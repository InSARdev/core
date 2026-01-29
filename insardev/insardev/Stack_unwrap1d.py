# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""
1D phase unwrapping along the temporal dimension using L1-norm IRLS.

Optimized implementation with:
- CPU batched linear solve (faster than GPU for small matrices)
- Efficient einsum-based AtWA computation
- Memory-efficient chunked processing
- Direct time series output option
"""
from .Stack_phasediff import Stack_phasediff
import numpy as np


class Stack_unwrap1d(Stack_phasediff):
    """1D phase unwrapping along the temporal dimension using L1-norm IRLS."""

    @staticmethod
    def wrap(data_pairs):
        """Wrap phase to [-pi, pi] range."""
        import xarray as xr
        import dask

        if isinstance(data_pairs, xr.DataArray):
            return xr.DataArray(dask.array.mod(data_pairs.data + np.pi, 2 * np.pi) - np.pi, data_pairs.coords)\
                .rename(data_pairs.name)
        return np.mod(data_pairs + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def _build_incidence_matrix(pair_dates):
        """
        Build incidence matrix for temporal network.

        Parameters
        ----------
        pair_dates : list of tuple
            List of (ref_date, rep_date) pairs.

        Returns
        -------
        A : np.ndarray
            Incidence matrix (n_pairs, n_intervals) where A[i,j] = 1 if
            date interval j is covered by pair i.
        dates : list
            Sorted unique dates.
        """
        all_dates = sorted(set(d for pair in pair_dates for d in pair))
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        n_pairs = len(pair_dates)
        n_intervals = len(all_dates) - 1

        A = np.zeros((n_pairs, n_intervals), dtype=np.float32)
        for i, (d1, d2) in enumerate(pair_dates):
            i1 = date_to_idx[d1]
            i2 = date_to_idx[d2]
            for j in range(i1, i2):
                A[i, j] = 1

        return A, all_dates

    @staticmethod
    def _irls_solve_1d(phi, W_corr, A_t, dev, max_iter=5, epsilon=0.1,
                       convergence_threshold=1e-3, return_increments=False):
        """
        Optimized IRLS solver using CPU batched operations.

        Uses CPU for batched linear solve (much faster than MPS/CUDA for small matrices)
        while keeping other operations on the specified device.

        Parameters
        ----------
        phi : torch.Tensor
            Wrapped or unwrapped phases (n_pairs, n_pixels).
        W_corr : torch.Tensor
            Correlation weights (n_pairs, n_pixels).
        A_t : torch.Tensor
            Incidence matrix (n_pairs, n_intervals).
        dev : torch.device
            PyTorch device (used for element-wise ops, solve always on CPU).
        max_iter : int
            Maximum IRLS iterations.
        epsilon : float
            IRLS regularization (larger = faster convergence, less L1-like).
        convergence_threshold : float
            Stop when weight change is below this.
        return_increments : bool
            If True, return phase increments instead of corrected pairs.

        Returns
        -------
        result : torch.Tensor
            Corrected phases (n_pairs, n_pixels) or increments (n_intervals, n_pixels).
        """
        import torch

        n_pairs, n_pixels = phi.shape
        n_intervals = A_t.shape[1]

        # Handle NaN
        nan_mask = torch.isnan(phi) | torch.isnan(W_corr)
        phi_clean = torch.where(nan_mask, torch.zeros_like(phi), phi)
        W_corr_clean = torch.where(nan_mask, torch.zeros_like(W_corr), W_corr)

        # Detect if input is wrapped (values mostly in [-π, π]) or 2D-unwrapped
        phi_max = torch.abs(phi_clean).max()
        wrapped_input = phi_max < 4 * np.pi  # Heuristic: if max < 4π, likely wrapped

        # Initialize IRLS weights
        W_irls = torch.ones_like(phi_clean)

        # Move A to CPU for batched solve (CPU is much faster for small batched linalg)
        A_cpu = A_t.cpu()
        reg_cpu = 1e-4 * torch.eye(n_intervals)

        for iteration in range(max_iter):
            # Combined weight
            W = W_irls * W_corr_clean  # (n_pairs, n_pixels)
            W_T = W.T  # (n_pixels, n_pairs)

            # Move to CPU for solve
            W_T_cpu = W_T.cpu()
            phi_cpu = phi_clean.cpu()
            W_cpu = W.cpu()

            # Build per-pixel AtWA using einsum (efficient on CPU)
            # AtWA[p,i,j] = sum_k W[p,k] * A[k,i] * A[k,j]
            AtWA = torch.einsum('pk,ki,kj->pij', W_T_cpu, A_cpu, A_cpu)
            AtWA = AtWA + reg_cpu

            # RHS: A^T W phi
            Wphi = W_cpu * phi_cpu  # (n_pairs, n_pixels)
            AtWphi = (A_cpu.T @ Wphi).T  # (n_pixels, n_intervals)

            # CPU batched solve (much faster than MPS/CUDA for small matrices)
            x = torch.linalg.solve(AtWA, AtWphi.unsqueeze(2)).squeeze(2)  # (n_pixels, n_intervals)
            x = x.T  # (n_intervals, n_pixels)

            # Move back to device for residual computation
            x = x.to(dev)

            # Reconstruct pair phases: φ_recon = A @ x
            phi_recon = A_t @ x  # (n_pairs, n_pixels)

            # Residuals
            residuals = phi_clean - phi_recon  # (n_pairs, n_pixels)
            # Wrap residuals for wrapped input to handle 2π ambiguity
            if wrapped_input:
                residuals = torch.atan2(torch.sin(residuals), torch.cos(residuals))

            # Update IRLS weights
            W_irls_new = 1.0 / (torch.abs(residuals) + epsilon)

            # Check convergence
            weight_change = torch.abs(W_irls_new - W_irls).mean()
            if weight_change < convergence_threshold:
                break

            W_irls = W_irls_new

        if return_increments:
            # Return phase increments per date interval
            result = x  # (n_intervals, n_pixels)
            # Mark pixels with all NaN as NaN
            all_nan = nan_mask.all(dim=0)
            result[:, all_nan] = float('nan')
            return result
        else:
            # Return corrected pair phases
            # Integer correction: k = round((φ_recon - φ) / 2π)
            corrections = torch.round((phi_recon - phi_clean) / (2 * np.pi))
            unwrapped = phi_clean + corrections * 2 * np.pi

            # Restore NaN
            unwrapped = torch.where(nan_mask, torch.full_like(unwrapped, float('nan')), unwrapped)
            return unwrapped

    @staticmethod
    def _unwrap1d_pairs_numpy(phase_stack, weight_stack, pair_dates, device='auto',
                               max_iter=5, epsilon=0.1, batch_size=50000, debug=False):
        """
        L1-norm IRLS temporal phase unwrapping on numpy arrays.

        Parameters
        ----------
        phase_stack : np.ndarray
            Wrapped or 2D-unwrapped phases, shape (n_pairs, height, width).
        weight_stack : np.ndarray or None
            Correlation weights, shape (n_pairs, height, width) or None.
        pair_dates : list of tuple
            List of (ref_date, rep_date) for each pair.
        device : str
            PyTorch device ('auto', 'cuda', 'mps', 'cpu').
        max_iter : int
            Maximum IRLS iterations.
        epsilon : float
            IRLS regularization parameter.
        batch_size : int
            Pixels per batch for memory efficiency.
        debug : bool
            Print debug information.

        Returns
        -------
        unwrapped : np.ndarray
            Temporally unwrapped phases, shape (n_pairs, height, width).
        """
        import torch

        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width

        # Build incidence matrix
        A, dates = Stack_unwrap1d._build_incidence_matrix(pair_dates)
        dev = Stack_unwrap1d._get_torch_device(device)

        if debug:
            print(f'IRLS 1D unwrap: {n_pairs} pairs, {len(dates)} dates, {n_pixels} pixels')
            print(f'  Device: {dev}, batch_size: {batch_size}')

        # Move matrix to device
        A_t = torch.from_numpy(A).to(dev)

        # Reshape to (n_pairs, n_pixels)
        phases_flat = phase_stack.reshape(n_pairs, n_pixels)
        if weight_stack is not None:
            weights_flat = weight_stack.reshape(n_pairs, n_pixels)
        else:
            weights_flat = np.ones_like(phases_flat)

        # Process in batches
        unwrapped_flat = np.zeros_like(phases_flat)
        n_batches = (n_pixels + batch_size - 1) // batch_size

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_pixels)

            if debug and (b == 0 or (b + 1) % 10 == 0 or b == n_batches - 1):
                print(f'  Batch {b+1}/{n_batches}')

            # Move batch to device
            phi = torch.from_numpy(phases_flat[:, start:end].astype(np.float32)).to(dev)
            W = torch.from_numpy(weights_flat[:, start:end].astype(np.float32)).to(dev)

            # Solve
            result = Stack_unwrap1d._irls_solve_1d(
                phi, W, A_t, dev, max_iter=max_iter, epsilon=epsilon,
                return_increments=False
            )

            unwrapped_flat[:, start:end] = result.cpu().numpy()

            del phi, W, result
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

        # Reshape back
        return unwrapped_flat.reshape(n_pairs, height, width)

    @staticmethod
    def _unwrap1d_to_dates_numpy(phase_stack, weight_stack, pair_dates, device='auto',
                                  max_iter=5, epsilon=0.1, batch_size=50000,
                                  cumsum=True, debug=False):
        """
        L1-norm IRLS temporal unwrapping directly to date-based time series.

        Two-step approach:
        1. Unwrap pairs (apply 2π corrections for wrapped input)
        2. Network inversion to get per-date time series

        Parameters
        ----------
        phase_stack : np.ndarray
            Wrapped or 2D-unwrapped phases, shape (n_pairs, height, width).
        weight_stack : np.ndarray or None
            Correlation weights, shape (n_pairs, height, width) or None.
        pair_dates : list of tuple
            List of (ref_date, rep_date) for each pair.
        device : str
            PyTorch device ('auto', 'cuda', 'mps', 'cpu').
        max_iter : int
            Maximum IRLS iterations.
        epsilon : float
            IRLS regularization parameter.
        batch_size : int
            Pixels per batch.
        cumsum : bool
            If True, return cumulative displacement. If False, return increments.
        debug : bool
            Print debug information.

        Returns
        -------
        time_series : np.ndarray
            Phase time series, shape (n_dates, height, width).
        dates : list
            Sorted unique dates.
        """
        import torch

        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width

        # Build incidence matrix
        A, dates = Stack_unwrap1d._build_incidence_matrix(pair_dates)
        n_intervals = len(dates) - 1
        dev = Stack_unwrap1d._get_torch_device(device)

        if debug:
            print(f'IRLS 1D to dates: {n_pairs} pairs -> {len(dates)} dates')
            print(f'  Device: {dev}, batch_size: {batch_size}')

        # Move matrix to device
        A_t = torch.from_numpy(A).to(dev)
        A_cpu = A_t.cpu()
        reg_cpu = 1e-4 * torch.eye(n_intervals)

        # Reshape
        phases_flat = phase_stack.reshape(n_pairs, n_pixels)
        if weight_stack is not None:
            weights_flat = weight_stack.reshape(n_pairs, n_pixels)
        else:
            weights_flat = np.ones_like(phases_flat)

        # Step 1: Unwrap pairs (apply 2π corrections)
        # Step 2: Compute increments from unwrapped pairs
        increments_flat = np.zeros((n_intervals, n_pixels), dtype=np.float32)
        n_batches = (n_pixels + batch_size - 1) // batch_size

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_pixels)

            if debug and (b == 0 or (b + 1) % 10 == 0 or b == n_batches - 1):
                print(f'  Batch {b+1}/{n_batches}')

            phi = torch.from_numpy(phases_flat[:, start:end].astype(np.float32)).to(dev)
            W = torch.from_numpy(weights_flat[:, start:end].astype(np.float32)).to(dev)

            # Step 1: Unwrap pairs (return_increments=False applies 2π corrections)
            unwrapped = Stack_unwrap1d._irls_solve_1d(
                phi, W, A_t, dev, max_iter=max_iter, epsilon=epsilon,
                return_increments=False
            )

            # Step 2: Compute increments from unwrapped pairs using weighted least squares
            # Solve: A @ x = unwrapped_pairs
            unwrapped_cpu = unwrapped.cpu()
            W_cpu = W.cpu()
            W_T_cpu = W_cpu.T

            # Build AtWA
            AtWA = torch.einsum('pk,ki,kj->pij', W_T_cpu, A_cpu, A_cpu)
            AtWA = AtWA + reg_cpu

            # RHS: A^T W unwrapped
            Wphi = W_cpu * unwrapped_cpu
            AtWphi = (A_cpu.T @ Wphi).T

            # Solve for increments
            x = torch.linalg.solve(AtWA, AtWphi.unsqueeze(2)).squeeze(2)
            x = x.T  # (n_intervals, n_pixels)

            increments_flat[:, start:end] = x.numpy()

            del phi, W, unwrapped, x
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

        # Build time series
        if cumsum:
            ts = np.zeros((len(dates), n_pixels), dtype=np.float32)
            ts[1:, :] = np.cumsum(increments_flat, axis=0)
        else:
            ts = np.zeros((len(dates), n_pixels), dtype=np.float32)
            ts[1:, :] = increments_flat

        # Reshape: (n_dates, n_pixels) -> (n_dates, height, width)
        time_series = ts.reshape(len(dates), height, width)

        return time_series, dates

    @staticmethod
    def _lstsq_to_dates_numpy(phase_stack, weight_stack, pair_dates, device='auto',
                               cumsum=True, debug=False):
        """
        Weighted least squares network inversion from pairs to dates.

        Uses PyTorch batched least squares for GPU acceleration.

        Parameters
        ----------
        phase_stack : np.ndarray
            Unwrapped phases, shape (n_pairs, height, width).
        weight_stack : np.ndarray or None
            Correlation weights, shape (n_pairs, height, width).
        pair_dates : list of tuple
            List of (ref_date, rep_date) for each pair.
        device : str
            PyTorch device ('auto', 'cuda', 'mps', 'cpu').
        cumsum : bool
            If True, return cumulative displacement.
        debug : bool
            Print debug information.

        Returns
        -------
        time_series : np.ndarray
            Phase time series, shape (n_dates, height, width).
        dates : list
            Sorted unique dates.
        """
        import torch

        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width

        # Build incidence matrix
        A, dates = Stack_unwrap1d._build_incidence_matrix(pair_dates)
        n_dates = len(dates)
        n_intervals = n_dates - 1

        dev = Stack_unwrap1d._get_torch_device(device)
        if debug:
            print(f'lstsq: {n_pairs} pairs -> {n_dates} dates, {n_pixels} pixels, device={dev}')

        # Reshape to (n_pairs, n_pixels)
        phases_flat = phase_stack.reshape(n_pairs, n_pixels)  # (n_pairs, n_pixels)
        if weight_stack is not None:
            weights_flat = weight_stack.reshape(n_pairs, n_pixels)
            # Clamp weights to (0, 1-eps) for numerical stability
            weights_flat = np.clip(weights_flat, 1e-6, 1 - 1e-6)
        else:
            weights_flat = None

        # Move to device
        A_t = torch.from_numpy(A.astype(np.float32)).to(dev)  # (n_pairs, n_intervals)
        phi = torch.from_numpy(phases_flat.astype(np.float32)).to(dev)  # (n_pairs, n_pixels)

        # Handle NaN in phase
        nan_mask_phi = torch.isnan(phi)
        phi = torch.where(nan_mask_phi, torch.zeros_like(phi), phi)

        if weights_flat is not None:
            W = torch.from_numpy(weights_flat.astype(np.float32)).to(dev)
            # Handle NaN in weights too
            nan_mask_w = torch.isnan(W)
            nan_mask = nan_mask_phi | nan_mask_w
            W = torch.where(nan_mask, torch.zeros_like(W), W)
            # Transform correlation to least squares weight: w / sqrt(1 - w^2)
            # Clamp to avoid Inf (W=1) and NaN (W>1)
            W_clamped = torch.clamp(W, 0.0, 0.9999)
            W_lstsq = W_clamped / torch.sqrt(1 - W_clamped**2)
        else:
            nan_mask = nan_mask_phi
            W_lstsq = torch.where(nan_mask, torch.zeros(1, device=dev), torch.ones(1, device=dev))
            W_lstsq = W_lstsq.expand(n_pairs, n_pixels)

        # Weighted least squares: solve (W*A)x = W*b for each pixel
        # Transpose to (n_pixels, n_pairs) for batched solve
        phi_T = phi.T  # (n_pixels, n_pairs)
        W_T = W_lstsq.T  # (n_pixels, n_pairs)

        # Apply weights: WA and Wb
        # WA: (n_pixels, n_pairs, n_intervals) = W_T[:, :, None] * A_t[None, :, :]
        # Wb: (n_pixels, n_pairs) = W_T * phi_T
        WA = W_T.unsqueeze(2) * A_t.unsqueeze(0)  # (n_pixels, n_pairs, n_intervals)
        Wb = (W_T * phi_T).unsqueeze(2)  # (n_pixels, n_pairs, 1)

        # Batched least squares solve on CPU (faster for small matrices)
        WA_cpu = WA.cpu()
        Wb_cpu = Wb.cpu()

        # Replace any non-finite values with 0 (CUDA lstsq is strict)
        WA_cpu = torch.nan_to_num(WA_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        Wb_cpu = torch.nan_to_num(Wb_cpu, nan=0.0, posinf=0.0, neginf=0.0)

        # torch.linalg.lstsq returns (solution, residuals, rank, singular_values)
        result = torch.linalg.lstsq(WA_cpu, Wb_cpu)
        x = result.solution.squeeze(2)  # (n_pixels, n_intervals)

        # Mark all-NaN pixels
        all_nan = nan_mask.all(dim=0).cpu()
        x[all_nan, :] = float('nan')

        # Build time series
        if cumsum:
            # Cumulative sum: first date = 0, then cumsum of increments
            ts = torch.zeros((n_pixels, n_dates), dtype=torch.float32)
            ts[:, 1:] = torch.cumsum(x, dim=1)
        else:
            ts = torch.zeros((n_pixels, n_dates), dtype=torch.float32)
            ts[:, 1:] = x

        # Reshape: (n_pixels, n_dates) -> (n_dates, height, width)
        time_series = ts.T.numpy().reshape(n_dates, height, width)

        # Cleanup GPU memory
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()

        return time_series, dates

    def lstsq(self, data, weight=None, device='auto', cumsum=True, debug=False):
        """
        Weighted least squares network inversion to date-based time series.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase. Uses PyTorch batched least squares for GPU acceleration.

        Parameters
        ----------
        data : Batch
            Unwrapped phase data with 'pair' dimension.
        weight : BatchUnit, optional
            Correlation weights for each pair.
        device : str, optional
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        cumsum : bool, optional
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        Batch
            Phase time series with 'date' dimension instead of 'pair'.

        Notes
        -----
        Typical workflow:
        1. stack.unwrap1d(intf, corr) - unwrap pairs temporally
        2. stack.lstsq(unwrapped, corr) - network inversion to dates

        Examples
        --------
        >>> unwrapped = stack.unwrap1d(intf, corr)
        >>> displacement = stack.lstsq(unwrapped, corr)
        """
        import xarray as xr
        import pandas as pd
        from .Batch import Batch, BatchWrap, BatchUnit

        # Validate input types
        if isinstance(data, BatchWrap):
            raise TypeError(
                'lstsq() requires unwrapped phase (Batch), got BatchWrap. '
                'Use unwrap1d() first to unwrap wrapped phase data.'
            )
        if not isinstance(data, Batch):
            raise TypeError(
                f'data must be Batch (unwrapped phase), got {type(data).__name__}.'
            )
        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(data) to convert correlation data.'
            )

        # Process each burst
        results = {}
        for key in data.keys():
            ds = data[key]
            w_ds = weight[key] if weight is not None else None

            result_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for pol in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da = ds[pol]
                w_da = w_ds[pol] if w_ds is not None else None

                result = self._lstsq_dataarray(da, w_da, device, cumsum, debug)
                result_vars[pol] = result

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            if ds.rio.crs is not None:
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        return Batch(results)

    def _lstsq_dataarray(self, data, weight, device, cumsum, debug):
        """Internal method for lstsq on DataArray - LAZY dask processing."""
        import xarray as xr
        import pandas as pd
        import dask
        import dask.array as da

        # Get pairs
        pairs = self._get_pairs(data)
        pair_dates = [(str(row.ref), str(row.rep)) for _, row in pairs.iterrows()]

        # Build incidence matrix once (needed for all blocks)
        A, dates = Stack_unwrap1d._build_incidence_matrix(pair_dates)
        n_dates = len(dates)
        n_pairs = len(pair_dates)

        # Use dask auto-chunking for y,x based on actual memory usage
        # WA tensor is (n_pixels, n_pairs, n_intervals) - dominant memory consumer
        # Memory per pixel = n_pairs * n_intervals * 4 bytes
        # Use 2D auto-chunks with memory-equivalent dtype
        n_intervals = n_dates - 1
        mem_per_pixel = n_pairs * n_intervals * 4
        auto_chunks = dask.array.core.normalize_chunks(
            'auto', (data.y.size, data.x.size), dtype=np.dtype(f'V{mem_per_pixel}')
        )
        chunks_y, chunks_x = auto_chunks[0][0], auto_chunks[1][0]

        # Rechunk: all pairs together (-1), auto-chunked y,x
        first_dim = data.dims[0]
        data = data.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})
        if weight is not None:
            weight = weight.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})

        # Use blockwise to avoid embedding large arrays in the graph
        def process_block(data_block, weight_block=None):
            """Process a spatial block - all pairs, subset of y,x."""
            # PyTorch batched weighted least squares
            ts_block, _ = Stack_unwrap1d._lstsq_to_dates_numpy(
                data_block, weight_block, pair_dates,
                device=device, cumsum=cumsum, debug=False
            )
            return ts_block.astype(np.float32)

        data_dask = data.data
        if weight is not None:
            weight_dask = weight.data
            result_dask = da.map_blocks(
                process_block, data_dask, weight_dask,
                dtype=np.float32,
                drop_axis=0,
                new_axis=0,
                chunks=(n_dates,) + data_dask.chunks[1:],
            )
        else:
            def process_block_no_weight(data_block):
                return process_block(data_block, None)
            result_dask = da.map_blocks(
                process_block_no_weight, data_dask,
                dtype=np.float32,
                drop_axis=0,
                new_axis=0,
                chunks=(n_dates,) + data_dask.chunks[1:],
            )

        # Rechunk to (1, -1, -1) for efficient per-slice downstream operations (e.g., plot)
        ts_dask = result_dask.rechunk({0: 1, 1: -1, 2: -1})

        # Build coordinates
        coords = {
            'date': pd.to_datetime(dates),
            'y': data.coords['y'],
            'x': data.coords['x']
        }

        return xr.DataArray(
            ts_dask,
            coords=coords,
            dims=('date', 'y', 'x'),
            name='displacement'
        )

    def unwrap1d(self, data, weight=None, device='auto', max_iter=5,
                 epsilon=0.1, batch_size=50000, debug=False):
        """
        L1-norm IRLS temporal phase unwrapping returning unwrapped pairs.

        Performs temporal unwrapping across the interferogram network,
        applying 2π corrections to make pairs consistent.

        Parameters
        ----------
        data : BatchWrap or Batch
            Phase data with 'pair' dimension (wrapped or 2D-unwrapped).
        weight : BatchUnit, optional
            Correlation weights for each pair.
        device : str, optional
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        max_iter : int, optional
            Maximum IRLS iterations. Default 5.
        epsilon : float, optional
            IRLS regularization parameter. Default 0.1.
        batch_size : int, optional
            Pixels per batch for memory efficiency. Default 50000.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        Batch
            Temporally unwrapped phase data with 'pair' dimension.

        Notes
        -----
        Typical workflow:
        1. stack.unwrap1d(intf, corr) - unwrap pairs temporally
        2. stack.lstsq(unwrapped, corr) - network inversion to dates

        Examples
        --------
        >>> unwrapped = stack.unwrap1d(intf, corr)
        >>> displacement = stack.lstsq(unwrapped, corr)
        """
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        # Auto-detect device based on Dask cluster resources and hardware
        device = Stack_unwrap1d._get_torch_device(device, debug=debug)

        if debug:
            print(f"DEBUG: unwrap1d using device={device}")

        # Validate input types
        if not isinstance(data, (Batch, BatchWrap)):
            raise TypeError(
                f'data must be BatchWrap or Batch, got {type(data).__name__}.'
            )
        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(data) to convert correlation data.'
            )

        # Process each burst
        results = {}
        for key in data.keys():
            ds = data[key]
            w_ds = weight[key] if weight is not None else None

            result_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for pol in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da = ds[pol]
                w_da = w_ds[pol] if w_ds is not None else None

                result = self._unwrap1d_pairs_dataarray(
                    da, w_da, device, max_iter, epsilon, batch_size, debug
                )
                result_vars[pol] = result

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            if ds.rio.crs is not None:
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        return Batch(results)

    def _unwrap1d_pairs_dataarray(self, data, weight, device, max_iter,
                                   epsilon, batch_size, debug):
        """Internal method for IRLS unwrapping on DataArray returning pairs - LAZY."""
        import xarray as xr
        import dask
        import dask.array

        # Get pairs
        pairs = self._get_pairs(data)
        pair_dates = [(str(row.ref), str(row.rep)) for _, row in pairs.iterrows()]
        n_pairs = len(pair_dates)

        # Save original coordinates before chunking
        original_coords = {}
        for k, v in data.coords.items():
            if hasattr(v, 'data') and hasattr(v.data, 'compute'):
                vals = v.compute().values
            elif hasattr(v, 'values'):
                vals = v.values
            else:
                vals = v
            if hasattr(v, 'dims') and len(v.dims) > 0 and v.dims != (k,):
                original_coords[k] = (v.dims, vals)
            else:
                original_coords[k] = vals

        # Use dask auto-chunking for y,x based on memory
        # Multiplier accounts for unwrap1d internal memory (IRLS iterations)
        import dask.array as da
        auto_chunks = dask.array.core.normalize_chunks(
            'auto', (4 * n_pairs, data.y.size, data.x.size), dtype=np.complex128
        )
        chunks_y, chunks_x = auto_chunks[1][0], auto_chunks[2][0]

        # Rechunk: all pairs together (-1), auto-chunked y,x
        first_dim = data.dims[0]
        data = data.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})
        if weight is not None:
            weight = weight.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})

        # Use blockwise to avoid embedding large arrays in the graph
        def process_block(data_block, weight_block=None):
            """Process a spatial block - all pairs, subset of y,x."""
            unwrapped_block = Stack_unwrap1d._unwrap1d_pairs_numpy(
                data_block, weight_block, pair_dates,
                device=device, max_iter=max_iter, epsilon=epsilon,
                batch_size=batch_size, debug=False
            )
            return unwrapped_block.astype(np.float32)

        data_dask = data.data
        if weight is not None:
            weight_dask = weight.data
            with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                unwrapped_dask = da.blockwise(
                    process_block, 'pyx',
                    data_dask, 'pyx',
                    weight_dask, 'pyx',
                    dtype=np.float32,
                )
        else:
            def process_block_no_weight(data_block):
                return process_block(data_block, None)
            with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                unwrapped_dask = da.blockwise(
                    process_block_no_weight, 'pyx',
                    data_dask, 'pyx',
                    dtype=np.float32,
                )

        # Rechunk to (1, -1, -1) for efficient per-slice downstream operations (e.g., plot)
        unwrapped_dask = unwrapped_dask.rechunk({0: 1, 1: -1, 2: -1})

        result = xr.DataArray(
            unwrapped_dask,
            dims=data.dims,
            name='unwrap'
        )
        result = result.assign_coords(original_coords)

        return result

