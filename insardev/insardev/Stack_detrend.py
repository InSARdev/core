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
from contextlib import nullcontext
from .Stack_unwrap2d import Stack_unwrap2d
from .Batch import Batch, BatchWrap

class Stack_detrend(Stack_unwrap2d):
    import numpy as np
    import xarray as xr

    @staticmethod
    def _regression1d_array(data, dim_values, weight, device, degree=1, wrap=False, iterations=1):
        """
        Fit 1D polynomial along first dimension at each (y, x) pixel using PyTorch.

        Parameters
        ----------
        data : np.ndarray
            3D array (n_samples, y, x) - polynomial fit along first dimension
        dim_values : np.ndarray
            1D array of x-values for fitting (length n_samples)
        weight : np.ndarray or None
            Weight array, same shape as data
        device : torch.device
            PyTorch device
        degree : int
            Polynomial degree (1=linear, 2=quadratic, etc.)
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase
        iterations : int
            Number of fitting iterations (for wrap=True, captures more trend per iteration)

        Returns
        -------
        np.ndarray
            Fitted values, same shape as data
        """
        import torch
        import numpy as np

        n_samples, ny, nx = data.shape
        n_pixels = ny * nx

        # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
        if device.type == 'cpu':
            dtype = torch.float64
        else:
            dtype = torch.float32

        # Normalize dim_values to [-1, 1] for numerical stability
        dim_min = dim_values.min()
        dim_max = dim_values.max()
        dim_range = dim_max - dim_min
        if dim_range > 0:
            dim_norm = 2 * (dim_values - dim_min) / dim_range - 1
            # Where does 0 map to in normalized space?
            x_origin_norm = 2 * (0.0 - dim_min) / dim_range - 1
        else:
            dim_norm = np.zeros_like(dim_values)
            x_origin_norm = 0.0

        # Build Vandermonde matrix: [1, x, x^2, ..., x^degree]
        # Shape: (n_samples, degree+1)
        X = np.column_stack([dim_norm**d for d in range(degree + 1)])
        X_t = torch.tensor(X, dtype=dtype, device=device)

        # For wrap mode, we need value at x=0 (original coordinates) to compute angle offset
        X_origin = torch.tensor([x_origin_norm**d for d in range(degree + 1)], dtype=dtype, device=device)

        # Flatten spatial dimensions: data becomes (n_samples, n_pixels)
        data_flat = data.reshape(n_samples, n_pixels)
        if weight is not None:
            weight_flat = weight.reshape(n_samples, n_pixels)

        # Move data to GPU
        data_t = torch.tensor(data_flat, dtype=dtype, device=device)
        if weight is not None:
            weight_t = torch.tensor(weight_flat, dtype=dtype, device=device)
        else:
            weight_t = None

        # Find valid pixels (no NaN in any sample)
        valid_mask = torch.isfinite(data_t).all(dim=0)  # (n_pixels,)
        if weight_t is not None:
            valid_mask &= torch.isfinite(weight_t).all(dim=0)

        valid_indices = torch.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            return np.full_like(data, np.nan, dtype=np.float32)

        # Extract valid pixels: (n_samples, n_valid)
        data_valid = data_t[:, valid_indices]
        if weight_t is not None:
            weight_valid = weight_t[:, valid_indices]
            sqrt_w = torch.sqrt(weight_valid)
        else:
            sqrt_w = None

        # Precompute (X^T X)^-1 X^T for unweighted case (normal equations)
        # This avoids lstsq which is not supported on MPS
        XtX = X_t.T @ X_t
        XtX_inv = torch.linalg.inv(XtX + 1e-10 * torch.eye(XtX.shape[0], dtype=dtype, device=device))
        XtX_inv_Xt = XtX_inv @ X_t.T  # (degree+1, n_samples)

        # Helper function to fit sin/cos once
        def fit_sincos_once(current_data):
            """Fit sin/cos to data and return angle trend."""
            sin_data = torch.sin(current_data)
            cos_data = torch.cos(current_data)

            if sqrt_w is not None:
                # Weighted case: solve via normal equations per pixel
                sin_weighted = (sin_data * sqrt_w).T  # (n_valid, n_samples)
                cos_weighted = (cos_data * sqrt_w).T
                X_batch = X_t[None, :, :] * sqrt_w.T[:, :, None]  # (n_valid, n_samples, d+1)
                # Normal equations: (X^T X) coeffs = X^T y
                XtX_batch = X_batch.transpose(1, 2) @ X_batch  # (n_valid, d+1, d+1)
                Xty_sin = X_batch.transpose(1, 2) @ sin_weighted.unsqueeze(-1)  # (n_valid, d+1, 1)
                Xty_cos = X_batch.transpose(1, 2) @ cos_weighted.unsqueeze(-1)
                # Add regularization for stability
                reg = 1e-10 * torch.eye(XtX_batch.shape[-1], dtype=dtype, device=device)
                coeffs_sin = torch.linalg.solve(XtX_batch + reg, Xty_sin)
                coeffs_cos = torch.linalg.solve(XtX_batch + reg, Xty_cos)
            else:
                # Unweighted case: use precomputed pseudo-inverse
                # coeffs = (X^T X)^-1 X^T y
                sin_T = sin_data.T  # (n_valid, n_samples)
                cos_T = cos_data.T
                coeffs_sin = (XtX_inv_Xt @ sin_T.T).T.unsqueeze(-1)  # (n_valid, d+1, 1)
                coeffs_cos = (XtX_inv_Xt @ cos_T.T).T.unsqueeze(-1)

            # Evaluate fit
            fit_sin = (X_t @ coeffs_sin.squeeze(-1).T).T  # (n_valid, n_samples)
            fit_cos = (X_t @ coeffs_cos.squeeze(-1).T).T

            # Compute angle offset at origin
            sin_origin = (X_origin @ coeffs_sin.squeeze(-1).T)  # (n_valid,)
            cos_origin = (X_origin @ coeffs_cos.squeeze(-1).T)
            angle_origin = torch.atan2(sin_origin, cos_origin)

            # Convert back to angle (relative to origin)
            fit_angle = torch.atan2(fit_sin, fit_cos) - angle_origin[:, None]
            return fit_angle

        if wrap:
            # Iterative circular fitting
            # data_valid has shape (n_samples, n_valid)
            cumulative_trend = torch.zeros_like(data_valid)
            current_data = data_valid.clone()

            for _ in range(iterations):
                # Fit current data - returns (n_valid, n_samples), need to transpose
                fit_angle = fit_sincos_once(current_data).T  # Now (n_samples, n_valid)

                # Accumulate trend
                cumulative_trend = cumulative_trend + fit_angle

                # Update residual for next iteration (subtract trend circularly)
                current_data = current_data - fit_angle

            # Wrap final result to [-π, π]
            cumulative_trend = torch.remainder(cumulative_trend + np.pi, 2 * np.pi) - np.pi

            # Put back into full array
            result_t = torch.full((n_samples, n_pixels), float('nan'), dtype=dtype, device=device)
            result_t[:, valid_indices] = cumulative_trend

        else:
            # Standard polynomial fitting (iterations not needed - one fit is exact)
            if sqrt_w is not None:
                # Weighted case: solve via normal equations per pixel
                y_weighted = (data_valid * sqrt_w).T  # (n_valid, n_samples)
                X_batch = X_t[None, :, :] * sqrt_w.T[:, :, None]  # (n_valid, n_samples, d+1)
                XtX_batch = X_batch.transpose(1, 2) @ X_batch  # (n_valid, d+1, d+1)
                Xty = X_batch.transpose(1, 2) @ y_weighted.unsqueeze(-1)  # (n_valid, d+1, 1)
                reg = 1e-10 * torch.eye(XtX_batch.shape[-1], dtype=dtype, device=device)
                coeffs = torch.linalg.solve(XtX_batch + reg, Xty)
            else:
                # Unweighted case: use precomputed pseudo-inverse
                data_T = data_valid.T  # (n_valid, n_samples)
                coeffs = (XtX_inv_Xt @ data_T.T).T.unsqueeze(-1)  # (n_valid, d+1, 1)

            # Evaluate fit
            fit = (X_t @ coeffs.squeeze(-1).T).T  # (n_valid, n_samples)

            # Put back into full array
            result_t = torch.full((n_samples, n_pixels), float('nan'), dtype=dtype, device=device)
            result_t[:, valid_indices] = fit.T

        # Reshape back to (n_samples, ny, nx)
        result = result_t.cpu().numpy().reshape(n_samples, ny, nx)
        return result.astype(np.float32)

    @staticmethod
    def _trend2d_array(phase, weight, variables, device, degree=1):
        """
        Fit 2D polynomial trend using PyTorch least squares (pure GPU implementation).

        Parameters
        ----------
        phase : np.ndarray
            2D or 3D phase array (pair, y, x) or (y, x)
        weight : np.ndarray or None
            Weight array, same shape as phase
        variables : list of np.ndarray
            List of 2D variable arrays (y, x) to use as regressors
        device : torch.device
            PyTorch device
        degree : int
            Polynomial degree for each variable (1=linear, 2=quadratic, etc.)

        Returns
        -------
        np.ndarray
            Trend surface, same shape as phase
        """
        import torch
        import numpy as np
        from itertools import combinations_with_replacement

        # Handle 2D vs 3D input
        squeeze = phase.ndim == 2
        if squeeze:
            phase = phase[np.newaxis, ...]
            if weight is not None:
                weight = weight[np.newaxis, ...]

        n_pairs, ny, nx = phase.shape
        n_pixels = ny * nx

        # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
        if device.type == 'cpu':
            dtype = torch.float64
        else:
            dtype = torch.float32

        # Flatten variables and track valid coordinates (on CPU first)
        valid_coords = np.ones(n_pixels, dtype=bool)
        vars_np = []
        for var in variables:
            v_flat = var.ravel()
            valid_coords &= np.isfinite(v_flat)
            vars_np.append(np.nan_to_num(v_flat, nan=0.0))

        # Move variables to GPU and build polynomial features there
        vars_t = [torch.tensor(v, dtype=dtype, device=device) for v in vars_np]
        n_vars = len(vars_t)

        # Build polynomial features on GPU (matching sklearn PolynomialFeatures order)
        # Order: degree 1 first, then degree 2, etc. (same as sklearn include_bias=False)
        features = []
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n_vars), d):
                term = torch.ones(n_pixels, dtype=dtype, device=device)
                for idx in combo:
                    term = term * vars_t[idx]
                features.append(term)

        # Stack features: (n_pixels, n_features)
        X_poly = torch.stack(features, dim=1)

        # Add bias column
        ones = torch.ones(n_pixels, 1, dtype=dtype, device=device)
        A_all = torch.cat([X_poly, ones], dim=1)

        # Valid coordinates mask on GPU
        valid_coords_t = torch.tensor(valid_coords, dtype=torch.bool, device=device)

        trends = np.empty_like(phase)

        for i in range(n_pairs):
            phase_flat = phase[i].ravel()

            # Get valid (non-NaN) mask
            valid_np = np.isfinite(phase_flat) & valid_coords
            if weight is not None:
                weight_flat = weight[i].ravel()
                valid_np &= np.isfinite(weight_flat)

            n_valid = valid_np.sum()
            if n_valid < 10:
                trends[i] = np.nan
                continue

            valid_t = torch.tensor(valid_np, dtype=torch.bool, device=device)

            # Extract valid rows
            A_valid = A_all[valid_t]  # (n_valid, n_features+1)
            b_valid = torch.tensor(phase_flat[valid_np], dtype=dtype, device=device)

            # Standardize features on GPU (excluding bias column)
            # Compute mean and std on valid data only
            feature_mean = A_valid[:, :-1].mean(dim=0, keepdim=True)
            feature_std = A_valid[:, :-1].std(dim=0, keepdim=True) + 1e-10
            A_valid_scaled = torch.cat([
                (A_valid[:, :-1] - feature_mean) / feature_std,
                A_valid[:, -1:]  # Keep bias column as-is
            ], dim=1)

            # Apply weights if provided
            if weight is not None:
                w = torch.tensor(np.sqrt(weight_flat[valid_np]), dtype=dtype, device=device)
                A_valid_scaled = A_valid_scaled * w[:, None]
                b_valid = b_valid * w

            # Solve normal equations: (A^T A) coeffs = A^T b
            AtA = A_valid_scaled.T @ A_valid_scaled
            Atb = A_valid_scaled.T @ b_valid
            coeffs = torch.linalg.solve(AtA, Atb)

            # Compute trend for all pixels using the same scaling
            A_all_scaled = torch.cat([
                (A_all[:, :-1] - feature_mean) / feature_std,
                A_all[:, -1:]
            ], dim=1)
            trend = (A_all_scaled @ coeffs).cpu().numpy()
            trends[i] = trend.reshape(ny, nx)

        if squeeze:
            trends = trends[0]

        return trends.astype(np.float32)

    def trend2d(self, phase, weight=None, transform=None, degree=1, device='auto', debug=False):
        """
        Compute 2D polynomial trend using PyTorch (GPU-accelerated).

        Parameters
        ----------
        phase : Batch or BatchWrap
            Phase data to detrend.
        weight : Batch or None
            Optional weights (e.g., correlation).
        transform : Batch or None
            Transform with variables to use as regressors (e.g., 'azi', 'rng', 'ele').
            All data_vars in transform will be used. If None, uses y/x coordinates.
        degree : int
            Polynomial degree for each variable (default 1):
            - degree=1: linear (a₁*v₁ + a₂*v₂ + ... + c)
            - degree=2: includes v₁², v₂², ... terms
            - degree=3: includes v₁³, v₂³, ... terms
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Trend surface (same type as input phase, lazy).

        Examples
        --------
        >>> trend = stack.trend2d(phase, weight=corr, transform=transform[['azi','rng','ele']], degree=1)
        >>> detrended = phase - trend
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np

        # Auto-detect device based on Dask cluster resources and hardware
        device = Stack_detrend._get_torch_device(device, debug=debug)

        if debug:
            print(f"DEBUG: using device={device}")

        # Warn about MPS precision issues with high-degree polynomials
        if device == 'mps' and degree >= 3:
            print(f"NOTE: MPS has float32 precision issues for degree>={degree}. Use device='cpu' for better accuracy.")

        wrap = isinstance(phase, BatchWrap)

        # Unify transform keys to phase
        if transform is not None:
            transform = transform.sel(phase)

        result = {}
        for key in phase.keys():
            ds = phase[key]

            # Get polarization variables (spatial, with y/x dims) - excludes converted attributes
            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            # Get variables from transform or use y/x coordinates
            if transform is not None:
                trans_ds = transform[key]
                # Filter for spatial variables (with y, x dims) - excludes converted attributes
                var_names = [v for v in trans_ds.data_vars
                            if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]
                # Get transform variables (compute if lazy)
                var_arrays = [trans_ds[v] for v in var_names]
                if any(hasattr(arr.data, 'compute') for arr in var_arrays):
                    computed = dask.compute(*var_arrays)
                    variables = [arr.values if hasattr(arr, 'values') else arr for arr in computed]
                else:
                    variables = [arr.values for arr in var_arrays]
            else:
                # Use y/x coordinates as default variables
                y = ds[pols[0]].y.values
                x = ds[pols[0]].x.values
                Y, X = np.meshgrid(y, x, indexing='ij')
                variables = [Y.astype(np.float32), X.astype(np.float32)]
                var_names = ['y', 'x']

            if debug:
                print(f"DEBUG {key}: variables={var_names}, shapes={[v.shape for v in variables]}")

            result_ds = {}
            for pol in pols:
                phase_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                # Save coordinates for output
                coords = dict(phase_da.coords)
                dims = phase_da.dims

                # Create wrapper function for blockwise
                def process_chunk(phase_chunk, weight_chunk=None, variables=variables, device=device, degree=degree):
                    import torch
                    # Ensure 3D input (pair, y, x)
                    if phase_chunk.ndim == 2:
                        phase_chunk = phase_chunk[np.newaxis, ...]
                        if weight_chunk is not None:
                            weight_chunk = weight_chunk[np.newaxis, ...]
                    result = Stack_detrend._trend2d_array(
                        phase_chunk, weight_chunk, variables, torch.device(device), degree
                    )
                    return result

                # Use da.blockwise for efficient dask integration
                phase_dask = phase_da.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(phase_dask.ndim))

                with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                    if weight_da is not None:
                        weight_dask = weight_da.data
                        def process_with_weight(phase_block, weight_block):
                            return process_chunk(phase_block, weight_block)
                        result_dask = da.blockwise(
                            process_with_weight, dim_str,
                            phase_dask, dim_str,
                            weight_dask, dim_str,
                            dtype=np.float32,
                        )
                    else:
                        result_dask = da.blockwise(
                            process_chunk, dim_str,
                            phase_dask, dim_str,
                            dtype=np.float32,
                        )

                trend_da = xr.DataArray(
                    result_dask,
                    dims=phase_da.dims,
                    coords=phase_da.coords
                )

                result_ds[pol] = trend_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchWrap(result) if wrap else Batch(result)

    def trend2d_dataset(self, phase, weight=None, transform=None, degree=1, device='auto', debug=False):
        """
        Compute 2D trend from phase Dataset using PyTorch (GPU-accelerated).

        Convenience wrapper around trend2d() for working with merged datasets
        instead of per-burst batches. Useful when you have already dissolved
        and merged your data.

        Parameters
        ----------
        phase : xr.Dataset
            Phase dataset (from phase.to_dataset() or unwrap2d_dataset()).
        weight : xr.Dataset, optional
            Correlation/weight dataset (from corr.to_dataset()).
        transform : xr.Dataset, optional
            Transform dataset with variables to use as regressors (e.g., 'azi', 'rng', 'ele').
        degree : int, optional
            Polynomial degree for each variable (default 1).
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        xr.Dataset
            Trend dataset with same structure as input phase.

        Examples
        --------
        Basic usage with merged datasets:
        >>> phase_ds = phase.to_dataset()
        >>> corr_ds = corr.to_dataset()
        >>> transform_ds = transform[['azi','rng']].to_dataset()
        >>> trend = stack.trend2d_dataset(phase_ds, corr_ds, transform_ds, degree=1)

        Convert back to per-burst Batch:
        >>> trend_batch = phase.from_dataset(trend)
        """
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        # Validate input types
        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")
        if transform is not None and not isinstance(transform, xr.Dataset):
            raise TypeError(f"transform must be xr.Dataset, got {type(transform).__name__}")

        # Wrap datasets in single-key Batches for trend2d()
        dummy_key = '__dataset__'
        phase_batch = BatchWrap({dummy_key: phase})
        weight_batch = BatchUnit({dummy_key: weight}) if weight is not None else None
        transform_batch = Batch({dummy_key: transform}) if transform is not None else None

        # Call PyTorch-based trend2d()
        trend_batch = self.trend2d(
            phase_batch,
            weight=weight_batch,
            transform=transform_batch,
            degree=degree,
            device=device,
            debug=debug
        )

        # Extract result Dataset
        output = trend_batch[dummy_key]
        output.attrs = phase.attrs
        return output

    def regression1d_baseline(self, data, weight=None, baseline='BPR', degree=1, wrap=False, iterations=1, device='auto', debug=False):
        """
        Fit 1D polynomial trend along perpendicular baseline at each (y, x) pixel using PyTorch.

        Parameters
        ----------
        data : Batch or BatchWrap
            Data with dimensions (pair/date, y, x) or (pair/date, pol, y, x).
        weight : Batch or None, optional
            Optional weights (e.g., correlation), same shape as data.
        baseline : str, Batch, np.ndarray, or xr.DataArray
            Baseline values to regress against:
            - str: variable name from data (default 'BPR' for perpendicular baseline)
            - Batch: values from Batch variable (e.g., data['BPR'])
            - array: explicit values (length must match pair dimension)
        degree : int
            Polynomial degree (1=linear, 2=quadratic, etc.)
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1). For wrap=True, multiple
            iterations capture more of the trend. Each iteration fits the
            residual from the previous iteration and accumulates the result.
            Typically 2-3 iterations are sufficient.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Fitted values, same type and shape as input data (lazy).

        Examples
        --------
        >>> # Fit linear trend along BPR (default)
        >>> trend = stack.regression1d_baseline(intf)
        >>> detrended = intf - trend
        >>>
        >>> # Use multiple iterations for wrapped phase
        >>> trend = stack.regression1d_baseline(intf, wrap=True, iterations=3)
        >>>
        >>> # Fit against custom baseline values
        >>> trend = stack.regression1d_baseline(intf, baseline=custom_values, degree=1)
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np
        import pandas as pd

        from .Batch import BatchWrap

        # Auto-detect device based on Dask cluster resources and hardware
        device = Stack_detrend._get_torch_device(device, debug=debug)

        if debug:
            print(f"DEBUG: using device={device}")

        is_wrap = isinstance(data, BatchWrap)

        result = {}
        for key in data.keys():
            ds = data[key]

            # Get polarization variables (spatial, with y/x dims)
            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            if not pols:
                result[key] = ds
                continue

            # Get reference DataArray to determine dimensions
            ref_da = ds[pols[0]]

            # Detect stack dimension (pair or date)
            stack_dims = [d for d in ['pair', 'date'] if d in ref_da.dims]
            if not stack_dims:
                raise ValueError(f"Data must have 'pair' or 'date' dimension, got: {ref_da.dims}")
            stack_dim = stack_dims[0]

            # Determine baseline_values for regression
            if hasattr(baseline, 'keys') and callable(baseline.keys):
                # Batch object - extract values for this key
                baseline_ds = baseline[key]
                # Get the first data variable or the DataArray itself
                if hasattr(baseline_ds, 'data_vars') and len(baseline_ds.data_vars) > 0:
                    baseline_var = list(baseline_ds.data_vars)[0]
                    baseline_values = np.asarray(baseline_ds[baseline_var].values, dtype=np.float64)
                else:
                    baseline_values = np.asarray(baseline_ds.values, dtype=np.float64)
            elif isinstance(baseline, str):
                # Use named variable or coordinate from data
                if baseline in ds.data_vars:
                    # It's a data variable (like 'BPR')
                    baseline_values = np.asarray(ds[baseline].values, dtype=np.float64)
                elif baseline in ref_da.coords:
                    # It's a coordinate
                    baseline_coord = ref_da.coords[baseline]
                    if np.issubdtype(baseline_coord.dtype, np.datetime64):
                        dates = pd.to_datetime(baseline_coord.values)
                        baseline_values = (dates - dates.min()).total_seconds().values / 86400.0
                    else:
                        baseline_values = baseline_coord.values.astype(np.float64)
                else:
                    raise ValueError(f"'{baseline}' not found in data variables or coordinates")
            elif isinstance(baseline, (xr.DataArray, pd.DataFrame, pd.Series)):
                baseline_values = np.asarray(baseline.values, dtype=np.float64)
            else:
                baseline_values = np.asarray(baseline, dtype=np.float64)

            if debug:
                print(f"DEBUG {key}: stack_dim={stack_dim}, n_samples={len(baseline_values)}, baseline range=[{baseline_values.min():.2f}, {baseline_values.max():.2f}]")

            result_ds = {}
            for pol in pols:
                data_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                # Ensure stack dimension is first
                if data_da.dims[0] != stack_dim:
                    data_da = data_da.transpose(stack_dim, ...)
                    if weight_da is not None:
                        weight_da = weight_da.transpose(stack_dim, ...)

                # Create wrapper function for blockwise
                def process_chunk(data_chunk, weight_chunk=None, baseline_values=baseline_values, device=device, degree=degree, wrap=wrap, iterations=iterations):
                    import torch
                    # Ensure 3D input (n_samples, y, x)
                    squeeze = False
                    if data_chunk.ndim == 2:
                        data_chunk = data_chunk[np.newaxis, ...]
                        if weight_chunk is not None:
                            weight_chunk = weight_chunk[np.newaxis, ...]
                        squeeze = True
                    result = Stack_detrend._regression1d_array(
                        data_chunk, baseline_values, weight_chunk, torch.device(device), degree, wrap, iterations
                    )
                    if squeeze:
                        result = result[0]
                    return result

                # Use da.blockwise for efficient dask integration
                # Rechunk to have all pairs in one chunk (regression needs all pairs together)
                data_dask = data_da.data
                if hasattr(data_dask, 'rechunk'):
                    data_dask = data_dask.rechunk({0: -1})
                dim_str = ''.join(chr(ord('a') + i) for i in range(data_dask.ndim))

                with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                    if weight_da is not None:
                        weight_dask = weight_da.data
                        if hasattr(weight_dask, 'rechunk'):
                            weight_dask = weight_dask.rechunk({0: -1})
                        def process_with_weight(data_block, weight_block):
                            return process_chunk(data_block, weight_block)
                        result_dask = da.blockwise(
                            process_with_weight, dim_str,
                            data_dask, dim_str,
                            weight_dask, dim_str,
                            dtype=np.float32,
                        )
                    else:
                        result_dask = da.blockwise(
                            process_chunk, dim_str,
                            data_dask, dim_str,
                            dtype=np.float32,
                        )

                # Rechunk back to per-slice for efficient downstream operations
                if hasattr(result_dask, 'rechunk'):
                    result_dask = result_dask.rechunk({0: 1})

                fit_da = xr.DataArray(
                    result_dask,
                    dims=data_da.dims,
                    coords=data_da.coords
                )

                result_ds[pol] = fit_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchWrap(result) if is_wrap else Batch(result)

    @staticmethod
    def _polyfit1d_pytorch(data, time_values, weight, sign, device, degree=0, wrap=False, iterations=1):
        """
        Fit 1D polynomial along first dimension at each pixel using PyTorch.

        Parameters
        ----------
        data : np.ndarray
            3D array (n_samples, ny, nx)
        time_values : np.ndarray
            1D array of time values for fitting (length n_samples)
        weight : np.ndarray or None
            Weight array, same shape as data
        sign : np.ndarray
            Sign array (n_samples,) to flip phase for pairs where date is rep
        device : torch.device
            PyTorch device
        degree : int
            Polynomial degree (0=mean, 1=linear)
        wrap : bool
            If True, use circular (sin/cos) fitting
        iterations : int
            Number of fitting iterations (for wrap=True, captures more trend per iteration)

        Returns
        -------
        np.ndarray or tuple
            If wrap=False: model coefficients (ny, nx) for the requested degree
            If wrap=True: (sin_coeff, cos_coeff) tuple of (ny, nx) arrays
        """
        import torch
        import numpy as np

        n_samples, ny, nx = data.shape
        n_pixels = ny * nx

        # Use float64 on CPU, float32 on GPU
        if device.type == 'cpu':
            dtype = torch.float64
        else:
            dtype = torch.float32

        # Apply sign to data (flip for pairs where date is rep)
        sign_3d = sign[:, None, None]
        data_signed = data * sign_3d
        if weight is not None:
            # Apply sqrt(weight) for weighted least squares
            data_signed = data_signed * np.sqrt(weight)

        # Normalize time values to [-1, 1] for numerical stability
        t_min, t_max = time_values.min(), time_values.max()
        t_range = t_max - t_min
        if t_range > 0:
            t_norm = 2 * (time_values - t_min) / t_range - 1
        else:
            t_norm = np.zeros_like(time_values)

        # Build design matrix: [1, t, t^2, ...]
        X = np.column_stack([t_norm**d for d in range(degree + 1)])
        X_t = torch.tensor(X, dtype=dtype, device=device)

        # Flatten spatial dimensions
        data_flat = data_signed.reshape(n_samples, n_pixels)
        data_t = torch.tensor(data_flat, dtype=dtype, device=device)

        # Find valid pixels
        valid_mask = torch.isfinite(data_t).all(dim=0)
        valid_indices = torch.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            if wrap:
                return np.full((ny, nx), np.nan, dtype=np.float32), np.full((ny, nx), np.nan, dtype=np.float32)
            else:
                return np.full((ny, nx), np.nan, dtype=np.float32)

        data_valid = data_t[:, valid_indices]  # (n_samples, n_valid)

        # Precompute pseudo-inverse for normal equations
        XtX = X_t.T @ X_t
        XtX_inv = torch.linalg.inv(XtX + 1e-10 * torch.eye(XtX.shape[0], dtype=dtype, device=device))
        XtX_inv_Xt = XtX_inv @ X_t.T  # (degree+1, n_samples)

        if wrap:
            # Iterative circular fitting
            current_data = data_valid.clone()
            cumulative_sin = torch.zeros(n_valid, dtype=dtype, device=device)
            cumulative_cos = torch.ones(n_valid, dtype=dtype, device=device)  # cos(0) = 1

            for _ in range(iterations):
                # Fit sin and cos components separately
                sin_data = torch.sin(current_data)
                cos_data = torch.cos(current_data)

                # coeffs = (X^T X)^-1 X^T y
                coeffs_sin = (XtX_inv_Xt @ sin_data).T  # (n_valid, degree+1)
                coeffs_cos = (XtX_inv_Xt @ cos_data).T

                # Extract coefficient for requested degree
                iter_sin = coeffs_sin[:, degree]  # (n_valid,)
                iter_cos = coeffs_cos[:, degree]

                # Accumulate using angle addition formulas:
                # sin(A+B) = sin(A)cos(B) + cos(A)sin(B)
                # cos(A+B) = cos(A)cos(B) - sin(A)sin(B)
                new_sin = cumulative_sin * iter_cos + cumulative_cos * iter_sin
                new_cos = cumulative_cos * iter_cos - cumulative_sin * iter_sin
                cumulative_sin = new_sin
                cumulative_cos = new_cos

                # Compute fitted angle for this iteration and subtract from data
                fit_angle = torch.atan2(iter_sin, iter_cos)
                current_data = current_data - fit_angle[None, :]

            # Put back into full arrays (NaN for invalid pixels)
            result_sin = torch.full((n_pixels,), float('nan'), dtype=dtype, device=device)
            result_cos = torch.full((n_pixels,), float('nan'), dtype=dtype, device=device)
            result_sin[valid_indices] = cumulative_sin
            result_cos[valid_indices] = cumulative_cos

            return (result_sin.cpu().numpy().reshape(ny, nx).astype(np.float32),
                    result_cos.cpu().numpy().reshape(ny, nx).astype(np.float32))
        else:
            # Standard polynomial fitting
            coeffs = (XtX_inv_Xt @ data_valid).T  # (n_valid, degree+1)

            # Extract coefficient for requested degree
            coeff = coeffs[:, degree]  # (n_valid,)

            # Put back into full array (NaN for invalid pixels)
            result = torch.full((n_pixels,), float('nan'), dtype=dtype, device=device)
            result[valid_indices] = coeff

            return result.cpu().numpy().reshape(ny, nx).astype(np.float32)

    @staticmethod
    def _regression1d_pairs_chunk(data_chunk, weight_chunk, ref_values, rep_values,
                                   device, degree, days_filter, count_filter, wrap, iterations):
        """
        Process a spatial chunk for regression1d_pairs.

        Parameters
        ----------
        data_chunk : np.ndarray
            3D array (n_pairs, chunk_y, chunk_x)
        weight_chunk : np.ndarray or None
            Weight array, same shape as data_chunk
        ref_values : np.ndarray
            1D array of ref dates as int64 (nanoseconds since epoch)
        rep_values : np.ndarray
            1D array of rep dates as int64 (nanoseconds since epoch)
        device : torch.device
            PyTorch device
        degree : int
            Polynomial degree (0=mean, 1=linear)
        days_filter : int or None
            Maximum time interval filter
        count_filter : int or None
            Maximum pairs per date filter
        wrap : bool
            Use circular fitting
        iterations : int
            Number of fitting iterations

        Returns
        -------
        np.ndarray
            Trend array (n_pairs, chunk_y, chunk_x)
        """
        import torch
        import numpy as np

        n_pairs, ny, nx = data_chunk.shape

        # Convert int64 nanoseconds to days (relative to first date)
        # This avoids datetime64 issues in the chunk function
        ns_per_day = 86400 * 1e9
        ref_days = ref_values / ns_per_day
        rep_days = rep_values / ns_per_day

        # Get unique dates
        all_days = np.concatenate([ref_days, rep_days])
        unique_days = np.unique(all_days)

        # Build per-date models
        if wrap:
            models_sin = {}
            models_cos = {}
        else:
            models = {}

        for date_day in unique_days:
            # Find pairs containing this date
            is_ref = np.isclose(ref_days, date_day)
            is_rep = np.isclose(rep_days, date_day)
            mask = is_ref | is_rep
            pair_indices = np.where(mask)[0]

            if len(pair_indices) == 0:
                continue

            # Compute time intervals (days) and signs
            time_values = []
            signs = []
            for idx in pair_indices:
                if is_rep[idx]:
                    # date is rep, pair goes ref -> date
                    days_val = date_day - ref_days[idx]
                    signs.append(-1.0)
                else:
                    # date is ref, pair goes date -> rep
                    days_val = rep_days[idx] - date_day
                    signs.append(1.0)
                time_values.append(days_val)

            time_values = np.array(time_values, dtype=np.float32)
            signs = np.array(signs, dtype=np.float32)

            # Sort by absolute time interval
            sort_idx = np.argsort(np.abs(time_values))
            time_values = time_values[sort_idx]
            signs = signs[sort_idx]
            pair_indices = pair_indices[sort_idx]

            # Apply count filter
            if count_filter is not None and len(pair_indices) > count_filter:
                time_values = time_values[:count_filter]
                signs = signs[:count_filter]
                pair_indices = pair_indices[:count_filter]

            # Apply days filter
            if days_filter is not None:
                keep = np.abs(time_values) <= days_filter
                time_values = time_values[keep]
                signs = signs[keep]
                pair_indices = pair_indices[keep]

            if len(pair_indices) == 0:
                continue

            # Extract data for selected pairs
            selected_data = data_chunk[pair_indices]
            selected_weight = weight_chunk[pair_indices] if weight_chunk is not None else None

            # Fit polynomial using PyTorch
            result = Stack_detrend._polyfit1d_pytorch(
                selected_data, time_values, selected_weight, signs,
                device, degree=degree, wrap=wrap, iterations=iterations
            )

            if wrap:
                sin_coeff, cos_coeff = result
                models_sin[date_day] = sin_coeff
                models_cos[date_day] = cos_coeff
            else:
                models[date_day] = result

        # Reconstruct pair-wise trends
        trend_data = np.full((n_pairs, ny, nx), np.nan, dtype=np.float32)

        if wrap:
            for i in range(n_pairs):
                ref_day = ref_days[i]
                rep_day = rep_days[i]

                # Find matching keys (use isclose for float comparison)
                ref_key = None
                rep_key = None
                for key in models_sin.keys():
                    if np.isclose(key, ref_day):
                        ref_key = key
                    if np.isclose(key, rep_day):
                        rep_key = key

                if ref_key is None or rep_key is None:
                    continue

                sin_ref = models_sin[ref_key]
                cos_ref = models_cos[ref_key]
                sin_rep = models_sin[rep_key]
                cos_rep = models_cos[rep_key]

                # Angle difference: atan2(sin(A-B), cos(A-B))
                sin_diff = sin_ref * cos_rep - cos_ref * sin_rep
                cos_diff = cos_ref * cos_rep + sin_ref * sin_rep
                trend_data[i] = np.arctan2(sin_diff, cos_diff)
        else:
            for i in range(n_pairs):
                ref_day = ref_days[i]
                rep_day = rep_days[i]

                # Find matching keys
                ref_key = None
                rep_key = None
                for key in models.keys():
                    if np.isclose(key, ref_day):
                        ref_key = key
                    if np.isclose(key, rep_day):
                        rep_key = key

                if ref_key is None or rep_key is None:
                    continue

                trend_data[i] = models[ref_key] - models[rep_key]

        return trend_data

    def regression1d_pairs(self, data, weight=None, degree=0, days=None, count=None, wrap=False, iterations=1, device='auto', debug=False):
        """
        Fit 1D polynomial trend along temporal pairs for each date using PyTorch.

        For each date, fits a polynomial along the time dimension using interferometric
        pairs that contain that date, then reconstructs the trend for each pair as
        the difference between the ref and rep date models.

        Parameters
        ----------
        data : Batch or BatchWrap
            Interferogram data with dimensions (pair, y, x).
        weight : Batch or None, optional
            Optional weights (e.g., correlation), same shape as data.
        degree : int
            Polynomial degree (0=mean, 1=linear). Default 0.
        days : int or None
            Maximum time interval (in days) to include. Default None (all pairs).
        count : int or None
            Maximum number of pairs per date to use. Default None (all pairs).
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1). For wrap=True, multiple
            iterations capture more of the trend. Each iteration fits the
            residual from the previous iteration and accumulates the result.
            Typically 2-3 iterations are sufficient.
        device : str
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Fitted trend values, same shape as input data (lazy).

        Examples
        --------
        >>> # Fit mean (degree=0) for each date
        >>> trend = stack.regression1d_pairs(intf, degree=0, days=100)
        >>> detrended = intf - trend
        >>>
        >>> # Fit linear trend (degree=1) for wrapped phase with iterations
        >>> trend = stack.regression1d_pairs(intf, degree=1, wrap=True, iterations=3)
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np
        import pandas as pd

        # Auto-detect device
        device = Stack_detrend._get_torch_device(device, debug=debug)

        if debug:
            print(f"DEBUG: regression1d_pairs using device={device}")

        # Handle Batch input
        if isinstance(data, (Batch, BatchWrap)):
            is_wrap = isinstance(data, BatchWrap)
            results = {}
            for burst_id in data.keys():
                burst_ds = data[burst_id]
                burst_weight = weight[burst_id] if weight is not None else None

                # Process all polarizations
                result_ds = {}
                for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                    data_da = burst_ds[pol]
                    weight_da = burst_weight[pol] if burst_weight is not None else None

                    if debug:
                        pairs, dates = self._get_pairs(data_da, dates=True)
                        print(f"DEBUG {burst_id}: {len(dates)} dates, {len(pairs)} pairs")

                    # Get ref/rep as int64 nanoseconds (avoids datetime64 serialization issues)
                    ref_values = burst_ds['ref'].values.astype('datetime64[ns]').astype(np.int64)
                    rep_values = burst_ds['rep'].values.astype('datetime64[ns]').astype(np.int64)

                    # Ensure pair dimension is first
                    if data_da.dims[0] != 'pair':
                        data_da = data_da.transpose('pair', ...)
                        if weight_da is not None:
                            weight_da = weight_da.transpose('pair', ...)

                    # Get dask arrays and rechunk (all pairs in one chunk)
                    data_dask = data_da.data
                    if hasattr(data_dask, 'rechunk'):
                        data_dask = data_dask.rechunk({0: -1})

                    # Create wrapper that captures parameters
                    def make_wrapper(ref_vals, rep_vals, dev, deg, days_f, count_f, wrp, iters):
                        def process_chunk(data_block, weight_block=None):
                            return Stack_detrend._regression1d_pairs_chunk(
                                data_block, weight_block, ref_vals, rep_vals,
                                torch.device(dev), deg, days_f, count_f, wrp, iters
                            )
                        return process_chunk

                    wrapper = make_wrapper(ref_values, rep_values, str(device), degree, days, count, wrap, iterations)

                    dim_str = ''.join(chr(ord('a') + i) for i in range(data_dask.ndim))

                    with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                        if weight_da is not None:
                            weight_dask = weight_da.data
                            if hasattr(weight_dask, 'rechunk'):
                                weight_dask = weight_dask.rechunk({0: -1})

                            def make_weighted_wrapper(wrapper_fn):
                                def process_with_weight(data_block, weight_block):
                                    return wrapper_fn(data_block, weight_block)
                                return process_with_weight

                            result_dask = da.blockwise(
                                make_weighted_wrapper(wrapper), dim_str,
                                data_dask, dim_str,
                                weight_dask, dim_str,
                                dtype=np.float32,
                            )
                        else:
                            result_dask = da.blockwise(
                                wrapper, dim_str,
                                data_dask, dim_str,
                                dtype=np.float32,
                            )

                    # Rechunk back to per-slice for efficient downstream operations
                    if hasattr(result_dask, 'rechunk'):
                        result_dask = result_dask.rechunk({0: 1})

                    # Create output DataArray
                    trend_da = xr.DataArray(
                        result_dask,
                        dims=data_da.dims,
                        coords=data_da.coords
                    )
                    result_ds[pol] = trend_da

                results[burst_id] = xr.Dataset(result_ds, attrs=burst_ds.attrs)

            return BatchWrap(results) if is_wrap else Batch(results)

        else:
            raise ValueError("data must be a Batch or BatchWrap object")

    def _polyfit(self, data, weight=None, degree=0, days=None, count=None, wrap=False):
        print ('NOTE: Function is deprecated. Use Stack.regression_pairs() instead.')
        return self.regression_pairs(data=data, weight=weight, degree=degree, days=days, count=count, wrap=wrap)

    def _regression_pairs(self, data, weight=None, degree=0, days=None, count=None, wrap=False):
        import xarray as xr
        import pandas as pd
        import numpy as np
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')

        multi_index = None
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            multi_index = data['stack']
            data = data.reset_index('stack')
            if weight is not None:
                if not ('stack' in weight.dims and isinstance(weight.coords['stack'].to_index(), pd.MultiIndex)):
                    raise ValueError('ERROR: "weight", if provided, must be stacked consistently with "data".')
                data = data.reset_index('stack')
        else:
            if 'stack' in weight.dims and isinstance(weight.coords['stack'].to_index(), pd.MultiIndex):
                raise ValueError('ERROR: "weight", if provided, must be stacked consistently with "data".')

        pairs, dates = self._get_pairs(data, dates=True)

        models = []
        if wrap:
            models_sin = []
            models_cos = []

        for date in dates:
            data_pairs = pairs[(pairs.ref==date)|(pairs.rep==date)].pair.values
            if weight is None:
                stack = data.sel(pair=data_pairs)
            else:
                stack = data.sel(pair=data_pairs) * np.sqrt(weight.sel(pair=data_pairs))
            del data_pairs

            stack_days = xr.where(stack.ref < pd.Timestamp(date),
                           (stack.ref - stack.rep).dt.days,
                           (stack.rep - stack.ref).dt.days)
            # select smallest intervals
            stack_days_selected = stack_days[np.argsort(np.abs(stack_days.values))][:count]
            if days is not None:
                stack_days_selected = stack_days_selected[np.abs(stack_days_selected)<=days]

            selected_pairs = (np.sign(stack_days)*stack).assign_coords(time=stack_days)\
                [stack.pair.isin(stack_days_selected.pair)]\
                .swap_dims({'pair': 'time'})\
                .sortby(['ref', 'rep'])
            del stack, stack_days, stack_days_selected

            if not wrap:
                linear_fit = selected_pairs.polyfit(dim='time', deg=degree)
                model = linear_fit.polyfit_coefficients.sel(degree=degree).astype(np.float32)
                models.append(model.assign_coords(date=pd.to_datetime(date)))
                del model, linear_fit
            else:
                # fit sine and cosine components
                linear_fit_sin = np.sin(selected_pairs).polyfit(dim='time', deg=degree)
                linear_fit_cos = np.cos(selected_pairs).polyfit(dim='time', deg=degree)

                model_sin = linear_fit_sin.polyfit_coefficients.sel(degree=degree).astype(np.float32)
                model_cos = linear_fit_cos.polyfit_coefficients.sel(degree=degree).astype(np.float32)

                models_sin.append(model_sin.assign_coords(date=pd.to_datetime(date)))
                models_cos.append(model_cos.assign_coords(date=pd.to_datetime(date)))
                del model_sin, model_cos, linear_fit_sin, linear_fit_cos

            del selected_pairs

        if not wrap:
            model = xr.concat(models, dim='date')
            del models
            out = xr.concat(
                [
                    (model.sel(date=ref).drop('date') - model.sel(date=rep).drop('date'))
                    .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep)
                    for ref, rep in zip(pairs['ref'], pairs['rep'])
                ],
                dim='pair'
            ).rename(data.name)
        else:
            # combine separate sin and cos models
            model_sin = xr.concat(models_sin, dim='date')
            model_cos = xr.concat(models_cos, dim='date')
            del models_sin, models_cos

            angle_diffs = []
            for ref, rep in zip(pairs['ref'], pairs['rep']):
                sin_ref = model_sin.sel(date=ref).drop('date')
                cos_ref = model_cos.sel(date=ref).drop('date')
                sin_rep = model_sin.sel(date=rep).drop('date')
                cos_rep = model_cos.sel(date=rep).drop('date')

                # compute angle differences using sin/cos difference formula
                # sin(A−B) = sin A * cos B − cos A * sin B
                sin_diff = sin_ref * cos_rep - cos_ref * sin_rep
                # cos(A−B) = cos A * cos B+ sin A * sin B
                cos_diff = cos_ref * cos_rep + sin_ref * sin_rep
                del sin_ref, cos_ref, sin_rep, cos_rep

                angle_diff = np.arctan2(sin_diff, cos_diff)\
                             .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep)
                angle_diffs.append(angle_diff)
                del angle_diff, sin_diff, cos_diff

            out = xr.concat(angle_diffs, dim='pair').rename(data.name)
            del angle_diffs

        if multi_index is not None:
            return out.assign_coords(stack=multi_index)
        return out

    def _turbulence(self, phase, weight=None):
        import xarray as xr
        import pandas as pd

        print ('NOTE: this function is deprecated, use instead Stack.polyfit()')

        pairs, dates = self._get_pairs(phase, dates=True)

        turbos = []
        for date in dates:
            ref = pairs[pairs.ref==date]
            rep = pairs[pairs.rep==date]
            #print (date, len(ref), len(rep))
            ref_data = phase.sel(pair=ref.pair.values)
            #print (ref_data)
            rep_data = phase.sel(pair=rep.pair.values)
            #print (rep_data)
            if weight is not None:
                ref_weight = weight.sel(pair=ref.pair.values)
                rep_weight = weight.sel(pair=rep.pair.values)
                turbo = xr.concat([ref_data*ref_weight, -rep_data*rep_weight], dim='pair').sum('pair')/\
                    xr.concat([ref_weight, rep_weight], dim='pair').sum('pair')
                del ref_weight, rep_weight
            else:
                turbo = xr.concat([ref_data, -rep_data], dim='pair').mean('pair')
            del ref_data, rep_data
            turbos.append(turbo.assign_coords(date=pd.to_datetime(date)))
            del turbo
        turbo = xr.concat(turbos, dim='date')
        del turbos

        phase_turbo = xr.concat([(turbo.sel(date=ref).drop('date') - turbo.sel(date=rep).drop('date'))\
                                 .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep) \
                          for ref, rep in zip(pairs['ref'], pairs['rep'])], dim='pair')

        return phase_turbo.rename('turbulence')

    def _velocity(self, data):
        import pandas as pd
        import numpy as np
        #years = ((data.date.max() - data.date.min()).dt.days/365.25).item()
        #nanoseconds = (data.date.max().astype(int) - data.date.min().astype(int)).item()
        #print ('years', np.round(years, 3), 'nanoseconds', nanoseconds)
        multi_index = None
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            multi_index = data.coords['stack']
            # replace multiindex by sequential numbers 0,1,...
            data = data.reset_index('stack')
        #velocity = nanoseconds*data.polyfit('date', 1).polyfit_coefficients.sel(degree=1)/years
        nanoseconds_per_year = 365.25*24*60*60*1e9
        # calculate slope per year
        velocity = nanoseconds_per_year*data.polyfit('date', 1).polyfit_coefficients.sel(degree=1).astype(np.float32).rename('trend')
        if multi_index is not None:
            return velocity.assign_coords(stack=multi_index)
        return velocity

    def plot_velocity(self, data, caption='Velocity, [rad/year]',
                      quantile=None, vmin=None, vmax=None, symmetrical=False, aspect=None, alpha=1, **kwargs):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            data = data.unstack('stack')

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"
    
        if quantile is not None:
            vmin, vmax = np.nanquantile(data, quantile)
    
        # define symmetrical boundaries
        if symmetrical is True and vmax > 0:
            minmax = max(abs(vmin), vmax)
            vmin = -minmax
            vmax =  minmax
    
        plt.figure()
        data.plot.imshow(vmin=vmin, vmax=vmax, alpha=alpha, cmap='turbo', interpolation='none')
        #self.plot_AOI(**kwargs)
        #self.plot_POI(**kwargs)
        if aspect is not None:
            plt.gca().set_aspect(aspect)
        #plt.xlabel('Range')
        #plt.ylabel('Azimuth')
        plt.title(caption)

    def plot_velocity_los_mm(self, data, caption='Velocity, [mm/year]',
                      quantile=None, vmin=None, vmax=None, symmetrical=False, aspect=None, alpha=1, **kwargs):
        self.plot_velocity(self.los_displacement_mm(data),
                           caption=caption, aspect=aspect, alpha=alpha,
                           quantile=quantile, vmin=vmin, vmax=vmax, symmetrical=symmetrical, **kwargs)

