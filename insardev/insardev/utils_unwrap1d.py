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
Static utility functions for 1D temporal phase unwrapping.

These functions contain the core algorithms for temporal phase unwrapping
and least squares network inversion.
"""
import numpy as np


def wrap(data_pairs):
    """Wrap phase to [-pi, pi] range."""
    import xarray as xr
    import dask

    if isinstance(data_pairs, xr.DataArray):
        return xr.DataArray(dask.array.mod(data_pairs.data + np.pi, 2 * np.pi) - np.pi, data_pairs.coords)\
            .rename(data_pairs.name)
    return np.mod(data_pairs + np.pi, 2 * np.pi) - np.pi


def build_incidence_matrix(pair_dates):
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


def irls_solve_1d(phi, W_corr, A_t, dev, max_iter=5, epsilon=0.1,
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


def unwrap1d_pairs_numpy(phase_stack, weight_stack, pair_dates, device='auto',
                         max_iter=5, epsilon=0.1, batch_size=None, debug=False):
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
    from .BatchCore import BatchCore

    # Accept chunk lists or 3D array → flat (n_pairs, n_pixels) views.
    if isinstance(phase_stack, list):
        flat_chunks = [np.asarray(c) for c in phase_stack]
        height, width = flat_chunks[0].shape[1], flat_chunks[0].shape[2]
        n_pixels = height * width
        n_pairs = sum(c.shape[0] for c in flat_chunks)
        flat_chunks = [c.reshape(c.shape[0], n_pixels) for c in flat_chunks]
        multi_chunk = True
    else:
        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width
        flat_chunks = [phase_stack.reshape(n_pairs, n_pixels)]
        multi_chunk = False

    if weight_stack is not None:
        if isinstance(weight_stack, list):
            flat_w_chunks = [np.asarray(c) for c in weight_stack]
            flat_w_chunks = [c.reshape(c.shape[0], n_pixels) for c in flat_w_chunks]
        else:
            flat_w_chunks = [weight_stack.reshape(weight_stack.shape[0], n_pixels)]
    else:
        flat_w_chunks = None

    # Build incidence matrix
    A, dates = build_incidence_matrix(pair_dates)
    dev = BatchCore._get_torch_device(device)

    if debug:
        print(f'IRLS 1D unwrap: {n_pairs} pairs, {len(dates)} dates, {n_pixels} pixels')
        print(f'  Device: {dev}, batch_size: {batch_size}')

    # Move matrix to device
    A_t = torch.from_numpy(A).to(dev)

    # Compute batch size from dask config if not provided
    if batch_size is None:
        from .utils_dask import get_dask_chunk_size_mb
        n_intervals = len(dates) - 1
        elem_bytes = max(1, n_pairs * n_intervals * 4)
        batch_size = max(1024, (get_dask_chunk_size_mb() * 1024 * 1024) // elem_bytes)

    # Gather size: pixel columns to concatenate at once (~dask chunk budget)
    from .utils_dask import get_dask_chunk_size_mb
    gather_size = max(batch_size, (get_dask_chunk_size_mb() * 1024 * 1024) // max(1, n_pairs * 4))

    unwrapped_flat = np.zeros((n_pairs, n_pixels), dtype=np.float32)

    for g_start in range(0, n_pixels, gather_size):
        g_end = min(g_start + gather_size, n_pixels)

        # Concatenate per-pair slices for this pixel range (one concat per gather block)
        if multi_chunk:
            phases_block = np.concatenate([c[:, g_start:g_end] for c in flat_chunks], axis=0)
            weights_block = np.concatenate([c[:, g_start:g_end] for c in flat_w_chunks], axis=0) \
                if flat_w_chunks is not None else None
        else:
            phases_block = flat_chunks[0][:, g_start:g_end]
            weights_block = flat_w_chunks[0][:, g_start:g_end] if flat_w_chunks is not None else None

        # Inner loop: torch batches from the concatenated block
        block_pixels = g_end - g_start
        for b_start in range(0, block_pixels, batch_size):
            b_end = min(b_start + batch_size, block_pixels)

            phi = torch.from_numpy(phases_block[:, b_start:b_end].astype(np.float32)).to(dev)
            if weights_block is not None:
                W = torch.from_numpy(weights_block[:, b_start:b_end].astype(np.float32)).to(dev)
            else:
                W = torch.ones(n_pairs, b_end - b_start, device=dev)

            result = irls_solve_1d(
                phi, W, A_t, dev, max_iter=max_iter, epsilon=epsilon,
                return_increments=False
            )
            unwrapped_flat[:, g_start + b_start:g_start + b_end] = result.cpu().numpy()

            del phi, W, result
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

        del phases_block, weights_block

    # Reshape back
    return unwrapped_flat.reshape(n_pairs, height, width)


def unwrap1d_to_dates_numpy(phase_stack, weight_stack, pair_dates, device='auto',
                            max_iter=5, epsilon=0.1, batch_size=None,
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
    from .BatchCore import BatchCore

    n_pairs, height, width = phase_stack.shape
    n_pixels = height * width

    # Build incidence matrix
    A, dates = build_incidence_matrix(pair_dates)
    n_intervals = len(dates) - 1
    dev = BatchCore._get_torch_device(device)

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

    # Compute batch size from dask config if not provided
    if batch_size is None:
        from .utils_dask import get_dask_chunk_size_mb
        elem_bytes = max(1, n_pairs * n_intervals * 4)
        batch_size = max(1024, (get_dask_chunk_size_mb() * 1024 * 1024) // elem_bytes)

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
        unwrapped = irls_solve_1d(
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

    # Build time series — np.cumsum propagates NaN (all-NaN pixels stay NaN)
    ts = np.zeros((len(dates), n_pixels), dtype=np.float32)
    if cumsum:
        ts[1:, :] = np.cumsum(increments_flat, axis=0)
    else:
        ts[1:, :] = increments_flat

    # Reshape: (n_dates, n_pixels) -> (n_dates, height, width)
    time_series = ts.reshape(len(dates), height, width)

    return time_series, dates


def lstsq_to_dates_numpy(phase_stack, weight_stack, pair_dates, device='auto',
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
    from .BatchCore import BatchCore

    # Accept chunk lists or 3D array → flat (n_pairs, n_pixels) views.
    if isinstance(phase_stack, list):
        flat_chunks = [np.asarray(c) for c in phase_stack]
        height, width = flat_chunks[0].shape[1], flat_chunks[0].shape[2]
        n_pixels = height * width
        n_pairs = sum(c.shape[0] for c in flat_chunks)
        flat_chunks = [c.reshape(c.shape[0], n_pixels) for c in flat_chunks]
        multi_chunk = True
    else:
        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width
        flat_chunks = [phase_stack.reshape(n_pairs, n_pixels)]
        multi_chunk = False

    if weight_stack is not None:
        if isinstance(weight_stack, list):
            flat_w_chunks = [np.asarray(c) for c in weight_stack]
            flat_w_chunks = [c.reshape(c.shape[0], n_pixels) for c in flat_w_chunks]
        else:
            flat_w_chunks = [weight_stack.reshape(weight_stack.shape[0], n_pixels)]
    else:
        flat_w_chunks = None

    # Build incidence matrix
    A, dates = build_incidence_matrix(pair_dates)
    n_dates = len(dates)
    n_intervals = n_dates - 1

    dev = BatchCore._get_torch_device(device)
    if debug:
        print(f'lstsq: {n_pairs} pairs -> {n_dates} dates, {n_pixels} pixels, device={dev}')

    # Compute batch size from dask config
    from .utils_dask import get_dask_chunk_size_mb
    elem_bytes = max(1, n_pairs * n_intervals * 4)
    batch_size = max(1024, (get_dask_chunk_size_mb() * 1024 * 1024) // elem_bytes)

    # Gather size: pixel columns to concatenate at once (~dask chunk budget)
    gather_size = max(batch_size, (get_dask_chunk_size_mb() * 1024 * 1024) // max(1, n_pairs * 4))

    A_t = torch.from_numpy(A.astype(np.float32)).to(dev)
    reg = 1e-4 * torch.eye(n_intervals, device='cpu')

    increments = np.full((n_pixels, n_intervals), np.nan, dtype=np.float32)

    for g_start in range(0, n_pixels, gather_size):
        g_end = min(g_start + gather_size, n_pixels)

        # Concatenate per-pair slices for this pixel range (one concat per gather block)
        if multi_chunk:
            phases_block = np.concatenate([c[:, g_start:g_end] for c in flat_chunks], axis=0)
            weights_block = np.concatenate([c[:, g_start:g_end] for c in flat_w_chunks], axis=0) \
                if flat_w_chunks is not None else None
        else:
            phases_block = flat_chunks[0][:, g_start:g_end]
            weights_block = flat_w_chunks[0][:, g_start:g_end] if flat_w_chunks is not None else None

        # Inner loop: torch batches from the concatenated block
        block_pixels = g_end - g_start
        for b_start in range(0, block_pixels, batch_size):
            b_end = min(b_start + batch_size, block_pixels)
            b_len = b_end - b_start

            phi_b = torch.from_numpy(phases_block[:, b_start:b_end].astype(np.float32)).to(dev)
            nan_mask_b = torch.isnan(phi_b)
            phi_b = torch.where(nan_mask_b, torch.zeros_like(phi_b), phi_b)

            if weights_block is not None:
                w_batch = weights_block[:, b_start:b_end].astype(np.float32)
                np.clip(w_batch, 1e-6, 1 - 1e-6, out=w_batch)
                W_b = torch.from_numpy(w_batch).to(dev)
                del w_batch
                nan_mask_w = torch.isnan(W_b)
                nan_mask_b = nan_mask_b | nan_mask_w
                W_b = torch.where(nan_mask_b, torch.zeros_like(W_b), W_b)
                W_clamped = torch.clamp(W_b, 0.0, 0.9999)
                W_lstsq_b = W_clamped / torch.sqrt(1 - W_clamped**2)
            else:
                W_lstsq_b = torch.where(nan_mask_b, torch.zeros(1, device=dev),
                                         torch.ones(1, device=dev)).expand(n_pairs, b_len)

            phi_T = phi_b.T
            W_T = W_lstsq_b.T

            WA = W_T.unsqueeze(2) * A_t.unsqueeze(0)
            Wb = (W_T * phi_T).unsqueeze(2)

            WA_cpu = torch.nan_to_num(WA.cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            Wb_cpu = torch.nan_to_num(Wb.cpu(), nan=0.0, posinf=0.0, neginf=0.0)

            result = torch.linalg.lstsq(WA_cpu, Wb_cpu)
            x = result.solution.squeeze(2)

            all_nan = nan_mask_b.all(dim=0).cpu()
            x[all_nan, :] = float('nan')
            increments[g_start + b_start:g_start + b_end] = x.numpy()

        del phases_block, weights_block

    # Build time series — np.cumsum propagates NaN (all-NaN pixels stay NaN)
    ts = np.zeros((n_pixels, n_dates), dtype=np.float32)
    if cumsum:
        ts[:, 1:] = np.cumsum(increments, axis=1)
    else:
        ts[:, 1:] = increments

    # Reshape: (n_pixels, n_dates) -> (n_dates, height, width)
    time_series = ts.T.reshape(n_dates, height, width)

    # Cleanup GPU memory
    if dev.type == 'mps':
        torch.mps.empty_cache()
    elif dev.type == 'cuda':
        torch.cuda.empty_cache()

    return time_series, dates
