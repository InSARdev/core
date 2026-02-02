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
    from .BatchCore import BatchCore

    n_pairs, height, width = phase_stack.shape
    n_pixels = height * width

    # Build incidence matrix
    A, dates = build_incidence_matrix(pair_dates)
    dev = BatchCore._get_torch_device(device)

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
        result = irls_solve_1d(
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


def unwrap1d_to_dates_numpy(phase_stack, weight_stack, pair_dates, device='auto',
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

    n_pairs, height, width = phase_stack.shape
    n_pixels = height * width

    # Build incidence matrix
    A, dates = build_incidence_matrix(pair_dates)
    n_dates = len(dates)
    n_intervals = n_dates - 1

    dev = BatchCore._get_torch_device(device)
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
