# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""
Static utility functions for detrending operations.

These functions contain the core algorithms for 1D and 2D polynomial
trend fitting using PyTorch for GPU acceleration.
"""
import numpy as np
import threading

# Lock for CUDA torch.linalg lazy initialization (prevents multi-threading bug)
# See: https://github.com/pytorch/pytorch/issues/90613
_linalg_init_lock = threading.Lock()
_linalg_initialized = False


def trend1d_array(data, dim_values, weight, device, degree=1, intercept=True, slope=True):
    """
    Fit 1D polynomial along first dimension at each (y, x) pixel using PyTorch.

    Two modes:
    - Complex input: unit-circle fitting (normalize to unit magnitude, fit complex
      polynomial, normalize result), returns complex64 unit-magnitude trend
    - Real input: standard polynomial fit, returns float32 trend

    Parameters
    ----------
    data : np.ndarray or list
        3D array (n_samples, y, x) or list of chunk arrays.
        Polynomial fit along first dimension. Real or complex.
    dim_values : np.ndarray
        1D array of x-values for fitting (length n_samples)
    weight : np.ndarray or list or None
        Weight array (real), same shape as data, or list of chunk arrays.
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (1=linear, 2=quadratic, etc.)
    intercept : bool
        If True, include intercept (constant term) in output. If False, zero it out.
    slope : bool
        If True, include slope (and higher-order terms) in output. If False, zero them out.

    Returns
    -------
    np.ndarray
        Fitted values, shape (n_samples, y, x).
        Complex input returns complex64 unit-magnitude trend.
        Real input returns float32 trend.
    """
    if degree == 0 and slope:
        raise ValueError("slope=True requires degree >= 1 (degree=0 has no slope to fit)")

    import torch
    from .utils_dask import get_dask_chunk_size_mb

    # Accept chunk lists or 3D array.
    # Build flat chunk views for on-demand pixel batch extraction.
    if isinstance(data, list):
        chunks = [np.asarray(c) for c in data]
        ny, nx = chunks[0].shape[1], chunks[0].shape[2]
        n_pixels = ny * nx
        n_samples = sum(c.shape[0] for c in chunks)
        is_complex = np.iscomplexobj(chunks[0])
        # Flatten each chunk to (chunk_n, n_pixels) — reshape is a view
        flat_chunks = [c.reshape(c.shape[0], n_pixels) for c in chunks]
    else:
        n_samples, ny, nx = data.shape
        n_pixels = ny * nx
        is_complex = np.iscomplexobj(data)
        flat_chunks = [data.reshape(n_samples, n_pixels)]

    if isinstance(weight, list):
        w_chunks = [np.asarray(c) for c in weight]
        flat_w_chunks = [c.reshape(c.shape[0], n_pixels) for c in w_chunks]
    elif weight is not None:
        flat_w_chunks = [weight.reshape(weight.shape[0], n_pixels)]
    else:
        flat_w_chunks = None

    out_dtype = np.complex64 if is_complex else np.float32

    # Helper: gather pixel columns from chunk list → (n_samples, len(idx))
    def _gather_cols(fc, idx):
        if len(fc) == 1:
            return fc[0][:, idx]
        parts = [c[:, idx] for c in fc]
        return np.concatenate(parts, axis=0)

    # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # Normalize dim_values so that 0 maps to 0 (intercept = value at dim=0)
    dim_absmax = np.max(np.abs(dim_values))
    if dim_absmax > 0:
        dim_norm = dim_values / dim_absmax
    else:
        dim_norm = np.zeros_like(dim_values)

    # Build Vandermonde matrix: [1, x, x^2, ..., x^degree]
    X = np.column_stack([dim_norm**d for d in range(degree + 1)])
    X_t = torch.tensor(X, dtype=dtype, device=device)

    # Compute per-pixel valid sample count; require degree+2 minimum for fit
    # (degree+1 gives exact fit with zero residual — not meaningful).
    # NaN samples get zero weight automatically (partial NaN always handled).
    min_valid = degree + 2

    valid_count = np.zeros(n_pixels, dtype=np.int32)
    if is_complex:
        any_nonzero = np.zeros(n_pixels, dtype=bool)
        for fc in flat_chunks:
            finite = np.isfinite(fc.real) & np.isfinite(fc.imag)
            valid_count += np.sum(finite, axis=0).astype(np.int32)
            any_nonzero |= np.any((fc != 0) & finite, axis=0)
        valid_mask = (valid_count >= min_valid) & any_nonzero
        del any_nonzero
    else:
        for fc in flat_chunks:
            valid_count += np.sum(np.isfinite(fc), axis=0).astype(np.int32)
        valid_mask = valid_count >= min_valid

    has_partial_nan = np.any(valid_count < n_samples) if valid_mask.any() else False
    del valid_count

    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    del valid_mask

    if n_valid == 0:
        out_shape = (n_samples, ny, nx)
        if is_complex:
            return np.full(out_shape, complex('nan'), dtype=np.complex64)
        return np.full(out_shape, np.nan, dtype=np.float32)

    # Precompute shared pseudo-inverse (tiny: degree+1 × n_samples)
    XtX = X_t.T @ X_t
    reg = 1e-10 * torch.eye(XtX.shape[0], dtype=dtype, device=device)
    XtX_inv_Xt = torch.linalg.inv(XtX + reg) @ X_t.T  # (degree+1, n_samples)

    # Compute pixel batch size from dask block budget
    elem_bytes = n_samples * (16 if is_complex else 8)
    budget_bytes = get_dask_chunk_size_mb() * 1024 * 1024
    batch_pixels = max(1024, budget_bytes // elem_bytes)

    # Initialize output in numpy
    nan_val = complex(float('nan'), 0) if is_complex else np.nan
    result_flat = np.full((n_samples, n_pixels), nan_val, dtype=out_dtype)

    # Process valid pixels in batches — gather only needed columns from chunks
    for b_start in range(0, n_valid, batch_pixels):
        b_end = min(b_start + batch_pixels, n_valid)
        idx = valid_indices[b_start:b_end]

        # Gather batch columns from chunk list: O(n_samples × batch_size)
        batch_data = _gather_cols(flat_chunks, idx)

        if is_complex:
            # For complex data: extract angles and fit with iterative 2π jump
            # correction (same principle as irls_solve_1d in unwrap1d).
            # 1. Fit polynomial to angles + jumps*2π with IRLS weights
            # 2. Update jumps: k = round((fit - angles) / 2π)
            # 3. Wrapped residuals → update IRLS weights: 1/(|r| + ε)
            # 4. Converge when jumps stabilize
            batch_abs = np.abs(batch_data)
            with np.errstate(invalid='ignore', divide='ignore'):
                angles_np = np.where(batch_abs > 0, np.angle(batch_data), np.nan)
            del batch_abs, batch_data

            angles_raw = torch.tensor(angles_np, dtype=dtype, device=device)
            del angles_np
            nan_mask = ~torch.isfinite(angles_raw)
            angles_safe = torch.where(nan_mask, torch.zeros(1, dtype=dtype, device=device), angles_raw)
            del angles_raw

            # Setup correlation weight + NaN masking (constant across iterations)
            has_weight_corr = flat_w_chunks is not None
            if has_weight_corr:
                batch_w = _gather_cols(flat_w_chunks, idx)
                sqrt_w_corr = torch.sqrt(torch.tensor(batch_w, dtype=dtype, device=device))
                del batch_w
            else:
                sqrt_w_corr = torch.ones_like(angles_safe)

            if has_partial_nan:
                sqrt_w_corr = sqrt_w_corr * (~nan_mask).to(dtype)

            # IRLS iteration with jump correction
            epsilon = 0.1
            convergence_threshold = 1e-3
            jumps = torch.zeros_like(angles_safe)  # (n_samples, batch_pixels)
            W_irls = torch.ones_like(angles_safe)   # (n_samples, batch_pixels)

            for iteration in range(10):
                data_b = angles_safe + jumps * (2 * np.pi)
                sqrt_w = sqrt_w_corr * torch.sqrt(W_irls)

                # Weighted polynomial fit
                y_w = (data_b * sqrt_w).T  # (batch, samp)
                X_wi = X_t[None, :, :] * sqrt_w.T[:, :, None]
                XtX_wi = X_wi.transpose(1, 2) @ X_wi + reg
                Xty = X_wi.transpose(1, 2) @ y_w.unsqueeze(-1)
                coeffs = torch.linalg.solve(XtX_wi, Xty)

                fit_raw = X_t @ coeffs.squeeze(-1).T  # (n_samples, batch_pixels)

                # Update integer 2π jumps
                new_jumps = torch.round((fit_raw - angles_safe) / (2 * np.pi))
                new_jumps[nan_mask] = 0

                # Wrapped residuals → IRLS weights (L1-norm, same as unwrap1d)
                residuals = data_b - fit_raw
                residuals = torch.atan2(torch.sin(residuals), torch.cos(residuals))
                W_irls_new = 1.0 / (torch.abs(residuals) + epsilon)
                W_irls_new[nan_mask] = 0

                # Check convergence: jumps stabilized and weights converged
                weight_change = torch.abs(W_irls_new - W_irls).mean()
                jumps_stable = (new_jumps == jumps).all()
                if jumps_stable and weight_change < convergence_threshold:
                    break

                jumps = new_jumps
                W_irls = W_irls_new

            if not intercept and not slope:
                fit_raw = torch.zeros_like(fit_raw)
            elif not intercept:
                c0 = coeffs.squeeze(-1)[:, 0]  # (batch_pixels,)
                fit_raw = fit_raw - c0[None, :]
            elif not slope:
                c0 = coeffs.squeeze(-1)[:, 0]
                fit_raw = c0[None, :].expand_as(fit_raw)

            result_flat[:, idx] = torch.exp(1j * fit_raw.to(cdtype)).cpu().numpy().astype(out_dtype)

        else:
            data_b = torch.tensor(batch_data, dtype=dtype, device=device)
            del batch_data

            has_weight = flat_w_chunks is not None
            if has_weight:
                batch_w = _gather_cols(flat_w_chunks, idx)
                sqrt_w = torch.sqrt(torch.tensor(batch_w, dtype=dtype, device=device))
                del batch_w

            # Handle partial NaN: zero out NaN values and give them zero weight
            if has_partial_nan:
                nan_mask = ~torch.isfinite(data_b)
                data_b = torch.where(nan_mask, torch.zeros(1, dtype=dtype, device=device), data_b)
                nan_w = (~nan_mask).to(dtype)  # 1 for valid, 0 for NaN
                if has_weight:
                    sqrt_w = sqrt_w * nan_w
                else:
                    sqrt_w = nan_w
                    has_weight = True

            # Real polynomial fit
            if has_weight:
                y_weighted = (data_b * sqrt_w).T
                X_batch = X_t[None, :, :] * sqrt_w.T[:, :, None]
                XtX_b = X_batch.transpose(1, 2) @ X_batch
                Xty = X_batch.transpose(1, 2) @ y_weighted.unsqueeze(-1)
                coeffs = torch.linalg.solve(XtX_b + reg, Xty)
            else:
                coeffs = (XtX_inv_Xt @ data_b).T.unsqueeze(-1)

            fit = (X_t @ coeffs.squeeze(-1).T).T
            if not intercept and not slope:
                fit = torch.zeros_like(fit)
            elif not intercept:
                c0 = coeffs.squeeze(-1)[:, 0]
                fit = fit - c0.unsqueeze(-1)
            elif not slope:
                c0 = coeffs.squeeze(-1)[:, 0]
                fit = c0.unsqueeze(-1).expand_as(fit)

            result_flat[:, idx] = fit.T.cpu().numpy().astype(out_dtype)

    out = result_flat.reshape(n_samples, ny, nx)

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return out


# Backward compatibility alias
regression1d_array = trend1d_array


def trend2d_array(phase, weight, variables, device, degree=1):
    """
    Fit 2D polynomial trend using PyTorch least squares (pure GPU implementation).

    Two modes:
    - Complex input: unit-circle fitting (normalize to unit magnitude, fit complex
      polynomial, normalize result), returns complex64 unit-magnitude trend
    - Real input: standard real polynomial fit, returns float32 trend

    Parameters
    ----------
    phase : np.ndarray
        2D or 3D array (pair, y, x) or (y, x). Real or complex.
    weight : np.ndarray or None
        Weight array (real), same spatial shape as phase
    variables : list of np.ndarray
        List of 2D variable arrays (y, x) to use as regressors
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree for each variable (1=linear, 2=quadratic, etc.)

    Returns
    -------
    np.ndarray
        Trend surface, same shape as phase.
        Complex input returns complex64 unit-magnitude trend.
        Real input returns float32 trend.
    """
    import torch
    from itertools import combinations_with_replacement

    # Force torch.linalg lazy initialization to avoid CUDA multi-threading bug
    # See: https://github.com/pytorch/pytorch/issues/90613
    global _linalg_initialized
    if device.type == 'cuda' and not _linalg_initialized:
        with _linalg_init_lock:
            if not _linalg_initialized:
                with torch.cuda.device(device):
                    _ = torch.linalg.solve(torch.eye(2, device=device), torch.ones(2, device=device))
                _linalg_initialized = True

    # Handle 2D vs 3D input
    squeeze = phase.ndim == 2
    if squeeze:
        phase = phase[np.newaxis, ...]
        if weight is not None:
            weight = weight[np.newaxis, ...]

    n_pairs, ny, nx = phase.shape
    n_pixels = ny * nx

    # Detect complex input
    is_complex = np.iscomplexobj(phase)

    # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32

    # Flatten variables and track fitting mask (NaN = excluded from fit, not from output)
    fit_mask = np.ones(n_pixels, dtype=bool)
    vars_np = []
    for var in variables:
        v_flat = var.ravel()
        fit_mask &= np.isfinite(v_flat)
        vars_np.append(v_flat)

    # For polynomial features: use nan_to_num so all pixels get valid features
    # NaN pixels get 0.0 features — but we use standardized features, so the
    # prediction at these pixels uses the mean value (bias term dominates).
    # This is acceptable for a smooth polynomial trend.
    vars_t = [torch.tensor(np.nan_to_num(v, nan=0.0), dtype=dtype, device=device) for v in vars_np]
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

    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    n_feat = A_all.shape[1]

    # Common standardization from transform-valid pixels (fit_mask)
    fit_indices = torch.tensor(np.where(fit_mask)[0], device=device)
    A_fit = A_all[fit_indices]
    feature_mean = A_fit[:, :-1].mean(dim=0, keepdim=True)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='std\\(\\): degrees of freedom')
        feature_std = A_fit[:, :-1].std(dim=0, keepdim=True) + 1e-10

    # Standardize design matrix once — (n_pixels, n_feat), stays on device
    A_std = torch.cat([
        (A_all[:, :-1] - feature_mean) / feature_std,
        A_all[:, -1:]
    ], dim=1)

    del vars_t, features, X_poly, ones, A_all, A_fit, fit_indices

    # Flatten phase/weight to (n_pairs, n_pixels) — stay on CPU
    phase_flat = phase.reshape(n_pairs, n_pixels)
    weight_flat = weight.reshape(n_pairs, n_pixels) if weight is not None else None

    # Compute batch size from dask chunk memory budget
    # Dominant memory per pixel: WA_batch (n_pairs × n_feat) + W,b (n_pairs × 2)
    from .utils_dask import get_dask_chunk_size_mb
    dtype_size = 8 if dtype == torch.float64 else 4
    mem_per_pixel = n_pairs * (n_feat + 2) * dtype_size
    batch_size = max(1024, (get_dask_chunk_size_mb() * 1024 * 1024) // mem_per_pixel)

    # Accumulate normal equations incrementally over spatial batches
    AtWA = torch.zeros((n_pairs, n_feat, n_feat), dtype=dtype, device=device)
    acc_dtype = cdtype if is_complex else dtype
    AtWb = torch.zeros((n_pairs, n_feat), dtype=acc_dtype, device=device)
    n_valid = np.zeros(n_pairs, dtype=np.int64)

    n_batches = (n_pixels + batch_size - 1) // batch_size
    for i in range(n_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, n_pixels)
        p_batch = phase_flat[:, s:e]
        fm_batch = fit_mask[s:e]

        # Per-batch valid mask (stays on CPU)
        if is_complex:
            valid_b = np.isfinite(p_batch) & (p_batch != 0)
        else:
            valid_b = np.isfinite(p_batch)
        valid_b = valid_b & fm_batch[None, :]
        if weight_flat is not None:
            w_batch = weight_flat[:, s:e]
            valid_b = valid_b & np.isfinite(w_batch)
        n_valid += valid_b.sum(axis=1)

        # W_batch: (n_pairs, batch) on device
        if weight_flat is not None:
            sqrt_w = np.sqrt(np.nan_to_num(w_batch, nan=0.0)).astype(np.float32)
            W_b = torch.tensor(np.where(valid_b, sqrt_w, 0.0), dtype=dtype, device=device)
        else:
            W_b = torch.tensor(valid_b.astype(np.float32), dtype=dtype, device=device)

        # b_batch: (n_pairs, batch) on device
        if is_complex:
            p_abs = np.abs(p_batch)
            with np.errstate(invalid='ignore', divide='ignore'):
                p_unit = np.where(p_abs > 0, p_batch / p_abs, 0 + 0j)
            b_b = torch.tensor(np.nan_to_num(p_unit, nan=0.0), dtype=cdtype, device=device)
        else:
            b_b = torch.tensor(np.nan_to_num(p_batch, nan=0.0), dtype=dtype, device=device)

        # WA_batch: (n_pairs, batch, n_feat) — bounded memory
        A_b = A_std[s:e]  # view on device
        WA_b = W_b[:, :, None] * A_b[None, :, :]

        # Accumulate AtWA (real) and AtWb (real or complex)
        AtWA += WA_b.transpose(1, 2) @ WA_b
        Wb_b = (W_b * b_b)[:, :, None]
        if is_complex:
            AtWb += (WA_b.transpose(1, 2).to(cdtype) @ Wb_b.to(cdtype)).squeeze(2)
        else:
            AtWb += (WA_b.transpose(1, 2) @ Wb_b).squeeze(2)

        del WA_b, W_b, b_b, Wb_b

    # Regularize and solve
    reg = 1e-10 * torch.eye(n_feat, dtype=dtype, device=device)
    if is_complex:
        AtWA = AtWA.to(cdtype)
    AtWA = AtWA + reg
    coeffs = torch.linalg.solve(AtWA.cpu(), AtWb.cpu()).to(device)

    # Skip pairs with too few valid pixels
    skip_mask = n_valid < 10

    # Predict in batches — bounded memory
    out_dtype = np.complex64 if is_complex else np.float32
    trends_np = np.empty((n_pairs, n_pixels), dtype=out_dtype)
    for i in range(n_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, n_pixels)
        A_b = A_std[s:e]
        if is_complex:
            t_b = (A_b.to(cdtype) @ coeffs.T).T.cpu().numpy()
            with np.errstate(invalid='ignore'):
                t_abs = np.abs(t_b)
                t_b = np.where(t_abs > 0, t_b / t_abs, 0)
            t_b[~np.isfinite(t_b)] = 0
        else:
            t_b = (A_b @ coeffs.T).T.cpu().numpy()
        trends_np[:, s:e] = t_b

    # Set skipped pairs to NaN
    if skip_mask.any():
        trends_np[skip_mask] = np.nan

    trends = trends_np.reshape(n_pairs, ny, nx)

    if squeeze:
        trends = trends[0]

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    if is_complex:
        return trends.astype(np.complex64)
    return trends.astype(np.float32)


def polyfit1d_pytorch(data, time_values, weight, sign, device, degree=0):
    """
    Fit 1D polynomial along first dimension at each pixel using PyTorch.

    Two modes:
    - Complex input: extract angles, apply IRLS with 2π jump correction
      (same principle as irls_solve_1d in unwrap1d). Normalize time by
      t_absmax so t=0 maps to 0 and the intercept = model at the date.
      Returns exp(i * intercept) as unit-magnitude complex model.
    - Real input: standard polynomial fit with sign applied multiplicatively

    Parameters
    ----------
    data : np.ndarray
        3D array (n_samples, ny, nx). Real or complex.
    time_values : np.ndarray
        1D array of time values for fitting (length n_samples)
    weight : np.ndarray or None
        Weight array (real), same shape as data
    sign : np.ndarray
        Sign array (n_samples,): +1 for ref dates, -1 for rep dates.
        Real: multiplied into data. Complex: -1 negates angle.
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (0=mean, 1=linear)

    Returns
    -------
    np.ndarray
        Model at date (ny, nx). complex64 for complex input, float32 for real.
    """
    import torch
    from .utils_dask import get_dask_chunk_size_mb

    n_samples, ny, nx = data.shape
    n_pixels = ny * nx
    is_complex = np.iscomplexobj(data)
    out_dtype = np.complex64 if is_complex else np.float32
    nan_val = np.nan + 0j if is_complex else np.nan

    # Use float64 on CPU, float32 on GPU
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32

    if is_complex:
        # Extract angles, negate for rep dates (equivalent to conj)
        data_abs = np.abs(data)
        with np.errstate(invalid='ignore', divide='ignore'):
            angles_np = np.where(data_abs > 0, np.angle(data), np.nan)
        del data_abs
        for s in range(n_samples):
            if sign[s] < 0:
                angles_np[s] = -angles_np[s]

        # Flatten spatial: (n_samples, n_pixels)
        angles_flat = angles_np.reshape(n_samples, n_pixels)
        del angles_np

        # Weight array flattened (correlation weights, not yet applied)
        if weight is not None:
            weight_flat = weight.reshape(n_samples, n_pixels)
        else:
            weight_flat = None

        # Valid mask: at least degree+2 finite samples per pixel
        min_valid = degree + 2
        finite_count = np.sum(np.isfinite(angles_flat), axis=0)
        valid_mask = finite_count >= min_valid
        has_partial_nan = np.any(finite_count[valid_mask] < n_samples) if valid_mask.any() else False
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            return np.full((ny, nx), nan_val, dtype=out_dtype)

        # Normalize time: t / absmax so t=0 maps to 0, intercept = value at date
        t_absmax = np.max(np.abs(time_values))
        if t_absmax > 0:
            t_norm = time_values / t_absmax
        else:
            t_norm = np.zeros_like(time_values)

        # Build design matrix: [1, t, t^2, ...]
        X = np.column_stack([t_norm**d for d in range(degree + 1)])
        X_t = torch.tensor(X, dtype=dtype, device=device)
        reg = 1e-10 * torch.eye(degree + 1, dtype=dtype, device=device)

        # Compute pixel batch size from dask block budget
        elem_bytes = n_samples * 8  # float64 angles
        budget_bytes = get_dask_chunk_size_mb() * 1024 * 1024
        batch_pixels = max(1024, budget_bytes // elem_bytes)

        # Initialize output in numpy
        result_np = np.full((n_pixels,), nan_val, dtype=out_dtype)

        # Process valid pixels in batches with IRLS + jump correction
        for b_start in range(0, n_valid, batch_pixels):
            b_end = min(b_start + batch_pixels, n_valid)
            idx = valid_indices[b_start:b_end]

            angles_raw = torch.tensor(angles_flat[:, idx], dtype=dtype, device=device)
            nan_mask = ~torch.isfinite(angles_raw)
            angles_safe = torch.where(nan_mask, torch.zeros(1, dtype=dtype, device=device), angles_raw)
            del angles_raw

            # Correlation weight + NaN masking
            if weight_flat is not None:
                sqrt_w_corr = torch.sqrt(torch.tensor(weight_flat[:, idx], dtype=dtype, device=device))
            else:
                sqrt_w_corr = torch.ones_like(angles_safe)

            if has_partial_nan:
                sqrt_w_corr = sqrt_w_corr * (~nan_mask).to(dtype)

            # IRLS iteration with jump correction
            epsilon = 0.1
            jumps = torch.zeros_like(angles_safe)
            W_irls = torch.ones_like(angles_safe)

            for iteration in range(10):
                data_b = angles_safe + jumps * (2 * np.pi)
                sqrt_w = sqrt_w_corr * torch.sqrt(W_irls)

                # Weighted polynomial fit: per-pixel solve
                y_w = (data_b * sqrt_w).T  # (batch, samp)
                X_wi = X_t[None, :, :] * sqrt_w.T[:, :, None]
                XtX_wi = X_wi.transpose(1, 2) @ X_wi + reg
                Xty = X_wi.transpose(1, 2) @ y_w.unsqueeze(-1)
                coeffs = torch.linalg.solve(XtX_wi, Xty)  # (batch, deg+1, 1)

                fit_raw = X_t @ coeffs.squeeze(-1).T  # (n_samples, batch)

                # Update integer 2π jumps
                new_jumps = torch.round((fit_raw - angles_safe) / (2 * np.pi))
                new_jumps[nan_mask] = 0

                # Wrapped residuals → IRLS weights
                residuals = data_b - fit_raw
                residuals = torch.atan2(torch.sin(residuals), torch.cos(residuals))
                W_irls_new = 1.0 / (torch.abs(residuals) + epsilon)
                W_irls_new[nan_mask] = 0

                # Convergence check
                weight_change = torch.abs(W_irls_new - W_irls).mean()
                jumps_stable = (new_jumps == jumps).all()
                if jumps_stable and weight_change < 1e-3:
                    break

                jumps = new_jumps
                W_irls = W_irls_new

            # Intercept c0 = model at date (t=0 maps to t_norm=0)
            c0 = coeffs.squeeze(-1)[:, 0]  # (batch,)
            result_np[idx] = np.exp(1j * c0.cpu().numpy()).astype(out_dtype)

        result_np = result_np.reshape(ny, nx)

    else:
        # Real data path — don't pre-weight; IRLS loop handles all weighting
        work_data = data * sign[:, None, None]

        # Normalize time: t / absmax so t=0 maps to 0, intercept = value at date
        t_absmax = np.max(np.abs(time_values))
        if t_absmax > 0:
            t_norm = time_values / t_absmax
        else:
            t_norm = np.zeros_like(time_values)

        # Build design matrix: [1, t, t^2, ...]
        X = np.column_stack([t_norm**d for d in range(degree + 1)])
        X_t = torch.tensor(X, dtype=dtype, device=device)

        # Flatten spatial dimensions
        data_flat = work_data.reshape(n_samples, n_pixels)
        del work_data

        valid_mask = np.all(np.isfinite(data_flat), axis=0)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            return np.full((ny, nx), nan_val, dtype=out_dtype)

        reg = 1e-10 * torch.eye(degree + 1, dtype=dtype, device=device)

        # Compute pixel batch size from dask block budget
        elem_bytes = n_samples * 8
        budget_bytes = get_dask_chunk_size_mb() * 1024 * 1024
        batch_pixels = max(1024, budget_bytes // elem_bytes)

        # Weight array flattened
        if weight is not None:
            weight_flat = np.sqrt(weight).reshape(n_samples, n_pixels)
        else:
            weight_flat = None

        # Initialize output in numpy
        result_np = np.full((n_pixels,), nan_val, dtype=out_dtype)

        # IRLS parameters
        epsilon = 0.1
        max_irls = 5

        # Process valid pixels in batches with IRLS L1
        for b_start in range(0, n_valid, batch_pixels):
            b_end = min(b_start + batch_pixels, n_valid)
            idx = valid_indices[b_start:b_end]
            data_b = torch.tensor(data_flat[:, idx], dtype=dtype, device=device)

            # Correlation weights
            if weight_flat is not None:
                sqrt_w_corr = torch.tensor(weight_flat[:, idx], dtype=dtype, device=device)
            else:
                sqrt_w_corr = torch.ones_like(data_b)

            W_irls = torch.ones_like(data_b)

            for iteration in range(max_irls):
                sqrt_w = sqrt_w_corr * torch.sqrt(W_irls)
                y_w = (data_b * sqrt_w).T  # (batch, samp)
                X_wi = X_t[None, :, :] * sqrt_w.T[:, :, None]
                XtX_wi = X_wi.transpose(1, 2) @ X_wi + reg
                Xty = X_wi.transpose(1, 2) @ y_w.unsqueeze(-1)
                coeffs = torch.linalg.solve(XtX_wi, Xty)  # (batch, deg+1, 1)

                fit = (X_t @ coeffs.squeeze(-1).T)  # (n_samples, batch)
                residuals = data_b - fit
                W_irls_new = 1.0 / (torch.abs(residuals) + epsilon)
                weight_change = torch.abs(W_irls_new - W_irls).mean()
                if weight_change < 1e-3:
                    break
                W_irls = W_irls_new

            # Intercept c0 = model at date (t=0 maps to t_norm=0)
            c0 = coeffs.squeeze(-1)[:, 0]
            result_np[idx] = c0.cpu().numpy().astype(out_dtype)

        result_np = result_np.reshape(ny, nx)

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return result_np


def trend1d_pairs_array(data_chunk, weight_chunk, ref_values, rep_values,
                         device, degree):
    """
    Process a spatial chunk for trend1d_pairs.

    For each unique date, gathers all pairs sharing that date, fits a
    polynomial to phase vs temporal baseline using all pairs, and stores
    the model at zero temporal baseline (intercept). Pair trends are
    reconstructed as model[ref] op model[rep].

    Two modes:
    - Complex input: unit-circle fitting per date, reconstruct pair trend as
      model[ref] * conj(model[rep]) (complex64 output)
    - Real input: standard polynomial fit per date, reconstruct as
      model[ref] - model[rep] (float32 output)

    Parameters
    ----------
    data_chunk : np.ndarray
        3D array (n_pairs, chunk_y, chunk_x). Real or complex.
    weight_chunk : np.ndarray or None
        Weight array (real), same shape as data_chunk
    ref_values : np.ndarray
        1D array of ref dates as int64 (nanoseconds since epoch)
    rep_values : np.ndarray
        1D array of rep dates as int64 (nanoseconds since epoch)
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (0=mean, 1=linear)

    Returns
    -------
    np.ndarray
        Trend array (n_pairs, chunk_y, chunk_x). complex64 or float32.
    """
    import torch

    # Accept chunk lists: extract only needed pairs without full concatenation.
    # _select(indices) extracts specific dim-0 slices from chunk list or 3D array.
    if isinstance(data_chunk, list):
        chunks = [np.asarray(c) for c in data_chunk]
        ny, nx = chunks[0].shape[1], chunks[0].shape[2]
        n_pairs = sum(c.shape[0] for c in chunks)
        is_complex = np.iscomplexobj(chunks[0])
        # Build cumulative size boundaries for index mapping
        cum = np.zeros(len(chunks) + 1, dtype=np.intp)
        for i, c in enumerate(chunks):
            cum[i + 1] = cum[i] + c.shape[0]

        def _select(indices):
            out = np.empty((len(indices), ny, nx), dtype=chunks[0].dtype)
            for ci, c in enumerate(chunks):
                mask = (indices >= cum[ci]) & (indices < cum[ci + 1])
                if mask.any():
                    out[mask] = c[indices[mask] - cum[ci]]
            return out
    else:
        n_pairs, ny, nx = data_chunk.shape
        is_complex = np.iscomplexobj(data_chunk)
        _select = lambda indices: data_chunk[indices]

    if isinstance(weight_chunk, list):
        w_chunks = [np.asarray(c) for c in weight_chunk]
        w_cum = np.zeros(len(w_chunks) + 1, dtype=np.intp)
        for i, c in enumerate(w_chunks):
            w_cum[i + 1] = w_cum[i] + c.shape[0]

        def _select_w(indices):
            out = np.empty((len(indices), ny, nx), dtype=w_chunks[0].dtype)
            for ci, c in enumerate(w_chunks):
                mask = (indices >= w_cum[ci]) & (indices < w_cum[ci + 1])
                if mask.any():
                    out[mask] = c[indices[mask] - w_cum[ci]]
            return out
    elif weight_chunk is not None:
        _select_w = lambda indices: weight_chunk[indices]
    else:
        _select_w = None

    # Convert int64 nanoseconds to days (relative to first date)
    ns_per_day = 86400 * 1e9
    ref_days = ref_values / ns_per_day
    rep_days = rep_values / ns_per_day

    # Get unique dates
    all_days = np.concatenate([ref_days, rep_days])
    unique_days = np.unique(all_days)

    # Build per-date models
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

        if len(pair_indices) == 0:
            continue

        # Extract only selected pairs from chunks (no full concatenation)
        selected_data = _select(pair_indices)
        selected_weight = _select_w(pair_indices) if _select_w is not None else None

        # Fit polynomial using PyTorch
        models[date_day] = polyfit1d_pytorch(
            selected_data, time_values, selected_weight, signs,
            device, degree=degree
        )

    # Reconstruct pair-wise trends
    out_dtype = np.complex64 if is_complex else np.float32
    nan_val = np.nan + 0j if is_complex else np.nan
    trend_data = np.full((n_pairs, ny, nx), nan_val, dtype=out_dtype)

    for i in range(n_pairs):
        ref_day = ref_days[i]
        rep_day = rep_days[i]

        # Find matching keys (use isclose for float comparison)
        ref_key = None
        rep_key = None
        for key in models.keys():
            if np.isclose(key, ref_day):
                ref_key = key
            if np.isclose(key, rep_day):
                rep_key = key

        if ref_key is None or rep_key is None:
            continue

        if is_complex:
            trend_data[i] = models[ref_key] * np.conj(models[rep_key])
        else:
            trend_data[i] = models[ref_key] - models[rep_key]

    return trend_data


# ============================================================================
# Chunked trend2d pipeline kernels
# ============================================================================
# These functions implement a 4-phase pipeline that avoids the full-spatial
# rechunk required by trend2d_array(). Normal equations (AtWA, AtWb) are
# accumulated per spatial chunk then summed via tree reduction.

def _build_poly_features(var_flat_list, n_pixels, degree):
    """Build polynomial feature matrix from flattened variable arrays.

    Features are ordered: degree 1 first, then degree 2, etc.
    Same order as sklearn PolynomialFeatures(include_bias=False).

    Parameters
    ----------
    var_flat_list : list of ndarray (n_pixels,)
        Flattened variable arrays (NaN-free, use nan_to_num before calling).
    n_pixels : int
        Number of pixels.
    degree : int
        Polynomial degree.

    Returns
    -------
    ndarray (n_pixels, n_poly_features)
        Polynomial feature matrix in float64.
    """
    from itertools import combinations_with_replacement

    n_vars = len(var_flat_list)
    features = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            term = np.ones(n_pixels, dtype=np.float64)
            for idx in combo:
                term = term * var_flat_list[idx]
            features.append(term)

    if len(features) == 0:
        return np.empty((n_pixels, 0), dtype=np.float64)
    return np.column_stack(features)


def _compute_feature_stats(var_dask_list, degree):
    """Phase 0: Compute global feature_mean and feature_std for standardization.

    Computes statistics from transform (pair-independent) using dask tree
    reductions.  Single .compute() call for efficiency.

    Parameters
    ----------
    var_dask_list : list of dask.array.Array
        Transform variables (2D: y, x).
    degree : int
        Polynomial degree.

    Returns
    -------
    feature_mean : ndarray (n_poly_features,)
    feature_std : ndarray (n_poly_features,)
    """
    import dask
    import dask.array as da
    from itertools import combinations_with_replacement

    n_vars = len(var_dask_list)

    # Valid mask: all variables finite
    valid_mask = da.ones(var_dask_list[0].shape, dtype=bool,
                         chunks=var_dask_list[0].chunks)
    for v in var_dask_list:
        valid_mask = valid_mask & da.isfinite(v)

    # Build polynomial features lazily and schedule reductions
    to_compute = []
    n_features = 0
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            term = da.ones_like(var_dask_list[0], dtype=np.float64)
            for idx in combo:
                v64 = var_dask_list[idx].astype(np.float64)
                term = term * da.where(da.isfinite(v64), v64, 0.0)
            masked = da.where(valid_mask, term, np.nan)
            to_compute.append(da.nanmean(masked))
            to_compute.append(da.nanstd(masked))
            n_features += 1

    if n_features == 0:
        return np.empty(0, dtype=np.float64), np.ones(0, dtype=np.float64)

    results = dask.compute(*to_compute)
    feature_mean = np.array([float(results[2 * i]) for i in range(n_features)],
                            dtype=np.float64)
    feature_std = np.array([float(results[2 * i + 1]) for i in range(n_features)],
                           dtype=np.float64) + 1e-10

    return feature_mean, feature_std


def _accumulate_chunk(phase_chunk, weight_chunk, var_chunks,
                      feature_mean, feature_std, degree, is_complex):
    """Phase 1: Accumulate normal equations for one spatial tile.

    Builds local A_std from transform tile using global feature_mean /
    feature_std, then accumulates AtWA and AtWb per pair with internal
    pixel batching for bounded memory.

    Parameters
    ----------
    phase_chunk : ndarray (n_pairs, cy, cx)
        Phase data.  Complex or real.
    weight_chunk : ndarray or None
        Weight data, same shape as phase_chunk.
    var_chunks : tuple/list of ndarray (cy, cx)
        Transform variable arrays.
    feature_mean : ndarray (n_poly_features,)
    feature_std : ndarray (n_poly_features,)
    degree : int
    is_complex : bool

    Returns
    -------
    ndarray (n_pairs, 1, 1, n_accum)
        Packed: [AtWA.ravel() | AtWb(.real,.imag) | n_valid] in float64.
    """
    n_pairs = phase_chunk.shape[0]
    cy, cx = phase_chunk.shape[1], phase_chunk.shape[2]
    n_pixels = cy * cx

    # Flatten variables and build fit mask
    var_flat_list = []
    fit_mask = np.ones(n_pixels, dtype=bool)
    for v in var_chunks:
        v_flat = v.ravel().astype(np.float64)
        fit_mask &= np.isfinite(v_flat)
        var_flat_list.append(np.nan_to_num(v_flat, nan=0.0))

    n_poly = len(feature_mean)
    n_feat = n_poly + 1  # +1 for bias
    n_feat_b = 2 * n_feat if is_complex else n_feat
    n_accum = n_feat * n_feat + n_feat_b + 1

    phase_flat = phase_chunk.reshape(n_pairs, n_pixels)
    weight_flat = (weight_chunk.reshape(n_pairs, n_pixels)
                   if weight_chunk is not None else None)

    # Batch size: keep A_std_batch + WA under half dask chunk budget
    from .utils_dask import get_dask_chunk_size_mb
    _budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
    batch_size = max(1024, _budget // max(1, 2 * n_feat * 8))
    n_batches = (n_pixels + batch_size - 1) // batch_size

    result = np.zeros((n_pairs, 1, 1, n_accum), dtype=np.float64)

    for p in range(n_pairs):
        AtWA = np.zeros((n_feat, n_feat), dtype=np.float64)
        AtWb = np.zeros(n_feat,
                        dtype=np.complex128 if is_complex else np.float64)
        n_valid_total = 0

        for bi in range(n_batches):
            s = bi * batch_size
            e = min((bi + 1) * batch_size, n_pixels)
            batch_len = e - s

            p_batch = phase_flat[p, s:e]
            fm_batch = fit_mask[s:e]

            if is_complex:
                valid = np.isfinite(p_batch) & (p_batch != 0) & fm_batch
            else:
                valid = np.isfinite(p_batch) & fm_batch
            if weight_flat is not None:
                valid &= np.isfinite(weight_flat[p, s:e])

            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            n_valid_total += n_valid

            # Build A_std for this batch (bounded memory)
            var_batch_list = [v[s:e] for v in var_flat_list]
            X_poly_b = _build_poly_features(var_batch_list, batch_len, degree)
            A_std_b = np.concatenate([
                (X_poly_b - feature_mean) / feature_std,
                np.ones((batch_len, 1), dtype=np.float64)
            ], axis=1)

            A_v = A_std_b[valid]  # (n_valid, n_feat)

            if weight_flat is not None:
                sqrt_w = np.sqrt(
                    np.clip(weight_flat[p, s:e][valid], 0, None))
                WA = A_v * sqrt_w[:, None]
            else:
                WA = A_v

            if is_complex:
                p_vals = p_batch[valid]
                p_abs = np.abs(p_vals)
                with np.errstate(invalid='ignore', divide='ignore'):
                    p_unit = np.where(p_abs > 0, p_vals / p_abs, 0 + 0j)
                b_vals = np.nan_to_num(p_unit, nan=0.0)
            else:
                b_vals = np.nan_to_num(p_batch[valid],
                                       nan=0.0).astype(np.float64)

            if weight_flat is not None:
                Wb = sqrt_w * b_vals
            else:
                Wb = b_vals

            AtWA += WA.T @ WA
            if is_complex:
                AtWb += (WA.astype(np.complex128).T
                         @ Wb.astype(np.complex128))
            else:
                AtWb += WA.T @ Wb

        # Pack into result
        result[p, 0, 0, :n_feat * n_feat] = AtWA.ravel()
        if is_complex:
            result[p, 0, 0,
                   n_feat * n_feat:n_feat * n_feat + n_feat] = AtWb.real
            result[p, 0, 0,
                   n_feat * n_feat + n_feat:
                   n_feat * n_feat + 2 * n_feat] = AtWb.imag
        else:
            result[p, 0, 0,
                   n_feat * n_feat:n_feat * n_feat + n_feat] = AtWb.real
        result[p, 0, 0, -1] = n_valid_total

    return result


def _solve_chunk(accum_block, n_feat, is_complex):
    """Phase 3: Solve for coefficients from accumulated normal equations.

    Parameters
    ----------
    accum_block : ndarray (n_pairs, n_accum)
        Packed accumulators per pair.
    n_feat : int
        Number of features (including bias).
    is_complex : bool

    Returns
    -------
    ndarray (n_pairs, n_coeff_out)
        Coefficients packed as float64.
        Real: (n_pairs, n_feat).  Complex: (n_pairs, 2*n_feat) with [re | im].
    """
    n_pairs = accum_block.shape[0]
    n_coeff_out = 2 * n_feat if is_complex else n_feat
    result = np.empty((n_pairs, n_coeff_out), dtype=np.float64)

    for p in range(n_pairs):
        accum = accum_block[p]

        AtWA = accum[:n_feat * n_feat].reshape(n_feat, n_feat)
        if is_complex:
            AtWb_re = accum[n_feat * n_feat:n_feat * n_feat + n_feat]
            AtWb_im = accum[n_feat * n_feat + n_feat:
                            n_feat * n_feat + 2 * n_feat]
            AtWb = AtWb_re + 1j * AtWb_im
        else:
            AtWb = accum[n_feat * n_feat:n_feat * n_feat + n_feat]
        n_valid = accum[-1]

        if n_valid < 10:
            result[p] = np.nan
            continue

        AtWA = AtWA + 1e-10 * np.eye(n_feat, dtype=np.float64)

        if is_complex:
            coeffs = np.linalg.solve(AtWA.astype(np.complex128), AtWb)
            result[p] = np.concatenate([coeffs.real, coeffs.imag])
        else:
            result[p] = np.linalg.solve(AtWA, AtWb)

    return result


def _apply_chunk(phase_chunk, coeffs_packed, var_chunks,
                 feature_mean, feature_std, degree, is_complex,
                 detrend_mode):
    """Phase 4: Apply polynomial trend to one spatial tile.

    Parameters
    ----------
    phase_chunk : ndarray (n_pairs, cy, cx)
    coeffs_packed : ndarray (n_pairs, n_coeff_out)
        Packed coefficients (float64).
    var_chunks : tuple/list of ndarray (cy, cx)
        Transform variable arrays.
    feature_mean, feature_std : ndarray (n_poly_features,)
    degree : int
    is_complex : bool
    detrend_mode : bool
        If True, return detrended data.

    Returns
    -------
    ndarray (n_pairs, cy, cx)
    """
    n_pairs = phase_chunk.shape[0]
    cy, cx = phase_chunk.shape[1], phase_chunk.shape[2]
    n_pixels = cy * cx

    n_poly = len(feature_mean)
    n_feat = n_poly + 1

    # Flatten variables
    var_flat_list = [np.nan_to_num(v.ravel().astype(np.float64), nan=0.0)
                     for v in var_chunks]

    # Batch size: keep A_std under half dask chunk budget
    from .utils_dask import get_dask_chunk_size_mb
    _budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
    batch_size = max(1024, _budget // max(1, n_feat * 8))
    n_batches = (n_pixels + batch_size - 1) // batch_size

    out_dtype = np.complex64 if is_complex else np.float32
    result = np.empty((n_pairs, cy, cx), dtype=out_dtype)

    for p in range(n_pairs):
        cp = coeffs_packed[p]

        if np.any(np.isnan(cp)):
            result[p] = np.nan
            continue

        if is_complex:
            coeffs = cp[:n_feat] + 1j * cp[n_feat:]
        else:
            coeffs = cp

        trend_flat = np.empty(
            n_pixels,
            dtype=np.complex128 if is_complex else np.float64)

        for bi in range(n_batches):
            s = bi * batch_size
            e = min((bi + 1) * batch_size, n_pixels)
            batch_len = e - s

            var_batch_list = [v[s:e] for v in var_flat_list]
            X_poly_b = _build_poly_features(var_batch_list, batch_len,
                                            degree)
            A_std_b = np.concatenate([
                (X_poly_b - feature_mean) / feature_std,
                np.ones((batch_len, 1), dtype=np.float64)
            ], axis=1)

            if is_complex:
                trend_flat[s:e] = A_std_b.astype(np.complex128) @ coeffs
            else:
                trend_flat[s:e] = A_std_b @ coeffs

        if is_complex:
            trend_abs = np.abs(trend_flat)
            with np.errstate(invalid='ignore', divide='ignore'):
                trend_flat = np.where(trend_abs > 0,
                                      trend_flat / trend_abs, 0)
            trend_flat[~np.isfinite(trend_flat)] = 0
            trend = trend_flat.reshape(cy, cx).astype(np.complex64)
        else:
            trend = trend_flat.reshape(cy, cx).astype(np.float32)

        if detrend_mode:
            if is_complex:
                result[p] = phase_chunk[p] * np.conj(trend)
            else:
                result[p] = phase_chunk[p] - trend
        else:
            result[p] = trend

    return result
