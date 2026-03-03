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


def trend1d_array(data, dim_values, weight, device, degree=1):
    """
    Fit 1D polynomial along first dimension at each (y, x) pixel using PyTorch.

    Two modes:
    - Complex input: unit-circle fitting (normalize to unit magnitude, fit complex
      polynomial, normalize result), returns complex64 unit-magnitude trend
    - Real input: standard polynomial fit, returns float32 trend

    Parameters
    ----------
    data : np.ndarray
        3D array (n_samples, y, x) - polynomial fit along first dimension.
        Real or complex.
    dim_values : np.ndarray
        1D array of x-values for fitting (length n_samples)
    weight : np.ndarray or None
        Weight array (real), same shape as data
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (1=linear, 2=quadratic, etc.)

    Returns
    -------
    np.ndarray
        Fitted values, same shape as data.
        Complex input returns complex64 unit-magnitude trend.
        Real input returns float32 trend.
    """
    import torch

    n_samples, ny, nx = data.shape
    n_pixels = ny * nx

    is_complex = np.iscomplexobj(data)

    # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # Normalize dim_values to [-1, 1] for numerical stability
    dim_min = dim_values.min()
    dim_max = dim_values.max()
    dim_range = dim_max - dim_min
    if dim_range > 0:
        dim_norm = 2 * (dim_values - dim_min) / dim_range - 1
    else:
        dim_norm = np.zeros_like(dim_values)

    # Build Vandermonde matrix: [1, x, x^2, ..., x^degree]
    X = np.column_stack([dim_norm**d for d in range(degree + 1)])
    X_t = torch.tensor(X, dtype=dtype, device=device)

    # Flatten spatial dimensions: data becomes (n_samples, n_pixels)
    data_flat = data.reshape(n_samples, n_pixels)
    if weight is not None:
        weight_flat = weight.reshape(n_samples, n_pixels)

    # Prepare data tensor
    if is_complex:
        data_abs = np.abs(data_flat)
        with np.errstate(invalid='ignore', divide='ignore'):
            data_unit = np.where(data_abs > 0, data_flat / data_abs, 0 + 0j)
        data_t = torch.tensor(data_unit, dtype=cdtype, device=device)
    else:
        data_t = torch.tensor(data_flat, dtype=dtype, device=device)

    if weight is not None:
        weight_t = torch.tensor(weight_flat, dtype=dtype, device=device)
    else:
        weight_t = None

    # Find valid pixels (no NaN in any sample)
    if is_complex:
        valid_mask = torch.isfinite(data_t.real).all(dim=0) & torch.isfinite(data_t.imag).all(dim=0)
        valid_mask &= (data_t != 0).any(dim=0)
    else:
        valid_mask = torch.isfinite(data_t).all(dim=0)
    if weight_t is not None:
        valid_mask &= torch.isfinite(weight_t).all(dim=0)

    valid_indices = torch.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        if is_complex:
            return np.full(data.shape, complex('nan'), dtype=np.complex64)
        return np.full_like(data, np.nan, dtype=np.float32)

    # Extract valid pixels: (n_samples, n_valid)
    data_valid = data_t[:, valid_indices]
    if weight_t is not None:
        weight_valid = weight_t[:, valid_indices]
        sqrt_w = torch.sqrt(weight_valid)
    else:
        sqrt_w = None

    # Precompute (X^T X)^-1 X^T for unweighted case
    XtX = X_t.T @ X_t
    XtX_inv = torch.linalg.inv(XtX + 1e-10 * torch.eye(XtX.shape[0], dtype=dtype, device=device))
    XtX_inv_Xt = XtX_inv @ X_t.T  # (degree+1, n_samples)

    if is_complex:
        # ---- Complex unit-circle fitting ----
        if sqrt_w is not None:
            z_weighted = (data_valid * sqrt_w).T  # (n_valid, n_samples)
            X_batch = X_t[None, :, :] * sqrt_w.T[:, :, None]  # (n_valid, n_samples, d+1)
            XtX_batch = X_batch.transpose(1, 2) @ X_batch
            Xtz = X_batch.transpose(1, 2).to(cdtype) @ z_weighted.unsqueeze(-1)
            reg = 1e-10 * torch.eye(XtX_batch.shape[-1], dtype=dtype, device=device)
            coeffs = torch.linalg.solve(
                (XtX_batch + reg).to(cdtype).cpu(), Xtz.cpu()
            ).to(device)
        else:
            data_T = data_valid.T
            coeffs = (XtX_inv_Xt.to(cdtype) @ data_T.T).T.unsqueeze(-1)

        fit = (X_t.to(cdtype) @ coeffs.squeeze(-1).T).T
        fit_abs = fit.abs().clamp(min=1e-10)
        fit = fit / fit_abs

        result_t = torch.full((n_samples, n_pixels), complex(float('nan'), 0), dtype=cdtype, device=device)
        result_t[:, valid_indices] = fit.T
        out = result_t.cpu().numpy().reshape(n_samples, ny, nx).astype(np.complex64)
    else:
        # ---- Standard real polynomial fitting ----
        if sqrt_w is not None:
            y_weighted = (data_valid * sqrt_w).T
            X_batch = X_t[None, :, :] * sqrt_w.T[:, :, None]
            XtX_batch = X_batch.transpose(1, 2) @ X_batch
            Xty = X_batch.transpose(1, 2) @ y_weighted.unsqueeze(-1)
            reg = 1e-10 * torch.eye(XtX_batch.shape[-1], dtype=dtype, device=device)
            coeffs = torch.linalg.solve(XtX_batch + reg, Xty)
        else:
            data_T = data_valid.T
            coeffs = (XtX_inv_Xt @ data_T.T).T.unsqueeze(-1)

        fit = (X_t @ coeffs.squeeze(-1).T).T
        result_t = torch.full((n_samples, n_pixels), float('nan'), dtype=dtype, device=device)
        result_t[:, valid_indices] = fit.T
        out = result_t.cpu().numpy().reshape(n_samples, ny, nx).astype(np.float32)

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
    - Complex input: unit-circle fitting (normalize to unit magnitude, apply sign
      via conj for rep dates, fit complex polynomial, normalize result)
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
        Real: multiplied into data. Complex: -1 triggers conj().
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (0=mean, 1=linear)

    Returns
    -------
    np.ndarray
        Model coefficients (ny, nx). complex64 for complex input, float32 for real.
    """
    import torch

    n_samples, ny, nx = data.shape
    n_pixels = ny * nx
    is_complex = np.iscomplexobj(data)

    # Use float64 on CPU, float32 on GPU
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32

    if is_complex:
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        # Complex unit-circle fitting: normalize to unit magnitude, apply sign via conj
        data_unit = data.copy()
        with np.errstate(invalid='ignore'):
            data_unit /= np.abs(data_unit)
        # Apply sign: conj for rep dates (sign=-1)
        for s in range(n_samples):
            if sign[s] < 0:
                data_unit[s] = np.conj(data_unit[s])
    else:
        # Real: apply sign multiplicatively
        sign_3d = sign[:, None, None]
        data_signed = data * sign_3d

    if weight is not None:
        sqrt_w = np.sqrt(weight)
        if is_complex:
            data_unit = data_unit * sqrt_w
        else:
            data_signed = data_signed * sqrt_w

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
    work_data = data_unit if is_complex else data_signed
    data_flat = work_data.reshape(n_samples, n_pixels)

    if is_complex:
        data_t = torch.tensor(data_flat, dtype=cdtype, device=device)
        # Valid: finite and nonzero
        valid_mask = torch.isfinite(data_t.real).all(dim=0) & torch.isfinite(data_t.imag).all(dim=0)
    else:
        data_t = torch.tensor(data_flat, dtype=dtype, device=device)
        valid_mask = torch.isfinite(data_t).all(dim=0)

    valid_indices = torch.where(valid_mask)[0]
    n_valid = len(valid_indices)

    nan_val = np.nan + 0j if is_complex else np.nan
    out_dtype = np.complex64 if is_complex else np.float32

    if n_valid == 0:
        return np.full((ny, nx), nan_val, dtype=out_dtype)

    data_valid = data_t[:, valid_indices]  # (n_samples, n_valid)

    # Precompute pseudo-inverse for normal equations
    XtX = X_t.T @ X_t
    XtX_inv = torch.linalg.inv(XtX + 1e-10 * torch.eye(XtX.shape[0], dtype=dtype, device=device))
    XtX_inv_Xt = XtX_inv @ X_t.T  # (degree+1, n_samples)

    if is_complex:
        # Complex fit: real design matrix, complex data
        coeffs = (XtX_inv_Xt.to(cdtype) @ data_valid).T  # (n_valid, degree+1)
        coeff = coeffs[:, degree]  # (n_valid,)

        # Normalize to unit magnitude
        coeff_abs = coeff.abs()
        coeff = torch.where(coeff_abs > 0, coeff / coeff_abs, coeff)

        result = torch.full((n_pixels,), float('nan'), dtype=cdtype, device=device)
        result[valid_indices] = coeff
        result_np = result.cpu().numpy().reshape(ny, nx).astype(np.complex64)
    else:
        # Standard polynomial fitting
        coeffs = (XtX_inv_Xt @ data_valid).T  # (n_valid, degree+1)
        coeff = coeffs[:, degree]  # (n_valid,)

        result = torch.full((n_pixels,), float('nan'), dtype=dtype, device=device)
        result[valid_indices] = coeff
        result_np = result.cpu().numpy().reshape(ny, nx).astype(np.float32)

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return result_np


def trend1d_pairs_array(data_chunk, weight_chunk, ref_values, rep_values,
                         device, degree, days_filter, count_filter):
    """
    Process a spatial chunk for trend1d_pairs.

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
    days_filter : int or None
        Maximum time interval filter
    count_filter : int or None
        Maximum pairs per date filter

    Returns
    -------
    np.ndarray
        Trend array (n_pairs, chunk_y, chunk_x). complex64 or float32.
    """
    import torch

    n_pairs, ny, nx = data_chunk.shape
    is_complex = np.iscomplexobj(data_chunk)

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
