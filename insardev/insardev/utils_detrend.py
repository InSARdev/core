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


def regression1d_array(data, dim_values, weight, device, degree=1, wrap=False, iterations=1):
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

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return result.astype(np.float32)


def trend2d_array(phase, weight, variables, device, degree=1):
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
        # Move to CPU for solve (faster for small matrices and avoids CUDA lazy init race)
        AtA = (A_valid_scaled.T @ A_valid_scaled).cpu()
        Atb = (A_valid_scaled.T @ b_valid).cpu()
        coeffs = torch.linalg.solve(AtA, Atb).to(device)

        # Compute trend for all pixels using the same scaling
        A_all_scaled = torch.cat([
            (A_all[:, :-1] - feature_mean) / feature_std,
            A_all[:, -1:]
        ], dim=1)
        trend = (A_all_scaled @ coeffs).cpu().numpy()
        trends[i] = trend.reshape(ny, nx)

    if squeeze:
        trends = trends[0]

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return trends.astype(np.float32)


def polyfit1d_pytorch(data, time_values, weight, sign, device, degree=0, wrap=False, iterations=1):
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

        result = (result_sin.cpu().numpy().reshape(ny, nx).astype(np.float32),
                  result_cos.cpu().numpy().reshape(ny, nx).astype(np.float32))

        # Cleanup GPU memory
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

        return result
    else:
        # Standard polynomial fitting
        coeffs = (XtX_inv_Xt @ data_valid).T  # (n_valid, degree+1)

        # Extract coefficient for requested degree
        coeff = coeffs[:, degree]  # (n_valid,)

        # Put back into full array (NaN for invalid pixels)
        result = torch.full((n_pixels,), float('nan'), dtype=dtype, device=device)
        result[valid_indices] = coeff

        result_np = result.cpu().numpy().reshape(ny, nx).astype(np.float32)

        # Cleanup GPU memory
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

        return result_np


def regression1d_pairs_chunk(data_chunk, weight_chunk, ref_values, rep_values,
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
        result = polyfit1d_pytorch(
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
