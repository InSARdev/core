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


def _round_half_away(x):
    """Round to nearest integer, breaking ties away from zero.

    torch.round() uses banker's rounding (round half to even), which
    rounds 0.5 → 0 and 1.5 → 2. For 2π jump correction we need
    round(±0.5) = ±1, otherwise corrections near the wrapping boundary
    (atmospheric phase ≈ ±π) never fire.
    """
    import torch
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


import numba as nb

@nb.njit(cache=True)
def _round_half_away_numba(x):
    """Round to nearest integer, breaking ties away from zero (numba version)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _c = np.zeros((3, 1), dtype=np.complex64)
    _w = np.ones((1, 1), dtype=np.float32)
    _d = np.array([-1.0, 0.0, 1.0])
    _trend1d_numba_kernel(_c, _w, _d, True, True, True, False)
    _trend1d_pairs_numba_kernel(
        _c, 1, 2, 3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, -1.0, 0.5]),
        np.array([1.0, -1.0, 1.0]),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        0, True,
    )
    _v = np.zeros((2, 3), dtype=np.float32)
    _wn = np.array([[0, 2, 0, 3]], dtype=np.int64)
    _trend2d_window_numba_kernel(
        _c[:, :3].reshape(1, 3).view(np.complex64),
        _v, _v, _v, np.ones((1, 1), dtype=np.float32),
        2, 3, False, True, _wn,
    )


@nb.njit(cache=True)
def _trend1d_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex128 or float64
    n_pixels,
    n_dates,
    n_pairs,
    date_pair_flat,    # flattened pair indices
    date_time_flat,    # flattened time values (normalized)
    date_sign_flat,    # flattened signs
    date_offsets,      # (n_dates+1,) start offsets into flat arrays
    pair_ref_didx,     # (n_pairs,) ref date index
    pair_rep_didx,     # (n_pairs,) rep date index
    max_refine,
    use_jumps=True,    # True for wrapped (complex), False for unwrapped (real)
):
    """Per-pixel IRLS fitting of all dates, with iterative refinement.

    Accepts complex or real input directly — extracts angles per-pixel
    inside the loop to avoid full-size intermediate arrays.

    Returns
    -------
    trend : (n_pairs, n_pixels) complex64
        Per-pair trend: exp(1j * (model[ref] - model[rep])) for complex,
        model[ref] - model[rep] for real. NaN where input is invalid or overfitting.
    """
    model_angles = np.zeros((n_dates, n_pixels), dtype=np.float64)
    trend = np.full((n_pairs, n_pixels), np.nan + 0j, dtype=np.complex64)

    # Per-pixel working array for angles (reused across pixels)
    pixel_angles = np.empty(n_pairs, dtype=np.float64)

    for px in range(n_pixels):
        # Extract angles in float64 precision (complex64 float32 causes cstd blow-up)
        if use_jumps:
            for p in range(n_pairs):
                c = data_flat[p, px]
                re = np.float64(c.real)
                im = np.float64(c.imag)
                if re == 0.0 and im == 0.0:
                    pixel_angles[p] = np.nan
                else:
                    pixel_angles[p] = np.arctan2(im, re)
        else:
            for p in range(n_pairs):
                pixel_angles[p] = data_flat[p, px].real

        corrected = np.empty(n_pairs, dtype=np.float64)
        local_models = np.zeros(n_dates, dtype=np.float64)

        cstd_sum = 0.0
        cstd_count = 0

        for iteration in range(1 + max_refine):
            is_filter = (iteration % 3 == 0)

            # Correct data using accumulated models
            if iteration == 0:
                for p in range(n_pairs):
                    corrected[p] = pixel_angles[p]
            else:
                for p in range(n_pairs):
                    corrected[p] = pixel_angles[p] \
                                   - local_models[pair_ref_didx[p]] \
                                   + local_models[pair_rep_didx[p]]

            # Fit each date
            for d in range(n_dates):
                d_start = date_offsets[d]
                d_end = date_offsets[d + 1]
                n_d = d_end - d_start
                if n_d < 4:
                    continue

                # Count valid pairs
                n_valid = 0
                for k in range(n_d):
                    pidx = date_pair_flat[d_start + k]
                    val = corrected[pidx] * date_sign_flat[d_start + k]
                    if np.isfinite(val):
                        n_valid += 1
                if n_valid < 4:
                    continue

                # Prepare per-pair arrays for this date
                phases = np.empty(n_d, dtype=np.float64)
                t_vals = np.empty(n_d, dtype=np.float64)
                valid = np.empty(n_d, dtype=nb.boolean)
                jumps = np.empty(n_d, dtype=np.float64)
                w_irls = np.ones(n_d, dtype=np.float64)

                if use_jumps:
                    # Circular mean for jump initialization (wrapped phase)
                    sin_sum = 0.0
                    cos_sum = 0.0
                    for k in range(n_d):
                        pidx = date_pair_flat[d_start + k]
                        val = corrected[pidx] * date_sign_flat[d_start + k]
                        if np.isfinite(val):
                            sin_sum += np.sin(val)
                            cos_sum += np.cos(val)
                    circ_mean = np.arctan2(sin_sum, cos_sum)

                    for k in range(n_d):
                        pidx = date_pair_flat[d_start + k]
                        val = corrected[pidx] * date_sign_flat[d_start + k]
                        phases[k] = val
                        t_vals[k] = date_time_flat[d_start + k]
                        valid[k] = np.isfinite(val)
                        if valid[k]:
                            jumps[k] = _round_half_away_numba(
                                (circ_mean - val) / (2 * np.pi))
                        else:
                            jumps[k] = 0.0
                            phases[k] = 0.0
                            w_irls[k] = 0.0
                else:
                    # No jump correction for unwrapped phase
                    for k in range(n_d):
                        pidx = date_pair_flat[d_start + k]
                        val = corrected[pidx] * date_sign_flat[d_start + k]
                        phases[k] = val
                        t_vals[k] = date_time_flat[d_start + k]
                        valid[k] = np.isfinite(val)
                        jumps[k] = 0.0
                        if not valid[k]:
                            phases[k] = 0.0
                            w_irls[k] = 0.0

                # IRLS loop: degree-1 analytical 2×2 solve
                # Always uses soft weights during IRLS. Hard rejection
                # applied post-convergence with N-scaled threshold.
                c0 = 0.0
                b = 0.0
                for irls_iter in range(10):
                    sw = 0.0; swt = 0.0; swt2 = 0.0
                    swy = 0.0; swty = 0.0
                    for k in range(n_d):
                        if not valid[k]:
                            continue
                        y = phases[k] + jumps[k] * (2 * np.pi)
                        w = w_irls[k]
                        t = t_vals[k]
                        sw += w
                        swt += w * t
                        swt2 += w * t * t
                        swy += w * y
                        swty += w * t * y

                    det = sw * swt2 - swt * swt + 1e-30
                    a = (swt2 * swy - swt * swty) / det
                    b = (sw * swty - swt * swy) / det
                    c0 = a

                    # Update jumps and weights
                    if use_jumps:
                        converged = True
                        max_dw = 0.0
                        for k in range(n_d):
                            if not valid[k]:
                                continue
                            fit_val = a + b * t_vals[k]
                            new_jump = _round_half_away_numba(
                                (fit_val - phases[k]) / (2 * np.pi))
                            if new_jump != jumps[k]:
                                converged = False
                                jumps[k] = new_jump

                            y = phases[k] + jumps[k] * (2 * np.pi)
                            res = y - fit_val
                            res = res - (2.0 * np.pi) * _round_half_away_numba(res / (2.0 * np.pi))

                            new_w = 1.0 / (abs(res) + 0.1)
                            dw = abs(new_w - w_irls[k])
                            if dw > max_dw:
                                max_dw = dw
                            w_irls[k] = new_w

                        if converged or max_dw < 1e-3:
                            break
                    else:
                        for k in range(n_d):
                            if not valid[k]:
                                continue
                            fit_val = a + b * t_vals[k]
                            res = phases[k] - fit_val
                            w_irls[k] = 1.0 / (abs(res) + 0.1)

                # Compute cstd and accumulate for overfitting check (iteration 0)
                if iteration == 0:
                    cos_s = 0.0
                    sin_s = 0.0
                    nv = 0
                    for k in range(n_d):
                        if not valid[k]:
                            continue
                        y = phases[k] + jumps[k] * (2 * np.pi)
                        raw_res = y - (c0 + b * t_vals[k])
                        cos_s += np.cos(raw_res)
                        sin_s += np.sin(raw_res)
                        nv += 1
                    if nv >= 2:
                        R = np.sqrt((cos_s / nv)**2 + (sin_s / nv)**2)
                        if R > 1 - 1e-10:
                            R = 1 - 1e-10
                        cstd_sum += np.sqrt(-2 * np.log(R))
                        cstd_count += 1

                # Accumulate model
                if iteration == 0:
                    local_models[d] = c0
                else:
                    local_models[d] += c0

        for d in range(n_dates):
            model_angles[d, px] = local_models[d]

        # Overfitting check: avg cstd across dates must be < π/2
        if cstd_count > 0 and (cstd_sum / cstd_count) > (np.pi / 2):
            continue  # leave trend as NaN for this pixel

        # Reconstruct per-pair trend for this pixel
        for p in range(n_pairs):
            if np.isfinite(pixel_angles[p]):
                diff = local_models[pair_ref_didx[p]] - local_models[pair_rep_didx[p]]
                if use_jumps:
                    trend[p, px] = np.complex64(np.exp(1j * diff))
                else:
                    trend[p, px] = np.complex64(diff)

    return trend


@nb.njit(cache=True)
def _trend1d_numba_kernel(
    data_flat,      # (n_samples, n_pixels) complex128 or float64
    w_flat,         # (n_samples, n_pixels) float32 or None-like
    dim_norm,       # (n_samples,) float64  — normalized dim values
    intercept,      # bool — include intercept in output
    slope,          # bool — include slope in output
    use_jumps,      # bool — True for wrapped (complex) phase, False for unwrapped (real)
    has_weight,     # bool — True if w_flat contains real weights, False if unit weights
):
    """Per-pixel IRLS linear fitting for detrend1d.

    Accepts complex or real input directly — extracts angles per-pixel
    inside the loop to avoid full-size intermediate arrays.

    Analytical 2x2 weighted least squares solve per pixel:
    y = a + b*t, 5 accumulators (sw, swt, swt2, swy, swty), Cramer's rule.

    Returns
    -------
    result : (n_samples, n_pixels) complex64 if use_jumps, else float32.
        Complex: unit-magnitude trend exp(1j*fit). Real: fitted values.
    """
    n_samples, n_pixels = data_flat.shape
    result = np.full((n_samples, n_pixels), np.nan + 0j, dtype=np.complex64)

    # Per-pixel working arrays (reused across pixels)
    angles = np.empty(n_samples, dtype=np.float64)
    jumps = np.empty(n_samples, dtype=np.float64)
    w_irls = np.empty(n_samples, dtype=np.float64)
    valid = np.empty(n_samples, dtype=nb.boolean)

    for px in range(n_pixels):
        # Extract angles per-pixel from complex input, or use values directly
        n_valid = 0
        if use_jumps:
            for s in range(n_samples):
                c = data_flat[s, px]
                re = np.float64(c.real)
                im = np.float64(c.imag)
                if re == 0.0 and im == 0.0:
                    angles[s] = np.nan
                    valid[s] = False
                else:
                    a = np.arctan2(im, re)
                    if np.isfinite(a):
                        angles[s] = a
                        valid[s] = True
                        n_valid += 1
                    else:
                        angles[s] = np.nan
                        valid[s] = False
        else:
            for s in range(n_samples):
                val = data_flat[s, px].real  # real input stored as complex with imag=0
                if np.isfinite(val):
                    angles[s] = val
                    valid[s] = True
                    n_valid += 1
                else:
                    angles[s] = np.nan
                    valid[s] = False

        if n_valid < 3:
            continue

        # Initialize jumps and IRLS weights
        if use_jumps:
            sin_sum = 0.0
            cos_sum = 0.0
            for s in range(n_samples):
                if valid[s]:
                    sin_sum += np.sin(angles[s])
                    cos_sum += np.cos(angles[s])
            circ_mean = np.arctan2(sin_sum, cos_sum)

            for s in range(n_samples):
                if valid[s]:
                    jumps[s] = _round_half_away_numba((circ_mean - angles[s]) / (2 * np.pi))
                    w_irls[s] = np.sqrt(w_flat[s, px]) if has_weight else 1.0
                else:
                    jumps[s] = 0.0
                    w_irls[s] = 0.0
        else:
            for s in range(n_samples):
                jumps[s] = 0.0
                w_irls[s] = (np.sqrt(w_flat[s, px]) if has_weight else 1.0) if valid[s] else 0.0

        # IRLS loop with analytical 2x2 solve
        epsilon = 0.1
        c0 = 0.0
        c1 = 0.0

        for irls_iter in range(10):
            sw = 0.0; swt = 0.0; swt2 = 0.0
            swy = 0.0; swty = 0.0
            for s in range(n_samples):
                if not valid[s]:
                    continue
                y = angles[s] + jumps[s] * (2 * np.pi)
                w = w_irls[s]
                t = dim_norm[s]
                sw += w
                swt += w * t
                swt2 += w * t * t
                swy += w * y
                swty += w * t * y
            det = sw * swt2 - swt * swt + 1e-30
            c0 = (swt2 * swy - swt * swty) / det
            c1 = (sw * swty - swt * swy) / det

            if use_jumps:
                converged = True
                max_dw = 0.0
                for s in range(n_samples):
                    if not valid[s]:
                        continue
                    t = dim_norm[s]
                    fit_val = c0 + c1 * t

                    new_jump = _round_half_away_numba(
                        (fit_val - angles[s]) / (2 * np.pi))
                    if new_jump != jumps[s]:
                        converged = False
                        jumps[s] = new_jump

                    y = angles[s] + jumps[s] * (2 * np.pi)
                    res = y - fit_val
                    res = res - (2.0 * np.pi) * _round_half_away_numba(res / (2.0 * np.pi))
                    base_w = np.sqrt(w_flat[s, px]) if has_weight else 1.0
                    new_w = base_w / (abs(res) + epsilon)
                    dw = abs(new_w - w_irls[s])
                    if dw > max_dw:
                        max_dw = dw
                    w_irls[s] = new_w

                if converged or max_dw < 1e-3:
                    break
            else:
                for s in range(n_samples):
                    if not valid[s]:
                        continue
                    t = dim_norm[s]
                    res = angles[s] - (c0 + c1 * t)
                    base_w = np.sqrt(w_flat[s, px]) if has_weight else 1.0
                    w_irls[s] = base_w / (abs(res) + epsilon)

        # Write final values directly as output type
        for s in range(n_samples):
            t = dim_norm[s]
            if not intercept and not slope:
                fit_val = 0.0
            elif not intercept:
                fit_val = c1 * t
            elif not slope:
                fit_val = c0
            else:
                fit_val = c0 + c1 * t
            if use_jumps:
                result[s, px] = np.complex64(np.exp(1j * fit_val))
            else:
                result[s, px] = np.complex64(fit_val)

    return result


def trend1d_array(data, dim_values, weight, intercept=True, slope=True, is_complex=True):
    """
    Fit linear trend along first dimension at each (y, x) pixel.

    Passes data directly to numba kernel — no intermediate float64 arrays.
    Complex: kernel extracts angles per-pixel. Real: passes values through.

    Parameters
    ----------
    data : np.ndarray or list
        3D array (n_samples, y, x) — complex or real. Or list of chunk arrays.
    dim_values : np.ndarray
        1D array of x-values for fitting (length n_samples).
    weight : np.ndarray or list or None
        Weight array (real), same shape as data, or list of chunk arrays.
    intercept : bool
        If True, include intercept (constant term) in output. If False, zero it out.
    slope : bool
        If True, include slope in output. If False, zero it out.
    is_complex : bool
        If True (default), treat as complex wrapped phase. If False, treat as real unwrapped phase.

    Returns
    -------
    np.ndarray
        Fitted values, shape (n_samples, y, x).
        Complex: complex64 unit-magnitude trend. Real: float32 trend.
    """
    if isinstance(data, list):
        data = np.asarray(data[0]) if len(data) == 1 else np.concatenate([np.asarray(c) for c in data], axis=0)
    n_samples, ny, nx = data.shape
    n_pixels = ny * nx

    # Pass data directly to kernel — no intermediate float64 arrays
    if is_complex:
        data[data == 0] = np.nan + 0j
    data_flat = np.ascontiguousarray(data.reshape(n_samples, n_pixels))

    # Weights: pass raw float32 to kernel, sqrt done per-pixel inside
    if isinstance(weight, list):
        weight = np.asarray(weight[0]) if len(weight) == 1 else np.concatenate([np.asarray(c) for c in weight], axis=0)
    has_weight = weight is not None
    if has_weight:
        w_flat = weight.reshape(n_samples, n_pixels).astype(np.float32)
    else:
        w_flat = np.empty((1, 1), dtype=np.float32)  # dummy, not accessed

    # Normalize dim values
    dim_absmax = np.max(np.abs(dim_values))
    if dim_absmax > 0:
        dim_norm = (dim_values / dim_absmax).astype(np.float64)
    else:
        dim_norm = np.zeros(n_samples, dtype=np.float64)

    result = _trend1d_numba_kernel(data_flat, w_flat, dim_norm,
                                    intercept, slope, is_complex, has_weight)
    if is_complex:
        return result.reshape(n_samples, ny, nx)
    else:
        return result.real.astype(np.float32).reshape(n_samples, ny, nx)


# Backward compatibility alias
regression1d_array = trend1d_array



@nb.njit(cache=True)
def _solve4x4(A, b):
    """Solve 4x4 system Ax=b via Gaussian elimination with partial pivoting."""
    M = np.empty((4, 5), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            M[i, j] = A[i, j]
        M[i, 4] = b[i]
    for col in range(4):
        max_val = abs(M[col, col])
        max_row = col
        for row in range(col + 1, 4):
            if abs(M[row, col]) > max_val:
                max_val = abs(M[row, col])
                max_row = row
        if max_val < 1e-15:
            return np.zeros(4, dtype=np.float64)
        if max_row != col:
            for j in range(5):
                M[col, j], M[max_row, j] = M[max_row, j], M[col, j]
        for row in range(col + 1, 4):
            factor = M[row, col] / M[col, col]
            for j in range(col, 5):
                M[row, j] -= factor * M[col, j]
    x = np.zeros(4, dtype=np.float64)
    for i in range(3, -1, -1):
        s = M[i, 4]
        for j in range(i + 1, 4):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


@nb.njit(cache=True)
def _trend2d_window_numba_kernel(phase_2d, var0, var1, var2, weight_2d,
                                  win_y, win_x, has_weight, is_complex,
                                  windows):
    """Numba kernel: windowed polynomial fit degree=1 with 3 variables.

    Three passes per window:
    1. Compute mean/std of variables on valid pixels
    2. Accumulate 4x4 normal equations and solve
    3. Predict on all pixels and accumulate into trend_sum/count
    """
    ny, nx = phase_2d.shape

    trend_sum_re = np.zeros((ny, nx), dtype=np.float64)
    trend_sum_im = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int32)

    for w in range(windows.shape[0]):
        ys = windows[w, 0]
        ye = windows[w, 1]
        xs = windows[w, 2]
        xe = windows[w, 3]

        # Pass 1: mean/std of variables on valid pixels
        sum_v0 = 0.0; sum_v1 = 0.0; sum_v2 = 0.0
        sum2_v0 = 0.0; sum2_v1 = 0.0; sum2_v2 = 0.0
        n_valid = 0
        for i in range(ys, ye):
            for j in range(xs, xe):
                if is_complex:
                    pr = np.float64(phase_2d[i, j].real)
                    pi = np.float64(phase_2d[i, j].imag)
                    if not (np.isfinite(pr) and np.isfinite(pi)):
                        continue
                    if pr == 0.0 and pi == 0.0:
                        continue
                else:
                    pf = np.float64(phase_2d[i, j].real)
                    if not np.isfinite(pf):
                        continue
                v0 = np.float64(var0[i, j])
                v1 = np.float64(var1[i, j])
                v2 = np.float64(var2[i, j])
                if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
                    continue
                sum_v0 += v0; sum_v1 += v1; sum_v2 += v2
                sum2_v0 += v0*v0; sum2_v1 += v1*v1; sum2_v2 += v2*v2
                n_valid += 1

        if n_valid < 4:
            continue

        mu0 = sum_v0 / n_valid
        mu1 = sum_v1 / n_valid
        mu2 = sum_v2 / n_valid
        std0 = max(np.sqrt(max(sum2_v0/n_valid - mu0*mu0, 0.0)), 1e-10)
        std1 = max(np.sqrt(max(sum2_v1/n_valid - mu1*mu1, 0.0)), 1e-10)
        std2 = max(np.sqrt(max(sum2_v2/n_valid - mu2*mu2, 0.0)), 1e-10)

        # Pass 2: accumulate normal equations
        AtA = np.zeros((4, 4), dtype=np.float64)
        Atb_re = np.zeros(4, dtype=np.float64)
        Atb_im = np.zeros(4, dtype=np.float64)

        for i in range(ys, ye):
            for j in range(xs, xe):
                if is_complex:
                    pr = np.float64(phase_2d[i, j].real)
                    pi = np.float64(phase_2d[i, j].imag)
                    if not (np.isfinite(pr) and np.isfinite(pi)):
                        continue
                    if pr == 0.0 and pi == 0.0:
                        continue
                else:
                    pf = np.float64(phase_2d[i, j].real)
                    if not np.isfinite(pf):
                        continue

                v0 = np.float64(var0[i, j])
                v1 = np.float64(var1[i, j])
                v2 = np.float64(var2[i, j])
                if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
                    continue

                a1 = (v0 - mu0) / std0
                a2 = (v1 - mu1) / std1
                a3 = (v2 - mu2) / std2

                w_val = 1.0
                if has_weight:
                    ww = np.float64(weight_2d[i, j])
                    if not np.isfinite(ww) or ww <= 0:
                        continue
                    w_val = np.sqrt(max(ww, 1e-6))

                wa0 = w_val
                wa1 = a1 * w_val
                wa2 = a2 * w_val
                wa3 = a3 * w_val

                AtA[0,0] += wa0*wa0; AtA[0,1] += wa0*wa1; AtA[0,2] += wa0*wa2; AtA[0,3] += wa0*wa3
                AtA[1,1] += wa1*wa1; AtA[1,2] += wa1*wa2; AtA[1,3] += wa1*wa3
                AtA[2,2] += wa2*wa2; AtA[2,3] += wa2*wa3
                AtA[3,3] += wa3*wa3

                if is_complex:
                    mag = np.sqrt(pr*pr + pi*pi)
                    if mag > 0:
                        ur = pr / mag
                        ui = pi / mag
                    else:
                        continue
                    Atb_re[0] += wa0 * ur * w_val; Atb_re[1] += wa1 * ur * w_val
                    Atb_re[2] += wa2 * ur * w_val; Atb_re[3] += wa3 * ur * w_val
                    Atb_im[0] += wa0 * ui * w_val; Atb_im[1] += wa1 * ui * w_val
                    Atb_im[2] += wa2 * ui * w_val; Atb_im[3] += wa3 * ui * w_val
                else:
                    Atb_re[0] += wa0 * pf * w_val; Atb_re[1] += wa1 * pf * w_val
                    Atb_re[2] += wa2 * pf * w_val; Atb_re[3] += wa3 * pf * w_val

        # Fill symmetric lower triangle + regularize
        AtA[1,0] = AtA[0,1]; AtA[2,0] = AtA[0,2]; AtA[3,0] = AtA[0,3]
        AtA[2,1] = AtA[1,2]; AtA[3,1] = AtA[1,3]; AtA[3,2] = AtA[2,3]
        for k in range(4):
            AtA[k,k] += 1e-8

        c_re = _solve4x4(AtA, Atb_re)
        c_im = _solve4x4(AtA, Atb_im)

        # Pass 3: predict on ALL pixels and accumulate
        for i in range(ys, ye):
            for j in range(xs, xe):
                v0 = np.float64(var0[i, j])
                v1 = np.float64(var1[i, j])
                v2 = np.float64(var2[i, j])

                a1 = (v0 - mu0) / std0
                a2 = (v1 - mu1) / std1
                a3 = (v2 - mu2) / std2

                if is_complex:
                    pred_re = c_re[0] + a1*c_re[1] + a2*c_re[2] + a3*c_re[3]
                    pred_im = c_im[0] + a1*c_im[1] + a2*c_im[2] + a3*c_im[3]
                    mag = np.sqrt(pred_re*pred_re + pred_im*pred_im)
                    if mag > 0:
                        trend_sum_re[i,j] += pred_re / mag
                        trend_sum_im[i,j] += pred_im / mag
                else:
                    trend_sum_re[i,j] += c_re[0] + a1*c_re[1] + a2*c_re[2] + a3*c_re[3]
                count[i,j] += 1

    return trend_sum_re, trend_sum_im, count


def trend2d_window_array(phase_2d, variables, weight_2d, device, degree, win_y, win_x):
    """Fit local polynomials on 4 half-overlapping grids, average predictions.

    For degree=1 with 3 variables, uses a fast numba kernel (no array allocation
    per window). Falls back to numpy for degree>=2 or != 3 variables.

    Parameters
    ----------
    phase_2d : (ny, nx) complex64 or float32
    variables : list of (ny, nx) float32 — regressors (azi, rng, ele)
    weight_2d : (ny, nx) float32 or None
    device : ignored (numpy only)
    degree : int — polynomial degree
    win_y, win_x : int — window size in pixels

    Returns
    -------
    trend : (ny, nx) same dtype — averaged trend from 4 grids
    """
    ny, nx = phase_2d.shape
    is_complex = np.iscomplexobj(phase_2d)

    wy = min(win_y, ny)
    wx = min(win_x, nx)
    hy = wy // 2
    hx = wx // 2

    def _window_extents(size, win, half):
        """Return list of (start, end) covering all pixels, extending last window to edge."""
        extents = []
        for s in range(0, size - win + 1, win):
            extents.append((s, s + win))
        if extents and extents[-1][1] < size:
            extents[-1] = (extents[-1][0], size)
        if not extents and size > 0:
            extents.append((0, size))
        for s, e in extents:
            assert e - s >= half, f"Window [{s}:{e}] size {e-s} < half_window {half}"
        return extents

    y_offsets = [0] + ([hy] if ny - hy >= hy else [])
    x_offsets = [0] + ([hx] if nx - hx >= hx else [])

    full_half = (wy * wx) // 2
    windows = []
    for off_y in y_offsets:
        for off_x in x_offsets:
            y_extents = _window_extents(ny - off_y, wy, hy)
            x_extents = _window_extents(nx - off_x, wx, hx)
            for (y0, y1) in y_extents:
                ys, ye = y0 + off_y, y1 + off_y
                for (x0, x1) in x_extents:
                    xs, xe = x0 + off_x, x1 + off_x
                    if (ye - ys) * (xe - xs) >= full_half:
                        windows.append((ys, ye, xs, xe))

    windows_arr = np.array(windows, dtype=np.int64)

    # Fast numba path for degree=1, 3 variables
    if degree == 1 and len(variables) == 3:
        has_weight = weight_2d is not None
        w2d = weight_2d if has_weight else np.empty((1, 1), dtype=np.float32)
        trend_re, trend_im, count = _trend2d_window_numba_kernel(
            phase_2d, variables[0], variables[1], variables[2],
            w2d, win_y, win_x, has_weight, is_complex, windows_arr)

        assert count.max() <= 4, f"Over-covered pixels: max count={count.max()}"
        mask = count > 0
        if is_complex:
            result = np.full((ny, nx), np.nan + 0j, dtype=np.complex64)
            c = count[mask].astype(np.float64)
            result.real[mask] = (trend_re[mask] / c).astype(np.float32)
            result.imag[mask] = (trend_im[mask] / c).astype(np.float32)
        else:
            result = np.full((ny, nx), np.nan, dtype=np.float32)
            result[mask] = (trend_re[mask] / count[mask]).astype(np.float32)
        return result

    # Fallback: numpy path for degree>=2 or != 3 variables
    if is_complex:
        trend_sum = np.zeros((ny, nx), dtype=np.complex64)
    else:
        trend_sum = np.zeros((ny, nx), dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.int32)

    for ys, ye, xs, xe in windows:
        p_win = phase_2d[ys:ye, xs:xe]
        v_wins = [v[ys:ye, xs:xe] for v in variables]
        w_win = weight_2d[ys:ye, xs:xe] if weight_2d is not None else None

        if is_complex:
            valid = np.isfinite(p_win) & (p_win != 0)
        else:
            valid = np.isfinite(p_win)

        n_valid = valid.sum()
        min_pixels = (degree + 1) * (len(variables) + 1)
        if n_valid < min_pixels:
            continue

        idx = np.where(valid.ravel())[0]
        n = len(idx)
        cols = [np.ones(n, dtype=np.float64)]
        standardize = []
        for v in v_wins:
            vf = v.ravel()[idx].astype(np.float64)
            mu, std = vf.mean(), max(vf.std(), 1e-10)
            cols.append((vf - mu) / std)
            standardize.append((mu, std))
        if degree >= 2:
            base = list(cols[1:])
            for d in range(2, degree + 1):
                for c in base:
                    cols.append(c ** d)

        A = np.column_stack(cols)
        nf = A.shape[1]
        if n < nf:
            continue

        w_sqrt = None
        if w_win is not None:
            w_sqrt = np.sqrt(np.clip(w_win.ravel()[idx].astype(np.float64), 1e-6, None))
            Aw = A * w_sqrt[:, np.newaxis]
        else:
            Aw = A

        AtA = Aw.T @ Aw + np.eye(nf) * 1e-8
        win_h, win_w = ye - ys, xe - xs

        all_cols = [np.ones(win_h * win_w, dtype=np.float64)]
        for v, (mu, std) in zip(v_wins, standardize):
            all_cols.append((v.ravel().astype(np.float64) - mu) / std)
        if degree >= 2:
            base = list(all_cols[1:])
            for d in range(2, degree + 1):
                for c in base:
                    all_cols.append(c ** d)
        A_all = np.column_stack(all_cols)

        if is_complex:
            p_flat = p_win.ravel()[idx].astype(np.complex128)
            p_unit = p_flat / np.abs(p_flat)
            b_real = p_unit.real
            b_imag = p_unit.imag
            if w_sqrt is not None:
                b_real = b_real * w_sqrt
                b_imag = b_imag * w_sqrt
            c_real = np.linalg.solve(AtA, Aw.T @ b_real)
            c_imag = np.linalg.solve(AtA, Aw.T @ b_imag)
            pred = (A_all @ c_real + 1j * A_all @ c_imag).reshape(win_h, win_w)
            pred_abs = np.abs(pred)
            pred_abs[~(pred_abs > 0)] = 1
            trend_win = (pred / pred_abs).astype(np.complex64)
        else:
            p_flat = p_win.ravel()[idx].astype(np.float64)
            if w_sqrt is not None:
                p_flat = p_flat * w_sqrt
            coeffs = np.linalg.solve(AtA, Aw.T @ p_flat)
            trend_win = (A_all @ coeffs).reshape(win_h, win_w).astype(np.float32)

        trend_sum[ys:ye, xs:xe] += trend_win
        count[ys:ye, xs:xe] += 1

    assert count.max() <= 4, f"Over-covered pixels: max count={count.max()}"

    mask = count > 0
    if is_complex:
        result = np.full((ny, nx), np.nan + 0j, dtype=np.complex64)
    else:
        result = np.full((ny, nx), np.nan, dtype=np.float32)
    result[mask] = trend_sum[mask] / count[mask]
    return result


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


def polyfit1d_pairs_pytorch(data, time_values, weight, sign, device, degree=0, threshold=None):
    """
    Fit 1D polynomial along first dimension at each pixel using PyTorch.

    Extracts angles from complex input, applies IRLS with 2π jump correction
    (same principle as irls_solve_1d in unwrap1d). Normalizes time by
    t_absmax so t=0 maps to 0 and the intercept = model at the date.
    Returns exp(i * intercept) as unit-magnitude complex model.

    Parameters
    ----------
    data : np.ndarray
        3D complex array (n_samples, ny, nx).
    time_values : np.ndarray
        1D array of time values for fitting (length n_samples)
    weight : np.ndarray or None
        Weight array (real), same shape as data
    sign : np.ndarray
        Sign array (n_samples,): +1 for ref dates, -1 for rep dates.
        -1 negates angle (equivalent to conjugation).
    device : torch.device
        PyTorch device
    degree : int
        Polynomial degree (0=mean, 1=linear)

    Returns
    -------
    np.ndarray or tuple
        Model at date (ny, nx), complex64.
        When threshold is not None, returns (model, kept_mask, cstd_map) where
        kept_mask is (n_samples, ny, nx) bool — True for pairs kept by threshold,
        cstd_map is (ny, nx) float32 — circular std of fit residuals for ALL
        valid pairs (used for averaging across dates in trend1d_pairs_array).
    """
    import torch
    from .utils_dask import get_dask_chunk_size_mb

    n_samples, ny, nx = data.shape
    n_pixels = ny * nx

    if not np.iscomplexobj(data):
        raise TypeError("polyfit1d_pairs_pytorch requires complex input (wrapped phase).")

    out_dtype = np.complex64
    nan_val = np.nan + 0j

    # Minimum pairs for a meaningful fit: degree+3 (4 for degree=1)
    min_pairs = degree + 3
    if n_samples < min_pairs:
        model = np.full((ny, nx), nan_val, dtype=out_dtype)
        if threshold is not None:
            return model, np.zeros((n_samples, ny, nx), dtype=bool), \
                   np.full((ny, nx), np.inf, dtype=np.float32)
        return model

    # Initialize kept mask and cstd map when threshold filtering is active
    kept_np = np.zeros((n_samples, n_pixels), dtype=bool) if threshold is not None else None
    cstd_np = np.full(n_pixels, np.inf, dtype=np.float32) if threshold is not None else None

    # Use float64 on CPU, float32 on GPU
    if device.type == 'cpu':
        dtype = torch.float64
    else:
        dtype = torch.float32

    # Extract angles, negate for rep dates (equivalent to conjugation)
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

    # Valid mask: at least min_pairs finite samples per pixel
    min_valid = min_pairs
    finite_count = np.sum(np.isfinite(angles_flat), axis=0)
    valid_mask = finite_count >= min_valid
    has_partial_nan = np.any(finite_count[valid_mask] < n_samples) if valid_mask.any() else False
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        model = np.full((ny, nx), nan_val, dtype=out_dtype)
        if threshold is not None:
            return model, np.zeros((n_samples, ny, nx), dtype=bool), \
                   np.full((ny, nx), np.inf, dtype=np.float32)
        return model

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
        W_irls = torch.ones_like(angles_safe)

        # Initialize jumps from circular mean (handles wrapping correctly).
        # Without this, when atmospheric phase is near ±pi, wrapped angles
        # split into two clusters and the initial fit averages to ~0,
        # making round((fit-angle)/2pi)=0 so jumps never activate.
        circ_mean = torch.atan2(
            torch.mean(torch.sin(angles_safe), dim=0),
            torch.mean(torch.cos(angles_safe), dim=0),
        )  # (n_pixels_batch,)
        jumps = _round_half_away((circ_mean.unsqueeze(0) - angles_safe) / (2 * np.pi))
        jumps[nan_mask] = 0

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
            new_jumps = _round_half_away((fit_raw - angles_safe) / (2 * np.pi))
            new_jumps[nan_mask] = 0

            # Wrapped residuals → weights
            residuals = data_b - fit_raw
            residuals = torch.atan2(torch.sin(residuals), torch.cos(residuals))
            if threshold is not None:
                # Hard threshold: uniform weight for kept pairs, zero for rejected
                W_irls_new = torch.where(torch.abs(residuals) <= threshold,
                                         torch.ones_like(angles_safe),
                                         torch.zeros_like(angles_safe))
            else:
                # IRLS L1-norm soft reweighting (no threshold)
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

        # Save kept mask and compute per-date cstd for this batch
        if kept_np is not None:
            kept_np[:, idx] = (W_irls > 0).cpu().numpy()
            # Require minimum kept pairs for a meaningful fit
            n_kept = (W_irls > 0).sum(dim=0)  # (batch,)
            min_kept = degree + 3  # 4 for degree=1, 3 for degree=0
            insufficient = (n_kept < min_kept).cpu().numpy()
            if insufficient.any():
                result_np[idx[insufficient]] = nan_val
                kept_np[:, idx[insufficient]] = False
            # Per-date cstd of ALL valid pairs against the fit
            valid_float = (~nan_mask).to(dtype)
            n_valid_b = valid_float.sum(dim=0).clamp(min=1)
            cos_sum = (torch.cos(residuals) * valid_float).sum(dim=0)
            sin_sum = (torch.sin(residuals) * valid_float).sum(dim=0)
            R = torch.sqrt((cos_sum / n_valid_b)**2 + (sin_sum / n_valid_b)**2).clamp(max=1 - 1e-10)
            cstd_np[idx] = torch.sqrt(-2 * torch.log(R)).cpu().numpy().astype(np.float32)

    result_np = result_np.reshape(ny, nx)

    # Cleanup GPU memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    if threshold is not None:
        return result_np, kept_np.reshape(n_samples, ny, nx), \
               cstd_np.reshape(ny, nx)
    return result_np


def trend1d_pairs_array(data_chunk, weight_chunk, ref_values, rep_values,
                         max_refine=3, is_complex=True):
    """
    Estimate per-date atmospheric phase from interferometric network.

    For each unique date, gathers all pairs sharing that date, fits a
    linear model (intercept + slope) to phase vs temporal baseline using
    all pairs, and stores the model at zero temporal baseline (intercept).
    Pair trends are reconstructed as model[ref] - model[rep] (real) or
    model[ref] * conj(model[rep]) (complex).

    Iterative refinement (max_refine > 0): after the initial per-date fit,
    pair-wise corrections from accumulated models are subtracted from the
    original data, and the per-date fit is repeated on the corrected data.

    Uses Numba-compiled per-pixel parallel loop.

    Parameters
    ----------
    data_chunk : np.ndarray or list
        3D array (n_pairs, chunk_y, chunk_x) — complex or real.
    weight_chunk : np.ndarray or None
        Weight array (real), same shape as data_chunk.
    ref_values : np.ndarray
        1D array of ref dates as int64 (nanoseconds since epoch).
    rep_values : np.ndarray
        1D array of rep dates as int64 (nanoseconds since epoch).
    max_refine : int
        Maximum refinement iterations (0 = single-pass). Default 3.
    is_complex : bool
        If True (default), treat as complex wrapped phase. If False, treat as real unwrapped phase.

    Returns
    -------
    np.ndarray
        Trend array (n_pairs, chunk_y, chunk_x), complex64 or float32.
    """
    # Materialize data from chunk list (avoid copy for single chunk)
    if isinstance(data_chunk, list):
        data_np = np.asarray(data_chunk[0]) if len(data_chunk) == 1 else np.concatenate([np.asarray(c) for c in data_chunk], axis=0)
    else:
        data_np = np.asarray(data_chunk)

    n_pairs, ny, nx = data_np.shape

    if is_complex:
        # Convert 0+0j to NaN in-place (skipped dask blocks)
        data_np[data_np == 0] = np.nan + 0j

    out_dtype = np.complex64 if is_complex else np.float32
    n_pixels = ny * nx

    # Convert int64 nanoseconds to days
    ns_per_day = 86400 * 1e9
    ref_days = ref_values / ns_per_day
    rep_days = rep_values / ns_per_day
    unique_days = np.unique(np.concatenate([ref_days, rep_days]))
    n_dates = len(unique_days)

    # Build per-date info flattened for numba (ragged arrays → flat + offsets)
    day_to_idx = {d: i for i, d in enumerate(unique_days)}
    all_pairs, all_times, all_signs = [], [], []
    offsets = [0]
    for date_day in unique_days:
        is_ref = np.isclose(ref_days, date_day)
        is_rep = np.isclose(rep_days, date_day)
        mask = is_ref | is_rep
        pidx = np.where(mask)[0]
        for idx in pidx:
            all_pairs.append(idx)
            if is_rep[idx]:
                all_times.append(ref_days[idx] - date_day)
                all_signs.append(-1.0)
            else:
                all_times.append(rep_days[idx] - date_day)
                all_signs.append(1.0)
        offsets.append(len(all_pairs))

    # Normalize time per date
    all_times_np = np.array(all_times, dtype=np.float64)
    offsets_np = np.array(offsets, dtype=np.int64)
    for d in range(n_dates):
        s, e = offsets_np[d], offsets_np[d + 1]
        if e > s:
            t_absmax = np.max(np.abs(all_times_np[s:e]))
            if t_absmax > 0:
                all_times_np[s:e] /= t_absmax

    # Pair → date index mapping
    pair_ref_didx = np.array([day_to_idx[d] for d in
                              unique_days[np.searchsorted(unique_days, ref_days)]])
    pair_rep_didx = np.array([day_to_idx[d] for d in
                              unique_days[np.searchsorted(unique_days, rep_days)]])

    # Pass data directly to kernel — no intermediate float64 arrays
    if is_complex:
        data_np[data_np == 0] = np.nan + 0j
    data_flat = np.ascontiguousarray(data_np.reshape(n_pairs, n_pixels))
    del data_np

    # Run numba kernel — extracts angles, fits, reconstructs trend, checks overfitting
    trend_data = _trend1d_pairs_numba_kernel(
        data_flat, n_pixels, n_dates, n_pairs,
        np.array(all_pairs, dtype=np.int64),
        all_times_np,
        np.array(all_signs, dtype=np.float64),
        offsets_np,
        pair_ref_didx, pair_rep_didx,
        max_refine,
        is_complex,
    )
    del data_flat

    return trend_data.reshape(n_pairs, ny, nx)


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


# Populate numba file cache on first import so dask workers skip compilation
_warmup_numba_cache()
