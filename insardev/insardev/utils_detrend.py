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
    _trend1d_numba_kernel(_c, _w, _d, True, True, True, False, 128)
    _wf = np.ones((3, 1), dtype=np.float32)
    _threshold_pairs_numba_kernel(_c, _wf, 1, 3, np.pi * 0.5)
    _sd = np.array([0.0, 0.1, -0.1]); _cd = np.array([0.1, 0.0, -0.1])
    _v, _r = _velocity_pairs_numba_kernel(_c, _wf, 1, 3, np.array([1.0, 1.0, 2.0]), _sd, _cd, 3, np.pi, False)
    _trend1d_pairs_numba_kernel(
        _c, _wf, 1, 2, 3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, -1.0, 0.5]),
        np.array([1.0, -1.0, 1.0]),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        np.array([1.0, 1.0, 2.0]),  # pair_dt
        0, True,
    )
    _v = np.zeros((2, 3), dtype=np.float32)
    _g = np.zeros((2, 2), dtype=np.float64)
    _gy = np.array([0.0, 1.0]); _gx = np.array([0.0, 2.0])
    _bilinear_interp_trend(_g, _g, _gy, _gx, 2, 3, True)
    _p = _c[:, :3].reshape(1, 3).view(np.complex64)
    _trend2d_sliding_numba_kernel(
        _p, _v, _v, _v, np.ones((1, 1), dtype=np.float32),
        1, 1, 1, 1, False, True, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    )


@nb.njit(cache=True)
def _threshold_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex64/128
    weight_flat,       # (n_pairs, n_pixels) float32
    n_pixels,
    n_pairs,
    threshold,         # cstd threshold in radians
):
    """Per-pixel weighted cstd check. Returns mask: True = keep, False = reject."""
    mask = np.zeros(n_pixels, dtype=nb.boolean)
    for px in range(n_pixels):
        wcos = 0.0; wsin = 0.0; wsum = 0.0
        for p in range(n_pairs):
            c = data_flat[p, px]
            re = np.float64(c.real)
            im = np.float64(c.imag)
            ang = np.arctan2(im, re)
            if (re == 0.0 and im == 0.0) or not np.isfinite(ang):
                continue
            pw = np.float64(weight_flat[p, px])
            if pw <= 0.0:
                continue
            wcos += pw * np.cos(ang)
            wsin += pw * np.sin(ang)
            wsum += pw
        if wsum < 1e-10:
            continue
        R = np.sqrt((wcos / wsum)**2 + (wsin / wsum)**2)
        if R < 1e-10:
            continue
        R = min(R, 1 - 1e-10)
        if np.sqrt(-2.0 * np.log(R)) < threshold:
            mask[px] = True
    return mask


def threshold_pairs_array(data_chunk, weight_chunk, threshold=np.pi/2):
    """Apply cstd threshold to complex pair data. Returns filtered copy.

    Pixels with weighted cstd >= threshold have all pairs set to 0+0j.
    """
    import numpy as np

    if isinstance(data_chunk, list):
        data_np = np.asarray(data_chunk[0]) if len(data_chunk) == 1 else np.concatenate([np.asarray(c) for c in data_chunk], axis=0)
    else:
        data_np = np.asarray(data_chunk)

    if data_np.ndim == 2:
        n_pairs, nx = data_np.shape
        ny = 1
        data_np = data_np.reshape(n_pairs, ny, nx)
    else:
        n_pairs, ny, nx = data_np.shape
    n_pixels = ny * nx

    data_flat = np.ascontiguousarray(data_np.reshape(n_pairs, n_pixels))

    if isinstance(weight_chunk, list):
        weight_np = np.asarray(weight_chunk[0]) if len(weight_chunk) == 1 else np.concatenate([np.asarray(c) for c in weight_chunk], axis=0)
    elif weight_chunk is not None:
        weight_np = np.asarray(weight_chunk)
    else:
        weight_np = None

    if weight_np is not None:
        weight_flat = np.ascontiguousarray(weight_np.reshape(n_pairs, n_pixels).astype(np.float32))
        weight_flat[~np.isfinite(weight_flat)] = 0.0
        weight_flat[weight_flat < 0] = 0.0
    else:
        weight_flat = np.ones((n_pairs, n_pixels), dtype=np.float32)
    del weight_np

    mask = _threshold_pairs_numba_kernel(data_flat, weight_flat, n_pixels, n_pairs, threshold)

    # NaN rejected pixels
    result = data_np.copy()
    mask_2d = mask.reshape(ny, nx)
    nan_val = np.complex64(np.nan + 0j)
    for iy in range(ny):
        for ix in range(nx):
            if not mask_2d[iy, ix]:
                result[:, iy, ix] = nan_val
    return result


@nb.njit(cache=True)
def _velocity_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex128
    weight_flat,       # (n_pairs, n_pixels) float32
    n_pixels,
    n_pairs,
    pair_dt,           # (n_pairs,) temporal baseline (years)
    sin_diff,          # (n_pairs,) sin(2π*t_ref) - sin(2π*t_rep)
    cos_diff,          # (n_pairs,) cos(2π*t_ref) - cos(2π*t_rep)
    max_refine,        # refinement levels (0=coarse only, 3=0.5mm/yr accuracy)
    vmax,              # max velocity in rad/year (0 = use Nyquist)
    seasonal,          # bool — enable annual seasonal projection
):
    """Global velocity estimation per pixel via multi-level periodogram
    with annual seasonal projection (IRLS).

    For each velocity candidate, fits and removes annual seasonal component
    A*sin_diff + B*cos_diff before scoring. Prevents seasonal signal from
    biasing the velocity estimate.

    Returns
    -------
    velocity : (n_pixels,) float32
        Velocity in rad/year. NaN where insufficient valid pairs.
    rmse : (n_pixels,) float32
        RMSE of residuals in rad after velocity+seasonal removal. NaN where invalid.
    """
    TWO_PI = 2.0 * np.pi
    N_BIN = 16
    velocity = np.full(n_pixels, np.nan, dtype=np.float32)
    rmse = np.full(n_pixels, np.nan, dtype=np.float32)

    # Search range: user-specified vmax or Nyquist
    if vmax > 0:
        gv_range = vmax
    else:
        gv_dt_min = 1e30
        for p in range(n_pairs):
            adt = abs(pair_dt[p])
            if adt > 1e-10 and adt < gv_dt_min:
                gv_dt_min = adt
        if gv_dt_min > 1e20:
            gv_dt_min = 1.0
        gv_range = np.pi / gv_dt_min

    # Precompute seasonal normal equation components (same for all pixels)
    ss = 0.0; sc = 0.0; cc = 0.0
    for p in range(n_pairs):
        ss += sin_diff[p] * sin_diff[p]
        sc += sin_diff[p] * cos_diff[p]
        cc += cos_diff[p] * cos_diff[p]
    seas_det = ss * cc - sc * sc + 1e-30

    # Per-pixel working arrays
    pixel_ang = np.empty(n_pairs, dtype=np.float64)
    pixel_w = np.empty(n_pairs, dtype=np.float64)
    pixel_valid = np.empty(n_pairs, dtype=nb.boolean)
    res = np.empty(n_pairs, dtype=np.float64)

    for px in range(n_pixels):
        n_valid = 0
        for p in range(n_pairs):
            c = data_flat[p, px]
            re = np.float64(c.real)
            im = np.float64(c.imag)
            pw = np.float64(weight_flat[p, px])
            ang = np.arctan2(im, re)
            if (re == 0.0 and im == 0.0) or not np.isfinite(ang) or pw <= 0.0:
                pixel_ang[p] = 0.0
                pixel_w[p] = 0.0
                pixel_valid[p] = False
            else:
                pixel_ang[p] = ang
                pixel_w[p] = pw
                pixel_valid[p] = True
                n_valid += 1
        if n_valid < 4:
            continue

        # Precompute cos/sin of phases for trig recurrence
        cos_ph = np.empty(n_pairs, dtype=np.float64)
        sin_ph = np.empty(n_pairs, dtype=np.float64)
        for p in range(n_pairs):
            if pixel_valid[p]:
                cos_ph[p] = np.cos(pixel_ang[p])
                sin_ph[p] = np.sin(pixel_ang[p])
            else:
                cos_ph[p] = 0.0; sin_ph[p] = 0.0

        step = 2.0 * gv_range / N_BIN
        best_S = -1.0
        best_v = 0.0
        scan_lo = -gv_range

        cos_step = np.empty(n_pairs, dtype=np.float64)
        sin_step = np.empty(n_pairs, dtype=np.float64)
        cos_cur = np.empty(n_pairs, dtype=np.float64)
        sin_cur = np.empty(n_pairs, dtype=np.float64)

        for level in range(1 + max_refine):
            for p in range(n_pairs):
                if pixel_valid[p]:
                    st = step * pair_dt[p]
                    cos_step[p] = np.cos(st); sin_step[p] = np.sin(st)
                    bt = scan_lo * pair_dt[p]
                    cos_cur[p] = np.cos(bt); sin_cur[p] = np.sin(bt)

            use_seasonal = seasonal

            if use_seasonal:
                # Compute residual angles once at level start via arctan2
                for p in range(n_pairs):
                    if pixel_valid[p]:
                        r_cos = cos_ph[p]*cos_cur[p] + sin_ph[p]*sin_cur[p]
                        r_sin = sin_ph[p]*cos_cur[p] - cos_ph[p]*sin_cur[p]
                        res[p] = np.arctan2(r_sin, r_cos)
                # Precompute angle step per pair for this level
                angle_step = step  # res advances by -step*dt per bin

            for bi in range(N_BIN):
                if use_seasonal:
                    A_s = 0.0; B_s = 0.0
                    for irls in range(3):
                        sy = 0.0; cy = 0.0
                        for p in range(n_pairs):
                            if not pixel_valid[p]: continue
                            r = res[p] - A_s*sin_diff[p] - B_s*cos_diff[p]
                            r = r - TWO_PI*np.floor((r+np.pi)/TWO_PI)
                            y = A_s*sin_diff[p] + B_s*cos_diff[p] + r
                            sy += sin_diff[p]*y; cy += cos_diff[p]*y
                        A_s = (cc*sy - sc*cy)/seas_det
                        B_s = (ss*cy - sc*sy)/seas_det

                    sr = 0.0; si = 0.0
                    for p in range(n_pairs):
                        if not pixel_valid[p]: continue
                        a = res[p] - A_s*sin_diff[p] - B_s*cos_diff[p]
                        a = a - TWO_PI*np.floor((a+np.pi)/TWO_PI)
                        sr += pixel_w[p]*np.cos(a); si += pixel_w[p]*np.sin(a)
                else:
                    sr = 0.0; si = 0.0
                    for p in range(n_pairs):
                        if not pixel_valid[p]: continue
                        sr += pixel_w[p]*(cos_ph[p]*cos_cur[p] + sin_ph[p]*sin_cur[p])
                        si += pixel_w[p]*(sin_ph[p]*cos_cur[p] - cos_ph[p]*sin_cur[p])

                S = sr*sr + si*si
                if S > best_S:
                    best_S = S
                    best_v = scan_lo + step*bi

                # Advance: trig recurrence + angle subtraction
                for p in range(n_pairs):
                    if pixel_valid[p]:
                        c = cos_cur[p]*cos_step[p] - sin_cur[p]*sin_step[p]
                        s = sin_cur[p]*cos_step[p] + cos_cur[p]*sin_step[p]
                        cos_cur[p] = c; sin_cur[p] = s
                if use_seasonal:
                    for p in range(n_pairs):
                        if pixel_valid[p]:
                            res[p] -= angle_step * pair_dt[p]
                            res[p] = res[p] - TWO_PI*np.floor((res[p]+np.pi)/TWO_PI)

            scan_lo = best_v - step
            step = 2.0 * step / N_BIN

        velocity[px] = np.float32(best_v)

        # RMSE after velocity removal
        rms_sum = 0.0; w_sum = 0.0
        for p in range(n_pairs):
            if not pixel_valid[p]: continue
            bt = best_v * pair_dt[p]
            cv = np.cos(bt); sv = np.sin(bt)
            r_cos = cos_ph[p]*cv + sin_ph[p]*sv
            r_sin = sin_ph[p]*cv - cos_ph[p]*sv
            r = np.arctan2(r_sin, r_cos)
            rms_sum += pixel_w[p]*r*r; w_sum += pixel_w[p]
        if w_sum > 1e-10:
            rmse[px] = np.float32(np.sqrt(rms_sum / w_sum))

    return velocity, rmse


def velocity_pairs_array(data_chunk, weight_chunk, ref_values, rep_values, max_refine=3, seasonal=False):
    """Estimate global velocity from interferometric pair network.

    Parameters
    ----------
    data_chunk : np.ndarray or list
        3D complex array (n_pairs, chunk_y, chunk_x).
    weight_chunk : np.ndarray or None
        Weight array (real), same shape as data_chunk.
    ref_values : np.ndarray
        1D array of ref dates as int64 (nanoseconds since epoch).
    rep_values : np.ndarray
        1D array of rep dates as int64 (nanoseconds since epoch).
    max_refine : int
        Refinement levels (0=coarse ~32mm/yr, 3=fine ~0.5mm/yr). Default 3.

    Returns
    -------
    np.ndarray
        Velocity array (chunk_y, chunk_x), float32, in rad/year.
    """
    import numpy as np

    if isinstance(data_chunk, list):
        data_np = np.asarray(data_chunk[0]) if len(data_chunk) == 1 else np.concatenate([np.asarray(c) for c in data_chunk], axis=0)
    else:
        data_np = np.asarray(data_chunk)

    if data_np.ndim == 2:
        n_pairs, nx = data_np.shape
        ny = 1
        data_np = data_np.reshape(n_pairs, ny, nx)
    else:
        n_pairs, ny, nx = data_np.shape
    n_pixels = ny * nx

    # Convert 0+0j to NaN
    data_np[data_np == 0] = np.nan + 0j
    data_flat = np.ascontiguousarray(data_np.reshape(n_pairs, n_pixels))
    del data_np

    # Prepare weights
    if isinstance(weight_chunk, list):
        weight_np = np.asarray(weight_chunk[0]) if len(weight_chunk) == 1 else np.concatenate([np.asarray(c) for c in weight_chunk], axis=0)
    elif weight_chunk is not None:
        weight_np = np.asarray(weight_chunk)
    else:
        weight_np = None

    if weight_np is not None:
        weight_flat = np.ascontiguousarray(weight_np.reshape(n_pairs, n_pixels).astype(np.float32))
        weight_flat[~np.isfinite(weight_flat)] = 0.0
        weight_flat[weight_flat < 0] = 0.0
    else:
        weight_flat = np.ones((n_pairs, n_pixels), dtype=np.float32)
    del weight_np

    # Temporal baseline in years
    ns_per_year = 365.25 * 86400 * 1e9
    pair_dt = ((rep_values - ref_values) / ns_per_year).astype(np.float64)

    # Seasonal basis: sin/cos difference between ref and rep dates
    ref_years = ref_values.astype(np.float64) / ns_per_year
    rep_years = rep_values.astype(np.float64) / ns_per_year
    sin_diff = (np.sin(2 * np.pi * ref_years) - np.sin(2 * np.pi * rep_years)).astype(np.float64)
    cos_diff = (np.cos(2 * np.pi * ref_years) - np.cos(2 * np.pi * rep_years)).astype(np.float64)

    # Max velocity: π/2 per shortest interval (unambiguous for noisy phase)
    dt_min_yr = np.min(np.abs(pair_dt[pair_dt != 0]))
    vmax = (np.pi / 2) / dt_min_yr  # rad/year

    vel, rmse = _velocity_pairs_numba_kernel(data_flat, weight_flat, n_pixels, n_pairs,
                                              pair_dt, sin_diff, cos_diff, max_refine, vmax, seasonal)
    del data_flat, weight_flat

    return vel.reshape(ny, nx), rmse.reshape(ny, nx)


@nb.njit(cache=True)
def _trend1d_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex128 or float64
    weight_flat,       # (n_pairs, n_pixels) float32 — correlation weights
    n_pixels,
    n_dates,
    n_pairs,
    date_pair_flat,    # flattened pair indices
    date_time_flat,    # flattened time values (normalized)
    date_sign_flat,    # flattened signs
    date_offsets,      # (n_dates+1,) start offsets into flat arrays
    pair_ref_didx,     # (n_pairs,) ref date index
    pair_rep_didx,     # (n_pairs,) rep date index
    pair_dt,           # (n_pairs,) temporal baseline in intervals (unnormalized)
    max_refine,
    is_complex=True,   # True for wrapped (complex), False for unwrapped (real)
):
    """Per-pixel atmospheric phase estimation using global velocity derotation
    + weighted circular mean.

    1. Global velocity: 16-bin periodogram on all pairs vs temporal baseline.
    2. Derotate pair phases by global velocity.
    3. Per-date weighted circular mean of derotated signed phases (iterative).
    4. Output trend = atmospheric model only (velocity preserved in detrended data).

    Returns
    -------
    trend : (n_pairs, n_pixels) complex64
        Per-pair atmospheric trend. NaN where input is invalid.
    """
    model_angles = np.zeros((n_dates, n_pixels), dtype=np.float64)
    trend = np.full((n_pairs, n_pixels), np.nan + 0j, dtype=np.complex64)

    pixel_angles = np.empty(n_pairs, dtype=np.float64)
    pixel_weights = np.empty(n_pairs, dtype=np.float64)

    for px in range(n_pixels):
        # Extract angles and correlation weights in float64
        if is_complex:
            for p in range(n_pairs):
                c = data_flat[p, px]
                re = np.float64(c.real)
                im = np.float64(c.imag)
                ang = np.arctan2(im, re)
                if (re == 0.0 and im == 0.0) or not np.isfinite(ang):
                    pixel_angles[p] = np.nan
                    pixel_weights[p] = 0.0
                else:
                    pixel_angles[p] = ang
                    pw = np.float64(weight_flat[p, px])
                    pixel_weights[p] = pw if pw > 0.0 else 0.0
        else:
            for p in range(n_pairs):
                pixel_angles[p] = data_flat[p, px].real
                pw = np.float64(weight_flat[p, px])
                pixel_weights[p] = pw if pw > 0.0 else 0.0

        corrected = np.empty(n_pairs, dtype=np.float64)
        local_models = np.zeros(n_dates, dtype=np.float64)

        # Global velocity estimation: periodogram on all pairs vs temporal
        # baseline. Removes the dominant velocity trend so per-date periodogram
        # only needs to find residual (seasonal/nonlinear) slope + atmospheric.
        global_v = 0.0
        if is_complex:
            n_valid_gv = 0
            for p in range(n_pairs):
                if np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0:
                    n_valid_gv += 1
            if n_valid_gv >= 4:
                # Multi-level periodogram (16 coarse + 16 fine = level 1).
                # Range: π/2 per shortest baseline (unambiguous for noisy phase).
                gv_dt_min = 1e30
                for p in range(n_pairs):
                    if np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0:
                        adt = abs(pair_dt[p])
                        if adt > 1e-10 and adt < gv_dt_min:
                            gv_dt_min = adt
                if gv_dt_min > 1e20:
                    gv_dt_min = 1.0
                gv_range = (np.pi * 0.5) / gv_dt_min
                gv_step = 2.0 * gv_range / 16
                best_gS = -1.0
                best_gv = 0.0
                scan_lo = -gv_range
                for level in range(1 + max_refine):
                    for bi in range(16):
                        v_try = scan_lo + gv_step * bi
                        sr = 0.0; si = 0.0
                        for p in range(n_pairs):
                            if not (np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0):
                                continue
                            ang = pixel_angles[p] - v_try * pair_dt[p]
                            ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                            sr += pixel_weights[p] * np.cos(ang)
                            si += pixel_weights[p] * np.sin(ang)
                        S = sr * sr + si * si
                        if S > best_gS:
                            best_gS = S; best_gv = v_try
                    scan_lo = best_gv - gv_step
                    gv_step = 2.0 * gv_step / 16
                global_v = best_gv

        # Single-pass atmospheric fit: derotate by velocity, then per-date circular mean.
        for p in range(n_pairs):
            corrected[p] = pixel_angles[p] - global_v * pair_dt[p]

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
                    if np.isfinite(val) and pixel_weights[pidx] > 0.0:
                        n_valid += 1
                if n_valid < 4:
                    continue

                # Prepare per-pair arrays for this date
                phases = np.empty(n_d, dtype=np.float64)
                t_vals = np.empty(n_d, dtype=np.float64)
                valid = np.empty(n_d, dtype=nb.boolean)
                w_irls = np.empty(n_d, dtype=np.float64)

                for k in range(n_d):
                    pidx = date_pair_flat[d_start + k]
                    val = corrected[pidx] * date_sign_flat[d_start + k]
                    phases[k] = val
                    t_vals[k] = date_time_flat[d_start + k]
                    pw = pixel_weights[pidx]
                    valid[k] = np.isfinite(val) and pw > 0.0
                    if valid[k]:
                        w_irls[k] = pw  # correlation as initial IRLS weight
                    else:
                        phases[k] = 0.0
                        w_irls[k] = 0.0

                # Per-date periodogram search (with global velocity already removed).
                # Finds residual slope (seasonal/nonlinear) + atmospheric intercept.
                if is_complex:
                    # Search range π/4: after global velocity removal, residual
                    # slope is from seasonal variations only. π/4 = π/2 (noisy
                    # limit) / 2 (two dates per pair) — the maximum stable slope.
                    b_range = np.pi * 0.25
                    n_scan = 32
                    scan_step = 2.0 * b_range / n_scan
                    # Precompute cos/sin of phases
                    cos_ph = np.empty(n_d, dtype=np.float64)
                    sin_ph = np.empty(n_d, dtype=np.float64)
                    for k in range(n_d):
                        if valid[k]:
                            cos_ph[k] = np.cos(phases[k])
                            sin_ph[k] = np.sin(phases[k])
                        else:
                            cos_ph[k] = 0.0; sin_ph[k] = 0.0
                    # Coarse scan with trig recurrence
                    cos_step = np.empty(n_d, dtype=np.float64)
                    sin_step = np.empty(n_d, dtype=np.float64)
                    cos_cur = np.empty(n_d, dtype=np.float64)
                    sin_cur = np.empty(n_d, dtype=np.float64)
                    b0 = -b_range
                    for k in range(n_d):
                        if valid[k]:
                            st = scan_step * t_vals[k]
                            cos_step[k] = np.cos(st); sin_step[k] = np.sin(st)
                            bt = b0 * t_vals[k]
                            cos_cur[k] = np.cos(bt); sin_cur[k] = np.sin(bt)
                        else:
                            cos_step[k] = 1.0; sin_step[k] = 0.0
                            cos_cur[k] = 1.0; sin_cur[k] = 0.0
                    best_S = -1.0; best_b = 0.0; best_a = 0.0
                    for bi in range(n_scan):
                        sr = 0.0; si = 0.0
                        for k in range(n_d):
                            if not valid[k]: continue
                            sr += w_irls[k] * (cos_ph[k]*cos_cur[k] + sin_ph[k]*sin_cur[k])
                            si += w_irls[k] * (sin_ph[k]*cos_cur[k] - cos_ph[k]*sin_cur[k])
                        S = sr*sr + si*si
                        if S > best_S:
                            best_S = S; best_b = b0 + scan_step*bi
                            best_a = np.arctan2(si, sr)
                        for k in range(n_d):
                            if valid[k]:
                                c = cos_cur[k]*cos_step[k] - sin_cur[k]*sin_step[k]
                                s = sin_cur[k]*cos_step[k] + cos_cur[k]*sin_step[k]
                                cos_cur[k] = c; sin_cur[k] = s
                    # Fine refinement
                    fine_step = 2.0 * scan_step / n_scan
                    fine_lo = best_b - scan_step
                    for k in range(n_d):
                        if valid[k]:
                            st = fine_step * t_vals[k]
                            cos_step[k] = np.cos(st); sin_step[k] = np.sin(st)
                            bt = fine_lo * t_vals[k]
                            cos_cur[k] = np.cos(bt); sin_cur[k] = np.sin(bt)
                    for bi in range(n_scan):
                        sr = 0.0; si = 0.0
                        for k in range(n_d):
                            if not valid[k]: continue
                            sr += w_irls[k] * (cos_ph[k]*cos_cur[k] + sin_ph[k]*sin_cur[k])
                            si += w_irls[k] * (sin_ph[k]*cos_cur[k] - cos_ph[k]*sin_cur[k])
                        S = sr*sr + si*si
                        if S > best_S:
                            best_S = S; best_b = fine_lo + fine_step*bi
                            best_a = np.arctan2(si, sr)
                        for k in range(n_d):
                            if valid[k]:
                                c = cos_cur[k]*cos_step[k] - sin_cur[k]*sin_step[k]
                                s = sin_cur[k]*cos_step[k] + cos_cur[k]*sin_step[k]
                                cos_cur[k] = c; sin_cur[k] = s
                    c0 = best_a - 2.0 * np.pi * np.floor((best_a + np.pi) / (2.0 * np.pi))
                else:
                    wsum = 0.0; wval = 0.0
                    for k in range(n_d):
                        if valid[k]:
                            wsum += w_irls[k]; wval += w_irls[k] * phases[k]
                    c0 = wval / (wsum + 1e-30)

                local_models[d] = c0

        # Remove linear trend from per-date models — prevents atmospheric
        # model from absorbing net deformation after global velocity removal.
        # Uses periodogram on models vs date index to find the trend slope,
        # then subtracts it. Handles wrapping correctly (models can be near ±π).
        if is_complex and n_dates > 2:
            d_arr = np.empty(n_dates, dtype=np.float64)
            for d in range(n_dates):
                d_arr[d] = np.float64(d) / np.float64(n_dates - 1)  # normalize to [0, 1]
            # Periodogram: find slope of models vs date index
            # Search b ∈ [-π/4, π/4] (same limit as per-date slopes)
            mt_range = np.pi * 0.25
            mt_scan = 16
            mt_step = 2.0 * mt_range / mt_scan
            mt_best_S = -1.0; mt_best_b = 0.0; mt_best_a = 0.0
            for bi in range(mt_scan):
                b_try = -mt_range + mt_step * bi
                sr = 0.0; si = 0.0
                for d in range(n_dates):
                    ang = local_models[d] - b_try * d_arr[d]
                    ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                    sr += np.cos(ang); si += np.sin(ang)
                S = sr * sr + si * si
                if S > mt_best_S:
                    mt_best_S = S; mt_best_b = b_try; mt_best_a = np.arctan2(si, sr)
            # Fine
            mt_fine_lo = mt_best_b - mt_step
            mt_fine_step = 2.0 * mt_step / mt_scan
            for bi in range(mt_scan):
                b_try = mt_fine_lo + mt_fine_step * bi
                sr = 0.0; si = 0.0
                for d in range(n_dates):
                    ang = local_models[d] - b_try * d_arr[d]
                    ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                    sr += np.cos(ang); si += np.sin(ang)
                S = sr * sr + si * si
                if S > mt_best_S:
                    mt_best_S = S; mt_best_b = b_try; mt_best_a = np.arctan2(si, sr)
            # Subtract trend: model[d] -= (a + b*d), wrapped
            for d in range(n_dates):
                correction = mt_best_a + mt_best_b * d_arr[d]
                local_models[d] = local_models[d] - correction
                local_models[d] = local_models[d] - 2.0 * np.pi * np.floor(
                    (local_models[d] + np.pi) / (2.0 * np.pi))

        for d in range(n_dates):
            model_angles[d, px] = local_models[d]

        # Reconstruct per-pair trend (atmospheric only)
        for p in range(n_pairs):
            if np.isfinite(pixel_angles[p]):
                diff = local_models[pair_ref_didx[p]] - local_models[pair_rep_didx[p]]
                if is_complex:
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
    is_complex,      # bool — True for wrapped (complex) phase, False for unwrapped (real)
    has_weight,     # bool — True if w_flat contains real weights, False if unit weights
    bins,           # int — periodogram bins (0=skip periodogram, use circular mean init)
):
    """Per-pixel IRLS linear fitting for detrend1d.

    For wrapped (complex) phase: periodogram init finds the slope globally
    (handles multi-cycle wrapping), then IRLS refines from that init.
    bins controls the periodogram search: range = bins/2, step = 1 rad.
    bins=256 covers DEM errors up to ~500m for C-band Sentinel-1.

    Analytical 2x2 weighted least squares solve per pixel:
    y = a + b*t, 5 accumulators (sw, swt, swt2, swy, swty), Cramer's rule.

    Returns
    -------
    result : (n_samples, n_pixels) complex64 if is_complex, else float32.
        Complex: unit-magnitude trend exp(1j*fit). Real: fitted values.
    slopes : (n_pixels,) float64
        Fitted slope c1 per pixel in normalized dim units. NaN where invalid.
    """
    n_samples, n_pixels = data_flat.shape
    result = np.full((n_samples, n_pixels), np.nan + 0j, dtype=np.complex64)
    slopes = np.full(n_pixels, np.nan, dtype=np.float64)

    # Per-pixel working arrays (reused across pixels)
    angles = np.empty(n_samples, dtype=np.float64)
    w_irls = np.empty(n_samples, dtype=np.float64)
    valid = np.empty(n_samples, dtype=nb.boolean)

    for px in range(n_pixels):
        # Extract angles per-pixel from complex input, or use values directly
        n_valid = 0
        if is_complex:
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

        # Initialize IRLS weights
        for s in range(n_samples):
            if valid[s]:
                w_irls[s] = np.sqrt(w_flat[s, px]) if has_weight else 1.0
            else:
                w_irls[s] = 0.0

        # Init: periodogram (bins>0) or circular mean (bins=0)
        if is_complex and bins > 0:
            # Periodogram init — single-level scan with trig recurrence.
            # range = bins/2, step = 1 rad. Finds slope globally, IRLS refines.
            cos_ph = np.empty(n_samples, dtype=np.float64)
            sin_ph = np.empty(n_samples, dtype=np.float64)
            for s in range(n_samples):
                if valid[s]:
                    cos_ph[s] = np.cos(angles[s])
                    sin_ph[s] = np.sin(angles[s])
                else:
                    cos_ph[s] = 0.0; sin_ph[s] = 0.0

            p_range = 0.5 * bins
            p_step = 2.0 * p_range / bins  # = 1.0
            scan_lo = -p_range
            best_S = -1.0; best_b = 0.0; best_a = 0.0

            # Precompute step and initial rotations per sample
            p_cos_step = np.empty(n_samples, dtype=np.float64)
            p_sin_step = np.empty(n_samples, dtype=np.float64)
            p_cos_cur = np.empty(n_samples, dtype=np.float64)
            p_sin_cur = np.empty(n_samples, dtype=np.float64)
            for s in range(n_samples):
                if valid[s]:
                    st = p_step * dim_norm[s]
                    p_cos_step[s] = np.cos(st); p_sin_step[s] = np.sin(st)
                    bt = scan_lo * dim_norm[s]
                    p_cos_cur[s] = np.cos(bt); p_sin_cur[s] = np.sin(bt)
                else:
                    p_cos_step[s] = 1.0; p_sin_step[s] = 0.0
                    p_cos_cur[s] = 1.0; p_sin_cur[s] = 0.0

            for bi in range(bins):
                sr = 0.0; si = 0.0
                for s in range(n_samples):
                    if not valid[s]: continue
                    sr += w_irls[s] * (cos_ph[s]*p_cos_cur[s] + sin_ph[s]*p_sin_cur[s])
                    si += w_irls[s] * (sin_ph[s]*p_cos_cur[s] - cos_ph[s]*p_sin_cur[s])
                S = sr*sr + si*si
                if S > best_S:
                    best_S = S; best_b = scan_lo + p_step*bi
                    best_a = np.arctan2(si, sr)
                # Trig recurrence: rotate by step
                for s in range(n_samples):
                    if valid[s]:
                        c = p_cos_cur[s]*p_cos_step[s] - p_sin_cur[s]*p_sin_step[s]
                        sn = p_sin_cur[s]*p_cos_step[s] + p_cos_cur[s]*p_sin_step[s]
                        p_cos_cur[s] = c; p_sin_cur[s] = sn

            c0 = best_a - 2.0 * np.pi * np.floor((best_a + np.pi) / (2.0 * np.pi))
            c1 = best_b
        elif is_complex:
            # Circular mean init (bins=0)
            re_sum = 0.0; im_sum = 0.0
            for s in range(n_samples):
                if valid[s]:
                    re_sum += np.float64(data_flat[s, px].real)
                    im_sum += np.float64(data_flat[s, px].imag)
            c0 = np.arctan2(im_sum, re_sum)
            c1 = 0.0
        else:
            c0 = 0.0
            c1 = 0.0

        epsilon = 0.1
        for irls_iter in range(10):
            sw = 0.0; swt = 0.0; swt2 = 0.0
            swy = 0.0; swty = 0.0
            max_dw = 0.0
            for s in range(n_samples):
                if not valid[s]:
                    continue
                t = dim_norm[s]
                fit_val = c0 + c1 * t
                if is_complex:
                    # Wrap residual, then "unwrap" around current model
                    res = angles[s] - fit_val
                    res = res - 2.0 * np.pi * np.floor((res + np.pi) / (2.0 * np.pi))
                    y = fit_val + res
                else:
                    y = angles[s]
                    res = y - fit_val
                w = w_irls[s]
                sw += w; swt += w * t; swt2 += w * t * t
                swy += w * y; swty += w * t * y

                base_w = np.sqrt(w_flat[s, px]) if has_weight else 1.0
                new_w = base_w / (abs(res) + epsilon)
                if new_w > 10.0 * base_w:
                    new_w = 10.0 * base_w
                dw = abs(new_w - w_irls[s])
                if dw > max_dw:
                    max_dw = dw
                w_irls[s] = new_w

            det = sw * swt2 - swt * swt + 1e-30
            c0 = (swt2 * swy - swt * swty) / det
            c1 = (sw * swty - swt * swy) / det

            if max_dw < 1e-3:
                break

        # Store slope (normalized units — caller denormalizes)
        slopes[px] = c1

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
            if is_complex:
                result[s, px] = np.complex64(np.exp(1j * fit_val))
            else:
                result[s, px] = np.complex64(fit_val)

    return result, slopes


def trend1d_array(data, dim_values, weight, intercept=True, slope=True, is_complex=True, bins=128):
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

    result, slopes_norm = _trend1d_numba_kernel(data_flat, w_flat, dim_norm,
                                    intercept, slope, is_complex, has_weight, bins)
    # Denormalize slope: kernel fits in normalized dim, convert to original units
    slopes_2d = (slopes_norm / dim_absmax).astype(np.float32).reshape(ny, nx) if dim_absmax > 0 \
        else np.full((ny, nx), np.nan, dtype=np.float32)

    if is_complex:
        return result.reshape(n_samples, ny, nx), slopes_2d
    else:
        return result.real.astype(np.float32).reshape(n_samples, ny, nx), slopes_2d


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
def _trend2d_sliding_numba_kernel(phase_2d, var0, var1, var2, weight_2d,
                                   half_y, half_x, stride_y, stride_x,
                                   has_weight, is_complex,
                                   g_mu0, g_std0, g_mu1, g_std1, g_mu2, g_std2):
    """Sliding window polynomial fit with column-cached incremental updates.

    Column cache: per-column AtA/Atb/n_valid for the current vertical range.
    Horizontal step: add/remove cached column stats (O(19) per column, not O(wy)).
    Vertical step: update each column cache by adding/removing stride_y rows.
    """
    ny, nx = phase_2d.shape
    NC = 19  # 10 AtA upper-tri + 4 Atb_re + 4 Atb_im + 1 n_valid

    gy_list = list(range(0, ny, stride_y))
    if gy_list[-1] != ny - 1:
        gy_list.append(ny - 1)
    gx_list = list(range(0, nx, stride_x))
    if gx_list[-1] != nx - 1:
        gx_list.append(nx - 1)
    n_gy = len(gy_list)
    n_gx = len(gx_list)

    grid_re = np.full((n_gy, n_gx), np.nan, dtype=np.float64)
    grid_im = np.full((n_gy, n_gx), np.nan, dtype=np.float64)

    # Precompute
    sv0 = np.empty((ny, nx), dtype=np.float64)
    sv1 = np.empty((ny, nx), dtype=np.float64)
    sv2 = np.empty((ny, nx), dtype=np.float64)
    valid = np.zeros((ny, nx), dtype=nb.boolean)
    p_re = np.zeros((ny, nx), dtype=np.float64)
    p_im = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            sv0[i,j] = (np.float64(var0[i,j]) - g_mu0) / g_std0
            sv1[i,j] = (np.float64(var1[i,j]) - g_mu1) / g_std1
            sv2[i,j] = (np.float64(var2[i,j]) - g_mu2) / g_std2
            if not (np.isfinite(sv0[i,j]) and np.isfinite(sv1[i,j]) and np.isfinite(sv2[i,j])):
                continue
            if is_complex:
                pr = np.float64(phase_2d[i,j].real); pi = np.float64(phase_2d[i,j].imag)
                if not (np.isfinite(pr) and np.isfinite(pi)): continue
                mag = np.sqrt(pr*pr + pi*pi)
                if mag == 0: continue
                p_re[i,j] = pr/mag; p_im[i,j] = pi/mag
            else:
                pf = np.float64(phase_2d[i,j].real)
                if not np.isfinite(pf): continue
                p_re[i,j] = pf
            valid[i,j] = True

    min_valid = 8
    # Column cache: cc[j, 0:10]=AtA, cc[j, 10:14]=Atb_re, cc[j, 14:18]=Atb_im, cc[j, 18]=n_valid
    cc = np.zeros((nx, NC), dtype=np.float64)

    # --- Helper: add one pixel (i,j) to column cache cc[j] with sign +1 or -1 ---
    # Inlined below for speed, but the mapping is:
    # cc[j,0]=AtA00 cc[j,1]=AtA01 cc[j,2]=AtA02 cc[j,3]=AtA03
    # cc[j,4]=AtA11 cc[j,5]=AtA12 cc[j,6]=AtA13
    # cc[j,7]=AtA22 cc[j,8]=AtA23 cc[j,9]=AtA33
    # cc[j,10:14]=Atb_re cc[j,14:18]=Atb_im cc[j,18]=n_valid

    # Build initial column caches for first grid row
    wy0 = max(0, gy_list[0] - half_y)
    wy1 = min(ny, gy_list[0] + half_y + 1)
    for j in range(nx):
        for i in range(wy0, wy1):
            if not valid[i,j]: continue
            a1=sv0[i,j]; a2=sv1[i,j]; a3=sv2[i,j]
            if has_weight:
                ww = np.float64(weight_2d[i,j])
                if not (np.isfinite(ww) and ww > 0): continue
                w = np.sqrt(max(ww, 1e-6))
            else:
                w = 1.0
            wa0=w; wa1=a1*w; wa2=a2*w; wa3=a3*w
            cc[j,0]+=wa0*wa0; cc[j,1]+=wa0*wa1; cc[j,2]+=wa0*wa2; cc[j,3]+=wa0*wa3
            cc[j,4]+=wa1*wa1; cc[j,5]+=wa1*wa2; cc[j,6]+=wa1*wa3
            cc[j,7]+=wa2*wa2; cc[j,8]+=wa2*wa3; cc[j,9]+=wa3*wa3
            bw=p_re[i,j]*w
            cc[j,10]+=wa0*bw; cc[j,11]+=wa1*bw; cc[j,12]+=wa2*bw; cc[j,13]+=wa3*bw
            if is_complex:
                bw2=p_im[i,j]*w
                cc[j,14]+=wa0*bw2; cc[j,15]+=wa1*bw2; cc[j,16]+=wa2*bw2; cc[j,17]+=wa3*bw2
            cc[j,18] += 1.0

    prev_wy0 = wy0; prev_wy1 = wy1

    for gi in range(n_gy):
        cy = gy_list[gi]
        wy0 = max(0, cy - half_y)
        wy1 = min(ny, cy + half_y + 1)

        if gi > 0:
            # Update column caches vertically: remove top rows, add bottom rows
            for j in range(nx):
                for i in range(prev_wy0, wy0):
                    if not valid[i,j]: continue
                    a1=sv0[i,j]; a2=sv1[i,j]; a3=sv2[i,j]
                    if has_weight:
                        ww = np.float64(weight_2d[i,j])
                        if not (np.isfinite(ww) and ww > 0): continue
                        w = np.sqrt(max(ww, 1e-6))
                    else:
                        w = 1.0
                    wa0=w; wa1=a1*w; wa2=a2*w; wa3=a3*w
                    cc[j,0]-=wa0*wa0; cc[j,1]-=wa0*wa1; cc[j,2]-=wa0*wa2; cc[j,3]-=wa0*wa3
                    cc[j,4]-=wa1*wa1; cc[j,5]-=wa1*wa2; cc[j,6]-=wa1*wa3
                    cc[j,7]-=wa2*wa2; cc[j,8]-=wa2*wa3; cc[j,9]-=wa3*wa3
                    bw=p_re[i,j]*w
                    cc[j,10]-=wa0*bw; cc[j,11]-=wa1*bw; cc[j,12]-=wa2*bw; cc[j,13]-=wa3*bw
                    if is_complex:
                        bw2=p_im[i,j]*w
                        cc[j,14]-=wa0*bw2; cc[j,15]-=wa1*bw2; cc[j,16]-=wa2*bw2; cc[j,17]-=wa3*bw2
                    cc[j,18] -= 1.0
                for i in range(prev_wy1, wy1):
                    if not valid[i,j]: continue
                    a1=sv0[i,j]; a2=sv1[i,j]; a3=sv2[i,j]
                    if has_weight:
                        ww = np.float64(weight_2d[i,j])
                        if not (np.isfinite(ww) and ww > 0): continue
                        w = np.sqrt(max(ww, 1e-6))
                    else:
                        w = 1.0
                    wa0=w; wa1=a1*w; wa2=a2*w; wa3=a3*w
                    cc[j,0]+=wa0*wa0; cc[j,1]+=wa0*wa1; cc[j,2]+=wa0*wa2; cc[j,3]+=wa0*wa3
                    cc[j,4]+=wa1*wa1; cc[j,5]+=wa1*wa2; cc[j,6]+=wa1*wa3
                    cc[j,7]+=wa2*wa2; cc[j,8]+=wa2*wa3; cc[j,9]+=wa3*wa3
                    bw=p_re[i,j]*w
                    cc[j,10]+=wa0*bw; cc[j,11]+=wa1*bw; cc[j,12]+=wa2*bw; cc[j,13]+=wa3*bw
                    if is_complex:
                        bw2=p_im[i,j]*w
                        cc[j,14]+=wa0*bw2; cc[j,15]+=wa1*bw2; cc[j,16]+=wa2*bw2; cc[j,17]+=wa3*bw2
                    cc[j,18] += 1.0
            prev_wy0 = wy0; prev_wy1 = wy1

        # Build window AtA from column caches for first grid column
        AtA = np.zeros((4, 4), dtype=np.float64)
        Atb_re = np.zeros(4, dtype=np.float64)
        Atb_im = np.zeros(4, dtype=np.float64)
        n_valid = 0
        wx0_init = max(0, gx_list[0] - half_x)
        wx1_init = min(nx, gx_list[0] + half_x + 1)
        for j in range(wx0_init, wx1_init):
            AtA[0,0]+=cc[j,0]; AtA[0,1]+=cc[j,1]; AtA[0,2]+=cc[j,2]; AtA[0,3]+=cc[j,3]
            AtA[1,1]+=cc[j,4]; AtA[1,2]+=cc[j,5]; AtA[1,3]+=cc[j,6]
            AtA[2,2]+=cc[j,7]; AtA[2,3]+=cc[j,8]; AtA[3,3]+=cc[j,9]
            Atb_re[0]+=cc[j,10]; Atb_re[1]+=cc[j,11]; Atb_re[2]+=cc[j,12]; Atb_re[3]+=cc[j,13]
            if is_complex:
                Atb_im[0]+=cc[j,14]; Atb_im[1]+=cc[j,15]; Atb_im[2]+=cc[j,16]; Atb_im[3]+=cc[j,17]
            n_valid += int(cc[j,18])

        # Solve at first grid column
        if n_valid >= min_valid:
            A = AtA.copy()
            A[1,0]=A[0,1]; A[2,0]=A[0,2]; A[3,0]=A[0,3]
            A[2,1]=A[1,2]; A[3,1]=A[1,3]; A[3,2]=A[2,3]
            for k in range(4): A[k,k]+=1e-8
            c = _solve4x4(A, Atb_re)
            cx0 = gx_list[0]
            pr = c[0]+sv0[cy,cx0]*c[1]+sv1[cy,cx0]*c[2]+sv2[cy,cx0]*c[3]
            if is_complex:
                ci = _solve4x4(A, Atb_im)
                ppi = ci[0]+sv0[cy,cx0]*ci[1]+sv1[cy,cx0]*ci[2]+sv2[cy,cx0]*ci[3]
                mag = np.sqrt(pr*pr+ppi*ppi)
                if mag > 0: grid_re[gi,0]=pr/mag; grid_im[gi,0]=ppi/mag
            else:
                grid_re[gi,0] = pr

        # Horizontal sweep using column caches
        prev_wx0 = wx0_init; prev_wx1 = wx1_init

        for gj in range(1, n_gx):
            cx = gx_list[gj]
            new_wx0 = max(0, cx - half_x)
            new_wx1 = min(nx, cx + half_x + 1)

            # Remove left columns from window
            for j in range(prev_wx0, new_wx0):
                AtA[0,0]-=cc[j,0]; AtA[0,1]-=cc[j,1]; AtA[0,2]-=cc[j,2]; AtA[0,3]-=cc[j,3]
                AtA[1,1]-=cc[j,4]; AtA[1,2]-=cc[j,5]; AtA[1,3]-=cc[j,6]
                AtA[2,2]-=cc[j,7]; AtA[2,3]-=cc[j,8]; AtA[3,3]-=cc[j,9]
                Atb_re[0]-=cc[j,10]; Atb_re[1]-=cc[j,11]; Atb_re[2]-=cc[j,12]; Atb_re[3]-=cc[j,13]
                if is_complex:
                    Atb_im[0]-=cc[j,14]; Atb_im[1]-=cc[j,15]; Atb_im[2]-=cc[j,16]; Atb_im[3]-=cc[j,17]
                n_valid -= int(cc[j,18])

            # Add right columns to window
            for j in range(prev_wx1, new_wx1):
                AtA[0,0]+=cc[j,0]; AtA[0,1]+=cc[j,1]; AtA[0,2]+=cc[j,2]; AtA[0,3]+=cc[j,3]
                AtA[1,1]+=cc[j,4]; AtA[1,2]+=cc[j,5]; AtA[1,3]+=cc[j,6]
                AtA[2,2]+=cc[j,7]; AtA[2,3]+=cc[j,8]; AtA[3,3]+=cc[j,9]
                Atb_re[0]+=cc[j,10]; Atb_re[1]+=cc[j,11]; Atb_re[2]+=cc[j,12]; Atb_re[3]+=cc[j,13]
                if is_complex:
                    Atb_im[0]+=cc[j,14]; Atb_im[1]+=cc[j,15]; Atb_im[2]+=cc[j,16]; Atb_im[3]+=cc[j,17]
                n_valid += int(cc[j,18])

            prev_wx0 = new_wx0; prev_wx1 = new_wx1

            if n_valid >= min_valid:
                A = AtA.copy()
                A[1,0]=A[0,1]; A[2,0]=A[0,2]; A[3,0]=A[0,3]
                A[2,1]=A[1,2]; A[3,1]=A[1,3]; A[3,2]=A[2,3]
                for k in range(4): A[k,k]+=1e-8
                c = _solve4x4(A, Atb_re)
                pr = c[0]+sv0[cy,cx]*c[1]+sv1[cy,cx]*c[2]+sv2[cy,cx]*c[3]
                if is_complex:
                    ci = _solve4x4(A, Atb_im)
                    ppi = ci[0]+sv0[cy,cx]*ci[1]+sv1[cy,cx]*ci[2]+sv2[cy,cx]*ci[3]
                    mag = np.sqrt(pr*pr+ppi*ppi)
                    if mag > 0: grid_re[gi,gj]=pr/mag; grid_im[gi,gj]=ppi/mag
                else:
                    grid_re[gi,gj] = pr

    return grid_re, grid_im, gy_list, gx_list


@nb.njit(cache=True)
def _bilinear_interp_trend(grid_re, grid_im, gy, gx, ny, nx, is_complex):
    """Bilinear interpolation of sparse trend grid to full (ny, nx) resolution."""
    n_gy = len(gy)
    n_gx = len(gx)
    out_re = np.full((ny, nx), np.nan, dtype=np.float64)
    out_im = np.full((ny, nx), np.nan, dtype=np.float64)

    # Precompute per-pixel grid indices and fractional weights
    # O(ny + nx) instead of O(ny * nx) searches
    row_gi = np.empty(ny, dtype=np.int64)
    row_fy = np.empty(ny, dtype=np.float64)
    gi = 0
    for i in range(ny):
        while gi < n_gy - 2 and gy[gi + 1] < i:
            gi += 1
        row_gi[i] = gi
        span = gy[gi + 1] - gy[gi]
        row_fy[i] = (i - gy[gi]) / span if span > 0 else 0.0

    col_gj = np.empty(nx, dtype=np.int64)
    col_fx = np.empty(nx, dtype=np.float64)
    gj = 0
    for j in range(nx):
        while gj < n_gx - 2 and gx[gj + 1] < j:
            gj += 1
        col_gj[j] = gj
        span = gx[gj + 1] - gx[gj]
        col_fx[j] = (j - gx[gj]) / span if span > 0 else 0.0

    for i in range(ny):
        gi1 = row_gi[i]
        gi2 = gi1 + 1
        fy = row_fy[i]
        for j in range(nx):
            gj1 = col_gj[j]
            gj2 = gj1 + 1
            fx = col_fx[j]

            # Bilinear weights for 4 corners
            w00 = (1-fy)*(1-fx); w01 = (1-fy)*fx
            w10 = fy*(1-fx);     w11 = fy*fx

            # Accumulate from valid corners only
            wsum = 0.0; vr = 0.0; vi = 0.0
            r = grid_re[gi1, gj1]
            if np.isfinite(r):
                wsum += w00; vr += w00 * r
                if is_complex: vi += w00 * grid_im[gi1, gj1]
            r = grid_re[gi1, gj2]
            if np.isfinite(r):
                wsum += w01; vr += w01 * r
                if is_complex: vi += w01 * grid_im[gi1, gj2]
            r = grid_re[gi2, gj1]
            if np.isfinite(r):
                wsum += w10; vr += w10 * r
                if is_complex: vi += w10 * grid_im[gi2, gj1]
            r = grid_re[gi2, gj2]
            if np.isfinite(r):
                wsum += w11; vr += w11 * r
                if is_complex: vi += w11 * grid_im[gi2, gj2]
            if wsum > 0:
                out_re[i, j] = vr / wsum
                if is_complex:
                    out_im[i, j] = vi / wsum
    return out_re, out_im


@nb.njit(cache=True)
def _gauss_solve(N, rhs, n):
    """Solve N*x = rhs via Gaussian elimination with partial pivoting. In-place."""
    x = np.empty(n, dtype=np.float64)
    for col in range(n):
        max_val = abs(N[col, col]); max_row = col
        for row in range(col+1, n):
            if abs(N[row, col]) > max_val:
                max_val = abs(N[row, col]); max_row = row
        if max_val < 1e-30:
            for i in range(n): x[i] = 0.0
            return x
        if max_row != col:
            for j in range(col, n):
                N[col, j], N[max_row, j] = N[max_row, j], N[col, j]
            rhs[col], rhs[max_row] = rhs[max_row], rhs[col]
        for row in range(col+1, n):
            factor = N[row, col] / N[col, col]
            for j in range(col+1, n):
                N[row, j] -= factor * N[col, j]
            rhs[row] -= factor * rhs[col]
            N[row, col] = 0.0
    for i in range(n-1, -1, -1):
        s = rhs[i]
        for j in range(i+1, n):
            s -= N[i, j] * x[j]
        x[i] = s / N[i, i] if abs(N[i, i]) > 1e-30 else 0.0
    return x


@nb.njit(cache=True)
def _lstsq_baseline_kernel(
    data_flat,         # (n_pairs, n_grid_pixels) complex64 — trend at grid points
    weight_flat,       # (n_pairs, n_grid_pixels) float32 or dummy
    n_grid_pixels,
    n_pairs,
    n_dates,
    pair_ref_didx,     # (n_pairs,) int64
    pair_rep_didx,     # (n_pairs,) int64
    bpr,               # (n_pairs,) float64 or dummy
    has_bpr,
    has_weight,
):
    """IRLS decomposition of per-pair trend into per-date + optional BPR at grid points.

    Returns per-date model values at each grid point: (n_dates, n_grid_pixels) float64.
    """
    TWO_PI = 2.0 * np.pi
    n_unknowns = n_dates + (1 if has_bpr else 0)
    date_model = np.full((n_dates, n_grid_pixels), np.nan, dtype=np.float64)

    angles = np.empty(n_pairs, dtype=np.float64)
    pair_valid = np.empty(n_pairs, dtype=nb.boolean)
    base_w = np.empty(n_pairs, dtype=np.float64)
    N = np.empty((n_unknowns, n_unknowns), dtype=np.float64)
    rhs = np.empty(n_unknowns, dtype=np.float64)
    model = np.empty(n_unknowns, dtype=np.float64)

    for gp in range(n_grid_pixels):
        n_valid = 0
        for p in range(n_pairs):
            c = data_flat[p, gp]
            re = np.float64(c.real); im = np.float64(c.imag)
            if (re == 0.0 and im == 0.0) or not np.isfinite(re):
                pair_valid[p] = False; angles[p] = 0.0; base_w[p] = 0.0
            else:
                pair_valid[p] = True; n_valid += 1
                angles[p] = np.arctan2(im, re)
                base_w[p] = np.sqrt(np.float64(weight_flat[p, gp])) if has_weight else 1.0

        if n_valid < n_unknowns + 3:
            continue

        for i in range(n_unknowns):
            model[i] = 0.0

        epsilon = 0.1
        for irls_iter in range(10):
            for i in range(n_unknowns):
                rhs[i] = 0.0
                for j in range(n_unknowns):
                    N[i, j] = 0.0

            for p in range(n_pairs):
                if not pair_valid[p]: continue
                di = pair_ref_didx[p]
                dj = pair_rep_didx[p]
                pred = model[di] - model[dj]
                if has_bpr:
                    pred += model[n_dates] * bpr[p]
                res = angles[p] - pred
                res = res - TWO_PI * np.floor((res + np.pi) / TWO_PI)
                y = pred + res
                irls_w = base_w[p] / (abs(res) + epsilon)
                if irls_w > 10.0 * base_w[p]:
                    irls_w = 10.0 * base_w[p]

                N[di, di] += irls_w; N[dj, dj] += irls_w
                N[di, dj] -= irls_w; N[dj, di] -= irls_w
                rhs[di] += irls_w * y; rhs[dj] -= irls_w * y
                if has_bpr:
                    b = bpr[p]
                    N[di, n_dates] += irls_w * b; N[n_dates, di] += irls_w * b
                    N[dj, n_dates] -= irls_w * b; N[n_dates, dj] -= irls_w * b
                    N[n_dates, n_dates] += irls_w * b * b
                    rhs[n_dates] += irls_w * b * y

            # Pin first date to 0
            for j in range(n_unknowns):
                N[0, j] = 0.0; N[j, 0] = 0.0
            N[0, 0] = 1.0; rhs[0] = 0.0

            new_model = _gauss_solve(N, rhs, n_unknowns)
            max_dw = 0.0
            for i in range(n_unknowns):
                dw = abs(new_model[i] - model[i])
                if dw > max_dw: max_dw = dw
                model[i] = new_model[i]
            if max_dw < 1e-4:
                break

        for d in range(n_dates):
            date_model[d, gp] = model[d]

    return date_model


def lstsq_baseline_array(data, weight, ref_values, rep_values, bpr_values=None, stride=1):
    """Decompose per-pair complex trend into per-date + optional BPR.

    Subsamples at stride, decomposes at grid points, interpolates per-date
    model back to full resolution, reconstructs consistent per-pair trend.

    Parameters
    ----------
    data : np.ndarray (n_pairs, ny, nx) complex64
        Per-pair trend from trend2d_window.
    weight : np.ndarray or None
        Optional per-pair weight.
    ref_values, rep_values : np.ndarray (n_pairs,) int64
        Pair date values as nanoseconds.
    bpr_values : np.ndarray (n_pairs,) or None
        Perpendicular baseline per pair.
    stride : int
        Subsample step for grid computation. Default 1.

    Returns
    -------
    np.ndarray (n_pairs, ny, nx) complex64
        Network-consistent per-pair trend (BPR component removed if bpr_values given).
    """
    if isinstance(data, list):
        data = np.asarray(data[0]) if len(data) == 1 else np.concatenate([np.asarray(c) for c in data], axis=0)
    n_pairs, ny, nx = data.shape

    # Build date indices
    ns_per_day = 86400 * 1e9
    ref_days = ref_values.astype(np.float64) / ns_per_day
    rep_days = rep_values.astype(np.float64) / ns_per_day
    unique_days = np.unique(np.concatenate([ref_days, rep_days]))
    n_dates = len(unique_days)
    day_to_idx = {d: i for i, d in enumerate(unique_days)}
    pair_ref_didx = np.array([day_to_idx[d] for d in ref_days], dtype=np.int64)
    pair_rep_didx = np.array([day_to_idx[d] for d in rep_days], dtype=np.int64)

    has_bpr = bpr_values is not None
    bpr = bpr_values.astype(np.float64) if has_bpr else np.empty(1, dtype=np.float64)

    # Build stride grid
    if isinstance(stride, (tuple, list)):
        stride_y, stride_x = int(stride[0]), int(stride[1])
    else:
        stride_y = stride_x = int(stride)

    gy_list = list(range(0, ny, stride_y))
    if gy_list[-1] != ny - 1:
        gy_list.append(ny - 1)
    gx_list = list(range(0, nx, stride_x))
    if gx_list[-1] != nx - 1:
        gx_list.append(nx - 1)
    gy = np.array(gy_list, dtype=np.float64)
    gx = np.array(gx_list, dtype=np.float64)
    n_gy, n_gx = len(gy_list), len(gx_list)

    # Subsample trend at grid points
    data_grid = np.empty((n_pairs, n_gy, n_gx), dtype=np.complex64)
    for gi, yi in enumerate(gy_list):
        for gj, xj in enumerate(gx_list):
            data_grid[:, gi, gj] = data[:, int(yi), int(xj)]
    n_grid_pixels = n_gy * n_gx
    data_grid_flat = np.ascontiguousarray(data_grid.reshape(n_pairs, n_grid_pixels))

    # Subsample weights
    has_weight = weight is not None
    if has_weight:
        w_grid = np.empty((n_pairs, n_gy, n_gx), dtype=np.float32)
        for gi, yi in enumerate(gy_list):
            for gj, xj in enumerate(gx_list):
                w_grid[:, gi, gj] = weight[:, int(yi), int(xj)]
        w_grid_flat = np.ascontiguousarray(w_grid.reshape(n_pairs, n_grid_pixels))
    else:
        w_grid_flat = np.empty((1, 1), dtype=np.float32)

    # Decompose at grid points
    date_model_flat = _lstsq_baseline_kernel(
        data_grid_flat, w_grid_flat, n_grid_pixels, n_pairs, n_dates,
        pair_ref_didx, pair_rep_didx, bpr, has_bpr, has_weight,
    )
    # date_model_flat: (n_dates, n_grid_pixels)

    # Interpolate per-date model to full resolution
    date_model_grid = date_model_flat.reshape(n_dates, n_gy, n_gx)

    # Interpolate per-date models to full resolution and reconstruct pairs
    dummy_im = np.zeros((n_gy, n_gx), dtype=np.float64)
    if stride_y > 1 or stride_x > 1:
        date_fullres = np.empty((n_dates, ny, nx), dtype=np.float64)
        for d in range(n_dates):
            date_fullres[d], _ = _bilinear_interp_trend(
                date_model_grid[d], dummy_im, gy, gx, ny, nx, False
            )
    else:
        date_fullres = date_model_grid

    result = np.full((n_pairs, ny, nx), np.nan + 0j, dtype=np.complex64)
    for p in range(n_pairs):
        di = pair_ref_didx[p]; dj = pair_rep_didx[p]
        phase = date_fullres[di] - date_fullres[dj]
        valid = np.isfinite(phase)
        result[p][valid] = np.exp(1j * phase[valid]).astype(np.complex64)

    return result


def trend2d_window_array(phase_2d, variables, weight_2d, win_y, win_x, stride=1):
    """Sliding window local polynomial fit with column-cached incremental updates.

    Per-pixel degree=1 polynomial fit using a sliding window with 3 regressors.
    Uses a fast numba kernel with column-cached incremental normal equations.
    With stride>1, computes at grid points and bilinearly interpolates the trend.

    Parameters
    ----------
    phase_2d : (ny, nx) complex64 or float32
    variables : list of 3 (ny, nx) float32 — regressors (azi, rng, ele)
    weight_2d : (ny, nx) float32 or None
    win_y, win_x : int — window size in pixels
    stride : int or tuple(int, int) — step between grid points (default 1).
        E.g. stride=(10, 40) evaluates every 10th row, 40th column, then
        bilinearly interpolates the trend to full resolution.

    Returns
    -------
    trend : (ny, nx) same dtype
    """
    ny, nx = phase_2d.shape
    is_complex = np.iscomplexobj(phase_2d)

    if not (1 <= len(variables) <= 3):
        raise ValueError("trend2d_window_array requires 1 to 3 variables (e.g. ele, azi+rng, azi+rng+ele)")
    # Pad to 3 variables with zeros if fewer provided
    while len(variables) < 3:
        variables = list(variables) + [np.zeros_like(variables[0])]

    has_weight = weight_2d is not None
    w2d = weight_2d if has_weight else np.empty((1, 1), dtype=np.float32)
    half_y = min(win_y, ny) // 2
    half_x = min(win_x, nx) // 2

    if isinstance(stride, (tuple, list)):
        stride_y, stride_x = int(stride[0]), int(stride[1])
    else:
        stride_y = stride_x = int(stride)

    # Global standardization
    v0, v1, v2 = variables[0], variables[1], variables[2]
    mask = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(v2)
    g_mu0 = np.float64(v0[mask].mean()); g_std0 = max(np.float64(v0[mask].std()), 1e-10)
    g_mu1 = np.float64(v1[mask].mean()); g_std1 = max(np.float64(v1[mask].std()), 1e-10)
    g_mu2 = np.float64(v2[mask].mean()); g_std2 = max(np.float64(v2[mask].std()), 1e-10)

    grid_re, grid_im, gy_list, gx_list = _trend2d_sliding_numba_kernel(
        phase_2d, v0, v1, v2, w2d,
        half_y, half_x, stride_y, stride_x, has_weight, is_complex,
        g_mu0, g_std0, g_mu1, g_std1, g_mu2, g_std2)

    if stride_y == 1 and stride_x == 1:
        trend_re = grid_re
        trend_im = grid_im
    else:
        gy = np.array(gy_list, dtype=np.float64)
        gx = np.array(gx_list, dtype=np.float64)
        trend_re, trend_im = _bilinear_interp_trend(grid_re, grid_im, gy, gx,
                                                      ny, nx, is_complex)

    if is_complex:
        result = np.empty((ny, nx), dtype=np.complex64)
        result.real = trend_re.astype(np.float32)
        result.imag = trend_im.astype(np.float32)
        # Normalize to unit circle
        mag = np.abs(result)
        mag[~(mag > 0)] = 1
        result /= mag
        result[np.isnan(trend_re)] = np.nan + 0j
        return result
    else:
        result = trend_re.astype(np.float32)
        result[np.isnan(trend_re)] = np.nan
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

    # Prepare weight array (correlation)
    if isinstance(weight_chunk, list):
        weight_np = np.asarray(weight_chunk[0]) if len(weight_chunk) == 1 else np.concatenate([np.asarray(c) for c in weight_chunk], axis=0)
    elif weight_chunk is not None:
        weight_np = np.asarray(weight_chunk)
    else:
        weight_np = None

    if weight_np is not None:
        weight_flat = np.ascontiguousarray(weight_np.reshape(n_pairs, n_pixels).astype(np.float32))
        weight_flat[~np.isfinite(weight_flat)] = 0.0
        weight_flat[weight_flat < 0] = 0.0
    else:
        weight_flat = np.ones((n_pairs, n_pixels), dtype=np.float32)
    del weight_np

    # Run numba kernel
    trend_data = _trend1d_pairs_numba_kernel(
        data_flat, weight_flat, n_pixels, n_dates, n_pairs,
        np.array(all_pairs, dtype=np.int64),
        all_times_np,
        np.array(all_signs, dtype=np.float64),
        offsets_np,
        pair_ref_didx, pair_rep_didx,
        (rep_days - ref_days).astype(np.float64),  # pair_dt in days
        max_refine,
        is_complex,
    )
    del data_flat, weight_flat

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
