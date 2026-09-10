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
trend fitting.
"""
import numpy as np
import numba as nb


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _c = np.zeros((3, 1), dtype=np.complex64)
    _w = np.ones((1, 1), dtype=np.float32)
    _d = np.array([-1.0, 0.0, 1.0])
    _trend1d_numba_kernel(_c, _w, _d, True, True, True, False, 128)
    _wf = np.ones((3, 1), dtype=np.float32)
    _threshold_pairs_numba_kernel(_c, _wf, 1, 3, np.pi * 0.5)
    _trend1d_pairs_numba_kernel(
        _c, _wf, 1, 2, 3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, -1.0, 0.5]),
        np.array([1.0, -1.0, 1.0]),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        np.array([1.0, 1.0, 2.0]),  # pair_dt
        np.array([0.0, 0.5, 1.0]),  # date_days_norm (real times)
        0, True,
    )
    # the gridded transform's spreaders, one call per rank: a worker that has
    # to compile them itself does it while every other worker compiles the
    # same thing into the same cache
    for _k in (1, 2, 3):
        trend2d_spread(np.ones((1, 1), np.complex128),
                       np.zeros((1, _k)), 8)


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
    date_days_norm,    # (n_dates,) REAL acquisition times normalized to [0, 1]
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
                # symmetric grid: both endpoints scanned (see velocity kernel)
                gv_step = 2.0 * gv_range / 15
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
                    gv_step = 2.0 * gv_step / 15
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
        # Uses periodogram on models vs REAL acquisition time (NOT date index:
        # acquisition intervals vary — e.g. a 36-day gap in a 12-day sequence —
        # and a physically linear-in-time ramp is kinked in index space, so an
        # index-based fit mis-removes it). Handles wrapping (models near ±π).
        if is_complex and n_dates > 2:
            d_arr = date_days_norm
            # Periodogram: find slope of models vs date index
            # Search b ∈ [-π/4, π/4] (same limit as per-date slopes)
            mt_range = np.pi * 0.25
            mt_scan = 16
            # symmetric grid: both endpoints scanned (see velocity kernel)
            mt_step = 2.0 * mt_range / (mt_scan - 1)
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

    return trend, model_angles


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


def trend1d_pairs_array(data_chunk, weight_chunk, ref_values, rep_values,
                         max_refine=3, is_complex=True, return_models=False):
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
    # REAL per-date times, normalized to [0, 1] (intervals vary; index != time)
    _span = float(unique_days[-1] - unique_days[0])
    date_days_norm = ((unique_days - unique_days[0]) /
                      (_span if _span > 0 else 1.0)).astype(np.float64)

    trend_data, model_data = _trend1d_pairs_numba_kernel(
        data_flat, weight_flat, n_pixels, n_dates, n_pairs,
        np.array(all_pairs, dtype=np.int64),
        all_times_np,
        np.array(all_signs, dtype=np.float64),
        offsets_np,
        pair_ref_didx, pair_rep_didx,
        (rep_days - ref_days).astype(np.float64),  # pair_dt in days
        date_days_norm,
        max_refine,
        is_complex,
    )
    del data_flat, weight_flat

    if return_models:
        # per-date models; the caller can interpolate THESE and difference them,
        # which keeps the correction per-date and closure exact. Interpolating
        # the per-pair trend instead is nonlinear in the phase and destroys
        # triplet closure.
        return model_data.reshape(n_dates, ny, nx)
    return trend_data.reshape(n_pairs, ny, nx)







# ============================================================================
# Complex-phase 2-D trend: a robust regression, solved on the spread grid
# ============================================================================
# ---------------------------------------------------------------------------
# The coherent sum by gridding: S(g) = sum u exp(-i g.v) on a lattice of
# candidates IS a type-1 non-uniform DFT of the phasors at the positions v.
# Spreading each sample onto a grid in VARIABLE space with a smooth kernel and
# transforming once evaluates every candidate at the cost of the kernel's
# footprint -- w**k per sample instead of one pass per candidate. The grid is
# linear in the samples, so blocks and bursts add exactly as before.
# ---------------------------------------------------------------------------

TREND2D_W = 7                 # kernel half-support in cells, each side
TREND2D_BETA = 2.30           # exponential-of-semicircle shape, per unit width
TREND2D_PROFILE = 64          # coarse cells per axis for the ramp-start scan


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread1(A, ur, ui, cells, w, beta, M, st, gr, gi):
    for p in range(A.shape[0]):
        t = (A[p, 0] + 0.5) * cells[0] + w
        i0 = int(np.ceil(t - 0.5 * w))
        for d in range(w):
            i = i0 + d
            z = 2.0 * (i - t) / w
            if z <= -1.0 or z >= 1.0:
                continue
            kw = np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
            for q in range(ur.shape[0]):
                gr[q, i] += ur[q, p] * kw
                gi[q, i] += ui[q, p] * kw


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread2(A, ur, ui, cells, w, beta, M, st, gr, gi):
    kv = np.empty((2, w))
    base = np.empty(2, np.int64)
    for p in range(A.shape[0]):
        for a in range(2):
            t = (A[p, a] + 0.5) * cells[a] + w
            i0 = int(np.ceil(t - 0.5 * w))
            base[a] = i0
            for d in range(w):
                z = 2.0 * (i0 + d - t) / w
                kv[a, d] = (np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
                            if -1.0 < z < 1.0 else 0.0)
        for d0 in range(w):
            k0 = kv[0, d0]
            if k0 == 0.0:
                continue
            r0 = (base[0] + d0) * st[0]
            for d1 in range(w):
                kk = k0 * kv[1, d1]
                if kk == 0.0:
                    continue
                idx = r0 + (base[1] + d1) * st[1]
                for q in range(ur.shape[0]):
                    gr[q, idx] += ur[q, p] * kk
                    gi[q, idx] += ui[q, p] * kk


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread3(A, ur, ui, cells, w, beta, M, st, gr, gi):
    kv = np.empty((3, w))
    base = np.empty(3, np.int64)
    for p in range(A.shape[0]):
        for a in range(3):
            t = (A[p, a] + 0.5) * cells[a] + w
            i0 = int(np.ceil(t - 0.5 * w))
            base[a] = i0
            for d in range(w):
                z = 2.0 * (i0 + d - t) / w
                kv[a, d] = (np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
                            if -1.0 < z < 1.0 else 0.0)
        for d0 in range(w):
            k0 = kv[0, d0]
            if k0 == 0.0:
                continue
            r0 = (base[0] + d0) * st[0]
            for d1 in range(w):
                k1 = k0 * kv[1, d1]
                if k1 == 0.0:
                    continue
                r1 = r0 + (base[1] + d1) * st[1]
                for d2 in range(w):
                    kk = k1 * kv[2, d2]
                    if kk == 0.0:
                        continue
                    idx = r1 + (base[2] + d2) * st[2]
                    for q in range(ur.shape[0]):
                        gr[q, idx] += ur[q, p] * kk
                        gi[q, idx] += ui[q, p] * kk


def trend2d_grid_shape(cells):
    """The cells the samples can reach: the extent plus the kernel's footprint."""
    import numpy as _np
    return _np.atleast_1d(_np.asarray(cells, _np.int64)) + 2 * TREND2D_W


def trend2d_spread(z, A, cells):
    """Samples -> the spread grid, (nd, prod(cells + 2w)) real and imaginary.

    A is in [-0.5, 0.5] per axis, the variable centred on its midpoint and
    divided by its extent. Only the cells the samples can reach are held;
    trend2d_read() evaluates the grid at any continuous gradient later.
    """
    import numpy as _np
    w = TREND2D_W
    beta = TREND2D_BETA * w
    k = A.shape[1]
    cells = _np.broadcast_to(_np.asarray(cells, _np.float64).ravel(),
                             (k,)).copy()
    M = trend2d_grid_shape(cells)
    st = _np.ones(k, _np.int64)
    for a in range(k - 2, -1, -1):
        st[a] = st[a + 1] * M[a + 1]
    z = _np.asarray(z)
    if z.ndim == 1:
        z = z[None]
    A = _np.ascontiguousarray(A, dtype=_np.float64)
    ur = _np.ascontiguousarray(z.real, dtype=_np.float64)
    ui = _np.ascontiguousarray(z.imag, dtype=_np.float64)
    size = int(_np.prod(M))
    gr = _np.zeros((z.shape[0], size), _np.float64)
    gi = _np.zeros((z.shape[0], size), _np.float64)
    if z.shape[1]:
        fn = {1: _trend2d_spread1, 2: _trend2d_spread2,
              3: _trend2d_spread3}.get(k)
        if fn is None:
            raise ValueError(f"trend2d(): {k} variables, the gridded transform "
                             f"is written for one, two or three.")
        fn(A, ur, ui, cells, w, beta, M, st, gr, gi)
    return gr, gi


def trend2d_kernel(cells):
    """The spreading kernel as it lands on the grid, one axis.

    ONE UNIT SAMPLE AT THE CENTRE, put through the same routine the data goes
    through, so whatever spreading does to it -- the discretisation included,
    which an analytic kernel transform would miss -- is exactly what divides
    out again.
    """
    import numpy as _np
    gr, gi = trend2d_spread(_np.ones((1, 1), _np.complex128),
                            _np.zeros((1, 1)), [int(cells)])
    return gr[0] + 1j * gi[0]


def trend2d_nodes(cells):
    """Where each cell of the grid sits in the variable, in [-1/2, 1/2]."""
    import numpy as _np
    M = int(cells) + 2 * TREND2D_W
    return (_np.arange(M) - TREND2D_W) / float(cells) - 0.5


def trend2d_reach(cells):
    """Turns the grid can carry, in cycles across the extent.

    HALF the computed band: past that the kernel's transform has decayed and
    dividing by it amplifies noise. The fit is not allowed to walk out here --
    not because the answer would be clipped, but because beyond it the grid no
    longer represents the samples.
    """
    return cells / 4.0


def trend2d_read(S, cells, k, g, kg=None):
    """The spread grid read at ONE arbitrary g, with its gradient.

    NO LATTICE. The grid is a sum of kernels at the sample positions, so
    contracting it against exp(-2 pi i g A) evaluates the coherent sum at any
    g at all, and differentiating the exponentials gives d/dg in the same
    pass. That is what lets the trend be SOLVED instead of searched: a lattice
    only ever offers the candidates it was built with, and its far nodes are
    where the sampling's own sidelobes live, not where the trend is.

    Returns (T, dT), T = sum u exp(-2 pi i g.A) and dT[a] = dT/dg_a, each
    divided by the kernel's own transform so one sample at A contributes
    exp(-2 pi i g.A) and nothing besides.
    """
    import numpy as _np
    cells = int(cells)
    M = cells + 2 * TREND2D_W
    Am = trend2d_nodes(cells)
    if kg is None:
        kg = trend2d_kernel(cells)
    g = _np.atleast_1d(_np.asarray(g, _np.float64))
    ph, dph = [], []
    kh = 1.0 + 0j
    dkh = _np.zeros(k, _np.complex128)
    for a in range(k):
        e = _np.exp(-2j * _np.pi * g[a] * Am)
        d = e * (-2j * _np.pi * Am)
        ph.append(e)
        dph.append(d)
        ka = complex((kg * e).sum())
        kh *= ka
        dkh[a] = complex((kg * d).sum())
    if abs(kh) < 1e-12:
        return _np.nan + 0j, _np.full(k, _np.nan + 0j)
    # the per-axis factor cancels out of dkh/kh, so build it as a log-derivative
    for a in range(k):
        dkh[a] = kh * dkh[a] / complex((kg * ph[a]).sum())

    def _contract(vecs):
        out = _np.asarray(S, _np.complex128).reshape((M,) * k)
        for v in vecs:
            out = _np.tensordot(out, v, axes=([0], [0]))
        return complex(out)

    That = _contract(ph)
    dThat = _np.empty(k, _np.complex128)
    for a in range(k):
        dThat[a] = _contract([dph[b] if b == a else ph[b] for b in range(k)])
    T = That / kh
    dT = (dThat - T * dkh) / kh
    return T, dT


def trend2d_width(cells, k, m=0):
    """Total accumulator columns: grids, count, moments, and -- when m axis
    covariates (dims 'y'/'x') are present -- their coarse profile grid."""
    M = int(cells) + 2 * TREND2D_W
    return (4 * M ** k + trend2d_moment_width(k)
            + (2 * TREND2D_PROFILE ** m if m else 0))


def trend2d_profile(u, Aax):
    """Coarse complex sums over the axis covariates, per date. Additive."""
    import numpy as np
    P = TREND2D_PROFILE
    nd = u.shape[0]
    m = Aax.shape[1]
    b = np.clip(((Aax + 0.5) * P).astype(np.int64), 0, P - 1)
    flat = b[:, 0] if m == 1 else b[:, 0] * P + b[:, 1]
    out = np.zeros((nd, 2 * P ** m))
    for d in range(nd):
        out[d, :P ** m] = np.bincount(flat, weights=u[d].real,
                                      minlength=P ** m)
        out[d, P ** m:] = np.bincount(flat, weights=u[d].imag,
                                      minlength=P ** m)
    return out


def trend2d_start(prof, m, gmax):
    """Ramp start from the marginal profiles, one alternation round.

    An axis covariate samples near-uniformly, so its marginal profile is a
    clean 1-D tone whose dense scan has neither basin nor lobe problems --
    this is what lets a multi-cycle ramp be REACHED. When both axes carry
    big ramps each profile smears the other, so the second axis is scanned
    after removing the first estimate, and the first is scanned again.
    """
    import numpy as np
    P = TREND2D_PROFILE
    z = prof[:P ** m] + 1j * prof[P ** m:]
    Ac = (np.arange(P) + 0.5) / P - 0.5
    gs = np.arange(-gmax, gmax + 1e-9, 0.02)
    E = np.exp(-2j * np.pi * np.outer(gs, Ac))

    def scan(pr):
        return float(gs[int(np.argmax(np.abs(E @ pr)))])

    if m == 1:
        return (scan(z),)
    G = z.reshape(P, P)

    def peak(pr):
        v = np.abs(E @ pr)
        j = int(np.argmax(v))
        return float(gs[j]), float(v[j])

    # start from the LESS smeared marginal -- when both axes carry big ramps
    # each profile attenuates the other, and beginning with the weaker one
    # can lock onto a smear artefact -- then alternate to a fixed point
    g0, v0 = peak(G.sum(1))
    g1, v1 = peak(G.sum(0))
    first = 0 if v0 >= v1 else 1
    g = [0.0, 0.0]
    for r in range(6):
        a = (first + r) % 2
        other = g[1 - a]
        corr = np.exp(-2j * np.pi * other * Ac)
        pr = ((G * corr[None, :]).sum(1) if a == 0
              else (G * corr[:, None]).sum(0))
        gnew = scan(pr)
        if r >= 2 and abs(gnew - g[a]) < 0.02:
            g[a] = gnew
            break
        g[a] = gnew
    return (g[0], g[1])


def trend2d_moment_width(k):
    """Columns the accumulator carries past the grid: n, then the moments."""
    return 1 + k + k * (k + 1) // 2


def trend2d_fit(total, cells, k, axes=(), half=None, maxiter=1000,
                tol=1e-10):
    """The accumulator -> one gradient and one constant per date, SOLVED.

    THE OBJECTIVE IS ALREADY A ROBUST REGRESSION. Maximising
    `sum cos(phi - 2 pi g.A - c)` is an M-estimator whose score is `sin` of
    the residual: bounded, redescending, one pixel can pull the fit by at most
    one unit. That is the same bounded influence the phasor form has always
    bought, written as a regression.

    IT IS SOLVED FROM ZERO, NOT SEARCHED. The old code took the global argmax
    over a lattice reaching `range`. Where the samples crowd into a fraction
    of the variable's extent -- real topography -- the sampling's own
    transform keeps grating lobes far out in the band, and the global argmax
    landed on one of them whenever the date was weak, answering with the
    histogram rather than the phase. The ascent from zero instead follows the
    objective to the stationary point CONNECTED to zero. That is a choice,
    not a theorem: a strong trend on a smooth slope of the objective is still
    reached (dates on this estimator's own test stack converge well past the
    half-power width), but a trend separated from zero by a null of a
    near-uniform sampling is not, and comes back as the small stationary
    point near zero. The reported `limit` is what tells the two sampling
    regimes apart.

    THE STEP IS NEWTON-SCALED WITH A GUARANTEED FALLBACK. The minorant step
    `Xg^-1 s` (Xg the centred second moment of A, s = d|T|/dg) is provably
    ascent -- each cosine term's curvature is bounded by its regressors'
    outer product, so Xg majorises the true curvature -- but by exactly the
    mean alignment |T|/n, so on a weak date it crawls: measured before this
    fix, 13 of 90 real dates were still moving at maxiter and one returned
    1.9 rad short of its own stationary point, flagged as solved. The true
    expected information carries that factor, so the step is rescaled by
    n/|T|; if the scaled trial ever fails to raise |T| the minorant step is
    taken instead, which cannot descend. Tens of iterations instead of
    hundreds, and the tolerance is actually met.

    Returns (gradients in cycles across the extent, constants, coherence at
    the solution, coherence at ZERO trend, resolved, why, limit, err --
    per-variable one-sigma from the checkerboard halves when `half` is given) -- the pair
    of coherences is the verification: the ascent starts at zero, so the
    solution's coherence can only exceed the zero-trend one, and their
    difference is what removing the trend actually bought. why: 0 solved; 1 no pixels; 2 the
    trend walked out of the grid's domain (`range`); 3 the covariate is
    degenerate on this date (singular moments, or no phasor sum at all);
    4 the iteration did not meet `tol` within `maxiter` -- returned as NaN,
    because an unconverged number is a plausible wrong value. `limit` is
    REPORTED, never applied: the half-power width of the sampling's own
    transform per variable, the closest two gradients can be and still be
    told apart.
    """
    import numpy as np
    total = np.asarray(total, np.float64)
    cells = int(cells)
    M = cells + 2 * TREND2D_W
    K = M ** k
    nd = total.shape[0]
    S = total[:, :K] + 1j * total[:, K:2 * K]
    H = total[:, 2 * K:3 * K] + 1j * total[:, 3 * K:4 * K]
    n = total[:, 4 * K]
    m1 = total[:, 4 * K + 1:4 * K + 1 + k]
    m2f = total[:, 4 * K + 1 + k:4 * K + trend2d_moment_width(k)]
    axes = tuple(axes)
    m = len(axes)
    prof = (total[:, 4 * K + trend2d_moment_width(k):] if m else None)
    kg = trend2d_kernel(cells)
    reach = trend2d_reach(cells)
    two_pi = 2 * np.pi

    g = np.full((nd, k), np.nan)
    c = np.full(nd, np.nan)
    coh = np.zeros(nd)
    coh0 = np.zeros(nd)
    why = np.zeros(nd, np.int64)
    lim = np.full((nd, k), np.nan)
    for d in range(nd):
        if n[d] <= 0:
            why[d] = 1
            continue
        # THE HALF-POWER WIDTH OF THE SAMPLING, per variable: walk W -- what
        # a perfectly coherent, trend-free date would score -- until it
        # halves or turns back up into its own lobes. Diagnostic only.
        for a in range(k):
            step = 0.02
            prev = 1.0
            lim[d, a] = reach
            gg = np.zeros(k)
            for i in range(1, int(reach / step) + 1):
                gg[a] = i * step
                Wv = abs(trend2d_read(H[d], cells, k, gg, kg)[0]) / n[d]
                if Wv <= 0.5 or Wv > prev:
                    lim[d, a] = gg[a]
                    break
                prev = Wv
        # THE NORMAL MATRIX OF THE SLOPE, which is the CENTRED second
        # moment of A: the constant is not iterated, it is profiled out, so
        # what the step divides by is the lever the slope actually has.
        m2 = np.empty((k, k))
        p = 0
        for a in range(k):
            for b in range(a, k):
                m2[a, b] = m2[b, a] = m2f[d, p]
                p += 1
        Xg = (two_pi ** 2) * (m2 - np.outer(m1[d], m1[d]) / n[d])

        # THE CONSTANT IS CLOSED FORM, never a free parameter: for any g the
        # best constant is angle(T(g)) and the objective is then just |T(g)|.
        # Iterating it instead lets it settle on the other stationary branch,
        # angle(T) + pi, where the objective is -|T| and the gradient step
        # walks downhill -- measured on this estimator's own test stack,
        # seven dates of ninety converged BELOW where they started.
        T0, dT0 = trend2d_read(S[d], cells, k, np.zeros(k), kg)
        if not np.isfinite(T0) or abs(T0) < 1e-30:
            why[d] = 3
            continue
        coh0[d] = float(abs(T0) / max(n[d], 1.0))

        # branch starts: always zero; plus the profile-scan ramp start for
        # the axis covariates, whose marginals are clean 1-D tones -- this is
        # what reaches a multi-cycle ramp the ascent from zero cannot
        starts = [np.zeros(k)]
        if m:
            gmax = min(reach - 0.25, TREND2D_PROFILE / 4.0)
            sv = trend2d_start(prof[d], m, gmax)
            gs = np.zeros(k)
            for j, ai in enumerate(axes):
                gs[ai] = sv[j]
            if np.max(np.abs(gs)) > 1e-9:
                starts.append(gs)

        best = None
        failed0 = 0
        for bi, g_init in enumerate(starts):
            r = _trend2d_ascend(S[d], cells, k, g_init, kg, Xg, n[d], reach,
                                maxiter, tol)
            if r is None:
                if bi == 0:
                    failed0 = _trend2d_ascend.last_fail
                continue
            if best is None or r[2] > best[2]:
                best = r
        if best is None:
            why[d] = failed0 if failed0 else 2
            continue
        g[d] = best[0]
        c[d] = float(np.angle(best[1]))
        coh[d] = float(best[2] / max(n[d], 1.0))
    # THE ERROR BAR IS MEASURED, NOT DERIVED: the same fit on two coarse
    # checkerboard halves, each started from the full solution so the halves
    # answer for noise and not for basins; half the disagreement is the
    # one-sigma scale of the estimate. One degree of freedom -- honest about
    # the size, noisy about itself.
    err = np.full((nd, k), np.nan)
    if half is not None:
        half = np.asarray(half, np.float64)
        for d in range(nd):
            if why[d]:
                continue
            gh = []
            for tot_h in (half[d], total[d] - half[d]):
                Sh = tot_h[:K] + 1j * tot_h[K:2 * K]
                nh = tot_h[4 * K]
                if nh <= 0:
                    break
                m1h = tot_h[4 * K + 1:4 * K + 1 + k]
                m2h = np.empty((k, k))
                pp = 4 * K + 1 + k
                for a in range(k):
                    for b in range(a, k):
                        m2h[a, b] = m2h[b, a] = tot_h[pp]
                        pp += 1
                Xh = (two_pi ** 2) * (m2h - np.outer(m1h, m1h) / nh)
                r = _trend2d_ascend(Sh, cells, k, g[d], kg, Xh, nh, reach,
                                    maxiter, tol)
                if r is None:
                    break
                gh.append(r[0])
            if len(gh) == 2:
                err[d] = 0.5 * np.abs(gh[0] - gh[1])
    resolved = why == 0
    g[~resolved] = np.nan
    c[~resolved] = np.nan
    return g, c, coh, coh0, resolved, why, lim, err


def _trend2d_ascend(Sd, cells, k, g_init, kg, Xg, nd_, reach, maxiter, tol):
    """One ascent to a stationary point. Returns (g, T, |T|) or None, the
    failure reason left in .last_fail (2 domain, 3 degenerate, 4 no
    convergence)."""
    import numpy as np
    _trend2d_ascend.last_fail = 0
    gd = np.array(g_init, np.float64)
    T, dT = trend2d_read(Sd, cells, k, gd, kg)
    if not np.isfinite(T) or abs(T) < 1e-30:
        _trend2d_ascend.last_fail = 3
        return None
    converged = False
    for _ in range(maxiter):
        absT = abs(T)
        score = (np.conj(T) / absT * dT).real
        try:
            base = np.linalg.solve(Xg, score)
        except np.linalg.LinAlgError:
            _trend2d_ascend.last_fail = 3
            return None
        taken = None
        # Newton first, backing off geometrically to the minorant step,
        # which cannot descend; one cycle per step at most, so the iterate
        # only ever moves along a path the objective supports
        scale = nd_ / absT
        _b = float(np.max(np.abs(base)))
        if _b * scale > 1.0:
            scale = 1.0 / max(_b, 1e-30)
        trials = []
        while scale > 1.0:
            trials.append(scale)
            scale *= 0.5
        trials.append(1.0)
        # the BEST rung, not the first acceptable one: near a peak the
        # Newton-scaled step reflects across it and the far side can sit
        # microscopically higher, so first-accept zig-zags for hundreds of
        # iterations; taking the rung with the highest |T| walks in instead
        best_t = None
        for t_ in trials:
            st = base * t_
            cand = gd + st
            if np.any(np.abs(cand) > reach):
                continue
            Tc, dTc = trend2d_read(Sd, cells, k, cand, kg)
            if np.isfinite(Tc) and abs(Tc) >= absT and                     (best_t is None or abs(Tc) > best_t[1]):
                best_t = (cand, abs(Tc), Tc, dTc, st)
        if best_t is not None:
            gd, T, dT, taken = best_t[0], best_t[2], best_t[3], best_t[4]
        if taken is None:
            # a rejected minorant step means the ascent has hit working
            # precision -- converged -- unless it left the grid's domain
            if np.any(np.abs(gd + base) > reach):
                _trend2d_ascend.last_fail = 2
                return None
            converged = True
            break
        if np.max(np.abs(taken)) < tol:
            converged = True
            break
    if not converged:
        _trend2d_ascend.last_fail = 4
        return None
    return gd, T, abs(T)


def trend2d_accumulate(data_blk, transform_blk, stats, cells, dims=None,
                       checker=None, extent=None, coords=None):
    """One spatial block -> the spread grid the fit reads, per date.

    Everything the estimator reads is a sum over pixels, so a block
    contributes its share and the caller adds them: the phasor grid, the
    ones grid (the sampling's own transform, for the reach), the per-date
    sample count, and the first and second moments of A (the regression's
    normal matrix). All real+imaginary interleaved, all additive.
    """
    import numpy as np
    k = len(transform_blk)
    nb = data_blk.shape[0]
    stats = np.asarray(stats, np.float64).ravel()
    mu = stats[:k]
    span = np.maximum(stats[k:2 * k], 1e-30)
    M = int(cells) + 2 * TREND2D_W
    K = M ** k
    ax_idx = ([i for i, d in enumerate(dims) if d in ('y', 'x')][:2]
              if dims is not None else [])
    out = np.zeros((nb, 1, 1, trend2d_width(cells, k, len(ax_idx))),
                   np.float64)

    # the positions this block can contribute: geometry, and a date with phase
    ny, nx = data_blk.shape[-2:]
    if dims is None:
        dims = ['yx'] * len(transform_blk)
    V = [np.asarray(b, np.float32) for b in transform_blk]
    keep = np.ones(ny * nx, bool)
    for v, d in zip(V, dims):
        if d == 'yx':
            keep &= np.isfinite(v.reshape(-1))
    have = np.zeros(ny * nx, bool)
    for t in range(nb):
        a = np.abs(data_blk[t]).reshape(-1)
        have |= np.isfinite(a) & (a > 0)
    keep &= have
    # A VARIABLE ALONG ONE AXIS RULES OUT WHOLE ROWS OR COLUMNS, and normally
    # none, so the raster-sized mask is only built if it has to be
    for v, d in zip(V, dims):
        if d != 'yx' and not np.isfinite(v).all():
            bad = ~np.isfinite(v)
            keep &= ~(np.repeat(bad, nx) if d == 'y' else np.tile(bad, ny))
    # ONE CHECKERBOARD HALF, when asked: an 8x8 board over the burst
    # extent, so the halves are spatially coarse enough to carry independent
    # atmosphere. The other half is total minus this one, for free.
    if checker is not None:
        yb, xb = coords
        y0, y1, x0, x1 = extent
        iy = np.minimum((np.asarray(yb, np.float64) - y0)
                        / max(y1 - y0, 1e-30) * 8, 7).astype(np.int64)
        ix = np.minimum((np.asarray(xb, np.float64) - x0)
                        / max(x1 - x0, 1e-30) * 8, 7).astype(np.int64)
        par = ((iy[:, None] + ix[None, :]) % 2 == int(checker))
        keep &= par.reshape(-1)
    idx = np.flatnonzero(keep)
    npts = idx.size
    if npts == 0:
        return out

    # INDEXED, NEVER BROADCAST, and scaled to the box the grid spans: the
    # midpoint centring is what puts every sample inside [-1/2, 1/2]
    rows = idx // nx
    A = np.empty((npts, k), np.float64)
    for i, (v, d) in enumerate(zip(V, dims)):
        if d == 'yx':
            A[:, i] = v.reshape(-1)[idx]
        elif d == 'y':
            A[:, i] = v[rows]
        else:
            A[:, i] = v[idx - rows * nx]
        A[:, i] = (A[:, i] - mu[i]) / span[i]
    u = np.zeros((nb, npts), np.complex128)
    for t in range(nb):
        z = data_blk[t].reshape(-1)[idx]
        a = np.abs(z)
        np.divide(z, a, out=u[t], where=np.isfinite(a) & (a > 0))
    # THE MOMENTS TRAVEL WITH THE GRID. The fit is a regression, so it needs
    # the normal matrix of [1, 2 pi A] as well as the coherent sum -- and it
    # must be built from the pixels each DATE actually has, which differ where
    # a date is missing. Three numbers per variable, summed like everything
    # else, so the answer still does not depend on the chunking.
    have = (np.abs(u) > 0)
    out[:, 0, 0, 4 * K] = have.sum(1)
    p = 4 * K + 1
    for i in range(k):
        out[:, 0, 0, p + i] = have @ A[:, i]
    p += k
    for i in range(k):
        for j in range(i, k):
            out[:, 0, 0, p] = have @ (A[:, i] * A[:, j])
            p += 1

    gr, gi = trend2d_spread(u, A, cells)
    out[:, 0, 0, :K] = gr
    out[:, 0, 0, K:2 * K] = gi
    # THE SAMPLING'S OWN TRANSFORM, spread the same way with the phase taken
    # out. It is what a perfectly coherent, TREND-FREE date would score, so it
    # says how far apart two gradients have to be before this variable's
    # distribution can tell them apart -- and where its far lobes are, which
    # is where a global search would land on a weak date. Same routine, same
    # additivity, one extra grid.
    hr, hi = trend2d_spread(have.astype(np.complex128), A, cells)
    out[:, 0, 0, 2 * K:3 * K] = hr
    out[:, 0, 0, 3 * K:4 * K] = hi
    # coarse profiles over the axis covariates: the ramp-start's raw material
    if ax_idx:
        out[:, 0, 0, 4 * K + trend2d_moment_width(k):] = \
            trend2d_profile(u, A[:, ax_idx])
    return out


# Populate numba file cache on first import so dask workers skip compilation.
# LAST, so every kernel above is defined by the time it runs.

# ---------------------------------------------------------------------------
# POLYNOMIAL 2D TREND (degree 1..n over transform variables or the y/x grid).
# A DIFFERENT ESTIMATOR FROM trend2d_* above: those solve one gradient per
# variable on the phasors, this fits a least-squares polynomial surface per
# pair. BatchCore.trend2d() drives these; BatchComplex.trend2d() drives the
# gradient solve. Restored from 14180b7 (2026-08-29) unchanged.
# ---------------------------------------------------------------------------

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

        if n_valid < 2 * n_feat:
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
                 detrend_mode, extrapolate=False):
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
    extrapolate : bool
        If False (default) and not detrend_mode, mask trend to input's valid region.

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
            if not extrapolate:
                nan_mask = np.isnan(phase_chunk[p].real) if is_complex else np.isnan(phase_chunk[p])
                trend[nan_mask] = np.nan
            result[p] = trend

    return result


_warmup_numba_cache()
