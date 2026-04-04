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
and least squares network inversion, implemented with numba per-pixel kernels.
"""
import numpy as np
import numba as nb
from .utils_detrend import _round_half_away_numba


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


def build_triplets(A):
    """
    Enumerate all valid triplets from incidence matrix.

    For each pair [s..e], find all splits at point m where both [s..m]
    and [m+1..e] exist as pairs. Precompute once, reuse across pixel batches.

    Parameters
    ----------
    A : np.ndarray
        Incidence matrix (n_pairs, n_intervals).

    Returns
    -------
    triplets : tuple or None
        (long_idx, left_idx, right_idx, n_trip) as numpy arrays,
        or None if no triplets exist.
        n_trip[p] = number of triplets where pair p is the long pair.
    """
    n_pairs = A.shape[0]
    pair_start = np.zeros(n_pairs, dtype=np.int32)
    pair_end = np.zeros(n_pairs, dtype=np.int32)
    for p in range(n_pairs):
        nz = np.nonzero(A[p])[0]
        if len(nz) > 0:
            pair_start[p] = nz[0]
            pair_end[p] = nz[-1]

    pair_lookup = {}
    for p in range(n_pairs):
        pair_lookup[(int(pair_start[p]), int(pair_end[p]))] = p

    long_list, left_list, right_list = [], [], []
    for p in range(n_pairs):
        s, e = int(pair_start[p]), int(pair_end[p])
        if s == e:
            continue
        for m in range(s, e):
            if (s, m) in pair_lookup and (m + 1, e) in pair_lookup:
                long_list.append(p)
                left_list.append(pair_lookup[(s, m)])
                right_list.append(pair_lookup[(m + 1, e)])

    if len(long_list) == 0:
        return None

    long_idx = np.array(long_list, dtype=np.int64)
    left_idx = np.array(left_list, dtype=np.int64)
    right_idx = np.array(right_list, dtype=np.int64)
    n_trip = np.bincount(long_idx, minlength=n_pairs).astype(np.int64)

    # Build full reverse index: for each pair, all triplets it participates in
    # (as long, left, or right). Stored as flat array + offsets.
    n_triplets = len(long_idx)
    pair_trip_count = np.zeros(n_pairs, dtype=np.int64)
    for t in range(n_triplets):
        pair_trip_count[long_idx[t]] += 1
        pair_trip_count[left_idx[t]] += 1
        pair_trip_count[right_idx[t]] += 1
    pair_trip_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    pair_trip_offsets[1:] = np.cumsum(pair_trip_count)
    pair_trip_flat = np.empty(pair_trip_offsets[-1], dtype=np.int64)
    fill = np.zeros(n_pairs, dtype=np.int64)
    for t in range(n_triplets):
        for pid in (long_idx[t], left_idx[t], right_idx[t]):
            pair_trip_flat[pair_trip_offsets[pid] + fill[pid]] = t
            fill[pid] += 1

    return long_idx, left_idx, right_idx, n_trip, pair_trip_flat, pair_trip_offsets


@nb.njit(cache=True)
def _triplet_irls_unwrap_numba(
    phi_flat,           # (n_pairs, n_pixels) float32
    w_flat,             # (n_pairs, n_pixels) float32
    pair_start,         # (n_pairs,) int32 — first interval index
    pair_end,           # (n_pairs,) int32 — last interval index (inclusive)
    n_intervals,        # int
    trip_long,          # (n_triplets,) int64
    trip_left,          # (n_triplets,) int64
    trip_right,         # (n_triplets,) int64
    trip_offsets,        # (n_pairs+1,) int64 — offsets into trip arrays by long pair
    pair_trip_flat,     # (sum,) int64 — ALL triplets per pair (long+left+right)
    pair_trip_offsets,  # (n_pairs+1,) int64 — offsets into pair_trip_flat
    threshold,          # float — triplet cstd threshold
    max_iter,           # int — IRLS iterations
    epsilon,            # float — IRLS regularization
    min_probability,    # float — minimum fraction of pairs passing closure (0-1)
):
    """Per-pixel triplet filtering + integer-aware IRLS unwrapping.

    Three robustness mechanisms:
    1. Integer corrections applied inside IRLS loop (not single-shot at end)
    2. Interval wrapping: x[i] → [-π,π], ambiguous intervals (|x[i]|≥π/2) zeroed
    3. Post-unwrap triplet closure check on all pairs

    Returns unwrapped phases (n_pairs, n_pixels).
    """
    n_pairs, n_pixels = phi_flat.shape
    result = np.full((n_pairs, n_pixels), np.nan, dtype=np.float32)

    # Pre-allocate working arrays
    AtWA = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    AtWphi = np.zeros(n_intervals, dtype=np.float64)
    A_work = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    b_work = np.zeros(n_intervals, dtype=np.float64)
    x = np.zeros(n_intervals, dtype=np.float64)
    w_irls = np.ones(n_pairs, dtype=np.float64)
    selected = np.ones(n_pairs, dtype=nb.boolean)
    pair_cstd = np.zeros(n_pairs, dtype=np.float64)
    k_corr = np.zeros(n_pairs, dtype=np.int64)
    k_prev = np.zeros(n_pairs, dtype=np.int64)

    for px in range(n_pixels):
        # --- Step 1: Triplet filtering ---
        n_triplets = len(trip_long)
        if n_triplets > 0 and threshold >= 0.0:
            for p in range(n_pairs):
                pair_cstd[p] = np.inf
                selected[p] = False

            for p in range(n_pairs):
                t_start = trip_offsets[p]
                t_end = trip_offsets[p + 1]
                n_t = t_end - t_start
                if n_t < 2:
                    pair_cstd[p] = np.inf
                    continue

                sum_cos = 0.0
                sum_sin = 0.0
                for t in range(t_start, t_end):
                    cl = phi_flat[trip_long[t], px] - phi_flat[trip_left[t], px] - phi_flat[trip_right[t], px]
                    cl = np.arctan2(np.sin(cl), np.cos(cl))
                    sum_cos += np.cos(cl)
                    sum_sin += np.sin(cl)
                R = np.sqrt((sum_cos / n_t)**2 + (sum_sin / n_t)**2)
                if R >= 1.0 - 1e-10:
                    R = 1.0 - 1e-10
                pair_cstd[p] = np.sqrt(-2.0 * np.log(R))

            for p in range(n_pairs):
                if pair_cstd[p] < threshold:
                    selected[p] = True

            for t in range(n_triplets):
                lp = trip_long[t]
                if selected[lp]:
                    selected[trip_left[t]] = True
                    selected[trip_right[t]] = True

            for p in range(n_pairs):
                if not np.isfinite(phi_flat[p, px]) or not np.isfinite(w_flat[p, px]):
                    selected[p] = False
        else:
            for p in range(n_pairs):
                selected[p] = np.isfinite(phi_flat[p, px]) and np.isfinite(w_flat[p, px])

        n_valid = 0
        for p in range(n_pairs):
            if selected[p]:
                n_valid += 1
        if n_valid < 3:
            continue

        # --- Step 2: Integer-aware IRLS solve ---
        for p in range(n_pairs):
            w_irls[p] = 1.0
            k_corr[p] = 0
            k_prev[p] = 0

        converged = False
        for iteration in range(max_iter):
            # Save previous corrections to detect oscillation
            for p in range(n_pairs):
                k_prev[p] = k_corr[p]

            for i in range(n_intervals):
                AtWphi[i] = 0.0
                for j in range(n_intervals):
                    AtWA[i, j] = 0.0

            for p in range(n_pairs):
                if not selected[p]:
                    continue
                w = w_flat[p, px] * w_irls[p]
                phi_corr = np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi
                s = pair_start[p]
                e = pair_end[p]
                for i in range(s, e + 1):
                    AtWphi[i] += w * phi_corr
                    for j in range(s, e + 1):
                        AtWA[i, j] += w

            for i in range(n_intervals):
                AtWA[i, i] += 1e-4

            # Inline GE solve
            for i in range(n_intervals):
                b_work[i] = AtWphi[i]
                for j in range(n_intervals):
                    A_work[i, j] = AtWA[i, j]
            for i in range(n_intervals):
                max_val = abs(A_work[i, i])
                max_row = i
                for r in range(i + 1, n_intervals):
                    if abs(A_work[r, i]) > max_val:
                        max_val = abs(A_work[r, i])
                        max_row = r
                if max_row != i:
                    for c in range(n_intervals):
                        A_work[i, c], A_work[max_row, c] = A_work[max_row, c], A_work[i, c]
                    b_work[i], b_work[max_row] = b_work[max_row], b_work[i]
                for r in range(i + 1, n_intervals):
                    factor = A_work[r, i] / A_work[i, i]
                    for c in range(i, n_intervals):
                        A_work[r, c] -= factor * A_work[i, c]
                    b_work[r] -= factor * b_work[i]
            for i in range(n_intervals - 1, -1, -1):
                s_val = b_work[i]
                for j in range(i + 1, n_intervals):
                    s_val -= A_work[i, j] * x[j]
                x[i] = s_val / A_work[i, i]

            # Update integer corrections and IRLS weights
            k_changed = False
            for p in range(n_pairs):
                if not selected[p]:
                    continue
                phi_recon = 0.0
                for i in range(pair_start[p], pair_end[p] + 1):
                    phi_recon += x[i]
                phi_corr = np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi
                delta = _round_half_away_numba((phi_recon - phi_corr) / (2.0 * np.pi))
                if delta != 0:
                    k_corr[p] += int(delta)
                    k_changed = True
                res = np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi - phi_recon
                res = np.arctan2(np.sin(res), np.cos(res))
                w_irls[p] = 1.0 / (abs(res) + epsilon)

            if not k_changed:
                converged = True
                break

        # If not converged, mark oscillating pairs as unreliable
        if not converged:
            for p in range(n_pairs):
                if selected[p] and k_corr[p] != k_prev[p]:
                    selected[p] = False

        # --- Step 3: Wrap intervals to [-π, π] and handle ambiguous intervals ---
        # A single interval can't have a 2π jump — enforce minimal phase.
        for i in range(n_intervals):
            x[i] = np.arctan2(np.sin(x[i]), np.cos(x[i]))

        # Ambiguous intervals (|x[i]| ≥ π/2): set to 0 and let pair corrections
        # absorb the phase. This removes the ambiguity — pairs with ref/rep at
        # the jumping date get NaN'd by the triplet closure check if inconsistent.
        for i in range(n_intervals):
            if abs(x[i]) >= np.pi / 2:
                x[i] = 0.0

        for p in range(n_pairs):
            if not selected[p]:
                continue
            phi_recon = 0.0
            for i in range(pair_start[p], pair_end[p] + 1):
                phi_recon += x[i]
            k_corr[p] = int(_round_half_away_numba(
                (phi_recon - np.float64(phi_flat[p, px])) / (2.0 * np.pi)))

        # --- Step 4: Triplet closure check (ALL triplets per pair) ---
        PERTURB_EPS = 0.1
        CLOSURE_THRESHOLD = np.pi * 0.5

        n_passed = 0
        n_selected = 0
        for p in range(n_pairs):
            if not selected[p]:
                continue
            n_selected += 1
            phi_corr = np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi

            # Check ALL triplets containing this pair (as long, left, or right)
            pt_start = pair_trip_offsets[p]
            pt_end = pair_trip_offsets[p + 1]
            n_bad = 0
            n_checked = 0
            for ti in range(pt_start, pt_end):
                t = pair_trip_flat[ti]
                tl = trip_long[t]; tleft = trip_left[t]; tright = trip_right[t]
                if not (selected[tl] and selected[tleft] and selected[tright]):
                    continue
                ul = np.float64(phi_flat[tl, px]) + k_corr[tl] * 2.0 * np.pi
                uleft = np.float64(phi_flat[tleft, px]) + k_corr[tleft] * 2.0 * np.pi
                uright = np.float64(phi_flat[tright, px]) + k_corr[tright] * 2.0 * np.pi
                closure = ul - uleft - uright
                n_checked += 1
                if abs(closure) > CLOSURE_THRESHOLD:
                    n_bad += 1
            if n_bad > 0:
                selected[p] = False
                continue

            # Perturbation check for pairs with no triplets
            if n_checked == 0:
                phi_recon = 0.0
                for i in range(pair_start[p], pair_end[p] + 1):
                    phi_recon += x[i]
                raw_diff = phi_recon - np.float64(phi_flat[p, px])
                k_base = _round_half_away_numba(raw_diff / (2.0 * np.pi))
                k_plus = _round_half_away_numba((raw_diff - PERTURB_EPS) / (2.0 * np.pi))
                k_minus = _round_half_away_numba((raw_diff + PERTURB_EPS) / (2.0 * np.pi))
                if k_plus != k_base or k_minus != k_base:
                    selected[p] = False
                    continue

            n_passed += 1

        # If too few pairs passed closure → pixel is unsolvable
        if n_selected > 0 and n_passed < int(n_selected * min_probability):
            continue  # NaN entire pixel

        # RMS residual check: random/incoherent input has σ_res ≈ π/√3 ≈ 1.8
        # Good data has σ_res ≈ noise level (0.1-0.3). Reject if σ_res > π/4.
        rms_sum = 0.0
        rms_n = 0
        for p in range(n_pairs):
            if not selected[p]:
                continue
            phi_recon = 0.0
            for i in range(pair_start[p], pair_end[p] + 1):
                phi_recon += x[i]
            res = np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi - phi_recon
            res = np.arctan2(np.sin(res), np.cos(res))
            rms_sum += res * res
            rms_n += 1
        if rms_n > 0 and np.sqrt(rms_sum / rms_n) > np.pi * 0.25:
            continue  # high residual → random/incoherent input

        for p in range(n_pairs):
            if selected[p]:
                result[p, px] = np.float32(np.float64(phi_flat[p, px]) + k_corr[p] * 2.0 * np.pi)

    return result


@nb.njit(cache=True)
def _lstsq_numba(
    phi_flat,           # (n_pairs, n_pixels) float32  — unwrapped phases
    w_flat,             # (n_pairs, n_pixels) float32
    pair_start,         # (n_pairs,) int32
    pair_end,           # (n_pairs,) int32
    n_intervals,        # int
    max_iter,           # int
    epsilon,            # float
    x_tol,              # float — solution convergence tolerance
    cumsum,             # bool — if True, return cumulative sum (time series)
):
    """Per-pixel IRLS network inversion from pairs to date time series.

    Uses Fisher-transformed correlation weights and solution-based convergence.
    Returns time series (n_dates, n_pixels) with first date = 0 (or NaN if all-NaN).
    """
    n_pairs, n_pixels = phi_flat.shape
    n_dates = n_intervals + 1
    result = np.full((n_dates, n_pixels), np.nan, dtype=np.float32)

    # Pre-allocate all working arrays once
    AtWA = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    AtWphi = np.zeros(n_intervals, dtype=np.float64)
    A_work = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    b_work = np.zeros(n_intervals, dtype=np.float64)
    x = np.zeros(n_intervals, dtype=np.float64)
    x_new = np.zeros(n_intervals, dtype=np.float64)
    w_irls = np.ones(n_pairs, dtype=np.float64)

    # Pre-compute Fisher-transformed weights: W / sqrt(1 - W²)
    w_fisher = np.empty((n_pairs, n_pixels), dtype=np.float64)
    for p in range(n_pairs):
        for px2 in range(n_pixels):
            wv = w_flat[p, px2]
            if not np.isfinite(wv):
                w_fisher[p, px2] = 0.0
                continue
            if wv < 1e-6:
                wv = 1e-6
            elif wv > 0.9999:
                wv = 0.9999
            w_fisher[p, px2] = wv / np.sqrt(1.0 - wv * wv)

    for px in range(n_pixels):
        n_valid = 0
        for p in range(n_pairs):
            if np.isfinite(phi_flat[p, px]) and np.isfinite(w_flat[p, px]):
                n_valid += 1
        if n_valid < 3:
            continue

        for p in range(n_pairs):
            w_irls[p] = 1.0
        for i in range(n_intervals):
            x[i] = 0.0

        for iteration in range(max_iter):
            for i in range(n_intervals):
                AtWphi[i] = 0.0
                for j in range(n_intervals):
                    AtWA[i, j] = 0.0

            for p in range(n_pairs):
                if not np.isfinite(phi_flat[p, px]) or not np.isfinite(w_flat[p, px]):
                    continue
                w = w_fisher[p, px] * w_irls[p]
                s = pair_start[p]
                e = pair_end[p]
                for i in range(s, e + 1):
                    AtWphi[i] += w * phi_flat[p, px]
                    for j in range(s, e + 1):
                        AtWA[i, j] += w

            for i in range(n_intervals):
                AtWA[i, i] += 1e-4

            # Inline GE solve with pre-allocated workspace
            for i in range(n_intervals):
                b_work[i] = AtWphi[i]
                for j in range(n_intervals):
                    A_work[i, j] = AtWA[i, j]
            for i in range(n_intervals):
                max_val = abs(A_work[i, i])
                max_row = i
                for r in range(i + 1, n_intervals):
                    if abs(A_work[r, i]) > max_val:
                        max_val = abs(A_work[r, i])
                        max_row = r
                if max_row != i:
                    for c in range(n_intervals):
                        A_work[i, c], A_work[max_row, c] = A_work[max_row, c], A_work[i, c]
                    b_work[i], b_work[max_row] = b_work[max_row], b_work[i]
                for r in range(i + 1, n_intervals):
                    factor = A_work[r, i] / A_work[i, i]
                    for c in range(i, n_intervals):
                        A_work[r, c] -= factor * A_work[i, c]
                    b_work[r] -= factor * b_work[i]
            for i in range(n_intervals - 1, -1, -1):
                s_val = b_work[i]
                for j in range(i + 1, n_intervals):
                    s_val -= A_work[i, j] * x_new[j]
                x_new[i] = s_val / A_work[i, i]

            # Check solution convergence
            dx = 0.0
            for i in range(n_intervals):
                dx += (x_new[i] - x[i])**2
            dx = np.sqrt(dx / n_intervals)

            for i in range(n_intervals):
                x[i] = x_new[i]

            if dx < x_tol:
                break

            # Update IRLS weights
            for p in range(n_pairs):
                if not np.isfinite(phi_flat[p, px]) or not np.isfinite(w_flat[p, px]):
                    continue
                phi_recon = 0.0
                for i in range(pair_start[p], pair_end[p] + 1):
                    phi_recon += x[i]
                res = phi_flat[p, px] - phi_recon
                w_irls[p] = 1.0 / (abs(res) + epsilon)

        # RMS residual check: reject if fit residual is too high
        rms_sum = 0.0
        rms_n = 0
        for p in range(n_pairs):
            if not np.isfinite(phi_flat[p, px]) or not np.isfinite(w_flat[p, px]):
                continue
            phi_recon = 0.0
            for i in range(pair_start[p], pair_end[p] + 1):
                phi_recon += x[i]
            res = phi_flat[p, px] - phi_recon
            rms_sum += res * res
            rms_n += 1
        if rms_n > 0 and np.sqrt(rms_sum / rms_n) > np.pi * 0.25:
            continue  # noisy/inconsistent input → NaN

        # Build time series for this pixel
        result[0, px] = 0.0
        if cumsum:
            cs = 0.0
            for i in range(n_intervals):
                cs += x[i]
                result[i + 1, px] = np.float32(cs)
        else:
            for i in range(n_intervals):
                result[i + 1, px] = np.float32(x[i])

    return result


def _build_pair_spans(A):
    """Extract pair start/end interval indices from incidence matrix."""
    n_pairs = A.shape[0]
    pair_start = np.zeros(n_pairs, dtype=np.int32)
    pair_end = np.zeros(n_pairs, dtype=np.int32)
    for p in range(n_pairs):
        nz = np.nonzero(A[p])[0]
        if len(nz) > 0:
            pair_start[p] = nz[0]
            pair_end[p] = nz[-1]
    return pair_start, pair_end


def _build_sorted_triplet_arrays(triplets, n_pairs):
    """Convert triplets tuple into sorted arrays with offsets for numba kernel."""
    if triplets is not None:
        long_np, left_np, right_np, _, pair_trip_flat, pair_trip_offsets = triplets
        n_triplets = len(long_np)
        sort_idx = np.argsort(long_np)
        trip_long = long_np[sort_idx]
        trip_left = left_np[sort_idx]
        trip_right = right_np[sort_idx]
        trip_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
        for t in range(n_triplets):
            trip_offsets[trip_long[t] + 1] += 1
        for p in range(n_pairs):
            trip_offsets[p + 1] += trip_offsets[p]
        # Remap pair_trip_flat indices from original to sorted order
        inv_sort = np.empty(n_triplets, dtype=np.int64)
        for i in range(n_triplets):
            inv_sort[sort_idx[i]] = i
        pair_trip_flat = inv_sort[pair_trip_flat]
    else:
        trip_long = np.array([], dtype=np.int64)
        trip_left = np.array([], dtype=np.int64)
        trip_right = np.array([], dtype=np.int64)
        trip_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
        pair_trip_flat = np.array([], dtype=np.int64)
        pair_trip_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    return trip_long, trip_left, trip_right, trip_offsets, pair_trip_flat, pair_trip_offsets


def _flatten_input(phase_stack, weight_stack):
    """Flatten 3D or list inputs to (n_pairs, n_pixels) arrays.

    Returns (phases_flat, weights_flat, n_pairs, height, width).
    """
    if isinstance(phase_stack, list):
        if len(phase_stack) == 1:
            phase_stack = np.asarray(phase_stack[0])
        else:
            phase_stack = np.concatenate([np.asarray(c) for c in phase_stack], axis=0)
    n_pairs, height, width = phase_stack.shape
    n_pixels = height * width
    phases_flat = phase_stack.reshape(n_pairs, n_pixels)

    if weight_stack is not None:
        if isinstance(weight_stack, list):
            if len(weight_stack) == 1:
                weight_stack = np.asarray(weight_stack[0])
            else:
                weight_stack = np.concatenate([np.asarray(c) for c in weight_stack], axis=0)
        weights_flat = weight_stack.reshape(weight_stack.shape[0], n_pixels)
    else:
        weights_flat = np.ones_like(phases_flat)

    return (np.ascontiguousarray(phases_flat, dtype=np.float32),
            np.ascontiguousarray(weights_flat, dtype=np.float32),
            n_pairs, height, width)


def unwrap1d_pairs_numpy(phase_stack, weight_stack, pair_dates,
                         max_iter=20, epsilon=0.1,
                         threshold=0.5, min_probability=0.5, debug=False):
    """
    Temporal phase unwrapping on numpy arrays.

    Uses triplet phase closure pre-filtering to identify consistent pairs,
    then IRLS unwrapping on selected pairs. Rejected pairs are set to NaN.

    Parameters
    ----------
    phase_stack : np.ndarray or list
        Wrapped or 2D-unwrapped phases, shape (n_pairs, height, width).
    weight_stack : np.ndarray, list, or None
        Correlation weights, shape (n_pairs, height, width) or None.
    pair_dates : list of tuple
        List of (ref_date, rep_date) for each pair.
    max_iter : int
        Maximum IRLS iterations. Default 5.
    epsilon : float
        IRLS regularization parameter. Default 0.1.
    threshold : float or None
        Pair consistency threshold. Lower = more conservative filtering.
        None disables triplet filtering (all pairs used). Default 0.5.
    debug : bool
        Print debug information.

    Returns
    -------
    unwrapped : np.ndarray
        Temporally unwrapped phases, shape (n_pairs, height, width).
    """
    phases_flat, weights_flat, n_pairs, height, width = _flatten_input(
        phase_stack, weight_stack)

    A, dates = build_incidence_matrix(pair_dates)
    n_intervals = len(dates) - 1
    pair_start, pair_end = _build_pair_spans(A)

    triplets = build_triplets(A)
    trip_long, trip_left, trip_right, trip_offsets, pair_trip_flat, pair_trip_offsets = \
        _build_sorted_triplet_arrays(triplets, n_pairs)

    if debug:
        print(f'1D unwrap: {n_pairs} pairs, {len(dates)} dates, '
              f'{phases_flat.shape[1]} pixels')

    # threshold=None → -1.0 to disable triplet filtering in numba kernel
    thr = -1.0 if threshold is None else float(threshold)

    result = _triplet_irls_unwrap_numba(
        phases_flat, weights_flat, pair_start, pair_end, n_intervals,
        trip_long, trip_left, trip_right, trip_offsets,
        pair_trip_flat, pair_trip_offsets,
        thr, max_iter, epsilon, float(min_probability))

    return result.reshape(n_pairs, height, width)


def lstsq_to_dates_numpy(phase_stack, weight_stack, pair_dates,
                          cumsum=True, max_iter=5, epsilon=0.1,
                          x_tol=0.001, debug=False):
    """
    L1-norm IRLS network inversion from pairs to dates.

    Uses iteratively reweighted least squares (IRLS) with L1-norm to be
    robust against outlier pairs (e.g. from unwrapping errors).
    Fisher-transformed correlation weights and solution-based convergence.

    Parameters
    ----------
    phase_stack : np.ndarray or list
        Unwrapped phases, shape (n_pairs, height, width).
    weight_stack : np.ndarray, list, or None
        Correlation weights, shape (n_pairs, height, width).
    pair_dates : list of tuple
        List of (ref_date, rep_date) for each pair.
    cumsum : bool
        If True, return cumulative displacement. Default True.
    max_iter : int
        Maximum IRLS iterations. Default 5.
    epsilon : float
        IRLS regularization (larger = faster convergence, less L1-like).
        Default 0.1.
    x_tol : float
        Solution convergence tolerance. Stop when RMS change in solution
        vector falls below this. Default 0.001.
    debug : bool
        Print debug information.

    Returns
    -------
    time_series : np.ndarray
        Phase time series, shape (n_dates, height, width).
    dates : list
        Sorted unique dates.
    """
    phases_flat, weights_flat, n_pairs, height, width = _flatten_input(
        phase_stack, weight_stack)
    n_pixels = height * width

    A, dates = build_incidence_matrix(pair_dates)
    n_dates = len(dates)
    n_intervals = n_dates - 1
    pair_start, pair_end = _build_pair_spans(A)

    if debug:
        print(f'lstsq: {n_pairs} pairs -> {n_dates} dates, {n_pixels} pixels')

    time_series = _lstsq_numba(
        phases_flat, weights_flat, pair_start, pair_end,
        n_intervals, max_iter, epsilon, x_tol, cumsum)

    return time_series.reshape(n_dates, height, width), dates


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _phi = np.zeros((3, 1), dtype=np.float32)
    _w = np.ones((3, 1), dtype=np.float32)
    _ps = np.array([0, 0, 0], dtype=np.int32)
    _pe = np.array([1, 1, 1], dtype=np.int32)
    _tl = np.array([2], dtype=np.int64)
    _tL = np.array([0], dtype=np.int64)
    _tR = np.array([1], dtype=np.int64)
    _to = np.array([0, 0, 0, 1], dtype=np.int64)
    # pair_trip: all 3 pairs participate in triplet 0
    _ptf = np.array([0, 0, 0], dtype=np.int64)
    _pto = np.array([0, 1, 2, 3], dtype=np.int64)
    _triplet_irls_unwrap_numba(_phi, _w, _ps, _pe, 2, _tl, _tL, _tR, _to,
                                _ptf, _pto, 1.0, 1, 0.1, 0.5)
    _lstsq_numba(_phi, _w, _ps, _pe, 2, 1, 0.1, 1e-4, True)

_warmup_numba_cache()
