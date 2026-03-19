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
    return long_idx, left_idx, right_idx, n_trip


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
    trip_offsets,        # (n_pairs+1,) int64 — offsets into trip arrays per pair
    threshold,          # float — triplet cstd threshold
    max_iter,           # int — IRLS iterations
    epsilon,            # float — IRLS regularization
):
    """Per-pixel triplet filtering + IRLS unwrapping.

    Returns unwrapped phases (n_pairs, n_pixels).
    """
    n_pairs, n_pixels = phi_flat.shape
    result = np.full((n_pairs, n_pixels), np.nan, dtype=np.float32)

    # Pre-allocate all working arrays once
    AtWA = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    AtWphi = np.zeros(n_intervals, dtype=np.float64)
    A_work = np.zeros((n_intervals, n_intervals), dtype=np.float64)
    b_work = np.zeros(n_intervals, dtype=np.float64)
    x = np.zeros(n_intervals, dtype=np.float64)
    w_irls = np.ones(n_pairs, dtype=np.float64)
    selected = np.ones(n_pairs, dtype=nb.boolean)
    pair_cstd = np.zeros(n_pairs, dtype=np.float64)

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

        # --- Step 2: IRLS solve ---
        for p in range(n_pairs):
            w_irls[p] = 1.0

        for iteration in range(max_iter):
            for i in range(n_intervals):
                AtWphi[i] = 0.0
                for j in range(n_intervals):
                    AtWA[i, j] = 0.0

            for p in range(n_pairs):
                if not selected[p]:
                    continue
                w = w_flat[p, px] * w_irls[p]
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
                    s_val -= A_work[i, j] * x[j]
                x[i] = s_val / A_work[i, i]

            # Update IRLS weights
            for p in range(n_pairs):
                if not selected[p]:
                    continue
                phi_recon = 0.0
                for i in range(pair_start[p], pair_end[p] + 1):
                    phi_recon += x[i]
                res = phi_flat[p, px] - phi_recon
                res = np.arctan2(np.sin(res), np.cos(res))
                w_irls[p] = 1.0 / (abs(res) + epsilon)

        # --- Step 3: Apply 2π corrections ---
        for p in range(n_pairs):
            if not selected[p]:
                continue
            phi_recon = 0.0
            for i in range(pair_start[p], pair_end[p] + 1):
                phi_recon += x[i]
            correction = _round_half_away_numba((phi_recon - phi_flat[p, px]) / (2.0 * np.pi))
            result[p, px] = phi_flat[p, px] + correction * 2.0 * np.pi

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
):
    """Per-pixel IRLS network inversion from pairs to date increments.

    Uses Fisher-transformed correlation weights and solution-based convergence.
    Returns increments (n_intervals, n_pixels).
    """
    n_pairs, n_pixels = phi_flat.shape
    increments = np.full((n_intervals, n_pixels), np.nan, dtype=np.float32)

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

        for i in range(n_intervals):
            increments[i, px] = x[i]

    return increments


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
        long_np, left_np, right_np, _ = triplets
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
    else:
        trip_long = np.array([], dtype=np.int64)
        trip_left = np.array([], dtype=np.int64)
        trip_right = np.array([], dtype=np.int64)
        trip_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    return trip_long, trip_left, trip_right, trip_offsets


def _flatten_input(phase_stack, weight_stack):
    """Flatten 3D or list inputs to (n_pairs, n_pixels) arrays.

    Returns (phases_flat, weights_flat, n_pairs, height, width).
    """
    if isinstance(phase_stack, list):
        flat_chunks = [np.asarray(c) for c in phase_stack]
        height, width = flat_chunks[0].shape[1], flat_chunks[0].shape[2]
        n_pairs = sum(c.shape[0] for c in flat_chunks)
        n_pixels = height * width
        phases_flat = np.concatenate(
            [c.reshape(c.shape[0], n_pixels) for c in flat_chunks], axis=0)
    else:
        n_pairs, height, width = phase_stack.shape
        n_pixels = height * width
        phases_flat = phase_stack.reshape(n_pairs, n_pixels)

    if weight_stack is not None:
        if isinstance(weight_stack, list):
            flat_w = [np.asarray(c) for c in weight_stack]
            weights_flat = np.concatenate(
                [c.reshape(c.shape[0], n_pixels) for c in flat_w], axis=0)
        else:
            weights_flat = weight_stack.reshape(weight_stack.shape[0], n_pixels)
    else:
        weights_flat = np.ones_like(phases_flat)

    return (np.ascontiguousarray(phases_flat, dtype=np.float32),
            np.ascontiguousarray(weights_flat, dtype=np.float32),
            n_pairs, height, width)


def unwrap1d_pairs_numpy(phase_stack, weight_stack, pair_dates,
                         max_iter=5, epsilon=0.1,
                         threshold=0.5, debug=False):
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
    trip_long, trip_left, trip_right, trip_offsets = _build_sorted_triplet_arrays(
        triplets, n_pairs)

    if debug:
        print(f'1D unwrap: {n_pairs} pairs, {len(dates)} dates, '
              f'{phases_flat.shape[1]} pixels')

    # threshold=None → -1.0 to disable triplet filtering in numba kernel
    thr = -1.0 if threshold is None else float(threshold)

    result = _triplet_irls_unwrap_numba(
        phases_flat, weights_flat, pair_start, pair_end, n_intervals,
        trip_long, trip_left, trip_right, trip_offsets,
        thr, max_iter, epsilon)

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

    increments = _lstsq_numba(
        phases_flat, weights_flat, pair_start, pair_end,
        n_intervals, max_iter, epsilon, x_tol)

    # Build time series — np.cumsum propagates NaN (all-NaN pixels stay NaN)
    ts = np.zeros((n_dates, n_pixels), dtype=np.float32)
    if cumsum:
        ts[1:, :] = np.cumsum(increments, axis=0)
    else:
        ts[1:, :] = increments
    # Set first date to NaN when all increments are NaN
    all_nan_pixels = np.all(np.isnan(increments), axis=0)
    ts[:, all_nan_pixels] = np.nan

    time_series = ts.reshape(n_dates, height, width)

    return time_series, dates
