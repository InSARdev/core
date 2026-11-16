"""
Burst alignment on plain arrays: BatchCore.polyval, residuals, _align_coeffs and align, and Stack.align.

The classes keep the batch part (per-burst dispatch, dask graphs, coordinates, argument checks) and call these functions.
"""
import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse.csgraph import connected_components
from .utils_circular import wrap_phase

MIN_OVERLAP_PIXELS = 50
MIN_ROW_PIXELS = 10
MIN_VALID_ROWS = 5
MIN_INLIER_SAMPLES = 10
MAD_OUTLIER_THRESHOLD = 2.5
OUTPUT_PRECISION = 3
RAMP_PRECISION = 9


def ramp(x, slope, intercept):
    """slope * x + intercept: one pair's linear polynomial on the coordinate x (array or DataArray)."""
    return slope * x + intercept


def extent(y, x):
    """Burst extent (y min, y max, x min, x max) of its coordinate arrays."""
    return (y.min(), y.max(), x.min(), x.max())


def x_centre(x):
    """Burst x centre: the mean of its x coordinates, as a Python float."""
    return float(np.mean(x))


def extents_overlap(e1, e2):
    """Burst overlap test: the extents e1 and e2 (y min, y max, x min, x max) share a box."""
    y1_min, y1_max, x1_min, x1_max = e1
    y2_min, y2_max, x2_min, x2_max = e2
    y_overlap = not (y1_max < y2_min or y2_max < y1_min)
    x_overlap = not (x1_max < x2_min or x2_max < x1_min)
    return y_overlap and x_overlap


def overlap_pairs(extents):
    """Overlapping burst pairs (i, j), i < j, in burst order, of the extents of the bursts in order."""
    pairs = []
    for i, e1 in enumerate(extents):
        for j in range(i + 1, len(extents)):
            if extents_overlap(e1, extents[j]):
                pairs.append((i, j))
    return pairs


def overlap_window(y1, y2, x1, x2):
    """The common y and x slices of two bursts from their coordinate arrays."""
    return (slice(max(y1.min(), y2.min()), min(y1.max(), y2.max())),
            slice(max(x1.min(), x2.min()), min(x1.max(), x2.max())))


def overlap_difference(first, second):
    """Phase difference of an overlap, second burst minus first (lazy for dask or xarray arrays)."""
    return second - first


def edge_weight(count):
    """Weight of an overlap measure: the square root of its valid pixel count."""
    return np.sqrt(float(count))


def double_difference(dd_sum, count, min_pixels=MIN_OVERLAP_PIXELS):
    """An overlap's double difference (phase, count) from its complex sum and pixel count; None below min_pixels."""
    cnt = int(count)
    if cnt >= min_pixels:
        return (float(np.angle(dd_sum / cnt)), cnt)
    return None


def double_difference_sums(ref_1, rep_1, ref_2, rep_2):
    """Lazy scalars of an overlap's double difference: the nansum of its complex values and the count of finite
    pixels. ref_k and rep_k are burst k's overlap at the reference and at the repeat date (DataArrays or arrays)."""
    import dask.array as da_module
    intf1 = ref_1 * rep_1.conj()
    intf2 = ref_2 * rep_2.conj()
    dd = intf1 * intf2.conj()  # lazy (y, x) overlap array
    dd_dask = dd.data  # underlying dask array
    return da_module.nansum(dd_dask), da_module.sum(da_module.isfinite(dd_dask))


def solve_offsets(rows, n_bursts):
    """Corrections of all bursts of one date from its overlap rows (i, j, dd_phase, count): one weighted least-squares
    solve, one anchor row (first burst) per connected component, constraint weight 100 * sum of the weights.
    Returns the correction of every burst (float64, n_bursts)."""
    rows_data = [(i, j, -dd_phase, edge_weight(cnt)) for i, j, dd_phase, cnt in rows]
    n_edges = len(rows_data)
    A = sparse.lil_matrix((n_edges, n_bursts))
    b = np.zeros(n_edges)
    W = np.zeros(n_edges)

    for r, (i, j, offset, weight) in enumerate(rows_data):
        A[r, i] = 1
        A[r, j] = -1
        b[r] = offset
        W[r] = weight

    A = A.tocsr()

    # Connected components for per-component constraints
    adjacency = sparse.lil_matrix((n_bursts, n_bursts))
    for (i, j, _, _) in rows_data:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    n_comp, labels = connected_components(adjacency.tocsr(), directed=False)

    # Constraint: first burst in each component = 0
    constraint_weight = np.sum(W) * 100 if np.sum(W) > 0 else 1e6
    constraints = []
    for comp in range(n_comp):
        members = np.where(labels == comp)[0]
        row = sparse.lil_matrix((1, n_bursts))
        row[0, members[0]] = 1.0
        constraints.append(row.tocsr())

    A_constrained = sparse.vstack([A] + constraints)
    b_constrained = np.concatenate([b, np.zeros(n_comp)])
    W_constrained = np.concatenate([W, np.full(n_comp, constraint_weight)])

    sqrt_W = np.sqrt(W_constrained)
    result = lsqr(sparse.diags(sqrt_W) @ A_constrained.tocsr(), sqrt_W * b_constrained)
    return result[0]


def date_residual(rows, corr):
    """Residual of one date: (after, before) the correction corr (per burst), the weighted mean |double difference|
    over its overlap rows (i, j, dd_phase, count). None without overlaps."""
    abs_discrepancies = []
    abs_before = []
    weights = []
    for i, j, dd_phase, cnt in rows:
        corrected = -dd_phase - (corr[i] - corr[j])
        corrected = wrap_phase(corrected)
        abs_discrepancies.append(abs(corrected))
        abs_before.append(abs(wrap_phase(-dd_phase)))
        weights.append(float(cnt))
    if not abs_discrepancies:
        return None
    return (float(np.average(abs_discrepancies, weights=weights)),
            float(np.average(abs_before, weights=weights)))


def correction_lines(ids, corrections):
    """Debug lines: the corrections (rad) of every burst that has a non-zero one."""
    lines = []
    for i, bid in enumerate(ids):
        corr = corrections[i, :]
        if np.any(corr != 0):
            lines.append(f'  {bid}: corrections = {[f"{c:.4f}" for c in corr]} rad')
    return lines


def rose_dates(residuals, residuals_before, rep_dates):
    """Dates whose residual rose by the correction, at the precision shown (3 decimals)."""
    return [d for d in rep_dates if round(residuals[d], 3) > round(residuals_before[d], 3)]


def combine_coefficients(ramps, offsets1, offsets2, n_pairs):
    """The 3-step coefficients per burst: [ramp, ramp intercept + offset1 + offset2]; per pair when n_pairs is set."""
    if n_pairs is None:
        return {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
    return {b: [[ramps[b][p][0], ramps[b][p][1] + offsets1[b][p] + offsets2[b][p]]
                for p in range(n_pairs)] for b in offsets1}


def overlap_residual(diff, circular: bool):
    """One burst overlap's term of residuals(): (|median phase difference|, valid pixel count, median).

    `diff` is the overlap's phase difference, burst 2 minus burst 1; `circular` wraps it and its median to
    [-pi, pi) first. None when no pixel is valid. align() takes the same term from the overlaps its solve
    already holds, so its 'residual' is this measure without a second pass over the grids.
    """
    wrap = wrap_phase if circular else (lambda x: x)
    valid = np.asarray(diff).ravel()
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return None
    median_diff = np.median(wrap(valid))
    return np.abs(wrap(median_diff)), len(valid), median_diff


def residual_mean(terms, n_pairs: int) -> list:
    """residuals() per pair from (pair index, |median|, weight) overlap terms, in overlap order: the weighted
    mean rounded to 3 decimals, 0.0 for a pair without a valid overlap."""
    sums = [0.0] * n_pairs
    weights = [0.0] * n_pairs
    for p, value, weight in terms:
        sums[p] += value * weight
        weights[p] += weight
    return [0.0 if weights[p] == 0 else round(sums[p] / weights[p], 3) for p in range(n_pairs)]


def residual_report(results, burst_track, n_pairs, discrepancies):
    """Debug output of residuals(): the discrepancy per pair (with its std over the overlaps), and per track of pair 0."""
    per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std
    subswath_stats = {}  # {(track, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
    for pair_idx, abs_discrepancy, weight, id1, id2, _ in results:
        per_overlap_discrepancies[pair_idx].append(abs_discrepancy)

        # Categorize: same track or cross-track
        track1 = burst_track[id1]
        track2 = burst_track[id2]
        if track1 == track2:
            track_key = track1
        else:
            track_key = f'{track1}-{track2}'

        key = (track_key, pair_idx)
        if key not in subswath_stats:
            subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0, 'values': []}
        subswath_stats[key]['sum'] += abs_discrepancy * weight
        subswath_stats[key]['weight'] += weight
        subswath_stats[key]['count'] += 1
        subswath_stats[key]['values'].append(abs_discrepancy)

    # Compute std for overall discrepancy
    for p in range(n_pairs):
        vals = per_overlap_discrepancies[p]
        if len(vals) > 1:
            std = np.std(vals)
            print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} ± {std:.3f} rad ({len(vals)} overlaps)', flush=True)
        else:
            print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} rad ({len(vals)} overlaps)', flush=True)

    # Print per-track stats (only for pair_idx=0 to avoid clutter)
    print('Per-track discrepancy (pair 0):', flush=True)
    for (track, pair_idx), stats in sorted(subswath_stats.items()):
        if pair_idx == 0 and stats['weight'] > 0:
            track_disc = stats['sum'] / stats['weight']
            vals = stats['values']
            if len(vals) > 1:
                track_std = np.std(vals)
                print(f'  {track}: {track_disc:.3f} ± {track_std:.3f} rad ({stats["count"]} overlaps)', flush=True)
            else:
                print(f'  {track}: {track_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)


def _maybe_wrap(x, circular):
    """Wrap to [-π, π) for circular stats, identity otherwise."""
    if circular:
        return wrap_phase(x)
    return x


def _phase_diff(a, center, circular):
    """Circular or linear difference from center."""
    if circular:
        return _maybe_wrap(a - center, circular)
    return a - center


def _phase_mean(a, circular):
    """Circular or linear mean."""
    if circular:
        return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    return np.mean(a)


def _phase_mad(a, center, circular):
    """Circular or linear MAD."""
    return np.median(np.abs(_phase_diff(a, center, circular)))


def row_offset(diff_np, x_coords, degree, method, circular):
    """Offset (and ramp for degree 1) of one overlap from its phase difference diff_np (y by x) and the x coordinates
    of its columns: row means with outlier rejection, the median or mean offset, the ramp of the rows weighted by
    their valid pixels. Returns (offset, ramp, x centroid, valid pixel count), or None for a too small overlap."""
    all_valid = diff_np.ravel()
    all_valid = all_valid[np.isfinite(all_valid)]

    if len(all_valid) < MIN_OVERLAP_PIXELS:
        return None

    if diff_np.ndim < 2:
        return None

    # Row-wise processing
    row_phases = []
    row_x_centroids = []
    row_weights = []

    for y_idx in range(diff_np.shape[0]):
        row = diff_np[y_idx, :]
        valid_mask = np.isfinite(row)
        n_valid = np.sum(valid_mask)
        if n_valid >= MIN_ROW_PIXELS:
            x_valid = x_coords[valid_mask]
            phase_valid = row[valid_mask]
            # Unwrap for row mean computation (needed for both circular and linear)
            if circular:
                phase_unwrapped = np.unwrap(phase_valid)
                row_mean = _maybe_wrap(np.mean(phase_unwrapped), circular)
            else:
                row_mean = np.mean(phase_valid)
            row_phases.append(row_mean)
            row_x_centroids.append(np.mean(x_valid))
            row_weights.append(n_valid)

    if len(row_phases) < MIN_VALID_ROWS:
        return None

    a = np.array(row_phases)
    x_row = np.array(row_x_centroids)
    weights = np.array(row_weights)
    a = _maybe_wrap(a, circular)

    # Outlier rejection
    if method == 'median':
        offset_initial = np.median(a)
        mad = _phase_mad(a, offset_initial, circular)
        if mad > 0:
            inliers = np.abs(_phase_diff(a, offset_initial, circular)) <= MAD_OUTLIER_THRESHOLD * mad
            if np.sum(inliers) >= MIN_INLIER_SAMPLES:
                a = a[inliers]
                x_row = x_row[inliers]
                weights = weights[inliers]

    n_valid = int(np.sum(weights))
    x_centroid = float(np.average(x_row, weights=weights))

    # Compute offset
    if method == 'median':
        sorted_idx = np.argsort(a)
        cumsum = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        offset = a[sorted_idx[median_idx]]
    else:
        offset = _phase_mean(a, circular)

    # Compute ramp if degree=1
    ramp_val = None
    if degree == 1 and len(a) >= MIN_VALID_ROWS:
        x_centered = x_row - x_centroid
        x_range = np.max(x_row) - np.min(x_row)
        if x_range > 100:
            residuals = a - offset
            Swxx = np.sum(weights * x_centered**2)
            Swxr = np.sum(weights * x_centered * residuals)
            if Swxx > 1e-10:
                ramp_val = Swxr / Swxx

    return (_maybe_wrap(offset, circular), ramp_val, x_centroid, n_valid)


def _solve_component(ia, ib, rhs, weights, nc):
    """Weighted least squares of one component of nc bursts: rows x[ib] - x[ia] = rhs with the weights, and the
    anchor row (burst 0) of weight 100 * sum of the weights. Returns the solution (nc)."""
    ncp = len(ia)
    Am = sparse.lil_matrix((ncp + 1, nc))
    bv = np.zeros(ncp + 1)
    Wv = np.zeros(ncp + 1)
    for kk in range(ncp):
        Am[kk, ia[kk]] = -1
        Am[kk, ib[kk]] = +1
        bv[kk] = rhs[kk]
        Wv[kk] = weights[kk]
    cw = np.sum(Wv[:-1]) * 100 if np.sum(Wv[:-1]) > 0 else 1e6
    Am[ncp, 0] = 1
    Wv[ncp] = cw
    sqW = np.sqrt(Wv)
    res = lsqr(sparse.diags(sqW) @ Am.tocsr(), sqW * bv)
    return res[0]


def solve_pair(pairs, ids, id_to_idx, degree, circular, x_centers):
    """Coefficients of one pair from its overlap terms: (id1, id2, offset, weight) for degree 0,
    (id1, id2, offset, ramp, x centroid, weight) for degree 1. One least-squares solve per connected component of
    the overlap graph; bursts without overlaps get zero. Returns {burst id: float32 offset} for degree 0 and
    {burst id: [ramp, intercept]} for degree 1."""
    n_bursts = len(ids)
    if len(pairs) == 0:
        if degree == 0:
            return {bid: np.float32(0.0) for bid in ids}
        else:
            return {bid: [np.float32(0.0), np.float32(0.0)] for bid in ids}

    adj = sparse.lil_matrix((n_bursts, n_bursts))
    for p in pairs:
        adj[id_to_idx[p[0]], id_to_idx[p[1]]] = 1
        adj[id_to_idx[p[1]], id_to_idx[p[0]]] = 1
    n_comp_all, labels = connected_components(adj.tocsr(), directed=False)

    out = {}
    for comp in range(n_comp_all):
        ci = np.where(labels == comp)[0]
        cids = [ids[ii] for ii in ci]
        cmap = {bid: ii for ii, bid in enumerate(cids)}
        nc = len(cids)
        if degree == 0:
            if nc == 1:
                out[cids[0]] = np.float32(0.0)
                continue
            cp = [(a, b, o, w) for a, b, o, w in pairs if a in cmap and b in cmap]
            if not cp:
                for bid in cids:
                    out[bid] = np.float32(0.0)
                continue
            res = _solve_component([cmap[t[0]] for t in cp], [cmap[t[1]] for t in cp],
                                   [t[2] for t in cp], [t[3] for t in cp], nc)
            for ii, bid in enumerate(cids):
                out[bid] = np.float32(round(float(_maybe_wrap(res[ii], circular)), OUTPUT_PRECISION))
        else:
            if nc == 1:
                out[cids[0]] = [np.float32(0.0), np.float32(0.0)]
                continue
            cp = [(a, b, o, r, xc, w) for a, b, o, r, xc, w in pairs if a in cmap and b in cmap]
            if not cp:
                for bid in cids:
                    out[bid] = [np.float32(0.0), np.float32(0.0)]
                continue
            res = _solve_component([cmap[t[0]] for t in cp], [cmap[t[1]] for t in cp],
                                   [t[3] for t in cp], [t[5] for t in cp], nc)
            for ii, bid in enumerate(cids):
                ramp_v = np.float32(round(float(res[ii]), RAMP_PRECISION))
                intercept = np.float32(round(-ramp_v * x_centers[bid], OUTPUT_PRECISION))
                out[bid] = [ramp_v, intercept]
    return out


def wrapped_residual(terms, degree, circular):
    """Weighted mean |offset| of one pair's overlap terms (degree 0: weight at index 3; degree 1: index 5), rounded
    to 3 decimals; 0.0 without terms or without weight."""
    if not terms:
        return 0.0
    offs = [abs(_maybe_wrap(t[2], circular)) for t in terms]
    ws = [t[3] for t in terms] if degree == 0 else [t[5] for t in terms]
    tw = sum(ws)
    return round(sum(o * w for o, w in zip(offs, ws)) / tw, 3) if tw > 0 else 0.0


def solve_coefficients(burst_strips, ids, burst_y, burst_x, pairs, pair_boxes, n_pairs, has_pair_dim, degree, method, circular,
                       x_centers, return_residuals, lazy_residual):
    """The solve of BatchCore._align_coeffs on the burst strips (numpy, in the order of ids): per burst the list of
    (box, array) strips, the boxes (y0, y1, x0, x1) in the burst's index space, covering the union of its overlap windows;
    burst_y and burst_x are the bursts' full coordinates; pairs and pair_boxes give each overlap pair and the window box of
    its first and second burst. The overlap statistics of every pair and pair, the per-component solve, and the residuals.
    Returns a dict with the 'offsets', 'residuals' (None unless return_residuals) and 'residual_after' (None unless lazy_residual)."""
    id_to_idx = {bid: i for i, bid in enumerate(ids)}
    lead = (n_pairs,) if has_pair_dim else ()

    def _window(i, box):
        """The window box of burst i as one numpy array (pair, y, x), copied from the strips that cover it."""
        y0, y1, x0, x1 = box
        strips = burst_strips[i]
        dtype = strips[0][1].dtype if strips else np.float32
        if y1 <= y0 or x1 <= x0:
            return np.empty(lead + (max(y1 - y0, 0), max(x1 - x0, 0)), dtype=dtype)
        out = np.empty(lead + (y1 - y0, x1 - x0), dtype=dtype)
        for (sy0, sy1, sx0, sx1), arr in strips:
            a0, a1 = max(y0, sy0), min(y1, sy1)
            b0, b1 = max(x0, sx0), min(x1, sx1)
            if a0 < a1 and b0 < b1:
                out[..., a0 - y0:a1 - y0, b0 - x0:b1 - x0] = arr[..., a0 - sy0:a1 - sy0, b0 - sx0:b1 - sx0]
        return out

    def _pair_diffs(k):
        """The overlap phase differences of pair k, burst 2 minus burst 1, on the common coordinates: one per pair."""
        id1, id2 = pairs[k]
        i1_idx, i2_idx = id_to_idx[id1], id_to_idx[id2]
        box1, box2 = pair_boxes[k]
        d1 = _window(i1_idx, box1)
        d2 = _window(i2_idx, box2)
        out = []
        for pair_idx in range(n_pairs):
            d1_p = d1[pair_idx] if has_pair_dim else d1
            d2_p = d2[pair_idx] if has_pair_dim else d2
            # Build xarray DataArrays for coordinate-aware overlap
            da1 = xr.DataArray(d1_p, dims=['y', 'x'],
                               coords={'y': burst_y[i1_idx][box1[0]:box1[1]],
                                       'x': burst_x[i1_idx][box1[2]:box1[3]]})
            da2 = xr.DataArray(d2_p, dims=['y', 'x'],
                               coords={'y': burst_y[i2_idx][box2[0]:box2[1]],
                                       'x': burst_x[i2_idx][box2[2]:box2[3]]})
            out.append(da2 - da1)
        return out

    # Compute overlap diffs and process statistics
    _pbp = {p: [] for p in range(n_pairs)}
    for k, (id1, id2) in enumerate(pairs):
        for pair_idx, diff in enumerate(_pair_diffs(k)):
            stat = row_offset(diff.values, diff.coords['x'].values, degree, method, circular)
            if stat is None:
                continue
            _off, _rv, _xcent, _nu = stat
            _w = np.sqrt(_nu)
            if degree == 0:
                _pbp[pair_idx].append((id1, id2, _off, _w))
            else:
                if _rv is not None:
                    _pbp[pair_idx].append((id1, id2, _off, _rv, _xcent, _w))

    rpp = [solve_pair(_pbp[p], ids, id_to_idx, degree, circular, x_centers) for p in range(n_pairs)]

    # Format output
    if n_pairs == 1 and not has_pair_dim:
        offsets = rpp[0]
    else:
        offsets = {bid: [rpp[p][bid] for p in range(n_pairs)] for bid in ids}

    # Residuals (if requested)
    residuals = None
    if return_residuals:
        disc = [wrapped_residual(_pbp[p], degree, circular) for p in range(n_pairs)]
        residuals = disc[0] if (n_pairs == 1 and not has_pair_dim) else disc

    # Residual after the correction (degree 0): residuals()' measure on the
    # overlaps this task already holds, each shifted by its offset difference
    residual_after = None
    if lazy_residual:
        terms = []
        for k, (id1, id2) in enumerate(pairs):
            for pair_idx, diff in enumerate(_pair_diffs(k)):
                shift = rpp[pair_idx][id2] - rpp[pair_idx][id1]
                term = overlap_residual(diff.values - shift, circular)
                if term is not None:
                    terms.append((pair_idx, term[0], term[1]))
        disc = residual_mean(terms, n_pairs)
        residual_after = np.asarray(disc if has_pair_dim else disc[0], dtype=np.float32)

    return {'offsets': offsets, 'residuals': residuals, 'residual_after': residual_after}
