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
import numpy as np
import pandas as pd


def network_pairs(dates, bpr, days=None, meters=None, invert=False,
                  max_connections=None, min_connections=2, cleanup=True):
    """Baseline pair table (ref, rep, ref_baseline, rep_baseline, pair, baseline, duration) of the dates and their BPR."""
    if days is None:
        days = int(1e6)

    # Build baseline table
    tbl = pd.DataFrame({'date': dates, 'BPR': bpr}).set_index('date')

    # Generate pairs
    data = []
    for line1 in tbl.itertuples():
        for line2 in tbl.itertuples():
            if not (line1.Index < line2.Index and (line2.Index - line1.Index).days <= days):
                continue
            if meters is not None and not (abs(line1.BPR - line2.BPR) <= meters):
                continue

            if not invert:
                data.append({
                    'ref': line1.Index,
                    'rep': line2.Index,
                    'ref_baseline': np.round(line1.BPR, 2),
                    'rep_baseline': np.round(line2.BPR, 2)
                })
            else:
                data.append({
                    'ref': line2.Index,
                    'rep': line1.Index,
                    'ref_baseline': np.round(line2.BPR, 2),
                    'rep_baseline': np.round(line1.BPR, 2)
                })

    if not data:
        raise ValueError("No valid baseline pairs found. "
                         "Try increasing 'days' or 'meters'.")

    df = pd.DataFrame(data).sort_values(['ref', 'rep']).reset_index(drop=True)

    if max_connections is not None:
        _dur = (df['rep'] - df['ref']).dt.days
        keep = pd.Series(True, index=df.index)
        all_dates = sorted(set(df['ref']) | set(df['rep']))
        for date in all_dates:
            # Outgoing pairs (date as ref)
            ref_mask = (df['ref'] == date) & keep
            if ref_mask.sum() > max_connections:
                ref_dur = _dur[ref_mask].sort_values()
                keep[ref_dur.index[max_connections:]] = False
            # Incoming pairs (date as rep)
            rep_mask = (df['rep'] == date) & keep
            if rep_mask.sum() > max_connections:
                rep_dur = _dur[rep_mask].sort_values()
                keep[rep_dur.index[max_connections:]] = False
        df = df[keep].reset_index(drop=True)

    if cleanup:
        from .Baseline import _cleanup_network
        df = _cleanup_network(df, min_connections=min_connections)

    if len(df) == 0:
        raise ValueError("No valid baseline pairs remain after filtering. "
                         "Try increasing 'days' or 'meters'.")

    df = df.reset_index(drop=True)
    df = df.assign(
        pair=[f'{ref.date()} {rep.date()}' for ref, rep in zip(df['ref'], df['rep'])],
        baseline=df['rep_baseline'] - df['ref_baseline'],
        duration=(df['rep'] - df['ref']).dt.days
    )
    return df


def filter_pairs(ref, rep, bpr, days=None, meters=None, exclude_dates=None, exclude_pairs=None, count=None,
                 min_connections=2, cleanup=True):
    """Bool mask of the pairs left by the filters of Batches.filter (days, meters, dates, pairs, connection counts, cleanup)."""
    ref = pd.DatetimeIndex(ref).normalize()
    rep = pd.DatetimeIndex(rep).normalize()
    n_pairs = len(ref)

    # Build mask of valid pairs
    mask = np.ones(n_pairs, dtype=bool)

    if days is not None:
        duration = (rep - ref).days
        mask &= duration <= days

    if meters is not None:
        mask &= np.abs(bpr) <= meters

    if exclude_dates is not None:
        if isinstance(exclude_dates, str):
            exclude_dates = [exclude_dates]
        exclude_dates = pd.to_datetime(exclude_dates).normalize()
        mask &= ~ref.isin(exclude_dates) & ~rep.isin(exclude_dates)

    if exclude_pairs is not None:
        if isinstance(exclude_pairs, str):
            exclude_pairs = [exclude_pairs]
        excluded = set()
        for p in exclude_pairs:
            parts = str(p).split()
            r, s = pd.Timestamp(parts[0]).normalize(), pd.Timestamp(parts[1]).normalize()
            excluded.add((r, s))
        for i in range(n_pairs):
            if (ref[i], rep[i]) in excluded:
                mask[i] = False

    # Build DataFrame for pruning
    df = pd.DataFrame({'ref': ref[mask], 'rep': rep[mask],
                       'idx': np.where(mask)[0]})

    if len(df) > 0:
        from .Baseline import _cleanup_network
        min_conn = max(min_connections, count) if count is not None else min_connections
        if cleanup:
            df = _cleanup_network(df, min_connections=min_conn)
        elif count is not None:
            counts = pd.concat([df['ref'], df['rep']]).value_counts()
            low_dates = set(counts[counts < count].index)
            df = df[~df['ref'].isin(low_dates) & ~df['rep'].isin(low_dates)]

    keep = np.zeros(n_pairs, dtype=bool)
    keep[np.asarray(df['idx'].values, dtype=np.intp)] = True
    return keep


def baseline_difference(b1, a1, b2, a2):
    """Baseline vector rep - ref from lengths b and angles a (degrees): (length, angle in degrees)."""
    hv = []
    for b, a in ((b1, a1), (b2, a2)):
        b = np.asarray(b, dtype=np.float64)
        a = np.radians(np.asarray(a, dtype=np.float64))
        hv.append((b * np.cos(a), b * np.sin(a)))
    vec = (hv[1][0] - hv[0][0], hv[1][1] - hv[0][1])
    return np.hypot(*vec), np.degrees(np.arctan2(vec[1], vec[0]))


def pair_indices(pairs, n_dates, reference, index=None):
    """Index pairs (n, 2) of one burst. Rows [i, j] are positions (index() maps any other entry, a date, first); a None in
    a pair is the burst's reference, reference() gives its position and is called only when a pair needs it. The empty
    list is the sugar: [i, reference] for every date i, the reference with itself included. [None, None] raises."""
    if len(pairs) == 0:
        rows = []
    elif isinstance(pairs[0], (list, tuple, np.ndarray)):
        rows = [list(row) for row in pairs]
    else:
        rows = [list(pairs)]
    for row in rows:
        if all(d is None for d in row):
            raise ValueError(f"ERROR: pairs(): {row} names no date: a pair needs one date, the other entry may be None "
                             "(the reference)")
    if not rows:
        ref = reference()
        return np.array([[i, ref] for i in range(n_dates)], dtype=np.int64)
    ref = None
    out = []
    for row in rows:
        vals = []
        for d in row:
            if d is None:
                if ref is None:
                    ref = reference()
                vals.append(ref)
            elif index is None or isinstance(d, (int, np.integer)):
                vals.append(d)
            else:
                vals.append(index(d, row))
        out.append(vals)
    return np.array(out)
