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
Static utility functions for 3D per-date phase unwrapping.

Unwraps the per-date stack directly from the complex scenes: the pair
ambiguities are not free (k_ij = n_i - n_j), so every triplet closes by
construction and no network inversion follows.
"""
import numpy as np
import numba as nb


def wrap(data_pairs):
    """Wrap phase to [-pi, pi] range."""
    import xarray as xr
    import dask

    if isinstance(data_pairs, xr.DataArray):
        return xr.DataArray(dask.array.mod(data_pairs.data + np.pi, 2 * np.pi) - np.pi, data_pairs.coords)\
            .rename(data_pairs.name)
    return np.mod(data_pairs + np.pi, 2 * np.pi) - np.pi



@nb.njit(cache=True)
def _unwrap3d_dates_kernel(phi, ok, i1, i2, wgt, tyr, max_iter, search,
                           short_yr, n_short, min_disagree, min_dates, n_trend):
    """Temporal unwrapping on the DATES: one integer per date, not per pair.

    Single-look pair phase is exactly phi_i - phi_j, so the pair ambiguities
    are not free -- k_ij = n_i - n_j. The network has n_dates unknowns rather
    than n_pairs, and every triplet closes BY CONSTRUCTION: there is nothing
    to verify and no pixel to reject for failing to close.

    The wrapped data alone cannot fix n (every n satisfies every pair
    exactly), so the integers come from smoothness: nearby dates differ
    little. Minimise sum_ij w_ij |(phi_i - phi_j) + 2*pi*(n_i - n_j)| by
    coordinate descent from the lag-1 branch.

    A date that cannot be reconciled with its neighbours is DROPPED (left NaN)
    and the chain bridges it with the longer interval to the next good date,
    instead of carrying its error into every later date.
    """
    n_dates, n_pix = phi.shape
    out = np.full((n_dates, n_pix), np.nan, dtype=np.float64)
    n_pairs = i1.shape[0]
    # Serial on purpose: dask parallelises across blocks, and every other numba
    # kernel here is serial for the same reason. A parallel=True kernel called
    # from a multi-threaded dask worker trips numba's workqueue layer --
    # "not threadsafe ... concurrent access has been detected" -- and takes the
    # whole worker down, surfacing only as KilledWorker.
    for px in range(n_pix):
        okp = np.empty(n_dates, dtype=nb.boolean)
        phid = np.empty(n_dates, dtype=np.float64)
        rate = np.empty(n_pairs, dtype=np.float64)
        nok = 0
        for d in range(n_dates):
            okp[d] = ok[d, px]
            if okp[d]:
                nok += 1
        if nok < min_dates:
            continue
        nn = np.zeros(n_dates, dtype=np.float64)
        Phi = np.zeros(n_dates, dtype=np.float64)
        pred = np.zeros(n_short, dtype=np.float64)
        # Remove the trend BEFORE unwrapping. The smoothness objective prefers
        # small |dPhi|, so a genuinely fast pixel gets wrapped down to a
        # slower solution entirely. Demodulating
        # by an estimated rate leaves a small residual, which is where that
        # objective is actually valid; the rate is added back at the end.
        #
        # Seed the rate from the SHORTEST intervals only. Acquisitions are not
        # evenly spaced -- winter gaps reach 90-120 d -- and a wrapped increment
        # over dt resolves rates only below lambda/(4 dt): 422 mm/yr at 12 d but
        # 42 mm/yr at 120 d. Averaging over all consecutive gaps therefore feeds
        # aliased rates into the median exactly where motion is fast enough to
        # care about. So take dtmin from the pixel's own valid dates and keep
        # only band pairs at that sampling minimum, which share the widest
        # unambiguous range the acquisition plan allows; widen only if that
        # leaves too few samples for a median.
        dtmin = 1e30
        for d in range(n_dates - 1):
            if ok[d, px] and ok[d + 1, px]:
                dt = tyr[d + 1] - tyr[d]
                if dt > 1e-9 and dt < dtmin:
                    dtmin = dt
        nd1 = 0
        if dtmin < 1e29:
            for wid in range(4):
                if wid == 0:
                    dtcut = 1.5 * dtmin
                elif wid == 1:
                    dtcut = 2.5 * dtmin
                elif wid == 2:
                    dtcut = 4.0 * dtmin
                else:
                    dtcut = 1e30
                nd1 = 0
                for pp in range(n_pairs):
                    aa = i1[pp]; bb = i2[pp]
                    if not (ok[aa, px] and ok[bb, px]):
                        continue
                    dt = tyr[bb] - tyr[aa]
                    if dt > 1e-9 and dt <= dtcut:
                        dif = phi[bb, px] - phi[aa, px]
                        dif = dif - 2 * np.pi * np.floor((dif + np.pi) / (2 * np.pi))
                        rate[nd1] = dif / dt
                        nd1 += 1
                if nd1 >= 3:
                    break
        vhat = 0.0
        if nd1 >= 3:
            for a in range(nd1):
                for b in range(a + 1, nd1):
                    if rate[b] < rate[a]:
                        tmpv = rate[a]; rate[a] = rate[b]; rate[b] = tmpv
            vhat = rate[nd1 // 2]
        for _tr in range(n_trend):
            for d in range(n_dates):
                phid[d] = phi[d, px] - vhat * tyr[d]
                phid[d] = phid[d] - 2 * np.pi * np.floor((phid[d] + np.pi) / (2 * np.pi))
            # at most one date can be dropped per attempt, and dropping stops
            # at min_dates, so that bound IS the iteration bound -- no separate
            # max_drop knob to disagree with it
            for _att in range(nok - min_dates + 1):
                # lag-1 branch across the CURRENTLY good dates
                prev = -1
                run = 0.0
                for d in range(n_dates):
                    if not okp[d]:
                        continue
                    if prev < 0:
                        run = phid[d]
                        nn[d] = 0.0
                    else:
                        dif = phid[d] - phid[prev]
                        dif = dif - 2 * np.pi * np.floor((dif + np.pi) / (2 * np.pi))
                        run = run + dif
                        nn[d] = np.rint((run - phid[d]) / (2 * np.pi))
                        run = phid[d] + 2 * np.pi * nn[d]
                    prev = d
                # coordinate descent on the integers
                for _it in range(max_iter):
                    changed = False
                    for d in range(n_dates):
                        if not okp[d]:
                            continue
                        best = 1e300
                        bn = nn[d]
                        for c in range(-search, search + 1):
                            cand = nn[d] + c
                            acc = 0.0
                            for p in range(n_pairs):
                                a = i1[p]; b = i2[p]
                                if a != d and b != d:
                                    continue
                                if not (okp[a] and okp[b]):
                                    continue
                                na = cand if a == d else nn[a]
                                nb2 = cand if b == d else nn[b]
                                r = (phid[a] - phid[b]) + 2 * np.pi * (na - nb2)
                                acc += wgt[p] * abs(r)
                            if acc < best - 1e-12:
                                best = acc; bn = cand
                        if bn != nn[d]:
                            nn[d] = bn; changed = True
                    if not changed:
                        break
                for d in range(n_dates):
                    Phi[d] = phid[d] + 2 * np.pi * nn[d]
                # Which date is unrecoverable? Ask its SHORTEST intervals, which
                # make no assumption about the trajectory: each near partner p
                # independently predicts d's integer from the wrapped difference,
                #     n_d^(p) = round((Phi_p + wrap(phi_d - phi_p) - phi_d)/2pi)
                # and that prediction is right whenever the motion over that
                # interval stays inside a half cycle. A date its neighbours agree
                # on is recoverable; one they disagree on is not, however smooth
                # or unsmooth the ground actually is.
                worst = -1
                worstbad = min_disagree - 1
                for d in range(n_dates):
                    if not okp[d]:
                        continue
                    nc = 0
                    for q in range(n_dates):
                        if q == d or (not okp[q]) or nc >= n_short:
                            continue
                        if abs(tyr[q] - tyr[d]) > short_yr:
                            continue
                        dif = phid[d] - phid[q]
                        dif = dif - 2 * np.pi * np.floor((dif + np.pi) / (2 * np.pi))
                        pred[nc] = np.rint((Phi[q] + dif - phid[d]) / (2 * np.pi))
                        nc += 1
                    if nc < 3:
                        continue
                    # majority vote among the predictions
                    bestcnt = 0
                    for a in range(nc):
                        cnt = 0
                        for b in range(nc):
                            if pred[b] == pred[a]:
                                cnt += 1
                        if cnt > bestcnt:
                            bestcnt = cnt
                    nbad = nc - bestcnt
                    if nbad > worstbad:
                        worstbad = nbad; worst = d
                if worst < 0:
                    break
                if nok <= min_dates:
                    break
                okp[worst] = False
                nok -= 1
            # Re-estimate the trend from the unwrapped series and solve again.
            # The seed rate is a median over lag-1 increments only, so it carries
            # the noise of a single 12-day difference; once the series is
            # unwrapped the slope can be fit over the whole span, which is far
            # tighter. Converges in 2-3 passes. vhat is only overwritten when
            # another pass follows, so the emitted phid/nn/vhat always agree.
            if _tr == n_trend - 1:
                break
            sN = 0.0; sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
            for d in range(n_dates):
                if okp[d]:
                    yv = phid[d] + 2 * np.pi * nn[d] + vhat * tyr[d]
                    sN += 1.0; sx += tyr[d]; sy += yv
                    sxx += tyr[d] * tyr[d]; sxy += tyr[d] * yv
            den = sN * sxx - sx * sx
            if den <= 1e-12:
                break
            vnew = (sN * sxy - sx * sy) / den
            if np.abs(vnew - vhat) < 1e-3:
                break
            vhat = vnew
        # Reject rather than alias. Wrapped phase constrains the rate only
        # modulo 2*pi/dt_min, so every solution has siblings at +-that spacing
        # that fit the observations EXACTLY -- no residual, closure or coherence
        # test can separate them, because they are not worse fits, they are
        # equally perfect ones. The only way a solution is knowable is if all
        # its siblings are implausible, and what makes them implausible is the
        # acquisition plan, not an opinion: dt_min fixes the resolvable range
        # at +-pi/dt_min, and a rate outside it is not something this sampling
        # can represent at all. So the gate is symmetric and needs no user
        # prior -- keep the solution when it lands inside the pixel's own
        # range, emit NaN when it does not. dt_min is taken over the SURVIVING
        # dates, since dropping an acquisition widens a gap and can shrink the
        # resolvable range.
        dtf = 1e30
        prevd = -1
        for d in range(n_dates):
            if okp[d]:
                if prevd >= 0:
                    dt = tyr[d] - tyr[prevd]
                    if dt > 1e-9 and dt < dtf:
                        dtf = dt
                prevd = d
        if dtf > 1e29:
            continue
        spac = 2 * np.pi / dtf
        if np.abs(vhat) >= 0.5 * spac:
            continue
        # Reference the series to its first surviving date, so it starts at 0.
        # Without this the emitted value is the ABSOLUTE per-date phase, whose
        # d=0 term is -phi_0 -- an arbitrary per-pixel constant anywhere in
        # [-pi, pi], i.e. +-13.9 mm at C-band. lstsq_to_dates(cumsum=True)
        # builds a cumulative series from zero, and this is the same series;
        # velocities are unaffected by the constant but the displacement curves
        # are not, so they must agree here too.
        base = 0.0
        for d in range(n_dates):
            if okp[d]:
                base = phid[d] + 2 * np.pi * nn[d] + vhat * tyr[d]
                break
        for d in range(n_dates):
            if okp[d]:
                out[d, px] = -(phid[d] + 2 * np.pi * nn[d] + vhat * tyr[d] - base)
    return out


def unwrap3d_dates_array(scenes, date_values, duration=90.0, max_iter=40,
                         search=2, short_days=40.0, n_short=6,
                         min_disagree=2, min_dates=3, n_trend=3):
    """Unwrapped per-date phase straight from the complex scenes.

    Replaces pairs() -> interferogram() -> unwrap2d() -> lstsq(): the dates are
    the unknowns, so no interferograms are formed and no network inversion is
    needed. min_dates is the floor on surviving dates per pixel: closure is
    structural here, so a resolved triplet is already usable and the default
    is 3 -- there is no statistical sample-size argument for a larger floor. Requires closure-exact input (single-look scenes with per-date
    corrections only), which is what fit1d_network() + predict() produce.
    """
    if isinstance(scenes, list):
        scenes = np.asarray(scenes[0]) if len(scenes) == 1 else np.concatenate(
            [np.asarray(c) for c in scenes], axis=0)
    scenes = np.ascontiguousarray(scenes, dtype=np.complex64)
    n_dates, ny, nx = scenes.shape

    def _as_ns(v):
        v = np.asarray(v)
        if v.dtype.kind == 'M':
            return v.astype('datetime64[ns]').astype(np.int64)
        return v.astype(np.int64)
    tyr = (_as_ns(date_values) / (86400e9 * 365.25)).astype(np.float64)
    tyr = tyr - tyr[0]
    i1, i2 = np.triu_indices(n_dates, k=1)
    band = (tyr[i2] - tyr[i1]) <= float(duration) / 365.25
    i1, i2 = np.ascontiguousarray(i1[band]), np.ascontiguousarray(i2[band])
    wgt = 1.0 / np.maximum(tyr[i2] - tyr[i1], 1e-6)

    flat = scenes.reshape(n_dates, ny * nx)
    amp = np.abs(flat)
    okf = np.ascontiguousarray(amp > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        phif = np.ascontiguousarray(
            np.where(okf, np.angle(flat), 0.0).astype(np.float64))
    out = _unwrap3d_dates_kernel(phif, okf, i1, i2, wgt, tyr, int(max_iter),
                                 int(search),
                                 float(short_days) / 365.25, int(n_short),
                                 int(min_disagree), int(min_dates), int(n_trend))
    return out.reshape(n_dates, ny, nx).astype(np.float32)

