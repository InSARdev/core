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
import numba as _numba
import numpy as np
import threading as _threading
import collections.abc as _abc
import time
import warnings
from .BatchCore import BatchCore
from . import utils_xarray
from .Batch import _Fit3dChain, _Fit3dSlice, _fit3d_keep

# pixels a tile: the loops over a tile's pixels are innermost, so the compiler vectorises them
_TILE = 64
# L1-IRLS: iterations and the residual floor of the weights 1 / |r|
_N_IRLS = 5
_EPS_IRLS = 0.1


@_numba.njit(nogil=True, cache=True)
def _fit1d_kernel(y, g, has_w, An, scale, lam_scale, hc, use_prior, th_out, gam_out, rmse_out):
    """
    The weighted least squares of fit1d() per pixel, float64 inside, over tiles of _TILE pixels: y (n_p, N) the pair
    phases (NaN refuses the pixel: all pairs or nothing), g (n_p, N) the correlation when has_w (the weight is
    min(g, 0.999)^2), An (n_p, k) the design scaled to unit columns, scale its column norms; with use_prior, a pass of
    plain IRLS gives sigma_phi^2, and the height column hc is then shrunk by lambda = sigma_phi^2 * lam_scale in a
    second IRLS pass from the base weights. A pivot that is not positive refuses the pixel. Every iteration is one sweep
    over the pairs: the residual of the previous solution, its weight, and the pair's terms of the normal equations.
    Writes th_out (k, N) unscaled, gam_out = exp(-rmse^2 / 2) and rmse_out (from the base weights, n / (n - k)
    inflated); NaN for a refused pixel.
    """
    n_p, N = y.shape
    k = An.shape[1]
    kk = k * (k + 1) // 2
    Q = np.empty((n_p, kk))
    for p in range(n_p):
        q = 0
        for a in range(k):
            for c in range(a + 1):
                Q[p, q] = An[p, a] * An[p, c]
                q += 1
    T = _TILE
    yv = np.zeros((n_p, T))
    w0 = np.zeros((n_p, T))
    rr = np.zeros(T)
    wv = np.zeros(T)
    wyv = np.zeros(T)
    Mt = np.zeros((kk, T))
    Bt = np.zeros((k, T))
    TH = np.zeros((k, T))
    lam = np.zeros(T)
    s2 = np.zeros(T)
    wsum0 = np.zeros(T)
    okv = np.zeros(T, np.bool_)
    solv = np.zeros(T, np.bool_)
    L = np.empty((k, k))
    z = np.empty(k)
    th = np.empty(k)
    dof = max(n_p - k, 1)
    infl = n_p / dof
    n_it = (2 if use_prior else 1) * _N_IRLS
    for j0 in range(0, N, T):
        nt = min(T, N - j0)
        for t in range(T):
            solv[t] = t < nt
            wsum0[t] = 0.0
            lam[t] = 0.0
            for a in range(k):
                TH[a, t] = 0.0
        for p in range(n_p):
            for t in range(nt):
                v = y[p, j0 + t]
                if v != v:
                    solv[t] = False
                    v = 0.0
                yv[p, t] = v
                if has_w:
                    gv = np.float64(g[p, j0 + t])
                    if not np.isfinite(gv) or gv < 0.0:
                        gv = 0.0
                    elif gv > 0.999:
                        gv = 0.999
                    w0[p, t] = gv * gv
                else:
                    w0[p, t] = 1.0
                wsum0[t] += w0[p, t]
            for t in range(nt, T):
                yv[p, t] = 0.0
                w0[p, t] = 0.0
        for t in range(T):
            solv[t] = solv[t] and wsum0[t] > 0.0
        # a refused pixel runs on zero weights and is written NaN
        for p in range(n_p):
            for t in range(T):
                if not solv[t]:
                    w0[p, t] = 0.0
                    yv[p, t] = 0.0
        have_lam = False
        for i in range(n_it + 1):
            it = i % _N_IRLS
            final = i == n_it
            # a pass starts from the base weights, later iterations reweight by the previous residual
            reweight = it > 0 and not final
            # the prior from the residual of the plain pass, at the first iteration of the second
            upd = use_prior and i == _N_IRLS
            for q in range(kk):
                for t in range(T):
                    Mt[q, t] = 0.0
            for a in range(k):
                for t in range(T):
                    Bt[a, t] = 0.0
            for t in range(T):
                s2[t] = 0.0
            for p in range(n_p):
                if i > 0:
                    for t in range(T):
                        rr[t] = yv[p, t]
                    for a in range(k):
                        ap = An[p, a]
                        for t in range(T):
                            rr[t] -= ap * TH[a, t]
                if final:
                    for t in range(T):
                        s2[t] += w0[p, t] * rr[t] * rr[t]
                    continue
                if upd:
                    for t in range(T):
                        s2[t] += rr[t] * rr[t]
                if reweight:
                    for t in range(T):
                        ar = abs(rr[t])
                        wv[t] = w0[p, t] / (ar if ar > _EPS_IRLS else _EPS_IRLS)
                else:
                    for t in range(T):
                        wv[t] = w0[p, t]
                for t in range(T):
                    wyv[t] = wv[t] * yv[p, t]
                for a in range(k):
                    ap = An[p, a]
                    for t in range(T):
                        Bt[a, t] += ap * wyv[t]
                for q in range(kk):
                    qp = Q[p, q]
                    for t in range(T):
                        Mt[q, t] += wv[t] * qp
            if final:
                break
            if upd:
                for t in range(T):
                    lam[t] = s2[t] / dof * lam_scale
                have_lam = True
            for t in range(T):
                ok = solv[t]
                if ok:
                    for a in range(k):
                        for c in range(a + 1):
                            acc = Mt[a * (a + 1) // 2 + c, t]
                            if have_lam and a == hc and c == hc:
                                acc += lam[t]
                            for qq in range(c):
                                acc -= L[a, qq] * L[c, qq]
                            if a == c:
                                if not acc > 0.0:
                                    ok = False
                                    break
                                L[a, a] = np.sqrt(acc)
                            else:
                                L[a, c] = acc / L[c, c]
                        if not ok:
                            break
                if ok:
                    for a in range(k):
                        acc = Bt[a, t]
                        for qq in range(a):
                            acc -= L[a, qq] * z[qq]
                        z[a] = acc / L[a, a]
                    for a in range(k - 1, -1, -1):
                        acc = z[a]
                        for qq in range(a + 1, k):
                            acc -= L[qq, a] * th[qq]
                        th[a] = acc / L[a, a]
                    for a in range(k):
                        if not np.isfinite(th[a]):
                            ok = False
                okv[t] = ok
                for a in range(k):
                    TH[a, t] = th[a] if ok else 0.0
        for t in range(nt):
            if okv[t]:
                rm = np.sqrt(max(s2[t] / max(wsum0[t], 1e-12), 0.0) * infl)
                rmse_out[j0 + t] = rm
                gam_out[j0 + t] = min(max(np.exp(-0.5 * rm * rm), 0.0), 1.0)
                for a in range(k):
                    th_out[a, j0 + t] = TH[a, t] / scale[a]
            else:
                for a in range(k):
                    th_out[a, j0 + t] = np.nan
                gam_out[j0 + t] = np.nan
                rmse_out[j0 + t] = np.nan


def fit1d_pairs(data, A, weight=None, dh_prior=None, dh_col=None):
    """
    Weighted least squares of a per-pixel model on per-PAIR unwrapped phase (the estimator of Batch.fit1d()).

    The design A (n_pairs, k) is the same for every pixel; its columns are scaled to unit norm for the solve (the rate
    column is O(1) years, the height column O(1e-4)). The weight is the correlation squared, bounded at 0.999 (the
    Cramer-Rao weight gamma^2 / (1 - gamma^2) is unbounded and lets one pair decide a low-correlation pixel). A pixel
    with a NaN pair is refused: all pairs or nothing. Five iterations of L1-IRLS (weights w0 / max(|r|, 0.1)) absorb an
    unwrapping cycle on one pair. With dh_prior (the height's sigma, radians), a plain IRLS pass gives sigma_phi^2 per
    pixel and a second pass shrinks the height column dh_col by lambda = sigma_phi^2 / dh_prior^2. rmse is the base-
    weighted RMS of the residual, n / (n - k) inflated, and gamma = exp(-rmse^2 / 2). Float64 inside, one GIL-free call.

    Parameters
    ----------
    data : np.ndarray
        (n_pairs, ...) float32 unwrapped phase; NaN refuses the pixel.
    A : np.ndarray
        (n_pairs, k) design.
    weight : np.ndarray or None
        Correlation of data's shape, or None for equal weights.
    dh_prior : float or None
        Sigma of the height prior in the height column's units; None for none.
    dh_col : int or None
        The height column (default: the last).

    Returns
    -------
    tuple
        theta (k, N) float64, gamma (N,), rmse (N,); NaN for a refused pixel.

    Examples
    --------
    >>> theta, gamma, rmse = fit1d_pairs(phase, A, weight=corr, dh_prior=2266.0, dh_col=1)
    """
    n_p = data.shape[0]
    y = np.ascontiguousarray(np.asarray(data).reshape(n_p, -1), dtype=np.float32)
    N = y.shape[1]
    Am = np.asarray(A, np.float64)
    k = Am.shape[1]
    scale = np.maximum(np.linalg.norm(Am, axis=0), 1e-30)
    An = np.ascontiguousarray(Am / scale)
    hc = k - 1 if dh_col is None else int(dh_col)
    use_prior = bool(dh_prior) and 0 <= hc < k
    lam_scale = 1.0 / (float(dh_prior) ** 2 * scale[hc] ** 2) if use_prior else 0.0
    has_w = weight is not None
    g = (np.ascontiguousarray(np.asarray(weight).reshape(n_p, -1), dtype=np.float32) if has_w
         else np.empty((0, 0), np.float32))
    if has_w and g.shape != y.shape:
        raise ValueError(f'fit1d: weight block {g.shape} differs from the data block {y.shape}')
    th = np.empty((k, N))
    gam = np.empty(N)
    rmse = np.empty(N)
    _fit1d_kernel(y, g, has_w, An, scale, lam_scale, hc, use_prior, th, gam, rmse)
    return th, gam, rmse


# ----------------------------------------------------------------------------------------------------------------------
# fit1d_network() -- fit3d() until 2026-10-24: the PS network and level 1 on the date stack. Moved here from Batch.py and
# utils_arcs.py with the behaviour of the library's HEAD (and the patches 0030, 0031, 0033); level 2+ dropped.
# ----------------------------------------------------------------------------------------------------------------------

# The scan runs this much wider than the caller's max_dh/max_dv so that their
# range is entirely interior: a solution at 0.99 * max is found on its merits,
# not pinned against a boundary. Solutions landing in the guard band are
# rejected -- their truth is beyond the range and cannot be recovered.
_GUARD = 1.1


# The refinement after the lattice: `_ZOOM_LEVELS` finer lattices around the
# winner, each `_ZOOM_SUB` times finer than the one before and `_ZOOM_BOX`
# cells wide either side of it. sqrt(2) of a cell, because a winner that is
# one cell off in BOTH parameters -- the diagonal, where a fine step in one
# moves the best value of the other -- must still be inside the box. Two
# levels of five leave a cell twenty-five times finer than the coarse one,
# already far below what the noise on a single arc can resolve.
_ZOOM_SUB = 5


_ZOOM_BOX = 1.4


_ZOOM_LEVELS = 2


# A SEEDED fit -- the caller predicts the arc from the network -- is refined
# inside a fixed PHYSICAL tolerance, metres and mm/yr, not a multiple of the
# caller's lattice step: the prediction's error is set by the network's
# measurements, and a caller refining the grid must not tighten who survives.
_SEED_TOL_H = 8.0


_SEED_TOL_V = 4.0


# The most components one solve may return. It is the int8 label's capacity,
# not a preference: past 127 a label folds onto another and two unrelated
# datums read as one. It also bounds the work -- every component costs its own
# kriging pass, so a shattered network would otherwise turn into thousands of
# solves over a single raster.
_MAX_COMPONENTS = 127


# Reweighting passes used to find arcs the network contradicts. The weights
# converge quickly because each pass only has to separate a heavy tail from a
# tight core, not to fit anything.
# A per-node sigma can collapse to zero when a node's few measurements happen
# to land identically, and a zero scale rejects everything else. The floor is
# this fraction of the scene's own scatter -- a scale cannot be finer than the
# measurement is. The SAME fraction on both halves of the solve: the two
# residuals are defined differently, because an arc is one equation in a joint
# system while a partner is a direct estimate, but nothing justifies judging
# them at different resolutions.
# THE FLOOR IS HUBER'S TUNING ON A CALIBRATED SCALE. A robust scale needs a
# lower bound only so a degenerate set cannot drive it to zero, and the MAD is
# already that scale: sigma = 1.4826 * MAD. Huber's constant is 1.345 sigma,
# which is 2 MAD, so the bound is stated in MADs and the constant is the one
# the estimator was designed around rather than a number chosen here.
#
# Set as a small FRACTION of the MAD it did real damage: the IRLS weight is
# `gamma / max(|r|, floor)`, so at a twentieth of a MAD a measurement sitting
# on the running centre outweighed one a MAD away by twenty to one and the
# estimate collapsed onto whichever happened to be nearest. On arcs it rejected
# twice as many as it should, and what it rejected was not outliers.
class _ThreadStats(_abc.MutableMapping):
    """The kernels' stats as PER-THREAD state (`_fit_stats`), so any dask
    shape is a valid one.

    A worker with more than one thread runs several blocks in one PROCESS, and
    a module-level dict is then shared by all of them: whatever the last block
    wrote is what the next one reads. That is a wrong answer whenever the two
    happen to be the same length and a crash when they are not, and neither is
    something a caller can be asked to avoid by choosing a different cluster --
    `threads_per_worker=2` exists so a worker can read while it computes.

    Every mapping operation resolves against the CALLING thread's dict, so the
    existing `_fit_stats[...]` call sites need no change; only the
    whole-dict assignments become `reset()`, since rebinding the attribute
    would swap the proxy out from under the other threads.
    """

    __slots__ = ('_tl',)

    def __init__(self):
        self._tl = _threading.local()

    @property
    def _d(self):
        d = getattr(self._tl, 'd', None)
        if d is None:
            d = {}
            self._tl.d = d
        return d

    def reset(self, mapping=None, **kw):
        d = dict(mapping) if mapping else {}
        d.update(kw)
        self._tl.d = d
        return self

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __delitem__(self, k):
        del self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f'_ThreadStats({self._d!r})'


# DEBUG NUMBERS ARE REDUCED IN THE BLOCK THAT MADE THEM. A level's report is
# debug output; it never needs the values themselves. Shipping them made the
# node table grow with the ARC COUNT, and that table is handed to every later
# level's every block -- so on a large scene every worker received hundreds of
# megabytes of samples to print six numbers from. What travels now is a fixed
# handful of scalars per key: counts and threshold tallies SUM exactly across
# blocks, extremes take the min/max of the blocks' own, and a percentile is
# reported as the RANGE over the blocks rather than pretending a median of
# medians is the median.
# PER-THREAD FROM THE START, so no call site ever meets a plain dict here.
_fit_stats = _ThreadStats()


_LE12 = (('le', 1.0), ('le', 2.0))


def _lvl_stat(a, thr=()):
    """Per-block scalars for the level report: n, extremes, percentiles and
    exact tallies at the thresholds the report prints."""
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    d = dict(n=int(a.size), min=float(a.min()), max=float(a.max()),
             p50=float(np.median(a)), p90=float(np.percentile(a, 90)),
             p99=float(np.percentile(a, 99)))
    for op, x in thr:
        d[f'{op}{x:g}'] = int((a <= x).sum() if op == 'le'
                              else (a >= x).sum() if op == 'ge'
                              else (a > x).sum())
    if thr:
        # THE BOUND TRAVELS WITH THE TALLY. A count over a threshold means
        # nothing without the threshold, and these are the CALLER's err_dh /
        # err_dv -- the report used to print "over 1 m" whatever the caller
        # had set, flagging half the network as bad against a bound nobody
        # asked for.
        d['_thr'] = [[op, float(x)] for op, x in thr]
    return d


def _3d_consensus(consensus):
    """How many agreeing measurements a value must rest on. An int.

    One argument for both halves of the solve, because it is one question asked
    twice: an arc must agree with the network, a partner must agree with the
    other partners.

    IT USED TO CARRY THE OUTLIER BOUND AND THE PASS COUNT AS WELL, as
    `(n, k, i)`. Neither belonged with it. The bound is now stated in physical
    units by `err_dh`/`err_dv`, where a caller can reason about it -- a robust
    sigma is a property of whatever scatter happens to be present, so a pixel
    whose partners were uniformly wrong widened the bar that judged them and
    passed. The pass count is `iterations`, which already governs the other
    iterative refinement in the same solve.

    A COUNT IS REQUIRED. `None` used to mean "ask for none of it", but the
    network stage has no path without something to judge against: it leaves its
    residuals unset and fails hundreds of lines later, inside a dask block.
    Parsed in ONE place and called from the public method as well, so a bad
    value raises where it was written rather than inside a dask block minutes
    later.
    """
    if consensus is None or isinstance(consensus, bool):
        raise ValueError(
            'consensus must be an integer count of agreeing measurements; '
            f'got {consensus!r}')
    if isinstance(consensus, (tuple, list)):
        raise ValueError(
            'consensus is now a plain count; the outlier bound moved to '
            f'err_dh/err_dv and the pass count to iterations. Got {consensus!r}')
    try:
        ma = int(consensus)
    except (TypeError, ValueError):
        raise ValueError(
            f'consensus must be an integer count; got {consensus!r}') from None
    if ma != consensus or ma < 1:
        raise ValueError(
            f'consensus must be a whole number >= 1; got {consensus!r}')
    return ma


def _3d_windows(window):
    """(wy, wx) or (wy, wx, py, px) -> the DS box and the PS extent, validated.

    Two numbers give the DS box and the PS extent follows as THREE TIMES it,
    which is the layout where the nine patches are all equal: the centre box is
    the DS window and the eight around it are the same size again. Four numbers
    set the two independently, because the range over which the atmosphere is
    common is a property of the site and nothing in the data states it. Where
    coherence is poor a caller wants a SMALLER DS box -- fewer, better
    neighbours -- and a much larger PS extent, so that pixels which the short
    test cannot certify are still reachable at range: (24, 96, 256, 1024) is
    a 192 x 192 m DS box inside a 2048 x 2048 m PS search.

    Both are FULL extents of a box centred on the pixel, like `window`
    everywhere else here, so the PS extent must exceed the DS box on both axes
    -- otherwise the ring between them is empty and there is nowhere to look.
    """
    w = tuple(int(v) for v in window)
    if len(w) == 2:
        w = w + (3 * w[0], 3 * w[1])
    if len(w) != 4:
        raise ValueError(
            f'window takes 2 values (wy, wx) -- the PS extent is then 3x it -- '
            f'or 4 (wy, wx, ps_y, ps_x); got {len(w)}: {window}')
    wy, wx, py, px = w
    if wy < 2 or wx < 2:
        raise ValueError(f'the DS window must be at least 2 pixels per axis, '
                         f'got ({wy}, {wx})')
    if py < wy + 2 or px < wx + 2:
        raise ValueError(
            f'the PS extent ({py}, {px}) must exceed the DS window ({wy}, {wx}) '
            f'by at least 2 pixels on each axis: the PS partners come from the '
            f'ring between them, and this one is empty')
    return wy, wx, py, px


def _3d_arc_product(A, B):
    """The arc `A * conj(B)` from explicit float32 parts, complex64 out.

    numpy's complex multiply takes a different arithmetic path below a few
    hundred columns, so the SAME pair formed in a small batch and in a large
    one differs in its last bits. A continuous refinement never noticed; a
    lattice can answer a near-tie one cell apart on the two, and an arc's
    fit then depended on how many arcs were formed alongside it. Four plain
    float32 products and two sums are the same IEEE operations at any size,
    so the bytes -- and the fit -- are a property of the pair alone.
    """
    out = np.empty(np.broadcast_shapes(A.shape, B.shape), np.complex64)
    ar, ai, br, bi = A.real, A.imag, B.real, B.imag
    np.add(ar * br, ai * bi, out=out.real)
    np.subtract(ai * br, ar * bi, out=out.imag)
    return out


def _3d_lattice_response(hh, tt, step_h, step_v):
    """The least a lattice of steps (step_h, step_v) can read of a peak.

    A peak lies at most half a step from the nearest lattice point in each
    parameter, and the coherence read there is the peak's times the stack's
    response at that offset, |mean_d exp(i (hh_d dh + tt_d dv))| -- lowest on
    a corner of the cell. Both corners are read: height and rate trade against
    each other along one diagonal and not the other. Radians in, as the
    lattice's own steps; a zero `hh` (no baseline) leaves the rate alone.
    """
    hh = np.asarray(hh, dtype=np.float64)
    tt = np.asarray(tt, dtype=np.float64)
    return float(min(abs(np.mean(np.exp(1j * (hh * (step_h / 2.0)
                                              + s * tt * (step_v / 2.0)))))
                     for s in (1.0, -1.0)))


def _3d_arc_fit(arc, ele2phase, t, meter2rad, max_dh=25.0, max_dv=25.0,
                step_dh=8.0, step_dv=2.0, budget=None, max_seasonal=0.0,
                iterations=8, seed_th=None, threshold=None):
    """Joint (height, velocity) fit on many arcs at once, WITHOUT priors.

    arc     : (n_dates, n_arcs) COMPLEX arc, u_i * conj(u_j). Phase never
              leaves the complex plane here: taking np.angle only to feed
              np.exp(1j.) straight back is a round trip that buys nothing and
              invites the wrapping bugs it looks like it is avoiding.
    ele2phase    : (n_dates,)  height-to-phase factor, B_perp / (R sin theta)
    t       : (n_dates,)  time in years from the reference epoch
    meter2rad    : float       4 pi / wavelength
    max_dh  : metres,  half-width of the height search. These are DIFFERENTIAL
              heights between neighbours a few tens of metres apart, so 200 m
              is already generous.
    max_dv  : mm/yr,   half-width of the rate search, likewise differential.
    step_dh : metres,  lattice step. It sets which BASIN is found, not the
              accuracy: the refinement below is continuous and absorbs the
              quantisation over a wide range of steps.
    step_dv : mm/yr,   lattice step in rate.
    iterations : 0 stops at the lattice argmax -- enough to RANK candidates,
              which is what the shortlist stages ask for. Any positive value
              runs the zoom below; the count itself no longer means anything.
    max_seasonal : must be 0. The annual term is not fitted by this kernel and
              a positive bound raises ValueError -- see the end of this note.
    seed_th : (n_arcs, 2) prior solutions in radians. The lattice is skipped
              and the zoom runs around the seed inside a fixed physical
              tolerance -- see UNRESOLVABLE ARCS.
    threshold : the coherence the caller keeps an arc at. Only arcs whose
              LATTICE coherence could still reach it are zoomed: threshold
              times the stack's response half a lattice step off
              (`_3d_lattice_response`), the most a lattice point can under-read
              a peak. The others return their lattice values, which are below
              it, and every arc that can clear it gets the zoom it always got.
              None zooms every arc; a seeded fit has no lattice and zooms
              every arc.

    Two stages, because neither alone is both correct and affordable.

    LATTICE. gamma = |sum_d z_d exp(-i theta.u_d)| / n_valid over a grid, as
    one (arcs x candidates) product. The grid is built as arange(-k, k+1)*step
    so it always CONTAINS ZERO: a grid that misses the origin biases every
    solution by half a cell, and at +-200 m with a 16 m step -- 200/16 = 12.5
    -- that took the largest connected component from 862 nodes to 63.

    REFINEMENT. A ZOOM: finer lattices around the winner, each five times
    finer than the last and 1.4 cells wide either side of it, every level on
    every date and the same objective. The coarse lattice only has to land in
    the right basin -- its cell is a fraction of the main lobe in both
    parameters, which is what the lobe widths above the steps guarantee --
    and the zoom then resolves the peak to a small fraction of a cell, below
    what the noise on one arc can resolve. Nothing iterates, nothing is
    inverted and nothing has to converge: the same input gives the same
    answer on any machine, and a solution can never drift out of the basin
    it was found in, because every box is centred on the last winner.

    WHY NOT A GRID ALONE. A grid argmax is discontinuous in the data: perturb
    every phase slightly and a small fraction of arcs jump a FULL cell while
    the median does not move at all -- the instability is rare, large, and
    invisible in any summary statistic. After the zoom the jump is a
    twenty-fifth of a cell and the answer moves as the data does.

    UNRESOLVABLE ARCS RETURN NaN. Two conditions.

      edge   the solution sits outside max_dh/max_dv, i.e. in the guard band
             the scan adds beyond them, so the truth is past the range the
             caller asked for and the peak is a boundary, not a maximum
      seed   a SEEDED fit whose first zoom lands on the tolerance box: the
             arc's own optimum lies beyond what the prediction is trusted
             to, so it is a different solution, not a refined one. A LATTICE
             winner on its box is not gated -- the coarse argmax was a cell
             off, the next level re-centres on the boundary point and the
             answer is unchanged.

    Height and rate are always solved TOGETHER: the perpendicular baseline is
    not a smooth function of time, so they separate only jointly -- chained, a
    planted (+20 m, -12.7 mm/yr) comes back as (+370 m, -0.65).

    Invalid samples are zeros and stay zeros, so they drop out of every sum and
    the normalisation counts only what is actually there. Going through angles
    lost this: np.angle(0) is 0 and np.exp(1j.0) is 1, which silently turned
    every masked date into a perfectly coherent observation.

    Returns (gamma, height_rad, velocity_rad_yr, seasonal_rad), each (n_arcs,).
    Height is radians per unit ele2phase and rate is radians per year: the
    library works in phase throughout and only displacement_los()
    converts to a length. `max_dh`/`max_dv` remain physical, since they
    state what the caller wants bounded.

    `seasonal_rad` is zeros, complex: the model holds no annual term, which
    is a value and not an absence. `max_seasonal > 0` raises ValueError. The
    annual was fitted here once, as one complex amplitude against the yearly
    carrier with its own lattice of sidebands; on the arcs this kernel serves
    it read as noise, and the freedom to fit it cost the rate more than the
    term returned. It comes back when a stack with a real annual signal
    exists to design it against.
    """
    arc = np.asarray(arc)
    if not np.iscomplexobj(arc):
        raise TypeError('arc must be the COMPLEX arc u_i * conj(u_j), '
                        f'got {arc.dtype}; passing np.angle() of it is the '
                        'round trip this signature exists to prevent')
    if not (step_dh > 0 and step_dv > 0 and max_dh > 0 and max_dv > 0):
        raise ValueError('max_dh, max_dv, step_dh, step_dv must all be > 0, '
                         f'got {max_dh}, {max_dv}, {step_dh}, {step_dv}')
    if max_seasonal and max_seasonal > 0:
        raise ValueError(
            f'max_seasonal={max_seasonal}: the annual term is not supported by '
            'the arc kernel; pass max_seasonal=0. It returns when a stack with '
            'a real annual signal exists to design it against.')
    n, m = arc.shape

    # SCAN 10% WIDER than the caller asked for, and reject what lands outside
    # THEIR range. max_dv=100 then means exactly what it says: 99 mm/yr is
    # detected, because it sits inside the scan with headroom either side and
    # is nowhere near the boundary where a peak is the edge rather than a
    # maximum. The guard band is what makes the promise exact, and it stays an
    # implementation detail -- the caller never reasons about it.
    #
    # arange(-k, k+1) * step is symmetric about zero BY CONSTRUCTION, so the
    # no-model solution is always a candidate and no solution is biased by a
    # half cell. See the docstring: getting this wrong is catastrophic and
    # entirely silent.
    # the epsilon is not cosmetic: 1.1 * 200 is 220.00000000000003 in binary
    # floating point, so a bare ceil() adds a whole cell at each end whenever
    # the guard lands exactly on the lattice
    # ele2phase=None means the series carries no usable baseline, so the height
    # term is NOT estimated: its grid collapses to {0} and dh comes back NaN.
    # Passing zeros instead would leave dh free to take any value at no cost,
    # and the reported height would be a random number rather than an absence.
    no_h = ele2phase is None
    # PHASE THROUGHOUT, CONVERTED ONCE AT THE DOOR. `max_dh` and `step_dh` are
    # stated in metres and `max_dv`/`step_dv` in mm/yr because that is what a
    # caller can reason about, but everything inside is radians, as everywhere
    # else in the library. Scaling a bound is exact, so `|dh| > max_dh` and
    # `|dh_rad| > max_dh * meter2rad` are the same gate -- and the constants
    # come out cleaner in phase.
    #
    # The alternative, converting at the RETURN, put a unit boundary in the
    # middle of the function: `dh` meant metres above it and radians below,
    # and a seeded caller handing back a value it had just been given was
    # wrong by meter2rad with nothing to catch it.
    _m2h = float(meter2rad)                     # rad per metre of dh
    _m2v = float(meter2rad) * 1e-3              # rad/yr per mm/yr of dv
    max_dh_r, step_dh_r = float(max_dh) * _m2h, float(step_dh) * _m2h
    max_dv_r, step_dv_r = float(max_dv) * _m2v, float(step_dv) * _m2v
    kh = 0 if no_h else int(np.ceil(_GUARD * max_dh_r / step_dh_r - 1e-9))
    kv = int(np.ceil(_GUARD * max_dv_r / step_dv_r - 1e-9))
    gh = np.arange(-kh, kh + 1) * step_dh_r
    gv = np.arange(-kv, kv + 1) * step_dv_r
    # The origin is the no-model solution. It being a candidate is what makes
    # "the fit is never worse than not fitting" true, and the refinement is
    # monotone FROM the lattice argmax, so losing it costs that guarantee as
    # well as the half cell. When a grid misses zero the network collapses to a
    # fraction of its nodes with no error raised anywhere, which is why this is
    # asserted rather than trusted.
    assert (gh == 0).any() and (gv == 0).any(), (
        f'search grid lost the origin: dh {gh[0]}..{gh[-1]} step {step_dh}, '
        f'dv {gv[0]}..{gv[-1]} step {step_dv}')
    ncand = gh.size * gv.size

    # The (arcs x candidates) product is the peak by a wide margin; the
    # candidate bank and the (dates x arcs) working copies are small beside
    # it. Arcs are independent, so splitting them changes nothing but memory.
    step = max(1, int(_3d_budget_mb(budget) * 1e6 / max(ncand * 8, 1)))
    if m > step:
        gs = np.empty(m, np.float32)
        hs = np.empty(m)
        vs = np.empty(m)
        ss = np.empty(m, np.complex128)
        for s0 in range(0, m, step):
            sl = slice(s0, min(s0 + step, m))
            gs[sl], hs[sl], vs[sl], ss[sl] = _3d_arc_fit(
                arc[:, sl], ele2phase, t, meter2rad, max_dh, max_dv,
                step_dh, step_dv, budget, max_seasonal,
                iterations,
                seed_th=None if seed_th is None else seed_th[sl],
                threshold=threshold)
        return gs, hs, vs, ss

    A = np.abs(arc)
    Z = np.where(A > 0, arc / np.where(A > 0, A, 1.0), 0).astype(np.complex64)
    nv = (A > 0).sum(axis=0)
    del A
    # ONE ORIENTATION PER ARC, CHOSEN BY THE ARC. An arc and its reverse are
    # the same measurement, and a continuous refinement returned exactly
    # negated answers for them; a lattice does not quite: between two
    # candidates the data cannot tell apart it takes the first in the bank's
    # order, and that is not the mirror of the first among the mirrored
    # candidates. So every arc is fitted in the orientation that makes the
    # sum of its imaginary parts non-negative -- a sum the reverse arc
    # negates exactly -- and the answer is negated back. fit(reverse) is
    # then -fit(arc) to the bit, whichever way a tree, a tile or a batch
    # handed the arc over, and no caller has to know.
    _rev = Z.imag.sum(axis=0) < 0
    if _rev.any():
        Z[:, _rev] = np.conj(Z[:, _rev])
    # the model phase is dh_rad * ele2phase_t + dv_rad * t_t, so with the
    # parameters in phase the design columns are the geometry itself
    tt = np.asarray(t, dtype=np.float64)
    hh = (np.zeros_like(tt) if no_h
          else np.asarray(ele2phase, dtype=np.float64))

    # ---- stage 1: lattice, one product ---------------------------------
    P = np.stack(np.meshgrid(gh, gv, indexing='ij'), -1).reshape(-1, 2)
    # ORDERED BY DISTANCE FROM (0, 0), THE ORIGIN FIRST, as fit1d's rate grid
    # is: an argmax returns the FIRST of equal scores, so a tie between
    # candidates that explain an arc equally well resolves to the smallest
    # model rather than to whichever corner the row-major grid happened to
    # start at. Rings of the lattice (Chebyshev distance in steps), then the
    # Manhattan distance inside a ring; stable, so the order is deterministic.
    # Nothing downstream reads the grid's shape -- every use is P[k].
    _ring = np.maximum(np.abs(P[:, 0]) / max(step_dh_r, 1e-30),
                       np.abs(P[:, 1]) / max(step_dv_r, 1e-30))
    _diag = (np.abs(P[:, 0]) / max(step_dh_r, 1e-30)
             + np.abs(P[:, 1]) / max(step_dv_r, 1e-30))
    P = P[np.lexsort((_diag, np.round(_ring, 6)))]
    C = np.exp(-1j * (np.outer(hh, P[:, 0])
                      + np.outer(tt, P[:, 1]))).astype(np.complex64)
    # no division by nv here: it is constant per arc, so it cannot move the
    # argmax, and gamma is computed once at the end from the refined model
    # THE PRODUCT IS BATCHED BY ITS OWN OUTPUT, not by the arcs going in. The
    # callers size their batches so the ARC array fits `budget`, but this
    # allocates (arcs x lattice), which is thousands of times larger -- so the
    # peak here is nothing like what `budget` accounted for, and it is batched
    # down to what `budget` allows.
    # A SEED SKIPS THE LATTICE ENTIRELY. The caller already knows where this
    # arc's optimum is -- from the network, which has solved both ends onto one
    # datum -- so the search that finds the basin has nothing left to find. The
    # refinement below still runs, because the seed is a prediction and the arc
    # is entitled to move within its own cell. What is dropped is the (arcs x
    # candidates) product, which is the whole cost.
    _mb = _3d_budget_mb(budget)
    if seed_th is not None:
        # SEEDED: the caller already knows where this arc's optimum is, because
        # the network has solved both ends onto one datum. The search that
        # finds the basin has nothing left to find, so the (arcs x candidates)
        # product -- the whole cost -- is skipped. The zoom below still runs:
        # a seed is a prediction, and the arc is entitled to move within the
        # tolerance a prediction is trusted to, and no further.
        _S = np.asarray(seed_th, dtype=np.float64).reshape(m, -1)
        TH = np.zeros((m, 2))
        TH[:, 1] = _S[:, -1]
        if not no_h and _S.shape[1] > 1:
            TH[:, 0] = _S[:, 0]
        TH[_rev] = -TH[_rev]          # the seed follows the arc's orientation
        del C
    else:
        _L = C.shape[1]
        _blk = max(1, int(_mb * 1024 * 1024 // max(_L * 8, 1)))
        k = np.empty(m, dtype=np.int64)
        _amax = np.empty(m, dtype=np.float32)    # the lattice coherence
        for _b0 in range(0, m, _blk):
            _sl = slice(_b0, min(_b0 + _blk, m))
            _A = np.abs(Z[:, _sl].T @ C)
            k[_sl] = _3d_argmax_first(_A)
            _amax[_sl] = _A[np.arange(_A.shape[0]), k[_sl]]
            del _A
        del C
        # (m, 2) whatever the design: without a baseline the height column
        # is identically zero and the zoom never moves it
        TH = P[k].astype(np.float64)

    # ---- stage 2: zoom ----------------------------------------------------
    # See the module constants: `_ZOOM_LEVELS` lattices around the winner,
    # each `_ZOOM_SUB` times finer, `_ZOOM_BOX` cells either side. Every
    # level divides the winner so far out of the data, so ONE bank of fine
    # offsets serves every arc and the level is a single product, exactly as
    # the coarse stage is.
    seed_edge = np.zeros(m, dtype=bool)
    if iterations <= 0:
        # NO REFINEMENT: the lattice argmax IS the answer. Its value already
        # orders candidates the way the refined one does -- close enough to
        # choose WHICH partners are worth refining; the chosen few are then
        # refined normally. Same code, same model, stopped one stage early.
        R = _3d_rotate(Z, np.outer(hh, TH[:, 0]) + np.outer(tt, TH[:, 1]))
        gam = (np.abs(R.sum(axis=0)) / np.maximum(nv, 1)).astype(np.float32)
        del R
    else:
        if seed_th is None:
            bh, bv = _ZOOM_BOX * step_dh_r, _ZOOM_BOX * step_dv_r
        else:
            bh, bv = _SEED_TOL_H * _m2h, _SEED_TOL_V * _m2v
        _nz = int(round(_ZOOM_BOX * _ZOOM_SUB))   # sub-cells either side
        if threshold is not None and seed_th is None:
            # ONLY THE ARCS THE CALLER CAN KEEP ARE ZOOMED (see `threshold`):
            # below threshold x the lattice's response half a step off, an arc
            # cannot reach the threshold wherever its peak sits in the cell
            gam = (_amax / np.maximum(nv, 1)).astype(np.float32)
            _zs = np.flatnonzero(gam >= threshold * _3d_lattice_response(
                hh, tt, step_dh_r, step_dv_r))
        else:
            gam = np.empty(m, dtype=np.float32)
            _zs = np.arange(m)
        for _lv in range(_ZOOM_LEVELS):
            ch, cv = bh / _nz, bv / _nz            # this level's cell
            # THE GRID IS GLOBAL, NOT THE SEED'S. The box is centred on the
            # nearest point of a lattice with this level's cell anchored at
            # the origin, so two runs that arrive with slightly different
            # seeds -- the network solved from batches in another order --
            # search the same points and return the same answer. A lattice
            # winner already sits on that grid; a predicted seed does not,
            # and moving it by up to half a cell costs nothing the box does
            # not have.
            TH[_zs, 0] = np.round(TH[_zs, 0] / ch) * ch if not no_h else 0.0
            TH[_zs, 1] = np.round(TH[_zs, 1] / cv) * cv
            fh = np.zeros(1) if no_h else np.arange(-_nz, _nz + 1) * ch
            fv = np.arange(-_nz, _nz + 1) * cv
            F = np.stack(np.meshgrid(fh, fv, indexing='ij'), -1).reshape(-1, 2)
            # ORIGIN FIRST, as the coarse lattice: a tie resolves to the
            # smallest move, never to a corner of the box
            _rh = np.abs(F[:, 0]) / max(ch, 1e-30)
            _rv = np.abs(F[:, 1]) / max(cv, 1e-30)
            _o = np.lexsort((_rh + _rv, np.round(np.maximum(_rh, _rv), 6)))
            F = F[_o]
            _ring = np.maximum(_rh, _rv)[_o] >= _nz - 1e-9   # on the box
            Cf = np.exp(-1j * (np.outer(hh, F[:, 0])
                               + np.outer(tt, F[:, 1]))).astype(np.complex64)
            # the product AND the divided-out copy of the block fit the budget
            _blk = max(1, int(_mb * 1024 * 1024
                              // max((Cf.shape[1] + n) * 8, 1)))
            for _b0 in range(0, len(_zs), _blk):
                _sl = _zs[_b0:_b0 + _blk]
                Zr = _3d_rotate(Z[:, _sl], np.outer(hh, TH[_sl, 0])
                                + np.outer(tt, TH[_sl, 1]))
                A_ = np.abs(Zr.T @ Cf)
                kf = _3d_argmax_first(A_)
                TH[_sl] += F[kf]
                gam[_sl] = A_[np.arange(len(kf)), kf] / np.maximum(nv[_sl], 1)
                if _lv == 0 and seed_th is not None:
                    seed_edge[_sl] = _ring[kf]
                del Zr, A_
            del Cf
            bh, bv = _ZOOM_BOX * ch, _ZOOM_BOX * cv   # the next box, on this cell

    TH[_rev] = -TH[_rev]              # back to the caller's orientation
    dh = np.full(m, np.nan) if no_h else TH[:, 0]
    dv = TH[:, 1]
    # An arc we cannot resolve returns NaN, never a plausible number: a wrong
    # value that clears the threshold is invisible to everything downstream,
    # while a NaN is simply not an arc. The two conditions are documented
    # above (UNRESOLVABLE ARCS): outside the caller's bounds, or a seeded fit
    # whose first zoom sits on the tolerance box.
    edge = np.abs(dv) > max_dv_r
    if not no_h:
        edge = edge | (np.abs(dh) > max_dh_r)
    bad = (nv < 1) | edge | seed_edge
    gam = np.where(bad, np.nan, gam).astype(np.float32)
    # no annual in the model is a value and not an absence: zero, where NaN
    # would claim it could not be assessed
    seas = np.zeros(m, dtype=np.complex128)
    # RADIANS OUT. The gates above run in the units the ARGUMENTS are stated in
    # -- max_dh in metres, max_dv in mm/yr -- because that is what the caller
    # asked to bound. Everything downstream works in phase, so the conversion
    # happens once, here, and nothing converts again: velocity() returns this
    # value untouched and displacement_los() is the single place a length is
    # produced.
    return (gam,
            np.where(bad, np.nan, dh),                 # rad per unit ele2phase
            np.where(bad, np.nan, dv),                 # rad/yr
            np.where(bad, np.nan, seas))               # rad, complex


def _3d_budget_mb(budget):
    """Working-set budget in MB; None reads the dask chunk size.

    The project sizes every transient against `array.chunk-size` so one
    setting governs the whole pipeline on a given machine. A hardcoded default
    -- this returned 1024 MB regardless -- silently ignored that, and on a
    machine configured for small chunks it would allocate eight times what the
    caller asked for.
    """
    if budget is not None:
        return float(budget)
    from .utils_dask import get_dask_chunk_size_mb
    return float(get_dask_chunk_size_mb())


def _3d_arc_batch(Us, Ut, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
                  step_dh, step_dv, budget, iterations, seed_th=None,
                  threads=1, threshold=None):
    """Fit every (src, tgt) arc between two phasor sets, in budgeted batches.

    NO DIFFERENTIAL ANNUAL. Both callers attach a pixel to a neighbour tens of
    metres away, and there is no seasonal GRADIENT at that scale: stratified
    and thermal delay vary with elevation and over kilometres, not across a
    courtyard. Fitting one is fitting noise, and two free parameters on a
    marginal arc buy enough coherence to carry the rate a whole sideband away.

    threads : arcs are independent, so slices of them fit concurrently; the
    GEMMs and ufuncs release the GIL. Only a caller that OWNS the host may
    raise it -- the fit1d_network gate does, a per-chunk dask task must not.

    threshold : the caller's arc threshold, for `_3d_arc_fit`: only arcs whose
    lattice coherence could still reach it are refined.
    """
    n = Us.shape[0]
    ga = np.empty(len(src), np.float32)
    dha = np.empty(len(src)); dva = np.empty(len(src))
    dsa = np.empty(len(src), np.complex128)
    _th = max(1, int(threads))
    if _th > 1 and len(src) > _th:
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        bnd = np.linspace(0, len(src), _th * 4 + 1).astype(np.int64)

        def _slice(i):
            sl = slice(bnd[i], bnd[i + 1])
            ga[sl], dha[sl], dva[sl], dsa[sl] = _3d_arc_batch(
                Us, Ut, src[sl], tgt[sl], ele2phase, t, meter2rad, max_dh,
                max_dv, step_dh, step_dv, _mb, iterations,
                seed_th=None if seed_th is None else seed_th[sl],
                threshold=threshold)
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_slice, range(_th * 4)))
        return ga, dha, dva, dsa
    # arcs within ONE set are formed lower column first, as _3d_ps_network
    # forms them, so the same arc has the same bytes whoever enumerated it
    src = np.asarray(src); tgt = np.asarray(tgt)
    if Us is Ut:
        _neg = src > tgt
        src, tgt = np.minimum(src, tgt), np.maximum(src, tgt)
        if seed_th is not None:
            seed_th = np.where(_neg[:, None], -np.asarray(seed_th), seed_th)
    else:
        _neg = np.zeros(len(src), dtype=bool)
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 16, 1)))
    for b0 in range(0, len(src), step):
        sl = slice(b0, min(b0 + step, len(src)))
        arc = _3d_arc_product(Us[:, src[sl]], Ut[:, tgt[sl]])
        ga[sl], dha[sl], dva[sl], dsa[sl] = _3d_arc_fit(
            arc, ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
            budget, 0.0, iterations=iterations,
            seed_th=None if seed_th is None else seed_th[sl],
            threshold=threshold)
    if _neg.any():
        dha[_neg] = -dha[_neg]
        dva[_neg] = -dva[_neg]
    return ga, dha, dva, dsa


def _3d_ds_solve(n_ds, ei, ep, e_dv, e_dh, e_g, ps_vel, ps_hgt, err_v,
                 err_h, passes):
    """Rate and height for every DS, each from its own equations.

        DS_i - PS_p = dv_ip     PS FIXED, so this pins the DS to the datum

    Returns (vel, hgt, n_anchor) -- the values and, per DS, how many of its
    equations came through the gate.

    ONLY DS-TO-PS ARCS CARRY THE SOLVE, at every level. The PS are the
    certified layer and the datum; a DS value inherits noise, so nothing is
    ever solved against one, and two DS are never tied to each other. Peer
    equations between the DS of one solve were tried and measured inert --
    one per several pixels, no change to the product -- and were removed
    together with the search that produced them. With no coupling the system
    is one weighted fit per pixel, solved in closed form.

    REJECT, RE-SOLVE, RE-CHECK, WITH THE SCALE HELD -- as the node network
    does. Reweighting alone leaves an outlier pulling and its error spread
    over the equations it contradicts; only removal stops that, and the scale
    must not be re-estimated from the survivors or the gate feeds on itself
    and erodes the set instead of settling.

    RATE AND HEIGHT ARE JUDGED TOGETHER, also as the node network does. Both
    are solved every pass and an arc answers for the worse of its two
    residuals, each against its own scale. Judged on the rate alone, an arc
    that is metres out in height -- a facade, a roof edge, two scatterers at
    different elevations inside one window -- passes the gate and its height
    enters the answer unchallenged, so the heights could not be trusted even
    where the rates were sound.
    """
    # published per thread for the debug readers; nothing iterates here
    _3d_ds_solve._tl.conv = []
    ei = np.asarray(ei, dtype=np.int64)
    m1 = len(ei)
    if m1 == 0:
        return (np.full(n_ds, np.nan), np.full(n_ds, np.nan),
                np.zeros(n_ds, np.int64))
    w0 = np.asarray(e_g, float)
    rhs_v = ps_vel[np.asarray(ep)] + np.asarray(e_dv, float)
    rhs_h = ps_hgt[np.asarray(ep)] + np.asarray(e_dh, float)

    def _both(w_, live):
        """The weighted least-squares value of every pixel from its live
        equations: with one unknown per row the normal equations are
        diagonal, so this is the exact solution, not an iteration."""
        ww = np.where(live, np.maximum(w_, 0.0), 0.0)
        sw = np.bincount(ei, weights=ww, minlength=n_ds)
        d = np.where(sw > 0, sw, np.nan)
        return (np.bincount(ei, weights=ww * rhs_v, minlength=n_ds) / d,
                np.bincount(ei, weights=ww * rhs_h, minlength=n_ds) / d)

    # BOTH RESIDUALS DECIDE, as the node network's gate does. An arc carries a
    # differential rate AND a differential height, and the two fail
    # independently: a facade or a roof edge can be metres out in height while
    # its rate looks ordinary, and judged on the rate alone that arc is kept
    # and its height goes into the answer unchallenged. Each residual is
    # scored against its own scale, since a metre and a millimetre per year
    # are not comparable numbers, and the worse of the two is what the arc is
    # judged by. THE SCALE IS THE BOUND THE CALLER STATED, not one read off
    # the residuals: a set that is uniformly wrong produces a wide robust
    # sigma and passes itself. Same rule as the network and the vote.
    def _z(xv, xh):
        return np.maximum(np.abs(xv[ei] - rhs_v) / err_v,
                          np.abs(xh[ei] - rhs_h) / err_h)

    live = np.ones(m1, dtype=bool)
    w = w0.copy()
    xv, xh = _both(w, live)
    for _ in range(max(1, int(passes))):
        w = w0 / np.maximum(_z(xv, xh), 1.0)
        xv, xh = _both(w, live)
    for _ in range(max(1, int(passes))):
        xv, xh = _both(w, live)
        keep = live & (_z(xv, xh) <= 1.0)
        if keep.sum() == live.sum() or keep.sum() < 2:
            break
        live = keep
    vel, hgt = _both(w, live)
    nanch = np.bincount(ei[live], minlength=n_ds)
    return vel, hgt, nanch


_3d_ds_solve._tl = _threading.local()


def _3d_lap(t0):
    """Seconds since `t0`, and the mark for the next stage.

    Stage timings ride the debug stream because the work runs inside dask
    tasks on other processes, where a profiler attached to the caller sees
    nothing. Returned as a pair so a caller can print one stage and start the
    next from the same instant, leaving no gap between them.
    """
    _t = time.monotonic()
    return _t - t0, _t


def _3d_argmax_first(A, rtol=1e-6):
    """The first column within `rtol` of each row's maximum.

    A plain argmax picks between two candidates that score within the last
    bit of each other by whichever a GEMM of this particular shape rounded
    up, so a result could depend on how the arcs were batched. The banks are
    ordered origin-first, so taking the FIRST candidate inside a tolerance
    that sits above float32 rounding makes the pick a property of the data:
    the smallest model among those the data cannot tell apart, whatever the
    batch. Values inside the tolerance differ by less than any noise on a
    single arc, so nothing of the answer is spent on this.
    """
    mx = A.max(axis=1, keepdims=True)
    return np.argmax(A >= mx * (1.0 - rtol), axis=1)


def _3d_rotate(Z, X):
    """`Z * exp(-i X)` in complex64, without the complex128 detour.

    `np.exp(-1j * X)` evaluates the transcendental in float64 and builds a
    complex128 array at sixteen bytes an element, which is then copied down.
    cos and sin written straight into the halves of a complex64 result skip
    both, and the reduction below is what makes the narrower evaluation safe.
    """
    # REDUCED BEFORE IT IS NARROWED. cos and sin are accurate in float32 only
    # for a small argument, and these phases run to hundreds of radians, where
    # a float32 argument costs three decimal digits. Folding into [-pi, pi] in
    # float64 first keeps the error at the float32 floor whatever the phase.
    x = np.asarray(X, np.float64)
    x = (x - (2 * np.pi) * np.round(x / (2 * np.pi))).astype(np.float32)
    e = np.empty(x.shape, np.complex64)
    np.cos(x, out=e.real)
    np.sin(x, out=e.imag)
    np.negative(e.imag, out=e.imag)
    e *= Z
    return e


def _3d_shortlist_ds_ps(Us, Ut, base_lab, nsrc, ele2phase, t, meter2rad,
                        max_dh, max_dv, step_dh, step_dv, budget,
                        iterations, min_agreeing, threshold,
                        stats=None, prefix='ds_', debug=False,
                        fix_h=None, fix_v=None, threads=1, pos=None):
    """DS to PS inside the HALF-WINDOW BOX: every arc is fitted, and only each
    candidate's best `_ARC_CAP` leave.

    THE NEIGHBOURHOOD IS THE DS WINDOW CENTRED ON THE CANDIDATE: the nodes
    within `wy // 2` rows and `wx // 2` columns of it, the one rule every
    partner search in the library draws -- the level-2 search, the cascade's
    own exclusion, the node network's box. It is a distance, so it is the
    same for every pixel wherever it sits and whatever the chunking. The
    window LATTICE it replaced -- a candidate took the nodes of its own
    window and the eight around it -- reached one to two windows depending
    on where the pixel sat in its window, and the lattice was cut from the
    block's corner, so the partner set changed with the chunking.

    FITTED IN CANDIDATE RANGES, RANKED AND CUT INSIDE EACH. A range of
    candidates -- as many as `budget` holds arcs for -- is built, fitted,
    ranked by FITTED coherence, and cut to `_ARC_CAP` per candidate before the
    next range exists. Nothing of graph length is ever materialised, and what
    leaves is at most `_ARC_CAP` arcs per candidate.

    NOTHING IS RANKED BEFORE THE FIT. A raw score is a coherence, so it decays
    with the arc's own height at `ele2phase * meter2rad` radians per metre: a
    tall scatterer's true partner scores like noise and ranks last, which is
    exactly where the height is the thing being measured. The ranking here is
    by the coherence the lattice reaches once it has found that height, which
    is the ordering the consensus and the solve use anyway.

    ONE COMPONENT PER CANDIDATE. Components carry their own free datum, so a
    candidate holding partners from two of them measures the offset between
    the datums rather than its own value. The best arc names the component,
    and only that component's arcs are ranked and kept.

    Returns (ksrc, ktgt, ga, dha, dva, dsa, good), aligned, grouped by
    candidate with the best arc first. Every arc that leaves clears
    `threshold` and belongs to its candidate's component, so `good` is True
    throughout; it is returned because the callers' contract names it.
    """
    if pos is None:
        raise ValueError('_3d_shortlist_ds_ps needs `pos` to name the window')
    dy_, dx_, ny_, nx_, _ry, _rx, wy, wx = pos
    dy_, dx_ = np.asarray(dy_, np.int64), np.asarray(dx_, np.int64)
    ny_, nx_ = np.asarray(ny_, np.int64), np.asarray(nx_, np.int64)
    nsrc = int(nsrc)
    hy, hx = max(int(wy) // 2, 1), max(int(wx) // 2, 1)
    # THE NODES ARE SORTED BY ROW ONCE, IN INTEGERS. A candidate's partners
    # are the nodes in the rows within `hy` of it -- a slice of that order
    # found by two searches -- that also lie within `hx` columns. Integer
    # pixel arithmetic throughout, so the box edge is exact: a scaled tree
    # query put the edge at a float comparison and lost the boundary nodes.
    _ord = np.argsort(ny_, kind='stable')
    _ys = ny_[_ord]
    _lo = np.searchsorted(_ys, dy_ - hy, 'left')
    _hi = np.searchsorted(_ys, dy_ + hy, 'right')

    def _partners(c0, c1):
        # the row slabs of candidates c0..c1, then the column test
        _n = _hi[c0:c1] - _lo[c0:c1]
        _src = np.repeat(np.arange(c0, c1, dtype=np.int64), _n)
        _off = np.arange(int(_n.sum()), dtype=np.int64) \
            - np.repeat(np.r_[0, np.cumsum(_n)[:-1]], _n)
        _tgt = _ord[np.repeat(_lo[c0:c1], _n) + _off]
        _in = np.abs(nx_[_tgt] - dx_[_src]) <= hx
        return _src[_in], _tgt[_in]

    # arcs per candidate, counted the same way, so the ranges below are
    # sized by what they will hold
    _cnt = np.zeros(nsrc, np.int64)
    for _c0 in range(0, nsrc, 4096):
        _s, _ = _partners(_c0, min(nsrc, _c0 + 4096))
        _cnt += np.bincount(_s, minlength=nsrc)
    _total = int(_cnt.sum())
    _cap = int(max(_ARC_CAP, int(min_agreeing) if min_agreeing else 0))
    _lb = np.asarray(base_lab, np.int64)
    _multi = len(np.unique(_lb)) > 1
    # A RANGE IS AS MANY CANDIDATES AS `budget` HOLDS ARCS FOR. The range's
    # aligned arrays -- two indices, coherence, height, rate, seasonal, the
    # ranking permutation and its gathers -- are of order a hundred bytes per
    # arc, so this keeps a range's working set at the budget; the fit inside
    # sizes its own transients against the same number.
    _B = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // 128))
    _cum = np.cumsum(_cnt)
    _thr = float(threshold)
    outs = []
    c0 = 0
    while c0 < nsrc:
        _base = int(_cum[c0 - 1]) if c0 else 0
        c1 = int(np.searchsorted(_cum, _base + _B, 'right'))
        c1 = min(max(c1, c0 + 1), nsrc)
        c_prev, c0 = c0, c1
        if not _cnt[c_prev:c1].sum():
            continue
        ksr, ktr = _partners(c_prev, c1)
        ga_r, dha_r, dva_r, dsa_r = _3d_arc_batch(
            Us, Ut, ksr, ktr, ele2phase, t, meter2rad, max_dh, max_dv,
            step_dh, step_dv, budget, iterations, threads=threads,
            threshold=_thr)
        ga_r = np.asarray(ga_r, np.float32)
        good_r = np.isfinite(ga_r) & (ga_r >= _thr) & np.isfinite(dha_r)
        # best arc first within each candidate; the ones below threshold sort
        # last and are never read again
        _o = np.lexsort((-np.where(good_r, ga_r, -np.inf), ksr))
        ks_o = ksr[_o] - c_prev
        good_o = good_r[_o]
        if _multi:
            _first = np.r_[True, ks_o[1:] != ks_o[:-1]]
            _own = np.full(c1 - c_prev, -1, np.int64)
            _fi = np.flatnonzero(_first & good_o)
            _lbo = _lb[ktr[_o]]
            _own[ks_o[_fi]] = _lbo[_fi]
            good_o &= (_own[ks_o] >= 0) & (_lbo == _own[ks_o])
            del _lbo
        _gi = np.flatnonzero(good_o)
        if not len(_gi):
            continue
        _cg = np.bincount(ks_o[_gi], minlength=c1 - c_prev)
        _rank = (np.arange(len(_gi))
                 - np.repeat(np.r_[0, np.cumsum(_cg)[:-1]], _cg))
        _keep = _o[_gi[_rank < _cap]]
        outs.append((ksr[_keep], ktr[_keep], ga_r[_keep], dha_r[_keep],
                     dva_r[_keep], dsa_r[_keep]))
        del ksr, ktr, ga_r, dha_r, dva_r, dsa_r, _o, ks_o, good_o, good_r
    if outs:
        ksrc, ktgt, ga, dha, dva, dsa = (np.concatenate(z) for z in zip(*outs))
    else:
        ksrc = np.empty(0, np.int64)
        ktgt = np.empty(0, np.int64)
        ga = np.empty(0, np.float32)
        dha = np.empty(0)
        dva = np.empty(0)
        dsa = np.empty(0, np.complex128)
    good = np.ones(len(ksrc), dtype=bool)
    if stats is not None:
        stats[prefix + 'ranked_arcs'] = 0
        stats[prefix + 'searched_arcs'] = _total
        stats[prefix + 'kept_arcs'] = int(len(ksrc))
        stats[prefix + 'provisional'] = int(np.count_nonzero(_cnt))
    return ksrc, ktgt, ga, dha, dva, dsa, good


def _3d_partner_consensus(src, ga, good, v_abs, nsrc, min_agreeing,
                          err_v, passes, labels=None, stats=None,
                          prefix='ds_', h_abs=None, err_h=None):
    """Do a pixel's best `min_agreeing` partners agree? Returns (first, votes, ok).

    ONLY THE BEST `min_agreeing` PARTNERS ENTER, AND NOTHING ELSE DOES. A pixel
    sees tens of candidates spanning every quality from just above `threshold`
    upwards, and a centre taken across that mixture estimates nothing: it
    summarises several populations, so it describes neither the good arcs nor
    the bad. So the partners are CHOSEN first, by arc coherence, which is
    settled before any value is read and so cannot be picked to suit the
    answer.

    THE TEST: EVERY PARTNER WITHIN `err` OF THE MEDIAN OF THEM. Each partner
    is a complete measurement of the pixel -- the partner's own value plus
    the arc -- so `n` partners are `n` measurements of one quantity. The
    median is one of those measurements rather than an average of them, so
    the band is `err` itself, the same bound the solve later holds every
    equation to, with nothing to correct for how much of the centre a
    measurement made itself. For three partners it reads plainly: no two
    consecutive values more than `err` apart -- the chain of steps that the
    height bound is for arcs. A pair that agrees with a third value a little
    further off is refused where three values spread evenly over the same
    range are admitted, and that preference is measured, not assumed: at the
    margin the lone dissenter is the worse sign.

    NO REWEIGHTING. The previous centre started at the median and then
    reweighted the partners by coherence over distance, with a scale read off
    the whole block; with `n` values that decides the closest ones are right
    and judges the rest against them, and it made the effective band depend
    on the block. The median alone keeps the preference without the
    machinery.

    THE VALUE IS THE COHERENCE-WEIGHTED MEAN. The gate certifies; what the
    partners collectively say is their weighted mean, exported as the centre
    as the provisional value. All of them lie
    within `err` of the median, so the mean does too.

    UNANIMOUS. Naming WHICH partners have to agree is what makes it a test
    rather than "some few of many", which any unimodal scatter passes on its
    shape alone. Allowing one dissenter is not the mild relaxation it reads
    as: there are only `n` columns, so a pixel holding `n - 1` admissible
    partners fills every column it has and passes, spending the tolerance
    meant for one partner DISAGREEING on one being ABSENT.

    `passes` is accepted for the callers' sake and unused: nothing here
    iterates.
    """
    _ma = None if min_agreeing is None else int(min_agreeing)
    o2 = np.lexsort((-np.where(good, ga, -np.inf), src))
    o2 = o2[good[o2]]
    cnt = np.bincount(src[o2], minlength=nsrc)
    if not len(cnt) or not cnt.max():
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32),
                np.zeros(nsrc, dtype=bool))
    off = np.r_[0, np.cumsum(cnt)[:-1]]
    col = np.arange(len(o2)) - np.repeat(off, cnt)
    # columns are gamma-descending, so the first `_ma` ARE the best `_ma`; a
    # row with fewer admissible arcs cannot fill them and is rejected for
    # having too few
    if _ma is not None:
        take = col < _ma
        o2, col = o2[take], col[take]
        cnt = np.minimum(cnt, _ma)
    kmax = int(cnt.max())
    row = src[o2]
    V = np.full((nsrc, kmax), np.nan)
    G = np.zeros((nsrc, kmax))
    IDX = np.full((nsrc, kmax), -1, dtype=np.int64)
    V[row, col] = v_abs[o2]
    G[row, col] = ga[o2]
    IDX[row, col] = o2
    fin = np.isfinite(V)
    Gf = np.where(fin, G, 0.0)
    sw = np.maximum(Gf.sum(axis=1), 1e-30)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN rows
        med_v = np.nanmedian(V, axis=1)
    cen_v = (Gf * np.where(fin, V, 0.0)).sum(axis=1) / sw
    keep = fin & (np.abs(V - med_v[:, None]) <= float(err_v))
    cen_h = None
    if h_abs is not None:
        Hm = np.full((nsrc, kmax), np.nan)
        Hm[row, col] = np.asarray(h_abs)[o2]
        cen_h = (Gf * np.where(fin, Hm, 0.0)).sum(axis=1) / sw
        if err_h is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med_h = np.nanmedian(Hm, axis=1)
            keep &= np.abs(Hm - med_h[:, None]) <= float(err_h)
    nkeep = keep.sum(axis=1)
    if _ma is not None:
        ok = keep[:, :_ma].sum(axis=1) == _ma
    else:
        ok = nkeep >= 1
    # rows are gamma-descending, so the first survivor is the best arc of the
    # consistent set
    sel_col = np.argmax(keep, axis=1)
    first = IDX[np.flatnonzero(ok), sel_col[ok]]
    votes = nkeep[ok].astype(np.int32)
    if stats is not None:
        stats[prefix + 'centre_v'] = np.where(nkeep > 0, cen_v, np.nan)
        stats[prefix + 'centre_h'] = (None if cen_h is None
                                      else np.where(nkeep > 0, cen_h, np.nan))
        # THE ARCS THAT VOTED. A pixel's consistency has to be read over the
        # partners its consensus actually rested on, the way the network's
        # closure is read over the arcs its solve rested on.
        _vm = keep & ok[:, None]
        _vi = IDX[_vm]
        stats[prefix + 'vote_arcs'] = _vi[_vi >= 0]
        reach = int(np.count_nonzero(cnt))
        stats[prefix + 'admissible'] = reach
        stats[prefix + 'no_consensus'] = reach - len(first)
        stats[prefix + 'too_few'] = int(
            reach - np.count_nonzero(fin[:, :_ma].sum(axis=1) >= _ma)
            if _ma is not None else 0)
        if labels is not None:
            # THE INVARIANT, CHECKED RATHER THAN ARGUED. The shortlist behind
            # an attached pixel must lie in ONE component: components carry
            # unrelated datums, so a decision taken across two is a numerical
            # error, not a noisier answer. Must read 0 on any scene with any
            # parameters. On `fin`, NOT on `keep` -- a cross-component partner
            # differs by the datum offset, which is what makes it an outlier,
            # so the rejection discards it before `keep` exists.
            _rw = np.flatnonzero(ok)
            _ix2 = IDX[_rw]
            _lb2 = np.where(fin[_rw] & (_ix2 >= 0),
                            labels[np.clip(_ix2, 0, None)].astype(np.int64), -1)
            _hi2 = _lb2.max(axis=1)
            _lo2 = np.where(_lb2 >= 0, _lb2, np.iinfo(np.int64).max).min(axis=1)
            stats[prefix + 'cross_component_votes'] = int(
                np.count_nonzero((_hi2 >= 0) & (_hi2 != _lo2)))
    return first, votes, ok


def _3d_fit_frame(date_values, bperp, geometry, n):
    """The time base and the height-to-phase factors the three stages share.

    Pulled out because the stages run in separate tasks under `union=True` and
    each has to rebuild the same frame; deriving it twice from the same inputs
    is exact, while passing it between tasks would let the two drift.
    """
    t = np.asarray(date_values)
    if t.dtype.kind == 'M':
        t = t.astype('datetime64[D]').astype(np.float64)
    if bperp is None:
        # an all-zero baseline makes the height lattice degenerate: ties
        # resolve to an extreme cell and the edge gate then rejects EVERY
        # arc, which reads as an empty result rather than a missing input
        raise ValueError(
            'fit1d_network() needs a perpendicular baseline per date (the BPR '
            'variable) to separate height from velocity; none was found.')
    B = np.asarray(bperp, dtype=np.float64)
    # zero at the master, where the phase is zero by construction and the
    # height term vanishes with the baseline that carries it
    t = (t - t[int(np.argmin(np.abs(B)))]) / 365.25
    wavelength, r_sin = geometry
    meter2rad = 4.0 * np.pi / wavelength
    ele2phase = B / r_sin
    car = np.exp(2j * np.pi * t)
    return t, ele2phase, meter2rad, car


# THE NETWORK'S LEAST SQUARES ARE A LAPLACIAN SOLVE. Every solve of the
# network is min_x sum_a w_a (x[i_a] - x[j_a] - d_a)^2 over its arcs, and its
# normal equations are the arc graph's weighted Laplacian, L x = G^T W d:
# symmetric, positive semi-definite, one free constant per component. LSQR
# solved the same problem on the arcs themselves, without preconditioning, and
# stopped at its 2000-iteration limit on every solve of the GasteinValley
# burst (70-100 s each, 38 of the run's 40): a node holds from one arc to over
# a thousand, and the unscaled system is conditioned accordingly. Scaled by
# its diagonal (Jacobi) the Laplacian reaches a relative residual of 1e-12 in
# 400-700 conjugate-gradient iterations -- a solution closer to the exact
# least squares than the iteration-limited one by orders of magnitude, in a
# few seconds instead of a minute and a half.
_LAP_BLOCK = 2048


@_numba.njit(nogil=True, cache=True)
def _3d_lap_mv(indptr, indices, data, P, Q, part, b0, b1, blk):
    """Q = L P on the rows of blocks [b0, b1), both columns at once, GIL-free.

    part[b] is block b's P.Q per column, so the caller sums the blocks in
    order and the dot product does not depend on how the blocks were shared
    among threads.
    """
    n = P.shape[0]
    for b in range(b0, b1):
        q0 = 0.0
        q1 = 0.0
        for r in range(b * blk, min(n, (b + 1) * blk)):
            s0 = 0.0
            s1 = 0.0
            for p in range(indptr[r], indptr[r + 1]):
                j = indices[p]
                v = data[p]
                s0 += v * P[j, 0]
                s1 += v * P[j, 1]
            Q[r, 0] = s0
            Q[r, 1] = s1
            q0 += P[r, 0] * s0
            q1 += P[r, 1] * s1
        part[b, 0] = q0
        part[b, 1] = q1


@_numba.njit(nogil=True, cache=True)
def _3d_cg_step(X, R, P, Q, dinv, alpha, part, b0, b1, blk):
    """X += alpha P and R -= alpha Q on blocks [b0, b1); part[b] is block b's
    (R.D^-1 R, R.R) per column."""
    n = X.shape[0]
    a0 = alpha[0]
    a1 = alpha[1]
    for b in range(b0, b1):
        z0 = 0.0
        z1 = 0.0
        e0 = 0.0
        e1 = 0.0
        for r in range(b * blk, min(n, (b + 1) * blk)):
            X[r, 0] += a0 * P[r, 0]
            X[r, 1] += a1 * P[r, 1]
            r0 = R[r, 0] - a0 * Q[r, 0]
            r1 = R[r, 1] - a1 * Q[r, 1]
            R[r, 0] = r0
            R[r, 1] = r1
            d = dinv[r]
            z0 += r0 * r0 * d
            z1 += r1 * r1 * d
            e0 += r0 * r0
            e1 += r1 * r1
        part[b, 0] = z0
        part[b, 1] = z1
        part[b, 2] = e0
        part[b, 3] = e1


@_numba.njit(nogil=True, cache=True)
def _3d_cg_dir(P, R, dinv, beta, b0, b1, blk):
    """P = D^-1 R + beta P on blocks [b0, b1)."""
    n = P.shape[0]
    g0 = beta[0]
    g1 = beta[1]
    for b in range(b0, b1):
        for r in range(b * blk, min(n, (b + 1) * blk)):
            d = dinv[r]
            P[r, 0] = d * R[r, 0] + g0 * P[r, 0]
            P[r, 1] = d * R[r, 1] + g1 * P[r, 1]


def _3d_lap_solve(ai, aj, w, N, D, threads=1, rtol=1e-12, maxiter=20000):
    """x minimising sum_a w_a (x[ai_a] - x[aj_a] - D[a, c])^2, per column c.

    ai, aj : (m,) the arcs' end nodes, indices into the N nodes
    w      : (m,) the arcs' weights, >= 0
    D      : (m, 2) the arcs' measured differences, one column per quantity
    Returns (N, 2). A node without an arc is 0, as LSQR's minimum-norm answer
    left it; within a component the answer is free by one constant, which
    the caller's datum removes.

    BOTH QUANTITIES ON ONE PASS OVER THE MATRIX. Height and rate share the
    arcs and the weights, so they share the Laplacian: every iteration reads it
    once for the two of them, and each column stops on its own residual.

    THE ANSWER DOES NOT DEPEND ON THE THREAD COUNT. The rows are handed to the
    pool in whole blocks of `_LAP_BLOCK`, every row is summed in its own order,
    and every dot product is summed per block and then over the blocks in
    order, so one thread and sixteen give the same bits.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.sparse import coo_matrix
    ai = np.asarray(ai, np.int64)
    aj = np.asarray(aj, np.int64)
    w = np.asarray(w, np.float64)
    D = np.asarray(D, np.float64).reshape(len(ai), 2)
    deg = np.bincount(ai, w, N) + np.bincount(aj, w, N)
    X = np.zeros((N, 2))
    B = np.empty((N, 2))
    for c in range(2):
        wd = w * D[:, c]
        B[:, c] = np.bincount(ai, wd, N) - np.bincount(aj, wd, N)
    nb = np.sqrt((B * B).sum(axis=0))
    done = nb == 0
    if done.all():
        return X
    _n = np.arange(N)
    L = coo_matrix((np.r_[-w, -w, deg], (np.r_[ai, aj, _n], np.r_[aj, ai, _n])),
                   shape=(N, N)).tocsr()
    L.sort_indices()
    ip = L.indptr.astype(np.int64)
    ix = L.indices.astype(np.int32)
    dt = L.data
    del L
    # JACOBI: the diagonal is each node's summed arc weight, the scale its
    # arc count sets; a node without an arc has no row and stays where it is
    dinv = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)
    blk = _LAP_BLOCK
    nblk = -(-N // blk)
    # the pool's shares in whole blocks, balanced by the rows' stored entries
    bnnz = ip[np.minimum(np.arange(nblk + 1) * blk, N)]
    th = max(1, min(int(threads), nblk))
    cut = np.unique(np.searchsorted(bnnz, np.linspace(0, bnnz[-1], th + 1)))
    cut[0], cut[-1] = 0, nblk
    spans = [(int(cut[i]), int(cut[i + 1])) for i in range(len(cut) - 1)
             if cut[i + 1] > cut[i]]
    R = B
    P = dinv[:, None] * R
    Q = np.empty((N, 2))
    part1 = np.zeros((nblk, 2))
    part2 = np.zeros((nblk, 4))
    rz = (R * R * dinv[:, None]).sum(axis=0)
    rr = nb * nb
    ex = ThreadPoolExecutor(len(spans)) if len(spans) > 1 else None

    def _run(f, *args):
        if ex is None:
            f(*args, *spans[0], blk)
        else:
            list(ex.map(lambda s: f(*args, *s, blk), spans))

    try:
        for _ in range(int(maxiter)):
            _run(_3d_lap_mv, ip, ix, dt, P, Q, part1)
            pq = part1.sum(axis=0)
            alpha = np.where(done | (pq == 0), 0.0,
                             rz / np.where(pq != 0, pq, 1.0))
            _run(_3d_cg_step, X, R, P, Q, dinv, alpha, part2)
            s = part2.sum(axis=0)
            rz_new, rr = s[:2], s[2:]
            done = done | (np.sqrt(rr) <= rtol * nb)
            if done.all():
                break
            beta = np.where(done | (rz == 0), 0.0,
                            rz_new / np.where(rz != 0, rz, 1.0))
            _run(_3d_cg_dir, P, R, dinv, beta)
            rz = rz_new
    finally:
        if ex is not None:
            ex.shutdown()
    if not done.all():
        warnings.warn(
            f'fit1d_network(): a network solve stopped at {int(maxiter)} iterations, '
            f'relative residual {float(np.max(np.sqrt(rr) / np.where(nb > 0, nb, 1))):.1e} '
            f'(asked {rtol:g}); its answer is the last iterate')
    return X


# HOW MANY ARCS A NODE MAY BRING TO THE NETWORK. Not a quality threshold --
# `threshold` is that -- but a bound on redundancy: beyond this many arcs a
# node is drawing repeatedly from the same neighbourhood, so the rows stop
# carrying independent information while the solve keeps paying for them.
# Applied AFTER the fits, by fitted coherence, so no pair that only fits
# coherent once its model is removed can be lost to a raw ranking; the cap
# only sizes the solve. Thirty-six is the independence cell's own count on
# the reference grid, and with one candidate per 2 x 2 cells it holds the
# solve near the size it had with the coarser winner boxes.
_ARC_CAP = 36


def _3d_ps_network(U, iy, ix, date_values, *, bperp=None, window=(32, 128),
                   threshold=0.5, geometry, budget=None, consensus,
                   iterations=8, max_dh=25.0, max_dv=25.0, step_dh=8.0,
                   step_dv=2.0, max_seasonal=5.0, err_dh=4.0, err_dv=1.0,
                   threads=None, debug=False, arcs=None):
    """The network over the nodes ALONE -- no raster, so no scene in memory.

    Every argument is a node quantity, which is why this stage can be run once
    over nodes gathered from several blocks or several bursts: the arcs are
    fitted between phasor columns and the datum is per component, neither of
    which asks where the pixels were stored.

    Returns the solved node table, or None when no network could be formed.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from .utils_dask import cluster_cores
    _nth = max(1, int(threads) if threads else cluster_cores()[1])
    _ma = _3d_consensus(consensus)
    _ii = max(1, int(iterations))
    Un = np.ascontiguousarray(U, dtype=np.complex64)
    iy = np.asarray(iy)
    ix = np.asarray(ix)
    n = Un.shape[0]
    wy, wx, pey, pex = _3d_windows(window)
    _fit_stats.reset(nodes=0, arcs=0, dropped=0, components=[],
                                 fill_order=[])
    _mark = time.monotonic()
    if n < 2 or len(iy) < 2:
        return None
    t, ele2phase, meter2rad, car = _3d_fit_frame(date_values, bperp,
                                                 geometry, n)
    # THE BOUNDS ARE STATED IN METRES AND MM/YR AND USED IN RADIANS. A metre of
    # height and a mm/yr of rate carry different amounts of phase, so the two
    # are not interchangeable numbers -- they are one statement about how far a
    # measurement may sit from the solve, each in its own unit.
    _err_h = float(err_dh) * meter2rad
    _err_v = float(err_dv) * meter2rad / 1e3
    # ---- the arcs: every pair inside the PS window, IN BATCHES ----------
    # scaled so the window becomes the unit box, then a Chebyshev query is
    # exactly "inside the window" and costs O(N k) rather than O(N^2)
    #
    # NOTHING OF PAIR LENGTH IS EVER HELD. Every pair inside the extent is
    # fitted, but only the few that clear `threshold` are kept, so the pairs
    # arrive a chunk of nodes at a time, each chunk is fitted in batches
    # sized by `budget`, and a batch's survivors are appended before the next
    # batch exists. The pair list the tree would hand back in one piece and
    # the per-pair results -- tens of gigabytes on a dense scene, for a set
    # the threshold and the cap then threw away -- never exist. The cap below
    # still sees the complete set it always saw: everything above threshold.
    hy, hx = max(pey // 2, 1), max(pex // 2, 1)
    _thr = float(threshold)
    # a batch's own arrays -- two indices, coherence, height, rate, seasonal
    # -- are of order sixty bytes per arc; the fit sizes its own transients
    # against the same budget inside
    _B = max(_nth * 4, int(_3d_budget_mb(budget) * 1024 * 1024 // 64))
    _acc = {k: [] for k in ('i', 'j', 'g', 'dh', 'dv', 'ds')}
    _n_fit = 0

    def _fit_batch(a_, b_):
        """One batch of arcs, threaded across `_nth` slices."""
        m = len(a_)
        g_ = np.empty(m, np.float32)
        dh_ = np.empty(m)
        dv_ = np.empty(m)
        ds2 = np.empty(m, np.complex128)

        # ONE ORIENTATION PER ARC IN THE NODE TABLE. The tree hands pairs
        # over as (min, max) of ITS indices and a caller as it likes; formed
        # the two ways the same arc differs in its last bits -- the complex
        # product is not symmetric under a swap -- and a lattice fit can
        # answer a near-tie one cell apart on them, where a continuous one
        # converged to the same point. Formed lower column first and negated
        # back, the arc's bytes, and so its fit, are the same whoever
        # enumerated it.
        _lo, _hi, _neg = np.minimum(a_, b_), np.maximum(a_, b_), a_ > b_

        def _run(sel, budget_):
            step = max(1, int(_3d_budget_mb(budget_) * 1024 * 1024
                              // max(n * 16, 1)))
            for b0 in range(0, len(sel), step):
                s_ = sel[b0:min(b0 + step, len(sel))]
                arc = _3d_arc_product(Un[:, _lo[s_]], Un[:, _hi[s_]])
                g_[s_], dh_[s_], dv_[s_], ds2[s_] = _3d_arc_fit(
                    arc, ele2phase, t, meter2rad, max_dh, max_dv, step_dh,
                    step_dv, budget_, max_seasonal, iterations=iterations,
                    threshold=_thr)
                _n = s_[_neg[s_]]
                dh_[_n] = -dh_[_n]
                dv_[_n] = -dv_[_n]
        if _nth > 1 and m > _nth:
            # arcs are independent; slices of them fit concurrently
            from concurrent.futures import ThreadPoolExecutor
            _bnd = np.linspace(0, m, _nth * 4 + 1).astype(np.int64)
            _idx = np.arange(m)
            with ThreadPoolExecutor(_nth) as _ex:
                list(_ex.map(lambda i: _run(
                    _idx[_bnd[i]:_bnd[i + 1]],
                    _3d_budget_mb(budget) / _nth), range(_nth * 4)))
        else:
            _run(np.arange(m), budget)
        return g_, dh_, dv_, ds2

    def _batches(a_, b_):
        nonlocal _n_fit
        for b0 in range(0, len(a_), _B):
            sl = slice(b0, min(b0 + _B, len(a_)))
            g_, dh_, dv_, ds2 = _fit_batch(a_[sl], b_[sl])
            _n_fit += sl.stop - sl.start
            ok = np.isfinite(g_) & (g_ >= _thr)
            if ok.any():
                _acc['i'].append(a_[sl][ok])
                _acc['j'].append(b_[sl][ok])
                _acc['g'].append(g_[ok])
                _acc['dh'].append(dh_[ok])
                _acc['dv'].append(dv_[ok])
                _acc['ds'].append(ds2[ok])

    if arcs is None:
        # A CHUNK OF NODES AT A TIME: the chunk's nodes are queried against
        # the whole tree and each pair is taken once, from its lower end. The
        # chunk is sized from the degree the previous one measured, so that
        # its pair records stay near one batch whatever the density.
        # IN THE TREE'S LEAF ORDER, NOT INDEX ORDER. The arcs keep the order
        # they are found in, and the solve below walks its sparse system in
        # that order: chunks of spatial neighbours give it the locality the
        # one-piece tree query gave, chunks of consecutive indices do not,
        # and the robust pass was measured well slower on the same arcs
        # ordered the second way.
        pts = np.c_[iy / hy, ix / hx]
        tree = cKDTree(pts)
        _leaf = np.asarray(tree.indices, dtype=np.int64)
        _rank = np.empty(len(iy), np.int64)
        _rank[_leaf] = np.arange(len(iy))
        _chunk = min(len(iy), 256)
        c0 = 0
        while c0 < len(iy):
            c1 = min(len(iy), c0 + _chunk)
            _sub = _leaf[c0:c1]
            rec = cKDTree(pts[_sub]).sparse_distance_matrix(
                tree, 1.0, p=np.inf, output_type='ndarray')
            gi = _sub[rec['i'].astype(np.int64)]
            gj = rec['j'].astype(np.int64)
            # each pair once: from the chunk's node when the partner is not
            # in an earlier chunk, which the leaf rank decides. LOWER INDEX
            # FIRST, as the one-piece query oriented them: the cap below
            # ranks a node's arcs as their first end and as their second end
            # separately, so the orientation is part of what it keeps.
            m_ = _rank[gj] > _rank[gi]
            a_ = np.minimum(gi, gj)[m_]
            b_ = np.maximum(gi, gj)[m_]
            del rec, gi, gj, m_
            _deg = max(1.0, 2.0 * len(a_) / max(c1 - c0, 1))
            _chunk = int(min(len(iy), max(64, _B // _deg)))
            _batches(a_, b_)
            del a_, b_
            c0 = c1
    else:
        # THE CALLER'S ARCS, as (n_arcs, 2) node indices. An exhaustive
        # candidate search (fit3d_brute) has already met every pair inside
        # the window and knows which ones hold a coherent arc; over its dense
        # node set every pair would be hundreds of millions of fits for arcs
        # the search has already rejected. Nothing below this line changes:
        # the arcs are fitted, capped at _ARC_CAP, weighted, gated and solved
        # exactly as the tree's pairs would be.
        pairs = np.ascontiguousarray(np.asarray(arcs, dtype=np.int64)
                                     .reshape(-1, 2))
        _batches(pairs[:, 0], pairs[:, 1])
        del pairs
    if _n_fit < 3:
        return None
    _n_good = sum(len(v) for v in _acc['g'])
    _lap, _mark = _3d_lap(_mark)
    if debug:
        print(f'DEBUG: arcs     {_n_fit:,} pairs fitted, {_n_good:,} '
              f'>= {_thr}  ({100 * _n_good / max(_n_fit, 1):.1f}%)'
              f'   {_lap:.1f}s', flush=True)
        if _n_good:
            _gk = np.concatenate(_acc['g'])
            print(f'DEBUG:          arc gamma p50 {np.median(_gk):.3f}  '
                  f'p90 {np.percentile(_gk, 90):.3f}  max {_gk.max():.3f}',
                  flush=True)
            del _gk
    if _n_good < 3:
        if debug:
            print('DEBUG: fewer than 3 arcs cleared the threshold', flush=True)
        return None
    ai, aj, gk, dh, dv, ds_ = (np.concatenate(_acc[k]) for k in
                               ('i', 'j', 'g', 'dh', 'dv', 'ds'))
    del _acc

    # ---- A NODE'S BEST `_ARC_CAP` ARCS, AND NO MORE ---------------------
    # A node's degree is its PS density times the window's area, so it grows
    # without bound as a scene gets denser while the window stays fixed -- and
    # the least-squares system grows with it. Past some number of arcs a node
    # is not being measured any better: the extra ones are drawn from the same
    # neighbourhood as the ones already kept, so they add rows without adding
    # independent constraint, and the robust pass has to judge every one of
    # them. On a sparse scene the cap is inert; on a dense one it is what keeps
    # the solve proportional to the ground rather than to the density.
    #
    # AN ARC SURVIVES IF EITHER END STILL WANTS IT, which is what makes this
    # safe. Capping each node's own list independently would drop the long
    # arcs that tie distant groups together -- they rank low for both ends
    # because coherence falls with distance -- and the network would fall into
    # pieces, each with its own free datum, which is far worse than any noise
    # the cap removes. Keeping an arc that either end still ranks means a node
    # cannot be isolated by another node's budget, and a long arc survives as
    # long as one of its ends has room for it.
    #
    # Redundancy is still what the screen is made of; only the excess beyond
    # what a node can use is rationed, and the robust pass below decides which
    # of the rest the network believes.
    if _ARC_CAP and len(ai) > 1:
        _cap_keep = np.zeros(len(ai), dtype=bool)
        for _ends in (ai, aj):
            _o = np.lexsort((-gk, _ends))
            _e = _ends[_o]
            _starts = np.r_[0, np.flatnonzero(np.diff(_e)) + 1]
            _group = np.cumsum(np.r_[True, np.diff(_e) != 0]) - 1
            _cap_keep[_o[np.arange(len(_e)) - _starts[_group] < _ARC_CAP]] = True
        if not _cap_keep.all():
            _lap, _mark = _3d_lap(_mark)
            if debug:
                print(f'DEBUG: cap      {int((~_cap_keep).sum()):,} of '
                      f'{len(ai):,} arcs beyond {_ARC_CAP} per node dropped'
                      f'   {_lap:.1f}s', flush=True)
            ai, aj = ai[_cap_keep], aj[_cap_keep]
            dh, dv, ds_ = dh[_cap_keep], dv[_cap_keep], ds_[_cap_keep]
            gk = gk[_cap_keep]
    # ONE ORDER FOR THE SOLVE. The arcs arrive in whatever order the tree or
    # a caller enumerated them, and the solve below is iterative: its last
    # bits, and with them a knife-edge rejection, follow the row order.
    # Sorted by their ends the same arcs give the same system whoever handed
    # them over, so the network is a function of the arcs alone.
    if len(ai):
        _o = np.lexsort((aj, ai))
        ai, aj, dh, dv, ds_, gk = ai[_o], aj[_o], dh[_o], dv[_o], ds_[_o], gk[_o]
    N = len(iy)
    gtake = gk

    def _incidence(a_, b_, m):
        return coo_matrix((np.tile([1.0, -1.0], m),
                           (np.repeat(np.arange(m), 2), np.c_[a_, b_].ravel())),
                          shape=(m, N)).tocsr()

    def _wsolve2(a_, b_, rhs_a, rhs_b, w):
        """The height and the rate solved at once, on the arcs (a_, b_).

        Every pass of the reweighting and every pass of the gate solves the
        network TWICE, once for each quantity, and the two share the arcs and
        the weights and depend on nothing of each other's -- so they share
        every pass over the Laplacian (`_3d_lap_solve`), on the stage's threads.
        These solves were the great majority of the stage.
        """
        x = _3d_lap_solve(a_, b_, w, N, np.c_[rhs_a, rhs_b], threads=_nth)
        return x[:, 0], x[:, 1]

    def _mad(r):
        return 1.4826 * float(np.median(np.abs(r - np.median(r))))

    # ---- REJECT ARCS THE NETWORK CONTRADICTS ---------------------------
    # An arc's coherence says how well it fits its OWN phase; it does not say
    # whether it agrees with the rest of the network, and the two are close to
    # independent. Plain least squares cannot tell the difference: it has no
    # way to reject one equation, so a contradicted arc is absorbed by
    # spreading its error over every arc that shares a node with it. On a
    # thinly connected node that is the whole solution.
    #
    # IRLS finds them, then they are DROPPED rather than down-weighted, and
    # the survivors are solved exactly. A down-weighted arc still perturbs the
    # answer and still holds its nodes in the component; a dropped one leaves
    # a node unsupported, which is the honest outcome -- that node had no
    # consistent measurement.
    G = _incidence(ai, aj, len(ai))
    w_ = gtake.astype(np.float64)
    r_h = r_v = None
    for _ in range(_ii):
        _xh, _xv = _wsolve2(ai, aj, dh, dv, w_)
        r_h = G @ _xh - dh
        r_v = G @ _xv - dv
        # per NODE, not pooled: judged against its own arcs' scatter
        # ONE ROBUST SCALE FOR THE NETWORK, per pass: the scale of every
        # arc's residual against the current solve. A scale per node was
        # tried, floored at twice the network's and capped at the network's
        # own -- which is the network's own for every node, so it changed
        # nothing and cost a pass over the nodes in Python each iteration.
        s_h, s_v = max(_mad(r_h), 1e-12), max(_mad(r_v), 1e-12)
        z = np.maximum(np.abs(r_h) / s_h, np.abs(r_v) / s_v)
        w_ = gtake / np.maximum(z, 1.0)
    # scale from the ROBUST fit, so the outliers do not set their own bar
    # ---- REJECT, RE-SOLVE, RE-CHECK, UNTIL IT SETTLES -------------------
    # `reject_sigma` promises that no surviving arc sits further than that
    # many node scales from the solution. Rejecting once cannot deliver it:
    # removing arcs MOVES the solution, and an arc validated against the old
    # one may sit well outside the gate of the new one -- measured, a fifth of
    # the arcs were rejected, and 2.4% of the survivors then had a residual
    # more than three times what the gate had seen.
    #
    # So the gate is re-applied to the solution it produced, until the
    # surviving set stops changing. Then the promise is true of the answer
    # that ships rather than of an intermediate nobody receives. It converges
    # in a few passes because each one removes less than the last; the cap is
    # only to bound the work if it ever oscillates between two sets.
    # THE SCALE IS ESTIMATED ONCE AND HELD. It describes the measurement
    # noise, which does not change because arcs were deleted -- only the
    # SOLUTION does. Re-estimating it every pass makes the gate feed on
    # itself: a cleaner set gives a tighter scale, which rejects more, which
    # tightens it again, and the loop erodes the network instead of settling.
    # Measured that way it removed a further 2 000 arcs and 35 nodes and still
    # left arcs at z = 101, because the bar moved under them.

    def _gate(idx):
        """Residual of the solve on `idx`, scored against the HELD scale."""
        a_, b_ = ai[idx], aj[idx]
        G_ = _incidence(a_, b_, len(idx))
        _xh_, _xv_ = _wsolve2(a_, b_, dh[idx], dv[idx], w_[idx])
        rh_ = G_ @ _xh_ - dh[idx]
        rv_ = G_ @ _xv_ - dv[idx]
        # ABSOLUTE, NOT RELATIVE. Scored against a robust sigma an arc could
        # widen the bar that judged it: a node whose arcs were uniformly wrong
        # got a scale to match and kept them all. The bound the caller stated
        # cannot be widened by what it is judging.
        z_ = np.maximum(np.abs(rh_) / _err_h, np.abs(rv_) / _err_v)
        return z_, rv_

    if True:
        keep_arc = np.ones(len(ai), dtype=bool)
        _passes = 0
        # HOW MANY TIMES TO REJECT AND RE-SOLVE. The same count that governs
        # the IRLS reweighting: both are iterative refinements of one solve
        # and there is no reason for a caller to reason about them separately.
        # The loop stops as soon as the surviving set repeats, so this only
        # bounds an oscillation -- how many are actually used depends on
        # `reject_sigma`, since a tighter gate removes more per pass and takes
        # longer to settle.
        for _passes in range(1, _ii + 1):
            _idx = np.flatnonzero(keep_arc)
            _z, _rv2 = _gate(_idx)
            _ok = _z <= 1.0
            if _ok.all():
                break                      # the gate holds on its own solution
            _new = np.zeros(len(ai), dtype=bool)
            _new[_idx[_ok]] = True
            if _new.sum() < 3:
                break
            keep_arc = _new
        if debug:
            # evaluated ONCE on the set that actually survived, so the numbers
            # describe the arcs the answer is built from
            _idx = np.flatnonzero(keep_arc)
            _dbg_z, _dbg_rv = _gate(_idx)
            _gate_passes = int(_passes)
    if keep_arc.sum() < 3:
        return None
    rejected = int((~keep_arc).sum())
    _lap, _mark = _3d_lap(_mark)
    if debug:
        print(f'DEBUG: IRLS     {rejected:,} of {len(ai):,} arcs rejected '
              f'beyond {err_dh:g} m / {err_dv:g} mm/yr  '
              f'({100 * rejected / max(len(ai), 1):.1f}%)'
              f'   {_lap:.1f}s', flush=True)
    # how much of each node's own support the rejection took away
    drej = np.bincount(np.r_[ai[~keep_arc], aj[~keep_arc]], minlength=N)

    # ---- THE FLOOR IS A FIXED POINT, NOT ONE PASS ----------------------
    # Dropping a node whose support is too thin takes its arcs with it, and
    # that lowers its PARTNERS' counts -- which can put them under the floor
    # in turn. Testing once leaves nodes standing on support that has since
    # been removed.
    #
    # The worse half is what it does to the COMPONENTS. A candidate below the
    # floor is never reported, but its arcs stay in the graph, so it still
    # BRIDGES: two groups that share no tested path are welded into one
    # component and handed a common datum on the strength of a node the same
    # rule just refused to report. The output then shows plainly disconnected
    # ground carrying the main component's label, which is not a thin answer
    # but a wrong one -- values presented as comparable that rest on no
    # measured connection.
    #
    # So the test is applied until it stops removing anything. That is the
    # k-core of the surviving-arc graph at k = `consensus`, and it is what
    # the single pass was always reaching for.
    _flr = _ma
    # counted BEFORE the floor runs: nodes the robust pass left with no arc at
    # all. Afterwards every removed node has none, so the two stop being
    # different questions and the distinction has to be taken here.
    _iso0 = int((np.bincount(np.r_[ai[keep_arc], aj[keep_arc]],
                             minlength=N) == 0).sum())
    _kalive = np.ones(N, dtype=bool)
    _kpass = 0
    while True:
        _kd = np.bincount(np.r_[ai[keep_arc], aj[keep_arc]], minlength=N)
        _kdrop = _kalive & (_kd < _flr)
        if not _kdrop.any():
            break
        _kalive &= ~_kdrop
        keep_arc = keep_arc & _kalive[ai] & _kalive[aj]
        _kpass += 1
        if keep_arc.sum() < 3:
            break
    if debug and _kpass > 1:
        print(f'DEBUG: floor    {_kpass} passes to reach the fixed point; '
              f'{int((~_kalive).sum())} of {N} candidates under {_flr} arc(s)',
              flush=True)
    if keep_arc.sum() < 3:
        if debug:
            print('DEBUG: fewer than 3 arcs survive the consensus floor',
                  flush=True)
        return None

    if debug:
        # RE-EVALUATED ON THE ARCS THAT SHIP. The gate above ran before the
        # floor removed anything, so its residuals described a larger set than
        # the answer is built from -- and `arc_z` would no longer line up with
        # `arc_dh`, which is the one thing a caller reading them together
        # needs.
        _idx = np.flatnonzero(keep_arc)
        _dbg_z, _dbg_rv = _gate(_idx)

    # THE CLOSURE BELOW IS CIRCULAR IF IT SEES SURVIVORS ONLY. IRLS rejects
    # the arcs that disagree with the solve, so a residual measured over what
    # is left is a property of the selection as much as of the solution. Keep
    # the arcs as they were, and report both.
    _pre = ((ai.copy(), aj.copy(), dh.copy(), dv.copy(), keep_arc.copy())
            if debug else None)
    ai, aj, dh, dv, ds_, gtake = (ai[keep_arc], aj[keep_arc], dh[keep_arc],
                                  dv[keep_arc], ds_[keep_arc], gtake[keep_arc])

    dcount = np.bincount(np.r_[ai, aj], minlength=N)
    # each node's coherence is the mean over the arcs that actually hold it
    _gsum = np.bincount(np.r_[ai, aj], weights=np.r_[gtake, gtake], minlength=N)
    gnode = np.where(dcount > 0, _gsum / np.maximum(dcount, 1), np.nan)
    ncomp, lab = connected_components(
        coo_matrix((np.ones(len(ai)), (ai, aj)), shape=(N, N)), directed=False)
    # CONNECTED IS THE ONLY REQUIREMENT. A node with no surviving arc has no
    # datum -- nothing places it against anything else -- so it cannot be
    # reported. One surviving arc is enough, because of WHAT survival means
    # here: the robust pass has already rejected every arc the network
    # contradicts, so a node holding one arc holds one the network AGREES
    # with. Counting arcs a second time would re-ask a question the rejection
    # already answered.
    #
    # `consensus` SURVIVING ARCS, THE SAME NUMBER THE ATTACHMENT ASKS OF A DS.
    # One question asked twice: an arc must agree with the network, a partner
    # must agree with the other partners. A node reported on fewer than that
    # rests on measurements too few to have been checked against each other.
    #
    # These are SURVIVORS, so the count is what remains after the robust pass
    # rejected every arc the network contradicted -- the agreement is tested
    # there, and this requires enough of it to have been tested at all.
    #
    # It was briefly relaxed to 1, while `degree` capped arcs per node: the cap
    # starved nodes below the count and the ARC BUDGET decided how many PS
    # existed, which is not a property of the data. `degree` is gone and every
    # arc clearing `threshold` now enters, so a node short of survivors is
    # genuinely short of support.
    live = dcount >= _ma
    _lap, _mark = _3d_lap(_mark)
    if debug:
        # NODES LOST BEFORE ANY COMPONENT EXISTS. A node whose every arc was
        # rejected holds no datum and cannot be reported. Counted separately
        # from the component floor below: the two are different losses at
        # different stages, and a single "kept" total hides which is which.
        print(f'DEBUG: solve     {int((~live).sum())} of {len(live)} nodes '
              f'left under {_flr} surviving arc(s)'
              f'  ({_iso0} held none even before the floor)'
              f'   {_lap:.1f}s', flush=True)
    if not live.any():
        return None

    # ---- integrate, one free datum per component -----------------------
    # THE SOLUTION THE GATE VALIDATED IS THE SOLUTION THAT SHIPS. The robust
    # pass rejects arcs whose residual exceeds `reject_sigma` node scales --
    # but of ITS OWN solution. Re-solving the survivors under different
    # weights answers a different question, and the guarantee then attaches to
    # a solution nobody receives: measured, 4% of surviving arcs had a
    # residual more than three times larger in the coherence-weighted resolve
    # than in the one the gate approved.
    #
    # So the robust weights are carried through rather than discarded. They
    # already contain the coherence -- `w_ = gtake / max(z, 1)` -- so an arc
    # is weighted by its own quality AND by how far it sits from the
    # consensus, which is strictly more information than coherence alone.
    w_fin = w_[keep_arc]
    hgt, vel = _wsolve2(ai, aj, dh, dv, w_fin)
    anr, ani = _wsolve2(ai, aj, ds_.real, ds_.imag, w_fin)
    for c in np.unique(lab[live]):
        k = live & (lab == c)
        hgt[k] -= np.median(hgt[k])
        vel[k] -= np.median(vel[k])
        # the annual datum is a complex MEAN: a componentwise median does not
        # transform as a complex number, so the gauge it fixes would depend on
        # which epoch t was measured from
        anr[k] -= np.mean(anr[k])
        ani[k] -= np.mean(ani[k])

    sel = np.where(live)[0]
    # ---- model parameters at every node ---------------------------------
    # NO PHASE IS RETURNED. fit1d_network() emits the model only; phase is
    # reconstructed on demand by predict(model), which lets the caller
    # choose what to remove. Assembling n_dates complex planes here and
    # throwing them away cost one plane per date per chunk for nothing.
    # TOPOGRAPHY IS THE ONLY THING REMOVED. The height term is the one part
    # of the fitted model that is not ground motion, so taking it out leaves
    # a DISPLACEMENT series: rate, seasonal and whatever else the scatterer
    # actually did all stay. Removing rate and seasonal as well would leave a
    # residual, which is a different product and the one the atmospheric
    # screen used to be built from.
    #
    # Nothing is interpolated. A node has a measurement and its neighbours do
    # not, and spreading one node's value over ground that was never measured
    # is what the kriged screen did -- with per-node noise dominating the
    # correlated signal, it cost coherence at every separation.
    # Pixels without a node stay NaN, which is what they are.

    sy_, sx_ = iy[sel], ix[sel]
    comps = []
    # A COMPONENT MUST BE ABLE TO MEET THE CONSENSUS IT IS JUDGED BY. Each
    # component carries its own free datum, so a small one is not a sparser
    # answer -- it is a separately-datumed one, resting on however few arcs
    # its handful of nodes could form. `consensus` already states how many
    # agreeing measurements a value must rest on before it is reported; a
    # component with fewer nodes than that cannot supply them even in
    # principle. It is the caller's own requirement applied to the network
    # itself, not a second threshold: ask for less and smaller ones qualify,
    # and with `consensus=None` the check is off along with all the others.
    _cmin = int(_ma) if _ma is not None else 1
    for c in np.unique(lab[sel]):
        k = np.where(lab[sel] == c)[0]
        if len(k) < _cmin:
            continue
        comps.append((len(k), float(np.median(dcount[sel][k])), k))
    if not comps:
        return None
    # AT MOST _MAX_COMPONENTS, LARGEST FIRST -- past it an int8 label would
    # fold component 128 onto -128 and two unrelated datums would read as one.
    # What is dropped is the smallest, which is also the least trustworthy.
    comps.sort(key=lambda c: -c[0])
    if debug:
        _seen = sorted((int((lab[sel] == c).sum())
                        for c in np.unique(lab[sel])), reverse=True)
        _drop = [z for z in _seen if z < _cmin]
        print(f'DEBUG: network  {len(_seen)} connected component(s); '
              f'{len(_drop)} below the {_cmin}-node consensus floor',
              flush=True)
        # THE FIVE LARGEST, then one line for the rest: a scene solves into
        # one or a few real components and a long tail of two- and three-node
        # fragments, and a screen of those tells nothing the count above did.
        for _n_, _d_, _k_ in comps[:5]:
            print(f'DEBUG:          size {_n_:>6,}   arcs/node {_d_:5.1f}',
                  flush=True)
        if len(comps) > 5:
            _rest = comps[5:]
            print(f'DEBUG:          ... and {len(_rest)} smaller: sizes '
                  f'{_rest[-1][0]:,}-{_rest[0][0]:,}, '
                  f'{sum(c[0] for c in _rest):,} nodes in all', flush=True)
        if _drop:
            # SIZES, not just a count. "3 dropped" hides whether that is six
            # nodes or twenty; the floor is per COMPONENT, so a total far
            # above it can still be made of pieces every one of which is
            # below it.
            print(f'DEBUG:          dropped sizes {_drop}  '
                  f'= {sum(_drop)} nodes, each below {_cmin}', flush=True)
    dropped = max(0, len(comps) - _MAX_COMPONENTS)
    comps = comps[:_MAX_COMPONENTS]
    order_size = sorted(range(len(comps)), key=lambda z: -comps[z][0])
    label_of = {z: r for r, z in enumerate(order_size)}
    order_prio = sorted(range(len(comps)), key=lambda z: (-comps[z][1],
                                                          -comps[z][0]))
    # Each node belongs to exactly one component, so writing them cannot
    # contest a pixel -- the arbitration the kriged version needed went with
    # the interpolation.
    keep = []
    for z in order_prio:
        k = comps[z][2]
        keep.append(k)
    kk = np.concatenate(keep)
    lab_all = np.array([label_of[z] for z in order_prio
                        for _ in range(len(comps[z][2]))], dtype=np.int8)
    k_mm = meter2rad * 1e-3

    _fit_stats.reset(
        nodes=int(len(kk)), arcs=int(len(ai)), dropped=int(dropped),
        arcs_rejected=int(rejected),
        degree_rejected=drej[sel][kk].astype(np.int32),
        components=[dict(label=label_of[z], size=comps[z][0],
                         degree=comps[z][1],
                         iy=sy_[comps[z][2]].copy(), ix=sx_[comps[z][2]].copy())
                    for z in order_prio],
        fill_order=[label_of[z] for z in order_prio],
        # the network solution itself, per node, in physical units -- the
        # datum is per component and already applied, so these are relative
        # to their own component and free by one constant each
        iy=sy_[kk].copy(), ix=sx_[kk].copy(),
        label=lab_all,
        degree=dcount[sel][kk].astype(np.int32),
        # RADIANS, matching `_3d_arc_fit`'s (gamma, height_rad,
        # velocity_rad_yr, seasonal_rad). Converting to mm or metres in here
        # would put a second length convention inside the library, when
        # `displacement_los()` is meant to be the only place one appears.
        height_rad=hgt[sel][kk].astype(np.float32),
        velocity_rad_yr=vel[sel][kk].astype(np.float32),
        seasonal_rad=(anr[sel][kk] + 1j * ani[sel][kk]).astype(np.complex64))

    if debug:
        _dn = dcount[sel][kk]
        _gn = gnode[sel][kk]
        _gn = _gn[np.isfinite(_gn)]
        print(f'DEBUG: solved   {len(kk)} of {len(live)} nodes kept, '
              f'{int(dropped)} component(s) dropped past the int8 label limit'
              f'   {_3d_lap(_mark)[0]:.1f}s', flush=True)
        print(f'DEBUG:          arcs/node p50 {np.median(_dn):.0f}  '
              f'min {_dn.min()}  max {_dn.max()}', flush=True)
        if len(_gn):
            print(f'DEBUG:          node gamma p50 {np.median(_gn):.3f}  '
                  f'p10 {np.percentile(_gn, 10):.3f}', flush=True)
        # DOES THE SOLUTION SATISFY ITS OWN ARCS? The network solve defines the
        # datum every attached pixel inherits, so its self-consistency is the
        # property to report -- not how coherent the arcs were, which is a
        # different question already answered above. Differences, so the free
        # datum cannot enter.
        _rv = np.abs((vel[ai] - vel[aj]) - dv) / meter2rad * 1e3      # mm/yr
        _rh = np.abs((hgt[ai] - hgt[aj]) - dh) / meter2rad            # m

        # PER COMPONENT, NOT POOLED. Each carries its own datum, so its
        # closure is its own property: a small component whose arcs agree is
        # usable on that datum, and pooling would hide it behind a larger one
        # that does not. Arcs belong to a component when BOTH ends do.
        if _pre is not None:
            _pa, _pj, _pdh, _pdv, _pk = _pre
            _prv = np.abs((vel[_pa] - vel[_pj]) - _pdv) / meter2rad * 1e3
            _prh = np.abs((hgt[_pa] - hgt[_pj]) - _pdh) / meter2rad
            _f = np.isfinite(_prv) & np.isfinite(_prh)
            if _f.any():
                print(f'DEBUG:          PS closure over ALL {int(_f.sum()):,} '
                      f'fitted arcs, rejected included -- the unselected view:',
                      flush=True)
                print(f'DEBUG:            per arc   rate p50 '
                      f'{np.median(_prv[_f]):.3f} p90 '
                      f'{np.percentile(_prv[_f], 90):.3f} mm/yr   height p50 '
                      f'{np.median(_prh[_f]):.2f} p90 '
                      f'{np.percentile(_prh[_f], 90):.2f} m', flush=True)
        _gt = np.asarray(gtake, float)
        _gt = _gt[np.isfinite(_gt)]
        if len(_gt):
            # WHAT THE SOLVE ACTUALLY RESTED ON. The cap keeps a node's best
            # arcs, so the solved set sits far above the threshold that let
            # them in, and judging the residual against `threshold` judges a
            # population that was never used.
            print(f'DEBUG:          solved arcs gamma p50 {np.median(_gt):.3f}'
                  f'  p10 {np.percentile(_gt, 10):.3f}'
                  f'  p90 {np.percentile(_gt, 90):.3f}'
                  f'   (threshold was {float(threshold):.2f})', flush=True)
        print(f'DEBUG:          PS closure by component, SURVIVING arcs only '
              f'(selected for agreeing -- optimistic; the 5 largest):',
              flush=True)
        for _z in sorted(order_prio, key=lambda z: -comps[z][0])[:5]:
            _kk2 = comps[_z][2]
            _nodes = sel[_kk2]
            _mask = np.zeros(N, dtype=bool); _mask[_nodes] = True
            _sel_a = _mask[ai] & _mask[aj]
            if not _sel_a.any():
                continue
            _cv, _ch = _rv[_sel_a], _rh[_sel_a]
            # THE TAIL, NOT JUST THE EXTREME. The rejection cuts at a multiple
            # of a robust scale, so the largest survivor is near that bar by
            # construction and says nothing on its own; how many arcs sit out
            # there is the question the bar cannot answer.
            # WHERE THE BAD ARCS ARE. Most arcs close far better than their
            # own coherence implies, so the ones that do not are a separate
            # population rather than the tail of one. If they concentrate on a
            # few nodes the fault is those nodes; if they are spread evenly it
            # is the arc fit.
            _bd = np.isfinite(_ch) & (_ch > err_dh)
            if _bd.any():
                _ea = ai[_sel_a]; _eb = aj[_sel_a]
                _bc = np.bincount(np.r_[_ea[_bd], _eb[_bd]], minlength=N)
                _tc = np.bincount(np.r_[_ea, _eb], minlength=N)
                _bn = _bc[_nodes]; _tn = _tc[_nodes]
                _sv = np.sort(_bn)[::-1]
                _tp = max(1, int(round(0.05 * len(_bn))))
                _fr = np.where(_tn > 0, _bn / np.maximum(_tn, 1), np.nan)
                print(f'DEBUG:             bad arcs (>{err_dh:g} m): '
                      f'{int(_bd.sum()):,}'
                      f' on {int((_bn > 0).sum()):,} of {len(_bn):,} nodes;'
                      f' worst 5% of nodes carry '
                      f'{_sv[:_tp].sum() / max(_sv.sum(), 1):.0%};'
                      f' per-node share p50 {np.nanmedian(_fr):.1%}'
                      f' p90 {np.nanpercentile(_fr, 90):.1%}'
                      f' max {np.nanmax(_fr):.0%}', flush=True)
            _cf = np.isfinite(_ch)
            if _cf.any():
                _hh = _ch[_cf]
                print(f'DEBUG:             height tail  p90 '
                      f'{np.percentile(_hh, 90):.2f}  p99 '
                      f'{np.percentile(_hh, 99):.2f} m   '
                      f'over {err_dh:g} m: {int((_hh > err_dh).sum()):,} '
                      f'({(_hh > err_dh).mean():.1%})   over '
                      f'{2 * err_dh:g} m: {int((_hh > 2 * err_dh).sum()):,} '
                      f'({(_hh > 2 * err_dh).mean():.2%})',
                      flush=True)
            # PER NODE, OVER THIS COMPONENT'S ARCS ONLY -- the same arcs the
            # per-arc figures above describe. Accumulating over every arc
            # would pull in ones reaching outside the component, including
            # nodes below the survival floor that never received a datum, and
            # a node's mean could then exceed the worst arc it averages.
            _acc = np.zeros(N); _cnt2 = np.zeros(N); _acch = np.zeros(N)
            np.add.at(_acc, ai[_sel_a], _cv); np.add.at(_acc, aj[_sel_a], _cv)
            np.add.at(_acch, ai[_sel_a], _ch); np.add.at(_acch, aj[_sel_a], _ch)
            np.add.at(_cnt2, ai[_sel_a], 1.0); np.add.at(_cnt2, aj[_sel_a], 1.0)
            # ONE MASK FOR VALUES AND SUPPORTS: compacting the values
            # while the supports stay in node order pairs each scatter with
            # some other node's support once any node holds no arc
            _fin2 = _cnt2[_nodes] > 0
            _cn = (_acc[_nodes] / np.maximum(_cnt2[_nodes], 1.0))[_fin2]
            _cnh = (_acch[_nodes] / np.maximum(_cnt2[_nodes], 1.0))[_fin2]
            _deg2 = _cnt2[_nodes][_fin2]
            if not len(_cn):
                continue
            # PER ARC and PER NODE are different scales and are labelled as
            # such: a node averages its own arcs, so its worst is always
            # milder than the worst single arc. Printing both unlabelled on
            # one line reads as a contradiction.
            print(f'DEBUG:           label {label_of[_z]}  size {comps[_z][0]:>6,}'
                  f'  arcs {int(_sel_a.sum()):>7,}', flush=True)
            print(f'DEBUG:             per arc   rate p50 {np.median(_cv):.3f}'
                  f' max {_cv.max():.3f} mm/yr'
                  f'   height p50 {np.median(_ch):.2f} max {_ch.max():.2f} m',
                  flush=True)
            print(f'DEBUG:             per node  rate p50 {np.median(_cn):.3f}'
                  f' max {_cn.max():.3f} mm/yr'
                  f'   over {err_dv:g} mm/yr: {int((_cn > err_dv).sum())} '
                  f'of {len(_cn)}',
                  flush=True)
            # THE OUTPUT, NOT THE INPUT. What a caller receives is the node
            # value, and its precision is the arc scatter divided by the
            # support that averaged it -- the arcs are the measurements, the
            # nodes are the answer, and only the second is delivered.
            if len(_cnh) and _deg2.max() > 0:
                _sd = np.sqrt(np.maximum(_deg2, 1.0))
                print(f'DEBUG:             per node  height p50 '
                      f'{np.median(_cnh):.2f} max {_cnh.max():.2f} m'
                      f'   over {err_dh:g} m: '
                      f'{int((_cnh > err_dh).sum()):,} of {len(_cnh):,}',
                      flush=True)
                print(f'DEBUG:             node precision (arc scatter / '
                      f'sqrt support, support p50 {np.median(_deg2):.0f}): '
                      f'rate {np.median(_cn / _sd):.4f} mm/yr   '
                      f'height {np.median(_cnh / _sd):.3f} m',
                      flush=True)

    if debug:
        # THE ARCS THEMSELVES, so the network's self-consistency can be
        # measured on the solve that actually ran rather than on a copy of it.
        # Set AFTER the stats dict is rebuilt above, or they would be wiped.
        # THE ARCS TO A FILE WHEN ASKED. `.stats` is per-thread, so a caller in
        # the main thread cannot read what a dask worker thread wrote; a dump
        # is the only way to get the solved network out for offline analysis.
        _dump = __import__('os').environ.get('INSARDEV_DUMP_ARCS')
        if _dump:
            # ONE FILE PER NETWORK: under union=False every chunk solves its
            # own, at once, so the name carries the network's first node
            _dump = f'{_dump}_{int(iy.min())}_{int(ix.min())}'
            np.savez(_dump, ai=ai, aj=aj, dh=dh, dv=dv,
                     g=np.asarray(gtake, dtype=np.float32),
                     z=_dbg_z, resid=_dbg_rv,
                     node_iy=iy, node_ix=ix, node_index=sel[kk],
                     vel=vel[sel][kk], hgt=hgt[sel][kk])
            print(f'DEBUG: dumped {len(ai):,} network arcs to {_dump}', flush=True)
        _fit_stats.update(
            arc_i=ai.copy(), arc_j=aj.copy(),
            arc_dh=dh.copy(), arc_dv=dv.copy(),
            arc_gamma=np.asarray(gtake, dtype=np.float32).copy(),
            # arcs index the FULL node space; these map it to the reported
            # arrays, and to the raster, so an arc can be located
            arc_gate_passes=_gate_passes,
            arc_z=_dbg_z.copy(),
            arc_sigma_v=np.full(len(ai), float(s_v), np.float64),
            arc_resid_irls=_dbg_rv.copy(),
            node_index=sel[kk].copy(),
            node_iy=iy.copy(), node_ix=ix.copy(),
            node_vel=vel.copy(), node_hgt=hgt.copy())

    # THE NETWORK SOLUTION, and nothing raster-shaped: the caller writes these
    # into whichever block holds each node. Full precision -- the attachment
    # reads a node's model to give a DS its datum, so rounding here would
    # round every DS that leans on it.
    return dict(iy=sy_[kk], ix=sx_[kk], label=lab_all,
                U=np.ascontiguousarray(Un[:, sel][:, kk]),
                vel=vel[sel][kk], hgt=hgt[sel][kk], coh=gnode[sel][kk],
                sea=(anr[sel][kk] + 1j * ani[sel][kk]),
                # WHAT THE SOLVE COUNTED, carried with the table. The stats
                # holder is per thread, so under `union=True` -- where the
                # network runs in one process and the attachments in others
                # -- they cannot be read where they were written.
                stats=dict(_fit_stats))


def _3d_ps_attach(scenes, q, nodes, date_values, *, spacing, bperp=None,
                  window=(32, 128), threshold=0.5, cell=(2, 8), geometry,
                  budget=None, level=1, max_dh=25.0, max_dv=25.0,
                  step_dh=8.0, step_dv=2.0, consensus, iterations=8,
                  err_dh=4.0, err_dv=1.0, threads=None, debug=False,
                  out_stats=None):
    """PASS 2, per block: the network written out, and the DS hung off it.

    `nodes` is the solved table from `_3d_ps_network`, with positions in THIS
    block's index space. Under `union=True` those nodes were solved together
    with other blocks' -- so the datum crossing the seam is the point -- but
    the pixels never left their own block.
    """
    from .utils_dask import cluster_cores
    _nth = max(1, int(threads) if threads else cluster_cores()[1])
    _ma = _3d_consensus(consensus)
    _ii = max(1, int(iterations))
    # A VIEW IS ENOUGH: only the shape and fancy indexing are read, so the
    # per-chunk driver's strided owned sub-block costs no copy here
    S = np.asarray(scenes, dtype=np.complex64)
    n, ny, nx = S.shape
    wy, wx, pey, pex = _3d_windows(window)
    lab_out = np.full((ny, nx), -1, dtype=np.int8)
    vel_out = np.full((ny, nx), np.nan, dtype=np.float32)
    hgt_out = np.full((ny, nx), np.nan, dtype=np.float32)
    coh_out = np.full((ny, nx), np.nan, dtype=np.float32)
    sea_out = np.full((ny, nx), np.nan + 1j * np.nan, dtype=np.complex64)
    # THE LEVEL THAT PRODUCED EACH PIXEL: 0 for a PS node, 1 for a DS attached
    # to the network, n for one attached to the level n-1 DS. -1 where nothing
    # was measured. Carried so the ladder can be looked at, not just counted.
    lvl_out = np.full((ny, nx), -1, dtype=np.int8)
    # CLEARED BEFORE THE EARLY RETURN, so a block that solves nothing
    # cannot leave the previous block's DS table visible on this thread.
    _fit_stats.reset(nodes=0, arcs=0, dropped=0,
                                 components=[], fill_order=[])
    if nodes is None or len(np.asarray(nodes['iy'])) == 0:
        return lab_out, vel_out, hgt_out, sea_out, coh_out, lvl_out
    t, ele2phase, meter2rad, car = _3d_fit_frame(date_values, bperp,
                                                 geometry, n)
    # stated in metres and mm/yr, used in radians -- see the network
    _err_h = float(err_dh) * meter2rad
    _err_v = float(err_dv) * meter2rad / 1e3
    iy = np.asarray(nodes['iy'])
    ix = np.asarray(nodes['ix'])
    Un = np.ascontiguousarray(nodes['U'], dtype=np.complex64)
    lab_all = np.asarray(nodes['label'], dtype=np.int8)
    vel = np.asarray(nodes['vel'])
    hgt = np.asarray(nodes['hgt'])
    gnode = np.asarray(nodes['coh'])
    anr = np.asarray(np.real(nodes['sea']))
    ani = np.asarray(np.imag(nodes['sea']))
    # the table arrives already reduced to the kept nodes, so the selections
    # the attachment applies are the identity -- kept so the code below reads
    # the same whether it ran here or in a network solved somewhere else
    sel = np.arange(len(iy))
    kk = np.arange(len(iy))
    # Each node belongs to exactly one component, so writing them cannot
    # contest a pixel.
    # A NODE MAY STAND OUTSIDE THIS BLOCK. Under `union=True` the table is the
    # whole scene's, and a node beyond these bounds is still a partner a DS
    # here can reach -- the PS extent is far wider than a chunk -- but its
    # model belongs to the block that holds its pixel, and only that block
    # writes it. This is the reach a per-block network cannot offer: without
    # it a DS near a chunk edge is judged on the handful of nodes its own
    # chunk happens to contain.
    _own = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    _oy, _ox = iy[_own], ix[_own]
    lab_out[_oy, _ox] = lab_all[_own]
    vel_out[_oy, _ox] = vel[_own].astype(np.float32)           # rad/yr
    lvl_out[_oy, _ox] = np.where(np.isfinite(vel[_own]), 0, -1)  # PS nodes
    hgt_out[_oy, _ox] = hgt[_own].astype(np.float32)           # rad
    coh_out[_oy, _ox] = gnode[_own].astype(np.float32)
    sea_out[_oy, _ox] = (anr[_own] + 1j * ani[_own]).astype(np.complex64)
    if q is None:
        q = np.full((ny, nx), np.nan, dtype=np.float32)
    # THE STATS BELONG TO WHOEVER SOLVED, and are REBUILT here for every
    # block. The holder is per thread, so under `union=True` -- the network
    # in one process, the attachments in others -- they cannot be read where
    # they were written, and a worker thread that fitted a previous block
    # still holds that block's dict. Seeded from the table this block was
    # handed, so what is reported is what this block actually wrote.
    _fit_stats.reset(nodes=0, arcs=0, dropped=0, components=[],
                                 fill_order=[])
    _fit_stats.update(nodes.get('stats') or {})
    _fit_stats['iy'] = iy
    _fit_stats['ix'] = ix
    # ---- attach the DS to the network ----------------------------------
    # THE PS EXTENT IS THE REACH, NOT THE DS WINDOW. A PS is defined by holding
    # a coherent arc out to the PS extent -- that is what separates it from a DS
    # -- so limiting the attachment to the DS window contradicts the test that
    # selected the partner in the first place.
    #
    # The DS window was justified by a reach table showing DS->PS coherence
    # dying past ~200 m, but that decays that fast only for a RAW inner product.
    # These arcs are FITTED (`_3d_arc_fit` solves the differential height and
    # rate), and a fitted arc carries much further -- the same distinction
    # arcs() documents for the long PS test, where raw coherence selected
    # nothing at km range and the fitted test still found pairs.
    #
    # It matters because the DS window holds too few PS to vote with: the
    # consensus rule needs `min_agreeing` partners and the window rarely supplies
    # them, so DS were being rejected for the reach of the search rather than
    # for the quality of their arcs.
    #
    # BEST, not nearest. Where several nodes are in reach the arc coherence says
    # which one actually carries the datum, and distance does not: gamma over a
    # DS window varies far more between partners than with the few tens of
    # metres separating them.
    #
    # A DS that clears the gate inherits its partner's height, rate, seasonal
    # AND component label, so it lands on the same datum as the node -- that is
    # the whole point of attaching rather than solving it alone. One that does
    # not clear it stays NaN: a DS with no coherent path to the network has no
    # datum, and a value written without one would be a different network's.
    if level >= 1:
        cand_ds = np.isfinite(q) & (q >= float(threshold))
        cand_ds[_oy, _ox] = False                        # nodes are not DS here
        dy_, dx_ = np.where(cand_ds)
        if len(dy_):
            ny_ps, nx_ps = iy[sel][kk], ix[sel][kk]
            # THE PARTNERS ARE THE DS WINDOW CENTRED ON THE CANDIDATE. The DS
            # window is the scale over which the atmosphere is taken to be
            # common, so it is also the scale over which an arc means
            # anything; a candidate takes every node within half a window of
            # it, the one rule every partner search here draws, and fits
            # each one. The set is bounded by the window, so the cost is a
            # property of the window rather than of the scene.
            _fit_stats['ds_candidates'] = int(len(dy_))
            _fit_stats['ds_reached'] = int(len(dy_))
            # unit phasors once for the candidates and once for the nodes,
            # rather than re-slicing the scene inside every batch
            _ad = np.abs(S[:, dy_, dx_])
            Ud_all = np.ascontiguousarray(
                np.where(_ad > 0, S[:, dy_, dx_] / np.where(_ad > 0, _ad, 1),
                         0).astype(np.complex64))
            del _ad
            Ups = np.ascontiguousarray(Un[:, sel][:, kk])
            if len(ny_ps):
                # the DS window names the neighbourhood; the two extent
                # slots are carried for shape and are not a reach any more
                _pos = (dy_, dx_, ny_ps, nx_ps, 0, 0, wy, wx)
                _t_ds = time.monotonic()
                ksrc, ktgt, ga, dha, dva, dsa, good = _3d_shortlist_ds_ps(
                    Ud_all, Ups, lab_all, len(dy_),
                    ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
                    budget, iterations, _ma, threshold,
                    stats=_fit_stats, prefix='ds_', debug=debug,
                    # THE PS ARE THE FIXED LAYER: solved onto one datum, so
                    # a partner's value is its own plus what the arc measures.
                    fix_h=hgt[sel][kk], fix_v=vel[sel][kk], threads=_nth,
                    pos=_pos)
                _fit_stats['ds_arcs'] = int(len(ksrc))
                _lap, _t_ds = _3d_lap(_t_ds)
                _fit_stats['ds_fit_s'] = _lap
                # THE PARTNERS ARE MEASUREMENTS, SO THEY ARE JUDGED THE WAY
                # THE ARCS ARE. Each partner gives the DS a complete answer, so
                # several partners are repeated measurements of one quantity --
                # the same situation as a node's arcs, and it gets the same
                # test: every one of the best `min_agreeing` within the stated
                # bound of their median, before the value counts as measured
                # at all. With one an error is invisible
                # and with two it cannot be localised, whether the two are
                # arcs or partners.
                #
                # Where the pixel is a clean scatterer the partners agree and
                # this changes nothing. Where it is a MIXTURE its phase belongs
                # to no single target, the arcs fit different parameters, and
                # the strongest one is as likely to hold the wrong component as
                # the right one.
                #
                # The representative is the best surviving arc, not the fitted
                # centre: height, seasonal and component label still come from
                # one real partner exactly as before -- only WHICH partner
                # changes. A fitted centre would sit between partners and be
                # backed by none of them.
                if debug:
                    # THE WHOLE SHORTLIST, not just the partner that won. A
                    # joint DS solve needs every DS->PS equation, and the star
                    # attachment keeps only one per pixel.
                    _gsl = np.flatnonzero(good)
                    _fit_stats.update(
                        dsarc_src=ksrc[_gsl].copy(),        # index into dy_/dx_
                        dsarc_ds_iy=dy_.copy(), dsarc_ds_ix=dx_.copy(),
                        dsarc_tgt=ktgt[_gsl].copy(),        # index into the nodes
                        dsarc_dv=dva[_gsl].copy(),
                        dsarc_dh=dha[_gsl].copy(),
                        dsarc_gamma=ga[_gsl].copy())
                v_abs = vel[sel][kk][ktgt] + dva
                h_abs = hgt[sel][kk][ktgt] + dha
                # A LOCAL DICT, NOT THE MODULE'S. `stats` is a function
                # attribute shared by every task in the worker PROCESS, and
                # the centres come back out of it to become the DS values --
                # so with more than one thread per worker a block reads the
                # centres of whichever block wrote last. That is a wrong
                # answer when the lengths happen to match and an IndexError
                # when they do not. Results must not travel through global
                # state; only the diagnostics may, and they are merged after.
                _cst = {}
                first, votes, _okds = _3d_partner_consensus(
                    ksrc, ga, good, v_abs, len(dy_), _ma, _err_v, _ii,
                    h_abs=h_abs, err_h=_err_h,
                    labels=(lab_all[ktgt] if debug else None),
                    stats=_cst, prefix='ds_')
                _fit_stats.update(_cst)
                _lap, _t_ds = _3d_lap(_t_ds)
                _fit_stats['ds_consensus_s'] = _lap
                if len(first):
                    ds_i, ps_i = ksrc[first], ktgt[first]
                    yy2, xx2 = dy_[ds_i], dx_[ds_i]
                    # THE VALUE IS THE CONSENSUS, NOT ITS BEST WITNESS. The
                    # partners that voted are repeated measurements of this
                    # pixel, and the vote already found which of them agree;
                    # taking the highest-gamma one and dropping the rest
                    # spends the redundancy on the test and keeps none of it
                    # for the answer. The component still comes from the best
                    # partner: a label is named, not averaged.
                    if debug:
                        # ARE THE PARTNERS INDEPENDENT OF EACH OTHER? The
                        # independence cell is enforced between the CANDIDATE
                        # and each partner, never between the partners: five
                        # of them inside one cell are one sample of the ground
                        # counted five times, and they agree because they are
                        # the same measurement, not because the value is right.
                        _va0 = _cst.get('ds_vote_arcs')
                        if _va0 is not None and len(_va0):
                            _vk = np.asarray(_va0, np.int64)
                            _cy0, _cx0 = int(cell[0]), int(cell[1])
                            _pc = (ny_ps[ktgt[_vk]] // max(_cy0, 1)
                                   ).astype(np.int64) * (1 << 20) + \
                                  (nx_ps[ktgt[_vk]] // max(_cx0, 1))
                            _o0 = np.lexsort((_pc, ksrc[_vk]))
                            _ss, _pp = ksrc[_vk][_o0], _pc[_o0]
                            _new = np.r_[True, (_ss[1:] != _ss[:-1]) |
                                         (_pp[1:] != _pp[:-1])]
                            _ncell = np.bincount(_ss[_new], minlength=len(dy_))
                            _nvote = np.bincount(ksrc[_vk], minlength=len(dy_))
                            _hv0 = _nvote > 0
                            _fit_stats['lvl_pcells'] = \
                                _lvl_stat(_ncell[_hv0], _LE12)
                            _fit_stats['lvl_pvotes'] = \
                                _lvl_stat(_nvote[_hv0])
                            # HOW FAR the agreeing partners actually are. The
                            # PS extent is the reach, and an isolated DS takes
                            # whatever it can reach -- five properly separated
                            # partners, good coherence, and every arc far past
                            # the separation where a differential model still
                            # describes one piece of ground.
                            _dyp = (ny_ps[ktgt[_vk]] - dy_[ksrc[_vk]]) \
                                * float(spacing[0])
                            _dxp = (nx_ps[ktgt[_vk]] - dx_[ksrc[_vk]]) \
                                * float(spacing[1])
                            _rr = np.sqrt(_dyp**2 + _dxp**2)
                            _sd = np.zeros(len(dy_)); _sn = np.zeros(len(dy_))
                            np.add.at(_sd, ksrc[_vk], _rr)
                            np.add.at(_sn, ksrc[_vk], 1.0)
                            # NO FIXED BOUND. What counts as a far partner
                            # is set by the window and the pixel spacing, not
                            # by a distance measured once on one stack: the
                            # same metres are well inside the reach on one
                            # geometry and past it on another. The percentiles
                            # and the max say it for whatever area this is.
                            _fit_stats['lvl_parcm'] = _lvl_stat(
                                _sd[_hv0] / np.maximum(_sn[_hv0], 1))
                    _cv = _cst.get('ds_centre_v')
                    _ch = _cst.get('ds_centre_h')
                    _si = ksrc[first]
                    h_ds = (hgt[sel][kk][ps_i] + dha[first] if _ch is None
                            else np.asarray(_ch)[_si])
                    lab_ds = lab_all[ps_i]
                    v_ds = (vel[sel][kk][ps_i] + dva[first] if _cv is None
                            else np.asarray(_cv)[_si])
                    if debug:
                        # ---- DS CLOSURE ---------------------------------
                        # Every accepted partner against the value the pixel
                        # adopted. The PS are the fixed layer, so a DS holds
                        # no free datum and each partner is a complete answer
                        # for it -- their spread IS the attachment's
                        # consistency. Unlike the network's, this set is not
                        # selected for agreeing, so it needs no second view.
                        _vad = np.full(len(dy_), np.nan)
                        _had = np.full(len(dy_), np.nan)
                        _vad[ds_i] = v_ds
                        _had[ds_i] = h_ds
                        _va = _cst.get('ds_vote_arcs')
                        _m2 = np.zeros(len(ksrc), dtype=bool)
                        if _va is not None and len(_va):
                            _m2[np.asarray(_va, np.int64)] = True
                        _m2 &= np.isfinite(_vad[ksrc])
                        if _m2.any():
                            _crv = np.abs(
                                (vel[sel][kk][ktgt[_m2]] + dva[_m2])
                                - _vad[ksrc[_m2]]) / meter2rad * 1e3
                            _crh = np.abs(
                                (hgt[sel][kk][ktgt[_m2]] + dha[_m2])
                                - _had[ksrc[_m2]]) / meter2rad
                            # PER PIXEL, the same way the network reports per
                            # node: a DS averages its own partners, so its
                            # worst is milder than the worst single partner and
                            # the two scales must be labelled apart.
                            _na = np.zeros(len(dy_)); _nc = np.zeros(len(dy_))
                            _nb = np.zeros(len(dy_))
                            np.add.at(_na, ksrc[_m2], _crv)
                            np.add.at(_nb, ksrc[_m2], _crh)
                            np.add.at(_nc, ksrc[_m2], 1.0)
                            _hv = _nc > 0
                            _pv2 = _na[_hv] / _nc[_hv]
                            _ph2 = _nb[_hv] / _nc[_hv]
                            # THE SAMPLES, for the level's reducer. The
                            # summaries below describe THIS block; percentiles
                            # do not average, so a level spanning many blocks
                            # has to pool the values themselves.
                            _fit_stats.update(
                                lvl_clo_arc_v=_crv.astype(np.float32),
                                lvl_clo_arc_h=_crh.astype(np.float32),
                                lvl_clo_ds_v=_pv2.astype(np.float32),
                                lvl_clo_ds_h=_ph2.astype(np.float32))
                            _fit_stats.update(
                                ds_clo_n=int(_m2.sum()),
                                ds_clo_rv=float(np.nanmedian(_crv)),
                                ds_clo_rv90=float(np.nanpercentile(_crv, 90)),
                                ds_clo_rh=float(np.nanmedian(_crh)),
                                ds_clo_rh90=float(np.nanpercentile(_crh, 90)),
                                ds_clo_pix=int(_hv.sum()),
                                ds_clo_nv=float(np.median(_pv2)),
                                ds_clo_nvmax=float(_pv2.max()),
                                ds_clo_nh=float(np.median(_ph2)),
                                ds_clo_nhmax=float(_ph2.max()),
                                ds_clo_over1=int((_pv2 > err_dv).sum()),
                                ds_clo_at=float(err_dv))

                    # ---- SOLVE THE DS, EACH FROM ITS OWN ARCS ---------
                    # Each DS has just been validated against `consensus`
                    # PS partners. The attachment's equations are KEPT rather
                    # than reduced to one partner each, and every pixel is
                    # solved from all of them against the FIXED network:
                    #
                    #     DS_i - PS_p = dv_ip     the PS are FIXED
                    #
                    # The PS stay fixed because they are the certified layer
                    # and define the datum: a hundred thousand weak DS solved
                    # WITH a few hundred nodes would outvote the network that
                    # anchors them. Fixing them also leaves the DS system
                    # with no free constant of its own. Nothing ties one DS
                    # to another: a DS value carries noise, not a datum.
                    _att = np.full(len(dy_), -1, dtype=np.int64)
                    _att[ds_i] = np.arange(len(ds_i))
                    _e = np.flatnonzero(good & (_att[ksrc] >= 0))
                    _ei, _ep = _att[ksrc[_e]], ktgt[_e]
                    _st2 = _fit_stats
                    v_ds, h_ds, _anc1 = _3d_ds_solve(
                        len(ds_i), _ei, _ep, dva[_e], dha[_e], ga[_e],
                        vel[sel][kk], hgt[sel][kk], _err_v, _err_h, _ii)
                    # THE SAME RULE THE NODES GET: enough of a pixel's own
                    # equations must survive the gate. One that keeps too few
                    # was not confirmed by the network, and a value carried by
                    # what is left would be a compromise nothing backs.
                    # the same rule at level 1: five surviving DS->PS
                    # equations, not five equations of any kind
                    _live2 = _anc1 >= _ma
                    if debug:
                        _st2['lvl_passed'] = int((_anc1 >= _ma).sum())
                    _st2['ds_unconfirmed'] = int((~_live2).sum())
                    v_ds = np.where(_live2, v_ds, np.nan)
                    h_ds = np.where(_live2, h_ds, np.nan)

                    # THE LABEL IS PART OF THE ANSWER, SO IT IS GATED WITH
                    # IT. `conncomp` says which datum a value belongs to, and a
                    # pixel the joint solve declined to place has no value and
                    # therefore no datum -- writing the partner's label anyway
                    # reports membership of a component the solve refused to
                    # grant, and a caller masking on `conncomp` rather than on
                    # the value would take it.
                    lab_out[yy2, xx2] = np.where(_live2, lab_ds, -1)
                    vel_out[yy2, xx2] = v_ds.astype(np.float32)
                    lvl_out[yy2, xx2] = np.where(_live2, 1, -1)  # DS on the PS
                    hgt_out[yy2, xx2] = h_ds.astype(np.float32)
                    # coherence and seasonal are part of the answer too: a
                    # pixel the solve refused has no measurement, and a finite
                    # gamma or annual on a NaN velocity reads as one
                    coh_out[yy2, xx2] = np.where(
                        _live2, ga[first], np.nan).astype(np.float32)
                    sea_out[yy2, xx2] = np.where(
                        _live2,
                        (anr[sel][kk][ps_i] + dsa[first].real)
                        + 1j * (ani[sel][kk][ps_i] + dsa[first].imag),
                        np.nan + 1j * np.nan).astype(np.complex64)
                    # THE NODE TABLE IS A RESULT, so it also goes somewhere
                    # this CALL owns. Level 2 stands on it, and read back from
                    # a process-global dict a block can inherit the nodes of
                    # whichever block shared its worker.
                    _dsn = dict(
                        ds_attached=int(len(first)),
                        ds_partners=np.bincount(
                            ksrc[good], minlength=len(dy_))[ksrc[first]],
                        ds_gamma=ga[first].copy(),
                        ds_votes=np.asarray(votes, dtype=np.int32),
                        ds_iy=yy2.copy(), ds_ix=xx2.copy(), ds_label=lab_ds,
                        ds_height_rad=h_ds.astype(np.float32),
                        ds_velocity_rad_yr=v_ds.astype(np.float32),
                        # the annual travels with the DS as well: it is fitted
                        # on the attaching arc and made absolute by its
                        # partner's own value, exactly as height and rate are
                        ds_seasonal_rad=((anr[sel][kk][ps_i] + dsa[first].real)
                                         + 1j * (ani[sel][kk][ps_i]
                                                 + dsa[first].imag)
                                         ).astype(np.complex64))
                    _dsn['lvl_cands'] = int(len(dy_))
                    _dsn['lvl_fixed'] = int(len(np.unique(ktgt)))
                    _dsn['lvl_arcs'] = int(len(ksrc))
                    _dsn['lvl_left'] = int(_fit_stats.get(
                        'ds_admissible', 0))
                    _dsn['lvl_attached'] = int(len(first))
                    for _k in ('lvl_clo_arc_v', 'lvl_clo_arc_h',
                               'lvl_clo_ds_v', 'lvl_clo_ds_h',
                               'lvl_passed',
                               'lvl_pcells', 'lvl_pvotes',
                               'lvl_parcm',
                               ):
                        _v = _fit_stats.get(_k)
                        if _v is not None:
                            _dsn[_k] = _v
                    _fit_stats.update(_dsn)
                    if out_stats is not None:
                        out_stats.update(_dsn)
    if debug:
        _s = _fit_stats
        _att = int(_s.get('ds_attached', 0))
        _cnd = int(_s.get('ds_candidates', 0))
        if level < 1:
            print('DEBUG: DS       level=0 -- the PS network only', flush=True)
        elif _cnd:
            print(f'DEBUG: DS       {_cnd:,} candidates, '
                  f'{int(_s.get("ds_reached", 0)):,} reached a node over '
                  f'{int(_s.get("ds_searched_arcs", 0)):,} arcs fitted, '
                  f'{int(_s.get("ds_arcs", 0)):,} kept'
                  f'   fit {_s.get("ds_fit_s", 0.0):.1f}s'
                  f' + consensus {_s.get("ds_consensus_s", 0.0):.1f}s',
                  flush=True)
            # THE BLOCK'S NUMBERS ARE NOT THE LEVEL'S. Level 1 runs once per
            # block, so printing here repeats every line as many times as
            # there are chunks -- unreadable at a few dozen, and still not the
            # answer, since the level's yield is the sum and its error
            # distribution the pooled samples. Recorded for the level's
            # reducer, which prints once when every block has finished.
            _s['lvl_no_consensus'] = int(_s.get('ds_no_consensus', 0))
            _s['lvl_too_few'] = int(_s.get('ds_too_few', 0))
            _s['lvl_multi_comp'] = int(_s.get('ds_multi_component', 0))
            _s['lvl_straddled'] = int(_s.get('ds_shortlist_straddled', 0))
            _s['lvl_cross_votes'] = int(_s.get('ds_cross_component_votes') or 0)
            _g = _s.get('ds_gamma')
            if _g is not None and len(_g):
                _s['lvl_gamma'] = _lvl_stat(_g)
        else:
            print('DEBUG: DS       no candidates cleared the threshold',
                  flush=True)
        # the closure of this level is reported once, by the level's reducer,
        # over the pooled samples of every block -- see `lvl_clo_*` above
        # the total is the LEVEL's, and one block does not know it
        _s['lvl_ps'] = int(len(_s.get('iy', ())))
        # THE CALLER'S DICT IS WHAT TRAVELS. The level report reads the
        # `out_stats` each block handed back, not this thread's holder, so
        # the keys set here reach it only if they are copied across.
        if out_stats is not None:
            out_stats.update({k: v for k, v in _s.items()
                              if k.startswith('lvl_')})
    return lab_out, vel_out, hgt_out, sea_out, coh_out, lvl_out


# PASS 1's BLOCKING (`_cascade_count_max`): how many own rows share one
# product, and how many own pixels a tile is wide. Neither changes a number --
# every pair is the same dot product whatever the blocking -- they set how much
# of each product the window masks out (the span is the tile plus a window) and
# whether it is still in cache when it is counted. Measured on GasteinValley
# chunks: 40 px tiles against the window's 321 px, 64 rows a batch.
_CASCADE_ROWS = 64


_CASCADE_COLS = 40


@_numba.njit(nogil=True, cache=True, inline='always')
def _cascade_run(trow, crow, base, a, b, thrb, xi, cr, mr, vb, vbi, oc, om):
    """One run of admissible pairs -- one own pixel, one neighbour row, the
    offsets dx in [a, b] -- into one floor's accumulators.

    A SCORE IS NEVER NEGATIVE, so its float32 bit pattern orders as its value
    does: the threshold test and both maxima run on the int32 bits, read back
    from a buffer the scores are written to once. A float maximum carried
    across the loop kept numba's loop scalar; the integer one vectorizes, and
    the answer is the same bits.
    """
    if a > b:
        return oc, om
    n = b - a + 1
    # SLICED ONCE, INDEXED FROM ZERO: an offset index the compiler cannot prove
    # non-negative carries numba's wraparound branch on every access, and that
    # branch alone kept these loops scalar
    tr = trow[base + a:base + a + n]
    cs = crow[base + a:base + a + n]
    for k in range(n):
        vb[k] = tr[k] * tr[k] + cs[k] * cs[k]
    c1 = cr[xi + a:xi + a + n]
    m1 = mr[xi + a:xi + a + n]
    for k in range(n):
        x = vbi[k]
        h = np.int32(1) if x >= thrb else np.int32(0)
        oc += h
        om = max(om, x)
        c1[k] += h
        m1[k] = max(m1[k], x)
    return oc, om


@_numba.njit(nogil=True, cache=True)
def _cascade_row(out, jlo, jhi, r, w, hx, cy, cx, f2y, f2x, thrb, y, x0,
                 valid, cnt2, mx2, cnte, mxe, oc2, om2, oce, ome, vb, vbi):
    """Neighbour row `y + r` against own rows [jlo, jhi) of a batch, GIL-free.

    `out` holds, per own row, `w` real then `w` imaginary rows of coherent sums
    over the tile's span. Every pair is visited ONCE: the pairs the longer
    floor admits go to (cnt2, mx2), the pairs only the cell floor admits to
    (cnte, mxe), and the cell floor is their sum and their maximum. The masks
    are index ranges, so a masked pair is never read, and the partner credits
    land on the neighbour row's contiguous accumulators. Maxima are int32 bit
    patterns (`_cascade_run`); the own pixel's running totals live in
    (oc2, om2, oce, ome) across the batch's neighbour rows.
    """
    nx = cnt2.shape[1]
    yr = y + r
    cr2, mr2, cre, mre = cnt2[yr], mx2[yr], cnte[yr], mxe[yr]
    for j in range(jlo, jhi):
        dy = r - j
        yo = y + j
        o = (j - jlo) * 2 * w
        for i in range(w):
            xi = x0 + i
            # an invalid pixel's phasors are zero: none of its products counts
            if valid[yo, xi] == 0:
                continue
            # one-sided: the own row only to the right; past the raster the
            # partners are zero padding and change nothing
            lo = 1 if dy == 0 else -hx
            if lo < -xi:
                lo = -xi
            hi = hx
            if hi > nx - 1 - xi:
                hi = nx - 1 - xi
            if lo > hi:
                continue
            trow = out[o + i]
            crow = out[o + w + i]
            base = hx + i
            a2 = oc2[j, i]
            b2 = om2[j, i]
            ae = oce[j, i]
            be = ome[j, i]
            for s in range(2):
                # inside the cell the pixel measures itself: its two flanks
                if dy < cy:
                    a, b = (lo, min(hi, -cx)) if s == 0 else (max(lo, cx), hi)
                else:
                    if s == 1:
                        break
                    a, b = lo, hi
                if a > b:
                    continue
                if dy >= f2y:
                    a2, b2 = _cascade_run(trow, crow, base, a, b, thrb, xi,
                                          cr2, mr2, vb, vbi, a2, b2)
                else:
                    # the longer floor leaves out |dx| < f2x as well: that
                    # middle belongs to the cell floor alone
                    a2, b2 = _cascade_run(trow, crow, base, a, min(b, -f2x),
                                          thrb, xi, cr2, mr2, vb, vbi, a2, b2)
                    ae, be = _cascade_run(trow, crow, base, max(a, 1 - f2x),
                                          min(b, f2x - 1), thrb, xi, cre, mre,
                                          vb, vbi, ae, be)
                    a2, b2 = _cascade_run(trow, crow, base, max(a, f2x), b,
                                          thrb, xi, cr2, mr2, vb, vbi, a2, b2)
            oc2[j, i] = a2
            om2[j, i] = b2
            oce[j, i] = ae
            ome[j, i] = be


def _cascade_count_max(S, wy, wx, cell, thr, floor2=None, own=None):
    """Per-pixel coherent-arc COUNT and best-arc coherence, in one sweep.

    The same one-sided rectangular products as `_3d_arcs_kernel`, reduced to
    the two numbers the cascade needs: how many admissible arcs clear `thr`,
    and the best coherence reached. Returns (count int32, best float32) for
    the WHOLE array handed in -- the caller slices owned pixels out.

    `floor2` = (fy, fx): a second, longer arc floor scored from the SAME
    products -- pairs closer than it in both axes are left out -- returning
    (count, best, count2, best2). The PS ranking reads the second pair: the
    number of coherent arcs a pixel holds at more than a cell or two away is
    what tells a scatterer from the mixture copies beside it, which share its
    shortest arcs; the GEMM is the cost and it is paid once.

    `own` = (r0, r1, c0, c1): only those pixels are taken as the near end of a
    pair; the rest of the array is data for their products. A pixel's numbers
    are then complete where every pixel within a window above and beside it is
    own -- the caller hands the region its reads need (`_cascade_pass1`).

    ONE PRODUCT PER NEIGHBOUR ROW, FOR A BATCH OF OWN ROWS. A batch of
    `_CASCADE_ROWS` rows of `_CASCADE_COLS` own pixels shares its neighbour
    rows, so each neighbour row is one product against every own row it is in
    reach of, read in place from the phasor raster: no transposed copy of the
    window per tile (it was 29% of the scan), and a product small enough to be
    counted while it is still in cache. The counting visits only the pairs the
    masks admit, once for both floors (`_cascade_row`).

    NOTHING IS COMPUTED WHERE NOTHING IS VALID. A pixel missing any date has
    zero phasors, so its products are zero and change no count and no maximum.
    A batch with no valid own pixel and a neighbour row with none in reach are
    skipped -- over the GasteinValley burst at least 54% of the scan.

    The numbers are the old ones to the bit: every pair is the same dot
    product, and counts and maxima do not depend on the order they are taken
    in. Measured on GasteinValley chunks (91 dates, window 80 x 320 px), one
    thread: 92% valid 160 -> 23 s, 52% valid 152 -> 14 s, 10% valid 146 -> 4 s.
    """
    n, ny, nx = S.shape
    cy, cx = int(cell[0]), int(cell[1])
    # without a second floor every admitted pair is the first floor's
    f2y, f2x = (0, 0) if floor2 is None else (int(floor2[0]), int(floor2[1]))
    hy, hx = wy // 2, wx // 2
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
    valid = np.zeros((ny, nx), np.uint8)
    slab = max(1, min(ny, int(64 * 1024 * 1024 // max(n * nx * 8, 1))))
    for y0 in range(0, ny, slab):
        y1 = min(y0 + slab, ny)
        blk = S[:, y0:y1, :]
        a = np.abs(blk)
        f = np.isfinite(a) & (a > 0)
        o = f.all(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.where(f, blk / np.where(f, a, 1), 0)
        u *= o[None, :, :]
        Xp[y0:y1, hx:hx + nx, :n] = np.moveaxis(u.real, 0, -1)
        Xp[y0:y1, hx:hx + nx, n:] = np.moveaxis(u.imag, 0, -1)
        valid[y0:y1] = o
        del blk, a, f, o, u
    cum = np.zeros((ny + 1, nx + 1), np.int32)
    np.cumsum(np.cumsum(valid, axis=0, dtype=np.int32), axis=1, out=cum[1:, 1:])
    cnt2 = np.zeros((ny, nx), np.int32)
    mx2 = np.zeros((ny, nx), np.float32)
    cnte = np.zeros((ny, nx), np.int32)
    mxe = np.zeros((ny, nx), np.float32)
    mx2i, mxei = mx2.view(np.int32), mxe.view(np.int32)
    # the threshold as the smallest float32 at or above it: on float32 scores
    # that is exactly the float64 comparison the scores were always held to
    thr2 = (float(thr) * n) ** 2
    t32 = np.float32(thr2)
    if float(t32) < thr2:
        t32 = np.nextafter(t32, np.float32(np.inf))
    thrb = np.int32(t32.view(np.int32))
    vb = np.empty(2 * hx + 1, np.float32)
    vbi = vb.view(np.int32)
    r0, r1, q0, q1 = (0, ny, 0, nx) if own is None else (int(v) for v in own)
    RB, W = _CASCADE_ROWS, _CASCADE_COLS
    A = np.empty((RB * 2 * W, K), dtype=np.float32)
    acc = np.zeros((4, RB, W), np.int32)
    for y in range(r0, r1, RB):
        rb = min(RB, r1 - y)
        R = min(rb + hy, ny - y)
        for x0 in range(q0, q1, W):
            w = min(W, q1 - x0)
            span = w + 2 * hx
            if (cum[y + rb, x0 + w] - cum[y, x0 + w] - cum[y + rb, x0]
                    + cum[y, x0]) == 0:
                continue
            for j in range(rb):
                A1 = Xp[y + j, hx + x0:hx + x0 + w, :]
                o = j * 2 * w
                A[o:o + w] = A1
                A[o + w:o + 2 * w, :n] = A1[:, n:]
                A[o + w:o + 2 * w, n:] = -A1[:, :n]
            acc[...] = 0
            c0, c1 = max(0, x0 - hx), min(nx, x0 + w + hx)
            for r in range(R):
                yr = y + r
                if (cum[yr + 1, c1] - cum[yr, c1] - cum[yr + 1, c0]
                        + cum[yr, c0]) == 0:
                    continue
                jlo, jhi = max(0, r - hy), min(rb, r + 1)
                out = A[jlo * 2 * w:jhi * 2 * w] @ Xp[yr, x0:x0 + span, :].T
                _cascade_row(out, jlo, jhi, r, w, hx, cy, cx, f2y, f2x, thrb,
                             y, x0, valid, cnt2, mx2i, cnte, mxei,
                             acc[0], acc[1], acc[2], acc[3], vb, vbi)
            sl = (slice(y, y + rb), slice(x0, x0 + w))
            cnt2[sl] += acc[0, :rb, :w]
            np.maximum(mx2i[sl], acc[1, :rb, :w], out=mx2i[sl])
            cnte[sl] += acc[2, :rb, :w]
            np.maximum(mxei[sl], acc[3, :rb, :w], out=mxei[sl])
    cnt = cnt2 + cnte
    mx = np.maximum(mx2i, mxei).view(np.float32)
    best = np.sqrt(mx, out=mx) / n
    if floor2 is None:
        return cnt, best
    return cnt, best, cnt2, np.sqrt(mx2, out=mx2) / n


def _3d_check_window_cell(wy, wx, cell, name='fit1d_network', spacing=None):
    """The DS window must span at least FOUR independence cells per dimension.

    A window is a patch of independent ground samples in which a distributed
    scatterer is looked for, and 4 x 4 cells -- sixteen samples -- is already
    a small one: fewer is not a search for distributed scattering at all, and
    the consensus a DS needs cannot be drawn from a handful of samples. Raised
    here rather than answered with an empty product. The counts are pixels;
    with the grid `spacing` the message speaks in the metres the caller used.
    """
    cy, cx = int(cell[0]), int(cell[1])
    if int(wy) < 4 * cy or int(wx) < 4 * cx:
        if spacing is not None:
            dy, dx = abs(float(spacing[0])), abs(float(spacing[1]))
            size = (f'DS window ({int(wy) * dy:g}, {int(wx) * dx:g}) m must span at least '
                    f'four independence cells per dimension -- cell=({cy * dy:g}, {cx * dx:g}) m '
                    f'needs a DS window of ({4 * cy * dy:g}, {4 * cx * dx:g}) m or larger')
        else:
            size = (f'DS window ({int(wy)}, {int(wx)}) px must span at least four '
                    f'independence cells per dimension -- cell={(cy, cx)} px needs a '
                    f'window of ({4 * cy}, {4 * cx}) px or larger')
        raise ValueError(
            f'{name}(): the {size}. A window of fewer than 4 x 4 cells is too '
            f'small a patch of independent samples to search for a distributed '
            f'scatterer.')


def _3d_ps_lattice(cell):
    """The PS-candidate lattice in pixels: one winner per 2 x 2 independence
    cells. One per cell is every independent sample the raster holds and the
    exhaustive solver's reference; it also quadruples the candidates the
    long-arc test scores and the nodes the network fits, and the network's
    cost is quadratic in its nodes. Two cells a side keeps the best pixel of
    a neighbourhood no larger than four samples -- a persistent scatterer is
    still the best of its own cell and of the three beside it -- for a
    quarter of the candidates. The same lattice must be used wherever the
    winner grid is cut, merged or reached across: this is the one place it
    is defined."""
    return max(2 * int(cell[0]), 1), max(2 * int(cell[1]), 1)


def _cascade_pass1(block, owned, origin, wy, wx, cell, thr, consensus,
                   threads=1, token=None):
    """One chunk of the cascade's dense scan: the DS-candidate rank raster and
    the PS-candidate winners, from a single pass over the chunk's data.

    `block` is the chunk WITH its full-window halo; `owned` = (y0, y1, x0, x1)
    names the region this task answers for, in block-local indices, and
    `origin` = (gy, gx) is the block's [0,0] in FULL-raster coordinates -- the
    winner cells live on the GLOBAL PS lattice (`_3d_ps_lattice(cell)`, two
    independence cells a side), so a chunk that does not start on a lattice
    line must still cut cells where the raster does. The halo is a full DS window per
    side because a winner cell owned by its origin reaches half a window past
    the owned edge, and gating those pixels needs THEIR windows complete.

    Returns (rank, Wser, wiy, wix):
      rank : (owned) float32 -- best arc coherence where the pixel holds at
             least `consensus` admissible arcs over `thr`, NaN otherwise.
      Wser : (dates, cells_y, cells_x) complex64 -- the winner pixel's raw
             series per PS lattice cell whose origin is owned; NaN series
             where the cell holds no gated pixel.
      wiy, wix : (cells_y, cells_x) int32 -- the winner's pixel position in
             the FULL raster, -1 where none.
    """
    S = np.asarray(block, dtype=np.complex64)
    n, ny, nx = S.shape
    y0, y1, x0, x1 = owned
    hy, hx = wy // 2, wx // 2
    # THE PS RANK COUNTS ARCS LONGER THAN TWO CELLS. Measured against the
    # exhaustive search: of the lattice cells that hold a persistent
    # scatterer, the best-coherence pixel was it in 30%, the pixel with the
    # most coherent arcs of length >= 2 x cell in 39% -- the best any
    # in-window statistic reached. The copies of a scatterer in the pixels
    # beside it share its shortest arcs; the count at more than two cells
    # away is where the cleanest copy pulls ahead.
    _floor2 = (2 * int(cell[0]), 2 * int(cell[1]))
    # ONLY WHAT IS READ IS SCANNED. The rank is read over the owned pixels and
    # the winners over the owned lattice span, which runs to the next block's
    # first cell (below); a pixel's count takes as near ends only the pixels
    # within a window above and beside it. Those rows and columns are the scan,
    # and the rest of the halo is data for their products: at a chunk a third
    # of a window wide its columns were a third of the old scan. The numbers
    # read are the same pairs, so the same bits.
    _pl = _3d_ps_lattice(cell)
    _qy1 = min((-(-(origin[0] + y1) // _pl[0]) * _pl[0]) - origin[0], ny)
    _qx1 = min((-(-(origin[1] + x1) // _pl[1]) * _pl[1]) - origin[1], nx)
    _c0, _c1 = max(0, x0 - hx), min(nx, _qx1 + hx)
    cnt = np.zeros((ny, nx), np.int32)
    best = np.zeros((ny, nx), np.float32)
    cnt2 = np.zeros((ny, nx), np.int32)
    best2 = np.zeros((ny, nx), np.float32)
    _th = max(1, int(threads))
    H = max(hy + 1, -(-(_qy1 - y0) // _th))
    bands = [(a, min(a + H, _qy1)) for a in range(y0, _qy1, H)]

    def _band(b):
        ya, yb = b
        a0 = max(0, ya - hy)
        b0 = min(ny, yb + hy)
        c_, m_, c2_, m2_ = _cascade_count_max(S[:, a0:b0], wy, wx, cell, thr,
                                              floor2=_floor2,
                                              own=(0, yb - a0, _c0, _c1))
        cnt[ya:yb] = c_[ya - a0:yb - a0]
        best[ya:yb] = m_[ya - a0:yb - a0]
        cnt2[ya:yb] = c2_[ya - a0:yb - a0]
        best2[ya:yb] = m2_[ya - a0:yb - a0]
    if len(bands) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
    else:
        for b in bands:
            _band(b)
    # DS AND PS ARE CHECKED SEPARATELY, NOT DS -> PS. A distributed scatterer
    # is a NEIGHBOURHOOD property -- coherent WITH its surroundings -- so the
    # DS raster keeps the `consensus` gate on how many coherent arcs a pixel
    # holds inside the window. A persistent scatterer only needs to hold ONE
    # coherent arc to reach the long-arc test that actually decides it, so
    # gating PS candidacy at `consensus` too discarded every PS in a sparse
    # area: it had fewer than `consensus` coherent neighbours within the DS
    # window and never reached the test that would have confirmed it.
    #
    # SELF-COHERENCE CANNOT REPLACE THE ARC. Per-pixel lag-1 coherence was
    # measured on the confirmed PS here and read 0.05-0.12, at the noise
    # floor, while their arc coherence was 0.6-0.7: a single pixel's phase
    # carries the atmosphere, the DEM-error on the drifting baseline and the
    # deformation, none of which cancel until two nearby pixels are
    # DIFFERENCED. The arc is what makes a scatterer visible; the count is the
    # only thing relaxed here.
    #   - `rank` (the DS raster) keeps the consensus gate;
    #   - the PS winner grid draws from every pixel holding at least one
    #     coherent arc longer than two cells, ranked by how many it holds.
    rank_full = np.where(cnt >= int(consensus), best, np.nan).astype(np.float32)
    rank = rank_full[y0:y1, x0:x1]
    # PS candidacy: any pixel holding a coherent arc longer than two cells,
    # ranked by HOW MANY it holds, ties by the best of them
    ps_rank_full = np.where(cnt2 >= 1, cnt2.astype(np.float32) + 1e-3 * best2,
                            np.nan).astype(np.float32)

    # winners: the argmax of the SELF-COHERENT rank in every PS lattice cell
    # -- 2 x 2 independence cells, `_3d_ps_lattice` -- on the GLOBAL lattice.
    # The independence cell is the scale below which two pixels are one
    # ground sample measured twice, so the lattice holds at most four
    # independent samples and the long-arc test (`_cascade_ps`, told the
    # lattice as `pcell`) decides which winners are persistent scatterers.
    # The half-window boxes slid by a quarter window that stood here before
    # were a hypothesis about where a PS may be -- the best of a 15 x 60 px
    # neighbourhood -- and lost the PS that was second in its box behind an
    # unrelated brighter one; the cells are disjoint, so no pixel can win
    # twice and no claim is needed.
    pcy, pcx = _3d_ps_lattice(cell)
    gy, gx = origin
    gy0, gx0 = gy + y0, gx + x0
    cy0 = (-(-gy0 // pcy) * pcy) - gy
    cx0 = (-(-gx0 // pcx) * pcx) - gx
    oy = np.arange(cy0, y1, pcy)
    ox = np.arange(cx0, x1, pcx)
    # WHERE THIS BLOCK MAY TAKE WINNERS FROM: its own lattice span, which runs
    # to the NEXT block's first origin -- not to the chunk edge. The two are
    # different when the chunk edge falls between lattice lines, and the
    # difference is exactly the band no cell of either block would search if
    # each stopped at its chunk edge. Bounded this way the blocks TILE the
    # raster: every pixel lies in the span of exactly one block, so no pixel
    # is offered twice and none is skipped, and a winner is never taken from
    # ground another block answers for.
    cy1 = min((-(-(gy + y1) // pcy) * pcy) - gy, ny)
    cx1 = min((-(-(gx + x1) // pcx) * pcx) - gx, nx)
    Wser = np.full((n, len(oy), len(ox)), np.nan, dtype=np.complex64)
    wiy = np.full((len(oy), len(ox)), -1, dtype=np.int32)
    wix = np.full((len(oy), len(ox)), -1, dtype=np.int32)
    for a, ya in enumerate(oy):
        for b, xa in enumerate(ox):
            yb_, xb_ = min(ya + pcy, cy1), min(xa + pcx, cx1)
            cell_r = ps_rank_full[ya:yb_, xa:xb_]
            if not np.isfinite(cell_r).any():
                continue
            k = np.nanargmax(cell_r)
            dy_, dx_ = divmod(int(k), cell_r.shape[1])
            Wser[:, a, b] = S[:, ya + dy_, xa + dx_]
            wiy[a, b] = gy + ya + dy_
            wix[a, b] = gx + xa + dx_
    return rank, Wser, wiy, wix


def _cascade_ps(Wser, wiy, wix, ele2phase, t, meter2rad, wy, wx, py, px,
                thr, consensus, max_dh=25.0, max_dv=25.0, step_dh=8.0,
                step_dv=2.0, iterations=8, threads=1, pcell=None,
                budget=None):
    """PS candidates from the merged winner grid: fitted long arcs only.

    The winner grid is the raster one pyramid level up: one candidate per
    `pcell` = (pcy, pcx) pixels, so reach and the short-arc exclusion are
    counted in THOSE cells -- partners closer than the DS window in both axes
    are the short-arc regime and excluded; partners beyond the PS window carry
    no common atmosphere and are not attempted. Partners are ranked raw and
    the best few FITTED -- a long arc only counts fitted.

    `pcell` is the winner lattice `_cascade_pass1` built -- `_3d_ps_lattice`,
    two independence cells a side -- so the reach stays a fixed distance in
    PIXELS whatever the cell;
    the default (wy//2, wx//2) is the lattice of older winner grids.

    Returns per-cell (gamma, dh, dv, arcs): the best fitted long-arc
    coherence, its differentials, and how many fitted long arcs cleared
    `thr`; NaN/0 where the cell has no winner or no coherent long arc.
    """
    pcy, pcx = (max(wy // 2, 1), max(wx // 2, 1)) if pcell is None \
        else (max(int(pcell[0]), 1), max(int(pcell[1]), 1))
    n, NY, NX = Wser.shape
    flat = Wser.reshape(n, -1)
    a = np.abs(flat)
    ok = np.isfinite(a).all(axis=0) & (a > 0).all(axis=0)
    idx = np.flatnonzero(ok)
    gamma = np.full((NY, NX), np.nan, np.float32)
    dh = np.full((NY, NX), np.nan, np.float32)
    dv = np.full((NY, NX), np.nan, np.float32)
    arcs = np.zeros((NY, NX), np.int32)
    if len(idx) < 2:
        return gamma, dh, dv, arcs
    with np.errstate(invalid='ignore', divide='ignore'):
        U = (flat[:, idx] / a[:, idx]).astype(np.complex64)
    cy_ = np.asarray(idx // NX, np.int64)
    cx_ = np.asarray(idx % NX, np.int64)
    # REACH IN CELLS: beyond the DS window, inside the PS window. The extent is
    # a FULL box centred on the cell, so the reach is HALF of it -- the same
    # rule the node network gets from its Chebyshev query and the attachment
    # from its tiles. Counted from the extent itself, this test reached twice
    # as far as the window it documents, and twice as far as fit1d_network for the
    # same parameter.
    ry = max(1, (py // 2) // pcy)
    rx = max(1, (px // 2) // pcx)
    # THE DS WINDOW IS A FULL BOX TOO, so the short-arc exclusion is HALF of
    # it a side, in cells -- the same boundary every other stage and the
    # brute reference draw. Excluding a full window a side refused twice the
    # ground it meant to.
    ey, ex = max(1, (wy // 2) // pcy), max(1, (wx // 2) // pcx)
    m = len(idx)
    kk = min(int(consensus), m - 1)
    # THE REACH IS APPLIED TO THE OPERAND, NOT THE PRODUCT. Scoring every
    # candidate against every other and masking afterwards computes the arcs
    # the window has already refused -- most of them, since the extent covers a
    # fraction of the lattice -- and then sorts all of it. Tiling the
    # candidates and taking the cells a tile can reach multiplies only what can
    # be kept, and turns the sort into a partition over that much smaller set.
    # The winners sit on a regular lattice in row-major order, so a tile's
    # reachable rows are a contiguous slice and only the columns need a test.
    ty, tx = max(1, ry // 2), max(1, rx // 2)
    # THE TILE IS SIZED BY MEMORY, NOT BY THE REACH. A tile scores its
    # candidates against every candidate within reach, so its working set is
    # |tile| x |reach| x (the complex product, its modulus and the two offset
    # grids). Half the reach as a tile is right when the reach holds a few
    # thousand candidates; with one candidate per independence cell and a PS
    # extent spanning the scene the reach holds every candidate there is, and
    # a half-reach tile of them against all of them is a working set of tens
    # of gigabytes PER THREAD. Bound the tile so that, against the worst-case
    # reach (all `m` candidates), one thread's tile stays inside the budget --
    # the dask chunk size when none is given: the scheduler already holds
    # many chunks at once, so one more working set of that size is nothing
    # the caller has not already allowed for.
    _bytes = _3d_budget_mb(budget) * 1024.0 * 1024.0
    _per = max(1.0, 20.0 * m)                      # bytes per tile candidate
    _dens = max(1e-6, m / float(max(NY * NX, 1)))  # candidates per cell
    _side = int(np.sqrt(max(1.0, _bytes / _per) / _dens))
    ty, tx = max(1, min(ty, _side)), max(1, min(tx, _side))
    # THE TILES ARE INDEPENDENT, so they run on the thread pool: each reads
    # shared arrays and returns its own arc list, and the results are
    # concatenated IN TILE ORDER -- the order the serial loop produced -- so
    # a score tie downstream resolves identically whatever the thread count.
    # The matmuls dominate and release the GIL, which is where the scaling
    # comes from; this stage is one task for the whole scene, and without the
    # pool it held one core of however many the caller granted.
    jobs = []
    for r0 in range(0, NY, ty):
        r1 = min(r0 + ty, NY)
        s0, s1 = np.searchsorted(cy_, r0), np.searchsorted(cy_, r1)
        if s1 <= s0:
            continue
        t0 = np.searchsorted(cy_, max(0, r0 - ry))
        t1 = np.searchsorted(cy_, r1 - 1 + ry, side='right')
        for c0 in range(0, NX, tx):
            jobs.append((s0, s1, t0, t1, c0, min(c0 + tx, NX)))

    def _tile(job):
        s0, s1, t0, t1, c0, c1 = job
        si = s0 + np.flatnonzero((cx_[s0:s1] >= c0) & (cx_[s0:s1] < c1))
        if not len(si):
            return None
        ti = t0 + np.flatnonzero((cx_[t0:t1] >= c0 - rx)
                                 & (cx_[t0:t1] <= c1 - 1 + rx))
        if not len(ti):
            return None
        G = np.abs(U[:, si].conj().T @ U[:, ti]) / n
        ddy = np.abs(cy_[si][:, None].astype(np.int32) - cy_[ti][None, :].astype(np.int32))
        ddx = np.abs(cx_[si][:, None].astype(np.int32) - cx_[ti][None, :].astype(np.int32))
        G[(ddy < ey) & (ddx < ex)] = -1.0      # short-arc regime
        G[(ddy > ry) | (ddx > rx)] = -1.0      # the tile over-reaches a
        #                                        little; the pair test does
        #                                        not
        k_ = min(kk, G.shape[1])
        if k_ < 1:
            return None
        top = (np.argpartition(-G, k_ - 1, axis=1)[:, :k_]
               if k_ < G.shape[1] else np.argsort(-G, axis=1)[:, :k_])
        keep = np.take_along_axis(G, top, 1) > 0
        return (np.repeat(si, k_)[keep.ravel()],
                ti[top.ravel()][keep.ravel()])

    _th = max(1, int(threads))
    if _th > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(_th) as _ex:
            outs = list(_ex.map(_tile, jobs))
    else:
        outs = [_tile(j) for j in jobs]
    src_l = [o[0] for o in outs if o is not None]
    tgt_l = [o[1] for o in outs if o is not None]
    src = np.concatenate(src_l) if src_l else np.empty(0, np.int64)
    tgt = np.concatenate(tgt_l) if tgt_l else np.empty(0, np.int64)
    if len(src):
        ga, dha, dva, _ = _3d_arc_batch(
            U, U, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
            step_dh, step_dv, _3d_budget_mb(budget), iterations,
            threads=threads, threshold=float(thr))
        good = np.isfinite(ga) & (ga >= float(thr))
        np.add.at(arcs.ravel(), idx[src[good]], 1)
        order = np.lexsort((-np.where(good, ga, -np.inf), src))
        first = order[np.r_[True, np.diff(src[order]) > 0]]
        first = first[good[first]]
        gamma.ravel()[idx[src[first]]] = ga[first]
        dh.ravel()[idx[src[first]]] = dha[first]
        dv.ravel()[idx[src[first]]] = dva[first]
    return gamma, dh, dv, arcs


def _fit3d_scan_for_dask(block, owned, origin, cell_origin, kw, threads,
                         token=None):
    """PASS 1: the cascade's dense scan, one chunk, read inside the task.

    What leaves is what the plan always specified: the DS-candidate RANK
    RASTER for the owned pixels, and the PS-candidate WINNER GRID -- one
    candidate per independence cell, its series and the pixel it came from.
    The winner grid is the raster one pyramid level up, so the whole scene's
    candidates weigh megabytes and the PS test can be asked ONCE, over all of
    them, instead of once per chunk against whatever that chunk happened to
    contain.

    `cell_origin` names where this chunk's winner grid sits on the scene's own
    cell lattice, so the level-1 stage can lay the parts side by side.
    """
    import numpy as np
    if isinstance(block, _Fit3dSlice):
        block = block.read(threads)
    # PARTIAL PER-DATE COVERAGE IS NORMAL AND TOLERATED. A pixel valid on some
    # dates and NaN on others is not a broken stack: the burst footprint drifts
    # sub-pixel between acquisitions, so a thin strip along the slanted edge is
    # imaged by only some dates. The pipeline handles it -- `_cascade_ps` only
    # ever uses pixels valid on EVERY date as candidates, so an edge pixel with
    # gaps simply is not a candidate, while the fully covered interior forms
    # the network. An earlier hard check here raised on that edge strip and
    # failed the whole fit; it was wrong. The one fatal case -- a date empty
    # over the WHOLE scene, so no all-dates pixel exists anywhere -- surfaces
    # as an empty network in `_fit3d_select_ps`, which raises there and names
    # the absent dates from their real coverage.
    wy, wx, _pey, _pex = kw['window']
    rank, W, wiy, wix = _cascade_pass1(
        block, owned, origin, wy, wx, tuple(kw['cell']),
        float(kw['threshold']), int(kw['min_agreeing']), threads=threads)
    # SCENE PIXELS ALREADY: the origin is handed to the scan in scene
    # coordinates, so the cells are cut on the SCENE lattice and the winner
    # positions come back on it -- one lattice for every burst, which is what
    # lets the level-1 stage lay the parts side by side exactly.
    return rank, W, wiy, wix


def _fit3d_select_ps(parts, kw, threads):
    """The PS test over the candidates in `parts`, on ONE scene lattice.

    Each part is `((W, iy, ix), (cell_y0, cell_x0))`; the grids are laid
    side by side by cell index and tested TOGETHER, a later part painting
    over an earlier one where both hold a winner. Returns None when no part
    holds a candidate, else a dict:
      g, wiy, wix : the test's coherence per lattice cell and the winner's
          scene pixel (-1 where none);
      U, iy, ix : the CERTIFIED winners' unit phasors and scene pixels, the
          network's input;
      cands, lat : how many candidates were tested and the lattice shape.
    Split out so the SELECTION can be varied -- all candidates together, or
    each chunk's alone -- while the network solve that follows stays the same.
    """
    import numpy as np
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    n = parts[0][0][0].shape[0]
    NY = max(int(p[1][0]) + p[0][0].shape[1] for p in parts)
    NX = max(int(p[1][1]) + p[0][0].shape[2] for p in parts)
    Wser = np.full((n, NY, NX), np.nan, np.complex64)
    wiy = np.full((NY, NX), -1, np.int32)
    wix = np.full((NY, NX), -1, np.int32)
    for (W, iy_, ix_), (cy0, cx0) in parts:
        sy = slice(int(cy0), int(cy0) + W.shape[1])
        sx = slice(int(cx0), int(cx0) + W.shape[2])
        take = np.asarray(iy_) >= 0
        Wser[:, sy, sx] = np.where(take[None], W, Wser[:, sy, sx])
        wiy[sy, sx] = np.where(take, iy_, wiy[sy, sx])
        wix[sy, sx] = np.where(take, ix_, wix[sy, sx])

    # THE ONE FATAL DISAGREEMENT: no candidate is valid on EVERY date, so the
    # all-dates test that PS and the network both need leaves fewer than two.
    # Partial edge coverage does not cause this -- the interior is imaged by
    # every date and its candidates are all-dates valid; only a date that is
    # empty across essentially the WHOLE candidate footprint empties the
    # intersection. When that happens the fit would return a model of NaN with
    # no error, so it is caught here and the ABSENT dates are named -- those
    # far below the coverage the rest reach, not merely short of a stray edge
    # pixel.
    _amp = np.abs(Wser)
    _good = np.isfinite(_amp)
    _have = wiy >= 0
    if _have.any() and int((_good.all(axis=0) & _have).sum()) < 2:
        _cov = _good[:, _have].mean(axis=1)
        _cmax = float(_cov.max())
        _bad = np.flatnonzero(_cov < 0.5 * _cmax)
        if len(_bad) and _cmax > 0.0:
            _dvs = np.asarray(kw['date_values'])
            _nm = ', '.join(f'{str(_dvs[i])[:10]} ({100*_cov[i]:.0f}%)'
                            for i in _bad[:8])
            raise ValueError(
                f'fit1d_network(): no scatterer is coherent on every date -- '
                f'{len(_bad)} of {len(_cov)} dates cover under half of what '
                f'the rest do: {_nm}'
                + (' ...' if len(_bad) > 8 else ''))
    del _amp, _good
    wy, wx, pey, pex = kw['window']
    t, ele2phase, meter2rad, _car = _3d_fit_frame(
        kw['date_values'], kw['bperp'], kw['geometry'], n)
    g, _dh, _dv, _arcs = _cascade_ps(
        Wser, wiy, wix, ele2phase, t, meter2rad, wy, wx, pey, pex,
        float(kw['threshold']), int(kw['min_agreeing']),
        max_dh=kw['max_dh'], max_dv=kw['max_dv'], step_dh=kw['step_dh'],
        step_dv=kw['step_dv'], iterations=int(kw['iterations']),
        threads=threads, budget=kw['budget'],
        # the winner grid is built on the PS lattice in _cascade_pass1;
        # reach and the short-arc exclusion count in it
        pcell=_3d_ps_lattice(kw['cell']))
    out = dict(g=np.asarray(g, np.float32), wiy=wiy, wix=wix,
               cands=int(np.count_nonzero(wiy >= 0)), lat=(NY, NX))
    sel = np.isfinite(g) & (g >= float(kw['threshold'])) & (wiy >= 0)
    if not int(sel.sum()):
        out.update(U=np.zeros((n, 0), np.complex64),
                   iy=np.zeros(0, np.int64), ix=np.zeros(0, np.int64))
        return out
    W = Wser[:, sel]
    a = np.abs(W)
    with np.errstate(invalid='ignore', divide='ignore'):
        U = np.ascontiguousarray(
            np.where(a > 0, W / np.where(a > 0, a, 1), 0).astype(np.complex64))
    out.update(U=U, iy=wiy[sel].astype(np.int64), ix=wix[sel].astype(np.int64))
    return out


def _fit3d_ps_raster_for_dask(test, oy, ox, ny, nx):
    """The PS test written back onto ONE chunk of a burst's pixel grid.

    The test answers per lattice cell; the winner knows which pixel it is,
    so its coherence goes back to that pixel, NaN elsewhere. EVERY winner
    inside the chunk is written, whichever burst it was scanned from --
    the rule fit1d_network() writes its nodes by, so a burst whose grid overlaps
    another's shows the same points the model will carry there.
    """
    import numpy as np
    out = np.full((int(ny), int(nx)), np.nan, np.float32)
    if test is None:
        return out
    g = np.asarray(test['g']).ravel()
    yy = np.asarray(test['wiy']).ravel()
    xx = np.asarray(test['wix']).ravel()
    m = (yy >= 0) & (xx >= 0) & np.isfinite(g)
    yy = yy[m] - int(oy)
    xx = xx[m] - int(ox)
    ok = (yy >= 0) & (yy < int(ny)) & (xx >= 0) & (xx < int(nx))
    out[yy[ok], xx[ok]] = g[m][ok]
    return out


def _fit3d_level1_for_dask(parts, kw, threads):
    """THE PS TEST, then the one network over whatever it certified.

    The winner grids are laid on one cell lattice -- the bursts share a ground
    grid, so a chunk's grid sits a whole number of cells from the scene's
    origin -- and the long-arc test runs on the result. A candidate is
    therefore judged against every candidate within the PS extent, not against
    the ones that happen to share its chunk: at this chunking a chunk is a
    fraction of that extent wide, so the chunk-local test was asking a question
    the data in that chunk could not answer.

    THE LATER BURST WINS THE OVERLAP, as everywhere else here.
    """
    import numpy as np
    import time
    _mark = time.monotonic()
    got = _fit3d_select_ps(parts, kw, threads)
    if got is not None and got['U'].shape[1] == 0:
        got = None
    cands = None if got is None else got['cands']
    lat = None if got is None else got['lat']
    # NO NETWORK IS FATAL, AND SAID SO HERE. This stage answers for the WHOLE
    # scene: with no certified PS there is no datum, so every later level has
    # nothing to attach to and every pixel comes back NaN. Returning None let
    # the run continue -- an hour of DS work writing NaN into planes nothing
    # anchors, ending in a model that is empty with no error anywhere. A block
    # holding no nodes is ordinary and still handled below; a SCENE holding
    # none is a stop.
    if got is None:
        raise ValueError(
            'fit1d_network(): the PS test certified no scatterer anywhere in the '
            f'scene{kw.get("tag", "")} at threshold={float(kw["threshold"]):g}. '
            'Nothing can be '
            'attached without a network, so the fit would return an empty '
            'model. Lower `threshold`, widen the PS extent (the second window '
            'value, or window[2:] in the four-value form), or check that the '
            'stack carries coherent scatterers.')
    U, iy, ix = got['U'], got['iy'], got['ix']
    if kw['debug']:
        # LEVEL 0, which is what this stage produces: the PS test and the
        # network solved over what it certified. It is the union driver's
        # FIRST pass, which is all the function's name says -- printing that
        # ordinal as the level labelled the PS network as level 1 and left
        # `level=0` reporting a level it had been asked not to run.
        print(f'DEBUG: level 0{kw.get("tag", "")}  {cands:,} candidates'
              f' on a {lat[0]} x {lat[1]} cell lattice'
              + f' -> {U.shape[1]:,} PS'
              + f'   {time.monotonic() - _mark:.1f}s', flush=True)
    if U.shape[1] < 2:
        raise ValueError(
            f'fit1d_network(): the PS test certified {U.shape[1]} scatterer(s) in the '
            f'whole scene{kw.get("tag", "")} at '
            f'threshold={float(kw["threshold"]):g}; a network '
            'needs at least two. The fit would return an empty model. Lower '
            '`threshold`, widen the PS extent (the second window value, or '
            'window[2:] in the four-value form), or check the stack.')
    return _3d_ps_network(
        U, iy, ix, kw['date_values'], bperp=kw['bperp'], window=kw['window'],
        threshold=float(kw['threshold']), geometry=kw['geometry'],
        budget=kw['budget'], consensus=kw['consensus'],
        iterations=int(kw['iterations']), max_dh=kw['max_dh'],
        max_dv=kw['max_dv'], step_dh=kw['step_dh'], step_dv=kw['step_dv'],
        max_seasonal=kw['max_seasonal'],
        err_dh=kw.get('err_dh', 5.0), err_dv=kw.get('err_dv', 1.0),
        threads=threads, debug=kw['debug'])


def _fit3d_attach_for_dask(block, part, net, kw, origin, threads, token=None,
                           emit_nodes=False):
    """PASS 2 of the union: the shared network written onto this block's grid.

    The block is read a SECOND time. That is the price of solving one network:
    pass 1 could not keep the pixels -- keeping them is what overloaded the
    host -- so the series come back from the store while the network does not.
    A node's model reaches this block only if the node stands in it; the datum
    it carries is the one the whole scene agreed on.
    """
    import numpy as np
    if isinstance(block, _Fit3dSlice):
        block = block.read(threads)
    n, ny, nx = block.shape
    q = part
    # THE NODES WITHIN THIS BLOCK'S PS EXTENT, in the block's own index space.
    # Nodes outside the block are partners and nothing else -- the extent is a
    # full box centred on a pixel, so a candidate here reaches half of it
    # either side and no further. That bound is what lets the attachment go on
    # ranking against every node it is handed: a block is narrower than the
    # extent, so inside one burst the two rules already agreed, and across the
    # seam this is the reach the window always claimed. Handing it the whole
    # scene instead would rank a candidate against nodes its own window
    # refuses, which is neither cheaper nor right.
    nodes = None
    if net is not None and len(np.asarray(net['iy'])):
        _iy = np.asarray(net['iy']) - int(origin[0])
        _ix = np.asarray(net['ix']) - int(origin[1])
        _ry = max(int(kw['window'][2]) // 2, 1)
        _rx = max(int(kw['window'][3]) // 2, 1)
        m = ((_iy >= -_ry) & (_iy < ny + _ry)
             & (_ix >= -_rx) & (_ix < nx + _rx))
        if m.any():
            nodes = dict(iy=_iy[m], ix=_ix[m],
                         U=np.ascontiguousarray(net['U'][:, m]),
                         label=np.asarray(net['label'])[m],
                         vel=np.asarray(net['vel'])[m],
                         hgt=np.asarray(net['hgt'])[m],
                         coh=np.asarray(net['coh'])[m],
                         sea=np.asarray(net['sea'])[m])
            if kw['debug']:
                # COUNTED HERE, REPORTED ONCE. One line per block said the
                # same thing every time: the PS extent is wider than a block
                # by design, so nearly the whole network clears the filter and
                # only the owned count varies. The level's reducer prints the
                # spread over the blocks instead.
                _inb = ((nodes['iy'] >= 0) & (nodes['iy'] < ny)
                        & (nodes['ix'] >= 0) & (nodes['ix'] < nx))
                _offered, _ownedn = int(m.sum()), int(_inb.sum())
    _own = {}
    l, v, h, sa, cg, lv = _3d_ps_attach(
        block, q, nodes, kw['date_values'], out_stats=_own,
        spacing=kw['spacing'],
        bperp=kw['bperp'], window=kw['window'], threshold=kw['threshold'],
        cell=kw['cell'], geometry=kw['geometry'], budget=kw['budget'],
        level=kw['level'], max_dh=kw['max_dh'], max_dv=kw['max_dv'],
        step_dh=kw['step_dh'], step_dv=kw['step_dv'],
        consensus=kw['consensus'], iterations=kw['iterations'],
        err_dh=kw.get('err_dh', 5.0), err_dv=kw.get('err_dv', 1.0),
        threads=threads, debug=kw['debug'])
    # ONE CONVENTION ACROSS EVERY FIT, whichever driver called this
    v = -v
    sa = -sa
    out = np.concatenate(
        [l[None].astype(np.complex64), v[None].astype(np.complex64),
         h[None].astype(np.complex64), sa[None].astype(np.complex64),
         cg[None].astype(np.complex64), lv[None].astype(np.complex64)],
        axis=0)
    if not emit_nodes:
        return out
    # THE LEVEL-1 NODES, IN SCENE COORDINATES. Level 2 is a separate stage over
    # these; a chunk needs the ones its neighbours own as well as its own, so
    # they leave here in a frame every chunk shares.
    # THIS CALL'S OWN NODES, WITH NO FALLBACK. `stats` outlives the task --
    # dask reuses its worker threads, so the dict still holds whatever the
    # previous block on this thread wrote. Falling back to it when this call
    # produced nothing is not a safe default: the stale coordinates get
    # re-based to THIS block's origin and shipped as this block's level-1
    # nodes, so level 2 attaches to pixels that were never solved, carrying
    # another block's velocities and another datum. An empty table is the
    # correct answer for a block that attached nothing.
    _st = _own
    import numpy as _np
    if int(_st.get('ds_attached', 0)) and 'ds_iy' in _st:
        _nodes = dict(
            iy=_np.asarray(_st['ds_iy'], _np.int64) + int(origin[0]),
            ix=_np.asarray(_st['ds_ix'], _np.int64) + int(origin[1]),
            vel=_np.asarray(_st['ds_velocity_rad_yr'], float),
            hgt=_np.asarray(_st['ds_height_rad'], float),
            sea=_np.asarray(_st['ds_seasonal_rad']),
            label=_np.asarray(_st['ds_label']),
            gamma=(_np.asarray(_st['ds_gamma'], float)
                   if _st.get('ds_gamma') is not None else None),
            level=_np.ones(len(_st['ds_iy']), _np.int16))
    else:
        _nodes = dict(iy=_np.zeros(0, _np.int64), ix=_np.zeros(0, _np.int64),
                      vel=_np.zeros(0), hgt=_np.zeros(0),
                      sea=_np.zeros(0, _np.complex64),
                      label=_np.zeros(0, _np.int8), gamma=None)
    _nodes['_stats'] = {k: v for k, v in _st.items() if k.startswith('lvl_')}
    if kw['debug'] and nodes is not None:
        _nodes['_stats']['lvl_net_total'] = int(len(_np.asarray(net['iy'])))
        _nodes['_stats']['lvl_net_offered'] = _offered
        _nodes['_stats']['lvl_net_owned'] = _ownedn
    return out, _nodes


def _fit3d_level_report(level_id, infos, debug=False, tag=''):
    """ONE REPORT PER LEVEL, not one per chunk.

    A chunk's numbers describe a chunk. With dozens of blocks the per-chunk
    lines are unreadable and, worse, unusable: a level's yield is the sum over
    its blocks and its error distribution is the pooled samples, neither of
    which a reader can recover from a scrolling list. The blocks of a level are
    all finished by the time this runs -- that is what the merge barrier is for
    -- so this is the first point where the level can be described at all.

    Returns the level's node tables, so it sits on the path the next level
    already depends on rather than being a side branch that has to be kept
    alive artificially.

    Under union=False every chunk is its own level and calls this with its
    one table and a `tag` naming the chunk, so the lines say what they
    describe.
    """
    import numpy as np
    tabs = [i for i in infos if i is not None]
    if not debug:
        return tabs
    st = [t.get('_stats') or {} for t in tabs]
    st = [x for x in st if x]
    if not st:
        return tabs
    tot = lambda k: sum(int(x.get(k, 0)) for x in st)

    # THE BLOCKS ALREADY REDUCED THESE. What arrives per key is a handful of
    # scalars per block (see `_lvl_stat`): counts and threshold tallies add up
    # exactly, extremes are the extreme of the blocks' own, and a percentile
    # is reported as the RANGE across blocks -- a median of medians is not the
    # median, so it is not printed as one.
    def agg(k):
        xs = [x[k] for x in st
              if isinstance(x.get(k), dict) and int(x[k].get('n', 0))]
        if not xs:
            return None
        a = dict(n=sum(int(x['n']) for x in xs),
                 blocks=len(xs),
                 min=min(x['min'] for x in xs),
                 max=max(x['max'] for x in xs),
                 p50lo=min(x['p50'] for x in xs),
                 p50hi=max(x['p50'] for x in xs),
                 p90hi=max(x['p90'] for x in xs),
                 p99hi=max(x['p99'] for x in xs))
        for key in set().union(*(set(x) for x in xs)):
            if key[:2] in ('le', 'ge', 'gt'):
                a[key] = sum(int(x.get(key, 0)) for x in xs)
        a['_thr'] = xs[0].get('_thr') or []
        return a
    def over(a):
        """(count, bound) for the single `gt` tally a key carries."""
        for op, x in a.get('_thr', []):
            if op == 'gt':
                return a.get(f'gt{x:g}', 0), x
        return 0, float('nan')
    pct = lambda a, key: 100.0 * a.get(key, 0) / max(a['n'], 1)
    rng = lambda a: (f"{a['p50lo']:.3g}" if a['p50lo'] == a['p50hi']
                     else f"{a['p50lo']:.3g}..{a['p50hi']:.3g}")

    left, att = tot('lvl_left'), tot('lvl_attached')
    print(f'DEBUG: LEVEL {level_id} over {len(st)} block(s){tag}: '
          f'{tot("lvl_cands"):,} candidates x {tot("lvl_fixed"):,} fixed nodes '
          f'-> {tot("lvl_arcs"):,} arcs; {left:,} left, {att:,} attached '
          f'({100.0 * att / max(left, 1):.1f}%)', flush=True)
    _tn = [int(x['lvl_net_total']) for x in st if 'lvl_net_total' in x]
    if _tn:
        # THE ONE FACT WORTH A LINE. The PS extent is wider than a block by
        # design, so every block is normally offered the whole network and
        # saying so 48 times says nothing. It matters only when the extent
        # STOPS reaching -- then some block was offered fewer, and that is
        # what gets printed.
        _net_n = max(_tn)
        _short = min((int(x.get('lvl_net_offered', 0)) for x in st
                      if 'lvl_net_total' in x), default=_net_n)
        print(f'DEBUG:   network {_net_n:,} nodes, '
              + ('all of them within the PS extent of every block'
                 if _short >= _net_n
                 else f'as few as {_short:,} within the PS extent of some '
                      f'block -- the extent no longer spans the scene'),
              flush=True)
    _no, _few = tot('lvl_no_consensus'), tot('lvl_too_few')
    if _no or _few:
        print(f'DEBUG:   of the {_no:,} not attached: {_few:,} had too few '
              f'partners, {_no - _few:,} had enough and disagreed', flush=True)
        _xc, _sd = tot('lvl_multi_comp'), tot('lvl_straddled')
        _xv = tot('lvl_cross_votes')
        print(f'DEBUG:   {_xc:,} candidates saw more than one component; '
              f'{_sd:,} straddled one, {_xv:,} votes crossed', flush=True)
    _ps = tot('lvl_ps')
    if _ps:
        print(f'DEBUG:   {_ps:,} PS + {att:,} DS = {_ps + att:,} measured '
              f'pixels after this level', flush=True)
    _a = agg('lvl_gamma')
    if _a:
        print(f'DEBUG:   attaching arc gamma p50 per block {rng(_a)}',
              flush=True)
    _vet, _sin = tot('lvl_vetted'), tot('lvl_solve_in')
    if _sin:
        print(f'DEBUG:   funnel: consensus vetted {_vet:,} arcs; solve was '
              f'handed {_sin:,} equations', flush=True)
    _pc, _pv = agg('lvl_pcells'), agg('lvl_pvotes')
    if _pc:
        print(f'DEBUG:   partner INDEPENDENCE: votes p50 per block '
              f'{rng(_pv) if _pv else "-"}, but distinct independence cells '
              f'p50 {rng(_pc)}; all partners in ONE cell: '
              f'{pct(_pc, "le1"):.1f}%, in <=2 cells: {pct(_pc, "le2"):.1f}%',
              flush=True)
    _pa2 = agg('lvl_parcm')
    if _pa2:
        print(f'DEBUG:   partner DISTANCE (mean over the voting arcs): p50 '
              f'per block {rng(_pa2)} worst-block p90 {_pa2["p90hi"]:.0f} p99 '
              f'{_pa2["p99hi"]:.0f} min {_pa2["min"]:.0f} max '
              f'{_pa2["max"]:.0f} m', flush=True)
    _oc = agg('lvl_offcentre')
    if _oc:
        _n_oc, _at_oc = over(_oc)
        print(f'DEBUG:   arcs entering the solve, vs the pixel\'s consensus '
              f'centre: p50 per block {rng(_oc)} worst-block p90 '
              f'{_oc["p90hi"]:.3f} p99 {_oc["p99hi"]:.3f} mm/yr; over '
              f'{_at_oc:g} mm/yr (err_dv) {_n_oc:,} of {_oc["n"]:,} '
              f'({100.0 * _n_oc / max(_oc["n"], 1):.1f}%)', flush=True)
    par = agg('lvl_partners')
    if par:
        kk = max((int(x.get('lvl_kk', 0)) for x in st), default=0)
        print(f'DEBUG:   shortlist k={kk}: partners found p50 per block '
              f'{rng(par)} min {int(par["min"])} max {int(par["max"])}',
              flush=True)
    av, ah = agg('lvl_clo_arc_v'), agg('lvl_clo_arc_h')
    dv, dh = agg('lvl_clo_ds_v'), agg('lvl_clo_ds_h')
    if av:
        print(f'DEBUG:   closure over {av["n"]:,} VOTING partners:', flush=True)
        _h = (f'   height p50 {rng(ah)} worst-block p90 {ah["p90hi"]:.2f} m'
              if ah else '')
        print(f'DEBUG:     per arc   rate p50 {rng(av)} worst-block p90 '
              f'{av["p90hi"]:.3f} mm/yr{_h}', flush=True)
    if dv:
        _n_dv, _at_dv = over(dv)
        _h = (f'   height p50 {rng(dh)} max {dh["max"]:.2f} m' if dh else '')
        print(f'DEBUG:     per DS    rate p50 {rng(dv)} max {dv["max"]:.3f} '
              f'mm/yr{_h}   over {_at_dv:g} mm/yr (err_dv): '
              f'{_n_dv:,} of {dv["n"]:,}', flush=True)
    return tabs


def _fit3d_model(ds, da_xr, both, date_values, bp):
    """The six planes of one burst as the model dataset, variables named by
    QUANTITY alone. `both` is (6, y, x) complex64: label, velocity, height,
    seasonal, coherence, level -- the planes every stage writes.

    rmse is sqrt(-2 ln gamma), derived here rather than carried as a seventh
    plane: the two are exact inverses, so shipping both through the graph
    would move the same information twice. The `date` coordinate is the
    MASTER, where B_perp is smallest -- where the fit zeroes its t -- so
    predict() reads the origin instead of reconstructing it.
    """
    import numpy as np
    import xarray as xr
    import dask.array as da
    lb = both[0].real.astype(np.int8)
    vv = both[1].real.astype(np.float32)
    hh_ = both[2].real.astype(np.float32)
    sa_ = both[3].astype(np.complex64)
    cg_ = both[4].real.astype(np.float32)
    lv_ = both[5].real.astype(np.int8)
    rr = da.sqrt(da.maximum(
        -2.0 * da.log(da.clip(cg_, 1e-9, 1.0)), 0.0)).astype(np.float32)
    coords = {k_: v for k_, v in da_xr.coords.items()
              if k_ in ('y', 'x', 'spatial_ref')}
    mvars = {}
    for nm_, arr_ in (('velocity', vv), ('height', hh_),
                      ('seasonal', sa_), ('coherence', cg_),
                      ('rmse', rr), ('conncomp', lb),
                      ('level', lv_)):
        mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
    mds = xr.Dataset(mvars, attrs=ds.attrs)
    _dd = (np.asarray(date_values).astype('datetime64[D]')
           .astype(np.float64))
    _b3 = (np.zeros_like(_dd) if bp is None
           else np.asarray(bp, float).ravel())
    if _b3.shape != _dd.shape:
        _b3 = np.zeros_like(_dd)
    mds = mds.assign_coords(
        date=np.datetime64(int(_dd[int(np.argmin(np.abs(_b3)))]), 'D'))
    if 'spatial_ref' in ds.coords:
        mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
    return mds


def _fit3d_setup(self, threshold, window, cell, baseline, level,
                 max_dh, max_dv, step_dh, step_dv, max_seasonal,
                 consensus, err_dh, err_dv, iterations, debug,
                 chain=None, tag='', ep=None):
    """WHAT EVERY fit1d_network() DRIVER SHARES: the polarisation, the windows,
    the cluster's shape, each burst's frame and the kwargs the stage
    functions take, and the scene lattice the bursts sit on.

    One place, so the union driver and the per-chunk driver cannot read
    the same stack two ways. Returns a dict; `_fit3d_scan` adds the
    pass-1 graph to it. `ep` is `_elevation_phase_approximate()` of a
    stack holding these bursts, computed once by a caller that loops over
    bursts; None computes it here.
    """
    import numpy as np
    import xarray as xr
    import dask as _dask
    import dask.array as da
    from .Batch import Batch
    from .utils_dask import get_dask_chunk_size_mb

    _pols = [v for ds in self.values() for v in ds.data_vars
             if ds[v].dtype.kind == 'c' and 'y' in ds[v].dims]
    if len(set(_pols)) != 1:
        raise ValueError(f"fit1d_network() takes ONE polarisation, this stack "
                         f"carries {sorted(set(_pols))}.")
    pol = sorted(set(_pols))[0]
    # THE INPUT CHUNKS STATE THE SIZE. arcs() takes the same number the
    # same way: a caller who wants the work blocked differently rechunks
    # the stack, and a second knob saying the same thing could only
    # contradict it.
    budget_mb = get_dask_chunk_size_mb()

    # THE CLUSTER STATES THE SHAPE, as the per-burst path reads it: the
    # declared cores first, the process's own count only without them
    from .utils_dask import cluster_cores
    _slots, _cores = cluster_cores()
    _threads = max(1, _cores // max(1, _slots))
    _width = _slots
    if chain is None:
        chain = _Fit3dChain(_width)
    elif chain.width is None:
        chain.width = int(_width)

    # EARLIEST BURST FIRST, so the later one wins the seam it shares: the one
    # acquisition order of every burst merge, as to_dataset() lays the bursts
    _keys = BatchCore._acquisition_order(self, 'fit1d_network')
    _dss = [self[k] for k in _keys]

    # EVERY CHUNK ANSWERS TO ITS OWN BURST'S GEOMETRY. Averaging the
    # wavelength, the elevation phase and the baseline across bursts
    # describes no acquisition that ever happened: each is recorded per
    # burst and is exactly known, so a chunk is fitted with the numbers its
    # own burst carries. The network is the one stage that spans bursts,
    # and it takes the EARLIEST burst's frame -- a real geometry rather
    # than a mean of several -- which is also the burst the datum rests on.
    def _scalar(v):
        a = np.asarray(v, dtype=float).ravel()
        return float(a[0]) if a.size == 1 else float(np.mean(a))
    _ep = Batch._elevation_phase_approximate(self) if ep is None else ep

    def _frame(key, ds):
        lam_ = _scalar(ds['radar_wavelength'].values)
        bp_ = (np.asarray(ds[baseline].values, float).ravel()
               if baseline and baseline in ds.data_vars else None)
        yv_ = np.asarray(ds['y'].values, dtype=float)
        xv_ = np.asarray(ds['x'].values, dtype=float)
        return dict(
            date_values=np.asarray(ds.coords['date'].values),
            spacing=(abs(float(yv_[1] - yv_[0])) if yv_.size > 1 else 1.0,
                     abs(float(xv_[1] - xv_[0])) if xv_.size > 1 else 1.0),
            bperp=bp_,
            geometry=(lam_, (4.0 * np.pi / lam_) / _ep[key]))

    # THE SCENE LATTICE. The bursts are geocoded on one ground grid, so a
    # node's place in the scene is its place in its own burst plus a whole
    # number of pixels -- no resampling, and nothing to interpolate.
    yv0 = np.asarray(_dss[0]['y'].values, dtype=float)
    xv0 = np.asarray(_dss[0]['x'].values, dtype=float)
    dy = float(yv0[1] - yv0[0]) if yv0.size > 1 else 1.0
    dx = float(xv0[1] - xv0[0]) if xv0.size > 1 else 1.0
    spacing = (abs(dy), abs(dx))
    # METRES IN, PIXELS INSIDE. `window` and `cell` are ground distances,
    # so the same call means the same neighbourhood on any posting. They
    # are divided by the scene's spacing ONCE, here, and everything below
    # -- halos, lattices, the dask tasks -- sees pixels only. The pixel
    # validators run after the division, on the counts they were written
    # for.
    wy, wx, pey, pex = utils_xarray.window_meters_to_pixels(window, spacing)
    cell = utils_xarray.meters_to_pixels(cell, spacing, minimum=1,
                                         name='cell')
    wy, wx, pey, pex = _3d_windows((wy, wx, pey, pex))
    _3d_check_window_cell(wy, wx, cell, 'fit1d_network', spacing=spacing)
    _y0s = [float(np.asarray(d['y'].values, dtype=float)[0]) for d in _dss]
    _x0s = [float(np.asarray(d['x'].values, dtype=float)[0]) for d in _dss]
    y_org = max(_y0s) if dy < 0 else min(_y0s)
    x_org = max(_x0s) if dx < 0 else min(_x0s)

    # the cascade gates a DS candidate on how many in-window arcs it
    # holds, which is the same count the rest of the solve answers to
    _ma_ds = _3d_consensus(consensus)
    _common = dict(
        window=(wy, wx, pey, pex), threshold=float(threshold),
        min_agreeing=int(_ma_ds),
        cell=tuple(cell), budget=budget_mb,
        level=int(level), max_dh=float(max_dh), max_dv=float(max_dv),
        step_dh=float(step_dh), step_dv=float(step_dv),
        max_seasonal=float(max_seasonal),
        consensus=int(_ma_ds),
        err_dh=float(err_dh), err_dv=float(err_dv),
        iterations=int(iterations), debug=bool(debug), tag=str(tag))
    _kw_of = {k: dict(_common, **_frame(k, ds))
              for k, ds in zip(_keys, _dss)}
    # the network spans the bursts; it answers to the earliest one's
    _kw_net = _kw_of[_keys[0]]
    date_values = _kw_net['date_values']
    bp = _kw_net['bperp']
    return dict(pol=pol, keys=_keys, dss=_dss, kw_of=_kw_of, kw_net=_kw_net,
                cores=_cores, threads=_threads, width=_width,
                chain=chain, tag=str(tag), cell=tuple(cell),
                window=(wy, wx, pey, pex), origin=(y_org, x_org),
                step=(dy, dx), date_values=date_values, bperp=bp,
                budget=budget_mb)


def _fit3d_scan(self, threshold, window, cell, baseline, level,
                max_dh, max_dv, step_dh, step_dv, max_seasonal,
                consensus, err_dh, err_dv, iterations, debug,
                chain=None, tag='', ep=None):
    """PASS 1, SHARED: every chunk scanned on the scene lattice.

    This is the first half of fit1d_network()'s union driver, and the whole of
    what arcs() needs: the graph that reads each chunk with its halo,
    scans it for the DS rank raster and the PS-candidate winner grid, and
    names where each winner grid sits on ONE scene lattice. fit1d_network() goes
    on to the PS test, the network and the levels; arcs() stops at the
    PS test and returns the two rasters. ONE driver, so the debug tool
    sees exactly the candidates the delivery sees -- a second copy of
    this loop drifted in every detail it was not kept up with.

    Returns a dict: `keys` (earliest burst first), `dss`, `pol`, `grid`
    (per key: rows, cols, the array, the dataset), `blocks` (one entry
    per chunk, see the loop), `kw_of` (per key), `kw_net`, `cores`,
    `threads`, `width`, `date_values`, `bperp`.
    """
    import numpy as np
    import dask as _dask
    import dask.array as da
    _su = _fit3d_setup(self, threshold, window, cell, baseline, level,
                            max_dh, max_dv, step_dh, step_dv, max_seasonal,
                            consensus, err_dh, err_dv, iterations, debug,
                            chain=chain, tag=tag, ep=ep)
    pol, _keys, _dss = _su['pol'], _su['keys'], _su['dss']
    _kw_of, _kw_net = _su['kw_of'], _su['kw_net']
    _cores, _threads, _width = _su['cores'], _su['threads'], _su['width']
    chain = _su['chain']
    # the PIXEL cell, converted from metres in _fit3d_setup: the block
    # lattice below is scene-pixel arithmetic
    cell = _su['cell']
    wy, wx, _pey, _pex = _su['window']
    y_org, x_org = _su['origin']
    dy, dx = _su['step']
    date_values, bp = _su['date_values'], _su['bperp']

    # ---- PASS 1: the cascade scan, one chunk at a time ---------------
    # Each chunk is read WITH A FULL DS WINDOW OF HALO, as arcs() reads
    # it: the owned pixels' own windows have to be complete, and the cells
    # the chunk owns must be able to see every pixel that could win them.
    # What comes back is the owned rank raster and the chunk's winner
    # grid -- one candidate per independence cell.
    _blocks = []
    _grid = {}
    _hy2, _hx2 = wy // 2, wx // 2
    for key, ds in zip(_keys, _dss):
        da_xr = ds[pol]
        if da_xr.dims[0] != 'date':
            da_xr = da_xr.transpose('date', ...)
        # NOT RECHUNKED HERE. Every window below is read through
        # `_Fit3dSlice`, which assembles the blocks it is given inside the
        # fit task; forcing the date axis into one chunk first only adds a
        # merge layer to the graph and materialises the window twice. The
        # caller's chunking is the caller's to state.
        dsk = da_xr.data
        _ny, _nx = dsk.shape[1], dsk.shape[2]
        yv = np.asarray(ds['y'].values, dtype=float)
        xv = np.asarray(ds['x'].values, dtype=float)
        boy = int(round((float(yv[0]) - y_org) / dy)) if yv.size else 0
        box = int(round((float(xv[0]) - x_org) / dx)) if xv.size else 0
        _cy, _cx = dsk.chunks[1], dsk.chunks[2]
        _y0 = np.r_[0, np.cumsum(_cy)][:-1]
        _x0 = np.r_[0, np.cumsum(_cx)][:-1]
        _grid[key] = (len(_cy), len(_cx), da_xr, ds)
        for _i in range(len(_cy)):
            gy0, gy1 = int(_y0[_i]), int(_y0[_i]) + int(_cy[_i])
            ya, yb = max(0, gy0 - wy), min(_ny, gy1 + wy)
            for _j in range(len(_cx)):
                gx0, gx1 = int(_x0[_j]), int(_x0[_j]) + int(_cx[_j])
                xa, xb = max(0, gx0 - wx), min(_nx, gx1 + wx)
                _gate = chain.gate()
                # ONE LATTICE, THE SCENE'S. The scan cuts its cells on
                # the lattice of the origin it is handed, and the origin
                # is handed in scene pixels below -- so the part's first
                # cell is exact integer arithmetic, never a rounded
                # burst-lattice reconstruction: a burst whose origin is
                # not a multiple of the half-window would land its whole
                # grid up to half a cell off, duplicating nodes at seams.
                # ON THE PS LATTICE: _cascade_pass1 lays its winner grid
                # on `_3d_ps_lattice(cell)` and names the first cell as
                # ceil(scene_origin / lattice); the merge in _fit3d_select_ps
                # places the grid by THIS cell index, so any other unit
                # here puts every block at the wrong offset and lets the
                # next block overwrite part of the previous one -- a hole
                # through the middle of the scene and thinned chunk edges
                _pl = _3d_ps_lattice(cell)
                _cell = (-(-(boy + gy0) // _pl[0]),
                         -(-(box + gx0) // _pl[1]))
                _owned = (gy0 - ya, gy1 - ya, gx0 - xa, gx1 - xa)
                # nout: the rank raster and the winner grid leave the scan
                # as SEPARATE keys, so the one level-1 task depends on the
                # winners alone. Returned as one value they travel
                # together, and every block's raster would be shipped to
                # the worker that runs the PS test and never read there.
                _part = _dask.delayed(_fit3d_scan_for_dask, nout=4)(
                    _Fit3dSlice(dsk[:, ya:yb, xa:xb]), _owned,
                    (boy + ya, box + xa),
                    _cell, _kw_of[key], _threads, _gate)
                chain.push(_part[2])
                _sub = dsk[:, gy0:gy1, gx0:gx1]
                # LEVEL 2 READS WIDER THAN IT WRITES. Its partners are
                # level-1 nodes another chunk owns and solved; the values
                # come from that chunk's table, but the ARC still needs
                # their phasor series, and only a haloed read has them.
                # Candidates stay owned-only, so nothing in the halo is
                # ever attached here -- it is there to be attached TO.
                #
                # ITS OWN, NARROWER SLICE. The scan above needs a FULL
                # window per side -- a winner cell reaches half a window
                # past the owned edge and gating those pixels needs their
                # windows whole -- but level 2 only reaches +-wy//2, the
                # window box. Sharing the scan's slice made every level
                # read twice the halo it uses, on every block of every
                # level, which is pure memory.
                _hy, _hx = max(wy // 2, 1), max(wx // 2, 1)
                _ya2, _yb2 = max(0, gy0 - _hy), min(_ny, gy1 + _hy)
                _xa2, _xb2 = max(0, gx0 - _hx), min(_nx, gx1 + _hx)
                _own2 = (gy0 - _ya2, gy1 - _ya2, gx0 - _xa2, gx1 - _xa2)
                _blocks.append((key, _sub, (gy0 + boy, gx0 + box),
                                int(_cy[_i]), int(_cx[_j]), _part,
                                _i, _j, _cell,
                                dsk[:, _ya2:_yb2, _xa2:_xb2],
                                (_ya2 + boy, _xa2 + box), _own2))

    return dict(_su, grid=_grid, blocks=_blocks)


def _fit3d_union(self, threshold, window, cell, baseline, level,
                 max_dh, max_dv, step_dh, step_dv, max_seasonal,
                 consensus, err_dh, err_dv, iterations, debug,
                 chain=None, tag='', ep=None):
    """One network over the union of the bursts, returned on the burst grid.

    A node's partners are whatever lies inside the window, and a burst edge
    is not a fact about the ground: solved per burst, a node near the seam
    reaches only half its neighbourhood and the two bursts answer the same
    question from different networks. Merged, the network crosses the seam.

    ONLY THE NODES ARE MERGED, NEVER THE RASTERS. Each block is scanned
    where it is stored and yields a node table -- some thousands of phasor
    columns -- and those tables are all the shared solve ever sees. Merging
    the stacks instead, into one array over the scene, makes every block as
    wide as the scene: the chunking the caller asked for stops applying,
    and the host pays tens of gigabytes to carry a network that weighs
    megabytes. arcs() unions its winner grids the same way, for the same
    reason.

    Only the SOLVE is unioned. The model comes back on each burst's own
    grid, carrying only that burst's pixels, so nothing downstream sees a
    different geometry than it handed in.
    """
    import numpy as np
    import xarray as xr
    import dask as _dask
    import dask.array as da
    from .Batch import Batch
    _s = _fit3d_scan(self, threshold, window, cell, baseline, level,
                          max_dh, max_dv, step_dh, step_dv, max_seasonal,
                          consensus, err_dh, err_dv, iterations, debug,
                          chain=chain, tag=tag, ep=ep)
    _keys, _grid, _blocks = _s['keys'], _s['grid'], _s['blocks']
    chain = _s['chain']
    _kw_of, _kw_net = _s['kw_of'], _s['kw_net']
    _cores, _width = _s['cores'], _s['width']
    date_values, bp = _s['date_values'], _s['bperp']
    # ---- LEVEL 1: the PS test over the WHOLE scene's candidates ------
    # The winner grids are merged onto one cell lattice and tested
    # together. A chunk here is a fraction of the PS extent wide, so a
    # chunk-local test asks each candidate about partners that mostly do
    # not exist in its chunk; on the merged grid it is asked about all of
    # them. The grid is the raster one pyramid level up, so the scene's
    # whole candidate set is megabytes and this is one in-memory task.
    _net = _dask.delayed(_fit3d_level1_for_dask)(
        [((b[5][1], b[5][2], b[5][3]), b[8]) for b in _blocks], _kw_net,
        _cores)

    # ---- PASS 2: the shared network written onto each burst's grid ----
    # THE THREAD BUDGET IS PER CONCURRENT TASK, NOT PER SLOT. This pass
    # submits one task per block and the gate runs at most `_width` of them
    # at once, so when there are fewer blocks than slots the cores divided
    # among the slots are divided among tasks that do not exist and the
    # rest of the machine idles. Attachment is the fitting-heavy stage, so
    # that idleness is the wall clock.
    _conc2 = max(1, min(int(_width) if _width else len(_blocks),
                        len(_blocks)))
    _threads2 = max(1, _cores // _conc2)
    _cells = {}
    _dly = {}
    _l2tab = []
    for (key, _sub, _origin, _ny, _nx, _part, _i, _j, _c,
         _hsub, _horg, _hown) in _blocks:
        _kwb = _kw_of[key]
        _gate = chain.gate()
        # The table is a few megabytes; the planes stay put. It emits
        # although nothing consumes the nodes, because the table is how the
        # level's numbers reach its report.
        # THE CALLER'S LEVEL, CAPPED AT ONE. This stage IS level 1. Capping
        # instead of hardcoding keeps `level=0` meaning what it documents: the
        # PS network alone, every other pixel NaN.
        _o = _dask.delayed(_fit3d_attach_for_dask, nout=2)(
            _Fit3dSlice(_sub), _part[0], _net,
            dict(_kwb, level=min(int(level), 1)),
            _origin, _threads2, _gate, emit_nodes=True)
        chain.push(_o[0])
        _l2tab.append(_o[1])
        _cells[(key, _i, _j)] = da.from_delayed(
            _o[0], shape=(6, _ny, _nx), dtype=np.complex64)
        _dly[(key, _i, _j)] = _o[0]

    # THE LEVEL-1 REPORT, once every block of it has finished. At
    # level=1 nothing consumes the tables, so the report is hung off a
    # plane -- the only way to tell the graph to run it at all.
    _rep1 = _dask.delayed(_fit3d_level_report)(1, _l2tab, bool(debug),
                                               str(tag))
    _kk0 = next(iter(_cells))
    _ny0, _nx0 = _cells[_kk0].shape[1], _cells[_kk0].shape[2]
    # THE DELAYED, NOT THE ARRAY: a dask array handed to delayed() is
    # finalized under new keys that carry a second copy of its whole
    # upstream graph, so every scan, the network and one attach ran
    # twice per compute
    _cells[_kk0] = da.from_delayed(
        _dask.delayed(_fit3d_keep)(_dly[_kk0], _rep1),
        shape=(6, _ny0, _nx0), dtype=np.complex64)

    model_result = {}
    for key in _keys:
        _nr, _nc, da_xr, ds = _grid[key]
        both = da.concatenate(
            [da.concatenate([_cells[(key, i, j)] for j in range(_nc)],
                            axis=2) for i in range(_nr)], axis=1)
        model_result[key] = _fit3d_model(ds, da_xr, both, date_values, bp)
    return Batch(model_result)


def fit1d_network(self, threshold: float = 0.5, window: tuple = (400, 4000),
            cell: 'float | tuple' = 24,
            baseline: str = 'BPR',
            level: int = 1,
            max_dh: float = 25.0, max_dv: float = 25.0,
            step_dh: float = 8.0, step_dv: float = 2.0,
            max_seasonal: float = 0.0,
            consensus: int = 3,
            err_dh: float = 4.0, err_dv: float = 1.0,
            union: bool = False,
            iterations: int = 8,
            debug: bool = False) -> 'Batch':
    """BatchComplex.fit1d_network() (its docstring says what it does): the PS network and level 1."""
    # the baselines and the radar geometry are the Dataset's (N81)
    BatchCore._needs_dataset(self, 'fit1d_network', ('BPR',) + BatchCore._GEOMETRY)
    # DELEGATE BY STACK TYPE, exactly as fit1d() does: pairs take the
    # pair branch (not implemented), dates take the PS network below.
    if any('pair' in ds[v].dims
           for ds in self.values() for v in ds.data_vars):
        # `window` is the PS network's (DS, PS extent) pair in metres on
        # the date path; on the pair path it is the side of the box the
        # covariance is estimated over, so a scalar.
        # the split is kept so a pair-domain fit has a home when one works
        raise NotImplementedError(
            'fit1d_network() does not support complex PAIRS. Use the per-DATE stack, '
            'or unwrap and call Batch.fit1d() on the unwrapped pairs.')
    # validated HERE, not only in the kernel: fit1d_network() returns a lazy Batch,
    # so a bad value would otherwise surface at compute time far from where
    # it was written
    _3d_consensus(consensus)
    if int(level) < 0:
        raise ValueError(f'level must be >= 0; got {level!r}')
    # LEVEL 2+ IS DROPPED (the user, 2026-10-24): it added about 30 % more DS of
    # clearly lower accuracy at a high cost in time and memory
    if int(level) >= 2:
        print(f'WARNING: fit1d_network() level={level} is dropped; level 1 is used.')
        level = 1
    _kw = dict(threshold=threshold, window=window, cell=cell,
               baseline=baseline, level=level,
               max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
               step_dv=step_dv, max_seasonal=max_seasonal,
               consensus=consensus, err_dh=err_dh, err_dv=err_dv,
               iterations=iterations, debug=debug)
    if union or len(self) == 1:
        return _fit3d_union(self, **_kw)
    # union=False: EACH BURST ITS OWN SCENE, as trend2d() fits each burst
    # on its own pixels. The same driver runs once per burst -- there is
    # no second code path -- and the bursts share ONE chain, so the
    # cluster never holds more heavy tasks at once than a single scene
    # would. Where bursts overlap the two models disagree over the shared
    # ground, exactly as two per-burst trends do; union=True is the way
    # to ask for one answer there.
    from .Batch import Batch
    chain = _Fit3dChain()
    # ONCE FOR THE CALL, not once per burst: a missing ref_height warns
    # once, naming every burst. Each burst's value is its own either way.
    ep = Batch._elevation_phase_approximate(self)
    out = {}
    for k in self.keys():
        out.update(_fit3d_union(type(self)({k: self[k]}), 
            chain=chain, tag=f' [{k}]', ep=ep, **_kw).items())
    return Batch(out)


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _3d_lap_solve(np.array([0, 1]), np.array([1, 2]), np.ones(2), 3,
                  np.ones((2, 2)))
    _cascade_row(np.zeros((2, 3), np.float32), 0, 1, 0, 1, 1, 1, 1, 0, 0,
                 np.int32(1), 0, 0, np.zeros((1, 1), np.uint8),
                 np.zeros((1, 3), np.int32), np.zeros((1, 3), np.int32),
                 np.zeros((1, 3), np.int32), np.zeros((1, 3), np.int32),
                 np.zeros((1, 1), np.int32), np.zeros((1, 1), np.int32),
                 np.zeros((1, 1), np.int32), np.zeros((1, 1), np.int32),
                 np.zeros(3, np.float32), np.zeros(3, np.int32))


_warmup_numba_cache()
