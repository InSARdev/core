"""Arc counting and the atmospheric solve on a persistent-scatterer network.

An arc is the double difference between two pixels on the same dates, so the
atmospheric screen cancels in it and the arc measures pixel quality with the
atmosphere already removed. Two pixels closer than the INDEPENDENCE CELL in
both axes are one sample of the ground, not two, so a coherence between them
measures the sensor's impulse response rather than the terrain -- every arc
here clears the cell.

  `arcs()`     -> sparse selection of independent, connected pixels
  `fit3d()`      -> best arc coherence selects the nodes; a joint (height, velocity)
                  model is fitted on the arcs, integrated over the network, and
                  the temporally-rough remainder is the atmospheric phase.
"""
import numba as _numba
import numpy as np
import threading as _threading
import collections.abc as _abc
import time
import warnings


_RAYLEIGH_MEAN = np.sqrt(np.pi) / 2.0
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

# How many partners each pixel offers the spanning forest, ranked by raw
# coherence. The forest takes only what connectivity needs -- about one arc per
# node -- so this only has to be deep enough that a pixel's usable partner is
# among them.
_ARC_CANDIDATES = 6

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

_LE123 = (('le', 1.0), ('le', 2.0), ('le', 3.0))
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
def _3d_arc_offsets(window_y, window_x, cell_y=2, cell_x=8):
    """Neighbour offsets inside the window that are not the same sample.

    A pair is INADMISSIBLE when it lies inside `cell` in BOTH axes:
    |dy| < cell_y and |dx| < cell_x. The default (2, 8) is 16 x 16 m on a
    standalone Sentinel-1 IW grid (8 x 2 m, no upscaling) -- isotropic in
    ground units, which a rectangular pixel grid is not.

    This replaces a measured independence cell, which did not measure what its
    name said. That estimate came from how far coherence stays high with
    offset, and coherence stays high wherever the SCENE stays similar -- land
    cover, terrain -- not merely across the impulse response. It therefore
    returned cells many times the ground-range resolution, varying block to
    block, and excluding out that far discarded real scatterers a few metres
    apart: far more candidates survive once the exclusion is cut to the
    resolution scale.

    The window is the BOX, so the offsets run to half of it either way:
    (16, 64) on an 8 x 2 m grid is 128 x 128 m of ground and the offsets reach
    +-64 m. A scatterer whose only partners lie beyond that needs a bigger
    window -- widening the reach inside a given window would just relabel the
    argument.

    A pair is INADMISSIBLE only when the two pixels touch. Such a pair is a
    single sample measured twice, so its coherence reflects the impulse
    response, not the ground. Oversampled copies still enter the network --
    they are independent of any ORIGINATOR further away, and are found there.
    """
    return [(dy, dx)
            for dy in range(-(window_y // 2), window_y // 2 + 1)
            for dx in range(-(window_x // 2), window_x // 2 + 1)
            if (dy or dx) and not (abs(dy) < cell_y and abs(dx) < cell_x)]

@_numba.njit(nogil=True, cache=True)
def _3d_ds_partners(Uv, Ub, cy, cx, fg, fyi, fxi, hy, hx,
                    cell_y, cell_x, kk, thr, out_v, out_j, lo, hi):
    """Each candidate's best `kk` FIXED partners inside its DS window, one
    per independence cell.

    ONE BOX, THE WINDOW'S HALF-EXTENT AS RADIUS. The DS window is the range
    over which the caller states the atmosphere is common, and every partner
    a DS is measured against must lie inside it; every candidate is offered
    the same box. The search used to walk a quarter-radius box first and the
    half-radius box only for candidates the small box could not fill, so a
    candidate's partners depended on how dense its neighbourhood happened to
    be -- the nearest few in a dense one, the whole window in a sparse one --
    which is two definitions of the same measurement.

    THE SLOTS HOLD INDEPENDENT SAMPLES. Two fixed nodes closer than the
    independence cell in both axes are one sample of the ground measured
    twice, and their agreement says nothing: they agree because they are the
    same measurement. So a partner inside the cell of one already held
    competes for THAT slot -- the better of the two keeps it -- and never
    takes a second; the `kk` partners a candidate ends up with are pairwise
    at least a cell apart, and a consensus among them is a consensus of
    distinct samples. `fyi`, `fxi` give each fixed node's pixel so the held
    slots can be placed.

    Nothing is materialised per pair: a candidate holds `kk` slots and a
    partner either displaces the weakest or is forgotten, so the working set
    is `kk` per candidate rather than one entry per pair in reach.

    `fg` carries the fixed node's index at its pixel and -1 elsewhere, which
    is what makes the search bipartite -- a candidate never partners another
    candidate, whose value does not exist yet.
    """
    n = Uv.shape[0]
    ny, nx = fg.shape
    # SERIAL ON PURPOSE, threaded by the caller in candidate bands. A
    # parallel=True kernel called from a multi-threaded dask worker trips
    # numba's workqueue layer -- "not threadsafe ... concurrent access has
    # been detected" -- and takes the whole worker down, surfacing only as a
    # crash. Every other kernel here is serial for the same reason, and a
    # candidate writes only its own row, so bands need no coordination.
    for c in range(lo, hi):
        for s in range(kk):
            out_v[c, s] = -1.0
            out_j[c, s] = -1
        yc = cy[c]
        xc = cx[c]
        if True:
            ya = yc - hy
            if ya < 0:
                ya = 0
            yb = yc + hy + 1
            if yb > ny:
                yb = ny
            xa = xc - hx
            if xa < 0:
                xa = 0
            xb = xc + hx + 1
            if xb > nx:
                xb = nx
            for py in range(ya, yb):
                dy = py - yc
                ady = dy if dy >= 0 else -dy
                for px in range(xa, xb):
                    j = fg[py, px]
                    if j < 0:
                        continue
                    dx = px - xc
                    adx = dx if dx >= 0 else -dx
                    # ONE SAMPLE OF THE GROUND is not an arc: inside the
                    # independence cell the two pixels share an impulse
                    # response and their coherence reports it, not the terrain.
                    if ady < cell_y and adx < cell_x:
                        continue
                    sr = 0.0
                    si = 0.0
                    for d in range(n):
                        ar = Uv[d, c].real
                        ai = Uv[d, c].imag
                        br = Ub[d, j].real
                        bi = Ub[d, j].imag
                        sr += ar * br + ai * bi
                        si += ai * br - ar * bi
                    v = np.sqrt(sr * sr + si * si) / n
                    mi = 0
                    mv = out_v[c, 0]
                    for s in range(1, kk):
                        if out_v[c, s] < mv:
                            mv = out_v[c, s]
                            mi = s
                    # BELOW THE WEAKEST SLOT IT CAN ENTER NOTHING: not a free
                    # or weakest slot, and not a same-cell slot either, whose
                    # holder is at least as good as the weakest. Most visits
                    # end here, before the cell scan below is paid for.
                    if v <= mv:
                        continue
                    # HELD PARTNERS INSIDE THIS ONE'S CELL are the same sample
                    # as it. If the best of them is at least as good, the
                    # newcomer is dropped and the held set, pairwise
                    # independent already, stays as it is. If the newcomer is
                    # better it takes the place of EVERY held partner in its
                    # cell -- not just the first met -- because two held
                    # partners can each be within a cell of it without being
                    # within a cell of each other, and replacing one would
                    # leave the other beside the newcomer. The invariant kept
                    # is that the held partners are pairwise independent.
                    same = -1
                    best = -1.0
                    for s in range(kk):
                        jj = out_j[c, s]
                        if jj < 0:
                            continue
                        ddy = fyi[jj] - py
                        if ddy < 0:
                            ddy = -ddy
                        ddx = fxi[jj] - px
                        if ddx < 0:
                            ddx = -ddx
                        if ddy < cell_y and ddx < cell_x:
                            if same < 0:
                                same = s
                            if out_v[c, s] > best:
                                best = out_v[c, s]
                    if same >= 0:
                        if v > best:
                            for s in range(kk):
                                jj = out_j[c, s]
                                if jj < 0 or s == same:
                                    continue
                                ddy = fyi[jj] - py
                                if ddy < 0:
                                    ddy = -ddy
                                ddx = fxi[jj] - px
                                if ddx < 0:
                                    ddx = -ddx
                                if ddy < cell_y and ddx < cell_x:
                                    out_v[c, s] = -1.0
                                    out_j[c, s] = -1
                            out_v[c, same] = v
                            out_j[c, same] = j
                        continue
                    out_v[c, mi] = v
                    out_j[c, mi] = j


def _3d_arcs_kernel(block, window_y, window_x, cell=(2, 8), budget=None,
                    topk=None, topk_mask=None, threads=1):
    """The BEST arc coherence each pixel reaches -- its PS quality.

    For every admissible separation (dy, dx) the whole raster is correlated
    against itself shifted by it, in one pass:

        gamma(dy, dx) = |sum_d u[d, y, x] * conj(u[d, y+dy, x+dx])| / n_valid

    and each pixel keeps the MAXIMUM over all separations. Not a count of how
    many partners cleared some level, and not a mean: a count reports how
    crowded a neighbourhood is, and a mean is dragged to the noise floor by the
    many partners any pixel has that are simply unrelated to it. The maximum
    answers the only question the selection asks -- does this pixel have a
    partner it agrees with.

    No tiles. Tiling made a pixel's answer depend on where it fell relative to
    the tile edges; sliding the whole array by each offset evaluates every pair
    exactly once, at full resolution, and credits it to BOTH endpoints, so half
    the offsets suffice.

    A pair is skipped when it lies inside the independence cell in BOTH axes:
    those two pixels are one sample of the ground, so their coherence reports
    the impulse response and not the terrain.

    The neighbourhood is a BOX CENTRED on the pixel: separations run to
    +-window//2, so `window` is the full extent, not the reach in one
    direction.

    budget sizes the transient working set and MUST be resolved by the
    caller, in the main process: dask workers are separate processes and do
    not inherit dask.config, so reading `array.chunk-size` in here would
    return the 128 MB default whatever the notebook set (see Stack.py:1435
    for the same trap). None falls back to that read, which is right only
    when the kernel is called directly.

    block : (n_dates, ny, nx) complex
    cell  : (dy, dx) independence cell in pixels. If its exclusion covers the
            whole +-window//2 box no pair is admissible and the result is all
            NaN -- an unmeasurable setting reports nothing rather than a
            number that looks like an answer.

    Returns (ny, nx) float32, the best arc coherence per pixel; NaN where no
    arc was observable -- a pixel that cannot be assessed is not a pixel that
    failed. Threshold it yourself, e.g. `>= 0.6`.

    topk : int or None
        With a count, additionally return the BEST `topk` partners per pixel
        rather than only the best one, as (coherence, dy, dx) arrays shaped
        (ny, nx, topk). The correlation block this reads from is the one the
        maximum is already taken over, so the extra cost is a partial sort of
        values that were computed anyway -- nothing like searching pair by
        pair. A pixel with fewer admissible partners than `topk` has its
        remaining slots at coherence -1 and offset 0.

        This is what a network over these pixels needs and the maximum cannot
        give: which partners, and where.

    topk_mask : (ny, nx) bool or None
        The pixels this call is about. They are the only ones that collect
        partners AND the only ones that can be partners, so a mask makes the
        kernel work over that selection alone -- the caller does not zero a
        copy of the scene to express it.

        The correlation stays dense, which is what makes it fast; what the
        mask removes is the top-`topk` bookkeeping at pixels nobody asked
        about. A network is built over a few per cent of a raster, and
        keeping a sorted list for the other 97% is the whole cost of the
        option: the correlation is a BLAS product, the bookkeeping is a
        partial sort and a scatter per tile.
    """
    S = np.asarray(block)
    n, ny, nx = S.shape
    wy, wx = int(window_y), int(window_x)
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    hy, hx = wy // 2, wx // 2
    if n < 2 or ny == 0 or nx == 0:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    # ROW BANDS WITH AN hy HALO. A pixel's partners live within +-hy rows, so
    # a band's own rows carry the full-raster answer; the halo pairs are
    # recomputed, which is the whole price. Only a caller that OWNS the host
    # may raise `threads` -- the fit3d gate does, a per-chunk task must not.
    _th = max(1, int(threads))
    if _th > 1 and ny >= 2 * (hy + 1):
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        H = max(hy + 1, -(-ny // _th))
        bands = [(a, min(a + H, ny)) for a in range(0, ny, H)]
        kk0 = int(topk) if topk else 0
        best_o = np.empty((ny, nx), np.float32)
        if kk0:
            tv_o = np.empty((ny, nx, kk0), np.float32)
            ty_o = np.empty((ny, nx, kk0), np.int16)
            tx_o = np.empty((ny, nx, kk0), np.int16)

        def _band(band):
            ya, yb = band
            a0 = max(0, ya - hy); b0 = min(ny, yb + hy)
            m = None if topk_mask is None else np.asarray(topk_mask)[a0:b0]
            r = _3d_arcs_kernel(S[:, a0:b0], wy, wx, (cy, cx), _mb,
                                topk=topk, topk_mask=m)
            sl = slice(ya - a0, yb - a0)
            if kk0:
                best_o[ya:yb] = r[0][sl]
                tv_o[ya:yb] = r[1][sl]
                ty_o[ya:yb] = r[2][sl]
                tx_o[ya:yb] = r[3][sl]
            else:
                best_o[ya:yb] = r[sl]
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
        return (best_o, tv_o, ty_o, tx_o) if kk0 else best_o

    # ---- BUILD THE OPERAND IN SLABS -------------------------------------
    # The whole-pixel rule first: a pixel is inside the radar extent or it is
    # not, and out there the samples are noise. A pixel valid on all but a few
    # dates is not a scatterer, and keeping it would cost a per-PAIR valid
    # count -- a second reduction as large as the phasor one, half the
    # arithmetic -- to serve a negligible share of pixels. Dropped, n_valid is
    # the constant n for every surviving pair and stops being computed at all.
    #
    # Built a slab of rows at a time, because materialising |S|, the unit
    # phasors and their real and imaginary planes whole costs several times the
    # output array in temporaries. A slab bounds that, and the peak becomes the
    # operand itself.
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
    ok = np.zeros((ny, nx), dtype=bool)
    slab = max(1, min(ny, int(64 * 1024 * 1024 // max(n * nx * 8, 1))))
    for y0 in range(0, ny, slab):
        y1 = min(y0 + slab, ny)
        blk = S[:, y0:y1, :]
        a = np.abs(blk)
        f = np.isfinite(a) & (a > 0)
        o = f.all(axis=0)
        if topk_mask is not None:
            # THE SELECTION IS THE RASTER, as far as this call is concerned.
            # A network is built over chosen pixels, so the unchosen may
            # neither collect a partner nor BE one -- and applying that here,
            # where observability is already decided, spares the caller
            # zeroing a copy of the scene to say the same thing.
            o = o & np.asarray(topk_mask, bool)[y0:y1]
        ok[y0:y1] = o
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.where(f, blk / np.where(f, a, 1), 0)
        u *= o[None, :, :]
        Xp[y0:y1, hx:hx + nx, :n] = np.moveaxis(u.real, 0, -1)
        Xp[y0:y1, hx:hx + nx, n:] = np.moveaxis(u.imag, 0, -1)
        del blk, a, f, o, u

    # ---- 2-D TILES ------------------------------------------------------
    # gamma(p, d) = |sum_dates u[p] conj(u[p+d])| / n is an inner product, so
    # the offsets at a fixed row separation are a BAND of a matrix product.
    # Walking one offset at a time re-reads the whole array once per offset --
    # far more memory traffic than arithmetic, so it runs nowhere near compute
    # bound.
    #
    # The shape of the product decides everything: a matmul small in BOTH
    # dimensions leaves most of the machine idle, while a wide right-hand
    # operand reaches full rate. So one tile gathers ALL the row separations
    # into N at once: a (Bx x K) @ (K x (hy+1)*span) per tile, trading a gather
    # for a much faster matmul.
    #
    # The dy halo is ONE-SIDED (0..hy, and dx > 0 at dy == 0) so each pair is
    # still evaluated once and credited to both of its ends; a symmetric halo
    # would double the arithmetic for nothing.
    #
    # The complex product needs only TWO real matrix products: with
    # A1 = [Re, Im] and A2 = [Im, -Re] against the same operand, A1 @ B is the
    # real part and A2 @ B the imaginary, contracting over 2n.
    # SEEDED BELOW ZERO, not at it. A pixel whose window holds no admissible
    # partner -- every neighbour inside the independence cell, or the window
    # clipped at a block corner -- would otherwise keep its seed and be
    # returned as coherence 0.0, which reads as a measured failure rather than
    # as nothing measured. The accumulator only ever takes maxima of squared
    # magnitudes, so a negative seed cannot be reached by a real arc and marks
    # exactly the pixels no pair ever touched.
    best = np.full((ny, nx), -1.0, dtype=np.float32)
    kk = int(topk) if topk else 0
    if kk:
        want = (np.ones((ny, nx), bool) if topk_mask is None
                else np.asarray(topk_mask, bool))
        tk_v = np.full((ny, nx, kk), -1.0, dtype=np.float32)
        tk_y = np.zeros((ny, nx, kk), dtype=np.int16)
        tk_x = np.zeros((ny, nx, kk), dtype=np.int16)

        def _merge(sy, sx, v, oy, ox):
            """Keep the best `kk` of what is held and what just arrived."""
            av = np.concatenate([tk_v[sy, sx], v], axis=1)
            ay = np.concatenate([tk_y[sy, sx], oy], axis=1)
            ax = np.concatenate([tk_x[sy, sx], ox], axis=1)
            j = np.argpartition(av, -kk, axis=1)[:, -kk:]
            tk_v[sy, sx] = np.take_along_axis(av, j, axis=1)
            tk_y[sy, sx] = np.take_along_axis(ay, j, axis=1)
            tk_x[sy, sx] = np.take_along_axis(ax, j, axis=1)
    masks = {}
    # Bx FOLLOWS THE HALO, BUT IS CAPPED. The block computes Bx x (Bx + 2 hx)
    # pairs of which Bx x (2 hx + 1) are wanted, so a small block wastes little
    # but gives BLAS a thin matrix and a large one recomputes the halo.
    # Runtime is flat between hx and 2 hx, so Bx follows hx.
    #
    # The cap exists because the tile's working set grows as hx^2 hy: the
    # score block is Bx x (hy+1) x (Bx + 2 hx) floats and appears twice, so a
    # large window reaches hundreds of MB per tile -- and dask runs one per
    # thread.
    # Shrinking Bx bounds it without touching the result, and costs nothing at
    # the window sizes where Bx is already below the cap.
    # The ceiling is the DASK CHUNK BUDGET, as core sizes every working set
    # (utils_dask.rechunk3d, BatchCore.velocity) -- not a constant of its own.
    # One setting the caller has already tuned for its machine governs this
    # too, so a large window cannot silently allocate hundreds of MB per dask
    # thread while still honouring a raised budget when there is room.
    tile_cap = _3d_budget_mb(budget) * 1024 * 1024
    Bx = max(1, hx)
    while Bx > 8:
        span_ = Bx + 2 * hx
        if (hy + 1) * span_ * 4 * (K + 2 * Bx) <= tile_cap:
            break
        Bx //= 2
    for y in range(ny):
        ndy = min(hy + 1, ny - y)
        for x0 in range(0, nx, Bx):
            w = min(Bx, nx - x0)
            span = w + 2 * hx
            # the tile's own pixels, and the same vectors rotated for Im
            A1 = Xp[y, hx + x0:hx + x0 + w, :]
            A2 = np.empty((w, K), dtype=np.float32)
            A2[:, :n] = A1[:, n:]
            A2[:, n:] = -A1[:, :n]
            # every partner the tile can reach, as one (K, ndy*span) operand
            Bk = np.ascontiguousarray(
                Xp[y:y + ndy, x0:x0 + span, :].transpose(2, 0, 1)
            ).reshape(K, ndy * span)
            t = A1 @ Bk
            Ci = A2 @ Bk
            np.multiply(t, t, out=t)
            np.multiply(Ci, Ci, out=Ci)
            t += Ci
            del Ci, Bk, A2
            key = (w, ndy)
            if key not in masks:
                dxm = (np.arange(span)[None, None, :] - hx
                       - np.arange(w)[:, None, None])
                dyv = np.arange(ndy)[None, :, None]
                mm = ((np.abs(dxm) <= hx)
                      & ~((dyv < cy) & (np.abs(dxm) < cx))
                      & ~((dyv == 0) & (dxm <= 0)))
                masks[key] = mm.reshape(w, ndy * span).astype(np.float32)
            t *= masks[key]
            tv = t.reshape(w, ndy, span)
            # the pixel's own best, and the same values at the partner ends
            np.maximum(best[y, x0:x0 + w], tv.max(axis=(1, 2)),
                       out=best[y, x0:x0 + w])
            tg = tv.max(axis=0)                     # (ndy, span)
            gx0 = x0 - hx
            xa, xb = max(0, gx0), min(nx, gx0 + span)
            if xb > xa:
                np.maximum(best[y:y + ndy, xa:xb], tg[:, xa - gx0:xb - gx0],
                           out=best[y:y + ndy, xa:xb])
            if kk:
                # THE SAME BLOCK, PARTIALLY SORTED. `tv` holds every partner
                # this tile can reach; the maximum above is one reduction of
                # it and the best `kk` is another.
                wsel = np.flatnonzero(want[y, x0:x0 + w])
                if len(wsel):
                    fl = tv.reshape(w, -1)[wsel]
                    m_ = min(kk, fl.shape[1])
                    j = np.argpartition(fl, -m_, axis=1)[:, -m_:]
                    vv = np.take_along_axis(fl, j, axis=1)
                    oy = (j // span).astype(np.int16)
                    ox = ((j % span) - hx
                          - wsel[:, None]).astype(np.int16)
                    _merge(y, x0 + wsel, vv, oy, ox)
                # and credited to the partner ends, where the offset reverses
                if xb > xa:
                    nxs = xb - xa
                    wm = want[y:y + ndy, xa:xb].ravel()
                    if wm.any():
                        tpm = tv[:, :, xa - gx0:xb - gx0].transpose(
                            1, 2, 0).reshape(ndy * nxs, w)[wm]
                        m2 = min(kk, w)
                        j2 = np.argpartition(tpm, -m2, axis=1)[:, -m2:]
                        v2 = np.take_along_axis(tpm, j2, axis=1)
                        gy = np.repeat(np.arange(ndy), nxs)[wm][:, None]
                        gx = np.tile(np.arange(xa, xb), ndy)[wm][:, None]
                        o2y = np.broadcast_to((-gy).astype(np.int16), v2.shape)
                        o2x = ((x0 + j2) - gx).astype(np.int16)
                        ry = (y + np.repeat(np.arange(ndy), nxs)[wm])
                        rx = np.tile(np.arange(xa, xb), ndy)[wm]
                        _merge(ry, rx, v2, o2y, o2x)
                        del tpm, j2, v2
            del t, tv, tg

    seen = best >= 0
    out = np.sqrt(best, out=best, where=seen) / n
    res = np.where(ok & seen, out, np.nan).astype(np.float32)
    if kk:
        good = tk_v > 0
        _sq = np.full(tk_v.shape, -1.0, dtype=np.float32)
        np.sqrt(tk_v, out=_sq, where=good)
        tk_v = np.where(good, _sq / n, -1.0).astype(np.float32)
        tk_v[~(ok & seen)] = -1.0
        return res, tk_v, tk_y, tk_x
    return res


def _3d_depth(chunks, window):
    """Halo depth per axis, and the check that the given chunks can carry it.

    `chunks` is (chunks_y, chunks_x) as dask reports them. Returns
    (depth_y, depth_x).

    An arc reaches at most half the PS extent from the pixel, so half is what a
    block must see beyond its own edge. An axis held in ONE chunk already has
    the whole raster and needs no halo -- and asking for one raises, because
    dask refuses a depth wider than the array.

    THE CHUNKING IS THE CALLER'S AND IS NEVER CHANGED HERE, only checked --
    and checked before `da.overlap` is reached, because that calls
    `ensure_minimum_chunksize()`, which silently re-splits any chunk shorter
    than the depth into lengths of its own choosing. Refusing first is what
    keeps the layout the caller asked for.

    Blocks are solved independently, so a different chunking gives a different
    set of scatterers near the seams. That is the design, not an error: a
    pixel at a block edge sees the neighbourhood its block affords.
    """
    # THE DS BOX IS THE REACH. What a block computes beyond its own edge is
    # the neighbourhood the DS window looks over; the PS extent bounds nothing
    # a halo can carry, since a node reaches every other node of its block.
    wy, wx, _, _ = _3d_windows(window)
    depth = []
    for _cs, _w in zip((tuple(chunks[0]), tuple(chunks[1])), (wy, wx)):
        if len(_cs) == 1:
            depth.append(0)
            continue
        if min(_cs) < _w:
            raise ValueError(f'chunk size {min(_cs)} less than processing '
                             f'window size {_w}, enlarge chunks or decrease '
                             f'window')
        depth.append(_w // 2)
    return tuple(depth)


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


def _3d_arcs_select(U, quality, window, threshold, cell=(2, 8)):
    """Sparse, independent, MUTUALLY CONNECTED pixels -- what the network uses.

    Counting arcs per pixel says how well connected a pixel is, but not to
    WHOM, and that distinction decides the network. Ranking candidates
    independently and suppressing a cell box around each winner keeps only a
    fraction of a winner's verified partners, because coherent partnership is a
    specific pairing and independent thinning does not respect it. The
    triangulation that follows then joins pixels never tested against each
    other, so a node's Delaunay neighbours score far below the best partner
    actually available to it and most of those edges fail the arc test.

    Selecting here keeps the pairing, because here the partners are known. Take
    the best-connected unblocked candidate, suppress its cell neighbourhood,
    then test it against the candidates in its window and promote the ones that
    PASS -- they are cell-independent by the same rule and joined to it by an
    arc already known to work -- and carry on from them. Note the partners are
    never inside the seed's own cell: the kernel excluded intra-cell pairs
    before counting, so every partner is an independent sample by construction.

    Same threshold and the kernel's own validity rule: no new parameter.

    Returns (selection, edges): the best arc coherence at the selected pixels with
    NaN everywhere else, and the VERIFIED pairs as (y0, x0, y1, x1) rows. The edges
    are not a by-product to be discarded -- they are the arcs this stage proved
    work, and re-deriving the network by triangulation instead throws them
    away: a pixel selected here on three good close partners can be
    triangulated to whichever nodes happen to be nearest, fail all of them, and
    drop out with degree zero.
    """
    from collections import deque
    n, ny, nx = U.shape
    wy, wx = int(window[0]), int(window[1])
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    cy_, cx_ = cy - 1, cx - 1
    # THE SAME WHOLE-PIXEL RULE THE KERNEL USES. A pixel is inside the radar
    # extent or it is not; out there the samples are noise. Carrying a
    # per-PAIR valid count here would be a second, different notion of
    # validity in the same file -- and it would let a pixel be selected on an
    # arc to noise, which is a false positive by construction, not a weak
    # scatterer. Two stages must not disagree about what a valid pixel is.
    pix_ok = (np.abs(U) > 0).all(axis=0)
    # A pixel is a candidate only if its OWN best arc reached the threshold.
    # The growth below promotes a partner on a measured arc >= threshold, but a
    # SEED is taken untested at _take() -- so with `quality > 0` every pixel
    # that is not blocked becomes a point, whatever its arcs did. Ungated
    # against planted truth, most selected pixels are decorrelated ground and
    # few of the triangulated arcs survive the fit; gated, essentially all of
    # them are real. The seed's best arc is itself a measured arc, so this is
    # the same rule the growth applies, not a second parameter.
    cand = np.isfinite(quality) & (quality >= threshold) & pix_ok
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    edges = []
    if not cand.any():
        return out, np.zeros((0, 4), dtype=np.int64)

    blocked = np.zeros((ny, nx), dtype=bool)
    taken = np.zeros((ny, nx), dtype=bool)
    owner = np.full((ny, nx), -1, dtype=np.int64)
    ys, xs = np.where(cand)
    # rank by the BEST ARC COHERENCE, not the arc count: of a touching
    # group -- one sample seen several times -- the pixel whose best arc works
    # best is the one to keep, whereas the count mostly reports how crowded
    # that pixel's neighbourhood is
    order = np.argsort(-quality[ys, xs], kind='stable')

    def _take(y_, x_):
        taken[y_, x_] = True
        out[y_, x_] = quality[y_, x_]
        sl = (slice(max(0, y_ - cy_), y_ + cy_ + 1),
              slice(max(0, x_ - cx_), x_ + cx_ + 1))
        # remember WHICH node covers each blocked pixel: a blocked partner is
        # within one cell of that node, i.e. the same ground sample, so an arc
        # verified against the partner is an arc against the node
        owner[sl] = np.where(blocked[sl], owner[sl], y_ * nx + x_)
        blocked[sl] = True

    for p in order:
        y0, x0 = int(ys[p]), int(xs[p])
        if blocked[y0, x0]:
            continue
        _take(y0, x0)
        q = deque([(y0, x0)])
        while q:
            yi, xi = q.popleft()
            ya, yb = max(0, yi - wy // 2), min(ny, yi + wy // 2 + 1)
            xa, xb = max(0, xi - wx // 2), min(nx, xi + wx // 2 + 1)
            sub = cand[ya:yb, xa:xb] & ~blocked[ya:yb, xa:xb]
            if not sub.any():
                continue
            ly, lx = np.where(sub)
            gy, gx = ly + ya, lx + xa
            keep = ~((np.abs(gy - yi) < cy) & (np.abs(gx - xi) < cx))
            if not keep.any():
                continue
            gy, gx = gy[keep], gx[keep]
            ok_ = pix_ok[gy, gx]
            if not ok_.any():
                continue
            gy, gx = gy[ok_], gx[ok_]
            g = np.abs(U[:, yi, xi].conj() @ U[:, gy, gx]) / n
            hit = np.where(g >= threshold)[0]
            # Only touching pixels are one sample, and the offsets excluded
            # those already -- everything reaching here is a real partner.
            for k in hit[np.argsort(-g[hit])]:
                yk, xk = int(gy[k]), int(gx[k])
                if not taken[yk, xk]:
                    if blocked[yk, xk]:
                        # the partner is one cell from an existing node, so it
                        # IS that node's sample: keep the arc, against the node
                        o = int(owner[yk, xk])
                        if o >= 0:
                            oy, ox = divmod(o, nx)
                            if not ((abs(oy - yi) < cy) and (abs(ox - xi) < cx)):
                                edges.append((yi, xi, oy, ox))
                        continue
                    _take(yk, xk)
                    q.append((yk, xk))
                edges.append((yi, xi, yk, xk))
    E = (np.array(edges, dtype=np.int64) if edges
         else np.zeros((0, 4), dtype=np.int64))
    return out, E


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


def _3d_arc_fit(arc, ele2phase, t, meter2rad, max_dh=25.0, max_dv=25.0,
                step_dh=8.0, step_dv=2.0, budget=None, max_seasonal=0.0,
                iterations=8, seed_th=None):
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
                seed_th=None if seed_th is None else seed_th[sl])
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
        for _b0 in range(0, m, _blk):
            _sl = slice(_b0, min(_b0 + _blk, m))
            k[_sl] = _3d_argmax_first(np.abs(Z[:, _sl].T @ C))
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
        gam = np.empty(m, dtype=np.float32)
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
            TH[:, 0] = np.round(TH[:, 0] / ch) * ch if not no_h else 0.0
            TH[:, 1] = np.round(TH[:, 1] / cv) * cv
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
            for _b0 in range(0, m, _blk):
                _sl = slice(_b0, min(_b0 + _blk, m))
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


def _3d_arc_fit_brute(arc, ele2phase, t, meter2rad, h_range=150.0, v_range=60.0,
                      h_step=0.5, v_step=0.25, budget=None):
    """Exhaustive (height, rate) scan -- the REFERENCE the ladder is checked against.

    _3d_arc_fit walks a coarse-to-fine ladder and can land in the wrong basin;
    that is not hypothetical, it happened: it returned a low coherence at a
    wildly wrong height where a much better solution existed nearby. Nothing in
    the ladder detects that, because a search cannot
    report a maximum it never visited. An exhaustive scan can.

    Every candidate is scored for every arc as ONE matrix product,

        gamma(a, c) = |sum_d Z[d, a] conj(E[d, c])| / n_valid

    with E the (dates x candidates) model bank, so it is a GEMM like the arc
    kernel's, and a full grid over thousands of arcs stays cheap.

    UNLIKE the ladder this takes a RANGE, which is a prior. That is why it is a
    reference and not the estimator: the ladder's search window is set by the
    baselines alone. Use it to verify, to debug a suspect pixel, or in a test
    that asserts the ladder finds what is there.

    Returns (gamma, height_rad, velocity_rad_yr), each (n_arcs,).
    Height is radians per unit ele2phase and rate is radians per year: the
    library works in phase throughout and only displacement_los()
    converts to a length. `max_dh`/`max_dv` remain physical, since they
    state what the caller wants bounded.
    """
    arc = np.asarray(arc)
    if not np.iscomplexobj(arc):
        raise TypeError(f'arc must be COMPLEX phasors, got {arc.dtype}')
    A = np.abs(arc)
    Z = np.where(A > 0, arc / np.where(A > 0, A, 1.0), 0).astype(np.complex64)
    n, m = Z.shape
    nv = np.maximum((A > 0).sum(axis=0), 1)
    # PHASE THROUGHOUT, as `_3d_arc_fit` does -- the ranges
    # and steps are the caller's, stated in metres and mm/yr, and converted
    # once here so the scan, the gates and the return all speak one unit.
    hh = np.asarray(ele2phase, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    _m2h, _m2v = float(meter2rad), float(meter2rad) * 1e-3
    # (-k, k+1) * step rather than arange(-range, range, step): the latter
    # MISSES THE ORIGIN whenever the step does not divide the range -- 200 m
    # in 3 m steps runs .. -2, 1, 4 .. -- which biases every solution by up to
    # half a cell and silently removes the no-model candidate.
    kh = int(np.ceil(float(h_range) / float(h_step) - 1e-9))
    kv = int(np.ceil(float(v_range) / float(v_step) - 1e-9))
    gh = np.arange(-kh, kh + 1) * float(h_step) * _m2h
    gv = np.arange(-kv, kv + 1) * float(v_step) * _m2v
    assert (gh == 0).any() and (gv == 0).any(), (
        f'scan grid lost the origin: dh {gh[0]}..{gh[-1]} step {h_step}, '
        f'dv {gv[0]}..{gv[-1]} step {v_step}')

    best_g = np.zeros(m, np.float32)
    best_h = np.zeros(m)
    best_v = np.zeros(m)
    # one velocity at a time keeps the model bank to (n x n_h): the full
    # product would be (n x n_h n_v) and that is where the memory goes
    cap = _3d_budget_mb(budget) * 1024 * 1024
    hchunk = max(1, min(len(gh), int(cap // max(n * 8 * 4, 1))))
    for v0 in gv:
        Zv = Z * np.exp(-1j * (tt * v0))[:, None]
        for a in range(0, len(gh), hchunk):
            gsub = gh[a:a + hchunk]
            E = np.exp(-1j * np.outer(hh, gsub)).astype(np.complex64)
            sc = np.abs(Zv.T @ E) / nv[:, None]        # (m, n_h)
            k = np.argmax(sc, axis=1)
            g = sc[np.arange(m), k]
            up = g > best_g
            best_g = np.where(up, g, best_g)
            best_h = np.where(up, gsub[k], best_h)
            best_v = np.where(up, v0, best_v)
    # an arc with nothing measured never rose above the zero seed; zero
    # height at zero rate would read as a perfect no-motion answer
    _bad = best_g <= 0.0
    return (np.where(_bad, np.nan, best_g).astype(np.float32),
            np.where(_bad, np.nan, best_h), np.where(_bad, np.nan, best_v))





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
                  threads=1):
    """Fit every (src, tgt) arc between two phasor sets, in budgeted batches.

    NO DIFFERENTIAL ANNUAL. Both callers attach a pixel to a neighbour tens of
    metres away, and there is no seasonal GRADIENT at that scale: stratified
    and thermal delay vary with elevation and over kilometres, not across a
    courtyard. Fitting one is fitting noise, and two free parameters on a
    marginal arc buy enough coherence to carry the rate a whole sideband away.

    threads : arcs are independent, so slices of them fit concurrently; the
    GEMMs and ufuncs release the GIL. Only a caller that OWNS the host may
    raise it -- the fit3d gate does, a per-chunk dask task must not.
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
                seed_th=None if seed_th is None else seed_th[sl])
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
            seed_th=None if seed_th is None else seed_th[sl])
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

def _3d_model_removed(U, ele2phase, t, h, v):
    """Divide each column's OWN fitted model out of its phasors.

    An arc's model phase is `dh * e2p_t + dv * t_t` with dh = h_i - h_p, so the
    exponential splits and each end can be corrected on its own. With both ends
    corrected the arc coherence collapses to a plain inner product,

        gamma_ip = |sum_t u~_i,t conj(u~_p,t)| / n

    and the two-parameter search that used to find it has nothing left to find.

    HEIGHT AND RATE ONLY -- the annual stays. It is long-wavelength, so it is
    shared across an arc and cancels there; dividing it out of one end alone
    would leave the other end's annual exposed as a residual the model has no
    term for, and the arc pays its full amplitude in lost coherence.
    """
    return (np.asarray(U) * np.exp(-1j * (np.outer(ele2phase, h)
                                          + np.outer(t, v)))).astype(np.complex64)


def _3d_predict_gamma(Us_c, Ut_c, src, tgt, budget, threads=1):
    """Arc coherence at the PREDICTED model, for corrected phasors.

    Batched, because the gather is what costs memory here: the product itself
    is one column per arc, but `Us_c[:, src]` materialises (dates x arcs).
    """
    n = Us_c.shape[0]
    out = np.empty(len(src), dtype=np.float32)
    _th = max(1, int(threads))
    if _th > 1 and len(src) > _th:
        # batches are independent; einsum releases the GIL
        from concurrent.futures import ThreadPoolExecutor
        bnd = np.linspace(0, len(src), _th * 4 + 1).astype(np.int64)

        def _slice(i):
            sl = slice(bnd[i], bnd[i + 1])
            out[sl] = _3d_predict_gamma(Us_c, Ut_c, src[sl], tgt[sl],
                                        _3d_budget_mb(budget) / _th)
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_slice, range(_th * 4)))
        return out
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 32, 1)))
    for b0 in range(0, len(src), step):
        sl = slice(b0, min(b0 + step, len(src)))
        out[sl] = (np.abs(np.einsum('tj,tj->j', Us_c[:, src[sl]],
                                    np.conj(Ut_c[:, tgt[sl]]))) / n)
    return out


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


@_numba.njit(nogil=True, cache=True)
def _3d_topk_stream(score, src, nsrc, k):
    """One pass of replace-the-minimum per source -- the lexsort's answer
    at O(arcs x k) instead of a sort of the whole graph. Strict `>` keeps
    the EARLIEST arc on ties, exactly as the stable sort did."""
    vals = np.full((nsrc, k), -np.inf, np.float64)
    idxs = np.full((nsrc, k), -1, np.int64)
    for a in range(len(score)):
        v = score[a]
        if not np.isfinite(v):
            continue
        s = src[a]
        mi = 0
        mv = vals[s, 0]
        for j in range(1, k):
            if vals[s, j] < mv or (vals[s, j] == mv
                                   and idxs[s, j] > idxs[s, mi]):
                mv = vals[s, j]; mi = j
        if v > mv:
            # evict the LATEST arc among the minimum slots, so ties keep
            # the earliest -- the stable sort's choice
            vals[s, mi] = v
            idxs[s, mi] = a
    return idxs


def _3d_topk_per_src(score, src, nsrc, k):
    """Indices of the best `k` arcs of each source, NaN ranked last."""
    idxs = _3d_topk_stream(np.asarray(score, np.float64),
                           np.asarray(src, np.int64), int(nsrc), int(k))
    out = idxs.ravel()
    return out[out >= 0]


def _3d_reach_tiles(pos):
    """DS candidates by tile, with the nodes that tile can reach.

    `pos` is `(src_y, src_x, tgt_y, tgt_x, ry, rx, hy, hx)` in pixels: the
    candidate and node positions, the HALF PS extent -- the same radius the
    node-to-node arcs get from their Chebyshev query, so a DS reaches exactly
    as far as a PS does -- and the DS window, which is the tile.

    A TILE OF CANDIDATES IS ANSWERED BY ONE DENSE PRODUCT. The operand is the
    nodes within the PS extent of the tile's own bounds, and since every pixel
    in the tile lies inside those bounds, its whole window is inside the
    operand: no pair in the product needs testing, and none that belongs is
    missing. The reach the tile adds is its own half-size, and a tile is
    counted in DS windows -- the scale the DS window is DEFINED by, the area
    over which the atmospheric phase does not change -- so the extent and the
    extent grown by a tile stand in the same atmosphere and the arcs mean the
    same thing.

    THE TILE IS THE DS WINDOW, and nothing else has to be decided. Larger
    tiles reach further and so rank each candidate against more nodes, which
    costs more than the products it saves; smaller ones cut the product into
    pieces too thin to be worth a call. Neither margin is close, so there is
    no size to tune and no memory to budget: a window's worth of candidates
    against the nodes it reaches is a few megabytes whatever the block is,
    and it falls out of the two windows the caller already declared.
    """
    sy, sx, ty, tx, ry, rx, wy, wx = pos
    wy, wx = max(int(wy), 1), max(int(wx), 1)
    kx = (sx // wx).astype(np.int64)
    key = (sy // wy).astype(np.int64) * (int(kx.max()) + 1) + kx
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cut = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        idx = order[a:b]
        _y, _x = sy[idx], sx[idx]
        tsel = np.nonzero((ty >= _y.min() - ry) & (ty <= _y.max() + ry)
                          & (tx >= _x.min() - rx) & (tx <= _x.max() + rx))[0]
        yield idx, tsel


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
            step_dh, step_dv, budget, iterations, threads=threads)
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


def _3d_seed_ds_ds(Us, Ut, src, tgt, nsrc, k, budget, threads=1):
    """DS to DS: the best `k` partners of every candidate, over the LISTED arcs.

    The vouching window is a small box on purpose -- a candidate is attached by
    neighbours it can be compared against, and short arcs are the better ones.
    The graph is therefore sparse, and a dense product would score pairs the
    window deliberately excluded.
    """
    g0 = _3d_predict_gamma(np.asarray(Us, np.complex64), Ut, src, tgt,
                           budget, threads=threads)
    sd = _3d_topk_per_src(g0, src, nsrc, k)
    return src[sd], tgt[sd]


def _3d_partner_shortlist(Us, Ut, src, tgt, base_lab, nsrc, ele2phase, t,
                          meter2rad, max_dh, max_dv, step_dh, step_dv, budget,
                          iterations, min_agreeing, threshold,
                          stats=None, prefix='ds_', debug=False,
                          fix_h=None, fix_v=None, seed=None, threads=1):
    """Rank every arc, name ONE component, refine the best `min_agreeing`.

    RANK FIRST, REFINE THE SHORTLIST. Only `min_agreeing` partners are used, so
    refining all of them spends the larger half of the fit on candidates that
    are discarded. Ranking needs the model only well enough to order it, and
    the refinement cannot leave its own lattice cell, so it reorders
    neighbours at most.

    ONE PASS TO RANK, AND THAT IS A CORRECTNESS FLOOR RATHER THAN A SETTING.
    The lattice scores every candidate at a QUANTISED model, so neighbours can
    share a grid point and tie exactly. One pass takes each onto its own
    optimum, which is what makes the comparison mean anything.

    Returns (ga, dha, dva, dsa, good).
    """
    if fix_h is not None:
        # ---- SEED, PREDICT, SCORE --------------------------------------
        # The partners are already solved onto one datum, so their model
        # divides out and an arc's coherence becomes a plain inner product:
        # gamma = |sum_t u~_i conj(u~_p)| / n, n multiply-adds and no lattice.
        # What the pixel still needs is its OWN (h, v), and that is the only
        # thing a search is spent on.
        Ut_c = _3d_model_removed(Ut, ele2phase, t, fix_h, fix_v)

        # (a) a cheap prior, to choose what to search. Scoring against the
        # corrected partner while assuming the pixel carries no offset is
        # already far better than raw coherence, and it only has to be good
        # enough to put `consensus` usable arcs in front.
        _nk = int(min_agreeing) if min_agreeing is not None else 1
        _ssel, _tsel = seed(Us, Ut_c, src, tgt, nsrc, _nk,
                            budget, threads=threads)

        # (b) the seeds get the FULL search, and ALL of them are used. Each
        # returns a complete answer for the pixel -- h_p + dh -- so they are
        # `consensus` measurements of one quantity, and the provisional model
        # is their robust centre. Taking the best single one would rest the
        # pixel on one arc, which is the star attachment this replaces.
        gs_, hs_, vs_, _ = _3d_arc_batch(
            Us, Ut, _ssel, _tsel, ele2phase, t, meter2rad, max_dh,
            max_dv, step_dh, step_dv, budget, iterations, threads=threads)
        _ph = np.full((nsrc, _nk), np.nan)
        _pv = np.full((nsrc, _nk), np.nan)
        _o = np.lexsort((-np.where(np.isfinite(gs_), gs_, -np.inf), _ssel))
        _cn = np.bincount(_ssel[_o], minlength=nsrc)
        _cl = np.arange(len(_o)) - np.repeat(np.r_[0, np.cumsum(_cn)[:-1]], _cn)
        _ph[_ssel[_o], _cl] = (fix_h[_tsel] + hs_)[_o]
        _pv[_ssel[_o], _cl] = (fix_v[_tsel] + vs_)[_o]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN rows
            prov_h = np.nanmedian(_ph, axis=1)
            prov_v = np.nanmedian(_pv, axis=1)
        _ok = np.isfinite(prov_h) & np.isfinite(prov_v)

        # (c) with the pixel's own model divided out too, EVERY candidate is
        # scored at n multiply-adds -- including ones the prior ranked poorly,
        # which is the point: the prior chose what to search, the model
        # chooses what to keep.
        Us_c = np.zeros_like(np.asarray(Us, np.complex64))
        Us_c[:, _ok] = _3d_model_removed(
            np.asarray(Us, np.complex64)[:, _ok], ele2phase, t,
            prov_h[_ok], prov_v[_ok])
        ga = _3d_predict_gamma(Us_c, Ut_c, src, tgt, budget, threads=threads)
        ga = np.where(_ok[src], ga, np.nan).astype(np.float32)
        del Us_c
        if min_agreeing is not None and len(np.unique(base_lab)) > 1:
            # the best partner names the component, as on the lattice path
            _l0 = base_lab[tgt].astype(np.int64)
            _t1 = np.full(nsrc, -np.inf)
            np.maximum.at(_t1, src, np.where(np.isfinite(ga), ga, -np.inf))
            _w = np.full(nsrc, -1, dtype=np.int64)
            _is1 = (ga >= _t1[src]) & np.isfinite(ga)
            _w[src[_is1]] = _l0[_is1]
            ga = np.where(_l0 != _w[src], np.nan, ga).astype(np.float32)

        # (d) what is kept becomes MEASUREMENTS -- the consensus test and the
        # network solve consume dh, dv, so they are refined for real. Seeded
        # from the prediction, so the refinement runs without a lattice.
        _short = _nk
        _keep = _3d_topk_per_src(ga, src, nsrc, _short)
        dha = np.full(len(src), np.nan)
        dva = np.full(len(src), np.nan)
        dsa = np.zeros(len(src), dtype=np.complex128)
        if len(_keep):
            _seed = np.stack([prov_h[src[_keep]] - fix_h[tgt[_keep]],
                              prov_v[src[_keep]] - fix_v[tgt[_keep]]], axis=1)
            g2, h2, v2, s2 = _3d_arc_batch(
                Us, Ut, src[_keep], tgt[_keep], ele2phase, t, meter2rad,
                max_dh, max_dv, step_dh, step_dv, budget, iterations,
                seed_th=_seed, threads=threads)
            ga[_keep], dha[_keep], dva[_keep], dsa[_keep] = g2, h2, v2, s2
        _unref = np.ones(len(src), dtype=bool)
        _unref[_keep] = False
        ga[_unref] = np.nan
        if stats is not None:
            stats[prefix + 'ranked_arcs'] = int(len(src))
            stats[prefix + 'searched_arcs'] = int(len(_ssel))
            stats[prefix + 'provisional'] = int(_ok.sum())
        good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)
        return ga, dha, dva, dsa, good

    ga, dha, dva, dsa = _3d_arc_batch(
        Us, Ut, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
        step_dh, step_dv, budget, 1, threads=threads)
    _lab = base_lab[tgt].astype(np.int64)
    if min_agreeing is not None and len(np.unique(base_lab)) > 1:
        # THE BEST PARTNER NAMES THE COMPONENT, AND THEN ONLY ITS NODES ARE
        # USED. Components carry unrelated datums, so partners drawn from two
        # of them disagree by that offset however good every arc is. The
        # single most coherent arc says which network this pixel belongs to;
        # if that component then cannot field `min_agreeing` partners, or they
        # disagree, the pixel is unmeasured -- no other component is tried.
        # Arc coherence is settled before any velocity is read, so the best
        # arc cannot be picked to produce a result.
        _gf = np.where(np.isfinite(ga), ga, -np.inf)
        _top1 = np.full(nsrc, -np.inf)
        np.maximum.at(_top1, src, _gf)
        _is1 = (_gf >= _top1[src]) & (_gf > -np.inf)
        _win = np.full(nsrc, -1, dtype=np.int64)
        _win[src[_is1]] = _lab[_is1]
        if debug and stats is not None:
            # what the restriction forbids, counted before it acts: shortlists
            # that would have spanned two components. Not "saw more than one",
            # which nearly every pixel does once a second one exists in reach.
            _so = np.lexsort((-_gf, src))
            _sn = np.bincount(src[_so], minlength=nsrc)
            _sf = np.r_[0, np.cumsum(_sn)[:-1]]
            _tp = _so[(np.arange(len(_so)) - np.repeat(_sf, _sn)) < min_agreeing]
            _lo = np.full(nsrc, np.iinfo(np.int64).max, np.int64)
            _hi = np.full(nsrc, -1, dtype=np.int64)
            np.minimum.at(_lo, src[_tp], _lab[_tp])
            np.maximum.at(_hi, src[_tp], _lab[_tp])
            stats[prefix + 'shortlist_straddled'] = int(
                np.count_nonzero((_hi >= 0) & (_hi != _lo)))
            stats[prefix + 'multi_component'] = int(np.count_nonzero(
                np.bincount(src[_lab != _win[src]], minlength=nsrc) > 0))
        ga[_lab != _win[src]] = np.nan
    _short = int(min_agreeing) if min_agreeing is not None else len(src)
    _ord = np.lexsort((-np.where(np.isfinite(ga), ga, -np.inf), src))
    _cnt0 = np.bincount(src[_ord], minlength=nsrc)
    _off0 = np.r_[0, np.cumsum(_cnt0)[:-1]]
    _col0 = np.arange(len(_ord)) - np.repeat(_off0, _cnt0)
    # ADMISSIBLE ONLY. Ranking puts NaN last, but taking the first `_short`
    # POSITIONS still reaches them when a pixel has fewer admissible
    # candidates -- and refitting one hands it a fresh finite coherence, so an
    # arc ruled out before the ranking would come back holding a vote.
    _keep = _ord[(_col0 < _short) & np.isfinite(ga[_ord])]
    if len(_keep):
        g2, h2, v2, s2 = _3d_arc_batch(
            Us, Ut, src[_keep], tgt[_keep], ele2phase, t, meter2rad,
            max_dh, max_dv, step_dh, step_dv, budget, iterations,
            threads=threads)
        ga[_keep], dha[_keep], dva[_keep], dsa[_keep] = g2, h2, v2, s2
    # anything not refined cannot be used, whatever its lattice value
    _unref = np.ones(len(src), dtype=bool)
    _unref[_keep] = False
    ga[_unref] = np.nan
    good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)
    return ga, dha, dva, dsa, good


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
            'fit3d() needs a perpendicular baseline per date (the BPR '
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
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr
    from scipy.spatial import cKDTree
    import os as _os
    _nth = max(1, int(threads) if threads else (_os.process_cpu_count() or 1))
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
                    step_dv, budget_, max_seasonal, iterations=iterations)
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

    def _wsolve(Gm, rhs, w):
        sw = np.sqrt(np.maximum(w, 0.0))
        return lsqr(diags(sw) @ Gm, sw * rhs,
                    atol=1e-12, btol=1e-12, iter_lim=2000)[0]

    def _wsolve2(Gm, rhs_a, rhs_b, w):
        """The height and the rate solved at once: one matrix, two threads.

        Every pass of the reweighting and every pass of the gate solves the
        network TWICE, once for each quantity, and the two share the matrix and
        depend on nothing of each other's. `lsqr` spends its time in kernels
        that release the GIL, so the pair costs little more than one of them --
        and these solves are the great majority of the stage.
        """
        import threading
        out = [None, None]

        def _run(i, r):
            out[i] = _wsolve(Gm, r, w)

        th = (threading.Thread(target=_run, args=(0, rhs_a)),
              threading.Thread(target=_run, args=(1, rhs_b)))
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out[0], out[1]

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
        _xh, _xv = _wsolve2(G, dh, dv, w_)
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
        _xh_, _xv_ = _wsolve2(G_, dh[idx], dv[idx], w_[idx])
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
    m_ = len(ai)
    G = _incidence(ai, aj, m_)
    w_fin = w_[keep_arc]
    hgt, vel = _wsolve2(G, dh, dv, w_fin)
    anr = _wsolve(G, ds_.real, w_fin)
    ani = _wsolve(G, ds_.imag, w_fin)
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
    # NO PHASE IS RETURNED. fit3d() emits the model only; phase is
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


def _3d_ds_attach(S, cand_ds, ds_nodes, _oy, _ox, lab_out, vel_out,
                  hgt_out, sea_out, coh_out, lvl_out, level_id,
                  ele2phase, t, meter2rad, *,
                  ny, nx, wy, wx, cell, budget, threshold, level,
                  max_dh, max_dv, step_dh, step_dv, iterations,
                  _ma, _ii, _err_h, _err_v, _nth, _st, spacing=(1.0, 1.0),
                  debug=False):
    """LEVELS 2+: DS hung off the PS network, with the DS of earlier levels
    admitted to the VOTE and to nothing else.

    Level 1 attaches DS that muster `consensus` agreeing PS arcs. What it left
    behind mostly holds one to four coherent PS arcs -- too few to vote alone.
    Here the fixed layer (PS and every DS attached so far) arrives as an
    argument and is never recomputed; the earlier DS complete the quorum, and
    the value is solved from the pixel's PS arcs only. Only the PS hold the
    datum, so nothing stands on a DS value and no error climbs the ladder: a
    higher `level` finds a few more good pixels, never more coverage.

    The output planes are updated IN PLACE; nothing is returned.
    """
    _fy = np.asarray(ds_nodes['iy']).copy()
    _fx = np.asarray(ds_nodes['ix']).copy()
    _fv = ds_nodes['vel'].astype(float)
    _fh = ds_nodes['hgt'].astype(float)
    _fs = np.asarray(ds_nodes['sea']).copy()
    _fl = np.asarray(ds_nodes['label']).copy()
    # THE LEVEL OF EVERY FIXED NODE: 0 the PS network, n a DS attached at
    # level n. Only the PS hold the datum -- they are the low-noise pixels by
    # construction -- so the VALUE is solved from PS arcs alone; a DS of any
    # level may vote, it never anchors, and error cannot climb the ladder.
    # Tables without the field are DS.
    _flv = (np.asarray(ds_nodes['level']).astype(np.int16).copy()
            if ds_nodes.get('level') is not None
            else np.ones(len(_fy), dtype=np.int16))
    _done = np.zeros((ny, nx), dtype=bool)
    _done[_oy, _ox] = True
    _done[_fy, _fx] = True
    # THE CALLER'S BOUNDS IN THE UNITS THE DEBUG LINES PRINT, so a tally over
    # a threshold is a tally against what the caller actually asked for
    _evmm = float(_err_v) / float(meter2rad) * 1e3     # mm/yr
    _ehm = float(_err_h) / float(meter2rad)            # m
    sy, sx = float(spacing[0]), float(spacing[1])      # ground metres/pixel
    _st['vouch_rounds'] = []
    for _rnd in range(int(level) - 1):
        vy, vx = np.where(cand_ds & ~_done)
        _st['vouch_candidates'] = int(len(vy))
        if not len(vy):
            break
        _before = int(_st.get('vouch_attached', 0))
        _st['vouch_attached'] = 0
        if True:
            _av = np.abs(S[:, vy, vx])
            Uv = np.ascontiguousarray(
                np.where(_av > 0, S[:, vy, vx] / np.where(_av > 0, _av, 1),
                         0).astype(np.complex64))
            del _av
            # REACH: THE DS WINDOW, as everywhere else in the library. A
            # `window` of (wy, wx) is a BOX of that size, so its reach is
            # +-wy//2 -- that is what `_3d_arc_offsets` ranges over for the
            # DS test and what the PS ring abuts. This stage used to read it
            # as a RADIUS and search +-wy, twice as far as levels 0 and 1 for
            # the same stated window, past the separation where a differential
            # model still describes one piece of ground -- and it doubled the
            # halo every block had to read. The near pass runs first and the
            # full window is tried only where it did not serve.
            #
            # THE SHORTLIST IS BOUNDED, `consensus` partners per candidate.
            # Pairing every fixed node with every candidate in reach is the
            # same selection arrived at by materialising it first: the count
            # is n_fixed x the density of its box, it GROWS with the network
            # every level, and the arcs beyond the best few are fitted only
            # to be discarded by the consensus. Selecting first costs `kk`
            # slots per candidate instead.
            # the DS window's half-extent: the radius inside which the
            # atmosphere is stated to be common, and the one box every
            # candidate is searched over
            _hy2, _hx2 = max(wy // 2, 1), max(wx // 2, 1)
            _ab = np.abs(S[:, _fy, _fx])
            Ub = np.ascontiguousarray(
                np.where(_ab > 0, S[:, _fy, _fx] / np.where(_ab > 0, _ab, 1),
                         0).astype(np.complex64))
            del _ab
            # THE SAME CAP THE NETWORK USES, and for the same reason. The
            # shortlist is chosen on raw coherence while the consensus votes
            # on FITTED values, so every partner the gates reject is one the
            # vote never sees and `consensus` partners offered is `consensus`
            # only if none is rejected. What the cap must prevent is the
            # opposite case -- a candidate with thousands of nodes in reach --
            # and the network already answers how many arcs are worth keeping
            # per node: beyond `_ARC_CAP` they were measured to add nothing.
            # Level 1's thirty-six is NOT the precedent here; that bound is
            # structural, four PS per window times the nine-window
            # neighbourhood, and the level-2 fixed layer is dense DS with no
            # such lattice.
            _kk = int(max(_ARC_CAP, _ma))
            _vy64, _vx64 = vy.astype(np.int64), vx.astype(np.int64)

            def _search(_mask, _sub=None):
                # the partner search over the fixed nodes in `_mask`, for
                # every candidate or for the subset `_sub`
                _fg = np.full((ny, nx), -1, dtype=np.int64)
                _jj = np.flatnonzero(_mask)
                _fg[_fy[_jj], _fx[_jj]] = _jj
                if _sub is None:
                    _U, _cy, _cx = Uv, _vy64, _vx64
                else:
                    _U = np.ascontiguousarray(Uv[:, _sub])
                    _cy, _cx = _vy64[_sub], _vx64[_sub]
                _nc = len(_cy)
                _ovx = np.empty((_nc, _kk), dtype=np.float32)
                _ojx = np.empty((_nc, _kk), dtype=np.int64)
                _args = (_U, Ub, _cy, _cx, _fg,
                         np.ascontiguousarray(_fy, dtype=np.int64),
                         np.ascontiguousarray(_fx, dtype=np.int64),
                         _hy2, _hx2,
                         int(cell[0]), int(cell[1]), int(_kk),
                         float(threshold), _ovx, _ojx)
                if _nth > 1 and _nc > _nth:
                    from concurrent.futures import ThreadPoolExecutor
                    _step = -(-_nc // _nth)
                    _bnd = [(a, min(a + _step, _nc))
                            for a in range(0, _nc, _step)]
                    with ThreadPoolExecutor(_nth) as _ex:
                        list(_ex.map(lambda b: _3d_ds_partners(
                            *_args, b[0], b[1]), _bnd))
                else:
                    _3d_ds_partners(*_args, 0, _nc)
                return _ovx, _ojx

            # ONE SEARCH FOR THE VOTE, over PS and DS alike: the best `kk`
            # partners by raw coherence. Whether a PS is among them does not
            # matter here -- the vote only asks whether the pixel's best
            # partners agree. The PS that carry its VALUE are collected
            # afterwards, for the few candidates that pass.
            _ov, _oj = _search(np.ones(len(_fy), dtype=bool))
            if debug:
                # RECORDED, NOT PRINTED. One chunk's numbers describe one
                # chunk; the caller has every chunk of the level and reduces
                # them to the one line a reader can actually use.
                _np_ = (_oj >= 0).sum(1)
                _st['lvl_kk'] = int(_kk)
                _st['lvl_partners'] = _lvl_stat(_np_)
            _kp = _oj >= 0
            ds_s = np.repeat(np.arange(len(vy)), _kk).reshape(-1, _kk)[_kp]
            ds_t = _oj[_kp]
            del _ov, _oj, _kp
            _st['vouch_arcs'] = int(len(ds_s))
            if debug:
                # THE SIZE OF THE PROBLEM, recorded where it is decided rather
                # than after the fit has already allocated over it.
                _st['lvl_cands'] = int(len(vy))
                _st['lvl_fixed'] = int(len(_fy))
                _st['lvl_arcs'] = int(len(ds_s))
            if len(ds_s):
                gd, hd, vd, sd, goodd = _3d_partner_shortlist(
                    Uv, Ub, ds_s, ds_t, _fl, len(vy), ele2phase, t,
                    meter2rad, max_dh, max_dv, step_dh, step_dv, budget,
                    iterations, _ma, threshold,
                    stats=_st, prefix='vouch_', debug=debug,
                    fix_h=_fh, fix_v=_fv, seed=_3d_seed_ds_ds,
                    threads=_nth)
                # THE REPRESENTATIVE IS THE HIGHEST-GAMMA SURVIVOR.
                # Two error terms are in play -- the new arc's, and the
                # base's own inherited one -- and only the first is
                # selectable here. Choosing instead on the base's
                # attachment coherence was measured and is WORSE: base
                # gamma says how coherently that DS attached, not how
                # correct its value is, so optimising for it selects
                # confidence rather than accuracy. The inherited error is
                # bounded by the gate, which has already discarded every
                # partner whose value disagreed.
                _vst = {}
                # RATE AND HEIGHT TOGETHER, as level 1 does and as the
                # solve does. `err_h` was being passed without the heights it
                # judges, so the gate could never run: level 2 was accepting
                # partners on the rate alone, and an arc metres out in height
                # -- a facade, a roof edge, two scatterers at different
                # elevations in one window -- entered the answer unchallenged.
                v2first, v2votes, _okv = _3d_partner_consensus(
                    ds_s, gd, goodd, _fv[ds_t] + vd, len(vy),
                    _ma, _err_v, _ii, h_abs=_fh[ds_t] + hd, err_h=_err_h,
                    labels=(_fl[ds_t] if debug else None),
                    stats=_vst, prefix='vouch_')
                _st.update(_vst)
                _st['vouch_attached'] = int(len(v2first))
                # THE VALUE STANDS ON PS ARCS ONLY. Every coherent arc voted;
                # the solve is handed the arcs to PS partners and nothing
                # else. A pixel with no coherent PS arc is not attached: the
                # ladder exists to find a few more good pixels, never to
                # manufacture coverage out of DS values -- those carry noise,
                # not a datum, and a value built on them inherits it unchecked.
                if len(v2first):
                    # THE VALUE STANDS ON PS ARCS ONLY, AND EVERY PS ARC IN
                    # REACH. A second search, over the PS alone and only for
                    # the candidates that passed the vote -- a few percent of
                    # them -- so no PS can be crowded off a shortlist by
                    # denser DS, and the cost is a few percent of the first.
                    # Those arcs are fitted and REPLACE the vote's arcs for
                    # everything downstream: the named partner, the seasonal,
                    # the equations. A pixel with no coherent PS arc is not
                    # attached; a DS value carries noise, not a datum.
                    _votes_c = np.zeros(len(vy), dtype=np.int32)
                    _votes_c[ds_s[v2first]] = np.asarray(v2votes, np.int32)
                    # the vote's arcs stay addressable for the closure
                    # diagnostic below, which reads `vouch_vote_arcs`
                    _vs_s, _vs_t, _vs_vd, _vs_hd = ds_s, ds_t, vd, hd
                    _si0 = ds_s[v2first]
                    _ov2, _oj2 = _search(_flv == 0, _sub=_si0)
                    _kp2 = _oj2 >= 0
                    _pss = np.repeat(_si0, _kk).reshape(-1, _kk)[_kp2]
                    _pst = _oj2[_kp2]
                    del _ov2, _oj2, _kp2
                    v2first = np.zeros(0, dtype=np.int64)
                    if len(_pss):
                        _vps = {}
                        gd, hd, vd, sd, goodd = _3d_partner_shortlist(
                            Uv, Ub, _pss, _pst, _fl, len(vy), ele2phase, t,
                            meter2rad, max_dh, max_dv, step_dh, step_dv,
                            budget, iterations, _ma, threshold,
                            stats=_vps, prefix='vouchps_', debug=debug,
                            fix_h=_fh, fix_v=_fv, seed=_3d_seed_ds_ds,
                            threads=_nth)
                        ds_s, ds_t = _pss, _pst
                        _st['vouchps_arcs'] = int(len(ds_s))
                        # one arc per attached candidate: its best PS arc
                        _ordv = np.lexsort((-np.where(goodd, gd, -np.inf), ds_s))
                        _ordv = _ordv[goodd[_ordv]]
                        if len(_ordv):
                            _fst = np.r_[True,
                                         ds_s[_ordv][1:] != ds_s[_ordv][:-1]]
                            v2first = _ordv[_fst]
                    v2votes = _votes_c[ds_s[v2first]] if len(v2first) \
                        else np.zeros(0, dtype=np.int32)
                    _st['vouch_nops'] = int(len(_si0) - len(v2first))
                _st['vouch_attached'] = int(len(v2first))   # after the filter
                if len(v2first):
                    _si, _bj = ds_s[v2first], ds_t[v2first]
                    vy2, vx2 = vy[_si], vx[_si]

                    # ---- SOLVE THE LEVEL-2 DS, EACH FROM ITS PS ARCS --
                    # Every extension solves its new layer against the
                    # layers already solved, held FIXED. The DS partners
                    # took part in the vote and in nothing else: the value
                    # rests on the pixel's arcs to PS alone, and no DS is
                    # ever tied to another DS.
                    _a2 = np.full(len(vy), -1, dtype=np.int64)
                    _a2[_si] = np.arange(len(_si))
                    _e2 = np.flatnonzero(goodd & (_a2[ds_s] >= 0))
                    _e2i, _e2p = _a2[ds_s[_e2]], ds_t[_e2]
                    _v3, _h3, _anc3 = _3d_ds_solve(
                        len(_si), _e2i, _e2p, vd[_e2], hd[_e2], gd[_e2],
                        _fv, _fh, _err_v, _err_h, _ii)
                    if debug:
                        # THE FUNNEL: what the consensus vetted, what the
                        # solve was handed, and what it kept. The gap between
                        # the first two is the arcs no gate in physical units
                        # ever saw.
                        _vv = _st.get('vouch_vote_arcs')
                        _st['lvl_vetted'] = int(len(_vv)) if _vv is not None \
                            else 0
                        _st['lvl_solve_in'] = int(len(_e2))
                        _st['lvl_solve_kept'] = int(_anc3.sum())
                        # how far the UNVETTED arcs sit from the pixel's own
                        # consensus centre, in the units the bound is stated in
                        _cvv = _st.get('vouch_centre_v')
                        if _cvv is not None and len(_e2):
                            _cc = np.asarray(_cvv)[ds_s[_e2]]
                            _pp = _fv[ds_t[_e2]] + vd[_e2]
                            _dd = np.abs(_pp - _cc) / meter2rad * 1e3
                            _dd = _dd[np.isfinite(_dd)]
                            if len(_dd):
                                _st['lvl_offcentre'] = _lvl_stat(
                                    _dd, (('gt', _evmm),))
                    # CONSENSUS TIES TO THE NETWORK, NOT TO PEERS. A
                    # `DS - DS` equation relates a pixel to another pixel of
                    # its OWN level and carries no datum; counted toward the
                    # same floor it lets a cluster certify itself on its
                    # internal edges. The consensus demanded five partners in
                    # the FIXED layer before the solve, so the solve must
                    # leave five standing.
                    # ... FIVE OF THE PS ARCS WHERE FIVE EXIST. The vote was
                    # taken over every partner; the solve is handed the PS
                    # arcs only, and a pixel holding three PS arcs plus two
                    # DS votes is handed three equations. The floor is
                    # therefore min(consensus, equations handed), never less
                    # than one: fewer standing than handed means the solve
                    # rejected some, which is the gate's business.
                    _nval = np.bincount(_e2i, minlength=len(_si))
                    _lv3 = (_anc3 >= np.minimum(_ma, _nval)) & (_anc3 >= 1)
                    if debug:
                        _st['lvl_passed'] = int((_anc3 >= _ma).sum())
                    _st['vouch_unconfirmed'] = int((~_lv3).sum())
                    _v3 = np.where(_lv3, _v3, np.nan)
                    _h3 = np.where(_lv3, _h3, np.nan)
                    # ---- WHAT THIS LEVEL IS WORTH -----------------------
                    # Yield alone cannot say whether a level added good
                    # pixels or merely more of them. This is the level-1 DS
                    # closure applied to this layer: each voting partner
                    # predicts the candidate from its own FIXED value plus
                    # the arc, and the disagreement with what the solve
                    # returned is the error, in mm/yr and m. Like level 1's,
                    # this set is not selected for agreeing, so it needs no
                    # second view.
                    _va3 = _vst.get('vouch_vote_arcs')
                    if debug and _va3 is not None and len(_va3):
                        _vk = np.asarray(_va3, np.int64)
                        _sl = _a2[_vs_s[_vk]]
                        _mv = _sl >= 0
                        _sl, _vk = _sl[_mv], _vk[_mv]
                        if len(_sl):
                            _fw = np.isfinite(_v3[_sl]) & np.isfinite(_h3[_sl])
                            _sl, _vk = _sl[_fw], _vk[_fw]
                    else:
                        _sl = np.zeros(0, np.int64)
                    if debug and len(_sl):
                        _cv3 = np.abs((_fv[_vs_t[_vk]] + _vs_vd[_vk])
                                      - _v3[_sl]) / meter2rad * 1e3
                        _ch3 = np.abs((_fh[_vs_t[_vk]] + _vs_hd[_vk])
                                      - _h3[_sl]) / meter2rad
                        # PER PIXEL, as the network reports per node: a DS
                        # averages its partners, so its worst is milder than
                        # the worst single partner and the two scales must
                        # be labelled apart.
                        _qa = np.zeros(len(_si)); _qb = np.zeros(len(_si))
                        _qc = np.zeros(len(_si))
                        np.add.at(_qa, _sl, _cv3)
                        np.add.at(_qb, _sl, _ch3)
                        np.add.at(_qc, _sl, 1.0)
                        _qh = _qc > 0
                        _pv3 = _qa[_qh] / _qc[_qh]
                        _ph3 = _qb[_qh] / _qc[_qh]
                        # THE SAMPLES, NOT A SUMMARY OF THEM. Percentiles do
                        # not average across chunks -- a median of medians is
                        # not the median -- so the level's reducer needs the
                        # values themselves to answer for the whole level.
                        # ARE THE PARTNERS INDEPENDENT OF EACH OTHER? The
                        # independence cell is enforced between the CANDIDATE
                        # and each partner, never between the partners --
                        # exactly as at level 1, and the diagnostic was only
                        # ever computed there. Five partners inside one cell
                        # are one sample of the ground counted five times, and
                        # they agree because they are the same measurement.
                        _cy0, _cx0 = int(cell[0]), int(cell[1])
                        _pcell = ((_fy[_vs_t[_vk]] // max(_cy0, 1)
                                   ).astype(np.int64) * (1 << 20)
                                  + (_fx[_vs_t[_vk]] // max(_cx0, 1)))
                        _psrc = np.asarray(_vs_s[_vk], np.int64)
                        _o0 = np.lexsort((_pcell, _psrc))
                        _ss0, _pp0 = _psrc[_o0], _pcell[_o0]
                        _nw0 = np.r_[True, (_ss0[1:] != _ss0[:-1])
                                     | (_pp0[1:] != _pp0[:-1])]
                        _ncell = np.bincount(_ss0[_nw0], minlength=len(vy))
                        _nvote = np.bincount(_psrc, minlength=len(vy))
                        _hv0 = _nvote > 0
                        if _hv0.any():
                            _st['lvl_pcells'] = _lvl_stat(_ncell[_hv0], _LE12)
                            _st['lvl_pvotes'] = _lvl_stat(_nvote[_hv0])
                            # HOW FAR the agreeing partners are, in ground
                            # units, as level 1 reports it
                            _dyp = (_fy[_vs_t[_vk]] - vy[_psrc]) * float(sy)
                            _dxp = (_fx[_vs_t[_vk]] - vx[_psrc]) * float(sx)
                            _rr0 = np.sqrt(_dyp ** 2 + _dxp ** 2)
                            _sd0 = np.zeros(len(vy)); _sn0 = np.zeros(len(vy))
                            np.add.at(_sd0, _psrc, _rr0)
                            np.add.at(_sn0, _psrc, 1.0)
                            _st['lvl_parcm'] = _lvl_stat(
                                _sd0[_hv0] / np.maximum(_sn0[_hv0], 1))
                        _st['lvl_clo_gamma'] = float(np.median(gd[v2first]))
                        _st['lvl_clo_arc_v'] = _lvl_stat(_cv3)
                        _st['lvl_clo_arc_h'] = _lvl_stat(_ch3)
                        _st['lvl_clo_ds_v'] = _lvl_stat(
                            _pv3, (('gt', _evmm),))
                        _st['lvl_clo_ds_h'] = _lvl_stat(_ph3)
                    # gated on the solve, and written AFTER it --
                    # the label used to be set before the solve ran
                    lab_out[vy2, vx2] = np.where(_lv3, _fl[_bj], -1)
                    vel_out[vy2, vx2] = _v3.astype(np.float32)
                    # WHICH LEVEL FOUND THIS PIXEL, so a caller can see the
                    # ladder rather than only its total.
                    lvl_out[vy2, vx2] = np.where(
                        _lv3, np.int8(level_id + _rnd), np.int8(-1))
                    hgt_out[vy2, vx2] = _h3.astype(np.float32)
                    coh_out[vy2, vx2] = np.where(
                        _lv3, gd[v2first], np.nan).astype(np.float32)
                    sea_out[vy2, vx2] = np.where(
                        _lv3,
                        (_fs[_bj].real + sd[v2first].real)
                        + 1j * (_fs[_bj].imag + sd[v2first].imag),
                        np.nan + 1j * np.nan).astype(np.complex64)
                    _st.update(
                        vouch_iy=vy2.copy(), vouch_ix=vx2.copy(),
                        vouch_label=_fl[_bj],
                        vouch_gamma=gd[v2first].copy(),
                        vouch_votes=np.asarray(v2votes, dtype=np.int32),
                        vouch_velocity_rad_yr=_v3.astype(np.float32),
                        vouch_height_rad=_h3.astype(np.float32))
        _got = int(_st.get('vouch_attached', 0))
        _st['vouch_rounds'].append(_got)
        if debug:
            _st['lvl_left'] = int(len(vy))
            _st['lvl_attached'] = int(_got)
            _st['lvl_network'] = int(len(_fy) + _got)
        if not _got:
            break                  # nothing added: further rounds cannot
        _ny_ = _st['vouch_iy']; _nx_ = _st['vouch_ix']
        _keepn = np.isfinite(_st['vouch_velocity_rad_yr'])
        _fy = np.r_[_fy, _ny_[_keepn]]
        _fx = np.r_[_fx, _nx_[_keepn]]
        _fv = np.r_[_fv, _st['vouch_velocity_rad_yr'][_keepn].astype(float)]
        _fh = np.r_[_fh, _st['vouch_height_rad'][_keepn].astype(float)]
        _fs = np.r_[_fs, sea_out[_ny_[_keepn], _nx_[_keepn]]]
        _fl = np.r_[_fl, _st['vouch_label'][_keepn]]
        _flv = np.r_[_flv, np.full(int(_keepn.sum()), level_id + _rnd,
                                   dtype=np.int16)]
        # only what the solve KEPT is settled; a winner the anchor gate
        # refused stays a candidate -- the next round's larger fixed layer
        # may hold the anchors this round could not offer it
        _done[_ny_[_keepn], _nx_[_keepn]] = True
        if not _keepn.any():
            break                  # nothing kept: the fixed layer did not
                                   # grow, so retrying is the same round again


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
    import os as _os
    _nth = max(1, int(threads) if threads else (_os.process_cpu_count() or 1))
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


@_numba.njit(nogil=True, cache=True)
def _cascade_tile(t2, mask, thr2, cnt, mx, y, x0, hx, gx0, ny_cnt, nx_cnt):
    """One tile's fused count+max with both ends credited, GIL-free.

    `t2` holds squared coherent-sum magnitudes for the tile's own pixels
    against their one-sided neighbourhood; every admissible score is counted
    against the threshold and folded into the running maximum AT BOTH ENDS.
    Credits land wherever they fall -- the caller clips to the owned region
    by slicing the accumulators it hands in.
    """
    w, ndy, span = t2.shape
    for i in range(w):
        xi = x0 + i
        for dy in range(ndy):
            base = dy * span
            for c in range(span):
                if mask[i, base + c] == 0.0:
                    continue
                v = t2[i, dy, c]
                col = gx0 + c
                if v >= thr2:
                    cnt[y, xi] += 1
                    if 0 <= col < nx_cnt and y + dy < ny_cnt:
                        cnt[y + dy, col] += 1
                if v > mx[y, xi]:
                    mx[y, xi] = v
                if 0 <= col < nx_cnt and y + dy < ny_cnt:
                    if v > mx[y + dy, col]:
                        mx[y + dy, col] = v


def _cascade_count_max(S, wy, wx, cell, thr, floor2=None):
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
    """
    n, ny, nx = S.shape
    cy, cx = int(cell[0]), int(cell[1])
    f2 = None if floor2 is None else (int(floor2[0]), int(floor2[1]))
    hy, hx = wy // 2, wx // 2
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
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
        del blk, a, f, o, u
    cnt = np.zeros((ny, nx), np.int32)
    mx = np.zeros((ny, nx), np.float32)
    if f2 is not None:
        cnt2 = np.zeros((ny, nx), np.int32)
        mx2 = np.zeros((ny, nx), np.float32)
    thr2 = (float(thr) * n) ** 2
    Bx = max(8, hx)
    masks = {}
    masks2 = {}
    for y in range(ny):
        ndy = min(hy + 1, ny - y)
        for x0 in range(0, nx, Bx):
            w = min(Bx, nx - x0)
            span = w + 2 * hx
            A1 = Xp[y, hx + x0:hx + x0 + w, :]
            A2 = np.empty((w, K), dtype=np.float32)
            A2[:, :n] = A1[:, n:]
            A2[:, n:] = -A1[:, :n]
            Bk = np.ascontiguousarray(
                Xp[y:y + ndy, x0:x0 + span, :].transpose(2, 0, 1)
            ).reshape(K, ndy * span)
            t = A1 @ Bk
            Ci = A2 @ Bk
            np.multiply(t, t, out=t)
            np.multiply(Ci, Ci, out=Ci)
            t += Ci
            key = (w, ndy)
            if key not in masks:
                dxm = (np.arange(span)[None, None, :] - hx
                       - np.arange(w)[:, None, None])
                dyv = np.arange(ndy)[None, :, None]
                mm = ((np.abs(dxm) <= hx)
                      & ~((dyv < cy) & (np.abs(dxm) < cx))
                      & ~((dyv == 0) & (dxm <= 0)))
                masks[key] = mm.reshape(w, ndy * span).astype(np.float32)
                if f2 is not None:
                    masks2[key] = (mm & ~((dyv < f2[0]) & (np.abs(dxm) < f2[1]))
                                   ).reshape(w, ndy * span).astype(np.float32)
            m = masks[key]
            t *= m
            t3 = t.reshape(w, ndy, span)
            _cascade_tile(t3, m, thr2, cnt, mx, y, x0, hx, x0 - hx, ny, nx)
            if f2 is not None:
                # the longer floor is a subset of the admissible pairs, so the
                # same masked products serve; the kernel skips what m2 zeroes
                _cascade_tile(t3, masks2[key], thr2, cnt2, mx2,
                              y, x0, hx, x0 - hx, ny, nx)
    best = np.sqrt(mx, out=mx) / n
    if f2 is None:
        return cnt, best
    return cnt, best, cnt2, np.sqrt(mx2, out=mx2) / n


def _3d_check_window_cell(wy, wx, cell, name='fit3d'):
    """The DS window must span at least FOUR independence cells per dimension.

    A window is a patch of independent ground samples in which a distributed
    scatterer is looked for, and 4 x 4 cells -- sixteen samples -- is already
    a small one: fewer is not a search for distributed scattering at all, and
    the consensus a DS needs cannot be drawn from a handful of samples. Raised
    here rather than answered with an empty product.
    """
    cy, cx = int(cell[0]), int(cell[1])
    if int(wy) < 4 * cy or int(wx) < 4 * cx:
        raise ValueError(
            f'{name}(): the DS window ({int(wy)}, {int(wx)}) must span at '
            f'least four independence cells per dimension -- cell={(cy, cx)} '
            f'needs a window of ({4 * cy}, {4 * cx}) or larger. A window of '
            f'fewer than 4 x 4 cells is too small a patch of independent '
            f'samples to search for a distributed scatterer.')


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
    _th = max(1, int(threads))
    if _th > 1 and ny >= 2 * (hy + 1):
        from concurrent.futures import ThreadPoolExecutor
        H = max(hy + 1, -(-ny // _th))
        bands = [(a, min(a + H, ny)) for a in range(0, ny, H)]
        cnt = np.empty((ny, nx), np.int32)
        best = np.empty((ny, nx), np.float32)
        cnt2 = np.empty((ny, nx), np.int32)
        best2 = np.empty((ny, nx), np.float32)

        def _band(b):
            ya, yb = b
            a0 = max(0, ya - hy)
            b0 = min(ny, yb + hy)
            c_, m_, c2_, m2_ = _cascade_count_max(S[:, a0:b0], wy, wx, cell,
                                                  thr, floor2=_floor2)
            cnt[ya:yb] = c_[ya - a0:yb - a0]
            best[ya:yb] = m_[ya - a0:yb - a0]
            cnt2[ya:yb] = c2_[ya - a0:yb - a0]
            best2[ya:yb] = m2_[ya - a0:yb - a0]
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
    else:
        cnt, best, cnt2, best2 = _cascade_count_max(S, wy, wx, cell, thr,
                                                    floor2=_floor2)
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
    # as far as the window it documents, and twice as far as fit3d for the
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
            threads=threads)
        good = np.isfinite(ga) & (ga >= float(thr))
        np.add.at(arcs.ravel(), idx[src[good]], 1)
        order = np.lexsort((-np.where(good, ga, -np.inf), src))
        first = order[np.r_[True, np.diff(src[order]) > 0]]
        first = first[good[first]]
        gamma.ravel()[idx[src[first]]] = ga[first]
        dh.ravel()[idx[src[first]]] = dha[first]
        dv.ravel()[idx[src[first]]] = dva[first]
    return gamma, dh, dv, arcs


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _3d_topk_stream(np.array([0.5, np.nan, 0.7]),
                    np.array([0, 0, 0], np.int64), 1, 2)
    _cascade_tile(np.ones((1, 1, 3), np.float32), np.ones((1, 3), np.float32),
                  0.5, np.zeros((1, 1), np.int32), np.zeros((1, 1), np.float32),
                  0, 0, 1, -1, 1, 1)


_warmup_numba_cache()
