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
    _wf = np.ones((3, 1), dtype=np.float32)
    _threshold_pairs_numba_kernel(_c, _wf, 1, 3, np.pi * 0.5)
    # the gridded transform's spreaders, one call per rank: a worker that has
    # to compile them itself does it while every other worker compiles the
    # same thing into the same cache
    for _k in (1, 2, 3):
        trend2d_spread(np.ones((1, 1), np.complex128),
                       np.zeros((1, _k)), 8)
    # the accumulator's own kernels and the reader, on the shapes of covariates
    # a caller can name: a raster, vectors along y and x with a raster, none
    _z = np.ones((2, 3, 4), np.complex64)
    _r = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    _y = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    _x = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    for _tb, _dm in (((_r,), ['yx']), ((_y, _x, _r), ['y', 'x', 'yx']), ((_y,), ['y'])):
        _k = len(_tb)
        _st = np.array([0.5] * _k + [1.0] * _k)
        _a = trend2d_accumulate(_z, _tb, _st, 8, dims=_dm, extent=(0.0, 1.0, 0.0, 1.0),
                                coords=(_y, _x))
        trend2d_fit(_a[:, 0, 0, :], 8, _k,
                    axes=tuple(i for i, d in enumerate(_dm) if d != 'yx')[:2])


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
# `resolution` is the HALF-POWER width of the sampling's own transform: |W|/n is
# an amplitude, so half of the power is this level of it
TREND2D_RESOLUTION_LEVEL = float(np.sqrt(0.5))
# the searched start reads each covariate's whole reach on a lattice this many
# times finer than a turn; False leaves the starts at zero and the axis profiles
TREND2D_SEARCH_OVER = 8
TREND2D_SEARCHED_START = True


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


# ---------------------------------------------------------------------------
# The accumulator's own kernels. The samples are held as (pixels, dates) and
# every grid as (cells, dates), so the DATE is the innermost loop and walks
# memory in order: a pixel's kernel weights are computed once and serve every
# date of the call. Both checkerboard halves come out of the one pass -- a
# pixel goes into the grid of ITS half and the total is their sum.
# ---------------------------------------------------------------------------

@nb.njit(cache=True, nogil=True)
def _trend2d_anyvalid(blk, flags):
    """blk (dates, pixels) complex -> flags[p] = 1 where any date holds a
    finite non-zero sample. Stops reading once every pixel is flagged."""
    nq, npix = blk.shape
    left = npix
    for t in range(nq):
        row = blk[t]
        for p in range(npix):
            if flags[p] == 0:
                z = row[p]
                zr = np.float64(z.real)
                zi = np.float64(z.imag)
                a = zr * zr + zi * zi
                if a > 0.0 and np.isfinite(a):
                    flags[p] = 1
                    left -= 1
        if left == 0:
            break


@nb.njit(cache=True, nogil=True)
def _trend2d_gather(blk, idx, ur, ui, cnt):
    """blk (dates, pixels) complex -> the unit phasors of the kept pixels as
    ur, ui (kept pixels, dates), zero where a date has no sample; cnt per date.

    In runs of pixels short enough that a run's (pixels, dates) rows stay in
    cache while every date writes its column of them."""
    nq = blk.shape[0]
    n = idx.size
    step = 4096
    for p0 in range(0, n, step):
        p1 = min(p0 + step, n)
        for t in range(nq):
            row = blk[t]
            c = 0
            for p in range(p0, p1):
                z = row[idx[p]]
                zr = np.float64(z.real)
                zi = np.float64(z.imag)
                a = np.sqrt(zr * zr + zi * zi)
                if a > 0.0 and np.isfinite(a):
                    ur[p, t] = zr / a
                    ui[p, t] = zi / a
                    c += 1
                else:
                    ur[p, t] = 0.0
                    ui[p, t] = 0.0
            cnt[t] += c


@nb.njit(cache=True, fastmath=True, nogil=True)
def _trend2d_spread_rows(rows, colsi, par, ur, ui, full, w, beta, cells, M,
                         ncols, WY, BY, AY, WX, BX, AX, AR, xlo, xhi,
                         oslot, ovar, ikind, islot, ivar,
                         vclass, vslot, pclass, pslot, PY, PX, P,
                         tr, ti, gr, gi, Hg, mn, m1, m2, pr, pi):
    """Every sum of one call, row by row.

    THE KERNEL IS A PRODUCT, and a covariate that is a vector along y has one
    weight vector per ROW. So a row's pixels are spread over the OTHER axes only
    -- the inner grid tr, ti -- and that small grid then goes into the rows of
    the full grid the row's own kernel touches: the same sums in another order,
    at the footprint of the inner axes per pixel instead of all of them. Inner
    axes are vectors along x (weights from a per-column table) and rasters
    (weights per pixel). With no y-vector the inner grid IS the grid.

    Alongside, from the same weights: the sampling's transform Hg per VARIABLE
    (one-dimensional -- the fit reads it one gradient at a time), the count and
    the moments per half, and the coarse profile over the axis covariates.
    `full` says every kept pixel has every date of the call, so what depends on
    the sampling alone is summed once (last axis of length one) not per date.
    """
    n = rows.size
    nq = ur.shape[1]
    nqh = mn.shape[1]
    nout = oslot.size
    nin = ikind.size
    k = vclass.size
    m = pclass.size
    itap = np.ones(3, np.int64)
    ist = np.zeros(3, np.int64)
    otap = np.ones(3, np.int64)
    ost = np.zeros(3, np.int64)
    s = 1
    for j in range(nin - 1, -1, -1):
        itap[j] = w
        ist[j] = s
        s *= M
    s = 1
    for j in range(nout - 1, -1, -1):
        otap[j] = w
        ost[j] = s
        s *= M
    win = np.zeros((3, w))
    win[:, 0] = 1.0
    wout = np.zeros((3, w))
    wout[:, 0] = 1.0
    ib = np.zeros(3, np.int64)
    ob = np.zeros(3, np.int64)
    lo = np.zeros(3, np.int64)
    hi = np.ones(3, np.int64)
    Av = np.zeros(3)
    rowcnt = np.zeros(nqh)
    colcnt = np.zeros((ncols, nqh))
    anyx = False
    for j in range(nin):
        if ikind[j] == 1:
            anyx = True
    p = 0
    while p < n:
        r = rows[p]
        q_end = p
        while q_end < n and rows[q_end] == r:
            q_end += 1
        if nout > 0:
            # the window of the inner grid this row can touch
            for j in range(nin):
                if ikind[j] == 1:
                    lo[j] = xlo[islot[j]]
                    hi[j] = xhi[islot[j]]
                else:
                    a0 = M
                    a1 = 0
                    for s_ in range(p, q_end):
                        t = (AR[islot[j], s_] + 0.5) * cells + w
                        i0 = int(np.ceil(t - 0.5 * w))
                        if i0 < a0:
                            a0 = i0
                        if i0 + w > a1:
                            a1 = i0 + w
                    lo[j] = a0
                    hi[j] = a1
            for hh in range(2):
                for i0_ in range(lo[0], hi[0]):
                    for i1_ in range(lo[1], hi[1]):
                        for i2_ in range(lo[2], hi[2]):
                            fi = i0_ * ist[0] + i1_ * ist[1] + i2_ * ist[2]
                            for q in range(nq):
                                tr[hh, fi, q] = 0.0
                                ti[hh, fi, q] = 0.0
        for qh in range(nqh):
            rowcnt[qh] = 0.0
        for s_ in range(p, q_end):
            c = colsi[s_]
            h = par[s_]
            for j in range(nin):
                sl = islot[j]
                if ikind[j] == 1:
                    ib[j] = BX[sl, c]
                    for d in range(w):
                        win[j, d] = WX[sl, c, d]
                else:
                    t = (AR[sl, s_] + 0.5) * cells + w
                    i0 = int(np.ceil(t - 0.5 * w))
                    ib[j] = i0
                    for d in range(w):
                        z = 2.0 * (i0 + d - t) / w
                        win[j, d] = (np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
                                     if -1.0 < z < 1.0 else 0.0)
            for d0 in range(itap[0]):
                w0 = win[0, d0]
                if w0 == 0.0:
                    continue
                f0 = (ib[0] + d0) * ist[0]
                for d1 in range(itap[1]):
                    w1 = w0 * win[1, d1]
                    if w1 == 0.0:
                        continue
                    f1 = f0 + (ib[1] + d1) * ist[1]
                    for d2 in range(itap[2]):
                        ww = w1 * win[2, d2]
                        if ww == 0.0:
                            continue
                        fi = f1 + (ib[2] + d2) * ist[2]
                        for q in range(nq):
                            tr[h, fi, q] += ur[s_, q] * ww
                            ti[h, fi, q] += ui[s_, q] * ww
            for a in range(k):
                if vclass[a] == 0:
                    Av[a] = AY[vslot[a], r]
                elif vclass[a] == 1:
                    Av[a] = AX[vslot[a], c]
                else:
                    Av[a] = AR[vslot[a], s_]
            if full:
                mn[h, 0] += 1.0
                pp = 0
                for a in range(k):
                    m1[h, a, 0] += Av[a]
                    for b in range(a, k):
                        m2[h, pp, 0] += Av[a] * Av[b]
                        pp += 1
                rowcnt[0] += 1.0
                if anyx:
                    colcnt[c, 0] += 1.0
                for j in range(nin):
                    if ikind[j] == 2:
                        for d in range(w):
                            Hg[ivar[j], ib[j] + d, 0] += win[j, d]
            else:
                for q in range(nq):
                    if ur[s_, q] != 0.0 or ui[s_, q] != 0.0:
                        mn[h, q] += 1.0
                        pp = 0
                        for a in range(k):
                            m1[h, a, q] += Av[a]
                            for b in range(a, k):
                                m2[h, pp, q] += Av[a] * Av[b]
                                pp += 1
                        rowcnt[q] += 1.0
                        if anyx:
                            colcnt[c, q] += 1.0
                        for j in range(nin):
                            if ikind[j] == 2:
                                for d in range(w):
                                    Hg[ivar[j], ib[j] + d, q] += win[j, d]
            if m > 0:
                flat = 0
                for j in range(m):
                    if pclass[j] == 0:
                        flat = flat * P + PY[pslot[j], r]
                    else:
                        flat = flat * P + PX[pslot[j], c]
                for q in range(nq):
                    pr[flat, q] += ur[s_, q]
                    pi[flat, q] += ui[s_, q]
        if nout > 0:
            for j in range(nout):
                sl = oslot[j]
                ob[j] = BY[sl, r]
                for d in range(w):
                    wout[j, d] = WY[sl, r, d]
                    for qh in range(nqh):
                        Hg[ovar[j], ob[j] + d, qh] += wout[j, d] * rowcnt[qh]
            for e0 in range(otap[0]):
                v0 = wout[0, e0]
                if v0 == 0.0:
                    continue
                g0 = (ob[0] + e0) * ost[0]
                for e1 in range(otap[1]):
                    v1 = v0 * wout[1, e1]
                    if v1 == 0.0:
                        continue
                    g1 = g0 + (ob[1] + e1) * ost[1]
                    for e2 in range(otap[2]):
                        vv = v1 * wout[2, e2]
                        if vv == 0.0:
                            continue
                        fo = g1 + (ob[2] + e2) * ost[2]
                        for hh in range(2):
                            for i0_ in range(lo[0], hi[0]):
                                for i1_ in range(lo[1], hi[1]):
                                    for i2_ in range(lo[2], hi[2]):
                                        fi = (i0_ * ist[0] + i1_ * ist[1]
                                              + i2_ * ist[2])
                                        for q in range(nq):
                                            gr[hh, fo, fi, q] += vv * tr[hh, fi, q]
                                            gi[hh, fo, fi, q] += vv * ti[hh, fi, q]
        p = q_end
    # a vector along x: its sampling is the count of kept pixels per column
    for j in range(nin):
        if ikind[j] == 1:
            sl = islot[j]
            for c in range(ncols):
                for d in range(w):
                    for qh in range(nqh):
                        Hg[ivar[j], BX[sl, c] + d, qh] += WX[sl, c, d] * colcnt[c, qh]


def _trend2d_axis_tables(Aax, cells, bad=None):
    """A covariate that is a vector along one axis: its kernel weights and first
    cell per row (or column), the routine the pixels go through made a table."""
    w = TREND2D_W
    beta = TREND2D_BETA * w
    Aax = np.asarray(Aax, np.float64)
    if bad is not None and bad.any():
        Aax = np.where(bad, 0.0, Aax)
    t = (Aax + 0.5) * float(cells) + w
    i0 = np.ceil(t - 0.5 * w).astype(np.int64)
    z = 2.0 * (i0[:, None] + np.arange(w)[None, :] - t[:, None]) / w
    inside = (z > -1.0) & (z < 1.0)
    kv = np.where(inside, np.exp(beta * (np.sqrt(np.where(inside, 1.0 - z * z, 1.0)) - 1.0)), 0.0)
    if bad is not None and bad.any():
        kv[bad] = 0.0
    return np.ascontiguousarray(kv), np.ascontiguousarray(i0)


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
    Sr, Si = _trend2d_parts(S)
    if kg is None:
        kg = trend2d_kernel(cells)
    return _trend2d_read_parts(Sr, Si, cells, k, g, kg)


def _trend2d_parts(S):
    """A spread grid as the two contiguous float64 vectors the reader walks:
    (real, imaginary) handed over as they are, a complex vector split."""
    import numpy as _np
    if isinstance(S, tuple):
        return S
    S = _np.asarray(S)
    return (_np.ascontiguousarray(S.real, dtype=_np.float64).ravel(),
            _np.ascontiguousarray(S.imag, dtype=_np.float64).ravel()
            if _np.iscomplexobj(S) else _np.zeros(S.size, _np.float64))


def _trend2d_read_parts(Sr, Si, cells, k, g, kg):
    """trend2d_read() on the grid's real and imaginary parts."""
    import numpy as _np
    cells = int(cells)
    g = _np.atleast_1d(_np.asarray(g, _np.float64))
    kgr = _np.ascontiguousarray(_np.asarray(kg).real, dtype=_np.float64)
    T, dT, ok = _trend2d_read_kernel(Sr, Si, int(k), cells + 2 * TREND2D_W,
                                     float(cells), TREND2D_W, g, kgr)
    if not ok:
        return _np.nan + 0j, _np.full(k, _np.nan + 0j)
    return complex(T), dT


@nb.njit(cache=True, fastmath=True, nogil=True)
def _trend2d_read_kernel(Sr, Si, k, M, cells, w, g, kg):
    """The contraction of trend2d_read(), ONE pass over the grid: the last axis
    is contracted against the phasors and against their derivative together, so
    the value and its k derivatives share the walk through the cells."""
    two_pi = 2.0 * np.pi
    pr = np.empty((k, M))
    pi = np.empty((k, M))
    dr = np.empty((k, M))
    di = np.empty((k, M))
    ka = np.empty(k, np.complex128)
    dka = np.empty(k, np.complex128)
    kh = 1.0 + 0.0j
    for a in range(k):
        sr = 0.0
        si = 0.0
        tr = 0.0
        ti = 0.0
        for c in range(M):
            am = (c - w) / cells - 0.5
            ang = -two_pi * g[a] * am
            cr = np.cos(ang)
            sn = np.sin(ang)
            f = two_pi * am
            pr[a, c] = cr
            pi[a, c] = sn
            # e * (-2 pi i am)
            dr[a, c] = f * sn
            di[a, c] = -f * cr
            sr += kg[c] * cr
            si += kg[c] * sn
            tr += kg[c] * dr[a, c]
            ti += kg[c] * di[a, c]
        ka[a] = complex(sr, si)
        dka[a] = complex(tr, ti)
        kh *= ka[a]
    dT = np.zeros(k, np.complex128)
    if abs(kh) < 1e-12:
        return 0.0 + 0.0j, dT, False
    That = 0.0 + 0.0j
    la = k - 1
    if k == 1:
        ar = 0.0
        ai = 0.0
        br = 0.0
        bi = 0.0
        for l in range(M):
            ar += Sr[l] * pr[0, l] - Si[l] * pi[0, l]
            ai += Sr[l] * pi[0, l] + Si[l] * pr[0, l]
            br += Sr[l] * dr[0, l] - Si[l] * di[0, l]
            bi += Sr[l] * di[0, l] + Si[l] * dr[0, l]
        That = complex(ar, ai)
        dT[0] = complex(br, bi)
    elif k == 2:
        t_ = 0.0 + 0.0j
        d0 = 0.0 + 0.0j
        d1 = 0.0 + 0.0j
        for i in range(M):
            base = i * M
            ar = 0.0
            ai = 0.0
            br = 0.0
            bi = 0.0
            for l in range(M):
                s_r = Sr[base + l]
                s_i = Si[base + l]
                ar += s_r * pr[la, l] - s_i * pi[la, l]
                ai += s_r * pi[la, l] + s_i * pr[la, l]
                br += s_r * dr[la, l] - s_i * di[la, l]
                bi += s_r * di[la, l] + s_i * dr[la, l]
            p0 = complex(pr[0, i], pi[0, i])
            t_ += p0 * complex(ar, ai)
            d0 += complex(dr[0, i], di[0, i]) * complex(ar, ai)
            d1 += p0 * complex(br, bi)
        That = t_
        dT[0] = d0
        dT[1] = d1
    else:
        t_ = 0.0 + 0.0j
        d0 = 0.0 + 0.0j
        d1 = 0.0 + 0.0j
        d2 = 0.0 + 0.0j
        for i in range(M):
            a1 = 0.0 + 0.0j
            b1 = 0.0 + 0.0j
            c1 = 0.0 + 0.0j
            for j in range(M):
                base = (i * M + j) * M
                ar = 0.0
                ai = 0.0
                br = 0.0
                bi = 0.0
                for l in range(M):
                    s_r = Sr[base + l]
                    s_i = Si[base + l]
                    ar += s_r * pr[la, l] - s_i * pi[la, l]
                    ai += s_r * pi[la, l] + s_i * pr[la, l]
                    br += s_r * dr[la, l] - s_i * di[la, l]
                    bi += s_r * di[la, l] + s_i * dr[la, l]
                p1 = complex(pr[1, j], pi[1, j])
                va = complex(ar, ai)
                a1 += p1 * va
                b1 += complex(dr[1, j], di[1, j]) * va
                c1 += p1 * complex(br, bi)
            p0 = complex(pr[0, i], pi[0, i])
            t_ += p0 * a1
            d0 += complex(dr[0, i], di[0, i]) * a1
            d1 += p0 * b1
            d2 += p0 * c1
        That = t_
        dT[0] = d0
        dT[1] = d1
        dT[2] = d2
    T = That / kh
    for a in range(k):
        # the per-axis factor cancels out of dkh/kh: a log-derivative
        dkh = kh * dka[a] / ka[a]
        dT[a] = (dT[a] - T * dkh) / kh
    return T, dT, True


@nb.njit(cache=True, fastmath=True, nogil=True)
def _trend2d_resolution_kernel(H, M, cells, w, kg, n, reach, step, level):
    """Walk the sampling's own transform along ONE variable until it falls to
    `level` of its value at zero or turns back up into its own lobes."""
    two_pi = 2.0 * np.pi
    prev = 1.0
    steps = int(reach / step)
    for i in range(1, steps + 1):
        g = i * step
        ar = 0.0
        ai = 0.0
        kr = 0.0
        ki = 0.0
        for c in range(M):
            ang = -two_pi * g * ((c - w) / cells - 0.5)
            cr = np.cos(ang)
            sn = np.sin(ang)
            ar += H[c] * cr
            ai += H[c] * sn
            kr += kg[c] * cr
            ki += kg[c] * sn
        kabs = np.sqrt(kr * kr + ki * ki)
        if kabs < 1e-12:
            # past what the grid represents: nothing to compare any more
            return reach
        Wv = np.sqrt(ar * ar + ai * ai) / kabs / n
        if Wv <= level or Wv > prev:
            return g
        prev = Wv
    return reach


def trend2d_layout(cells, k, m=0):
    """Where one date's accumulator keeps what.

    TWO CHECKERBOARD HALVES, each its own phasor grid (real, imaginary), count
    and moments -- `half` columns each, the second straight after the first; the
    total is their sum, so nothing is accumulated twice. Then what only the
    total is read for: the sampling's own transform, ONE-DIMENSIONAL per
    variable (k grids of M cells, real), and the coarse profile over the m axis
    covariates.
    """
    M = int(cells) + 2 * TREND2D_W
    K = M ** k
    half = 2 * K + trend2d_moment_width(k)
    prof = 2 * half + k * M
    return {'M': M, 'K': K, 'half': half, 'H': 2 * half, 'profile': prof,
            'width': prof + (2 * TREND2D_PROFILE ** m if m else 0)}


def trend2d_width(cells, k, m=0):
    """Total accumulator columns, see trend2d_layout()."""
    return trend2d_layout(cells, k, m)['width']


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


def trend2d_fit(acc, cells, k, axes=(), maxiter=1000, tol=1e-10, degree=1):
    """The accumulator -> one gradient and one constant per date, SOLVED.

    `acc` is (dates, trend2d_width()) as trend2d_accumulate() lays it out: the
    two checkerboard halves, whose sum is the total this fits, and the
    sampling's one-dimensional transforms and axis profiles of the total.

    degree=0 fits the constant alone: the same closed form the constant has at
    any gradient, read at zero, so the gradients come back zero and the
    coherence is the zero-trend one; no ascent and no limit. Its error bar is
    the same halves' disagreement, of the constant, in radians, in the first
    error column.

    THE OBJECTIVE IS ALREADY A ROBUST REGRESSION. Maximising
    `sum cos(phi - 2 pi g.A - c)` is an M-estimator whose score is `sin` of
    the residual: bounded, redescending, one pixel can pull the fit by at most
    one unit. That is the same bounded influence the phasor form has always
    bought, written as a regression.

    IT IS SOLVED, AND THE SEARCH ONLY OFFERS STARTS. The old code took the
    global argmax over a lattice reaching `range`. Where the samples crowd
    into a fraction of the variable's extent -- real topography -- the
    sampling's own transform keeps grating lobes far out in the band, and the
    global argmax landed on one of them whenever the date was weak, answering
    with the histogram rather than the phase. The ascent from zero follows the
    objective to the stationary point CONNECTED to zero instead, which a
    strong trend on a smooth slope still reaches; a peak across a dip of the
    objective it does not. So the ascent is given more starts and the highest
    stationary point wins: the ramp read off the axis covariates' profiles,
    and the date's own grid SEARCHED over every covariate's whole reach
    (trend2d_search) -- the latter only where the two checkerboard halves,
    independent pixels of the same sampling, searched on their own, land
    within the sampling's resolution of the same point. A lobe lifted by
    noise is not lifted in both halves, and a date whose halves disagree keeps
    the starts it had. The reported `limit` is that resolution.

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
    per-variable one-sigma from the two checkerboard halves) -- the pair
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
    acc = np.asarray(acc, np.float64)
    cells = int(cells)
    axes = tuple(axes)
    m = len(axes)
    L = trend2d_layout(cells, k, m)
    M, K, hw = L['M'], L['K'], L['half']
    nd = acc.shape[0]
    halves = (acc[:, :hw], acc[:, hw:2 * hw])
    # THE TOTAL IS THE SUM OF THE HALVES: every pixel is in exactly one
    total = np.ascontiguousarray(halves[0] + halves[1])
    Sr, Si = total[:, :K], total[:, K:2 * K]
    H = acc[:, L['H']:L['H'] + k * M].reshape(nd, k, M)
    n = total[:, 2 * K]
    m1 = total[:, 2 * K + 1:2 * K + 1 + k]
    m2f = total[:, 2 * K + 1 + k:hw]
    prof = (acc[:, L['profile']:] if m else None)
    kg = trend2d_kernel(cells)
    kgr = np.ascontiguousarray(kg.real, dtype=np.float64)
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
        # a perfectly coherent, trend-free date would score -- until its power
        # halves or it turns back up into its own lobes. Diagnostic only. The
        # walk moves one gradient at a time, so each variable's own
        # one-dimensional transform is all it reads; dates that share their
        # pixels share the walk.
        for a in (range(k) if degree else ()):
            if d and n[d] == n[d - 1] and np.array_equal(H[d, a], H[d - 1, a]):
                lim[d, a] = lim[d - 1, a]
                continue
            lim[d, a] = _trend2d_resolution_kernel(
                np.ascontiguousarray(H[d, a]), M, float(cells), TREND2D_W, kgr,
                float(n[d]), float(reach), 0.02, TREND2D_RESOLUTION_LEVEL)
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
        Sd = (Sr[d], Si[d])
        T0, dT0 = trend2d_read(Sd, cells, k, np.zeros(k), kg)
        if not np.isfinite(T0) or abs(T0) < 1e-30:
            why[d] = 3
            continue
        coh0[d] = float(abs(T0) / max(n[d], 1.0))
        if not degree:
            g[d] = 0.0
            c[d] = float(np.angle(T0))
            coh[d] = coh0[d]
            continue

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
        # A SEARCHED START FOR EVERY COVARIATE, PROVED BY THE HALVES. The
        # date's own grid is searched over each covariate's whole reach -- a
        # raster like elevation has no profile to scan, and its peak can sit
        # across a dip from zero. What the old global argmax got wrong is that
        # a weak date's largest peak may be a lobe of the SAMPLING lifted by
        # noise: so the search is repeated on the two checkerboard halves,
        # independent pixels of the same sampling, and its point is a start
        # only where both halves' own searches land within the sampling's
        # resolution of it. A lobe that noise lifted is not lifted twice.
        if TREND2D_SEARCHED_START and halves[0][d, 2 * K] > 0 \
                and halves[1][d, 2 * K] > 0:
            gs = trend2d_search(Sd, cells, k, kg, reach)
            if np.max(np.abs(gs)) > 1e-9:
                for tot_h in (halves[0][d], halves[1][d]):
                    gh = trend2d_search((np.ascontiguousarray(tot_h[:K]),
                                         np.ascontiguousarray(tot_h[K:2 * K])),
                                        cells, k, kg, reach)
                    if np.any(np.abs(gh - gs) > lim[d]):
                        break
                else:
                    starts.append(gs)

        best = None
        failed0 = 0
        for bi, g_init in enumerate(starts):
            r = _trend2d_ascend(Sd, cells, k, g_init, kg, Xg, n[d], reach,
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
    if degree:
        for d in range(nd):
            if why[d]:
                continue
            gh = []
            for tot_h in (halves[0][d], halves[1][d]):
                Sh = (np.ascontiguousarray(tot_h[:K]),
                      np.ascontiguousarray(tot_h[K:2 * K]))
                nh = tot_h[2 * K]
                if nh <= 0:
                    break
                m1h = tot_h[2 * K + 1:2 * K + 1 + k]
                m2h = np.empty((k, k))
                pp = 2 * K + 1 + k
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
    else:
        # the constant of each half, closed form like the full one; the
        # difference is an angle, so it is taken on the circle
        for d in range(nd):
            if why[d]:
                continue
            ch = []
            for tot_h in (halves[0][d], halves[1][d]):
                if tot_h[2 * K] <= 0:
                    break
                Th = trend2d_read((np.ascontiguousarray(tot_h[:K]),
                                   np.ascontiguousarray(tot_h[K:2 * K])),
                                  cells, k, np.zeros(k), kg)[0]
                if not np.isfinite(Th) or abs(Th) < 1e-30:
                    break
                ch.append(float(np.angle(Th)))
            if len(ch) == 2:
                err[d, 0] = 0.5 * abs(float(np.angle(np.exp(1j * (ch[0] - ch[1])))))
    resolved = why == 0
    g[~resolved] = np.nan
    c[~resolved] = np.nan
    return g, c, coh, coh0, resolved, why, lim, err


def trend2d_search(Sd, cells, k, kg, reach):
    """One spread grid searched for a START, every covariate over its whole reach.

    The grid holds the coherent sum at every gradient, so along one covariate,
    through the current point, it is a one-dimensional transform: the other
    axes are contracted at the current gradients and ONE padded FFT reads the
    whole line. Each round reads every covariate's line from the same point and
    takes only the single change that raises the sum most, until none does --
    so the order the covariates were named in cannot matter. Starts from zero
    and returns zero when nothing is higher. A lattice point, not a solution:
    the ascent still produces the number.
    """
    import numpy as np
    Sr, Si = _trend2d_parts(Sd)
    cells = int(cells)
    M = cells + 2 * TREND2D_W
    S = (Sr + 1j * Si).reshape((M,) * k)
    N = TREND2D_SEARCH_OVER * cells
    while N < 2 * M:
        N *= 2
    gj = np.fft.fftfreq(N) * cells                 # turns across the extent
    # the kernel's own transform on the same lattice divides out of the line
    Kf = np.abs(np.fft.fft(np.asarray(kg).real, N))
    band = (np.abs(gj) <= reach - 0.25) & (Kf > 1e-12)
    Am = trend2d_nodes(cells)
    gd = np.zeros(k)
    best = abs(_trend2d_read_parts(Sr, Si, cells, k, gd, kg)[0])
    if not np.isfinite(best):
        return gd
    for _ in range(4 * k):
        pick = None
        for a in range(k):
            line = S
            for b in range(k - 1, -1, -1):
                if b != a:
                    line = np.tensordot(line, np.exp(-2j * np.pi * gd[b] * Am),
                                        axes=([b], [0]))
            F = np.abs(np.fft.fft(line, N))
            F = np.where(band, F / np.where(band, Kf, 1.0), 0.0)
            cand = gd.copy()
            cand[a] = gj[int(np.argmax(F))]
            val = abs(_trend2d_read_parts(Sr, Si, cells, k, cand, kg)[0])
            if np.isfinite(val) and val > best * (1.0 + 1e-9) \
                    and (pick is None or val > pick[0]):
                pick = (val, cand)
        if pick is None:
            break
        best, gd = pick
    return gd


def _trend2d_ascend(Sd, cells, k, g_init, kg, Xg, nd_, reach, maxiter, tol):
    """One ascent to a stationary point. Returns (g, T, |T|) or None, the
    failure reason left in .last_fail (2 domain, 3 degenerate, 4 no
    convergence)."""
    import numpy as np
    _trend2d_ascend.last_fail = 0
    gd = np.array(g_init, np.float64)
    Sd = _trend2d_parts(Sd)
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


def trend2d_accumulate_samples(u, A, cells, par=None, axes=()):
    """The accumulator of a LIST of samples, (dates, trend2d_width()).

    `u` (dates, samples) are the unit phasors, zero where a date has no sample;
    `A` (samples, k) the covariates already centred and scaled into
    [-1/2, 1/2]; `par` (samples,) the checkerboard half of each, 0 or 1 -- left
    out, every sample is in the first; `axes` names the covariates the coarse
    profile is kept over. The same sums trend2d_accumulate() makes of a block,
    through the plain spreader: what a test or an experiment holding pixels
    rather than rasters feeds trend2d_fit().
    """
    import numpy as np
    u = np.atleast_2d(np.asarray(u, np.complex128))
    A = np.asarray(A, np.float64)
    if A.ndim == 1:
        A = A[:, None]
    nd, k = u.shape[0], A.shape[1]
    axes = tuple(axes)
    L = trend2d_layout(cells, k, len(axes))
    M, K, hw = L['M'], L['K'], L['half']
    out = np.zeros((nd, L['width']), np.float64)
    par = (np.zeros(A.shape[0], np.int64) if par is None
           else np.asarray(par, np.int64))
    have = (np.abs(u) > 0).astype(np.float64)
    for h in range(2):
        sel = par == h
        if not sel.any():
            continue
        gr, gi = trend2d_spread(u[:, sel], A[sel], cells)
        o = h * hw
        out[:, o:o + K] = gr
        out[:, o + K:o + 2 * K] = gi
        out[:, o + 2 * K] = have[:, sel].sum(axis=1)
        p = o + 2 * K + 1
        for i in range(k):
            out[:, p + i] = have[:, sel] @ A[sel, i]
        p += k
        for i in range(k):
            for j in range(i, k):
                out[:, p] = have[:, sel] @ (A[sel, i] * A[sel, j])
                p += 1
    for a in range(k):
        out[:, L['H'] + a * M:L['H'] + (a + 1) * M] = trend2d_spread(
            have.astype(np.complex128), A[:, [a]], cells)[0]
    if axes:
        out[:, L['profile']:] = trend2d_profile(u, A[:, list(axes)])
    return out


def trend2d_accumulate(data_blk, transform_blk, stats, cells, dims=None,
                       extent=None, coords=None):
    """One spatial block, every date it is handed -> the sums the fit reads,
    (dates, 1, 1, trend2d_width()), laid out as trend2d_layout() says.

    Everything the estimator reads is a sum over pixels, so a block
    contributes its share and the caller adds them: per checkerboard half the
    phasor grid, the sample count and the first and second moments of A (the
    regression's normal matrix); for the total the sampling's own transform
    per variable and the coarse profile over the axis covariates. All additive.

    THE HALVES ARE AN 8x8 BOARD OVER `extent` (y0, y1, x0, x1), read off the
    block's own `coords` (y, x): spatially coarse enough to carry independent
    atmosphere. Without them every pixel is in the first half and there is no
    second to disagree with.

    THE DATES ARE TAKEN AS THEY COME AND FED TO THE KERNEL IN GROUPS: a pixel's
    kernel weights do not depend on the date, so they are computed once per
    group, and the group is as many dates as hold the samples and the grids
    within one dask chunk -- the size every other intermediate of a task is
    budgeted at, whatever the caller's date chunks are.
    """
    import numpy as np
    from .utils_dask import get_dask_chunk_size_mb
    k = len(transform_blk)
    nb = data_blk.shape[0]
    stats = np.asarray(stats, np.float64).ravel()
    mu = stats[:k]
    span = np.maximum(stats[k:2 * k], 1e-30)
    w = TREND2D_W
    beta = TREND2D_BETA * w
    P = TREND2D_PROFILE
    ny, nx = data_blk.shape[-2:]
    if dims is None:
        dims = ['yx'] * k
    dims = list(dims)
    ax_idx = [i for i, d in enumerate(dims) if d in ('y', 'x')][:2]
    m = len(ax_idx)
    L = trend2d_layout(cells, k, m)
    M, K, hw = L['M'], L['K'], L['half']
    mw = trend2d_moment_width(k)
    out = np.zeros((nb, 1, 1, L['width']), np.float64)

    # the positions this block can contribute: geometry, and a date with phase
    V = [np.asarray(b, np.float32) for b in transform_blk]
    blk = np.ascontiguousarray(data_blk).reshape(nb, ny * nx)
    flags = np.zeros(ny * nx, np.uint8)
    _trend2d_anyvalid(blk, flags)
    keep = flags.view(np.bool_)
    for v, d in zip(V, dims):
        if d == 'yx':
            keep &= np.isfinite(v.reshape(-1))
    # A VARIABLE ALONG ONE AXIS RULES OUT WHOLE ROWS OR COLUMNS, and normally
    # none, so the raster-sized mask is only built if it has to be
    bad = {}
    for i, (v, d) in enumerate(zip(V, dims)):
        if d != 'yx':
            bad[i] = ~np.isfinite(v)
            if bad[i].any():
                keep &= ~(np.repeat(bad[i], nx) if d == 'y' else np.tile(bad[i], ny))
    idx = np.flatnonzero(keep)
    npts = idx.size
    if npts == 0:
        return out
    rows = idx // nx
    colsi = idx - rows * nx
    # the half each pixel belongs to
    if coords is not None and extent is not None:
        yb, xb = coords
        y0, y1, x0, x1 = extent
        iy = np.minimum((np.asarray(yb, np.float64) - y0)
                        / max(y1 - y0, 1e-30) * 8, 7).astype(np.int64)
        ix = np.minimum((np.asarray(xb, np.float64) - x0)
                        / max(x1 - x0, 1e-30) * 8, 7).astype(np.int64)
        par = ((iy[rows] + ix[colsi]) % 2).astype(np.int64)
    else:
        par = np.zeros(npts, np.int64)

    # the covariates by what they are: vectors along y lead the grid, vectors
    # along x and rasters follow -- the kernel's own order, undone at the end.
    # Scaled to the box the grid spans: the midpoint centring is what puts
    # every sample inside [-1/2, 1/2]
    ys = [i for i, d in enumerate(dims) if d == 'y']
    xs = [i for i, d in enumerate(dims) if d == 'x']
    rs = [i for i, d in enumerate(dims) if d == 'yx']

    def scaled(i, values):
        return (np.asarray(values, np.float64) - mu[i]) / span[i]
    WY = np.zeros((max(len(ys), 1), ny, w))
    BY = np.zeros((max(len(ys), 1), ny), np.int64)
    AY = np.zeros((max(len(ys), 1), ny))
    for j, i in enumerate(ys):
        AY[j] = np.where(bad[i], 0.0, scaled(i, V[i]))
        WY[j], BY[j] = _trend2d_axis_tables(AY[j], cells, bad[i])
    WX = np.zeros((max(len(xs), 1), nx, w))
    BX = np.zeros((max(len(xs), 1), nx), np.int64)
    AX = np.zeros((max(len(xs), 1), nx))
    for j, i in enumerate(xs):
        AX[j] = np.where(bad[i], 0.0, scaled(i, V[i]))
        WX[j], BX[j] = _trend2d_axis_tables(AX[j], cells, bad[i])
    xlo = np.ascontiguousarray(BX.min(axis=1))
    xhi = np.ascontiguousarray(BX.max(axis=1) + w)
    AR = np.zeros((max(len(rs), 1), npts))
    for j, i in enumerate(rs):
        AR[j] = scaled(i, V[i].reshape(-1)[idx])
    vclass = np.array([0 if d == 'y' else 1 if d == 'x' else 2 for d in dims], np.int64)
    vslot = np.array([(ys if d == 'y' else xs if d == 'x' else rs).index(i)
                      for i, d in enumerate(dims)], np.int64)
    oslot = np.arange(len(ys), dtype=np.int64)
    ovar = np.array(ys, np.int64)
    ikind = np.array([1] * len(xs) + [2] * len(rs), np.int64)
    islot = np.array(list(range(len(xs))) + list(range(len(rs))), np.int64)
    ivar = np.array(xs + rs, np.int64)
    # the coarse profiles' bins, per row or per column
    pclass = np.array([0 if dims[i] == 'y' else 1 for i in ax_idx], np.int64)
    pslot = np.array([(ys if dims[i] == 'y' else xs).index(i) for i in ax_idx], np.int64)
    PY = np.clip(((AY + 0.5) * P).astype(np.int64), 0, P - 1)
    PX = np.clip(((AX + 0.5) * P).astype(np.int64), 0, P - 1)
    nout, nin = len(ys), len(xs) + len(rs)
    Nout, Nin = M ** nout, M ** nin
    canon = ys + xs + rs
    back = (0,) + tuple(1 + canon.index(a) for a in range(k)) + (k + 1,)

    # as many dates per pass as keep the samples and the grids within a chunk
    budget = get_dask_chunk_size_mb() * 2 ** 20
    per_date = 16 * npts + 16 * 2 * K + 16 * 2 * Nin + 8 * L['width']
    group = int(max(1, min(nb, budget // per_date)))
    for t0 in range(0, nb, group):
        nq = min(group, nb - t0)
        ur = np.empty((npts, nq))
        ui = np.empty((npts, nq))
        cnt = np.zeros(nq, np.int64)
        _trend2d_gather(blk[t0:t0 + nq], idx, ur, ui, cnt)
        full = bool((cnt == npts).all())
        nqh = 1 if full else nq
        gr = np.zeros((2, Nout, Nin, nq))
        gi = np.zeros((2, Nout, Nin, nq))
        if nout:
            tr = np.zeros((2, Nin, nq))
            ti = np.zeros((2, Nin, nq))
        else:
            tr, ti = gr[:, 0], gi[:, 0]
        Hg = np.zeros((k, M, nqh))
        mn = np.zeros((2, nqh))
        m1 = np.zeros((2, k, nqh))
        m2 = np.zeros((2, k * (k + 1) // 2, nqh))
        pr = np.zeros((P ** m if m else 1, nq))
        pi = np.zeros((P ** m if m else 1, nq))
        _trend2d_spread_rows(rows, colsi, par, ur, ui, full, w, beta, float(cells), M,
                             nx, WY, BY, AY, WX, BX, AX, AR, xlo, xhi,
                             oslot, ovar, ikind, islot, ivar,
                             vclass, vslot, pclass, pslot, PY, PX, P,
                             tr, ti, gr, gi, Hg, mn, m1, m2, pr, pi)
        del ur, ui
        o = out[t0:t0 + nq, 0, 0]
        # the grid back in the variables' own order, the date first
        for part, col in ((gr, 0), (gi, K)):
            g_ = np.transpose(part.reshape((2,) + (M,) * k + (nq,)), back)
            g_ = np.moveaxis(g_, -1, 0).reshape(nq, 2, K)
            for h in range(2):
                o[:, h * hw + col:h * hw + col + K] = g_[:, h]
        for h in range(2):
            o[:, h * hw + 2 * K] = mn[h]
            o[:, h * hw + 2 * K + 1:h * hw + 2 * K + 1 + k] = m1[h].T
            o[:, h * hw + 2 * K + 1 + k:h * hw + 2 * K + mw] = m2[h].T
        o[:, L['H']:L['H'] + k * M] = np.moveaxis(Hg, -1, 0).reshape(nqh, k * M)
        if m:
            o[:, L['profile']:L['profile'] + P ** m] = pr.T
            o[:, L['profile'] + P ** m:] = pi.T
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
