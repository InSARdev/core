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

_F0 = np.float32(0.0)


def _kernel1d(s, trunc):
    """The normalized float32 Gaussian of sigma `s` samples, truncated at `trunc` sigmas; [1] for sigma 0."""
    if s <= 0:
        return np.ones(1, np.float32)
    size = int(2 * np.ceil(s * trunc) + 1)
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / s) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


@_numba.njit(nogil=True, cache=True, fastmath={'contract'})
def _xpass(s, k, d):
    """d = s filtered along x by the symmetric kernel k, the edge values repeated. Taps t and -t are added before their
    multiply, four tap pairs a pass in a fixed order; the inner loops run over slices by their own counter (no
    negative-index branch, so they vectorize)."""
    n = s.size
    r = k.size // 2
    kr = k[r]
    for i in range(n):
        d[i] = kr * s[i]
    t = 1
    while t + 3 <= r:
        k1, k2, k3, k4 = k[r - t], k[r - t - 1], k[r - t - 2], k[r - t - 3]
        e = t + 3
        m = n - 2 * e
        if m > 0:
            dd = d[e:n - e]
            l1, r1 = s[3:3 + m], s[e + t:e + t + m]
            l2, r2 = s[2:2 + m], s[e + t + 1:e + t + 1 + m]
            l3, r3 = s[1:1 + m], s[e + t + 2:e + t + 2 + m]
            l4, r4 = s[0:m], s[e + t + 3:e + t + 3 + m]
            for i in range(m):
                dd[i] = (((dd[i] + k1 * (l1[i] + r1[i])) + k2 * (l2[i] + r2[i])) + k3 * (l3[i] + r3[i])) \
                    + k4 * (l4[i] + r4[i])
        for x in range(min(e, n)):
            for tt in range(t, e + 1):
                d[x] += k[r - tt] * (s[max(x - tt, 0)] + s[min(x + tt, n - 1)])
        for x in range(max(n - e, e), n):
            for tt in range(t, e + 1):
                d[x] += k[r - tt] * (s[max(x - tt, 0)] + s[min(x + tt, n - 1)])
        t += 4
    while t <= r:
        kt = k[r - t]
        m = n - 2 * t
        if m > 0:
            dd = d[t:n - t]
            sl = s[0:m]
            sr = s[2 * t:n]
            for i in range(m):
                dd[i] += kt * (sl[i] + sr[i])
        for x in range(min(t, n)):
            d[x] += kt * (s[max(x - t, 0)] + s[min(x + t, n - 1)])
        for x in range(max(n - t, t), n):
            d[x] += kt * (s[max(x - t, 0)] + s[min(x + t, n - 1)])
        t += 1


@_numba.njit(nogil=True, cache=True, fastmath={'contract'})
def _ypass(rows, k, r, out):
    """out = rows r : r + k.size combined by the symmetric kernel k, four tap pairs a pass in a fixed order."""
    ry = k.size // 2
    c = rows[r + ry]
    kc = k[ry]
    for x in range(out.size):
        out[x] = kc * c[x]
    t = 1
    while t + 3 <= ry:
        k1, k2, k3, k4 = k[ry - t], k[ry - t - 1], k[ry - t - 2], k[ry - t - 3]
        u1, v1 = rows[r + ry - t], rows[r + ry + t]
        u2, v2 = rows[r + ry - t - 1], rows[r + ry + t + 1]
        u3, v3 = rows[r + ry - t - 2], rows[r + ry + t + 2]
        u4, v4 = rows[r + ry - t - 3], rows[r + ry + t + 3]
        for x in range(out.size):
            out[x] = (((out[x] + k1 * (u1[x] + v1[x])) + k2 * (u2[x] + v2[x])) + k3 * (u3[x] + v3[x])) \
                + k4 * (u4[x] + v4[x])
        t += 4
    while t <= ry:
        kt = k[ry - t]
        u, v = rows[r + ry - t], rows[r + ry + t]
        for x in range(out.size):
            out[x] += kt * (u[x] + v[x])
        t += 1


@_numba.njit(nogil=True, cache=True, fastmath={'contract'})
def _conv_real(data, weight, has_w, kx, ky, lim, out):
    """out = sum(k w v) / (sum(k w) + 1e-17) of a real grid, w the weight (1 without one) where v is not NaN and 0
    where it is; NaN where the weight sum is at most lim (with a weight: under lim). Bands of rows, each row filtered
    along x once, then along y."""
    H, W = data.shape
    ry = ky.size // 2
    band = max(64, 4 * ry)
    xv = np.empty((min(band, H) + 2 * ry, W), np.float32)
    xw = np.empty((min(band, H) + 2 * ry, W), np.float32)
    fv = np.empty(W, np.float32)
    fw = np.empty(W, np.float32)
    av = np.empty(W, np.float32)
    aw = np.empty(W, np.float32)
    nan32 = np.float32(np.nan)
    tiny = np.float32(1e-17)
    one = np.float32(1.0)
    for r0 in range(0, H, band):
        nr = min(band, H - r0)
        for q in range(nr + 2 * ry):
            y = min(max(r0 - ry + q, 0), H - 1)
            row = data[y]
            if has_w:
                wrow = weight[y]
                for x in range(W):
                    v = row[x]
                    w = wrow[x]
                    w = w if w == w else _F0
                    ok = v == v
                    fw[x] = w if ok else _F0
                    fv[x] = v * w if ok else _F0
            else:
                for x in range(W):
                    v = row[x]
                    ok = v == v
                    fw[x] = one if ok else _F0
                    fv[x] = v if ok else _F0
            _xpass(fv, kx, xv[q])
            _xpass(fw, kx, xw[q])
        for r in range(nr):
            _ypass(xv, ky, r, av)
            _ypass(xw, ky, r, aw)
            orow = out[r0 + r]
            for x in range(W):
                d = aw[x]
                bad = (d < lim) if has_w else (d <= lim)
                orow[x] = nan32 if bad else av[x] / (d + tiny)


@_numba.njit(nogil=True, cache=True, fastmath={'contract'})
def _conv_complex(data, weight, has_w, kx, ky, lim, out):
    """_conv_real of both parts of a complex grid at once, one weight sum for both (each part gets what _conv_real gives
    it). False, out left unfinished, when a value has one NaN part: its parts have different weights."""
    H, W = data.shape
    ry = ky.size // 2
    band = max(64, 4 * ry)
    xr = np.empty((min(band, H) + 2 * ry, W), np.float32)
    xi = np.empty((min(band, H) + 2 * ry, W), np.float32)
    xw = np.empty((min(band, H) + 2 * ry, W), np.float32)
    fr = np.empty(W, np.float32)
    fi = np.empty(W, np.float32)
    fw = np.empty(W, np.float32)
    ar = np.empty(W, np.float32)
    ai = np.empty(W, np.float32)
    aw = np.empty(W, np.float32)
    nanc = np.complex64(complex(np.nan, np.nan))
    tiny = np.float32(1e-17)
    one = np.float32(1.0)
    for r0 in range(0, H, band):
        nr = min(band, H - r0)
        for q in range(nr + 2 * ry):
            y = min(max(r0 - ry + q, 0), H - 1)
            row = data[y]
            for x in range(W):
                v = row[x]
                if (v.real != v.real) != (v.imag != v.imag):
                    return False
            if has_w:
                wrow = weight[y]
                for x in range(W):
                    v = row[x]
                    w = wrow[x]
                    w = w if w == w else _F0
                    ok = v.real == v.real
                    fw[x] = w if ok else _F0
                    fr[x] = v.real * w if ok else _F0
                    fi[x] = v.imag * w if ok else _F0
            else:
                for x in range(W):
                    v = row[x]
                    ok = v.real == v.real
                    fw[x] = one if ok else _F0
                    fr[x] = v.real if ok else _F0
                    fi[x] = v.imag if ok else _F0
            _xpass(fr, kx, xr[q])
            _xpass(fi, kx, xi[q])
            _xpass(fw, kx, xw[q])
        for r in range(nr):
            _ypass(xr, ky, r, ar)
            _ypass(xi, ky, r, ai)
            _ypass(xw, ky, r, aw)
            orow = out[r0 + r]
            for x in range(W):
                d = aw[x]
                if (d < lim) if has_w else (d <= lim):
                    orow[x] = nanc
                else:
                    d = d + tiny
                    orow[x] = np.complex64(complex(ar[x] / d, ai[x] / d))
    return True


def nanconvolve2d_gaussian(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5):
    """
    2D Gaussian convolution with NaN handling, directly: two 1-D passes in numba, the edge values repeated.

    A NaN value has no weight, and the output is NaN where the weight sum is at most `threshold` (with a weight: under
    1e-10). The parts of a complex grid are filtered with their own NaN masks. Requires float32 or complex64 input
    and a float32 weight.

    Parameters
    ----------
    data_np : np.ndarray
        2D numpy array (float32 or complex64) to convolve.
    weight_np : np.ndarray, optional
        Weight array (float32) for weighted convolution.
    sigma : float or tuple of floats
        Standard deviation of Gaussian kernel (y, x).
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).

    Returns
    -------
    np.ndarray
        Convolved array (float32 or complex64) with same shape as input.

    Examples
    --------
    >>> out = nanconvolve2d_gaussian(data, None, sigma=(2.0, 8.0))
    """
    if sigma is None:
        return data_np

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)

    is_complex = np.issubdtype(data_np.dtype, np.complexfloating)
    assert data_np.dtype == (np.complex64 if is_complex else np.float32), \
        f"Input must be float32 or complex64, got {data_np.dtype}"
    has_w = weight_np is not None
    if has_w:
        assert weight_np.dtype == np.float32, f"Weight must be float32, got {weight_np.dtype}"
        # the kernels index both blocks alike and check no bounds
        if weight_np.shape != data_np.shape:
            raise ValueError(f'gaussian: weight block {weight_np.shape} differs from the data block {data_np.shape}')
    w = weight_np if has_w else np.empty((0, 0), np.float32)
    kernel_y = _kernel1d(sigma[0], truncate)
    kernel_x = _kernel1d(sigma[1], truncate)
    lim = np.float32(1e-10 if has_w else threshold)

    if not is_complex:
        result = np.empty(data_np.shape, np.float32)
        _conv_real(data_np, w, has_w, kernel_x, kernel_y, lim, result)
        return result
    result = np.empty(data_np.shape, np.complex64)
    if not _conv_complex(data_np, w, has_w, kernel_x, kernel_y, lim, result):
        # a value with one NaN part: each part with its own mask
        _conv_real(data_np.real, w, has_w, kernel_x, kernel_y, lim, result.real)
        _conv_real(data_np.imag, w, has_w, kernel_x, kernel_y, lim, result.imag)
    return result


# GMTSAR constant: the half-gain wavelength is 5.3 sigma
GAUSSIAN_CUTOFF = 5.3
# the decimated filter's top cell: at most a sixteenth of the wavelength
_CELLS = 16
# a step's anti-aliasing Gaussian, in its input samples: narrower, a sampled Gaussian is not one (its spectrum at the
# Nyquist frequency, exp(-pi^2 sigma^2 / 2), is 3e-9 at 2 samples and 0.49 at 0.4)
_AA_SIGMA = 2.0


@_numba.njit(nogil=True, cache=True)
def _fields_real(data, weight, has_w, out):
    """out[0] = v w and out[1] = w of a real grid: w the weight (1 without one, 0 where it is NaN), 0 where v is NaN."""
    H, W = data.shape
    one = np.float32(1.0)
    for y in range(H):
        for x in range(W):
            v = data[y, x]
            w = one
            if has_w:
                w = weight[y, x]
                w = w if w == w else _F0
            ok = v == v
            out[1, y, x] = w if ok else _F0
            out[0, y, x] = v * w if ok else _F0


@_numba.njit(nogil=True, cache=True)
def _fields_complex(data, weight, has_w, out):
    """out = (re wr, im wi, wr, wi) of a complex grid, each part with its own NaN mask (_fields_real)."""
    H, W = data.shape
    one = np.float32(1.0)
    for y in range(H):
        for x in range(W):
            v = data[y, x]
            w = one
            if has_w:
                w = weight[y, x]
                w = w if w == w else _F0
            okr = v.real == v.real
            oki = v.imag == v.imag
            out[2, y, x] = w if okr else _F0
            out[3, y, x] = w if oki else _F0
            out[0, y, x] = v.real * w if okr else _F0
            out[1, y, x] = v.imag * w if oki else _F0


@_numba.njit(nogil=True, cache=True, fastmath={'contract'})
def _filter_fields(src, ky, kx, out):
    """out = every field of src filtered along y by ky and along x by kx (symmetric kernels), the edge values repeated."""
    F, H, W = src.shape
    ry = ky.size // 2
    xb = np.empty((H + 2 * ry, W), np.float32)
    for f in range(F):
        for q in range(H + 2 * ry):
            y = min(max(q - ry, 0), H - 1)
            if kx.size > 1:
                _xpass(src[f, y], kx, xb[q])
            else:
                xb[q] = src[f, y]
        for r in range(H):
            if ry > 0:
                _ypass(xb, ky, r, out[f, r])
            else:
                out[f, r] = xb[r]


@_numba.njit(nogil=True, cache=True)
def _coarsen_fields(src, oy, ox, fy, fx, cy0, cx0, ny, nx, out):
    """out[:, i, j] = the mean of the fy x fx samples of cell (cy0 + i, cx0 + j) (the cells counted from the first sample
    of the array) in src, whose first sample is (oy, ox) of an array of ny x nx; a sample past its end is its last one."""
    F, nci, ncj = out.shape
    inv = np.float32(1.0) / np.float32(fy * fx)
    for f in range(F):
        for i in range(nci):
            for j in range(ncj):
                s = _F0
                for a in range(fy):
                    row = src[f, min((cy0 + i) * fy + a, ny - 1) - oy]
                    for b in range(fx):
                        s += row[min((cx0 + j) * fx + b, nx - 1) - ox]
                out[f, i, j] = s * inv


@_numba.njit(nogil=True, cache=True)
def _axis_weights(n, origin, s, nc, c0):
    """For n samples from `origin`: the cell before each (counted from the cells' first, c0), the next one and the
    weight of the next. A cell's value sits at its centre, (i + 1/2) s - 1/2; past the outer centres, the outer cell."""
    j0 = np.empty(n, np.int64)
    j1 = np.empty(n, np.int64)
    t = np.empty(n, np.float32)
    for k in range(n):
        u = ((origin + k) - (s - 1) * 0.5) / s
        j = int(np.floor(u))
        w = u - j
        if j < 0:
            j = 0
            w = 0.0
        if j >= nc - 1:
            j = nc - 1
            w = 0.0
        j0[k] = j - c0
        j1[k] = min(j + 1, nc - 1) - c0
        t[k] = w
    return j0, j1, t


@_numba.njit(nogil=True, cache=True)
def _lines(cells, i, j0, j1, t, out):
    """out[f] = row i of every field of cells interpolated along x (_axis_weights)."""
    one = np.float32(1.0)
    for f in range(cells.shape[0]):
        row = cells[f, i]
        o = out[f]
        for x in range(o.size):
            o[x] = (one - t[x]) * row[j0[x]] + t[x] * row[j1[x]]


@_numba.njit(nogil=True, cache=True)
def _upsample_real(cells, cy0, cx0, ncy, ncx, sy, sx, y0, x0, has_w, lim, out):
    """out = at every pixel from (y0, x0), the top cells' weighted sum over their weight sum, both interpolated
    bilinearly (_axis_weights); NaN where the weight sum is at most lim (with a weight: under lim)."""
    H, W = out.shape
    i0s, i1s, ty = _axis_weights(H, y0, sy, ncy, cy0)
    j0, j1, tx = _axis_weights(W, x0, sx, ncx, cx0)
    la = np.empty((2, W), np.float32)
    lb = np.empty((2, W), np.float32)
    ra = -1
    rb = -1
    nan32 = np.float32(np.nan)
    tiny = np.float32(1e-17)
    one = np.float32(1.0)
    for y in range(H):
        i0, i1, t = i0s[y], i1s[y], ty[y]
        if ra != i0:
            if rb == i0:
                la, lb = lb, la
                ra, rb = rb, ra
            else:
                _lines(cells, i0, j0, j1, tx, la)
                ra = i0
        if rb != i1:
            _lines(cells, i1, j0, j1, tx, lb)
            rb = i1
        for x in range(W):
            num = (one - t) * la[0, x] + t * lb[0, x]
            den = (one - t) * la[1, x] + t * lb[1, x]
            bad = (den < lim) if has_w else (den <= lim)
            out[y, x] = nan32 if bad else num / (den + tiny)


@_numba.njit(nogil=True, cache=True)
def _upsample_complex(cells, cy0, cx0, ncy, ncx, sy, sx, y0, x0, has_w, lim, out):
    """_upsample_real of a complex grid's fields (re wr, im wi, wr, wi): each part over its own weight sum."""
    H, W = out.shape
    i0s, i1s, ty = _axis_weights(H, y0, sy, ncy, cy0)
    j0, j1, tx = _axis_weights(W, x0, sx, ncx, cx0)
    la = np.empty((4, W), np.float32)
    lb = np.empty((4, W), np.float32)
    ra = -1
    rb = -1
    nan32 = np.float32(np.nan)
    tiny = np.float32(1e-17)
    one = np.float32(1.0)
    for y in range(H):
        i0, i1, t = i0s[y], i1s[y], ty[y]
        if ra != i0:
            if rb == i0:
                la, lb = lb, la
                ra, rb = rb, ra
            else:
                _lines(cells, i0, j0, j1, tx, la)
                ra = i0
        if rb != i1:
            _lines(cells, i1, j0, j1, tx, lb)
            rb = i1
        for x in range(W):
            nr = (one - t) * la[0, x] + t * lb[0, x]
            ni = (one - t) * la[1, x] + t * lb[1, x]
            dr = (one - t) * la[2, x] + t * lb[2, x]
            di = (one - t) * la[3, x] + t * lb[3, x]
            badr = (dr < lim) if has_w else (dr <= lim)
            badi = (di < lim) if has_w else (di <= lim)
            out[y, x] = np.complex64(complex(nan32 if badr else nr / (dr + tiny), nan32 if badi else ni / (di + tiny)))


def _pow2_floor(v):
    """The largest power of 2 at most v (1 below 2)."""
    p = 1
    while 2 * p <= v:
        p *= 2
    return p


def decimation_plan(sigma, pixel_sizes, truncate=4.0):
    """
    The decimation of a Gaussian of `sigma` pixels (y, x) on a grid of `pixel_sizes` metres (y, x), or None when no
    axis is decimated and the filter runs directly.

    Per axis the top cell is the largest power-of-2 multiple of the pixel that is at most a sixteenth of the
    wavelength, reached in steps of 2 (an axis with fewer steps takes them first). A step averages pairs of cells after
    a Gaussian of 2 samples against aliasing. The top filter keeps what remains of the Gaussian's variance after the
    anti-aliasing filters, the cell averages and the bilinear interpolation back to the pixels: about 3 to 6 top cells,
    as the 16 to 32 cells of a wavelength make it.

    Returns a dict: levels, one dict per step with f (y, x) and aa (y, x), the anti-aliasing sigma in the step's input
    samples (0 for none); scale (y, x), the top cell in pixels; sigma (y, x), the top filter in top cells.

    Examples
    --------
    >>> plan = decimation_plan((9.4, 37.7), (8.0, 2.0))
    """
    if pixel_sizes is None:
        return None
    d = np.asarray(pixel_sizes, dtype=np.float64)
    lam = np.asarray(sigma, dtype=np.float64) * GAUSSIAN_CUTOFF * d
    scale = [_pow2_floor(lam[a] / (_CELLS * d[a])) for a in (0, 1)]
    if scale == [1, 1]:
        return None
    steps = [[2] * (s.bit_length() - 1) for s in scale]
    n = max(len(s) for s in steps)
    steps = [s + [1] * (n - len(s)) for s in steps]
    # the anti-aliasing kernel's own variance, in samples squared
    k = _kernel1d(_AA_SIGMA, truncate).astype(np.float64)
    var_k = float(np.sum(k * (np.arange(k.size) - k.size // 2) ** 2))
    cell = d.copy()
    var_aa = np.zeros(2)
    levels = []
    for i in range(n):
        for a in (0, 1):
            if steps[a][i] > 1:
                var_aa[a] += var_k * cell[a] ** 2
                cell[a] *= steps[a][i]
        levels.append(dict(f=(steps[0][i], steps[1][i]),
                           aa=tuple(_AA_SIGMA if steps[a][i] > 1 else 0.0 for a in (0, 1))))
    # the cell averages add (top**2 - pixel**2) / 12 and the bilinear interpolation (top**2 - pixel**2) / 6
    var = (lam / GAUSSIAN_CUTOFF) ** 2 - var_aa - (cell ** 2 - d ** 2) / 4
    return dict(levels=levels, scale=tuple(scale), sigma=tuple(np.sqrt(var) / cell))


def _down_step(fields, oy, ox, level, n_in, cells, truncate):
    """One decimation step: fields (F, h, w), whose first sample is (oy, ox) of the step's input of n_in samples, to the
    means of the cells cells[0] (y range) x cells[1] (x range), after the step's anti-aliasing filter."""
    fy, fx = level['f']
    if level['aa'][0] > 0 or level['aa'][1] > 0:
        src = np.empty_like(fields)
        _filter_fields(fields, _kernel1d(level['aa'][0], truncate), _kernel1d(level['aa'][1], truncate), src)
    else:
        src = fields
    out = np.empty((fields.shape[0], cells[0][1] - cells[0][0], cells[1][1] - cells[1][0]), np.float32)
    _coarsen_fields(src, oy, ox, fy, fx, cells[0][0], cells[1][0], n_in[0], n_in[1], out)
    return out


def _down_reach(level, truncate):
    """The anti-aliasing reach (y, x) of a step, in its input samples."""
    return tuple(int(np.ceil(s * truncate)) if s > 0 else 0 for s in level['aa'])


def _fields(data, weight):
    """The weighted values and the weights of a real (2 fields) or complex (4 fields) grid."""
    is_complex = np.issubdtype(data.dtype, np.complexfloating)
    out = np.empty((4 if is_complex else 2,) + data.shape, np.float32)
    has_w = weight is not None
    (_fields_complex if is_complex else _fields_real)(data, weight if has_w else np.empty((0, 0), np.float32), has_w,
                                                      out)
    return out


def _upsample(top, c0, ncells, scale, origin, shape, has_w, lim, dtype):
    """The pixels (`shape` from `origin`) from the top fields (whose first cell is c0 of ncells)."""
    out = np.empty(shape, dtype)
    (_upsample_complex if np.issubdtype(dtype, np.complexfloating) else _upsample_real)(
        top, c0[0], c0[1], ncells[0], ncells[1], scale[0], scale[1], origin[0], origin[1], has_w, lim, out)
    return out


def _call_window(fn, sources, ranges, kw):
    """fn of the windows of the sources: per source, its pieces (rows of chunks) cut to their local ranges and joined,
    with the window's first sample (y, x)."""
    wins = []
    for src in sources:
        if src is None:
            wins.append(None)
            continue
        grid, ys, xs, lo = src
        rows = [np.concatenate([p[..., y0:y1, x0:x1] for p, (x0, x1) in zip(row, xs)], axis=-1)
                for row, (y0, y1) in zip(grid, ys)]
        wins.append((np.concatenate(rows, axis=-2), lo))
    return fn(wins, ranges, **kw)


def _window_graph(name, fn, sources, windows, lead, chunks, dtype, kw):
    """A dask array of the leading chunks `lead` and the (y, x) `chunks`, its block fn(windows, ranges, **kw): per
    source (a dask array (n, y, x) or (n, F, y, x), a 2-D one shared by every n, or None), the samples windows[i](lo, hi)
    around the block's (y, x) ranges, clamped to the source and gathered from the chunks they lie in."""
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
    off = [np.concatenate([[0], np.cumsum(c)]).astype(int) for c in chunks]
    info = []
    for src in sources:
        if src is None:
            info.append(None)
            continue
        info.append((src.__dask_keys__(), [np.concatenate([[0], np.cumsum(c)]).astype(int) for c in src.chunks[-2:]],
                     src.shape[-2:], src.ndim))
    layer = {}
    for k in range(len(lead[0])):
        for i in range(len(chunks[0])):
            for j in range(len(chunks[1])):
                ranges = (int(off[0][i]), int(off[0][i + 1]), int(off[1][j]), int(off[1][j + 1]))
                specs = []
                for src_info, win in zip(info, windows):
                    if src_info is None:
                        specs.append(None)
                        continue
                    keys, soff, shape, ndim = src_info
                    keys = keys if ndim == 2 else keys[k] if ndim == 3 else keys[k][0]
                    lo, hi = [], []
                    for a, (r0, r1) in enumerate(((ranges[0], ranges[1]), (ranges[2], ranges[3]))):
                        w0, w1 = win[a](r0, r1)
                        lo.append(max(int(w0), 0))
                        hi.append(min(int(w1), int(shape[a])))
                    ia = range(int(np.searchsorted(soff[0], lo[0], 'right')) - 1, int(np.searchsorted(soff[0], hi[0], 'left')))
                    ib = range(int(np.searchsorted(soff[1], lo[1], 'right')) - 1, int(np.searchsorted(soff[1], hi[1], 'left')))
                    grid = [[keys[a][b] for b in ib] for a in ia]
                    ys = [(max(lo[0], int(soff[0][a])) - int(soff[0][a]), min(hi[0], int(soff[0][a + 1])) - int(soff[0][a]))
                          for a in ia]
                    xs = [(max(lo[1], int(soff[1][b])) - int(soff[1][b]), min(hi[1], int(soff[1][b + 1])) - int(soff[1][b]))
                          for b in ib]
                    specs.append([grid, ys, xs, (lo[0], lo[1])])
                layer[(name, k) + (0,) * (len(lead) - 1) + (i, j)] = (_call_window, fn, specs, ranges, kw)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[s for s in sources if s is not None])
    allchunks = tuple(lead) + tuple(chunks)
    return da.Array(graph, name, allchunks, meta=np.empty((0,) * len(allchunks), dtype))


def _block_weight(wins, k):
    """The weight window of block k, or None."""
    if len(wins) < 2 or wins[1] is None:
        return None
    w = wins[1][0]
    return w if w.ndim == 2 else w[k]


def _task_direct(wins, ranges, sigma, truncate, threshold):
    data, (oy, ox) = wins[0]
    out = [nanconvolve2d_gaussian(data[k], _block_weight(wins, k), sigma=sigma, truncate=truncate,
                                        threshold=threshold)[ranges[0] - oy:ranges[1] - oy, ranges[2] - ox:ranges[3] - ox]
           for k in range(data.shape[0])]
    return np.stack(out)


def _task_down(wins, ranges, level, n_in, truncate, first):
    src, (oy, ox) = wins[0]
    cells = ((ranges[0], ranges[1]), (ranges[2], ranges[3]))
    out = [_down_step(_fields(src[k], _block_weight(wins, k)) if first else src[k], oy, ox, level, n_in, cells, truncate)
           for k in range(src.shape[0])]
    return np.stack(out)


def _task_top(wins, ranges, ky, kx):
    src, (oy, ox) = wins[0]
    out = []
    for k in range(src.shape[0]):
        f = np.empty_like(src[k])
        _filter_fields(np.ascontiguousarray(src[k]), ky, kx, f)
        out.append(f[:, ranges[0] - oy:ranges[1] - oy, ranges[2] - ox:ranges[3] - ox])
    return np.stack(out)


def _task_up(wins, ranges, ncells, scale, has_w, lim, dtype):
    src, c0 = wins[0]
    return np.stack([_upsample(np.ascontiguousarray(src[k]), c0, ncells, scale, (ranges[0], ranges[2]),
                               (ranges[1] - ranges[0], ranges[3] - ranges[2]), has_w, lim, dtype)
                     for k in range(src.shape[0])])


def _cell_chunks(prev, f, n):
    """Chunks of n cells after a step of f from chunks `prev`: about prev / f, at least 64."""
    size = max(64, -(-max(prev) // f))
    return tuple(min(size, n - s) for s in range(0, n, size))


def gaussian_dask(data, weight, sigma, pixel_sizes, threshold=0.5, truncate=4.0):
    """
    2D Gaussian convolution of a dask array (y, x) or (n, y, x) with chunks of 1 along n, NaN-aware and weighted, real
    or complex: directly (nanconvolve2d_gaussian()), or decimated when the pixels are fine against the wavelength
    (decimation_plan()): the weighted values and the weights averaged in steps, filtered on the top cells, interpolated
    back bilinearly and divided, NaN where the weight sum is at most `threshold` (with a weight: under 1e-10), as
    directly. Every output block is computed from the window of input it needs, gathered from the chunks it lies in,
    so the result does not depend on the chunks, and a long wavelength costs about one pass over the data. With a
    weight, a pixel with no weight within ceil(4 sigma) is NaN directly, while the decimated filter, which reaches a
    little farther, gives it the weighted mean of its far tail. The weight is a dask array of the data's shape, a
    (y, x) one shared by every n, or None. The output has the data's chunks.

    Parameters
    ----------
    data : dask.array.Array
        (y, x) or (n, y, x) real or complex values.
    weight : dask.array.Array or None
        Weights: the data's shape, or (y, x) for every n.
    sigma : tuple of floats
        Standard deviation of the Gaussian (y, x) in pixels.
    pixel_sizes : tuple of floats
        Pixel sizes (dy, dx) in metres.
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).

    Returns
    -------
    dask.array.Array
        float32 or complex64, the data's shape and chunks.

    Examples
    --------
    >>> out = gaussian_dask(data, None, sigma=(2.0, 8.0), pixel_sizes=(8.0, 2.0))
    """
    from dask.base import tokenize
    three = data.ndim == 3
    D = data if three else data[None]
    W = weight if weight is None or weight.ndim == 2 and three else (weight if three else weight[None])
    is_complex = np.issubdtype(D.dtype, np.complexfloating)
    dtype = np.complex64 if is_complex else np.float32
    D = D.astype(dtype)
    W = None if W is None else W.astype(np.float32)
    token = tokenize(D, W, sigma, pixel_sizes, threshold, truncate)
    lead = (D.chunks[0],)
    plan = decimation_plan(sigma, pixel_sizes, truncate)
    if plan is None:
        r = [int(np.ceil(s * truncate)) for s in sigma]
        win = (lambda a, b, r=r[0]: (a - r, b + r), lambda a, b, r=r[1]: (a - r, b + r))
        out = _window_graph(f'gaussian-{token}', _task_direct, [D, W], [win, win], lead, D.chunks[1:], dtype,
                            dict(sigma=tuple(sigma), truncate=truncate, threshold=threshold))
        return out if three else out[0]
    F = 4 if is_complex else 2
    src, n_in, prev = D, D.shape[1:], D.chunks[1:]
    for i, level in enumerate(plan['levels']):
        f = level['f']
        reach = _down_reach(level, truncate)
        n_out = tuple(-(-n_in[a] // f[a]) for a in (0, 1))
        chunks = tuple(_cell_chunks(prev[a], f[a], n_out[a]) for a in (0, 1))
        win = tuple(lambda c0, c1, f=f[a], r=reach[a]: (c0 * f - r, c1 * f + r) for a in (0, 1))
        src = _window_graph(f'gaussian-step{i}-{token}', _task_down, [src, W if i == 0 else None], [win, win],
                            lead + ((F,),), chunks, np.float32,
                            dict(level=level, n_in=tuple(int(v) for v in n_in), truncate=truncate, first=i == 0))
        n_in, prev = n_out, chunks
    r = [int(np.ceil(s * truncate)) for s in plan['sigma']]
    win = (lambda a, b, r=r[0]: (a - r, b + r), lambda a, b, r=r[1]: (a - r, b + r))
    top = _window_graph(f'gaussian-top-{token}', _task_top, [src], [win], lead + ((F,),), prev, np.float32,
                        dict(ky=_kernel1d(plan['sigma'][0], truncate), kx=_kernel1d(plan['sigma'][1], truncate)))
    s = plan['scale']
    win = tuple(lambda p0, p1, s=s[a]: (int(np.floor((p0 - (s - 1) / 2) / s)), int(np.floor((p1 - 1 - (s - 1) / 2) / s)) + 2)
                for a in (0, 1))
    has_w = W is not None
    out = _window_graph(f'gaussian-up-{token}', _task_up, [top], [win], lead, D.chunks[1:], dtype,
                        dict(ncells=tuple(int(v) for v in n_in), scale=s, has_w=has_w,
                             lim=np.float32(1e-10 if has_w else threshold), dtype=dtype))
    return out if three else out[0]
