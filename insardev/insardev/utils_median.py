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
import numba as _numba


@_numba.njit(nogil=True, cache=True, inline='always')
def _median_select(buf, k, n):
    """The k-th smallest of buf[:n], in place. Quickselect, no allocation."""
    lo, hi = 0, n - 1
    while True:
        if hi <= lo + 1:
            if hi == lo + 1 and buf[hi] < buf[lo]:
                buf[lo], buf[hi] = buf[hi], buf[lo]
            return buf[k]
        mid = (lo + hi) >> 1
        buf[mid], buf[lo + 1] = buf[lo + 1], buf[mid]
        if buf[lo] > buf[hi]:
            buf[lo], buf[hi] = buf[hi], buf[lo]
        if buf[lo + 1] > buf[hi]:
            buf[lo + 1], buf[hi] = buf[hi], buf[lo + 1]
        if buf[lo] > buf[lo + 1]:
            buf[lo], buf[lo + 1] = buf[lo + 1], buf[lo]
        i, j, piv = lo + 1, hi, buf[lo + 1]
        while True:
            i += 1
            while buf[i] < piv:
                i += 1
            j -= 1
            while buf[j] > piv:
                j -= 1
            if j < i:
                break
            buf[i], buf[j] = buf[j], buf[i]
        buf[lo + 1], buf[j] = buf[j], buf[lo + 1]
        if j >= k:
            hi = j - 1
        if j <= k:
            lo = i


@_numba.njit(nogil=True, cache=True)
def _median_plane(a, wy, wx, out):
    """One (y, x) plane: the NaN-SKIPPING median of a centred box per pixel.

    Serial on purpose, as every numba kernel here is: dask parallelises across
    blocks, and a parallel=True kernel called from a multi-threaded dask worker
    trips numba's workqueue layer and takes the worker down.

    NaN is skipped, never propagated and never treated as a value -- a caller
    who prefilters the raster leaves holes everywhere, and a box that straddles
    one must still answer from the samples it has. A box with nothing finite in
    it returns NaN, which is an absence rather than a number. The window is
    truncated at the raster edge for the same reason.

    The samples are gathered into one reusable buffer and the median is taken
    by quickselect, so nothing is allocated per pixel and nothing is sorted
    that does not have to be.
    """
    ny, nx = a.shape
    hy, hx = wy // 2, wx // 2
    buf = np.empty(wy * wx, dtype=a.dtype)
    for y in range(ny):
        y0 = y - hy if y - hy > 0 else 0
        y1 = y + hy + 1 if y + hy + 1 < ny else ny
        for x in range(nx):
            x0 = x - hx if x - hx > 0 else 0
            x1 = x + hx + 1 if x + hx + 1 < nx else nx
            k = 0
            for yy in range(y0, y1):
                row = a[yy]
                for xx in range(x0, x1):
                    v = row[xx]
                    if v == v:                      # NaN fails this, and is skipped
                        buf[k] = v
                        k += 1
            if k == 0:
                out[y, x] = np.nan
            elif k & 1:
                out[y, x] = _median_select(buf, k >> 1, k)
            else:
                upper = _median_select(buf, k >> 1, k)
                lower = buf[0]
                for t in range(1, k >> 1):
                    if buf[t] > lower:
                        lower = buf[t]
                out[y, x] = 0.5 * (lower + upper)


def median2d(block, window_y, window_x):
    """A centred box MEDIAN over (y, x), NaN-skipping, on one dask block.

    Module level so dask ships it by name.

    THE MEDIAN OF A COMPLEX SAMPLE IS TAKEN PER COMPONENT, real and imaginary
    separately. A set of phasors has no total order, so there is no single
    sample to pick out; numpy's own complex median picks one by lexicographic
    order, which depends on where the phase happens to sit relative to the real
    axis and means nothing for a phasor. The per-component median keeps the
    estimator robust in the way a caller asks for: one wild pixel in the box
    cannot move the answer, which is exactly what the mean fails at.
    """
    import numpy as np
    squeeze = block.ndim == 2
    B = block[np.newaxis, ...] if squeeze else block
    out = np.empty_like(B)
    for i in range(B.shape[0]):
        plane = np.ascontiguousarray(B[i])
        if np.iscomplexobj(B):
            re = np.empty(plane.shape, dtype=plane.real.dtype)
            im = np.empty(plane.shape, dtype=plane.real.dtype)
            _median_plane(np.ascontiguousarray(plane.real), window_y, window_x, re)
            _median_plane(np.ascontiguousarray(plane.imag), window_y, window_x, im)
            out[i] = re + 1j * im
        else:
            res = np.empty(plane.shape, dtype=plane.dtype)
            _median_plane(plane, window_y, window_x, res)
            out[i] = res
    return out[0] if squeeze else out


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _a = np.array([[0.0, np.nan, 1.0],
                   [2.0, 3.0, np.nan],
                   [4.0, 5.0, 6.0]], dtype=np.float32)
    # one call per dtype the dask task can hand it, and _median_select compiles
    # with them because it is inlined: float32 from a float32 raster or the
    # components of complex64, float64 from float64 or complex128
    _median_plane(_a, 3, 3, np.empty((3, 3), np.float32))
    _median_plane(_a.astype(np.float64), 3, 1, np.empty((3, 3), np.float64))


_warmup_numba_cache()
