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
import math

import numba as _numba
import numpy as np
import scipy.fft
from llvmlite import ir as _llir
from numba.extending import intrinsic as _intrinsic

_F0 = np.float32(0.0)


@_intrinsic
def _fma32(typingctx, a, b, c):
    """float32 fused multiply-add, one rounding (llvm.fma)."""
    f32 = _numba.types.float32
    sig = f32(f32, f32, f32)

    def codegen(context, builder, signature, args):
        t = _llir.FloatType()
        fn = builder.module.declare_intrinsic('llvm.fma', [t], _llir.FunctionType(t, [t, t, t]))
        return builder.call(fn, args)
    return sig, codegen


@_numba.njit(nogil=True, cache=True, inline='always')
def _cabs32(re, im):
    """The complex64 absolute value as numpy rounds it: the larger part times the square root of 1 + r * r (one fma),
    r the smaller part over the larger."""
    re = abs(re)
    im = abs(im)
    lg = max(re, im)
    sm = min(im, re)
    if lg == _F0 or sm == np.float32(np.inf):
        ratio = _F0
    else:
        ratio = sm / lg
    return np.float32(np.sqrt(_fma32(ratio, ratio, np.float32(1.0))) * lg)


@_numba.njit(nogil=True, cache=True, inline='always')
def _einsum_dot(w, c, n):
    """The sum of w * c over n float32 values in np.einsum's order: four lanes, each group of 16 values chained by fma
    from its last four to its first, a zero-padded tail, the lanes summed in pairs."""
    a0 = _F0
    a1 = _F0
    a2 = _F0
    a3 = _F0
    g = (n // 16) * 16
    for e0 in range(0, g, 16):
        for u in range(3, -1, -1):
            e = e0 + 4 * u
            a0 = _fma32(w[e], c[e], a0)
            a1 = _fma32(w[e + 1], c[e + 1], a1)
            a2 = _fma32(w[e + 2], c[e + 2], a2)
            a3 = _fma32(w[e + 3], c[e + 3], a3)
    for e in range(g, n, 4):
        m = n - e
        a0 = _fma32(w[e], c[e], a0)
        if m > 1:
            a1 = _fma32(w[e + 1], c[e + 1], a1)
        if m > 2:
            a2 = _fma32(w[e + 2], c[e + 2], a2)
        if m > 3:
            a3 = _fma32(w[e + 3], c[e + 3], a3)
    return _F0 + ((a0 + a1) + (a2 + a3))


@_numba.njit(nogil=True, cache=True)
def _strip(phase, corr, sy, sx, gny, gnx, py, px, step_y, step_x, row0, row1, n_px, wgt, wflat, wgt_sum, min_valid,
           out, count):
    """Patch rows row0:row1 of the grid into out and count: the patches (an invalid pixel and the padding 0), alpha = 1
    - sum(wgt corr) / sum(wgt), the in-place FFT of the strip, the spectrum times its modulus over the patch maximum to
    the power alpha, the in-place inverse FFT, the triangular weight, folded in patch order; a patch under min_valid
    valid pixels adds nothing."""
    nr = row1 - row0
    n = nr * n_px
    size = py * px
    patches = np.empty((n, py, px), np.complex64)
    alpha = np.empty(n, np.float32)
    valid = np.empty(n, np.bool_)
    cbuf = np.empty(size, np.float32)
    lo = np.float32(1e-10)
    for i in range(nr):
        for j in range(n_px):
            k = i * n_px + j
            ys = (row0 + i) * step_y
            xs = j * step_x
            nv = 0
            for a in range(py):
                y = ys + a
                for b in range(px):
                    x = xs + b
                    v = np.complex64(0)
                    c = _F0
                    if y < gny and x < gnx:
                        pv = phase[sy + y, sx + x]
                        if pv.real == pv.real and pv.imag == pv.imag:
                            v = pv
                            nv += 1
                        cv = corr[sy + y, sx + x]
                        if cv == cv:
                            c = cv
                    patches[k, a, b] = v
                    cbuf[a * px + b] = c
            alpha[k] = (_einsum_dot(wflat, cbuf, size) / (-wgt_sum)) + np.float32(1.0)
            valid[k] = nv >= min_valid
    spec = scipy.fft.fft2(patches, axes=(1, 2), overwrite_x=True, workers=1)
    for k in range(n):
        al = alpha[k]
        # (|S| / max |S|)^alpha: the weight does not depend on the patch's amplitude level, so the overlapping patches
        # add as the input weighs them, not by their level to the power 1 + alpha; the maximum from the squared moduli,
        # one square root a patch
        msq = lo * lo
        for a in range(py):
            for b in range(px):
                f = spec[k, a, b]
                q = f.real * f.real + f.imag * f.imag
                if q > msq:
                    msq = q
        inv = np.float32(1.0) / np.float32(np.sqrt(msq))
        for a in range(py):
            for b in range(px):
                f = spec[k, a, b]
                m = _cabs32(f.real, f.imag)
                if m < lo:
                    m = lo
                w = np.float32(math.pow(m * inv, al))
                spec[k, a, b] = np.complex64(complex(f.real * w, f.imag * w))
    filtered = scipy.fft.ifft2(spec, axes=(1, 2), overwrite_x=True, workers=1)
    for i in range(nr):
        for j in range(n_px):
            k = i * n_px + j
            if not valid[k]:
                continue
            ys = sy + (row0 + i) * step_y
            xs = sx + j * step_x
            for a in range(py):
                for b in range(px):
                    w = wgt[a, b]
                    g = filtered[k, a, b]
                    out[ys + a, xs + b] += np.complex64(complex(g.real * w, g.imag * w))
                    count[ys + a, xs + b] += w


@_numba.njit(nogil=True, cache=True)
def _finish(phase, out, count, ny, nx):
    """out / count in place (times the float32 reciprocal, as numpy divides a complex value by a real one), NaN where
    the input is NaN or no valid patch reaches."""
    nanc = np.complex64(complex(np.nan, np.nan))
    tiny = np.float32(1e-6)
    lo = np.float32(1e-10)
    one = np.float32(1.0)
    for y in range(ny):
        for x in range(nx):
            c = count[y, x]
            pv = phase[y, x]
            if c < tiny or pv.real != pv.real or pv.imag != pv.imag:
                out[y, x] = nanc
            else:
                scl = one / max(c, lo)
                o = out[y, x]
                out[y, x] = np.complex64(complex(o.real * scl, o.imag * scl))


def goldstein(phase_np, corr_np, psize_y, psize_x, threshold=0.5, skip=(0, 0)):
    """
    Goldstein adaptive filter, in numba (scipy.fft through rocket-fft).

    Each patch's spectrum S is weighted by (|S| / max |S|)^alpha, alpha = 1 - the patch's mean correlation: the result
    scales with the input's amplitude and its phase does not depend on it.

    The patches are filtered in strips of patch rows that fit half of the Dask chunk size
    (dask.config['array.chunk-size']), one GIL-free call a strip, and accumulated in the same order as one batch of all
    patches: the same patches, the same fold order, so the strip size can only change the rounding of a patch's FFT
    (about 1 ulp for some patch sizes). Requires complex64/float32 input.

    Parameters
    ----------
    phase_np : np.ndarray
        2D complex64 numpy array of phase data.
    corr_np : np.ndarray
        2D float32 numpy array of correlation values.
    psize_y : int
        Patch size in y dimension.
    psize_x : int
        Patch size in x dimension.
    threshold : float
        Minimum fraction of valid (non-NaN) pixels required to process a patch.
        Default 0.5 means at least 50% of pixels must be valid.
    skip : tuple of int
        (rows, columns) at the start of the block that no patch covers: the patch grid starts at that row and
        column of the block, and those pixels are NaN. Used for a block of a larger array, whose patch grid belongs
        to the whole array (see Batch._apply_goldstein_2d_for_dask). Default (0, 0): the grid starts at the block.

    Returns
    -------
    np.ndarray
        Filtered complex64 array with same shape as input.

    Examples
    --------
    >>> out = goldstein(phase, corr, 32, 32)
    """
    from .utils_dask import get_dask_chunk_size_mb

    # Enforce input types
    assert phase_np.dtype == np.complex64, f"phase must be complex64, got {phase_np.dtype}"
    assert corr_np.dtype == np.float32, f"corr must be float32, got {corr_np.dtype}"
    # the kernels index both blocks alike and check no bounds
    if corr_np.shape != phase_np.shape:
        raise ValueError(f'goldstein: corr block {corr_np.shape} differs from the phase block {phase_np.shape}')

    ny, nx = phase_np.shape
    # the patch grid starts at row sy and column sx of the block; the rows and columns before them stay uncovered
    sy, sx = skip
    gny, gnx = ny - sy, nx - sx
    step_y, step_x = psize_y // 2, psize_x // 2
    # The padded grid covers every pixel by at least one patch; the strips pad their own rows
    padded_ny, padded_nx = gny + (psize_y - step_y), gnx + (psize_x - step_x)

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                    [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt_sum = wgt.sum()

    # Threshold for valid patches
    min_valid_pixels = int(psize_y * psize_x * threshold)

    # Patches at step intervals: n_patches_y rows of n_patches_x patches
    n_patches_y = (padded_ny - psize_y) // step_y + 1
    n_patches_x = (padded_nx - psize_x) // step_x + 1

    # Patch rows per strip: about 40 bytes per patch pixel within half of the Dask chunk size
    budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
    rows_per_strip = max(1, budget // (n_patches_x * psize_y * psize_x * 40))

    out = np.zeros((sy + padded_ny, sx + padded_nx), dtype=np.complex64)
    count = np.zeros((sy + padded_ny, sx + padded_nx), dtype=np.float32)
    wflat = np.ascontiguousarray(wgt.reshape(-1))
    for row0 in range(0, n_patches_y, rows_per_strip):
        _strip(phase_np, corr_np, sy, sx, gny, gnx, psize_y, psize_x, step_y, step_x, row0,
               min(row0 + rows_per_strip, n_patches_y), n_patches_x, wgt, wflat, np.float32(wgt_sum),
               min_valid_pixels, out, count)
    # normalized by the accumulated weights in place; NaN where the input is NaN or no valid patch reaches
    _finish(phase_np, out, count, ny, nx)
    del count
    return out[:ny, :nx]
