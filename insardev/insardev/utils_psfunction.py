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


@_numba.njit(nogil=True, cache=True)
def _date_sums(chunk, sums, counts):
    """sums[t] += |z|^2 over the finite pixels of date t of a (height, width, n) chunk, counts[t] += their number."""
    H, W, n = chunk.shape
    for y in range(H):
        for x in range(W):
            for t in range(n):
                z = chunk[y, x, t]
                v = np.float64(z.real) * z.real + np.float64(z.imag) * z.imag
                if np.isfinite(v):
                    sums[t] += v
                    counts[t] += 1.0


@_numba.njit(nogil=True, cache=True)
def _welford(chunk, factors, count, mean, m2):
    """Fold the dates of a (height, width, n) chunk whose factor is finite into the per-pixel running count, mean and M2
    of factor * |z|^2."""
    H, W, n = chunk.shape
    for y in range(H):
        for x in range(W):
            c = count[y, x]
            m = mean[y, x]
            s = m2[y, x]
            for t in range(n):
                f = factors[t]
                if not np.isfinite(f):
                    continue
                z = chunk[y, x, t]
                v = (np.float64(z.real) * z.real + np.float64(z.imag) * z.imag) * f
                if np.isfinite(v):
                    c += 1.0
                    d = v - m
                    m += d / c
                    s += d * (v - m)
            count[y, x] = c
            mean[y, x] = m
            m2[y, x] = s


@_numba.njit(nogil=True, cache=True)
def _finish(count, mean, m2, out):
    """psfunction = mean / (2 std), std with n - 1; NaN where no value or the ratio is not finite."""
    H, W = count.shape
    for y in range(H):
        for x in range(W):
            c = count[y, x]
            v = np.nan
            if c > 0:
                v = mean[y, x] / (2.0 * np.sqrt(m2[y, x] / max(c - 1.0, 1.0)))
            out[y, x] = np.float32(v) if np.isfinite(v) else np.float32(np.nan)


def psfunction_block(chunks):
    """
    The PS function of a spatial block from its per-date SLC chunks: mean / (2 std) of the normalised intensity over
    the dates.

    Two passes, each chunk read once per pass: the mean intensity |z|^2 of every date over the block's finite pixels,
    which scales every date to the mean of all dates (a date with no finite pixel is skipped), then per pixel the
    running mean and variance (Welford, n - 1) of the scaled intensity. Float64 inside, GIL-free kernels, no copy of a
    date. NaN where the pixel has no value or the ratio is not finite.

    Parameters
    ----------
    chunks : list of np.ndarray
        Complex SLC chunks of the block, each (height, width, n) with the dates on the last axis.

    Returns
    -------
    np.ndarray
        (height, width) float32.

    Examples
    --------
    >>> psf = psfunction_block([slc])
    """
    H, W = chunks[0].shape[:2]
    sums, counts = [], []
    for c in chunks:
        s, q = np.zeros(c.shape[-1]), np.zeros(c.shape[-1])
        _date_sums(c, s, q)
        sums.append(s)
        counts.append(q)
    sums, counts = np.concatenate(sums), np.concatenate(counts)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
        factors = (np.nanmean(means) if np.isfinite(means).any() else np.nan) / means
    count, mean, m2 = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    i = 0
    for c in chunks:
        k = c.shape[-1]
        _welford(c, np.ascontiguousarray(factors[i:i + k]), count, mean, m2)
        i += k
    out = np.empty((H, W), np.float32)
    _finish(count, mean, m2, out)
    return out
