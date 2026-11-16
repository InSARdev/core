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
"""The rotation of a complex block by a phase in one numba pass: `d * exp(-1j * phase)` without the temporaries of numpy's expression.

`exp(-1j * phase)` is cos(phase) - 1j * sin(phase), and the kernel evaluates it in the phase's own precision:
- a float32 phase (detrend's): cos and sin are taken in float64 and rounded to float32 once (the library's complex64 product rounds
  the same way), and the product is formed in the data's precision on the interleaved (real, imag) floats. numpy's
  `d * np.exp(np.complex64(-1j) * phase)` evaluates the exponential in complex64, so the two differ by rounding only (at most one
  float32 step of |d| per component, measured by tests/core/core/numerics/rotate_pairs);
- a float64 phase (align's polyval, the offsets of Stack.align): cos, sin and the product stay in float64 and only the result is cast
  to the data's complex dtype, so a phase of hundreds of rad is not rounded to float32 first.
The result has the data's complex dtype (complex64 at least).
"""
import math

import numpy as np
import numba as nb
import xarray as xr


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _rotate_pairs(df, p, of, single):
    """df, of: the interleaved (real, imag) floats of the data and of the result (one float dtype); p: flat phases, one per complex
    value. `single` True: cos and sin are rounded to float32 once and the product is formed in the data's precision (the float32
    phase, numpy's complex64 numbers); False: cos, sin and the product stay in float64 and only the store casts to the result's float
    dtype (the float64 phase). One pass, no temporaries.

    >>> import numpy as np
    >>> df = np.array([1, 0, 0, 1], np.float32)
    >>> of = np.empty(4, np.float32)
    >>> _rotate_pairs(df, np.zeros(2, np.float32), of, True)
    >>> of.tolist()
    [1.0, 0.0, 0.0, 1.0]
    >>> _rotate_pairs(df, np.zeros(2), of, False)
    >>> of.tolist()
    [1.0, 0.0, 0.0, 1.0]
    """
    if single:
        for k in range(p.shape[0]):
            ph = np.float64(p[k])
            c = np.float32(math.cos(ph))
            s = np.float32(math.sin(ph))
            re = df[2 * k]
            im = df[2 * k + 1]
            of[2 * k] = re * c + im * s
            of[2 * k + 1] = im * c - re * s
    else:
        for k in range(p.shape[0]):
            ph = np.float64(p[k])
            c64 = math.cos(ph)
            s64 = math.sin(ph)
            re64 = np.float64(df[2 * k])
            im64 = np.float64(df[2 * k + 1])
            of[2 * k] = re64 * c64 + im64 * s64
            of[2 * k + 1] = im64 * c64 - re64 * s64


def rotate_block(d, p):
    """`d * exp(-1j * p)` for a complex block `d` and a real phase `p` (numpy broadcasting of the two shapes). The result has the
    complex dtype of `d` (complex64 at least). A float64 phase is rotated in float64 (the result cast once), any other phase dtype
    as float32.

    >>> import numpy as np
    >>> d = np.array([1 + 0j, 0 + 1j], np.complex64)
    >>> r = rotate_block(d, np.array([0.0, np.pi / 2], np.float32))
    >>> r.dtype.name, (np.round(r.real, 6) + 0.0).tolist(), (np.round(r.imag, 6) + 0.0).tolist()
    ('complex64', [1.0, 1.0], [0.0, 0.0])
    >>> r = rotate_block(d, np.array([0.0, np.pi / 2]))
    >>> r.dtype.name, (np.round(r.real, 6) + 0.0).tolist(), (np.round(r.imag, 6) + 0.0).tolist()
    ('complex64', [1.0, 1.0], [0.0, 0.0])
    """
    p = np.asarray(p)
    pdtype = np.float64 if p.dtype == np.float64 else np.float32
    shape = np.broadcast_shapes(d.shape, p.shape)
    dtype = np.result_type(d.dtype, np.complex64)
    rdtype = np.zeros(0, dtype).real.dtype
    out = np.empty(shape, dtype)
    dc = np.ascontiguousarray(np.broadcast_to(d, shape), dtype).reshape(-1)
    pc = np.ascontiguousarray(np.broadcast_to(np.asarray(p, pdtype), shape)).reshape(-1)
    _rotate_pairs(dc.view(rdtype), pc, out.reshape(-1).view(rdtype), pdtype is np.float32)
    return out


def rotate_grid(v, p):
    """`v * exp(-1j * p)` for a complex grid DataArray `v` and a phase DataArray `p`, lazy for dask, with the attributes xarray's
    arithmetic `v * p` gives: the variable's attributes are those of the arithmetic of two 0-d Variables carrying the operands'
    attributes (xarray's own rule, under the keep_attrs option), and the coordinates keep the first operand's attributes, as the
    arithmetic does.

    >>> import numpy as np
    >>> import xarray as xr
    >>> v = xr.DataArray(np.ones((1, 2), np.complex64), dims=('y', 'x'), attrs={'units': 'mag', 'a': 1})
    >>> p = xr.DataArray(np.zeros((1, 2), np.float32), dims=('y', 'x'), attrs={'units': 'rad', 'b': 2})
    >>> g = rotate_grid(v, p)
    >>> g.dtype.name, g.attrs == (v * p).attrs
    ('complex64', True)
    """
    out = xr.apply_ufunc(rotate_block, v, p, dask='parallelized', keep_attrs='override',
                         join=xr.get_options()['arithmetic_join'],
                         output_dtypes=[np.result_type(v.dtype, np.complex64)])
    out.attrs = (xr.Variable((), 0, v.attrs) * xr.Variable((), 0, p.attrs)).attrs
    return out
