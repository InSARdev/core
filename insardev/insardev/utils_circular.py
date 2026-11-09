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
import numba
import numpy as np
import xarray as xr

# 2 pi in three parts (Cody-Waite): k * _C1 and k * _C2 are exact for |k| < 2**23
_PI = math.pi
_TWO_PI = 6.283185307179586
_C1 = 6.283185303211212
_C2 = 3.9683743166540886e-09
_C3 = 2.068073192717642e-18
_INV_TWO_PI = 1.0 / _TWO_PI

@numba.njit(inline='always', cache=True)
def wrap_phase_scalar(x):
    """
    Wrap one value to [-pi, pi) in float64: the numba core of wrap_phase, for numba kernels.
    A value that is not finite (inf, NaN) has no phase to wrap: it gives NaN.

    >>> wrap_phase_scalar(4.0)
    -2.2831853071795867
    """
    y = np.float64(x)
    if not math.isfinite(y):
        return np.nan
    k = math.floor((y + _PI) * _INV_TWO_PI)
    r = ((y - k * _C1) - k * _C2) - k * _C3
    # the reduction can land one rounding step outside the range: fold it back
    if r >= _PI:
        r -= _TWO_PI
    elif r < -_PI:
        r += _TWO_PI
    return r

@numba.vectorize(['float32(float32)', 'float64(float64)'], nopython=True, cache=True)
def wrap_phase(x):
    """
    Wrap phase to [-pi, pi) as a numpy ufunc: float64 inside, the dtype of the input is kept.
    Lazy on dask arrays and dask-backed xarray objects.

    >>> wrap_phase(4.0)
    np.float64(-2.2831853071795867)
    """
    return wrap_phase_scalar(x)

# numba's cpu target returns a DUFunc (kept alive as _dufunc): the numpy ufunc of it is the public wrap_phase, named as the
# module attribute, so it pickles by reference (dask workers unpickle it)
_dufunc = wrap_phase
wrap_phase = _dufunc.ufunc

def _std_of_R(R):
    """
    The circular standard deviation sqrt(-2 ln R) of the mean resultant length R (numpy, dask or xarray).
    R is clipped to [1e-16, 1] first: R = 0 (opposed angles) gives a finite value, not inf, and R a rounding step
    above 1 (identical angles) gives 0, not NaN. Shared by circular_std and BatchWrap.std().
    """
    R = R.clip(min=1e-16, max=1.0)
    return np.sqrt(-2 * np.log(R))

def circular_std(x: xr.DataArray | np.ndarray, axis=None) -> float:
    """
    Compute the circular standard deviation of angles in x (radians),
    returning a nonnegative real number in radians.

    Based on R = |⟨e^{iθ}⟩| and sigma = √(-2 ln R).  
    See: https://en.wikipedia.org/wiki/Circular_standard_deviation

    Parameters
    ----------
    x : xarray.DataArray or numpy.ndarray
        Input array of angles, in radians.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the circular standard deviation.
        If None, compute over the flattened array.

    Returns
    -------
    float or ndarray
        Circular standard deviation. If `axis` is None, returns a scalar;
        otherwise returns an array with `axis` removed.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils import utils
    >>> angles = np.array([0, np.pi/4, np.pi/2])
    >>> utils.circular_std(angles)
    0.6591489993143684

    >>> # along a specific axis
    >>> angles2d = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]])
    >>> utils.circular_std(angles2d, axis=0)
    array([8.58386411, 8.58386411])
    """
    import numpy as np

    # Ensure we work with a numpy array
    arr = x.values if hasattr(x, "values") else x
    # Wrap into [-π, π)
    θ = wrap_phase(arr)

    # Compute mean resultant vector
    z = np.nanmean(np.exp(1j * θ), axis=axis)
    R = np.abs(z)

    # Invert to get circular standard deviation (R clipped, see _std_of_R)
    return _std_of_R(R)

def circular_mean(x: xr.DataArray|np.ndarray, axis=None) -> float:
    """
    Calculate the circular mean of an xarray.DataArray or numpy.ndarray.

    Parameters
    ----------
    x : xarray.DataArray or numpy.ndarray
        Input array.
    axis : int, optional
        Axis along which the circular mean is computed.

    Returns
    -------
    float
        Circular mean of the input array.

    Examples
    --------
    from insardev.utils import utils
    data.reduce(circular_mean, dim='x')
    """
    import numpy as np
    # flawrap to [-π,π)
    x = wrap_phase(x)
    mean = np.nanmean(np.exp(1j*x), axis=axis)
    # undefined circular mean
    if np.all(np.abs(mean) < 1e-10):
        return np.nan
    return np.arctan2(mean.imag, mean.real)

def resultant_length(obj, dim=None, carry=(), **kwargs):
    """
    The mean resultant length R = |E[e^(iθ)]| of wrapped phase (an xarray object), clipped to [1e-16, 1]: the input of the
    circular std (_std_of_R) and of the var (1 - R) of BatchWrap.std() and var(). Reduced over `dim` in the complex domain,
    or a coarsen() window (already the complex phasors, see coarsen_phasors()) by its own mean(). The metadata `carry`
    names are left out. The clip keeps the var in [0, 1]: a rounding step above 1 (identical angles) gives 0.
    """
    import inspect
    sig = inspect.signature(obj.mean)
    if 'dim' in sig.parameters:
        complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
        R = complex_obj.mean(dim=dim, **kwargs)
    else:
        R = obj.mean(**kwargs)
        if carry:
            R = R[[v for v in R.data_vars if v not in carry]]
    return xr.ufuncs.abs(R).clip(min=1e-16, max=1.0)

def circular_aggregate(obj, name: str, dim=None, carry=(), **kwargs):
    """
    Aggregate wrapped phase (an xarray object) by `name` in the complex domain, the angle of the result back to phase.

    The phase reductions only (mean, sum, median, min, max): the angle is in (-pi, pi] already, so it is not wrapped again.
    The spread is not phase: see resultant_length() and BatchWrap.std() / var(). `carry` names the variables that pass
    through unchanged (the metadata), so the reductions leave them out. Returns float32 (BatchWrap._agg).
    """
    import inspect
    # get the aggregation function
    fn = getattr(obj, name)
    sig = inspect.signature(fn)

    # perform aggregation in complex domain
    if 'dim' in sig.parameters:
        # intfs.mean('pair').isel(0)
        complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
        fn_complex = getattr(complex_obj, name)
        agg_result = fn_complex(dim=dim, **kwargs)
        # convert back to wrapped phase
        agg_result = xr.ufuncs.angle(agg_result)
    else:
        # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
        # already in complex domain, see coarsen_phasors()
        agg_result = fn(**kwargs)
        if carry:
            agg_result = agg_result[[v for v in agg_result.data_vars if v not in carry]]
        agg_result = xr.ufuncs.angle(agg_result)

    return agg_result.astype('float32')

def coarsen_phasors(grid, meta: dict, window: dict):
    """
    The complex phasors exp(1j*theta) of one burst's grids, with the metadata `meta` (name -> DataArray) carried beside
    them, each dimension aligned to the coarse grid of `window` (utils_xarray.coarsen_start). The coarsen itself is the caller's.
    """
    from . import utils_xarray
    # convert to complex numbers for proper circular statistics
    ds2 = xr.ufuncs.exp(1j * grid.astype('float32'))
    if meta:
        ds2 = ds2.assign(meta)
    # align each dimension to the coarse grid
    for dim, factor in window.items():
        start = utils_xarray.coarsen_start(ds2, dim, factor)
        if start is not None:
            ds2 = ds2.isel({dim: slice(start, None)})
    return ds2

def dissolve_pol(current_vals, others_vals: list, wrap: bool, extend: bool, weight, dtype):
    """
    The weighted mean of one polarization over the overlapping bursts on their common grid (BatchCore._dissolve_pol_for_dask):
    the circular mean of the phase when wrap, the linear mean otherwise. `current_vals` and `others_vals` are numpy arrays on
    that grid, NaN where a burst has no data; NaN stays NaN where the current burst has none, unless extend.
    """
    import warnings

    n_others = len(others_vals)

    if weight is None:
        w_current, w_other = 1.0, 1.0
    else:
        w_current = weight
        w_other = (1.0 - weight) / n_others if n_others > 0 else 0.0

    current_valid = np.isfinite(current_vals)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        if wrap:
            weighted_sum = np.where(current_valid, np.exp(1j * current_vals).astype(np.complex64) * w_current, np.complex64(0))
            weight_sum = np.where(current_valid, w_current, 0.0)
            for vals in others_vals:
                valid = np.isfinite(vals)
                weighted_sum += np.where(valid, np.exp(1j * vals).astype(np.complex64) * w_other, np.complex64(0))
                weight_sum += np.where(valid, w_other, 0.0)
            valid_weights = weight_sum > 0
            normalized = np.divide(weighted_sum, weight_sum, out=np.zeros_like(weighted_sum), where=valid_weights)
            out = np.where(valid_weights, np.arctan2(normalized.imag, normalized.real), np.nan)
        else:
            weighted_sum = np.where(current_valid, current_vals * w_current, 0.0)
            weight_sum = np.where(current_valid, w_current, 0.0)
            for vals in others_vals:
                valid = np.isfinite(vals)
                weighted_sum += np.where(valid, vals * w_other, 0.0)
                weight_sum += np.where(valid, w_other, 0.0)
            out = np.divide(weighted_sum, weight_sum, out=np.full_like(weighted_sum, np.nan), where=weight_sum > 0)

        if not extend:
            out = np.where(current_valid, out, np.nan)

    return out.astype(dtype)
