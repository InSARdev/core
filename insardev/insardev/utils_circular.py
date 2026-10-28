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
import xarray as xr

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
    θ = (arr + np.pi) % (2 * np.pi) - np.pi

    # Compute mean resultant vector
    z = np.nanmean(np.exp(1j * θ), axis=axis)
    R = np.abs(z)

    # Clip for numerical stability
    R = np.clip(R, 1e-16, 1.0)

    # Invert to get circular standard deviation
    return np.sqrt(-2 * np.log(R))

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
    x = (x + np.pi) % (2*np.pi) - np.pi
    mean = np.nanmean(np.exp(1j*x), axis=axis)
    # undefined circular mean
    if np.all(np.abs(mean) < 1e-10):
        return np.nan
    return np.arctan2(mean.imag, mean.real)
