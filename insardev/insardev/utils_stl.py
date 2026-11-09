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
Static utility functions for Seasonal-Trend decomposition using LOESS (STL).

These functions contain the core algorithms for STL decomposition
applied to InSAR time series data.
"""
import numpy as np


def stl1d(ts, dt, dt_periodic, periods=52, robust=False):
    """
    Perform Seasonal-Trend decomposition using LOESS (STL) on the input time series data.

    The function performs the following steps:
    1. Check for NaN values in the input time series and return arrays filled with NaNs if found.
    2. Create an interpolation function using the input time series and corresponding time values.
    3. Interpolate the time series data for the periodic time values.
    4. Perform STL decomposition using the interpolated time series data.
    5. Return the trend, seasonal, and residual components of the decomposed time series.

    Parameters
    ----------
    ts : numpy.ndarray
        Input time series data.
    dt : numpy.ndarray
        Corresponding time values for the input time series data.
    dt_periodic : numpy.ndarray
        Periodic time values for interpolation.
    periods : int
        Number of periods for seasonal decomposition.
    robust : bool, optional
        Whether to use a robust fitting procedure for the STL decomposition (default is False).

    Returns
    -------
    numpy.ndarray
        Trend component of the decomposed time series.
    numpy.ndarray
        Seasonal component of the decomposed time series.
    numpy.ndarray
        Residual component of the decomposed time series.
    """
    from scipy.interpolate import interp1d
    from statsmodels.tsa.seasonal import STL

    # Check for NaNs in the input time series; this check is faster than letting STL handle NaNs
    if np.any(np.isnan(ts)):
        nodata = np.nan * np.zeros(dt_periodic.size)
        return nodata, nodata, nodata

    # Create an interpolation function for the input time series and time values
    interp_func = interp1d(dt, ts, kind='nearest', fill_value='extrapolate', assume_sorted=True)
    # Interpolate the time series data for the periodic time values
    ts = interp_func(dt_periodic)

    # Perform STL decomposition on the interpolated time series data
    stl = STL(ts, period=periods, robust=robust)
    res = stl.fit()

    return res.trend, res.seasonal, res.resid


def stl_block(data_block, dt, dt_periodic, periods=52, robust=False, *, budget_bytes):
    """
    Decompose each pixel of a (date, y, x) block by stl1d, tile by tile.

    A tile holds as many pixels as fit in budget_bytes (at least 256 pixels).

    Parameters
    ----------
    data_block : numpy.ndarray
        Input block with dimensions (date, y, x).
    dt : numpy.ndarray
        Original dates as int64 nanoseconds (see stl_periodic).
    dt_periodic : numpy.ndarray
        Periodic dates as int64 nanoseconds (see stl_periodic).
    periods : int, optional
        Number of periods for seasonal decomposition (default 52).
    robust : bool, optional
        Whether to use a robust fitting procedure (default False).
    budget_bytes : int
        Memory budget of one tile in bytes.

    Returns
    -------
    numpy.ndarray
        float32 array with dimensions (component, date, y, x): trend, seasonal and residual.
    """
    import math
    n_dates_out = len(dt_periodic)
    ny, nx = data_block.shape[1], data_block.shape[2]
    n_dates_in_local = data_block.shape[0]
    result = np.empty((3, n_dates_out, ny, nx), dtype=np.float32)
    vec_stl = np.vectorize(
        lambda ts: stl1d(ts, dt, dt_periodic, periods, robust),
        signature='(n)->(m),(m),(m)'
    )
    per_pixel_bytes = (n_dates_in_local + 3 * n_dates_out) * 4
    max_sub_pixels = max(256, budget_bytes // max(1, per_pixel_bytes))
    sub_side = int(math.sqrt(max_sub_pixels))
    sub_h = min(sub_side, ny)
    sub_w = min(sub_side, nx)
    for ty0 in range(0, ny, sub_h):
        ty1 = min(ty0 + sub_h, ny)
        for tx0 in range(0, nx, sub_w):
            tx1 = min(tx0 + sub_w, nx)
            tile = data_block[:, ty0:ty1, tx0:tx1]
            tile_t = tile.transpose(1, 2, 0)
            del tile
            block = np.asarray(vec_stl(tile_t))
            del tile_t
            result[:, :, ty0:ty1, tx0:tx1] = block.transpose(0, 3, 1, 2)
            del block
    del vec_stl
    return result


def stl_periodic(dates, freq='W'):
    """
    Compute periodic time values for STL decomposition.

    Parameters
    ----------
    dates : array-like
        Input dates for the time series.
    freq : str, optional
        Frequency string for unifying date intervals (default is 'W' for weekly).

    Returns
    -------
    dt : numpy.ndarray
        Original dates as int64 nanoseconds.
    dt_periodic : numpy.ndarray
        Periodic time values as int64 nanoseconds.
    """
    import pandas as pd
    import xarray as xr

    # convert coordinate to valid dates
    dates = pd.to_datetime(dates)
    # original dates
    dt = dates.astype(np.int64)
    # Unify date intervals; using weekly intervals should be suitable for a mix of 6 and 12 days intervals
    dates_weekly = pd.date_range(dates[0], dates[-1], freq=freq)
    dt_weekly = xr.DataArray(dates_weekly, dims=['date'])
    dt_periodic = dt_weekly.astype(np.int64)
    return (dt, dt_periodic.values)
