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
    dt_periodic : xarray.DataArray
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
    return (dt, dt_periodic)
