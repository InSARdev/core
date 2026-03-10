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
from .Stack_sbas import Stack_sbas
from .Batch import Batch
from . import utils_stl

class Stack_stl(Stack_sbas):

    def stl(self, data, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL) on Batch data.

        Decomposes time series into trend, seasonal, and residual components.
        The input Batch must have a 'date' dimension.

        Parameters
        ----------
        data : Batch
            Input Batch with 'date' dimension containing time series data.
        freq : str, optional
            Frequency string for resampling (default 'W' for weekly).
            Examples: '1W' for 1 week, '2W' for 2 weeks, '10d' for 10 days.
        periods : int, optional
            Number of periods for seasonal decomposition (default 52 for weekly data = 1 year).
        robust : bool, optional
            Whether to use robust fitting (slower but handles outliers better). Default False.

        Returns
        -------
        Batch
            Batch containing 'trend', 'seasonal', and 'resid' variables for each polarization.

        Examples
        --------
        >>> displacement = stack.lstsq(phase, corr)
        >>> stl_result = stack.stl(displacement, freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, resid components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        import xarray as xr
        from .BatchCore import BatchCore

        # Validate input
        if not isinstance(data, dict):
            raise TypeError(f"data must be a Batch, got {type(data).__name__}")

        # Validate lazy data
        BatchCore._require_lazy(data, 'stl')

        sample_ds = next(iter(data.values()))
        if 'date' not in sample_ds.dims:
            raise ValueError("Input Batch must have 'date' dimension for STL decomposition")

        # Get polarizations from the first dataset (spatial, with y/x dims) - excludes converted attributes
        polarizations = [v for v in sample_ds.data_vars
                        if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]

        results = {}
        for key, ds in data.items():
            result_vars = {}
            for pol in polarizations:
                if pol not in ds.data_vars:
                    continue
                da = ds[pol]
                # Apply STL decomposition
                stl_ds = self._stl(da, freq=freq, periods=periods, robust=robust)
                # Rename variables to include polarization
                for var in ['trend', 'seasonal', 'resid']:
                    result_vars[f'{pol}_{var}'] = stl_ds[var]

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            # Preserve CRS if available
            if hasattr(ds, 'rio') and ds.rio.crs is not None:
                import rioxarray
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        return Batch(results)

    # Aggregate data for varying frequencies (e.g., 12+ days for 6 days S1AB images interval)
    # Use frequency strings like '1W' for 1 week, '2W' for 2 weeks, '10d' for 10 days, '1M' for 1 month, etc.
    def _stl(self, data, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL) on the input time series data in parallel.

        The function performs the following steps:
        1. Convert the 'date' coordinate to valid dates.
        2. Unify date intervals to a specified frequency (e.g., weekly) for a mix of time intervals.
        3. Apply the Stack.stl1d function in parallel using Dask.
        4. Rename the output date dimension to match the original irregular date dimension.
        5. Return the STL decomposition results as an xarray Dataset.

        Parameters
        ----------
        self : Stack
            Instance of the Stack class.
        data : xarray.DataArray
            Input time series data as an xarray DataArray.
        freq : str, optional
            Frequency string for unifying date intervals (default is 'W' for weekly).
        periods : int, optional
            Number of periods for seasonal decomposition (default is 52).
        robust : bool, optional
            Whether to use a slower robust fitting procedure for the STL decomposition (default is False).

        Returns
        -------
        xarray.Dataset or None
            An xarray Dataset containing the trend, seasonal, and residual components of the decomposed time series,
            or None if the results are saved to a file.

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
        """
        import xarray as xr
        import numpy as np
        import pandas as pd
        import dask
        # disable "distributed.utils_perf - WARNING - full garbage collections ..."
        try:
            from dask.distributed import utils_perf
            utils_perf.disable_gc_diagnosis()
        except ImportError:
            from distributed.gc import disable_gc_diagnosis
            disable_gc_diagnosis()

        assert data.dims[0] == 'date', 'The first data dimension should be date'

        # Default chunk sizes if not set on Stack
        netcdf_chunksize = getattr(self, 'netcdf_chunksize', 512)
        chunksize1d = getattr(self, 'chunksize1d', 10000)

        if not isinstance(data, xr.DataArray):
            raise Exception('Invalid input: The "data" parameter should be of type xarray.DataArray.')

        dt, dt_periodic = utils_stl.stl_periodic(data.date, freq)
        n_dates_out = len(dt_periodic)
        n_dates_in = data.date.size

        data_dask = data.data

        def _stl_block(data_block, _dt=dt, _dt_periodic=dt_periodic,
                       _n_dates_out=n_dates_out, _periods=periods, _robust=robust):
            import math
            from .utils_dask import get_dask_chunk_size_mb
            ny, nx = data_block.shape[1], data_block.shape[2]
            n_dates_in_local = data_block.shape[0]
            result = np.empty((3, _n_dates_out, ny, nx), dtype=np.float32)
            vec_stl = np.vectorize(
                lambda ts: utils_stl.stl1d(ts, _dt, _dt_periodic, _periods, _robust),
                signature='(n)->(m),(m),(m)'
            )
            per_pixel_bytes = (n_dates_in_local + 3 * _n_dates_out) * 4
            budget_bytes = int(get_dask_chunk_size_mb() * 1024 * 1024)
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

        models = dask.array.blockwise(
            _stl_block, 'cdyx',
            data_dask, 'pyx',
            new_axes={'c': 3, 'd': n_dates_out},
            concatenate=True,
            dtype=np.float32,
            meta=np.empty((0, 0, 0, 0), dtype=np.float32),
        )

        coords = {'date': dt_periodic.astype('datetime64[ns]'), 'y': data.y, 'x': data.x}

        # transform to separate variables
        varnames = ['trend', 'seasonal', 'resid']
        keys_vars = {}
        for varidx, varname in enumerate(varnames):
            var_data = models[varidx]
            keys_vars[varname] = xr.DataArray(var_data, coords=coords)
        model = xr.Dataset({**keys_vars})
        del models

        return model
