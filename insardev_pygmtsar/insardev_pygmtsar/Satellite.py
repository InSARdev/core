# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from insardev_toolkit import progressbar_joblib
from insardev_toolkit import datagrid


class Satellite(progressbar_joblib, datagrid):
    """Abstract base class for satellite processing with common utilities.

    Provides shared functionality for S1 (Sentinel-1) and NISAR processing.
    Subclasses must have a `df` attribute (GeoDataFrame) with MultiIndex.
    """
    import geopandas as gpd
    import xarray as xr
    import pandas as pd

    def __repr__(self):
        return 'Object %s %d items\n%r' % (self.__class__.__name__, len(self.df), self.df)

    def to_dataframe(self, crs: int = 4326, ref: str = None) -> pd.DataFrame:
        """
        Return a Pandas DataFrame for all records.

        Parameters
        ----------
        crs : int
            Coordinate reference system EPSG code. Default is 4326 (WGS84).
        ref : str, optional
            Reference date (YYYY-MM-DD) to filter by. If provided, only records
            with matching ref_ids are returned.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing records, reprojected to the specified CRS.

        Examples
        --------
        >>> df = stack.to_dataframe()
        >>> df_ref = stack.to_dataframe(ref='2023-01-15')
        """
        if ref is None:
            df = self.df
        else:
            # Get ref_ids from reference date and filter all records by them
            ref_ids = self.df[self.df.startTime.dt.date.astype(str) == ref].index.get_level_values(0).unique()
            if len(ref_ids) == 0:
                raise ValueError(f"Reference date '{ref}' not found in the data")
            df = self.df[self.df.index.get_level_values(0).isin(ref_ids)]
        return df.set_crs(4326).to_crs(crs)

    def get_record(self, record_id: str) -> pd.DataFrame:
        """
        Return dataframe record for a given identifier.

        Parameters
        ----------
        record_id : str
            Record identifier (can be full ID at level 2 or ref_id at level 0).

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the record.

        Raises
        ------
        AssertionError
            If no record is found for the given identifier.
        """
        df = self.df[self.df.index.get_level_values(2) == record_id]
        if len(df) == 0:
            df = self.df[self.df.index.get_level_values(0) == record_id]
        assert len(df) > 0, f'Record not found: {record_id}'
        return df

    def fullBurstId(self, record_id: str) -> str:
        """Get the fullBurstId/sceneId (level 0 index) for a record.

        Parameters
        ----------
        record_id : str
            Record identifier (burst or scene).

        Returns
        -------
        str
            The fullBurstId/sceneId (level 0 index value).
        """
        df = self.get_record(record_id)
        return df.index.get_level_values(0)[0]

    def sceneId(self, record_id: str) -> str:
        """Alias for fullBurstId() - get the sceneId (level 0 index) for a record."""
        return self.fullBurstId(record_id)

    def plot(self, records: pd.DataFrame = None, ref: str = None,
             alpha: float = 0.7, caption: str = 'Estimated Footprint',
             cmap: str = 'turbo', aspect: float = None, _size: tuple[int, int] = None,
             ax=None):
        """
        Plot scene/burst footprints on a map.

        Parameters
        ----------
        records : pd.DataFrame, optional
            Records to plot. Default is all records.
        ref : str, optional
            Reference date to filter records.
        alpha : float, optional
            Transparency of the DEM overlay.
        caption : str, optional
            Plot title.
        cmap : str, optional
            Colormap for scene colors.
        aspect : float, optional
            Aspect ratio for the plot.
        _size : tuple[int, int], optional
            Screen size in pixels for decimation.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib

        if _size is None:
            _size = (2000, 1000)

        if records is None:
            records = self.to_dataframe(ref=ref)

        if ax is None:
            plt.figure()
            ax = plt.gca()
        if self.DEM is not None:
            dem = self.get_dem()
            size_y, size_x = dem.shape
            factor_y = int(np.round(size_y / _size[1]))
            factor_x = int(np.round(size_x / _size[0]))
            dem = dem[::max(1, factor_y), ::max(1, factor_x)]
            dem.plot.imshow(cmap='gray', alpha=alpha, add_colorbar=True, ax=ax)

        cmap_obj = matplotlib.colormaps[cmap]
        colors = dict([(v, cmap_obj(k)) for k, v in enumerate(records.index.unique())])

        # Calculate overlaps
        overlap_count = [sum(1 for geom2 in records.geometry if geom1.intersects(geom2))
                         for geom1 in records.geometry]
        _alpha = max(1 / max(overlap_count), 0.002)
        _alpha = min(_alpha, alpha / 2)

        records.reset_index().plot(color=[colors[k] for k in records.index],
                                   alpha=_alpha, edgecolor='black', ax=ax)
        if aspect is not None:
            ax.set_aspect(aspect)
        ax.set_title(caption)

    def consolidate_metadata(self, target: str, record_id: str = None):
        """
        Consolidate zarr metadata for a given target directory.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        record_id : str, optional
            The scene/burst identifier. If provided, consolidates metadata
            for that specific record's subdirectory.
        """
        import zarr
        import os

        root_dir = target
        if record_id:
            root_dir = os.path.join(target, self.fullBurstId(record_id))
        root_store = zarr.storage.LocalStore(root_dir)
        zarr.group(store=root_store, zarr_format=3, overwrite=False)
        zarr.consolidate_metadata(root_store)

    def get_repref(self, ref: str, records: pd.DataFrame = None) -> dict:
        """
        Get the reference and repeat records for a given reference date.

        Parameters
        ----------
        ref : str
            The reference date (YYYY-MM-DD).
        records : pd.DataFrame, optional
            The DataFrame containing the records. If None, uses to_dataframe(ref=ref).

        Returns
        -------
        dict
            A dictionary mapping ref_id -> (ref_list, rep_list) where each list
            contains tuples of record indices.
        """
        if records is None:
            records = self.to_dataframe(ref=ref)

        recs_ref = records[records.startTime.dt.date.astype(str) == ref]
        refs_dict = {}
        for rec in recs_ref.itertuples():
            refs_dict.setdefault(rec.Index[0], []).append(rec.Index)

        recs_rep = records[records.startTime.dt.date.astype(str) != ref]
        reps_dict = {}
        for rec in recs_rep.itertuples():
            reps_dict.setdefault(rec.Index[0], []).append(rec.Index)

        for key in refs_dict:
            if key not in reps_dict:
                print(f'NOTE: {key} has no repeat records, ignore.')
        for key in reps_dict:
            if key not in refs_dict:
                print(f'NOTE: {key} has no reference records, ignore.')

        # Return only pairs with both reference and repeat records
        return {key: (refs_dict[key], reps_dict[key]) for key in refs_dict if key in reps_dict}

    def julian_to_datetime(self, julian_timestamp: float) -> pd.Timestamp:
        """
        Convert Julian timestamp to datetime.

        Parameters
        ----------
        julian_timestamp : float
            Timestamp in format YYYYDOY.FRACTION, e.g., 2023040.1484139557
            where YYYY is year, DOY is day of year, and FRACTION is fractional day.

        Returns
        -------
        pd.Timestamp
            Converted datetime.

        Examples
        --------
        >>> stack.julian_to_datetime(2023040.5)
        Timestamp('2023-02-10 12:00:00')
        """
        import pandas as pd

        year = int(julian_timestamp / 1000)
        doy = int(julian_timestamp % 1000)
        fraction = julian_timestamp - int(julian_timestamp)

        base_date = pd.Timestamp(f"{year}-01-01")
        date = base_date + pd.Timedelta(days=doy) + pd.Timedelta(days=fraction)

        return date

    def get_geoid(self, grid: xr.DataArray | xr.Dataset = None) -> xr.DataArray:
        """
        Get EGM96 geoid heights.

        Parameters
        ----------
        grid : xarray array or dataset, optional
            Interpolate geoid heights on the grid.

        Returns
        -------
        xr.DataArray
            Geoid heights in meters.

        Notes
        -----
        See EGM96 geoid heights on http://icgem.gfz-potsdam.de/tom_longtime
        """
        from .utils_satellite import get_geoid
        return get_geoid(grid, netcdf_engine=self.netcdf_engine_read)

    def get_dem(self, geometry: gpd.GeoDataFrame = None, buffer_degrees: float = 0):
        """
        Load and preprocess digital elevation model (DEM) data.

        Parameters
        ----------
        geometry : geopandas.GeoDataFrame, optional
            The geometry of the area to crop the DEM.
        buffer_degrees : float, optional
            The buffer in degrees to add to the geometry.

        Returns
        -------
        xarray.DataArray
            The DEM data array.
        """
        import xarray as xr
        import numpy as np
        import rioxarray as rio
        import pandas as pd
        import os

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not specified.')

        if geometry is None:
            geometry = self.df

        if isinstance(self.DEM, xr.Dataset):
            ortho = self.DEM[list(self.DEM.data_vars)[0]]
        elif isinstance(self.DEM, xr.DataArray):
            ortho = self.DEM
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.tiff', '.tif', '.TIF']:
            ortho = rio.open_rasterio(self.DEM).squeeze(drop=True) \
                .rename({'y': 'lat', 'x': 'lon'}) \
                .drop('spatial_ref')
            if ortho.lat.diff('lat')[0].item() < 0:
                ortho = ortho.reindex(lat=ortho.lat[::-1])
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.nc', '.netcdf', '.grd']:
            ortho = xr.open_dataarray(self.DEM, engine=self.netcdf_engine_read)
        elif isinstance(self.DEM, str):
            raise ValueError('ERROR: filename extension not recognized. Use .tiff, .tif, .TIF, .nc, .netcdf, .grd')
        else:
            raise ValueError('ERROR: argument is not an Xarray object and it is not a file name')
        ortho = ortho.transpose('lat', 'lon')

        # Unique indices required for interpolation
        lat_index = pd.Index(ortho.coords['lat'])
        lon_index = pd.Index(ortho.coords['lon'])
        duplicates = lat_index[lat_index.duplicated()].tolist() + lon_index[lon_index.duplicated()].tolist()
        assert len(duplicates) == 0, 'ERROR: DEM grid includes duplicated coordinates'

        # Crop to the geometry extent
        bounds = self.get_bounds(geometry.buffer(buffer_degrees))
        ortho = ortho.sel(lat=slice(bounds[1], bounds[3]), lon=slice(bounds[0], bounds[2]))

        ds = ortho.astype(np.float32).transpose('lat', 'lon').rename("dem")
        return self.spatial_ref(ds, 4326)

    def get_dem_wgs84ellipsoid(self, geometry: gpd.GeoDataFrame = None, buffer_degrees: float = 0.04):
        """
        Load DEM with EGM96 geoid correction (heights relative to WGS84 ellipsoid).

        Parameters
        ----------
        geometry : geopandas.GeoDataFrame, optional
            The geometry of the area to crop the DEM.
        buffer_degrees : float, optional
            The buffer in degrees to add to the geometry.

        Returns
        -------
        xarray.DataArray
            WGS84 ellipsoid DEM data array.
        """
        ortho = self.get_dem(geometry, buffer_degrees)
        geoid = self.get_geoid(ortho)
        ds = (ortho + geoid).rename("dem")
        return self.spatial_ref(ds, 4326)

    def _get_topo_llt(self, record_id: str, degrees: float, debug: bool = False):
        """
        Get the topography coordinates (lon, lat, z) for decimated DEM.

        Memory-efficient version using netCDF4 direct slicing - never loads full DEM.
        Supports 200GB+ global DEMs referenced for all scenes/bursts.

        Parameters
        ----------
        record_id : str
            Scene or burst identifier.
        degrees : float
            Number of degrees for decimation.
        debug : bool, optional
            Enable debug mode. Default is False.

        Returns
        -------
        numpy.ndarray
            Array containing the topography coordinates (lon, lat, z), NaN filtered.
        """
        import numpy as np
        import os

        record = self.get_record(record_id)
        geometry = record.geometry

        # Get bounds with buffer
        buffer_degrees = 0.04
        bounds = geometry.buffer(buffer_degrees).total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = bounds

        # Open DEM file directly (never loads full array!)
        dem_path = self.DEM
        if not isinstance(dem_path, str) or not os.path.exists(dem_path):
            raise ValueError(f'DEM path must be a valid file: {dem_path}')

        # Use configured netcdf engine for optimized strided access
        if self.netcdf_engine_read == 'h5netcdf':
            import h5netcdf
            nc = h5netcdf.File(dem_path, 'r')
        else:
            from netCDF4 import Dataset
            nc = Dataset(dem_path, 'r')

        try:
            # Get coordinate arrays (these are small - just 1D indices)
            lat_var = nc.variables.get('lat') or nc.variables.get('y')
            lon_var = nc.variables.get('lon') or nc.variables.get('x')
            lat_arr = lat_var[:]
            lon_arr = lon_var[:]

            # Find indices for the required region
            lat_idx = np.where((lat_arr >= lat_min) & (lat_arr <= lat_max))[0]
            lon_idx = np.where((lon_arr >= lon_min) & (lon_arr <= lon_max))[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                raise ValueError(f'DEM does not cover bounds: {bounds}')

            lat_start, lat_end = lat_idx[0], lat_idx[-1] + 1
            lon_start, lon_end = lon_idx[0], lon_idx[-1] + 1

            # Compute decimation factor
            dem_res = abs(lat_arr[1] - lat_arr[0])
            dec_factor = max(1, int(np.round(degrees / dem_res)))

            if debug:
                print(f'DEBUG: DEM decimation factor={dec_factor}, region={lat_end-lat_start}x{lon_end-lon_start}')

            # Read DEM data with striding (decimated read - memory efficient!)
            dem_var = nc.variables.get('z') or nc.variables.get('elevation') or nc.variables.get('dem')
            if dem_var is None:
                # Try first 2D variable
                for name, var in nc.variables.items():
                    if len(var.dimensions) == 2:
                        dem_var = var
                        break

            # Strided read: only loads every dec_factor-th point
            z_vals = dem_var[lat_start:lat_end:dec_factor, lon_start:lon_end:dec_factor].astype(np.float32)
            lat_vals = lat_arr[lat_start:lat_end:dec_factor]
            lon_vals = lon_arr[lon_start:lon_end:dec_factor]
        finally:
            nc.close()

        # Apply geoid correction (EGM96 -> WGS84 ellipsoid)
        from .utils_satellite import get_geoid_correction
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        geoid = get_geoid_correction(lat_grid.ravel(), lon_grid.ravel(), netcdf_engine=self.netcdf_engine_read)
        z_wgs84 = z_vals.ravel() + geoid.astype(np.float32)
        del geoid

        # Build topo_llt array
        topo_llt = np.column_stack([
            lon_grid.ravel(),
            lat_grid.ravel(),
            z_wgs84
        ])
        del lon_grid, lat_grid, z_vals, z_wgs84, lat_vals, lon_vals

        # Filter out NaN elevation values
        valid_mask = ~np.isnan(topo_llt[:, 2])
        result = topo_llt[valid_mask].astype(np.float64)
        del topo_llt, valid_mask

        if debug:
            print(f'DEBUG: topo_llt points={len(result)}')

        return result

    def geocode(self, transform: xr.Dataset, data: xr.DataArray,
                resolution: tuple[float, float] = None) -> xr.DataArray:
        """
        Perform geocoding from radar to projected coordinates using inverse transform.

        The inverse transform has coords (y, x) and vars (rng, azi, ele).
        Uses cv2.remap with Lanczos interpolation directly on the inverse maps.

        Parameters
        ----------
        transform : xarray.Dataset
            The inverse transform with coords (y, x) and vars (rng, azi, ele).
        data : xarray.DataArray
            Grid(s) in radar coordinates (a, r).
        resolution : tuple[float, float], optional
            Output resolution (dy, dx) in meters. If None, uses transform resolution.

        Returns
        -------
        xarray.DataArray
            The geocoded grid(s) in projected coordinates (y, x).
        """
        import cv2
        import xarray as xr
        import numpy as np

        # get transform arrays - inverse transform has coords (y, x), vars (azi, rng)
        trans_azi = transform.azi.values  # 2D: (n_y, n_x)
        trans_rng = transform.rng.values  # 2D: (n_y, n_x)
        out_y = transform.y.values  # 1D
        out_x = transform.x.values  # 1D

        # get data arrays - data is in radar coords (a, r)
        data_vals = data.values
        coord_a = data.a.values
        coord_r = data.r.values

        # Convert transform azi/rng to fractional radar indices
        # inv_map_a[i,j] = row index in radar data, inv_map_r[i,j] = col index
        inv_map_a = ((trans_azi - coord_a[0]) / (coord_a[1] - coord_a[0])).astype(np.float32)
        inv_map_r = ((trans_rng - coord_r[0]) / (coord_r[1] - coord_r[0])).astype(np.float32)

        n_y, n_x = inv_map_a.shape
        OPENCV_MAX = 32766  # cv2.remap requires dimensions < 32767

        # Use cv2.remap with Lanczos to sample radar data at inverse map coordinates
        # map_r is x (column), map_a is y (row) in source image
        if n_x <= OPENCV_MAX:
            # Direct remap - no chunking needed
            if np.iscomplexobj(data_vals):
                grid_proj_re = cv2.remap(data_vals.real.astype(np.float32), inv_map_r, inv_map_a,
                                         interpolation=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj_im = cv2.remap(data_vals.imag.astype(np.float32), inv_map_r, inv_map_a,
                                         interpolation=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj = (grid_proj_re + 1j * grid_proj_im).astype(data.dtype)
            else:
                grid_proj = cv2.remap(data_vals.astype(np.float32), inv_map_r, inv_map_a,
                                      interpolation=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        else:
            # Chunked remap - split x dimension into minimal equal chunks
            n_chunks = (n_x + OPENCV_MAX - 1) // OPENCV_MAX
            x_indices = np.arange(n_x)
            chunk_indices = np.array_split(x_indices, n_chunks)

            if np.iscomplexobj(data_vals):
                grid_proj = np.empty((n_y, n_x), dtype=data.dtype)
                data_re = data_vals.real.astype(np.float32)
                data_im = data_vals.imag.astype(np.float32)
                for idx in chunk_indices:
                    x_slice = slice(idx[0], idx[-1] + 1)
                    re_chunk = cv2.remap(data_re, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                         interpolation=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                    im_chunk = cv2.remap(data_im, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                         interpolation=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                    grid_proj[:, x_slice] = (re_chunk + 1j * im_chunk).astype(data.dtype)
                del data_re, data_im
            else:
                grid_proj = np.empty((n_y, n_x), dtype=np.float32)
                data_f32 = data_vals.astype(np.float32)
                for idx in chunk_indices:
                    x_slice = slice(idx[0], idx[-1] + 1)
                    grid_proj[:, x_slice] = cv2.remap(data_f32, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                                      interpolation=cv2.INTER_CUBIC,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                del data_f32

        coords = {'y': out_y, 'x': out_x}
        return xr.DataArray(grid_proj, coords=coords, dims=['y', 'x']).rename(data.name)

    def get_transform(self, outdir: str, scene: str = None) -> xr.Dataset:
        """
        Retrieve the inverse transform data.

        The inverse transform has coords (y, x) and vars (rng, azi, ele, look_E, look_N, look_U).
        For each projected pixel, stores the corresponding radar coordinates and look vectors.

        Parameters
        ----------
        outdir : str
            Output directory containing transform zarr.
        scene : str, optional
            Scene/burst name (not used, kept for API compatibility).

        Returns
        -------
        xarray.Dataset
            An xarray dataset with the transform data.
        """
        import xarray as xr
        import numpy as np
        import os

        ds = xr.open_zarr(store=os.path.join(outdir, 'transform'),
                         consolidated=True,
                         zarr_format=3,
                         chunks='auto')
        # variables are stored as int32 with _FillValue and scale_factor
        # inverse transform has vars (rng, azi, ele)
        for v in ('rng', 'azi', 'ele'):
            if v not in ds:
                continue
            fill_value = ds[v].attrs.get('_FillValue')
            scale_factor = ds[v].attrs.get('scale_factor', 1.0)
            if fill_value is not None:
                # xarray didn't decode - apply manually
                data = ds[v].astype('float32')
                data = data.where(ds[v] != fill_value)
                ds[v] = data * scale_factor
            else:
                # xarray already decoded - ensure float32 and mask extreme values
                data = ds[v].astype('float32')
                ds[v] = data.where(np.abs(data) < 1e8)
        return ds
