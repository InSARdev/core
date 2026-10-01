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

    def _repr_html_(self) -> str:
        """Render the records as a table in a notebook, the way the records themselves render.

        Defined explicitly because __getattr__ refuses private names, so this cannot be reached
        by delegation, and without it a notebook falls back to the plain text __repr__.
        """
        return 'Object %s %d items%s' % (self.__class__.__name__, len(self.df), self.df._repr_html_())

    def __len__(self) -> int:
        """Number of records."""
        return len(self.df)

    def __contains__(self, key) -> bool:
        """Whether a column exists, as for the records."""
        return key in self.df

    def __iter__(self):
        """Iterate the column names, as the records do."""
        return iter(self.df)

    def __getattr__(self, name: str):
        """Attribute access to the records, so that s1.mission is the mission column.

        Only reached when the attribute is defined neither on the object nor on its class, so
        it never shadows a method. Private and dunder names are refused: the copy and pickle
        protocols look their hooks up with getattr, and answering those from the records would
        copy and pickle the records instead of this object, which transform() sends to the
        joblib workers.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        df = self.__dict__.get('df')
        if df is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(df, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    def __getitem__(self, key):
        """Select records or columns, as for the records themselves.

        Selecting records returns an object of the same kind, so the result is still ready for
        processing, e.g. s1 = s1[s1.mission == 'S1A']. Selecting a column returns the column.
        """
        records = self.df[key]
        if isinstance(records, self.gpd.GeoDataFrame) and records.index.names == self.df.index.names:
            return self.with_records(records)
        return records

    def with_records(self, records) -> 'Satellite':
        """Return this object holding the given records, without scanning the data again."""
        import copy
        obj = copy.copy(self)
        obj.df = records
        return obj

    def baselines(self, debug: bool = False):
        """Perpendicular baseline of every record in meters.

        A baseline is orbit geometry, so it needs neither the images nor a DEM nor a
        transform, and the reference date can be chosen before any processing. The value is
        measured from the first date of the group, which is only the origin of the scale: a
        pair baseline is a difference of two of them, so the origin cancels.

        THIS ORIGIN IS NOT THE STACK'S REFERENCE. Only `transform(ref=...)` knows which
        date the stack is referenced to, and it measures each burst from that one, so a
        stored stack has BPR == 0 exactly at its reference; a value from here must never
        overwrite it. The origin is
        reported for every group, because a value on its own means nothing without it, and it
        is always the first date: any date would do, so a baseline that cannot be solved is
        reported as an error rather than worked around by choosing another origin.

        Sentinel-1 reads the orbits from files downloaded after the bursts, so its baselines
        stay undefined until the orbits are there and the data directory is scanned again.
        Nisar carries the orbit inside the scene and has no such wait.

        Parameters
        ----------
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        pandas.Series
            Perpendicular baseline per record, NaN where it cannot be computed.

        Raises
        ------
        OSError
            An input file that cannot be read: empty (utils_files.EmptyFileError), missing, or
            unreadable. Any other error reading the inputs raises too. Only a baseline that the
            orbits do not solve (ValueError or IndexError of SAT_baseline) is reported as an
            ERROR and left NaN.
        ValueError
            An orbit that cannot be used: a Sentinel-1 orbit file that does not cover its burst or
            holds a state vector that is not finite, or a baseline that is not finite.
        """
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        BPRs = pd.Series(np.nan, index=self.df.index, name='BPR')
        if 'orbit' in self.df.columns and self.df.orbit.isnull().all():
            print('NOTE: Orbits are not downloaded, baselines are undefined. '
                  'Download the orbits and scan the data directory again.')
            return BPRs

        # bursts are keyed by fullBurstID and burst, scenes by sceneId and scene
        group_level, record_level = self.df.index.names[0], self.df.index.names[2]
        for group, records in self.df.groupby(level=group_level):
            dates = records.startTime.dt.date
            # the polarizations share the acquisition, so one record per date is enough
            firsts = records[~dates.duplicated()]
            if 'orbit' in firsts.columns and firsts.orbit.isnull().any():
                print(f'NOTE: {group} misses orbits, its baselines are undefined.')
                continue

            names = list(firsts.index.get_level_values(record_level))
            name_dates = list(dates[~dates.duplicated()])
            # the file holding the orbit of each record: the orbit file (Sentinel-1) or the scene (Nisar)
            sources = dict(zip(names, firsts['orbit' if 'orbit' in firsts.columns else 'path']))

            prms = {}
            def prm_of(name):
                if name not in prms:
                    prms[name], _, _ = self.align_ref(name, debug=debug, return_slc=False)
                return prms[name]

            # the origin is the first date and never moves: it only shifts the whole scale,
            # so moving it would hide a defect instead of reporting one
            origin, origin_date = names[0], name_dates[0]
            values = {}
            with tqdm(desc='Computing Baselines'.ljust(25), total=len(names)) as pbar:
                for name, date in zip(names, name_dates):
                    if name == origin:
                        values[date] = 0.0
                    else:
                        # the inputs are read outside the try: an empty or missing file (EmptyFileError,
                        # FileNotFoundError, any OSError) or any other read error raises, it is no unsolved baseline
                        prm_origin, prm_name = prm_of(origin), prm_of(name)
                        try:
                            value = float(prm_origin.SAT_baseline(prm_name).get('B_perpendicular'))
                        except (ValueError, IndexError) as e:
                            # the orbits do not solve it: SAT_baseline raises ValueError for a missing orbit or a
                            # repeat orbit that does not cover the reference, IndexError for an orbit without state
                            # vectors. Every date is a valid reference, so an unsolved baseline is a
                            # defect and not a property of the data. Only this one is left out.
                            print(f'ERROR: {group} baseline of {date} is unsolved: {e}')
                        else:
                            # an invalid value (an orbit with NaN state vectors gives NaN) raises
                            if not np.isfinite(value):
                                raise ValueError(f'ERROR: {group} baseline of {date} is {value}. Check the orbits in '
                                                 f'{sources[origin]} and {sources[name]}, and download the '
                                                 f'invalid file again.')
                            values[date] = value
                    pbar.update(1)

            print(f'NOTE: {group} baselines are measured from {origin_date}.')
            BPRs[records.index] = [values.get(date, np.nan) for date in dates]

        return BPRs

    def _check_orbits(self, refreps: list, target: str, overwrite: bool, append: bool):
        """Check the orbit of every record a transform processes, as the processing reads it, before the
        transform removes or writes anything: an orbit that cannot be used raises there, not after the earlier
        records are written. Each orbit file is read once for all the records that use it (the mission's
        _orbit_file and _check_orbit_file). A completed target that the transform keeps (neither overwrite nor
        append) is not processed and needs no check.

        Parameters
        ----------
        refreps : list
            (reference records, repeat records) per group, the values of get_repref().
        target, overwrite, append
            The transform's arguments.

        Raises
        ------
        FileNotFoundError
            A record without its orbit file (Sentinel-1).
        ValueError
            An orbit that does not cover its record, has a gap or a state vector that is not finite
            (utils_satellite.orbit_defect).
        """
        import os
        from tqdm.auto import tqdm

        if not (overwrite or append) and os.path.isfile(os.path.join(target, 'zarr.json')):
            return
        records = [record[-1] for refs, reps in refreps for record in refs + reps]
        by_file = {}
        for record in records:
            by_file.setdefault(self._orbit_file(record, self.get_record(record)), []).append(record)
        for orbit_file, group in tqdm(by_file.items(), desc='Checking Orbits'.ljust(25)):
            self._check_orbit_file(orbit_file, group)

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
        # the records are WGS84 already, so return them as they are and reproject only when
        # another projection is asked for. Without a reference date this is the records object
        # itself, so that a selection assigned to it is not silently lost on a copy.
        if crs == 4326:
            return df
        return df.to_crs(crs)

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

    def plot(self, ref: str = None,
             alpha: float = 0.7, caption: str = 'Estimated Footprint',
             cmap: str = 'turbo', aspect: float = None, _size: tuple[int, int] = None,
             ax=None):
        """
        Plot scene/burst footprints on a map.

        Parameters
        ----------
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

    def get_repref(self, ref: str) -> dict:
        """
        Get the reference and repeat records for a given reference date.

        Parameters
        ----------
        ref : str
            The reference date (YYYY-MM-DD).

        Returns
        -------
        dict
            A dictionary mapping ref_id -> (ref_list, rep_list) where each list
            contains tuples of record indices.
        """
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

    def dem_datum(self) -> str:
        """
        Get the vertical datum of the DEM: 'EGM2008', 'EGM96' or 'ellipsoid'.

        Resolved once per DEM by insardev_toolkit utils_geoid.dem_datum(): the datum the DEM declares, the tile names
        of older toolkit downloads, otherwise EGM2008 with a warning. The DEM height plus the geoid height of this
        datum is the WGS84 ellipsoidal height of the processing.

        Returns
        -------
        str
            The vertical datum.
        """
        if self.DEM is None:
            raise ValueError('ERROR: DEM is not specified.')
        cached = getattr(self, '_dem_datum', None)
        if cached is None or cached[0] is not self.DEM:
            from insardev_toolkit import utils_geoid
            self._dem_datum = cached = (self.DEM, utils_geoid.dem_datum(self.DEM))
        return cached[1]

    def get_geoid(self, grid: xr.DataArray | xr.Dataset = None) -> xr.DataArray:
        """
        Get the geoid heights of the DEM's vertical datum (dem_datum()).

        Parameters
        ----------
        grid : xarray array or dataset, optional
            Interpolate geoid heights on the grid (its lat and lon coordinates). Default is the DEM grid (get_dem()).

        Returns
        -------
        xr.DataArray
            Geoid heights in meters.

        Notes
        -----
        The NGA EGM2008 and EGM96 grids of insardev_toolkit, cubic B-spline (utils_geoid.geoid_height()).
        """
        import xarray as xr
        from insardev_toolkit import utils_geoid
        if grid is None:
            grid = self.get_dem()
        lat, lon = grid.lat.values, grid.lon.values
        values = utils_geoid.geoid_height(lat, lon, self.dem_datum(), grid=True)
        return xr.DataArray(values, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'), name='geoid')

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
        elif isinstance(self.DEM, str):
            # NetCDF4 grid or VRT of NetCDF4 tiles, opened lazily through h5py
            from insardev_toolkit import utils_tiles
            ortho = utils_tiles.open_dem(self.DEM)
        else:
            raise ValueError('ERROR: argument is not an Xarray object and it is not a file name')
        ortho = ortho.transpose('lat', 'lon')

        # Unique indices required for interpolation
        lat_index = pd.Index(ortho.coords['lat'])
        lon_index = pd.Index(ortho.coords['lon'])
        duplicates = lat_index[lat_index.duplicated()].tolist() + lon_index[lon_index.duplicated()].tolist()
        assert len(duplicates) == 0, 'ERROR: DEM grid includes duplicated coordinates'

        # Crop to the geometry extent. The buffer is in degrees on purpose, so silence the
        # warning about buffering a geographic CRS.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*geographic CRS.*')
            bounds = self.get_bounds(geometry.buffer(buffer_degrees))
        # the same crop rule as the window readers: a bound on a pixel centre keeps that pixel
        from insardev_toolkit import utils_tiles
        ortho = utils_tiles.crop(ortho, (bounds[0], bounds[1], bounds[2], bounds[3]))

        ds = ortho.astype(np.float32).transpose('lat', 'lon').rename("dem")
        return self.spatial_ref(ds, 4326)

    def get_dem_wgs84ellipsoid(self, geometry: gpd.GeoDataFrame = None, buffer_degrees: float = 0.04):
        """
        Load DEM with the geoid correction of its vertical datum (heights relative to WGS84 ellipsoid).

        Parameters
        ----------
        geometry : geopandas.GeoDataFrame, optional
            The geometry of the area to crop the DEM.
        buffer_degrees : float, optional
            The buffer in degrees to add to the geometry.

        Returns
        -------
        xarray.DataArray
            WGS84 ellipsoid DEM data array (insardev_toolkit utils_geoid.ellipsoidal_height()).
        """
        from insardev_toolkit import utils_geoid
        ortho = self.get_dem(geometry, buffer_degrees)
        ds = utils_geoid.ellipsoidal_height(ortho, self.dem_datum()).rename("dem")
        return self.spatial_ref(ds, 4326)

    def _get_topo_llt(self, record_id: str, degrees: float, debug: bool = False):
        """
        Get the topography coordinates (lon, lat, z) for decimated DEM.

        Memory-efficient version reading the decimated window through h5py - never loads full DEM.
        Supports 200GB+ global DEMs referenced for all scenes/bursts, merged or as a VRT of tiles.

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

        record = self.get_record(record_id)
        geometry = record.geometry

        # Get bounds with buffer. The buffer is in degrees on purpose, so silence the
        # warning about buffering a geographic CRS.
        import warnings
        buffer_degrees = 0.04
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*geographic CRS.*')
            bounds = geometry.buffer(buffer_degrees).total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = bounds

        # Open DEM file directly (never loads full array!); a missing file is reported by the reader
        dem_path = self.DEM
        if not isinstance(dem_path, str):
            raise ValueError(f'DEM path must be a file name: {dem_path}')
        from insardev_toolkit import utils_tiles

        # Compute decimation factor
        dem_res = utils_tiles.dem_step(dem_path)[0]
        dec_factor = max(1, int(np.round(degrees / dem_res)))

        # Strided read: only keeps every dec_factor-th point (decimated read - memory efficient!)
        window = utils_tiles.read_dem(dem_path, (lon_min, lat_min, lon_max, lat_max), stride=dec_factor)
        if window is None:
            raise ValueError(f'DEM does not cover bounds: {bounds}')
        z_vals, lat_vals, lon_vals = window

        if debug:
            print(f'DEBUG: DEM decimation factor={dec_factor}, decimated region={z_vals.shape[0]}x{z_vals.shape[1]}')

        # Apply geoid correction (DEM heights -> WGS84 ellipsoid): the geoid of the DEM's vertical datum
        import xarray as xr
        from insardev_toolkit import utils_geoid
        ortho = xr.DataArray(z_vals, coords={'lat': lat_vals, 'lon': lon_vals}, dims=('lat', 'lon'))
        z_wgs84 = utils_geoid.ellipsoidal_height(ortho, self.dem_datum()).values.ravel()
        del ortho
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

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
                                         interpolation=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj_im = cv2.remap(data_vals.imag.astype(np.float32), inv_map_r, inv_map_a,
                                         interpolation=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj = (grid_proj_re + 1j * grid_proj_im).astype(data.dtype)
            else:
                grid_proj = cv2.remap(data_vals.astype(np.float32), inv_map_r, inv_map_a,
                                      interpolation=cv2.INTER_LANCZOS4,
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
                                         interpolation=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                    im_chunk = cv2.remap(data_im, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                         interpolation=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                    grid_proj[:, x_slice] = (re_chunk + 1j * im_chunk).astype(data.dtype)
                del data_re, data_im
            else:
                grid_proj = np.empty((n_y, n_x), dtype=np.float32)
                data_f32 = data_vals.astype(np.float32)
                for idx in chunk_indices:
                    x_slice = slice(idx[0], idx[-1] + 1)
                    grid_proj[:, x_slice] = cv2.remap(data_f32, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                                      interpolation=cv2.INTER_LANCZOS4,
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
