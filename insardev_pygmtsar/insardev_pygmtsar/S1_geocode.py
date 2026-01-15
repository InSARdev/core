# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_align import S1_align

class S1_geocode(S1_align):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def geocode(self, transform: xr.Dataset, data: xr.DataArray) -> xr.DataArray:
        """
        Perform geocoding from radar to projected coordinates.

        Parameters
        ----------
        transform : xarray.Dataset
            The transform matrix.
        data : xarray.DataArray
            Grid(s) representing the interferogram(s) in radar coordinates.

        Returns
        -------
        xarray.DataArray
            The geocoded grid(s) in projected coordinates.

        Examples
        --------
        Geocode 3D unwrapped phase grid stack:
        unwraps_ll = stack.intf_ra2ll(stack.open_grids(pairs, 'unwrap'))
        # or use "geocode" option for open_grids() instead:
        unwraps_ll = stack.open_grids(pairs, 'unwrap', geocode=True)
        """
        from scipy.interpolate import RegularGridInterpolator
        import xarray as xr
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')

        # check for duplicates to avoid inconsistent indexing errors
        if np.unique(data.a.values).size != data.a.values.size:
            raise AssertionError("Duplicate azimuth coordinates detected")
        if np.unique(data.r.values).size != data.r.values.size:
            raise AssertionError("Duplicate range coordinates detected")
        if not data.indexes['a'].is_unique:
            raise AssertionError("Duplicate azimuth coordinates in indexes detected")
        if not data.indexes['r'].is_unique:
            raise AssertionError("Duplicate range coordinates in indexes detected")

        # get transform arrays
        trans_azi = transform.azi.values
        trans_rng = transform.rng.values

        # get data arrays
        coord_a = data.a.values
        coord_r = data.r.values
        data_values = data.values

        # create interpolation points from transform
        points = np.column_stack([trans_azi.ravel(), trans_rng.ravel()])

        # create interpolator for data
        interp = RegularGridInterpolator((coord_a, coord_r), data_values, method='nearest', bounds_error=False)

        # interpolate
        grid_proj = interp(points).reshape(trans_azi.shape).astype(data.dtype)

        da_out = xr.DataArray(grid_proj, transform.ele.coords).rename(data.name)
        return da_out

    @staticmethod
    def get_utm_epsg(lat: float, lon: float) -> int:
        zone_num = int((lon + 180) // 6) + 1
        if lat >= 0:
            return 32600 + zone_num
        else:
            return 32700 + zone_num
    
    @staticmethod
    def proj(ys: np.ndarray, xs: np.ndarray, to_epsg: int, from_epsg: int) -> tuple[np.ndarray, np.ndarray]:
        from pyproj import CRS, Transformer
        from_crs = CRS.from_epsg(from_epsg)
        to_crs = CRS.from_epsg(to_epsg)
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        xs_new, ys_new = transformer.transform(xs, ys)
        del transformer, from_crs, to_crs
        return ys_new, xs_new

    def get_transform(self, outdir: str, burst: str) -> xr.Dataset:
        """
        Retrieve the transform data.

        This function opens a NetCDF dataset, which contains data mapping from radar
        coordinates to geographical coordinates (from azimuth-range to latitude-longitude domain).

        Parameters
        ----------
        burst : str
            The burst name.

        Returns
        -------
        xarray.Dataset or list of xarray.Dataset
            An xarray dataset(s) with the transform data.

        Examples
        --------
        Get the inverse transform data:
        get_trans()
        """
        import xarray as xr
        import os
        import numpy as np
        ds = xr.open_zarr(store=os.path.join(outdir, 'transform'),
                         consolidated=True,
                          zarr_format=3,
                         chunks='auto')
        # variables are stored as int32 with _FillValue and scale_factor
        # modern xarray applies mask_and_scale automatically (attrs become empty)
        # older versions may not - handle both cases
        # also mask extreme values that may come from different fill_value conventions
        for v in ('azi','rng','ele'):
            fill_value = ds[v].attrs.get('_FillValue')
            scale_factor = ds[v].attrs.get('scale_factor', 1.0)
            if fill_value is not None:
                # xarray didn't decode - apply manually
                data = ds[v].astype('float32')
                data = data.where(ds[v] != fill_value)
                ds[v] = data * scale_factor
            else:
                # xarray already decoded - ensure float32 and mask extreme values
                # (handles cases where zarr fill_value differs from _FillValue attr)
                data = ds[v].astype('float32')
                ds[v] = data.where(np.abs(data) < 1e8)
        return ds
        #.dropna(dim='y', how='all')
        #.dropna(dim='x', how='all')

    def compute_transform(self,
                          outdir: str,
                          burst_ref: str,
                          basedir: str,
                          resolution: tuple[int, int]=(10, 2.5),
                          scale_factor: float=2.0,
                          epsg: int=None):
        """
        Retrieve or calculate the transform data. This transform data is then saved as
        a NetCDF file for future use.

        This function generates data mapping from geographical coordinates to radar coordinates (azimuth-range domain).
        The function uses a Digital Elevation Model (DEM) to derive the geographical coordinates, and then uses the
        `SAT_llt2rat` function to map these to radar coordinates.

        Parameters
        ----------
        burst_ref : str
            The reference burst name.
        resolution : tuple, optional
            The resolution in the azimuth and range direction.
            Default is (10, 2.5).

        Returns
        -------
        None

        Examples
        --------
        Calculate and get the transform data:
        >>> Stack.compute_trans_dat(1)
        """
        import xarray as xr
        import numpy as np
        import os
        import cv2
        import warnings
        warnings.filterwarnings('ignore')

        # range, azimuth, elevation(ref to radius in PRM), look_E, look_N, look_U
        llt2ratlook_map = {0: 'rng', 1: 'azi', 2: 'ele', 3: 'look_E', 4: 'look_N', 5: 'look_U'}

        prm = self.PRM(burst_ref, basedir)

        def SAT_llt2ratlook(lats, lons, zs):
            coords3d = np.column_stack([lons, lats, np.nan_to_num(zs)])
            rae = prm.SAT_llt2rat(coords3d, precise=1, binary=False).astype(np.float32)
            rae = rae.reshape(zs.size, 5)[...,:3]
            look = prm.SAT_look(coords3d, binary=True).astype(np.float32).reshape(zs.size, 6)[...,3:]
            out = np.concatenate([rae, look], axis=-1)
            out[~np.isfinite(zs), :] = np.nan
            return out

        def compute_trans(ys, xs, coarsen, epsg, borders):
            amin, amax, rmin, rmax = borders['amin'], borders['amax'], borders['rmin'], borders['rmax']

            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            lats, lons = self.proj(yy, xx, from_epsg=epsg, to_epsg=4326)

            dlat = float(dem.lat.diff('lat')[0])
            dlon = float(dem.lon.diff('lon')[0])
            elev = dem.sel(lat=slice(np.nanmin(lats)-dlat, np.nanmax(lats)+dlat),
                          lon=slice(np.nanmin(lons)-dlon, np.nanmax(lons)+dlon)).values

            if not elev.size:
                return np.nan * np.zeros((6, ys.size, xs.size), np.float32)

            # apply coarsen when needed
            lats_coarsen = lats[::coarsen[0], ::coarsen[1]]
            lons_coarsen = lons[::coarsen[0], ::coarsen[1]]

            # interpolate elevation at coarsened lat/lon points
            elev_da = dem.sel(lat=slice(np.nanmin(lats)-dlat, np.nanmax(lats)+dlat),
                             lon=slice(np.nanmin(lons)-dlon, np.nanmax(lons)+dlon))
            elev_coarsen = elev_da.interp({'lat': xr.DataArray(lats_coarsen), 'lon': xr.DataArray(lons_coarsen)}).values
            shape = elev_coarsen.shape

            # compute 3D radar coordinates for all the geographical 3D points
            rae = SAT_llt2ratlook(lats_coarsen.astype(np.float32).ravel(),
                                  lons_coarsen.astype(np.float32).ravel(),
                                  elev_coarsen.astype(np.float32).ravel())

            # mask invalid values for better compression
            mask = (rae[...,0]>=rmin - 2*coarsen[1]) & (rae[...,0]<=rmax + 2*coarsen[1]) \
                 & (rae[...,1]>=amin - 2*coarsen[0]) & (rae[...,1]<=amax + 2*coarsen[0])
            rae[~mask] = np.nan
            rae_coarsen = rae.reshape(shape[0], shape[1], -1)

            if coarsen[0] > 1 or coarsen[1] > 1:
                src_y_coords = np.interp(yy, ys[::coarsen[0]], np.arange(shape[0])).astype(np.float32)
                src_x_coords = np.interp(xx, xs[::coarsen[1]], np.arange(shape[1])).astype(np.float32)

                rae = np.stack([
                    cv2.remap(
                        rae_coarsen[...,i],
                        src_x_coords,
                        src_y_coords,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    for i in range(6)
                ], axis=0)
            else:
                rae = rae_coarsen.transpose(2,0,1)

            return rae

        # do not use coordinate names lat,lon because the output grid saved as (lon,lon) in this case...
        record = self.get_record(burst_ref)
        dem = self.get_dem_wgs84ellipsoid(geometry=record.geometry)

        if epsg is None:
            epsg = self.get_utm_epsg(float(dem.lat.mean()), float(dem.lon.mean()))

        a_max, r_max = prm.bounds()
        borders = {'amin': 0, 'amax': a_max, 'rmin': 0, 'rmax': r_max}

        # check DEM corners
        dem_corners = dem[::dem.lat.size-1, ::dem.lon.size-1].values
        dem_lats = dem.lat.values[[0, -1]]
        dem_lons = dem.lon.values[[0, -1]]
        lats_corners, lons_corners = np.meshgrid(dem_lats, dem_lons, indexing='ij')
        yy, xx = self.proj(lats_corners, lons_corners, from_epsg=4326, to_epsg=epsg)
        dem_y_min = np.min(resolution[0] * ((yy/resolution[0]).round() + 0.5))
        dem_y_max = np.max(resolution[0] * ((yy/resolution[0]).round() - 0.5))
        dem_x_min = np.min(resolution[1] * ((xx/resolution[1]).round() + 0.5))
        dem_x_max = np.max(resolution[1] * ((xx/resolution[1]).round() - 0.5))
        ys = np.arange(dem_y_min, dem_y_max + resolution[0], resolution[0])
        xs = np.arange(dem_x_min, dem_x_max + resolution[1], resolution[1])

        # OpenCV remap requires dimensions strictly < SHRT_MAX (32767)
        # Crop symmetrically from center if exceeded (border pixels are typically NaN)
        SHRT_MAX = 32767
        max_dim = SHRT_MAX - 1  # OpenCV requires strictly less than SHRT_MAX
        if ys.size > max_dim:
            excess = ys.size - max_dim
            ys = ys[excess//2 : ys.size - (excess - excess//2)]
            print(f'NOTE: y dimension cropped by {excess} pixels to fit OpenCV limit ({SHRT_MAX})')
        if xs.size > max_dim:
            excess = xs.size - max_dim
            xs = xs[excess//2 : xs.size - (excess - excess//2)]
            print(f'NOTE: x dimension cropped by {excess} pixels to fit OpenCV limit ({SHRT_MAX})')

        dem_spacing = ((dem_y_max - dem_y_min)/dem.lat.size, (dem_x_max - dem_x_min)/dem.lon.size)

        # transform user-specified grid resolution to coarsen factor
        coarsen = (
            max(1, int(np.round(dem_spacing[0]/resolution[0]))),
            max(1, int(np.round(dem_spacing[1]/resolution[1])))
        )

        # estimate the radar extent on decimated grid
        decimation = 10
        rae_est = compute_trans(ys[::decimation], xs[::decimation], coarsen, epsg, borders)
        ele_est = rae_est[2]
        # find valid extent
        valid_mask = np.isfinite(ele_est)
        if valid_mask.any():
            valid_rows = np.any(valid_mask, axis=1)
            valid_cols = np.any(valid_mask, axis=0)
            y_indices = np.where(valid_rows)[0]
            x_indices = np.where(valid_cols)[0]
            y_min = ys[::decimation][y_indices[0]] - 2*decimation*resolution[0]*coarsen[0]
            y_max = ys[::decimation][y_indices[-1]] + 2*decimation*resolution[0]*coarsen[0]
            x_min = xs[::decimation][x_indices[0]] - 2*decimation*resolution[1]*coarsen[1]
            x_max = xs[::decimation][x_indices[-1]] + 2*decimation*resolution[1]*coarsen[1]
            ys = ys[(ys>=y_min)&(ys<=y_max)]
            xs = xs[(xs>=x_min)&(xs<=x_max)]

        # compute for the radar extent
        rae = compute_trans(ys, xs, coarsen, epsg, borders)

        # transform to separate variables
        trans = xr.Dataset({val: xr.DataArray(rae[key].round(4), coords={'y': ys,'x': xs}, dims=['y', 'x'])
                          for (key, val) in llt2ratlook_map.items()})

        # scale to integers for better compression
        # use explicit int32 fill to avoid architecture-dependent float->int overflow behavior
        fill_value = np.int32(np.iinfo(np.int32).max)
        for varname in ['azi', 'rng', 'ele']:
            scaled = (scale_factor * trans[varname]).round()
            mask = np.isfinite(scaled)
            # convert to int32 first, then fill NaN positions
            int_data = scaled.fillna(0).astype(np.int32)
            int_data = int_data.where(mask, fill_value)
            trans[varname] = int_data
            trans[varname].attrs['scale_factor'] = 1/scale_factor
            trans[varname].attrs['add_offset'] = 0
            # _FillValue will be set in encoding, not attrs (xarray requirement)

        # add georeference attributes
        trans = self.spatial_ref(trans, epsg)
        trans.attrs['spatial_ref'] = trans.spatial_ref.attrs['spatial_ref']
        trans = trans.drop_vars('spatial_ref')
        for var in list(trans.data_vars):
            trans[var].attrs.pop('grid_mapping', None)

        # use a single chunk for efficient storage
        shape = (trans.y.size, trans.x.size)
        encoding = {var: {'chunks': shape} for var in trans.data_vars}
        # set zarr fill_value to match _FillValue for proper masking on read
        for varname in ['azi', 'rng', 'ele']:
            encoding[varname]['_FillValue'] = np.iinfo(np.int32).max
        trans.to_zarr(
            store=os.path.join(outdir, 'transform'),
            mode='w',
            zarr_format=3,
            consolidated=True,
            encoding=encoding
        )
        del trans
