# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_tidal import S1_tidal

class S1_dem(S1_tidal):
    import geopandas as gpd
    import xarray as xr
    import pandas as pd

    # Xarray's interpolation can be inefficient for large grids;
    # this custom function handles the task more effectively.
    def _interp2d_like(self, data, grid, method='cubic', **kwargs):
        import numpy as np
        import xarray as xr
        import warnings
        warnings.filterwarnings('ignore')

        # detect dimensions and coordinates for 2D or 3D grid
        dims = grid.dims[-2:]
        dim1, dim2 = dims
        coords = {dim1: grid[dim1], dim2: grid[dim2]}

        # get coordinate values
        out_coords1 = grid[dim1].values
        out_coords2 = grid[dim2].values

        # add buffer around for interpolation
        d1 = float(data[dim1].diff(dim1)[0])
        d2 = float(data[dim2].diff(dim2)[0])

        # select data subset with buffer
        data_subset = data.sel({
            dim1: slice(out_coords1[0]-2*d1, out_coords1[-1]+2*d1),
            dim2: slice(out_coords2[0]-2*d2, out_coords2[-1]+2*d2)
        })

        # interpolate
        result = data_subset.interp({dim1: out_coords1, dim2: out_coords2}, method=method, **kwargs)

        da_out = xr.DataArray(result.values, coords=coords, dims=dims).rename(data.name).astype(np.float32)
        # append all the input coordinates
        return da_out.assign_coords({k: v for k, v in data.coords.items() if k not in coords})

    def get_geoid(self, grid: xr.DataArray|xr.Dataset=None) -> xr.DataArray:
        """
        Get EGM96 geoid heights.

        Parameters
        ----------
        grid : xarray array or dataset, optional
            Interpolate geoid heights on the grid. Default is None.

        Returns
        -------
        xr.DataArray
            Geoid heights in meters.

        Examples
        --------
        stack.get_geoid()

        Notes
        -----
        See EGM96 geoid heights on http://icgem.gfz-potsdam.de/tom_longtime
        """
        from .utils_satellite import get_geoid
        return get_geoid(grid)

    def get_dem(self, geometry: gpd.GeoDataFrame=None, buffer_degrees: float=0):
        """
        Load and preprocess digital elevation model (DEM) data from specified datafile or variable.

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
        import geopandas as gpd
        import pandas as pd
        import os

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not specified.')

        if geometry is None:
            geometry = self.df

        if isinstance(self.DEM, (xr.Dataset)):
            ortho = self.DEM[list(self.DEM.data_vars)[0]]
        elif isinstance(self.DEM, (xr.DataArray)):
            ortho = self.DEM
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.tiff', '.tif', '.TIF']:
            ortho = rio.open_rasterio(self.DEM).squeeze(drop=True)\
                .rename({'y': 'lat', 'x': 'lon'})\
                .drop('spatial_ref')
            if ortho.lat.diff('lat')[0].item() < 0:
                ortho = ortho.reindex(lat=ortho.lat[::-1])
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.nc', '.netcdf', '.grd']:
            ortho = xr.open_dataarray(self.DEM, engine=self.netcdf_engine_read)
        elif isinstance(self.DEM, str):
            print ('ERROR: filename extension is not recognized. Should be one from .tiff, .tif, .TIF, .nc, .netcdf, .grd')
        else:
            print ('ERROR: argument is not an Xarray object and it is not a file name')
        ortho = ortho.transpose('lat','lon')

        # unique indices required for interpolation
        lat_index = pd.Index(ortho.coords['lat'])
        lon_index = pd.Index(ortho.coords['lon'])
        duplicates = lat_index[lat_index.duplicated()].tolist() + lon_index[lon_index.duplicated()].tolist()
        assert len(duplicates) == 0, 'ERROR: DEM grid includes duplicated coordinates, possibly on merged tiles edges'

        # crop to the geometry extent
        bounds = self.get_bounds(geometry.buffer(buffer_degrees))
        ortho = ortho.sel(lat=slice(bounds[1], bounds[3]), lon=slice(bounds[0], bounds[2]))

        # preserve NaN for areas outside DEM coverage
        ds = ortho.astype(np.float32).transpose('lat','lon').rename("dem")
        return self.spatial_ref(ds, 4326)

    # buffer required to get correct (binary) results from SAT_llt2rat tool
    # small buffer produces incomplete area coverage and restricted NaNs
    # 0.02 degrees works well worldwide but not in Siberia
    # minimum buffer size: 8 arc seconds for 90 m DEM
    def get_dem_wgs84ellipsoid(self, geometry: gpd.GeoDataFrame=None, buffer_degrees: float=0.04):
        """
        Load and preprocess digital elevation model (DEM) data from specified datafile or variable.

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

        Examples
        --------
        Load and crop from local NetCDF file:
        stack.load_dem('GEBCO_2020/GEBCO_2020.nc')

        Load and crop from local GeoTIF file:
        stack.load_dem('GEBCO_2019.tif')

        Load from Xarray DataArray or Dataset:
        stack.set_dem(None).load_dem(dem)
        stack.set_dem(None).load_dem(dem.to_dataset())

        Notes
        -----
        This method loads DEM from the user specified file. The data is then preprocessed by removing
        the EGM96 geoid to make the heights relative to the WGS84 ellipsoid.
        """
        import numpy as np
        
        # DEM
        ortho = self.get_dem(geometry, buffer_degrees)
        # heights correction
        geoid = self.get_geoid(ortho)
        # suppose missed values are water surface
        ds = (ortho + geoid).rename("dem")
        return self.spatial_ref(ds, 4326)
