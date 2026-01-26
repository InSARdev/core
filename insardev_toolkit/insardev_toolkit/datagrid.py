# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------

class datagrid:
    import numpy as np

    # NetCDF options, see https://docs.xarray.dev/en/stable/user-guide/io.html#zarr-compressors-and-filters
    # Use netcdf4 engine for better compatibility with parallel/concurrent access (joblib workers)
    netcdf_engine_read: str = 'netcdf4'
    netcdf_engine_write: str = 'netcdf4'
    netcdf_format: str = 'NETCDF4'
    netcdf_chunksize: int = 1280
    netcdf_compression_algorithm: str = 'zlib'
    netcdf_complevel: int = 3
    netcdf_shuffle: bool = True

    # define lost class variables due to joblib via arguments
    def get_encoding_netcdf(self, shape=None, chunksize=None):
        """
        Return the compression options for a data grid.

        Parameters
        ----------
        shape : tuple, list, np.ndarray, optional
            The shape of the data grid. Required if chunksize is less than grid dimension sizes. Default is None.
        chunksize : int or tuple, optional
            The chunk size for data compression. If not specified, the class attribute chunksize is used.

        Returns
        -------
        dict
            A dictionary containing the compression options for the data grid.

        Examples
        --------
        Get the compression options for a data grid with shape (1000, 1000):

        >>> get_encoding_netcdf(shape=(1000, 1000))
        {'zlib': True, 'complevel': 3, 'chunksizes': (512, 512)}

        Get the compression options for a data grid with chunksize 256:

        >>> get_encoding_netcdf(chunksize=256)
        {'zlib': True, 'complevel': 3, 'chunksizes': (256, 256)}
        """
        import numpy as np

        if chunksize is None and len(shape) == 1:
            # (stacked) single-dimensional grid 
            chunksize = self.netcdf_chunksize1d
        elif chunksize is None:
            # common 2+D grid
            chunksize = self.netcdf_chunksize

        assert chunksize is not None, 'compression() chunksize is None'
        if isinstance(chunksize, (tuple, list, np.ndarray)):
            # use as is, it can be 2D or 3D grid (even 1D while it is not used for now)
            if shape is not None:
                assert len(shape) == len(chunksize), f'ERROR: defined shape and chunksize dimensions are not equal: {len(shape)} != {len(chunksize)}'
                chunksizes = tuple([chunksize[dim] if chunksize[dim]<shape[dim] else shape[dim] for dim in range(len(shape))])
            else:
                chunksizes = chunksize
        else:
            if shape is not None:
                # 2D or 3D grid
                chunksizes = []
                for idim in range(len(shape)):
                    chunksizes.append(chunksize if chunksize<shape[idim] else shape[idim])
                # set first dimension chunksize to 1 for 3D array
                if len(chunksizes) == 3:
                    chunksizes[0] = 1
                chunksizes = tuple(chunksizes)
            else:
                chunksizes=(chunksize, chunksize)
        opts = dict(chunksizes=chunksizes)
        if self.netcdf_compression_algorithm is not None and self.netcdf_complevel >= 0:
            opts[self.netcdf_compression_algorithm] = True
            opts['complevel'] = self.netcdf_complevel
            opts['shuffle'] = self.netcdf_shuffle
        return opts


    @staticmethod
    def get_spacing(data, coarsen=None):
        import numpy as np
        if isinstance(data, (dict)):
            da = next(iter(data.values()))
        elif isinstance(data, (list, tuple)):
            da = data[0]
        else:
            da = data
        if coarsen is None:
            coarsen = (1, 1)
        if not isinstance(coarsen, (list, tuple, np.ndarray)):
            coarsen = (coarsen, coarsen)
        #print ('get_spacing', da)
        dy = da.y.diff('y').item(0)
        dx = da.x.diff('x').item(0)
        if coarsen is not None:
            dy *= coarsen[0]
            dx *= coarsen[1]
        return (dy, dx)

    @staticmethod
    def spatial_ref(da, target=None):
        """
        Add geospatial attributes (CRS and spatial dimensions) to allow raster operations using RioXarray.

        Parameters
        ----------
        da : xarray.DataArray or xarray.Dataset
            The input 2D or 3D grid to be converted to geospatial.
        target : int, xarray.DataArray, or xarray.Dataset, optional
            The target EPSG code or an xarray object from which to derive the CRS.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The geospatial 2D or 3D grid with spatial attributes.

        Examples
        --------
        Convert a raster to geospatial and mask it using a Shapely vector geometry:
        Stack.spatial_ref(grid).rio.clip([geometry])
        """
        import xarray as xr
        import rioxarray
        import sys
        #assert 'rioxarray' in sys.modules, 'rioxarray module is not found'

        if target is None:
            return da
        
        if isinstance(target, (list, tuple)):
            target = target[0]
        elif isinstance(target, dict):
            target = next(iter(target.values()))

        # extract EPSG from target xarray object or use provided EPSG
        if isinstance(target, (xr.DataArray, xr.Dataset)):
            epsg = None
            # first try rioxarray CRS
            if target.rio.crs is not None:
                epsg = target.rio.crs.to_epsg()
            else:
                # fallback: try to get EPSG from spatial_ref (check both coords and data_vars)
                spatial_ref_var = None
                if 'spatial_ref' in target.coords:
                    spatial_ref_var = target.coords['spatial_ref']
                elif isinstance(target, xr.Dataset) and 'spatial_ref' in target.data_vars:
                    spatial_ref_var = target.data_vars['spatial_ref']
                
                if spatial_ref_var is not None:
                    spatial_ref_attrs = spatial_ref_var.attrs
                    # try crs_wkt first, then spatial_ref attr
                    crs_wkt = spatial_ref_attrs.get('crs_wkt') or spatial_ref_attrs.get('spatial_ref')
                    if crs_wkt:
                        from rasterio.crs import CRS
                        try:
                            epsg = CRS.from_wkt(crs_wkt).to_epsg()
                        except Exception:
                            epsg = None
            
            if epsg is None:
                print('WARNING: Target xarray object has no CRS defined.')
                return da
        else:
            epsg = target
        #print ('spatial_ref: target epsg', epsg)

        if epsg == 4326:
            # EPSG:4326 (WGS84, lat/lon)
            da_spatial = (
                da.rio.write_crs(4326)
                  .rio.set_spatial_dims(y_dim='lat', x_dim='lon')
                  #.rio.write_grid_mapping()
                  .assign_coords(lat=da.lat.assign_attrs(axis='Y', 
                                                       standard_name='latitude',
                                                       long_name='latitude'),
                               lon=da.lon.assign_attrs(axis='X', 
                                                       standard_name='longitude',
                                                       long_name="longitude"))
                  .assign_attrs({'Conventions': 'CF-1.8'}))
        else:
            # projected coordinates
            da_spatial = (
                da.rio.write_crs(epsg)
                  .rio.set_spatial_dims(y_dim='y', x_dim='x')
                  #.rio.write_grid_mapping()
                  .assign_coords(y=da.y.assign_attrs(axis='Y', 
                                                   standard_name='projection_y_coordinate',
                                                   long_name='northing'),
                               x=da.x.assign_attrs(axis='X', 
                                                   standard_name='projection_x_coordinate',
                                                   long_name="easting"))
                  .assign_attrs({'Conventions': 'CF-1.8'}))
    
        return da_spatial.assign_attrs(coordinates='spatial_ref')

    @staticmethod
    def get_bounds(geometry, epsg=4326):
        import geopandas as gpd
        import xarray as xr
        from shapely.geometry import Polygon
    
        if isinstance(geometry, (xr.DataArray, xr.Dataset)) and ('lat' in geometry.dims and 'lon' in geometry.dims):
            # WGS84 coordinates expected
            lon_start = geometry.lon.min().item()
            lat_start = geometry.lat.min().item()
            lon_end   = geometry.lon.max().item()
            lat_end   = geometry.lat.max().item()
            bounds = lon_start, lat_start, lon_end, lat_end
        elif isinstance(geometry, (xr.DataArray, xr.Dataset)):
            # x_start = geometry.x.min().item()
            # y_start = geometry.y.min().item()
            # x_end   = geometry.x.max().item()
            # y_end   = geometry.y.max().item()
            # bounds = x_start, y_start, x_end, y_end
            xmin, xmax = float(geometry.x.min()), float(geometry.x.max())
            ymin, ymax = float(geometry.y.min()), float(geometry.y.max())
            corners = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
            geom = gpd.GeoDataFrame({'geometry': [Polygon(corners)]})
            if epsg is not None:
                try:
                    epsg_code = int(geometry.rio.crs.to_epsg())
                    geom = geom.set_crs(epsg_code).to_crs(epsg)
                except Exception:
                    pass
            bounds = geom.dissolve().envelope.item().bounds
        elif isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            #print ('Geometry is a GeoDataFrame')
            geom = geometry
            if epsg is not None:
                try:
                    geom = geometry.to_crs(epsg)
                except:
                    pass
            bounds = geom.union_all().envelope.bounds if isinstance(geometry, gpd.GeoSeries) else geom.dissolve().envelope.item().bounds 
        elif isinstance(geometry, tuple):
            # geometry is already bounds
            bounds = geometry
        else:
            bounds = geometry.bounds
        #print ('bounds', bounds)
        #lon_start, lat_start, lon_end, lat_end
        return bounds
