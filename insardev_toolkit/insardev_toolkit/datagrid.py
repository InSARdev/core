# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------

import threading

_MEM_ENV = threading.local()


def _mem_open_env():
    """The rasterio Env of this thread that lets GDAL open a MEM connection string (GDAL 3.10 and later refuse it
    otherwise), entered ONCE per thread and kept: entering one around every warp costs about 10 microseconds, a few percent
    of a small block's warp. Its only option is that one, set for this thread only."""
    if getattr(_MEM_ENV, 'env', None) is None:
        import rasterio
        env = rasterio.Env(GDAL_MEM_ENABLE_OPEN='YES')
        env.__enter__()
        _MEM_ENV.env = env


class datagrid:
    import numpy as np


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
                da.copy(deep=False).rio.write_crs(epsg, inplace=True)
                  .rio.set_spatial_dims(y_dim='y', x_dim='x', inplace=True)
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
    def reproject(da, crs, resampling='nearest', chunks=2048):
        """
        Reproject a 2D grid to the CRS crs lazily, on the grid of rio.reproject(crs): as rio.reproject() but block by
        block, where rio.reproject() reads the whole grid and warps it in memory.

        One dask task per output block warps the input blocks under it (the block's bounds in the input CRS, densified,
        and one input pixel more), assembled, with rasterio.warp.reproject(), as odc-geo's dask reprojection does. The
        warp is GDAL's, with its approximate transformer of 0.125 pixel, so a few pixels near the input pixel edges
        can take the neighbour rio.reproject() takes on its own split of the grid. The result is lazy: every compute
        that uses it warps its blocks again, and .compute() keeps the grid in memory.

        Parameters
        ----------
        da : xarray.DataArray
            2D grid with a CRS (rio.crs), dask-backed: its blocks are the inputs of the output blocks.
        crs : int, str or CRS
            Output CRS.
        resampling : str, optional
            rasterio.enums.Resampling name. Default 'nearest'.
        chunks : int, optional
            Output block size on both axes. Default 2048.

        Returns
        -------
        xarray.DataArray
            The lazy (y, x) grid with its CRS and nodata, NaN (0 for an integer grid) without an input nodata.

        Examples
        --------
        land = Tiles().reproject(Tiles().open('land.vrt'), stack.crs)
        """
        import numpy as np
        import dask.array as dask_array
        import xarray as xr
        import rioxarray
        import rasterio.warp
        from uuid import uuid4
        from affine import Affine
        from dask.highlevelgraph import HighLevelGraph
        from rasterio.crs import CRS
        from rasterio.enums import Resampling

        if not hasattr(da.data, 'dask'):
            da = da.chunk(chunks)
        src_crs = da.rio.crs
        dst_crs = CRS.from_user_input(crs)
        src_transform = da.rio.transform()
        ny, nx = da.shape
        # THE DTYPE, NOT THE GRID, in the task function below: `da` in its closure was pickled with every task, a
        # grid in memory included (1.9 MB of source gave a 1.9 MB function)
        dtype = da.dtype
        nodata = da.rio.nodata
        if nodata is None:
            nodata = np.nan if np.issubdtype(dtype, np.floating) else 0
        # the grid of rio.reproject(crs)
        dst_transform, width, height = rasterio.warp.calculate_default_transform(src_crs, crs, nx, ny, *da.rio.bounds())
        method = getattr(Resampling, resampling)
        data = da.data
        ychunks, xchunks = data.chunks
        yedges, xedges = np.cumsum((0,) + ychunks), np.cumsum((0,) + xchunks)
        out_y = tuple(min(chunks, height - i) for i in range(0, height, chunks))
        out_x = tuple(min(chunks, width - i) for i in range(0, width, chunks))
        name = f'reproject-{uuid4().hex}'
        keys = data.__dask_keys__()
        gdal_type = {'uint8': 'Byte', 'int8': 'Int8', 'uint16': 'UInt16', 'int16': 'Int16', 'uint32': 'UInt32', 'int32': 'Int32',
                     'uint64': 'UInt64', 'int64': 'Int64', 'float32': 'Float32', 'float64': 'Float64',
                     'complex64': 'CFloat32', 'complex128': 'CFloat64'}.get(np.dtype(dtype).name)
        if gdal_type is None:
            raise TypeError(f'reproject() does not support the grid dtype {dtype}: convert it to a numeric type, e.g. astype("uint8").')

        def mem_string(array, transform):
            """The GDAL connection string of a C-contiguous 2D numpy array wrapped as a MEM dataset, with its transform."""
            h, w = array.shape
            return (f'MEM:::DATAPOINTER={array.ctypes.data},PIXELS={w},LINES={h},BANDS=1,DATATYPE={gdal_type},'
                    f'GEOTRANSFORM={"/".join(repr(float(v)) for v in transform.to_gdal())}')

        def warp(window, block_transform, shape, *blocks):
            (r0, r1), (c0, c1), rows, cols = window
            grid = np.block([[blocks[i * len(cols) + j] for j in range(len(cols))] for i in range(len(rows))])
            # the assembled input blocks start at the first block's row and column
            source = np.ascontiguousarray(grid[r0 - yedges[rows[0]]:r1 - yedges[rows[0]], c0 - xedges[cols[0]]:c1 - xedges[cols[0]]])
            out = np.full(shape, nodata, dtype=dtype)
            # Both grids are GDAL MEM datasets that WRAP the arrays (no copy) and carry their transform from the moment
            # they are opened (the connection string), passed to rasterio.warp.reproject as bands. Given numpy arrays,
            # rasterio builds a MemoryDataset itself, which reads its transform before it has one, warns by design and
            # silences the warning with warnings.catch_warnings(): not thread-safe, so with 2 or more threads per worker
            # the warning escapes. No lock and no warning filter: nothing here warns, and the warps run in parallel.
            _mem_open_env()
            with rasterio.open(mem_string(source, src_transform * Affine.translation(c0, r0)), 'r+') as src, \
                    rasterio.open(mem_string(out, block_transform), 'r+') as dst:
                src.crs, dst.crs = src_crs, dst_crs
                rasterio.warp.reproject(rasterio.band(src, 1), rasterio.band(dst, 1), resampling=method,
                                        src_nodata=nodata, dst_nodata=nodata)
            return out

        dsk = {}
        inverse = ~src_transform
        for i, (y0, by) in enumerate(zip(np.cumsum((0,) + out_y[:-1]), out_y)):
            for j, (x0, bx) in enumerate(zip(np.cumsum((0,) + out_x[:-1]), out_x)):
                block_transform = dst_transform * Affine.translation(x0, y0)
                left, top = block_transform * (0, 0)
                right, bottom = block_transform * (bx, by)
                # the block's bounds in the input CRS, densified along its edges, one input pixel wider
                w, s, e, n = rasterio.warp.transform_bounds(crs, src_crs, min(left, right), min(bottom, top),
                                                            max(left, right), max(bottom, top), densify_pts=21)
                cols_f, rows_f = zip(*(inverse * (x, y) for x, y in ((w, n), (e, n), (w, s), (e, s))))
                r0, r1 = max(0, int(np.floor(min(rows_f))) - 1), min(ny, int(np.ceil(max(rows_f))) + 1)
                c0, c1 = max(0, int(np.floor(min(cols_f))) - 1), min(nx, int(np.ceil(max(cols_f))) + 1)
                if r1 <= r0 or c1 <= c0:
                    # no input under the block
                    dsk[(name, i, j)] = (np.full, (by, bx), nodata, dtype)
                    continue
                rows = [k for k in range(len(ychunks)) if yedges[k] < r1 and yedges[k + 1] > r0]
                cols = [k for k in range(len(xchunks)) if xedges[k] < c1 and xedges[k + 1] > c0]
                deps = [keys[r][c] for r in rows for c in cols]
                dsk[(name, i, j)] = (warp, ((r0, r1), (c0, c1), rows, cols), block_transform, (by, bx), *deps)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=(data,))
        out = dask_array.Array(graph, name, chunks=(out_y, out_x), dtype=dtype)
        # the pixel centres of the output grid
        xs = (dst_transform * (np.arange(width) + 0.5, np.full(width, 0.5)))[0]
        ys = (dst_transform * (np.full(height, 0.5), np.arange(height) + 0.5))[1]
        result = xr.DataArray(out, coords={'y': ys, 'x': xs}, dims=('y', 'x'), name=da.name, attrs=da.attrs)
        return result.rio.write_crs(crs).rio.write_nodata(nodata)

    @staticmethod
    def get_bounds(geometry, epsg=4326):
        import geopandas as gpd
        import xarray as xr
        from shapely.geometry import Polygon
    
        # an S1 or Nisar object holds its records in .df, accept it as the geometry itself.
        # Checked on .df rather than on a to_dataframe() method, which Xarray objects have too.
        if isinstance(getattr(geometry, 'df', None), (gpd.GeoDataFrame, gpd.GeoSeries)):
            geometry = geometry.df
    
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
