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
    def suggested_grid(src_crs, src_transform, nx, ny, dst_crs):
        """
        The output grid of a reprojection of an nx x ny grid with the affine src_transform from src_crs to dst_crs, by the rule
        GDAL documents for GDALSuggestedWarpOutput2 (https://gdal.org/en/stable/api/gdal_alg.html): about 20 points are
        sampled along each edge of the source, the sampled points are transformed to dst_crs and their extent is the output
        extent; the output pixels are square, and their size is chosen so that the diagonal from the top left to the bottom
        right corner of the output spans the same number of pixels as the diagonal of the source; the width and height are the
        extent over the pixel size, rounded to the nearest integer. The transformation is a pyproj Transformer.

        Parameters
        ----------
        src_crs : rasterio.crs.CRS
            Source CRS.
        src_transform : affine.Affine
            Source affine transform of the cell edges.
        nx : int
            Source width in cells.
        ny : int
            Source height in cells.
        dst_crs : rasterio.crs.CRS
            Output CRS.

        Returns
        -------
        tuple
            (transform, width, height): the output affine transform (square pixels, origin at the top left corner of the
            extent) and the output width and height in pixels.

        Examples
        --------
        transform, width, height = datagrid.suggested_grid(src_crs, src_transform, 600, 600, dst_crs)
        """
        import numpy as np
        from affine import Affine
        from pyproj import Transformer

        to_dst = Transformer.from_crs(src_crs.to_wkt(), dst_crs.to_wkt(), always_xy=True)
        # the cell edges sampled at 21 points each, corners included, as (column, row) of the source: the top edge (indices
        # 0-20), the bottom edge (21-41), the left (42-62) and the right (63-83)
        u, v = np.linspace(0.0, nx, 21), np.linspace(0.0, ny, 21)
        cols = np.concatenate([u, u, np.zeros(21), np.full(21, float(nx))])
        rows = np.concatenate([np.zeros(21), np.full(21, float(ny)), v, v])
        X, Y = src_transform * (cols, rows)
        xs, ys = to_dst.transform(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not finite.any():
            raise ValueError('ERROR: no edge of the source grid transforms to the output CRS.')
        left, right = xs[finite].min(), xs[finite].max()
        bottom, top = ys[finite].min(), ys[finite].max()
        # the top left corner (column 0, row 0) is index 0, the bottom right corner (column nx, row ny) is index 41
        pixel = np.hypot(xs[41] - xs[0], ys[41] - ys[0]) / np.hypot(nx, ny)
        if not (np.isfinite(pixel) and pixel > 0):
            raise ValueError('ERROR: the corners of the source grid do not give a pixel size in the output CRS.')
        width = max(1, int(np.floor((right - left) / pixel + 0.5)))
        height = max(1, int(np.floor((top - bottom) / pixel + 0.5)))
        return Affine(pixel, 0.0, left, 0.0, -pixel, top), width, height

    @staticmethod
    def reproject(da, crs, resampling='nearest', chunks=2048):
        """
        Reproject a grid to the CRS crs lazily, on the grid of suggested_grid() (GDAL's suggested output, computed with pyproj),
        block by block, where rio.reproject() reads the whole grid and warps it in memory.

        The grid is 2D (y, x), or N-D with the leading dimensions before (y, x), e.g. (term, row, part, y, x): every slice
        of the leading dimensions is reprojected with the same sampling, so the transform and the sampling indices are
        computed once per output block whatever the number of slices. An N-D result equals the stack of the 2D results of
        its slices bitwise.

        Every output pixel centre is transformed exactly to the input CRS with a pyproj Transformer, turned into a
        fractional input row and column by the input affine, and sampled: 'nearest' takes the input cell that contains
        it, 'bilinear' interpolates the four input cells around it, by their centres. No GDAL warp is used, so the values
        do not depend on the block size: chunks=2048 and chunks=32 give the same grid. A pixel outside the input is NaN
        (0 for an integer grid without a nodata value); with 'bilinear' so is a pixel whose four cells do not all lie in
        the input or one of which is nodata. Complex grids (complex64, complex128) are sampled the same way, bilinear on their
        complex values.

        One dask task per output block reads the input blocks its sampling can reach (the input blocks under the block's
        bounds in the input CRS, with two input cells more on each side), transforms the block's pixel centres once and
        samples the slices in strips of rows. The result is lazy: every compute that uses it samples its blocks again,
        and .compute() keeps the grid in memory.

        Parameters
        ----------
        da : xarray.DataArray
            2D grid (y, x) or N-D grid with the leading dimensions before (y, x), with a CRS (rio.crs), dask-backed: its
            blocks are the inputs of the output blocks.
        crs : int, str or CRS
            Output CRS.
        resampling : str, optional
            'nearest' or 'bilinear'. Default 'nearest'.
        chunks : int, optional
            Output block size on both axes. Default 2048.

        Returns
        -------
        xarray.DataArray
            The lazy grid with its leading dimensions and (y, x), its CRS and nodata, NaN (0 for an integer grid) without
            an input nodata.

        Examples
        --------
        land = Tiles().reproject(Tiles().open('land.vrt'), stack.crs)
        """
        import numpy as np
        import dask.array as dask_array
        import xarray as xr
        import rioxarray
        from uuid import uuid4
        from dask.highlevelgraph import HighLevelGraph
        from pyproj import Transformer
        from rasterio.crs import CRS

        if resampling not in ('nearest', 'bilinear'):
            raise ValueError(f"ERROR: reproject() supports the resampling 'nearest' and 'bilinear', not '{resampling}'")
        if da.ndim < 2:
            raise ValueError('reproject() needs a grid of at least two dimensions (y, x)')
        if not hasattr(da.data, 'dask'):
            da = da.chunk(chunks)
        src_crs = da.rio.crs
        dst_crs = CRS.from_user_input(crs)
        src_transform = da.rio.transform()
        ny, nx = da.shape[-2:]
        lead = da.shape[:-2]
        nlead = len(lead)
        if resampling == 'bilinear' and (nx < 2 or ny < 2):
            raise ValueError('reproject() bilinear needs an input of at least 2 x 2 cells')
        # THE DTYPE, NOT THE GRID, in the task function below: `da` in its closure was pickled with every task, a grid in
        # memory included (1.9 MB of source gave a 1.9 MB function)
        dtype = da.dtype
        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.inexact)):
            raise TypeError(f'reproject() does not support the grid dtype {dtype}: convert it to a numeric type, e.g. astype("uint8").')
        nodata = da.rio.nodata
        if nodata is None:
            nodata = np.nan if np.issubdtype(dtype, np.inexact) else 0
        nodata_is_nan = bool(np.issubdtype(dtype, np.inexact) and np.isnan(nodata))
        # the output grid: GDAL's suggested output, computed with pyproj
        dst_transform, width, height = datagrid.suggested_grid(src_crs, src_transform, nx, ny, dst_crs)
        inverse = ~src_transform
        # every leading slice in one input block: an output task holds all the slices of its block
        data = da.data.rechunk({i: -1 for i in range(nlead)}) if nlead else da.data
        ychunks, xchunks = data.chunks[nlead:]
        yedges, xedges = np.cumsum((0,) + ychunks), np.cumsum((0,) + xchunks)
        out_y = tuple(min(chunks, height - i) for i in range(0, height, chunks))
        out_x = tuple(min(chunks, width - i) for i in range(0, width, chunks))
        name = f'reproject-{uuid4().hex}'
        keys = data.__dask_keys__()
        for _ in range(nlead):
            keys = keys[0]
        src_wkt, dst_wkt = src_crs.to_wkt(), dst_crs.to_wkt()
        to_dst = Transformer.from_crs(src_wkt, dst_wkt, always_xy=True)
        # the bounds in the output CRS of every input block, padded by two input cells: a pixel samples its cell and the
        # neighbours of that cell, so its input block is one of these
        boxes = np.empty((len(ychunks), len(xchunks), 4))
        for r in range(len(ychunks)):
            for c in range(len(xchunks)):
                left, top = src_transform * (xedges[c] - 2, yedges[r] - 2)
                right, bottom = src_transform * (xedges[c + 1] + 2, yedges[r + 1] + 2)
                w, s, e, n = to_dst.transform_bounds(min(left, right), min(bottom, top), max(left, right), max(bottom, top), densify_pts=21)
                # a block the transform cannot place is read by every output block
                boxes[r, c] = (w, s, e, n) if np.all(np.isfinite((w, s, e, n))) else (-np.inf, -np.inf, np.inf, np.inf)
        STRIP = 512   # output rows of the sampled slices computed at once: the memory of a block is that of its strips

        def is_nodata(values):
            return np.isnan(values) if nodata_is_nan else values == nodata

        def check(index, size):
            """A sampled cell outside the input blocks the task reads is a defect: raised, never a silent nodata."""
            if index.size and (index.min() < 0 or index.max() >= size):
                raise RuntimeError('reproject() samples input cells outside the input blocks its task reads')

        def warp(meta, *blocks):
            """The output block of meta = (first input row, first input column, input block rows, input block columns, first
            output row, first output column, rows, columns) from the input blocks: the sampling of the whole block once, then
            its slices in strips of rows."""
            r0, c0, nr, nc, y0, x0, by, bx = meta
            source = np.block([[blocks[i * nc + j] for j in range(nc)] for i in range(nr)])
            lead = source.shape[:-2]
            sh, sw = source.shape[-2:]
            to_source = Transformer.from_crs(dst_wkt, src_wkt, always_xy=True)
            # the pixel centres of the block, transformed once
            cols = np.arange(x0, x0 + bx) + 0.5
            rows = np.arange(y0, y0 + by) + 0.5
            X, Y = dst_transform * (cols[None, :], rows[:, None])
            xs, ys = to_source.transform(np.ascontiguousarray(X).ravel(), np.ascontiguousarray(Y).ravel())
            # fractional input column and row: the cell (i, j) spans [j, j + 1) x [i, i + 1), its centre is at j + 0.5
            cf, rf = inverse * (np.asarray(xs), np.asarray(ys))
            cf, rf = cf.reshape(by, bx), rf.reshape(by, bx)
            if resampling == 'nearest':
                inside = (cf >= 0) & (cf < nx) & (rf >= 0) & (rf < ny)
                ci = np.floor(np.where(inside, cf, c0)).astype(np.intp) - c0
                ri = np.floor(np.where(inside, rf, r0)).astype(np.intp) - r0
                check(ci[inside], sw)
                check(ri[inside], sh)
            else:
                # the cell centres sit at j + 0.5: the four cells around the point, all of them in the input
                u, w = cf - 0.5, rf - 0.5
                inside = (u >= 0) & (u <= nx - 1) & (w >= 0) & (w <= ny - 1)
                u, w = np.where(inside, u, 0.0), np.where(inside, w, 0.0)
                gc, gr = np.minimum(np.floor(u), nx - 2), np.minimum(np.floor(w), ny - 2)
                tx, ty = u - gc, w - gr
                ci = np.where(inside, gc.astype(np.intp) - c0, 0)
                ri = np.where(inside, gr.astype(np.intp) - r0, 0)
                check(ci[inside], sw - 1)
                check(ri[inside], sh - 1)
            out = np.full(lead + (by, bx), nodata, dtype=dtype)
            acc = np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
            for s in range(0, by, STRIP):
                e = min(by, s + STRIP)
                if resampling == 'nearest':
                    val = np.where(inside[s:e], source[..., ri[s:e], ci[s:e]], nodata)
                else:
                    v00, v01 = source[..., ri[s:e], ci[s:e]], source[..., ri[s:e], ci[s:e] + 1]
                    v10, v11 = source[..., ri[s:e] + 1, ci[s:e]], source[..., ri[s:e] + 1, ci[s:e] + 1]
                    missing = is_nodata(v00) | is_nodata(v01) | is_nodata(v10) | is_nodata(v11)
                    f00, f01, f10, f11 = (v.astype(acc) for v in (v00, v01, v10, v11))
                    txs, tys = tx[s:e], ty[s:e]
                    val = (1 - tys) * ((1 - txs) * f00 + txs * f01) + tys * ((1 - txs) * f10 + txs * f11)
                    if np.issubdtype(dtype, np.integer):
                        val = np.rint(val)
                    val = np.where(inside[s:e] & ~missing, val, nodata)
                out[..., s:e, :] = val.astype(dtype)
            return out

        dsk = {}
        for i, (y0, by) in enumerate(zip(np.cumsum((0,) + out_y[:-1]), out_y)):
            for j, (x0, bx) in enumerate(zip(np.cumsum((0,) + out_x[:-1]), out_x)):
                key = (name,) + (0,) * nlead + (i, j)
                # the bounds of the output block in its CRS: its pixel edges
                xa, ya = dst_transform * (x0, y0)
                xb, yb = dst_transform * (x0 + bx, y0 + by)
                hit = ((boxes[..., 0] <= max(xa, xb)) & (boxes[..., 2] >= min(xa, xb)) &
                       (boxes[..., 1] <= max(ya, yb)) & (boxes[..., 3] >= min(ya, yb)))
                if not hit.any():
                    # no input under the block
                    dsk[key] = (np.full, lead + (by, bx), nodata, dtype)
                    continue
                rows, cols = np.nonzero(hit.any(axis=1))[0], np.nonzero(hit.any(axis=0))[0]
                r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                deps = [keys[r][c] for r in range(r0, r1) for c in range(c0, c1)]
                meta = (int(yedges[r0]), int(xedges[c0]), r1 - r0, c1 - c0, int(y0), int(x0), int(by), int(bx))
                dsk[key] = (warp, meta, *deps)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=(data,))
        out = dask_array.Array(graph, name, chunks=tuple((n,) for n in lead) + (out_y, out_x), dtype=dtype)
        # the pixel centres of the output grid
        xs = (dst_transform * (np.arange(width) + 0.5, np.full(width, 0.5)))[0]
        ys = (dst_transform * (np.full(height, 0.5), np.arange(height) + 0.5))[1]
        coords = {d: da.coords[d].values for d in da.dims[:-2] if d in da.coords and da.coords[d].dims == (d,)}
        coords.update({'y': ys, 'x': xs})
        result = xr.DataArray(out, coords=coords, dims=tuple(da.dims[:-2]) + ('y', 'x'), name=da.name, attrs=da.attrs)
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
