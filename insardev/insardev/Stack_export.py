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
from .Stack_ps import Stack_ps
from .utils_vtk import as_vtk as _as_vtk
from insardev_toolkit import progressbar

class Stack_export(Stack_ps):

    @staticmethod
    def as_vtk(dataset):
        # Wrapper retained for backward compatibility
        return _as_vtk(dataset)

    # def export_geotiff(self, data, name, caption='Exporting WGS84 GeoTIFF(s)', compress='LZW'):
    #     """
    #     Export the provided data to a GeoTIFF file.

    #     Parameters
    #     ----------
    #     data : xarray.DataArray
    #         The data to be exported as a GeoTIFF.
    #     name : str
    #         The base name for the GeoTIFF file(s).
    #     caption : str, optional
    #         A description for the export process, used for progress display. Default is 'Exporting WGS84 GeoTIFF(s)'.
    #     compress : str, optional
    #         The compression method to use for the GeoTIFF. Default is 'LZW'.

    #     Returns
    #     -------
    #     None
    #         The function writes the GeoTIFF file(s) to disk with the specified name.
            
    #     Examples
    #     --------
    #     Export a single GeoTIFF file "velocity.tif":
    #     sbas.export_geotiff(velocity, 'velocity')

    #     Export a stack of GeoTIFF files like corr.2024-01-01_2024-01-02.tif:
    #     sbas.export_geotiff(corr, 'corr')

    #     Export date-based stack of GeoTIFF files like disp.2024-01-01.tif:
    #     sbas.export_geotiff(disp, 'disp')
    #     """
    #     import xarray as xr
    #     import numpy as np
    #     from tqdm.auto import tqdm

    #     assert isinstance(data, xr.DataArray), 'Argument data is not an xr.DataArray object'

    #     # determine if data has a stack dimension and what it is
    #     stackvar = data.dims[0] if len(data.dims) == 3 else None

    #     # prepare the progress bar
    #     with tqdm(desc=caption, total=len(data[stackvar]) if stackvar is not None else 1) as pbar:
    #         for grid in data.transpose(stackvar, ...) if stackvar is not None else [data]:
    #             # convert the stack variable value to a string suitable for filenames
    #             if stackvar is not None and np.issubdtype(grid[stackvar].dtype, np.datetime64):
    #                 stackval = grid[stackvar].dt.date.item()
    #             elif stackvar is not None:
    #                 stackval = grid[stackvar].astype(str).item().replace(' ', '_')
    #             else:
    #                 stackval = ''
    #             #print ('stackval', stackval)
    #             filename = f'{name}.{stackval}.tif' if stackvar is not None else f'{name}.tif'
    #             #print ('filename', filename)
    #             # convert the data to geographic coordinates if necessary and export to GeoTIFF
    #             self.spatial_ref(grid).rio.to_raster(filename, compress=compress)
    #             pbar.update(1)

    # def export_geojson(self, data, name, caption='Exporting WGS84 GeoJSON', pivotal=True, digits=2, coord_digits=6):
    #     """
    #     Export the provided data to a GeoJSON file.

    #     Parameters
    #     ----------
    #     data : xarray.DataArray
    #         The data to be exported as GeoJSON.
    #     name : str
    #         The base name for the GeoJSON file(s).
    #     caption : str, optional
    #         A description for the export process, used for progress display. Default is 'Exporting WGS84 GeoJSON'.
    #     pivotal : bool, optional
    #         Whether to pivot the data. Default is True.
    #     digits : int, optional
    #         Number of decimal places to round the data values. Default is 2.
    #     coord_digits : int, optional
    #         Number of decimal places to round the coordinates. Default is 6.
    
    #     Returns
    #     -------
    #     None
    #         The function writes the GeoJSON file(s) to disk with the specified name.
            
    #     Examples
    #     --------
    #     Export a GeoJSON file "velocity.geojson":
    #     sbas.export_geojson(velocity, 'velocity')
    #     """
    #     import xarray as xr
    #     import geopandas as gpd
    #     import numpy as np
    #     from tqdm.auto import tqdm
    #     # disable "distributed.utils_perf - WARNING - full garbage collections ..."
    #     try:
    #         from dask.distributed import utils_perf
    #         utils_perf.disable_gc_diagnosis()
    #     except ImportError:
    #         from distributed.gc import disable_gc_diagnosis
    #         disable_gc_diagnosis()

    #     assert isinstance(data, xr.DataArray), 'Argument data is not an xr.DataArray object'

    #     # determine if data has a stack dimension and what it is
    #     stackvar = data.dims[0] if len(data.dims) == 3 else None

    #     # convert the data to geographic coordinates if necessary
    #     grid = data

    #     def block_as_json(block, stackvar, name):
    #         df = block.compute().to_dataframe().dropna().reset_index()
    #         df[name] = df[name].apply(lambda x: float(f"{x:.{digits}f}"))
    #         for col in df.columns:
    #             if np.issubdtype(df[col].dtype, np.datetime64):
    #                 df[col] = df[col].dt.date.astype(str)
    #         if stackvar is not None and np.issubdtype(df[stackvar].dtype, np.datetime64):
    #             df[stackvar] = df[stackvar].dt.date.astype(str)
    #         if stackvar is not None and pivotal:
    #             df = df.pivot_table(index=['lat', 'lon'], columns=stackvar,
    #                                 values=name, fill_value=np.nan).reset_index()
    #         # convert to geodataframe with value and geometry columns
    #         gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon.round(coord_digits), df.lat.round(coord_digits)))
    #         del df, gdf['lat'], gdf['lon']
    #         chunk_json = None
    #         if len(gdf):
    #             chunk_json = gdf.to_json(drop_id=True)
    #             # crop GeoJSON header and footer and split lines
    #             chunk_json = chunk_json[43:-2].replace('}}, ', '}},\n')
    #         del gdf
    #         return chunk_json

    #     # json header flag
    #     empty = True
    #     with open(f'{name}.geojson', 'w') as f:
    #         # GeoJSON header
    #         f.write('{"type": "FeatureCollection", "features": [\n')
    
    #         if 'stack' in data.dims:
    #             stack_blocks = np.array_split(np.arange(grid['stack'].size), np.arange(0, grid['stack'].size, self.chunksize1d)[1:])
    #             # prepare the progress bar
    #             with tqdm(desc=caption, total=len(stack_blocks)) as pbar:
    #                 for stack_block in stack_blocks:
    #                     block = grid.isel(stack=stack_block).drop_vars(['y','x'])
    #                     chunk_json = block_as_json(block, stackvar, data.name)
    #                     del block
    #                     if chunk_json is not None:
    #                         f.write(('' if empty else ',') + chunk_json)
    #                         empty = False
    #                     pbar.update(1)
    #         else:
    #             # split to equal chunks and rest
    #             # 1/4 NetCDF chunk is the smallest reasonable processing chunk
    #             lats_blocks = np.array_split(np.arange(grid.lat.size), np.arange(0, grid.lat.size, self.netcdf_chunksize//2)[1:])
    #             lons_blocks = np.array_split(np.arange(grid.lon.size), np.arange(0, grid.lon.size, self.netcdf_chunksize//2)[1:])
    #             # prepare the progress bar
    #             with tqdm(desc=caption, total=len(lats_blocks)*len(lons_blocks)) as pbar:
    #                 for lats_block in lats_blocks:
    #                     for lons_block in lons_blocks:
    #                         block = grid.isel(lat=lats_block, lon=lons_block)
    #                         chunk_json = block_as_json(block, stackvar, data.name)
    #                         del block
    #                         if chunk_json is not None:
    #                             f.write(('' if empty else ',') + chunk_json)
    #                             empty = False
    #                         pbar.update(1)

    #         # GeoJSON footer
    #         f.write(']}')
    #     del grid

    # def export_csv(self, data, name, caption='Exporting WGS84 CSV', delimiter=',', digits=2, coord_digits=6):
    #     """
    #     Export the provided data to a CSV file.
    
    #     Parameters
    #     ----------
    #     data : xarray.DataArray
    #         The data to be exported as CSV.
    #     name : str
    #         The base name for the CSV file(s).
    #     caption : str, optional
    #         A description for the export process, used for progress display. Default is 'Exporting WGS84 CSV'.
    #     delimiter : str, optional
    #         The delimiter to use in the CSV file. Default is ','.
    #     digits : int, optional
    #         Number of decimal places to round the data values. Default is 2.
    #     coord_digits : int, optional
    #         Number of decimal places to round the coordinates. Default is 6.

    #     Returns
    #     -------
    #     None
    #         The function writes the CSV file(s) to disk with the specified name.
            
    #     Examples
    #     --------
    #     Export a CSV file "velocity.csv":
    #     sbas.export_csv(velocity, 'velocity')
    #     """
    #     import xarray as xr
    #     import numpy as np
    #     from tqdm.auto import tqdm
    #     # disable "distributed.utils_perf - WARNING - full garbage collections ..."
    #     try:
    #         from dask.distributed import utils_perf
    #         utils_perf.disable_gc_diagnosis()
    #     except ImportError:
    #         from distributed.gc import disable_gc_diagnosis
    #         disable_gc_diagnosis()

    #     assert isinstance(data, xr.DataArray), 'Argument data is not an xr.DataArray object'
    
    #     # convert the data to geographic coordinates if necessary
    #     grid = data

    #     # determine if data has a stack dimension and what it is
    #     if 'stack' in data.dims:
    #         stackvar = data.dims[0] if len(data.dims) == 2 else None
    #     else:
    #         stackvar = data.dims[0] if len(data.dims) == 3 else None
    
    #     with open(f'{name}.csv', 'w') as f:
    #         # CSV header
    #         f.write(delimiter.join(filter(None, [stackvar, 'lon', 'lat', data.name])) + '\n')
    #         if 'stack' in data.dims:
    #             stack_blocks = np.array_split(np.arange(grid['stack'].size), np.arange(0, grid['stack'].size, self.chunksize1d)[1:])
    #             # prepare the progress bar
    #             with tqdm(desc=caption, total=len(stack_blocks)) as pbar:
    #                 for stack_block in stack_blocks:
    #                     block = grid.isel(stack=stack_block).drop_vars(['y','x']).compute()
    #                     block_val = block.round(digits).values
    #                     block_lat = block.lat.round(coord_digits).values
    #                     block_lon = block.lon.round(coord_digits).values
    #                     if stackvar is not None:
    #                         stackvals = block[stackvar]
    #                         if np.issubdtype(stackvals.dtype, np.datetime64):
    #                             stackvals = stackvals.dt.date.astype(str)
    #                         block_csv = np.column_stack((np.repeat(stackvals, block_lon.size),
    #                                                      np.repeat(block_lon, stackvals.size),
    #                                                      np.repeat(block_lat, stackvals.size),
    #                                                      block_val.ravel()))
    #                     else:
    #                         block_csv = np.column_stack((block_lon, block_lat, block_val.astype(str).ravel()))
    #                     del block, block_lat, block_lon
    #                     block_csv = block_csv[np.isfinite(block_val.ravel())]
    #                     del block_val
    #                     if block_csv.size > 0:
    #                         np.savetxt(f, block_csv, delimiter=delimiter, fmt='%s')
    #                     del block_csv
    #                     pbar.update(1)
    #         else:
    #             # split to equal chunks and rest
    #             # 1/4 NetCDF chunk is the smallest reasonable processing chunk
    #             lats_blocks = np.array_split(np.arange(grid.lat.size), np.arange(0, grid.lat.size, self.netcdf_chunksize//2)[1:])
    #             lons_blocks = np.array_split(np.arange(grid.lon.size), np.arange(0, grid.lon.size, self.netcdf_chunksize//2)[1:])
    #             # prepare the progress bar
    #             with tqdm(desc=caption, total=len(lats_blocks)*len(lons_blocks)) as pbar:
    #                 for lats_block in lats_blocks:
    #                     for lons_block in lons_blocks:
    #                         block = grid.isel(lat=lats_block, lon=lons_block).compute()
    #                         block_val = block.round(digits).values
    #                         block_lat = block.lat.round(coord_digits).values
    #                         block_lon = block.lon.round(coord_digits).values
    #                         if stackvar is not None:
    #                             stackvals = block[stackvar]
    #                             if np.issubdtype(stackvals.dtype, np.datetime64):
    #                                 stackvals = stackvals.dt.date.astype(str)
    #                             stackvals, lats, lons = np.meshgrid(stackvals, block_lat, block_lon, indexing='ij')
    #                             block_csv = np.column_stack((stackvals.ravel(), lons.ravel(), lats.ravel(), block_val.ravel()))
    #                             del stackvals, lats, lons
    #                         else:
    #                             lats, lons = np.meshgrid(block_lat, block_lon, indexing='ij')
    #                             block_csv = np.column_stack((lons.ravel(), lats.ravel(), block_val.astype(str).ravel()))
    #                             del lats, lons
    #                         del block, block_lat, block_lon
    #                         block_csv = block_csv[np.isfinite(block_val.ravel())]
    #                         del block_val
    #                         if block_csv.size > 0:
    #                             np.savetxt(f, block_csv, delimiter=delimiter, fmt='%s')
    #                         del block_csv
    #                         pbar.update(1)
    #     del grid

    # def export_netcdf(self, data, name, caption='Exporting WGS84 NetCDF', engine='netcdf4', format='NETCDF3_64BIT'):
    #     """
    #     Export the provided data to a NetCDF file.
    
    #     Parameters
    #     ----------
    #     data : xarray.DataArray
    #         The data to be exported as NetCDF.
    #     name : str
    #         The base name for the NetCDF file(s).
    #     caption : str, optional
    #         A description for the export process, used for progress display. Default is 'Exporting WGS84 NetCDF'.
    #     engine : str, optional
    #         The NetCDF engine to use (e.g., 'netcdf4'). Default is 'netcdf4'.
    #     format : str, optional
    #         The NetCDF format to use (e.g., 'NETCDF3_64BIT'). Default is 'NETCDF3_64BIT'.

    #     Returns
    #     -------
    #     None
    #         The function writes the NetCDF file to disk with the specified name.
            
    #     Examples
    #     --------
    #     Export a NetCDF file "velocity.nc":
    #     sbas.export_netcdf(velocity, 'velocity')
    #     """
    #     import xarray as xr
    #     import pandas as pd
    #     import numpy as np
    #     import dask
    #     import os
    #     # disable "distributed.utils_perf - WARNING - full garbage collections ..."
    #     try:
    #         from dask.distributed import utils_perf
    #         utils_perf.disable_gc_diagnosis()
    #     except ImportError:
    #         from distributed.gc import disable_gc_diagnosis
    #         disable_gc_diagnosis()
    
    #     assert isinstance(data, xr.DataArray), 'Argument data is not an xr.DataArray object'
    
    #     # convert the data to geographic coordinates if necessary
    #     grid = data

    #     if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         print (f"NOTE: open as xr.open_dataarray('{name}.nc').set_index(stack=['lat', 'lon'])")
    #         grid = grid.reset_index('stack')
    
    #     filename = f'{name}.nc'
    #     if os.path.exists(filename):
    #         os.remove(filename)
    #     encoding = {data.name: self.get_encoding_netcdf(grid.shape)}
    #     delayed = grid.to_netcdf(filename, engine=engine, encoding=encoding, format=format, compute=False)
    #     progressbar(dask.persist(delayed), desc=caption)
    #     del grid
