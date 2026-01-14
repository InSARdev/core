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
from __future__ import annotations
from .Stack_plot import Stack_plot
from .BatchCore import BatchCore
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
#from collections.abc import Mapping
from . import utils_io
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import rasterio as rio
    import pandas as pd
    import xarray as xr

class Stack(Stack_plot, BatchCore):
    
    def __init__(self, mapping:dict[str, xr.Dataset] | None = None):
        #print('Stack __init__', 0 if mapping is None else len(mapping))
        super().__init__(mapping)

    @property
    def coords(self):
        """Return coordinates from the first dataset in the stack.
        
        All datasets in a Stack share the same coordinate structure,
        so we expose the first one's coords for convenience.
        """
        if not self:
            return None
        first_ds = next(iter(self.values()))
        return first_ds.coords

    def PRM(self, keys: str | list[str] | None = None) -> dict:
        """Return platform parameters per burst.

        Parameters
        ----------
        keys : str | list[str] | None
            Parameter name(s) to extract. If ``None``, all scalar attrs
            and 0-D data_vars are returned per burst.

        Returns
        -------
        dict
            Mapping of burst key -> param dict (or burst key -> single value
            when a single key is requested).
        """
        if not self:
            return {}

        # normalize key selection
        select_all = keys is None
        if isinstance(keys, str):
            keys_list = [keys]
        else:
            keys_list = keys if keys is not None else None

        result: dict[str, object] = {}
        for burst, ds in self.items():
            params: dict[str, object] = dict(getattr(ds, 'attrs', {}))
            for name, var in getattr(ds, 'data_vars', {}).items():
                if var.ndim == 0:
                    try:
                        params.setdefault(name, var.item())
                    except Exception:
                        params.setdefault(name, var.values)

            if not select_all:
                if keys_list is None:
                    params = {}
                else:
                    params = {k: params.get(k) for k in keys_list}
                    if isinstance(keys, str):
                        params = params.get(keys)

            result[burst] = params

        return result

    def __getitem__(self, key):
        """Access by key or variable list."""
        # Handle variable selection like sbas[['ele']]
        if isinstance(key, list):
            if len(key) == 1 and self:
                var = key[0]
                # Build a Batch containing only the requested variable for each scene
                subset = {k: ds[[var]] for k, ds in self.items() if var in ds.data_vars}
                if not subset:
                    raise KeyError(var)
                return Batch(subset)
            # Multiple variables - return Stack with subset
            return type(self)({k: ds[key] for k, ds in self.items()})
        return dict.__getitem__(self, key)

    def __getattr__(self, name: str):
        """
        Access variables (e.g., 'ele') from the Stack as Batch.

        This allows accessing variables stored in burst datasets:
            sbas.ele  -> BatchVar containing elevation data
        """
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self:
            sample = next(iter(self.values()), None)
            if sample is not None and hasattr(sample, 'data_vars'):
                if name in sample.data_vars or name in sample.coords:
                    subset = {k: ds[[name]] for k, ds in self.items() if name in ds.data_vars or name in ds.coords}
                    if subset:
                        return Batch(subset)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def transform(self) -> Batch:
        """Return a Batch view of this Stack (including 1D/2D non-complex vars)."""
        return Batch(self)

    def to_vtk(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK.

        Merges bursts using to_dataset() and exports one VTK file per data variable
        (e.g., VV.vtk). Within each file, pairs become separate VTK arrays named
        by date (e.g., 20190708_20190702).

        Parameters
        ----------
        path : str
            Output directory for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Accepts any mapping convertible to ``Batch``.
        transform : Batch | None, optional
            Optional transform Batch providing topography (``ele`` or ``z``).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.
        """
        import os
        import numpy as np
        import xarray as xr
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        # Default to self.transform() when called on Stack and transform not provided
        tfm_is_default = transform is None
        tfm = transform if transform is not None else self.transform()
        if tfm is not None and not isinstance(tfm, BatchCore):
            tfm = Batch(tfm)

        if not target:
            return

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"{_format_dt(val[0])}_{_format_dt(val[1])}"
            # Handle string format from to_dataset(): "2019-07-02 2019-07-08"
            if isinstance(val, str) and ' ' in val:
                parts = val.split()
                if len(parts) == 2:
                    return f"{_format_dt(parts[0])}_{_format_dt(parts[1])}"
            return _format_dt(val)

        os.makedirs(path, exist_ok=True)

        # Merge bursts into unified dataset(s) per variable
        merged = target.to_dataset()
        if isinstance(merged, xr.DataArray):
            merged = merged.to_dataset()

        # Get transform elevation merged via to_dataset()
        # Decimate default transform to match input batch resolution for efficiency
        topo_merged = None
        if tfm is not None:
            if tfm_is_default:
                # Decimate each burst's transform to match corresponding input burst
                # Use index-based nearest neighbor selection (much faster than reindex)
                def _nearest_indices(source_coords, target_coords):
                    """Find indices in source_coords nearest to target_coords."""
                    # Handle descending coordinates (e.g., y going north to south)
                    descending = len(source_coords) > 1 and source_coords[0] > source_coords[-1]
                    if descending:
                        source_coords = source_coords[::-1]
                    indices = np.searchsorted(source_coords, target_coords)
                    indices = np.clip(indices, 0, len(source_coords) - 1)
                    # Check if previous index is closer
                    prev_indices = np.clip(indices - 1, 0, len(source_coords) - 1)
                    prev_diff = np.abs(source_coords[prev_indices] - target_coords)
                    curr_diff = np.abs(source_coords[indices] - target_coords)
                    indices = np.where(prev_diff < curr_diff, prev_indices, indices)
                    # Convert back to original order if descending
                    if descending:
                        indices = len(source_coords) - 1 - indices
                    return indices

                decimated = {}
                for k in target.keys():
                    if k not in tfm:
                        continue
                    tfm_ds = tfm[k][['ele']]
                    tgt_ds = target[k]
                    # Find nearest indices for y and x coordinates
                    y_idx = _nearest_indices(tfm_ds.y.values, tgt_ds.y.values)
                    x_idx = _nearest_indices(tfm_ds.x.values, tgt_ds.x.values)
                    # Select using indices and assign target coordinates
                    selected = tfm_ds.isel(y=y_idx, x=x_idx)
                    selected = selected.assign_coords(y=tgt_ds.y, x=tgt_ds.x)
                    decimated[k] = selected
                topo_merged = Batch(decimated).to_dataset()
            else:
                # User-provided transform: use as-is
                topo_merged = tfm[['ele']].to_dataset()

        # Group by data variable (polarization)
        data_vars = list(merged.data_vars)

        with tqdm(total=len(data_vars), desc='Exporting VTK') as pbar:
            for data_var in data_vars:
                da = merged[data_var]

                # Handle pair dimension
                if 'pair' in da.dims:
                    pair_coord = da.coords.get('pair')
                    pair_values = pair_coord.values if pair_coord is not None else range(da.sizes.get('pair', 0))
                    export_items = []
                    for i, pair_val in enumerate(pair_values):
                        da_slice = da.isel(pair=i)
                        if 'pair' in da_slice.dims:
                            da_slice = da_slice.squeeze('pair', drop=True)
                        export_items.append((pair_val, da_slice))
                else:
                    export_items = [(None, da)]

                if not export_items:
                    pbar.update(1)
                    continue

                ref_da = export_items[0][1]
                layers = []

                # Add topography from transform
                if topo_merged is not None:
                    topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
                    if topo_da is not None:
                        topo_da = topo_da.interp(y=ref_da.y, x=ref_da.x, method='linear')
                        if mask:
                            topo_da = topo_da.where(np.isfinite(ref_da))
                        layers.append(topo_da.rename('z'))

                # Add overlay
                if overlay is not None:
                    if not isinstance(overlay, xr.DataArray):
                        raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")

                    ov = overlay
                    if 'band' not in ov.dims:
                        ov = ov.expand_dims('band')
                    try:
                        ov = ov.sel(y=slice(float(ref_da.y.min()), float(ref_da.y.max())),
                                    x=slice(float(ref_da.x.min()), float(ref_da.x.max())))
                    except Exception:
                        try:
                            ov = ov.sel(lat=slice(float(ref_da.lat.min()), float(ref_da.lat.max())),
                                        lon=slice(float(ref_da.lon.min()), float(ref_da.lon.max())))
                        except Exception:
                            pass
                    ov = ov.interp(y=ref_da.y, x=ref_da.x, method='linear')
                    layers.append(ov.rename('colors'))

                # Add data arrays with pair names
                for pair_val, da_item in export_items:
                    var_name = _format_pair(pair_val) if pair_val is not None else data_var
                    layers.append(da_item.rename(var_name))

                ds_out = xr.merge(layers, compat='override', join='left')
                vtk_grid = as_vtk(ds_out)

                filename = os.path.join(path, f"{data_var}.vtk")

                writer = vtkStructuredGridWriter()
                writer.SetFileName(filename)
                writer.SetInputData(vtk_grid)
                writer.SetFileType(VTK_BINARY)
                writer.Write()

                pbar.update(1)

    def to_vtks(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK from a Batch.

        Parameters
        ----------
        path : str
            Output directory for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Accepts any mapping convertible to ``Batch``.
        transform : Batch | None, optional
            Optional transform Batch providing topography (`ele`).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.
        """
        import os
        import numpy as np
        import xarray as xr
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        tfm = transform
        if tfm is not None and not isinstance(tfm, BatchCore):
            tfm = Batch(tfm)

        if not target:
            return

        def _interp_to_grid(source: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
            if {'y', 'x'}.issubset(source.dims):
                return source.interp(y=target_da.y, x=target_da.x, method='linear')
            if {'lat', 'lon'}.issubset(source.dims):
                if {'lat', 'lon'}.issubset(target_da.coords):
                    return source.interp(lat=target_da.lat, lon=target_da.lon, method='linear')
                return source.rename({'lat': 'y', 'lon': 'x'}).interp(y=target_da.y, x=target_da.x, method='linear')
            return source

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"{_format_dt(val[0])}_{_format_dt(val[1])}"
            return _format_dt(val)

        os.makedirs(path, exist_ok=True)

        with tqdm(total=len(target), desc='Exporting VTK') as pbar:
            for burst, ds in target.items():
                if not ds.data_vars:
                    pbar.update(1)
                    continue

                data_var = next(iter(ds.data_vars))
                base_da = ds[data_var]

                if 'pair' in ds.dims:
                    pair_coord = ds.coords.get('pair')
                    pair_values = pair_coord.values if pair_coord is not None else range(ds.sizes.get('pair', 0))
                    export_items = []
                    for i, pair_val in enumerate(pair_values):
                        ds_slice = ds.isel(pair=i)
                        if 'pair' in ds_slice.dims:
                            ds_slice = ds_slice.squeeze('pair', drop=True)
                        else:
                            ds_slice = ds_slice.squeeze(drop=True)
                        export_items.append((pair_val, ds_slice))
                else:
                    export_items = [(None, ds)]

                for pair_val, ds_item in export_items:
                    base_da_item = ds_item[data_var]
                    layers = [base_da_item.rename(data_var)]

                    if tfm is not None and burst in tfm:
                        tfm_ds = tfm[burst]
                        topo_da = tfm_ds.get('ele') if 'ele' in tfm_ds else tfm_ds.get('z') if 'z' in tfm_ds else None
                        if topo_da is not None:
                            topo_da = _interp_to_grid(topo_da, base_da_item)
                            if mask:
                                topo_da = topo_da.where(np.isfinite(base_da_item))
                            layers.append(topo_da.rename('z'))

                    if overlay is not None:
                        if not isinstance(overlay, xr.DataArray):
                            raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")

                        ov = overlay
                        if 'band' not in ov.dims:
                            ov = ov.expand_dims('band')
                        try:
                            ov = ov.sel(y=slice(float(base_da_item.y.min()), float(base_da_item.y.max())),
                                        x=slice(float(base_da_item.x.min()), float(base_da_item.x.max())))
                        except Exception:
                            try:
                                ov = ov.sel(lat=slice(float(base_da_item.lat.min()), float(base_da_item.lat.max())),
                                            lon=slice(float(base_da_item.lon.min()), float(base_da_item.lon.max())))
                            except Exception:
                                pass
                        ov = _interp_to_grid(ov, base_da_item)
                        layers.append(ov.rename('colors'))

                    ds_out = xr.merge(layers, compat='override', join='left')
                    vtk_grid = as_vtk(ds_out)

                    pair_suffix = ''
                    if pair_val is not None:
                        pair_suffix = f"_{_format_pair(pair_val)}"

                    filename = os.path.join(path, f"{burst}{pair_suffix}.vtk")

                    writer = vtkStructuredGridWriter()
                    writer.SetFileName(filename)
                    writer.SetInputData(vtk_grid)
                    writer.SetFileType(VTK_BINARY)
                    writer.Write()

                pbar.update(1)

    def compute(self, *batches: BatchCore) -> tuple:
        """Compute multiple Batch objects together efficiently.

        This method materializes multiple dependent Batch objects in a single
        dask graph execution, which is faster than computing them separately
        because shared computations are only performed once.

        Parameters
        ----------
        *batches : BatchCore
            One or more Batch objects to compute together.

        Returns
        -------
        tuple
            Tuple of computed Batch objects in the same order as input.
            If only one batch is provided, returns a single Batch (not a tuple).

        Examples
        --------
        Compute two dependent batches together (faster than separate compute):

        >>> intf, corr = stack.compute(
        ...     intf.mask(landmask).downsample(100),
        ...     corr.mask(landmask).downsample(100)
        ... )

        Instead of separate execution (about 2x slower):

        >>> intf = intf.mask(landmask).downsample(100).compute()
        >>> corr = corr.mask(landmask).downsample(100).compute()

        Compute three batches:

        >>> intf, corr, elevation = stack.compute(intf, corr, ele)
        """
        import dask
        from insardev_toolkit.progressbar import progressbar

        if not batches:
            raise ValueError('At least one Batch must be provided')

        # Convert all batches to dicts for dask.persist
        batch_dicts = [dict(b) for b in batches]

        # Persist all batches together in a single graph execution
        progressbar(
            result := dask.persist(*batch_dicts),
            desc='Computing Batches...'.ljust(25)
        )

        # Convert back to Batch objects with computed coordinates
        computed_batches = []
        for i, batch_dict in enumerate(result):
            computed = {}
            for key, ds in batch_dict.items():
                # Compute any lazy coordinates, preserving their dims
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)
                computed[key] = ds
            # Preserve original Batch type
            computed_batches.append(type(batches[i])(computed))

        if len(computed_batches) == 1:
            return computed_batches[0]
        return tuple(computed_batches)

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str = 'Snapshotting...', n_jobs: int = -1, debug=False):
        if len(args) > 2:
            raise ValueError(f'ERROR: snapshot() accepts only one or two Batch or dict objects or no arguments.')
        datas = utils_io.snapshot(*args, store=store, storage_options=storage_options, compat=True, caption=caption, n_jobs=n_jobs, debug=debug)
        return datas

    # def downsample(self, *args, coarsen=None, resolution=60, func='mean', debug:bool=False):
    #     datas = []
    #     for arg in args:
    #         print ('type(arg)', type(arg))
    #         if isinstance(arg, (Stack, BatchComplex)):
    #             arg = Batch(arg)
    #             print (arg.isel(0)['033_069722_IW3'].data_vars)
    #         wrap = True if isinstance(arg, BatchWrap) else False
    #         print ('\ttype(arg)', type(arg), 'wrap', wrap)
    #         sample = next(iter(arg.values()))
    #         callback = utils_xarray.downsampler(sample, coarsen=coarsen, resolution=resolution, func=func, wrap=wrap, debug=debug)
    #         data = callback(arg)
    #         datas.append(BatchWrap(data) if wrap else Batch(data))
    #     return datas

    def to_dataframe(self,
                     datas: dict[str, xr.Dataset | xr.DataArray] | None = None,
                     crs:str|None='auto',
                     attr_start:str='BPR',
                     debug:bool=False
                     ) -> pd.DataFrame:
        """
        Return a Pandas DataFrame for all Stack scenes.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing Stack scenes.

        Examples
        --------
        df = stack.to_dataframe()
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd
        import numpy as np

        if datas is not None and not isinstance(datas, dict):
            raise ValueError(f'ERROR: datas is not None or a dict: {type(datas)}')
    
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            crs = self.crs

        if datas is None:
            datas = self

        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in next(iter(datas.values())).data_vars]
        #print ('polarizations', polarizations)

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in datas.values():
            #print (data.id)
            attrs = [data_var for data_var in ds if ds[data_var].dims==('date',)][::-1]
            attr_start_idx = attrs.index(attr_start)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs[:attr_start_idx+1]:
                    #NotImplementedError: 'item' is not yet a valid method on dask arrays
                    value = ds[attr].item(date_idx)
                    #value = ds[attr].values[date_idx]
                    #print (attr, date_idx, date, value)
                    #processed_attr['date'] = date
                    if hasattr(value, 'item'):
                        processed_attr[attr] = value.item()
                    elif attr == 'geometry':
                        processed_attr[attr] = wkt.loads(value)
                    else:
                        processed_attr[attr] = value
                processed_attrs.append(processed_attr)
                #print (processed_attr)
        df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        #del df['date']
        #df['polarization'] = ','.join(polarizations)
        # convert polarizations to strings like "VV,VH" to pevent confusing with tuples in the dataframe
        df = df.assign(polarization=','.join(map(str, polarizations)))
        # reorder columns to the same order as preprocessor uses
        pol = df.pop("polarization")
        df.insert(3, "polarization", pol)
        # round for human readability
        df['BPR'] = df['BPR'].round(1)

        group_col = df.columns[0]
        burst_col = df.columns[1]
        #print ('df.columns[0]', df.columns[0])
        #print ('df.columns[:2][::-1].tolist()', df.columns[:2][::-1].tolist())
        df['startTime'] = pd.to_datetime(df['startTime'])
        #df['date'] = df['startTime'].dt.date.astype(str)
        df = df.sort_values(by=[group_col, burst_col]).set_index([group_col, burst_col])
        # move geometry to the end of the dataframe to be the most similar to insar_pygmtsar output
        df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]
        
        return df.to_crs(crs) if crs is not None else df

    def load(self, urls:str | list | dict[str, str], storage_options:dict[str, str]|None=None, debug:bool=False):
        import numpy as np
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        import zarr
        from shapely import wkt
        import os
        from insardev_toolkit import progressbar_joblib
        from tqdm.auto import tqdm
        import joblib
        import warnings
        # suppress the "Sending large graph of size …"
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module=r'distributed\.client',
            message=r'Sending large graph of size .*'
        )
        from distributed import get_client, WorkerPlugin
        class IgnoreDaskDivide(WorkerPlugin):
            def setup(self, worker):
                # suppress the "RuntimeWarning: invalid value encountered in divide"
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r'dask\._task_spec'
                )
        client = get_client()
        client.register_plugin(IgnoreDaskDivide(), name='ignore_divide')

        def burst_preprocess(ds, debug:bool=False):
            #print ('ds_preprocess', ds)
            # Convert all attrs to vars (except xarray/CRS metadata)
            for key in list(ds.attrs.keys()):
                if key in ['Conventions', 'spatial_ref']:
                    continue
                # Create a new DataArray with the original value
                val = ds.attrs[key]
                # Handle array values (e.g., polynomial coefficients)
                if isinstance(val, (list, tuple)):
                    val = np.array(val)
                val_arr = np.asarray(val)
                if val_arr.ndim == 0:
                    ds[key] = xr.DataArray(val_arr, dims=[])
                else:
                    # For 1D arrays (like polynomial coefficients), create with 'coef' dim
                    ds[key] = xr.DataArray(val_arr, dims=['coef'])
                # remove the attribute
                del ds.attrs[key]
            
            # remove attributes for repeat bursts to unify the attributes
            BPR = ds['BPR'].values.item(0)
            if BPR != 0:
                ds.attrs = {}

            ds['data'] = (ds.re + 1j*ds.im).astype(np.complex64)
            if not debug:
                del ds['re'], ds['im']
            #date = pd.to_datetime(ds['startTime'].item())
            #date64s = np.array([np.datetime64(date)], dtype="datetime64[s]")
            #return ds.expand_dims({'date': np.array([date.date()], dtype='U10')})
            #return ds.expand_dims({'date': date64s})
            return ds.expand_dims(
                date=[np.datetime64(ds["startTime"].item(), 's')]
            )

        def _bursts_transform_preprocess(bursts, transform):
            #print ('_bursts_transform_preprocess')
            
            # in case of multiple polarizations, merge them into a single dataset
            polarizations = np.unique(bursts.polarization)
            if len(polarizations) > 1:
                datas = []
                for polarization in polarizations:
                    data = bursts.isel(date=bursts.polarization == polarization)\
                                .rename({'data': polarization})
                    # cannot combine in a single value VV and VH polarizations and corresponding burst names
                    data.burst.values = [
                        v.replace(polarization, 'XX') for v in data.burst.values
                    ]
                    del data['polarization']
                    datas.append(data.chunk(-1))
                ds = xr.merge(datas)
                del datas
            else:
                ds = bursts.rename({'data': polarizations[0]})

            for var in transform.data_vars:
                #if var not in ['re', 'im']:
                ds[var] = transform[var].chunk(-1)

            ds.rio.write_crs(bursts.attrs['spatial_ref'], inplace=True)
            return ds

        def bursts_transform_preprocess(dss, transform):
            """
            Combine bursts and transform into a single dataset.
            Only reference burst for every polarization has attributes (see burst_preprocess)
            """
            #print ('bursts_transform_preprocess')

            polarizations = np.unique([ds.polarization for ds in dss])
            #print ('polarizations', polarizations)

            # convert generic 'data' variable for all polarizations to VV, VH,... variables
            datas = []
            spatial_ref = None
            for polarization in polarizations:
                data = [ds for ds in dss if ds.polarization==polarization]
                # newer xarray tightened attribute checks; keep first dataset attrs
                data = xr.concat(data, dim='date', combine_attrs='override').rename({'data': polarization})
                #data = xr.concat(data, dim='date', combine_attrs='override').rename({'data': polarization}).sortby('date')
                # cannot combine in a single value VV and VH polarizations and corresponding burst names
                data.burst.values = [v.replace(polarization, 'XX') for v in data.burst.values]
                del data['polarization']
                if spatial_ref is None:
                    spatial_ref = data.attrs.get('spatial_ref')
                datas.append(data)
                del data
            ds = xr.merge(datas, combine_attrs='override')
            # only reference burst has spatial_ref attribute, capture before merge and restore if needed
            if spatial_ref is None:
                spatial_ref = transform.attrs.get('spatial_ref')
            if spatial_ref is None:
                raise KeyError('spatial_ref')
            ds.attrs['spatial_ref'] = spatial_ref
            del datas

            # add transform variables
            for var in transform.data_vars:
                ds[var] = transform[var]

            # set the coordinate reference system
            ds.rio.write_crs(spatial_ref, inplace=True)
            return ds

        # if isinstance(urls, str):
        #     print ('NOTE: urls is a string, convert to dict with burst as key and list of URLs as value.')
        # elif isinstance(urls, dict):
        #     print ('NOTE: urls is a dict, using it as is.')
        #     groups = urls
        # elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2:
        #     print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
        #     #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
        #     groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
        # elif isinstance(urls, list):
        #     print ('NOTE: urls is a list, convert to dict with burst as key and list of URLs as value.')
        #     groups = {}
        #     for url in urls:
        #         parent = url.rsplit('/', 2)[1]
        #         groups.setdefault(parent, []).append(url)
        # else:
        #     raise ValueError(f'ERROR: urls is not a dict, list, or Pandas Dataframe: {type(urls)}')

        # def store_open_burst(grp):
        #     #ds = xr.open_zarr(root.store, group=f'021_043788_IW1/{burst}', consolidated=True, zarr_format=3)
        #     #grp = root['021_043788_IW1'][burst]
        #     ds = xr.open_zarr(grp.store, group=grp.path, consolidated=True, zarr_format=3)
        #     return burst_preprocess(ds)
        
        def store_open_group(root, group):
            # open group (fullBurstID)
            grp = root[group]
            # get all subgroups (bursts) except transform
            grp_bursts = [grp[k] for k in grp.keys() if k!='transform']
            dss = [xr.open_zarr(_grp.store, group=_grp.path, consolidated=True, zarr_format=3) for _grp in grp_bursts]
            dss = [burst_preprocess(ds) for ds in dss]
            # get transform subgroup
            grp_transform = grp['transform']
            transform = xr.open_zarr(grp_transform.store, group=grp_transform.path, consolidated=True, zarr_format=3)
            # combine bursts and transform casted to 32 bit floats
            ds = bursts_transform_preprocess(dss, transform.astype(np.float32))
            del dss, transform
            return group, ds

        if isinstance(urls, str):
            # note: isinstance(urls, zarr.storage.ZipStore) can be loaded too but it is less efficient
            urls = os.path.expanduser(urls)
            root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
            with progressbar_joblib.progressbar_joblib(tqdm(desc='Loading Dataset...'.ljust(25), total=len(list(root.group_keys())))) as progress_bar:
                dss = joblib.Parallel(n_jobs=-1, backend='loky')\
                    (joblib.delayed(store_open_group)(root, group) for group in list(root.group_keys()))
            # list of key - dataset converted to dict and appended to the existing dict
            self.update(dss)
        # elif isinstance(urls, FsspecStore):
        #     root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
        #     dss = []
        #     for group in tqdm(list(root.group_keys()), desc='Loading Store'):
        #         dss.append(store_open_group(root, group))
        #     self.dss = dict(dss)
        #     del dss
        elif isinstance(urls, list) or isinstance(urls, pd.DataFrame):
            # load bursts and transform specified by URLs
            # this allows to load from multiple locations with precise control of the data
            if isinstance(urls, list):
                print ('NOTE: urls is a list, using it as is.')
                df = pd.DataFrame(urls, columns=['url'])
                df['fullBurstID'] = df['url'].str.rsplit('/', n=2).str[1]
                df['burst'] = df["url"].str.rsplit("/", n=2).str[2]
                urls = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
                print (urls.head())
            elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2 and len(urls.columns) == 1:
                print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
                #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
                #groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
            else:
                raise ValueError(f'ERROR: urls is not a list, or Pandas Dataframe with multiindex: {type(urls)}')

            dss = {}
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets...'.ljust(25)):
                #print ('fullBurstID', fullBurstID)
                df = urls[urls.index.get_level_values(0) == fullBurstID]
                bases = df[df.index.get_level_values(1) != 'transform'].iloc[:,0].values
                #print ('fullBurstID', fullBurstID, '=>', bases)
                base = df[df.index.get_level_values(1) == 'transform'].iloc[:,0].values[0]
                #print ('fullBurstID', fullBurstID, '=>', base)
                bursts = xr.open_mfdataset(
                    bases,
                    engine='zarr',
                    zarr_format=3,
                    consolidated=True,
                    parallel=True,
                    concat_dim='date',
                    combine='nested',
                    preprocess=burst_preprocess,
                    storage_options=storage_options,
                )
                # some variables are stored as int32 with scale factor, convert to float32 instead of default float64
                transform = xr.open_dataset(base, engine='zarr', zarr_format=3, consolidated=True, storage_options=storage_options).astype('float32')

                ds = _bursts_transform_preprocess(bursts, transform)
                dss[fullBurstID] = ds
                del ds, bursts, transform

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            self.update(dss)

        # Check for duplicate dates in each burst
        for key, ds in self.items():
            dates = ds.date.values
            unique, counts = np.unique(dates, return_counts=True)
            if (counts > 1).any():
                duplicates = unique[counts > 1]
                raise ValueError(f'Burst {key} contains duplicate dates: {duplicates}. '
                                 'This may be caused by corrupted data or library issues. '
                                 'Try restarting the runtime and reloading the data.')
        return self

    def elevation(self, phase: Batch | float | list | "np.ndarray", baseline: float | None = None, transform: Batch | None = None) -> "Batch | float | np.ndarray":
        """Compute elevation (meters) from unwrapped phase grids in radar coordinates.

        Parameters
        ----------
        phase : Batch | float | list | np.ndarray
            Unwrapped phase grids (e.g., output of unwrap2d()), or a scalar/array
            of phase values for quick height-per-fringe calculations.
        baseline : float | None, optional
            Perpendicular baseline in meters. Required when phase is scalar/array.
            If ``None`` and phase is Batch, use the burst-specific ``BPR``.
        transform : Batch | None, optional
            Transform batch containing look vectors; its `.incidence()` is used.
            If ``None``, defaults to ``self.transform()``.

        Returns
        -------
        Batch | float | np.ndarray
            Elevation grids as float32 datasets, or scalar/array if input was scalar/array.
        """
        import xarray as xr
        import numpy as np

        # Handle scalar/list/array input for quick calculations
        if not isinstance(phase, BatchCore):
            if baseline is None:
                raise ValueError("baseline is required when phase is a scalar/list/array")
            if transform is None:
                transform = self.transform()

            # Get average parameters from first burst
            first_key = next(iter(transform.keys()))
            tfm = transform[first_key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return var.item()
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            slant_start = _scalar_from_ds(tfm, 'SC_height_start')
            slant_end = _scalar_from_ds(tfm, 'SC_height_end')
            slant_range = (slant_start + slant_end) / 2  # average slant range

            # Get average incidence angle
            inc_batch = transform.incidence()
            inc_da = inc_batch[first_key]['incidence']
            incidence = float(inc_da.mean().compute())

            # Convert input to numpy array
            phase_arr = np.asarray(phase)
            is_scalar = phase_arr.ndim == 0

            # Height from phase formula: h = -λ * φ * R * cos(incidence) / (4π * B⊥)
            elev = -(wavelength * phase_arr * slant_range * np.cos(incidence) / (4 * np.pi * baseline))

            # Return same type as input, rounded to 3 decimals
            if is_scalar:
                return round(float(elev), 3)
            return np.round(elev, 3)

        # Default to self.transform() when called on Stack and transform not provided
        if transform is None:
            transform = self.transform()
        incidence_batch = transform.incidence()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in phase.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return var.item()
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            slant_start = _scalar_from_ds(tfm, 'SC_height_start')
            slant_end = _scalar_from_ds(tfm, 'SC_height_end')
            if wavelength is None or slant_start is None or slant_end is None:
                raise KeyError(f"Missing parameters in transform for burst {key}: radar_wavelength, SC_height_start, SC_height_end")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                # Use BPR as DataArray to broadcast correctly across pairs
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            inc_da = incidence_batch[key]['incidence']
            slant_range = xr.DataArray(
                np.linspace(slant_start, slant_end, inc_da.sizes['x']),
                coords={'x': inc_da.coords['x']},
                dims=('x',)
            )

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                    slant = slant_range.interp(x=data.x)
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')
                    slant = slant_range.reindex_like(data.x, method='nearest')

                # Height from phase formula (PyGMTSAR convention):
                # h = -λ * φ * R * cos(incidence) / (4π * B⊥)
                # BPR broadcasts across pair dimension automatically
                elev = -(wavelength * data * slant * xr.ufuncs.cos(incidence) / (4 * np.pi * bpr))
                name = 'ele' if len(phase_ds.data_vars) == 1 else f'{var_name}_ele'
                elev_vars[name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def displacement_los(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Parameters
        ----------
        phase : Batch
            Unwrapped phase grids in radar coordinates.
        transform : Batch, optional
            Transform batch providing mission constants; only the first burst is
            queried for ``radar_wavelength``. If None, uses self.transform().

        Returns
        -------
        Batch
            LOS displacement grids (meters), lazily scaled by the mission wavelength.
        """
        import numpy as np
        import xarray as xr

        # Default to self.transform() when not provided
        if transform is None:
            transform = self.transform()

        if not transform:
            raise ValueError('transform must contain at least one burst with radar_wavelength')

        transform_first = next(iter(transform.values()))

        def _scalar_from_ds(ds, name: str):
            if name in ds:
                var = ds[name]
                if var.ndim == 0:
                    return var.item()
                # Handle 1D+ arrays - verify all values are identical
                elif var.ndim >= 1:
                    values = var.values.flatten()
                    unique = np.unique(values)
                    if len(unique) != 1:
                        raise ValueError(f'{name} has multiple distinct values: {unique}')
                    return unique[0]
            return ds.attrs.get(name)

        wavelength = _scalar_from_ds(transform_first, 'radar_wavelength')
        if wavelength is None:
            raise KeyError('Missing radar_wavelength in transform')

        # scale factor from phase in radians to displacement in meters
        # constant is negative to make LOS = -1 * range change
        scale = -float(wavelength) / (4 * np.pi)

        out: dict[str, xr.Dataset] = {}
        for key, phase_ds in phase.items():
            disp_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                disp = (data * scale).astype('float32')
                #name = 'los' if len(phase_ds.data_vars) == 1 else f'{var_name}_los'
                disp_vars[var_name] = disp
            out[key] = xr.Dataset(disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, phase: Batch, transform: Batch = None, func=None, suffix: str = '') -> Batch:
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np

        # Default to self.transform() when not provided, decimated to match phase resolution
        tfm_is_default = transform is None
        if transform is None:
            transform = self.transform()

        # Decimate default transform to match input phase resolution for efficiency
        if tfm_is_default and transform is not None:
            transform = Batch({k: transform[k].reindex(y=phase[k].y, x=phase[k].x, method='nearest')
                               for k in phase.keys() if k in transform})

        los_batch = self.displacement_los(phase, transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
                # align incidence to data grid
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')

                comp = (data / func(incidence)).astype('float32')

                if len(los_ds.data_vars) == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            out[key] = xr.Dataset(comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        return Batch(out)

    def displacement_vertical(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute vertical displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute east-west displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.sin, suffix='eastwest')

    def align_elevation(self, debug: bool = False) -> "Stack":
        """Apply elevation correction for TOPS burst processing.

        Corrects for satellite height variation along track when processing
        per-burst (as opposed to GMTSAR's approach of stitching before
        interferometry).

        Parameters
        ----------
        debug : bool, optional
            Print debug information including elevation corrections.

        Returns
        -------
        Stack
            New Stack with corrected elevation values.

        Notes
        -----
        The satellite height varies along the orbit (30-60m between adjacent
        bursts). This causes an apparent elevation offset between bursts
        when they are processed independently. This function computes and
        applies cumulative elevation corrections to align the bursts.

        The elevation correction is computed from the satellite height
        difference using radar geometry (incidence angle, Earth radius,
        slant range).
        """
        import numpy as np
        import xarray as xr

        if not self:
            return type(self)({})

        # Group bursts by (pathNumber, subswath) and sort by burst number
        # NOTE: Extract pathNumber and subswath from burst ID string, not from
        # per-date data variables which may vary (e.g., S1A vs S1B satellites).
        # Burst ID format: "{pathNumber}_{burstNumber}_{subswath}"
        ids = sorted(self.keys())
        groups = {}
        for bid in ids:
            parts = bid.split('_')
            if len(parts) >= 3:
                path_num = int(parts[0])
                burst_num = int(parts[1])
                subswath = parts[2]
            else:
                # Fallback to data variable if burst ID format is unexpected
                ds = self[bid]
                path_num = int(ds['pathNumber'].values[0])
                subswath = str(ds['subswath'].values[0])
                burst_num = int(parts[1]) if len(parts) >= 2 else 0
            key = (path_num, subswath)
            if key not in groups:
                groups[key] = []
            groups[key].append((bid, burst_num))

        for key in groups:
            groups[key].sort(key=lambda x: x[1])

        if debug:
            print(f"align_tops(): {len(ids)} bursts")

        def compute_elevation_correction(ds_prev, ds_curr):
            """Compute elevation correction from satellite height difference.

            Returns the apparent elevation offset (meters) to add to current burst.
            """
            prf = float(ds_prev.attrs['PRF'])
            lpb = int(ds_prev.attrs['linesPerBurst'])
            clock_start_prev = float(ds_prev.attrs['clock_start'])
            clock_start_curr = float(ds_curr.attrs['clock_start'])

            ksr_prev = int(ds_prev.attrs['ksr'])
            ker_prev = int(ds_prev.attrs['ker'])
            ksr_curr = int(ds_curr.attrs['ksr'])
            ker_curr = int(ds_curr.attrs['ker'])

            # Lines from burst1 start to burst2 start
            nl = int(np.floor((clock_start_curr - clock_start_prev) * 86400.0 * prf + 0.5))

            # Overlap of valid regions
            valid_overlap_start = max(ksr_prev, nl + ksr_curr)
            valid_overlap_end = min(ker_prev, nl + ker_curr)

            # Stitch at middle of valid overlap
            azi_prev = round((valid_overlap_start + valid_overlap_end) / 2)
            azi_curr = azi_prev - nl

            # Satellite height at overlap midpoint for each burst
            H_prev_start = float(ds_prev.attrs['SC_height_start'])
            H_prev_end = float(ds_prev.attrs['SC_height_end'])
            H_curr_start = float(ds_curr.attrs['SC_height_start'])
            H_curr_end = float(ds_curr.attrs['SC_height_end'])

            H_prev = H_prev_start + (H_prev_end - H_prev_start) * azi_prev / lpb
            H_curr = H_curr_start + (H_curr_end - H_curr_start) * azi_curr / lpb
            dH_satellite = H_prev - H_curr

            # Compute apparent elevation difference from satellite height change
            # Using radar geometry: range, Earth radius, incidence angle
            earth_radius = float(ds_prev.attrs['earth_radius'])
            near_range = float(ds_prev.attrs['near_range'])
            H_avg = (H_prev + H_curr) / 2
            c = earth_radius + H_avg
            ret = earth_radius
            cos_theta = (c**2 + ret**2 - near_range**2) / (2 * c * ret)
            drho_dH = (c - ret * cos_theta) / near_range
            drho_dh = (ret - c * cos_theta) / near_range
            dh_elevation = -dH_satellite * drho_dH / drho_dh

            return dh_elevation

        # Compute cumulative elevation corrections per burst
        burst_ele_corrections = {}

        for (path_num, subswath), burst_list in groups.items():
            if debug:
                print(f"\n=== Path {path_num}, Subswath {subswath} ===")

            ref_bid = burst_list[0][0]
            burst_ele_corrections[ref_bid] = 0.0

            if debug:
                print(f"{ref_bid}: reference burst (ele_corr=0)")

            cumulative_ele = 0.0

            for i in range(1, len(burst_list)):
                prev_bid = burst_list[i - 1][0]
                curr_bid = burst_list[i][0]
                ds_prev = self[prev_bid]
                ds_curr = self[curr_bid]

                dh_elevation = compute_elevation_correction(ds_prev, ds_curr)
                cumulative_ele += dh_elevation
                burst_ele_corrections[curr_bid] = cumulative_ele

                if debug:
                    print(f"{curr_bid}: dh={dh_elevation:.2f}m, cumulative={cumulative_ele:.2f}m")

        # Apply elevation corrections
        output = {}
        for bid, ds in self.items():
            ele_corr = burst_ele_corrections.get(bid, 0.0)

            # Skip if no correction needed
            if abs(ele_corr) < 1e-6:
                output[bid] = ds
                continue

            new_ds = ds.copy()

            # Apply elevation correction
            if 'ele' in ds.data_vars:
                new_ds['ele'] = ds['ele'] + ele_corr

            output[bid] = new_ds

        return type(self)(output)

    def baseline(self, days: int | None = None, meters: float | None = None,
                 invert: bool = False) -> "Baseline":
        """Generate baseline pairs table from the Stack.

        Creates a Baseline DataFrame containing all valid interferometric pairs
        with their temporal and spatial baselines.

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline difference in meters. If None, no limit.
        invert : bool, optional
            If True, invert reference and repeat dates. Default is False.

        Returns
        -------
        Baseline
            DataFrame subclass with columns: ref, rep, ref_baseline, rep_baseline,
            pair, baseline, duration. Has custom plot() and hist() methods.

        Examples
        --------
        >>> # Get all baseline pairs
        >>> bl = stack.baseline()
        >>> bl.plot()  # Plot baseline network
        >>> bl.hist()  # Plot duration histogram

        >>> # Filter by temporal separation
        >>> bl = stack.baseline(days=48)

        >>> # Filter by baseline
        >>> bl = stack.baseline(meters=100)
        """
        import numpy as np
        import pandas as pd
        from .Baseline import Baseline

        if days is None:
            days = int(1e6)

        # Get baseline table: date -> BPR
        # Extract BPR per date from first burst (all bursts have same dates)
        if not self:
            return Baseline()

        first_key = next(iter(self.keys()))
        first_ds = self[first_key]

        if 'date' not in first_ds.dims:
            raise ValueError("Stack must have 'date' dimension to compute baselines")

        # Normalize to date only (no time component)
        dates = pd.DatetimeIndex(first_ds.coords['date'].values).normalize()

        # Get BPR values - they are stored as a data variable along date dimension
        if 'BPR' in first_ds.data_vars:
            bpr_values = first_ds['BPR'].values
        elif 'BPR' in first_ds.coords:
            bpr_values = first_ds.coords['BPR'].values
        else:
            raise ValueError("Stack must have 'BPR' variable to compute baselines")

        # Build baseline table
        tbl = pd.DataFrame({'date': dates, 'BPR': bpr_values}).set_index('date')

        # Generate pairs
        data = []
        for line1 in tbl.itertuples():
            for line2 in tbl.itertuples():
                if not (line1.Index < line2.Index and (line2.Index - line1.Index).days <= days):
                    continue
                if meters is not None and not (abs(line1.BPR - line2.BPR) <= meters):
                    continue

                if not invert:
                    data.append({
                        'ref': line1.Index,
                        'rep': line2.Index,
                        'ref_baseline': np.round(line1.BPR, 2),
                        'rep_baseline': np.round(line2.BPR, 2)
                    })
                else:
                    data.append({
                        'ref': line2.Index,
                        'rep': line1.Index,
                        'ref_baseline': np.round(line2.BPR, 2),
                        'rep_baseline': np.round(line1.BPR, 2)
                    })

        if not data:
            return Baseline(burst_id=first_key, dates=dates)

        df = pd.DataFrame(data).sort_values(['ref', 'rep']).reset_index(drop=True)
        df = df.assign(
            pair=[f'{ref.date()} {rep.date()}' for ref, rep in zip(df['ref'], df['rep'])],
            baseline=df['rep_baseline'] - df['ref_baseline'],
            duration=(df['rep'] - df['ref']).dt.days
        )

        return Baseline(df, burst_id=first_key, dates=dates)
