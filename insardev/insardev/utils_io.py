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
from insardev_toolkit.progressbar_joblib import progressbar_joblib

# Zarr v3 performance fix: disable expensive all_equal() check on every chunk
# See https://github.com/zarr-developers/zarr-python/issues/2710
import zarr
zarr.config.set({'array.write_empty_chunks': True})

# def snapshot_interleave(*args, store: str | None = None, storage_options: dict[str, str] | None = None,
#                         compat: bool = True, n_jobs: int = -1, debug=False):
#     """
#     Save and open a Zarr store or just open it when no data arguments are provided.
#     This function is a shortcut for snapshot(...) with interleave=True.
#     """
#     return snapshot(*args, store=store, storage_options=storage_options, compat=compat, interleave=True, n_jobs=n_jobs, debug=debug)

def snapshot(*args, store: str | None = None, storage_options: dict[str, str] | None = None,
                compat: bool = True, caption: str | None = 'Snapshotting...',
                allow_rechunk: bool = False, n_jobs: int = 1, debug=False):
    """
    Save and open a Zarr store or just open it when no data arguments are provided.
    This function wraps save(...) and open(...) functions.

    Parameters
    ----------
    *args : xr.Dataset or dict[str, xr.Dataset]
        Multiple datasets to save. Can be:
        - Individual xarray.Datasets
        - Dictionary of xarray.Datasets
    store : str, optional
        Path to the store (directory name).
    storage_options : dict[str, str], optional
        Storage options for the store.
    compat : bool, optional
        If True, automatically pack datasets saved as a single dataset into a dictionary.
        If False, only allow a dictionary of datasets. Default is True.
    caption : str, optional
        Caption for the progress bar. Default is 'Snapshotting...'.
    allow_rechunk : bool, optional
        If True, rechunk loaded data to optimal chunk sizes based on dask.config['array.chunk-size'].
        If False (default), rechunk to single spatial chunks (-1 for y,x) for MPS/GPU compatibility.
    n_jobs : int, optional
        Number of bursts to process in parallel during save. Default is 1 (sequential).
        Higher values increase parallelism but also memory usage and file descriptors.
    debug : bool, optional
        If True, print debug information. Default is False.
    """
    # call the function without data arguments to only open existing store
    if len(args) > 0:
        save(*args, store=store, storage_options=storage_options, compat=compat, caption=caption, n_jobs=n_jobs, debug=debug)
    # Always use -1 for open (fast parallel loading)
    return open(store=store, storage_options=storage_options, compat=compat,
                allow_rechunk=allow_rechunk, n_jobs=-1, debug=debug)

def save(*args, store, storage_options: dict[str, str] | None = None, compat: bool = True,
            caption: str | None = 'Saving...', n_jobs: int = 1, debug=False):
    """
    Save multiple xarray.Datasets into one Zarr store, each under its own subgroup.

    Parameters
    ----------
    *args : xr.Dataset or dict[str, xr.Dataset]
        Multiple datasets to save. Can be:
        - Individual xarray.Datasets
        - Dictionary of xarray.Datasets
        - List of xarray.Datasets
    store : str or zarr.storage.Store
        Path to the store directory or a Zarr storage object.
    storage_options : dict[str, str], optional
        Storage options for cloud stores.
    compat : bool, optional
        If True, automatically pack datasets saved as a list or a single dataset into a dictionary.
        If False, only allow a dictionary of datasets. Default is True.
    n_jobs : int, optional
        Number of bursts to process in parallel. Default is 1 (sequential).
        Higher values increase parallelism but also memory usage and file descriptors.
        Interleaved dependent outputs (e.g., phase+conncomp) are always written together
        to share upstream computation.
    debug : bool, optional
        If True, print debug information. Default is False.

    Notes
    -----
    Uses dask.array.store() to write arrays with shared upstream computation.
    Interleaved outputs from the same burst are written together to avoid
    duplicate computation.
    """
    import xarray as xr
    import dask
    import dask.array as da
    import numpy as np
    import zarr
    import gc
    from dask.distributed import get_client
    from tqdm.auto import tqdm

    interleave = len(args) > 1

    # process all arguments into a single dictionary
    datas = {}
    if interleave:
        if len(args) == 2 and (isinstance(args[0], dict) or isinstance(args[1], dict)):
            for (k0, v0), (k1, v1) in zip(args[0].items(), args[1].items()):
                datas[f'i0_{k0}'] = v0
                datas[f'i1_{k1}'] = v1
        elif len(args) == 2 and (isinstance(args[0], xr.Dataset) and isinstance(args[1], xr.Dataset)):
            datas['i0_default'] = args[0]
            datas['i1_default'] = args[1]
        else:
            raise ValueError('Arguments must be two xarray.Datasets or dictionaries of xarray.Datasets when interleave is True')
    elif compat and len(args) == 1 and isinstance(args[0], xr.Dataset):
        datas = {'default': args[0]}
    else:
        for i, arg in enumerate(args):
            if isinstance(arg, xr.Dataset):
                if not compat:
                    raise ValueError('Arguments must be dictionaries of xarray.Datasets when compat is False')
                datas[f'default_{i}'] = arg
            elif isinstance(arg, dict):
                datas.update(arg)
            else:
                raise ValueError('Arguments must be xarray.Datasets or dictionaries of xarray.Datasets when compat is True')

    # Check if store is a string path or a store object
    is_store_object = not isinstance(store, str)

    # open to drop existing store (for string paths) or initialize (for store objects)
    root = zarr.group(store=store, zarr_format=3, overwrite=True)
    root.attrs['__class__'] = [
        f'{cls.__module__}.{cls.__qualname__}'
        for cls in (type(arg) for arg in args)
    ]

    # Group datasets by burst (keeping interleaved pairs together for shared computation)
    burst_groups = {}  # burst_id -> [(grp, ds), ...]
    for grp, ds in datas.items():
        # Extract burst_id (strip i0_/i1_ prefix for interleaved data)
        if grp.startswith(('i0_', 'i1_')):
            burst_id = grp.split('_', 1)[1]
        else:
            burst_id = grp
        burst_groups.setdefault(burst_id, []).append((grp, ds))

    burst_ids = list(burst_groups.keys())
    n_bursts = len(burst_ids)

    # Process bursts in batches of n_jobs
    for batch_start in tqdm(range(0, n_bursts, n_jobs), desc=caption.ljust(25), total=(n_bursts + n_jobs - 1) // n_jobs):
        batch_burst_ids = burst_ids[batch_start:batch_start + n_jobs]

        # Collect all datasets for this batch
        batch_datas = {}
        for burst_id in batch_burst_ids:
            for grp, ds in burst_groups[burst_id]:
                batch_datas[grp] = ds

        # Write this batch using dask.array.store() for shared computation
        sources = []
        targets = []
        grp_info = []  # Track (grp, ds, grp_store) for metadata writing

        for grp, ds in batch_datas.items():
            # Create zarr group for this dataset
            if is_store_object:
                grp_path = grp
                grp_store = store
            else:
                grp_path = None
                grp_store = f'{store}/{grp}'

            # Initialize the group structure
            ds_grp = zarr.group(store=grp_store, path=grp_path, zarr_format=3, overwrite=True)
            grp_info.append((grp, ds, grp_store))

            for var_name in ds.data_vars:
                da_xr = ds[var_name]
                # Get dimension names for zarr v3 metadata
                dim_names = list(da_xr.dims)
                if hasattr(da_xr.data, 'dask'):
                    # Create zarr array target with proper shape/chunks/dtype and dimension_names
                    chunks = da_xr.data.chunksize
                    z = ds_grp.create_array(
                        var_name,
                        shape=da_xr.shape,
                        chunks=chunks,
                        dtype=da_xr.dtype,
                        dimension_names=dim_names,
                        overwrite=True
                    )
                    sources.append(da_xr.data)
                    targets.append(z)
                else:
                    # Non-dask array - write directly with dimension_names
                    ds_grp.create_array(
                        var_name,
                        data=np.asarray(da_xr.data),
                        chunks=da_xr.shape,
                        dtype=da_xr.dtype,
                        dimension_names=dim_names,
                        overwrite=True
                    )

        # Store all dask arrays in this batch together - shares upstream computation
        if sources:
            da.store(sources, targets, lock=False)

        # Write xarray metadata (coords, attrs) AFTER data is written
        for grp, ds, grp_store in grp_info:
            ds_clean = ds.copy()
            for v in list(ds_clean.data_vars):
                ds_clean = ds_clean.drop_vars(v)
            for coord in ds_clean.coords:
                ds_clean[coord].encoding.pop('chunks', None)
            if is_store_object:
                ds_clean.to_zarr(store=store, group=grp, mode='a', consolidated=True, zarr_format=3)
            else:
                ds_clean.to_zarr(store=grp_store, mode='a', consolidated=True, zarr_format=3)

        # Clear memory after each batch
        del sources, targets, batch_datas, grp_info
        gc.collect()

    # consolidate metadata for the groups in the store
    zarr.consolidate_metadata(store, zarr_format=3)
    del datas, burst_groups
    gc.collect()
    try:
        client = get_client()
        client.run(lambda: __import__('gc').collect())
    except ValueError:
        pass

def open(store: str, storage_options: dict[str, str] | None = None, compat: bool = True,
            caption: str | None = 'Opening...', allow_rechunk: bool = False, n_jobs: int = -1, debug=False):
    """
    Load a Zarr store created by save(...).

    Parameters
    ----------
    store : str | zarr.storage.Store
        Path to the store (directory '<name>.zarr') or a Zarr storage object.
    storage_options : dict[str, str], optional
        Storage options for the store.
    compat : bool, optional
        If True, automatically unpack datasets saved as a list or a single dataset.
        If False, return a dictionary of datasets. Default is True.
    allow_rechunk : bool, optional
        If True, rechunk to optimal chunk sizes based on dask.config['array.chunk-size'].
        If False (default), rechunk to single spatial chunks (-1 for y,x) for MPS/GPU compatibility.
    n_jobs : int, optional
        Number of parallel jobs to use for opening. Default is -1 (use all available cores).
    debug : bool, optional
        If True, print debug information and use sequential backend. Default is False.

    Returns
    -------
    Stack or Batch or Batches
    """
    import os
    import dask
    import zarr
    import xarray as xr
    import joblib
    from tqdm.auto import tqdm
    from pydoc import locate
    from .utils_dask import rechunk2d

    # Check if store is a string path or a store object
    is_store_object = not isinstance(store, str)

    root = zarr.open_consolidated(store, storage_options=storage_options, zarr_format=3, mode='r')
    classes = [locate(c) for c in root.attrs.get('__class__')]
    interleave = len(classes) > 1
    groups = list(root.group_keys())

    joblib_backend = 'sequential' if debug else 'threading'

    def _load_grp(grp):
        import rioxarray
        # Use consolidated=False for subgroups - they share root's consolidated metadata
        if is_store_object:
            ds = xr.open_zarr(store, group=grp, storage_options=storage_options, consolidated=False, zarr_format=3)
        else:
            ds = xr.open_zarr(f'{store}/{grp}', storage_options=storage_options, consolidated=False, zarr_format=3)
        # restore rioxarray CRS from spatial_ref coordinate/variable if present
        if ds.rio.crs is None:
            # check both coords and data_vars for spatial_ref
            spatial_ref_var = None
            if 'spatial_ref' in ds.coords:
                spatial_ref_var = ds.coords['spatial_ref']
            elif 'spatial_ref' in ds.data_vars:
                spatial_ref_var = ds.data_vars['spatial_ref']
            if spatial_ref_var is not None:
                spatial_ref_attrs = spatial_ref_var.attrs
                crs_wkt = spatial_ref_attrs.get('crs_wkt') or spatial_ref_attrs.get('spatial_ref')
                if crs_wkt:
                    ds = ds.rio.write_crs(crs_wkt)
        return grp, ds

    with progressbar_joblib.progressbar_joblib(tqdm(desc=caption.ljust(25), total=len(groups))) as progress_bar:
        results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(_load_grp) (grp) for grp in groups)
    dss = dict(results)
    del results

    # Apply rechunking based on allow_rechunk setting
    # allow_rechunk=True: rechunk to dask config size (for efficient processing)
    # allow_rechunk=False: rechunk to single spatial chunks (for MPS/GPU compatibility)
    for key, ds in dss.items():
        rechunked_vars = {}
        for var_name in ds.data_vars:
            arr = ds[var_name]
            if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                continue

            if allow_rechunk:
                # Rechunk to dask config size
                y_size, x_size = arr.shape[-2], arr.shape[-1]
                element_bytes = arr.dtype.itemsize
                optimal = rechunk2d((y_size, x_size), element_bytes)
                if arr.ndim == 3:
                    chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                else:
                    chunks = optimal
            else:
                # Single spatial chunk for MPS/GPU compatibility
                if arr.ndim == 3:
                    chunks = {arr.dims[0]: 1, 'y': -1, 'x': -1}
                else:
                    chunks = {'y': -1, 'x': -1}

            rechunked_vars[var_name] = arr.chunk(chunks)
        if rechunked_vars:
            dss[key] = ds.assign(rechunked_vars)

    if debug:
        if allow_rechunk:
            dask_chunk_mb = int(dask.config.get('array.chunk-size', '128 MiB').replace(' MiB', '').replace(' MB', ''))
            print(f"NOTE open: rechunking to dask.config['array.chunk-size']={dask_chunk_mb} MB")
        else:
            print(f"NOTE open: using single spatial chunk (allow_rechunk=False)")

    if interleave:
        # unpack interleaved datasets and return as Batches for chaining
        from .Batch import Batches
        dss0 = {k[len('i0_'):]: v for k, v in dss.items() if k.startswith('i0_')}
        dss1 = {k[len('i1_'):]: v for k, v in dss.items() if k.startswith('i1_')}
        return Batches((classes[0](dss0), classes[1](dss1)))
    elif compat and len(dss) == 1 and 'default' in dss:
        # unpack single dataset
        return dss['default']
    elif compat and 'default_0' in dss:
        # default_0, default_1, etc. means that the datasets were saved as a list, unpack them
        return [dss[f'default_{i}'] for i in range(len(dss))]

    return classes[0](dss)
