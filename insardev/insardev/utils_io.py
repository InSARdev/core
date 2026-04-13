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

# TensorStore target for concurrent-safe zarr writes from dask workers.
# zarr-python v3 has no synchronization for concurrent writes (zarr-python#1596),
# and da.store lock parameter does not work with distributed (dask#12109).
# TensorStore handles concurrent writes safely via its own locking.
# Imported lazily inside _TensorStoreTarget._open() to avoid hard dependency at load time.
import numpy as np

def _store_path_to_kvstore(store_path, storage_options=None):
    """Convert filesystem path to TensorStore kvstore spec."""
    import os
    opts = storage_options or {}
    if store_path.startswith('s3://'):
        parts = store_path[5:].split('/', 1)
        kvstore = {'driver': 's3', 'bucket': parts[0], 'path': parts[1] if len(parts) > 1 else ''}
        if 'endpoint_url' in opts: kvstore['endpoint'] = opts['endpoint_url']
        if 'key' in opts and 'secret' in opts:
            kvstore['aws_credentials'] = {'access_key': opts['key'], 'secret_key': opts['secret']}
        if 'region_name' in opts: kvstore['aws_region'] = opts['region_name']
        return kvstore
    elif store_path.startswith(('gs://', 'gcs://')):
        prefix = 'gs://' if store_path.startswith('gs://') else 'gcs://'
        parts = store_path[len(prefix):].split('/', 1)
        return {'driver': 'gcs', 'bucket': parts[0], 'path': parts[1] if len(parts) > 1 else ''}
    return {'driver': 'file', 'path': os.path.abspath(store_path) + '/'}

def _zarr_codecs_to_ts(zarr_codecs):
    """Convert zarr v3 codec config to TensorStore codec list."""
    ts_codecs = []
    for codec in zarr_codecs:
        name = codec.name if hasattr(codec, 'name') else str(codec)
        if 'bytes' in name.lower():
            endian = getattr(codec, 'endian', None)
            if endian is not None:
                endian = str(endian.value) if hasattr(endian, 'value') else str(endian)
            ts_codecs.append({'name': 'bytes', 'configuration': {'endian': endian or 'little'}})
        elif 'zstd' in name.lower():
            ts_codecs.append({'name': 'zstd', 'configuration': {'level': int(getattr(codec, 'level', 0))}})
        elif 'blosc' in name.lower():
            ts_codecs.append({'name': 'blosc', 'configuration': {
                'cname': str(getattr(codec, 'cname', 'lz4')), 'clevel': int(getattr(codec, 'clevel', 5))}})
        elif 'gzip' in name.lower():
            ts_codecs.append({'name': 'gzip', 'configuration': {'level': int(getattr(codec, 'level', 6))}})
    return ts_codecs or [{'name': 'bytes', 'configuration': {'endian': 'little'}}]

def _make_ts_target(zarr_array, store_path, var_name, group_path=None, storage_options=None):
    """Build a picklable TensorStore target from an existing zarr v3 array."""
    array_path = f'{store_path}/{group_path}/{var_name}' if group_path else f'{store_path}/{var_name}'
    dtype = zarr_array.dtype
    dtype_map = {
        np.float32: 'float32', np.float64: 'float64',
        np.int16: 'int16', np.int32: 'int32', np.int64: 'int64',
        np.uint8: 'uint8', np.uint16: 'uint16', np.uint32: 'uint32',
        np.complex64: 'complex64', np.complex128: 'complex128',
    }
    codecs = zarr_array.metadata.codecs if hasattr(zarr_array.metadata, 'codecs') else []
    spec = {
        'driver': 'zarr3',
        'kvstore': _store_path_to_kvstore(array_path, storage_options),
        'metadata': {
            'shape': list(zarr_array.shape),
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': list(zarr_array.chunks)}},
            'codecs': _zarr_codecs_to_ts(codecs),
            'data_type': dtype_map.get(dtype.type, str(dtype)),
        },
    }
    fill = zarr_array.fill_value
    if fill is not None:
        if np.issubdtype(dtype, np.complexfloating):
            spec['metadata']['fill_value'] = [float('nan'), 0.0] if np.isnan(fill) else [float(np.real(fill)), float(np.imag(fill))]
        elif np.issubdtype(dtype, np.floating):
            spec['metadata']['fill_value'] = float(fill)
        else:
            spec['metadata']['fill_value'] = int(fill)
    return _TensorStoreTarget(spec)

class _TensorStoreTarget:
    """Picklable da.store target that writes via TensorStore (concurrent-safe)."""
    def __init__(self, spec):
        self.spec = spec
        self._store = None
    def _open(self):
        if self._store is None:
            import tensorstore as ts
            self._store = ts.open(self.spec, open=True, write=True).result()
        return self._store
    def __setitem__(self, key, value):
        self._open()[key] = value
    @property
    def shape(self):
        return tuple(self.spec['metadata']['shape'])
    def __getstate__(self):
        return {'spec': self.spec}
    def __setstate__(self, state):
        self.spec = state['spec']
        self._store = None


# def snapshot_interleave(*args, store: str | None = None, storage_options: dict[str, str] | None = None,
#                         compat: bool = True, n_jobs: int = -1, debug=False):
#     """
#     Save and open a Zarr store or just open it when no data arguments are provided.
#     This function is a shortcut for snapshot(...) with interleave=True.
#     """
#     return snapshot(*args, store=store, storage_options=storage_options, compat=compat, interleave=True, n_jobs=n_jobs, debug=debug)

def snapshot(*args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Snapshotting...',
                debug=False, **kwargs):
    """
    Save and open a Zarr store or just open it when no data arguments are provided.
    This function wraps save(...) and open(...) functions.

    Parameters
    ----------
    *args : dict[str, xr.Dataset]
        Batch/BatchComplex dicts to save. Each arg is a dict mapping burst ID to Dataset.
    store : str, optional
        Path to the store (directory name).
    storage_options : dict[str, str], optional
        Storage options for the store.
    caption : str, optional
        Caption for the progress bar. Default is 'Snapshotting...'.
    debug : bool, optional
        If True, print debug information. Default is False.
    """
    # call the function without data arguments to only open existing store
    if len(args) > 0:
        save(*args, store=store, storage_options=storage_options, caption=caption,
             debug=debug, **kwargs)
    # Release input references and worker memory before opening the result
    import gc
    del args, kwargs
    gc.collect()
    try:
        from dask.distributed import get_client
        get_client().run(gc.collect)
    except (ValueError, ImportError):
        pass
    # Always use -1 for open (fast parallel loading)
    return open(store=store, storage_options=storage_options,
                n_jobs=-1, debug=debug)

def save(*args, store, storage_options: dict[str, str] | None = None,
            caption: str | None = 'Saving...', debug=False,
            wrapper: type | None = None):
    """
    Save Batch/BatchComplex dicts into one Zarr store, each burst under its own subgroup.

    Parameters
    ----------
    *args : dict[str, xr.Dataset]
        One or two Batch/BatchComplex dicts to save. Each maps burst ID to Dataset.
        Two args = interleaved save (e.g., phase + correlation).
    store : str or zarr.storage.Store
        Path to the store directory or a Zarr storage object.
    storage_options : dict[str, str], optional
        Storage options for cloud stores.
    debug : bool, optional
        If True, print debug information. Default is False.

    Notes
    -----
    Uses dask.array.store() to write arrays with shared upstream computation.
    Interleaved outputs from the same burst are written together to avoid
    duplicate computation.
    """
    import warnings
    import xarray as xr
    import dask
    import dask.array as da
    import zarr
    import gc
    from dask.distributed import get_client, futures_of
    from tqdm.auto import tqdm

    # Suppress zarr v3 consolidated metadata and .DS_Store warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

    interleave = len(args) > 1

    # All args must be dicts (Batch/BatchComplex are dict subclasses)
    datas = {}
    if interleave:
        for idx, arg in enumerate(args):
            if not isinstance(arg, dict):
                raise ValueError(f'Interleaved save: arg {idx} is {type(arg).__name__}, expected Batch dict')
            for k, v in arg.items():
                datas[f'i{idx}_{k}'] = v
    else:
        if not isinstance(args[0], dict):
            raise ValueError(f'Expected Batch dict, got {type(args[0]).__name__}')
        datas.update(args[0])

    # Check if store is a string path or a store object
    is_store_object = not isinstance(store, str)

    # Remove existing store manually before zarr.group to avoid OS errors
    # (macOS .DS_Store files from Spotlight can prevent zarr's shutil.rmtree)
    import os, shutil
    if not is_store_object and os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)
    root = zarr.group(store=store, zarr_format=3, overwrite=True)
    root.attrs['__class__'] = [
        type(arg).__name__ for arg in args
    ]
    if wrapper is not None:
        root.attrs['__wrapper__'] = wrapper.__name__

    # Group datasets by burst (keeping interleaved pairs together for shared computation)
    burst_groups = {}  # burst_id -> [(grp, ds), ...]
    for grp, ds in datas.items():
        burst_id = grp.split('_', 1)[1] if interleave else grp
        burst_groups.setdefault(burst_id, []).append((grp, ds))

    burst_ids = list(burst_groups.keys())

    # Get dask distributed client (if available)
    try:
        _client = get_client()
    except ValueError:
        _client = None

    # Pre-scan to get total pairs/chunks for progress bar
    n_pairs = 0
    for grp, ds in burst_groups[burst_ids[0]]:
        for var_name in ds.data_vars:
            da_xr = ds[var_name]
            if hasattr(da_xr.data, 'dask') and da_xr.ndim >= 3:
                n_pairs = da_xr.shape[0]
                break
        if n_pairs > 0:
            break

    # Process all bursts together
    pbar = tqdm(desc=caption.ljust(25), total=0)  # set total after path selection

    # Prepare zarr targets for dask arrays
    sources = []
    targets = []
    grp_info = []  # Track (grp, ds, grp_store) for metadata writing

    for burst_id in burst_ids:
        for grp, ds in burst_groups[burst_id]:
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
                    chunks = da_xr.data.chunksize
                    z = ds_grp.create_array(
                        var_name,
                        shape=da_xr.shape,
                        chunks=chunks,
                        dtype=da_xr.dtype,
                        dimension_names=dim_names,
                        overwrite=True,
                        fill_value=np.nan if (np.issubdtype(da_xr.dtype, np.floating) or np.issubdtype(da_xr.dtype, np.complexfloating)) else 0,
                    )
                    sources.append(da_xr.data)
                    if da_xr.ndim >= 3 and not is_store_object:
                        targets.append(_make_ts_target(z, store, var_name, group_path=grp,
                                                        storage_options=storage_options))
                    else:
                        targets.append(z)
                else:
                    # Non-dask array - write directly with dimension_names
                    ds_grp.create_array(
                        var_name,
                        data=np.asarray(da_xr.data),
                        chunks=da_xr.shape,
                        dimension_names=dim_names,
                        overwrite=True
                    )

    # Write dask arrays to zarr in memory-controlled batches.
    # Each batch slice is optimized (dask.optimize culls unused keys).
    if sources:
        if _client is not None:
            n_workers = max(len(_client.nthreads()), 1)
        else:
            n_workers = 1

        ref_3d = next((s for s in sources if s.ndim >= 3), None)
        n_pairs = ref_3d.shape[0] if ref_3d is not None else 1
        dim0_merged = ref_3d is not None and ref_3d.chunksize[0] == ref_3d.shape[0]

        if debug:
            print(f'DEBUG save: {len(sources)} sources, ref_3d={ref_3d is not None}, '
                  f'n_pairs={n_pairs}, dim0_merged={dim0_merged}')
            for i, s in enumerate(sources):
                print(f'  source[{i}]: ndim={s.ndim}, shape={s.shape}, '
                      f'chunks={s.chunks if hasattr(s, "chunks") else "N/A"}')

        idx_3d = [i for i, s in enumerate(sources) if s.ndim >= 3]
        idx_2d = [i for i, s in enumerate(sources) if s.ndim < 3]

        # Write 2D sources directly (small, no contention)
        if idx_2d:
            da.store([dask.optimize(sources[i])[0] for i in idx_2d],
                     [targets[i] for i in idx_2d], lock=False)

        # Write 3D sources via TensorStore targets (concurrent-safe, no lock needed).
        if idx_3d:
            if dim0_merged:
                # 1D chunks: single pass, scheduler pipelines optimally
                arrays = list(dask.optimize(*[sources[i] for i in idx_3d]))
                store_arr = da.store(arrays, [targets[i] for i in idx_3d],
                                    lock=False, compute=False)
                if _client is not None:
                    persisted = _client.persist(store_arr)
                    all_futures = futures_of(persisted)
                    pbar.total = len(all_futures)
                    for batch in dask.distributed.as_completed(all_futures, with_results=False).batches():
                        for f in batch:
                            if f.status == 'error':
                                raise f.exception()
                            f.cancel()
                        pbar.update(len(batch))
                    del all_futures, persisted
                else:
                    pbar.total = 1
                    dask.compute(store_arr)
                    pbar.update(1)
                del arrays, store_arr
            else:
                # 2D chunks: batch by pairs, same pipeline as 1D per batch.
                batch_pairs = n_workers
                pbar.total = (n_pairs + batch_pairs - 1) // batch_pairs
                for k in range(0, n_pairs, batch_pairs):
                    k_end = min(k + batch_pairs, n_pairs)
                    arrays = list(dask.optimize(*[sources[i][k:k_end] for i in idx_3d]))
                    store_arr = da.store(arrays, [targets[i] for i in idx_3d],
                                         lock=False, compute=False,
                                         regions=[(slice(k, k_end),)] * len(arrays))
                    if _client is not None:
                        persisted = _client.persist(store_arr)
                        all_futures = futures_of(persisted)
                        n_total = len(all_futures)
                        n_done = 0
                        for batch in dask.distributed.as_completed(all_futures, with_results=False).batches():
                            for f in batch:
                                if f.status == 'error':
                                    raise f.exception()
                                f.cancel()
                            n_done += len(batch)
                            pbar.set_postfix_str(f'{n_done}/{n_total}')
                            pbar.refresh()
                        del all_futures, persisted
                    else:
                        dask.compute(store_arr)
                    del arrays, store_arr
                    pbar.set_postfix_str('')
                    pbar.update(1)

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

    del sources, targets, grp_info, idx_3d, idx_2d
    pbar.close()

    # consolidate metadata for the groups in the store
    zarr.consolidate_metadata(store, zarr_format=3)
    del datas, burst_groups

    gc.collect()
    try:
        client = get_client()
        client.run(gc.collect)
    except ValueError:
        pass

def open(store: str, storage_options: dict[str, str] | None = None,
            caption: str | None = 'Opening...', n_jobs: int = -1, debug=False, **kwargs):
    """
    Load a Zarr store created by save(...).

    Parameters
    ----------
    store : str | zarr.storage.Store
        Path to the store (directory) or a Zarr storage object.
    storage_options : dict[str, str], optional
        Storage options for the store.
    n_jobs : int, optional
        Number of parallel jobs to use for opening. Default is -1 (use all available cores).
    debug : bool, optional
        If True, print debug information and use sequential backend. Default is False.

    Returns
    -------
    Batch or Batches
    """
    import os
    import warnings
    import dask
    import zarr
    import xarray as xr
    import joblib
    from tqdm.auto import tqdm
    # Suppress zarr v3 consolidated metadata and .DS_Store warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

    # Check if store is a string path or a store object
    is_store_object = not isinstance(store, str)

    root = zarr.open_consolidated(store, storage_options=storage_options, zarr_format=3, mode='r')
    from .Batch import Batch, BatchComplex, BatchUnit, BatchWrap
    _class_map = {
        'Batch': Batch, 'BatchComplex': BatchComplex,
        'BatchUnit': BatchUnit, 'BatchWrap': BatchWrap,
    }
    classes = [_class_map.get(c.rsplit('.', 1)[-1], Batch) for c in root.attrs.get('__class__')]
    interleave = len(classes) > 1
    groups = list(root.group_keys())

    joblib_backend = 'sequential' if debug else 'threading'

    def _load_grp(grp):
        import uuid
        import dask.array as da
        import rioxarray
        # Use consolidated=False for subgroups - they share root's consolidated metadata
        if is_store_object:
            ds = xr.open_zarr(store, group=grp, storage_options=storage_options, consolidated=False, zarr_format=3)
        else:
            ds = xr.open_zarr(f'{store}/{grp}', storage_options=storage_options, consolidated=False, zarr_format=3)
        # Unique dask layer to separate different opens of the same zarr.
        # NOTE: The underlying open_dataset keys are deterministic from the
        # path. If the zarr was overwritten between opens, the scheduler may
        # serve stale cached source reads. This is mitigated by:
        # 1. save() calls gc.collect + client.run(gc.collect) to release old futures
        # 2. map_blocks identity prevents downstream key collisions
        # 3. Stale source reads only occur if old futures survive gc — rare in practice
        # Note: xr.open_zarr keys are deterministic from the path.
        # Stale cache collisions are mitigated by save()'s gc.collect cleanup.
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
        # xr.open_zarr does not preserve coordinate ordering — reorder to
        # match the dimension order of data variables so that downstream
        # code (e.g. plot.scatter) picks the correct default axis.
        # Zero-cost: reuses same dask arrays, only rebuilds metadata.
        dim_order = None
        for vname in ds.data_vars:
            if ds[vname].ndim >= 2:
                dim_order = list(ds[vname].dims)
                break
        if dim_order is not None:
            dim_coords = [c for c in dim_order if c in ds.coords]
            non_dim_coords = [c for c in ds.coords if c not in set(dim_order)]
            ordered = dim_coords + non_dim_coords
            if list(ds.coords) != ordered:
                new_vars = {v: (ds[v].dims, ds[v].data, ds[v].attrs) for v in ds.data_vars}
                new_coords = {}
                for c in ordered:
                    coord = ds.coords[c]
                    new_coords[c] = (coord.dims, coord.values, coord.attrs) if coord.dims \
                        else xr.Variable((), coord.values, coord.attrs)
                ds = xr.Dataset(new_vars, coords=new_coords, attrs=ds.attrs)
        return grp, ds

    with progressbar_joblib.progressbar_joblib(tqdm(desc=caption.ljust(25), total=len(groups))) as progress_bar:
        results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(_load_grp) (grp) for grp in groups)
    dss = dict(results)
    del results

    # Read chunks exactly as stored in zarr — no rechunking.
    # The zarr chunks match what was computed and stored by save().

    if interleave:
        # unpack interleaved datasets and return as Batches for chaining
        from .Batch import Batches
        batches = []
        for idx, cls in enumerate(classes):
            prefix = f'i{idx}_'
            batch_dss = {k[len(prefix):]: v for k, v in dss.items() if k.startswith(prefix)}
            batches.append(cls(batch_dss))
        return Batches(batches)
    result = classes[0](dss)
    wrapper_name = root.attrs.get('__wrapper__')
    if wrapper_name is not None:
        from .Batch import Batches
        _wrapper_map = {'Batches': Batches}
        wrapper_cls = _wrapper_map.get(wrapper_name.rsplit('.', 1)[-1])
        if wrapper_cls is not None:
            return wrapper_cls((result,))
    return result
