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
                caption: str | None = 'Snapshotting...',
                n_chunks: int = 1, debug=False, **kwargs):
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
             n_chunks=n_chunks, debug=debug, **kwargs)
    # Always use -1 for open (fast parallel loading)
    return open(store=store, storage_options=storage_options,
                n_jobs=-1, debug=debug)

def save(*args, store, storage_options: dict[str, str] | None = None,
            caption: str | None = 'Saving...', n_chunks: int = 1, debug=False,
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
    import numpy as np
    import zarr
    import gc
    from dask.distributed import get_client, wait as dask_wait, futures_of
    from tqdm.auto import tqdm

    # Suppress zarr v3 consolidated metadata and .DS_Store warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

    interleave = len(args) > 1

    # All args must be dicts (Batch/BatchComplex are dict subclasses)
    datas = {}
    if interleave:
        if len(args) != 2 or not isinstance(args[0], dict) or not isinstance(args[1], dict):
            raise ValueError('Interleaved save requires exactly two Batch dicts')
        for (k0, v0), (k1, v1) in zip(args[0].items(), args[1].items()):
            datas[f'i0_{k0}'] = v0
            datas[f'i1_{k1}'] = v1
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
        f'{cls.__module__}.{cls.__qualname__}'
        for cls in (type(arg) for arg in args)
    ]
    if wrapper is not None:
        root.attrs['__wrapper__'] = f'{wrapper.__module__}.{wrapper.__qualname__}'

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
    n_bursts_total = len(burst_ids)

    # Get dask distributed client (if available)
    try:
        _client = get_client()
    except ValueError:
        _client = None

    # Pre-scan to get total pairs/chunks for progress bar
    n_pairs = 0
    _ref_dask = None
    for grp, ds in burst_groups[burst_ids[0]]:
        for var_name in ds.data_vars:
            da_xr = ds[var_name]
            if hasattr(da_xr.data, 'dask') and da_xr.ndim >= 3:
                n_pairs = da_xr.shape[0]
                _ref_dask = da_xr.data
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
        import numpy as np

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

        def _persist_store(arrays, targets, regions, pbar):
            """Persist on workers, wait, store to zarr.
            Raises on computation error. Store writes from worker memory
            directly to zarr — no data transfer to client."""
            with dask.config.set({'optimization.fuse.active': False}):
                arrays = list(dask.optimize(*arrays))
            persisted = list(dask.persist(*arrays))
            futures = futures_of(persisted)
            n_total = len(futures)
            n_done = 0
            for batch in dask.distributed.as_completed(futures, with_results=False).batches():
                for f in batch:
                    if f.status == 'error':
                        raise f.exception()
                n_done += len(batch)
                pbar.set_postfix_str(f'computing {n_done}/{n_total}')
            pbar.set_postfix_str('writing')
            da.store(persisted, targets, lock=False, regions=regions)
            del persisted
            pbar.set_postfix_str('')

        # Spatial metadata from the best available reference source.
        ref_spatial = ref_3d if ref_3d is not None else \
            next((s for s in sources if s.ndim >= 2 and hasattr(s, 'chunks')), None)
        if ref_spatial is not None:
            n_y = len(ref_spatial.chunks[-2])
            n_x = len(ref_spatial.chunks[-1])
            x_chunks = ref_spatial.chunks[-1]
            x_boundaries = [0] + list(np.cumsum(x_chunks))
        else:
            n_y, n_x = 1, 1

        idx_3d = [i for i, s in enumerate(sources) if s.ndim >= 3]
        idx_2d = [i for i, s in enumerate(sources) if s.ndim < 3]

        if idx_3d:
            # Write 2D sources once (non-batched).
            if idx_2d:
                full_reg = [tuple(slice(None) for _ in range(sources[i].ndim)) for i in idx_2d]
                _persist_store([sources[i] for i in idx_2d],
                               [targets[i] for i in idx_2d], full_reg, pbar)

            if dim0_merged:
                # 1D chunks (all pairs merged): batch by spatial chunks.
                # Each chunk is ~1GB — n_chunks chunks per batch.
                ref_3d_sample = sources[idx_3d[0]]
                n_spatial = len(ref_3d_sample.chunks[-2]) * len(ref_3d_sample.chunks[-1])
                n_3d = len(idx_3d)
                batch_size = max(1, n_chunks * n_workers // max(n_3d, 1))
                x_chunks = ref_3d_sample.chunks[-1]
                x_boundaries = [0] + list(np.cumsum(x_chunks))
                ndim = ref_3d_sample.ndim
                batch_x = max(1, batch_size // len(ref_3d_sample.chunks[-2]))
                n_batches = (len(x_chunks) + batch_x - 1) // batch_x
                pbar.total = n_batches

                for kx in range(0, len(x_chunks), batch_x):
                    kx_end = min(kx + batch_x, len(x_chunks))
                    x0, x1 = x_boundaries[kx], x_boundaries[kx_end]
                    reg = (slice(None),) * (ndim - 1) + (slice(x0, x1),)
                    _persist_store([sources[i][..., x0:x1] for i in idx_3d],
                                   [targets[i] for i in idx_3d],
                                   [reg] * n_3d, pbar)
                    pbar.update(1)
            else:
                # 2D chunks (one pair per chunk): batch by pairs.
                batch_pairs = n_chunks * n_workers
                pbar.total = (n_pairs + batch_pairs - 1) // batch_pairs

                for k in range(0, n_pairs, batch_pairs):
                    k_end = min(k + batch_pairs, n_pairs)
                    _persist_store([sources[i][k:k_end] for i in idx_3d],
                                   [targets[i] for i in idx_3d],
                                   [(slice(k, k_end),)] * len(idx_3d), pbar)
                    pbar.update(1)
        elif idx_2d:
            # All sources are 2D (e.g. ADI).
            pbar.total = 1
            full_reg = [tuple(slice(None) for _ in range(sources[i].ndim)) for i in idx_2d]
            _persist_store([sources[i] for i in idx_2d],
                           [targets[i] for i in idx_2d], full_reg, pbar)
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

    del sources, targets, grp_info
    gc.collect()

    pbar.close()

    # consolidate metadata for the groups in the store
    zarr.consolidate_metadata(store, zarr_format=3)
    del datas, burst_groups
    gc.collect()
    try:
        client = get_client()
        client.run(lambda: __import__('gc').collect())
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
    from pydoc import locate
    # Suppress zarr v3 consolidated metadata and .DS_Store warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

    # Check if store is a string path or a store object
    is_store_object = not isinstance(store, str)

    root = zarr.open_consolidated(store, storage_options=storage_options, zarr_format=3, mode='r')
    classes = [locate(c) for c in root.attrs.get('__class__')]
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
        uid = uuid.uuid4().hex[:8]
        for var in list(ds.data_vars):
            arr = ds[var].data
            if hasattr(arr, 'dask'):
                new_name = f'{grp}-{var}-{uid}'
                ds[var].data = da.map_blocks(
                    lambda x: x, arr, dtype=arr.dtype,
                    name=new_name, meta=arr._meta)
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
        dss0 = {k[len('i0_'):]: v for k, v in dss.items() if k.startswith('i0_')}
        dss1 = {k[len('i1_'):]: v for k, v in dss.items() if k.startswith('i1_')}
        return Batches((classes[0](dss0), classes[1](dss1)))
    result = classes[0](dss)
    wrapper_name = root.attrs.get('__wrapper__')
    if wrapper_name is not None:
        wrapper_cls = locate(wrapper_name)
        if wrapper_cls is not None:
            return wrapper_cls((result,))
    return result
