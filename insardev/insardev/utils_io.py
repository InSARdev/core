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
                n_bursts: int = 2, debug=False, **kwargs):
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
    n_bursts : int, optional
        Number of bursts to process in parallel during save. Default 2.
    debug : bool, optional
        If True, print debug information. Default is False.
    """
    # call the function without data arguments to only open existing store
    if len(args) > 0:
        save(*args, store=store, storage_options=storage_options, compat=compat, caption=caption, n_bursts=n_bursts, debug=debug)
    # Always use -1 for open (fast parallel loading)
    return open(store=store, storage_options=storage_options, compat=compat,
                n_jobs=-1, debug=debug)

def save(*args, store, storage_options: dict[str, str] | None = None, compat: bool = True,
            caption: str | None = 'Saving...', n_bursts: int = 2, debug=False):
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
    n_bursts : int, optional
        Number of bursts to process in parallel. Default 2.
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
    import warnings
    import xarray as xr
    import dask
    import dask.array as da
    import numpy as np
    import zarr
    import gc
    from dask.distributed import get_client, wait
    from tqdm.auto import tqdm

    # Suppress zarr v3 consolidated metadata and .DS_Store warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

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

    # Pre-scan to get total pairs for progress bar
    n_pairs = 0
    for grp, ds in burst_groups[burst_ids[0]]:
        for var_name in ds.data_vars:
            da_xr = ds[var_name]
            if hasattr(da_xr.data, 'dask') and da_xr.ndim >= 3:
                n_pairs = da_xr.shape[0]
                break
        if n_pairs > 0:
            break
    total = n_bursts_total * max(n_pairs, 1)

    # Process bursts in batches of n_bursts
    pbar = tqdm(desc=caption.ljust(25), total=total)
    for batch_start in range(0, n_bursts_total, n_bursts):
        batch_burst_ids = burst_ids[batch_start:batch_start + n_bursts]

        # Collect all datasets for this batch
        batch_datas = {}
        for burst_id in batch_burst_ids:
            for grp, ds in burst_groups[burst_id]:
                batch_datas[grp] = ds

        # Prepare zarr targets for dask arrays
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
                    # Use dask array's spatial chunk sizes for zarr storage,
                    # but always store pair/date dim as 1 so snapshots work
                    # for both 1D (along pairs) and 2D (per pair) processing.
                    chunks = da_xr.data.chunksize
                    if da_xr.ndim >= 3:
                        chunks = (1,) * (da_xr.ndim - 2) + chunks[-2:]
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
                        dtype=da_xr.dtype,
                        dimension_names=dim_names,
                        overwrite=True
                    )

        # Write 3D data in batches of ~2*n_workers.
        # Two paths depending on pair-chunk structure:
        #   Per-pair (chunksize[0]==1): batch by pairs — many pair-chunks,
        #     few spatial blocks. Joint optimize is safe (different vars have
        #     independent source layers).
        #   Merged (chunksize[0]>1): batch by spatial blocks — few pair-chunks,
        #     many spatial blocks. Each slice optimized SEPARATELY (joint
        #     optimize merges shared layers, cull can't remove tasks).
        # Both paths use optimize_graph=False on client.compute to prevent
        # the client from re-merging separately-culled graphs.
        if sources:
            idx_3d = [i for i, s in enumerate(sources) if s.ndim >= 3]
            idx_2d = [i for i, s in enumerate(sources) if s.ndim < 3]
            # 2D sources: small, write directly
            if idx_2d:
                da.store([sources[i] for i in idx_2d],
                         [targets[i] for i in idx_2d], lock=False)
            # 3D sources: write to zarr
            if idx_3d:
                n_pairs = sources[idx_3d[0]].shape[0]
                n_batch_progress = len(batch_burst_ids) * n_pairs

                n_vars = max(len(idx_3d) // len(batch_burst_ids), 1)
                if _client is not None:
                    n_workers = max(len(_client.nthreads()), 1)
                else:
                    n_workers = 1
                batch_size = max(n_workers // n_vars, 1)

                all_per_pair = all(sources[i].chunksize[0] == 1 for i in idx_3d)

                if all_per_pair:
                    # --- Per-pair path: batch by pairs ---
                    # Compute on workers, gather to client, write from client.
                    # Writing from workers via da.store() loses ~0.7% chunks
                    # due to zarr v3 async event loop issues under concurrent
                    # write pressure (even to different chunk files).
                    batch_tgt = [targets[i] for i in idx_3d]
                    for k in range(0, n_pairs, batch_size):
                        k_end = min(k + batch_size, n_pairs)
                        batch_src = list(dask.optimize(
                            *[sources[i][k:k_end] for i in idx_3d]))
                        if _client is not None:
                            futures = _client.compute(batch_src)
                            results = _client.gather(futures)
                        else:
                            results = dask.compute(*batch_src)
                        for vi, arr in enumerate(results):
                            batch_tgt[vi][k:k_end] = arr
                        del results
                        pbar.update((k_end - k) * len(batch_burst_ids))
                else:
                    # Merged pairs (from blockwise ops like trend1d).
                    # Batch by spatial tiles (y-chunk × x-chunk).
                    # Optimize full sources once upfront — slicing the
                    # optimized graph per tile is cheap, while calling
                    # dask.optimize per batch on a deep graph is O(graph).
                    if _client is not None:
                        n_workers = max(len(_client.nthreads()), 1)
                    else:
                        n_workers = 1
                    n_vars = max(len(idx_3d) // len(batch_burst_ids), 1)
                    tiles_per_batch = max(n_workers // n_vars, 1)

                    # Optimize once
                    opt_sources = list(dask.optimize(
                        *[sources[i] for i in idx_3d]))

                    y_chunks = opt_sources[0].chunks[1]
                    x_chunks = opt_sources[0].chunks[2]
                    y_offsets = [0] + list(np.cumsum(y_chunks))
                    x_offsets = [0] + list(np.cumsum(x_chunks))
                    tiles = [(y_offsets[iy], y_offsets[iy+1],
                              x_offsets[ix], x_offsets[ix+1])
                             for iy in range(len(y_chunks))
                             for ix in range(len(x_chunks))]
                    n_tiles = len(tiles)
                    opt_targets = [targets[i] for i in idx_3d]

                    if _client is not None:
                        done_tiles = 0
                        for tb in range(0, n_tiles, tiles_per_batch):
                            tb_end = min(tb + tiles_per_batch, n_tiles)
                            batch_src_all = []
                            batch_tgt_all = []
                            batch_reg_all = []
                            for t in range(tb, tb_end):
                                ys, ye, xs, xe = tiles[t]
                                for vi, src in enumerate(opt_sources):
                                    batch_src_all.append(src[:, ys:ye, xs:xe])
                                    batch_tgt_all.append(opt_targets[vi])
                                    batch_reg_all.append((slice(None), slice(ys, ye), slice(xs, xe)))
                            d = da.store(batch_src_all, batch_tgt_all, lock=False,
                                         regions=batch_reg_all, compute=False)
                            with dask.config.set(delayed_optimize=None):
                                futures = _client.compute(d)
                            wait(futures)
                            # Raise if any task failed (otherwise zarr chunk stays at fill_value=0)
                            _client.gather(futures)
                            new_done = n_pairs * tb_end // n_tiles
                            pbar.update(new_done - done_tiles)
                            done_tiles = new_done
                    else:
                        done_tiles = 0
                        for tb in range(0, n_tiles, tiles_per_batch):
                            tb_end = min(tb + tiles_per_batch, n_tiles)
                            batch_src_all = []
                            batch_tgt_all = []
                            batch_reg_all = []
                            for t in range(tb, tb_end):
                                ys, ye, xs, xe = tiles[t]
                                for vi, src in enumerate(opt_sources):
                                    batch_src_all.append(src[:, ys:ye, xs:xe])
                                    batch_tgt_all.append(opt_targets[vi])
                                    batch_reg_all.append((slice(None), slice(ys, ye), slice(xs, xe)))
                            da.store(batch_src_all, batch_tgt_all, lock=False,
                                     regions=batch_reg_all)
                            new_done = n_pairs * tb_end // n_tiles
                            pbar.update(new_done - done_tiles)
                            done_tiles = new_done
            else:
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

        # Clear memory after each batch
        del sources, targets, batch_datas, grp_info
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

def open(store: str, storage_options: dict[str, str] | None = None, compat: bool = True,
            caption: str | None = 'Opening...', n_jobs: int = -1, debug=False, **kwargs):
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
    n_jobs : int, optional
        Number of parallel jobs to use for opening. Default is -1 (use all available cores).
    debug : bool, optional
        If True, print debug information and use sequential backend. Default is False.

    Returns
    -------
    Stack or Batch or Batches
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
    elif compat and len(dss) == 1 and 'default' in dss:
        # unpack single dataset
        return dss['default']
    elif compat and 'default_0' in dss:
        # default_0, default_1, etc. means that the datasets were saved as a list, unpack them
        return [dss[f'default_{i}'] for i in range(len(dss))]

    return classes[0](dss)
