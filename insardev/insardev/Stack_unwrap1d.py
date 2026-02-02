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
"""
1D phase unwrapping along the temporal dimension using L1-norm IRLS.

Optimized implementation with:
- CPU batched linear solve (faster than GPU for small matrices)
- Efficient einsum-based AtWA computation
- Memory-efficient chunked processing
- Direct time series output option
"""
from .BatchCore import BatchCore
from . import utils_unwrap1d
import numpy as np


class Stack_unwrap1d(BatchCore):
    """1D phase unwrapping along the temporal dimension using L1-norm IRLS."""

    def lstsq(self, data, weight=None, device='auto', cumsum=True, debug=False):
        """
        Weighted least squares network inversion to date-based time series.

        .. deprecated::
            Use ``data.lstsq(weight=corr)`` on Batch instead.
            This Stack method will be removed in a future version.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase. Uses PyTorch batched least squares for GPU acceleration.

        Parameters
        ----------
        data : Batch
            Unwrapped phase data with 'pair' dimension.
        weight : BatchUnit, optional
            Correlation weights for each pair.
        device : str, optional
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        cumsum : bool, optional
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        Batch
            Phase time series with 'date' dimension instead of 'pair'.

        Notes
        -----
        Typical workflow:
        1. stack.unwrap1d(intf, corr) - unwrap pairs temporally
        2. stack.lstsq(unwrapped, corr) - network inversion to dates

        Examples
        --------
        >>> unwrapped = stack.unwrap1d(intf, corr)
        >>> displacement = stack.lstsq(unwrapped, corr)
        """
        import xarray as xr
        import pandas as pd
        from .Batch import Batch, BatchWrap, BatchUnit

        # Validate input types
        if isinstance(data, BatchWrap):
            raise TypeError(
                'lstsq() requires unwrapped phase (Batch), got BatchWrap. '
                'Use unwrap1d() first to unwrap wrapped phase data.'
            )
        if not isinstance(data, Batch):
            raise TypeError(
                f'data must be Batch (unwrapped phase), got {type(data).__name__}.'
            )
        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(data) to convert correlation data.'
            )

        # Process each burst
        results = {}
        for key in data.keys():
            ds = data[key]
            w_ds = weight[key] if weight is not None else None

            result_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for pol in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da = ds[pol]
                w_da = w_ds[pol] if w_ds is not None else None

                result = self._lstsq_dataarray(da, w_da, device, cumsum, debug)
                result_vars[pol] = result

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            if ds.rio.crs is not None:
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        return Batch(results)

    def _lstsq_dataarray(self, data, weight, device, cumsum, debug):
        """Internal method for lstsq on DataArray - LAZY dask processing."""
        import xarray as xr
        import pandas as pd
        import dask
        import dask.array as da

        # Extract pairs from data coords (inline _get_pairs logic for DataArray)
        refs = data.coords['ref'].values
        reps = data.coords['rep'].values
        refs = refs if isinstance(refs, np.ndarray) else [refs]
        reps = reps if isinstance(reps, np.ndarray) else [reps]
        pair_dates = [(str(pd.Timestamp(r)), str(pd.Timestamp(p))) for r, p in zip(refs, reps)]

        # Build incidence matrix once (needed for all blocks)
        A, dates = utils_unwrap1d.build_incidence_matrix(pair_dates)
        n_dates = len(dates)
        n_pairs = len(pair_dates)

        # Use dask auto-chunking for y,x based on actual memory usage
        # WA tensor is (n_pixels, n_pairs, n_intervals) - dominant memory consumer
        # Memory per pixel = n_pairs * n_intervals * 4 bytes
        # Use 2D auto-chunks with memory-equivalent dtype
        n_intervals = n_dates - 1
        mem_per_pixel = n_pairs * n_intervals * 4
        auto_chunks = dask.array.core.normalize_chunks(
            'auto', (data.y.size, data.x.size), dtype=np.dtype(f'V{mem_per_pixel}')
        )
        chunks_y, chunks_x = auto_chunks[0][0], auto_chunks[1][0]

        # Rechunk: all pairs together (-1), auto-chunked y,x
        first_dim = data.dims[0]
        data = data.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})
        if weight is not None:
            weight = weight.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})

        # Use blockwise to avoid embedding large arrays in the graph
        def process_block(data_block, weight_block=None):
            """Process a spatial block - all pairs, subset of y,x."""
            # PyTorch batched weighted least squares
            ts_block, _ = utils_unwrap1d.lstsq_to_dates_numpy(
                data_block, weight_block, pair_dates,
                device=device, cumsum=cumsum, debug=False
            )
            return ts_block.astype(np.float32)

        data_dask = data.data
        if weight is not None:
            weight_dask = weight.data
            result_dask = da.map_blocks(
                process_block, data_dask, weight_dask,
                dtype=np.float32,
                drop_axis=0,
                new_axis=0,
                chunks=(n_dates,) + data_dask.chunks[1:],
            )
        else:
            def process_block_no_weight(data_block):
                return process_block(data_block, None)
            result_dask = da.map_blocks(
                process_block_no_weight, data_dask,
                dtype=np.float32,
                drop_axis=0,
                new_axis=0,
                chunks=(n_dates,) + data_dask.chunks[1:],
            )

        # Rechunk to (1, -1, -1) for efficient per-slice downstream operations (e.g., plot)
        ts_dask = result_dask.rechunk({0: 1, 1: -1, 2: -1})

        # Build coordinates
        coords = {
            'date': pd.to_datetime(dates),
            'y': data.coords['y'],
            'x': data.coords['x']
        }

        return xr.DataArray(
            ts_dask,
            coords=coords,
            dims=('date', 'y', 'x'),
            name='displacement'
        )

    def unwrap1d(self, data, weight=None, device='auto', max_iter=5,
                 epsilon=0.1, batch_size=50000, debug=False):
        """
        L1-norm IRLS temporal phase unwrapping returning unwrapped pairs.

        .. deprecated::
            Use ``phase.unwrap1d(weight=corr)`` on BatchWrap instead.
            This Stack method will be removed in a future version.

        Performs temporal unwrapping across the interferogram network,
        applying 2π corrections to make pairs consistent.

        Parameters
        ----------
        data : BatchWrap or Batch
            Phase data with 'pair' dimension (wrapped or 2D-unwrapped).
        weight : BatchUnit, optional
            Correlation weights for each pair.
        device : str, optional
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        max_iter : int, optional
            Maximum IRLS iterations. Default 5.
        epsilon : float, optional
            IRLS regularization parameter. Default 0.1.
        batch_size : int, optional
            Pixels per batch for memory efficiency. Default 50000.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        Batch
            Temporally unwrapped phase data with 'pair' dimension.

        Notes
        -----
        Typical workflow:
        1. stack.unwrap1d(intf, corr) - unwrap pairs temporally
        2. stack.lstsq(unwrapped, corr) - network inversion to dates

        Examples
        --------
        >>> unwrapped = stack.unwrap1d(intf, corr)
        >>> displacement = stack.lstsq(unwrapped, corr)
        """
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        # Auto-detect device based on Dask cluster resources and hardware
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack_unwrap1d._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: unwrap1d using device={device}")

        # Validate input types
        if not isinstance(data, (Batch, BatchWrap)):
            raise TypeError(
                f'data must be BatchWrap or Batch, got {type(data).__name__}.'
            )
        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(data) to convert correlation data.'
            )

        # Process each burst
        results = {}
        for key in data.keys():
            ds = data[key]
            w_ds = weight[key] if weight is not None else None

            result_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for pol in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da = ds[pol]
                w_da = w_ds[pol] if w_ds is not None else None

                result = self._unwrap1d_pairs_dataarray(
                    da, w_da, device, max_iter, epsilon, batch_size, debug
                )
                result_vars[pol] = result

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            if ds.rio.crs is not None:
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        return Batch(results)

    def _unwrap1d_pairs_dataarray(self, data, weight, device, max_iter,
                                   epsilon, batch_size, debug):
        """Internal method for IRLS unwrapping on DataArray returning pairs - LAZY."""
        import xarray as xr
        import pandas as pd
        import dask
        import dask.array

        # Extract pairs from data coords (inline _get_pairs logic for DataArray)
        refs = data.coords['ref'].values
        reps = data.coords['rep'].values
        refs = refs if isinstance(refs, np.ndarray) else [refs]
        reps = reps if isinstance(reps, np.ndarray) else [reps]
        pair_dates = [(str(pd.Timestamp(r)), str(pd.Timestamp(p))) for r, p in zip(refs, reps)]
        n_pairs = len(pair_dates)

        # Save original coordinates before chunking
        original_coords = {}
        for k, v in data.coords.items():
            if hasattr(v, 'data') and hasattr(v.data, 'compute'):
                vals = v.compute().values
            elif hasattr(v, 'values'):
                vals = v.values
            else:
                vals = v
            if hasattr(v, 'dims') and len(v.dims) > 0 and v.dims != (k,):
                original_coords[k] = (v.dims, vals)
            else:
                original_coords[k] = vals

        # Use dask auto-chunking for y,x based on memory
        # Multiplier accounts for unwrap1d internal memory (IRLS iterations)
        import dask.array as da
        auto_chunks = dask.array.core.normalize_chunks(
            'auto', (4 * n_pairs, data.y.size, data.x.size), dtype=np.complex128
        )
        chunks_y, chunks_x = auto_chunks[1][0], auto_chunks[2][0]

        # Rechunk: all pairs together (-1), auto-chunked y,x
        first_dim = data.dims[0]
        data = data.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})
        if weight is not None:
            weight = weight.chunk({first_dim: -1, 'y': chunks_y, 'x': chunks_x})

        # Use blockwise to avoid embedding large arrays in the graph
        def process_block(data_block, weight_block=None):
            """Process a spatial block - all pairs, subset of y,x."""
            unwrapped_block = utils_unwrap1d.unwrap1d_pairs_numpy(
                data_block, weight_block, pair_dates,
                device=device, max_iter=max_iter, epsilon=epsilon,
                batch_size=batch_size, debug=False
            )
            return unwrapped_block.astype(np.float32)

        data_dask = data.data
        # Resource annotation limits concurrent unwrap operations
        task_resources = {'unwrap1d': 1, 'gpu': 1} if device != 'cpu' else {'unwrap1d': 1}
        meta = np.empty((0, 0, 0), dtype=np.float32)
        if weight is not None:
            weight_dask = weight.data
            with dask.annotate(resources=task_resources):
                unwrapped_dask = da.blockwise(
                    process_block, 'pyx',
                    data_dask, 'pyx',
                    weight_dask, 'pyx',
                    dtype=np.float32,
                    meta=meta,
                )
        else:
            def process_block_no_weight(data_block):
                return process_block(data_block, None)
            with dask.annotate(resources=task_resources):
                unwrapped_dask = da.blockwise(
                    process_block_no_weight, 'pyx',
                    data_dask, 'pyx',
                    dtype=np.float32,
                    meta=meta,
                )

        # Rechunk to (1, -1, -1) for efficient per-slice downstream operations (e.g., plot)
        unwrapped_dask = unwrapped_dask.rechunk({0: 1, 1: -1, 2: -1})

        result = xr.DataArray(
            unwrapped_dask,
            dims=data.dims,
            name='unwrap'
        )
        result = result.assign_coords(original_coords)

        return result

