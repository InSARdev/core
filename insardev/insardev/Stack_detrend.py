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
from contextlib import nullcontext
from .Stack_unwrap2d import Stack_unwrap2d
from .Batch import Batch, BatchWrap
from . import utils_detrend

class Stack_detrend(Stack_unwrap2d):
    import numpy as np
    import xarray as xr

    def trend2d(self, phase, weight=None, transform=None, degree=1, device='auto', debug=False):
        """
        Compute 2D polynomial trend using PyTorch (GPU-accelerated).

        .. deprecated::
            Use ``phase.trend2d(transform, weight=corr)`` on Batch/BatchWrap instead.
            This Stack method will be removed in a future version.

        Parameters
        ----------
        phase : Batch or BatchWrap
            Phase data to detrend.
        weight : Batch or None
            Optional weights (e.g., correlation).
        transform : Batch or None
            Transform with variables to use as regressors (e.g., 'azi', 'rng', 'ele').
            All data_vars in transform will be used. If None, uses y/x coordinates.
        degree : int
            Polynomial degree for each variable (default 1):
            - degree=1: linear (a₁*v₁ + a₂*v₂ + ... + c)
            - degree=2: includes v₁², v₂², ... terms
            - degree=3: includes v₁³, v₂³, ... terms
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Trend surface (same type as input phase, lazy).

        Examples
        --------
        >>> trend = stack.trend2d(phase, weight=corr, transform=transform[['azi','rng','ele']], degree=1)
        >>> detrended = phase - trend
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np

        # Auto-detect device based on Dask cluster resources and hardware
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack_detrend._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: using device={device}")

        # Warn about MPS precision issues with high-degree polynomials
        if device == 'mps' and degree >= 3:
            print(f"NOTE: MPS has float32 precision issues for degree>={degree}. Use device='cpu' for better accuracy.")

        wrap = isinstance(phase, BatchWrap)

        # Unify transform keys to phase
        if transform is not None:
            transform = transform.sel(phase)

        result = {}
        for key in phase.keys():
            ds = phase[key]

            # Get polarization variables (spatial, with y/x dims) - excludes converted attributes
            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            # Get variables from transform (required)
            if transform is None:
                raise ValueError("transform is required for trend2d. Use stack.transform()[['azi','rng','ele']] or stack.transform()[['azi','rng']].")

            trans_ds = transform[key]
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            var_names = [v for v in trans_ds.data_vars
                        if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]

            # Check that transform and weight resolutions match phase resolution
            phase_da_ref = ds[pols[0]]
            phase_shape = phase_da_ref.shape[-2:]  # (y, x)
            phase_dy = float(phase_da_ref.y.diff('y')[0])
            phase_dx = float(phase_da_ref.x.diff('x')[0])

            trans_da_ref = trans_ds[var_names[0]]
            trans_shape = trans_da_ref.shape
            if phase_shape != trans_shape:
                trans_dy = float(trans_da_ref.y.diff('y')[0])
                trans_dx = float(trans_da_ref.x.diff('x')[0])
                raise ValueError(
                    f"Transform shape {trans_shape} does not match phase shape {phase_shape}. "
                    f"Phase spacing: dy={phase_dy:.1f}, dx={phase_dx:.1f}. "
                    f"Transform spacing: dy={trans_dy:.1f}, dx={trans_dx:.1f}. "
                    f"Use stack.transform()[['azi','rng','ele']].downsample(N) to match."
                )

            if weight is not None:
                weight_ds = weight[key]
                weight_pols = [v for v in weight_ds.data_vars
                              if 'y' in weight_ds[v].dims and 'x' in weight_ds[v].dims]
                if weight_pols:
                    weight_da_ref = weight_ds[weight_pols[0]]
                    weight_shape = weight_da_ref.shape[-2:]
                    if phase_shape != weight_shape:
                        weight_dy = float(weight_da_ref.y.diff('y')[0])
                        weight_dx = float(weight_da_ref.x.diff('x')[0])
                        raise ValueError(
                            f"Weight shape {weight_shape} does not match phase shape {phase_shape}. "
                            f"Phase spacing: dy={phase_dy:.1f}, dx={phase_dx:.1f}. "
                            f"Weight spacing: dy={weight_dy:.1f}, dx={weight_dx:.1f}. "
                            f"Use weight.downsample(N) to match."
                        )

            if debug:
                print(f"DEBUG {key}: variables={var_names}")

            result_ds = {}
            for pol in pols:
                phase_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                # Use da.blockwise for efficient dask integration
                phase_dask = phase_da.data

                # Get transform variables as dask arrays (keep lazy, already have correct 2D chunks)
                var_dask_list = [trans_ds[v].data for v in var_names]

                # Create wrapper function for blockwise - receives variable chunks
                def make_process_chunk(n_vars, device, degree):
                    def process_chunk(*args):
                        import torch
                        # First arg is phase, optional second is weight, rest are variable chunks
                        phase_chunk = args[0]
                        if len(args) == 1 + n_vars:
                            # No weight
                            weight_chunk = None
                            var_chunks = args[1:]
                        else:
                            # With weight
                            weight_chunk = args[1]
                            var_chunks = args[2:]
                        # Ensure 3D input (pair, y, x)
                        if phase_chunk.ndim == 2:
                            phase_chunk = phase_chunk[np.newaxis, ...]
                            if weight_chunk is not None:
                                weight_chunk = weight_chunk[np.newaxis, ...]
                        result = utils_detrend.trend2d_array(
                            phase_chunk, weight_chunk, list(var_chunks), torch.device(device), degree
                        )
                        return result
                    return process_chunk

                process_fn = make_process_chunk(len(var_dask_list), str(device), degree)
                dim_str = ''.join(chr(ord('a') + i) for i in range(phase_dask.ndim))
                spatial_dim_str = dim_str[-2:]  # Last 2 chars for y, x

                # Resource annotation limits concurrent detrend operations
                task_resources = {'detrend': 1, 'gpu': 1} if device != 'cpu' else {'detrend': 1}
                meta = np.empty((0,) * phase_dask.ndim, dtype=np.float32)
                with dask.annotate(resources=task_resources):
                    # Build blockwise args: phase, [weight], *variables
                    blockwise_args = [phase_dask, dim_str]
                    if weight_da is not None:
                        weight_dask = weight_da.data
                        blockwise_args.extend([weight_dask, dim_str])
                    # Add all variable arrays with spatial-only dims
                    for var_dask in var_dask_list:
                        blockwise_args.extend([var_dask, spatial_dim_str])

                    result_dask = da.blockwise(
                        process_fn, dim_str,
                        *blockwise_args,
                        dtype=np.float32,
                        meta=meta,
                    )

                trend_da = xr.DataArray(
                    result_dask,
                    dims=phase_da.dims,
                    coords=phase_da.coords
                )

                result_ds[pol] = trend_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchWrap(result) if wrap else Batch(result)

    def trend2d_dataset(self, phase, weight=None, transform=None, degree=1, device='auto', debug=False):
        """
        Compute 2D trend from phase Dataset using PyTorch (GPU-accelerated).

        Convenience wrapper around trend2d() for working with merged datasets
        instead of per-burst batches. Useful when you have already dissolved
        and merged your data.

        Parameters
        ----------
        phase : xr.Dataset
            Phase dataset (from phase.to_dataset() or unwrap2d_dataset()).
        weight : xr.Dataset, optional
            Correlation/weight dataset (from corr.to_dataset()).
        transform : xr.Dataset, optional
            Transform dataset with variables to use as regressors (e.g., 'azi', 'rng', 'ele').
        degree : int, optional
            Polynomial degree for each variable (default 1).
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        xr.Dataset
            Trend dataset with same structure as input phase.

        Examples
        --------
        Basic usage with merged datasets:
        >>> phase_ds = phase.to_dataset()
        >>> corr_ds = corr.to_dataset()
        >>> transform_ds = transform[['azi','rng']].to_dataset()
        >>> trend = stack.trend2d_dataset(phase_ds, corr_ds, transform_ds, degree=1)

        Convert back to per-burst Batch:
        >>> trend_batch = phase.from_dataset(trend)
        """
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        # Validate input types
        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")
        if transform is not None and not isinstance(transform, xr.Dataset):
            raise TypeError(f"transform must be xr.Dataset, got {type(transform).__name__}")

        # Wrap datasets in single-key Batches for trend2d()
        dummy_key = '__dataset__'
        phase_batch = BatchWrap({dummy_key: phase})
        weight_batch = BatchUnit({dummy_key: weight}) if weight is not None else None
        transform_batch = Batch({dummy_key: transform}) if transform is not None else None

        # Call PyTorch-based trend2d()
        trend_batch = self.trend2d(
            phase_batch,
            weight=weight_batch,
            transform=transform_batch,
            degree=degree,
            device=device,
            debug=debug
        )

        # Extract result Dataset
        output = trend_batch[dummy_key]
        output.attrs = phase.attrs
        return output

    def regression1d_baseline(self, data, weight=None, baseline='BPR', degree=1, wrap=False, iterations=1, device='auto', debug=False):
        """
        Fit 1D polynomial trend along perpendicular baseline at each (y, x) pixel using PyTorch.

        Parameters
        ----------
        data : Batch or BatchWrap
            Data with dimensions (pair/date, y, x) or (pair/date, pol, y, x).
        weight : Batch or None, optional
            Optional weights (e.g., correlation), same shape as data.
        baseline : str, Batch, np.ndarray, or xr.DataArray
            Baseline values to regress against:
            - str: variable name from data (default 'BPR' for perpendicular baseline)
            - Batch: values from Batch variable (e.g., data['BPR'])
            - array: explicit values (length must match pair dimension)
        degree : int
            Polynomial degree (1=linear, 2=quadratic, etc.)
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1). For wrap=True, multiple
            iterations capture more of the trend. Each iteration fits the
            residual from the previous iteration and accumulates the result.
            Typically 2-3 iterations are sufficient.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Fitted values, same type and shape as input data (lazy).

        Examples
        --------
        >>> # Fit linear trend along BPR (default)
        >>> trend = stack.regression1d_baseline(intf)
        >>> detrended = intf - trend
        >>>
        >>> # Use multiple iterations for wrapped phase
        >>> trend = stack.regression1d_baseline(intf, wrap=True, iterations=3)
        >>>
        >>> # Fit against custom baseline values
        >>> trend = stack.regression1d_baseline(intf, baseline=custom_values, degree=1)
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np
        import pandas as pd

        from .Batch import BatchWrap

        # Auto-detect device based on Dask cluster resources and hardware
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack_detrend._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: using device={device}")

        is_wrap = isinstance(data, BatchWrap)

        result = {}
        for key in data.keys():
            ds = data[key]

            # Get polarization variables (spatial, with y/x dims)
            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            if not pols:
                result[key] = ds
                continue

            # Get reference DataArray to determine dimensions
            ref_da = ds[pols[0]]

            # Detect stack dimension (pair or date)
            stack_dims = [d for d in ['pair', 'date'] if d in ref_da.dims]
            if not stack_dims:
                raise ValueError(f"Data must have 'pair' or 'date' dimension, got: {ref_da.dims}")
            stack_dim = stack_dims[0]

            # Determine baseline_values for regression
            if hasattr(baseline, 'keys') and callable(baseline.keys):
                # Batch object - extract values for this key
                baseline_ds = baseline[key]
                # Get the first data variable or the DataArray itself
                if hasattr(baseline_ds, 'data_vars') and len(baseline_ds.data_vars) > 0:
                    baseline_var = list(baseline_ds.data_vars)[0]
                    baseline_values = np.asarray(baseline_ds[baseline_var].values, dtype=np.float64)
                else:
                    baseline_values = np.asarray(baseline_ds.values, dtype=np.float64)
            elif isinstance(baseline, str):
                # Use named variable or coordinate from data
                if baseline in ds.data_vars:
                    # It's a data variable (like 'BPR')
                    baseline_values = np.asarray(ds[baseline].values, dtype=np.float64)
                elif baseline in ref_da.coords:
                    # It's a coordinate
                    baseline_coord = ref_da.coords[baseline]
                    if np.issubdtype(baseline_coord.dtype, np.datetime64):
                        dates = pd.to_datetime(baseline_coord.values)
                        baseline_values = (dates - dates.min()).total_seconds().values / 86400.0
                    else:
                        baseline_values = baseline_coord.values.astype(np.float64)
                else:
                    raise ValueError(f"'{baseline}' not found in data variables or coordinates")
            elif isinstance(baseline, (xr.DataArray, pd.DataFrame, pd.Series)):
                baseline_values = np.asarray(baseline.values, dtype=np.float64)
            else:
                baseline_values = np.asarray(baseline, dtype=np.float64)

            if debug:
                print(f"DEBUG {key}: stack_dim={stack_dim}, n_samples={len(baseline_values)}, baseline range=[{baseline_values.min():.2f}, {baseline_values.max():.2f}]")

            result_ds = {}
            for pol in pols:
                data_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                # Ensure stack dimension is first
                if data_da.dims[0] != stack_dim:
                    data_da = data_da.transpose(stack_dim, ...)
                    if weight_da is not None:
                        weight_da = weight_da.transpose(stack_dim, ...)

                # Create wrapper function for blockwise
                def process_chunk(data_chunk, weight_chunk=None, baseline_values=baseline_values, device=device, degree=degree, wrap=wrap, iterations=iterations):
                    import torch
                    # Ensure 3D input (n_samples, y, x)
                    squeeze = False
                    if data_chunk.ndim == 2:
                        data_chunk = data_chunk[np.newaxis, ...]
                        if weight_chunk is not None:
                            weight_chunk = weight_chunk[np.newaxis, ...]
                        squeeze = True
                    result = utils_detrend.regression1d_array(
                        data_chunk, baseline_values, weight_chunk, torch.device(device), degree, wrap, iterations
                    )
                    if squeeze:
                        result = result[0]
                    return result

                # Use da.blockwise for efficient dask integration
                # Rechunk to have all pairs in one chunk (regression needs all pairs together)
                data_dask = data_da.data
                if hasattr(data_dask, 'rechunk'):
                    data_dask = data_dask.rechunk({0: -1})
                dim_str = ''.join(chr(ord('a') + i) for i in range(data_dask.ndim))

                # Resource annotation limits concurrent regression operations
                task_resources = {'regression': 1, 'gpu': 1} if device != 'cpu' else {'regression': 1}
                meta = np.empty((0,) * data_dask.ndim, dtype=np.float32)
                with dask.annotate(resources=task_resources):
                    if weight_da is not None:
                        weight_dask = weight_da.data
                        if hasattr(weight_dask, 'rechunk'):
                            weight_dask = weight_dask.rechunk({0: -1})
                        def process_with_weight(data_block, weight_block):
                            return process_chunk(data_block, weight_block)
                        result_dask = da.blockwise(
                            process_with_weight, dim_str,
                            data_dask, dim_str,
                            weight_dask, dim_str,
                            dtype=np.float32,
                            meta=meta,
                        )
                    else:
                        result_dask = da.blockwise(
                            process_chunk, dim_str,
                            data_dask, dim_str,
                            dtype=np.float32,
                            meta=meta,
                        )

                # Rechunk back to per-slice for efficient downstream operations
                if hasattr(result_dask, 'rechunk'):
                    result_dask = result_dask.rechunk({0: 1})

                fit_da = xr.DataArray(
                    result_dask,
                    dims=data_da.dims,
                    coords=data_da.coords
                )

                result_ds[pol] = fit_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchWrap(result) if is_wrap else Batch(result)

    def regression1d_pairs(self, data, weight=None, degree=0, days=None, count=None, wrap=False, iterations=1, device='auto', debug=False):
        """
        Fit 1D polynomial trend along temporal pairs for each date using PyTorch.

        For each date, fits a polynomial along the time dimension using interferometric
        pairs that contain that date, then reconstructs the trend for each pair as
        the difference between the ref and rep date models.

        Parameters
        ----------
        data : Batch or BatchWrap
            Interferogram data with dimensions (pair, y, x).
        weight : Batch or None, optional
            Optional weights (e.g., correlation), same shape as data.
        degree : int
            Polynomial degree (0=mean, 1=linear). Default 0.
        days : int or None
            Maximum time interval (in days) to include. Default None (all pairs).
        count : int or None
            Maximum number of pairs per date to use. Default None (all pairs).
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1). For wrap=True, multiple
            iterations capture more of the trend. Each iteration fits the
            residual from the previous iteration and accumulates the result.
            Typically 2-3 iterations are sufficient.
        device : str
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch or BatchWrap
            Fitted trend values, same shape as input data (lazy).

        Examples
        --------
        >>> # Fit mean (degree=0) for each date
        >>> trend = stack.regression1d_pairs(intf, degree=0, days=100)
        >>> detrended = intf - trend
        >>>
        >>> # Fit linear trend (degree=1) for wrapped phase with iterations
        >>> trend = stack.regression1d_pairs(intf, degree=1, wrap=True, iterations=3)
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        import numpy as np
        import pandas as pd

        # Auto-detect device
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack_detrend._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: regression1d_pairs using device={device}")

        # Handle Batch input
        if isinstance(data, (Batch, BatchWrap)):
            is_wrap = isinstance(data, BatchWrap)
            results = {}
            for burst_id in data.keys():
                burst_ds = data[burst_id]
                burst_weight = weight[burst_id] if weight is not None else None

                # Process all polarizations
                result_ds = {}
                for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                    data_da = burst_ds[pol]
                    weight_da = burst_weight[pol] if burst_weight is not None else None

                    if debug:
                        pairs, dates = self._get_pairs(data_da, dates=True)
                        print(f"DEBUG {burst_id}: {len(dates)} dates, {len(pairs)} pairs")

                    # Get ref/rep as int64 nanoseconds (avoids datetime64 serialization issues)
                    ref_values = burst_ds['ref'].values.astype('datetime64[ns]').astype(np.int64)
                    rep_values = burst_ds['rep'].values.astype('datetime64[ns]').astype(np.int64)

                    # Ensure pair dimension is first
                    if data_da.dims[0] != 'pair':
                        data_da = data_da.transpose('pair', ...)
                        if weight_da is not None:
                            weight_da = weight_da.transpose('pair', ...)

                    # Get dask arrays and rechunk (all pairs in one chunk)
                    data_dask = data_da.data
                    if hasattr(data_dask, 'rechunk'):
                        data_dask = data_dask.rechunk({0: -1})

                    # Create wrapper that captures parameters
                    def make_wrapper(ref_vals, rep_vals, dev, deg, days_f, count_f, wrp, iters):
                        def process_chunk(data_block, weight_block=None):
                            return utils_detrend.regression1d_pairs_chunk(
                                data_block, weight_block, ref_vals, rep_vals,
                                torch.device(dev), deg, days_f, count_f, wrp, iters
                            )
                        return process_chunk

                    wrapper = make_wrapper(ref_values, rep_values, str(device), degree, days, count, wrap, iterations)

                    dim_str = ''.join(chr(ord('a') + i) for i in range(data_dask.ndim))

                    # Resource annotation limits concurrent regression operations
                    task_resources = {'regression': 1, 'gpu': 1} if device != 'cpu' else {'regression': 1}
                    meta = np.empty((0,) * data_dask.ndim, dtype=np.float32)
                    with dask.annotate(resources=task_resources):
                        if weight_da is not None:
                            weight_dask = weight_da.data
                            if hasattr(weight_dask, 'rechunk'):
                                weight_dask = weight_dask.rechunk({0: -1})

                            def make_weighted_wrapper(wrapper_fn):
                                def process_with_weight(data_block, weight_block):
                                    return wrapper_fn(data_block, weight_block)
                                return process_with_weight

                            result_dask = da.blockwise(
                                make_weighted_wrapper(wrapper), dim_str,
                                data_dask, dim_str,
                                weight_dask, dim_str,
                                dtype=np.float32,
                                meta=meta,
                            )
                        else:
                            result_dask = da.blockwise(
                                wrapper, dim_str,
                                data_dask, dim_str,
                                dtype=np.float32,
                                meta=meta,
                            )

                    # Rechunk back to per-slice for efficient downstream operations
                    if hasattr(result_dask, 'rechunk'):
                        result_dask = result_dask.rechunk({0: 1})

                    # Create output DataArray
                    trend_da = xr.DataArray(
                        result_dask,
                        dims=data_da.dims,
                        coords=data_da.coords
                    )
                    result_ds[pol] = trend_da

                results[burst_id] = xr.Dataset(result_ds, attrs=burst_ds.attrs)

            return BatchWrap(results) if is_wrap else Batch(results)

        else:
            raise ValueError("data must be a Batch or BatchWrap object")

    def _polyfit(self, data, weight=None, degree=0, days=None, count=None, wrap=False):
        print ('NOTE: Function is deprecated. Use Stack.regression_pairs() instead.')
        return self.regression_pairs(data=data, weight=weight, degree=degree, days=days, count=count, wrap=wrap)

    def _regression_pairs(self, data, weight=None, degree=0, days=None, count=None, wrap=False):
        import xarray as xr
        import pandas as pd
        import numpy as np
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')

        multi_index = None
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            multi_index = data['stack']
            data = data.reset_index('stack')
            if weight is not None:
                if not ('stack' in weight.dims and isinstance(weight.coords['stack'].to_index(), pd.MultiIndex)):
                    raise ValueError('ERROR: "weight", if provided, must be stacked consistently with "data".')
                data = data.reset_index('stack')
        else:
            if 'stack' in weight.dims and isinstance(weight.coords['stack'].to_index(), pd.MultiIndex):
                raise ValueError('ERROR: "weight", if provided, must be stacked consistently with "data".')

        pairs, dates = self._get_pairs(data, dates=True)

        models = []
        if wrap:
            models_sin = []
            models_cos = []

        for date in dates:
            data_pairs = pairs[(pairs.ref==date)|(pairs.rep==date)].pair.values
            if weight is None:
                stack = data.sel(pair=data_pairs)
            else:
                stack = data.sel(pair=data_pairs) * np.sqrt(weight.sel(pair=data_pairs))
            del data_pairs

            stack_days = xr.where(stack.ref < pd.Timestamp(date),
                           (stack.ref - stack.rep).dt.days,
                           (stack.rep - stack.ref).dt.days)
            # select smallest intervals
            stack_days_selected = stack_days[np.argsort(np.abs(stack_days.values))][:count]
            if days is not None:
                stack_days_selected = stack_days_selected[np.abs(stack_days_selected)<=days]

            selected_pairs = (np.sign(stack_days)*stack).assign_coords(time=stack_days)\
                [stack.pair.isin(stack_days_selected.pair)]\
                .swap_dims({'pair': 'time'})\
                .sortby(['ref', 'rep'])
            del stack, stack_days, stack_days_selected

            if not wrap:
                linear_fit = selected_pairs.polyfit(dim='time', deg=degree)
                model = linear_fit.polyfit_coefficients.sel(degree=degree).astype(np.float32)
                models.append(model.assign_coords(date=pd.to_datetime(date)))
                del model, linear_fit
            else:
                # fit sine and cosine components
                linear_fit_sin = np.sin(selected_pairs).polyfit(dim='time', deg=degree)
                linear_fit_cos = np.cos(selected_pairs).polyfit(dim='time', deg=degree)

                model_sin = linear_fit_sin.polyfit_coefficients.sel(degree=degree).astype(np.float32)
                model_cos = linear_fit_cos.polyfit_coefficients.sel(degree=degree).astype(np.float32)

                models_sin.append(model_sin.assign_coords(date=pd.to_datetime(date)))
                models_cos.append(model_cos.assign_coords(date=pd.to_datetime(date)))
                del model_sin, model_cos, linear_fit_sin, linear_fit_cos

            del selected_pairs

        if not wrap:
            model = xr.concat(models, dim='date')
            del models
            out = xr.concat(
                [
                    (model.sel(date=ref).drop('date') - model.sel(date=rep).drop('date'))
                    .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep)
                    for ref, rep in zip(pairs['ref'], pairs['rep'])
                ],
                dim='pair'
            ).rename(data.name)
        else:
            # combine separate sin and cos models
            model_sin = xr.concat(models_sin, dim='date')
            model_cos = xr.concat(models_cos, dim='date')
            del models_sin, models_cos

            angle_diffs = []
            for ref, rep in zip(pairs['ref'], pairs['rep']):
                sin_ref = model_sin.sel(date=ref).drop('date')
                cos_ref = model_cos.sel(date=ref).drop('date')
                sin_rep = model_sin.sel(date=rep).drop('date')
                cos_rep = model_cos.sel(date=rep).drop('date')

                # compute angle differences using sin/cos difference formula
                # sin(A−B) = sin A * cos B − cos A * sin B
                sin_diff = sin_ref * cos_rep - cos_ref * sin_rep
                # cos(A−B) = cos A * cos B+ sin A * sin B
                cos_diff = cos_ref * cos_rep + sin_ref * sin_rep
                del sin_ref, cos_ref, sin_rep, cos_rep

                angle_diff = np.arctan2(sin_diff, cos_diff)\
                             .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep)
                angle_diffs.append(angle_diff)
                del angle_diff, sin_diff, cos_diff

            out = xr.concat(angle_diffs, dim='pair').rename(data.name)
            del angle_diffs

        if multi_index is not None:
            return out.assign_coords(stack=multi_index)
        return out

    def _turbulence(self, phase, weight=None):
        import xarray as xr
        import pandas as pd

        print ('NOTE: this function is deprecated, use instead Stack.polyfit()')

        pairs, dates = self._get_pairs(phase, dates=True)

        turbos = []
        for date in dates:
            ref = pairs[pairs.ref==date]
            rep = pairs[pairs.rep==date]
            #print (date, len(ref), len(rep))
            ref_data = phase.sel(pair=ref.pair.values)
            #print (ref_data)
            rep_data = phase.sel(pair=rep.pair.values)
            #print (rep_data)
            if weight is not None:
                ref_weight = weight.sel(pair=ref.pair.values)
                rep_weight = weight.sel(pair=rep.pair.values)
                turbo = xr.concat([ref_data*ref_weight, -rep_data*rep_weight], dim='pair').sum('pair')/\
                    xr.concat([ref_weight, rep_weight], dim='pair').sum('pair')
                del ref_weight, rep_weight
            else:
                turbo = xr.concat([ref_data, -rep_data], dim='pair').mean('pair')
            del ref_data, rep_data
            turbos.append(turbo.assign_coords(date=pd.to_datetime(date)))
            del turbo
        turbo = xr.concat(turbos, dim='date')
        del turbos

        phase_turbo = xr.concat([(turbo.sel(date=ref).drop('date') - turbo.sel(date=rep).drop('date'))\
                                 .assign_coords(pair=str(ref.date()) + ' ' + str(rep.date()), ref=ref, rep=rep) \
                          for ref, rep in zip(pairs['ref'], pairs['rep'])], dim='pair')

        return phase_turbo.rename('turbulence')

    def _velocity(self, data):
        import pandas as pd
        import numpy as np
        #years = ((data.date.max() - data.date.min()).dt.days/365.25).item()
        #nanoseconds = (data.date.max().astype(int) - data.date.min().astype(int)).item()
        #print ('years', np.round(years, 3), 'nanoseconds', nanoseconds)
        multi_index = None
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            multi_index = data.coords['stack']
            # replace multiindex by sequential numbers 0,1,...
            data = data.reset_index('stack')
        #velocity = nanoseconds*data.polyfit('date', 1).polyfit_coefficients.sel(degree=1)/years
        nanoseconds_per_year = 365.25*24*60*60*1e9
        # calculate slope per year
        velocity = nanoseconds_per_year*data.polyfit('date', 1).polyfit_coefficients.sel(degree=1).astype(np.float32).rename('trend')
        if multi_index is not None:
            return velocity.assign_coords(stack=multi_index)
        return velocity

    def plot_velocity(self, data, caption='Velocity, [rad/year]',
                      quantile=None, vmin=None, vmax=None, symmetrical=False, aspect=None, alpha=1, **kwargs):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            data = data.unstack('stack')

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"
    
        if quantile is not None:
            vmin, vmax = np.nanquantile(data, quantile)
    
        # define symmetrical boundaries
        if symmetrical is True and vmax > 0:
            minmax = max(abs(vmin), vmax)
            vmin = -minmax
            vmax =  minmax
    
        plt.figure()
        data.plot.imshow(vmin=vmin, vmax=vmax, alpha=alpha, cmap='turbo', interpolation='none')
        #self.plot_AOI(**kwargs)
        #self.plot_POI(**kwargs)
        if aspect is not None:
            plt.gca().set_aspect(aspect)
        #plt.xlabel('Range')
        #plt.ylabel('Azimuth')
        plt.title(caption)

    def plot_velocity_los_mm(self, data, caption='Velocity, [mm/year]',
                      quantile=None, vmin=None, vmax=None, symmetrical=False, aspect=None, alpha=1, **kwargs):
        self.plot_velocity(self.los_displacement_mm(data),
                           caption=caption, aspect=aspect, alpha=alpha,
                           quantile=quantile, vmin=vmin, vmax=vmax, symmetrical=symmetrical, **kwargs)

