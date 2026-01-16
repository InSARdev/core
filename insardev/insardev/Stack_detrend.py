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
from .utils_regression2d import regression2d
from . import utils_xarray
from .Batch import Batch, BatchWrap
class Stack_detrend(Stack_unwrap2d):
    import numpy as np
    import xarray as xr

    # def trend2d_interferogram(self, datas, weights=None, variables=['azi', 'rng'], compute=False, **kwarg):
    #     return self.trend2d(datas, weights, variables, compute, wrap=True, **kwarg)

    def trend2d_sklearn(self, datas, weights, transform, debug=False, **kwarg):
        """
        Compute 2D polynomial trend using sklearn (legacy implementation).

        For most use cases, use trend2d() instead which uses PyTorch and is faster.
        """
        if debug:
            print(f"DEBUG trend2d_sklearn: degree={kwarg.get('degree', 1)}")

        def _regression2d(data, weight, transform, **kwarg):
            key = kwarg.pop('key')
            trend = regression2d(data,
                                 variables=[transform[v] for v in transform.data_vars],
                                 weight=weight,
                                 **kwarg)
            return trend

        if transform is not None:
            # unify keys to datas
            transform = transform.sel(datas)

        # prevent chunking of the stack dimension, it produces performance issues and incorrect results in 2D functions
        assert datas.chunks['pair']==1, 'ERROR: datas must be chunked as (1, ...)'
        assert weights is None or weights.chunks['pair']==1, 'ERROR: weights must be chunked as (1, ...)'

        wrap = True if isinstance(datas, BatchWrap) else False
        data = utils_xarray.apply_pol(datas, weights, transform, func=_regression2d, add_key=True, wrap=wrap, **kwarg)
        return BatchWrap(data) if wrap else Batch(data)

    @staticmethod
    def _trend2d_array(phase, weight, variables, device, degree=1):
        """
        Fit 2D polynomial trend using PyTorch least squares (pure GPU implementation).

        Parameters
        ----------
        phase : np.ndarray
            2D or 3D phase array (pair, y, x) or (y, x)
        weight : np.ndarray or None
            Weight array, same shape as phase
        variables : list of np.ndarray
            List of 2D variable arrays (y, x) to use as regressors
        device : torch.device
            PyTorch device
        degree : int
            Polynomial degree for each variable (1=linear, 2=quadratic, etc.)

        Returns
        -------
        np.ndarray
            Trend surface, same shape as phase
        """
        import torch
        import numpy as np
        from itertools import combinations_with_replacement

        # Handle 2D vs 3D input
        squeeze = phase.ndim == 2
        if squeeze:
            phase = phase[np.newaxis, ...]
            if weight is not None:
                weight = weight[np.newaxis, ...]

        n_pairs, ny, nx = phase.shape
        n_pixels = ny * nx

        # Use float64 on CPU, float32 on GPU (MPS doesn't support float64)
        if device.type == 'cpu':
            dtype = torch.float64
        else:
            dtype = torch.float32

        # Flatten variables and track valid coordinates (on CPU first)
        valid_coords = np.ones(n_pixels, dtype=bool)
        vars_np = []
        for var in variables:
            v_flat = var.ravel()
            valid_coords &= np.isfinite(v_flat)
            vars_np.append(np.nan_to_num(v_flat, nan=0.0))

        # Move variables to GPU and build polynomial features there
        vars_t = [torch.tensor(v, dtype=dtype, device=device) for v in vars_np]
        n_vars = len(vars_t)

        # Build polynomial features on GPU (matching sklearn PolynomialFeatures order)
        # Order: degree 1 first, then degree 2, etc. (same as sklearn include_bias=False)
        features = []
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n_vars), d):
                term = torch.ones(n_pixels, dtype=dtype, device=device)
                for idx in combo:
                    term = term * vars_t[idx]
                features.append(term)

        # Stack features: (n_pixels, n_features)
        X_poly = torch.stack(features, dim=1)

        # Add bias column
        ones = torch.ones(n_pixels, 1, dtype=dtype, device=device)
        A_all = torch.cat([X_poly, ones], dim=1)

        # Valid coordinates mask on GPU
        valid_coords_t = torch.tensor(valid_coords, dtype=torch.bool, device=device)

        trends = np.empty_like(phase)

        for i in range(n_pairs):
            phase_flat = phase[i].ravel()

            # Get valid (non-NaN) mask
            valid_np = np.isfinite(phase_flat) & valid_coords
            if weight is not None:
                weight_flat = weight[i].ravel()
                valid_np &= np.isfinite(weight_flat)

            n_valid = valid_np.sum()
            if n_valid < 10:
                trends[i] = np.nan
                continue

            valid_t = torch.tensor(valid_np, dtype=torch.bool, device=device)

            # Extract valid rows
            A_valid = A_all[valid_t]  # (n_valid, n_features+1)
            b_valid = torch.tensor(phase_flat[valid_np], dtype=dtype, device=device)

            # Standardize features on GPU (excluding bias column)
            # Compute mean and std on valid data only
            feature_mean = A_valid[:, :-1].mean(dim=0, keepdim=True)
            feature_std = A_valid[:, :-1].std(dim=0, keepdim=True) + 1e-10
            A_valid_scaled = torch.cat([
                (A_valid[:, :-1] - feature_mean) / feature_std,
                A_valid[:, -1:]  # Keep bias column as-is
            ], dim=1)

            # Apply weights if provided
            if weight is not None:
                w = torch.tensor(np.sqrt(weight_flat[valid_np]), dtype=dtype, device=device)
                A_valid_scaled = A_valid_scaled * w[:, None]
                b_valid = b_valid * w

            # Solve normal equations: (A^T A) coeffs = A^T b
            AtA = A_valid_scaled.T @ A_valid_scaled
            Atb = A_valid_scaled.T @ b_valid
            coeffs = torch.linalg.solve(AtA, Atb)

            # Compute trend for all pixels using the same scaling
            A_all_scaled = torch.cat([
                (A_all[:, :-1] - feature_mean) / feature_std,
                A_all[:, -1:]
            ], dim=1)
            trend = (A_all_scaled @ coeffs).cpu().numpy()
            trends[i] = trend.reshape(ny, nx)

        if squeeze:
            trends = trends[0]

        return trends.astype(np.float32)

    def trend2d(self, phase, weight=None, transform=None, degree=1, device=None, debug=False):
        """
        Compute 2D polynomial trend using PyTorch (GPU-accelerated).

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
        device : str or None
            PyTorch device ('cuda', 'mps', 'cpu'). Auto-detected if None.
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
        device = Stack_detrend._get_torch_device(device if device is not None else 'auto', debug=debug)

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

            # Get variables from transform or use y/x coordinates
            if transform is not None:
                trans_ds = transform[key]
                # Filter for spatial variables (with y, x dims) - excludes converted attributes
                var_names = [v for v in trans_ds.data_vars
                            if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]
                # Get transform variables (compute if lazy)
                var_arrays = [trans_ds[v] for v in var_names]
                if any(hasattr(arr.data, 'compute') for arr in var_arrays):
                    computed = dask.compute(*var_arrays)
                    variables = [arr.values if hasattr(arr, 'values') else arr for arr in computed]
                else:
                    variables = [arr.values for arr in var_arrays]
            else:
                # Use y/x coordinates as default variables
                y = ds[pols[0]].y.values
                x = ds[pols[0]].x.values
                Y, X = np.meshgrid(y, x, indexing='ij')
                variables = [Y.astype(np.float32), X.astype(np.float32)]
                var_names = ['y', 'x']

            if debug:
                print(f"DEBUG {key}: variables={var_names}, shapes={[v.shape for v in variables]}")

            result_ds = {}
            for pol in pols:
                phase_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                # Save coordinates for output
                coords = dict(phase_da.coords)
                dims = phase_da.dims

                # Create wrapper function for apply_ufunc
                def process_chunk(phase_chunk, weight_chunk=None, variables=variables, device=device, degree=degree):
                    import torch
                    # Ensure 3D input (pair, y, x)
                    if phase_chunk.ndim == 2:
                        phase_chunk = phase_chunk[np.newaxis, ...]
                        if weight_chunk is not None:
                            weight_chunk = weight_chunk[np.newaxis, ...]
                    # Use MPS lock to serialize GPU access on Apple Silicon
                    with Stack_detrend._mps_lock() if device.type == 'mps' else nullcontext():
                        result = Stack_detrend._trend2d_array(
                            phase_chunk, weight_chunk, variables, torch.device(device), degree
                        )
                    return result

                # Use apply_ufunc with dask='parallelized' for efficient processing
                # This avoids serializing variables for each chunk
                with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                    if weight_da is not None:
                        trend_da = xr.apply_ufunc(
                            process_chunk,
                            phase_da,
                            weight_da,
                            input_core_dims=[['y', 'x'], ['y', 'x']],
                            output_core_dims=[['y', 'x']],
                            dask='parallelized',
                            output_dtypes=[np.float32],
                            dask_gufunc_kwargs={'allow_rechunk': True},
                        )
                    else:
                        trend_da = xr.apply_ufunc(
                            process_chunk,
                            phase_da,
                            input_core_dims=[['y', 'x']],
                            output_core_dims=[['y', 'x']],
                            dask='parallelized',
                            output_dtypes=[np.float32],
                            dask_gufunc_kwargs={'allow_rechunk': True},
                        )

                result_ds[pol] = trend_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchWrap(result) if wrap else Batch(result)

    def trend2d_dataset(self, phase, weight=None, transform=None, **kwargs):
        """
        Compute 2D trend from phase Dataset using regression.

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
            Transform dataset with 'azi' and 'rng' variables.
        **kwargs
            Additional arguments passed to regression2d (degree, algorithm, etc.).

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
        >>> trend = stack.trend2d_dataset(phase_ds, corr_ds, transform_ds, degree=1, algorithm='linear')

        Convert back to per-burst Batch:
        >>> trend_batch = phase.from_dataset(trend)
        """
        import xarray as xr
        import numpy as np

        # Validate input types
        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")
        if transform is not None and not isinstance(transform, xr.Dataset):
            raise TypeError(f"transform must be xr.Dataset, got {type(transform).__name__}")

        # Helper to chunk only dimensions that exist in a dataset
        def safe_chunk(ds, base_spec):
            valid_spec = {k: v for k, v in base_spec.items() if k in ds.dims}
            return ds.chunk(valid_spec)

        # Rechunk: 1 pair per chunk, full y/x
        chunk_spec = {'y': -1, 'x': -1}
        if 'pair' in phase.dims:
            chunk_spec['pair'] = 1
        phase = safe_chunk(phase, chunk_spec)
        if weight is not None:
            weight = safe_chunk(weight, chunk_spec)

        # Get polarization variables (spatial, with y/x dims) - excludes converted attributes
        polarizations = [v for v in phase.data_vars
                        if 'y' in phase[v].dims and 'x' in phase[v].dims]

        # Detect stack dimension
        sample_da = phase[polarizations[0]]
        dims = sample_da.dims
        stackvar = dims[0] if len(dims) > 2 else None
        n_stack = sample_da.sizes[stackvar] if stackvar else 1

        # Prepare transform variables list
        transform_vars = [transform[v] for v in transform.data_vars] if transform is not None else []

        results = {}
        for pol in polarizations:
            phase_da = phase[pol]
            weight_da = weight[pol] if weight is not None else None

            if stackvar and n_stack > 1:
                # Process each stack element using regression2d directly
                trend_slices = []
                for i in range(n_stack):
                    phase_slice = phase_da.isel({stackvar: i})
                    weight_slice = weight_da.isel({stackvar: i}) if weight_da is not None else None

                    # Call regression2d directly - it handles dask arrays properly via apply_ufunc
                    trend_slice = regression2d(
                        phase_slice,
                        variables=transform_vars,
                        weight=weight_slice,
                        **kwargs
                    )
                    trend_slices.append(trend_slice.expand_dims({stackvar: [phase_da[stackvar].values[i]]}))

                # Concatenate along stack dimension and rechunk
                result_da = xr.concat(trend_slices, dim=stackvar).chunk({stackvar: 1, 'y': -1, 'x': -1})
            else:
                # Single 2D case - call regression2d directly
                result_da = regression2d(
                    phase_da,
                    variables=transform_vars,
                    weight=weight_da,
                    **kwargs
                )

            results[pol] = result_da.rename(pol)

        output = xr.merge(list(results.values()))
        output.attrs = phase.attrs
        return output

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

    def _trend(self, data, dim='auto', degree=1):
        print ('NOTE: Function is deprecated. Use Stack.regression1d() instead.')
        return self.regression1d(data=data, dim=dim, degree=degree)


    def _gaussian(self, data, wavelength, truncate=3.0, resolution=60, debug=False):
        """
        Apply a lazy Gaussian filter to an input 2D or 3D data array.

        Parameters
        ----------
        data : xarray.DataArray
            The input data array with NaN values allowed.
        wavelength : float
            The cut-off wavelength for the Gaussian filter in meters.
        truncate : float, optional
            Size of the Gaussian kernel, defined in terms of standard deviation, or 'sigma'. 
            It is the number of sigmas at which the window (filter) is truncated. 
            For example, if truncate = 3.0, the window will cut off at 3 sigma. Default is 3.0.
        resolution : float, optional
            The processing resolution for the Gaussian filter in meters.
        debug : bool, optional
            Whether to print debug information.

        Returns
        -------
        xarray.DataArray
            The filtered data array with the same coordinates as the input.

        Examples
        --------
        Detrend ionospheric effects and solid Earth's tides on a large area and save to disk:
        stack.stack_gaussian2d(slcs, wavelength=400)
        For band-pass filtering apply the function twice and save to disk:
        model = stack.stack_gaussian2d(slcs, wavelength=400, interactive=True) \
            - stack.stack_gaussian2d(slcs, wavelength=2000, interactive=True)
        stack.save_cube(model, caption='Gaussian Band-Pass filtering')

        Detrend and return lazy xarray dataarray:
        stack.stack_gaussian2d(slcs, wavelength=400, interactive=True)
        For band-pass filtering apply the function twice:
        stack.stack_gaussian2d(slcs, wavelength=400, interactive=True) \
            - stack.stack_gaussian2d(slcs, wavelength=2000, interactive=True) 

        """
        import xarray as xr
        import numpy as np
    #         import warnings
    #         # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
    #         warnings.filterwarnings('ignore')
    #         warnings.filterwarnings('ignore', module='dask')
    #         warnings.filterwarnings('ignore', module='dask.core')

        assert np.issubdtype(data.dtype, np.floating), 'ERROR: expected float datatype input data'
        assert wavelength is not None, 'ERROR: Gaussian filter cut-off wavelength is not defined'

        # ground pixel size
        dy, dx = self.get_spacing(data)
        # downscaling
        yscale, xscale = int(np.round(resolution/dy)), int(np.round(resolution/dx))
        # gaussian kernel
        #sigma_y = np.round(wavelength / dy / yscale, 1)
        #sigma_x = np.round(wavelength / dx / xscale, 1)
        if debug:
            print (f'DEBUG: gaussian: ground pixel size in meters: y={dy:.1f}, x={dx:.1f}')
        if (xscale <=1 and yscale <=1) or (wavelength/resolution <= 3):
            # decimation is useless
            return self.multilooking(data, wavelength=wavelength, coarsen=None, debug=debug)

        # define filter on decimated grid, the correction value is typically small
        wavelength_dec = np.sqrt(wavelength**2 - resolution**2)
        if debug:
            print (f'DEBUG: gaussian: downscaling to resolution {resolution}m using yscale {yscale}, xscale {xscale}')
            #print (f'DEBUG: gaussian: filtering on {resolution}m grid using sigma_y0 {sigma_y}, sigma_x0 {sigma_x}')
            print (f'DEBUG: gaussian: filtering on {resolution}m grid using wavelength {wavelength_dec:.1f}')

        # find stack dim
        stackvar = data.dims[0] if len(data.dims) == 3 else 'stack'
        #print ('stackvar', stackvar)

        data_dec = self.multilooking(data, wavelength=resolution, coarsen=(yscale,xscale), debug=debug)
        data_dec_gauss = self.multilooking(data_dec, wavelength=wavelength_dec, debug=debug)
        del data_dec

        stack = []
        for stackval in data[stackvar].values if len(data.dims) == 3 else [None]:
            data_in = data_dec_gauss.sel({stackvar: stackval}) if stackval is not None else data_dec_gauss
            data_out = data_in.reindex({'y': data.y, 'x': data.x}, method='nearest')
            del data_in
            stack.append(data_out)
            del data_out

        # wrap lazy Dask array to Xarray dataarray
        if len(data.dims) == 2:
            out = stack[0]
        else:
            out = xr.concat(stack, dim=stackvar)
        del stack

        # append source data coordinates excluding removed y, x ones
        for (k,v) in data.coords.items():
            if k not in ['y','x']:
                out[k] = v

        return out
