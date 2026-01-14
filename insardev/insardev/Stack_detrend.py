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
from .Stack_unwrap2d import Stack_unwrap2d
from .utils_regression2d import regression2d
from . import utils_xarray
from .Batch import Batch, BatchWrap
class Stack_detrend(Stack_unwrap2d):
    import numpy as np
    import xarray as xr

    # def trend2d_interferogram(self, datas, weights=None, variables=['azi', 'rng'], compute=False, **kwarg):
    #     return self.trend2d(datas, weights, variables, compute, wrap=True, **kwarg)

    def trend2d(self, datas, weights, transform, compute=False, **kwarg):
        def _regression2d(data, weight, transform, **kwarg):
            #print ('kwarg', kwarg)
            key = kwarg.pop('key')
            #print ('shapes', data.shape, weight.shape, [transform[v].shape for v in variables])
            #if transform is None:
            #    # find nearest transform matrix values on the original grid for potentially multilooked data
            #    transform = self.dss[key][variables].reindex_like(data, method='nearest')
            trend = regression2d(data,
                                 variables=[transform[v] for v in transform.data_vars],
                                 weight=weight,
                                 **kwarg)
            #print ('trend', trend)
            return trend
        if transform is not None:
            # unify keys to datas
            transform = transform.sel(datas)

        # prevent chunking of the stack dimension, it produces performance issues and incorrect results in 2D functions
        assert datas.chunks['pair']==1, 'ERROR: datas must be chunked as (1, ...)'
        assert weights is None or weights.chunks['pair']==1, 'ERROR: weights must be chunked as (1, ...)'

        wrap = True if isinstance(datas, BatchWrap) else False
        data = utils_xarray.apply_pol(datas, weights, transform, func=_regression2d, add_key=True, compute=compute, wrap=wrap, **kwarg)
        # Preserve pair coordinate type from input (xr.merge in apply_pol can lose MultiIndex)
        for key in data:
            if 'pair' in datas[key].coords and 'pair' in data[key].coords:
                data[key] = data[key].assign_coords(pair=datas[key].coords['pair'])
        return BatchWrap(data) if wrap else Batch(data)

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

        # Get polarization variables
        polarizations = [v for v in phase.data_vars if v != 'spatial_ref']

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
        data.plot.imshow(vmin=vmin, vmax=vmax, alpha=alpha, cmap='turbo')
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
