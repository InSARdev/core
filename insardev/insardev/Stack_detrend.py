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

class Stack_detrend(Stack_unwrap2d):
    import numpy as np
    import xarray as xr

    # trend2d, trend1d, trend1d_pairs — implementations moved to BatchCore.
    # Use phase.trend2d(...), data.trend1d(...), data.trend1d_pairs(...) instead.
    # trend2d_dataset — moved to BatchCore.trend2d_dataset() classmethod.

    def regression1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("regression1d_pairs() is removed. Use Batch.trend1d_pairs() instead.")

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

    def _plot_velocity(self, data, caption='Velocity, [rad/year]',
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

    def _plot_velocity_los_mm(self, data, caption='Velocity, [mm/year]',
                      quantile=None, vmin=None, vmax=None, symmetrical=False, aspect=None, alpha=1, **kwargs):
        self.plot_velocity(self.los_displacement_mm(data),
                           caption=caption, aspect=aspect, alpha=alpha,
                           quantile=quantile, vmin=vmin, vmax=vmax, symmetrical=symmetrical, **kwargs)

