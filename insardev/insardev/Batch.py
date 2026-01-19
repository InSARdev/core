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
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect

class Batch(BatchCore):
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the real 2D vars from Stack
        if isinstance(mapping, Stack):
            #print ('Batch __init__: Stack')
            real_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only non-complex data_vars that live on the ('y','x') grid
                # and include 1D non-complex variables (e.g., per-axis metadata)
                real_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind != 'c'
                    and (
                        tuple(ds[v].dims) == ('y', 'x')
                        or len(ds[v].dims) == 1
                    )
                ]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('Batch __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})
    
    def clip(self, min=None, max=None, **kwargs):
        """
        used for correlation in [0,1] range
        """
        return BatchUnit(super().clip(min=min, max=max, **kwargs))
    
    def plot(
        self,
        cmap = 'turbo',
        alpha = 0.5,
        caption = None,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

    @staticmethod
    def _velocity_torch(data, times_years, min_valid=3, device='auto', debug=False):
        """
        Compute velocity using PyTorch closed-form linear regression.

        Uses the closed-form solution for weighted linear regression which is
        much more GPU-efficient than batched lstsq:
            slope = (Σw·Σ(w·t·y) - Σ(w·t)·Σ(w·y)) / (Σw·Σ(w·t²) - (Σ(w·t))²)

        Parameters
        ----------
        data : np.ndarray
            3D array (n_times, height, width) or 2D (n_times, n_pixels).
        times_years : np.ndarray
            1D array of time values in years from first date.
        min_valid : int
            Minimum valid points required.
        device : str
            PyTorch device.
        debug : bool
            Print debug info.

        Returns
        -------
        np.ndarray
            2D array (height, width) or 1D (n_pixels) of velocities.
        """
        import torch
        import numpy as np

        # Select device using shared helper (inherited from BatchCore)
        dev = Batch._get_torch_device(device)

        if debug:
            print(f'DEBUG: _velocity_torch device={dev}, shape={data.shape}')

        original_shape = data.shape
        n_times = data.shape[0]

        # Reshape to 2D: (n_times, n_pixels)
        if data.ndim == 3:
            data_2d = data.reshape(n_times, -1)
        else:
            data_2d = data

        # Move to device
        y = torch.from_numpy(data_2d.astype(np.float32)).to(dev)  # (n_times, n_pixels)
        t = torch.from_numpy(times_years.astype(np.float32)).to(dev)  # (n_times,)

        # Handle NaN: create weight mask (1 for valid, 0 for NaN)
        nan_mask = torch.isnan(y)
        valid_count = (~nan_mask).sum(dim=0)  # (n_pixels,)
        w = (~nan_mask).float()  # (n_times, n_pixels)

        # Replace NaN with 0 for computation
        y_filled = torch.where(nan_mask, torch.zeros_like(y), y)

        # Closed-form weighted linear regression
        # slope = (Σw·Σ(w·t·y) - Σ(w·t)·Σ(w·y)) / (Σw·Σ(w·t²) - (Σ(w·t))²)
        # All sums are over the time dimension (dim=0)

        # Broadcast t to match y shape: (n_times,) -> (n_times, n_pixels)
        t_expanded = t.unsqueeze(1)  # (n_times, 1)

        # Compute weighted sums (all results are (n_pixels,))
        sum_w = w.sum(dim=0)                           # Σw
        sum_wt = (w * t_expanded).sum(dim=0)           # Σ(w·t)
        sum_wy = (w * y_filled).sum(dim=0)             # Σ(w·y)
        sum_wt2 = (w * t_expanded * t_expanded).sum(dim=0)  # Σ(w·t²)
        sum_wty = (w * t_expanded * y_filled).sum(dim=0)    # Σ(w·t·y)

        # Compute denominator: Σw·Σ(w·t²) - (Σ(w·t))²
        denom = sum_w * sum_wt2 - sum_wt * sum_wt

        # Compute numerator: Σw·Σ(w·t·y) - Σ(w·t)·Σ(w·y)
        numer = sum_w * sum_wty - sum_wt * sum_wy

        # Compute slope (velocity)
        velocity = numer / denom

        # Compute intercept: intercept = (Σ(w·y) - slope * Σ(w·t)) / Σw
        intercept = (sum_wy - velocity * sum_wt) / sum_w

        # Mask pixels with insufficient valid points or zero denominator
        valid_mask = (valid_count >= min_valid) & (denom.abs() > 1e-10)
        velocity = torch.where(valid_mask, velocity, torch.tensor(float('nan'), device=dev))
        intercept = torch.where(valid_mask, intercept, torch.tensor(float('nan'), device=dev))

        # Reshape back
        vel_np = velocity.cpu().numpy()
        int_np = intercept.cpu().numpy()
        if len(original_shape) == 3:
            vel_np = vel_np.reshape(original_shape[1], original_shape[2])
            int_np = int_np.reshape(original_shape[1], original_shape[2])

        # Cleanup GPU memory
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()

        return vel_np, int_np

    def velocity(self, min_valid=3, device='auto', debug=False) -> tuple["Batch", "Batch"]:
        """
        Compute velocity (linear trend) and intercept from time series.

        Calculates the slope per year and intercept for each pixel using linear
        regression on the 'date' dimension. Uses PyTorch for GPU acceleration.

        Parameters
        ----------
        min_valid : int, optional
            Minimum number of valid (non-NaN) data points required to compute
            velocity. Pixels with fewer valid points will be set to NaN.
            Default is 3.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print debug information. Default False.

        Returns
        -------
        tuple[Batch, Batch]
            (velocity, intercept) - velocity is slope per year, intercept is
            the y-value at t=0 (first date). Both are lazy Batch objects.

        Examples
        --------
        >>> displacement = stack.lstsq(detrend, corr)
        >>> velocity, intercept = displacement.velocity()
        """
        import dask
        import numpy as np
        import pandas as pd
        import xarray as xr
        import rioxarray  # for .rio accessor

        nanoseconds_per_year = 365.25 * 24 * 60 * 60 * 1e9

        # Auto-detect device based on Dask cluster resources and hardware
        device = Batch._get_torch_device(device, debug=debug)

        if debug:
            print(f"DEBUG: velocity using device={device}")

        # Get CRS from input batch
        crs = self.crs

        vel_results = {}
        int_results = {}
        for key, ds in self.items():
            vel_vars = {}
            int_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for var in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da = ds[var]

                # Convert dates to years from first date
                dates = pd.to_datetime(da.date.values)
                times_ns = (dates - dates[0]).total_seconds() * 1e9
                times_years = np.array(times_ns / nanoseconds_per_year, dtype=np.float32)

                # Create wrapper for apply_ufunc
                def compute_velocity(data, times_years=times_years, min_valid=min_valid, device=device):
                    return Batch._velocity_torch(data, times_years, min_valid=min_valid, device=device)

                # Use apply_ufunc with dask='parallelized' for lazy evaluation
                with dask.annotate(resources={'gpu': 1} if device.type != 'cpu' else {}):
                    vel_da, int_da = xr.apply_ufunc(
                        compute_velocity,
                        da,
                        input_core_dims=[['date', 'y', 'x']],
                        output_core_dims=[['y', 'x'], ['y', 'x']],
                        dask='parallelized',
                        output_dtypes=[np.float32, np.float32],
                        dask_gufunc_kwargs={'allow_rechunk': True},
                    )

                # Assign coordinates
                vel_da = vel_da.assign_coords({'y': da.y, 'x': da.x})
                int_da = int_da.assign_coords({'y': da.y, 'x': da.x})
                vel_vars[var] = vel_da
                int_vars[var] = int_da

            vel_ds = xr.Dataset(vel_vars)
            vel_ds.attrs = ds.attrs
            int_ds = xr.Dataset(int_vars)
            int_ds.attrs = ds.attrs
            # Preserve CRS
            if crs is not None:
                vel_ds = vel_ds.rio.write_crs(crs)
                int_ds = int_ds.rio.write_crs(crs)
            vel_results[key] = vel_ds
            int_results[key] = int_ds

        return Batch(vel_results), Batch(int_results)

    def incidence(self) -> "Batch":
        """Compute incidence angle from look vector components."""
        import rioxarray  # for .rio accessor

        # Get CRS from input batch
        crs = self.crs

        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            look_E = tfm["look_E"]
            look_N = tfm["look_N"]
            look_U = tfm["look_U"]
            incidence = xr.ufuncs.atan2(xr.ufuncs.sqrt(look_E ** 2 + look_N ** 2), look_U) * xr.ufuncs.sign(look_E).astype("float32")
            result_ds = xr.Dataset({"incidence": incidence})
            result_ds.attrs = tfm.attrs
            # Preserve CRS
            if crs is not None:
                result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        return Batch(out)

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) to convert phase to complex phasor.

        Parameters
        ----------
        sign : int, optional
            Sign of the exponent. Default is -1 for exp(-1j * phase).

        Returns
        -------
        BatchComplex
            Complex phasor representation.
        """
        import xarray as xr
        return BatchComplex(self.map_da(lambda da: xr.ufuncs.exp(sign * 1j * da), **kwargs))

class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None, wrap: bool = True):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        # skip wrapping for intermediate objects like DatasetCoarsen
        if not wrap:
            dict.__init__(self, mapping or {})
        else:
            wrapped = {k: self.wrap(v) for k, v in (mapping or {}).items()}
            dict.__init__(self, wrapped)

    @staticmethod
    def wrap(data):
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def __add__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: Batch):
        import xarray as xr
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    # Get a spatial variable (with y, x dims) to check for pair dimension
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)
                    first_elem = val[0]

                    if isinstance(first_elem, (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = ds - offsets
                    elif len(val) == 1:
                        # Single value wrapped in list: [offset]
                        result[k] = ds - val[0]
                    else:
                        # Single pair degree=1: [ramp, offset]
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                else:
                    result[k] = ds - val
        return type(self)(result)

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] * other[k] if k in other else self[k] for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] / other[k] if k in other else self[k] for k in keys})

    def sin(self, **kwargs) -> Batch:
        """
        Return a Batch of the sin(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da: xr.ufuncs.sin(da), **kwargs))

    def cos(self, **kwargs) -> Batch:
        """
        Return a Batch of the cos(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da: xr.ufuncs.cos(da), **kwargs))
    
    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) like np.exp(-1j * intfs)
        
        - If sign = -1 (the default), this is exp(-1j * da).
        - If sign = +1, this is exp(+1j * da).
        """
        from .Batch import BatchComplex
        return BatchComplex(self.map_da(lambda da: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
        """
        #print ('wrap _agg')
        import inspect
        import xarray as xr
        import pandas as pd
        out = {}
        for key, obj in self.items():
            # get the aggregation function
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            
            # perform aggregation in complex domain
            if 'dim' in sig.parameters:
                # intfs.mean('pair').isel(0)
                #agg_result = fn(dim=dim, **kwargs)
                complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
                #fn_complex = getattr(complex_obj, name)
                #agg_result = fn_complex(dim=dim, **kwargs)
                if name in ('var', 'std'):
                    # |E[e^(iθ)]|
                    R = xr.ufuncs.abs(complex_obj.mean(dim=dim, **kwargs))
                    if name == 'var':
                        # 1 - |E[e^(iθ)]|
                        agg_result = (1 - R)
                    else:  # std
                        # √(-2 ln|E[e^(iθ)]|)
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    fn_complex = getattr(complex_obj, name)
                    agg_result = fn_complex(dim=dim, **kwargs)
                    # convert back to wrapped phase
                    agg_result = xr.ufuncs.angle(agg_result)
            else:
                # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
                # already in complex domain, see coarsen()
                if name in ('var', 'std'):
                    R = xr.ufuncs.abs(obj.mean(**kwargs))
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    agg_result = xr.ufuncs.angle(agg_result)
            
            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')
            
        #print ('wrap _agg self.chunks', self.chunks)
        #return type(self)(out).chunk(self.chunks)
        #print ('wrap _agg self.chunks', self.chunks)
        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        #print ('wrap chunks', chunks)
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        #print ('wrap coarsen')
        chunks = self.chunks
        #print ('self.chunks', chunks)
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics
            ds2 = xr.ufuncs.exp(1j * ds.astype('float32'))
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds2 = ds2.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
            out[key] = ds2.coarsen(window, **kwargs)

        # wrap=False since these are DatasetCoarsen objects, not actual data
        return type(self)(out, wrap=False)

    def plot(
        self,
        cmap = 'gist_rainbow_r',
        alpha = 0.7,
        caption='Phase, [rad]',
        vmin=-np.pi,
        vmax=np.pi,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        return super().plot(*args, **kwargs)

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing for wrapped phase data.
    #     """
    #     return self.iexp().gaussian(*args, **kwargs).angle()

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
    #     then recombining via atan2.  No complex dtype ever created.
    #     """
    #     from .Batch import Batch
    #     import xarray as xr

    #     keep_attrs = kwargs.pop('keep_attrs', None)
    #     # build two Batches of the real sin and cos components and filter them
    #     sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
    #     cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

    #     # compute wrapped phase using np.arctan2
    #     out = {k: xr.Dataset({
    #         var: xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
    #         for var in sin[k].data_vars
    #     }) for k in self.keys()}

    #     return BatchWrap(out)

    def gaussian(self, *args, **kwargs):
        """
        Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
        then recombining via arctan2.
        """
        from .Batch import Batch
        import xarray as xr

        keep_attrs = kwargs.pop('keep_attrs', False)
        data_vars = next(iter(self.values())).data_vars

        # build two Batches of the real sin and cos components and filter them
        sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
        cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

        # compute wrapped phase using arctan2
        out: dict[str, xr.Dataset] = {}
        for k in self.keys():
            phase_vars = {}
            for var in data_vars:
                phase = xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
                if keep_attrs:
                    phase.attrs = self[k][var].attrs.copy()
                phase_vars[var] = phase
            ds = xr.Dataset(phase_vars)
            if keep_attrs:
                ds.attrs = self[k].attrs.copy()
            out[k] = ds

        return BatchWrap(out)

class BatchUnit(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores correlation in the range [0,1].
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchWrap, BatchComplex)):
            raise ValueError(f'ERROR: BatchUnit does not support Stack, BatchWrap or BatchComplex objects.')
        dict.__init__(self, mapping or {})

    def plot(
        self,
        cmap = 'auto',
        caption=None,
        alpha=1,
        vmin=0,
        vmax=1,
        *args,
        **kwargs
    ):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        kwargs["alpha"] = alpha
        return super().plot(*args, **kwargs)

class BatchComplex(BatchCore):
    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the complex vars from Stack
        if isinstance(mapping, Stack):
            complex_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only complex data_vars
                complex_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                ]
                complex_dict[key] = ds[complex_vars]
            mapping = complex_dict

        #print('BatchComplex __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})

    def real(self, **kwargs):
        """
        Return the real part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            # ds.map() applies the lambda to each DataArray in the Dataset
            ds_real = ds.map(lambda da: da.real, **kwargs)
            out[key] = ds_real
        return Batch(out)

    def imag(self, **kwargs):
        """
        Return the imaginary part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            ds_imag = ds.map(lambda da: da.imag, **kwargs)
            out[key] = ds_imag
        return Batch(out)

    def abs(self, **kwargs):
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs))

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        # Optimized: avoid sqrt in abs() by computing real² + imag² directly
        return Batch(self.map_da(lambda da: da.real**2 + da.imag**2, **kwargs))

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle), returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        return BatchWrap(self.map_da(lambda da: np.arctan2(da.imag, da.real).astype(np.float32), **kwargs))

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            # select only the vars whose dtype is complex
            complex_vars = [
                var for var in ds.data_vars
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def plot(self, *args, **kwargs):
        """
        Plotting is not supported on raw complex batches.
        Convert to real values first (e.g. with .angle() or .abs()).
        """
        raise NotImplementedError(
            "BatchComplex objects do not support plot().\n"
            "Convert to a real-valued batch first, e.g.:\n"
            "  • use `.angle()` to get wrapped phase → BatchWrap\n"
            "  • use `.abs()` or `.power()` to get magnitude → Batch"
        )

    @staticmethod
    def _goldstein(phase_np, corr_np, psize=32, device='auto', batch_size=None):
        """
        Apply Goldstein adaptive filter using PyTorch with vectorized unfold/fold.

        Uses torch.nn.functional.unfold/fold for efficient patch extraction and
        accumulation, avoiding slow Python loops.

        Parameters
        ----------
        phase_np : np.ndarray
            2D complex numpy array of phase data.
        corr_np : np.ndarray
            2D real numpy array of correlation values.
        psize : int or dict
            Patch size for the filter. Default is 32.
        device : str, optional
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.
        batch_size : int, optional
            Ignored (kept for API compatibility).

        Returns
        -------
        np.ndarray
            Filtered complex array with same shape as input.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        from .BatchCore import BatchCore

        if psize is None:
            return phase_np

        # Handle (1, y, x) arrays from apply_ufunc
        squeeze = False
        if phase_np.ndim == 3 and phase_np.shape[0] == 1:
            phase_np = phase_np[0]
            corr_np = corr_np[0] if corr_np.ndim == 3 else corr_np
            squeeze = True

        if isinstance(psize, dict):
            psize_y, psize_x = psize['y'], psize['x']
        else:
            psize_y, psize_x = int(psize), int(psize)

        dev = BatchCore._get_torch_device(device)
        ny, nx = phase_np.shape
        step_y, step_x = psize_y // 2, psize_x // 2

        # Save NaN mask before modifying
        nan_mask = ~np.isfinite(phase_np)

        # Create triangular weight matrix
        wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
        wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
        quadrant = np.outer(wy, wx)
        wgt_np = np.block([[quadrant, np.flip(quadrant, axis=1)],
                           [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
        wgt = torch.from_numpy(wgt_np).to(dev)
        wgt_sum = wgt.sum()
        del wx, wy, quadrant, wgt_np

        # Prepare data - fill NaN with 0
        phase_clean = np.nan_to_num(phase_np, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
        corr_clean = np.nan_to_num(corr_np, nan=0.0, copy=True)

        with torch.no_grad():
            # Pad to make dimensions work with unfold/fold
            pad_h = (step_y - ((ny - psize_y) % step_y)) % step_y
            pad_w = (step_x - ((nx - psize_x) % step_x)) % step_x

            if pad_h > 0 or pad_w > 0:
                phase_clean = np.pad(phase_clean, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                corr_clean = np.pad(corr_clean, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            padded_h, padded_w = phase_clean.shape

            # Convert to torch tensors - split complex into 2 channels: (1, 2, H, W)
            phase_t = torch.from_numpy(
                np.stack([phase_clean.real, phase_clean.imag], axis=0)
            ).unsqueeze(0).float().to(dev)
            corr_t = torch.from_numpy(corr_clean).unsqueeze(0).unsqueeze(0).to(dev)
            del phase_clean, corr_clean

            # Use unfold to extract all patches at once
            # Output: (1, C*kH*kW, num_patches)
            phase_patches = F.unfold(phase_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            corr_patches = F.unfold(corr_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            del phase_t, corr_t

            num_patches = phase_patches.shape[2]

            # Reshape phase: (1, 2*psize*psize, L) -> (L, 2, psize, psize) -> complex (L, psize, psize)
            phase_patches = phase_patches.squeeze(0).view(2, psize_y, psize_x, num_patches).permute(3, 0, 1, 2)
            phase_patches = torch.complex(phase_patches[:, 0], phase_patches[:, 1])

            # Reshape corr: (1, psize*psize, L) -> (L, psize, psize)
            corr_patches = corr_patches.squeeze(0).view(psize_y, psize_x, num_patches).permute(2, 0, 1)

            # Compute alpha for each patch
            weighted_corr = (wgt.unsqueeze(0) * corr_patches).sum(dim=(1, 2))
            alpha = 1.0 - weighted_corr / wgt_sum
            del corr_patches, weighted_corr

            # Check valid patches (at least 50% non-zero)
            valid_count = (phase_patches != 0).sum(dim=(1, 2)).float()
            valid_mask = valid_count >= (psize_y * psize_x * 0.5)
            del valid_count

            # Batched FFT and power spectrum filtering
            fft_patches = torch.fft.fft2(phase_patches)
            del phase_patches
            magnitude = torch.abs(fft_patches).clamp(min=1e-10)
            pspec_weight = torch.pow(magnitude, alpha.unsqueeze(1).unsqueeze(2))
            del magnitude, alpha
            fft_filtered = pspec_weight * fft_patches
            del pspec_weight, fft_patches

            # Batched IFFT
            filtered = torch.fft.ifft2(fft_filtered)
            del fft_filtered

            # Apply weight and mask invalid patches
            weighted = wgt.unsqueeze(0) * filtered
            del filtered
            weighted = torch.where(valid_mask.unsqueeze(1).unsqueeze(2), weighted, torch.zeros_like(weighted))

            # Prepare for fold: (1, 2*psize*psize, num_patches)
            weighted_ri = torch.stack([weighted.real, weighted.imag], dim=1)
            del weighted
            weighted_ri = weighted_ri.permute(1, 2, 3, 0).reshape(1, 2 * psize_y * psize_x, num_patches)

            # Fold valid_mask for counting overlaps
            valid_for_fold = valid_mask.float().view(1, 1, num_patches).expand(1, psize_y * psize_x, num_patches)
            del valid_mask

            # Fold back - accumulates overlapping patches
            result_ri = F.fold(weighted_ri, output_size=(padded_h, padded_w),
                               kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            count = F.fold(valid_for_fold, output_size=(padded_h, padded_w),
                           kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            del weighted_ri, valid_for_fold

            # Normalize
            result_ri = result_ri / count.clamp(min=1)
            result_ri = result_ri.squeeze(0)  # (2, H, W)
            del count

            # Crop to original size and combine real/imag
            result = torch.complex(result_ri[0, :ny, :nx], result_ri[1, :ny, :nx])
            result = result.cpu().numpy()
            del result_ri

        # Clear GPU cache
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()
        del wgt

        # Mask where original was NaN
        result[nan_mask] = np.nan + 1j * np.nan

        if squeeze:
            result = result[np.newaxis, ...]
        return result

    def goldstein(self, corr: BatchUnit, psize: int | dict[str, int] = 32, device: str = 'auto', debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.

        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        psize : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        BatchComplex
            New batch with filtered phase values
        """
        import numpy as np
        import dask
        import dask.array as da

        if debug:
            print('DEBUG: goldstein')

        if psize is None:
            return self

        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")

        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")

        if isinstance(psize, int):
            psize = {'y': psize, 'x': psize}

        # Determine if GPU resource annotation is needed
        use_gpu = device in ('cuda', 'mps')

        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            corr_ds = corr[k]
            filtered_vars = {}

            # Process each complex data variable in the dataset
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    corr_da = corr_ds[var_name]
                    phase_dask = var_data.data
                    corr_dask = corr_da.data

                    # Wrapper for _goldstein
                    def goldstein_block(phase_block, corr_block):
                        return BatchComplex._goldstein(phase_block, corr_block, psize=psize, device=device)

                    # Build dimension string based on ndim (e.g., 'yx' for 2D, 'pyx' for 3D)
                    dim_str = ''.join(chr(ord('a') + i) for i in range(phase_dask.ndim))

                    # Use da.blockwise for efficient dask integration
                    with dask.annotate(resources={'gpu': 1} if use_gpu else {}):
                        filtered_dask = da.blockwise(
                            goldstein_block, dim_str,
                            phase_dask, dim_str,
                            corr_dask, dim_str,
                            dtype=np.complex64,
                        )
                    filtered_vars[var_name] = xr.DataArray(
                        filtered_dask,
                        dims=var_data.dims,
                        coords=var_data.coords
                    )
                else:
                    filtered_vars[var_name] = var_data

            # Create a new dataset with the filtered variables
            result[k] = xr.Dataset(
                filtered_vars,
                coords=ds.coords,
                attrs=ds.attrs
            )

        return type(self)(result)


class BatchList(tuple):
    """
    A tuple-like container for multiple Batch objects that allows chained operations.

    Enables operations like:
        mintf, mcorr = stack.phasediff(...).downsample(20).compute()
        mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        mintf, mcorr = BatchList.open('mintf_corr')

    Instead of:
        mintf, mcorr = stack.phasediff(...)
        mintf, mcorr = stack.compute(mintf.downsample(20), mcorr.downsample(20))
    """

    def __new__(cls, batches=()):
        return super().__new__(cls, batches)

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None, n_jobs: int = -1, debug: bool = False):
        """Save or open a BatchList snapshot.

        When called on a BatchList with data, saves all batches to Zarr store.
        When called on an empty BatchList(), opens an existing store.

        Parameters
        ----------
        store : str
            Path to the Zarr store.
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        debug : bool
            Print debug information.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        >>> # Open
        >>> mintf, mcorr = BatchList().snapshot('mintf_corr')
        """
        from . import utils_io

        if len(self) == 0:
            # Open mode - no data args
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       compat=True, caption=caption or 'Opening...', n_jobs=n_jobs, debug=debug)
        else:
            # Save mode - pass batches directly to preserve types
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       compat=True, caption=caption or 'Snapshotting...', n_jobs=n_jobs, debug=debug)

        if isinstance(result, tuple):
            return result
        return (result,)

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                n_jobs: int = -1, debug: bool = False):
        """Save or open a BatchList archive as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Parameters
        ----------
        store : str
            Path to the ZIP file. Must end with '.zip'.
        caption : str, optional
            Progress bar caption.
        compression : int
            ZIP compression level 0-9 (0=no compression, 9=max). Default 6.
            Higher values produce smaller files but take longer.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        debug : bool
            Print debug information.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save to zip
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).archive('mintf_corr.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> mintf, mcorr = stack.phasediff(...).archive('mintf_corr.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> mintf, mcorr = stack.phasediff(...).archive('gs://bucket/mintf_corr.zip')
        >>> # Open from zip
        >>> mintf, mcorr = BatchList().archive('mintf_corr.zip')
        """
        import zarr
        import zipfile
        import tempfile
        import os
        import fsspec

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

        if len(self) == 0:
            # Open mode - check file exists first
            if is_cloud:
                fs, path = fsspec.core.url_to_fs(store)
                if not fs.exists(path):
                    raise FileNotFoundError(f"Archive not found: {store}")
            elif not os.path.exists(store):
                raise FileNotFoundError(f"Archive not found: {store}")
            # Use ZipStore directly for reading
            zip_store = zarr.storage.ZipStore(store, mode='r')
            result = self.snapshot(store=zip_store, caption=caption or 'Opening archive...', n_jobs=n_jobs, debug=debug)
            zip_store.close()
            return result
        else:
            # Save mode - write to temp directory, then zip
            # This avoids ZipStore's duplicate entry problem
            temp_dir = tempfile.mkdtemp()
            try:
                result = self.snapshot(store=temp_dir, caption=caption or 'Archiving...', n_jobs=n_jobs, debug=debug)
                # Create zip with specified compression level
                # Use fsspec for cloud storage support
                with fsspec.open(store, 'wb') as f:
                    with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression) as zf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zf.write(file_path, arcname)
            finally:
                import shutil
                shutil.rmtree(temp_dir)
            return result

    def downsample(self, *args, **kwargs):
        """Apply downsample to all batches."""
        return BatchList([b.downsample(*args, **kwargs) for b in self])

    def crop(self, *args, **kwargs):
        """Apply crop to all batches."""
        return BatchList([b.crop(*args, **kwargs) for b in self])

    def sel(self, *args, **kwargs):
        """Apply sel to all batches."""
        return BatchList([b.sel(*args, **kwargs) for b in self])

    def isel(self, *args, **kwargs):
        """Apply isel to all batches."""
        return BatchList([b.isel(*args, **kwargs) for b in self])

    def compute(self):
        """Compute all batches together efficiently and return as tuple for unpacking.

        Computes all batches in a single dask graph execution, which is faster
        than computing them separately because shared computations are only
        performed once.
        """
        import dask
        from insardev_toolkit.progressbar import progressbar

        # persist() computes shared graph with progress tracking
        batch_dicts = [dict(b) for b in self]
        progressbar(
            result := dask.persist(*batch_dicts),
            desc='Computing Batches...'.ljust(25)
        )
        # materialize coordinates to local memory (same as BatchCore.compute)
        computed_batches = []
        for i, batch_dict in enumerate(result):
            computed = {}
            for key, ds in batch_dict.items():
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)
                computed[key] = ds
            computed_batches.append(type(self[i])(computed))
        return BatchList(computed_batches)

    def persist(self):
        """Persist all batches together efficiently and return as BatchList.

        Persists all batches in a single dask graph execution, which is faster
        than persisting them separately because shared computations are only
        performed once.
        """
        import dask
        from insardev_toolkit.progressbar import progressbar

        # Convert all batches to dicts for dask.persist
        batch_dicts = [dict(b) for b in self]

        # Persist all batches together in a single graph execution
        progressbar(
            result := dask.persist(*batch_dicts),
            desc='Persisting Batches...'.ljust(25)
        )

        # Convert back to Batch objects preserving types
        persisted_batches = [type(self[i])(batch_dict) for i, batch_dict in enumerate(result)]
        return BatchList(persisted_batches)

    def __getattr__(self, name):
        """Proxy unknown attributes to all batches if they're callable."""
        if name.startswith('_'):
            raise AttributeError(f"BatchList has no attribute '{name}'")

        # Check if all batches have this attribute and it's callable
        attrs = [getattr(b, name, None) for b in self]
        if all(callable(a) for a in attrs if a is not None):
            def method(*args, **kwargs):
                results = [getattr(b, name)(*args, **kwargs) for b in self]
                # If results are Batch-like, wrap in BatchList
                if results and hasattr(results[0], 'keys') and callable(results[0].keys):
                    return BatchList(results)
                return tuple(results)
            return method

        raise AttributeError(f"BatchList has no attribute '{name}'")
