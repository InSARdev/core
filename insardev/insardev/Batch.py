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

        return vel_np, int_np

    def velocity(self, min_valid=3, device=None, debug=False) -> tuple["Batch", "Batch"]:
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
            PyTorch device ('cuda', 'mps', 'cpu'). Auto-detected if None.
            Respects Dask cluster resources={'gpu': 0} to disable GPU.
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
        device = Batch._get_torch_device(device if device is not None else 'auto', debug=debug)

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
                    from contextlib import nullcontext
                    # Use MPS lock to serialize GPU access on Apple Silicon
                    with Batch._mps_lock() if device.type == 'mps' else nullcontext():
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
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs))

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
    def _goldstein(phase, corr, psize=32, debug=False):
        import xarray as xr
        import numpy as np
        import dask
        from numbers import Real
        from collections.abc import Mapping
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            module=r'dask\._task_spec'
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            'ignore',
            message='invalid value encountered in divide',
            category=RuntimeWarning,
            module=r'dask\._task_spec'
        )

        if debug:
            print ('DEBUG: goldstein')

        if psize is None:
            # miss the processing
            return phase
        
        if not isinstance(psize, (Real, Mapping)):
            raise ValueError('ERROR: psize should be an integer, float, or dictionary')

        if isinstance(psize, Real):
            psize = {'y': psize, 'x': psize}

        # Handle Dataset objects by extracting the first DataArray
        if isinstance(phase, xr.Dataset):
            phase = next(iter(phase.data_vars.values()))
        if isinstance(corr, xr.Dataset):
            corr = next(iter(corr.data_vars.values()))

        def apply_pspec(data, alpha):
            # NaN is allowed value
            assert not(alpha < 0), f'Invalid parameter value {alpha} < 0'
            wgt = np.power(np.abs(data)**2, alpha / 2)
            data = wgt * data
            return data

        def make_wgt(psize):
            nyp, nxp = psize['y'], psize['x']
            # Create arrays of horizontal and vertical weights
            wx = 1.0 - np.abs(np.arange(nxp // 2) - (nxp / 2.0 - 1.0)) / (nxp / 2.0 - 1.0)
            wy = 1.0 - np.abs(np.arange(nyp // 2) - (nyp / 2.0 - 1.0)) / (nyp / 2.0 - 1.0)
            # Compute the outer product of wx and wy to create the top-left quadrant of the weight matrix
            quadrant = np.outer(wy, wx)
            # Create a full weight matrix by mirroring the quadrant along both axes
            wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                            [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]])
            return wgt

        def patch_goldstein_filter(data, corr, wgt, psize):
            """
            Apply the Goldstein adaptive filter to the given data.

            Args:
                data: 2D numpy array of complex values representing the data to be filtered.
                corr: 2D numpy array of correlation values. Must have the same shape as `data`.

            Returns:
                2D numpy array of filtered data.
            """
            # Calculate alpha
            alpha = 1 - (wgt * corr).sum() / wgt.sum()
            data = np.fft.fft2(data, s=(psize['y'], psize['x']))
            data = apply_pspec(data, alpha)
            data = np.fft.ifft2(data, s=(psize['y'], psize['x']))
            return wgt * data

        def apply_goldstein_filter(data, corr, psize, wgt_matrix):
            # Create an empty array for the output initialized with NaN
            out = np.full(data.shape, np.nan + 1j*np.nan, dtype=np.complex64)
            # ignore processing for empty chunks
            if np.all(np.isnan(data)):
                return out
            # Track which pixels are processed (for proper overlap accumulation)
            processed = np.zeros(data.shape, dtype=np.float32)
            # Temporary array for accumulation
            acc = np.zeros(data.shape, dtype=np.complex64)
            # Iterate over windows of the data
            for i in range(0, data.shape[0] - psize['y'], psize['y'] // 2):
                for j in range(0, data.shape[1] - psize['x'], psize['x'] // 2):
                    # Create proocessing windows
                    data_window = data[i:i+psize['y'], j:j+psize['x']]
                    corr_window = corr[i:i+psize['y'], j:j+psize['x']]
                    # do not process NODATA areas filled with zeros
                    fraction_valid = np.count_nonzero(data_window != 0) / data_window.size
                    if fraction_valid >= 0.5:
                        wgt_window = wgt_matrix[:data_window.shape[0],:data_window.shape[1]]
                        # Apply the filter to the window
                        filtered_window = patch_goldstein_filter(data_window, corr_window, wgt_window, psize)
                        # Add the result to the accumulation array
                        slice_i = slice(i, min(i + psize['y'], out.shape[0]))
                        slice_j = slice(j, min(j + psize['x'], out.shape[1]))
                        acc[slice_i, slice_j] += filtered_window[:slice_i.stop - slice_i.start, :slice_j.stop - slice_j.start]
                        processed[slice_i, slice_j] += 1
            # Only set output where pixels were processed
            mask = processed > 0
            out[mask] = acc[mask]
            return out

        assert phase.shape == corr.shape, f'ERROR: phase and correlation variables have different shape \
                                          ({phase.shape} vs {corr.shape})'

        stack =[]
        for ind in range(len(phase)):
            # Apply function with overlap; psize//2 overlap is not enough (some empty lines produced)
            # use complex data and real correlation
            # fill NaN values in correlation by zeroes to prevent empty output blocks
            block = dask.array.map_overlap(apply_goldstein_filter,
                                           phase[ind].fillna(0).data,
                                           corr[ind].fillna(0).data,
                                           depth=(psize['y'] // 2 + 2, psize['x'] // 2 + 2),
                                           dtype=np.complex64, 
                                           meta=np.array(()),
                                           psize=psize,
                                           wgt_matrix = make_wgt(psize))
            # Calculate the phase
            stack.append(block)
            del block

        # Create DataArray with proper coordinates and attributes
        ds = xr.DataArray(
            dask.array.stack(stack),
            coords=phase.coords,
            dims=phase.dims,
            name=phase.name,
            attrs=phase.attrs
        )
        del stack
        # replace zeros produces in NODATA areas
        return ds.where(np.isfinite(phase))

    def goldstein(self, corr: BatchUnit, psize: int | dict[str, int] = 32, debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.
        
        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        psize : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        debug : bool, optional
            Print debug information. Default is False.
            
        Returns
        -------
        BatchComplex
            New batch with filtered phase values
        """
        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")
            
        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")
            
        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            filtered_vars = {}
            
            # Process each complex data variable in the dataset
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    filtered_data = self._goldstein(
                        phase=var_data,
                        corr=corr[k],
                        psize=psize,
                        debug=debug
                    )
                    filtered_vars[var_name] = filtered_data
                else:
                    filtered_vars[var_name] = var_data
            
            # Create a new dataset with the filtered variables
            result[k] = xr.Dataset(
                filtered_vars,
                coords=ds.coords,
                attrs=ds.attrs
            )
            
        return type(self)(result)
