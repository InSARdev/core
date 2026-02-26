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


def _apply_goldstein_for_dask(phase_block, corr_block, psize, threshold, device):
    """Module-level function for Goldstein filter blockwise operation (DEPRECATED).

    Defined at module level to avoid dask serialization issues with nested functions.
    Closures capturing variables can cause memory explosions in dask workers.
    """
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize,
                                   threshold=threshold, device=device)


def _apply_goldstein_2d_for_dask(phase_block, corr_block, psize=32, threshold=0.5, device='cpu'):
    """Module-level function for Goldstein filter map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles both 2D (y, x) and 3D (1, y, x) blocks - _goldstein() handles squeeze/unsqueeze.

    Parameters
    ----------
    phase_block : np.ndarray
        Complex array from dask, shape (y, x) or (1, y, x)
    corr_block : np.ndarray
        Real array from dask, shape (y, x) or (1, y, x)
    psize : int or dict
        Patch size for the filter
    threshold : float
        Minimum fraction of valid pixels
    device : str
        PyTorch device

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input
    """
    # _goldstein handles (1, y, x) -> squeeze -> process -> unsqueeze
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize,
                                   threshold=threshold, device=device)


def _apply_velocity_block(data_block, times_years, min_valid, device):
    """Module-level function for velocity computation with da.blockwise.

    Defined at module level to avoid dask serialization issues with nested functions.

    Parameters
    ----------
    data_block : np.ndarray
        3D array (n_dates, chunk_y, chunk_x) - all dates, spatial chunk
    times_years : tuple or list
        Time values in years from first date (converted from numpy by dask)
    min_valid : int
        Minimum valid points required
    device : str
        PyTorch device string

    Returns
    -------
    np.ndarray
        3D array (2, chunk_y, chunk_x) where [0] is velocity, [1] is intercept
    """
    import numpy as np
    # Convert times_years back to numpy array (dask serialization may convert to list)
    times_years = np.asarray(times_years, dtype=np.float32)
    vel, intercept = Batch._velocity_torch(data_block, times_years, min_valid=min_valid, device=device)
    # Stack velocity and intercept along a new first dimension
    return np.stack([vel, intercept], axis=0).astype(np.float32)


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

    @staticmethod
    def _compute_rgb(copol: np.ndarray, xpol: np.ndarray,
                     gamma: float = 1.0, brightness: float = 2.0,
                     quantile: list = None) -> np.ndarray:
        """
        Compute RGB composite from co-pol and cross-pol arrays.

        Parameters
        ----------
        copol : np.ndarray
            Co-polarization data (HH or VV), shape (..., y, x)
        xpol : np.ndarray
            Cross-polarization data (HV or VH), shape (..., y, x)
        gamma : float
            Gamma correction (>1 brightens dark areas)
        brightness : float
            Linear brightness multiplier
        quantile : list
            Quantile range for normalization, default [0.02, 0.98]

        Returns
        -------
        np.ndarray
            RGB array as float32 [0-1], shape (..., y, x, 3)
        """
        # Normalize each channel to [0, 1] using quantile stretch
        def normalize_channel(data):
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                return np.zeros_like(data)
            q_vals = quantile if quantile is not None else [0.02, 0.98]
            if np.isscalar(q_vals):
                q_vals = [q_vals, 1 - q_vals] if q_vals < 0.5 else [1 - q_vals, q_vals]
            q = np.nanquantile(valid, q_vals)
            vmin_ch, vmax_ch = q[0], q[-1]
            if vmax_ch <= vmin_ch:
                vmax_ch = vmin_ch + 1e-10
            normalized = (data - vmin_ch) / (vmax_ch - vmin_ch)
            return np.clip(normalized, 0, 1)

        # R=copol, G=xpol, B=copol
        r_norm = normalize_channel(copol)
        g_norm = normalize_channel(xpol)
        b_norm = normalize_channel(copol)

        # Apply gamma correction
        if gamma != 1.0:
            r_norm = np.power(r_norm, 1.0 / gamma)
            g_norm = np.power(g_norm, 1.0 / gamma)
            b_norm = np.power(b_norm, 1.0 / gamma)

        # Handle NaN (set to 0)
        nan_mask = ~np.isfinite(copol) | ~np.isfinite(xpol)
        r_norm = np.where(nan_mask, 0, r_norm)
        g_norm = np.where(nan_mask, 0, g_norm)
        b_norm = np.where(nan_mask, 0, b_norm)

        # Stack to RGB
        rgb_float = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # Apply brightness
        if brightness != 1.0:
            rgb_float = rgb_float * brightness
            rgb_float = np.clip(rgb_float, 0, 1)

        return rgb_float.astype(np.float32)

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

    def plot2(self, *args, **kwargs):
        """
        Plot dual-pol RGB composite (shortcut for plot(composite=True)).

        This is a convenience method for dual-polarization data that creates
        an RGB composite where R=co-pol, G=cross-pol, B=co-pol.

        All arguments are passed to plot() with composite=True.

        See Also
        --------
        plot : Full plotting method with all options.
        """
        kwargs["composite"] = True
        return self.plot(*args, **kwargs)

    def rgb(self, gamma: float = 1.0, brightness: float = 2.0, quantile: list = None):
        """
        Create RGB composite from dual-pol data as xarray DataArray.

        Standard dual-pol RGB decomposition: R=co-pol, G=cross-pol, B=co-pol
        - Magenta/pink: high co-pol, low cross-pol (surface scattering, urban)
        - Green: high cross-pol (volume scattering, vegetation)
        - White/gray: both high (mixed scattering)
        - Dark: both low (smooth surfaces, water)

        Parameters
        ----------
        gamma : float, optional
            Gamma correction for brightness. Default 1.0.
            Values > 1 brighten dark areas, < 1 increase contrast.
        brightness : float, optional
            Linear brightness multiplier. Default 2.0.
        quantile : list, optional
            Quantile range for normalization. Default [0.02, 0.98].

        Returns
        -------
        xr.DataArray
            RGB array with dims (band, y, x) or (date/pair, band, y, x).
            Values are uint8 [0-255]. NaN pixels have value 0.

        Examples
        --------
        >>> rgb = stack[['HH','HV']].isel(date=[0]).power().gaussian(60).rgb()
        >>> rgb.shape  # (3, y, x) - band first for rasterio compatibility

        >>> # Save as GeoTIFF (direct - no transpose needed)
        >>> import rasterio
        >>> from rasterio.transform import from_bounds
        >>> transform = from_bounds(float(rgb.x.min()), float(rgb.y.min()),
        ...                         float(rgb.x.max()), float(rgb.y.max()),
        ...                         rgb.sizes['x'], rgb.sizes['y'])
        >>> with rasterio.open('polsar.tif', 'w', driver='GTiff',
        ...                    height=rgb.sizes['y'], width=rgb.sizes['x'],
        ...                    count=3, dtype='uint8', crs=stack.crs,
        ...                    transform=transform) as dst:
        ...     dst.write(rgb.values)  # Already (3, H, W)
        """
        import numpy as np
        import xarray as xr
        import dask
        from insardev_toolkit import progressbar

        # Check for exactly 2 polarizations
        sample = next(iter(self.values()))
        polarizations = [v for v in sample.data_vars
                        if sample[v].dims[-2:] == ('y', 'x')]
        if len(polarizations) != 2:
            raise ValueError(f"rgb() requires exactly 2 polarizations, found {len(polarizations)}: {polarizations}")

        pol1, pol2 = polarizations[0], polarizations[1]

        # Get stack variable (date or pair)
        stackvar = list(sample[pol1].dims)[0] if len(sample[pol1].dims) > 2 else None

        # Merge to single dataset
        ds = self.to_dataset()
        da_copol = ds[pol1]
        da_xpol = ds[pol2]

        if stackvar is None:
            stackvar = 'fake'
            da_copol = da_copol.expand_dims({stackvar: [0]})
            da_xpol = da_xpol.expand_dims({stackvar: [0]})

        # Materialize
        da_copol, da_xpol = dask.persist(da_copol, da_xpol)
        progressbar([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

        # Compute RGB using shared method from BatchCore
        copol = da_copol.values
        xpol = da_xpol.values
        rgb_float = Batch._compute_rgb(copol, xpol, gamma=gamma,
                                       brightness=brightness, quantile=quantile)
        rgb_uint8 = (rgb_float * 255).astype(np.uint8)

        # Create DataArray with band-first order for rasterio compatibility
        if stackvar == 'fake':
            # Remove fake dimension: (1, y, x, 3) -> (y, x, 3) -> (3, y, x)
            rgb_uint8 = rgb_uint8[0]
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=['y', 'x', 'band'],
                coords={'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose('band', 'y', 'x')
        else:
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=[stackvar, 'y', 'x', 'band'],
                coords={stackvar: da_copol[stackvar], 'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose(stackvar, 'band', 'y', 'x')

        rgb_da.attrs['crs'] = self.crs
        return rgb_da

    def lee(self, *args, **kwargs):
        """
        Apply Enhanced Lee speckle filter to reduce noise while preserving edges.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "lee() requires insardev_backscatter extension"
        )

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
        import dask.array as da
        import numpy as np
        import pandas as pd
        import xarray as xr
        import rioxarray  # for .rio accessor

        nanoseconds_per_year = 365.25 * 24 * 60 * 60 * 1e9

        # Resolve device ONCE here, not in every task - use string for serialization
        if device == 'auto':
            resolved_device = Batch._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: velocity using device={device}")

        # Get CRS from input batch
        crs = self.crs

        # Resource annotation for GPU tasks
        task_resources = {'gpu': 1} if device != 'cpu' else {}

        vel_results = {}
        int_results = {}
        for key, ds in self.items():
            vel_vars = {}
            int_vars = {}
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            for var in [v for v in ds.data_vars
                       if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                data_arr = ds[var]

                # Convert dates to years from first date
                dates = pd.to_datetime(data_arr.date.values)
                times_ns = (dates - dates[0]).total_seconds() * 1e9
                times_years = np.array(times_ns / nanoseconds_per_year, dtype=np.float32)
                n_dates = len(dates)

                # Use dask auto-chunking for y,x based on memory usage
                # Velocity computation: data (n_dates * n_pixels * 4 bytes) + temps
                # Estimate ~3x data size for temps during computation
                mem_per_pixel = n_dates * 4 * 3  # 3x for intermediate tensors
                auto_chunks = dask.array.core.normalize_chunks(
                    'auto', (data_arr.y.size, data_arr.x.size),
                    dtype=np.dtype(f'V{mem_per_pixel}')
                )
                chunks_y, chunks_x = auto_chunks[0][0], auto_chunks[1][0]

                if debug:
                    print(f"DEBUG: velocity auto-chunks: y={chunks_y}, x={chunks_x}")

                # Rechunk: all dates together (-1), auto-chunked y,x
                data_arr = data_arr.chunk({'date': -1, 'y': chunks_y, 'x': chunks_x})
                data_dask = data_arr.data

                # Use da.map_blocks for efficient dask integration
                # Input: (n_dates, chunk_y, chunk_x) -> Output: (2, chunk_y, chunk_x)
                def process_block(data_block):
                    return _apply_velocity_block(data_block, times_years, min_valid, device)

                with dask.annotate(resources=task_resources):
                    result_dask = da.map_blocks(
                        process_block, data_dask,
                        dtype=np.float32,
                        drop_axis=0,
                        new_axis=0,
                        chunks=(2,) + data_dask.chunks[1:],
                    )

                # Unpack velocity (index 0) and intercept (index 1)
                vel_da = xr.DataArray(
                    result_dask[0],
                    dims=['y', 'x'],
                    coords={'y': data_arr.y, 'x': data_arr.x}
                )
                int_da = xr.DataArray(
                    result_dask[1],
                    dims=['y', 'x'],
                    coords={'y': data_arr.y, 'x': data_arr.x}
                )

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
        """Compute incidence angle from azi, rng, ele, and radar geometry parameters.

        Uses spherical Earth geometry with per-pixel satellite height interpolation
        and terrain elevation correction. Matches GMTSAR look vector results within ~0.07%.

        Required vars: azi, rng, ele, near_range, SC_height_start, SC_height_end,
                       earth_radius, rng_samp_rate, num_lines
        """
        import numpy as np
        import rioxarray  # for .rio accessor

        c = 299792458.0  # speed of light

        # Get CRS from input batch
        crs = self.crs

        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            # Get scalar parameters (mean if per-date)
            near_range = float(tfm['near_range'].mean().item())
            SC_height_start = float(tfm['SC_height_start'].mean().item())
            SC_height_end = float(tfm['SC_height_end'].mean().item())
            earth_radius = float(tfm['earth_radius'].mean().item())
            rng_samp_rate = float(tfm['rng_samp_rate'].mean().item())
            num_lines = float(tfm['num_lines'].mean().item())

            # Get per-pixel coordinates
            azi = tfm['azi']
            rng = tfm['rng']
            ele = tfm['ele']

            # Compute slant range to the actual elevated ground point
            range_pixel_size = c / (2 * rng_samp_rate)
            slant_range = near_range + rng * range_pixel_size

            # Interpolate satellite height based on azimuth position
            SC_height = SC_height_start + (SC_height_end - SC_height_start) * azi / (num_lines - 1)

            # Ground at earth_radius + ele from Earth center
            ground_dist = earth_radius + ele

            # Satellite at earth_radius + SC_height from Earth center
            sat_dist = earth_radius + SC_height

            # Spherical Earth geometry: law of cosines + law of sines
            cos_earth = (ground_dist**2 + sat_dist**2 - slant_range**2) / (2 * ground_dist * sat_dist)
            cos_earth = xr.where(cos_earth > 1, 1, xr.where(cos_earth < -1, -1, cos_earth))
            earth_angle = xr.ufuncs.arccos(cos_earth)
            sin_inc = sat_dist * xr.ufuncs.sin(earth_angle) / slant_range
            sin_inc = xr.where(sin_inc > 1, 1, xr.where(sin_inc < -1, -1, sin_inc))
            incidence = xr.ufuncs.arcsin(sin_inc).astype('float32')

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

    def lstsq(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
              cumsum: bool = True, debug: bool = False) -> 'Batch':
        """
        Weighted least squares network inversion to date-based time series.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the inversion (typically correlation).
        device : str
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        cumsum : bool
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch
            Phase time series with 'date' dimension instead of 'pair'.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap1d(weight=corr)
        >>> disp = unwrapped.lstsq(weight=corr)
        """
        from .Stack_unwrap1d import Stack_unwrap1d

        return Stack_unwrap1d.lstsq(Stack_unwrap1d(), self, weight=weight,
                                    device=device, cumsum=cumsum, debug=debug)

    def displacement_los(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing mission constants (radar_wavelength),
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            LOS displacement grids (meters), lazily scaled by the mission wavelength.

        Examples
        --------
        >>> disp_los = unwrapped.displacement_los(stack)
        >>> disp_los = unwrapped.displacement_los(stack.transform())
        """
        import numpy as np
        import xarray as xr
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        if not transform:
            raise ValueError('transform must contain at least one burst with radar_wavelength')

        transform_first = next(iter(transform.values()))

        def _scalar_from_ds(ds, name: str):
            if name in ds:
                var = ds[name]
                if var.ndim == 0:
                    return var.item()
                elif var.ndim >= 1:
                    values = var.values.flatten()
                    unique = np.unique(values)
                    if len(unique) != 1:
                        raise ValueError(f'{name} has multiple distinct values: {unique}')
                    return unique[0]
            return ds.attrs.get(name)

        wavelength = _scalar_from_ds(transform_first, 'radar_wavelength')
        if wavelength is None:
            raise KeyError('Missing radar_wavelength in transform')

        # scale factor from phase in radians to displacement in meters
        # constant is negative to make LOS = -1 * range change
        scale = -float(wavelength) / (4 * np.pi)

        out: dict[str, xr.Dataset] = {}
        for key, phase_ds in self.items():
            disp_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                disp = (data * scale).astype('float32')
                disp_vars[var_name] = disp
            out[key] = xr.Dataset(disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, transform: 'Batch | Stack', func, suffix: str = '') -> 'Batch':
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        # Decimate transform to match phase resolution for efficiency
        transform = Batch({k: transform[k].reindex(y=self[k].y, x=self[k].x, method='nearest')
                           for k in self.keys() if k in transform})

        los_batch = self.displacement_los(transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
                # align incidence to data grid
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')

                comp = (data / func(incidence)).astype('float32')

                if len(los_ds.data_vars) == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            out[key] = xr.Dataset(comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        return Batch(out)

    def displacement_vertical(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute vertical displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            Vertical displacement grids (meters).

        Examples
        --------
        >>> disp_v = unwrapped.displacement_vertical(stack)
        >>> disp_v = unwrapped.displacement_vertical(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute east-west displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            East-west displacement grids (meters).

        Examples
        --------
        >>> disp_ew = unwrapped.displacement_eastwest(stack)
        >>> disp_ew = unwrapped.displacement_eastwest(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.sin, suffix='eastwest')

    def elevation(self, transform: 'Batch | Stack', baseline: float | None = None) -> 'Batch':
        """Compute elevation (meters) from unwrapped phase grids.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch containing look vectors for incidence calculation,
            or a Stack (will call .transform() internally).
        baseline : float | None, optional
            Perpendicular baseline in meters. If None, uses burst-specific BPR
            from phase coordinates.

        Returns
        -------
        Batch
            Elevation grids as float32 datasets.

        Examples
        --------
        >>> elev = unwrapped.elevation(stack)
        >>> elev = unwrapped.elevation(stack.transform())
        """
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        incidence_batch = transform.incidence()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in self.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            slant_start = _scalar_from_ds(tfm, 'SC_height_start')
            slant_end = _scalar_from_ds(tfm, 'SC_height_end')
            if wavelength is None or slant_start is None or slant_end is None:
                raise KeyError(f"Missing parameters in transform for burst {key}: radar_wavelength, SC_height_start, SC_height_end")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            inc_da = incidence_batch[key]['incidence']
            slant_range = xr.DataArray(
                np.linspace(slant_start, slant_end, inc_da.sizes['x']),
                coords={'x': inc_da.coords['x']},
                dims=('x',)
            )

            ref_height = _scalar_from_ds(tfm, 'ref_height') or 0.0

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                    slant = slant_range.interp(x=data.x)
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')
                    slant = slant_range.reindex_like(data.x, method='nearest')

                # Height from phase formula: h = ref_height - λ * φ * R * cos(incidence) / (4π * B⊥)
                elev = ref_height - (wavelength * data * slant * xr.ufuncs.cos(incidence) / (4 * np.pi * bpr))
                name = var_name
                elev_vars[name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def stl(self, freq: str = 'W', periods: int = 52, robust: bool = False) -> 'Batch':
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Decomposes time series into trend, seasonal, and residual components.
        The Batch must have a 'date' dimension.

        Parameters
        ----------
        freq : str, optional
            Frequency string for resampling (default 'W' for weekly).
            Examples: '1W' for 1 week, '2W' for 2 weeks, '10d' for 10 days.
        periods : int, optional
            Number of periods for seasonal decomposition (default 52 for weekly data = 1 year).
        robust : bool, optional
            Whether to use robust fitting (slower but handles outliers better). Default False.

        Returns
        -------
        Batch
            Batch containing 'trend', 'seasonal', and 'resid' variables for each polarization.

        Examples
        --------
        >>> displacement = unwrapped.lstsq(weight=corr)
        >>> stl_result = displacement.stl(freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, resid components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        from .Stack_stl import Stack_stl

        return Stack_stl.stl(Stack_stl(), self, freq=freq, periods=periods, robust=robust)

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
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.sin(da), **kwargs))

    def cos(self, **kwargs) -> Batch:
        """
        Return a Batch of the cos(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.cos(da), **kwargs))

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) like np.exp(-1j * intfs)

        - If sign = -1 (the default), this is exp(-1j * da).
        - If sign = +1, this is exp(+1j * da).
        """
        from .Batch import BatchComplex
        return BatchComplex(self.map_da(lambda da, **kw: xr.ufuncs.exp(sign * 1j * da), **kwargs))

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

    def unwrap2d(self, weight: 'BatchUnit | None' = None, conncomp: bool = False,
                 conncomp_size: int = 1000, conncomp_gap: int | None = None,
                 conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                 device: str = 'auto', debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = phase.unwrap2d(weight=corr)  # With weights
        """
        from .Stack_unwrap2d import Stack_unwrap2d

        return Stack_unwrap2d.unwrap2d(Stack_unwrap2d(), self, weight=weight,
                                       conncomp=conncomp, conncomp_size=conncomp_size,
                                       conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                                       conncomp_linkcount=conncomp_linkcount, device=device,
                                       debug=debug, **kwargs)

    def unwrap2d_irls(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
                      max_iter: int = 50, tol: float = 1e-2, cg_max_iter: int = 10,
                      cg_tol: float = 1e-3, epsilon: float = 1e-2,
                      conncomp_size: int = 30, debug: bool = False) -> 'Batches':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

        This is the core unwrapping algorithm. Disconnected components are
        unwrapped independently and aligned using per-component circular mean.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        max_iter : int
            Maximum IRLS iterations. Default 50.
        tol : float
            Convergence tolerance. Default 1e-2.
        cg_max_iter : int
            Maximum conjugate gradient iterations. Default 10.
        cg_tol : float
            Conjugate gradient tolerance. Default 1e-3.
        epsilon : float
            Smoothing parameter for L1 approximation. Default 1e-2.
        conncomp_size : int
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default 30.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, BatchUnit):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: BatchUnit of component labels (uint16, 0=invalid, 1=largest, ...)

        Notes
        -----
        Uses a novel DCT+IRLS algorithm that combines DCT efficiency with IRLS
        robustness. See `utils_unwrap2d.irls_unwrap_2d` for algorithm details
        and references.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped, conncomp = phase.unwrap2d_irls(weight=corr)
        """
        from .Stack_unwrap2d import Stack_unwrap2d

        return Stack_unwrap2d.unwrap2d_irls(Stack_unwrap2d(), self, weight=weight,
                                            device=device, max_iter=max_iter, tol=tol,
                                            cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                                            epsilon=epsilon, conncomp_size=conncomp_size,
                                            debug=debug)

    def unwrap2d_link(self, conncomp_size: int = 10_000, conncomp_gap: int | None = None,
                      conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                      debug: bool = False) -> 'Batch':
        """
        Link disconnected components in already unwrapped phase.

        This function applies component linking to already unwrapped phase data
        by finding optimal 2π offsets between disconnected components.
        Use this to correct phase jumps between components after unwrapping.

        Parameters
        ----------
        conncomp_size : int
            Minimum pixels for a connected component. Default 10,000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.

        Examples
        --------
        >>> # First unwrap without linking
        >>> unwrapped = phase.unwrap2d_irls(weight=corr)
        >>>
        >>> # Then link components separately
        >>> linked = unwrapped.unwrap2d_link(conncomp_size=10_000, debug=True)
        """
        from .Stack_unwrap2d import Stack_unwrap2d

        return Stack_unwrap2d.unwrap2d_link(Stack_unwrap2d(), self,
                                            conncomp_size=conncomp_size,
                                            conncomp_gap=conncomp_gap,
                                            conncomp_linksize=conncomp_linksize,
                                            conncomp_linkcount=conncomp_linkcount,
                                            debug=debug)

    def unwrap1d(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
                 debug: bool = False, **kwargs) -> 'Batch':
        """
        1D temporal phase unwrapping using IRLS optimization.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, epsilon, batch_size.

        Returns
        -------
        Batch
            Temporally unwrapped phase.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap1d()  # Without weights
        >>> unwrapped = phase.unwrap1d(weight=corr)  # With weights
        """
        from .Stack_unwrap1d import Stack_unwrap1d

        return Stack_unwrap1d.unwrap1d(Stack_unwrap1d(), self, weight=weight,
                                       device=device, debug=debug, **kwargs)

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

    def backscatter(self, *args, **kwargs):
        """
        Compute backscatter intensity (sigma0) from radiometrically calibrated SLC data.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "backscatter() requires insardev_backscatter extension"
        )

    def adi(self, *args, **kwargs):
        """
        Compute Amplitude Dispersion Index (ADI) for calibrated σ₀ data.

        ADI = std(amplitude) / mean(amplitude) over time.
        Lower ADI indicates more stable scatterers (PS candidates).

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "adi() requires insardev_backscatter extension"
        )

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

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
    def _goldstein(phase_np, corr_np, psize=32, threshold=0.5, device='auto'):
        """
        Apply Goldstein adaptive filter.

        Uses loop-based processing for CPU (constant memory) and
        PyTorch unfold/fold for GPU (vectorized).

        Parameters
        ----------
        phase_np : np.ndarray
            2D complex numpy array of phase data.
        corr_np : np.ndarray
            2D real numpy array of correlation values.
        psize : int or dict
            Patch size for the filter. Default is 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str, optional
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        np.ndarray
            Filtered complex array with same shape as input.
        """
        import numpy as np
        from .BatchCore import BatchCore
        from .utils_goldstein import goldstein_numpy, goldstein_pytorch

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

        # Ensure correct dtypes (goldstein functions require complex64/float32)
        if phase_np.dtype != np.complex64:
            phase_np = phase_np.astype(np.complex64)
        if corr_np.dtype != np.float32:
            corr_np = corr_np.astype(np.float32)

        # Dispatch based on device
        dev = BatchCore._get_torch_device(device)

        if dev.type == 'cpu':
            result = goldstein_numpy(phase_np, corr_np, psize_y, psize_x, threshold=threshold)
        else:
            result = goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, dev, threshold=threshold)

        if squeeze:
            result = result[np.newaxis, ...]
        return result

    def goldstein(self, corr: BatchUnit, window: int | dict[str, int] = 32, threshold: float = 0.5,
                  device: str = 'auto', debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.

        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        window : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        threshold : float, optional
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
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

        if window is None:
            return self

        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")

        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")

        # Validate lazy data
        BatchCore._require_lazy(self, 'goldstein')

        if isinstance(window, int):
            window = {'y': window, 'x': window}

        # Resolve device ONCE here, not in every task
        if device == 'auto':
            resolved_device = BatchCore._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

        # Resource annotation limits concurrent torch operations
        # Workers must be configured with resources={'goldstein': N} to limit concurrency
        use_gpu = device in ('cuda', 'mps')
        task_resources = {'goldstein': 1, 'gpu': 1} if use_gpu else {'goldstein': 1}

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

                    # Require first dimension chunked as 1 (avoid hidden rechunking overhead)
                    chunks = phase_dask.chunks
                    if var_data.ndim == 3 and chunks[0][0] != 1:
                        raise ValueError(
                            f"goldstein() requires first dimension chunked as 1, got chunks {chunks[0]}. "
                            f"Data should already have pair=1 chunks from load()."
                        )

                    # Calculate overlap depth: window//2 + 2 (PyGMTSAR formula)
                    depth_y = window['y'] // 2 + 2
                    depth_x = window['x'] // 2 + 2

                    if debug:
                        print(f'DEBUG: goldstein map_overlap depth=({depth_y}, {depth_x})')

                    # Require corr to have same shape as phase
                    if corr_dask.shape != phase_dask.shape:
                        raise ValueError(
                            f"goldstein() requires corr shape {corr_dask.shape} to match phase shape {phase_dask.shape}. "
                            f"Correlation should have the same dimensions as phase data."
                        )

                    # Require corr to have same chunks as phase (no hidden rechunking)
                    if hasattr(corr_dask, 'chunks') and corr_dask.chunks != phase_dask.chunks:
                        raise ValueError(
                            f"goldstein() requires corr chunks {corr_dask.chunks} to match phase chunks {phase_dask.chunks}. "
                            f"Use .chunk() to align chunks before calling goldstein()."
                        )

                    with dask.annotate(resources=task_resources):
                        depth_2d = {0: depth_y, 1: depth_x}
                        depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                        filtered_dask = da.map_overlap(
                            _apply_goldstein_2d_for_dask,
                            phase_dask,
                            corr_dask,
                            depth= depth_3d if var_data.ndim == 3 else depth_2d,
                            boundary='nearest',
                            dtype=np.complex64,
                            psize=window,
                            threshold=threshold,
                            device=device,
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


class Batches(tuple):
    """
    A tuple-like container for multiple Batch objects that allows chained operations.

    Enables operations like:
        mintf, mcorr = stack.phasediff(...).downsample(20).compute()
        mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        mintf, mcorr = Batches.open('mintf_corr')

    Instead of:
        mintf, mcorr = stack.phasediff(...)
        mintf, mcorr = stack.compute(mintf.downsample(20), mcorr.downsample(20))
    """

    def __new__(cls, batches=()):
        return super().__new__(cls, batches)

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None, allow_rechunk: bool = False,
                 n_jobs: int = 1, debug: bool = False):
        """Save or open a Batches snapshot.

        When called on a Batches with data, saves all batches to Zarr store.
        When called on an empty Batches(), opens an existing store.

        Parameters
        ----------
        store : str
            Path to the Zarr store.
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        allow_rechunk : bool
            If True, rechunk loaded data to optimal chunk sizes.
            If False (default), preserve single spatial chunks for MPS/GPU compatibility.
        n_jobs : int
            Number of bursts to process in parallel (default 1 = sequential).
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
        >>> mintf, mcorr = Batches().snapshot('mintf_corr')
        """
        from . import utils_io

        if len(self) == 0:
            # Open mode - no data args
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       compat=True, caption=caption or 'Opening...',
                                       allow_rechunk=allow_rechunk, n_jobs=n_jobs, debug=debug)
        else:
            # Save mode - pass batches directly to preserve types
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       compat=True, caption=caption or 'Snapshotting...',
                                       allow_rechunk=allow_rechunk, n_jobs=n_jobs, debug=debug)

        if isinstance(result, tuple):
            return result
        return (result,)

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                n_jobs: int = -1, debug: bool = False):
        """Save or open a Batches archive as a single ZIP file.

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
        >>> mintf, mcorr = Batches().archive('mintf_corr.zip')
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
        return Batches([b.downsample(*args, **kwargs) for b in self])

    def where(self, cond, other=np.nan, **kwargs):
        """Apply where mask to all batches."""
        return Batches([b.where(cond, other, **kwargs) for b in self])

    def crop(self, *args, **kwargs):
        """Apply crop to all batches."""
        return Batches([b.crop(*args, **kwargs) for b in self])

    def sel(self, *args, **kwargs):
        """Apply sel to all batches."""
        return Batches([b.sel(*args, **kwargs) for b in self])

    def isel(self, *args, **kwargs):
        """Apply isel to all batches."""
        return Batches([b.isel(*args, **kwargs) for b in self])

    def angle(self):
        """Apply angle() to BatchComplex batches, return others unchanged.

        Returns
        -------
        Batches
            Batches with BatchComplex converted to BatchWrap (phase angles),
            other batch types unchanged.

        Examples
        --------
        >>> phase, corr = stack.phasediff2(pairs).angle()
        >>> # phase is now BatchWrap with angles, corr is unchanged BatchUnit
        """
        results = []
        for b in self:
            if isinstance(b, BatchComplex):
                results.append(b.angle())
            else:
                results.append(b)
        return Batches(results)

    def goldstein(self, window: int | list[int, int] = 32, threshold: float = 0.5, device: str = 'auto'):
        """Apply Goldstein filter to phase using correlation as weight.

        Expects Batches with [BatchComplex (phase), BatchUnit (correlation)].

        Parameters
        ----------
        window : int or list[int, int]
            Goldstein filter patch size, default 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with Goldstein-filtered phase and unchanged correlation.

        Examples
        --------
        >>> phase, corr = stack.phasediff(pairs, wavelength=30).goldstein(32).angle()
        """
        if len(self) < 2:
            raise ValueError("goldstein() requires Batches with at least 2 elements: [phase, correlation]")

        phase, corr = self[0], self[1]

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"First element must be BatchComplex, got {type(phase).__name__}")
        if not isinstance(corr, BatchUnit):
            raise TypeError(f"Second element must be BatchUnit, got {type(corr).__name__}")

        filtered_phase = phase.goldstein(corr, window, threshold=threshold, device=device)
        return Batches([filtered_phase, corr] + list(self[2:]))

    def interferogram(self,
                  weight: 'BatchUnit | None' = None,
                  phase: 'BatchComplex | None' = None,
                  wavelength: float | None = None,
                  gaussian_threshold: float = 0.5,
                  device: str = 'auto') -> 'Batches':
        """
        Compute phase difference from paired SLC data.

        Expects Batches from pairs() with [ref, rep] BatchComplex objects.

        Parameters
        ----------
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking.
        phase : BatchComplex or None
            Optional phase to subtract (e.g., topographic phase).
        wavelength : float or None
            Gaussian filter wavelength for multilooking.
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with [phase, correlation].

        Examples
        --------
        >>> ref, rep = stack.pairs(baseline.tolist())
        >>> phase, corr = ref.interferogram(rep, wavelength=30)
        >>> # Or chained:
        >>> phase, corr = stack.pairs(baseline.tolist()).interferogram(wavelength=30)
        """
        if len(self) != 2:
            raise ValueError("interferogram() requires Batches with exactly 2 elements: [ref, rep]")

        ref, rep = self[0], self[1]

        if not isinstance(ref, BatchComplex) or not isinstance(rep, BatchComplex):
            raise TypeError("Both elements must be BatchComplex")

        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(stack.from_dataset(data)) to convert a single DataArray.'
            )

        # Resource annotation limits concurrent interferogram operations
        # Workers must be configured with resources={'interferogram': N} to limit concurrency
        import dask
        with dask.annotate(resources={'interferogram': 1}):
            intf = ref * rep.conj()
            if phase is not None:
                if isinstance(phase, BatchComplex):
                    intf = intf * phase
                else:
                    intf = intf * phase.iexp(-1)

            corr_look = BatchUnit()
            if wavelength is not None:
                intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
                intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
                intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
                del ref, rep
                corr_look = (intf_look.abs() / (intensity_ref * intensity_rep).sqrt()).clip(0, 1)
                del intensity_ref, intensity_rep
            else:
                intf_look = intf
                del ref, rep
            del intf

            if weight is not None:
                intf_look = intf_look.where(weight.isfinite())
                corr_look = corr_look.where(weight.isfinite()) if corr_look else None

        return Batches([intf_look, corr_look])

    def interferogram2(self, *args, **kwargs):
        """
        Compute optimized interferogram using dual-polarization coherence optimization.

        This method requires the insardev_polsar extension package.
        """
        raise ImportError(
            "interferogram2() requires insardev_polsar extension"
        )

    def compute(self, allow_rechunk: bool = False):
        """Compute all batches efficiently with sequential burst processing.

        Iterates through bursts sequentially, computing all dependent variables
        (e.g., intf and corr) together for each burst. This preserves shared
        computations within a burst while limiting memory usage by not holding
        all bursts in memory simultaneously.

        Parameters
        ----------
        allow_rechunk : bool, optional
            If True (default), auto-rechunk output for efficient further processing:
            - 3D data: chunk size 1 for first dim (date/pair), auto for y,x
            - 2D data: auto for y,x based on dask.config['array.chunk-size']
            If False, use single spatial chunk (-1 for y,x).

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        from tqdm.auto import tqdm
        from .utils_dask import get_dask_chunk_size_mb, rechunk2d

        # Get all burst keys (should be same across all batches)
        keys = list(self[0].keys())

        # Initialize result dicts for each batch
        computed_dicts = [{} for _ in self]

        # Print NOTE before progress bar
        dask_chunk_mb = get_dask_chunk_size_mb()
        if allow_rechunk:
            print(f"NOTE compute: rechunking to dask.config['array.chunk-size']={dask_chunk_mb} MB")
        else:
            print(f"NOTE compute: using single spatial chunk (allow_rechunk=False)")

        # Process one burst at a time across all batches
        for key in tqdm(keys, desc='Computing bursts'.ljust(25)):
            # Collect datasets for this burst from all batches
            burst_datasets = [batch[key] for batch in self]

            # Compute all datasets for this burst together (preserves shared computation)
            computed_datasets = dask.compute(*burst_datasets)

            # Store results and materialize coordinates
            for i, ds in enumerate(computed_datasets):
                # Materialize coordinates to local memory
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)

                # Rechunk computed data for efficient further processing
                # xarray's .chunk() handles both numpy and dask arrays
                rechunked_vars = {}
                for var in ds.data_vars:
                    arr = ds[var]
                    # Skip non-spatial variables
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue

                    if allow_rechunk:
                        # Use rechunk2d for uniform chunk sizes
                        y_size, x_size = arr.shape[-2], arr.shape[-1]
                        element_bytes = arr.dtype.itemsize
                        optimal = rechunk2d((y_size, x_size), element_bytes)
                        if arr.ndim == 3:
                            chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                        else:
                            chunks = {'y': optimal['y'], 'x': optimal['x']}
                    else:
                        # Single spatial chunk
                        if arr.ndim == 3:
                            chunks = {arr.dims[0]: 1, 'y': -1, 'x': -1}
                        else:
                            chunks = {'y': -1, 'x': -1}

                    rechunked_vars[var] = arr.chunk(chunks)

                # Update dataset with rechunked variables
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)

                computed_dicts[i][key] = ds

        # Reconstruct batches with correct types
        computed_batches = [type(self[i])(computed_dicts[i]) for i in range(len(self))]
        return Batches(computed_batches)

    def persist(self):
        """Persist all batches efficiently with sequential burst processing.

        Iterates through bursts sequentially, persisting all dependent variables
        (e.g., intf and corr) together for each burst. This preserves shared
        computations within a burst while limiting memory usage.

        Returns
        -------
        Batches
            Persisted batches with data in cluster memory.
        """
        import dask
        from tqdm.auto import tqdm

        # Get all burst keys (should be same across all batches)
        keys = list(self[0].keys())

        # Initialize result dicts for each batch
        persisted_dicts = [{} for _ in self]

        # Process one burst at a time across all batches
        for key in tqdm(keys, desc='Persisting bursts'.ljust(25)):
            # Collect datasets for this burst from all batches
            burst_datasets = [batch[key] for batch in self]

            # Persist all datasets for this burst together (preserves shared computation)
            persisted_datasets = dask.persist(*burst_datasets)

            # Store results
            for i, ds in enumerate(persisted_datasets):
                persisted_dicts[i][key] = ds

        # Reconstruct batches with correct types
        persisted_batches = [type(self[i])(persisted_dicts[i]) for i in range(len(self))]
        return Batches(persisted_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, device='auto', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        Expects Batches with [BatchWrap (phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = Batches([phase, corr]).unwrap2d()  # With weights
        """
        if len(self) < 1:
            raise ValueError("unwrap2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, device=device,
                              debug=debug, **kwargs)

    def unwrap1d(self, device='auto', debug=False, **kwargs):
        """
        1D temporal phase unwrapping using IRLS optimization.

        Expects Batches with [BatchWrap (phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, epsilon, batch_size.

        Returns
        -------
        Batch
            Temporally unwrapped phase.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = Batches([phase, corr]).unwrap1d()
        """
        if len(self) < 1:
            raise ValueError("unwrap1d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap1d
        return phase.unwrap1d(weight=weight, device=device, debug=debug, **kwargs)

    def trend2d(self, transform, degree=1, device='auto', debug=False):
        """
        Compute 2D polynomial trend (ramp) from phase.

        Expects Batches with [Batch/BatchWrap (phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Trend surface.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> trend = Batches([phase, corr]).trend2d(stack.transform())
        >>> detrended = phase - trend
        """
        if len(self) < 1:
            raise ValueError("trend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchWrap, BatchComplex)):
            raise TypeError(f"First element must be Batch, BatchWrap, or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchCore.trend2d
        return phase.trend2d(transform, weight=weight, degree=degree, device=device, debug=debug)

    def detrend2d(self, transform, degree=1, device='auto', debug=False):
        """
        Detrend 2D polynomial trend and return Batches with detrended data.

        Expects Batches from interferogram(): [BatchComplex (intf), BatchUnit (corr)]
        or from angle(): [BatchWrap (phase), BatchUnit (corr)].

        For complex input: detrended = intf * trend.conj()
        For wrapped input: detrended = wrap(phase - trend)
        For real input: detrended = phase - trend

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Batches with [detrended_phase, weight] preserving original types.

        Examples
        --------
        >>> # Complex interferogram detrending (chained)
        >>> intf, corr = stack.pairs(baseline).interferogram(wavelength=30).detrend2d(transform)
        >>> # Wrapped phase detrending
        >>> phase, corr = stack.pairs(baseline).phasediff(wavelength=30).angle().detrend2d(transform)
        """
        if len(self) < 1:
            raise ValueError("detrend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchWrap, BatchComplex)):
            raise TypeError(f"First element must be Batch, BatchWrap, or BatchComplex, got {type(phase).__name__}")

        trend = phase.trend2d(transform, weight=weight, degree=degree, device=device, debug=debug)

        if isinstance(phase, BatchComplex):
            detrended = phase * trend.conj()
        else:
            detrended = phase - trend

        # Rebuild Batches preserving all original elements except first
        elements = [detrended] + list(self[1:])
        return Batches(elements)

    def lstsq(self, device='auto', cumsum=True, debug=False):
        """
        Weighted least squares network inversion to date-based time series.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase. Expects Batches with [Batch (unwrapped phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        device : str
            PyTorch device ('auto', 'cuda', 'mps', 'cpu'). Default 'auto'.
        cumsum : bool
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        debug : bool
            Print debug information.

        Returns
        -------
        Batch
            Phase time series with 'date' dimension instead of 'pair'.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = Batches([phase, corr]).unwrap1d()
        >>> disp = Batches([unwrapped, corr]).lstsq()
        """
        if len(self) < 1:
            raise ValueError("lstsq() requires Batches with at least 1 element: [data]")

        data = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Delegate to Batch.lstsq
        return data.lstsq(weight=weight, device=device, cumsum=cumsum, debug=debug)

    def regression1d_baseline(self, baseline='BPR', degree=1, wrap=False, iterations=1,
                               device='auto', debug=False):
        """
        Fit 1D polynomial trend along perpendicular baseline at each (y, x) pixel.

        Expects Batches with [Batch/BatchWrap (phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        baseline : str
            Variable name to regress against (default 'BPR' for perpendicular baseline).
        degree : int
            Polynomial degree (1=linear, 2=quadratic). Default 1.
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Fitted trend values, same shape as input data.

        Examples
        --------
        >>> trend = Batches([intf, corr]).regression1d_baseline()
        >>> detrended = intf - trend
        """
        if len(self) < 1:
            raise ValueError("regression1d_baseline() requires Batches with at least 1 element: [data]")

        data = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Delegate to BatchCore.regression1d_baseline
        return data.regression1d_baseline(weight=weight, baseline=baseline, degree=degree,
                                           wrap=wrap, iterations=iterations,
                                           device=device, debug=debug)

    def regression1d_pairs(self, degree=0, days=None, count=None, wrap=False, iterations=1,
                            device='auto', debug=False):
        """
        Fit 1D polynomial trend along temporal pairs for each date.

        Expects Batches with [Batch/BatchWrap (phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        degree : int
            Polynomial degree (0=mean, 1=linear). Default 0.
        days : int or None
            Maximum time interval (in days) to include. Default None (all pairs).
        count : int or None
            Maximum number of pairs per date to use. Default None (all pairs).
        wrap : bool
            If True, use circular (sin/cos) fitting for wrapped phase.
        iterations : int
            Number of fitting iterations (default 1).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Fitted trend values, same shape as input data.

        Examples
        --------
        >>> trend = Batches([intf, corr]).regression1d_pairs(degree=0, days=100)
        >>> detrended = intf - trend
        """
        if len(self) < 1:
            raise ValueError("regression1d_pairs() requires Batches with at least 1 element: [data]")

        data = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Delegate to BatchCore.regression1d_pairs
        return data.regression1d_pairs(weight=weight, degree=degree, days=days, count=count,
                                        wrap=wrap, iterations=iterations,
                                        device=device, debug=debug)

    def stl(self, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Expects Batches with [Batch (time series data)]. No weight parameter needed.

        Parameters
        ----------
        freq : str
            Frequency string for resampling (default 'W' for weekly).
        periods : int
            Number of periods for seasonal decomposition (default 52).
        robust : bool
            Whether to use robust fitting. Default False.

        Returns
        -------
        Batch
            Batch containing 'trend', 'seasonal', and 'resid' variables.

        Examples
        --------
        >>> stl_result = Batches([displacement]).stl(freq='W', periods=52)
        """
        if len(self) < 1:
            raise ValueError("stl() requires Batches with at least 1 element: [data]")

        data = self[0]

        # Delegate to Batch.stl
        return data.stl(freq=freq, periods=periods, robust=robust)

    def __getattr__(self, name):
        """Proxy unknown attributes to all batches if they're callable."""
        if name.startswith('_'):
            raise AttributeError(f"Batches has no attribute '{name}'")

        # Check if all batches have this attribute and it's callable
        attrs = [getattr(b, name, None) for b in self]
        if all(callable(a) for a in attrs if a is not None):
            def method(*args, **kwargs):
                results = [getattr(b, name)(*args, **kwargs) for b in self]
                # If results are Batch-like, wrap in Batches
                if results and hasattr(results[0], 'keys') and callable(results[0].keys):
                    return Batches(results)
                return tuple(results)
            return method

        raise AttributeError(f"Batches has no attribute '{name}'")
