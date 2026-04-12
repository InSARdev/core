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
        3D array (2, chunk_y, chunk_x) where [0] is velocity, [1] is RMSE
    """
    import numpy as np
    times_years = np.asarray(times_years, dtype=np.float32)
    vel, rmse = Batch._velocity_torch(data_block, times_years, min_valid=min_valid, device=device)
    return np.stack([vel, rmse], axis=0).astype(np.float32)


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

        # Harmonic weighted regression: y = c0 + c1*t + c2*sin(2πt) + c3*cos(2πt)
        # Separates velocity from annual seasonal — unbiased for any time span.
        # Uses normal equations: (A^T W A) x = A^T W y, solved via Cholesky.

        t_expanded = t.unsqueeze(1)  # (n_times, 1)
        sin_t = torch.sin(2 * np.pi * t).unsqueeze(1)  # (n_times, 1)
        cos_t = torch.cos(2 * np.pi * t).unsqueeze(1)  # (n_times, 1)

        # Build A^T W A (4x4) and A^T W y (4) per pixel, vectorized
        # A columns: [1, t, sin, cos], each (n_times, n_pixels) after broadcast
        ones = torch.ones_like(t_expanded)  # (n_times, 1)
        cols = [ones, t_expanded, sin_t, cos_t]  # each (n_times, 1)

        # Weighted columns: w * col, each (n_times, n_pixels)
        wcols = [w * c for c in cols]

        # A^T W A: (4, 4, n_pixels)
        n_cols = 4
        AtWA = torch.zeros(n_cols, n_cols, data_2d.shape[1], device=dev)
        AtWy = torch.zeros(n_cols, data_2d.shape[1], device=dev)
        for i in range(n_cols):
            AtWy[i] = (wcols[i] * y_filled).sum(dim=0)
            for j in range(i, n_cols):
                val = (wcols[i] * cols[j]).sum(dim=0)
                AtWA[i, j] = val
                AtWA[j, i] = val

        # Solve per pixel: transpose to (n_pixels, 4, 4) and (n_pixels, 4)
        AtWA = AtWA.permute(2, 0, 1)  # (n_pixels, 4, 4)
        AtWy = AtWy.permute(1, 0)      # (n_pixels, 4)

        # Batch solve via torch.linalg.solve
        try:
            coeffs = torch.linalg.solve(AtWA, AtWy)  # (n_pixels, 4)
        except Exception:
            # Fallback: pseudo-inverse for singular matrices
            coeffs = torch.zeros(data_2d.shape[1], n_cols, device=dev)
            for px in range(data_2d.shape[1]):
                try:
                    coeffs[px] = torch.linalg.solve(AtWA[px], AtWy[px])
                except Exception:
                    coeffs[px] = float('nan')

        velocity = coeffs[:, 1]   # c1 = velocity (unbiased)

        # Compute RMSE: residuals = y - A @ coeffs
        # A @ coeffs per pixel: sum over basis functions
        predicted = torch.zeros_like(y_filled)
        for i, c in enumerate(cols):
            predicted += c * coeffs[:, i].unsqueeze(0)  # (n_times, n_pixels)
        residuals = (y_filled - predicted) * w  # zero out NaN positions
        rmse = torch.sqrt((residuals ** 2).sum(dim=0) / valid_count.clamp(min=1))

        # Mask pixels with insufficient valid points
        valid_mask = valid_count >= min_valid
        velocity = torch.where(valid_mask, velocity, torch.tensor(float('nan'), device=dev))
        rmse = torch.where(valid_mask, rmse, torch.tensor(float('nan'), device=dev))

        # Reshape back
        vel_np = velocity.cpu().numpy()
        rmse_np = rmse.cpu().numpy()
        if len(original_shape) == 3:
            vel_np = vel_np.reshape(original_shape[1], original_shape[2])
            rmse_np = rmse_np.reshape(original_shape[1], original_shape[2])

        # Cleanup GPU memory
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()

        return vel_np, rmse_np

    def velocity(self, min_valid=5, device='auto', debug=False) -> "Batches":
        """
        Compute velocity and RMSE from time series.

        Harmonic regression (linear + seasonal) per pixel on the 'date' dimension.
        Uses PyTorch for GPU acceleration.

        Parameters
        ----------
        min_valid : int, optional
            Minimum number of valid (non-NaN) data points required.
            Default is 5.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
        debug : bool, optional
            Print debug information. Default False.

        Returns
        -------
        Batches[Batch, Batch]
            (velocity, rmse) — velocity is slope per year, RMSE is residual
            root-mean-square error. Both are lazy Batch objects.

        Examples
        --------
        >>> velocity, rmse = displacement.velocity()
        >>> vel, rmse = detrend0.velocity().displacement_los(transform).compute()
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

        vel_results = {}
        rmse_results = {}
        for key, ds in self.items():
            vel_vars = {}
            rmse_vars = {}
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

                result_dask = da.map_blocks(
                    process_block, data_dask,
                    dtype=np.float32,
                    drop_axis=0,
                    new_axis=0,
                    chunks=(2,) + data_dask.chunks[1:],
                )

                vel_da = xr.DataArray(
                    result_dask[0], dims=['y', 'x'],
                    coords={'y': data_arr.y, 'x': data_arr.x}
                )
                rmse_da = xr.DataArray(
                    result_dask[1], dims=['y', 'x'],
                    coords={'y': data_arr.y, 'x': data_arr.x}
                )
                vel_vars[var] = vel_da
                rmse_vars[var] = rmse_da

            vel_ds = xr.Dataset(vel_vars)
            vel_ds.attrs = ds.attrs
            rmse_ds = xr.Dataset(rmse_vars)
            rmse_ds.attrs = ds.attrs
            if crs is not None:
                vel_ds = vel_ds.rio.write_crs(crs)
                rmse_ds = rmse_ds.rio.write_crs(crs)
            vel_results[key] = vel_ds
            rmse_results[key] = rmse_ds

        return Batches((Batch(vel_results), Batch(rmse_results)))

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

    def lstsq(self, weight: 'BatchUnit | None' = None,
              cumsum: bool = True, max_iter: int = 5, epsilon: float = 0.1,
              x_tol: float = 0.001, debug: bool = False) -> 'Batch':
        """
        L1-norm IRLS network inversion to date-based time series.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase. Uses IRLS with L1-norm for robustness against
        outlier pairs.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the inversion (typically correlation).
        cumsum : bool
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        max_iter : int
            Maximum IRLS iterations. Default 5.
        epsilon : float
            IRLS regularization parameter. Default 0.1.
        x_tol : float
            Solution convergence tolerance. Default 0.001.
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
                                    cumsum=cumsum,
                                    max_iter=max_iter, epsilon=epsilon,
                                    x_tol=x_tol, debug=debug)

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

    def trend1d(self, *args, **kwargs):
        raise TypeError(
            "trend1d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    def trend2d(self, *args, **kwargs):
        raise TypeError(
            "trend2d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    def trend1d_pairs(self, *args, **kwargs):
        raise TypeError(
            "trend1d_pairs() requires BatchComplex (complex wrapped phase). "
            "Place detrend1d_pairs() before unwrapping in the pipeline."
        )

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
                        # Use da.stack for dask 0-d arrays to avoid triggering .compute()
                        if any(hasattr(v, 'dask') for v in val):
                            import dask.array as _da
                            offsets = xr.DataArray(_da.stack(val), dims=['pair'])
                        else:
                            offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = ds - offsets
                    elif len(val) == 1:
                        # Single value wrapped in list: [offset]
                        result[k] = ds - val[0]
                    else:
                        # Single pair degree=1: [ramp, offset]
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                elif isinstance(val, (int, float)) \
                        or (hasattr(val, 'ndim') and val.ndim == 0):
                    # Scalar subtraction (concrete or dask 0-d array)
                    result[k] = ds - val
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

    def unwrap2d_chunk(self, weight: 'BatchUnit | None' = None, overlap=None,
                       device: str = 'auto', debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Unlike unwrap2d() which requires a single spatial chunk (global unwrapping),
        this method unwraps each spatial chunk independently with overlap margins.
        Suitable for large rasters where global unwrapping would exceed memory.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size (0.25 = 25%).
            Int = pixels. Tuple (y, x) for different overlap per axis. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batch
            Batch of unwrapped phase.
        """
        from .Stack_unwrap2d import Stack_unwrap2d

        return Stack_unwrap2d.unwrap2d_chunk(Stack_unwrap2d(), self, weight=weight,
                                              overlap=overlap, device=device,
                                              debug=debug, **kwargs)

    def unwrap2d_irls(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
                      max_iter: int = 50, tol: float = 1e-2, cg_max_iter: int = 10,
                      cg_tol: float = 1e-3, epsilon: float = 1e-2,
                      conncomp_size: int = 30, semaphore: int = 8, debug: bool = False) -> 'Batches':
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
        semaphore : int
            Maximum concurrent CPU IRLS tasks per process. Default 8.
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
                                            semaphore=semaphore, debug=debug)

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

    def unwrap1d(self, weight: 'BatchUnit | None' = None,
                 debug: bool = False, **kwargs) -> 'Batch':
        """
        1D temporal phase unwrapping using triplet pre-filtering and IRLS.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, epsilon.

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
                                       debug=debug, **kwargs)

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

    def threshold(self, weight=None, threshold=np.pi/2) -> "BatchComplex":
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Computes weighted cstd across all pairs per pixel. Pixels with
        cstd >= threshold are set to 0+0j (all pairs). Useful for rejecting
        incoherent pixels before velocity estimation or detrending.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional correlation weight for weighted cstd.
        threshold : float
            Maximum cstd in radians. Default π/2. Use π/4 for stricter filtering.

        Returns
        -------
        BatchComplex
            Filtered copy with incoherent pixels zeroed.
        """
        import dask.array as da
        import xarray as xr
        from . import utils_detrend

        BatchCore._require_lazy(self, 'threshold')

        results = {}
        for burst_id, burst_ds in self.items():
            burst_weight = weight[burst_id] if weight is not None else None
            filtered_vars = {}
            for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                data_da = burst_ds[pol]
                weight_da = burst_weight[pol] if burst_weight is not None else None

                if data_da.dims[0] != 'pair':
                    data_da = data_da.transpose('pair', ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                n_pairs_val = data_da.shape[0]

                def _threshold_block(data_block, weight_block=None,
                                     _threshold=threshold):
                    return utils_detrend.threshold_pairs_array(
                        [data_block],
                        [weight_block] if weight_block is not None else None,
                        threshold=_threshold,
                    )

                if weight_dask is not None:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )
                else:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )

                filtered_vars[pol] = xr.DataArray(filtered_dask, dims=data_da.dims,
                                                   coords=data_da.coords, name=pol)

            filtered_ds = burst_ds.assign(filtered_vars)
            results[burst_id] = filtered_ds

        return BatchComplex(results)

    def velocity(self, weight=None, max_refine=3, seasonal=False) -> "Batch":
        """
        Estimate global velocity from interferometric pair network using
        periodogram on the unit circle. Fast shortcut that skips the full
        detrend1d_pairs → unwrap1d → lstsq pipeline.

        Returns velocity in rad/year. Use displacement_los(transform) to
        convert to m/year, then multiply by 1000 for mm/year.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional correlation weight.
        max_refine : int
            Refinement levels (0=coarse ~32mm/yr, 3=fine ~0.5mm/yr). Default 3.

        Returns
        -------
        Batch
            Velocity (y, x) in rad/year per burst, as a lazy Batch.
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from . import utils_detrend

        BatchCore._require_lazy(self, 'velocity')

        vel_results = {}
        rmse_results = {}
        for burst_id, burst_ds in self.items():
            burst_weight = weight[burst_id] if weight is not None else None
            vel_vars = {}
            rmse_vars = {}
            for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                data_da = burst_ds[pol]
                weight_da = burst_weight[pol] if burst_weight is not None else None

                ref_values = burst_ds['ref'].values.astype('datetime64[ns]').astype(np.int64)
                rep_values = burst_ds['rep'].values.astype('datetime64[ns]').astype(np.int64)

                if data_da.dims[0] != 'pair':
                    data_da = data_da.transpose('pair', ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None

                def _velocity_block(data_block, weight_block=None,
                                    _ref=ref_values, _rep=rep_values,
                                    _max_refine=max_refine, _seasonal=seasonal):
                    vel, rmse = utils_detrend.velocity_pairs_array(
                        [data_block],
                        [weight_block] if weight_block is not None else None,
                        _ref, _rep,
                        max_refine=_max_refine, seasonal=_seasonal,
                    )
                    # Stack (2, y, x) so blockwise can return both
                    return np.stack([vel, rmse], axis=0)

                # Include seasonal in name to prevent dask cache collision
                blk_name = f'velocity_block_s{seasonal}'
                if weight_dask is not None:
                    stacked_dask = da.blockwise(
                        _velocity_block, 'nyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'n': 2},
                        concatenate=True,
                        dtype=np.float32,
                        meta=np.empty((0, 0, 0), dtype=np.float32),
                        name=blk_name,
                    )
                else:
                    stacked_dask = da.blockwise(
                        _velocity_block, 'nyx',
                        data_dask, 'pyx',
                        new_axes={'n': 2},
                        concatenate=True,
                        dtype=np.float32,
                        meta=np.empty((0, 0, 0), dtype=np.float32),
                        name=blk_name,
                    )

                coords = {k: v for k, v in data_da.coords.items() if k in ('y', 'x', 'spatial_ref')}
                vel_da = xr.DataArray(stacked_dask[0], dims=('y', 'x'), coords=coords, name=pol)
                rmse_da = xr.DataArray(stacked_dask[1], dims=('y', 'x'), coords=coords, name=pol)
                vel_vars[pol] = vel_da
                rmse_vars[pol] = rmse_da

            vel_ds = xr.Dataset(vel_vars)
            rmse_ds = xr.Dataset(rmse_vars)
            if 'spatial_ref' in burst_ds.coords:
                vel_ds = vel_ds.assign_coords(spatial_ref=burst_ds.spatial_ref)
                rmse_ds = rmse_ds.assign_coords(spatial_ref=burst_ds.spatial_ref)
            vel_results[burst_id] = vel_ds
            rmse_results[burst_id] = rmse_ds

        return Batches((Batch(vel_results), Batch(rmse_results)))

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

    def pairs(self, pairs):
        """Select date pairs from per-date data, returning ref and rep stacks.

        Parameters
        ----------
        pairs : array-like (n_pairs, 2)
            Pairs as [[ref_date, rep_date], ...]. Dates as datetime64 or indices.

        Returns
        -------
        tuple (ref, rep)
            Two BatchComplex with 'pair' dimension instead of 'date'.
        """
        import numpy as np
        pairs = np.asarray(pairs)
        ref_dates = pairs[:, 0]
        rep_dates = pairs[:, 1]

        # Map dates to integer indices (match by day to handle precision differences)
        key0 = list(self.keys())[0]
        date_coords = self[key0].coords['date'].values
        # Truncate to day precision for matching
        date_days = np.array(date_coords, dtype='datetime64[D]')
        date_to_idx = {d: i for i, d in enumerate(date_days)}
        ref_idx = [date_to_idx[np.datetime64(d, 'D')] for d in ref_dates]
        rep_idx = [date_to_idx[np.datetime64(d, 'D')] for d in rep_dates]

        # Select, rename date→pair, and assign pair coords matching the caller
        n_pairs = len(ref_idx)
        pair_coords = np.arange(n_pairs)
        screen_ref = self.isel(date=ref_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))
        screen_rep = self.isel(date=rep_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))

        return screen_ref, screen_rep

    def lstsq_baseline(self, weight=None, baseline='BPR', stride=1, debug=False):
        """
        Decompose per-pair complex trend into network-consistent per-date model.

        IRLS least-squares on the pair network with optional BPR regressor.
        Subsamples at stride, solves at grid points, interpolates back.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional correlation weight.
        baseline : str or None
            Variable name for BPR regressor (default 'BPR'). None to skip.
        stride : int
            Subsample step. Default 1.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        BatchComplex
            Network-consistent per-pair trend (BPR component removed).
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from . import utils_detrend

        BatchCore._require_lazy(self, 'lstsq_baseline')

        result = {}
        for key, ds in self.items():
            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                result[key] = ds
                continue

            ref_values = ds.coords['ref'].values.astype('datetime64[ns]').astype(np.int64)
            rep_values = ds.coords['rep'].values.astype('datetime64[ns]').astype(np.int64)
            has_bpr = baseline is not None and baseline in ds.coords
            bpr_values = ds.coords[baseline].values.astype(np.float64) if has_bpr else None

            # Compute unique dates (same logic as lstsq_baseline_array)
            ns_per_day = 86400 * 1e9
            ref_days = ref_values.astype(np.float64) / ns_per_day
            rep_days = rep_values.astype(np.float64) / ns_per_day
            unique_days = np.unique(np.concatenate([ref_days, rep_days]))
            n_dates = len(unique_days)
            # Map unique_days back to datetime64
            date_coords = (unique_days * ns_per_day).astype('datetime64[ns]')

            weight_ds = weight[key] if weight is not None else None

            result_ds = {}
            for pol in pols:
                data_da = ds[pol]
                data_dask = data_da.data

                weight_da = weight_ds[pol] if weight_ds is not None else None

                def _block(data_block, weight_block=None,
                           _ref=ref_values, _rep=rep_values,
                           _bpr=bpr_values, _stride=stride):
                    w = [weight_block] if weight_block is not None else None
                    return utils_detrend.lstsq_baseline_array(
                        [data_block], w, _ref, _rep,
                        bpr_values=_bpr, stride=_stride,
                    )

                if weight_da is not None:
                    weight_dask = weight_da.data
                    result_dask = da.blockwise(
                        _block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_dates},
                        concatenate=True,
                        dtype=np.complex64,
                        meta=np.empty((0, 0, 0), dtype=np.complex64),
                    )
                else:
                    result_dask = da.blockwise(
                        _block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_dates},
                        concatenate=True,
                        dtype=np.complex64,
                        meta=np.empty((0, 0, 0), dtype=np.complex64),
                    )

                # Per-date output with date coordinates
                result_ds[pol] = xr.DataArray(
                    result_dask,
                    dims=['date', 'y', 'x'],
                    coords={'date': date_coords,
                            'y': data_da.coords['y'],
                            'x': data_da.coords['x']}
                )

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        return BatchComplex(result)

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

    def unwrap2d(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d(*args, **kwargs)

    def unwrap2d_irls(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d_irls(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot complex phase as wrapped phase via .angle() conversion.
        """
        return self.angle().plot(*args, **kwargs
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
            import torch
            result = goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, dev, threshold=threshold)
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

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
        elif isinstance(window, (tuple, list)):
            window = {'y': window[0], 'x': window[1]}

        # Resolve device ONCE here, not in every task
        if device == 'auto':
            resolved_device = BatchCore._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

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

                    depth_2d = {0: depth_y, 1: depth_x}
                    depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                    filtered_dask = da.map_overlap(
                        _apply_goldstein_2d_for_dask,
                        phase_dask,
                        corr_dask,
                        depth= depth_3d if var_data.ndim == 3 else depth_2d,
                        boundary='none',
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


def _subtract_date_from_pair(first, second):
    """Subtract per-date atmospheric screens from per-pair data.

    Uses BatchComplex.pairs() to select ref/rep screens,
    then: result = data * conj(screen_ref) * screen_rep
    """
    import numpy as np

    key0 = list(first.keys())[0]
    ref_dates = first[key0].coords['ref'].values
    rep_dates = first[key0].coords['rep'].values
    pairs = np.column_stack([ref_dates, rep_dates])

    screen_ref, screen_rep = second.pairs(pairs)
    return first * screen_ref.conj() * screen_rep


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

    @staticmethod
    def _preserve_nonspatial(source, target):
        """Copy non-spatial variables (e.g. BPR) from source to target batch."""
        import dask.array as da
        for key in source:
            src_ds = source[key]
            tgt_ds = target[key]
            extra = {}
            for v in src_ds.data_vars:
                if v not in tgt_ds.data_vars:
                    var = src_ds[v]
                    if not isinstance(var.data, da.Array):
                        var = var.chunk()
                    extra[v] = var
            if extra:
                target[key] = tgt_ds.assign(extra)
        return target

    def phase(self) -> 'BatchComplex | BatchWrap | Batch | None':
        """Extract phase from Batches.

        Returns the first BatchComplex, or first BatchWrap,
        or first Batch found. Returns None if none found.
        """
        for b in self:
            if isinstance(b, BatchComplex):
                return b
        for b in self:
            if isinstance(b, BatchWrap):
                return b
        for b in self:
            if isinstance(b, Batch) and not isinstance(b, BatchUnit):
                return b
        return None

    def correlation(self) -> 'BatchUnit | None':
        """Extract correlation weights from Batches.

        Returns the first BatchUnit found, or None.
        """
        for b in self:
            if isinstance(b, BatchUnit):
                return b
        return None

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None,
                 debug: bool = False, **kwargs):
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
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       caption=caption or 'Opening...',
                                       debug=debug)
        else:
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       caption=caption or 'Snapshotting...',
                                       debug=debug, wrapper=Batches)

        if isinstance(result, Batches):
            return result
        return Batches((result,))  # fallback for stores without __wrapper__

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                debug: bool = False):
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
            result = self.snapshot(store=zip_store, caption=caption or 'Opening archive...', debug=debug)
            zip_store.close()
            return result
        else:
            # Save mode - write to temp directory, then zip
            # This avoids ZipStore's duplicate entry problem
            temp_dir = tempfile.mkdtemp()
            try:
                result = self.snapshot(store=temp_dir, caption=caption or 'Archiving...', debug=debug)
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

    def chunk(self, *args, **kwargs):
        """Apply chunk to all batches."""
        return Batches([b.chunk(*args, **kwargs) for b in self])

    def chunk2d(self, *args, **kwargs):
        """Apply chunk2d to all batches."""
        return Batches([b.chunk2d(*args, **kwargs) for b in self])

    def chunk1d(self, *args, **kwargs):
        """Apply chunk1d to all batches."""
        return Batches([b.chunk1d(*args, **kwargs) for b in self])

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

    def filter(self, days=None, meters=None, date=None, pair=None, count=None,
               min_connections=2, cleanup=True):
        """Filter pairs in the baseline network.

        Selects pairs matching the given temporal/spatial criteria. By default,
        removes degraded dates (hanging or single-side connected).

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline in meters. If None, no limit.
        date : str or list, optional
            Date(s) to exclude. Accepts a single date string or a list,
            any format parseable by ``pd.to_datetime``.
        pair : str or list, optional
            Pair(s) to exclude. Each pair is a string ``'YYYY-MM-DD YYYY-MM-DD'``.
            Accepts a single pair string or a list.
        count : int, optional
            Remove dates with fewer than this many connections.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        cleanup : bool, optional
            If True (default), iteratively remove hanging dates and dates
            connected only to predecessors or only to successors.
            Set to False to keep the raw network for testing.

        Returns
        -------
        Batches
            Filtered Batches with valid pairs only.

        Examples
        --------
        >>> intfcorr.filter(days=100, meters=80).detrend1d().unwrap1d().lstsq()
        >>> intfcorr.filter(date='2024-12-30').detrend1d().unwrap1d().lstsq()
        >>> intfcorr.filter(date=['2024-12-30', '2024-06-21'])
        >>> intfcorr.filter(pair='2024-06-21 2024-12-30')
        >>> intfcorr.filter(count=3)  # remove dates with < 3 connections
        >>> intfcorr.filter(days=100, meters=80, cleanup=False)  # raw network
        """
        import numpy as np
        import pandas as pd

        if days is None and meters is None and date is None and pair is None and count is None:
            return self

        # Get pair coordinates from the first batch element
        first_batch = self[0]
        first_key = next(iter(first_batch.keys()))
        ds = first_batch[first_key]
        ref = pd.DatetimeIndex(ds.coords['ref'].values).normalize()
        rep = pd.DatetimeIndex(ds.coords['rep'].values).normalize()
        bpr = ds.coords['BPR'].values
        n_pairs = len(ref)

        # Build mask of valid pairs
        mask = np.ones(n_pairs, dtype=bool)

        if days is not None:
            duration = (rep - ref).days
            mask &= duration <= days

        if meters is not None:
            mask &= np.abs(bpr) <= meters

        if date is not None:
            if isinstance(date, str):
                date = [date]
            exclude_dates = pd.to_datetime(date).normalize()
            mask &= ~ref.isin(exclude_dates) & ~rep.isin(exclude_dates)

        if pair is not None:
            if isinstance(pair, str):
                pair = [pair]
            exclude_pairs = set()
            for p in pair:
                parts = str(p).split()
                r, s = pd.Timestamp(parts[0]).normalize(), pd.Timestamp(parts[1]).normalize()
                exclude_pairs.add((r, s))
            for i in range(n_pairs):
                if (ref[i], rep[i]) in exclude_pairs:
                    mask[i] = False

        # Build DataFrame for pruning
        df = pd.DataFrame({'ref': ref[mask], 'rep': rep[mask],
                           'idx': np.where(mask)[0]})

        if len(df) > 0:
            from .Baseline import _cleanup_network
            min_conn = max(min_connections, count) if count is not None else min_connections
            if cleanup:
                df = _cleanup_network(df, min_connections=min_conn)
            elif count is not None:
                counts = pd.concat([df['ref'], df['rep']]).value_counts()
                low_dates = set(counts[counts < count].index)
                df = df[~df['ref'].isin(low_dates) & ~df['rep'].isin(low_dates)]

        if len(df) == 0:
            raise ValueError("No valid pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")

        valid_idx = df['idx'].values
        return self.isel(pair=valid_idx)

    def coherent(self, threshold=0.5):
        """Mask low-coherence pixels using mean correlation.

        Computes mean correlation across pairs from the BatchUnit item
        and sets pixels with mean correlation below threshold to NaN
        in all batch items.

        Parameters
        ----------
        threshold : float
            Minimum mean correlation to keep. Default 0.5.

        Returns
        -------
        Batches
            Same structure with NaN where mean correlation < threshold.
        """
        corr = next((b for b in self if isinstance(b, BatchUnit)), None)
        if corr is None:
            raise ValueError('coherent() requires a BatchUnit (correlation) in Batches')
        results = []
        for b in self:
            out = {}
            for key in b:
                corr_ds = corr[key]
                corr_var = next(v for v in corr_ds.data_vars if 'y' in corr_ds[v].dims)
                corr_da = corr_ds[corr_var]
                mask = corr_da.mean('pair') >= threshold if 'pair' in corr_da.dims else corr_da >= threshold
                out[key] = b[key].where(mask)
            results.append(type(b)(out))
        return Batches(results)

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

        intf = ref * rep.conj()
        if phase is not None:
            if isinstance(phase, BatchComplex):
                intf = intf * phase
            else:
                intf = intf * phase.iexp(-1)

        if wavelength is not None:
            intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            del ref, rep
            corr_look = (intf_look.abs() / (intensity_ref * intensity_rep).sqrt()).clip(0, 1)
            del intensity_ref, intensity_rep
        else:
            intf_look = intf
            corr_look = None
            del ref, rep
        del intf

        if weight is not None:
            intf_look = intf_look.where(weight.isfinite())
            corr_look = corr_look.where(weight.isfinite()) if corr_look else None

        if corr_look is None:
            return Batches([intf_look])
        return Batches([intf_look, corr_look])

    def interferogram2(self, *args, **kwargs):
        """
        Compute optimized interferogram using dual-polarization coherence optimization.

        This method requires the insardev_polsar extension package.
        """
        raise ImportError(
            "interferogram2() requires insardev_polsar extension"
        )

    def compute(self):
        """Compute all batches at once via dask.persist().

        Persists all bursts across all batches in a single scheduler
        submission. Data stays in worker memory. Preserves shared computation
        between dependent batches (e.g., phase and correlation). For
        memory-constrained sequential processing, use snapshot().

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        import numpy as np
        from insardev_toolkit.progressbar import progressbar

        # Get all burst keys (should be same across all batches)
        keys = list(self[0].keys())
        n_batches = len(self)

        # Save input chunk structure per batch per burst
        all_input_chunks = []  # list of {burst_key: {var_name: chunks_dict}}
        for batch in self:
            batch_chunks = {}
            for key, ds in batch.items():
                ic = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if hasattr(arr.data, 'chunks'):
                        ic[var_name] = dict(zip(arr.dims, arr.data.chunks))
                batch_chunks[key] = ic
            all_input_chunks.append(batch_chunks)

        # Persist all batches at once — single scheduler submission
        # progressbar extracts futures and blocks until completion
        all_dicts = [dict(batch) for batch in self]
        all_results = list(dask.persist(*all_dicts))
        progressbar(all_results, desc='Computing bursts'.ljust(25))

        # Finalize: materialize coordinates and rechunk to match input
        computed_batches = []
        for bi in range(n_batches):
            result = all_results[bi]
            computed = {}
            for key, ds in result.items():
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)
                input_chunks = all_input_chunks[bi][key]
                rechunked_vars = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if var_name in input_chunks:
                        chunks = input_chunks[var_name]
                        if isinstance(arr.data, np.ndarray):
                            arr = arr.chunk(chunks)
                        elif hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks:
                            arr = arr.chunk(chunks)
                        rechunked_vars[var_name] = arr
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                computed[key] = ds
            computed_batches.append(type(self[bi])(computed))
        return Batches(computed_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, device='auto', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

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

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, device=device,
                              debug=debug, **kwargs)

    def unwrap2d_chunk(self, overlap=None, device='auto', debug=False, **kwargs):
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Unlike unwrap2d() which requires a single spatial chunk, this method
        unwraps each spatial chunk independently with overlap margins.

        Parameters
        ----------
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batches
            Batches with [unwrapped_phase, weight] preserving original types.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline).interferogram(wavelength=30).angle()
        >>> unwrapped, corr = phase.chunk2d('128MiB').unwrap2d_chunk()
        """
        if len(self) < 1:
            raise ValueError("unwrap2d_chunk() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        unwrapped = phase.unwrap2d_chunk(weight=weight, overlap=overlap,
                                          device=device, debug=debug, **kwargs)

        elements = [unwrapped] + list(self[1:])
        return Batches(elements)

    def unwrap1d(self, debug=False, **kwargs):
        """
        1D temporal phase unwrapping using IRLS optimization.

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Parameters
        ----------
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, epsilon.

        Returns
        -------
        Batches
            Batches with [unwrapped Batch, weight] preserving original elements.

        Examples
        --------
        >>> # With explicit .angle()
        >>> phase, corr = mintfcorr.detrend1d().angle()
        >>> unwrapped, corr = Batches([phase, corr]).unwrap1d()
        >>> # Or directly from complex phase (angle applied automatically)
        >>> unwrapped, corr = mintfcorr.detrend1d().unwrap1d()
        """
        if len(self) < 1:
            raise ValueError("unwrap1d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap1d
        unwrapped = phase.unwrap1d(weight=weight, debug=debug, **kwargs)

        # Preserve non-spatial variables (e.g. BPR) that may be dropped by unwrapping
        unwrapped = Batches._preserve_nonspatial(phase, unwrapped)

        # Rebuild Batches preserving all original elements except first
        elements = [unwrapped] + list(self[1:])
        return Batches(elements)

    def trend2d(self, transform=None, degree=1, window=None, stride=1, device='auto', debug=False):
        """
        Compute 2D spatial trend and append it to Batches.

        Appends the per-pair trend as a new BatchComplex element, preserving
        original data unchanged. Use with lstsq_baseline() + subtract() for
        network-consistent detrending.

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi', 'rng'.
        degree : int
            Polynomial degree (1=plane). Default 1.
        window : int, tuple, or None
            Window size in pixels. None = global fit.
        stride : int
            Subsample step for windowed fit. Default 1.
        device : str
            PyTorch device.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Original Batches with appended trend BatchComplex.

        Examples
        --------
        >>> # Append trend for later network-consistent detrending
        >>> intfcorr2d = intfcorr.trend2d(transform, window=(500,2000), stride=10)
        >>> # intfcorr2d = [intfs, corr, trend]  or  [intfs, trend]
        """
        if len(self) < 1:
            raise ValueError("trend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(f"First element must be Batch or BatchComplex, got {type(phase).__name__}")

        if window is None:
            trend = phase.trend2d(transform, weight=weight, degree=degree,
                                  device=device, detrend=False, debug=debug)
        else:
            if degree != 1:
                raise ValueError("Windowed trend2d only supports degree=1.")
            if transform is not None:
                n_vars = len([v for v in transform[list(transform.keys())[0]].data_vars
                              if 'y' in transform[list(transform.keys())[0]][v].dims])
                if not (1 <= n_vars <= 3):
                    raise ValueError(f"Windowed trend2d requires 1-3 transform variables, got {n_vars}.")
            trend = phase.trend2d_window(transform, weight=weight,
                                         window=window, stride=stride,
                                         detrend=False, debug=debug)

        # Preserve non-spatial variables (e.g. BPR, ref, rep)
        trend = Batches._preserve_nonspatial(phase, trend)

        # Append trend to Batches
        elements = list(self) + [trend]
        return Batches(elements)

    def detrend2d(self, transform=None, degree=1, window=None, stride=1, device='auto', debug=False):
        """
        Detrend 2D polynomial trend and return Batches with detrended data.

        Two modes:
        - window=None (default): global polynomial fit across full spatial extent.
        - window=N or window=(Ny, Nx): local windowed fit with 4 half-overlapping
          grids averaged per pixel. Extent-independent. Window size in pixels.

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi', 'rng', 'ele'.
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        window : int, tuple, or None
            Window size in pixels. None = global fit. int = square window.
            tuple (win_y, win_x) = rectangular window.
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
        >>> # Global detrend (default)
        >>> intf = stack.pairs(bl).interferogram(wl=30).detrend2d(transform)
        >>> # Local windowed detrend (extent-independent)
        >>> intf = stack.pairs(bl).interferogram(wl=30).detrend2d(transform, window=250)
        """
        if len(self) < 1:
            raise ValueError("detrend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(f"First element must be Batch or BatchComplex, got {type(phase).__name__}")

        if window is None:
            # Global polynomial fit (Pattern D: three-phase)
            detrended = phase.trend2d(transform, weight=weight, degree=degree,
                                      device=device, detrend=True, debug=debug)
        else:
            if degree != 1:
                raise ValueError("Windowed detrend2d only supports degree=1. "
                                 "Use window=None for higher-degree global fit.")
            if transform is not None:
                n_vars = len([v for v in transform[list(transform.keys())[0]].data_vars
                              if 'y' in transform[list(transform.keys())[0]][v].dims])
                if not (1 <= n_vars <= 3):
                    raise ValueError(f"Windowed detrend2d requires 1-3 transform variables "
                                     f"(e.g. ele, azi+rng, azi+rng+ele), got {n_vars}. "
                                     f"Use window=None for global fit with more variables.")
            # Local sliding window fit
            detrended = phase.trend2d_window(transform, weight=weight,
                                              window=window, stride=stride,
                                              detrend=True, debug=debug)

        # Preserve non-spatial variables (e.g. BPR) that may be dropped by arithmetic
        detrended = Batches._preserve_nonspatial(phase, detrended)

        # Rebuild Batches preserving all original elements except first
        elements = [detrended] + list(self[1:])
        return Batches(elements)

    def lstsq_baseline(self, baseline='BPR', stride=1, batch=-1, debug=False):
        """
        Make per-pair trend network-consistent via SBAS decomposition.

        Decomposes the specified BatchComplex (trend from trend2d) into per-date
        components via IRLS least-squares on the pair network. Optionally
        separates BPR-correlated component (DEM leak). Replaces the target
        BatchComplex with the network-consistent reconstruction.

        Parameters
        ----------
        baseline : str or None
            If str (default 'BPR'), include BPR as regressor to separate
            DEM contamination. If None, decompose without BPR.
        stride : int
            Subsample step — process at grid points, interpolate back.
            Match with trend2d stride for efficiency.
        batch : int
            Index of batch element to process. Default -1 (last).
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Same structure with target element replaced by consistent trend.

        Examples
        --------
        >>> intfcorr = intfs.trend2d(transform, window=(500,2000), stride=10)
        >>> intfcorr = intfcorr.lstsq_baseline(stride=10)
        >>> intfcorr = intfcorr.subtract()  # apply consistent trend
        """
        if len(self) < 2:
            raise ValueError("lstsq_baseline requires at least 2 elements (data + trend from trend2d)")
        trend_idx = batch if batch >= 0 else len(self) + batch
        trend_batch = self[trend_idx]

        if not isinstance(trend_batch, BatchComplex):
            raise TypeError(f"lstsq_baseline target must be BatchComplex, got {type(trend_batch).__name__}")

        # Find weight (BatchUnit) if present
        weight = None
        for el in self:
            if isinstance(el, BatchUnit):
                weight = el
                break

        consistent = trend_batch.lstsq_baseline(
            weight=weight, baseline=baseline, stride=stride, debug=debug
        )

        elements = list(self)
        elements[trend_idx] = consistent
        return Batches(elements)

    def subtract(self):
        """
        Subtract the next same-type batch from the first batch.

        Finds the first batch element, then the next element of the same type,
        subtracts the second from the first, replaces the first with the result,
        and drops the second.

        Type-specific subtraction:
        - BatchComplex: first * conj(second) (phase subtraction on unit circle)
        - Batch: first - second (real subtraction)
        - BatchWrap: wrap(first - second) (wrapped phase subtraction)
        - BatchUnit: not supported (raises error)

        Returns
        -------
        Batches
            With first element replaced by subtracted result, second dropped.

        Examples
        --------
        >>> intfcorr = intfs.trend2d(transform, ...).lstsq_baseline(stride=10).subtract()
        """
        if len(self) < 2:
            raise ValueError("subtract() requires at least 2 elements")

        first_type = type(self[0])
        if isinstance(self[0], BatchUnit):
            raise TypeError("subtract() cannot be applied to BatchUnit")

        # Find next element of the same type
        second_idx = None
        for i in range(1, len(self)):
            if type(self[i]) is first_type:
                second_idx = i
                break
        if second_idx is None:
            raise ValueError(f"subtract() requires a second {first_type.__name__} element")

        first = self[0]
        second = self[second_idx]

        # Check if second is per-date (from lstsq_baseline) and first is per-pair
        is_date_to_pair = False
        for key in first.keys():
            first_ds = first[key]
            second_ds = second[key]
            first_pol = [v for v in first_ds.data_vars if 'y' in first_ds[v].dims][0]
            second_pol = [v for v in second_ds.data_vars if 'y' in second_ds[v].dims][0]
            if 'pair' in first_ds[first_pol].dims and 'date' in second_ds[second_pol].dims:
                is_date_to_pair = True
            break

        if is_date_to_pair:
            result = _subtract_date_from_pair(first, second)
        elif isinstance(first, BatchComplex):
            result = first * second.conj()
        else:
            result = first - second

        result = Batches._preserve_nonspatial(first, result)

        elements = list(self)
        elements[0] = result
        elements.pop(second_idx)
        return Batches(elements)

    def detrend1d_baseline(self, baseline='BPR', intercept=False, slope=True,
                           bins=128, debug=False):
        """
        Detrend linear trend along baseline variable (default BPR) and return Batches.

        Removes phase proportional to perpendicular baseline at each pixel.
        Uses numba per-pixel periodogram init + IRLS with analytical 2x2 solve.

        Parameters
        ----------
        baseline : str
            Variable name to regress against (default 'BPR' for perpendicular baseline).
        intercept : bool
            If True, include intercept in the trend to subtract.
            If False (default), zero out the intercept (preserve it in data).
        slope : bool
            If True (default), include slope in the trend to subtract.
            If False, zero out the slope (preserve it in data).
        bins : int
            Periodogram bins for slope initialization (default 128).
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Batches with detrended first element, preserving other elements.

        Examples
        --------
        >>> intfcorr.detrend1d_baseline()          # remove BPR slope (DEM residual)
        >>> intfcorr.detrend1d_baseline(bins=256)   # larger search range for big DEM errors
        """
        if len(self) < 1:
            raise ValueError("detrend1d_baseline() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(f"detrend1d_baseline() requires Batch or BatchComplex, got {type(phase).__name__}.")

        detrended = phase.trend1d_baseline(weight=weight, baseline=baseline,
                                           detrend=True,
                                           intercept=intercept, slope=slope,
                                           bins=bins, debug=debug)

        detrended = Batches._preserve_nonspatial(phase, detrended)

        elements = [detrended] + list(self[1:])
        return Batches(elements)

    def detrend1d(self, *args, **kwargs):
        """Alias for detrend1d_baseline(). Use detrend1d_baseline() directly for clarity."""
        print("NOTE: detrend1d() is an alias. Use detrend1d_baseline() directly.")
        return self.detrend1d_baseline(*args, **kwargs)

    def lstsq(self, cumsum=True,
              max_iter=5, epsilon=0.1, x_tol=0.001, debug=False):
        """
        L1-norm IRLS network inversion to date-based time series.

        Takes unwrapped pair phases and inverts the network to get per-date
        accumulated phase. Uses IRLS with L1-norm for robustness against
        outlier pairs. Expects Batches with [Batch (unwrapped phase), BatchUnit (weight, optional)].

        Parameters
        ----------
        cumsum : bool
            If True (default), return cumulative displacement time series.
            If False, return incremental phase changes between dates.
        max_iter : int
            Maximum IRLS iterations. Default 5.
        epsilon : float
            IRLS regularization parameter. Default 0.1.
        x_tol : float
            Solution convergence tolerance. Default 0.001.
        debug : bool
            Print debug information.

        Returns
        -------
        Batches
            Batches with [displacement Batch] preserving original elements except first.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped, corr = Batches([phase, corr]).unwrap1d()
        >>> disp, corr = Batches([unwrapped, corr]).lstsq()
        """
        if len(self) < 1:
            raise ValueError("lstsq() requires Batches with at least 1 element: [data]")

        data = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Delegate to Batch.lstsq
        result = data.lstsq(weight=weight, cumsum=cumsum,
                            max_iter=max_iter, epsilon=epsilon,
                            x_tol=x_tol, debug=debug)

        # Rebuild Batches preserving all original elements except first
        elements = [result] + list(self[1:])
        return Batches(elements)

    def displacement_los(self, transform):
        """
        Convert phase to line-of-sight displacement (meters).

        Applies Batch.displacement_los() to the first element.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch or Stack providing radar_wavelength.

        Returns
        -------
        Batches
            Batches with [displacement Batch] preserving other elements.

        Examples
        --------
        >>> disp, corr = mintfcorr.detrend1d().unwrap1d().lstsq().displacement_los(stack.transform())
        """
        data = self[0]
        result = data.displacement_los(transform)
        elements = [result] + list(self[1:])
        return Batches(elements)

    def regression1d_baseline(self, *args, **kwargs):
        raise NotImplementedError("Batches.regression1d_baseline() is removed. Use Batches.detrend1d() or Batch.trend1d() instead.")

    def detrend1d_pairs(self, max_refine=3, debug=False):
        """
        Detrend 1D linear trend along temporal pairs and return Batches.

        For each date, fits a linear model (intercept + slope) to phase vs
        temporal baseline across all pairs sharing that date, then subtracts
        the fit. This removes constant atmospheric delay (intercept at zero
        temporal baseline).

        Iterative refinement (max_refine > 0): re-estimates atmospheric phase
        after removing current pair-wise model from the data. Each pass
        reduces cross-date bias by ~sqrt(N_dates).

        Requires complex input (BatchComplex): unit-circle fitting, detrend
        multiplicatively (phase * trend.conj()). Must be placed before unwrapping.
        Incoherent pixels (circular std > π/2) are automatically NaN'd.

        Parameters
        ----------
        max_refine : int
            Maximum refinement iterations (0 = single-pass). Default 3.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Batches with [detrended_phase, weight] preserving original types.

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline).interferogram(wavelength=30).detrend1d_pairs()
        """
        if len(self) < 1:
            raise ValueError("detrend1d_pairs() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(
                f"detrend1d_pairs() requires Batch or BatchComplex, "
                f"got {type(phase).__name__}."
            )

        # Fuse fit+subtract into one blockwise call (detrend=True) so the input
        # phase is referenced only once in the dask graph.
        detrended = phase.trend1d_pairs(weight=weight,
                                        detrend=True,
                                        max_refine=max_refine,
                                        debug=debug)

        # Preserve non-spatial variables (e.g. BPR) that may be dropped by arithmetic
        detrended = Batches._preserve_nonspatial(phase, detrended)

        # Rebuild Batches preserving all original elements except first
        elements = [detrended] + list(self[1:])
        return Batches(elements)

    def threshold(self, threshold=np.pi/2):
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Pixels with cstd >= threshold are set to NaN. Uses correlation
        weights from the second element if available.

        Parameters
        ----------
        threshold : float
            Maximum cstd in radians. Default π/2.

        Returns
        -------
        Batches
            Batches with filtered phase, preserving other elements.
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"threshold() requires BatchComplex, got {type(phase).__name__}")

        filtered = phase.threshold(weight=weight, threshold=threshold)
        elements = [filtered] + list(self[1:])
        return Batches(elements)

    def velocity(self, max_refine=3, **kwargs):
        """
        Estimate velocity from pair network (BatchComplex) or time series (Batch).

        For BatchComplex: uses periodogram on pairs — fast shortcut for global
        velocity in rad/year. Use displacement_los(transform) to convert to m/year.
        For Batch: uses linear regression on time series (existing method).

        Parameters
        ----------
        max_refine : int
            For BatchComplex: refinement levels (0=coarse ~32mm/yr, 3=fine ~0.5mm/yr).

        Returns
        -------
        Batches
            For BatchComplex: Batches[velocity, rmse] — both in rad/year.
            For Batch: Batches[velocity, intercept].
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if isinstance(phase, BatchComplex):
            return phase.velocity(weight=weight, max_refine=max_refine, **kwargs)
        else:
            return phase.velocity(**kwargs)

    def rmse(self, solution):
        """RMSE of phase vs solution, using correlation weight if present.

        Extracts phase from self[0] and optional weight from self[1] (BatchUnit).
        Weight is automatically passed to the RMSE calculation and reduced
        to (y, x) via mean over the temporal dimension.

        Parameters
        ----------
        solution : Batch
            Velocity (y, x), pair-based, or date-based solution.

        Returns
        -------
        Batches
            [RMSE Batch (y, x), mean weight BatchUnit (y, x)] when weight present,
            [RMSE Batch (y, x)] otherwise.
        """
        if len(self) < 1:
            raise ValueError("rmse() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        rmse_result = phase.rmse(solution, weight=weight)

        if weight is not None:
            # Reduce weight to (y, x) — detect temporal dimension
            w_sample_ds = next(iter(weight.values()))
            w_spatial = [v for v in w_sample_ds.data_vars if 'y' in w_sample_ds[v].dims]
            tdim = next((d for d in ('pair', 'date')
                         if w_spatial and d in w_sample_ds[w_spatial[0]].dims), None)
            reduced_weight = weight.mean(tdim) if tdim else weight
            elements = [rmse_result, reduced_weight]
        else:
            elements = [rmse_result]

        return Batches(elements)

    def regression1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("Batches.regression1d_pairs() is removed. Use Batches.detrend1d_pairs() instead.")

    def trend1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("Batches.trend1d_pairs() is removed. Use Batches.detrend1d_pairs() instead.")

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
