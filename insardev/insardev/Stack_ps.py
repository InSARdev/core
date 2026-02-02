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
from .Stack_stl import Stack_stl
from insardev_toolkit import progressbar

class Stack_ps(Stack_stl):

    def compute_ps(self, *args, **kwargs):
        """
        Deprecated. Use psfunction() instead which computes PS function directly using PyTorch.

        Example:
            psf = stack.psfunction()
            sintf, scorr = stack.phasediff_multilook(pairs, wavelength=30, weight=psf)
        """
        raise NotImplementedError(
            "compute_ps() is deprecated. Use psfunction() instead:\n"
            "    psf = stack.psfunction()\n"
            "    sintf, scorr = stack.phasediff_multilook(pairs, wavelength=30, weight=psf)"
        )

    @staticmethod
    def _psfunction_torch(amplitudes, device='auto', debug=False):
        """
        Compute PS function using PyTorch for GPU acceleration.

        Parameters
        ----------
        amplitudes : np.ndarray
            3D array of shape (n_dates, height, width) containing amplitude values.
        device : str
            Device to use ('auto', 'mps', 'cuda', 'cpu').
        debug : bool
            Print debug information.

        Returns
        -------
        np.ndarray
            2D array of shape (height, width) containing PS function values.
        """
        import torch
        import numpy as np

        # Select device using shared helper
        dev = Stack_ps._get_torch_device(device)

        if debug:
            print(f'DEBUG: _psfunction_torch using device={dev}')

        n_dates, height, width = amplitudes.shape

        # Convert to intensity (|z|^2) and move to device
        intensity = torch.from_numpy(amplitudes.astype(np.float32) ** 2).to(dev)

        # Compute mean intensity per date for normalization
        # Shape: (n_dates,)
        mean_intensity_per_date = intensity.nanmean(dim=(1, 2))

        # Global mean across all dates
        global_mean = mean_intensity_per_date.nanmean()

        # Normalize each date: intensity * (global_mean / date_mean)
        # Reshape for broadcasting: (n_dates, 1, 1)
        norm_factor = global_mean / mean_intensity_per_date
        norm_factor = norm_factor.view(n_dates, 1, 1)
        intensity_norm = intensity * norm_factor

        # Compute mean and std across date dimension
        # Use masked operations to handle NaN
        mean_amp = intensity_norm.nanmean(dim=0)  # (height, width)

        # PyTorch nanstd - compute manually since not built-in
        # std = sqrt(E[(x - mean)^2])
        diff = intensity_norm - mean_amp.unsqueeze(0)
        # Count non-NaN values
        valid_mask = ~torch.isnan(intensity_norm)
        n_valid = valid_mask.sum(dim=0).float()
        n_valid = torch.clamp(n_valid, min=1)  # Avoid division by zero

        # Variance with Bessel's correction (n-1)
        variance = torch.where(
            valid_mask, diff ** 2, torch.tensor(0.0, device=dev)
        ).sum(dim=0) / (n_valid - 1).clamp(min=1)
        std_amp = torch.sqrt(variance)

        # PS function: mean / (2 * std)
        psf = mean_amp / (2 * std_amp)

        # Handle invalid values (inf, nan)
        psf = torch.where(torch.isfinite(psf), psf, torch.tensor(float('nan'), device=dev))

        # Move back to CPU and convert to numpy
        result = psf.cpu().numpy()

        # Cleanup GPU memory
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    def psfunction(self, device='auto', debug=False):
        """
        Compute PS (Persistent Scatterer) function for weighting in single-look processing.

        The PS function identifies stable scatterers by computing the ratio of mean
        amplitude to amplitude standard deviation across the temporal stack:

            psfunction = mean_intensity / (2 * std_intensity)

        Higher values indicate more stable scatterers (consistent backscatter).

        Parameters
        ----------
        device : str
            Device for PyTorch computation ('auto', 'mps', 'cuda', 'cpu').
        debug : bool
            Print debug information.

        Returns
        -------
        BatchUnit
            Per-burst PS function values for use as weight in phasediff_multilook() (lazy).

        Examples
        --------
        # Use PS function as weight for single-look interferograms
        psf = stack.psfunction()
        sintf, scorr = stack.phasediff_multilook(pairs, wavelength=30, weight=psf)
        """
        import dask
        import dask.array
        import numpy as np
        import torch
        import xarray as xr
        from .Batch import BatchUnit

        # Auto-detect device based on Dask cluster resources and hardware
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack_ps._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        if debug:
            print(f"DEBUG: psfunction using device={device}")

        results = {}
        for key, ds in self.items():
            # Get complex SLC data variable (usually 'VV' or 'VH')
            complex_vars = [v for v in ds.data_vars if ds[v].dtype.kind == 'c']
            if not complex_vars:
                raise ValueError(f"No complex data found in burst {key}")

            # Use first complex variable
            var_name = complex_vars[0]
            slc_data = ds[var_name]

            # Ensure data is chunked for lazy processing (chunk in y,x, not date)
            if not isinstance(slc_data.data, dask.array.Array):
                slc_data = slc_data.chunk({'y': 512, 'x': 512})

            if debug:
                print(f'DEBUG: psfunction for {key}: shape={slc_data.shape}, chunks={slc_data.chunks}')

            # Create wrapper that captures device and debug
            def make_wrapper(dev, dbg):
                def process_wrapper(slc_chunk):
                    """Process spatial chunk: (chunk_y, chunk_x, n_dates) -> (chunk_y, chunk_x)

                    Note: input_core_dims=[['date']] moves date to last axis.
                    """
                    from contextlib import nullcontext
                    # Transpose to (n_dates, chunk_y, chunk_x) for _psfunction_torch
                    slc_transposed = np.moveaxis(slc_chunk, -1, 0)
                    # Compute amplitude |z|
                    amplitudes = np.abs(slc_transposed)
                    # Compute PS function using PyTorch
                    psf_values = Stack_ps._psfunction_torch(amplitudes, device=dev, debug=dbg)
                    return psf_values.astype(np.float32)
                return process_wrapper

            wrapper = make_wrapper(device, debug)

            # Rechunk so date is a single chunk (required for core dim reduction)
            slc_data = slc_data.chunk({'date': -1})

            # Use xr.apply_ufunc with dask='parallelized' for lazy execution
            # Core dim is 'date' (reduction), chunked dims are y, x
            # Note: input_core_dims moves 'date' to last axis, wrapper transposes back
            # Use GPU annotation to prevent MPS command buffer conflicts
            # Provide explicit meta to avoid ComplexWarning when dask infers
            # output type from complex input (we intentionally convert to real)
            with dask.annotate(resources={'gpu': 1} if device != 'cpu' else {}):
                psf_da = xr.apply_ufunc(
                    wrapper,
                    slc_data,
                    input_core_dims=[['date']],
                    output_core_dims=[[]],
                    dask='parallelized',
                    dask_gufunc_kwargs={'meta': np.array((), dtype=np.float32)},
                )

            # Assign name to match SLC variable
            psf_da.name = var_name

            results[key] = xr.Dataset({var_name: psf_da})

        return BatchUnit(results)

    # def plot_psfunction(self, data='auto', caption='PS Function', cmap='gray', quantile=None, vmin=None, vmax=None, **kwargs):
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     if isinstance(data, str) and data == 'auto':
    #         data = self.psfunction()
    #     elif 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         data = data.unstack('stack')

    #     if quantile is not None:
    #         assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

    #     if quantile is not None:
    #         vmin, vmax = np.nanquantile(data, quantile)

    #     plt.figure()
    #     data.plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    #     #self.plot_AOI(**kwargs)
    #     #self.plot_POI(**kwargs)
    #     #plt.xlabel('Range')
    #     #plt.ylabel('Azimuth')
    #     plt.title(caption)

#     def get_adi_threshold(self, threshold):
#         """
#         Vectorize Amplitude Dispersion Index (ADI) raster values selected using the specified threshold.
#         """
#         import numpy as np
#         import dask
#         import pandas as pd
#         import geopandas as gpd
# 
#         def adi_block(ys, xs):
#             from scipy.interpolate import griddata
#             # we can calculate more accurate later
#             dy = dx = 10
#             trans_inv_block = trans_inv.sel(y=slice(min(ys)-dy,max(ys)+dy), x=slice(min(xs)-dx,max(xs)+dx))
#             lt_block = trans_inv_block.lt.compute(n_workers=1).data.ravel()
#             ll_block = trans_inv_block.ll.compute(n_workers=1).data.ravel()
#             block_y, block_x = np.meshgrid(trans_inv_block.y.data, trans_inv_block.x.data)
#             points = np.column_stack([block_y.ravel(), block_x.ravel()])
#             # following NetCDF indices 0.5,1.5,...
#             adi_block = adi.sel(y=slice(min(ys),max(ys)+1), x=slice(min(xs),max(xs)+1))
#             adi_block_value = adi_block.compute(n_workers=1).data.ravel()
#             adi_block_mask = adi_block_value<=threshold
#             adi_block_value = adi_block_value[adi_block_mask]
#             adi_block_y, adi_block_x = np.meshgrid(adi_block.y, adi_block.x)
#             adi_block_y = adi_block_y.ravel()[adi_block_mask]
#             adi_block_x = adi_block_x.ravel()[adi_block_mask]
#             # interpolate geographic coordinates, coarsen=2 grid is required for the best accuracy
#             grid_lt = griddata(points, lt_block, (adi_block_y, adi_block_x), method='linear').astype(np.float32)
#             grid_ll = griddata(points, ll_block, (adi_block_y, adi_block_x), method='linear').astype(np.float32)
#             # return geographic coordinates and values
#             return np.column_stack([grid_lt, grid_ll, adi_block_value])
#     
#         # data grid and transform table
#         adi = self.get_adi()
#         trans_inv = self.get_trans_inv()
#     
#         # split to equal chunks and rest
#         ys_blocks = np.array_split(np.arange(adi.y.size), np.arange(0, adi.y.size, self.chunksize)[1:])
#         xs_blocks = np.array_split(np.arange(adi.x.size), np.arange(0, adi.x.size, self.chunksize)[1:])
#         # arrays size is unknown so we cannot construct dask array
#         blocks = []
#         for ys_block in ys_blocks:
#             for xs_block in xs_blocks:
#                 block = dask.delayed(adi_block)(ys_block, xs_block)
#                 blocks.append(block)
#                 del block
#     
#         # materialize the result as a set of numpy arrays
#         progressbar(model := dask.persist(blocks), desc='Amplitude Dispersion Index (ADI) Threshold')
#         del blocks
#         # the result is already calculated and compute() returns the result immediately
#         model = np.concatenate(dask.compute(model)[0][0])
#         # convert to geopandas object
#         columns = {'adi': model[:,2], 'geometry': gpd.points_from_xy(model[:,1], model[:,0])}
#         df = gpd.GeoDataFrame(columns, crs="EPSG:4326")
#         del columns
#         return df

#     sbas.plot_amplitudes(dates=sbas.df.index[:8], intensity=True, quantile=[0.01, 0.99],
#                         func=lambda data: data.sel(y=slice(920,960), x=slice(4500,4550)),
#                         marker='x', marker_size=200, POI=sbas.geocode(POI))
#     #AOI=sbas.geocode(AOI.buffer(-0.001))
    def plot_amplitudes(self, dates=None, data='auto', norm='auto', func=None, intensity=False,
                       caption='auto', cmap='gray', cols=4, size=4, nbins=5, aspect=1.2, y=1.05,
                       quantile=None, vmin=None, vmax=None, symmetrical=False, **kwargs):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import types

        if isinstance(data, str) and data == 'auto':
            # open SLC data as real amplitudes
            #data = np.abs(self.open_data(dates=dates))
            # magick scale for better plots readability
            scale = np.sqrt(2.5e-07) if intensity else 2.5e-07
            data = np.abs(self.open_data(dates=dates, scale=scale))
            if intensity:
                data = np.square(data)
        elif 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            data = data.unstack('stack')

        if func is not None:
            data = func(data)

        if isinstance(norm, str) and norm == 'auto':
            # normilize SLC grids using average
            stack_average = data.mean(['y','x'])
            norm_multiplier = stack_average.mean(dim='date') / stack_average
            #print ('norm_multiplier', norm_multiplier.values)
            data = norm_multiplier * data
        elif norm is not None:
            data = norm * data

        if isinstance(caption, str) and caption == 'auto':
            caption = 'SLC Intensity' if intensity else 'SLC Amplitude'

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        if quantile is not None:
            vmin, vmax = np.nanquantile(data, quantile)

        # define symmetrical boundaries
        if symmetrical is True and vmax > 0:
            minmax = max(abs(vmin), vmax)
            vmin = -minmax
            vmax =  minmax

        # multi-plots ineffective for linked lazy data
        fg = data.plot.imshow(
            col='date',
            col_wrap=cols, size=size, aspect=aspect,
            vmin=vmin, vmax=vmax, cmap=cmap,
            interpolation='none' # Disable interpolation
        )
        #fg.set_axis_labels('Range', 'Azimuth')
        fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
        fg.fig.suptitle(caption, y=y)

        self.plots_AOI(fg, **kwargs)
        self.plots_POI(fg, **kwargs)
