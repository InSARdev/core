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
from .Stack_export import Stack_export
from insardev_toolkit import progressbar

class Stack_plot(Stack_export):
    import xarray as xr
    import numpy as np
    import pandas as pd
    import matplotlib

    def plot(self, cmap='turbo', alpha=1):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        df = self.to_dataframe().reset_index()
        df['date'] = df['startTime'].dt.date

        # Create group key for orbit direction + path number
        df['orbit_path'] = df.apply(
            lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} [{rec['pathNumber']}]", axis=1
        )

        # Get unique orbit/path combinations and assign colors
        unique_orbit_paths = sorted(df['orbit_path'].unique())
        n = len(unique_orbit_paths)
        colormap = matplotlib.cm.get_cmap(cmap, n)
        color_map = {op: colormap(i) for i, op in enumerate(unique_orbit_paths)}

        fig, ax = plt.subplots()

        # Plot each burst with color based on orbit/path
        for orbit_path, group in df.groupby('orbit_path'):
            group.plot(ax=ax, edgecolor=color_map[orbit_path], facecolor='none', lw=0.25, label=orbit_path)

        # Create consolidated legend labels with date ranges
        legend_labels = []
        for orbit_path in unique_orbit_paths:
            group = df[df['orbit_path'] == orbit_path]
            dates = sorted(group['date'].unique())
            if len(dates) == 1:
                label = f"{orbit_path.split()[0]} {dates[0]} {orbit_path.split()[1]}"
            else:
                label = f"{orbit_path.split()[0]} {dates[0]} - {dates[-1]} {orbit_path.split()[1]}"
            legend_labels.append((orbit_path, label))

        handles = [matplotlib.lines.Line2D([0], [0], color=color_map[op], lw=1, label=label)
                   for op, label in legend_labels]
        ax.legend(handles=handles, loc='upper right')

        col = df.columns[0]
        for _, row in df.drop_duplicates(subset=[col]).iterrows():
            # compute centroid
            x, y = row.geometry.centroid.coords[0]
            ax.annotate(
                str(row[col]),
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                color=color_map[row['orbit_path']],
                path_effects=[patheffects.withStroke(linewidth=0.25, foreground='black')],
                alpha=1
            )

        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('easting [m]')
        ax.set_ylabel('northing [m]')

    # def plot_dataset(self,
    #                  data: xr.Dataset | xr.DataArray,
    #                  polarizations: list[str] | tuple[str] | str | None,
    #                  cmap: matplotlib.colors.Colormap | str | None,
    #                  vmin: float | None,
    #                  vmax: float | None,
    #                  quantile: float | None,
    #                  symmetrical: bool,
    #                  caption: str,
    #                  cols: int,
    #                  rows: int,
    #                  size: float,
    #                  nbins: int,
    #                  aspect: float,
    #                  y: float,
    #                  wrap: bool,
    #                  _size: tuple[int, int] | None,
    #                  ):
    #     import xarray as xr
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     # no data means no plot and no error
    #     if data is None:
    #         return

    #     assert isinstance(data, (dict, list, tuple, xr.Dataset, xr.DataArray)), 'ERROR: data should be a dict or list or tuple or Dataset or DataArray'

    #     # screen size in pixels (width, height) to estimate reasonable number pixels per plot
    #     # this is quite large to prevent aliasing on 600dpi plots without additional processing
    #     if _size is None:
    #         _size = (8000,4000)

    #     def plot_polarization(data, polarization):

    #         if isinstance(data, dict):
    #             data = list(data.values())

    #         if isinstance(data, xr.Dataset):
    #             stackvar = list(data[polarization].dims)[0]
    #             da = data[polarization].isel({stackvar: slice(0, rows)})
    #         else:
    #             stackvar = list(data[0].dims)[0]
    #             das = [da[polarization].isel({stackvar: slice(0, rows)}) for da in data]
    #             da = self.to_dataset(das)
    #             del das

    #         if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
    #             da = da.unstack('stack')
            
    #         # there is no reason to plot huge arrays much larger than screen size for small plots
    #         #print ('screen_size', screen_size)
    #         size_y, size_x = da.shape[1:]
    #         #print ('size_x, size_y', size_x, size_y)
    #         factor_y = int(np.round(size_y / (_size[1] / rows)))
    #         factor_x = int(np.round(size_x / (_size[0] / cols)))
    #         #print ('factor_x, factor_y', factor_x, factor_y)
    #         # decimate for faster plot, do not coarsening without antialiasing
    #         # maybe data is already smoothed and maybe not, decimation is the only safe option
    #         da = da[:,::max(1, factor_y), ::max(1, factor_x)]
    #         # materialize for all the calculations and plotting
    #         progressbar(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

    #         # calculate min, max when needed
    #         if quantile is not None:
    #             _vmin, _vmax = np.nanquantile(da, quantile)
    #         else:
    #             _vmin, _vmax = vmin, vmax
    #         # define symmetrical boundaries
    #         if symmetrical is True and _vmax > 0:
    #             minmax = max(abs(_vmin), _vmax)
    #             _vmin = -minmax
    #             _vmax =  minmax
            
    #         # multi-plots ineffective for linked lazy data
    #         fg = (self.wrap(da) if wrap else da).rename(caption)\
    #             .plot.imshow(
    #             col=stackvar,
    #             col_wrap=min(cols, da[stackvar].size), size=size, aspect=aspect,
    #             vmin=_vmin, vmax=_vmax, cmap=cmap
    #         )
    #         #fg.set_axis_labels('Range', 'Azimuth')
    #         fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
    #         fg.fig.suptitle(f'{polarization} {caption}', y=y)
    #         return fg

    #     if quantile is not None:
    #         assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

    #     if not isinstance(data, (xr.Dataset, xr.DataArray)):
    #         raise ValueError(f'ERROR: invalid data type {type(data)}. Should be xr.Dataset or xr.DataArray')

    #     if isinstance(data, xr.DataArray):
    #         # convert DataArray to Dataset to plot a single polarization
    #         data = data.to_dataset()

    #     if polarizations is None:
    #         polarizations = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data[0].data_vars)
    #     elif isinstance(polarizations, str):
    #         polarizations = [polarizations]
    #     #print ('polarizations', polarizations)

    #     # process polarizations one by one
    #     fgs = []
    #     for pol in polarizations:
    #         fg = plot_polarization(data, polarization=pol)
    #         fgs.append(fg)
    #     return fgs

    def plot_displacement_mm(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [mm]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        data_los_mm = self.los_displacement_mm(data)
        return self.plot_dataset(data_los_mm, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_displacement(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_phase(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_interferogram(self, data, polarizations=None,
                           cmap='gist_rainbow_r',
                           caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        import numpy as np
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=-np.pi, vmax=np.pi, quantile=None, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size)

    def plot_correlation(self, data, polarizations=None,
                         cmap='auto', vmin=0, vmax=1, quantile=None, symmetrical=False,
                         caption='Correlation', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_stack_correlation(self, data, threshold=None, caption='Correlation Stack', bins=100, cmap='auto'):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            data = data.unstack('stack')

        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
    
        data_flatten = data.values.ravel()
    
        fig, axs = plt.subplots(1, 2)
    
        ax2 = axs[0].twinx()
        axs[0].hist(data_flatten, range=(0, 1), bins=bins, density=False, cumulative=False, color='gray', edgecolor='black', alpha=0.5)
        ax2.hist(data_flatten, range=(0, 1), bins=bins, density=False, cumulative=True, color='orange', edgecolor='black', alpha=0.25)
    
        mean_value = np.nanmean(data_flatten)
        axs[0].axvline(mean_value, color='b', label=f'Average {mean_value:0.3f}')
        median_value = np.nanmedian(data_flatten)
        axs[0].axvline(median_value, color='g', label=f'Median {median_value:0.3f}')
        axs[0].set_xlim([0, 1])
        axs[0].grid()
        axs[0].set_xlabel('Correlation')
        axs[0].set_ylabel('Count')
        ax2.set_ylabel('Cumulative Count', color='orange')
    
        axs[0].set_title('Histogram')
        if threshold is not None:
            data.where(data > threshold).plot.imshow(cmap=cmap, vmin=0, vmax=1, ax=axs[1])
            axs[1].set_title(f'Threshold = {threshold:0.3f}')
            axs[0].axvline(threshold, linestyle='dashed', color='black', label=f'Threshold {threshold:0.3f}')
        else:
            data.where(data).plot.imshow(cmap=cmap, vmin=0, vmax=1, ax=axs[1])
        axs[0].legend()
        plt.suptitle(caption)
        plt.tight_layout()
