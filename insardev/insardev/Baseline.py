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
import pandas as pd
import numpy as np


class Baseline(pd.DataFrame):
    """DataFrame subclass for InSAR baseline pairs with custom plotting methods.

    Contains baseline pair information with columns:
        - ref: Reference date
        - rep: Repeat date
        - ref_baseline: Perpendicular baseline at reference date (meters)
        - rep_baseline: Perpendicular baseline at repeat date (meters)
        - pair: String representation of the pair
        - baseline: Baseline difference (rep_baseline - ref_baseline)
        - duration: Temporal separation in days

    Examples
    --------
    >>> baseline = stack.baseline(days=100)
    >>> baseline.plot()  # Plot baseline network
    >>> baseline.hist()  # Plot duration histogram
    """

    # Preserve subclass through pandas operations
    _metadata = ['_burst_id', '_dates']

    def __init__(self, data=None, burst_id=None, dates=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._burst_id = burst_id
        self._dates = dates

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return Baseline(*args, **kwargs).__finalize__(self)
        return _c

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def burst_id(self) -> str | None:
        """Return the burst ID used to generate this baseline table."""
        return self._burst_id

    def tolist(self) -> list[list[int]]:
        """Convert baseline pairs to date indices for use with phasediff_multilook.

        Returns a list of [ref_index, rep_index] pairs that can be passed directly
        to stack.phasediff_multilook() as the first argument.

        Returns
        -------
        list[list[int]]
            List of [ref_index, rep_index] pairs.

        Examples
        --------
        >>> baseline = stack.baseline(days=48)
        >>> pairs = baseline.tolist()
        >>> intf, corr = stack.phasediff_multilook(pairs, wavelength=100)
        """
        if self._dates is None:
            raise ValueError("Dates not available. Baseline must be created from Stack.baseline()")

        if len(self) == 0:
            return []

        # Build date to index mapping
        date_to_idx = {date: idx for idx, date in enumerate(self._dates)}

        # Convert ref/rep dates to indices
        pairs = []
        for _, row in self.iterrows():
            ref_idx = date_to_idx.get(row['ref'])
            rep_idx = date_to_idx.get(row['rep'])
            if ref_idx is not None and rep_idx is not None:
                pairs.append([ref_idx, rep_idx])

        return pairs

    def plot(self, caption: str = 'Baseline', figsize: tuple[float, float] | None = None,
             show_labels: bool = True, ax=None):
        """Plot baseline network diagram.

        Parameters
        ----------
        caption : str, optional
            Plot title. Default is 'Baseline'.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        show_labels : bool, optional
            Whether to show date labels on points. Default is True.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Build caption with burst ID prefix if available
        if self._burst_id:
            caption = f'{self._burst_id} {caption}'

        if len(self) == 0:
            ax.set_title(caption)
            return ax

        # Build unique dates/baselines DataFrame
        df_points = pd.DataFrame(
            np.concatenate([
                self[['ref', 'ref_baseline']].values,
                self[['rep', 'rep_baseline']].values
            ]),
            columns=['date', 'baseline']
        ).drop_duplicates()

        # Plot date points
        ax.scatter(df_points['date'], df_points['baseline'], marker='o', c='b', s=40, zorder=10)

        # Plot pair connections
        for _, row in self.iterrows():
            ax.plot([row['ref'], row['rep']], [row['ref_baseline'], row['rep_baseline']],
                    c='#30a2da', lw=0.5, zorder=5)

        # Add date labels
        if show_labels:
            try:
                import adjustText
                texts = []
                for date, baseline in df_points.values:
                    texts.append(ax.text(date, baseline, str(date.date()), ha='center', va='bottom'))
                adjustText.adjust_text(texts, ax=ax)
            except ImportError:
                # Fall back to simple labels without adjustment
                for date, baseline in df_points.values:
                    ax.text(date, baseline, str(date.date()), ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Timeline')
        ax.set_ylabel('Perpendicular Baseline [m]')
        ax.set_title(caption)
        ax.grid(True)

        # Format x-axis labels: short date format without century, rotated
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        return ax

    def hist(self, interval_days: int = 6, caption: str = 'Durations Histogram',
             column: str | None = None, ascending: bool | None = None,
             cmap: str = 'turbo', vmin: float | None = None, vmax: float | None = None,
             figsize: tuple[float, float] | None = None, ax=None):
        """Plot histogram of baseline pair durations.

        Parameters
        ----------
        interval_days : int, optional
            Bin width in days. Default is 6.
        caption : str, optional
            Plot title. Default is 'Durations Histogram'.
        column : str, optional
            Column name for color-coding bars. If None, uniform color.
        ascending : bool, optional
            Sort direction for stacked bars when column is specified.
        cmap : str, optional
            Colormap name. Default is 'turbo'.
        vmin, vmax : float, optional
            Color scale limits.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Build caption with burst ID prefix if available
        if self._burst_id:
            caption = f'{self._burst_id} {caption}'

        if len(self) == 0 or 'duration' not in self.columns:
            ax.set_title(caption)
            return ax

        max_duration = self['duration'].max()
        bins = np.arange(interval_days / 2, max_duration + interval_days, interval_days)
        bin_midpoints = (bins[:-1] + bins[1:]) / 2

        if column is not None and column in self.columns and ascending is None:
            # Calculate histogram with average column values
            counts, edges = np.histogram(self['duration'], bins=bins)
            sums, _ = np.histogram(self['duration'], bins=bins, weights=self[column])
            averages = sums / np.where(counts > 0, counts, 1)
            averages = np.where(counts > 0, averages, np.nan)

            # Normalize the average values for coloring
            norm = mcolors.Normalize(
                vmin=vmin if vmin is not None else np.nanmin(averages),
                vmax=vmax if vmax is not None else np.nanmax(averages)
            )
            colormap = plt.colormaps[cmap]

            for i in range(len(bin_midpoints)):
                bin_color = 'white' if np.isnan(averages[i]) else colormap(norm(averages[i]))
                ax.bar(bin_midpoints[i], counts[i], width=bins[i+1] - bins[i],
                       color=bin_color, edgecolor='black', align='center', zorder=3)

            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, label=f'Average {column}')

        elif column is not None and column in self.columns:
            # Stacked bars sorted by column value
            norm = mcolors.Normalize(
                vmin=vmin if vmin is not None else self[column].min(),
                vmax=vmax if vmax is not None else self[column].max()
            )
            colormap = plt.colormaps[cmap]

            for i in range(len(bins) - 1):
                bin_data = self[(self['duration'] >= bins[i]) & (self['duration'] < bins[i + 1])]
                bin_data = bin_data.sort_values(by=column, ascending=ascending)

                bottom = 0
                for _, row in bin_data.iterrows():
                    color = colormap(norm(row[column]))
                    ax.bar(bin_midpoints[i], 1, bottom=bottom, width=bins[i+1] - bins[i],
                           color=color, edgecolor='black', align='center', zorder=3)
                    bottom += 1

            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, label=f'{column} value')
        else:
            # Simple histogram
            ax.hist(self['duration'], bins=bins, color='skyblue', edgecolor='black', align='mid', zorder=3)

        ax.set_xlabel('Duration [days]')
        ax.set_ylabel('Number of Interferograms')
        ax.set_title(caption)
        ax.grid(True, zorder=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        return ax

    def filter(self, days: int | None = None, meters: float | None = None) -> "Baseline":
        """Filter baseline pairs by temporal and spatial criteria.

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days.
        meters : float, optional
            Maximum baseline difference in meters.

        Returns
        -------
        Baseline
            Filtered baseline pairs.
        """
        result = self.copy()

        if days is not None:
            result = result[result['duration'] <= days]

        if meters is not None:
            result = result[np.abs(result['baseline']) <= meters]

        return Baseline(result, burst_id=self._burst_id, dates=self._dates)

    def limit(self, n: int = 2, iterations: int = 1) -> "Baseline":
        """Limit pairs to ensure each date has at least n connections.

        Parameters
        ----------
        n : int, optional
            Minimum number of pairs per date. Default is 2.
        iterations : int, optional
            Number of filtering iterations. Default is 1.

        Returns
        -------
        Baseline
            Filtered baseline pairs.
        """
        result = self.copy()

        for _ in range(iterations):
            # Count connections per date
            ref_counts = result['ref'].value_counts()
            rep_counts = result['rep'].value_counts()
            all_dates = set(result['ref'].unique()) | set(result['rep'].unique())

            # Filter dates with too few connections
            valid_dates = []
            for date in all_dates:
                count = ref_counts.get(date, 0) + rep_counts.get(date, 0)
                if count >= n:
                    valid_dates.append(date)

            # Keep pairs where both dates are valid
            result = result[result['ref'].isin(valid_dates) & result['rep'].isin(valid_dates)]

        return Baseline(result, burst_id=self._burst_id, dates=self._dates)
