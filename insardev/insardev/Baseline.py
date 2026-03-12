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


def _cleanup_network(df, min_connections=2):
    """Iteratively remove degraded dates from a baseline network.

    Removes:
    1. Dates with fewer than ``min_connections`` pairs
    2. Internal dates connected only as ref (not rep) — except the first date
    3. Internal dates connected only as rep (not ref) — except the last date

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'ref' and 'rep' columns (datetime).
    min_connections : int
        Minimum pairs per date. Dates below this are removed.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (same columns as input).
    """
    while len(df) > 0:
        all_dates = sorted(set(df['ref']) | set(df['rep']))
        if len(all_dates) < 2:
            break
        first_date, last_date = all_dates[0], all_dates[-1]
        counts = pd.concat([df['ref'], df['rep']]).value_counts()
        hanging = set(counts[counts < min_connections].index)
        ref_only = set(df['ref'].unique()) - set(df['rep'].unique()) - {first_date}
        rep_only = set(df['rep'].unique()) - set(df['ref'].unique()) - {last_date}
        bad_dates = hanging | ref_only | rep_only
        if not bad_dates:
            break
        df = df[~df['ref'].isin(bad_dates) & ~df['rep'].isin(bad_dates)]
    return df


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
    >>> baseline = stack.baseline()
    >>> baseline.plot()  # Plot baseline network
    >>> baseline.filter(days=100).plot()  # Plot filtered network
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

        # Count connections per date
        date_counts = pd.concat([self['ref'], self['rep']]).value_counts()

        # Plot pair connections
        for _, row in self.iterrows():
            ax.plot([row['ref'], row['rep']], [row['ref_baseline'], row['rep_baseline']],
                    c='#6cc0e8', lw=0.5, zorder=5)

        # Plot date points colored by connection count
        colors = []
        for date in df_points['date']:
            n = date_counts.get(date, 0)
            if n <= 1:
                colors.append('red')
            elif n == 2:
                colors.append('yellow')
            elif n == 3:
                colors.append('orange')
            else:
                colors.append('#6cc0e8')
        ax.scatter(df_points['date'], df_points['baseline'], marker='o', c=colors, s=40,
                   edgecolors='white', linewidths=0.5, zorder=10)

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
        n_pairs = len(self)
        n_dates = len(df_points)
        ax.set_title(f'{caption}\n{n_pairs} pairs, {n_dates} dates')
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

    def filter(self, days=None, meters=None, date=None, pair=None, count=None,
               min_connections=2, cleanup=True) -> "Baseline":
        """Filter baseline pairs by temporal/spatial criteria and date/pair exclusion.

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days.
        meters : float, optional
            Maximum baseline difference in meters.
        date : str or list, optional
            Date(s) to exclude. Accepts a single date string or a list.
        pair : str or list, optional
            Pair(s) to exclude. Each pair is ``'YYYY-MM-DD YYYY-MM-DD'``.
        count : int, optional
            Remove dates with fewer than this many connections.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        cleanup : bool, optional
            If True (default), iteratively remove hanging and single-side
            connected dates. Set to False for raw network.

        Returns
        -------
        Baseline
            Filtered baseline pairs.

        Examples
        --------
        >>> bl = stack.baseline()
        >>> bl.filter(days=100, meters=80).plot()
        >>> bl.filter(date='2024-12-30').plot()
        >>> bl.filter(pair='2024-06-21 2024-12-30')
        >>> bl.filter(count=3)  # remove dates with < 3 connections
        """
        if days is None and meters is None and date is None and pair is None and count is None:
            return self

        df = self.copy()

        if days is not None:
            df = df[df['duration'] <= days]

        if meters is not None:
            df = df[np.abs(df['baseline']) <= meters]

        if date is not None:
            if isinstance(date, str):
                date = [date]
            exclude_dates = pd.to_datetime(date).normalize()
            df = df[~df['ref'].isin(exclude_dates) & ~df['rep'].isin(exclude_dates)]

        if pair is not None:
            if isinstance(pair, str):
                pair = [pair]
            exclude_pairs = set()
            for p in pair:
                parts = str(p).split()
                r, s = pd.Timestamp(parts[0]).normalize(), pd.Timestamp(parts[1]).normalize()
                exclude_pairs.add((r, s))
            mask = [True] * len(df)
            for i, (_, row) in enumerate(df.iterrows()):
                if (row['ref'], row['rep']) in exclude_pairs:
                    mask[i] = False
            df = df[mask]

        if len(df) > 0:
            min_conn = max(min_connections, count) if count is not None else min_connections
            if cleanup:
                df = _cleanup_network(df, min_connections=min_conn)
            elif count is not None:
                counts = pd.concat([df['ref'], df['rep']]).value_counts()
                low_dates = set(counts[counts < count].index)
                df = df[~df['ref'].isin(low_dates) & ~df['rep'].isin(low_dates)]

        if len(df) == 0:
            raise ValueError("No valid baseline pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")

        return Baseline(df.reset_index(drop=True), burst_id=self._burst_id, dates=self._dates)

