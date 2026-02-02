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

