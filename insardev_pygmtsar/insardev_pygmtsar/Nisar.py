# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Nisar_transform import Nisar_transform
from .PRM import PRM


class Nisar(Nisar_transform):
    """
    Nisar RSLC data manager and preprocessor.

    Main class for preprocessing Nisar RSLC HDF5 data to geocoded Zarr stacks.
    Nisar uses stripmap mode (simpler than S1 TOPS) - no deramp/reramp needed.

    Usage
    -----
    >>> from insardev_pygmtsar import Nisar
    >>> nisar = Nisar('/path/to/nisar/data', DEM='/path/to/dem.tif')
    >>> nisar.to_dataframe()
    >>> nisar.plot()
    >>> nisar.transform('/output/stack.zarr', ref='2025-11-22')
    """
    import pandas as pd
    import xarray as xr

    # Class variables
    datadir: str | None = None
    DEM: str | xr.DataArray | xr.Dataset | None = None
    df: pd.DataFrame | None = None
    frequency: str | None = None

    def info(self) -> dict:
        """
        Get summary information about the Nisar dataset.

        Returns
        -------
        dict
            Dataset information including dates, tracks, frequencies, etc.
        """
        if self.df is None or len(self.df) == 0:
            return {'error': 'No data loaded'}

        dates = self.df.startTime.dt.date.unique()
        tracks = self.df.track.unique() if 'track' in self.df.columns else []
        pols = self.df.index.get_level_values(1).unique().tolist()

        return {
            'n_scenes': len(self.df),
            'n_dates': len(dates),
            'date_range': (str(min(dates)), str(max(dates))),
            'tracks': list(tracks),
            'polarizations': pols,
            'frequency': self.frequency,
        }
