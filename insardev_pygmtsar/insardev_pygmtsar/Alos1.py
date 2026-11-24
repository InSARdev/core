# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Alos1_transform import Alos1_transform


class Alos1(Alos1_transform):
    """
    ALOS-1 PALSAR L1.1 (SLC) data manager and preprocessor.

    Scans the crops that insardev_toolkit `ASF.download` stores (`<path>_<frame>_<offnadir>/AL1_*_<pol>.img` and their `.led`
    files) into records, one per date and polarization, with their baselines and Doppler centroids, and transforms them to the
    geocoded zarr stack (transform): each date from its native-Doppler image geometry, the polarizations of a date on HH's
    alignment.

    Usage
    -----
    >>> from insardev_pygmtsar import Alos1
    >>> alos1 = Alos1('/path/to/data', DEM='/path/to/dem.vrt')
    >>> alos1.to_dataframe()
    >>> alos1.plot()
    >>> alos1.transform('/path/to/stack', ref='2007-07-03')
    """
    import pandas as pd
    import xarray as xr

    # Class variables
    datadir: str | None = None
    DEM: str | xr.DataArray | xr.Dataset | None = None
    df: pd.DataFrame | None = None

    def info(self) -> dict:
        """
        Get summary information about the ALOS-1 dataset.

        Returns
        -------
        dict
            Dataset information: number of records and dates, the date range, the paths and the polarizations.

        Examples
        --------
        >>> alos1.info()
        """
        if self.df is None or len(self.df) == 0:
            return {'error': 'No data loaded'}

        dates = self.df.startTime.dt.date.unique()
        paths = self.df.pathNumber.unique() if 'pathNumber' in self.df.columns else []
        pols = self.df.index.get_level_values(1).unique().tolist()

        return {
            'n_scenes': len(self.df),
            'n_dates': len(dates),
            'date_range': (str(min(dates)), str(max(dates))),
            'paths': list(paths),
            'polarizations': pols,
        }
