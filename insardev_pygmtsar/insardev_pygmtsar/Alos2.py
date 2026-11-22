# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Alos2_transform import Alos2_transform


class Alos2(Alos2_transform):
    """
    ALOS-2 PALSAR-2 ScanSAR L1.1 data manager and preprocessor.

    Scans the subswath crops that insardev_toolkit `ASF.download` stores (`<path>_<frame>_<Fn>/AL2_*_<pol>.img` and their
    `.led` files) into records, one per date, subswath and polarization, with their baselines, and transforms them to the
    geocoded zarr stack (transform), the polarizations of a date on the co-pol's alignment.

    Usage
    -----
    >>> from insardev_pygmtsar import Alos2
    >>> alos2 = Alos2('/path/to/data', DEM='/path/to/dem.vrt')
    >>> alos2.to_dataframe()
    >>> alos2.plot()
    >>> alos2.transform('/path/to/stack', ref='2015-04-04')
    """
    import pandas as pd
    import xarray as xr

    # Class variables
    datadir: str | None = None
    DEM: str | xr.DataArray | xr.Dataset | None = None
    df: pd.DataFrame | None = None

    def info(self) -> dict:
        """
        Get summary information about the ALOS-2 dataset.

        Returns
        -------
        dict
            Dataset information: number of records and dates, the date range, the paths and the polarizations.

        Examples
        --------
        >>> alos2.info()
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
