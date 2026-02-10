# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_transform import S1_transform
from .PRM import PRM

class S1(S1_transform):
    import pandas as pd
    import xarray as xr

    # class variables
    datadir: str|None = None
    DEM: str|xr.DataArray|xr.Dataset|None = None
    df: pd.DataFrame|None = None
