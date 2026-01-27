# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_slc import S1_slc
from .PRM import PRM
from .utils_s1 import make_burst


class S1_gmtsar(S1_slc):

    def _make_burst(self, burst: str, mode: int = 0,
                    rshift_grid=None, ashift_grid=None, debug: bool = False):
        """
        Extract PRM and orbit data (and optionally SLC) for a burst.

        Pure Python implementation - no GMTSAR binaries required.
        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst : str
            Burst identifier
        mode : int, optional
            0 - PRM and orbit only (no SLC)
            1 - PRM, orbit, and SLC data
            2 - PRM, orbit, deramped SLC, and reramp params
            Defaults to 0.
        rshift_grid : np.ndarray, optional
            Range shift grid for alignment (mode=1 only)
        ashift_grid : np.ndarray, optional
            Azimuth shift grid for alignment (mode=1 only)
        debug : bool, optional
            Enable debug output. Defaults to False.

        Returns
        -------
        tuple
            (prm, orbit_df) for mode=0 where prm is a PRM object with orbit_df attached
            (prm, orbit_df, slc_data) for mode=1

        Examples
        --------
        >>> prm, orbit_df = s1._make_burst(burst, mode=0)
        >>> prm, orbit_df, slc = s1._make_burst(burst, mode=1)
        """
        import os

        df = self.get_record(burst)
        prefix = self.fullBurstId(burst)

        # File paths
        xml_file = os.path.join(self.datadir, prefix, 'annotation', f'{burst}.xml')
        tiff_file = os.path.join(self.datadir, prefix, 'measurement', f'{burst}.tiff')
        orbit_file = os.path.join(self.datadir, df['orbit'].iloc[0])

        if mode == 2:
            from .utils_s1 import deramped_burst
            return deramped_burst(xml_file, tiff_file, orbit_file)

        return make_burst(
            xml_file=xml_file,
            tiff_file=tiff_file,
            orbit_file=orbit_file,
            mode=mode,
            rshift_grid=rshift_grid,
            ashift_grid=ashift_grid,
            debug=debug
        )
