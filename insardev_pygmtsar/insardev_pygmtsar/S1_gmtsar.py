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

    def _orbit_file(self, burst: str, record) -> str:
        """
        Return the orbit file path of a burst from its record, FileNotFoundError when it has no orbit file.
        """
        import os
        orbit_val = record['orbit'].iloc[0]
        if not isinstance(orbit_val, str):
            raise FileNotFoundError(f'ERROR: No orbit file for burst {burst} ({record["startTime"].iloc[0].date()}). '
                                    f'Download the orbits.')
        return os.path.join(self.datadir, orbit_val)

    def _burst_files(self, burst: str) -> tuple:
        """
        Return the (annotation, measurement, orbit) files of a burst; an annotation or orbit file emptied after
        the scan raises, as the measurement does (utils_files.exists).
        """
        import os
        from insardev_toolkit.utils_S1 import measurement_path
        from insardev_toolkit.utils_files import exists

        df = self.get_record(burst)
        prefix = self.fullBurstId(burst)

        # File paths: the burst measurement is <burst>.nc, or the legacy <burst>.tiff
        xml_file = os.path.join(self.datadir, prefix, 'annotation', f'{burst}.xml')
        tiff_file = measurement_path(os.path.join(self.datadir, prefix, 'measurement'), burst)
        orbit_file = self._orbit_file(burst, df)
        exists(xml_file)
        exists(orbit_file)
        return xml_file, tiff_file, orbit_file

    def _check_orbit_file(self, orbit_file: str, bursts: list):
        """
        Check the orbit file of the bursts as the processing reads it for each of them (satellite_orbit), with
        one read of the file (utils_s1.check_orbit_file).
        """
        from .utils_s1 import satellite_prm, check_orbit_file

        spans = []
        for burst in bursts:
            xml_file, tiff_file, _ = self._burst_files(burst)
            prm = satellite_prm(xml_file, tiff_file)
            spans.append((burst, prm['SC_clock_start'], prm['SC_clock_stop']))
        check_orbit_file(orbit_file, spans)

    def _make_burst(self, burst: str, mode: int = 0, debug: bool = False):
        """
        Extract PRM and orbit data (and optionally the deramped SLC) for a burst.

        Pure Python implementation - no GMTSAR binaries required.
        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst : str
            Burst identifier
        mode : int, optional
            0 - PRM and orbit only (no SLC)
            2 - PRM, orbit, deramped SLC, and reramp params
            Defaults to 0.
        debug : bool, optional
            Enable debug output. Defaults to False.

        Returns
        -------
        tuple
            (prm, orbit_df) for mode=0 where prm is a PRM object with orbit_df attached
            (prm_dict, orbit_df, slc_data, reramp_params) for mode=2

        Examples
        --------
        >>> prm, orbit_df = s1._make_burst(burst, mode=0)
        >>> prm_dict, orbit_df, slc, reramp_params = s1._make_burst(burst, mode=2)
        """
        xml_file, tiff_file, orbit_file = self._burst_files(burst)

        if mode == 2:
            from .utils_s1 import deramped_burst
            return deramped_burst(xml_file, tiff_file, orbit_file)
        if mode != 0:
            raise ValueError(f'mode must be 0 or 2, got {mode}')

        return make_burst(
            xml_file=xml_file,
            tiff_file=tiff_file,
            orbit_file=orbit_file,
            debug=debug
        )
