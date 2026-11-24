# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Satellite import Satellite

# JAXA synchronised the ALOS-2 ScanSAR burst timing on this date
ALOS2_BURST_SYNC_DATE = '2015-02-08'


def _burst_sync_warning(dates):
    """
    The WARNING about ALOS-2 ScanSAR records acquired before JAXA synchronised the burst timing (2015-02-08), or None.

    The bursts of those records are not synchronised with other acquisitions, so their interferograms lose coherence; they remain
    usable for interferograms of large events. The dates themselves are visible in the records table.

    Parameters
    ----------
    dates : iterable of datetime.date, datetime.datetime or str
        The acquisition date of each record ('YYYY-MM-DD' for a string).

    Returns
    -------
    str or None
        One line with the number of the given dates before 2015-02-08, or None when there is none.

    Examples
    --------
    >>> _burst_sync_warning(['2015-01-10', '2015-02-21', '2015-04-04'])
    'WARNING: 1 ALOS-2 ScanSAR records before 2015-02-08 have unsynchronised bursts.'
    """
    count = sum(1 for date in dates if str(date)[:10] < ALOS2_BURST_SYNC_DATE)
    if count == 0:
        return None
    return f'WARNING: {count} ALOS-2 ScanSAR records before {ALOS2_BURST_SYNC_DATE} have unsynchronised bursts.'


class Alos2_slc(Satellite):
    """ALOS-2 PALSAR-2 ScanSAR L1.1 data manager - scans a directory for the subswath IMG and LED files."""
    import xarray as xr

    # Pattern for subdirectories: path_frame_subswath (e.g., 129_4450_F4)
    pattern_prefix: str = '[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_F[1-5]'
    # Pattern for the IMG files: AL2_{path}_{frame}_{subswath}_{yyyymmdd}_{pol}.img
    pattern_alos2: str = 'AL2_*_[HV][HV].img'

    def __init__(self, datadir: str, DEM: str | xr.DataArray | xr.Dataset | None = None):
        """
        Scan a directory for ALOS-2 subswath IMG files and build the records.

        One record per IMG file (date, subswath and polarization); its LED file (the IMG name without `_<pol>.img`, plus `.led`)
        holds the orbit, so the records carry it in the 'orbit' column. The records acquired before JAXA synchronised the ScanSAR
        burst timing (2015-02-08) are counted in one WARNING line (_burst_sync_warning).

        Parameters
        ----------
        datadir : str
            Directory containing the ALOS-2 files in subdirectories like 129_4450_F4/.
        DEM : str or xarray, optional
            Path to DEM file or DEM DataArray.

        Raises
        ------
        ValueError
            No IMG files.

        Examples
        --------
        >>> Alos2_slc('raw/alos2_calbuco', DEM='raw/alos2_calbuco/dem.vrt')
        """
        import os
        from glob import glob
        from datetime import timedelta
        import pandas as pd
        import geopandas as gpd
        from insardev_toolkit.utils_files import exists
        from .utils_alos2 import alos2_prm, alos2_timing, alos2_footprint

        self.datadir = datadir
        # a DEM file name is resolved here once: a dem.nc that the downloader replaced by dem.vrt is read from
        # the VRT, and the note about it prints in this process, not in every worker
        if isinstance(DEM, str):
            from insardev_toolkit import utils_tiles
            DEM = utils_tiles.resolve(DEM)
        self.DEM = DEM
        # its vertical datum as well (dem_datum()): a DEM without one warns here, and a DEM of a vertical datum that
        # is not supported, or mixing providers, raises before any processing
        if DEM is not None:
            self.dem_datum()

        img_files = []
        for prefix in sorted(glob(self.pattern_prefix, root_dir=self.datadir)):
            for name in sorted(glob(self.pattern_alos2, root_dir=os.path.join(self.datadir, prefix))):
                img_files.append((prefix, os.path.join(self.datadir, prefix, name)))
        if len(img_files) == 0:
            raise ValueError(f'No ALOS-2 IMG files found in {datadir} subdirectories matching '
                             f'{self.pattern_prefix}/{self.pattern_alos2}')
        # an empty file raises before any file is read
        for _, img_path in img_files:
            exists(img_path)
            exists(self._led_path(img_path))

        records = []
        for prefix, img_path in img_files:
            led_path = self._led_path(img_path)
            prm = alos2_prm(img_path, led_path)
            timing = alos2_timing(img_path, led_path)
            records.append({
                'sceneId': prefix,
                'polarization': prm['polarization'],
                'scene': os.path.splitext(os.path.basename(img_path))[0],
                # the first line's time, to the whole second
                'startTime': timing['day'] + timedelta(seconds=int(timing['start'])),
                'path': img_path,
                'orbit': led_path,
                'pathNumber': prm['path'],
                'frameNumber': prm['frame'],
                'subswath': prm['subswath'],
                'flightDirection': prm['orbdir'],
                'lookDirection': prm['lookdir'],
                'band': 'L',
                'geometry': alos2_footprint(img_path)
            })

        df = pd.DataFrame(records)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326) \
            .sort_values(by=['sceneId', 'polarization', 'scene']) \
            .set_index(['sceneId', 'polarization', 'scene'])

        print(f'NOTE: Loaded {len(df)} ALOS-2 records.')
        # the records acquired before the burst synchronisation, once per scan
        warning = _burst_sync_warning(df.startTime.dt.date)
        if warning is not None:
            print(warning)
        self.df = df
        # a record attribute, placed before the geometry, which is kept last as the long field
        self.df.insert(self.df.columns.get_loc('geometry'), 'BPR', self.baselines())

    @staticmethod
    def _led_path(img_path: str) -> str:
        """
        The LED file of an IMG file: the IMG name without `_<pol>.img`, plus `.led`.

        Parameters
        ----------
        img_path : str
            Path to the IMG file.

        Returns
        -------
        str
            Path to the LED file.

        Examples
        --------
        >>> Alos2_slc._led_path('129_4450_F4/AL2_129_4450_F4_20150404_HH.img')
        '129_4450_F4/AL2_129_4450_F4_20150404.led'
        """
        return img_path.rsplit('_', 1)[0] + '.led'

    def _orbit_file(self, scene: str, record) -> str:
        """
        Return the file that holds the orbit of a scene from its record: its LED file.

        Parameters
        ----------
        scene : str
            Scene identifier.
        record : pandas.DataFrame
            The record of the scene (get_record).

        Returns
        -------
        str
            Path to the LED file.

        Examples
        --------
        >>> alos2._orbit_file('AL2_129_4450_F4_20150404_HH', alos2.get_record('AL2_129_4450_F4_20150404_HH'))
        """
        return record['orbit'].iloc[0]

    def _check_orbit_file(self, led_path: str, scenes: list):
        """
        Check the orbit of a LED file as the processing reads it (utils_alos2.alos2_orbit), against every scene that uses it
        (its polarizations).

        Parameters
        ----------
        led_path : str
            Path to the LED file.
        scenes : list
            The scene identifiers of the records that use it.

        Raises
        ------
        ValueError
            An orbit that does not cover a scene, has a gap or a state vector that is not finite.

        Examples
        --------
        >>> alos2._check_orbit_file('129_4450_F4/AL2_129_4450_F4_20150404.led', ['AL2_129_4450_F4_20150404_HH'])
        """
        from .utils_alos2 import alos2_orbit
        from insardev_toolkit.utils_files import exists

        # a LED file emptied after the scan raises (utils_files.exists)
        exists(led_path)
        for scene in scenes:
            alos2_orbit(led_path, self.get_record(scene)['path'].iloc[0])

    def _make_scene(self, scene: str, mode: int = 2, debug: bool = False):
        """
        Extract PRM, orbit, and optionally SLC data for an ALOS-2 scene.

        Parameters
        ----------
        scene : str
            Scene identifier.
        mode : int
            0 = PRM only, 2 = PRM + SLC data
        debug : bool
            Enable debug output.

        Returns
        -------
        tuple
            (PRM with orbit_df, orbit_df) if mode=0
            (prm_dict, orbit_df, slc_data, None) if mode=2
            Note: reramp_params is always None for ALOS-2 (no TOPS deramp)

        Examples
        --------
        >>> alos2._make_scene('AL2_129_4450_F4_20150404_HH', mode=0)
        """
        from .utils_alos2 import alos2_prm, alos2_orbit, alos2_slc
        from .PRM import PRM
        from insardev_toolkit.utils_files import exists

        record = self.get_record(scene)
        img_path = record['path'].iloc[0]
        led_path = record['orbit'].iloc[0]
        # a scene file emptied after the scan raises (utils_files.exists)
        exists(img_path)
        exists(led_path)

        prm_dict = alos2_prm(img_path, led_path)
        orbit_df = alos2_orbit(led_path, img_path)

        if mode == 0:
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            return prm, orbit_df

        slc_data = alos2_slc(img_path)
        return prm_dict, orbit_df, slc_data, None
