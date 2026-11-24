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


def _alos1_date_groups(items: list, records) -> list:
    """
    The items (sceneId, polarization, scene) of an ALOS-1 scene grouped by date, HH first: the polarizations of a date share
    ONE alignment, defined by HH (a PLR date's HH, HV, VH and VV are one group; an FBD date's HH and HV as well).

    A date without HH is grouped as the stripmap base does (Stripmap_transform._date_groups: the co-pol with its cross-pol, a
    cross-pol alone forms its own group). The groups come in the order of first appearance of their dates.

    Parameters
    ----------
    items : list of tuple
        The record indices (sceneId, polarization, scene), as get_repref() gives them.
    records : pandas.DataFrame
        The records (the scan's df): their startTime by the scene (index level 2).

    Returns
    -------
    list of list of tuple
        The groups, each the item that defines its alignment first.

    Examples
    --------
    >>> import pandas as pd
    >>> records = pd.DataFrame({'startTime': [pd.Timestamp('2008-04-02 06:21:44')] * 4},
    ...                        index=pd.MultiIndex.from_tuples([('s', p, f's_{p}') for p in ('HH', 'HV', 'VH', 'VV')]))
    >>> _alos1_date_groups([('s', 'VV', 's_VV'), ('s', 'HV', 's_HV'), ('s', 'HH', 's_HH'), ('s', 'VH', 's_VH')], records)
    [[('s', 'HH', 's_HH'), ('s', 'VV', 's_VV'), ('s', 'HV', 's_HV'), ('s', 'VH', 's_VH')]]
    """
    from .Stripmap_transform import _date_groups

    scenes = records.index.get_level_values(2)
    dates = {}
    for item in items:
        dates.setdefault(records.startTime[scenes == item[-1]].iloc[0], []).append(item)
    groups = []
    for date_items in dates.values():
        if any(item[1] == 'HH' for item in date_items):
            # HH first, then the co-pol and the cross-pols in their order
            groups.append(sorted(date_items, key=lambda item: (item[1] != 'HH', item[1][0] != item[1][1])))
        else:
            groups.extend(_date_groups(date_items, records))
    return groups


class Alos1_slc(Satellite):
    """ALOS-1 PALSAR L1.1 (SLC) data manager - scans a directory for the IMG and LED files."""
    import xarray as xr

    # Pattern for subdirectories: path_frame_offnadir (e.g., 216_0680_343, the off-nadir angle in 0.1 degree); the raw
    # (L1.0) folders end with _raw and do not match
    pattern_prefix: str = '[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]'
    # Pattern for the IMG files: AL1_{path}_{frame}_{offnadir}_{yyyymmdd}_{pol}.img
    pattern_alos1: str = 'AL1_*_[HV][HV].img'

    def __init__(self, datadir: str, DEM: str | xr.DataArray | xr.Dataset | None = None):
        """
        Scan a directory for ALOS-1 IMG files and build the records.

        The layout is the one insardev_toolkit `ASF.download` stores: `<path>_<frame>_<offnadir>/AL1_<path>_<frame>_<offnadir>
        _<yyyymmdd>_<pol>.img`, with the date's `AL1_<path>_<frame>_<offnadir>_<yyyymmdd>.led` beside it. One record per IMG
        file (date and polarization); the LED holds the orbit, so the records carry it in the 'orbit' column. The column
        'subswath' is the off-nadir code of the name (the beam), 'doppler' the date's Doppler centroid at mid swath in Hz
        (utils_alos1.alos1_doppler_centroid: the image is focused at it, not at zero Doppler), and the geometry is the footprint
        of JAXA's per-line positions (utils_alos2.alos2_footprint), the image's own native-Doppler geometry.

        FBS (32 MHz range sampling) and FBD (16 MHz) dates of one path, frame and off-nadir angle form one stack: each date
        keeps its own PRM (sampling, chirp, near range), and the alignment maps each date's own radar coordinates
        (Stripmap_align).

        Parameters
        ----------
        datadir : str
            Directory containing the ALOS-1 files in subdirectories like 216_0680_343/.
        DEM : str or xarray, optional
            Path to DEM file or DEM DataArray.

        Raises
        ------
        ValueError
            No IMG files.

        Examples
        --------
        >>> Alos1_slc('raw/alos1_rosamond', DEM='raw/alos1_rosamond/dem.vrt')
        """
        import os
        from glob import glob
        from datetime import timedelta
        import pandas as pd
        import geopandas as gpd
        from insardev_toolkit.utils_files import exists
        from .utils_alos1 import alos1_prm, alos1_timing, alos1_doppler_centroid
        from .utils_alos2 import alos2_footprint

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
            for name in sorted(glob(self.pattern_alos1, root_dir=os.path.join(self.datadir, prefix))):
                img_files.append((prefix, os.path.join(self.datadir, prefix, name)))
        if len(img_files) == 0:
            raise ValueError(f'No ALOS-1 IMG files found in {datadir} subdirectories matching '
                             f'{self.pattern_prefix}/{self.pattern_alos1}')
        # an empty file raises before any file is read
        for _, img_path in img_files:
            exists(img_path)
            exists(self._led_path(img_path))

        records = []
        for prefix, img_path in img_files:
            led_path = self._led_path(img_path)
            prm = alos1_prm(img_path, led_path)
            timing = alos1_timing(img_path, led_path)
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
                'doppler': alos1_doppler_centroid(prm),
                'geometry': alos2_footprint(img_path)
            })

        df = pd.DataFrame(records)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326) \
            .sort_values(by=['sceneId', 'polarization', 'scene']) \
            .set_index(['sceneId', 'polarization', 'scene'])

        print(f'NOTE: Loaded {len(df)} ALOS-1 records.')
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
        >>> Alos1_slc._led_path('216_0680_343/AL1_216_0680_343_20070703_HH.img')
        '216_0680_343/AL1_216_0680_343_20070703.led'
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
        >>> alos1._orbit_file('AL1_216_0680_343_20070703_HH', alos1.get_record('AL1_216_0680_343_20070703_HH'))
        """
        return record['orbit'].iloc[0]

    def _check_orbit_file(self, led_path: str, scenes: list):
        """
        Check the orbit of a LED file as the processing reads it (utils_alos1.alos1_orbit), against every scene that uses it
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
        >>> alos1._check_orbit_file('216_0680_343/AL1_216_0680_343_20070703.led', ['AL1_216_0680_343_20070703_HH'])
        """
        from .utils_alos1 import alos1_orbit
        from insardev_toolkit.utils_files import exists

        # a LED file emptied after the scan raises (utils_files.exists)
        exists(led_path)
        for scene in scenes:
            alos1_orbit(led_path, self.get_record(scene)['path'].iloc[0])

    def _make_scene(self, scene: str, mode: int = 2, debug: bool = False):
        """
        Extract PRM, orbit, and optionally SLC data for an ALOS-1 scene.

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
            Note: reramp_params is always None for ALOS-1 (no TOPS deramp)

        Examples
        --------
        >>> alos1._make_scene('AL1_216_0680_343_20070703_HH', mode=0)
        """
        from .utils_alos1 import alos1_prm, alos1_orbit
        from .utils_alos2 import alos2_slc
        from .PRM import PRM
        from insardev_toolkit.utils_files import exists

        record = self.get_record(scene)
        img_path = record['path'].iloc[0]
        led_path = record['orbit'].iloc[0]
        # a scene file emptied after the scan raises (utils_files.exists)
        exists(img_path)
        exists(led_path)

        prm_dict = alos1_prm(img_path, led_path)
        orbit_df = alos1_orbit(led_path, img_path)

        if mode == 0:
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            return prm, orbit_df

        slc_data = alos2_slc(img_path)
        return prm_dict, orbit_df, slc_data, None
