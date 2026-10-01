# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Satellite import Satellite


class Nisar_slc(Satellite):
    """Nisar RSLC data manager - scans directory for HDF5 files."""
    import xarray as xr

    # Pattern for subdirectories: track_frame (e.g., 172_008)
    pattern_prefix: str = '[0-9]*_[0-9]*'
    # Pattern for Nisar HDF5 files: NSR_{track}_{frame}_{datetime}_{pol}.h5
    pattern_nisar: str = 'NSR_*.h5'

    def __init__(self, datadir: str, DEM: str | xr.DataArray | xr.Dataset | None = None,
                 frequency: str = None):
        """
        Scan directory for Nisar RSLC HDF5 files and build metadata DataFrame.

        Parameters
        ----------
        datadir : str
            Directory containing Nisar HDF5 files (in subdirectories like 172_008/).
        DEM : str or xarray, optional
            Path to DEM file or DEM DataArray.
        frequency : str, optional
            Frequency band to use: 'A' (20 MHz, high res) or 'B' (5 MHz, quick look).
            If None, auto-detects. Raises error if both present without explicit selection.
        """
        import os
        from glob import glob
        import re
        import pandas as pd
        import geopandas as gpd
        from datetime import datetime, timedelta
        import h5py
        from .utils_nisar import nisar_get_frequencies, nisar_get_polarizations, nisar_start_time

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
        self.frequency = frequency

        # Scan for subdirectories with track_frame pattern (like S1)
        prefixes = glob(self.pattern_prefix, root_dir=self.datadir)

        # Collect all HDF5 files from subdirectories
        h5_files = []
        for prefix in prefixes:
            subdir = os.path.join(self.datadir, prefix)
            files = glob(self.pattern_nisar, root_dir=subdir)
            for f in files:
                h5_files.append(os.path.join(prefix, f))

        # Also check root directory for backwards compatibility
        root_files = glob(self.pattern_nisar, root_dir=self.datadir)
        h5_files.extend(root_files)

        if len(h5_files) == 0:
            raise ValueError(f'No Nisar HDF5 files found in {datadir} or subdirectories matching pattern {self.pattern_nisar}')
        # an empty file raises before any file is read
        from insardev_toolkit.utils_files import exists
        for h5_file in h5_files:
            exists(os.path.join(self.datadir, h5_file))

        records = []
        detected_frequency = None
        # bbox crops whose identification/zeroDopplerStartTime is still the uncropped frame's start
        frame_starts = []

        for h5_file in h5_files:
            h5_path = os.path.join(self.datadir, h5_file)

            with h5py.File(h5_path, 'r') as f:
                    ident = f['science/LSAR/identification']

                    # Extract identification metadata
                    track = int(ident['trackNumber'][()])
                    frame = int(ident['frameNumber'][()])
                    orbit_dir = ident['orbitPassDirection'][()].decode() if isinstance(
                        ident['orbitPassDirection'][()], bytes) else str(ident['orbitPassDirection'][()])
                    look_dir = ident['lookDirection'][()].decode() if isinstance(
                        ident['lookDirection'][()], bytes) else str(ident['lookDirection'][()])

                    # Start time: the first SLC line's, to the whole second
                    day, sec = nisar_start_time(f)
                    start_time = day + timedelta(seconds=int(sec))
                    # a bbox crop downloaded before the toolkit set the crop's own start keeps the frame's there
                    zdt_str = ident['zeroDopplerStartTime'][()].decode() if isinstance(
                        ident['zeroDopplerStartTime'][()], bytes) else str(ident['zeroDopplerStartTime'][()])
                    ident_start = datetime.strptime(zdt_str[:19], '%Y-%m-%dT%H:%M:%S')
                    ident_frac = float('0.' + zdt_str[20:]) if len(zdt_str) > 20 else 0.0
                    lines = (((day - ident_start).total_seconds() + sec - ident_frac)
                             / float(f['science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing'][()]))
                    if abs(lines) >= 0.5:
                        frame_starts.append(f'{h5_file} ({lines:+.0f} lines)')

                    # Get available frequencies
                    available_freqs = nisar_get_frequencies(h5_path)

                    # Determine frequency to use
                    if frequency is not None:
                        if frequency not in available_freqs:
                            print(f'WARNING: Requested frequency {frequency} not available in {h5_file}, skipping.')
                            continue
                        use_freq = frequency
                    elif len(available_freqs) == 1:
                        use_freq = available_freqs[0]
                    else:
                        raise ValueError(
                            f"Both frequencyA and frequencyB available in {h5_file}. "
                            f"Please specify frequency='A' (high res) or frequency='B' (quick look)."
                        )

                    # Track detected frequency for consistency check
                    if detected_frequency is None:
                        detected_frequency = use_freq
                    elif detected_frequency != use_freq:
                        print(f'WARNING: Mixed frequencies in dataset. Using {detected_frequency}.')
                        continue

                    # Get polarizations for this frequency
                    pols = nisar_get_polarizations(h5_path, use_freq)

                    # Get geolocation grid for approximate footprint (WGS84 EPSG:4326)
                    import numpy as np
                    from shapely.geometry import Polygon
                    geoloc = f['science/LSAR/RSLC/metadata/geolocationGrid']
                    # coordinateX = longitude, coordinateY = latitude (EPSG 4326)
                    # Shape: (height_layers, azimuth, range) - use first height layer
                    lons = geoloc['coordinateX'][0]  # (azimuth, range)
                    lats = geoloc['coordinateY'][0]

                    # Get corner points for polygon (counterclockwise)
                    corners = [
                        (lons[0, 0], lats[0, 0]),      # top-left
                        (lons[0, -1], lats[0, -1]),    # top-right
                        (lons[-1, -1], lats[-1, -1]),  # bottom-right
                        (lons[-1, 0], lats[-1, 0]),    # bottom-left
                        (lons[0, 0], lats[0, 0]),      # close polygon
                    ]
                    geometry = Polygon(corners)

                    # Create scene ID: track_frame
                    scene_id = f'{track:03d}_{frame:03d}'

                    # Create scene name from filename
                    scene_name = os.path.splitext(h5_file)[0]

                    # Add record for each polarization
                    for pol in pols:
                        record = {
                            'sceneId': scene_id,
                            'polarization': pol,
                            'scene': scene_name,
                            'startTime': start_time,
                            'path': h5_path,
                            'track': track,
                            'frame': frame,
                            'flightDirection': 'A' if orbit_dir.lower().startswith('a') else 'D',
                            'lookDirection': look_dir[0].upper(),
                            'band': 'L',
                            'frequency': use_freq,
                            'geometry': geometry
                        }
                        records.append(record)

        if frame_starts:
            print(f"NOTE: identification/zeroDopplerStartTime is the uncropped frame's start, not the first line's "
                  f"time, in {len(frame_starts)} bbox crop(s) downloaded before insardev_toolkit set a crop's own "
                  f"start; the first line's time (swaths/zeroDopplerTime) is used: {', '.join(frame_starts)}.")

        if len(records) == 0:
            raise ValueError(f'No valid Nisar data found in {datadir}')

        # Build GeoDataFrame
        df = pd.DataFrame(records)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326) \
            .sort_values(by=['sceneId', 'polarization', 'scene']) \
            .set_index(['sceneId', 'polarization', 'scene'])

        # Store frequency
        if self.frequency is None:
            self.frequency = detected_frequency

        print(f'NOTE: Loaded {len(df)} Nisar scenes (frequency{self.frequency}).')
        self.df = df
        # a record attribute, placed before the geometry, which is kept last as the long field
        self.df.insert(self.df.columns.get_loc('geometry'), 'BPR', self.baselines())

    def _orbit_file(self, scene: str, record) -> str:
        """
        Return the file that holds the orbit of a scene from its record: the scene file itself.
        """
        return record['path'].iloc[0]

    def _check_orbit_file(self, h5_path: str, scenes: list):
        """
        Check the orbit of a scene file as the processing reads it (utils_nisar.nisar_orbit), once for all the
        records of the file (its polarizations).
        """
        from .utils_nisar import nisar_orbit
        from insardev_toolkit.utils_files import exists

        # a scene file emptied after the scan raises (utils_files.exists)
        exists(h5_path)
        nisar_orbit(h5_path)

    def _make_scene(self, scene: str, mode: int = 2, debug: bool = False):
        """
        Extract PRM, orbit, and optionally SLC data for a Nisar scene.

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
            (prm_dict, orbit_df) if mode=0
            (prm_dict, orbit_df, slc_data, None) if mode=2
            Note: reramp_params is always None for Nisar (stripmap mode)
        """
        from .utils_nisar import nisar_prm, nisar_orbit, nisar_slc
        from .PRM import PRM

        from insardev_toolkit.utils_files import exists

        record = self.get_record(scene)
        h5_path = record['path'].iloc[0]
        # a scene file emptied after the scan raises (utils_files.exists)
        exists(h5_path)
        pol = record.index.get_level_values(1)[0]
        frequency = self.frequency

        # Get PRM parameters
        prm_dict = nisar_prm(h5_path, pol=pol, frequency=frequency)

        # Get orbit
        orbit_df = nisar_orbit(h5_path)

        if mode == 0:
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            return prm, orbit_df

        # Mode 2: Also get SLC data
        slc_data = nisar_slc(h5_path, pol=pol, frequency=frequency)

        # No reramp_params for Nisar (stripmap mode)
        return prm_dict, orbit_df, slc_data, None
