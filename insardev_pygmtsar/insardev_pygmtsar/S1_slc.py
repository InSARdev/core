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
from insardev_toolkit.utils_S1 import path_number, measurement_path
from insardev_toolkit.utils_files import exists
from insardev_toolkit import utils_eof


class S1_slc(Satellite):
    import xarray as xr

    pattern_prefix: str = '[0-9]*_[0-9]*_IW?'
    pattern_burst: str = 'S1_[0-9]*_IW?_[0-9]*T[0-9]*_[HV][HV]_*-BURST'
    pattern_orbit: str = 'S1?_OPER_AUX_???ORB_OPOD_[0-9]*_V[0-9]*_[0-9]*.EOF'

    def __init__(self, datadir: str, DEM: str|xr.DataArray|xr.Dataset|None=None):
        """
        Scans the specified directory for Sentinel-1 SLC (Single Look Complex) data and filters it based on the provided parameters.
    
        Parameters
        ----------
        datadir : str
            The directory containing the data files.
        DEMfilename : str, optional
            The filename of the DEM file.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing metadata about the found burst, including their paths and other relevant properties.
    
        Raises
        ------
        ValueError
            If the bursts contain inconsistencies, such as mismatched measurement (.nc or .tiff) and .xml files, or if invalid filter parameters are provided.
        """
        import os
        from glob import glob
        import pandas as pd
        import geopandas as gpd
        from datetime import datetime

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

        # mission, product, production and validity of the orbit files from their names; an empty file raises
        orbits = utils_eof.parse_files(glob(self.pattern_orbit, root_dir=self.datadir), root_dir=self.datadir)

        # scan directories with patterns
        prefixes = glob(self.pattern_prefix, root_dir=self.datadir)
        records = []
        for prefix in prefixes:
            #print('prefix', prefix)
            meta_dir = os.path.join(self.datadir, prefix, 'annotation')
            metas = glob(self.pattern_burst + '.xml', root_dir=meta_dir)
            #print('metas', metas)
            for meta in metas:
                #print('meta', meta)
                # the files of the burst that the processing reads; an empty one raises
                exists(os.path.join(meta_dir, meta))
                measurement_path(os.path.join(self.datadir, prefix, 'measurement'), os.path.splitext(meta)[0])
                for sub in ('calibration', 'noise'):
                    exists(os.path.join(self.datadir, prefix, sub, meta))
                ann = self.parse_annotation(os.path.join(meta_dir, meta))
                start_time = datetime.strptime(ann['startTime'], '%Y-%m-%dT%H:%M:%S.%f')
                # validate startTime matches burst name date (detect corrupted XML from parallel download race condition)
                burst_name = os.path.splitext(meta)[0]
                burst_date_str = burst_name.split('_')[3]  # e.g., '20210211T135237'
                expected_date = datetime.strptime(burst_date_str, '%Y%m%dT%H%M%S').date()
                if start_time.date() != expected_date:
                    raise ValueError(f'ERROR: Corrupted XML annotation for burst {burst_name}: '
                                   f'startTime {start_time.date()} does not match expected date {expected_date}. '
                                   f'This is likely caused by a race condition during parallel download. '
                                   f'Delete the corrupted files and re-download with n_jobs=1 or re-run the download.')
                # the orbit file of the burst's own mission that covers the burst time, a precise orbit first,
                # then the newest production, as EOF.download selects it; None when no file covers the burst
                orbit = utils_eof.select(orbits, ann['missionId'], start_time)
                # Build record from parsed annotation
                record = {
                    'fullBurstID': prefix,
                    'burst': burst_name,
                    'startTime': start_time,
                    'polarization': ann['polarisation'],
                    'flightDirection': ann['flightDirection'],
                    # the path of the burst ID, the same for the burst on every date and satellite
                    'pathNumber': path_number(burst_name.split('_')[1]),
                    'subswath': ann['swath'],
                    'mission': ann['missionId'],
                    # the radar band, C on every Sentinel-1 satellite (the NISAR records say L); the transform
                    # stores it with the other record attributes
                    'band': 'C',
                    'beamModeType': ann['mode'],
                    'orbit': orbit,
                    'geometry': ann['geometry']
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        assert len(df), f'Bursts not found'
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)\
            .sort_values(by=['fullBurstID','polarization','burst'])\
            .set_index(['fullBurstID','polarization','burst'])

        path_numbers = df.pathNumber.unique().tolist()
        min_dates = [str(df[df.pathNumber==path].startTime.dt.date.min()) for path in path_numbers]
        if len(path_numbers) > 1:
            print (f'NOTE: Multiple path numbers found in the dataset: {", ".join(map(str, path_numbers))}.')
            print (f'NOTE: The following reference dates are available: {", ".join(min_dates)}.')
        df = self._drop_duplicates(df)
        print (f'NOTE: Loaded {len(df)} bursts.')
        self.df = df
        # a record attribute, placed before the geometry, which is kept last as the long field
        self.df.insert(self.df.columns.get_loc('geometry'), 'BPR', self.baselines())

    def _drop_duplicates(self, df):
        """Ignore bursts that repeat an acquisition, keeping the first of them.

        An acquisition is one burst, one polarization and one time. The Sentinel-1 archive can
        distribute a single acquisition more than once, as separate products of the same
        datatake, holding the same image. Keeping them all puts one date twice in the stack,
        which is rejected only much later when the stack is loaded, so the repeats are dropped
        here and reported.

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            The scanned bursts.

        Returns
        -------
        geopandas.GeoDataFrame
            The bursts without the repeated acquisitions.
        """
        # the stack compares the dates by the second, so an acquisition is keyed the same way
        key = df.reset_index()[['fullBurstID', 'polarization']].assign(
            time=df.startTime.dt.floor('s').values)
        duplicated = key.duplicated(keep='first').values
        if not duplicated.any():
            return df

        bursts = df.index.get_level_values('burst')
        for idx in duplicated.nonzero()[0]:
            same = ((key.fullBurstID == key.fullBurstID.iloc[idx])
                    & (key.polarization == key.polarization.iloc[idx])
                    & (key.time == key.time.iloc[idx])).values.nonzero()[0][0]
            print(f'NOTE: {bursts[idx]} repeats the acquisition of {bursts[same]}, ignored.')
        print(f'NOTE: {int(duplicated.sum())} of {len(df)} bursts repeat an acquisition, '
              f'{len(df) - int(duplicated.sum())} left.')
        return df[~duplicated]

    def parse_annotation(self, filename: str) -> dict:
        """
        Parse XML annotation using ElementTree (fast, extracts only required fields).

        Parameters
        ----------
        filename : str
            The filename of the XML scene annotation.

        Returns
        -------
        dict
            Flat dict with metadata fields and geometry.
        """
        import xml.etree.ElementTree as ET
        from shapely.geometry import LineString, Polygon, MultiPolygon

        tree = ET.parse(filename)
        root = tree.getroot()

        # Extract adsHeader fields
        header = root.find('.//adsHeader')
        result = {
            'startTime': header.find('startTime').text,
            'polarisation': header.find('polarisation').text,
            'absoluteOrbitNumber': header.find('absoluteOrbitNumber').text,
            'swath': header.find('swath').text,
            'missionId': header.find('missionId').text,
            'mode': header.find('mode').text,
            'flightDirection': root.find('.//productInformation/pass').text,
        }

        # Extract geolocation grid points and build geometry
        geoloc_list = root.find('.//geolocationGridPointList')
        lines_dict = {}  # line_num -> [(lon, lat), ...]
        for gcp in geoloc_list.findall('geolocationGridPoint'):
            line = int(gcp.find('line').text)
            lon = float(gcp.find('longitude').text)
            lat = float(gcp.find('latitude').text)
            if line not in lines_dict:
                lines_dict[line] = []
            lines_dict[line].append((lon, lat))

        # Build polygons from consecutive lines
        bursts = []
        prev_coords = None
        for line_num in sorted(lines_dict.keys()):
            coords = lines_dict[line_num]
            if len(coords) > 1 and prev_coords is not None and len(prev_coords) > 1:
                bursts.append(Polygon([*prev_coords, *coords[::-1]]))
            prev_coords = coords

        result['geometry'] = MultiPolygon(bursts)
        return result
