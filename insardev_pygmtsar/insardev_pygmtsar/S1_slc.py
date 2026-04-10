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
            If the bursts contain inconsistencies, such as mismatched .tiff and .xml files, or if invalid filter parameters are provided.
        """
        import os
        from glob import glob
        import re
        import pandas as pd
        import geopandas as gpd
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        oneday = relativedelta(days=1)
        
        self.datadir = datadir
        self.DEM = DEM

        orbits = glob(self.pattern_orbit, root_dir=self.datadir)
        #print ('orbits', orbits)
        orbits_dict = {}
        # Extract validity dates from filename (no file I/O needed)
        # Pattern: S1A_OPER_AUX_POEORB_OPOD_20210207T122351_V20210117T225942_20210119T005942.EOF
        filename_pattern = re.compile(r'_V(\d{8})T\d{6}_(\d{8})T\d{6}\.EOF$')
        for orbit in orbits:
            match = filename_pattern.search(orbit)
            if match:
                validity_start = datetime.strptime(match.group(1), '%Y%m%d').date()
                validity_stop = datetime.strptime(match.group(2), '%Y%m%d').date()
                orbits_dict[(validity_start, validity_stop)] = orbit
        #print('orbits_dict', orbits_dict)
        
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
                # match orbit file
                date = start_time.date()
                orbit= (orbits_dict.get((date-oneday, date+oneday)) or
                                     orbits_dict.get((date-oneday, date)) or
                                     orbits_dict.get((date, date+oneday)) or
                                     orbits_dict.get((date, date)))
                # Build record from parsed annotation
                record = {
                    'fullBurstID': prefix,
                    'burst': burst_name,
                    'startTime': start_time,
                    'polarization': ann['polarisation'],
                    'flightDirection': ann['flightDirection'],
                    'pathNumber': ((int(ann['absoluteOrbitNumber']) - 73) % 175) + 1,
                    'subswath': ann['swath'],
                    'mission': ann['missionId'],
                    'beamModeType': ann['mode'],
                    'orbit': orbit,
                    'geometry': ann['geometry']
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        assert len(df), f'Bursts not found'
        df = gpd.GeoDataFrame(df, geometry='geometry')\
            .sort_values(by=['fullBurstID','polarization','burst'])\
            .set_index(['fullBurstID','polarization','burst'])

        path_numbers = df.pathNumber.unique().tolist()
        min_dates = [str(df[df.pathNumber==path].startTime.dt.date.min()) for path in path_numbers]
        if len(path_numbers) > 1:
            print (f'NOTE: Multiple path numbers found in the dataset: {", ".join(map(str, path_numbers))}.')
            print (f'NOTE: The following reference dates are available: {", ".join(min_dates)}.')
        print (f'NOTE: Loaded {len(df)} bursts.')
        self.df = df

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
