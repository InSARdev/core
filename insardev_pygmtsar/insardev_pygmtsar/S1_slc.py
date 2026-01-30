# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_base import S1_base

class S1_slc(S1_base):
    import geopandas as gpd
    from shapely.geometry import MultiPolygon
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
        import pandas as pd
        import geopandas as gpd
        import shapely
        import numpy as np
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
        import re
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
                annotation = self.read_xml(os.path.join(meta_dir, meta))
                start_time = annotation['product']['adsHeader']['startTime']
                start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f')
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
                                     orbits_dict.get((date, date)))
                # Extract all required fields from annotation
                record = {
                    'fullBurstID': prefix,
                    'burst': os.path.splitext(meta)[0],
                    'startTime': start_time,
                    'polarization': annotation['product']['adsHeader']['polarisation'],
                    'flightDirection': annotation['product']['generalAnnotation']['productInformation']['pass'],
                    'pathNumber': ((int(annotation['product']['adsHeader']['absoluteOrbitNumber']) - 73) % 175) + 1,
                    'subswath': annotation['product']['adsHeader']['swath'],
                    'mission': annotation['product']['adsHeader']['missionId'],
                    'beamModeType': annotation['product']['adsHeader']['mode'],
                    'orbit': orbit,
                    'geometry': self.geoloc2geometry(annotation)
                }
                records.append(record)
                #print(record)
        
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

    def geoloc2geometry(self, annotation: dict) -> MultiPolygon:
        """
        Read approximate bursts locations from annotation
        """
        from shapely.geometry import LineString, Polygon, MultiPolygon
        df = self.get_geoloc(annotation)
        # this code line works for a single scene
        #lines = df.groupby('line')['geometry'].apply(lambda x: LineString(x.tolist()))
        # more complex code is required for stitched scenes processing with repeating 'line' series
        df['line_change'] = df['line'].diff().ne(0).cumsum()
        # single-point lines possible for stitched scenes
        grouped_lines = df.groupby('line_change')['geometry'].apply(lambda x: LineString(x.tolist()) if len(x) > 1 else None)
        lines = grouped_lines.reset_index(drop=True)
        #bursts = [Polygon([*line1.coords, *line2.coords[::-1]]) for line1, line2 in zip(lines[:-1], lines[1:])]
        # to ignore None for single-point lines
        bursts = []
        prev_line = None
        for line in lines:
            if line is not None and prev_line is not None:
                bursts.append(Polygon([*prev_line.coords, *line.coords[::-1]]))
            prev_line = line
        return MultiPolygon(bursts)

    def read_xml(self, filename: str) -> dict:
        """
        Return the XML scene annotation as a dictionary.

        Parameters
        ----------
        filename : str
            The filename of the XML scene annotation.

        Returns
        -------
        dict
            The XML scene annotation as a dictionary.
        """
        import xmltodict

        with open(filename) as fd:
            # fix wrong XML tags to process cropped scenes
            # GMTSAR assemble_tops.c produces malformed xml
            # https://github.com/gmtsar/gmtsar/issues/354
            #doc = xmltodict.parse(fd.read().replace('/></','></'))
            try:
                doc = xmltodict.parse(fd.read())
            except Exception as e:
                print(f'Error parsing XML file {filename}: {e}')
                raise e
        return doc

    def get_geoloc(self, annotation: dict) -> gpd.GeoDataFrame:
        """
        Build approximate scene polygons using Ground Control Points (GCPs) from XML scene annotation.

        Parameters
        ----------
        filename : str, optional
            The filename of the XML scene annotation. If None, print a note and return an empty DataFrame. Default is None.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing the approximate scene polygons.

        annotation = S1.read_annotation(filename)
        S1.get_geoloc(annotation)
        """
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import os

        geoloc = annotation['product']['geolocationGrid']['geolocationGridPointList']
        # check data consistency
        assert int(geoloc['@count']) == len(geoloc['geolocationGridPoint'])
    
        gcps = pd.DataFrame(geoloc['geolocationGridPoint'])
        # convert to numeric values excluding azimuthTime & slantRangeTime
        for column in gcps.columns[2:]:
            gcps[column] = pd.to_numeric(gcps[column])

        # return approximate location as set of GCP
        return gpd.GeoDataFrame(gcps, geometry=gpd.points_from_xy(x=gcps.longitude, y=gcps.latitude))
