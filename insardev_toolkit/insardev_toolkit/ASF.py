# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
from .progressbar_joblib import progressbar_joblib

class ASF(progressbar_joblib):
    import pandas as pd
    from datetime import timedelta

    def __init__(self, username=None, password=None):
        import asf_search
        import getpass
        if username is None:
            username = getpass.getpass('Please enter your ASF username and press Enter key:')
        if password is None:
            password = getpass.getpass('Please enter your ASF password and press Enter key:')
        self.username = username
        self.password = password

    def _get_asf_session(self):
        import asf_search
        return asf_search.ASFSession().auth_with_creds(self.username, self.password)

    @staticmethod
    def _detect_mission(granule_name):
        """Detect satellite mission from granule/burst name.

        Parameters
        ----------
        granule_name : str
            Granule identifier (S1 burst or NISAR granule).

        Returns
        -------
        str
            'S1' for Sentinel-1, 'NISAR' for NISAR.

        Raises
        ------
        ValueError
            If mission cannot be detected from name.
        """
        if granule_name.startswith('S1_') and granule_name.endswith('-BURST'):
            return 'S1'
        elif granule_name.startswith('NISAR_'):
            return 'NISAR'
        else:
            raise ValueError(f"Unknown mission for granule: {granule_name}. "
                           f"Expected S1_*-BURST or NISAR_* format.")

    @staticmethod
    def _normalize_polarization(polarization):
        """Convert polarization to list format.

        Parameters
        ----------
        polarization : None, str, or list
            Polarization specification.

        Returns
        -------
        list or None
            None if input is None, otherwise list of uppercase polarizations.
        """
        if polarization is None:
            return None
        if isinstance(polarization, str):
            return [polarization.upper()]
        return [p.upper() for p in polarization]

    @staticmethod
    def _burst_exists(basedir, burst):
        """
        Check if a burst is completely downloaded with all required files.

        Parameters
        ----------
        basedir : str
            Base directory containing burst data.
        burst : str
            Burst name like 'S1_370328_IW1_20150121T134421_VV_DBBE-BURST'.

        Returns
        -------
        bool
            True if all 4 files exist, are regular files, and have non-zero size.
        """
        import os
        from glob import glob

        # Extract burstId pattern from burst name (orbital path unknown, use wildcard)
        # burst: S1_370328_IW1_20150121T134421_VV_DBBE-BURST
        # burstId: 071_370328_IW1 (path number varies)
        parts = burst.split('_')
        burstid_pattern = f'*_{parts[1]}_{parts[2]}'

        # Find matching burstId directory
        matching_dirs = glob(burstid_pattern, root_dir=basedir)
        if not matching_dirs:
            return False
        if len(matching_dirs) > 1:
            raise ValueError(f'ERROR: Multiple burstId directories found for {burst}: {matching_dirs}. '
                           f'This indicates inconsistent data that cannot be processed.')

        burst_dir = os.path.join(basedir, matching_dirs[0])

        # Define expected file paths
        files = [
            os.path.join(burst_dir, 'measurement', f'{burst}.tiff'),
            os.path.join(burst_dir, 'annotation', f'{burst}.xml'),
            os.path.join(burst_dir, 'calibration', f'{burst}.xml'),
            os.path.join(burst_dir, 'noise', f'{burst}.xml'),
        ]

        # Check all files: exist, regular file, non-zero size
        for filepath in files:
            if not os.path.exists(filepath):
                return False
            if not os.path.isfile(filepath):
                return False
            if os.path.getsize(filepath) == 0:
                return False

        return True

    @staticmethod
    def _nisar_exists(basedir, granule_id, polarization):
        """Check if NISAR per-pol file exists.

        Parameters
        ----------
        basedir : str
            Base directory containing NISAR data.
        granule_id : str
            Full NISAR granule ID.
        polarization : str
            Single polarization to check (e.g., 'HH').

        Returns
        -------
        bool
            True if output file exists.
        """
        import os

        # Parse granule ID to get output filename
        # NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_...
        # Output: track_frame/NSR_172_008_20251204T024618_HH.h5
        parts = granule_id.replace('.h5', '').split('_')
        track = int(parts[5])  # 172
        frame = int(parts[7])  # 008
        datetime_str = parts[11][:15]  # 20251204T024618

        # Files are stored in track_frame subdirectory
        subdir = f"{track:03d}_{frame:03d}"
        out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{polarization}.h5"
        out_path = os.path.join(basedir, subdir, out_name)

        return os.path.exists(out_path)

    # https://asf.alaska.edu/datasets/data-sets/derived-data-sets/sentinel-1-bursts/
    def download(self, basedir, bursts, polarization=None, frequency=None, session=None, n_jobs=8, joblib_backend='loky', skip_exist=True,
                        retries=30, timeout_second=3, debug=False):
        """
        Download SAR data from ASF.

        Supports both Sentinel-1 bursts and NISAR RSLC granules. Mission is auto-detected
        from ID format.

        Parameters
        ----------
        basedir : str
            Output directory.
        bursts : str or list
            Burst/granule identifiers. Can be:
            - S1 burst: 'S1_262885_IW2_20190702T032452_VV_69C5-BURST'
            - NISAR RSLC: 'NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_...'
            - Newline-separated string of multiple IDs
        polarization : None, str, or list, optional
            Polarization(s) to download.
            - None: S1 uses pol from name, NISAR downloads all available
            - 'VV': Download only VV (S1) or 'HH' (NISAR)
            - ['VV', 'VH']: Download both polarizations
        frequency : None or str, optional (NISAR only)
            Which frequency band(s) to download:
            - None: Both frequencyA (20MHz) and frequencyB (5MHz) together (default)
            - 'A': Only frequencyA (high resolution, primary InSAR)
            - 'B': Only frequencyB (4x less data, quick look or iono correction)
        session : asf_search.ASFSession, optional
            Authenticated session. Created automatically if None.
        n_jobs : int, optional
            Parallel download jobs. Default 8.
        joblib_backend : str, optional
            Backend for parallel processing. Default 'loky'.
        skip_exist : bool, optional
            Skip already downloaded data. Default True.
        retries : int, optional
            Number of retry attempts. Default 30.
        timeout_second : int, optional
            Seconds between retries. Default 3.
        debug : bool, optional
            Print debug information. Default False.

        Returns
        -------
        pandas.DataFrame or None
            Downloaded files info, or None if all existed.

        Examples
        --------
        >>> asf = ASF('user', 'pass')
        >>> # S1: download VV only
        >>> asf.download('data/', 'S1_262885_IW2_20190702T032452_VV_69C5-BURST')
        >>> # S1: download both pols
        >>> asf.download('data/', 'S1_262885_IW2_20190702T032452_VV_69C5-BURST', polarization=['VV','VH'])
        >>> # NISAR: download HH with both frequencies (default, for iono correction)
        >>> asf.download('data/', 'NISAR_L1_PR_RSLC_006_172_A_008_...', polarization='HH')
        >>> # NISAR: download HH frequencyB only (quick look, 4x less data)
        >>> asf.download('data/', 'NISAR_L1_PR_RSLC_006_172_A_008_...', polarization='HH', frequency='B')
        >>> # NISAR: download HH frequencyA only (when iono correction not needed)
        >>> asf.download('data/', 'NISAR_L1_PR_RSLC_006_172_A_008_...', polarization='HH', frequency='A')
        """
        import pandas as pd
        import os

        # Normalize inputs
        if isinstance(bursts, str):
            bursts = list(filter(None, map(str.strip, bursts.split('\n'))))
        pols = self._normalize_polarization(polarization)

        # Create output directory
        os.makedirs(basedir, exist_ok=True)

        # Group by mission (auto-detect from ID format)
        s1_bursts = []
        nisar_granules = []
        for burst in bursts:
            mission = self._detect_mission(burst)
            if mission == 'S1':
                s1_bursts.append(burst)
            elif mission == 'NISAR':
                nisar_granules.append(burst)

        # Check if any downloads needed BEFORE creating session (avoid network call)
        if skip_exist:
            # Filter S1 bursts that need download
            s1_needed = [b for b in s1_bursts if not self._burst_exists(basedir, b)]
            # Filter NISAR granules that need download (check all requested pols)
            nisar_needed = []
            nisar_existing = []  # Track existing files for return
            for g in nisar_granules:
                check_pols = pols if pols else ['HH', 'HV', 'VH', 'VV']
                g_needed = False
                for p in check_pols:
                    if not self._nisar_exists(basedir, g, p):
                        g_needed = True
                    else:
                        # Build existing file name
                        parts = g.replace('.h5', '').split('_')
                        track = int(parts[5])
                        frame = int(parts[7])
                        datetime_str = parts[11][:15]
                        nisar_existing.append(f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{p}.h5")
                if g_needed:
                    nisar_needed.append(g)
        else:
            s1_needed = s1_bursts
            nisar_needed = nisar_granules
            nisar_existing = []

        # Return early if nothing to download (no network call!)
        if not s1_needed and not nisar_needed:
            # Return existing files as DataFrame (NISAR only for now)
            if nisar_existing:
                return pd.DataFrame({'file': nisar_existing})
            return None

        # Prepare session only when actually needed
        if session is None:
            session = self._get_asf_session()

        results = []

        # Download S1 bursts
        if s1_needed:
            df = self._download_s1(basedir, s1_needed, pols, session,
                                   n_jobs, joblib_backend, skip_exist, retries, timeout_second, debug)
            if df is not None:
                results.append(df)

        # Download NISAR granules
        if nisar_needed:
            df = self._download_nisar(basedir, nisar_needed, pols, frequency, session,
                                       n_jobs, joblib_backend, skip_exist, retries, timeout_second, debug)
            if df is not None:
                results.append(df)

        if results:
            return pd.concat(results, ignore_index=True)
        return None

    def _download_s1(self, basedir, bursts, polarizations, session, n_jobs,
                      joblib_backend, skip_exist, retries, timeout_second, debug):
        """Internal: Download Sentinel-1 bursts.

        Parameters
        ----------
        bursts : list
            List of S1 burst IDs.
        polarizations : list or None
            If None, use polarization from burst name.
            If list, replace polarization in burst name with each requested pol.
        """
        import rioxarray as rio
        from tifffile import TiffFile
        import xmltodict
        from xml.etree import ElementTree
        import pandas as pd
        import asf_search
        import joblib
        from tqdm.auto import tqdm
        import os
        from datetime import datetime, timedelta
        import time
        import warnings
        # supress asf_search 'UserWarning: File already exists, skipping download'
        warnings.filterwarnings("ignore", category=UserWarning)

        # Expand bursts by polarization if specified
        if polarizations is not None:
            expanded_bursts = []
            for burst in bursts:
                # S1_262885_IW2_20190702T032452_VV_69C5-BURST
                #                              ^^ pol at position 4
                parts = burst.split('_')
                for pol in polarizations:
                    new_parts = parts.copy()
                    new_parts[4] = pol  # Replace polarization
                    new_burst = '_'.join(new_parts)
                    expanded_bursts.append(new_burst)
            # Remove duplicates while preserving order
            seen = set()
            bursts = [b for b in expanded_bursts if not (b in seen or seen.add(b))]

        def filter_azimuth_time(items, start_utc_dt, stop_utc_dt, delta=3):
            return [item for item in items if
                 datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') >= start_utc_dt - timedelta(seconds=delta) and
                 datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') <= stop_utc_dt + timedelta(seconds=delta)]

        # skip existing bursts (check all 4 files: tiff + 3 xml, regular files, non-zero size)
        if skip_exist:
            bursts_missed = [burst for burst in bursts if not self._burst_exists(basedir, burst)]
        else:
            bursts_missed = bursts
        # do not use internet connection, work offline when all the scenes already available
        if len(bursts_missed) == 0:
            return None

        def download_burst(result, basedir, session):
            properties = result.geojson()['properties']
            #print ('result properties', properties)
            burst = properties['fileID']
            burstId = properties['burst']['fullBurstID']
            burstIndex = properties['burst']['burstIndex']
            platform = properties['platform'][-2:]
            polarization = properties['polarization']
            #print ('polarization', polarization)
            subswath = properties['burst']['subswath']

            # create the directories if needed
            burst_dir = os.path.join(basedir, burstId)
            tif_dir = os.path.join(burst_dir, 'measurement')
            xml_annot_dir = os.path.join(burst_dir, 'annotation')
            xml_noise_dir = os.path.join(burst_dir, 'noise')
            xml_calib_dir = os.path.join(burst_dir, 'calibration')
            # save annotation using the burst and scene names
            xml_file = os.path.join(xml_annot_dir, f'{burst}.xml')
            xml_noise_file = os.path.join(xml_noise_dir, f'{burst}.xml')
            xml_calib_file = os.path.join(xml_calib_dir, f'{burst}.xml')
            #rint ('xml_file', xml_file)
            tif_file = os.path.join(tif_dir, f'{burst}.tiff')
            #print ('tif_file', tif_file)
            for dirname in [burst_dir, tif_dir, xml_annot_dir, xml_noise_dir, xml_calib_dir]:
                os.makedirs(dirname, exist_ok=True)

            # check if all files already exist
            all_exist = (os.path.exists(tif_file) and os.path.getsize(tif_file) >= int(properties['bytes'])
                        and os.path.exists(xml_file) and os.path.getsize(xml_file) > 0
                        and os.path.exists(xml_noise_file) and os.path.getsize(xml_noise_file) > 0
                        and os.path.exists(xml_calib_file) and os.path.getsize(xml_calib_file) > 0)

            if all_exist:
                # validate existing TIFF dimensions using local annotation XML
                with open(xml_file, 'r') as f:
                    local_annotation = xmltodict.parse(f.read())['product']
                lines_per_burst = int(local_annotation['swathTiming']['linesPerBurst'])
                samples_per_burst = int(local_annotation['imageAnnotation']['imageInformation']['numberOfSamples'])
                with TiffFile(tif_file) as tif:
                    page = tif.pages[0]
                    actual_lines, actual_samples = page.shape
                if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                    raise Exception(f'ERROR: Existing TIFF dimensions mismatch for {burst}: '
                                  f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                  f'Delete the corrupted file and re-download.')
                # all files valid, skip download
                return

            # download manifest to memory to get dimensions for TIFF validation
            manifest_url = properties['additionalUrls'][0]
            response = session.get(manifest_url)
            response.raise_for_status()
            xml_content = response.text
            if len(xml_content) == 0:
                raise Exception(f'ERROR: Downloaded manifest is empty: {manifest_url}')
            # check XML file validity by parsing it
            _ = ElementTree.fromstring(xml_content)

            subswathidx = int(subswath[-1:]) - 1
            content = xmltodict.parse(xml_content)['burst']['metadata']['product'][subswathidx]
            assert polarization == content['polarisation'], 'ERROR: XML polarization differs from burst polarization'
            annotation = content['content']

            # get dimensions from manifest
            lines_per_burst = int(annotation['swathTiming']['linesPerBurst'])
            samples_per_burst = int(annotation['swathTiming']['samplesPerBurst'])

            annotation_burst = annotation['swathTiming']['burstList']['burst'][burstIndex]
            start_utc = annotation_burst['azimuthTime']
            start_utc_dt = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')

            # validate startTime matches burst name date (detect manifest mix-up)
            burst_date_str = burst.split('_')[3]  # e.g., '20210211T135237'
            expected_date = datetime.strptime(burst_date_str, '%Y%m%dT%H%M%S').date()
            if start_utc_dt.date() != expected_date:
                raise Exception(f'ERROR: Manifest data mismatch for burst {burst}: '
                              f'parsed startTime {start_utc_dt.date()} does not match expected date {expected_date}. '
                              f'This indicates corrupted manifest data.')

            # download tif if needed
            if os.path.exists(tif_file) and os.path.getsize(tif_file) >= int(properties['bytes']):
                # validate existing file dimensions
                with TiffFile(tif_file) as tif:
                    page = tif.pages[0]
                    actual_lines, actual_samples = page.shape
                if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                    raise Exception(f'ERROR: Existing TIFF dimensions mismatch for {burst}: '
                                  f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                  f'Delete the corrupted file and re-download.')
            else:
                # Download and validate TIFF entirely in memory before writing to disk
                import io
                import rasterio
                from rasterio.io import MemoryFile

                # Download TIFF to memory
                tiff_url = properties['url']
                response = session.get(tiff_url)
                response.raise_for_status()
                tiff_bytes = response.content
                if len(tiff_bytes) == 0:
                    raise Exception(f'ERROR: Downloaded TIFF is empty: {tiff_url}')

                # Validate TIFF structure and dimensions in memory using TiffFile
                with TiffFile(io.BytesIO(tiff_bytes)) as tif:
                    page = tif.pages[0]
                    actual_lines, actual_samples = page.shape
                    if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                        raise Exception(f'ERROR: Downloaded TIFF dimensions mismatch for {burst}: '
                                      f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                      f'ASF burst extraction may have failed.')
                    # Also get offset for XML creation
                    tiff_offset = page.dataoffsets[0]

                # Validate TIFF can be read by rasterio/GDAL (detects corruption)
                with MemoryFile(tiff_bytes) as memfile:
                    with memfile.open() as ds:
                        if ds.width != samples_per_burst or ds.height != lines_per_burst:
                            raise Exception(f'ERROR: Rasterio dimensions mismatch for {burst}')
                        # Read a small portion to verify data is accessible
                        _ = ds.read(1, window=rasterio.windows.Window(0, 0, min(100, ds.width), min(100, ds.height)))

                # TIFF validated - now build XML content in memory before writing anything

            # Build XML content in memory (or skip if files exist)
            xml_contents = {}  # {filepath: content_string}
            need_xml = not (os.path.exists(xml_file) and os.path.getsize(xml_file) > 0
                           and os.path.exists(xml_noise_file) and os.path.getsize(xml_noise_file) > 0
                           and os.path.exists(xml_calib_file) and os.path.getsize(xml_calib_file) > 0)

            if need_xml:
                # Get TIFF offset (already have it if we downloaded, otherwise read from existing file)
                if 'tiff_offset' not in dir():
                    with TiffFile(tif_file) as tif:
                        page = tif.pages[0]
                        tiff_offset = page.dataoffsets[0]
                offset = tiff_offset

                azimuth_time_interval = annotation['imageAnnotation']['imageInformation']['azimuthTimeInterval']
                burst_time_interval = timedelta(seconds=(lines_per_burst - 1) * float(azimuth_time_interval))
                stop_utc_dt = start_utc_dt + burst_time_interval
                stop_utc = stop_utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
                #print ('stop_utc', stop_utc, stop_utc_dt)

                # output xml
                product = {}

                adsHeader = annotation['adsHeader']
                adsHeader['startTime'] = start_utc
                adsHeader['stopTime'] = stop_utc
                adsHeader['imageNumber'] = '001'
                product = product   | {'adsHeader': adsHeader}

                qualityInformation = {'productQualityIndex': annotation['qualityInformation']['productQualityIndex']} |\
                                      {'qualityDataList':     annotation['qualityInformation']['qualityDataList']}
                product = product   | {'qualityInformation': qualityInformation}

                generalAnnotation = annotation['generalAnnotation']
                # filter annotation['generalAnnotation']['replicaInformationList'] by azimuthTime
                product = product   | {'generalAnnotation': generalAnnotation}

                imageAnnotation = annotation['imageAnnotation']
                imageAnnotation['imageInformation']['productFirstLineUtcTime'] = start_utc
                imageAnnotation['imageInformation']['productLastLineUtcTime'] = stop_utc
                imageAnnotation['imageInformation']['productComposition'] = 'Assembled'
                imageAnnotation['imageInformation']['sliceNumber'] = '0'
                imageAnnotation['imageInformation']['sliceList'] = {'@count': '0'}
                imageAnnotation['imageInformation']['numberOfLines'] = str(lines_per_burst)
                # imageStatistics and inputDimensionsList are not updated
                product = product   | {'imageAnnotation': imageAnnotation}

                dopplerCentroid = annotation['dopplerCentroid']
                items = filter_azimuth_time(dopplerCentroid['dcEstimateList']['dcEstimate'], start_utc_dt, stop_utc_dt)
                dopplerCentroid['dcEstimateList'] = {'@count': len(items), 'dcEstimate': items}
                product = product   | {'dopplerCentroid': dopplerCentroid}

                antennaPattern = annotation['antennaPattern']
                items = filter_azimuth_time(antennaPattern['antennaPatternList']['antennaPattern'], start_utc_dt, stop_utc_dt)
                antennaPattern['antennaPatternList'] = {'@count': len(items), 'antennaPattern': items}
                product = product   | {'antennaPattern': antennaPattern}

                swathTiming = annotation['swathTiming']
                items = filter_azimuth_time(swathTiming['burstList']['burst'], start_utc_dt, start_utc_dt, 1)
                assert len(items) == 1, 'ERROR: unexpected bursts count, should be 1'
                # add TiFF file information
                items[0]['byteOffset'] = offset
                swathTiming['burstList'] = {'@count': len(items), 'burst': items}
                product = product   | {'swathTiming': swathTiming}

                geolocationGrid = annotation['geolocationGrid']
                items = filter_azimuth_time(geolocationGrid['geolocationGridPointList']['geolocationGridPoint'], start_utc_dt, stop_utc_dt, 1)
                # re-numerate line numbers for the burst
                for item in items: item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                geolocationGrid['geolocationGridPointList'] = {'@count': len(items), 'geolocationGridPoint': items}
                product = product   | {'geolocationGrid': geolocationGrid}

                product = product   | {'coordinateConversion': annotation['coordinateConversion']}
                product = product   | {'swathMerging': annotation['swathMerging']}

                xml_contents[xml_file] = xmltodict.unparse({'product': product}, pretty=True, indent='  ')

                # output noise xml
                content = xmltodict.parse(xml_content)['burst']['metadata']['noise'][subswathidx]
                assert polarization == content['polarisation'], 'ERROR: XML polarization differs from burst polarization'
                annotation = content['content']

                noise = {}

                adsHeader = annotation['adsHeader']
                adsHeader['startTime'] = start_utc
                adsHeader['stopTime'] = stop_utc
                adsHeader['imageNumber'] = '001'
                noise = noise   | {'adsHeader': adsHeader}

                if 'noiseVectorList' in annotation:
                    noiseRangeVector = annotation['noiseVectorList']
                    items = filter_azimuth_time(noiseRangeVector['noiseVector'], start_utc_dt, stop_utc_dt)
                    # re-numerate line numbers for the burst
                    for item in items: item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                    noiseRangeVector = {'@count': len(items), 'noiseVector': items}
                    noise = noise   | {'noiseVectorList': noiseRangeVector}

                if 'noiseRangeVectorList' in annotation:
                    noiseRangeVector = annotation['noiseRangeVectorList']
                    items = filter_azimuth_time(noiseRangeVector['noiseRangeVector'], start_utc_dt, stop_utc_dt)
                    # re-numerate line numbers for the burst
                    for item in items: item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                    noiseRangeVector = {'@count': len(items), 'noiseRangeVector': items}
                    noise = noise   | {'noiseRangeVectorList': noiseRangeVector}

                if 'noiseAzimuthVectorList' in annotation:
                    noiseAzimuthVector = annotation['noiseAzimuthVectorList']
                    items = noiseAzimuthVector['noiseAzimuthVector']['line']['#text'].split(' ')
                    items = [int(item) for item in items]
                    lowers = [item for item in items if item <= burstIndex * lines_per_burst] or items[0]
                    uppers = [item for item in items if item >= (burstIndex + 1) * lines_per_burst - 1] or items[-1]
                    mask = [True if item>=lowers[-1] and item<=uppers[0] else False for item in items]
                    items = [item - burstIndex * lines_per_burst for item, m in zip(items, mask) if m]
                    noiseAzimuthVector['noiseAzimuthVector']['firstAzimuthLine'] = lowers[-1] - burstIndex * lines_per_burst
                    noiseAzimuthVector['noiseAzimuthVector']['lastAzimuthLine'] = uppers[0] - burstIndex * lines_per_burst
                    noiseAzimuthVector['noiseAzimuthVector']['line'] = {'@count': len(items), '#text': ' '.join([str(item) for item in items])}
                    items = noiseAzimuthVector['noiseAzimuthVector']['noiseAzimuthLut']['#text'].split(' ')
                    items = [item for item, m in zip(items, mask) if m]
                    noiseAzimuthVector['noiseAzimuthVector']['noiseAzimuthLut'] = {'@count': len(items), '#text': ' '.join(items)}
                    noise = noise   | {'noiseAzimuthVectorList': noiseAzimuthVector}

                xml_contents[xml_noise_file] = xmltodict.unparse({'noise': noise}, pretty=True, indent='  ')

                # output calibration xml
                content = xmltodict.parse(xml_content)['burst']['metadata']['calibration'][subswathidx]
                assert polarization == content['polarisation'], 'ERROR: XML polarization differs from burst polarization'
                annotation = content['content']

                calibration = {}

                adsHeader = annotation['adsHeader']
                adsHeader['startTime'] = start_utc
                adsHeader['stopTime'] = stop_utc
                adsHeader['imageNumber'] = '001'
                calibration = calibration   | {'adsHeader': adsHeader}

                calibration = calibration   | {'calibrationInformation': annotation['calibrationInformation']}

                calibrationVector = annotation['calibrationVectorList']
                items = filter_azimuth_time(calibrationVector['calibrationVector'], start_utc_dt, stop_utc_dt)
                # re-numerate line numbers for the burst
                for item in items: item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                calibrationVector = {'@count': len(items), 'calibrationVector': items}
                calibration = calibration   | {'calibrationVectorList': calibrationVector}

                xml_contents[xml_calib_file] = xmltodict.unparse({'calibration': calibration}, pretty=True, indent='  ')

            # All validations passed - now write everything to disk atomically
            # Write TIFF if we downloaded it (tiff_bytes exists in local scope)
            if 'tiff_bytes' in dir():
                with open(tif_file, 'wb') as f:
                    f.write(tiff_bytes)

            # Write all XML files
            for filepath, content in xml_contents.items():
                with open(filepath, 'w') as f:
                    f.write(content)

        with tqdm(desc=f'Downloading ASF Catalog'.ljust(25), total=1) as pbar:
            results = asf_search.granule_search(bursts_missed)
            pbar.update(1)

        # Check for conflicting bursts from different paths with same burstNum_subswath pattern
        # Such data cannot be stored in the same basedir without conflicts
        pattern_to_fullburstid = {}
        for result in results:
            props = result.geojson()['properties']
            full_burst_id = props['burst']['fullBurstID']  # e.g., '071_151226_IW3'
            # Extract pattern without path: '151226_IW3'
            parts = full_burst_id.split('_')
            pattern = f'{parts[1]}_{parts[2]}'
            if pattern in pattern_to_fullburstid:
                if pattern_to_fullburstid[pattern] != full_burst_id:
                    raise ValueError(f'ERROR: Conflicting bursts from different paths: '
                                   f'{pattern_to_fullburstid[pattern]} and {full_burst_id} '
                                   f'both match pattern *_{pattern}. '
                                   f'Download bursts from different paths into separate directories.')
            else:
                pattern_to_fullburstid[pattern] = full_burst_id

        if n_jobs is None or debug == True:
            print ('Note: sequential joblib processing is applied when "n_jobs" is None or "debug" is True.')
            joblib_backend = 'sequential'

        def download_burst_with_retry(result, basedir, session, retries, timeout_second):
            for retry in range(retries):
                try:
                    download_burst(result, basedir, session)
                    return True
                except Exception as e:
                    print(f'ERROR: download attempt {retry+1} failed for {result}: {e}')
                    if retry + 1 == retries:
                        return False
                time.sleep(timeout_second)

        # download bursts
        with self.progressbar_joblib(tqdm(desc='Downloading ASF SLC'.ljust(25), total=len(bursts_missed))) as progress_bar:
            statuses = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_burst_with_retry)\
                                    (result, basedir, session, retries=retries, timeout_second=timeout_second) for result in results)

        failed_count = statuses.count(False)
        if failed_count > 0:
            raise Exception(f'Bursts downloading failed for {failed_count} items.')
        # parse processed bursts and convert to dataframe
        bursts_downloaded = pd.DataFrame(bursts_missed, columns=['burst'])
        # return the results in a user-friendly dataframe
        return bursts_downloaded

    def _download_nisar(self, basedir, granules, polarizations, frequency, session,
                         n_jobs, joblib_backend, skip_exist, retries, timeout_second, debug):
        """Internal: Download NISAR RSLC granules with per-polarization output.

        Uses single HTTP Range request approach (verified 9.2 min for one pol):
        1. Query chunk byte offsets from remote HDF5 (metadata only, ~10 sec)
        2. Download entire byte span in ONE HTTP Range request (~9 min)
        3. Parse chunks locally and write output HDF5 (~2 sec)

        This is ~46x faster than per-chunk HTTP requests.
        Uses joblib for parallel downloads when n_jobs > 1.

        Output naming: NSR_{track}_{frame}_{datetime}_{pol}.h5
        (same naming for all frequency modes - check datasets to see what's included)

        Parameters
        ----------
        granules : list
            List of NISAR granule IDs.
        polarizations : list or None
            If None, download all available polarizations.
            If list, download only specified polarizations.
        frequency : str or None
            If None, download both frequencyA and frequencyB together (default).
            If 'A', download only frequencyA (20 MHz, high resolution).
            If 'B', download only frequencyB (5 MHz, 4x less data, for quick look).
        """
        import h5py
        import fsspec
        import aiohttp
        import requests
        import numpy as np
        import pandas as pd
        import asf_search
        from tqdm.auto import tqdm
        from io import BytesIO
        from datetime import datetime, timedelta
        import os
        import time
        import threading

        # Initialize tqdm lock for thread-safe progress bars
        tqdm.set_lock(threading.RLock())

        # Batch ASF search for all granules at once (instead of per-granule)
        # Show connecting status first, then search status
        granule_ids = [g.replace('.h5', '') for g in granules]
        print(f"Connecting to ASF...", end='\r', flush=True)
        search_results = asf_search.granule_search(granule_ids)
        print(f"Searching ASF for {len(granules)} granule(s)... done ({len(search_results)} found)")

        # Build URL lookup: granule_id -> url
        url_lookup = {}
        for result in search_results:
            props = result.geojson()['properties']
            # Match by granule name (without .h5)
            gid = props['fileID'].replace('.h5', '')
            url_lookup[gid] = props['url']

        # Verify all granules found
        missing = [g for g in granule_ids if g not in url_lookup]
        if missing:
            raise ValueError(f"Granules not found in ASF: {missing}")

        def get_chunk_info(h5_remote, pol, frequency='A'):
            """Query chunk byte offsets from remote HDF5.

            Returns dict with chunk info and byte range.
            """
            slc = h5_remote[f'science/LSAR/RSLC/swaths/frequency{frequency}/{pol}']
            shape = slc.shape
            chunk_shape = slc.chunks

            n_az = (shape[0] + chunk_shape[0] - 1) // chunk_shape[0]
            n_rg = (shape[1] + chunk_shape[1] - 1) // chunk_shape[1]

            chunks = []
            for row in range(n_az):
                for col in range(n_rg):
                    coord = (row * chunk_shape[0], col * chunk_shape[1])
                    info = slc.id.get_chunk_info_by_coord(coord)
                    chunks.append({
                        'offset': info.byte_offset,
                        'size': info.size,
                        'row': row,
                        'col': col,
                        'coord': coord
                    })

            min_offset = min(c['offset'] for c in chunks)
            max_end = max(c['offset'] + c['size'] for c in chunks)

            return {
                'chunks': chunks,
                'min_offset': min_offset,
                'max_end': max_end,
                'shape': shape,
                'chunk_shape': chunk_shape,
                'dtype': slc.dtype,
                'compression': slc.compression,
                'compression_opts': slc.compression_opts,
                'shuffle': slc.shuffle,  # Important: shuffle filter must match for write_direct_chunk
                'n_az': n_az,
                'n_rg': n_rg
            }

        def download_byte_range(url, start, end, auth_tuple, pbar=None, http_session=None):
            """Download byte range using single HTTP Range request.

            If pbar is provided, updates it instead of creating a new one.
            If http_session is provided, reuses the connection.
            """
            size = end - start
            headers = {'Range': f'bytes={start}-{end-1}'}  # HTTP Range is inclusive

            chunks_data = []
            if http_session:
                r = http_session.get(url, headers=headers, stream=True)
            else:
                r = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
            with r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024*1024):
                    chunks_data.append(chunk)
                    if pbar:
                        pbar.update(len(chunk))

            return b''.join(chunks_data)

        def parse_nisar_granule_id(granule_id):
            """Parse NISAR granule ID to extract track, frame, and datetime.

            Format: NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_20251204T024653_X05007_N_F_J_001.h5
                    ^product^     ^cycle^track^dir^frame^...^pols^..^start_time^     ^end_time^
            """
            parts = granule_id.replace('.h5', '').split('_')
            # parts[5] = track (172), parts[7] = frame (008), parts[11] = start datetime
            track = int(parts[5])
            frame = int(parts[7])
            # Start time: 20251204T024618 -> 20251204T024618
            datetime_str = parts[11]
            return track, frame, datetime_str

        def download_nisar_granule(granule_id, basedir, polarizations, skip_exist, debug, position=None):
            """Download single NISAR granule, split by polarization.

            position: tqdm position for parallel downloads (None for sequential)
            """

            # 1. Parse granule ID for output naming (fast, no network)
            track, frame, datetime_str = parse_nisar_granule_id(granule_id)

            # 2. Construct output paths to check existence early
            subdir = f"{track:03d}_{frame:03d}"
            out_dir = os.path.join(basedir, subdir)

            # 3. Check if all requested files already exist (skip remote HDF5 access)
            pols_to_download = polarizations if polarizations else ['HH', 'HV', 'VH', 'VV']

            if skip_exist:
                existing_files = []
                for pol in pols_to_download:
                    out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                    out_path = os.path.join(out_dir, out_name)
                    if os.path.exists(out_path):
                        existing_files.append(out_name)

                if len(existing_files) == len(pols_to_download):
                    if debug:
                        print(f"NISAR {track}_{frame}: all files exist, skipping download")
                    return existing_files

            # 4. Get URL from pre-fetched lookup (batch search done at start)
            short_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}"
            granule_search_id = granule_id.replace('.h5', '')
            url = url_lookup[granule_search_id]

            if debug:
                print(f"NISAR URL: {url}")

            # 5. Setup auth and session for connection reuse
            # Use ASF session which handles OAuth for Earthdata Cloud
            auth_tuple = (self.username, self.password)
            http_session = self._get_asf_session()

            # Get file size (HEAD request) and detect layout
            head_resp = http_session.head(url, allow_redirects=True)
            file_size = int(head_resp.headers['Content-Length'])

            # Fast layout detection (8MB download) - also returns probe buffer for reuse
            layout, probe_buffer = self._detect_nisar_layout_fast(url, auth_tuple, file_size, http_session=http_session)

            # Get available polarizations from probe buffer (no remote access)
            from io import BytesIO
            with h5py.File(BytesIO(bytes(probe_buffer)), 'r') as h5_probe:
                swaths_path = 'science/LSAR/RSLC/swaths'
                if swaths_path not in h5_probe:
                    raise ValueError(
                        f"Unsupported NISAR file format: '{swaths_path}' not found in {granule_id}. "
                        f"This may be an older simulated scene with incompatible structure."
                    )
                swaths_grp = h5_probe[swaths_path]

                # Check for frequencyA (required for current implementation)
                if 'frequencyA' not in swaths_grp:
                    available_keys = list(swaths_grp.keys())
                    raise ValueError(
                        f"Unsupported NISAR file format: 'frequencyA' not found in {granule_id}. "
                        f"Available keys in swaths: {available_keys}. "
                        f"This may be an older simulated scene with incompatible structure."
                    )

                swaths = swaths_grp['frequencyA']
                available_pols = [k for k in swaths.keys() if k in ['HH', 'HV', 'VH', 'VV']]

                if not available_pols:
                    raise ValueError(
                        f"No polarization data found in {granule_id}/frequencyA. "
                        f"Available keys: {list(swaths.keys())}. "
                        f"This may be an older simulated scene with incompatible structure."
                    )

                has_freq_b = 'frequencyB' in swaths_grp
                if has_freq_b:
                    freq_b_pols = [k for k in swaths_grp['frequencyB'].keys()
                                   if k in ['HH', 'HV', 'VH', 'VV']]
                else:
                    freq_b_pols = []

            # Determine which pols to download
            if polarizations is None:
                pols_to_download = available_pols
            else:
                pols_to_download = [p for p in polarizations if p in available_pols]
                missing = set(polarizations) - set(available_pols)
                if missing:
                    print(f"WARNING: Polarizations {missing} not available in {granule_id}")

            # Update frequency download flags based on actual availability
            download_freq_a = frequency is None or frequency == 'A'
            download_freq_b = (frequency is None or frequency == 'B') and has_freq_b

            if debug and position is None:
                if frequency is None:
                    freq_str = 'A+B' if has_freq_b else 'A'
                else:
                    freq_str = frequency
                print(f"NISAR {track}_{frame}: downloading {pols_to_download} (frequency{freq_str})")

            if layout == 'A':
                # Layout A: metadata at start - NO FSSPEC NEEDED
                # Get chunk info from 8MB probe
                from io import BytesIO
                with h5py.File(BytesIO(bytes(probe_buffer)), 'r') as h5_probe:
                    # Find where ANY SLC starts (to determine metadata boundary)
                    slc_min = float('inf')
                    for freq in ['A', 'B']:
                        freq_path = f'science/LSAR/RSLC/swaths/frequency{freq}'
                        if freq_path not in h5_probe:
                            continue
                        for p in ['HH', 'HV', 'VH', 'VV']:
                            slc_path = f'{freq_path}/{p}'
                            if slc_path in h5_probe:
                                slc = h5_probe[slc_path]
                                if slc.chunks:
                                    info = slc.id.get_chunk_info(0)  # First chunk
                                    slc_min = min(slc_min, info.byte_offset)

                    # Get chunk info for pols we're actually downloading
                    all_chunk_info = {}
                    for pol in pols_to_download:
                        chunk_info_a = None
                        chunk_info_b = None
                        if download_freq_a:
                            chunk_info_a = get_chunk_info(h5_probe, pol, 'A')
                        if download_freq_b and pol in freq_b_pols:
                            chunk_info_b = get_chunk_info(h5_probe, pol, 'B')
                        all_chunk_info[pol] = (chunk_info_a, chunk_info_b)

                # Use slc_min with small 5% margin for HDF5 internal structures
                # (safe because _read_nisar_metadata_direct skips other-pol paths)
                metadata_end = int(slc_min * 1.05)

                if debug and position is None:
                    print(f"  File size: {file_size/1e9:.2f} GB, Layout: {layout}")
                    print(f"  Metadata size: {metadata_end/1e6:.1f} MB")

                # Download metadata block with progress bar
                with tqdm(total=metadata_end, unit='B', unit_scale=True,
                          desc=f"{short_name} metadata", leave=False,
                          dynamic_ncols=False, ncols=80, mininterval=0.3,
                          disable=(position is not None)) as meta_pbar:
                    metadata_buffer = self._download_nisar_metadata_start(
                        url, auth_tuple, metadata_end, pbar=meta_pbar if position is None else None,
                        http_session=http_session)

            else:
                # Layout B not supported yet
                raise NotImplementedError(f"NISAR Layout B (metadata at end) not supported: {granule_id}")

            # Calculate total SLC size
            total_slc_size = 0
            for pol, (chunk_info_a, chunk_info_b) in all_chunk_info.items():
                if chunk_info_a:
                    total_slc_size += chunk_info_a['max_end'] - chunk_info_a['min_offset']
                if chunk_info_b:
                    total_slc_size += chunk_info_b['max_end'] - chunk_info_b['min_offset']

            # Check for empty SLC data (incompatible format)
            if total_slc_size == 0:
                raise ValueError(
                    f"No SLC data found in {granule_id}. "
                    f"Requested polarizations: {pols_to_download}. "
                    f"This may be an older simulated scene with incompatible structure."
                )

            if debug and position is None:
                for pol, (chunk_info_a, chunk_info_b) in all_chunk_info.items():
                    if chunk_info_a:
                        total_data = sum(c['size'] for c in chunk_info_a['chunks'])
                        span_size = chunk_info_a['max_end'] - chunk_info_a['min_offset']
                        overhead = (span_size - total_data) / total_data * 100
                        print(f"    FreqA: {len(chunk_info_a['chunks'])} chunks, "
                              f"Data: {total_data/1e9:.2f} GB, "
                              f"Span: {span_size/1e9:.2f} GB ({overhead:.1f}% overhead)")
                    if chunk_info_b:
                        total_data_b = sum(c['size'] for c in chunk_info_b['chunks'])
                        span_size_b = chunk_info_b['max_end'] - chunk_info_b['min_offset']
                        overhead_b = (span_size_b - total_data_b) / total_data_b * 100
                        print(f"    FreqB: {len(chunk_info_b['chunks'])} chunks, "
                              f"Data: {total_data_b/1e9:.2f} GB, "
                              f"Span: {span_size_b/1e9:.2f} GB ({overhead_b:.1f}% overhead)")

            # Create progress bar for SLC download
            desc = f"{short_name}"
            with tqdm(total=total_slc_size, unit='B', unit_scale=True, desc=desc,
                      position=position, leave=True,
                      dynamic_ncols=False, ncols=80, mininterval=0.3) as pbar:

                # Download and write each polarization
                downloaded_files = []
                os.makedirs(out_dir, exist_ok=True)

                for pol in pols_to_download:
                    out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                    out_path = os.path.join(out_dir, out_name)

                    chunk_info_a, chunk_info_b = all_chunk_info[pol]

                    if debug and position is None:
                        if chunk_info_a:
                            total_data = sum(c['size'] for c in chunk_info_a['chunks'])
                            span_size = chunk_info_a['max_end'] - chunk_info_a['min_offset']
                            overhead = (span_size - total_data) / total_data * 100
                            print(f"    FreqA: {len(chunk_info_a['chunks'])} chunks, "
                                  f"Data: {total_data/1e9:.2f} GB, "
                                  f"Span: {span_size/1e9:.2f} GB ({overhead:.1f}% overhead)")
                        if chunk_info_b:
                            total_data_b = sum(c['size'] for c in chunk_info_b['chunks'])
                            span_size_b = chunk_info_b['max_end'] - chunk_info_b['min_offset']
                            overhead_b = (span_size_b - total_data_b) / total_data_b * 100
                            print(f"    FreqB: {len(chunk_info_b['chunks'])} chunks, "
                                  f"Data: {total_data_b/1e9:.2f} GB, "
                                  f"Span: {span_size_b/1e9:.2f} GB ({overhead_b:.1f}% overhead)")

                    # Read metadata from pre-downloaded buffer (no remote access)
                    metadata = self._read_nisar_metadata_direct(pol, metadata_buffer)

                    # Download frequencyA byte span if needed
                    downloaded_data_a = None
                    if chunk_info_a is not None:
                        downloaded_data_a = download_byte_range(
                            url,
                            chunk_info_a['min_offset'],
                            chunk_info_a['max_end'],
                            auth_tuple,
                            pbar=pbar,
                            http_session=http_session
                        )

                    # Download frequencyB byte span if needed
                    downloaded_data_b = None
                    if chunk_info_b is not None:
                        downloaded_data_b = download_byte_range(
                            url,
                            chunk_info_b['min_offset'],
                            chunk_info_b['max_end'],
                            auth_tuple,
                            pbar=pbar,
                            http_session=http_session
                        )

                    # Write output HDF5 (suppress debug output in parallel mode)
                    self._write_nisar_pol_h5_from_bytes(
                        downloaded_data_a, chunk_info_a, metadata, pol, out_path,
                        track, frame, datetime_str, debug and (position is None),
                        downloaded_data_b=downloaded_data_b, chunk_info_b=chunk_info_b
                    )

                    downloaded_files.append(out_name)

            return downloaded_files

        def download_with_retry(granule, position=None):
            """Download single granule with retry logic."""
            for retry in range(retries):
                try:
                    return download_nisar_granule(
                        granule, basedir, polarizations, skip_exist, debug, position=position
                    )
                except ValueError:
                    # Format errors are permanent - don't retry
                    raise
                except Exception as e:
                    print(f"ERROR downloading {granule} (attempt {retry+1}/{retries}): {e}")
                    if retry + 1 == retries:
                        raise
                    time.sleep(timeout_second)

        # Process granules in parallel using joblib
        if n_jobs == 1 or len(granules) == 1:
            # Sequential processing (no position needed)
            all_downloaded = []
            for granule in granules:
                files = download_with_retry(granule)
                all_downloaded.extend(files)
        else:
            # Sequential downloads with clear progress bars
            # (parallel progress bars have display issues; sequential is cleaner UX
            # and similar speed since downloads share bandwidth anyway)
            all_downloaded = []
            for i, granule in enumerate(granules):
                files = download_with_retry(granule)  # No position = clean single bar
                all_downloaded.extend(files)

        if all_downloaded:
            return pd.DataFrame({'file': all_downloaded})
        return None

    def _detect_nisar_layout_fast(self, url, auth_tuple, file_size, http_session=None):
        """Fast layout detection by downloading 8MB and testing.

        Downloads first 8MB, patches HDF5 superblock, and tries to read metadata.
        If successful, layout is A (metadata at start). If fails, layout is B.

        Returns: (layout, probe_buffer) where layout is 'A' or 'B' and
                 probe_buffer is the patched 8MB buffer for reuse.
        """
        import requests
        import struct
        import h5py
        from io import BytesIO

        PROBE_SIZE = 8 * 1024 * 1024  # 8 MB - needed to reach attitude/time dataset

        # Download first 8MB (reuse session if provided)
        headers = {'Range': f'bytes=0-{PROBE_SIZE-1}'}
        if http_session:
            response = http_session.get(url, headers=headers)
        else:
            response = requests.get(url, headers=headers, auth=auth_tuple)
        response.raise_for_status()
        data = bytearray(response.content)

        # Patch HDF5 superblock EOF
        version = data[8]
        if version == 0:
            data[40:48] = struct.pack('<Q', len(data))
        else:
            data[28:36] = struct.pack('<Q', len(data))
            data[44:48] = struct.pack('<I', self._hdf5_lookup3_hash(bytes(data[0:44])))

        # Try to read a metadata dataset
        try:
            with h5py.File(BytesIO(bytes(data)), 'r') as h5_test:
                # Try to read a small metadata dataset (attitude/time is usually small)
                _ = h5_test['science/LSAR/RSLC/metadata/attitude/time'][()]
                return 'A', data  # Success - metadata is at start
        except:
            return 'B', data  # Failed - metadata is at end

    def _detect_nisar_layout(self, h5_remote):
        """Detect NISAR file layout offsets using open remote HDF5.

        Returns tuple: (metadata_end, slc_start, slc_end)
        - metadata_end: byte offset where metadata ends
        - slc_start: byte offset where SLC starts
        - slc_end: byte offset where SLC ends
        """
        import h5py

        main_slc_a = {f'science/LSAR/RSLC/swaths/frequencyA/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        main_slc_b = {f'science/LSAR/RSLC/swaths/frequencyB/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        all_slc = main_slc_a | main_slc_b

        metadata_max = 0
        slc_min = float('inf')
        slc_max = 0

        def analyze_offsets(name, obj):
            nonlocal metadata_max, slc_min, slc_max
            if not isinstance(obj, h5py.Dataset):
                return

            if name in all_slc:
                # SLC dataset - get chunk offset range
                if obj.chunks:
                    try:
                        # First chunk for min
                        info = obj.id.get_chunk_info(0)
                        slc_min = min(slc_min, info.byte_offset)
                        # Last chunk for max
                        n_chunks = obj.id.get_num_chunks()
                        info = obj.id.get_chunk_info(n_chunks - 1)
                        slc_max = max(slc_max, info.byte_offset + info.size)
                    except:
                        pass
            else:
                # Metadata dataset - get max end offset
                if obj.chunks:
                    for idx in range(obj.id.get_num_chunks()):
                        try:
                            info = obj.id.get_chunk_info(idx)
                            metadata_max = max(metadata_max, info.byte_offset + info.size)
                        except:
                            pass
                else:
                    offset = obj.id.get_offset()
                    if offset:
                        metadata_max = max(metadata_max, offset + obj.id.get_storage_size())

        h5_remote.visititems(analyze_offsets)

        return int(metadata_max * 1.01), slc_min, slc_max  # Add 1% buffer to metadata_max

    def _download_nisar_metadata_start(self, url, auth_tuple, download_size, pbar=None, http_session=None):
        """Download metadata from file start (Layout A).

        Used when metadata is stored before SLC data.
        Downloads bytes 0 to download_size and patches HDF5 superblock.

        Returns bytearray with patched HDF5 data.
        """
        import requests
        import struct

        headers = {'Range': f'bytes=0-{download_size-1}'}
        if http_session:
            response = http_session.get(url, headers=headers, stream=True)
        else:
            response = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
        response.raise_for_status()

        data = bytearray()
        for chunk in response.iter_content(chunk_size=1024*1024):
            data.extend(chunk)
            if pbar:
                pbar.update(len(chunk))

        # Detect superblock version and patch EOF
        version = data[8]
        if version == 0:
            # v0: EOF at bytes 40-47, no checksum
            data[40:48] = struct.pack('<Q', len(data))
        else:
            # v2/3: EOF at bytes 28-35, checksum at 44-47
            data[28:36] = struct.pack('<Q', len(data))
            data[44:48] = struct.pack('<I', self._hdf5_lookup3_hash(bytes(data[0:44])))

        return data

    def _download_nisar_metadata_sparse(self, url, auth_tuple, file_size, header_size, tail_start, pbar=None):
        """Download NISAR metadata using sparse buffer approach (Layout B).

        Downloads header (0 to header_size) + tail (tail_start to file_size) and
        creates a sparse buffer with zeros for the SLC region in between.

        Parameters
        ----------
        header_size : int
            Size of header to download (bytes before SLC).
        tail_start : int
            Byte offset where tail begins (after SLC ends).

        Returns bytearray with sparse buffer (header + zeros + tail).
        """
        import requests

        # Download header (0 to header_size)
        headers_h = {'Range': f'bytes=0-{header_size-1}'}
        response_h = requests.get(url, headers=headers_h, auth=auth_tuple, stream=True)
        response_h.raise_for_status()
        header = bytearray()
        for chunk in response_h.iter_content(chunk_size=1024*1024):
            header.extend(chunk)
            if pbar:
                pbar.update(len(chunk))

        # Download tail (tail_start to file_size)
        headers_t = {'Range': f'bytes={tail_start}-{file_size-1}'}
        response_t = requests.get(url, headers=headers_t, auth=auth_tuple, stream=True)
        response_t.raise_for_status()
        tail = bytearray()
        for chunk in response_t.iter_content(chunk_size=1024*1024):
            tail.extend(chunk)
            if pbar:
                pbar.update(len(chunk))

        # Create sparse buffer: header + zeros + tail
        gap_size = tail_start - header_size
        data = bytearray(header)
        data.extend(b'\x00' * gap_size)
        data.extend(tail)

        return data

    def _read_nisar_metadata_direct(self, pol, metadata_buffer):
        """Read metadata from pre-downloaded buffer.

        Uses metadata_buffer (from layout A download) to read all metadata
        datasets - NO remote access needed.

        Parameters
        ----------
        pol : str
            Polarization being processed (e.g., 'HH').
        metadata_buffer : bytearray
            Pre-downloaded metadata buffer (from _download_nisar_metadata_start).

        Returns dict with all metadata datasets to copy.
        """
        import h5py
        from io import BytesIO

        # ALL main SLC datasets to skip (stored later in file)
        all_slc = {f'science/LSAR/RSLC/swaths/frequency{f}/{p}'
                   for f in ['A', 'B'] for p in ['HH', 'HV', 'VH', 'VV']}

        # Other polarizations to skip (their metadata may be stored later)
        other_pols = {'HH', 'HV', 'VH', 'VV'} - {pol}

        def should_skip_path(path):
            """Check if path should be skipped - SLC or other-pol data."""
            if path in all_slc:
                return True
            # Skip paths containing other polarizations
            for other_pol in other_pols:
                if f'/{other_pol}' in path or path.endswith(f'/{other_pol}'):
                    return True
            return False

        def iterate_safe(group, prefix=''):
            """Iterate HDF5 checking paths before accessing objects."""
            datasets = []
            groups = []
            for name in group.keys():
                path = f"{prefix}/{name}" if prefix else name
                if should_skip_path(path):
                    continue
                try:
                    item = group[name]
                    if isinstance(item, h5py.Group):
                        groups.append((path, item))
                        sub_ds, sub_grp = iterate_safe(item, path)
                        datasets.extend(sub_ds)
                        groups.extend(sub_grp)
                    elif isinstance(item, h5py.Dataset):
                        datasets.append((path, item))
                except Exception:
                    pass  # Skip inaccessible items
            return datasets, groups

        metadata = {}

        # Read everything from local buffer - no remote access
        with h5py.File(BytesIO(bytes(metadata_buffer)), 'r') as h5_local:
            # Use safe iteration that checks paths before accessing
            datasets, groups = iterate_safe(h5_local)

            # Collect group attributes
            metadata['_group_attrs'] = {}
            for path, grp in groups:
                if grp.attrs:
                    metadata['_group_attrs'][path] = dict(grp.attrs)

            # Root attributes
            metadata['_root_attrs'] = dict(h5_local.attrs)

            # Read dataset values
            for path, ds in datasets:
                try:
                    metadata[path] = {
                        'data': ds[()],
                        'dtype': ds.dtype,
                        'shape': ds.shape,
                        'attrs': dict(ds.attrs)
                    }
                except Exception:
                    pass

        return metadata

    def _get_nisar_metadata_size(self, h5_remote, pol):
        """Calculate metadata download size without downloading.

        Returns size in bytes needed to download all metadata (excludes all SLC datasets).
        """
        import h5py

        # ALL main SLC datasets to skip (they're downloaded separately)
        main_slc_a = {f'science/LSAR/RSLC/swaths/frequencyA/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        main_slc_b = {f'science/LSAR/RSLC/swaths/frequencyB/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        all_slc = main_slc_a | main_slc_b

        max_offset = 0

        def find_max_offset(name, obj):
            nonlocal max_offset
            if not isinstance(obj, h5py.Dataset):
                return
            # Skip ALL SLC datasets - they're downloaded separately
            if name in all_slc:
                return

            if obj.chunks:
                for idx in range(obj.id.get_num_chunks()):
                    try:
                        info = obj.id.get_chunk_info(idx)
                        end = info.byte_offset + info.size
                        if end > max_offset:
                            max_offset = end
                    except:
                        pass
            else:
                offset = obj.id.get_offset()
                if offset:
                    end = offset + obj.id.get_storage_size()
                    if end > max_offset:
                        max_offset = end

        h5_remote.visititems(find_max_offset)
        return int(max_offset * 1.01)  # Add 1% buffer

    def _download_nisar_metadata(self, url, auth_tuple, download_size, pol, pbar=None):
        """Download and parse NISAR metadata with shared progress bar.

        Returns dict with all metadata datasets to copy.
        """
        import h5py
        import requests
        import struct
        from io import BytesIO

        main_slc_a = {f'science/LSAR/RSLC/swaths/frequencyA/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        main_slc_b = {f'science/LSAR/RSLC/swaths/frequencyB/{p}' for p in ['HH', 'HV', 'VH', 'VV']}

        def should_skip(name):
            if name in main_slc_a and name != f'science/LSAR/RSLC/swaths/frequencyA/{pol}':
                return True
            if name in main_slc_b and name != f'science/LSAR/RSLC/swaths/frequencyB/{pol}':
                return True
            return False

        # Download metadata block
        headers = {'Range': f'bytes=0-{download_size-1}'}
        response = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
        response.raise_for_status()

        data = bytearray()
        for chunk in response.iter_content(chunk_size=1024*1024):
            data.extend(chunk)
            if pbar:
                pbar.update(len(chunk))

        # Patch HDF5 superblock
        data[28:36] = struct.pack('<Q', len(data))
        new_checksum = self._hdf5_lookup3_hash(bytes(data[0:44]))
        data[44:48] = struct.pack('<I', new_checksum)

        # Parse metadata from BytesIO
        metadata = {}
        buf = BytesIO(bytes(data))

        with h5py.File(buf, 'r') as h5_local:
            # Collect dataset names
            datasets_to_read = []
            def collect_datasets(name, obj):
                if should_skip(name):
                    return
                if isinstance(obj, h5py.Dataset) and name not in main_slc_a and name not in main_slc_b:
                    datasets_to_read.append(name)
            h5_local.visititems(collect_datasets)

            # Read all metadata datasets
            for name in datasets_to_read:
                try:
                    ds = h5_local[name]
                    metadata[name] = {
                        'data': ds[()],
                        'dtype': ds.dtype,
                        'shape': ds.shape,
                        'attrs': dict(ds.attrs)
                    }
                except:
                    pass

            # Collect group attributes
            metadata['_group_attrs'] = {}
            def collect_group_attrs(name, obj):
                if isinstance(obj, h5py.Group) and obj.attrs:
                    metadata['_group_attrs'][name] = dict(obj.attrs)
            h5_local.visititems(collect_group_attrs)

            # Root attributes
            metadata['_root_attrs'] = dict(h5_local.attrs)

        return metadata

    def _read_nisar_metadata(self, url, auth_tuple, h5_remote, pol, out_name, debug=False, position=None):
        """Read ALL metadata from remote HDF5 using single-block download.

        Copies the entire HDF5 structure except:
        - SLC datasets for polarizations other than `pol` (frequencyA and frequencyB)

        FrequencyB is INCLUDED because it's needed for ionospheric correction
        (split-spectrum technique within L-band, NOT a different radar band).

        Uses optimized in-memory approach:
        1. Query byte offsets for all metadata datasets (fast - just metadata)
        2. Download required bytes in ONE HTTP Range request
        3. Patch HDF5 superblock to allow opening truncated data
        4. Read all metadata from BytesIO (no temp file!)

        This is ~30x faster than reading each dataset individually via remote HDF5.

        Returns dict with all datasets to copy.
        """
        import h5py
        import requests
        import struct
        from io import BytesIO
        from tqdm.auto import tqdm

        # Main SLC datasets to skip (downloaded separately via direct chunk API)
        main_slc_a = {f'science/LSAR/RSLC/swaths/frequencyA/{p}' for p in ['HH', 'HV', 'VH', 'VV']}
        main_slc_b = {f'science/LSAR/RSLC/swaths/frequencyB/{p}' for p in ['HH', 'HV', 'VH', 'VV']}

        def should_skip(name):
            """Check if this path should be skipped."""
            # Skip SLC data for other polarizations in frequencyA
            if name in main_slc_a and name != f'science/LSAR/RSLC/swaths/frequencyA/{pol}':
                return True
            # Skip SLC data for other polarizations in frequencyB
            if name in main_slc_b and name != f'science/LSAR/RSLC/swaths/frequencyB/{pol}':
                return True
            return False

        # Step 1: Find maximum byte offset across all metadata datasets
        if debug:
            print("    Calculating metadata byte range...")

        max_offset = 0
        datasets_to_read = []

        def find_max_offset(name, obj):
            nonlocal max_offset
            if should_skip(name):
                return
            if not isinstance(obj, h5py.Dataset):
                return
            if name in main_slc_a or name in main_slc_b:
                return

            datasets_to_read.append(name)

            # Check byte offsets (handle both chunked and contiguous)
            if obj.chunks:
                # Chunked dataset - check all chunks
                for idx in range(obj.id.get_num_chunks()):
                    try:
                        info = obj.id.get_chunk_info(idx)
                        end = info.byte_offset + info.size
                        if end > max_offset:
                            max_offset = end
                    except:
                        pass
            else:
                # Contiguous dataset
                offset = obj.id.get_offset()
                if offset:
                    end = offset + obj.id.get_storage_size()
                    if end > max_offset:
                        max_offset = end

        h5_remote.visititems(find_max_offset)

        # Add 1% buffer
        download_size = int(max_offset * 1.01)
        if debug:
            print(f"    Metadata requires {max_offset:,} bytes ({max_offset/1e6:.1f} MB)")
            print(f"    Downloading {download_size:,} bytes ({download_size/1e6:.1f} MB)...")

        # Step 2: Download metadata block in single Range request
        headers = {'Range': f'bytes=0-{download_size-1}'}
        response = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
        response.raise_for_status()

        data = bytearray()
        # Always show progress for metadata (it's ~100 MB)
        # out_name is like "NSR_172_008_20251204T024618_HH.h5"
        desc = f"{out_name[:-3]} metadata" if out_name else f"{pol} metadata"
        with tqdm(total=download_size, unit='B', unit_scale=True, desc=desc,
                  position=position, leave=(position is None)) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                data.extend(chunk)
                pbar.update(len(chunk))

        # Step 3: Patch HDF5 superblock (EOF and checksum)
        # Superblock v2/3: EOF at offset 28, checksum at offset 44
        data[28:36] = struct.pack('<Q', len(data))
        new_checksum = self._hdf5_lookup3_hash(bytes(data[0:44]))
        data[44:48] = struct.pack('<I', new_checksum)

        # Step 4: Read metadata from BytesIO (no temp file needed!)
        metadata = {}
        buf = BytesIO(bytes(data))

        with h5py.File(buf, 'r') as h5_local:
            if debug:
                print(f"    Reading {len(datasets_to_read)} metadata datasets...")

            pbar = tqdm(datasets_to_read, desc='    Reading metadata',
                       leave=False) if debug else datasets_to_read

            for name in pbar:
                try:
                    obj = h5_local[name]
                    metadata[name] = {
                        'data': obj[()],
                        'dtype': obj.dtype,
                        'shape': obj.shape,
                        'attrs': dict(obj.attrs)  # Copy dataset attributes (description, units, etc.)
                    }
                except Exception:
                    pass  # Skip datasets that can't be read

            # Also copy root attributes
            metadata['_root_attrs'] = dict(h5_local.attrs)

            # Copy group attributes (important for HDF5 structure)
            metadata['_group_attrs'] = {}
            def collect_group_attrs(name, obj):
                if isinstance(obj, h5py.Group) and obj.attrs:
                    metadata['_group_attrs'][name] = dict(obj.attrs)
            h5_local.visititems(collect_group_attrs)

        return metadata

    @staticmethod
    def _hdf5_lookup3_hash(data):
        """Jenkins lookup3 hash for HDF5 superblock checksum."""
        def rot(x, k):
            return ((x << k) | (x >> (32 - k))) & 0xffffffff

        def mix(a, b, c):
            a = (a - c) & 0xffffffff; a ^= rot(c, 4); c = (c + b) & 0xffffffff
            b = (b - a) & 0xffffffff; b ^= rot(a, 6); a = (a + c) & 0xffffffff
            c = (c - b) & 0xffffffff; c ^= rot(b, 8); b = (b + a) & 0xffffffff
            a = (a - c) & 0xffffffff; a ^= rot(c, 16); c = (c + b) & 0xffffffff
            b = (b - a) & 0xffffffff; b ^= rot(a, 19); a = (a + c) & 0xffffffff
            c = (c - b) & 0xffffffff; c ^= rot(b, 4); b = (b + a) & 0xffffffff
            return a, b, c

        def final(a, b, c):
            c ^= b; c = (c - rot(b, 14)) & 0xffffffff
            a ^= c; a = (a - rot(c, 11)) & 0xffffffff
            b ^= a; b = (b - rot(a, 25)) & 0xffffffff
            c ^= b; c = (c - rot(b, 16)) & 0xffffffff
            a ^= c; a = (a - rot(c, 4)) & 0xffffffff
            b ^= a; b = (b - rot(a, 14)) & 0xffffffff
            c ^= b; c = (c - rot(b, 24)) & 0xffffffff
            return a, b, c

        import struct
        length = len(data)
        a = b = c = (0xdeadbeef + length) & 0xffffffff

        i = 0
        while i + 12 <= length:
            a = (a + struct.unpack('<I', data[i:i+4])[0]) & 0xffffffff
            b = (b + struct.unpack('<I', data[i+4:i+8])[0]) & 0xffffffff
            c = (c + struct.unpack('<I', data[i+8:i+12])[0]) & 0xffffffff
            a, b, c = mix(a, b, c)
            i += 12

        remaining = length - i
        if remaining > 0:
            tail = data[i:] + bytes(12 - remaining)
            for j, shift in enumerate([0, 8, 16, 24][:min(remaining, 4)]):
                a = (a + (tail[j] << shift)) & 0xffffffff
            for j, shift in enumerate([0, 8, 16, 24][:max(0, min(remaining - 4, 4))]):
                b = (b + (tail[4 + j] << shift)) & 0xffffffff
            for j, shift in enumerate([0, 8, 16, 24][:max(0, min(remaining - 8, 4))]):
                c = (c + (tail[8 + j] << shift)) & 0xffffffff
            a, b, c = final(a, b, c)

        return c

    def _write_nisar_pol_h5_from_bytes(self, downloaded_data, chunk_info, metadata, pol,
                                        out_path, track, frame, datetime_str, debug,
                                        downloaded_data_b=None, chunk_info_b=None):
        """Write single-polarization NISAR HDF5 from downloaded byte data.

        Copies ALL metadata from source file (geolocationGrid, attitude, calibration,
        processingInformation, etc.) - only excludes SLC data for other polarizations.

        Supports three modes:
        - FrequencyA only: downloaded_data present, downloaded_data_b is None
        - FrequencyB only: downloaded_data is None, downloaded_data_b present
        - Both frequencies: both present

        Parameters
        ----------
        downloaded_data : bytes or None
            Raw bytes from HTTP Range request covering frequencyA chunks.
        chunk_info : dict or None
            Chunk metadata from get_chunk_info() for frequencyA.
        metadata : dict
            ALL metadata from _read_nisar_metadata().
        downloaded_data_b : bytes, optional
            Raw bytes for frequencyB (ionospheric correction / quick look).
        chunk_info_b : dict, optional
            Chunk metadata for frequencyB.
        """
        import h5py
        from io import BytesIO
        from tqdm.auto import tqdm

        # Determine which frequencies are being written
        has_freq_a = downloaded_data is not None and chunk_info is not None
        has_freq_b = downloaded_data_b is not None and chunk_info_b is not None

        # Count total chunks
        n_chunks_total = 0
        if has_freq_a:
            n_chunks_total += len(chunk_info['chunks'])
        if has_freq_b:
            n_chunks_total += len(chunk_info_b['chunks'])

        if debug:
            print(f"    Writing {n_chunks_total} chunks + {len(metadata)-1} metadata datasets...")

        # Build HDF5 in memory
        mem_buffer = BytesIO()

        with h5py.File(mem_buffer, 'w') as h5_mem:
            # 1. Write frequencyA SLC if available
            if has_freq_a:
                shape = chunk_info['shape']
                chunk_shape = chunk_info['chunk_shape']
                dtype = chunk_info['dtype']
                compression = chunk_info['compression']
                compression_opts = chunk_info['compression_opts']
                shuffle = chunk_info.get('shuffle', False)  # Must match for write_direct_chunk
                min_offset = chunk_info['min_offset']

                dst_slc = h5_mem.create_dataset(
                    f'science/LSAR/RSLC/swaths/frequencyA/{pol}',
                    shape=shape, dtype=dtype, chunks=chunk_shape,
                    compression=compression, compression_opts=compression_opts,
                    shuffle=shuffle
                )

                chunk_iter = tqdm(chunk_info['chunks'], desc='    Writing freqA SLC', leave=False) if debug else chunk_info['chunks']
                for chunk in chunk_iter:
                    offset_in_data = chunk['offset'] - min_offset
                    chunk_bytes = downloaded_data[offset_in_data:offset_in_data + chunk['size']]
                    dst_slc.id.write_direct_chunk(chunk['coord'], chunk_bytes)

            # 2. Write frequencyB SLC if available (ionospheric correction / quick look)
            if has_freq_b:
                shape_b = chunk_info_b['shape']
                chunk_shape_b = chunk_info_b['chunk_shape']
                dtype_b = chunk_info_b['dtype']
                compression_b = chunk_info_b['compression']
                compression_opts_b = chunk_info_b['compression_opts']
                shuffle_b = chunk_info_b.get('shuffle', False)  # Must match for write_direct_chunk
                min_offset_b = chunk_info_b['min_offset']

                dst_slc_b = h5_mem.create_dataset(
                    f'science/LSAR/RSLC/swaths/frequencyB/{pol}',
                    shape=shape_b, dtype=dtype_b, chunks=chunk_shape_b,
                    compression=compression_b, compression_opts=compression_opts_b,
                    shuffle=shuffle_b
                )

                chunk_iter_b = tqdm(chunk_info_b['chunks'], desc='    Writing freqB SLC', leave=False) if debug else chunk_info_b['chunks']
                for chunk in chunk_iter_b:
                    offset_in_data = chunk['offset'] - min_offset_b
                    chunk_bytes = downloaded_data_b[offset_in_data:offset_in_data + chunk['size']]
                    dst_slc_b.id.write_direct_chunk(chunk['coord'], chunk_bytes)

            # 3. Write metadata datasets (filter by frequency if needed)
            for ds_path, ds_info in metadata.items():
                if ds_path in ('_root_attrs', '_group_attrs'):
                    continue  # Handle separately

                # Skip frequency-specific metadata when not writing that frequency
                if not has_freq_a and 'frequencyA' in ds_path:
                    continue
                if not has_freq_b and 'frequencyB' in ds_path:
                    continue

                # Ensure parent groups exist
                parent_path = '/'.join(ds_path.split('/')[:-1])
                if parent_path and parent_path not in h5_mem:
                    h5_mem.create_group(parent_path)

                # Create dataset with attributes
                try:
                    ds = h5_mem.create_dataset(ds_path, data=ds_info['data'])
                    # Copy dataset attributes (description, units, etc.)
                    if 'attrs' in ds_info:
                        for attr_key, attr_val in ds_info['attrs'].items():
                            try:
                                ds.attrs[attr_key] = attr_val
                            except Exception:
                                pass
                except Exception as e:
                    if debug:
                        print(f"    Warning: could not create {ds_path}: {e}")

            # 4. Copy group attributes (filter by frequency if needed)
            if '_group_attrs' in metadata:
                for grp_path, grp_attrs in metadata['_group_attrs'].items():
                    # Skip frequency-specific groups when not writing that frequency
                    if not has_freq_a and 'frequencyA' in grp_path:
                        continue
                    if not has_freq_b and 'frequencyB' in grp_path:
                        continue
                    if grp_path in h5_mem:
                        for attr_key, attr_val in grp_attrs.items():
                            try:
                                h5_mem[grp_path].attrs[attr_key] = attr_val
                            except Exception:
                                pass

            # 5. Copy root attributes
            if '_root_attrs' in metadata:
                for key, value in metadata['_root_attrs'].items():
                    h5_mem.attrs[key] = value

        # Single disk write
        with open(out_path, 'wb') as f:
            f.write(mem_buffer.getvalue())

        if debug:
            import os
            file_size = os.path.getsize(out_path)
            print(f"    Done: {file_size / 1e9:.2f} GB written")

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None,
               platform='SENTINEL-1', processingLevel='auto', polarization='VV', beamMode='IW'):
        import geopandas as gpd
        import asf_search
        import shapely

        # cover defined time interval
        if len(startTime)==10:
            startTime=f'{startTime} 00:00:01'
        if len(stopTime)==10:
            stopTime=f'{stopTime} 23:59:59'

        if flightDirection == 'D':
            flightDirection = 'DESCENDING'
        elif flightDirection == 'A':
            flightDirection = 'ASCENDING'

        # convert to a single geometry
        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geometry = geometry.geometry.union_all()
        # convert closed linestring to polygon
        if geometry.type == 'LineString' and geometry.coords[0] == geometry.coords[-1]:
            geometry = shapely.geometry.Polygon(geometry.coords)
        if geometry.type == 'Polygon':
            # force counterclockwise orientation.
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        #print ('wkt', geometry.wkt)

        if isinstance(processingLevel, str) and processingLevel=='auto' and platform == 'SENTINEL-1':
            processingLevel = asf_search.PRODUCT_TYPE.BURST

        # search bursts
        results = asf_search.search(
            start=startTime,
            end=stopTime,
            flightDirection=flightDirection,
            intersectsWith=geometry.wkt,
            platform=platform,
            processingLevel=processingLevel,
            polarization=polarization,
            beamMode=beamMode,
        )
        return gpd.GeoDataFrame.from_features([product.geojson() for product in results], crs="EPSG:4326")

    @staticmethod
    def plot(bursts, ax=None, figsize=None):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt

        bursts['date'] = pd.to_datetime(bursts['startTime']).dt.strftime('%Y-%m-%d')
        bursts['label'] = bursts.apply(lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} {rec['date']} [{rec['pathNumber']}]", axis=1)
        unique_labels = sorted(bursts['label'].unique())
        unique_paths = sorted(bursts['pathNumber'].astype(str).unique())
        colors = {label[-4:-1]: 'orange' if label[0] == 'A' else 'cyan' for i, label in enumerate(unique_labels)}
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        for label, group in bursts.groupby('label'):
            group.plot(ax=ax, edgecolor=colors[label[-4:-1]], facecolor='none', linewidth=1, alpha=1, label=label)
        burst_handles = [matplotlib.lines.Line2D([0], [0], color=colors[label[-4:-1]], lw=1, label=label) for label in unique_labels]
        aoi_handle = matplotlib.lines.Line2D([0], [0], color='red', lw=1, label='AOI')
        handles = burst_handles + [aoi_handle]
        ax.legend(handles=handles, loc='upper right')
        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
