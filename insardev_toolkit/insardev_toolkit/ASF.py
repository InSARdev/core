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

# ============================================================================
# Minimal asf_search replacement (replaces ~9000 lines with ~150 lines)
# ============================================================================
import requests

# ASF/Earthdata authentication constants
_EDL_HOST = 'urs.earthdata.nasa.gov'
_EDL_CLIENT_ID = 'BO_n7nTIlMljdvU6kRRB3g'
_ASF_AUTH_HOST = 'cumulus.asf.alaska.edu'
_AUTH_DOMAINS = ['asf.alaska.edu', 'earthdata.nasa.gov', 'daac.asf.alaska.edu']
_AUTH_COOKIES = ['urs_user_already_logged', 'uat_urs_user_already_logged', 'asf-urs']
_ASF_SEARCH_URL = 'https://api.daac.asf.alaska.edu/services/search/param'


class _PRODUCT_TYPE:
    """ASF product type constants."""
    BURST = 'BURST'
    SLC = 'SLC'
    GRD = 'GRD_HD'
    RAW = 'RAW'


class _ASFSearchResult:
    """Minimal ASF search result wrapper."""
    def __init__(self, geojson_feature):
        self._geojson = geojson_feature

    def geojson(self):
        return self._geojson


class _ASFSession(requests.Session):
    """Authenticated session for ASF/Earthdata downloads.

    Handles OAuth2 authentication to NASA Earthdata Login (EDL).
    Uses the same auth flow as the original asf_search library.
    """

    def __init__(self):
        super().__init__()
        self._authenticated = False
        self._username = None
        self._password = None

    def auth_with_creds(self, username, password):
        """Authenticate with Earthdata Login credentials.

        Parameters
        ----------
        username : str
            Earthdata Login username.
        password : str
            Earthdata Login password.

        Returns
        -------
        _ASFSession
            Self, for method chaining.
        """
        self._username = username
        self._password = password

        # Earthdata OAuth2 token endpoint
        token_url = f'https://{_EDL_HOST}/oauth/token'

        # Get bearer token using client credentials
        # This is how asf_search authenticates
        try:
            response = self.post(
                token_url,
                data={'grant_type': 'client_credentials'},
                auth=(self._username, self._password),
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            if response.status_code == 200:
                token_data = response.json()
                if 'access_token' in token_data:
                    self.headers['Authorization'] = f"Bearer {token_data['access_token']}"
                    self._authenticated = True
                    return self
        except Exception:
            pass

        # Fallback: use basic auth (works for many ASF endpoints)
        self.auth = (username, password)
        self._authenticated = True
        return self

    def rebuild_auth(self, prepared_request, response):
        """Maintain auth across redirects to authorized domains."""
        # Check if redirecting to an authorized domain
        url = prepared_request.url.lower()
        if any(domain in url for domain in _AUTH_DOMAINS):
            # Keep existing auth header
            return
        # For other domains, use default behavior
        super().rebuild_auth(prepared_request, response)


def _asf_granule_search(granule_list):
    """Search ASF by granule names (burst IDs or product names).

    Parameters
    ----------
    granule_list : list
        List of granule identifiers.

    Returns
    -------
    list
        List of _ASFSearchResult objects.
    """
    if not granule_list:
        return []

    # ASF SearchAPI accepts comma-separated granule list
    params = {
        'granule_list': ','.join(granule_list),
        'output': 'geojson'
    }

    import time
    for attempt in range(30):
        try:
            response = requests.get(_ASF_SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except Exception as e:
            if attempt + 1 == 30:
                raise
            print(f'ASF catalog search attempt {attempt+1} failed: {e}, retrying in 3s...')
            time.sleep(3)

    data = response.json()
    features = data.get('features', [])

    return [_ASFSearchResult(f) for f in features]


def _asf_search(start=None, end=None, flightDirection=None, intersectsWith=None,
                platform=None, processingLevel=None, polarization=None, beamMode=None):
    """Search ASF catalog with various filters.

    Parameters
    ----------
    start : str, optional
        Start datetime (ISO format or 'YYYY-MM-DD HH:MM:SS').
    end : str, optional
        End datetime.
    flightDirection : str, optional
        'ASCENDING' or 'DESCENDING'.
    intersectsWith : str, optional
        WKT geometry string.
    platform : str, optional
        e.g., 'SENTINEL-1', 'SENTINEL-1A', 'SENTINEL-1B'.
    processingLevel : str, optional
        e.g., 'BURST', 'SLC', 'GRD_HD'.
    polarization : str, optional
        e.g., 'VV', 'VH', 'HH', 'HV', 'VV+VH'.
    beamMode : str, optional
        e.g., 'IW', 'EW', 'SM'.

    Returns
    -------
    list
        List of _ASFSearchResult objects.
    """
    params = {'output': 'geojson'}

    if start:
        params['start'] = start
    if end:
        params['end'] = end
    if flightDirection:
        params['flightDirection'] = flightDirection
    if intersectsWith:
        params['intersectsWith'] = intersectsWith
    if platform:
        params['platform'] = platform
    if processingLevel:
        params['processingLevel'] = processingLevel
    if polarization:
        params['polarization'] = polarization
    if beamMode:
        params['beamMode'] = beamMode

    import time
    for attempt in range(30):
        try:
            response = requests.get(_ASF_SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except Exception as e:
            if attempt + 1 == 30:
                raise
            print(f'ASF catalog search attempt {attempt+1} failed: {e}, retrying in 3s...')
            time.sleep(3)

    data = response.json()
    features = data.get('features', [])

    return [_ASFSearchResult(f) for f in features]


# Module-like namespace for compatibility with: import asf_search; asf_search.search()
class _asf_search_module:
    """Namespace mimicking asf_search module interface."""
    ASFSession = _ASFSession
    PRODUCT_TYPE = _PRODUCT_TYPE()
    search = staticmethod(_asf_search)
    granule_search = staticmethod(_asf_granule_search)

asf_search = _asf_search_module()
# ============================================================================

# Cloudflare Worker cache proxy for S1 bursts (handles auth internally)
_S1_CACHE_PROXY = 'https://s1-cache-asf.insar.dev'
_ASF_BURST_HOST = 'https://sentinel1-burst.asf.alaska.edu'

# Cloudflare Worker cache proxy for NISAR (handles auth internally)
# API: /GRANULE_ID/OFFSET.bin → 128MB block at OFFSET
_NISAR_CACHE_PROXY = 'https://nisar-cache-asf.insar.dev'
_NISAR_BLOCK_SIZE = 128 * 1024 * 1024  # 128 MB blocks


class ASF(progressbar_joblib):
    import pandas as pd
    from datetime import timedelta

    def __init__(self, username=None, password=None):
        """Initialize ASF downloader.

        Parameters
        ----------
        username : str, optional
            Earthdata Login username. If not provided, uses cache proxy.
        password : str, optional
            Earthdata Login password. If not provided, uses cache proxy.

        Notes
        -----
        When no credentials provided, downloads use Cloudflare cache proxy
        at s1-cache-asf.insar.dev which handles authentication internally.
        """
        self.username = username
        self.password = password
        if username is None:
            print("NOTE: Using insar.dev Cache API. Free for non-commercial use; license required for funded academic, institutional, or professional use.")

    def _get_asf_session(self):
        """Get authenticated session for ASF downloads.

        Returns plain requests.Session if no credentials (uses cache proxy).
        """
        if self.username is None:
            # Cache proxy handles auth - just need a plain session
            return requests.Session()
        return asf_search.ASFSession().auth_with_creds(self.username, self.password)

    def _get_burst_url(self, original_url):
        """Convert ASF burst URL to cache proxy URL if no credentials."""
        if self.username is None and original_url.startswith(_ASF_BURST_HOST):
            return original_url.replace(_ASF_BURST_HOST, _S1_CACHE_PROXY)
        return original_url

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
    def download(self, basedir, bursts, polarization=None, frequency=None, bbox=None, session=None, n_jobs=None, joblib_backend='loky', skip_exist=True,
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
        frequency : str or list, required for NISAR
            Which frequency band(s) to download (NISAR stores two frequencies with different resolutions):
            - 'A': frequencyA (20MHz bandwidth, ~7m range resolution, ~10GB per scene)
            - 'B': frequencyB (5MHz bandwidth, ~25m range resolution, ~1.5GB per scene)
            - ['A', 'B']: Both frequencies in same file (~14GB per scene)
        bbox : tuple or None, optional (NISAR cache proxy only)
            Bounding box in WGS84 coordinates: (west, south, east, north).
            When provided, only downloads aligned blocks covering the bbox.
            Uses cache-optimized multi-offset endpoint for efficient partial extraction.
            If None, downloads full scene using aligned blocks for caching benefit.
        session : asf_search.ASFSession, optional
            Authenticated session. Created automatically if None.
        n_jobs : int or None, optional
            Parallel download jobs. None uses mission-specific defaults (S1: 8, NISAR: 2).
        joblib_backend : str, optional
            Backend for parallel processing. Default 'loky' (multiprocessing, faster on Colab).
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
        >>> # NISAR: download HH frequencyA (primary InSAR, ~7m range resolution)
        >>> asf.download('data/freqA/', 'NISAR_L1_PR_RSLC_006_172_A_008_...', polarization='HH', frequency='A')
        >>> # NISAR: download HH frequencyB (quick look or iono correction, 4x less data)
        >>> asf.download('data/freqB/', 'NISAR_L1_PR_RSLC_006_172_A_008_...', polarization='HH', frequency='B')
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

        # Require explicit frequency for NISAR to prevent accidental large downloads
        if nisar_granules and frequency is None:
            raise ValueError(
                "NISAR data requires explicit frequency parameter:\n"
                "  frequency='B': 5MHz bandwidth (~25m res, ~1.5GB) - recommended for quick look\n"
                "  frequency='A': 20MHz bandwidth (~7m res, ~10GB) - full resolution InSAR\n"
                "  frequency=['A','B']: Both frequencies in same file (~14GB)"
            )

        # Expand S1 burst names by requested polarizations before skip-exist check
        if s1_bursts and pols:
            expanded = []
            for burst in s1_bursts:
                parts = burst.split('_')
                for pol in pols:
                    new_parts = parts.copy()
                    new_parts[4] = pol
                    expanded.append('_'.join(new_parts))
            seen = set()
            s1_bursts = [b for b in expanded if not (b in seen or seen.add(b))]

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
            df = self._download_nisar(basedir, nisar_needed, pols, frequency, bbox, session,
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
        # S1-specific default: 8 parallel jobs
        if n_jobs is None:
            n_jobs = 8

        import rioxarray as rio
        from tifffile import TiffFile
        import xmltodict
        from xml.etree import ElementTree
        import pandas as pd
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

        # URL transformer for cache proxy
        def get_burst_url(url):
            return self._get_burst_url(url)

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
            manifest_url = get_burst_url(properties['additionalUrls'][0])
            response = session.get(manifest_url, timeout=(10, 60))
            response.raise_for_status()
            xml_content = response.text
            if debug:
                cache_status = response.headers.get('x-cache', 'N/A')
                size_mb = len(xml_content.encode()) / 1024 / 1024
                print(f'  XML  {cache_status:4} {size_mb:5.1f}MB {burst}')
            if len(xml_content) == 0:
                raise Exception(f'ERROR: Downloaded manifest is empty: {manifest_url}')
            # check if server returned JSON error instead of XML
            if xml_content.lstrip().startswith('{'):
                try:
                    import json
                    error_json = json.loads(xml_content)
                    error_msg = error_json.get('message', error_json.get('error', str(error_json)))
                    raise Exception(f'ERROR: ASF server returned error instead of manifest for {burst}: {error_msg}')
                except json.JSONDecodeError:
                    raise Exception(f'ERROR: ASF server returned invalid response for {burst}: {xml_content[:200]}')
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

                # Download TIFF fully into memory (no streaming — requests will raise
                # on incomplete/truncated responses instead of silently returning partial data)
                tiff_url = get_burst_url(properties['url'])
                response = session.get(tiff_url, timeout=(10, 300))
                response.raise_for_status()
                tiff_bytes = response.content
                if debug:
                    cache_status = response.headers.get('x-cache', 'N/A')
                    size_mb = len(tiff_bytes) / 1024 / 1024
                    print(f'  TIFF {cache_status:4} {size_mb:5.1f}MB {burst}')
                if len(tiff_bytes) == 0:
                    raise Exception(f'ERROR: Downloaded TIFF is empty: {tiff_url}')

                # Check if server returned JSON error instead of TIFF
                # TIFF magic bytes: II*\x00 (little-endian) or MM\x00* (big-endian)
                if tiff_bytes[:2] not in (b'II', b'MM'):
                    # Not a TIFF - likely JSON error from server
                    try:
                        import json
                        error_json = json.loads(tiff_bytes.decode('utf-8', errors='replace'))
                        error_msg = error_json.get('message', error_json.get('error', str(error_json)))
                        raise Exception(f'ERROR: ASF server returned error instead of TIFF for {burst}: {error_msg}')
                    except json.JSONDecodeError:
                        raise Exception(f'ERROR: ASF server returned invalid response for {burst}: {tiff_bytes[:100]!r}')

                # Validate TIFF structure, dimensions, and completeness using TiffFile
                with TiffFile(io.BytesIO(tiff_bytes)) as tif:
                    page = tif.pages[0]
                    actual_lines, actual_samples = page.shape
                    if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                        raise Exception(f'ERROR: Downloaded TIFF dimensions mismatch for {burst}: '
                                      f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                      f'ASF burst extraction may have failed.')
                    # Verify all strip data fits within the downloaded bytes
                    for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
                        if offset + bytecount > len(tiff_bytes):
                            raise Exception(f'ERROR: Downloaded TIFF truncated for {burst}: '
                                          f'strip at offset {offset} needs {bytecount} bytes '
                                          f'but file is only {len(tiff_bytes)} bytes.')
                    # Also get offset for XML creation
                    tiff_offset = page.dataoffsets[0]

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

            # All validations passed - write to temp files then atomic rename.
            # This guarantees no partial files on disk if interrupted mid-write.
            if 'tiff_bytes' in dir():
                tmp = tif_file + '.tmp'
                with open(tmp, 'wb') as f:
                    f.write(tiff_bytes)
                os.rename(tmp, tif_file)

            for filepath, content in xml_contents.items():
                tmp = filepath + '.tmp'
                with open(tmp, 'w') as f:
                    f.write(content)
                os.rename(tmp, filepath)

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
            burst_id = result.geojson()['properties']['fileID']
            for retry in range(retries):
                try:
                    download_burst(result, basedir, session)
                    return True
                except Exception as e:
                    print(f'ERROR: download attempt {retry+1} failed for {burst_id}: {e}')
                    if retry + 1 == retries:
                        return False
                time.sleep(timeout_second)

        # download bursts
        with self.progressbar_joblib(tqdm(desc='Downloading ASF SLC'.ljust(25), total=len(results))) as progress_bar:
            statuses = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_burst_with_retry)\
                                    (result, basedir, session, retries=retries, timeout_second=timeout_second) for result in results)

        failed_count = statuses.count(False)
        if failed_count > 0:
            raise Exception(f'Bursts downloading failed for {failed_count} items.')
        # parse processed bursts and convert to dataframe
        bursts_downloaded = pd.DataFrame(bursts_missed, columns=['burst'])
        # return the results in a user-friendly dataframe
        return bursts_downloaded

    # =========================================================================
    # NISAR Helper Methods (shared between direct and cache downloads)
    # =========================================================================

    @staticmethod
    def _nisar_get_chunk_info(h5, pol, frequency='A'):
        """Query chunk byte offsets from HDF5 file.

        Parameters
        ----------
        h5 : h5py.File
            Open HDF5 file handle
        pol : str
            Polarization ('HH', 'HV', 'VH', 'VV')
        frequency : str
            Frequency band ('A' or 'B')

        Returns
        -------
        dict with keys:
            'chunks': list of {'offset', 'size', 'row', 'col', 'coord'}
            'min_offset', 'max_end': byte range span
            'shape', 'chunk_shape': array dimensions
            'dtype', 'compression', 'compression_opts', 'shuffle': HDF5 dataset properties
            'n_az', 'n_rg': number of chunks in each dimension
        """
        slc = h5[f'science/LSAR/RSLC/swaths/frequency{frequency}/{pol}']
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
            'shuffle': slc.shuffle,
            'n_az': n_az,
            'n_rg': n_rg
        }

    @staticmethod
    def _nisar_parse_granule_id(granule_id):
        """Parse NISAR granule ID to extract track, frame, and datetime.

        Format: NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_20251204T024653_X05007_N_F_J_001.h5
                ^product^     ^cycle^track^dir^frame^...^pols^..^start_time^     ^end_time^
        """
        parts = granule_id.replace('.h5', '').split('_')
        track = int(parts[5])
        frame = int(parts[7])
        datetime_str = parts[11]
        return track, frame, datetime_str

    @staticmethod
    def _nisar_merge_chunks_to_regions(chunks, gap_threshold=1024*1024):
        """Merge adjacent chunks into contiguous regions allowing small gaps.

        Parameters
        ----------
        chunks : list
            List of {'offset': int, 'size': int, ...} dicts, or (offset, size) tuples
        gap_threshold : int
            Maximum gap in bytes to merge (default 1MB)

        Returns
        -------
        list of (region_start, region_size, chunk_list)
            chunk_list contains the original chunks in this region
        """
        if not chunks:
            return []

        # Normalize to list of dicts
        if isinstance(chunks[0], tuple):
            chunks = [{'offset': off, 'size': sz} for off, sz in chunks]

        # Sort by offset
        sorted_chunks = sorted(chunks, key=lambda c: c['offset'])

        regions = []
        region_start = sorted_chunks[0]['offset']
        region_end = sorted_chunks[0]['offset'] + sorted_chunks[0]['size']
        region_chunks = [sorted_chunks[0]]

        for chunk in sorted_chunks[1:]:
            if chunk['offset'] <= region_end + gap_threshold:
                # Extend current region
                region_end = max(region_end, chunk['offset'] + chunk['size'])
                region_chunks.append(chunk)
            else:
                # Save current region and start new one
                regions.append((region_start, region_end - region_start, region_chunks))
                region_start = chunk['offset']
                region_end = chunk['offset'] + chunk['size']
                region_chunks = [chunk]

        regions.append((region_start, region_end - region_start, region_chunks))
        return regions

    @staticmethod
    def _nisar_bbox_to_pixel_indices(h5, bbox, chunk_info_a, chunk_info_b=None):
        """Convert WGS84 bbox to pixel indices using geolocationGrid.

        Parameters
        ----------
        h5 : h5py.File
            Open HDF5 file with geolocationGrid
        bbox : tuple
            (west, south, east, north) in WGS84 degrees
        chunk_info_a : dict
            Chunk info for frequencyA (has shape)
        chunk_info_b : dict, optional
            Chunk info for frequencyB

        Returns
        -------
        dict with keys:
            'az_start', 'az_end': azimuth pixel range
            'rg_start_a', 'rg_end_a': range pixel range for freqA
            'rg_start_b', 'rg_end_b': range pixel range for freqB (if provided)
        """
        import numpy as np

        # Bbox format: (west, south, east, north) = (lon_min, lat_min, lon_max, lat_max)
        west, south, east, north = bbox
        geo = h5['science/LSAR/RSLC/metadata/geolocationGrid']

        # Get geolocation coordinates (use height=0 layer, index 10 of 20)
        # NISAR EPSG 4326 convention: coordinateX = longitude, coordinateY = latitude
        lon = geo['coordinateX'][10, :, :]  # (n_az_geo, n_rg_geo) - longitude
        lat = geo['coordinateY'][10, :, :]  # latitude
        geo_az_time = geo['zeroDopplerTime'][:]
        geo_slant_range = geo['slantRange'][:]

        # Get full grid coordinates
        swaths = h5['science/LSAR/RSLC/swaths']
        full_az_time = swaths['zeroDopplerTime'][:]
        full_slant_range_a = swaths['frequencyA/slantRange'][:]

        # Find geolocation grid cells inside bbox
        in_bbox = ((lon >= west) & (lon <= east) &
                   (lat >= south) & (lat <= north))

        if not np.any(in_bbox):
            raise ValueError(f"Bbox {bbox} does not intersect scene")

        # Get azimuth and range indices in geolocation grid
        az_geo_idx, rg_geo_idx = np.where(in_bbox)
        az_geo_min, az_geo_max = az_geo_idx.min(), az_geo_idx.max()
        rg_geo_min, rg_geo_max = rg_geo_idx.min(), rg_geo_idx.max()

        # Convert to full grid indices via time/range interpolation
        az_time_min = geo_az_time[az_geo_min]
        az_time_max = geo_az_time[az_geo_max]
        rg_min = geo_slant_range[rg_geo_min]
        rg_max = geo_slant_range[rg_geo_max]

        # Find full grid indices
        az_start = int(np.searchsorted(full_az_time, az_time_min))
        az_end = int(np.searchsorted(full_az_time, az_time_max)) + 1
        rg_start_a = int(np.searchsorted(full_slant_range_a, rg_min))
        rg_end_a = int(np.searchsorted(full_slant_range_a, rg_max)) + 1

        # Clamp to valid range
        n_az_pixels = chunk_info_a['shape'][0]
        n_rg_a_pixels = chunk_info_a['shape'][1]
        az_start = max(0, az_start)
        az_end = min(n_az_pixels, az_end)
        rg_start_a = max(0, rg_start_a)
        rg_end_a = min(n_rg_a_pixels, rg_end_a)

        result = {
            'az_start': az_start,
            'az_end': az_end,
            'rg_start_a': rg_start_a,
            'rg_end_a': rg_end_a,
        }

        # Handle frequencyB if present
        if chunk_info_b:
            full_slant_range_b = swaths['frequencyB/slantRange'][:]
            n_rg_b_pixels = chunk_info_b['shape'][1]

            rg_start_b = int(np.searchsorted(full_slant_range_b, rg_min))
            rg_end_b = int(np.searchsorted(full_slant_range_b, rg_max)) + 1
            rg_start_b = max(0, rg_start_b)
            rg_end_b = min(n_rg_b_pixels, rg_end_b)

            result['rg_start_b'] = rg_start_b
            result['rg_end_b'] = rg_end_b

        return result

    @staticmethod
    def _nisar_filter_chunks_by_pixel_range(chunk_info, az_start, az_end, rg_start, rg_end):
        """Filter chunks to only those overlapping the given pixel range.

        Parameters
        ----------
        chunk_info : dict
            Output from _nisar_get_chunk_info()
        az_start, az_end : int
            Azimuth pixel range (start inclusive, end exclusive)
        rg_start, rg_end : int
            Range pixel range (start inclusive, end exclusive)

        Returns
        -------
        list of chunk dicts that overlap the pixel range
        """
        chunk_shape = chunk_info['chunk_shape']
        filtered = []

        for chunk in chunk_info['chunks']:
            # Chunk covers pixels [row*chunk_az, (row+1)*chunk_az) in azimuth
            chunk_az_start = chunk['row'] * chunk_shape[0]
            chunk_az_end = (chunk['row'] + 1) * chunk_shape[0]
            chunk_rg_start = chunk['col'] * chunk_shape[1]
            chunk_rg_end = (chunk['col'] + 1) * chunk_shape[1]

            # Check overlap
            if (chunk_az_end > az_start and chunk_az_start < az_end and
                chunk_rg_end > rg_start and chunk_rg_start < rg_end):
                filtered.append(chunk)

        return filtered

    def _download_nisar(self, basedir, granules, polarizations, frequency, bbox, session,
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
        frequency : str
            'A' for frequencyA (20 MHz, high resolution).
            'B' for frequencyB (5 MHz, 4x less data, for quick look).
        """
        # NISAR-specific default: 4 parallel jobs (optimal for Colab with decompression)
        if n_jobs is None:
            n_jobs = 4

        import h5py
        import fsspec
        import aiohttp
        import requests
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from io import BytesIO
        from datetime import datetime, timedelta
        import os
        import time
        import threading

        # Use cache proxy when no credentials provided
        if self.username is None:
            return self._download_nisar_via_cache(
                basedir, granules, polarizations, frequency, bbox,
                n_jobs, skip_exist, retries, timeout_second, debug
            )

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

        # Use shared helper for chunk info
        get_chunk_info = ASF._nisar_get_chunk_info

        def extract_signed_url(url, auth_tuple, http_session=None):
            """Extract signed CloudFront URL by following OAuth redirects.

            Makes a small Range request to trigger OAuth flow and capture
            the final signed URL for direct reuse.
            """
            headers = {'Range': 'bytes=0-0'}  # Minimal request

            if http_session:
                # Follow redirects manually to capture final URL
                resp = http_session.get(url, headers=headers, allow_redirects=False)
                while resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get('Location')
                    if not location:
                        break
                    resp = http_session.get(location, headers=headers, allow_redirects=False)
                # Final URL after all redirects
                if 'cloudfront.net' in resp.url:
                    return resp.url
            return None

        def download_byte_range(url, start, end, auth_tuple, pbar=None, http_session=None, signed_url=None):
            """Download byte range using single HTTP Range request.

            If signed_url is provided, uses it directly (skipping OAuth).
            If pbar is provided, updates it instead of creating a new one.
            If http_session is provided, reuses the connection.

            Returns: (bytes, signed_url) - signed_url for reuse in subsequent requests
            """
            size = end - start
            headers = {'Range': f'bytes={start}-{end-1}'}  # HTTP Range is inclusive

            chunks_data = []
            returned_signed_url = signed_url

            if signed_url:
                # Use signed URL directly (much faster - no OAuth redirects)
                r = requests.get(signed_url, headers=headers, stream=True)
            elif http_session:
                r = http_session.get(url, headers=headers, stream=True)
                # Capture signed URL from redirect chain
                if 'cloudfront.net' in r.url:
                    returned_signed_url = r.url
            else:
                r = requests.get(url, headers=headers, auth=auth_tuple, stream=True)

            with r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024*1024):
                    chunks_data.append(chunk)
                    if pbar:
                        pbar.update(len(chunk))

            return b''.join(chunks_data), returned_signed_url

        def download_filtered_chunks(url, chunk_info, bbox_info, freq, auth_tuple, pbar=None,
                                      http_session=None, signed_url=None, gap_threshold=16*1024*1024):
            """Download only chunks overlapping bbox, merging with 16MB gap threshold.

            Uses larger gap threshold (16MB) to reduce HTTP requests while still
            skipping large gaps between chunk regions.

            Parameters
            ----------
            chunk_info : dict
                Output from _nisar_get_chunk_info()
            bbox_info : dict
                Output from _nisar_bbox_to_pixel_indices()
            freq : str
                'A' or 'B' (to look up correct rg_start/rg_end keys)
            gap_threshold : int
                Maximum gap in bytes to merge (default 16MB)

            Returns
            -------
            tuple: (chunk_data_dict, signed_url)
                chunk_data_dict maps chunk byte offset -> chunk bytes
            """
            # Filter chunks to bbox
            rg_start_key = f'rg_start_{freq.lower()}'
            rg_end_key = f'rg_end_{freq.lower()}'
            filtered_chunks = ASF._nisar_filter_chunks_by_pixel_range(
                chunk_info,
                bbox_info['az_start'], bbox_info['az_end'],
                bbox_info[rg_start_key], bbox_info[rg_end_key]
            )

            if not filtered_chunks:
                return {}, signed_url

            # Merge into regions with 16MB gap threshold
            regions = ASF._nisar_merge_chunks_to_regions(filtered_chunks, gap_threshold)

            # Download each region
            chunk_data = {}
            current_signed_url = signed_url

            for region_start, region_size, region_chunks in regions:
                region_end = region_start + region_size
                region_bytes, current_signed_url = download_byte_range(
                    url, region_start, region_end, auth_tuple,
                    pbar=pbar, http_session=http_session, signed_url=current_signed_url
                )

                # Extract individual chunks from region
                for chunk in region_chunks:
                    rel_offset = chunk['offset'] - region_start
                    chunk_data[chunk['offset']] = region_bytes[rel_offset:rel_offset + chunk['size']]

            return chunk_data, current_signed_url

        # Use shared helper for granule ID parsing
        parse_nisar_granule_id = ASF._nisar_parse_granule_id

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

            # Get file size (HEAD request)
            head_resp = http_session.head(url, allow_redirects=True)
            file_size = int(head_resp.headers['Content-Length'])

            # Download 128MB metadata block (same as cache path) with progress bar
            with tqdm(total=128*1024*1024, unit='B', unit_scale=True,
                      desc=f"{short_name} metadata", leave=False,
                      dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0,
                      disable=(position is not None)) as meta_pbar:
                layout, metadata_buffer = self._detect_nisar_layout_fast(
                    url, auth_tuple, file_size, http_session=http_session,
                    pbar=meta_pbar if position is None else None
                )

            if debug and position is None:
                print(f"  File size: {file_size/(1024**3):.2f} GB, Layout: {layout}, Metadata: 128MB")

            # Get available polarizations from metadata buffer (no remote access)
            from io import BytesIO
            with h5py.File(BytesIO(bytes(metadata_buffer)), 'r') as h5_meta:
                swaths_path = 'science/LSAR/RSLC/swaths'
                if swaths_path not in h5_meta:
                    raise ValueError(
                        f"Unsupported NISAR file format: '{swaths_path}' not found in {granule_id}. "
                        f"This may be an older simulated scene with incompatible structure."
                    )
                swaths_grp = h5_meta[swaths_path]

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
            # Normalize frequency to list for consistent checking
            freq_list = [frequency] if isinstance(frequency, str) else frequency
            download_freq_a = 'A' in freq_list
            download_freq_b = 'B' in freq_list and has_freq_b

            if debug and position is None:
                freq_str = '+'.join(f for f in freq_list if f == 'A' or (f == 'B' and has_freq_b))
                print(f"NISAR {track}_{frame}: downloading {pols_to_download} (frequency{freq_str})")

            if layout != 'A':
                raise NotImplementedError(f"NISAR Layout B (metadata at end) not supported: {granule_id}")

            # Layout A: metadata at start - use 128MB buffer for everything
            with h5py.File(BytesIO(bytes(metadata_buffer)), 'r') as h5_meta:
                # Get chunk info for pols we're actually downloading
                all_chunk_info = {}
                for pol in pols_to_download:
                    chunk_info_a = None
                    chunk_info_b = None
                    if download_freq_a:
                        chunk_info_a = get_chunk_info(h5_meta, pol, 'A')
                    if download_freq_b and pol in freq_b_pols:
                        chunk_info_b = get_chunk_info(h5_meta, pol, 'B')
                    all_chunk_info[pol] = (chunk_info_a, chunk_info_b)

                # Calculate bbox pixel indices if bbox provided (uses geolocationGrid from 128MB buffer)
                bbox_info = None
                if bbox is not None:
                    first_ci_a = next((ci_a for ci_a, _ in all_chunk_info.values() if ci_a), None)
                    first_ci_b = next((ci_b for _, ci_b in all_chunk_info.values() if ci_b), None)
                    if first_ci_a or first_ci_b:
                        bbox_info = ASF._nisar_bbox_to_pixel_indices(
                            h5_meta, bbox, first_ci_a or first_ci_b, first_ci_b
                        )
                        if debug and position is None:
                            print(f"  Bbox {bbox} -> pixels az[{bbox_info['az_start']}:{bbox_info['az_end']}], "
                                  f"rg_a[{bbox_info.get('rg_start_a', 'N/A')}:{bbox_info.get('rg_end_a', 'N/A')}]")

            # Calculate total SLC size (filtered by bbox if provided)
            total_slc_size = 0
            GAP_THRESHOLD = 16 * 1024 * 1024  # 16MB - same as download_filtered_chunks
            for pol, (chunk_info_a, chunk_info_b) in all_chunk_info.items():
                if chunk_info_a:
                    if bbox_info is not None:
                        # Calculate size of merged regions (16MB gap threshold)
                        filtered = ASF._nisar_filter_chunks_by_pixel_range(
                            chunk_info_a, bbox_info['az_start'], bbox_info['az_end'],
                            bbox_info['rg_start_a'], bbox_info['rg_end_a']
                        )
                        regions = ASF._nisar_merge_chunks_to_regions(filtered, GAP_THRESHOLD)
                        total_slc_size += sum(r[1] for r in regions)
                    else:
                        total_slc_size += chunk_info_a['max_end'] - chunk_info_a['min_offset']
                if chunk_info_b:
                    if bbox_info is not None:
                        filtered = ASF._nisar_filter_chunks_by_pixel_range(
                            chunk_info_b, bbox_info['az_start'], bbox_info['az_end'],
                            bbox_info['rg_start_b'], bbox_info['rg_end_b']
                        )
                        regions = ASF._nisar_merge_chunks_to_regions(filtered, GAP_THRESHOLD)
                        total_slc_size += sum(r[1] for r in regions)
                    else:
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
                              f"Data: {total_data/(1024**3):.2f} GB, "
                              f"Span: {span_size/(1024**3):.2f} GB ({overhead:.1f}% overhead)")
                    if chunk_info_b:
                        total_data_b = sum(c['size'] for c in chunk_info_b['chunks'])
                        span_size_b = chunk_info_b['max_end'] - chunk_info_b['min_offset']
                        overhead_b = (span_size_b - total_data_b) / total_data_b * 100
                        print(f"    FreqB: {len(chunk_info_b['chunks'])} chunks, "
                              f"Data: {total_data_b/(1024**3):.2f} GB, "
                              f"Span: {span_size_b/(1024**3):.2f} GB ({overhead_b:.1f}% overhead)")

            # Create progress bar for SLC download
            desc = f"{short_name}"
            with tqdm(total=total_slc_size, unit='B', unit_scale=True, desc=desc,
                      position=position, leave=True,
                      dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0) as pbar:

                # Download and write each polarization
                downloaded_files = []
                os.makedirs(out_dir, exist_ok=True)
                signed_url = None  # Will be extracted from first request and reused

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
                                  f"Data: {total_data/(1024**3):.2f} GB, "
                                  f"Span: {span_size/(1024**3):.2f} GB ({overhead:.1f}% overhead)")
                        if chunk_info_b:
                            total_data_b = sum(c['size'] for c in chunk_info_b['chunks'])
                            span_size_b = chunk_info_b['max_end'] - chunk_info_b['min_offset']
                            overhead_b = (span_size_b - total_data_b) / total_data_b * 100
                            print(f"    FreqB: {len(chunk_info_b['chunks'])} chunks, "
                                  f"Data: {total_data_b/(1024**3):.2f} GB, "
                                  f"Span: {span_size_b/(1024**3):.2f} GB ({overhead_b:.1f}% overhead)")

                    # Read metadata from pre-downloaded buffer (no remote access)
                    metadata = self._read_nisar_metadata_direct(pol, metadata_buffer)

                    # Download frequencyA data
                    downloaded_data_a = None
                    if chunk_info_a is not None:
                        if bbox_info is not None:
                            # Bbox mode: merged regions with 16MB gap threshold
                            downloaded_data_a, signed_url = download_filtered_chunks(
                                url, chunk_info_a, bbox_info, 'A', auth_tuple,
                                pbar=pbar, http_session=http_session, signed_url=signed_url
                            )
                        else:
                            # Full download: single Range request for entire span
                            downloaded_data_a, signed_url = download_byte_range(
                                url,
                                chunk_info_a['min_offset'],
                                chunk_info_a['max_end'],
                                auth_tuple,
                                pbar=pbar,
                                http_session=http_session,
                                signed_url=signed_url
                            )

                    # Download frequencyB data
                    downloaded_data_b = None
                    if chunk_info_b is not None:
                        if bbox_info is not None:
                            # Bbox mode: merged regions with 16MB gap threshold
                            downloaded_data_b, signed_url = download_filtered_chunks(
                                url, chunk_info_b, bbox_info, 'B', auth_tuple,
                                pbar=pbar, http_session=http_session, signed_url=signed_url
                            )
                        else:
                            # Full download: single Range request for entire span
                            downloaded_data_b, signed_url = download_byte_range(
                                url,
                                chunk_info_b['min_offset'],
                                chunk_info_b['max_end'],
                                auth_tuple,
                                pbar=pbar,
                                http_session=http_session,
                                signed_url=signed_url
                            )

                    # Write output HDF5 (suppress debug output in parallel mode)
                    self._write_nisar_pol_h5_from_bytes(
                        downloaded_data_a, chunk_info_a, metadata, pol, out_path,
                        track, frame, datetime_str, debug and (position is None),
                        downloaded_data_b=downloaded_data_b, chunk_info_b=chunk_info_b,
                        crop_info=bbox_info
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

        # Process granules (sequential when n_jobs=1, single granule, or debug=True)
        if n_jobs == 1 or len(granules) == 1 or debug:
            # Sequential processing (also when debug=True for easier debugging)
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

    def _download_nisar_via_cache(self, basedir, granules, polarizations, frequency, bbox,
                                   n_jobs, skip_exist, retries, timeout_second, debug):
        """Download NISAR via Cloudflare cache proxy (no credentials required).

        Uses cache proxy at nisar-cache-asf.insar.dev with two APIs:
        1. Single block: /GRANULE_ID/OFFSET/LENGTH.bin → single byte range (≤128MB)
        2. Multi-offset: /GRANULE_ID/off1_len1,off2_len2,.../ranges.bin → 25x25km blocks (8-96MB)

        When bbox is provided, downloads only the aligned blocks covering the bbox.
        For full downloads, also uses aligned blocks to maximize cache efficiency.

        Parameters
        ----------
        granules : list
            List of NISAR granule IDs.
        polarizations : list or None
            If None, download all available polarizations.
        frequency : str or None
            If None, download both frequencyA and frequencyB.
            If 'A', download only frequencyA. If 'B', download only frequencyB.
        bbox : tuple or None
            Bounding box (west, south, east, north) in WGS84.
            If provided, only download blocks covering this area.
        n_jobs : int
            Number of parallel granule downloads (uses loky backend).
        """
        # NISAR-specific default: 4 parallel jobs (optimal for Colab with decompression)
        if n_jobs is None:
            n_jobs = 4

        import h5py
        import requests
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from io import BytesIO
        import os
        import struct
        import time
        from joblib import Parallel, delayed

        MIN_BLOCK = 64 * 1024 * 1024   # 64 MB min block
        MAX_BLOCK = 128 * 1024 * 1024  # 128 MB max block

        # 25x25km aligned block constants (for cache efficiency)
        # 512 pixels × 4.46m = 2.28km azimuth, 512 pixels × 9.44m = 4.83km range
        ALIGNED_AZ_CHUNKS = 15  # 15 chunks azimuth (~34km, ~105MB blocks for FreqB)
        ALIGNED_RG_CHUNKS = 7   # FreqB has 13 rg chunks, use 7 to get 7+6=2 blocks
        MAX_MULTI_BLOCK = 128 * 1024 * 1024  # 128 MB max for multi-offset (15×7 aligned blocks)

        # Use shared helpers
        parse_nisar_granule_id = ASF._nisar_parse_granule_id
        get_chunk_info = ASF._nisar_get_chunk_info

        # Note: get_chunk_info already returns dict with keys:
        # 'chunks', 'min_offset', 'max_end', 'shape', 'chunk_shape',
        # 'dtype', 'compression', 'compression_opts', 'shuffle', 'n_az', 'n_rg'

        def bbox_to_block_indices(h5, bbox, chunk_info_a, chunk_info_b=None):
            """Convert WGS84 bbox to chunk indices for bbox-optimized download.

            Uses shared _nisar_bbox_to_pixel_indices for pixel conversion,
            then calculates exact chunk indices (not aligned blocks) for minimal traffic.

            Returns dict with pixel indices + chunk indices for bbox area.
            """
            # Get pixel indices from shared helper
            result = ASF._nisar_bbox_to_pixel_indices(h5, bbox, chunk_info_a, chunk_info_b)

            # Calculate exact chunk indices for bbox (not aligned blocks)
            chunk_az = chunk_info_a['chunk_shape'][0]
            chunk_rg = chunk_info_a['chunk_shape'][1]

            result['az_chunk_start'] = result['az_start'] // chunk_az
            result['az_chunk_end'] = (result['az_end'] + chunk_az - 1) // chunk_az
            result['rg_chunk_start_a'] = result['rg_start_a'] // chunk_rg
            result['rg_chunk_end_a'] = (result['rg_end_a'] + chunk_rg - 1) // chunk_rg

            # Handle frequencyB chunk indices if present
            if chunk_info_b and 'rg_start_b' in result:
                chunk_rg_b = chunk_info_b['chunk_shape'][1]
                result['rg_chunk_start_b'] = result['rg_start_b'] // chunk_rg_b
                result['rg_chunk_end_b'] = (result['rg_end_b'] + chunk_rg_b - 1) // chunk_rg_b

            return result

        def chunks_to_bbox_blocks(chunk_info, az_chunk_start, az_chunk_end, rg_chunk_start, rg_chunk_end):
            """Create optimized blocks for bbox download - only exact chunks needed.

            Groups chunks into ~64MB blocks for efficient HTTP requests while
            downloading only the chunks that cover the bbox area.
            Worker handles defragmentation (merging with 16MB gap threshold).

            Returns list of dicts with 'chunks' (raw chunk offsets) and 'total_size'.
            """
            if not chunk_info:
                return []

            chunks = chunk_info['chunks']
            chunk_lookup = {(c['row'], c['col']): c for c in chunks}

            # Collect only chunks within bbox range
            raw_chunks = []
            for row in range(az_chunk_start, az_chunk_end):
                for col in range(rg_chunk_start, rg_chunk_end):
                    if (row, col) in chunk_lookup:
                        c = chunk_lookup[(row, col)]
                        raw_chunks.append((c['offset'], c['size']))

            if not raw_chunks:
                return []

            # Pass raw chunks directly - worker handles defragmentation
            total_size = sum(size for _, size in raw_chunks)

            # Return as single block (or split if too large)
            if total_size <= MAX_MULTI_BLOCK:
                return [{
                    'az_block': 0,
                    'rg_block': 0,
                    'chunks': raw_chunks,
                    'total_size': total_size
                }]
            else:
                # Split into multiple blocks if too large
                # Group by azimuth rows
                blocks = []
                current_chunks = []
                current_size = 0
                BLOCK_TARGET = 64 * 1024 * 1024  # 64MB target per block

                for row in range(az_chunk_start, az_chunk_end):
                    row_chunks = []
                    for col in range(rg_chunk_start, rg_chunk_end):
                        if (row, col) in chunk_lookup:
                            c = chunk_lookup[(row, col)]
                            row_chunks.append((c['offset'], c['size']))

                    row_size = sum(s for _, s in row_chunks)
                    if current_size + row_size > BLOCK_TARGET and current_chunks:
                        # Flush current block
                        blocks.append({
                            'az_block': len(blocks),
                            'rg_block': 0,
                            'chunks': current_chunks,
                            'total_size': current_size
                        })
                        current_chunks = []
                        current_size = 0

                    current_chunks.extend(row_chunks)
                    current_size += row_size

                # Flush remaining
                if current_chunks:
                    blocks.append({
                        'az_block': len(blocks),
                        'rg_block': 0,
                        'chunks': current_chunks,
                        'total_size': current_size
                    })

                return blocks

        def chunks_to_aligned_blocks(chunk_info, az_block_start=None, az_block_end=None,
                                      rg_block_start=None, rg_block_end=None):
            """Group chunks into aligned blocks for cache efficiency.

            Each aligned block covers ALIGNED_AZ_CHUNKS × ALIGNED_RG_CHUNKS HDF5 chunks.
            All clients requesting the same scene get identical block boundaries,
            ensuring consistent cache hits.

            Parameters
            ----------
            chunk_info : dict
                Output from get_chunk_info()
            az_block_start, az_block_end : int, optional
                Azimuth block indices to extract (0-indexed). If None, extract all.
            rg_block_start, rg_block_end : int, optional
                Range block indices to extract. If None, extract all.

            Returns
            -------
            list of dict
                Each dict: {'az_block': int, 'rg_block': int, 'chunks': list, 'total_size': int}
                where 'chunks' is list of (offset, size) tuples
            """
            if not chunk_info:
                return []

            n_az = chunk_info['n_az']
            n_rg = chunk_info['n_rg']
            chunks = chunk_info['chunks']

            # Build lookup: (row, col) -> chunk
            chunk_lookup = {(c['row'], c['col']): c for c in chunks}

            # Calculate number of aligned blocks
            n_az_blocks = (n_az + ALIGNED_AZ_CHUNKS - 1) // ALIGNED_AZ_CHUNKS
            n_rg_blocks = (n_rg + ALIGNED_RG_CHUNKS - 1) // ALIGNED_RG_CHUNKS

            # Apply block range filters
            if az_block_start is None:
                az_block_start = 0
            if az_block_end is None:
                az_block_end = n_az_blocks
            if rg_block_start is None:
                rg_block_start = 0
            if rg_block_end is None:
                rg_block_end = n_rg_blocks

            aligned_blocks = []

            for ab in range(az_block_start, min(az_block_end, n_az_blocks)):
                for rb in range(rg_block_start, min(rg_block_end, n_rg_blocks)):
                    # Chunk range for this aligned block
                    az_start = ab * ALIGNED_AZ_CHUNKS
                    az_end = min((ab + 1) * ALIGNED_AZ_CHUNKS, n_az)
                    rg_start = rb * ALIGNED_RG_CHUNKS
                    rg_end = min((rb + 1) * ALIGNED_RG_CHUNKS, n_rg)

                    # Collect chunks for this block
                    raw_chunks = []
                    for row in range(az_start, az_end):
                        for col in range(rg_start, rg_end):
                            if (row, col) in chunk_lookup:
                                c = chunk_lookup[(row, col)]
                                raw_chunks.append((c['offset'], c['size']))

                    if raw_chunks:
                        # Pass raw chunks directly - worker handles defragmentation
                        total_size = sum(size for _, size in raw_chunks)
                        aligned_blocks.append({
                            'az_block': ab,
                            'rg_block': rb,
                            'chunks': raw_chunks,  # Raw chunks - worker merges with 16MB gap
                            'total_size': total_size
                        })

            return aligned_blocks

        def build_multi_offset_url(granule_id, block_chunks):
            """Build multi-offset URL for 25x25km aligned block.

            URL format: /{GRANULE}/off1_len1,off2_len2,.../ranges.bin

            Parameters
            ----------
            granule_id : str
                NISAR granule ID
            block_chunks : list of (offset, size) tuples
                Chunks to fetch, sorted by offset

            Returns
            -------
            str
                Multi-offset URL for cache proxy
            """
            offsets_str = ','.join(f"{off}_{size}" for off, size in block_chunks)
            return f"{_NISAR_CACHE_PROXY}/{granule_id}/{offsets_str}/ranges.bin"

        def fetch_aligned_block(granule_id, block_chunks, session=None):
            """Fetch aligned 25x25km block via multi-offset endpoint.

            Returns (content, cache_hit, chunk_offsets) tuple where chunk_offsets
            maps each original chunk offset to its position in the returned data.
            """
            total_size = sum(size for _, size in block_chunks)

            # Validate block size
            if total_size > MAX_MULTI_BLOCK:
                # Block too large - shouldn't happen with 25x25km blocks
                raise ValueError(f"Block size {total_size} exceeds max {MAX_MULTI_BLOCK}")

            url = build_multi_offset_url(granule_id, block_chunks)
            sess = session or requests.Session()
            resp = sess.get(url)
            resp.raise_for_status()
            if len(resp.content) != total_size:
                raise ValueError(
                    f"Response size mismatch: got {len(resp.content)}, expected {total_size}")

            cache_hit = resp.headers.get('cf-cache-status', '').upper() == 'HIT'

            # Build offset map: original chunk offset -> (position_in_data, size)
            chunk_offsets = {}
            pos = 0
            for off, size in block_chunks:
                chunk_offsets[off] = (pos, size)
                pos += size

            return resp.content, cache_hit, chunk_offsets

        def split_range_to_blocks(start, end):
            """Split byte range into deterministic 64-128MB blocks.

            All clients requesting the same range get identical blocks,
            ensuring consistent cache hits.
            """
            total = end - start
            if total <= MAX_BLOCK:
                return [(start, total)]

            # Calculate number of chunks so each is 64-128MB
            n_blocks = (total + MAX_BLOCK - 1) // MAX_BLOCK
            block_size = total // n_blocks

            # Ensure blocks are >= MIN_BLOCK
            while n_blocks > 1 and block_size < MIN_BLOCK:
                n_blocks -= 1
                block_size = total // n_blocks

            blocks = []
            offset = start
            for i in range(n_blocks):
                if i == n_blocks - 1:
                    # Last block gets remainder
                    blocks.append((offset, end - offset))
                else:
                    blocks.append((offset, block_size))
                    offset += block_size
            return blocks

        # Cache statistics for debug mode
        cache_stats = {'hits': 0, 'misses': 0, 'bytes_hit': 0, 'bytes_miss': 0}

        def fetch_cache_range(granule_id, offset, length, session=None):
            """Fetch range from cache proxy using new API.

            Returns (content, cache_hit) tuple.
            """
            url = f"{_NISAR_CACHE_PROXY}/{granule_id}/{offset}/{length}.bin"
            sess = session or requests.Session()
            resp = sess.get(url)
            resp.raise_for_status()
            # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
            cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                        resp.headers.get('cf-cache-status', '').upper() == 'HIT')
            return resp.content, cache_hit

        def fetch_region_via_cache(granule_id, start, end, pbar=None, session=None):
            """Fetch byte region using deterministic 64-128MB blocks.

            Splits large regions into cacheable blocks.
            Returns contiguous bytes from start to end.
            """
            sess = session or requests.Session()
            blocks = split_range_to_blocks(start, end)

            all_data = bytearray()
            for block_offset, block_length in blocks:
                block_data, cache_hit = fetch_cache_range(granule_id, block_offset, block_length, session=sess)
                all_data.extend(block_data)

                # Track cache stats
                if cache_hit:
                    cache_stats['hits'] += 1
                    cache_stats['bytes_hit'] += block_length
                else:
                    cache_stats['misses'] += 1
                    cache_stats['bytes_miss'] += block_length
                    # Log missed blocks for debugging
                    if debug:
                        print(f"    MISS: offset={block_offset} len={block_length/(1024**2):.0f}MB")

                if pbar:
                    pbar.update(block_length)
                    # Show cache stats in progress bar
                    pbar.set_postfix_str(f"H{cache_stats['hits']}M{cache_stats['misses']}")

            return bytes(all_data)

        def fetch_chunks_via_cache(granule_id, chunks, pbar=None, session=None):
            """Fetch HDF5 chunks by downloading their containing regions.

            Groups adjacent chunks into contiguous regions (1MB gap threshold),
            then fetches each region using deterministic 64-128MB blocks.

            Returns dict: {chunk_offset: chunk_bytes}
            """
            sess = session or requests.Session()

            # Use shared helper to merge chunks with 1MB gap threshold (consistent with aligned blocks)
            regions = ASF._nisar_merge_chunks_to_regions(chunks, gap_threshold=1024*1024)

            # Download each region and extract chunks
            chunk_data = {}
            for region_start, region_size, region_chunks in regions:
                region_end = region_start + region_size
                region_bytes = fetch_region_via_cache(
                    granule_id, region_start, region_end, pbar=pbar, session=sess
                )

                for chunk in region_chunks:
                    rel_offset = chunk['offset'] - region_start
                    chunk_data[chunk['offset']] = region_bytes[rel_offset:rel_offset + chunk['size']]

            return chunk_data

        def fetch_chunks_via_aligned_blocks(granule_id, chunk_info, pbar=None, session=None,
                                             az_block_start=None, az_block_end=None,
                                             rg_block_start=None, rg_block_end=None):
            """Fetch HDF5 chunks using aligned blocks for cache efficiency.

            Uses multi-offset API: /GRANULE/off1_len1,off2_len2,.../ranges.bin
            All clients requesting the same scene get identical block boundaries.

            Parameters
            ----------
            granule_id : str
                NISAR granule ID
            chunk_info : dict
                Output from get_chunk_info()
            pbar : tqdm, optional
                Progress bar to update
            session : requests.Session, optional
                HTTP session for connection reuse
            az_block_start, az_block_end : int, optional
                Azimuth block indices to extract (for bbox subsetting)
            rg_block_start, rg_block_end : int, optional
                Range block indices to extract (for bbox subsetting)

            Returns dict: {chunk_offset: chunk_bytes}
            """
            sess = session or requests.Session()

            # Get aligned 25x25km blocks
            aligned_blocks = chunks_to_aligned_blocks(
                chunk_info,
                az_block_start=az_block_start, az_block_end=az_block_end,
                rg_block_start=rg_block_start, rg_block_end=rg_block_end
            )

            if not aligned_blocks:
                return {}

            # Debug: show block alignment stats
            if debug:
                n_az = max(b['az_block'] for b in aligned_blocks) + 1
                n_rg = max(b['rg_block'] for b in aligned_blocks) + 1
                sizes_mb = [b['total_size'] / (1024**2) for b in aligned_blocks]
                print(f"    Aligned blocks: {n_az}×{n_rg} = {len(aligned_blocks)} blocks, "
                      f"size: {min(sizes_mb):.1f}-{max(sizes_mb):.1f}MB (avg {sum(sizes_mb)/len(sizes_mb):.1f}MB)")

            chunk_data = {}

            for block in aligned_blocks:
                chunks = block['chunks']  # Raw chunks (offset, size) - worker handles defragmentation
                total_size = block['total_size']

                # Try multi-offset API if block is in valid size range
                if total_size <= MAX_MULTI_BLOCK:
                    data, cache_hit, chunk_offsets = fetch_aligned_block(
                        granule_id, chunks, session=sess
                    )

                    if data is not None:
                        # Track cache stats
                        if cache_hit:
                            cache_stats['hits'] += 1
                            cache_stats['bytes_hit'] += total_size
                        else:
                            cache_stats['misses'] += 1
                            cache_stats['bytes_miss'] += total_size
                            if debug:
                                print(f"    MISS: aligned block ({block['az_block']},{block['rg_block']}) "
                                      f"{total_size/(1024**2):.0f}MB ({len(chunks)} chunks)")

                        # Extract individual chunks - worker returns them concatenated in order
                        for chunk_off, chunk_size in chunks:
                            pos, _ = chunk_offsets[chunk_off]
                            chunk_data[chunk_off] = data[pos:pos + chunk_size]

                        if pbar:
                            pbar.update(total_size)
                            if debug:
                                pbar.set_postfix_str(f"H{cache_stats['hits']}M{cache_stats['misses']}")

                        continue

                # Block exceeds MAX_MULTI_BLOCK - code bug, should not happen with proper aligned blocks
                raise RuntimeError(
                    f"Block ({block['az_block']},{block['rg_block']}) size={total_size/(1024**2):.1f}MB "
                    f"exceeds MAX_MULTI_BLOCK={MAX_MULTI_BLOCK/(1024**2):.0f}MB - this is a code issue"
                )

            return chunk_data

        def patch_hdf5_superblock(data):
            """Patch HDF5 superblock EOF to match buffer size."""
            data = bytearray(data)
            version = data[8]
            if version == 0:
                data[40:48] = struct.pack('<Q', len(data))
            else:
                data[28:36] = struct.pack('<Q', len(data))
                data[44:48] = struct.pack('<I', self._hdf5_lookup3_hash(bytes(data[0:44])))
            return data

        def download_granule_via_cache(granule_id, position=None, shared_pbar=None, shared_progress=None):
            """Download single NISAR granule via cache proxy.

            shared_pbar: optional (pbar_list, lock) tuple for threading parallel mode
            shared_progress: optional (counter, lock) tuple for loky parallel mode
            """
            track, frame, datetime_str = parse_nisar_granule_id(granule_id)

            # Output directory
            subdir = f"{track:03d}_{frame:03d}"
            out_dir = os.path.join(basedir, subdir)

            # Check what pols to download
            pols_to_download = polarizations if polarizations else ['HH', 'HV', 'VH', 'VV']

            # Check existing files
            if skip_exist:
                existing_files = []
                for pol in pols_to_download:
                    out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                    out_path = os.path.join(out_dir, out_name)
                    if os.path.exists(out_path):
                        existing_files.append(out_name)
                if len(existing_files) == len(pols_to_download):
                    if debug:
                        print(f"NISAR {track}_{frame}: all files exist, skipping")
                    return existing_files

            # Create session for connection reuse
            session = requests.Session()

            # Fetch metadata block (offset 0, 128MB for NISAR metadata)
            short_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}"

            metadata_raw, meta_cache_hit = fetch_cache_range(granule_id, 0, MAX_BLOCK, session=session)
            # Track metadata block in cache stats
            if meta_cache_hit:
                cache_stats['hits'] += 1
                cache_stats['bytes_hit'] += MAX_BLOCK
            else:
                cache_stats['misses'] += 1
                cache_stats['bytes_miss'] += MAX_BLOCK
            if debug:
                print(f"  {short_name}: metadata {MAX_BLOCK/(1024*1024):.0f}MB {'HIT' if meta_cache_hit else 'MISS'}")
            metadata_buffer = patch_hdf5_superblock(metadata_raw)

            # Parse metadata to get available pols and chunk info
            with h5py.File(BytesIO(bytes(metadata_buffer)), 'r') as h5_meta:
                swaths_path = 'science/LSAR/RSLC/swaths'
                if swaths_path not in h5_meta:
                    raise ValueError(f"Unsupported NISAR format: {swaths_path} not found")

                swaths_grp = h5_meta[swaths_path]
                if 'frequencyA' not in swaths_grp:
                    raise ValueError(f"Unsupported NISAR format: frequencyA not found")

                available_pols = [k for k in swaths_grp['frequencyA'].keys()
                                  if k in ['HH', 'HV', 'VH', 'VV']]
                has_freq_b = 'frequencyB' in swaths_grp
                freq_b_pols = ([k for k in swaths_grp['frequencyB'].keys()
                               if k in ['HH', 'HV', 'VH', 'VV']] if has_freq_b else [])

                # Filter to requested pols
                if polarizations is None:
                    pols_to_download = available_pols
                else:
                    pols_to_download = [p for p in polarizations if p in available_pols]
                    missing = set(polarizations) - set(available_pols)
                    if missing:
                        print(f"WARNING: Polarizations {missing} not available")

                # Frequency flags - normalize to list for consistent checking
                freq_list = [frequency] if isinstance(frequency, str) else frequency
                download_freq_a = 'A' in freq_list
                download_freq_b = 'B' in freq_list and has_freq_b

                # Get chunk info for all pols
                all_chunk_info = {}
                for pol in pols_to_download:
                    chunk_info_a = None
                    chunk_info_b = None
                    if download_freq_a:
                        chunk_info_a = get_chunk_info(h5_meta, pol, 'A')
                    if download_freq_b and pol in freq_b_pols:
                        chunk_info_b = get_chunk_info(h5_meta, pol, 'B')
                    all_chunk_info[pol] = (chunk_info_a, chunk_info_b)

            # Calculate total SLC size (actual chunk bytes, not span)
            total_slc_size = 0
            for pol, (chunk_info_a, chunk_info_b) in all_chunk_info.items():
                if chunk_info_a:
                    total_slc_size += sum(c['size'] for c in chunk_info_a['chunks'])
                if chunk_info_b:
                    total_slc_size += sum(c['size'] for c in chunk_info_b['chunks'])

            if total_slc_size == 0:
                raise ValueError(f"No SLC data found in {granule_id}")

            if debug:
                print(f"  Downloading {total_slc_size/(1024**3):.2f} GB SLC data via cache...")

            # Setup progress bar (shared or own)
            if shared_pbar:
                pbar_list, pbar_lock = shared_pbar
                with pbar_lock:
                    if pbar_list[0] is None:
                        # First worker initializes shared bar
                        pbar_list[0] = tqdm(total=0, unit='B', unit_scale=True,
                                           desc=f"Downloading {len(granule_ids)} granules",
                                           dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0)
                    pbar_list[0].total += total_slc_size
                    pbar_list[0].refresh()
                pbar = pbar_list[0]
                own_pbar = False
            else:
                pbar = tqdm(total=total_slc_size, unit='B', unit_scale=True, desc=short_name,
                           position=position, leave=True,
                           dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0)
                own_pbar = True

            downloaded_files = []
            os.makedirs(out_dir, exist_ok=True)

            for pol in pols_to_download:
                out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                out_path = os.path.join(out_dir, out_name)

                chunk_info_a, chunk_info_b = all_chunk_info[pol]

                # Read metadata from buffer
                metadata = self._read_nisar_metadata_direct(pol, metadata_buffer)

                # Download frequencyA chunks using aligned 25x25km blocks
                downloaded_data_a = None
                if chunk_info_a is not None:
                    downloaded_data_a = fetch_chunks_via_aligned_blocks(
                        granule_id,
                        chunk_info_a,
                        pbar=pbar if own_pbar else None,
                        session=session
                    )
                    if shared_pbar:
                        size = sum(c['size'] for c in chunk_info_a['chunks'])
                        with pbar_lock:
                            pbar.update(size)

                # Download frequencyB chunks using aligned 25x25km blocks
                downloaded_data_b = None
                if chunk_info_b is not None:
                    downloaded_data_b = fetch_chunks_via_aligned_blocks(
                        granule_id,
                        chunk_info_b,
                        pbar=pbar if own_pbar else None,
                        session=session
                    )
                    if shared_pbar:
                        size = sum(c['size'] for c in chunk_info_b['chunks'])
                        with pbar_lock:
                            pbar.update(size)

                # Write output HDF5
                self._write_nisar_pol_h5_from_bytes(
                    downloaded_data_a, chunk_info_a, metadata, pol, out_path,
                    track, frame, datetime_str, debug and (position is None),
                    downloaded_data_b=downloaded_data_b, chunk_info_b=chunk_info_b
                )

                downloaded_files.append(out_name)

            if own_pbar:
                pbar.close()

            # Print cache stats in debug mode
            if debug and (cache_stats['hits'] > 0 or cache_stats['misses'] > 0):
                total_blocks = cache_stats['hits'] + cache_stats['misses']
                hit_pct = 100 * cache_stats['hits'] / total_blocks if total_blocks > 0 else 0
                print(f"  Cache: {cache_stats['hits']} HIT / {cache_stats['misses']} MISS "
                      f"({hit_pct:.0f}% hit rate, {cache_stats['bytes_hit']/(1024**2):.0f} MB cached)")

            return downloaded_files

        def download_with_retry(granule, position=None):
            """Download single granule with retry logic."""
            for retry in range(retries):
                try:
                    return download_granule_via_cache(granule, position=position)
                except ValueError:
                    raise  # Format errors are permanent
                except Exception as e:
                    print(f"ERROR downloading {granule} (attempt {retry+1}/{retries}): {e}")
                    if retry + 1 == retries:
                        raise
                    time.sleep(timeout_second)

        # Process granules
        granule_ids = [g.replace('.h5', '') for g in granules]

        # Handle bbox parameter
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("bbox must be (west, south, east, north) in WGS84 coordinates")
            west, south, east, north = bbox
            if west >= east or south >= north:
                raise ValueError("Invalid bbox: west must be < east and south must be < north")
            if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
                raise ValueError("bbox coordinates must be valid WGS84 (lon: -180 to 180, lat: -90 to 90)")

        print(f"Downloading {len(granule_ids)} NISAR granule(s) via cache proxy (aligned blocks)...")

        # Unified download path - only backend differs for debug/sequential mode
        import joblib
        backend = 'sequential' if (n_jobs == 1 or debug) else 'loky'
        effective_n_jobs = 1 if backend == 'sequential' else n_jobs

        def download_aligned_block(gid, offsets_str, total_size):
            """Download aligned 25x25km block via multi-offset API."""
            import requests
            url = f"{_NISAR_CACHE_PROXY}/{gid}/{offsets_str}/ranges.bin"
            for retry in range(retries):
                try:
                    resp = requests.get(url, timeout=120)
                    resp.raise_for_status()
                    if len(resp.content) != total_size:
                        raise ValueError(
                            f"Response size mismatch: got {len(resp.content)}, expected {total_size}")
                    # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
                    cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                                resp.headers.get('cf-cache-status', '').upper() == 'HIT')
                    return (offsets_str, resp.content, cache_hit)
                except Exception as e:
                    if debug:
                        print(f'ERROR: block download attempt {retry+1}/{retries} failed: {e}')
                    if retry + 1 == retries:
                        raise
                    time.sleep(timeout_second)

        def download_single_chunk(gid, offset, length):
            """Download single chunk via single-block API (for small blocks)."""
            import requests
            url = f"{_NISAR_CACHE_PROXY}/{gid}/{offset}/{length}.bin"
            for retry in range(retries):
                try:
                    resp = requests.get(url, timeout=120)
                    resp.raise_for_status()
                    return (offset, length, resp.content)
                except Exception as e:
                    if debug:
                        print(f'ERROR: single chunk download attempt {retry+1}/{retries} failed: {e}')
                    if retry + 1 == retries:
                        raise
                    time.sleep(timeout_second)

        all_downloaded = []

        for gid in granule_ids:

            # 1. Fetch metadata for this granule
            if debug:
                print(f"Fetching metadata for {gid}...")
            meta_raw, _ = fetch_cache_range(gid, 0, MAX_BLOCK)
            meta_buf = patch_hdf5_superblock(meta_raw)
            if debug:
                print(f"  Metadata: {len(meta_buf)/(1024**2):.1f}MB")

            with h5py.File(BytesIO(bytes(meta_buf)), 'r') as h5:
                swaths = h5['science/LSAR/RSLC/swaths']
                avail_pols = [k for k in swaths['frequencyA'].keys() if k in ['HH','HV','VH','VV']]
                pols = [p for p in (polarizations or avail_pols) if p in avail_pols]
                has_freq_b = 'frequencyB' in swaths
                freq_b_pols = [k for k in swaths['frequencyB'].keys() if k in ['HH','HV','VH','VV']] if has_freq_b else []
                # Normalize frequency to list for consistent checking
                freq_list = [frequency] if isinstance(frequency, str) else frequency
                dl_a = 'A' in freq_list
                dl_b = 'B' in freq_list and has_freq_b

                # Pre-collect chunk info for all pols while h5 is open
                all_chunk_info = {}
                for pol in pols:
                    ci_a, ci_b = None, None
                    if dl_a:
                        ci_a = get_chunk_info(h5, pol, 'A')
                    if dl_b and pol in freq_b_pols:
                        ci_b = get_chunk_info(h5, pol, 'B')
                    all_chunk_info[pol] = (ci_a, ci_b)

                # Calculate bbox chunk indices if bbox provided
                bbox_info = None
                if bbox is not None:
                    # Use first available chunk_info for bbox calculation
                    first_ci_a = next((ci_a for ci_a, _ in all_chunk_info.values() if ci_a), None)
                    first_ci_b = next((ci_b for _, ci_b in all_chunk_info.values() if ci_b), None)
                    if first_ci_a or first_ci_b:
                        bbox_info = bbox_to_block_indices(h5, bbox, first_ci_a or first_ci_b, first_ci_b)
                        if debug:
                            print(f"  Bbox {bbox} -> chunks az[{bbox_info['az_chunk_start']}:{bbox_info['az_chunk_end']}], "
                                  f"rg_a[{bbox_info.get('rg_chunk_start_a', 'N/A')}:{bbox_info.get('rg_chunk_end_a', 'N/A')}]")

            track, frame, datetime_str = parse_nisar_granule_id(gid)
            subdir = f"{track:03d}_{frame:03d}"
            out_dir = os.path.join(basedir, subdir)
            os.makedirs(out_dir, exist_ok=True)

            # Process one polarization at a time to limit memory usage
            for pol in pols:
                ci_a, ci_b = all_chunk_info[pol]
                blocks_to_download = []
                small_chunks_to_download = []

                # Use bbox-optimized blocks when bbox provided, otherwise aligned blocks
                if bbox_info:
                    # Bbox mode: download only exact chunks needed
                    az_start = bbox_info['az_chunk_start']
                    az_end = bbox_info['az_chunk_end']

                    if ci_a:
                        rg_start = bbox_info.get('rg_chunk_start_a', 0)
                        rg_end = bbox_info.get('rg_chunk_end_a', ci_a['n_rg'])
                        for block in chunks_to_bbox_blocks(ci_a, az_start, az_end, rg_start, rg_end):
                            if block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])

                    if ci_b:
                        rg_start = bbox_info.get('rg_chunk_start_b', 0)
                        rg_end = bbox_info.get('rg_chunk_end_b', ci_b['n_rg'])
                        for block in chunks_to_bbox_blocks(ci_b, az_start, az_end, rg_start, rg_end):
                            if block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])
                else:
                    # Full download: use cache-aligned blocks
                    if ci_a:
                        for block in chunks_to_aligned_blocks(ci_a):
                            if block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])

                    if ci_b:
                        for block in chunks_to_aligned_blocks(ci_b):
                            if block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])

                # Debug: show block stats for this pol
                if debug and blocks_to_download:
                    sizes_mb = [blk[1] / (1024**2) for blk in blocks_to_download]
                    mode = "bbox-optimized" if bbox_info else "aligned"
                    print(f"    {pol}: {len(blocks_to_download)} {mode} blocks, "
                          f"size: {min(sizes_mb):.1f}-{max(sizes_mb):.1f}MB (avg {sum(sizes_mb)/len(sizes_mb):.1f}MB)")

                total_size = sum(blk[1] for blk in blocks_to_download) + sum(c[1] for c in small_chunks_to_download)
                block_data = {}  # {chunk_offset: chunk_bytes}
                cache_hits = 0
                cache_misses = 0

                with tqdm(desc=f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}",
                          total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                          smoothing=0) as pbar:

                    # Download blocks via multi-offset API
                    if blocks_to_download:
                        results = joblib.Parallel(n_jobs=effective_n_jobs, backend=backend, return_as='generator')(
                            joblib.delayed(download_aligned_block)(gid, offsets_str, total_sz)
                            for offsets_str, total_sz, _ in blocks_to_download
                        )
                        for (offsets_str, total_sz, chunks), (_, data, cache_hit) in zip(blocks_to_download, results):
                            if cache_hit:
                                cache_hits += 1
                            else:
                                cache_misses += 1
                            pbar.update(total_sz)
                            pbar.set_postfix_str(f"H{cache_hits}M{cache_misses}")
                            # Extract chunks - worker returns them concatenated in order
                            pos = 0
                            for chunk_off, chunk_size in chunks:
                                block_data[chunk_off] = data[pos:pos + chunk_size]
                                pos += chunk_size

                    # Download small chunks via single-block API
                    if small_chunks_to_download:
                        results = joblib.Parallel(n_jobs=effective_n_jobs, backend=backend, return_as='generator')(
                            joblib.delayed(download_single_chunk)(gid, off, size)
                            for off, size in small_chunks_to_download
                        )
                        for off, size, data in results:
                            pbar.update(size)
                            block_data[off] = data

                # Write file for this pol immediately
                out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                out_path = os.path.join(out_dir, out_name)
                metadata = self._read_nisar_metadata_direct(pol, meta_buf)

                def extract_chunks_from_data(ci, freq_label=''):
                    if not ci:
                        return None
                    # Only extract chunks that were actually downloaded (subset when bbox applied)
                    result = {}
                    missing = 0
                    empty = 0
                    for c in ci['chunks']:
                        if c['offset'] not in block_data:
                            missing += 1
                        else:
                            chunk_bytes = block_data[c['offset']]
                            if len(chunk_bytes) == 0:
                                empty += 1
                            result[c['offset']] = chunk_bytes
                    if missing > 0 or empty > 0:
                        raise ValueError(
                            f"freq{freq_label} {pol}: {missing} missing + {empty} empty chunks "
                            f"out of {len(ci['chunks'])} total ({len(block_data)} in block_data)")
                    return result

                data_a = extract_chunks_from_data(ci_a, 'A')
                data_b = extract_chunks_from_data(ci_b, 'B')

                self._write_nisar_pol_h5_from_bytes(
                    data_a, ci_a, metadata, pol, out_path,
                    track, frame, datetime_str, False,
                    downloaded_data_b=data_b, chunk_info_b=ci_b,
                    crop_info=bbox_info
                )
                all_downloaded.append(out_name)

                # Free memory before next pol
                del block_data

            # Free metadata buffer after all pols done
            del meta_buf

        if all_downloaded:
            return pd.DataFrame({'file': all_downloaded})
        return None

    def _detect_nisar_layout_fast(self, url, auth_tuple, file_size, http_session=None, pbar=None):
        """Download 128MB metadata block and detect layout.

        Downloads first 128MB (same as cache path), patches HDF5 superblock,
        and tries to read metadata. Contains all metadata including geolocationGrid.

        Returns: (layout, metadata_buffer) where layout is 'A' or 'B' and
                 metadata_buffer is the patched 128MB buffer for reuse.
        """
        import requests
        import struct
        import h5py
        from io import BytesIO

        METADATA_SIZE = 128 * 1024 * 1024  # 128 MB - matches cache path

        # Download first 128MB (reuse session if provided)
        headers = {'Range': f'bytes=0-{METADATA_SIZE-1}'}
        if http_session:
            response = http_session.get(url, headers=headers, stream=True)
        else:
            response = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
        response.raise_for_status()

        # Stream download with progress
        chunks = []
        for chunk in response.iter_content(chunk_size=1024*1024):
            chunks.append(chunk)
            if pbar:
                pbar.update(len(chunk))
        data = bytearray(b''.join(chunks))

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
            print(f"    Metadata requires {max_offset:,} bytes ({max_offset/(1024**2):.1f} MB)")
            print(f"    Downloading {download_size:,} bytes ({download_size/(1024**2):.1f} MB)...")

        # Step 2: Download metadata block in single Range request
        headers = {'Range': f'bytes=0-{download_size-1}'}
        response = requests.get(url, headers=headers, auth=auth_tuple, stream=True)
        response.raise_for_status()

        data = bytearray()
        # Always show progress for metadata (it's ~100 MB)
        # out_name is like "NSR_172_008_20251204T024618_HH.h5"
        desc = f"{out_name[:-3]} metadata" if out_name else f"{pol} metadata"
        with tqdm(total=download_size, unit='B', unit_scale=True, desc=desc,
                  position=position, leave=(position is None), smoothing=0) as pbar:
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
                                        downloaded_data_b=None, chunk_info_b=None,
                                        crop_info=None, data_offset_a=None, data_offset_b=None):
        """Write single-polarization NISAR HDF5 from downloaded byte data.

        Copies ALL metadata from source file (geolocationGrid, attitude, calibration,
        processingInformation, etc.) - only excludes SLC data for other polarizations.

        Supports three modes:
        - FrequencyA only: downloaded_data present, downloaded_data_b is None
        - FrequencyB only: downloaded_data is None, downloaded_data_b present
        - Both frequencies: both present

        When crop_info is provided, crops SLC data and coordinate arrays to bbox extent.

        Parameters
        ----------
        downloaded_data : bytes, dict, or None
            Raw bytes from HTTP Range request covering frequencyA chunks,
            OR dict mapping chunk_offset -> chunk_bytes (for cache proxy).
        chunk_info : dict or None
            Chunk metadata from get_chunk_info() for frequencyA.
        metadata : dict
            ALL metadata from _read_nisar_metadata().
        downloaded_data_b : bytes, dict, or None
            Raw bytes or dict for frequencyB (ionospheric correction / quick look).
        chunk_info_b : dict, optional
            Chunk metadata for frequencyB.
        data_offset_a : int, optional
            Override min_offset for bytes mode (for bbox-filtered downloads).
        data_offset_b : int, optional
            Override min_offset for frequencyB bytes.
        """
        import h5py
        import numpy as np
        from io import BytesIO
        from tqdm.auto import tqdm

        import os

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

        # Calculate chunk-aligned crop extents if cropping
        crop_az_start, crop_az_end = 0, None
        crop_rg_start_a, crop_rg_end_a = 0, None
        crop_rg_start_b, crop_rg_end_b = 0, None

        if crop_info:
            # Get chunk-aligned bounds (full chunks, not pixels)
            if has_freq_a:
                chunk_az = chunk_info['chunk_shape'][0]
                chunk_rg = chunk_info['chunk_shape'][1]
                # Align to chunk boundaries
                crop_az_start = (crop_info['az_start'] // chunk_az) * chunk_az
                crop_az_end = ((crop_info['az_end'] + chunk_az - 1) // chunk_az) * chunk_az
                # Only set freqA crop bounds if they exist in crop_info
                if 'rg_start_a' in crop_info:
                    crop_rg_start_a = (crop_info['rg_start_a'] // chunk_rg) * chunk_rg
                    crop_rg_end_a = ((crop_info['rg_end_a'] + chunk_rg - 1) // chunk_rg) * chunk_rg
            if has_freq_b:
                chunk_az_b = chunk_info_b['chunk_shape'][0]
                chunk_rg_b = chunk_info_b['chunk_shape'][1]
                if crop_az_start == 0:  # Not set by freqA
                    crop_az_start = (crop_info['az_start'] // chunk_az_b) * chunk_az_b
                    crop_az_end = ((crop_info['az_end'] + chunk_az_b - 1) // chunk_az_b) * chunk_az_b
                # Only set freqB crop bounds if they exist in crop_info
                if 'rg_start_b' in crop_info:
                    crop_rg_start_b = (crop_info['rg_start_b'] // chunk_rg_b) * chunk_rg_b
                    crop_rg_end_b = ((crop_info['rg_end_b'] + chunk_rg_b - 1) // chunk_rg_b) * chunk_rg_b

        # Build HDF5 in memory
        mem_buffer = BytesIO()

        with h5py.File(mem_buffer, 'w') as h5_mem:
            # 1. Write frequencyA SLC if available
            if has_freq_a:
                orig_shape = chunk_info['shape']
                chunk_shape = chunk_info['chunk_shape']
                dtype = chunk_info['dtype']
                compression = chunk_info['compression']
                compression_opts = chunk_info['compression_opts']
                shuffle = chunk_info.get('shuffle', False)
                min_offset = data_offset_a if data_offset_a is not None else chunk_info['min_offset']

                # Calculate output shape (cropped or full)
                if crop_info:
                    out_shape = (min(crop_az_end, orig_shape[0]) - crop_az_start,
                                 min(crop_rg_end_a, orig_shape[1]) - crop_rg_start_a)
                else:
                    out_shape = orig_shape

                dst_slc = h5_mem.create_dataset(
                    f'science/LSAR/RSLC/swaths/frequencyA/{pol}',
                    shape=out_shape, dtype=dtype, chunks=chunk_shape,
                    compression=compression, compression_opts=compression_opts,
                    shuffle=shuffle
                )

                chunk_iter = tqdm(chunk_info['chunks'], desc='    Writing freqA SLC', leave=False) if debug else chunk_info['chunks']
                is_dict = isinstance(downloaded_data, dict)
                for chunk in chunk_iter:
                    row, col = chunk['row'], chunk['col']
                    orig_coord = chunk['coord']  # (row_idx * chunk_az, col_idx * chunk_rg)

                    # Skip chunks outside crop area
                    if crop_info:
                        pixel_row = row * chunk_shape[0]
                        pixel_col = col * chunk_shape[1]
                        if pixel_row < crop_az_start or pixel_row >= crop_az_end:
                            continue
                        if pixel_col < crop_rg_start_a or pixel_col >= crop_rg_end_a:
                            continue
                        # Adjust coord for cropped output
                        new_coord = (pixel_row - crop_az_start, pixel_col - crop_rg_start_a)
                    else:
                        new_coord = orig_coord

                    if is_dict:
                        chunk_bytes = downloaded_data[chunk['offset']]
                    else:
                        offset_in_data = chunk['offset'] - min_offset
                        chunk_bytes = downloaded_data[offset_in_data:offset_in_data + chunk['size']]
                    dst_slc.id.write_direct_chunk(new_coord, chunk_bytes)

            # 2. Write frequencyB SLC if available
            if has_freq_b:
                orig_shape_b = chunk_info_b['shape']
                chunk_shape_b = chunk_info_b['chunk_shape']
                dtype_b = chunk_info_b['dtype']
                compression_b = chunk_info_b['compression']
                compression_opts_b = chunk_info_b['compression_opts']
                shuffle_b = chunk_info_b.get('shuffle', False)
                min_offset_b = data_offset_b if data_offset_b is not None else chunk_info_b['min_offset']

                # Calculate output shape (cropped or full)
                if crop_info:
                    out_shape_b = (min(crop_az_end, orig_shape_b[0]) - crop_az_start,
                                   min(crop_rg_end_b, orig_shape_b[1]) - crop_rg_start_b)
                else:
                    out_shape_b = orig_shape_b

                dst_slc_b = h5_mem.create_dataset(
                    f'science/LSAR/RSLC/swaths/frequencyB/{pol}',
                    shape=out_shape_b, dtype=dtype_b, chunks=chunk_shape_b,
                    compression=compression_b, compression_opts=compression_opts_b,
                    shuffle=shuffle_b
                )

                chunk_iter_b = tqdm(chunk_info_b['chunks'], desc='    Writing freqB SLC', leave=False) if debug else chunk_info_b['chunks']
                is_dict_b = isinstance(downloaded_data_b, dict)
                for chunk in chunk_iter_b:
                    row, col = chunk['row'], chunk['col']
                    orig_coord = chunk['coord']

                    # Skip chunks outside crop area
                    if crop_info:
                        pixel_row = row * chunk_shape_b[0]
                        pixel_col = col * chunk_shape_b[1]
                        if pixel_row < crop_az_start or pixel_row >= crop_az_end:
                            continue
                        if pixel_col < crop_rg_start_b or pixel_col >= crop_rg_end_b:
                            continue
                        new_coord = (pixel_row - crop_az_start, pixel_col - crop_rg_start_b)
                    else:
                        new_coord = orig_coord

                    if is_dict_b:
                        chunk_bytes = downloaded_data_b[chunk['offset']]
                    else:
                        offset_in_data = chunk['offset'] - min_offset_b
                        chunk_bytes = downloaded_data_b[offset_in_data:offset_in_data + chunk['size']]
                    dst_slc_b.id.write_direct_chunk(new_coord, chunk_bytes)

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

                # Get data, potentially cropping coordinate arrays
                data = ds_info['data']
                if crop_info:
                    # Crop zeroDopplerTime (shared azimuth coordinate)
                    if ds_path == 'science/LSAR/RSLC/swaths/zeroDopplerTime':
                        az_end = min(crop_az_end, len(data)) if crop_az_end else len(data)
                        data = data[crop_az_start:az_end]
                    # Crop frequencyA slantRange
                    elif ds_path == 'science/LSAR/RSLC/swaths/frequencyA/slantRange':
                        rg_end = min(crop_rg_end_a, len(data)) if crop_rg_end_a else len(data)
                        data = data[crop_rg_start_a:rg_end]
                    # Crop frequencyB slantRange
                    elif ds_path == 'science/LSAR/RSLC/swaths/frequencyB/slantRange':
                        rg_end = min(crop_rg_end_b, len(data)) if crop_rg_end_b else len(data)
                        data = data[crop_rg_start_b:rg_end]
                    # Crop validSamplesSubSwath arrays (azimuth dimension)
                    elif 'validSamplesSubSwath' in ds_path and len(data.shape) == 2:
                        az_end = min(crop_az_end, data.shape[0]) if crop_az_end else data.shape[0]
                        data = data[crop_az_start:az_end, :]

                # GeolocationGrid: select sea level height layer (index 1 = 0m)
                # Keep 3D shape (1, az, rg) for compatibility - saves ~162MB (20 layers -> 1 layer)
                if 'geolocationGrid' in ds_path and len(data.shape) == 3:
                    data = data[1:2, :, :]
                # GeolocationGrid 1D height coordinate: keep only sea level value
                elif 'geolocationGrid' in ds_path and len(data.shape) == 1 and 'height' in ds_path.lower():
                    data = data[1:2]

                # Create dataset with compression for metadata (matches source HDF5)
                try:
                    # Use gzip compression for arrays, skip for scalars
                    if hasattr(data, 'shape') and len(data.shape) > 0 and data.size > 100:
                        ds = h5_mem.create_dataset(ds_path, data=data, compression='gzip', compression_opts=4)
                    else:
                        ds = h5_mem.create_dataset(ds_path, data=data)
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

        # Single disk write to temp file, then atomic rename
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            f.write(mem_buffer.getvalue())
        os.rename(tmp_path, out_path)

        if debug:
            file_size = os.path.getsize(out_path)
            print(f"    Done: {file_size / (1024**3):.2f} GB written")

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None,
               platform='SENTINEL-1', processingLevel='auto', polarization='VV', beamMode='IW'):
        import geopandas as gpd
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
