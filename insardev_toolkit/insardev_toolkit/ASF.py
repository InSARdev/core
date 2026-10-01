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
# the (URL, error type) of the Earthdata token failures warned about in this process (_ASFSession.auth_with_creds)
_TOKEN_WARNED = set()
_ASF_SEARCH_URL = 'https://api.daac.asf.alaska.edu/services/search/param'
# granules per catalog request, matching the ASF/CMR page size
_ASF_GRANULE_CHUNK = 250


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


def _short_reason(error):
    """The short reason of a request that got no answer: the text of the operating system error inside the requests
    error, such as 'Connection refused', or the name of the error type when it holds none."""
    seen, todo = set(), [error]
    while todo:
        e = todo.pop()
        if not isinstance(e, BaseException) or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, OSError) and e.strerror:
            return e.strerror
        todo += [arg for arg in e.args if isinstance(arg, BaseException)]
        todo += [getattr(e, 'reason', None), e.__cause__, e.__context__]
    return type(error).__name__


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
        from .HTTP import send, returned
        try:
            # (connect, read) timeouts, as the other toolkit requests (HTTP.fetch)
            response = send(
                self.post,
                token_url,
                data={'grant_type': 'client_credentials'},
                auth=(self._username, self._password),
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=(10, 300)
            )
            token = response.json().get('access_token') if response.status_code == 200 else None
            if token:
                self.headers['Authorization'] = f"Bearer {token}"
                self._authenticated = True
                return self
            key, failure = (token_url, None), f'{token_url}: no access token in the answer'
        except Exception as e:
            # an HTTP error names the URL and what the server returned (HTTP.send); an error with no answer gives
            # its short reason, as its full text holds object addresses that differ for every session
            key = (token_url, type(e))
            failure = str(e) if returned(e) else f'{token_url}: {_short_reason(e)}'

        # Fallback: use basic auth (works for many ASF endpoints); one WARNING per URL and error type in a process
        if key not in _TOKEN_WARNED:
            _TOKEN_WARNED.add(key)
            print(f'WARNING: Earthdata token request failed ({failure}), basic auth is used.')
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


def _asf_query(params, retries=30, timeout_second=3):
    """POST a query to the ASF SearchAPI and return the parsed GeoJSON.

    POST is used instead of GET because query strings are limited to a few
    kilobytes by the server: a granule list of ~180 bursts, or a detailed WKT
    geometry, exceeds that and the request is rejected with HTTP 414
    (Request-URI Too Large). The SearchAPI accepts exactly the same parameters
    form-encoded in the request body, without any length limit.

    Parameters
    ----------
    params : dict
        SearchAPI parameters, e.g. {'granule_list': '...', 'output': 'geojson'}.
    retries : int, optional
        Number of attempts, the first one included; 0 makes one attempt (HTTP.attempts). A failure that a retry
        cannot change (HTTP.final) raises at its first attempt. Default 30.
    timeout_second : int, optional
        Seconds to wait between attempts. Default 3.

    Returns
    -------
    dict
        Parsed GeoJSON response.
    """
    import time
    from .HTTP import final, attempts, send
    n = attempts(retries)
    for attempt in range(n):
        try:
            # (connect, read) timeouts: the catalog is slow for large granule lists
            response = send(requests.post, _ASF_SEARCH_URL, data=params, timeout=(30, 300))
            return response.json()
        except Exception as e:
            stop = final(e)
            if stop:
                print(f'ASF catalog search attempt {attempt+1} failed (not retried): {e}')
            if stop or attempt + 1 == n:
                raise
            print(f'ASF catalog search attempt {attempt+1} failed: {e}, retrying in {timeout_second}s...')
            time.sleep(timeout_second)


def _asf_granule_search(granule_list):
    """Search ASF by granule names (burst IDs or product names).

    The list is split into chunks so that a single request stays within the
    catalog limits regardless of how many granules are requested.

    Parameters
    ----------
    granule_list : list
        List of granule identifiers.

    Returns
    -------
    list
        List of _ASFSearchResult objects, in the order the granules were requested.
    """
    if not granule_list:
        return []

    if isinstance(granule_list, str):
        granule_list = [granule_list]
    else:
        granule_list = list(granule_list)

    features = {}
    for chunk in [granule_list[idx:idx + _ASF_GRANULE_CHUNK]
                  for idx in range(0, len(granule_list), _ASF_GRANULE_CHUNK)]:
        # ASF SearchAPI accepts comma-separated granule list
        data = _asf_query({'granule_list': ','.join(chunk), 'output': 'geojson'})
        for feature in data.get('features', []):
            fileID = feature.get('properties', {}).get('fileID')
            # the catalog can report the same granule twice, keep the first entry
            features.setdefault(fileID, feature)

    # return in the requested order, ignoring granules missing from the catalog
    ordered = [features.pop(granule) for granule in granule_list if granule in features]
    # keep any extra entries the catalog returned under a different name
    return [_ASFSearchResult(f) for f in ordered + list(features.values())]


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

    data = _asf_query(params)
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


def _asf_burst_records(bursts, missing_note=None):
    """Find Sentinel-1 bursts in the ASF catalog by name, in one batched granule search.

    The catalog occasionally omits one polarization channel of an otherwise complete dual-polarization scene. The
    channels differ only by the polarization in the name and in the url path, so the record of such a burst is rebuilt
    from a sibling channel, which a second search finds, and a NOTE names the record it was rebuilt from.

    Parameters
    ----------
    bursts : list
        Burst names, e.g. 'S1_262885_IW2_20190702T032452_VV_69C5-BURST'.
    missing_note : str, optional
        The NOTE printed for each burst that neither the catalog nor a sibling gives, with '{burst}' for its name.
        None prints nothing.

    Returns
    -------
    list
        _ASFSearchResult records: those of the catalog, in the requested order, then those rebuilt from a sibling.
        A burst that neither the catalog nor a sibling gives has none.
    """
    from tqdm.auto import tqdm

    with tqdm(desc=f'Downloading ASF Catalog'.ljust(25), total=1) as pbar:
        results = asf_search.granule_search(bursts)
        pbar.update(1)

    def polarization_siblings(burst):
        # the polarization channels of a scene share every part of the name but the channel
        parts = burst.split('_')
        names = ['_'.join(parts[:4] + [pol] + parts[5:]) for pol in ['VV', 'VH', 'HH', 'HV']]
        return [name for name in names if name != burst]

    def replace_polarization(url, polarization):
        # burst urls end with .../<subswath>/<polarization>/<burstIndex>.<ext>
        parts = url.split('/')
        parts[-2] = polarization
        return '/'.join(parts)

    catalog = {result.geojson()['properties']['fileID']: result for result in results}
    bursts_absent = [burst for burst in bursts if burst not in catalog]
    if bursts_absent:
        # a sibling is not necessarily requested here, it can be downloaded already
        siblings = {name for burst in bursts_absent for name in polarization_siblings(burst)}
        for result in asf_search.granule_search(sorted(siblings - set(catalog))):
            catalog.setdefault(result.geojson()['properties']['fileID'], result)
    for burst in bursts_absent:
        sibling = next((catalog[name] for name in polarization_siblings(burst)
                        if name in catalog), None)
        if sibling is None:
            if missing_note is not None:
                print(missing_note.format(burst=burst))
            continue
        feature = sibling.geojson()
        properties = dict(feature['properties'])
        polarization = burst.split('_')[4]
        properties['fileID'] = burst
        properties['sceneName'] = burst
        # the catalog names the served TIFF; the measurement is stored as <burst>.nc (or the legacy <burst>.tiff)
        properties['fileName'] = f'{burst}.tiff'
        properties['polarization'] = polarization
        properties['url'] = replace_polarization(properties['url'], polarization)
        properties['additionalUrls'] = [replace_polarization(url, polarization)
                                        for url in properties['additionalUrls']]
        results.append(_ASFSearchResult(dict(feature, properties=properties)))
        print(f'NOTE: burst {burst} is missing in the ASF catalog, '
              f'catalog record rebuilt from {feature["properties"]["fileID"]}.')
    return results

# Cloudflare Worker cache proxy for S1 bursts (handles auth internally)
_S1_CACHE_PROXY = 'https://s1-cache-asf.insar.dev'
_ASF_BURST_HOST = 'https://sentinel1-burst.asf.alaska.edu'

# Cloudflare Worker cache proxy for NISAR (handles auth internally)
# API: /GRANULE_ID/OFFSET.bin → 128MB block at OFFSET
_NISAR_CACHE_PROXY = 'https://nisar-cache-asf.insar.dev'
_NISAR_BLOCK_SIZE = 128 * 1024 * 1024  # 128 MB blocks
# Minimum side of a NISAR bbox crop on the ground: smaller crops cannot be processed accurately
_NISAR_MIN_CROP_KM = 20


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
            True if all 4 files exist and are regular files. An empty one raises.
        """
        import os
        from glob import glob
        from .utils_files import exists

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

        # Define expected file paths: the measurement is <burst>.nc, or the legacy <burst>.tiff
        from .utils_S1 import measurement_path
        files = [
            measurement_path(os.path.join(burst_dir, 'measurement'), burst),
            os.path.join(burst_dir, 'annotation', f'{burst}.xml'),
            os.path.join(burst_dir, 'calibration', f'{burst}.xml'),
            os.path.join(burst_dir, 'noise', f'{burst}.xml'),
        ]

        # Check all files: exist and regular file; every one is checked, so that an empty one raises
        present = [exists(filepath) and os.path.isfile(filepath) for filepath in files]
        return all(present)

    @staticmethod
    def _nisar_path(basedir, granule_id, polarization):
        """The output file of one polarization of a NISAR granule."""
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
        return os.path.join(basedir, subdir, out_name)

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
            True if output file exists. An empty one raises.
        """
        from .utils_files import exists
        return exists(ASF._nisar_path(basedir, granule_id, polarization))

    @staticmethod
    def _nisar_complete(basedir, granule_id, polarizations):
        """Whether every file of a NISAR granule already exists, for skip_exist=True.

        Parameters
        ----------
        basedir : str
            Base directory containing NISAR data.
        granule_id : str
            Full NISAR granule ID.
        polarizations : list or None
            The requested polarizations. None: the polarizations the product has, read from the metadata of a file
            of the granule that exists (the listOfPolarizations of the frequencies it holds).

        Returns
        -------
        bool
            True when every file of the granule exists; False with polarizations=None and no file of the granule.
            An empty file raises.
        """
        import h5py

        def present(pols):
            # every file is checked, so that an empty one raises
            return [ASF._nisar_exists(basedir, granule_id, p) for p in pols]

        linear = ['HH', 'HV', 'VH', 'VV']
        if polarizations is None:
            first = next((p for p in linear if ASF._nisar_exists(basedir, granule_id, p)), None)
            if first is None:
                return False
            path = ASF._nisar_path(basedir, granule_id, first)
            # a file holds the frequencies it was downloaded with, each with the product's list
            with h5py.File(path, 'r') as h5:
                keys = [f'science/LSAR/RSLC/swaths/{freq}/listOfPolarizations' for freq in ('frequencyA', 'frequencyB')]
                lists = [h5[key][()] for key in keys if key in h5]
            if not lists:
                print(f'WARNING: {path} has no science/LSAR/RSLC/swaths/frequency*/listOfPolarizations, so the '
                      f'polarizations of the product are unknown here: its metadata is downloaded to find them, and '
                      f'the files that exist are not downloaded again.')
                present(linear)
                return False
            listed = {v.decode() if isinstance(v, bytes) else str(v) for values in lists for v in values}
            polarizations = [p for p in linear if p in listed]
        return bool(polarizations) and all(present(polarizations))

    @staticmethod
    def _nisar_pols(polarizations, available, granule_id):
        """The polarizations of a NISAR granule to download: the requested ones that the product has (a WARNING
        names the others), or every one it has for polarizations=None."""
        if polarizations is None:
            return list(available)
        missing = [p for p in polarizations if p not in available]
        if missing:
            print(f'WARNING: Polarizations {missing} not available in {granule_id}')
        return [p for p in polarizations if p in available]

    # https://asf.alaska.edu/datasets/data-sets/derived-data-sets/sentinel-1-bursts/
    def download(self, basedir, bursts, polarization=None, frequency=None, bbox=None, session=None, n_jobs=None, joblib_backend='loky', skip_exist=True,
                        retries=30, timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
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
            An empty list raises a ValueError.
        frequency : str or list, required for NISAR
            Which frequency band(s) to download (NISAR stores two frequencies with different resolutions):
            - 'A': frequencyA (20MHz bandwidth, ~7m range resolution, ~10GB per scene)
            - 'B': frequencyB (5MHz bandwidth, ~25m range resolution, ~1.5GB per scene)
            - ['A', 'B']: Both frequencies in same file (~14GB per scene)
        bbox : tuple or None, optional (NISAR only)
            Bounding box in WGS84 coordinates: (west, south, east, north).
            When provided, only the part of the frame covering the bbox is downloaded: the exact radar extent
            of the bbox over every height layer of the product's geolocation grid (such as -500 to 9000 m),
            with no margin added, rounded out to the file's chunk grid (such as 512 x 512 pixels).
            Each side must be at least 20 km long on the ground (the south and north sides along their
            parallels): smaller crops cannot be processed accurately, so a smaller bbox raises a ValueError
            before anything is downloaded. The bbox is never enlarged. To process a smaller area, download a
            bbox of at least 20 x 20 km and pass the smaller bbox to the preprocessor, which can use the whole
            downloaded area for alignment.
            The bbox checks (this minimum size and the coordinate format) apply to every download that will
            happen, and to nothing else: with skip_exist=True, a granule whose files already exist is skipped
            without any check; every other granule is checked before any network call, and when the check fails
            nothing is downloaded and no folder is created. With skip_exist=False the check always runs.
            The cache proxy uses its cache-optimized multi-offset endpoint for the partial extraction.
            If None, downloads the full frame (the cache proxy uses aligned blocks for caching benefit).
        session : asf_search.ASFSession, optional
            Authenticated session. Created automatically if None.
        n_jobs : int or None, optional
            Parallel download jobs. None uses mission-specific defaults (S1: 8, NISAR: 2).
        joblib_backend : str, optional
            Backend for parallel processing. Default 'loky' (multiprocessing, faster on Colab).
        skip_exist : bool, optional
            skip_exist=True skips downloading files that already exist; skip_exist=False downloads them again.
            Default True. The files of a NISAR granule are those of the requested polarizations, or with
            polarization=None those the product has, read from the metadata of a file of the granule that exists
            (with none, every file is missing). Only the missing files are downloaded, and a granule whose files
            all exist is skipped without network access. The NISAR bbox checks apply to every download that will
            happen, and to nothing else.
        retries : int, optional
            Number of attempts of each request, the first one included; 0 makes one attempt, as 1 does
            (HTTP.attempts). A failure that a retry cannot change (HTTP.final, such as HTTP 404) is not retried.
            Default 30.
        timeout_second : int, optional
            Seconds between retries. Default 3.
        min_rate : str or float, optional
            Bytes per second every download request (a Sentinel-1 burst file, a NISAR byte range or cache block)
            must keep, a size string such as '100KB' or a number, averaged over min_rate_window seconds from its
            first byte, or it is retried on a new connection (HTTP.read_body); the rate every one of the n_jobs
            parallel downloads must reach. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default 60.
        debug : bool, optional
            Print debug information. Default False.

        Returns
        -------
        pandas.DataFrame or None
            The files downloaded by this call (the column 'burst' for Sentinel-1, 'file' for NISAR), or None when
            nothing is downloaded (every file exists).

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
        from .HTTP import attempts
        from .utils_S1 import polarizations

        # a negative retries raises before any file is checked or downloaded
        attempts(retries)

        # Normalize inputs
        import geopandas as gpd
        if isinstance(bursts, gpd.GeoDataFrame):
            bursts = bursts['sceneName'].tolist()
        elif isinstance(bursts, str):
            bursts = list(filter(None, map(str.strip, bursts.split('\n'))))

        # No output directory here: each writer creates its own subdirectory (which creates basedir) once every
        # check has passed, so a refused request leaves nothing behind

        # Group by mission (auto-detect from ID format)
        s1_bursts = []
        nisar_granules = []
        for burst in bursts:
            mission = self._detect_mission(burst)
            if mission == 'S1':
                s1_bursts.append(burst)
            elif mission == 'NISAR':
                nisar_granules.append(burst)

        # an empty polarization list selects nothing to download and raises, before any request or folder
        pols = polarizations(polarization, "'HH' or ['HH', 'HV']" if nisar_granules and not s1_bursts
                             else "'VV' or ['VV', 'VH']")

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
            # Filter NISAR granules that need download: a granule with a missing file; the downloaders then skip its
            # files that exist, so only the missing ones are downloaded
            nisar_needed = [g for g in nisar_granules if not self._nisar_complete(basedir, g, pols)]
        else:
            s1_needed = s1_bursts
            nisar_needed = nisar_granules

        # NISAR bbox crop: checked for the granules that will be downloaded, before any network call (session, S1
        # and both NISAR paths); a granule skipped as existing is not checked
        if nisar_needed and bbox is not None:
            self._nisar_check_bbox(bbox)

        # Return early if nothing to download (no network call!)
        if not s1_needed and not nisar_needed:
            return None

        # Prepare session only when actually needed
        if session is None:
            session = self._get_asf_session()

        results = []

        # Download S1 bursts
        if s1_needed:
            df = self._download_s1(basedir, s1_needed, pols, session,
                                   n_jobs, joblib_backend, skip_exist, retries, timeout_second,
                                   min_rate, min_rate_window, debug)
            if df is not None:
                results.append(df)

        # Download NISAR granules
        if nisar_needed:
            df = self._download_nisar(basedir, nisar_needed, pols, frequency, bbox, session,
                                       n_jobs, joblib_backend, skip_exist, retries, timeout_second,
                                       min_rate, min_rate_window, debug)
            if df is not None:
                results.append(df)

        if results:
            return pd.concat(results, ignore_index=True)
        return None

    def _download_s1(self, basedir, bursts, polarizations, session, n_jobs,
                      joblib_backend, skip_exist, retries, timeout_second, min_rate, min_rate_window, debug):
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
        from .utils_S1 import measurement_path, slc_shape, tiff_data_offset, write_slc, burst_xmls, PAIRS_TIFF_OFFSET
        from .utils_files import exists, EmptyFileError, write_file
        from .HTTP import final, attempts, send, read_body
        import xmltodict
        from xml.etree import ElementTree
        import pandas as pd
        import joblib
        from tqdm.auto import tqdm
        import os
        from datetime import datetime
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
            # the burst is stored as <burst>.nc; bursts downloaded before keep their <burst>.tiff
            nc_file = os.path.join(tif_dir, f'{burst}.nc')
            tif_file = measurement_path(tif_dir, burst)
            #print ('tif_file', tif_file)
            for dirname in [burst_dir, tif_dir, xml_annot_dir, xml_noise_dir, xml_calib_dir]:
                os.makedirs(dirname, exist_ok=True)

            def measurement_exists():
                if not exists(tif_file):
                    return False
                if tif_file.endswith('.tiff'):
                    return os.path.getsize(tif_file) >= int(properties['bytes'])
                return True

            # check if all files already exist; every one is checked before any download, so that an empty one raises
            xml_exist = [exists(filepath) for filepath in (xml_file, xml_noise_file, xml_calib_file)]
            all_exist = measurement_exists() and all(xml_exist)

            if all_exist:
                # validate existing measurement dimensions using local annotation XML
                with open(xml_file, 'r') as f:
                    local_annotation = xmltodict.parse(f.read())['product']
                lines_per_burst = int(local_annotation['swathTiming']['linesPerBurst'])
                samples_per_burst = int(local_annotation['imageAnnotation']['imageInformation']['numberOfSamples'])
                actual_lines, actual_samples = slc_shape(tif_file)
                if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                    raise Exception(f'ERROR: Existing measurement dimensions mismatch for {burst}: '
                                  f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                  f'Delete the corrupted file and re-download.')
                # all files valid, skip download
                return

            # download manifest to memory to get dimensions for TIFF validation.
            # A burst absent from the cache is extracted from the archived scene on demand,
            # and the server answers nothing at all while it does that, measured at 150s for
            # a single manifest. The read timeout has to outlast that silence, otherwise every
            # attempt aborts before the first byte and no number of retries ever succeeds.
            # Once the body flows, a transfer slower than min_rate is cut (HTTP.read_body) and retried.
            manifest_url = get_burst_url(properties['additionalUrls'][0])
            with send(session.get, manifest_url, stream=True, timeout=(10, 300)) as response:
                xml_content = read_body(response, min_rate, min_rate_window).decode(response.encoding or 'utf-8')
                cache_status = response.headers.get('x-cache', 'N/A')
            if debug:
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
            metadata = xmltodict.parse(xml_content)['burst']['metadata']
            content = metadata['product'][subswathidx]
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
            tiff_bytes = None
            if measurement_exists():
                # validate existing file dimensions
                actual_lines, actual_samples = slc_shape(tif_file)
                if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                    raise Exception(f'ERROR: Existing measurement dimensions mismatch for {burst}: '
                                  f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                  f'Delete the corrupted file and re-download.')
            else:
                # Download and validate TIFF entirely in memory before writing to disk
                import io

                # Download TIFF fully into memory; a transfer slower than min_rate is cut (HTTP.read_body) and retried
                tiff_url = get_burst_url(properties['url'])
                with send(session.get, tiff_url, stream=True, timeout=(10, 300)) as response:
                    tiff_bytes = read_body(response, min_rate, min_rate_window)
                    cache_status = response.headers.get('x-cache', 'N/A')
                    original_size = int(response.headers.get('X-Original-Size', 0))
                if debug:
                    size_mb = len(tiff_bytes) / 1024 / 1024
                    print(f'  TIFF {cache_status:4} {size_mb:5.1f}MB {burst}')
                if len(tiff_bytes) == 0:
                    raise Exception(f'ERROR: Downloaded TIFF is empty: {tiff_url}')

                # Early truncation check using expected size from server or ASF metadata
                expected_size = original_size or int(properties['bytes'])
                if expected_size > 0 and len(tiff_bytes) < expected_size:
                    pct = 100 * len(tiff_bytes) / expected_size
                    raise Exception(f'ERROR: Downloaded TIFF truncated for {burst}: '
                                  f'got {len(tiff_bytes)} bytes ({pct:.0f}%) of {expected_size} expected '
                                  f'(cache: {cache_status}). '
                                  f'The cache proxy may have timed out fetching from upstream.')

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

                # TIFF validated - now build XML content in memory before writing anything

            # Build XML content in memory (or skip if files exist)
            xml_contents = {}  # {filepath: content}

            if not all(xml_exist):
                # the XMLs of the burst, as every source writes them (utils_S1.burst_xmls); the byteOffset is that of
                # the measurement stored: the .nc written below, or the existing one
                byte_offset = PAIRS_TIFF_OFFSET if tiff_bytes is not None else tiff_data_offset(tif_file)
                parts = {}
                for kind in ('noise', 'calibration'):
                    content = metadata[kind][subswathidx]
                    assert polarization == content['polarisation'], 'ERROR: XML polarization differs from burst polarization'
                    parts[kind] = content['content']
                xml_contents[xml_file], xml_contents[xml_noise_file], xml_contents[xml_calib_file] = \
                    burst_xmls(annotation, parts['noise'], parts['calibration'], burstIndex, byte_offset)

            # All validations passed - write to temp files then atomic rename.
            # This guarantees no partial files on disk if interrupted mid-write.
            if tiff_bytes is not None:
                # the burst is stored as compressed NetCDF4, converted and verified in memory
                write_slc(tiff_bytes, nc_file)

            for filepath, content in xml_contents.items():
                write_file(filepath, content)

        # the catalog records of the bursts, rebuilt from a sibling polarization channel the catalog omits
        results = _asf_burst_records(bursts_missed,
                                     missing_note='NOTE: burst {burst} is missing in the ASF catalog and is not downloaded.')

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
            # retries=0 makes one attempt (HTTP.attempts); a failure that a retry cannot change (HTTP.final) ends the
            # attempts of the burst at once
            burst_id = result.geojson()['properties']['fileID']
            n = attempts(retries)
            for retry in range(n):
                try:
                    download_burst(result, basedir, session)
                    return True
                except EmptyFileError:
                    raise
                except Exception as e:
                    stop = final(e)
                    print(f'ERROR: download attempt {retry+1} failed{" (not retried)" if stop else ""} '
                          f'for {burst_id}: {e}')
                    if stop or retry + 1 == n:
                        return False
                time.sleep(timeout_second)

        # download bursts
        with self.progressbar_joblib(tqdm(desc='Downloading ASF SLC'.ljust(25), total=len(results))) as progress_bar:
            statuses = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_burst_with_retry)\
                                    (result, basedir, session, retries=retries, timeout_second=timeout_second) for result in results)

        failed_count = statuses.count(False)
        if failed_count > 0:
            raise Exception(f'Bursts downloading failed for {failed_count} items.')
        # parse processed bursts and convert to dataframe, reporting the bursts really
        # downloaded, which excludes any burst the catalog does not know about
        bursts_downloaded = pd.DataFrame([result.geojson()['properties']['fileID'] for result in results],
                                         columns=['burst'])
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
    def _nisar_check_bbox(bbox):
        """Validate a NISAR crop bbox (west, south, east, north) in WGS84 degrees before any download.

        Each side must be at least _NISAR_MIN_CROP_KM long on the WGS84 ellipsoid: the south and north sides
        along their parallels, the west and east sides along the meridian. The bbox is never enlarged.

        Raises
        ------
        ValueError
            For a malformed bbox, or a side shorter than the minimum.
        """
        import numpy as np
        from pyproj import Geod

        if len(bbox) != 4:
            raise ValueError("bbox must be (west, south, east, north) in WGS84 coordinates")
        west, south, east, north = bbox
        if west >= east or south >= north:
            raise ValueError("Invalid bbox: west must be < east and south must be < north")
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox coordinates must be valid WGS84 (lon: -180 to 180, lat: -90 to 90)")

        geod = Geod(ellps='WGS84')

        def parallel_km(lat):
            """Arc of the parallel at lat over the bbox longitudes."""
            phi = np.radians(lat)
            return geod.a * np.cos(phi) / np.sqrt(1 - geod.es * np.sin(phi) ** 2) * np.radians(east - west) / 1000

        sides = {'south': parallel_km(south), 'north': parallel_km(north),
                 'west and east': geod.inv(west, south, west, north)[2] / 1000}
        # compared at 1 mm
        short = [k for k, v in sides.items() if round(v * 1e6) < _NISAR_MIN_CROP_KM * 10**6]
        if short:
            sizes = ', '.join(f'{k} {v:.3f} km' for k, v in sides.items())
            raise ValueError(
                f"NISAR bbox {tuple(float(v) for v in bbox)} is too small to crop (too short: {', '.join(short)}). "
                f"The minimum is {_NISAR_MIN_CROP_KM} km per side on the ground; the requested sides are {sizes}. "
                f"Smaller crops cannot be processed accurately, and the bbox is not enlarged automatically. "
                f"Pass a bbox of at least {_NISAR_MIN_CROP_KM} x {_NISAR_MIN_CROP_KM} km here; a smaller "
                f"processing bbox can still be passed to the preprocessor, which can use the whole downloaded "
                f"area for alignment. Nothing was downloaded.")

    @staticmethod
    def _nisar_bbox_to_pixel_indices(h5, bbox, chunk_info_a, chunk_info_b=None):
        """Convert WGS84 bbox to pixel indices using geolocationGrid.

        The geolocation grid gives longitude and latitude on a regular (zeroDopplerTime, slantRange) node grid
        for every height layer (such as -500 to 9000 m). A lon/lat box is curved in radar coordinates, so
        its outline is sampled at a quarter of the smallest node spacing and mapped to fractional nodes on
        every layer, by inverting the bilinear interpolation of the grid (Newton
        iterations, extrapolated from the edge cells beyond the grid), then to swath lines and range bins. The
        window is the extreme extent over all layers, so it covers the bbox at any terrain height within the
        layers. No margin is added: the window ends at the pixels nearest to the extreme outline points. The
        callers round the window out to the file's chunk grid.

        Parameters
        ----------
        h5 : h5py.File
            Open HDF5 file with geolocationGrid
        bbox : tuple
            (west, south, east, north) in WGS84 degrees
        chunk_info_a : dict or None
            Chunk info for frequencyA (has shape), None when only frequencyB is requested
        chunk_info_b : dict, optional
            Chunk info for frequencyB

        Returns
        -------
        dict with keys:
            'az_start', 'az_end': azimuth pixel range
            'rg_start_a', 'rg_end_a': range pixel range for freqA (if provided)
            'rg_start_b', 'rg_end_b': range pixel range for freqB (if provided)

        Raises
        ------
        ValueError
            "does not intersect scene" when the bbox misses the swath of a requested band (only the grid nodes
            inside the swath count) or a window would be empty.
        """
        import numpy as np

        def interp(x, xp, fp):
            """Linear interpolation over increasing xp, extrapolated linearly beyond both ends."""
            y = np.interp(x, xp, fp)
            lo, hi = x < xp[0], x > xp[-1]
            y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
            y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return y

        def window(pixels, n):
            """Pixels nearest to the extreme fractional positions, clamped to [0, n)."""
            return max(0, int(np.floor(pixels.min() + 0.5))), min(n, int(np.floor(pixels.max() + 0.5)) + 1)

        # Bbox format: (west, south, east, north) = (lon_min, lat_min, lon_max, lat_max)
        west, south, east, north = bbox
        geo = h5['science/LSAR/RSLC/metadata/geolocationGrid']

        # All height layers, (n_height, n_az_geo, n_rg_geo)
        # NISAR EPSG 4326 convention: coordinateX = longitude, coordinateY = latitude
        lon = geo['coordinateX'][:]
        lat = geo['coordinateY'][:]
        n_layers, n_az_geo, n_rg_geo = lon.shape

        # The part of the bbox beyond the grid's own lon/lat extent is outside the scene
        w, e = max(west, lon.min()), min(east, lon.max())
        s, n = max(south, lat.min()), min(north, lat.max())
        if w >= e or s >= n:
            raise ValueError(f"Bbox {bbox} does not intersect scene")

        # Outline of the (clipped) bbox, sampled at a quarter of the smallest node spacing in degrees
        step = 0.25 * min(np.hypot(np.diff(lon, axis=axis), np.diff(lat, axis=axis)).min() for axis in (1, 2))
        xs = np.linspace(w, e, int(np.ceil((e - w) / step)) + 1)
        ys = np.linspace(s, n, int(np.ceil((n - s) / step)) + 1)
        px = np.concatenate([xs, np.full(ys.size, e), xs, np.full(ys.size, w)])
        py = np.concatenate([np.full(xs.size, s), ys, np.full(xs.size, n), ys])

        # Fractional nodes (u along azimuth, v along range) of the outline on every layer, (n_height, n_points).
        # Start: the affine map through each layer's first node and the two grid axes.
        k = np.arange(n_layers)[:, None]
        ox, oy = lon[:, 0, 0][:, None], lat[:, 0, 0][:, None]
        ux, uy = (lon[:, -1, 0][:, None] - ox) / (n_az_geo - 1), (lat[:, -1, 0][:, None] - oy) / (n_az_geo - 1)
        vx, vy = (lon[:, 0, -1][:, None] - ox) / (n_rg_geo - 1), (lat[:, 0, -1][:, None] - oy) / (n_rg_geo - 1)
        det = ux * vy - uy * vx
        u = ((px - ox) * vy - (py - oy) * vx) / det
        v = ((py - oy) * ux - (px - ox) * uy) / det

        def bilinear(g, i, j, a, b):
            """Value and its derivatives along u and v of grid g in cells (i, j) at cell fractions (a, b)."""
            g00, g10, g01, g11 = g[k, i, j], g[k, i + 1, j], g[k, i, j + 1], g[k, i + 1, j + 1]
            c = g11 - g10 - g01 + g00
            return g00 + a * (g10 - g00) + b * (g01 - g00) + a * b * c, (g10 - g00) + b * c, (g01 - g00) + a * c

        for _ in range(50):
            i = np.clip(np.floor(u).astype(np.int64), 0, n_az_geo - 2)
            j = np.clip(np.floor(v).astype(np.int64), 0, n_rg_geo - 2)
            fx, xu, xv = bilinear(lon, i, j, u - i, v - j)
            fy, yu, yv = bilinear(lat, i, j, u - i, v - j)
            det = xu * yv - xv * yu
            du = ((px - fx) * yv - (py - fy) * xv) / det
            dv = ((py - fy) * xu - (px - fx) * yu) / det
            u, v = u + du, v + dv
            if max(np.abs(du).max(), np.abs(dv).max()) < 1e-6:
                break
        else:
            raise ValueError(f"Bbox {bbox}: the geolocation grid inversion did not converge")

        # Fractional nodes -> zeroDopplerTime / slantRange -> fractional swath lines and range bins
        swaths = h5['science/LSAR/RSLC/swaths']
        full_az_time = swaths['zeroDopplerTime'][:]
        az_time = interp(u.ravel(), np.arange(n_az_geo), geo['zeroDopplerTime'][:])
        slant_range = interp(v.ravel(), np.arange(n_rg_geo), geo['slantRange'][:])
        lines = interp(az_time, full_az_time, np.arange(full_az_time.size))
        az_start, az_end = window(lines, (chunk_info_a or chunk_info_b)['shape'][0])

        # The grid reaches beyond the swath (such as 5 nodes in time and range), so only its nodes inside the
        # swath count: their lines and each band's bins
        node_lines = interp(geo['zeroDopplerTime'][:], full_az_time, np.arange(full_az_time.size))
        in_bbox = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)

        def in_swath(lines, bins, n_bins):
            """Fractional lines and bins inside the swath of a band with n_bins range bins."""
            return (lines > -0.5) & (lines < full_az_time.size - 0.5) & (bins > -0.5) & (bins < n_bins - 0.5)

        # Every requested band must intersect the bbox: an outline point inside the band's swath on some layer, or
        # a node inside the band's swath within the bbox (a bbox containing the whole scene); its window is not empty
        result = {'az_start': az_start, 'az_end': az_end}
        missed = []
        for band, chunk_info in (('A', chunk_info_a), ('B', chunk_info_b)):
            if not chunk_info:
                continue
            full_slant_range = swaths[f'frequency{band}/slantRange'][:]
            bins = interp(slant_range, full_slant_range, np.arange(full_slant_range.size))
            node_bins = interp(geo['slantRange'][:], full_slant_range, np.arange(full_slant_range.size))
            hit = (in_swath(lines, bins, full_slant_range.size).any() or
                   (in_bbox & in_swath(node_lines[:, None], node_bins[None, :], full_slant_range.size)).any())
            rg_start, rg_end = window(bins, chunk_info['shape'][1])
            if not hit or az_start >= az_end or rg_start >= rg_end:
                missed.append(band)
            result[f'rg_start_{band.lower()}'], result[f'rg_end_{band.lower()}'] = rg_start, rg_end
        if missed:
            requested = [band for band, ci in (('A', chunk_info_a), ('B', chunk_info_b)) if ci]
            raise ValueError(f"Bbox {bbox} does not intersect scene" +
                             ('' if missed == requested else f" in frequency{missed[0]}"))

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

    @staticmethod
    def _nisar_crop_times(metadata, az_start, az_end):
        """identification/zeroDopplerStartTime and zeroDopplerEndTime of a crop of the swath lines [az_start, az_end).

        They are the times of the crop's first and last lines: swaths/zeroDopplerTime on the epoch of its units
        ("seconds since 2025-12-24T00:00:00"), in the source's format (2025-12-24T01:13:36.042105263). Only a
        moved side is returned: a crop that starts at line 0 or ends at the frame's last line keeps the source's
        string there, byte for byte.

        Returns
        -------
        dict
            {dataset path: numpy.bytes_} of the times to replace (empty without a moved side); numpy.bytes_ as
            read from the source, so the dataset stays a fixed-length string.
        """
        import numpy as np
        from datetime import datetime, timedelta

        zdt = metadata['science/LSAR/RSLC/swaths/zeroDopplerTime']
        times = zdt['data']
        az_end = min(az_end, len(times)) if az_end else len(times)
        sides = {'zeroDopplerStartTime': az_start if az_start > 0 else None,
                 'zeroDopplerEndTime': az_end - 1 if az_end < len(times) else None}
        sides = {f'science/LSAR/identification/{k}': v for k, v in sides.items() if v is not None}
        sides = {k: v for k, v in sides.items() if k in metadata}
        if not sides:
            return {}
        units = zdt['attrs'].get('units', b'')
        units = units.decode() if isinstance(units, bytes) else str(units)
        if 'since' not in units:
            print(f"WARNING: swaths/zeroDopplerTime has no 'seconds since <UTC epoch>' units ({units!r}), so the "
                  f"crop keeps the frame's identification/zeroDopplerStartTime and zeroDopplerEndTime instead of "
                  f"its own first and last line times.")
            return {}
        epoch = datetime.fromisoformat(units.split('since', 1)[1].strip().replace(' ', 'T')[:19])
        out = {}
        for path, line in sides.items():
            old = metadata[path]['data']
            old = old.decode() if isinstance(old, bytes) else str(old)
            digits = len(old.split('.', 1)[1]) if '.' in old else 0
            ns = int(round(float(times[line]) * 1e9))
            text = (epoch + timedelta(seconds=ns // 10**9)).strftime('%Y-%m-%dT%H:%M:%S')
            if digits:
                text += '.' + f'{ns % 10**9:09d}'[:digits]
            out[path] = np.bytes_(text.encode())
        return out

    def _download_nisar(self, basedir, granules, polarizations, frequency, bbox, session,
                         n_jobs, joblib_backend, skip_exist, retries, timeout_second, min_rate, min_rate_window, debug):
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
        from .utils_files import EmptyFileError
        from .HTTP import final, attempts, send, read_body

        # Use cache proxy when no credentials provided
        if self.username is None:
            return self._download_nisar_via_cache(
                basedir, granules, polarizations, frequency, bbox,
                n_jobs, skip_exist, retries, timeout_second, min_rate, min_rate_window, debug
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

        def download_byte_range(url, start, end, auth_tuple, pbar=None, http_session=None, signed_url=None):
            """Download byte range using single HTTP Range request.

            If signed_url is provided, uses it directly (skipping OAuth).
            If pbar is provided, updates it instead of creating a new one.
            If http_session is provided, reuses the connection.

            Returns: (bytes, signed_url) - signed_url for reuse in subsequent requests
            """
            size = end - start
            headers = {'Range': f'bytes={start}-{end-1}'}  # HTTP Range is inclusive

            returned_signed_url = signed_url

            # a failed request raises with its response closed (HTTP.http_error), the response of any other is
            # closed by the with block
            if signed_url:
                # Use signed URL directly (much faster - no OAuth redirects); an error names the file URL
                r = send(requests.get, signed_url, what=url, headers=headers, stream=True, timeout=(10, 300))
            elif http_session:
                r = send(http_session.get, url, headers=headers, stream=True, timeout=(10, 300))
            else:
                r = send(requests.get, url, headers=headers, auth=auth_tuple, stream=True, timeout=(10, 300))

            with r:
                # Capture signed URL from redirect chain
                if http_session and not signed_url and 'cloudfront.net' in r.url:
                    returned_signed_url = r.url
                content_length = int(r.headers.get('Content-Length', 0))
                if content_length > 0 and content_length < size:
                    raise Exception(f'Truncated range response: expected {size} bytes, server reports {content_length}')
                # a transfer slower than min_rate is cut (HTTP.read_body) and retried
                result = read_body(r, min_rate, min_rate_window, progress=pbar.update if pbar else None)

            if len(result) < size:
                raise Exception(f'Truncated download: got {len(result)} bytes of {size} expected')
            return result, returned_signed_url

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

            # 3. download() passes only granules with a missing file (skip_exist); the files that exist are not
            # downloaded again (below, once the metadata names the polarizations of the product)

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
            head_resp = send(http_session.head, url, allow_redirects=True, timeout=(10, 30))
            file_size = head_resp.headers.get('Content-Length')
            if file_size is None:
                # a broken answer, retried
                raise IOError(f'{url}: no Content-Length in the answer')
            file_size = int(file_size)

            # Download 128MB metadata block (same as cache path) with progress bar
            with tqdm(total=128*1024*1024, unit='B', unit_scale=True,
                      desc=f"{short_name} metadata", leave=False,
                      dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0,
                      disable=(position is not None)) as meta_pbar:
                layout, metadata_buffer = self._detect_nisar_layout_fast(
                    url, auth_tuple, file_size, http_session=http_session,
                    pbar=meta_pbar if position is None else None,
                    min_rate=min_rate, min_rate_window=min_rate_window
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
            pols_to_download = ASF._nisar_pols(polarizations, available_pols, granule_id)
            # the files that exist are not downloaded again
            if skip_exist:
                pols_to_download = [p for p in pols_to_download if not ASF._nisar_exists(basedir, granule_id, p)]
                if not pols_to_download:
                    return []

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
                            h5_meta, bbox, first_ci_a, first_ci_b
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
            """Download single granule with retry logic: retries=0 makes one attempt (HTTP.attempts), and a failure
            that a retry cannot change (HTTP.final) raises at its first attempt."""
            n = attempts(retries)
            for retry in range(n):
                try:
                    return download_nisar_granule(
                        granule, basedir, polarizations, skip_exist, debug, position=position
                    )
                except (ValueError, EmptyFileError):
                    # Format errors and empty files are permanent - don't retry
                    raise
                except Exception as e:
                    stop = final(e)
                    print(f"ERROR downloading {granule} (attempt {retry+1}/{n}){' (not retried)' if stop else ''}: "
                          f"{e}")
                    if stop or retry + 1 == n:
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
                                   n_jobs, skip_exist, retries, timeout_second, min_rate, min_rate_window, debug):
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
        from .HTTP import final, attempts, send, read_body, MAGIC_HDF5

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

            # Calculate exact chunk indices for bbox (not aligned blocks); chunk_info_a is None for frequencyB only
            chunk_az = (chunk_info_a or chunk_info_b)['chunk_shape'][0]

            result['az_chunk_start'] = result['az_start'] // chunk_az
            result['az_chunk_end'] = (result['az_end'] + chunk_az - 1) // chunk_az
            if chunk_info_a:
                chunk_rg = chunk_info_a['chunk_shape'][1]
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

        def fetch_cache_range(granule_id, offset, length, session=None):
            """Fetch range from cache proxy using new API.

            Returns (content, cache_hit) tuple.
            """
            url = f"{_NISAR_CACHE_PROXY}/{granule_id}/{offset}/{length}.bin"
            sess = session or requests.Session()
            # a transfer slower than min_rate is cut (HTTP.read_body) and retried
            with send(sess.get, url, stream=True, timeout=120) as resp:
                content = read_body(resp, min_rate, min_rate_window)
                # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
                cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                            resp.headers.get('cf-cache-status', '').upper() == 'HIT')
            if len(content) != length:
                raise ValueError(f'Cache response size mismatch: got {len(content)}, expected {length}')
            return content, cache_hit

        def retried(what, request):
            """request() with the retries of the cache proxy: retries=0 makes one attempt (HTTP.attempts), and a
            failure that a retry cannot change (HTTP.final, such as HTTP 404) raises at its first attempt, logged as
            not retried; the last failure is logged too."""
            n = attempts(retries)
            for retry in range(n):
                try:
                    return request()
                except Exception as e:
                    stop = final(e)
                    if debug or stop or retry + 1 == n:
                        print(f'ERROR: {what} download attempt {retry+1}/{n} failed'
                              f'{" (not retried)" if stop else ""}: {e}')
                    if stop or retry + 1 == n:
                        raise
                    time.sleep(timeout_second)

        def fetch_metadata(granule_id):
            """The metadata block of a granule (fetch_cache_range); a body that is not HDF5 is retried, as a
            broken transfer."""
            def request():
                content, cache_hit = fetch_cache_range(granule_id, 0, MAX_BLOCK)
                if not content.startswith(MAGIC_HDF5):
                    raise IOError(f'Cache returned invalid metadata (not HDF5): {content[:100]!r}')
                return content, cache_hit
            return retried('metadata', request)

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

        # Process granules (the bbox is validated by download())
        granule_ids = [g.replace('.h5', '') for g in granules]

        print(f"Downloading {len(granule_ids)} NISAR granule(s) via cache proxy (aligned blocks)...")

        # Unified download path - only backend differs for debug/sequential mode
        import joblib
        backend = 'sequential' if (n_jobs == 1 or debug) else 'loky'
        effective_n_jobs = 1 if backend == 'sequential' else n_jobs

        def download_aligned_block(gid, offsets_str, total_size):
            """Download aligned 25x25km block via multi-offset API."""
            import requests
            url = f"{_NISAR_CACHE_PROXY}/{gid}/{offsets_str}/ranges.bin"
            def request():
                # a transfer slower than min_rate is cut (HTTP.read_body) and retried
                with send(requests.get, url, stream=True, timeout=120) as resp:
                    content = read_body(resp, min_rate, min_rate_window)
                    # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
                    cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                                resp.headers.get('cf-cache-status', '').upper() == 'HIT')
                if len(content) != total_size:
                    raise ValueError(
                        f"Response size mismatch: got {len(content)}, expected {total_size}")
                return (offsets_str, content, cache_hit)
            return retried('block', request)

        def download_single_chunk(gid, offset, length):
            """Download single chunk via single-block API (for small blocks)."""
            import requests
            url = f"{_NISAR_CACHE_PROXY}/{gid}/{offset}/{length}.bin"
            def request():
                # a transfer slower than min_rate is cut (HTTP.read_body) and retried
                with send(requests.get, url, stream=True, timeout=120) as resp:
                    return (offset, length, read_body(resp, min_rate, min_rate_window))
            return retried('single chunk', request)

        all_downloaded = []

        for gid in granule_ids:

            # 1. Fetch metadata for this granule
            if debug:
                print(f"Fetching metadata for {gid}...")
            meta_raw, _ = fetch_metadata(gid)
            meta_buf = patch_hdf5_superblock(meta_raw)
            if debug:
                print(f"  Metadata: {len(meta_buf)/(1024**2):.1f}MB")

            with h5py.File(BytesIO(bytes(meta_buf)), 'r') as h5:
                swaths = h5['science/LSAR/RSLC/swaths']
                avail_pols = [k for k in swaths['frequencyA'].keys() if k in ['HH','HV','VH','VV']]
                pols = ASF._nisar_pols(polarizations, avail_pols, gid)
                # the files that exist are not downloaded again
                if skip_exist:
                    pols = [p for p in pols if not ASF._nisar_exists(basedir, gid, p)]
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
                        bbox_info = bbox_to_block_indices(h5, bbox, first_ci_a, first_ci_b)
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
                    # Extract downloaded chunks — with bbox, only a subset is present
                    result = {}
                    empty = 0
                    for c in ci['chunks']:
                        if c['offset'] in block_data:
                            chunk_bytes = block_data[c['offset']]
                            if len(chunk_bytes) == 0:
                                empty += 1
                            result[c['offset']] = chunk_bytes
                    if empty > 0:
                        raise ValueError(
                            f"freq{freq_label} {pol}: {empty} empty chunks")
                    if len(result) == 0:
                        raise ValueError(
                            f"freq{freq_label} {pol}: no chunks downloaded "
                            f"(0 of {len(ci['chunks'])} in block_data)")
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

    def _detect_nisar_layout_fast(self, url, auth_tuple, file_size, http_session=None, pbar=None,
                                  min_rate='100KB', min_rate_window=60):
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
        from .HTTP import send, read_body

        METADATA_SIZE = 128 * 1024 * 1024  # 128 MB - matches cache path

        # Download first 128MB (reuse session if provided)
        headers = {'Range': f'bytes=0-{METADATA_SIZE-1}'}
        if http_session:
            response = send(http_session.get, url, headers=headers, stream=True, timeout=(10, 300))
        else:
            response = send(requests.get, url, headers=headers, auth=auth_tuple, stream=True, timeout=(10, 300))

        # Stream download with progress; a transfer slower than min_rate is cut (HTTP.read_body) and retried
        with response:
            data = bytearray(read_body(response, min_rate, min_rate_window, progress=pbar.update if pbar else None))

        # Validate HDF5 magic bytes
        if data[:4] != b'\x89HDF':
            try:
                import json
                error = json.loads(bytes(data).decode('utf-8', errors='replace'))
                msg = error.get('message', error.get('error', str(error)))
                raise Exception(f'Server returned error instead of HDF5: {msg}')
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise Exception(f'Server returned invalid response (not HDF5): {bytes(data[:100])!r}')

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

    def _read_nisar_metadata_direct(self, pol, metadata_buffer):
        """Read metadata from pre-downloaded buffer.

        Uses metadata_buffer (from layout A download) to read all metadata
        datasets - NO remote access needed.

        Parameters
        ----------
        pol : str
            Polarization being processed (e.g., 'HH').
        metadata_buffer : bytearray
            Pre-downloaded metadata buffer: the patched first 128 MB of the file (_detect_nisar_layout_fast, or
            the metadata block of the cache proxy).

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

        When crop_info is provided, crops SLC data and coordinate arrays to bbox extent, and sets
        identification/zeroDopplerStartTime and zeroDopplerEndTime to the crop's first and last lines.

        Parameters
        ----------
        downloaded_data : bytes, dict, or None
            Raw bytes from HTTP Range request covering frequencyA chunks,
            OR dict mapping chunk_offset -> chunk_bytes (for cache proxy).
        chunk_info : dict or None
            Chunk metadata from get_chunk_info() for frequencyA.
        metadata : dict
            ALL metadata from _read_nisar_metadata_direct().
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

            # 3. Write metadata datasets (filter by frequency if needed); a crop gets the identification start and
            # end times of its own first and last lines
            crop_times = self._nisar_crop_times(metadata, crop_az_start, crop_az_end) if crop_info else {}
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
                    # identification/zeroDopplerStartTime and zeroDopplerEndTime of the cropped lines
                    elif ds_path in crop_times:
                        data = crop_times[ds_path]

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

        # Validate in-memory HDF5 before writing to disk
        import h5py
        with h5py.File(BytesIO(mem_buffer.getvalue()), 'r') as h5_check:
            swaths = h5_check['science/LSAR/RSLC/swaths']
            for freq_key in ['frequencyA', 'frequencyB']:
                if freq_key in swaths and pol in swaths[freq_key]:
                    ds = swaths[freq_key][pol]
                    if ds.shape[0] == 0 or ds.shape[1] == 0:
                        raise ValueError(f'{freq_key}/{pol} has zero dimension: {ds.shape}')
                    break
            else:
                raise ValueError(f'No SLC dataset found for {pol}')

        # Write to temp file, then atomic rename
        from .utils_files import write_file
        write_file(out_path, mem_buffer.getvalue())

        if debug:
            file_size = os.path.getsize(out_path)
            print(f"    Done: {file_size / (1024**3):.2f} GB written")

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None,
               platform='SENTINEL-1', processingLevel='auto', polarization=None, beamMode='IW'):
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
        if geometry.geom_type == 'LineString' and geometry.coords[0] == geometry.coords[-1]:
            geometry = shapely.geometry.Polygon(geometry.coords)
        if geometry.geom_type == 'Polygon':
            # force counterclockwise orientation.
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        #print ('wkt', geometry.wkt)

        # one platform name, a comma-separated string or a list; the catalog takes several as a comma-separated string
        from .utils_S1 import S1_PLATFORMS, platform_names
        names = platform_names(platform) if platform else []
        # 'auto' is the burst level of every Sentinel-1 platform name
        if isinstance(processingLevel, str) and processingLevel=='auto' and names \
                and all(name.upper() in S1_PLATFORMS for name in names):
            processingLevel = asf_search.PRODUCT_TYPE.BURST

        # search bursts
        results = asf_search.search(
            start=startTime,
            end=stopTime,
            flightDirection=flightDirection,
            intersectsWith=geometry.wkt,
            platform=','.join(names) if names else platform,
            processingLevel=processingLevel,
            polarization=polarization,
            beamMode=beamMode,
        )
        gdf = gpd.GeoDataFrame.from_features([product.geojson() for product in results], crs="EPSG:4326")
        if 'burst' in gdf.columns:
            gdf['fullBurstID'] = gdf['burst'].apply(lambda b: b['fullBurstID'])
        return gdf

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
