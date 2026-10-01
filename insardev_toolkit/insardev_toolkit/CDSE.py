# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
Copernicus Data Space Ecosystem (CDSE) Sentinel-1 Burst Access Module.

Provides search and download capabilities for Sentinel-1 SLC bursts from the
Copernicus Data Space Ecosystem, with API compatible with the ASF module.
"""
from .progressbar_joblib import progressbar_joblib
from .utils_S1 import path_number, S1_PLATFORMS, platform_names
from .HTTP import send
import requests

# CDSE authentication constants
_CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/cdse/protocol/openid-connect/token"
_CDSE_CLIENT_ID = "cdse-public"
_CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
# bursts per catalogue request, the server rejects anything above this
_CDSE_PAGE_SIZE = 1000

# Cloudflare Worker cache proxy for CDSE bursts
_CDSE_CACHE_PROXY = 'https://s1-cache-cdse.insar.dev'


def _cdse_pages(params):
    """Follow the @odata.nextLink pages of one catalogue query: the bursts read and the catalogue count of its matches."""
    values, url, first, count = [], _CDSE_CATALOGUE_URL, True, None
    while url is not None:
        # the next link carries its own query, so pass params for the first request only
        response = send(requests.get, url, params=dict(params, **{'$count': 'true'}) if first else None,
                        timeout=(30, 300))
        data = response.json()
        if first:
            count, first = data.get('@odata.count'), False
        values += data.get('value', [])
        url = data.get('@odata.nextLink')
    return values, count


def _cdse_query(params):
    """Query the CDSE catalogue and return all matching bursts.

    The catalogue never returns more than _CDSE_PAGE_SIZE bursts per request and reports the
    rest through an @odata.nextLink, so a query matching more bursts than that is silently
    truncated unless the link is followed. The link stops at the catalogue's $skip limit of 10000
    too, so a query ordered by 'ContentDate/Start desc' goes on from the start time of the last
    burst read until one query is read whole; any other query matching more raises an error, and
    so does a result with fewer distinct bursts than the first query matches.

    Parameters
    ----------
    params : dict
        OData parameters, e.g. {'$filter': '...', '$top': 1000}.

    Returns
    -------
    list
        List of CDSE burst records.
    """
    values, ids, query, total = [], set(), params, None
    while True:
        more, count = _cdse_pages(query)
        # the bursts the first query matches, which the continuations have to read in all
        total = count if total is None else total
        new = [v for v in more if v['Id'] not in ids]
        values += new
        ids.update(v['Id'] for v in new)
        if count is None:
            print(f'WARNING: the CDSE catalogue gave no @odata.count, so {len(values)} bursts are returned '
                  f'without knowing whether the query matches more')
            return values
        if len(more) >= count:
            if len(ids) < total:
                raise ValueError(f'ERROR: the CDSE catalogue returned {len(ids)} distinct bursts of the {total} '
                                 f'matching the query: narrow the query.')
            return values
        if params.get('$orderby') != 'ContentDate/Start desc' or not new:
            raise ValueError(f'ERROR: the CDSE catalogue returned {len(more)} of the {count} bursts matching the '
                             f'query and pages no further: narrow the query.')
        # the bursts at and before the last start time read, which are read again and dropped
        after = f"ContentDate/Start le {more[-1]['ContentDate']['Start']}"
        query = dict(params, **{'$filter': f"({params['$filter']}) and {after}" if params.get('$filter') else after})


class _CDSESession(requests.Session):
    """Authenticated session for CDSE downloads.

    Handles OAuth2 authentication to Copernicus Data Space Ecosystem.
    """

    def __init__(self):
        super().__init__()
        self._authenticated = False
        self._token = None

    def auth_with_creds(self, username, password):
        """Authenticate with CDSE credentials.

        Parameters
        ----------
        username : str
            CDSE username (email).
        password : str
            CDSE password.

        Returns
        -------
        _CDSESession
            Self, for method chaining.
        """
        # (connect, read) timeouts, as the other toolkit requests (HTTP.fetch, the Earthdata token POST of ASF)
        response = send(
            self.post,
            _CDSE_TOKEN_URL,
            data={
                "client_id": _CDSE_CLIENT_ID,
                "username": username,
                "password": password,
                "grant_type": "password",
            },
            timeout=(10, 300),
        )

        self._token = response.json()["access_token"]
        self.headers["Authorization"] = f"Bearer {self._token}"
        self._authenticated = True
        return self

    def refresh_token(self, username, password):
        """Refresh the access token."""
        return self.auth_with_creds(username, password)


def _cdse_search(start=None, end=None, flightDirection=None, intersectsWith=None,
                 polarization=None, swath=None, burstId=None, relativeOrbit=None,
                 beamMode=None, platform=None):
    """Search CDSE catalog for Sentinel-1 bursts.

    Parameters
    ----------
    start : str, optional
        Start datetime (ISO format).
    end : str, optional
        End datetime (ISO format).
    flightDirection : str, optional
        'ASCENDING' or 'DESCENDING'.
    intersectsWith : str, optional
        WKT geometry string.
    polarization : str, optional
        'VV', 'VH', 'HH', 'HV'.
    swath : str, optional
        'IW1', 'IW2', 'IW3'.
    burstId : int, optional
        Burst ID number.
    relativeOrbit : int, optional
        Relative orbit number.
    beamMode : str, optional
        'IW' or 'EW', the OperationalMode of the burst.
    platform : str or list, optional
        'SENTINEL-1' for every satellite, or 'SENTINEL-1A' .. 'SENTINEL-1D' for one, as ASF names them. A list or a
        comma-separated string names several.

    Returns
    -------
    list
        List of burst metadata dictionaries, every match.
    """
    filters = []

    if start:
        # Ensure proper ISO format
        if len(start) == 10:
            start = f"{start}T00:00:00.000Z"
        elif not start.endswith('Z'):
            start = start.replace(' ', 'T') + '.000Z'
        filters.append(f"ContentDate/Start ge {start}")

    if end:
        if len(end) == 10:
            end = f"{end}T23:59:59.999Z"
        elif not end.endswith('Z'):
            end = end.replace(' ', 'T') + '.999Z'
        filters.append(f"ContentDate/Start le {end}")

    if flightDirection:
        if flightDirection.upper() in ('A', 'ASC', 'ASCENDING'):
            flightDirection = 'ASCENDING'
        elif flightDirection.upper() in ('D', 'DESC', 'DESCENDING'):
            flightDirection = 'DESCENDING'
        filters.append(f"OrbitDirection eq '{flightDirection}'")

    if intersectsWith:
        filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{intersectsWith}')")

    if polarization:
        filters.append(f"PolarisationChannels eq '{polarization.upper()}'")

    if swath:
        filters.append(f"SwathIdentifier eq '{swath.upper()}'")

    if burstId is not None:
        filters.append(f"BurstId eq {burstId}")

    if relativeOrbit is not None:
        filters.append(f"RelativeOrbitNumber eq {relativeOrbit}")

    if beamMode:
        filters.append(f"OperationalMode eq '{beamMode.upper()}'")

    if platform:
        # the catalog names a satellite by its serial letter, 'SENTINEL-1C' -> 'C'; 'SENTINEL-1' is every one of them
        names = [name.upper() for name in platform_names(platform)]
        for name in names:
            if name not in S1_PLATFORMS:
                raise ValueError(f"ERROR: unknown platform {name!r}, expected 'SENTINEL-1' or 'SENTINEL-1A' .. "
                                 f"'SENTINEL-1D'")
        if 'SENTINEL-1' not in names:
            filters.append('(' + ' or '.join(f"PlatformSerialIdentifier eq '{name[-1]}'" for name in names) + ')')

    params = {
        '$top': _CDSE_PAGE_SIZE,
        '$orderby': 'ContentDate/Start desc',
    }
    if filters:
        params['$filter'] = ' and '.join(filters)

    # the catalog returns the matches a page at a time, so every page is read
    return _cdse_query(params)


def _parse_asf_burst_id(burst_id):
    """Parse ASF-format burst ID to components.

    Parameters
    ----------
    burst_id : str
        ASF burst ID like 'S1_262887_IW2_20190702T032458_VV_69C5-BURST'.

    Returns
    -------
    dict
        Dictionary with burstId, swath, datetime, polarization, sceneHash.
    """
    if not burst_id.startswith('S1_') or not burst_id.endswith('-BURST'):
        raise ValueError(f"Invalid ASF burst ID format: {burst_id}")

    # S1_262887_IW2_20190702T032458_VV_69C5-BURST
    parts = burst_id[:-6].split('_')  # Remove '-BURST' suffix
    if len(parts) != 6:
        raise ValueError(f"Invalid ASF burst ID format: {burst_id}")

    return {
        'burstId': int(parts[1]),
        'swath': parts[2],
        'datetime': parts[3],
        'polarization': parts[4],
        'sceneHash': parts[5],
    }


def _make_asf_burst_id(cdse_burst):
    """Convert CDSE burst metadata to ASF-format burst ID.

    Parameters
    ----------
    cdse_burst : dict
        CDSE burst metadata from search results.

    Returns
    -------
    str
        ASF-format burst ID.
    """
    # Extract components from CDSE burst
    burst_id = cdse_burst.get('BurstId')
    swath = cdse_burst.get('SwathIdentifier', 'IW1')
    polarization = cdse_burst.get('PolarisationChannels', 'VV')

    # Name the burst after its azimuth time, which is the value the annotation carries in
    # adsHeader/startTime and the value ASF names a burst after. ContentDate/Start is the
    # sensing start, 1-2 seconds later, and naming a burst after it stores it under a name
    # ASF cannot recognize, so the same burst is downloaded twice into one directory.
    content_date = cdse_burst.get('ContentDate', {})
    start_time = cdse_burst.get('AzimuthTime') or content_date.get('Start', '')
    # Convert 2019-07-02T03:24:58.123Z to 20190702T032458
    dt_str = start_time[:19].replace('-', '').replace(':', '')

    # Generate scene hash from parent product or ID
    parent_name = cdse_burst.get('ParentProductName', '')
    if parent_name:
        # Extract hash from parent product name (last 4 chars before extension)
        scene_hash = parent_name.split('_')[-1][:4].upper()
    else:
        # Use last 4 chars of UUID
        uuid = cdse_burst.get('Id', '0000')
        scene_hash = uuid[-4:].upper()

    # the burst ID is zero-padded to 6 digits, as ASF names it
    return f"S1_{int(burst_id):06d}_{swath}_{dt_str}_{polarization}_{scene_hash}-BURST"


def _cdse_to_geojson_feature(cdse_burst):
    """Convert CDSE burst metadata to GeoJSON feature (ASF-compatible format).

    Parameters
    ----------
    cdse_burst : dict
        CDSE burst metadata from search results.

    Returns
    -------
    dict
        GeoJSON feature with properties matching ASF format.
    """
    content_date = cdse_burst.get('ContentDate', {})
    geo_footprint = cdse_burst.get('GeoFootprint', {})

    # Build ASF-compatible burst ID
    file_id = _make_asf_burst_id(cdse_burst)

    # the relative orbit of the product, which the ASF catalog reports as pathNumber too
    rel_orbit = cdse_burst.get('RelativeOrbitNumber', 0)
    beam_mode = cdse_burst.get('OperationalMode', 'IW')
    # the path of the burst directory: path_number() gives it from an IW burst ID, and a burst of another mode
    # keeps the relative orbit of the product
    path = path_number(cdse_burst['BurstId']) if beam_mode == 'IW' else rel_orbit

    # CDSE returns single-letter platform ('A','B','C','D'); ASF uses 'SENTINEL-1A'/'SENTINEL-1C'/...
    platform_letter = cdse_burst.get('PlatformSerialIdentifier', 'A')
    platform = f'SENTINEL-1{platform_letter}'

    properties = {
        'fileID': file_id,
        'url': f"{_CDSE_CATALOGUE_URL}({cdse_burst['Id']})/$value",
        'additionalUrls': [],  # CDSE returns zip with all files
        'bytes': 0,  # CDSE OData has no per-burst size; ByteOffset is offset within parent SLC, not size
        'startTime': content_date.get('Start', ''),
        'stopTime': content_date.get('End', ''),
        'flightDirection': cdse_burst.get('OrbitDirection', 'ASCENDING'),
        'pathNumber': rel_orbit,
        'polarization': cdse_burst.get('PolarisationChannels', 'VV'),
        'platform': platform,
        'processingLevel': 'BURST',
        'beamModeType': beam_mode,
        'burst': {
            # the burst directory, named as ASF names it: the path, which for an IW burst differs from the relative
            # orbit of a product that starts before the ascending node, and the zero-padded burst ID
            'fullBurstID': f"{path:03d}_{int(cdse_burst['BurstId']):06d}_{cdse_burst.get('SwathIdentifier', 'IW1')}",
            'burstIndex': cdse_burst.get('BurstId', 0),
            'subswath': cdse_burst.get('SwathIdentifier', 'IW1'),
            'absoluteBurstID': cdse_burst.get('AbsoluteBurstId', 0),
            'relativeBurstID': cdse_burst.get('BurstId', 0),
            'id': cdse_burst.get('Id'),  # CDSE UUID for download
        },
    }

    return {
        'type': 'Feature',
        'geometry': geo_footprint,
        'properties': properties,
    }


class _CDSESearchResult:
    """Minimal CDSE search result wrapper (ASF-compatible interface)."""

    def __init__(self, geojson_feature):
        self._geojson = geojson_feature

    def geojson(self):
        return self._geojson


class CDSE(progressbar_joblib):
    """Copernicus Data Space Ecosystem Sentinel-1 Burst Downloader.

    Drop-in replacement for ASF module, using CDSE as data source.
    Supports same burst ID format as ASF for compatibility.

    Parameters
    ----------
    username : str, optional
        CDSE username (email). If not provided, uses cache proxy.
    password : str, optional
        CDSE password. If not provided, uses cache proxy.

    Examples
    --------
    >>> cdse = CDSE()  # Use cache proxy
    >>> bursts = CDSE.search(aoi, startTime='2024-01-01', stopTime='2024-01-31')
    >>> cdse.download('data/', bursts.fileID.tolist())

    >>> cdse = CDSE('user@email.com', 'password')  # Direct CDSE access
    >>> cdse.download('data/', ['S1_262887_IW2_20190702T032458_VV_69C5-BURST'])
    """
    import pandas as pd
    from datetime import timedelta

    def __init__(self, username=None, password=None):
        """Initialize CDSE downloader.

        Parameters
        ----------
        username : str, optional
            CDSE username (email). If not provided, uses cache proxy.
        password : str, optional
            CDSE password. If not provided, uses cache proxy.
        """
        self.username = username
        self.password = password
        self._session = None
        self._token_time = None
        if username is None:
            print("NOTE: Using insar.dev Cache API. Free for non-commercial use; license required for funded academic, institutional, or professional use.")

    def _get_session(self):
        """Get authenticated session for CDSE downloads."""
        import time

        if self.username is None:
            # Cache proxy handles auth
            return requests.Session()

        # Check if we need to refresh token (tokens expire after ~10 minutes)
        if self._session is None or self._token_time is None or \
           (time.time() - self._token_time) > 540:  # 9 minutes
            self._session = _CDSESession().auth_with_creds(self.username, self.password)
            self._token_time = time.time()

        return self._session

    def _get_burst_url(self, burst_uuid):
        """Get download URL for burst, using cache proxy if no credentials."""
        if self.username is None:
            return f"{_CDSE_CACHE_PROXY}/{burst_uuid}"
        return f"{_CDSE_CATALOGUE_URL}({burst_uuid})/$value"

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None,
               platform='SENTINEL-1', polarization=None, beamMode='IW'):
        """Search for Sentinel-1 bursts in CDSE catalog.

        Parameters
        ----------
        geometry : GeoDataFrame, GeoSeries, or shapely geometry
            Area of interest.
        startTime : str
            Start date (YYYY-MM-DD or ISO format).
        stopTime : str
            Stop date (YYYY-MM-DD or ISO format).
        flightDirection : str, optional
            'A'/'ASCENDING' or 'D'/'DESCENDING'.
        platform : str or list, optional
            'SENTINEL-1' (default) for every satellite, or 'SENTINEL-1A' .. 'SENTINEL-1D' for one, as in ASF.search. A
            list or a comma-separated string names several.
        polarization : str, optional
            Polarization (default 'VV').
        beamMode : str, optional
            Beam mode, 'IW' (default) or 'EW'. None searches every mode.

        Returns
        -------
        GeoDataFrame
            Search results with ASF-compatible schema.
        """
        import geopandas as gpd
        import shapely

        # Normalize time format
        if startTime and len(startTime) == 10:
            startTime = f'{startTime} 00:00:01'
        if stopTime and len(stopTime) == 10:
            stopTime = f'{stopTime} 23:59:59'

        # Normalize flight direction
        if flightDirection == 'D':
            flightDirection = 'DESCENDING'
        elif flightDirection == 'A':
            flightDirection = 'ASCENDING'

        # Convert geometry
        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geometry = geometry.geometry.union_all()
        if geometry.geom_type == 'LineString' and geometry.coords[0] == geometry.coords[-1]:
            geometry = shapely.geometry.Polygon(geometry.coords)
        if geometry.geom_type == 'Polygon':
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)

        # Search CDSE
        results = _cdse_search(
            start=startTime,
            end=stopTime,
            flightDirection=flightDirection,
            intersectsWith=geometry.wkt,
            polarization=polarization,
            beamMode=beamMode,
            platform=platform,
        )

        # Convert to GeoJSON features
        features = [_cdse_to_geojson_feature(r) for r in results]

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        if 'burst' in gdf.columns:
            gdf['fullBurstID'] = gdf['burst'].apply(lambda b: b['fullBurstID'])
        return gdf

    @staticmethod
    def search_by_burst_id(burst_ids):
        """Search CDSE catalog by ASF-format burst IDs.

        Parameters
        ----------
        burst_ids : str or list
            ASF-format burst ID(s).

        Returns
        -------
        list
            List of _CDSESearchResult objects.
        """
        if isinstance(burst_ids, str):
            burst_ids = [burst_ids]

        # Parse all burst IDs
        parsed_bursts = {}
        for burst_id in burst_ids:
            parsed = _parse_asf_burst_id(burst_id)
            # Key: (burstId, swath, date, polarization)
            dt = parsed['datetime']
            date_str = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
            key = (parsed['burstId'], parsed['swath'], date_str, parsed['polarization'])
            parsed_bursts[key] = burst_id

        if not parsed_bursts:
            return []

        # Build single batched query
        burst_nums = list(set(k[0] for k in parsed_bursts.keys()))
        swaths = list(set(k[1] for k in parsed_bursts.keys()))
        dates = list(set(k[2] for k in parsed_bursts.keys()))
        polarizations = list(set(k[3] for k in parsed_bursts.keys()))

        min_date = min(dates)
        max_date = max(dates)

        # OData filter with all constraints
        filters = []
        filters.append('(' + ' or '.join(f'BurstId eq {b}' for b in burst_nums) + ')')
        filters.append('(' + ' or '.join(f"SwathIdentifier eq '{s}'" for s in swaths) + ')')
        filters.append(f"ContentDate/Start ge {min_date}T00:00:00.000Z")
        filters.append(f"ContentDate/Start le {max_date}T23:59:59.999Z")
        if len(polarizations) == 1:
            filters.append(f"PolarisationChannels eq '{polarizations[0]}'")

        # ordered by start time, so that a query matching more bursts than the catalogue pages goes on from the last
        params = {
            '$top': _CDSE_PAGE_SIZE,
            '$orderby': 'ContentDate/Start desc',
            '$filter': ' and '.join(filters),
        }

        cdse_results = _cdse_query(params)

        # Filter to exact matches (swath, date, polarization)
        results = []
        for cdse_burst in cdse_results:
            burst_id = cdse_burst.get('BurstId')
            swath = cdse_burst.get('SwathIdentifier')
            pol = cdse_burst.get('PolarisationChannels')
            content_date = cdse_burst.get('ContentDate', {}).get('Start', '')[:10]

            key = (burst_id, swath, content_date, pol)
            if key in parsed_bursts:
                feature = _cdse_to_geojson_feature(cdse_burst)
                results.append(_CDSESearchResult(feature))

        return results

    def download(self, basedir, bursts, polarization=None, session=None, n_jobs=4,
                 joblib_backend='loky', skip_exist=True, retries=30, timeout_second=3,
                 min_rate='100KB', min_rate_window=60, debug=False):
        """Download Sentinel-1 bursts from CDSE.

        Parameters
        ----------
        basedir : str
            Output directory.
        bursts : str or list
            Burst identifiers (ASF format).
        polarization : str or list, optional
            Polarization(s) to download. An empty list raises a ValueError.
        session : requests.Session, optional
            Authenticated session.
        n_jobs : int, optional
            Parallel download jobs (default 8).
        joblib_backend : str, optional
            Joblib backend (default 'loky').
        skip_exist : bool, optional
            Skip already downloaded (default True).
        retries : int, optional
            Attempts of each burst, the first one included; 0 makes one attempt, as 1 does (HTTP.attempts). A
            failure that a retry cannot change (HTTP.final, such as HTTP 404) is not retried. Default 30.
        timeout_second : int, optional
            Seconds between retries (default 3).
        min_rate : str or float, optional
            Bytes per second a burst download must keep, a size string such as '100KB' or a number, averaged over
            min_rate_window seconds from its first byte, or it is retried on a new connection (HTTP.read_body); the
            rate every one of the n_jobs parallel downloads must reach. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default 60.
        debug : bool, optional
            Print debug info (default False).

        Returns
        -------
        DataFrame or None
            Downloaded bursts info.
        """
        import os
        import zipfile
        import io
        import pandas as pd
        import joblib
        from tqdm.auto import tqdm
        import time
        import warnings
        from .utils_files import EmptyFileError
        from .utils_S1 import polarizations
        from .HTTP import final, attempts
        warnings.filterwarnings("ignore", category=UserWarning)

        # a negative retries and an empty polarization list raise before any file is checked or downloaded
        attempts(retries)
        pols = polarizations(polarization)

        # Normalize bursts to list
        if isinstance(bursts, str):
            bursts = [b.strip() for b in bursts.strip().split('\n') if b.strip()]

        # Filter S1 bursts only
        bursts = [b for b in bursts if b.startswith('S1_') and b.endswith('-BURST')]

        if not bursts:
            print("No valid S1 bursts to download")
            return None

        # Expand bursts by polarization if specified
        if pols is not None:
            expanded_bursts = []
            for burst in bursts:
                # S1_262885_IW2_20190702T032452_VV_69C5-BURST
                #                              ^^ pol at position 4
                parts = burst.split('_')
                for pol in pols:
                    new_parts = parts.copy()
                    new_parts[4] = pol  # Replace polarization
                    new_burst = '_'.join(new_parts)
                    expanded_bursts.append(new_burst)
            # Remove duplicates while preserving order
            seen = set()
            bursts = [b for b in expanded_bursts if not (b in seen or seen.add(b))]

        # Create output directory
        os.makedirs(basedir, exist_ok=True)

        # Check which bursts need downloading
        if skip_exist:
            bursts_missed = [b for b in bursts if not self._burst_exists(basedir, b)]
            if debug and len(bursts) != len(bursts_missed):
                print(f"Skipping {len(bursts) - len(bursts_missed)} already downloaded bursts")
        else:
            bursts_missed = bursts

        if len(bursts_missed) == 0:
            print(f"All {len(bursts)} bursts already downloaded")
            return None

        # Search for burst UUIDs
        print(f"Searching CDSE catalog for {len(bursts_missed)} bursts...")
        results = self.search_by_burst_id(bursts_missed)

        if len(results) != len(bursts_missed):
            print(f"Warning: Found {len(results)} of {len(bursts_missed)} bursts in CDSE")

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

        session = session or self._get_session()
        use_post = self.username is not None  # CDSE requires POST for authenticated downloads
        get_burst_url = self._get_burst_url  # Capture method for closure
        # Extract token for pickling (session headers don't survive loky serialization)
        auth_token = getattr(session, '_token', None) if use_post else None

        def download_burst(result):
            """Download and extract single burst with full ASF-compatible XML filtering."""
            import requests
            import xmltodict
            from datetime import datetime
            from tifffile import TiffFile
            import rasterio
            from rasterio.io import MemoryFile
            from .utils_S1 import measurement_path, slc_shape, write_slc, burst_xmls, PAIRS_TIFF_OFFSET
            from .utils_files import exists, write_file
            from .HTTP import send, http_error, NotFound, read_body

            props = result.geojson()['properties']
            burst = props['fileID']
            cdse_uuid = props['burst']['id']
            burst_info = props['burst']
            burstId = burst_info['fullBurstID']
            polarization = props['polarization']

            # Create directories
            burst_dir = os.path.join(basedir, burstId)
            tif_dir = os.path.join(burst_dir, 'measurement')
            xml_annot_dir = os.path.join(burst_dir, 'annotation')
            xml_noise_dir = os.path.join(burst_dir, 'noise')
            xml_calib_dir = os.path.join(burst_dir, 'calibration')

            def burst_files(burst):
                # the annotation, noise and calibration XMLs; the burst is stored as <burst>.nc
                return (os.path.join(xml_annot_dir, f'{burst}.xml'), os.path.join(xml_noise_dir, f'{burst}.xml'),
                        os.path.join(xml_calib_dir, f'{burst}.xml'), os.path.join(tif_dir, f'{burst}.nc'))

            xml_file, xml_noise_file, xml_calib_file, nc_file = burst_files(burst)
            # bursts downloaded before keep their <burst>.tiff
            tif_file = measurement_path(tif_dir, burst)

            for dirname in [burst_dir, tif_dir, xml_annot_dir, xml_noise_dir, xml_calib_dir]:
                os.makedirs(dirname, exist_ok=True)

            # Check if all files already exist and validate; every one is checked before any download, so that an
            # empty one raises
            all_exist = all([exists(filepath) for filepath in (tif_file, xml_file, xml_noise_file, xml_calib_file)])

            if all_exist:
                with open(xml_file, 'r') as f:
                    local_annotation = xmltodict.parse(f.read())['product']
                lines_per_burst = int(local_annotation['swathTiming']['linesPerBurst'])
                samples_per_burst = int(local_annotation['imageAnnotation']['imageInformation']['numberOfSamples'])
                actual_lines, actual_samples = slc_shape(tif_file)
                if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                    raise Exception(f'ERROR: Existing measurement dimensions mismatch for {burst}: '
                                  f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                                  f'Delete the corrupted file and re-download.')
                return True

            # Download burst zip
            url = get_burst_url(cdse_uuid)

            if use_post and auth_token:
                # Direct CDSE download with manual redirect handling
                # (auth header is lost when following redirect to different domain)
                headers = {'Authorization': f'Bearer {auth_token}'}

                # Initial request
                response = send(requests.post, url, headers=headers, allow_redirects=False, timeout=30,
                                stream=True, check=False)

                # Handle redirect (CDSE redirects to bursts.dataspace.copernicus.eu)
                if 300 <= response.status_code < 400:
                    redirect_url = response.headers.get('Location')
                    if redirect_url:
                        response.close()
                        # Follow redirect with auth header, allow further redirects
                        response = send(requests.post, redirect_url, what=url, headers=headers,
                                        timeout=(10, 300), allow_redirects=True, stream=True, check=False)
            else:
                # Cache proxy - simple GET
                response = send(session.get, url, timeout=(10, 300), stream=True, check=False)

            with response:
                if response.status_code != 200:
                    # an HTTPError with its response, so that HTTP.final judges the status
                    raise http_error(response, url)

                # Get cache status for debug output
                cache_status = response.headers.get('cf-cache-status', response.headers.get('x-cache', 'N/A'))
                cache_enc = response.headers.get('content-encoding', 'none')

                # a transfer slower than min_rate is cut (HTTP.read_body) and retried on a new connection
                zip_bytes = read_body(response, min_rate, min_rate_window)
            if len(zip_bytes) == 0:
                raise Exception(f'ERROR: Downloaded ZIP is empty for {burst}')

            if debug:
                zip_mb = len(zip_bytes) / 1024 / 1024
                via = 'proxy' if _CDSE_CACHE_PROXY in url else 'direct'
                print(f'  {cache_status:4} {via:6} {zip_mb:5.1f}MB {burst}')

            # Validate ZIP magic bytes
            if zip_bytes[:2] != b'PK':
                # Not an archive - likely JSON error from server
                try:
                    import json
                    error_json = json.loads(zip_bytes.decode('utf-8', errors='replace'))
                    error_msg = error_json.get('message', error_json.get('detail',
                                               error_json.get('error', str(error_json))))
                    raise Exception(f'ERROR: CDSE server returned error instead of ZIP for {burst}: {error_msg}')
                except json.JSONDecodeError:
                    raise Exception(f'ERROR: Invalid ZIP magic bytes for {burst}: {zip_bytes[:4]!r}')

            # The archive is streamed with chunked encoding and without a content length, so an
            # upstream failure ends the stream early and the short body arrives without any error.
            # Such a partial archive has no end of central directory record and zipfile reports
            # that as the misleading 'File is not a zip file'.
            try:
                archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
            except zipfile.BadZipFile as e:
                raise Exception(f'ERROR: Downloaded ZIP incomplete for {burst}: {e}. '
                              f'Got {len(zip_bytes)} bytes (cache: {cache_status}), the cache proxy '
                              f'stream ended before the archive was complete.')

            # Extract zip to memory first for validation
            tiff_bytes = None
            annotation_xml = None
            noise_xml = None
            calibration_xml = None

            with archive as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)

                    if filename.endswith('.tiff') or filename.endswith('.tif'):
                        with zf.open(member) as src:
                            tiff_bytes = src.read()

                    elif '/annotation/' in member and filename.endswith('.xml') and '/calibration/' not in member and '/rfi/' not in member:
                        with zf.open(member) as src:
                            annotation_xml = src.read().decode('utf-8')

                    elif filename.startswith('noise-') and filename.endswith('.xml'):
                        with zf.open(member) as src:
                            noise_xml = src.read().decode('utf-8')

                    elif filename.startswith('calibration-') and filename.endswith('.xml'):
                        with zf.open(member) as src:
                            calibration_xml = src.read().decode('utf-8')

            # the archive is not needed anymore, and the conversion below must not hold it in memory
            del zip_bytes, archive

            # a file missing from an archive that opens is final (HTTP.final), a retry delivers the same archive
            if not tiff_bytes:
                raise NotFound(f'No TIFF in the CDSE archive of {burst}')
            if not annotation_xml:
                raise NotFound(f'No annotation XML in the CDSE archive of {burst}')

            # Validate TIFF magic bytes
            if tiff_bytes[:2] not in (b'II', b'MM'):
                # Not a TIFF - likely JSON error from server
                try:
                    import json
                    error_json = json.loads(tiff_bytes.decode('utf-8', errors='replace'))
                    error_msg = error_json.get('message', error_json.get('error', str(error_json)))
                    raise Exception(f'ERROR: CDSE server returned error instead of TIFF for {burst}: {error_msg}')
                except json.JSONDecodeError:
                    raise Exception(f'ERROR: Invalid TIFF magic bytes for {burst}: {tiff_bytes[:4]!r}')

            # Parse annotation to get expected dimensions
            annotation = xmltodict.parse(annotation_xml)['product']
            lines_per_burst = int(annotation['swathTiming']['linesPerBurst'])
            samples_per_burst = int(annotation['imageAnnotation']['imageInformation']['numberOfSamples'])

            # Validate TIFF dimensions with TiffFile
            with TiffFile(io.BytesIO(tiff_bytes)) as tif:
                page = tif.pages[0]
                actual_lines, actual_samples = page.shape
            if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
                raise Exception(f'ERROR: Downloaded TIFF dimensions mismatch for {burst}: '
                              f'got {actual_lines}x{actual_samples}, expected {lines_per_burst}x{samples_per_burst}. '
                              f'CDSE burst extraction may have failed.')

            # Validate TIFF can be read by rasterio/GDAL (detects corruption)
            with MemoryFile(tiff_bytes) as memfile:
                with memfile.open() as ds:
                    if ds.width != samples_per_burst or ds.height != lines_per_burst:
                        raise Exception(f'ERROR: Rasterio dimensions mismatch for {burst}: '
                                      f'got {ds.height}x{ds.width}, expected {lines_per_burst}x{samples_per_burst}')
                    # Read a small portion to verify data is accessible
                    _ = ds.read(1, window=rasterio.windows.Window(0, 0, min(100, ds.width), min(100, ds.height)))

            # Get burst timing info
            burst_list = annotation['swathTiming']['burstList']['burst']
            if not isinstance(burst_list, list):
                burst_list = [burst_list]

            # CDSE returns single burst - burstIndex is always 0
            assert len(burst_list) == 1, f'Expected 1 burst, got {len(burst_list)}'
            burst_data = burst_list[0]
            start_utc = burst_data['azimuthTime']
            start_utc_dt = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')

            # Validate startTime matches burst name date (detect manifest mix-up)
            burst_date_str = burst.split('_')[3]  # e.g., '20210211T135237'
            expected_date = datetime.strptime(burst_date_str, '%Y%m%dT%H%M%S').date()
            if start_utc_dt.date() != expected_date:
                raise Exception(f'ERROR: Manifest data mismatch for burst {burst}: '
                              f'parsed startTime {start_utc_dt.date()} does not match expected date {expected_date}. '
                              f'This indicates corrupted manifest data.')
            # the files are named after the burst azimuth time of the annotation, as ASF names them, while the
            # catalog AzimuthTime gave the name the burst was looked up and checked for above
            parts = burst.split('_')
            parts[3] = start_utc_dt.strftime('%Y%m%dT%H%M%S')
            burst = '_'.join(parts)
            xml_file, xml_noise_file, xml_calib_file, nc_file = burst_files(burst)
            # the XMLs of the burst, as every source writes them (utils_S1.burst_xmls), for the .nc written below.
            # A CDSE burst product counts the lines of its XMLs from the first line of the burst, except the noise
            # azimuth vector, which keeps the lines of the whole subswath. In the subswath the vector and the
            # geolocation grid both start at line 0, so the burst starts at vector line (first vector line - first
            # grid line).
            # a file missing from the archive is final (HTTP.final), a retry delivers the same archive
            if not noise_xml:
                raise NotFound(f'No noise XML in the CDSE archive of {burst}')
            if not calibration_xml:
                raise NotFound(f'No calibration XML in the CDSE archive of {burst}')
            noise = xmltodict.parse(noise_xml)['noise']
            calibration = xmltodict.parse(calibration_xml)['calibration']
            vector = noise.get('noiseAzimuthVectorList', {}).get('noiseAzimuthVector')
            noise_azimuth_line = None
            if isinstance(vector, dict):
                points = annotation['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']
                points = points if isinstance(points, list) else [points]
                noise_azimuth_line = int(vector['firstAzimuthLine']) - min(int(point['line']) for point in points)
            xml_contents = dict(zip((xml_file, xml_noise_file, xml_calib_file),
                                    burst_xmls(annotation, noise, calibration, 0, PAIRS_TIFF_OFFSET,
                                               noise_azimuth_line)))

            # All validations passed - write to temp files then atomic rename.
            # This guarantees no partial files on disk if interrupted mid-write.
            # The burst is stored as compressed NetCDF4, converted and verified in memory, the same
            # file the ASF download writes.
            write_slc(tiff_bytes, nc_file)

            for filepath, content in xml_contents.items():
                write_file(filepath, content)

            return cache_status  # Return cache status (HIT/MISS/etc)

        def download_burst_with_retry(result, retries, timeout_second):
            # retries=0 makes one attempt (HTTP.attempts); a failure that a retry cannot change (HTTP.final) ends the
            # attempts of the burst at once
            burst_id = result.geojson()['properties']['fileID']
            n = attempts(retries)
            for retry in range(n):
                try:
                    return download_burst(result)  # Returns cache_status or True (for existing)
                except EmptyFileError:
                    raise
                except Exception as e:
                    stop = final(e)
                    print(f'ERROR: download attempt {retry+1} failed{" (not retried)" if stop else ""} '
                          f'for {burst_id}: {e}')
                    if stop or retry + 1 == n:
                        return False
                time.sleep(timeout_second)

        if n_jobs is None or debug:
            print('Note: sequential processing applied when n_jobs is None or debug is True.')
            # Simple loop with tqdm for sequential/debug mode
            statuses = []
            hits, misses = 0, 0
            pbar = tqdm(results, desc='Downloading CDSE SLC'.ljust(25))
            for result in pbar:
                status = download_burst_with_retry(result, retries, timeout_second)
                statuses.append(status)
                # Update HIT/MISS counts
                if status == 'HIT':
                    hits += 1
                elif status in ('MISS', 'EXPIRED', 'STALE'):
                    misses += 1
                if hits + misses > 0:
                    pbar.set_postfix_str(f'HIT:{hits} MISS:{misses}')
        else:
            # Parallel download with joblib
            with self.progressbar_joblib(tqdm(desc='Downloading CDSE SLC'.ljust(25), total=len(results))) as progress_bar:
                statuses = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                    joblib.delayed(download_burst_with_retry)(result, retries, timeout_second)
                    for result in results
                )

        failed_count = sum(1 for s in statuses if s is False)
        if failed_count > 0:
            raise Exception(f'Bursts downloading failed for {failed_count} items.')

        return pd.DataFrame(bursts_missed, columns=['burst'])

    @staticmethod
    def _burst_exists(basedir, burst):
        """Check if burst is completely downloaded. An empty file of the burst raises."""
        import os
        from glob import glob
        from .utils_files import exists

        # Parse burst ID: S1_043813_IW1_20230210T033452_VV_E5B0-BURST
        parts = burst.split('_')
        burst_num = f'{int(parts[1]):06d}'  # zero-padded as ASF and CDSE name it: 43813 -> 043813
        swath = parts[2]                # IW1
        datetime_full = parts[3]        # 20230210T033452
        datetime_prefix = datetime_full[:13]  # 20230210T0334 (ignore seconds)
        pol = parts[4]                  # VV

        # Find directory matching *_burstnum_swath
        dir_pattern = f'*_{burst_num}_{swath}'
        matching_dirs = glob(dir_pattern, root_dir=basedir)
        if not matching_dirs:
            return False

        burst_dir = os.path.join(basedir, matching_dirs[0])

        # Find files matching S1_burstnum_swath_datetime*_pol_* (flexible on seconds)
        file_pattern = f'S1_{burst_num}_{swath}_{datetime_prefix}*_{pol}_*'

        present = True
        for subdir in ['measurement', 'annotation', 'calibration', 'noise']:
            subdir_path = os.path.join(burst_dir, subdir)
            # the measurement is <burst>.nc, or the legacy <burst>.tiff
            exts = ('.nc', '.tiff') if subdir == 'measurement' else ('.xml',)
            matches = [m for ext in exts for m in glob(file_pattern + ext, root_dir=subdir_path)] \
                if os.path.isdir(subdir_path) else []
            # every matching file of every subdirectory is checked, so that an empty one raises
            for m in matches:
                exists(os.path.join(subdir_path, m))
            present = present and bool(matches)

        return present

    @staticmethod
    def plot(bursts, ax=None, figsize=None):
        """Plot burst footprints on map."""
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt

        bursts['date'] = pd.to_datetime(bursts['startTime']).dt.strftime('%Y-%m-%d')
        bursts['label'] = bursts.apply(
            lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} {rec['date']} [{rec['pathNumber']}]",
            axis=1
        )
        unique_labels = sorted(bursts['label'].unique())
        colors = {label[-4:-1]: 'orange' if label[0] == 'A' else 'cyan' for label in unique_labels}

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for label, group in bursts.groupby('label'):
            group.plot(ax=ax, edgecolor=colors[label[-4:-1]], facecolor='none', linewidth=1, alpha=1, label=label)

        burst_handles = [matplotlib.lines.Line2D([0], [0], color=colors[label[-4:-1]], lw=1, label=label)
                        for label in unique_labels]
        aoi_handle = matplotlib.lines.Line2D([0], [0], color='red', lw=1, label='AOI')
        handles = burst_handles + [aoi_handle]
        ax.legend(handles=handles, loc='upper right')
        ax.set_title('Sentinel-1 Burst Footprints (CDSE)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
