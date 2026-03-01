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
import requests

# CDSE authentication constants
_CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/cdse/protocol/openid-connect/token"
_CDSE_CLIENT_ID = "cdse-public"
_CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"

# Cloudflare Worker cache proxy for CDSE bursts
_CDSE_CACHE_PROXY = 'https://s1-cache-cdse.insar.dev'


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
        response = self.post(
            _CDSE_TOKEN_URL,
            data={
                "client_id": _CDSE_CLIENT_ID,
                "username": username,
                "password": password,
                "grant_type": "password",
            },
        )
        response.raise_for_status()

        self._token = response.json()["access_token"]
        self.headers["Authorization"] = f"Bearer {self._token}"
        self._authenticated = True
        return self

    def refresh_token(self, username, password):
        """Refresh the access token."""
        return self.auth_with_creds(username, password)


def _cdse_search(start=None, end=None, flightDirection=None, intersectsWith=None,
                 polarization=None, swath=None, burstId=None, relativeOrbit=None,
                 top=1000):
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
    top : int, optional
        Maximum results (default 1000).

    Returns
    -------
    list
        List of burst metadata dictionaries.
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

    params = {
        '$top': top,
        '$orderby': 'ContentDate/Start desc',
    }
    if filters:
        params['$filter'] = ' and '.join(filters)

    response = requests.get(_CDSE_CATALOGUE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    return data.get('value', [])


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

    # Parse datetime from ContentDate
    content_date = cdse_burst.get('ContentDate', {})
    start_time = content_date.get('Start', '')
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

    return f"S1_{burst_id}_{swath}_{dt_str}_{polarization}_{scene_hash}-BURST"


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

    # Extract relative orbit for path number
    rel_orbit = cdse_burst.get('RelativeOrbitNumber', 0)

    properties = {
        'fileID': file_id,
        'url': f"{_CDSE_CATALOGUE_URL}({cdse_burst['Id']})/$value",
        'additionalUrls': [],  # CDSE returns zip with all files
        'bytes': cdse_burst.get('ByteOffset', 0),
        'startTime': content_date.get('Start', ''),
        'stopTime': content_date.get('End', ''),
        'flightDirection': cdse_burst.get('OrbitDirection', 'ASCENDING'),
        'pathNumber': rel_orbit,
        'polarization': cdse_burst.get('PolarisationChannels', 'VV'),
        'platform': cdse_burst.get('PlatformSerialIdentifier', 'S1A'),
        'processingLevel': 'BURST',
        'beamModeType': cdse_burst.get('OperationalMode', 'IW'),
        'burst': {
            'fullBurstID': f"{rel_orbit:03d}_{cdse_burst.get('BurstId')}_{cdse_burst.get('SwathIdentifier', 'IW1')}",
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
               platform='SENTINEL-1', polarization='VV', beamMode='IW'):
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
        platform : str, optional
            Platform name (default 'SENTINEL-1').
        polarization : str, optional
            Polarization (default 'VV').
        beamMode : str, optional
            Beam mode (default 'IW').

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
        )

        # Convert to GeoJSON features
        features = [_cdse_to_geojson_feature(r) for r in results]

        return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

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

        params = {
            '$top': 1000,
            '$filter': ' and '.join(filters),
        }

        response = requests.get(_CDSE_CATALOGUE_URL, params=params)
        response.raise_for_status()
        cdse_results = response.json().get('value', [])

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
                 debug=False):
        """Download Sentinel-1 bursts from CDSE.

        Parameters
        ----------
        basedir : str
            Output directory.
        bursts : str or list
            Burst identifiers (ASF format).
        polarization : str or list, optional
            Polarization(s) to download.
        session : requests.Session, optional
            Authenticated session.
        n_jobs : int, optional
            Parallel download jobs (default 8).
        joblib_backend : str, optional
            Joblib backend (default 'loky').
        skip_exist : bool, optional
            Skip already downloaded (default True).
        retries : int, optional
            Retry attempts (default 30).
        timeout_second : int, optional
            Seconds between retries (default 3).
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
        warnings.filterwarnings("ignore", category=UserWarning)

        # Normalize bursts to list
        if isinstance(bursts, str):
            bursts = [b.strip() for b in bursts.strip().split('\n') if b.strip()]

        # Filter S1 bursts only
        bursts = [b for b in bursts if b.startswith('S1_') and b.endswith('-BURST')]

        if not bursts:
            print("No valid S1 bursts to download")
            return None

        # Normalize and expand bursts by polarization if specified
        polarizations = self._normalize_polarization(polarization)
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
            from datetime import datetime, timedelta
            from tifffile import TiffFile
            import rasterio
            from rasterio.io import MemoryFile

            def filter_azimuth_time(items, start_utc_dt, stop_utc_dt, delta=3):
                if not isinstance(items, list):
                    items = [items]
                return [item for item in items if
                    datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') >= start_utc_dt - timedelta(seconds=delta) and
                    datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') <= stop_utc_dt + timedelta(seconds=delta)]

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
            xml_file = os.path.join(xml_annot_dir, f'{burst}.xml')
            xml_noise_file = os.path.join(xml_noise_dir, f'{burst}.xml')
            xml_calib_file = os.path.join(xml_calib_dir, f'{burst}.xml')
            tif_file = os.path.join(tif_dir, f'{burst}.tiff')

            for dirname in [burst_dir, tif_dir, xml_annot_dir, xml_noise_dir, xml_calib_dir]:
                os.makedirs(dirname, exist_ok=True)

            # Check if all files already exist and validate
            all_exist = (os.path.exists(tif_file) and os.path.getsize(tif_file) > 0
                        and os.path.exists(xml_file) and os.path.getsize(xml_file) > 0
                        and os.path.exists(xml_noise_file) and os.path.getsize(xml_noise_file) > 0
                        and os.path.exists(xml_calib_file) and os.path.getsize(xml_calib_file) > 0)

            if all_exist:
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
                return True

            # Download burst zip
            url = get_burst_url(cdse_uuid)

            if use_post and auth_token:
                # Direct CDSE download with manual redirect handling
                # (auth header is lost when following redirect to different domain)
                headers = {'Authorization': f'Bearer {auth_token}'}

                # Initial request
                response = requests.post(url, headers=headers, allow_redirects=False, timeout=30)

                # Handle redirect (CDSE redirects to bursts.dataspace.copernicus.eu)
                if 300 <= response.status_code < 400:
                    redirect_url = response.headers.get('Location')
                    if redirect_url:
                        # Follow redirect with auth header, allow further redirects
                        response = requests.post(redirect_url, headers=headers,
                                                timeout=(10, 300), allow_redirects=True)
            else:
                # Cache proxy - simple GET
                response = session.get(url, timeout=(10, 300))

            if response.status_code != 200:
                try:
                    error_body = response.text[:500] if response.text else 'No response body'
                except:
                    error_body = 'Could not read response body'
                raise Exception(f"CDSE {response.status_code} {response.reason}: {error_body}")

            # Get cache status for debug output
            cache_status = response.headers.get('cf-cache-status', response.headers.get('x-cache', 'N/A'))
            cache_enc = response.headers.get('content-encoding', 'none')

            zip_bytes = response.content
            if len(zip_bytes) == 0:
                raise Exception(f'ERROR: Downloaded ZIP is empty for {burst}')

            if debug:
                zip_mb = len(zip_bytes) / 1024 / 1024
                via = 'proxy' if _CDSE_CACHE_PROXY in url else 'direct'
                print(f'  {cache_status:4} {via:6} {zip_mb:5.1f}MB {burst}')

            # Extract zip to memory first for validation
            tiff_bytes = None
            annotation_xml = None
            noise_xml = None
            calibration_xml = None

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
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

            if not tiff_bytes:
                raise Exception(f'ERROR: No TIFF found in ZIP for {burst}')
            if not annotation_xml:
                raise Exception(f'ERROR: No annotation XML found in ZIP for {burst}')

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
                tiff_offset = page.dataoffsets[0]
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
            azimuth_time_interval = annotation['imageAnnotation']['imageInformation']['azimuthTimeInterval']
            burst_list = annotation['swathTiming']['burstList']['burst']
            if not isinstance(burst_list, list):
                burst_list = [burst_list]

            # CDSE returns single burst - burstIndex is always 0
            assert len(burst_list) == 1, f'Expected 1 burst, got {len(burst_list)}'
            burstIndex = 0
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
            burst_time_interval = timedelta(seconds=(lines_per_burst - 1) * float(azimuth_time_interval))
            stop_utc_dt = start_utc_dt + burst_time_interval
            stop_utc = stop_utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

            # Build filtered annotation XML (matching ASF.py structure)
            xml_contents = {}

            # Build product annotation
            product = {}

            adsHeader = annotation['adsHeader']
            adsHeader['startTime'] = start_utc
            adsHeader['stopTime'] = stop_utc
            adsHeader['imageNumber'] = '001'
            product['adsHeader'] = adsHeader

            if 'qualityInformation' in annotation:
                qualityInformation = {}
                if 'productQualityIndex' in annotation['qualityInformation']:
                    qualityInformation['productQualityIndex'] = annotation['qualityInformation']['productQualityIndex']
                if 'qualityDataList' in annotation['qualityInformation']:
                    qualityInformation['qualityDataList'] = annotation['qualityInformation']['qualityDataList']
                product['qualityInformation'] = qualityInformation

            if 'generalAnnotation' in annotation:
                product['generalAnnotation'] = annotation['generalAnnotation']

            imageAnnotation = annotation['imageAnnotation']
            imageAnnotation['imageInformation']['productFirstLineUtcTime'] = start_utc
            imageAnnotation['imageInformation']['productLastLineUtcTime'] = stop_utc
            imageAnnotation['imageInformation']['productComposition'] = 'Assembled'
            imageAnnotation['imageInformation']['sliceNumber'] = '0'
            imageAnnotation['imageInformation']['sliceList'] = {'@count': '0'}
            imageAnnotation['imageInformation']['numberOfLines'] = str(lines_per_burst)
            product['imageAnnotation'] = imageAnnotation

            if 'dopplerCentroid' in annotation:
                dopplerCentroid = annotation['dopplerCentroid']
                items = filter_azimuth_time(dopplerCentroid['dcEstimateList']['dcEstimate'], start_utc_dt, stop_utc_dt)
                dopplerCentroid['dcEstimateList'] = {'@count': len(items), 'dcEstimate': items}
                product['dopplerCentroid'] = dopplerCentroid

            if 'antennaPattern' in annotation:
                antennaPattern = annotation['antennaPattern']
                items = filter_azimuth_time(antennaPattern['antennaPatternList']['antennaPattern'], start_utc_dt, stop_utc_dt)
                antennaPattern['antennaPatternList'] = {'@count': len(items), 'antennaPattern': items}
                product['antennaPattern'] = antennaPattern

            swathTiming = annotation['swathTiming']
            items = filter_azimuth_time(swathTiming['burstList']['burst'], start_utc_dt, start_utc_dt, 1)
            assert len(items) == 1, 'ERROR: unexpected bursts count, should be 1'
            items[0]['byteOffset'] = tiff_offset  # Add TIFF offset
            swathTiming['burstList'] = {'@count': len(items), 'burst': items}
            product['swathTiming'] = swathTiming

            geolocationGrid = annotation['geolocationGrid']
            geoloc_points = geolocationGrid['geolocationGridPointList']['geolocationGridPoint']
            if not isinstance(geoloc_points, list):
                geoloc_points = [geoloc_points]
            items = filter_azimuth_time(geoloc_points, start_utc_dt, stop_utc_dt, 1)
            # Re-numerate line numbers for the burst (burstIndex=0, so no change needed)
            for item in items:
                item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
            geolocationGrid['geolocationGridPointList'] = {'@count': len(items), 'geolocationGridPoint': items}
            product['geolocationGrid'] = geolocationGrid

            if 'coordinateConversion' in annotation:
                product['coordinateConversion'] = annotation['coordinateConversion']
            if 'swathMerging' in annotation:
                product['swathMerging'] = annotation['swathMerging']

            xml_contents[xml_file] = xmltodict.unparse({'product': product}, pretty=True, indent='  ')

            # Build filtered noise XML
            if noise_xml:
                noise_annot = xmltodict.parse(noise_xml)['noise']
                noise = {}

                if 'adsHeader' in noise_annot:
                    noise_adsHeader = noise_annot['adsHeader']
                    noise_adsHeader['startTime'] = start_utc
                    noise_adsHeader['stopTime'] = stop_utc
                    noise_adsHeader['imageNumber'] = '001'
                    noise['adsHeader'] = noise_adsHeader

                if 'noiseVectorList' in noise_annot:
                    noiseRangeVector = noise_annot['noiseVectorList']
                    nv_items = noiseRangeVector.get('noiseVector', [])
                    if not isinstance(nv_items, list):
                        nv_items = [nv_items]
                    items = filter_azimuth_time(nv_items, start_utc_dt, stop_utc_dt)
                    for item in items:
                        item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                    noise['noiseVectorList'] = {'@count': len(items), 'noiseVector': items}

                if 'noiseRangeVectorList' in noise_annot:
                    noiseRangeVector = noise_annot['noiseRangeVectorList']
                    nrv_items = noiseRangeVector.get('noiseRangeVector', [])
                    if not isinstance(nrv_items, list):
                        nrv_items = [nrv_items]
                    items = filter_azimuth_time(nrv_items, start_utc_dt, stop_utc_dt)
                    for item in items:
                        item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                    noise['noiseRangeVectorList'] = {'@count': len(items), 'noiseRangeVector': items}

                if 'noiseAzimuthVectorList' in noise_annot:
                    noiseAzimuthVector = noise_annot['noiseAzimuthVectorList']
                    nav = noiseAzimuthVector.get('noiseAzimuthVector', {})
                    if nav and 'line' in nav:
                        line_data = nav['line']
                        if isinstance(line_data, dict) and '#text' in line_data:
                            line_items = [int(x) for x in line_data['#text'].split()]
                        elif isinstance(line_data, str):
                            line_items = [int(x) for x in line_data.split()]
                        else:
                            line_items = []

                        if line_items:
                            # Find lines within burst range
                            lowers = [item for item in line_items if item <= burstIndex * lines_per_burst] or [line_items[0]]
                            uppers = [item for item in line_items if item >= (burstIndex + 1) * lines_per_burst - 1] or [line_items[-1]]
                            mask = [lowers[-1] <= item <= uppers[0] for item in line_items]
                            filtered_lines = [item - burstIndex * lines_per_burst for item, m in zip(line_items, mask) if m]

                            nav['firstAzimuthLine'] = str(lowers[-1] - burstIndex * lines_per_burst)
                            nav['lastAzimuthLine'] = str(uppers[0] - burstIndex * lines_per_burst)
                            nav['line'] = {'@count': str(len(filtered_lines)), '#text': ' '.join(str(x) for x in filtered_lines)}

                            # Filter noiseAzimuthLut similarly
                            if 'noiseAzimuthLut' in nav:
                                lut_data = nav['noiseAzimuthLut']
                                if isinstance(lut_data, dict) and '#text' in lut_data:
                                    lut_items = lut_data['#text'].split()
                                elif isinstance(lut_data, str):
                                    lut_items = lut_data.split()
                                else:
                                    lut_items = []
                                filtered_lut = [item for item, m in zip(lut_items, mask) if m]
                                nav['noiseAzimuthLut'] = {'@count': str(len(filtered_lut)), '#text': ' '.join(filtered_lut)}

                            noise['noiseAzimuthVectorList'] = {'noiseAzimuthVector': nav}

                xml_contents[xml_noise_file] = xmltodict.unparse({'noise': noise}, pretty=True, indent='  ')

            # Build filtered calibration XML
            if calibration_xml:
                calib_annot = xmltodict.parse(calibration_xml)['calibration']
                calibration = {}

                if 'adsHeader' in calib_annot:
                    calib_adsHeader = calib_annot['adsHeader']
                    calib_adsHeader['startTime'] = start_utc
                    calib_adsHeader['stopTime'] = stop_utc
                    calib_adsHeader['imageNumber'] = '001'
                    calibration['adsHeader'] = calib_adsHeader

                if 'calibrationInformation' in calib_annot:
                    calibration['calibrationInformation'] = calib_annot['calibrationInformation']

                if 'calibrationVectorList' in calib_annot:
                    calibrationVector = calib_annot['calibrationVectorList']
                    cv_items = calibrationVector.get('calibrationVector', [])
                    if not isinstance(cv_items, list):
                        cv_items = [cv_items]
                    items = filter_azimuth_time(cv_items, start_utc_dt, stop_utc_dt)
                    for item in items:
                        item['line'] = str(int(item['line']) - (lines_per_burst * burstIndex))
                    calibration['calibrationVectorList'] = {'@count': len(items), 'calibrationVector': items}

                xml_contents[xml_calib_file] = xmltodict.unparse({'calibration': calibration}, pretty=True, indent='  ')

            # All validations passed - write to temp files then atomic rename.
            # This guarantees no partial files on disk if interrupted mid-write.
            tmp = tif_file + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(tiff_bytes)
            os.rename(tmp, tif_file)

            for filepath, content in xml_contents.items():
                tmp = filepath + '.tmp'
                with open(tmp, 'w') as f:
                    f.write(content)
                os.rename(tmp, filepath)

            return cache_status  # Return cache status (HIT/MISS/etc)

        def download_burst_with_retry(result, retries, timeout_second):
            burst_id = result.geojson()['properties']['fileID']
            for retry in range(retries):
                try:
                    return download_burst(result)  # Returns cache_status or True (for existing)
                except Exception as e:
                    print(f'ERROR: download attempt {retry+1} failed for {burst_id}: {e}')
                    if retry + 1 == retries:
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
        """Check if burst is completely downloaded."""
        import os
        from glob import glob

        # Parse burst ID: S1_043813_IW1_20230210T033452_VV_E5B0-BURST
        parts = burst.split('_')
        burst_num = str(int(parts[1]))  # Remove leading zeros: 043813 -> 43813
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

        for subdir in ['measurement', 'annotation', 'calibration', 'noise']:
            subdir_path = os.path.join(burst_dir, subdir)
            if not os.path.isdir(subdir_path):
                return False
            ext = '.tiff' if subdir == 'measurement' else '.xml'
            matches = glob(file_pattern + ext, root_dir=subdir_path)
            if not matches:
                return False
            filepath = os.path.join(subdir_path, matches[0])
            if os.path.getsize(filepath) == 0:
                return False

        return True

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
