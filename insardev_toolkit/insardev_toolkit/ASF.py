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
import re

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


# the query of a URL or of a request path in a text: a signed URL's query is its signature
_URL_QUERY = re.compile(r'''(/[^\s?'"]*)\?[^\s'")]*[^\s'"):]''')


def _redact(text):
    """
    A text with the query of every URL and request path in it replaced by '?<query removed>' (scheme, host and path
    kept): the text of a failed request to a signed URL (requests names the URL it could not reach, its query the
    signature that opens the file for about an hour) is printed or raised without it.

    Parameters
    ----------
    text : str
        The text, such as the message of a requests error.

    Returns
    -------
    str
        The text without the queries.

    Examples
    --------
    >>> _redact("Max retries exceeded with url: /data/file.zip?Expires=1&Signature=abc (Caused by ...)")
    'Max retries exceeded with url: /data/file.zip?<query removed> (Caused by ...)'
    """
    return _URL_QUERY.sub(r'\1?<query removed>', text)


def _redacted(error):
    """
    The error with the queries removed from its text (_redact), raised by the download code in its place: when its
    text holds a query, its arguments become the redacted text and its chained errors (which hold the URL too) are
    dropped; the error's type is kept. An error whose text holds no query is returned unchanged.

    Parameters
    ----------
    error : BaseException
        The error of a failed request.

    Returns
    -------
    BaseException
        The same error object.

    Examples
    --------
    >>> raise _redacted(e)
    """
    text = str(error)
    clean = _redact(text)
    if clean != text:
        error.args = (clean,)
        error.__cause__ = None
        error.__suppress_context__ = True
    return error


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


def _asf_granule_search(granule_list, retries=30, timeout_second=3):
    """Search ASF by granule names (burst IDs or product names).

    The list is split into chunks so that a single request stays within the
    catalog limits regardless of how many granules are requested.

    Parameters
    ----------
    granule_list : list
        List of granule identifiers.
    retries : int, optional
        Attempts of each catalog request, passed to _asf_query. Default 30.
    timeout_second : int, optional
        Seconds between the attempts, passed to _asf_query. Default 3.

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
        data = _asf_query({'granule_list': ','.join(chunk), 'output': 'geojson'}, retries, timeout_second)
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

import io


class _RangeReader(io.RawIOBase):
    """A read-only, seekable file over the byte ranges of a remote file, for zipfile: the reads go through
    `read(start, end)` in aligned 64 KiB blocks, which are kept, so that the ZIP directory and the local headers of
    a large archive are read with a few small requests."""

    def __init__(self, read, size):
        """
        Parameters
        ----------
        read : callable
            read(start, end) -> the bytes of the remote file from start to end (end exclusive).
        size : int
            The size of the remote file in bytes.

        The error of the last failed read is kept in the attribute error (None at start): zipfile reports a failed
        read of the archive's directory as BadZipFile ('File is not a zip file'), so the caller raises the error
        itself.

        Examples
        --------
        >>> _RangeReader(lambda start, end: data[start:end], len(data))
        """
        super().__init__()
        self._read = read
        self._size = size
        self._position = 0
        self._block = 64 * 1024
        self._cache = {}
        self.error = None

    def readable(self):
        """
        Whether the file can be read: always True.

        Examples
        --------
        >>> reader.readable()
        True
        """
        return True

    def seekable(self):
        """
        Whether the file supports seek: always True.

        Examples
        --------
        >>> reader.seekable()
        True
        """
        return True

    def tell(self):
        """
        The current position in the file.

        Examples
        --------
        >>> reader.tell()
        """
        return self._position

    def seek(self, offset, whence=0):
        """
        Move the position: from the start (whence 0), the current position (1) or the end (2).

        Parameters
        ----------
        offset : int
            The offset in bytes.
        whence : int, optional
            0, 1 or 2 as io.IOBase.seek. Default 0.

        Returns
        -------
        int
            The new position.

        Raises
        ------
        ValueError
            For another whence or a negative position.

        Examples
        --------
        >>> reader.seek(-22, 2)
        """
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = self._size + offset
        else:
            raise ValueError(f'whence {whence} is not 0, 1 or 2')
        if position < 0:
            raise ValueError(f'negative seek position {position}')
        self._position = position
        return position

    def readinto(self, buffer):
        """
        Read bytes from the current position into a buffer: the blocks not kept yet are requested in one range.

        Parameters
        ----------
        buffer : bytearray or memoryview
            The buffer to fill.

        Returns
        -------
        int
            The number of bytes read: 0 at the end of the file.

        Examples
        --------
        >>> reader.readinto(bytearray(30))
        """
        n = min(len(buffer), max(0, self._size - self._position))
        if n == 0:
            return 0
        first = self._position // self._block
        last = (self._position + n - 1) // self._block
        missing = [k for k in range(first, last + 1) if k not in self._cache]
        if missing:
            start = missing[0] * self._block
            try:
                data = self._read(start, min(self._size, (missing[-1] + 1) * self._block))
            except Exception as e:
                # kept for the caller: zipfile replaces it by BadZipFile
                self.error = e
                raise
            for k in range(missing[0], missing[-1] + 1):
                self._cache[k] = data[k * self._block - start:(k + 1) * self._block - start]
        data = b''.join(self._cache[k] for k in range(first, last + 1))
        offset = self._position - first * self._block
        buffer[:n] = data[offset:offset + n]
        self._position += n
        return n


def _asf_burst_records(bursts, missing_note=None, retries=30, timeout_second=3):
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
    retries : int, optional
        Attempts of each catalog search, passed to the granule search. Default 30.
    timeout_second : int, optional
        Seconds between the attempts of a catalog search. Default 3.

    Returns
    -------
    list
        _ASFSearchResult records: those of the catalog, in the requested order, then those rebuilt from a sibling.
        A burst that neither the catalog nor a sibling gives has none.
    """
    from tqdm.auto import tqdm

    with tqdm(desc=f'Downloading ASF Catalog'.ljust(25), total=1) as pbar:
        results = asf_search.granule_search(bursts, retries=retries, timeout_second=timeout_second)
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
        for result in asf_search.granule_search(sorted(siblings - set(catalog)), retries=retries,
                                                timeout_second=timeout_second):
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
# The first request of a NISAR granule's metadata (64 MiB, _NisarRemoteFile): the superblock and most of the structure
_NISAR_HEAD_BYTES = 64 * 1024 * 1024
# The page of every further metadata read of a NISAR granule (16 MiB, _NisarRemoteFile): the structure past the head
_NISAR_PAGE_BYTES = 16 * 1024 * 1024
# The largest byte range the NISAR direct route reads at once (64 MiB): one block of chunks, written before the next is read
_NISAR_BLOCK_BYTES = 64 * 1024 * 1024
# Cloudflare Worker cache proxy for ALOS-2 (handles auth internally)
# API: /FILE_ID/OFFSET/LENGTH.bin → LENGTH bytes (≤128MB) of the granule's ZIP at OFFSET
_ALOS2_CACHE_PROXY = 'https://alos2-cache-asf.insar.dev'
_NISAR_BLOCK_SIZE = 128 * 1024 * 1024  # 128 MB blocks
# Minimum side of a NISAR bbox crop on the ground: smaller crops cannot be processed accurately
_NISAR_MIN_CROP_KM = 20
# The ALOS-2 line window of a bbox is rounded outward to multiples of this many image lines: a margin for the
# resampling and for the shifts between dates, as the 512 x 512 chunks of a NISAR crop give
_ALOS2_LINE_BLOCK = 512
# The largest block of ALOS-2 image records read and written at once (64 MiB), below the proxy's 128 MB per range
_ALOS2_BLOCK_BYTES = 64 * 1024 * 1024
# The ALOS-2 ScanSAR subswath footprints in the catalog polygon of a scene (ASF.search: the catalog holds one polygon per
# scene, none per subswath), measured against JAXA's line coordinates: session/patches/alos2/PLAN.md, U0 and its
# 2026-11-20 extension (23 right- and 18 left-looking scenes). Lengths in km, in the tangent plane at the scene centre.
_ALOS2_FOOTPRINT = {
    # a catalog polygon is a scene outline when its 4 sides and 4 angles lie in these ranges (measured: 351-370 km and
    # 89.5-90.5 degrees; the broken ones, near the South Pole, have sides of 9-506 km and angles of 44-167 degrees)
    'side_km': (330.0, 390.0),
    'angle_deg': (85.0, 95.0),
    # the heading at the scene from its frame F: argument of latitude frame_deg * F, orbit inclination, and the Earth
    # rotation term of the ground track (earth_rotation * cos(latitude)); it picks the near edge (40 of 40 scenes)
    'frame_deg': 0.05,
    'inclination_deg': 97.9,
    'earth_rotation': 0.067,
    # per subswath: the across-track fractions of the scene width (near edge 0, far edge 1, the same on both
    # across-track sides) and the terrain margins of its near and far side for heights -500..9000 m
    # (500 m / tan(near incidence), 9000 m / tan(far incidence))
    'subswaths': {
        'F1': (-0.0033, 0.2305, 1.05, 14.33),
        'F2': (0.2165, 0.4078, 0.81, 12.10),
        'F3': (0.3941, 0.6134, 0.68, 10.22),
        'F4': (0.5996, 0.8049, 0.57, 8.90),
        'F5': (0.7912, 1.0036, 0.50, 7.81),
    },
    # the margin of the fractions across track per look side: the largest measured error is 3.3 km over 23
    # right-looking scenes and 8.5 km over 13 left-looking ones
    'model_km': {'R': 4.0, 'L': 9.0},
    # the margin along track per look side, beyond the start and end edges of the catalog polygon: the true subswaths
    # reach up to 2.92 km (right-looking) and 4.52 km (left-looking) beyond them (43 scenes); terrain height does not
    # move points along track (zero-Doppler images)
    'along_km': {'R': 3.0, 'L': 5.0},
}
# NISAR product name (fileID): level config product cycle track direction frame mode polarization x start stop crid processing coverage source counter
_NISAR_NAME = re.compile(
    r'^NISAR_L(?P<level>\d)_(?P<config>[A-Z]{2})_(?P<product>[A-Z]+)_(?P<cycle>\d{3})_(?P<track>\d{3})'
    r'_(?P<direction>[AD])_(?P<frame>\d{3})_(?P<mode>\d{4})_(?P<polarization>[A-Z]{4})_(?P<x>[A-Z])'
    r'_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<crid>[A-Z]\d{5})_(?P<processing>[A-Z])'
    r'_(?P<coverage>[A-Z])_(?P<source>[A-Z])_(?P<counter>\d{3})$')


class _NisarRemoteFile(io.RawIOBase):
    """A read-only, seekable file over one remote NISAR granule, for h5py: `h5py.File(remote, 'r')` reads the granule's
    metadata wherever the granule stores it, without downloading the granule.

    Design: on creation the first _NISAR_HEAD_BYTES (64 MiB, at most the file) are read in one request; they hold the
    HDF5 superblock, whose end-of-file address is the size of the file, and most of the structure (groups, object
    headers, chunk indexes). Every later read of bytes that are not kept fetches the _NISAR_PAGE_BYTES-aligned pages
    (16 MiB, the last page clipped at the size) that cover them, one request per missing page: the structure past the
    head arrives on demand. prefetch(ranges) fetches the given byte ranges that are not kept (the values of the metadata
    datasets, which a granule may store gigabytes past its structure) in ONE request of the route, and keeps them as
    they are. Everything fetched is kept for the life of the object (one granule); the requests and bytes fetched are
    counted in the attributes requests and bytes_read.

    Every network operation is the route's read_ranges(ranges): it takes a list of (offset, length) and returns a list
    of bytes of exactly those lengths, with the route's own retries.

    Examples
    --------
    >>> remote = _NisarRemoteFile(lambda ranges: [data[offset:offset + length] for offset, length in ranges])
    """

    def __init__(self, read_ranges, size=None):
        """
        Parameters
        ----------
        read_ranges : callable
            read_ranges(ranges) -> a list of bytes, one per (offset, length) of ranges, of exactly that length: one
            network operation of the route, with its retries.
        size : int or None
            The size of the file when the route knows it (the direct route's HEAD); None reads it from the superblock.

        Raises
        ------
        IOError
            When the head does not start with the HDF5 signature (the route retries it), or a read gives other
            lengths than the ones asked for.
        ValueError
            When size is given and differs from the end-of-file address of the superblock, or the superblock version
            is not 0, 1, 2 or 3.

        Examples
        --------
        >>> _NisarRemoteFile(read_ranges, size=file_size)
        """
        import struct
        super().__init__()
        self._read_ranges = read_ranges
        self._position = 0
        # page index -> the bytes of the page; the ranges of prefetch: sorted starts and (start, end, bytes)
        self._pages = {}
        self._starts = []
        self._ranges = []
        self.requests = 0
        self.bytes_read = 0
        length = _NISAR_HEAD_BYTES if size is None else min(_NISAR_HEAD_BYTES, size)
        head = self._fetch([(0, length)])[0]
        if not bytes(head[:8]) == b'\x89HDF\r\n\x1a\n':
            raise IOError(f'NISAR metadata is not HDF5: the first bytes are {bytes(head[:100])!r}')
        # the end-of-file address of the superblock (little-endian unsigned 64 bit): version 0 at bytes 40-48,
        # version 1 at 44-52 (4 more bytes of the indexed storage K), versions 2 and 3 at 28-36
        version = head[8]
        field = {0: 40, 1: 44, 2: 28, 3: 28}.get(version)
        if field is None:
            raise ValueError(f'NISAR metadata: HDF5 superblock version {version} is not supported')
        eof = struct.unpack('<Q', bytes(head[field:field + 8]))[0]
        if size is not None and size != eof:
            raise ValueError(f'NISAR file size {size} differs from the end of file {eof} of its HDF5 superblock')
        self.size = eof
        # the head kept as pages (a head padded past a smaller file is cut at its end)
        view = memoryview(head)[:min(len(head), eof)]
        for k in range(0, len(view), _NISAR_PAGE_BYTES):
            self._pages[k // _NISAR_PAGE_BYTES] = view[k:k + _NISAR_PAGE_BYTES]

    def _fetch(self, ranges):
        """
        The bytes of the ranges by one call of read_ranges, counted in requests and bytes_read.

        Parameters
        ----------
        ranges : list of tuple
            (offset, length) of each range.

        Returns
        -------
        list of bytes
            The bytes of each range.

        Raises
        ------
        IOError
            When read_ranges gives other lengths than the ones asked for.

        Examples
        --------
        >>> remote._fetch([(0, 16)])
        """
        data = self._read_ranges(list(ranges))
        if len(data) != len(ranges) or any(len(d) != n for d, (_, n) in zip(data, ranges)):
            raise IOError(f'NISAR metadata read: got {[len(d) for d in data]} bytes for the ranges {list(ranges)}')
        self.requests += 1
        self.bytes_read += sum(n for _, n in ranges)
        return data

    def _range(self, start, end):
        """
        The kept range of prefetch that holds the bytes start..end, as (its start, its bytes), or None.

        Parameters
        ----------
        start : int
            First byte.
        end : int
            Byte after the last one.

        Returns
        -------
        tuple or None
            (range start, range bytes) or None.

        Examples
        --------
        >>> remote._range(0, 16)
        """
        import bisect
        i = bisect.bisect_right(self._starts, start) - 1
        if i >= 0:
            first, last, data = self._ranges[i]
            if end <= last:
                return first, data
        return None

    def _pieces(self, start, end, fetch=True):
        """
        The bytes start..end as pieces, page by page: a kept page, else a kept range of prefetch, else (fetch) the page
        fetched now and kept.

        Parameters
        ----------
        start : int
            First byte.
        end : int
            Byte after the last one, at most the size.
        fetch : bool
            True fetches the missing pages; False gives None when a piece is not kept.

        Returns
        -------
        list of memoryview or None
            The pieces in order, or None (fetch False) when some bytes are not kept.

        Examples
        --------
        >>> remote._pieces(0, 16)
        """
        pieces = []
        page = _NISAR_PAGE_BYTES
        for k in range(start // page, (end - 1) // page + 1):
            a, b = max(start, k * page), min(end, (k + 1) * page)
            if k not in self._pages:
                kept = self._range(a, b)
                if kept is not None:
                    first, data = kept
                    pieces.append(memoryview(data)[a - first:b - first])
                    continue
                if not fetch:
                    return None
                length = min(page, self.size - k * page)
                self._pages[k] = memoryview(self._fetch([(k * page, length)])[0])
            pieces.append(self._pages[k][a - k * page:b - k * page])
        return pieces

    def prefetch(self, ranges):
        """
        Fetch the given byte ranges that are not kept in ONE call of read_ranges and keep them as they are (not as
        pages); ranges past the end of the file are cut at it, empty ones ignored.

        Parameters
        ----------
        ranges : iterable of tuple
            (offset, length) of each range, such as the storage of the values of the metadata datasets.

        Examples
        --------
        >>> remote.prefetch([(0, 16), (1024, 16)])
        """
        import bisect
        missing = set()
        for offset, length in ranges:
            length = min(length, self.size - offset)
            if offset < 0 or length <= 0:
                continue
            if self._pieces(offset, offset + length, fetch=False) is None:
                missing.add((offset, length))
        if not missing:
            return
        missing = sorted(missing)
        for (offset, length), data in zip(missing, self._fetch(missing)):
            i = bisect.bisect_right(self._starts, offset)
            self._starts.insert(i, offset)
            self._ranges.insert(i, (offset, offset + length, data))

    def readable(self):
        """
        Whether the file can be read: always True.

        Examples
        --------
        >>> remote.readable()
        True
        """
        return True

    def seekable(self):
        """
        Whether the file supports seek: always True.

        Examples
        --------
        >>> remote.seekable()
        True
        """
        return True

    def tell(self):
        """
        The current position in the file.

        Examples
        --------
        >>> remote.tell()
        """
        return self._position

    def seek(self, offset, whence=0):
        """
        Move the position: from the start (whence 0), the current position (1) or the end (2).

        Parameters
        ----------
        offset : int
            The offset in bytes.
        whence : int, optional
            0, 1 or 2 as io.IOBase.seek. Default 0.

        Returns
        -------
        int
            The new position.

        Raises
        ------
        ValueError
            For another whence or a negative position.

        Examples
        --------
        >>> remote.seek(0, 2)
        """
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f'whence {whence} is not 0, 1 or 2')
        if position < 0:
            raise ValueError(f'negative seek position {position}')
        self._position = position
        return position

    def readinto(self, buffer):
        """
        Read bytes from the current position into a buffer: the bytes not kept are fetched as the pages that cover
        them.

        Parameters
        ----------
        buffer : bytearray or memoryview
            The buffer to fill.

        Returns
        -------
        int
            The number of bytes read: 0 at the end of the file.

        Examples
        --------
        >>> remote.readinto(bytearray(96))
        """
        n = min(len(buffer), max(0, self.size - self._position))
        if n == 0:
            return 0
        out = memoryview(buffer).cast('B')
        pos = 0
        for piece in self._pieces(self._position, self._position + n):
            out[pos:pos + len(piece)] = piece
            pos += len(piece)
        self._position += n
        return n


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
            Granule identifier (S1 burst, NISAR granule, ALOS-2 subswath or ALOS-1 PALSAR product).

        Returns
        -------
        str
            'S1' for Sentinel-1, 'NISAR' for NISAR, 'ALOS2' for ALOS-2 (a name starting with ALOS2), 'ALOS1' for
            ALOS-1 PALSAR (a name starting with ALPSRP).

        Raises
        ------
        ValueError
            If mission cannot be detected from name.

        Examples
        --------
        >>> ASF._detect_mission('ALOS2592361150-250512-WBDR1.1__A-F2')
        'ALOS2'
        >>> ASF._detect_mission('ALPSRP131296360-L1.1')
        'ALOS1'
        """
        if granule_name.startswith('S1_') and granule_name.endswith('-BURST'):
            return 'S1'
        elif granule_name.startswith('NISAR_'):
            return 'NISAR'
        elif granule_name.startswith('ALOS2'):
            return 'ALOS2'
        elif granule_name.startswith('ALPSRP'):
            return 'ALOS1'
        else:
            raise ValueError(f"Unknown mission for granule: {granule_name}. "
                           f"Expected S1_*-BURST, NISAR_*, ALOS2* or ALPSRP* format.")

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
            True if all files of the burst exist and are regular files: the 4 of the burst, and for an IW1 or IW3 burst
            its IW2 annotation (annotation/<name>.xml with the burst name changed to IW2), which a burst downloaded
            before lacks (ASF fetches the burst again). An empty one raises.
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
        # the IW2 annotation of an IW1 or IW3 burst: the burst name with its subswath token changed to IW2
        if parts[2] in ('IW1', 'IW3'):
            files.append(os.path.join(burst_dir, 'annotation', '_'.join(parts[:2] + ['IW2'] + parts[3:]) + '.xml'))

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

    @staticmethod
    def _alos2_parse(granule):
        """
        Parse the ID of one ALOS-2 ScanSAR subswath, `<fileID>-F<n>`, without network.

        The fileID is ALOS2 + orbit (5 digits) + frame (4 digits) + '-' + YYMMDD + '-' + the product code (mode, one
        letter, '1.1__', one letter), as the column sceneName of ASF.search gives it; F<n> is the subswath 1..5.

        Parameters
        ----------
        granule : str
            The subswath ID, such as 'ALOS2592361150-250512-WBDR1.1__A-F2'.

        Returns
        -------
        dict
            'scene' (ALOS2 + orbit + frame + '-' + YYMMDD), 'fileID', 'orbit', 'frame', 'path'
            ((14 * orbit + 23) mod 207 + 1), 'date' ('YYYYMMDD'), 'mode' (the three mode letters of the product
            code: one for the HH products, one for the HH+HV products) and 'subswath' (int).

        Raises
        ------
        ValueError
            One line, for an ID that is not an ALOS-2 subswath ID, a scene name without the product code, a fileID
            without its subswath, a product other than the 5-subswath ScanSAR products in HH or HH+HV, or a subswath
            outside F1..F5.

        Examples
        --------
        >>> ASF._alos2_parse('ALOS2592361150-250512-WBDR1.1__A-F2')['path']
        86
        """
        import re
        m = re.match(r'^(ALOS2(\d{5})(\d{4})-(\d{6}))(?:-([A-Z]{3})[A-Z]1\.1__[A-Z])?(?:-F(\d))?$', granule)
        if m is None:
            raise ValueError(f"Not an ALOS-2 subswath ID: {granule}")
        if m.group(5) is None:
            raise ValueError(f"ALOS-2 ID {granule} needs the fileID with its subswath, such as <fileID>-F1 "
                             f"(column sceneName of ASF.search)")
        fileID = granule if m.group(6) is None else granule[:m.start(6) - 2]
        if m.group(6) is None:
            raise ValueError(f"ALOS-2 ID {granule} names no subswath: use {fileID}-F1, {fileID}-F2, {fileID}-F3, "
                             f"{fileID}-F4 or {fileID}-F5")
        mode = m.group(5)
        if mode not in ('WBD', 'WBS'):
            raise ValueError(f"ALOS-2 product {granule} is not supported: only the 5-subswath ScanSAR products in HH or "
                             f"HH+HV are")
        n = int(m.group(6))
        if not 1 <= n <= 5:
            raise ValueError(f"ALOS-2 subswath F{n} of {granule} does not exist (F1..F5)")
        orbit = int(m.group(2))
        return {'scene': m.group(1), 'fileID': fileID, 'orbit': orbit, 'frame': int(m.group(3)),
                'path': (14 * orbit + 23) % 207 + 1, 'date': '20' + m.group(4), 'mode': mode, 'subswath': n}

    @staticmethod
    def _alos2_path(basedir, info, polarization=None):
        """
        The file of an ALOS-2 subswath: `<basedir>/<path>_<frame>_F<n>/AL2_<path>_<frame>_F<n>_<YYYYMMDD>_<pol>.img`,
        or its `.led` (no polarization) for polarization=None.

        Parameters
        ----------
        basedir : str
            The download folder.
        info : dict
            The subswath as ASF._alos2_parse gives it.
        polarization : str or None, optional
            The polarization of the image file; None for the leader file. Default None.

        Returns
        -------
        str
            The path of the file.

        Examples
        --------
        >>> ASF._alos2_path('data', ASF._alos2_parse('ALOS2592361150-250512-WBDR1.1__A-F2'), 'HH')
        'data/086_1150_F2/AL2_086_1150_F2_20250512_HH.img'
        """
        import os
        name = f"{info['path']:03d}_{info['frame']:04d}_F{info['subswath']}"
        suffix = '.led' if polarization is None else f'_{polarization}.img'
        return os.path.join(basedir, name, f"AL2_{name}_{info['date']}{suffix}")

    @staticmethod
    def _alos2_pols(polarizations, info):
        """
        The polarizations of an ALOS-2 subswath to download: its product has HH and HV (HH+HV) or HH (HH); the
        requested ones that it has (a WARNING names the others), or every one it has for polarizations=None.

        Parameters
        ----------
        polarizations : list or None
            The requested polarizations.
        info : dict
            The subswath as ASF._alos2_parse gives it.

        Returns
        -------
        list
            The polarizations to download.

        Examples
        --------
        >>> ASF._alos2_pols(None, ASF._alos2_parse('ALOS2592361150-250512-WBDR1.1__A-F2'))
        ['HH', 'HV']
        """
        available = ['HH', 'HV'] if info['mode'] == 'WBD' else ['HH']
        if polarizations is None:
            return list(available)
        missing = [p for p in polarizations if p not in available]
        if missing:
            print(f"WARNING: Polarizations {missing} not available in {info['fileID']}-F{info['subswath']}")
        return [p for p in polarizations if p in available]

    @staticmethod
    def _alos2_complete(basedir, info, polarizations):
        """
        Whether every file of an ALOS-2 subswath already exists, for skip_exist=True, without network: its `.led`
        and the `.img` of every polarization of ASF._alos2_pols.

        Parameters
        ----------
        basedir : str
            The download folder.
        info : dict
            The subswath as ASF._alos2_parse gives it.
        polarizations : list or None
            The requested polarizations (None: every one the product has).

        Returns
        -------
        bool
            True when every file exists. An empty file raises.

        Examples
        --------
        >>> ASF._alos2_complete('data', ASF._alos2_parse('ALOS2592361150-250512-WBDR1.1__A-F2'), None)
        """
        from .utils_files import exists
        files = [ASF._alos2_path(basedir, info)] + [ASF._alos2_path(basedir, info, p)
                                                    for p in ASF._alos2_pols(polarizations, info)]
        # every file is checked, so that an empty one raises
        return all([exists(path) for path in files])

    @staticmethod
    def _alos1_parse(granule):
        """
        Parse the ID of one ALOS-1 PALSAR L1.1 (SLC) product, `ALPSRP<orbit><frame>-L1.1`, or L1.0 (raw signal data)
        product, `ALPSRP<orbit><frame>-L1.0`, without network.

        The ID is the product's fileID in the ASF catalog (the column sceneName of ASF.search): ALPSRP + orbit (5
        digits) + frame (4 digits) + '-L1.1' or '-L1.0' (measured 2026-11-23: the L1.0 fileID, such as
        'ALPSRP055006360-L1.0', and its ZIP 'ALPSRP055006360-L1.0.zip'). The path is (46 * orbit + 84) mod 671 + 1
        (checked against the catalog's pathNumber of 17,500 L1.1 products of 283 paths, 3..671, 2026-11-22). The ID
        holds no date: 'date' is the UTC date of the scene centre predicted from the orbit and the frame, t =
        2006-01-24T01:30:51.5 UTC + 5923.0978 s * (orbit + frame / 7200), fitted to the catalog's scene centres of the
        same 17,500 products (largest error 214 s): it can be one day off for a scene within minutes of midnight. The
        files of a product are named with the date of its first image line (ASF._download_alos1); ASF._alos1_complete
        looks for them within one day of this date (a path and frame repeat every 46 days).

        Parameters
        ----------
        granule : str
            The product ID, such as 'ALPSRP131296360-L1.1' or 'ALPSRP055006360-L1.0'.

        The level is the ID's suffix, 'L1.1' or 'L1.0': the supported levels, which ASF.download and the file names
        (ASF._alos1_path) follow (info['level']).

        Returns
        -------
        dict
            'scene' (ALPSRP + orbit + frame), 'fileID', 'level' ('L1.1' or 'L1.0'), 'orbit', 'frame', 'path' and
            'date' ('YYYYMMDD', predicted).

        Raises
        ------
        ValueError
            One line, for an ID that is not an ALOS-1 PALSAR product ID, a scene name without the '-L1.1' or '-L1.0'
            suffix, or a product of another level than L1.1 and L1.0.

        Examples
        --------
        >>> ASF._alos1_parse('ALPSRP131296360-L1.1')['path']
        119
        >>> ASF._alos1_parse('ALPSRP131296360-L1.1')['date']
        '20080712'
        >>> ASF._alos1_parse('ALPSRP055006360-L1.0')['level']
        'L1.0'
        """
        import re
        from datetime import datetime, timedelta, timezone
        m = re.match(r'^(ALPSRP(\d{5})(\d{4}))(?:-(.+))?$', granule)
        if m is None:
            raise ValueError(f"Not an ALOS-1 PALSAR product ID: {granule}")
        if m.group(4) is None:
            raise ValueError(f"ALOS-1 ID {granule} needs the product suffix: use {granule}-L1.1 (SLC) or "
                             f"{granule}-L1.0 (raw) (column sceneName of ASF.search)")
        # the supported levels (the ID suffix)
        if m.group(4) not in ('L1.1', 'L1.0'):
            raise ValueError(f"ALOS-1 product {granule} is not supported: only the L1.1 (SLC) and L1.0 (raw) products "
                             f"are")
        orbit, frame = int(m.group(2)), int(m.group(3))
        centre = datetime(2006, 1, 24, 1, 30, 51, 500000, tzinfo=timezone.utc) + \
            timedelta(seconds=5923.0978 * (orbit + frame / 7200))
        return {'scene': m.group(1), 'fileID': granule, 'level': m.group(4), 'orbit': orbit, 'frame': frame,
                'path': (46 * orbit + 84) % 671 + 1, 'date': centre.strftime('%Y%m%d')}

    @staticmethod
    def _alos1_path(basedir, info, polarization=None):
        """
        The file of an ALOS-1 product: `<basedir>/<path>_<frame>_<offnadir>/AL1_<path>_<frame>_<offnadir>_<YYYYMMDD>_<pol>.img`,
        or its `.led` (no polarization) for polarization=None; <offnadir> is the off-nadir angle in tenths of a degree
        (3 digits). The one place where the names of the ALOS-1 files are formed, per level (info['level']): an L1.1
        (SLC) product as above, an L1.0 (raw) product in the folder `<path>_<frame>_<offnadir>_raw` with the same code,
        `AL1_<path>_<frame>_<offnadir>_<YYYYMMDD>_<pol>.raw` and its own `.led` (the leader files of the two levels
        differ). Raw and SLC files never share a folder.

        Parameters
        ----------
        basedir : str
            The download folder.
        info : dict
            The product as ASF._alos1_parse gives it, with 'offnadir' (degrees, from the product's leader file) and
            'date' ('YYYYMMDD', of the first image line) added by the download.
        polarization : str or None, optional
            The polarization of the image file; None for the leader file. Default None.

        Returns
        -------
        str
            The path of the file.

        Examples
        --------
        >>> ASF._alos1_path('data', dict(ASF._alos1_parse('ALPSRP131296360-L1.1'), offnadir=34.3), 'HV')
        'data/119_6360_343/AL1_119_6360_343_20080712_HV.img'
        >>> ASF._alos1_path('data', dict(ASF._alos1_parse('ALPSRP055006360-L1.0'), offnadir=34.3, date='20070205'), 'HH')
        'data/118_6360_343_raw/AL1_118_6360_343_20070205_HH.raw'
        """
        import os
        name = f"{info['path']:03d}_{info['frame']:04d}_{int(round(info['offnadir'] * 10)):03d}"
        raw = info['level'] == 'L1.0'
        suffix = '.led' if polarization is None else f"_{polarization}{'.raw' if raw else '.img'}"
        return os.path.join(basedir, f"{name}_raw" if raw else name, f"AL1_{name}_{info['date']}{suffix}")

    @staticmethod
    def _alos1_pols(polarizations, available, granule):
        """
        The polarizations of an ALOS-1 product to download: the requested ones that it has (a WARNING names the
        others), or every one it has for polarizations=None. The product's polarizations are its image files, the same
        for its L1.1 (SLC) and L1.0 (raw) products (FBS: HH or VV; FBD: HH and HV or VV and VH; PLR: HH, HV, VH and
        VV).

        Parameters
        ----------
        polarizations : list or None
            The requested polarizations.
        available : list
            The polarizations of the product.
        granule : str
            The product ID, for the WARNING.

        Returns
        -------
        list
            The polarizations to download.

        Examples
        --------
        >>> ASF._alos1_pols(None, ['HH', 'HV'], 'ALPSRP131296360-L1.1')
        ['HH', 'HV']
        >>> ASF._alos1_pols(['HV'], ['HH', 'HV'], 'ALPSRP131296360-L1.1')
        ['HV']
        """
        if polarizations is None:
            return list(available)
        missing = [p for p in polarizations if p not in available]
        if missing:
            print(f'WARNING: Polarizations {missing} not available in {granule}')
        return [p for p in polarizations if p in available]

    @staticmethod
    def _alos1_complete(basedir, info, polarizations):
        """
        Whether every file of an ALOS-1 product already exists, for skip_exist=True, without network.

        The files of a product are found by its path and frame and a date within one day of the predicted date of
        ASF._alos1_parse, at any off-nadir angle (a path and frame repeat every 46 days, so at most one product
        matches), in the folder of its level (ASF._alos1_path: `<path>_<frame>_<offnadir>` for L1.1 with `.img` image
        files, `<path>_<frame>_<offnadir>_raw` for L1.0 with `.raw` image files). The product is complete when its
        `.led` exists and either the image file of every requested polarization exists, or the image files that exist
        are as many as the product's image channels (the number of channels of the data set summary record of the
        `.led`, bytes 389-392): every polarization the product has. With polarizations=None only the second rule
        applies.

        Parameters
        ----------
        basedir : str
            The download folder.
        info : dict
            The product as ASF._alos1_parse gives it.
        polarizations : list or None
            The requested polarizations (None: every one the product has).

        Returns
        -------
        bool
            True when every file exists. An empty file raises.

        Raises
        ------
        ValueError
            When more than one leader file matches the product.

        Examples
        --------
        >>> ASF._alos1_complete('data', ASF._alos1_parse('ALPSRP131296360-L1.1'), None)
        False
        >>> ASF._alos1_complete('data', ASF._alos1_parse('ALPSRP055006360-L1.0'), ['HH'])
        False
        """
        import os
        from glob import glob
        from datetime import datetime, timedelta
        from .utils_files import exists
        from .utils_alos2_orbit import _record, _field
        day = datetime.strptime(info['date'], '%Y%m%d')
        dates = [(day + timedelta(days=k)).strftime('%Y%m%d') for k in (-1, 0, 1)]
        name = f"{info['path']:03d}_{info['frame']:04d}"
        raw = info['level'] == 'L1.0'
        folder, extension = (f'{name}_???_raw', '.raw') if raw else (f'{name}_???', '.img')
        leds = sorted({p for d in dates for p in glob(os.path.join(folder, f'AL1_{name}_???_{d}.led'),
                                                         root_dir=basedir)})
        if not leds:
            return False
        if len(leds) > 1:
            raise ValueError(f"Multiple ALOS-1 leader files found for {info['fileID']}: {leds}")
        led_path = os.path.join(basedir, leds[0])
        if not exists(led_path):
            return False
        stem = led_path[:-len('.led')]
        # every file is checked, so that an empty one raises
        stored = [p for p in ('HH', 'HV', 'VH', 'VV') if exists(f'{stem}_{p}{extension}')]
        if polarizations is not None and all(p in stored for p in polarizations):
            return True
        with open(led_path, 'rb') as f:
            led = f.read()
        record = _record(led, 10)
        if record is None:
            raise ValueError(f'{led_path} has no data set summary record (type 10)')
        return len(stored) == int(_field(record, 389, 392))

    @staticmethod
    def _alos1_catalog(ids, retries, timeout_second):
        """
        The catalog entries of ALOS-1 products, found by one catalog search before any of them is downloaded.

        ASF.download calls it before the download of any mission starts, so a product missing from the catalog leaves
        nothing of the call written; ASF._download_alos1 reads its products' URLs and footprints from it.

        Parameters
        ----------
        ids : list
            ALOS-1 product IDs (ASF._alos1_parse), such as 'ALPSRP131296360-L1.1' or 'ALPSRP055006360-L1.0'.
        retries : int
            Attempts of the catalog search, the first one included (HTTP.attempts).
        timeout_second : float
            Seconds waited between the attempts.

        Returns
        -------
        dict
            {fileID: (url, footprint GeoJSON geometry or None, beamModeType or None)}.

        Raises
        ------
        ValueError
            One line, for a product the catalog does not hold.

        Examples
        --------
        >>> ASF._alos1_catalog(['ALPSRP131296360-L1.1', 'ALPSRP055006360-L1.0'], 30, 3)
        """
        products = {i: ASF._alos1_parse(i) for i in dict.fromkeys(ids)}
        print(f"Connecting to ASF...", end='\r', flush=True)
        search_results = asf_search.granule_search(list(products), retries=retries, timeout_second=timeout_second)
        print(f"Searching ASF for {len(products)} product(s)... done ({len(search_results)} found)")
        catalog = {}
        for result in search_results:
            feature = result.geojson()
            props = feature['properties']
            catalog.setdefault(props['fileID'], (props['url'], feature.get('geometry'), props.get('beamModeType')))
        missing = [g for g in products if g not in catalog]
        if missing:
            raise ValueError(f"Granules not found in ASF: {missing}")
        return catalog

    # https://asf.alaska.edu/datasets/data-sets/derived-data-sets/sentinel-1-bursts/
    def download(self, basedir, bursts, polarization=None, frequency=None, bbox=None, session=None, n_jobs=None, joblib_backend='loky', skip_exist=True,
                        retries=30, timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Download SAR data from ASF.

        Supports Sentinel-1 bursts, NISAR RSLC granules, ALOS-2 PALSAR-2 ScanSAR L1.1 subswaths (the 5-subswath
        products in HH or HH+HV), ALOS-1 PALSAR L1.1 (SLC) products and ALOS-1 PALSAR L1.0 (raw signal data) products
        of the stripmap modes FBS, FBD and PLR. Mission is auto-detected from ID format.

        An ALOS-1 product is named by its fileID in the ASF catalog (the column sceneName of ASF.search):
        'ALPSRP131296360-L1.1' (SLC) or 'ALPSRP055006360-L1.0' (raw). A scene name without '-L1.1' or '-L1.0' or a
        product of another level raises a one-line ValueError before anything is downloaded. An FBS SLC (one
        polarization, HH or VV, with twice the range samples of FBD: 32 MHz sampling) is downloaded the same way as an
        FBD SLC, its samples unchanged. The
        catalog entries of the ALOS-1 products are looked up before the download of any mission starts, so a missing
        ALOS-1 product leaves nothing of the call written (Sentinel-1, NISAR and ALOS-2 IDs of the same call included).
        ALOS-1 needs an Earthdata account, ASF(username, password): there is no cache proxy for it, and ASF() raises a
        one-line ValueError for a product to download. Its files are cropped CEOS files in a folder per (path, frame,
        off-nadir angle in tenths of a degree), with the path (46 * orbit + 84) mod 671 + 1, the off-nadir angle of the
        product's leader file and the date of its first image line:
        <basedir>/119_6360_343/AL1_119_6360_343_20080712_HH.img (and _HV.img) and AL1_119_6360_343_20080712.led; the SLC
        dates of one path, frame and off-nadir angle (FBS, FBD and PLR) share it. The raw
        products of one path, frame and off-nadir angle (FBS, FBD and PLR dates) share the folder
        <basedir>/118_6360_343_raw, with the same code: AL1_118_6360_343_20070205_HH.raw and its own leader file
        AL1_118_6360_343_20070205.led (the line records of the raw signal data, ASF._alos1_path); raw and SLC files
        never share a folder. The image and leader files are written as for ALOS-2. The ZIP of a product stores its
        files uncompressed or deflate-compressed (the '-L1.1.zip' packaging of some FBS and PLR products): the lines of
        a stored image file are read by byte ranges, those of a compressed one by decompressing it from its start to the
        last line kept (never the whole file in memory).

        An ALOS-2 subswath is named by the granule's fileID (the column sceneName of ASF.search) and the subswath
        F1..F5, as JAXA names its files: 'ALOS2592361150-250512-WBDR1.1__A-F2'. A fileID without its subswath, a
        scene name without the product code, another product or subswath raise a one-line ValueError before anything is
        downloaded. The Earthdata account is optional, as for NISAR: with one, ASF(username, password), ALOS-2 is read
        directly from ASF; without one, ASF(), through the cache proxy alos2-cache-asf.insar.dev. Its
        files are cropped CEOS files in a folder per (path, frame, subswath), with the path
        (14 * orbit + 23) mod 207 + 1 and the date of the ID: <basedir>/086_1150_F2/AL2_086_1150_F2_20250512_HH.img
        (and _HV.img) and AL2_086_1150_F2_20250512.led. The image files keep the line records (prefix and samples) of
        the source unchanged, with the line counts of the file descriptor and the record sequence numbers set for the
        crop; the leader file is the source's.

        Sentinel-1 bursts of subswaths IW1 and IW3 also get the annotation of the IW2 product of the scene, from the
        burst manifest, in the burst folder as annotation/<name>.xml, where <name> is the burst name with the subswath
        token changed to IW2: S1_135523_IW1_20221016T015057_VV_6681-BURST gets
        annotation/S1_135523_IW2_20221016T015057_VV_6681-BURST.xml. An IW2 burst gets no such file. When the manifest
        holds no IW2 product, a WARNING is printed and the file is not stored. The file is mandatory for ASF: a burst
        folder without it is incomplete, and a rerun over it (skip_exist=True) fetches the burst again, which adds that
        file when the manifest holds the IW2 product. CDSE does not need the file (see CDSE.download).

        The annotation of a Sentinel-1 burst holds its orbit: generalAnnotation/orbitList, with the attribute source =
        the name of the orbit (EOF) file it comes from, holds the orbit state vectors of that file from the last one at
        or before 120 s (utils_eof.ORBIT_MARGIN) before the burst start to the first one at or after 120 s after the
        burst stop. The file is the one of the burst's mission whose validity covers the burst, a precise orbit (POEORB)
        first, then the newest production. The orbits are downloaded with the bursts and kept in memory: no orbit file
        is written. A burst whose orbit is not published is not downloaded and gets a WARNING; a burst that fails after
        its attempts gets a WARNING; the call does not raise for them, and one WARNING line counts the bursts not stored
        (not in the catalog, without an orbit, failed to download). The next download fetches them. A
        repeated call over complete burst folders changes no file and makes no network request.

        Parameters
        ----------
        basedir : str
            Output directory.
        bursts : str or list
            Burst/granule identifiers. Can be:
            - S1 burst: 'S1_262885_IW2_20190702T032452_VV_69C5-BURST'
            - NISAR RSLC: 'NISAR_L1_PR_RSLC_006_172_A_008_2005_DHDH_A_20251204T024618_...'
            - ALOS-2 subswath: 'ALOS2592361150-250512-WBDR1.1__A-F2'
            - ALOS-1 product: 'ALPSRP131296360-L1.1' (SLC), 'ALPSRP055006360-L1.0' (raw)
            - Newline-separated string of multiple IDs
        polarization : None, str, or list, optional
            Polarization(s) to download.
            - None: S1 uses pol from name, NISAR downloads all available, ALOS-2 HH and HV of an HH+HV product,
              HH of an HH product (a requested polarization the product lacks gets a WARNING and is skipped),
              ALOS-1 every polarization of the product (FBS one, FBD two, PLR four)
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
            ALOS-2: optional as well. Without it every listed subswath is downloaded whole; with it only the lines
            of the subswath that the bbox reaches at any height from -500 to 9000 m (all range pixels), from the
            orbit of the leader file, rounded outward to multiples of 512 lines (a margin for the resampling and the
            shifts between dates; the image's last line ends the last block). A listed subswath that the bbox does
            not reach gets a WARNING and nothing is stored for it (no folder, no file). The same 20 km minimum and
            checks apply.
            ALOS-1: the same as ALOS-2, for the whole product image. An ALOS-1 raw (L1.0) frame keeps the lines that
            hold every echo of the transform's grid: its beam-centre lines widened on each side by half the synthetic
            aperture (utils_alos2_orbit.raw_window), rounded outward to multiples of 512 lines.
        session : asf_search.ASFSession, optional
            Authenticated session. Created automatically if None.
        n_jobs : int or None, optional
            Parallel download jobs, at most two per CPU this process may use (a larger number is capped, with a
            note). Negative values follow joblib: -1 all CPUs, -2 all but one, and so on; 0 raises ValueError.
            None uses mission-specific defaults (S1: 8, NISAR: 4). ALOS-2 subswaths are downloaded one after
            another whatever the value.
        joblib_backend : str, optional
            Backend for parallel processing. Default 'loky' (multiprocessing, faster on Colab).
        skip_exist : bool, optional
            skip_exist=True skips downloading files that already exist; skip_exist=False downloads them again.
            Default True. With skip_exist=False every requested Sentinel-1 burst is downloaded again and all its files
            are written again, its annotation with the newest published orbit. The files of a NISAR granule are those of the requested polarizations, or with
            polarization=None those the product has, read from the metadata of a file of the granule that exists
            (with none, every file is missing). Only the missing files are downloaded, and a granule whose files
            all exist is skipped without network access. The NISAR bbox checks apply to every download that will
            happen, and to nothing else. The files of an ALOS-2 subswath are its .led and the .img of every
            polarization to download, known from the ID alone: a subswath whose files all exist is skipped without
            network access and without an account. The files of an ALOS-1 product are its .led and the .img (.raw for
            L1.0) of every requested polarization, or of every polarization the product has (the channel count of its
            stored .led):
            a product whose files all exist is skipped without network access and without an account.
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
            The files downloaded by this call (the column 'burst' for Sentinel-1: the bursts stored by this call;
            'file' for NISAR and for the ALOS-2 and ALOS-1 image files), or None when nothing is downloaded (every file exists,
            or no burst is stored).

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
        >>> # ALOS-2: HH and HV of subswath F3 of an HH+HV scene, the lines a 22 x 22 km bbox reaches
        >>> asf.download('data/', 'ALOS2100352900-160401-WBDR1.1__D-F3', bbox=(-121.228, 35.504, -120.978, 35.704))
        >>> # ALOS-1: HH and HV of an FBD product, the lines a 20 x 20 km bbox reaches
        >>> asf.download('data/', 'ALPSRP131296360-L1.1', bbox=(-72.2092, -40.545, -72.0908, -40.455))
        >>> # ALOS-1: HH of an FBS product, the lines a 20 x 21 km bbox reaches
        >>> asf.download('data/', 'ALPSRP055006360-L1.1', bbox=(-72.27, -40.595, -72.03, -40.405))
        >>> # ALOS-1 raw: HH of an FBS frame, the echoes of a 20 x 21 km bbox
        >>> asf.download('data/', 'ALPSRP055006360-L1.0', bbox=(-72.27, -40.595, -72.03, -40.405))
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
        alos2_ids = []
        alos1_ids = []
        for burst in bursts:
            mission = self._detect_mission(burst)
            if mission == 'S1':
                s1_bursts.append(burst)
            elif mission == 'NISAR':
                nisar_granules.append(burst)
            elif mission == 'ALOS2':
                alos2_ids.append(burst)
            elif mission == 'ALOS1':
                alos1_ids.append(burst)
        # every ALOS-2 and ALOS-1 ID is checked before any request or folder (a one-line ValueError)
        alos2_info = {i: self._alos2_parse(i) for i in alos2_ids}
        alos1_info = {i: self._alos1_parse(i) for i in alos1_ids}

        # an empty polarization list selects nothing to download and raises, before any request or folder
        pols = polarizations(polarization, "'HH' or ['HH', 'HV']"
                             if (nisar_granules or alos2_ids or alos1_ids) and not s1_bursts
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

        # the polarizations of each ALOS-2 subswath (a WARNING names the requested ones its product lacks); a subswath
        # with none of them has nothing to download
        alos2_pols = {i: self._alos2_pols(pols, alos2_info[i]) for i in dict.fromkeys(alos2_ids)}

        # Check if any downloads needed BEFORE creating session (avoid network call)
        if skip_exist:
            # Filter S1 bursts that need download
            s1_needed = [b for b in s1_bursts if not self._burst_exists(basedir, b)]
            # Filter NISAR granules that need download: a granule with a missing file; the downloaders then skip its
            # files that exist, so only the missing ones are downloaded
            nisar_needed = [g for g in nisar_granules if not self._nisar_complete(basedir, g, pols)]
            # ALOS-2 subswaths with a missing file (the polarizations of a product follow from its mode)
            alos2_needed = [i for i, p in alos2_pols.items()
                            if p and not self._alos2_complete(basedir, alos2_info[i], p)]
            # ALOS-1 products with a missing file (the polarizations of a product are known from its stored leader file)
            alos1_needed = [i for i in dict.fromkeys(alos1_ids) if not self._alos1_complete(basedir, alos1_info[i], pols)]
        else:
            s1_needed = s1_bursts
            nisar_needed = nisar_granules
            alos2_needed = [i for i, p in alos2_pols.items() if p]
            alos1_needed = list(dict.fromkeys(alos1_ids))

        # NISAR bbox crop: checked for the granules that will be downloaded, before any network call (session, S1
        # and both NISAR paths); a granule skipped as existing is not checked
        if nisar_needed and bbox is not None:
            self._nisar_check_bbox(bbox)
        # the same minimum for the ALOS-2 subswaths that will be downloaded
        if alos2_needed and bbox is not None:
            self._nisar_check_bbox(bbox, mission='ALOS-2')
        # and for the ALOS-1 products that will be downloaded, which need an account (no cache proxy for ALOS-1)
        if alos1_needed and bbox is not None:
            self._nisar_check_bbox(bbox, mission='ALOS-1')
        if alos1_needed and self.username is None:
            raise ValueError("ALOS-1 needs an Earthdata account, ASF(username, password): there is no cache proxy for "
                             "ALOS-1. Nothing was downloaded.")

        # Return early if nothing to download (no network call!)
        if not s1_needed and not nisar_needed and not alos2_needed and not alos1_needed:
            return None

        # the catalog entries of the ALOS-1 products that will be downloaded, before the download of any mission: a
        # product missing from the catalog raises with nothing of the call written
        alos1_catalog = self._alos1_catalog(alos1_needed, retries, timeout_second) if alos1_needed else None

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

        # Download ALOS-2 subswaths
        if alos2_needed:
            df = self._download_alos2(basedir, alos2_needed, pols, bbox, n_jobs, skip_exist, retries, timeout_second,
                                      min_rate, min_rate_window, debug)
            if df is not None:
                results.append(df)

        # Download ALOS-1 products of both levels (the ID suffix of ASF._alos1_parse): L1.1, the cropped SLC products,
        # and L1.0, the cropped raw frames (ASF._download_alos1 follows info['level'])
        if alos1_needed:
            df = self._download_alos1(basedir, alos1_needed, pols, bbox, n_jobs, skip_exist, retries, timeout_second,
                                      min_rate, min_rate_window, debug, catalog=alos1_catalog)
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
        basedir : str
            Output directory; each burst is written into its own folder under it.
        bursts : list
            List of S1 burst IDs.
        polarizations : list or None
            If None, use polarization from burst name.
            If list, replace polarization in burst name with each requested pol.
        session : asf_search.ASFSession or requests.Session
            Session that every burst request is sent through (send(session.get, ...)), created by download().
        n_jobs : int or None
            Parallel burst downloads (joblib), at most two per CPU this process may use (a larger number is capped,
            with a note). Negative values follow joblib: -1 all CPUs, -2 all but one, and so on; 0 raises ValueError.
            None gives the Sentinel-1 default of 8.
        joblib_backend : str
            Backend of the parallel burst downloads; replaced by 'sequential' when debug is True.
        skip_exist : bool
            True skips the bursts whose four files already exist (_burst_exists), and keeps every file of a burst
            that exists; False downloads every burst again and writes all its files again, the annotation with the
            newest published orbit. When no burst is missing, no network request is made.
        retries : int
            Attempts per burst, the first one included; 0 makes one attempt, as 1 does (HTTP.attempts). A failure
            that a retry cannot change (HTTP.final, such as HTTP 404) is not retried.
        timeout_second : float
            Seconds waited between the attempts of one burst, not after the last one.
        min_rate : str or float
            Bytes per second each burst request (the manifest and the TIFF) must keep, a size string such as '100KB'
            or a number, averaged over min_rate_window seconds from its first byte, or the request is retried
            (HTTP.read_body).
        min_rate_window : float
            Seconds over which the rate is averaged.
        debug : bool
            Print debug information of the burst requests; with True the bursts are downloaded one after another.

        The orbit of every burst is read before the bursts are downloaded (EOF._orbit_records, with the same n_jobs,
        backend and request settings) and stored in its annotation (utils_S1.burst_xmls with orbit): the orbitList
        with the attribute source = the orbit file name, a precise orbit (POEORB) first, then the newest production,
        and the records from the last one at or before 120 s before the burst start to the first one at or after 120 s
        after the burst stop. No orbit file is written. A burst without a published orbit is not downloaded, and a
        burst that fails after its attempts is not stored: each gets a WARNING, the call does not raise, and one
        WARNING line counts the bursts not stored.

        Returns
        -------
        pandas.DataFrame or None
            The bursts stored by this call (column 'burst'), or None when none is.
        """
        # S1-specific default: 8 parallel jobs
        if n_jobs is None:
            n_jobs = 8

        import rioxarray as rio
        from tifffile import TiffFile
        from .utils_S1 import measurement_path, slc_shape, tiff_data_offset, write_slc, burst_xmls, iw2_annotation_xml, PAIRS_TIFF_OFFSET
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

        def download_burst(result, basedir, session, orbit):
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
            # the IW2 annotation of an IW1 or IW3 burst, named as the burst with its subswath token changed to IW2
            burst_parts = burst.split('_')
            iw2_file = None
            if subswath in ('IW1', 'IW3'):
                iw2_file = os.path.join(xml_annot_dir, '_'.join(burst_parts[:2] + ['IW2'] + burst_parts[3:]) + '.xml')
            for dirname in [burst_dir, tif_dir, xml_annot_dir, xml_noise_dir, xml_calib_dir]:
                os.makedirs(dirname, exist_ok=True)

            def measurement_exists():
                # with skip_exist=False every file of the burst is written again
                if not skip_exist:
                    return False
                if not exists(tif_file):
                    return False
                if tif_file.endswith('.tiff'):
                    return os.path.getsize(tif_file) >= int(properties['bytes'])
                return True

            # check if all files already exist; every one is checked before any download, so that an empty one raises.
            # With skip_exist=False none is kept: the burst is written again, as one that has none of its files.
            if skip_exist:
                xml_exist = [exists(filepath) for filepath in (xml_file, xml_noise_file, xml_calib_file)]
                iw2_exist = iw2_file is None or exists(iw2_file)
            else:
                xml_exist = [False, False, False]
                iw2_exist = iw2_file is None
            all_exist = measurement_exists() and all(xml_exist) and iw2_exist

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

            metadata = xmltodict.parse(xml_content)['burst']['metadata']

            def of_subswath(kind):
                # the item (product, noise or calibration) of this burst's subswath, found by the swath its own annotation
                # names: the position in the list is not the subswath, as a manifest may lack the IW2 product
                items = [item for item in (metadata[kind] if isinstance(metadata[kind], list) else [metadata[kind]])
                         if item['content'].get('adsHeader', {}).get('swath') == subswath]
                if len(items) != 1:
                    raise Exception(f'ERROR: The burst manifest holds {len(items)} {kind} annotations of {subswath} for {burst}; '
                                    f'one is expected.')
                return items[0]

            content = of_subswath('product')
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
                # the measurement stored: the .nc written below, or the existing one. An XML that exists is kept as it is.
                # The annotation holds the orbit of the burst (utils_eof.set_orbit).
                byte_offset = PAIRS_TIFF_OFFSET if tiff_bytes is not None else tiff_data_offset(tif_file)
                parts = {}
                for kind in ('noise', 'calibration'):
                    content = of_subswath(kind)
                    assert polarization == content['polarisation'], 'ERROR: XML polarization differs from burst polarization'
                    parts[kind] = content['content']
                built = burst_xmls(annotation, parts['noise'], parts['calibration'], burstIndex, byte_offset,
                                   orbit=orbit)
                for filepath, present, content in zip((xml_file, xml_noise_file, xml_calib_file), xml_exist, built):
                    if not present:
                        xml_contents[filepath] = content
            # the IW2 annotation, when the burst lacks it: from the manifest, which holds the IW2 product; none when the
            # manifest has none (a WARNING from iw2_annotation_xml)
            if iw2_file is not None and not iw2_exist:
                iw2_content = iw2_annotation_xml(metadata, polarization)
                if iw2_content is not None:
                    xml_contents[iw2_file] = iw2_content

            # All validations passed - write to temp files then atomic rename.
            # This guarantees no partial files on disk if interrupted mid-write.
            if tiff_bytes is not None:
                # the burst is stored as compressed NetCDF4, converted and verified in memory
                write_slc(tiff_bytes, nc_file)

            for filepath, content in xml_contents.items():
                write_file(filepath, content)

        # the catalog records of the bursts, rebuilt from a sibling polarization channel the catalog omits
        results = _asf_burst_records(bursts_missed,
                                     missing_note='NOTE: burst {burst} is missing in the ASF catalog and is not downloaded.',
                                     retries=retries, timeout_second=timeout_second)

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

        from .HTTP import download_jobs
        from .utils_pool import release_workers
        n_jobs = download_jobs(n_jobs)
        if n_jobs is None or debug == True:
            print ('Note: sequential joblib processing is applied when "n_jobs" is None or "debug" is True.')
            joblib_backend = 'sequential'

        # the orbit of every burst, kept in memory and stored in its annotation; no orbit file is written. A burst
        # whose orbit is not published is not downloaded (the WARNING is printed by EOF._orbit_records)
        orbits = []
        if results:
            from .EOF import EOF
            names = [result.geojson()['properties']['fileID'] for result in results]
            scenes = pd.DataFrame({'burst': names,
                                   'mission': ['S1' + result.geojson()['properties']['platform'][-1].upper()
                                               for result in results],
                                   'startTime': [datetime.strptime(name.split('_')[3], '%Y%m%dT%H%M%S')
                                                 for name in names]})
            orbits = EOF._orbit_records(scenes, n_jobs=n_jobs, joblib_backend=joblib_backend, retries=retries,
                                        timeout_second=timeout_second, min_rate=min_rate,
                                        min_rate_window=min_rate_window)
        no_orbit = [result.geojson()['properties']['fileID'] for result, orbit in zip(results, orbits) if orbit is None]
        tasks = [(result, orbit) for result, orbit in zip(results, orbits) if orbit is not None]

        def download_burst_with_retry(result, basedir, session, orbit, retries, timeout_second):
            # retries=0 makes one attempt (HTTP.attempts); a failure that a retry cannot change (HTTP.final) ends the
            # attempts of the burst at once
            burst_id = result.geojson()['properties']['fileID']
            n = attempts(retries)
            for retry in range(n):
                try:
                    download_burst(result, basedir, session, orbit)
                    return True
                except EmptyFileError:
                    raise
                except Exception as e:
                    stop = final(e)
                    print(f'ERROR: download attempt {retry+1} failed{" (not retried)" if stop else ""} '
                          f'for {burst_id}: {e}')
                    if stop or retry + 1 == n:
                        error = ' '.join(str(e).splitlines())
                        print(f'WARNING: {burst_id} is not stored: {error}.')
                        return False
                time.sleep(timeout_second)

        # download bursts
        with self.progressbar_joblib(tqdm(desc='Downloading ASF SLC'.ljust(25), total=len(tasks))) as progress_bar:
            statuses = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_burst_with_retry)\
                                    (result, basedir, session, orbit, retries=retries, timeout_second=timeout_second)
                                    for result, orbit in tasks)

        # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
        release_workers()
        stored = [result.geojson()['properties']['fileID'] for (result, _), status in zip(tasks, statuses) if status]
        failed = [result.geojson()['properties']['fileID'] for (result, _), status in zip(tasks, statuses) if not status]
        found = {result.geojson()['properties']['fileID'] for result in results}
        not_in_catalog = [burst for burst in bursts_missed if burst not in found]
        missed = set(not_in_catalog) | set(no_orbit) | set(failed)
        if missed:
            names = [burst for burst in bursts_missed if burst in missed]
            print(f'WARNING: {len(names)} of {len(bursts_missed)} bursts are not stored ({len(not_in_catalog)} not in the '
                  f'catalog, {len(no_orbit)} without an orbit, {len(failed)} failed to download): '
                  f'{", ".join(names)}. The next download fetches them.')
        if not stored:
            return None
        # the bursts stored by this call, which excludes any burst the catalog does not know about, without a
        # published orbit or failed
        return pd.DataFrame(stored, columns=['burst'])

    # =========================================================================
    # NISAR Helper Methods (shared between direct and cache downloads)
    # =========================================================================

    @staticmethod
    def _byte_range(url, start, end, auth_tuple, min_rate, min_rate_window, pbar=None, http_session=None,
                    signed_url=None):
        """Download byte range using single HTTP Range request.

        The request is shared by the NISAR and ALOS-2 direct downloads.
        If signed_url is provided, uses it directly (skipping OAuth).
        If pbar is provided, updates it instead of creating a new one.
        If http_session is provided, reuses the connection.

        A signed URL expires about an hour after the redirect that gave it, and the server refuses it with HTTP 403
        from then on: a request to signed_url answered 403 is made once more for the same range without it, through
        http_session (its redirect gives a new signed URL, returned) or else with auth_tuple. A 403 of that repeated
        request, or of a request made without signed_url, raises; every other error raises as it is. A response
        longer than the range (a server that ignores Range answers 200 with the whole file) raises IOError, and is
        never read beyond the range (HTTP.read_body limit). The request asks for an uncoded answer
        (Accept-Encoding: identity), so the response comes with its Content-Length and a longer one is refused before
        a byte of it is read.

        Parameters
        ----------
        url : str
            URL of the file; the request goes to it, unless signed_url is given.
        start : int
            First byte of the range.
        end : int
            Byte after the last one of the range.
        auth_tuple : tuple
            (user, password) for the basic-auth request, used when neither signed_url nor http_session is given.
        min_rate : str or float
            Minimum transfer rate (HTTP.read_body), a size string such as '100KB' or a number; a slower transfer is
            cut and retried.
        min_rate_window : float
            Seconds over which min_rate is averaged from the first byte.
        pbar : tqdm or None
            Progress bar updated with the bytes received; None shows no progress.
        http_session : requests.Session or None
            Session that reuses its connection.
        signed_url : str or None
            Signed URL the request goes to directly, without the OAuth redirect. None: the request goes to url, and
            with http_session the CloudFront URL it is redirected to is returned.

        Returns
        -------
        tuple
            (bytes, signed_url) - signed_url for reuse in subsequent requests: the one given, the one of the redirect
            of http_session, or the one of the repeated request when signed_url was refused with 403 (None when that
            request went with auth_tuple)
        """
        import requests
        from .HTTP import send, read_body, answer

        size = end - start
        # HTTP Range is inclusive; an uncoded answer carries its Content-Length (HTTP.read_body refuses a longer one)
        headers = {'Range': f'bytes={start}-{end-1}', 'Accept-Encoding': 'identity'}

        returned_signed_url = signed_url

        # a failed request raises with its response closed (HTTP.http_error), the response of any other is
        # closed by the with block
        if signed_url:
            # Use signed URL directly (much faster - no OAuth redirects); an error names the file URL
            try:
                r = send(requests.get, signed_url, what=url, headers=headers, stream=True, timeout=(10, 300))
            except (requests.HTTPError, requests.exceptions.ProxyError) as e:
                # the status as HTTP.final reads it: 403 is an expired or refused signed URL
                line = answer(e)
                if line is None or int(line.split()[0]) != 403:
                    raise
                r = None
            if r is None:
                # the same range once more without the signed URL (outside the except block: a failure of the
                # repeated request raises on its own)
                return ASF._byte_range(url, start, end, auth_tuple, min_rate, min_rate_window, pbar=pbar,
                                       http_session=http_session)
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
            # never more than the range is read: a server that ignores Range (200, the whole file) raises
            result = read_body(r, min_rate, min_rate_window, progress=pbar.update if pbar else None, limit=size)

        if len(result) < size:
            raise Exception(f'Truncated download: got {len(result)} bytes of {size} expected')
        return result, returned_signed_url

    @staticmethod
    def _cache_range(proxy, granule_id, offset, length, min_rate, min_rate_window, session=None):
        """Fetch a byte range of a granule from a cache proxy: GET <proxy>/<granule_id>/<offset>/<length>.bin.

        The request is shared by the NISAR and ALOS-2 downloads through the cache proxies (no account). An error
        raises as it is (HTTP.send); the caller retries. The request asks for an uncoded answer
        (Accept-Encoding: identity), so the response comes with its Content-Length and a longer one is refused before
        a byte of it is read. Both proxies, alos2-cache-asf.insar.dev and nisar-cache-asf.insar.dev (from its next
        deploy), name their upstream answer in the headers X-Upstream-* (status, length, range, via, host, redirected,
        ms), which the errors quote, and an upstream status other than 206 (the Range ignored, the body the whole file)
        is refused before a byte of the body is read; a proxy answer without these headers keeps the texts without the
        note.

        Parameters
        ----------
        proxy : str
            Base URL of the cache proxy, such as _NISAR_CACHE_PROXY or _ALOS2_CACHE_PROXY.
        granule_id : str
            The granule ID the proxy knows: a NISAR granule ID, or the fileID of an ALOS-2 granule.
        offset : int
            First byte of the range in the granule's file.
        length : int
            Number of bytes of the range, at most 128 MB (the proxy's limit).
        min_rate : str or float
            Minimum transfer rate (HTTP.read_body), a size string such as '100KB' or a number; a slower transfer is
            cut.
        min_rate_window : float
            Seconds over which min_rate is averaged from the first byte.
        session : requests.Session or None
            Session that reuses its connection; None makes a new one.

        Returns
        -------
        tuple
            (content, cache_hit): the bytes of the range, and whether the proxy or the CDN served them from its cache
            (the header X-Cache or cf-cache-status is HIT).

        Raises
        ------
        IOError
            When the response does not hold exactly length bytes, a truncated transfer that the caller retries; a
            response longer than length is never read beyond it (HTTP.read_body limit). Also when the proxy's
            X-Upstream-Status is not 206, before the body is read.

        Examples
        --------
        >>> content, cache_hit = ASF._cache_range(_ALOS2_CACHE_PROXY, 'ALOS2592361150-250512-WBDR1.1__A', 0, 16,
        ...                                       '100KB', 60)
        """
        import requests
        from .HTTP import send, read_body

        url = f"{proxy}/{granule_id}/{offset}/{length}.bin"
        sess = session or requests.Session()
        # a transfer slower than min_rate is cut (HTTP.read_body) and retried; an uncoded answer carries its
        # Content-Length (HTTP.read_body refuses a longer one)
        with send(sess.get, url, headers={'Accept-Encoding': 'identity'}, stream=True, timeout=120) as resp:
            # both proxies (ALOS-2 and NISAR, the latter from its next deploy) name their upstream answer
            # (X-Upstream-*); an answer without these headers keeps the texts without the note
            h = resp.headers
            upstream = h.get('X-Upstream-Status')
            note = '' if upstream is None else (
                f" (proxy upstream: status {upstream}, length {h.get('X-Upstream-Length', 'none')}, "
                f"range {h.get('X-Upstream-Range', 'none')}, via {h.get('X-Upstream-Via', 'none')}, "
                f"host {h.get('X-Upstream-Host', 'none')}, redirected {h.get('X-Upstream-Redirected', 'none')}, "
                f"{h.get('X-Upstream-Ms', 'none')} ms, cache {h.get('cf-cache-status', 'none')})")
            # an upstream answer other than 206 ignored the Range: refused before a byte of its body is read
            if upstream is not None and upstream != '206':
                raise IOError(f'Cache proxy: the upstream answered {upstream} for {length} bytes at {offset}, '
                              f'not the range{note}')
            # never more than the range is read: a longer response raises
            content = read_body(resp, min_rate, min_rate_window, limit=length)
            # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
            cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                        resp.headers.get('cf-cache-status', '').upper() == 'HIT')
        if len(content) != length:
            raise IOError(f'Cache response size mismatch: got {len(content)}, expected {length}{note}')
        return content, cache_hit

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
    def _nisar_merge_chunks_to_regions(chunks, gap_threshold=1024*1024, max_size=None):
        """Merge adjacent chunks into contiguous regions allowing small gaps.

        Parameters
        ----------
        chunks : list
            List of {'offset': int, 'size': int, ...} dicts, or (offset, size) tuples
        gap_threshold : int
            Maximum gap in bytes to merge (default 1MB)
        max_size : int or None
            The longest region in bytes: a region is closed before a chunk that would make it longer than max_size
            (from its first byte to the chunk's end), and a single chunk longer than max_size is a region of its own.
            None: no limit.

        Returns
        -------
        list of (region_start, region_size, chunk_list)
            chunk_list contains the original chunks in this region

        Examples
        --------
        >>> ASF._nisar_merge_chunks_to_regions([(0, 40), (40, 40), (80, 40)], gap_threshold=0, max_size=100)
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
            if chunk['offset'] <= region_end + gap_threshold and \
                    (max_size is None or chunk['offset'] + chunk['size'] - region_start <= max_size):
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
    def _nisar_check_bbox(bbox, mission='NISAR'):
        """Validate a NISAR crop bbox (west, south, east, north) in WGS84 degrees before any download.

        Each side must be at least _NISAR_MIN_CROP_KM long on the WGS84 ellipsoid: the south and north sides
        along their parallels, the west and east sides along the meridian. The bbox is never enlarged.

        Parameters
        ----------
        bbox : tuple
            (west, south, east, north) in WGS84 degrees.
        mission : str
            The mission name of the message; the check is the same for every mission.

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
                f"{mission} bbox {tuple(float(v) for v in bbox)} is too small to crop (too short: {', '.join(short)}). "
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

        With an account (the direct route) each granule is read through HTTP Range requests on one login session:
        1. Its size (HEAD) and its metadata, read by h5py through _NisarRemoteFile wherever the granule stores it (no
           128 MB limit): a 64 MiB head (_NISAR_HEAD_BYTES), 16 MiB pages (_NISAR_PAGE_BYTES) on demand for the rest
           of the structure (the chunk byte offsets, the geolocation grid), and the values of the copied metadata
           datasets in one batch of ranges, merged into regions of at most _NISAR_BLOCK_BYTES (gaps up to 16 MB read
           through), one Range request each. The copy omits metadata/RFI (see _read_nisar_metadata_direct). The
           metadata progress bar counts the bytes fetched.
        2. Its data in blocks of at most _NISAR_BLOCK_BYTES (64 MiB): the chunks (all of them, or those the bbox
           reaches) merged into chunk-aligned regions (ASF._nisar_merge_chunks_to_regions, gaps up to 16 MB read
           through), one Range request each; the chunks of a block are written to the output at once
           (ASF._nisar_stream_writer) and its bytes released before the next block is read.
        Every network operation of a granule is a read retried on its own (the session does not raise:
        _ASFSession.auth_with_creds falls back to basic auth), so there is no granule-level retry, which would only
        repeat a read that has used its attempts. A read that fails after its attempts raises through
        _nisar_stream_writer, which removes its .tmp file. Without an account the cache proxy is used
        (_download_nisar_via_cache).

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
        basedir : str
            Output directory; the files of each granule go into its subfolder track_frame under it.
        bbox : tuple or None
            Bounding box (west, south, east, north) in WGS84. With it only the byte ranges of the chunks overlapping
            the box are downloaded, and the file is cropped to the radar extent of the box (crop_info); None downloads
            the full frame.
        session : asf_search.ASFSession or requests.Session
            Not used by the NISAR download: each granule opens its own session with the ASF login (_get_asf_session).
            Kept for the signature shared with the Sentinel-1 download.
        n_jobs : int or None
            Without a username (the cache-proxy path) the number of parallel joblib workers for the blocks and small
            chunks, at most two per CPU (HTTP.download_jobs; negative values follow joblib: -1 all CPUs); 1 or debug downloads them sequentially. With a username
            the granules are downloaded one after another whatever the value. None is set to the NISAR default of 4
            first.
        joblib_backend : str
            Not read by the NISAR download: the cache-proxy path picks its own backend (loky, or sequential for
            n_jobs 1 or debug), and the credential path starts no joblib pool. Kept for the signature shared with the
            Sentinel-1 download.
        skip_exist : bool
            True skips the polarization files that exist (ASF._nisar_exists), and a granule whose files all exist is
            not downloaded; False downloads them again.
        retries : int
            Attempts of each read of a granule (the HEAD of its size, its metadata, each block of data), the first
            one included; 0 makes one attempt, as 1 does (HTTP.attempts). A failure that a retry cannot change
            (HTTP.final), a format error (ValueError) and an empty file (EmptyFileError) are not retried. Every failed
            attempt is printed, and the bytes it counted in the progress bar are taken back.
        timeout_second : float
            Seconds waited between the attempts of one read, not after the last one.
        min_rate : str or float
            Bytes per second each byte-range request of a granule (the metadata read and the data ranges) must keep,
            a size string such as '100KB' or a number, averaged over min_rate_window seconds from its first byte, or
            the request is retried (HTTP.read_body).
        min_rate_window : float
            Seconds over which the rate is averaged.
        debug : bool
            Print debug information of the granules; with True the granules are downloaded one after another.
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
        search_results = asf_search.granule_search(granule_ids, retries=retries, timeout_second=timeout_second)
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

        def read(what, request, pbar=None):
            """The result of request() (a network operation of a granule) with the attempts of retries
            (HTTP.attempts): a format error (ValueError) or an empty file (EmptyFileError) raises at once, a failure
            that a retry cannot change (HTTP.final) at its first attempt; every failed attempt is printed, and the
            bytes it counted in pbar are taken back. The printed and the raised text hold no URL query (_redacted)."""
            n = attempts(retries)
            for k in range(1, n + 1):
                before = None if pbar is None else pbar.n
                try:
                    return request()
                except (ValueError, EmptyFileError):
                    # format errors and empty files are permanent - don't retry
                    raise
                except Exception as e:
                    if pbar is not None:
                        # the bytes are counted while they are read: each byte is counted once
                        pbar.update(before - pbar.n)
                    stop = final(e)
                    # the text of a failed request to the signed URL without its query (its signature): _redact
                    print(f"ERROR downloading {what} (attempt {k}/{n}){' (not retried)' if stop else ''}: "
                          f"{_redact(str(e))}")
                    if stop or k == n:
                        raise _redacted(e)
                    time.sleep(timeout_second)

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

            def size():
                # Get file size (HEAD request)
                head_resp = send(http_session.head, url, allow_redirects=True, timeout=(10, 30))
                file_size = head_resp.headers.get('Content-Length')
                if file_size is None:
                    # a broken answer, retried
                    raise IOError(f'{url}: no Content-Length in the answer')
                return int(file_size)

            file_size = read(f'{granule_id} size', size)

            # the signed URL of a read is the one the next read uses (the metadata reads and the data blocks)
            signed_url = None
            # the metadata progress bar of the current stage: the bytes the metadata reads fetch (no fixed total)
            meta_bar = [None]

            def metadata_bar():
                """A progress bar of the metadata bytes fetched, current while it is open: its total grows by each
                region before it is read."""
                bar = tqdm(total=0, unit='B', unit_scale=True, desc=f"{short_name} metadata", leave=False,
                           dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0,
                           disable=(position is not None))
                meta_bar[0] = bar if position is None else None
                return bar

            def read_ranges(ranges):
                """The bytes of each (offset, length) of ranges (_NisarRemoteFile): the ranges merged into regions of
                at most _NISAR_BLOCK_BYTES (gaps up to 16 MB read through), each region read on its own (read, with its
                attempts), then each range sliced out of its region."""
                import bisect
                regions = []
                for start, length, _ in ASF._nisar_merge_chunks_to_regions(
                        [(int(o), int(n)) for o, n in ranges], 16 * 1024 * 1024, max_size=_NISAR_BLOCK_BYTES):
                    end = start + length
                    if meta_bar[0] is not None:
                        meta_bar[0].total += length
                        meta_bar[0].refresh()

                    def fetch(start=start, end=end):
                        nonlocal signed_url
                        data, signed_url = ASF._byte_range(
                            url, start, end, auth_tuple, min_rate, min_rate_window,
                            pbar=meta_bar[0], http_session=http_session, signed_url=signed_url
                        )
                        # the head of the granule is HDF5, or the answer is broken (retried)
                        if start == 0 and not data.startswith(b'\x89HDF'):
                            raise IOError(f'Server returned invalid response (not HDF5): {bytes(data[:100])!r}')
                        return data

                    regions.append((start, read(f'{granule_id} metadata bytes {start}..{end}', fetch, meta_bar[0])))
                starts = [r[0] for r in regions]
                out = []
                for offset, length in ranges:
                    start, data = regions[bisect.bisect_right(starts, offset) - 1]
                    out.append(data[offset - start:offset - start + length])
                return out

            # the metadata through a file object over the granule: its 64 MiB head, then pages on demand
            with metadata_bar():
                remote = _NisarRemoteFile(read_ranges, size=file_size)

            if debug and position is None:
                print(f"  File size: {file_size/(1024**3):.2f} GB, metadata head: "
                      f"{remote.bytes_read/(1024**2):.0f} MiB")

            # Get available polarizations from the granule's metadata
            with metadata_bar(), h5py.File(remote, 'r') as h5_meta:
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

            # the chunk indexes and the geolocation grid through the same file object
            with metadata_bar(), h5py.File(remote, 'r') as h5_meta:
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

                # Calculate bbox pixel indices if bbox provided (uses the granule's geolocationGrid)
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

            def blocks(chunk_info, freq):
                """The blocks read of one frequency: the chunks (those the bbox reaches with a bbox) merged into
                chunk-aligned regions of at most _NISAR_BLOCK_BYTES, gaps up to 16 MB read through."""
                if bbox_info is not None:
                    chunks = ASF._nisar_filter_chunks_by_pixel_range(
                        chunk_info, bbox_info['az_start'], bbox_info['az_end'],
                        bbox_info[f'rg_start_{freq.lower()}'], bbox_info[f'rg_end_{freq.lower()}']
                    )
                else:
                    chunks = chunk_info['chunks']
                return ASF._nisar_merge_chunks_to_regions(chunks, 16 * 1024 * 1024, max_size=_NISAR_BLOCK_BYTES)

            # Calculate total SLC size: the bytes of the blocks read
            total_slc_size = 0
            for pol, (chunk_info_a, chunk_info_b) in all_chunk_info.items():
                for freq, chunk_info in (('A', chunk_info_a), ('B', chunk_info_b)):
                    if chunk_info:
                        total_slc_size += sum(r[1] for r in blocks(chunk_info, freq))

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

                    # the metadata of the polarization: the values of its datasets fetched in one batch of ranges
                    with metadata_bar():
                        metadata = self._read_nisar_metadata_direct(pol, remote)

                    write_debug = debug and (position is None)
                    if write_debug:
                        n_chunks_total = sum(len(ci['chunks']) for ci in (chunk_info_a, chunk_info_b) if ci is not None)
                        print(f"    Writing {n_chunks_total} chunks + {len(metadata)-1} metadata datasets...")

                    # frequencyA, then frequencyB: each block read (its own attempts) and its chunks written at once
                    with self._nisar_stream_writer(out_path, pol, chunk_info_a, chunk_info_b, lambda: metadata,
                                                   bbox_info) as write:
                        for freq, chunk_info in (('A', chunk_info_a), ('B', chunk_info_b)):
                            if chunk_info is None:
                                continue
                            for start, length, chunks in blocks(chunk_info, freq):
                                end = start + length

                                def fetch(start=start, end=end):
                                    # the signed URL of a read is the one the next read uses
                                    nonlocal signed_url
                                    data, signed_url = ASF._byte_range(
                                        url, start, end, auth_tuple, min_rate, min_rate_window,
                                        pbar=pbar, http_session=http_session, signed_url=signed_url
                                    )
                                    return data

                                data = read(f'{granule_id} bytes {start}..{end}', fetch, pbar)
                                write((c['offset'], data[c['offset'] - start:c['offset'] - start + c['size']])
                                      for c in chunks)
                                # the block's bytes are released before the next block is read
                                del data

                    if write_debug:
                        print(f"    Done: {os.path.getsize(out_path) / (1024**3):.2f} GB written")

                    downloaded_files.append(out_name)

            return downloaded_files

        # Process granules one after another (n_jobs and debug do not change it): each read of a granule has its own
        # attempts, so a granule is not retried as a whole
        all_downloaded = []
        for granule in granules:
            files = download_nisar_granule(granule, basedir, polarizations, skip_exist, debug)
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

        The metadata of a granule is read by h5py through _NisarRemoteFile wherever the granule stores it (no 128 MB
        limit): a 64 MiB head (_NISAR_HEAD_BYTES) and 16 MiB pages (_NISAR_PAGE_BYTES) on demand for the structure,
        each by the single-block API, and the values of the copied metadata datasets in one batch: one multi-offset
        request, or as few as the proxy's limits allow (at most _NISAR_HEAD_BYTES per request; a total below 1 MiB, the
        proxy's minimum, is padded with bytes of the file around the ranges, which are discarded). Every request has the
        retries of the route. The copy omits metadata/RFI (see _read_nisar_metadata_direct).

        Parameters
        ----------
        basedir : str
            Directory of the per-polarization NISAR files, checked by skip_exist.
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
            Number of parallel granule downloads (uses loky backend), at most two per CPU this process may use (a larger
            number is capped, with a note). Negative values follow joblib: -1 all CPUs, -2 all but one, and so on; 0 raises
            ValueError.
        skip_exist : bool
            Skip the polarizations whose files already exist in basedir.
        retries : int
            Attempts of each cache request, the first one included: 0 makes one attempt. A failure that
            a retry cannot change (such as HTTP 404) is raised at once.
        timeout_second : float
            Seconds to wait before the next attempt of a failed request: a pause between
            retries, not a request timeout.
        min_rate : str or float
            Minimum transfer rate in bytes per second, a size string such as '100KB' or a
            number. A transfer slower than it is cut and retried.
        min_rate_window : float
            Seconds over which min_rate is measured.
        debug : bool
            Print the progress and each failed attempt; the download then uses the sequential backend.
        """
        # NISAR-specific default: 4 parallel jobs (optimal for Colab with decompression)
        if n_jobs is None:
            n_jobs = 4

        import h5py
        import requests
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        import os
        import time
        from joblib import Parallel, delayed
        from .HTTP import final, attempts, send, read_body, MAGIC_HDF5

        # Minimum total of a multi-range request (.../ranges.bin), the proxy's own message: "Total size 426567 below min 1048576".
        # A block below it is fetched chunk by chunk with the single-range form (.../OFFSET/LENGTH.bin): measured on 2026-10-12,
        # the proxy answers the two chunks of the failing block (228477 and 198090 bytes) with HTTP 200.
        MIN_MULTI_BLOCK = 1024 * 1024  # 1 MB

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

        def retried(what, request):
            """request() with the retries of the cache proxy: retries=0 makes one attempt (HTTP.attempts), and a
            failure that a retry cannot change (HTTP.final, such as HTTP 404) raises at its first attempt, logged as
            not retried; every failed attempt is logged."""
            n = attempts(retries)
            for retry in range(n):
                try:
                    return request()
                except Exception as e:
                    stop = final(e)
                    print(f'ERROR: {what} download attempt {retry+1}/{n} failed'
                          f'{" (not retried)" if stop else ""}: {e}')
                    if stop or retry + 1 == n:
                        raise
                    time.sleep(timeout_second)

        # The most bytes of one metadata request to the proxy (single-block or multi-offset): 64 MiB, below the proxy's
        # 128 MB top limit
        MAX_METADATA_REQUEST = _NISAR_HEAD_BYTES
        # The longest list of ranges of one multi-offset metadata request, in characters of its URL path
        MAX_METADATA_RANGES_CHARS = 8192

        def metadata_ranges(gid, file_size):
            """read_ranges of _NisarRemoteFile for granule gid: the bytes of each (offset, length) of ranges, each
            request retried on its own ('metadata'). One range: single-block requests (ASF._cache_range) of at most
            MAX_METADATA_REQUEST each; a head (offset 0) that is not HDF5 is retried, as a broken transfer. Several
            ranges: merged where they touch, cut into pieces of at most MAX_METADATA_REQUEST and packed in order into
            as few multi-offset requests (/GID/off_len,.../ranges.bin) as the limits allow (a total of at most
            MAX_METADATA_REQUEST, a list of at most MAX_METADATA_RANGES_CHARS); a request below MIN_MULTI_BLOCK is
            padded with the bytes after its last piece, up to the end of the file (file_size(), the size from the
            superblock), then before its first one, and the padding discarded."""
            import bisect

            def single(offset, length):
                def request():
                    content, _ = ASF._cache_range(_NISAR_CACHE_PROXY, gid, offset, length, min_rate, min_rate_window)
                    if offset == 0 and not content.startswith(MAGIC_HDF5):
                        raise IOError(f'Cache returned invalid metadata (not HDF5): {content[:100]!r}')
                    return content
                return retried('metadata', request)

            def multi(pieces):
                total = sum(n for _, n in pieces)
                url = f"{_NISAR_CACHE_PROXY}/{gid}/{','.join(f'{o}_{n}' for o, n in pieces)}/ranges.bin"

                def request():
                    # an uncoded answer carries its Content-Length; never more than the ranges is read
                    with send(requests.get, url, headers={'Accept-Encoding': 'identity'}, stream=True,
                              timeout=120) as resp:
                        content = read_body(resp, min_rate, min_rate_window, limit=total)
                    if len(content) != total:
                        raise IOError(f'Response size mismatch: got {len(content)}, expected {total}')
                    return content
                return retried('metadata', request)

            def read_ranges(ranges):
                if len(ranges) == 1:
                    offset, length = ranges[0]
                    return [b''.join(single(o, min(MAX_METADATA_REQUEST, offset + length - o))
                                     for o in range(offset, offset + length, MAX_METADATA_REQUEST))]
                spans = []
                for o, n in sorted(set(ranges)):
                    if spans and o <= spans[-1][1]:
                        spans[-1][1] = max(spans[-1][1], o + n)
                    else:
                        spans.append([o, o + n])
                pieces = [(o, min(MAX_METADATA_REQUEST, end - o))
                          for start, end in spans for o in range(start, end, MAX_METADATA_REQUEST)]
                groups, current, total, chars = [], [], 0, 0
                for o, n in pieces:
                    text = len(f'{o}_{n},')
                    if current and (total + n > MAX_METADATA_REQUEST or chars + text > MAX_METADATA_RANGES_CHARS):
                        groups.append(current)
                        current, total, chars = [], 0, 0
                    current.append((o, n))
                    total += n
                    chars += text
                groups.append(current)
                got = {}
                for group in groups:
                    sent = list(group)
                    need = MIN_MULTI_BLOCK - sum(n for _, n in group)
                    if need > 0:
                        o, n = sent[-1]
                        after = max(0, min(need, file_size() - (o + n)))
                        sent[-1] = (o, n + after)
                        o, n = sent[0]
                        before = max(0, min(need - after, o))
                        sent[0] = (o - before, n + before)
                    content = multi(sent)
                    # the pieces in the answer: each sent piece in order, the requested bytes inside it
                    pos = 0
                    for (o, n), (sent_o, sent_n) in zip(group, sent):
                        got[o] = content[pos + o - sent_o:pos + o - sent_o + n]
                        pos += sent_n
                data = {start: b''.join(got[o] for o in range(start, end, MAX_METADATA_REQUEST)) for start, end in spans}
                starts = [s for s, _ in spans]
                out = []
                for offset, length in ranges:
                    start = starts[bisect.bisect_right(starts, offset) - 1]
                    out.append(data[start][offset - start:offset - start + length])
                return out
            return read_ranges

        # Process granules (the bbox is validated by download())
        granule_ids = [g.replace('.h5', '') for g in granules]

        print(f"Downloading {len(granule_ids)} NISAR granule(s) via cache proxy (aligned blocks)...")

        # Unified download path - only backend differs for debug/sequential mode
        import joblib
        backend = 'sequential' if (n_jobs == 1 or debug) else 'loky'
        from .HTTP import download_jobs
        from .utils_pool import release_workers
        effective_n_jobs = 1 if backend == 'sequential' else download_jobs(n_jobs)

        def download_aligned_block(gid, offsets_str, total_size):
            """Download aligned 25x25km block via multi-offset API."""
            import requests
            url = f"{_NISAR_CACHE_PROXY}/{gid}/{offsets_str}/ranges.bin"
            def request():
                # a transfer slower than min_rate is cut (HTTP.read_body) and retried; an uncoded answer carries
                # its Content-Length (HTTP.read_body refuses a longer one)
                with send(requests.get, url, headers={'Accept-Encoding': 'identity'}, stream=True,
                          timeout=120) as resp:
                    # never more than the ranges is read: a longer response raises
                    content = read_body(resp, min_rate, min_rate_window, limit=total_size)
                    # Check both X-Cache (proxy) and cf-cache-status (CDN) headers
                    cache_hit = (resp.headers.get('X-Cache', '').upper() == 'HIT' or
                                resp.headers.get('cf-cache-status', '').upper() == 'HIT')
                if len(content) != total_size:
                    raise ValueError(
                        f"Response size mismatch: got {len(content)}, expected {total_size}")
                return (offsets_str, content, cache_hit)
            return retried('block', request)

        def download_single_chunk(gid, offset, length):
            """Download single chunk via single-block API (for small blocks); an answer that is not exactly the
            chunk raises IOError (ASF._cache_range), retried."""
            def request():
                content, _ = ASF._cache_range(_NISAR_CACHE_PROXY, gid, offset, length, min_rate, min_rate_window)
                return (offset, length, content)
            return retried('single chunk', request)

        all_downloaded = []

        for gid in granule_ids:

            # 1. The metadata of this granule through a file object over it: its 64 MiB head, then pages on demand
            if debug:
                print(f"Fetching metadata for {gid}...")
            remote = _NisarRemoteFile(metadata_ranges(gid, lambda: remote.size))
            if debug:
                print(f"  Metadata head: {remote.bytes_read/(1024**2):.1f}MB")

            with h5py.File(remote, 'r') as h5:
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
                            if MIN_MULTI_BLOCK <= block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])

                    if ci_b:
                        rg_start = bbox_info.get('rg_chunk_start_b', 0)
                        rg_end = bbox_info.get('rg_chunk_end_b', ci_b['n_rg'])
                        for block in chunks_to_bbox_blocks(ci_b, az_start, az_end, rg_start, rg_end):
                            if MIN_MULTI_BLOCK <= block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])
                else:
                    # Full download: use cache-aligned blocks
                    if ci_a:
                        for block in chunks_to_aligned_blocks(ci_a):
                            if MIN_MULTI_BLOCK <= block['total_size'] <= MAX_MULTI_BLOCK:
                                offsets_str = ','.join(f"{off}_{size}" for off, size in block['chunks'])
                                blocks_to_download.append((offsets_str, block['total_size'], block['chunks']))
                            else:
                                small_chunks_to_download.extend(block['chunks'])

                    if ci_b:
                        for block in chunks_to_aligned_blocks(ci_b):
                            if MIN_MULTI_BLOCK <= block['total_size'] <= MAX_MULTI_BLOCK:
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
                cache_hits = 0
                cache_misses = 0
                out_name = f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}.h5"
                out_path = os.path.join(out_dir, out_name)
                # The chunks are written to the output at its temporary name as the blocks arrive, and each block's bytes are
                # released after its chunks are written; the output is renamed to its final name when complete.
                with tqdm(desc=f"NSR_{track:03d}_{frame:03d}_{datetime_str}_{pol}",
                          total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                          smoothing=0) as pbar, \
                        self._nisar_stream_writer(out_path, pol, ci_a, ci_b,
                                                  lambda: self._read_nisar_metadata_direct(pol, remote), bbox_info) as write:

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
                            pairs = []
                            pos = 0
                            for chunk_off, chunk_size in chunks:
                                pairs.append((chunk_off, data[pos:pos + chunk_size]))
                                pos += chunk_size
                            write(pairs)
                            del data, pairs

                    # Download small chunks via single-block API
                    if small_chunks_to_download:
                        results = joblib.Parallel(n_jobs=effective_n_jobs, backend=backend, return_as='generator')(
                            joblib.delayed(download_single_chunk)(gid, off, size)
                            for off, size in small_chunks_to_download
                        )
                        for off, size, data in results:
                            pbar.update(size)
                            write([(off, data)])
                            del data

                    # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
                    release_workers()
                all_downloaded.append(out_name)

            # the granule's metadata file object is released after all pols
            del remote

        if all_downloaded:
            return pd.DataFrame({'file': all_downloaded})
        return None

    @staticmethod
    def _crop_window(bbox, orbit, image, sampling_rate, prefix, prefix_length, label, doppler=None):
        """Internal: the line window of an ALOS-2 subswath or an ALOS-1 image that a cropped download stores: the lines
        that the transform's UTM grid of the bbox reaches at any height from -500 to 9000 m.

        The grid is utils_grid.transform_region(bbox, zones) (the grid rectangles of the bbox plus 100 m, in lon/lat),
        mapped by utils_alos2_orbit.bbox_window with the image's Doppler centroid, and the window is rounded outward to
        multiples of _ALOS2_LINE_BLOCK lines (512), ending at the image's last line at most. The zones: first the UTM zone
        of the bbox centre; the transform takes the zone of the centroid of the stored footprint, so the prefixes of the
        window's first and last lines are read (JAXA's lon/lat of their first, mid and last pixel) and, when their span
        reaches another UTM zone or hemisphere, the window is computed again over the grids of all those zones (once).

        Parameters
        ----------
        bbox : tuple
            (west, south, east, north) in WGS84 degrees.
        orbit : dict
            The LED orbit (utils_alos2_orbit.led_orbit).
        image : dict
            The image (utils_alos2_ceos.image_info).
        sampling_rate : float
            The range sampling rate in Hz (utils_alos2_orbit.led_sampling_rate).
        prefix : callable
            prefix(line) returns the prefix bytes of the 0-based image line (one read).
        prefix_length : int
            The length of the line prefix: utils_alos2_ceos.PREFIX (544) for ALOS-2, PREFIX_ALOS1 (412) for ALOS-1.
        label : str
            The mission and the ID, for the error message ('ALOS-2 <subswath ID>').
        doppler : tuple of float or None, optional
            (constant, slope, wavelength) of the image's Doppler centroid (utils_alos2_orbit.geo2rdr). None (default):
            zero Doppler (ALOS-2 L1.1).

        Returns
        -------
        tuple
            (window, zones): window (start, end), 0-based lines, end exclusive, or None when the bbox reaches the image
            at no height; zones the set of the EPSG codes of the grid.

        Raises
        ------
        RuntimeError
            The lines of the second window reach a zone outside its grid.

        Examples
        --------
        >>> ASF._crop_window((-72.9, -41.39, -72.5, -41.19), orbit, image, 32e6, prefix, 544, 'ALOS-2 <ID>')
        """
        import numpy as np
        from .utils_alos2_orbit import bbox_window
        from .utils_alos2_ceos import prefix_lonlat
        from .utils_grid import get_utm_epsg, transform_region

        def span_zones(window):
            """The UTM EPSG codes of every zone and hemisphere between the minimum and maximum of JAXA's lon/lat of the
            first, mid and last pixel of the window's first and last lines (one prefix read each): the span of the stored
            footprint, which holds its centroid.

            Examples
            --------
            >>> span_zones((6144, 22016))
            """
            lon, lat = [], []
            for line in (window[0], window[1] - 1):
                line_lon, line_lat = prefix_lonlat(prefix(line), prefix_length)
                lon.extend(line_lon)
                lat.extend(line_lat)
            # a longitude every 6 degrees from the west end, and the east end: one in every zone between
            lons = np.r_[np.arange(min(lon), max(lon), 6.0), max(lon)]
            return {get_utm_epsg(y, x) for y in (min(lat), max(lat)) for x in lons}

        # the transform's UTM grid of the bbox: first in the zone of the bbox centre, then (once) in every zone the stored
        # lines span
        epsgs = {get_utm_epsg(0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[0] + bbox[2]))}
        for attempt in (1, 2):
            window = bbox_window(orbit, image, sampling_rate, transform_region(tuple(bbox), epsgs), doppler=doppler)
            if window is None:
                break
            # rounded outward on the image's grid of _ALOS2_LINE_BLOCK lines
            window = ((window[0] // _ALOS2_LINE_BLOCK) * _ALOS2_LINE_BLOCK,
                      min(image['lines'], -(-window[1] // _ALOS2_LINE_BLOCK) * _ALOS2_LINE_BLOCK))
            zones = span_zones(window)
            if zones <= epsgs:
                break
            if attempt == 2:
                raise RuntimeError(f"{label}: lines {window[0]}..{window[1]} reach UTM {sorted(zones - epsgs)} outside "
                                   f"the grid of {sorted(epsgs)}")
            epsgs |= zones
        return window, epsgs

    def _alos_zip(self, fileID, url, size, label, retries, timeout_second, min_rate, min_rate_window, debug):
        """Internal: the byte-range access to one product ZIP at ASF, shared by the ALOS-2 and the ALOS-1 downloads
        (ASF._download_alos2, ASF._download_alos1): the reads with their attempts, the ZIP directory and the offset of a
        member's data.

        With an account every read is a byte range of ASF._byte_range on one login session, the first request following
        the login redirect and the next ones going to its signed URL. Without one (ALOS-2 only) every read is
        ASF._cache_range of the fileID on _ALOS2_CACHE_PROXY, on one requests.Session (no Earthdata session, no signed
        URL). The ZIP directory is read through byte ranges (zipfile over _RangeReader).

        Parameters
        ----------
        fileID : str
            The product's fileID: the name of its reads in the messages, and the ID the cache proxy knows.
        url : str
            The catalog URL of the product's ZIP.
        size : int or None
            The ZIP's size in bytes. None (with an account only): the size the server reports, the Content-Range of the
            one-byte range 0-0
            (the catalog size can differ from it, ASF._download_alos1), with the attempts of retries; its one-byte body
            is read, so the connection goes back to the pool for the next reads, and the signed URL of its redirect is
            kept for them.
        label : str
            'ALOS-2' or 'ALOS-1': the mission in the messages.
        retries, timeout_second, min_rate, min_rate_window, debug
            As ASF._download_alos2.

        Returns
        -------
        tuple
            (read, infos, data_offset): read(start, end, pbar=None) the bytes start..end (end exclusive) of the ZIP with
            the attempts of retries (HTTP.attempts): a format error (ValueError) or an empty file (EmptyFileError)
            raises at once, a failure that a retry cannot change (HTTP.final) at its first attempt; every failed
            attempt is printed, the bytes it counted in pbar taken back, and the query of a URL in its text removed
            (_redacted: a signed URL's query is its signature), also in the error raised; infos the members
            (zipfile.ZipFile.infolist); data_offset(member) the offset of a member's data, after its local header.

        Examples
        --------
        >>> read, infos, data_offset = ASF('user', 'pass')._alos_zip('ALPSRP055006360-L1.0', url, None, 'ALOS-1', 30, 3,
        ...                                                          '100KB', 60, False)
        """
        import os
        import time
        import zipfile
        from .HTTP import final, attempts, send
        from .utils_files import EmptyFileError

        def attempt(what, request, pbar=None):
            """request() with the attempts of retries; every failed one printed as 'ERROR <what> ...'."""
            n = attempts(retries)
            for k in range(1, n + 1):
                before = None if pbar is None else pbar.n
                try:
                    return request()
                except (ValueError, EmptyFileError):
                    # format errors and empty files are permanent - don't retry
                    raise
                except Exception as e:
                    if pbar is not None:
                        # the direct route counts the bytes while it reads: each byte is counted once
                        pbar.update(before - pbar.n)
                    stop = final(e)
                    print(f"ERROR {what} (attempt {k}/{n}){' (not retried)' if stop else ''}: {_redact(str(e))}")
                    if stop or k == n:
                        raise _redacted(e)
                    time.sleep(timeout_second)

        signed = [None]
        if self.username is None:
            # through the cache proxy: no Earthdata session, no signed URL
            if debug:
                print(f"{label} URL: {_ALOS2_CACHE_PROXY}/{fileID}")
            session = requests.Session()

            def fetch(start, end, pbar=None):
                data, _ = ASF._cache_range(_ALOS2_CACHE_PROXY, fileID, start, end - start, min_rate, min_rate_window,
                                           session)
                if pbar is not None:
                    pbar.update(len(data))
                return data
        else:
            if debug:
                print(f"{label} URL: {url}")
            session = self._get_asf_session()

            def fetch(start, end, pbar=None):
                # the first request follows the login redirect, the next ones go to its signed URL
                data, signed[0] = ASF._byte_range(url, start, end, None, min_rate, min_rate_window, pbar=pbar,
                                                  http_session=session, signed_url=signed[0])
                return data

        def read(start, end, pbar=None):
            return attempt(f'downloading {fileID} bytes {start}..{end}', lambda: fetch(start, end, pbar), pbar)

        def remote_size():
            with send(session.get, url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'}, stream=True,
                      timeout=(10, 300)) as r:
                if 'cloudfront.net' in r.url:
                    signed[0] = r.url
                m = re.match(r'^bytes 0-0/(\d+)$', r.headers.get('Content-Range', ''))
                if m is None:
                    raise ValueError(f"{label} product {fileID}: no file size in the answer to a byte range "
                                     f"(HTTP {r.status_code}, Content-Range {r.headers.get('Content-Range')!r})")
                # the one byte of the range 0-0, read only after its Content-Range is checked
                r.content
                return int(m.group(1))

        if size is None:
            size = attempt(f'reading the size of {fileID}', remote_size)
        reader = _RangeReader(read, size)
        try:
            with zipfile.ZipFile(reader) as archive:
                infos = archive.infolist()
        except zipfile.BadZipFile as e:
            # zipfile reports a failed read as 'File is not a zip file': the read's own error is raised (after its
            # attempts)
            if reader.error is not None:
                raise reader.error from e
            raise

        def data_offset(member):
            # the data follows the local header
            reader.seek(member.header_offset)
            header = reader.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError(f"{label} member {os.path.basename(member.filename)}: no local header at "
                                 f"{member.header_offset}")
            name_length = int.from_bytes(header[26:28], 'little')
            extra_length = int.from_bytes(header[28:30], 'little')
            return member.header_offset + 30 + name_length + extra_length

        return read, infos, data_offset

    def _download_alos2(self, basedir, ids, polarizations, bbox, n_jobs, skip_exist, retries, timeout_second, min_rate,
                        min_rate_window, debug):
        """Internal: download ALOS-2 ScanSAR subswaths from ASF, cropped to a bbox: directly with an Earthdata
        account, through the cache proxy _ALOS2_CACHE_PROXY without one (the account is optional, as for NISAR).

        A granule at ASF is one stored (uncompressed) ZIP archive. Its directory is read through byte ranges
        (zipfile over _RangeReader), then only the leader file (LED) and the line records of the subswaths' image
        files (IMG) that the bbox reaches are read. With an account every read is a byte range of ASF._byte_range on
        one login session per granule, reusing the signed URL of the first redirect; without one every read is
        ASF._cache_range of the granule's fileID on _ALOS2_CACHE_PROXY, on one requests.Session per granule (no
        Earthdata session, no signed URL). Each image file is streamed to disk in blocks of at most
        _ALOS2_BLOCK_BYTES (64 MiB), below the proxy's 128 MB (utils_alos2_ceos.write_crop: a temporary file renamed
        when complete). Each byte-range read has its own attempts: a failed block is read again, not the whole file.
        The catalog search for the size of each granule is the public one in both cases.

        Parameters
        ----------
        basedir : str
            Output directory; the files of a subswath go into its folder <path>_<frame>_F<n> under it.
        ids : list
            ALOS-2 subswath IDs (ASF._alos2_parse), such as 'ALOS2592361150-250512-WBDR1.1__A-F2'.
        polarizations : list or None
            The requested polarizations; those the product lacks are skipped (download() names them in a WARNING).
            None: every one the product has (HH and HV of an HH+HV product, HH of an HH product).
        bbox : tuple or None
            (west, south, east, north) in WGS84 degrees: only the lines of a subswath that the transform's UTM grid
            of the bbox reaches at any height from -500 to 9000 m are stored (utils_alos2_orbit.bbox_window over
            utils_grid.transform_region: the grid rectangles of the bbox plus 100 m, in lon/lat), rounded outward to
            multiples of _ALOS2_LINE_BLOCK lines (512) and ending at the image's last line at most. The grid is first
            taken in the UTM zone of the bbox centre; the transform takes the zone of the centroid of the stored
            footprint, so the prefixes of the window's first and last lines are read (JAXA's lon/lat of their first,
            mid and last pixel) and, when their span reaches another UTM zone or hemisphere, the window is computed
            again over the grids of all those zones (once; a zone still outside raises RuntimeError): ASF._crop_window,
            zero Doppler. A subswath the bbox does not reach gets a WARNING and nothing is stored. None stores the whole
            subswath.
        n_jobs : int or None
            Not used: the subswaths are downloaded one after another. Kept for the signature shared with the other
            downloads.
        skip_exist : bool
            True skips the image files that exist; False downloads them again.
        retries : int
            Attempts of the catalog search and of each byte-range read (the ZIP directory, the LED, the image heads,
            each block of image records), the first one included; 0 makes one attempt (HTTP.attempts). A failure that
            a retry cannot change (HTTP.final), a format error (ValueError) and an empty file (EmptyFileError) are not
            retried. Every failed attempt is printed.
        timeout_second : float
            Seconds waited between the attempts.
        min_rate : str or float
            Bytes per second each byte-range request must keep (HTTP.read_body).
        min_rate_window : float
            Seconds over which the rate is averaged.
        debug : bool
            Print the URLs and the line windows.

        Returns
        -------
        pandas.DataFrame or None
            The column 'file': the names of the image files written, or None when none is written.

        Examples
        --------
        >>> # without an account, through the cache proxy
        >>> ASF()._download_alos2('data', ['ALOS2100352900-160401-WBDR1.1__D-F3'], None, None, None, True, 30, 3, '100KB',
        ...                       60, False)
        >>> asf = ASF('user', 'pass')
        >>> asf._download_alos2('data', ['ALOS2100352900-160401-WBDR1.1__D-F3'], None, None, None, True, 30, 3, '100KB', 60,
        ...                     False)
        """
        import os
        import pandas as pd
        from tqdm.auto import tqdm
        from .utils_files import exists
        from .utils_alos2_orbit import led_orbit, led_sampling_rate
        from .utils_alos2_ceos import image_info, write_crop, DESCRIPTOR, PREFIX

        # the subswaths of each granule, in the order of the IDs
        granules = {}
        for subswath_id in dict.fromkeys(ids):
            info = ASF._alos2_parse(subswath_id)
            granules.setdefault(info['fileID'], []).append((subswath_id, info))

        print(f"Connecting to ASF...", end='\r', flush=True)
        search_results = asf_search.granule_search(list(granules), retries=retries, timeout_second=timeout_second)
        print(f"Searching ASF for {len(granules)} granule(s)... done ({len(search_results)} found)")
        catalog = {}
        for result in search_results:
            props = result.geojson()['properties']
            catalog.setdefault(props['fileID'], (props['url'], int(props['bytes'])))
        missing = [g for g in granules if g not in catalog]
        if missing:
            raise ValueError(f"Granules not found in ASF: {missing}")

        def download_granule(fileID, subswaths, written):
            """One granule, with no retry of its own: every network operation of a granule is a byte-range read
            with its own attempts (the ZIP directory, the LED, the image heads, the blocks; the login session does
            not raise, _ASFSession.auth_with_creds falls back to basic auth), so a granule-level retry would only
            repeat a read that has used its attempts. A read that fails after its attempts raises (write_crop
            removes its .tmp)."""
            url, size = catalog[fileID]
            # the reads with their attempts, the ZIP directory, the members' data (shared with ALOS-1)
            read, infos, data_offset = self._alos_zip(fileID, url, size, 'ALOS-2', retries, timeout_second, min_rate,
                                                      min_rate_window, debug)
            members = {os.path.basename(m.filename): m for m in infos}

            def member_data(name):
                # (offset of the member's data, its size)
                member = members.get(name)
                if member is None:
                    raise ValueError(f"ALOS-2 granule {fileID} has no {name}")
                if member.compress_type != 0:
                    raise ValueError(f"ALOS-2 member {name} is compressed (method {member.compress_type}), not stored")
                return data_offset(member), member.file_size

            led_offset, led_size = member_data(f'LED-{fileID}')
            led = read(led_offset, led_offset + led_size)
            if bbox is not None:
                orbit = led_orbit(led)
                sampling_rate = led_sampling_rate(led)

            for subswath_id, info in subswaths:
                available = ASF._alos2_pols(None, info)
                pols = available if polarizations is None else [p for p in polarizations if p in available]
                if not pols:
                    continue
                img = {p: member_data(f"IMG-{p}-{fileID}-F{info['subswath']}") for p in pols}
                head = read(img[pols[0]][0], img[pols[0]][0] + 1264)
                image = image_info(head)
                if bbox is not None:

                    def prefix(line):
                        # the 544-byte prefix of an image line of the first polarization (one byte-range read)
                        first = img[pols[0]][0] + DESCRIPTOR + line * image['record_length']
                        return read(first, first + PREFIX)

                    window, epsgs = ASF._crop_window(bbox, orbit, image, sampling_rate, prefix, PREFIX,
                                                     f'ALOS-2 {subswath_id}')
                    if window is None:
                        print(f"WARNING: bbox {tuple(bbox)} does not reach {subswath_id} at heights -500 to 9000 m: "
                              f"nothing stored")
                        continue
                    if debug:
                        print(f"{subswath_id}: the grid of the bbox in EPSG {sorted(epsgs)}")
                else:
                    window = (0, image['lines'])
                start, end = window
                record_length = image['record_length']
                if debug:
                    print(f"{subswath_id}: lines {start}..{end} of {image['lines']}, {record_length} bytes each")

                led_path = ASF._alos2_path(basedir, info)
                os.makedirs(os.path.dirname(led_path), exist_ok=True)
                if not exists(led_path):
                    with open(led_path + '.tmp', 'wb') as f:
                        f.write(led)
                    os.replace(led_path + '.tmp', led_path)

                todo = [p for p in pols if not (skip_exist and exists(ASF._alos2_path(basedir, info, p)))]
                desc = os.path.basename(led_path)[:-len('.led')]
                with tqdm(total=len(todo) * (end - start) * record_length, unit='B', unit_scale=True, desc=desc,
                          leave=True, dynamic_ncols=False, ncols=80, mininterval=0.3, smoothing=0) as pbar:
                    for p in todo:
                        offset, member_size = img[p]
                        own = head if p == pols[0] else read(offset, offset + 1264)
                        own_image = image_info(own)
                        if any(own_image[k] != image[k] for k in ('lines', 'record_length', 'pixels')):
                            raise ValueError(f"ALOS-2 IMG-{p} of {subswath_id} has another size than IMG-{pols[0]}")
                        if member_size != 720 + image['lines'] * record_length:
                            raise ValueError(f"ALOS-2 IMG-{p} of {subswath_id} has {member_size} bytes, expected "
                                             f"{720 + image['lines'] * record_length}")
                        first = offset + 720
                        step = max(1, _ALOS2_BLOCK_BYTES // record_length)
                        # one block of records in memory at a time
                        blocks = (read(first + a * record_length, first + min(end, a + step) * record_length, pbar)
                                  for a in range(start, end, step))
                        img_path = ASF._alos2_path(basedir, info, p)
                        write_crop(img_path, own[:720], blocks, end - start, record_length)
                        name = os.path.basename(img_path)
                        if name not in written:
                            written.append(name)

        # the subswaths one after another (each read with its own attempts)
        written = []
        for fileID, subswaths in granules.items():
            download_granule(fileID, subswaths, written)

        if written:
            return pd.DataFrame({'file': written})
        return None

    def _download_alos1(self, basedir, ids, polarizations, bbox, n_jobs, skip_exist, retries, timeout_second, min_rate,
                        min_rate_window, debug, catalog=None):
        """Internal: download ALOS-1 PALSAR L1.1 (SLC) and L1.0 (raw signal data) products directly from ASF with an
        Earthdata account, cropped to a bbox (download() refuses ASF() without an account for ALOS-1: there is no cache
        proxy).

        A product at ASF is one ZIP archive. Its size is the one the server reports (the Content-Range of a one-byte
        range), never the catalog size, which can differ from it: for 79 of the 109 L1.0 products of orbit 5500 (-14
        to +11 bytes) and for none of 232 L1.1 products (measured 2026-11-23). Its directory is read through byte ranges
        (zipfile over _RangeReader), then only the leader file (LED) and the line records of the image files (IMG, one
        per polarization) that the bbox reaches are read, as for ALOS-2 (ASF._alos_zip, shared with
        ASF._download_alos2): every read is a byte range of ASF._byte_range on one login session per product, reusing
        the signed URL of the first redirect. The members are found by their names
        (LED-<scene>-<code>, IMG-<pol>-<scene>-<code>, in a folder or not), each by its own ZIP method: a stored member
        (0) is read by byte ranges of the window's lines; a deflate-compressed member (8) is read from its start and
        decompressed as it arrives (raw deflate) up to the window's last line, nothing after it, the lines before the
        window dropped, at most _ALOS2_BLOCK_BYTES of compressed and of decompressed data in memory at a time. The LED is
        read whole either way; another method raises a one-line ValueError. The off-nadir angle of the file names is the nominal off-nadir angle of the data set
        summary record of the LED (bytes 1839-1854), the date the date of the first image line. Each image file is
        streamed to disk in blocks of at most _ALOS2_BLOCK_BYTES (64 MiB) (utils_alos2_ceos.write_crop: a temporary file
        renamed when complete). The image line prefix of ALOS-1 is 412 bytes (utils_alos2_ceos.PREFIX_ALOS1).

        The L1.1 (SLC) products of the modes FBS, FBD and PLR are read and written the same way: the image channels of
        the data set summary record (bytes 389-392) are FBS 1, FBD 2, PLR 4 (measured on L1.1 and L1.0 leader files),
        and the record length of each image file is its own (its file descriptor), so the twice as many range samples
        of an FBS image (32 MHz sampling, FBD 16 MHz) change nothing in the crop.

        An L1.0 (raw) product is stored as its leader file and the whole line records (412-byte line header, 8-bit I, Q
        echo samples, fill) of the window, as they are in the source (only the record sequence numbers renumbered, as
        for L1.1; no missing-line repair): ASF._alos1_path, `<path>_<frame>_<offnadir>_raw/...<pol>.raw`. Its members
        are deflate-compressed (measured on 21 products and the 109 of orbit 5500), read as above; the leader file is
        stored with the first image file, once its lines are found. A raw download reads no SLC (L1.1) product.

        Parameters
        ----------
        basedir : str
            Output directory; the files of a product go into its folder <path>_<frame>_<offnadir> under it.
        ids : list
            ALOS-1 product IDs (ASF._alos1_parse), such as 'ALPSRP131296360-L1.1' or 'ALPSRP055006360-L1.0'.
        polarizations : list or None
            The requested polarizations; those the product lacks get a WARNING and are skipped (ASF._alos1_pols).
            None: every one the product has.
        bbox : tuple or None
            (west, south, east, north) in WGS84 degrees: only the lines of the image that the transform's UTM grid of
            the bbox reaches at any height from -500 to 9000 m are stored, rounded outward to multiples of
            _ALOS2_LINE_BLOCK lines (512) and ending at the image's last line at most: the rule of ALOS-2
            (ASF._crop_window), with the image's Doppler centroid of the LED (ALOS-1 L1.1 is not zero-Doppler) and the
            412-byte prefixes of the window's first and last lines (of a compressed member: one decompression stream
            from its start). A raw (L1.0) frame keeps the lines whose own time tags (the line headers) fall in the time
            span of utils_alos2_orbit.raw_window: the beam-centre times of the transform's region
            (utils_grid.transform_region of the bbox in every UTM zone of the frame's longitude span, the catalog
            footprint: the raw line prefixes hold no ground coordinates) at the beam-centre Doppler of the raw leader
            file's own yaw and orbit (utils_alos2_orbit.raw_doppler: the raw leader files hold no Doppler centroid),
            widened on each side by half the synthetic aperture and by the margin of the frame's steering state, from
            JAXA's yaw steering mode flag of the leader file (utils_alos2_orbit.raw_margin: ALOS1_DOPPLER_MARGIN steered,
            ALOS1_DOPPLER_MARGIN_UNSTEERED not); a raw leader file without a readable attitude data record or with
            another flag value raises a one-line ValueError naming the product. A span
            outside the frame's own time span (from its first line: its time, and its lines at the line rate of line 1
            plus utils_alos2_orbit.ALOS1_FRAME_END_SLACK) gets a WARNING before the member is read. The lines are
            selected while the first image streams from its first line (utils_alos2_ceos.select_lines), rounded outward
            to _ALOS2_LINE_BLOCK lines and clipped to the frame, so a PRF change inside the frame moves no line; the
            other polarizations take the same lines, and so does a resumed download: the lines of an image of the
            product already stored (utils_alos2_ceos.crop_lines). A product it does not reach gets a WARNING and
            nothing is stored (no folder is created). None stores the whole image.
        n_jobs : int or None
            Not used: the products are downloaded one after another. Kept for the signature shared with the other
            downloads.
        skip_exist : bool
            True skips the image files that exist; False downloads them again.
        retries : int
            Attempts of the catalog search and of each byte-range read (the ZIP directory, the LED, the image heads,
            each block of image records), the first one included; 0 makes one attempt (HTTP.attempts). A failure that
            a retry cannot change (HTTP.final), a format error (ValueError) and an empty file (EmptyFileError) are not
            retried. Every failed attempt is printed.
        timeout_second : float
            Seconds waited between the attempts.
        min_rate : str or float
            Bytes per second each byte-range request must keep (HTTP.read_body).
        min_rate_window : float
            Seconds over which the rate is averaged.
        debug : bool
            Print the URLs and the line windows.
        catalog : dict or None
            The catalog entries of the products as ASF._alos1_catalog gives them (ASF.download looks them up before
            the download of any mission). None: looked up here (ASF._alos1_catalog).

        Returns
        -------
        pandas.DataFrame or None
            The column 'file': the names of the image files written, or None when none is written.

        Examples
        --------
        >>> asf = ASF('user', 'pass')
        >>> asf._download_alos1('data', ['ALPSRP131296360-L1.1'], None, None, None, True, 30, 3, '100KB', 60, False)
        >>> asf._download_alos1('data', ['ALPSRP055006360-L1.0'], ['HH'], (-72.27, -40.595, -72.03, -40.405), None, True, 30, 3, '100KB', 60, False)
        """
        import os
        import re
        import zlib
        import zipfile
        import numpy as np
        import pandas as pd
        import shapely.geometry
        from tqdm.auto import tqdm
        from .utils_files import exists
        from itertools import chain
        from .utils_alos2_orbit import (led_orbit, led_sampling_rate, led_attitude, raw_doppler, raw_margin, raw_window,
                                        _record, _field, ALOS1_FRAME_END_SLACK)
        from .utils_alos2_ceos import image_info, write_crop, select_lines, crop_lines, DESCRIPTOR, PREFIX_ALOS1
        from .utils_grid import get_utm_epsg, transform_region

        products = {i: ASF._alos1_parse(i) for i in dict.fromkeys(ids)}

        # the catalog entries (a product missing from the catalog raises there, before any product is downloaded)
        if catalog is None:
            catalog = ASF._alos1_catalog(list(products), retries, timeout_second)

        def download_product(fileID, info, written):
            """One product, with no retry of its own: every network operation is a byte-range read with its own
            attempts (as ASF._download_alos2). A read that fails after its attempts raises (write_crop removes its
            .tmp)."""
            url, footprint, _ = catalog[fileID]
            raw = info['level'] == 'L1.0'
            # the reads with their attempts, the ZIP directory (its size the one the server reports), the members' data
            # (shared with ALOS-2)
            read, infos, data_offset = self._alos_zip(fileID, url, None, 'ALOS-1', retries, timeout_second, min_rate,
                                                      min_rate_window, debug)

            # the LED and the image files of the product, by their names (in a folder or not)
            scene = info['scene']
            members = {}
            # the code of the level in the member names: H1.1__A, P1.1__D (L1.1), H1.0__A, P1.0__A (L1.0)
            digit = '0' if raw else '1'
            for member in infos:
                m = re.match(rf'^(LED|IMG-(HH|HV|VH|VV))-{scene}-[A-Z]1\.{digit}__[AD]$', os.path.basename(member.filename))
                if m is not None:
                    members[m.group(2) or 'LED'] = member
            if 'LED' not in members:
                raise ValueError(f"ALOS-1 product {fileID} has no LED-{scene}-* member")
            available = [p for p in ('HH', 'HV', 'VH', 'VV') if p in members]
            if not available:
                raise ValueError(f"ALOS-1 product {fileID} has no IMG-*-{scene}-* member")

            def member_data(key):
                # (offset of the member's data, the member): stored (method 0) or deflate-compressed (method 8) members
                member = members[key]
                if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError(f"ALOS-1 member {os.path.basename(member.filename)} of {fileID} has the compression "
                                     f"method {member.compress_type}, neither stored (0) nor deflate (8)")
                return data_offset(member), member

            def inflate(offset, member, first, last, chunk):
                """The bytes first..last (end exclusive) of a deflate-compressed member, decompressed as its data
                arrives (raw deflate): the compressed data is read from its start in reads of at most `chunk` bytes
                until the byte `last` is decompressed, and nothing after it; every piece holds at most
                _ALOS2_BLOCK_BYTES decompressed bytes (bounded memory, never the whole member). The member ends early
                only when no compressed data is left and the decompressor gives no more output: a call that reaches the
                output limit can hold back output with all its input consumed, which the next call gives. Yields the
                pieces."""
                name = os.path.basename(member.filename)
                inflater = zlib.decompressobj(-15)
                position, stop, at, pending = offset, offset + member.compress_size, 0, b''
                while at < last:
                    if not pending and position < stop and not inflater.eof:
                        pending = read(position, min(stop, position + chunk))
                        position += len(pending)
                    out = inflater.decompress(pending, _ALOS2_BLOCK_BYTES)
                    pending = inflater.unconsumed_tail
                    if not out and not pending and (position >= stop or inflater.eof):
                        raise ValueError(f"ALOS-1 member {name} of {fileID} ends at {at} bytes, before byte {last}")
                    lo, hi = max(at, first), min(at + len(out), last)
                    if lo < hi:
                        yield out[lo - at:hi - at]
                    at += len(out)

            def member_bytes(key, first, last, chunk=_ALOS2_BLOCK_BYTES):
                """The bytes first..last of a member, stored or compressed, in one piece (the LED, the image heads)."""
                offset, member = member_data(key)
                if member.compress_type == zipfile.ZIP_STORED:
                    return read(offset + first, offset + last)
                return b''.join(inflate(offset, member, first, last, chunk))

            def records(key, start, end, record_length, pbar):
                """The image records start..end (0-based lines, end exclusive) of a member in blocks of whole records,
                each at most _ALOS2_BLOCK_BYTES: byte ranges of a stored member, or the streamed decompression of a
                compressed one up to the last record (pbar counts the records' bytes)."""
                offset, member = member_data(key)
                step = max(1, _ALOS2_BLOCK_BYTES // record_length)
                first = DESCRIPTOR + start * record_length
                if member.compress_type == zipfile.ZIP_STORED:
                    for a in range(start, end, step):
                        yield read(offset + first + (a - start) * record_length,
                                   offset + first + (min(end, a + step) - start) * record_length, pbar)
                    return
                block = bytearray()
                for piece in inflate(offset, member, first, DESCRIPTOR + end * record_length, _ALOS2_BLOCK_BYTES):
                    block += piece
                    while len(block) >= step * record_length:
                        out = bytes(block[:step * record_length])
                        del block[:step * record_length]
                        pbar.update(len(out))
                        yield out
                if block:
                    pbar.update(len(block))
                    yield bytes(block)

            # the LED first
            led = member_bytes('LED', 0, members['LED'].file_size)
            summary = _record(led, 10)
            if summary is None:
                raise ValueError(f"ALOS-1 LED of {fileID} has no data set summary record (type 10)")
            pols = ASF._alos1_pols(polarizations, available, fileID)
            if not pols:
                return
            head_length = DESCRIPTOR + PREFIX_ALOS1
            # the head of a compressed member: read in 64 KiB of compressed data at a time (one read holds it)
            head = member_bytes(pols[0], 0, head_length, 64 * 1024)
            image = image_info(head, PREFIX_ALOS1, raw)
            record_length = image['record_length']
            # the off-nadir angle of the LED and the date of the first image line name the files
            info = dict(info, offnadir=float(_field(summary, 1839, 1854)), date=image['date'].replace('-', ''))
            if bbox is not None and raw:
                # the beam-centre Doppler from the raw leader file's own yaw and orbit (the raw leader files hold no
                # Doppler centroid; a raw download reads no L1.1 product): utils_alos2_orbit.raw_doppler; raw_window adds
                # the margin of the frame's steering state (raw_margin) on both sides. The line rate: the header PRF,
                # half of it for PLR (4 channels)
                wavelength = float(_field(summary, 501, 516))
                rate = image['prf'] / (2 if int(_field(summary, 389, 392)) == 4 else 1)
                orbit, sampling_rate = led_orbit(led), led_sampling_rate(led)
                try:
                    attitude = led_attitude(led)
                    doppler = raw_doppler(orbit, attitude, image, sampling_rate, wavelength, rate)
                    margin = raw_margin(led)
                except ValueError as e:
                    raise ValueError(f"ALOS-1 product {fileID}: {e}") from None
                # the transform's region in every UTM zone of the frame's longitude span (the catalog footprint): the
                # zone of the focused image's footprint is one of them
                if not footprint:
                    raise ValueError(f"ALOS-1 product {fileID}: the catalog gives no footprint")
                west, south, east, north = shapely.geometry.shape(footprint).bounds
                epsgs = {get_utm_epsg(y, x) for y in (south, north) for x in np.r_[np.arange(west, east, 6.0), east]}
                # the time span of the echoes of the region; the lines are selected by their own time tags while the
                # first image streams (below), so the line rate of the frame (a PLR frame's lines come at half its
                # header PRF) and a PRF change inside it move no line
                span = raw_window(orbit, image, sampling_rate, transform_region(tuple(bbox), epsgs), wavelength, doppler,
                                  margin=margin)
                if span is None:
                    print(f"WARNING: bbox {tuple(bbox)} does not reach {fileID} at heights -500 to 9000 m: "
                          f"nothing stored")
                    return
                # the frame's own time span, from its first line: a span before its first line or after its end (its
                # lines at the line rate of line 1, plus ALOS1_FRAME_END_SLACK for a PRF change) holds none of its
                # lines, and the member is not read
                if span[1] < image['time'] or span[0] > image['time'] + image['lines'] / rate + ALOS1_FRAME_END_SLACK:
                    print(f"WARNING: bbox {tuple(bbox)} does not reach {fileID} at heights -500 to 9000 m "
                          f"(its time span is outside the frame): nothing stored")
                    return
                window = None
                if debug:
                    print(f"{fileID}: the grid of the bbox in EPSG {sorted(epsgs)}, Doppler {doppler[0]:.3f} + "
                          f"{doppler[1]:.6g} * R Hz, margin {margin[0]} s, seconds {span[0]:.3f}..{span[1]:.3f} of "
                          f"{image['date']}")
            elif bbox is not None:
                # the image's Doppler centroid fd1 + fdd1 * R (Hz, R the slant range in m), as GMTSAR's ALOS
                # preprocessing reads it from the data set summary record: bytes 1735-1750 (Hz) and 1751-1766 (Hz/km);
                # the radar wavelength bytes 501-516 (m).
                # Measured 2026-11-23 against JAXA's per-line ground points (prefix bytes 193-216): with this Doppler the
                # predicted line is constant across the swath (spread 0.04 lines) but 4.24 (FBD, path 119) and 5.23 (FBS, path 118)
                # lines off. The line times are whole milliseconds, plus a fixed 2.74 ms ALOS-1 offset between the line time tag
                # and JAXA's geolocation (GMTSAR adds +4 lines, ALOS_llt2rat.c). It is not applied here: the windows hold the
                # transform grid with at least 252 lines to spare. It belongs to the ALOS-1 preprocessor.
                doppler = (float(_field(summary, 1735, 1750)), 1e-3 * float(_field(summary, 1751, 1766)),
                           float(_field(summary, 501, 516)))
                stream = {'pieces': None, 'at': 0, 'kept': b'', 'next': 0, 'prefixes': {}}

                def prefix(line):
                    """The 412-byte prefix of an image line of the first polarization: a byte-range read of a stored
                    member; of a compressed one, from one decompression stream from its start that goes on forward as
                    far as a call needs and keeps the prefix of every line it passes (412 bytes a line), so that a
                    second pass reads no line again; one piece of decompressed data in memory."""
                    first = DESCRIPTOR + line * record_length
                    offset, member = member_data(pols[0])
                    if member.compress_type == zipfile.ZIP_STORED:
                        return read(offset + first, offset + first + PREFIX_ALOS1)
                    if stream['pieces'] is None:
                        stream['pieces'] = inflate(offset, member, 0, member.file_size, _ALOS2_BLOCK_BYTES)
                    while line not in stream['prefixes']:
                        piece = next(stream['pieces'])
                        # the end of the previous piece stays: a prefix may begin in it
                        stream['kept'] = stream['kept'][-PREFIX_ALOS1:] + piece
                        stream['at'] += len(piece)
                        start = stream['at'] - len(stream['kept'])
                        while (stream['next'] < image['lines']
                               and DESCRIPTOR + stream['next'] * record_length + PREFIX_ALOS1 <= stream['at']):
                            a = DESCRIPTOR + stream['next'] * record_length - start
                            stream['prefixes'][stream['next']] = stream['kept'][a:a + PREFIX_ALOS1]
                            stream['next'] += 1
                    return stream['prefixes'][line]

                # the line window of the bbox: the window rule shared with ALOS-2
                window, epsgs = ASF._crop_window(bbox, led_orbit(led), image, led_sampling_rate(led), prefix,
                                                 PREFIX_ALOS1, f'ALOS-1 {fileID}', doppler)
                # the decompressed data kept for the prefixes is released
                stream.update(pieces=None, kept=b'', prefixes={})
                if window is None:
                    print(f"WARNING: bbox {tuple(bbox)} does not reach {fileID} at heights -500 to 9000 m: "
                          f"nothing stored")
                    return
                if debug:
                    print(f"{fileID}: the grid of the bbox in EPSG {sorted(epsgs)}, Doppler {doppler[0]} + "
                          f"{doppler[1]} * R Hz")
            else:
                window = (0, image['lines'])
            if debug and window is not None:
                print(f"{fileID}: lines {window[0]}..{window[1]} of {image['lines']}, {record_length} bytes each")

            led_path = ASF._alos1_path(basedir, info)

            def store_led():
                # the folder is created when a file is written
                os.makedirs(os.path.dirname(led_path), exist_ok=True)
                if not exists(led_path):
                    with open(led_path + '.tmp', 'wb') as f:
                        f.write(led)
                    os.replace(led_path + '.tmp', led_path)

            todo = [p for p in pols if not (skip_exist and exists(ASF._alos1_path(basedir, info, p)))]
            if window is None:
                # a resumed raw crop: the lines of an image of the product already stored (and not written again), so
                # that every polarization holds the same lines (the line tags of a PLR frame's VH and VV lie half a line
                # period from HH and HV: their own selection can move the window by one block)
                for q in available:
                    stored = ASF._alos1_path(basedir, info, q)
                    if q not in todo and exists(stored):
                        with open(stored, 'rb') as f:
                            window = crop_lines(f.read(head_length), record_length)
                        if window[1] > image['lines']:
                            raise ValueError(f"ALOS-1 {os.path.basename(stored)} holds lines {window[0]}..{window[1]}, "
                                             f"beyond the {image['lines']} lines of {fileID}")
                        break
            # the LED of a raw crop is stored with its first image, once its lines are found
            if window is not None:
                store_led()
            desc = os.path.basename(led_path)[:-len('.led')]
            total = None if window is None else len(todo) * (window[1] - window[0]) * record_length
            with tqdm(total=total, unit='B', unit_scale=True, desc=desc, leave=True, dynamic_ncols=False, ncols=80,
                      mininterval=0.3, smoothing=0) as pbar:
                for p in todo:
                    own = head if p == pols[0] else member_bytes(p, 0, head_length, 64 * 1024)
                    own_image = image_info(own, PREFIX_ALOS1, raw)
                    if any(own_image[k] != image[k] for k in ('lines', 'record_length', 'pixels')):
                        raise ValueError(f"ALOS-1 IMG-{p} of {fileID} has another size than IMG-{pols[0]}")
                    member_size = members[p].file_size
                    if member_size != DESCRIPTOR + image['lines'] * record_length:
                        raise ValueError(f"ALOS-1 IMG-{p} of {fileID} has {member_size} bytes, expected "
                                         f"{DESCRIPTOR + image['lines'] * record_length}")
                    img_path = ASF._alos1_path(basedir, info, p)
                    if window is None:
                        # the first image of a raw crop: the records stream from line 1 and the lines whose time tags meet
                        # the span are kept (utils_alos2_ceos.select_lines), the window rounded outward to
                        # _ALOS2_LINE_BLOCK lines and clipped to the frame; nothing is read after its last line, and the
                        # other polarizations take the same lines
                        found = {}
                        blocks = select_lines(records(p, 0, image['lines'], record_length, pbar), record_length,
                                              image['lines'], image['date'], span[0], span[1], _ALOS2_LINE_BLOCK, found)
                        # the first block of the window, before the folder is created: none when no line lies in the span
                        first = next(blocks, None)
                        if first is None:
                            print(f"WARNING: bbox {tuple(bbox)} does not reach {fileID} at heights -500 to 9000 m "
                                  f"(no line of the frame in its time span): nothing stored")
                            return
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        write_crop(img_path, own[:DESCRIPTOR], chain([first], blocks), None, record_length)
                        window = (found['start'], found['end'])
                        pbar.total = pbar.n + (len(todo) - 1) * (window[1] - window[0]) * record_length
                        pbar.refresh()
                        if debug:
                            print(f"{fileID}: lines {window[0]}..{window[1]} of {image['lines']}, {record_length} bytes "
                                  f"each")
                        store_led()
                    else:
                        # one block of records in memory at a time
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        write_crop(img_path, own[:DESCRIPTOR], records(p, window[0], window[1], record_length, pbar),
                                   window[1] - window[0], record_length)
                    name = os.path.basename(img_path)
                    if name not in written:
                        written.append(name)
            store_led()

        # the products one after another (each read with its own attempts)
        written = []
        for fileID, info in products.items():
            download_product(fileID, info, written)

        if written:
            return pd.DataFrame({'file': written})
        return None

    def _read_nisar_metadata_direct(self, pol, remote):
        """Read the metadata datasets of one polarization of a NISAR granule, to copy into its output file.

        The granule is opened by h5py through remote (_NisarRemoteFile): its structure is read from the head and the
        pages remote fetches on demand. The datasets to copy are collected (every dataset but the SLC rasters and the
        objects of the other polarizations; an object whose structure cannot be read is skipped), then the storage of
        their values (contiguous: its offset and size; chunked: the offset and size of every chunk; compact or not
        allocated: nothing) is fetched by remote.prefetch in one batch, wherever the granule stores it, and the values
        are read. A value that cannot be read raises, with the dataset's path: nothing is dropped silently.

        The group science/LSAR/RSLC/metadata/RFI and everything below it is not copied (never listed, prefetched or
        read): it holds the RFI detection statistics of the focusing stage in raw-data geometry (the raw slow time and
        raw slant range of analysis blocks; hitCount is a 1024-bin spectrum per block, 61 MiB uncompressed in a validated
        granule), which no processor of the focused SLC uses; it is not downloaded, to limit the downloaded size.
        calibrationInformation/.../rfiLikelihood is outside it and is copied.

        Parameters
        ----------
        pol : str
            Polarization being processed (e.g., 'HH').
        remote : _NisarRemoteFile
            The file object over the granule.

        Returns
        -------
        dict
            {dataset path: {'data', 'dtype', 'shape', 'attrs'}}, '_group_attrs' {group path: attributes} and
            '_root_attrs'.

        Raises
        ------
        OSError
            When the values of a dataset cannot be read (the text names the dataset).

        Examples
        --------
        >>> metadata = asf._read_nisar_metadata_direct('HH', remote)
        """
        import h5py

        # ALL main SLC datasets to skip (stored later in file)
        all_slc = {f'science/LSAR/RSLC/swaths/frequency{f}/{p}'
                   for f in ['A', 'B'] for p in ['HH', 'HV', 'VH', 'VV']}

        # Other polarizations to skip (their metadata may be stored later)
        other_pols = {'HH', 'HV', 'VH', 'VV'} - {pol}

        # RFI detection statistics in raw-data geometry, unused for the focused SLC: not downloaded
        rfi = 'science/LSAR/RSLC/metadata/RFI'

        def should_skip_path(path):
            """Check if path should be skipped - SLC, other-pol data or metadata/RFI."""
            if path in all_slc:
                return True
            if path == rfi or path.startswith(f'{rfi}/'):
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

        def storage(ds):
            """The (offset, size) byte ranges that store the values of ds in the file."""
            layout = ds.id.get_create_plist().get_layout()
            if layout == h5py.h5d.CONTIGUOUS:
                offset, size = ds.id.get_offset(), ds.id.get_storage_size()
                return [(offset, size)] if offset is not None and size else []
            if layout == h5py.h5d.CHUNKED:
                chunks = (ds.id.get_chunk_info(i) for i in range(ds.id.get_num_chunks()))
                return [(c.byte_offset, c.size) for c in chunks if c.byte_offset is not None and c.size]
            # compact (the values are in the object header, read with the structure) or virtual
            return []

        metadata = {}

        with h5py.File(remote, 'r') as h5:
            # Use safe iteration that checks paths before accessing
            datasets, groups = iterate_safe(h5)

            # Collect group attributes
            metadata['_group_attrs'] = {}
            for path, grp in groups:
                if grp.attrs:
                    metadata['_group_attrs'][path] = dict(grp.attrs)

            # Root attributes
            metadata['_root_attrs'] = dict(h5.attrs)

            # the values of every dataset to copy, fetched in one batch wherever the granule stores them
            remote.prefetch([r for _, ds in datasets for r in storage(ds)])

            # Read dataset values: a value that cannot be read raises
            for path, ds in datasets:
                try:
                    data = ds[()]
                except Exception as e:
                    raise OSError(f'NISAR metadata {path}: its values cannot be read: {e}') from e
                metadata[path] = {
                    'data': data,
                    'dtype': ds.dtype,
                    'shape': ds.shape,
                    'attrs': dict(ds.attrs)
                }

        return metadata

    @staticmethod
    def _nisar_stream_writer(out_path, pol, chunk_info_a, chunk_info_b, get_metadata, crop_info=None):
        """Write one polarization of a NISAR HDF5 file to out_path + '.tmp' as its chunks arrive, then rename it to out_path.

        Returns a context manager whose value is write(pairs): pairs is an iterable of (chunk_offset, chunk_bytes). Each chunk of the
        output is written to the file at once, so its bytes are not kept; a chunk outside the output (outside the crop) is ignored.
        The output datasets are created on entry. On a normal exit the metadata is written, the file is validated and renamed to
        out_path (os.replace). On an exception the temporary file is removed and nothing is written under out_path.

        Parameters
        ----------
        out_path : str
            The final file name.
        pol : str
            Polarization, e.g. 'HH'.
        chunk_info_a, chunk_info_b : dict or None
            Chunk metadata from get_chunk_info() for frequencyA and frequencyB; None when that frequency is not written.
        get_metadata : callable
            Returns ALL metadata from _read_nisar_metadata_direct(), called on a normal exit (after the chunks, as the
            metadata is read after the downloads).
        crop_info : dict, optional
            The bbox crop: the output is cropped to chunk-aligned bounds, and the identification times to the crop's lines.
        """
        import contextlib
        import os
        import h5py

        def stream():
            has_freq_a = chunk_info_a is not None
            has_freq_b = chunk_info_b is not None
            tmp_path = out_path + '.tmp'
            try:
                with h5py.File(tmp_path, 'w') as h5_out:
                    # Calculate chunk-aligned crop extents if cropping
                    crop_az_start, crop_az_end = 0, None
                    crop_rg_start_a, crop_rg_end_a = 0, None
                    crop_rg_start_b, crop_rg_end_b = 0, None

                    if crop_info:
                        # Get chunk-aligned bounds (full chunks, not pixels)
                        if has_freq_a:
                            chunk_az = chunk_info_a['chunk_shape'][0]
                            chunk_rg = chunk_info_a['chunk_shape'][1]
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

                    wanted = {}   # chunk offset -> (dataset, coordinate in the output, frequency label)
                    written = {}  # frequency label -> chunks written

                    # 1. frequencyA SLC if available: the dataset now, its chunks as they arrive
                    if has_freq_a:
                        orig_shape = chunk_info_a['shape']
                        chunk_shape = chunk_info_a['chunk_shape']
                        dtype = chunk_info_a['dtype']
                        compression = chunk_info_a['compression']
                        compression_opts = chunk_info_a['compression_opts']
                        shuffle = chunk_info_a.get('shuffle', False)

                        # Calculate output shape (cropped or full)
                        if crop_info:
                            out_shape = (min(crop_az_end, orig_shape[0]) - crop_az_start,
                                         min(crop_rg_end_a, orig_shape[1]) - crop_rg_start_a)
                        else:
                            out_shape = orig_shape

                        dst_slc = h5_out.create_dataset(
                            f'science/LSAR/RSLC/swaths/frequencyA/{pol}',
                            shape=out_shape, dtype=dtype, chunks=chunk_shape,
                            compression=compression, compression_opts=compression_opts,
                            shuffle=shuffle
                        )

                        for chunk in chunk_info_a['chunks']:
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
                            wanted[chunk['offset']] = (dst_slc, new_coord, 'A')

                    # 2. frequencyB SLC if available
                    if has_freq_b:
                        orig_shape_b = chunk_info_b['shape']
                        chunk_shape_b = chunk_info_b['chunk_shape']
                        dtype_b = chunk_info_b['dtype']
                        compression_b = chunk_info_b['compression']
                        compression_opts_b = chunk_info_b['compression_opts']
                        shuffle_b = chunk_info_b.get('shuffle', False)

                        # Calculate output shape (cropped or full)
                        if crop_info:
                            out_shape_b = (min(crop_az_end, orig_shape_b[0]) - crop_az_start,
                                           min(crop_rg_end_b, orig_shape_b[1]) - crop_rg_start_b)
                        else:
                            out_shape_b = orig_shape_b

                        dst_slc_b = h5_out.create_dataset(
                            f'science/LSAR/RSLC/swaths/frequencyB/{pol}',
                            shape=out_shape_b, dtype=dtype_b, chunks=chunk_shape_b,
                            compression=compression_b, compression_opts=compression_opts_b,
                            shuffle=shuffle_b
                        )

                        for chunk in chunk_info_b['chunks']:
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
                            wanted[chunk['offset']] = (dst_slc_b, new_coord, 'B')

                    def write(pairs):
                        for offset, chunk_bytes in pairs:
                            target = wanted.pop(offset, None)
                            if target is None:
                                continue  # not in the output
                            dataset, coord, label = target
                            if len(chunk_bytes) == 0:
                                raise ValueError(f'freq{label} {pol}: empty chunk at offset {offset}')
                            dataset.id.write_direct_chunk(coord, chunk_bytes)
                            written[label] = written.get(label, 0) + 1

                    yield write

                    # every chunk of the output has to have arrived
                    for label, has_freq in (('A', has_freq_a), ('B', has_freq_b)):
                        if has_freq and written.get(label, 0) == 0:
                            raise ValueError(f'freq{label} {pol}: no chunks downloaded')
                    if wanted:
                        raise ValueError(f'{pol}: {len(wanted)} chunk(s) of the output not downloaded')

                    # 3. Write metadata datasets (filter by frequency if needed); a crop gets the identification start and
                    # end times of its own first and last lines
                    metadata = get_metadata()
                    crop_times = ASF._nisar_crop_times(metadata, crop_az_start, crop_az_end) if crop_info else {}
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
                        if parent_path and parent_path not in h5_out:
                            h5_out.create_group(parent_path)

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
                                ds = h5_out.create_dataset(ds_path, data=data, compression='gzip', compression_opts=4)
                            else:
                                ds = h5_out.create_dataset(ds_path, data=data)
                            # Copy dataset attributes (description, units, etc.)
                            if 'attrs' in ds_info:
                                for attr_key, attr_val in ds_info['attrs'].items():
                                    try:
                                        ds.attrs[attr_key] = attr_val
                                    except Exception:
                                        pass
                        except Exception as e:
                            print(f"    Warning: could not create {ds_path}: {e}")

                    # 4. Copy group attributes (filter by frequency if needed)
                    if '_group_attrs' in metadata:
                        for grp_path, grp_attrs in metadata['_group_attrs'].items():
                            # Skip frequency-specific groups when not writing that frequency
                            if not has_freq_a and 'frequencyA' in grp_path:
                                continue
                            if not has_freq_b and 'frequencyB' in grp_path:
                                continue
                            if grp_path in h5_out:
                                for attr_key, attr_val in grp_attrs.items():
                                    try:
                                        h5_out[grp_path].attrs[attr_key] = attr_val
                                    except Exception:
                                        pass

                    # 5. Copy root attributes
                    if '_root_attrs' in metadata:
                        for key, value in metadata['_root_attrs'].items():
                            h5_out.attrs[key] = value

                # Validate the finished file before it gets its final name
                with h5py.File(tmp_path, 'r') as h5_check:
                    swaths = h5_check['science/LSAR/RSLC/swaths']
                    for freq_key in ['frequencyA', 'frequencyB']:
                        if freq_key in swaths and pol in swaths[freq_key]:
                            ds = swaths[freq_key][pol]
                            if ds.shape[0] == 0 or ds.shape[1] == 0:
                                raise ValueError(f'{freq_key}/{pol} has zero dimension: {ds.shape}')
                            break
                    else:
                        raise ValueError(f'No SLC dataset found for {pol}')

                os.replace(tmp_path, out_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

        return contextlib.contextmanager(stream)()

    @staticmethod
    def update(basedir, skip_exist=True, n_jobs=8, joblib_backend='loky', retries=30, timeout_second=3, min_rate='100KB',
               min_rate_window=60):
        """
        Update the orbit stored in the Sentinel-1 bursts of a data directory.

        The orbit stored in the annotation of every Sentinel-1 burst in basedir (generalAnnotation/orbitList, with the
        attribute source = the orbit file name) is replaced by the most precise, most recent published orbit: the
        orbit file of the burst's mission that covers the burst, a precise orbit (POEORB) first, then the newest
        production. With skip_exist=True only the bursts whose stored orbit (source) differs from the selected file
        are rewritten, such as a restituted orbit (RESORB) once the POEORB is published (about 20 days after the
        acquisition), a reprocessed POEORB, or a burst downloaded before the orbit was stored in it; with
        skip_exist=False the orbit of every stored burst is rewritten. Only the annotation files change, each written
        to a temporary file and renamed when complete; the measurement, noise and calibration files and the IW2
        annotation of IW1 and IW3 bursts are not changed. The orbit files are read from the orbit server: the call
        needs internet and no account. A burst with no published orbit keeps its own, with a WARNING.

        Parameters
        ----------
        basedir : str
            The data directory of the burst folders.
        skip_exist : bool, optional
            True rewrites only the bursts whose stored orbit differs from the selected one; False rewrites the orbit
            of every stored burst. Default True.
        n_jobs : int, optional
            The number of concurrent orbit downloads, as EOF.download. Default 8.
        joblib_backend : str, optional
            The backend for parallel processing. Default 'loky'.
        retries : int, optional
            The number of attempts of each orbit index and orbit file request, as EOF.download. Default 30.
        timeout_second : float, optional
            Seconds between the attempts. Default 3.
        min_rate : str or float, optional
            Bytes per second an orbit download must keep, as EOF.download. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default 60.

        Returns
        -------
        pandas.DataFrame or None
            The rewritten bursts (column burst) and their new orbit (column orbit, the orbit file name), or None when
            no burst is rewritten.

        Examples
        --------
        >>> asf = ASF()
        >>> asf.update(DATADIR)
        >>> # rewrite the orbit of every stored burst
        >>> asf.update(DATADIR, skip_exist=False)
        """
        from .EOF import EOF
        return EOF._update_bursts(basedir, skip_exist=skip_exist, n_jobs=n_jobs, joblib_backend=joblib_backend,
                                  retries=retries, timeout_second=timeout_second, min_rate=min_rate,
                                  min_rate_window=min_rate_window)

    @staticmethod
    def _alos2_subswath_rows(features, geometry):
        """
        The rows of ASF.search for ALOS-2 scenes: per scene, one row per subswath F1..F5 that the AOI may reach.

        The catalog holds one polygon per ALOS-2 scene and none per subswath. Each subswath is a fixed across-track
        fraction of the scene polygon (_ALOS2_FOOTPRINT), measured from the near edge, which follows from the frame in
        the fileID (the heading) and the look side (R right, L left). A subswath is kept when the AOI intersects its
        footprint widened across track by the model error and the terrain margins (heights -500..9000 m) and along
        track by the measured reach of the subswaths beyond the start and end edges of the scene polygon. The work is
        done in the tangent plane at the scene centre (WGS84), so it holds near the poles and across the antimeridian.
        A catalog polygon that is not a scene outline (its sides or angles out of the measured ranges, or not 4 corners)
        lists all 5 subswaths without the AOI filter, with one WARNING naming the fileID.

        Parameters
        ----------
        features : list of dict
            The GeoJSON features of ALOS-2 scenes as the catalog returns them (properties with fileID, Polygon geometry).
        geometry : shapely geometry
            The AOI in longitude and latitude (degrees).

        Returns
        -------
        geopandas.GeoDataFrame
            One row per kept subswath, in scene order and then F1..F5: the scene's catalog properties, `subswath`
            ('F1'..'F5'), `sceneName` = `<fileID>-F<n>` (the ID ASF.download takes), `fileID` unchanged, and the
            subswath footprint as geometry (the scene polygon for a scene listed without the AOI filter). Footprint
            longitudes are kept continuous around the scene centre, so near the antimeridian they may pass 180 or -180.

        Examples
        --------
        >>> scene = {'type': 'Feature', 'properties': {'fileID': 'ALOS2516422900-231215-WBSR1.1__D'},
        ...          'geometry': {'type': 'Polygon', 'coordinates': [[[-123.323, 34.277], [-119.548, 33.723],
        ...                       [-118.793, 36.9], [-122.726, 37.45], [-123.323, 34.277]]]}}
        >>> ASF._alos2_subswath_rows([scene], shapely.geometry.box(-121.228, 35.504, -120.978, 35.704))['sceneName'].tolist()
        ['ALOS2516422900-231215-WBSR1.1__D-F3']
        """
        import re
        import numpy as np
        import shapely
        import geopandas as gpd

        a, e2 = 6378137.0, 6.69437999014e-3
        table = _ALOS2_FOOTPRINT

        def ecef(lonlat):
            """WGS84 ECEF (m) of longitude/latitude points (degrees) at h = 0, shape (n, 3)."""
            lon, lat = np.radians(lonlat[:, 0]), np.radians(lonlat[:, 1])
            n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
            return np.stack([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon),
                             n * (1 - e2) * np.sin(lat)], axis=1)

        def plane(origin):
            """The east, north and up unit vectors at the geodetic position of an ECEF point."""
            x, y, z = origin
            p = np.hypot(x, y)
            lon = np.arctan2(y, x)
            lat = np.arctan2(z, p * (1 - e2))
            for _ in range(5):
                n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
                h = p / np.cos(lat) - n
                lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
            east = np.array([-np.sin(lon), np.cos(lon), 0.0])
            north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
            up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            return east, north, up, np.degrees(lon)

        def to_plane(lonlat, origin, east, north):
            """Tangent-plane east/north (km) of longitude/latitude points, shape (n, 2)."""
            d = ecef(np.asarray(lonlat, dtype=float)) - origin
            return np.stack([d @ east, d @ north], axis=1) / 1000.0

        def to_lonlat(points, origin, east, north, up, lon0):
            """Longitude/latitude of tangent-plane points: the ellipsoid point (h = 0) on the plane normal through each,
            the inverse of to_plane; longitudes continuous around lon0."""
            q = origin + 1000.0 * (points[:, :1] * east + points[:, 1:2] * north)
            w = np.array([1 / a ** 2, 1 / a ** 2, 1 / (a ** 2 * (1 - e2))])
            qa, qb, qc = (up * up * w).sum(), (q * up * w).sum(axis=1), (q * q * w).sum(axis=1) - 1
            # the root of the quadratic nearer the plane
            root = np.sqrt(qb * qb - qa * qc)
            s1, s2 = (-qb + root) / qa, (-qb - root) / qa
            s = np.where(np.abs(s1) <= np.abs(s2), s1, s2)
            r = q + s[:, None] * up
            lon = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
            lat = np.degrees(np.arctan2(r[:, 2], (1 - e2) * np.hypot(r[:, 0], r[:, 1])))
            lon = lon0 + (lon - lon0 + 180.0) % 360.0 - 180.0
            return np.stack([lon, lat], axis=1)

        def unit(v):
            return v / np.linalg.norm(v)

        def cosine(u, v):
            return abs(float(np.dot(u, v))) / (np.linalg.norm(u) * np.linalg.norm(v))

        # the AOI densified to at most 0.1 degree per edge before it is projected into each scene's plane
        aoi = shapely.segmentize(geometry, 0.1)
        rows = []
        for feature in features:
            props = dict(feature['properties'])
            fileID = props['fileID']
            m = re.match(r'^ALOS2\d{5}(\d{4})-\d{6}-[A-Z]{3}([RL])1\.1__[A-Z]$', fileID)
            shape = shapely.geometry.shape(feature['geometry'])
            corners = None
            if shape.geom_type == 'Polygon':
                coords = np.asarray(shape.exterior.coords, dtype=float)[:, :2]
                if len(coords) > 1 and np.array_equal(coords[0], coords[-1]):
                    coords = coords[:-1]
                if len(coords) == 4:
                    corners = coords
            footprints = {}
            if corners is not None and m is not None:
                xyz = ecef(corners)
                origin = xyz.mean(axis=0)
                east, north, up, lon0 = plane(origin)
                c = to_plane(corners, origin, east, north)
                sides = [c[(k + 1) % 4] - c[k] for k in range(4)]
                lengths = [float(np.linalg.norm(s)) for s in sides]
                angles = [float(np.degrees(np.arccos(np.clip(np.dot(-sides[k - 1], sides[k]) /
                                                             (lengths[k - 1] * lengths[k]), -1, 1))))
                          for k in range(4)]
                if all(table['side_km'][0] <= v <= table['side_km'][1] for v in lengths) \
                        and all(table['angle_deg'][0] <= v <= table['angle_deg'][1] for v in angles):
                    # the heading at the scene from its frame (argument of latitude u, inclination i)
                    u = np.radians(table['frame_deg'] * int(m.group(1)))
                    i = np.radians(table['inclination_deg'])
                    phs = np.arcsin(np.sin(u) * np.sin(i))
                    ratio = np.cos(i) / np.cos(phs)
                    tE = ratio - table['earth_rotation'] * np.cos(phs)
                    tN = np.sign(np.cos(u)) * np.sqrt(max(0.0, 1 - min(1.0, ratio ** 2)))
                    t = unit(np.array([tE, tN]))
                    right = np.array([t[1], -t[0]])
                    look = m.group(2)
                    # the along-track sides: the pair of opposite sides more parallel to the heading
                    j = 0 if cosine(sides[0], t) + cosine(sides[2], t) >= cosine(sides[1], t) + cosine(sides[3], t) else 1
                    mids = {k: float(np.dot((c[k] + c[(k + 1) % 4]) / 2, right)) for k in (j, j + 2)}
                    near = min(mids, key=mids.get) if look == 'R' else max(mids, key=mids.get)
                    # the two across-track sides, each from its near corner to its far corner
                    pairs = [(c[near], c[(near + 3) % 4]), (c[(near + 1) % 4], c[(near + 2) % 4])]
                    aoi_plane = shapely.transform(aoi, lambda xy: to_plane(xy, origin, east, north))
                    model = table['model_km'][look]
                    along = table['along_km'][look]
                    for name, (f0, f1, h0, h1) in table['subswaths'].items():
                        # near side: fraction f0, moved toward the near edge; far side: f1, moved toward the far edge
                        near_side, far_side = (f0, -(model + h0)), (f1, model + h1)
                        outline, selection = [], []
                        for (n0, n1), order in zip(pairs, ((near_side, far_side), (far_side, near_side))):
                            d = n1 - n0
                            for f, shift in order:
                                outline.append(n0 + f * d)
                                selection.append(n0 + f * d + shift * unit(d))
                        # along track: the corners of each across-track side moved away from the other side's corners
                        # of the same near or far side (outline order: near, far on one side, far, near on the other)
                        for k, other in ((0, 3), (1, 2), (2, 1), (3, 0)):
                            selection[k] = selection[k] + along * unit(outline[k] - outline[other])
                        if aoi_plane.intersects(shapely.geometry.Polygon(selection)):
                            lonlat = to_lonlat(np.array(outline), origin, east, north, up, lon0)
                            footprints[name] = shapely.geometry.polygon.orient(shapely.geometry.Polygon(lonlat), sign=1.0)
                else:
                    corners = None
            if corners is None or m is None:
                print(f'WARNING: The catalog footprint of the ALOS-2 scene {fileID} is not a scene outline: its 5 '
                      f'subswaths are listed without the AOI filter.')
                footprints = {name: shape for name in table['subswaths']}
            for name, footprint in footprints.items():
                rows.append({'type': 'Feature', 'geometry': shapely.geometry.mapping(footprint),
                             'properties': {**props, 'sceneName': f'{fileID}-{name}', 'subswath': name}})
        return gpd.GeoDataFrame.from_features(rows, crs='EPSG:4326')

    @staticmethod
    def _alos1_rows(features, polarization=None):
        """
        The rows of ASF.search for ALOS-1 PALSAR: one row per L1.1 (SLC) product and per L1.0 (raw signal data) product
        of the stripmap modes FBS, FBD and PLR, with the products holding every requested polarization.

        A product of another level or mode (found with an explicit processingLevel or beamMode) is dropped with one
        WARNING (ASF.download does not support it). The polarizations
        of a product are its catalog polarization: 'HH', 'VV', 'HH+HV', 'VV+VH', or 'quadrature' (PLR: HH, HV, VH and
        VV). The catalog's own polarization filter matches
        the whole value ('HH' finds no HH+HV product, measured 2026-11-22), so the filter is applied to the rows here.

        Parameters
        ----------
        features : list of dict
            The GeoJSON features of ALOS-1 products as the catalog returns them (properties with fileID, beamModeType,
            polarization).
        polarization : str, list or None, optional
            The polarizations a product must hold: 'HH' keeps the HH, HH+HV and PLR products, 'HH+HV' or 'HV' the
            HH+HV and PLR products, 'VV' the VV, VV+VH and PLR products. None (default) keeps every product (VV and
            VV+VH included).

        Returns
        -------
        geopandas.GeoDataFrame
            One row per product in catalog order: the catalog properties with `sceneName` = the fileID (the ID
            ASF.download takes, such as 'ALPSRP131296360-L1.1' or 'ALPSRP055006360-L1.0'), `beamMode` = the catalog
            beamModeType, the catalog processingLevel ('L1.1' or 'L1.0'), and the catalog footprint as geometry; CRS
            EPSG:4326.

        Examples
        --------
        >>> product = {'type': 'Feature', 'properties': {'fileID': 'ALPSRP131296360-L1.1', 'beamModeType': 'FBD',
        ...            'polarization': 'HH+HV'}, 'geometry': {'type': 'Polygon', 'coordinates': [[[-72.89, -40.297],
        ...            [-72.684, -40.813], [-71.885, -40.621], [-72.097, -40.107], [-72.89, -40.297]]]}}
        >>> ASF._alos1_rows([product], 'HV')['sceneName'].tolist()
        ['ALPSRP131296360-L1.1']
        """
        import geopandas as gpd

        def pols_of(value):
            # the catalog polarization of a product as a set of polarizations
            value = value or ''
            if isinstance(value, str):
                value = ['HH', 'HV', 'VH', 'VV'] if value.strip().lower() == 'quadrature' \
                    else value.replace(',', '+').split('+')
            return {p.strip().upper() for p in value}

        def modes(fileID):
            # the modes of the products ASF.download takes, for an ID that ASF._alos1_parse accepts (an SLC, L1.1, or
            # a raw product, L1.0): FBS, FBD and PLR
            try:
                ASF._alos1_parse(fileID or '')
            except ValueError:
                return ()
            return ('FBS', 'FBD', 'PLR')

        supported = [f for f in features if f['properties'].get('beamModeType') in modes(f['properties'].get('fileID'))]
        if len(supported) < len(features):
            print(f'WARNING: {len(features) - len(supported)} ALOS-1 products other than the L1.1 and L1.0 products '
                  f'of the modes FBS, FBD and PLR are dropped (ASF.download does not support them).')
        if polarization is not None:
            requested = pols_of('+'.join([polarization] if isinstance(polarization, str) else list(polarization)))
            supported = [f for f in supported if requested <= pols_of(f['properties'].get('polarization'))]
        rows = [{'type': 'Feature', 'geometry': f['geometry'],
                 'properties': {**f['properties'], 'sceneName': f['properties']['fileID'],
                                'beamMode': f['properties'].get('beamModeType')}}
                for f in supported]
        if not rows:
            return gpd.GeoDataFrame(columns=['geometry', 'sceneName', 'beamMode'], geometry='geometry', crs='EPSG:4326')
        return gpd.GeoDataFrame.from_features(rows, crs='EPSG:4326')

    @staticmethod
    def _nisar_latest(features):
        """
        The NISAR catalog features with, for each acquisition, only the most recent production product.

        An acquisition is the (level, product, track, direction, frame, mode, polarization, start) of the product name
        (_NISAR_NAME). Its products are ranked by the config PR (production) above any other (such as UR, urgent
        response), then the five digits of the CRID as a number (P05036 > P05023 > X05010 > X05009), then the product
        counter as a number; the highest rank is kept. The kept features keep their catalog order. A feature whose
        fileID is not a NISAR product name is kept as it is.

        Parameters
        ----------
        features : list of dict
            The GeoJSON features as the catalog returns them (properties with fileID).

        Returns
        -------
        list of dict
            The kept features, in catalog order.

        Examples
        --------
        >>> old = {'properties': {'fileID': 'NISAR_L1_PR_RSLC_033_113_D_079_4005_DHDH_A_20261020T011329_20261020T011403_P05023_N_F_J_001'}}
        >>> new = {'properties': {'fileID': 'NISAR_L1_PR_RSLC_033_113_D_079_4005_DHDH_A_20261020T011329_20261020T011403_P05036_P_F_J_001'}}
        >>> ASF._nisar_latest([old, new]) == [new]
        True
        """
        keys, best = [], {}
        for index, feature in enumerate(features):
            m = _NISAR_NAME.match(feature['properties'].get('fileID') or '')
            if m is None:
                keys.append(None)
                continue
            key = tuple(m.group(name) for name in
                        ('level', 'product', 'track', 'direction', 'frame', 'mode', 'polarization', 'start'))
            rank = (m.group('config') == 'PR', int(m.group('crid')[1:]), int(m.group('counter')))
            keys.append(key)
            if key not in best or rank > best[key][0]:
                best[key] = (rank, index)
        return [feature for index, (feature, key) in enumerate(zip(features, keys))
                if key is None or best[key][1] == index]

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None,
               platform='SENTINEL-1', processingLevel='auto', polarization=None, beamMode='auto'):
        """
        Search the ASF catalog for the products that intersect an AOI.

        Sentinel-1 (the default): one row per burst (processingLevel 'auto' is the burst level, beamMode 'auto' is IW),
        with the column fullBurstID. ALOS-2 (platform='ALOS-2', searched alone): one row per subswath F1..F5 of the
        5-subswath ScanSAR products in HH and in HH+HV that the AOI may reach (ASF._alos2_subswath_rows): the catalog
        holds one polygon per scene, so the subswath footprints are predicted from it and the selection is widened by
        the footprint error, the terrain and the along-track reach; a neighbouring subswath may be listed, and
        ASF.download with a bbox warns about a subswath the bbox does not reach. The column sceneName is the subswath ID
        `<fileID>-F<n>` that ASF.download takes, `subswath` is 'F1'..'F5' and the geometry is the predicted subswath
        footprint.

        ALOS-1 PALSAR (platform='ALOS', searched alone): one row per L1.1 (SLC) product of the stripmap modes FBS, FBD
        and PLR, or with processingLevel='L1.0' one row per raw signal data product of the same modes
        (ASF._alos1_rows): the default search lists SLC products only, the L1.0 search raw products only. The column
        sceneName is the product ID that ASF.download
        takes, such as 'ALPSRP131296360-L1.1' or 'ALPSRP055006360-L1.0', with the catalog columns pathNumber,
        frameNumber, offNadirAngle, polarization, startTime, processingLevel, beamMode (the catalog beamModeType) and
        the catalog footprint.

        NISAR (platform='NISAR'): one row per acquisition, the most recent production product of it
        (ASF._nisar_latest). The catalog holds BETA, PROVISIONAL and VALIDATED versions of one acquisition and the
        reprocessed products of earlier acquisitions, and ASF.download writes them all to the same file. An acquisition
        is the (level, product, track, direction, frame, mode, polarization, start) of the product name; its products
        are ranked by the config PR (production) above any other (such as UR, urgent response), then the five digits of
        the CRID as a number (P05036 > P05023 > X05010 > X05009), then the product counter as a number.

        Parameters
        ----------
        geometry : shapely geometry, geopandas.GeoDataFrame or geopandas.GeoSeries
            The AOI in longitude and latitude; a closed LineString is taken as its Polygon.
        startTime, stopTime : str, optional
            The time interval; a date 'YYYY-MM-DD' covers the whole day.
        flightDirection : str, optional
            'A' or 'ASCENDING', 'D' or 'DESCENDING'; None for both.
        platform : str or list, optional
            One platform name, a comma-separated string or a list, such as 'SENTINEL-1', 'SENTINEL-1C', 'ALOS-2' or
            'ALOS' (ALOS-1 PALSAR). ALOS-2 and ALOS cannot be mixed with other platforms. Default 'SENTINEL-1'.
        processingLevel : str, optional
            The catalog processing level; 'auto' is the burst level for Sentinel-1, no filter for ALOS-2 and L1.1 (SLC)
            for ALOS; 'L1.0' selects the ALOS raw products. An ALOS search is of one level: None (no filter) or a value
            naming both 'L1.0' and 'L1.1' raises ValueError, as raw and SLC products are never listed together.
            Default 'auto'.
        polarization : str, optional
            For Sentinel-1 the catalog filter, such as 'VV'. For ALOS-2 the products holding every named
            polarization: None or 'HH' keeps the HH and HH+HV products, 'HH+HV' or 'HV' the HH+HV products only.
            For ALOS the same rule (a PLR product holds HH, HV, VH and VV). Default None.
        beamMode : str, optional
            The catalog beam mode; 'auto' is IW for Sentinel-1, the 5-subswath ScanSAR products in HH and HH+HV for
            ALOS-2, FBS, FBD and PLR for the ALOS L1.1 and L1.0 products, and no filter for other platforms. An ALOS-2
            or ALOS product of another mode (or an ALOS product of another level), found with an explicit beamMode or
            processingLevel, is dropped with a WARNING
            (ASF.download does not support it). Default 'auto'.

        Returns
        -------
        geopandas.GeoDataFrame
            The catalog rows (Sentinel-1 bursts, ALOS-2 subswaths, ALOS-1 products, NISAR products) with their
            footprints, CRS EPSG:4326.

        Raises
        ------
        ValueError
            For a platform list that names ALOS-2 or ALOS and other platforms, or an ALOS search with processingLevel
            None or naming both 'L1.0' and 'L1.1'.

        Examples
        --------
        >>> ASF.search(geometry=AOI, startTime='2016-01-01', stopTime='2023-12-31', flightDirection='D', platform='ALOS-2')
        >>> ASF.search(geometry=AOI, startTime='2006-01-01', stopTime='2011-12-31', flightDirection='A', platform='ALOS')
        >>> ASF.search(geometry=AOI, startTime='2006-01-01', stopTime='2011-12-31', platform='ALOS', processingLevel='L1.0')
        """
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
        alos2 = [name for name in names if name.upper() == 'ALOS-2']
        if alos2 and len(alos2) != len(names):
            raise ValueError(f'ALOS-2 is searched alone: platform={platform!r} names other platforms too.')
        alos1 = [name for name in names if name.upper() == 'ALOS']
        if alos1 and len(alos1) != len(names):
            raise ValueError(f'ALOS is searched alone: platform={platform!r} names other platforms too.')
        sentinel1 = bool(names) and all(name.upper() in S1_PLATFORMS for name in names)
        # 'auto' is the burst level of every Sentinel-1 platform name; ALOS-2 has no processing level in the catalog;
        # ALOS-1 is the L1.1 (SLC) level
        if isinstance(processingLevel, str) and processingLevel=='auto':
            if sentinel1:
                processingLevel = asf_search.PRODUCT_TYPE.BURST
            elif alos2:
                processingLevel = None
            elif alos1:
                processingLevel = 'L1.1'
        # ALOS: raw (L1.0) and SLC (L1.1) products are never listed together
        if alos1:
            levels = [] if processingLevel is None else \
                [processingLevel] if isinstance(processingLevel, str) else list(processingLevel)
            levels = {level.strip().upper() for name in levels for level in str(name).split(',')} - {''}
            if not levels or {'L1.0', 'L1.1'} <= levels:
                raise ValueError(f"ALOS is searched at one level: processingLevel='L1.1' (SLC) or 'L1.0' (raw), "
                                 f"not {processingLevel!r}.")
        # 'auto' is IW for Sentinel-1, the 5-subswath ScanSAR products (WBD: HH+HV, WBS: HH) for ALOS-2 and for ALOS-1
        # the stripmap modes FBS, FBD and PLR of the SLC or of the raw products
        if isinstance(beamMode, str) and beamMode=='auto':
            beamMode = 'IW' if sentinel1 else 'WBD,WBS' if alos2 else 'FBS,FBD,PLR' if alos1 else None

        # search bursts; the ALOS-2 and ALOS-1 polarizations are filtered on the rows (the catalog's 'HH+HV' filter
        # returns no ALOS-2 product; for ALOS-1 it matches the whole value)
        results = asf_search.search(
            start=startTime,
            end=stopTime,
            flightDirection=flightDirection,
            intersectsWith=geometry.wkt,
            platform=','.join(names) if names else platform,
            processingLevel=processingLevel,
            polarization=None if (alos2 or alos1) else polarization,
            beamMode=beamMode,
        )
        if alos1:
            return ASF._alos1_rows([product.geojson() for product in results], polarization)
        if alos2:
            import re
            features = [product.geojson() for product in results]
            # the products ASF.download takes: the 5-subswath ScanSAR products in HH (WBS) and HH+HV (WBD)
            supported = [f for f in features
                         if re.match(r'^ALOS2\d{9}-\d{6}-(WBD|WBS)[RL]1\.1__[AD]$', f['properties'].get('fileID') or '')]
            if len(supported) < len(features):
                print(f'WARNING: {len(features) - len(supported)} ALOS-2 products other than the 5-subswath ScanSAR '
                      f'products in HH or HH+HV are dropped (ASF.download does not support them).')
            if polarization is not None:
                requested = [polarization] if isinstance(polarization, str) else list(polarization)
                requested = {p.strip().upper() for name in requested for p in name.replace(',', '+').split('+')}
                def held(f):
                    pols = f['properties'].get('polarization') or []
                    pols = pols.replace(',', '+').split('+') if isinstance(pols, str) else pols
                    return requested <= {p.strip().upper() for p in pols}
                supported = [f for f in supported if held(f)]
            return ASF._alos2_subswath_rows(supported, geometry)
        features = [product.geojson() for product in results]
        # NISAR: per acquisition the most recent production product (the versions are written to the same file)
        if names and all(name.upper() == 'NISAR' for name in names):
            features = ASF._nisar_latest(features)
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
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
