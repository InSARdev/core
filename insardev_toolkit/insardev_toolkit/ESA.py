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
ESA: search and cropped download of Envisat ASAR and ERS-1 / ERS-2 Image Mode SLC products (ASA_IMS_1P, SAR_IMS_1P) from ESA's
(A)SAR Online Dissemination service, with the precise orbit embedded (utils_envisat, utils_esa_orbit).

Facts for a future switch of source or reader (the reports are in session/patches/esa_alos1/reports):

The two kinds of catalogue records (SO-Cat):
- ARCHIVE records: ESA's catalogue of its archive, one record per 15 s scene of every acquisition, complete. ESA.search lists
  them and ESA.download requests them by their productId: ESA processes an archive record's scene on request with its current
  processor and serves it, so every scene of an AOI is available, all of one processor version. This is the route ESA's OTF user
  manual v2.5 (2025-10-02) gives: "L1 products generated on-the-fly are provided as fixed standard scenes in Envisat format,
  processed from Level 0 on user request by the system", chosen by mission phase, year, relative orbit and frame.
- DELIVERY records: the copies ESA delivered on request (created in 2025: the 18 s scenes, the short scenes inside them and the
  REOD records), partial (only the scenes someone requested) and duplicating archive scenes. They are never listed.

The selection by archive spans: ESA delivers L1 as fixed standard scenes processed on request, with "an overlap between
scenes ... added" (ESA OTF user manual v2.5): the delivered 18 s scene of an archive record is its 15 s span, 1.5 s before and
2.5 s after (U2a; an ERS extra record is served as kind A [b - 0.998, b + 16.997] or kind B [b - 1.498, e + 0.996] around its
acquisition b..e, U2c Q3). ESA.search selects the scenes that reach the AOI by their archive spans alone (the catalogue span of
a regular record, the acquisition of an extra record), never by the delivered overlap: the same slots serve the AOI on every
date, and no neighbour is listed for the overlap only. The archive spans tile the track except at data-take changes and missing
records: on 323 of 5,840 cached dates the listed spans leave 0.02-3.58 s of the AOI's time window (U2bD3_report.md,
U2bD3_verify.md), which the delivered overlap of the listed scenes covers. At the end of a data take ESA's delivered scene ends with
the acquisition (measured: 0.023 s before the catalogue end, orbit 40149), so there no scene holds the missing part. The crop of a listed scene
keeps every line of its delivered scene that holds the AOI, the overlap included.

Sources and access (measured 2026-11-23):
- ESA's (A)SAR service (esar-ds.eo.esa.int): the SO-Cat catalogue is public, with no login; the products are served by their
  archive record IDs (the 15 s records ESA.search lists) to an ESA EO Sign In account (SAML sign-in). The first request of a
  product not yet processed answers HTTP 202 with Retry-After while ESA processes it on request (1.7-17 min; the ERS extra
  records 2-3 min), then 206 to Range requests. ESA drops kept-alive connections: every request sends Connection: close
  (PLAN.md, the measured ESA facts). A standard account may request 20 L1 scenes per day and collection (OTF user manual
  v2.5); no counter is shown, and whether Range reads or 202 polls count is not established (U2a_report_r3.md,
  U2c_report_final.md, U2d_report.md).
- Each product request costs one of the 20 per day: ESA.download first reads the scene's precise orbit (ersorbits.insar.dev, no
  ESA request) and skips, with no request, a scene whose orbit is refused or whose delivered scene the bbox misses in time or
  in the swath's range (_precheck; U2bD4_report.md). ESA.search itself reads no orbit: its selection is the nominal model's.
- No ESA module reads ~/.netrc: the catalogue queries, the orbit reads and the downloads go through sessions that never look
  it up (utils_esa_orbit._NetrcFreeSession); the account of the downloads is the one given to ESA(username, password).
- CDSE lists ASA_IMS_1P only offline (Online false, no size, no S3 path) and no ERS SLC (PLAN.md, U2e_report.md).
- ASF took ERS-1 and ERS-2 out of its search and download tools on 2026-10-01; its files still answer a direct URL with an
  Earthdata login, but nothing lists them (U2d_report.md, U2e_report.md).
- Level 0 (ASA_IM__0P, SAR_IM__0P) is listed by SO-Cat but answers 403 NOT_AUTHORIZED: restricted data, granted only to
  projects ESA approves; it is not supported (U2d_report.md).

Processor versions (2026-11-23):
- An archive record ID serves ESA's on-request product of the scene, processed with ASAR/7.00 (every archive record request
  measured: U2a's 6 and U2c's, and the live crops of U2b-D2), under its delivered 18 s name. The 18 s name of a delivery record
  requested directly serves the 2023 version (ASAR/6.03 for Envisat, ASAR/6.05 for ERS) (U2a_report_r3.md "Products
  requested").
- The two versions of one scene (Envisat 2008-01-16, lines 14336-16383): equal record headers, 11.75 % of the samples differ, by
  at most 15 counts (RMS amplitude 104.7), coherence 0.999994, phase 0.0012 deg (U2b-D2 round 1, its m7_sample_vs_crop.json).
- ESA.search lists and ESA.download requests the archive record IDs only: one processor version per stack while ESA serves
  them so. The delivery records' 18 s names are not supported and no test uses them. The library does not compare versions;
  the MPH SOFTWARE_VER and PROC_TIME of each stored file name its processor.
"""
from .progressbar_joblib import progressbar_joblib
from .utils_esa_orbit import _NetrcFreeSession
import re

# ESA (A)SAR Online Dissemination: the public catalogue (SO-Cat, no login)
_SOCAT_URL = 'https://esar-ds.eo.esa.int/socat/{}/search'
# the largest catalogue answer read, in bytes: a longer one is asked again for half the time window
_SOCAT_LIMIT = 64 * 1024 * 1024
# the mission spans of the two collections, the search's time window when startTime or stopTime is None
_SOCAT_SPAN = {'ASA_IMS_1P': ('2002-03-01', '2012-05-01'), 'SAR_IMS_1P': ('1991-07-01', '2011-08-01')}
# the bbox query is widened by this many degrees of latitude (longitude scaled by the latitude): the records of the
# neighbouring grid slots, whose archive spans may meet the AOI's window within its margin (_FRAME_MARGIN), are in the answer
# (the width was set for the delivered 18 s scenes, which reach 2.5 s, 17 km, beyond the archive span: a margin now)
_SOCAT_WIDEN = 0.3
# the ERS collection's query is widened by this many degrees of latitude instead: the catalogue footprint of an extra record is
# trimmed at the start of the record (measured on the cached records: the served span of 0.3 % of the extra records reaches
# more than 1.2 degree beyond their footprint, 0.9 % beyond 1.0, 45 % beyond 0.3; the acquisition lies inside the served span),
# and an extra record covers the AOI by its acquisition
_SOCAT_WIDEN_ERS = 1.2

# the products of the ESA (A)SAR Online Dissemination service (collection, productId, extension) and its EO Sign In pages
_OADS_URL = 'https://esar-ds.eo.esa.int/oads/data/{}/{}{}'
_OADS_LOGIN = 'https://esar-ds.eo.esa.int/oads/access/login'
_OADS_ACCESS = 'https://esar-ds.eo.esa.int/oads/access/'
# ESA's limit for a standard account: 20 L1 products per day and collection (ESA OTF manual v2.5; no counter is shown)
_QUOTA = 20
# The largest block of MDS1 line records read and written at once (64 MiB), as the ALOS-2 and NISAR downloaders do
# (ASF._ALOS2_BLOCK_BYTES, _NISAR_BLOCK_BYTES): one Range request per block of whole records, resumed within the block after
# a drop, then one write of the block, so a scene's download holds one block in memory; a write per network read (64 kB) is
# slow on Google Colab's disks
_BLOCK_BYTES = 64 * 1024 * 1024
# the first request of a product reads this many bytes: its header through MDS1 (24,854-25,896 bytes in every product
# measured, U2a); a longer header is read again whole
_HEAD_BYTES = 65536
# the longest wait for a product that ESA processes on the request (HTTP 202 with Retry-After), seconds per scene: the
# processing took 1.7-17 min for the regular scenes and 2-3 min for the ERS extra records (U2a, U2c)
_PROCESSING_WAIT = 1800
# the wait when a 202 answer carries no Retry-After, seconds
_RETRY_AFTER = 60
# the parallel scene downloads for n_jobs=None. Measured 2026-11-23 (U2b-D2, 15 MB Range reads of three processed products):
# one connection alone 0.22 MB/s, three at once 0.20, 0.20 and 1.19 MB/s each, so the rate per connection does not drop
# with three (the full products of the same day: 0.44-0.54 MB/s, one at a time); more were not measured. The 202 processing
# runs on ESA's side, so parallel requests overlap it as well.
_JOBS = 3
# the delivered scene of a listed scene, seconds from its catalogue begin, before the product is requested (_precheck): the
# delivered 18 s scene of a regular record is its 15 s archive span 1.5 s before and 2.5 s after (U2a), and an archive span
# lasts at most 15.1 s (_archive_record), so it ends at most 17.6 s after the begin; an ERS extra record is served as kind A
# [b - 0.998, b + 16.997] or kind B [b - 1.498, e + 0.996] (U2c Q3), inside the same bounds. Measured on the 15 served
# products whose requested archive ID is cached (U2bD4_report.md): MPH SENSING_START -1.500..+0.001 s and SENSING_STOP
# +16.086..+17.592 s from the catalogue begin; the bounds hold 0.1 s more on each side
_DELIVERED = (-1.6, 17.7)
# the swath of each satellite and swath name: the look angles of its near and far edge, degrees off the geodetic nadir at the
# satellite, on the WGS84 ellipsoid (the antenna's elevation beams are fixed to the platform; the slant range of an edge moves
# with the orbit's height, so the edges are kept as angles and turned into slant ranges at each scene, _swath_ranges).
# Envisat: ESA, "Impact of the Envisat Mission Extension on SAR data" v1.0 (2010-10-12), Table 1 (the Image Mode swaths at
# the ascending node: swath width, near ground range, near slant range; the beams are those of the earlier phases, its
# section 5). The look angles are derived at the altitude the table itself implies, not the 782.4 km of its section 4 (the
# orbit's mean altitude): on a sphere of radius a = 6378.137 km with the satellite at a + H, the near slant range R and the
# near ground range g of a swath satisfy R^2 = (a + H)^2 + a^2 - 2 (a + H) a cos(g / a), which gives H = 769.66..769.67 km
# for each of the 7 swaths (the semi-major axis 7142.047 km of its section 4 is 763.9 km above the sphere, and the orbit's
# radius at the node differs from it by at most a e = 8.3 km). At that height the near slant range gives the same near look angle as the near ground
# range (to 0.01 degree), the far look angle is that of g + swath width: asin(a sin(g / a) / R), and the table's mid
# elevation angle is reproduced within 0.40 degree (IS1) to 0.02 degree (IS7). The 5 cached Envisat IS2 headers (39.7-40.6 N,
# 2007-2008, phase 2) have near 16.50..16.57 and far 23.13..23.16 degrees: 0.05-0.12 degree inside the table's 16.62 and
# 23.19. The other swaths have no cached header. ERS (IS2 only): the 13 cached ERS-1 and ERS-2 headers of 1993-2008 at
# 40.3-41.2 N: near 16.53..17.23, far 22.91..23.38 degrees (12 of them start at 17.12..17.23; the 2008 ERS-2 product of
# track 36 starts 3.4 km nearer, U2c_report_final.md). The headers' look angles: their GEOLOCATION GRID tie points of the
# first and last samples against the precise orbit (session/patches/esa_alos1 U2bD4_report.md)
_SWATH_LOOK = {'ENVISAT': {'IS1': (12.57, 19.87), 'IS2': (16.62, 23.19), 'IS3': (22.82, 27.52), 'IS4': (27.17, 31.74),
                           'IS5': (31.32, 34.32), 'IS6': (33.87, 37.15), 'IS7': (36.93, 39.24)},
               'ERS-1': {'IS2': (16.53, 23.38)}, 'ERS-2': {'IS2': (16.53, 23.38)}}
# the swath edges are widened by this many degrees of look angle on each side (about 8 km on the ground at 20 degrees). It
# covers the 0.12 degree by which the 5 Envisat IS2 headers differ from the table's edges, and a difference of the same
# size for the swaths without a header (not measured); the ERS edges are the widest measured, so the margin lies beyond them
_SWATH_MARGIN = 0.5
# the heights of the AOI in the checks before a request, as the crop takes them (utils_envisat.aoi_window), metres
_HEIGHTS = (-500.0, 9000.0)
# the slowest product transfer accepted (HTTP.iter_body), bytes per second over 60 s: a slower one is read again from where
# it stopped. ESA served 22-45 KB/s per connection for minutes on 2026-11-23 (U2b-D2: a 50 KB/s floor cut one transfer 7
# times, each new connection as slow), so the floor only catches a stalled connection
_MIN_RATE = '10KB'
# a file found by skip_exist without a request: the served scene of the ID (its MPH ABS_ORBIT, and its SENSING_START
# within this window of the ID's start, seconds; measured -0.81, -1.24, -1.31 s on the three served products of
# U2b-D2, the served scenes start 1.0-1.5 s before the catalogue's, U2a and U2c; the next scene of the orbit starts 15 s
# later, an off-grid one at least 4 s)
_SKIP_MATCH = (-3.0, 1.0)

# platform names of ESA.search, and the satellites each one means
_PLATFORMS = {'ENVISAT': ('ENVISAT',), 'ERS-1': ('ERS-1',), 'ERS-2': ('ERS-2',), 'ERS': ('ERS-1', 'ERS-2')}
_COLLECTION = {'ENVISAT': 'ASA_IMS_1P', 'ERS-1': 'SAR_IMS_1P', 'ERS-2': 'SAR_IMS_1P'}
# code prefix and file extension per satellite, and the satellite of a code prefix
_PREFIX = {'ENVISAT': 'ENV', 'ERS-1': 'ER1', 'ERS-2': 'ER2'}
_EXTENSION = {'ENV': '.N1', 'ER1': '.E1', 'ER2': '.E2'}
_SATELLITE = {prefix: satellite for satellite, prefix in _PREFIX.items()}
_SWATHS = ('IS1', 'IS2', 'IS3', 'IS4', 'IS5', 'IS6', 'IS7')
_POLARIZATIONS = ('HH', 'HV', 'VH', 'VV')

# orbit grids per (satellite, missionPhase) (session/patches/esa_alos1 U2a Q1 and Q7): the repeat period T = cycle / orbits,
# the node longitude of track 1 lam1 (degrees; None where it is not measured) and whether it is a 35-day grid (a 3-digit track
# field in the code; the other grids carry the phase letter before a 4-digit track)
_DAY = 86400.0
_T35, _T3, _T168, _T30 = 35 * _DAY / 501, 3 * _DAY / 43, 168 * _DAY / 2411, 30 * _DAY / 431
_GRIDS = {
    ('ENVISAT', '2'): (_T35, 0.204, True), ('ENVISAT', '3'): (_T30, None, False),
    ('ERS-1', 'R'): (_T35, 0.204, True), ('ERS-1', 'C'): (_T35, 0.204, True), ('ERS-1', 'G'): (_T35, 0.204, True),
    ('ERS-1', 'A'): (_T3, 7.84, False), ('ERS-1', 'B'): (_T3, 5.96, False), ('ERS-1', 'D'): (_T3, 5.98, False),
    ('ERS-1', 'E'): (_T168, 0.21, False), ('ERS-1', 'F'): (_T168, 0.25, False),
    ('ERS-2', 'A'): (_T35, 0.204, True), ('ERS-2', 'B'): (_T3, None, False),
}
# nominal circular orbits (semi-major axis m, inclination deg) and the fitted correction of the zero-Doppler frame to the
# catalogue's time from the ascending node, in frames: c0 + c1 (1 - cos u) + c2 sin u (U2a Q1, fitted on 2,412 Envisat records)
_ORBIT = {'ENVISAT': (7159.5e3, 98.55), 'ERS-1': (7153.1e3, 98.52), 'ERS-2': (7153.1e3, 98.52)}
_FRAME_FIT = (-0.389, 2.276, 1.026)
_EARTH_RATE = 2 * 3.141592653589793 / 86164.0905
_SUN_RATE = 2 * 3.141592653589793 / (365.2422 * _DAY)
# the kinds of an ERS extra record (U2c Q3, measured on 8 products; ESA serves the two kinds, the others not as a scene) are
# told apart by (begin - the catalogue's start time from the node) less the orbit's node time: 1 s for A, 0 for B
_KIND_TOLERANCE = 0.05
# margin of the AOI's time window, in frames: the model frame less the AOI's zero-Doppler frame from the precise orbit is
# Envisat -1.27..+2.37 (p05 -0.93, p95 1.58; 2 of 146 beyond the 2-frame margin, inside the delivered overlap), ERS-1 phases C
# and G -0.11..+0.78, ERS-2 -0.18..+0.73, ERS-1 phases A, B, D, E, F +0.10..+1.19 (FABLE_ESA_review.md F6, its b4_model.py)
_FRAME_MARGIN = 2.0
# a regular record starts on its grid when its start frame lies this close to a multiple of 18 frames (ERS: within 0.048)
_ON_GRID = 0.25

_ID = re.compile(r'^(?P<collection>ASA_IMS_1P|SAR_IMS_1P)(?P<centre>N[A-Z0-9]{3}|REOD)(?P<date>\d{8})_(?P<time>\d{6})_'
                 r'(?P<duration>\d{8})(?P<phase>[0-9A-Z])(?P<cycle>\d{3})_(?P<track>\d{5})_(?P<orbit>\d{5})_(?P<counter>\d{4})$')
_CODE = re.compile(r'^(?P<prefix>ENV|ER1|ER2)_(?P<track>\d{3}|[0-9A-Z]\d{4})_(?P<frame>\d{4})_(?P<swath>IS[1-7])_'
                   r'(?P<date>\d{8})_(?P<polarization>HH|HV|VH|VV)$')


def _names(value, allowed, what):
    """
    The names of a search argument: None for all, one name, a comma-separated string or a list, upper case.

    Parameters
    ----------
    value : str, list or None
        The argument.
    allowed : iterable of str
        The accepted names.
    what : str
        The argument's name for the error message.

    Returns
    -------
    list of str or None
        The names, None for None.

    Raises
    ------
    ValueError
        For a name not in allowed, or a value that is neither a string nor a list (a number, say).

    Examples
    --------
    >>> _names('is2, IS6', ('IS2', 'IS6'), 'swath')
    ['IS2', 'IS6']
    """
    if value is None:
        return None
    if not isinstance(value, (str, list, tuple, set)):
        raise ValueError(f'Unknown {what} {value!r}: use {", ".join(allowed)}.')
    items = value.split(',') if isinstance(value, str) else list(value)
    names = [str(item).strip().upper() for item in items if str(item).strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown or not names:
        raise ValueError(f'Unknown {what} {unknown or value!r}: use {", ".join(allowed)}.')
    return names


def _epoch(text):
    """
    Seconds since 1970 (UTC) of a catalogue time such as '2008-01-16T20:47:12.123Z'.

    Parameters
    ----------
    text : str
        ISO time, with or without 'Z'.

    Returns
    -------
    float

    Examples
    --------
    >>> _epoch('1970-01-01T00:00:01.500Z')
    1.5
    """
    from datetime import datetime, timezone
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _socat_page(collection, bbox, start, stop):
    """
    One SO-Cat request: the records of a collection whose footprint meets a bbox and whose acquisition begins in [start, stop].

    POST with Connection: close (the service drops kept-alive connections, PLAN.md) through a session that never reads
    ~/.netrc (utils_esa_orbit._NetrcFreeSession); the answer is read through a capped stream and its
    row count is checked against the X-ESA-SOCat-Record-Count header. The service returns every record at once: pageCount and
    the page arguments change nothing (measured 2026-11-23), so the search splits long time windows itself (_socat).

    Parameters
    ----------
    collection : str
        'ASA_IMS_1P' or 'SAR_IMS_1P'.
    bbox : tuple of float
        (minlon, minlat, maxlon, maxlat).
    start, stop : str
        Dates 'YYYY-MM-DD'.

    Returns
    -------
    list of dict
        The records, column name to text.

    Raises
    ------
    IOError
        An answer longer than _SOCAT_LIMIT, or with another row count than its header.

    Examples
    --------
    >>> _socat_page('ASA_IMS_1P', (14.0, 40.7, 14.2, 40.9), '2008-01-01', '2008-02-01')
    """
    from .HTTP import send, read_body
    data = {'service': 'SimpleOnlineCatalogue', 'version': '1.2', 'request': 'search', 'format': 'text/tab-separated-values',
            'query.footprint.minlon': f'{bbox[0]:.6f}', 'query.footprint.minlat': f'{bbox[1]:.6f}',
            'query.footprint.maxlon': f'{bbox[2]:.6f}', 'query.footprint.maxlat': f'{bbox[3]:.6f}',
            'query.beginAcquisition.start': start, 'query.beginAcquisition.stop': stop}
    headers = {'Connection': 'close', 'User-Agent': 'insardev-toolkit'}
    with _NetrcFreeSession() as session:
        with send(session.post, _SOCAT_URL.format(collection), data=data, stream=True, timeout=(30, 300),
                  headers=headers) as response:
            body = read_body(response, limit=_SOCAT_LIMIT)
            count = response.headers.get('X-ESA-SOCat-Record-Count')
    lines = [line for line in body.decode('utf-8', 'replace').splitlines() if line.strip()]
    if not lines:
        rows = []
    else:
        header = lines[0].split('\t')
        rows = [dict(zip(header, line.split('\t'))) for line in lines[1:]]
    if count is not None and int(count) != len(rows):
        raise IOError(f'SO-Cat answer of {len(rows)} rows, its header counts {count}')
    return rows


def _socat(collection, bbox, start, stop, retries=30, timeout_second=3):
    """
    The records of a collection over a bbox and time window, by one SO-Cat request per calendar year, each retried; a year
    whose answer exceeds the read limit is asked again in two halves. Records listed twice are kept once (by productId).

    Parameters
    ----------
    collection : str
        'ASA_IMS_1P' or 'SAR_IMS_1P'.
    bbox : tuple of float
        (minlon, minlat, maxlon, maxlat).
    start, stop : str
        Dates 'YYYY-MM-DD'. SO-Cat includes the stop day (the window 2008-01-15..2008-01-16 returns the records of
        2008-01-16, FABLE_ESA_review.md F6), so the yearly windows share their end days; a record listed twice is kept once,
        and the caller filters the times. start equal to stop sends no request.
    retries : int, optional
        Attempts per request, the first one included (HTTP.attempts). Default 30.
    timeout_second : float, optional
        Seconds between attempts. Default 3.

    Returns
    -------
    list of dict

    Examples
    --------
    >>> _socat('SAR_IMS_1P', (14.0, 40.7, 14.2, 40.9), '1996-01-01', '1997-01-01')
    """
    import time
    from datetime import date, timedelta
    from .HTTP import final, attempts
    first, last = date.fromisoformat(start), date.fromisoformat(stop)
    windows = []
    while first < last:
        end = min(date(first.year + 1, 1, 1), last)
        windows.append((first, end))
        first = end
    records = {}
    while windows:
        a, b = windows.pop(0)
        n = attempts(retries)
        for attempt in range(n):
            try:
                rows = _socat_page(collection, bbox, a.isoformat(), b.isoformat())
                break
            except IOError as e:
                if 'longer than' in str(e) and (b - a).days > 1:
                    middle = a + timedelta(days=(b - a).days // 2)
                    windows[:0] = [(a, middle), (middle, b)]
                    rows = None
                    break
                error = e
            except Exception as e:
                error = e
            stop_now = final(error)
            print(f'ESA catalogue search attempt {attempt + 1} failed{" (not retried)" if stop_now else ""}: {error}')
            if stop_now or attempt + 1 == n:
                raise error
            time.sleep(timeout_second)
        for row in rows or []:
            records.setdefault(row.get('productId', ''), row)
    records.pop('', None)
    return list(records.values())


def _delivery_record(row):
    """
    Whether a catalogue record is a delivery record by its number formats: one of the records ESA created in 2025 for the
    copies it delivered on request (the 18 s scenes and the short scenes of the same orbits), partial and duplicating archive
    scenes. Its orbitNumber is zero-padded to 6 digits ('023029') and its wrsLongitudeGrid to 4 digits ('0129'), or its
    productId has the centre field REOD ('SAR_IMS_1PREOD20110505_...'; 1,271 ERS and 217 Envisat records of the cached
    catalogue, all created in 2025 and mostly 28-30 s long). The archive records have none of these (their orbit numbers have
    at most 5 digits and no leading zero, and the 4-digit tracks of the ERS-1 168-day phases E and F, such as '1672', have no
    leading zero either). Not every delivery record has a format mark: 7 of the cached catalogue (ERS-2 orbit 11061 of
    1997-06-01, created in 2025-09, 16-18 s) have none; the second mark, the orbit's node time (_node_marked), finds them.

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    bool

    Examples
    --------
    >>> _delivery_record({'orbitNumber': '023029', 'wrsLongitudeGrid': '0129'})
    True
    >>> _delivery_record({'orbitNumber': '83866', 'wrsLongitudeGrid': '29', 'productId': 'SAR_IMS_1PREOD20110505_211439_00000028B189_00029_83866_0000'})
    True
    >>> _delivery_record({'orbitNumber': '16593', 'wrsLongitudeGrid': '1672', 'productId': 'SAR_IMS_1PNESA19940503_211738_00000015E012_01672_16593_0000'})
    False
    """
    track = row['wrsLongitudeGrid']
    return (row['orbitNumber'].startswith('0') or (len(track) >= 4 and track.startswith('0'))
            or row.get('productId', '')[10:14] == 'REOD')


def _archive_record(row):
    """
    Whether a catalogue record is an archive record of a scene: a record in the archive's format (not a delivery record,
    _delivery_record), whose productId has a duration field of at most '00000015' and whose acquisition lasts at most 15.1 s (a
    shorter scene ends a data take: the Envisat records of 13 s are legitimate scenes). The delivery records, the 18 s scenes
    and short scenes inside them (36 ERS records of 3-15 s pass the duration rule), are never listed. The year of the record
    plays no part: 1,278 ERS and 217 Envisat records in the archive's format updated in 2025 last 16-30 s and fail the duration
    rule.

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    bool

    Examples
    --------
    >>> _archive_record({'productId': 'SAR_IMS_1PNESA19960227_211738_00000015G154_00129_24165_0000', 'orbitNumber': '24165',
    ...                  'wrsLongitudeGrid': '129',
    ...                  'beginAcquisition': '1996-02-27T21:17:38.000Z', 'endAcquisition': '1996-02-27T21:17:53.090Z'})
    True
    """
    if _delivery_record(row):
        return False
    try:
        field = int(row['productId'][30:38])
    except ValueError:
        return False
    duration = _epoch(row['endAcquisition']) - _epoch(row['beginAcquisition'])
    return field <= 15 and duration <= 15.1


def _span_matches(row):
    """
    Whether the time from the ascending node of a record spans its acquisition (completion - start = end - begin within 0.05 s):
    true for a regular scene, false for an ERS extra record (1 s longer).

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    bool or None
        None for a record without readable times from the node.

    Examples
    --------
    >>> _span_matches({'startTimeFromAscendingNode': '679042', 'completionTimeFromAscendingNode': '694132',
    ...                'beginAcquisition': '2002-11-13T20:47:33.390Z', 'endAcquisition': '2002-11-13T20:47:48.480Z'})
    True
    """
    try:
        span = (float(row['completionTimeFromAscendingNode']) - float(row['startTimeFromAscendingNode'])) / 1000
    except ValueError:
        return None
    return abs(span - (_epoch(row['endAcquisition']) - _epoch(row['beginAcquisition']))) < 0.05


def _orbit_key(row):
    """
    The orbit of a record: (satellite, orbit number as an integer).

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    tuple

    Examples
    --------
    >>> _orbit_key({'productId': 'ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000', 'orbitNumber': '030744'})
    ('ENVISAT', 30744)
    """
    return _platform(row), int(row['orbitNumber'])


def _node(row):
    """
    The ascending node time a record implies: its begin less its time from the node (startTimeFromAscendingNode), seconds
    since 1970; None for a record without a readable time from the node.

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    float or None

    Examples
    --------
    >>> _node({'beginAcquisition': '1970-01-01T00:11:20.000Z', 'startTimeFromAscendingNode': '679000'})
    1.0
    """
    try:
        return _epoch(row['beginAcquisition']) - float(row['startTimeFromAscendingNode']) / 1000
    except ValueError:
        return None


def _node_marked(rows):
    """
    The second mark of a delivery record, independent of the number formats (_delivery_record): its node time (_node). The
    archive records of an orbit share one node time: the regular scenes within 1 ms (6,758 cached orbits), the ERS extra
    records 1 s (kind A) or 0 s (kind B) after it (U2c Q3). The delivery records do not: 4,915 of the 4,936 cached ones
    differ from it by more than 0.1 s, the 7 without a format mark by 1273-1333 s (FABLE_ESA_review.md F4). The orbit's node
    time is the median over its archive records that span their acquisition (_archive_record, _span_matches), or, for an
    orbit without one, the least over its other archive records (as _extras takes it). A record is marked when its node time
    differs from its orbit's by neither 0 nor 1 s (within _KIND_TOLERANCE).

    Parameters
    ----------
    rows : list of dict
        SO-Cat records (any satellite and orbit).

    Returns
    -------
    set of str
        The productIds of the marked records (the records marked by the formats are left to _delivery_record).

    Examples
    --------
    >>> _node_marked([])
    set()
    """
    import statistics
    regular, other = {}, {}
    for row in rows:
        node = _node(row)
        if node is None or not _archive_record(row):
            continue
        (regular if _span_matches(row) else other).setdefault(_orbit_key(row), []).append(node)
    marked = set()
    for row in rows:
        key, node = _orbit_key(row), _node(row)
        if node is None or _delivery_record(row) or not (key in regular or key in other):
            continue
        reference = statistics.median(regular[key]) if key in regular else min(other[key])
        shift = node - reference
        if abs(shift) >= _KIND_TOLERANCE and abs(shift - 1) >= _KIND_TOLERANCE:
            marked.add(row['productId'])
    return marked


def _dedupe(rows):
    """
    One record per group of records of one orbit whose starts lie within 0.05 s of each other (U2a E4): the latest recordLastUpdate,
    then the smallest productId.

    Parameters
    ----------
    rows : list of dict
        SO-Cat records (any satellite and orbit).

    Returns
    -------
    list of dict
        The kept records, in time order per orbit.

    Examples
    --------
    >>> _dedupe([])
    []
    """
    orbits = {}
    for row in rows:
        orbits.setdefault(_orbit_key(row), []).append(row)
    out = []
    for orbit in orbits.values():
        orbit.sort(key=lambda r: (_epoch(r['beginAcquisition']), r['productId']))
        groups = [[orbit[0]]]
        for row in orbit[1:]:
            if _epoch(row['beginAcquisition']) - _epoch(groups[-1][-1]['beginAcquisition']) < 0.05:
                groups[-1].append(row)
            else:
                groups.append([row])
        for group in groups:
            latest = max(r['recordLastUpdate'] for r in group)
            out.append(sorted((r for r in group if r['recordLastUpdate'] == latest), key=lambda r: r['productId'])[0])
    return out


def _regular(rows):
    """
    The regular scenes of catalogue records (U2a E3, E4): the archive records (_archive_record) whose time from the ascending node spans their
    acquisition (_span_matches; the ERS extra records span 1 s more) and that carry their orbit's node time (not _node_marked), then
    one record per group of records of one orbit whose starts lie within 0.05 s of each other (_dedupe). Envisat has neither extra
    records nor such groups.

    Parameters
    ----------
    rows : list of dict
        SO-Cat records (any satellite and orbit).

    Returns
    -------
    list of dict
        The regular records, in time order per orbit.

    Examples
    --------
    >>> _regular([])
    []
    """
    marked = _node_marked(rows)
    return _dedupe([row for row in rows if _archive_record(row) and _span_matches(row) and row['productId'] not in marked])


def _extras(rows, regular):
    """
    The ERS extra records of catalogue records: the archive records (_archive_record) whose time from the ascending node spans 1 s more than
    their acquisition, one per group of records that start together (_dedupe, U2a E4), of the kind A or B (U2c Q3). The ascending
    node time of the orbit is the median of (begin - start from the node) over its regular records; an extra record of kind A
    begins 1 s after the start it carries, one of kind B at it (within 0.05 s), the others are not served as a scene and are
    dropped. An orbit without a regular record takes the least of its extras' values (U2c). Each record is returned as a copy with
    '_kind' ('A' or 'B') and '_span': its archive span, the acquisition begin..end, in seconds after the ascending node (the span
    ESA serves for its URL holds it: kind A [b - 0.998, b + 16.997], kind B [b - 1.498, e + 0.996], U2c Q3).

    Parameters
    ----------
    rows : list of dict
        SO-Cat records (any satellite and orbit).
    regular : list of dict
        The regular records of the same catalogue answer (_regular), for the node time of each orbit.

    Returns
    -------
    list of dict
        The extra records.

    Examples
    --------
    >>> _extras([], [])
    []
    """
    import statistics
    nodes = {}
    for row in regular:
        nodes.setdefault(_orbit_key(row), []).append(_epoch(row['beginAcquisition']) - float(row['startTimeFromAscendingNode']) / 1000)
    candidates = _dedupe([row for row in rows if _archive_record(row) and _span_matches(row) is False])
    starts = {}
    for row in candidates:
        starts.setdefault(_orbit_key(row), []).append(_epoch(row['beginAcquisition']) - float(row['startTimeFromAscendingNode']) / 1000)
    out = []
    for row in candidates:
        key = _orbit_key(row)
        node = statistics.median(nodes[key]) if key in nodes else min(starts[key])
        b, e = _epoch(row['beginAcquisition']), _epoch(row['endAcquisition'])
        shift = (b - float(row['startTimeFromAscendingNode']) / 1000) - node
        if abs(shift - 1) < _KIND_TOLERANCE:
            kind = 'A'
        elif abs(shift) < _KIND_TOLERANCE:
            kind = 'B'
        else:
            continue
        out.append(dict(row, _kind=kind, _span=(b - node, e - node)))
    return out


def _parseable(rows):
    """
    Split catalogue records into those whose productId has the known fields and a real calendar date and time (ESA._parse_id reads
    it) and those that have not: a productId of another format, and one of the format with an impossible date or time (month 13,
    hour 25). ESA.search leaves the others out with a WARNING naming them, so that one odd record never stops a search.

    Parameters
    ----------
    rows : list of dict
        SO-Cat records.

    Returns
    -------
    tuple
        (kept, left_out): two lists of records.

    Examples
    --------
    >>> _parseable([{'productId': 'ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000'}, {'productId': 'x'}])[1]
    [{'productId': 'x'}]
    >>> _parseable([{'productId': 'ASA_IMS_1PNESA20081316_254713_000000152065_00129_30744_0000'}])[0]
    []
    """
    kept, left_out = [], []
    for row in rows:
        try:
            ESA._parse_id(row.get('productId', ''))
        except ValueError:
            left_out.append(row)
        else:
            kept.append(row)
    return kept, left_out


def _platform(row):
    """
    The satellite of a catalogue record: 'ENVISAT', 'ERS-1' or 'ERS-2' (platformSerialIdentifier for ERS, else the productId).

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    str

    Examples
    --------
    >>> _platform({'productId': 'SAR_IMS_1PNESA19960228_211738_00000015A009_00129_04492_0000', 'platformSerialIdentifier': '2'})
    'ERS-2'
    """
    serial = row.get('platformSerialIdentifier', '')
    if row['productId'].startswith('SAR') and serial in ('1', '2'):
        return f'ERS-{serial}'
    return ESA._parse_id(row['productId'])['platform']


def _slot(row, T):
    """
    The grid slot of a regular record and the frame of its code. Regular scenes start on frames 18 k of the 7200 frames of an
    orbit (frame = 7200 t / T, t the time from the ascending node), fixed to the node within 1.2 ms on the 35-day grids and
    21 ms on the 3-day grids over 1991-2011 (cached records, U2b-S); the frame of the slot is its centre, 18 k + 9, the same
    code on every date. On the ground the Envisat grid moves with the season: the zero-Doppler frame of one point of track 129
    runs 748.3..749.8 through each year, with single dates at 746.70 and 750.24 (3.5 frames, 20 km, over 39 dates; 6 tracks:
    median peak-to-peak 1.6 frames), so an AOI near a slot edge is held by slot k on some dates and k + 1 on others; the ERS
    grids are stable within 0.09 (ERS-1) and 0.13 (ERS-2) frames median, 0.44 at most (FABLE_ESA_review.md F6). For an Envisat scene shifted off the grid (at a data-take edge) it is the slot of its centre (_select may name
    such a scene by another slot it overlaps).

    Parameters
    ----------
    row : dict
        A regular SO-Cat record.
    T : float
        The orbit period of its grid, seconds.

    Returns
    -------
    tuple
        (k, frame, on_grid): the slot number, 18 k + 9 and whether the record starts on the grid (within _ON_GRID frames).

    Examples
    --------
    >>> _slot({'startTimeFromAscendingNode': '679042', 'completionTimeFromAscendingNode': '694132'}, 35 * 86400 / 501)
    (45, 819, True)
    """
    start = 7200 * float(row['startTimeFromAscendingNode']) / 1000 / T
    centre = 7200 * (float(row['startTimeFromAscendingNode']) + float(row['completionTimeFromAscendingNode'])) / 2000 / T
    on_grid = abs(start - 18 * round(start / 18)) < _ON_GRID
    k = round(start / 18) if on_grid else int(centre // 18)
    return k, 18 * k + 9, on_grid


def _gain(start, stop, spans):
    """
    The length of the interval [start, stop] that lies outside the union of the spans.

    Parameters
    ----------
    start, stop : float
        The interval.
    spans : list of tuple
        (start, stop) intervals.

    Returns
    -------
    float
        0.0 for an empty interval (stop <= start).

    Examples
    --------
    >>> _gain(0.0, 10.0, [(-1.0, 4.0), (6.0, 11.0)])
    2.0
    """
    if stop <= start:
        return 0.0
    total, cursor = 0.0, start
    for a, b in sorted(spans):
        if a > cursor:
            total += min(a, stop) - cursor
        cursor = max(cursor, b)
        if cursor >= stop:
            return total
    return total + (stop - cursor)


def _centre_frame(centre, used):
    """
    The frame of a scene that no slot is free for: its own centre frame, round(7200 t / T), or, when a listed scene of the orbit
    uses that frame, the nearest unused integer frame on the side of the slot centre where the scene's centre lies.

    Parameters
    ----------
    centre : float
        The centre of the scene's catalogue span, in frames (7200 t / T).
    used : set of int
        The frames of the scenes listed so far.

    Returns
    -------
    int

    Examples
    --------
    >>> _centre_frame(2769.88, {2770, 2771})
    2772
    """
    frame, step = round(centre), (1 if centre >= 18 * (centre // 18) + 9 else -1)
    while frame in used:
        frame += step
    return frame


def _select(rows, window, T):
    """
    The scenes of one orbit, swath and polarization that an AOI needs, and the frames of their codes. A scene reaches the AOI
    when its archive span meets the AOI's time window: its catalogue span for a regular scene, its acquisition (_extras, '_span')
    for an ERS extra record; the overlap ESA adds to the delivered scene plays no part (the module docstring).
    - Every reaching regular scene on the grid is listed with the frame of its slot (the slot centre 18 k + 9), the earliest one when
      two scenes on the grid start in one slot (overlapping Envisat data takes, 55 ms apart).
    - Then the other reaching scenes (regular scenes off the grid, at a data-take edge, or the later of such a pair, and the extra
      records) are listed one at a time: the one that adds the most window length that no listed scene covers (recomputed after each
      listing; a tie goes to a regular scene, then to the earlier start, then to the smaller productId), until none adds any. The
      rest are redundant. The result does not depend on the order of rows.
    - A listed scene of these takes the slot of its centre, or, when a listed scene holds that frame, the nearest slot its
      catalogue span overlaps that no listed scene holds. When there is none, it takes its own centre frame, round(7200 t / T),
      or the nearest unused integer frame beyond it on the side of the slot centre (the scene is returned as renamed).

    Parameters
    ----------
    rows : list of dict
        Regular SO-Cat records (_regular) of one orbit, swath and polarization, and its extra records (_extras).
    window : tuple of float
        (start, stop) of the AOI, seconds after the ascending node.
    T : float
        The orbit period of the grid, seconds.

    Returns
    -------
    tuple
        (listed, redundant, renamed): listed is a list of (record, frame) in time order; redundant is the list of the productIds
        of the reaching scenes that are not listed; renamed is the list of the productIds of the listed scenes named by their own
        centre frame, not by a slot centre.

    Examples
    --------
    >>> _select([{'productId': 'a', 'startTimeFromAscendingNode': '679042', 'completionTimeFromAscendingNode': '694132'}],
    ...         (680.0, 690.0), 35 * 86400 / 501)[0][0][1]
    819
    """
    def seconds(row):
        return float(row['startTimeFromAscendingNode']) / 1000, float(row['completionTimeFromAscendingNode']) / 1000

    def archive(row):
        return row['_span'] if '_span' in row else seconds(row)

    ordered = sorted(rows, key=lambda row: (seconds(row)[0], row['productId']))
    listed, spans, used, rest = [], [], set(), []
    for row in ordered:
        a, b = archive(row)
        if a > window[1] or b < window[0]:
            continue
        _, frame, on_grid = _slot(row, T)
        if on_grid and frame not in used and '_kind' not in row:
            listed.append((row, frame))
            spans.append((a, b))
            used.add(frame)
        else:
            rest.append(row)
    renamed = []
    while rest:
        # the gains are rounded to the catalogue's millisecond; the first of equal gains is the earliest scene
        gains = [round(_gain(max(archive(row)[0], window[0]), min(archive(row)[1], window[1]), spans), 6) for row in rest]
        best = max(range(len(rest)), key=lambda i: (gains[i], '_kind' not in rest[i]))
        if gains[best] <= 0:
            break
        row = rest.pop(best)
        s, c = (7200 * value / T for value in seconds(row))
        centre = (s + c) / 2
        free = [j for j in range(int(s // 18), -int(-c // 18)) if 18 * j + 9 not in used]
        if free:
            frame = 18 * min(free, key=lambda j: (abs(18 * j + 9 - centre), j)) + 9
        else:
            frame = _centre_frame(centre, used)
            renamed.append(row['productId'])
        listed.append((row, frame))
        spans.append(archive(row))
        used.add(frame)
    listed.sort(key=lambda item: (seconds(item[0])[0], item[0]['productId']))
    return listed, [row['productId'] for row in rest], renamed


def _check_aoi(geometry):
    """
    Refuse an AOI that the search does not handle: one that crosses the 180 degree meridian (longitudes beyond -180..180 or a
    longitude extent above 180 degrees), or one with latitudes beyond -90..90.

    Parameters
    ----------
    geometry : shapely geometry
        The AOI, longitude and latitude.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For an AOI that crosses 180 degrees or lies outside the longitude and latitude ranges.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> _check_aoi(box(14.0, 40.7, 14.2, 40.9))
    """
    minlon, minlat, maxlon, maxlat = geometry.bounds
    if minlon < -180 or maxlon > 180 or maxlon - minlon > 180:
        raise ValueError('An AOI crossing 180 degrees of longitude is not supported: search each side separately.')
    if minlat < -90 or maxlat > 90:
        raise ValueError(f'AOI latitudes {minlat}..{maxlat} are outside -90..90.')


def _model_frames(lons, lats, track, direction, platform, phase):
    """
    Frames (7200 t / T, unrounded) of ground points on a track of the nominal orbit (U2a Q1): the zero-Doppler time t after the
    ascending node of each point, on a circular sun-synchronous orbit with node longitude lam1 - (track - 1) (omega_E -
    omega_sun) T, less the fitted correction c0 + c1 (1 - cos u) + c2 sin u to the catalogue's time.

    Parameters
    ----------
    lons, lats : array-like
        Points, degrees.
    track : int
        wrsLongitudeGrid.
    direction : str
        'ASCENDING' or 'DESCENDING' (the pass that holds the points).
    platform : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    phase : str
        missionPhase.

    Returns
    -------
    numpy.ndarray or None
        The frames; None for a grid whose lam1 is not measured.

    Examples
    --------
    >>> _model_frames([14.125], [40.825], 129, 'ASCENDING', 'ENVISAT', '2')
    array([811.40637747])
    """
    import numpy as np
    T, lam1, _ = _GRIDS[(platform, phase)]
    if lam1 is None:
        return None
    a, inc = _ORBIT[platform]
    lons, lats = np.radians(np.asarray(lons, float)), np.radians(np.asarray(lats, float))
    rate = _EARTH_RATE - _SUN_RATE
    node = np.radians(lam1) - (track - 1) * rate * T
    ci, si = np.cos(np.radians(inc)), np.sin(np.radians(inc))
    # WGS84 points on the ellipsoid
    e2 = 0.00669437999014
    n = 6378137.0 / np.sqrt(1 - e2 * np.sin(lats) ** 2)
    points = np.stack([n * np.cos(lats) * np.cos(lons), n * np.cos(lats) * np.sin(lons), n * (1 - e2) * np.sin(lats)])

    def orbit(t):
        u = 2 * np.pi * t / T
        ln = node - rate * t
        return a * np.stack([np.cos(u) * np.cos(ln) - np.sin(u) * ci * np.sin(ln),
                             np.cos(u) * np.sin(ln) + np.sin(u) * ci * np.cos(ln), np.sin(u) * si])

    def doppler(t):
        return np.sum((points - orbit(t)) * (orbit(t + 0.01) - orbit(t - 0.01)) / 0.02, axis=0)

    u0 = np.arcsin(np.clip(np.sin(lats) / si, -1, 1))
    if direction.upper().startswith('D'):
        u0 = np.pi - u0
    t = (u0 % (2 * np.pi)) / (2 * np.pi) * T
    for _ in range(20):
        step = doppler(t) / ((doppler(t + 0.05) - doppler(t - 0.05)) / 0.1)
        t = t - step
        if np.all(np.abs(step) < 1e-5):
            break
    u = 2 * np.pi * t / T
    c0, c1, c2 = _FRAME_FIT
    return 7200 * t / T - (c0 + c1 * (1 - np.cos(u)) + c2 * np.sin(u))


def _aoi_window(geometry, track, direction, platform, phase):
    """
    The AOI's time window after the ascending node on a track, seconds: the model frames (_model_frames) of the vertices of the
    AOI's convex hull, widened by _FRAME_MARGIN frames on each side.

    Parameters
    ----------
    geometry : shapely geometry
        The AOI, longitude and latitude.
    track : int
        wrsLongitudeGrid.
    direction : str
        'ASCENDING' or 'DESCENDING'.
    platform : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    phase : str
        missionPhase.

    Returns
    -------
    tuple or None
        (start, stop) seconds after the ascending node; None for a grid whose lam1 is not measured.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> _aoi_window(box(14.0, 40.7, 14.2, 40.9), 129, 'ASCENDING', 'ENVISAT', '2')
    """
    hull = geometry.convex_hull
    coords = list(hull.exterior.coords) if hull.geom_type == 'Polygon' else list(hull.coords)
    frames = _model_frames([c[0] for c in coords], [c[1] for c in coords], track, direction, platform, phase)
    if frames is None:
        return None
    T = _GRIDS[(platform, phase)][0]
    return (float(frames.min()) - _FRAME_MARGIN) * T / 7200, (float(frames.max()) + _FRAME_MARGIN) * T / 7200


def _footprint(row):
    """
    The footprint polygon of a catalogue record ('lat lon lat lon ...' in the record) in longitude and latitude.

    Parameters
    ----------
    row : dict
        A SO-Cat record.

    Returns
    -------
    shapely.geometry.Polygon

    Examples
    --------
    >>> _footprint({'footprint': '41.18 13.82 40.30 13.64 40.21 14.38 41.09 14.57 41.18 13.82'}).bounds
    (13.64, 40.21, 14.57, 41.18)
    """
    from shapely.geometry import Polygon
    values = [float(v) for v in row['footprint'].split()]
    return Polygon(list(zip(values[1::2], values[0::2])))


def _query_bbox(geometry, collection):
    """
    The bbox of the catalogue query of an AOI: its bounds widened by _SOCAT_WIDEN degrees of latitude (_SOCAT_WIDEN_ERS for the
    ERS collection) and the same distance in longitude (scaled by the latitude), within -180..180 and -90..90.

    Parameters
    ----------
    geometry : shapely geometry
        The AOI, longitude and latitude.
    collection : str
        'ASA_IMS_1P' or 'SAR_IMS_1P'.

    Returns
    -------
    tuple of float
        (minlon, minlat, maxlon, maxlat).

    Examples
    --------
    >>> from shapely.geometry import box
    >>> _query_bbox(box(14.0, 40.7, 14.2, 40.9), 'ASA_IMS_1P')[1]
    40.400000000000006
    """
    import math
    minlon, minlat, maxlon, maxlat = geometry.bounds
    widen_lon = _SOCAT_WIDEN / max(math.cos(math.radians(max(abs(minlat), abs(maxlat)))), 0.1)
    widen_lat = _SOCAT_WIDEN_ERS if collection == 'SAR_IMS_1P' else _SOCAT_WIDEN
    return (max(minlon - widen_lon, -180.0), max(minlat - widen_lat, -90.0),
            min(maxlon + widen_lon, 180.0), min(maxlat + widen_lat, 90.0))


def _listing(records, geometry, wanted):
    """
    The rows of ESA.search over an AOI from the catalogue records of its query: the regular scenes and the ERS extra records
    (_regular, _extras) that wanted keeps, grouped by orbit, swath and polarization; the orbits whose footprints meet the AOI
    widened by 0.1 degree; per orbit the AOI's time window (_aoi_window, or the span of the scenes whose footprint meets the AOI
    widened by 0.2 degree where the grid's node longitude is not measured) and the scenes it needs with the frames of their
    codes (_select). ESA.download takes the codes of its scenes from it, with the same AOI, so that its files carry the codes of
    the search's rows.

    Parameters
    ----------
    records : list of dict
        SO-Cat records with a readable productId (_parseable).
    geometry : shapely geometry
        The AOI, longitude and latitude.
    wanted : callable
        A record -> bool filter (satellite, time, direction, swath, polarization).

    Returns
    -------
    tuple
        (rows, notes): rows is the list of the row dicts (sceneName, code, platform, track, phase, frame, swath, polarization,
        flightDirection, orbitNumber, startTime, kind, geometry), in the order of the groups; notes is the list of the WARNING
        lines (a scene named by its own centre frame, the grids with no orbit model, the grids whose node longitude is not
        measured), in the order ESA.search prints them.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> _listing([], box(14.0, 40.7, 14.2, 40.9), lambda row: True)
    ([], [])
    """
    import shapely

    across = geometry.buffer(0.1)
    near = geometry.buffer(0.2)
    every = _regular(records)
    regular = [row for row in every if wanted(row)] + [row for row in _extras(records, every) if wanted(row)]

    groups = {}
    for row in regular:
        groups.setdefault((_platform(row), int(row['orbitNumber']), row['swathIdentifier'],
                           row['polarisationChannels'].strip().upper()), []).append(row)
    windows, unknown, rows, notes = {}, set(), [], []
    for (sat, orbit, sw, pol), group in sorted(groups.items()):
        footprints = {row['productId']: _footprint(row) for row in group}
        # the convex hull of the orbit's footprints: it spans the along-track gaps that the trimmed footprints of extra records leave
        vertices = [point for f in footprints.values() for point in f.exterior.coords]
        if not shapely.geometry.MultiPoint(vertices).convex_hull.intersects(across):
            continue
        # one orbit has one mission phase, track and direction
        phase, track, pass_ = group[0]['missionPhase'], int(group[0]['wrsLongitudeGrid']), group[0]['orbitDirection']
        if (sat, phase) not in _GRIDS:
            unknown.add((sat, phase))
            continue
        T = _GRIDS[(sat, phase)][0]
        grid = (sat, phase, track, pass_)
        if grid not in windows:
            windows[grid] = _aoi_window(geometry, track, pass_, sat, phase)
        window = windows[grid]
        if window is None:
            # no orbit model: the regular scenes whose footprint meets the widened AOI, and the span of their catalogue times
            # (the extra records, whose footprints are trimmed, stay candidates)
            group = [row for row in group if '_kind' in row or footprints[row['productId']].intersects(near)]
            meeting = [row for row in group if '_kind' not in row]
            if not meeting:
                continue
            window = (min(float(row['startTimeFromAscendingNode']) for row in meeting) / 1000,
                      max(float(row['completionTimeFromAscendingNode']) for row in meeting) / 1000)
        listed, _, renamed = _select(group, window, T)
        for row, frame in listed:
            if row['productId'] in renamed:
                notes.append(f'WARNING: ESA scene {row["productId"]} is {"an extra record" if "_kind" in row else "off the grid"} and '
                             f'every slot it overlaps holds another listed scene of orbit {orbit}: it is named by its own centre frame '
                             f'{frame}, not by a slot centre.')
            begin = row['beginAcquisition']
            rows.append(dict(sceneName=row['productId'],
                             code=ESA._code(sat, phase, track, frame, sw, begin[:10].replace('-', ''), pol),
                             platform=sat, track=track, phase=phase, frame=frame, swath=sw, polarization=pol,
                             flightDirection=pass_, orbitNumber=orbit, startTime=begin,
                             kind='extra' if '_kind' in row else ('regular' if _slot(row, T)[2] else 'offgrid'),
                             geometry=footprints[row['productId']]))
    if unknown:
        notes.append(f'WARNING: no orbit grid known for {sorted(unknown)}: their scenes are not listed.')
    lam1 = sorted({(g[0], g[1]) for g, w in windows.items() if w is None})
    if lam1:
        notes.append(f'WARNING: the node longitude of {lam1} is not measured: their AOI window is the span of the scenes whose '
                     f'footprint meets the AOI widened by 0.2 degree.')
    return rows, notes


def _bbox(bbox):
    """
    The bbox of ESA.download: None, or four numbers (west, south, east, north) in WGS84 degrees with west < east and
    south < north, not crossing 180 degrees (_check_aoi).

    Parameters
    ----------
    bbox : tuple of float or None
        The bbox.

    Returns
    -------
    tuple of float or None
        (west, south, east, north) as floats.

    Raises
    ------
    ValueError
        A bbox of another form, an empty one, or one beyond the longitude and latitude ranges.

    Examples
    --------
    >>> _bbox((14.34, 40.323, 14.4, 40.363))
    (14.34, 40.323, 14.4, 40.363)
    """
    import math
    from shapely.geometry import box
    if bbox is None:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        values = ()
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f'ESA bbox {bbox!r}: expected (west, south, east, north) in degrees, west < east and south < north.')
    _check_aoi(box(*values))
    return values


class _ESASession(_NetrcFreeSession):
    """
    The HTTP session of ESA.download: its credentials are those given to ESA(username, password), never a ~/.netrc entry
    (_NetrcFreeSession: no netrc lookup for a request or a redirect; the sign-in redirects to eoiam-idp.eo.esa.int). It
    sends a browser's User-Agent. The environment's proxies and CA bundle apply as usual.
    """

    def __init__(self):
        """
        A new session with no credentials.

        Examples
        --------
        >>> _ESASession().auth is not None
        True
        """
        super().__init__()
        self.headers['User-Agent'] = 'Mozilla/5.0 (insardev-toolkit)'


def _login(username, password, retries=30, timeout_second=3):
    """
    Sign in to the ESA (A)SAR Online Dissemination service with an ESA EO Sign In account, as a browser does: the login page
    of the service, the account form of eoiam-idp.eo.esa.int, then the SAML answer forms back to the service. The account is
    sent to the sign-in form only and never printed.

    Parameters
    ----------
    username, password : str
        The ESA EO Sign In account.
    retries : int, optional
        Attempts of the sign-in, the first one included (HTTP.attempts). Default 30.
    timeout_second : float, optional
        Seconds between attempts. Default 3.

    Returns
    -------
    _ESASession
        The signed-in session.

    Raises
    ------
    ValueError
        The service does not accept the account.

    Examples
    --------
    >>> _login('user', 'pass')
    """
    import html
    import time
    from urllib.parse import parse_qs, urljoin, urlparse
    from .HTTP import attempts, final, send

    n = attempts(retries)
    for attempt in range(n):
        session = _ESASession()
        try:
            r = send(session.get, _OADS_LOGIN, timeout=(30, 120))
            body = r.text[:200000]
            key = parse_qs(urlparse(r.url).query).get('sessionDataKey', [None])[0]
            form = re.search(r'<form[^>]+action="([^"]+)"', body)
            r = send(session.post, urljoin(r.url, html.unescape(form.group(1)) if form else '../commonauth'), timeout=(30, 120),
                     data={'username': username, 'password': password, 'sessionDataKey': key, 'tocommonauth': 'true'})
            body = r.text[:200000]
            # the SAML answer forms, posted on as a browser does
            for _ in range(4):
                found = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', body, re.S)
                if not found or 'SAMLResponse' not in found.group(2):
                    break
                fields = {}
                for tag in re.findall(r'<input[^>]*>', found.group(2)):
                    name = re.search(r'name\s*=\s*[\'"]([^\'"]+)[\'"]', tag)
                    value = re.search(r'value\s*=\s*[\'"]([^\'"]*)[\'"]', tag)
                    if name:
                        fields[name.group(1)] = html.unescape(value.group(1)) if value else ''
                r = send(session.post, html.unescape(found.group(1)), data=fields, timeout=(30, 120))
                body = r.text[:200000]
            page = send(session.get, _OADS_ACCESS, timeout=(30, 120)).text[:300000]
        except Exception as e:
            session.close()
            stop = final(e)
            print(f'ESA sign-in attempt {attempt + 1} failed{" (not retried)" if stop else ""}: {e}')
            if stop or attempt + 1 == n:
                raise
            time.sleep(timeout_second)
            continue
        if 'Not signed in' in page:
            session.close()
            raise ValueError('ESA EO Sign In refused the account of ESA(username, password).')
        return session


def _complete(path):
    """
    The header of a complete crop file (ESA.download), read from the file: its MPH TOT_SIZE equals the file size and its 19th
    DSD is the precise-orbit dataset (utils_esa_orbit.ORBIT_DS_NAME); None for a missing, short or malformed file.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    dict or None
        As utils_envisat.read_header.

    Examples
    --------
    >>> _complete('data/ENV_129_0801_IS2/ENV_129_0801_IS2_20080116_VV.N1')
    """
    import os
    from . import utils_envisat
    from .utils_esa_orbit import ORBIT_DS_NAME
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(min(size, 1 << 20))
        header = utils_envisat.read_header(head)
    except (OSError, ValueError):
        return None
    dsds = header['dsds']
    if header['mph']['TOT_SIZE'] != size or len(dsds) != 19 or dsds[18]['name'] != ORBIT_DS_NAME:
        return None
    return header


class _NotServed(Exception):
    """A product that ESA does not serve within _PROCESSING_WAIT (HTTP 202 on every request)."""


class _Block:
    """
    The sink of _ranged for one block of MDS1 records: a buffer allocated once per scene and filled in place, so a block is
    held once in memory and written to the file in one call (_fetch_scene). ``len`` counts the bytes read into it, ``+=``
    puts the next ones after them.

    Parameters
    ----------
    buffer : bytearray
        The scene's block buffer, at least as long as the block.
    size : int
        The block's length in bytes.

    Examples
    --------
    >>> block = _Block(bytearray(8), 4)
    >>> block += b'ab'
    >>> len(block), bytes(block.view())
    (2, b'ab')
    """

    def __init__(self, buffer, size):
        self.buffer, self.size, self.count = memoryview(buffer)[:size], size, 0

    def __len__(self):
        return self.count

    def __iadd__(self, data):
        self.buffer[self.count:self.count + len(data)] = data
        self.count += len(data)
        return self

    def view(self):
        """
        The bytes read so far, a view of the buffer (no copy).

        Returns
        -------
        memoryview

        Examples
        --------
        >>> bytes(_Block(bytearray(4), 4).view())
        b''
        """
        return self.buffer[:self.count]


def _ranged(task, start, stop, sink, state):
    """
    Read the bytes [start, stop) of a product by HTTP Range requests (Connection: close), resumed from the last byte read when
    a request fails, each one retried as the toolkit's downloaders do (HTTP.final, HTTP.attempts). The attempts count the
    failed requests in a row that read no byte: a request that fails after reading bytes is no failed attempt and starts the
    count again, so a slow transfer cut many times still ends. A 206 answer whose body brings no byte (its headers consistent)
    is a failed attempt as well, so a server that keeps answering so ends the reads after the attempts. An HTTP 202 answer
    (ESA processes the product on the request) is waited for as its Retry-After asks, up to the scene's _PROCESSING_WAIT in
    all; those waits are not attempts.

    Parameters
    ----------
    task : dict
        The scene's task (_fetch_scene): url, session, retries, timeout_second, wait, debug.
    start, stop : int
        The byte range, stop excluded.
    sink : bytearray or _Block
        The bytes read are appended to it (``sink += chunk``; ``len(sink)`` counts them).
    state : dict
        The scene's counters: 'gets' (requests sent), 'bytes' (bytes read), 'waited' (seconds), 'total' (the product's
        size, from Content-Range).

    Returns
    -------
    None

    Raises
    ------
    _NotServed
        Every answer within the scene's wait is 202.
    Exception
        A failure a retry cannot change (HTTP.final), or the last failure.

    Examples
    --------
    >>> _ranged(task, 0, 65536, bytearray(), dict(gets=0, bytes=0, waited=0.0, total=None))
    """
    import time
    from .HTTP import attempts, final, iter_body, raise_for_status, returned, send

    session, url = task['session'], task['url']
    n = attempts(task['retries'])
    attempt, base = 0, len(sink)
    while base + (stop - start) > len(sink):
        offset = start + len(sink) - base
        before = len(sink)
        try:
            state['gets'] += 1
            with send(session.get, url, headers={'Range': f'bytes={offset}-{stop - 1}', 'Connection': 'close'}, stream=True,
                      timeout=(30, 300), check=False) as response:
                if response.status_code == 202:
                    text = response.headers.get('Retry-After', '')
                    wait = float(text) if text.strip().isdigit() else _RETRY_AFTER
                    left = task['wait'] - state['waited']
                    if left <= 0:
                        raise _NotServed(f'ESA did not serve it within {task["wait"]:.0f} s of processing (HTTP 202)')
                    wait = min(max(wait, 1.0), left)
                    if task['debug']:
                        print(f'ESA processes {url}: waiting {wait:.0f} s (HTTP 202)')
                    time.sleep(wait)
                    state['waited'] += wait
                    continue
                raise_for_status(response, url)
                if response.status_code != 206:
                    raise IOError(f'{url}: HTTP {response.status_code} for a Range request, 206 expected')
                found = re.match(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not found or int(found.group(1)) != offset or int(found.group(2)) > stop - 1:
                    raise IOError(f'{url}: Content-Range {response.headers.get("Content-Range")!r} for bytes {offset}-{stop - 1}')
                state['total'] = int(found.group(3))
                for chunk in iter_body(response, min_rate=_MIN_RATE):
                    sink += chunk[:max(0, base + (stop - start) - len(sink))]
                    state['bytes'] += len(chunk)
                if len(sink) == before:
                    raise IOError(f'{url}: an empty body for bytes {offset}-{stop - 1}')
        except _NotServed:
            raise
        except Exception as e:
            stop_now = final(e)
            if len(sink) > before and not stop_now:
                # the request read bytes before it failed: no failed attempt, the next one resumes after them
                attempt = 0
                if task['debug']:
                    print(f'ESA transfer cut after {len(sink) - before} bytes for {url}: {returned(e) or e}; resumed')
                continue
            attempt += 1
            if task['debug'] or stop_now or attempt == n:
                print(f'ERROR: ESA download attempt {attempt}/{n} failed{" (not retried)" if stop_now else ""} '
                      f'for {url}: {returned(e) or e}')
            if stop_now or attempt == n:
                raise
            time.sleep(task['timeout_second'])


def _swath_ranges(position, velocity, looks):
    """
    The slant ranges at which lines of sight at given look angles meet the WGS84 ellipsoid, right of the track (ERS and
    Envisat look right): in the zero-Doppler plane of the satellite, at the look angle off its geodetic nadir.

    Parameters
    ----------
    position, velocity : array-like
        The satellite's Earth-fixed position (m) and velocity (m/s).
    looks : iterable of float
        Look angles, degrees.

    Returns
    -------
    list of float
        The slant ranges, metres; NaN for a line of sight that misses the ellipsoid.

    Examples
    --------
    >>> [round(r) for r in _swath_ranges([7159500.0, 0.0, 0.0], [0.0, 0.0, 7450.0], [0.0])]
    [781363]
    """
    import math
    import numpy as np
    a, e2 = 6378137.0, 0.00669437999014
    b2 = a * a * (1 - e2)
    S, V = np.asarray(position, float), np.asarray(velocity, float)
    # the geodetic latitude of the satellite, by iteration
    p = math.hypot(S[0], S[1])
    lat = math.atan2(S[2], p * (1 - e2))
    for _ in range(6):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        height = p / math.cos(lat) - n
        lat = math.atan2(S[2], p * (1 - e2 * n / (n + height)))
    lon = math.atan2(S[1], S[0])
    down = -np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    along = V / np.linalg.norm(V)
    down = down - (down @ along) * along
    down /= np.linalg.norm(down)
    right = np.cross(down, along)
    out = []
    for look in looks:
        u = math.cos(math.radians(look)) * down + math.sin(math.radians(look)) * right
        qa = (u[0] ** 2 + u[1] ** 2) / a ** 2 + u[2] ** 2 / b2
        qb = 2 * ((S[0] * u[0] + S[1] * u[1]) / a ** 2 + S[2] * u[2] / b2)
        qc = (S[0] ** 2 + S[1] ** 2) / a ** 2 + S[2] ** 2 / b2 - 1
        disc = qb * qb - 4 * qa * qc
        out.append(float('nan') if disc < 0 else (-qb - math.sqrt(disc)) / (2 * qa))
    return out


def _precheck(task, session, cache):
    """
    The checks of a scene before its product is requested, from the precise orbit of its delivered scene
    (utils_esa_orbit.orbit_vectors over the catalogue begin + _DELIVERED; ersorbits.insar.dev, no ESA request): the orbit is
    not refused, and, with a bbox, the bbox reaches the delivered scene. The bbox is the transform's grid region of the crop
    (utils_grid.transform_region in the UTM zone of the bbox centre) at every height of _HEIGHTS; it reaches the scene when
    utils_alos2_orbit.bbox_window maps it, on the precise orbit, into the scene's zero-Doppler time span and then into the
    swath's slant range: the near and far range of _SWATH_LOOK widened by _SWATH_MARGIN on each side, turned into slant
    ranges at the scene's middle (_swath_ranges). The crop's own window on the served header is taken after the request
    (_fetch_scene), as before.

    Parameters
    ----------
    task : dict
        The scene's task (ESA.download): scene, satellite, begin (the catalogue begin, seconds since 1970), swath, bbox (or
        None), and begin_late (optional: seconds the true begin may lie after begin, 1 for the whole seconds of a productId).
    session : requests.Session
        The session of the orbit reads.
    cache : dict
        The orbit files read in this call (utils_esa_orbit.orbit_vectors).

    Returns
    -------
    tuple
        (status, reason): (None, None) to request the scene; ('no_orbit', the refusal), ('outside_time', the reason) or
        ('outside_swath', the reason) to skip it with no product request.

    Examples
    --------
    >>> _precheck(dict(scene='ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000', satellite='ENVISAT',
    ...                begin=1200516433.0, swath='IS2', bbox=(14.34, 40.323, 14.4, 40.363)), _NetrcFreeSession(), {})
    (None, None)
    """
    import numpy as np
    from . import utils_envisat, utils_esa_orbit
    from .utils_alos2_orbit import bbox_window, orbit_state
    from .utils_grid import get_utm_epsg, transform_region

    second = np.timedelta64(1_000_000_000, 'ns')
    epoch = np.datetime64('1970-01-01T00:00:00', 'ns')
    first = epoch + int(round((task['begin'] + _DELIVERED[0]) * 1e9)) * np.timedelta64(1, 'ns')
    last = epoch + int(round((task['begin'] + _DELIVERED[1] + task.get('begin_late', 0.0)) * 1e9)) * np.timedelta64(1, 'ns')
    try:
        vectors = utils_esa_orbit.orbit_vectors(task['satellite'], first, last, session=session, cache=cache)
    except ValueError as e:
        return 'no_orbit', str(e)
    bbox = task['bbox']
    if bbox is None:
        return None, None
    orbit = utils_envisat.orbit_dict(vectors)
    region = transform_region(tuple(bbox), {get_utm_epsg(0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[0] + bbox[2]))})
    if utils_envisat._beyond(orbit, region, _HEIGHTS):
        return 'outside_time', 'the precise orbit puts the bbox outside its delivered scene in time'
    day = np.datetime64(orbit['date'], 'D').astype('datetime64[ns]')
    time0 = (first - day) / second
    span = (last - first) / second
    prf = 100.0
    lines = int(np.ceil(span * prf)) + 1
    looks = _SWATH_LOOK[task['satellite']][task['swath']]
    position, velocity = orbit_state(orbit, time0 + span / 2)
    near, far = _swath_ranges(position, velocity, (looks[0] - _SWATH_MARGIN, looks[1] + _SWATH_MARGIN))
    if not (np.isfinite(near) and np.isfinite(far)):
        return None, None
    # range bins of 10 m: the time span alone (every slant range up to 4000 km), then the swath
    rate = 299792458.0 / 20.0
    for status, label, low, high in (('outside_time', 'in time', 0.0, 4.0e6), ('outside_swath', 'in range: outside its swath', near, far)):
        image = dict(lines=lines, pixels=int(np.ceil((high - low) / 10.0)) + 1, time=time0, prf=prf, near_range=low,
                     date=orbit['date'])
        try:
            window = bbox_window(orbit, image, rate, region, _HEIGHTS)
        except ValueError:
            # the bbox reaches beyond the orbit's span with a part only: the crop decides
            return None, None
        if window is None:
            return status, f'the precise orbit puts the bbox outside its delivered scene {label}'
    return None, None


def _fetch_scene(task):
    """
    Download one scene of ESA.download: its header by Range, the precise orbit of the served scene
    (utils_esa_orbit.orbit_vectors), the line window of the bbox (utils_envisat.aoi_window; the whole product without a
    bbox), and the crop file (the bytes of utils_envisat.crop) written under a temporary name in its folder and renamed. The
    crop is written in blocks, never held in memory whole: the records i and j are read first by Range (their times are in
    the rewritten header, utils_envisat.crop_header), the header is written, the MDS1 records i..j are read by one Range
    request per block of whole records of at most _BLOCK_BYTES (resumed within the block after a drop) into one buffer
    (_Block) and each block is written in one call, and the precise-orbit dataset (utils_envisat.crop_orbit) ends the file.
    The records i and j are read back from the file and must equal their first reads. ESA may serve the scene under another
    name than the one requested (U2a): the served product is cropped.

    Parameters
    ----------
    task : dict
        scene (the productId), satellite, url, path (the file), bbox (or None), session, retries, timeout_second, wait
        (seconds of 202 processing accepted), debug.

    Returns
    -------
    dict
        scene, status ('stored', 'no_orbit', 'no_aoi', 'not_served' or 'failed'), reason (None or the text of the reason),
        requested (whether a product request was sent), served (the served product's name), lines ((i, j) or None),
        gets, bytes (read from ESA), seconds, file (the file stored or None).

    Examples
    --------
    >>> _fetch_scene(task)['status']
    'stored'
    """
    import os
    import time
    from . import utils_envisat, utils_esa_orbit

    t0 = time.monotonic()
    out = dict(scene=task['scene'], status='failed', reason=None, requested=False, served=None, lines=None, file=None)
    state = dict(gets=0, bytes=0, waited=0.0, total=None)
    tmp = None
    try:
        out['requested'] = True
        head = bytearray()
        size = _HEAD_BYTES
        while True:
            _ranged(task, len(head), size, head, state)
            try:
                header = utils_envisat.read_header(bytes(head))
                break
            except ValueError:
                # a header longer than the bytes read: read it again, twice as long (at most 4 MB)
                if size >= state['total'] or size >= (4 << 20):
                    raise
                size = min(2 * size, state['total'])
        out['served'] = header['mph'].get('PRODUCT')
        if header['mph']['TOT_SIZE'] != state['total']:
            raise IOError(f'{task["url"]}: TOT_SIZE {header["mph"]["TOT_SIZE"]} of {out["served"]} is not the served '
                          f'size {state["total"]}')
        main = header['main']
        try:
            orbit = utils_esa_orbit.orbit_vectors(task['satellite'], main['first_time'], main['last_time'])
        except ValueError as e:
            out.update(status='no_orbit', reason=str(e))
            return out
        if task['bbox'] is None:
            window = (0, main['lines'] - 1)
        else:
            window = utils_envisat.aoi_window(header, task['bbox'], orbit)
            if window is None:
                out.update(status='no_aoi', reason=f'its served scene {out["served"]} holds no line of the bbox')
                return out
        i, j = window
        out['lines'] = window
        mds1 = header['mds1']
        size = mds1['dsr_size']
        start, stop = mds1['offset'] + i * size, mds1['offset'] + (j + 1) * size
        # the records i and j: their times are in the header, which is written before the records
        first = bytearray()
        _ranged(task, start, start + size, first, state)
        last = first
        if j > i:
            last = bytearray()
            _ranged(task, stop - size, stop, last, state)
        crop_head = utils_envisat.crop_header(bytes(head[:header['size']]), i, j, bytes(first), bytes(last), orbit)
        crop_tail = utils_envisat.crop_orbit(orbit)
        records = stop - start
        # the records per block, and the one buffer of the scene's blocks
        step = max(1, _BLOCK_BYTES // size)
        buffer = bytearray(min(step * size, records))
        folder = os.path.dirname(task['path'])
        os.makedirs(folder, exist_ok=True)
        tmp = os.path.join(folder, f'.{os.path.basename(task["path"])}.{os.getpid()}.part')
        with open(tmp, 'w+b') as f:
            f.write(crop_head)
            for a in range(start, stop, step * size):
                block = _Block(buffer, min(step * size, stop - a))
                _ranged(task, a, a + block.size, block, state)
                f.write(block.view())
            f.write(crop_tail)
            f.truncate()
            # the records i and j written are the ones the header was written for
            f.seek(len(crop_head))
            written_first = f.read(size)
            f.seek(len(crop_head) + records - size)
            written_last = f.read(size)
        if (written_first, written_last) != (bytes(first), bytes(last)):
            raise IOError(f'{task["url"]}: the records of lines {i + 1} and {j + 1} differ between two reads')
        os.replace(tmp, task['path'])
        tmp = None
        out.update(status='stored', file=task['path'])
    except _NotServed as e:
        out.update(status='not_served', reason=str(e))
    except Exception as e:
        out.update(status='failed', reason=' '.join(str(e).splitlines()) or type(e).__name__)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)
        out.update(gets=state['gets'], bytes=state['bytes'], seconds=round(time.monotonic() - t0, 1))
    return out


class ESA(progressbar_joblib):
    """
    Envisat ASAR and ERS-1/ERS-2 SAR single-look complex scenes (ASA_IMS_1P, SAR_IMS_1P) of the ESA (A)SAR Online Dissemination
    service.

    ESA.search needs no account. Downloads need an ESA EO Sign In account until a cache proxy exists.
    """

    def __init__(self, username=None, password=None):
        """
        Keep the ESA EO Sign In account of the downloads; no network call is made.

        Parameters
        ----------
        username : str, optional
            ESA EO Sign In username. ESA.search needs none; downloads need one until a cache proxy exists.
        password : str, optional
            ESA EO Sign In password.

        Examples
        --------
        >>> ESA(username=None, password=None)
        """
        self.username = username
        self.password = password

    @staticmethod
    def _parse_id(productId):
        """
        The fields of a productId, the ID that the download takes.

        Parameters
        ----------
        productId : str
            Such as 'SAR_IMS_1PNESA19960227_211738_00000015G154_00129_24165_0000'. The 1,488 delivery records of the cached
            catalogue whose centre field is REOD instead of NESA (all created in 2025, 28-30 s long, _delivery_record) have the
            same fields and are read as well.

        Returns
        -------
        dict
            collection ('ASA_IMS_1P' or 'SAR_IMS_1P'), platform ('ENVISAT', 'ERS-1', 'ERS-2'), start (datetime, UTC), orbit,
            track, phase, cycle (int), duration (seconds, int) and centre ('NESA' and the like, or 'REOD'). ERS-2 is told from
            ERS-1 by its phases: A or B from 1995 on, and X.

        Raises
        ------
        ValueError
            For a malformed ID.

        Examples
        --------
        >>> ESA._parse_id('ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000')['platform']
        'ENVISAT'
        """
        from datetime import datetime, timezone
        match = _ID.match(productId) if isinstance(productId, str) else None
        try:
            start = datetime.strptime(match['date'] + match['time'], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ValueError(f'Malformed ESA productId {productId!r}: expected such as '
                             f'SAR_IMS_1PNESA19960227_211738_00000015G154_00129_24165_0000.') from None
        phase = match['phase']
        if match['collection'] == 'ASA_IMS_1P':
            platform = 'ENVISAT'
        elif phase == 'X' or (phase in 'AB' and start.year >= 1995):
            platform = 'ERS-2'
        else:
            platform = 'ERS-1'
        return dict(collection=match['collection'], platform=platform, start=start, orbit=int(match['orbit']),
                    track=int(match['track']), phase=phase, cycle=int(match['cycle']), duration=int(match['duration']),
                    centre=match['centre'])

    @staticmethod
    def _code(platform, phase, track, frame, swath, date, polarization):
        """
        The code of a scene: ENV_|ER1_|ER2_<track>_<frame:04d>_<swath>_<YYYYMMDD>_<pol>. The track field is <track:03d> on the
        35-day grids (Envisat phase 2, ERS-1 phases C, R and G, ERS-2 phase A) and <phase><track:04d> on the others, so that
        codes that look alike are always the same ground track.

        Parameters
        ----------
        platform : str
            'ENVISAT', 'ERS-1' or 'ERS-2'.
        phase : str
            missionPhase.
        track : int
            wrsLongitudeGrid.
        frame : int
            The slot frame (_slot).
        swath : str
            'IS1'..'IS7'.
        date : str
            'YYYYMMDD'.
        polarization : str
            'HH', 'HV', 'VH' or 'VV'.

        Returns
        -------
        str

        Raises
        ------
        ValueError
            For a (platform, phase) of no known orbit grid.

        Examples
        --------
        >>> ESA._code('ERS-2', 'A', 129, 819, 'IS2', '19971218', 'VV')
        'ER2_129_0819_IS2_19971218_VV'
        """
        if (platform, phase) not in _GRIDS:
            raise ValueError(f'No orbit grid known for {platform} phase {phase!r}.')
        field = f'{track:03d}' if _GRIDS[(platform, phase)][2] else f'{phase}{track:04d}'
        return f'{_PREFIX[platform]}_{field}_{frame:04d}_{swath}_{date}_{polarization}'

    @staticmethod
    def _parse_code(code):
        """
        The fields of a scene code (ESA._code).

        Parameters
        ----------
        code : str
            Such as 'ER1_E2117_2777_IS2_19940503_VV'.

        Returns
        -------
        dict
            prefix, track (the code's field), frame (int), swath, date, polarization.

        Raises
        ------
        ValueError
            For a malformed code, or a 5-character track field whose phase letter is not a grid of the satellite with a
            phase-letter track (a 35-day phase takes a 3-digit track).

        Examples
        --------
        >>> ESA._parse_code('ENV_129_0819_IS2_20080116_VV')['frame']
        819
        """
        match = _CODE.match(code) if isinstance(code, str) else None
        if match is None:
            raise ValueError(f'Malformed ESA scene code {code!r}: expected such as ENV_129_0819_IS2_20080116_VV.')
        out = match.groupdict()
        out['frame'] = int(out['frame'])
        if len(out['track']) == 5:
            grid = _GRIDS.get((_SATELLITE[out['prefix']], out['track'][0]))
            if grid is None or grid[2]:
                raise ValueError(f'Malformed ESA scene code {code!r}: phase {out["track"][0]!r} is not an orbit grid of '
                                 f'{_SATELLITE[out["prefix"]]} with a phase-letter track.')
        return out

    @staticmethod
    def _path(basedir, code):
        """
        The file of a scene: <basedir>/<folder>/<code>.<N1|E1|E2>, one folder per grid slot:
        - a 3-digit track (the 35-day phases, where ERS-1 and ERS-2 share their ground tracks): ERS_<track>_<frame>_<swath>, shared by
          both satellites (U2b-S round 5: on the cached records the ERS-2 footprints of the 35-day phase A lie 0.03-0.10 degree east of
          the ERS-1 footprints of the same track in the archive records, a catalogue artefact, and within 0.02 degree in the delivery
          records, against a track spacing of 0.72 degree);
        - a phase letter plus a track (every other phase): the satellite's own folder, ER1_<phase><track>_<frame>_<swath> or
          ER2_<phase><track>_<frame>_<swath> (ERS-1 phase B and ERS-2 phase B of one track number are ground tracks 0.67 degree apart);
        - Envisat: ENV_<track>_<frame>_<swath> (Envisat and ERS-2 share the 35-day track numbers).

        Parameters
        ----------
        basedir : str
            The download folder.
        code : str
            The scene code (ESA._code).

        Returns
        -------
        str

        Examples
        --------
        >>> ESA._path('data', 'ER1_129_0819_IS2_19960227_VV')
        'data/ERS_129_0819_IS2/ER1_129_0819_IS2_19960227_VV.E1'
        >>> ESA._path('data', 'ER2_B0022_2782_IS2_20110315_VV')
        'data/ER2_B0022_2782_IS2/ER2_B0022_2782_IS2_20110315_VV.E2'
        """
        import os
        info = ESA._parse_code(code)
        shared = info['prefix'] != 'ENV' and len(info['track']) == 3
        folder = f'{"ERS" if shared else info["prefix"]}_{info["track"]}_{info["frame"]:04d}_{info["swath"]}'
        return os.path.join(basedir, folder, code + _EXTENSION[info['prefix']])

    @staticmethod
    def _existing(basedir, productId):
        """
        The complete crop file of a scene in basedir, found with no request (the code, which holds the frame, needs the
        catalogue): the files <basedir>/<folder>/<prefix>_<track>_*_<date>_*.<ext> of the scene's satellite, track field and
        date whose path is their code's (ESA._path), whose header is complete (_complete) and whose served scene is the ID's:
        the MPH ABS_ORBIT equals the ID's orbit and the MPH SENSING_START lies within _SKIP_MATCH seconds of the ID's start.
        None unless exactly one file matches.

        Parameters
        ----------
        basedir : str
            The download folder.
        productId : str
            The scene (ESA.search's sceneName).

        Returns
        -------
        str or None
            The file.

        Examples
        --------
        >>> ESA._existing('data', 'ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000')
        """
        import glob
        import os
        import numpy as np
        from .utils_envisat import _sph_time

        info = ESA._parse_id(productId)
        grid = _GRIDS.get((info['platform'], info['phase']))
        if grid is None:
            return None
        prefix = _PREFIX[info['platform']]
        field = f'{info["track"]:03d}' if grid[2] else f'{info["phase"]}{info["track"]:04d}'
        pattern = f'{prefix}_{field}_*_{info["start"]:%Y%m%d}_*{_EXTENSION[prefix]}'
        start = info['start'].replace(tzinfo=None)
        found = []
        for path in sorted(glob.glob(os.path.join(glob.escape(basedir), '*', pattern))):
            try:
                if ESA._path(basedir, os.path.basename(path)[:-len(_EXTENSION[prefix])]) != path:
                    continue
            except ValueError:
                continue
            header = _complete(path)
            if header is None or header['mph'].get('ABS_ORBIT') != info['orbit']:
                continue
            try:
                sensing = _sph_time(header['mph'].get('SENSING_START'))
            except ValueError:
                continue
            delta = (sensing - np.datetime64(start, 'ns')) / np.timedelta64(1, 's')
            if _SKIP_MATCH[0] <= delta <= _SKIP_MATCH[1]:
                found.append(path)
        return found[0] if len(found) == 1 else None

    @staticmethod
    def _scene_codes(scenes, bbox, retries=30, timeout_second=3):
        """
        The codes of scenes as ESA.search gives them over the bbox: per orbit of the scenes one catalogue query (the bbox
        widened as the search widens it, _query_bbox; without a bbox the whole globe; the days of the scene's start +-1 h),
        then the search's own listing (_listing) of the orbit's records over the bbox, every swath and polarization. Without a
        bbox the AOI of each scene is its own catalogue footprint.

        Parameters
        ----------
        scenes : list of str
            productIds.
        bbox : tuple of float or None
            (west, south, east, north).
        retries : int, optional
            Attempts of each catalogue request (_socat). Default 30.
        timeout_second : float, optional
            Seconds between attempts. Default 3.

        Returns
        -------
        dict
            productId -> its row (as _listing gives it); a scene that the listing does not hold is absent.

        Examples
        --------
        >>> ESA._scene_codes(['ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000'], (14.34, 40.323, 14.4, 40.363))
        """
        from datetime import timedelta
        from shapely.geometry import box

        geometry = None if bbox is None else box(*bbox)
        groups = {}
        for scene in scenes:
            info = ESA._parse_id(scene)
            first = (info['start'] - timedelta(hours=1)).date()
            last = (info['start'] + timedelta(hours=1)).date() + timedelta(days=1)
            groups.setdefault((info['collection'], info['platform'], info['orbit'], first, last), []).append(scene)
        codes = {}
        for (collection, platform, orbit, first, last), names in groups.items():
            query = (-180.0, -90.0, 180.0, 90.0) if geometry is None else _query_bbox(geometry, collection)
            records, _ = _parseable(_socat(collection, query, first.isoformat(), last.isoformat(), retries, timeout_second))
            records = [row for row in records if _orbit_key(row) == (platform, orbit)]
            if geometry is not None:
                rows, _ = _listing(records, geometry, lambda row: True)
                codes.update({row['sceneName']: row for row in rows if row['sceneName'] in names})
                continue
            for name in names:
                record = next((row for row in records if row['productId'] == name), None)
                if record is None:
                    continue
                rows, _ = _listing(records, _footprint(record), lambda row: True)
                codes.update({row['sceneName']: row for row in rows if row['sceneName'] == name})
        return codes

    def download(self, basedir, scenes, bbox=None, session=None, n_jobs=None, joblib_backend='loky', skip_exist=True,
                 retries=30, timeout_second=3, debug=False):
        """
        Download Envisat and ERS scenes from the ESA (A)SAR Online Dissemination service, cropped to a bbox, with the precise
        orbit embedded.

        The scenes are the sceneName values of ESA.search (productIds of the archive records). Each is stored as one
        file, <basedir>/<folder>/<code>.<N1|E1|E2>, with the code and folder of its ESA.search row over the same bbox:
        ENV_129_0801_IS2/ENV_129_0801_IS2_20080116_VV.N1, ERS_129_0819_IS2/ER1_129_0819_IS2_19960227_VV.E1 (the 35-day ERS
        tracks share their folders), ER1_E1672_2655_IS2/... for the other ERS phases (ESA._path). Given the search's rows (a
        GeoDataFrame with sceneName and code), each scene is stored under its row's code, with no catalogue query: a search
        over a polygon keeps its codes when the download takes the polygon's bounds. Given IDs, the code needs the frame of
        the scene's grid slot, which comes from the catalogue: one catalogue query per orbit (no login), then the search's own
        selection over the bbox. Without a bbox the scene's own footprint is the AOI. A scene that this selection does not list
        (another bbox than the search's, or a neighbour that adds nothing) is not stored, with a WARNING.

        Before any product request, the precise orbit of each scene's delivered span (its catalogue begin 1.6 s before to 17.7
        s after, _DELIVERED; from ersorbits.insar.dev, no ESA request) is read: a refused orbit, or a bbox that the orbit puts
        outside the delivered scene in time or outside the swath's slant range (the swath's near and far look angles, ESA's
        Envisat swath table and the measured ERS headers, widened by 0.5 degree, _SWATH_LOOK), skips the scene with no
        product request. The crop's lines are then taken from the served header, as below.

        Each product is requested with the account of ESA(username, password): a browser-like sign-in to the service once per
        call, unless session is given; the account is never read from ~/.netrc. ESA processes a scene on its first request
        (HTTP 202 with Retry-After, 1.7-17 min measured; the ERS extra records 2-3 min) and keeps it; the wait is capped at 30
        minutes per scene. ESA serves the scene of a 15 s archive record as its 18 s scene, under another name (the served name
        is the PRODUCT field of the file): the served product is cropped. Every product request is a Range request with
        Connection: close (ESA drops kept-alive connections), resumed from the last byte read when it fails.

        The file is the Envisat format of the product (MPH, SPH, DSDs, ADS, MDS1) with the MDS1 line records of the bbox,
        unchanged: the lines of the transform's UTM grid of the bbox at every height from -500 to 9000 m, from the precise
        orbit, rounded outward to multiples of 512 lines of the product (utils_envisat.aoi_window), all range samples; the
        whole product without a bbox. The records are read and written in blocks of at most 64 MiB (one Range request and one
        write per block, as the ALOS-2 and NISAR downloads): a scene's download holds one block in memory. The header is
        rewritten for the crop (utils_envisat.crop_header): MPH TOT_SIZE, SPH_SIZE,
        NUM_DSD and NUM_DATA_SETS; SPH FIRST_LINE_TIME and LAST_LINE_TIME; the MAIN PROCESSING PARAMS first and last
        zero-Doppler times, line count and its 5 orbit state vectors, replaced by the precise orbit at their epochs. The
        precise orbit of the scene (utils_esa_orbit.orbit_vectors, from https://ersorbits.insar.dev/: Envisat DORIS GDR-D,
        ERS Delft ODR or ESA PRC, the most recent public set covering the scene) is embedded as a 19th DSD and dataset,
        'PRECISE ORBIT STATE VECTORS', about +-14 min around the scene (Envisat 60 s, ERS 30 s steps), its FILENAME the
        source orbit files (utils_envisat.read_precise_orbit reads it). GMTSAR's envi_slc_decode reads the file.

        A scene is not stored, with a WARNING naming the reason, and the call goes on: the precise orbit puts the bbox outside
        its delivered scene (no request); its orbit is refused (no orbit, a jump between joined arcs, a window of the
        repository's no_orbit.csv, a leap second inside the orbit's nodes; before the request, and again for the served
        scene's times); its served scene holds no line of the bbox; ESA does not serve it within 30 minutes; it fails after
        its attempts. One WARNING line counts the scenes not stored with their reasons. One line counts the product requests
        of the call per collection, with ESA's limit for a standard account: 20 L1 products per day and collection (no counter
        is shown by ESA), and the scenes skipped with no request by the precise orbit, counted apart: outside the bbox in time,
        outside its swath, and with a refused orbit (the WARNING line counts the same three kinds).

        Parameters
        ----------
        basedir : str
            Output directory.
        scenes : str, list or geopandas.GeoDataFrame
            The sceneName values of ESA.search, such as 'ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000', a
            newline-separated string of them, or the search's rows: with a code column, each scene is stored under its row's
            code (a code of another satellite, date or track than the scene's is refused), and its startTime is the catalogue begin
            of the check before the request (without startTime, the productId's start and one second more).
        bbox : tuple or None, optional
            (west, south, east, north) in WGS84 degrees, the bbox of the search (its codes are those of the search over it).
            None downloads the whole products. Default None.
        session : requests.Session, optional
            A signed-in session; None signs in with ESA(username, password). Default None.
        n_jobs : int or None, optional
            Parallel scene downloads, at most two per CPU this process may use (HTTP.download_jobs); negative values follow
            joblib. None uses 3 (the count measured without a slower connection). debug=True downloads one after another.
        joblib_backend : str, optional
            Backend of the parallel downloads. Default 'loky'.
        skip_exist : bool, optional
            True skips a scene whose file exists complete (its MPH TOT_SIZE equals the file size and its 19th DSD is the
            precise orbit) with no request at all: no catalogue query, no sign-in, no product request; the file is found by
            the served scene's orbit and sensing start (ESA._existing). False downloads every scene again. Default True.
        retries : int, optional
            Attempts of each request, the first one included; 0 makes one attempt, as 1 does (HTTP.attempts). A failure a
            retry cannot change (HTTP.final) is not retried. The 202 waits are not attempts, nor is a product request that
            fails after reading bytes (the next one resumes after them, so a slow transfer cut many times still ends).
            Default 30.
        timeout_second : int, optional
            Seconds between attempts. Default 3.
        debug : bool, optional
            Print every attempt and wait. Default False.

        Returns
        -------
        pandas.DataFrame or None
            The files stored by this call (sceneName, code, file), or None when none is.

        Raises
        ------
        ValueError
            Before any request: no scene, a malformed sceneName or code, a row's code of another satellite, date or track, one scene
            given with two codes, a malformed bbox, or no ESA EO Sign In account (and no session) when a scene is to be
            downloaded. A refused sign-in.

        Examples
        --------
        >>> esa = ESA('user', 'pass')
        >>> # one Envisat scene, the lines of a 5 x 4 km bbox
        >>> esa.download('data', 'ASA_IMS_1PNESA20080116_204713_000000152065_00129_30744_0000', bbox=(14.34, 40.323, 14.4, 40.363))
        >>> # the rows of a search
        >>> esa.download('data', ESA.search(geometry=AOI, startTime='2008-01-01', stopTime='2008-12-31'), bbox=AOI.bounds)
        """
        import time
        import joblib
        import pandas as pd
        from tqdm.auto import tqdm
        from .HTTP import attempts, download_jobs
        from .utils_pool import release_workers

        # every check before any request or folder
        attempts(retries)
        given = {}
        if hasattr(scenes, 'columns') and 'sceneName' in scenes.columns:
            if 'code' in scenes.columns:
                # the search's rows: their codes and folders are taken as they are
                for _, row in scenes.iterrows():
                    item = {key: row[key] for key in ('code', 'startTime', 'swath') if key in scenes.columns}
                    name = row['sceneName'].strip() if isinstance(row['sceneName'], str) else row['sceneName']
                    if name in given and given[name]['code'] != item['code']:
                        raise ValueError(f'ESA scene {name} is given with two codes: {given[name]["code"]}, {item["code"]}.')
                    given[name] = item
            scenes = scenes['sceneName'].tolist()
        elif isinstance(scenes, str):
            scenes = scenes.split('\n')
        elif hasattr(scenes, 'tolist'):
            scenes = scenes.tolist()
        elif not isinstance(scenes, (list, tuple)):
            raise ValueError(f'ESA scenes {type(scenes).__name__}: give the sceneName values of ESA.search.')
        scenes = list(dict.fromkeys(s.strip() if isinstance(s, str) else s for s in scenes if not isinstance(s, str) or s.strip()))
        if not scenes:
            raise ValueError('No ESA scene to download: give the sceneName values of ESA.search.')
        infos = {scene: ESA._parse_id(scene) for scene in scenes}
        for scene, item in given.items():
            # a row's code names the scene's satellite, date and track (its track field: <track:03d>, or a phase letter and
            # <track:04d>; the catalogue's track equals the productId's, measured on the 97,942 cached records)
            code, info = ESA._parse_code(item['code']), infos[scene]
            if code['prefix'] != _PREFIX[info['platform']] or code['date'] != f'{info["start"]:%Y%m%d}':
                raise ValueError(f'ESA scene {scene}: its code {item["code"]} names another satellite or date.')
            if int(code['track'][-4:]) != info['track']:
                raise ValueError(f'ESA scene {scene}: its code {item["code"]} names another track than the scene\'s {info["track"]}.')
        bbox = _bbox(bbox)
        jobs = download_jobs(_JOBS if n_jobs is None else n_jobs)

        # the complete files, found with no request
        existing = {scene: ESA._existing(basedir, scene) for scene in scenes} if skip_exist else {}
        pending = [scene for scene in scenes if existing.get(scene) is None]
        if not pending:
            return None
        if session is None and not (self.username and self.password):
            raise ValueError('ESA downloads need an ESA EO Sign In account: ESA(username, password).')

        # the codes of the search: the rows given, or the search's selection from the catalogue
        if given:
            rows = {scene: given[scene] for scene in pending}
        else:
            rows = ESA._scene_codes(pending, bbox, retries=retries, timeout_second=timeout_second)
        reasons, tasks = {}, []
        for scene in pending:
            if scene not in rows:
                reasons[scene] = ('not_listed', 'ESA.search does not list it over ' + ('the bbox' if bbox else 'its footprint'))
                continue
            path = ESA._path(basedir, rows[scene]['code'])
            if skip_exist and _complete(path) is not None:
                continue
            info = infos[scene]
            start = rows[scene].get('startTime')
            # the catalogue begin; the productId's start (whole seconds) widened by one second when the row has none
            begin = _epoch(str(start)) if isinstance(start, str) and start else info['start'].timestamp()
            tasks.append(dict(scene=scene, satellite=info['platform'], collection=info['collection'],
                              url=_OADS_URL.format(info['collection'], scene, _EXTENSION[_PREFIX[info['platform']]]),
                              path=path, bbox=bbox, retries=retries, timeout_second=timeout_second, wait=_PROCESSING_WAIT,
                              debug=debug, begin=begin, swath=ESA._parse_code(rows[scene]['code'])['swath']))
            if not (isinstance(start, str) and start):
                tasks[-1]['begin_late'] = 1.0

        # the precise orbit of each scene before its request: a refused orbit, or a bbox outside the delivered scene, is
        # skipped with no product request
        skipped = {}
        if tasks:
            cache = {}
            with _NetrcFreeSession() as orbit_session:
                for task in tasks:
                    t0 = time.monotonic()
                    status, reason = _precheck(task, orbit_session, cache)
                    if debug:
                        print(f'ESA {task["scene"]}: precise orbit check {status or "requested"} in {time.monotonic() - t0:.1f} s')
                    if status is not None:
                        skipped[task['scene']] = status
                        reasons[task['scene']] = (status, reason)
            tasks = [task for task in tasks if task['scene'] not in skipped]

        results = []
        if tasks:
            if session is None:
                session = _login(self.username, self.password, retries=retries, timeout_second=timeout_second)
            for task in tasks:
                task['session'] = session
            if jobs is None or jobs < 1 or debug:
                jobs = 1
            with self.progressbar_joblib(tqdm(desc='Downloading ESA SLC'.ljust(25), total=len(tasks))):
                results = joblib.Parallel(n_jobs=min(jobs, len(tasks)), backend=joblib_backend)(
                    joblib.delayed(_fetch_scene)(task) for task in tasks)
            if joblib_backend == 'loky' and min(jobs, len(tasks)) > 1:
                # the idle joblib workers keep their memory: stop them
                release_workers()
        for task, result in zip(tasks, results):
            if debug:
                print(f'ESA {task["scene"]}: {result["status"]}, lines {result["lines"]}, {result["gets"]} requests, '
                      f'{result["bytes"]} bytes in {result["seconds"]} s')
            if result['status'] != 'stored':
                reasons[task['scene']] = (result['status'], result['reason'])

        for scene in pending:
            if scene in reasons:
                print(f'WARNING: ESA scene {scene} is not stored: {reasons[scene][1]}.')
        counts = {collection: 0 for collection in dict.fromkeys(infos[scene]['collection'] for scene in pending)}
        for task, result in zip(tasks, results):
            counts[task['collection']] += bool(result['requested'])
        kinds = list(skipped.values())
        print(f'ESA product requests in this call: {", ".join(f"{c} {n}" for c, n in counts.items())} (ESA allows a standard '
              f'account {_QUOTA} L1 products per day and collection); skipped with no request by the precise orbit: '
              f'{kinds.count("outside_time")} outside the bbox in time, {kinds.count("outside_swath")} outside its swath, '
              f'{kinds.count("no_orbit")} with a refused orbit.')
        if reasons:
            kinds = [kind for kind, _ in reasons.values()]
            print(f'WARNING: {len(reasons)} of {len(scenes)} ESA scenes are not stored ({kinds.count("not_listed")} not listed '
                  f'for the {"bbox" if bbox else "footprint"}, {kinds.count("outside_time")} skipped with no request as the '
                  f'precise orbit puts the bbox outside them in time, {kinds.count("outside_swath")} outside their swath, '
                  f'{kinds.count("no_orbit")} without an orbit, {kinds.count("no_aoi")} '
                  f'without a line of the bbox, {kinds.count("not_served")} not served by ESA within {_PROCESSING_WAIT / 60:g} '
                  f'min, {kinds.count("failed")} failed to download): {", ".join(s for s in pending if s in reasons)}.')
        stored = [(task['scene'], rows[task['scene']]['code'], result['file'])
                  for task, result in zip(tasks, results) if result['status'] == 'stored']
        if not stored:
            return None
        return pd.DataFrame(stored, columns=['sceneName', 'code', 'file'])

    @staticmethod
    def search(geometry, startTime=None, stopTime=None, flightDirection=None, platform='ENVISAT', polarization=None, swath=None):
        """
        Search the ESA catalogue (SO-Cat, no login) for the Envisat and ERS scenes that the AOI may reach.

        One row per scene that the AOI may reach, per orbit, swath and polarization. Every scene is downloaded and cropped on its own
        and stored as its own file; the core library merges them. The scenes are the regular scenes (U2a: the archive records
        of 15 s, with one record of each group of records that start together) and, for ERS, the extra records (the records whose
        catalogue span is 1 s longer than the acquisition) that fill a part of the AOI that no regular scene covers, such as a slot
        whose regular record is missing from the catalogue. A record is an archive record of a scene when it is not a delivery
        record (the records ESA created in 2025 for the copies it delivered on request have a zero-padded orbit number and track,
        or the centre field REOD, and are never listed) and its acquisition lasts at most 15.1 s. A scene may reach the AOI when its
        archive span meets the AOI's time window after the ascending node: the nominal-orbit model of the AOI's vertices (U2a Q1)
        widened by 2 frames (1.7 s) for the model's error. The archive span of a regular scene is its catalogue span, that of an
        extra record its acquisition. ESA delivers fixed standard scenes with "an overlap between scenes ... added" (ESA OTF user
        manual v2.5: 1.5 s before and 2.5 s after the 15 s span, U2a); the overlap selects nothing, so the same slots serve the AOI on
        every date (the crop keeps the overlap's lines of a listed scene). The archive spans tile the track except at data-take
        changes and missing records, where the delivered overlap of the listed scenes covers the gap; at the end of a data take no
        scene holds data past the acquisition. A
        neighbour that holds no AOI line is skipped by the download with a WARNING. The orbits are those with a scene whose footprint meets the AOI widened by 0.1 degree
        (the ERS-2 footprints of the catalogue sit up to 0.08 degree east). On the grids whose node longitude is not measured (Envisat
        phase 3, ERS-2 phase B) the AOI's time window is the span of the scenes whose footprint meets the AOI widened by 0.2 degree,
        with a WARNING. A catalogue record whose productId has no known format, or holds an impossible date or time (month 13,
        hour 25), is left out with a WARNING naming it (the delivery records with the centre field REOD are read; they are never
        listed, like the other delivery records).

        The frame of a code is its scene's grid slot: regular scenes start on every 18th frame of the 7200 frames of an orbit,
        fixed to the ascending node, and the slot centre 18 k + 9 is the same code on every date. On the ground the Envisat
        grid moves with the season (up to 3.5 frames, 20 km, on track 129), so an Envisat AOI near a slot edge is held by one
        slot on some dates and the next on others; ERS is stable within 0.44 frame (FABLE_ESA_review.md F6). Every regular scene on the grid that may
        reach the AOI is listed. A regular scene shifted off the grid at a data-take edge (or the later of two scenes starting
        together) and the extra records are listed only when they cover a part of the AOI's window that no other listed scene covers:
        the scene that adds the most is listed first (a regular scene before an extra record when they add the same), and the listing
        goes on until none adds any (the result does not depend on the order of the catalogue's answer). Such a scene takes the slot
        of its centre, or, when a listed scene of its orbit holds that slot, the nearest free slot it overlaps. When there is none, it
        is named by its own centre frame, round(7200 t / T), with a WARNING.

        The folder of a scene (ESA._path): ERS_<track>_<frame>_<swath> for the 35-day phases, shared by ERS-1 and ERS-2 (a 3-digit track
        in the code); ER1_<phase><track>_<frame>_<swath> or ER2_<phase><track>_<frame>_<swath> for every other phase of ERS (ERS-1 and
        ERS-2 of one track number there are different ground tracks); ENV_<track>_<frame>_<swath> for Envisat.

        Parameters
        ----------
        geometry : shapely geometry, geopandas.GeoDataFrame or geopandas.GeoSeries
            The AOI in longitude and latitude; a closed LineString is taken as its Polygon.
        startTime, stopTime : str, optional
            The time interval; a date 'YYYY-MM-DD' covers the whole day. None: the start or end of the missions.
        flightDirection : str, optional
            'A' or 'ASCENDING', 'D' or 'DESCENDING'; None for both.
        platform : str or list, optional
            'ENVISAT', 'ERS-1', 'ERS-2' or 'ERS' (both ERS satellites), one name, a comma-separated string or a list.
            Envisat is the ASA_IMS_1P collection, ERS-1 and ERS-2 the SAR_IMS_1P collection. Default 'ENVISAT'.
        polarization : str or list, optional
            'HH', 'HV', 'VH' or 'VV', one name, a comma-separated string or a list; None for all. Default None.
        swath : str or list, optional
            'IS1'..'IS7', one name, a comma-separated string or a list; None for all. Default None.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per scene, sorted by startTime, CRS EPSG:4326: sceneName (the productId, the ID the download takes), code,
            platform, track, phase, frame, swath, polarization, flightDirection, orbitNumber, startTime (beginAcquisition), kind and
            geometry (the catalogue footprint). kind tells how the scene is served: 'regular' for a regular scene that starts on the
            orbit's frame grid, 'offgrid' for a regular scene shifted off the grid (Envisat, at a data-take edge) and 'extra' for an
            ERS extra record (processed by ESA on request).

        Raises
        ------
        ValueError
            For an unknown platform (also one that is neither a string nor a list), swath, polarization or flightDirection, or an
            AOI that crosses 180 degrees of longitude.

        Examples
        --------
        >>> ESA.search(geometry=AOI, startTime='2005-01-01', stopTime='2008-12-31', flightDirection='A', platform='ENVISAT', swath='IS2')
        """
        import warnings
        import geopandas as gpd
        import pandas as pd
        import shapely
        from datetime import date, timedelta, datetime, timezone

        satellites = []
        for name in _names(platform, tuple(_PLATFORMS), 'platform') or []:
            satellites += [s for s in _PLATFORMS[name] if s not in satellites]
        if not satellites:
            raise ValueError(f'Unknown platform {platform!r}: use ENVISAT, ERS-1, ERS-2, ERS.')
        swaths = _names(swath, _SWATHS, 'swath')
        polarizations = _names(polarization, _POLARIZATIONS, 'polarization')
        directions = {None: None, 'A': 'ASCENDING', 'ASCENDING': 'ASCENDING', 'D': 'DESCENDING', 'DESCENDING': 'DESCENDING'}
        key = flightDirection.upper() if isinstance(flightDirection, str) else flightDirection
        if key not in directions:
            raise ValueError(f'Unknown flightDirection {flightDirection!r}: use A, ASCENDING, D, DESCENDING or None.')
        direction = directions[key]

        if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geometry = geometry.geometry.union_all()
        if geometry.geom_type == 'LineString' and geometry.coords[0] == geometry.coords[-1]:
            geometry = shapely.geometry.Polygon(geometry.coords)
        _check_aoi(geometry)

        def bound(value, end):
            if value is None:
                return None
            text = str(value)
            if len(text) == 10:
                text += 'T23:59:59.999999' if end else 'T00:00:00'
            return _epoch(text)
        t0, t1 = bound(startTime, False), bound(stopTime, True)

        records = []
        for collection in dict.fromkeys(_COLLECTION[s] for s in satellites):
            span = _SOCAT_SPAN[collection]
            first = datetime.fromtimestamp(t0, timezone.utc).date().isoformat() if t0 is not None else span[0]
            last = (datetime.fromtimestamp(t1, timezone.utc).date() + timedelta(days=1)).isoformat() if t1 is not None else span[1]
            records += _socat(collection, _query_bbox(geometry, collection), first, last)
        records, left_out = _parseable(records)
        for row in left_out:
            print(f'WARNING: ESA record {row.get("productId")!r} has no known productId format or holds an impossible date or time: '
                  f'it is not listed.')

        def wanted(row):
            b = _epoch(row['beginAcquisition'])
            return (_platform(row) in satellites
                    and (t0 is None or b >= t0) and (t1 is None or b <= t1)
                    and (direction is None or row['orbitDirection'] == direction)
                    and (swaths is None or row['swathIdentifier'] in swaths)
                    and (polarizations is None or row['polarisationChannels'].strip().upper() in polarizations))
        rows, notes = _listing(records, geometry, wanted)
        for note in notes:
            print(note)
        columns = ['sceneName', 'code', 'platform', 'track', 'phase', 'frame', 'swath', 'polarization', 'flightDirection',
                   'orbitNumber', 'startTime', 'kind', 'geometry']
        if not rows:
            return gpd.GeoDataFrame(pd.DataFrame(columns=columns), geometry='geometry', crs='EPSG:4326')
        gdf = gpd.GeoDataFrame(pd.DataFrame(rows, columns=columns), geometry='geometry', crs='EPSG:4326')
        return gdf.sort_values(['startTime', 'sceneName'], kind='stable').reset_index(drop=True)
