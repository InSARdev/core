# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://InSAR.dev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
Precise orbits of Envisat, ERS-1 and ERS-2 from the public orbit repository served at https://ersorbits.insar.dev/:
one orbit set per satellite and UTC day, listed in ``<SATELLITE>/<YYYY>/<MM>/<DD>/index.csv``, each orbit file stored
unchanged in its own zip (the repository's README gives the sets, their ranking and the per-day rule).

The orbit files come in three formats:

- Delft ODR (TU Delft DEOS): a 32-byte header (``xODR``, the satellite in 8 characters, then big-endian int32 begin
  time, repeat period, arc number, record count and version) and 16-byte records of big-endian int32: seconds since
  1985-01-01 00:00 UTC, geodetic latitude and longitude in 1e-7 degree, height in mm. Positions only. The latitude,
  longitude and height refer to the ellipsoid of the getorb software, GRS80 with a rounded flattening: a = 6378137 m,
  1/f = 298.257 (DEOS, ``dorvor_to_odr.m``, https://doris.tudelft.nl/software/dorvor_to_odr.m: "getorb ODR files
  contain data in phi,lambda,height with respect to the GRS80 ellipsoid (rounded flattening)", case 'GRS80_getorb'
  a = 6378137, f = 1/298.257; GMTSAR's odr2prc.f includes getorb's GRS80.inc). The velocities are the derivative of
  the local interpolating polynomial of the positions (``interpolate``).
- D-PAF PRC (ESA STEP, ``ERS1.ORB.PRC`` / ``ERS2.ORB.PRC`` POSVEL): 130-byte ASCII records. The STTERR records hold
  the Earth-fixed state: day in 0.1 d since 2000-01-01 12:00 TDT (columns 15-20), microseconds of that day (21-31),
  x y z in mm (32-67, 3 x I12), vx vy vz in um/s (68-100, 3 x I11) (D-PAF "Altimeter and Orbit Global Products
  Manual" ERS-D-GPM-31200, section 3.2; GMTSAR odr2prc.f formats 103 and 104). The time scale is TDT: UTC = TDT -
  32.184 s - (TAI - UTC), with the leap-second table LEAP_SECONDS. The Earth-fixed frame is D-PAF's CTS, whose
  z-axis is the BIH mean pole; the manual gives its offsets to the IERS pole as 45 and 286 mas ("x_ERS - 0.045" =
  x_DPAF, y_ERS - 0.286" = y_DPAF"). The PRC positions and velocities are rotated by these offsets into the IERS-pole
  frame of the other sets (PRC_POLE_OFFSET).
- Envisat DORIS DOR_VOR (ESA STEP, GDR-D): an MPH and SPH (``SPH_DESCRIPTOR="ORBITE POE_REST SAT ENV1"``) with one DSD
  ``DORIS PRECISE ORBIT``, then 129-byte ASCII records at 60 s: UTC time ``DD-MMM-YYYY hh:mm:ss.ffffff``, UT1 - UTC,
  absolute orbit, x y z in m, vx vy vz in m/s, quality flag. The header names no frame; its records carry UT1 - UTC
  beside the time, so the time is UTC, and the vectors are Earth-fixed: the product state vector of an Envisat SLC
  (Earth-fixed by the product specification) equals the DORIS record of the same second.

Arcs of one set are never joined across a jump (``cut_at_jumps``), a scene within JUMP_EDGE seconds of a jump of D-PAF PRC arcs
is refused (``check_jumps``: 1020 s), a scene inside one arc takes that arc alone (``own_arc``), and a window that overlaps a span listed in the satellite's ``no_orbit.csv`` (windows no set gives a verified orbit for) is
refused (``check_no_orbit``). A window cut at the end of its set, a jump or a joint must reach EDGE seconds beyond the scene
(0 for D-PAF PRC, 160 for Delft ODR, 90 for DORIS: the measured distance from which the one-sided interpolation near the cut
stays within 5 mm).

The functions are pure (no file access) except ``orbit_vectors``, which reads the repository over HTTP through a session that
never reads ~/.netrc (``_NetrcFreeSession``).

Facts for a future switch of orbit source (2026-11-23; the reports are in session/patches/esa_alos1/reports):
- The repository: https://ersorbits.insar.dev/, one set per satellite and UTC day; a scene takes the set of its first line's
  day only, with arcs of other days used only if they belong to that set (``select_orbit``; U2bO_round2_report.md).
- ``no_orbit.csv`` per satellite lists the windows no set gives a verified orbit for; an overlapping window is refused
  (U2bO_round3_report.md).
- A cut window keeps the scene EDGE seconds from the cut by the format's measured one-sided interpolation error (EDGE;
  U2bD4_round2_report.md).
- A scene keeps JUMP_EDGE seconds (1020 for D-PAF PRC) from a joint of its set's arcs above JUMP, by the measured distance from
  which the arcs at such joints agree with the other producers within the repository's confirmation threshold, 10 m (JUMP_EDGE;
  U2bD5_round2_report.md).
- The D-PAF PRC positions and velocities are rotated by the CTS-to-IERS pole offsets (45, 286 mas, ERS-D-GPM-31200) into the
  IERS frame of the other sets (U2bO_report.md).
- Leap seconds: PRC TDT is converted to UTC with LEAP_SECONDS (IERS Bulletin C); the Delft ODR node labelled 00:00:00 of a leap
  date is the inserted second and is dropped; a window whose interpolation nodes include a leap second is refused
  (U2bO_report.md 3.3, U2bO_round2_report.md 3, U2bO_round3_report.md 2).
"""
import hashlib
import io
import re
import zipfile

import numpy as np
import requests

# the repository and its satellite folders
BASE_URL = 'https://ersorbits.insar.dev/'
SATELLITES = {'ENVISAT': 'ENVISAT', 'ERS-1': 'ERS1', 'ERS-2': 'ERS2'}
# the output grid: whole multiples of the step, from MARGIN seconds before the scene start to MARGIN after its stop
STEPS = {'ENVISAT': 60, 'ERS-1': 30, 'ERS-2': 30}
MARGIN = 840
# a window cut to the arcs of its set (select_orbit), at a jump (cut_at_jumps) or at the joint of two arcs (own_arc) must reach
# at least EDGE seconds beyond each end of the scene, by the format of the arcs (the prefix of the zip name, ``_edge``). Near a
# cut end the interpolation is one-sided twice: ``interpolate`` shifts its 9-point stencil inside the arc's nodes, and the crop's
# reader (utils_alos2_orbit.orbit_state, 8 points) shifts its stencil inside the window. Measured on real arcs of the repository
# (50 D-PAF PRC, 20 Delft ODR, 10 DORIS; 10 interior nodes of each taken as the cut end, both sides; the cut window from the
# nodes on one side against the whole window from all nodes, through the crop's reader; scene times 0-300 s from the cut end;
# session/patches/esa_alos1/reports/U2bD4_round2_report.md): PRC (30 s nodes in mm, on the 30 s grid) differs by at most
# 4.6 mm at any distance; Delft ODR (60 s nodes in 1e-7 degree) by up to 51 mm, 5 mm from 159 s on; DORIS (60 s nodes at
# hh:mm:27-28) by up to 17 mm, 5 mm from 85 s on. Each EDGE is that distance rounded up to 10 s.
EDGE = {'PRC_': 0, 'ODR_': 160, 'DOR_VOR': 90}
# consecutive arcs of a set are joined only where their positions at the joint differ by at most JUMP metres (see
# ``cut_at_jumps``). Measured on the 5,847 joints of the repository (every kept arc and its successor by the chain rule of
# ``select_arcs``; the difference of the two arcs at the joint, session/patches/esa_alos1/reports/U2bO_round3_report.md): the
# bulk has medians 0.07-0.35 m (Delft and ERS-2 PRC), 0.81 m (ERS-1 PRC), 0.0 m (DORIS) and no gap above it, a heavy tail runs
# from about 1 m to 100 km. 2 m is above the 99.9 % point of the Rayleigh bulk of every Delft set (0.77-1.09 m) and of ERS-2 PRC
# (0.22 m), and is the threshold ORB2 used for a stretch where two sets disagree.
JUMP = 2.0
# a scene keeps at least JUMP_EDGE seconds from a joint of two consecutive arcs of its set whose positions differ there by more than
# JUMP metres (``check_jumps``), by the format of the arcs (the prefix of the zip name); a format not listed has no such distance (its
# window is cut at the jump, ``cut_at_jumps``). At such a joint one of the arcs is wrong near its end. The criterion is the
# repository's own: an arc is confirmed where another producer agrees with it within 10 m. Measured on the 138 D-PAF PRC joints of
# the repository above JUMP (75 ERS-1, 63 ERS-2; session/patches/esa_alos1/reports/U2bD5_round2_report.md): the 30 s nodes of each
# arc out to 1800 s from the joint, those outside no_orbit.csv, against the nearest other producer (Delft ODR of every set, GFZ
# VER13); on the 121 sides with such nodes (72 ERS-1, 49 ERS-2) every node from at most 720 s from the joint on agrees within 10 m
# (ERS-2: worst 720 s, 90 % 240 s; ERS-1: worst 210 s, 90 % 0 s). 1020 s is the worst case, 720 s, with a margin of 300 s.
JUMP_EDGE = {'PRC_': 1020}
# the points of the Lagrange interpolation
POINTS = 9
# the list of windows without a verified orbit, in the satellite's folder
NO_ORBIT = 'no_orbit.csv'
# the largest index (bytes, decoded), zip and orbit file read
INDEX_LIMIT = 1 << 20
ZIP_LIMIT = 64 << 20
FILE_LIMIT = 64 << 20
# TAI - UTC in seconds from each date, 1991-2012: IERS Bulletin C, Leap_Second.dat
# (https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat, updated through Bulletin 72)
LEAP_SECONDS = (('1991-01-01', 26), ('1992-07-01', 27), ('1993-07-01', 28), ('1994-07-01', 29), ('1996-01-01', 30),
                ('1997-07-01', 31), ('1999-01-01', 32), ('2006-01-01', 33), ('2009-01-01', 34), ('2012-07-01', 35))
# TT (TDT) - TAI
TDT_TAI = 32.184
# the ellipsoid of the Delft ODR latitude, longitude and height (getorb's GRS80 with a rounded flattening)
ODR_A = 6378137.0
ODR_F = 1.0 / 298.257
ODR_EPOCH = np.datetime64('1985-01-01T00:00:00', 'ns')
PRC_EPOCH = np.datetime64('2000-01-01T12:00:00', 'ns')
MJD_EPOCH = np.datetime64('2000-01-01T00:00:00', 'ns')
# the offsets of the IERS pole from the D-PAF CTS pole of the PRC Earth-fixed vectors, x and y, arcseconds
# (D-PAF ERS-D-GPM-31200, PRC Product Specification, Reference System)
PRC_POLE_OFFSET = (0.045, 0.286)
# the 5 state vectors of the MAIN PROCESSING PARAMS ADS record: offset of the first, size of each
SV_OFFSET = 1765
SV_SIZE = 36
SV_COUNT = 5
# the precise-orbit dataset: its DSD name (at most 28 characters) and record size
ORBIT_DS_NAME = 'PRECISE ORBIT STATE VECTORS'
ORBIT_DSR_SIZE = 60
_INT32 = 2 ** 31


def _ns(times):
    """
    UTC times as numpy datetime64[ns].

    Parameters
    ----------
    times : datetime-like or array_like
        A datetime, numpy datetime64, pandas Timestamp or ISO string, or an array of them.

    Returns
    -------
    numpy.ndarray or numpy.datetime64
        datetime64[ns], the shape of the input.

    Examples
    --------
    >>> _ns('1997-12-18T09:49:12')
    np.datetime64('1997-12-18T09:49:12.000000000')
    """
    if hasattr(times, 'to_datetime64'):
        times = times.to_datetime64()
    return np.asarray(times, dtype='datetime64[ns]')[()]


def _seconds(times, epoch):
    """
    Seconds of datetime64 times after an epoch, as float.

    Parameters
    ----------
    times : numpy.ndarray or numpy.datetime64
        datetime64 times.
    epoch : numpy.datetime64
        The epoch.

    Returns
    -------
    numpy.ndarray or float
        Seconds after the epoch.

    Examples
    --------
    >>> _seconds(np.datetime64('2000-01-01T00:01:00'), MJD_EPOCH)
    np.float64(60.0)
    """
    return (_ns(times) - _ns(epoch)).astype('timedelta64[ns]').astype(np.int64) / 1e9


def _leaps():
    """
    The leap-second table as arrays.

    Returns
    -------
    tuple of numpy.ndarray
        The dates (datetime64[ns], UTC midnight) from which each TAI - UTC applies, and TAI - UTC in seconds (int64).

    Examples
    --------
    >>> _leaps()[1][0]
    np.int64(26)
    """
    dates = np.array([np.datetime64(d, 'ns') for d, _ in LEAP_SECONDS])
    offsets = np.array([o for _, o in LEAP_SECONDS], dtype=np.int64)
    return dates, offsets


def tai_utc(times):
    """
    TAI - UTC at UTC times, from the leap-second table LEAP_SECONDS.

    Parameters
    ----------
    times : datetime-like or array_like
        UTC times from 1991-01-01 on.

    Returns
    -------
    numpy.ndarray or numpy.int64
        TAI - UTC in seconds.

    Raises
    ------
    ValueError
        A time before 1991-01-01, the start of the table, or NaT.

    Examples
    --------
    >>> tai_utc('1997-07-01T00:00:00')
    np.int64(31)
    >>> tai_utc('1997-06-30T23:59:59')
    np.int64(30)
    """
    dates, offsets = _leaps()
    utc = _ns(times)
    if np.any(np.isnat(utc)):
        raise ValueError('TAI - UTC of NaT')
    k = np.searchsorted(dates, utc, side='right') - 1
    if np.any(k < 0):
        raise ValueError(f'Leap-second table starts at {LEAP_SECONDS[0][0]}')
    return offsets[k]


def tdt_to_utc(times):
    """
    UTC of Terrestrial Dynamic Time (TT) instants: UTC = TDT - 32.184 s - (TAI - UTC).

    TAI - UTC is the value in force at the UTC result, from LEAP_SECONDS. An instant inside an inserted leap second
    (23:59:60 UTC, which datetime64 cannot hold) gives NaT.

    Parameters
    ----------
    times : datetime-like or array_like
        TDT instants (as naive datetimes), from 1991-01-01 on.

    Returns
    -------
    numpy.ndarray or numpy.datetime64
        UTC, datetime64[ns]; NaT inside an inserted leap second.

    Raises
    ------
    ValueError
        An instant before the leap-second table, or NaT.

    Examples
    --------
    >>> tdt_to_utc('1997-07-01T00:01:03.184')
    np.datetime64('1997-07-01T00:00:00.000000000')
    >>> tdt_to_utc('1997-07-01T00:01:02.184')
    np.datetime64('NaT','ns')
    >>> tdt_to_utc('1997-07-01T00:01:01.184')
    np.datetime64('1997-06-30T23:59:59.000000000')
    """
    dates, offsets = _leaps()
    if np.any(np.isnat(_ns(times))):
        raise ValueError('UTC of TDT NaT')
    tai = _ns(times) - np.timedelta64(int(round(TDT_TAI * 1e9)), 'ns')
    # TAI at which each offset starts: UTC midnight of its date plus the offset
    starts = dates + offsets * np.timedelta64(1_000_000_000, 'ns')
    k = np.searchsorted(starts, tai, side='right') - 1
    if np.any(k < 0):
        raise ValueError(f'Leap-second table starts at {LEAP_SECONDS[0][0]}')
    utc = tai - offsets[k] * np.timedelta64(1_000_000_000, 'ns')
    # inside the inserted second: the old offset reaches the next leap date
    following = np.where(k + 1 < len(dates), dates[np.minimum(k + 1, len(dates) - 1)], np.datetime64('NaT', 'ns'))
    inside = (k + 1 < len(dates)) & (utc >= following)
    return np.where(inside, np.datetime64('NaT', 'ns'), utc)[()]


def parse_index(text):
    """
    Parse a day index (``index.csv``) of the orbit repository.

    Parameters
    ----------
    text : str or bytes
        The index content: the header ``path,start,end,source,set,model,produced,sha256`` and one row per arc.

    Returns
    -------
    list of dict
        One dict per arc: ``'path'`` (relative to the repository root), ``'start'`` and ``'end'`` (datetime64[ns]
        UTC), ``'source'``, ``'set'``, ``'model'``, ``'produced'`` and ``'sha256'`` (str).

    Raises
    ------
    ValueError
        The header is not the index header.

    Examples
    --------
    >>> parse_index('path,start,end,source,set,model,produced,sha256\\n'
    ...             'ERS2/1997/12/16/ODR_dgm-e04_278.zip,1997-12-16T00:00:00,1997-12-21T12:00:00,TU Delft DEOS,'
    ...             'ODR.ERS-2/dgm-e04,DGM-E04,1998-12-15T17:33:32,c1e0')[0]['set']
    'ODR.ERS-2/dgm-e04'
    """
    import csv

    if isinstance(text, bytes):
        text = text.decode('utf-8')
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    columns = ['path', 'start', 'end', 'source', 'set', 'model', 'produced', 'sha256']
    if header != columns:
        raise ValueError(f'Not an orbit index header: {header}')
    rows = []
    for values in reader:
        if not values:
            continue
        row = dict(zip(columns, values))
        row['start'], row['end'] = _ns(row['start']), _ns(row['end'])
        rows.append(row)
    return rows


def select_arcs(rows, first, last):
    """
    The arcs of a day index that give the orbit over a time window, by the repository's rule.

    Of the arcs whose span covers the window, the one whose span centre is nearest the window centre (the first in
    the rows on a tie). When no single arc covers it, consecutive arcs of one set that together cover it, each
    starting at or before the end of the one before (PRC arcs meet end to end): from the arc that covers the window
    start with the latest end, each time the arc starting at or before the current end with the latest end; the sets
    are tried in the order of the rows.

    Parameters
    ----------
    rows : list of dict
        The arcs, as ``parse_index`` returns them (``orbit_vectors`` passes the arcs of one set, see ``select_orbit``);
        a path listed twice is taken once.
    first, last : datetime-like
        The window, UTC.

    Returns
    -------
    list of dict
        The selected arcs in time order: one, several joined, or none when no arc or chain covers the window.

    Examples
    --------
    >>> select_arcs(parse_index(open('index.csv').read()), '1997-12-18T09:35:00', '1997-12-18T10:04:00')
    """
    first, last = _ns(first), _ns(last)
    unique = {}
    for row in rows:
        unique.setdefault(row['path'], row)
    rows = list(unique.values())
    covering = [row for row in rows if row['start'] <= first and last <= row['end']]
    if covering:
        centre = first + (last - first) / 2
        return [min(covering, key=lambda row: abs(row['start'] + (row['end'] - row['start']) / 2 - centre))]
    for name in dict.fromkeys(row['set'] for row in rows):
        arcs = [row for row in rows if row['set'] == name]
        starting = [row for row in arcs if row['start'] <= first]
        if not starting:
            continue
        chain = [max(starting, key=lambda row: row['end'])]
        while chain[-1]['end'] < last:
            following = [row for row in arcs if row['start'] <= chain[-1]['end'] < row['end']]
            if not following:
                break
            chain.append(max(following, key=lambda row: row['end']))
        if chain[-1]['end'] >= last:
            return chain
    return []


def _geodetic_to_ecef(lat, lon, height):
    """
    Earth-fixed XYZ of geodetic coordinates on the ODR ellipsoid (ODR_A, ODR_F).

    Parameters
    ----------
    lat, lon : array_like
        Geodetic degrees.
    height : array_like
        Metres above the ellipsoid.

    Returns
    -------
    numpy.ndarray
        X, Y, Z in metres, shape (..., 3).

    Examples
    --------
    >>> _geodetic_to_ecef(0.0, 0.0, 0.0)
    array([6378137.,       0.,       0.])
    """
    e2 = ODR_F * (2.0 - ODR_F)
    lat, lon = np.radians(lat), np.radians(lon)
    normal = ODR_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.stack([(normal + height) * np.cos(lat) * np.cos(lon),
                     (normal + height) * np.cos(lat) * np.sin(lon),
                     (normal * (1.0 - e2) + height) * np.sin(lat)], axis=-1)


def parse_odr(data):
    """
    Decode a Delft ODR orbit file.

    The header's record count is read; bytes after the counted records are ignored. Times are UTC as written
    (seconds since 1985-01-01 counted in days of 86400 s).

    An arc that crosses a leap second holds 60 s records labelled ... 23:59:00, 00:00:00, 00:00:59, 00:01:59 ...: the
    node labelled 00:00:00 of the leap date, followed by a 59 s step, is physically the inserted second 23:59:60, which
    datetime64 cannot hold (fits of the neighbouring nodes on both sides predict its position to mm at 23:59:60 and
    7.5 km off at 00:00:00), and the nodes after it carry the correct UTC labels. Such a node is dropped. An arc
    whose labels stay on :00 after a leap (the 60 s step is kept) has no such node.

    Parameters
    ----------
    data : bytes
        The ODR file.

    Returns
    -------
    dict
        ``'times'`` (datetime64[ns] UTC), ``'positions'`` (n, 3) Earth-fixed metres on the ODR ellipsoid,
        ``'velocities'`` None, ``'satellite'`` (str), ``'arc'`` (int).

    Raises
    ------
    ValueError
        Not an ``xODR`` file, or shorter than its record count.

    Examples
    --------
    >>> parse_odr(zipfile.ZipFile('ODR_dgm-e04_278.zip').read('ODR.278'))['positions'].shape
    """
    if data[:4] != b'xODR' or len(data) < 32:
        raise ValueError(f'Not a Delft ODR file: {bytes(data[:4])!r}')
    _, _, arc, count, _ = np.frombuffer(data[12:32], dtype='>i4')
    if len(data) < 32 + 16 * int(count):
        raise ValueError(f'ODR file of {len(data)} bytes is shorter than its {count} records')
    records = np.frombuffer(data[32:32 + 16 * int(count)], dtype='>i4').reshape(int(count), 4).astype(np.int64)
    times = ODR_EPOCH + records[:, 0] * np.timedelta64(1_000_000_000, 'ns')
    positions = _geodetic_to_ecef(records[:, 1] / 1e7, records[:, 2] / 1e7, records[:, 3] / 1e3)
    # the inserted second: the node labelled 00:00:00 of a leap date (the table's first date is no leap) with a 59 s step
    # to the next node
    leaps = _leaps()[0][1:]
    inserted = np.isin(times[:-1], leaps) & (np.diff(times) == np.timedelta64(59, 's'))
    if inserted.any():
        keep = np.ones(len(times), dtype=bool)
        keep[:-1] = ~inserted
        times, positions = times[keep], positions[keep]
    return {'times': times, 'positions': positions, 'velocities': None,
            'satellite': data[4:12].decode('ascii').strip(), 'arc': int(arc)}


def _pole_rotation():
    """
    The rotation from D-PAF's CTS (BIH pole) to the IERS-pole frame: I + [w]x with w = (y_offset, x_offset, 0) rad.

    Returns
    -------
    numpy.ndarray
        3 x 3 matrix applied as ``R @ v``.

    Examples
    --------
    >>> _pole_rotation().shape
    (3, 3)
    """
    wx, wy = np.radians(PRC_POLE_OFFSET[1] / 3600.0), np.radians(PRC_POLE_OFFSET[0] / 3600.0)
    return np.array([[1.0, 0.0, wy], [0.0, 1.0, -wx], [-wy, wx, 1.0]])


def parse_prc(data):
    """
    Decode a D-PAF PRC orbit file (POSVEL): the STTERR (Earth-fixed) records.

    TDT is converted to UTC by ``tdt_to_utc``; a record inside an inserted leap second is dropped. Positions and
    velocities are rotated from D-PAF's CTS to the IERS pole (PRC_POLE_OFFSET).

    Parameters
    ----------
    data : bytes
        The PRC file.

    Returns
    -------
    dict
        ``'times'`` (datetime64[ns] UTC), ``'positions'`` (n, 3) m, ``'velocities'`` (n, 3) m/s.

    Raises
    ------
    ValueError
        No ``DSIDP`` first record or no STTERR record.

    Examples
    --------
    >>> parse_prc(zipfile.ZipFile('PRC_090529_73746_rev2.zip').read('PRC_090529_73746_rev2'))['times'][0]
    """
    if not data.startswith(b'DSIDP'):
        raise ValueError(f'Not a PRC file: {bytes(data[:6])!r}')
    records = [data[i:i + 130] for i in range(0, len(data) - 129, 130)]
    terr = [record.decode('ascii') for record in records if record.startswith(b'STTERR')]
    if not terr:
        raise ValueError('PRC file holds no STTERR record')
    day = np.array([int(r[14:20]) for r in terr], dtype=np.int64)
    usec = np.array([int(r[20:31]) for r in terr], dtype=np.int64)
    position = np.array([[int(r[a:a + 12]) for a in (31, 43, 55)] for r in terr], dtype=np.int64) / 1e3
    velocity = np.array([[int(r[a:a + 11]) for a in (67, 78, 89)] for r in terr], dtype=np.int64) / 1e6
    tdt = PRC_EPOCH + day * np.timedelta64(8_640_000_000_000, 'ns') + usec * np.timedelta64(1000, 'ns')
    utc = tdt_to_utc(tdt)
    keep = ~np.isnat(utc)
    rotation = _pole_rotation()
    return {'times': utc[keep], 'positions': position[keep] @ rotation.T, 'velocities': velocity[keep] @ rotation.T}


def parse_dor(data):
    """
    Decode an Envisat DORIS DOR_VOR orbit file: the records of its ``DORIS PRECISE ORBIT`` data set.

    Parameters
    ----------
    data : bytes
        The DOR_VOR file.

    Returns
    -------
    dict
        ``'times'`` (datetime64[ns] UTC), ``'positions'`` (n, 3) m, ``'velocities'`` (n, 3) m/s.

    Raises
    ------
    ValueError
        No ``DORIS PRECISE ORBIT`` DSD, or a data set beyond the file.

    Examples
    --------
    >>> parse_dor(zipfile.ZipFile('DOR_VOR_AXVF-P20120425_171900_20080115_215527_20080117_002327.zip').read(
    ...     'DOR_VOR_AXVF-P20120425_171900_20080115_215527_20080117_002327'))['times'][0]
    """
    match = re.search(rb'DS_NAME="DORIS PRECISE ORBIT\s*"\s*DS_TYPE=M\s*FILENAME="[^"]*"\s*DS_OFFSET=\+(\d+)<bytes>\s*'
                      rb'DS_SIZE=\+(\d+)<bytes>\s*NUM_DSR=\+(\d+)\s*DSR_SIZE=\+(\d+)<bytes>', data[:20000])
    if match is None:
        raise ValueError('Not a DORIS orbit file: no DORIS PRECISE ORBIT data set')
    offset, size, count, length = (int(value) for value in match.groups())
    if offset + count * length > len(data):
        raise ValueError(f'DORIS data set ends beyond the file ({offset + count * length} > {len(data)} bytes)')
    months = {m: i + 1 for i, m in enumerate(('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
                                              'NOV', 'DEC'))}
    times, values = [], []
    for k in range(count):
        record = data[offset + k * length:offset + (k + 1) * length].decode('ascii')
        stamp = record[:27]
        iso = f'{stamp[7:11]}-{months[stamp[3:6]]:02d}-{stamp[0:2]}T{stamp[12:27]}'
        times.append(np.datetime64(iso, 'ns'))
        values.append([float(v) for v in record[27:].split()[2:8]])
    values = np.array(values)
    return {'times': np.array(times, dtype='datetime64[ns]'), 'positions': values[:, :3], 'velocities': values[:, 3:]}


def parse_orbit_file(name, data):
    """
    Decode an orbit file of the repository by its name: ``ODR.*`` Delft ODR, ``PRC_*`` D-PAF PRC, ``DOR_VOR*`` DORIS.

    Parameters
    ----------
    name : str
        The orbit file name (the zip member).
    data : bytes
        The orbit file.

    Returns
    -------
    dict
        As ``parse_odr``, ``parse_prc`` or ``parse_dor``.

    Raises
    ------
    ValueError
        A name of another format.

    Examples
    --------
    >>> parse_orbit_file('ODR.278', zipfile.ZipFile('ODR_dgm-e04_278.zip').read('ODR.278'))['positions'].shape
    """
    if name.startswith('ODR.'):
        return parse_odr(data)
    if name.startswith('PRC_'):
        return parse_prc(data)
    if name.startswith('DOR_VOR'):
        return parse_dor(data)
    raise ValueError(f'Unsupported orbit file: {name}')


def _lagrange(nodes, t):
    """
    The Lagrange weights and their derivatives of nodes at times.

    Parameters
    ----------
    nodes : numpy.ndarray
        (m, P) node times per time, float.
    t : numpy.ndarray
        (m,) times, float.

    Returns
    -------
    tuple of numpy.ndarray
        ``(weights, derivatives)``, each (m, P): the value and the time derivative of each basis polynomial at t.

    Examples
    --------
    >>> _lagrange(np.array([[0.0, 1.0, 2.0]]), np.array([0.5]))[0]
    array([[ 0.375,  0.75 , -0.125]])
    """
    count = nodes.shape[1]
    diff = t[:, None] - nodes
    weights = np.empty_like(nodes)
    derivatives = np.zeros_like(nodes)
    for j in range(count):
        others = [m for m in range(count) if m != j]
        denominator = np.prod(nodes[:, [j]] - nodes[:, others], axis=1)
        weights[:, j] = np.prod(diff[:, others], axis=1) / denominator
        for k in others:
            rest = [m for m in others if m != k]
            derivatives[:, j] += np.prod(diff[:, rest], axis=1)
        derivatives[:, j] /= denominator
    return weights, derivatives


def _nodes(node_seconds, epoch_seconds, points=POINTS):
    """
    The indexes of the nodes each epoch is interpolated from: ``points`` consecutive nodes centred on the node nearest
    the epoch, shifted inside the nodes at the ends.

    Parameters
    ----------
    node_seconds : numpy.ndarray
        (n,) node times, float seconds, increasing; n >= points.
    epoch_seconds : numpy.ndarray
        (m,) epoch times, float seconds.
    points : int, optional
        The number of nodes. Default POINTS (9).

    Returns
    -------
    numpy.ndarray
        (m, points) node indexes.

    Examples
    --------
    >>> _nodes(np.arange(20) * 60.0, np.array([0.0, 610.0, 1140.0]))[:, [0, -1]]
    array([[ 0,  8],
           [ 6, 14],
           [11, 19]])
    """
    count = len(node_seconds)
    nearest = np.searchsorted(node_seconds, epoch_seconds)
    nearest = np.clip(nearest, 1, count - 1)
    nearest = np.where(np.abs(node_seconds[nearest - 1] - epoch_seconds) <= np.abs(node_seconds[nearest] - epoch_seconds),
                       nearest - 1, nearest)
    start = np.clip(nearest - points // 2, 0, count - points)
    return start[:, None] + np.arange(points)[None, :]


def interpolate(times, positions, velocities, epochs, points=POINTS):
    """
    Interpolate an orbit at epochs: Lagrange interpolation on the ``points`` nodes nearest each epoch.

    The window of nodes is centred on the node nearest the epoch and shifted inside the nodes at the ends. The
    positions are the Lagrange polynomial of the node positions; the velocities are the Lagrange polynomial of the
    node velocities, or, when the nodes have none (Delft ODR), the time derivative of the position polynomial.
    Node times may be uneven. No extrapolation.

    Parameters
    ----------
    times : array_like
        The node times, datetime64, increasing.
    positions : array_like
        (n, 3) node positions, m.
    velocities : array_like or None
        (n, 3) node velocities, m/s, or None.
    epochs : array_like
        The times to interpolate at, datetime64.
    points : int, optional
        The number of nodes of each interpolation. Default POINTS (9).

    Returns
    -------
    numpy.ndarray
        (m, 6): x, y, z (m) and vx, vy, vz (m/s) at the epochs.

    Raises
    ------
    ValueError
        An epoch outside the nodes, or fewer nodes than points.

    Examples
    --------
    >>> interpolate(orbit['times'], orbit['vectors'][:, :3], orbit['vectors'][:, 3:], orbit['times'][:3])
    """
    times = np.atleast_1d(_ns(times))
    epochs = np.atleast_1d(_ns(epochs))
    positions = np.asarray(positions, dtype=float)
    count = len(times)
    if count < points:
        raise ValueError(f'Orbit has {count} vectors, {points} needed')
    if np.any(epochs < times[0]) or np.any(epochs > times[-1]):
        raise ValueError(f'Orbit time outside {times[0]} .. {times[-1]}')
    reference = epochs[0]
    node_seconds = _seconds(times, reference)
    epoch_seconds = _seconds(epochs, reference)
    index = _nodes(node_seconds, epoch_seconds, points)
    weights, derivatives = _lagrange(node_seconds[index], epoch_seconds)
    position = np.einsum('kj,kjc->kc', weights, positions[index])
    if velocities is None:
        velocity = np.einsum('kj,kjc->kc', derivatives, positions[index])
    else:
        velocity = np.einsum('kj,kjc->kc', weights, np.asarray(velocities, dtype=float)[index])
    return np.hstack([position, velocity])


def window(satellite, start, stop):
    """
    The epochs of the orbit window of a scene: whole multiples of the satellite's step (STEPS) from MARGIN seconds
    before the start to MARGIN seconds after the stop, both ends rounded outwards.

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.

    Returns
    -------
    numpy.ndarray
        datetime64[ns] epochs.

    Raises
    ------
    ValueError
        An unknown satellite, or stop before start.

    Examples
    --------
    >>> window('ERS-2', '1997-12-18T09:49:12', '1997-12-18T09:49:30')[[0, -1]]
    array(['1997-12-18T09:35:00.000000000', '1997-12-18T10:03:30.000000000'],
          dtype='datetime64[ns]')
    """
    if satellite not in STEPS:
        raise ValueError(f'Unknown satellite {satellite!r}, expected one of {list(STEPS)}')
    start, stop = _ns(start), _ns(stop)
    if stop < start:
        raise ValueError(f'Scene stop {stop} is before its start {start}')
    step = STEPS[satellite]
    epoch = np.datetime64('1970-01-01T00:00:00', 'ns')
    first = int(np.floor((_seconds(start, epoch) - MARGIN) / step))
    last = int(np.ceil((_seconds(stop, epoch) + MARGIN) / step))
    return epoch + np.arange(first, last + 1, dtype=np.int64) * step * np.timedelta64(1_000_000_000, 'ns')


class _NoAuth(requests.auth.AuthBase):
    """
    The authentication of a _NetrcFreeSession: none, the request passes unchanged. A session with an auth never looks up
    ~/.netrc for a request (requests reads it only for a session and a request without one), and the environment's
    proxies still apply.
    """

    def __call__(self, request):
        """
        The request, unchanged.

        Parameters
        ----------
        request : requests.PreparedRequest
            The request.

        Returns
        -------
        requests.PreparedRequest

        Examples
        --------
        >>> _NoAuth()(requests.Request('GET', 'https://ersorbits.insar.dev/').prepare()).url
        'https://ersorbits.insar.dev/'
        """
        return request


class _NetrcFreeSession(requests.Session):
    """
    The HTTP session of every ESA module (the orbit reads here, ESA's catalogue queries and downloads): it never reads
    ~/.netrc. Its auth (_NoAuth) skips the netrc lookup of every request, and rebuild_auth that of every redirect (requests
    reads the netrc entry of the redirect's host there whatever the session's auth). The environment's proxies and CA bundle
    apply as usual.
    """

    def __init__(self):
        """
        A new session with no credentials.

        Examples
        --------
        >>> _NetrcFreeSession().auth is not None
        True
        """
        super().__init__()
        self.auth = _NoAuth()

    def rebuild_auth(self, prepared_request, response):
        """
        On a redirect, drop the Authorization header when the host changes, as requests does, and never add one from
        ~/.netrc.

        Parameters
        ----------
        prepared_request : requests.PreparedRequest
            The redirected request.
        response : requests.Response
            The redirect answer.

        Returns
        -------
        None

        Examples
        --------
        >>> _NetrcFreeSession().rebuild_auth(prepared_request, response)
        """
        headers = prepared_request.headers
        if 'Authorization' in headers and self.should_strip_auth(response.request.url, prepared_request.url):
            del headers['Authorization']


def _fetch(session, url, limit, retries=10, wait=3.0, timeout=(10, 120)):
    """
    GET a body into memory, read at most ``limit`` bytes; retried on failures a retry can change (HTTP.final).

    Parameters
    ----------
    session : requests.Session
        The session.
    url : str
        The URL.
    limit : int
        Largest body accepted, bytes (HTTP.read_body).
    retries : int, optional
        Attempts, the first included. Default 10.
    wait : float, optional
        Seconds between attempts. Default 3.
    timeout : tuple, optional
        (connect, read) timeouts, seconds. Default (10, 120).

    Returns
    -------
    bytes
        The body.

    Raises
    ------
    HTTP.NotFound
        HTTP 404 or 410.
    Exception
        A failure a retry cannot change, at once, or the last failure.

    Examples
    --------
    >>> _fetch(_NetrcFreeSession(), BASE_URL + 'ERS2/1997/12/18/index.csv', INDEX_LIMIT)[:4]
    b'path'
    """
    import time

    from . import HTTP

    count = HTTP.attempts(retries)
    for attempt in range(count):
        try:
            with HTTP.send(session.get, url, stream=True, timeout=timeout, check=False) as response:
                if response.status_code in (404, 410):
                    raise HTTP.NotFound(f'{url}: {HTTP.reply(response)}')
                HTTP.raise_for_status(response, url)
                return HTTP.read_body(response, limit=limit)
        except HTTP.NotFound:
            raise
        except Exception as e:
            stop = HTTP.final(e)
            if stop or attempt + 1 == count:
                print(f'ERROR: download attempt {attempt + 1}/{count} failed{" (not retried)" if stop else ""} '
                      f'for {url}: {HTTP.returned(e) or e}')
                raise
            time.sleep(wait)


def _member(blob, sha256):
    """
    The only member of an orbit zip, checked against the index's SHA-256.

    Parameters
    ----------
    blob : bytes
        The zip.
    sha256 : str
        The SHA-256 of the member, from the index.

    Returns
    -------
    tuple
        ``(name, data)``.

    Raises
    ------
    ValueError
        Not one member, a member above FILE_LIMIT, or a SHA-256 mismatch.

    Examples
    --------
    >>> _member(open('ODR_dgm-e04_278.zip', 'rb').read(), 'c1e07436...')[0]
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if len(infos) != 1:
            raise ValueError(f'Orbit zip holds {len(infos)} members, 1 expected')
        if infos[0].file_size > FILE_LIMIT:
            raise ValueError(f'Orbit file {infos[0].filename} of {infos[0].file_size} bytes exceeds {FILE_LIMIT}')
        data = archive.read(infos[0])
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f'Orbit file {infos[0].filename}: SHA-256 differs from the index')
    return infos[0].filename, data


def _edge(rows):
    """
    The least distance between a scene and the end of its cut window: EDGE of the arcs' format (the prefix of the zip name), the
    largest of the arcs given.

    Parameters
    ----------
    rows : list of dict
        Index rows (``parse_index``).

    Returns
    -------
    int
        Seconds.

    Raises
    ------
    ValueError
        A zip name of no format of EDGE.

    Examples
    --------
    >>> _edge([{'path': 'ERS1/1998/08/07/PRC_980807_37118_rev2.zip'}, {'path': 'ERS2/1997/12/16/ODR_dgm-e04_278.zip'}])
    160
    """
    edge = 0
    for row in rows:
        name = row['path'].rsplit('/', 1)[-1]
        prefix = next((prefix for prefix in EDGE if name.startswith(prefix)), None)
        if prefix is None:
            raise ValueError(f'Unsupported orbit file: {name}')
        edge = max(edge, EDGE[prefix])
    return edge


def _jump_edge(rows):
    """
    The least distance between a scene and a joint above JUMP of its set's arcs: JUMP_EDGE of the arcs' format (the prefix of the zip
    name), the largest of the arcs given; 0 for a format not in JUMP_EDGE.

    Parameters
    ----------
    rows : list of dict
        Index rows (``parse_index``).

    Returns
    -------
    int
        Seconds.

    Examples
    --------
    >>> _jump_edge([{'path': 'ERS1/1998/08/07/PRC_980807_37118_rev2.zip'}, {'path': 'ERS2/1997/12/16/ODR_dgm-e04_278.zip'}])
    1020
    """
    names = [row['path'].rsplit('/', 1)[-1] for row in rows]
    return max((JUMP_EDGE[prefix] for name in names for prefix in JUMP_EDGE if name.startswith(prefix)), default=0)


def check_jumps(satellite, start, stop, joints):
    """
    Refuse a scene near a jump of its set: a joint of two consecutive arcs whose positions differ there by more than JUMP metres
    lies within JUMP_EDGE seconds (``_jump_edge``: 1020 for D-PAF PRC) of the scene's lines, before or after it, or inside it.

    At such a joint one of the two arcs is wrong near its end, by up to the jump, and the arcs of the repository agree with the
    other producers within the repository's confirmation threshold, 10 m, only from up to 720 s from the joint on (JUMP_EDGE:
    that worst case with a margin of 300 s). A scene that ends JUMP_EDGE seconds or more
    before the joint, or starts that long after it, is kept. A joint at or below JUMP is no jump (``own_arc``), and a format
    without a JUMP_EDGE has none (its window is cut at the jump, ``cut_at_jumps``).

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    joints : list of dict
        The joints of consecutive arcs of the scene's set near it: ``'at'`` the joint (the first record of the later arc),
        ``'jump'`` the position difference of the two arcs there in metres (``_splice_jump``; infinity where they cannot be
        compared, a jump), ``'first'`` and ``'second'`` the index rows of the two arcs.

    Raises
    ------
    ValueError
        One line naming the satellite, the scene, the joint nearest the scene, its two arcs, its jump and the distance.

    Examples
    --------
    >>> check_jumps('ERS-2', '2010-03-15T07:04:39', '2010-03-15T07:04:54', [dict(at=np.datetime64('2010-03-15T08:06:00'), jump=11.9,
    ...             first={'path': 'ERS2/2010/03/07/PRC_100307_77794_rev2.zip'}, second={'path': 'ERS2/2010/03/15/PRC_100315_77902_rev2.zip'})])
    """
    start, stop = _ns(start), _ns(stop)
    second = np.timedelta64(1_000_000_000, 'ns')
    centre = start + (stop - start) / 2
    near = []
    for joint in joints:
        if joint['jump'] <= JUMP:
            continue
        at = _ns(joint['at'])
        edge = _jump_edge([joint['first'], joint['second']])
        if start - edge * second < at < stop + edge * second:
            near.append((abs(at - centre), at, edge, joint))
    if near:
        _, at, edge, joint = min(near, key=lambda item: item[:2])
        first, after = (joint[key]['path'].rsplit('/', 1)[-1] for key in ('first', 'second'))
        raise ValueError(f'No precise orbit of {satellite} for the scene {start} .. {stop}: it lies within {edge} s of a jump of '
                         f'{joint["jump"]:.1f} m between {first} and {after} at {at}')


def _short(satellite, start, stop, where, edge):
    """
    The one-line refusal of a scene whose window, cut at ``where``, does not reach ``edge`` seconds beyond it.

    Parameters
    ----------
    satellite : str
        The satellite.
    start, stop : numpy.datetime64
        The scene's first and last line times.
    where : str
        The cut, as words.
    edge : int
        Seconds (``_edge``).

    Returns
    -------
    str

    Examples
    --------
    >>> _short('ERS-2', np.datetime64('1999-05-10T23:59:00'), np.datetime64('1999-05-10T23:59:16'), 'the end of set a', 160)
    'No precise orbit of ERS-2 for the scene 1999-05-10T23:59:00 .. 1999-05-10T23:59:16: the window cut at the end of set a does not reach 160 s beyond the scene'
    """
    need = 'does not hold the scene' if edge == 0 else f'does not reach {edge} s beyond the scene'
    return f'No precise orbit of {satellite} for the scene {start} .. {stop}: the window cut at {where} {need}'


def select_orbit(satellite, start, stop, indexes):
    """
    The set, arcs and epochs of a scene's orbit, from the day indexes of the repository (the per-day rule).

    The set is the one of the index of the scene's own UTC day (the day of its first line): all scenes of a day take
    one set. Only that set's arcs are used, listed in any of the given indexes (a window across midnight reads the
    other day's index too, whose rows of another set are left out): one set per window, never two. The window is
    ``window``; where the set's arcs (joined where consecutive) do not cover all of it, it is cut to their coverage,
    and it must still reach EDGE seconds (``_edge``: 0 for PRC, 160 for Delft ODR, 90 for DORIS) beyond each end of the
    scene; the interpolation near the cut is one-sided. The arcs over the (cut) window are chosen by ``select_arcs``.

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    indexes : dict
        The rows (``parse_index``) of each UTC day the window touches, keyed by the day as 'YYYY-MM-DD'; a day
        without an index is missing or None.

    Returns
    -------
    dict
        ``'times'`` the (cut) window epochs, datetime64[ns]; ``'arcs'`` the selected rows in time order; ``'set'``
        and ``'model'``; ``'cut'`` True when the window was cut.

    Raises
    ------
    ValueError
        One line naming the satellite and the scene: the scene's day has no index or lists several sets, the set
        does not cover the scene, the window cut to the set does not reach EDGE seconds beyond the scene, or an arc of
        the set has a zip name of no format of EDGE.

    Examples
    --------
    >>> row = parse_index('path,start,end,source,set,model,produced,sha256\\n'
    ...                   'ERS2/1997/12/16/ODR_dgm-e04_278.zip,1997-12-16T00:00:00,1997-12-21T12:00:00,TU Delft DEOS,'
    ...                   'ODR.ERS-2/dgm-e04,DGM-E04,1998-12-15T17:33:32,c1e0')
    >>> select_orbit('ERS-2', '1997-12-18T09:49:12', '1997-12-18T09:49:30', {'1997-12-18': row})['set']
    'ODR.ERS-2/dgm-e04'
    """
    epochs = window(satellite, start, stop)
    start, stop = _ns(start), _ns(stop)
    second = np.timedelta64(1_000_000_000, 'ns')
    day = str(start.astype('datetime64[D]'))
    own = indexes.get(day) or []
    if not own:
        raise ValueError(f'No precise orbit of {satellite} on {day}: the day has no index')
    names = list(dict.fromkeys(row['set'] for row in own))
    if len(names) != 1:
        raise ValueError(f'The orbit index of {satellite} {day} lists {len(names)} sets: {names}')
    name = names[0]
    unique = {}
    for key in sorted(indexes):
        for row in indexes[key] or []:
            if row['set'] == name:
                unique.setdefault(row['path'], row)
    rows = sorted(unique.values(), key=lambda row: row['start'])
    # the coverage of the set: consecutive arcs joined (each starting at or before the end of the ones before)
    spans = []
    for row in rows:
        if spans and row['start'] <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], row['end'])
        else:
            spans.append([row['start'], row['end']])
    span = [s for s in spans if s[0] <= start and stop <= s[1]]
    if not span:
        raise ValueError(f'No precise orbit of {satellite} covers the scene {start} .. {stop} (set {name})')
    low, high = span[0]
    times = epochs[(epochs >= low) & (epochs <= high)]
    edge = _edge(rows)
    reach = edge * second
    if len(times) == 0 or times[0] > start - reach or times[-1] < stop + reach:
        raise ValueError(_short(satellite, start, stop, f'the ends of set {name} ({low} .. {high})', edge))
    arcs = select_arcs(rows, times[0], times[-1])
    return {'times': times, 'arcs': arcs, 'set': name, 'model': arcs[0]['model'], 'cut': len(times) < len(epochs)}


def parse_no_orbit(text):
    """
    Parse the list of unverifiable orbit windows of a satellite (``<SATELLITE>/no_orbit.csv`` of the repository).

    Parameters
    ----------
    text : str or bytes
        The header ``start,end,reason`` and one row per window, ``start`` and ``end`` in UTC ISO.

    Returns
    -------
    list of dict
        One dict per window: ``'start'`` and ``'end'`` (datetime64[ns] UTC) and ``'reason'`` (str).

    Raises
    ------
    ValueError
        The header is not ``start,end,reason``, or a window ends before it starts.

    Examples
    --------
    >>> parse_no_orbit('start,end,reason\\n1995-03-21T05:45:00,1995-03-21T08:21:00,no set is verified')[0]['reason']
    'no set is verified'
    """
    import csv

    if isinstance(text, bytes):
        text = text.decode('utf-8')
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header != ['start', 'end', 'reason']:
        raise ValueError(f'Not a no_orbit.csv header: {header}')
    spans = []
    for values in reader:
        if not values:
            continue
        row = dict(zip(header, values))
        row['start'], row['end'] = _ns(row['start']), _ns(row['end'])
        if row['end'] < row['start']:
            raise ValueError(f'no_orbit.csv window ends before it starts: {values}')
        spans.append(row)
    return spans


def check_no_orbit(satellite, spans, first, last):
    """
    Refuse an orbit window that overlaps an unverifiable window of the satellite.

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    spans : list of dict
        The unverifiable windows, as ``parse_no_orbit`` returns them.
    first, last : datetime-like
        The orbit window, UTC.

    Raises
    ------
    ValueError
        One line naming the satellite, the orbit window, the listed window and its reason.

    Examples
    --------
    >>> check_no_orbit('ERS-2', [], '2002-03-09T10:00:00', '2002-03-09T10:30:00')
    """
    first, last = _ns(first), _ns(last)
    for span in spans:
        if span['start'] <= last and first <= span['end']:
            raise ValueError(f'No verified precise orbit of {satellite} for the window {first} .. {last}: it overlaps '
                             f'{span["start"]} .. {span["end"]} ({span["reason"]})')


def _splice_jump(first, second, at):
    """
    The position difference of two decoded arcs at one instant, the jump of a joint between them.

    Parameters
    ----------
    first, second : dict
        Decoded arcs (``parse_orbit_file``), cut to the span the repository serves.
    at : numpy.datetime64
        The instant, the first node of ``second``.

    Returns
    -------
    float
        Metres; infinity when either arc holds no 9 nodes around the instant (the arcs cannot be compared).

    Examples
    --------
    >>> _splice_jump(arc_a, arc_b, arc_b['times'][0])
    0.12
    """
    for part in (first, second):
        if len(part['times']) < POINTS or not part['times'][0] <= at <= part['times'][-1]:
            return float('inf')
    one, two = (interpolate(part['times'], part['positions'], part['velocities'], [at])[0, :3] for part in (first, second))
    return float(np.linalg.norm(one - two))


def cut_at_jumps(satellite, start, stop, chosen, jumps):
    """
    Never join across a jump: cut a joined selection to the arcs on the scene's side of every jump.

    The arcs of one set meet at manoeuvres (ESA's PRC arcs end to end; Delft arcs overlap, and one can run through a
    burn the other models), and the orbits can differ by metres to kilometres where they meet. Of the arcs
    ``select_orbit`` chose (joined where none covers the window), two consecutive ones stay joined only when their
    position difference at the joint is at most JUMP metres. Where it is larger the arcs are split there; the scene's
    side is the run of arcs whose span holds the whole scene (the one with the nearest span centre when several do), and
    the window is cut to that run's span. It must still reach EDGE seconds (``_edge``) beyond each end of the scene; the
    interpolation near the cut is one-sided, inside the run's nodes.

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    chosen : dict
        The selection of ``select_orbit``.
    jumps : list of float
        The position difference in metres at each joint of consecutive arcs of ``chosen['arcs']`` (``_splice_jump``),
        one fewer than the arcs; infinity where the arcs cannot be compared.

    Returns
    -------
    dict
        ``chosen`` itself when no joint is a jump, else a dict of the same keys: the cut ``'times'``, the ``'arcs'`` of
        the scene's side and ``'cut'`` True.

    Raises
    ------
    ValueError
        One line naming the satellite, the scene and the jump: the scene straddles a jump, or the window cut at the jump
        does not reach EDGE seconds beyond the scene.

    Examples
    --------
    >>> arcs = [row('PRC_A.zip', '1998-08-07T00:00', '1998-08-10T12:00'), row('PRC_B.zip', '1998-08-10T12:00', '1998-08-14T00:00')]
    >>> cut_at_jumps('ERS-1', '1998-08-10T11:50:00', '1998-08-10T11:50:16', dict(times=window('ERS-1', '1998-08-10T11:50:00',
    ...              '1998-08-10T11:50:16'), arcs=arcs, set='s', model='M', cut=False), [500.0])['arcs'][0]['path']
    'PRC_A.zip'
    """
    arcs, times = chosen['arcs'], chosen['times']
    if len(jumps) != len(arcs) - 1:
        raise ValueError(f'{len(jumps)} jumps for {len(arcs)} arcs')
    bad = [k for k, jump in enumerate(jumps) if not jump <= JUMP]
    if not bad:
        return chosen
    start, stop = _ns(start), _ns(stop)
    centre = start + (stop - start) / 2
    edge = _edge(arcs)
    reach = edge * np.timedelta64(1_000_000_000, 'ns')
    # the joint nearest the scene is named
    near = min(bad, key=lambda k: abs(arcs[k + 1]['start'] - centre))
    where = (f'a jump of {jumps[near]:.1f} m between {arcs[near]["path"].rsplit("/", 1)[-1]} and '
             f'{arcs[near + 1]["path"].rsplit("/", 1)[-1]} at {arcs[near + 1]["start"]}')
    cuts = [0] + [k + 1 for k in bad] + [len(arcs)]
    holding, found = [], []
    for low, high in zip(cuts[:-1], cuts[1:]):
        run = arcs[low:high]
        begin, end = run[0]['start'], max(row['end'] for row in run)
        if not (begin <= start and stop <= end):
            continue
        cut = times[(times >= begin) & (times <= end)]
        holding.append(run)
        if len(cut) and cut[0] <= start - reach and cut[-1] >= stop + reach:
            found.append((abs(begin + (end - begin) / 2 - centre), len(found), run, cut))
    if not holding:
        raise ValueError(f'No precise orbit of {satellite} for the scene {start} .. {stop}: it straddles {where}')
    if not found:
        raise ValueError(_short(satellite, start, stop, where, edge))
    _, _, run, cut = min(found, key=lambda item: item[:2])
    return dict(chosen, times=cut, arcs=run, cut=True)


def own_arc(satellite, start, stop, chosen, joints):
    """
    The scene's own arc where arcs are joined: a scene that lies within one of the joined arcs takes that arc alone, the
    window cut at the joints, so that the interpolation near the scene never reads the nodes of two arcs.

    ``cut_at_jumps`` leaves arcs joined where they differ by at most JUMP metres; the 9-point interpolation then reads
    nodes of both arcs near the joint, which moves the orbit near the joint by up to 1.35 m and 12 mm/s from the arc that
    covers the scene (325 PRC joints, session/patches/esa_alos1/reports/FABLE_ESA_review.md F2). Arc k holds the times
    from the joint before it (the first record of arc k) to the joint after it (the first record of arc k + 1), the ends of
    the joined span at the ends. A scene inside the holding of one arc takes it alone, with the window cut to that holding
    however few steps remain beyond the scene: the interpolation near the joint is one-sided, its 9-point stencil shifted
    inside that arc's nodes, never across the joint. The cut window must reach EDGE seconds (``_edge``: 0 for PRC, whose
    arcs meet end to end) beyond each end of the scene. A scene that itself straddles a joint keeps the joined arcs
    (``chosen`` unchanged).

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    chosen : dict
        The selection of ``select_orbit`` after ``cut_at_jumps``.
    joints : list of numpy.datetime64
        The first record of each arc of ``chosen['arcs']`` after the first, one fewer than the arcs.

    Returns
    -------
    dict
        ``chosen`` itself for one arc or a scene that straddles a joint, else a dict of the same keys: the cut ``'times'``,
        ``'arcs'`` the scene's arc alone and ``'cut'`` True.

    Raises
    ------
    ValueError
        One line naming the satellite, the scene and the joint: the scene lies within one arc, and the window cut at the
        joint does not reach EDGE seconds beyond the scene.

    Examples
    --------
    >>> arcs = [row('PRC_A.zip', '1998-08-07T00:00', '1998-08-10T12:00'), row('PRC_B.zip', '1998-08-10T12:00', '1998-08-14T00:00')]
    >>> own_arc('ERS-1', '1998-08-10T11:59:00', '1998-08-10T11:59:16', dict(times=window('ERS-1', '1998-08-10T11:59:00',
    ...         '1998-08-10T11:59:16'), arcs=arcs, set='s', model='M', cut=False), [np.datetime64('1998-08-10T12:00')])['arcs'][0]['path']
    'PRC_A.zip'
    """
    arcs, times = chosen['arcs'], chosen['times']
    if len(joints) != len(arcs) - 1:
        raise ValueError(f'{len(joints)} joints for {len(arcs)} arcs')
    if len(arcs) < 2:
        return chosen
    start, stop = _ns(start), _ns(stop)
    joints = [_ns(joint) for joint in joints]
    bounds = [times[0]] + joints + [times[-1]]
    for k in range(len(arcs)):
        low, high = bounds[k], bounds[k + 1]
        if not (low <= start and stop <= high):
            continue
        cut = times[(times >= low) & (times <= high)]
        edge = _edge([arcs[k]])
        reach = edge * np.timedelta64(1_000_000_000, 'ns')
        if len(cut) == 0 or cut[0] > start - reach or cut[-1] < stop + reach:
            near = low if k > 0 and (k + 1 == len(arcs) or start - low < high - stop) else high
            raise ValueError(_short(satellite, start, stop, f'the joint of two arcs at {near}', edge))
        return dict(chosen, times=cut, arcs=[arcs[k]], cut=True)
    return chosen


def _clip(part, row):
    """
    A decoded arc cut to the span the repository serves for it (the index row's ``start`` .. ``end``).

    Parameters
    ----------
    part : dict
        The arc, as ``parse_orbit_file`` returns it.
    row : dict
        The index row of the arc.

    Returns
    -------
    dict
        The arc with ``'times'``, ``'positions'`` and ``'velocities'`` (when it has them) of the nodes inside the span.

    Examples
    --------
    >>> _clip(parse_odr(odr_bytes), row)['times'][0]
    """
    keep = (part['times'] >= row['start']) & (part['times'] <= row['end'])
    return dict(part, times=part['times'][keep], positions=part['positions'][keep],
                velocities=None if part['velocities'] is None else part['velocities'][keep])


def orbit_vectors(satellite, start, stop, session=None, cache=None):
    """
    The precise orbit of a scene from the orbit repository (BASE_URL), on the scene's window (``window``).

    The day index of every UTC day the window touches is read (the scene's own day must have one); the set, the arc
    or joined arcs and the window (cut to the set where it does not cover all of it) are chosen by
    ``select_orbit``; the zips are read into memory (capped reads, timeouts and retries), checked against the
    index's SHA-256, decoded and cut to the span of their index row. A scene within JUMP_EDGE seconds (1020 for D-PAF PRC) of a
    joint of its set's arcs whose positions differ by more than JUMP metres is refused (``check_jumps``; the indexes of the days
    within that distance and the arcs of the joints there are read). Joined arcs are never joined across a jump
    (``cut_at_jumps``), and a scene that lies within one of the joined arcs takes that arc alone (``own_arc``). The list
    ``<SATELLITE>/no_orbit.csv`` of unverifiable windows is read once (a missing file, HTTP 404, means no list, with a
    WARNING); a window that overlaps a listed one is refused. The vectors are interpolated to the window epochs by
    ``interpolate``. No leap second may lie inside the span of the nodes the interpolation reads (the window and the nodes
    beyond its ends).

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    session : requests.Session, optional
        The HTTP session; when None, a new _NetrcFreeSession (it never reads ~/.netrc), closed at the end.
    cache : dict, optional
        Keeps the repository's files read (the indexes, the no_orbit list, the zips; a file not served as well) for the
        next calls given the same dict, so the scenes of one call read each file once. None reads every file anew.

    Returns
    -------
    dict
        ``'times'`` datetime64[ns] UTC epochs; ``'vectors'`` (n, 6) Earth-fixed x, y, z (m) and vx, vy, vz (m/s);
        ``'source'`` list of the zip names used; ``'set'`` and ``'model'`` of the arcs (index columns); ``'step'``
        seconds; ``'cut'`` True when the window was cut (to the set or at a jump).

    Raises
    ------
    ValueError
        One line naming the satellite and the scene: no orbit (``select_orbit``), a jump near the scene (``check_jumps``) or
        at the window's cut (``cut_at_jumps``), an
        unverifiable window (``check_no_orbit``), records that do not cover the window, a leap second inside the nodes
        read, or an unknown satellite.

    Examples
    --------
    >>> orbit_vectors('ERS-2', '1997-12-18T09:49:12', '1997-12-18T09:49:30')['source']
    ['ODR_dgm-e04_278.zip']
    """
    from . import HTTP

    window(satellite, start, stop)
    own = session is None
    session = _NetrcFreeSession() if own else session

    def get(path, limit):
        if cache is None:
            return _fetch(session, BASE_URL + path, limit)
        if path not in cache:
            try:
                cache[path] = _fetch(session, BASE_URL + path, limit)
            except HTTP.NotFound as e:
                cache[path] = e
        if isinstance(cache[path], HTTP.NotFound):
            raise cache[path]
        return cache[path]
    try:
        return _orbit(satellite, start, stop, get)
    finally:
        if own:
            session.close()


def _orbit(satellite, start, stop, get):
    """
    ``orbit_vectors`` over a function that reads the repository: ``get(path, limit)`` returns the bytes of a file of the
    repository (path relative to its root, at most ``limit`` bytes) or raises ``HTTP.NotFound``.

    Parameters
    ----------
    satellite : str
        'ENVISAT', 'ERS-1' or 'ERS-2'.
    start, stop : datetime-like
        The scene's first and last line times, UTC.
    get : callable
        The reader of the repository.

    Returns
    -------
    dict
        As ``orbit_vectors``.

    Examples
    --------
    >>> _orbit('ERS-2', '1997-12-18T09:49:12', '1997-12-18T09:49:30', get)['set']
    'ODR.ERS-2/dgm-e04'
    """
    from . import HTTP

    epochs = window(satellite, start, stop)
    step = STEPS[satellite]
    begin = _ns(start)
    folder = SATELLITES[satellite]
    try:
        unverified = parse_no_orbit(get(f'{folder}/{NO_ORBIT}', INDEX_LIMIT))
    except HTTP.NotFound:
        unverified = []
        print(f'WARNING: {BASE_URL}{folder}/{NO_ORBIT} is not served: the windows of {satellite} without a verified orbit '
              f'are not listed')
    indexes = {}
    day = epochs[0].astype('datetime64[D]')
    while day <= epochs[-1].astype('datetime64[D]'):
        try:
            indexes[str(day)] = parse_index(get(f'{folder}/{str(day).replace("-", "/")}/index.csv', INDEX_LIMIT))
        except HTTP.NotFound:
            indexes[str(day)] = None
        day = day + np.timedelta64(1, 'D')
    chosen = select_orbit(satellite, start, stop, indexes)
    decoded = {}

    def decode(arc):
        """
        The member name and the decoded arc of an index row, cut to its span (``_clip``), each zip read once per call.

        Examples
        --------
        >>> decode(chosen['arcs'][0])[0]
        """
        if arc['path'] not in decoded:
            name, data = _member(get(arc['path'], ZIP_LIMIT), arc['sha256'])
            decoded[arc['path']] = name, _clip(parse_orbit_file(name, data), arc)
        return decoded[arc['path']]
    # the joints of the set within JUMP_EDGE of the scene, beyond the window too (the indexes of those days are read)
    edge = _jump_edge(chosen['arcs'])
    if edge:
        reach = edge * np.timedelta64(1_000_000_000, 'ns')
        low, high = begin - reach, _ns(stop) + reach
        day = low.astype('datetime64[D]')
        while day <= high.astype('datetime64[D]'):
            if str(day) not in indexes:
                try:
                    indexes[str(day)] = parse_index(get(f'{folder}/{str(day).replace("-", "/")}/index.csv', INDEX_LIMIT))
                except HTTP.NotFound:
                    indexes[str(day)] = None
            day = day + np.timedelta64(1, 'D')
        rows = sorted({row['path']: row for key in sorted(indexes) for row in indexes[key] or []
                       if row['set'] == chosen['set']}.values(), key=lambda row: row['start'])
        joints = []
        for first, after in zip(rows[:-1], rows[1:]):
            if after['start'] <= first['end'] and low < after['start'] < high:
                one, two = decode(first)[1], decode(after)[1]
                at = two['times'][0] if len(two['times']) else after['start']
                joints.append(dict(at=at, jump=_splice_jump(one, two, at), first=first, second=after))
        check_jumps(satellite, start, stop, joints)
    parts = []
    for arc in chosen['arcs']:
        name, part = decode(arc)
        if len(part['times']) == 0:
            raise ValueError(f'No precise orbit of {satellite} for the scene {begin}: {name} holds no record inside its '
                             f'span {arc["start"]} .. {arc["end"]}')
        parts.append(part)
    chosen_arcs = chosen['arcs']
    jumps = [_splice_jump(parts[k], parts[k + 1], parts[k + 1]['times'][0]) for k in range(len(parts) - 1)]
    chosen = cut_at_jumps(satellite, start, stop, chosen, jumps)
    parts = [parts[next(k for k, row in enumerate(chosen_arcs) if row['path'] == arc['path'])] for arc in chosen['arcs']]
    # a scene within one of the joined arcs takes that arc alone, cut at the joints
    chosen_arcs = chosen['arcs']
    chosen = own_arc(satellite, start, stop, chosen, [part['times'][0] for part in parts[1:]])
    arcs, epochs = chosen['arcs'], chosen['times']
    parts = [parts[next(k for k, row in enumerate(chosen_arcs) if row['path'] == arc['path'])] for arc in arcs]
    check_no_orbit(satellite, unverified, epochs[0], epochs[-1])
    # joined arcs: each arc up to the first record of the next one
    times, positions, velocities = [], [], []
    for k, part in enumerate(parts):
        keep = slice(None) if k + 1 == len(parts) else part['times'] < parts[k + 1]['times'][0]
        times.append(part['times'][keep])
        positions.append(part['positions'][keep])
        velocities.append(None if part['velocities'] is None else part['velocities'][keep])
    times = np.concatenate(times)
    positions = np.concatenate(positions)
    velocities = None if any(v is None for v in velocities) else np.concatenate(velocities)
    if times[0] > epochs[0] or times[-1] < epochs[-1]:
        raise ValueError(f'No precise orbit of {satellite} covers the scene {begin}: the window is {epochs[0]} .. '
                         f'{epochs[-1]}, the records span {times[0]} .. {times[-1]}')
    if len(times) < POINTS:
        raise ValueError(f'No precise orbit of {satellite} for the scene {begin}: the arcs hold {len(times)} records, '
                         f'{POINTS} needed')
    # the span of the nodes the interpolation reads, the nodes beyond the window ends included
    index = _nodes(_seconds(times, epochs[0]), _seconds(epochs, epochs[0]))
    low, high = times[index.min()], times[index.max()]
    dates, _ = _leaps()
    if np.any((dates > low) & (dates <= high)):
        raise ValueError(f'A leap second lies inside the orbit nodes of {satellite} {low} .. {high} for the scene '
                         f'{begin}')
    return {'times': epochs, 'vectors': interpolate(times, positions, velocities, epochs),
            'source': [arc['path'].rsplit('/', 1)[-1] for arc in arcs], 'set': chosen['set'],
            'model': chosen['model'], 'step': step, 'cut': bool(chosen['cut'])}


def _mjd(times):
    """
    The Envisat MJD 2000 of UTC times: whole days since 2000-01-01 (may be negative), seconds of the day,
    microseconds of the second.

    Parameters
    ----------
    times : array_like
        datetime64 UTC times.

    Returns
    -------
    tuple of numpy.ndarray
        ``(days, seconds, microseconds)``, int64.

    Examples
    --------
    >>> _mjd(np.array(['1999-12-31T23:59:59.5'], dtype='datetime64[ns]'))
    (array([-1]), array([86399]), array([500000]))
    """
    micro = (_ns(np.atleast_1d(times)) - MJD_EPOCH).astype('timedelta64[us]').astype(np.int64)
    days = np.floor_divide(micro, 86_400_000_000)
    rest = micro - days * 86_400_000_000
    return days, rest // 1_000_000, rest % 1_000_000


def _from_mjd(days, seconds, micro):
    """
    UTC datetime64[ns] of Envisat MJD 2000 fields.

    Parameters
    ----------
    days, seconds, micro : array_like
        Days since 2000-01-01, seconds of the day, microseconds of the second.

    Returns
    -------
    numpy.ndarray
        datetime64[ns].

    Examples
    --------
    >>> _from_mjd([-1], [86399], [500000])
    array(['1999-12-31T23:59:59.500000000'], dtype='datetime64[ns]')
    """
    micro = (np.asarray(days, dtype=np.int64) * 86_400_000_000 + np.asarray(seconds, dtype=np.int64) * 1_000_000
             + np.asarray(micro, dtype=np.int64))
    return MJD_EPOCH + micro * np.timedelta64(1000, 'ns')


def read_state_vectors(main_params):
    """
    The 5 orbit state vectors of a MAIN PROCESSING PARAMS ADS record (offset SV_OFFSET, SV_SIZE bytes each:
    big-endian MJD 2000 as int32 days, uint32 seconds, uint32 microseconds, then int32 x, y, z in 1e-2 m and vx, vy,
    vz in 1e-5 m/s, Earth-fixed).

    Parameters
    ----------
    main_params : bytes
        The ADS record (10,069 bytes in ASAR and ERS SLC products).

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, vectors)``: 5 datetime64[ns] UTC epochs and (5, 6) m and m/s.

    Examples
    --------
    >>> read_state_vectors(header[7516:7516 + 10069])[0]
    """
    raw = np.frombuffer(main_params[SV_OFFSET:SV_OFFSET + SV_COUNT * SV_SIZE], dtype=np.uint8).reshape(SV_COUNT, SV_SIZE)
    days = raw[:, 0:4].copy().view('>i4').ravel()
    seconds = raw[:, 4:8].copy().view('>u4').ravel()
    micro = raw[:, 8:12].copy().view('>u4').ravel()
    values = raw[:, 12:36].copy().view('>i4').reshape(SV_COUNT, 6).astype(float)
    return _from_mjd(days, seconds, micro), np.hstack([values[:, :3] / 1e2, values[:, 3:] / 1e5])


def replace_state_vectors(main_params, orbit):
    """
    Replace the 5 orbit state vectors of a MAIN PROCESSING PARAMS ADS record by the precise orbit at their epochs.

    The epochs (MJD fields) are kept; positions and velocities are ``interpolate`` of the orbit's vectors at them,
    rounded to the stored units (1e-2 m, 1e-5 m/s). Every other byte is unchanged.

    Parameters
    ----------
    main_params : bytes
        The ADS record.
    orbit : dict
        The orbit as ``orbit_vectors`` returns it; it must cover the 5 epochs.

    Returns
    -------
    bytes
        The record with the 24 bytes of values of each state vector replaced.

    Raises
    ------
    ValueError
        A value that does not fit int32, or an epoch outside the orbit.

    Examples
    --------
    >>> replace_state_vectors(header[7516:7516 + 10069], orbit_vectors('ERS-2', start, stop))
    """
    times, _ = read_state_vectors(main_params)
    vectors = np.asarray(orbit['vectors'], dtype=float)
    values = interpolate(orbit['times'], vectors[:, :3], vectors[:, 3:], times)
    scaled = np.rint(np.hstack([values[:, :3] * 1e2, values[:, 3:] * 1e5]))
    if np.any(np.abs(scaled) >= _INT32):
        raise ValueError(f'State vector value {np.abs(scaled).max():.0f} does not fit int32')
    out = bytearray(main_params)
    for k in range(SV_COUNT):
        begin = SV_OFFSET + k * SV_SIZE + 12
        out[begin:begin + 24] = scaled[k].astype('>i4').tobytes()
    return bytes(out)


def orbit_ads(orbit):
    """
    The bytes of the precise-orbit dataset of a crop: one record (ORBIT_DSR_SIZE = 60 bytes, big-endian) per orbit
    epoch: MJD 2000 (int32 days since 2000-01-01, uint32 seconds, uint32 microseconds), then x, y, z (m) and vx, vy,
    vz (m/s) as float64.

    Parameters
    ----------
    orbit : dict
        The orbit as ``orbit_vectors`` returns it.

    Returns
    -------
    bytes
        n x 60 bytes.

    Examples
    --------
    >>> len(orbit_ads({'times': np.array(['2008-01-16T20:34:00'], dtype='datetime64[ns]'), 'vectors': np.zeros((1, 6))}))
    60
    """
    days, seconds, micro = _mjd(orbit['times'])
    dtype = np.dtype([('days', '>i4'), ('seconds', '>u4'), ('micro', '>u4'), ('vector', '>f8', (6,))])
    records = np.zeros(len(days), dtype=dtype)
    records['days'], records['seconds'], records['micro'] = days, seconds, micro
    records['vector'] = np.asarray(orbit['vectors'], dtype=float)
    return records.tobytes()


def parse_orbit_ads(data):
    """
    Read the precise-orbit dataset written by ``orbit_ads``.

    Parameters
    ----------
    data : bytes
        n x 60 bytes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, vectors)``: datetime64[ns] UTC and (n, 6).

    Raises
    ------
    ValueError
        A length that is not a multiple of 60.

    Examples
    --------
    >>> parse_orbit_ads(orbit_ads(orbit))[1].shape
    """
    if len(data) % ORBIT_DSR_SIZE:
        raise ValueError(f'Orbit dataset of {len(data)} bytes is not a multiple of {ORBIT_DSR_SIZE}')
    dtype = np.dtype([('days', '>i4'), ('seconds', '>u4'), ('micro', '>u4'), ('vector', '>f8', (6,))])
    records = np.frombuffer(data, dtype=dtype)
    return _from_mjd(records['days'], records['seconds'], records['micro']), records['vector'].astype(float)


def orbit_filename(orbit):
    """
    The FILENAME field of the precise-orbit DSD: the orbit file names of the source zips (without ``.zip``),
    joined by ``+``.

    Parameters
    ----------
    orbit : dict
        The orbit as ``orbit_vectors`` returns it.

    Returns
    -------
    str
        At most 62 characters.

    Raises
    ------
    ValueError
        The names exceed the 62 characters of the field.

    Examples
    --------
    >>> orbit_filename({'source': ['PRC_981231_39020_rev2.zip', 'PRC_990101_39024_rev2.zip']})
    'PRC_981231_39020_rev2+PRC_990101_39024_rev2'
    """
    names = '+'.join(name.removesuffix('.zip') for name in orbit['source'])
    if len(names) > 62:
        raise ValueError(f'Orbit source names exceed the 62 characters of a DSD FILENAME: {names}')
    return names


def orbit_dsd(orbit, offset):
    """
    The 280-byte DSD of the precise-orbit dataset (Envisat format: DS_NAME, DS_TYPE=A, FILENAME, DS_OFFSET,
    DS_SIZE, NUM_DSR, DSR_SIZE, then 32 blanks and a newline).

    Parameters
    ----------
    orbit : dict
        The orbit as ``orbit_vectors`` returns it.
    offset : int
        The byte offset of the dataset in the product.

    Returns
    -------
    bytes
        280 bytes.

    Examples
    --------
    >>> len(orbit_dsd({'times': np.zeros(28), 'source': ['ODR_dgm-e04_278.zip']}, 24854))
    280
    """
    count = len(orbit['times'])
    text = (f'DS_NAME="{ORBIT_DS_NAME:<28}"\n'
            f'DS_TYPE=A\n'
            f'FILENAME="{orbit_filename(orbit):<62}"\n'
            f'DS_OFFSET=+{offset:020d}<bytes>\n'
            f'DS_SIZE=+{count * ORBIT_DSR_SIZE:020d}<bytes>\n'
            f'NUM_DSR=+{count:010d}\n'
            f'DSR_SIZE=+{ORBIT_DSR_SIZE:010d}<bytes>\n'
            f'{"":32}\n')
    return text.encode('ascii')
