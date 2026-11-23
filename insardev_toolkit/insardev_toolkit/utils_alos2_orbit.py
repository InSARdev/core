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
ALOS-2 orbit of a CEOS leader (LED) file and the line window of a ScanSAR subswath that a bbox covers.

The LED file is a sequence of CEOS records (file descriptor, data set summary, platform position data, attitude,
radiometric, data quality, facility records). Each record starts with a 12-byte header: bytes 1-4 the sequence
number, byte 5 subtype 1, byte 6 the record type, bytes 7-8 subtypes 2-3, bytes 9-12 the record length (big-endian
int32, the header included). The platform position data record (type 30) holds the Earth-fixed state vectors; the
data set summary record (type 10) holds the range sampling rate.

ALOS-2 L1.1 subswath images are zero-Doppler: a ground point is imaged on the line whose time is the orbit's
zero-Doppler time of the point. ALOS-1 PALSAR L1.1 images are not: a point is imaged when its Doppler equals the
image's Doppler centroid, the LED terms fd1 + fdd1 * R (geo2rdr's ``doppler``). A bbox, mapped on height layers from -500 to 9000 m into (range bin, line) of a
subswath, gives the lines to read, the way the NISAR downloader uses its geolocation grid. An ALOS-1 raw (L1.0)
frame gives the time span of a region's echoes instead (raw_window), its lines then selected by their own time tags,
at the beam-centre Doppler of the platform's yaw (the attitude data record, type 40: raw_doppler) with the margin of
the frame's steering state (raw_margin).
The functions here are pure: no network and no file access.
"""
import math

import numpy as np

# the WGS84 ellipsoid and the speed of light
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_C = 299792458.0

# The Earth's rotation rate (WGS84), rad/s
_OMEGA = 7.292115e-5

# The margins of an ALOS-1 PALSAR raw (L1.0) window (raw_window), seconds before and after the region's beam-centre
# times: the raw leader files hold no Doppler centroid, so the beam centre comes from their yaw (raw_doppler), and the
# margin covers its difference to the image's own Doppler centroid; raw_margin picks it by JAXA's yaw steering mode flag
# of the raw leader file.
# Measured offline 2026-11-23 on 142 raw / L1.1 pairs of the same scenes (2006-2011; FBS, FBD, PLR; ascending and
# descending; the L1.1 Doppler centroid fd1 + fdd1 * R, data set summary bytes 1735-1766: the download reads no L1.1
# product), the beam-centre time of the yaw Doppler minus that of the L1.1 Doppler at the scene centre and the footprint
# corners (710 points; session/patches/esa_alos1/reports/AL1FIX3_round2_report.md):
# - the steering state is JAXA's own flag, the yaw steering mode flag of the raw leader file's data set summary record
#   (type 10, bytes 1831-1834, I4: ALOS PALSAR Level 1.0 Product Format Description Rev. J, Table 3.3-5 (12/13), field 133,
#   p. 3-37): 0 yaw steering, 1 none. It equals the true state on all 189 raw frames checked (2026-11-23); the yaw at the
#   frame's middle is no classifier: |yaw| 0.185..3.841 deg on the yaw-steered frames (it falls with the latitude, 0.185
#   at 83.4 deg), 0.000..0.067 deg on the frames without yaw steering (L1.1 Doppler centroids up to 2,489 Hz).
# - steered: -0.278..+0.170 s (|offset| median 0.032 s; -74..+133 Hz), largest 0.278 s (ALPSRP202855820, FBS,
#   2009-11-14); ALOS1_DOPPLER_MARGIN 0.38 s (1.25 x 0.302 s, the largest offset from zero Doppler of 340 yaw-steered L1.1
#   products of 2006-2011) holds it. The Fable re-check (653 raw/L1.1 pairs, 2006-09..2011-04, every second day, and every
#   frame of orbit 5785) found steered offsets up to 0.288 s; every pair held its lines, the thinnest steered spare 0.092 s
#   (ALPSRP210165790, 2010-01-03) and unsteered 0.466 s (FABLE_ALOS1_review.md, "Re-check of F1 after AL1FIX3").
# - not steered: -0.235..+0.632 s (|offset| median 0.066 s; -302..+112 Hz), largest 0.632 s (ALPSRP057875700, FBS,
#   2007-02-24, latitude -70.5; 0.578 s ALPSRP057875710): ALOS1_DOPPLER_MARGIN_UNSTEERED 1.25 x 0.632 s rounded up to
#   0.05 s, 0.80 s.
ALOS1_DOPPLER_MARGIN = (0.38, 0.38)
ALOS1_DOPPLER_MARGIN_UNSTEERED = (0.80, 0.80)
# A raw frame's end is estimated from its first line (its time, line count and line rate); a PRF change inside the frame
# moves it (2092.05 -> 2123.142 Hz measured on ALPSRP055006320-L1.0: 1.5 %, at most 0.25 s over the 16.4 s of a frame).
# A region is after the frame when its span starts more than this many seconds after the estimated end.
ALOS1_FRAME_END_SLACK = 1.0


def led_records(led):
    """
    List the CEOS records of LED file bytes.

    Parameters
    ----------
    led : bytes
        The LED file content.

    Returns
    -------
    list of tuple
        ``(offset, record_type, length)`` per record, walked from offset 0 by the record lengths; the walk stops at
        the end of the bytes or at a record length <= 0.

    Examples
    --------
    >>> led_records(open('AL2_086_1150_F1_20250512.led', 'rb').read())
    """
    records = []
    offset = 0
    while offset + 12 <= len(led):
        length = int.from_bytes(led[offset + 8:offset + 12], 'big', signed=True)
        if length <= 0:
            break
        records.append((offset, led[offset + 5], length))
        offset += length
    return records


def _record(led, record_type):
    """
    The bytes of the first CEOS record of a type.

    Parameters
    ----------
    led : bytes
        The LED file content.
    record_type : int
        The record type (byte 6 of the record header).

    Returns
    -------
    bytes or None
        The record, its header included; None when there is no record of the type.

    Examples
    --------
    >>> _record(open('AL2_086_1150_F1_20250512.led', 'rb').read(), 30)
    """
    for offset, kind, length in led_records(led):
        if kind == record_type:
            return led[offset:offset + length]
    return None


def _field(record, first, last):
    """
    The text of bytes of a record.

    Parameters
    ----------
    record : bytes
        The record, its header included.
    first, last : int
        The 1-based inclusive byte positions in the record.

    Returns
    -------
    str
        The ASCII text, stripped.

    Examples
    --------
    >>> _field(b' ' * 140 + b'  28', 141, 144)
    '28'
    """
    return record[first - 1:last].decode('ascii').strip()


def led_orbit(led):
    """
    Read the orbit state vectors of LED file bytes (the first platform position data record, type 30).

    Parameters
    ----------
    led : bytes
        The LED file content.

    Returns
    -------
    dict
        ``'date'`` 'YYYY-MM-DD' of the first point, ``'seconds'`` its seconds of the day, ``'step'`` the interval
        between the points in seconds, ``'vectors'`` an array (n, 6) of Earth-fixed x, y, z (m) and vx, vy, vz (m/s).

    Raises
    ------
    ValueError
        No type-30 record, or fewer than 8 points.

    Examples
    --------
    >>> led_orbit(open('AL2_086_1150_F1_20250512.led', 'rb').read())['step']
    """
    record = _record(led, 30)
    if record is None:
        raise ValueError('LED has no platform position record (type 30)')
    count = int(_field(record, 141, 144))
    if count < 8:
        raise ValueError(f'LED orbit has {count} points, 8 needed')
    if 386 + count * 132 > len(record):
        raise ValueError(f'LED orbit record too short for {count} points')
    year, month, day = (int(_field(record, a, a + 3)) for a in (145, 149, 153))
    seconds = float(_field(record, 161, 182))
    step = float(_field(record, 183, 204))
    vectors = np.array([float(record[386 + 22 * k:386 + 22 * (k + 1)].decode('ascii').strip())
                        for k in range(count * 6)]).reshape(count, 6)
    return {'date': f'{year:04d}-{month:02d}-{day:02d}', 'seconds': seconds, 'step': step, 'vectors': vectors}


def led_sampling_rate(led):
    """
    Read the range sampling rate of LED file bytes (the first data set summary record, type 10, bytes 711-726, MHz).

    Parameters
    ----------
    led : bytes
        The LED file content.

    Returns
    -------
    float
        The range sampling rate in Hz.

    Raises
    ------
    ValueError
        No type-10 record.

    Examples
    --------
    >>> led_sampling_rate(open('AL2_086_1150_F1_20250512.led', 'rb').read())
    """
    record = _record(led, 10)
    if record is None:
        raise ValueError('LED has no data set summary record (type 10)')
    return float(_field(record, 711, 726)) * 1e6


def led_attitude(led):
    """
    Read the attitude of ALOS-1 PALSAR LED file bytes (the first attitude data record, type 40).

    The layout of JAXA's PALSAR Level 1.0 format (Product Format Description for ALOS Data Processing Subsystem,
    PALSAR Level 1.0, Table 3.3-7 Attitude Data Record): bytes 13-16 the number of points (I4), then 120 bytes a point
    from byte 17: the day of the year (I4), the millisecond of the day (I8), the pitch, roll and yaw quality flags
    (3 x I4, 0 good, 1 NG), the pitch, roll and yaw (3 x E14.6, degrees), then their rate flags and rates. The points
    whose yaw flag is good are kept (raw_doppler and raw_margin use the yaw only). The day of the year is taken in the
    year of the orbit's date (the platform position data record, type 30), or in the year before or after when that is
    nearer to it.

    Parameters
    ----------
    led : bytes
        The LED file content.

    Returns
    -------
    dict
        ``'seconds'`` the times of the points in seconds of the orbit's day (ascending), ``'pitch'``, ``'roll'`` and
        ``'yaw'`` in degrees: numpy arrays.

    Raises
    ------
    ValueError
        No type-40 record, no point, a point that cannot be read, or no point with a good yaw flag.

    Examples
    --------
    >>> led_attitude(open('AL1_118_6360_343_20070205.led', 'rb').read())['yaw']
    """
    from datetime import date, timedelta

    record = _record(led, 40)
    if record is None:
        raise ValueError('LED has no attitude data record (type 40)')
    text = _field(record, 13, 16)
    count = int(text) if text.isdigit() else 0
    if count < 1:
        raise ValueError(f'LED attitude data record holds no point (count {text!r})')
    if 16 + 120 * count > len(record):
        raise ValueError(f'LED attitude data record too short for {count} points')
    year, month, day = (int(v) for v in led_orbit(led)['date'].split('-'))
    origin = date(year, month, day)
    seconds, angles = [], []
    for k in range(count):
        point = record[16 + 120 * k:16 + 120 * (k + 1)]
        try:
            day_of_year, ms = int(_field(point, 1, 4)), int(_field(point, 5, 12))
            flags = [int(_field(point, a, a + 3)) for a in (13, 17, 21)]
            values = [float(_field(point, a, a + 13)) for a in (25, 39, 53)]
        except ValueError:
            raise ValueError(f'LED attitude point {k + 1} of {count} cannot be read') from None
        if flags[2] != 0:
            continue
        days = min(((date(y, 1, 1) + timedelta(days=day_of_year - 1) - origin).days for y in (year - 1, year, year + 1)),
                   key=abs)
        seconds.append(86400.0 * days + ms / 1000.0)
        angles.append(values)
    if not seconds:
        raise ValueError(f'LED attitude data record holds no point with a good yaw flag ({count} points)')
    order = np.argsort(seconds, kind='stable')
    seconds, angles = np.asarray(seconds)[order], np.asarray(angles).reshape(-1, 3)[order]
    return {'seconds': seconds, 'pitch': angles[:, 0], 'roll': angles[:, 1], 'yaw': angles[:, 2]}


def orbit_state(orbit, t):
    """
    Interpolate the orbit state at times of its day: 8-point Lagrange interpolation of the six components.

    The 8 consecutive vectors start at ``clip(searchsorted(T, t) - 4, 0, n - 8)`` with
    ``T = seconds + step * arange(n)``. No extrapolation.

    Parameters
    ----------
    orbit : dict
        The orbit as ``led_orbit`` returns it.
    t : float or array_like
        Seconds of the orbit's day.

    Returns
    -------
    tuple of numpy.ndarray
        ``(position, velocity)``, shape (3,) for a scalar ``t``, (N, 3) for an array.

    Raises
    ------
    ValueError
        A time outside the orbit's first and last points.

    Examples
    --------
    >>> orbit_state(orbit, 36000.0)
    """
    vectors = np.asarray(orbit['vectors'], dtype=float)
    n = len(vectors)
    first, step = float(orbit['seconds']), float(orbit['step'])
    times = first + step * np.arange(n)
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if np.any(~(t >= times[0])) or np.any(~(t <= times[-1])):
        raise ValueError(f'orbit time outside {times[0]:.3f}..{times[-1]:.3f} s')
    start = np.clip(np.searchsorted(times, t) - 4, 0, n - 8)
    # the Lagrange weights on the unit node spacing: u is the time in steps from the window's first node
    u = (t - times[start]) / step
    nodes = np.arange(8, dtype=float)
    diff = u[:, None] - nodes[None, :]
    weights = np.empty((len(t), 8))
    for j in range(8):
        others = np.delete(np.arange(8), j)
        weights[:, j] = np.prod(diff[:, others], axis=1) / np.prod(j - others)
    window = vectors[start[:, None] + np.arange(8)[None, :]]
    state = np.einsum('kj,kjc->kc', weights, window)
    if scalar:
        return state[0, :3], state[0, 3:]
    return state[:, :3], state[:, 3:]


def _ecef(lon, lat, height):
    """
    Earth-fixed XYZ of geodetic coordinates on the WGS84 ellipsoid.

    Parameters
    ----------
    lon, lat : array_like
        Geodetic degrees.
    height : array_like
        Metres above the ellipsoid.

    Returns
    -------
    numpy.ndarray
        X, Y, Z in metres, shape (..., 3).

    Examples
    --------
    >>> _ecef(0.0, 0.0, 0.0)
    array([6378137.,       0.,       0.])
    """
    lon, lat = np.radians(lon), np.radians(lat)
    sin_lat = np.sin(lat)
    normal = _A / np.sqrt(1.0 - _E2 * sin_lat ** 2)
    return np.stack([(normal + height) * np.cos(lat) * np.cos(lon),
                     (normal + height) * np.cos(lat) * np.sin(lon),
                     (normal * (1.0 - _E2) + height) * sin_lat], axis=-1)


def geo2rdr(orbit, lon, lat, height, t0, doppler=None):
    """
    Imaging time and slant range of ground points: zero Doppler, or the image's Doppler centroid.

    Newton on ``f(t) = (P - S(t)) . V(t) - wavelength * fd(R) * R / 2`` with ``R = |P - S(t)|``: ``t <- t + f / (V . V)``
    from ``t0``, at most 50 iterations, until every step is below 1e-9 s. The point is imaged when its Doppler
    ``2 (P - S) . V / (wavelength R)`` equals the centroid ``fd(R) = constant + slope * R`` of its slant range (the LED
    Doppler terms as GMTSAR's ALOS preprocessing reads them: ``fd1 + fdd1 * R``). Zero Doppler (``doppler=None``) is
    ``f(t) = (P - S(t)) . V(t)``.

    Parameters
    ----------
    orbit : dict
        The orbit as ``led_orbit`` returns it.
    lon, lat : array_like
        Geodetic degrees.
    height : array_like
        Metres above the WGS84 ellipsoid.
    t0 : float or array_like
        The start times, seconds of the orbit's day.
    doppler : tuple of float or None, optional
        ``(constant, slope, wavelength)``: the Doppler centroid ``constant + slope * R`` in Hz at the slant range R
        (m), and the radar wavelength in metres. None (default): zero Doppler.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, slant_range)``: seconds of the orbit's day and metres, in the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        Newton does not converge in 50 iterations, or a time leaves the orbit.

    Examples
    --------
    >>> geo2rdr(orbit, 138.7, 35.4, 0.0, 36000.0)
    >>> geo2rdr(orbit, -72.15, -40.5, 0.0, 36000.0, (78.837, -1.009178e-4, 0.2360571))
    """
    lon, lat, height, t0 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lon, lat, height, t0)))
    shape = lon.shape
    target = _ecef(lon.ravel(), lat.ravel(), height.ravel())
    t = t0.ravel().copy()
    if doppler is not None:
        constant, slope, wavelength = (float(v) for v in doppler)
    for _ in range(50):
        position, velocity = orbit_state(orbit, t)
        f = np.einsum('kc,kc->k', target - position, velocity)
        if doppler is not None:
            slant_range = np.linalg.norm(target - position, axis=1)
            f = f - 0.5 * wavelength * (constant + slope * slant_range) * slant_range
        dt = f / np.einsum('kc,kc->k', velocity, velocity)
        t = t + dt
        if np.all(np.abs(dt) < 1e-9):
            break
    else:
        raise ValueError('geo2rdr did not converge in 50 iterations')
    position, _ = orbit_state(orbit, t)
    slant_range = np.linalg.norm(target - position, axis=1)
    return t.reshape(shape), slant_range.reshape(shape)


def _day_number(date):
    """
    Days since 1970-01-01 of a date.

    Parameters
    ----------
    date : str
        'YYYY-MM-DD'.

    Returns
    -------
    int
        The day number.

    Examples
    --------
    >>> _day_number('1970-01-02')
    1
    """
    return int(np.datetime64(date, 'D').astype(np.int64))


def _outline(bbox):
    """
    The outline of a bbox: each side sampled at 33 points, 128 points closed (the last point of each side is the
    first of the next).

    Parameters
    ----------
    bbox : tuple of float
        ``(west, south, east, north)`` in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lon, lat)``: 128 points each, counter-clockwise from the south-west corner.

    Examples
    --------
    >>> _outline((138.6, 35.3, 138.8, 35.5))
    """
    west, south, east, north = bbox
    side = np.linspace(0.0, 1.0, 33)[:-1]
    lon = np.concatenate([west + (east - west) * side, np.full(32, east),
                          east + (west - east) * side, np.full(32, west)])
    lat = np.concatenate([np.full(32, south), south + (north - south) * side,
                          np.full(32, north), north + (south - north) * side])
    return lon, lat


def bbox_window(orbit, image, sampling_rate, bbox, heights=(-500.0, 9000.0), doppler=None):
    """
    The line window of a subswath image that a bbox covers on any height between the given heights.

    The bbox outline (each side sampled at 33 points, 128 points closed) is mapped by ``geo2rdr`` (with the image's
    Doppler centroid ``doppler``) on 20 height layers into (range bin, line) of the image; each layer's polygon is
    intersected with the image; the window spans the union of the intersections.

    Parameters
    ----------
    orbit : dict
        The orbit as ``led_orbit`` returns it.
    image : dict
        The subswath: ``'lines'``, ``'pixels'``, ``'time'`` (seconds of day of line 1), ``'prf'`` (Hz),
        ``'near_range'`` (m) and ``'date'`` ('YYYY-MM-DD').
    sampling_rate : float
        Range sampling rate in Hz.
    bbox : tuple of float
        ``(west, south, east, north)`` in degrees.
    heights : tuple of float, optional
        The lowest and highest heights above the WGS84 ellipsoid, metres.
    doppler : tuple of float or None, optional
        ``(constant, slope, wavelength)`` of the image's Doppler centroid, as ``geo2rdr`` takes it. None (default):
        zero Doppler (ALOS-2 L1.1).

    Returns
    -------
    tuple of int or None
        ``(start, end)`` 0-based lines, end exclusive; None when the bbox reaches the image at no height.

    Examples
    --------
    >>> bbox_window(orbit, image, 17.465266e6, (138.6, 35.3, 138.8, 35.5))
    >>> bbox_window(orbit, image, 16e6, (-72.27, -40.595, -72.03, -40.405), doppler=(78.837, -1.009178e-4, 0.2360571))
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    lines, pixels, prf = int(image['lines']), int(image['pixels']), float(image['prf'])
    days = _day_number(image['date']) - _day_number(orbit['date'])
    time0 = float(image['time']) + 86400.0 * days
    middle = time0 + 0.5 * (lines - 1) / prf
    lon, lat = _outline(bbox)
    spacing = _C / (2.0 * float(sampling_rate))
    frame = box(0, 0, pixels, lines)
    parts = []
    for height in np.linspace(heights[0], heights[1], 20):
        t, slant_range = geo2rdr(orbit, lon, lat, height, middle, doppler)
        outline = Polygon(np.column_stack([(slant_range - float(image['near_range'])) / spacing, (t - time0) * prf]))
        if not outline.is_valid:
            outline = outline.buffer(0)
        part = outline.intersection(frame)
        if not part.is_empty:
            parts.append(part)
    if not parts:
        return None
    _, miny, _, maxy = unary_union(parts).bounds
    return max(0, int(math.floor(miny))), min(lines, int(math.ceil(maxy)))


def _frame_yaw(orbit, attitude, image, line_rate):
    """
    The frame's middle time ``image['time'] + lines / (2 line_rate)`` in seconds of the orbit's day, and the yaw of the
    attitude interpolated linearly at it (degrees). Raises ValueError when the time lies outside the attitude points.

    Examples
    --------
    >>> _frame_yaw(orbit, led_attitude(led), image, 2123.142)
    """
    days = _day_number(image['date']) - _day_number(orbit['date'])
    t = float(image['time']) + 86400.0 * days + 0.5 * int(image['lines']) / float(line_rate)
    times = np.asarray(attitude['seconds'], dtype=float)
    if not times[0] <= t <= times[-1]:
        raise ValueError(f'LED attitude covers {times[0]:.3f}..{times[-1]:.3f} s, not the frame middle {t:.3f} s')
    return t, float(np.interp(t, times, np.asarray(attitude['yaw'], dtype=float)))


def raw_doppler(orbit, attitude, image, sampling_rate, wavelength, line_rate):
    """
    The beam-centre Doppler centroid of an ALOS-1 PALSAR raw (L1.0) frame from the platform's yaw (the raw leader files
    hold no Doppler centroid): the Doppler of the ground point that the beam centre sees at a slant range, at the
    frame's first and last echo samples, at the frame's middle time.

    The reference frame of the attitude angles is the orbital frame of the inertial velocity: down
    ``z = -S / |S|``, right ``y = z x Vi / |z x Vi|``, ahead ``x = y x z``, with ``Vi = V + Omega x S`` the inertial
    velocity of the Earth-fixed orbit state S, V (Omega the Earth's rotation). The antenna's along-track axis is the
    body x axis after the yaw about z, ``a = cos(yaw) x + sin(yaw) y``. The pitch and roll are not used: the roll about
    the body x axis does not move it, and the sign of the pitch term is not established (fitted on raw / L1.1 pairs
    2026-11-23: -0.31 +- 0.48 of the term, which holds 0; the beam-centre offsets of the margins below were measured
    without it). The beam centre lies in the plane through S normal to a: the ground point P (WGS84 ellipsoid, height
    0) at the slant range R, right of the track in that plane, has the Doppler ``fd = 2 V . (P - S) / (wavelength R)``
    (geo2rdr's convention). The yaw is that of the attitude interpolated linearly at the frame's middle time,
    ``image['time'] + lines / (2 line_rate)``.

    Parameters
    ----------
    orbit : dict
        The orbit as ``led_orbit`` returns it.
    attitude : dict
        The attitude as ``led_attitude`` returns it (seconds of the orbit's day).
    image : dict
        The frame: ``'lines'``, ``'pixels'`` (echo samples per line), ``'time'`` (seconds of day of line 1),
        ``'near_range'`` (slant range of the first sample, m) and ``'date'`` ('YYYY-MM-DD').
    sampling_rate : float
        Range sampling rate in Hz.
    wavelength : float
        The radar wavelength in metres.
    line_rate : float
        The frame's lines per second (the header PRF; half of it for a PLR frame).

    Returns
    -------
    tuple of float
        ``(constant, slope, wavelength)``: the Doppler centroid ``constant + slope * R`` in Hz at the slant range R
        (m) through the two Dopplers, as ``geo2rdr`` and ``raw_window`` take it.

    Raises
    ------
    ValueError
        The frame's middle time outside the attitude points, or a slant range that does not reach the ellipsoid.

    Examples
    --------
    >>> raw_doppler(orbit, led_attitude(led), image, 32e6, 0.2360571, 2123.142)
    """
    t, yaw = _frame_yaw(orbit, attitude, image, line_rate)
    yaw = math.radians(yaw)
    position, velocity = orbit_state(orbit, t)
    inertial = velocity + np.cross(np.array([0.0, 0.0, _OMEGA]), position)
    down = -position / np.linalg.norm(position)
    right = np.cross(down, inertial)
    right /= np.linalg.norm(right)
    ahead = np.cross(right, down)
    axis = math.cos(yaw) * ahead + math.sin(yaw) * right
    # the plane normal to the axis: its direction nearest to the nadir, and its direction to the right
    lower = down - float(down @ axis) * axis
    lower /= np.linalg.norm(lower)
    side = np.cross(lower, axis)
    near = float(image['near_range'])
    ranges = np.array([near, near + int(image['pixels']) * _C / (2.0 * float(sampling_rate))])
    # the look angle in the plane, by bisection: below the ellipsoid at 0, above it at 90 degrees
    scale = np.array([1.0 / _A ** 2, 1.0 / _A ** 2, 1.0 / (_A * (1.0 - _F)) ** 2])

    def outside(angle):
        direction = np.cos(angle)[:, None] * lower + np.sin(angle)[:, None] * side
        point = position + ranges[:, None] * direction
        return (point * point * scale).sum(axis=1) > 1.0

    low, high = np.zeros(2), np.full(2, 0.5 * math.pi)
    if outside(low).any() or not outside(high).all():
        raise ValueError(f'slant ranges {ranges[0]:.0f}..{ranges[1]:.0f} m do not reach the ellipsoid')
    for _ in range(60):
        middle = 0.5 * (low + high)
        above = outside(middle)
        low, high = np.where(above, low, middle), np.where(above, middle, high)
    angle = 0.5 * (low + high)
    direction = np.cos(angle)[:, None] * lower + np.sin(angle)[:, None] * side
    doppler = 2.0 * (direction @ velocity) / float(wavelength)
    slope = float((doppler[1] - doppler[0]) / (ranges[1] - ranges[0]))
    return float(doppler[0] - slope * ranges[0]), slope, float(wavelength)


def raw_margin(led):
    """
    The margin of an ALOS-1 PALSAR raw (L1.0) window (raw_window) by the frame's steering state: ALOS1_DOPPLER_MARGIN
    for a yaw-steered frame, ALOS1_DOPPLER_MARGIN_UNSTEERED for a frame without yaw steering.

    The state is JAXA's yaw steering mode flag of the raw leader file: bytes 1831-1834 of the data set summary record
    (type 10, I4), '0' yaw steering and '1' none (ALOS PALSAR Level 1.0 Product Format Description Rev. J, Table 3.3-5
    (12/13), field 133, p. 3-37).

    Parameters
    ----------
    led : bytes
        The raw leader file content.

    Returns
    -------
    tuple of float
        ``(before, after)`` seconds, as ``raw_window`` takes its ``margin``.

    Raises
    ------
    ValueError
        No type-10 record, or a yaw steering mode flag that is neither 0 nor 1.

    Examples
    --------
    >>> raw_margin(open('AL1_118_6360_343_20070205.led', 'rb').read())
    """
    record = _record(led, 10)
    if record is None:
        raise ValueError('LED has no data set summary record (type 10)')
    flag = _field(record, 1831, 1834)
    if flag == '0':
        return ALOS1_DOPPLER_MARGIN
    if flag == '1':
        return ALOS1_DOPPLER_MARGIN_UNSTEERED
    raise ValueError(f"LED yaw steering mode flag (data set summary bytes 1831-1834) is {flag!r}, not 0 or 1")


def raw_window(orbit, image, sampling_rate, bbox, wavelength, doppler=None, heights=(-500.0, 9000.0),
               antenna_length=8.9, margin=None):
    """
    The time span of an ALOS-1 PALSAR raw (L1.0) frame that holds every echo of a region: the span of the region's
    beam-centre times, widened on each side by half the synthetic aperture and by a margin (the difference of the
    Doppler centroid given, from the yaw (raw_doppler), to the image's own: raw_margin by the frame's steering state).

    The beam-centre times: the region's outline (each side sampled at 33 points, 128 points closed) mapped by
    ``geo2rdr`` (with the Doppler centroid ``doppler``; zero Doppler for None) on 20 height layers into (range bin,
    time); each layer's polygon is intersected with the frame's swath (range bins 0..pixels, from the slant range of
    its first sample at the range sampling rate), and the span runs over the union of the intersections. The frame's
    lines do not enter: the download selects the lines by the time tag of each line
    (utils_alos2_ceos.select_lines), so a PRF change inside the frame moves no line.

    Half the synthetic aperture is ``wavelength * R_far / (2 * antenna_length * v_ground)`` seconds: the half of the
    beam's along-track width wavelength / antenna_length (radians) at the slant range R_far of the frame's last echo
    sample, crossed at the ground speed v_ground = |V| * r / |S| of the sub-satellite point (S, V the orbit state at
    the middle of the region's beam-centre span, r the ellipsoid's geocentric radius under S).

    Parameters
    ----------
    orbit : dict
        The orbit as ``led_orbit`` returns it.
    image : dict
        The frame: ``'lines'``, ``'pixels'`` (echo samples per line), ``'time'`` (seconds of day of line 1),
        ``'prf'`` (Hz), ``'near_range'`` (slant range of the first sample, m) and ``'date'`` ('YYYY-MM-DD').
        ``'lines'``, ``'time'`` and ``'prf'`` only give the start of geo2rdr's iterations.
    sampling_rate : float
        Range sampling rate in Hz.
    bbox : tuple of float
        ``(west, south, east, north)`` of the region in degrees (utils_grid.transform_region).
    wavelength : float
        The radar wavelength in metres.
    doppler : tuple of float or None, optional
        ``(constant, slope, wavelength)`` of the Doppler centroid, as ``geo2rdr`` takes it (``raw_doppler``). None
        (default): zero Doppler.
    heights : tuple of float, optional
        The lowest and highest heights above the WGS84 ellipsoid, metres.
    antenna_length : float, optional
        The antenna length along the track in metres. Default 8.9: the 8.9 x 3.1 m ALOS-1 PALSAR antenna (GMTSAR's
        ALOS preprocessing, ALOS_pre_process/roi_utils.c: ANTENNA_LENGTH = 8.9).
    margin : tuple of float or None, optional
        ``(before, after)`` seconds added before and after the span (``raw_margin``). None (default):
        ALOS1_DOPPLER_MARGIN (a yaw-steered frame).

    Returns
    -------
    tuple of float or None
        ``(start, end)``: seconds of the day of ``image['date']`` (as the lines' time tags count them); None when the
        region reaches the frame's swath at no height.

    Examples
    --------
    >>> raw_window(orbit, image, 32e6, (-72.3, -40.6, -72.0, -40.4), 0.2360571)
    >>> raw_window(orbit, image, 32e6, (-72.3, -40.6, -72.0, -40.4), 0.2360571, raw_doppler(orbit, attitude, image, 32e6,
    ...            0.2360571, 2123.142), margin=raw_margin(led))
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    lines, prf = int(image['lines']), float(image['prf'])
    days = _day_number(image['date']) - _day_number(orbit['date'])
    # the time of line 1 in seconds of the orbit's day: the origin of the polygons' time axis
    time0 = float(image['time']) + 86400.0 * days
    guess = time0 + 0.5 * (lines - 1) / prf
    lon, lat = _outline(bbox)
    spacing = _C / (2.0 * float(sampling_rate))
    # the swath in range, unbounded in time (a day either way)
    swath = box(0, -86400.0, int(image['pixels']), 86400.0)
    parts = []
    for height in np.linspace(heights[0], heights[1], 20):
        t, slant_range = geo2rdr(orbit, lon, lat, height, guess, doppler)
        outline = Polygon(np.column_stack([(slant_range - float(image['near_range'])) / spacing, t - time0]))
        if not outline.is_valid:
            outline = outline.buffer(0)
        part = outline.intersection(swath)
        if not part.is_empty:
            parts.append(part)
    if not parts:
        return None
    _, first, _, last = unary_union(parts).bounds
    position, velocity = orbit_state(orbit, time0 + 0.5 * (first + last))
    distance = float(np.linalg.norm(position))
    # the ellipsoid's geocentric radius under the satellite: a b / sqrt(b^2 cos^2 + a^2 sin^2) of its geocentric latitude
    sine = float(position[2]) / distance
    polar = _A * (1.0 - _F)
    radius = _A * polar / math.sqrt(polar ** 2 * (1.0 - sine ** 2) + _A ** 2 * sine ** 2)
    ground_speed = float(np.linalg.norm(velocity)) * radius / distance
    far_range = float(image['near_range']) + int(image['pixels']) * _C / (2.0 * float(sampling_rate))
    half = float(wavelength) * far_range / (2.0 * float(antenna_length) * ground_speed)
    before, after = ALOS1_DOPPLER_MARGIN if margin is None else (float(v) for v in margin)
    return (float(image['time']) + first - half - before, float(image['time']) + last + half + after)
