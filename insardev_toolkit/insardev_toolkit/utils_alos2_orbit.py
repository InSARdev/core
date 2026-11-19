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
zero-Doppler time of the point. A bbox, mapped on height layers from -500 to 9000 m into (range bin, line) of a
subswath, gives the lines to read, the way the NISAR downloader uses its geolocation grid. The functions here are
pure: no network and no file access.
"""
import math

import numpy as np

# the WGS84 ellipsoid and the speed of light
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_C = 299792458.0


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


def geo2rdr(orbit, lon, lat, height, t0):
    """
    Zero-Doppler time and slant range of ground points.

    Newton on ``f(t) = (P - S(t)) . V(t)``: ``t <- t + f / (V . V)`` from ``t0``, at most 50 iterations, until every
    step is below 1e-9 s.

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
    """
    lon, lat, height, t0 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lon, lat, height, t0)))
    shape = lon.shape
    target = _ecef(lon.ravel(), lat.ravel(), height.ravel())
    t = t0.ravel().copy()
    for _ in range(50):
        position, velocity = orbit_state(orbit, t)
        dt = np.einsum('kc,kc->k', target - position, velocity) / np.einsum('kc,kc->k', velocity, velocity)
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


def bbox_window(orbit, image, sampling_rate, bbox, heights=(-500.0, 9000.0)):
    """
    The line window of a subswath image that a bbox covers on any height between the given heights.

    The bbox outline (each side sampled at 33 points, 128 points closed) is mapped by ``geo2rdr`` on 20 height layers
    into (range bin, line) of the image; each layer's polygon is intersected with the image; the window spans the
    union of the intersections.

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

    Returns
    -------
    tuple of int or None
        ``(start, end)`` 0-based lines, end exclusive; None when the bbox reaches the image at no height.

    Examples
    --------
    >>> bbox_window(orbit, image, 17.465266e6, (138.6, 35.3, 138.8, 35.5))
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    west, south, east, north = bbox
    lines, pixels, prf = int(image['lines']), int(image['pixels']), float(image['prf'])
    days = _day_number(image['date']) - _day_number(orbit['date'])
    time0 = float(image['time']) + 86400.0 * days
    middle = time0 + 0.5 * (lines - 1) / prf
    side = np.linspace(0.0, 1.0, 33)[:-1]
    lon = np.concatenate([west + (east - west) * side, np.full(32, east),
                          east + (west - east) * side, np.full(32, west)])
    lat = np.concatenate([np.full(32, south), south + (north - south) * side,
                          np.full(32, north), north + (south - north) * side])
    spacing = _C / (2.0 * float(sampling_rate))
    frame = box(0, 0, pixels, lines)
    parts = []
    for height in np.linspace(heights[0], heights[1], 20):
        t, slant_range = geo2rdr(orbit, lon, lat, height, middle)
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
