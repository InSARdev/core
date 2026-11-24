# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
"""
ALOS-2 PALSAR-2 ScanSAR L1.1 subswath crops as insardev_toolkit `ASF.download` stores them: per date and subswath an IMG
file per polarization (`AL2_<path>_<frame>_<Fn>_<yyyymmdd>_<pol>.img`) and one leader file (`AL2_<path>_<frame>_<Fn>_<yyyymmdd>.led`).

IMG: a 720-byte file descriptor, then one record per line: a 544-byte prefix and the pixels as big-endian float32 I, Q pairs
(record length 544 + 8 x pixels). The prefix fields read here (1-based bytes, big-endian int32 unless noted): 13 image line
number, 25 pixels, 37 year, 41 day of year, 45 milliseconds of the day, 57 PRF (mHz), 85-92 sensor acquisition microseconds of
the day (int64), 117 slant range of the first pixel (integer m), 121 data record window position (ns of two-way delay),
193/197/201 latitude and 205/209/213 longitude of the first, mid and last pixel (1e-6 degree, on the WGS84 ellipsoid at height 0).
The lines are read through a memmap of the records: a box is one byte range per line, nothing to decode and no cache to tune.
The prefix length is the file descriptor's field "prefix data per record" (bytes 277-280, ASCII): 544 in ALOS-2 files, 412 in
ALOS-1 PALSAR L1.1 files (the first 412 bytes of the ALOS-2 prefix, the same fields; bytes 85-92 and 121-124 are spare and 0
there). So alos2_image, alos2_prefix, alos2_led, alos2_slc and alos2_footprint read ALOS-1 files too (utils_alos1).

LED: CEOS records; the data set summary (type 10) holds the radar constants and the ellipsoid, the platform position data
(type 30) the orbit: Earth-fixed state vectors 60 s apart.

Line timing (alos2_timing): the lines are start + k / PRF, start the microsecond field of line 0 and PRF the mHz field (GMTSAR's
line spacing too), and the near range is c / 2 x the window position. On the crops of raw/alos2_calbuco JAXA's per-line latitude
and longitude on the LED orbit follow exactly this line: a constant +47 us, a slope of -0.08..-0.10 us per 1000 lines, a scatter
of +-9.5 us (the 1e-6 degree rounding of the latitude is 16.9 us of line time); their first-pixel ranges are within 0.034 m of
the near range. The microsecond tags of later lines drift +1 us per 1000 lines against the line (19 us over 17408 lines, 2.5 ppm),
the millisecond field is their truncation, and the integer near-range field is off by up to 0.42 m.

Orbit: the LED vectors are densified to 10 s by the toolkit's 8-point Lagrange interpolation (alos2_orbit), the step on which
the package's Hermite interpolation is accurate.
"""
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import constants

from .utils_s1 import _sc_clock_whole_second

SOL = constants.speed_of_light
# the step of the orbit given to the package (s): see alos2_orbit
ALOS2_ORBIT_STEP = 10.0
# the timing reads every this many lines (and the last line): field checks and the check of the line positions
ALOS2_TIMING_LINES = 64
# the IMG file name: path, frame, subswath, date, polarization
ALOS2_NAME = re.compile(r'^AL2_(\d+)_(\d+)_(F[1-5])_(\d{8})_([HV][HV])\.img$')
# the fields of the data set summary record (type 10): name -> (first byte, last byte, scale), 1-based bytes of the record
ALOS2_DSS_FIELDS = {
    'semi_major_axis': (181, 196, 1e3),        # km
    'semi_minor_axis': (197, 212, 1e3),        # km
    'clock_angle': (477, 484, 1.0),            # degrees, +90 right looking
    'incidence': (485, 492, 1.0),              # degrees at the scene centre
    'wavelength': (501, 516, 1.0),             # m
    'chirp_linear': (551, 566, 1.0),           # Hz/s
    'rng_samp_rate': (711, 726, 1e6),          # MHz
    'pulse_dur': (743, 758, 1e-6),             # us
}


def alos2_image(img_path: str) -> dict:
    """
    Size, timing fields and near range of an ALOS-2 (or ALOS-1 L1.1) IMG file, checked against the file size.

    The line prefix length is the descriptor's field "prefix data per record" (bytes 277-280, ASCII): 544 for ALOS-2, 412 for
    ALOS-1 PALSAR L1.1.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.

    Returns
    -------
    dict
        `insardev_toolkit.utils_alos2_ceos.image_info` of the first 720 + prefix bytes: 'lines', 'record_length', 'pixels',
        'time' (seconds of day of line 1, the 1 ms field), 'prf' (Hz), 'near_range' (integer m), 'date' ('YYYY-MM-DD'); and
        'prefix' (the line prefix length, bytes).

    Raises
    ------
    ValueError
        The head is not that of a consistent IMG file (the prefix length field is not a number among them), or the file size is
        not 720 + lines x record_length.

    Examples
    --------
    >>> alos2_image('129_4450_F4/AL2_129_4450_F4_20150404_HH.img')['lines']
    """
    from insardev_toolkit.utils_alos2_ceos import image_info, DESCRIPTOR

    with open(img_path, 'rb') as f:
        head = f.read(DESCRIPTOR)
        text = head[276:280].decode('ascii', errors='replace').strip()
        if len(head) < DESCRIPTOR or not text.isdigit():
            raise ValueError(f'{img_path}: the prefix length (bytes 277-280 of the descriptor) is {text!r}, not a number; '
                             f'download it again')
        prefix_length = int(text)
        head += f.read(prefix_length)
    info = image_info(head, prefix_length)
    info['prefix'] = prefix_length
    size = os.path.getsize(img_path)
    lines, record_length = info['lines'], info['record_length']
    expected = DESCRIPTOR + lines * record_length
    if size != expected:
        raise ValueError(f'{img_path}: {size} bytes, expected {expected} for {lines} lines of {record_length} bytes; '
                         f'download it again')
    return info


def alos2_prefix(img_path: str, lines) -> dict:
    """
    Prefix fields of lines of an ALOS-2 (or ALOS-1 L1.1) IMG file.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.
    lines : array-like of int
        0-based line indices.

    Returns
    -------
    dict
        'line' (image line number), 'pixels', 'year', 'doy' (1-based day of year), 'ms' (milliseconds of the day), 'us'
        (sensor acquisition microseconds of the day, bytes 85-92, int64), 'chirp_linear' (the chirp linear coefficient, bytes
        77-80, the field's 4 bytes as a big-endian int32), 'near_range' (integer m), 'window' (data record window position, ns
        of two-way delay) as int64 arrays, 'prf' (Hz, the mHz field) a float64 array, and 'lat', 'lon' float64
        arrays (len, 3) of the first, mid and last pixel in degrees (height 0). In ALOS-1 files 'us' and 'window' are spare
        bytes (0).

    Examples
    --------
    >>> alos2_prefix('129_4450_F4/AL2_129_4450_F4_20150404_HH.img', [0, 64, 128])['prf']
    """
    from insardev_toolkit.utils_alos2_ceos import DESCRIPTOR

    info = alos2_image(img_path)
    lines = np.asarray(lines, dtype=np.int64).ravel()
    records = np.memmap(img_path, dtype=np.uint8, mode='r', offset=DESCRIPTOR, shape=(info['lines'], info['record_length']))
    prefix = np.array(records[lines, :info['prefix']])
    del records
    # the first 216 bytes as big-endian int32 words: the field at 1-based byte b is the word (b - 1) // 4
    words = np.ascontiguousarray(prefix[:, :216]).view('>i4').astype(np.int64)
    return {'line': words[:, 3],
            'pixels': words[:, 6],
            'year': words[:, 9],
            'doy': words[:, 10],
            'ms': words[:, 11],
            'us': np.ascontiguousarray(prefix[:, 84:92]).view('>i8').ravel().astype(np.int64),
            'prf': words[:, 14].astype(np.float64) / 1000.0,
            'chirp_linear': words[:, 19],
            'near_range': words[:, 29],
            'window': words[:, 30],
            'lat': words[:, 48:51].astype(np.float64) * 1e-6,
            'lon': words[:, 51:54].astype(np.float64) * 1e-6}


def _dss_fields(led: bytes, led_path: str, fields: dict) -> dict:
    """
    Numeric ASCII fields of the data set summary record (type 10) of a leader file's bytes.

    Parameters
    ----------
    led : bytes
        The leader file's bytes.
    led_path : str
        The leader file named in the error messages.
    fields : dict
        name -> (first byte, last byte, scale), 1-based bytes of the record, as ALOS2_DSS_FIELDS.

    Returns
    -------
    dict
        name -> the field's number x scale (float).

    Raises
    ------
    ValueError
        No data set summary record, or a field that is not a finite number.

    Examples
    --------
    >>> _dss_fields(open('AL2_129_4450_F4_20150404.led', 'rb').read(), 'AL2_129_4450_F4_20150404.led', ALOS2_DSS_FIELDS)
    """
    from insardev_toolkit.utils_alos2_orbit import led_records

    dss = [led[offset:offset + length] for offset, kind, length in led_records(led) if kind == 10]
    if not dss:
        raise ValueError(f'{led_path}: no data set summary record (type 10)')
    dss = dss[0]
    values = {}
    for name, (first, last, scale) in fields.items():
        text = dss[first - 1:last].decode('ascii', errors='replace').strip()
        try:
            values[name] = float(text) * scale
        except ValueError:
            raise ValueError(f'{led_path}: {name} (bytes {first}-{last} of the data set summary) is {text!r}, not a number') from None
        if not np.isfinite(values[name]):
            raise ValueError(f'{led_path}: {name} (bytes {first}-{last} of the data set summary) is {text!r}, not a number')
    return values


def alos2_led(led_path: str) -> dict:
    """
    Radar constants, ellipsoid and orbit of an ALOS-2 (or ALOS-1 L1.1) leader (LED) file.

    Parameters
    ----------
    led_path : str
        Path to the LED file.

    Returns
    -------
    dict
        'wavelength' (m), 'rng_samp_rate' (Hz), 'pulse_dur' (s), 'chirp_linear' (Hz/s, the linear range pulse coefficient of the
        data set summary, positive; the PRM's chirp_slope is the line prefix's coefficient, see alos2_prm), 'clock_angle'
        (degrees), 'lookdir' ('L' when the clock angle is negative, else 'R'), 'incidence' (degrees at
        the scene centre), 'equatorial_radius' and 'polar_radius' (m) of the data set summary record, and 'orbit' (the
        `insardev_toolkit.utils_alos2_orbit.led_orbit` dict).

    Raises
    ------
    ValueError
        No data set summary or platform position record, or a field that is not a number.

    Examples
    --------
    >>> alos2_led('129_4450_F4/AL2_129_4450_F4_20150404.led')['wavelength']
    """
    from insardev_toolkit.utils_alos2_orbit import led_orbit

    with open(led_path, 'rb') as f:
        led = f.read()
    values = _dss_fields(led, led_path, ALOS2_DSS_FIELDS)
    try:
        orbit = led_orbit(led)
    except ValueError as e:
        raise ValueError(f'{led_path}: {e}') from None
    return {'wavelength': values['wavelength'],
            'rng_samp_rate': values['rng_samp_rate'],
            'pulse_dur': values['pulse_dur'],
            'chirp_linear': values['chirp_linear'],
            'clock_angle': values['clock_angle'],
            'lookdir': 'L' if values['clock_angle'] < 0 else 'R',
            'incidence': values['incidence'],
            'equatorial_radius': values['semi_major_axis'],
            'polar_radius': values['semi_minor_axis'],
            'orbit': orbit}


def _fields_constant(img_path: str, prefix: dict, names) -> None:
    """
    Check that prefix fields keep one value over the sampled lines.

    Parameters
    ----------
    img_path : str
        The IMG file named in the error message.
    prefix : dict
        The fields of the sampled lines (alos2_prefix).
    names : iterable of str
        The fields to check.

    Raises
    ------
    ValueError
        A field takes more than one value.

    Examples
    --------
    >>> _fields_constant('AL2_129_4450_F4_20150404_HH.img', alos2_prefix('AL2_129_4450_F4_20150404_HH.img', [0, 64]), ('prf',))
    """
    for name in names:
        values = np.unique(prefix[name])
        if len(values) > 1:
            raise ValueError(f'{img_path}: the {name} field changes inside the file ({", ".join(str(v) for v in values)})')


def _line_positions(img_path: str, led_path: str, info: dict, orbit: dict, prefix: dict, doppler=None) -> tuple:
    """
    JAXA's per-line positions of the sampled lines mapped on the LED orbit.

    The latitude and longitude of the first, mid and last pixel (height 0) of each sampled line are mapped by
    `insardev_toolkit.utils_alos2_orbit.geo2rdr` to their imaging time and slant range, from the line's millisecond field: at
    zero Doppler, or at the Doppler centroid `doppler`.

    Parameters
    ----------
    img_path : str
        The IMG file named in the error message.
    led_path : str
        The LED file named in the error message.
    info : dict
        alos2_image of the IMG file.
    orbit : dict
        The LED orbit (alos2_led 'orbit').
    prefix : dict
        The fields of the sampled lines (alos2_prefix).
    doppler : tuple of float, optional
        (constant Hz, slope Hz/m, wavelength m) of the centroid constant + slope x R, as geo2rdr takes it; None: zero Doppler.

    Returns
    -------
    tuple
        (line_days, t_ms, t, slant_range): the day of each line from line 1's date (int64; the times count on past midnight),
        the millisecond field in seconds of line 1's day, and the imaging times (s of line 1's day) and slant ranges (m),
        arrays (lines, 3).

    Raises
    ------
    ValueError
        No solution on the orbit: the files do not belong together or one is damaged.

    Examples
    --------
    >>> _line_positions(img, led_path, alos2_image(img), alos2_led(led_path)['orbit'], alos2_prefix(img, [0, 64]))
    """
    from insardev_toolkit.utils_alos2_orbit import geo2rdr

    # the date of each line from its prefix, in days from line 1's date: the times count on past midnight
    pairs, inverse = np.unique(np.stack([prefix['year'], prefix['doy']], axis=1), axis=0, return_inverse=True)
    pair_days = np.array([(np.datetime64(f'{int(y):04d}-01-01') + np.timedelta64(int(d) - 1, 'D') - np.datetime64(info['date'])).astype(np.int64)
                          for y, d in pairs], dtype=np.int64)
    line_days = pair_days[np.ravel(inverse)]
    t_ms = prefix['ms'] / 1e3 + 86400.0 * line_days
    # the positions of the sampled lines on the orbit, in seconds of line 1's day
    day_offset = int((np.datetime64(info['date']) - np.datetime64(orbit['date'])).astype(np.int64))
    try:
        t, slant_range = geo2rdr(orbit, prefix['lon'], prefix['lat'], 0.0, (t_ms + 86400.0 * day_offset)[:, None], doppler)
    except ValueError as e:
        raise ValueError(f"{img_path}: JAXA's line positions do not follow the orbit of {led_path} ({e}): the files do not "
                         f"belong together or one is damaged; download them again") from None
    return line_days, t_ms, t - 86400.0 * day_offset, slant_range


def alos2_timing(img_path: str, led_path: str) -> dict:
    """
    Line timing and near range of an ALOS-2 IMG file from JAXA's fine prefix fields, checked against the per-line positions on
    the LED orbit.

    The lines are start + k / prf (k the 0-based line): start is the prefix field "sensor acquisition microseconds of day"
    (bytes 85-92, int64) of line 0 and prf the mHz PRF field (bytes 57-60), the line spacing of GMTSAR too. The near range is
    c / 2 x the field "data record window position" (bytes 121-124, ns of two-way delay). The millisecond field (a truncation)
    and the integer near-range field are not used for the values. The microsecond tags of later lines drift from this line:
    returned as 'us_drift', not used (measured on raw/alos2_calbuco: +1 us per 1000 lines, 19 us over 17408 lines).

    Only the sampled lines k = 0, 64, 128, ... and the last line are read (a PRF or window change persists, so the samples
    detect it): their field-change checks, the microsecond test below and the check of the positions. The check: the latitude
    and longitude of the first, mid and last pixel (height 0) are mapped by `insardev_toolkit.utils_alos2_orbit.geo2rdr` on the
    LED orbit to their zero-Doppler time and slant range, from the line's millisecond field. Measured on raw/alos2_calbuco:
    +47 us after the line times with a slope of -0.08..-0.10 us per 1000 lines, spread 19 us (the 1e-6 degree rounding of the
    latitude is 16.9 us of line time); first-pixel ranges within 0.034 m of the near range, spread 0.07 m.

    A file whose microsecond field is zero or more than 1 ms from its millisecond field on any sampled line prints a WARNING,
    and its start comes from the positions instead: the median of t_zd - k / prf.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.
    led_path : str
        Path to its LED file.

    Returns
    -------
    dict
        'day' (datetime, midnight UTC of the date of line 1), 'start' (seconds of that day of line 1, float64), 'prf' (Hz),
        'near_range' (m), 'lines', 'pixels', 'us_drift' (s: the largest distance of the microsecond field of the sampled lines
        from the line times; NaN without the field), 'latlon_time' (s: median of t_zd - (start + k / prf)), 'latlon_range' (m: median first-pixel slant
        range minus near_range), 'spread_time' (s) and 'spread_range' (m): max - min of those.

    Raises
    ------
    ValueError
        The PRF, pixel count, near-range or window-position field differs between the sampled lines; or the line positions do not follow
        the orbit (|latlon_time| or spread_time above 1 / prf, |latlon_range| or spread_range above c / (2 rng_samp_rate), or no
        zero-Doppler solution on the orbit): the IMG and the LED do not belong together or one is damaged.

    Examples
    --------
    >>> alos2_timing('129_4450_F4/AL2_129_4450_F4_20150404_HH.img', '129_4450_F4/AL2_129_4450_F4_20150404.led')['start']
    """
    info = alos2_image(img_path)
    led = alos2_led(led_path)
    lines, pixels = info['lines'], info['pixels']
    # the sampled lines: k = 0, 64, 128, ... and the last line
    k = np.unique(np.append(np.arange(0, lines, ALOS2_TIMING_LINES), lines - 1))
    prefix = alos2_prefix(img_path, k)
    _fields_constant(img_path, prefix, ('prf', 'pixels', 'near_range', 'window'))
    day = datetime.strptime(info['date'], '%Y-%m-%d')
    near_range = SOL / 2.0 * float(prefix['window'][0]) * 1e-9
    # the positions of the sampled lines on the orbit at zero Doppler, in seconds of line 1's day
    line_days, t_ms, t_zd, slant_range = _line_positions(img_path, led_path, info, led['orbit'], prefix)
    t_us = prefix['us'] / 1e6 + 86400.0 * line_days

    # the line spacing is the mHz PRF field
    prf = float(prefix['prf'][0])
    if np.all(prefix['us'] != 0) and np.all(np.abs(t_us - t_ms) <= 1e-3):
        # the microsecond tag of line 0; the tags of later lines drift from the line, reported only
        start = float(t_us[0])
        us_drift = float(np.abs(t_us - (start + k / prf)).max())
    else:
        print(f'WARNING: {img_path}: the microsecond time field of the lines is zero or more than 1 ms from the millisecond '
              f'field; the line times are fitted to the line positions on the orbit of {led_path} instead.')
        start = float(np.median(t_zd - k[:, None] / prf))
        us_drift = float('nan')

    model = t_zd - (start + k[:, None] / prf)
    first = slant_range[:, 0]
    latlon_time = float(np.median(model))
    latlon_range = float(np.median(first)) - near_range
    spread_time = float(model.max() - model.min())
    spread_range = float(first.max() - first.min())
    spacing = SOL / (2.0 * led['rng_samp_rate'])
    if not (abs(latlon_time) <= 1.0 / prf and spread_time <= 1.0 / prf
            and abs(latlon_range) <= spacing and spread_range <= spacing):
        raise ValueError(f"{img_path}: JAXA's line positions do not follow the orbit of {led_path} ({1e6 * latlon_time:+.1f} us "
                         f"and spread {1e6 * spread_time:.1f} us in time, {latlon_range:+.3f} m and spread {spread_range:.3f} m "
                         f"in range): the files do not belong together or one is damaged; download them again")
    # the day of the first line's time
    days = int(np.floor(start / 86400.0))
    return {'day': day + timedelta(days=days),
            'start': start - 86400.0 * days,
            'prf': prf,
            'near_range': near_range,
            'lines': lines,
            'pixels': pixels,
            'us_drift': us_drift,
            'latlon_time': latlon_time,
            'latlon_range': latlon_range,
            'spread_time': spread_time,
            'spread_range': spread_range}


def alos2_orbit(led_path: str, img_path: str, t1: float = None, t2: float = None) -> pd.DataFrame:
    """
    Orbit state vectors of an ALOS-2 scene: the LED vectors densified to 10 s.

    The LED holds Earth-fixed state vectors 60 s apart. The package interpolates orbits by Hermite interpolation with
    accelerations from finite differences, accurate on vectors 10 s apart: on vectors 60 s apart it is off by up to 7.5 cm in
    position and 3.8 cm/s in velocity (about 4 m along track), measured on a 10 s NISAR orbit thinned to 60 s. The toolkit's
    8-point Lagrange interpolation (`insardev_toolkit.utils_alos2_orbit.orbit_state`) on the 60 s vectors densified to 10 s,
    then the package's Hermite on those, is within 0.4 mm and 4 um/s. So the vectors here are the Lagrange states at the times
    first + 10 k within the LED span (first and last vector included).

    Parameters
    ----------
    led_path : str
        Path to the LED file.
    img_path : str
        Path to an IMG file of the LED: its line timing (alos2_timing) is the scene the orbit must cover.
    t1 : float, optional
        Start time as year * 1000 + 0-based day + day fraction for filtering.
    t2 : float, optional
        End time as year * 1000 + 0-based day + day fraction for filtering.

    Raises
    ------
    ValueError
        The orbit cannot be used (utils_satellite.orbit_defect): its state vectors do not cover the scene's first to last line,
        have a gap, or one is not finite.

    Returns
    -------
    pd.DataFrame
        The orbit as utils_nisar.nisar_orbit returns it: columns iy (year), id (0-based day), isec (seconds of day), px, py, pz
        (m), vx, vy, vz (m/s), clock (seconds from Jan 1 of the scene year, continuous through midnight and Jan 1); attrs nd, iy,
        id, isec and idsec (10.0).

    Examples
    --------
    >>> alos2_orbit('129_4450_F4/AL2_129_4450_F4_20150404.led', '129_4450_F4/AL2_129_4450_F4_20150404_HH.img')
    """
    from .utils_satellite import orbit_densify

    led = alos2_led(led_path)
    timing = alos2_timing(img_path, led_path)
    return orbit_densify(led['orbit'], timing['day'], timing['start'], timing['lines'], timing['prf'], led_path, t1, t2,
                         step=ALOS2_ORBIT_STEP)


def alos2_prm(img_path: str, led_path: str) -> dict:
    """
    PRM parameters of an ALOS-2 IMG file.

    The keys of utils_nisar.nisar_prm, from the LED (radar constants, ellipsoid, look direction), the line timing (alos2_timing)
    and the file name, without granule_duration and frequency: the closest-approach search window of an ALOS-2 repeat is half
    the record's own duration (utils_satellite.s1_burst_half_window). The name keys 'path', 'frame', 'subswath' and
    'polarization' are technical PRM keys, as NISAR's 'track' and 'frame'.

    Parameters
    ----------
    img_path : str
        Path to the IMG file, named AL2_<path>_<frame>_<Fn>_<yyyymmdd>_<pol>.img.
    led_path : str
        Path to its LED file.

    Returns
    -------
    dict
        The PRM parameters: PRM().set(**params). SC_identity 5 (ALOS in GMTSAR); fd1, fdd1 and fddd1 are 0 (zero-Doppler L1.1);
        orbdir is 'A' when the orbit's vz at the scene's middle line is positive, else 'D' (GMTSAR ALOS_ldr_orbit); chirp_slope
        is GMTSAR's final ALOS-2 value: the chirp linear coefficient of line 0's prefix x 1e6 (Hz/s, positive;
        read_ALOS_data_SLC.c:325 overwrites the leader's negated coefficient with it).

    Raises
    ------
    ValueError
        The file name is not that of an ALOS-2 IMG file (and the errors of alos2_led and alos2_timing).

    Examples
    --------
    >>> alos2_prm('129_4450_F4/AL2_129_4450_F4_20150404_HH.img', '129_4450_F4/AL2_129_4450_F4_20150404.led')['near_range']
    """
    from insardev_toolkit.utils_alos2_orbit import orbit_state

    match = ALOS2_NAME.match(os.path.basename(img_path))
    if match is None:
        raise ValueError(f'{img_path}: the file name is not AL2_<path>_<frame>_<Fn>_<yyyymmdd>_<pol>.img')
    path_number, frame_number, subswath, _, pol = match.groups()

    led = alos2_led(led_path)
    timing = alos2_timing(img_path, led_path)
    day, prf, lines, pixels = timing['day'], timing['prf'], timing['lines'], timing['pixels']
    year = day.year
    jd = day.timetuple().tm_yday - 1  # 0-based
    clock_start = jd + timing['start'] / 86400.0
    clock_stop = clock_start + lines / prf / 86400.0

    # the orbit direction from the orbit's vz at the scene's middle line (on the orbit's day)
    middle = timing['start'] + 0.5 * (lines - 1) / prf + 86400.0 * int((np.datetime64(day.date()) - np.datetime64(led['orbit']['date'])).astype(np.int64))
    _, velocity = orbit_state(led['orbit'], middle)

    return {
        'input_file': img_path,
        'first_line': 1,
        'st_rng_bin': 1,
        'nlooks': 1,
        'rshift': 0,
        'ashift': 0,
        'sub_int_r': 0.0,
        'sub_int_a': 0.0,
        'stretch_r': 0.0,
        'stretch_a': 0.0,
        'a_stretch_r': 0.0,
        'a_stretch_a': 0.0,
        'dtype': 'a',
        'rng_samp_rate': led['rng_samp_rate'],
        'PRF': prf,
        # 5 = ALOS in GMTSAR
        'SC_identity': 5,
        'radar_wavelength': led['wavelength'],
        # year * 1000 + day and whole seconds only (never for timing, see _sc_clock_whole_second)
        'SC_clock_start': _sc_clock_whole_second(year, clock_start),
        'SC_clock_stop': _sc_clock_whole_second(year, clock_stop),
        'clock_start': clock_start,
        'clock_stop': clock_stop,
        'near_range': timing['near_range'],
        'num_rng_bins': pixels,
        'bytes_per_line': pixels * 8,
        'nrows': lines,
        'num_lines': lines,
        'num_valid_az': lines,
        'num_patches': 1,
        'orbdir': 'A' if float(velocity[2]) > 0 else 'D',
        'lookdir': led['lookdir'],
        'polarization': pol,
        'path': int(path_number),
        'frame': int(frame_number),
        'subswath': subswath,
        # GMTSAR's ALOS-2 SLC reader: the line prefix's chirp linear coefficient (Hz/us) x 1e6 (read_ALOS_data_SLC.c:325)
        'chirp_slope': float(alos2_prefix(img_path, [0])['chirp_linear'][0]) * 1e6,
        'pulse_dur': led['pulse_dur'],
        'chirp_ext': 0,
        'fd1': 0.0,
        'fdd1': 0.0,
        'fddd1': 0.0,
        'equatorial_radius': led['equatorial_radius'],
        'polar_radius': led['polar_radius'],
    }


def alos2_slc(img_path: str, row_slice: slice = None, col_slice: slice = None) -> np.ndarray:
    """
    Read a box of an ALOS-2 (or ALOS-1 L1.1) IMG file.

    One byte range per line through a memmap of the line records: the pixels after the line prefix (544 bytes for ALOS-2, 412
    for ALOS-1: alos2_image), big-endian complex64, converted to native complex64. The memmap is closed before returning.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.
    row_slice : slice, optional
        0-based lines.
    col_slice : slice, optional
        0-based range bins.

    Returns
    -------
    numpy.ndarray
        complex64 (lines, bins) of the box.

    Examples
    --------
    >>> alos2_slc('129_4450_F4/AL2_129_4450_F4_20150404_HH.img', slice(0, 64), slice(0, 64))
    """
    from insardev_toolkit.utils_alos2_ceos import DESCRIPTOR

    info = alos2_image(img_path)
    prefix = info['prefix']
    if row_slice is None:
        row_slice = slice(None)
    if col_slice is None:
        col_slice = slice(None)
    first, stop, step = col_slice.indices(info['pixels'])
    if step < 0 or stop <= first:
        # a reversed or empty column range: read its covering bins, then take the slice of them
        first, stop, step, cols = 0, info['pixels'], 1, col_slice
    else:
        cols = slice(None, None, step)
    records = np.memmap(img_path, dtype=np.uint8, mode='r', offset=DESCRIPTOR, shape=(info['lines'], info['record_length']))
    box = records[row_slice, prefix + 8 * first:prefix + 8 * stop]
    slc = box.view('>c8')[:, cols].astype(np.complex64)
    del box, records
    return slc


def alos2_footprint(img_path: str):
    """
    Footprint of an ALOS-2 (or ALOS-1 L1.1) IMG file at height 0 from JAXA's per-line positions.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.

    Returns
    -------
    shapely.geometry.Polygon
        Longitude, latitude of the first line's first, mid and last pixel and of the last line's last, mid and first pixel,
        closed.

    Examples
    --------
    >>> alos2_footprint('129_4450_F4/AL2_129_4450_F4_20150404_HH.img').bounds
    """
    from shapely.geometry import Polygon

    info = alos2_image(img_path)
    prefix = alos2_prefix(img_path, [0, info['lines'] - 1])
    lon, lat = prefix['lon'], prefix['lat']
    points = [(lon[0, j], lat[0, j]) for j in (0, 1, 2)] + [(lon[1, j], lat[1, j]) for j in (2, 1, 0)]
    return Polygon(points + [points[0]])
