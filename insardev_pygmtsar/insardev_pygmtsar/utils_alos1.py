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
ALOS-1 PALSAR L1.1 (SLC) crops as insardev_toolkit `ASF.download` stores them: per date an IMG file per polarization
(`AL1_<path>_<frame>_<offnadir>_<yyyymmdd>_<pol>.img`, offnadir the off-nadir angle in 0.1 degree) and one leader file
(`AL1_<path>_<frame>_<offnadir>_<yyyymmdd>.led`).

The files have the CEOS layout of ALOS-2 with a 412-byte line prefix (the descriptor's prefix length field), so the image,
prefix, leader, box and footprint readers of utils_alos2 read them (alos2_image, alos2_prefix, alos2_led, alos2_slc,
alos2_footprint). ALOS-1 has no microsecond time field (bytes 85-92) and no window position field (bytes 121-124): they are
spare and 0. Its own functions here are the line timing, the PRM and the orbit.

The image is focused at JAXA's Doppler centroid, not at zero Doppler: a line holds the points whose Doppler is the centroid
fd1 + fdd1 R of their slant range R (the leader's data set summary, bytes 1735-1750 in Hz and 1751-1766 in Hz/km; GMTSAR reads
the same two fields). Measured on raw/alos1_rosamond (session/patches/preproc/reports/A0a_report.md, task 2): JAXA's per-line
positions of the first, mid and last pixel mapped on the LED orbit at that centroid follow start + k / PRF within +-9.5 us and
the near range + j c / (2 fs) within +-0.037 m (the 1e-6 degree rounding of the positions); at zero Doppler they are 230-280
lines away. The millisecond field of the lines is a truncation (0.99 ms of scatter, -1.74 .. -2.68 ms from the fitted start)
and the integer near-range field is off by up to 0.48 m, so both are fitted here (alos1_timing).
"""
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .utils_alos2 import SOL, alos2_image, alos2_prefix, alos2_led, _dss_fields, _fields_constant, _line_positions
from .utils_s1 import _sc_clock_whole_second

# the step of the orbit given to the package (s), as ALOS-2's (utils_alos2.ALOS2_ORBIT_STEP)
ALOS1_ORBIT_STEP = 10.0
# the timing reads every this many lines (and the last line)
ALOS1_TIMING_LINES = 64
# the largest residual of the timing fit: twice the largest measured by unit A0a on four Rosamond dates (time 9.5 us = 0.02
# lines, the 1e-6 degree rounding of JAXA's latitudes; range 0.037 m), a margin of 2 for other scenes and latitudes
ALOS1_RESIDUAL_TIME = 2e-5        # s
ALOS1_RESIDUAL_RANGE = 0.08       # m
# the IMG file name: path, frame, off-nadir angle (0.1 degree), date, polarization
ALOS1_NAME = re.compile(r'^AL1_(\d+)_(\d+)_(\d+)_(\d{8})_([HV][HV])\.img$')
# the Doppler centroid of the data set summary record (type 10): name -> (first byte, last byte, scale), 1-based bytes;
# the constant term in Hz and the linear term in Hz/km, scaled to Hz/m as GMTSAR does (read_ALOS_sarleader.c: 0.001 x)
ALOS1_DOPPLER_FIELDS = {
    'fd1': (1735, 1750, 1.0),
    'fdd1': (1751, 1766, 1e-3),
}


def alos1_led(led_path: str) -> dict:
    """
    Radar constants, ellipsoid, orbit and Doppler centroid of an ALOS-1 PALSAR L1.1 leader (LED) file.

    Parameters
    ----------
    led_path : str
        Path to the LED file.

    Returns
    -------
    dict
        utils_alos2.alos2_led of the file, and 'fd1' (Hz) and 'fdd1' (Hz/m): the Doppler centroid fd1 + fdd1 R at the slant
        range R (m) of the data set summary, GMTSAR's fd1 and fdd1; 'fddd1' 0.0 (the leader has no quadratic term, GMTSAR 0 too).

    Raises
    ------
    ValueError
        The errors of alos2_led; a Doppler field that is not a number.

    Examples
    --------
    >>> alos1_led('216_0680_343/AL1_216_0680_343_20070703.led')['fd1']
    """
    led = alos2_led(led_path)
    with open(led_path, 'rb') as f:
        data = f.read()
    led.update(_dss_fields(data, led_path, ALOS1_DOPPLER_FIELDS))
    led['fddd1'] = 0.0
    return led


def alos1_timing(img_path: str, led_path: str) -> dict:
    """
    Line timing and near range of an ALOS-1 PALSAR L1.1 IMG file, fitted to JAXA's per-line positions on the LED orbit with
    the LED Doppler centroid.

    The lines are start + k / prf (k the 0-based line, prf the mHz PRF field of the prefix, bytes 57-60). The sampled lines
    k = 0, 64, 128, ... and the last line are read: the latitude and longitude of their first, mid and last pixel (height 0,
    prefix bytes 193-216) are mapped by `insardev_toolkit.utils_alos2_orbit.geo2rdr` on the LED orbit to the time and slant
    range at which their Doppler is the leader's centroid fd1 + fdd1 R (alos1_led). start is the median of t - k / prf over
    the three pixels; the near range is the median slant range of the first pixel. The pixels are at the range bins 0,
    N / 2 - 1 and N - 1 (N pixels, 0-based: JAXA's mid pixel is N / 2 - 1, measured by unit A0a), c / (2 rng_samp_rate)
    apart. The millisecond field (a truncation, 0.99 ms of scatter) is only the start of the solve, and the integer
    near-range field is not used for the values.

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
        'near_range' (m), 'lines', 'pixels'; the fit: 'residual_time' (s) and 'residual_range' (m), the largest absolute
        residual of the three pixels of the sampled lines against start + k / prf and near_range + bin c / (2 rng_samp_rate);
        'ms_offset' (s: start minus the millisecond field of line 0) and 'near_offset' (m: near_range minus the integer
        near-range field).

    Raises
    ------
    ValueError
        The PRF, pixel count or near-range field differs between the sampled lines; or the line positions do not follow the
        orbit (residual_time above ALOS1_RESIDUAL_TIME, residual_range above ALOS1_RESIDUAL_RANGE, or no solution on the
        orbit): the IMG and the LED do not belong together or one is damaged.

    Examples
    --------
    >>> alos1_timing('216_0680_343/AL1_216_0680_343_20070703_HH.img', '216_0680_343/AL1_216_0680_343_20070703.led')['start']
    """
    info = alos2_image(img_path)
    led = alos1_led(led_path)
    lines, pixels = info['lines'], info['pixels']
    # the sampled lines: k = 0, 64, 128, ... and the last line
    k = np.unique(np.append(np.arange(0, lines, ALOS1_TIMING_LINES), lines - 1))
    prefix = alos2_prefix(img_path, k)
    _fields_constant(img_path, prefix, ('prf', 'pixels', 'near_range'))
    day = datetime.strptime(info['date'], '%Y-%m-%d')
    doppler = (led['fd1'], led['fdd1'], led['wavelength'])
    _, t_ms, t, slant_range = _line_positions(img_path, led_path, info, led['orbit'], prefix, doppler)

    # the line spacing is the mHz PRF field
    prf = float(prefix['prf'][0])
    start = float(np.median(t - k[:, None] / prf))
    near_range = float(np.median(slant_range[:, 0]))
    spacing = SOL / (2.0 * led['rng_samp_rate'])
    bins = np.array([0, pixels // 2 - 1, pixels - 1], dtype=np.float64)
    residual_time = float(np.abs(t - (start + k[:, None] / prf)).max())
    residual_range = float(np.abs(slant_range - (near_range + bins * spacing)).max())
    if not (residual_time <= ALOS1_RESIDUAL_TIME and residual_range <= ALOS1_RESIDUAL_RANGE):
        raise ValueError(f"{img_path}: JAXA's line positions do not follow the orbit of {led_path} (residuals "
                         f"{1e6 * residual_time:.1f} us, {residual_range:.3f} m): the files do not belong together or one is "
                         f"damaged; download them again")
    # the day of the first line's time
    days = int(np.floor(start / 86400.0))
    return {'day': day + timedelta(days=days),
            'start': start - 86400.0 * days,
            'prf': prf,
            'near_range': near_range,
            'lines': lines,
            'pixels': pixels,
            'residual_time': residual_time,
            'residual_range': residual_range,
            'ms_offset': start - float(t_ms[0]),
            'near_offset': near_range - float(prefix['near_range'][0])}


def alos1_orbit(led_path: str, img_path: str, t1: float = None, t2: float = None) -> pd.DataFrame:
    """
    Orbit state vectors of an ALOS-1 scene: the LED vectors (60 s apart) densified to 10 s by
    `utils_satellite.orbit_densify`, as utils_alos2.alos2_orbit.

    Parameters
    ----------
    led_path : str
        Path to the LED file.
    img_path : str
        Path to an IMG file of the LED: its line timing (alos1_timing) is the scene the orbit must cover.
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
        The orbit as utils_satellite.orbit_densify returns it: columns iy, id (0-based day), isec, px, py, pz, vx, vy, vz,
        clock; attrs nd, iy, id, isec and idsec (10.0).

    Examples
    --------
    >>> alos1_orbit('216_0680_343/AL1_216_0680_343_20070703.led', '216_0680_343/AL1_216_0680_343_20070703_HH.img')
    """
    from .utils_satellite import orbit_densify

    led = alos2_led(led_path)
    timing = alos1_timing(img_path, led_path)
    return orbit_densify(led['orbit'], timing['day'], timing['start'], timing['lines'], timing['prf'], led_path, t1, t2,
                         step=ALOS1_ORBIT_STEP)


def alos1_prm(img_path: str, led_path: str) -> dict:
    """
    PRM parameters of an ALOS-1 PALSAR L1.1 IMG file.

    The keys of utils_alos2.alos2_prm, with ALOS-1's values: the line timing and near range of alos1_timing, the Doppler
    centroid of the leader (alos1_led), the chirp slope of the line prefix read as the float it is. The name keys 'path',
    'frame', 'subswath' (the off-nadir code of the name, the beam: one per stack) and 'polarization' are technical PRM keys.

    Differences from GMTSAR's `ALOS_pre_process_SLC -ALOS1`:
    - clock_start, clock_stop, SC_clock_start, SC_clock_stop: the fitted line time of alos1_timing (GMTSAR: the millisecond
      field of line 0 minus 2.0835 ms, off by -0.60 .. +0.34 ms on the Rosamond dates); the 0-based day as alos2_prm (GMTSAR
      the 1-based day);
    - near_range: the fitted near range (GMTSAR: the integer field, off by up to 0.48 m);
    - chirp_slope: prefix bytes 77-80 as the big-endian float32 they are, Hz/us x 1e6 (FBD 5.1851853e11 Hz/s, the leader's
      value to float32 rounding; GMTSAR reads the 4 bytes as an int32: 1.22455e15);
    - bytes_per_line: 8 x pixels, the IMG's complex64 read directly as alos2_prm (GMTSAR: 4 x pixels of its int16 SLC file);
    - input_file: the IMG path, as alos2_prm (GMTSAR: its SLC file).

    Parameters
    ----------
    img_path : str
        Path to the IMG file, named AL1_<path>_<frame>_<offnadir>_<yyyymmdd>_<pol>.img.
    led_path : str
        Path to its LED file.

    Returns
    -------
    dict
        The PRM parameters: PRM().set(**params). SC_identity 5 (ALOS in GMTSAR); fd1 (Hz), fdd1 (Hz/m) and fddd1 (0) of the
        leader, in GMTSAR's units; orbdir is 'A' when the orbit's vz at the scene's middle line is positive, else 'D'.

    Raises
    ------
    ValueError
        The file name is not that of an ALOS-1 IMG file (and the errors of alos1_led and alos1_timing).

    Examples
    --------
    >>> alos1_prm('216_0680_343/AL1_216_0680_343_20070703_HH.img', '216_0680_343/AL1_216_0680_343_20070703.led')['chirp_slope']
    """
    from insardev_toolkit.utils_alos2_orbit import orbit_state

    match = ALOS1_NAME.match(os.path.basename(img_path))
    if match is None:
        raise ValueError(f'{img_path}: the file name is not AL1_<path>_<frame>_<offnadir>_<yyyymmdd>_<pol>.img')
    path_number, frame_number, offnadir, _, pol = match.groups()

    led = alos1_led(led_path)
    timing = alos1_timing(img_path, led_path)
    day, prf, lines, pixels = timing['day'], timing['prf'], timing['lines'], timing['pixels']
    year = day.year
    jd = day.timetuple().tm_yday - 1  # 0-based
    clock_start = jd + timing['start'] / 86400.0
    clock_stop = clock_start + lines / prf / 86400.0

    # the orbit direction from the orbit's vz at the scene's middle line (on the orbit's day)
    middle = timing['start'] + 0.5 * (lines - 1) / prf + 86400.0 * int((np.datetime64(day.date()) - np.datetime64(led['orbit']['date'])).astype(np.int64))
    _, velocity = orbit_state(led['orbit'], middle)
    # the chirp linear coefficient of line 0's prefix: the 4 bytes of a big-endian float32 in Hz/us
    chirp = float(np.array(alos2_prefix(img_path, [0])['chirp_linear'], dtype=np.int32).view(np.float32)[0])

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
        'subswath': offnadir,
        'chirp_slope': chirp * 1e6,
        'pulse_dur': led['pulse_dur'],
        'chirp_ext': 0,
        'fd1': led['fd1'],
        'fdd1': led['fdd1'],
        'fddd1': led['fddd1'],
        'equatorial_radius': led['equatorial_radius'],
        'polar_radius': led['polar_radius'],
    }


def alos1_doppler_centroid(prm) -> float:
    """
    The Doppler centroid of an ALOS-1 date at mid swath (Hz): the leader's fd1 + fdd1 R at the slant range R of the middle
    range bin, (num_rng_bins - 1) / 2, of the PRM.

    It is the record column 'doppler' of the scan (Alos1_slc) and, divided by the PRF, the date's centroid in cycles per line
    of the alignment and the transform (the stripmap hook _doppler_centroid).

    Parameters
    ----------
    prm : PRM or dict
        The PRM of the date or its parameters (alos1_prm): fd1, fdd1, near_range, rng_samp_rate and num_rng_bins are read.

    Returns
    -------
    float
        The centroid in Hz.

    Examples
    --------
    >>> alos1_doppler_centroid(alos1_prm('216_0680_343/AL1_216_0680_343_20070703_HH.img',
    ...                                  '216_0680_343/AL1_216_0680_343_20070703.led'))
    """
    mid = float(prm.get('near_range')) + 0.5 * (int(prm.get('num_rng_bins')) - 1) * SOL / (2.0 * float(prm.get('rng_samp_rate')))
    return float(prm.get('fd1')) + float(prm.get('fdd1')) * mid


def alos1_native_geometry(prm, orbit_df: pd.DataFrame, height: float) -> dict:
    """
    The native-geometry parameters of an ALOS-1 date: the terms of utils_satellite.native_doppler_shift() and
    native_doppler_phase(), from its PRM and orbit.

    The image is focused at the leader's Doppler centroid fd1 + fdd1 R (the module's docstring); the parameters are that
    centroid, the wavelength, the PRF, the near range and the bin spacing of the PRM, and the effective velocity at the first
    and the last range bin: v^2 = V.V + (S - P).A at the scene's middle line (S, V, A the position, velocity and
    acceleration of the orbit, Hermite-interpolated as utils_satellite.satellite_llt2rat does), P the zero-Doppler point at
    that slant range, on the look side, at the geocentric radius of the ellipsoid under the satellite plus height.

    The velocity depends on the height of the ground: measured on raw/alos1_rosamond (unit A3 of the SLC preprocessor plan),
    0.4-0.5 m/s per 1000 m, about 0.03 lines of the shift. With the scene's median DEM height the closed form is within
    0.026 lines of the exact solve over the whole swath (ground heights 630-1730 m), with height 0 within 0.054 lines.

    Parameters
    ----------
    prm : PRM
        The PRM of the date (alos1_prm): clock_start, PRF, num_lines, num_rng_bins, near_range, rng_samp_rate,
        radar_wavelength, fd1, fdd1 and lookdir are read.
    orbit_df : pandas.DataFrame
        The orbit of the date (alos1_orbit).
    height : float
        The ground height above the WGS84 ellipsoid (m) of the effective velocity: the scene's median DEM height.

    Returns
    -------
    dict
        'fd1' (Hz), 'fdd1' (Hz/m), 'wavelength' (m), 'prf' (Hz), 'near_range' (m), 'range_pixel' (m), 'velocity' (m/s at the
        first and the last bin, a tuple of float) and 'pixels': plain values (the transform's chunk workers receive them).

    Examples
    --------
    >>> alos1_native_geometry(prm, prm.orbit_df, 800.0)['velocity']
    """
    from .utils_satellite import _hermite_interp, _ecef_to_geodetic, geocentric_radius

    prf, lines, pixels = float(prm.get('PRF')), int(prm.get('num_lines')), int(prm.get('num_rng_bins'))
    near_range = float(prm.get('near_range'))
    range_pixel = SOL / (2.0 * float(prm.get('rng_samp_rate')))

    # the orbit at the scene's middle line, and 1 s either side for the acceleration (seconds from the year's start)
    clock = orbit_df['clock'].to_numpy()
    t_mid = 86400.0 * float(prm.get('clock_start')) + 0.5 * lines / prf
    times = np.array([t_mid - 1.0, t_mid, t_mid + 1.0])
    position = np.array([_hermite_interp(clock, orbit_df[p].to_numpy(), orbit_df[v].to_numpy(), times[1:2])[0]
                         for p, v in (('px', 'vx'), ('py', 'vy'), ('pz', 'vz'))])
    step = clock[1] - clock[0]
    velocities = np.array([_hermite_interp(clock, orbit_df[v].to_numpy(), np.gradient(orbit_df[v].to_numpy(), step), times)
                           for v in ('vx', 'vy', 'vz')])
    velocity, acceleration = velocities[:, 1], (velocities[:, 2] - velocities[:, 0]) / 2.0

    # the zero-Doppler plane: along = V / |V|, up = the part of S normal to it, right = along x up (the right-looking side)
    _, lat, _ = _ecef_to_geodetic(*position)
    radius = float(geocentric_radius(np.radians(lat))) + height
    along = velocity / np.linalg.norm(velocity)
    normal = position - (position @ along) * along
    normal_length = np.linalg.norm(normal)
    up = normal / normal_length
    side = (1.0 if prm.get('lookdir') == 'R' else -1.0) * np.cross(along, up)
    effective = []
    for slant in (near_range, near_range + (pixels - 1) * range_pixel):
        # the look vector u at slant range R to a point at the radius: |S + R u| = radius, u normal to V
        cos_look = (position @ position + slant * slant - radius * radius) / (2.0 * normal_length * slant)
        look = -cos_look * up + np.sqrt(1.0 - cos_look * cos_look) * side
        effective.append(float(np.sqrt(velocity @ velocity - slant * (look @ acceleration))))

    return {'fd1': float(prm.get('fd1')), 'fdd1': float(prm.get('fdd1')), 'wavelength': float(prm.get('radar_wavelength')),
            'prf': prf, 'near_range': near_range, 'range_pixel': range_pixel, 'velocity': tuple(effective), 'pixels': pixels}
