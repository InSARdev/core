# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
"""
Sentinel-1 utility functions to replace GMTSAR binaries.
Pure Python implementations without disk I/O or external binaries.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Days before the 1st of each month in a common year, for the 0-based julian day
# GMTSAR wants. Leap years add one from March on.
_DOY_CUM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def _ydf_to_iso(ydf: float) -> str:
    """`year.day_fraction` (GMTSAR's SC_clock format) as an ISO-8601 timestamp.

    Only used to bracket a text search, so the seconds are truncated rather than
    rounded -- widening the window by under a second either way is harmless when
    an exact numeric test follows. A window across Jan 1 runs below day 0 of the
    scene year (or past Dec 31), so the year is the nearest thousand.
    """
    import math
    from datetime import datetime, timedelta
    year = int(math.floor(ydf / 1000.0 + 0.5))
    rest = ydf - year * 1000
    jd = math.floor(rest)
    sec = (rest - jd) * 86400.0
    dt = datetime(year, 1, 1) + timedelta(days=jd, seconds=sec)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


def _jan1_days(year0: int, year: int) -> int:
    """Days from Jan 1 of year0 to Jan 1 of year: the day count of year0 carried into a neighbouring year."""
    return (datetime(int(year), 1, 1) - datetime(int(year0), 1, 1)).days


def _aztime_seconds(dt, day0) -> float:
    """Seconds of a datetime from 00:00 UTC of the date day0; on day0 itself exactly the seconds of the day.

    GMTSAR compares annotation times with the day kept (yyyyddd.fraction), so a time just past midnight stays
    seconds, not a day, from a burst before it.
    """
    return (86400 * (dt.date() - day0).days
            + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6)


def orbit_window(start: float, stop: float) -> tuple:
    """
    The time the orbit of a burst is read for, (t1, t2): its first to last line, start and stop as
    year.day_fraction (SC_clock_start, SC_clock_stop), extended by 1400 s (about 23 minutes) on each side.
    """
    return start - 1400.0 / 86400.0, stop + 1400.0 / 86400.0


def _orbit_for(burst: tuple, t1: float, t2: float) -> str:
    """What an orbit file error names: the burst (name, start, stop) and its date, or the time t1 to t2."""
    if burst is not None:
        return f'burst {burst[0]} ({_ydf_to_iso(burst[1])[:10]})'
    return f'{_ydf_to_iso(t1)[:19]} to {_ydf_to_iso(t2)[:19]}'


def satellite_orbit(xml_path: str, t1: float, t2: float, burst: tuple = None) -> pd.DataFrame:
    """
    Extract orbit state vectors from Sentinel-1 EOF XML file.

    Replaces GMTSAR ext_orb_s1a binary.

    Parameters
    ----------
    xml_path : str
        Path to orbit EOF XML file (e.g., S1A_OPER_AUX_POEORB_*.EOF)
    t1 : float
        Start time as year.day_fraction (e.g., 2015.021 + seconds/86400)
        This is GMTSAR's SC_clock_start format
    t2 : float
        End time as year.day_fraction
        This is GMTSAR's SC_clock_stop format
    burst : tuple, optional
        (name, start, stop) of the burst the orbit is read for, start and stop as year.day_fraction:
        the state vectors must cover this time, and the error names the burst and its date.

    Raises
    ------
    FileNotFoundError
        There is no orbit file at xml_path.
    ValueError
        The orbit file cannot be used: it has no state vectors over the time (the burst when given),
        a gap between them, or a state vector that is not finite (utils_satellite.orbit_defect).

    Returns
    -------
    pd.DataFrame
        Orbit state vectors with columns:
        - iy: year
        - id: julian day
        - isec: seconds of day
        - px, py, pz: ECEF position (meters)
        - vx, vy, vz: ECEF velocity (m/s)
        - clock: seconds from Jan 1 of the scene year (for interpolation), continuous through
          midnight and Jan 1 (a vector of the next year counts on from day 365/366, one of the
          previous year from day -1), like t1/t2 and PRM clock_start

        DataFrame.attrs contains metadata:
        - nd: number of records
        - idsec: time step (seconds)

    Examples
    --------
    >>> # Get orbit for a burst (extend time range by ~23 minutes on each side)
    >>> t1, t2 = orbit_window(prm.get('SC_clock_start'), prm.get('SC_clock_stop'))
    >>> orbit_df = satellite_orbit(eof_path, t1, t2)
    """
    df, _ = _eof_state_vectors(xml_path, t1, t2, burst)
    # an orbit file that cannot be used raises: the state vectors must cover the burst, without a gap, and be finite
    error = _eof_error(xml_path, df, t1, t2, burst)
    if error is not None:
        raise error

    # Store metadata
    if len(df) > 1:
        dt = df['isec'].iloc[1] - df['isec'].iloc[0]
        if dt < 0:  # day boundary crossing
            dt = (df['clock'].iloc[1] - df['clock'].iloc[0])
    else:
        dt = 10.0  # default 10s for single record

    df.attrs = {
        'nd': len(df),
        'iy': int(df['iy'].iloc[0]),
        'id': int(df['id'].iloc[0]),
        'isec': float(df['isec'].iloc[0]),
        'idsec': dt
    }

    return df


def check_orbit_file(xml_path: str, bursts: list) -> None:
    """
    Check an EOF file for every burst that uses it with one read of the file: the state vectors of each burst
    (those satellite_orbit reads for it, over orbit_window) are checked as satellite_orbit checks them.

    Parameters
    ----------
    xml_path : str
        Path to orbit EOF XML file.
    bursts : list
        (name, start, stop) of each burst, start and stop as year.day_fraction (SC_clock_start, SC_clock_stop).

    Raises
    ------
    FileNotFoundError, ValueError
        The error of satellite_orbit for a burst whose orbit cannot be used.
    """
    windows = [orbit_window(start, stop) for _, start, stop in bursts]
    # one read per day count (year * 1000 + day, the year of t1): the bursts of a date across Jan 1 have two
    years = [int(np.floor(t1 / 1000.0 + 0.5)) for t1, _ in windows]
    for year in dict.fromkeys(years):
        group = [i for i, y in enumerate(years) if y == year]
        df, ydf = _eof_state_vectors(xml_path, min(windows[i][0] for i in group),
                                     max(windows[i][1] for i in group), bursts[group[0]])
        for i in group:
            t1, t2 = windows[i]
            error = _eof_error(xml_path, df[(ydf >= t1) & (ydf <= t2)], t1, t2, bursts[i])
            if error is not None:
                raise error


def _eof_error(xml_path: str, df: pd.DataFrame, t1: float, t2: float, burst: tuple):
    """The error of the state vectors df of an EOF file read from t1 to t2 for the burst (name, start, stop) when
    they cannot be used (utils_satellite.orbit_defect), None when they can."""
    from .utils_satellite import orbit_defect
    year0 = int(np.floor(t1 / 1000.0 + 0.5))
    span = ((burst[1] - year0 * 1000) * 86400.0, (burst[2] - year0 * 1000) * 86400.0) if burst is not None else ()
    defect = orbit_defect(df, *span)
    if defect is None:
        return None
    return ValueError(f'ERROR: Orbit file {xml_path} {defect} {_orbit_for(burst, t1, t2)}. '
                      f'Delete it and download the orbits again.')


def _eof_state_vectors(xml_path: str, t1: float, t2: float, burst: tuple = None) -> tuple:
    """
    The state vectors of an EOF file from t1 to t2 (year.day_fraction), in file order, without a check.

    Returns
    -------
    tuple
        (pd.DataFrame with the columns of satellite_orbit and no attrs, np.ndarray of each state vector's
        year.day_fraction on the day count of t1); the DataFrame holds only 'clock' when no state vector is in
        the time.

    Raises
    ------
    FileNotFoundError
        There is no file at xml_path; the error names the burst (name, start, stop) when given.
    """
    import xml.etree.ElementTree as ET
    from insardev_toolkit.utils_files import exists

    if not exists(xml_path):
        raise FileNotFoundError(f'ERROR: Orbit file {xml_path} not found for {_orbit_for(burst, t1, t2)}. '
                                f'Download the orbits again.')

    # ONLY THE WINDOW IS PARSED. A precise-orbit file covers 26 hours at 10 s --
    # 9,361 state vectors -- and a burst asks for about 47 minutes of it, so
    # building the whole DOM and converting every timestamp throws away 97% of
    # the work, once per burst. ISO-8601 timestamps sort lexicographically in
    # chronological order, so the blocks can be selected as TEXT first and only
    # the survivors handed to the XML parser. The string window is widened by a
    # margin and the exact numeric test below still decides, so the result is
    # what parsing the whole file would have given.
    _margin = 60.0 / 86400.0
    root = None
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='replace') as f:
            _text = f.read()
        _blocks = _text.split('<OSV>')
        if len(_blocks) > 1:
            _lo = _ydf_to_iso(t1 - _margin)
            _hi = _ydf_to_iso(t2 + _margin)
            _kept = []
            for _b in _blocks[1:]:
                _i = _b.find('<UTC>UTC=')
                _j = _b.find('</OSV>')
                if _i < 0 or _j < 0:
                    _kept = None
                    break
                _ts = _b[_i + 9:_i + 35]
                if _lo <= _ts <= _hi:
                    _kept.append('<OSV>' + _b[:_j + 6])
            if _kept:
                root = ET.fromstring('<List_of_OSVs>' + ''.join(_kept) + '</List_of_OSVs>')
    except (OSError, UnicodeError, ET.ParseError):
        root = None
    if root is None:
        # anything unexpected about the file -- read it whole, as before
        root = ET.parse(xml_path).getroot()

    # Find all OSV (Orbit State Vector) elements
    osvs = root.findall('.//OSV')

    # t1/t2 count the days of the scene year (year * 1000 + day), running below day 0 or past Dec 31 when the
    # window crosses Jan 1: the vectors of the other year are put on that count too, and so is 'clock'
    year0 = int(np.floor(t1 / 1000.0 + 0.5))

    records = []
    ydfs = []
    for osv in osvs:
        # Parse UTC timestamp: "UTC=2015-01-20T22:59:44.000000". Fixed width, so
        # it is sliced rather than passed to strptime, which dominated this
        # function: one call per state vector, almost all of them discarded.
        utc_str = osv.find('UTC').text
        if utc_str.startswith('UTC='):
            utc_str = utc_str[4:]
        year = int(utc_str[0:4])
        month = int(utc_str[5:7])
        day = int(utc_str[8:10])
        # GMTSAR uses 0-based julian day (Jan 1 = day 0)
        jd = _DOY_CUM[month - 1] + day - 1
        if month > 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
            jd += 1
        sec = (int(utc_str[11:13]) * 3600 + int(utc_str[14:16]) * 60
               + int(utc_str[17:19]) + float(utc_str[19:]))

        # Compute year.day_fraction for filtering (GMTSAR format), on the day count of t1/t2
        if year == year0:
            ydf = year * 1000 + jd + sec / 86400.0
        else:
            ydf = year0 * 1000 + (jd + _jan1_days(year0, year)) + sec / 86400.0

        # Filter by time range
        if ydf < t1 or ydf > t2:
            continue

        # Extract position and velocity
        x = float(osv.find('X').text)
        y = float(osv.find('Y').text)
        z = float(osv.find('Z').text)
        vx = float(osv.find('VX').text)
        vy = float(osv.find('VY').text)
        vz = float(osv.find('VZ').text)

        records.append({
            'iy': year,
            'id': jd,
            'isec': sec,
            'px': x,
            'py': y,
            'pz': z,
            'vx': vx,
            'vy': vy,
            'vz': vz
        })
        ydfs.append(ydf)

    if len(records) == 0:
        return pd.DataFrame({'clock': np.empty(0)}), np.empty(0)

    df = pd.DataFrame(records)

    # Compute clock (seconds from Jan 1 of the scene year, continuous through Jan 1) for interpolation
    day = df['id']
    if (df['iy'] != year0).any():
        day = day + [_jan1_days(year0, y) for y in df['iy']]
    df['clock'] = (24 * 60 * 60) * day + df['isec']

    return df, np.array(ydfs, dtype=np.float64)


def doppler_centroid(orbit_df: pd.DataFrame,
                     clock_start: float,
                     prf: float,
                     near_range: float,
                     num_rng_bins: int,
                     num_valid_az: int,
                     num_patches: int,
                     nrows: int,
                     ra: float = 6378137.0,
                     rc: float = 6356752.31424518) -> dict:
    """
    Compute Doppler orbit parameters from orbit state vectors.

    Replaces GMTSAR calc_dop_orb binary.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit state vectors from satellite_orbit() or PRM.read_LED()
    clock_start : float
        Image start time in days (from PRM clock_start)
    prf : float
        Pulse repetition frequency in Hz
    near_range : float
        Near range distance in meters
    num_rng_bins : int
        Number of range bins
    num_valid_az : int
        Number of valid azimuth lines per patch
    num_patches : int
        Number of patches
    nrows : int
        Total number of rows
    ra : float
        Semi-major axis of reference ellipsoid (default WGS84)
    rc : float
        Semi-minor axis of reference ellipsoid (default WGS84)

    Returns
    -------
    dict
        Dictionary containing:
        - earth_radius: Local earth radius at scene center (meters)
        - SC_height: Spacecraft height above earth_radius (meters)
        - SC_height_start: Height at start of image
        - SC_height_end: Height at end of image
        - SC_vel: Ground velocity (m/s)
        - orbdir: Orbit direction ('A' for ascending, 'D' for descending)

    Examples
    --------
    >>> params = doppler_centroid(
    ...     orbit_df,
    ...     prm.get('clock_start'),
    ...     prm.get('PRF'),
    ...     prm.get('near_range'),
    ...     prm.get('num_rng_bins'),
    ...     prm.get('num_valid_az'),
    ...     prm.get('num_patches'),
    ...     prm.get('nrows')
    ... )
    >>> earth_radius = params['earth_radius']
    """
    from .utils_satellite import _hermite_interp

    # Prepare orbit data
    orbit_time = orbit_df['clock'].values
    px = orbit_df['px'].values
    py = orbit_df['py'].values
    pz = orbit_df['pz'].values
    vx = orbit_df['vx'].values
    vy = orbit_df['vy'].values
    vz = orbit_df['vz'].values

    # Compute acceleration for Hermite interpolation
    dt = orbit_time[1] - orbit_time[0]
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)

    # Time computation (matches GMTSAR ldr_orbit.c)
    t1 = 86400.0 * clock_start + (nrows - num_valid_az) / (2.0 * prf)
    t2 = t1 + num_patches * num_valid_az / prf
    t0 = (t1 + t2) / 2.0

    def calc_height_velocity(t_center, t_start, t_end):
        """Calculate height and velocity at given times."""
        # Interpolate orbit at center time
        # centre, and 2 s either side for the velocity difference -- three
        # epochs per axis, so three calls rather than nine
        _t3 = np.array([t_center - 2.0, t_center, t_center + 2.0])
        _x = _hermite_interp(orbit_time, px, vx, _t3)
        _y = _hermite_interp(orbit_time, py, vy, _t3)
        _z = _hermite_interp(orbit_time, pz, vz, _t3)
        x1, xs, x2 = _x[0], _x[1], _x[2]
        y1, ys, y2 = _y[0], _y[1], _y[2]
        z1, zs, z2 = _z[0], _z[1], _z[2]

        # Satellite distance from earth center
        rs = np.sqrt(xs**2 + ys**2 + zs**2)

        # Velocity (4 second interval)
        vx_sat = (x2 - x1) / 4.0
        vy_sat = (y2 - y1) / 4.0
        vz_sat = (z2 - z1) / 4.0
        vs = np.sqrt(vx_sat**2 + vy_sat**2 + vz_sat**2)

        # Geodetic latitude of satellite
        rlat = np.arcsin(zs / rs)

        # Local earth radius (ellipsoid)
        st = np.sin(rlat)
        ct = np.cos(rlat)
        arg = (ct * ct) / (ra * ra) + (st * st) / (rc * rc)
        re = 1.0 / np.sqrt(arg)

        # Height above ellipsoid
        height = rs - re

        # Ground velocity computation (follows GMTSAR approach)
        # Uses range-time polynomial fit
        ro = near_range

        # Compute target position at near range
        a = np.array([xs/rs, ys/rs, zs/rs])  # radial unit vector
        b = np.array([vx_sat/vs, vy_sat/vs, vz_sat/vs])  # velocity unit vector
        c = np.cross(a, b)  # cross-track

        # Look angle
        ct_look = (rs**2 + ro**2 - re**2) / (2.0 * rs * ro)
        st_look = np.sin(np.arccos(np.clip(ct_look, -1, 1)))

        # Target position
        xe = xs + ro * (-st_look * c[0] - ct_look * a[0])
        ye = ys + ro * (-st_look * c[1] - ct_look * a[1])
        ze = zs + ro * (-st_look * c[2] - ct_look * a[2])

        # Compute ground velocity from range-time polynomial
        nt = 100
        dt_sample = 200.0 / prf
        times = np.linspace(-dt_sample * nt / 2, dt_sample * nt / 2, nt)
        # _hermite_interp is vectorised over its query points and each point is
        # independent of the others, so the whole sample set goes in ONE call per
        # axis. Asking for one point at a time cost 300 calls here, three times
        # per burst, and was the largest remaining item in S1().
        t_k = t_center + times
        xk = _hermite_interp(orbit_time, px, vx, t_k)
        yk = _hermite_interp(orbit_time, py, vy, t_k)
        zk = _hermite_interp(orbit_time, pz, vz, t_k)
        ranges = np.sqrt((xe - xk)**2 + (ye - yk)**2 + (ze - zk)**2) - ro

        # Fit second-order polynomial
        coeffs = np.polyfit(times, ranges, 2)
        vg = np.sqrt(ro * 2.0 * coeffs[0])  # ground velocity

        return height, re, vg, vz_sat

    # Compute at start, center, and end
    height_start, re_start, vg_start, _ = calc_height_velocity(t1, t1, t1)
    height_end, re_end, vg_end, _ = calc_height_velocity(t2, t2, t2)
    height, re_c, vg, vz_center = calc_height_velocity(t0, t1, t2)

    # Use center earth radius
    re = re_c

    # Determine orbit direction
    orbdir = 'A' if vz_center > 0 else 'D'

    return {
        'earth_radius': re,
        'SC_height': height + re_c - re,
        'SC_height_start': height_start + re_start - re,
        'SC_height_end': height_end + re_end - re,
        'SC_vel': vg,
        'orbdir': orbdir
    }


def satellite_prm(xml_path: str, tiff_path: str) -> dict:
    """
    Extract PRM parameters from Sentinel-1 burst annotation XML file.

    Replaces GMTSAR make_s1a_tops pop_burst() function.

    Parameters
    ----------
    xml_path : str
        Path to burst annotation XML file
    tiff_path : str
        Path to burst GeoTIFF file (used for input_file parameter)

    Returns
    -------
    dict
        Dictionary containing all PRM parameters needed for processing.
        Can be used to create a PRM object via PRM().set(**params)

    Notes
    -----
    This extracts parameters matching GMTSAR's make_s1a_tops.c pop_burst() function.
    For single-burst processing, num_patches=1 and num_valid_az equals total valid lines.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime
    from scipy import constants

    SOL = constants.speed_of_light

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Helper functions
    def get_text(xpath):
        elem = root.find(xpath)
        return elem.text if elem is not None else None

    def get_float(xpath):
        text = get_text(xpath)
        return float(text) if text else None

    def get_int(xpath):
        text = get_text(xpath)
        return int(float(text)) if text else None

    # Extract parameters following GMTSAR make_s1a_tops.c pop_burst()
    prm = {}

    # Processing parameters
    prm['first_line'] = 1
    prm['st_rng_bin'] = 1
    prm['nlooks'] = get_int('.//rangeProcessing/numberOfLooks')
    prm['rshift'] = 0
    prm['ashift'] = 0
    prm['sub_int_r'] = 0.0
    prm['sub_int_a'] = 0.0
    prm['stretch_r'] = 0.0
    prm['stretch_a'] = 0.0
    prm['a_stretch_r'] = 0.0
    prm['a_stretch_a'] = 0.0
    prm['dtype'] = 'a'

    # Sampling rate
    fs = get_float('.//productInformation/rangeSamplingRate')
    prm['rng_samp_rate'] = fs

    # Satellite identity (10 = Sentinel-1)
    prm['SC_identity'] = 10

    # Wavelength from radar frequency (GMTSAR uses 'lambda' but that's a Python keyword)
    radar_freq = get_float('.//productInformation/radarFrequency')
    wavelength = SOL / radar_freq
    prm['radar_wavelength'] = wavelength

    # Chirp parameters
    tx_pulse_length = get_float('.//downlinkValues/txPulseLength')
    look_bandwidth = get_float('.//rangeProcessing/lookBandwidth')
    prm['chirp_slope'] = look_bandwidth / tx_pulse_length
    prm['pulse_dur'] = tx_pulse_length

    # I/Q mean (GMTSAR sets to 0, uses 'xmi' and 'xmq')
    prm['I_mean'] = 0.0
    prm['Q_mean'] = 0.0

    # PRF and timing
    azi_time_interval = get_float('.//imageInformation/azimuthTimeInterval')
    prm['PRF'] = 1.0 / azi_time_interval

    # Near range with GMTSAR correction (subtract 1/fs before multiplying by SOL/2)
    slant_range_time = get_float('.//imageInformation/slantRangeTime')
    prm['near_range'] = (slant_range_time - 1.0 / fs) * SOL / 2.0

    # Ellipsoid parameters
    prm['equatorial_radius'] = get_float('.//ellipsoidSemiMajorAxis') or 6378137.0
    prm['polar_radius'] = get_float('.//ellipsoidSemiMinorAxis') or 6356752.31

    # Orbit direction
    pass_dir = get_text('.//productInformation/pass')
    prm['orbdir'] = pass_dir[0].upper() if pass_dir else 'D'  # 'A' or 'D'
    prm['lookdir'] = 'R'  # Right looking

    # File paths
    prm['input_file'] = tiff_path
    # LED and SLC filenames will be set by caller

    # SLC parameters
    prm['SLC_scale'] = 1.0
    prm['Flip_iq'] = 'n'
    prm['deskew'] = 'n'
    prm['offset_video'] = 'n'

    # Image dimensions (make width divisible by 4)
    n_samples = get_int('.//imageInformation/numberOfSamples')
    n_samples = n_samples - (n_samples % 4)
    prm['bytes_per_line'] = n_samples * 4
    prm['good_bytes_per_line'] = prm['bytes_per_line']
    prm['num_rng_bins'] = n_samples

    # Misc parameters
    prm['caltone'] = 0.0
    prm['rm_az_band'] = 0.0
    prm['rm_rng_band'] = 0.2
    prm['rng_spec_wgt'] = 1.0
    prm['scnd_rng_mig'] = 0.0
    prm['fd1'] = 0.0
    prm['az_res'] = 0.0
    prm['fdd1'] = 0.0
    prm['fddd1'] = 0.0

    # Lines per burst and burst count
    lpb = get_int('.//swathTiming/linesPerBurst')
    burst_count_elem = root.find('.//swathTiming/burstList')
    burst_count = int(burst_count_elem.get('count')) if burst_count_elem is not None else 1

    # Parse firstValidSample to find valid line range
    burst = root.find('.//swathTiming/burstList/burst')
    fvs_text = burst.find('firstValidSample').text
    fvs = [int(x) for x in fvs_text.split()]

    # Find first and last valid lines (where flag >= 0)
    k_start = None
    k_end = None
    first_samp = 1
    for j, flag in enumerate(fvs):
        if flag >= 0:
            if k_start is None:
                k_start = j
            k_end = j
            first_samp = max(first_samp, flag)

    prm['first_sample'] = first_samp

    # Number of valid lines (GMTSAR uses line span, not count)
    n_valid = k_end - k_start if k_start is not None else lpb
    # Make divisible by 4
    prm['num_lines'] = n_valid - (n_valid % 4)
    prm['nrows'] = prm['num_lines']
    prm['num_valid_az'] = prm['num_lines']
    prm['num_patches'] = 1
    prm['chirp_ext'] = 0

    # Clock times - parse productFirstLineUtcTime
    first_line_time = get_text('.//imageInformation/productFirstLineUtcTime')
    dt_first = datetime.strptime(first_line_time, '%Y-%m-%dT%H:%M:%S.%f')

    # Year and julian day (GMTSAR uses 0-based julian day)
    year = dt_first.year
    jd = dt_first.timetuple().tm_yday - 1  # 0-based

    # Seconds of day as fractional day
    sec = dt_first.hour * 3600 + dt_first.minute * 60 + dt_first.second + dt_first.microsecond / 1e6
    clock_start_day = jd + sec / 86400.0  # Day fraction

    # GMTSAR format: year*1000 + julian_day + fractional_day
    prm['clock_start'] = clock_start_day

    # Advance start time to account for invalid lines at beginning
    if k_start is not None:
        prm['clock_start'] += k_start / prm['PRF'] / 86400.0

    # SC_clock includes year offset (year * 1000)
    prm['SC_clock_start'] = prm['clock_start'] + year * 1000

    # Stop times
    prm['clock_stop'] = prm['clock_start'] + prm['num_lines'] / prm['PRF'] / 86400.0
    prm['SC_clock_stop'] = prm['clock_stop'] + year * 1000

    return prm


def reference_burst(xml_path: str, tiff_path: str, eof_path: str) -> tuple:
    """
    Extract reference burst PRM and orbit data (mode=0).

    Replaces GMTSAR make_s1a_tops + ext_orb_s1a for mode=0.

    Parameters
    ----------
    xml_path : str
        Path to burst annotation XML file
    tiff_path : str
        Path to burst GeoTIFF file
    eof_path : str
        Path to precise orbit EOF file

    Returns
    -------
    prm_dict : dict
        Dictionary of PRM parameters
    orbit_df : pd.DataFrame
        Orbit state vectors from EOF file

    Notes
    -----
    For mode=0, no SLC data is extracted. Only PRM parameters and orbit data
    are returned. This is typically used for computing geometry and preparing
    for burst alignment.
    """
    import os

    # Extract PRM parameters from XML
    prm_dict = satellite_prm(xml_path, tiff_path)

    # Compute time range for orbit extraction (extend by ~23 minutes on each side)
    t1, t2 = orbit_window(prm_dict['SC_clock_start'], prm_dict['SC_clock_stop'])

    # Extract orbit from EOF file; it must cover the burst (named after its annotation file)
    burst = (os.path.splitext(os.path.basename(xml_path))[0], prm_dict['SC_clock_start'], prm_dict['SC_clock_stop'])
    orbit_df = satellite_orbit(eof_path, t1, t2, burst=burst)

    return prm_dict, orbit_df


def satellite_slc(tiff_path: str) -> "xr.DataArray":
    """
    Read Sentinel-1 burst SLC data as xarray DataArray.

    Parameters
    ----------
    tiff_path : str
        Path to the burst measurement file: `<burst>.nc`, or the legacy `<burst>.tiff`.

    Returns
    -------
    xr.DataArray
        Complex64 SLC data with dimensions (azimuth, range).
        Coordinates are pixel indices.

    Notes
    -----
    Sentinel-1 bursts are complex int16, stored as compressed NetCDF4 (or GeoTIFF when downloaded before).
    This function reads the data as complex64 for processing.

    Examples
    --------
    >>> slc = satellite_slc('burst.nc')
    >>> print(slc.shape)  # (lines, samples)
    >>> print(slc.dtype)  # complex64
    """
    import xarray as xr
    from insardev_toolkit.utils_S1 import read_slc

    data = read_slc(tiff_path)
    n_lines, n_samples = data.shape

    return xr.DataArray(
        data,
        dims=['azimuth', 'range'],
        coords={
            'azimuth': range(n_lines),
            'range': range(n_samples)
        },
        name='slc',
        attrs={'dtype': 'complex64', 'source': tiff_path}
    )


def dc_polynomial(dc_estimate, xml_path: str) -> tuple:
    """
    The Doppler centroid polynomial of a dcEstimate record as the S-1 IPF used it: the geometryDcPolynomial when the
    record is flagged dataDcRmsErrorAboveThreshold (the IPF falls back to the Doppler centroid from geometry, S-1 L1
    Detailed Algorithm Definition, section 5), the dataDcPolynomial otherwise.

    Parameters
    ----------
    dc_estimate : element or dict
        The dcEstimate record: an ElementTree / lxml element, or the dict of xmltodict.
    xml_path : str
        The burst annotation XML file (for the error message).

    Returns
    -------
    tuple
        ([c0, c1, c2], t0): the polynomial coefficients about its slant range time t0.
    """
    def text(name):
        if isinstance(dc_estimate, dict):
            value = dc_estimate.get(name)
            value = value.get('#text') if isinstance(value, dict) else value
        else:
            element = dc_estimate.find(name)
            value = element.text if element is not None else None
        if value is None:
            raise ValueError(f'ERROR: {xml_path}: a dcEstimate record has no {name}. Download the burst again.')
        return value

    name = 'geometryDcPolynomial' if text('dataDcRmsErrorAboveThreshold').strip() == 'true' else 'dataDcPolynomial'
    return [float(x) for x in text(name).split()[:3]], float(text('t0'))


def deramped_burst(xml_path: str, tiff_path: str, eof_path: str) -> tuple:
    """
    Extract deramped burst SLC data without alignment shift or reramp.

    Returns the deramped SLC (azimuth phase ramp removed) and the parameters
    needed to compute the reramp phase analytically at any coordinates.
    This enables merging the alignment and geocoding interpolations into one.

    Parameters
    ----------
    xml_path : str
        Path to burst annotation XML file
    tiff_path : str
        Path to burst GeoTIFF file
    eof_path : str
        Path to precise orbit EOF file

    Returns
    -------
    prm_dict : dict
        Dictionary of PRM parameters
    orbit_df : pd.DataFrame
        Orbit state vectors from EOF file
    slc_data : np.ndarray
        Deramped SLC as complex64 array with shape (n_valid, num_rng_bins).
        Values are raw DN (digital numbers) without any scaling.
    reramp_params : dict
        Parameters for analytical reramp phase computation:
        fka, fnc, ks, dta, dts, ts0, tau0, lpb, k_start, n_valid; fka is the FM rate polynomial about its t0
        (tau0), fnc the Doppler centroid polynomial the IPF used (dc_polynomial()) re-expanded about tau0.
    """
    import numpy as np
    from scipy import constants
    import xml.etree.ElementTree as ET
    from datetime import datetime

    SOL = constants.speed_of_light

    # Get PRM parameters and orbit
    prm_dict, orbit_df = reference_burst(xml_path, tiff_path, eof_path)

    # Read SLC data
    slc_da = satellite_slc(tiff_path)
    data_complex = slc_da.values

    # Get valid line range
    n_lines, n_cols = data_complex.shape
    lpb = n_lines
    width = prm_dict['num_rng_bins']

    # Parse firstValidSample from XML for valid region
    tree = ET.parse(xml_path)
    root = tree.getroot()
    burst = root.find('.//swathTiming/burstList/burst')
    fvs_text = burst.find('firstValidSample').text
    fvs = [int(x) for x in fvs_text.split()]

    # Find valid line range
    k_start = None
    k_end = None
    for j, flag in enumerate(fvs):
        if flag >= 0:
            if k_start is None:
                k_start = j
            k_end = j

    if k_start is None:
        k_start = 0
        k_end = lpb - 1

    n_valid = k_end - k_start
    n_valid = n_valid - (n_valid % 4)

    # Get parameters for deramp/reramp computation
    prf = prm_dict['PRF']
    radar_freq = SOL / prm_dict['radar_wavelength']
    azi_steering_rate = float(root.find('.//productInformation/azimuthSteeringRate').text)
    kpsi = np.pi * azi_steering_rate / 180.0

    dta = 1.0 / prf
    dts = 1.0 / prm_dict['rng_samp_rate']
    ts0 = prm_dict['near_range'] * 2.0 / SOL + 1.0 / prm_dict['rng_samp_rate']

    # Compute burst center time for finding nearest Doppler/FM rate estimates
    t_brst_str = root.find('.//swathTiming/burstList/burst/azimuthTime').text
    dt_brst = datetime.strptime(t_brst_str, '%Y-%m-%dT%H:%M:%S.%f')
    sec_brst = dt_brst.hour * 3600 + dt_brst.minute * 60 + dt_brst.second + dt_brst.microsecond / 1e6
    t_brst = sec_brst + dta * lpb / 2.0

    def parse_aztime(aztime_str):
        # seconds from 00:00 UTC of the burst day, the clock of t_brst
        return _aztime_seconds(datetime.strptime(aztime_str, '%Y-%m-%dT%H:%M:%S.%f'), dt_brst.date())

    # Get Doppler centroid polynomial
    dc_estimates = root.findall('.//dopplerCentroid/dcEstimateList/dcEstimate')
    best_dc = None
    best_dc_dist = float('inf')
    for dc in dc_estimates:
        dc_time = parse_aztime(dc.find('azimuthTime').text)
        dist = abs(dc_time - t_brst)
        if dist < best_dc_dist:
            best_dc_dist = dist
            best_dc = dc
    # the Doppler centroid the IPF used
    fnc, dc_t0 = dc_polynomial(best_dc, xml_path)

    # Get FM rate polynomial
    fm_rates = root.findall('.//generalAnnotation/azimuthFmRateList/azimuthFmRate')
    best_fm = None
    best_fm_dist = float('inf')
    for fm in fm_rates:
        fm_time = parse_aztime(fm.find('azimuthTime').text)
        dist = abs(fm_time - t_brst)
        if dist < best_fm_dist:
            best_fm_dist = dist
            best_fm = fm
    tau0 = float(best_fm.find('t0').text)
    if best_fm.find('azimuthFmRatePolynomial') is not None:
        fka = [float(x) for x in best_fm.find('azimuthFmRatePolynomial').text.split()[:3]]
    else:
        fka = [float(best_fm.find('c0').text),
               float(best_fm.find('c1').text),
               float(best_fm.find('c2').text)]

    # Each polynomial is given about its own t0: the FM rate about its record's t0 (tau0), the Doppler centroid
    # about the dcEstimate t0. The deramp / reramp evaluate both at tau - tau0, so the Doppler polynomial is
    # re-expanded exactly about tau0: fnc(tau - dc_t0) = fnc'(tau - tau0)
    d = tau0 - dc_t0
    fnc = [fnc[0] + fnc[1] * d + fnc[2] * d * d, fnc[1] + 2.0 * fnc[2] * d, fnc[2]]

    # Find velocity at burst center (orbit times from 00:00 UTC of the burst day, the clock of t_brst)
    from .utils_satellite import orbit_seconds
    orbit_time = orbit_seconds(orbit_df, dt_brst.timetuple().tm_yday - 1)
    vx = np.interp(t_brst, orbit_time, orbit_df['vx'].values)
    vy = np.interp(t_brst, orbit_time, orbit_df['vy'].values)
    vz = np.interp(t_brst, orbit_time, orbit_df['vz'].values)
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    ks = 2.0 * vtot * radar_freq * kpsi / SOL

    # Compute deramp phase and apply (chunked, memory-efficient)
    eta = (np.arange(lpb) - lpb / 2.0 + 0.5) * dta
    jj = np.arange(width)
    taus = ts0 + jj * dts - tau0

    ka = fka[0] + fka[1] * taus + fka[2] * taus**2
    kt = ka * ks / (ka - ks)
    fnct_arr = fnc[0] + fnc[1] * taus + fnc[2] * taus**2
    etaref = -fnct_arr / ka + fnc[0] / fka[0]

    n_chunks = 8
    chunk_size = (lpb + n_chunks - 1) // n_chunks
    # Only the valid region is returned - as complex64 (raw DN values, no scaling). Each chunk is deramped as a
    # whole, as before, and its valid lines go straight into the output: no full-burst deramped array and no copy
    # of its valid region
    slc_valid = np.empty((n_valid, width), dtype=np.complex64)

    for chunk_idx in range(n_chunks):
        azi_start = chunk_idx * chunk_size
        azi_end = min((chunk_idx + 1) * chunk_size, lpb)
        if azi_start >= lpb:
            break
        # the chunk's lines inside the valid region
        v0 = max(azi_start, k_start)
        v1 = min(azi_end, k_start + n_valid)
        if v0 >= v1:
            continue
        eta_chunk = eta[azi_start:azi_end, np.newaxis]
        pramp = -np.pi * kt * (eta_chunk - etaref)**2
        pmod = -2.0 * np.pi * fnct_arr * eta_chunk
        phase_chunk = pramp + pmod
        del pramp, pmod
        deramp = np.exp(1j * phase_chunk)
        del phase_chunk
        deramped = (data_complex[azi_start:azi_end, :width] * deramp).astype(np.complex64)
        del deramp
        slc_valid[v0 - k_start:v1 - k_start] = deramped[v0 - azi_start:v1 - azi_start]
        del deramped

    # slc_da too: it held the raw SLC alive until return
    del data_complex, slc_da, eta, jj, taus, ka, kt, fnct_arr, etaref

    reramp_params = {
        'fka': fka,
        'fnc': fnc,
        'ks': ks,
        'dta': dta,
        'dts': dts,
        'ts0': ts0,
        'tau0': tau0,
        'lpb': lpb,
        'k_start': k_start,
        'n_valid': n_valid,
    }

    return prm_dict, orbit_df, slc_valid, reramp_params


# Re-export satellite_llt2rat from utils_satellite for backwards compatibility
from .utils_satellite import satellite_llt2rat


def make_burst(xml_file: str, tiff_file: str, orbit_file: str,
               debug: bool = False) -> tuple:
    """
    Pure Python replacement for GMTSAR make_s1a_tops + ext_orb_s1a (mode=0: PRM and orbit, no SLC).

    Returns PRM object with attached orbit data, without writing any files to disk.
    The SLC is read by deramped_burst().

    Parameters
    ----------
    xml_file : str
        Path to burst annotation XML file
    tiff_file : str
        Path to burst GeoTIFF file
    orbit_file : str
        Path to precise orbit EOF file
    debug : bool, optional
        Enable debug output. Defaults to False.

    Returns
    -------
    tuple
        (prm, orbit_df) where prm is a PRM object

    Notes
    -----
    The returned PRM object has orbit_df attached via prm.orbit_df attribute.
    This allows in-memory processing without file I/O.
    """
    import time

    if debug:
        print(f'DEBUG: make_burst xml_file={xml_file}')
        print(f'DEBUG: make_burst tiff_file={tiff_file}')
        print(f'DEBUG: make_burst orbit_file={orbit_file}')

    start_time = time.perf_counter()

    # PRM and orbit only
    prm_dict, orbit_df = reference_burst(xml_file, tiff_file, orbit_file)

    elapsed = time.perf_counter() - start_time
    if debug:
        print(f'PROFILE: make_burst {elapsed:.3f}s')

    # Create PRM object from dict
    from .PRM import PRM
    prm = PRM().set(**prm_dict)

    # Attach orbit data to PRM object for in-memory processing
    prm.orbit_df = orbit_df

    return prm, orbit_df
