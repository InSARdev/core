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


def satellite_orbit(xml_path: str, t1: float, t2: float) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        Orbit state vectors with columns:
        - iy: year
        - id: julian day
        - isec: seconds of day
        - px, py, pz: ECEF position (meters)
        - vx, vy, vz: ECEF velocity (m/s)
        - clock: seconds from Jan 1 of the year (for interpolation)

        DataFrame.attrs contains metadata:
        - nd: number of records
        - idsec: time step (seconds)

    Examples
    --------
    >>> # Get orbit for a burst (extend time range by ~23 minutes on each side)
    >>> t1 = prm.get('SC_clock_start') - 1400.0/86400.0
    >>> t2 = prm.get('SC_clock_stop') + 1400.0/86400.0
    >>> orbit_df = satellite_orbit(eof_path, t1, t2)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all OSV (Orbit State Vector) elements
    osvs = root.findall('.//OSV')

    records = []
    for osv in osvs:
        # Parse UTC timestamp: "UTC=2015-01-20T22:59:44.000000"
        utc_str = osv.find('UTC').text
        utc_str = utc_str.replace('UTC=', '')
        dt = datetime.strptime(utc_str, '%Y-%m-%dT%H:%M:%S.%f')

        # Convert to year, julian day, seconds
        # Note: GMTSAR uses 0-based julian day (Jan 1 = day 0)
        year = dt.year
        jd = dt.timetuple().tm_yday - 1  # Convert to 0-based
        sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        # Compute year.day_fraction for filtering (GMTSAR format)
        ydf = year * 1000 + jd + sec / 86400.0

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

    if len(records) == 0:
        raise ValueError(f'No orbit data found in time range {t1} to {t2}')

    df = pd.DataFrame(records)

    # Compute clock (seconds from Jan 1) for interpolation
    df['clock'] = (24 * 60 * 60) * df['id'] + df['isec']

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
        xs = _hermite_interp(orbit_time, px, vx, np.array([t_center]))[0]
        ys = _hermite_interp(orbit_time, py, vy, np.array([t_center]))[0]
        zs = _hermite_interp(orbit_time, pz, vz, np.array([t_center]))[0]

        # Get positions 2 seconds apart for velocity
        t_minus = t_center - 2.0
        t_plus = t_center + 2.0

        x1 = _hermite_interp(orbit_time, px, vx, np.array([t_minus]))[0]
        y1 = _hermite_interp(orbit_time, py, vy, np.array([t_minus]))[0]
        z1 = _hermite_interp(orbit_time, pz, vz, np.array([t_minus]))[0]

        x2 = _hermite_interp(orbit_time, px, vx, np.array([t_plus]))[0]
        y2 = _hermite_interp(orbit_time, py, vy, np.array([t_plus]))[0]
        z2 = _hermite_interp(orbit_time, pz, vz, np.array([t_plus]))[0]

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
        ranges = np.zeros(nt)

        for k, time_offset in enumerate(times):
            t_k = t_center + time_offset
            xk = _hermite_interp(orbit_time, px, vx, np.array([t_k]))[0]
            yk = _hermite_interp(orbit_time, py, vy, np.array([t_k]))[0]
            zk = _hermite_interp(orbit_time, pz, vz, np.array([t_k]))[0]
            ranges[k] = np.sqrt((xe - xk)**2 + (ye - yk)**2 + (ze - zk)**2) - ro

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
    # Extract PRM parameters from XML
    prm_dict = satellite_prm(xml_path, tiff_path)

    # Compute time range for orbit extraction (extend by ~23 minutes on each side)
    t1 = prm_dict['SC_clock_start'] - 1400.0 / 86400.0
    t2 = prm_dict['SC_clock_stop'] + 1400.0 / 86400.0

    # Extract orbit from EOF file
    orbit_df = satellite_orbit(eof_path, t1, t2)

    return prm_dict, orbit_df


def satellite_slc(tiff_path: str) -> "xr.DataArray":
    """
    Read Sentinel-1 burst SLC data from GeoTIFF as xarray DataArray.

    Parameters
    ----------
    tiff_path : str
        Path to burst GeoTIFF file

    Returns
    -------
    xr.DataArray
        Complex64 SLC data with dimensions (azimuth, range).
        Coordinates are pixel indices.

    Notes
    -----
    Sentinel-1 burst GeoTIFFs are stored as complex_int16 format.
    This function reads the data as complex64 for processing.

    Examples
    --------
    >>> slc = satellite_slc('burst.tiff')
    >>> print(slc.shape)  # (lines, samples)
    >>> print(slc.dtype)  # complex64
    """
    import xarray as xr
    import rasterio

    with rasterio.open(tiff_path) as src:
        data = src.read(1).astype('complex64')
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
        Deramped SLC as int16 array with shape (n_valid, num_rng_bins*2).
        Format: [re0, im0, re1, im1, ...] matching GMTSAR SLC format.
    reramp_params : dict
        Parameters for analytical reramp phase computation:
        fka, fnc, ks, dta, dts, ts0, tau0, lpb, k_start, n_valid
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
    tau0 = float(root.find('.//azimuthFmRateList/azimuthFmRate/t0').text)

    # Compute burst center time for finding nearest Doppler/FM rate estimates
    t_brst_str = root.find('.//swathTiming/burstList/burst/azimuthTime').text
    dt_brst = datetime.strptime(t_brst_str, '%Y-%m-%dT%H:%M:%S.%f')
    sec_brst = dt_brst.hour * 3600 + dt_brst.minute * 60 + dt_brst.second + dt_brst.microsecond / 1e6
    t_brst = sec_brst + dta * lpb / 2.0

    def parse_aztime(aztime_str):
        dt = datetime.strptime(aztime_str, '%Y-%m-%dT%H:%M:%S.%f')
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

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
    fnc = [float(x) for x in best_dc.find('dataDcPolynomial').text.split()[:3]]

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
    if best_fm.find('azimuthFmRatePolynomial') is not None:
        fka = [float(x) for x in best_fm.find('azimuthFmRatePolynomial').text.split()[:3]]
    else:
        fka = [float(best_fm.find('c0').text),
               float(best_fm.find('c1').text),
               float(best_fm.find('c2').text)]

    # Find velocity at burst center
    orbit_time = orbit_df['clock'].values
    vx = np.interp(t_brst, orbit_time % 86400, orbit_df['vx'].values)
    vy = np.interp(t_brst, orbit_time % 86400, orbit_df['vy'].values)
    vz = np.interp(t_brst, orbit_time % 86400, orbit_df['vz'].values)
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
    slc_deramped = np.empty((lpb, width), dtype=np.complex64)

    for chunk_idx in range(n_chunks):
        azi_start = chunk_idx * chunk_size
        azi_end = min((chunk_idx + 1) * chunk_size, lpb)
        if azi_start >= lpb:
            break
        eta_chunk = eta[azi_start:azi_end, np.newaxis]
        pramp = -np.pi * kt * (eta_chunk - etaref)**2
        pmod = -2.0 * np.pi * fnct_arr * eta_chunk
        phase_chunk = pramp + pmod
        del pramp, pmod
        deramp = np.exp(1j * phase_chunk)
        del phase_chunk
        slc_deramped[azi_start:azi_end] = (data_complex[azi_start:azi_end, :width] * deramp).astype(np.complex64)
        del deramp

    del data_complex, eta, jj, taus, ka, kt, fnct_arr, etaref

    # Extract valid region and return as complex64 (raw DN values)
    slc_valid = slc_deramped[k_start:k_start + n_valid, :width].astype(np.complex64)
    del slc_deramped

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


def repeat_burst(xml_path: str, tiff_path: str, eof_path: str,
                 rshift_grid: "np.ndarray | None" = None,
                 ashift_grid: "np.ndarray | None" = None,
                 grid_step: int = 16) -> tuple:
    """
    Extract repeat burst SLC data with alignment (mode=1).

    Replaces GMTSAR make_s1a_tops for mode=1 with OpenCV Lanczos interpolation.

    Parameters
    ----------
    xml_path : str
        Path to burst annotation XML file
    tiff_path : str
        Path to burst GeoTIFF file
    eof_path : str
        Path to precise orbit EOF file
    rshift_grid : np.ndarray, optional
        Range shift grid for alignment (shape matches output SLC).
        If None, no alignment is applied.
    ashift_grid : np.ndarray, optional
        Azimuth shift grid for alignment (shape matches output SLC).
        If None, no alignment is applied.
    grid_step : int
        Step size of shift grids relative to full resolution SLC.
        Default is 16 (matching GMTSAR convention).

    Returns
    -------
    prm_dict : dict
        Dictionary of PRM parameters
    orbit_df : pd.DataFrame
        Orbit state vectors from EOF file
    slc_data : np.ndarray
        Complex SLC data as int16 array with shape (num_lines, num_rng_bins*2).
        Format: [re0, im0, re1, im1, ...] matching GMTSAR SLC format.

    Notes
    -----
    This implements the TOPS deramp-shift-reramp procedure:
    1. Read raw burst data from GeoTIFF
    2. Apply deramp (remove azimuth phase ramp)
    3. Shift using OpenCV Lanczos interpolation (better than GMTSAR's sinc)
    4. Apply reramp (restore azimuth phase ramp with shifted parameters)

    OpenCV Lanczos interpolation is used instead of GMTSAR's 8-point sinc
    interpolation for better quality and faster processing.

    Theory: TOPS mode introduces an azimuth phase ramp due to antenna steering.
    The phase must be removed before resampling (deramp) and restored after
    (reramp) to maintain phase coherence between bursts.
    """
    import numpy as np
    import cv2
    from scipy import constants

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
    import xml.etree.ElementTree as ET
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

    # Extract valid region (GMTSAR uses line span, not count)
    n_valid = k_end - k_start
    n_valid = n_valid - (n_valid % 4)  # Make divisible by 4

    # If no shift grids provided, just extract without alignment
    if rshift_grid is None or ashift_grid is None:
        # Extract valid region and crop to width
        slc = data_complex[k_start:k_start + n_valid, :width].copy()

        # Convert to int16 format (real, imag interleaved)
        slc_out = np.zeros((n_valid, width * 2), dtype=np.int16)
        slc_out[:, 0::2] = np.clip(slc.real * 2, -32768, 32767).astype(np.int16)
        slc_out[:, 1::2] = np.clip(slc.imag * 2, -32768, 32767).astype(np.int16)

        return prm_dict, orbit_df, slc_out

    # ======================================================================
    # TOPS Deramp-Shift-Reramp procedure
    # ======================================================================

    # Get parameters for deramp computation
    prf = prm_dict['PRF']
    radar_freq = SOL / prm_dict['radar_wavelength']
    azi_steering_rate = float(root.find('.//productInformation/azimuthSteeringRate').text)
    kpsi = np.pi * azi_steering_rate / 180.0

    dta = 1.0 / prf  # azimuth time interval
    dts = 1.0 / prm_dict['rng_samp_rate']  # range time interval
    ts0 = prm_dict['near_range'] * 2.0 / SOL + 1.0 / prm_dict['rng_samp_rate']  # slant range time at near range
    tau0 = float(root.find('.//azimuthFmRateList/azimuthFmRate/t0').text)

    # Compute burst center time for finding nearest Doppler/FM rate estimates
    t_brst_str = root.find('.//swathTiming/burstList/burst/azimuthTime').text
    from datetime import datetime
    dt_brst = datetime.strptime(t_brst_str, '%Y-%m-%dT%H:%M:%S.%f')
    sec_brst = dt_brst.hour * 3600 + dt_brst.minute * 60 + dt_brst.second + dt_brst.microsecond / 1e6
    t_brst = sec_brst + dta * lpb / 2.0  # burst center time (seconds of day)

    # Helper to parse azimuth time string to seconds of day
    def parse_aztime(aztime_str):
        dt = datetime.strptime(aztime_str, '%Y-%m-%dT%H:%M:%S.%f')
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

    # Get Doppler centroid polynomial - find NEAREST to burst center time (like GMTSAR)
    dc_estimates = root.findall('.//dopplerCentroid/dcEstimateList/dcEstimate')
    best_dc = None
    best_dc_dist = float('inf')
    for dc in dc_estimates:
        dc_time = parse_aztime(dc.find('azimuthTime').text)
        dist = abs(dc_time - t_brst)
        if dist < best_dc_dist:
            best_dc_dist = dist
            best_dc = dc
    fnc = [float(x) for x in best_dc.find('dataDcPolynomial').text.split()[:3]]

    # Get FM rate polynomial - find NEAREST to burst center time (like GMTSAR)
    fm_rates = root.findall('.//generalAnnotation/azimuthFmRateList/azimuthFmRate')
    best_fm = None
    best_fm_dist = float('inf')
    for fm in fm_rates:
        fm_time = parse_aztime(fm.find('azimuthTime').text)
        dist = abs(fm_time - t_brst)
        if dist < best_fm_dist:
            best_fm_dist = dist
            best_fm = fm
    if best_fm.find('azimuthFmRatePolynomial') is not None:
        fka = [float(x) for x in best_fm.find('azimuthFmRatePolynomial').text.split()[:3]]
    else:
        fka = [float(best_fm.find('c0').text),
               float(best_fm.find('c1').text),
               float(best_fm.find('c2').text)]

    # Find velocity at burst center by interpolating orbit
    orbit_time = orbit_df['clock'].values
    vx = np.interp(t_brst, orbit_time % 86400, orbit_df['vx'].values)
    vy = np.interp(t_brst, orbit_time % 86400, orbit_df['vy'].values)
    vz = np.interp(t_brst, orbit_time % 86400, orbit_df['vz'].values)
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)

    # Steering rate contribution to Doppler rate
    ks = 2.0 * vtot * radar_freq * kpsi / SOL

    # ======================================================================
    # Compute deramp phase in chunks along azimuth (memory-efficient)
    # Keep float64 for phase precision (TOPS phases can be thousands of radians)
    # ======================================================================
    eta = (np.arange(lpb) - lpb / 2.0 + 0.5) * dta  # azimuth time relative to center
    jj = np.arange(width)
    taus = ts0 + jj * dts - tau0  # slant range time relative to tau0

    # FM rate at each range (1D arrays, reused for all chunks)
    ka = fka[0] + fka[1] * taus + fka[2] * taus**2
    kt = ka * ks / (ka - ks)
    fnct = fnc[0] + fnc[1] * taus + fnc[2] * taus**2
    etaref = -fnct / ka + fnc[0] / fka[0]

    # Process deramp in chunks along azimuth
    n_chunks = 8
    chunk_size = (lpb + n_chunks - 1) // n_chunks
    slc_deramped = np.empty((lpb, width), dtype=np.complex64)

    for chunk_idx in range(n_chunks):
        azi_start = chunk_idx * chunk_size
        azi_end = min((chunk_idx + 1) * chunk_size, lpb)
        if azi_start >= lpb:
            break

        # Compute phase for this chunk (float64 precision)
        eta_chunk = eta[azi_start:azi_end, np.newaxis]
        pramp = -np.pi * kt * (eta_chunk - etaref)**2
        pmod = -2.0 * np.pi * fnct * eta_chunk
        phase_chunk = pramp + pmod
        del pramp, pmod

        # Apply deramp and store as complex64
        deramp = np.exp(1j * phase_chunk)
        del phase_chunk
        slc_deramped[azi_start:azi_end] = (data_complex[azi_start:azi_end, :width] * deramp).astype(np.complex64)
        del deramp

    del data_complex

    # ======================================================================
    # Apply shift using OpenCV Lanczos interpolation
    # ======================================================================
    # Upsample shift grids to full resolution if needed
    if rshift_grid.shape != (lpb, width):
        rshift_full = cv2.resize(rshift_grid.astype(np.float32),
                                  (width, lpb),
                                  interpolation=cv2.INTER_LINEAR)
        ashift_full = cv2.resize(ashift_grid.astype(np.float32),
                                  (width, lpb),
                                  interpolation=cv2.INTER_LINEAR)
    else:
        rshift_full = rshift_grid.astype(np.float32)
        ashift_full = ashift_grid.astype(np.float32)

    # Create coordinate maps for remap (reuse arrays)
    col_idx = np.arange(width, dtype=np.float32)[np.newaxis, :] + np.zeros((lpb, 1), dtype=np.float32)
    row_idx = np.arange(lpb, dtype=np.float32)[:, np.newaxis] + np.zeros((1, width), dtype=np.float32)
    map_x = col_idx + rshift_full
    map_y = row_idx + ashift_full
    del col_idx, row_idx

    # Apply shift using OpenCV remap with Lanczos interpolation
    slc_real_shifted = cv2.remap(slc_deramped.real,
                                  map_x, map_y,
                                  interpolation=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    slc_imag_shifted = cv2.remap(slc_deramped.imag,
                                  map_x, map_y,
                                  interpolation=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    del slc_deramped, map_x, map_y

    # Combine into complex (reuse one of the arrays)
    slc_shifted = (slc_real_shifted + 1j * slc_imag_shifted).astype(np.complex64)
    del slc_real_shifted, slc_imag_shifted

    # ======================================================================
    # Compute reramp phase in chunks along azimuth (memory-efficient)
    # Keep float64 for phase precision
    # ======================================================================
    slc_final = np.empty((lpb, width), dtype=np.complex64)

    for chunk_idx in range(n_chunks):
        azi_start = chunk_idx * chunk_size
        azi_end = min((chunk_idx + 1) * chunk_size, lpb)
        if azi_start >= lpb:
            break

        # Shifted coordinates for this chunk (float64 precision)
        eta_chunk = eta[azi_start:azi_end, np.newaxis]
        eta_shifted = eta_chunk + ashift_full[azi_start:azi_end] * dta
        taus_shifted = ts0 + (jj[np.newaxis, :] + rshift_full[azi_start:azi_end]) * dts - tau0

        # FM rate at shifted range
        ka_shifted = fka[0] + fka[1] * taus_shifted + fka[2] * taus_shifted**2
        kt_shifted = ka_shifted * ks / (ka_shifted - ks)

        # Doppler centroid at shifted range
        fnct_shifted = fnc[0] + fnc[1] * taus_shifted + fnc[2] * taus_shifted**2

        # Reference time at shifted range
        etaref_shifted = -fnct_shifted / ka_shifted + fnc[0] / fka[0]

        # Reramp phase (negative of deramp)
        pramp_reramp = -np.pi * kt_shifted * (eta_shifted - etaref_shifted)**2
        pmod_reramp = -2.0 * np.pi * fnct_shifted * eta_shifted
        phase_chunk = pramp_reramp + pmod_reramp
        del pramp_reramp, pmod_reramp, eta_shifted, taus_shifted, ka_shifted, kt_shifted, fnct_shifted, etaref_shifted

        # Apply reramp (conjugate = negative phase)
        reramp = np.exp(-1j * phase_chunk)
        del phase_chunk
        slc_final[azi_start:azi_end] = (slc_shifted[azi_start:azi_end] * reramp).astype(np.complex64)
        del reramp

    del slc_shifted, ashift_full, rshift_full, eta, jj, taus, ka, kt, fnct, etaref

    # Extract valid region and return as complex64 (raw DN values)
    slc_valid = slc_final[k_start:k_start + n_valid, :width].astype(np.complex64)
    del slc_final

    return prm_dict, orbit_df, slc_valid


# Re-export satellite_llt2rat from utils_satellite for backwards compatibility
from .utils_satellite import satellite_llt2rat


def make_burst(xml_file: str, tiff_file: str, orbit_file: str,
               mode: int = 0,
               rshift_grid: "np.ndarray | None" = None,
               ashift_grid: "np.ndarray | None" = None,
               debug: bool = False) -> tuple:
    """
    Pure Python replacement for GMTSAR make_s1a_tops + ext_orb_s1a.

    Returns PRM object with attached orbit data and optionally SLC data,
    without writing any files to disk.

    Parameters
    ----------
    xml_file : str
        Path to burst annotation XML file
    tiff_file : str
        Path to burst GeoTIFF file
    orbit_file : str
        Path to precise orbit EOF file
    mode : int, optional
        0 - PRM and orbit only (no SLC)
        1 - PRM, orbit, and SLC data
        Defaults to 0.
    rshift_grid : np.ndarray, optional
        Range shift grid for alignment (mode=1 only)
    ashift_grid : np.ndarray, optional
        Azimuth shift grid for alignment (mode=1 only)
    debug : bool, optional
        Enable debug output. Defaults to False.

    Returns
    -------
    tuple
        (prm, orbit_df) for mode=0 where prm is a PRM object
        (prm, orbit_df, slc_data) for mode=1

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

    if mode == 0:
        # Mode 0: PRM and orbit only
        prm_dict, orbit_df = reference_burst(xml_file, tiff_file, orbit_file)
    else:
        # Mode 1: PRM, orbit, and SLC
        prm_dict, orbit_df, slc_data = repeat_burst(
            xml_file, tiff_file, orbit_file,
            rshift_grid=rshift_grid,
            ashift_grid=ashift_grid
        )

    elapsed = time.perf_counter() - start_time
    if debug:
        print(f'PROFILE: make_burst mode={mode} {elapsed:.3f}s')

    # Create PRM object from dict
    from .PRM import PRM
    prm = PRM().set(**prm_dict)

    # Attach orbit data to PRM object for in-memory processing
    prm.orbit_df = orbit_df

    if mode == 0:
        return prm, orbit_df
    else:
        return prm, orbit_df, slc_data
