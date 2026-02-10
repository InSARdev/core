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
NISAR utility functions for RSLC preprocessing.
Pure Python implementations extracting parameters from NISAR HDF5 files.

NISAR uses stripmap mode (simpler than S1 TOPS) with all metadata embedded in HDF5.
No deramp/reramp is needed - direct SLC interpolation works.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import constants

SOL = constants.speed_of_light


def nisar_orbit(h5_path: str, t1: float = None, t2: float = None) -> pd.DataFrame:
    """
    Extract orbit state vectors from NISAR HDF5 file.

    Replaces S1's satellite_orbit() which reads EOF XML.
    NISAR embeds orbit directly in the HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    t1 : float, optional
        Start time as year.day_fraction for filtering
    t2 : float, optional
        End time as year.day_fraction for filtering

    Returns
    -------
    pd.DataFrame
        Orbit state vectors with columns:
        - iy: year
        - id: julian day (0-based)
        - isec: seconds of day
        - px, py, pz: ECEF position (meters)
        - vx, vy, vz: ECEF velocity (m/s)
        - clock: seconds from Jan 1 of the year (for interpolation)

        DataFrame.attrs contains metadata:
        - nd: number of records
        - idsec: time step (seconds)
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        # Get reference date from zeroDopplerStartTime
        ident = f['science/LSAR/identification']
        zdt_start_str = ident['zeroDopplerStartTime'][()].decode() if isinstance(
            ident['zeroDopplerStartTime'][()], bytes) else str(ident['zeroDopplerStartTime'][()])
        # Format: "2025-11-22T02:46:18.000000000"
        ref_date = datetime.strptime(zdt_start_str.split('T')[0], '%Y-%m-%d')

        # Orbit data
        orbit_grp = f['science/LSAR/RSLC/metadata/orbit']
        # Orbit time is seconds since midnight UTC of the acquisition day
        orbit_time = orbit_grp['time'][:]  # shape (N,)
        position = orbit_grp['position'][:]  # shape (N, 3) - ECEF XYZ
        velocity = orbit_grp['velocity'][:]  # shape (N, 3) - ECEF XYZ

    records = []
    for i in range(len(orbit_time)):
        # Convert orbit time (seconds since midnight) to absolute datetime
        sec_of_day = float(orbit_time[i])
        dt = ref_date + timedelta(seconds=sec_of_day)

        # Convert to year, julian day, seconds (GMTSAR format)
        year = dt.year
        jd = dt.timetuple().tm_yday - 1  # 0-based julian day
        sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        # Year.day_fraction for filtering
        ydf = year * 1000 + jd + sec / 86400.0

        # Filter by time range if specified
        if t1 is not None and ydf < t1:
            continue
        if t2 is not None and ydf > t2:
            continue

        records.append({
            'iy': year,
            'id': jd,
            'isec': sec,
            'px': position[i, 0],
            'py': position[i, 1],
            'pz': position[i, 2],
            'vx': velocity[i, 0],
            'vy': velocity[i, 1],
            'vz': velocity[i, 2]
        })

    if len(records) == 0:
        raise ValueError(f'No orbit data found in NISAR file: {h5_path}')

    df = pd.DataFrame(records)

    # Compute clock (seconds from Jan 1) for interpolation
    df['clock'] = (24 * 60 * 60) * df['id'] + df['isec']

    # Store metadata
    if len(df) > 1:
        dt_step = df['clock'].iloc[1] - df['clock'].iloc[0]
    else:
        dt_step = 10.0

    df.attrs = {
        'nd': len(df),
        'iy': int(df['iy'].iloc[0]),
        'id': int(df['id'].iloc[0]),
        'isec': float(df['isec'].iloc[0]),
        'idsec': dt_step
    }

    return df


def nisar_prm(h5_path: str, pol: str = 'HH', frequency: str = 'B') -> dict:
    """
    Extract PRM parameters from NISAR HDF5 file.

    Replaces S1's satellite_prm() which reads annotation XML.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    frequency : str
        Frequency band ('A' or 'B')

    Returns
    -------
    dict
        Dictionary containing all PRM parameters needed for processing.
        Can be used to create a PRM object via PRM().set(**params)

    Notes
    -----
    NISAR uses stripmap mode - no deramp/reramp needed.
    Key parameters:
    - radar_wavelength = c / processedCenterFrequency
    - rng_samp_rate = c / (2 * slantRangeSpacing)
    - PRF = 1 / zeroDopplerTimeSpacing
    - near_range = slantRange[0]
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        # Identification
        ident = f['science/LSAR/identification']
        track = int(ident['trackNumber'][()])
        frame = int(ident['frameNumber'][()])
        orbit_dir = ident['orbitPassDirection'][()].decode() if isinstance(
            ident['orbitPassDirection'][()], bytes) else str(ident['orbitPassDirection'][()])
        look_dir = ident['lookDirection'][()].decode() if isinstance(
            ident['lookDirection'][()], bytes) else str(ident['lookDirection'][()])

        # Swath parameters
        freq_path = f'science/LSAR/RSLC/swaths/frequency{frequency}'
        swath = f[freq_path]

        # Radar frequency and wavelength
        radar_freq = float(swath['processedCenterFrequency'][()])
        wavelength = SOL / radar_freq

        # Range sampling
        slant_range_spacing = float(swath['slantRangeSpacing'][()])
        rng_samp_rate = SOL / (2.0 * slant_range_spacing)

        # Slant range array
        slant_range = swath['slantRange'][:]
        near_range = float(slant_range[0])
        num_rng_bins = len(slant_range)

        # Azimuth timing
        zdt = f['science/LSAR/RSLC/swaths/zeroDopplerTime'][:]
        zdt_spacing = float(f['science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing'][()])
        prf = 1.0 / zdt_spacing
        num_lines = len(zdt)

        # Convert zeroDopplerStartTime to clock_start (year.day_fraction)
        zdt_start_str = ident['zeroDopplerStartTime'][()].decode() if isinstance(
            ident['zeroDopplerStartTime'][()], bytes) else str(ident['zeroDopplerStartTime'][()])
        # Format: "2025-11-22T02:46:18.000000000"
        zdt_start = datetime.strptime(zdt_start_str.split('.')[0], '%Y-%m-%dT%H:%M:%S')
        # Add fractional seconds
        frac_sec = float('0.' + zdt_start_str.split('.')[1][:6]) if '.' in zdt_start_str else 0.0

        year = zdt_start.year
        jd = zdt_start.timetuple().tm_yday - 1  # 0-based
        sec = zdt_start.hour * 3600 + zdt_start.minute * 60 + zdt_start.second + frac_sec
        clock_start = jd + sec / 86400.0

        # End time
        duration = num_lines / prf
        clock_stop = clock_start + duration / 86400.0

        # SLC dimensions
        slc = swath[pol]
        slc_shape = slc.shape  # (azimuth, range)

        # PRF from nominal value (for reference)
        nominal_prf = float(swath['nominalAcquisitionPRF'][()])

    # Build PRM dictionary (matching GMTSAR format)
    prm = {
        # File info
        'input_file': h5_path,

        # Processing parameters
        'first_line': 1,
        'st_rng_bin': 1,
        'nlooks': 1,  # SLC is single-look
        'rshift': 0,
        'ashift': 0,
        'sub_int_r': 0.0,
        'sub_int_a': 0.0,
        'stretch_r': 0.0,
        'stretch_a': 0.0,
        'a_stretch_r': 0.0,
        'a_stretch_a': 0.0,
        'dtype': 'a',  # complex

        # Sampling
        'rng_samp_rate': rng_samp_rate,
        'PRF': prf,

        # Satellite identity (14 = NSR/NISAR per GMTSAR)
        'SC_identity': 14,

        # Wavelength
        'radar_wavelength': wavelength,

        # Timing
        'SC_clock_start': year * 1000 + clock_start,
        'SC_clock_stop': year * 1000 + clock_stop,
        'clock_start': clock_start,
        'clock_stop': clock_stop,

        # Range
        'near_range': near_range,
        'num_rng_bins': num_rng_bins,
        'bytes_per_line': num_rng_bins * 8,  # complex64 = 8 bytes

        # Azimuth
        'nrows': num_lines,
        'num_lines': num_lines,
        'num_valid_az': num_lines,
        'num_patches': 1,  # Stripmap mode

        # Orbit direction
        'orbdir': 'A' if orbit_dir.lower().startswith('a') else 'D',

        # Look direction (NISAR is left-looking)
        'lookdir': 'L' if look_dir.lower().startswith('l') else 'R',

        # NISAR-specific
        'frequency': frequency,
        'polarization': pol,
        'track': track,
        'frame': frame,

        # Chirp parameters (approximate for NISAR L-band)
        'chirp_slope': 0.0,  # Not directly available, compute if needed
        'pulse_dur': 0.0,
        'chirp_ext': 0,  # No chirp extension for NISAR stripmap

        # Doppler (NISAR provides dopplerCentroid if needed)
        'fd1': 0.0,
        'fdd1': 0.0,
        'fddd1': 0.0,

        # WGS84 ellipsoid parameters (needed for SAT_llt2rat)
        'equatorial_radius': 6378137.0,  # WGS84 semi-major axis
        'polar_radius': 6356752.31424518,  # WGS84 semi-minor axis
    }

    return prm


def nisar_slc(h5_path: str, pol: str = 'HH', frequency: str = 'B',
              row_slice: slice = None, col_slice: slice = None) -> np.ndarray:
    """
    Read NISAR SLC data from HDF5.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    frequency : str
        Frequency band ('A' or 'B')
    row_slice : slice, optional
        Azimuth slice for partial read
    col_slice : slice, optional
        Range slice for partial read

    Returns
    -------
    np.ndarray
        Complex64 SLC data array (azimuth, range)

    Notes
    -----
    NISAR uses stripmap mode - NO deramp needed.
    Data is returned as-is from the HDF5 file.
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        slc_path = f'science/LSAR/RSLC/swaths/frequency{frequency}/{pol}'
        slc_ds = f[slc_path]

        if row_slice is None:
            row_slice = slice(None)
        if col_slice is None:
            col_slice = slice(None)

        slc = slc_ds[row_slice, col_slice]

    return slc.astype(np.complex64)


def nisar_calibration(h5_path: str, pol: str = 'HH', frequency: str = 'B') -> dict:
    """
    Read NISAR calibration LUTs from HDF5.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    frequency : str
        Frequency band ('A' or 'B')

    Returns
    -------
    dict
        Calibration data:
        - scaleFactor: per-polarization calibration constant
        - beta0: 2D LUT for beta0 (usually 1.0 everywhere)
        - sigma0: 2D LUT for sigma0 conversion
        - gamma0: 2D LUT for gamma0 conversion
        - slantRange: 1D coordinate for LUT range dimension
        - zeroDopplerTime: 1D coordinate for LUT azimuth dimension
        - nesz: noise equivalent sigma-zero (if available)

    Notes
    -----
    Calibration formulas:
    - beta0 = |SLC|² / scaleFactor²
    - sigma0 = beta0 * sigma0_LUT
    - gamma0 = beta0 * gamma0_LUT
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        # Per-polarization scale factor
        sf_path = f'science/LSAR/RSLC/metadata/calibrationInformation/frequency{frequency}/{pol}/scaleFactor'
        scale_factor = float(f[sf_path][()])

        # Geometry calibration LUTs
        geom = f['science/LSAR/RSLC/metadata/calibrationInformation/geometry']
        beta0 = geom['beta0'][:]
        sigma0 = geom['sigma0'][:]
        gamma0 = geom['gamma0'][:]
        lut_slant_range = geom['slantRange'][:]
        lut_zdt = geom['zeroDopplerTime'][:]

        # Noise equivalent backscatter (NESZ)
        nesz_path = f'science/LSAR/RSLC/metadata/calibrationInformation/frequency{frequency}/noiseEquivalentBackscatter/{pol}'
        if nesz_path in f:
            nesz = f[nesz_path][:]
            nesz_sr = f[f'science/LSAR/RSLC/metadata/calibrationInformation/frequency{frequency}/noiseEquivalentBackscatter/slantRange'][:]
            nesz_zdt = f[f'science/LSAR/RSLC/metadata/calibrationInformation/frequency{frequency}/noiseEquivalentBackscatter/zeroDopplerTime'][:]
        else:
            nesz = None
            nesz_sr = None
            nesz_zdt = None

    return {
        'scaleFactor': scale_factor,
        'beta0': beta0,
        'sigma0': sigma0,
        'gamma0': gamma0,
        'slantRange': lut_slant_range,
        'zeroDopplerTime': lut_zdt,
        'nesz': nesz,
        'nesz_slantRange': nesz_sr,
        'nesz_zeroDopplerTime': nesz_zdt,
    }


def nisar_burst(h5_path: str, pol: str = 'HH', frequency: str = 'B') -> tuple:
    """
    Main entry point - extract PRM, orbit, and prepare for geocoding.

    Equivalent to S1's deramped_burst() but simpler (no reramp needed).

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    frequency : str
        Frequency band ('A' or 'B')

    Returns
    -------
    tuple
        (prm_dict, orbit_df, None)
        - prm_dict: PRM parameters
        - orbit_df: Orbit state vectors
        - None: placeholder for reramp_params (not needed for NISAR)
    """
    # Extract PRM parameters
    prm = nisar_prm(h5_path, pol, frequency)

    # Extract orbit with time padding (~23 minutes on each side, like S1)
    t1 = prm['SC_clock_start'] - 1400.0 / 86400.0
    t2 = prm['SC_clock_stop'] + 1400.0 / 86400.0
    orbit_df = nisar_orbit(h5_path, t1, t2)

    # No reramp params for NISAR (stripmap mode)
    reramp_params = None

    return prm, orbit_df, reramp_params


def nisar_geolocation_grid(h5_path: str) -> dict:
    """
    Read NISAR geolocation grid for quick geocoding reference.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file

    Returns
    -------
    dict
        Geolocation grid data:
        - coordinateX: ECEF X coordinates
        - coordinateY: ECEF Y coordinates
        - coordinateZ: ECEF Z coordinates
        - incidenceAngle: local incidence angle
        - losUnitVectorX/Y/Z: line-of-sight unit vectors
        - slantRange: slant range coordinates
        - zeroDopplerTime: azimuth time coordinates
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        geoloc = f['science/LSAR/RSLC/metadata/geolocationGrid']

        result = {}
        for key in ['coordinateX', 'coordinateY', 'coordinateZ',
                    'incidenceAngle', 'losUnitVectorX', 'losUnitVectorY', 'losUnitVectorZ',
                    'slantRange', 'zeroDopplerTime']:
            if key in geoloc:
                result[key] = geoloc[key][:]

    return result


def nisar_get_frequencies(h5_path: str) -> list:
    """
    Get available frequency bands in NISAR HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file

    Returns
    -------
    list
        Available frequencies ('A', 'B', or both)
    """
    import h5py

    frequencies = []
    with h5py.File(h5_path, 'r') as f:
        swaths = f['science/LSAR/RSLC/swaths']
        if 'frequencyA' in swaths:
            frequencies.append('A')
        if 'frequencyB' in swaths:
            frequencies.append('B')

    return frequencies


def nisar_get_polarizations(h5_path: str, frequency: str = 'B') -> list:
    """
    Get available polarizations for a frequency band.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    frequency : str
        Frequency band ('A' or 'B')

    Returns
    -------
    list
        Available polarizations (e.g., ['HH', 'HV'])
    """
    import h5py

    pols = []
    with h5py.File(h5_path, 'r') as f:
        freq_path = f'science/LSAR/RSLC/swaths/frequency{frequency}'
        if freq_path in f:
            swath = f[freq_path]
            for pol in ['HH', 'HV', 'VH', 'VV']:
                if pol in swath:
                    pols.append(pol)

    return pols
