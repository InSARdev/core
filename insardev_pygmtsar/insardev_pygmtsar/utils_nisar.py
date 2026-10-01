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


def _nisar_epoch(ds):
    """The UTC epoch of a NISAR time dataset from its units ("seconds since 2025-10-25T00:00:00"), or None."""
    units = ds.attrs.get('units')
    if units is None:
        return None
    units = units.decode() if isinstance(units, bytes) else str(units)
    if 'since' not in units:
        return None
    try:
        return datetime.fromisoformat(units.split('since', 1)[1].strip().replace(' ', 'T')[:19])
    except ValueError:
        return None


def nisar_start_time(f) -> tuple:
    """
    Zero-Doppler time of the first SLC line of an open NISAR RSLC file.

    It is swaths/zeroDopplerTime[0] on the epoch of its units. identification/zeroDopplerStartTime is not used:
    it is the uncropped frame's start in bbox crops downloaded by insardev_toolkit before it set the crop's own.

    Returns
    -------
    tuple
        (day, sec): the UTC day of the first line (datetime at midnight) and its seconds of that day (float64).
    """
    zdt = f['science/LSAR/RSLC/swaths/zeroDopplerTime']
    epoch = _nisar_epoch(zdt)
    if epoch is None:
        raise ValueError(f"{f.filename}: swaths/zeroDopplerTime has no 'seconds since <UTC epoch>' units, so the "
                         f"time of the first line is unknown")
    day = epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    sec = (epoch - day).total_seconds() + float(zdt[0])
    days = int(np.floor(sec / 86400.0))
    return day + timedelta(days=days), sec - days * 86400.0


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

    Raises
    ------
    ValueError
        The orbit cannot be used (utils_satellite.orbit_defect): its state vectors do not cover the scene's
        first to last line, have a gap, or one is not finite.

    Returns
    -------
    pd.DataFrame
        Orbit state vectors with columns:
        - iy: year
        - id: julian day (0-based)
        - isec: seconds of day
        - px, py, pz: ECEF position (meters)
        - vx, vy, vz: ECEF velocity (m/s)
        - clock: seconds from Jan 1 of the scene year (for interpolation), continuous through
          midnight and Jan 1 (a vector of the next year counts on from day 365/366, one of the
          previous year from day -1), like t1/t2 and PRM clock_start

        DataFrame.attrs contains metadata:
        - nd: number of records
        - idsec: time step (seconds)
    """
    import h5py

    with h5py.File(h5_path, 'r') as f:
        # Reference date: the day of the first SLC line
        ref_date, _ = nisar_start_time(f)

        # Orbit data
        orbit_grp = f['science/LSAR/RSLC/metadata/orbit']
        # Orbit time is seconds since the epoch of its units, midnight UTC of the frame's acquisition day
        orbit_time = orbit_grp['time'][:]  # shape (N,)
        epoch = _nisar_epoch(orbit_grp['time'])
        if epoch is None:
            print(f"WARNING: {h5_path}: metadata/orbit/time has no 'seconds since <UTC epoch>' units; its epoch is "
                  f"taken as midnight UTC of the first line's day ({ref_date.date()}).")
            epoch = ref_date
        position = orbit_grp['position'][:]  # shape (N, 3) - ECEF XYZ
        velocity = orbit_grp['velocity'][:]  # shape (N, 3) - ECEF XYZ
        # the zero-Doppler times of the scene's first and last lines: the orbit must cover them
        zdt = f['science/LSAR/RSLC/swaths/zeroDopplerTime']
        scene = [_nisar_epoch(zdt) + timedelta(seconds=float(zdt[i])) for i in (0, -1)]

    def unusable(defect):
        return ValueError(f'ERROR: Orbit of scene file {h5_path} {defect} the scene ({ref_date.date()}). '
                          f'Delete the scene file and download it again.')

    # t1/t2 count the days of the scene year (year * 1000 + day), running below day 0 or past Dec 31 when the
    # record crosses Jan 1: the vectors of the other year are put on that count too, and so is 'clock'
    year0 = ref_date.year

    records = []
    for i in range(len(orbit_time)):
        # Convert orbit time (seconds since the epoch) to absolute datetime
        sec_of_day = float(orbit_time[i])
        dt = epoch + timedelta(seconds=sec_of_day)

        # Convert to year, julian day, seconds (GMTSAR format)
        year = dt.year
        jd = dt.timetuple().tm_yday - 1  # 0-based julian day
        sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        # Year.day_fraction for filtering, on the day count of t1/t2
        if year == year0:
            ydf = year * 1000 + jd + sec / 86400.0
        else:
            ydf = year0 * 1000 + (jd + (datetime(year, 1, 1) - datetime(year0, 1, 1)).days) + sec / 86400.0

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
        raise unusable('does not cover')

    df = pd.DataFrame(records)

    # Compute clock (seconds from Jan 1 of the scene year, continuous through Jan 1) for interpolation
    day = df['id']
    if (df['iy'] != year0).any():
        day = day + [(datetime(int(y), 1, 1) - datetime(year0, 1, 1)).days for y in df['iy']]
    df['clock'] = (24 * 60 * 60) * day + df['isec']

    # an orbit that cannot be used raises: the state vectors must cover the scene, without a gap, and be finite
    from .utils_satellite import orbit_defect
    defect = orbit_defect(df, *[(t - datetime(year0, 1, 1)).total_seconds() for t in scene])
    if defect is not None:
        raise unusable(defect)

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

        # The first line's time to clock_start (year.day_fraction)
        day, sec = nisar_start_time(f)
        year = day.year
        jd = day.timetuple().tm_yday - 1  # 0-based
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


def nisar_doppler_centroid(h5_path: str, frequency: str, azi, rng):
    """
    Doppler centroid of an RSLC band at swath positions, in cycles per line.

    The azimuth spectrum of the RSLC is centred on the processor's Doppler centroid, stored in
    science/LSAR/RSLC/metadata/processingInformation/parameters/frequency<X>/dopplerCentroid [Hz] over the
    zeroDopplerTime x slantRange axes of the same group. It is the absolute value, ambiguity included (about
    +950 Hz at a 1520 Hz line rate, 0.61-0.66 cycles per line), not the alias f - PRF: the band B minus band A
    difference of the data's own spectral centroid follows the carrier ratio at the absolute value.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    frequency : str
        Frequency band ('A' or 'B')
    azi, rng : float or array-like
        0-based pixel-centre line and range bin of the swath, the transform's azi and rng (zeroDopplerTime[i] and
        slantRange[j] are the centres of line i and bin j). Arrays (broadcast together) give one value per
        position from one read of the LUT.

    Returns
    -------
    float or numpy.ndarray
        f_dc times zeroDopplerTimeSpacing at (azi, rng): the LUT interpolated bilinearly, held at its edges. A float
        for scalar positions, else an array of their broadcast shape.
    """
    import h5py

    root = 'science/LSAR/RSLC/'
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'{root}metadata/processingInformation/parameters/frequency{frequency}']
        values = grp['dopplerCentroid'][:].astype(np.float64)
        # the band's own axes, else the shared ones of processingInformation/parameters
        base = grp if 'zeroDopplerTime' in grp else f[f'{root}metadata/processingInformation/parameters']
        lut_time = base['zeroDopplerTime'][:].astype(np.float64)
        lut_range = base['slantRange'][:].astype(np.float64)
        zdt0 = float(f[f'{root}swaths/zeroDopplerTime'][0])
        dt = float(f[f'{root}swaths/zeroDopplerTimeSpacing'][()])
        sw = f[f'{root}swaths/frequency{frequency}']
        sr0 = float(sw['slantRange'][0])
        dr = float(sw['slantRangeSpacing'][()])
    assert values.shape == (len(lut_time), len(lut_range)), \
        f'dopplerCentroid {values.shape} does not match its axes ({len(lut_time)}, {len(lut_range)})'
    # increasing axes for the interpolation
    if len(lut_time) > 1 and lut_time[-1] < lut_time[0]:
        lut_time, values = lut_time[::-1], values[::-1]
    if len(lut_range) > 1 and lut_range[-1] < lut_range[0]:
        lut_range, values = lut_range[::-1], values[:, ::-1]
    if np.any(np.diff(lut_time) <= 0) or np.any(np.diff(lut_range) <= 0):
        raise ValueError(f'{h5_path}: dopplerCentroid axes of frequency{frequency} are not monotonic')
    # the LUT axes as fractional line and bin indices of the swath
    lines = (lut_time - zdt0) / dt
    bins = (lut_range - sr0) / dr
    if np.ndim(azi) == 0 and np.ndim(rng) == 0:
        return float(np.interp(azi, lines, [np.interp(rng, bins, row) for row in values * dt]))
    # per position the same two interpolations as above: in range on every LUT line, then in azimuth
    azi, rng = np.broadcast_arrays(np.asarray(azi, dtype=np.float64), np.asarray(rng, dtype=np.float64))
    on_lines = np.array([np.interp(rng.ravel(), bins, row) for row in values * dt])
    out = np.array([np.interp(a, lines, on_lines[:, i]) for i, a in enumerate(azi.ravel())], dtype=np.float64)
    return out.reshape(azi.shape)


# the zero-Doppler time span of a full NISAR frame: a nominal constant (a full frame, e.g. track 172 frame 8, has
# 53,200 lines over 34.999 s); only the telling of a full frame from a crop uses it
NISAR_FRAME_SECONDS = 35.0


def nisar_frame_fraction(h5_path: str, frequency: str) -> tuple:
    """
    The extent of an RSLC file against its full frame, per axis: (azimuth, range).

    Azimuth: the swaths/zeroDopplerTime span over NISAR_FRAME_SECONDS (35 s, the nominal full frame). Range: the
    swaths/frequency<X>/slantRange span over the full frame's slant-range span, which every RSLC carries exactly in
    metadata/calibrationInformation/frequency<X>/noiseEquivalentBackscatter/slantRange (a crop keeps that axis whole;
    it also follows modes with a narrower swath). A full frame gives about 1 on both axes, a crop less on the
    cropped axis.

    Parameters
    ----------
    h5_path : str
        Path to NISAR RSLC HDF5 file
    frequency : str
        Frequency band ('A' or 'B')

    Returns
    -------
    tuple
        (azimuth fraction, range fraction), floats.

    Raises
    ------
    ValueError
        When the file has no full-frame slant-range axis: a full frame cannot be told from a crop without it.
    """
    import h5py

    root = 'science/LSAR/RSLC/'
    full = f'{root}metadata/calibrationInformation/frequency{frequency}/noiseEquivalentBackscatter/slantRange'
    with h5py.File(h5_path, 'r') as f:
        zdt = f[f'{root}swaths/zeroDopplerTime']
        t0, t1 = float(zdt[0]), float(zdt[-1])
        sr = f[f'{root}swaths/frequency{frequency}/slantRange']
        r0, r1 = float(sr[0]), float(sr[-1])
        if full not in f:
            raise ValueError(f'{h5_path}: {full} is missing, so the full frame\'s slant-range extent is unknown and '
                             f'a full frame cannot be told from a crop')
        sr_full = f[full]
        f0, f1 = float(sr_full[0]), float(sr_full[-1])
    if not (np.isfinite(f1 - f0) and abs(f1 - f0) > 0):
        raise ValueError(f'{h5_path}: {full} spans {f0}..{f1} m, not a valid full-frame slant-range extent')
    return abs(t1 - t0) / NISAR_FRAME_SECONDS, abs(r1 - r0) / abs(f1 - f0)


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
