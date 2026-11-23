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
Envisat-format SLC products: Envisat ASAR Image Mode (.N1, collection ASA_IMS_1P) and ERS-1 / ERS-2 (.E1 / .E2, collection
SAR_IMS_1P, the same format). The product header, the line window of an AOI and the crop file.

The layout (ESA PO-RS-MDA-GS-2009; the same in every delivered product measured,
session/patches/esa_alos1/reports/U2a_report_r3.md Q4 and U2bO_report.md 5.2):

- MPH: MPH_SIZE (1247) ASCII bytes of ``KEY=value`` lines (TOT_SIZE, SPH_SIZE, NUM_DSD, DSD_SIZE, NUM_DATA_SETS, ...).
- SPH: SPH_SIZE bytes: ASCII ``KEY=value`` lines (FIRST_LINE_TIME, LAST_LINE_TIME, LINE_LENGTH, ...), then NUM_DSD data
  set descriptors (DSD) of DSD_SIZE (280) bytes: DS_NAME, DS_TYPE (A annotation, M measurement, R reference), FILENAME,
  DS_OFFSET, DS_SIZE, NUM_DSR, DSR_SIZE. MPH_SIZE + SPH_SIZE equals the DS_OFFSET of the first DSD.
- The annotation data sets (ADS), contiguous: MDS1 SQ ADS, MAIN PROCESSING PARAMS ADS (one record: the first and last
  zero-Doppler times, the line time interval, the line and sample counts, the range sampling rate, the radar frequency
  and 5 orbit state vectors), DOP CENTROID COEFFS ADS, CHIRP PARAMS ADS, GEOLOCATION GRID ADS (GEO_RECORD bytes per
  record: the tie points of the first and last line of a block of lines, TIE_POINTS across the swath).
- MDS1, last: one record per line: the zero-Doppler time (MJD 2000), a quality flag byte, the line number (uint32, from 1),
  then I and Q as big-endian int16 per sample (RECORD_HEADER + 4 x samples bytes).

MJD 2000 times are big-endian int32 days since 2000-01-01, uint32 seconds of the day and uint32 microseconds; all numbers
are big-endian. The field offsets of the MAIN PROCESSING PARAMS and GEOLOCATION GRID records are those of the EPR data
dictionary (ASAR_Main_ADSR, ASAR_Geo_Grid_ADSR) bundled in GMTSAR (preproc/ENVI_preproc/ENVI_SLC_decode/epr_api-2.3).

The crop file (``crop``; ``crop_header`` and ``crop_orbit`` give its parts) is the product's lines i..j with the header rewritten for them and the precise orbit embedded,
the layout U2a Q4 measured (option a: the GEOLOCATION GRID kept unchanged, its records refer to the lines of the full
product):

- MPH: TOT_SIZE, SPH_SIZE (+ DSD_SIZE), NUM_DSD (+ 1) and NUM_DATA_SETS (+ 1) rewritten; PRODUCT and all other fields kept.
- SPH: FIRST_LINE_TIME and LAST_LINE_TIME rewritten to the times of records i and j; the corner coordinates
  (FIRST_/LAST_ NEAR/MID/FAR _LAT/_LONG) kept: they describe the full product.
- DSDs: every non-zero DS_OFFSET shifted by DSD_SIZE (the 19th DSD is inserted after the last one, ahead of the ADS);
  MDS1 DS_SIZE and NUM_DSR for the lines kept; then the DSD of the precise-orbit dataset
  (``utils_esa_orbit.orbit_dsd``, DS_NAME ``PRECISE ORBIT STATE VECTORS``).
- ADS unchanged, except the MAIN PROCESSING PARAMS record: the first and last zero-Doppler times (the bytes of records i
  and j), num_output_lines, and the 5 state vectors replaced by the precise orbit at their epochs
  (``utils_esa_orbit.replace_state_vectors``).
- MDS1: the records i..j as read from the product, unchanged.
- The precise-orbit dataset (``utils_esa_orbit.orbit_ads``) after MDS1.

The functions are pure: no network and no file access.

Facts for a future switch of reader (2026-11-23; the reports are in session/patches/esa_alos1/reports):
- The crop layout above (U2a_report_r3.md Q4, option a): the header rewrite, MDS1 unchanged, and the 19th DSD 'PRECISE ORBIT
  STATE VECTORS' with its dataset after MDS1.
- GMTSAR's envi_slc_decode (a local clang build) read our crops correctly: Envisat, ERS-1 and ERS-2 crops of 2,048 lines (the
  Envisat SLC equal to the original's rows 14336..16383, the LED identical, the ERS SLCs equal to the records decoded
  independently; the 19th DSD breaks neither envi_slc_decode nor EPR; U2bD1_verify.md 5, U2bD1_round2_report.md), and a live
  Envisat crop of 2,048 lines whose SLC equals the crop's records (U2b-D2 round 1, m5_gmtsar_live.json). No test depends on
  GMTSAR.
- The products this module reads are those ESA serves for archive record IDs (ASAR/7.00); the 2023 versions (ASAR/6.03, 6.05)
  have the same layout: the two versions of one scene have identical line times, range windows and sample counts
  (U2a_report_r3.md Q3).
"""
import re

import numpy as np

# the main product header size, the DSD size and the MDS1 record header (time 12, quality 1, line number 4)
MPH_SIZE = 1247
DSD_SIZE = 280
RECORD_HEADER = 17
# the data set names the functions read
MAIN_PARAMS = 'MAIN PROCESSING PARAMS ADS'
GEOLOCATION = 'GEOLOCATION GRID ADS'
MDS1 = 'MDS1'
# a GEOLOCATION GRID record and its tie points per line
GEO_RECORD = 521
TIE_POINTS = 11
# the offsets of the MAIN PROCESSING PARAMS fields read (EPR ASAR_Main_ADSR): MJD times, float32, uint32
MAIN_FIRST_TIME = 0
MAIN_LAST_TIME = 13
MAIN_LINE_TIME_INTERVAL = 52
MAIN_LINES = 56
MAIN_SAMPLES = 60
MAIN_RANGE_SAMPLING_RATE = 983
MAIN_RADAR_FREQUENCY = 987
# the line window of an AOI is rounded outward to multiples of LINE_BLOCK lines of the product (as ALOS-2's 512 lines:
# a margin for the resampling and the shifts between dates)
LINE_BLOCK = 512
# the speed of light
_C = 299792458.0
_MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
_FIELD = re.compile(r'^([A-Z0-9_]+)=(.*)$', re.M)


def _text(data, what):
    """
    ASCII text of header bytes.

    Parameters
    ----------
    data : bytes
        The bytes.
    what : str
        The part named in the error message.

    Returns
    -------
    str
        The text.

    Raises
    ------
    ValueError
        Bytes that are not ASCII.

    Examples
    --------
    >>> _text(b'NUM_DSD=+0000000018\\n', 'MPH')
    'NUM_DSD=+0000000018\\n'
    """
    try:
        return bytes(data).decode('ascii')
    except UnicodeDecodeError:
        raise ValueError(f'Envisat header: the {what} is not ASCII') from None


def _value(raw):
    """
    The value of an MPH, SPH or DSD field: a quoted string without its quotes and trailing blanks, a number without its
    unit (``<bytes>``, ``<s>``, ...) as int or float, or the raw text.

    Parameters
    ----------
    raw : str
        The text after ``KEY=``.

    Returns
    -------
    str, int or float
        The value.

    Examples
    --------
    >>> _value('+0000006099<bytes>')
    6099
    >>> _value('"ASCENDING "')
    'ASCENDING'
    """
    raw = raw.rstrip()
    if raw.startswith('"'):
        return raw.strip('"').rstrip()
    number = re.sub(r'<[^>]*>$', '', raw)
    if re.fullmatch(r'[+-]?\d+', number):
        return int(number)
    try:
        return float(number)
    except ValueError:
        return raw


def _fields(text):
    """
    The ``KEY=value`` fields of a header text, values parsed by ``_value``.

    Parameters
    ----------
    text : str
        The MPH, the ASCII SPH or a DSD.

    Returns
    -------
    dict
        Field name to value.

    Examples
    --------
    >>> _fields('NUM_DSD=+0000000018\\nSWATH="IS2"\\n')
    {'NUM_DSD': 18, 'SWATH': 'IS2'}
    """
    return {key: _value(raw) for key, raw in _FIELD.findall(text)}


def _number(fields, key, what):
    """
    An integer field of a header part.

    Parameters
    ----------
    fields : dict
        The parsed fields.
    key : str
        The field name.
    what : str
        The part named in the error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    ValueError
        A missing or non-integer field.

    Examples
    --------
    >>> _number({'NUM_DSD': 18}, 'NUM_DSD', 'MPH')
    18
    """
    value = fields.get(key)
    if not isinstance(value, int):
        raise ValueError(f'Envisat header: the {what} has no integer {key} ({value!r})')
    return value


def _sph_time(text):
    """
    A UTC time of the ``DD-MON-YYYY hh:mm:ss.ffffff`` header format.

    Parameters
    ----------
    text : str
        The time text.

    Returns
    -------
    numpy.datetime64
        datetime64[ns].

    Raises
    ------
    ValueError
        Text of another format.

    Examples
    --------
    >>> _sph_time('16-JAN-2008 20:47:27.343088')
    np.datetime64('2008-01-16T20:47:27.343088000')
    """
    match = re.fullmatch(r'(\d{2})-([A-Z]{3})-(\d{4}) (\d{2}:\d{2}:\d{2}\.\d{6})', str(text))
    if match is None or match.group(2) not in _MONTHS:
        raise ValueError(f'Envisat header: time {text!r} is not DD-MON-YYYY hh:mm:ss.ffffff')
    day, month, year, clock = match.groups()
    return np.datetime64(f'{year}-{_MONTHS.index(month) + 1:02d}-{day}T{clock}', 'ns')


def _sph_text(time):
    """
    The ``DD-MON-YYYY hh:mm:ss.ffffff`` header text of a UTC time, to the microsecond (truncated).

    Parameters
    ----------
    time : numpy.datetime64
        The time.

    Returns
    -------
    str
        27 characters.

    Examples
    --------
    >>> _sph_text(np.datetime64('2008-01-16T20:47:27.343088', 'ns'))
    '16-JAN-2008 20:47:27.343088'
    """
    text = str(np.datetime64(time, 'us'))
    date, clock = text.split('T')
    year, month, day = date.split('-')
    return f'{day}-{_MONTHS[int(month) - 1]}-{year} {clock}'


def _mjd_time(data, offset=0):
    """
    The UTC time of a big-endian MJD 2000 field.

    Parameters
    ----------
    data : bytes
        The record.
    offset : int, optional
        The offset of the field.

    Returns
    -------
    numpy.datetime64
        datetime64[ns].

    Examples
    --------
    >>> _mjd_time(bytes.fromhex('00000b79') + (74847).to_bytes(4, 'big') + (343088).to_bytes(4, 'big'))
    np.datetime64('2008-01-16T20:47:27.343088000')
    """
    from .utils_esa_orbit import _from_mjd

    days = int.from_bytes(data[offset:offset + 4], 'big', signed=True)
    seconds = int.from_bytes(data[offset + 4:offset + 8], 'big')
    micro = int.from_bytes(data[offset + 8:offset + 12], 'big')
    return _from_mjd([days], [seconds], [micro])[0]


def _geolocation(data, count):
    """
    The GEOLOCATION GRID records (EPR ASAR_Geo_Grid_ADSR, GEO_RECORD bytes each).

    Parameters
    ----------
    data : bytes
        The data set.
    count : int
        The number of records.

    Returns
    -------
    dict
        Arrays over the records: ``'first_time'``, ``'last_time'`` (datetime64[ns]), ``'line_num'`` (the first line, from
        1), ``'num_lines'``, and for the first and the last line of each record (prefix ``first_`` / ``last_``),
        (count, TIE_POINTS) arrays ``samples`` (from 1), ``range_time`` (two-way slant range time, s), ``angle``
        (incidence, degrees), ``lat`` and ``lon`` (degrees).

    Examples
    --------
    >>> _geolocation(head[19123:19123 + 11 * 521], 11)['line_num']
    """
    dtype = np.dtype([('first_days', '>i4'), ('first_seconds', '>u4'), ('first_micro', '>u4'), ('attach', 'u1'),
                      ('line_num', '>u4'), ('num_lines', '>u4'), ('track', '>f4'),
                      ('first_samples', '>u4', TIE_POINTS), ('first_range_time', '>f4', TIE_POINTS),
                      ('first_angle', '>f4', TIE_POINTS), ('first_lat', '>i4', TIE_POINTS), ('first_lon', '>i4', TIE_POINTS),
                      ('spare1', 'V22'), ('last_days', '>i4'), ('last_seconds', '>u4'), ('last_micro', '>u4'),
                      ('last_samples', '>u4', TIE_POINTS), ('last_range_time', '>f4', TIE_POINTS),
                      ('last_angle', '>f4', TIE_POINTS), ('last_lat', '>i4', TIE_POINTS), ('last_lon', '>i4', TIE_POINTS),
                      ('spare2', 'V22')])
    assert dtype.itemsize == GEO_RECORD
    from .utils_esa_orbit import _from_mjd

    records = np.frombuffer(bytes(data[:count * GEO_RECORD]), dtype=dtype, count=count)
    out = {'line_num': records['line_num'].astype(np.int64), 'num_lines': records['num_lines'].astype(np.int64)}
    for side in ('first', 'last'):
        out[f'{side}_time'] = _from_mjd(records[f'{side}_days'], records[f'{side}_seconds'], records[f'{side}_micro'])
        out[f'{side}_samples'] = records[f'{side}_samples'].astype(np.int64)
        out[f'{side}_range_time'] = records[f'{side}_range_time'].astype(float) * 1e-9
        out[f'{side}_angle'] = records[f'{side}_angle'].astype(float)
        out[f'{side}_lat'] = records[f'{side}_lat'].astype(float) * 1e-6
        out[f'{side}_lon'] = records[f'{side}_lon'].astype(float) * 1e-6
    return out


def read_header(data):
    """
    Read the header of an Envisat-format product (.N1, .E1, .E2): the MPH, the SPH, the DSDs, the MAIN PROCESSING
    PARAMS fields the line window and the crop need, and the GEOLOCATION GRID.

    Parameters
    ----------
    data : bytes
        The product's first bytes, at least through the GEOLOCATION GRID ADS (the bytes up to MDS1 are enough); more
        bytes are ignored.

    Returns
    -------
    dict
        ``'mph'`` and ``'sph'``: the fields (numbers parsed by their type, units dropped, strings without quotes).
        ``'dsds'``: per DSD a dict ``name``, ``type``, ``filename``, ``offset``, ``size``, ``num_dsr``, ``dsr_size``
        (a spare DSD has the name '' and zeros). ``'main'``: ``first_time``, ``last_time`` (datetime64[ns], the first and
        last zero-Doppler times), ``lines`` (num_output_lines), ``samples`` (num_samples_per_line),
        ``line_time_interval`` (s), ``range_sampling_rate`` (Hz), ``radar_frequency`` (Hz), ``near_range_time`` (the
        two-way slant range time of the first sample, s: the MAIN PROCESSING PARAMS record holds none, it is the
        GEOLOCATION GRID's first tie point, as GMTSAR's envi_slc_decode takes it, moved to sample 1 by the range
        sampling), ``state_vectors`` (``utils_esa_orbit.read_state_vectors``: times and (5, 6) vectors).
        ``'geolocation'``: the records (``_geolocation``). ``'mds1'``: the MDS1 DSD. ``'size'``: the MDS1 offset (the
        header's length).

    Raises
    ------
    ValueError
        A one-line message for a malformed header: too short, not ASCII, a missing MPH field, DSD_SIZE not 280,
        MPH_SIZE + SPH_SIZE not equal to DSD[0].DS_OFFSET, no MAIN PROCESSING PARAMS, GEOLOCATION GRID or MDS1 data
        set, or an MDS1 inconsistent with the MAIN PROCESSING PARAMS (record size, record count).

    Examples
    --------
    >>> read_header(open('ASA_IMS_1PNESA20080116_204727_000000162065_00129_30744_0000.N1', 'rb').read(24854))['main']['lines']
    """
    from .utils_esa_orbit import read_state_vectors

    data = bytes(data)
    if len(data) < MPH_SIZE:
        raise ValueError(f'Envisat header: {len(data)} bytes, the MPH needs {MPH_SIZE}')
    mph = _fields(_text(data[:MPH_SIZE], 'MPH'))
    for key in ('TOT_SIZE', 'SPH_SIZE', 'NUM_DSD', 'DSD_SIZE', 'NUM_DATA_SETS'):
        _number(mph, key, 'MPH')
    if mph['DSD_SIZE'] != DSD_SIZE:
        raise ValueError(f'Envisat header: DSD_SIZE is {mph["DSD_SIZE"]}, expected {DSD_SIZE}')
    sph_size, count = mph['SPH_SIZE'], mph['NUM_DSD']
    if count < 1 or sph_size < count * DSD_SIZE:
        raise ValueError(f'Envisat header: SPH_SIZE {sph_size} cannot hold NUM_DSD {count} DSDs')
    if len(data) < MPH_SIZE + sph_size:
        raise ValueError(f'Envisat header: {len(data)} bytes, the SPH ends at {MPH_SIZE + sph_size}')
    dsd_start = MPH_SIZE + sph_size - count * DSD_SIZE
    sph = _fields(_text(data[MPH_SIZE:dsd_start], 'SPH'))
    dsds = []
    for k in range(count):
        block = data[dsd_start + k * DSD_SIZE:dsd_start + (k + 1) * DSD_SIZE]
        if not block.strip():
            dsds.append(dict(name='', type='', filename='', offset=0, size=0, num_dsr=0, dsr_size=0))
            continue
        fields = _fields(_text(block, f'DSD {k}'))
        dsd = dict(name=str(fields.get('DS_NAME', '')), type=str(fields.get('DS_TYPE', '')),
                   filename=str(fields.get('FILENAME', '')))
        for key, name in (('DS_OFFSET', 'offset'), ('DS_SIZE', 'size'), ('NUM_DSR', 'num_dsr'), ('DSR_SIZE', 'dsr_size')):
            dsd[name] = _number(fields, key, f'DSD {k}')
        dsds.append(dsd)
    if dsds[0]['offset'] != MPH_SIZE + sph_size:
        raise ValueError(f'Envisat header: MPH_SIZE + SPH_SIZE = {MPH_SIZE + sph_size} differs from DSD[0].DS_OFFSET = '
                         f'{dsds[0]["offset"]}')
    named = {}
    for name in (MAIN_PARAMS, GEOLOCATION, MDS1):
        found = [d for d in dsds if d['name'] == name and d['size'] > 0]
        if not found:
            raise ValueError(f'Envisat header: no {name} data set')
        named[name] = found[0]
    main_dsd, geo_dsd, mds1 = named[MAIN_PARAMS], named[GEOLOCATION], named[MDS1]
    for dsd in (main_dsd, geo_dsd):
        if len(data) < dsd['offset'] + dsd['size']:
            raise ValueError(f'Envisat header: {len(data)} bytes, the {dsd["name"]} ends at {dsd["offset"] + dsd["size"]}')
    if geo_dsd['dsr_size'] != GEO_RECORD or geo_dsd['size'] != geo_dsd['num_dsr'] * GEO_RECORD:
        raise ValueError(f'Envisat header: GEOLOCATION GRID records of {geo_dsd["dsr_size"]} bytes, expected {GEO_RECORD}')
    record = data[main_dsd['offset']:main_dsd['offset'] + main_dsd['dsr_size']]
    if len(record) < MAIN_RADAR_FREQUENCY + 4:
        raise ValueError(f'Envisat header: MAIN PROCESSING PARAMS record of {len(record)} bytes is too short')
    lines = int.from_bytes(record[MAIN_LINES:MAIN_LINES + 4], 'big')
    samples = int.from_bytes(record[MAIN_SAMPLES:MAIN_SAMPLES + 4], 'big')
    (line_time_interval,) = np.frombuffer(record[MAIN_LINE_TIME_INTERVAL:MAIN_LINE_TIME_INTERVAL + 4], '>f4')
    rate, frequency = np.frombuffer(record[MAIN_RANGE_SAMPLING_RATE:MAIN_RANGE_SAMPLING_RATE + 8], '>f4')
    if mds1['dsr_size'] != RECORD_HEADER + 4 * samples:
        raise ValueError(f'Envisat header: MDS1 records of {mds1["dsr_size"]} bytes, {samples} samples need '
                         f'{RECORD_HEADER + 4 * samples}')
    if mds1['num_dsr'] != lines or mds1['size'] != lines * mds1['dsr_size']:
        raise ValueError(f'Envisat header: MDS1 holds {mds1["num_dsr"]} records of {mds1["size"]} bytes, '
                         f'num_output_lines is {lines}')
    if geo_dsd['num_dsr'] < 1:
        raise ValueError('Envisat header: the GEOLOCATION GRID has no record')
    geolocation = _geolocation(data[geo_dsd['offset']:geo_dsd['offset'] + geo_dsd['size']], geo_dsd['num_dsr'])
    times, vectors = read_state_vectors(record)
    near_range_time = float(geolocation['first_range_time'][0, 0]
                            - (geolocation['first_samples'][0, 0] - 1) / float(rate))
    main = dict(first_time=_mjd_time(record, MAIN_FIRST_TIME), last_time=_mjd_time(record, MAIN_LAST_TIME), lines=lines,
                samples=samples, line_time_interval=float(line_time_interval), range_sampling_rate=float(rate),
                radar_frequency=float(frequency), near_range_time=near_range_time,
                state_vectors={'times': times, 'vectors': vectors})
    return dict(mph=mph, sph=sph, dsds=dsds, main=main, geolocation=geolocation, mds1=mds1, size=mds1['offset'])


def orbit_dict(orbit):
    """
    The precise orbit of ``utils_esa_orbit.orbit_vectors`` as a ``utils_alos2_orbit`` orbit, for its ``orbit_state``
    (8-point Lagrange interpolation), ``geo2rdr`` and ``bbox_window``.

    Parameters
    ----------
    orbit : dict
        As ``utils_esa_orbit.orbit_vectors`` returns it: ``'times'`` datetime64[ns] UTC epochs at a regular step,
        ``'vectors'`` (n, 6) Earth-fixed x, y, z (m) and vx, vy, vz (m/s).

    Returns
    -------
    dict
        ``'date'`` 'YYYY-MM-DD' of the first epoch, ``'seconds'`` the first epoch's seconds of that day, ``'step'`` the
        seconds between the epochs, ``'vectors'`` (n, 6) float.

    Raises
    ------
    ValueError
        Fewer than 8 epochs, vectors not (n, 6), or epochs not at one regular step.

    Examples
    --------
    >>> orbit_dict({'times': np.datetime64('2008-01-16T20:33', 'ns') + np.arange(8) * np.timedelta64(60, 's'),
    ...             'vectors': np.zeros((8, 6))})['seconds']
    np.float64(73980.0)
    """
    times = np.asarray(orbit['times'], dtype='datetime64[ns]')
    vectors = np.asarray(orbit['vectors'], dtype=float)
    if len(times) < 8:
        raise ValueError(f'Envisat orbit: {len(times)} epochs, the interpolation needs 8')
    if vectors.shape != (len(times), 6):
        raise ValueError(f'Envisat orbit: vectors of shape {vectors.shape} for {len(times)} epochs')
    steps = np.diff(times).astype(np.int64)
    if steps[0] <= 0 or np.any(steps != steps[0]):
        raise ValueError(f'Envisat orbit: epochs not at one regular step ({steps.min() / 1e9}..{steps.max() / 1e9} s)')
    day = times[0].astype('datetime64[D]')
    return {'date': str(day), 'seconds': (times[0] - day).astype(np.int64) / 1e9, 'step': steps[0] / 1e9,
            'vectors': vectors}


def product_image(header):
    """
    The product's line and range geometry as a ``utils_alos2_orbit.bbox_window`` image.

    Parameters
    ----------
    header : dict
        As ``read_header`` returns it.

    Returns
    -------
    dict
        ``'lines'``, ``'pixels'``, ``'time'`` (seconds of the first line's day of line 1), ``'prf'`` (1 / line time
        interval), ``'near_range'`` (m, the first sample) and ``'date'``.

    Examples
    --------
    >>> product_image(read_header(head))['lines']
    """
    main = header['main']
    day = np.datetime64(main['first_time'], 'D')
    return {'lines': main['lines'], 'pixels': main['samples'],
            'time': (np.datetime64(main['first_time'], 'ns') - day).astype(np.int64) / 1e9,
            'prf': 1.0 / main['line_time_interval'], 'near_range': 0.5 * _C * main['near_range_time'], 'date': str(day)}


def tie_lonlat(header, line):
    """
    The longitudes and latitudes of the GEOLOCATION GRID tie points at a line: linear in the line between the tie lines
    (the first and the last line of every record), held at the first or last tie line beyond them.

    Parameters
    ----------
    header : dict
        As ``read_header`` returns it.
    line : float
        The 0-based line.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lon, lat)``, TIE_POINTS each, degrees.

    Examples
    --------
    >>> tie_lonlat(read_header(head), 0)
    """
    geo = header['geolocation']
    lines = np.concatenate([geo['line_num'] - 1, geo['line_num'] + geo['num_lines'] - 2]).astype(float)
    order = np.argsort(lines, kind='stable')
    lon = np.concatenate([geo['first_lon'], geo['last_lon']])[order]
    lat = np.concatenate([geo['first_lat'], geo['last_lat']])[order]
    lines = lines[order]
    return (np.array([np.interp(line, lines, lon[:, k]) for k in range(TIE_POINTS)]),
            np.array([np.interp(line, lines, lat[:, k]) for k in range(TIE_POINTS)]))


def _beyond(orbit, region, heights):
    """
    Whether an AOI lies wholly before the first epoch or wholly after the last epoch of an orbit along the track: the
    zero-Doppler function ``f(t) = (P - S(t)) . V(t)`` (positive while the point is ahead of the satellite) is
    negative at the first epoch for every corner of the region at both heights (all before), or positive at the last
    epoch for all of them (all after).

    Parameters
    ----------
    orbit : dict
        As ``orbit_dict`` returns it.
    region : tuple of float
        ``(west, south, east, north)`` degrees.
    heights : tuple of float
        The lowest and highest heights above the WGS84 ellipsoid, metres.

    Returns
    -------
    bool
        True when every corner's zero-Doppler time lies outside the orbit's span on the same side.

    Examples
    --------
    >>> _beyond(orbit_dict(orbit_vectors('ENVISAT', start, stop)), (14.0, 40.7, 14.5, 41.0), (-500.0, 9000.0))
    False
    """
    from .utils_alos2_orbit import _ecef, orbit_state

    west, south, east, north = region
    lon, lat, height = np.meshgrid([west, east], [south, north], [heights[0], heights[-1]])
    points = _ecef(lon.ravel(), lat.ravel(), height.ravel())
    ends = float(orbit['seconds']), float(orbit['seconds']) + float(orbit['step']) * (len(orbit['vectors']) - 1)
    (first, v_first), (last, v_last) = (orbit_state(orbit, t) for t in ends)
    return bool(np.all((points - first) @ v_first < 0) or np.all((points - last) @ v_last > 0))


def aoi_window(header, bbox, orbit, heights=(-500.0, 9000.0)):
    """
    The MDS1 lines of a product that hold an AOI.

    The transform's UTM grid of the bbox (``utils_grid.transform_region``: the grid rectangles plus 100 m, in lon/lat)
    is mapped by ``utils_alos2_orbit.bbox_window`` on the product's zero-Doppler geometry: the precise orbit of the
    download (``utils_esa_orbit.orbit_vectors``, converted by ``orbit_dict``, 8-point Lagrange interpolation), the
    product's line times and its range sampling (``product_image``), at every height between ``heights``. The window
    is rounded outward to multiples of LINE_BLOCK (512) lines of the product and clipped to its lines. The grid is first
    taken in the UTM zone of the bbox centre; when the GEOLOCATION GRID tie points of the window's first and last lines
    (``tie_lonlat``) span another UTM zone or hemisphere, the window is computed again over the grids of all those zones
    (once), as ALOS-2's download does.

    An AOI whose zero-Doppler times all lie before the orbit's first epoch, or all after its last (``_beyond``), reaches
    no line of the product (the orbit covers the scene): None.

    Parameters
    ----------
    header : dict
        As ``read_header`` returns it.
    bbox : tuple of float
        ``(west, south, east, north)`` in WGS84 degrees.
    orbit : dict
        The precise orbit as ``utils_esa_orbit.orbit_vectors`` returns it; it covers the product's lines.
    heights : tuple of float, optional
        The lowest and highest heights above the WGS84 ellipsoid, metres. Default (-500, 9000).

    Returns
    -------
    tuple of int or None
        ``(i, j)``: the first and the last line (0-based, both included); None when the AOI reaches no line of the
        product at any of the heights, or lies wholly beyond the orbit's span.

    Raises
    ------
    RuntimeError
        The window computed over the zones its tie points span reaches still another zone.
    ValueError
        An orbit that does not cover the product's lines (or is not regular, ``orbit_dict``); an AOI that reaches
        beyond the orbit's span with part of it only.

    Examples
    --------
    >>> aoi_window(read_header(head), (14.0, 40.7, 14.5, 41.0), orbit_vectors('ENVISAT', start, stop))
    """
    from .utils_alos2_orbit import bbox_window
    from .utils_grid import get_utm_epsg, transform_region

    orbit = orbit_dict(orbit)
    image = product_image(header)
    rate = header['main']['range_sampling_rate']
    lines = image['lines']
    days = (np.datetime64(image['date'], 'D') - np.datetime64(orbit['date'], 'D')).astype(np.int64)
    first = image['time'] + 86400.0 * days
    last = first + (lines - 1) / image['prf']
    span = orbit['seconds'], orbit['seconds'] + orbit['step'] * (len(orbit['vectors']) - 1)
    if first < span[0] or last > span[1]:
        raise ValueError(f'Envisat AOI: the orbit {orbit["date"]} {span[0]:.3f}..{span[1]:.3f} s does not cover the '
                         f'product\'s lines {first:.3f}..{last:.3f} s')
    epsgs = {get_utm_epsg(0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[0] + bbox[2]))}
    for attempt in (1, 2):
        region = transform_region(tuple(bbox), epsgs)
        if _beyond(orbit, region, heights):
            return None
        try:
            window = bbox_window(orbit, image, rate, region, heights)
        except ValueError as exc:
            raise ValueError(f'Envisat AOI {tuple(bbox)}: {exc}') from None
        if window is None:
            return None
        window = ((window[0] // LINE_BLOCK) * LINE_BLOCK, min(lines, -(-window[1] // LINE_BLOCK) * LINE_BLOCK))
        lon, lat = [], []
        for line in (window[0], window[1] - 1):
            line_lon, line_lat = tie_lonlat(header, line)
            lon.extend(line_lon)
            lat.extend(line_lat)
        # a longitude every 6 degrees from the west end, and the east end: one in every zone between
        lons = np.r_[np.arange(min(lon), max(lon), 6.0), max(lon)]
        zones = {get_utm_epsg(y, x) for y in (min(lat), max(lat)) for x in lons}
        if zones <= epsgs:
            break
        if attempt == 2:
            raise RuntimeError(f'Envisat lines {window[0]}..{window[1] - 1} reach UTM {sorted(zones - epsgs)} outside the '
                               f'grid of {sorted(epsgs)}')
        epsgs |= zones
    return int(window[0]), int(window[1] - 1)


def _replace(block, key, value):
    """
    Replace the value of a ``KEY=value`` line of a header block, keeping the line's length.

    Parameters
    ----------
    block : bytes
        The MPH, the ASCII SPH or a DSD.
    key : str
        The field name.
    value : str
        The new value text, of the old one's length.

    Returns
    -------
    bytes
        The block with the value replaced.

    Raises
    ------
    ValueError
        No such field, or a value of another length.

    Examples
    --------
    >>> _replace(b'NUM_DSD=+0000000018\\n', 'NUM_DSD', '+0000000019')
    b'NUM_DSD=+0000000019\\n'
    """
    match = re.search(rb'(?m)^' + key.encode('ascii') + rb'=([^\n]*)\n', block)
    if match is None:
        raise ValueError(f'Envisat header: no field {key}')
    new = value.encode('ascii')
    if len(new) != len(match.group(1)):
        raise ValueError(f'Envisat header: {key} value {value!r} does not fit {match.group(1).decode("ascii")!r}')
    return block[:match.start(1)] + new + block[match.end(1):]


def _width(block, key):
    """
    The digits of a signed integer field (``+0000006099<bytes>`` has 10).

    Parameters
    ----------
    block : bytes
        The header block.
    key : str
        The field name.

    Returns
    -------
    tuple
        ``(digits, unit)``: the digit count and the unit text after them ('' when none).

    Raises
    ------
    ValueError
        No such integer field.

    Examples
    --------
    >>> _width(b'SPH_SIZE=+0000006099<bytes>\\n', 'SPH_SIZE')
    (10, '<bytes>')
    """
    match = re.search(rb'(?m)^' + key.encode('ascii') + rb'=[+-](\d+)(<[^>]*>)?\n', block)
    if match is None:
        raise ValueError(f'Envisat header: no integer field {key}')
    return len(match.group(1)), (match.group(2) or b'').decode('ascii')


def _set_int(block, key, value):
    """
    Replace a signed integer field of a header block, keeping its digits and unit.

    Parameters
    ----------
    block : bytes
        The header block.
    key : str
        The field name.
    value : int
        The new value, not negative.

    Returns
    -------
    bytes
        The block with the field replaced.

    Examples
    --------
    >>> _set_int(b'SPH_SIZE=+0000006099<bytes>\\n', 'SPH_SIZE', 6379)
    b'SPH_SIZE=+0000006379<bytes>\\n'
    """
    digits, unit = _width(block, key)
    return _replace(block, key, f'+{value:0{digits}d}{unit}')


def crop_header(head, i, j, first, last, orbit):
    """
    The rewritten header of the crop file of a product's lines i..j (``crop``): the bytes before its MDS1 records. Every
    rewritten field comes from the product's header, i, j, the zero-Doppler times of records i and j and the orbit; none
    from the other records, so the records can follow it from the network (ESA.download streams them).

    Parameters
    ----------
    head : bytes
        The product's bytes up to MDS1 (more bytes are ignored).
    i, j : int
        The first and the last line (0-based, both included).
    first, last : bytes
        The MDS1 records i and j as read from the product (the same record when i == j).
    orbit : dict
        The precise orbit as ``utils_esa_orbit.orbit_vectors`` returns it; it covers the 5 state-vector epochs.

    Returns
    -------
    bytes
        The rewritten MPH, SPH, the 19 DSDs and the ADS; the crop file is these bytes, the records i..j unchanged, then
        ``crop_orbit(orbit)``.

    Raises
    ------
    ValueError
        A malformed header (``read_header``); lines outside the product or ``j < i``; records of another length or whose
        line numbers are not i + 1 and j + 1; a product whose MDS1 is not its last data set or that already holds a
        precise-orbit dataset; an orbit that does not cover the state vectors or whose source names do not fit the DSD.

    Examples
    --------
    >>> crop_header(head, 0, 1023, record0, record1023, orbit_vectors('ENVISAT', start, stop))
    """
    from .utils_esa_orbit import ORBIT_DS_NAME, orbit_ads, orbit_dsd, replace_state_vectors

    header = read_header(head)
    head = bytes(head)[:header['size']]
    main, mds1, dsds = header['main'], header['mds1'], header['dsds']
    total = main['lines']
    if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and 0 <= i <= j < total):
        raise ValueError(f'Envisat crop: lines {i}..{j} are not inside the product\'s 0..{total - 1}')
    i, j = int(i), int(j)
    count = j - i + 1
    size = mds1['dsr_size']
    first, last = bytes(first), bytes(last)
    if len(first) != size or len(last) != size:
        raise ValueError(f'Envisat crop: records of {len(first)} and {len(last)} bytes, MDS1 records have {size}')
    first_num = int.from_bytes(first[13:17], 'big')
    last_num = int.from_bytes(last[13:17], 'big')
    if (first_num, last_num) != (i + 1, j + 1):
        raise ValueError(f'Envisat crop: the records hold lines {first_num}..{last_num}, expected {i + 1}..{j + 1}')
    if any(d['name'] == ORBIT_DS_NAME for d in dsds):
        raise ValueError('Envisat crop: the product already holds a precise-orbit dataset')
    for d in dsds:
        if d['size'] > 0 and d is not mds1 and d['offset'] + d['size'] > mds1['offset']:
            raise ValueError(f'Envisat crop: the {d["name"]} data set lies beyond MDS1')
    if header['mph']['TOT_SIZE'] != mds1['offset'] + mds1['size']:
        raise ValueError(f'Envisat crop: TOT_SIZE {header["mph"]["TOT_SIZE"]} is not the end of MDS1 '
                         f'{mds1["offset"] + mds1["size"]}')
    main_dsd = next(d for d in dsds if d['name'] == MAIN_PARAMS and d['size'] > 0)
    if main_dsd['num_dsr'] != 1:
        raise ValueError(f'Envisat crop: {main_dsd["num_dsr"]} MAIN PROCESSING PARAMS records, expected 1')
    first_bytes = first[:12]
    last_bytes = last[:12]

    mph_size = MPH_SIZE
    sph_size = header['mph']['SPH_SIZE']
    num_dsd = header['mph']['NUM_DSD']
    dsd_start = mph_size + sph_size - num_dsd * DSD_SIZE
    mds_offset = mds1['offset'] + DSD_SIZE
    orbit_offset = mds_offset + count * size
    tot_size = orbit_offset + len(orbit_ads(orbit))

    mph = head[:mph_size]
    mph = _set_int(mph, 'TOT_SIZE', tot_size)
    mph = _set_int(mph, 'SPH_SIZE', sph_size + DSD_SIZE)
    mph = _set_int(mph, 'NUM_DSD', num_dsd + 1)
    mph = _set_int(mph, 'NUM_DATA_SETS', header['mph']['NUM_DATA_SETS'] + 1)
    sph = head[mph_size:dsd_start]
    sph = _replace(sph, 'FIRST_LINE_TIME', f'"{_sph_text(_mjd_time(first_bytes))}"')
    sph = _replace(sph, 'LAST_LINE_TIME', f'"{_sph_text(_mjd_time(last_bytes))}"')
    blocks = []
    for k, dsd in enumerate(dsds):
        block = head[dsd_start + k * DSD_SIZE:dsd_start + (k + 1) * DSD_SIZE]
        if dsd['offset'] > 0:
            block = _set_int(block, 'DS_OFFSET', dsd['offset'] + DSD_SIZE)
        if dsd is mds1:
            block = _set_int(block, 'DS_SIZE', count * size)
            block = _set_int(block, 'NUM_DSR', count)
        blocks.append(block)
    blocks.append(orbit_dsd(orbit, orbit_offset))
    ads = bytearray(head[dsd_start + num_dsd * DSD_SIZE:])
    begin = main_dsd['offset'] - (mph_size + sph_size)
    record = bytearray(replace_state_vectors(bytes(ads[begin:begin + main_dsd['dsr_size']]), orbit))
    record[MAIN_FIRST_TIME:MAIN_FIRST_TIME + 12] = first_bytes
    record[MAIN_LAST_TIME:MAIN_LAST_TIME + 12] = last_bytes
    record[MAIN_LINES:MAIN_LINES + 4] = count.to_bytes(4, 'big')
    ads[begin:begin + main_dsd['dsr_size']] = record
    out = b''.join([mph, sph] + blocks + [bytes(ads)])
    if len(out) != mds_offset:
        raise ValueError(f'Envisat crop: a header of {len(out)} bytes, MDS1 starts at {mds_offset}')
    return out


def crop_orbit(orbit):
    """
    The precise-orbit dataset that ends the crop file (``crop``), after its MDS1 records.

    Parameters
    ----------
    orbit : dict
        The precise orbit as ``utils_esa_orbit.orbit_vectors`` returns it.

    Returns
    -------
    bytes
        The dataset (``utils_esa_orbit.orbit_ads``) the 19th DSD of ``crop_header`` describes.

    Examples
    --------
    >>> crop_orbit(orbit_vectors('ENVISAT', start, stop))
    """
    from .utils_esa_orbit import orbit_ads

    return orbit_ads(orbit)


def crop(head, lines, i, j, orbit):
    """
    The bytes of the crop file of a product's lines i..j, with the precise orbit embedded (the layout in the module
    docstring: U2a Q4, option a): ``crop_header``, the records, ``crop_orbit``. It holds the whole crop in memory;
    ESA.download writes the same bytes with the records streamed to the file.

    Parameters
    ----------
    head : bytes
        The product's bytes up to MDS1 (more bytes are ignored).
    lines : bytes
        The MDS1 records i..j as read from the product.
    i, j : int
        The first and the last line (0-based, both included).
    orbit : dict
        The precise orbit as ``utils_esa_orbit.orbit_vectors`` returns it; it covers the 5 state-vector epochs.

    Returns
    -------
    bytes
        The crop file: the rewritten header, the records unchanged, the precise-orbit dataset.

    Raises
    ------
    ValueError
        A malformed header (``read_header``); lines outside the product or ``j < i``; records of another length or whose
        line numbers are not i + 1 and j + 1; a product whose MDS1 is not its last data set or that already holds a
        precise-orbit dataset; an orbit that does not cover the state vectors or whose source names do not fit the DSD.

    Examples
    --------
    >>> crop(head, lines, 0, 1023, orbit_vectors('ENVISAT', start, stop))
    """
    header = read_header(head)
    lines = bytes(lines)
    total = header['main']['lines']
    if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and 0 <= i <= j < total):
        raise ValueError(f'Envisat crop: lines {i}..{j} are not inside the product\'s 0..{total - 1}')
    count = int(j) - int(i) + 1
    size = header['mds1']['dsr_size']
    if len(lines) != count * size:
        raise ValueError(f'Envisat crop: {len(lines)} bytes of records, lines {i}..{j} need {count * size}')
    # crop_header's TOT_SIZE is its own length (the MDS1 offset) + the records + the orbit dataset: the joined bytes
    return b''.join([crop_header(head, i, j, lines[:size], lines[(count - 1) * size:], orbit), lines, crop_orbit(orbit)])


def read_precise_orbit(data):
    """
    Read the precise-orbit dataset of a crop file (``crop``).

    Parameters
    ----------
    data : bytes
        The crop file's bytes (at least through the end of the dataset).

    Returns
    -------
    dict
        ``'times'`` datetime64[ns] UTC, ``'vectors'`` (n, 6) Earth-fixed x, y, z (m) and vx, vy, vz (m/s),
        ``'source'`` the orbit file names of the DSD's FILENAME (the source zips without ``.zip``).

    Raises
    ------
    ValueError
        A malformed header, no precise-orbit DSD, or data shorter than the dataset.

    Examples
    --------
    >>> read_precise_orbit(crop(head, lines, 0, 1023, orbit))['source']
    """
    from .utils_esa_orbit import ORBIT_DS_NAME, ORBIT_DSR_SIZE, parse_orbit_ads

    data = bytes(data)
    header = read_header(data)
    found = [d for d in header['dsds'] if d['name'] == ORBIT_DS_NAME]
    if not found:
        raise ValueError('Envisat crop: no precise-orbit dataset')
    dsd = found[0]
    if dsd['dsr_size'] != ORBIT_DSR_SIZE or dsd['size'] != dsd['num_dsr'] * ORBIT_DSR_SIZE:
        raise ValueError(f'Envisat crop: precise-orbit records of {dsd["dsr_size"]} bytes, expected {ORBIT_DSR_SIZE}')
    if len(data) < dsd['offset'] + dsd['size']:
        raise ValueError(f'Envisat crop: {len(data)} bytes, the precise-orbit dataset ends at {dsd["offset"] + dsd["size"]}')
    times, vectors = parse_orbit_ads(data[dsd['offset']:dsd['offset'] + dsd['size']])
    return {'times': times, 'vectors': vectors, 'source': dsd['filename'].split('+')}
