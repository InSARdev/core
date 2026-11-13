# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://InSAR.dev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""Ionospheric delay of a folder of CODE IONEX maps: model(), verify(), delay() (one-way slant delay per node and time, lazy, on the native lattice in lon and lat) and phase() (interferometric phase, radians)."""
import os

import numpy as np
import pandas as pd

from . import utils_epochs
from .utils_tropo import (_plan, _times)

_NAME = 'IONEX'
_VARIABLE = 'tec'
_LEVEL = '0'
_STEP_HOURS = 1
_K = 40.3                 # m3/s2 per electron/m2: the leading Appleton-Hartree term (TN-200 Eq. 21)
_C = 299792458.0          # m/s
_TOL = 1e-6               # degrees: a latitude or longitude of a day file is a lattice node within this
_NODATA = 9999            # IONEX v1 Table 2: the published no-data value of the maps


def _vrt(path):
    """The absolute path of the IONEX.vrt index in a model folder.

    Parameters
    ----------
    path : str - The folder of the downloaded maps.

    Returns
    -------
    str

    Examples
    --------
    >>> _vrt('dem_tiles')
    """
    return os.path.join(os.path.abspath(path), _NAME + '.vrt')


def _wavelength(wavelength):
    """The wavelength as a float in metres.

    Parameters
    ----------
    wavelength : float - The radar wavelength, metres.

    Returns
    -------
    float

    Raises
    ------
    ValueError: When the wavelength is not positive.

    Examples
    --------
    >>> _wavelength(299792458.0 / 5.405e9)
    """
    value = float(wavelength)
    if not value > 0:
        raise ValueError('ERROR: wavelength must be a positive length in metres')
    return value


def _factor(factor):
    """The ionospheric factor as a float, dimensionless.

    Parameters
    ----------
    factor : float - The ratio of the ionospheric to the group delay.

    Returns
    -------
    float

    Raises
    ------
    ValueError: When the factor is not finite.

    Examples
    --------
    >>> _factor(0.78)
    """
    value = float(factor)
    if not np.isfinite(value):
        raise ValueError('ERROR: factor must be a finite number')
    return value


def _index(vrt):
    """
    The multidimensional IONEX.vrt index: the node axes, the epochs and the day file of each map.

    Parameters
    ----------
    vrt : str - The path of the IONEX.vrt index.

    Returns
    -------
    dict - {'path': the index path; 'lat', 'lon': float64 node latitudes and longitudes, degrees; 'seconds': float64 epochs,
    seconds since 1970; 'where': (day file absolute path, map index in the file) of each epoch; 'first': {integer seconds:
    index of its first map}}.

    Raises
    ------
    ValueError: When the index cannot be read, is not an IONEX index, lists a file that is not a day file as published (an index
    written before 2026-10-29), or an epoch has no day file.

    Examples
    --------
    >>> _index('dem_tiles/IONEX.vrt')
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(vrt).getroot()
    except (OSError, ET.ParseError) as e:
        raise ValueError(f'ERROR: {vrt} cannot be read: {type(e).__name__}') from None
    group = root.find('Group')
    if group is None or utils_epochs._attribute(group, 'model') != _NAME:
        raise ValueError(f'ERROR: {vrt} is not a multidimensional IONEX index: index_days() again over the day files')
    base = os.path.dirname(os.path.abspath(vrt))
    seconds = np.asarray(utils_epochs._inline(group, 'epoch'), dtype=np.float64)
    starts = []
    for arr in group.findall('Array'):
        if arr.get('name') != _VARIABLE:
            continue
        for src in arr.findall('Source'):
            name, slab = src.find('SourceFilename'), src.find('DestSlab')
            if name is None or slab is None or slab.get('offset') is None:
                raise ValueError(f'ERROR: {vrt} has a Source without SourceFilename or DestSlab: index_days() again')
            text = (name.text or '').strip()
            path = os.path.join(base, text) if name.get('relativeToVRT', '0') == '1' else text
            if utils_epochs._DAY_FILE.fullmatch(os.path.basename(path)) is None:
                raise ValueError(f'ERROR: {vrt} lists {os.path.basename(path)}, not an IONEX day file as published: index the folder '
                                 f'again (IONEX().download again fetches no file that is present)')
            starts.append((int(slab.get('offset').split(',')[0]), os.path.normpath(path)))
    starts.sort()
    where = [None] * seconds.size
    for n, (start, path) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else seconds.size
        for k in range(start, stop):
            where[k] = (path, k - start)
    if any(w is None for w in where):
        raise ValueError(f'ERROR: {vrt} has maps without a day file: index_days() again over the day files')
    first = {}
    for k, s in enumerate(seconds):
        first.setdefault(int(s), k)
    return {'path': os.path.abspath(vrt), 'lat': np.asarray(utils_epochs._inline(group, 'lat'), dtype=np.float64),
            'lon': np.asarray(utils_epochs._inline(group, 'lon'), dtype=np.float64),
            'seconds': seconds, 'where': where, 'first': first}


def _lattice(index):
    """
    The lattice latitudes and longitudes of an IONEX index, ascending.

    Parameters
    ----------
    index : dict - The index of _index().

    Returns
    -------
    tuple of numpy.ndarray - lat and lon, float64 degrees.

    Raises
    ------
    ValueError: When the index holds fewer than two nodes on an axis.

    Examples
    --------
    >>> _lattice(_index('dem_tiles/IONEX.vrt'))
    """
    lat, lon = np.sort(index['lat']), np.sort(index['lon'])
    if lat.size < 2 or lon.size < 2:
        raise ValueError(f'ERROR: {index["path"]} holds fewer than two nodes on an axis')
    return lat, lon


def _match(axis, want, path):
    """The index in axis of each wanted node: the nearest node of a day file within a tolerance, each node used once.

    Parameters
    ----------
    axis : numpy.ndarray - The latitudes or longitudes of the day file, degrees.
    want : numpy.ndarray - The wanted latitudes or longitudes, degrees.
    path : str - The day file, for the message.

    Returns
    -------
    numpy.ndarray of int

    Raises
    ------
    ValueError: When a wanted node is not a node of the day file.

    Examples
    --------
    >>> _match(np.array([40.0, 40.5]), np.array([40.0, 40.5]), 'IONEX/codg1320.16i.Z')
    """
    d = np.abs(axis[None, :] - want[:, None])
    idx = d.argmin(axis=1)
    if d[np.arange(want.size), idx].max() > _TOL or np.unique(idx).size != want.size:
        raise ValueError(f'ERROR: the nodes of {path} are not the nodes of the IONEX index: repeat the IONEX download')
    return idx


def _day_header(path, lat, lon):
    """One read of a day file: its TEC maps as published and its header values, in one IONEX._read.

    Parameters
    ----------
    path : str - The day file as published (IONEX/<name>).
    lat, lon : numpy.ndarray - The lattice latitudes and longitudes, degrees, ascending.

    Returns
    -------
    dict - path; tec: the integer maps (epoch, lat, lon) as published, on the file's own lattice (ascending); rows, cols: the
    lattice's index in the file's latitudes and longitudes; scale: electrons per square metre per stored integer; radius and hgt:
    BASE RADIUS and HGT1 in km.

    Raises
    ------
    ValueError: When the file cannot be read or decoded (its header included), or its nodes are not the lattice nodes.

    Examples
    --------
    >>> _day_header('dem_tiles/IONEX/codg1320.16i.Z', np.array([40.0, 40.5]), np.array([10.0, 11.0]))
    """
    from .IONEX import IONEX
    text = IONEX._read(path)
    try:
        maps = IONEX._decode_maps(text)
    except ValueError as e:
        raise ValueError(f'ERROR: {path}: {str(e).removeprefix("ERROR: ")}') from None
    if maps['lat'].size != lat.size or maps['lon'].size != lon.size:
        raise ValueError(f'ERROR: the nodes of {path} are not the nodes of the IONEX index: repeat the IONEX download')
    head = maps['header']
    return {'path': path, 'tec': maps['tec'], 'rows': _match(maps['lat'], lat, path), 'cols': _match(maps['lon'], lon, path),
            'scale': 10.0 ** maps['exponent'] * 1e16, 'radius': head['BASE_RADIUS'], 'hgt': head['HGT1_HGT2_DHGT'][0]}


def _shell(reads):
    """The shell R + H in metres from the day file headers (BASE RADIUS and HGT1); the day files must agree.

    Parameters
    ----------
    reads : list of dict - The day file reads of _day_header.

    Returns
    -------
    float - R + H, metres.

    Raises
    ------
    ValueError: When two day files differ in BASE RADIUS or HGT1; the message names them.

    Examples
    --------
    >>> _shell([{'path': 'dem_tiles/IONEX/codg1320.16i.Z', 'radius': 6371.0, 'hgt': 450.0}, {'path': 'dem_tiles/IONEX/codg1330.16i.Z', 'radius': 6371.0, 'hgt': 450.0}])
    """
    first = reads[0]
    for read in reads[1:]:
        if (read['radius'], read['hgt']) != (first['radius'], first['hgt']):
            raise ValueError(f'ERROR: the day files {os.path.basename(first["path"])} and {os.path.basename(read["path"])} differ: '
                             f'BASE RADIUS {first["radius"]} and {read["radius"]} km, HGT1 {first["hgt"]} and {read["hgt"]} km')
    return first['radius'] * 1e3 + first['hgt'] * 1e3


def _source(index, epoch):
    """
    The day file and the map index in it of the map of one epoch.

    Parameters
    ----------
    index : dict - The index of _index().
    epoch : pandas.Timestamp - The epoch of the map (UTC, on the whole hour).

    Returns
    -------
    tuple - (day file absolute path, map index in the file).

    Raises
    ------
    ValueError: When the index has no map of the epoch: repeat the IONEX download of its day.

    Examples
    --------
    >>> _source(_index('dem_tiles/IONEX.vrt'), pd.Timestamp('2016-05-11 05:00'))
    """
    k = index['first'].get(int(utils_epochs._seconds(epoch)))
    if k is None:
        raise ValueError(f'ERROR: the IONEX map of {epoch:%Y-%m-%d %H:%M} is not in {index["path"]}: repeat the IONEX '
                         f'download of {epoch:%Y-%m-%d}')
    return index['where'][k]


def _day_maps(read, k, epoch):
    """The TEC grid of one map of a read day file over the lattice, in electrons per square metre; the no-data nodes are filled.

    Parameters
    ----------
    read : dict - The day file read of _day_header.
    k : int - The map index in the file.
    epoch : str - The ISO datetime of the map, for the message.

    Returns
    -------
    numpy.ndarray - float64, (lat, lon).

    Raises
    ------
    ValueError: When the file has no map k: the index and the file disagree.

    Examples
    --------
    >>> _day_maps({'path': 'dem_tiles/IONEX/codg1320.16i.Z', 'tec': np.zeros((1, 2, 2), dtype=np.int32), 'rows': np.array([0, 1]), 'cols': np.array([0, 1]), 'scale': 1e15}, 0, '2016-05-11T00:00:00')
    """
    from .utils_atmosphere import atm_fill_nodata
    path, tecs = read['path'], read['tec']
    if not 0 <= k < tecs.shape[0]:
        raise ValueError(f'ERROR: {path} has no map {k}: the index and the file disagree, repeat the download')
    tec = np.asarray(tecs[k], dtype=float)
    nodata = tec == _NODATA
    if nodata.any():
        tec, filled = atm_fill_nodata(np.where(nodata, np.nan, tec))
        print(f'WARNING: IONEX: {filled} no-data nodes ({_NODATA}) of the map {epoch} filled (linear, nearest at the border)')
    return tec[np.ix_(read['rows'], read['cols'])] * read['scale']


def _days(index, epochs, lat, lon):
    """The day file reads and the lazy TEC grids of some epochs: one read task per day file, whatever the epochs it serves.

    Parameters
    ----------
    index : dict - The index of _index().
    epochs : list of pandas.Timestamp - The epochs.
    lat, lon : numpy.ndarray - The lattice latitudes and longitudes, degrees.

    Returns
    -------
    tuple - (reads, grids): reads {day file path: delayed _day_header}; grids {ISO datetime: delayed TEC grid}.

    Examples
    --------
    >>> _days(_index('dem_tiles/IONEX.vrt'), [pd.Timestamp('2016-05-11 05:00')], np.array([40.0, 40.5]), np.array([10.0, 11.0]))
    """
    import dask
    reads, grids = {}, {}
    for e in epochs:
        path, k = _source(index, e)
        if path not in reads:
            reads[path] = dask.delayed(_day_header)(path, lat, lon)
        grids[e.isoformat()] = dask.delayed(_day_maps)(reads[path], k, e.isoformat())
    return reads, grids


def _value(tec0, tec1, w0, w1, scale, shell):
    """The IONEX value at every node of one time, in metres of one-way vertical delay.

    Parameters
    ----------
    tec0, tec1 : numpy.ndarray or None - The TEC grids (electrons per square metre) of the two bracketing epochs; None takes tec0 alone.
    w0, w1 : float - The time weights, summing to 1.
    scale : float - The metres of delay per electron per square metre.
    shell : float - The shell R + H of the day files, metres: a dependency only, so the value is computed when the day files agree.

    Returns
    -------
    numpy.ndarray - float64, the shape of tec0.

    Examples
    --------
    >>> _value(np.ones((2, 2)), None, 1.0, 0.0, 2.0, 6821e3)
    """
    value = w0 * tec0
    if tec1 is not None:
        value = value + w1 * tec1
    return scale * value


def model(path):
    """The name of the ionospheric model of a folder: 'IONEX' when it holds the IONEX.vrt index.

    Parameters
    ----------
    path : str - The folder of the downloaded model fields.

    Returns
    -------
    str - 'IONEX'.

    Raises
    ------
    ValueError: When the folder holds no IONEX.vrt.

    Examples
    --------
    >>> model('dem_tiles')
    'IONEX'
    """
    if _NAME in utils_epochs.sources(path):
        return _NAME
    raise ValueError(f'ERROR: no IONEX index (IONEX.vrt) in {path}')


_VERIFIED = set()   # (IONEX.vrt path, its mtime, the epochs) verified in this process


def verify(path, times):
    """Check that the maps and day files that the times need are stored in a folder; reads no map values. The result is
    kept per process for the index and the epochs, and each day file is checked once (utils_epochs.exists_once).

    Parameters
    ----------
    path : str - The folder of the downloaded model fields.
    times : list of str or pandas.Timestamp - The acquisition times, UTC.

    Raises
    ------
    ValueError: Naming the first missing map (day and hour) or day file, and the IONEX download to repeat.

    Examples
    --------
    >>> verify('dem_tiles', ['2016-05-11 05:15'])
    """
    model(path)
    stamps = _times(times)
    vrt = _vrt(path)
    index = _index(vrt)
    epochs = sorted({e for e0, e1, w in _plan(stamps, _STEP_HOURS) for e in (e0, e1) if e is not None})
    key = (vrt, os.stat(vrt).st_mtime_ns, tuple(epochs))
    if key in _VERIFIED:
        return
    for e in epochs:
        f, _ = _source(index, e)
        if not utils_epochs.exists_once(f):
            raise ValueError(f'ERROR: {f} listed in {vrt} is missing: repeat the IONEX download of {e:%Y-%m-%d}')
    _VERIFIED.add(key)


def delay(path, times, wavelength, factor):
    """The one-way slant delay of an IONEX folder, lazy: one value per node and time, in metres, on the native lattice in lon and lat.

    Parameters
    ----------
    path : str - The folder of the downloaded maps.
    times : list of str or pandas.Timestamp - The acquisition times, UTC; the first axis of the result.
    wavelength : float - The radar wavelength, metres (C-band Sentinel-1: 299792458 / 5.405e9).
    factor : float - The ionospheric factor applied to the delay (e.g. 0.78).

    Returns
    -------
    xarray.Dataset - Lazy; the variable value (time, y, x) in metres and shell (no dims), R + H in metres from the day file headers,
        checked to agree (the value depends on that check); the coordinates time, term, epoch, weight, y, x, and the CRS EPSG:4326 (spatial_ref).

    Raises
    ------
    ValueError: When the folder has no IONEX index, the wavelength or factor is not valid, or the day files differ in their shell
        parameters (raised when the values are computed). A missing map or day file is named when it is read.

    Examples
    --------
    >>> delay('dem_tiles', ['2016-05-11 05:15'], 299792458.0 / 5.405e9, 0.78)
    """
    import dask
    import dask.array as da
    import rioxarray  # noqa: F401  (the .rio accessor)
    import xarray as xr
    model(path)
    stamps = _times(times)
    lam = _wavelength(wavelength)
    fac = _factor(factor)
    index = _index(_vrt(path))
    lat, lon = _lattice(index)
    plan = _plan(stamps, _STEP_HOURS)
    epochs = sorted({e for e0, e1, w in plan for e in (e0, e1) if e is not None})
    reads, grids = _days(index, epochs, lat, lon)
    shell = dask.delayed(_shell)(list(reads.values()))
    scale = _K * (lam / _C) ** 2 * fac
    shape = (lat.size, lon.size)
    values, epoch, weights = [], [], []
    for e0, e1, w in plan:
        w0, w1 = (1.0, 0.0) if e1 is None else (1.0 - w, w)
        e1s = e0 if e1 is None else e1
        node = dask.delayed(_value)(grids[e0.isoformat()], None if e1 is None else grids[e1.isoformat()], w0, w1, scale, shell)
        values.append(da.from_delayed(node, shape=shape, dtype=float))
        epoch.append([e0.to_datetime64(), e1s.to_datetime64()])
        weights.append([w0, w1])
    ds = xr.Dataset(
        {'value': (('time', 'y', 'x'), da.stack(values)), 'shell': ((), da.from_delayed(shell, shape=(), dtype=float))},
        coords={'time': pd.DatetimeIndex(stamps), 'term': np.arange(2), 'y': lat, 'x': lon,
                'epoch': (('time', 'term'), np.array(epoch, dtype='datetime64[ns]')),
                'weight': (('time', 'term'), np.array(weights, dtype=float))})
    return ds.rio.write_crs('EPSG:4326')


def phase(path, times, wavelength, factor):
    """The interferometric phase of the ionospheric delay, in radians, lazy.

    Parameters
    ----------
    path : str - The folder of the downloaded maps.
    times : list of str or pandas.Timestamp - The acquisition times, UTC.
    wavelength : float - The radar wavelength, metres.
    factor : float - The ionospheric factor applied to the delay (e.g. 0.78).

    Returns
    -------
    xarray.Dataset - As delay, the variable value in radians.

    Raises
    ------
    ValueError: When the wavelength is not positive, or as delay.

    Examples
    --------
    >>> phase('dem_tiles', ['2016-05-11 05:15'], 299792458.0 / 5.405e9, 0.78)
    """
    lam = _wavelength(wavelength)
    ds = delay(path, times, lam, factor)
    return ds.assign(value=ds['value'] * (4.0 * np.pi / lam))


def zenith_tiles(path, times, wavelength, factor):
    """The ionospheric terms of a folder for the stack: the zenith TEC phase on the native lattice and the shell height, lazy.

    Parameters
    ----------
    path : str - The folder of the downloaded maps.
    times : list of str or pandas.Timestamp - The acquisition times, UTC; the first axis of the result.
    wavelength : float - The radar wavelength, metres.
    factor : float - The ionospheric factor.

    Returns
    -------
    xarray.Dataset - Lazy, one dask chunk per time: tec (time, lat, lon), the zenith TEC phase in radians (-4 pi / wavelength is not
        +4 pi / wavelength times the one-way delay, the sign of the model); shell (no dims), R + H in metres from the day files,
        checked to agree and computed with the tec tables. Coordinates time, lat and lon (the IONEX nodes).
    """
    import xarray as xr
    ds = phase(path, times, wavelength, factor)
    return xr.Dataset({'tec': (('time', 'lat', 'lon'), ds['value'].data), 'shell': ((), ds['shell'].data)},
                      coords={'time': ds['time'].values, 'lat': ds['y'].values, 'lon': ds['x'].values})

