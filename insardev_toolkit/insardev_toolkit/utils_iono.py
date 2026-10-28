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
from . import utils_tiles
from .utils_files import exists
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
    ValueError: When the index cannot be read, is not an IONEX index, or an epoch has no day file.

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
    >>> _match(np.array([40.0, 40.5]), np.array([40.0, 40.5]), 'IONEX/20160511.nc')
    """
    d = np.abs(axis[None, :] - want[:, None])
    idx = d.argmin(axis=1)
    if d[np.arange(want.size), idx].max() > _TOL or np.unique(idx).size != want.size:
        raise ValueError(f'ERROR: the nodes of {path} are not the nodes of the IONEX index: repeat the IONEX download')
    return idx


def _require(f, path, names):
    """Raise the ValueError of a day file that lacks one of the datasets names."""
    for name in names:
        if name not in f:
            raise ValueError(f'ERROR: {path} has no {name} dataset: repeat the IONEX download')


def _day_header(path, lat, lon):
    """The header of a day file: the scale, the base radius and the shell height.

    Parameters
    ----------
    path : str - The day file (IONEX/<yyyymmdd>.nc).
    lat, lon : numpy.ndarray - The lattice latitudes and longitudes, degrees, ascending.

    Returns
    -------
    tuple - (path, scale, radius, shell): the scale in electrons per square metre per stored integer; the radius and shell in km.

    Raises
    ------
    ValueError: When the file cannot be read, lacks an attribute, or its nodes are not the lattice nodes.

    Examples
    --------
    >>> _day_header('dem_tiles/IONEX/20160511.nc', np.array([40.0, 40.5]), np.array([10.0, 11.0]))
    """
    utils_tiles.hdf5_plugins()
    import h5py
    try:
        with h5py.File(path, 'r') as f:
            _require(f, path, ('lat', 'lon', _VARIABLE))
            var = f[_VARIABLE]
            if 'exponent' not in var.attrs:
                raise ValueError(f'ERROR: {path} has no exponent attribute on {_VARIABLE}')
            scale = 10.0 ** int(var.attrs['exponent']) * 1e16
            for key in ('BASE_RADIUS', 'HGT1_HGT2_DHGT'):
                if key not in f.attrs:
                    raise ValueError(f'ERROR: {path} has no {key} attribute: repeat the IONEX download')
            radius = float(f.attrs['BASE_RADIUS'])
            shell = float(np.asarray(f.attrs['HGT1_HGT2_DHGT'], dtype=float).ravel()[0])
            file_lat = np.asarray(f['lat'][()], dtype=float)
            file_lon = np.asarray(f['lon'][()], dtype=float)
    except OSError as e:
        raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then download the IONEX maps again'
                         ) from None
    if file_lat.size != lat.size or file_lon.size != lon.size:
        raise ValueError(f'ERROR: the nodes of {path} are not the nodes of the IONEX index: repeat the IONEX download')
    _match(file_lat, lat, path)
    _match(file_lon, lon, path)
    return path, scale, radius, shell


def _shell(headers):
    """The base radius and shell height in metres from the day file headers; the day files must agree.

    Parameters
    ----------
    headers : list of tuple - The day file headers, radius and shell in km.

    Returns
    -------
    tuple of float - (radius, shell) in metres.

    Raises
    ------
    ValueError: When two day files differ in BASE_RADIUS or HGT1_HGT2_DHGT; the message names them.

    Examples
    --------
    >>> _shell([('dem_tiles/IONEX/20160511.nc', 1e15, 6371.0, 450.0), ('dem_tiles/IONEX/20160512.nc', 1e15, 6371.0, 450.0)])
    """
    first = headers[0]
    for head in headers[1:]:
        if (head[2], head[3]) != (first[2], first[3]):
            raise ValueError(f'ERROR: the day files {os.path.basename(first[0])} and {os.path.basename(head[0])} differ: '
                             f'BASE_RADIUS {first[2]} and {head[2]} km, HGT1 {first[3]} and {head[3]} km')
    return first[2] * 1e3, first[3] * 1e3


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


def _day_maps(path, epochs, ks, lat, lon, scale):
    """The TEC grids of some epochs of one day file, over the whole lattice, in electrons per square metre.

    Parameters
    ----------
    path : str - The day file.
    epochs : tuple of str - The ISO datetimes of the maps.
    ks : tuple of int - The band index of each epoch.
    lat, lon : numpy.ndarray - The lattice latitudes and longitudes, degrees.
    scale : float - The day file's scale, electrons per square metre per stored integer.

    Returns
    -------
    dict - {epoch: grid}, each grid float64 (lat, lon).

    Raises
    ------
    ValueError: When the file cannot be read or its nodes are not the lattice nodes.

    Examples
    --------
    >>> _day_maps('dem_tiles/IONEX/20160511.nc', ('2016-05-11T05:00:00',), (5,), np.array([40.0, 40.5]), np.array([10.0, 11.0]), 1e15)
    """
    import h5py
    from .utils_atmosphere import atm_fill_nodata
    utils_tiles.hdf5_plugins()
    try:
        with h5py.File(path, 'r') as f:
            _require(f, path, ('lat', 'lon', _VARIABLE))
            file_lat = np.asarray(f['lat'][()], dtype=float)
            file_lon = np.asarray(f['lon'][()], dtype=float)
            var = f[_VARIABLE]
            for k in ks:
                if not 0 <= k < var.shape[0]:
                    raise ValueError(f'ERROR: {path} has no map {k}: the index and the file disagree, repeat the download')
            tecs = [np.asarray(var[k], dtype=float) for k in ks]
    except OSError as e:
        raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then download the IONEX maps again'
                         ) from None
    rows, cols = _match(file_lat, lat, path), _match(file_lon, lon, path)
    out = {}
    for epoch, tec in zip(epochs, tecs):
        nodata = tec == _NODATA
        if nodata.any():
            tec, filled = atm_fill_nodata(np.where(nodata, np.nan, tec))
            print(f'WARNING: IONEX: {filled} no-data nodes ({_NODATA}) of the map {epoch} filled (linear, nearest at the border)')
        out[epoch] = tec[np.ix_(rows, cols)] * scale
    return out


def _days(index, epochs, lat, lon):
    """The day file headers and the lazy TEC grids of some epochs.

    Parameters
    ----------
    index : dict - The index of _index().
    epochs : list of pandas.Timestamp - The epochs.
    lat, lon : numpy.ndarray - The lattice latitudes and longitudes, degrees.

    Returns
    -------
    tuple - (headers, maps): the headers in epoch order; maps {ISO datetime: delayed TEC grid}.

    Examples
    --------
    >>> _days(_index('dem_tiles/IONEX.vrt'), [pd.Timestamp('2016-05-11 05:00')], np.array([40.0, 40.5]), np.array([10.0, 11.0]))
    """
    import dask
    headers, items = {}, {}
    for e in epochs:
        iso = e.isoformat()
        path, k = _source(index, e)
        if path not in headers:
            headers[path] = _day_header(path, lat, lon)
            items[path] = ([], [])
        items[path][0].append(iso)
        items[path][1].append(k)
    maps = {}
    for path, (isos, ks) in items.items():
        grids = dask.delayed(_day_maps)(path, tuple(isos), tuple(ks), lat, lon, headers[path][1])
        for iso in isos:
            maps[iso] = grids[iso]
    return list(headers.values()), maps


def _value(tec0, tec1, w0, w1, scale):
    """The IONEX value at every node of one time, in metres of one-way vertical delay.

    Parameters
    ----------
    tec0, tec1 : numpy.ndarray or None - The TEC grids (electrons per square metre) of the two bracketing epochs; None takes tec0 alone.
    w0, w1 : float - The time weights, summing to 1.
    scale : float - The metres of delay per electron per square metre.

    Returns
    -------
    numpy.ndarray - float64, the shape of tec0.

    Examples
    --------
    >>> _value(np.ones((2, 2)), None, 1.0, 0.0, 2.0)
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


def verify(path, times):
    """Check that the maps and day files that the times need are stored in a folder; reads no map values.

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
    for e in epochs:
        f, _ = _source(index, e)
        if not exists(f):
            raise ValueError(f'ERROR: {f} listed in {vrt} is missing: repeat the IONEX download of {e:%Y-%m-%d}')


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
    xarray.Dataset - Lazy; the variable value (time, y, x) in metres, the attributes shell_k (float, R / (R + H)) and
        shell (float, R + H in metres), the coordinates time, y, x, term, epoch and weight, and the CRS EPSG:4326 (spatial_ref).

    Raises
    ------
    ValueError: When the folder has no IONEX index, a map or day file is missing, the wavelength or factor is not valid, or the day files differ in their shell parameters.

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
    verify(path, stamps)
    lam = _wavelength(wavelength)
    fac = _factor(factor)
    vrt = _vrt(path)
    index = _index(vrt)
    lat, lon = _lattice(index)
    plan = _plan(stamps, _STEP_HOURS)
    epochs = sorted({e for e0, e1, w in plan for e in (e0, e1) if e is not None})
    headers, maps = _days(index, epochs, lat, lon)
    radius, height = _shell(headers)
    scale = _K * (lam / _C) ** 2 * fac
    shape = (lat.size, lon.size)
    values, epoch, weights = [], [], []
    for e0, e1, w in plan:
        w0, w1 = (1.0, 0.0) if e1 is None else (1.0 - w, w)
        e1s = e0 if e1 is None else e1
        node = dask.delayed(_value)(maps[e0.isoformat()], None if e1 is None else maps[e1.isoformat()], w0, w1, scale)
        values.append(da.from_delayed(node, shape=shape, dtype=float))
        epoch.append([e0.to_datetime64(), e1s.to_datetime64()])
        weights.append([w0, w1])
    ds = xr.Dataset(
        {'value': (('time', 'y', 'x'), da.stack(values))},
        coords={'time': pd.DatetimeIndex(stamps), 'term': np.arange(2), 'y': lat, 'x': lon,
                'epoch': (('time', 'term'), np.array(epoch, dtype='datetime64[ns]')),
                'weight': (('time', 'term'), np.array(weights, dtype=float))},
        attrs={'shell_k': radius / (radius + height), 'shell': radius + height})
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
    """The ionospheric terms as one lazy tile of the whole lattice, with one table entry per node and time and the shell mapping of every node.

    Parameters
    ----------
    path : str - The folder of the downloaded maps.
    times : list of str or pandas.Timestamp - The acquisition times, UTC; the first axis of the result.
    wavelength : float - The radar wavelength, metres.
    factor : float - The ionospheric factor.

    Returns
    -------
    xarray.Dataset - Lazy, one dask chunk per time: hydro (time, tile, ty, tx, height) with one height, wet and mapping zeros; the attributes shell_k (R / (R + H)) and shell (R + H, metres) from the day file of the first time.
    """
    import dask.array as da
    import xarray as xr
    ds = phase(path, times, wavelength, factor)
    value = da.asarray(ds['value'].data)
    ntime, ny, nx = value.shape
    hydro = value[:, None, :, :, None]
    wet = da.zeros_like(hydro)
    mapping = da.zeros((ntime, 1, ny, nx, 2, 3), chunks=(value.chunks[0], (1,), (ny,), (nx,), (2,), (3,)), dtype=float)
    return xr.Dataset(
        {'hydro': (('time', 'tile', 'ty', 'tx', 'height'), hydro),
         'wet': (('time', 'tile', 'ty', 'tx', 'height'), wet),
         'mapping': (('time', 'tile', 'ty', 'tx', 'component', 'coef'), mapping)},
        coords={'time': ds['time'].values, 'lat': ds['y'].values, 'lon': ds['x'].values,
                'tile y0': ('tile', np.array([0], dtype=np.int64)), 'tile x0': ('tile', np.array([0], dtype=np.int64)),
                'height': [0.0], 'component': ['hydrostatic', 'wet'], 'coef': ['a', 'b', 'c']},
        attrs={'h0': 0.0, 'step': 1.0, 'count': 1, 'niell': [0.0, 0.0, 0.0, 0.0], 'shell_k': float(ds.attrs['shell_k']),
               'shell': float(ds.attrs['shell'])})
