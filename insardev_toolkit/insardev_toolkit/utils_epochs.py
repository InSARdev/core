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
Epochs, 1-degree tiles and the VRT index of the model fields that the ERA5, VMF3 and IONEX downloaders store.

Each (epoch, tile) is one NetCDF4 file basedir/<SOURCE>/<tile>_<YYYYmmddTHHMMSS>.nc with one variable per source
variable, dims (level, lat, lon). The index basedir/<SOURCE>.vrt is one GDAL multidimensional VRT over the tile files
present. Tile files are never rewritten; the readers fill the no-data nodes (NaN) of each stored block with
utils_atmosphere atm_fill_nodata and convert nothing else.
"""
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd

from . import utils_tiles
from .utils_atmosphere import atm_fill_nodata
from .utils_files import exists, write_file

# grid per source: node spacing (deg), nodes per tile side (edge nodes kept), epoch step (h), node offset (deg)
# node offset 0.0 for whole degrees, 0.5 for the VMF3 node centres; the VMF3 products share the VMF3 grid
SOURCES = {
    'ERA5': {'step': 0.25, 'nodes': 5, 'hours': 1, 'offset': 0.0},
    'VMF3': {'step': 1.0, 'nodes': 2, 'hours': 6, 'offset': 0.5},
    'VMF3_OP': {'step': 1.0, 'nodes': 2, 'hours': 6, 'offset': 0.5},
    'VMF3_EI': {'step': 1.0, 'nodes': 2, 'hours': 6, 'offset': 0.5},
    'VMF3_FC': {'step': 1.0, 'nodes': 2, 'hours': 6, 'offset': 0.5},
}
EPOCH_FORMAT = '%Y%m%dT%H%M%S'
# the level of a surface variable (no level dimension in the index)
_LEVEL = 'surface'
# a tile name: N40E010 for whole-degree corners, S27.5E150.5 for x.5 corners (both parts x.5 or both whole)
_FILE = re.compile(r'([NS]\d{2}(?:\.5)?[EW]\d{3}(?:\.5)?)_(\d{8}T\d{6})\.nc')
_TILE = re.compile(r'([NS])(\d{2}(?:\.5)?)([EW])(\d{3}(?:\.5)?)')


def _source(source):
    """
    The grid of a source: node spacing (deg), nodes per tile side, epoch step (h), node offset (deg).

    Parameters: source (str). Returns: dict.

    Examples
    --------
    >>> _source('ERA5')['nodes']
    5
    """
    if source not in SOURCES:
        raise ValueError(f'ERROR: unknown source {source!r}, use one of {", ".join(SOURCES)}')
    return SOURCES[source]


def _stamp(time):
    """
    A time as a naive UTC pandas.Timestamp.

    Parameters: time (str, pandas.Timestamp or datetime; a zoned time is converted to UTC). Returns: pandas.Timestamp.

    Examples
    --------
    >>> _stamp('2016-05-11 08:32:52')
    Timestamp('2016-05-11 08:32:52')
    """
    t = pd.Timestamp(time)
    if t.tzinfo is not None:
        t = t.tz_convert('UTC').tz_localize(None)
    return t


def _whole(value, name, offset=0.0):
    """
    A tile corner in degrees: an int for a whole number (offset 0.0), a float for a whole number plus 0.5 (offset 0.5);
    other values raise.

    Parameters: value (int or float); name (str, for the message); offset (0.0 or 0.5, default 0.0). Returns: int or
    float. Raises: ValueError, one ERROR: line, for a value of neither kind.

    Examples
    --------
    >>> _whole(40.0, 'lat')
    40
    >>> _whole(150.5, 'lon', 0.5)
    150.5
    """
    v = float(value)
    if not math.isfinite(v) or not (v - offset).is_integer():
        if offset == 0:
            raise ValueError(f'ERROR: {name} must be a whole degree, got {value}')
        raise ValueError(f'ERROR: {name} must be a whole degree plus 0.5 (a VMF3 node centre), got {value}')
    return int(v) if offset == 0 else v


def _origin(origin):
    """
    The node offset of a cell-corner grid: 0.0 for whole degrees, 0.5 for whole degrees plus 0.5.

    Parameters: origin (int or float, degrees). Returns: 0.0 or 0.5. Raises: ValueError, one ERROR: line, for an origin
    of neither kind.

    Examples
    --------
    >>> _origin(150.5)
    0.5
    >>> _origin(0)
    0.0
    """
    frac = float(origin) % 1
    if frac not in (0.0, 0.5):
        raise ValueError(f'ERROR: origin must be a whole degree or a whole degree plus 0.5, got {origin}')
    return frac


def _cells(lo, hi, size, origin):
    """
    Corner indices k0..k1 (inclusive) of the cells [origin + k*size, origin + (k+1)*size] that cover lo..hi; a cell
    touched only at its edge is excluded, and a zero-width span gives the cell that starts at lo.

    Parameters: lo, hi (degrees, lo <= hi); size (degrees, positive); origin (float, see _origin). Returns: tuple (k0,
    k1). Raises: ValueError, one ERROR: line, for an origin of neither kind.

    Examples
    --------
    >>> _cells(10.2, 10.8, 1, 0)
    (10, 10)
    """
    _origin(origin)
    k0 = math.floor(round((lo - origin) / size, 9))
    k1 = math.ceil(round((hi - origin) / size, 9)) - 1
    return k0, max(k0, k1)


def _corner(k, origin, size):
    """
    The corner position of cell k: an int when origin and size are whole numbers, else a float.

    Parameters: k (int); origin (float, see _origin); size (degrees). Returns: int or float. Raises: ValueError, one
    ERROR: line, for an origin of neither kind.

    Examples
    --------
    >>> _corner(10, 0, 1)
    10
    """
    _origin(origin)
    value = origin + k * size
    if float(origin).is_integer() and float(size).is_integer():
        return int(value)
    return float(value)


def _parse_tile(name, offset=None):
    """
    The corner (lon0, lat0) of a tile name: 'N40E010' gives ints, 'S27.5E150.5' floats.

    Parameters: name (str); offset (0.0 or 0.5, the node offset the name must match; None accepts either). Returns:
    tuple (lon0, lat0). Raises: ValueError, one ERROR: line, for a text that is not a tile name of that kind.

    Examples
    --------
    >>> _parse_tile('N40E010')
    (10, 40)
    >>> _parse_tile('S27.5E150.5', 0.5)
    (150.5, -27.5)
    """
    m = _TILE.fullmatch(name)
    if m is None:
        raise ValueError(f'ERROR: not a tile name: {name}')
    lat_text, lon_text = m.group(2), m.group(4)
    half = lat_text.endswith('.5')
    if half != lon_text.endswith('.5'):
        raise ValueError(f'ERROR: not a tile name: {name}')
    if offset is not None and half != (float(offset) != 0):
        kind = 'an x.5' if float(offset) != 0 else 'a whole-degree'
        raise ValueError(f'ERROR: tile name {name} is not {kind} name for this source')
    number = float if half else int
    lat0 = number(lat_text) * (1 if m.group(1) == 'N' else -1)
    lon0 = number(lon_text) * (1 if m.group(3) == 'E' else -1)
    return lon0, lat0


def bracket(time, step_hours):
    """
    The epochs that bracket a UTC time: the epoch at or before it and the one after; a time on an epoch gives that epoch
    alone.

    Parameters: time (str, pandas.Timestamp or datetime; a naive time is UTC); step_hours (int, 1 for ERA5, 6 for VMF3).
    Returns: list of pandas.Timestamp, one or two, ascending.

    Examples
    --------
    >>> bracket('2016-05-11 08:32:52', 1)
    [Timestamp('2016-05-11 08:00:00'), Timestamp('2016-05-11 09:00:00')]
    """
    if step_hours not in (1, 6):
        raise ValueError(f'ERROR: the epoch step must be 1 or 6 hours, got {step_hours}')
    t = _stamp(time)
    step = pd.Timedelta(hours=step_hours)
    floor = t.floor(step)
    return [floor] if floor == t else [floor, floor + step]


def tile_cells(west, south, east, north, size=1, origin=0):
    """
    The 1-degree tiles whose interior a box touches, with corners on origin + k * size. An edge that lies on a corner
    adds no cell in that direction; west > east crosses the antimeridian.

    Parameters: west, south, east, north (degrees); size (degrees, default 1); origin (whole degree for ERA5, whole
    degree plus 0.5 for VMF3, default 0). Returns: list of (lon0, lat0), sorted by lat0 then lon0. Raises: ValueError,
    one ERROR: line, for an origin of neither kind or bounds outside the grid.

    Examples
    --------
    >>> tile_cells(10.0, 40.0, 12.0, 41.0)
    [(10, 40), (11, 40)]
    """
    frac = _origin(origin)
    if not all(math.isfinite(float(v)) for v in (west, south, east, north)):
        raise ValueError('ERROR: the bounds must be finite degrees')
    if not float(size) > 0:
        raise ValueError(f'ERROR: the cell size must be positive, got {size}')
    if not (-90 <= south <= north <= 90) or not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError(f'ERROR: bounds {west}, {south}, {east}, {north} are outside -180..180 x -90..90, '
                         f'or south is above north')
    lat0_k0, lat0_k1 = _cells(south, north, size, origin)
    spans = [(west, east)] if west <= east else [(west, 180), (-180, east)]
    # lower-left corners of a grid with node offset origin % 1: lon -180..179 (wraps at 180), lat -90..89, both shifted by the offset
    # so that the pole nodes (-90 + offset, 90 - offset) bound the cells
    cells = set()
    for lo, hi in spans:
        lon_k0, lon_k1 = _cells(lo, hi, size, origin)
        for lat_k in range(lat0_k0, lat0_k1 + 1):
            for lon_k in range(lon_k0, lon_k1 + 1):
                lon0, lat0 = _corner(lon_k, origin, size), _corner(lat_k, origin, size)
                if lon0 < -180 + frac:
                    # the cell that starts west of -180 is the cell that ends at 180 + frac, wrapped (x.5 grids only)
                    lon0 += 360
                if -180 + frac <= lon0 <= 179 + frac and -90 + frac <= lat0 <= 89 - frac:
                    cells.add((lon0, lat0))
    return sorted(cells, key=lambda c: (c[1], c[0]))


def tile_nodes(source, lon0, lat0):
    """
    The grid nodes on the edges of one tile, both ends kept: ERA5 has 5 nodes per side at 0.25 degree, VMF3 its 2 corner
    nodes.

    Parameters: source (str); lon0, lat0 (degrees, the lower-left corner: whole degrees for ERA5, whole degrees plus 0.5
    for VMF3). Returns: tuple (lons, lats), float64 arrays, ascending. Raises: ValueError, one ERROR: line, for a corner
    of the other kind than the source's offset.

    Examples
    --------
    >>> lons, lats = tile_nodes('ERA5', 10, 40)
    >>> lons, lats = tile_nodes('VMF3', 150.5, -27.5)
    """
    cfg = _source(source)
    lon0, lat0 = _whole(lon0, 'lon0', cfg['offset']), _whole(lat0, 'lat0', cfg['offset'])
    k = np.arange(cfg['nodes'], dtype=np.float64)
    lons = float(lon0) + cfg['step'] * k
    lats = float(lat0) + cfg['step'] * k
    return lons, lats


def tile_name(lon0, lat0, offset=0):
    """
    The name of a 1-degree tile from its lower-left corner: 'N40E010' (sign letter, 2-digit latitude, sign letter,
    3-digit longitude); with offset 0.5 one decimal is kept, 'S27.5E150.5'.

    Parameters: lon0, lat0 (degrees: whole for offset 0, whole plus 0.5 for offset 0.5); offset (0 or 0.5, default 0).
    Returns: str. Raises: ValueError, one ERROR: line, for a corner of the other kind or outside the grid.

    Examples
    --------
    >>> tile_name(10, 40)
    'N40E010'
    >>> tile_name(150.5, -27.5, 0.5)
    'S27.5E150.5'
    """
    lon0, lat0 = _whole(lon0, 'lon0', offset), _whole(lat0, 'lat0', offset)
    if not (-180 + offset <= lon0 <= 179 + offset and -90 + offset <= lat0 <= 89 - offset):
        raise ValueError(f'ERROR: tile corner {lon0}, {lat0} is outside lon {-180 + offset:g}..{179 + offset:g}, '
                         f'lat {-90 + offset:g}..{89 - offset:g}')
    ns = 'N' if lat0 >= 0 else 'S'
    ew = 'E' if lon0 >= 0 else 'W'
    if offset == 0:
        return f'{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}'
    return f'{ns}{abs(lat0):04.1f}{ew}{abs(lon0):05.1f}'


def _name_start(name):
    """
    The start time of a Sentinel-1 scene name as pandas.Timestamp.

    Parameters: name (str, scene name). Returns: pandas.Timestamp.

    Examples
    --------
    >>> _name_start('S1A_IW_SLC__1SDV_20160511T083252_20160511T083320_011063_0147A3_0001')
    Timestamp('2016-05-11 08:32:52')
    """
    m = re.search(r'(\d{8}T\d{6})', name)
    if m is None:
        raise ValueError(f'ERROR: no start time in the scene name {name}')
    return pd.to_datetime(m.group(1), format=EPOCH_FORMAT)


def scene_items(scenes, source, geometry=None):
    """
    The (epoch, tile) items of the scenes: per scene its bracketing epochs times the 1-degree tiles of its extent,
    without duplicates; corners are on the source grid.

    Parameters: scenes (DataFrame with startTime and geometry columns, an object with a .df DataFrame, or a list of
    scene names); source (str, 'ERA5' hourly or 'VMF3' 6-hourly); geometry (shapely geometry, the area when scenes is a
    list of names). Returns: DataFrame with columns epoch, lon, lat, one row per unique item, sorted.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> scene_items(['S1A_IW_SLC__1SDV_20160511T083252_20160511T083320_011063_0147A3_0001'], 'ERA5',
    ...             geometry=box(10.2, 40.2, 10.8, 40.8))
    """
    cfg = _source(source)
    if isinstance(scenes, pd.DataFrame) or hasattr(scenes, 'df'):
        df = scenes if isinstance(scenes, pd.DataFrame) else scenes.df
        if not {'startTime', 'geometry'} <= set(df.columns):
            raise ValueError('ERROR: the scenes need startTime and geometry columns')
        starts, geoms = list(df['startTime']), list(df['geometry'])
    elif isinstance(scenes, (list, tuple)) and scenes and all(isinstance(s, str) for s in scenes):
        if geometry is None:
            raise ValueError('ERROR: scene names need geometry= for the area of the scenes')
        starts = [_name_start(s) for s in scenes]
        geoms = [geometry] * len(scenes)
    else:
        raise ValueError('ERROR: scenes must be a DataFrame with startTime and geometry, an object with .df, '
                         'or a list of scene names')
    if not starts:
        raise ValueError('ERROR: no scenes')
    rows = set()
    for start, geom in zip(starts, geoms):
        if getattr(geom, 'bounds', None) is None:
            raise ValueError('ERROR: a scene has no geometry')
        west, south, east, north = geom.bounds
        tiles = tile_cells(west, south, east, north, origin=cfg['offset'])
        for epoch in bracket(start, cfg['hours']):
            rows.update((epoch, lon0, lat0) for lon0, lat0 in tiles)
    return pd.DataFrame(sorted(rows), columns=['epoch', 'lon', 'lat'])


def _summary(structure):
    """
    The variables of a structure with their level counts, as text.

    Parameters: structure (list of [variable, dtype, [levels]]). Returns: str.

    Examples
    --------
    >>> _summary([['t', 'float32', ['1000', '850']], ['sp', 'float32', ['surface']]])
    't (2 levels), sp (1 level)'
    """
    return ', '.join(f'{var} ({len(labels)} level{"s" if len(labels) != 1 else ""})' for var, _, labels in structure)


def _normalise(served, source):
    """
    The served values of one (epoch, tile): level labels as str, 2-D arrays checked against the tile nodes, and the
    structure in served order.

    Parameters: served (dict {variable: {level: 2-D array}}); source (str). Returns: tuple (normalised dict, structure).

    Examples
    --------
    >>> _normalise({'sp': {'surface': np.zeros((5, 5), np.float32)}}, 'ERA5')[1]
    [['sp', 'float32', ['surface']]]
    """
    n = _source(source)['nodes']
    if not isinstance(served, dict) or not served:
        raise ValueError('ERROR: fetch must return {variable: {level: 2-D array}}')
    out, structure = {}, []
    for var, levels in served.items():
        if not isinstance(levels, dict) or not levels:
            raise ValueError(f'ERROR: fetch returned no levels for {var}')
        labels = [str(k) for k in levels]
        if len(set(labels)) != len(labels):
            raise ValueError(f'ERROR: fetch returned repeated levels for {var}')
        out[var], dtypes = {}, set()
        for k, arr in levels.items():
            a = np.asarray(arr)
            if a.shape != (n, n):
                raise ValueError(f'ERROR: fetch returned {var} level {k} with shape {a.shape}, expected {(n, n)}')
            out[var][str(k)] = a
            dtypes.add(a.dtype.name)
        if len(dtypes) != 1:
            raise ValueError(f'ERROR: fetch returned mixed dtypes for {var}: {sorted(dtypes)}')
        dtype = dtypes.pop()
        if dtype not in utils_tiles._GDAL_TYPES:
            raise ValueError(f'ERROR: fetch returned {var} as {dtype}, not a supported type')
        structure.append([var, dtype, labels])
    return out, structure


def _write_tile(path, source, lon0, lat0, served, structure, epoch):
    """
    Writes one tile file (variables, coordinates, CRS), built in memory, verified by reading back, and written
    atomically.

    Parameters: path (str); source (str); lon0, lat0 (int); served, structure (as _normalise returns them); epoch
    (pandas.Timestamp). Returns: None.

    Examples
    --------
    >>> _write_tile('ERA5/N40E010_20160511T080000.nc', 'ERA5', 10, 40, served, structure, epoch)
    """
    import io
    utils_tiles.hdf5_plugins()
    import h5py
    lons, lats = tile_nodes(source, lon0, lat0)
    stacks = {var: np.stack([served[var][lev] for lev in labels]) for var, _, labels in structure}
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        dl = f.create_dataset('lat', data=lats, track_times=False)
        dl.make_scale('lat')
        dl.attrs.update(units='degrees_north', standard_name='latitude', long_name='latitude', axis='Y')
        do = f.create_dataset('lon', data=lons, track_times=False)
        do.make_scale('lon')
        do.attrs.update(units='degrees_east', standard_name='longitude', long_name='longitude', axis='X')
        grid = f.create_dataset('crs', data=np.int32(0), track_times=False)
        grid.attrs.update(grid_mapping_name='latitude_longitude', semi_major_axis=6378137.0,
                          inverse_flattening=298.257223563, crs_wkt=utils_tiles.crs_wkt())
        for var, dtype, labels in structure:
            arr = stacks[var]
            dz = f.create_dataset(var, data=arr, chunks=(1, arr.shape[1], arr.shape[2]),
                                  compression=utils_tiles._compression(), track_times=False)
            dz.dims[1].attach_scale(dl)
            dz.dims[2].attach_scale(do)
            dz.attrs['grid_mapping'] = 'crs'
        f.attrs['Conventions'] = 'CF-1.8'
        f.attrs['source'] = source
        f.attrs['epoch'] = epoch.isoformat()
        f.attrs['variables'] = json.dumps(structure)
    data = buf.getvalue()
    with h5py.File(io.BytesIO(data), 'r') as f:
        for var, _, labels in structure:
            # the VRT declares NaN as nodata for float bands: NaN is read back as stored, in the same cells
            if not np.array_equal(f[var][()], stacks[var], equal_nan=np.issubdtype(stacks[var].dtype, np.floating)):
                raise ValueError(f'ERROR: tile verification failed for {path}')
    write_file(path, data)


def _read_structure(path):
    """
    The structure [[variable, dtype, [levels]], ...] of a tile file.

    Parameters: path (str). Returns: list.

    Examples
    --------
    >>> _read_structure('ERA5/N40E010_20160511T080000.nc')
    [['sp', 'float32', ['surface']]]
    """
    utils_tiles.hdf5_plugins()
    import h5py
    try:
        with h5py.File(path, 'r') as f:
            return json.loads(f.attrs['variables'])
    except (OSError, KeyError, ValueError) as e:
        raise ValueError(f'ERROR: tile file {path} cannot be read: {type(e).__name__}') from None


def _scan(folder, tolerant=False, offset=None):
    """
    The tile files of a source folder, sorted by name, with their epoch, corner and structure. An unreadable file raises
    its ERROR: line; with tolerant it is listed with structure None.

    Parameters: folder (str, e.g. basedir/ERA5); tolerant (bool, default False); offset (0.0 or 0.5, None accepts
    either). Returns: list of dict.

    Examples
    --------
    >>> _scan('ERA5', offset=0.0)
    []
    """
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        m = _FILE.fullmatch(name)
        if m is None:
            continue
        path = os.path.join(folder, name)
        lon0, lat0 = _parse_tile(m.group(1), offset)
        structure, error = None, None
        try:
            structure = _read_structure(path)
        except ValueError as e:
            # one ERROR: line that names the file and says to delete it; store() raises it before any fetch
            error = f'{e}, delete it, then store again'
            if not tolerant:
                raise ValueError(error) from None
        out.append({'path': path, 'epoch': pd.to_datetime(m.group(2), format=EPOCH_FORMAT), 'lon0': lon0,
                    'lat0': lat0, 'structure': structure, 'error': error})
    return out


def _listed(vrt):
    """
    The absolute paths of the tile files that a VRT lists.

    Parameters: vrt (str). Returns: list of str.

    Examples
    --------
    >>> _listed('ERA5.vrt')
    []
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(vrt).getroot()
    except (OSError, ET.ParseError) as e:
        raise ValueError(f'ERROR: {vrt} cannot be read: {type(e).__name__}') from None
    base = os.path.dirname(os.path.abspath(vrt))
    out = set()
    for name in root.iter('SourceFilename'):
        text = (name.text or '').strip()
        if text.startswith('NETCDF:'):
            text = text[len('NETCDF:'):].rsplit(':', 1)[0].strip('"')
        out.add(os.path.join(base, text) if name.get('relativeToVRT', '0') == '1' else text)
    return sorted(out)


def _xml(text):
    """
    Text escaped for an XML element.

    Parameters: text (str). Returns: str.

    Examples
    --------
    >>> _xml('a<b')
    'a&lt;b'
    """
    from xml.sax.saxutils import escape
    return escape(str(text))


def _seconds(epoch):
    """
    The epoch as whole seconds since 1970-01-01 UTC.

    Parameters: epoch (pandas.Timestamp or str, on the whole second). Returns: float.

    Examples
    --------
    >>> _seconds('2016-05-11T08:00:00')
    1462953600.0
    """
    return float(pd.Timestamp(epoch).value // 10**9)


def _numbers(values):
    """
    The values of a coordinate array as comma-separated reprs, so that each value reads back exactly.

    Parameters: values (iterable of float). Returns: str.

    Examples
    --------
    >>> _numbers([38.0, 38.25])
    '38.0,38.25'
    """
    return ','.join(repr(float(v)) for v in values)


def _index_xml(vrt, source, files):
    """
    The text of the GDAL multidimensional VRT over the tile files present: dimensions epoch (seconds since 1970), level
    (pressure levels, where the model has them), lat and lon; one array per variable with one source per tile file,
    placed by its offset.

    Parameters: vrt (str; tile paths are relative to its folder); source (str); files (list of dict, from _scan).
    Returns: str. Raises: ValueError, one ERROR: line, when the tiles differ in variables or levels, or a tile is
    north-up.

    Examples
    --------
    >>> _index_xml('dem/ERA5.vrt', 'ERA5', _scan('dem/ERA5'))
    """
    import h5py
    from xml.sax.saxutils import escape
    cfg = _source(source)
    step, n = cfg['step'], cfg['nodes']
    base = os.path.dirname(os.path.abspath(vrt))
    structure = files[0]['structure']
    for f in files:
        if f['structure'] != structure:
            raise ValueError(f'ERROR: {os.path.relpath(f["path"], base)} has variables {_summary(f["structure"])}, '
                             f'the other tiles have {_summary(structure)}')
    labels = None
    for var, dtype, labs in structure:
        if labs == ['surface']:
            continue
        if labels is None:
            labels = labs
        elif labs != labels:
            raise ValueError(f'ERROR: {var} has the levels {_summary([[var, dtype, labs]])}, the other variables have '
                             f'the levels of {labels[0]} to {labels[-1]}: one level dimension holds all of them')
    for f in files:
        with h5py.File(f['path'], 'r') as hf:
            lat = np.asarray(hf['lat'][()])
        if lat.size > 1 and lat[1] < lat[0]:
            raise ValueError(f'ERROR: {os.path.relpath(f["path"], base)} is north-up: delete it, then store() again')
    lon_min = float(min(f['lon0'] for f in files))
    lat_min = float(min(f['lat0'] for f in files))
    lon_max = float(max(f['lon0'] for f in files) + 1)
    lat_max = float(max(f['lat0'] for f in files) + 1)
    ncols = round((lon_max - lon_min) / step) + 1
    nrows = round((lat_max - lat_min) / step) + 1
    lon = lon_min + step * np.arange(ncols, dtype=np.float64)
    lat = lat_min + step * np.arange(nrows, dtype=np.float64)
    by_epoch = {}
    for f in files:
        by_epoch.setdefault(pd.Timestamp(f['epoch']), []).append(f)
    epochs = sorted(by_epoch)
    e_attr = '<Attribute name="{}"><DataType>String</DataType><Value>{}</Value></Attribute>'
    out = ['<VRTDataset>\n<Group name="/">\n',
           '  ' + e_attr.format('model', escape(source)) + '\n',
           f'  <Dimension name="epoch" size="{len(epochs)}" type="TEMPORAL" indexingVariable="epoch"/>\n']
    if labels is not None:
        out.append(f'  <Dimension name="level" size="{len(labels)}" type="VERTICAL" indexingVariable="level"/>\n')
    out.append(f'  <Dimension name="lat" size="{nrows}" type="HORIZONTAL_Y" indexingVariable="lat"/>\n'
               f'  <Dimension name="lon" size="{ncols}" type="HORIZONTAL_X" indexingVariable="lon"/>\n')
    out.append(f'  <Array name="epoch"><DataType>Float64</DataType><DimensionRef ref="epoch"/>'
               f'<InlineValues>{_numbers(_seconds(e) for e in epochs)}</InlineValues>'
               + e_attr.format('units', 'seconds since 1970-01-01 00:00:00') + '</Array>\n')
    if labels is not None:
        out.append(f'  <Array name="level"><DataType>Float64</DataType><DimensionRef ref="level"/>'
                   f'<InlineValues>{_numbers(float(t) for t in labels)}</InlineValues>'
                   + e_attr.format('labels', escape(','.join(labels))) + '</Array>\n')
    out.append(f'  <Array name="lat"><DataType>Float64</DataType><DimensionRef ref="lat"/>'
               f'<InlineValues>{_numbers(lat)}</InlineValues>' + e_attr.format('units', 'degrees_north') + '</Array>\n')
    out.append(f'  <Array name="lon"><DataType>Float64</DataType><DimensionRef ref="lon"/>'
               f'<InlineValues>{_numbers(lon)}</InlineValues>' + e_attr.format('units', 'degrees_east') + '</Array>\n')
    for var, dtype, labs in structure:
        surface = labs == ['surface']
        dims = ['epoch'] + ([] if surface else ['level']) + ['lat', 'lon']
        nodata = '<NoDataValue>nan</NoDataValue>' if dtype.startswith('float') else ''
        out.append(f'  <Array name="{escape(var)}"><DataType>{utils_tiles._GDAL_TYPES[dtype]}</DataType>'
                   + ''.join(f'<DimensionRef ref="{d}"/>' for d in dims) + nodata + '\n')
        for ei, epoch in enumerate(epochs):
            for f in sorted(by_epoch[epoch], key=lambda f: (f['lat0'], f['lon0'])):
                yoff = round((f['lat0'] - lat_min) / step)
                xoff = round((f['lon0'] - lon_min) / step)
                rel = os.path.relpath(f['path'], base)
                dest = f'{ei},{yoff},{xoff}' if surface else f'{ei},0,{yoff},{xoff}'
                view = '' if surface else '<SourceView>[newaxis,...]</SourceView>'
                out.append(f'    <Source><SourceFilename relativeToVRT="1">{escape(rel)}</SourceFilename>'
                           f'<SourceArray>{escape(var)}</SourceArray>{view}<DestSlab offset="{dest}"/></Source>\n')
        out.append('  </Array>\n')
    out.append('</Group>\n</VRTDataset>\n')
    return ''.join(out)


def _write_vrt(vrt, source, files):
    """
    Writes the index over the tile files present, atomically, as the VRT of _index_xml.

    Parameters: vrt (str); source (str); files (list of dict, from _scan). Returns: None.

    Examples
    --------
    >>> _write_vrt('dem/ERA5.vrt', 'ERA5', _scan('dem/ERA5'))
    """
    write_file(vrt, _index_xml(vrt, source, files))


def store(basedir, source, items, fetch, n_jobs=8, joblib_backend='loky'):
    """
    Stores the (epoch, tile) items of a source: each item without a tile file is fetched and written as one NetCDF4
    file; existing tile files are kept, and the VRT basedir/<source>.vrt is rebuilt from the files present. Fetches run
    in joblib workers, while this process writes the files and the VRT in item order, so the result equals that of
    n_jobs=1. An unreadable or missing listed tile file raises one ERROR: line before any fetch that needs it.

    Parameters: basedir (str, tiles go to basedir/<source>/, the index to basedir/<source>.vrt); source (str, 'ERA5',
    'VMF3', 'VMF3_OP', 'VMF3_EI' or 'VMF3_FC'); items (DataFrame with columns epoch, lon, lat; whole-degree corners for
    ERA5, whole plus 0.5 for VMF3, another corner raises one ERROR: line); fetch (callable fetch(epoch, lon, lat)
    returning {variable: {level: 2-D array}} on the tile_nodes; called for new items only, and picklable when n_jobs is
    not 1); n_jobs (int or None, at most two fetches per CPU; None or 1 fetches here; default 8); joblib_backend (str,
    default 'loky'). Returns: DataFrame with columns epoch, tile, path, status ('new' or 'existing'), one row per unique
    item; path is absolute.

    Examples
    --------
    >>> store('dem_tiles', 'ERA5', scene_items(scenes, 'ERA5'), fetch)
    >>> store('dem_tiles', 'ERA5', scene_items(scenes, 'ERA5'), fetch, n_jobs=4)
    """
    cfg = _source(source)
    if not isinstance(items, pd.DataFrame) or not {'epoch', 'lon', 'lat'} <= set(items.columns):
        raise ValueError('ERROR: items must be a DataFrame with epoch, lon and lat columns')
    if not callable(fetch):
        raise ValueError('ERROR: fetch must be a function fetch(epoch, lon, lat)')
    base = os.path.abspath(basedir)
    folder = os.path.join(base, source)
    vrt = os.path.join(base, f'{source}.vrt')
    wanted, seen = [], set()
    for e, lon, lat in zip(items['epoch'], items['lon'], items['lat']):
        epoch = _stamp(e)
        if bracket(epoch, cfg['hours'])[0] != epoch:
            raise ValueError(f'ERROR: epoch {epoch} is not on the {source} grid, every {cfg["hours"]} h')
        lon0, lat0 = _whole(lon, 'lon', cfg['offset']), _whole(lat, 'lat', cfg['offset'])
        tile_name(lon0, lat0, cfg['offset'])
        key = (epoch, lon0, lat0)
        if key not in seen:
            seen.add(key)
            wanted.append(key)
    if not wanted:
        raise ValueError('ERROR: no items to store')
    os.makedirs(folder, exist_ok=True)
    present = _scan(folder, tolerant=True, offset=cfg['offset'])
    bad = [f for f in present if f['structure'] is None]
    if bad:
        more = f' (and {len(bad) - 1} more unreadable tile files)' if len(bad) > 1 else ''
        raise ValueError(bad[0]['error'] + more)
    structure = present[0]['structure'] if present else None
    for f in present:
        if f['structure'] != structure:
            raise ValueError(f'ERROR: {os.path.relpath(f["path"], base)} has variables {_summary(f["structure"])}, '
                             f'the other tiles have {_summary(structure)}')
    def file_of(epoch, lon0, lat0):
        """
        The tile file of an item, basedir/<source>/<tile>_<epoch>.nc.

        Examples
        --------
        >>> file_of(pd.Timestamp('2016-05-11 08:00:00'), 10, 40)
        """
        return os.path.join(folder, f'{tile_name(lon0, lat0, cfg["offset"])}_{epoch.strftime(EPOCH_FORMAT)}.nc')

    def fetched(epoch, lon0, lat0):
        """
        The fetch of one item; an exception it raises becomes one ERROR: line that names the item.

        Examples
        --------
        >>> fetched(pd.Timestamp('2016-05-11 08:00:00'), 10, 40)
        """
        try:
            return fetch(epoch, lon0, lat0)
        except Exception as e:
            text = ' '.join(str(e).split())
            if isinstance(e, ValueError) and text.startswith('ERROR: '):
                text = text[len('ERROR: '):]
            else:
                text = f'{type(e).__name__}: {text}'
            raise ValueError(f'ERROR: fetch of {tile_name(lon0, lat0, cfg["offset"])} at {epoch.isoformat()} failed: '
                             f'{text}') from e

    requested = {file_of(epoch, lon0, lat0) for epoch, lon0, lat0 in wanted}
    if os.path.isfile(vrt):
        for path in _listed(vrt):
            if not os.path.isfile(path) and path not in requested:
                raise ValueError(f'ERROR: {os.path.basename(vrt)} lists the tile file '
                                 f'{os.path.relpath(path, base)}, which is missing: request its epoch and tile '
                                 f'again to fetch it')
    # the items without a tile file, in the order of the items: only these are fetched
    new = [key for key in wanted if not exists(file_of(*key))]
    new_set = set(new)
    parallel = bool(new) and n_jobs is not None and n_jobs != 1
    if parallel:
        import joblib
        from .HTTP import download_jobs
        from .utils_pool import release_workers
        n_jobs = download_jobs(n_jobs)
        # the fetches run in the workers and come back in the order of the items; the parent writes each as it arrives
        results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend, return_as='generator')(
            joblib.delayed(fetched)(epoch, lon0, lat0) for epoch, lon0, lat0 in new)
    else:
        results = (fetched(epoch, lon0, lat0) for epoch, lon0, lat0 in new)
    rows = []
    try:
        for epoch, lon0, lat0 in wanted:
            name = tile_name(lon0, lat0, cfg['offset'])
            path = file_of(epoch, lon0, lat0)
            if (epoch, lon0, lat0) not in new_set:
                status = 'existing'
            else:
                served, new_structure = _normalise(next(results), source)
                if structure is None:
                    structure = new_structure
                elif new_structure != structure:
                    raise ValueError(f'ERROR: {name} at {epoch} has variables {_summary(new_structure)}, the tiles '
                                     f'have {_summary(structure)}')
                _write_tile(path, source, lon0, lat0, served, structure, epoch)
                status = 'new'
            rows.append((epoch, name, os.path.abspath(path), status))
    finally:
        # an unfinished parallel call is aborted here, before its workers are stopped
        results.close()
        if parallel:
            release_workers()
        files = _scan(folder, offset=cfg['offset'])
        if files:
            _write_vrt(vrt, source, files)
    return pd.DataFrame(rows, columns=['epoch', 'tile', 'path', 'status'])


def _warn_filled(vrt_path, count):
    """
    Warns once per call that no-data nodes of the stored blocks were filled.

    Parameters: vrt_path (str, e.g. basedir/ERA5.vrt); count (int, filled nodes of the call; no warning for 0). Returns:
    None.

    Examples
    --------
    >>> _warn_filled('dem_tiles/ERA5.vrt', 3)
    """
    if count:
        model = os.path.splitext(os.path.basename(vrt_path))[0]
        warnings.warn(f'{model}: {count} no-data nodes of the stored tiles are filled from the valid nodes '
                      f'(linear, nearest on the border)', stacklevel=3)


def _node_range(first, step, count, lo, hi):
    """
    The first and last index of the nodes of an ascending axis within [lo - step, hi + step]: the nodes inside the
    bounds with one more on each side.

    Parameters: first (float, first node); step (float); count (int, nodes of the axis); lo, hi (float, bounds).
    Returns: tuple (first, last) indices, inclusive; last is below first when no node is there.

    Examples
    --------
    >>> _node_range(10.0, 0.25, 9, 10.3, 10.6)
    (1, 3)
    """
    i0 = math.ceil((lo - step - first) / step - 1e-6)
    i1 = math.floor((hi + step - first) / step + 1e-6)
    return max(i0, 0), min(i1, count - 1)


# the model name of an index: its group attribute
def _attribute(group, name):
    """
    A string attribute of a group of a multidimensional index; empty when missing.

    Parameters: group (xml.etree.ElementTree.Element); name (str). Returns: str.

    Examples
    --------
    >>> _attribute(ET.parse('dem/ERA5.vrt').getroot().find('Group'), 'model')
    """
    for attr in group.findall('Attribute'):
        if attr.get('name') == name:
            return (attr.findtext('Value') or '').strip()
    return ''


def _inline(group, name):
    """
    The values of a coordinate array of a multidimensional index, in index order.

    Parameters: group (xml.etree.ElementTree.Element); name (str). Returns: list of float. Raises: ValueError, one
    ERROR: line, when the array is missing.

    Examples
    --------
    >>> _inline(ET.parse('dem/ERA5.vrt').getroot().find('Group'), 'lat')
    """
    for arr in group.findall('Array'):
        if arr.get('name') == name:
            return [float(v) for v in (arr.findtext('InlineValues') or '').split(',')]
    raise ValueError(f'ERROR: the index has no {name} array')


class Index:
    """
    The multidimensional VRT index of one model folder, parsed once: model name, epochs, pressure levels, node axes and,
    per variable, its sources by epoch (tile file, file variable, lattice offsets).

    Parameters: vrt_path (str, e.g. basedir/ERA5.vrt). Raises: ValueError, one ERROR: line, when the index cannot be
    read or is not multidimensional.

    Examples
    --------
    >>> index = Index('dem_tiles/ERA5.vrt')
    """

    def __init__(self, vrt_path):
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(vrt_path).getroot()
        except (OSError, ET.ParseError) as e:
            raise ValueError(f'ERROR: {vrt_path} cannot be read: {type(e).__name__}') from None
        group = root.find('Group')
        if group is None:
            raise ValueError(f'ERROR: {vrt_path} is not a multidimensional index: store() again over its tile files')
        self.path = os.path.abspath(vrt_path)
        self.base = os.path.dirname(self.path)
        self.model = _attribute(group, 'model')
        if self.model not in SOURCES:
            raise ValueError(f'ERROR: {vrt_path} has the model {self.model!r}, not one of {", ".join(SOURCES)}')
        self.step = SOURCES[self.model]['step']
        self.nodes = SOURCES[self.model]['nodes']
        self.epochs = [int(v) for v in _inline(group, 'epoch')]
        self.epoch_index = {s: i for i, s in enumerate(self.epochs)}
        self.lat = np.asarray(_inline(group, 'lat'), dtype=np.float64)
        self.lon = np.asarray(_inline(group, 'lon'), dtype=np.float64)
        self.levels = []
        for arr in group.findall('Array'):
            if arr.get('name') == 'level':
                self.levels = [str(v) for v in _attribute_of(arr, 'labels').split(',')]
        self.arrays = {}
        for arr in group.findall('Array'):
            name = arr.get('name')
            if name in ('epoch', 'level', 'lat', 'lon'):
                continue
            dims = [d.get('ref') for d in arr.findall('DimensionRef')]
            sources = {}
            for src in arr.findall('Source'):
                fname = src.find('SourceFilename')
                text = (fname.text or '').strip()
                path = os.path.join(self.base, text) if fname.get('relativeToVRT', '0') == '1' else text
                dest = [int(v) for v in src.find('DestSlab').get('offset').split(',')]
                ei = dest[0]
                yoff, xoff = dest[-2], dest[-1]
                sources.setdefault(ei, []).append((os.path.normpath(path), src.findtext('SourceArray'), yoff, xoff))
            self.arrays[name] = {'surface': 'level' not in dims, 'sources': sources}

    def axes(self, bounds):
        """
        The lattice indices (i0, i1, j0, j1), inclusive, of the nodes inside the bounds with one more on each side,
        clipped to the index.

        Parameters: bounds (lon_min, lat_min, lon_max, lat_max, degrees). Returns: tuple (i0, i1, j0, j1). Raises:
        ValueError, one ERROR: line, when the window holds no node.

        Examples
        --------
        >>> Index('dem_tiles/ERA5.vrt').axes((10.3, 40.3, 10.6, 40.6))
        """
        i0, i1 = _node_range(self.lon[0], self.step, self.lon.size, bounds[0], bounds[2])
        j0, j1 = _node_range(self.lat[0], self.step, self.lat.size, bounds[1], bounds[3])
        if i0 > i1 or j0 > j1:
            raise ValueError(f'ERROR: window {tuple(bounds)} is outside the extent of {self.path}')
        return i0, i1, j0, j1

    def epoch_of(self, epoch):
        """
        The index of an epoch in the epoch dimension.

        Parameters: epoch (str or pandas.Timestamp). Returns: int. Raises: ValueError, one ERROR: line, when the index
        has no such epoch.

        Examples
        --------
        >>> Index('dem_tiles/ERA5.vrt').epoch_of('2016-05-11 08:00:00')
        """
        stamp = _stamp(epoch)
        ei = self.epoch_index.get(_seconds(stamp))
        if ei is None:
            raise ValueError(f'ERROR: no epoch {stamp.isoformat()} in {self.path}: store() that epoch first')
        return ei


def _attribute_of(arr, name):
    """
    An attribute of an array element of a multidimensional index; empty when missing.

    Parameters: arr (xml.etree.ElementTree.Element); name (str). Returns: str.

    Examples
    --------
    >>> _attribute_of(array_element, 'labels')
    """
    for attr in arr.findall('Attribute'):
        if attr.get('name') == name:
            return (attr.findtext('Value') or '').strip()
    return ''


def _read_blocks(index, plan):
    """
    The stored blocks of a plan, each file opened once and each variable read whole; the no-data nodes (NaN) of each 2-D
    block are filled (atm_fill_nodata), and one warning counts the filled nodes.

    Parameters: index (Index, of the model folder); plan (dict {path: variables}). Returns: dict {(path, variable):
    array (levels, rows, columns)}. Raises: ValueError, one ERROR: line, when a listed tile file is missing or
    unreadable, or lacks a listed variable.

    Examples
    --------
    >>> _read_blocks(Index('dem_tiles/ERA5.vrt'), {'dem_tiles/ERA5/10_40_20160511T080000.nc': ['t']})
    """
    utils_tiles.hdf5_plugins()
    import h5py
    blocks = {}
    filled = 0
    for path, names in plan.items():
        if not exists(path):
            raise ValueError(f'ERROR: {path} listed in {index.path} is missing: call store() again with the same items '
                             f'to fetch it')
        try:
            with h5py.File(path, 'r') as f:
                for sv in sorted(names):
                    if sv not in f:
                        raise ValueError(f'ERROR: {path} has no variable {sv}: call store() again with the same items '
                                         f'to fetch it')
                    whole = np.asarray(f[sv][()])
                    # the no-data nodes of each stored 2-D block are filled, then the window is cut from the block
                    levels = []
                    for k in range(whole.shape[0]):
                        block, count = atm_fill_nodata(whole[k])
                        filled += count
                        levels.append(block)
                    blocks[(path, sv)] = np.stack(levels)
        except OSError as e:
            raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then store() again') from None
    _warn_filled(index.path, filled)
    return blocks


def tile_window(index, epoch, bands, row0, col0):
    """
    The values of several bands of one epoch on the nodes of one tile, read from that tile's file only (no-data nodes
    filled as in _read_blocks); a shared edge node takes the value of the tile's own file.

    Parameters: index (Index); epoch (str or pandas.Timestamp); bands (list of (variable, level)); row0, col0 (int,
    lattice row and column of the tile's first node). Returns: dict {(variable, level): float32 array (nodes, nodes)},
    NaN where no file of the tile is listed. Raises: ValueError, one ERROR: line, as _read_blocks.

    Examples
    --------
    >>> tile_window(Index('dem_tiles/ERA5.vrt'), '2016-05-11 08:00:00', [('t', '1000')], 0, 0)
    """
    ei = index.epoch_of(epoch)
    n = index.nodes
    keys, wanted = [], {}
    for variable, level in bands:
        level = str(level)
        arr = index.arrays.get(variable)
        if arr is None:
            raise ValueError(f'ERROR: no band {index.epochs[ei]} {variable} {level} in {index.path}')
        if arr['surface']:
            if level != _LEVEL:
                raise ValueError(f'ERROR: no band {variable} at level {level} in {index.path}: {variable} is a surface variable')
            wanted.setdefault(variable, {})[_LEVEL] = 0
        else:
            if level not in index.levels:
                raise ValueError(f'ERROR: no band {variable} at level {level} in {index.path}')
            wanted.setdefault(variable, {})[level] = index.levels.index(level)
        keys.append((variable, level))
    # the file of the tile for each variable: the source of the epoch at the tile's offsets (the last one listed)
    own, plan = {}, {}
    for variable in wanted:
        for path, sv, yoff, xoff in index.arrays[variable]['sources'].get(ei, []):
            if yoff == row0 and xoff == col0:
                own[variable] = (path, sv)
        if variable in own:
            plan.setdefault(own[variable][0], set()).add(own[variable][1])
    blocks = _read_blocks(index, plan)
    out = {key: np.full((n, n), np.nan, dtype=np.float32) for key in keys}
    for variable, labels in wanted.items():
        if variable not in own:
            continue
        block = blocks[own[variable]]
        for label, lev in labels.items():
            out[(variable, label)][:] = block[lev, :n, :n]
    return out


# VRT indexes that sources() lists: the SOURCES grids except 'VMF3' (stored as VMF3_OP, VMF3_EI, VMF3_FC), and IONEX
_INDEXES = tuple(name for name in SOURCES if name != 'VMF3') + ('IONEX',)


def sources(folder):
    """
    The names of the VRT indexes in a folder (ERA5, VMF3_OP, VMF3_EI, VMF3_FC, IONEX) that exist there; none for a
    missing folder.

    Parameters: folder (str). Returns: list of str, in that order.

    Examples
    --------
    >>> sources('dem_tiles')
    ['ERA5', 'IONEX']
    """
    base = os.path.abspath(folder)
    return [name for name in _INDEXES if os.path.isfile(os.path.join(base, f'{name}.vrt'))]


# the IONEX day files (IONEX.download writes basedir/IONEX/<yyyymmdd>.nc): the maps of 00:00 to 24:00, one hour apart
_DAY_FILE = re.compile(r'(\d{8})\.nc')
_IONEX_MAPS = 25
_IONEX_UNITS = 'seconds since 1970-01-01 00:00:00'


def _axis(values, name, path):
    """
    The first node and step of a regular ascending axis of a day file.

    Parameters: values (numpy.ndarray); name (str, 'lat' or 'lon'); path (str). Returns: tuple (first, step) as floats.
    Raises: ValueError, one ERROR: line, for fewer than two nodes or an axis that is not ascending with one step.

    Examples
    --------
    >>> _axis(np.array([-87.5, -85.0, -82.5]), 'lat', 'IONEX/20160511.nc')
    """
    steps = np.diff(values)
    if len(values) < 2 or not steps[0] > 0 or not np.allclose(steps, steps[0], rtol=0, atol=1e-6):
        raise ValueError(f'ERROR: {path} {name} is not ascending with one step')
    return float(values[0]), float(steps[0])


def _read_day(path, day):
    """
    The map epochs and grid of one IONEX day file, read without its values; the rms maps are not read.

    Parameters: path (str, basedir/IONEX/<yyyymmdd>.nc); day (pandas.Timestamp, the UTC day of the file name). Returns:
    dict with epochs (DatetimeIndex) and grid (first node, step and count of lon and lat). Raises: ValueError, one
    ERROR: line, for an unreadable file, a missing variable, not 25 maps, or no regular ascending grid.

    Examples
    --------
    >>> _read_day('IONEX/20160511.nc', pd.Timestamp('2016-05-11'))
    """
    utils_tiles.hdf5_plugins()
    import h5py
    try:
        with h5py.File(path, 'r') as f:
            if not all(k in f for k in ('time', 'lat', 'lon', 'tec')):
                raise ValueError(f'ERROR: {path} has no time, lat, lon and tec variables')
            units = f['time'].attrs.get('units')
            units = units.decode() if isinstance(units, bytes) else units
            if units != _IONEX_UNITS:
                raise ValueError(f'ERROR: {path} time units are {units!r}, expected {_IONEX_UNITS!r}')
            seconds = np.asarray(f['time'][()], dtype=np.int64)
            lat = np.asarray(f['lat'][()], dtype=np.float64)
            lon = np.asarray(f['lon'][()], dtype=np.float64)
            shape, dtype = f['tec'].shape, f['tec'].dtype
    except OSError as e:
        raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then index again') from None
    if len(seconds) != _IONEX_MAPS:
        raise ValueError(f'ERROR: {path} has {len(seconds)} maps, {_IONEX_MAPS} expected')
    if shape != (len(seconds), len(lat), len(lon)) or dtype != np.int32:
        raise ValueError(f'ERROR: {path} tec is {dtype} {shape}, expected int32 (time, lat, lon)')
    epochs = pd.to_datetime(seconds, unit='s')
    if epochs[0] != day:
        raise ValueError(f'ERROR: {path} starts at {epochs[0]}, not at its day {day:%Y-%m-%d} 00:00')
    lon0, dlon = _axis(lon, 'lon', path)
    lat0, dlat = _axis(lat, 'lat', path)
    grid = {'lon0': lon0, 'dlon': dlon, 'nlon': len(lon), 'lat0': lat0, 'dlat': dlat, 'nlat': len(lat)}
    return {'epochs': epochs, 'grid': grid}


def index_days(basedir, source='IONEX'):
    """
    Writes the multidimensional VRT index basedir/IONEX.vrt over the IONEX day files, atomically: dimensions epoch (seconds
    since 1970, the maps of the day files in order), lat and lon (the node latitudes and longitudes of the day files, as
    ascending values); one array tec (Int32, as published) with one source per day file, placed at its first epoch. The
    rms maps are not indexed. The day files are only read; the index is rebuilt on each call.

    Parameters: basedir (str, the download folder with the IONEX subfolder); source (str, 'IONEX'). Returns: None.
    Raises: ValueError, one ERROR: line, when the source is not IONEX, there is no day file, a day file is unreadable or
    lacks its 25 maps, or the grids differ.

    Examples
    --------
    >>> index_days('dem_tiles', source='IONEX')
    """
    if source != 'IONEX':
        raise ValueError(f"ERROR: index_days indexes the IONEX day files, source must be 'IONEX', not {source!r}")
    base = os.path.abspath(basedir)
    folder = os.path.join(base, source)
    days = []
    for name in (sorted(os.listdir(folder)) if os.path.isdir(folder) else []):
        m = _DAY_FILE.fullmatch(name)
        if m is None:
            continue
        try:
            day = pd.to_datetime(m.group(1), format='%Y%m%d')
        except ValueError:
            continue
        path = os.path.join(folder, name)
        days.append((path, _read_day(path, day)))
    if not days:
        raise ValueError(f'ERROR: no IONEX day file <yyyymmdd>.nc in {folder}')
    first_path, first = days[0][0], days[0][1]['grid']
    for path, info in days[1:]:
        if info['grid'] != first:
            raise ValueError(f'ERROR: {path} has a grid other than {first_path}')
    lat = first['lat0'] + first['dlat'] * np.arange(first['nlat'], dtype=np.float64)
    lon = first['lon0'] + first['dlon'] * np.arange(first['nlon'], dtype=np.float64)
    seconds, sources, offset = [], [], 0
    for path, info in days:
        seconds.extend(_seconds(when) for when in info['epochs'])
        sources.append((offset, os.path.relpath(path, base)))
        offset += len(info['epochs'])
    e_attr = '<Attribute name="{}"><DataType>String</DataType><Value>{}</Value></Attribute>'
    out = ['<VRTDataset>\n<Group name="/">\n',
           '  ' + e_attr.format('model', source) + '\n',
           f'  <Dimension name="epoch" size="{len(seconds)}" type="TEMPORAL" indexingVariable="epoch"/>\n',
           f'  <Dimension name="lat" size="{first["nlat"]}" type="HORIZONTAL_Y" indexingVariable="lat"/>\n',
           f'  <Dimension name="lon" size="{first["nlon"]}" type="HORIZONTAL_X" indexingVariable="lon"/>\n',
           f'  <Array name="epoch"><DataType>Float64</DataType><DimensionRef ref="epoch"/>'
           f'<InlineValues>{_numbers(seconds)}</InlineValues>' + e_attr.format('units', _IONEX_UNITS) + '</Array>\n',
           f'  <Array name="lat"><DataType>Float64</DataType><DimensionRef ref="lat"/>'
           f'<InlineValues>{_numbers(lat)}</InlineValues>' + e_attr.format('units', 'degrees_north') + '</Array>\n',
           f'  <Array name="lon"><DataType>Float64</DataType><DimensionRef ref="lon"/>'
           f'<InlineValues>{_numbers(lon)}</InlineValues>' + e_attr.format('units', 'degrees_east') + '</Array>\n',
           f'  <Array name="tec"><DataType>{utils_tiles._GDAL_TYPES["int32"]}</DataType>'
           '<DimensionRef ref="epoch"/><DimensionRef ref="lat"/><DimensionRef ref="lon"/>\n']
    for start, rel in sources:
        out.append(f'    <Source><SourceFilename relativeToVRT="1">{_xml(rel)}</SourceFilename>'
                   f'<SourceArray>tec</SourceArray><DestSlab offset="{start},0,0"/></Source>\n')
    out.append('  </Array>\n</Group>\n</VRTDataset>\n')
    write_file(os.path.join(base, f'{source}.vrt'), ''.join(out))
