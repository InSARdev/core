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
Geoid heights of the NGA EGM2008 and EGM96 geoids, to convert DEM heights to WGS84 ellipsoidal heights:
ellipsoidal height = DEM height + geoid height of the DEM's vertical datum.

The grids ship with the package in data/geoid/ (sources, licenses and conversion in the README there):

    EGM2008   2.5' grid   NGA via PROJ us_nga_egm08_25.tif            vertical CRS EPSG:3855
    EGM96     7.5' grid   NGA F477 synthesis of NGA's EGM96 model      vertical CRS EPSG:5773

Both are int16 NetCDF4 grids with scale_factor 0.003 m, read through utils_tiles like a DEM. A call reads only the
grid window around the requested points, block by block, so no process ever holds a whole grid. The heights are
interpolated by the cubic B-spline through the grid nodes, the same spline as scipy.ndimage (order 3), with the
grid continued across the poles on the opposite meridian. Measured errors including the 3 mm storage step
(data/geoid/README.md): EGM2008 max 0.0053 m at all nodes of NGA's 1' grid and 0.0056 m at the 2.5' cell centres;
EGM96 max 0.0029 m at the 7.5' cell centres against NGA's EGM96 model; rms 0.0008 m or less for both.

The vertical datum of a DEM (dem_datum()) is found in this order:
1. the datum argument;
2. the vertical CRS the DEM declares: the VRT <SRS>, the NetCDF crs_wkt, spatial_ref or geoid_name attributes,
   the same attributes of an xarray object and of its spatial_ref coordinate;
3. the tile names of downloads made before the toolkit wrote the vertical CRS: Copernicus_DSM_COG_* tiles are
   EGM2008, SRTM (skadi) NxxWxxx and ALOS ALPSMLC30_* tiles are EGM96;
4. otherwise EGM2008, the datum of the Copernicus DEM and the newer geoid, with a warning once per DEM.
An xarray object read from a file (xarray's encoding source, the source attribute of utils_tiles.open_dem() and
Tiles.open()) that declares none of step 2 itself has the datum of that file. A Dataset is the DEM of its first data
variable, as Satellite.get_dem() reads it: the attributes and the file of that variable count as its own. A NetCDF4
file is the DEM of the variable utils_tiles reads from it (for an xarray object read from the file, of the variable
it holds), by the same rule: the attributes of that variable count as the file's own, and a file saved from a DEM
read from another file (its source attribute) has the datum of that file.

xarray keeps the file of xr.open_dataarray() and xr.open_dataset() in its encoding only, and where(), astype(),
arithmetic and interp() drop the encoding: pass the datum for such a DEM, or open the file with
utils_tiles.open_dem() or Tiles().open(), which keep the file as an attribute.

ellipsoidal_height() converts a DEM to heights above the ellipsoid and marks the result so.
"""
import os
import re
import weakref
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'geoid')
# datum name -> the bundled grid and the EPSG code of the vertical CRS
GEOIDS = {
    'EGM2008': dict(file='egm2008_2.5min.nc', epsg=3855),
    'EGM96': dict(file='egm96_7.5min.nc', epsg=5773),
}
# heights above the WGS84 ellipsoid need no geoid (geoid height 0)
ELLIPSOID = 'ellipsoid'
# the datum of a DEM that declares none
DEFAULT_DATUM = 'EGM2008'
# grid nodes per block of the windowed reads, and the spline margin around a block: the B-spline prefilter decays
# as 0.268^k, 0.268^16 < 1e-9, so a block read with this margin gives the coefficients of the whole grid
BLOCK = 256
MARGIN = 16
# points (or grid cells) evaluated at once; bounds the temporaries of a call to a few MB
CHUNK = 65536
_POLE = np.sqrt(3.0) - 2.0

_NAMES = {'EGM2008': 'EGM2008', 'EGM08': 'EGM2008', 'EGM96': 'EGM96', 'ELLIPSOID': ELLIPSOID,
          'ELLIPSOIDAL': ELLIPSOID, 'WGS84': ELLIPSOID}
# tile names of the toolkit downloads made before the vertical CRS was written (see Tiles)
_TILE_NAMES = (
    (re.compile(r'^Copernicus_DSM_(COG_)?\d+_[NS]\d{2}_\d{2}_[EW]\d{3}_\d{2}_DEM', re.I), 'EGM2008'),
    (re.compile(r'^ALPSMLC30_[NS]\d{3}[EW]\d{3}_DSM', re.I), 'EGM96'),
    (re.compile(r'^[NS]\d{2}[EW]\d{3}(\.hgt)?(\.gz)?(\.nc)?$', re.I), 'EGM96'),
)
_SAID = set()


def _say(key, message, owner=None):
    """Print a message once per process, or once per owner object while it exists."""
    if key not in _SAID:
        print(message)
        _SAID.add(key)
        if owner is not None:
            # the key of an object is its id, which a later object may reuse: the key goes with the object
            try:
                weakref.finalize(owner, _SAID.discard, key)
            except TypeError:
                pass


# -----------------------------------------------------------------------------------------------------------------
# datum names and CRS
# -----------------------------------------------------------------------------------------------------------------

def _crs_datum(crs):
    """
    The vertical datum a CRS declares: 'EGM2008', 'EGM96', 'ellipsoid' (a 3D geographic CRS such as EPSG:4979), or
    None for a CRS without heights or one that does not parse. Raises for another vertical datum.
    """
    from pyproj import CRS
    try:
        crs = CRS.from_user_input(crs)
    except Exception:
        return None
    if crs.type_name == 'Bound CRS' and crs.source_crs is not None:
        crs = crs.source_crs
    if crs.is_compound:
        vertical = [c for c in crs.sub_crs_list if c.type_name == 'Vertical CRS']
    elif crs.type_name == 'Vertical CRS':
        vertical = [crs]
    else:
        return ELLIPSOID if crs.is_geographic and len(crs.axis_info) == 3 else None
    if not vertical:
        return None
    v = vertical[0]
    code = v.to_epsg()
    for name, geoid in GEOIDS.items():
        if code == geoid['epsg'] or (code is None and name in v.name.upper().replace(' ', '')):
            return name
    raise ValueError(f'ERROR: vertical datum "{v.name}" is not supported. Use EGM2008, EGM96 or ellipsoidal heights.')


def datum_name(datum):
    """
    The datum name 'EGM2008', 'EGM96' or 'ellipsoid' of a name ('EGM2008', 'egm96', 'EGM2008 geoid', 'EGM96 height',
    'ellipsoid', 'WGS84'), an EPSG code (3855, 5773, 'EPSG:4326+3855', 4979) or a CRS. Raises for anything else.
    """
    if isinstance(datum, str):
        key = re.sub(r'[\s_-]', '', datum).upper()
        key = re.sub(r'(GEOID|HEIGHT)$', '', key)
        if key in _NAMES:
            return _NAMES[key]
    name = _crs_datum(datum)
    if name is None:
        raise ValueError(f'ERROR: vertical datum not recognized: {datum!r}. Use EGM2008, EGM96 or ellipsoid.')
    return name


def _name_datum(names):
    """The datum of old toolkit tiles from their names, None when no name is a known tile; raises when they mix
    datums."""
    found = set()
    for name in names:
        base = os.path.basename(str(name).rstrip('/'))
        for pattern, datum in _TILE_NAMES:
            if pattern.match(base):
                found.add(datum)
                break
    if len(found) > 1:
        raise ValueError(f'ERROR: the DEM mixes tiles of the vertical datums {", ".join(sorted(found))}. '
                         f'Download the DEM from one provider.')
    return found.pop() if found else None


def _attrs_datum(attrs):
    """The datum declared by CF/rioxarray attributes: crs_wkt, spatial_ref, geoid_name."""
    for key in ('crs_wkt', 'spatial_ref', 'crs'):
        value = attrs.get(key)
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str) and value.strip():
            name = _crs_datum(value)
            if name is not None:
                return name
    for key in ('geoid_name', 'vertical_datum'):
        value = attrs.get(key)
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str) and value.strip():
            return datum_name(value)
    return None


def _vrt_datum(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    srs = root.findtext('SRS')
    name = _crs_datum(srs) if srs and srs.strip() else None
    if name is not None:
        return name
    files = [s.text.strip() for s in root.iter('SourceFilename') if s.text]
    files = [f[len('NETCDF:'):].rsplit(':', 1)[0].strip('"') if f.startswith('NETCDF:') else f for f in files]
    return _name_datum(files)


def _file_datum(path, var=None, seen=None):
    """The datum a local DEM file declares or its tile names show: a VRT or a NetCDF4 grid; None for another file."""
    from . import utils_tiles
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.vrt':
        return _vrt_datum(path)
    if ext in utils_tiles.NETCDF_EXTENSIONS and utils_tiles.is_hdf5(path):
        return _netcdf_datum(path, var, seen)
    return None


def _netcdf_datum(path, var=None, seen=None):
    """
    The datum of a NetCDF4 grid, by the rule of an xarray DEM read from a file: the attributes of its CRS variables,
    of its DEM variable, of the variables that name a grid mapping and of the file; its tile name; the datum of the
    file in the source attribute of the DEM variable or of the file (a saved crop); the tile name of that source.
    The DEM variable is var, the 2D variable an xarray object read from the file holds, otherwise the one utils_tiles
    reads. seen holds the files followed, so sources that name each other end.
    """
    import h5py
    from . import utils_tiles
    seen = set() if seen is None else seen
    seen.add(os.path.realpath(path))
    with h5py.File(path, 'r') as f:
        def attrs(obj):
            return {k: obj.attrs[k] for k in obj.attrs}
        if isinstance(var, str) and var in f and getattr(f[var], 'ndim', 0) == 2:
            dem = var
        else:
            dem = utils_tiles.data_variable(f)
        # the DEM variable, then the variables that name a grid mapping
        names = [dem] if dem is not None else []
        names += [n for n in f if n != dem and isinstance(f[n], h5py.Dataset) and 'grid_mapping' in f[n].attrs]
        found, mappings = [], []
        for name in names:
            obj = f[name]
            if 'grid_mapping' in obj.attrs:
                gm = obj.attrs['grid_mapping']
                gm = gm.decode() if isinstance(gm, bytes) else str(gm)
                mappings.append(gm.split(':')[0].strip())
            found.append(attrs(obj))
        # the grid mapping variables: named by a data variable, or the usual names
        for name in mappings + ['crs', 'spatial_ref']:
            if name in f:
                found.insert(0, attrs(f[name]))
        found.append(attrs(f))
        sources = [f.attrs.get('source')] + ([f[dem].attrs.get('source')] if dem is not None else [])
    for a in found:
        name = _attrs_datum(a)
        if name is not None:
            return name
    name = _name_datum([os.path.splitext(os.path.basename(path))[0]])
    if name is not None:
        return name
    sources = [s.decode() if isinstance(s, bytes) else s for s in sources]
    sources = list(dict.fromkeys(s for s in sources if isinstance(s, str) and s))
    for source in sources:
        if '://' not in source and os.path.realpath(source) not in seen:
            name = _file_datum(source, seen=seen)
            if name is not None:
                return name
    for source in sources:
        name = _name_datum([source.split('?')[0]])
        if name is not None:
            return name
    return None


def _members(obj):
    """An xarray DEM and, for a Dataset, its first data variable: the grid Satellite.get_dem() reads from it."""
    names = list(obj.data_vars) if hasattr(obj, 'data_vars') else []
    return [obj, obj[names[0]]] if names else [obj]


def _crs_variable(obj, name):
    """The coordinate or the data variable of a CRS (grid mapping) name, or None."""
    if isinstance(name, str) and name in obj.coords:
        return obj.coords[name]
    if isinstance(name, str) and hasattr(obj, 'data_vars') and name in obj.data_vars:
        return obj.data_vars[name]
    return None


def _xarray_datum(obj):
    members = _members(obj)
    # the object's own attributes, then those of the first data variable of a Dataset, then the CRS variables
    found = [dict(m.attrs) for m in members]
    names = ['spatial_ref', 'crs'] + [m.encoding.get('grid_mapping') or m.attrs.get('grid_mapping') for m in members]
    for name in dict.fromkeys(n for n in names if isinstance(n, str)):
        var = _crs_variable(obj, name)
        if var is not None:
            found.append(dict(var.attrs))
    for a in found:
        name = _attrs_datum(a)
        if name is not None:
            return name
    # a grid opened from a file keeps the file name (xarray's encoding source, the source attribute of
    # utils_tiles.open_dem()): the datum of the file, as for its path. xarray records the file of the variable it
    # read, the DEM variable of this object; utils_tiles records the file of the variable it reads.
    files = {}
    for m in members:
        for source, var in ((m.encoding.get('source'), members[-1].name), (m.attrs.get('source'), None)):
            if isinstance(source, str) and source:
                files.setdefault(source, var)
    for source, var in files.items():
        name = _file_datum(source, var)
        if name is not None:
            return name
    return _name_datum([os.path.splitext(os.path.basename(n))[0] for n in files])


def _sources(obj):
    """The files or URLs an xarray object was read from: the encoding source, then the source attribute, of the
    object and of the first data variable of a Dataset."""
    found = [s for m in _members(obj) for s in (m.encoding.get('source'), m.attrs.get('source'))]
    return list(dict.fromkeys(s for s in found if isinstance(s, str) and s))


def _absolute(source):
    """A local file as its real path (absolute, symbolic links resolved), so one file has one key under any name,
    also through a symlinked folder or a symlink to the file; a URL as it is."""
    return source if '://' in source else os.path.realpath(source)


def _label(dem):
    """
    The DEM named in the fallback warning, the key that prints it once per DEM, and the object that owns the key:
    the file of a path or of an xarray object read from one, otherwise the xarray object itself.
    """
    if isinstance(dem, (str, os.PathLike)):
        from . import utils_tiles
        # the file read for the path ('dem.nc' replaced by 'dem.vrt'), as utils_tiles.open_dem() records it
        path = _absolute(utils_tiles.resolve(os.fspath(dem)))
        return path, ('default', path), None
    if hasattr(dem, 'attrs') and hasattr(dem, 'encoding') and hasattr(dem, 'sizes'):
        sources = _sources(dem)
        if sources:
            path = _absolute(sources[0])
            return path, ('default', path), None
        name = getattr(dem, 'name', None)
        sizes = ', '.join(f'{k}: {v}' for k, v in dem.sizes.items())
        return (f'{type(dem).__name__}{"" if name is None else " " + str(name)} ({sizes})', ('default', id(dem)),
                dem)
    return type(dem).__name__, ('default', type(dem).__name__), None


def declared_datum(dem):
    """
    The vertical datum a DEM declares or its tile names show (steps 2 and 3 of dem_datum()), or None.

    Parameters
    ----------
    dem : str or xarray.DataArray or xarray.Dataset
        A DEM file (NetCDF4 grid or VRT, as read by utils_tiles) or an xarray DEM.
    """
    if dem is None:
        return None
    if isinstance(dem, (str, os.PathLike)):
        from . import utils_tiles
        path = utils_tiles.resolve(os.fspath(dem))
        utils_tiles.check_format(path)
        return _vrt_datum(path) if path.lower().endswith('.vrt') else _netcdf_datum(path)
    if hasattr(dem, 'attrs') and hasattr(dem, 'coords'):
        return _xarray_datum(dem)
    return None


def dem_datum(dem=None, datum=None):
    """
    The vertical datum of a DEM: 'EGM2008', 'EGM96' or 'ellipsoid'.

    Found in this order: the datum argument; the vertical CRS the DEM declares (VRT <SRS>, NetCDF crs_wkt /
    spatial_ref / geoid_name attributes, xarray attributes); the tile names of older toolkit downloads
    (Copernicus_DSM_COG_* EGM2008, SRTM NxxWxxx and ALOS ALPSMLC30_* EGM96); an xarray DEM read from a file, the
    datum of that file. A Dataset is the DEM of its first data variable, a NetCDF4 file of the variable utils_tiles
    reads; a file saved from a DEM read from another file has the datum of that file. A DEM with none of these is
    taken as EGM2008, with a warning once per DEM file or object.

    A DEM opened by xr.open_dataarray() or xr.open_dataset() loses its file in where(), astype(), arithmetic and
    interp(): pass the datum, or open the file with utils_tiles.open_dem() or Tiles().open().

    Parameters
    ----------
    dem : str or xarray.DataArray or xarray.Dataset, optional
        A DEM file (NetCDF4 grid or VRT) or an xarray DEM.
    datum : str or int, optional
        The datum, overriding what the DEM declares: 'EGM2008', 'EGM96', 'ellipsoid' or an EPSG code.

    Examples
    --------
    dem_datum('dem.vrt')                  # 'EGM2008' for a Copernicus download
    dem_datum('my_dem.nc', datum='EGM96')
    """
    if datum is not None:
        return datum_name(datum)
    name = declared_datum(dem)
    if name is not None:
        return name
    label, key, owner = _label(dem)
    _say(key, f'WARNING: {label} has no vertical datum, {DEFAULT_DATUM} is assumed. '
              f'Pass the datum if it is not {DEFAULT_DATUM}.', owner)
    return DEFAULT_DATUM


# -----------------------------------------------------------------------------------------------------------------
# grids and the cubic B-spline
# -----------------------------------------------------------------------------------------------------------------

def grid_path(datum):
    """The bundled grid file of a geoid datum ('EGM2008' or 'EGM96')."""
    name = datum_name(datum)
    if name not in GEOIDS:
        raise ValueError(f'ERROR: no geoid grid for {name}')
    path = os.path.join(DATA_DIR, GEOIDS[name]['file'])
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f'ERROR: geoid grid {path} is missing or empty. Reinstall insardev_toolkit.')
    return path


class _Grid:
    """A global geoid grid: nodes from -90 to 90 and from -180 to 180 (the +180 column repeats -180)."""

    def __init__(self, path):
        from . import utils_tiles
        self.grid = utils_tiles.open_grid(path)
        lat, lon = self.grid.lat, self.grid.lon
        self.step = float(lat[1] - lat[0])
        self.nlat = lat.size
        self.nlon = lon.size - 1
        tol = 1e-6 * self.step
        if (abs(lat[0] + 90) > tol or abs(lat[-1] - 90) > tol or abs(lon[0] + 180) > tol or abs(lon[-1] - 180) > tol
                or abs(float(lon[1] - lon[0]) - self.step) > tol or self.nlon % 2):
            raise ValueError(f'ERROR: {path} is not a global geoid grid')

    def _cols(self, rows, c0, c1):
        """Stored rows (ascending, contiguous) by columns c0..c1-1 of the periodic grid."""
        n = self.nlon
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        if c1 - c0 >= n:
            block = self.grid.read(r0, r1, 0, n)
            return block[:, np.mod(np.arange(c0, c1), n)]
        a, b = c0 % n, c0 % n + (c1 - c0)
        if b <= n:
            return self.grid.read(r0, r1, a, b)
        return np.concatenate([self.grid.read(r0, r1, a, n), self.grid.read(r0, r1, 0, b - n)], axis=1)

    def window(self, r0, r1, c0, c1):
        """
        Node values of rows r0..r1-1 and columns c0..c1-1, float64. Rows past a pole are the rows on the other
        side of it on the opposite meridian (row -k at lon is row k at lon + 180); columns are periodic.
        """
        last = self.nlat - 1
        rows = np.arange(r0, r1)
        south, north = rows < 0, rows > last
        inside = ~(south | north)
        out = np.empty((rows.size, c1 - c0), dtype=np.float64)
        if inside.any():
            out[inside] = self._cols(rows[inside], c0, c1)
        half = self.nlon // 2
        for mask, mirror in ((south, -rows), (north, 2 * last - rows)):
            if mask.any():
                src = mirror[mask]
                block = self._cols(np.arange(src.min(), src.max() + 1), c0 + half, c1 + half)
                out[mask] = block[src - src.min()]
        return out


_GEOID_GRIDS = {}


def _geoid_grid(datum):
    path = grid_path(datum)
    stamp = os.path.getmtime(path)
    cached = _GEOID_GRIDS.get(path)
    if cached is None or cached[0] != stamp:
        _GEOID_GRIDS[path] = cached = (stamp, _Grid(path))
    return cached[1]


def _prefilter(c, axis):
    """Cubic B-spline coefficients along one axis, in place (float64), mirror boundary as scipy.ndimage."""
    c = np.moveaxis(c, axis, 0)
    n = c.shape[0]
    if n < 2:
        return
    z = _POLE
    c *= (1.0 - z) * (1.0 - 1.0 / z)
    # causal initialization, mirror boundary
    zn1 = z ** (n - 1)
    c0 = c[0] + zn1 * c[n - 1]
    zi = z
    for i in range(1, n - 1):
        c0 = c0 + zi * (c[i] + zn1 * c[n - 1 - i])
        zi *= z
    c[0] = c0 / (1.0 - zn1 * zn1)
    for i in range(1, n):
        c[i] += z * c[i - 1]
    # anticausal initialization, mirror boundary
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0)
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])


def _weights(t):
    """The four cubic B-spline weights of the nodes i-1, i, i+1, i+2 at the fraction t past node i."""
    t2 = t * t
    t3 = t2 * t
    return ((1.0 - t) ** 3 / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
            t3 / 6.0)


def _coefficients(grid, r0, r1, c0, c1):
    """Spline coefficients of the nodes r0..r1-1 x c0..c1-1, read with the margin that makes them exact."""
    w = grid.window(r0 - MARGIN, r1 + MARGIN, c0 - MARGIN, c1 + MARGIN)
    _prefilter(w, 0)
    _prefilter(w, 1)
    return w[MARGIN:w.shape[0] - MARGIN, MARGIN:w.shape[1] - MARGIN]


def _index(grid, lat, lon):
    """Fractional node indices: rows from -90, columns from -180 in [0, nlon)."""
    fi = (lat + 90.0) / grid.step
    fj = np.mod((lon + 180.0) / grid.step, grid.nlon)
    # a longitude just below 180 can round to nlon
    fj = np.where(fj >= grid.nlon, fj - grid.nlon, fj)
    return fi, fj


def _check_lat(lat):
    bad = np.abs(lat) > 90.0
    if np.any(bad):
        raise ValueError(f'ERROR: latitude out of [-90, 90]: {lat[bad].ravel()[0]}')


def _points(grid, lat, lon):
    """Geoid heights at points (1-D arrays of finite values), evaluated block by block."""
    fi, fj = _index(grid, lat, lon)
    i0 = np.minimum(np.floor(fi).astype(np.int64), grid.nlat - 1)
    j0 = np.floor(fj).astype(np.int64)
    out = np.empty(lat.size, dtype=np.float64)
    key = (i0 // BLOCK) * (grid.nlon // BLOCK + 1) + j0 // BLOCK
    order = np.argsort(key, kind='stable')
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for group in np.split(order, bounds):
        bi, bj = i0[group[0]] // BLOCK, j0[group[0]] // BLOCK
        r0, c0 = bi * BLOCK - 1, bj * BLOCK - 1
        coef = _coefficients(grid, r0, r0 + BLOCK + 3, c0, c0 + BLOCK + 3)
        # the points of a block in slices, so the temporaries stay small for any number of points
        for k in range(0, group.size, CHUNK):
            g = group[k:k + CHUNK]
            wi, wj = _weights(fi[g] - i0[g]), _weights(fj[g] - j0[g])
            ii, jj = i0[g] - r0 - 1, j0[g] - c0 - 1
            value = np.zeros(g.size)
            for a in range(4):
                for b in range(4):
                    value += wi[a] * wj[b] * coef[ii + a, jj + b]
            out[g] = value
    return out


def _tensor(grid, lat, lon):
    """Geoid heights on the grid lat x lon (1-D arrays of finite values), evaluated block by block."""
    fi, fj = _index(grid, lat, lon)
    i0 = np.minimum(np.floor(fi).astype(np.int64), grid.nlat - 1)
    j0 = np.floor(fj).astype(np.int64)
    out = np.empty((lat.size, lon.size), dtype=np.float64)
    rblocks = {b: np.flatnonzero(i0 // BLOCK == b) for b in np.unique(i0 // BLOCK)}
    cblocks = {b: np.flatnonzero(j0 // BLOCK == b) for b in np.unique(j0 // BLOCK)}
    for bi, ri_block in rblocks.items():
        r0 = bi * BLOCK - 1
        for bj, ci in cblocks.items():
            c0 = bj * BLOCK - 1
            coef = _coefficients(grid, r0, r0 + BLOCK + 3, c0, c0 + BLOCK + 3)
            wj = _weights(fj[ci] - j0[ci])
            jj = j0[ci] - c0 - 1
            rows = max(1, CHUNK // ci.size)
            # output rows in slices, so the temporaries stay small for any grid size
            for k in range(0, ri_block.size, rows):
                ri = ri_block[k:k + rows]
                wi = _weights(fi[ri] - i0[ri])
                ii = i0[ri] - r0 - 1
                # along latitude: (rows, block columns), then along longitude
                part = wi[0][:, None] * coef[ii]
                for a in range(1, 4):
                    part += wi[a][:, None] * coef[ii + a]
                value = wj[0][None, :] * part[:, jj]
                for b in range(1, 4):
                    value += wj[b][None, :] * part[:, jj + b]
                out[np.ix_(ri, ci)] = value
    return out


def geoid_height(lat, lon, datum=DEFAULT_DATUM, grid=False):
    """
    Geoid height N above the WGS84 ellipsoid, in metres: ellipsoidal height = DEM height + N.

    Cubic B-spline of the bundled grid, read window by window (see the module notes for the grids and errors).

    Parameters
    ----------
    lat, lon : array_like
        Latitude [-90, 90] and longitude (any, periodic) in degrees. Arrays that broadcast to one shape, or with
        grid=True the 1-D axes of a grid. NaN reads as NaN.
    datum : str or int, optional
        'EGM2008' (default), 'EGM96', 'ellipsoid' (all zeros) or an EPSG code; see dem_datum() for the datum of a DEM.
    grid : bool, optional
        True: lat and lon are the axes of a grid and the result is (lat.size, lon.size). Default False.

    Returns
    -------
    numpy.ndarray
        Geoid heights, float64.

    Examples
    --------
    geoid_height(19.4, -99.0)                                  # EGM2008 at Mexico City
    geoid_height(dem.lat, dem.lon, dem_datum('dem.vrt'), grid=True)
    """
    name = datum_name(datum)
    if grid:
        lat = np.asarray(lat, dtype=np.float64).ravel()
        lon = np.asarray(lon, dtype=np.float64).ravel()
        shape = (lat.size, lon.size)
    else:
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
        shape = lat.shape
        lat, lon = lat.ravel(), lon.ravel()
    ok_lat, ok_lon = np.isfinite(lat), np.isfinite(lon)
    _check_lat(lat[ok_lat])
    if name != ELLIPSOID and ok_lat.all() and ok_lon.all() and lat.size and lon.size:
        # no NaN coordinates: the result is filled directly
        g = _geoid_grid(name)
        return _tensor(g, lat, lon) if grid else _points(g, lat, lon).reshape(shape)
    out = np.full(shape, np.nan)
    if name == ELLIPSOID:
        if grid:
            out[np.ix_(ok_lat, ok_lon)] = 0.0
        else:
            out.reshape(-1)[ok_lat & ok_lon] = 0.0
        return out
    g = _geoid_grid(name)
    if grid:
        if ok_lat.any() and ok_lon.any():
            out[np.ix_(ok_lat, ok_lon)] = _tensor(g, lat[ok_lat], lon[ok_lon])
        return out
    ok = ok_lat & ok_lon
    if ok.any():
        flat = out.reshape(-1)
        flat[ok] = _points(g, lat[ok], lon[ok])
    return out


# -----------------------------------------------------------------------------------------------------------------
# DEM heights above the ellipsoid
# -----------------------------------------------------------------------------------------------------------------

# attributes that name the geoid of a grid's heights, false for its heights above the ellipsoid
_GEOID_ATTRS = ('geoid_name', 'geopotential_datum_name', 'vertical_datum')


def _horizontal(attrs):
    """CRS attributes without heights: a compound CRS becomes its horizontal CRS, the geoid names are dropped."""
    from pyproj import CRS
    out = {k: v for k, v in attrs.items() if k not in _GEOID_ATTRS}
    for key in ('crs_wkt', 'spatial_ref', 'crs'):
        value = out.get(key)
        value = value.decode() if isinstance(value, bytes) else value
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            crs = CRS.from_user_input(value)
        except Exception:
            continue
        if crs.type_name == 'Bound CRS' and crs.source_crs is not None:
            crs = crs.source_crs
        horizontal = [c for c in crs.sub_crs_list if c.type_name != 'Vertical CRS'] if crs.is_compound else None
        if crs.type_name == 'Vertical CRS' or horizontal == []:
            del out[key]
        elif horizontal:
            out[key] = horizontal[0].to_wkt()
    return out


def _geoid_block(lat, lon, datum):
    """Geoid heights of one block of a lazy grid, from the lat and lon blocks of its axes."""
    return geoid_height(lat, lon, datum, grid=True)


def ellipsoidal_height(dem, datum=None):
    """
    Heights of a DEM above the WGS84 ellipsoid: DEM height + geoid height N of the DEM's vertical datum.

    The result declares its heights: its vertical_datum attribute is 'ellipsoid', so dem_datum() of it is
    'ellipsoid'. The source file and the geoid of the DEM no longer describe these heights and are dropped, from
    the attributes and from the CRS coordinate (a compound CRS becomes its horizontal CRS). A dask DEM stays lazy:
    the geoid is evaluated block by block.

    Parameters
    ----------
    dem : xarray.DataArray
        DEM on 1-D lat and lon coordinates, e.g. utils_tiles.open_dem('dem.vrt').
    datum : str or int, optional
        The vertical datum of the DEM, overriding what it declares; see dem_datum().

    Returns
    -------
    xarray.DataArray
        Heights above the ellipsoid: float32, or the DEM's float type when it is wider (float64). An integer DEM
        (int16, int32, uint16, ...) gives float32.

    Examples
    --------
    h = ellipsoidal_height(utils_tiles.open_dem('dem.vrt'))
    dem_datum(h)                                                # 'ellipsoid'
    """
    import xarray as xr
    if not isinstance(dem, xr.DataArray):
        raise TypeError(f'ERROR: the DEM must be an xarray.DataArray, got {type(dem).__name__}. For a Dataset, '
                        f'pass its DEM variable.')
    for dim in ('lat', 'lon'):
        if dim not in dem.dims or dim not in dem.coords or dem.coords[dim].dims != (dim,):
            raise ValueError(f'ERROR: the DEM must have 1-D lat and lon coordinates, got dims {dem.dims}.')
    name = dem_datum(dem, datum)
    # float32, or a wider float type of the DEM; integers go to float32, not to their common type with float32
    # (float64 for int32)
    dtype = np.result_type(dem.dtype, np.float32) if np.issubdtype(dem.dtype, np.floating) else np.dtype(np.float32)
    if name == ELLIPSOID:
        h = dem.astype(dtype)
    else:
        lat, lon = dem.coords['lat'].values, dem.coords['lon'].values
        if dem.chunks is None:
            n = geoid_height(lat, lon, name, grid=True)
        else:
            import dask.array as da
            chunks = dict(zip(dem.dims, dem.chunks))
            n = da.blockwise(_geoid_block, 'ij', da.from_array(lat, chunks=(chunks['lat'],)), 'i',
                             da.from_array(lon, chunks=(chunks['lon'],)), 'j', dtype=np.float64, datum=name)
        n = xr.DataArray(n, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
        h = (dem.astype(np.float64) + n).astype(dtype)
    h.name = dem.name
    attrs = _horizontal(dem.attrs)
    attrs.pop('source', None)
    attrs['vertical_datum'] = ELLIPSOID
    h.attrs = attrs
    h.encoding = {k: v for k, v in dem.encoding.items() if k == 'grid_mapping'}
    # the CRS coordinate is shared with the DEM: a copy of it loses the geoid
    names = ('spatial_ref', 'crs', dem.encoding.get('grid_mapping'), dem.attrs.get('grid_mapping'))
    for crs_name in dict.fromkeys(n for n in names if isinstance(n, str) and n in h.coords):
        coord = h.coords[crs_name].copy()
        coord.attrs = _horizontal(coord.attrs)
        h = h.assign_coords({crs_name: coord})
    return h
