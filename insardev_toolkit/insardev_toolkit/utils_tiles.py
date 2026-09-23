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
Geographic raster grids (DEM, land mask) as NetCDF4 files read and written through h5py alone.

Downloads are stored as tiles, one NetCDF4 file per source tile under its original name, with a GDAL VRT file
over them as the index: `dem.vrt` next to the tile folder `dem/`. A tile is compressed by Blosc2 zstd with byte
shuffle in 512 x 512 chunks, which reads DEM windows faster than gzip at the same size. Downloads are not merged
into one file anymore, but a single NetCDF4 grid such as a `dem.nc` downloaded before is read as well.

Every reader of a DEM goes through open_grid(): NetCDF4 grids (.nc, .netcdf, .grd) and VRT files of NetCDF4
tiles, including a VRT written by gdalbuildvrt over the tile folder. A grid stored with descending latitude is
read ascending. GeoTIFF and classic NetCDF3 files are refused with a hint how to convert them.

The tiles need the Blosc2 filter of hdf5plugin, which h5py gets by importing hdf5plugin, as every reader here
does. GDAL, QGIS or netCDF-C read the tiles and the VRT only with HDF5_PLUGIN_PATH set to hdf5plugin.PLUGIN_PATH in
their own environment. The package never sets that variable itself: a process that inherits it before h5py loads
cannot write the tiles anymore (HDF5 loads the filter a second time and the compression fails).
"""
import os
import numpy as np

TILE_CHUNK = 512
NETCDF_EXTENSIONS = ('.nc', '.netcdf', '.grd')
# data variable names tried first, in this order, before the first 2D variable of a NetCDF grid
DATA_NAMES = ('z', 'elevation', 'dem', 'Band1', 'band_data')
# CF attributes of a packed or masked variable
CF_ATTRS = ('scale_factor', 'add_offset', '_FillValue', 'missing_value')


def hdf5_plugins():
    """The hdf5plugin module, whose import registers its filters with h5py."""
    import hdf5plugin
    return hdf5plugin


def _compression():
    hdf5plugin = hdf5_plugins()
    return hdf5plugin.Blosc2(cname='zstd', clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE)


def is_hdf5(path):
    if not os.path.isfile(path):
        return False
    with open(path, 'rb') as f:
        return f.read(8).startswith(b'\x89HDF')


def cf_decode(raw, attrs):
    """
    float32 values of a NetCDF variable decoded as xarray does it: _FillValue and missing_value become NaN, then
    raw * scale_factor + add_offset.

    Parameters
    ----------
    raw : numpy.ndarray
        Values as stored.
    attrs : mapping
        Attributes of the variable; only CF_ATTRS are used.
    """
    def numbers(name):
        return np.asarray(attrs[name]).ravel() if name in attrs else np.empty(0)
    scale, offset = numbers('scale_factor'), numbers('add_offset')
    scale = float(scale[0]) if scale.size else 1.0
    offset = float(offset[0]) if offset.size else 0.0
    packed = scale != 1.0 or offset != 0.0
    values = np.asarray(raw, dtype=np.float64 if packed else np.float32)
    fills = [m for m in np.concatenate([numbers('_FillValue'), numbers('missing_value')]) if not np.isnan(m)]
    if fills:
        missing = np.isin(raw, fills)
        if missing.any():
            values = np.where(missing, np.nan, values)
    if packed:
        values = values * scale + offset
    return values.astype(np.float32, copy=False)


def check_format(path):
    """Raise for any DEM file the readers do not take: GeoTIFF, classic NetCDF3, an unknown extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        raise ValueError(f'ERROR: GeoTIFF DEM files are not supported: {path}. Convert it to NetCDF4 with lat/lon '
                         f'coordinates, e.g. gdal_translate -of netCDF -co FORMAT=NC4 dem.tif dem.nc')
    if ext not in NETCDF_EXTENSIONS + ('.vrt',):
        raise ValueError(f'ERROR: DEM file extension not recognized: {path}. Use NetCDF4 (.nc, .netcdf, .grd) '
                         f'or a VRT file of NetCDF4 tiles (.vrt)')
    if not os.path.exists(path):
        also = f' nor {os.path.splitext(path)[0]}.vrt' if ext == '.nc' else ''
        raise FileNotFoundError(f'ERROR: DEM file not found: {path}{also}')
    if not os.path.isfile(path):
        raise ValueError(f'ERROR: DEM path is not a file: {path}')
    if ext in NETCDF_EXTENSIONS and not is_hdf5(path):
        raise ValueError(f'ERROR: classic NetCDF3 DEM files are not supported: {path}. Convert it to NetCDF4, '
                         f'e.g. nccopy -k nc4 dem.nc dem_nc4.nc')


# -----------------------------------------------------------------------------------------------------------------
# writing
# -----------------------------------------------------------------------------------------------------------------

def _define_grid(f, lat, lon, dtype, shape, units=None, source=None, chunks=None, fillvalue=None):
    """Coordinates, CF grid mapping and the data variable of a NetCDF4 lat/lon grid, in an open h5py file."""
    from rasterio.crs import CRS
    dl = f.create_dataset('lat', data=np.asarray(lat, dtype=np.float64), track_times=False)
    dl.make_scale('lat')
    dl.attrs.update(units='degrees_north', standard_name='latitude', long_name='latitude', axis='Y')
    do = f.create_dataset('lon', data=np.asarray(lon, dtype=np.float64), track_times=False)
    do.make_scale('lon')
    do.attrs.update(units='degrees_east', standard_name='longitude', long_name='longitude', axis='X')
    crs = f.create_dataset('crs', data=np.int32(0), track_times=False)
    crs.attrs.update(grid_mapping_name='latitude_longitude', semi_major_axis=6378137.0,
                     inverse_flattening=298.257223563, crs_wkt=CRS.from_epsg(4326).to_wkt())
    dtype = np.dtype(dtype)
    if fillvalue is None and dtype.kind == 'f':
        fillvalue = np.nan
    chunks = chunks or (min(TILE_CHUNK, shape[0]), min(TILE_CHUNK, shape[1]))
    dz = f.create_dataset('z', shape=shape, dtype=dtype, chunks=chunks, compression=_compression(),
                          fillvalue=fillvalue, track_times=False)
    dz.dims[0].attach_scale(dl)
    dz.dims[1].attach_scale(do)
    dz.attrs['grid_mapping'] = 'crs'
    if units is not None:
        dz.attrs['units'] = units
    f.attrs['Conventions'] = 'CF-1.8'
    if source is not None:
        f.attrs['source'] = source
    return dz


def write_tile(path, values, lat, lon, units=None, source=None):
    """
    Store one tile as a NetCDF4 file, built in memory, verified and written atomically.

    Parameters
    ----------
    path : str
        Output file.
    values : numpy.ndarray
        2D grid on (lat, lon), both ascending; its own dtype is kept.
    lat, lon : numpy.ndarray
        Pixel centre coordinates, ascending.
    units : str, optional
        Units of the values.
    source : str, optional
        Where the tile came from.
    """
    import io
    import h5py
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if values.shape != (lat.size, lon.size):
        raise ValueError(f'ERROR: tile values {values.shape} do not match coordinates ({lat.size}, {lon.size})')
    if lat.size > 1 and not lat[1] > lat[0] or lon.size > 1 and not lon[1] > lon[0]:
        raise ValueError('ERROR: tile coordinates must ascend')
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        dz = _define_grid(f, lat, lon, values.dtype, values.shape, units=units, source=source)
        dz[...] = values
    data = buf.getvalue()
    with h5py.File(io.BytesIO(data), 'r') as f:
        if not np.array_equal(f['z'][:], values, equal_nan=True):
            raise ValueError(f'ERROR: tile verification failed for {path}')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)


def tile_grid(path):
    """Lattice of one tile: first pixel centre and step on each axis, sizes and dtype."""
    import h5py
    with h5py.File(path, 'r') as f:
        lat, lon = f['lat'][:], f['lon'][:]
        dtype = f['z'].dtype
    if lat.size < 2 or lon.size < 2 or not (lat[1] > lat[0] and lon[1] > lon[0]):
        raise ValueError(f'ERROR: tile {path} must have ascending lat and lon with at least 2 pixels each')
    return dict(path=path, lat0=float(lat[0]), dlat=float((lat[-1] - lat[0]) / (lat.size - 1)), nlat=int(lat.size),
                lon0=float(lon[0]), dlon=float((lon[-1] - lon[0]) / (lon.size - 1)), nlon=int(lon.size),
                dtype=np.dtype(dtype))


_GDAL_TYPES = {'float32': 'Float32', 'float64': 'Float64', 'int16': 'Int16', 'uint16': 'UInt16',
               'int32': 'Int32', 'uint8': 'Byte', 'int8': 'Int8'}


def write_vrt(vrt_path, tile_paths):
    """
    Write the VRT index over NetCDF4 tiles, with paths relative to the VRT.

    All tiles must share one pixel size and one lattice. Tiles are listed south to north and west to east, so
    where tiles overlap by a shared edge (SRTM) the northern and eastern one gives the value, the same choice
    the merged grids made before.
    """
    from rasterio.crs import CRS
    infos = sorted((tile_grid(p) for p in tile_paths), key=lambda t: (t['lat0'], t['lon0']))
    if not infos:
        raise ValueError('ERROR: no tiles to index')
    a, e = infos[0]['dlon'], infos[0]['dlat']
    for t in infos:
        if abs(t['dlon'] - a) > 1e-9 * a or abs(t['dlat'] - e) > 1e-9 * e:
            raise ValueError(f'ERROR: tiles of different pixel sizes cannot be combined: {os.path.basename(t["path"])} '
                             f'has {t["dlat"] * 3600:.3f}" x {t["dlon"] * 3600:.3f}", the first one '
                             f'{e * 3600:.3f}" x {a * 3600:.3f}"')
    x0 = min(t['lon0'] - a / 2 for t in infos)
    y0 = max(t['lat0'] + (t['nlat'] - 0.5) * e for t in infos)
    body = []
    nx = ny = 0
    for t in infos:
        xoff = (t['lon0'] - a / 2 - x0) / a
        yoff = (y0 - (t['lat0'] + (t['nlat'] - 0.5) * e)) / e
        if abs(xoff - round(xoff)) > 1e-6 or abs(yoff - round(yoff)) > 1e-6:
            raise ValueError(f'ERROR: tile {os.path.basename(t["path"])} is not on the lattice of the other tiles')
        xoff, yoff = int(round(xoff)), int(round(yoff))
        nx, ny = max(nx, xoff + t['nlon']), max(ny, yoff + t['nlat'])
        rel = os.path.relpath(t['path'], os.path.dirname(os.path.abspath(vrt_path)))
        body.append(f'    <SimpleSource>\n'
                    f'      <SourceFilename relativeToVRT="1">{rel}</SourceFilename>\n'
                    f'      <SourceBand>1</SourceBand>\n'
                    f'      <SourceProperties RasterXSize="{t["nlon"]}" RasterYSize="{t["nlat"]}" '
                    f'DataType="{_GDAL_TYPES.get(t["dtype"].name, "Float32")}" BlockXSize="{TILE_CHUNK}" '
                    f'BlockYSize="{TILE_CHUNK}"/>\n'
                    f'      <SrcRect xOff="0" yOff="0" xSize="{t["nlon"]}" ySize="{t["nlat"]}"/>\n'
                    f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{t["nlon"]}" ySize="{t["nlat"]}"/>\n'
                    f'    </SimpleSource>\n')
    xml = (f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">\n'
           f'  <SRS dataAxisToSRSAxisMapping="2,1">{CRS.from_epsg(4326).to_wkt()}</SRS>\n'
           f'  <GeoTransform>{x0!r}, {a!r}, 0.0, {y0!r}, 0.0, {-e!r}</GeoTransform>\n'
           f'  <VRTRasterBand dataType="Float32" band="1">\n'
           f'    <NoDataValue>nan</NoDataValue>\n' + ''.join(body) +
           f'  </VRTRasterBand>\n</VRTDataset>\n')
    tmp = vrt_path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(xml)
    os.replace(tmp, vrt_path)


# -----------------------------------------------------------------------------------------------------------------
# reading
# -----------------------------------------------------------------------------------------------------------------

class _NetCDFGrid:
    """A NetCDF4 grid file: lat and lon ascending, a file stored the other way round is flipped on read."""

    def __init__(self, path):
        import h5py
        self.path = path
        with h5py.File(path, 'r') as f:
            lat = f['lat'] if 'lat' in f else f['y'] if 'y' in f else None
            lon = f['lon'] if 'lon' in f else f['x'] if 'x' in f else None
            if lat is None or lon is None:
                raise ValueError(f'ERROR: {path} has no lat/lon (or y/x) coordinates')
            self.lat = lat[:].astype(np.float64)
            self.lon = lon[:].astype(np.float64)
            coords = ('lat', 'lon', 'y', 'x')
            names = [n for n in DATA_NAMES if n in f and getattr(f[n], 'ndim', 0) == 2]
            names += sorted(n for n in f if n not in coords and getattr(f[n], 'ndim', 0) == 2)
            if not names:
                raise ValueError(f'ERROR: no 2D data variable found in {path}')
            self.var = names[0]
            if f[self.var].shape != (self.lat.size, self.lon.size):
                raise ValueError(f'ERROR: {path}: variable {self.var} is {f[self.var].shape}, not (lat, lon) '
                                 f'{(self.lat.size, self.lon.size)}')
            self.attrs = {k: f[self.var].attrs[k] for k in CF_ATTRS if k in f[self.var].attrs}
        # the readers work on ascending axes; a north-up file (descending lat) is flipped on read
        self.flip_lat = self.lat.size > 1 and self.lat[1] < self.lat[0]
        self.flip_lon = self.lon.size > 1 and self.lon[1] < self.lon[0]
        if self.flip_lat:
            self.lat = self.lat[::-1]
        if self.flip_lon:
            self.lon = self.lon[::-1]

    @staticmethod
    def _file_slice(n, i0, i1, s, flip):
        """The file slice of the ascending indices i0:i1:s, and whether the block must be reversed."""
        if not flip:
            return slice(i0, i1, s), False
        last = i0 + s * ((i1 - 1 - i0) // s)
        return slice(n - 1 - last, n - i0, s), True

    def read(self, r0, r1, c0, c1, sr=1, sc=1):
        import h5py
        hdf5_plugins()
        rows, flip_r = self._file_slice(self.lat.size, r0, r1, sr, self.flip_lat)
        cols, flip_c = self._file_slice(self.lon.size, c0, c1, sc, self.flip_lon)
        with h5py.File(self.path, 'r') as f:
            block = f[self.var][rows, cols]
        if flip_r:
            block = block[::-1]
        if flip_c:
            block = block[:, ::-1]
        return cf_decode(block, self.attrs)


class _VRTGrid:
    """A VRT of NetCDF4 tiles: one lattice (lat and lon ascending), windows stitched from the tiles they touch."""

    def __init__(self, path):
        import xml.etree.ElementTree as ET
        self.path = path
        root = ET.parse(path).getroot()
        self.nx, self.ny = int(root.get('rasterXSize')), int(root.get('rasterYSize'))
        x0, a, _, y0, _, e = (float(v) for v in root.findtext('GeoTransform').split(','))
        e = -e
        # lattice rows from the south: row g is the centre at y0 - (ny - g - 0.5) * e
        self.lat = y0 - (self.ny - np.arange(self.ny) - 0.5) * e
        self.lon = x0 + (np.arange(self.nx) + 0.5) * a
        base = os.path.dirname(os.path.abspath(path))
        self.sources = []
        # the sources as written here (SimpleSource) and as gdalbuildvrt writes them (ComplexSource with NODATA)
        for src in [s for s in root.iter() if s.tag in ('SimpleSource', 'ComplexSource')]:
            for element in ('ScaleOffset', 'ScaleRatio', 'LUT', 'ColorTableComponent'):
                if src.find(element) is not None:
                    raise ValueError(f'ERROR: {path}: VRT sources with {element} are not supported')
            name = src.find('SourceFilename')
            fname = name.text.strip()
            # GDAL may name a NetCDF variable as NETCDF:"file":variable
            if fname.startswith('NETCDF:'):
                fname = fname[len('NETCDF:'):].rsplit(':', 1)[0].strip('"')
            if name.get('relativeToVRT', '0') == '1':
                fname = os.path.join(base, fname)
            if not os.path.exists(fname):
                raise FileNotFoundError(f'ERROR: {path}: the tile {name.text.strip()} listed in the index is '
                                        f'missing; run the download again to fetch it')
            if os.path.splitext(fname)[1].lower() not in NETCDF_EXTENSIONS or not is_hdf5(fname):
                raise ValueError(f'ERROR: {path}: VRT sources must be NetCDF4 tiles as written by '
                                 f'insardev_toolkit.Tiles, found {name.text.strip()}')
            dst = src.find('DstRect')
            xoff, yoff = int(float(dst.get('xOff'))), int(float(dst.get('yOff')))
            xsize, ysize = int(float(dst.get('xSize'))), int(float(dst.get('ySize')))
            # the tile's rows from the south start at lattice row ny - yoff - ysize
            self.sources.append((fname, self.ny - yoff - ysize, ysize, xoff, xsize))
        if not self.sources:
            raise ValueError(f'ERROR: {path}: the VRT lists no tiles')

    def read(self, r0, r1, c0, c1, sr=1, sc=1):
        import h5py
        hdf5_plugins()
        rows = np.arange(r0, r1, sr)
        cols = np.arange(c0, c1, sc)
        out = np.full((rows.size, cols.size), np.nan, dtype=np.float32)
        if rows.size == 0 or cols.size == 0:
            return out
        for fname, g0, ysize, h0, xsize in self.sources:
            # output rows and columns that fall inside this tile, on the stride grid
            ri = np.nonzero((rows >= g0) & (rows < g0 + ysize))[0]
            ci = np.nonzero((cols >= h0) & (cols < h0 + xsize))[0]
            if ri.size == 0 or ci.size == 0:
                continue
            t_r0, t_c0 = rows[ri[0]] - g0, cols[ci[0]] - h0
            with h5py.File(fname, 'r') as f:
                block = f['z'][t_r0:t_r0 + (ri.size - 1) * sr + 1:sr, t_c0:t_c0 + (ci.size - 1) * sc + 1:sc]
            out[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1] = block
        return out


_GRIDS = {}
_SAID = set()


def _once(key, message):
    """Print a message once per process."""
    if key not in _SAID:
        print(message)
        _SAID.add(key)


def resolve(path):
    """
    The DEM file read for a path. The downloader writes 'dem.vrt' in place of a requested 'dem.nc', so a 'dem.nc'
    that does not exist is read from the 'dem.vrt' next to it, with a warning. An existing 'dem.nc', as downloaded
    by the versions before, is read itself, with a warning when a newer 'dem.vrt' sits next to it.
    """
    if not isinstance(path, str):
        raise TypeError(f'ERROR: the DEM must be a file name, a NetCDF4 grid or a VRT of tiles, got '
                        f'{type(path).__name__}')
    if path.lower().endswith('.nc'):
        vrt = os.path.splitext(path)[0] + '.vrt'
        if not os.path.exists(path):
            if os.path.exists(vrt):
                _once(('replaced', path), f'WARNING: {path} does not exist, {vrt} is read instead: the downloader '
                                          f'stores the tiles with this VRT index in place of a single .nc file.')
                return vrt
        elif os.path.exists(vrt) and os.path.getmtime(vrt) > os.path.getmtime(path):
            _once(('stale', path), f'WARNING: {path} is read, but the newer {vrt} next to it is not: delete '
                                   f'{path} to use the downloaded tiles.')
    return path


def open_grid(path):
    """The grid object of a DEM file, cached per process until the file changes."""
    path = resolve(path)
    check_format(path)
    key = os.path.abspath(path)
    stamp = os.path.getmtime(key)
    cached = _GRIDS.get(key)
    if cached is None or cached[0] != stamp:
        grid = _VRTGrid(key) if key.lower().endswith('.vrt') else _NetCDFGrid(key)
        grid.stamp = stamp
        _GRIDS[key] = cached = (stamp, grid)
    return cached[1]


def _index_range(coords, lo, hi):
    """
    First and last+1 index of the pixel centres inside [lo, hi], or None.

    A centre within a millionth of a pixel of a bound counts as inside: coordinates computed on different paths
    (a tile's own axis, the VRT lattice, a merged grid) differ in the last bits, and a bound that falls on a
    pixel centre -- a round-number area on an arc-second grid -- must not keep or drop that row by rounding.
    """
    tol = 1e-6 * abs(float(coords[1] - coords[0])) if coords.size > 1 else 0.0
    idx = np.nonzero((coords >= lo - tol) & (coords <= hi + tol))[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def crop(da, bounds):
    """Crop a DataArray on (lat, lon) to (lon_min, lat_min, lon_max, lat_max) by the rule of _index_range()."""
    lon_min, lat_min, lon_max, lat_max = bounds
    rows = _index_range(da.lat.values, lat_min, lat_max)
    cols = _index_range(da.lon.values, lon_min, lon_max)
    if rows is None or cols is None:
        raise ValueError(f'ERROR: the bounds {bounds} miss the grid')
    return da.isel(lat=slice(*rows), lon=slice(*cols))


def read_dem(path, bounds=None, stride=1):
    """
    Read a DEM window: the pixel centres inside the bounds, both ends included.

    Parameters
    ----------
    path : str
        NetCDF4 grid (.nc, .netcdf, .grd) or VRT of NetCDF4 tiles (.vrt).
    bounds : tuple, optional
        (lon_min, lat_min, lon_max, lat_max). The whole grid when None.
    stride : int, optional
        Keep every stride-th pixel on both axes, starting from the first selected one. Default 1.

    Returns
    -------
    tuple or None
        (values float32, lat, lon) with both coordinates ascending, or None when the bounds miss the grid. The
        cells of a VRT that no tile covers read as NaN.
    """
    grid = open_grid(path)
    if bounds is None:
        rows, cols = (0, grid.lat.size), (0, grid.lon.size)
    else:
        lon_min, lat_min, lon_max, lat_max = bounds
        rows = _index_range(grid.lat, lat_min, lat_max)
        cols = _index_range(grid.lon, lon_min, lon_max)
        if rows is None or cols is None:
            return None
    s = max(1, int(stride))
    values = grid.read(rows[0], rows[1], cols[0], cols[1], s, s)
    return values, grid.lat[rows[0]:rows[1]:s], grid.lon[cols[0]:cols[1]:s]


def dem_step(path):
    """(dlat, dlon) pixel size of a DEM file, absolute."""
    grid = open_grid(path)
    return abs(float(grid.lat[1] - grid.lat[0])), abs(float(grid.lon[1] - grid.lon[0]))


def _read_block(path, stamp, block_info=None):
    (r0, r1), (c0, c1) = block_info[None]['array-location']
    grid = open_grid(path)
    if grid.stamp != stamp:
        # a re-download rewrites the index; the lazy array's lattice would no longer be the file's
        raise RuntimeError(f'ERROR: {path} changed since it was opened; open it again')
    return grid.read(r0, r1, c0, c1)


def open_dem(path, chunks=2048):
    """
    Open a DEM file lazily as an xarray DataArray on (lat, lon), read through h5py block by block.

    Parameters
    ----------
    path : str
        NetCDF4 grid (.nc, .netcdf, .grd) or VRT of NetCDF4 tiles (.vrt).
    chunks : int, optional
        Dask chunk size on both axes. Default 2048.
    """
    import dask.array as da
    import xarray as xr
    grid = open_grid(path)
    ny, nx = grid.lat.size, grid.lon.size
    c = int(chunks)
    shape_chunks = (tuple(min(c, ny - i) for i in range(0, ny, c)), tuple(min(c, nx - i) for i in range(0, nx, c)))
    data = da.map_blocks(_read_block, grid.path, grid.stamp, dtype=np.float32, chunks=shape_chunks,
                         meta=np.empty((0, 0), dtype=np.float32))
    return xr.DataArray(data, coords={'lat': grid.lat, 'lon': grid.lon}, dims=('lat', 'lon'), name='z')
