# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
from .datagrid import datagrid
from .progressbar_joblib import progressbar_joblib

class XYZTiles(datagrid, progressbar_joblib):
    """
    Download XYZ map tiles into a folder of tiles as served, with a VRT index over them.

    A tile is stored exactly as the service sends it, one file per tile under its own name, `gmap.vrt` indexing
    the tiles in `gmap/{z}/{x}/{y}.png` (or .jpg). The tiles are Web Mercator, so the index is in EPSG:3857,
    where they all share one lattice; in latitude they do not, as their step grows towards the poles. Paletted
    tiles, which most map services serve, are expanded to RGB by the index.

    A tile is fetched, verified and written atomically, so an interrupted download resumes where it stopped, and
    the mosaic returned is always read back from the stored tiles and reprojected, whether it was just downloaded
    or downloaded before. No single merged file is written: `filename='gmap.nc'` is replaced by `gmap.vrt` with a
    warning.
    """

    http_timeout = 30
    # half of the Web Mercator (EPSG:3857) extent in meters, the map edge at the equator
    mercator_extent = 20037508.342789244
    # OSM tiles downloading requires the browser header (otherwise, server returns HTTP 403 code)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.example.com',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
    }

    def download_googlemaps(self, geometry, zoom, filename=None, **kwargs):
        kwargs['url'] = 'https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}'
        return self.download(geometry, zoom, filename, **kwargs)
        
    def download_googlesatellite(self, geometry, zoom, filename=None, **kwargs):
        kwargs['url'] = 'https://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}'
        return self.download(geometry, zoom, filename, **kwargs)

    def download_googlesatellitehybrid(self, geometry, zoom, filename=None, **kwargs):
        kwargs['url'] = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
        return self.download(geometry, zoom, filename, **kwargs)

    def download_openstreetmap(self, geometry, zoom, filename=None, **kwargs):
        kwargs['url'] = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        return self.download(geometry, zoom, filename, **kwargs)

    def download_openrailwaymap(self, geometry, zoom, filename=None, background='Mapnik', **kwargs):
        import xarray as xr
        if background is None:
            # [abc]
            kwargs['url'] = 'https://a.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png'
            tiles = self.download(geometry, zoom, filename, **kwargs)
            tiles = xr.concat([tiles, 255*(tiles.sum('band') != 0)], dim='band')
            return tiles
        elif background == 'Mapnik':
            kwargs['url'] = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        else:
            raise ValueError('Expected background "Mapnik" or None')
        return self.download(geometry, zoom, filename, **kwargs)

    @staticmethod
    def deg2num(lat_deg, lon_deg, zoom):
        """The tile holding a point at a zoom level."""
        import math
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat_deg))) / math.pi) / 2.0 * n)
        return (x, y)

    @staticmethod
    def _tile_extension(data):
        """The file type of a served tile, from its first bytes."""
        for magic, extension in ((b'\x89PNG', 'png'), (b'\xff\xd8\xff', 'jpg'), (b'GIF8', 'gif'), (b'RIFF', 'webp')):
            if data.startswith(magic):
                return extension
        raise ValueError(f'ERROR: the tile is not an image: {bytes(data[:8])!r}')

    def _download_tile(self, url, x, y, zoom, tiles_dir, skip_exist=True, retries=30, timeout_second=3,
                       min_rate='100KB', min_rate_window=60, debug=False):
        """
        Fetch one tile as served into `tiles_dir/{z}/{x}/{y}.<type>`.

        Returns
        -------
        tuple
            ('tile', path, x, y), ('missing', name) when the service does not have the tile, or ('failed', name, error).
        """
        import os
        from glob import glob
        from .HTTP import fetch, NotFound, MAGIC_IMAGE
        from .utils_files import exists, write_file

        stem = os.path.join(tiles_dir, str(zoom), str(x), str(y))
        name = f'{zoom}/{x}/{y}'
        tile_url = url.format(x=x, y=y, z=zoom)
        if debug:
            print('DEBUG _download_tile:', tile_url, '->', stem)
        if skip_exist:
            if os.path.exists(stem + '.missing'):
                return ('missing', name)
            for stored in glob(stem + '.*'):
                if stored.endswith(('.missing', '.tmp')):
                    continue
                # an empty tile raises
                exists(stored)
                with open(stored, 'rb') as f:
                    head = f.read(8)
                if head.startswith(MAGIC_IMAGE):
                    return ('tile', stored, x, y)
                # a stored file that is not an image is fetched again
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        try:
            data = fetch(tile_url, headers=self.headers, magic=MAGIC_IMAGE, retries=retries,
                         timeout_second=timeout_second, min_rate=min_rate, min_rate_window=min_rate_window, debug=debug)
            path = f'{stem}.{self._tile_extension(data)}'
            write_file(path, data)
            if os.path.exists(stem + '.missing'):
                os.remove(stem + '.missing')
            return ('tile', path, x, y)
        except NotFound:
            # a service does not always cover every tile of the area
            open(stem + '.missing', 'w').close()
            return ('missing', name)
        except Exception as e:
            return ('failed', name, f'{type(e).__name__}: {e}')

    @classmethod
    def _write_vrt(cls, vrt_path, tiles, zoom):
        """
        Write the VRT index over the stored tiles: EPSG:3857, paths relative to the VRT, RGB.

        All tiles must have one pixel size. A paletted tile, as most map services serve, is expanded to RGB by
        taking one component of its color table per band, the way `gdal_translate -expand rgb` writes it.

        Parameters
        ----------
        vrt_path : str
            Output file.
        tiles : list of tuple
            (path, x, y) of every tile.
        zoom : int
            Zoom level of the tiles, which sets the pixel size.
        """
        import os
        import rasterio
        from rasterio.crs import CRS
        from .utils_files import write_file

        infos = []
        for path, x, y in sorted(tiles, key=lambda tile: (tile[2], tile[1])):
            with rasterio.open(path) as src:
                try:
                    # a single band without a color table (a greyscale tile) is repeated into the RGB bands
                    palette = src.count == 1 and bool(src.colormap(1))
                except ValueError:
                    palette = False
                infos.append(dict(path=path, x=x, y=y, width=src.width, height=src.height, count=src.count,
                                  palette=palette))
        if not infos:
            raise ValueError('ERROR: no tiles to index')
        sizes = {(t['width'], t['height']) for t in infos}
        if len(sizes) > 1:
            raise ValueError(f'ERROR: tiles of different pixel sizes cannot be combined: {sorted(sizes)}')
        width, height = sizes.pop()
        bands = 4 if all(t['count'] == 4 for t in infos) else 3
        x_min, y_min = min(t['x'] for t in infos), min(t['y'] for t in infos)
        nx = (max(t['x'] for t in infos) - x_min + 1) * width
        ny = (max(t['y'] for t in infos) - y_min + 1) * height
        # the tiles of a zoom level split the Web Mercator extent evenly, so they share one lattice
        tile_size = 2 * cls.mercator_extent / 2 ** zoom
        x0 = -cls.mercator_extent + x_min * tile_size
        y0 = cls.mercator_extent - y_min * tile_size
        body = []
        for band in range(1, bands + 1):
            sources = []
            for t in infos:
                relative = os.path.relpath(t['path'], os.path.dirname(os.path.abspath(vrt_path)))
                xoff, yoff = (t['x'] - x_min) * width, (t['y'] - y_min) * height
                element = 'ComplexSource' if t['palette'] else 'SimpleSource'
                component = f'      <ColorTableComponent>{band}</ColorTableComponent>\n' if t['palette'] else ''
                sources.append(
                    f'    <{element}>\n'
                    f'      <SourceFilename relativeToVRT="1">{relative}</SourceFilename>\n'
                    f'      <SourceBand>{1 if t["palette"] or t["count"] < band else band}</SourceBand>\n'
                    f'      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Byte" '
                    f'BlockXSize="{width}" BlockYSize="{height}"/>\n'
                    f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
                    f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{width}" ySize="{height}"/>\n'
                    f'{component}'
                    f'    </{element}>\n')
            body.append(f'  <VRTRasterBand dataType="Byte" band="{band}">\n'
                        f'    <ColorInterp>{["Red", "Green", "Blue", "Alpha"][band - 1]}</ColorInterp>\n'
                        + ''.join(sources) + '  </VRTRasterBand>\n')
        xml = (f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">\n'
               f'  <SRS>{CRS.from_epsg(3857).to_wkt()}</SRS>\n'
               f'  <GeoTransform>{x0!r}, {tile_size / width!r}, 0.0, {y0!r}, 0.0, {-tile_size / height!r}</GeoTransform>\n'
               + ''.join(body) + '</VRTDataset>\n')
        write_file(vrt_path, xml)

    def _read_vrt(self, vrt_path, bounds, geometry, fill_value):
        """The mosaic of the indexed tiles, cropped to the bounds and reprojected as the geometry asks."""
        import geopandas as gpd
        import rioxarray as rio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds

        # the geometry's own projection when it has one, lat/lon otherwise
        try:
            crs = geometry.crs if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)) else geometry.rio.crs
        except Exception:
            crs = None
        crs = crs if crs is not None else 4326

        with rio.open_rasterio(vrt_path) as src:
            # crop in Web Mercator first, so only the tiles of the area are read; an area smaller than a pixel of
            # this zoom level is expanded to the pixels around it instead of coming back empty
            da = src.rio.clip_box(*transform_bounds(4326, 3857, *bounds), auto_expand=True)
            # nearest resampling preserves the map colors
            da = da.rio.reproject(crs, resampling=Resampling.nearest).load()
        if 'band' in da.coords:
            da = da.drop_vars('band')
        if da.rio.crs.is_geographic:
            da = da.rename({'y': 'lat', 'x': 'lon'})
            if da.lat.size > 1 and da.lat[1] < da.lat[0]:
                da = da.reindex(lat=da.lat[::-1])
            cropped = da.sel(lat=slice(bounds[1], bounds[3]), lon=slice(bounds[0], bounds[2]))
            # an area smaller than a pixel selects nothing, so the pixels around it are kept instead
            if cropped.lat.size and cropped.lon.size:
                da = cropped
            da = da.rio.set_spatial_dims(y_dim='lat', x_dim='lon', inplace=True).rio.write_crs(crs, inplace=True)
        da = da.rename('colors')
        # replace background pixels (255) with fill_value for dark theme support
        fill = da.attrs.get('_FillValue', 255)
        if fill_value != fill:
            mask = (da.values == fill)
            if mask.any():
                da.values[mask] = fill_value
            da.attrs['_FillValue'] = fill_value
        return da

    def download(self, geometry, zoom, filename=None, url='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                 n_jobs=8, skip_exist=True, fill_value=255, retries=30, timeout_second=3, min_rate='100KB',
                 min_rate_window=60, debug=False):
        """
        Downloads map tiles for a specified geometry and zoom level from a given tile map service.

        Every tile is stored as the service sends it, under its own name, with a VRT index over them; the mosaic
        returned is read back from the stored tiles and reprojected. A tile the service does not have is recorded
        as missing and left empty; a tile that fails to download stops the download with an error once every other
        tile is done.

        Parameters
        ----------
        geometry : object
            The area for which to download map tiles, defined as a Shapely geometry, GeoPandas object, or Xarray object.
            This is typically referred to as an Area of Interest (AOI).
        zoom : int
            The zoom level for the map tiles. Higher zoom levels correspond to higher resolution.
        filename : str or None, optional
            'gmap.vrt' stores the tiles in the folder 'gmap/' next to it with the VRT index over them;
            'gmap.nc' is replaced by 'gmap.vrt' with a warning; None keeps no files. Default is None.
        url : str, optional
            The URL template of the tile map service. The placeholders {x}, {y}, {z} should be present in the URL.
            Default is Google Satellite Hybrid 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'.
        n_jobs : int, optional
            The number of concurrent download jobs. Default is 8.
        skip_exist : bool, optional
            If True, keeps the tiles already downloaded. Default is True.
        fill_value : int, optional
            Value to replace the background (255) with, for dark themes. Default is 255, which changes nothing.
        retries : int, optional
            Download attempts per tile. Default is 30.
        timeout_second : float, optional
            Seconds between attempts. Default is 3.
        min_rate : str or float, optional
            Bytes per second a tile download must keep, a size string such as '100KB' or a number, averaged over
            min_rate_window seconds, or it is retried. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default 60.
        debug : bool, optional
            If True, prints debugging information. Default is False.

        Returns
        -------
        Xarray
            An Xarray object containing the RGB raster data for the downloaded map tiles. This object can be used for further analysis and visualization.

        Examples
        --------
        # Download map tiles at zoom level 10
        from pygmtsar import XYZTiles
        gmap = XYZTiles().download(AOI, zoom = 10)
        gmap.plot.imshow()

        XYZTiles().download(AOI, zoom=10, filename='gmap.vrt')
        """
        import os
        import shutil
        import tempfile
        from glob import glob
        from tqdm.auto import tqdm
        import joblib
        from .utils_files import exists

        if filename is not None and filename.lower().endswith('.nc'):
            vrt = os.path.splitext(filename)[0] + '.vrt'
            print(f'WARNING: {filename} is replaced by {vrt}: the downloaded tiles are stored in '
                  f'{os.path.splitext(vrt)[0]} with the VRT index over them, no single merged file is written. '
                  f'Use {vrt} to read the tiles.')
            filename = vrt
        if filename is not None and not filename.lower().endswith('.vrt'):
            raise ValueError(f'ERROR: filename must end with .vrt, the index over the tiles stored next to it: '
                             f'{filename}')

        bounds = self.get_bounds(geometry)
        lon_start, lat_start, lon_end, lat_end = bounds
        # the tile grid runs north to south, so the northern bound gives the first tile row
        x_start, y_start = self.deg2num(lat_end, lon_start, zoom)
        x_end, y_end = self.deg2num(lat_start, lon_end, zoom)

        workdir = None
        if filename is None:
            # the tiles are only a step on the way to the mosaic in memory, removed at the end
            workdir = tempfile.mkdtemp(prefix='xyztiles_')
            tiles_dir = workdir
            vrt = os.path.join(workdir, 'tiles.vrt')
        else:
            vrt = filename
            tiles_dir = os.path.splitext(filename)[0]
        os.makedirs(tiles_dir, exist_ok=True)

        joblib_backend = 'sequential' if n_jobs is None or debug else None

        try:
            # every tile of the zoom in the folder is indexed below; an empty one raises before any download
            stored = [path for path in glob(os.path.join(tiles_dir, str(zoom), '*', '*'))
                      if not path.endswith(('.missing', '.tmp'))]
            for path in stored:
                exists(path)
            cells = [(x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1)]
            with self.progressbar_joblib(tqdm(desc='Downloading Map Tiles'.ljust(25), total=len(cells))) as progress_bar:
                results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(self._download_tile)\
                                    (url, x, y, zoom, tiles_dir, skip_exist, retries, timeout_second, min_rate,
                                     min_rate_window, debug)\
                                    for x, y in cells)

            missing = sorted(r[1] for r in results if r[0] == 'missing')
            failed = [r for r in results if r[0] == 'failed']
            if missing:
                print(f'NOTE: {len(missing)} of {len(cells)} tiles do not exist on the server and are left empty: '
                      f'{", ".join(missing)}')
            if failed:
                raise Exception(f'ERROR: {len(failed)} of {len(cells)} tiles failed to download; the downloaded '
                                f'tiles are kept, run again to resume: ' +
                                '; '.join(f'{r[1]}: {r[2]}' for r in failed))
            if len(missing) == len(cells):
                raise ValueError(f'ERROR: none of the {len(cells)} tiles exist on the server for the bounds {bounds}.')

            # the index covers every tile of this zoom in the folder: a larger area downloaded later extends it
            tiles = [(path, int(path.split(os.sep)[-2]), int(os.path.splitext(os.path.basename(path))[0]))
                     for path in sorted(glob(os.path.join(tiles_dir, str(zoom), '*', '*')))
                     if not path.endswith(('.missing', '.tmp'))]
            self._write_vrt(vrt, tiles, zoom)
            da = self._read_vrt(vrt, bounds, geometry, fill_value)
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
        return da
