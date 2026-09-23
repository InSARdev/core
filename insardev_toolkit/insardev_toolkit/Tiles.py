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

class Tiles(datagrid, progressbar_joblib):
    """
    Download 1-degree raster tiles (DEM, land mask) into NetCDF4 tiles with a VRT index over them.

    Every tile is stored as it is served, one NetCDF4 file per tile under its original name, next to the VRT:
    `dem.vrt` indexes the tiles in `dem/`. A tile is fetched, decoded and verified in a worker and written
    atomically, so nothing larger than one tile is held in memory and an interrupted download resumes where it
    stopped. A tile the server does not have (ocean) is recorded as missing and reads as NaN; a tile that fails
    to download stops the download with an error once every other tile is done.

    No single merged file is written: `filename='dem.nc'` is replaced by `dem.vrt` with a warning. The DEM readers
    still take a merged NetCDF4 grid downloaded before.
    """

    http_timeout = 30
        # Define typical browser headers
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

    @staticmethod
    def _tile_params(product, lon, lat):
        """Template parameters of the tile at (lon, lat), the south-west corner of its 1-degree cell."""
        product1 = int(product[0])
        if product in ['1s', '01s']:
            resolution = '30'
        elif product in ['3s', '03s']:
            resolution = '90'
        else:
            resolution = ''
        return {
            'product': product,
            'lat': lat,
            'lon': lon,
            'product1': product1,
            'resolution': resolution,
            # 1 degree grid
            'SN2':    f'{"S" if lat<0 else "N"}{abs(lat):02}',
            'SN3':    f'{"S" if lat<0 else "N"}{abs(lat):03}',
            'WE3':    f'{"W" if lon<0 else "E"}{abs(lon):03}',
            # standalone 5 degree grid (SRTM and GLO DEM)
            'SN2x5': f'{"S" if lat<0 else "N"}{abs(lat) - (abs(lat) % 5):02}',
            'SN3x5': f'{"S" if lat<0 else "N"}{abs(lat) - (abs(lat) % 5):03}',
            'WE3x5': f'{"W" if lon<0 else "E"}{abs(lon) - (abs(lon) % 5):03}',
            # 5 degree alternative grid when negative grid cells aligning differ (ALOS DEM)
            'SN2x5alt': f'{"S" if lat < 0 else "N"}{(abs(lat) // 5 * 5 if lat >= 0 else (abs(lat) // 5 + 1) * 5):02}',
            'SN3x5alt': f'{"S" if lat < 0 else "N"}{(abs(lat) // 5 * 5 if lat >= 0 else (abs(lat) // 5 + 1) * 5):03}',
            'WE3x5alt': f'{"W" if lon < 0 else "E"}{(abs(lon) // 5 * 5 if lon >= 0 else (abs(lon) // 5 + 1) * 5):03}'
        }

    @staticmethod
    def _tile_name(tile_id, file_id, archive):
        """Stored name of a tile: the served raster's own name, as NetCDF4."""
        import os
        name = os.path.basename(file_id) if file_id else tile_id
        if not file_id and archive:
            name = name[:-(len(archive) + 1)]
        return os.path.splitext(name)[0] + '.nc'

    @staticmethod
    def _mask_nodata(values, nodata):
        """The values as float32 with every nodata value replaced by NaN, unchanged when there is no nodata."""
        import numpy as np
        nodata = [v for v in (nodata if isinstance(nodata, (list, tuple, set)) else [nodata]) if v is not None]
        if not nodata:
            return values
        values = np.asarray(values, dtype=np.float32)
        missing = np.isin(values, np.asarray(nodata, dtype=np.float32))
        if missing.any():
            values = np.where(missing, np.nan, values)
        return values

    @classmethod
    def _decode_geotiff(cls, data, nodata=None):
        """GeoTIFF bytes -> (values, lat, lon) with lat and lon ascending, at the pixel centres; the nodata value
        of the file (GDAL_NODATA tag) and the given one read as NaN."""
        import io
        import numpy as np
        from tifffile import TiffFile
        with TiffFile(io.BytesIO(data)) as tif:
            values = tif.pages[0].asarray()
            geo = tif.geotiff_metadata
            tag = tif.pages[0].tags.get('GDAL_NODATA')
        if values.ndim != 2 or not geo or 'ModelPixelScale' not in geo or 'ModelTiepoint' not in geo:
            raise ValueError(f'ERROR: not a single-band GeoTIFF raster: {values.shape}')
        nodata = [nodata] if nodata is not None else []
        if tag is not None:
            nodata.append(float(str(tag.value).strip('\x00 ')))
        values = cls._mask_nodata(values, nodata)
        sx, sy = float(geo['ModelPixelScale'][0]), float(geo['ModelPixelScale'][1])
        i0, j0, x0, y0 = (float(v) for v in (geo['ModelTiepoint'][0], geo['ModelTiepoint'][1],
                                              geo['ModelTiepoint'][3], geo['ModelTiepoint'][4]))
        # the tie point is a pixel centre for PixelIsPoint rasters and a pixel corner for PixelIsArea ones
        half = 0.0 if int(geo.get('GTRasterTypeGeoKey', 1)) == 2 else 0.5
        ny, nx = values.shape
        lon = x0 + (np.arange(nx) - i0 + half) * sx
        lat = y0 - (np.arange(ny) - j0 + half) * sy
        return values[::-1], lat[::-1], lon

    @classmethod
    def _decode_hgt(cls, data, lon, lat, nodata=-32768):
        """SRTM .hgt bytes of the cell (lon, lat) -> (values, lat, lon) ascending: big-endian int16 samples on
        the arc-second grid including both edges of the cell; the voids (-32768) read as NaN."""
        import numpy as np
        n = int(round((len(data) // 2) ** 0.5))
        if 2 * n * n != len(data):
            raise ValueError(f'ERROR: SRTM tile of {len(data)} bytes is not a square int16 grid')
        values = cls._mask_nodata(np.frombuffer(data, dtype='>i2').reshape(n, n).astype(np.int16), nodata)
        step = 1.0 / (n - 1)
        lats = lat + 1 - np.arange(n) * step
        lons = lon + np.arange(n) * step
        return values[::-1], lats[::-1], lons

    @staticmethod
    def _tile_source(path):
        """The URL a stored tile was fetched from, its `source` attribute."""
        import h5py
        with h5py.File(path, 'r') as f:
            source = f.attrs.get('source')
        return source.decode() if isinstance(source, bytes) else source

    @staticmethod
    def _decode_netcdf(data):
        """NetCDF4 bytes of a lat/lon grid -> (values, lat, lon) ascending, decoded by the CF attributes."""
        import io
        import h5py
        import numpy as np
        from . import utils_tiles
        with h5py.File(io.BytesIO(data), 'r') as f:
            lat = f['lat'][:] if 'lat' in f else f['y'][:]
            lon = f['lon'][:] if 'lon' in f else f['x'][:]
            name = next(n for n in ['z'] + sorted(f) if n in f and getattr(f[n], 'ndim', 0) == 2)
            values = utils_tiles.cf_decode(f[name][:], f[name].attrs)
        if lat[0] > lat[-1]:
            values, lat = values[::-1], lat[::-1]
        if lon[0] > lon[-1]:
            values, lon = values[:, ::-1], lon[::-1]
        return values, lat, lon

    def _download_tile(self, base_url, path_id, tile_id, file_id, archive, filetype, product, lon, lat,
                       tiles_dir, skip_exist=True, retries=30, timeout_second=3, units=None, nodata=None,
                       min_rate='100KB', min_rate_window=60, debug=False):
        """
        Fetch one tile into `tiles_dir` as NetCDF4.

        Returns
        -------
        tuple
            ('tile', path), ('missing', name) when the server does not have the tile, or ('failed', name, error).
        """
        import os
        import io
        import gzip
        import zipfile
        from .HTTP import fetch, NotFound, MAGIC_ZIP, MAGIC_GZIP, MAGIC_TIFF, MAGIC_HDF5
        from . import utils_tiles

        params = self._tile_params(product, lon, lat)
        url = base_url.format(**params)
        path = path_id.format(**params)
        file = file_id.format(**params) if file_id is not None else None
        tile = tile_id.format(**params)
        tile_url = f'{url}/{path}/{tile}'
        name = self._tile_name(tile, file, archive)
        out = os.path.join(tiles_dir, name)
        missing = out[:-3] + '.missing'
        if debug:
            print('DEBUG _download_tile:', tile_url, '->', out)
        if skip_exist and os.path.exists(missing):
            return ('missing', name)
        if skip_exist and os.path.exists(out):
            try:
                utils_tiles.tile_grid(out)
                return ('tile', out)
            except Exception:
                # an unreadable tile is fetched again
                pass
        magic = {'zip': MAGIC_ZIP, 'gz': MAGIC_GZIP}.get(archive) or (MAGIC_HDF5 if filetype == 'netcdf' else MAGIC_TIFF)
        try:
            data = fetch(tile_url, headers=self.headers, magic=magic, retries=retries, timeout_second=timeout_second,
                         min_rate=min_rate, min_rate_window=min_rate_window, debug=debug)
            if archive == 'zip':
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    if file not in zf.namelist():
                        raise ValueError(f'ERROR: the zip archive {tile_url} does not include {file}')
                    data = zf.read(file)
            elif archive == 'gz':
                data = gzip.decompress(data)
            if filetype == 'netcdf':
                values, lats, lons = self._decode_netcdf(data)
            elif data[:4] in MAGIC_TIFF:
                values, lats, lons = self._decode_geotiff(data, nodata=nodata)
            else:
                values, lats, lons = self._decode_hgt(data, lon, lat, nodata=nodata if nodata is not None else -32768)
            utils_tiles.write_tile(out, values, lats, lons, units=units, source=tile_url)
            if os.path.exists(missing):
                os.remove(missing)
            return ('tile', out)
        except NotFound:
            # offshore tiles are missed by design
            open(missing, 'w').close()
            return ('missing', name)
        except Exception as e:
            return ('failed', name, f'{type(e).__name__}: {e}')

    def download(self, base_url, path_id, tile_id, archive, filetype,
                  geometry, file_id=None, filename=None, product='1s',
                  n_jobs=4, joblib_backend='loky', skip_exist=True, retries=30, timeout_second=3,
                  units=None, nodata=None, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Download the tiles covering a geometry.

        Parameters
        ----------
        filename : str or None
            'dem.vrt': the tiles in the folder 'dem/' next to it and the VRT index over them.
            'dem.nc' is replaced by 'dem.vrt' with a warning, no single merged file is written.
            None: nothing is kept, the grid cropped to the geometry is returned in memory.
        nodata : float, optional
            Value of the served tiles that reads as NaN, besides the nodata the tiles declare themselves.
        min_rate : str or float, optional
            Bytes per second a tile download must keep, a size string such as '100KB' or a number, averaged over
            min_rate_window seconds, or it is retried; the rate every one of the n_jobs parallel downloads must
            reach. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default 60.

        Returns
        -------
        xarray.DataArray
            The grid cropped to the geometry, lazy when it is stored in a file.
        """
        import os
        import shutil
        import tempfile
        import numpy as np
        from tqdm.auto import tqdm
        import joblib
        from . import utils_tiles

        assert product in ['1s', '3s'], f'ERROR: product name is invalid: {product}. Expected names are "1s", "3s".'

        if filename is not None and filename.lower().endswith('.nc'):
            vrt = os.path.splitext(filename)[0] + '.vrt'
            print(f'WARNING: {filename} is replaced by {vrt}: the downloaded tiles are stored in '
                  f'{os.path.splitext(vrt)[0]} with the VRT index over them, no single merged file is written. '
                  f'Use {vrt} to read the grid.')
            if os.path.exists(filename):
                print(f'WARNING: {filename} exists and the readers keep using it in place of {vrt}: delete it to '
                      f'use the downloaded tiles.')
            filename = vrt
        if filename is not None and not filename.lower().endswith('.vrt'):
            raise ValueError(f'ERROR: filename must end with .vrt, the index over the tiles stored next to it: '
                             f'{filename}')

        bounds = self.get_bounds(geometry)

        # it produces 4 tiles for cases like (39.5, 39.5, 40.0, 40.0)
        #left, right = int(np.floor(lon_start)), int(np.floor(lon_end))
        #bottom, top = int(np.floor(lat_start)), int(np.floor(lat_end))
        # enhancement to produce a single tile for cases like (39.5, 39.5, 40.0, 40.0)
        lon_start, lat_start, lon_end, lat_end = bounds
        left = np.floor(min(lon_start, lon_end))
        right = np.ceil(max(lon_start, lon_end)) - 1
        bottom = np.floor(min(lat_start, lat_end))
        top = np.ceil(max(lat_start, lat_end)) - 1
        left, right = int(left), int(right)
        bottom, top = int(bottom), int(top)
        #print ('left, right', left, right, 'bottom, top', bottom, top)

        workdir = None
        if filename is None:
            # the tiles are only a step on the way to the grid in memory, removed at the end
            workdir = tempfile.mkdtemp(prefix='tiles_')
            tiles_dir = workdir
            vrt = os.path.join(workdir, 'tiles.vrt')
        else:
            vrt = filename
            tiles_dir = os.path.splitext(filename)[0]
            if os.path.exists(tiles_dir) and not os.path.isdir(tiles_dir):
                raise ValueError(f'ERROR: the tiles of {filename} go into the folder {tiles_dir}, but a file of '
                                 f'that name exists')
        os.makedirs(tiles_dir, exist_ok=True)

        if n_jobs is None or debug == True:
            print ('Note: sequential joblib processing is applied when "n_jobs" is None or "debug" is True.')
            joblib_backend = 'sequential'

        try:
            cells = [(x, y) for x in range(left, right + 1) for y in range(bottom, top + 1)]
            with self.progressbar_joblib(tqdm(desc=f'Downloading Raster Tiles'.ljust(25), total=len(cells))) as progress_bar:
                results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(self._download_tile)\
                                    (base_url, path_id, tile_id, file_id, archive, filetype, product, x, y,
                                     tiles_dir, skip_exist, retries, timeout_second, units, nodata,
                                     min_rate, min_rate_window, debug)\
                                    for x, y in cells)

            missing = sorted(r[1] for r in results if r[0] == 'missing')
            failed = [r for r in results if r[0] == 'failed']
            if missing:
                print(f'NOTE: {len(missing)} of {len(cells)} tiles do not exist on the server (offshore) and read '
                      f'as NaN: {", ".join(missing)}')
            if failed:
                raise Exception(f'ERROR: {len(failed)} of {len(cells)} tiles failed to download; the downloaded '
                                f'tiles are kept, run again to resume: ' +
                                '; '.join(f'{r[1]}: {r[2]}' for r in failed))
            if len(missing) == len(cells):
                raise ValueError(f'ERROR: none of the {len(cells)} tiles exist on the server for the bounds {bounds}.')

            # the index covers every tile of this provider in the folder, so a larger area downloaded later extends
            # the mosaic; tiles of another provider or files that are not tiles are left out of it
            provider = base_url.format(**self._tile_params(product, 0, 0))
            tiles, skipped = [], []
            for name in sorted(os.listdir(tiles_dir)):
                path = os.path.join(tiles_dir, name)
                if not name.endswith('.nc') or not os.path.isfile(path):
                    continue
                try:
                    source = self._tile_source(path)
                except Exception:
                    source = None
                (tiles if source is not None and source.startswith(provider) else skipped).append(path)
            if skipped:
                print(f'NOTE: {len(skipped)} files in {tiles_dir} are not tiles of {provider} and are not indexed: '
                      f'{", ".join(os.path.basename(p) for p in skipped)}')
            try:
                utils_tiles.write_vrt(vrt, tiles)
            except ValueError as e:
                raise ValueError(f'{e}. Copernicus DEM tiles above 50 degrees of latitude have coarser longitude '
                                 f'spacing than the tiles below; consider using the SRTM DEM instead.') from None

            crop = (min(lon_start, lon_end), min(lat_start, lat_end), max(lon_start, lon_end), max(lat_start, lat_end))
            da = utils_tiles.crop(utils_tiles.open_dem(vrt), crop)
            if filename is None:
                da = da.load()
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
        # set CRS and spatial dimensions for rioxarray compatibility
        return self.spatial_ref(da, 4326)

    def open(self, filename):
        """
        Open a downloaded grid lazily, with its CRS set for rioxarray: 'dem.vrt' with the tiles next to it, or a
        merged 'dem.nc' downloaded before. A 'dem.nc' that the download replaced by 'dem.vrt' is read from the VRT.

        land = Tiles().open('land.vrt')
        landmask = np.isfinite(land.rio.reproject(stack.crs))
        """
        from . import utils_tiles
        return self.spatial_ref(utils_tiles.open_dem(filename), 4326)

    def download_landmask(self, geometry, filename=None, product='1s', skip_exist=True, n_jobs=8, retries=30,
                          timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Download land mask tiles.

        from pygmtsar import Tiles
        landmask = Tiles().download_landmask(AOI)
        landmask.plot.imshow()

        Tiles().download_landmask(S1.scan_slc(DATADIR), 'landmask.vrt')
        """
        return self.download(
                         #base_url       = 'https://alexeypechnikov.github.io/gmtlandmask/{product}',
                         base_url       = 'https://gmtlandmask.insar.dev/{product}',
                         path_id        = '{SN2}',
                         tile_id        = '{SN2}{WE3}.nc.gz',
                         archive        = 'gz',
                         filetype       = 'netcdf',
                         geometry       = geometry,
                         filename       = filename,
                         product        = product,
                         n_jobs         = n_jobs,
                         joblib_backend = 'loky',
                         skip_exist     = skip_exist,
                         retries        = retries,
                         timeout_second = timeout_second,
                         min_rate       = min_rate,
                         min_rate_window = min_rate_window,
                         debug          = debug)

    # https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com
    # https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N38_00_E038_00_DEM/Copernicus_DSM_COG_10_N38_00_E038_00_DEM.tif
    def download_dem_glo(self, geometry, filename=None, product='1s', skip_exist=True, n_jobs=8, retries=30,
                         timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Download Copernicus GLO-30/GLO-90 Digital Elevation Model tiles from open AWS storage.

        from pygmtsar import Tiles
        dem = Tiles().download_dem_glo(AOI)
        dem.plot.imshow()

        Tiles().download_dem_glo(S1.scan_slc(DATADIR), 'dem.vrt')
        """
        assert product in ['1s', '3s'], f'ERROR: product name is invalid: {product} for Copernicus GLO DEM. Expected names are "1s", "3s".'
        return self.download(
                         #base_url       = 'https://copernicus-dem-{resolution}m.s3.amazonaws.com',
                         base_url       = 'https://copernicusdem{product1}s.insar.dev',
                         path_id        = 'Copernicus_DSM_COG_{product1}0_{SN2}_00_{WE3}_00_DEM',
                         tile_id        = 'Copernicus_DSM_COG_{product1}0_{SN2}_00_{WE3}_00_DEM.tif',
                         archive        = None,
                         filetype       = 'geotif',
                         geometry       = geometry,
                         filename       = filename,
                         product        = product,
                         skip_exist     = skip_exist,
                         n_jobs         = n_jobs,
                         retries        = retries,
                         timeout_second = timeout_second,
                         units          = 'm',
                         min_rate       = min_rate,
                         min_rate_window = min_rate_window,
                         debug          = debug)

    # aws s3 ls --no-sign-request s3://elevation-tiles-prod/skadi/
    # https://s3.amazonaws.com/elevation-tiles-prod/skadi/N20/N20E000.hgt.gz
    def download_dem_srtm(self, geometry, filename=None, product='1s', skip_exist=True, n_jobs=8, retries=30,
                          timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Download NASA SRTM Digital Elevation Model tiles from open AWS storage.

        from pygmtsar import Tiles
        dem = Tiles().download_dem_srtm(AOI)
        dem.plot.imshow()

        Tiles().download_dem_srtm(S1.scan_slc(DATADIR), 'dem.vrt')
        """
        assert product in ['1s'], f'ERROR: only product="1s" is supported for NASA SRTM DEM.'
        return self.download(
                         #base_url       = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi',
                         base_url       = 'https://srtmdem1s.insar.dev',
                         path_id        = '{SN2}',
                         tile_id        = '{SN2}{WE3}.hgt.gz',
                         archive        = 'gz',
                         filetype       = 'geotif',
                         geometry       = geometry,
                         filename       = filename,
                         product        = product,
                         skip_exist     = skip_exist,
                         n_jobs         = n_jobs,
                         retries        = retries,
                         timeout_second = timeout_second,
                         units          = 'm',
                         nodata         = -32768,
                         min_rate       = min_rate,
                         min_rate_window = min_rate_window,
                         debug          = debug)

    # Define new method to download ALOS DEM
    # https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2404/N025E040/N027E042.zip N027E042/ALPSMLC30_N027E042_DSM.tif
    def download_dem_alos(self, geometry, filename=None, product='1s', skip_exist=True, n_jobs=8, retries=30,
                          timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
            """
            Download JAXA ALOS Digital Elevation Model tiles from open JAXA storage.

            from pygmtsar import Tiles
            Tiles().download_dem_alos(AOI, filename='dem.vrt').plot.imshow(cmap='terrain')
            """
            assert product in ['1s'], f'ERROR: only product="1s" is supported for JAXA ALOS DEM.'
            return self.download(
                             #base_url       = 'https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2404/',
                             base_url       = 'https://alosdem1s.insar.dev',
                             path_id        = '{SN3x5alt}{WE3x5alt}',
                             tile_id        = '{SN3}{WE3}.zip',
                             file_id        = '{SN3}{WE3}/ALPSMLC30_{SN3}{WE3}_DSM.tif',
                             archive        = 'zip',
                             filetype       = 'geotif',
                             geometry       = geometry,
                             filename       = filename,
                             product        = product,
                             skip_exist     = skip_exist,
                             n_jobs         = n_jobs,
                             retries        = retries,
                             timeout_second = timeout_second,
                             units          = 'm',
                             nodata         = -9999,
                             min_rate       = min_rate,
                             min_rate_window = min_rate_window,
                             debug          = debug)

    def download_dem(self, geometry, filename=None, product='1s', provider='GLO', skip_exist=True, n_jobs=8,
                     retries=30, timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """
        Downloads Copernicus or SRTM Digital Elevation Model (DEM) at 30m or 90m resolution.

        Parameters
        ----------
        geometry : object
            The Shapely geometry or GeoPandas object or Xarray object for which to download the DEM.
        filename : str or None, optional
            'dem.vrt' stores the downloaded tiles in the folder 'dem/' next to it with the VRT index over them;
            'dem.nc' is replaced by 'dem.vrt' with a warning; None returns the grid in memory only.
            Default is None.
        product : str, optional
            The resolution of the DEM. Valid options are '1s' (for 30m) and '3s' (for 90m). Default is '1s'.
        provider : str, optional
            The provider of the DEM. Valid options are 'GLO' (for Copernicus Global Land Service), 'SRTM', and 'ALOS'. Default is 'GLO'.
        skip_exist : bool, optional
            If True, keeps the tiles already downloaded. Default is True.
        n_jobs : int, optional
            The number of concurrent download jobs. Default is 8.
        retries : int, optional
            Download attempts per tile. Default is 30.
        timeout_second : float, optional
            Seconds between attempts. Default is 3.
        debug : bool, optional
            If True, prints debugging information. Default is False.

        Returns
        -------
        xarray.DataArray
            The DEM cropped to the geometry.

        Raises
        ------
        AssertionError
            If an invalid provider or product is specified.

        Examples
        --------
        Tiles().download_dem(AOI, filename='dem.vrt', product='1s', provider='GLO', n_jobs=4, skip_exist=True, debug=True)
        """
        kwargs = dict(geometry   = geometry,
                      filename   = filename,
                      product    = product,
                      skip_exist = skip_exist,
                      n_jobs     = n_jobs,
                      retries    = retries,
                      timeout_second = timeout_second,
                      min_rate   = min_rate,
                      min_rate_window = min_rate_window,
                      debug=debug)
        assert provider in ['GLO', 'SRTM', 'ALOS'], f'ERROR: provider name is invalid: {provider}. Expected names are "GLO", "SRTM", "ALOS".'
        if provider == 'SRTM':
            return self.download_dem_srtm(**kwargs)
        elif provider == 'GLO':
            return self.download_dem_glo(**kwargs)
        elif provider == 'ALOS':
            return self.download_dem_alos(**kwargs)
