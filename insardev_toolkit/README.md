# InSAR.dev Toolkit

Utility functions and helper tools for the InSAR.dev ecosystem.

## Features

- DEM downloading (Copernicus GLO-30/GLO-90, NASA SRTM, JAXA ALOS)
- Sentinel-1 burst downloads store each burst's orbit in its annotation, the precise orbit (POEORB) once published, else the newest restituted orbit (RESORB); no orbit file is written; `update` rewrites it when a more precise or newer orbit is published
- Sentinel-1 orbit file (EOF) downloading with `EOF().download`, for bursts downloaded before the orbits were stored in them
- Sentinel-1 SLC downloading from ASF (Alaska Satellite Facility)
- Map tile downloading (XYZ, OpenStreetMap, Google Maps)
- ERA5 pressure-level downloading: the ECMWF reanalysis on 37 pressure levels from the Google ARCO v3 copy (ERA5T for the latest hours), cut to 1-degree tiles
- VMF3 tropospheric grid downloading (TU Wien): 6-hourly grids cut to 1-degree tiles by byte ranges, node heights kept
- IONEX global ionosphere maps downloading (CODE final, NASA CDDIS): one file per UTC day, stored and read as published
- Tropospheric and ionospheric phase of the stores through one evaluator (`utils_atmosphere.py`, with the tables of `utils_tropo.py` and `utils_iono.py`)
- Progress bars with parallel processing support
- NetCDF and xarray data grid operations
- 3D visualization export (VTK format)
- HTTP utilities for downloading and extraction

## Sentinel-1 orbits

`ASF.download` and `CDSE.download` store the orbit of each Sentinel-1 burst in its annotation: `generalAnnotation/orbitList`
with the attribute `source`, the name of the orbit (EOF) file it comes from. The list holds the state vectors of that file from
the last one at or before 120 s before the burst start to the first one at or after 120 s after its stop. The file is the one of
the burst's mission that covers the burst, a precise orbit (POEORB) first, then the newest production. No orbit file is written.
A burst without an orbit (none published, or its orbit index or file cannot be read) is not downloaded: a WARNING names it, and
the next download fetches it. `asf.update(DATADIR)` rewrites the stored orbit of the bursts whose `source` differs from the most
precise, most recent published orbit, such as a restituted orbit (RESORB) once the POEORB is published, and stores it in the
bursts downloaded without one; a burst whose orbit cannot be read keeps its own, with a WARNING; `asf.update(DATADIR,
skip_exist=False)` rewrites the orbit of every burst. With the default `skip_exist=True`, `asf.download` changes no stored file
and makes no network request once every burst is stored; `skip_exist=False` downloads and writes every burst again.
`EOF().download` still downloads the orbit files; the preprocessors read them for a burst without a stored orbit.

## Atmosphere downloads

The three atmosphere downloaders store the values as served; the phases are computed from these stores (section "Atmosphere delays" below), not by the downloaders. Each takes `scenes`: a
DataFrame with `startTime` and `geometry` columns, or an object with such a DataFrame as `.df`. ERA5 and VMF3 take a list
of Sentinel-1 scene names with `geometry=`. IONEX takes the list form too and reads the start time of each name from its
sixth field (the 8+6 digit field). Per scene ERA5 and VMF3 fetch the two bracketing epochs (an epoch on the scene
time gives that epoch alone) for the 1-degree tiles of the scene extent. IONEX fetches the UTC day of each scene instead.
Each download returns a DataFrame with the columns `epoch`, `tile`, `path` and `status` (`new` or `existing`).

```python
from insardev_toolkit import ERA5, IONEX, VMF3

ERA5().download('atmosphere', scenes)                 # hourly epochs, 37 pressure levels, one VRT index
ERA5().download('atmosphere', names, geometry=area)   # the same from a list of scene names
VMF3().download('atmosphere', scenes, product='OP')   # 6-hourly grids: OP, EI or FC; n_jobs=8 by default
IONEX().download('atmosphere', scenes)               # daily CODE final maps through the insar.dev cache, no account
IONEX().download('atmosphere', names)                # the same from a list of scene names
IONEX('<earthdata user>', '<earthdata password>').download('atmosphere', scenes)  # the same from CDDIS with an Earthdata Login account
```

IONEX takes an optional Earthdata Login account in its constructor, `IONEX(username, password)`: without it the day files come through the insar.dev cache (no account), with it they come from CDDIS directly. A day file already held in the download folder is never fetched again, and `download` raises for a username without its password.

Tiles: a tile is a 1-degree cell, lon -180..180, half-open, named by its lower-left corner. One NetCDF4 file per tile and
epoch, in `basedir/<SOURCE>/`; the index is `basedir/<SOURCE>.vrt`, one GDAL multidimensional VRT with the dimensions
`epoch` (seconds since 1970-01-01), `level` (pressure levels, where the model has them), `lat` and `lon`: the lat and lon
arrays are the node values, ascending, and each variable is one array over these dimensions, with one source per tile file
placed by its offset. VMF3 tile names use the lower-left node-centre corner (x.5), such as `S27.5E150.5`. The tiles are
Blosc2-compressed (as `utils_tiles` writes them) and GDAL does not read them (`RasterioIOError: Read failed` on a Blosc2
tile), so the index and the tiles are read by the toolkit's own readers (`utils_epochs`: the index parser and
`utils_epochs.tile_window`).
A box across lon 180 gives two cells, and cells are half-open. IONEX: CODE final maps only, values as published, with no
altitude factor.

Storage, for `basedir` = `atmosphere`. `<tile>` is the lower-left corner of the 1-degree cell: `N40E010` for ERA5 (whole
degrees) and `S27.5E150.5` for VMF3 (x.5 node centres). The VMF3 products are `VMF3_OP`, `VMF3_EI` and `VMF3_FC`:

```
atmosphere/
  ERA5/<tile>_<YYYYmmddTHHMMSS>.nc      one file per 1-degree tile and hour (5 x 5 nodes, edge nodes kept; 37 pressure levels, hPa)
  ERA5.vrt                              one multidimensional index: temperature, specific_humidity, specific_cloud_liquid_water_content and geopotential over (epoch, level, lat, lon), 37 levels; surface_pressure and geopotential_at_surface over (epoch, lat, lon)
  VMF3_OP/<tile>_<YYYYmmddTHHMMSS>.nc   one file per 1-degree tile and epoch (2 x 2 nodes at x.5, node heights kept)
  VMF3_OP.vrt                           one multidimensional index: ah, aw, zhd, zwd and orography over (epoch, lat, lon)
  VMF3_EI/, VMF3_EI.vrt                 the same for the EI product
  VMF3_FC/, VMF3_FC.vrt                 the same for the FC product
  IONEX/<name as published>             one file per UTC day as served (COD0OPSFIN_<yyyy><ddd>0000_01D_01H_GIM.INX.gz; before 2022-11-27 codg<ddd>0.<yy>i.Z), verified in memory before it is kept; the maps are read from it (gzip, or compress through ncompress)
  IONEX.vrt                             one multidimensional index over the day files present, from their headers: tec (Int32, as published) over (epoch, lat, lon); epoch in seconds since 1970, the maps in order
```

Append mode: a tile file that exists and reads is kept and listed as `existing`, and no byte of it is requested. New tiles
are fetched and written, and the VRT is rewritten from all the tile files present, so a larger area or a later date extends
the index. VMF3 and the epoch store fetch the new tiles in parallel: `n_jobs=8` by default, joblib with the loky backend,
and `n_jobs=1` runs in-process. `HTTP.download_jobs` caps the count at two workers per CPU this process may use. IONEX
fetches its days with `n_jobs=8` by default, under the same cap. ERA5.download takes no `n_jobs`. A tile file of the epoch
store that exists but cannot be read stops the store before any fetch or write, with one line starting `ERROR:` that names
the file and says to delete it; nothing is fetched again automatically. ERA5 writes its own tiles (`ERA5.py`). IONEX takes
`skip_exist=True` by default, and `skip_exist=False` downloads the days again. A missing day or a missing account raises
`ValueError` with a one-line message starting `ERROR:`.

Package exports (`insardev_toolkit/__init__.py`): `progressbar_joblib`, `progressbar`, `datagrid`, `EOF`, `NCubeVTK`, `ASF`,
`CDSE`, `ERA5`, `IONEX`, `VMF3`, `PyGMTSAR`, `Tiles`, `XYZTiles`, `MultiInstanceManager`, `download`, `unzip` and `UI`.
The atmosphere modules are not exported by the package; they are imported as modules, for example
`from insardev_toolkit import utils_tropo, utils_iono, utils_atmosphere`.

### Atmosphere delays

`model_class(folder)().open(folder, times, crs, wavelength, factor, heights)` (module `utils_atmosphere`) opens a model folder once per stack and returns an `OpenedModel`. `heights` is `(lo, hi)`, the ellipsoidal metres of the caller's pixels, given as two numbers (the caller's `ele` `actual_range`); no pixel is read. The ERA5 and VMF3 zenith tables cover that range on the global grid of the model's table step (2 m for ERA5, 5 m for VMF3), with one row beyond each end. The default `(-500.0, 9000.0)` is the whole domain. IONEX ignores it. `folder` is the download `basedir`, the folder that holds `ERA5.vrt`, `VMF3_OP.vrt` or `IONEX.vrt`.

`opened.phase()` takes no arguments and returns an `xarray.Dataset` of every opened date. It reads and computes nothing: its variables are lazy, one dask chunk per model tile and group of dates (the dates that share their bracketing model epochs).

```python
from insardev_toolkit.utils_atmosphere import model_class

opened = model_class('atmosphere')().open('atmosphere', times, 32637, 299792458.0 / 5.405e9, 0.78, heights=(lo, hi))
tables = opened.phase()    # ERA5, VMF3: hydro, wet (time, tile, ty, tx, height); mapping (time, tile, ty, tx, component, coef); height_term (height_coef). IONEX: tec (time, lat, lon) and shell (no dims)
```

- ERA5 and VMF3: `hydro` and `wet` are the zenith delays in phase units (-4 pi / wavelength times the delay), over the ellipsoidal heights `height` (m); `utils_tropo.zenith_tiles` builds them. IONEX: `tec` (time, lat, lon) is the zenith TEC phase in radians, and `shell` (no dims) is R + H in metres from the day files, checked to agree (`utils_iono.zenith_tiles`). Each tile is read once per opened model.
- `mapping` (ERA5, VMF3) holds the coefficients `a`, `b`, `c` and `K` of each component (hydrostatic, wet): GMF for ERA5, VMF3 for VMF3. `height_term` is the vector `v, a, b, c` of the height term (`v = 0`: no term). IONEX has no wet term, mapping or height axis.
- The coordinates `x` and `y` (tile, ty, tx) of ERA5 and VMF3 are the nodes in the output CRS (lon and lat when the CRS is None); `lat` and `lon` are the model's nodes; `tile y0` and `tile x0` locate each tile. IONEX has the coordinates time, lat and lon.
- The caller evaluates its own pixels from these tables: bilinear over the 4 nodes of the pixel's cell, linear in the height, with the mapping at the pixel's incidence. The public static helpers `mapping_cf` (the continued fraction), `mapping_form` (the one-division form), `height_term` (the Niell height term) and `pierce_point` (the IONEX single-layer pierce point and its effective incidence; the mapping factor is 1 / cos of that incidence) are the formulas of the callers' kernels (core's Stack, ecef), not for notebooks.
- IONEX is one layer on the shell `R + H`: the caller's mapping is `mapping_cf(0, 0, 0, s)`, `s = sqrt(cos^2 z + (1 - k^2) sin^2 z)`, `k = R / (R + H)`, that is `1 / sqrt(1 - (k sin z)^2)`; the look azimuth and the pierce point are the caller's (`pierce_point`).
- `Stack.load(..., tropo=, iono=)` keeps the folders as attributes of each burst. `Stack.troposphere_phase()` (ERA5 or VMF3) is -4 pi / wavelength times the delay; `Stack.ionosphere_phase(factor=0.78)` is +4 pi / wavelength times it. The corrected phase is the phase minus the returned phase.
- `utils_tropo.model(path)` and `utils_iono.model(path)` name the model of a folder; `verify(path, times)` checks that the stored epochs or maps the times need exist, and reads no values.
- `utils_iono.delay(path, times, wavelength, factor)` and `utils_iono.phase(path, times, wavelength, factor)` take no crs: they give the native IONEX lattice in lon and lat, as the variables `value` (time, y, x) in metres (delay) or radians (phase) and `shell` (no dims, R + H in metres from the day file headers); the coordinates are time, term, epoch, weight, y, x, and the CRS EPSG:4326 (`spatial_ref`). They reproject nothing; the projection of the nodes is the opened model's (`open(..., crs)`).

### Geoid

`utils_geoid.py` evaluates the bundled global geoid (EGM2008, or EGM96 by `datum=`). `geoid_height(lat, lon, datum)` returns the geoid height at points, as a 4 x 4 weighted sum of cubic B-spline coefficients. `geoid_coefficients(datum)` returns those coefficients on the grid's native nodes as a lazy, dask-backed xarray.DataArray (float64, dims lat and lon; index 0 at lat -90 and lon -180; the longitude wraps). The call reads nothing: a computed crop reads only the blocks of nodes it touches, with the margin nodes their coefficients need. The caller crops the coefficients to its own window and evaluates the weights at its points; the whole grid is never computed. Import: `from insardev_toolkit import utils_geoid`.

## License

This software is released under the **BSD 3-Clause License**.
See [LICENSE](./LICENSE) for full terms.

## Contact

- Author: Aleksei Pechnikov
- Email: alexey@pechnikov.dev
- ORCID: https://orcid.org/0000-0001-9626-8615

## Bug Reports

Bug reports and suggestions are welcome via the project's issue tracker.
