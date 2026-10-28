# InSAR.dev Toolkit

Utility functions and helper tools for the InSAR.dev ecosystem.

## Features

- DEM downloading (Copernicus GLO-30/GLO-90, NASA SRTM, JAXA ALOS)
- Sentinel-1 orbit file (EOF) downloading
- Sentinel-1 SLC downloading from ASF (Alaska Satellite Facility)
- Map tile downloading (XYZ, OpenStreetMap, Google Maps)
- ERA5 pressure-level downloading: the ECMWF reanalysis on 37 pressure levels from the Google ARCO v3 copy (ERA5T for the latest hours), cut to 1-degree tiles
- VMF3 tropospheric grid downloading (TU Wien): 6-hourly grids cut to 1-degree tiles by byte ranges, node heights kept
- IONEX global ionosphere maps downloading (CODE final, NASA CDDIS): one NetCDF4 file per UTC day
- Tropospheric and ionospheric phase of the stores through one evaluator (`utils_atmosphere.py`, with the tables of `utils_tropo.py` and `utils_iono.py`)
- Progress bars with parallel processing support
- NetCDF and xarray data grid operations
- 3D visualization export (VTK format)
- HTTP utilities for downloading and extraction

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
IONEX().download('atmosphere', scenes, username='<earthdata user>', password='<earthdata password>')  # daily CODE final maps
IONEX().download('atmosphere', names, username='<earthdata user>', password='<earthdata password>')   # the same from a list of scene names
```

IONEX needs the Earthdata Login account in its `username` and `password` arguments, and raises when either is missing.

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
  IONEX/<name as published>             the file as served, verified in memory before it is kept
  IONEX/<yyyymmdd>.nc                   one NetCDF4 file per UTC day, 25 maps, values as published
  IONEX.vrt                             one multidimensional index over the day files present: tec (Int32, as published) over (epoch, lat, lon); epoch in seconds since 1970, the maps in order
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

`model_class(folder)().open(folder, times, crs, wavelength, factor)` (module `utils_atmosphere`) opens a model folder once per stack and returns an `OpenedModel`. `phase(times, y, x, ele, inc, look)` gives the phase of one burst on one output chunk: numpy in, numpy out. `look` (degrees) is needed by IONEX only. A dask array of `ele` (ellipsoidal height, m) or `inc` (incidence, rad) gives a lazy `(time, y, x)` float32 array with the chunks of `ele`, one task per output chunk. `folder` is the download `basedir`, the folder that holds `ERA5.vrt`, `VMF3_OP.vrt` or `IONEX.vrt`.

```python
from insardev_toolkit.utils_atmosphere import model_class

opened = model_class('atmosphere')().open('atmosphere', times, 32637, 299792458.0 / 5.405e9, 0.78)
phase = opened.phase(times, y, x, ele, inc, look)   # ele in metres, inc in radians, look in degrees
```

- Per node and stack time, the zenith delays are tables over the ellipsoidal height (-500 m to 9000 m, 10 m step), in phase units. `utils_tropo.zenith_tiles` builds them for ERA5 and VMF3, `utils_iono.zenith_tiles` for IONEX. Each tile is read once per opened model and computed only when a phase chunk needs it.
- The mapping coefficients (`a`, `b`, `c`) are kept per node and time and applied at the pixel: GMF for ERA5, VMF3 for VMF3. IONEX is a table of one entry with the shell mapping `1 / sqrt(1 - (k sin z)^2)`, `k = R / (R + H)`.
- A pixel is bilinear over the 4 reprojected nodes of its lattice cell. For ERA5 and VMF3 the cell is the pixel's, and each node is evaluated at the pixel height (the tables interpolated linearly) and incidence. IONEX is evaluated at the line-of-sight pierce point of the pixel, which needs the look azimuth: its cell and its mapping (at the effective incidence z' of TN-200 Eq. 22) are those of the pierce point. One kernel serves all three.
- A pixel height outside the table range (ERA5, VMF3) or an incidence outside [0, 90) degrees raises one `ERROR:` line, and so does a non-finite pixel coordinate, for every model. A NaN height or incidence gives NaN for every model; a NaN look gives NaN for IONEX, the only model that uses the look. A pixel deeper below a node's surface than the ERA5 tables reach is NaN, with a warning.
- `Stack.load(..., tropo=, iono=)` keeps the folders as attributes of each burst. `Stack.troposphere_phase()` (ERA5 or VMF3) is -4 pi / wavelength times the delay; `Stack.ionosphere_phase(factor=0.78)` is +4 pi / wavelength times it. The corrected phase is the phase minus the returned phase.
- `utils_tropo.model(path)` and `utils_iono.model(path)` name the model of a folder; `verify(path, times)` checks that the stored epochs or maps the times need exist, and reads no values.
- `utils_iono.delay(path, times, wavelength, factor)` and `utils_iono.phase(path, times, wavelength, factor)` take no crs: they give the native IONEX lattice in lon and lat, every variable kept (value; coordinates time, term, epoch, weight, y, x; attributes shell_k, shell), with the CRS EPSG:4326 written (`spatial_ref`). They reproject nothing; the projection of the pixels is the opened model's (`open(..., crs)`).

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
