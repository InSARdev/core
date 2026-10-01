# Geoid grids

Geoid heights N above the WGS84 ellipsoid, for converting DEM heights to ellipsoidal heights (h = H + N).
They are read by `insardev_toolkit.utils_geoid` (`geoid_height()`, `dem_datum()`), one window at a time.

| File | Geoid | Grid | Vertical CRS | Bytes | SHA-256 |
|---|---|---|---|---|---|
| `egm2008_2.5min.nc` | NGA EGM2008 | 2.5' (4321 x 8641 nodes) | EPSG:3855 | 29,301,280 | `622ab4bed5f634e7c704eb102442ab707fa818dff75493eb904f86008223946a` |
| `egm96_7.5min.nc` | NGA EGM96 | 7.5' (1441 x 2881 nodes) | EPSG:5773 | 4,304,758 | `a8d3df58087d65a0435f98912dc8259b9ec7ddf35e1de26062ee7473067f6ac5` |

License: public domain, NGA public information. See `LICENSE` in this folder. The BSD-3-Clause license of
insardev_toolkit does not apply to these two files.

`egm2008_2.5min.py` and `egm96_7.5min.py` are the programs that wrote the two grids (see Conversion).
`egm2008_2.5min.py` converts PROJ's `us_nga_egm08_25.tif`. `egm96_7.5min.py` is NGA's program F477.F ported to
numpy. Each program holds its own grid writer and runs alone, with only its inputs. They document the grids;
insardev_toolkit does not import them. They are toolkit code with the toolkit's license header.

## Format

- NetCDF4 (HDF5), CF-1.8, read by `utils_tiles.read_dem()` like a DEM.
- Nodes from -90 to 90 in latitude and from -180 to 180 in longitude, both ascending. The +180 column repeats -180.
- Variable `z`: int16 with `scale_factor` 0.003 m and `add_offset`. The 3 mm step rounds by up to 1.5 mm.
- Compression: Blosc2 zstd level 9 with byte shuffle, 512 x 512 chunks. h5py reads it after `import hdf5plugin`.
  GDAL and netCDF-C need `HDF5_PLUGIN_PATH` set to `hdf5plugin.PLUGIN_PATH` in their own environment.
- Attribute `history`: the program in this folder that wrote the file, and its inputs by name with their SHA-256.
  It holds no time, host or path of the run, so a rebuild in any folder gives the same bytes.

## Sources

### EGM2008

- File: PROJ-data `us_nga_egm08_25.tif`, https://cdn.proj.org/us_nga_egm08_25.tif
  - 80,585,622 bytes, SHA-256 `4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a`.
  - PROJ-data: "License: Public Domain", source NGA. It was produced by GeographicLib from the EGM2008 model.
    https://github.com/OSGeo/PROJ-data/blob/master/us_nga/us_nga_README.txt and
    https://github.com/OSGeo/PROJ-data/blob/master/copyright_and_licenses.csv
  - PROJ's build script: GeographicLib 1.49, `GeoidToGTX egm2008 24 egm08_25.gtx`,
    https://raw.githubusercontent.com/OSGeo/proj-datumgrid/master/world/build_egm08_25_gtx.sh
- NGA EGM2008 page: https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
  (archived: https://web.archive.org/web/20260105104748/https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84)

### EGM96

- NGA EGM96 page: http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html
  (archived: https://web.archive.org/web/20061103210455/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html)
- Files used, NGA originals from the archive:

| File | Archived URL | SHA-256 |
|---|---|---|
| `egm96.z` (coefficients EGM96) | https://web.archive.org/web/20061120214908/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.z | `6e4fc4f2a00bf7b5cd91bed594ce636022cc844d81a4cfa5f76d83de490c68cd` |
| `corrcoef.z` (coefficients CORRCOEF) | https://web.archive.org/web/20070112050926/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/corrcoef.z | `45b61f280041e39a72d3f654e65124b33eb4e0a5f756b7f43112b855353cef3e` |
| `f477.f` (NGA synthesis program, ported in `egm96_7.5min.py`) | https://web.archive.org/web/20061029210851/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/f477.f | `e817d29b8c8db573946c67484b43df85dd76563ee7cb27fc565ca55c1e49924b` |
| `ww15mgh.grd.z` (NGA 15' grid, check only) | https://web.archive.org/web/20061120214936/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/ww15mgh.grd.z | `5b789d8c2a3163cc48f7d8f194ebe6fcbfcde8417ed687b8f113fe76d29508e9` |

## Conversion

- EGM2008: program `egm2008_2.5min.py` in this folder. Its header names the source file `us_nga_egm08_25.tif`
  with its URL and SHA-256, and its origin: PROJ built it with GeographicLib's GeoidToGTX from NGA's EGM2008 model.
  The GeoTIFF nodes, read by GDAL (rasterio), are stored unchanged apart from the 3 mm step (`add_offset`
  -10.542 m). No resampling. Run it with PROJ's file:

      curl -O https://cdn.proj.org/us_nga_egm08_25.tif
      python3.13 egm2008_2.5min.py us_nga_egm08_25.tif egm2008_2.5min.nc

  The output is the shipped `egm2008_2.5min.nc` byte for byte (SHA-256 above), checked with numpy 2.5.2, rasterio
  1.4.3 (GDAL 3.9.3), h5py 3.14.0 (HDF5 1.14.6) and hdf5plugin 7.1.0 on macOS arm64.
- EGM96: program `egm96_7.5min.py` in this folder. Its header names NGA's original program F477.F and the
  coefficient files EGM96 and CORRCOEF, with their NGA and archived URLs. It computes the geoid heights at the 7.5'
  nodes by F477.F ported to numpy. The port is the same algorithm: EGM96 to degree 360 minus the WGS84 even zonals
  J2..J10 (DHCSIN), the WGS84(G873) constants and normal gravity (RADGRA), NGA's Legendre recursion (LEGFDN), the
  CORRCOEF height-anomaly-to-undulation correction and the -0.53 m term (HUNDU). Stored with the 3 mm step
  (`add_offset` -10.827 m). Run it with NGA's files:

      gzip -d egm96.z corrcoef.z
      python3.13 egm96_7.5min.py egm96 corrcoef egm96_7.5min.nc

  The output is the shipped `egm96_7.5min.nc` byte for byte (SHA-256 above), checked with numpy 2.5.2, h5py 3.14.0
  (HDF5 1.14.6) and hdf5plugin 7.1.0 on macOS arm64.
- Check of the port (`tests/test_n31_geoid_toolkit/egm96_validate.py`):
  - NGA's test points (INPUT.DAT to OUTF477.DAT, 3 decimals): max difference 0.00046 m.
  - All 1,038,240 nodes of NGA's 15' grid WW15MGH.GRD (3 decimals): max 0.0005 m.
  - All nodes of PROJ `us_nga_egm96_15.tif`: max below 0.00005 m.

Why 7.5' for EGM96: the cubic error of each grid step was measured against F477 at the cell centres
(`egm96_resolution.py`). 7.5' is the coarsest step where the error is no larger than the 3 mm storage step.

| EGM96 grid | Bytes (variable z only) | Max cubic error, cell centres, stored int16 |
|---|---|---|
| 15' | 1,236,986 | 0.0268 m |
| 10' | 2,575,662 | 0.0054 m |
| **7.5'** | 4,262,890 | **0.0029 m** |
| 5' | 8,484,483 | 0.0026 m |

## Accuracy of `utils_geoid.geoid_height()`

Cubic B-spline through the nodes (the spline of `scipy.ndimage`, order 3), with the grid continued across the poles
on the opposite meridian. Errors include the 3 mm storage step. Script: `tests/test_n31_geoid_toolkit/geoid_accuracy.py`.

| Geoid | Reference | Points | Max | RMS |
|---|---|---|---|---|
| EGM2008 | NGA 1' grid, exact node values | all 233,301,600 nodes | 0.0053 m | 0.00076 m |
| EGM2008 | cubic spline of the NGA 1' grid | all 2.5' cell centres | 0.0056 m | 0.00066 m |
| EGM2008 | cubic spline of the NGA 1' grid | 1,000,000 random points | 0.0034 m | 0.00076 m |
| EGM2008 | source GeoTIFF nodes (GDAL) | all 2.5' nodes | 0.0015 m | 0.00087 m |
| EGM96 | F477 | all 7.5' cell centres | 0.0029 m | 0.00066 m |
| EGM96 | F477 | 1,000,000 random points | 0.0024 m | 0.00076 m |
| EGM96 | NGA WW15MGH.GRD | all 15' nodes | 0.0020 m | 0.00082 m |

For comparison, PROJ's bilinear interpolation of the same sources (`vgridshift`) differs from the references by up
to 0.082 m (EGM2008, 2.5' grid) and 1.009 m (EGM96, 15' grid) at the same random points.
