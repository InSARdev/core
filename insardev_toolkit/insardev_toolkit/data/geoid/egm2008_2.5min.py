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
The program that wrote egm2008_2.5min.nc in this folder: EGM2008 geoid heights at the nodes of a 2.5' grid.

It converts PROJ's grid us_nga_egm08_25.tif to the grid file of this folder. The nodes are the GeoTIFF's nodes, read
by GDAL (rasterio): no resampling and no new values. Only the storage changes, to int16 with a 3 mm step. The file is
written by write_grid() of this file.

This file documents the shipped grid. insardev_toolkit does not import it, and it runs nothing on import.

Original file and its origin (public domain; see LICENSE and README.md in this folder)
-------------------------------------------------------------------------------------
us_nga_egm08_25.tif, PROJ-data: EGM2008 geoid heights at 2.5' nodes, float32 GeoTIFF
    https://cdn.proj.org/us_nga_egm08_25.tif
    80,585,622 bytes, SHA-256 4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a
PROJ-data on this file: source NGA, "License: Public Domain", "GeoTIFF converted from GTX", "produced by
GeographicLib using the EGM2008 gravity model"; copyright_and_licenses.csv: us_nga_egm08_25.tif,Disclaimed,Public
domain
    https://github.com/OSGeo/PROJ-data/blob/master/us_nga/us_nga_README.txt
    https://github.com/OSGeo/PROJ-data/blob/master/copyright_and_licenses.csv
The GTX, as PROJ built it: GeographicLib 1.49, program GeoidToGTX, from GeographicLib's EGM2008 gravity model file
(egm2008.zip), "GeoidToGTX egm2008 24 egm08_25.gtx" (24 nodes per degree, 2.5')
    https://raw.githubusercontent.com/OSGeo/proj-datumgrid/master/world/build_egm08_25_gtx.sh
    https://geographiclib.sourceforge.io/C++/doc/gravity.html
EGM2008, NGA's Earth gravitational model (NGA EGM2008 page)
    https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
    archived: https://web.archive.org/web/20260105104748/https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84

Run
---
This file needs no other file of this folder. Requires numpy, rasterio (GDAL), h5py and hdf5plugin.

    curl -O https://cdn.proj.org/us_nga_egm08_25.tif
    python3.13 egm2008_2.5min.py us_nga_egm08_25.tif egm2008_2.5min.nc

Input: PROJ's GeoTIFF, as downloaded.
Output: egm2008_2.5min.nc, 29,301,280 bytes, SHA-256 622ab4bed5f634e7c704eb102442ab707fa818dff75493eb904f86008223946a
(the shipped file). The run prints the SHA-256 of the file it wrote and whether it equals the shipped one. The bytes
were reproduced with numpy 2.5.2, rasterio 1.4.3 (GDAL 3.9.3), h5py 3.14.0 (HDF5 1.14.6) and hdf5plugin 7.1.0 on
macOS arm64. With other versions, compare the int16 values of the variable z instead. The history attribute of the
file names this program and its input us_nga_egm08_25.tif with its SHA-256, and no time, host or path of the run.

Format (README.md, "Format"): NetCDF4 (HDF5), CF-1.8. Nodes from -90 to 90 in latitude and from -180 to 180 in
longitude, both ascending; the +180 column repeats -180. Variable z int16 with scale_factor 0.003 m and add_offset
(a 3 mm step, up to 1.5 mm rounding), Blosc2 zstd level 9 with byte shuffle in 512 x 512 chunks.

The function read_tif() of this file, for any PROJ or NGA geoid GeoTIFF, is also used by the tests of the InSARdev
repository (tests/test_n31_geoid_toolkit/).
"""
import numpy as np

# the grid
STEP = 2.5 / 60.0
SCALE = 0.003
CHUNK = 512
# EPSG:4326 as GDAL 3.9 writes it (rasterio.crs.CRS.from_epsg(4326).to_wkt()), the crs_wkt of the shipped grid
WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
             'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
             'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],'
             'AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
# the attributes of the shipped file. build() fills the history with the SHA-256 of the input it read. The history
# names this program and the input by fixed names, with nothing of the run (time, host, path), so the same input
# gives the same file in any folder.
HISTORY = ('converted by data/geoid/egm2008_2.5min.py from us_nga_egm08_25.tif (SHA-256 {tif}): nodes unchanged, '
           'int16 3 mm step')
ATTRS = dict(
    title="EGM2008 geoid heights, 2.5' grid",
    institution='US National Geospatial-Intelligence Agency (NGA)',
    source='https://cdn.proj.org/us_nga_egm08_25.tif (PROJ-data us_nga, NGA EGM2008, public domain)',
    references='https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84',
    history=HISTORY,
    license='Public domain (NGA public information); see README.md')
CRS_ATTRS = dict(geoid_name='EGM2008')

# SHA-256 of PROJ's GeoTIFF and of the shipped grid
SHA256 = {'us_nga_egm08_25.tif': '4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a',
          'egm2008_2.5min.nc': '622ab4bed5f634e7c704eb102442ab707fa818dff75493eb904f86008223946a'}


# ---------------------------------------------------------------------------------------------------------------
# the source GeoTIFF
# ---------------------------------------------------------------------------------------------------------------
def read_tif(path):
    """
    The nodes of a PROJ or NGA geoid GeoTIFF, read by GDAL (rasterio). Returns z and the grid step (degrees).

    z: float64 nodes, lat from -90 (rows), lon from -180 (columns, periodic, without the +180 column).
    """
    import rasterio
    with rasterio.open(path) as src:
        z = src.read(1).astype(np.float64)
        t = src.transform
    lon = t.c + t.a * (np.arange(z.shape[1]) + 0.5)
    lat = t.f + t.e * (np.arange(z.shape[0]) + 0.5)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    step = float(t.a)
    assert abs(lon[0] + 180.0) < 1e-6 and abs(lat[0] + 90.0) < 1e-6, (lon[0], lat[0])
    n = int(round(360.0 / step))
    if z.shape[1] > n:
        assert np.max(np.abs(z[:, n] - z[:, 0])) < 1e-6
    return np.ascontiguousarray(z[:, :n]), step


# ---------------------------------------------------------------------------------------------------------------
# the grid file
# ---------------------------------------------------------------------------------------------------------------
def write_grid(path, z, step, attrs, crs_attrs):
    """
    Write a geoid grid file in the format of this folder. Returns the largest rounding error (m) and add_offset.

    z: float64 nodes, lat from -90 (rows), lon from -180 (columns, periodic, without the +180 column).
    """
    import io
    import h5py
    import hdf5plugin
    nlat, nlon = z.shape
    assert abs((nlat - 1) * step - 180.0) < 1e-9 and abs(nlon * step - 360.0) < 1e-9, (z.shape, step)
    z = np.concatenate([z, z[:, :1]], axis=1)
    lat = -90.0 + step * np.arange(nlat)
    lon = -180.0 + step * np.arange(nlon + 1)
    lat[-1], lon[-1] = 90.0, 180.0
    offset = round(0.5 * (z.min() + z.max()) / SCALE) * SCALE
    q = np.round((z - offset) / SCALE)
    assert np.abs(q).max() <= 32767, np.abs(q).max()
    q = q.astype(np.int16)
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        dl = f.create_dataset('lat', data=lat, track_times=False)
        dl.make_scale('lat')
        dl.attrs.update(units='degrees_north', standard_name='latitude', long_name='latitude', axis='Y')
        do = f.create_dataset('lon', data=lon, track_times=False)
        do.make_scale('lon')
        do.attrs.update(units='degrees_east', standard_name='longitude', long_name='longitude', axis='X')
        crs = f.create_dataset('crs', data=np.int32(0), track_times=False)
        crs.attrs.update(grid_mapping_name='latitude_longitude', semi_major_axis=6378137.0,
                         inverse_flattening=298.257223563, crs_wkt=WGS84_WKT, **crs_attrs)
        dz = f.create_dataset('z', data=q, chunks=(CHUNK, CHUNK), track_times=False,
                              **hdf5plugin.Blosc2(cname='zstd', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE))
        dz.dims[0].attach_scale(dl)
        dz.dims[1].attach_scale(do)
        dz.attrs.update(units='m', long_name='geoid height above the WGS84 ellipsoid', grid_mapping='crs',
                        scale_factor=np.float64(SCALE), add_offset=np.float64(offset))
        f.attrs['Conventions'] = 'CF-1.8'
        for k, v in attrs.items():
            f.attrs[k] = v
    with open(path, 'wb') as fh:
        fh.write(buf.getvalue())
    back = q.astype(np.float64) * SCALE + offset
    return float(np.max(np.abs(back - z))), offset


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(tif, output):
    """Convert PROJ's us_nga_egm08_25.tif to the 2.5' EGM2008 grid and write it to output (egm2008_2.5min.nc)."""
    import os
    import time
    t0 = time.time()
    tif_digest = sha256(tif)
    print(f'input: {tif}, {os.path.getsize(tif)} bytes, sha256 {tif_digest}')
    if tif_digest != SHA256['us_nga_egm08_25.tif']:
        print(f"WARNING: {tif}: not PROJ's us_nga_egm08_25.tif (sha256 {SHA256['us_nga_egm08_25.tif']}). Use "
              f"PROJ's file to reproduce the shipped grid.")
    z, step = read_tif(tif)
    if abs(step - STEP) > 1e-9:
        raise ValueError(f"{tif}: grid step {step * 60:g}', not 2.5'. Use PROJ's us_nga_egm08_25.tif.")
    print(f'GeoTIFF nodes {z.shape[0]} x {z.shape[1]} read in {time.time() - t0:.1f} s')
    attrs = dict(ATTRS, history=HISTORY.format(tif=tif_digest))
    err, offset = write_grid(output, z, step, attrs, CRS_ATTRS)
    digest = sha256(output)
    same = 'equal to' if digest == SHA256['egm2008_2.5min.nc'] else 'not equal to'
    print(f'{output}: {os.path.getsize(output)} bytes, add_offset {offset}, largest rounding {err:.6f} m')
    print(f'sha256 {digest}, {same} the shipped egm2008_2.5min.nc')
    return z


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EGM2008 geoid heights on a 2.5' grid, from PROJ's GeoTIFF.")
    parser.add_argument('tif', help="PROJ's us_nga_egm08_25.tif (https://cdn.proj.org/us_nga_egm08_25.tif)")
    parser.add_argument('output', help='the grid file to write, e.g. egm2008_2.5min.nc')
    args = parser.parse_args()
    build(args.tif, args.output)


if __name__ == '__main__':
    main()
