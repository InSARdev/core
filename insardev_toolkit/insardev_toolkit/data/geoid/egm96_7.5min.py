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
The program that computed egm96_7.5min.nc in this folder: EGM96 geoid heights at the nodes of a 7.5' grid.

It is NGA's program F477.F ported to numpy, plus the writer of the grid file. The port is the same algorithm as
F477.F: EGM96 to degree and order 360 minus the WGS84 even zonals J2..J10 (DHCSIN), the WGS84(G873) constants and
normal gravity (RADGRA), NGA's Legendre recursion (LEGFDN), the CORRCOEF height-anomaly-to-undulation correction and
the -0.53 m term (HUNDU). The names in parentheses are the subroutines of F477.F.

This file documents the shipped grid. insardev_toolkit does not import it, and it runs nothing on import.

Original program and inputs (NGA, public domain; see LICENSE and README.md in this folder)
------------------------------------------------------------------------------------------
NGA EGM96 page
    http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html
    archived: https://web.archive.org/web/20061103210455/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html
F477.F, NGA's geoid height program (R. H. Rapp, December 1996)
    http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/f477.f
    archived: https://web.archive.org/web/20061029210851/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/f477.f
    f477.f SHA-256 e817d29b8c8db573946c67484b43df85dd76563ee7cb27fc565ca55c1e49924b
EGM96, the potential coefficients (F477.F reads the file 'EGM96'), compressed as egm96.z
    http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.z
    archived: https://web.archive.org/web/20061120214908/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.z
    egm96.z SHA-256 6e4fc4f2a00bf7b5cd91bed594ce636022cc844d81a4cfa5f76d83de490c68cd
    uncompressed SHA-256 5b2773f3bb532576811baaba650f7da8592ee0e8e3dc84d8604fcae14ab67a47
CORRCOEF, the correction coefficients (F477.F reads the file 'CORRCOEF'), compressed as corrcoef.z
    http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/corrcoef.z
    archived: https://web.archive.org/web/20070112050926/http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/corrcoef.z
    corrcoef.z SHA-256 45b61f280041e39a72d3f654e65124b33eb4e0a5f756b7f43112b855353cef3e
    uncompressed SHA-256 ac5662cddee90c127eba244a78c097bdfe13e2e4584cd63c4e80f2b8b469076c

Run
---
The .z files are Unix compress files; gzip -d decompresses them to egm96 and corrcoef. This file needs no other file
of this folder. Requires numpy, h5py and hdf5plugin.

    gzip -d egm96.z corrcoef.z
    python3.13 egm96_7.5min.py egm96 corrcoef egm96_7.5min.nc

Inputs: the two uncompressed coefficient files, in NGA's free format (degree, order, C, S, ...).
Output: egm96_7.5min.nc, 4,304,758 bytes, SHA-256 a8d3df58087d65a0435f98912dc8259b9ec7ddf35e1de26062ee7473067f6ac5
(the shipped file). The run prints the SHA-256 of the file it wrote and whether it equals the shipped one. The bytes
were reproduced with numpy 2.5.2, h5py 3.14.0 (HDF5 1.14.6) and hdf5plugin 7.1.0 on macOS arm64. With other
versions, compare the int16 values of the variable z instead. The history attribute of the file names this program
and its inputs egm96 and corrcoef with their SHA-256, and no time, host or path of the run.

Format (README.md, "Format"): NetCDF4 (HDF5), CF-1.8. Nodes from -90 to 90 in latitude and from -180 to 180 in
longitude, both ascending; the +180 column repeats -180. Variable z int16 with scale_factor 0.003 m and add_offset
(a 3 mm step, up to 1.5 mm rounding), Blosc2 zstd level 9 with byte shuffle in 512 x 512 chunks.

The other functions of this file, f477_points() for single points and write_grid() for any grid, are used by the
tests of the InSARdev repository (tests/test_n31_geoid_toolkit/).
"""
import numpy as np

NMAX = 360
# WGS84(G873) constants of F477.F
GM = 0.3986004418e15
AE = 6378137.0
E2 = 0.00669437999013
GEQT = 9.7803253359
KSOM = 0.00193185265246
# WGS84 even zonal harmonics, removed from EGM96 (DHCSIN)
J = {2: 0.108262982131e-2, 4: -.237091120053e-05, 6: 0.608346498882e-8, 8: -0.142681087920e-10,
     10: 0.121439275882e-13}

# the grid
STEP = 7.5 / 60.0
SCALE = 0.003
CHUNK = 512
# EPSG:4326 as GDAL 3.9 writes it (rasterio.crs.CRS.from_epsg(4326).to_wkt()), the crs_wkt of the shipped grid
WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
             'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
             'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],'
             'AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
# the attributes of the shipped file. build() fills the history with the SHA-256 of the inputs it read. The history
# names this program and the inputs by fixed names, with nothing of the run (time, host, path), so the same inputs
# give the same file in any folder.
HISTORY = ('computed by data/geoid/egm96_7.5min.py from egm96 (SHA-256 {EGM96}) and corrcoef (SHA-256 {CORRCOEF}), '
           'int16 3 mm step')
ATTRS = dict(
    title="EGM96 geoid heights, 7.5' grid",
    institution='US National Geospatial-Intelligence Agency (NGA)',
    source='NGA EGM96 coefficients EGM96 and CORRCOEF, synthesized with NGA F477.F ported to numpy '
           '(https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html)',
    references='https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html',
    history=HISTORY,
    license='Public domain (NGA public information); see README.md')
CRS_ATTRS = dict(geoid_name='EGM96')

# SHA-256 of the uncompressed NGA coefficient files and of the shipped grid
SHA256 = {'EGM96': '5b2773f3bb532576811baaba650f7da8592ee0e8e3dc84d8604fcae14ab67a47',
          'CORRCOEF': 'ac5662cddee90c127eba244a78c097bdfe13e2e4584cd63c4e80f2b8b469076c',
          'egm96_7.5min.nc': 'a8d3df58087d65a0435f98912dc8259b9ec7ddf35e1de26062ee7473067f6ac5'}


# ---------------------------------------------------------------------------------------------------------------
# NGA F477.F in numpy
# ---------------------------------------------------------------------------------------------------------------
def _read_coef(path, ncol):
    """(n, m) -> columns 3 and 4 (C, S) of an NGA coefficient file (free format, Fortran D exponents)."""
    with open(path) as f:
        text = f.read().replace('D', 'E').replace('d', 'e')
    a = np.array(text.split(), dtype=np.float64).reshape(-1, ncol)
    c = np.zeros((NMAX + 1, NMAX + 1))
    s = np.zeros((NMAX + 1, NMAX + 1))
    n, m = a[:, 0].astype(int), a[:, 1].astype(int)
    keep = n <= NMAX
    c[n[keep], m[keep]] = a[keep, 2]
    s[n[keep], m[keep]] = a[keep, 3]
    return c, s


def load_f477(egm96, corrcoef):
    """
    HC, HS (EGM96 with the WGS84 even zonals removed, DHCSIN) and CC, CS (CORRCOEF), as (n, m) arrays.

    egm96, corrcoef: paths of NGA's uncompressed files EGM96 (6 columns) and CORRCOEF (4 columns).
    """
    hc, hs = _read_coef(egm96, 6)
    for n, jn in J.items():
        hc[n, 0] += jn / np.sqrt(2.0 * n + 1.0)
    cc, cs = _read_coef(corrcoef, 4)
    return hc, hs, cc, cs


def _radgra(flat_deg):
    """RADGRA at height 0: geocentric latitude (rad), normal gravity, geocentric radius."""
    flatr = np.deg2rad(flat_deg)
    t1 = np.sin(flatr) ** 2
    nn = AE / np.sqrt(1.0 - E2 * t1)
    t2 = nn * np.cos(flatr)
    z = nn * (1.0 - E2) * np.sin(flatr)
    re = np.sqrt(t2 ** 2 + z ** 2)
    rlat = np.arctan2(z, np.abs(t2))
    gr = GEQT * (1.0 + KSOM * t1) / np.sqrt(1.0 - E2 * t1)
    return rlat, gr, re


def _row_sums(flat_deg, coef):
    """Per latitude and order m: potential sums (AE/RE)^n P_nm HC/HS and correction sums P_nm CC/CS over n."""
    hc, hs, cc, cs = coef
    rlat, gr, re = _radgra(np.asarray(flat_deg, np.float64))
    theta = np.pi / 2 - rlat
    ct, st = np.cos(theta), np.sin(theta)
    k = rlat.size
    ar = AE / re
    # (n, k): (AE/RE)^n per degree and latitude
    arn = np.arange(NMAX + 1, dtype=np.float64)[:, None]
    arn = ar[None, :] ** arn
    arn[:2] = 0.0                         # the potential sum starts at degree 2 (HUNDU)
    drts = np.sqrt(np.arange(2 * NMAX + 2, dtype=np.float64))
    dirt = np.zeros_like(drts)
    dirt[1:] = 1.0 / drts[1:]
    pa_c = np.zeros((k, NMAX + 1)); pa_s = np.zeros((k, NMAX + 1))
    pc_c = np.zeros((k, NMAX + 1)); pc_s = np.zeros((k, NMAX + 1))
    pmm = np.ones(k)
    for m in range(NMAX + 1):
        # sectoral P_mm (RLNN of LEGFDN)
        if m == 1:
            pmm = st * drts[3]
        elif m > 1:
            pmm = drts[2 * m + 1] * dirt[2 * m] * st * pmm
        # p[n - m] = P_nm for n = m..NMAX (rows), per latitude (columns)
        p = np.zeros((NMAX + 1 - m, k))
        p[0] = pmm
        if m + 1 <= NMAX:
            p[1] = drts[2 * m + 3] * ct * pmm
        for n in range(m + 2, NMAX + 1):
            p[n - m] = drts[2 * n + 1] * dirt[n + m] * dirt[n - m] * (
                drts[2 * n - 1] * ct * p[n - m - 1] - drts[n + m - 1] * drts[n - m - 1] * dirt[2 * n - 3] * p[n - m - 2])
        w = arn[m:] * p
        pa_c[:, m] = hc[m:, m] @ w
        pa_s[:, m] = hs[m:, m] @ w
        pc_c[:, m] = cc[m:, m] @ p
        pc_s[:, m] = cs[m:, m] @ p
    scale = GM / (gr * re)
    return scale, pa_c, pa_s, pc_c, pc_s


def f477_grid(lat_deg, lon_deg, coef, lon_block=2048):
    """EGM96 geoid heights (m) on the tensor grid lat x lon (degrees), NGA F477."""
    lat_deg = np.asarray(lat_deg, np.float64)
    lon_deg = np.asarray(lon_deg, np.float64)
    scale, pa_c, pa_s, pc_c, pc_s = _row_sums(lat_deg, coef)
    m = np.arange(NMAX + 1)[:, None]
    out = np.empty((lat_deg.size, lon_deg.size))
    for j0 in range(0, lon_deg.size, lon_block):
        lam = np.deg2rad(lon_deg[j0:j0 + lon_block])[None, :]
        cm, sm = np.cos(m * lam), np.sin(m * lam)
        a = pa_c @ cm + pa_s @ sm
        c = pc_c @ cm + pc_s @ sm
        out[:, j0:j0 + lon_block] = a * scale[:, None] + c / 100.0 - 0.53
    return out


def f477_points(lat_deg, lon_deg, coef):
    """EGM96 geoid heights (m) at points (1-D arrays), NGA F477; each point its own latitude row."""
    lat_deg = np.asarray(lat_deg, np.float64).ravel()
    lon_deg = np.asarray(lon_deg, np.float64).ravel()
    out = np.empty(lat_deg.size)
    for i0 in range(0, lat_deg.size, 4096):
        sl = slice(i0, i0 + 4096)
        scale, pa_c, pa_s, pc_c, pc_s = _row_sums(lat_deg[sl], coef)
        lam = np.deg2rad(lon_deg[sl])[:, None]
        m = np.arange(NMAX + 1)[None, :]
        cm, sm = np.cos(m * lam), np.sin(m * lam)
        a = np.sum(pa_c * cm + pa_s * sm, axis=1)
        c = np.sum(pc_c * cm + pc_s * sm, axis=1)
        out[sl] = a * scale + c / 100.0 - 0.53
    return out


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


def build(egm96, corrcoef, output):
    """Compute the 7.5' EGM96 grid from NGA's coefficient files and write it to output (egm96_7.5min.nc)."""
    import os
    import time
    t0 = time.time()
    digests = {}
    for label, path in (('EGM96', egm96), ('CORRCOEF', corrcoef)):
        digests[label] = digest = sha256(path)
        print(f'{label}: {path}, {os.path.getsize(path)} bytes, sha256 {digest}')
        if digest != SHA256[label]:
            print(f"WARNING: {path}: not NGA's {label} file (sha256 {SHA256[label]}). Use NGA's file to reproduce "
                  f"the shipped grid.")
    coef = load_f477(egm96, corrcoef)
    lat = -90.0 + STEP * np.arange(int(round(180 / STEP)) + 1)
    lon = -180.0 + STEP * np.arange(int(round(360 / STEP)))
    z = f477_grid(lat, lon, coef)
    print(f'F477 heights at {z.shape[0]} x {z.shape[1]} nodes in {time.time() - t0:.1f} s')
    attrs = dict(ATTRS, history=HISTORY.format(**digests))
    err, offset = write_grid(output, z, STEP, attrs, CRS_ATTRS)
    digest = sha256(output)
    same = 'equal to' if digest == SHA256['egm96_7.5min.nc'] else 'not equal to'
    print(f'{output}: {os.path.getsize(output)} bytes, add_offset {offset}, largest rounding {err:.6f} m')
    print(f'sha256 {digest}, {same} the shipped egm96_7.5min.nc')
    return z


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EGM96 geoid heights on a 7.5' grid, NGA F477.F in numpy.")
    parser.add_argument('egm96', help="NGA's file EGM96 (egm96.z uncompressed)")
    parser.add_argument('corrcoef', help="NGA's file CORRCOEF (corrcoef.z uncompressed)")
    parser.add_argument('output', help='the grid file to write, e.g. egm96_7.5min.nc')
    args = parser.parse_args()
    build(args.egm96, args.corrcoef, args.output)


if __name__ == '__main__':
    main()
