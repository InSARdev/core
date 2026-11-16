# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""The atmosphere phase of one data block, pixelwise. burst_constants, once per burst: the geometry, the CRS and the shell flags (and
the height rows of the burst's ele range for a one-off call of cut_block). block_tables, once per chunk at the graph build: the
height rows hs (HS_STEP of the kind, absolute multiples, one row beyond each end of the chunk's ele range) and the second-order least-squares
polynomials, per row, in (x - x_c, y - y_c) about the absolute 10 km centres the chunk's pixels
need (a 3 x 3 stencil at +-100 m) of the incidence and, with a shell, of the ionosphere's pierce latitude, pierce longitude and
effective incidence (the look azimuth enters through them only). cut_block, per chunk: each pixel's lat/lon, bilinear on the absolute geometry lattice
(utils_geometry._lattice_step; pyproj only at its nodes), and the box of the model's lattice the chunk reads. grid_block, one
numba pass per block for all of the block's dates (per row the date-independent part of each pixel once, then each date reads only its
tables; chunking the dates of the stack, k dates per chunk, makes the phase cheaper per date): the troposphere (at each of the four model nodes around the pixel, the zenith delays at the pixel's height
(linear between the 10 m rows) times the mapping at the pixel's incidence plus the height term, then the bilinear of the node
delays; date minus reference) or the
ionosphere (the pierce point from the polynomials, the TEC bilinear times the mapping). Each pixel's value depends on its own
coordinates and the burst constants only: nothing depends on the chunking.

The formulas are the toolkit's public helpers (mapping_form, height_term, pierce_point, iono_mapping); none is copied here.
The tables arrive as arrays on the model's lattice (phase_block gathers them from the held tiles).
"""
import math
import warnings
from typing import NamedTuple

import numpy as np
import numba as nb
from insardev_toolkit.utils_atmosphere import mapping_form, height_term, pierce_point, iono_mapping
from .utils_geometry import _Pix, _latlon, look_azimuth, _node_latlon, METRES_PER_DEGREE

# m: the height rows of the model's zenith tables (the toolkit's step)
HEIGHT_STEP = 10.0
# m: the spacing of the height rows of the polynomials, per kind (the incidence of the troposphere's mapping function; the
# ionosphere's pierce geometry, nearly linear in height)
HS_STEP = dict(tropo=250.0, iono=1000.0)
# m: the stencil step of the polynomials (degrees of a geographic CRS: scaled)
STENCIL = 100.0
# m: the spacing of the absolute centres of the polynomials (degrees of a geographic CRS: scaled)
CELL = 10000.0


class Cut(NamedTuple):
    """The per-chunk step of one block (cut_block): the box (i0, i1, j0, j1) of the model's lattice the block reads, the pixel
    coordinates, the pixel lattice (node indices from the first node and weights) with its node latitudes and longitudes (degrees),
    the model's axes (first node, step, size) and the burst constants."""
    i0: int
    i1: int
    j0: int
    j1: int
    ycoord: np.ndarray
    xcoord: np.ndarray
    jy: np.ndarray
    wy: np.ndarray
    jx: np.ndarray
    wx: np.ndarray
    LAT: np.ndarray
    LON: np.ndarray
    lat0: float
    dla: float
    lon0: float
    dlo: float
    nla: int
    nlo: int
    const: dict


def _fit(F):
    """The second-order least-squares polynomial of each 3 x 3 stencil F (rows y, columns x; the last two axes) over its nine
    points at the offsets -1, 0, 1 (in units of the stencil step): the coefficients c0, cx, cy, cxx, cyy, cxy of
    P(u, v) = c0 + cx u + cy v + cxx u^2 + cyy v^2 + cxy u v.

    Parameters
    ----------
    F : numpy.ndarray - The stencil values, shape (..., 3, 3).

    Returns
    -------
    numpy.ndarray - The coefficients, shape (..., 6).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils_atmosphere import _fit
    >>> X, Y = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    >>> (_fit(1.0 + 2.0 * X + 3.0 * Y + 4.0 * X * Y + 5.0 * X * X).round(9) + 0.0).tolist()
    [1.0, 2.0, 3.0, 5.0, 0.0, 4.0]
    """
    U, V = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    u, v = U.ravel(), V.ravel()
    A = np.stack([np.ones(9), u, v, u * u, v * v, u * v], axis=1)
    W = np.linalg.pinv(A)
    F9 = F.reshape(F.shape[:-2] + (9,))
    # the nine products added in a fixed order: the same bits whatever the number of stencils (a matrix product would not be)
    out = []
    for j in range(6):
        acc = F9[..., 0] * W[j, 0]
        for k in range(1, 9):
            acc = acc + F9[..., k] * W[j, k]
        out.append(acc)
    return np.stack(out, axis=-1)


def _unwrap_about_centre(F, ref=None):
    """The degrees of each 3 x 3 stencil F (last two axes) unwrapped (period 360) about its centre value. With ref (degrees), each
    stencil whose centre lies a whole number of turns away from it is moved by those turns (the others keep their bits): the
    move is exact near +-180, where the values and the result share a binade and 360 is a multiple of its spacing.

    Parameters
    ----------
    F : numpy.ndarray - Angles in degrees, shape (..., 3, 3).
    ref : float or None - The degrees whose branch the stencils take; None: no move.

    Returns
    -------
    numpy.ndarray - The angles without a jump of 360 across the stencil, shape (..., 3, 3).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils_atmosphere import _unwrap_about_centre
    >>> _unwrap_about_centre(np.array([[359.0, 0.5, 1.0], [359.5, 0.0, 1.0], [359.0, 0.5, 1.0]])).round(9).tolist()[1]
    [-0.5, 0.0, 1.0]
    >>> F = np.array([np.full((3, 3), -179.998), np.full((3, 3), 179.997)])
    >>> _unwrap_about_centre(F, 179.99)[:, 1, 1].round(9).tolist()
    [180.002, 179.997]
    """
    c = F[..., 1:2, 1:2]
    G = c + ((F - c + 180.0) % 360.0 - 180.0)
    if ref is None:
        return G
    k = np.round((c - ref) / 360.0)
    return np.where(k != 0.0, G - 360.0 * k, G)


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _pierce_loop(lat, lon, h, z, look, shell, geocentric, plat, plon, zeff):
    """The toolkit's pierce_point per point, in place: lat and lon in degrees, h, z and look as pierce_point takes them."""
    dummy = np.zeros((1, 1))
    for q in range(lat.size):
        a, b, c = pierce_point(lat[q], lon[q], h[q], z[q], look[q], shell, geocentric, False, 0, 0, 1.0, 1.0, dummy, dummy)
        plat[q] = a
        plon[q] = b
        zeff[q] = c


def _cell_step(crs):
    """The spacing of the absolute centres and the stencil step (CELL, STENCIL): map units, or the same length in degrees
    for a geographic CRS (as utils_geometry._lattice_step converts its step)."""
    from pyproj import CRS
    scale = 1.0 / METRES_PER_DEGREE if CRS.from_user_input(crs).is_geographic else 1.0
    return CELL * scale, STENCIL * scale


def _centres(const, hs, kx, ky, cell, step):
    """The polynomial tables of the absolute centres (kx x ky, integer multiples of cell): per centre and height row, the
    incidence and, with a shell, the pierce latitude, the pierce longitude (degrees, unwrapped about the centre's own value
    and moved by whole turns to the centre's ground longitude, so the height rows of one centre share a branch) and the
    effective incidence. Each centre is evaluated alone on its own 3 x 3 stencil, so its values depend on the centre
    only.

    Parameters
    ----------
    const : dict - The burst constants (burst_constants).
    hs : numpy.ndarray - The height rows, metres (_rows).
    kx, ky : numpy.ndarray - The integer multiples of cell, per axis.
    cell, step : float - The centre spacing and the stencil step, map units (_cell_step).

    Returns
    -------
    dict - tables, each (len(kx), len(ky), len(hs), 6): without a shell (the troposphere) inc (radians) only, with a shell
        (the ionosphere) pl (degrees), pn (degrees) and zf (radians) only; the others are None.

    Examples
    --------
    >>> tabs = _centres(const, hs, np.arange(3, 5), np.arange(-2, 0), 10000.0, 100.0)
    """
    geo, crs, shell, geocentric = const['geo'], const['crs'], const['shell'], const['geocentric']
    off = step * np.array([-1.0, 0.0, 1.0])
    OX, OY = (a.ravel() for a in np.meshgrid(off, off))
    nh = hs.size
    H = np.repeat(hs, 9)
    shape = (len(kx), len(ky), nh, 6)
    inc = np.empty(shape) if shell is None else None
    pl = np.empty(shape) if shell is not None else None
    pn = np.empty(shape) if shell is not None else None
    zf = np.empty(shape) if shell is not None else None
    for a, kxv in enumerate(kx):
        for b, kyv in enumerate(ky):
            lat, lon = _latlon(crs, float(kxv) * cell + OX, float(kyv) * cell + OY)
            LAT, LON = np.tile(lat, nh), np.tile(lon, nh)
            z = np.asarray(_Pix(geo, LAT, LON, H).inc, np.float64)
            if shell is None:
                inc[a, b] = _fit(z.reshape(nh, 3, 3))
            else:
                look = np.asarray(look_azimuth(geo, LAT, LON, H), np.float64)
                plat, plon, zeff = np.empty(nh * 9), np.empty(nh * 9), np.empty(nh * 9)
                _pierce_loop(np.degrees(LAT), np.degrees(LON), H, z, look, float(shell), bool(geocentric), plat, plon, zeff)
                pl[a, b] = _fit(plat.reshape(nh, 3, 3))
                pn[a, b] = _fit(_unwrap_about_centre(plon.reshape(nh, 3, 3), float(np.degrees(lon[4]))))
                zf[a, b] = _fit(zeff.reshape(nh, 3, 3))
    return dict(inc=inc, pl=pl, pn=pn, zf=zf)


def _hstep(shell):
    """The row spacing of the kind: the ionosphere's with a shell, the troposphere's without (HS_STEP).

    Parameters
    ----------
    shell : float or None - R + H, metres; None: the troposphere.

    Returns
    -------
    float - The spacing, metres.

    Examples
    --------
    >>> _hstep(None) == HS_STEP['tropo']
    True
    """
    return HS_STEP['tropo' if shell is None else 'iono']


def _rows(heights, hstep):
    """The height rows of the polynomials: every hstep metres at the absolute multiples, one row beyond each end of the range.

    Parameters
    ----------
    heights : tuple of float - (lo, hi), metres.
    hstep : float - The row spacing, metres (HS_STEP of the kind).

    Returns
    -------
    numpy.ndarray - The row heights hs, metres.

    Examples
    --------
    >>> _rows((112.0, 231.0), 100.0).tolist()
    [0.0, 100.0, 200.0, 300.0, 400.0]
    """
    lo, hi = (float(h) for h in heights)
    return hstep * np.arange(math.floor(lo / hstep) - 1, math.ceil(hi / hstep) + 2, dtype=np.float64)


def burst_constants(geo, crs, heights, shell=None, geocentric=True, centre=None):
    """The constants of one burst, computed once per burst: the height rows hs (every HS_STEP of the kind, one row beyond each end of the
    ele range; cut_block's default), the geometry, the CRS, the shell flags and the longitude of the burst centre (the seam rule:
    cut_block unwraps the nodes' longitudes about it). The polynomials of the absolute centres are computed per chunk
    (block_tables).

    Parameters
    ----------
    geo : dict - The burst's geometry (utils_geometry.burst_geometry).
    crs : str - The output CRS of the burst.
    heights : tuple of float - (lo, hi): the ellipsoidal heights of the ele range (its actual_range).
    shell : float or None - R + H, metres; None: no pierce polynomials.
    geocentric : bool - The ionosphere's zenith flag (pierce_point).
    centre : tuple of float or None - (x_c, y_c): the centre of the burst's whole output grid in the output CRS, never of a chunk;
        None: cut_block unwraps about its own block's middle node.

    Returns
    -------
    dict - geo, crs, hs (m), shell (float or None), geocentric and lon_c (degrees, the centre's longitude; None without centre).

    Examples
    --------
    >>> const = burst_constants(geo, crs, (lo, hi))
    """
    hs = _rows(heights, _hstep(shell))
    lon_c = None if centre is None else float(np.degrees(_latlon(crs, np.float64(centre[0]), np.float64(centre[1]))[1]))
    return dict(geo=geo, crs=crs, hs=hs, shell=None if shell is None else float(shell), geocentric=bool(geocentric), lon_c=lon_c)


def block_tables(ycoord, xcoord, const, heights=None):
    """The polynomial tables of a block, which depend on its geometry and height range only (no date): the height rows of the
    range and the absolute centres its pixels need, the multiples of cell nearest the block's own ranges (_centres). The Stack
    builds them once per chunk at the graph build, for the chunk's own ele range; every date of the chunk reads them.

    Parameters
    ----------
    ycoord, xcoord : numpy.ndarray - The block's pixel row and column coordinates, map units.
    const : dict - The burst constants (burst_constants).
    heights : tuple of float or None - (lo, hi): the block's ele range, metres; None: the rows of the burst constants.

    Returns
    -------
    dict - hs (the height rows, m), hstep (their spacing, m), cell, step (map units), kx0, ky0 (the integer multiples of the first centre per axis) and
        the tables of _centres: inc without a shell (the troposphere); pl, pn and zf with one (the ionosphere); the others
        are None.

    Examples
    --------
    >>> tables = block_tables(ycoord, xcoord, const, (lo, hi))
    """
    ycoord = np.asarray(ycoord, np.float64)
    xcoord = np.asarray(xcoord, np.float64)
    hstep = _hstep(const['shell'])
    hs = const['hs'] if heights is None else _rows(heights, hstep)
    cell, step = _cell_step(const['crs'])
    kx = np.arange(int(np.rint(xcoord.min() / cell)), int(np.rint(xcoord.max() / cell)) + 1)
    ky = np.arange(int(np.rint(ycoord.min() / cell)), int(np.rint(ycoord.max() / cell)) + 1)
    return dict(hs=hs, hstep=hstep, kx0=int(kx[0]), ky0=int(ky[0]), cell=cell, step=step, **_centres(const, hs, kx, ky, cell, step))


def ele_range(ele):
    """The range (lo, hi) of the finite heights of a block, metres; (0.0, 0.0) for a block without any (its phase is NaN).

    Parameters
    ----------
    ele : numpy.ndarray - The block's pixel heights.

    Returns
    -------
    tuple of float - (lo, hi).

    Examples
    --------
    >>> ele_range(np.array([[1.0, np.nan], [4.0, 2.0]]))
    (1.0, 4.0)
    """
    a = np.asarray(ele, np.float64)
    a = a[np.isfinite(a)]
    return (0.0, 0.0) if a.size == 0 else (float(a.min()), float(a.max()))


def cut_block(ycoord, xcoord, const, node_lat, node_lon, tables=None):
    """The per-chunk step of one data block: the block's pixel lattice (utils_geometry._node_latlon: the absolute nodes of the
    geometry lattice, pyproj at them only) as node latitudes and longitudes in degrees (the longitudes unwrapped about the burst centre), the box (i0, i1, j0, j1) of the model's
    lattice the block reads, and the polynomial tables (block_tables) in the cut's constants. A pixel outside the
    model's lattice is flagged by grid_block. No pixel is evaluated here.

    Parameters
    ----------
    ycoord, xcoord : numpy.ndarray - The block's pixel row and column coordinates, map units.
    const : dict - The burst constants (burst_constants).
    node_lat, node_lon : numpy.ndarray - The model's regular lattice latitudes (nlat) and longitudes (nlon), degrees; the
        longitudes ascend.
    tables : dict or None - The block's tables (block_tables), built once per chunk; None computes them here for the block's own
        pixels and the rows of the burst constants; {} gives a box-only cut (no tables), which grid_block refuses.

    Returns
    -------
    Cut - The box, the pixel lattice, the model's axes and the burst constants with the tables.

    Examples
    --------
    >>> cut = cut_block(ycoord, xcoord, const, lat, lon, tables)
    """
    ycoord = np.asarray(ycoord, np.float64)
    xcoord = np.asarray(xcoord, np.float64)
    jy, wy, jx, wx, lat, lon = _node_latlon({'crs': const['crs']}, np.arange(ycoord.size), np.arange(xcoord.size),
                                            ycoord, xcoord)
    LAT = np.degrees(lat)
    LON = np.degrees(lon)
    # the seam rule: the nodes' longitudes unwrapped about the burst centre (lookups are wrapped into the model's window)
    lon_c = const.get('lon_c')
    if lon_c is None:
        lon_c = float(LON[LON.shape[0] // 2, LON.shape[1] // 2])
    LON = lon_c + ((LON - lon_c + 180.0) % 360.0 - 180.0)
    if tables is None:
        tables = block_tables(ycoord, xcoord, const)
    const = dict(const, **tables)
    node_lat = np.asarray(node_lat, np.float64)
    node_lon = np.asarray(node_lon, np.float64)
    nla, nlo = node_lat.size, node_lon.size
    lat0, dla = float(node_lat[0]), float(node_lat[1] - node_lat[0])
    lon0, dlo = float(node_lon[0]), float(node_lon[1] - node_lon[0])
    if dlo <= 0.0:
        raise ValueError('ERROR: the model longitudes must ascend')
    tl = (LAT - lat0) / dla
    i0 = min(max(int(math.floor(tl.min())), 0), nla - 1)
    i1 = min(max(int(math.floor(tl.max())) + 2, i0 + 1), nla)
    # the longitudes in the model's window [lon0, lon0 + 360); a block straddling the window's end reads all columns
    LONw = lon0 + (LON - lon0 - 360.0 * np.floor((LON - lon0) / 360.0))
    tn = (LONw - lon0) / dlo
    if tn.max() - tn.min() <= 180.0 / dlo:
        j0 = min(max(int(math.floor(tn.min())), 0), nlo - 1)
        j1 = min(max(int(math.floor(tn.max())) + 2, j0 + 1), nlo)
    else:
        j0, j1 = 0, nlo
    return Cut(i0, i1, j0, j1, ycoord, xcoord, jy, wy, jx, wx, LAT, LON, lat0, dla, lon0, dlo, nla, nlo, const)


def gather(tiles, arrays, i0, i1, j0, j1):
    """The tile arrays on the box of the model's lattice: one array (i1 - i0, j1 - j0, ...), NaN where no tile holds a node of
    the box.

    Parameters
    ----------
    tiles : sequence of (int, int) - The first node (lattice row, column) of each tile.
    arrays : sequence of numpy.ndarray - The tile's values (ty, tx, ...), one per tile, the same trailing shape.
    i0, i1, j0, j1 : int - The box (cut_block).

    Returns
    -------
    numpy.ndarray - The values on the box, float64.

    Examples
    --------
    >>> from insardev.utils_atmosphere import gather
    >>> gather([(0, 0)], [np.ones((2, 2, 3))], 0, 2, 1, 3).shape
    (2, 2, 3)
    """
    tail = np.shape(arrays[0])[2:]
    out = np.full((i1 - i0, j1 - j0) + tail, np.nan)
    for (y0, x0), a in zip(tiles, arrays):
        ty, tx = a.shape[0], a.shape[1]
        r0, r1 = max(i0, y0), min(i1, y0 + ty)
        c0, c1 = max(j0, x0), min(j1, x0 + tx)
        if r0 < r1 and c0 < c1:
            out[r0 - i0:r1 - i0, c0 - j0:c1 - j0] = a[r0 - y0:r1 - y0, c0 - x0:c1 - x0]
    return out


def pixel_axis(first, step, i0, i1):
    """The pixel centre coordinates i0 .. i1 - 1 of one axis of a transform lattice: first + step * i, numpy's own arange fill,
    so a stored axis is reproduced bitwise. The Stack passes (first, step) and the block's index range instead of the arrays.

    Parameters
    ----------
    first, step : float - The first centre and the spacing, map units (step 0.0 for an axis of one pixel).
    i0, i1 : int - The index range (i1 exclusive).

    Returns
    -------
    numpy.ndarray - The coordinates, float64, shape (i1 - i0,).

    Examples
    --------
    >>> from insardev.utils_atmosphere import pixel_axis
    >>> pixel_axis(100.0, 10.0, 2, 5).tolist()
    [120.0, 130.0, 140.0]
    """
    return first + step * np.arange(i0, i1)


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _nearest(x, y, cell, step, kx0, ky0):
    """The table indices (ix, iy) of the nearest absolute centre of the pixel (x, y) and its offsets from that centre, in
    stencil steps."""
    kx = np.rint(x / cell)
    ky = np.rint(y / cell)
    return int(kx) - kx0, int(ky) - ky0, (x - kx * cell) / step, (y - ky * cell) / step


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _height_row(e, h0, nhs, hstep):
    """The row (kh, wq) of the polynomial rows (first height h0, nhs rows every hstep) around the height e: row kh and the
    weight of row kh + 1, kh = -1 for a height outside the rows. The weight comes from the absolute height (e / hstep less its
    floor), never from h0, so a pixel's value does not depend on the rows' range."""
    th = e / hstep
    kf = math.floor(th)
    kh = int(kf) - int(round(h0 / hstep))
    wq = th - kf
    if kh == nhs - 1 and wq == 0.0:
        return nhs - 2, 1.0
    if kh < 0 or kh > nhs - 2:
        return -1, 0.0
    return kh, wq


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _poly(T, ix, iy, k, dx, dy):
    """The polynomial of the centre (ix, iy) and height row k of T (nkx, nky, nh, 6) at (dx, dy), in stencil steps."""
    c = T[ix, iy, k]
    return c[0] + c[1] * dx + c[2] * dy + c[3] * dx * dx + c[4] * dy * dy + c[5] * dx * dy


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _node_delay(hyd, wet, mp, p, m, kf, wh, s, hz):
    """The slant delay of one model node (phase units): the zenith delays at the height row kf + wh (linear between the rows)
    times the mapping factors at cos of the incidence s (toolkit mapping_form of the node's coefficients), the hydrostatic one
    plus the height term hz.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils_atmosphere import _node_delay
    >>> mp = np.zeros((1, 1, 2, 4))
    >>> mp[..., 3] = 1.0
    >>> round(_node_delay(np.ones((1, 1, 3)), np.ones((1, 1, 3)), mp, 0, 0, 0, 0.5, 0.5, 0.0), 6)
    4.0
    """
    zh = (1.0 - wh) * hyd[p, m, kf] + wh * hyd[p, m, kf + 1]
    zw = (1.0 - wh) * wet[p, m, kf] + wh * wet[p, m, kf + 1]
    mh = mapping_form(mp[p, m, 0, 0], mp[p, m, 0, 1], mp[p, m, 0, 2], mp[p, m, 0, 3], s) + hz
    mw = mapping_form(mp[p, m, 1, 0], mp[p, m, 1, 1], mp[p, m, 1, 2], mp[p, m, 1, 3], s)
    return zh * mh + zw * mw


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _grid_tropo(ele, ycoord, xcoord, jy, wy, jx, wx, LAT, LON, lat0, dla, lon0, dlo, nla, nlo, i0, j0,
                hyd_d, wet_d, map_d, hyd_r, wet_r, map_r, hcoef, h_first, hs, hstep, cell, step, kx0, ky0, inc_t, out, flags):
    """The troposphere phase of a block in one serial pass, every date inside the pass: the date minus the reference, per pixel (see
    grid_block). Per row, the date-independent part of each pixel (its lat/lon cell and weights, the height rows, the incidence and
    the reference's four node delays) is computed once into row buffers of nx values; then each date reads only its own tables.
    The tables of the dates carry a leading date axis: hyd_d, wet_d (nd, na, nb, nh), map_d (nd, na, nb, 2, 4); the reference's do
    not; out is (nd, ny, nx). flags counts: [0] pixels whose height is outside the held rows, [1] NaN table values (pixel-dates), [2]
    pixels outside the model's lattice or its box.
    """
    nd = hyd_d.shape[0]
    ny = ele.shape[0]
    nx = ele.shape[1]
    na = hyd_d.shape[1]
    nbb = hyd_d.shape[2]
    nh = hyd_d.shape[3]
    nhs = hs.shape[0]
    hv = hcoef[0]
    ha = hcoef[1]
    hb = hcoef[2]
    hc = hcoef[3]
    ok = np.zeros(nx, np.bool_)
    P = np.zeros(nx, np.int64)
    M = np.zeros(nx, np.int64)
    KF = np.zeros(nx, np.int64)
    WH = np.zeros(nx, np.float64)
    S = np.zeros(nx, np.float64)
    HZ = np.zeros(nx, np.float64)
    W4 = np.zeros((nx, 4), np.float64)
    R4 = np.zeros((nx, 4), np.float64)
    for r in range(ny):
        a = jy[r]
        w1 = wy[r]
        for c in range(nx):
            ok[c] = False
            e = ele[r, c]
            if e != e:
                continue
            b = jx[c]
            w2 = wx[c]
            # the pixel's lat/lon: bilinear on the absolute lattice nodes (degrees, the longitude unwrapped about the centre)
            lat = (1.0 - w1) * ((1.0 - w2) * LAT[a, b] + w2 * LAT[a, b + 1]) + w1 * ((1.0 - w2) * LAT[a + 1, b] + w2 * LAT[a + 1, b + 1])
            lon = (1.0 - w1) * ((1.0 - w2) * LON[a, b] + w2 * LON[a, b + 1]) + w1 * ((1.0 - w2) * LON[a + 1, b] + w2 * LON[a + 1, b + 1])
            d = lon - lon0
            lonw = lon0 + (d - 360.0 * math.floor(d / 360.0))
            tl = (lat - lat0) / dla
            tn = (lonw - lon0) / dlo
            if not (tl >= 0.0 and tl <= nla - 1.0 and tn >= 0.0 and tn <= nlo - 1.0):
                flags[2] += 1
                continue
            il = int(math.floor(tl))
            if il > nla - 2:
                il = nla - 2
            jn = int(math.floor(tn))
            if jn > nlo - 2:
                jn = nlo - 2
            fl = tl - il
            fn = tn - jn
            ia = il - i0
            jb = jn - j0
            if ia < 0 or ia + 1 >= na or jb < 0 or jb + 1 >= nbb:
                flags[2] += 1
                continue
            # the height rows: the zenith tables (10 m) and the polynomials (HS_STEP of the kind), both weighted from the absolute height
            kf, wh = _height_row(e, h_first, nh, HEIGHT_STEP)
            kh, wq = _height_row(e, hs[0], nhs, hstep)
            if kf < 0 or kh < 0:
                flags[0] += 1
                continue
            # the incidence: the polynomials of the nearest centre, the two rows around e, linear in height
            ix, iy, dx, dy = _nearest(xcoord[c], ycoord[r], cell, step, kx0, ky0)
            inc = (1.0 - wq) * _poly(inc_t, ix, iy, kh, dx, dy) + wq * _poly(inc_t, ix, iy, kh + 1, dx, dy)
            s_inc = math.cos(inc)
            hz = height_term(hv, ha, hb, hc, e, inc)
            ok[c] = True
            P[c] = ia
            M[c] = jb
            KF[c] = kf
            WH[c] = wh
            S[c] = s_inc
            HZ[c] = hz
            for q in range(4):
                W4[c, q] = (fl if (q >> 1) else 1.0 - fl) * (fn if (q & 1) else 1.0 - fn)
                R4[c, q] = _node_delay(hyd_r, wet_r, map_r, ia + (q >> 1), jb + (q & 1), kf, wh, s_inc, hz)
        # the dates: each reads only its own tables
        for k in range(nd):
            hd = hyd_d[k]
            wd = wet_d[k]
            md = map_d[k]
            for c in range(nx):
                if not ok[c]:
                    out[k, r, c] = np.nan
                    continue
                acc = 0.0
                for q in range(4):
                    acc += W4[c, q] * (_node_delay(hd, wd, md, P[c] + (q >> 1), M[c] + (q & 1), KF[c], WH[c], S[c], HZ[c]) - R4[c, q])
                if acc != acc:
                    flags[1] += 1
                out[k, r, c] = acc


@nb.njit(nogil=True, cache=True, error_model='numpy')
def _grid_iono(ele, ycoord, xcoord, hs, hstep, cell, step, kx0, ky0, pl_t, pn_t, zf_t, tec_d, tec_r, lat0, dla, lon0, dlo, out, flags):
    """The ionosphere phase of a block in one serial pass, every date inside the pass: the TEC of the date minus the reference at the
    pierce point, times the mapping of its effective incidence (see grid_block). Per row, the date-independent part of each pixel
    (the pierce point, its lattice cell, the mapping and the reference's TEC) is computed once into row buffers of nx values; then
    each date reads only its own table. tec_d is (nd, nlat, nlon), tec_r (nlat, nlon), out (nd, ny, nx). flags counts: [0] pixels
    whose height is outside the held rows, [1] NaN table values (pixel-dates), [2] pixels whose pierce point is outside the lattice."""
    nd = tec_d.shape[0]
    ny = ele.shape[0]
    nx = ele.shape[1]
    nla = tec_d.shape[1]
    nlo = tec_d.shape[2]
    nhs = hs.shape[0]
    ok = np.zeros(nx, np.bool_)
    IL = np.zeros(nx, np.int64)
    JN = np.zeros(nx, np.int64)
    FL = np.zeros(nx, np.float64)
    FN = np.zeros(nx, np.float64)
    MF = np.zeros(nx, np.float64)
    DR = np.zeros(nx, np.float64)
    for r in range(ny):
        for c in range(nx):
            ok[c] = False
            e = ele[r, c]
            if e != e:
                continue
            kh, wq = _height_row(e, hs[0], nhs, hstep)
            if kh < 0:
                flags[0] += 1
                continue
            ix, iy, dx, dy = _nearest(xcoord[c], ycoord[r], cell, step, kx0, ky0)
            pl = (1.0 - wq) * _poly(pl_t, ix, iy, kh, dx, dy) + wq * _poly(pl_t, ix, iy, kh + 1, dx, dy)
            pn = (1.0 - wq) * _poly(pn_t, ix, iy, kh, dx, dy) + wq * _poly(pn_t, ix, iy, kh + 1, dx, dy)
            zf = (1.0 - wq) * _poly(zf_t, ix, iy, kh, dx, dy) + wq * _poly(zf_t, ix, iy, kh + 1, dx, dy)
            # the pierce longitude wrapped into [-180, 180) before the lookup
            pn = pn - 360.0 * math.floor((pn + 180.0) / 360.0)
            mf = iono_mapping(zf)
            tl = (pl - lat0) / dla
            tn = (pn - lon0) / dlo
            if not (tl >= 0.0 and tl <= nla - 1.0 and tn >= 0.0 and tn <= nlo - 1.0):
                flags[2] += 1
                continue
            il = int(math.floor(tl))
            if il > nla - 2:
                il = nla - 2
            jn = int(math.floor(tn))
            if jn > nlo - 2:
                jn = nlo - 2
            fl = tl - il
            fn = tn - jn
            ok[c] = True
            IL[c] = il
            JN[c] = jn
            FL[c] = fl
            FN[c] = fn
            MF[c] = mf
            DR[c] = ((1.0 - fl) * ((1.0 - fn) * tec_r[il, jn] + fn * tec_r[il, jn + 1])
                     + fl * ((1.0 - fn) * tec_r[il + 1, jn] + fn * tec_r[il + 1, jn + 1]))
        # the dates: each reads only its own table
        for k in range(nd):
            td = tec_d[k]
            for c in range(nx):
                if not ok[c]:
                    out[k, r, c] = np.nan
                    continue
                il = IL[c]
                jn = JN[c]
                fl = FL[c]
                fn = FN[c]
                dd = ((1.0 - fl) * ((1.0 - fn) * td[il, jn] + fn * td[il, jn + 1])
                      + fl * ((1.0 - fn) * td[il + 1, jn] + fn * td[il + 1, jn + 1]))
                val = (dd - DR[c]) * MF[c]
                if val != val:
                    flags[1] += 1
                out[k, r, c] = val


def grid_block(ele, cut, troposphere=None, ionosphere=None):
    """The phase of one block from its cut, for every date of the call in ONE numba pass: the troposphere or the ionosphere (date minus
    reference), one of the two per call. The date-independent work of each pixel (its lattice cell, height rows, incidence or pierce
    point and the reference's node delays) is done once per pixel, then each date reads only its own tables: a block of k dates costs
    far less than k blocks of one date. Chunking the dates of the stack (k dates per chunk) makes the phase cheaper per date.
    Warns once per block when pixels are NaN for a height outside the held rows, a NaN table value or a pixel outside the model's
    lattice.

    Parameters
    ----------
    ele : numpy.ndarray - The block's pixel heights (ny, nx), metres; NaN gives NaN.
    cut : Cut - The per-chunk step of the block (cut_block).
    troposphere : tuple or None - (hyd_d, wet_d, map_d, hyd_r, wet_r, map_r, hcoef, h_first): the tables of the dates on the box of the
        cut, with a leading date axis (nd, na, nb, nh), the mapping (nd, na, nb, 2, 4); the reference's tables without it (na, nb, nh),
        (na, nb, 2, 4); the height term (v, a, b, c) and the first held height, metres.
    ionosphere : tuple or None - (tec_d, tec_r): the TEC of the dates (nd, nlat, nlon) and of the reference (nlat, nlon) on the whole
        model lattice, phase units.

    Returns
    -------
    numpy.ndarray - The phase (nd, ny, nx), float32, radians.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils_atmosphere import cut_block, grid_block
    >>> y, x = 5220764.0 + 8.0 * np.arange(4), 358279.0 + 2.0 * np.arange(4)
    >>> poly = np.zeros((3, 3, 5, 6))
    >>> pl, pn, zf = poly.copy(), poly.copy(), poly.copy()
    >>> pl[..., 0], pn[..., 0], zf[..., 0] = 47.127, 13.132, 0.3
    >>> tables = dict(hs=1000.0 * np.arange(-1, 4), hstep=1000.0, cell=10000.0, step=100.0, kx0=35, ky0=521, pl=pl, pn=pn, zf=zf)
    >>> cut = cut_block(y, x, {'crs': 'EPSG:32633', 'lon_c': 13.13}, 47.0 + 0.05 * np.arange(8), 13.0 + 0.05 * np.arange(8), tables)
    >>> tec = np.ones((2, 8, 8))
    >>> grid_block(np.full((4, 4), 900.0), cut, ionosphere=(tec, np.zeros((8, 8)))).shape
    (2, 4, 4)
    """
    ele = np.asarray(ele, np.float64)
    ny, nx = cut.ycoord.size, cut.xcoord.size
    if ele.shape != (ny, nx):
        raise ValueError('ERROR: the ele block and the cut differ in shape')
    if (troposphere is None) == (ionosphere is None):
        raise ValueError('ERROR: give the troposphere or the ionosphere tables, one of them')
    const = cut.const
    if 'cell' not in const:
        raise ValueError('ERROR: the cut carries no polynomial tables (cut_block with tables=None or block_tables)')
    flags = np.zeros(3, np.int64)
    if troposphere is not None:
        if const['inc'] is None:
            raise ValueError('ERROR: the troposphere needs the burst constants without a shell (burst_constants)')
        hyd_d, wet_d, map_d, hyd_r, wet_r, map_r, hcoef, h_first = troposphere
        arrays = [np.ascontiguousarray(v, np.float64) for v in (hyd_d, wet_d, map_d, hyd_r, wet_r, map_r)]
        box = (cut.i1 - cut.i0, cut.j1 - cut.j0)
        if any(a.shape[1:3] != box for a in arrays[:3]) or any(a.shape[:2] != box for a in arrays[3:]):
            raise ValueError('ERROR: the troposphere tables and the cut differ in the box of the model lattice')
        if len({a.shape[0] for a in arrays[:3]}) != 1:
            raise ValueError('ERROR: the troposphere tables of the dates differ in the number of dates')
        out = np.empty((arrays[0].shape[0], ny, nx), np.float32)
        _grid_tropo(ele, cut.ycoord, cut.xcoord, cut.jy, cut.wy, cut.jx, cut.wx, cut.LAT, cut.LON, cut.lat0, cut.dla,
                    cut.lon0, cut.dlo, cut.nla, cut.nlo, cut.i0, cut.j0, *arrays,
                    np.ascontiguousarray(hcoef, np.float64), float(h_first), const['hs'], const['hstep'], const['cell'], const['step'],
                    const['kx0'], const['ky0'], const['inc'], out, flags)
    else:
        if const['pl'] is None:
            raise ValueError('ERROR: the ionosphere needs the burst constants of a shell (burst_constants)')
        tec_d, tec_r = (np.ascontiguousarray(v, np.float64) for v in ionosphere)
        if tec_d.shape[1:] != (cut.nla, cut.nlo) or tec_r.shape != (cut.nla, cut.nlo):
            raise ValueError('ERROR: the ionosphere tables and the cut differ in the model lattice')
        out = np.empty((tec_d.shape[0], ny, nx), np.float32)
        _grid_iono(ele, cut.ycoord, cut.xcoord, const['hs'], const['hstep'], const['cell'], const['step'], const['kx0'], const['ky0'],
                   const['pl'], const['pn'], const['zf'], tec_d, tec_r, cut.lat0, cut.dla, cut.lon0, cut.dlo, out, flags)
    parts = []
    if flags[0]:
        parts.append(f'{int(flags[0])} pixel(s) lie at a height outside the held atmosphere rows')
    if flags[1]:
        parts.append(f'{int(flags[1])} pixel-dates take a NaN table value (deeper below the model surface than the tables '
                     f'reach, or no data)')
    if flags[2]:
        parts.append(f'{int(flags[2])} pixel(s) lie outside the atmosphere lattice')
    if parts:
        warnings.warn('; '.join(parts) + ': their phase is NaN')
    return out


def phase_block(kind, ele, axes, span, static, tables, plan, *refs):
    """The atmosphere phase of one data block for ALL of its dates in one task: the per-chunk step (cut_block, with the chunk's tables)
    and ONE grid kernel call (grid_block) for every date of the plan, so the date-independent work of each pixel is done once, not
    once per date. The block's pixel coordinates are not passed: they are computed here from the transform's lattice (pixel_axis).
    The held tables arrive as flat arguments (refs): per date of the plan, the tables of the date and, when the plan says so, of
    the reference, each for the tiles the plan names (troposphere: hydro, wet, mapping per tile; ionosphere: the tec lattice). The
    dates of one block share their tiles and their reference (the Stack's plan always does). Chunking the dates of the stack (k dates
    per chunk) makes the phase cheaper per date: one task and one kernel call per stored block for its k dates.

    Parameters
    ----------
    kind : str - 'tropo' or 'iono'.
    ele : numpy.ndarray - The block's pixel heights (ny, nx), metres.
    axes : tuple of float - (y_first, y_step, x_first, x_step), map units: the first pixel centre and the spacing per axis
        (step 0.0 for an axis of one pixel).
    span : tuple of int - (i0, i1, j0, j1): the block's pixel rows and columns in its burst.
    static : dict - The burst constants (const), the model's node_lat and node_lon (degrees), h_first (the first held height) and
        hcoef (v, a, b, c of the height term).
    tables : dict - The chunk's polynomial tables (block_tables), built once at the graph build.
    plan : list of (tuple, bool) - Per date: the tiles (y0, x0) of the box and whether the reference is subtracted; the same for
        every date.
    refs : numpy.ndarray - The tables of the plan, in its order.

    Returns
    -------
    numpy.ndarray - The phase (len(plan), ny, nx), float32, radians.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev.utils_atmosphere import phase_block
    >>> poly = np.zeros((3, 3, 5, 6))
    >>> pl, pn, zf = poly.copy(), poly.copy(), poly.copy()
    >>> pl[..., 0], pn[..., 0], zf[..., 0] = 47.127, 13.132, 0.3
    >>> tables = dict(hs=1000.0 * np.arange(-1, 4), hstep=1000.0, cell=10000.0, step=100.0, kx0=35, ky0=521, pl=pl, pn=pn, zf=zf)
    >>> static = dict(const={'crs': 'EPSG:32633', 'lon_c': 13.13}, node_lat=47.0 + 0.05 * np.arange(8),
    ...               node_lon=13.0 + 0.05 * np.arange(8), h_first=0.0, hcoef=np.zeros(4))
    >>> tec = np.ones((1, 8, 8))
    >>> phase_block('iono', np.full((4, 4), 900.0), (5220764.0, 8.0, 358279.0, 2.0), (0, 4, 0, 4), static, tables,
    ...             [((), True)] * 2, 2 * tec, tec, 3 * tec, tec).shape
    (2, 4, 4)
    """
    ele = np.asarray(ele, np.float64)
    ny, nx = ele.shape
    y_first, y_step, x_first, x_step = axes
    i0, i1, j0, j1 = span
    if not plan:
        return np.empty((0, ny, nx), np.float32)
    tiles, has_ref = plan[0]
    if any(p[0] != tiles or p[1] != has_ref for p in plan):
        raise ValueError('ERROR: the dates of one block must share their tiles and their reference')
    ycoord = pixel_axis(y_first, y_step, i0, i1)
    xcoord = pixel_axis(x_first, x_step, j0, j1)
    cut = cut_block(ycoord, xcoord, static['const'], static['node_lat'], static['node_lon'], tables)
    n = 3 * len(tiles) if kind == 'tropo' else 1
    stride = n * (2 if has_ref else 1)
    dates = [refs[k * stride:k * stride + n] for k in range(len(plan))]
    ref = refs[n:2 * n] if has_ref else None
    if has_ref:
        for k in range(1, len(plan)):
            r_k = refs[k * stride + n:k * stride + 2 * n]
            if any(a is not b and not np.array_equal(a, b, equal_nan=True) for a, b in zip(r_k, ref)):
                raise ValueError('ERROR: the dates of one block must share their reference')
    if kind == 'tropo':
        box = (cut.i0, cut.i1, cut.j0, cut.j1)
        # a held chunk is (1 date, 1 tile, ...): its own tile values
        hyd_d, wet_d, map_d = (np.stack([gather(tiles, [np.asarray(a)[0, 0] for a in d[q::3]], *box) for d in dates]) for q in range(3))
        if ref is None:
            hyd_r, wet_r, map_r = np.zeros_like(hyd_d[0]), np.zeros_like(wet_d[0]), np.zeros_like(map_d[0])
        else:
            hyd_r, wet_r, map_r = (gather(tiles, [np.asarray(a)[0, 0] for a in ref[q::3]], *box) for q in range(3))
        return grid_block(ele, cut, troposphere=(hyd_d, wet_d, map_d, hyd_r, wet_r, map_r, static['hcoef'], static['h_first']))
    # a held chunk is (1 date, lat, lon): the whole lattice of the date
    tec_d = np.stack([np.asarray(d[0])[0] for d in dates])
    tec_r = np.zeros_like(tec_d[0]) if ref is None else np.asarray(ref[0])[0]
    return grid_block(ele, cut, ionosphere=(tec_d, tec_r))
