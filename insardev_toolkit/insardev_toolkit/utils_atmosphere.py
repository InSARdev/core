#!/usr/bin/env python
# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""Atmosphere model evaluator for ERA5, VMF3 and IONEX: each node holds zenith delay tables over the ellipsoidal height.

A pixel is bilinear over the 4 nodes of its cell: in the output CRS for ERA5 and VMF3, in latitude and longitude for IONEX.
"""
import math
import warnings

import numpy as np
import pandas as pd
from numba import njit

_EPS = 1e-9


def atm_fill_nodata(grid):
    """Fill the NaN nodes of a 2-D atmosphere map by linear interpolation, then by the nearest valid node at the border.

    Parameters
    ----------
    grid : numpy.ndarray - 2-D map (lat x lon) of node values; NaN marks a no-data node.

    Returns
    -------
    tuple - The filled map and the count of filled nodes; a map without NaN is returned as it is.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import atm_fill_nodata
    >>> grid = np.array([[1.0, 2.0, 3.0], [2.0, np.nan, 4.0], [3.0, 4.0, 5.0]])
    >>> filled, count = atm_fill_nodata(grid)
    >>> round(float(filled[1, 1]), 6), count
    (3.0, 1)
    """
    from scipy.interpolate import griddata
    from scipy.spatial import QhullError
    grid = np.asarray(grid)
    mask = np.isnan(grid)
    count = int(mask.sum())
    if count == 0:
        return grid, 0
    if count == grid.size:
        raise ValueError('ERROR: atmosphere map has no valid node to fill from')
    valid = np.argwhere(~mask).astype(np.float64)
    values = grid[~mask]
    query = np.argwhere(mask).astype(np.float64)
    try:
        filled = griddata(valid, values, query, method='linear')
    except QhullError:
        # fewer than three valid nodes or valid nodes on one line: linear reaches no node
        filled = np.full(len(query), np.nan)
    empty = np.isnan(filled)
    if empty.any():
        filled[empty] = griddata(valid, values, query[empty], method='nearest')
    out = grid.copy()
    out[mask] = filled
    return out, count


@njit(nogil=True, cache=True)
def _inverse(px, py, ax, ay, bx, by, cx, cy, dx, dy):
    """Bilinear coordinates (u, v) of a point in a quad, clipped to [0, 1].

    Parameters
    ----------
    px, py : float - The point, in the units of the quad.
    ax, ay, bx, by, cx, cy, dx, dy : float - The quad corners a, b, c and d in order.

    Returns
    -------
    tuple - (found, u, v); found is True when the point lies inside the quad.
    """
    ex = bx - ax
    ey = by - ay
    fx = dx - ax
    fy = dy - ay
    gx = ax - bx + cx - dx
    gy = ay - by + cy - dy
    hx = px - ax
    hy = py - ay
    k2 = gx * fy - gy * fx
    k1 = (ex * fy - ey * fx) + (hx * gy - hy * gx)
    k0 = hx * ey - hy * ex
    if k2 == 0.0:
        if k1 == 0.0:
            return False, 0.0, 0.0
        r1 = -k0 / k1
        r2 = r1
    else:
        disc = k1 * k1 - 4.0 * k2 * k0
        if disc < 0.0:
            if disc < -1e-12 * (k1 * k1 + abs(4.0 * k2 * k0)):
                return False, 0.0, 0.0
            disc = 0.0
        sq = math.sqrt(disc)
        if k1 >= 0.0:
            q = -0.5 * (k1 + sq)
        else:
            q = -0.5 * (k1 - sq)
        r1 = q / k2
        r2 = k0 / q if q != 0.0 else r1
    for v in (r1, r2):
        if v < -_EPS or v > 1.0 + _EPS:
            continue
        exv = ex + v * gx
        eyv = ey + v * gy
        fxv = hx - v * fx
        fyv = hy - v * fy
        den = exv * exv + eyv * eyv
        if den == 0.0:
            continue
        u = (exv * fxv + eyv * fyv) / den
        if u >= -_EPS and u <= 1.0 + _EPS:
            return True, min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
    return False, 0.0, 0.0


@njit(nogil=True, cache=True)
def _locate_all(px, py, X, Y, nxn, meta, imeta, start, items, ncol, cell, uu, vv):
    """The cell of the node lattice that holds each pixel, -1 where no cell does (of the holding cells, the smallest).

    Parameters
    ----------
    px, py : numpy.ndarray - The pixel coordinates; output CRS (ERA5, VMF3) or longitude and latitude (IONEX).
    X, Y : numpy.ndarray - The node coordinates, flattened in rows of nxn; output CRS (ERA5, VMF3) or longitude and latitude (IONEX).
    meta, imeta, start, items : numpy.ndarray - The bucket index of the cells.
    nxn, ncol : int - The row length of the node arrays and the number of cell columns.
    cell, uu, vv : numpy.ndarray - Output: the cell of each pixel and its bilinear coordinates in that cell.

    Returns
    -------
    int - The count of pixels that no cell holds.
    """
    x0 = meta[0]
    y0 = meta[1]
    xmax = meta[2]
    ymax = meta[3]
    bdx = meta[4]
    bdy = meta[5]
    nbx = imeta[0]
    nby = imeta[1]
    missing = 0
    for k in range(px.shape[0]):
        cell[k] = -1
        x = px[k]
        y = py[k]
        if not (x >= x0 and x <= xmax and y >= y0 and y <= ymax):
            missing += 1
            continue
        bx = min(int((x - x0) / bdx), nbx - 1)
        by = min(int((y - y0) / bdy), nby - 1)
        b = by * nbx + bx
        best = 0.0
        for q in range(start[b], start[b + 1]):
            c = items[q]
            j = c // ncol
            i = c - j * ncol
            a = j * nxn + i
            found, u, v = _inverse(x, y, X[a], Y[a], X[a + 1], Y[a + 1], X[a + nxn + 1], Y[a + nxn + 1], X[a + nxn], Y[a + nxn])
            if found:
                # the smallest holding quad: a cell stretched across the projection cut holds the pixel too
                ex = X[a + 1] - X[a]
                ey = Y[a + 1] - Y[a]
                fx = X[a + nxn + 1] - X[a]
                fy = Y[a + nxn + 1] - Y[a]
                gx = X[a + nxn] - X[a]
                gy = Y[a + nxn] - Y[a]
                s = abs(ex * fy - ey * fx) + abs(fx * gy - fy * gx)
                if cell[k] < 0 or s < best:
                    cell[k] = c
                    uu[k] = u
                    vv[k] = v
                    best = s
        if cell[k] < 0:
            missing += 1
    return missing


@njit(nogil=True, cache=True, error_model='numpy')
def _height_slot(h, h0, step, count):
    """The table entry index and weight of an ellipsoidal height, by linear interpolation between entries.

    Parameters
    ----------
    h : float - The ellipsoidal height, metres; NaN gives a NaN weight.
    h0 : float - The height of the first entry, metres.
    step : float - The entry spacing, metres.
    count : int - The number of entries, at least 2.

    Returns
    -------
    tuple - (i, w): the entry i and the weight w of entries i and i + 1.

    Raises
    ------
    ValueError: When h is outside the first and the last entries.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import _height_slot
    >>> _height_slot(2.5, 0.0, 5.0, 2)
    (0, 0.5)
    """
    f = (h - h0) / step
    if f != f:
        return 0, np.nan
    if not (f >= 0.0 and f <= count - 1):
        raise ValueError('ERROR: a pixel height is outside the height range of the atmosphere model tables')
    i = min(int(f), count - 2)
    return i, f - i


@njit(nogil=True, cache=True, error_model='numpy')
def _form(a, b, c, s):
    """The dimensionless mapping factor of the coefficients a, b, c at s = cos of the incidence.

    Parameters
    ----------
    a, b, c : float - The dimensionless mapping coefficients.
    s : float - cos of the incidence, dimensionless.

    Returns
    -------
    float - The mapping factor; with a = b = c = 0 it is 1 / s.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import _form
    >>> round(_form(0.0, 0.0, 0.0, 0.5), 6)
    2.0
    """
    if a == 0.0 and b == 0.0 and c == 0.0:
        return 1.0 / s
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (s + a / (s + b / (s + c)))


@njit(nogil=True, cache=True, error_model='numpy')
def _mapping_form(a, b, c, K, s):
    """The mapping factor in one division, with the constant K of the coefficients.

    Parameters
    ----------
    a, b, c, K : float - The coefficients and the constant K.
    s : float - cos of the incidence, dimensionless.

    Returns
    -------
    float - The dimensionless mapping factor.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import _mapping_form
    >>> round(_mapping_form(0.0, 0.0, 0.0, 1.0, 0.5), 6)
    2.0
    """
    r = s + c
    num = s * r + b
    return K * num / (s * num + a * r)


def _mapping_coefs(mapping):
    """Append the constant K to the coefficients (a, b, c) of each component.

    Parameters
    ----------
    mapping : numpy.ndarray - Coefficients with the last axis (a, b, c).

    Returns
    -------
    numpy.ndarray - The last axis extended to (a, b, c, K).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _mapping_coefs
    >>> _mapping_coefs(np.zeros((1, 1, 1, 2, 3)))[..., 3].ravel().tolist()
    [1.0, 1.0]
    """
    a, b, c = mapping[..., 0], mapping[..., 1], mapping[..., 2]
    q = 1.0 + c
    return np.concatenate([mapping, ((q + b + a * q) / (q + b))[..., None]], axis=-1)


@njit(nogil=True, cache=True, error_model='numpy')
def _node_phase(hyd, wet, mapping, t, r, c, i, w, s, hterm):
    """The phase in radians of one node at one time for one pixel: the zenith delay tables times their mapping, plus the
    height term.

    Parameters
    ----------
    hyd, wet : numpy.ndarray - The zenith delay tables in phase units, indexed (time, row, col, height).
    mapping : numpy.ndarray - The (a, b, c, K) coefficients per component.
    t, r, c : int - The time index and the node row and column.
    i, w : int, float - The height entry and its weight.
    s : float - cos of the incidence, dimensionless.
    hterm : float - The height term, added to the hydrostatic mapping.

    Returns
    -------
    float - The phase, radians.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _node_phase
    >>> hyd = np.ones((1, 1, 1, 2)); wet = np.zeros((1, 1, 1, 2)); mp = np.zeros((1, 1, 1, 2, 4)); mp[..., 3] = 1.0
    >>> round(_node_phase(hyd, wet, mp, 0, 0, 0, 0, 0.5, 1.0, 0.0), 6)
    1.0
    """
    zh = (1.0 - w) * hyd[t, r, c, i] + w * hyd[t, r, c, i + 1]
    zw = (1.0 - w) * wet[t, r, c, i] + w * wet[t, r, c, i + 1]
    mh = _mapping_form(mapping[t, r, c, 0, 0], mapping[t, r, c, 0, 1], mapping[t, r, c, 0, 2], mapping[t, r, c, 0, 3], s) + hterm
    mw = _mapping_form(mapping[t, r, c, 1, 0], mapping[t, r, c, 1, 1], mapping[t, r, c, 1, 2], mapping[t, r, c, 1, 3], s)
    return zh * mh + zw * mw


@njit(nogil=True, cache=True, error_model='numpy')
def _shell_phase(hyd, wet, t, r, c, m, inv):
    """The phase in radians of one node at one time for one pixel of a shell model: the table entry times the shell mapping.

    Parameters
    ----------
    hyd, wet : numpy.ndarray - The table entries, indexed (time, row, col, entry).
    t, r, c : int - The time index and the node row and column.
    m, inv : float - The shell mapping, 1 / s, for both components.

    Returns
    -------
    float - The phase, radians.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _shell_phase
    >>> hyd = np.full((1, 1, 1, 1), 2.0); wet = np.zeros((1, 1, 1, 1))
    >>> _shell_phase(hyd, wet, 0, 0, 0, 1.5, 1.5)
    3.0
    """
    return hyd[t, r, c, 0] * m + wet[t, r, c, 0] * inv


@njit(nogil=True, cache=True, error_model='numpy')
def _table_gap(hyd, wet, t, r, c, i, count):
    """True when a node at one time has a gap at a slot: a NaN entry at the slot or the next one, with a finite top entry.

    Parameters
    ----------
    hyd, wet : numpy.ndarray - The zenith delay tables, indexed (time, row, col, height).
    t, r, c, i : int - The time index, the node row and column, and the slot.
    count : int - The number of table entries.

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _table_gap
    >>> hyd = np.array([[[[np.nan, 1.0, 2.0]]]]); wet = hyd * 0.1
    >>> _table_gap(hyd, wet, 0, 0, 0, 0, 3), _table_gap(hyd, wet, 0, 0, 0, 1, 3)
    (True, False)
    """
    top = hyd[t, r, c, count - 1] == hyd[t, r, c, count - 1] and wet[t, r, c, count - 1] == wet[t, r, c, count - 1]
    a = hyd[t, r, c, i] + hyd[t, r, c, i + 1] + wet[t, r, c, i] + wet[t, r, c, i + 1]
    return top and a != a


@njit(nogil=True, cache=True, error_model='numpy')
def _pixel_kernel(hyd, wet, mapping, loc, lr, lc, uu, vv, h, z, h0, step, count, niell, k, gaps, out):
    """Phase of a pixel chunk on one tile: out[p, q] is the phase (radians, float32) of pixel q at the time loc[p].

    Each pixel is bilinear over the 4 nodes of its cell; a pixel with a NaN height or incidence is NaN for every model,
    with no raise and no warning.

    Parameters
    ----------
    hyd, wet : numpy.ndarray - The zenith tables of the tile, (time, y, x, height), in phase units.
    mapping : numpy.ndarray - The (a, b, c, K) coefficients of the tile, (time, y, x, component, 4).
    loc : numpy.ndarray - The index into the tables of each time.
    lr, lc : numpy.ndarray - The local lower-left node of each pixel's cell.
    uu, vv : numpy.ndarray - The bilinear weights of each pixel in its cell.
    h : numpy.ndarray - The ellipsoidal height of each pixel, metres.
    z : numpy.ndarray - The incidence of each pixel, radians.
    h0, step : float - The first table height and the entry spacing, metres.
    count : int - The number of table entries.
    niell : tuple of float - The height term (v, a, b, c); v is 0 for IONEX.
    k : float - The shell factor (IONEX); 1 for the mapping forms.
    gaps : numpy.ndarray - Output: gaps[0] counts the pixel-dates whose phase is NaN from a table gap.
    out : numpy.ndarray - Output: float32 phases, radians, shape (times, pixels).

    Raises
    ------
    ValueError: For an incidence outside [0, 90) degrees or a height outside the tables.
    """
    npos = loc.size
    for q in range(h.size):
        if h[q] != h[q] or z[q] != z[q]:
            for p in range(npos):
                out[p, q] = np.float32(np.nan)
            continue
        zd = z[q] * (180.0 / np.pi)
        if not (zd >= 0.0 and zd < 90.0):
            raise ValueError('ERROR: the incidence of the burst is outside [0, 90) degrees')
        cz = np.cos(z[q])
        sz = np.sin(z[q])
        s = np.sqrt(cz * cz + (1.0 - k * k) * sz * sz)
        inv = 1.0 / s
        r0 = lr[q]
        c0 = lc[q]
        u = uu[q]
        v = vv[q]
        w00 = (1.0 - u) * (1.0 - v)
        w10 = u * (1.0 - v)
        w11 = u * v
        w01 = (1.0 - u) * v
        if count == 1:
            m = inv
            for p in range(npos):
                t = loc[p]
                acc = 0.0
                if w00 != 0.0:
                    acc += w00 * _shell_phase(hyd, wet, t, r0, c0, m, inv)
                if w10 != 0.0:
                    acc += w10 * _shell_phase(hyd, wet, t, r0, c0 + 1, m, inv)
                if w11 != 0.0:
                    acc += w11 * _shell_phase(hyd, wet, t, r0 + 1, c0 + 1, m, inv)
                if w01 != 0.0:
                    acc += w01 * _shell_phase(hyd, wet, t, r0 + 1, c0, m, inv)
                out[p, q] = np.float32(acc)
        else:
            hterm = 0.0
            if niell[0] != 0.0:
                hterm = niell[0] * h[q] * (1.0 / cz - _form(niell[1], niell[2], niell[3], s))
            i, wh = _height_slot(h[q], h0, step, count)
            for p in range(npos):
                t = loc[p]
                acc = 0.0
                if w00 != 0.0:
                    acc += w00 * _node_phase(hyd, wet, mapping, t, r0, c0, i, wh, s, hterm)
                if w10 != 0.0:
                    acc += w10 * _node_phase(hyd, wet, mapping, t, r0, c0 + 1, i, wh, s, hterm)
                if w11 != 0.0:
                    acc += w11 * _node_phase(hyd, wet, mapping, t, r0 + 1, c0 + 1, i, wh, s, hterm)
                if w01 != 0.0:
                    acc += w01 * _node_phase(hyd, wet, mapping, t, r0 + 1, c0, i, wh, s, hterm)
                out[p, q] = np.float32(acc)
                if acc != acc and h[q] == h[q]:
                    gap = False
                    if w00 != 0.0:
                        gap = gap or _table_gap(hyd, wet, t, r0, c0, i, count)
                    if w10 != 0.0:
                        gap = gap or _table_gap(hyd, wet, t, r0, c0 + 1, i, count)
                    if w11 != 0.0:
                        gap = gap or _table_gap(hyd, wet, t, r0 + 1, c0 + 1, i, count)
                    if w01 != 0.0:
                        gap = gap or _table_gap(hyd, wet, t, r0 + 1, c0, i, count)
                    if gap:
                        gaps[0] += 1
    return out


def _nodes_xy(lon, lat, crs):
    """The nodes in the output CRS as x and y.

    Parameters
    ----------
    lon, lat : numpy.ndarray - The node longitudes and latitudes, degrees, one shape.
    crs : int, str, rasterio CRS or None - The output CRS; None gives lon and lat.

    Returns
    -------
    tuple of numpy.ndarray - x and y, float64, the shape of lon.
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs('EPSG:4326', 4326 if crs is None else crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64).reshape(lon.shape), np.asarray(y, dtype=np.float64).reshape(lon.shape)


def _cell_owner(y0, x0, ty, tx, ny, nx):
    """The tile index of each cell of a lattice, -1 where no tile holds the cell.

    Parameters
    ----------
    y0, x0 : numpy.ndarray - The first node row and column of each tile.
    ty, tx : int - The tile size in nodes.
    ny, nx : int - The lattice size in nodes.

    Returns
    -------
    numpy.ndarray - The int32 tile index of each of the (ny - 1) * (nx - 1) cells, row by row.

    Examples
    --------
    >>> _cell_owner(np.array([0, 4]), np.array([0, 0]), 5, 5, 9, 5)
    """
    owner = np.full((max(ny - 1, 0), max(nx - 1, 0)), -1, dtype=np.int32)
    for k in range(len(y0)):
        owner[y0[k]:y0[k] + ty - 1, x0[k]:x0[k] + tx - 1] = k
    return owner.ravel()


def _cell_table(X, Y, owner):
    """The cells of a node lattice, indexed for locating pixels.

    Parameters
    ----------
    X, Y : numpy.ndarray - The node coordinates, shape (ny, nx); output CRS (ERA5, VMF3) or longitude and latitude (IONEX).
    owner : numpy.ndarray - The tile index of each cell.

    Returns
    -------
    tuple - (meta, imeta, start, items, ncol, owner): the bucket index of the cells and the tile of each cell.
    """
    ny, nx = X.shape
    ncol = nx - 1
    empty = (np.array([0.0, 0.0, -1.0, -1.0, 1.0, 1.0]), np.array([1, 1], dtype=np.int64),
             np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64), max(ncol, 1), owner)
    if ny < 2 or nx < 2:
        return empty
    CX = np.stack([X[:-1, :-1], X[:-1, 1:], X[1:, 1:], X[1:, :-1]])
    CY = np.stack([Y[:-1, :-1], Y[:-1, 1:], Y[1:, 1:], Y[1:, :-1]])
    finite = np.isfinite(CX).all(axis=0) & np.isfinite(CY).all(axis=0)
    ids = np.flatnonzero(finite.ravel())
    if ids.size == 0:
        return empty
    cx0 = CX.min(axis=0).ravel()[ids]
    cx1 = CX.max(axis=0).ravel()[ids]
    cy0 = CY.min(axis=0).ravel()[ids]
    cy1 = CY.max(axis=0).ravel()[ids]
    x0, y0 = cx0.min(), cy0.min()
    xmax, ymax = cx1.max(), cy1.max()
    W = xmax - x0
    H = ymax - y0
    aspect = W / H if (W > 0 and H > 0) else 1.0
    nbx = max(1, int(round(np.sqrt(ids.size * aspect))))
    nby = max(1, int(round(np.sqrt(ids.size / aspect))))
    bdx = W / nbx if W > 0 else 1.0
    bdy = H / nby if H > 0 else 1.0
    buckets = [[] for _ in range(nbx * nby)]
    for k in range(ids.size):
        ix0 = min(int((cx0[k] - x0) / bdx), nbx - 1)
        ix1 = min(int((cx1[k] - x0) / bdx), nbx - 1)
        iy0 = min(int((cy0[k] - y0) / bdy), nby - 1)
        iy1 = min(int((cy1[k] - y0) / bdy), nby - 1)
        for by in range(iy0, iy1 + 1):
            for bx in range(ix0, ix1 + 1):
                buckets[by * nbx + bx].append(int(ids[k]))
    start = np.zeros(nbx * nby + 1, dtype=np.int64)
    start[1:] = np.cumsum([len(b) for b in buckets])
    items = np.array([c for b in buckets for c in b], dtype=np.int64)
    meta = np.array([x0, y0, xmax, ymax, bdx, bdy], dtype=np.float64)
    imeta = np.array([nbx, nby], dtype=np.int64)
    return meta, imeta, start, items, ncol, owner


_A_WGS = 6378137.0                                                # m, WGS84 semi-major axis
_E2_WGS = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)      # WGS84 first eccentricity squared


def _control(y, x, crs, k=9):
    """The control mesh of a projected CRS for the pierce point: the latitude and longitude of up to k x k pixels of the
    whole grid (one pyproj call on them); None for no CRS or a geographic CRS.

    Parameters
    ----------
    y, x : numpy.ndarray - The pixel axes in the output CRS, 1-D or 2-D of one shape.
    crs : int, str, rasterio CRS or None - The output CRS.
    k : int - The most control points along an axis.

    Returns
    -------
    tuple or None - (ri, ci, lat, lon): the control rows and columns (indices) and the latitudes and longitudes in degrees,
    shape (len(ri), len(ci)).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _control
    >>> _ = _control(np.array([5.2e6, 5.21e6]), np.array([3.5e5, 3.6e5]), 32633)
    """
    from pyproj import CRS, Transformer
    if crs is None or CRS.from_user_input(crs).is_geographic:
        return None
    ny, nx = (y.size, x.size) if y.ndim == 1 else y.shape
    ri, ci = (np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(np.int64)) for n in (ny, nx))
    cy, cx = (y[ri][:, None], x[ci][None, :]) if y.ndim == 1 else (y[np.ix_(ri, ci)], x[np.ix_(ri, ci)])
    cy, cx = np.broadcast_arrays(np.asarray(cy, dtype=np.float64), np.asarray(cx, dtype=np.float64))
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(cx, cy)
    return ri, ci, np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def _lerp(vals, ri, ci, rows, cols):
    """Bilinear interpolation in index space of control values on the control rows ri and columns ci to the pixels rows
    x cols (global indices)."""
    def weights(idx, pos):
        if idx.size == 1:
            return np.zeros(pos.size, dtype=np.int64), np.zeros(pos.size, dtype=np.int64), np.zeros(pos.size)
        j = np.clip(np.searchsorted(idx, pos, side='right') - 1, 0, idx.size - 2)
        return j, j + 1, (pos - idx[j]) / (idx[j + 1] - idx[j])
    j0, j1, t = weights(ci, np.asarray(cols, dtype=np.float64))
    part = vals[:, j0] * (1.0 - t) + vals[:, j1] * t
    i0, i1, s = weights(ri, np.asarray(rows, dtype=np.float64))
    return part[i0] * (1.0 - s)[:, None] + part[i1] * s[:, None]


def _pierce_offset(yy, xx, ele, inc, look, shell, control, geocentric, rows, cols):
    """The IONEX pierce point of each pixel by the sine law, and the effective incidence of its shell mapping.

    With R = |target ECEF|, S = R + H = shell and zc the geocentric zenith of the ray at the target: beta = zc -
    arcsin(R sin zc / S), d = R beta. The pierce point lies at d along the geodesic of the look azimuth + 180 degrees
    (the great circle of the geocentric frame, geodetic latitude) in latitude and longitude, whatever the output CRS.
    A projected CRS takes the pixel latitude and longitude interpolated from the control mesh _control, so chunking does
    not matter. The pierce point is the same for both conventions (TN-200 step 2); only the mapping zenith differs.

    Parameters
    ----------
    yy, xx : numpy.ndarray - The pixel coordinates, shape (ny, nx): latitude and longitude for a geographic CRS; the output
        CRS for a projected one (then control gives the latitude and longitude).
    ele, inc, look : numpy.ndarray - The ellipsoidal heights (metres), incidences (radians) and look azimuths
        (degrees clockwise from north, satellite to target) of the pixels.
    shell : float - R + H, metres.
    control : tuple or None - The control mesh of _control; None for a geographic CRS.
    geocentric : bool - True: the mapping at the geocentric zenith of the exact ray (ETAD ATBD 2.7 Eq. 45-46, the default);
        False: TN-200 Eq. 22 at the ellipsoid-normal zenith z (FRM4SAR TN-200), for verifications against TN-200.
    rows, cols : numpy.ndarray - The indices of the pixel rows and columns in the whole grid.

    Returns
    -------
    tuple of numpy.ndarray - (lat, lon, zeff): the pierce point in degrees and the effective incidence z' in radians,
    shape (ny, nx).

    Raises
    ------
    ValueError: For an incidence outside [0, 90) degrees.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _pierce_offset
    >>> _ = _pierce_offset(np.full((2, 2), 47.0), np.full((2, 2), 13.0), np.zeros((2, 2)),
    ...                    np.full((2, 2), 0.6), np.full((2, 2), 77.0), 6821e3, None, True, np.arange(2), np.arange(2))
    """
    z = np.asarray(inc, dtype=np.float64)
    zd = np.degrees(z)
    if (~np.isnan(zd) & ~((zd >= 0.0) & (zd < 90.0))).any():
        raise ValueError('ERROR: the incidence of the burst is outside [0, 90) degrees')
    yy, xx = np.asarray(yy, dtype=np.float64), np.asarray(xx, dtype=np.float64)
    if control is None:
        lat, lon = yy, xx
    else:
        ri, ci, lat_c, lon_c = control
        lat, lon = _lerp(lat_c, ri, ci, rows, cols), _lerp(lon_c, ri, ci, rows, cols)
    phi = np.radians(lat)
    big_n = _A_WGS / np.sqrt(1.0 - _E2_WGS * np.sin(phi) ** 2)
    h = np.asarray(ele, dtype=np.float64)
    pt = (big_n + h) * np.cos(phi)                                    # T = (pt cos lon, pt sin lon, zt), ECEF
    zt = (big_n * (1.0 - _E2_WGS) + h) * np.sin(phi)
    R = np.hypot(pt, zt)
    # geocentric frame: the radial is the normal tilted by d = phi - phi_c; the LOS in ENU, azimuth look + 180
    sd, cd = (np.sin(phi) * pt - np.cos(phi) * zt) / R, (np.cos(phi) * pt + np.sin(phi) * zt) / R
    a = np.radians(np.asarray(look, dtype=np.float64) + 180.0)
    ue, un, uu = np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)
    uh = cd * un + sd * uu                                        # LOS north component, geocentric
    zc = np.arctan2(np.hypot(ue, uh), -sd * un + cd * uu)         # angle at T between the radial and the LOS
    az = np.arctan2(ue, uh)                                       # azimuth of the LOS, geocentric
    zray = np.arcsin(R * np.sin(zc) / shell)                      # the zenith at the pierce point (sine law)
    beta = zc - zray                                              # the ray: sine law at the geocentric angle zc
    zeff = zray if geocentric else np.arcsin(R * np.sin(z) / shell)   # the mapping: ETAD, or TN-200 Eq. 22 at the normal z
    sf, cf = zt / R, pt / R
    s2 = sf * np.cos(beta) + cf * np.sin(beta) * np.cos(az)           # sin of the pierce latitude, geocentric
    dl = np.arctan2(np.sin(az) * np.sin(beta) * cf, np.cos(beta) - sf * s2)
    p, zz = shell * np.sqrt(1.0 - s2 * s2), shell * s2                # the pierce point in ECEF
    # geodetic latitude of the pierce point: TN-200 Eq. 36 (Bowring)
    b = _A_WGS * np.sqrt(1.0 - _E2_WGS)
    th = np.arctan2(zz * _A_WGS, p * b)
    lat_g = np.arctan2(zz + _E2_WGS / (1.0 - _E2_WGS) * b * np.sin(th) ** 3, p - _E2_WGS * _A_WGS * np.cos(th) ** 3)
    return np.degrees(lat_g), lon + np.degrees(dl), zeff


def _chunk_cells(y, x, ny, nx, yoff, xoff, chunk, nodes, index, pier=None):
    """The cell of each pixel of one output chunk with its bilinear coordinates; with pier, the cell of its IONEX pierce point
    (in the latitude and longitude nodes).

    Parameters
    ----------
    y, x : numpy.ndarray - The pixel axes, 1-D or 2-D of one shape, in the output CRS.
    ny, nx : int - The chunk size in pixels.
    yoff, xoff : sequence of int - The chunk offsets of the axes.
    chunk : tuple of int - The chunk index (bi, bj).
    nodes, index : tuple - The lattice nodes and the cell index.
    pier : None or tuple - None, or (ele, inc, look, shell, control, geocentric) of the chunk, see _pierce_offset.

    Returns
    -------
    tuple - Flat arrays of the pixels: the cell (-1 where none holds the pixel), u, v, the count of pixels no cell
    holds, and with pier the effective incidence (NaN where the pierce point is NaN; else None).
    """
    bi, bj = chunk
    if y.ndim == 1:
        yy, xx = np.meshgrid(y[yoff[bi]:yoff[bi] + ny], x[xoff[bj]:xoff[bj] + nx], indexing='ij')
    else:
        yy = y[yoff[bi]:yoff[bi] + ny, xoff[bj]:xoff[bj] + nx]
        xx = x[yoff[bi]:yoff[bi] + ny, xoff[bj]:xoff[bj] + nx]
    py, px, zeff = yy, xx, None
    if pier is not None:
        py, px, zeff = _pierce_offset(yy, xx, *pier, yoff[bi] + np.arange(ny), xoff[bj] + np.arange(nx))
    X, Y = nodes
    meta, imeta, start, items, ncol = index[:5]
    qx = np.ascontiguousarray(px, dtype=np.float64).ravel()
    qy = np.ascontiguousarray(py, dtype=np.float64).ravel()
    good = np.isfinite(qx) & np.isfinite(qy)
    cell = np.zeros(qx.size, dtype=np.int64)
    uu = np.zeros(qx.size, dtype=np.float64)
    vv = np.zeros(qx.size, dtype=np.float64)
    missing = 0
    if pier is None:
        # ERA5 and VMF3: a pixel outside the node mesh is missing (phase() has checked that the coordinates are finite)
        missing = _locate_all(qx, qy, np.ascontiguousarray(X).ravel(), np.ascontiguousarray(Y).ravel(), X.shape[1],
                              meta, imeta, start, items, ncol, cell, uu, vv)
    elif good.any():
        # IONEX: a NaN pierce point (from a NaN height, incidence or look) takes a valid location; the pixel stays NaN
        fill = np.flatnonzero(good)[0]
        missing = _locate_all(np.where(good, qx, qx[fill]), np.where(good, qy, qy[fill]), np.ascontiguousarray(X).ravel(),
                              np.ascontiguousarray(Y).ravel(), X.shape[1], meta, imeta, start, items, ncol, cell, uu, vv)
    if zeff is not None:
        zeff = np.where(good, np.ascontiguousarray(zeff, dtype=np.float64).ravel(), np.nan)
    return cell, uu, vv, missing, zeff


def _plan(cell, owner):
    """The tiles that an output chunk needs: those holding the cells its pixels fall in.

    Parameters
    ----------
    cell : numpy.ndarray - The cell of each pixel; -1 for none.
    owner : numpy.ndarray - The tile index of each cell.

    Returns
    -------
    numpy.ndarray - The sorted unique tile indices.

    Examples
    --------
    >>> _plan(np.array([0, 5, -1]), np.array([0, 0, 1, 1, 1, 1]))
    """
    valid = cell[cell >= 0]
    return [int(k) for k in np.unique(owner[valid]) if k >= 0]


def _time_groups(idx, tedges):
    """Group the times of a burst by the time chunks of a model.

    Parameters
    ----------
    idx : numpy.ndarray - The indices of the burst's times on the model's time axis.
    tedges : numpy.ndarray - The ascending first time index of each time chunk of the model.

    Returns
    -------
    tuple - (groups, used): groups[c] holds (positions, local indices) of the times in time chunk c; used[c] is that chunk.

    Examples
    --------
    >>> _time_groups(np.array([0, 2, 3]), np.array([0, 2, 4]))
    """
    tc = np.searchsorted(tedges, idx, side='right') - 1
    used = sorted(set(tc.tolist()))
    groups = tuple((np.nonzero(tc == t)[0], idx[tc == t] - tedges[t]) for t in used)
    return groups, used


def _evaluate_tile(out, pix, tabs, groups, lr, lc, uu, vv, h, z, consts):
    """Write the phases of the pixels of one chunk that lie on one tile's tables into out. A pixel whose height reads a
    table gap is NaN, with one warning per call.

    Parameters
    ----------
    out : numpy.ndarray - Output phases, (times, pixels), float32, radians.
    pix : numpy.ndarray - The flat indices of the pixels.
    tabs : list - The hydrostatic, wet and mapping tables of the tile, one array per time group.
    groups : list of tuple - The times of each time group.
    lr, lc, uu, vv : numpy.ndarray - The pixels' local cells and weights.
    h, z : numpy.ndarray - The pixels' heights, metres, and incidences, radians.
    consts : tuple - (h0, step, count, niell, k) of the model.

    Returns
    -------
    None - out is updated in place.
    """
    h0, step, count, niell, k = consts
    for g, (pos, loc) in enumerate(groups):
        hyd, wet, mapping = (np.ascontiguousarray(tabs[a][g][:, 0]) for a in range(3))
        mapping = _mapping_coefs(mapping)
        sub = np.empty((loc.size, pix.size), dtype=np.float32)
        gaps = np.zeros(1, dtype=np.int64)
        _pixel_kernel(hyd, wet, mapping, loc, lr, lc, uu, vv, h, z, h0, step, count, niell, k, gaps, sub)
        out[np.ix_(pos, pix)] = sub
        if gaps[0]:
            warnings.warn(f'{int(gaps[0])} pixel-dates lie deeper below the model surface than the atmosphere tables reach '
                          f'(the law below the surface ends there): their phase is NaN')


def _phase_block(ele, inc, look, tabs, *, y, x, yoff, xoff, chunk, groups, nodes, index, name, tiles, origins, consts, pier=None):
    """One output chunk (time, y, x) of an opened model, float32 radians.

    Each pixel is evaluated on the 4 nodes of its cell (of its pierce point with pier), which lie in one tile; a pixel
    whose cell has no tile, or whose height or incidence is NaN, is NaN.

    Parameters
    ----------
    ele : numpy.ndarray - The ellipsoidal heights of the chunk, metres.
    inc : numpy.ndarray - The incidences of the chunk, radians.
    look : numpy.ndarray or None - The look azimuths of the chunk, degrees (with pier).
    tabs : list - The tables of the chunk's tiles.
    y, x : numpy.ndarray - The pixel axes in the output CRS.
    yoff, xoff : sequence of int - The chunk offsets of the axes.
    chunk : tuple of int - The chunk index.
    groups : list of tuple - The times of each time group.
    nodes, index : tuple - The lattice nodes and the cell index.
    name : str - The model name, for the messages.
    tiles : numpy.ndarray - The tiles of the chunk.
    origins : list of tuple - The first node (y0, x0) of each tile.
    consts : tuple - (h0, step, count, niell, k) of the model.
    pier : None or tuple - None, or (shell, control, geocentric): the IONEX pierce point with the shell R + H (metres), the
    control mesh of the output CRS (_control, None for a geographic CRS) and the zenith flag (_pierce_offset).

    Returns
    -------
    numpy.ndarray - float32 phases, (time, y, x).

    Raises
    ------
    ValueError: For a pixel outside the node mesh, an incidence outside [0, 90) degrees, or a height outside the tables.
    """
    ny, nx = np.shape(ele)
    pierce = None if pier is None else (ele, inc, look, pier[0], pier[1], pier[2])
    cell, uu, vv, missing, zeff = _chunk_cells(y, x, ny, nx, yoff, xoff, chunk, nodes, index, pierce)
    if missing:
        raise ValueError(f'ERROR: {name} phase(): {missing} pixel(s) of the chunk lie outside the model nodes '
                         f'(ERA5 and VMF3 reprojected to the output crs, IONEX in latitude and longitude)')
    ncol = index[4]
    h = np.ascontiguousarray(np.asarray(ele, dtype=np.float64)).ravel()
    z = np.ascontiguousarray(np.asarray(inc, dtype=np.float64) if zeff is None else zeff).ravel()
    lj = cell // ncol
    li = cell - lj * ncol
    own = index[5][cell]
    ntime = sum(pos.size for pos, _ in groups)
    out = np.empty((ntime, ny * nx), dtype=np.float32)
    for k, tile in enumerate(tiles):
        pix = np.flatnonzero(own == tile)
        if pix.size:
            _evaluate_tile(out, pix, tabs[k], groups, np.ascontiguousarray(lj[pix] - origins[k][0]),
                           np.ascontiguousarray(li[pix] - origins[k][1]), uu[pix], vv[pix], h[pix], z[pix], consts)
    pix = np.flatnonzero(own < 0)
    if pix.size:
        count = consts[2]
        nan = [[np.full((ntime, 1, 2, 2, count), np.nan)], [np.full((ntime, 1, 2, 2, count), np.nan)],
               [np.full((ntime, 1, 2, 2, 2, 3), np.nan)]]
        zero = np.zeros(pix.size, dtype=np.int64)
        _evaluate_tile(out, pix, nan, ((np.arange(ntime), np.arange(ntime)),), zero, zero, uu[pix], vv[pix], h[pix], z[pix],
                       consts)
    return out.reshape(ntime, ny, nx)


def _tile_chunks(a):
    """The terms of one term grid as a dask array with one chunk per tile.

    Parameters
    ----------
    a : dask.array.Array - The terms, (time, tile, y, x, ...).

    Returns
    -------
    dask.array.Array - The same array with one chunk per tile; time chunks are kept and the other axes are one chunk each.

    Examples
    --------
    >>> _tile_chunks(da.zeros((2, 3, 5, 5, 4, 1, 9), chunks=(1, 3, 5, 5, 4, 1, 9))).chunks
    """
    import dask.array as da
    a = da.asarray(a)
    return a.rechunk({0: a.chunks[0], 1: 1, **{d: -1 for d in range(2, a.ndim)}})


def _edges(sizes):
    """The first node of each block of nodes, and the node count at the end.

    Parameters
    ----------
    sizes : sequence of int - The node count of each block.

    Returns
    -------
    numpy.ndarray - int64 start index of each block, then the total count.
    """
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


def _first_key(keys):
    """The first key of a nested list of dask keys: the key of the chunk at index 0 of every axis.

    Parameters
    ----------
    keys : list - Nested lists of dask keys.

    Returns
    -------
    object - The first key.

    Examples
    --------
    >>> _first_key([[('x', 0, 0)]])
    """
    while isinstance(keys, list):
        keys = keys[0]
    return keys


def _node_grid(terms):
    """The lattice axes lat and lon of a model's terms, and the first node (y0, x0) of each tile.

    Parameters
    ----------
    terms : xarray.Dataset - The model's terms.

    Returns
    -------
    tuple - (lat, lon, y0, x0): float64 degrees and int64 node indices.
    """
    lat = np.asarray(terms['lat'].values, dtype=np.float64)
    lon = np.asarray(terms['lon'].values, dtype=np.float64)
    y0 = np.asarray(terms['tile y0'].values, dtype=np.int64)
    x0 = np.asarray(terms['tile x0'].values, dtype=np.int64)
    return lat, lon, y0, x0


def _consts(terms):
    """The constants of a model's tables, read from the attributes of its terms.

    Parameters
    ----------
    terms : xarray.Dataset - The model's terms.

    Returns
    -------
    tuple - (h0, step, count, niell, k): the first height and the height step in metres, the entries per table, the
    height term (v, a, b, c), and the shell factor (1 for the mapping forms).
    """
    attrs = terms.attrs
    return (float(attrs['h0']), float(attrs['step']), int(attrs['count']),
            np.asarray(attrs['niell'], dtype=np.float64), float(attrs['shell_k']))


class OpenedModel:
    """An atmosphere model of a folder, opened for a stack by AtmosphereModel.open.

    Attributes
    ----------
    name : str - 'ERA5', 'VMF3' or 'IONEX'.
    terms : xarray.Dataset - The lazy tables of the time union of the bursts, one dask chunk per tile: hydro and wet
    (time, tile, ty, tx, height) in phase units, and mapping (time, tile, ty, tx, component, coef).
    times : pandas.DatetimeIndex - The times of the terms, UTC.
    crs : int, str or rasterio CRS - The output CRS.
    x, y : numpy.ndarray - The nodes, shape (ny, nx): in the output CRS (ERA5, VMF3), in longitude and latitude (IONEX).
    """

    def __init__(self, name, terms, crs):
        self.name = name
        self.terms = terms
        self.times = pd.DatetimeIndex(terms['time'].values)
        self.crs = crs
        lat, lon, y0, x0 = _node_grid(terms)
        LON, LAT = np.meshgrid(lon, lat)
        # IONEX: the pierce point and the TEC are in latitude and longitude (the published single-layer model)
        self.x, self.y = (LON, LAT) if name == 'IONEX' else _nodes_xy(LON, LAT, crs)
        # the tables stay lazy: each tile is one dask chunk, computed only when a phase chunk needs it
        self._arrays = [_tile_chunks(terms[v].data) for v in ('hydro', 'wet', 'mapping')]
        ty, tx = self._arrays[0].shape[2], self._arrays[0].shape[3]
        if (y0 + ty > lat.size).any() or (x0 + tx > lon.size).any():
            raise ValueError(f'ERROR: {name} the terms do not cover the model nodes')
        self._y0, self._x0 = y0, x0
        self._index = _cell_table(self.x, self.y, _cell_owner(y0, x0, ty, tx, lat.size, lon.size))
        self._tedges = _edges(self._arrays[0].chunks[0])
        self._keys = [a.__dask_keys__() for a in self._arrays]
        self._tiles_done = {}
        self._consts = _consts(terms)

    def _origins(self, tiles):
        """The first node (y0, x0) of each tile of a plan.

        Parameters
        ----------
        tiles : list of int - The tile indices of the plan.

        Returns
        -------
        list of tuple - (y0, x0) of each tile.
        """
        return [(int(self._y0[k]), int(self._x0[k])) for k in tiles]

    def _tiles(self, tiles):
        """The tables of the tiles as float64 numpy arrays, in tile order; each tile is computed once.

        Parameters
        ----------
        tiles : list of int - The tile indices.

        Returns
        -------
        list - The tables of each tile.
        """
        import dask
        todo = [k for k in tiles if k not in self._tiles_done]
        if todo:
            arrays = [a[:, k:k + 1] for k in todo for a in self._arrays]
            values = dask.compute(*arrays, scheduler='synchronous')
            for q, k in enumerate(todo):
                self._tiles_done[k] = [[np.asarray(values[3 * q + j], dtype=np.float64)] for j in range(3)]
        return [self._tiles_done[k] for k in tiles]

    def phase(self, times, y, x, ele, inc, look=None, geocentric=True):
        """The phase in radians (float32, shape (time, y, x)) of the burst times on one output chunk or a whole grid.

        A dask ele, inc or look gives a lazy result with the chunks of ele; each pixel is bilinear over the 4
        reprojected nodes around it (IONEX: over the 4 latitude-longitude nodes around its pierce point).
        IONEX evaluates each pixel at its geodesic pierce point along the look azimuth (TN-200 steps 2 and 4; exact in a
        projected CRS) and needs look; ERA5 and VMF3 ignore it.

        Parameters
        ----------
        times : list of str or pandas.Timestamp - The acquisition times of one burst, UTC.
        y, x : numpy.ndarray - The pixel axes in the output CRS, 1-D or 2-D of one shape (numpy only).
        ele : numpy.ndarray or dask.array.Array - The ellipsoidal heights of the pixels, metres.
        inc : numpy.ndarray or dask.array.Array - The incidences of the pixels, radians.
        look : numpy.ndarray or dask.array.Array or None - The look azimuths of the pixels, degrees clockwise from
        north, satellite to target; the shape of ele.
        geocentric : bool - IONEX only. True (default): the mapping at the geocentric zenith of the exact ray at the pierce
        point (ETAD ATBD 2.7 Eq. 45-46). False: the mapping at the ellipsoid-normal zenith of the pixel (FRM4SAR TN-200
        Eq. 22), for verifications against TN-200. ERA5 and VMF3 ignore it.

        Returns
        -------
        numpy.ndarray or dask.array.Array - The phase, radians, (time, y, x). A pixel deeper below the surface than the
        ERA5 tables reach is NaN, with one warning per call.

        Raises
        ------
        ValueError: For a pixel with a NaN coordinate or outside the node mesh, outside the height range of the
        tables, or at an
        incidence outside [0, 90) degrees; for IONEX without look.
        """
        import dask.array as da
        import functools
        from dask.base import tokenize
        from dask.highlevelgraph import HighLevelGraph
        from . import utils_tropo
        stamps = utils_tropo._times(times)
        idx = self.times.get_indexer(pd.DatetimeIndex(stamps))
        if (idx < 0).any():
            raise ValueError(f'ERROR: {self.name} phase(): a time of the burst is not in the opened model')
        pierce = self.name == 'IONEX'
        if pierce and look is None:
            raise ValueError('ERROR: IONEX phase() needs the look azimuth (look=): the pierce point lies along it')
        shell = self.terms.attrs.get('shell') if pierce else None
        if pierce and shell is None:
            raise ValueError('ERROR: IONEX terms have no shell height R + H: repeat the IONEX open')
        if not pierce:
            look = None
        consts = self._consts[:4] + (1.0,) if pierce else self._consts
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if y.ndim == 1 and x.ndim == 1:
            shape = (y.size, x.size)
        elif y.ndim == 2 and y.shape == x.shape:
            shape = y.shape
        else:
            raise ValueError(f'ERROR: {self.name} phase(): y and x are a grid of 1-D axes or two arrays of one shape')
        # a NaN coordinate is missing on every model, before the control mesh (_control) could spread it to finite pixels
        yy, xx = np.broadcast_arrays(y[:, None] if y.ndim == 1 else y, x[None, :] if x.ndim == 1 else x)
        missing = int(np.count_nonzero(~np.isfinite(yy) | ~np.isfinite(xx)))
        if missing:
            raise ValueError(f'ERROR: {self.name} phase(): {missing} pixel(s) of the chunk lie outside the model nodes '
                             f'(ERA5 and VMF3 reprojected to the output crs, IONEX in latitude and longitude)')
        nodes = (self.x, self.y)
        pier = (float(shell), _control(y, x, self.crs), bool(geocentric)) if pierce else None
        zero = np.zeros(1, dtype=np.int64)
        if isinstance(ele, da.Array) or isinstance(inc, da.Array) or isinstance(look, da.Array):
            if y.ndim != 1:
                raise ValueError(f'ERROR: {self.name} phase(): the lazy phase takes the 1-D axes y and x')
            ele = da.asarray(ele)
            inc = da.asarray(inc)
            if ele.shape != shape or inc.shape != shape:
                raise ValueError(f'ERROR: {self.name} phase(): ele and inc must have the shape of the pixels')
            chunks = ele.chunks
            inc = inc.rechunk(chunks)
            if look is not None:
                look = da.asarray(look)
                if look.shape != shape:
                    raise ValueError(f'ERROR: {self.name} phase(): look must have the shape of the pixels')
                look = look.rechunk(chunks)
            yoff = np.concatenate(([0], np.cumsum(chunks[0])[:-1])).astype(np.int64)
            xoff = np.concatenate(([0], np.cumsum(chunks[1])[:-1])).astype(np.int64)
            name = 'phase-' + tokenize(self.name, ele, inc, look, idx, y, x, bool(geocentric),
                                       *[a.name for a in self._arrays])
            ekeys = ele.__dask_keys__()
            ikeys = inc.__dask_keys__()
            lkeys = None if look is None else look.__dask_keys__()
            groups, used = _time_groups(idx, self._tedges)
            layer = {}
            for bi in range(len(chunks[0])):
                for bj in range(len(chunks[1])):
                    if pierce:
                        tiles = list(range(self._arrays[0].shape[1]))     # IONEX is one tile of the whole lattice
                    else:
                        cell = _chunk_cells(y, x, chunks[0][bi], chunks[1][bj], yoff, xoff, (bi, bj), nodes, self._index)[0]
                        tiles = _plan(cell, self._index[5])
                    keys = [[[_first_key(self._keys[a][tc][k]) for tc in used] for a in range(3)] for k in tiles]
                    layer[(name, 0, bi, bj)] = (
                        functools.partial(_phase_block, y=y, x=x, yoff=yoff, xoff=xoff, chunk=(bi, bj), groups=groups,
                                          nodes=nodes, index=self._index, name=self.name, tiles=tiles,
                                          origins=self._origins(tiles), consts=consts, pier=pier),
                        ekeys[bi][bj], ikeys[bi][bj], None if lkeys is None else lkeys[bi][bj], keys)
            deps = [ele, inc] + ([] if look is None else [look]) + self._arrays
            graph = HighLevelGraph.from_collections(name, layer, dependencies=deps)
            return da.Array(graph, name, chunks=((len(idx),),) + chunks, dtype=np.float32,
                            meta=np.empty((0, 0, 0), dtype=np.float32))
        ele = np.asarray(ele, dtype=np.float64)
        inc = np.asarray(inc, dtype=np.float64)
        if ele.shape != shape or inc.shape != shape:
            raise ValueError(f'ERROR: {self.name} phase(): ele and inc must have the shape of the pixels')
        if look is not None:
            look = np.asarray(look, dtype=np.float64)
            if look.shape != shape:
                raise ValueError(f'ERROR: {self.name} phase(): look must have the shape of the pixels')
        if pierce:
            tiles = list(range(self._arrays[0].shape[1]))
        else:
            cell = _chunk_cells(y, x, shape[0], shape[1], zero, zero, (0, 0), nodes, self._index)[0]
            tiles = _plan(cell, self._index[5])
        tabs = self._tiles(tiles)
        groups = ((np.arange(idx.size), idx),)
        return _phase_block(ele, inc, look, tabs, y=y, x=x, yoff=zero, xoff=zero, chunk=(0, 0), groups=groups, nodes=nodes,
                            index=self._index, name=self.name, tiles=tiles, origins=self._origins(tiles), consts=consts,
                            pier=pier)


class AtmosphereModel:
    """Mixin of the atmosphere model classes ERA5, VMF3 and IONEX; open() opens a model folder once per stack."""

    def open(self, folder, times, crs, wavelength, factor=1.0):
        """Open the model of a folder once for a stack: lazy terms over the time union of the bursts, and the nodes (in the
        output CRS; IONEX in latitude and longitude).

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields (the folder that holds its .vrt index).
        times : list of str or pandas.Timestamp - The acquisition times of all bursts of the stack, UTC (their union).
        crs : int, str, rasterio CRS or None - The output CRS of the stack; None for lon and lat.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor (IONEX only), dimensionless.

        Returns
        -------
        OpenedModel
        """
        terms = self._phase(folder, times, wavelength, factor)
        return OpenedModel(type(self).__name__, terms, crs)

    def _phase(self, folder, times, wavelength, factor):
        """The terms of the model on its native nodes, in phase units.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor (IONEX only), dimensionless.
        """
        raise NotImplementedError('ERROR: the model class names its terms in _phase')


def model_class(folder):
    """The atmosphere model class of a folder (IONEX, ERA5 or VMF3), from its index file.

    Parameters
    ----------
    folder : str - The folder of the downloaded model fields.

    Returns
    -------
    type - ERA5, VMF3 or IONEX.

    Raises
    ------
    ValueError: When the folder holds no atmosphere index, or IONEX with a tropospheric index.
    """
    from . import utils_epochs
    from . import utils_tropo
    from .ERA5 import ERA5
    from .IONEX import IONEX
    from .VMF3 import VMF3
    names = utils_epochs.sources(folder)
    tropo = [n for n in names if n != 'IONEX']
    if 'IONEX' in names:
        if tropo:
            raise ValueError('ERROR: ' + str(folder) + ' holds IONEX and a tropospheric index: ' + ', '.join(names))
        return IONEX
    name = utils_tropo.model(folder)
    return ERA5 if name == 'ERA5' else VMF3
