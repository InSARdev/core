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
# the IONEX control mesh: nodes at the multiples of CONTROL_M metres in the output CRS, from zero (see _control);
# CONTROL_M chosen by measurement (GMH U4)
CONTROL_M = 800.0


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


@njit(nogil=True, cache=True, error_model='numpy', inline='always')
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


@njit(nogil=True, cache=True, error_model='numpy', inline='always')
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


@njit(nogil=True, cache=True, error_model='numpy', inline='always')
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


@njit(nogil=True, cache=True, inline='always')
def _locate_one(x, y, X, Y, nxn, meta, imeta, start, items, ncol):
    """The cell of one point among the reprojected nodes: the search of the kernel (the former _locate_all, removed by GAZ A2).

    Parameters
    ----------
    x, y : float - The point, in the CRS of the nodes.
    X, Y : numpy.ndarray - The node coordinates, flat (row-major, nxn nodes per row).
    nxn : int - The number of nodes per row.
    meta, imeta, start, items : numpy.ndarray - The cell index (see _cell_table): the box origin and sizes, the boxes and
        the cells of each box.
    ncol : int - The number of cells per row.

    Returns
    -------
    tuple - (cell, u, v): the cell, -1 when no cell holds the point, and the bilinear weights of the point in the cell.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _locate_one
    >>> X = np.array([0.0, 1.0, 0.0, 1.0]); Y = np.array([0.0, 0.0, 1.0, 1.0])
    >>> _ = _locate_one(0.25, 0.5, X, Y, 2, np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), np.array([1, 1]),
    ...                 np.array([0, 1]), np.array([0]), 1)
    """
    x0 = meta[0]
    y0 = meta[1]
    xmax = meta[2]
    ymax = meta[3]
    bdx = meta[4]
    bdy = meta[5]
    nbx = imeta[0]
    nby = imeta[1]
    if not (x >= x0 and x <= xmax and y >= y0 and y <= ymax):
        return -1, 0.0, 0.0
    bx = min(int((x - x0) / bdx), nbx - 1)
    by = min(int((y - y0) / bdy), nby - 1)
    b = by * nbx + bx
    cell = -1
    cu = 0.0
    cv = 0.0
    best = 0.0
    for q in range(start[b], start[b + 1]):
        c = items[q]
        j = c // ncol
        i = c - j * ncol
        a = j * nxn + i
        found, u, v = _inverse(x, y, X[a], Y[a], X[a + 1], Y[a + 1], X[a + nxn + 1], Y[a + nxn + 1], X[a + nxn], Y[a + nxn])
        if found:
            ex = X[a + 1] - X[a]
            ey = Y[a + 1] - Y[a]
            fx = X[a + nxn + 1] - X[a]
            fy = Y[a + nxn + 1] - Y[a]
            gx = X[a + nxn] - X[a]
            gy = Y[a + nxn] - Y[a]
            s = abs(ex * fy - ey * fx) + abs(fx * gy - fy * gx)
            if cell < 0 or s < best:
                cell = c
                cu = u
                cv = v
                best = s
    return cell, cu, cv


@njit(nogil=True, cache=True, error_model='numpy', inline='always')
def _pierce_one(qy, qx, h, z, look, shell, geocentric, ctrl, ky, kx, S0, S1, lat_c, lon_c):
    """The IONEX pierce point of one pixel and its effective incidence: the arithmetic of the sine law (TN-200 step 2).

    Parameters
    ----------
    qy, qx : float - The pixel latitude and longitude, degrees; with ctrl, the pixel's coordinates in the output CRS.
    h : float - The ellipsoidal height, metres.
    z : float - The incidence, radians.
    look : float - The look azimuth, degrees clockwise from north (satellite to target).
    shell : float - R + H, metres.
    geocentric : bool - True: the mapping at the geocentric zenith of the exact ray; False: at the ellipsoid-normal zenith.
    ctrl : bool - True: the pierce point's start is the control mesh (bilinear in the cell, the longitude unwrapped per cell).
    ky, kx, S0, S1 : int, int, float, float - The first node row and column of the mesh and its spacings (metres).
    lat_c, lon_c : numpy.ndarray - The control mesh latitudes and longitudes, degrees, (nodes y, nodes x).

    Returns
    -------
    tuple of float - (pierce latitude, pierce longitude, effective incidence), degrees, degrees and radians.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _pierce_one
    >>> _ = _pierce_one(47.0, 13.0, 0.0, 0.6, 77.0, 6821e3, True, False, 0, 0, 1.0, 1.0, np.zeros((1, 1)), np.zeros((1, 1)))
    """
    rad = math.pi / 180.0
    deg = 180.0 / math.pi
    c1 = 1.0 - _E2_WGS
    bb = _A_WGS * math.sqrt(1.0 - _E2_WGS)
    k3 = _E2_WGS / (1.0 - _E2_WGS) * bb
    k4 = _E2_WGS * _A_WGS
    if ctrl:
        ty_ = qy / S0
        jy = math.floor(ty_)
        iy = np.int64(jy - ky)
        wy = ty_ - jy
        tx_ = qx / S1
        jx = math.floor(tx_)
        ix = np.int64(jx - kx)
        wx = tx_ - jx
        la = lat_c[iy, ix] * (1.0 - wx) + lat_c[iy, ix + 1] * wx
        lb = lat_c[iy + 1, ix] * (1.0 - wx) + lat_c[iy + 1, ix + 1] * wx
        lat = la * (1.0 - wy) + lb * wy
        cc = lon_c[iy, ix]
        top = ((lon_c[iy, ix + 1] - cc + 180.0) % 360.0 - 180.0) * wx
        bot = ((lon_c[iy + 1, ix] - cc + 180.0) % 360.0 - 180.0) * (1.0 - wx)
        bot += ((lon_c[iy + 1, ix + 1] - cc + 180.0) % 360.0 - 180.0) * wx
        top = top * (1.0 - wy) + bot * wy
        lon = (cc + top + 180.0) % 360.0 - 180.0
    else:
        lat = qy
        lon = qx
    phi = lat * rad
    sp = math.sin(phi)
    cp = math.cos(phi + 0.0)   # + 0.0: an operand apart from sin's, so LLVM does not fuse them into sincos (1 ulp apart)
    big_n = _A_WGS / math.sqrt(1.0 - _E2_WGS * (sp * sp))
    pt = (big_n + h) * cp
    zt = (big_n * c1 + h) * sp
    R = math.hypot(pt, zt)
    sd = (sp * pt - cp * zt) / R
    cd = (cp * pt + sp * zt) / R
    a = (look + 180.0) * rad
    sz = math.sin(z)
    ue = sz * math.sin(a)
    un = sz * math.cos(a + 0.0)
    uu = math.cos(z + 0.0)
    uh = cd * un + sd * uu
    zc = math.atan2(math.hypot(ue, uh), -sd * un + cd * uu)
    az = math.atan2(ue, uh)
    zray = math.asin(R * math.sin(zc) / shell)
    beta = zc - zray
    zeff = zray if geocentric else math.asin(R * math.sin(z) / shell)
    sf = zt / R
    cf = pt / R
    s2 = sf * math.cos(beta + 0.0) + cf * math.sin(beta) * math.cos(az + 0.0)
    dl = math.atan2(math.sin(az) * math.sin(beta) * cf, math.cos(beta + 0.0) - sf * s2)
    p = shell * math.sqrt(1.0 - s2 * s2)
    zz = shell * s2
    th = math.atan2(zz * _A_WGS, p * bb)
    lat_g = math.atan2(zz + k3 * math.pow(math.sin(th), 3.0), p - k4 * math.pow(math.cos(th + 0.0), 3.0))
    lon = lon + dl * deg
    if lon < -180.0 or lon > 180.0:
        lon = (lon + 180.0) % 360.0 - 180.0
    return lat_g * deg, lon, zeff


@njit(nogil=True, cache=True, error_model='numpy', inline='always')
def _pixel_phase(hyd, wet, mapping, r0, c0, u, v, h, z, h0, step, count, niell, k, out):
    """The phase of one pixel at every date of a block (option (a)): its 4 nodes, each evaluated at the pixel's height
    (linear in the table entries) and at the pixel's incidence (the mapping), bilinear with the pixel's weights. Written to
    out, float32 radians; a pixel with a NaN height or incidence is NaN for every date, with no error and no warning.

    Parameters
    ----------
    hyd, wet : numpy.ndarray - The zenith tables of one tile slot, (date, row, col, height), in phase units.
    mapping : numpy.ndarray - The (a, b, c, K) coefficients of the slot, (date, row, col, component, 4).
    r0, c0 : int - The local lower-left node of the pixel's cell in the tables.
    u, v : float - The bilinear weights of the pixel in its cell.
    h : float - The ellipsoidal height of the pixel, metres.
    z : float - The incidence of the pixel, radians (IONEX: the effective incidence at the pierce point).
    h0, step : float - The first table height and the entry spacing, metres.
    count : int - The number of table entries.
    niell : numpy.ndarray - The height term (v, a, b, c); v is 0 for IONEX.
    k : float - The shell factor (IONEX); 1 for the mapping forms.
    out : numpy.ndarray - Output: the phase of each date, float32, 1-D.

    Returns
    -------
    tuple of int - (code, gaps): code 0 when the pixel is evaluated, 1 for an incidence outside [0, 90) degrees, 2 for a
    height outside the tables (out is NaN then); gaps counts the dates whose phase is NaN from a table gap.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _pixel_phase
    >>> hyd = np.ones((1, 2, 2, 2)); wet = np.zeros((1, 2, 2, 2)); mp = np.zeros((1, 2, 2, 2, 4)); mp[..., 3] = 1.0
    >>> out = np.zeros(1, np.float32)
    >>> _pixel_phase(hyd, wet, mp, 0, 0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 2, np.zeros(4), 1.0, out)
    (0, 0)
    >>> round(float(out[0]), 6)
    1.0
    """
    ntime = out.shape[0]
    if h != h or z != z:
        for p in range(ntime):
            out[p] = np.float32(np.nan)
        return 0, 0
    zd = z * (180.0 / math.pi)
    if not (zd >= 0.0 and zd < 90.0):
        for p in range(ntime):
            out[p] = np.float32(np.nan)
        return 1, 0
    cz = np.cos(z)
    sz = np.sin(z)
    s = np.sqrt(cz * cz + (1.0 - k * k) * sz * sz)
    inv = 1.0 / s
    w00 = (1.0 - u) * (1.0 - v)
    w10 = u * (1.0 - v)
    w11 = u * v
    w01 = (1.0 - u) * v
    if count == 1:
        for p in range(ntime):
            acc = 0.0
            if w00 != 0.0:
                acc += w00 * _shell_phase(hyd, wet, p, r0, c0, inv, inv)
            if w10 != 0.0:
                acc += w10 * _shell_phase(hyd, wet, p, r0, c0 + 1, inv, inv)
            if w11 != 0.0:
                acc += w11 * _shell_phase(hyd, wet, p, r0 + 1, c0 + 1, inv, inv)
            if w01 != 0.0:
                acc += w01 * _shell_phase(hyd, wet, p, r0 + 1, c0, inv, inv)
            out[p] = np.float32(acc)
        return 0, 0
    hterm = 0.0
    if niell[0] != 0.0:
        hterm = niell[0] * h * (1.0 / cz - _form(niell[1], niell[2], niell[3], s))
    f = (h - h0) / step
    if not (f >= 0.0 and f <= count - 1):
        for p in range(ntime):
            out[p] = np.float32(np.nan)
        return 2, 0
    i = min(int(f), count - 2)
    wh = f - i
    gaps = 0
    for p in range(ntime):
        acc = 0.0
        if w00 != 0.0:
            acc += w00 * _node_phase(hyd, wet, mapping, p, r0, c0, i, wh, s, hterm)
        if w10 != 0.0:
            acc += w10 * _node_phase(hyd, wet, mapping, p, r0, c0 + 1, i, wh, s, hterm)
        if w11 != 0.0:
            acc += w11 * _node_phase(hyd, wet, mapping, p, r0 + 1, c0 + 1, i, wh, s, hterm)
        if w01 != 0.0:
            acc += w01 * _node_phase(hyd, wet, mapping, p, r0 + 1, c0, i, wh, s, hterm)
        out[p] = np.float32(acc)
        if acc != acc:
            gap = False
            if w00 != 0.0:
                gap = gap or _table_gap(hyd, wet, p, r0, c0, i, count)
            if w10 != 0.0:
                gap = gap or _table_gap(hyd, wet, p, r0, c0 + 1, i, count)
            if w11 != 0.0:
                gap = gap or _table_gap(hyd, wet, p, r0 + 1, c0 + 1, i, count)
            if w01 != 0.0:
                gap = gap or _table_gap(hyd, wet, p, r0 + 1, c0, i, count)
            if gap:
                gaps += 1
    return 0, gaps


@njit(nogil=True, cache=True, error_model='numpy')
def _block_kernel(axes1d, y1, x1, y2, x2, ele, inc, look, X, Y, nxn, meta, imeta, start, items, ncol, owner, slot_of, oy, ox,
                  HYD, WET, MAPC, h0, step, count, niell, k, pierce, shell, geocentric, ctrl, ky, kx, S0, S1, lat_c, lon_c,
                  flags, out):
    """The phase of one output block in one serial pass over its pixels: per pixel the pierce point (IONEX), the cell among
    the nodes, the tile slot, and _pixel_phase, written to out[:, r, c]. Errors are counted in flags (see _phase_block).

    Parameters
    ----------
    axes1d : bool - True: y1, x1 are the block's 1-D axes; False: y2, x2 are the 2-D coordinates.
    y1, x1, y2, x2, ele, inc, look : numpy.ndarray - The pixel coordinates, heights, incidences and looks of the block.
    X, Y, nxn, meta, imeta, start, items, ncol, owner : the node coordinates and the cell index (see _locate_one, _cell_table).
    slot_of, oy, ox : numpy.ndarray - The tile slot of each tile index, and the first node of each slot (the last entry is
        the slot of no tile, whose NaN tables give NaN).
    HYD, WET, MAPC : numpy.ndarray - The block tables (slot, date, row, col, ...) of _block_tables.
    h0, step, count, niell, k : the table constants of the model (see _pixel_phase).
    pierce, shell, geocentric, ctrl : bool, float, bool, bool - The IONEX pierce point switches, the shell R + H, the zenith
        flag, and whether the control mesh is used.
    ky, kx, S0, S1, lat_c, lon_c : the control mesh; each pixel finds its cell and the longitude offsets from its first node.
    flags : numpy.ndarray - Output: the error counts (missing, table gaps, incidence before the pierce point, first kernel error).
    out : numpy.ndarray - Output: the phases, float32, (date, row, col).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _block_kernel
    >>> X = np.array([0.0, 1.0, 0.0, 1.0]); Y = np.array([0.0, 0.0, 1.0, 1.0])
    >>> meta = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); imeta = np.array([1, 1]); start = np.array([0, 1]); items = np.array([0])
    >>> HYD = np.ones((2, 1, 2, 2, 1)); WET = np.zeros((2, 1, 2, 2, 1)); MAPC = np.zeros((2, 1, 2, 2, 2, 4))
    >>> HYD[1] = np.nan; WET[1] = np.nan; MAPC[1] = np.nan
    >>> out = np.zeros((1, 1, 1), np.float32); flags = np.zeros(4, np.int64); z2 = np.zeros((1, 1))
    >>> _block_kernel(True, np.array([0.5]), np.array([0.5]), z2, z2, np.zeros((1, 1)), np.full((1, 1), 0.5), z2, X, Y, 2, meta,
    ...               imeta, start, items, 1, np.array([0]), np.array([0, -1]), np.zeros(2, np.int64), np.zeros(2, np.int64),
    ...               HYD, WET, MAPC, 0.0, 1.0, 1, np.zeros(4), 1.0, False, 0.0, True, False, 0, 0, 1.0, 1.0,
    ...               z2, z2, flags, out)
    >>> round(float(out[0, 0, 0]), 4)
    1.1395
    """
    ny, nx = ele.shape
    nslot = HYD.shape[0] - 1
    pout = out.shape[0]
    nan32 = np.float32(np.nan)
    for r in range(ny):
        for c in range(nx):
            hq = np.float64(ele[r, c])
            zq = np.float64(inc[r, c])
            if axes1d:
                qy = y1[r]
                qx = x1[c]
            else:
                qy = y2[r, c]
                qx = x2[r, c]
            if pierce:
                zd0 = zq * (180.0 / math.pi)
                if zd0 == zd0 and not (zd0 >= 0.0 and zd0 < 90.0):
                    flags[_INC_PRE] = 1
                plat, plon, zeff = _pierce_one(qy, qx, hq, zq, np.float64(look[r, c]), shell, geocentric, ctrl,
                                               ky, kx, S0, S1, lat_c, lon_c)
                if not (math.isfinite(plon) and math.isfinite(plat)):
                    for p in range(pout):
                        out[p, r, c] = nan32
                    continue
                px = plon
                py = plat
                zq = zeff
            else:
                px = qx
                py = qy
            cell, u, v = _locate_one(px, py, X, Y, nxn, meta, imeta, start, items, ncol)
            if cell < 0:
                flags[_MISSING] += 1
                for p in range(pout):
                    out[p, r, c] = nan32
                continue
            # the slot of the cell's tile; the last slot (NaN tables) for a cell of no tile
            s_ = nslot
            own = owner[cell]
            if own >= 0 and own < slot_of.size and slot_of[own] >= 0:
                s_ = slot_of[own]
            r0 = 0
            c0 = 0
            if s_ < nslot:
                j = cell // ncol
                i = cell - j * ncol
                r0 = j - oy[s_]
                c0 = i - ox[s_]
            code, gaps = _pixel_phase(HYD[s_], WET[s_], MAPC[s_], r0, c0, u, v, hq, zq, h0, step, count, niell, k,
                                      out[:, r, c])
            if code == 1 and flags[_KERR] == 0:
                flags[_KERR] = 1
            if code == 2 and flags[_KERR] == 0:
                flags[_KERR] = 2
            flags[_GAPS] += gaps


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
    tuple - (meta, imeta, start, items, ncol, owner, boxes, box_owner): the bucket index of the cells, the tile of each cell,
    the box (x0, x1, y0, y1) of each finite cell and the tile of each box.
    """
    ny, nx = X.shape
    ncol = nx - 1
    empty = (np.array([0.0, 0.0, -1.0, -1.0, 1.0, 1.0]), np.array([1, 1], dtype=np.int64),
             np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64), max(ncol, 1), owner,
             np.zeros((0, 4)), np.zeros(0, dtype=np.int32))
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
    boxes = np.stack([cx0, cx1, cy0, cy1], axis=1)
    return meta, imeta, start, items, ncol, owner, boxes, owner[ids]


_A_WGS = 6378137.0                                                # m, WGS84 semi-major axis
_E2_WGS = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)      # WGS84 first eccentricity squared

# flags of _block_kernel: missing pixels, table gaps, incidence of the burst before the pierce point (IONEX), first kernel error
_MISSING, _GAPS, _INC_PRE, _KERR = 0, 1, 2, 3



def _control(y, x, crs):
    """The control mesh of a projected CRS for the pierce point: the latitude and longitude at the nodes of the absolute
    lattice of CONTROL_M metres in the output CRS (the multiples of CONTROL_M from zero) that the axes need; None for no
    CRS or a geographic CRS. A pixel's mesh values depend on its own coordinates only, never on the grid.

    Parameters
    ----------
    y, x : numpy.ndarray - The pixel axes in the output CRS, 1-D or 2-D of one shape.
    crs : int, str, rasterio CRS or None - The output CRS.

    Returns
    -------
    tuple or None - (ky, kx, S, lat, lon): the first lattice indices of the node rows and columns, the spacings S =
    (Sy, Sx) in metres and the latitudes and longitudes in degrees as the transform returns them (in [-180, 180]; the
    +/-180 seam is crossed by the control offsets of _phase_block, cell by cell), shape (nodes y, nodes x).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _control
    >>> _ = _control(np.array([5.2e6, 5.21e6]), np.array([3.5e5, 3.6e5]), 32633)
    """
    from pyproj import CRS, Transformer
    if crs is None or CRS.from_user_input(crs).is_geographic:
        return None
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    S = CONTROL_M

    def nodes(a):
        k0 = int(np.floor(a.min() / S))
        return k0, int(np.floor(a.max() / S)) - k0 + 2
    ky, ny = nodes(y)
    kx, nx = nodes(x)
    Y, X = np.meshgrid((ky + np.arange(ny)) * S, (kx + np.arange(nx)) * S, indexing='ij')
    # one transform of the mesh
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(X.ravel(), Y.ravel())
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return ky, kx, (S, S), lat.reshape(X.shape), lon.reshape(X.shape)


def _extent(y, x):
    """The extent of the pixels of a block: reductions of the 1-D axes or of the 2-D arrays (no copy).

    Parameters
    ----------
    y, x : numpy.ndarray - The pixel axes in the output CRS: 1-D or 2-D of one shape.

    Returns
    -------
    tuple of float - (ymin, ymax, xmin, xmax).

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _extent
    >>> _extent(np.array([1.0, 3.0]), np.array([-2.0, 5.0]))
    (1.0, 3.0, -2.0, 5.0)
    """
    return float(np.min(y)), float(np.max(y)), float(np.min(x)), float(np.max(x))


def _plan(index, ymin, ymax, xmin, xmax):
    """The tiles that a block needs: those holding the cells whose box meets the block's extent. A superset of the tiles of
    the block's own cells; exact for the kernel, where each pixel still finds its own cell.

    Parameters
    ----------
    index : tuple - The cell index (see _cell_table): the cell boxes and the tile of each box.
    ymin, ymax, xmin, xmax : float - The extent of the block's pixels in the output CRS.

    Returns
    -------
    list of int - The sorted unique tile indices.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _cell_table, _plan
    >>> index = _cell_table(np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]),
    ...                     np.array([0], dtype=np.int32))
    >>> _plan(index, 0.2, 0.4, 0.2, 0.4)
    [0]
    """
    boxes, box_owner = index[6], index[7]
    meet = (boxes[:, 0] <= xmax) & (boxes[:, 1] >= xmin) & (boxes[:, 2] <= ymax) & (boxes[:, 3] >= ymin)
    return [int(k) for k in np.unique(box_owner[meet]) if k >= 0]


def _missing(y, x):
    """The number of pixels with a coordinate that is not finite: O(ny + nx) for 1-D axes (the rows and columns of NaN), and
    reductions for 2-D arrays (the count itself only when a coordinate is not finite).

    Parameters
    ----------
    y, x : numpy.ndarray - The pixel axes in the output CRS: 1-D (the axes of a grid) or 2-D of one shape.

    Returns
    -------
    int - The number of pixels with a coordinate that is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _missing
    >>> _missing(np.array([np.nan, 1.0]), np.array([0.0, 1.0, 2.0]))
    3
    """
    if y.size == 0 or x.size == 0:
        return 0
    if y.ndim == 1 and x.ndim == 1:
        by = int(np.count_nonzero(~np.isfinite(y)))
        bx = int(np.count_nonzero(~np.isfinite(x)))
        return by * x.size + y.size * bx - by * bx
    if all(math.isfinite(float(v)) for v in (np.min(y), np.max(y), np.min(x), np.max(x))):
        return 0
    return int(np.count_nonzero(~np.isfinite(y) | ~np.isfinite(x)))


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


class _Lattice:
    """The cell index of a model's lattice as one graph value: the phase tasks of a stack share it (dask sends it once per worker)."""

    def __init__(self, index):
        self.index = index


def _overlap(edges, lo, hi):
    """The blocks of one axis that hold the pixels [lo, hi), each with the slice of its pixels inside the block.

    Parameters
    ----------
    edges : numpy.ndarray - The first pixel of each block and the pixel count at the end (see _edges).
    lo, hi : int - The pixels of the output block, [lo, hi).

    Returns
    -------
    list of tuple - (block index, slice) of each block that holds some of the pixels.
    """
    out = []
    for k in range(len(edges) - 1):
        a, b = max(lo, int(edges[k])), min(hi, int(edges[k + 1]))
        if a < b:
            out.append((k, slice(a - int(edges[k]), b - int(edges[k]))))
    return out


def _output_chunks(chunks, ndate, shape):
    """The chunks (date, y, x) of a lazy phase as tuples of int, checked to cover the dates and the pixel axes.

    Parameters
    ----------
    chunks : tuple of three sequences of int - The chunk sizes of the dates, y and x.
    ndate : int - The number of dates of the burst.
    shape : tuple of int - The pixel shape (y, x).

    Returns
    -------
    tuple - (date, y, x) chunk sizes.
    """
    out = tuple(tuple(int(n) for n in c) for c in chunks)
    if len(out) != 3:
        raise ValueError('ERROR: phase(): chunks is (date, y, x) chunk sizes')
    for got, want, what in zip(out, (ndate,) + tuple(shape), ('date', 'y', 'x')):
        if not got or min(got) < 1 or sum(got) != want:
            raise ValueError(f'ERROR: phase(): the {what} chunks {got} do not cover the {want} {what} of the phase')
    return out


def _block_tables(tabs, groups, ntime, count):
    """The tables of the block's tiles, per tile slot and date of the block (only the block's dates), and one more slot of
    NaN tables for the pixels of no tile.

    Parameters
    ----------
    tabs : list - The hydrostatic, wet and mapping tables of each tile (see OpenedModel.phase).
    groups : list of tuple - The dates of the block by model time chunk: (positions, local indices).
    ntime : int - The number of dates of the block.
    count : int - The number of table entries.

    Returns
    -------
    tuple of numpy.ndarray - (HYD, WET, MAPC): (slot, date, row, col, height), (slot, date, row, col, height) and
    (slot, date, row, col, component, 4) with the constant K.

    Examples
    --------
    >>> import numpy as np
    >>> from insardev_toolkit.utils_atmosphere import _block_tables
    >>> HYD, WET, MAPC = _block_tables([], ((np.arange(1), np.arange(1)),), 1, 1)
    >>> HYD.shape
    (1, 1, 2, 2, 1)
    """
    nslot = len(tabs)
    if nslot:
        t0 = tabs[0][0][0]
        ty, tx = t0.shape[2], t0.shape[3]
    else:
        ty = tx = 2
    HYD = np.full((nslot + 1, ntime, ty, tx, count), np.nan)
    WET = np.full((nslot + 1, ntime, ty, tx, count), np.nan)
    MAPC = np.full((nslot + 1, ntime, ty, tx, 2, 4), np.nan)
    for s, tab in enumerate(tabs):
        for g, (pos, loc) in enumerate(groups):
            HYD[s, pos] = np.asarray(tab[0][g], np.float64)[loc, 0]
            WET[s, pos] = np.asarray(tab[1][g], np.float64)[loc, 0]
            MAPC[s, pos] = _mapping_coefs(np.asarray(tab[2][g], np.float64)[loc, 0])
    return HYD, WET, MAPC


def _phase_task(lattice, ele, inc, look, tabs, *, cut, block, **kw):
    """One output block of the lazy phase: the block called on the VIEWS of the input blocks it overlaps, each writing its
    part of one output allocated once (out=); no copy of the inputs. A block that returns another array than its out is
    copied into the view.

    Parameters
    ----------
    lattice : _Lattice - The cell index of the model.
    ele, inc, look : list of list of numpy.ndarray or None - The overlapping blocks of the heights, incidences and looks.
    tabs : list - The model tables of the block's tiles (see OpenedModel.phase).
    cut : tuple of two lists of slice - The rows and the columns of the block in each overlapping input block.
    block : callable - _phase_block, bound when the graph is built (as before the chunked lazy phase).
    kw : dict - The other arguments of _phase_block (y and x: the block's 1-D axes).

    Returns
    -------
    numpy.ndarray - float32 phases, (time, y, x).
    """
    rsl, csl = cut
    y, x = kw.pop('y'), kw.pop('x')
    hs = [s.stop - s.start for s in rsl]
    ws = [s.stop - s.start for s in csl]
    ntime = sum(pos.size for pos, _ in kw['groups'])
    out = np.empty((ntime, sum(hs), sum(ws)), dtype=np.float32)
    r0 = 0
    for i, rs in enumerate(rsl):
        c0 = 0
        for j, cs in enumerate(csl):
            part = out[:, r0:r0 + hs[i], c0:c0 + ws[j]]
            res = block(ele[i][j][rs, cs], inc[i][j][rs, cs], None if look is None else look[i][j][rs, cs], tabs,
                        y=y[r0:r0 + hs[i]], x=x[c0:c0 + ws[j]], index=lattice.index, out=part, **kw)
            if res is not part:
                part[...] = res
            c0 += ws[j]
        r0 += hs[i]
    return out


def _phase_block(ele, inc, look, tabs, *, y, x, groups, nodes, index, name, tiles, origins, consts, pier=None, out=None):
    """One output block (time, y, x) of an opened model, float32 radians, in one numba pass per pixel.

    Each pixel is evaluated on the 4 nodes of its cell (of its pierce point with pier), which lie in one tile; a pixel
    whose cell has no tile, or whose height or incidence is NaN, is NaN.

    Parameters
    ----------
    ele : numpy.ndarray - The ellipsoidal heights of the block, metres.
    inc : numpy.ndarray - The incidences of the block, radians.
    look : numpy.ndarray or None - The look azimuths of the block, degrees (with pier).
    tabs : list - The tables of the block's tiles.
    y, x : numpy.ndarray - The pixel axes of the block in the output CRS: 1-D, or 2-D of the shape of ele.
    groups : list of tuple - The dates of the block, by model time chunk: (positions, local indices).
    nodes, index : tuple - The lattice nodes and the cell index.
    name : str - The model name, for the messages.
    tiles : numpy.ndarray - The tiles of the block.
    origins : list of tuple - The first node (y0, x0) of each tile.
    consts : tuple - (h0, step, count, niell, k) of the model.
    pier : None or tuple - None, or (shell, control, geocentric): the IONEX pierce point with the shell R + H (metres), the
    control mesh of the output CRS (_control, None for a geographic CRS) and the zenith flag (_pierce_one).
    out : numpy.ndarray or None - The output, (time, y, x) float32, to fill (a view of a larger output); None allocates it.

    Returns
    -------
    numpy.ndarray - float32 phases, (time, y, x): out.

    Raises
    ------
    ValueError: For a pixel outside the node mesh, an incidence outside [0, 90) degrees, or a height outside the tables.
    """
    ele = np.asarray(ele)
    inc = np.asarray(inc)
    ny, nx = ele.shape
    ntime = sum(pos.size for pos, _ in groups)
    h0, step, count, niell, k = consts
    X, Y = nodes
    meta, imeta, start, items, ncol, owner = index[:6]
    tiles = [int(t) for t in tiles]
    slot_of = np.full(max([int(owner.max()) if owner.size else -1] + tiles) + 2, -1, np.int64)
    for s, t in enumerate(tiles):
        slot_of[t] = s
    oy = np.array([o[0] for o in origins] + [0], np.int64)
    ox = np.array([o[1] for o in origins] + [0], np.int64)
    HYD, WET, MAPC = _block_tables(tabs, groups, ntime, count)
    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    axes1d = y.ndim == 1
    y1, x1 = (y, x) if axes1d else (np.zeros(1), np.zeros(1))
    y2, x2 = (np.zeros((1, 1)), np.zeros((1, 1))) if axes1d else (y, x)
    z2 = np.zeros((1, 1))
    pierce = pier is not None
    shell, geocentric, ctrl = 0.0, True, False
    ky = kx = 0
    S0 = S1 = 1.0
    lat_c = lon_c = z2
    if pierce:
        shell, control, geocentric = float(pier[0]), pier[1], bool(pier[2])
        if look is None:
            raise ValueError('ERROR: IONEX phase() needs the look azimuth (look=): the pierce point lies along it')
        look = np.asarray(look)
        if control is not None:
            ctrl = True
            ky, kx, (S0, S1), lat_c, lon_c = control
            lat_c = np.ascontiguousarray(lat_c, np.float64)
            lon_c = np.ascontiguousarray(lon_c, np.float64)

    else:
        look = z2
    flags = np.zeros(4, np.int64)
    if out is None:
        out = np.empty((ntime, ny, nx), dtype=np.float32)
    _block_kernel(axes1d, y1, x1, y2, x2, ele, inc, look, np.ascontiguousarray(X, np.float64).ravel(),
                  np.ascontiguousarray(Y, np.float64).ravel(), X.shape[1], meta, imeta, start, items, ncol, owner, slot_of,
                  oy, ox, HYD, WET, MAPC, float(h0), float(step), int(count), np.asarray(niell, np.float64), float(k), pierce,
                  shell, geocentric, ctrl, int(ky), int(kx), float(S0), float(S1), lat_c, lon_c, flags, out)
    if flags[_INC_PRE]:
        raise ValueError('ERROR: the incidence of the burst is outside [0, 90) degrees')
    if flags[_MISSING]:
        raise ValueError(f'ERROR: {name} phase(): {int(flags[_MISSING])} pixel(s) of the chunk lie outside the model nodes '
                         f'(ERA5 and VMF3 reprojected to the output crs, IONEX in latitude and longitude)')
    if flags[_KERR] == 1:
        raise ValueError('ERROR: the incidence of the burst is outside [0, 90) degrees')
    if flags[_KERR] == 2:
        raise ValueError('ERROR: a pixel height is outside the height range of the atmosphere model tables')
    if flags[_GAPS]:
        warnings.warn(f'{int(flags[_GAPS])} pixel-dates lie deeper below the model surface than the atmosphere tables reach '
                      f'(the law below the surface ends there): their phase is NaN')
    return out


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
    import dask
    # the terms may be lazy: computed on the synchronous scheduler (no Client, no thread pool)
    lat, lon, y0, x0 = dask.compute(terms['lat'].data, terms['lon'].data, terms['tile y0'].data, terms['tile x0'].data,
                                    scheduler='synchronous')
    return (np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64),
            np.asarray(y0, dtype=np.int64), np.asarray(x0, dtype=np.int64))


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

    def phase(self, times, y, x, ele, inc, look=None, geocentric=True, chunks=None):
        """The phase in radians (float32, shape (time, y, x)) of the burst times on one output chunk or a whole grid.

        A dask ele, inc or look gives a lazy result: each output block is one task that evaluates only its own dates and
        pixels, reads only the model tables of its dates, and takes the ele, inc and look blocks it overlaps (cut in numpy
        inside the task). inc and look in other chunks than ele are rechunked to ele's (one WARNING per call). Each pixel is bilinear over the 4
        reprojected nodes around it (IONEX: over the 4 latitude-longitude nodes around its pierce point). IONEX evaluates
        each pixel at its geodesic pierce point along the look azimuth (TN-200 steps 2 and 4; exact in a projected CRS) and
        needs look; ERA5 and VMF3 ignore it.

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
        chunks : None or tuple of three sequences of int - The chunks (date, y, x) of a lazy result, e.g. the stack data's
        (Stack._data_chunks). None: one chunk of all dates and the chunks of ele. Lazy inputs only.

        Returns
        -------
        numpy.ndarray or dask.array.Array - The phase, radians, (time, y, x). A pixel deeper below the surface than the
        ERA5 tables reach is NaN, with one warning per call.

        Raises
        ------
        ValueError: For a pixel with a NaN coordinate or outside the node mesh, outside the height range of the
        tables, or at an incidence outside [0, 90) degrees; for IONEX without look; for chunks that do not cover the dates and
        the pixels.
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
        missing = _missing(y, x)
        if missing:
            raise ValueError(f'ERROR: {self.name} phase(): {missing} pixel(s) of the chunk lie outside the model nodes '
                             f'(ERA5 and VMF3 reprojected to the output crs, IONEX in latitude and longitude)')
        nodes = (self.x, self.y)
        pier = (float(shell), _control(y, x, self.crs), bool(geocentric)) if pierce else None
        if isinstance(ele, da.Array) or isinstance(inc, da.Array) or isinstance(look, da.Array):
            if y.ndim != 1:
                raise ValueError(f'ERROR: {self.name} phase(): the lazy phase takes the 1-D axes y and x')
            ele = da.asarray(ele)
            inc = da.asarray(inc)
            if ele.shape != shape or inc.shape != shape:
                raise ValueError(f'ERROR: {self.name} phase(): ele and inc must have the shape of the pixels')
            # the secondaries inc and look take the ele (y, x) chunks, the primary's: rechunked with one WARNING per call
            moved = []
            if inc.chunks != ele.chunks:
                inc = inc.rechunk(ele.chunks)
                moved.append('inc')
            if look is not None:
                look = da.asarray(look)
                if look.shape != shape:
                    raise ValueError(f'ERROR: {self.name} phase(): look must have the shape of the pixels')
                if look.chunks != ele.chunks:
                    look = look.rechunk(ele.chunks)
                    moved.append('look')
            if moved:
                print(f"WARNING: {self.name} phase(): {', '.join(moved)} rechunked to the ele (y, x) chunks.", flush=True)
            out = ((len(idx),),) + ele.chunks if chunks is None else _output_chunks(chunks, len(idx), shape)
            dedges, yedges, xedges = _edges(out[0]), _edges(out[1]), _edges(out[2])
            eedges = (_edges(ele.chunks[0]), _edges(ele.chunks[1]))
            ekeys = ele.__dask_keys__()
            ikeys = inc.__dask_keys__()
            lkeys = None if look is None else look.__dask_keys__()
            name = 'phase-' + tokenize(self.name, ele, inc, look, idx, y, x, bool(geocentric), out,
                                       *[a.name for a in self._arrays])
            lattice = (name, 'lattice')
            layer = {lattice: _Lattice(self._index)}
            dgroups = [_time_groups(idx[dedges[c]:dedges[c + 1]], self._tedges) for c in range(len(out[0]))]
            for bi in range(len(out[1])):
                r0, r1 = int(yedges[bi]), int(yedges[bi + 1])
                ye = _overlap(eedges[0], r0, r1)
                y_blk = y[r0:r1]
                for bj in range(len(out[2])):
                    c0, c1 = int(xedges[bj]), int(xedges[bj + 1])
                    xe = _overlap(eedges[1], c0, c1)
                    x_blk = x[c0:c1]
                    if pierce:
                        tiles = list(range(self._arrays[0].shape[1]))     # IONEX is one tile of the whole lattice
                    else:
                        tiles = _plan(self._index, *_extent(y_blk, x_blk))
                    cut = ([s for _, s in ye], [s for _, s in xe])
                    eb = [[ekeys[i][j] for j, _ in xe] for i, _ in ye]
                    ib = [[ikeys[i][j] for j, _ in xe] for i, _ in ye]
                    lb = None if lkeys is None else [[lkeys[i][j] for j, _ in xe] for i, _ in ye]
                    for dc in range(len(out[0])):
                        groups, used = dgroups[dc]
                        keys = [[[_first_key(self._keys[a][tc][k]) for tc in used] for a in range(3)] for k in tiles]
                        layer[(name, dc, bi, bj)] = (
                            functools.partial(_phase_task, cut=cut, block=_phase_block, y=y_blk, x=x_blk,
                                              groups=groups,
                                              nodes=nodes, name=self.name, tiles=tiles, origins=self._origins(tiles),
                                              consts=consts, pier=pier),
                            lattice, eb, ib, lb, keys)
            deps = [ele, inc] + ([] if look is None else [look]) + self._arrays
            graph = HighLevelGraph.from_collections(name, layer, dependencies=deps)
            return da.Array(graph, name, chunks=out, dtype=np.float32, meta=np.empty((0, 0, 0), dtype=np.float32))
        if chunks is not None:
            raise ValueError(f'ERROR: {self.name} phase(): chunks= needs a dask ele, inc or look (the lazy phase)')
        ele = np.asarray(ele)                  # the kernel reads each pixel as float64: the inputs are not copied
        inc = np.asarray(inc)
        if ele.shape != shape or inc.shape != shape:
            raise ValueError(f'ERROR: {self.name} phase(): ele and inc must have the shape of the pixels')
        if look is not None:
            look = np.asarray(look)
            if look.shape != shape:
                raise ValueError(f'ERROR: {self.name} phase(): look must have the shape of the pixels')
        if pierce:
            tiles = list(range(self._arrays[0].shape[1]))
        else:
            tiles = _plan(self._index, *_extent(y, x))
        tabs = self._tiles(tiles)
        groups = ((np.arange(idx.size), idx),)
        return _phase_block(ele, inc, look, tabs, y=y, x=x, groups=groups, nodes=nodes,
                            index=self._index, name=self.name, tiles=tiles, origins=self._origins(tiles), consts=consts,
                            pier=pier)


class AtmosphereModel:
    """Mixin of the atmosphere model classes ERA5, VMF3 and IONEX; open() opens a model folder once per stack."""

    def open(self, folder, times, crs, wavelength, factor=1.0, heights=None):
        """Open the model of a folder once for a stack: lazy terms over the time union of the bursts, and the nodes (in the
        output CRS; IONEX in latitude and longitude).

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields (the folder that holds its .vrt index).
        times : list of str or pandas.Timestamp - The acquisition times of all bursts of the stack, UTC (their union).
        crs : int, str, rasterio CRS or None - The output CRS of the stack; None for lon and lat.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor (IONEX only), dimensionless.
        heights : tuple of float or None - (lo, hi), the ellipsoidal heights (metres) of the caller's pixels: the zenith tables
            cover that range (ERA5, VMF3; IONEX ignores it). None: the envelope at the model's table step (for callers without a
            known range). Heights outside -500..9000 m raise one ERROR line.

        Returns
        -------
        OpenedModel
        """
        terms = self._phase(folder, times, wavelength, factor, heights)
        return OpenedModel(type(self).__name__, terms, crs)

    def _phase(self, folder, times, wavelength, factor, heights=None):
        """The terms of the model on its native nodes, in phase units.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor (IONEX only), dimensionless.
        heights : tuple or None - (lo, hi) ellipsoidal metres of the caller's pixels, or None (the envelope); see open.
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
