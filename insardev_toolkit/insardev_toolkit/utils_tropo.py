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
Tropospheric delays of a model folder: model() names its index, verify() checks its files, zenith_tiles() returns the
zenith delay tables per tile as lazy dask chunks.

The folder holds one index written by utils_epochs.store(): VMF3_OP, VMF3_EI or VMF3_FC (6-hourly) or ERA5 (hourly,
pressure levels). Tables are in metres over the ellipsoidal height _TABLE_H0 + k _TABLE_STEP.
"""
import datetime
from collections import namedtuple
import math
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

from . import utils_epochs
from . import utils_geoid
from .utils_files import exists

_NAME_ERA5 = 'ERA5'
# VMF3 index variables (level 'surface').
_VARIABLES = ('ah', 'aw', 'zhd', 'zwd', 'orography')
_LEVEL = 'surface'
# ERA5 index variables: four fields per pressure level, two at the surface.
_ERA5_LEVEL_VARIABLES = ('temperature', 'specific_humidity', 'specific_cloud_liquid_water_content', 'geopotential')
_ERA5_SURFACE_VARIABLES = ('surface_pressure', 'geopotential_at_surface')
# TN-200 Eq. 10 refractivity coefficients (1e-6), molar mass ratio, gas constants (J/(kg K)), standard gravity.
_K_DRY, _K_WET, _K_WET2, _K_LIQ = 77.6890, 71.2952, 375463.0, 1.45
_EPS, _RD, _RV, _G0 = 0.62198, 287.05, 461.5, 9.80665
# WGS84 normal gravity and ellipsoid constants (closed form): _GE to _M.
_GE, _KG, _E2, _A = 9.7803253359, 0.00193185265241, 0.00669437999014, 6378137.0
_F, _M = 1.0 / 298.257223563, 0.00344978600308
_NMAX = 12
# 91 (n, m) terms of the spherical harmonic model, in table order
_TERMS = [(n, m) for n in range(_NMAX + 1) for m in range(n + 1)]
# Niell height correction of the hydrostatic mapping: a, b, c (vmf3_grid.m 397-399)
_HT_COEF = (2.53e-5, 5.49e-3, 1.14e-3)
# sub-steps of the geometric heights within a layer between two levels (ln p rule, _zenith_node)
_SUB = 16


def _mapping_tables(name):
    """
    Loads the [table] blocks of data/mapping/<name> as float64 arrays.

    Parameters: name (str). Returns: dict {table name: 1-D float64 array}.
    """
    tables = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mapping', name), encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            if text.startswith('['):
                rows = tables[text[1:-1]] = []
            else:
                rows.extend(text.split())
    return {key: np.array(rows, dtype=float) for key, rows in tables.items()}

_SH_TABLES = {name: table.reshape(91, 5) for name, table in _mapping_tables('vmf3_grid.txt').items()}


_GMF_NMAX = 9
_GMF_NTERMS = (_GMF_NMAX + 1) * (_GMF_NMAX + 2) // 2
_GMF_TABLES = _mapping_tables('gmf.txt')
# the GMF constants of gmf.f: the hydrostatic b, and the wet b and c (its hydrostatic c is _gmf_nodes)
_GMF_BH, _GMF_BW, _GMF_CW = 0.0029, 0.00146, 0.04391


def _doy(time):
    """
    The day of the year (1 to 366, with the fraction of the day) of a UTC time.

    Parameters: time (str or pandas.Timestamp; a naive time is UTC). Returns: float.

    Examples
    --------
    >>> _doy('2016-05-11 12:00')
    """
    t = utils_epochs._stamp(time)
    return t.dayofyear + (t - t.normalize()).total_seconds() / 86400.0


def _legendre(lat, lon, nmax):
    """
    Normalised Legendre functions V[n, m] (with cos m lon) and W[n, m] (with sin m lon) of a node.

    Parameters: lat, lon (degrees); nmax (int). Returns: V and W, arrays (nmax + 1, nmax + 1).

    Examples
    --------
    >>> _legendre(40.0, 10.0, 9)
    """
    th = math.pi / 2 - math.radians(lat)
    lam = math.radians(lon)
    x, y, z = math.sin(th) * math.cos(lam), math.sin(th) * math.sin(lam), math.cos(th)
    V = np.zeros((nmax + 1, nmax + 1))
    W = np.zeros((nmax + 1, nmax + 1))
    V[0, 0] = 1.0
    V[1, 0] = z * V[0, 0]
    for n in range(2, nmax + 1):
        V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / n
    for m in range(1, nmax + 1):
        V[m, m] = (2 * m - 1) * (x * V[m - 1, m - 1] - y * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (x * W[m - 1, m - 1] + y * V[m - 1, m - 1])
        if m < nmax:
            V[m + 1, m] = (2 * m + 1) * z * V[m, m]
            W[m + 1, m] = (2 * m + 1) * z * W[m, m]
        for n in range(m + 2, nmax + 1):
            V[n, m] = ((2 * n - 1) * z * V[n - 1, m] - (n + m - 1) * V[n - 2, m]) / (n - m)
            W[n, m] = ((2 * n - 1) * z * W[n - 1, m] - (n + m - 1) * W[n - 2, m]) / (n - m)
    return V, W


def _sh_epochs(parts, doys):
    """
    The spherical harmonic bh, bw, ch and cw of every node on every day of the year.

    Parameters: parts (numpy.ndarray (nodes, 4, 5), from _sh_table_parts); doys (list of float, from _doy). Returns:
    numpy.ndarray (epochs, 4, nodes).

    Examples
    --------
    >>> _sh_epochs(_sh_table_parts(np.array([40.55]), np.array([10.55])), [132.5])
    """
    out = np.empty((len(doys), 4, parts.shape[0]))
    for k, doy in enumerate(doys):
        a = doy / 365.25 * 2 * math.pi
        c = parts.transpose(2, 1, 0)
        out[k] = c[0] + c[1] * math.cos(a) + c[2] * math.sin(a) + c[3] * math.cos(2 * a) + c[4] * math.sin(2 * a)
    return out


_TERM_N = np.array([n for n, m in _TERMS])
_TERM_M = np.array([m for n, m in _TERMS])
_SH_ANM = np.stack([_SH_TABLES['anm_' + p] for p in ('bh', 'bw', 'ch', 'cw')])
_SH_BNM = np.stack([_SH_TABLES['bnm_' + p] for p in ('bh', 'bw', 'ch', 'cw')])


@njit(nogil=True, cache=True, error_model='numpy')
def _sh_table_kernel(lat, lon, nmax, tn, tm, anm, bnm, out):
    """
    Writes the spherical harmonic sums of every node into out.

    Parameters: lat, lon (float64 (nodes,), degrees); nmax (int); tn, tm (int64 (terms,), degree and order); anm, bnm
    (float64 (4, terms, 5), coefficients); out (float64 (nodes, 4, 5)). Returns: None.

    Examples
    --------
    >>> _sh_table_kernel(np.array([40.55]), np.array([10.55]), _NMAX, _TERM_N, _TERM_M, _SH_ANM, _SH_BNM, np.empty((1, 4, 5)))
    """
    V = np.zeros((nmax + 1, nmax + 1))
    W = np.zeros((nmax + 1, nmax + 1))
    for k in range(lat.size):
        th = math.pi / 2 - math.radians(lat[k])
        lam = math.radians(lon[k])
        x = math.sin(th) * math.cos(lam)
        y = math.sin(th) * math.sin(lam)
        z = math.cos(th)
        V[0, 0] = 1.0
        V[1, 0] = z
        for n in range(2, nmax + 1):
            V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / n
        for m in range(1, nmax + 1):
            V[m, m] = (2 * m - 1) * (x * V[m - 1, m - 1] - y * W[m - 1, m - 1])
            W[m, m] = (2 * m - 1) * (x * W[m - 1, m - 1] + y * V[m - 1, m - 1])
            if m < nmax:
                V[m + 1, m] = (2 * m + 1) * z * V[m, m]
                W[m + 1, m] = (2 * m + 1) * z * W[m, m]
            for n in range(m + 2, nmax + 1):
                V[n, m] = ((2 * n - 1) * z * V[n - 1, m] - (n + m - 1) * V[n - 2, m]) / (n - m)
                W[n, m] = ((2 * n - 1) * z * W[n - 1, m] - (n + m - 1) * W[n - 2, m]) / (n - m)
        for q in range(4):
            for c in range(5):
                s = 0.0
                for t in range(tn.size):
                    s += anm[q, t, c] * V[tn[t], tm[t]] + bnm[q, t, c] * W[tn[t], tm[t]]
                out[k, q, c] = s


def _sh_table_parts(lat, lon):
    """
    The spherical harmonic coefficients of the nodes.

    Parameters: lat, lon (1-D float arrays, degrees). Returns: numpy.ndarray (nodes, 4, 5).

    Examples
    --------
    >>> _sh_table_parts(np.array([40.55]), np.array([10.55]))
    """
    out = np.empty((lat.size, 4, 5))
    _sh_table_kernel(np.ascontiguousarray(lat, dtype=float), np.ascontiguousarray(lon, dtype=float), _NMAX,
                     _TERM_N, _TERM_M, _SH_ANM, _SH_BNM, out)
    return out


def _gmf_nodes(lat, lon):
    """
    The epoch-independent GMF parts of each node: the mean and annual amplitudes of the hydrostatic and wet a, and
    cos(lat).

    Parameters: lat, lon (1-D float arrays, degrees). Returns: tuple (ah_mean, ah_amp, aw_mean, aw_amp, cos_lat).

    Examples
    --------
    >>> _gmf_nodes(np.array([40.0, 40.25]), np.array([10.0, 10.25]))
    """
    t = _GMF_TABLES
    out = np.empty((5, lat.size))
    for k in range(lat.size):
        V, W = _legendre(lat[k], lon[k], _GMF_NMAX)
        v = np.array([V[n, m] for n, m in _TERMS[:_GMF_NTERMS]])
        w = np.array([W[n, m] for n, m in _TERMS[:_GMF_NTERMS]])
        out[0, k] = t['ah_mean'] @ v + t['bh_mean'] @ w
        out[1, k] = t['ah_amp'] @ v + t['bh_amp'] @ w
        out[2, k] = t['aw_mean'] @ v + t['bw_mean'] @ w
        out[3, k] = t['aw_amp'] @ v + t['bw_amp'] @ w
        out[4, k] = math.cos(math.radians(lat[k]))
    return tuple(out)


def _gmf_epochs(parts, lat, epochs):
    """
    The GMF ah, aw and ch of every node at every epoch.

    Parameters: parts (from _gmf_nodes); lat (1-D float, degrees; its sign selects the hemisphere); epochs (list of
    pandas.Timestamp, UTC). Returns: numpy.ndarray (epochs, 3, nodes).

    Examples
    --------
    >>> _gmf_epochs(_gmf_nodes(np.array([40.0]), np.array([10.0])), np.array([40.0]), [pd.Timestamp('2016-05-11 08:00')])
    """
    ah_mean, ah_amp, aw_mean, aw_amp, cos_lat = parts
    south = lat < 0
    c11 = np.where(south, 0.007, 0.005)
    c10 = np.where(south, 0.002, 0.001)
    out = np.empty((len(epochs), 3, lat.size))
    for k, epoch in enumerate(epochs):
        doy = (utils_epochs._stamp(epoch) - pd.Timestamp('1980-01-01')).total_seconds() / 86400.0 - 27.0
        cyc = 2 * math.pi * doy / 365.25
        cos_cyc = math.cos(cyc)
        out[k, 0] = (ah_mean + ah_amp * cos_cyc) * 1e-5
        out[k, 1] = (aw_mean + aw_amp * cos_cyc) * 1e-5
        cos_h = np.where(south, math.cos(cyc + math.pi), math.cos(cyc + 0.0))
        out[k, 2] = 0.062 + ((cos_h + 1) * c11 / 2 + c10) * (1 - cos_lat)
    return out


def _times(times):
    """
    The times as naive UTC pandas.Timestamp values, in a list.

    Parameters: times (str or pandas.Timestamp, or a list of them). Returns: list of pandas.Timestamp. Raises:
    ValueError, one ERROR: line, when there is no time.

    Examples
    --------
    >>> _times(['2016-05-11 08:30'])
    """
    if isinstance(times, (str, pd.Timestamp, datetime.datetime)):
        times = [times]
    out = [utils_epochs._stamp(t) for t in times]
    if not out:
        raise ValueError('ERROR: no times given')
    return out


def _plan(times, hours):
    """
    For each time, the two bracketing epochs and the weight of the second; a time on an epoch gives (e0, None, 0.0).

    Parameters: times (list of pandas.Timestamp); hours (int, epoch step in hours). Returns: list of (e0, e1, w).

    Examples
    --------
    >>> _plan([pd.Timestamp('2016-05-11 08:30')], 6)
    """
    step = pd.Timedelta(hours=hours)
    out = []
    for t in times:
        e0 = t.floor(step)
        if e0 == t:
            out.append((e0, None, 0.0))
        else:
            out.append((e0, e0 + step, (t - e0) / step))
    return out


def _tiles(index, name):
    """
    The sorted lower-left corners (lon0, lat0) of the tile files that an index lists.

    Parameters: index (utils_epochs.Index); name (str, model name). Returns: list of (lon0, lat0). Raises: ValueError,
    one ERROR: line, for a listed source that is not a tile file.

    Examples
    --------
    >>> _tiles(utils_epochs.Index('dem_tiles/ERA5.vrt'), 'ERA5')
    """
    offset = utils_epochs.SOURCES[name]['offset']
    corners = set()
    for arr in index.arrays.values():
        for sources in arr['sources'].values():
            for path, _, _, _ in sources:
                m = utils_epochs._FILE.search(os.path.basename(path))
                if m is None:
                    raise ValueError(f'ERROR: {index.path} lists a source that is not a tile file: {path}')
                corners.add(utils_epochs._parse_tile(m.group(1), offset))
    return sorted(corners)


# lapse rate (K/m) of the IFS law below the node surface, Gamma_st (eqs 1-6)
_GAMMA_ST = 0.0065
# steps in ln p of the law below the node surface, and its largest ln(p / p_s).
_BELOW_UMAX = 2.0


_ZENITH_LAW_STEPS = 4000
# GMF coefficients and Niell height term per node and time (pixel mapping)
ZenithTables = namedtuple('ZenithTables', 'h0 step hydro wet mapping niell')
# Counters of _zenith_kernel (see _zenith_warnings).
_ZENITH_COUNTS = 6


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_refractivity(p, t, q, c):
    """
    Dry and wet refractivity of one point, in units of 1e-6.

    Parameters: p (hPa); t (K); q, c (kg/kg). Returns: tuple (dry, wet).

    Examples
    --------
    >>> _zenith_refractivity(1000.0, 288.0, 0.01, 0.0)
    """
    den = _EPS + (1.0 - _EPS) * q
    r = 1.0 / (t * den)
    e = p * q * t * r
    it = den * r
    pd = p - e
    rho = (pd / _RD + e / _RV) * 100.0 * it
    return _K_DRY * pd * it, (_K_WET * e + _K_WET2 * e * it) * it + _K_LIQ * c * rho * 1000.0


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_gravity(g0, c1, h):
    """
    Normal gravity (m/s2) at the geometric height h of a node.

    Parameters: g0 (m/s2, gravity at the ellipsoid); c1 (1/m); h (m). Returns: float.

    Examples
    --------
    >>> _zenith_gravity(9.8, 3e-7, 100.0)
    """
    return g0 * (1.0 - c1 * h + 3.0 * h * h / (_A * _A))


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_height(h, ig, dphi, g0, c1):
    """
    The geometric height and the inverse gravity after a geopotential step dphi (m2/s2, positive up) from h.

    Parameters: h (m); ig (inverse gravity at h, s2/m); dphi (m2/s2); g0 (m/s2); c1 (1/m). Returns: (height, inverse
    gravity).

    Examples
    --------
    >>> _zenith_height(0.0, 1.0 / 9.8, 98.0, 9.8, 3e-7)
    """
    hp = h + dphi * ig
    drop = (hp - h) * g0 * (1.0 - 0.5 * c1 * (hp + h) + (hp * hp + hp * h + h * h) / (_A * _A))
    ig = 1.0 / _zenith_gravity(g0, c1, hp)
    return hp + (dphi - drop) * ig, ig


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_climb(h, ig, dphi, g0, c1, counts):
    """
    _zenith_height for a step up the column; a height that would fall is held and counted in counts[4].

    Parameters: as _zenith_height, and counts (int64 array). Returns: (height, inverse gravity).

    Examples
    --------
    >>> _zenith_climb(0.0, 1.0 / 9.8, 98.0, 9.8, 3e-7, np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    hn, ign = _zenith_height(h, ig, dphi, g0, c1)
    if hn < h:
        counts[4] += 1
        return h, ig
    return hn, ign


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_column(sp, phis, geo, tt, qq, cl, P, work, counts):
    """
    Fills work (float64 (5, levels + 1)) with the column of one node and epoch: node 0 is the surface, then the levels
    bottom first. Levels below the surface take the surface values, counted in counts[0:4].

    Parameters: sp (Pa, surface pressure); phis (m2/s2); geo, tt, qq, cl (m2/s2, K, kg/kg, kg/kg; levels); P (Pa,
    levels); work; counts (int64 array). Returns: the number of levels below the surface. Raises: ValueError, one ERROR:
    line, when every level is below the surface.

    Examples
    --------
    >>> work = np.empty((5, 3)); counts = np.zeros(_ZENITH_COUNTS, dtype=np.int64)
    >>> _zenith_column(101325.0, 0.0, np.array([1000.0, 50000.0]), np.array([288.0, 255.0]), np.array([0.01, 0.001]), np.zeros(2), np.array([100000.0, 50000.0]), work, counts)
    """
    nun = 0
    for j in range(P.size):
        if P[j] > sp:
            nun += 1
    if nun >= P.size:
        raise ValueError('ERROR: a column of the ERA5 index has every pressure level below its surface pressure')
    lsp = math.log(sp)
    work[0, 0] = phis
    work[1, 0] = lsp
    work[2, 0] = tt[nun]
    work[3, 0] = qq[nun]
    work[4, 0] = cl[nun]
    for j in range(P.size):
        under = P[j] > sp
        k = max(j, nun)
        raw = phis if under else geo[k]
        if raw < phis:
            counts[0] += 1
        if j < nun and not under:
            counts[1] += 1
        lnp = lsp if under else math.log(P[k])
        if lnp > lsp:
            counts[2] += 1
        keep = min(lnp, lsp)
        low = min(keep, work[1, j])
        if low != keep:
            counts[3] += 1
        work[0, j + 1] = max(raw, phis)
        work[1, j + 1] = low
        work[2, j + 1] = tt[k]
        work[3, j + 1] = qq[k]
        work[4, j + 1] = cl[k]
    return nun


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_first(top, h0, step, count):
    """
    The number of table entries (heights h0 + k step, k < count) below the height top.

    Parameters: top, h0, step (m); count (int). Returns: int.

    Examples
    --------
    >>> _zenith_first(12.0, 0.0, 5.0, 100)
    """
    k = 0
    if top > h0:
        k = min(count, int(math.ceil((top - h0) / step)))
    while k > 0 and h0 + (k - 1) * step >= top:
        k -= 1
    while k < count and h0 + k * step < top:
        k += 1
    return k


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_basis(t, dz):
    """
    The four cubic Hermite weights at the fraction t of a span of length dz (m); the slope weights carry dz and 1e-6.

    Parameters: t (float, 0 to 1); dz (m). Returns: tuple (value and slope weights at the start, value and slope weights
    at the end).

    Examples
    --------
    >>> _zenith_basis(0.5, 10.0)
    """
    t2 = t * t
    t3 = t2 * t
    return 2.0 * t3 - 3.0 * t2 + 1.0, (t3 - 2.0 * t2 + t) * dz * 1e-6, 3.0 * t2 - 2.0 * t3, (t3 - t2) * dz * 1e-6


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_span(hyd, wet, k, h0, step, ha, hb, ca, cb, na, nb):
    """
    Writes the cumulative hydrostatic and wet delays of the table entries from k on, at heights in [ha, hb), by Hermite
    interpolation; an empty span writes nothing.

    Parameters: hyd, wet (float64 tables, output); k (int); h0, step (m); ha, hb (m); ca, cb, na, nb ((dry, wet) delays
    and refractivities at the ends). Returns: the first entry not reached.

    Examples
    --------
    >>> hyd, wet = np.zeros(4), np.zeros(4)
    >>> _zenith_span(hyd, wet, 0, 0.0, 5.0, 0.0, 12.0, (0.0, 0.0), (1e-3, 1e-3), (100.0, 50.0), (100.0, 50.0))
    """
    dz = hb - ha
    if dz <= 0.0:
        return k
    inv = 1.0 / dz
    while k < hyd.size:
        h = h0 + k * step
        if h >= hb:
            break
        w0, w1, w2, w3 = _zenith_basis((h - ha) * inv, dz)
        hyd[k] = w0 * ca[0] + w1 * na[0] + w2 * cb[0] + w3 * nb[0]
        wet[k] = w0 * ca[1] + w1 * na[1] + w2 * cb[1] + w3 * nb[1]
        k += 1
    return k


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_mix(work, m, s):
    """
    The temperature, humidity and cloud at sub-step s of the layer that ends at node m, linear between its two nodes.

    Parameters: work (from _zenith_column); m, s (int). Returns: tuple (temperature, humidity, cloud).

    Examples
    --------
    >>> _zenith_mix(np.array([[0.0, 5000.0], [11.5, 11.0], [288.0, 255.0], [0.01, 0.001], [0.0, 0.0]]), 1, 8)
    """
    f = s / _SUB
    return (work[2, m - 1] + f * (work[2, m] - work[2, m - 1]), work[3, m - 1] + f * (work[3, m] - work[3, m - 1]),
            work[4, m - 1] + f * (work[4, m] - work[4, m - 1]))


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_above(work, g0, c1, hs, und, h0, step, ks, hyd, wet, counts):
    """
    The delay of the column above the surface, from the bottom to the top, accumulated into hyd and wet from entry ks on
    (Simpson rule over the ln p sub-steps of each layer).

    Parameters: work (from _zenith_column); g0, c1 (as _zenith_gravity); hs (m, surface height); und (m, undulation);
    h0, step (m); ks (int); hyd, wet (output); counts (int64 array). Returns: (dry delay, wet delay, inverse gravity at
    the top).

    Examples
    --------
    >>> work = np.array([[0.0, 5000.0], [11.5, 11.0], [288.0, 255.0], [0.01, 0.001], [0.0, 0.0]])
    >>> hyd, wet = np.zeros(8), np.zeros(8)
    >>> _zenith_above(work, 9.8, 3e-7, 0.0, 0.0, 0.0, 5.0, 0, hyd, wet, np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    k = ks
    h = hs
    ig = 1.0 / _zenith_gravity(g0, c1, h)
    cd = 0.0
    cw = 0.0
    for m in range(1, work.shape[1]):
        dphi = work[0, m] - work[0, m - 1]
        dx = work[1, m - 1] - work[1, m]
        if dx <= 0.0 and dphi == 0.0:
            continue
        sub = dx / _SUB
        dpg = dphi / _SUB
        fac = 1e-6 * dphi / dx if dx > 0.0 else 0.0
        ratio = math.exp(-sub)
        p = math.exp(work[1, m - 1]) / 100.0
        t, q, c = _zenith_mix(work, m, 0)
        nd0, nw0 = _zenith_refractivity(p, t, q, c)
        for s in range(0, _SUB, 2):
            h1, ig1 = _zenith_climb(h, ig, dpg, g0, c1, counts)
            h2, ig2 = _zenith_climb(h1, ig1, dpg, g0, c1, counts)
            p1 = p * ratio
            p2 = p1 * ratio
            t, q, c = _zenith_mix(work, m, s + 1)
            nd1, nw1 = _zenith_refractivity(p1, t, q, c)
            t, q, c = _zenith_mix(work, m, s + 2)
            nd2, nw2 = _zenith_refractivity(p2, t, q, c)
            fd0, fd1, fd2 = fac * ig * nd0, fac * ig1 * nd1, fac * ig2 * nd2
            fw0, fw1, fw2 = fac * ig * nw0, fac * ig1 * nw1, fac * ig2 * nw2
            cd1 = cd + sub / 12.0 * (5.0 * fd0 + 8.0 * fd1 - fd2)
            cw1 = cw + sub / 12.0 * (5.0 * fw0 + 8.0 * fw1 - fw2)
            cd2 = cd + sub / 3.0 * (fd0 + 4.0 * fd1 + fd2)
            cw2 = cw + sub / 3.0 * (fw0 + 4.0 * fw1 + fw2)
            k = _zenith_span(hyd, wet, k, h0, step, h + und, h1 + und, (cd, cw), (cd1, cw1), (nd0, nw0), (nd1, nw1))
            k = _zenith_span(hyd, wet, k, h0, step, h1 + und, h2 + und, (cd1, cw1), (cd2, cw2), (nd1, nw1), (nd2, nw2))
            cd, cw = cd2, cw2
            h, ig, p, nd0, nw0 = h2, ig2, p2, nd2, nw2
    while k < hyd.size:
        hyd[k] = cd
        wet[k] = cw
        k += 1
    return cd, cw, ig


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_lapse(hz, ts):
    """
    The lapse rate (K/m) of IFS equations 4 and 5 at the height hz (m) and the temperature ts (K).

    Parameters: hz (m); ts (K). Returns: float (K/m).

    Examples
    --------
    >>> _zenith_lapse(0.0, 288.0)
    """
    if hz < 2000.0:
        return _GAMMA_ST
    a = ts + _GAMMA_ST * hz
    w = min(max((hz - 2000.0) / 500.0, 0.0), 1.0)
    return max((1.0 - w) * a + w * min(a, 298.0) - ts, 0.0) / max(hz, 1.0)


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_below(sp, phis, work, pl, g0, c1, hs, und, ds, h0, step, ks, hyd, wet, counts):
    """
    Writes the delays below the node surface (IFS law, equations 1 to 6) into entries 0 to ks - 1; entries deeper than
    the last step are NaN and counted in counts[5].

    Parameters: sp (Pa); phis (m2/s2); work (from _zenith_column); pl (Pa, pressure of the lowest level above the
    surface); g0, c1; hs, und (m); ds ((dry, wet) delays at the surface); h0, step (m); ks (int); hyd, wet (output);
    counts (int64 array). Returns: None.

    Examples
    --------
    >>> work = np.array([[0.0, 5000.0], [11.5, 11.0], [288.0, 255.0], [0.01, 0.001], [0.0, 0.0]])
    >>> hyd, wet = np.zeros(8), np.zeros(8)
    >>> _zenith_below(101325.0, 0.0, work, 100000.0, 9.8, 3e-7, 0.0, 0.0, (2.0, 0.1), 0.0, 5.0, 8, hyd, wet, np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    k = ks - 1
    if k < 0:
        return
    ts = work[2, 0] + _GAMMA_ST * (_RD / _G0) * (sp / pl - 1.0) * work[2, 0]
    kap = _zenith_lapse(phis / _G0, ts) * _RD / _G0
    kap6 = _GAMMA_ST * _RD / _G0
    q = work[3, 0]
    c = work[4, 0]
    top = hs + und
    du = _BELOW_UMAX / _ZENITH_LAW_STEPS
    grow = math.exp(du)
    eu = 1.0
    nd0, nw0 = _zenith_refractivity(sp / 100.0, ts, q, c)
    h = hs
    ig = 1.0 / _zenith_gravity(g0, c1, h)
    d0 = 0.0
    pdef0 = 0.0
    ad0 = 0.0
    aw0 = 0.0
    for i in range(1, _ZENITH_LAW_STEPS + 1):
        u = du * i
        eu *= grow
        y = kap * u
        y6 = kap6 * u
        t = ts * (1.0 + y + y * y / 2.0 + y * y * y / 6.0)
        pdef = _RD * ts * u * (1.0 + y6 / 2.0 + y6 * y6 / 6.0)
        h, ig = _zenith_height(h, ig, pdef0 - pdef, g0, c1)
        d1 = hs - h
        nd1, nw1 = _zenith_refractivity(eu * sp / 100.0, t, q, c)
        ad1 = ad0 + 0.5 * (nd0 + nd1) * (d1 - d0) * 1e-6
        aw1 = aw0 + 0.5 * (nw0 + nw1) * (d1 - d0) * 1e-6
        while k >= 0 and top - (h0 + k * step) <= d1:
            w0, w1, w2, w3 = _zenith_basis((top - (h0 + k * step) - d0) / (d1 - d0), d1 - d0)
            hyd[k] = ds[0] + w0 * ad0 + w1 * nd0 + w2 * ad1 + w3 * nd1
            wet[k] = ds[1] + w0 * aw0 + w1 * nw0 + w2 * aw1 + w3 * nw1
            k -= 1
        if k < 0:
            return
        d0, pdef0, ad0, aw0, nd0, nw0 = d1, pdef, ad1, aw1, nd1, nw1
    counts[5] += k + 1
    for j in range(k + 1):
        hyd[j] = np.nan
        wet[j] = np.nan


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_node(sp, phis, geo, tt, qq, cl, P, und, g0, c1, h0, step, hyd, wet, work, counts):
    """
    The zenith delay tables (hyd, wet) of one node at one epoch over the heights h0 + k step: the column, the delay
    above the surface and the law below it.

    Parameters: as _zenith_column, then und (m); g0, c1 (as _zenith_gravity); h0, step (m); hyd, wet (output); work;
    counts (int64 array). Returns: None.

    Examples
    --------
    >>> hyd, wet = np.zeros(8), np.zeros(8)
    >>> _zenith_node(101325.0, 0.0, np.array([1000.0, 50000.0]), np.array([288.0, 255.0]), np.array([0.01, 0.001]), np.zeros(2), np.array([100000.0, 50000.0]), 0.0, 9.8, 3e-7, 0.0, 5.0, hyd, wet, np.empty((5, 3)), np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    nun = _zenith_column(sp, phis, geo, tt, qq, cl, P, work, counts)
    hs = phis / _G0
    for _ in range(6):
        hs = phis / _zenith_gravity(g0, c1, hs / 2.0)
    ks = _zenith_first(hs + und, h0, step, hyd.size)
    cd, cw, ig = _zenith_above(work, g0, c1, hs, und, h0, step, ks, hyd, wet, counts)
    closure = 1e-6 * _K_DRY * (_RD / 100.0) * math.exp(work[1, work.shape[1] - 1]) * ig
    ds = (closure + cd, cw)
    for k in range(ks, hyd.size):
        hyd[k] = ds[0] - hyd[k]
        wet[k] = ds[1] - wet[k]
    _zenith_below(sp, phis, work, P[nun], g0, c1, hs, und, ds, h0, step, ks, hyd, wet, counts)


@njit(nogil=True, cache=True, error_model='numpy')
def _zenith_kernel(sp, phis, geo, tt, qq, cl, P, und, g0, c1, h0, step, hyd, wet, counts):
    """
    Writes the zenith delay tables of every node at every epoch, serially, into hyd and wet (epochs, nodes, entries).

    Parameters: sp, phis (epochs, nodes); geo, tt, qq, cl (epochs, nodes, levels, bottom level first); P (Pa, levels);
    und, g0, c1 (nodes); h0, step (m); hyd, wet (output); counts (int64 array). Returns: None.

    Examples
    --------
    >>> hyd, wet = np.zeros((1, 1, 8)), np.zeros((1, 1, 8))
    >>> _zenith_kernel(np.full((1, 1), 101325.0), np.zeros((1, 1)), np.array([[[1000.0, 50000.0]]]), np.array([[[288.0, 255.0]]]), np.array([[[0.01, 0.001]]]), np.zeros((1, 1, 2)), np.array([100000.0, 50000.0]), np.zeros(1), np.full(1, 9.8), np.full(1, 3e-7), 0.0, 5.0, hyd, wet, np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    work = np.empty((5, P.size + 1))
    for e in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            _zenith_node(sp[e, j], phis[e, j], geo[e, j], tt[e, j], qq[e, j], cl[e, j], P, und[j], g0[j], c1[j], h0, step,
                         hyd[e, j], wet[e, j], work, counts)


def _zenith_warnings(counts):
    """
    The warning lines for the nonzero counts of _zenith_kernel: the held heights, and the table entries below the
    surface law that are NaN.

    Parameters: counts (int64 array). Returns: None.

    Examples
    --------
    >>> _zenith_warnings(np.zeros(_ZENITH_COUNTS, dtype=np.int64))
    """
    texts = ('the ERA5 geopotential of {} levels above the surface lies below the surface geopotential; they are held at the surface value',
             'the ERA5 levels are not in pressure order: {} levels above the surface take the values of the level at the surface index',
             'the ERA5 levels are not in pressure order: {} levels have a pressure above the surface pressure; they are held at the surface pressure',
             'the ERA5 pressure is not monotone in {} levels; they are held at the level beneath',
             'the ERA5 geometric heights fell below the level beneath in {} node sub-steps (the geopotential is not monotone); they are held at that level',
             'the ERA5 zenith tables have {} entries deeper below the node surface than the law below the surface reaches (the entries are NaN)')
    for n, text in zip(counts, texts):
        if n:
            warnings.warn(text.format(int(n)))


def era5_zenith_tables(sp, phis, geo, tt, qq, cl, levels, lat, lon, undulation, epochs, first, second, w0, w1, h0, step, count):
    """
    The ERA5 zenith delay tables of the nodes of one tile over the ellipsoidal height, with the geoid height added, one
    computation per node and epoch; the two epochs of each time are folded by the time weights.

    Parameters: sp (Pa), phis (m2/s2), (epochs, nodes); geo, tt, qq, cl ((epochs, nodes, levels), bottom level first);
    levels (list of str, hPa, bottom first); lat, lon (degrees) and undulation (m), (nodes,); epochs (list of
    pandas.Timestamp); first, second, w0, w1 ((times,) epoch positions and weights; second is -1 for a time on an
    epoch); h0, step (m); count (int).

    Returns: ZenithTables (h0, step, hydro, wet, mapping, niell): hydro and wet (times, nodes, count) in m; mapping
    (times, nodes, 2, 3) of a, b, c for the hydrostatic (0) and wet (1) mapping; niell (4,). Raises: ValueError, one
    ERROR: line, for a step that is not positive or a table of fewer than two entries; and as _zenith_column.

    Examples
    --------
    >>> era5_zenith_tables(np.full((1, 1), 101325.0), np.zeros((1, 1)), np.array([[[1000.0, 50000.0]]]), np.array([[[288.0, 255.0]]]), np.array([[[0.01, 0.001]]]), np.zeros((1, 1, 2)), ['1000', '500'], np.array([40.0]), np.array([10.0]), np.zeros(1), [pd.Timestamp('2016-05-11 08:00')], np.array([0]), np.array([-1]), np.array([1.0]), np.array([0.0]), 0.0, 5.0, 8)
    """
    if not step > 0.0 or count < 2:
        raise ValueError('ERROR: the zenith tables need a positive height step and at least two entries')
    first, second = np.asarray(first), np.asarray(second)
    sp, phis, geo, tt, qq, cl = (np.ascontiguousarray(a, dtype=float) for a in (sp, phis, geo, tt, qq, cl))
    lat, lon, undulation = (np.asarray(a, dtype=float) for a in (lat, lon, undulation))
    P = np.array([float(t) for t in levels]) * 100.0
    s2 = np.sin(np.radians(lat)) ** 2
    g0 = _GE * (1.0 + _KG * s2) / np.sqrt(1.0 - _E2 * s2)
    c1 = 2.0 / _A * (1.0 + _F + _M - 2.0 * _F * s2)
    hyd = np.empty(sp.shape + (count,))
    wet = np.empty(sp.shape + (count,))
    counts = np.zeros(_ZENITH_COUNTS, dtype=np.int64)
    _zenith_kernel(sp, phis, geo, tt, qq, cl, P, undulation, g0, c1, float(h0), float(step), hyd, wet, counts)
    _zenith_warnings(counts)
    gmf = _gmf_epochs(_gmf_nodes(lat, lon), lat, list(epochs))
    mapping = np.empty(gmf.shape[:1] + (gmf.shape[2], 2, 3))
    mapping[:, :, 0, 0], mapping[:, :, 0, 1], mapping[:, :, 0, 2] = gmf[:, 0], _GMF_BH, gmf[:, 2]
    mapping[:, :, 1, 0], mapping[:, :, 1, 1], mapping[:, :, 1, 2] = gmf[:, 1], _GMF_BW, _GMF_CW
    return ZenithTables(float(h0), float(step), _fold(hyd, first, second, w0, w1), _fold(wet, first, second, w0, w1),
                        _fold(mapping, first, second, w0, w1), np.array([1e-3, *_HT_COEF]))


def _fold(values, first, second, w0, w1):
    """
    The two epochs of each time folded by the time weights: w0 times the first epoch plus w1 times the second; a time on
    an epoch (second -1) takes the first epoch alone.

    Parameters: values (epochs on the first axis); first, second (epoch positions); w0, w1 ((times,) weights). Returns:
    array with the times on its first axis.

    Examples
    --------
    >>> _fold(np.array([[1.0], [2.0]]), np.array([0, 0]), np.array([1, -1]), np.array([0.25, 1.0]), np.array([0.75, 0.0]))
    """
    shape = (-1,) + (1,) * (values.ndim - 1)
    has = np.asarray(second) >= 0
    other = np.where(has, second, first)
    return (np.asarray(w0, dtype=float).reshape(shape) * values[first]
            + np.where(has, np.asarray(w1, dtype=float), 0.0).reshape(shape) * values[other])


def _nodes(values, flat):
    """
    The values of a block at the nodes, as float64.

    Parameters: values (numpy.ndarray (rows, columns)); flat (row-major node indices). Returns: float64 array.

    Examples
    --------
    >>> _nodes(np.arange(9.0).reshape(3, 3), np.array([0, 4]))
    """
    return np.asarray(values, dtype=float).reshape(-1)[flat]


def _root(vrt):
    """
    The root element of a VRT index written by utils_epochs.store().

    Parameters: vrt (str). Returns: xml.etree.ElementTree.Element. Raises: ValueError, one ERROR: line, when the VRT
    cannot be read.

    Examples
    --------
    >>> _root('dem_tiles.vrt')
    """
    try:
        return ET.parse(vrt).getroot()
    except (OSError, ET.ParseError) as e:
        raise ValueError(f'ERROR: {vrt} cannot be read: {type(e).__name__}') from None


def _bands(vrt):
    """
    The VRTRasterBand elements of a VRT index by (description, variable, level); the description is the ISO datetime of
    the epoch.

    Parameters: vrt (str). Returns: dict {(description, variable, level): element}.

    Examples
    --------
    >>> _bands('dem_tiles.vrt')
    """
    out = {}
    for band in _root(vrt).findall('VRTRasterBand'):
        meta = {m.get('key'): m.text for m in band.findall('Metadata/MDI')}
        out.setdefault((band.findtext('Description'), meta.get('VARIABLE'), meta.get('LEVEL')), band)
    return out


def _listed(band, vrt):
    """
    The absolute paths of the tile files that a band lists.

    Parameters: band (VRTRasterBand element); vrt (str, for relative paths). Returns: list of str. Raises: ValueError,
    one ERROR: line, when a source is not a NETCDF variable.

    Examples
    --------
    >>> _listed(_bands('dem_tiles.vrt')[('2016-05-11T06:00:00', 'zhd', 'surface')], 'dem_tiles.vrt')
    """
    base = os.path.dirname(os.path.abspath(vrt))
    out = []
    for src in band.findall('SimpleSource'):
        name = src.find('SourceFilename')
        text = (name.text or '').strip()
        if not text.startswith('NETCDF:'):
            raise ValueError(f'ERROR: {vrt} lists a source without a NETCDF variable: {text}')
        rel = text[len('NETCDF:'):].rsplit(':', 1)[0].strip('"')
        out.append(os.path.join(base, rel) if name.get('relativeToVRT', '0') == '1' else rel)
    return out


def _levels(index):
    """
    The ERA5 pressure labels as stored, from the highest pressure down.

    Parameters: index (utils_epochs.Index of the ERA5 folder). Returns: list of str. Raises: ValueError, one ERROR:
    line, when the levels are not pressures in hPa or there are none.

    Examples
    --------
    >>> _levels(utils_epochs.Index('dem_tiles/ERA5.vrt'))
    """
    try:
        pairs = sorted(((float(t), t) for t in index.levels), reverse=True)
    except (TypeError, ValueError):
        raise ValueError('ERROR: the levels of the ERA5 index are not pressures in hPa') from None
    if not pairs:
        raise ValueError('ERROR: the ERA5 index holds no pressure level')
    return [t for _, t in pairs]


def _required(name, index):
    """
    The (variable, level) bands that the terms of a model need for each epoch.

    Parameters: name (str, model name); index (utils_epochs.Index; the ERA5 levels are read from it). Returns: list of
    (variable, level) tuples.

    Examples
    --------
    >>> _required('VMF3_OP', utils_epochs.Index('dem_tiles/VMF3_OP.vrt'))
    """
    if name == _NAME_ERA5:
        return ([(v, lev) for v in _ERA5_LEVEL_VARIABLES for lev in _levels(index)]
                + [(v, _LEVEL) for v in _ERA5_SURFACE_VARIABLES])
    return [(v, _LEVEL) for v in _VARIABLES]


def model(path):
    """
    The tropospheric model of a folder: its one VRT index among ERA5, VMF3_OP, VMF3_EI and VMF3_FC (IONEX is not
    tropospheric).

    Parameters: path (str, folder of the downloaded model fields). Returns: the model name, str. Raises: ValueError, one
    ERROR: line, when the folder holds none or more than one.

    Examples
    --------
    >>> model('dem_tiles')
    'VMF3_OP'
    """
    names = [n for n in utils_epochs.sources(path) if n != 'IONEX']
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(f'ERROR: no tropospheric index (ERA5.vrt, VMF3_OP.vrt, VMF3_EI.vrt or VMF3_FC.vrt) in {path}')
    raise ValueError(f'ERROR: {path} holds more than one tropospheric index: {", ".join(names)}')


def _verify(index, stamps):
    """
    Checks that an index holds the bands that the epochs need, and that their tile files exist.

    Parameters: index (utils_epochs.Index); stamps (list of pandas.Timestamp; each time needs the epochs around it).
    Raises: ValueError, one ERROR: line, for a missing epoch, band or tile file.

    Examples
    --------
    >>> _verify(utils_epochs.Index('dem_tiles/ERA5.vrt'), [pd.Timestamp('2016-05-11 06:30')])
    """
    name = index.model
    required = _required(name, index)
    epochs = sorted({e for e0, e1, w in _plan(stamps, utils_epochs.SOURCES[name]['hours'])
                     for e in (e0, e1) if e is not None})
    for e in epochs:
        ei = index.epoch_of(e)
        for v, lev in required:
            arr = index.arrays.get(v)
            if arr is None or not arr['sources'].get(ei):
                where = '' if lev == _LEVEL else f' at level {lev}'
                raise ValueError(f'ERROR: {name} has no {v}{where} of epoch {e:%Y-%m-%d %H:%M} in {index.path}: store() that '
                                 f'epoch first')
            for path, _, _, _ in arr['sources'][ei]:
                if not exists(path):
                    raise ValueError(f'ERROR: {path} listed in {index.path} is missing: call store() again with the same items '
                                     f'to fetch it')


def verify(path, times):
    """
    Checks that the epochs that the times need are stored, with their variables, and that their listed tile files exist;
    reads no data.

    Parameters: path (str, folder of the downloaded model fields); times (list of str or pandas.Timestamp, UTC). Raises:
    ValueError, one ERROR: line, naming the first missing epoch, source or tile file.

    Examples
    --------
    >>> verify('dem_tiles', ['2016-05-11 08:30'])
    """

    name = model(path)
    stamps = _times(times)
    _verify(utils_epochs.Index(os.path.join(os.path.abspath(path), name + '.vrt')), stamps)


_NLEVEL, _NPART = 1, 3


@njit(nogil=True, cache=True, error_model='numpy')
def _vmf3_table_kernel(zhd, zwd, hg, phi, h0, step, count, hyd, wet):
    """
    Writes the VMF3 zenith delays at the table entries: hydrostatic (1 - 2.26e-5 D)^5.225 / (1 - 0.28e-6 D / f_g) and
    wet exp(-D / 2000 m), with D = h - orography and f_g = 1 - 0.00266 cos 2 phi - 0.28e-6 orography.

    Parameters: zhd, zwd, hg ((epochs, nodes), m: zenith hydrostatic and wet delays, orography); phi (float64 (nodes,),
    radians); h0, step (m); count (int); hyd, wet ((epochs, nodes, count), m, output). Returns: None.

    Examples
    --------
    >>> hyd, wet = np.empty((1, 1, 2)), np.empty((1, 1, 2))
    >>> _vmf3_table_kernel(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)), np.zeros(1), 0.0, 100.0, 2, hyd, wet)
    """
    for e in range(zhd.shape[0]):
        for n in range(zhd.shape[1]):
            f_g = 1.0 - 0.00266 * math.cos(2.0 * phi[n]) - 0.28e-6 * hg[e, n]
            for k in range(count):
                d = h0 + step * k - hg[e, n]
                hyd[e, n, k] = zhd[e, n] * math.pow(1.0 - 2.26e-5 * d, 5.225) / (1.0 - 0.28e-6 * d / f_g)
                wet[e, n, k] = zwd[e, n] * math.exp(-d / 2000.0)


def vmf3_zenith_tables(ah, aw, zhd, zwd, orography, lat, lon, epochs, first, second, w0, w1, h0, step, count):
    """
    The VMF3 zenith delay tables of the nodes of one tile, one formula per node and epoch, folded into the times; the
    mapping coefficients are ah and aw, with b and c of the spherical harmonic model. The layout is that of
    era5_zenith_tables.

    Parameters: ah, aw, zhd, zwd, orography ((epochs, nodes) VMF3 values; zenith delays and orography in m); lat, lon
    (degrees, (nodes,)); epochs (list of pandas.Timestamp); first, second, w0, w1 (as in era5_zenith_tables); h0, step
    (m); count (int). Returns: ZenithTables as era5_zenith_tables. Raises: ValueError, one ERROR: line, for a step that
    is not positive or a table of fewer than two entries.

    Examples
    --------
    >>> vmf3_zenith_tables(np.full((1, 1), 0.0), np.full((1, 1), 0.0), np.full((1, 1), 2.3), np.full((1, 1), 0.1),
    ...                    np.full((1, 1), 100.0), np.array([40.0]), np.array([10.0]), [pd.Timestamp('2016-05-11 06:00')],
    ...                    np.array([0]), np.array([-1]), np.array([1.0]), np.array([0.0]), 0.0, 5.0, 8)
    """
    if not step > 0.0 or count < 2:
        raise ValueError('ERROR: the zenith tables need a positive height step and at least two entries')
    lat, lon = (np.asarray(a, dtype=float) for a in (lat, lon))
    phi = np.radians(lat)
    coef = _sh_epochs(_sh_table_parts(lat, lon), [_doy(e) for e in epochs])
    mapping = np.empty((len(epochs), lat.size, 2, 3))
    mapping[:, :, 0, 0], mapping[:, :, 0, 1], mapping[:, :, 0, 2] = ah, coef[:, 0], coef[:, 2]
    mapping[:, :, 1, 0], mapping[:, :, 1, 1], mapping[:, :, 1, 2] = aw, coef[:, 1], coef[:, 3]
    zhd, zwd, hg = (np.ascontiguousarray(a, dtype=float) for a in (zhd, zwd, orography))
    hyd, wet = np.empty((len(epochs), lat.size, count)), np.empty((len(epochs), lat.size, count))
    _vmf3_table_kernel(zhd, zwd, hg, phi, float(h0), float(step), int(count), hyd, wet)
    return ZenithTables(float(h0), float(step), _fold(hyd, first, second, w0, w1), _fold(wet, first, second, w0, w1),
                        _fold(mapping, first, second, w0, w1), np.array([1e-3, *_HT_COEF]))


# _TABLE_COUNT entries of _TABLE_STEP m from _TABLE_H0: the global envelope (ellipsoidal heights)
_TABLE_H0 = -500.0
_TABLE_STEP = 10.0
_TABLE_COUNT = 951


def _era5_fields(got, flat, levels):
    """
    The ERA5 surface pressure and surface geopotential, as (epochs, nodes), and the four level variables, as (epochs,
    nodes, levels), of the nodes of a tile.

    Parameters: got (one tile window per epoch, from utils_epochs.tile_window); flat (row-major node indices of the
    tile); levels (list of str). Returns: tuple (sp, phis, fields).
    """
    sp = np.stack([_nodes(g[('surface_pressure', _LEVEL)], flat) for g in got])
    phis = np.stack([_nodes(g[('geopotential_at_surface', _LEVEL)], flat) for g in got])
    fields = {v: np.stack([np.stack([_nodes(g[(v, lev)], flat) for lev in levels], axis=-1) for g in got])
              for v in _ERA5_LEVEL_VARIABLES}
    return sp, phis, fields


def _zenith_tile(index, name, epochs, row0, col0, lat_b, lon_b, levels, undulation, first, second, w0, w1, h0, step, count,
                 scale):
    """
    The zenith delay tables of one tile for all times, in phase units: hydro and wet (times, ny, nx, count), each
    multiplied by scale, and mapping (times, ny, nx, 2, 3). The tile's file of each epoch is read by
    utils_epochs.tile_window.

    Parameters: index (utils_epochs.Index); name ('ERA5' or the VMF3 name); epochs (list of pandas.Timestamp, each
    once); row0, col0 (int, lattice row and column of the tile's first node); lat_b, lon_b (degrees, ny and nx nodes);
    levels (list of str, ERA5; None for VMF3); undulation ((ny, nx) m, ERA5; None for VMF3); first, second, w0, w1 (as
    in era5_zenith_tables); h0, step (m); count (int); scale (float, -4 pi / wavelength). Returns: tuple (hydro, wet,
    mapping), float64.
    """
    ny, nx = lat_b.size, lon_b.size
    flat = np.arange(ny * nx)
    jj, ii = np.divmod(flat, nx)
    lat_n, lon_n = lat_b[jj], lon_b[ii]
    if name == _NAME_ERA5:
        bands = [('surface_pressure', _LEVEL), ('geopotential_at_surface', _LEVEL)] + \
            [(v, lev) for v in _ERA5_LEVEL_VARIABLES for lev in levels]
        got = [utils_epochs.tile_window(index, e, bands, row0, col0) for e in epochs]
        sp, phis, field = _era5_fields(got, flat, levels)
        und = np.asarray(undulation, dtype=float)[jj, ii]
        tables = era5_zenith_tables(sp, phis, field['geopotential'], field['temperature'], field['specific_humidity'],
                                    field['specific_cloud_liquid_water_content'], levels, lat_n, lon_n, und, epochs, first,
                                    second, w0, w1, h0, step, count)
    else:
        got = [utils_epochs.tile_window(index, e, [(v, _LEVEL) for v in _VARIABLES], row0, col0) for e in epochs]
        values = {v: np.stack([_nodes(g[(v, _LEVEL)], flat) for g in got]) for v in _VARIABLES}
        tables = vmf3_zenith_tables(values['ah'], values['aw'], values['zhd'], values['zwd'], values['orography'], lat_n, lon_n,
                                    epochs, first, second, w0, w1, h0, step, count)
    hydro = tables.hydro.reshape(-1, ny, nx, count) * scale
    wet = tables.wet.reshape(-1, ny, nx, count) * scale
    mapping = tables.mapping.reshape(-1, ny, nx, 2, 3)
    return hydro, wet, mapping


def zenith_tiles(path, times, wavelength):
    """
    The zenith delay tables of a tropospheric model folder per tile, lazily: one dask chunk per tile, holding the tile's
    full node set for all times. The tables are in phase units (-4 pi / wavelength times the delay).

    Parameters: path (str, folder of the downloaded model fields); times (list of str or pandas.Timestamp, UTC; the
    first axis of the result); wavelength (float, metres). Returns: lazy xarray.Dataset: hydro and wet with dims (time,
    tile, ty, tx, height); mapping with dims (time, tile, ty, tx, component, coef); coordinates time, lat and lon
    (degrees), tile y0 and x0 (lattice row and column of each tile's first node), height (m); attributes h0, step,
    count, niell, shell_k. Raises: ValueError, one ERROR: line, for a wavelength that is not positive, a folder without
    a tropospheric index, or a missing epoch or tile file.
    """
    import dask
    import dask.array as da
    import xarray as xr
    if not float(wavelength) > 0:
        raise ValueError('ERROR: wavelength must be a positive length in metres')
    name = model(path)
    stamps = _times(times)
    index = utils_epochs.Index(os.path.join(os.path.abspath(path), name + '.vrt'))
    _verify(index, stamps)
    step = utils_epochs.SOURCES[name]['step']
    n = utils_epochs.SOURCES[name]['nodes']
    lat, lon = index.lat, index.lon
    if name == _NAME_ERA5:
        levels = _levels(index)
        undulation = utils_geoid.geoid_height(lat, lon, grid=True)
    else:
        levels = None
        undulation = None
    corners = sorted((int(round((c[1] - lat[0]) / step)), int(round((c[0] - lon[0]) / step))) for c in _tiles(index, name))
    plan = _plan(stamps, utils_epochs.SOURCES[name]['hours'])
    uniq = sorted({e for e0, e1, w in plan for e in ((e0,) if e1 is None else (e0, e1))})
    position = {e: k for k, e in enumerate(uniq)}
    first = np.array([position[e0] for e0, e1, w in plan])
    second = np.array([-1 if e1 is None else position[e1] for e0, e1, w in plan])
    w0 = np.array([1.0 if e1 is None else 1.0 - w for e0, e1, w in plan], dtype=float)
    w1 = np.array([0.0 if e1 is None else w for e0, e1, w in plan], dtype=float)
    ntime = len(plan)
    scale = -4.0 * np.pi / float(wavelength)
    count = _TABLE_COUNT
    hydro_t, wet_t, mapping_t = [], [], []
    for j0, i0 in corners:
        und = None if undulation is None else undulation[j0:j0 + n, i0:i0 + n]
        node = dask.delayed(_zenith_tile, nout=3)(index, name, uniq, j0, i0, lat[j0:j0 + n], lon[i0:i0 + n], levels, und,
                                                  first, second, w0, w1, _TABLE_H0, _TABLE_STEP, count, scale)
        hydro_t.append(da.from_delayed(node[0], shape=(ntime, n, n, count), dtype=float))
        wet_t.append(da.from_delayed(node[1], shape=(ntime, n, n, count), dtype=float))
        mapping_t.append(da.from_delayed(node[2], shape=(ntime, n, n, 2, 3), dtype=float))
    ds = xr.Dataset(
        {'hydro': (('time', 'tile', 'ty', 'tx', 'height'), da.stack(hydro_t, axis=1)),
         'wet': (('time', 'tile', 'ty', 'tx', 'height'), da.stack(wet_t, axis=1)),
         'mapping': (('time', 'tile', 'ty', 'tx', 'component', 'coef'), da.stack(mapping_t, axis=1))},
        coords={'time': pd.DatetimeIndex(stamps), 'lat': lat, 'lon': lon,
                'tile y0': ('tile', np.array([j for j, i in corners], dtype=np.int64)),
                'tile x0': ('tile', np.array([i for j, i in corners], dtype=np.int64)),
                'height': _TABLE_H0 + _TABLE_STEP * np.arange(count),
                'component': ['hydrostatic', 'wet'], 'coef': ['a', 'b', 'c']},
        attrs={'h0': _TABLE_H0, 'step': _TABLE_STEP, 'count': count, 'niell': [1e-3, *_HT_COEF], 'shell_k': 1.0})
    return ds
