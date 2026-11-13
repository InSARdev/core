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
"""Atmosphere models for ERA5, VMF3 and IONEX: the opened model holds its node tables over the ellipsoidal height, lazily.

The toolkit reads no pixel: OpenedModel.phase() returns the tables of every opened date, and the caller evaluates its
own pixels from them (bilinear in position, linear in height). The public static helpers mapping_cf, mapping_form,
height_term, pierce_point and iono_mapping are the formulas of the pixel, for the callers' kernels (core's Stack, ecef), not for notebooks.
"""
import math

import numpy as np
import pandas as pd
from numba import njit

# the IONEX control mesh spacing of the removed pixel evaluator, metres (kept for the callers of the public helpers)
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


@njit(nogil=True, cache=True, error_model='numpy')
def mapping_cf(a, b, c, s):
    """The dimensionless mapping factor of the coefficients a, b, c at s = cos of the incidence (continued fraction).

    Parameters
    ----------
    a, b, c : float - The dimensionless mapping coefficients.
    s : float - cos of the incidence, dimensionless.

    Returns
    -------
    float - The mapping factor; with a = b = c = 0 it is 1 / s.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import mapping_cf
    >>> round(mapping_cf(0.0, 0.0, 0.0, 0.5), 6)
    2.0
    """
    if a == 0.0 and b == 0.0 and c == 0.0:
        return 1.0 / s
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (s + a / (s + b / (s + c)))


@njit(nogil=True, cache=True, error_model='numpy')
def mapping_form(a, b, c, K, s):
    """The mapping factor in one division, with the constant K of the coefficients (equal to mapping_cf with K = (q + b + a q) / (q + b),
    q = 1 + c).

    Parameters
    ----------
    a, b, c, K : float - The coefficients and the constant K.
    s : float - cos of the incidence, dimensionless.

    Returns
    -------
    float - The dimensionless mapping factor.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import mapping_form
    >>> round(mapping_form(0.0, 0.0, 0.0, 1.0, 0.5), 6)
    2.0
    """
    r = s + c
    num = s * r + b
    return K * num / (s * num + a * r)


@njit(nogil=True, cache=True, error_model='numpy')
def height_term(v, a, b, c, h, z):
    """The Niell height term of the hydrostatic mapping: v h (1 / cos z - mapping_cf(a, b, c, cos z)); v = 0 gives 0.

    Parameters
    ----------
    v, a, b, c : float - The height term coefficients (the model's height_term; v = 0: no term).
    h : float - The ellipsoidal height, metres.
    z : float - The incidence, radians.

    Returns
    -------
    float - The height term, dimensionless, added to the hydrostatic mapping factor.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import height_term
    >>> round(height_term(2.53e-5, 5.49e-3, 1.14e-3, 1.14e-3, 1000.0, 0.5), 9)
    4.6779e-05
    """
    if v == 0.0:
        return 0.0
    cz = math.cos(z)
    return v * h * (1.0 / cz - mapping_cf(a, b, c, cz))


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
def pierce_point(qy, qx, h, z, look, shell, geocentric, ctrl, ky, kx, S0, S1, lat_c, lon_c):
    """The IONEX single-layer pierce point of one pixel and its effective incidence (TN-200 step 2, the sine law). The mapping
    factor of the pixel is 1 / cos of the effective incidence.

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
    >>> from insardev_toolkit.utils_atmosphere import pierce_point
    >>> _ = pierce_point(47.0, 13.0, 0.0, 0.6, 77.0, 6821e3, True, False, 0, 0, 1.0, 1.0, np.zeros((1, 1)), np.zeros((1, 1)))
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


@njit(nogil=True, cache=True, error_model='numpy')
def iono_mapping(zeff):
    """The single-layer mapping factor of the effective incidence zeff of pierce_point: 1 / cos(zeff).

    Parameters
    ----------
    zeff : float - The effective incidence at the shell, radians (the third output of pierce_point).

    Returns
    -------
    float - The mapping factor, dimensionless.

    Examples
    --------
    >>> from insardev_toolkit.utils_atmosphere import iono_mapping
    >>> round(iono_mapping(0.0), 6), round(iono_mapping(1.0471975511965976), 6)
    (1.0, 2.0)
    """
    return 1.0 / math.cos(zeff)


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


_A_WGS = 6378137.0                                                # m, WGS84 semi-major axis
_E2_WGS = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)      # WGS84 first eccentricity squared


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


class OpenedModel:
    """An atmosphere model of a folder, opened for a stack by AtmosphereModel.open.

    Attributes
    ----------
    name : str - 'ERA5', 'VMF3' or 'IONEX'.
    terms : xarray.Dataset - The lazy tables of the time union of the bursts. ERA5 and VMF3: hydro and wet (time, tile, ty, tx,
        height) in phase units, mapping (time, tile, ty, tx, component, coef), height_term (height_coef), one dask chunk per
        tile, and the attributes h0, step, count and niell. IONEX: tec (time, lat, lon) in phase units and shell (no dims),
        R + H in metres, and no attributes; no wet term, mapping or height axis.
    times : pandas.DatetimeIndex - The times of the terms, UTC.
    crs : int, str, rasterio CRS or None - The output CRS; None for lon and lat.
    x, y : numpy.ndarray - The nodes, shape (ny, nx): in the output CRS (ERA5, VMF3), in longitude and latitude (IONEX).
    """

    def __init__(self, name: str, terms, crs):
        self.name = name
        self.terms = terms
        self.times = pd.DatetimeIndex(terms['time'].values)
        self.crs = crs
        if 'tec' in terms.data_vars:
            # IONEX: one layer on the lattice in latitude and longitude (the published single-layer model); no tiles, no mapping
            LON, LAT = np.meshgrid(np.asarray(terms['lon'].values, dtype=np.float64), np.asarray(terms['lat'].values, dtype=np.float64))
            self.x, self.y = LON, LAT
            self._arrays = None
            return
        lat, lon, y0, x0 = _node_grid(terms)
        LON, LAT = np.meshgrid(lon, lat)
        self.x, self.y = _nodes_xy(LON, LAT, crs)
        # the tables stay lazy: each tile is one dask chunk, computed only when a caller's block needs it
        self._arrays = [_tile_chunks(terms[v].data) for v in ('hydro', 'wet', 'mapping')]
        ty, tx = self._arrays[0].shape[2], self._arrays[0].shape[3]
        if (y0 + ty > lat.size).any() or (x0 + tx > lon.size).any():
            raise ValueError(f'ERROR: {name} the terms do not cover the model nodes')
        # the nodes of each tile in the output CRS, (tile, ty, tx)
        rows = y0[:, None, None] + np.arange(ty)[None, :, None]
        cols = x0[:, None, None] + np.arange(tx)[None, None, :]
        self._xy = (self.x[rows, cols], self.y[rows, cols])

    def phase(self):
        """The tables of every opened date as one lazy xarray.Dataset; this call reads and computes nothing.

        Returns
        -------
        xarray.Dataset - hydro and wet (time, tile, ty, tx, height): the zenith delays in phase units (-4 pi / wavelength
        times the delay, radians); mapping (time, tile, ty, tx, component, coef): the coefficients a, b, c and K of the
        hydrostatic and wet mapping; height_term (height_coef): v, a, b, c of the height term (v = 0: no term). Coordinates:
        time, lat and lon (the model's nodes), height (m), tile y0 and tile x0 (lattice row and column of each tile's first
        node), and x, y (tile, ty, tx): the nodes in the output CRS (lon and lat when the CRS is None). One dask chunk per
        model tile and group of dates (the dates that share their bracketing model epochs, as zenith_tiles builds it). The caller
        selects its dates and tiles by label.

        IONEX has one layer: tec (time, lat, lon), the zenith TEC phase in radians (the sign of the model), and shell (no dims), R + H
        in metres from the day files; no wet term, mapping or height axis. Its tables are the same lazy Dataset, on the lattice.
        """
        import dask.array as da
        import xarray as xr
        terms = self.terms
        if self._arrays is None:
            return terms
        mapping = da.map_blocks(_mapping_coefs, self._arrays[2], dtype=float,
                                chunks=self._arrays[2].chunks[:-1] + ((4,),))
        return xr.Dataset(
            {'hydro': (('time', 'tile', 'ty', 'tx', 'height'), self._arrays[0]),
             'wet': (('time', 'tile', 'ty', 'tx', 'height'), self._arrays[1]),
             'mapping': (('time', 'tile', 'ty', 'tx', 'component', 'coef'), mapping),
             'height_term': (('height_coef',), np.asarray(terms.attrs['niell'], dtype=np.float64))},
            coords={'time': terms['time'].values, 'lat': terms['lat'].values, 'lon': terms['lon'].values,
                    'height': terms['height'].values, 'tile y0': ('tile', terms['tile y0'].values),
                    'tile x0': ('tile', terms['tile x0'].values),
                    'x': (('tile', 'ty', 'tx'), self._xy[0]), 'y': (('tile', 'ty', 'tx'), self._xy[1]),
                    'component': ['hydrostatic', 'wet'], 'coef': ['a', 'b', 'c', 'K'],
                    'height_coef': ['v', 'a', 'b', 'c']},
            attrs={k: v for k, v in terms.attrs.items() if k != 'niell'})


class AtmosphereModel:
    """Mixin of the atmosphere model classes ERA5, VMF3 and IONEX; open() opens a model folder once per stack."""

    def open(self, folder: str, times: list, crs: int | str | None, wavelength: float, factor: float = 1.0,
             heights: tuple[float, float] = (-500.0, 9000.0)):
        """Open the model of a folder once for a stack: lazy tables over the time union of the bursts, and the nodes (in the
        output CRS; IONEX in latitude and longitude). No pixel is read: the caller gives its height range only.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields (the folder that holds its .vrt index).
        times : list of str or pandas.Timestamp - The acquisition times of all bursts of the stack, UTC (their union).
        crs : int, str or None - The output CRS of the stack; None for lon and lat.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor (IONEX only), dimensionless.
        heights : tuple of float - (lo, hi), the ellipsoidal heights (metres) of the caller's pixels (its ele actual_range).
            The zenith tables (ERA5, VMF3) cover the rows of that range on the global grid of the model's table step,
            one row beyond each end; the default (-500, 9000) is the whole domain. IONEX ignores it. Outside -500..9000 m
            raises one ERROR line.

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
