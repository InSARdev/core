# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
The UTM output grid of the preprocessors' transforms for a WGS84 bbox: the automatic UTM zone, the projection, the grid
edges (bbox_grid_bounds, used by insardev_pygmtsar's transforms) and the lon/lat region that grid covers
(transform_region, used by the downloaders to crop what the transform processes).
"""
import numpy as np


def get_utm_epsg(lat, lon):
    """Get UTM EPSG code for given lat/lon coordinates."""
    zone_num = int((lon + 180) // 6) + 1
    return 32600 + zone_num if lat >= 0 else 32700 + zone_num


def proj(ys, xs, to_epsg, from_epsg):
    """Project coordinates between EPSG codes.

    Parameters
    ----------
    ys : array-like
        Y coordinates (latitude for EPSG:4326)
    xs : array-like
        X coordinates (longitude for EPSG:4326)
    to_epsg : int
        Target EPSG code
    from_epsg : int
        Source EPSG code

    Returns
    -------
    tuple
        (ys_new, xs_new) in target CRS
    """
    from pyproj import CRS, Transformer
    from_crs = CRS.from_epsg(from_epsg)
    to_crs = CRS.from_epsg(to_epsg)
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    xs_new, ys_new = transformer.transform(xs, ys)
    del transformer, from_crs, to_crs
    return ys_new, xs_new


# the points projected on each edge of a bbox (bbox_grid_bounds): the extreme of a curved edge between two of them
# is missed by at most 0.06 mm for a bbox up to 1 degree wide and 2.7 mm up to 6 degrees (UTM, measured)
BBOX_EDGE_POINTS = 1025


def bbox_grid_bounds(bbox, epsg, dy, dx):
    """The grid edges (y_min, y_max, x_min, x_max) in EPSG epsg that cover a WGS84 bbox [lon_min, lat_min, lon_max,
    lat_max]: its projected extent, snapped outward to multiples of the spacing (dy, dx). The four EDGES are projected
    (BBOX_EDGE_POINTS points each), not two corners: a lon/lat rectangle is not a rectangle in the output CRS, and its
    extreme x/y can lie at any corner or inside a curved edge (a parallel across a UTM central meridian). The S1 and
    NISAR transforms crop their grid to these edges."""
    t = np.linspace(0.0, 1.0, BBOX_EDGE_POINTS)
    lon = np.r_[bbox[0] + (bbox[2] - bbox[0]) * t, np.full(t.size, bbox[2]),
                bbox[2] - (bbox[2] - bbox[0]) * t, np.full(t.size, bbox[0])]
    lat = np.r_[np.full(t.size, bbox[1]), bbox[1] + (bbox[3] - bbox[1]) * t,
                np.full(t.size, bbox[3]), bbox[3] - (bbox[3] - bbox[1]) * t]
    ys, xs = proj(lat, lon, from_epsg=4326, to_epsg=epsg)
    return (float(dy * np.floor(np.min(ys) / dy)), float(dy * np.ceil(np.max(ys) / dy)),
            float(dx * np.floor(np.min(xs) / dx)), float(dx * np.ceil(np.max(xs) / dx)))


def transform_region(bbox, epsgs, margin=100.0):
    """
    The lon/lat envelope of the output grid a transform makes for a WGS84 bbox, in every given UTM zone.

    For each EPSG code the grid rectangle is bbox_grid_bounds(bbox, epsg, 1.0, 1.0), the bbox's projected extent
    snapped outward to whole metres, expanded by `margin` metres on every side. A transform with the spacing d snaps
    the same extent outward to multiples of d, which moves each edge less than d beyond the 1 m edge: the default
    100 m covers that snapping for any spacing up to 100 m. The four edges of each rectangle are sampled at
    BBOX_EDGE_POINTS points each and projected to EPSG:4326 (a UTM rectangle is not a lon/lat rectangle); the
    envelope is the minimum and maximum of all of them.

    Parameters
    ----------
    bbox : tuple of float
        (west, south, east, north) in WGS84 degrees.
    epsgs : iterable of int
        The EPSG codes of the grids (such as the UTM zones get_utm_epsg gives); at least one.
    margin : float, optional
        Metres added on every side of each grid rectangle. Default 100.

    Returns
    -------
    tuple of float
        (west, south, east, north) in WGS84 degrees, holding every grid rectangle.

    Raises
    ------
    ValueError
        If no EPSG code is given.

    Examples
    --------
    >>> transform_region((-72.40, -40.65, -72.00, -40.30), {32718, 32719})
    """
    epsgs = sorted({int(epsg) for epsg in epsgs})
    if not epsgs:
        raise ValueError('transform_region needs at least one EPSG code')
    t = np.linspace(0.0, 1.0, BBOX_EDGE_POINTS)
    lons, lats = [], []
    for epsg in epsgs:
        y_min, y_max, x_min, x_max = bbox_grid_bounds(bbox, epsg, 1.0, 1.0)
        y_min, y_max, x_min, x_max = y_min - margin, y_max + margin, x_min - margin, x_max + margin
        xs = np.r_[x_min + (x_max - x_min) * t, np.full(t.size, x_max),
                   x_max - (x_max - x_min) * t, np.full(t.size, x_min)]
        ys = np.r_[np.full(t.size, y_min), y_min + (y_max - y_min) * t,
                   np.full(t.size, y_max), y_max - (y_max - y_min) * t]
        lat, lon = proj(ys, xs, to_epsg=4326, from_epsg=epsg)
        lons.append(np.asarray(lon))
        lats.append(np.asarray(lat))
    lon, lat = np.concatenate(lons), np.concatenate(lats)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())
