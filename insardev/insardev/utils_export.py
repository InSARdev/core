# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""Export helpers of the batch classes: GeoPackage blocks, GeoJSON pixel rectangles and their precision."""
import datetime
import os
import re
import numpy as np

_ISO_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')


def _tre_name(name):
    """
    `d` + YYYYMMDD for a column named by an ISO date (YYYY-MM-DD, a real calendar date); any other name is kept as it is.

    Examples
    --------
    >>> _tre_name('2018-10-17')
    'd20181017'
    >>> _tre_name('2018-13-45')
    '2018-13-45'
    >>> _tre_name('velocity')
    'velocity'
    """
    if isinstance(name, str) and _ISO_DATE.fullmatch(name):
        try:
            datetime.date.fromisoformat(name)
        except ValueError:
            return name
        return 'd' + name.replace('-', '')
    return name


def _table_to_geopackage(table, filename: str, crs: str | int | None = 4326, layer: str | None = None,
                         overwrite: bool = True) -> None:
    """
    Write a table of points (a GeoDataFrame, or a lazy one with compute()) to a GeoPackage as ONE layer.

    The columns named by an ISO date become `d` + YYYYMMDD, the TRE naming; every other column keeps its name. The points are
    reprojected to crs (None keeps the table's own CRS). A lazy table is computed here, before the write. The `.gpkg` extension is
    added when missing; the layer is named after the file unless layer is given. With overwrite an existing file is removed first.

    Parameters
    ----------
    table : GeoDataFrame | dask-geopandas GeoDataFrame
        The points, one row per point, with the date columns and the geometry.
    filename : str
        The output GeoPackage.
    crs : str | int | None, optional
        The CRS of the written points. Default 4326.
    layer : str, optional
        The layer name. Default: the file's stem.
    overwrite : bool, optional
        Replace an existing file. Default True. When False a new layer is added to the file, and an existing layer of the
        same name gets the rows appended.

    Returns
    -------
    None

    Examples
    --------
    >>> _table_to_geopackage(table, 'dates.gpkg', crs=4326)
    """
    names = {c: _tre_name(c) for c in table.columns}
    if hasattr(table, 'compute'):
        table = table.compute()
    if not filename.lower().endswith('.gpkg'):
        filename = filename + '.gpkg'
    if layer is None:
        layer = os.path.splitext(os.path.basename(filename))[0]
    table = table.rename(columns=names)
    if crs is not None:
        table = table.to_crs(crs)
    if overwrite and os.path.exists(filename):
        os.remove(filename)
    table.to_file(filename, layer=layer, driver='GPKG', mode='w' if overwrite else 'a')


def gpkg_block(blocks, yv, xv, decimals):
    """One spatial block reduced to its FINITE pixels, where it lives.

    `blocks` are the block's OWN arrays -- nothing of the scene's graph
    travels with them -- so what comes back is the sparse points rather
    than the raster they came from.
    """
    cols, keep = {}, None
    for name, v in blocks.items():
        v = np.asarray(v)
        if v.ndim == 2:
            cols[name] = v
        else:
            for i in range(v.shape[0]):
                cols[f'{name}#{i}'] = v[i]
    for v in cols.values():
        if np.issubdtype(v.dtype, np.floating):
            m = np.isfinite(v)
        elif np.issubdtype(v.dtype, np.complexfloating):
            m = np.isfinite(v.real) & np.isfinite(v.imag)
        else:
            continue
        keep = m if keep is None else (keep | m)
    if keep is None:
        keep = np.ones((len(yv), len(xv)), bool)
    if not keep.any():
        return None
    yy, xx = np.nonzero(keep)
    out = {'y': np.asarray(yv)[yy], 'x': np.asarray(xv)[xx]}
    for k, v in cols.items():
        v = v[yy, xx]
        if np.issubdtype(v.dtype, np.complexfloating):
            # A GEOPACKAGE HAS NO COMPLEX COLUMN, and `real`/`imag` are
            # the FIT'S basis, not a quantity: the annual term is
            # `real*cos(2 pi t) + imag*sin(2 pi t)`, so neither half means
            # anything alone. AMPLITUDE AND PHASE are what the same
            # quantity is called everywhere it is read -- amplitude is the
            # size of the seasonal swing, in whatever unit the variable
            # carries, and phase is WHEN it peaks. Phase costs one column
            # and is what makes the pair lossless: amplitude alone cannot
            # rebuild the model, nor say whether two points peak in the
            # same season, which is what separates a real seasonal signal
            # from a fitting artefact. Degrees, as the products this is
            # compared against report it.
            # ROUND IN FLOAT64, whatever came in. A float32 rounded to
            # 6 decimals cannot HOLD 6 decimals -- the nearest float32 to
            # -12.294427 is -12.29442691802978..., and the GeoPackage
            # column is an 8-byte REAL, so every reader displays that
            # full tail as if nothing had been rounded at all.
            out[f'{k}_amp'] = np.round(np.abs(v).astype(np.float64),
                                       decimals)
            out[f'{k}_phase'] = np.round(
                np.degrees(np.angle(v)).astype(np.float64),
                min(decimals, 3))
        elif np.issubdtype(v.dtype, np.floating):
            out[k] = np.round(v.astype(np.float64), decimals)
        else:
            out[k] = v
    return out


def pixel_rectangles(x, y, half_x, half_y, dtype=None):
    """Pixel rectangles about the centres (x, y): corners (m, 4, 2), counter-clockwise from the lower left.

    dtype: the common dtype of the frame the centres come from (its rows hold it), None keeps x and y as they are.
    """
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)
    return np.stack([
        np.stack([x - half_x, y - half_y], axis=-1),
        np.stack([x + half_x, y - half_y], axis=-1),
        np.stack([x + half_x, y + half_y], axis=-1),
        np.stack([x - half_x, y + half_y], axis=-1),
    ], axis=1)


def rounded(values, decimals):
    """Values as float, rounded to decimals (the GeoJSON properties of BatchCore.to_geojson)."""
    return values.astype(float).round(decimals)


def snap_geometry(geometry, grid_size=1e-6):
    """Geometries snapped to a grid of grid_size (the GeoJSON coordinates)."""
    import shapely
    return shapely.set_precision(geometry, grid_size=grid_size)
