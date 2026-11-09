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
import numpy as np
import xarray as xr

def get_spacing(da, coarsen=(1, 1)):
    import numpy as np
    if coarsen is None:
        coarsen = (1, 1)
    if not isinstance(coarsen, (list, tuple, np.ndarray)):
        coarsen = (coarsen, coarsen)
    dy = da.y.diff('y').item(0)
    dx = da.x.diff('x').item(0)
    if coarsen is not None:
        dy *= coarsen[0]
        dx *= coarsen[1]
    return (dy, dx)

def spacing_of(ds):
    """(dy, dx) ground metres per pixel of a burst, positive."""
    import numpy as np
    y = np.asarray(ds['y'].values, dtype=float)
    x = np.asarray(ds['x'].values, dtype=float)
    return (abs(float(y[1] - y[0])) if y.size > 1 else 1.0,
            abs(float(x[1] - x[0])) if x.size > 1 else 1.0)


def meters_to_pixels(value, spacing, minimum=1, odd=False, name='window'):
    """A GROUND SIZE in metres -> samples on THIS grid, per axis.

    Every spatial size a caller states is a distance on the ground, so the
    same call describes the same neighbourhood whatever the posting. The
    conversion happens once, at the public boundary, and everything inside
    the library keeps working in pixels.

    `minimum` is the smallest sensible count per axis; a request that rounds
    below it RAISES rather than collapsing to a degenerate window, and the
    message says what it would take on this grid. `odd` rounds up to the next
    odd count for kernels centred on a pixel.
    """
    import numpy as np
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = (value, value)
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f'{name} takes one value or two (y, x) in metres, '
                         f'got {value}')
    out = []
    for v, d, ax in zip(value, spacing, ('y', 'x')):
        v, d = float(v), abs(float(d))
        if not v > 0:
            raise ValueError(f'{name} must be positive metres, got {value}')
        n = int(round(v / d))
        if odd and n % 2 == 0:
            n += 1
        if n < minimum:
            raise ValueError(
                f'{name}={v:g} m is under {minimum} pixel(s) on this grid, '
                f'which posts {d:g} m per pixel in {ax}: ask for at least '
                f'{minimum * d:g} m.')
        out.append(n)
    return tuple(out)


def window_meters_to_pixels(window, spacing, name='window'):
    """(DS, PS) or (DSy, DSx, PSy, PSx) in metres -> (wy, wx, pey, pex) pixels.

    Two values are the DS window and the PS extent as GROUND SQUARES, so
    (400, 4000) is a 400 m box inside a 4000 m search on any posting, and the
    anisotropy of the grid is handled by the conversion rather than by the
    caller. Four values state the two axes separately, still in metres.
    """
    import numpy as np
    w = tuple(window) if isinstance(window, (list, tuple, np.ndarray)) else None
    if w is None or len(w) not in (2, 4):
        raise ValueError(
            f'{name} takes 2 values (DS, PS extent) or 4 (DSy, DSx, PSy, PSx), '
            f'in METRES; got {window!r}')
    if len(w) == 2:
        w = (w[0], w[0], w[1], w[1])
    wy, wx = meters_to_pixels((w[0], w[1]), spacing, minimum=2,
                              name=f'{name} (DS box)')
    pey, pex = meters_to_pixels((w[2], w[3]), spacing, minimum=2,
                                name=f'{name} (PS extent)')
    return wy, wx, pey, pex


def coarsen_start(da, name, spacing, grid_factor=1):
    """
    Calculate start coordinate to align coarsened grids.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    name : str
        Coordinate name to align
    spacing : int
        Coarsening spacing
    grid_factor : int, optional
        Grid factor for alignment, default is 1
        
    Returns
    -------
    int or None
        Start index for optimal alignment, or None if no good alignment found
    """
    import numpy as np
    
    # get coordinate values
    coords = da[name].values
    if len(coords) < spacing:
        print(f'_coarsen_start: Not enough points for spacing {spacing}')
        return None
        
    # calculate coordinate differences
    diffs = np.diff(coords)
    if not np.allclose(diffs, diffs[0], rtol=1e-5):
        print(f'_coarsen_start: Non-uniform spacing detected for {name}')
        return None
        
    # calculate target spacing
    target_spacing = diffs[0] * spacing * grid_factor
    
    # find best alignment point
    best_offset = None
    min_error = float('inf')
    
    for i in range(spacing):
        # get coarsened coordinates
        coarse_coords = coords[i::spacing]
        if len(coarse_coords) < 2:
            continue
            
        # calculate alignment error
        error = np.abs(coarse_coords[0] % target_spacing)
        if error < min_error:
            min_error = error
            best_offset = i
            
    if best_offset is not None:
        #print(f'_coarsen_start: {name} spacing={spacing} grid_factor={grid_factor} => {best_offset} (error={min_error:.2e})')
        return best_offset
        
    print(f'_coarsen_start: No good alignment found for {name}')
    return None

def to_dict(datas: dict[str, xr.Dataset | xr.DataArray] | None = None):
    """
    Convert a list of datasets or dictionaries of datasets to a dictionary.
    """
    if isinstance(datas, xr.Dataset):
        return {'default': datas}
    if isinstance(datas, xr.DataArray):
        return {'default': datas.to_dataset()}
    return datas

def apply(*args, **kwarg):
    """
    Apply a function to multiple datasets or dictionaries of datasets with the same keys.

    Parameters
    ----------
    *args : list of datasets or dictionaries of datasets
        The datasets to apply the function to.
    **kwarg : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    dict or dataset
        The result of applying the function to the datasets.
        If the input is a dictionary or a list of dictionaries, the result is a dictionary.
        If the input is a dataset or a list of datasets, the result is a dataset.
    
    Examples
    --------
    >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    """
    import dask

    func = kwarg.pop('func', None)
    if func is None:
        raise ValueError('`func` argument is required')
    compute = kwarg.pop('compute', False)
    #print ('compute', compute)
    add_key = kwarg.pop('add_key', False)
    if not args:
        return
    datas = [to_dict(arg) if arg is not None else None for arg in args]
    keys = list(datas[0].keys())
    if add_key:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
    else:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **kwarg) for key in keys}
    if compute:
        from .utils_dask import progress_persisted
        progress_persisted(dss := dask.persist(dss)[0], desc=f'Computing...'.ljust(25))
    # detect output type
    sample = next(iter(dss.values()))
    # multiple datasets or dictionaries
    if (isinstance(sample, (tuple, list))):
        n = len(sample)
        dicts = [{key: dss[key][i] for key in keys} for i in range(n)]
        if isinstance(args[0], dict):
            return tuple(dicts)
        return tuple(d['default'] for d in dicts)
    # single dataset or dictionary
    if isinstance(args[0], dict):
        return dss
    return dss['default']



def nearest_indices(source, target):
    """Find indices in source nearest to target."""
    # Handle descending coordinates (e.g., y going north to south)
    descending = len(source) > 1 and source[0] > source[-1]
    if descending:
        source = source[::-1]
    indices = np.searchsorted(source, target)
    indices = np.clip(indices, 0, len(source) - 1)
    # Check if previous index is closer
    prev_indices = np.clip(indices - 1, 0, len(source) - 1)
    prev_diff = np.abs(source[prev_indices] - target)
    curr_diff = np.abs(source[indices] - target)
    indices = np.where(prev_diff < curr_diff, prev_indices, indices)
    # Convert back to original order if descending
    if descending:
        indices = len(source) - 1 - indices
    return indices


def merge_tiles(tiles, offsets, out_shape, fill_dtype, nodata=np.nan):
    """
    Merge tiles into one date of an output block (the algorithm of to_dataset(); called per date by _merge_stack_for_dask).

    Defined at module level to avoid dask serialization issues with nested functions.
    When using dask distributed, closures with nested functions may not serialize correctly,
    causing random/incorrect behavior on workers.

    THE OVERLAP RULE: the tiles come in the bursts' acquisition order and each one is laid
    over the ones before it, so the MOST RECENT burst's valid value wins; its nodata
    (_nodata_of: NaN, or -1, 0, False for an integer or bool grid) is transparent and
    never hides an earlier burst's valid value; a pixel no tile covers stays nodata.

    Args:
        tiles: list of 3D arrays (1, ny, nx), one date of each tile (cut by _tile_for_dask() or a whole input block), earliest burst first
        offsets: list of (y_offset, x_offset) tuples for each tile
        out_shape: (ny, nx) output shape
        fill_dtype: output dtype, the grid's own
        nodata: the grid's nodata value, _nodata_of(fill_dtype)

    Returns:
        merged 2D numpy array
    """
    out = np.full(out_shape, nodata, dtype=fill_dtype)
    nan = isinstance(nodata, float) and np.isnan(nodata)

    for tile_3d, (y_off, x_off) in zip(tiles, offsets):
        # Tiles arrive as 3D (1, ny, nx), squeeze to 2D
        tile = tile_3d[0]

        # Tile position in output chunk
        y0, x0 = y_off, x_off
        y1, x1 = y0 + tile.shape[0], x0 + tile.shape[1]

        # Clip to output bounds
        y0c, x0c = max(0, y0), max(0, x0)
        y1c, x1c = min(out_shape[0], y1), min(out_shape[1], x1)

        if y1c > y0c and x1c > x0c:
            # Tile slice corresponding to clipped output region
            ty0, tx0 = y0c - y0, x0c - x0
            ty1, tx1 = ty0 + (y1c - y0c), tx0 + (x1c - x0c)

            # the later burst's valid values over the earlier ones; its nodata is transparent
            view = out[y0c:y1c, x0c:x1c]
            src = tile[ty0:ty1, tx0:tx1]
            np.copyto(view, src, where=~np.isnan(src) if nan else src != nodata)

    return out


def _overlap(edges, lo, hi):
    """[(chunk, start, stop within it)] of the chunks the range [lo, hi) overlaps."""
    first = int(np.searchsorted(edges, lo, 'right')) - 1
    last = int(np.searchsorted(edges, hi, 'left')) - 1
    return [(i, max(lo, edges[i]) - edges[i], min(hi, edges[i + 1]) - edges[i])
            for i in range(first, last + 1)]


def mosaic_axis(n, bursts, target, block):
    """The output block sizes of one axis of to_dataset(chunks='auto'), n px: `bursts` (start, inner block boundaries, end, blocks
    across) of each merged grid of each burst on the union's pixel index, `target` the budget tile, `block` the largest input block.
    A boundary sits where every burst covering it has a block boundary; intervals are joined up to `target`; an interval above
    max(target, 2 * block) is split at the bursts' block boundaries and edges that cut the fewest input blocks."""
    goal = max(target, 2 * block)

    def cut(p):
        # the input blocks an output boundary at p cuts
        return sum(w for s, inner, e, w in bursts if s < p < e and p not in inner)

    cand = sorted({p for s, inner, e, _ in bursts for p in (s, e, *inner) if 0 < p < n})
    sizes, acc = [], 0
    for size in np.diff([0] + [p for p in cand if cut(p) == 0] + [n]).tolist():
        if acc and acc + size > target:
            sizes.append(acc)
            acc = 0
        acc += size
    sizes.append(acc)
    edges = np.cumsum([0] + sizes).tolist()
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= goal:
            out.append(b)
            continue
        # positions at most goal apart: the candidates inside, and every goal px where none is (no burst covers there)
        pts = [a]
        for p in [q for q in cand if a < q < b] + [b]:
            while p - pts[-1] > goal:
                pts.append(pts[-1] + goal)
            pts.append(p)
        best = [(0, 0, -1)] + [None] * (len(pts) - 1)
        for i in range(1, len(pts)):
            c = 0 if i == len(pts) - 1 else cut(pts[i])
            for j in range(i - 1, -1, -1):
                if pts[i] - pts[j] > goal:
                    break
                if best[j] is not None and (best[i] is None or (best[j][0] + c, best[j][1] + 1) < best[i][:2]):
                    best[i] = (best[j][0] + c, best[j][1] + 1, j)
        chain, i = [], len(pts) - 1
        while i > 0:
            chain.append(pts[i])
            i = best[i][2]
        out.extend(chain[::-1])
    return tuple(np.diff([0] + out).tolist())


def mosaic_plan(y_list, x_list, dy, dx, chunks, yx_chunks=None):
    """The plan of to_dataset() for bursts with the coordinates y_list[i], x_list[i] on one grid of signed spacing dy, dx: the union
    axes ys, xs, every burst's first and last index on them (burst_info), the output blocks of each axis (y_sizes, x_sizes; the edges
    y_edges, x_edges) and the bursts of each block (chunk_index: (yi, xi) -> [(burst, coverage)] in acquisition order). 'auto' takes
    the blocks from the input blocks: yx_chunks[i] = the (y chunks, x chunks) of each merged grid of burst i (None: one block a burst)."""
    from collections import defaultdict
    # Find global min/max across all bursts
    # Use explicit float conversion to handle any lazy coordinate types
    y_min = min(float(np.asarray(y).min()) for y in y_list)
    y_max = max(float(np.asarray(y).max()) for y in y_list)
    x_min = min(float(np.asarray(x).min()) for x in x_list)
    x_max = max(float(np.asarray(x).max()) for x in x_list)

    # Determine first/last based on sign of spacing
    # If dy > 0 (increasing): first = min, last = max
    # If dy < 0 (decreasing): first = max, last = min
    y_first = y_min if dy > 0 else y_max
    y_last = y_max if dy > 0 else y_min
    x_first = x_min if dx > 0 else x_max
    x_last = x_max if dx > 0 else x_min

    # Build output grid using arange with signed spacing
    # Add dy/2 to last to ensure it's included (arange excludes endpoint)
    ys = np.arange(y_first, y_last + dy/2, dy)
    xs = np.arange(x_first, x_last + dx/2, dx)

    # Each burst on the union: its coordinates, its first index and its end index (exclusive)
    # Index formula: idx = (coord - y_first) / dy; works for both increasing and decreasing coords
    burst_info, starts, ends = [], [], []
    for y, x in zip(y_list, x_list):
        # Ensure coordinates are concrete numpy arrays (not dask or other lazy types)
        # This is important when data comes from delayed computations like dissolve()
        burst_ys = np.asarray(y, dtype=np.float64)
        burst_xs = np.asarray(x, dtype=np.float64)
        burst_info.append({'y_coords': burst_ys, 'x_coords': burst_xs})
        starts.append((int(round((burst_ys[0] - y_first) / dy)), int(round((burst_xs[0] - x_first) / dx))))
        ends.append((int(round((burst_ys[-1] - y_first) / dy)) + 1, int(round((burst_xs[-1] - x_first) / dx)) + 1))

    # The output blocks of each axis. 'auto': each axis from the input blocks (mosaic_axis) and the budget tile of
    # rechunk2d; an int or a (y, x) pair: uniform blocks
    if chunks == 'auto':
        from .utils_dask import rechunk2d, warn_floor
        # to_dataset creates a new output grid: the budget tile of the union at 16 bytes per pixel
        floors = []
        optimal = rechunk2d((ys.size, xs.size), element_bytes=16, floors=floors)
        warn_floor('to_dataset', floors)
        grids = yx_chunks if yx_chunks is not None else [[((len(y),), (len(x),))] for y, x in zip(y_list, x_list)]
        by, bx = [], []
        for (iy0, ix0), gs in zip(starts, grids):
            for cy, cx in gs:
                ey, ex = np.cumsum((0,) + tuple(cy)).tolist(), np.cumsum((0,) + tuple(cx)).tolist()
                by.append((iy0, {iy0 + e for e in ey[1:-1]}, iy0 + ey[-1], len(cx)))
                bx.append((ix0, {ix0 + e for e in ex[1:-1]}, ix0 + ex[-1], len(cy)))
        y_sizes = mosaic_axis(ys.size, by, optimal['y'], max(max(cy) for gs in grids for cy, _ in gs))
        x_sizes = mosaic_axis(xs.size, bx, optimal['x'], max(max(cx) for gs in grids for _, cx in gs))
    else:
        if isinstance(chunks, (int, np.integer)):
            # Single int: use same size for both dimensions
            chunks = (chunks, chunks)
        elif not (isinstance(chunks, (tuple, list)) and len(chunks) == 2):
            raise ValueError(
                f"chunks must be 'auto', int, or 2-tuple (y, x), got {type(chunks).__name__}: {chunks}. "
                "Note: 3D chunks are not supported; stack dimension is always processed per-slice."
            )
        # Tuple/list of (y_chunk, x_chunk): uniform blocks, the last one the remainder
        cy, cx = min(int(chunks[0]), ys.size), min(int(chunks[1]), xs.size)
        y_sizes = tuple([cy] * (ys.size // cy) + ([ys.size % cy] if ys.size % cy else []))
        x_sizes = tuple([cx] * (xs.size // cx) + ([xs.size % cx] if xs.size % cx else []))

    # Block edges on each axis: block i is [edges[i], edges[i + 1])
    y_edges = np.cumsum((0,) + y_sizes).tolist()
    x_edges = np.cumsum((0,) + x_sizes).tolist()
    n_y_chunks, n_x_chunks = len(y_sizes), len(x_sizes)

    # chunk_index[(yi, xi)] = list of (burst_idx, coverage), each burst where it overlaps the block
    chunk_index = defaultdict(list)
    for burst_idx, ((iy0, ix0), (iy1, ix1)) in enumerate(zip(starts, ends)):
        # the blocks the burst overlaps: the one holding its first index up to the one before its end
        y_range = range(max(0, int(np.searchsorted(y_edges, iy0, 'right')) - 1),
                        min(n_y_chunks, int(np.searchsorted(y_edges, iy1, 'left'))))
        x_range = range(max(0, int(np.searchsorted(x_edges, ix0, 'right')) - 1),
                        min(n_x_chunks, int(np.searchsorted(x_edges, ix1, 'left'))))
        for yi in y_range:
            for xi in x_range:
                # overlap in array index space (pixels)
                coverage = (max(0, min(y_edges[yi + 1], iy1) - max(y_edges[yi], iy0)) *
                            max(0, min(x_edges[xi + 1], ix1) - max(x_edges[xi], ix0)))
                if coverage > 0:
                    chunk_index[(yi, xi)].append((burst_idx, coverage))

    # Each chunk's bursts in acquisition order (burst_idx follows it): the most recent
    # is laid last and wins the overlaps
    for key in chunk_index:
        chunk_index[key].sort()

    return {'ys': ys, 'xs': xs, 'dy': dy, 'dx': dx, 'y_first': y_first, 'x_first': x_first,
            'n_y_chunks': n_y_chunks, 'n_x_chunks': n_x_chunks, 'burst_info': burst_info,
            'chunk_index': chunk_index, 'y_sizes': y_sizes, 'x_sizes': x_sizes,
            'y_edges': y_edges, 'x_edges': x_edges}


def mosaic_tiles(plan, yi, xi, yx_edges):
    """The tiles of output chunk (yi, xi) of mosaic_plan(), as (out_shape, tiles): one tile per input block it
    overlaps, (burst, iy, py0, py1, ix, px0, px1, off_y, off_x) -- the burst, the block index and the cut within
    the block, and the tile's offset in the chunk. yx_edges[burst] = (y edges, x edges) of the burst's blocks."""
    ys, xs, dy, dx = plan['ys'], plan['xs'], plan['dy'], plan['dx']
    y_first, x_first = plan['y_first'], plan['x_first']
    y_edges, x_edges = plan['y_edges'], plan['x_edges']
    yb0, yb1 = y_edges[yi], y_edges[yi + 1]
    xb0, xb1 = x_edges[xi], x_edges[xi + 1]
    out_shape = (yb1 - yb0, xb1 - xb0)

    # Output chunk coordinates
    out_ys = ys[yb0:yb1]
    out_xs = xs[xb0:xb1]

    # O(1) lookup of overlapping bursts via spatial index
    overlapping = plan['chunk_index'].get((yi, xi), [])

    # Output chunk coordinate bounds (with small tolerance for floating point)
    # Use tiny tolerance (1e-6 * spacing) to handle floating point precision
    # Apply tolerance to expand bounds (not shrink), regardless of coord direction
    tol_y = abs(dy) * 1e-6
    tol_x = abs(dx) * 1e-6
    y_lo = min(out_ys[0], out_ys[-1]) - tol_y
    y_hi = max(out_ys[0], out_ys[-1]) + tol_y
    x_lo = min(out_xs[0], out_xs[-1]) - tol_x
    x_hi = max(out_xs[0], out_xs[-1]) + tol_x

    tiles = []
    for burst_idx, _ in overlapping:
        info = plan['burst_info'][burst_idx]
        burst_ys = info['y_coords']
        burst_xs = info['x_coords']

        # Find burst indices that fall within output chunk bounds
        mask_y = (burst_ys >= y_lo) & (burst_ys <= y_hi)
        mask_x = (burst_xs >= x_lo) & (burst_xs <= x_hi)
        idx_y = np.where(mask_y)[0]
        idx_x = np.where(mask_x)[0]

        if len(idx_y) > 0 and len(idx_x) > 0:
            by0, by1 = int(idx_y[0]), int(idx_y[-1]) + 1
            bx0, bx1 = int(idx_x[0]), int(idx_x[-1]) + 1

            # Compute tile offset within this output chunk
            # Tile's first coord -> global array index -> offset in chunk
            tile_y0_coord = burst_ys[by0]
            tile_x0_coord = burst_xs[bx0]
            # Global array index of tile start
            tile_global_yi = int(round((tile_y0_coord - y_first) / dy))
            tile_global_xi = int(round((tile_x0_coord - x_first) / dx))
            # Offset within this chunk (yb0, xb0 is chunk start in global)
            y_off = tile_global_yi - yb0
            x_off = tile_global_xi - xb0

            # Validate offset is within reasonable bounds
            # (should be within chunk ± 1 for floating point tolerance)
            tile_h, tile_w = by1 - by0, bx1 - bx0
            if y_off < -1 or y_off + tile_h > out_shape[0] + 1:
                continue  # Skip misaligned tiles
            if x_off < -1 or x_off + tile_w > out_shape[1] + 1:
                continue  # Skip misaligned tiles

            ey, ex = yx_edges[burst_idx]
            for iy, py0, py1 in _overlap(ey, by0, by1):
                for ix, px0, px1 in _overlap(ex, bx0, bx1):
                    tiles.append((burst_idx, iy, py0, py1, ix, px0, px1,
                                  y_off + ey[iy] + py0 - by0, x_off + ex[ix] + px0 - bx0))
    return out_shape, tiles


def burst_overlaps(extents):
    """{burst index: the other bursts whose boxes overlap it} for the extents (ymin, ymax, xmin, xmax) of the bursts."""
    from shapely import box, STRtree
    burst_boxes = [box(xmin, ymin, xmax, ymax) for ymin, ymax, xmin, xmax in extents]
    tree = STRtree(burst_boxes)
    return {burst_idx: tuple(int(idx) for idx in tree.query(burst_boxes[burst_idx]) if idx != burst_idx)
            for burst_idx in range(len(burst_boxes))}


def format_date(val):
    """Date text of a file name: YYYYMMDD for a date-like value, str(val) for a value that is not a date."""
    import pandas as pd
    try:
        ts = pd.to_datetime(val)
        if pd.isna(ts):
            return str(val)
        return ts.strftime('%Y%m%d')
    except Exception:
        return str(val)


def overlay_height(oy, ox, elevations):
    """Elevation at the overlay nodes (oy, ox): each burst's nearest pixel inside it, burst by burst; the first burst with a value gives it.

    elevations: list of DataArrays (y, x) of the bursts' elevation.
    """
    z = np.full((len(oy), len(ox)), np.nan, dtype=np.float32)
    for ele in elevations:
        ys, xs = ele.y.values, ele.x.values
        iy = np.nonzero((oy >= min(ys[0], ys[-1])) & (oy <= max(ys[0], ys[-1])))[0]
        ix = np.nonzero((ox >= min(xs[0], xs[-1])) & (ox <= max(xs[0], xs[-1])))[0]
        if not len(iy) or not len(ix):
            continue
        values = np.asarray(ele.isel(y=nearest_indices(ys, oy[iy]), x=nearest_indices(xs, ox[ix])).values,
                            dtype=np.float32)
        block = z[np.ix_(iy, ix)]
        z[np.ix_(iy, ix)] = np.where(np.isnan(block), values, block)
    return z
