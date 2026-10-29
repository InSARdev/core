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
from __future__ import annotations
from .BatchCore import BatchCore, _SCALARS
import numpy as np
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect


def _ref_height_stored(ds) -> float | None:
    """The stack's flat-earth reference height from `ref_height`, signed (a height below sea level stays negative).

    Every preprocessor stores ONE value on every date: the reference height in flat-earth mode
    (remove_topo_phase=False), where the unwrapped grids give an absolute height, and NaN in DEM (InSAR) mode,
    where they give a residual height -- NaN reads as 0.0, the value the elevation arithmetic always used there.
    Older stores carry 0.0 in DEM mode, and older ecef flat-mode stores 0 on the reference date and the height on
    the repeat dates: the value of largest magnitude is the height in every case. None when the Dataset stores no
    ref_height at all: callers go through _ref_heights(), which warns.
    """
    if 'ref_height' in ds:
        v = np.asarray(ds['ref_height'].values, dtype=np.float64).ravel()
    elif ds.attrs.get('ref_height') is not None:
        v = np.asarray([ds.attrs['ref_height'] or 0.0], dtype=np.float64)
    else:
        return None
    v = v[np.isfinite(v)]
    return float(v[np.argmax(np.abs(v))]) if v.size else 0.0


def _computed(ds, names) -> dict:
    """{name: numpy values} of a burst Dataset's non-grid variables, ONE compute for the lazy ones."""
    import dask
    vals, = dask.compute({n: ds[n].data for n in names})
    return {n: np.asarray(v) for n, v in vals.items()}


def _ref_mode(values) -> tuple:
    """(reference height, DEM mode) from a burst's ref_height values, the rule of _ref_height_stored(): NaN on
    every date is DEM mode (height 0.0), else the finite value of largest magnitude is the flat-earth height."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return (float(v[np.argmax(np.abs(v))]), False) if v.size else (0.0, True)


def _crs_wkt(ds) -> str:
    """The CRS of a burst Dataset or DataArray as WKT."""
    return ds.rio.crs.to_wkt()


def _ele_midpoint(ds, key, where) -> float | None:
    """The midpoint of the `ele` actual_range ATTRIBUTE (never the raster), None without it."""
    if not isinstance(ds, xr.Dataset) or 'ele' not in ds.variables:
        return None
    rng = ds['ele'].attrs.get('actual_range')
    if rng is None or len(rng) != 2 or not np.all(np.isfinite(np.asarray(rng, dtype=np.float64))):
        return None
    return 0.5 * (float(rng[0]) + float(rng[1]))


def _warn_no_actual_range(where, keys) -> None:
    """ONE WARNING per call naming the bursts of an old store that carry no `ele` actual_range (the terrain height
    is taken as 0): the callers of _stack_geometry() collect the bursts and call this once after their loop."""
    if keys:
        print(f"WARNING: {where}: burst(s) {', '.join(str(k) for k in keys)} carry no ele actual_range: "
              f"the terrain height is taken as 0.", flush=True)


def _stack_geometry(ds, key, where, action, no_range) -> tuple:
    """THE GEOMETRY OF ONE BURST FROM THE STACK (or its transform() view): the per-date variables and the `ele`
    actual_range attribute, never a raster. Returns (burst model, b_h (date, 2), b_v (date, 2), dates).

    The REFERENCE date's numbers (BPR 0, N55): never a mean over the dates, never a pair's first date. Each date's
    baseline triple is its own baseline vector to the reference satellite. A stack without its reference date
    raises (N137): the geometry has no date to be the reference.

    `no_range` is the caller's list: a burst without an `ele` actual_range is appended to it (the terrain height
    is taken as 0) and the caller warns ONCE with _warn_no_actual_range() after its loop."""
    from . import utils_geometry as ug
    if not isinstance(ds, xr.Dataset) or 'date' not in ds.dims:
        raise TypeError(f'ERROR: {where} needs the stack: burst {key} carries no per-date geometry. {action}')
    names = ug.SCALARS + ('BPR', 'ref_height')
    missing = [n for n in names if n not in ds.variables]
    if missing:
        raise KeyError(f"ERROR: {where}: burst {key} carries no {', '.join(missing)}.")
    if not all(n in ds.variables for n in ug.TRIPLES):
        raise ValueError(f'ERROR: {where}: burst {key}: the store predates the baseline triples. '
                         f'Re-create it with transform().')
    values = _computed(ds, names + ug.TRIPLES)
    bpr = values['BPR'].astype(np.float64).ravel()
    zero = np.flatnonzero(bpr == 0.0)
    if not zero.size:
        raise ValueError(f'ERROR: {where}: burst {key} has no reference date (BPR 0) in the stack. '
                         f'Keep the reference date when selecting dates.')
    iref = int(zero[0])
    h_ref, dem_mode = _ref_mode(values['ref_height'])
    h_attr = _ele_midpoint(ds, key, where)
    if h_attr is None:
        no_range.append(key)
        h_attr = 0.0
    geo = ug.burst_geometry({n: values[n].ravel()[iref] for n in ug.SCALARS}, _crs_wkt(ds), h_ref, dem_mode, h_attr)
    bh, bv = ug.date_triples({n: values[n].ravel() for n in ug.TRIPLES})
    return geo, bh, bv, np.asarray(ds['date'].values)


def _pair_geometry(ds, key, where, h_attr) -> tuple:
    """THE GEOMETRY OF ONE BURST FROM THE PHASE ITSELF, the warned workaround without the stack: each pair's own
    numbers (its first date's, as pairs() carries them), the crop polygon and the CRS, the pair triple DIFFERENCE
    pairs() gives, and the terrain height h_attr. Returns (burst models, [(model index, 0, difference)])."""
    from . import utils_geometry as ug
    if not all(n in ds.variables for n in ug.TRIPLES):
        raise ValueError(f'ERROR: {where}: burst {key}: the phase carries no pair baseline triples (the store '
                         f'predates them). Re-create it with transform().')
    names = ug.SCALARS + ('ref_height',)
    missing = [n for n in names if n not in ds.variables]
    if missing:
        raise KeyError(f"ERROR: {where}: burst {key} carries no {', '.join(missing)}. Pass the stack: "
                       f"x.elevation(stack).")
    values = _computed(ds, names + ug.TRIPLES)
    npair = int(np.size(values['baseline_start']))
    h_ref, dem_mode = _ref_mode(values['ref_height'])
    crs = _crs_wkt(ds)
    geos, index, pairs = [], {}, []
    dh, dv = ug.date_triples({n: values[n].ravel() for n in ug.TRIPLES})
    zero = np.zeros(2)
    for p in range(npair):
        v = {n: (values[n].ravel()[p] if values[n].size == npair else values[n].ravel()[0]) for n in ug.SCALARS}
        k = tuple(str(v[n]) for n in ug.SCALARS)
        if k not in index:
            index[k] = len(geos)
            geos.append(ug.burst_geometry(v, crs, h_ref, dem_mode, h_attr))
        pairs.append((index[k], (zero, zero), (dh[p], dv[p])))
    return geos, pairs


def _pairs_of_stack(data, key, bh, bv, dates, where) -> list:
    """[(0, b_i, b_j)] per pair of a phase grid: each date's own triple, looked up by the pair's two dates."""
    if 'ref' not in data.coords or 'rep' not in data.coords:
        raise KeyError(f'ERROR: {where}: burst {key} carries no ref/rep dates of its pairs.')
    out = []
    for r, s in zip(np.atleast_1d(data.coords['ref'].values), np.atleast_1d(data.coords['rep'].values)):
        i, j = Batch._ref_index(r, dates), Batch._ref_index(s, dates)
        out.append((0, (bh[i], bv[i]), (bh[j], bv[j])))
    return out


def _geometry_grid(block, geo, y, x, chunks):
    """A lazy (y, x) float32 grid of a geometry block function on the coordinates y, x in the chunks (cy, cx).

    The blocks get their GLOBAL pixel indices (arange of the whole grid, chunked like y and x) and the whole axes:
    the geometry nodes are the global ones, so the result does not depend on the chunks."""
    import dask.array as da
    gy = da.from_array(np.arange(len(y)), chunks=(chunks[0],))
    gx = da.from_array(np.arange(len(x)), chunks=(chunks[1],))
    return da.blockwise(block, 'yx', gy, 'y', gx, 'x', dtype=np.float32, geo=geo,
                        ycoord=np.asarray(y, np.float64), xcoord=np.asarray(x, np.float64))


def _grid_chunks(ds):
    """The (y, x) chunks of a burst's first lazy grid, one chunk without any."""
    for v in BatchCore._vars_of(ds).values():
        if BatchCore._is_grid(v) and v.chunks is not None:
            d = dict(zip(v.dims, v.chunks))
            return d['y'], d['x']
    return (ds.sizes['y'],), (ds.sizes['x'],)


def _elevation_grid(data, geos, pairs, joint):
    """elevation() of one phase grid, (pair, y, x) or one pair's (y, x): lazy, the phase and its y, x only."""
    import dask.array as da
    from . import utils_geometry as ug
    arr = data.data[None] if data.dims == ('y', 'x') else data.data
    if not isinstance(arr, da.Array):
        # a computed (numpy) phase: one pair per chunk, and (y, x) in the library chunk budget, not the full grid
        from .utils_dask import rechunk2d
        c = rechunk2d(arr.shape[-2:], arr.dtype.itemsize)
        arr = da.from_array(np.asarray(arr), chunks=(1, c['y'], c['x']))
    # the blocks get their GLOBAL pixel indices and the whole axes (the nodes are the global ones, see _geometry_grid)
    gy = da.from_array(np.arange(arr.shape[1]), chunks=(arr.chunks[1],))
    gx = da.from_array(np.arange(arr.shape[2]), chunks=(arr.chunks[2],))
    pidx = da.from_array(np.arange(arr.shape[0]), chunks=(arr.chunks[0],))
    res = da.blockwise(ug.elevation_block, 'pyx', arr, 'pyx', pidx, 'p', gy, 'y', gx, 'x', dtype=np.float32,
                       geos=geos, pairs=pairs, joint=joint,
                       ycoord=np.asarray(data.y.values, np.float64), xcoord=np.asarray(data.x.values, np.float64))
    return xr.DataArray(res[0] if data.dims == ('y', 'x') else res, dims=data.dims, coords=data.coords)


def _ref_heights(batch, where: str | None = None) -> dict:
    """`{burst: height}` from _ref_height_stored() for every burst Dataset of `batch` (a Batch or a dict).

    A burst without ref_height gets 0.0, and the call prints ONE warning naming all of them: 0.0 is right for a
    DEM-mode stack and wrong by the reference height for a flat-earth one, and a stack that does not store the
    value cannot tell which it is.
    """
    heights = {key: _ref_height_stored(ds) for key, ds in batch.items()}
    missing = [str(key) for key, h in heights.items() if h is None]
    if missing:
        names = [k for k in missing if k]
        print(f'WARNING: {where + ": " if where else ""}ref_height is missing'
              f'{" in burst(s) " + ", ".join(names) if names else ""}, so 0.0 is used as the flat-earth reference '
              f'height. That is right for a DEM-mode stack (remove_topo_phase=True) and wrong by the reference '
              f'height for a flat-earth one (remove_topo_phase=False).', flush=True)
    return {key: 0.0 if h is None else h for key, h in heights.items()}


def _trend2d_accumulate_for_dask(data_blk, *args, stats=None, n_vars=0,
                                 **kwargs):
    """Module-level entry point for the trend2d() accumulator.

    Module level so dask ships it by name; blockwise hands over the covariate
    blocks and then the block's own y and x coordinates -- the checkerboard
    halves are read off them -- as separate positional arguments. The date
    axis is contracted, so `data_blk` is the task's date chunks merged.
    """
    from . import utils_detrend
    return utils_detrend.trend2d_accumulate(
        data_blk, tuple(args[:n_vars]), stats,
        coords=(args[n_vars], args[n_vars + 1]), **kwargs)


def _trend2d_finalize_for_dask(acc, dates=None, *, stats=None,
                               cells=0, k=0, axes=(), label='', degree=1):
    """One date block of accumulators -> its coefficients.

    Columns are the k gradients, the constant, why the date failed if it
    did (0 solved, 1 no pixels or a checkerboard half with none, 2 the trend walked out of `range`, 3 the
    covariate is degenerate on this date, 4 the iteration did not converge,
    5 a different trend beats the answer on the global lattice, 6 the
    coherence is not above the noise level). Every
    unresolved date has NaN in the gradients, the constant, the errors, the
    coherence, the coherence at ZERO trend and the gain: NaN means no data,
    a coherence is proved only by a trend that stands. The coherence at the
    solution, the coherence at ZERO trend (their difference is what the trend
    bought), the sample count, the k
    reaches -- the half-power width of each variable's own sampling -- the
    k errors and the k centroids of the date's pixels.
    At degree 0 the gradients are zero, the reaches NaN, and the first error
    column is the constant's one-sigma in radians.
    """
    import numpy as np
    from . import utils_detrend
    stats = np.asarray(stats, np.float64).ravel()
    acc = np.asarray(acc)
    g, c, coh, coh0, det, why, lim, err = utils_detrend.trend2d_fit(
        acc, cells, k, axes=axes, degree=degree)
    span = np.maximum(stats[k:2 * k], 1e-30)[None, :]
    # the transform answers in turns across the extent; the plane wants a rate
    g = g * (2 * np.pi) / span
    if not det.all():
        # NAME THEM. A date that comes back NaN takes every pixel of the stack
        # with it downstream, so the message has to say which date and what
        # stopped it rather than how many fell out of one block.
        _reason = {1: 'no pixels, or a checkerboard half with none',
                   2: 'the trend walked out of `range` -- widen it',
                   3: 'degenerate covariate on this date',
                   4: 'did not converge -- an unconverged number is a '
                      'plausible wrong value, so it is not returned',
                   5: 'a different trend beats it on the global lattice -- coherence NaN',
                   6: 'coherence not above the noise level -- not proved'}
        _d = (np.asarray(dates).ravel() if dates is not None
              else np.arange(det.size))
        for _i in np.flatnonzero(~det):
            _nm = (str(np.asarray(_d[_i], dtype='datetime64[D]'))
                   if dates is not None else f'index {_d[_i]}')
            print(f"trend2d('{label}'): {_nm} did not resolve and comes back "
                  f"NaN -- {_reason.get(int(why[_i]), 'unknown')} "
                  f"(coherence {coh[_i]:.3f})", flush=True)
    L = utils_detrend.trend2d_layout(cells, k, len(tuple(axes)))
    K, hw = L['K'], L['half']
    # the two halves' counts and first moments add up to the total's
    n = acc[:, 2 * K:2 * K + 1] + acc[:, hw + 2 * K:hw + 2 * K + 1]
    # the centroid of each date's pixels in the variable's own units: the
    # accumulator carries the first moment of the centred, scaled covariate
    _m1 = (acc[:, 2 * K + 1:2 * K + 1 + k]
           + acc[:, hw + 2 * K + 1:hw + 2 * K + 1 + k])
    cen = stats[:k][None, :] + (_m1 / np.maximum(n, 1.0)) * span
    # a gradient's error is in turns across the extent like the gradient; the
    # constant's is already radians
    if degree:
        err = err * (2 * np.pi) / span
    return np.concatenate(
        [g, c[:, None], why[:, None].astype(np.float64), coh[:, None],
         coh0[:, None], n, lim * (2 * np.pi) / span, err, cen], axis=1)


class _Fit3dChain:
    """ONE HEAVY TASK PER WORKER, across every pass and every burst.

    A gate is a task's dependency on the output `width` tasks before it,
    which keeps that many in flight whatever cluster the caller brought.
    Measured on the scene driver: eight tasks with two threads each beat
    sixteen with one by 15% on the bandwidth-bound scans, at the same
    memory, and the thread budget it fixes keeps the product bit-identical
    between runs. The chain runs THROUGH the passes -- the first attach
    gates on the last scans -- and through the bursts of a per-burst call,
    so a multi-burst stack under union=False never holds more tasks at once
    than a single scene does. Sized by the first setup that sees it.
    """
    __slots__ = ('width', 'outs', 'seed')

    def __init__(self, width=None):
        import dask as _dask
        self.width = None if width is None else int(width)
        self.outs = []
        self.seed = _dask.delayed('start', name='fit3d-seed')

    def gate(self):
        """The dependency the next task takes: the output `width` back, the
        seed until there are that many, nothing when no width gates."""
        if not self.width:
            return None
        return (self.outs[-self.width] if len(self.outs) >= self.width
                else self.seed)

    def push(self, out):
        self.outs.append(out)


class _Fit3dSlice:
    """A window of the stack, READ INSIDE THE FIT TASK.

    Handing `delayed` a dask array makes the window a task of its own: the
    scheduler computes it wherever a worker is free and holds it until the fit
    runs, so idle workers fill with windows they will never fit. Carried like
    this it is an ordinary argument, and the read happens on the worker that is
    about to fit it. Measured on arcs(), the same change took a wide cluster
    from 81 GB to 8.4 GB at the same wall time.

    COMPUTED AS IT IS CHUNKED, WITHOUT A RECHUNK. Asking dask for the window
    as ONE chunk makes it build a merge layer per read -- and the driver used
    to force the date axis into one chunk first, so the graph carried
    `rechunk-split-rechunk-merge-rechunk-merge` on top of that. Both are
    removed: the window is computed with the chunking it already has, and dask
    fuses the reads and concatenates them itself.

    GATHERING THE BLOCKS BY HAND IS SLOWER, MEASURED. The date axis is one
    chunk per acquisition, so a window is ~90 blocks; pulling them as separate
    futures and copying each into a buffer holds less memory at the peak but
    pays a scheduler round-trip and a transfer per block, and ran 1.6x slower
    on the same crop and cluster. One call is what the scheduler is for.
    """
    __slots__ = ('arr',)

    def __init__(self, arr):
        # THE CALLER'S CHUNKING IS KEPT. Nothing about the window needs a
        # different one, and re-chunking it here would put back the merge
        # layer this class exists without.
        self.arr = arr

    def read(self, threads=1):
        """The window as a numpy block, computed HERE, on this task's threads.

        THE TASK COMPUTES ITS OWN WINDOW. The window's graph is zarr chunk
        reads and elementwise arithmetic -- the detrend evaluates its model
        per chunk -- so the local threaded scheduler runs it inside the task:
        the chunks are decoded where they are used, nothing is shipped
        between workers, and the task keeps its thread slot. Handing the
        window to the cluster instead -- the old `worker_client` route --
        spread the reads over whichever workers were free and seceded this
        task from its slot while it waited, which let the scheduler stack
        further heavy tasks onto the same worker while others idled.

        A WINDOW THAT HOLDS CLUSTER-HELD PIECES -- a persisted stack, or a
        product still made of futures -- cannot be computed locally; those go
        through the worker's client as before.
        """
        import numpy as np
        from distributed import Future
        arr = self.arr
        if any(isinstance(v, Future) for v in dict(arr.__dask_graph__()).values()):
            from distributed import worker_client
            with worker_client() as _cl:
                return np.asarray(_cl.compute(arr, sync=True))
        return np.asarray(arr.compute(scheduler='threads',
                                      num_workers=max(1, int(threads))))


def _fit3d_keep(value, *deps):
    """Return `value`, having waited for `deps`. The graph has no other way
    to say "run this too" without making it a real input of something."""
    return value


# fit1d's per-block screen: the share of a block's pixels, ranked by raw
# resultant, whose fit residuals define the block's common phase per date
# (never fewer than one). A per-date constant is a mild correction that only
# has to be LOCAL; it is not an elevation slope, so a dask block is a fine
# unit for it and no window in metres is needed.
_FIT1D_SCREEN_TOP = 0.01


def _horn_gradient_2d_for_dask(block, dy=1.0, dx=1.0):
    """Horn (1981) 3 x 3 terrain gradient of a DEM block, for map_overlap.

    Module level so dask ships it by name. The block arrives WITH its halo and
    leaves the same shape, so dask trims the depth back off.

    The two derivatives are packed into ONE complex array -- gx + 1j*gy, per
    metre of easting and northing -- because they come from the same nine
    reads. Slope and aspect are then elementwise on the result and cost no
    second pass over the raster.

    `dy`/`dx` are SIGNED coordinate steps per index, so northing running either
    way down the array gives the same gradient.
    """
    import numpy as np
    z = np.asarray(block, dtype=np.float64)
    flat = z.ndim == 3
    if flat:
        z = z[0]
    out = np.full(z.shape, np.nan, dtype=np.complex64)
    if z.shape[0] >= 3 and z.shape[1] >= 3:
        a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
        d, f = z[1:-1, :-2], z[1:-1, 2:]
        g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
        # each side sums to four weights over a two-step span, hence the 8
        gx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
        gy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
        out[1:-1, 1:-1] = (gx + 1j * gy).astype(np.complex64)
    return out[None] if flat else out


def _apply_goldstein_2d_for_dask(phase_block, corr_block, psize=32, threshold=0.5, halo=None, block_info=None):
    """Module-level function for Goldstein filter map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles both 2D (y, x) and 3D (1, y, x) blocks - _goldstein() handles squeeze/unsqueeze.

    THE PATCH GRID BELONGS TO THE WHOLE ARRAY, NOT TO THE BLOCK. The patches sit at multiples of half a patch counted
    from pixel 0 of the array, so every pixel is filtered by the same patches whatever the dask chunks are. A block
    starts at (chunk start - halo), halo is the overlap depth map_overlap was given (goldstein() passes it here, so
    the two cannot drift apart), and the rows and columns before its first patch of the array grid are overlap that
    map_overlap trims: the filter leaves them out (skip). block_info locates the block in the OVERLAPPED array, where
    an inner chunk edge has grown by halo on both sides, so the block starts at location - 2 * halo * chunk index.
    block_info is None only for dask's zero-size probe of the output type.

    Parameters
    ----------
    phase_block : np.ndarray
        Complex array from dask, shape (y, x) or (1, y, x)
    corr_block : np.ndarray
        Real array from dask, shape (y, x) or (1, y, x)
    psize : dict
        Patch size {'y': ..., 'x': ...} for the filter
    threshold : float
        Minimum fraction of valid pixels
    halo : tuple of int
        The overlap depth (y, x) of the map_overlap that cut this block.
    block_info : dict
        Supplied by dask.

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input
    """
    skip = (0, 0)
    if block_info is not None:
        info = block_info[0]
        py, px = psize['y'], psize['x']
        skip = tuple((2 * h * i - loc[0]) % (p // 2)
                     for loc, i, h, p in zip(info['array-location'][-2:], info['chunk-location'][-2:], halo, (py, px)))
    # _goldstein handles (1, y, x) -> squeeze -> process -> unsqueeze
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize, threshold=threshold, skip=skip)


class Batch(BatchCore):

    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the real 2D vars from Stack
        if isinstance(mapping, Stack):
            #print ('Batch __init__: Stack')
            real_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only non-complex data_vars that live on the ('y','x') grid
                # and include 1D non-complex variables (e.g., per-axis metadata)
                # the 2D grid, plus every non-gridded var whatever its rank:
                # per-date POLYNOMIAL COEFFICIENTS are ('date', '<name>_coef'),
                # and the old `len(dims) == 1` test dropped them, taking the
                # orbit polynomials and every other per-date list out of the
                # transform view that exists to carry them
                real_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind != 'c'
                    and (
                        tuple(ds[v].dims) == ('y', 'x')
                        or len(ds[v].dims) == 1
                        or not ({'y', 'x'} & set(ds[v].dims))
                    )
                ]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('Batch __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})
    
    def clip(self, min=None, max=None, **kwargs):
        """
        used for correlation in [0,1] range: a BatchUnit when the values are
        float, otherwise a plain Batch (a BatchUnit holds float units only)
        """
        return BatchCore._as_class(BatchUnit, dict(super().clip(min=min, max=max, **kwargs)))

    @staticmethod
    def _compute_rgb(copol: np.ndarray, xpol: np.ndarray,
                     gamma: float = 1.0, brightness: float = 2.0,
                     quantile: list = None) -> np.ndarray:
        """
        Compute RGB composite from co-pol and cross-pol arrays.

        Parameters
        ----------
        copol : np.ndarray
            Co-polarization data (HH or VV), shape (..., y, x)
        xpol : np.ndarray
            Cross-polarization data (HV or VH), shape (..., y, x)
        gamma : float
            Gamma correction (>1 brightens dark areas)
        brightness : float
            Linear brightness multiplier
        quantile : list
            Quantile range for normalization, default [0.02, 0.98]

        Returns
        -------
        np.ndarray
            RGB array as float32 [0-1], shape (..., y, x, 3)
        """
        # Normalize each channel to [0, 1] using quantile stretch
        def normalize_channel(data):
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                return np.zeros_like(data)
            q_vals = quantile if quantile is not None else [0.02, 0.98]
            if np.isscalar(q_vals):
                q_vals = [q_vals, 1 - q_vals] if q_vals < 0.5 else [1 - q_vals, q_vals]
            q = np.nanquantile(valid, q_vals)
            vmin_ch, vmax_ch = q[0], q[-1]
            if vmax_ch <= vmin_ch:
                vmax_ch = vmin_ch + 1e-10
            normalized = (data - vmin_ch) / (vmax_ch - vmin_ch)
            return np.clip(normalized, 0, 1)

        # R=copol, G=xpol, B=copol
        r_norm = normalize_channel(copol)
        g_norm = normalize_channel(xpol)
        b_norm = normalize_channel(copol)

        # Apply gamma correction
        if gamma != 1.0:
            r_norm = np.power(r_norm, 1.0 / gamma)
            g_norm = np.power(g_norm, 1.0 / gamma)
            b_norm = np.power(b_norm, 1.0 / gamma)

        # Handle NaN (set to 0)
        nan_mask = ~np.isfinite(copol) | ~np.isfinite(xpol)
        r_norm = np.where(nan_mask, 0, r_norm)
        g_norm = np.where(nan_mask, 0, g_norm)
        b_norm = np.where(nan_mask, 0, b_norm)

        # Stack to RGB
        rgb_float = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # Apply brightness
        if brightness != 1.0:
            rgb_float = rgb_float * brightness
            rgb_float = np.clip(rgb_float, 0, 1)

        return rgb_float.astype(np.float32)

    def plot(
        self,
        cmap = 'turbo',
        alpha = 0.5,
        caption = None,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

    def plot2(self, *args, **kwargs):
        """
        Plot dual-pol RGB composite (shortcut for plot(composite=True)).

        This is a convenience method for dual-polarization data that creates
        an RGB composite where R=co-pol, G=cross-pol, B=co-pol.

        All arguments are passed to plot() with composite=True.

        See Also
        --------
        plot : Full plotting method with all options.
        """
        kwargs["composite"] = True
        return self.plot(*args, **kwargs)

    def rgb(self, gamma: float = 1.0, brightness: float = 2.0, quantile: list = None):
        """
        Create RGB composite from dual-pol data as xarray DataArray.

        Standard dual-pol RGB decomposition: R=co-pol, G=cross-pol, B=co-pol
        - Magenta/pink: high co-pol, low cross-pol (surface scattering, urban)
        - Green: high cross-pol (volume scattering, vegetation)
        - White/gray: both high (mixed scattering)
        - Dark: both low (smooth surfaces, water)

        Parameters
        ----------
        gamma : float, optional
            Gamma correction for brightness. Default 1.0.
            Values > 1 brighten dark areas, < 1 increase contrast.
        brightness : float, optional
            Linear brightness multiplier. Default 2.0.
        quantile : list, optional
            Quantile range for normalization. Default [0.02, 0.98].

        Returns
        -------
        xr.DataArray
            RGB array with dims (band, y, x) or (date/pair, band, y, x).
            Values are uint8 [0-255]. NaN pixels have value 0.

        Examples
        --------
        >>> model = stack.fit1d_network()
        >>> # modelled displacement per date, unwrapped
        >>> disp  = stack.predict(model=model).displacement_los(stack.transform())
        >>> # remove the modelled ground motion, keeping topography
        >>> ground = stack.complex() * stack.predict(model=model).iexp(-1)
        >>> # remove the whole model, topography included
        >>> resid  = stack.complex() * stack.predict(model=model, baseline='BPR').iexp(-1)
        """
        import numpy as np
        import xarray as xr
        import dask

        # Check for exactly 2 polarizations
        sample = next(iter(self.values()))
        polarizations = [v for v in sample.data_vars
                        if sample[v].dims[-2:] == ('y', 'x')]
        if len(polarizations) != 2:
            raise ValueError(f"rgb() requires exactly 2 polarizations, found {len(polarizations)}: {polarizations}")

        pol1, pol2 = polarizations[0], polarizations[1]

        # Get stack variable (date or pair)
        stackvar = list(sample[pol1].dims)[0] if len(sample[pol1].dims) > 2 else None

        # Merge to single dataset
        ds = self.to_dataset()
        da_copol = ds[pol1]
        da_xpol = ds[pol2]

        if stackvar is None:
            stackvar = 'fake'
            da_copol = da_copol.expand_dims({stackvar: [0]})
            da_xpol = da_xpol.expand_dims({stackvar: [0]})

        # Materialize
        da_copol, da_xpol = dask.persist(da_copol, da_xpol)
        from .utils_dask import progress_persisted
        progress_persisted([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

        # Compute RGB using shared method from BatchCore
        copol = da_copol.values
        xpol = da_xpol.values
        rgb_float = Batch._compute_rgb(copol, xpol, gamma=gamma,
                                       brightness=brightness, quantile=quantile)
        rgb_uint8 = (rgb_float * 255).astype(np.uint8)

        # Create DataArray with band-first order for rasterio compatibility
        if stackvar == 'fake':
            # Remove fake dimension: (1, y, x, 3) -> (y, x, 3) -> (3, y, x)
            rgb_uint8 = rgb_uint8[0]
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=['y', 'x', 'band'],
                coords={'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose('band', 'y', 'x')
        else:
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=[stackvar, 'y', 'x', 'band'],
                coords={stackvar: da_copol[stackvar], 'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose(stackvar, 'band', 'y', 'x')

        rgb_da.attrs['crs'] = self.crs
        return rgb_da

    def lee(self, *args, **kwargs):
        """
        Apply Enhanced Lee speckle filter to reduce noise while preserving edges.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "lee() requires insardev_backscatter extension"
        )

    @staticmethod
    def _ref_index(ref, dates):
        """Which acquisition a `ref` names, as an index into `dates`.

        Accepts what S1.transform(ref=...) accepts and one thing more:

          None                 the model's own anchor -- nothing is subtracted
          int                  position in the stack, Python-style, so 0 is the
                               first acquisition and -1 the last
          str / datetime       an acquisition date, matched to the day

        A date that is not in the stack raises rather than silently picking a
        neighbour: predict() re-references the output to it, and a near miss
        would shift every plane by a constant nobody asked for.
        """
        import numpy as np
        import pandas as pd
        d = np.asarray(dates).astype('datetime64[D]')
        if isinstance(ref, (bool, np.bool_)):
            raise TypeError(
                "ref takes a date or a stack position, not a bool. Use ref=0 "
                "for the first acquisition and ref=None for the model's anchor.")
        if isinstance(ref, (int, np.integer)):
            i = int(ref)
            if not -len(d) <= i < len(d):
                raise IndexError(
                    f"ref={ref} is out of range for {len(d)} acquisitions.")
            return i % len(d)
        try:
            want = np.datetime64(pd.to_datetime(ref), 'D')
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"ref must be None, an int position or a date, got {ref!r}") from e
        hit = np.nonzero(d == want)[0]
        if not len(hit):
            raise KeyError(
                f"ref={ref!r} is not one of this stack's acquisitions "
                f"({str(d[0])} .. {str(d[-1])}, {len(d)} dates).")
        return int(hit[0])

    def trend2d(self, transform: 'BatchCore | None' = None, weight: 'BatchUnit | None' = None,
                degree: int = 1, detrend: bool = False,
                extrapolate: bool = False, debug: bool = False) -> 'BatchCore':
        """
        Compute 2D polynomial trend (ramp) from data.

        Real data (Batch): standard polynomial, returns Batch. Complex data
        have their own BatchComplex.trend2d(): a per-date table without (y, x)
        grids, which no operator takes (N81); detrend() removes it.

        Parameters
        ----------
        transform : BatchCore or None
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
            If None, uses y,x grid coordinates as regressors.
        weight : BatchUnit or None
            Optional weight for the fitting (typically correlation).
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        detrend : bool
            If True, return detrended data instead of the trend surface.
            Fuses fit+subtract into one blockwise call so the input phase is
            referenced only once in the dask graph, avoiding memory pinning.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch or BatchComplex
            Trend surface (same type as input).

        Examples
        --------
        >>> # With radar coordinates, the transform at the phase posting
        >>> trend = phase.trend2d(stack.transform()[['azi', 'rng']].downsample(phase.spacing), weight=corr)
        >>> # With y,x grid coordinates (no transform needed)
        >>> trend = phase.trend2d(weight=corr)
        >>> # Complex data: a per-date trend table, removed by detrend()
        >>> flat = stack.detrend(stack.trend2d('northing', 'easting', 'ele'))
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from . import utils_detrend
        from .Batch import Batch, BatchComplex

        phase = self
        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, phase)

        # Validate lazy data
        BatchCore._require_lazy(phase, 'trend2d')

        is_complex = isinstance(phase, BatchComplex)

        # Unify transform keys to phase
        if transform is not None:
            transform = transform.sel(phase)

        result = {}
        for key in phase.keys():
            ds = phase[key]
            # the grids of a Dataset, or a DataArray as its own one (N81); none is _form's
            # short error, not an IndexError
            grids = BatchCore._grids_of(ds)
            if not grids:
                raise BatchCore._no_grid_error(ds)
            pols = list(grids)

            phase_da_ref = grids[pols[0]]
            phase_shape = phase_da_ref.shape[-2:]
            if min(phase_shape) < 3:
                raise ValueError(f"trend2d() needs a burst grid of at least 3 x 3 pixels, burst '{key}' is {phase_shape[0]} x {phase_shape[1]}.")
            if not result and degree: print("NOTE: Batch.trend2d() computes its feature statistics eagerly.")
            phase_dy = float(phase_da_ref.y.diff('y')[0])
            phase_dx = float(phase_da_ref.x.diff('x')[0])

            if transform is not None:
                trans_ds = transform[key]
                var_names = [v for v in trans_ds.data_vars
                            if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]

                # Check that transform resolution matches phase resolution
                trans_da_ref = trans_ds[var_names[0]]
                trans_shape = trans_da_ref.shape
                if phase_shape != trans_shape:
                    trans_dy = float(trans_da_ref.y.diff('y')[0])
                    trans_dx = float(trans_da_ref.x.diff('x')[0])
                    raise ValueError(
                        f"Transform shape {trans_shape} does not match phase shape {phase_shape}. "
                        f"Phase spacing: dy={phase_dy:.1f}, dx={phase_dx:.1f}. "
                        f"Transform spacing: dy={trans_dy:.1f}, dx={trans_dx:.1f}. "
                        f"Use stack.transform()[['azi','rng','ele']].downsample(N) to match."
                    )
            else:
                trans_ds = None
                var_names = ['y', 'x']

            if weight is not None:
                weight_grids = BatchCore._grids_of(weight[key])
                if weight_grids:
                    weight_da_ref = next(iter(weight_grids.values()))
                    weight_shape = weight_da_ref.shape[-2:]
                    if phase_shape != weight_shape:
                        weight_dy = float(weight_da_ref.y.diff('y')[0])
                        weight_dx = float(weight_da_ref.x.diff('x')[0])
                        raise ValueError(
                            f"Weight shape {weight_shape} does not match phase shape {phase_shape}. "
                            f"Phase spacing: dy={phase_dy:.1f}, dx={phase_dx:.1f}. "
                            f"Weight spacing: dy={weight_dy:.1f}, dx={weight_dx:.1f}. "
                            f"Use weight.downsample(N) to match."
                        )

            if debug:
                print(f"DEBUG {key}: variables={var_names}")

            result_ds = {}
            for pol in pols:
                phase_da = grids[pol]
                weight_da = BatchCore._weight_of(weight[key], pol) if weight is not None else None

                phase_dask = phase_da.data

                # Handle 2D input by promoting to 3D
                squeeze_pair = phase_dask.ndim == 2
                if squeeze_pair:
                    phase_dask = phase_dask[np.newaxis, ...]

                # Merged chunking: dim 0 is a single chunk spanning all pairs.
                # Skip rechunk to avoid expensive P2P shuffle — the kernels
                # (_accumulate_chunk, _solve_chunk, _apply_chunk) all handle
                # multi-pair blocks via internal loops.
                dim0_merged = (len(phase_dask.chunks[0]) == 1
                               and phase_dask.chunks[0][0] > 1)

                # Per-pair chunking: ensure pair dimension is chunked to 1
                if not dim0_merged and any(c != 1 for c in phase_dask.chunks[0]):
                    phase_dask = phase_dask.rechunk({0: 1})

                n_pairs = phase_dask.shape[0]

                # Build variable arrays for the fit
                phase_spatial_chunks = phase_dask.chunks[-2:]
                if trans_ds is not None:
                    var_dask_list = []
                    for v in var_names:
                        var_dask = trans_ds[v].data
                        if var_dask.chunks != phase_spatial_chunks:
                            var_dask = var_dask.rechunk(phase_spatial_chunks)
                        var_dask_list.append(var_dask)
                else:
                    # Use y,x grid coordinates as regressors
                    y_np, x_np = np.meshgrid(
                        phase_da.y.values.astype(np.float32),
                        phase_da.x.values.astype(np.float32),
                        indexing='ij')
                    var_dask_list = [
                        da.from_array(y_np, chunks=phase_spatial_chunks),
                        da.from_array(x_np, chunks=phase_spatial_chunks),
                    ]

                # Phase 0: Compute global per-variable standardization (pair-independent),
                # applied to every variable BEFORE the polynomial terms are built
                feature_mean, feature_std = utils_detrend._compute_feature_stats(
                    var_dask_list, degree)
                n_poly = utils_detrend._n_poly_features(len(var_dask_list), degree)
                n_feat = n_poly + 1  # +1 for bias
                n_feat_b = 2 * n_feat if is_complex else n_feat
                n_accum = n_feat * n_feat + n_feat_b + 1
                n_coeff_out = 2 * n_feat if is_complex else n_feat

                if debug:
                    print(f"DEBUG {key}/{pol}: n_feat={n_feat}, n_accum={n_accum}, "
                          f"n_coeff_out={n_coeff_out}, chunks={phase_dask.chunks}")

                # Phase 1: Accumulate partial normal equations per (pair, tile)
                n_vars = len(var_dask_list)
                has_weight = weight_da is not None

                def make_accumulate_fn(has_weight, n_vars, feature_mean,
                                       feature_std, degree, is_complex):
                    def fn(*args):
                        phase_c = args[0]
                        if has_weight:
                            weight_c = args[1]
                            var_cs = args[2:2 + n_vars]
                        else:
                            weight_c = None
                            var_cs = args[1:1 + n_vars]
                        return utils_detrend._accumulate_chunk(
                            phase_c, weight_c, var_cs,
                            feature_mean, feature_std, degree, is_complex)
                    return fn

                accumulate_fn = make_accumulate_fn(
                    has_weight, n_vars, feature_mean, feature_std,
                    degree, is_complex)

                blockwise_args = [phase_dask, 'pyx']
                if has_weight:
                    weight_dask = weight_da.data
                    if squeeze_pair:
                        weight_dask = weight_dask[np.newaxis, ...]
                    if weight_dask.chunks != phase_dask.chunks:
                        weight_dask = weight_dask.rechunk(phase_dask.chunks)
                    blockwise_args.extend([weight_dask, 'pyx'])
                for v_dask in var_dask_list:
                    blockwise_args.extend([v_dask, 'yx'])

                partials = da.blockwise(
                    accumulate_fn, 'pyxf',
                    *blockwise_args,
                    adjust_chunks={'y': 1, 'x': 1},
                    new_axes={'f': n_accum},
                    dtype=np.float64,
                    meta=np.empty((0, 0, 0, 0), dtype=np.float64),
                )

                # Phase 2: Sum across spatial chunks (tree reduction)
                summed = partials.sum(axis=(1, 2))  # (n_pairs, n_accum)

                # Phase 3: Solve per pair
                def make_solve_fn(n_feat, is_complex):
                    def fn(block):
                        return utils_detrend._solve_chunk(
                            block, n_feat, is_complex)
                    return fn

                solve_fn = make_solve_fn(n_feat, is_complex)
                coeffs = da.map_blocks(
                    solve_fn, summed,
                    dtype=np.float64,
                    chunks=(summed.chunks[0], (n_coeff_out,)),
                )  # (n_pairs, n_coeff_out)

                # Phase 4: Apply trend per (pair, tile)
                def make_apply_fn(n_vars, feature_mean, feature_std,
                                  degree, is_complex, detrend_mode, extrapolate):
                    def fn(*args):
                        phase_c = args[0]
                        coeffs_c = args[1]
                        var_cs = args[2:2 + n_vars]
                        return utils_detrend._apply_chunk(
                            phase_c, coeffs_c, var_cs,
                            feature_mean, feature_std,
                            degree, is_complex, detrend_mode, extrapolate)
                    return fn

                apply_fn = make_apply_fn(
                    n_vars, feature_mean, feature_std,
                    degree, is_complex, detrend, extrapolate)

                out_dtype = phase_da.dtype if is_complex else np.float32
                blockwise_args_apply = [phase_dask, 'pyx', coeffs, 'pf']
                for v_dask in var_dask_list:
                    blockwise_args_apply.extend([v_dask, 'yx'])

                result_dask = da.blockwise(
                    apply_fn, 'pyx',
                    *blockwise_args_apply,
                    concatenate=True,
                    dtype=out_dtype,
                    meta=np.empty((0, 0, 0), dtype=out_dtype),
                )

                if squeeze_pair:
                    result_dask = result_dask[0]

                trend_da = xr.DataArray(
                    result_dask,
                    dims=phase_da.dims,
                    coords=phase_da.coords
                )

                result_ds[pol] = trend_da

            # a DataArray in, a DataArray out (N81)
            result[key] = BatchCore._form(ds, result_ds, attrs=ds.attrs)

        if is_complex:
            return BatchComplex(result)
        return Batch(result)

    def fit1d(self, weight=None, baseline: str = 'BPR',
              max_dh: float = 30.0,
              max_seasonal: float = 0.0,
              debug: bool = False) -> 'Batch':
        """
        Full per-pixel model on UNWRAPPED per-PAIR phase -- NO network.

        The unwrapped twin of BatchComplex.fit1d(): the same model and the same
        output, fitted by weighted least squares instead of by coherence
        maximisation, because unwrapped phase makes the objective convex. Returns
        the MODEL ONLY, named and scaled as fit1d_network() names and scales it, so
        predict(model) is the one inverse for all three.

        WHY PAIRS AND NOT AN INVERTED SERIES. Multilooked phase has no zero
        closure, so a per-date series is not recoverable without committing to
        an inversion first. It does not need to be: every pair is the SAME
        absolute-time model differenced at two epochs,

            phi_p = -velocity * dt_p - height * ele2phase_p
                    - Re(seasonal) * dcos_p - Im(seasonal) * dsin_p

        with dt = t_rep - t_ref, ele2phase_p = dBPR_p / median(R sin(incidence)),
        and dcos_p = cos(2 pi t_rep) - cos(2 pi t_ref). The network the lstsq()
        inversion would solve is already IN those columns. Fitting here rather
        than after an inversion avoids a rank-deficient solve, a datum choice,
        and the propagation of one bad pair down a whole cumulative series --
        it is one bad ROW instead.

        The signs are the interferometric ones and are not a convention this
        function chose: interferogram() forms ref * conj(rep) while pairs() forms
        dt and dBPR as rep - ref, so a pair carries MINUS the model difference.

        IT DOES NOT FILTER. A long-wavelength screen in the pairs biases every
        parameter, and removing it is the caller's decision and the caller's
        scale:

        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)

        Parameters
        ----------
        weight : BatchUnit or None
            Per-pair correlation, aligned 1:1 with the pairs. Enters as
            gamma^2, bounded at 0.999: the unbounded gamma^2/(1-gamma^2) lets
            one pair decide a low-correlation pixel.
        baseline : str
            Per-pair perpendicular baseline coordinate, default 'BPR'. RAISES
            when it is absent: a height-free fit reports a velocity that has
            absorbed the topographic phase, and nothing downstream can tell.
            Pass None to ask for that fit deliberately -- which is only right
            when the topographic phase is already removed -- and `height` then
            comes back NaN.
        max_dh, max_seasonal : float
            In metres and in mm of LOS half-amplitude. BOTH ENTER THE FIT, which
            is why they are arguments: `max_dh` sets the shrinkage prior on the
            DEM-error column -- sigma_dh = max_dh/3, a property of the terrain
            model -- and `max_seasonal=0` leaves the annual out of the design
            entirely rather than filtering it afterwards.

            `max_dh` GATES THE PRIOR, NOT THE ANSWER, which is the distinction
            that decides what belongs in this signature. A height solving beyond
            it says the DEM the prior came from does not describe this pixel, so
            the fit standing on that DEM is garbage in, garbage out and NaN is
            the honest answer. A RATE bound would gate the measurement itself --
            an assumption about how fast the ground may move, imposed on the one
            quantity being measured -- which is why there is none here. On
            wrapped phase it is unavoidable, the aliases sitting one cycle per
            year apart and the search having to stay inside half that spacing;
            unwrapped phase has no such comb, so a rate limit could only mask a
            solved answer. `model.velocity.where(...)` does that at the call
            site, where it cannot silently empty a subsiding scene.

            Neither is a search range: the objective is convex on unwrapped
            phase, so there is no lattice to size and no guard band to keep a
            peak off a boundary -- which is why step_dh and step_dv have no
            counterpart in this signature either. Nothing is clipped; a height
            outside `max_dh` is reported as NaN, not pinned to the edge.

            The annual is identifiable only when the pairs sample different
            times of year. Over a span well short of one, the annual columns
            collapse onto the rate column; the design report below states by
            how much rather than letting the split look decided.

        Returns
        -------
        Batch
            The model, named and scaled exactly as fit1d_network() and
            BatchComplex.fit1d():

              `velocity`   rad/yr
              `height`     rad per unit ele2phase, NaN with no baseline
              `seasonal`   complex rad, 0 when max_seasonal=0
              `coherence`  exp(-rmse^2/2) of the reported model
              `rmse`       sqrt(-2 ln coherence), radians, n/(n-k) inflated

            plus a scalar `date` coordinate: the epoch the model is
            referenced to, which is what every fit here carries so predict()
            never has to guess. A pairs batch holds only baseline DIFFERENCES,
            so the master that fit1d_network() anchors on is not recoverable from it and
            the median acquisition is used instead -- recorded rather than
            assumed. Rate and height are indifferent to the origin; the annual
            is not, and a model fitted on one origin and removed on another
            leaves a residual annual of 2|sin(pi delta)| |seasonal|.

            NO `conncomp`: every pixel is solved alone.

            A pixel refused by `max_dh` has velocity, height and seasonal NaN
            while `coherence` and `rmse` survive, which is _3d_arc_fit()'s
            convention: they describe the fit that was attempted, and a caller
            diagnosing why a pixel was refused needs them.

        Examples
        --------
        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> noise = phase - stack.predict(model, baseline='BPR')
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        import dask
        import dask.array as da
        from .utils_fit1d import fit1d_pairs

        # the radar geometry is the Dataset's (N81): a DataArray batch raises here;
        # baseline=None reads radar_wavelength alone (the pair baselines are coordinates)
        BatchCore._needs_dataset(self, 'fit1d', BatchCore._GEOMETRY if baseline else ('radar_wavelength',))
        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, self)
        BatchCore._require_lazy(self, 'fit1d')
        model_result = {}
        _ep_all = None  # computed on first use: once per call, so a missing ref_height warns once and not once per burst
        for key in self.keys():
            ds = self[key]
            # a PAIR dim, not merely a grid: `rng` and friends are (y, x)
            # and would otherwise be counted as a second polarisation
            pols = [v for v in ds.data_vars
                    if 'pair' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                raise TypeError(
                    f"fit1d() found no (pair, y, x) variables in burst {key}. "
                    "It operates on per-PAIR unwrapped phase; a per-date "
                    "complex stack needs BatchComplex.fit1d().")
            if len(pols) > 1:
                raise ValueError(
                    f"fit1d() fits ONE polarisation; burst '{key}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two "
                    "polarisations would collide. Select one first, e.g. "
                    "batch[['VV']].")
            pol = pols[0]
            da_ = ds[pol]
            if da_.dims[0] != 'pair':
                da_ = da_.transpose('pair', ...)

            rd = (np.asarray(pd.to_datetime(da_.coords['ref'].values).values)
                  .astype('datetime64[D]').astype(np.float64))
            pd_ = (np.asarray(pd.to_datetime(da_.coords['rep'].values).values)
                   .astype('datetime64[D]').astype(np.float64))
            # the median acquisition, since the master is not recoverable from
            # differences; days, as every other origin in this library
            t0_day = float(np.median(np.unique(np.concatenate([rd, pd_]))))
            t_ref = (rd - t0_day) / 365.25
            t_rep = (pd_ - t0_day) / 365.25
            dt = t_rep - t_ref

            if 'radar_wavelength' not in ds:
                raise KeyError(
                    f"fit1d() needs 'radar_wavelength' on '{key}' to build the "
                    "height column and to read out in radians.")
            lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
            # only used to state the design's precision in physical units --
            # the columns and the solution are radians throughout
            meter2rad = 4.0 * np.pi / lam

            # ele2phase per PAIR: dBperp / (R sin(incidence)), from
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            bp = None
            for src in (da_.coords, ds):
                if baseline and baseline in src:
                    bp = np.asarray(src[baseline].values, dtype=float).ravel()
                    break
            # A MISSING BASELINE IS AN ERROR, NOT A SMALLER MODEL. Dropping the
            # height column silently returns a velocity that has absorbed the
            # topographic phase -- on a stack whose baselines drift with time
            # that is a rate wrong by tens of mm/yr, reported with the same
            # coherence as a good one. The usual way it goes missing is not a
            # pipeline that never had it: it is xarray dropping a non-index
            # coordinate when two batches disagree on its value, which happens
            # the moment phases from two bands or two stacks are combined.
            # Passing baseline=None is the way to ask for a height-free fit.
            if baseline and bp is None:
                _have = sorted(set(da_.coords) | set(ds.data_vars))
                raise KeyError(
                    f"fit1d() found no {baseline!r} for '{key}' to build the "
                    f"height column; the dataset carries {_have}. It is dropped "
                    f"when two operands disagree on it -- re-attach the one you "
                    f"mean with .assign_coords({baseline}=('pair', values)), or "
                    f"pass baseline=None for a deliberate height-free fit.")
            if bp is not None and bp.shape != dt.shape:
                raise ValueError(
                    f"fit1d(): {baseline!r} has {bp.shape[0]} value(s) for '{key}' "
                    f"but the batch has {dt.shape[0]} pair(s). Per-PAIR baselines "
                    f"are what this fit takes -- pairs() attaches them; a per-DATE "
                    f"BPR has to be differenced first.")
            e2p = None
            if bp is not None:
                # FROM THIS BATCH. A transform= argument used to select the
                # source of this scalar; every step of the pair pipeline carries
                # the metadata it reads, so it never supplied anything missing,
                # and the value only rescales `height` inversely and cancels in
                # the product.
                if _ep_all is None:
                    _ep_all = Batch._elevation_phase_approximate(self)
                _fac = _ep_all[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)
            if e2p is None:
                print(f"fit1d(): baseline=None for '{key}' -- the height column "
                      "is dropped and the topographic phase stays in the "
                      "residual.", flush=True)

            # THE PAIR CONVENTION, which is this method's own -- pairs are not
            # a per-date stack and this fit is not BatchComplex.fit1d.
            #
            #   phi_p = +velocity*dt - height*ele2phase_p + annual
            #
            # `velocity` is the rate of LOS displacement expressed in radians:
            # the sense lstsq() works in, since it sums intervals rep-ref
            # (the network inversion sums intervals rep-ref), and the sense
            # displacement_los() converts,
            # since its -lambda/4pi is derived for a pair. An SLC phase is
            # -(4pi/lambda)*r, so a pair ref*conj(rep) carries +m2r*dr while a
            # date series carries -m2r*dr -- opposite. Fitting the date sense
            # here makes model.displacement_los() report subsidence as uplift.
            #
            # `height` keeps the other sign: it is DEM error, not a displacement,
            # and nothing converts it with displacement_los(). It is RETURNED IN
            # RADIANS like everything else -- radians per unit ele2phase -- and
            # the metres only ever appear in the diagnostic printed below.
            cols = [dt]
            has_h = e2p is not None
            if has_h:
                cols.append(-e2p)
            fit_seasonal = bool(max_seasonal and max_seasonal > 0)
            if fit_seasonal:
                cols.append(np.cos(2 * np.pi * t_rep) - np.cos(2 * np.pi * t_ref))
                cols.append(np.sin(2 * np.pi * t_rep) - np.sin(2 * np.pi * t_ref))
            A = np.stack(cols, axis=1).astype(np.float64)
            k = A.shape[1]
            if A.shape[0] < k + 1:
                raise ValueError(
                    f"fit1d() needs more than {k} pairs to fit {k} parameters; "
                    f"burst '{key}' has {A.shape[0]}.")
            # HOW WELL THE PAIR SET DETERMINES EACH PARAMETER, decided once,
            # because the design is the same for every pixel. This is not the
            # residual -- `rmse` is that, and the two are independent. A short
            # stack fitted with an annual returns an UNBIASED rate with a huge
            # variance while the fit looks excellent: the rate is meaningless
            # and no residual statistic can say so. Var(theta) = sigma^2 (A'A)^-1, so
            # sigma_velocity = rmse * `rate` below, per pixel, in mm/yr.
            try:
                Cinv = np.linalg.inv(A.T @ A)
                sig = np.sqrt(np.clip(np.diag(Cinv), 0, None))
            except np.linalg.LinAlgError:
                sig = np.full(k, np.inf)
            s_v = float(sig[0]) / (meter2rad * 1e-3)
            s_h = (float(sig[1]) / meter2rad) if has_h else float('nan')
            # what the annual costs the rate: the same design without it
            vif = 1.0
            if fit_seasonal and k > (2 if has_h else 1):
                kk_ = 2 if has_h else 1
                A0 = A[:, :kk_]
                try:
                    vif = float(np.sqrt(np.linalg.inv(A0.T @ A0)[0, 0])
                                / max(float(sig[0]), 1e-30))
                    vif = 1.0 / max(vif, 1e-30)
                except np.linalg.LinAlgError:
                    vif = float('inf')
            terms = 'velocity' + (', height' if has_h else '') \
                + (', seasonal' if fit_seasonal else '')
            msg = (f"fit1d('{key}'): {A.shape[0]} pairs, {k} parameters "
                   f"({terms}), date {np.datetime64(int(t0_day), 'D')}\n"
                   f"  per radian of residual the design gives "
                   f"sigma_velocity {s_v:.3f} mm/yr"
                   + (f", sigma_height {s_h:.3f} m" if has_h else "")
                   + (f"; the annual inflates the rate by x{vif:.2f}"
                      if fit_seasonal else ""))
            if debug or not np.isfinite(s_v) or vif > 3.0:
                print(msg, flush=True)
            if vif > 3.0:
                print("  the annual is NOT separable from the rate on this "
                      "pair set -- too short a span for them to differ. The "
                      "rate stays unbiased but its error grows by that factor; "
                      "pass max_seasonal=0 if the annual is not wanted.",
                      flush=True)

            mem_per_pixel = A.shape[0] * 4 * 4
            ay, ax = dask.array.core.normalize_chunks(
                'auto', (da_.y.size, da_.x.size),
                dtype=np.dtype(f'V{mem_per_pixel}'))
            cy, cx = ay[0], ax[0]
            da_ = da_.chunk({'pair': -1, 'y': cy, 'x': cx})
            d_dask = da_.data
            w_dask = None
            if weight is not None:
                w_dask = BatchCore._weight_of(weight[key], pol).transpose('pair', ...).chunk(
                    {'pair': -1, 'y': cy, 'x': cx}).data
                if w_dask.chunks != d_dask.chunks:
                    w_dask = w_dask.rechunk(d_dask.chunks)

            # max_dh is a 3-sigma bound on a DEM error, so it also names the
            # PRIOR the height is shrunk under. One number, both jobs: what the
            # caller will accept, and what the terrain model is known to be.
            _dh_prior = meter2rad * float(max_dh) / 3.0 if has_h else None

            def _blk(b, wb=None, _A=A, _k=k, _hh=has_h, _se=fit_seasonal,
                     _dbg=bool(debug), _m2r=meter2rad,
                     _mh=float(max_dh),
                     _ms=float(max_seasonal), _dp=_dh_prior):
                shp = b.shape[1:]
                th, gam, rms = fit1d_pairs(
                    b, _A, weight=wb, dh_prior=_dp,
                    dh_col=(1 if _hh else None))
                nanp = np.full(shp, np.nan, np.float32)
                vel = th[0].reshape(shp).astype(np.float32)
                hgt = th[1].reshape(shp).astype(np.float32) if _hh else nanp
                if _se:
                    j = 2 if _hh else 1
                    sea = (th[j].reshape(shp)
                           + 1j * th[j + 1].reshape(shp)).astype(np.complex64)
                else:
                    # zero, not NaN: the model holds no annual, and that is a
                    # value rather than an absence -- the same convention fit1d_network
                    # uses at max_seasonal=0
                    sea = np.zeros(shp, np.complex64)
                gam = gam.reshape(shp).astype(np.float32)
                rms = rms.reshape(shp).astype(np.float32)
                # THE BOUNDS ARE THE GUARANTEE, not a hint: outside them the
                # pixel is NaN, never clipped to the edge and reported as if it
                # had solved there. coherence and rmse are kept -- they say how
                # the refused fit behaved, which is what a caller needs to see.
                bad = ~np.isfinite(vel)
                if _hh:
                    bad |= np.abs(hgt) > _mh * _m2r
                if _se:
                    bad |= np.abs(sea) > _ms * _m2r * 1e-3
                vel = np.where(bad, np.nan, vel).astype(np.float32)
                hgt = np.where(bad, np.nan, hgt).astype(np.float32)
                sea = np.where(bad, np.nan + 0j, sea).astype(np.complex64)
                return np.stack([vel.astype(np.complex64),
                                 hgt.astype(np.complex64),
                                 sea,
                                 gam.astype(np.complex64),
                                 rms.astype(np.complex64)], axis=0)

            args = (d_dask,) if w_dask is None else (d_dask, w_dask)
            res = da.map_blocks(_blk, *args, dtype=np.complex64,
                                drop_axis=0, new_axis=0,
                                chunks=(5,) + d_dask.chunks[1:],
                                meta=np.empty((0, 0, 0), np.complex64))

            coords = {kk: vv for kk, vv in da_.coords.items()
                      if kk in ('y', 'x', 'spatial_ref')}
            mvars = {}
            for nm_, arr_ in (('velocity', res[0].real.astype(np.float32)),
                              ('height', res[1].real.astype(np.float32)),
                              ('seasonal', res[2].astype(np.complex64)),
                              ('coherence', res[3].real.astype(np.float32)),
                              ('rmse', res[4].real.astype(np.float32))):
                mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            mds = mds.assign_coords(date=np.datetime64(int(t0_day), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[key] = mds
        return Batch(model_result)


    def predict(self, model, baseline: str = 'BPR',
                ref=None) -> 'Batch':
        """
        Phase predicted by a model, on THIS batch's own pairs or dates.

        The inverse of fit1d(), and the way to see what a fit did not explain:

        >>> model = phase - phase.gaussian(wavelength=40000)
        >>> model = model.fit1d()
        >>> noise = phase - phase.predict(model)

        A PAIRS batch gets PER-PAIR phase, rebuilt in the pair convention the
        model is published in, so it subtracts from the pairs directly:

            phi_p = velocity*dt_p - height*ele2phase_p
                    + Re(seasonal)*dcos_p + Im(seasonal)*dsin_p

        A per-DATE batch gets per-date phase, which runs OPPOSITE in velocity
        and seasonal -- an SLC phase is -(4pi/lambda)r while a pair
        ref*conj(rep) carries +m2r*dr -- and identical in height, whose
        per-date term +height*ele2phase_d already differences to
        -height*ele2phase_p. That asymmetry is why no global sign converts one
        into the other.

        The epoch comes from the model's scalar `date` coordinate, which every
        fit records, so a model fitted against one origin is never removed
        against another. Only the annual cares, and it cares completely: a
        mismatch of delta years leaves a residual annual of
        2|sin(pi delta)|*|seasonal|.

        Parameters
        ----------
        model : Batch
            Output of fit1d() or fit1d_network(): `velocity`, `height`, `seasonal`.
        baseline : str
            Per-pair or per-date perpendicular baseline, default 'BPR'. None
            drops the height term -- a prediction of the deformation alone,
            which is what to ask for when the topographic phase is to stay in
            the data. ABSENT RAISES, as in fit1d(): removing less than intended
            is invisible in the result and shows up only in the residual.

        ref : None, int, str or datetime
            Which acquisition reads zero in the returned per-date series.
            None keeps the model's own anchor -- the master, where B_perp is
            smallest and the height term vanishes with the baseline that
            carries it. That is the right anchor for the FIT and an awkward one
            to look at, since the series then runs negative before the master
            and positive after; `ref=0` re-reads the same model from the first
            acquisition, and a date string or datetime picks any other. Same
            spelling as S1.transform(ref=...).

            IT IS A DISPLAY CHOICE AND NOTHING MORE. Re-referencing shifts
            every date by one per-pixel constant, so every DIFFERENCE is
            untouched -- pair predictions, removals and velocities come out
            bit-identical whatever `ref` says. The fit is not re-referenced,
            only the picture. Meaningless on a PAIRS batch, where a pair is
            already a difference and the reference cancels, so passing it there
            raises rather than being quietly ignored.
        Returns
        -------
        Batch
            Real radians, on the MODEL's grid, over this batch's pairs or
            dates. The scatterer's own constant is not part of any model, so
            the prediction is right up to one constant per pixel -- which is
            what you want for removal, and what a difference cancels anyway.
        """
        import numpy as np
        import pandas as pd
        import xarray as xr

        BatchCore._require_model(model, 'predict')
        # radar_wavelength and the radar geometry are the Dataset's (N81)
        BatchCore._needs_dataset(self, 'predict', BatchCore._GEOMETRY)
        out = {}
        _ep_all = None  # computed on first use: once per call, so a missing ref_height warns once and not once per burst
        for key in self.keys():
            ds = self[key]
            if key not in model:
                raise KeyError(f"predict(): the model has no burst '{key}'.")
            mds = model[key]
            missing = [v for v in ('velocity', 'height') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit1d() or fit1d_network().")

            grids = [v for v in ds.data_vars
                     if 'y' in ds[v].dims and 'x' in ds[v].dims
                     and ('pair' in ds[v].dims or 'date' in ds[v].dims)]
            if not grids:
                raise TypeError(
                    f"predict() found no (pair|date, y, x) variables in '{key}'.")
            pol = grids[0]
            da_ = ds[pol]
            per_pair = 'pair' in da_.dims

            if 'radar_wavelength' not in ds:
                raise KeyError(f"predict() needs 'radar_wavelength' on '{key}'.")
            lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
            meter2rad = 4.0 * np.pi / lam

            # the epoch the model was fitted against
            t0 = None
            if 'date' in mds.coords and mds.coords['date'].ndim == 0:
                t0 = float(np.asarray(mds.coords['date'].values)
                           .astype('datetime64[D]').astype(np.float64))

            if per_pair:
                rd = (np.asarray(pd.to_datetime(da_.coords['ref'].values).values)
                      .astype('datetime64[D]').astype(np.float64))
                pd_ = (np.asarray(pd.to_datetime(da_.coords['rep'].values).values)
                       .astype('datetime64[D]').astype(np.float64))
                if t0 is None:
                    t0 = float(np.median(np.unique(np.concatenate([rd, pd_]))))
                t_ref = (rd - t0) / 365.25
                t_rep = (pd_ - t0) / 365.25
                bp = None
                for src in (da_.coords, ds):
                    if baseline and baseline in src:
                        bp = np.asarray(src[baseline].values, dtype=float).ravel()
                        break
            else:
                dday = (np.asarray(da_.coords['date'].values)
                        .astype('datetime64[D]').astype(np.float64))
                bp = None
                if baseline and baseline in ds:
                    bp = np.asarray(ds[baseline].values, dtype=float)
                    while bp.ndim > 1:
                        bp = np.nanmean(bp, axis=-1)
                if t0 is None:
                    _b = bp if (bp is not None and bp.shape == dday.shape) \
                        else np.zeros_like(dday)
                    t0 = float(dday[int(np.argmin(np.abs(_b)))])
                t_d = (dday - t0) / 365.25

            # ele2phase = B_perp / median(R sin(incidence)), from
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            #
            # ABSENT AND None ARE DIFFERENT ANSWERS, as in fit1d(). None asks
            # for a prediction without the topographic term; a baseline that
            # went missing -- xarray drops a non-index coordinate the moment two
            # operands disagree on its value, which is what combining two bands
            # or two stacks does -- would quietly remove LESS than the caller
            # believes, and the term left behind shows up only as a
            # baseline-correlated pattern in the residual.
            n_obs = da_.sizes['pair' if per_pair else 'date']
            if baseline and bp is None:
                _have = sorted(set(da_.coords) | set(ds.data_vars))
                raise KeyError(
                    f"predict() found no {baseline!r} for '{key}' to rebuild the "
                    f"height term; the dataset carries {_have}. Re-attach the one "
                    f"you mean with .assign_coords({baseline}=('pair', values)), "
                    f"or pass baseline=None to predict without topography.")
            if bp is not None and bp.shape != (n_obs,):
                raise ValueError(
                    f"predict(): {baseline!r} has {bp.shape[0]} value(s) for "
                    f"'{key}' but the batch has {n_obs} "
                    f"{'pair' if per_pair else 'date'}(s). The height term needs "
                    f"one baseline per observation.")
            e2p = None
            if bp is not None:
                if _ep_all is None:
                    _ep_all = Batch._elevation_phase_approximate(self)
                _fac = _ep_all[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)

            import dask.array as da
            vel = mds['velocity'].data
            hgt = mds['height'].data
            # OPTIONAL: fit1d_network() fits an annual term and fit1d() does not, and
            # a model without the variable HAS none rather than an unknown one.
            sea = mds['seasonal'].data if 'seasonal' in mds else None
            planes = []
            for i in range(n_obs):
                if per_pair:
                    # the pair convention, exactly fit1d's columns
                    phi = vel * float(t_rep[i] - t_ref[i])
                    if e2p is not None:
                        phi = phi - da.nan_to_num(hgt) * float(e2p[i])
                    dcos = float(np.cos(2 * np.pi * t_rep[i])
                                 - np.cos(2 * np.pi * t_ref[i]))
                    dsin = float(np.sin(2 * np.pi * t_rep[i])
                                 - np.sin(2 * np.pi * t_ref[i]))
                    phi = phi + (sea.real * dcos + sea.imag * dsin)
                else:
                    # per-date, published as master*conj(date) so it is the
                    # same radians displacement_los() converts -- identical in
                    # form to the pair branch, differenced against the epoch
                    phi = vel * float(t_d[i])
                    if e2p is not None:
                        phi = phi - da.nan_to_num(hgt) * float(e2p[i])
                    if sea is not None:
                        car = np.exp(2j * np.pi * float(t_d[i]))
                        phi = phi + (sea.real * car.real + sea.imag * car.imag)
                planes.append(phi.astype(np.float32))
            pred = da.stack(planes, axis=0)
            if ref is not None:
                if per_pair:
                    raise TypeError(
                        "ref re-references a per-DATE series, and a pair is "
                        "already a difference in which the reference cancels. "
                        "Call it on the per-date stack instead.")
                _i = Batch._ref_index(ref, da_.coords['date'].values)
                pred = pred - pred[_i]

            dim = 'pair' if per_pair else 'date'
            # BARE axes: the model's scalar `date` coordinate travels with any
            # DataArray taken from it and then collides with a `date` dimension
            coords = {dim: np.asarray(da_.coords[dim].values),
                      'y': np.asarray(mds.coords['y'].values),
                      'x': np.asarray(mds.coords['x'].values)}
            pds = xr.Dataset({pol: xr.DataArray(pred, dims=(dim, 'y', 'x'),
                                                coords=coords)}, attrs=ds.attrs)
            if per_pair:
                for c in ('ref', 'rep'):
                    if c in da_.coords:
                        pds = pds.assign_coords(
                            {c: ('pair', np.asarray(da_.coords[c].values))})
            sref = (mds['spatial_ref'] if 'spatial_ref' in mds.coords
                    else (ds['spatial_ref'] if 'spatial_ref' in ds.coords else None))
            if sref is not None:
                pds = pds.assign_coords(
                    spatial_ref=sref.drop_vars(list(sref.coords), errors='ignore'))
            out[key] = pds
        return Batch(out)

    def mix(self, model, baseline: str = 'BPR') -> 'Batch':
        """
        Per-pair displacement from a fit1d()/fit1d_network() model MIXED with the raw signal.

        The pair form of BatchComplex.mix(). Returns the model's displacement
        carrying the residual the interferograms actually have, rather than the
        smooth curve predict() draws through them:

            mix = predict(model, baseline=None) + (self - predict(model))

        PAIRS ONLY, and that is not a restriction but what the data is: an
        unwrapped interferogram is a difference of two acquisitions by
        construction, so this batch is pairwise or it is not unwrapped phase.
        A per-date batch is complex and belongs to BatchComplex.mix(), which
        differs in two ways that matter -- it forms the residual as a complex
        product, and it has a per-pixel constant to divide out first. Here there
        is none: a pair is already a difference, so the scatterer's own phase has
        cancelled before this method ever sees it, which is the same reason
        predict() refuses `ref` on a pairs batch.

        With the residual left unwrapped the whole thing reduces to
        `self - (predict('BPR') - predict(None))` -- the interferograms with the
        fitted topographic phase taken out, and every departure from the model
        kept at whatever size it really is. That is the useful product here: the
        complex form has to reconstruct an absolute scale the wrapped data never
        had, while unwrapped pairs already carry theirs.

        NOTHING IS WRAPPED HERE, unlike the complex form. There the phase lives
        on the unit circle and the residual can only be read as an angle; here it
        is already unwrapped, so the residual is a plain difference and stays
        whatever size it is. Wrapping it would fold every departure larger than
        half a cycle back into (-pi, pi] and throw away exactly the large
        deformation the unwrapper was run to recover.

        Parameters
        ----------
        model : Batch
            Output of fit1d() or fit1d_network(): `velocity`, `height`, `seasonal`.
        baseline : str or None
            Per-pair perpendicular baseline carrying the height term. Default
            'BPR'. Pass None when topography is already gone from these pairs,
            and the returned phase is then the reconstruction of what is here.

        Returns
        -------
        Batch
            Displacement phase in RADIANS, one plane per pair, in the pair
            convention the model is published in -- the same object predict()
            returns and displacement_los() converts.

        Examples
        --------
        >>> model = phase.fit1d(weight=corr)
        >>> disp  = phase.mix(model)
        >>> disp.displacement_los(stack.transform())
        """
        BatchCore._require_model(model, 'mix')
        # predict() reads the Dataset's radar geometry (N81)
        BatchCore._needs_dataset(self, 'mix', BatchCore._GEOMETRY)
        for key, ds in self.items():
            if 'pair' not in ds.dims:
                raise TypeError(
                    f"mix() is for PAIRS; burst '{key}' carries "
                    f"{sorted(ds.dims)}. An unwrapped interferogram is pairwise "
                    f"by construction -- a per-date complex stack wants "
                    f"BatchComplex.mix(), which removes a per-pixel constant "
                    f"this form has no need of.")
        # `full` CARRIES THE HEIGHT TERM AND `disp` DOES NOT, and the gap between
        # them is what leaves the series: with the residual unwrapped this reduces
        # to self - (full - disp), the observation minus the fitted topography.
        # Predict both rather than subtracting a topography by hand, so the two
        # can never disagree about the geometry or the epoch.
        full = self.predict(model, baseline=baseline)
        disp = (full if baseline is None
                else self.predict(model, baseline=None))
        # PLUS, not minus, and no constant removed -- both differ from
        # BatchComplex.mix() and both follow from the pair convention: predict()
        # publishes per-pair phase that "subtracts from the pairs directly", so
        # the residual is self - full and goes back on the same way round.
        return disp + (self - full)


    def elevation_phase(self) -> "Batch":
        """Radians of phase per metre of elevation per metre of perpendicular baseline.

        `4 pi kN / (lambda R sin(incidence))` per pixel on the WGS84 ellipsoid, at the
        terrain height (the midpoint of the `ele` actual_range attribute): the sensitivity
        elevation() divides by in its one-division form, where the per-pixel
        perpendicular baseline of a pair comes from the baseline triples. Multiply by
        B_perp and a height to get phase; divide a phase by it and B_perp to get a height.

        SLANT RANGE AND THE SINE OF THE INCIDENCE ANGLE, matching GMTSAR (`sbas.c:186`,
        `phase2topo.c`); kN is the ellipsoid normal's factor, 1 on a sphere.

        READS NO RASTER: x, y and the reference date's numbers only (see incidence()).
        Call it on the stack's transform() view: `stack.transform().elevation_phase()`.
        """
        from . import utils_geometry as ug
        BatchCore._needs_dataset(self, 'elevation_phase', ug.SCALARS, 'Call it on stack.transform().')
        out: dict[str, xr.Dataset] = {}
        no_range: list = []
        for key, ds in self.items():
            geo = _stack_geometry(ds, key, 'elevation_phase()', 'Call it on stack.transform().', no_range)[0]
            fac = _geometry_grid(ug.elevation_phase_block, geo, ds.y.values, ds.x.values, _grid_chunks(ds))
            res = xr.Dataset({'elevation_phase': xr.DataArray(fac, dims=('y', 'x'),
                                                              coords={'y': ds.y, 'x': ds.x})})
            res.attrs = ds.attrs
            if 'spatial_ref' in ds.coords:
                res = res.assign_coords(spatial_ref=ds.spatial_ref)
            out[key] = res
        _warn_no_actual_range('elevation_phase()', no_range)
        return Batch(out)

    def _elevation_phase_approximate(self) -> dict:
        """`elevation_phase()` at the centre of the PROCESSED area.

        `{burst_id: value}`, and it reads NO radar grids. `azi`, `rng` and
        `ele` are the special rasters `transform()` carries; a caller turning
        an ARGUMENT LIMIT into phase -- a height bound in metres into a lattice
        step -- must not depend on them, and PAIR data does not have them at
        all. Results are a different matter: they come out in radians and
        `displacement_los()` converts them with the full pixelwise geometry.

        WHERE THE CENTRE COMES FROM. `geometry` is the burst's exact radar
        extent as a polygon, and its four corners have known radar coordinates
        by construction -- (0, 0) to (num_lines-1, num_rng_bins-1), the
        0-based line and bin of the corner pixel centres.
        Inverting that mapping turns any map coordinate into (azi, rng), so the
        centre of the geocoded grid gives the centre of what was actually
        processed. That matters because `geometry` describes the WHOLE burst
        while a bbox may have cropped the data to a small part of it, and the
        burst centre would then sit outside the data entirely.

        The inverse mapping need not be accurate in range BINS -- the slant
        range barely moves across many of them -- only good enough to land in
        the processed area rather than at the far side of the burst.
        """
        import numpy as np
        return {key: (4.0 * np.pi / wavelength) / (slant_range * sin_inc)
                for key, (wavelength, slant_range, sin_inc)
                in Batch._centre_geometry_approximate(self, 'scene-centre elevation_phase()').items()}

    def _incidence_approximate(self) -> dict:
        """The incidence angle (radians) at the centre of the PROCESSED area, `{burst_id: value}`: the
        counterpart of `_elevation_phase_approximate()`, from the same metadata-only centre and the same
        sphere (arcsin of its sin(incidence)), equally approximate and reading NO radar grids. The
        displacement conversions of a burst without (y, x) -- a point's series, an area mean -- use it,
        one value per burst; gridded data take the per-pixel incidence()."""
        import numpy as np
        return {key: float(np.arcsin(sin_inc))
                for key, (_, _, sin_inc) in Batch._centre_geometry_approximate(self, 'scene-centre incidence()').items()}

    def _centre_geometry_approximate(self, where) -> dict:
        """`{burst_id: (wavelength, slant range, sin(incidence))}` at the centre of the PROCESSED area, on the
        sphere of earth_radius at ref_height: the shared geometry of `_elevation_phase_approximate()` (see there
        for where the centre comes from) and `_incidence_approximate()`. Metadata only, NO radar grids."""
        import numpy as np

        c_light = 299792458.0
        out: dict = {}
        ref_heights = _ref_heights(self, where)
        for key, ds in self.items():
            g = lambda n: float(np.asarray(ds[n].values).ravel().mean())
            wavelength = g('radar_wavelength')
            near_range = g('near_range')
            rng_samp_rate = g('rng_samp_rate')
            earth_radius = g('earth_radius')
            num_lines = g('num_lines')
            num_rng_bins = g('num_rng_bins')
            sc_height = 0.5 * (g('SC_height_start') + g('SC_height_end'))

            # The stack's burst `geometry` and CRS are the stored numbers of every real product: there is no
            # fallback to a burst centre (N138: none of 66 real calls on the pygmtsar and ecef stores reached one)
            rng_c = num_rng_bins / 2.0
            from shapely import wkt as _wkt
            from pyproj import Transformer as _Tr
            poly = _wkt.loads(str(np.asarray(ds['geometry'].values).ravel()[0]))
            lon, lat = (np.asarray(v) for v in poly.exterior.coords.xy)
            px, py = _Tr.from_crs('EPSG:4326', ds.rio.crs, always_xy=True).transform(lon[:4], lat[:4])
            a_c = np.array([0.0, 0.0, num_lines - 1, num_lines - 1])
            r_c = np.array([0.0, num_rng_bins - 1,
                            num_rng_bins - 1, 0.0])
            M, b = [], []
            for x_, y_, a_, r_ in zip(px, py, a_c, r_c):
                M.append([x_, y_, 1, 0, 0, 0, -a_ * x_, -a_ * y_]); b.append(a_)
                M.append([0, 0, 0, x_, y_, 1, -r_ * x_, -r_ * y_]); b.append(r_)
            h = np.linalg.solve(np.asarray(M, float), np.asarray(b, float))
            xc = 0.5 * (float(ds.x.min()) + float(ds.x.max()))
            yc = 0.5 * (float(ds.y.min()) + float(ds.y.max()))
            den = h[6] * xc + h[7] * yc + 1.0
            cand = (h[3] * xc + h[4] * yc + h[5]) / den
            if np.isfinite(cand) and 0.0 <= cand <= num_rng_bins:
                rng_c = float(cand)

            slant_range = near_range + rng_c * (c_light / (2.0 * rng_samp_rate))
            ground_dist = earth_radius + ref_heights[key]
            sat_dist = earth_radius + sc_height
            cos_earth = np.clip(
                (ground_dist ** 2 + sat_dist ** 2 - slant_range ** 2)
                / (2.0 * ground_dist * sat_dist), -1.0, 1.0)
            sin_inc = np.clip(
                sat_dist * np.sin(np.arccos(cos_earth)) / slant_range, -1.0, 1.0)
            out[key] = (wavelength, slant_range, sin_inc)
        return out

    def incidence(self) -> "Batch":
        """Incidence angle (radians) on the WGS84 ellipsoid, per pixel of the grid.

        READS NO RASTER (no azi, rng or ele): the satellite track is rebuilt from the
        REFERENCE date's numbers (near_range, rng_samp_rate, earth_radius,
        SC_height_start/end, num_lines, num_rng_bins, lookdir) and the exact crop polygon
        `geometry`, each pixel's x, y is mapped onto it at the terrain height (the
        midpoint of the `ele` actual_range attribute), on nodes every 128 pixels and
        bilinear in between. Measured against the exact orbit geometry: 0.013-0.048 deg
        median, at most 0.061 deg; at most 0.11 % in the vertical and east-west
        projections. The value covers the whole grid: no raster masks it.

        Call it on the stack (`stack.incidence()`) or its transform() view.
        """
        from . import utils_geometry as ug
        BatchCore._needs_dataset(self, 'incidence', ug.SCALARS, 'Call it on the stack: stack.incidence().')
        crs = self.crs
        out: dict[str, xr.Dataset] = {}
        no_range: list = []
        for key, ds in self.items():
            geo = _stack_geometry(ds, key, 'incidence()', 'Call it on the stack: stack.incidence().', no_range)[0]
            inc = _geometry_grid(ug.incidence_block, geo, ds.y.values, ds.x.values, _grid_chunks(ds))
            result_ds = xr.Dataset({'incidence': xr.DataArray(inc, dims=('y', 'x'), coords={'y': ds.y, 'x': ds.x})})
            result_ds.attrs = ds.attrs
            result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        _warn_no_actual_range('incidence()', no_range)
        return Batch(out)

    def look(self) -> "Batch":
        """Look azimuth (degrees, clockwise from geodetic north, satellite to target, in [0, 360)) per pixel of the grid.

        READS NO RASTER, as incidence(): the satellite position on each pixel's zero-Doppler line comes from the orbit numbers
        of the REFERENCE date and the exact crop polygon, at the terrain height (the midpoint of the `ele` actual_range), on
        nodes every 128 pixels and bilinear in between (in its sine and cosine, no wrap at 0 and 360). It is the orbit
        geometry, not the polygon heading. A decimated transform() gives the look on its own decimated grid.

        Call it on the stack (`stack.look()`) or its transform() view.
        """
        from . import utils_geometry as ug
        BatchCore._needs_dataset(self, 'look', ug.SCALARS, 'Call it on the stack: stack.look().')
        crs = self.crs
        out: dict[str, xr.Dataset] = {}
        no_range: list = []
        for key, ds in self.items():
            geo = _stack_geometry(ds, key, 'look()', 'Call it on the stack: stack.look().', no_range)[0]
            look = _geometry_grid(ug.look_block, geo, ds.y.values, ds.x.values, _grid_chunks(ds))
            result_ds = xr.Dataset({'look': xr.DataArray(look, dims=('y', 'x'), coords={'y': ds.y, 'x': ds.x})})
            result_ds.attrs = ds.attrs
            result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        _warn_no_actual_range('look()', no_range)
        return Batch(out)

    def aspect(self) -> "Batch":
        """Terrain slope and aspect from `ele`, by the Horn (1981) 3 x 3 kernel.

        Both attributes come out of the same pass, so both are returned, in
        DEGREES:

          slope  -- steepness from horizontal, zero on level ground;
          aspect -- the DOWNSLOPE direction clockwise from north, so a face
                    falling east reads 90. NaN where the ground is level and
                    there is no direction to report, rather than the 0 that
                    would read as due north.

        HORN, NOT A CENTRAL DIFFERENCE. The kernel weights the diagonals as
        well as the cardinal neighbours, which is steadier wherever the DEM is
        posted finer than its own resolution -- `ele` is resampled onto the
        radar grid, so it always is. A central difference sees only the four
        cardinal cells and passes the resampling's own structure straight
        through.

        The spacing comes from the y/x coordinates and each axis divides by
        its OWN, signed, so an anisotropic posting is exact rather than
        approximated by a mean, and northing running either way down the array
        gives the same answer.

        The kernel has no answer for the border row and column: those are NaN,
        not a one-sided estimate, which is a different quantity wearing the
        same name.

        `incidence()` returns radians; these return degrees, which is what
        every DEM tool reports (gdaldem, xarray-spatial, xDEM, GRASS) and what
        a reader of a slope map expects.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        import rioxarray  # for .rio accessor

        crs = self.crs

        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            # the transform's `ele`, or a DataArray batch of it, stack.ele (N81)
            if isinstance(tfm, xr.DataArray):
                z = tfm if tfm.name == 'ele' else None
            else:
                z = tfm['ele'] if 'ele' in tfm.variables else None
            if z is None:
                raise KeyError(
                    f'aspect(): {key!r} carries no {"ele"!r} to differentiate. '
                    'Call it on transform(), which delivers the DEM.')
            if z.dims[-2:] != ('y', 'x'):
                raise ValueError(
                    f'aspect(): {key!r} has ele on {z.dims}, not a (y, x) '
                    'raster; there is no map grid to take a gradient on.')
            yv = np.asarray(z.y.values, dtype=np.float64)
            xv = np.asarray(z.x.values, dtype=np.float64)
            if yv.size < 3 or xv.size < 3:
                raise ValueError(
                    f'aspect(): the 3 x 3 kernel needs three posts on each '
                    f'axis; {key!r} has y={yv.size}, x={xv.size}.')
            # SIGNED, because the derivative is with respect to the COORDINATE
            # and not the index
            dy = float(yv[1] - yv[0])
            dx = float(xv[1] - xv[0])

            # ONE OVERLAPPED PASS. The kernel reaches one pixel each way, so
            # that is the halo; NaN at the raster's own border comes from the
            # NaN the boundary pads with, and needs no special case. Float
            # first, since an integer DEM cannot carry that pad.
            dsk = z.data
            if not isinstance(dsk, da.Array):
                dsk = da.from_array(dsk, chunks=dsk.shape)
            dsk = dsk.astype('float32')
            depth = ({0: 0, 1: 1, 2: 1} if z.ndim == 3 else {0: 1, 1: 1})
            grad = da.map_overlap(
                _horn_gradient_2d_for_dask, dsk,
                depth=depth, boundary=np.nan, trim=True, dtype=np.complex64,
                dy=dy, dx=dx)

            mag = da.abs(grad)
            slope = da.rad2deg(da.arctan(mag)).astype('float32')
            # THE GRADIENT IS NEGATED because aspect names where the surface
            # FALLS, not where it rises; atan2(east, north) is then already
            # clockwise from north, which is the compass convention reported
            # everywhere else
            aspect = (da.rad2deg(da.arctan2(-grad.real, -grad.imag)) % 360.0)
            aspect = da.where(mag > 0, aspect, np.nan).astype('float32')

            # TWO NAMED OUTPUTS, a Dataset whatever the input form; a DataArray's own
            # attributes describe `ele`, not the result (N81)
            result_ds = xr.Dataset({
                'slope': xr.DataArray(slope, dims=z.dims, coords=z.coords),
                'aspect': xr.DataArray(aspect, dims=z.dims, coords=z.coords)})
            if isinstance(tfm, xr.Dataset):
                result_ds.attrs = tfm.attrs
            # Preserve CRS
            result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        return Batch(out)

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) to convert phase to complex phasor.

        Parameters
        ----------
        sign : int, optional
            Sign of the exponent. Default is -1 for exp(-1j * phase).

        Returns
        -------
        BatchComplex
            Complex phasor representation.
        """
        import xarray as xr
        return BatchComplex(self.map_da(lambda da: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def displacement_los(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing mission constants (radar_wavelength),
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            LOS displacement grids (meters), lazily scaled by the mission wavelength.

        Examples
        --------
        >>> disp_los = unwrapped.displacement_los(stack)
        >>> disp_los = unwrapped.displacement_los(stack.transform())
        """
        import numpy as np
        import xarray as xr
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        if not transform:
            raise ValueError('transform must contain at least one burst with radar_wavelength')

        transform_first = next(iter(transform.values()))

        def _scalar_from_ds(ds, name: str):
            if name in ds:
                var = ds[name]
                if var.ndim == 0:
                    return var.item()
                elif var.ndim >= 1:
                    values = var.values.flatten()
                    unique = np.unique(values)
                    if len(unique) != 1:
                        raise ValueError(f'{name} has multiple distinct values: {unique}')
                    return unique[0]
            return ds.attrs.get(name)

        wavelength = _scalar_from_ds(transform_first, 'radar_wavelength')
        if wavelength is None:
            raise KeyError('Missing radar_wavelength in transform')

        # scale factor from phase in radians to displacement in meters
        # constant is negative to make LOS = -1 * range change
        scale = -float(wavelength) / (4 * np.pi)

        out: dict[str, xr.Dataset] = {}
        for key, phase_ds in self.items():
            disp_vars: dict[str, xr.DataArray] = {}
            # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
            # radar_wavelength, burst ids and the rest ride along and are
            # carried through untouched; multiplying a <U43 burst name by
            # a float is the UFuncTypeError this prevents. A Dataset without
            # any (y, x) grid RAISES (N81): passed through, its phase came
            # back in radians labelled metres.
            # A DATAARRAY IS PROCESSED AS IT IS (N81): with or without (y, x)
            # -- a point's series, an area mean -- it is phase, converted by
            # the same scale.
            grids = BatchCore._grid_vars(phase_ds) if isinstance(phase_ds, xr.Dataset) else None
            # a Dataset's variables, or a DataArray as its own one (N81)
            for var_name, data in BatchCore._vars_of(phase_ds).items():
                if grids is not None and var_name not in grids:
                    disp_vars[var_name] = data
                    continue
                disp = (data * scale).astype('float32')
                disp_vars[var_name] = disp
            # a DataArray in, a DataArray out (N81)
            out[key] = BatchCore._form(phase_ds, disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, transform: 'Batch | Stack', func, suffix: str = '') -> 'Batch':
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin).

        The incidence is computed ON THE PHASE'S OWN GRID (its y, x and chunks) from the geometry source -- the
        stack or its transform() view -- with no raster read and nothing reindexed (see incidence()).

        A DATAARRAY WITHOUT (y, x) -- a point's series, an area mean -- is converted PER BURST by the incidence at
        the centre of that source's processed area (_incidence_approximate(), the approximate burst-centre path of
        the scalar height per fringe): one value per burst, the per-burst sensitivity. A Dataset without any grid
        raises in displacement_los()."""
        import xarray as xr
        from . import utils_geometry as ug

        los_batch = self.displacement_los(transform)

        out: dict[str, xr.Dataset] = {}
        # the burst-centre incidence of the bursts without (y, x), from the geometry source as it is passed
        centre = [key for key, los_ds in los_batch.items()
                  if isinstance(los_ds, xr.DataArray) and not BatchCore._is_grid(los_ds)]
        for key in centre:
            if key not in transform:
                raise KeyError(f'Missing incidence for key: {key}')
        centre = Batch._incidence_approximate({key: dict.__getitem__(transform, key) for key in centre}) \
            if centre else {}

        no_range: list = []
        for key, los_ds in los_batch.items():
            if key not in transform:
                raise KeyError(f'Missing incidence for key: {key}')
            if key in centre:
                comp = (los_ds / float(func(xr.DataArray(centre[key])))).astype('float32')
                out[key] = BatchCore._form(los_ds, {suffix: comp})
                continue
            geo = _stack_geometry(dict.__getitem__(transform, key), key, f'displacement_{suffix}()',
                                  f'Pass the stack: x.displacement_{suffix}(stack).', no_range)[0]
            comp_vars: dict[str, xr.DataArray] = {}
            # a Dataset's variables, or a DataArray as its own one (N81)
            los_vars = BatchCore._vars_of(los_ds)
            # the naming counts the GRIDS: the radar metadata a Dataset carries are not results (N133)
            n_grids = sum(1 for d in los_vars.values() if "y" in d.dims and "x" in d.dims)

            for var_name, data in los_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    comp_vars[var_name] = data
                    continue
                # the incidence on the data's own grid and chunks: the geometry is bent to the data, never the
                # other way round
                chunks = (tuple(data.chunks[data.dims.index('y')]), tuple(data.chunks[data.dims.index('x')])) \
                    if data.chunks is not None else ((data.sizes['y'],), (data.sizes['x'],))
                inc = _geometry_grid(ug.incidence_block, geo, data.y.values, data.x.values, chunks)
                incidence = xr.DataArray(inc, dims=('y', 'x'), coords={'y': data.y, 'x': data.x})

                comp = (data / func(incidence)).astype('float32')

                if n_grids == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            # a DataArray in, a DataArray out (N81), named as a one-variable Dataset names it
            out[key] = BatchCore._form(los_ds, comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        _warn_no_actual_range(f'displacement_{suffix}()', no_range)
        return Batch(out)

    def displacement_vertical(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute vertical displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Stack or Batch
            The stack, or its transform() view: the wavelength and the per-date
            geometry incidence() reads, no raster.

        Returns
        -------
        Batch
            Vertical displacement grids (meters).

        Examples
        --------
        >>> disp_v = unwrapped.displacement_vertical(stack)
        >>> disp_v = unwrapped.displacement_vertical(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute east-west displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Stack or Batch
            The stack, or its transform() view: the wavelength and the per-date
            geometry incidence() reads, no raster.

        Returns
        -------
        Batch
            East-west displacement grids (meters).

        Examples
        --------
        >>> disp_ew = unwrapped.displacement_eastwest(stack)
        >>> disp_ew = unwrapped.displacement_eastwest(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.sin, suffix='eastwest')

    def elevation(self, stack: 'Stack | Batch | None' = None, baseline: float | None = None) -> 'Batch':
        """Compute elevation (meters) from unwrapped phase grids, with the STACK.

        The phase gives its values, x, y and each pair's two dates; the stack gives everything
        that is not pairwise: each date's baseline (baseline_* and alpha_*, the baseline
        vector to the reference satellite at the scene start and end, linear in between), the
        REFERENCE date's numbers, the exact crop polygon, the CRS and the `ele` actual_range
        ATTRIBUTE. NO RASTER IS READ: the satellite track is rebuilt on WGS84 from those
        numbers and the polygon, the geometry is evaluated on nodes every 128 pixels and
        applied in one pass. A value comes out wherever the phase is finite.

        With the stack, the linearized joint solve: the exact secant of the two dates' range
        changes with the position moved to the solved height, three steps. Flat-earth mode
        (remove_topo_phase=False) gives the absolute height about the reference height; DEM
        mode gives the residual height about the DEM, anchored at the actual_range midpoint.

        WITHOUT THE STACK (a transform, or nothing): ONE WARNING, and the one-division form
        from what the phase itself carries -- each pair's baseline triple difference
        (pairs()), its own first-date numbers, polygon and CRS, and the transform's `ele`
        actual_range attribute when one is given (else the terrain height is taken as 0).

        Parameters
        ----------
        stack : Stack
            The stack the pairs were made from. A transform batch (stack.transform()) or
            None falls back to the one-division form with a WARNING.
        baseline : None
            Not for grids, whose baselines come from the triples; the scalar height of
            ambiguity is stack.elevation(2*np.pi, baseline).

        Returns
        -------
        Batch
            Elevation grids as float32 datasets.

        Examples
        --------
        >>> elev = unwrapped.elevation(stack)
        >>> elev = stack.elevation(unwrapped)
        """
        from .Stack import Stack

        if baseline is not None:
            raise ValueError('ERROR: elevation() of grids takes no baseline: the stack gives the triples. '
                             'For a scalar phase use stack.elevation(2*np.pi, baseline).')
        joint = isinstance(stack, Stack)
        if not joint:
            # the workaround reads the phase Dataset's own numbers, which a DataArray does not carry (N81)
            if BatchCore._is_dataarray_batch(self):
                raise TypeError('ERROR: elevation() without the stack needs the phase Dataset. Pass the stack: '
                                'x.elevation(stack).')
        out: dict[str, xr.Dataset] = {}
        no_range: list = []

        for key, phase_ds in self.items():
            grids = {n: d for n, d in BatchCore._vars_of(phase_ds).items() if 'y' in d.dims and 'x' in d.dims}
            if joint:
                if key not in stack:
                    raise KeyError(f'Missing geometry for key: {key}')
                geo, bh, bv, dates = _stack_geometry(dict.__getitem__(stack, key), key, 'elevation()',
                                                       'Pass the stack: x.elevation(stack).', no_range)
                geos = [geo]
                pairs = (_pairs_of_stack(next(iter(grids.values())), key, bh, bv, dates, 'elevation()')
                         if grids else [])
            else:
                src = dict.get(stack, key) if isinstance(stack, dict) else None
                h_attr = _ele_midpoint(src, key, 'elevation()') if src is not None else None
                geos, pairs = _pair_geometry(phase_ds, key, 'elevation()', 0.0 if h_attr is None else h_attr)

            elev_vars: dict[str, xr.DataArray] = {}
            # a Dataset's variables, or a DataArray as its own one (N81): the pair dates are coordinates of the
            # pairs, which a DataArray carries
            for var_name, data in BatchCore._vars_of(phase_ds).items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if var_name not in grids:
                    elev_vars[var_name] = data
                    continue
                elev_vars[var_name] = _elevation_grid(data, geos, pairs, joint)

            # a DataArray in, a DataArray out (N81)
            out[key] = BatchCore._form(phase_ds, elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        # AFTER THE LOOP, which is where the errors of the bursts are raised: an old phase raises first, without the WARNING
        if not joint:
            print('WARNING: elevation() needs the stack: use intf.elevation(stack); using the pair baselines only.',
                  flush=True)
        _warn_no_actual_range('elevation()', no_range)
        return Batch(out)

    def stl(self, freq: str = 'W', periods: int = 52, robust: bool = False) -> 'Batch':
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Decomposes time series into trend, seasonal, and residual components.
        The Batch must have a 'date' dimension and ONE (y, x) variable: x['VV']
        or x[['VV']]; several raise (the outputs are named by component alone).

        Parameters
        ----------
        freq : str, optional
            Frequency string for resampling (default 'W' for weekly).
            Examples: '1W' for 1 week, '2W' for 2 weeks, '10d' for 10 days.
        periods : int, optional
            Number of periods for seasonal decomposition (default 52 for weekly data = 1 year).
        robust : bool, optional
            Whether to use robust fitting (slower but handles outliers better). Default False.

        Returns
        -------
        Batch
            Batch of Datasets with the 'trend', 'seasonal' and 'residual' variables.

        Examples
        --------
        >>> model = (unwrapped - unwrapped.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> stl_result = displacement['VV'].stl(freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, residual components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        from .Stack import Stack

        return Stack.stl(Stack(), self, freq=freq, periods=periods, robust=robust)

class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None, wrap: bool = True):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        # skip wrapping for intermediate objects like DatasetCoarsen
        if not wrap:
            dict.__init__(self, mapping or {})
        else:
            # FLOAT PHASE ONLY (N81): a bool, integer or complex grid is not a phase
            BatchCore._require_float(mapping, 'BatchWrap wraps float phase only')
            wrapped = {k: self.wrap(v) for k, v in (mapping or {}).items()}
            dict.__init__(self, wrapped)

    @classmethod
    def _keeps(cls, dtype) -> bool:
        """A result keeps the class when it is float phase (BatchCore._as_class)."""
        return dtype.kind == 'f'

    def _like(self, mapping):
        """A BatchWrap over `mapping` WITHOUT the wrap: the angle of a complex mean is in (-pi, pi] already, and a wrap
        would only move +pi to -pi (BatchCore._restored, downsample)."""
        return type(self)(mapping, wrap=False)

    @staticmethod
    def wrap(data):
        """Wrap the (y, x) planes to [-pi, pi]; leave everything else alone.

        Wrapping the whole Dataset would fold the radar metadata too --
        near_range 800000 m comes back as 2.3 rad, silently -- so only variables
        that actually carry phase are wrapped: the gridded ones, by the operator
        rule. A DataArray is wrapped whole: BatchWrap(x['VV']) is an explicit
        conversion. A selection from a BatchWrap is never converted: w['VV']
        stays a BatchWrap as stored, and w['BPR'], w.residual, w.y are plain
        Batch objects, so no later isel() or compute() wraps them
        (BatchCore._select_var).

        Only float phase gets here: BatchWrap(x) raises for a bool, integer or
        complex grid, and a mask of a BatchWrap (w > 0, np.isfinite(w)) is a
        plain Batch (BatchCore._as_class), so no rebuild wraps it into floats.
        """
        if isinstance(data, xr.Dataset):
            out = data.copy()
            for v in data.data_vars:
                da_ = data[v]
                if BatchCore._is_grid(da_):
                    out[v] = np.mod(da_ + np.pi, 2 * np.pi) - np.pi
            out.attrs = data.attrs
            return out
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def trend2d(self, *args, **kwargs):
        for key, ds in self.items():
            ny, nx = ds.sizes.get('y', 3), ds.sizes.get('x', 3)
            if min(ny, nx) < 3:
                raise ValueError(f"trend2d() needs a burst grid of at least 3 x 3 pixels, burst '{key}' is {ny} x {nx}.")
        raise TypeError(
            "trend2d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    # THE OPERATOR RULE (BatchCore._binary_vars): the grids are operated on, the
    # rest passes through; the result is wrapped again, as wrapped-phase
    # arithmetic is. `w + w` doubled radar_wavelength and `w * 2` raised on the
    # burst strings before.
    def __add__(self, other: Batch):
        if BatchCore._is_stack(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(v, other, _operator.add) for k, v in self.items()})
        keys = self.keys()
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.add) if k in other else self[k])
                           for k in keys})

    def __sub__(self, other: Batch):
        if BatchCore._is_stack(other):
            return NotImplemented
        import xarray as xr
        import operator as _operator
        # SUBTRACT THE GRIDS, CARRY THE REST. A whole-Dataset `ds - val` reaches
        # the burst ids and the radar geometry too: numpy refuses to subtract
        # <U43 strings, so align() raised, and radar_wavelength minus itself is
        # 0, so meter2rad became infinite where it did not. _binary_vars is the
        # one place that rule lives.
        _sub_grids = lambda d, v: BatchCore._binary_vars(d, v, _operator.sub)
        if isinstance(other, _SCALARS):
            return self._result({k: _sub_grids(v, other) for k, v in self.items()})
        # the per-pair coefficients of align() go through the DataArray polyval() correction
        return self._result({k: (self._sub_coeffs(k, self[k], other[k]) if k in other else self[k])
                           for k in self.keys()})

    def __mul__(self, other: Batch):
        if BatchCore._is_stack(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(v, other, _operator.mul) for k, v in self.items()})
        keys = self.keys()
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.mul) if k in other else self[k])
                           for k in keys})

    def __rmul__(self, other):
        if BatchCore._is_stack(other):
            return NotImplemented
        # scalar * batch  → map scalar * each dataset
        return self._result({k: BatchCore._binary_vars(v, other, lambda a, b: b * a) for k, v in self.items()})

    def __truediv__(self, other: Batch):
        if BatchCore._is_stack(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(v, other, _operator.truediv) for k, v in self.items()})
        keys = self.keys()
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.truediv) if k in other else self[k])
                           for k in keys})

    def sin(self, **kwargs) -> Batch:
        """
        Return a Batch of the sin(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.sin(da), **kwargs))

    def cos(self, **kwargs) -> Batch:
        """
        Return a Batch of the cos(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.cos(da), **kwargs))

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) like np.exp(-1j * intfs)

        - If sign = -1 (the default), this is exp(-1j * da).
        - If sign = +1, this is exp(+1j * da).
        """
        from .Batch import BatchComplex
        # the phasors are built as a plain dict: rebuilding a BatchWrap first
        # wrapped the complex values, which numpy refuses (interferogram(phase=BatchWrap))
        return BatchComplex(self._map_grids(lambda da, **kw: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.

        THE OPERATOR RULE: the grids are aggregated, every other variable passes
        through unchanged (the string metadata raised in astype() before).
        """
        #print ('wrap _agg')
        import inspect
        import xarray as xr
        import pandas as pd
        out = {}
        for key, obj in self.items():
            carry = {}
            if isinstance(obj, xr.Dataset):
                grids = BatchCore._grid_vars(obj)
                carry = {v: obj[v] for v in obj.data_vars if v not in grids}
                obj = obj[grids]
            elif hasattr(obj, 'obj') and isinstance(obj.obj, xr.Dataset):
                # a coarsen() window over the complex grids, the metadata carried by coarsen()
                carry = {v: obj.obj[v] for v in obj.obj.data_vars if not BatchCore._is_grid(obj.obj[v])}
            # get the aggregation function
            fn = getattr(obj, name)
            sig = inspect.signature(fn)

            # perform aggregation in complex domain
            if 'dim' in sig.parameters:
                # intfs.mean('pair').isel(0)
                #agg_result = fn(dim=dim, **kwargs)
                complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
                #fn_complex = getattr(complex_obj, name)
                #agg_result = fn_complex(dim=dim, **kwargs)
                if name in ('var', 'std'):
                    # |E[e^(iθ)]|
                    R = xr.ufuncs.abs(complex_obj.mean(dim=dim, **kwargs))
                    if name == 'var':
                        # 1 - |E[e^(iθ)]|
                        agg_result = (1 - R)
                    else:  # std
                        # √(-2 ln|E[e^(iθ)]|)
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    fn_complex = getattr(complex_obj, name)
                    agg_result = fn_complex(dim=dim, **kwargs)
                    # convert back to wrapped phase
                    agg_result = xr.ufuncs.angle(agg_result)
            else:
                # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
                # already in complex domain, see coarsen()
                if name in ('var', 'std'):
                    R = obj.mean(**kwargs)
                    if carry:
                        R = R[[v for v in R.data_vars if v not in carry]]
                    R = xr.ufuncs.abs(R)
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    if carry:
                        agg_result = agg_result[[v for v in agg_result.data_vars if v not in carry]]
                    agg_result = xr.ufuncs.angle(agg_result)

            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')
            for v, da_ in carry.items():
                out[key][v] = da_

        # each burst back in its own chunks, along the dims its result still has. The angle of every name but var and std is in
        # (-pi, pi] already (no wrap); var and std are not phase and keep the wrap
        return self._restored(out, wrap=name in ('var', 'std'))

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics: the
            # grids, by the operator rule, and every other variable carried
            if isinstance(ds, xr.Dataset):
                grids = BatchCore._grid_vars(ds)
                ds2 = xr.ufuncs.exp(1j * ds[grids].astype('float32'))
                meta = [v for v in ds.data_vars if v not in grids]
                if meta:
                    ds2 = ds2.assign({v: ds[v] for v in meta})
            else:
                ds2 = xr.ufuncs.exp(1j * ds.astype('float32'))
            # align each dimension to the coarse grid
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                if start is not None:
                    ds2 = ds2.isel({dim: slice(start, None)})
            # a burst with fewer samples than a window along a dimension has no full cell: it is dropped, as
            # BatchCore.coarsen() drops it (patch 0007)
            if any(ds2.sizes.get(dim, factor) < factor for dim, factor in window.items()):
                continue
            # blocks of whole windows, as BatchCore.coarsen()
            sizes = BatchCore._window_chunks(ds2, window)
            if sizes:
                ds2 = ds2.chunk(sizes)
            out[key] = ds2.coarsen(window, **kwargs)

        # wrap=False since these are DatasetCoarsen objects, not actual data
        return type(self)(out, wrap=False)

    def plot(
        self,
        cmap = 'gist_rainbow_r',
        alpha = 0.7,
        caption='Phase, [rad]',
        vmin=-np.pi,
        vmax=np.pi,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        return super().plot(*args, **kwargs)

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing for wrapped phase data.
    #     """
    #     return self.iexp().gaussian(*args, **kwargs).angle()

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
    #     then recombining via atan2.  No complex dtype ever created.
    #     """
    #     from .Batch import Batch
    #     import xarray as xr

    #     keep_attrs = kwargs.pop('keep_attrs', None)
    #     # build two Batches of the real sin and cos components and filter them
    #     sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
    #     cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

    #     # compute wrapped phase using np.arctan2
    #     out = {k: xr.Dataset({
    #         var: xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
    #         for var in sin[k].data_vars
    #     }) for k in self.keys()}

    #     return BatchWrap(out)

    def gaussian(self, *args, **kwargs):
        """
        Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
        then recombining via arctan2.

        Takes what BatchCore.gaussian() takes and in the same order --
        `gaussian(wavelength, weight=None, threshold=0.5, ...)`, the wavelength
        in metres, first and required.
        """
        from .Batch import Batch
        import xarray as xr

        keep_attrs = kwargs.pop('keep_attrs', False)
        data_vars = list(BatchCore._vars_of(next(iter(self.values()))))

        # build two Batches of the real sin and cos components and filter them
        sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
        cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

        # compute wrapped phase using arctan2
        out: dict[str, xr.Dataset] = {}
        for k in self.keys():
            if isinstance(self[k], xr.DataArray):
                # a DataArray in, a DataArray out (N81)
                src = self[k]
                if not ('y' in src.dims and 'x' in src.dims):
                    out[k] = src
                    continue
                phase = xr.ufuncs.arctan2(sin[k], cos[k]).astype('float32').rename(src.name)
                if keep_attrs:
                    phase.attrs = src.attrs.copy()
                out[k] = phase
                continue
            phase_vars = {}
            for var in data_vars:
                src = self[k][var]
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # arctan2 on a <U43 burst id is a TypeError, and the geometry
                # riding beside the phase is not something to smooth.
                if not ('y' in src.dims and 'x' in src.dims):
                    phase_vars[var] = src
                    continue
                phase = xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
                if keep_attrs:
                    phase.attrs = src.attrs.copy()
                phase_vars[var] = phase
            ds = xr.Dataset(phase_vars)
            if keep_attrs:
                ds.attrs = self[k].attrs.copy()
            out[k] = ds

        return BatchWrap(out)

    def unwrap2d(self, weight: 'BatchUnit | None' = None, conncomp: bool = False,
                 conncomp_size: int = 1000, conncomp_gap: int | None = None,
                 conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                 union: bool = False,
                 debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase: L1 IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        union : bool
            False (default) solves each burst on its own, which is the form
            that scales. True unions the bursts and solves once over the
            result, so the answer is consistent across burst edges -- viable
            while the merged scene fits. A Batch either way: the merge is
            internal, and each burst comes back holding only its own pixels.
        debug : bool
            Print diagnostic information.
        **kwargs
            Passed to unwrap2d_irls(): accuracy (default 2e-3; 5e-4 near the optimum,
            about 1.9 times longer), max_iter (the safety cap, default 100), epsilon.
            The former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, Batch conncomp).

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(100).angle()
        >>> unwrapped = intf.unwrap2d()  # Without weights
        >>> unwrapped = intf.unwrap2d(corr, union=True)  # With weights, one solve over the bursts
        """
        from .Stack import Stack

        return Stack.unwrap2d(Stack(), self, weight=weight,
                                       conncomp=conncomp, conncomp_size=conncomp_size,
                                       conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                                       conncomp_linkcount=conncomp_linkcount, union=union,
                                       debug=debug, **kwargs)

    def unwrap2d_irls(self, weight: 'BatchUnit | None' = None, accuracy: float = 2e-3,
                      max_iter: int = 100, epsilon: float = 1e-2, conncomp_size: int = 30,
                      debug: bool = False, **kwargs) -> 'Batches':
        """
        Unwrap phase: L1 IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        This is the core unwrapping algorithm. Disconnected components are
        unwrapped independently and aligned using per-component circular mean.
        See Stack.unwrap2d_irls() for the tasks and the threads.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        accuracy : float, optional
            The stop: the relative decrease of the L1 objective over one step. Default 2e-3, which
            reaches the former strong set's result in less time than the former relaxed set took;
            5e-4 is near the optimum and takes about 1.9 times longer.
        max_iter : int, optional
            The safety cap on the steps; reaching it before `accuracy` prints one WARNING. Default 100.
        epsilon : float, optional
            Smoothing of the L1 norm at the end of its continuation. Default 1e-2.
        conncomp_size : int
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default 30.
        debug : bool
            Print diagnostic information.
        **kwargs
            The former tol, cg_max_iter, cg_tol, device, semaphore: accepted and ignored
            with one WARNING.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, Batch):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: Batch of component labels (uint16, 0=invalid, 1=largest, ...)

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(100).angle()
        >>> unwrapped, conncomp = intf.unwrap2d_irls(weight=corr)
        """
        from .Stack import Stack

        return Stack.unwrap2d_irls(Stack(), self, weight=weight, accuracy=accuracy,
                                   max_iter=max_iter, epsilon=epsilon, conncomp_size=conncomp_size,
                                   debug=debug, **kwargs)

    def unwrap2d_link(self, conncomp_size: int = 10_000, conncomp_gap: int | None = None,
                      conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                      debug: bool = False) -> 'Batch':
        """
        Link disconnected components in already unwrapped phase.

        This function applies component linking to already unwrapped phase data
        by finding optimal 2π offsets between disconnected components.
        Use this to correct phase jumps between components after unwrapping.

        Parameters
        ----------
        conncomp_size : int
            Minimum pixels for a connected component. Default 10,000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.

        Examples
        --------
        >>> # First unwrap without linking
        >>> unwrapped = phase.unwrap2d_irls(weight=corr)
        >>>
        >>> # Then link components separately
        >>> linked = unwrapped.unwrap2d_link(conncomp_size=10_000, debug=True)
        """
        from .Stack import Stack

        return Stack.unwrap2d_link(Stack(), self,
                                            conncomp_size=conncomp_size,
                                            conncomp_gap=conncomp_gap,
                                            conncomp_linksize=conncomp_linksize,
                                            conncomp_linkcount=conncomp_linkcount,
                                            debug=debug)


class BatchUnit(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores correlation in the range [0,1].
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchWrap, BatchComplex)):
            raise ValueError(f'ERROR: BatchUnit does not support Stack, BatchWrap or BatchComplex objects.')
        # FLOAT UNITS ONLY (N81): a bool, integer or complex grid is not a unit
        BatchCore._require_float(mapping, 'BatchUnit holds float units only')
        dict.__init__(self, mapping or {})

    @classmethod
    def _keeps(cls, dtype) -> bool:
        """A result keeps the class when it is float units (BatchCore._as_class)."""
        return dtype.kind == 'f'

    def plot(
        self,
        cmap = 'auto',
        caption=None,
        alpha=1,
        vmin=0,
        vmax=1,
        *args,
        **kwargs
    ):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        kwargs["alpha"] = alpha
        return super().plot(*args, **kwargs)


class BatchComplex(BatchCore):
    def fit1d(self, threshold: float = 0.5, baseline: str = 'BPR',
              max_dh: float = 25.0, max_dv: float = 100.0,
              step_dh: float = 8.0, step_dv: float = 2.0,
              adaptive: bool = False) -> 'Batch':
        """
        Per-pixel {height, rate} fit on the per-date complex stack, no network.

        The 1d twin of fit1d_network(): the same `_3d_arc_fit` kernel on each pixel's
        own series instead of on arcs. `velocity` and `height` are named and
        scaled as fit1d_network()'s, so predict(model) inverts both. No self-check: a
        pixel is NaN where its solution lies beyond `max_dh`/`max_dv` or it has
        no non-zero date, and its `coherence` says what the rest is worth.

        No annual term (the kernel refuses one). For evenly spread dates, an
        annual amplitude above 1.435 rad lets a sideband one cycle/yr away
        outscore the true rate.

        Parameters
        ----------
        threshold : float
            Input gate: |sum z_d| / n over the non-zero dates, after the block
            screen and before the pixel's fit; pixels below it are NaN.
        baseline : str
            Name of the per-date perpendicular baseline variable; the DEM error
            is solved jointly with the rate. None or a missing variable (no
            error raised) fits no height; the DEM error then biases the rate.
        max_dh : float
            Largest DEM error, metres; a solution beyond it is NaN. A geocoding
            limit, not a prior: a DEM error shifts the sample on the ground.
        max_dv : float
            Largest rate, mm/yr; a solution beyond it is NaN. The scan runs 10%
            past it, so a rate near the bound is a maximum, not an edge.
        step_dh, step_dv : float
            Lattice steps, metres and mm/yr, as fit1d_network()'s. Two zoom levels,
            each 5x finer and 1.4 cells either side, refine the lattice winner,
            so a step must stay inside its main lobe; a long stack narrows the
            rate lobe and wants a finer `step_dv`.
        adaptive : bool
            Remove each block's common phase per date first: fit its top 1% of
            finite pixels by raw resultant (at least one), average their
            residual phasors per date and rotate that angle out of each series.
            One constant per date per block, not a substitute for detrend().
            A block whose references all fail the bounds is NaN.

        Returns
        -------
        Batch
            `velocity` rad/yr, `height` rad per unit ele2phase, `coherence` the
            resultant length maximised, `rmse` sqrt(-2 ln gamma * n/(n-p)) with
            p = 2, or 3 with a baseline. No `conncomp` or `seasonal`; predict()
            treats a missing `seasonal` as none.

        Notes
        -----
        One task per spatial chunk, with the date axis contracted. With
        `adaptive` the chunk is also the screen's block, so rechunking the
        stack (e.g. chunk2d()) changes the result.

        Examples
        --------
        >>> model = stack.fit1d()
        >>> noise = stack.complex() * stack.predict(model, baseline='BPR').iexp(sign=1)
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from .Batch import Batch

        # the baselines and the radar geometry are the Dataset's (N81); baseline=None reads
        # radar_wavelength alone
        BatchCore._needs_dataset(self, 'fit1d', (baseline,) + BatchCore._GEOMETRY if baseline
                                 else ('radar_wavelength',))
        BatchCore._require_lazy(self, 'fit1d')

        # a stack with any `pair` variable is refused; use the per-date stack
        _pairs = any('pair' in ds[v].dims
                     for ds in self.values() for v in ds.data_vars)
        if _pairs:
            # kept as its own branch for a future pair-domain fit
            raise NotImplementedError(
                'fit1d() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')

        model_result = {}
        _ep_all = None  # computed on first use: once per call, so a missing ref_height warns once and not once per burst
        for burst_id, ds in self.items():

            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                    and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                raise TypeError(
                    f'fit1d() found no complex (date, y, x) variables in '
                    f'burst {burst_id}')
            # one polarisation: the model variables are named by quantity and
            # would collide
            if len(pols) > 1:
                raise ValueError(
                    f"fit1d() fits ONE polarisation; burst '{burst_id}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two "
                    "polarisations would collide. Select one first, e.g. "
                    "batch[['VV']].")
            pol = pols[0]

            # ele2phase_d = B_perp_d / (R sin theta), one R sin theta per burst
            dates = np.asarray(ds.coords['date'].values)
            dday = dates.astype('datetime64[D]').astype(np.float64)
            bp = None
            ele2phase = None
            meter2rad = None
            if 'radar_wavelength' in ds:
                lam_ = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
                meter2rad = 4.0 * np.pi / lam_
                if baseline and baseline in ds:
                    bp = np.asarray(ds[baseline].values, dtype=float)
                    if bp.ndim > 1:
                        bp = np.nanmean(bp.reshape(len(dates), -1), axis=1)
                    # elevation_phase() = 4 pi / (lambda R sin(inc))
                    if _ep_all is None:
                        _ep_all = Batch._elevation_phase_approximate(self)
                    _fac = _ep_all[burst_id]
                    ele2phase = bp / (meter2rad / _fac)
            if meter2rad is None:
                raise TypeError(
                    f'fit1d() needs radar_wavelength in burst {burst_id} to '
                    f'turn a rotation rate into a velocity')

            # t = 0 at the master (smallest |B_perp|, else the first date); it
            # is written to the model's `date`, which predict() reads as its
            # origin
            _b = bp if (bp is not None and bp.shape == dday.shape) \
                else np.zeros_like(dday)
            _master = int(np.argmin(np.abs(_b)))
            tyr = (dday - dday[_master]) / 365.25

            data_da = ds[pol]
            if data_da.dims[0] != 'date':
                data_da = data_da.transpose('date', ...)
            data_dask = data_da.data

            def _fit_block(block, _h=ele2phase, _t=tyr, _m=meter2rad,
                           _mh=float(max_dh), _mv=float(max_dv),
                           _sh=float(step_dh), _sv=float(step_dv),
                           _thr=float(threshold), _adaptive=bool(adaptive)):
                # imported inside the task
                from .utils_fit1d import _3d_arc_fit
                from .utils_dask import get_dask_chunk_size_mb
                S = np.asarray(block)
                nd = S.shape[0]
                shape = S.shape[1:]
                Z = np.ascontiguousarray(S.reshape(nd, -1))
                npix = Z.shape[1]
                if npix == 0:
                    e = np.empty(0, np.complex64).reshape(shape)
                    return np.stack([e, e, e, e], axis=0)
                _budget = get_dask_chunk_size_mb()
                # the pixels go in batches of a quarter of the chunk size (8 bytes
                # per date and pixel, complex64), each value as from the whole
                # block (pixels are independent): the task holds its input and
                # arrays of a batch, none of the block's size
                _step = max(1, int(_budget * 2**20 // 4 // (nd * 8)))
                _batches = [slice(j, min(j + _step, npix))
                            for j in range(0, npix, _step)]

                def _unit(cols, rot=None):
                    # the normalised phasors of the pixels cols, rotated by rot
                    # per date; a NaN on any date refuses the pixel, a zero
                    # date drops out of every sum and of n, and an all-zero
                    # series is NaN
                    Zc = Z[:, cols]
                    A = np.abs(Zc)
                    ok = np.isfinite(A) & (A > 0)
                    U = np.where(ok, Zc / np.where(ok, A, 1.0), 0)
                    if rot is not None:
                        U = (U * rot).astype(np.complex64)
                    # an index set of pixels (Z[:, cols]) comes out F-contiguous;
                    # _3d_arc_fit gets the C-contiguous array it always got
                    return np.ascontiguousarray(U), A, ok

                # input gate: resultant length of the normalised phasors, which
                # a constant scatterer phase leaves unchanged
                _finite = np.empty(npix, dtype=bool)
                _nok = np.empty(npix, dtype=np.int64)
                g0 = np.empty(npix)
                for _b in _batches:
                    _U, _A, _ok = _unit(_b)
                    _finite[_b] = np.isfinite(_A).all(axis=0)
                    _nok[_b] = np.maximum(_ok.sum(axis=0), 1)
                    g0[_b] = np.abs(_U.sum(axis=0)) / _nok[_b]
                    del _U, _A, _ok
                _rot = None
                # block screen: fit the top share of the finite pixels by raw
                # resultant (so their own rate and DEM error stay out), average
                # their residual phasors per date, rotate every series by that
                # angle and recompute the gate
                _k = max(int(round(_FIT1D_SCREEN_TOP * npix)), 1) if _adaptive else 0
                _ref = np.argsort(np.where(_finite, -g0, np.inf),
                                  kind='stable')[:_k]
                _ref = _ref[_finite[_ref]]
                if len(_ref):
                    _Ur = _unit(_ref)[0]
                    _, _hr, _vr = _3d_arc_fit(
                        _Ur, _h, _t, _m, max_dh=_mh, max_dv=_mv, step_dh=_sh,
                        step_dv=_sv, budget=_budget, max_seasonal=0.0)[:3]
                    _fok = np.isfinite(_vr)
                    if not _fok.any():
                        _finite[:] = False
                    else:
                        _ph = np.outer(_t, _vr[_fok])
                        if _h is not None:
                            _ph = _ph + np.outer(_h, _hr[_fok])
                        _R = _Ur[:, _fok] * np.exp(-1j * _ph)
                        _S = _R.sum(axis=0)
                        _R = _R * np.conj(_S / np.maximum(np.abs(_S), 1e-30))
                        # a date no reference holds gets angle 0, so it is not
                        # rotated
                        _c = np.angle(_R.mean(axis=1))
                        _rot = np.exp(-1j * _c)[:, None]
                        for _b in _batches:
                            g0[_b] = np.abs(_unit(_b, _rot)[0].sum(axis=0)) / _nok[_b]
                sel = np.nonzero(_finite & (g0 >= _thr))[0]
                # only pixels at or above the gate are fitted for output
                vel = np.full(npix, np.nan, np.float64)
                hgt = np.full(npix, np.nan, np.float64)
                gam = np.full(npix, np.nan, np.float32)
                for _a in range(0, len(sel), _step):
                    _cols = sel[_a:_a + _step]
                    Us = _unit(_cols, _rot)[0]
                    # NaN beyond either bound or with no non-zero date; the
                    # batching budget is the dask chunk size
                    g_, h_, v_ = _3d_arc_fit(
                        Us, _h, _t, _m, max_dh=_mh, max_dv=_mv, step_dh=_sh,
                        step_dv=_sv, budget=_budget, max_seasonal=0.0)[:3]
                    if _h is None:
                        # no baseline: the height is 0 by model, not unassessed
                        h_ = np.where(np.isfinite(v_), 0.0, np.nan)
                    vel[_cols] = v_
                    hgt[_cols] = h_
                    gam[_cols] = g_
                    del Us
                # publish the pair sign that displacement_los() converts: a
                # per-date series runs opposite to a pair, so the rate is
                # negated. The height keeps its sign; its per-date +h*e2p_d
                # differences to -h*e2p_p.
                vel = -vel
                # rmse = sqrt(-2 ln gamma * n/(n-p)): n non-zero dates, p
                # parameters spent
                nok = np.concatenate([np.maximum((np.abs(Z[:, _b]) > 0).sum(axis=0), 1)
                                      for _b in _batches])
                npar = 2 + (0 if _h is None else 1)
                infl = nok / np.maximum(nok - npar, 1)
                Rres = np.clip(gam.astype(np.float64), 1e-9, 1.0)
                rms = np.sqrt(np.maximum(-2.0 * np.log(Rres), 0.0) * infl)
                rms = np.where(np.isfinite(gam), rms, np.nan)
                # one complex64 array carries every plane
                return np.stack([vel.reshape(shape).astype(np.complex64),
                                 hgt.reshape(shape).astype(np.complex64),
                                 gam.reshape(shape).astype(np.complex64),
                                 rms.reshape(shape).astype(np.complex64)],
                                axis=0)

            # one task per spatial chunk, dates contracted; with `adaptive` the
            # screen couples the pixels of a block
            stacked = da.blockwise(
                _fit_block, 'nyx', data_dask, 'dyx',
                new_axes={'n': 4}, concatenate=True, dtype=np.complex64,
                meta=np.empty((0, 0, 0), dtype=np.complex64))

            coords = {kk: vv for kk, vv in data_da.coords.items()
                      if kk in ('y', 'x', 'spatial_ref')}
            mvars = {}
            for nm_, arr_ in (('velocity', stacked[0].real.astype(np.float32)),
                              ('height', stacked[1].real.astype(np.float32)),
                              ('coherence', stacked[2].real.astype(np.float32)),
                              ('rmse', stacked[3].real.astype(np.float32))):
                mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            # the epoch the model is referenced to
            mds = mds.assign_coords(date=np.datetime64(int(dday[_master]), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[burst_id] = mds
        return Batch(model_result)

    def predict(self, model, baseline: 'str | None' = 'BPR',
                ref=None, vars=None) -> 'Batch':
        """
        Predicted per-date phase from a fit1d_network() or fit1d() model.

        Reconstructs, for every date in this stack,

            phi_d = velocity * t_d + height * ele2phase_d + Re(car_d * conj(seasonal))

        with `car_d = exp(2j*pi*t_d)`, `t_d` in years from the first date, and
        `ele2phase_d = BPR_d / median(R sin(incidence))` -- the same geometry
        fit1d_network() fitted against, so the two are exact inverses.

        The scatterer's constant phase is NOT part of the model: fit1d_network() profiles
        it out by rotation rather than gauging it to an epoch, so the prediction
        is correct up to one constant per pixel. That is what you want for
        removal -- `stack.complex() * predict(model).iexp(sign=1)` cancels the modelled
        part and leaves the constant, which no interferometric measurement
        determines. SIGN=1, not the default: the prediction is published as
        master*conj(date), the same radians displacement_los() converts, and
        rotating it out of the stack therefore runs the other way.

        Parameters
        ----------
        model : Batch
            Output of fit1d_network() or fit1d(): variables `velocity`, `height`,
            `seasonal`
            (unprefixed, one polarisation). Also a trend2d() model, and an
            insardev_polsar trend3d() field, which its predict3d() evaluates.
        baseline : str or None
            Per-date perpendicular baseline variable. DEFAULT 'BPR', i.e. the
            topographic term is projected onto dates and the prediction is the
            raw phase. Pass None to leave it out: the prediction is then rate +
            seasonal only -- the object that matches phase which has already had
            topography removed. Verified: predict('BPR') / predict(None) is
            exactly exp(1j*ele2phase*height) to 4e-07 rad.
        vars : Batch or None
            Only for a trend2d() model: the covariates it names, at this
            stack's posting, when the stack does not carry them itself -- the
            same argument detrend() takes.

        Returns
        -------
        Batch
            Modelled phase in RADIANS, one plane per date, on the same grid as
            this stack. REAL and UNWRAPPED: `velocity * t` is rad/yr times years
            and is never reduced, so 20 mm/yr over 3 years is 13.59 rad and 50
            mm/yr over 5 years is 56.6 rad.

            Radians are the primitive because the conversion runs ONE WAY.
            `.iexp(-1)` turns this into a phasor whenever one is wanted, but no
            operation turns a phasor back: .angle() only returns [-pi, pi], and
            at 20 mm/yr the rate it implies comes back as -1.52 mm/yr, the wrong
            sign. Returning phasors here would destroy the model's one advantage
            over the data -- that its prediction is unwrapped by construction.

        Examples
        --------
        >>> model  = stack.fit1d_network()
        >>> ground = stack.complex() * stack.predict(model=model, baseline=None).iexp(sign=1)
        >>> resid  = stack.complex() * stack.predict(model=model).iexp(sign=1)
        >>> # modelled displacement per date, unwrapped (no height term):
        >>> disp = (stack.predict(model=model, baseline=None)
        ...              .displacement_los(stack.transform()))
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        BatchCore._require_model(model, 'predict')
        # A trend3d() FIELD is insardev_polsar's measurement, recognised by what it carries: its predict3d()
        # evaluates it at this stack's pixels
        if BatchComplex._trend3d_field(model, list(model.keys())):
            if vars is not None:
                raise TypeError("predict(): `vars` are the covariates of a trend2d() "
                                "model; a trend3d() field names none.")
            return self._trend3d_method('predict3d', 'predict')(self, model)
        # A trend2d() MODEL is a different animal from a fit model: no
        # velocity or height, but per-date coefficients over the stack's own
        # covariates. Recognised by what it carries, evaluated per chunk.
        if all('trend2d_vars' in model[k].attrs for k in model):
            print("NOTE: BatchComplex.predict() computes the trend2d() model's numbers eagerly.")
            return self._trend2d_predict(model, vars=vars)
        if vars is not None:
            raise TypeError("predict(): `vars` are the covariates of a trend2d() "
                            "model; a fit1d_network() / fit1d() model names none.")
        if baseline:
            # the height term reads the Dataset's baselines and radar geometry (N81);
            # baseline=None predicts from the model and the dates alone. A Dataset
            # without them raises too: the height term was dropped, silently
            BatchCore._needs_vars(self, 'predict', (baseline,) + BatchCore._GEOMETRY,
                                  'Call it on the Dataset batch x, or pass baseline=None.')
        out = {}
        _ep_all = None  # computed on first use: once per call, so a missing ref_height warns once and not once per burst
        for key, ds in self.items():
            # a Dataset's variables, or a DataArray as its own one (N81)
            vars_ = BatchCore._vars_of(ds)
            pols = [v for v, a in vars_.items()
                    if a.dtype.kind == 'c' and 'date' in a.dims
                    and 'y' in a.dims and 'x' in a.dims]
            if len(pols) > 1:
                raise ValueError(
                    f"predict() takes ONE polarisation, burst '{key}' carries "
                    f"{len(pols)}: {pols}. fit1d_network() has the same rule.")
            if not pols:
                raise TypeError(
                    f"predict() found no complex (date, y, x) variables in '{key}'.")
            pol = pols[0]
            da_xr = vars_[pol]
            if da_xr.dims[0] != 'date':
                da_xr = da_xr.transpose('date', ...)

            mds = model[key]
            missing = [v for v in ('velocity', 'height') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit1d_network().")

            # TIME ORIGIN MUST MATCH THE FIT. utils_fit1d zeroes t at the MASTER
            # -- argmin|B_perp| -- not at the first date, because there the phase
            # is zero by construction and the height term vanishes with the
            # baseline carrying it. Velocity and height absorb an origin shift
            # into the free constant, but car = exp(2j*pi*t) does NOT: getting
            # this wrong rotates the fitted annual by exp(2j*pi*delta) and leaves
            # a residual annual of amplitude 2|sin(pi*delta)|*|seasonal|, so
            # removing the model ADDS a seasonal. Days, as the kernel uses,
            # not seconds.
            dday = (np.asarray(ds.coords['date'].values)
                    .astype('datetime64[D]').astype(np.float64))
            # BPR is read for the ORIGIN even when baseline is None -- that
            # argument decides whether the height term is applied, not where
            # time starts. With no baseline the kernel has B = zeros, so
            # argmin|B| is index 0 and the origin is the first date; this
            # reproduces that too.
            _b = None
            if baseline and baseline in ds:
                _b = np.asarray(ds[baseline].values, dtype=float)
                while _b.ndim > 1:
                    _b = np.nanmean(_b, axis=-1)
            if _b is None or _b.shape != dday.shape:
                _b = np.zeros_like(dday)
            # THE MODEL CARRIES ITS OWN EPOCH, in a scalar `date` coordinate
            # every fit writes -- the master for the per-date fits, the median
            # acquisition for Batch.fit1d(), which sees only baseline
            # DIFFERENCES and cannot recover argmin|B| from them. Reading it
            # back is what makes predict() the exact inverse of whichever fit
            # produced the model rather than of one of them. The fallback stays
            # for a model written before the coordinate existed.
            t0 = None
            if 'date' in mds.coords and mds.coords['date'].ndim == 0:
                t0 = float(np.asarray(mds.coords['date'].values)
                           .astype('datetime64[D]').astype(np.float64))
            if t0 is None:
                t0 = float(dday[int(np.argmin(np.abs(_b)))])
            t = ((dday - t0) / 365.25).astype(np.float64)

            # ele2phase per date: BPR / median(R sin(inc)), and
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            ele2phase = None
            if baseline and baseline in ds:
                # FROM THIS STACK, always. A transform= argument used to be able
                # to point the geometry somewhere else, but fit1d()/fit1d_network() take
                # theirs from the stack they are given and cannot be redirected,
                # so another source could only make the prediction disagree with
                # the model it inverts. Nothing physical rode on the choice
                # either: the scalar rescales `height` inversely and cancels in
                # the product.
                if _ep_all is None:
                    _ep_all = Batch._elevation_phase_approximate(self)
                _fac = _ep_all[key]
                bp = np.asarray(ds[baseline].values, dtype=float)
                while bp.ndim > 1:
                    bp = np.nanmean(bp, axis=-1)
                _lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
                ele2phase = bp / ((4.0 * np.pi / _lam) / _fac)

            vel = mds['velocity'].data
            hgt = mds['height'].data
            # OPTIONAL: fit1d_network() fits an annual term and fit1d() does not, and
            # a model without the variable HAS none rather than an unknown one.
            sea = mds['seasonal'].data if 'seasonal' in mds else None
            planes = []
            for i in range(len(t)):
                # THE SAME RADIANS AS EVERYTHING ELSE: what displacement_los()
                # converts. A per-date series relates to displacement the
                # OPPOSITE way a pair does -- an SLC phase is -(4pi/lambda)r, so
                # d = +psi/m2r per date while d = -phi/m2r per pair -- and
                # displacement_los()'s -lambda/4pi is the pair relation. Emitting
                # the raw SLC phase here made a planted -20 mm/yr subsidence read
                # back as +20.00 mm/yr through displacement_los while the pair
                # branch gave -20.00. So the per-date prediction is published as
                # master*conj(date), which IS an interferometric phase and obeys
                # the one convention.
                #
                # REMOVAL THEREFORE ROTATES THE OTHER WAY:
                #     residual = stack.complex() * predict(model).iexp(sign=+1)
                # iexp() defaults to exp(-1j*phase), which is the removal
                # direction for the raw SLC phase this no longer returns.
                phi = vel * float(t[i])
                if ele2phase is not None:
                    # height is NaN where no baseline was available; a NaN would
                    # poison the whole date, so contribute only where it solved
                    phi = phi - da.nan_to_num(hgt) * float(ele2phase[i])
                if sea is not None:
                    car = np.exp(2j * np.pi * float(t[i]))
                    phi = phi + (sea.real * car.real + sea.imag * car.imag)
                planes.append(phi.astype(np.float32))
            pred = da.stack(planes, axis=0)
            if ref is not None:
                # WHICH ACQUISITION READS ZERO -- see the docstring. Every
                # difference is invariant to this, so it cannot change a
                # removal or a rate, only which plane sits at zero.
                _i = Batch._ref_index(ref, ds.coords['date'].values)
                pred = pred - pred[_i]

            # THE GRID IS THE MODEL'S, THE DATES ARE THIS STACK'S. A model is
            # routinely fitted on multilooked phase and then predicted against
            # the stack it came from -- `downsample(30)` in the usual pipeline
            # -- so the two carry the same extent at different postings. Taking
            # y and x from the stack's own variable asserted they were equal
            # and raised as soon as they were not: "conflicting sizes for
            # dimension 'x': length 2915 on the data but length 23330 on
            # coordinate 'x'". The planes are built from the model's rasters,
            # so the model's axes are the ones that describe them.
            # BARE AXES, not the model's DataArrays. The model carries its epoch
            # as a SCALAR `date` coordinate, and that scalar travels with any
            # coordinate or variable taken from it -- which then collides with
            # the `date` DIMENSION this prediction has:
            # "dimension 'date' already exists as a scalar variable".
            coords = {'date': np.asarray(da_xr.coords['date'].values),
                      'y': np.asarray(mds.coords['y'].values),
                      'x': np.asarray(mds.coords['x'].values)}
            # a DataArray in, a DataArray out (N81)
            pds = BatchCore._form(ds, {pol: xr.DataArray(pred, dims=('date', 'y', 'x'),
                                                         coords=coords)}, attrs=ds.attrs)
            sref = (mds['spatial_ref'] if 'spatial_ref' in mds.coords
                    else (ds['spatial_ref'] if 'spatial_ref' in ds.coords else None))
            if sref is not None:
                sref = sref.drop_vars(list(sref.coords), errors='ignore')
                pds = pds.assign_coords(spatial_ref=sref)
            out[key] = pds
        return Batch(out)

    def mix(self, model, baseline: 'str | None' = 'BPR', ref=None) -> 'Batch':
        """
        Per-date displacement from a fit1d()/fit1d_network() model MIXED with the raw signal.

        Returns the model's displacement carrying the residual the data actually
        has, instead of the smooth curve predict() draws through it:

            resid = stack.complex() * predict(model, baseline).iexp(sign=1)
            mix   = predict(model, baseline=None) - angle(resid * conj(mean_date(resid)))

        NOTHING IS UNWRAPPED HERE, and that is the point rather than an omission.
        fit1d()/fit1d_network() fit in the complex domain, predict() returns radians that
        were never wrapped -- a rate times a span of years is a real number, not a
        phase -- and the residual is read off a complex product. The model supplies
        the absolute scale an unwrapper would otherwise have to recover, which is
        what makes this usable where the phase is too noisy to unwrap at all and a
        linear fit is the only robust description left of it.

        THE RESIDUAL IS A WRAPPED SIGNAL IN (-pi, pi], BY DESIGN. Inside half a
        cycle a deviation from the model is unambiguous, so it needs no integer
        cycle solved for it and the series is well defined without an unwrapper --
        which is the whole mechanism. Departures larger than half a cycle fold back
        into the range, so the model has to be close enough that the data's
        deviation from it is sub-cycle; that is the condition the method asks for,
        and it is the same condition that makes the fit worth trusting.

        The per-pixel constant is divided out before the residual goes back on. It
        is the scatterer's own phase, which no interferometric measurement
        determines and which the fit profiles out by rotation rather than gauging
        to an epoch.

        The topographic term is projected onto dates for the residual, where it has
        to be present or the wrap would fold it into the noise, and left out of the
        returned series, which is displacement.

        Parameters
        ----------
        model : Batch
            Output of fit1d() or fit1d_network(): variables `velocity`, `height`,
            `seasonal` (unprefixed, one polarisation).
        baseline : str or None
            Per-date perpendicular baseline variable carrying the height term.
            Default 'BPR'. Pass None when topography has already been removed from
            this stack: the model then contributes no height term and the returned
            series reconstructs what is actually here.
        ref
            Passed through to predict().

        Returns
        -------
        Batch
            Displacement phase in RADIANS, one plane per date, on the model's grid.
            REAL and unwrapped in the same sense as predict(), so it converts the
            same way -- .displacement_los() and friends. Not a phasor: .iexp()
            makes one when a phasor is wanted, and no operation makes the trip
            back.

        Examples
        --------
        >>> model = stack.fit1d()
        >>> disp  = stack.mix(model=model)
        >>> disp.displacement_los(stack.transform())
        >>> # smooth model for comparison -- same convention, no noise:
        >>> stack.predict(model=model, baseline=None)
        """
        BatchCore._require_model(model, 'mix')
        if baseline:
            # the height term reads the Dataset's baselines and radar geometry (N81); a
            # Dataset without them raises too: the height term was dropped, silently
            BatchCore._needs_vars(self, 'mix', (baseline,) + BatchCore._GEOMETRY,
                                  'Call it on the Dataset batch x, or pass baseline=None.')
        # The height term's metres-to-radians scale comes from this stack on both
        # sides, as it does in fit1d()/fit1d_network(), so it cancels in the product.
        full = self.predict(model, baseline=baseline, ref=ref)
        # THE RESIDUAL NEEDS THE WHOLE MODEL, the height term included, even though
        # the output does not want it: left out of `full` it stays in the residual,
        # and the wrap then folds topography into what is reported as noise.
        disp = (full if baseline is None
                else self.predict(model, baseline=None, ref=ref))
        # SIGN=1, NOT THE DEFAULT -- predict() explains why: the prediction is
        # published as master*conj(date), so rotating it out of the stack runs the
        # other way. The residual is SUBTRACTED for the same reason. The pair is
        # what makes the reconstruction exact: rotate it back out of the stack and
        # only the constant is left, which is the check to repeat if either the
        # prediction's convention or iexp()'s default ever moves.
        resid = self * full.iexp(sign=1)
        cmean = resid.mean(dim='date')
        return disp - (resid * cmean.conj()).angle()

    def trend2d(self, *vars, transform=None, range: float = None,
                union: bool = False, debug: bool = False) -> 'Batch':
        """
        Spatial trend of the complex phase, PER DATE, as a unit-magnitude
        phasor: `phi_d = sum_i g_di * (v_i - z_i) + k_d`, or `phi_d = k_d`
        alone when no covariate is given, so removing it is a rotation.

        >>> trend = stack.where(stack.adi()['VV'] < 0.25).trend2d('northing', 'easting', 'ele')
        >>> trend = stack.where(stack.adi()['VV'] < 0.25).trend2d(
        ...     stack.transform()[['northing','easting','ele']])       # the same
        >>> flat  = stack.detrend(trend)

        IT RUNS ON THE RAW STACK, where detrending belongs. The scatterer phase
        is still there, and being constant in time is what cancels it: one
        multiplication by the epoch the baselines are measured from, where BPR
        is zero. Its own plane is then common to every date, so no velocity
        depends on it.

        NOTHING IS UNWRAPPED, AND THE ANSWER IS SOLVED FOR. Maximising
        `sum cos(phi - g.v - k)` is a bounded-influence regression whose score
        is the SINE of the residual, and it is SOLVED, by an ascent. One pass
        over the data; the accumulator is read at whatever gradient the
        iteration asks for, not at lattice nodes.

        THE OLD GLOBAL ARGMAX WAS THE BUG. A variable's own distribution has a
        transform -- what a perfectly coherent, TREND-FREE date would score --
        and real topography puts big far lobes in it, because the pixels crowd
        into a fraction of the elevation range. Taking the largest peak over a
        wide band then answered with the elevation histogram rather than the
        phase whenever a date was weak. The ascent from zero follows the
        objective instead, to the stationary point CONNECTED to zero -- and
        misses a peak that sits across a dip from it. So the ascent gets more
        STARTS, and the highest stationary point wins: the ramp read off the
        profiles of `northing`/`easting`, and the date's own accumulator
        searched over every covariate's whole reach -- that one taken ONLY
        where the two checkerboard halves of the scene, searched on their own,
        land within the sampling's resolution of the same point. The halves
        are independent pixels of the same sampling: a lobe that noise lifted
        is not lifted in both, and a date whose halves disagree keeps the
        starts it had. Fit map ramps with Batch.trend2d(degree=...) on real
        phase, not here; this estimator is for covariates whose sampling has
        structure, elevation above all.

        THE OBJECTIVE IS BOUNDED, AND THAT PROTECTS THE GROUND PHASE: residuals
        enter as UNIT phasors, never angles, so one pixel pulls the fit by at
        most one unit. Least squares on the angle would let a patch of
        deformation tilt the plane and take the signal with it.

        Dates are fitted alone -- no master, no network, no pairs -- so the
        correction differences into every pair and triplet closure is
        untouched. Blocks add up to one plane per date, so the answer does not
        depend on the chunking.

        NOTHING IS GATED, EVERYTHING IS REPORTED. Per-date variables ride
        on the result:
          coherence, coherence0 -- phasor alignment with and without the
              trend removed; both are NaN on every unresolved date (see below);
          gain -- their difference, NaN with it. Fitting noise alone gains up to ~0.07,
              so gain above that means a trend was found;
          slope_<var> -- the fitted slope, radians per unit of the variable;
          resolution_<var> -- how far apart two slopes must be for this
              variable's sampling to tell them apart, same units: the
              half-power width of the sampling's own transform. A fitted
              slope beyond it is real signal but its value is lobe-ambiguous;
          stderr_<var> -- one-sigma of the fitted slope, same units, measured
              as half the disagreement of two checkerboard halves of the
              scene (one degree of freedom: honest scale, noisy itself);
          stderr_intercept -- with no covariate, the constant's one-sigma in
              radians, from the same halves;
          pixels -- samples fitted.
        Every unresolved date -- no pixels or a checkerboard half with none,
        a trend walking out of `range`, a degenerate covariate, no convergence,
        a different trend beating the answer on the global lattice, or a
        coherence not above the noise level -- has NaN in the intercept, the
        slopes and their errors, and in coherence, coherence0 and gain: NaN
        means no data. A coherence is proved only by a trend that stands, so
        0 would claim a well determined coherence under the noise level.
        pixels and resolution are the sampling's and stay.
        THE LIMITS, MEASURED (2026-10-27; synthetic planted ramps, 3000 samples,
        checkerboard split by pixel position, 20 noise seeds per point, range 4,
        8 and 16 pi; coherence 1.0 noiseless, 0.6 and 0.4 add Gaussian phase
        noise): every answer inside `range` is within 0.033 cycles at coherence
        0.6, 0.058 at 0.4 and 1e-4 noiseless, and the noise level keeps all of
        them; the seeds answered inside `range` (of 180) are 180 at coherence
        1.0, 170 at 0.6 and 0.4 for one covariate, and 180, 163, 163 for three.
        Trends beyond `range` come back NaN: of 2240 seeds (112 truths), answers
        with the noise level are 0-2 at range 8 and 16 pi and 0-80 at range 4 pi
        (one covariate), 0-5 for three; without it 1585-1910 (one covariate) and
        311-1118 (three).
        THIS ONLY FITS; detrend() applies.

        Parameters
        ----------
        *vars : str, or one Batch
            The covariates, three at most, each one gradient -- a raster or,
            like the map coordinates, a vector along one axis. Either NAMED,
            `trend2d('ele')` or `trend2d('northing', 'easting', 'ele')`, and
            taken from `transform`; or the Batch itself,
            `trend2d(stack.transform()[['northing','easting','ele']])`, at this
            stack's posting.

            NONE GIVEN, the model is the constant alone -- the angle of the
            phasor sum of every valid pixel: it names no covariate and carries
            no slopes, and 'stderr_intercept' prices the constant instead. A
            higher degree is not an argument either: it is named, `'ele²'`.

            The geometry is the stack's transform: where() on a Stack masks
            the data only (N81). And mind the frame -- `azi` and `rng` restart at every burst,
            so one plane cannot be written in them across bursts; `northing`
            and `easting` are the same grid for all of them.

            A variable named `<name>²` is the square of `<name>`, built
            on demand about the midpoint of its actual_range and rebuilt
            about the model's own point wherever the model is evaluated:
            `stack.transform()[['ele', 'ele²']]`. Pass it alongside its
            base, since a squared term alone pins the parabola's vertex.
        transform : Batch or None
            Where named covariates are read: `stack.transform()`, this stack's
            own when left out. Not used when the covariates come as a Batch.
        range : float
            How much of gradient space the accumulator can represent, radians
            across each variable's extent. Default None self-sizes: 128 pi
            radians (64 cycles) at one variable -- an order of magnitude
            beyond any physical trend, for a grid of about a thousand numbers
            -- and 16 pi and 8 pi radians (8 and 4 cycles) at two and three,
            where the grid is cells**k and width costs real memory. A wider
            range can land a weak date's fit on another lobe, so the default
            is the intended setting; a date whose trend walks out of the range
            comes back NaN in its coherence, coherence0 and gain too, and the
            fit says so.
            Leave it alone.
        union : bool
            False (default) fits each burst on its own pixels, so overlapping
            bursts can disagree over the ground they share. True adds every
            burst's accumulators into one fit per date; nothing is merged or
            resampled, a sum over pixels not caring where they came from.
        debug : bool
            Print each date's turn across every variable (the constant when
            there is no covariate), its coherence and its reach, and name the dates that
            did not resolve.

        Returns
        -------
        Batch
            A model dataset per burst, nothing of the stack in it:
            The MODEL, per date, in scipy's terms and in float64:
            'intercept' (radians where every covariate sits at its
            'trend2d_zero' -- the midpoint of its actual_range, read from the
            store and never measured, so that building the model runs nothing
            -- relative to the reference date), per covariate 'slope_<var>' (radians per
            unit of it), 'resolution_<var>' (how far apart two slopes must be
            for this sampling to tell them apart, same units), 'stderr_<var>'
            (the slope's one-sigma, same units), and 'coherence', 'coherence0',
            'gain', 'pixels'; the covariate names ride as attributes. With no
            covariate only 'intercept', 'stderr_intercept', 'coherence',
            'coherence0', 'gain' and 'pixels', with no covariate named. No
            raster: `stack.predict(trend)` evaluates the phase per chunk when
            asked, `stack.detrend(trend)` removes it.
        """
        import numpy as np
        import builtins as _builtins
        import xarray as xr
        import dask as _dask
        import dask.array as da
        from . import utils_detrend

        # the reference epoch is where the Dataset's BPR is smallest (N81): without it the
        # fit is referenced to another date, silently -- for a Dataset without BPR too
        BatchCore._needs_vars(self, 'trend2d', ('BPR',))
        # the covariates: named, or the Batch they always were; a gradient per
        # covariate and the constant, or -- with none -- the constant alone
        degree = 1
        if not vars:
            degree = 0
            # EVERY VALID PIXEL COUNTS: the accumulator is handed a covariate
            # that is finite everywhere, which the constant then never reads
            vars = Batch({key: xr.Dataset({'northing': xr.DataArray(
                np.asarray(ds.y.values, np.float32), dims=('y',),
                attrs={'actual_range': [float(ds.y.min()), float(ds.y.max())]})})
                for key, ds in self.items()})
        elif len(vars) == 1 and not isinstance(vars[0], str):
            vars = vars[0]
        elif all(isinstance(v, str) for v in vars):
            if len(vars) > 3:
                raise ValueError(f"trend2d(): three covariates at most -- the transform "
                                 f"is gridded in them -- got {list(vars)}.")
            if len(set(vars)) != len(vars):
                raise ValueError(f"trend2d(): each covariate once, got {list(vars)}.")
            if transform is None:
                if not hasattr(self, 'transform'):
                    raise TypeError(
                        "trend2d(): named covariates are read from the geometry, and "
                        "this batch has none of its own -- pass "
                        "transform=stack.transform().")
                transform = self.transform()
            vars = transform[list(vars)]
        else:
            raise TypeError(
                "trend2d() takes the covariates' names -- trend2d('northing', "
                "'easting', 'ele') -- or ONE Batch of them -- "
                "trend2d(stack.transform()[['ele']]); got "
                f"{[type(v).__name__ for v in vars]}.")

        # ---- per burst: the lazy pieces, nothing computed yet --------------
        preps = []
        for key, ds in self.items():
            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' and 'y' in ds[v].dims
                    and 'x' in ds[v].dims]
            if len(pols) != 1:
                raise ValueError(
                    f"trend2d() fits ONE polarisation, burst '{key}' carries "
                    f"{len(pols)}: {pols}. The atmosphere is the same for "
                    f"every polarisation, and per-pol trends would break any "
                    f"PolSAR analysis -- select one (e.g. "
                    f"stack[['{pols[0]}']]), fit it, and detrend() applies "
                    f"the one trend to every polarisation.")
            data_da = ds[pols[0]]
            if 'pair' in data_da.dims:
                raise TypeError(
                    f"trend2d() fits DATES; burst '{key}' carries a 'pair' "
                    f"dimension. A per-pair trend is not expressible as a "
                    f"per-date rotation, so removing one breaks triplet closure "
                    f"-- fit the date stack and let the pairs inherit it.")
            if 'date' not in data_da.dims:
                raise TypeError(f"trend2d() needs a 'date' dimension, burst "
                                f"'{key}' has {list(data_da.dims)}.")
            data_da = data_da.transpose('date', 'y', 'x')
            # nothing reaches across the date axis, so it stays chunked as
            # the caller left it
            data_dask = data_da.data
            nd = data_da.sizes['date']
            if min(data_dask.shape[-2:]) < 3:
                raise ValueError(f"trend2d() needs a burst grid of at least 3 x 3 pixels, burst '{key}' is {data_dask.shape[-2]} x {data_dask.shape[-1]}.")

            tds = vars[key]
            # a map coordinate is constant along the other axis, so it
            # stays a vector: each block is handed a row or a column
            var_names = [v for v in tds.data_vars
                         if tds[v].dims and set(tds[v].dims) <= {'y', 'x'}]
            if not var_names:
                raise ValueError(f"trend2d() found no gridded variables in the "
                                 f"vars for '{key}'.")
            # the vars stays lazy and is bent to the phase's chunks, never
            # the other way round
            vars_dask, vars_dims = [], []
            for var in var_names:
                _d = tuple(tds[var].dims)
                _ch = tuple(data_dask.chunks[-2:][('y', 'x').index(a)]
                            for a in _d)
                _sh = tuple(data_dask.shape[-2:][('y', 'x').index(a)]
                            for a in _d)
                var_dask = tds[var].data
                if not isinstance(var_dask, da.Array):
                    var_dask = da.from_array(
                        np.asarray(tds[var].values, np.float32), chunks=_ch)
                var_dask = var_dask.astype(np.float32)
                if var_dask.chunks != _ch:
                    var_dask = var_dask.rechunk(_ch)
                if tuple(var_dask.shape) != _sh:
                    raise ValueError(
                        f"trend2d(): '{var}' is {tuple(var_dask.shape)} over "
                        f"{_d} but the phase is {tuple(data_dask.shape[-2:])} "
                        f"over ('y', 'x') for '{key}'.")
                vars_dask.append(var_dask)
                vars_dims.append(_d)
            k = len(vars_dask)

            # the scatterer cancels against the epoch the baselines are
            # measured from, where BPR is zero
            _b = None
            if 'BPR' in ds:
                _b = np.asarray(ds['BPR'].values, dtype=float)
                while _b.ndim > 1:
                    _b = np.nanmean(_b, axis=-1)
            if nd < 2 or _b is None or _b.shape != (nd,):
                _iref = 0
                _R = da.ones(data_dask.shape[-2:], dtype=np.complex64,
                             chunks=data_dask.chunks[-2:]) if nd < 2 \
                    else da.conj(data_da.isel(date=0).data)
            else:
                _iref = int(np.argmin(np.abs(_b)))
                # just the conjugate: samples are normalised to unit phasors
                # after this, so the reference's magnitude cancels
                _R = da.conj(data_da.isel(date=_iref).data)

            # THE VARIABLE'S OWN EXTENT, and it is READ, not measured: the
            # store records how far each raster reaches, so `range` -- a turn
            # across the ground the plane is applied to -- costs nothing. A
            # vector answers for itself. The midpoint is the centre, and min
            # and max bound |v - centre| by half the extent, which a mean does
            # not on a skewed variable.
            _mx, _mn = [], []
            for var in var_names:
                _ar = tds[var].attrs.get('actual_range')
                if _ar is None:
                    raise ValueError(
                        f"trend2d(): '{var}' of '{key}' carries no "
                        f"actual_range, so how far it reaches is unknown and "
                        f"`range` has nothing to be a turn across. Measuring "
                        f"it here would hide that the store was written "
                        f"without it -- write the attribute instead.")
                _mn.append(np.float32(_ar[0]))
                _mx.append(np.float32(_ar[1]))
            stats = np.asarray(_mx + _mn, np.float32)
            # where each variable contributes nothing: 0, or a square's centre
            _zeros = [float(tds[v].attrs.get('square_centre', 0.0))
                      for v in var_names]
            preps.append({'key': key, 'pol': pols[0], 'ds': ds,
                          'data_da': data_da, 'data_ref': data_dask * _R,
                          'vars_dask': vars_dask, 'var_names': var_names,
                          'nd': nd, 'k': k, 'stats': stats,
                          'zero': _zeros,
                          'vars_dims': vars_dims, 'iref': _iref})

        if not preps:
            return BatchComplex({})
        k = preps[0]['k']
        # THE GRID SIZES ITSELF. `range` is only how much of gradient space
        # the accumulator can represent, and the cost of representing more is
        # a longer vector -- linear in cells at one variable -- so the default
        # is deliberately absurd: 128 pi radians (64 cycles) across the
        # covariate's extent at k=1, an order of magnitude beyond any physical
        # trend, for a grid of ~a thousand numbers per date. Only at two and
        # three variables does width cost real memory (the grid is cells**k),
        # so there the default falls back to 16 pi and 8 pi radians (8 and 4
        # cycles) and the one message that can ask for more still exists.
        # Nobody should ever need to set this.
        if range is None:
            range = {1: 128, 2: 16, 3: 8}.get(k, 8) * np.pi
        _cells = int(np.ceil(2 * float(range) / np.pi))
        # axis-vector covariates (northing/easting) get a ramp start from
        # their marginal profiles -- their sampling is near-uniform, so the
        # 1-D scan is clean and a multi-cycle ramp becomes reachable
        _axes = tuple(i for i, _d in enumerate(preps[0]['vars_dims'])
                      if len(_d) == 1)[:2]
        if any(p['k'] != k for p in preps):
            raise ValueError("trend2d() got a different number of variables "
                             "for different bursts.")

        if union:
            # added, not merged: the accumulator is a sum over pixels, so
            # no mosaic is built and no pixel is resampled
            nd = preps[0]['nd']
            if any(p['nd'] != nd for p in preps):
                raise ValueError(
                    f"trend2d(union=True) adds the bursts date by date, so "
                    f"they must carry the same dates; got "
                    f"{[p['nd'] for p in preps]}.")
            # POSITION, NOT LABEL: bursts of one pass are acquired seconds
            # apart, so their date stamps differ and aligning on them would
            # split every date. The loader puts them in order.

        # ---- the search geometry, all of it known before any read --------
        _K = int(np.prod(utils_detrend.trend2d_grid_shape([_cells] * k)))
        from .utils_dask import get_dask_chunk_size_mb
        _budget = get_dask_chunk_size_mb() * 2 ** 20

        # ---- the fit, as a graph: nothing is read here, so asking for one
        # burst or one date later pays for that burst or that date -----------
        groups = [preps] if union else [[p] for p in preps]
        for grp in groups:
            _S = np.stack([p['stats'] for p in grp])             # (bursts, 2k)
            _hi = _S[:, :k].max(axis=0)
            _lo = _S[:, k:].min(axis=0)
            _stats = np.concatenate([0.5 * (_hi + _lo),          # centre
                                     _hi - _lo])                 # extent
            _acc = []
            for p in grp:
                _args = []
                for var_dask, _d in zip(p['vars_dask'], p['vars_dims']):
                    _args += [var_dask, ''.join(_d)]
                _w = utils_detrend.trend2d_width(_cells, k, len(_axes))
                # BOTH CHECKERBOARD HALVES IN THE ONE PASS: two independent
                # coarse pixel sets, whose disagreement prices the estimate
                # per date. A pixel is spread once, into the grid of its own
                # half, and the total is their sum; the board is read off the
                # block's own coordinates.
                _yc = np.asarray(p['data_da'].coords['y'].values, float)
                _xc = np.asarray(p['data_da'].coords['x'].values, float)
                _ref = p['data_ref']
                _yd = da.from_array(_yc, chunks=_ref.chunks[1])
                _xd = da.from_array(_xc, chunks=_ref.chunks[2])
                # THE CALLER'S DATE CHUNKS, WHOLE, MERGED INSIDE THE TASK. A
                # pixel's kernel weights do not depend on the date, so they
                # serve every date a task carries: the date axis is contracted,
                # as fit1d() does, over runs of consecutive date chunks that
                # together fit one dask chunk. A date chunk larger than that is
                # taken as it is. Nothing is rechunked and no chunk is split --
                # a run is a slice at chunk boundaries.
                _slice = (max(_ref.chunks[1]) * max(_ref.chunks[2])
                          * _ref.dtype.itemsize)
                _cap = max(1, int(_budget // max(_slice, 1)))
                _runs, _t0, _n = [], 0, 0
                for _c in _ref.chunks[0]:
                    if _n and _n + _c > _cap:
                        _runs.append((_t0, _t0 + _n))
                        _t0, _n = _t0 + _n, 0
                    _n += _c
                _runs.append((_t0, _t0 + _n))
                _parts = []
                for _a, _b in _runs:
                    _parts.append(da.blockwise(
                        _trend2d_accumulate_for_dask, 'Dyxf',
                        _ref if len(_runs) == 1 else _ref[_a:_b], 'dyx',
                        *_args, _yd, 'y', _xd, 'x',
                        stats=_stats, cells=_cells, n_vars=k,
                        dims=[''.join(_d) for _d in p['vars_dims']],
                        extent=(float(_yc.min()), float(_yc.max()),
                                float(_xc.min()), float(_xc.max())),
                        adjust_chunks={'y': 1, 'x': 1},
                        new_axes={'D': _b - _a, 'f': _w},
                        concatenate=True, dtype=np.float64,
                        meta=np.empty((0, 0, 0, 0), np.float64)
                    ).sum(axis=(1, 2)))
                _acc.append(_parts[0] if len(_parts) == 1
                            else da.concatenate(_parts, axis=0))
            # ONE DATE PER FIT, however many dates an accumulator task took:
            # the dates are solved alone, so they are solved side by side. The
            # accumulator is the fit's own small intermediate, a row per date.
            _total = sum(_acc[1:], _acc[0]).rechunk({0: 1})
            _dts = da.from_array(
                np.asarray(grp[0]['data_da'].coords['date'].values)
                .astype('datetime64[D]').astype(np.int64), chunks=1)
            _coef = da.blockwise(
                _trend2d_finalize_for_dask, 'dc',
                _total, 'df',
                _dts, 'd',
                concatenate=True, stats=_stats, cells=_cells, k=k,
                axes=_axes, degree=degree,
                label=grp[0]['key'] if not union else 'union',
                # k gradients, the constant, why, coh, coh0, pixels, k
                # reaches, k errors, k centroids
                new_axes={'c': 4 * k + 5},
                dtype=np.float64, meta=np.empty((0, 0), np.float64))
            if debug:
                # the table needs the numbers, so they are computed here, ONCE,
                # and the model carries them: computing it runs nothing again
                _coef = da.from_array(np.asarray(_coef), chunks=-1)

            for p in grp:
                p['coef'] = (_coef, _stats)

        # ---- the plane: a few numbers per date and the geometry, so the
        # raster is an expression rather than something a task builds --------
        out = {}
        for p in preps:
            _coef, _stats = p['coef']
            data_dask = p['data_da'].data
            _dc = data_dask.chunks[0]

            if debug:
                _cf = np.asarray(_coef)
                _det = _cf[:, k + 1] == 0
                _span = _stats[k:2 * k]
                # degree 0 has no gradient columns to show, only the constant
                _tv = p['var_names'] if degree else []
                _kt = len(_tv)
                print(f"trend2d('{p['key']}'): {p['nd']} dates, {k} "
                      f"variables {p['var_names']}, degree {degree}, "
                      f"referenced to date {p['iref']}"
                      + (" [one fit for every burst]" if union else ""),
                      flush=True)
                hdr = ("    date " + " ".join(f"{v:>12s}" for v in _tv)
                       + ("" if degree else f"{'intercept':>12s} {'stderr':>9s}"))
                print(hdr + f"{'coh':>9s} {'pixels':>12s}  "
                      + " ".join(f"{'resolution ' + v:>13s}" for v in _tv)
                      + ("   [rad across the variable's span]" if degree
                         else "   [rad]"), flush=True)
                _tag = {1: 'no pixels or a half with none', 2: 'walked out of `range`',
                        3: 'degenerate covariate', 4: 'did not converge',
                         5: 'beaten on the global lattice', 6: 'below the noise level'}
                for d in _builtins.range(p['nd']):
                    _coh = _cf[d, k + 2]
                    _reach = " ".join(
                        f"{_cf[d, k + 5 + i] * _span[i]:13.4f}"
                        for i in _builtins.range(_kt))
                    if not _det[d]:
                        print(f"    {d:4d} " + " ".join(f"{chr(45) * 2:>12s}"
                              for _ in _builtins.range(max(_kt, 1)))
                              + ("" if degree else f" {chr(45) * 2:>9s}")
                              + f" {_coh:8.5f} {int(_cf[d, k + 4]):12,d}  "
                              f"{_reach}   {_tag[int(_cf[d, k + 1])]}",
                              flush=True)
                        continue
                    row = (" ".join(f"{_cf[d, i] * _span[i]:12.4f}"
                                    for i in _builtins.range(_kt)) if degree
                           else f"{_cf[d, k]:12.4f} {_cf[d, 2 * k + 5]:9.4f}")
                    print(f"    {d:4d} {row} {_coh:8.5f} "
                          f"{int(_cf[d, k + 4]):12,d}  {_reach}", flush=True)
                if union:
                    debug = False        # the table is the same for every burst

            ds = p['ds']
            # THE MODEL IS THE PRODUCT, like fit1d_network(): the fit's own numbers
            # and nothing of the stack. A date's trend is
            # `intercept_d + sum_i slope_di * v_i` -- a handful of numbers
            # per date over the stack's own covariates. Evaluating it here
            # into a (date, y, x) phasor made a raster the size of the
            # stack, which compute() then persisted across the cluster and
            # every block read had to fetch pieces of. predict() evaluates
            # it lazily where it is asked for; detrend() applies it so.
            # SCIPY'S NAMES, IN FLOAT64: `slope_<var>` is the gradient per
            # unit of the covariate and `intercept` the phase where every
            # covariate sits at its 'trend2d_zero'. The fit itself is centred
            # on the extent's midpoint for the grid; the constant is moved
            # here, in float64.
            o = xr.Dataset(coords={'date': np.asarray(
                p['data_da'].coords['date'].values)}, attrs=dict(ds.attrs))
            # THE ZERO IS READ, NEVER MEASURED, so that nothing here needs the
            # fit: a variable's zero is the midpoint of its actual_range --
            # where the fit's own constant already sits -- and a square's is
            # the centre it was built about. An attribute has to be a number
            # now; a zero taken from the fitted pixels would run the whole fit
            # inside this call.
            _names = list(p['var_names']) if degree else []
            _orig = [_coef[:, i].astype(np.float64)
                     for i in _builtins.range(len(_names))]
            _zeros = [float(p['zero'][i]) if _v.endswith('²') else float(_stats[i])
                      for i, _v in enumerate(_names)]
            _icpt = _coef[:, k].astype(np.float64)
            _slopes = list(_orig)
            for i, _v in enumerate(_names):
                if _v.endswith('²'):
                    _d = np.float64(_zeros[i]) - np.float64(p['zero'][i])
                    _icpt = _icpt + _orig[i] * (_d * _d - np.float64(_stats[i]))
                    _b = _v[:-1]
                    if _b in _names and _d != 0.0:
                        _j = _names.index(_b)
                        _slopes[_j] = _slopes[_j] + 2.0 * _orig[i] * _d
                else:
                    _icpt = (_icpt + _orig[i]
                             * (np.float64(_zeros[i]) - np.float64(_stats[i])))
            o['intercept'] = xr.DataArray(_icpt, dims=('date',))
            # a constant names no covariate, so predict() needs none
            o.attrs['trend2d_vars'] = list(p['var_names']) if degree else []
            o.attrs['trend2d_dims'] = ([''.join(_d) for _d in p['vars_dims']]
                                       if degree else [])
            # aligned with trend2d_vars: where each variable contributes
            # nothing, and for a square the point it is squared about
            o.attrs['trend2d_zero'] = _zeros
            o.attrs['trend2d_ref'] = int(p['iref'])
            o['coherence'] = xr.DataArray(
                _coef[:, k + 2].astype(np.float32), dims=('date',))
            o['coherence0'] = xr.DataArray(
                _coef[:, k + 3].astype(np.float32), dims=('date',))
            # what removing the trend bought; ~0.07 is reachable by fitting
            # noise alone, so above that a trend was genuinely found
            o['gain'] = xr.DataArray(
                (_coef[:, k + 2] - _coef[:, k + 3]).astype(np.float32),
                dims=('date',))
            o['pixels'] = xr.DataArray(
                _coef[:, k + 4].astype(np.int64), dims=('date',))
            if not degree:
                # the constant's one-sigma, radians: half the disagreement of
                # the two checkerboard halves' constants
                o['stderr_intercept'] = xr.DataArray(
                    _coef[:, 2 * k + 5].astype(np.float64), dims=('date',))
            for i, var in enumerate(p['var_names'] if degree else []):
                # per unit of the covariate, all three, so they read against
                # each other: the slope, how far apart two slopes must be for
                # this sampling to tell them apart, and the slope's one-sigma
                o[f'slope_{var}'] = xr.DataArray(_slopes[i], dims=('date',))
                o[f'resolution_{var}'] = xr.DataArray(
                    _coef[:, k + 5 + i].astype(np.float64), dims=('date',))
                o[f'stderr_{var}'] = xr.DataArray(
                    _coef[:, 2 * k + 5 + i].astype(np.float64), dims=('date',))
            if 'spatial_ref' in ds.coords:
                o = o.assign_coords(spatial_ref=ds['spatial_ref'].drop_vars(
                    list(ds['spatial_ref'].coords), errors='ignore'))
            out[p['key']] = o
        return Batch(out)

    def _trend2d_predict(self, model, vars=None) -> 'Batch':
        """The trend2d() model evaluated on this stack's grid: a dask
        expression `intercept_d + sum_i slope_di * (v_i - zero_i)` over the
        covariate rasters, `(base - zero)**2` for a `<var>²`, evaluated
        in float64 and handed on as float32 phase.

        THE COVARIATES ARE THE STACK'S OWN. The model names them; each is
        taken from this stack's variables, or from its map coordinates for
        `northing`/`easting`, exactly as transform() exposes them, so nothing
        has to be carried along with the model. `vars` overrides that with a
        Batch of covariates at this posting when the stack does not hold
        them.

        THE MODEL'S DATES ARE THE OUTPUT'S DATES. A model holding a subset of
        the stack's acquisitions, or holding them in another order, selects
        the stack; only a date the model names and the stack does not is an
        error.

        LAZY. The per-date numbers are constants in the graph, a few
        kilobytes; the covariates are read from the store as the data are.
        Nothing of raster size is materialised anywhere, and a block window
        built on this holds no cluster-held piece.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        out = {}
        for key, ds in self.items():
            if key not in model:
                raise KeyError(f"predict(): the model has no burst '{key}'.")
            mds = model[key]
            names = list(mds.attrs['trend2d_vars'])
            dims = list(mds.attrs['trend2d_dims'])
            # a Dataset's variables, or a DataArray as its own one (N81)
            grids = [v for v, a in BatchCore._vars_of(ds).items()
                     if a.dtype.kind == 'c' and 'date' in a.dims
                     and 'y' in a.dims and 'x' in a.dims]
            if not grids:
                raise TypeError(
                    f"predict() found no (date, y, x) complex variable in "
                    f"'{key}' to evaluate the trend on.")
            _md = np.asarray(mds.coords['date'].values)
            _sd = np.asarray(ds.coords['date'].values)
            if _md.shape != _sd.shape or not np.array_equal(_md, _sd):
                # A MODEL MAY HOLD FEWER DATES THAN THE STACK, and in another
                # order: trend2d() leaves out a date it cannot fit, and a
                # caller drops the ones it does not trust. Neither is an error.
                # THE PREDICTION FOLLOWS THE MODEL -- the stack is selected
                # down to the model's acquisitions, in the model's order, so a
                # date carrying no trend is left out rather than predicted
                # from some other date's coefficients.
                _miss = _md[~np.isin(_md, _sd)]
                if _miss.size:
                    raise ValueError(
                        f"predict(): the trend2d() model of '{key}' holds "
                        f"{_miss.size} date(s) this stack does not: "
                        f"{[str(d) for d in _miss[:5]]}.")
                ds = ds.sel(date=_md)
            ref = BatchCore._vars_of(ds)[grids[0]].transpose('date', 'y', 'x')
            data = ref.data
            # the model per date, as CONSTANTS in the graph; the phase is
            # evaluated in float64 -- an intercept at zero and a slope times
            # a covariate in the thousands cancel to a few radians -- and
            # handed on as float32, the precision the raster always had
            _dc = data.chunks[0]
            # THE MODEL'S NUMBERS IN ONE COMPUTE: `intercept` and every `slope_<var>` are views of ONE lazy fit, and
            # asked for one by one each `.values` ran the whole fit again (4 runs for 3 covariates, and the line of a
            # date that did not resolve printed 4 times)
            import dask
            _need = ['intercept'] + [f'slope_{v}' for v in names]
            # NOTE: prints through BatchComplex.predict and BatchComplex.detrend
            _num = dict(zip(_need, dask.compute(*[mds[n].data for n in _need])))
            kd = da.from_array(np.asarray(_num['intercept'], np.float64),
                               chunks=(_dc,))
            phi = kd[:, None, None]
            src = vars[key] if vars is not None else ds
            # a DataArray burst carries no covariate but its map coordinates (N81)
            src_vars = BatchCore._vars_of(src)
            # absent in a model fitted before the attribute existed: zero
            _zero = list(mds.attrs.get('trend2d_zero', []))

            def _plain(nm):
                if nm in src_vars:
                    return src_vars[nm]
                if nm == 'northing' and 'y' in src.coords:
                    return xr.DataArray(np.asarray(src.y.values, np.float32),
                                        dims=('y',))
                if nm == 'easting' and 'x' in src.coords:
                    return xr.DataArray(np.asarray(src.x.values, np.float32),
                                        dims=('x',))
                return None

            for i, (v, d) in enumerate(zip(names, dims)):
                _z = float(_zero[i]) if i < len(_zero) else 0.0
                if v.endswith('²'):
                    # A SQUARE IS REBUILT, NEVER LOOKED UP: the model records
                    # the point it is squared about, so a crop with another
                    # extent still evaluates the model that was fitted
                    _bc = _plain(v[:-1])
                    cov = (None if _bc is None
                           else (_bc.astype(np.float64) - _z) ** 2)
                else:
                    cov = _plain(v)
                    if cov is not None and _z:
                        cov = cov.astype(np.float64) - _z
                if cov is None:
                    _extra = (f" It is the square of '{v[:-1]}', which is not "
                              f"there either." if v.endswith('²') else "")
                    raise KeyError(
                        f"predict(): the trend2d() model needs covariate "
                        f"'{v}', which '{key}' does not carry.{_extra} Pass "
                        f"`vars=stack.transform()[[...]]` at this posting.")
                _d = tuple(cov.dims)
                if ''.join(_d) != d:
                    raise ValueError(
                        f"predict(): covariate '{v}' of '{key}' is over "
                        f"{_d}, the model was fitted over ('{d}',).")
                _ch = tuple(data.chunks[1:][('y', 'x').index(a_)] for a_ in _d)
                cd = cov.data
                if not isinstance(cd, da.Array):
                    cd = da.from_array(np.asarray(cov.values, np.float32),
                                       chunks=_ch)
                cd = cd.astype(np.float64)
                if cd.chunks != _ch:
                    cd = cd.rechunk(_ch)
                gd = da.from_array(
                    np.asarray(_num[f'slope_{v}'], np.float64),
                    chunks=(_dc,))
                _v = cd
                if _d == ('y',):
                    _v = _v[:, None]
                elif _d == ('x',):
                    _v = _v[None, :]
                phi = phi + gd[:, None, None] * _v[None]
            # a constant, or a model over vectors along one axis only, does
            # not vary along the other, so the phase is broadcast to the
            # raster it is applied to
            if tuple(phi.shape) != tuple(data.shape):
                phi = da.broadcast_to(phi, data.shape, chunks=data.chunks)
            coords = {k_: v_ for k_, v_ in ref.coords.items()
                      if k_ in ('date', 'y', 'x', 'spatial_ref')}
            # a DataArray in, a DataArray out (N81): the trend's 'phase'
            out[key] = BatchCore._form(ds, {'phase': xr.DataArray(
                phi.astype(np.float32), dims=('date', 'y', 'x'),
                coords=coords)}, attrs=dict(ds.attrs))
        return Batch(out)

    @staticmethod
    def _trend3d_field(trend, keys) -> bool:
        """Whether `trend` is an insardev_polsar trend3d() FIELD for every one of `keys`: its bursts' attrs
        hold 'trend3d_cell'. A field is a measurement, not a model; predict() and detrend() hand it to the
        module's predict3d() and detrend3d()."""
        keys = list(keys)
        return len(keys) > 0 and all(k in trend and 'trend3d_cell' in trend[k].attrs for k in keys)

    def _trend3d_method(self, name, what):
        """insardev_polsar's `name` (predict3d, detrend3d), which takes a trend3d() field, as the class holds it:
        a Batch of DataArrays reaches it directly. Without the module, the common error names it."""
        method = getattr(type(self), name, None)
        if method is None:
            raise ImportError(f"{what}() of a trend3d() field requires insardev_polsar extension")
        return method

    def detrend(self, trend: 'Batch', vars=None) -> 'BatchComplex':
        """
        Remove a trend from EVERY pixel: `self * exp(-1j * phase)`. The trend says
        what it is, as predict()'s model does, so the name carries no dims: a
        trend2d() model, or an insardev_polsar trend3d() field, which its
        detrend3d() removes; detrend2d() is an alias.

        The phase is predict(trend): the model's per-date coefficients over
        the stack's own covariates, evaluated per chunk -- nothing of raster
        size is built, persisted or fetched. One trend, every polarisation:
        the atmosphere is the same for each complex raster.

        Parameters
        ----------
        trend : Batch
            What trend2d() returned for this stack, or a trend3d() field
            (insardev_polsar). It may hold FEWER dates
            than the stack: those dates are what comes back, since a date with
            no trend cannot be detrended.
        vars : Batch or None
            Covariates at this posting, only when the stack does not carry
            the ones the model names (a trend2d() model only).

        Returns
        -------
        BatchComplex
            The stack with the trend rotated out of every polarisation,
            everything else untouched, over the trend's dates.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        # a trend3d() FIELD, recognised by what it carries: insardev_polsar's detrend3d() removes it
        if BatchComplex._trend3d_field(trend, self.keys()):
            if vars is not None:
                raise TypeError("detrend(): `vars` are the covariates of a trend2d() "
                                "model; a trend3d() field names none.")
            return self._trend3d_method('detrend3d', 'detrend')(self, trend)
        for key in self.keys():
            if key not in trend or 'trend2d_vars' not in trend[key].attrs:
                raise ValueError(
                    f"detrend(): no trend2d() model for burst '{key}' -- "
                    f"pass what trend2d() returned for this stack.")
        print("NOTE: BatchComplex.detrend() computes the trend2d() model's numbers eagerly.")
        pred = self._trend2d_predict(trend, vars=vars)
        res = {}
        for key, ds in self.items():
            # the trend's phase: a DataArray burst's prediction is the DataArray itself (N81)
            phi = BatchCore._vars_of(pred[key])['phase']
            _pd = np.asarray(phi.coords['date'].values)
            _sd = np.asarray(ds.coords['date'].values)
            if _pd.shape != _sd.shape or not np.array_equal(_pd, _sd):
                # predict() gave back the MODEL's dates; the stack follows.
                # A date the trend does not carry is DROPPED, never passed
                # through untouched -- a stack mixing detrended and raw dates
                # is not a stack, and the mix is invisible downstream.
                ds = ds.sel(date=_pd)
            # THE SAME NUMBERS THE RASTER USED TO HOLD: float32 phase into a
            # complex64 exponential, rotated out
            rot = xr.DataArray(
                da.exp(np.complex64(-1j) * phi.data.astype(np.complex64)),
                dims=phi.dims, coords=phi.coords)
            upd = {v: a * rot for v, a in BatchCore._vars_of(ds).items()
                   if a.dtype.kind == 'c' and 'y' in a.dims
                   and 'x' in a.dims and 'date' in a.dims}
            if isinstance(ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81)
                res[key] = BatchCore._form(ds, upd)
                continue
            res[key] = ds.assign(upd)
        return type(self)(res)

    def detrend2d(self, trend: 'Batch', vars=None) -> 'BatchComplex':
        """Alias of detrend(), for code that names the dims: prints a WARNING and calls
        detrend(trend, vars), which reads what the trend is (user, 2026-10-01)."""
        for key, ds in self.items():
            ny, nx = ds.sizes.get('y', 3), ds.sizes.get('x', 3)
            if min(ny, nx) < 3:
                raise ValueError(f"detrend2d() needs a burst grid of at least 3 x 3 pixels, burst '{key}' is {ny} x {nx}.")
        print('WARNING: detrend2d() calls detrend(). Use detrend(trend).')
        return self.detrend(trend, vars=vars)

    def fit1d_network(self, threshold: float = 0.5, window: tuple = (400, 4000),
                cell: 'float | tuple' = 24,
                baseline: str = 'BPR',
                level: int = 1,
                max_dh: float = 25.0, max_dv: float = 25.0,
                step_dh: float = 8.0, step_dv: float = 2.0,
                max_seasonal: float = 0.0,
                consensus: int = 3,
                err_dh: float = 4.0, err_dv: float = 1.0,
                union: bool = False,
                iterations: int = 8,
                debug: bool = False) -> 'Batch':
        """
        Fit a per-pixel (height, velocity, seasonal) model on a PS network.

        Returns the MODEL ONLY -- one dataset of named parameters, no phase.
        Use predict(model=...) to reconstruct phase from it, and subtract
        whatever you actually want removed; the caller decides, not the fit. Solved
        PER DASK CHUNK with no inter-chunk state:

          nodes    every pixel `arcs()` certifies; nothing is thinned, since
                   the cell constrains ARCS, not nodes
          arcs     each node's best partners by raw coherence inside four
                   half-offset windows, then a maximum spanning forest and a
                   second arc per node, every edge clearing the independence
                   cell -- a pair inside the cell is one ground sample, and
                   its coherence is the impulse response
          model    joint (height, velocity) per arc; they are NOT separable
                   one at a time, because the perpendicular baseline is not a
                   smooth function of time
          network  least squares onto per-node values; each component's free
                   datum is estimated from its own mean residual and removed,
                   so a network in several pieces costs nothing
          ground   node phase minus its own HEIGHT term only. Pixels that
                   carry neither a node nor an attached DS are NaN; nothing is
                   interpolated into them
          level 1  every DS is fitted against the PS nodes inside the PS
                   EXTENT -- the same reach the network arcs use, not the
                   smaller DS window, since a PS is by definition a scatterer
                   that holds a fitted arc that far. It inherits the chosen
                   node's height, rate, seasonal and component LABEL, so it
                   lands on the same datum. A DS with no arc clearing
                   `threshold` stays NaN: without a coherent path to the
                   network it has no datum
          level 2+ dropped: it added about 30 % more DS of clearly lower
                   accuracy at a high cost in time and memory; a request
                   prints one WARNING and runs level 1

        Returns ONE dataset carrying the solve, its variables named
        by quantity:

          `velocity`   rad/yr
          `height`     rad per unit ele2phase
          `seasonal`   complex rad, the fitted annual
          `coherence`  arc coherence tying the pixel to the network
          `rmse`       sqrt(-2 ln coherence), rad
          `conncomp`   int8, -1 nodata, 0 the largest component
          `level`      int8, WHICH CASCADE STEP PLACED THE PIXEL: 0 a PS
                       network node, 1 a DS attached to that network, -1 nothing
                       solved

        all NaN (or -1) where nothing was solved. Names rather than positions,
        so a caller never counts commas and adding a quantity moves nothing.
        There is NO polarisation prefix: fit1d_network() takes exactly one polarisation
        and raises otherwise, so nothing needs disambiguating and predict() can
        look the names up directly.

        RADIANS OUT, as everywhere else here. `displacement_los()` stays the
        single place a length is produced. `coherence` is the arc quality that
        ties each pixel in -- the mean over its own arcs for a node, the
        attaching arc for a densified DS -- and `rmse` is its exact inverse
        transform, carried for convenience rather than as new information.

        VELOCITY IS A RASTER, not a stats entry. The kernels' stats live in
        `utils_fit1d._fit_stats`, a per-thread object written by whichever
        block ran last on that worker thread, so under dask it describes ONE
        chunk and misdescribes the others -- it reported 200 nodes for a
        raster carrying 29118. A product that cannot be rebuilt from what the
        method returns is not really returned; the stats stay for
        single-block diagnostics only.

        NO ATMOSPHERIC SCREEN IS COMPUTED, by measurement rather than
        omission. A per-node screen kriged to the ground lowered coherence at
        every separation: a node residual is dominated by its own noise rather
        than by correlated signal, so interpolation spreads mostly that error.
        Published kriging estimators were reproduced and behave the same way.
        The one term with a positive effect is a per-epoch stratified delay
        proportional to ELEVATION, and only marginally, concentrated at long
        range -- not enough to put in this path.

        Parameters
        ----------
        threshold : float
            Arc coherence for an arc to count and to be kept in the network.
            An arc is refined only where its coherence on the coarse lattice
            could still reach it -- threshold times the stack's own response
            half a lattice step off, the most a lattice point can under-read a
            peak -- so the refinement is skipped only where it cannot change
            which arcs are kept.
        window : tuple of float
            GROUND SIZES IN METRES, divided by the burst's posting once, here,
            so the same call describes the same neighbourhood on any grid.

            `(ds, ps)` is the DS window and the PS extent, both as squares on
            the ground: (400, 4000) is a 400 m box inside a 4000 m search.
            `(dsy, dsx, psy, psx)` states the two axes separately when the
            ground itself is anisotropic. Everything inside the fit works in
            pixels; nothing else takes a pixel count.

            The PS extent is the reach of everything that tests against a NODE
            -- the long arcs that prove a PS, the network arcs that carry the
            datum, and the DS attachment. It is therefore the caller's cost
            dial as well: widening it grows the candidate partners per pixel
            and the arc fits with them.
        cell : float or tuple of float
            The independence cell in METRES: two pixels closer than this in
            both axes are one sample of the ground measured twice, so arcs
            between them measure the sensor rather than the terrain. A single
            number is the same distance on both axes, which is what a
            resolution cell usually is.
        baseline : str
            Variable holding the perpendicular baseline per date.
        iterations : int
            0 stops the per-arc fit at the lattice argmax, which is enough to
            RANK candidates and is what the shortlist stages use. Any positive
            value runs the refinement -- a zoom of two lattices, each five
            times finer than the last, around the winner -- and the number
            itself no longer matters: the refinement used to iterate, and the
            count is kept only so existing calls still read.
        consensus : int
            How much agreement is required before a value is reported, asked
            once for both halves of the solve: a node must keep this many
            arcs to stay in the network and a component this many nodes to
            keep its datum, and a DS's best partners -- this many of them --
            must agree with each other on its height and rate within the
            absolute bounds `err_dh`, `err_dv`.

            THE PARTNERS ARE INDEPENDENT SAMPLES. A DS is offered its best
            partners by arc coherence, one per independence cell: two fixed
            nodes closer than `cell` in both axes are one sample of the ground
            measured twice, so the better of them holds one slot and the
            other is never a second vote. The vote is therefore a vote of
            distinct scatterers, and `consensus` means the same thing at every
            pixel. Before this rule five votes could be five pixels of one
            scatterer agreeing with itself, and those DS were the outlier
            population of their level.

            DEFAULTS TO 3, and that is measured, not chosen. With independent
            partners, the DS that only two votes certify carry twice the
            share of velocity outliers of the level they join (one in ten
            past 5 mm/yr against one in twenty), agree half as well with
            their own neighbourhood, and sit 60% further from an independent
            reference: two partners still agree by chance at the arc noise
            level. The third vote is where the certified set reaches the
            quality of its neighbourhood; a fifth trims the tail further at
            the cost of a fifth of the pixels. 2 is a coverage setting, 5 a
            conservative one, and the nodes are the same at every value.

            A node the rejection leaves below `consensus` arcs leaves the
            rasters, the stats and the DS attachment together; a DS whose
            partners cannot muster that many agreeing is not attached.

            None turns it off: the network is integrated by plain least squares
            and a DS takes its best arc. That is the right setting when the
            caller is tuning `threshold` themselves and does not want a second
            rule moving the answer underneath them -- the checks exist to make a
            LOW threshold usable, by removing what a low gate lets in.

        level : int
            How far to carry the solve, 0 or 1. Level 1 uses the network as its
            references.

            0   the PS network only. Nodes carry values; every other pixel is
                NaN. This is the answer when a caller wants only what the PS
                test certified.
            1   plus DS attached to PS, each by `consensus` agreeing PS
                partners inside the PS extent.

            The default is 1. Level 2+ is dropped: it added about 30 % more DS
            of clearly lower accuracy at a high cost in time and memory; a
            request prints one WARNING and runs level 1.

        union : bool
            What a scene is, the same word as in trend2d(). False, the
            default, makes each burst its own scene: the PS test sees every
            candidate of that burst within the PS extent, one network fixes
            one datum over the burst, and where two bursts overlap their
            models disagree over the shared ground, as two per-burst trends
            do. True makes the whole stack one scene: one network, one datum
            across the burst seams. The same driver runs either way; a
            multi-burst stack under False runs it once per burst. A caller
            who wants a small area crops the stack first -- the crop is then
            the scene, and the chunking is dask's business alone.

        debug : bool
            Print a stage-by-stage account of the solve: how many nodes the PS
            test found, how many pairs were fitted and how many cleared
            `threshold`, what the robust pass rejected, the connected
            components and their sizes, and where the DS candidates went --
            too few partners, no consensus, or attached. Off by default.

            The counts are the ones that explain a disappointing result. A
            thin answer is either few nodes, few arcs, a network in pieces, or
            DS that reached a node and failed consensus, and those call for
            different changes -- the returned rasters alone cannot tell them
            apart.

        max_dh, max_dv : float
            Largest DIFFERENTIAL height (m) and rate (mm/yr) an arc may carry:

            `max_dv` DEFAULTS BELOW ONE FRINGE PER YEAR. A rate of lambda/2 per
            year -- 27.7 mm/yr at Sentinel-1's 55.5 mm -- advances the phase by
            a whole cycle over a year, and the search finds a second maximum
            close to it. That maximum is not a true alias: with these dates the
            two are separable in principle. But it is high enough to win on a
            marginal arc, and ONE arc that takes it is then integrated by the
            network across everything behind it -- with closure staying clean,
            since every arc agrees with the shifted solution. Measured on a
            real stack, 55 nodes sat a full cycle above their neighbours 1 km
            away while the solve reported no node closing worse than 0.6 mm/yr.
            Keeping the bound under that rate denies the search the second
            maximum. It does NOT bound the velocity a pixel may be reported at:
            the bound is per ARC, and faster ground is reached by integrating a
            chain of arcs that each stay inside it. Raising it past lambda/2 per
            year re-admits the failure; the value is wavelength-specific and
            would need revisiting for another mission.

            `max_dh` DEFAULTS TO 25 m FOR THE SAME REASON. A scatterer 100 m
            above its neighbour is reached by a chain of arcs that each step
            under 25 m, never by one arc searched over 100 m: the wide search
            is exact on a real large height (a planted 40 m comes back as
            40 m) and gains nothing on a good arc, but on an arc whose
            coherence is near the floor it hands the fit a sidelobe height
            tens of metres off, and a few of those clear the coherence
            threshold and enter the network with the rate error that a wrong
            height carries on drifting baselines. Measured against an
            independent height reference, 96% of the arcs a 100 m search put
            beyond 25 m were wrong, the extra pixels it admitted were the
            weakest of the product, and the wider lattice cost most of the
            run time; 110 m also reaches the height ambiguity of the
            baselines. Widen it only for terrain whose DEM is known to be off
            by more than 25 m between neighbours.

            the difference between two neighbours a few tens of metres apart,
            not an absolute elevation or velocity. Anything solving beyond
            them returns NaN rather than a plausible wrong number, so these
            are the guarantee, not a hint -- the search runs wider than they
            say so that max_dv=100 detects 99 mm/yr on its merits and never
            against a boundary. Narrow them for speed when the terrain
            allows, widen them for a rapidly deforming one.
        step_dh, step_dv : float
            Lattice step in height (m) and rate (mm/yr). These pick which
            BASIN is found, NOT the accuracy -- the refinement is continuous
            and absorbs the quantisation over a wide range of steps. Raise
            them to go faster; the failure when they are finally too coarse is
            detected, not silent.

            HEIGHT AND RATE ARE TIED 4:1, in the steps and in the bounds
            below. Sentinel-1 holds its orbital tube by design, so the phase a
            metre of height carries at the widest baseline is fixed, and the
            phase a mm/yr of rate carries at the farthest date grows only
            with the stack's span: over the multi-year stacks in use one
            mm/yr is worth about four metres of height, so err_dh = 4 err_dv
            and step_dh = 4 step_dv make the two parameters one statement in
            phase. Measured on a real stack against a fine lattice and an
            independent reference: the products at (8 m, 2 mm/yr), (4 m,
            1 mm/yr) and (2 m, 1 mm/yr) agree on shared pixels to within the
            node precision and identically against the reference, so the
            coarse end of that family is the default and the fastest. The
            ratio is Sentinel-1's; another mission's tube sets its own.
        err_dh, err_dv : float
            The ABSOLUTE bound, in metres and mm/yr, on how far a
            measurement may sit from the solve: a node's arc residual in the
            network and a DS's disagreement with its partners' vote. Stated
            in physical units because a metre of height and a mm/yr of rate
            carry different amounts of phase; tied 4:1 as above so the two
            are one bound. Tightening err_dh from 5 to 4 m drops about 4% of
            the marginal DS at the gate and leaves the rest untouched.
        max_seasonal : float
            Must be 0, the default: the annual term is not fitted by the arc
            kernel and a positive value raises ValueError. The physics has not
            changed -- an annual of amplitude A radians leaves coherence
            |J0(A)| at the true rate and |J1(A)| one cycle/yr away, crossing at
            A = 1.435 rad, above which a {height, rate} fit returns the
            sideband -- but on the arcs fitted here the term read as noise and
            the freedom to fit it cost the rate more than it returned. It
            comes back when a stack with a real annual signal exists to design
            it against.
        Returns
        -------
        BatchCore
            complex64 unit phasors of the atmospheric phase, NaN where not
            solved.

        THE SCREEN COMES FROM PS AND FROM NOTHING ELSE. Distributed scatterers
        can receive it; they cannot source it, because their residuals are noise
        and averaging more of them converges to zero rather than to the delay.
        The nodes are therefore the PS raster -- a DS window's best pixel,
        verified against partners BEYOND the window -- and the network is bounded
        by the same PS window, so `window` is the only reach the caller sets.

        RETURNS TWO THINGS, AND THE SECOND IS NOT DECORATION. Each connected
        component of the network carries its own free constant per date, because
        arcs cancel exactly that, so two pixels are comparable only if they came
        from the same component. `labels` says which, exactly as the 2D
        unwrapping reports its own components, and a caller who ignores it will
        compare values that share no reference.

        Returns
        -------
        Batches of two Batch
            `model = stack.fit1d_network(...)`, then `stack.predict(model=model)`.

            screen  complex unit phasors per date, NaN where no component
                    reached the pixel. NaN is the answer there: away from the
                    nodes the field is extrapolated, not measured.
            labels  int8. 0 is the largest component, 1 the next, by node
                    count; -1 is nodata. A scene that needs more than 127 has
                    shattered rather than resolved, and says so rather than
                    folding one component's label onto another's. Where two
                    components reach one pixel the screen is taken from the one
                    with the more arcs per node, never averaged between --
                    their datums are unrelated, and mixing them is worse than
                    either alone.

        Examples
        --------
        >>> model = stack.fit1d_network()
        >>> predicted = stack.predict(model=model)
        >>> model['velocity'].plot(cmap='turbo')   # rad/yr
        >>> good = model.where(model['conncomp'] == 0)
        >>> main = velocity.where(labels == 0)   # one datum, comparable
        """
        from . import utils_fit1d
        return utils_fit1d.fit1d_network(self, threshold=threshold, window=window, cell=cell, baseline=baseline,
                                         level=level, max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
                                         step_dv=step_dv, max_seasonal=max_seasonal, consensus=consensus,
                                         err_dh=err_dh, err_dv=err_dv, union=union, iterations=iterations,
                                         debug=debug)

    def fit3d(self, *args, **kwargs):
        """Renamed fit1d_network(): prints one WARNING and calls it with the same arguments (its docstring has them)."""
        print('WARNING: fit3d() is renamed fit1d_network(); fit3d() calls it.')
        return self.fit1d_network(*args, **kwargs)


    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off the complex vars from Stack, PLUS the 1D metadata that rides
        # with them. Keeping only dtype.kind=='c' stranded the radar geometry:
        # radar_wavelength, near_range, earth_radius, SC_height_start,
        # rng_samp_rate and BPR are per-date 1D variables, so pairs() ->
        # BatchComplex dropped every one of them and nothing downstream could
        # convert units or build ele2phase. Batch.__init__ already keeps 1D
        # non-complex variables for exactly this reason; this makes the two
        # agree. Grids stay excluded -- a (y,x) real variable is data, not
        # metadata, and belongs in a Batch.
        if isinstance(mapping, Stack):
            complex_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                keep = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' or ds[v].ndim <= 1
                ]
                complex_dict[key] = ds[keep]
            mapping = complex_dict

        #print('BatchComplex __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})

    @classmethod
    def _keeps(cls, dtype) -> bool:
        """A result keeps the class unless it is a boolean mask (BatchCore._as_class):
        BatchComplex takes any dtype, a mask is a plain Batch in every class."""
        return dtype.kind != 'b'

    @staticmethod
    def _grids_map(ds, f, **kwargs):
        """f on the gridded variables of a Dataset, every other variable carried
        unchanged (the operator rule); f on a DataArray directly."""
        if isinstance(ds, xr.DataArray):
            return f(ds)
        grids = BatchCore._grid_vars(ds)
        res = ds[grids].map(f, **kwargs)
        for v in ds.data_vars:
            if v not in grids:
                res[v] = ds[v]
        return res

    @property
    def real(self):
        """The real part of the grids as a Batch of real-valued Datasets; the metadata passes through unchanged."""
        return Batch({key: BatchComplex._grids_map(ds, lambda da: da.real) for key, ds in self.items()})

    @property
    def imag(self):
        """The imaginary part of the grids as a Batch of real-valued Datasets; the metadata passes through unchanged."""
        return Batch({key: BatchComplex._grids_map(ds, lambda da: da.imag) for key, ds in self.items()})

    def abs(self, **kwargs):
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs))

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        # Optimized: avoid sqrt in abs() by computing real² + imag² directly
        return Batch(self.map_da(lambda da: da.real**2 + da.imag**2, **kwargs))

    def threshold(self, weight=None, threshold=np.pi/2) -> "BatchComplex":
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Computes weighted cstd across all pairs per pixel. Pixels with
        cstd >= threshold are set to 0+0j (all pairs). Useful for rejecting
        incoherent pixels before velocity estimation or detrending.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional correlation weight for weighted cstd.
        threshold : float
            Maximum cstd in radians. Default π/2. Use π/4 for stricter filtering.

        Returns
        -------
        BatchComplex
            Filtered copy with incoherent pixels zeroed.
        """
        import dask.array as da
        import xarray as xr
        from . import utils_detrend

        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, self)
        BatchCore._require_lazy(self, 'threshold')

        results = {}
        for burst_id, burst_ds in self.items():
            burst_weight = weight[burst_id] if weight is not None else None
            filtered_vars = {}
            # THE GRIDS ONLY (N81): the radar metadata a burst carries -- burst
            # strings, near_range, ... -- is not phase, and a name list could not
            # keep up with it; blockwise() raised on the first 1-D variable. A
            # DataArray burst is its own one grid
            for pol, data_da in BatchCore._grids_of(burst_ds).items():
                weight_da = BatchCore._weight_of(burst_weight, pol)

                if data_da.dims[0] != 'pair':
                    data_da = data_da.transpose('pair', ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                n_pairs_val = data_da.shape[0]

                def _threshold_block(data_block, weight_block=None,
                                     _threshold=threshold):
                    return utils_detrend.threshold_pairs_array(
                        [data_block],
                        [weight_block] if weight_block is not None else None,
                        threshold=_threshold,
                    )

                if weight_dask is not None:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )
                else:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )

                filtered_vars[pol] = xr.DataArray(filtered_dask, dims=data_da.dims,
                                                   coords=data_da.coords, name=pol)

            # a DataArray in, a DataArray out (N81)
            results[burst_id] = BatchCore._form(burst_ds, filtered_vars) \
                if isinstance(burst_ds, xr.DataArray) else burst_ds.assign(filtered_vars)

        return BatchComplex(results)

    def backscatter(self, *args, **kwargs):
        """
        Compute backscatter intensity (sigma0) from radiometrically calibrated SLC data.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "backscatter() requires insardev_backscatter extension"
        )

    def adi(self, *args, **kwargs):
        """
        Compute Amplitude Dispersion Index (ADI) for calibrated σ₀ data.

        ADI = std(amplitude) / mean(amplitude) over time.
        Lower ADI indicates more stable scatterers (PS candidates).

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "adi() requires insardev_backscatter extension"
        )

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

    def singlelook(self, baseline: 'str | None' = 'BPR') -> 'BatchComplex':
        """The stack referenced to ONE epoch: `u_d * conj(u_ref)`, per date.

        The scatterer's own phase is constant in time, so one multiplication by
        the reference epoch cancels it and what is left in each date is the
        change since that epoch -- the quantity every later step actually
        models. Nothing is filtered and nothing is resampled: this is still one
        look per pixel, which is what the name says.

        It is also the only form in which neighbouring pixels may be averaged.
        Raw SLC neighbours hold independent speckle phase and their phasors sum
        toward zero; referenced, they share the same temporal phase change and
        the sum is a real multilook -- see multilook(), which is this followed
        by gaussian().

        THE REFERENCE EPOCH IS WHERE THE BASELINE IS SMALLEST, so the height
        term vanishes there and fit1d_network()'s origin is the same epoch. The choice
        is free for anything that differences dates or pixels, since a constant
        per pixel cancels; matching fit1d_network only keeps the conventions aligned.
        `baseline=None` takes the middle epoch. A baseline NAMED AND ABSENT
        raises rather than quietly picking another epoch.

        Returns
        -------
        BatchComplex
            Same dates, same metadata; the reference epoch becomes real and
            positive, since a phasor times its own conjugate is its power.
        """
        import numpy as np

        ds0 = self[list(self.keys())[0]]
        if baseline:
            # the reference epoch is where the Dataset's baseline is smallest (N81)
            BatchCore._needs_dataset(self, 'singlelook', (baseline,),
                                     'Call it on the Dataset batch x, or pass baseline=None.')
            if baseline not in ds0.variables:
                raise KeyError(
                    f'singlelook(): no {baseline!r} to pick the reference epoch from; '
                    f'the stack carries {sorted(v for v in ds0.variables if v not in ds0.dims)}. '
                    f'Pass baseline=None to reference the middle epoch instead.')
            b = np.asarray(ds0[baseline].values, dtype=float).ravel()
            if b.size != ds0.sizes['date']:
                raise ValueError(
                    f'singlelook(): {baseline!r} holds {b.size} values for '
                    f'{ds0.sizes["date"]} dates')
            index = int(np.argmin(np.abs(b)))
        else:
            index = int(ds0.sizes['date']) // 2

        # dates differ between bursts of one acquisition, so the epoch is taken
        # by POSITION and every burst is referenced to its own copy of it
        return self * self.isel(date=index).conj()

    def multilook(self, wavelength: float, baseline: 'str | None' = 'BPR',
                  weight: 'BatchUnit | None' = None, threshold: float = 0.5,
                  debug: bool = False) -> 'BatchComplex':
        """Spatial looks on a per-date complex stack: singlelook() then gaussian().

        Averaging the stack itself is a random walk -- neighbouring SLC pixels
        hold independent speckle phase, so their phasors sum toward zero and
        the ground phase does not survive it. singlelook() first, and the
        neighbours share the SAME temporal phase change, which is the quantity
        being averaged; the same filter is then a real multilook.

        THIS IS WHAT A DISTRIBUTED SCATTERER NEEDS BEFORE A PER-PIXEL FIT.
        fit1d_network() builds every arc from single pixel columns, so a candidate's
        own phase noise is COMMON to all of its arcs: no number of partners, no
        `consensus` and no `err_dv` can average it out, because the consensus
        can only reject the pixel, never improve it. Supply the looks here and
        the fit is handed a phase it can resolve; leave them out and a
        vegetated target survives only with its gates opened, which is the same
        as not filtering at all.

        Parameters
        ----------
        wavelength : float
            Gaussian cutoff in metres, as in interferogram(). It states the
            ground scale over which the scatterers are taken to be alike, and
            nothing here guesses it. Where the posting is finer than the radar
            resolution the neighbours are already correlated, so looks taken up
            to that scale cost no resolution the data ever held.
        baseline : str or None
            Passed to singlelook() to name the reference epoch.
        weight, threshold, debug
            Passed to gaussian().

        Returns
        -------
        BatchComplex
            The referenced stack, spatially multilooked. Same dates, same
            metadata, still one value per pixel -- no resampling.

        Examples
        --------
        >>> model = stack.multilook(wavelength=60).fit1d_network(threshold=0)
        """
        if not wavelength > 0:
            raise ValueError(f'multilook() needs a positive wavelength in metres, got {wavelength}')
        if baseline:
            # singlelook()'s reference epoch is where the Dataset's baseline is smallest (N81)
            BatchCore._needs_dataset(self, 'multilook', (baseline,),
                                     'Call it on the Dataset batch x, or pass baseline=None.')
        # a BatchUnit (N81), checked here; gaussian() fits it to the grids
        BatchCore._weight(weight)
        return self.singlelook(baseline).gaussian(
            weight=weight, wavelength=wavelength, threshold=threshold, debug=debug)

    def pairs(self, pairs):
        """Select date pairs from per-date data, returning ref and rep stacks.

        Parameters
        ----------
        pairs : array-like (n_pairs, 2)
            Pairs as [[ref_date, rep_date], ...]. Dates as datetime64 or indices.

        Returns
        -------
        tuple (ref, rep)
            Two BatchComplex with 'pair' dimension instead of 'date'.
        """
        import numpy as np
        pairs = np.asarray(pairs)
        ref_dates = pairs[:, 0]
        rep_dates = pairs[:, 1]

        # Map dates to integer indices (match by day to handle precision differences)
        key0 = list(self.keys())[0]
        date_coords = self[key0].coords['date'].values
        # Truncate to day precision for matching
        date_days = np.array(date_coords, dtype='datetime64[D]')
        date_to_idx = {d: i for i, d in enumerate(date_days)}
        ref_idx = [date_to_idx[np.datetime64(d, 'D')] for d in ref_dates]
        rep_idx = [date_to_idx[np.datetime64(d, 'D')] for d in rep_dates]

        # Select, rename date→pair, and assign pair coords matching the caller.
        # The per-date 'residual' of Stack.align() is dropped, as Stack.pairs() does
        n_pairs = len(ref_idx)
        pair_coords = np.arange(n_pairs)
        screen_ref = self.isel(date=ref_idx).rename(date='pair').map(
            lambda ds: ds.drop_vars('residual', errors='ignore').assign_coords(pair=pair_coords))
        screen_rep = self.isel(date=rep_idx).rename(date='pair').map(
            lambda ds: ds.drop_vars('residual', errors='ignore').assign_coords(pair=pair_coords))

        return screen_ref, screen_rep

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            if isinstance(ds, xr.DataArray):
                # a Batch of DataArrays takes angle() directly
                out[k] = xr.ufuncs.angle(ds).astype('float32')
                continue
            # the COMPLEX GRIDS, by the operator rule; a Dataset without grids raises
            grids = BatchCore._grid_vars(ds)
            complex_vars = [
                var for var in grids
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray. EVERY
            # non-gridded variable rides along afterwards -- angle() of
            # near_range is meaningless, but DROPPING it strands every
            # downstream unit conversion, which is how radar_wavelength went
            # missing. A real grid (a Stack's ele) is not a phase and is left out.
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            meta = [v for v in ds.data_vars if v not in grids]
            if meta:
                ds_phase = ds_phase.assign({v: ds[v] for v in meta})
            ds_phase.attrs = ds.attrs
            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def unwrap2d(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion, see BatchWrap.unwrap2d().

        The solver stops at `accuracy` (default 2e-3; 5e-4 near the optimum, about 1.9 times
        longer); the former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.
        """
        return self.angle().unwrap2d(*args, **kwargs)

    def unwrap2d_irls(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion, see BatchWrap.unwrap2d_irls().

        The solver stops at `accuracy` (default 2e-3; 5e-4 near the optimum, about 1.9 times
        longer); the former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.
        """
        return self.angle().unwrap2d_irls(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot complex phase as wrapped phase via .angle() conversion.
        """
        return self.angle().plot(*args, **kwargs
        )

    @staticmethod
    def _goldstein(phase_np, corr_np, psize=32, threshold=0.5, skip=(0, 0)):
        """
        Apply Goldstein adaptive filter (utils_goldstein.goldstein).

        Parameters
        ----------
        phase_np : np.ndarray
            2D complex numpy array of phase data.
        corr_np : np.ndarray
            2D real numpy array of correlation values.
        psize : int or dict
            Patch size for the filter. Default is 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        skip : tuple of int
            (rows, columns) at the start of the block that no patch covers: the patch grid starts there (NaN
            before it). For a block of a larger array, whose patch grid belongs to the whole array.

        Returns
        -------
        np.ndarray
            Filtered complex array with same shape as input.
        """
        import numpy as np
        from .utils_goldstein import goldstein

        if psize is None:
            return phase_np

        # Handle (1, y, x) arrays from apply_ufunc
        squeeze = False
        if phase_np.ndim == 3 and phase_np.shape[0] == 1:
            phase_np = phase_np[0]
            corr_np = corr_np[0] if corr_np.ndim == 3 else corr_np
            squeeze = True

        if isinstance(psize, dict):
            psize_y, psize_x = psize['y'], psize['x']
        else:
            psize_y, psize_x = int(psize), int(psize)
        # the kernels mirror a half-size quadrant into the patch weight, so
        # the patch has to be even; see goldstein() above
        psize_y, psize_x = 2 * (int(psize_y) // 2), 2 * (int(psize_x) // 2)
        # the triangular weight divides by psize/2 - 1, so a 2-pixel patch has
        # no weight at all and the whole output would come back NaN
        if psize_y < 4 or psize_x < 4:
            raise ValueError(f'goldstein needs a patch of at least 4 pixels '
                             f'per axis after rounding down to even, got '
                             f'({psize_y}, {psize_x})')

        # Ensure correct dtypes (goldstein functions require complex64/float32)
        if phase_np.dtype != np.complex64:
            phase_np = phase_np.astype(np.complex64)
        if corr_np.dtype != np.float32:
            corr_np = corr_np.astype(np.float32)

        result = goldstein(phase_np, corr_np, psize_y, psize_x, threshold=threshold, skip=skip)

        if squeeze:
            result = result[np.newaxis, ...]
        return result

    def goldstein(self, corr: BatchUnit, window: 'float | tuple | dict' = 200, threshold: float = 0.5,
                  debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.

        The patches sit at multiples of half a patch counted from pixel 0 of each burst's grid, so the result
        depends on the filter window and on the burst's grid, and does not depend on the dask chunks (up to float
        rounding). It does depend on cropping the burst: a crop starts its own patch grid. Each patch's spectrum S
        is weighted by (|S| / max |S|)^alpha, alpha = 1 - the patch's mean correlation, so the phase does not
        depend on the amplitude scale of the interferogram.

        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        window : float or tuple or dict, optional
            Patch size in METRES on the ground, divided by each burst's
            posting here. A single number is a square patch; a pair or a
            {'y': ..., 'x': ...} dict states the axes apart. Default 200.

            The pixel count it becomes is rounded DOWN to even per axis: the
            filter weight is a mirrored quadrant, so odd patches have no
            weight. A patch longer than a burst raises ValueError, at the call.
        threshold : float, optional
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        BatchComplex
            New batch with filtered phase values
        """
        import numpy as np
        import dask.array as da

        if debug:
            print('DEBUG: goldstein')

        if window is None:
            return self

        # AN INHERENTLY SPATIAL FILTER: a burst without a (y, x) grid -- a point's series, an
        # area mean -- raises the short error (N81), not a dask chunk error further down
        for ds in self.values():
            if not BatchCore._grids_of(ds):
                raise BatchCore._no_grid_error(ds)

        # a BatchUnit for every burst, the shared check and messages (N81); a DataArray
        # BatchUnit weights every grid, a Dataset one names every grid
        corr = BatchCore._weight(corr, self, 'corr', required=True)

        # Validate lazy data
        BatchCore._require_lazy(self, 'goldstein')

        # METRES IN, PIXELS PER BURST. The patch is a ground size, so each
        # burst divides it by its own posting inside the loop below.
        if isinstance(window, dict):
            window_m = (window['y'], window['x'])
        else:
            window_m = window

        # THE PATCH OF EVERY BURST, CHECKED BEFORE ANYTHING IS BUILT.
        windows = {}
        for k in self.keys():
            ds = self[k]
            _wy, _wx = utils_xarray.meters_to_pixels(
                window_m, utils_xarray.spacing_of(ds), minimum=4,
                name='goldstein() window')
            # EVEN PATCHES, BECAUSE THE WEIGHT IS BUILT BY MIRRORING A
            # QUADRANT. The triangular weight is a (psize//2, psize//2)
            # quadrant reflected twice, so its side is 2*(psize//2) -- equal
            # to the patch only when the patch is even. An odd patch used to
            # fail deep inside the filter with a numpy broadcast error about
            # shapes one sample apart. Rounded DOWN, so the patch never grows
            # past what the caller asked for.
            window = {'y': 2 * (_wy // 2), 'x': 2 * (_wx // 2)}
            if window['y'] < 4 or window['x'] < 4:
                raise ValueError(
                    f'goldstein() window {window_m} m is under four pixels per '
                    f'axis on burst {k} after rounding down to even')
            # A patch longer than the burst has no room in it: the crop is too small
            if window['y'] > ds.sizes['y'] or window['x'] > ds.sizes['x']:
                raise ValueError(
                    f"goldstein() window {window_m} m is {window['y']} x {window['x']} pixels, longer than burst {k} "
                    f"of {ds.sizes['y']} x {ds.sizes['x']} pixels: crop a larger area or use a shorter window")
            windows[k] = window

        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            corr_ds = corr[k]
            filtered_vars = {}
            window = windows[k]

            # Process each complex data variable in the dataset; a DataArray burst is
            # its own one (N81)
            for var_name, var_data in BatchCore._vars_of(ds).items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    corr_da = BatchCore._weight_of(corr_ds, var_name)
                    phase_dask = var_data.data
                    corr_dask = corr_da.data

                    # Require first dimension chunked as 1 (avoid hidden rechunking overhead)
                    chunks = phase_dask.chunks
                    if var_data.ndim == 3 and chunks[0][0] != 1:
                        raise ValueError(
                            f"goldstein() requires first dimension chunked as 1, got chunks {chunks[0]}. "
                            f"Data should already have pair=1 chunks from load()."
                        )

                    # Overlap depth: the patch size. Every patch that touches a pixel of the chunk, at its place
                    # on the grid of the whole array, then lies inside the block (a patch is `window` long and
                    # starts at a multiple of half of it). The depth is handed to _apply_goldstein_2d_for_dask as halo.
                    depth_y = window['y']
                    depth_x = window['x']

                    if debug:
                        print(f'DEBUG: goldstein map_overlap depth=({depth_y}, {depth_x})')

                    depth_2d = {0: depth_y, 1: depth_x}
                    depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                    filtered_dask = da.map_overlap(
                        _apply_goldstein_2d_for_dask,
                        phase_dask,
                        corr_dask,
                        depth= depth_3d if var_data.ndim == 3 else depth_2d,
                        boundary='none',
                        dtype=np.complex64,
                        psize=window,
                        threshold=threshold,
                        halo=(depth_y, depth_x),
                    )

                    filtered_vars[var_name] = xr.DataArray(
                        filtered_dask,
                        dims=var_data.dims,
                        coords=var_data.coords
                    )
                else:
                    filtered_vars[var_name] = var_data

            # Create a new dataset with the filtered variables; a DataArray in, a
            # DataArray out (N81)
            result[k] = BatchCore._form(
                ds,
                filtered_vars,
                coords=ds.coords,
                attrs=ds.attrs
            )

        return type(self)(result)


def _subtract_date_from_pair(first, second):
    """Subtract per-date atmospheric screens from per-pair data.

    Uses BatchComplex.pairs() to select ref/rep screens,
    then: result = data * conj(screen_ref) * screen_rep
    """
    import numpy as np

    key0 = list(first.keys())[0]
    ref_dates = first[key0].coords['ref'].values
    rep_dates = first[key0].coords['rep'].values
    pairs = np.column_stack([ref_dates, rep_dates])

    screen_ref, screen_rep = second.pairs(pairs)
    # Rechunk screens to match first's chunks — isel produces fragmented
    # dim0 chunks (63,63,...,31) while first has merged (1165,) chunks.
    # Without this, the multiply triggers an implicit rechunk of first
    # that re-reads the full upstream graph 19× per spatial tile.
    ref_var = next(v for v in first[key0].data_vars if first[key0][v].ndim >= 3)
    ref_chunks = first[key0][ref_var].data.chunks
    for batch in (screen_ref, screen_rep):
        for key in batch:
            ds = batch[key]
            rechunked = {}
            for v in ds.data_vars:
                da_xr = ds[v]
                if da_xr.ndim >= 3 and hasattr(da_xr.data, 'chunks') and da_xr.data.chunks != ref_chunks:
                    rechunked[v] = da_xr.chunk(dict(zip(da_xr.dims, ref_chunks)))
            if rechunked:
                batch[key] = ds.assign(rechunked)
    return first * screen_ref.conj() * screen_rep


class Batches(tuple):
    """
    A tuple-like container for multiple Batch objects that allows chained operations.

    Enables operations like:
        mintf, mcorr = stack.phasediff(...).downsample(20).compute()
        mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        mintf, mcorr = Batches.open('mintf_corr')

    Instead of:
        mintf, mcorr = stack.phasediff(...)
        mintf, mcorr = stack.compute(mintf.downsample(20), mcorr.downsample(20))
    """

    def __new__(cls, batches=()):
        return super().__new__(cls, batches)

    @staticmethod
    def _preserve_nonspatial(source, target):
        """Copy non-spatial variables (e.g. BPR) from source to target batch."""
        import dask.array as da
        for key in source:
            src_ds = source[key]
            tgt_ds = target[key]
            extra = {}
            for v in src_ds.data_vars:
                if v not in tgt_ds.data_vars:
                    var = src_ds[v]
                    if not isinstance(var.data, da.Array):
                        var = var.chunk()
                    extra[v] = var
            if extra:
                target[key] = tgt_ds.assign(extra)
        return target

    def phase(self) -> 'BatchComplex | BatchWrap | Batch | None':
        """Extract phase from Batches.

        Returns the first BatchComplex, or first BatchWrap,
        or first Batch found. Returns None if none found.
        """
        for b in self:
            if isinstance(b, BatchComplex):
                return b
        for b in self:
            if isinstance(b, BatchWrap):
                return b
        for b in self:
            if isinstance(b, Batch) and not isinstance(b, BatchUnit):
                return b
        return None

    def correlation(self) -> 'BatchUnit | None':
        """Extract correlation weights from Batches.

        Returns the first BatchUnit found, or None.
        """
        for b in self:
            if isinstance(b, BatchUnit):
                return b
        return None

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None,
                 debug: bool = False, overwrite: bool = True, **kwargs):
        """Save or open a Batches snapshot.

        When called on a Batches with data, saves all batches to Zarr store.
        When called on an empty Batches(), opens an existing store.

        Parameters
        ----------
        store : str
            Path to the Zarr store.
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        debug : bool
            Print debug information.
        overwrite : bool
            True (default) replaces an existing store; False keeps a complete one and returns it opened.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        >>> # Open
        >>> mintf, mcorr = Batches().snapshot('mintf_corr')
        """
        from . import utils_io

        if len(self) == 0:
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       caption=caption or 'Opening...',
                                       debug=debug)
        else:
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       caption=caption or 'Snapshotting...',
                                       debug=debug, overwrite=overwrite, wrapper=Batches)

        if isinstance(result, Batches):
            return result
        return Batches((result,))  # fallback for stores without __wrapper__

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                debug: bool = False):
        """Save or open a Batches archive as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Parameters
        ----------
        store : str
            Path to the ZIP file. Must end with '.zip'.
        caption : str, optional
            Progress bar caption.
        compression : int
            ZIP compression level 0-9 (0=no compression, 9=max). Default 6.
            Higher values produce smaller files but take longer.
        debug : bool
            Print debug information.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save to zip
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).archive('mintf_corr.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> mintf, mcorr = stack.phasediff(...).archive('mintf_corr.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> mintf, mcorr = stack.phasediff(...).archive('gs://bucket/mintf_corr.zip')
        >>> # Open from zip
        >>> mintf, mcorr = Batches().archive('mintf_corr.zip')
        """
        import zarr
        import zipfile
        import tempfile
        import os
        import fsspec

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

        if len(self) == 0:
            # Open mode - check file exists first
            if is_cloud:
                fs, path = fsspec.core.url_to_fs(store)
                if not fs.exists(path):
                    raise FileNotFoundError(f"Archive not found: {store}")
            elif not os.path.exists(store):
                raise FileNotFoundError(f"Archive not found: {store}")
            # Use ZipStore directly for reading
            zip_store = zarr.storage.ZipStore(store, mode='r')
            result = self.snapshot(store=zip_store, caption=caption or 'Opening archive...', debug=debug)
            zip_store.close()
            return result
        else:
            # Save mode - write to temp directory, then zip
            # This avoids ZipStore's duplicate entry problem
            temp_dir = tempfile.mkdtemp()
            try:
                result = self.snapshot(store=temp_dir, caption=caption or 'Archiving...', debug=debug)
                # Create zip with specified compression level
                # Use fsspec for cloud storage support
                with fsspec.open(store, 'wb') as f:
                    with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression) as zf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zf.write(file_path, arcname)
            finally:
                import shutil
                shutil.rmtree(temp_dir)
            return result

    def downsample(self, *args, **kwargs):
        """Apply downsample to all batches."""
        return Batches([b.downsample(*args, **kwargs) for b in self])

    def chunk(self, *args, **kwargs):
        """Apply chunk to all batches."""
        return Batches([b.chunk(*args, **kwargs) for b in self])

    def chunk2d(self, *args, **kwargs):
        """Apply chunk2d to all batches."""
        return Batches([b.chunk2d(*args, **kwargs) for b in self])

    def chunk1d(self, *args, **kwargs):
        """Apply chunk1d to all batches."""
        return Batches([b.chunk1d(*args, **kwargs) for b in self])

    def where(self, cond, other=np.nan, **kwargs):
        """Apply where mask to all batches."""
        return Batches([b.where(cond, other, **kwargs) for b in self])

    def crop(self, *args, **kwargs):
        """Apply crop to all batches."""
        return Batches([b.crop(*args, **kwargs) for b in self])

    def sel(self, *args, **kwargs):
        """Apply sel to all batches."""
        return Batches([b.sel(*args, **kwargs) for b in self])

    def isel(self, *args, **kwargs):
        """Apply isel to all batches."""
        return Batches([b.isel(*args, **kwargs) for b in self])

    def filter(self, days=None, meters=None, date=None, pair=None, count=None,
               min_connections=2, cleanup=True):
        """Filter pairs in the baseline network.

        Selects pairs matching the given temporal/spatial criteria. By default,
        removes degraded dates (hanging or single-side connected).

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline in meters. If None, no limit.
        date : str or list, optional
            Date(s) to exclude. Accepts a single date string or a list,
            any format parseable by ``pd.to_datetime``.
        pair : str or list, optional
            Pair(s) to exclude. Each pair is a string ``'YYYY-MM-DD YYYY-MM-DD'``.
            Accepts a single pair string or a list.
        count : int, optional
            Remove dates with fewer than this many connections.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        cleanup : bool, optional
            If True (default), iteratively remove hanging dates and dates
            connected only to predecessors or only to successors.
            Set to False to keep the raw network for testing.

        Returns
        -------
        Batches
            Filtered Batches with valid pairs only.

        Examples
        --------
        >>> stack.filter(days=100, meters=80).unwrap3d()
        >>> stack.filter(date='2024-12-30').unwrap3d()
        >>> intfcorr.filter(date=['2024-12-30', '2024-06-21'])
        >>> intfcorr.filter(pair='2024-06-21 2024-12-30')
        >>> intfcorr.filter(count=3)  # remove dates with < 3 connections
        >>> intfcorr.filter(days=100, meters=80, cleanup=False)  # raw network
        """
        import numpy as np
        import pandas as pd

        if days is None and meters is None and date is None and pair is None and count is None:
            return self

        # Get pair coordinates from the first batch element
        first_batch = self[0]
        first_key = next(iter(first_batch.keys()))
        ds = first_batch[first_key]
        ref = pd.DatetimeIndex(ds.coords['ref'].values).normalize()
        rep = pd.DatetimeIndex(ds.coords['rep'].values).normalize()
        bpr = ds.coords['BPR'].values
        n_pairs = len(ref)

        # Build mask of valid pairs
        mask = np.ones(n_pairs, dtype=bool)

        if days is not None:
            duration = (rep - ref).days
            mask &= duration <= days

        if meters is not None:
            mask &= np.abs(bpr) <= meters

        if date is not None:
            if isinstance(date, str):
                date = [date]
            exclude_dates = pd.to_datetime(date).normalize()
            mask &= ~ref.isin(exclude_dates) & ~rep.isin(exclude_dates)

        if pair is not None:
            if isinstance(pair, str):
                pair = [pair]
            exclude_pairs = set()
            for p in pair:
                parts = str(p).split()
                r, s = pd.Timestamp(parts[0]).normalize(), pd.Timestamp(parts[1]).normalize()
                exclude_pairs.add((r, s))
            for i in range(n_pairs):
                if (ref[i], rep[i]) in exclude_pairs:
                    mask[i] = False

        # Build DataFrame for pruning
        df = pd.DataFrame({'ref': ref[mask], 'rep': rep[mask],
                           'idx': np.where(mask)[0]})

        if len(df) > 0:
            from .Baseline import _cleanup_network
            min_conn = max(min_connections, count) if count is not None else min_connections
            if cleanup:
                df = _cleanup_network(df, min_connections=min_conn)
            elif count is not None:
                counts = pd.concat([df['ref'], df['rep']]).value_counts()
                low_dates = set(counts[counts < count].index)
                df = df[~df['ref'].isin(low_dates) & ~df['rep'].isin(low_dates)]

        if len(df) == 0:
            raise ValueError("No valid pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")

        valid_idx = df['idx'].values
        return self.isel(pair=valid_idx)

    def coherent(self, threshold=0.5):
        """Mask low-coherence pixels using mean correlation.

        Computes mean correlation across pairs from the BatchUnit item
        and sets pixels with mean correlation below threshold to NaN
        in all batch items.

        Parameters
        ----------
        threshold : float
            Minimum mean correlation to keep. Default 0.5.

        Returns
        -------
        Batches
            Same structure with NaN where mean correlation < threshold.
        """
        corr = next((b for b in self if isinstance(b, BatchUnit)), None)
        if corr is None:
            raise ValueError('coherent() requires a BatchUnit (correlation) in Batches')
        results = []
        for b in self:
            # the shared weight check (N81): a burst corr lacks, or a DataArray corr without
            # (y, x) (BatchUnit(corr['BPR'])), raises. Not by name: one correlation grid
            # masks every grid of every batch
            BatchCore._weight(corr, b, 'corr', by_name=False)
            out = {}
            for key in b:
                corr_ds = corr[key]
                if isinstance(corr_ds, xr.DataArray):
                    # a DataArray BatchUnit (N81) is its own one grid
                    corr_da = corr_ds
                else:
                    corr_var = next(v for v in corr_ds.data_vars if 'y' in corr_ds[v].dims)
                    corr_da = corr_ds[corr_var]
                mask = corr_da.mean('pair') >= threshold if 'pair' in corr_da.dims else corr_da >= threshold
                ds = b[key]
                # THE GRIDS ONLY (N81): xarray's where() on the whole Dataset turned
                # the string metadata into objects that failed on compute
                out[key] = ds.where(mask) if isinstance(ds, xr.DataArray) else ds.assign(
                    {v: ds[v].where(mask) for v in ds.data_vars if BatchCore._is_grid(ds[v])})
            results.append(type(b)(out))
        return Batches(results)

    def angle(self):
        """Apply angle() to BatchComplex batches, return others unchanged.

        Returns
        -------
        Batches
            Batches with BatchComplex converted to BatchWrap (phase angles),
            other batch types unchanged.

        Examples
        --------
        >>> phase, corr = stack.phasediff2(pairs).angle()
        >>> # phase is now BatchWrap with angles, corr is unchanged BatchUnit
        """
        results = []
        for b in self:
            if isinstance(b, BatchComplex):
                results.append(b.angle())
            else:
                results.append(b)
        return Batches(results)

    def goldstein(self, window: 'float | tuple' = 200, threshold: float = 0.5):
        """Apply Goldstein filter to phase using correlation as weight.

        Expects Batches with [BatchComplex (phase), BatchUnit (correlation)].

        Parameters
        ----------
        window : float or tuple
            Goldstein filter patch size in METRES on the ground, default 200.
            Converted per burst and rounded down to an even pixel count.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.

        Returns
        -------
        Batches
            Batches with Goldstein-filtered phase and unchanged correlation.

        Examples
        --------
        >>> intf, corr = stack.pairs(pairs).interferogram(30).goldstein(200)
        """
        if len(self) < 2:
            raise ValueError("goldstein() requires Batches with at least 2 elements: [interferogram, correlation]")

        phase, corr = self[0], self[1]

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"First element must be BatchComplex, got {type(phase).__name__}")
        # the correlation is the weight: the shared check and messages (N81)
        BatchCore._weight(corr, phase, 'corr', required=True)

        filtered_phase = phase.goldstein(corr, window, threshold=threshold)
        return Batches([filtered_phase, corr] + list(self[2:]))

    def interferogram(self,
                  wavelength: float | None = None,
                  weight: 'BatchUnit | None' = None,
                  phase: 'BatchComplex | None' = None,
                  gaussian_threshold: float = 0.5) -> 'Batches':
        """
        Compute phase difference from paired SLC data.

        Expects Batches from pairs() with [ref, rep] BatchComplex objects.

        Parameters
        ----------
        wavelength : float or None
            Filter wavelength in metres, passed to gaussian(). FIRST, as
            gaussian() and multilook() take it -- it is the argument that
            decides what comes back, and it used to sit third behind two
            optional objects, so `interferogram(400)` bound 400 to the weight.

            None IS A MODE, not a missing value: the interferogram is returned
            single-look and there is no correlation to report with it, so the
            result holds ONE element instead of two. Unpacking
            `phase, corr = ...` then fails, which is the intended signal that
            a correlation was never computed.
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking.
        phase : BatchComplex or None
            Optional phase to subtract (e.g., topographic phase).
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).

        Returns
        -------
        Batches
            Batches with [interferogram, correlation]: the interferogram COMPLEX
            (BatchComplex) until angle() is called -- unwrap2d() and
            unwrap2d_irls() call it themselves -- and the correlation a BatchUnit.

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(30)
        >>> # single-look, ONE element back -- no correlation without a filter
        >>> intf, = stack.pairs(baseline.tolist()).interferogram()
        """
        if len(self) != 2:
            raise ValueError("interferogram() requires Batches with exactly 2 elements: [ref, rep]")

        ref, rep = self[0], self[1]

        if not isinstance(ref, BatchComplex) or not isinstance(rep, BatchComplex):
            raise TypeError("Both elements must be BatchComplex")

        # A WEIGHT OR A PHASE WHERE THE WAVELENGTH GOES is the old argument
        # order; caught here so it says so rather than failing later on a
        # comparison against a Batch.
        if wavelength is not None and not isinstance(wavelength, (int, float)):
            raise TypeError(
                f'interferogram() takes the wavelength first now, as gaussian() '
                f'does, and got a {type(wavelength).__name__}. Pass the others '
                f'by name: interferogram(wavelength, weight=..., phase=...)')

        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, ref)

        intf = ref * rep.conj()
        if phase is not None:
            if isinstance(phase, BatchComplex):
                intf = intf * phase
            else:
                intf = intf * phase.iexp(-1)

        if wavelength is not None:
            intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            del ref, rep
            corr_look = (intf_look.abs() / (intensity_ref * intensity_rep).sqrt()).clip(0, 1)
            del intensity_ref, intensity_rep
        else:
            intf_look = intf
            corr_look = None
            del ref, rep
        del intf

        if weight is not None:
            intf_look = intf_look.where(weight.isfinite())
            corr_look = corr_look.where(weight.isfinite()) if corr_look else None

        if corr_look is None:
            return Batches([intf_look])
        return Batches([intf_look, corr_look])

    def interferogram2(self, *args, **kwargs):
        """
        Compute optimized interferogram using dual-polarization coherence optimization.

        This method requires the insardev_polsar extension package.
        """
        raise ImportError(
            "interferogram2() requires insardev_polsar extension"
        )

    def compute(self):
        """Compute all batches at once via dask.persist().

        Persists all bursts across all batches in a single scheduler
        submission. Data stays in worker memory. Preserves shared computation
        between dependent batches (e.g., phase and correlation). For
        memory-constrained sequential processing, use snapshot().

        A batch of DataArrays is computed as the DataArrays it holds, and a
        batch with nothing lazy comes back as it is (BatchCore.compute, N81).

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        from .utils_dask import progress_persisted

        lazy = [BatchCore._is_lazy_any(batch) for batch in self]
        if not any(lazy):
            return self

        # Save input chunk structure per batch per burst
        all_input_chunks = [{key: BatchCore._input_chunks(v) for key, v in batch.items()}
                            for batch, l in zip(self, lazy) if l]

        # Persist all lazy batches at once — single scheduler submission
        # progress_persisted extracts futures and blocks until completion
        all_results = list(dask.persist(*[dict(batch) for batch, l in zip(self, lazy) if l]))
        progress_persisted(all_results, desc='Computing bursts'.ljust(25))

        # Finalize: materialize coordinates, hold the futures, rechunk to match input
        computed = iter(zip(all_results, all_input_chunks))
        computed_batches = []
        for batch, l in zip(self, lazy):
            if not l:
                computed_batches.append(batch)
                continue
            result, input_chunks = next(computed)
            computed_batches.append(type(batch)({key: BatchCore._persisted(v, input_chunks[key])
                                                 for key, v in result.items()}))
        return Batches(computed_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, union=False,
                 debug=False, **kwargs):
        """
        Unwrap phase: L¹ IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Parameters
        ----------
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        union : bool
            False (default) solves each burst on its own, which is the form
            that scales. True unions the bursts and solves once over the
            result, so the answer is consistent across burst edges -- viable
            while the merged scene fits. A Batch either way: the merge is
            internal, and each burst comes back holding only its own pixels.
        debug : bool
            Print diagnostic information.
        **kwargs
            Passed to unwrap2d_irls(): accuracy (default 2e-3; 5e-4 near the optimum,
            about 1.9 times longer), max_iter (the safety cap, default 100), epsilon.
            The former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, Batch conncomp).

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(100).angle()
        >>> unwrapped = intf.unwrap2d()  # Without weights
        >>> unwrapped = Batches([intf, corr]).unwrap2d()  # With weights
        """
        if len(self) < 1:
            raise ValueError("unwrap2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, union=union,
                              debug=debug, **kwargs)

    def subtract(self):
        """
        Subtract the next same-type batch from the first batch.

        Finds the first batch element, then the next element of the same type,
        subtracts the second from the first, replaces the first with the result,
        and drops the second.

        Type-specific subtraction:
        - BatchComplex: first * conj(second) (phase subtraction on unit circle)
        - Batch: first - second (real subtraction)
        - BatchWrap: wrap(first - second) (wrapped phase subtraction)
        - BatchUnit: not supported (raises error)

        Returns
        -------
        Batches
            With first element replaced by subtracted result, second dropped.

        Examples
        --------
        >>> intfcorr = Batches([intf, model_intf]).subtract()
        """
        if len(self) < 2:
            raise ValueError("subtract() requires at least 2 elements")

        first_type = type(self[0])
        if isinstance(self[0], BatchUnit):
            raise TypeError("subtract() cannot be applied to BatchUnit")

        # Find next element of the same type
        second_idx = None
        for i in range(1, len(self)):
            if type(self[i]) is first_type:
                second_idx = i
                break
        if second_idx is None:
            raise ValueError(f"subtract() requires a second {first_type.__name__} element")

        first = self[0]
        second = self[second_idx]

        # Check if second is per-date and first is per-pair
        is_date_to_pair = False
        for key in first.keys():
            first_ds = first[key]
            second_ds = second[key]
            first_pol = [v for v in first_ds.data_vars if 'y' in first_ds[v].dims][0]
            second_pol = [v for v in second_ds.data_vars if 'y' in second_ds[v].dims][0]
            if 'pair' in first_ds[first_pol].dims and 'date' in second_ds[second_pol].dims:
                is_date_to_pair = True
            break

        if is_date_to_pair:
            result = _subtract_date_from_pair(first, second)
        elif isinstance(first, BatchComplex):
            result = first * second.conj()
        else:
            result = first - second

        result = Batches._preserve_nonspatial(first, result)

        elements = list(self)
        elements[0] = result
        elements.pop(second_idx)
        return Batches(elements)

    def displacement_los(self, transform):
        """
        Convert phase to line-of-sight displacement (meters).

        Applies Batch.displacement_los(transform) to the first element.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch or Stack providing radar_wavelength.

        Returns
        -------
        Batches
            Batches with [displacement Batch] preserving other elements.

        Examples
        --------
        >>> disp = stack.unwrap3d().displacement_los(stack.transform())
        """
        data = self[0]
        result = data.displacement_los(transform)
        elements = [result] + list(self[1:])
        return Batches(elements)

    def threshold(self, threshold=np.pi/2):
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Pixels with cstd >= threshold are set to NaN. The second element,
        if any, is the correlation weight and must be a BatchUnit.

        Parameters
        ----------
        threshold : float
            Maximum cstd in radians. Default π/2.

        Returns
        -------
        Batches
            Batches with filtered phase, preserving other elements.
        """
        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"threshold() requires BatchComplex, got {type(phase).__name__}")

        filtered = phase.threshold(weight=weight, threshold=threshold)
        elements = [filtered] + list(self[1:])
        return Batches(elements)

    def fit1d(self, **kwargs):
        """
        Per-pixel model from the first element, dispatching on its type.

        Returns
        -------
        Batch
            The model: velocity, height, coherence, rmse, and `seasonal` from
            the variants that fit an annual term. Identical in name and unit to
            fit1d_network()'s, so predict() consumes either.
        """
        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)
        # BatchComplex.fit1d() normalises every sample to a unit phasor, so a
        # magnitude weight has nothing to act on and it takes none -- complex
        # data carries no correlation. Only the unwrapped fit is offered one;
        # a weight given with complex data raises instead of being dropped.
        if weight is not None and isinstance(phase, BatchComplex):
            raise TypeError('ERROR: fit1d() of complex data takes no weight. '
                            'Use Batches([phase]).fit1d().')
        if weight is None:
            return phase.fit1d(**kwargs)
        return phase.fit1d(weight=weight, **kwargs)

    def rmse(self, solution):
        """RMSE of phase vs solution, using correlation weight if present.

        Extracts phase from self[0] and optional weight from self[1] (BatchUnit).
        Weight is automatically passed to the RMSE calculation and reduced
        to (y, x) via mean over the temporal dimension.

        Parameters
        ----------
        solution : Batch
            Velocity (y, x), pair-based, or date-based solution.

        Returns
        -------
        Batches
            [RMSE Batch (y, x), mean weight BatchUnit (y, x)] when weight present,
            [RMSE Batch (y, x)] otherwise.
        """
        if len(self) < 1:
            raise ValueError("rmse() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        rmse_result = phase.rmse(solution, weight=weight)

        if weight is not None:
            # Reduce weight to (y, x) — detect temporal dimension
            w_sample = next(iter(dict.values(weight)))
            # a DataArray weight (N81) is its own one grid, taken as it is
            w_spatial = [w_sample] if isinstance(w_sample, xr.DataArray) else \
                [w_sample[v] for v in w_sample.data_vars if 'y' in w_sample[v].dims]
            tdim = next((d for d in ('pair', 'date')
                         if w_spatial and d in w_spatial[0].dims), None)
            reduced_weight = weight.mean(tdim) if tdim else weight
            elements = [rmse_result, reduced_weight]
        else:
            elements = [rmse_result]

        return Batches(elements)

    def stl(self, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Expects Batches with [Batch (time series data)]. No weight parameter needed.

        Parameters
        ----------
        freq : str
            Frequency string for resampling (default 'W' for weekly).
        periods : int
            Number of periods for seasonal decomposition (default 52).
        robust : bool
            Whether to use robust fitting. Default False.

        Returns
        -------
        Batch
            Batch of Datasets with the 'trend', 'seasonal' and 'residual' variables
            (one (y, x) variable in, see Batch.stl()).

        Examples
        --------
        >>> stl_result = Batches([displacement]).stl(freq='W', periods=52)
        """
        if len(self) < 1:
            raise ValueError("stl() requires Batches with at least 1 element: [data]")

        data = self[0]

        # Delegate to Batch.stl
        return data.stl(freq=freq, periods=periods, robust=robust)

    def __getattr__(self, name):
        """Proxy unknown attributes to all batches if they're callable."""
        if name.startswith('_'):
            raise AttributeError(f"Batches has no attribute '{name}'")

        # Check if all batches have this attribute and it's callable
        attrs = [getattr(b, name, None) for b in self]
        if all(callable(a) for a in attrs if a is not None):
            def method(*args, **kwargs):
                results = [getattr(b, name)(*args, **kwargs) for b in self]
                # If results are Batch-like, wrap in Batches
                if results and hasattr(results[0], 'keys') and callable(results[0].keys):
                    return Batches(results)
                return tuple(results)
            return method

        raise AttributeError(f"Batches has no attribute '{name}'")
