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
"""
Stack: the burst-stack InSAR processor.

Flattened from the former Stack_plot -> Stack_export -> Stack_ps -> Stack_stl ->
Stack_sbas -> Stack_detrend -> Stack_unwrap2d mixin chain, which added inheritance
levels without separating concerns.
"""
from __future__ import annotations
from .BatchCore import BatchCore
from . import utils_unwrap2d
from .utils_circular import wrap_phase
import numpy as np
import rioxarray
from contextlib import nullcontext
from . import utils_stl
from .Batch import Batch
from .BatchWrap import BatchWrap
from .BatchUnit import BatchUnit
from .BatchComplex import BatchComplex
from .Batches import Batches
from . import utils_io
from . import utils_xarray
from . import utils_baseline
from . import utils_align
from . import utils_median
from typing import TYPE_CHECKING
from .utils_dask import graph_check

if TYPE_CHECKING:
    import rasterio as rio
    import pandas as pd
    import xarray as xr

def _zarr_block(layout, *parts):
    """One block of Stack._zarr_dask(): cut from the disk chunks it overlaps (parts, row by row), joined.
    layout: (rows, columns, ((start, stop) in each row's chunk), ((start, stop) in each column's chunk), lead)."""
    ny, nx, ys, xs, lead = layout
    cut = [[parts[i * nx + j][ys[i][0]:ys[i][1], xs[j][0]:xs[j][1]] for j in range(nx)] for i in range(ny)]
    block = cut[0][0] if ny == 1 and nx == 1 else np.block(cut)
    return block[np.newaxis] if lead else block

def _on_data(name):
    """Stack.`name` of the registry (_ON_DATA): BatchComplex(self).`name` run on the stack's data and restacked (Stack._data_op)."""
    def method(self, *args, **kwargs):
        return self._data_op(name, args, kwargs)
    method.__name__ = name
    method.__qualname__ = f'Stack.{name}'
    method.__doc__ = getattr(BatchComplex, name).__doc__
    return method


class _StackView(BatchComplex):
    """The per-burst Datasets of a Stack, shown under the Stack's name by BatchCore.__repr__ (GKB U4). No method of the Stack is here."""


_StackView.__name__ = 'Stack'


# THE ATMOSPHERE OF A BURST (the user, 2026-10-20, RP/gfa_rulings.md; the redesign of 2026-11-08, GAM U4): the toolkit opens each
# atmosphere model once per stack, and opened.phase() gives its tables lazily on the model's own nodes (one chunk per date and
# tile). This module computes the phase with its own kernel (utils_atmosphere): burst_constants once per burst (the geometry and
# the shell flags), and at the graph build the one eager step (_chunk_ranges: the ele range of each chunk, with a progress bar) and
# block_tables once per spatial chunk (the height rows of its range, every HS_STEP of the kind, and the polynomials of the incidence or, for the
# ionosphere, the pierce point about the absolute 10 km centres its pixels need; held as data, one key per chunk), then per data
# block cut_block and grid_block, one numba pass per block, pixelwise. The tables of the stack's dates and tiles are computed
# once at the phase call and held (_held_group); each data block is one task that reads the held chunks of its tiles, its ele
# block, the chunk's tables and the burst's constants. The folder's kind comes from the tables' own variables; no model name is
# compared here.
DEFAULT_HEIGHTS = (-500.0, 9000.0)
_HELD = {}
_HELD_MAX = 8


def _atmosphere_open(folder, times, crs, lam, factor, kind, method, heights):
    """The atmosphere model of the folder, opened once for the stack: the toolkit's class of the folder and its open() with the
    union of the bursts' times, the output crs, the wavelength and heights (lo, hi): the union of the bursts' ele actual_range.
    Raises one ERROR line when the folder holds the other kind of tables (a 'tec' lattice is the ionosphere, a height axis the
    troposphere)."""
    from insardev_toolkit.utils_atmosphere import model_class
    opened = model_class(folder)().open(folder, list(times), crs, lam, 1.0 if factor is None else factor, heights=heights)
    is_iono = 'tec' in opened.terms.data_vars
    if is_iono != (kind == 'iono'):
        holds = 'the ionosphere' if is_iono else 'the troposphere'
        raise ValueError(f"ERROR: {method}() needs the {kind} model, the folder {folder} holds {holds}")
    return opened


def _atmosphere_reference(stack, baseline, ref, method):
    """Per burst, the index of the acquisition the atmosphere phase is referenced to, or None for the absolute phase
    (baseline=None and ref=None). ref given: its position in each burst's dates, parsed as predict() parses it
    (Batch._ref_index); a date a burst does not hold raises one ERROR line. ref=None: each burst's own date where its
    |baseline| is smallest (predict()'s rule per burst, the per-date baseline averaged over its other dims); a burst that
    lacks the baseline takes its first date (zeros, as predict() and trend2d()), no error. Nothing is computed: the
    per-date baseline is read from the bursts' own values."""
    if not baseline:
        baseline = None
    if baseline is None and ref is None:
        return None
    dates = {}
    for key, ds in stack.items():
        dates[key] = np.asarray(ds.coords['date'].values).astype('datetime64[D]')
    if ref is not None:
        out = {}
        for key, d in dates.items():
            try:
                out[key] = Batch._ref_index(ref, d)
            except (KeyError, IndexError):
                raise ValueError(f"ERROR: {method}(): ref={ref!r} is not an acquisition of burst {key} "
                                 f"({d[0]} .. {d[-1]}, {len(d)} dates)") from None
        return out
    out = {}
    for key, ds in stack.items():
        if baseline not in ds:
            out[key] = 0
            continue
        b = np.asarray(ds[baseline].values, dtype=np.float64)
        while b.ndim > 1:
            b = np.nanmean(b, axis=-1)
        if b.shape != dates[key].shape or not np.isfinite(b).any():
            raise ValueError(f"ERROR: {method}(): burst {key} has no finite per-date {baseline} to pick the reference date")
        out[key] = int(np.nanargmin(np.abs(b)))
    return out




def _data_chunks(ds):
    """The chunks (date, y, x) of the burst's gridded data, its complex variables first; None when the data is not chunked.
    The atmosphere phase takes them, so the phase and the data share one chunking (GLK U2)."""
    for v in sorted(ds.data_vars, key=lambda n: ds[n].dtype.kind != 'c'):
        a = ds[v]
        if set(a.dims) == {'date', 'y', 'x'} and a.chunks is not None:
            return tuple(a.chunks[a.dims.index(d)] for d in ('date', 'y', 'x'))
    return None


def _held_table(arrs, chunks):
    """The lazy tables of one model, computed once and held for the stack: arrs (dask arrays) with one rechunk spec each in
    chunks (one chunk per date and tile, the other dims whole). They are persisted in ONE call, so the reads they share run
    once; with a distributed client each is kept alive in every graph built from it (hold_persisted). Without one they are
    computed in ONE call on dask's synchronous scheduler into numpy (never the threaded scheduler). Returns the held arrays
    in order."""
    import dask
    import dask.array as da
    from .utils_dask import hold_persisted
    rechunked = [arr.rechunk(spec) for arr, spec in zip(arrs, chunks)]
    if _held_client() is None:
        values = dask.compute(*rechunked, scheduler='synchronous')
        return [da.from_array(v, chunks=arr.chunks) for v, arr in zip(values, rechunked)]
    return [hold_persisted(arr) for arr in dask.persist(*rechunked)]


def _held_group(opened, kind):
    """The held tables of one opened model: the model's lattice (lat, lon, degrees), the time labels, the tile first nodes, and
    the lazy tables computed and held (_held_table): troposphere hydro, wet, mapping (one chunk per date and tile) with the
    height rows and the height term; ionosphere tec (one chunk per date) with the shell R + H, a number."""
    import dask
    import pandas as pd
    T = opened.phase()
    lat = np.asarray(T['lat'].values, dtype=np.float64)
    lon = np.asarray(T['lon'].values, dtype=np.float64)
    held = {'times': pd.DatetimeIndex(opened.times), 'lat': lat, 'lon': lon}
    if kind == 'tropo':
        held['tiles'] = list(zip(np.asarray(T['tile y0'].values, dtype=int).tolist(),
                                 np.asarray(T['tile x0'].values, dtype=int).tolist()))
        held['ty'], held['tx'] = int(T.sizes['ty']), int(T.sizes['tx'])
        held['hydro'], held['wet'], held['mapping'] = _held_table(
            [T[v].data for v in ('hydro', 'wet', 'mapping')], [{0: 1, 1: 1}] * 3)
        held['h_first'] = float(np.asarray(T['height'].values, dtype=np.float64)[0])
        held['hcoef'] = np.asarray(T['height_term'].values, dtype=np.float64)
        held['shell'] = None
    else:
        held['tec'], shell = _held_table([T['tec'].data, T['shell'].data], [{0: 1, 1: -1, 2: -1}, {}])
        held['shell'] = float(dask.compute(shell, scheduler='synchronous' if _held_client() is None else None)[0])
        held['h_first'], held['hcoef'] = 0.0, np.zeros(4)
    return held


def _held_client():
    """The id of the distributed client that the default scheduler runs on, or None for any other scheduler (synchronous
    configured, or no client): the held tables belong to the client that persisted them, and a graph computed on another
    scheduler cannot resolve their futures."""
    import dask.base
    from distributed import Client
    client = getattr(dask.base.get_scheduler(), '__self__', None)
    return client.id if isinstance(client, Client) else None


def _held_alive(held):
    """True when every dask array of a held group has all its futures finished on the cluster; an array without futures
    (the synchronous path) is alive. False after client.restart() or a lost worker: the group is rebuilt."""
    import dask.array as da
    from distributed import futures_of
    for v in held.values():
        if isinstance(v, da.Array) and any(f.status != 'finished' for f in futures_of(v)):
            return False
    return True


def _held_for(folder, crs, lam, stamps, span, factor, kind, method):
    """The held tables of a group of bursts (one folder, crs and wavelength): reused by a second call with the same times,
    heights, factor and client while the held chunks are alive on the cluster (_HELD, the last _HELD_MAX groups); otherwise the
    model is opened once and its tables are held. The entries of another client are dropped."""
    client = _held_client()
    for k in [k for k in _HELD if k[-1] != client]:
        del _HELD[k]
    sig = (kind, folder, crs, lam, tuple(sorted(stamps)), tuple(span), factor, client)
    if sig in _HELD and not _held_alive(_HELD[sig]):
        del _HELD[sig]
    if sig not in _HELD:
        opened = _atmosphere_open(folder, tuple(sorted(stamps)), crs, lam, factor, kind, method, span)
        if len(_HELD) >= _HELD_MAX:
            _HELD.pop(next(iter(_HELD)))
        _HELD[sig] = _held_group(opened, kind)
    return _HELD[sig]


def _table_key(arr, i, t=None):
    """The dask key of one held chunk: the date index i and, for a table with tiles, the tile t; the other indices are 0."""
    idx = [i] if t is None else [i, t]
    return (arr.name,) + tuple(idx + [0] * (arr.ndim - len(idx)))


def _chunk_ranges(eles, method):
    """The ele range (lo, hi) of every spatial chunk of every burst, read now: the one eager step of the atmosphere phase (the
    tables of a chunk cover its own heights only). One WARNING names it, as align() does; the chunks are read in parallel on the
    cluster with a progress bar, without a Client on the synchronous scheduler (never the threaded one).

    Parameters
    ----------
    eles : dict - Per burst key: (the ele dask array, the data chunks (dates, y, x)).
    method : str - The calling method's name, for the message.

    Returns
    -------
    dict - Per burst key: the nested lists [iy][ix] of (lo, hi), metres.

    Examples
    --------
    >>> ranges = _chunk_ranges({key: (ele_array, ((1,), (ny,), (nx,)))}, 'troposphere_phase')
    """
    import dask
    import dask.array as da
    from .utils_atmosphere import ele_range
    jobs = []
    for key, (ele, dchunks) in eles.items():
        blocks = da.asarray(ele).rechunk((dchunks[1], dchunks[2])).blocks
        jobs += [((key, iy, ix), dask.delayed(ele_range)(blocks[iy, ix]))
                 for iy in range(len(dchunks[1])) for ix in range(len(dchunks[2]))]
    print(f'WARNING: {method}() is not lazy: it reads the height range of {len(jobs)} chunk(s) now, so the tables of each chunk '
          f'cover its own heights only.', flush=True)
    delayed = [d for _, d in jobs]
    if _held_client() is None:
        values = dask.compute(*delayed, scheduler='synchronous')
    else:
        from .utils_dask import progress_persisted
        persisted = dask.persist(*delayed)
        progress_persisted(list(persisted), desc='Heights of chunks'.ljust(25))
        values = dask.compute(*persisted)
    out = {key: [[None] * len(dchunks[2]) for _ in dchunks[1]] for key, (_, dchunks) in eles.items()}
    for ((key, iy, ix), _), v in zip(jobs, values):
        out[key][iy][ix] = v
    return out


def _atmosphere_phase(stack, kind, factor, method, geocentric=True, baseline=None, ref=None):
    """Batch of per burst Datasets with the variable 'phase' (date, y, x), float32 radians, one lazy dask array per burst,
    on the data's chunks. The folder of kind ('tropo' or 'iono') comes from the burst attributes set by load(). The tables
    are computed and held at this call (_held_for); each data block is one task of an explicit layer (phase_block): its ele
    block, the held chunks of its tiles for each date and for the reference, and the burst's geometry (Batch._stack_geometry).
    The reference is subtracted in the kernel. Bursts sharing a folder, crs and wavelength share one opened model."""
    import uuid
    import dask.array as da
    import xarray as xr
    import pandas as pd
    from dask._task_spec import DataNode, Task, TaskRef
    from dask.highlevelgraph import HighLevelGraph
    from .Batch import _ele_range, _stack_geometry, _warn_no_actual_range
    from .utils_atmosphere import block_tables, burst_constants, cut_block, phase_block, pixel_axis
    missing = [key for key, ds in stack.items() if kind not in ds.attrs]
    if missing:
        raise ValueError(f"ERROR: {method}() needs the {kind} folder of the bursts: call Stack().load(..., {kind}=folder) first")
    tf = stack.transform()
    import insardev.utils_geometry as ug
    for key in tf.keys():
        missing = [n for n in ug.SCALARS + ('BPR', 'ref_height') if n not in tf[key].variables]
        if missing:
            raise KeyError(f"ERROR: {method}(): burst {key} carries no {', '.join(missing)}.")
    bursts = []
    for key, ds in stack.items():
        if 'startTime' not in ds or 'radar_wavelength' not in ds:
            raise ValueError(f"ERROR: {method}() needs startTime and radar_wavelength on burst {key}")
        if 'ele' not in tf[key]:
            raise ValueError(f"ERROR: {method}() needs the heights (ele) of the transform on burst {key}")
        lam = np.unique(np.asarray(ds['radar_wavelength'].values, dtype=np.float64))
        if lam.size != 1:
            raise ValueError(f"ERROR: {method}() needs one radar_wavelength on burst {key}")
        times = tuple(pd.Timestamp(str(t)) for t in np.asarray(ds['startTime'].values))
        bursts.append((key, ds, str(ds.attrs[kind]), float(lam[0]), times, ds.rio.crs.to_string()))
    reference = _atmosphere_reference(stack, baseline, ref, method)
    # the heights of each opened model: the union of its bursts' ele actual_range (the transform's attribute, never the raster);
    # a burst without it counts the default range -500..9000 m, and ONE WARNING of this call names those bursts
    union = {}
    spans = {}
    nores = []
    for key, ds, folder, lam, times, crs in bursts:
        gkey = (folder, crs, lam)
        union.setdefault(gkey, set()).update(times)
        rng = _ele_range(tf[key])
        if rng is None:
            nores.append(str(key))
            rng = DEFAULT_HEIGHTS
        spans[gkey] = rng if gkey not in spans else (min(spans[gkey][0], rng[0]), max(spans[gkey][1], rng[1]))
    if nores:
        print(f"WARNING: {method}(): burst(s) {', '.join(nores)} carry no ele actual_range: the {kind} model of their folder "
              f"is opened for the default heights -500..9000 m.", flush=True)
    held = {}
    for gkey, stamps in union.items():
        folder, crs, lam = gkey
        held[gkey] = _held_for(folder, crs, lam, stamps, spans[gkey], factor, kind, method)
    uncut = []
    out = {}
    ranges = _chunk_ranges({key: (tf[key]['ele'].data, _data_chunks(ds) or ((ds.sizes['date'],), (ds.sizes['y'],), (ds.sizes['x'],)))
                            for key, ds, *_ in bursts}, method)
    for key, ds, folder, lam, times, crs in bursts:
        gkey = (folder, crs, lam)
        hg = held[gkey]
        geo = _stack_geometry(tf[key], key, method, 'Call it on the stack.', uncut)[0]
        rng = _ele_range(tf[key]) or DEFAULT_HEIGHTS
        dates = [int(i) for i in hg['times'].get_indexer(list(times))]
        if min(dates) < 0:
            raise ValueError(f"ERROR: {method}(): a date of burst {key} is not a time of its opened {kind} model")
        ref_i = None if reference is None else dates[reference[key]]
        dchunks = _data_chunks(ds) or ((ds.sizes['date'],), (ds.sizes['y'],), (ds.sizes['x'],))
        ele = da.asarray(tf[key]['ele'].data).rechunk((dchunks[1], dchunks[2]))
        uid = uuid.uuid4().hex
        name = f'atm-{kind}-{uid}'
        skey = (f'atm-static-{uid}', 0)
        # the burst's constants, once per burst: the geometry, the shell flags and the longitude of the centre of its whole output grid
        xs, ys = np.asarray(ds.x.values, dtype=np.float64), np.asarray(ds.y.values, dtype=np.float64)
        const = burst_constants(geo, crs, rng, shell=hg['shell'], geocentric=bool(geocentric),
                                centre=(0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())))
        static = dict(const=const, node_lat=hg['lat'], node_lon=hg['lon'], h_first=hg['h_first'], hcoef=hg['hcoef'])
        layer = {skey: DataNode(skey, static)}
        tables = [hg[v] for v in ('hydro', 'wet', 'mapping')] if kind == 'tropo' else [hg['tec']]
        tiles = hg.get('tiles', [])
        ty, tx = hg.get('ty', 0), hg.get('tx', 0)

        def refs_of(i, sel):
            """The held chunks of date index i for the tiles sel (troposphere: hydro, wet, mapping per tile)."""
            if kind == 'tropo':
                return [TaskRef(_table_key(hg[v], i, t)) for t in sel for v in ('hydro', 'wet', 'mapping')]
            return [TaskRef(_table_key(hg['tec'], i))]

        oy = np.cumsum((0,) + tuple(dchunks[1])).tolist()
        ox = np.cumsum((0,) + tuple(dchunks[2])).tolist()
        od = np.cumsum((0,) + tuple(dchunks[0])).tolist()
        # the burst's pixel axes: the transform's lattice as (first, step) per axis (a step of a one-pixel axis is 0.0); each block
        # computes its own coordinates from them (pixel_axis), so no coordinate array is held in a task or a key
        ystep = float(ys[1] - ys[0]) if ys.size > 1 else 0.0
        xstep = float(xs[1] - xs[0]) if xs.size > 1 else 0.0
        axes = (float(ys[0]), ystep, float(xs[0]), xstep)
        for jc in range(len(dchunks[0])):
            for iy in range(len(dchunks[1])):
                for ix in range(len(dchunks[2])):
                    ycoord = pixel_axis(axes[0], axes[1], oy[iy], oy[iy + 1])
                    xcoord = pixel_axis(axes[2], axes[3], ox[ix], ox[ix + 1])
                    span = (oy[iy], oy[iy + 1], ox[ix], ox[ix + 1])
                    sel = []
                    if kind == 'tropo':
                        # the box of the model's lattice only ({}: the burst's tables are not needed for it)
                        cut = cut_block(ycoord, xcoord, const, hg['lat'], hg['lon'], tables={})
                        sel = [t for t, (y0, x0) in enumerate(tiles)
                               if y0 < cut.i1 and y0 + ty > cut.i0 and x0 < cut.j1 and x0 + tx > cut.j0]
                    tiles_sel = tuple(tiles[t] for t in sel)
                    plan, refs = [], []
                    for dj in range(od[jc], od[jc + 1]):
                        plan.append((tiles_sel, ref_i is not None))
                        refs += refs_of(dates[dj], sel)
                        if ref_i is not None:
                            refs += refs_of(ref_i, sel)
                    # the chunk's polynomial tables: built here, at the graph build, for the chunk's own ele range (only what the
                    # kind needs), and held as DATA in one key per spatial chunk that the chunk's dates read; no task builds them
                    tkey = (f'atm-tables-{uid}', iy, ix)
                    if tkey not in layer:
                        layer[tkey] = DataNode(tkey, block_tables(ycoord, xcoord, const, ranges[key][iy][ix]))
                    bkey = (name, jc, iy, ix)
                    layer[bkey] = Task(bkey, phase_block, kind, TaskRef((ele.name, iy, ix)), axes, span,
                                       TaskRef(skey), TaskRef(tkey), plan, *refs)
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[ele] + tables)
        res = da.Array(graph, name, chunks=dchunks, dtype=np.float32, meta=np.empty((0, 0, 0), dtype=np.float32))
        phase = xr.DataArray(res, dims=('date', 'y', 'x'), coords={'date': ds.date, 'y': ds.y, 'x': ds.x},
                             attrs={'units': 'radians'})
        result_ds = xr.Dataset({'phase': phase})
        result_ds.attrs = dict(ds.attrs)
        out[key] = result_ds.rio.write_crs(ds.rio.crs)
    _warn_no_actual_range(method, uncut)
    result = Batch(out)
    graph_check(method, result, [ds for _, ds, *_ in bursts])
    return result


class Stack:
    """The per-burst SLC stack: the complex data (VV, ...), its 1-D radar metadata and
    the TRANSFORM, the geometry grids (azi, ele, rng, look_E/N/U) and the per-date
    geometry polynomials, which stack.transform() gives.

    A STACK IS NOT AN ARRAY (N81). It is a metaobject: the complex data, the transform
    and the attributes. Math operators, ordering comparisons and numpy functions refuse it
    (Python raises its TypeError); == and != compare identity. numpy and math on the data
    go through stack.complex(), which is BatchComplex(stack).

    THE TRANSFORM IS NOT DATA (N81). Every data operation of a Stack -- where(), mask(),
    the map helpers, the reductions, gaussian(), multilook(), goldstein(), fit1d(),
    fit1d_network(), trend2d(), predict(), ... -- runs on BatchComplex(stack), so the transform grids
    are never masked,
    weighted, filtered, smoothed or reduced as data were. A result on the stack's own
    grid that stays complex data comes back as a Stack with the transform untouched
    (stack.where(mask).transform() is the stack's transform, unmasked); any other
    result is what the operation gives: a mask is a plain Batch, angle() a BatchWrap,
    downsample() a BatchComplex on its new grid. A mask or a weight matches the data
    grids by name (a DataArray one applies to every data grid).

    Selection and structure -- sel(), isel(), crop(), chunk*(), compute(), assign(),
    load(), snapshot() -- take the transform along. The methods that need the
    geometry read it from the stack: trend2d() and predict() from stack.transform(),
    detrend() from the stack's own covariates, elevation() and displacement_*()
    from stack.transform().
    """

    _STRUCTURE_4CONN = utils_unwrap2d.STRUCTURE_4CONN

    # THE SELECTORS OF A STACK (GKB U3, the user 2026-10-30): these names run on both parts of a Stack, the data
    # (BatchComplex) and the transform (Batch); the results are a Stack again (_on_parts). Each part is given only the
    # names it holds (_restrict); the chunkers are not here but in _ON_MERGED. assign is explicit: each variable goes to the part
    # the split rule gives it (_part_keeps).
    PART_SELECTORS = ('assign_coords', 'crop', 'drop_isel', 'drop_sel', 'drop_vars',
                      'expand_dims', 'isel', 'reindex', 'reindex_like', 'rename', 'rename_vars', 'sel', 'set_index', 'transpose')

    # THE REGISTRY OF THE INHERITED CORE NAMES (GKB U4; the user, 2026-10-30: every selector, where, fit1d, fit1d_network, mix).
    # PART_SELECTORS run on both parts (_on_parts); _ON_DATA run on the data, BatchComplex(self), restacked (_data_op). Every other
    # inherited name is an explicit method in the class body (after the last method): a plain forward written out, or a wrapper.
    _ON_DATA = ('where', 'fit1d', 'fit1d_network', 'mix')

    # THE CHUNKERS RUN ONCE ON THE MERGED BURSTS (HEAD's Stack(BatchComplex); 'chunk*() ... take the transform along'): every grid
    # of a burst gets the chunks of one BatchCore call, which makes one graph check (_on_merged).
    _ON_MERGED = ('chunk', 'chunk1d', 'chunk2d')

    @staticmethod
    def _geometry_keeps(da) -> bool:
        """Whether Batch(stack) keeps a variable: a real grid (y, x), a 1-D one or a non-spatial one (Batch.__init__'s rule)."""
        return da.dtype.kind != 'c' and (tuple(da.dims) == ('y', 'x') or len(da.dims) == 1 or not ({'y', 'x'} & set(da.dims)))

    @staticmethod
    def _split(mapping):
        """(data, geometry) dicts of a mapping of per-burst Datasets. The geometry holds what Batch(stack) keeps; the data holds
        what BatchComplex(stack) keeps and every real variable that is neither (so each variable is in one part at least). The
        1-D and 0-D variables are in both, as the two views always had them."""
        data, geometry = {}, {}
        for key, ds in mapping.items():
            if not hasattr(ds, 'data_vars'):
                data[key] = geometry[key] = ds
                continue
            geo = [v for v in ds.data_vars if Stack._geometry_keeps(ds[v])]
            dat = [v for v in ds.data_vars if not Stack._geometry_keeps(ds[v]) or ds[v].ndim <= 1]
            data[key] = ds[dat]
            geometry[key] = ds[geo]
        return data, geometry

    @staticmethod
    def _names_of(part):
        """The variables, coordinates and dimensions the burst Datasets of a part hold."""
        names = set()
        for ds in dict.values(part):
            if hasattr(ds, 'variables'):
                names |= set(ds.variables) | set(ds.dims)
        return names

    @staticmethod
    def _part_of(value, which):
        """A Stack argument as its part of one side; any other argument as it is."""
        if isinstance(value, Stack):
            return value._data if which == 'data' else value._geometry
        return value

    @staticmethod
    def _restrict(name, args, kwargs, mine, other):
        """The name arguments of a selector restricted to the names this part holds. A name that neither part holds is kept, so
        the part raises the error it always raised."""
        def keep(n):
            return n in mine or n not in other
        if name == 'drop_vars' and args and isinstance(args[0], (str, list, tuple)):
            names = [args[0]] if isinstance(args[0], str) else list(args[0])
            args = [[n for n in names if keep(n)]] + list(args[1:])
        elif name in ('rename', 'isel', 'sel', 'drop_isel', 'drop_sel') and args and isinstance(args[0], dict):
            args = [{n: v for n, v in args[0].items() if keep(n)}] + list(args[1:])
        if name == 'transpose':
            args = [d for d in args if keep(d)]
        elif name in ('rename', 'rename_vars', 'isel', 'sel', 'drop_isel', 'drop_sel'):
            kwargs = {n: v for n, v in kwargs.items() if keep(n)}
        return args, kwargs

    @staticmethod
    def _on_parts(name):
        """Stack.`name` on both parts: the data (BatchComplex) and the transform (Batch) each run `name` with the arguments
        restricted to the names the part holds; the results are a Stack (GKB U3)."""
        def method(self, *args, **kwargs):
            names = {'data': Stack._names_of(self._data), 'geometry': Stack._names_of(self._geometry)}
            out = {}
            for which, part in (('data', self._data), ('geometry', self._geometry)):
                other = 'geometry' if which == 'data' else 'data'
                a = [Stack._part_of(x, which) for x in args]
                k = {key: Stack._part_of(v, which) for key, v in kwargs.items()}
                a, k = Stack._restrict(name, a, k, names[which], names[other])
                out[which] = getattr(part, name)(*a, **k)
            if not (isinstance(out['data'], BatchCore) and isinstance(out['geometry'], BatchCore)):
                return out['data']
            return Stack._of(out['data'], out['geometry'])
        method.__name__ = name
        method.__qualname__ = f'Stack.{name}'
        method.__doc__ = getattr(BatchComplex, name).__doc__
        return method

    @staticmethod
    def _on_merged(name):
        """Stack.`name` of _ON_MERGED: BatchCore.`name` runs once on the merged bursts (dict(self.items()) as one view), and the
        result is split into the parts of a Stack again."""
        def method(self, *args, **kwargs):
            return Stack(getattr(BatchCore._view_as(_StackView, dict(self.items())), name)(*args, **kwargs))
        method.__name__ = name
        method.__qualname__ = f'Stack.{name}'
        method.__doc__ = getattr(BatchComplex, name).__doc__
        return method

    @staticmethod
    def _part_keeps(value, which):
        """Whether assign() gives a value to a part: a Stack to both (each its own part); a batch by the grid of its first burst
        (BatchCore.assign takes one variable per burst); a DataArray by the split rule (_split): the transform keeps a real grid
        (y, x), the data the others, a 1-D or 0-D one to both; any other value (a Dataset, a number) to both."""
        if isinstance(value, BatchCore):
            grids = list(dict.values(value))
            value = grids[0] if grids else None
        if isinstance(value, Stack) or not (hasattr(value, 'dims') and hasattr(value, 'dtype')):
            return True
        geo = Stack._geometry_keeps(value)
        return geo if which == 'geometry' else (not geo or value.ndim <= 1)

    def assign(self, variables=None, **variables_kwargs):
        """The variables of the stack, as BatchComplex.assign (GKB U3): each goes to the part the split rule gives it (_part_keeps),
        so a new real grid is in the transform only and a new complex grid in the data only.

        >>> Stack().assign(X=1)
        """
        kwargs = dict(variables or {}, **variables_kwargs)
        out = {}
        for which, part in (('data', self._data), ('geometry', self._geometry)):
            mine = {k: Stack._part_of(v, which) for k, v in kwargs.items() if Stack._part_keeps(v, which)}
            out[which] = part.assign(**mine)
        if not (isinstance(out['data'], BatchCore) and isinstance(out['geometry'], BatchCore)):
            return out['data']
        return Stack._of(out['data'], out['geometry'])

    @staticmethod
    def _as_stack(out):
        """A BatchComplex result of a method run on _batch_view() is a Stack again (GKB U4); any other result is returned as it is."""
        return Stack(out) if isinstance(out, BatchComplex) else out

    def _batch_view(self):
        """The per-burst Datasets of the stack, merged (data and transform) as the dict Stack gave them, as a BatchComplex view.
        The base of the methods that read a whole burst: the trend3d() fields, compute(), the store, to_dataset(), map().
        No constructor runs (BatchCore._view_as)."""
        return BatchCore._view_as(BatchComplex, dict(self.items()))

    def _data_op(self, name, args, kwargs):
        """BatchComplex(self).`name`(*args, **kwargs) on the stack's data; a Stack argument is its data too. The result is
        restacked (_restack): a result on the stack's own grid is a Stack with the transform untouched."""
        data = BatchComplex(self)

        def arg(a):
            return data if a is self else (BatchComplex(a) if isinstance(a, Stack) else a)
        out = getattr(data, name)(*[arg(a) for a in args], **{k: arg(v) for k, v in kwargs.items()})
        return self._restack(data, out)

    _refuse = lambda self, *args, **kwargs: NotImplemented  # the refusal of a Stack (N81): numpy and Python raise their TypeError

    # A STACK IS NOT AN ARRAY (N81, GKB U5): the class defines no operator or comparison; a binary operator finds no method on
    # the Stack and its partner returns NotImplemented for it (BatchCore._is_operand). numpy refuses it by its own protocol:
    # __array_ufunc__ is None (a ufunc raises TypeError) and __array_function__ returns NotImplemented (a function raises
    # TypeError). Without them numpy reads the Stack as a sequence of its keys (np.add(stack, stack) returned an array) and
    # np.sum(stack) reaches stack.sum() (measured, GKB U5).
    __array_ufunc__ = None
    __array_function__ = _refuse
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = None

    def _restack(self, data, out):
        """The result of a data operation (`out`, computed on `data` =
        BatchComplex(self)) as the caller gets it (N81).

        A BatchComplex on the stack's own grid -- the same bursts, y and x --
        is a stack again: every variable BatchComplex(self) left out (the
        transform grids, the per-date geometry polynomials) rides along
        untouched, the same objects, where its dimensions still describe the
        result (a polynomial over dates does not ride once the dates are
        reduced). Anything else -- a mask, a real Batch, a BatchWrap, a model, a
        new grid -- is returned as the operation gave it.
        """
        import numpy as np
        import xarray as xr
        if type(out) is not BatchComplex or not out:
            return out
        res = {}
        for key, ods in dict.items(out):
            ds = self.get(key)
            dds = dict.get(data, key)
            if ds is None or dds is None or not isinstance(ods, xr.Dataset):
                return out

            def same(d):
                # the dimension as the stack has it: the same labels, or the same size unlabelled
                if d not in ods.dims or d not in ds.dims or (d in ds.indexes) != (d in ods.indexes):
                    return False
                if d in ds.indexes:
                    return np.array_equal(np.asarray(ds.indexes[d]), np.asarray(ods.indexes[d]))
                return ds.sizes[d] == ods.sizes[d]
            if not (same('y') and same('x')):
                return out
            ride = {v: ds[v] for v in ds.data_vars
                    if v not in dds.data_vars and v not in ods.variables
                    and all(same(d) if d in dds.dims else d not in ods.dims for d in ds[v].dims)}
            if ride:
                ods = ods.assign(ride)
                order = [v for v in ds.data_vars if v in ods.data_vars]
                ods = ods[order + [v for v in ods.data_vars if v not in order]]
            res[key] = ods
        return type(self)(res)

    def trend2d(self, *vars, transform=None, **kwargs):
        # the data is BatchComplex(self); the covariates named are read from stack.transform()
        data = BatchComplex(self)
        return self._restack(data, data.trend2d(
            *vars, transform=self.transform() if transform is None else transform, **kwargs))
    trend2d.__doc__ = BatchComplex.trend2d.__doc__

    def predict(self, model, baseline: 'str | None' = 'BPR', ref=None, vars=None) -> 'Batch':
        BatchCore._require_model(model, 'predict')
        # an insardev_polsar trend3d() FIELD is evaluated by its predict3d() at the stack's pixels and elevation,
        # as stack.predict3d(field) does: `ele` is on the stack, BatchComplex(self) leaves it out
        if BatchComplex._trend3d_field(model, list(model.keys())):
            return self._as_stack(BatchComplex.predict(self._batch_view(), model, baseline=baseline, ref=ref, vars=vars))
        # the data is BatchComplex(self); a trend2d() model's covariates come from stack.transform()
        data = BatchComplex(self)
        if vars is None and all('trend2d_vars' in model[k].attrs for k in model):
            vars = self.transform()
        # NOTE: prints through BatchComplex.predict
        return self._restack(data, data.predict(model, baseline=baseline, ref=ref, vars=vars))
    predict.__doc__ = BatchComplex.predict.__doc__

    def _trend2d_covariates(self, trend, vars):
        """The covariates of a trend2d() model when none are given (GKB U4): the stack's transform, as trend2d() and predict()
        read them, when every burst of the stack holds the model's trend2d_vars. Any other argument is returned as it is."""
        if vars is None and self.keys() and all(k in trend and 'trend2d_vars' in trend[k].attrs for k in self.keys()):
            return self.transform()
        return vars

    def detrend(self, trend, vars=None):
        # a trend3d() FIELD is removed by the module's detrend3d(), which reads the stack's elevation: BatchComplex(self) leaves it out
        if BatchComplex._trend3d_field(trend, self.keys()):
            return self._as_stack(BatchComplex.detrend(self._batch_view(), trend, vars=vars))
        # the data is BatchComplex(self); a trend2d() model's covariates come from stack.transform()
        data = BatchComplex(self)
        return self._restack(data, data.detrend(trend, vars=self._trend2d_covariates(trend, vars)))
    detrend.__doc__ = BatchComplex.detrend.__doc__

    def detrend2d(self, trend, vars=None):
        # as detrend(): a trend3d() field goes through the merged view, a trend2d() model reads the stack's transform
        if BatchComplex._trend3d_field(trend, self.keys()):
            return self._as_stack(BatchComplex.detrend2d(self._batch_view(), trend, vars=vars))
        data = BatchComplex(self)
        return self._restack(data, data.detrend2d(trend, vars=self._trend2d_covariates(trend, vars)))
    detrend2d.__doc__ = BatchComplex.detrend2d.__doc__

    @staticmethod
    def _carry_meta(src_ds, out_vars):
        """The 1-D metadata rides along with the grids it describes.

        Every step of the pipeline carries the radar geometry it does not
        itself use, because a later one does: `radar_wavelength`, `near_range`,
        `rng_samp_rate`, `SC_height_*`, `earth_radius`, `num_lines`,
        `num_rng_bins`. Rebuilding a Dataset from the computed grids alone
        drops them, and nothing notices until a fit asks -- `fit1d()` raising
        `KeyError: near_range` several cells after the step that lost it.

        Computed variables win on a name collision; the coordinates are already
        carried by the DataArrays themselves.
        """
        return {**{v: src_ds[v] for v in src_ds.data_vars
                   if v not in out_vars and src_ds[v].ndim <= 1},
                **out_vars}

    def unwrap2d_link(self, phase, conncomp_size=10_000, conncomp_gap=None,
                      conncomp_linksize=5, conncomp_linkcount=30, debug=False):
        """
        Link disconnected components in already unwrapped phase.

        This function applies component linking to already unwrapped phase data
        by finding optimal 2π offsets between disconnected components.
        Use this to correct phase jumps between components after unwrapping.

        Parameters
        ----------
        phase : Batch
            Batch of already unwrapped phase datasets (output from unwrap2d_irls).
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be linked.
            Components smaller than this are set to NaN. Default is 10,000.
        conncomp_gap : int or None, optional
            Maximum pixel distance between components to consider them connectable.
            If None (default), no distance limit - all components can connect.
        conncomp_linksize : int, optional
            Number of pixels to use on each side of a connection point for
            estimating the phase offset. Default is 5.
        conncomp_linkcount : int, optional
            Maximum number of nearest neighbor components to consider.
            Default is 30.
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.

        Examples
        --------
        Link components in already unwrapped phase:

        >>> # First unwrap without linking
        >>> unwrapped = phase.unwrap2d_irls(weight=corr)
        >>>
        >>> # Then link components separately
        >>> linked = stack.unwrap2d_link(unwrapped, conncomp_size=10_000, debug=True)

        Notes
        -----
        The linking algorithm:
        1. Identifies connected components in the valid phase data
        2. Filters out components smaller than conncomp_size (set to NaN)
        3. Finds connection points between components using STRtree spatial indexing
        4. Estimates phase offsets at connection points
        5. Uses Integer Linear Programming (ILP) to find globally optimal 2π offsets
        6. Applies offsets to align all components with the reference (largest) component
        """
        return Batch(phase).unwrap2d_link(conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
                                          conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
                                          debug=debug)

    def unwrap2d(self, phase, weight=None, conncomp=False,
                conncomp_size=1_000, conncomp_gap=None,
                conncomp_linksize=5, conncomp_linkcount=30, union=False,
                debug=False, **kwargs):
        """
        Unwrap phase: L¹ IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        .. deprecated::
            Use ``phase.unwrap2d(weight=corr)`` on BatchWrap instead.
            This Stack method will be removed in a future version.

        See unwrap2d_irls() for the solver and its parameters.

        When conncomp=False (default), disconnected components are automatically
        linked using ILP optimization to find optimal 2π offsets.

        When conncomp=True, components are kept separate and returned with
        size-ordered labels (1=largest, 2=second largest, etc.).

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. Higher values indicate
            more reliable phase measurements.
        conncomp : bool, optional
            If False (default), link disconnected components using ILP to find
            optimal 2π offsets, returning a single merged result.
            If True, keep components separate and return conncomp labels
            (1=largest component, 2=second largest, etc., 0=invalid).
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 1000.
        conncomp_gap : int or None, optional
            Maximum pixel distance between components to consider them connectable.
            If None (default), no distance limit - all direct connections are used.
            Only used when conncomp=False.
        conncomp_linksize : int, optional
            Number of pixels to use on each side of a connection point for
            estimating the phase offset between components. Uses median for
            robustness - 5 pixels is sufficient to tolerate 2 outliers (40%).
            Default is 5. Only used when conncomp=False.
        conncomp_linkcount : int, optional
            Maximum number of nearest neighbor components to consider for
            connections from each component. Higher values find more potential
            connections but increase computation. Default is 30.
            Only used when conncomp=False.
        union : bool, optional
            False (default) solves each burst separately -- the form that
            scales, at the cost of each carrying its own integer solution, so
            bursts need not agree across a shared edge. True merges and solves
            once, consistent across them, while the merged scene fits. A Batch
            either way, each burst back with only its own pixels.
        debug : bool, optional
            If True, print diagnostic information. Default is False.
        **kwargs
            Passed to unwrap2d_irls(): accuracy (default 2e-3; 5e-4 near the optimum,
            about 1.9 times longer), max_iter (the safety cap, default 100), epsilon.
            The former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase (components linked).
            If conncomp is True: tuple of (Batch unwrapped phase, Batch conncomp)
            where conncomp labels are ordered by size (1=largest).

        Notes
        -----
        Component Linking (when conncomp=False):
        1. Unwraps each connected component separately
        2. Finds direct connections between components (not crossing others)
        3. Estimates phase offsets using conncomp_linksize pixels per connection
        4. Uses ILP to find globally optimal integer 2π offsets

        Examples
        --------
        Unwrap phase with component linking (default):
        >>> unwrapped = stack.unwrap2d(intfs, corr)

        Unwrap without weighting:
        >>> unwrapped = stack.unwrap2d(intfs)

        Keep components separate (no linking), get labels:
        >>> unwrapped, conncomp = stack.unwrap2d(intfs, corr, conncomp=True)
        >>> main_component = unwrapped.where(conncomp == 1)  # largest component

        Closer to the optimum, about 1.9 times longer:
        >>> unwrapped = stack.unwrap2d(intfs, corr, accuracy=5e-4)
        """
        # Validate parameters
        if not conncomp and conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )
        if union:
            return BatchWrap._unwrap2d_union(
                phase, weight, conncomp=conncomp, conncomp_size=conncomp_size,
                conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                conncomp_linkcount=conncomp_linkcount,
                debug=debug, **kwargs)
        # the refusal of unwrap2d_irls(), which this path went through before the move
        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, union=False,
                              debug=debug, **kwargs)


    def unwrap2d_irls(self, phase, weight=None, accuracy=2e-3, max_iter=100, epsilon=1e-2,
                      conncomp_size=30, debug=False, **kwargs):
        """
        Unwrap phase: L¹ IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        Each pair of each burst is one raster and one task with unwrap_threads(rasters) numba threads: one
        raster takes every core, 8 take 2 each on 16 cores, 16 and more take one each. The result depends
        neither on the threads nor on the chunks.
        Disconnected components (separated by NaN regions) are unwrapped independently and aligned to the
        input's circular mean. See `utils_unwrap2d.irls_unwrap_2d` for the algorithm.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. Higher values indicate
            more reliable phase measurements.
        accuracy : float, optional
            The stop: the relative decrease of the L1 objective over one step. Default 2e-3, which
            reaches the former strong set's result in less time than the former relaxed set took;
            5e-4 is near the optimum and takes about 1.9 times longer.
        max_iter : int, optional
            The safety cap on the steps; reaching it before `accuracy` prints one WARNING. Default 100.
        epsilon : float, optional
            Smoothing of the L1 norm at the end of its continuation. Default 1e-2.
        conncomp_size : int, optional
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default is 30.
        debug : bool, optional
            If True, print diagnostic information. Default is False.
        **kwargs
            The former tol, cg_max_iter, cg_tol, device, semaphore: accepted and ignored
            with one WARNING.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, Batch):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: Batch of component labels (uint16, 0=invalid, 1=largest, 2=second, ...);
              labels, not units (a BatchUnit holds float units only, N81)

        Examples
        --------
        >>> unwrapped, conncomp = stack.unwrap2d_irls(intfs, corr)
        >>> unwrapped, conncomp = stack.unwrap2d_irls(intfs, corr, accuracy=5e-4)
        """
        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        return phase.unwrap2d_irls(weight, accuracy=accuracy, max_iter=max_iter, epsilon=epsilon,
                                   conncomp_size=conncomp_size, debug=debug, **kwargs)

    def unwrap2d_dataset_mask(self, phase, method='hough_focal', grad_threshold=2.0,
                               mask_width=3, debug=False):
        """
        Experimental (not tested): discontinuity mask of one phase Dataset, see detect_discontinuity_hough_focal.

        Detect discontinuities in wrapped phase and return a mask for unwrapping.

        This function identifies phase discontinuities (e.g., fault lines) that
        should be masked before phase unwrapping to prevent smoothing artifacts.

        Parameters
        ----------
        phase : xr.Dataset
            Wrapped phase dataset.
        method : str
            Detection method. Currently supported:
            - 'hough_focal': Hough transform with focal point (tip) detection.
              Best for linear discontinuities with a clear endpoint.
        grad_threshold : float
            Gradient magnitude threshold for edge detection (radians).
            Default 2.0 rad ≈ 0.64π. Lower values detect more edges.
        mask_width : int
            Half-width of the mask along detected discontinuities. Default 3.
        debug : bool
            Print diagnostic information. Default False.

        Returns
        -------
        mask : xr.Dataset
            Boolean mask dataset with same coordinates as input.
            True = discontinuity pixels that should be masked as NaN before unwrapping.

        Examples
        --------
        Detect and mask discontinuities before unwrapping:

        >>> # Get wrapped phase
        >>> intf_ds = intf.align().dissolve().compute().to_dataset()
        >>> corr_ds = corr.dissolve().compute().to_dataset()
        >>>
        >>> # Detect discontinuities
        >>> mask = stack.unwrap2d_dataset_mask(intf_ds, method='hough_focal')
        >>>
        >>> # Apply mask to phase
        >>> intf_masked = intf_ds.where(~mask)
        >>> corr_masked = corr_ds.where(~mask)
        >>>
        >>> # Unwrap
        >>> unwrapped = intf_masked.unwrap2d(corr_masked, union=True)

        Notes
        -----
        The 'hough_focal' method works by:
        1. Computing wrapped phase gradients
        2. Finding high-gradient pixels (potential discontinuities)
        3. Using Hough transform to detect dominant line direction
        4. Finding the "tip" where gradient magnitude drops
        5. Creating a mask from the tip along the fault direction

        This preserves the discontinuity while allowing smooth regions to
        unwrap correctly around the fault tip.
        """
        import xarray as xr

        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")

        # Get data variable name (first one)
        var_names = list(phase.data_vars)
        if not var_names:
            raise ValueError("phase dataset has no data variables")
        var_name = var_names[0]

        # Get phase data as numpy array
        phase_data = phase[var_name].values

        if phase_data.ndim != 2:
            raise ValueError(f"Expected 2D phase data, got {phase_data.ndim}D")

        # Detect discontinuities
        if method == 'hough_focal':
            mask_data, info = utils_unwrap2d.detect_discontinuity_hough_focal(
                phase_data,
                grad_threshold=grad_threshold,
                mask_width=mask_width,
                debug=debug
            )
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'hough_focal'")

        if debug:
            if info['focal_point'] is not None:
                print(f"Discontinuity detected:")
                print(f"  Focal point: {info['focal_point']}")
                print(f"  Angle: {np.rad2deg(info['angle']):.1f}°")
                print(f"  Masked pixels: {info['n_masked']}")
            else:
                print("No discontinuity detected")

        # Create output dataset with same structure
        mask_da = xr.DataArray(
            mask_data,
            dims=phase[var_name].dims,
            coords=phase[var_name].coords,
            name='mask'
        )

        return mask_da.to_dataset(name=var_name)

    def stl(self, data, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL) on Batch data.

        Decomposes time series into trend, seasonal, and residual components.
        The input Batch must have a 'date' dimension.

        ONE VARIABLE: a Batch of DataArrays, x['VV'], or of Datasets with one
        (y, x) variable, x[['VV']]. The outputs are named by component alone,
        so two variables would collide: several raise.

        Parameters
        ----------
        data : Batch
            Input Batch with 'date' dimension containing time series data.
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
        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> displacement = model.displacement_los(stack.transform())
        >>> stl_result = stack.stl(displacement['VV'], freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, residual components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        import xarray as xr
        from .BatchCore import BatchCore

        # Validate input
        if not isinstance(data, (dict, Stack)):
            raise TypeError(f"data must be a Batch, got {type(data).__name__}")

        # Validate lazy data
        BatchCore._require_lazy(data, 'stl')

        sample_ds = next(iter(data.values()))
        if 'date' not in sample_ds.dims:
            raise ValueError("Input Batch must have 'date' dimension for STL decomposition")

        # ONE (y, x) VARIABLE per burst, a DataArray or a Dataset's one grid (N81): the
        # outputs are named by component alone, so two variables would collide
        results = {}
        for key, ds in data.items():
            grids = BatchCore._grids_of(ds)
            if len(grids) != 1:
                if isinstance(ds, xr.DataArray):
                    raise TypeError(f"ERROR: stl(): x['{ds.name}'] has no (y, x) grid, dims {tuple(ds.dims)}.")
                if not grids:
                    BatchCore._grid_vars(ds)
                names = list(grids)
                raise ValueError(f"ERROR: stl() takes one variable; the batch has {', '.join(map(str, names))}. "
                                 f"Select one: x[['{names[0]}']] or x['{names[0]}'].")
            da = next(iter(grids.values()))
            # Apply STL decomposition: 'trend', 'seasonal', 'residual'
            stl_ds = self._stl(da, freq=freq, periods=periods, robust=robust)

            # THREE NAMED OUTPUTS, a Dataset whatever the input form; a DataArray's
            # own attributes describe the input, not the result (N81)
            result_ds = xr.Dataset({var: stl_ds[var] for var in ('trend', 'seasonal', 'residual')})
            if isinstance(ds, xr.Dataset):
                result_ds.attrs = ds.attrs
            # Preserve CRS if available
            if hasattr(ds, 'rio') and ds.rio.crs is not None:
                import rioxarray
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

        graph_check('stl', results, data)
        return Batch(results)

    def _stl(self, data, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL) on the input time series data in parallel.

        The function performs the following steps:
        1. Convert the 'date' coordinate to valid dates.
        2. Unify date intervals to a specified frequency (e.g., weekly) for a mix of time intervals.
        3. Apply the Stack.stl1d function in parallel using Dask.
        4. Rename the output date dimension to match the original irregular date dimension.
        5. Return the STL decomposition results as an xarray Dataset.

        Parameters
        ----------
        self : Stack
            Instance of the Stack class.
        data : xarray.DataArray
            Input time series data as an xarray DataArray.
        freq : str, optional
            Frequency string for unifying date intervals (default is 'W' for weekly).
        periods : int, optional
            Number of periods for seasonal decomposition (default is 52).
        robust : bool, optional
            Whether to use a slower robust fitting procedure for the STL decomposition (default is False).

        Returns
        -------
        xarray.Dataset or None
            An xarray Dataset containing the trend, seasonal, and residual components of the decomposed time series,
            or None if the results are saved to a file.

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
        """
        import xarray as xr
        import numpy as np
        import pandas as pd
        import dask
        # disable "distributed.utils_perf - WARNING - full garbage collections ..."
        try:
            from dask.distributed import utils_perf
            utils_perf.disable_gc_diagnosis()
        except ImportError:
            from distributed.gc import disable_gc_diagnosis
            disable_gc_diagnosis()

        assert data.dims[0] == 'date', 'The first data dimension should be date'

        if not isinstance(data, xr.DataArray):
            raise Exception('Invalid input: The "data" parameter should be of type xarray.DataArray.')

        dt, dt_periodic = utils_stl.stl_periodic(data.date, freq)
        n_dates_out = len(dt_periodic)
        n_dates_in = data.date.size

        data_dask = data.data

        def _stl_block(data_block, _dt=dt, _dt_periodic=dt_periodic, _periods=periods, _robust=robust):
            from .utils_dask import get_dask_chunk_size_mb
            budget_bytes = int(get_dask_chunk_size_mb() * 1024 * 1024)
            return utils_stl.stl_block(data_block, _dt, _dt_periodic, _periods, _robust, budget_bytes=budget_bytes)

        models = dask.array.blockwise(
            _stl_block, 'cdyx',
            data_dask, 'pyx',
            new_axes={'c': 3, 'd': n_dates_out},
            concatenate=True,
            dtype=np.float32,
            meta=np.empty((0, 0, 0, 0), dtype=np.float32),
        )

        coords = {'date': dt_periodic.astype('datetime64[ns]'), 'y': data.y, 'x': data.x}

        # transform to separate variables; 'residual', the word align() uses
        varnames = ['trend', 'seasonal', 'residual']
        keys_vars = {}
        for varidx, varname in enumerate(varnames):
            var_data = models[varidx]
            keys_vars[varname] = xr.DataArray(var_data, coords=coords)
        model = xr.Dataset({**keys_vars})
        del models

        return model

    def psfunction(self, debug=False):
        """
        Compute PS (Persistent Scatterer) function for weighting in single-look processing.

        The PS function identifies stable scatterers by computing the ratio of mean
        amplitude to amplitude standard deviation across the temporal stack:

            psfunction = mean_intensity / (2 * std_intensity)

        Higher values indicate more stable scatterers (consistent backscatter).

        The result depends on the processing chunk size, because each chunk is the spatial context of the
        PS detection; tune it with load().chunk2d(...).

        Parameters
        ----------
        debug : bool
            Print debug information.

        Returns
        -------
        BatchUnit
            Per-burst PS function values for use as weight in pairs(...).interferogram(..., weight=psf) (lazy).

        Examples
        --------
        # Use PS function as weight for single-look interferograms
        psf = stack.psfunction()
        sintf, scorr = stack.pairs(pairs).interferogram(30, weight=psf)
        """
        import dask.array
        import numpy as np
        import xarray as xr
        from .BatchUnit import BatchUnit
        from .utils_psfunction import psfunction_dyx

        print('NOTE: psfunction() takes each processing chunk as its spatial context: tune the chunk size with load().chunk2d(...).',
              flush=True)

        results = {}
        for key, ds in self.items():
            # Get complex SLC data variable (usually 'VV' or 'VH')
            complex_vars = [v for v in ds.data_vars if ds[v].dtype.kind == 'c']
            if not complex_vars:
                raise ValueError(f"No complex data found in burst {key}")

            # Use first complex variable
            var_name = complex_vars[0]
            slc_data = ds[var_name]

            # Ensure data is chunked for lazy processing (chunk in y,x, not date)
            if not isinstance(slc_data.data, dask.array.Array):
                slc_data = slc_data.chunk({'y': 512, 'x': 512})

            if debug:
                print(f'DEBUG: psfunction for {key}: shape={slc_data.shape}, chunks={slc_data.chunks}')

            # ONE TASK PER OUTPUT CHUNK (the per-output-chunk rule; the user 2026-11-04): the date chunks of a block reach one
            # task as they are; the dates move last inside it
            slc_dyx = slc_data.transpose('date', 'y', 'x')
            psf = dask.array.blockwise(psfunction_dyx, 'yx', slc_dyx.data, 'dyx', concatenate=False,
                                       dtype=np.float32, meta=np.empty((0, 0), dtype=np.float32))
            psf_da = xr.DataArray(psf, dims=('y', 'x'), coords=slc_dyx.isel(date=0, drop=True).coords, name=var_name)

            results[key] = xr.Dataset({var_name: psf_da})

        graph_check('psfunction', results, self._data)
        return BatchUnit(results)

    import xarray as xr

    import numpy as np

    import pandas as pd

    import matplotlib

    def plot(self, cmap='turbo', alpha=1, ax=None, figsize=None):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        df = self.to_dataframe().reset_index()
        df['date'] = df['startTime'].dt.date

        # Create group key for orbit direction + path number
        df['orbit_path'] = df.apply(
            lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} [{rec['pathNumber']}]", axis=1
        )

        # Get unique orbit/path combinations and assign colors
        unique_orbit_paths = sorted(df['orbit_path'].unique())
        n = len(unique_orbit_paths)
        colormap = matplotlib.colormaps.get_cmap(cmap).resampled(n)
        color_map = {op: colormap(i) for i, op in enumerate(unique_orbit_paths)}

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot each burst with color based on orbit/path
        for orbit_path, group in df.groupby('orbit_path'):
            group.plot(ax=ax, edgecolor=color_map[orbit_path], facecolor='none', lw=0.25, label=orbit_path)

        # Create consolidated legend labels with date ranges
        legend_labels = []
        for orbit_path in unique_orbit_paths:
            group = df[df['orbit_path'] == orbit_path]
            dates = sorted(group['date'].unique())
            if len(dates) == 1:
                label = f"{orbit_path.split()[0]} {dates[0]} {orbit_path.split()[1]}"
            else:
                label = f"{orbit_path.split()[0]} {dates[0]} - {dates[-1]} {orbit_path.split()[1]}"
            legend_labels.append((orbit_path, label))

        handles = [matplotlib.lines.Line2D([0], [0], color=color_map[op], lw=1, label=label)
                   for op, label in legend_labels]
        ax.legend(handles=handles, loc='upper right')

        col = df.columns[0]
        for _, row in df.drop_duplicates(subset=[col]).iterrows():
            # compute centroid
            x, y = row.geometry.centroid.coords[0]
            ax.annotate(
                str(row[col]),
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                color=color_map[row['orbit_path']],
                path_effects=[patheffects.withStroke(linewidth=0.25, foreground='black')],
                alpha=1
            )

        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('easting [m]')
        ax.set_ylabel('northing [m]')

    def __init__(self, mapping:dict[str, xr.Dataset] | None = None):
        """The stack over a mapping of per-burst Datasets (or another Stack): split into its data (BatchComplex) and its
        transform (Batch), each burst's variables in the part they belong to (_split)."""
        data, geometry = Stack._split(mapping or {})
        self._data = BatchComplex(data)
        self._geometry = Batch(geometry)

    @classmethod
    def _of(cls, data, geometry):
        """A Stack over its two parts as they are (the results of the selectors; no split)."""
        out = cls.__new__(cls)
        out._data = data
        out._geometry = geometry
        return out

    def _merged(self, key):
        """The Dataset of one burst: its data part, with the transform variables the data part lacks (the 1-D ones are in both)."""
        data = dict.__getitem__(self._data, key)
        geo = dict.get(self._geometry, key)
        if not hasattr(data, 'data_vars') or not hasattr(geo, 'data_vars'):
            return data
        extra = {v: geo[v] for v in geo.data_vars if v not in data.variables}
        return data.assign(extra) if extra else data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key):
        return key in self._data

    def __setitem__(self, key, value):
        data, geometry = Stack._split({key: value})
        self._data[key] = data[key]
        self._geometry[key] = geometry[key]

    def __delitem__(self, key):
        del self._data[key]
        del self._geometry[key]

    def __repr__(self):
        return BatchCore.__repr__(BatchCore._view_as(_StackView, dict(self.items())))

    def keys(self):
        return self._data.keys()

    def values(self):
        return [self._merged(k) for k in self._data.keys()]

    def items(self):
        return [(k, self._merged(k)) for k in self._data.keys()]

    def get(self, key, default=None):
        return self._merged(key) if key in self._data else default

    def pop(self, key, *default):
        if key in self._data:
            out = self._merged(key)
            del self[key]
            return out
        if default:
            return default[0]
        raise KeyError(key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @staticmethod
    def _batch_type_for_subset(subset: dict) -> type | None:
        """Determine appropriate Batch type based on dtypes and dims in subset.

        Returns
        -------
        type or None
            BatchComplex if all variables are complex with date/pair dimension.
            Batch if all variables are spatial-only (no date/pair dimension).
            None if variables have date/pair dimension but are not complex (should be Stack).
        """
        import numpy as np
        import xarray as xr
        dtypes = set()
        has_temporal_dim = False
        for ds in subset.values():
            # a DataArray is its own one variable
            for da_ in ([ds] if isinstance(ds, xr.DataArray) else [ds[v] for v in ds.data_vars]):
                dtypes.add(da_.dtype)
                # Check if variable has temporal dimension (date or pair)
                if 'date' in da_.dims or 'pair' in da_.dims:
                    has_temporal_dim = True
        # If all variables are complex with temporal dim, use BatchComplex
        if dtypes and all(np.issubdtype(dt, np.complexfloating) for dt in dtypes) and has_temporal_dim:
            return BatchComplex
        # If no temporal dimension, return Batch (spatial-only variables)
        if not has_temporal_dim:
            return Batch
        # Has temporal dimension but not complex - should be Stack
        return None

    @property
    def wavelength(self):
        """Radar wavelength in meters (scalar, constant across all bursts)."""
        if not self:
            return None
        ds = next(iter(self.values()))
        if 'radar_wavelength' in ds:
            val = ds['radar_wavelength']
            return float(val.values.flat[0]) if val.ndim >= 1 else float(val.item())
        if 'radar_wavelength' in ds.attrs:
            return float(ds.attrs['radar_wavelength'])
        return None

    @property
    def coords(self):
        """Return coordinates from the first dataset in the stack.
        
        All datasets in a Stack share the same coordinate structure,
        so we expose the first one's coords for convenience.
        """
        if not self:
            return None
        first_ds = next(iter(self.values()))
        return first_ds.coords

    def PRM(self, keys: str | list[str] | None = None) -> dict:
        """Return platform parameters per burst.

        Parameters
        ----------
        keys : str | list[str] | None
            Parameter name(s) to extract. If ``None``, all scalar attrs
            and 0-D data_vars are returned per burst.

        Returns
        -------
        dict
            Mapping of burst key -> param dict (or burst key -> single value
            when a single key is requested).
        """
        if not self:
            return {}

        # normalize key selection
        select_all = keys is None
        if isinstance(keys, str):
            keys_list = [keys]
        else:
            keys_list = keys if keys is not None else None

        result: dict[str, object] = {}
        for burst, ds in self.items():
            params: dict[str, object] = dict(getattr(ds, 'attrs', {}))
            for name, var in getattr(ds, 'data_vars', {}).items():
                if var.ndim == 0:
                    try:
                        params.setdefault(name, var.item())
                    except Exception:
                        params.setdefault(name, var.values)

            if not select_all:
                if keys_list is None:
                    params = {}
                else:
                    params = {k: params.get(k) for k in keys_list}
                    if isinstance(keys, str):
                        params = params.get(keys)

            result[burst] = params

        return result

    def __getitem__(self, key):
        """A burst, or variables: a list keeps its 1-D variables and the attributes; a gridded name keeps the attributes, a 1-D name is that variable alone."""
        # Handle variable selection like sbas[['ele']] or sbas[['VV', 'VH']]
        if isinstance(key, list):
            # a list with no gridded name raises TypeError (a 1-D variable alone is stack['BPR'])
            datasets = list(self.values())
            if datasets and not any(n in ds.data_vars and BatchCore._is_grid(ds[n]) for ds in datasets for n in key):
                if not key:
                    raise TypeError("ERROR: stack[[]] names no variable. Select a gridded one, e.g. stack[['VV']].")
                for n in key:
                    # a name no burst carries is missing; a 1-D variable or coordinate is named in the message
                    if not any(n in ds.variables for ds in datasets):
                        raise KeyError(n)
                raise TypeError(f"ERROR: stack[{key!r}] names no gridded variable (1-D variables and scalars only): "
                                f"{', '.join(map(str, key))}. Select one 1-D variable alone, e.g. stack['BPR'] (its DataArray).")
            subset = {}
            for k, ds in self.items():
                # a single name leaves out the bursts that do not carry it, as before
                if len(key) == 1 and key[0] not in ds.data_vars:
                    continue
                missing = [n for n in key if n not in ds.variables]
                if missing:
                    raise KeyError(missing[0])
                # the other spatial variables (with y or x) go; the 1-D variables and the attributes stay
                subset[k] = ds.drop_vars([v for v in ds.data_vars if v not in key and {'y', 'x'} & set(ds[v].dims)])
            if datasets and not subset:
                raise KeyError(key[0])
            # the class is the listed gridded variables' own: the 1-D variables beside them do not count
            batch_type = self._batch_type_for_subset(
                {k: ds[[v for v in ds.data_vars if BatchCore._is_grid(ds[v])]] for k, ds in subset.items()})
            # Return appropriate Batch type if spatial-only or complex, else Stack
            if batch_type is not None:
                return batch_type(subset)
            # Has temporal dimension but not complex - return Stack
            return type(self)(subset)
        if isinstance(key, str) and key not in self:
            # not a burst: the variable or coordinate of that name (N81)
            return self._select_var(key)
        return self._merged(key)

    def _select_var(self, key):
        """The variable or coordinate `key` of each burst (N81); a gridded one keeps the burst's attributes."""
        subset = {k: ds[key] for k, ds in self.items() if key in ds.data_vars or key in ds.coords}
        if not subset:
            raise KeyError(key)
        if not all(BatchCore._is_grid(v) for v in subset.values()):
            return Batch(subset)
        # a DataArray holds no other variable, so it carries the attributes of its burst
        grids = {k: da.copy(deep=False) for k, da in subset.items()}
        for k, da in grids.items():
            da.attrs = {**self[k].attrs, **da.attrs}
        batch_type = self._batch_type_for_subset(grids)
        # For single variable access, default to Batch if not BatchComplex
        if batch_type is None:
            batch_type = Batch
        return batch_type(grids)

    def __getattr__(self, name: str):
        """A variable: `stack.ele` is `stack['ele']`, a Batch of DataArrays, as xarray's ds.ele is ds['ele']. Nothing else is
        forwarded to BatchComplex (GKB U4): any other name raises AttributeError."""
        if not name.startswith('_') and self:
            try:
                return self._select_var(name)
            except KeyError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def complex(self) -> BatchComplex:
        """Return the complex data of this Stack as a BatchComplex, without the transform:
        stack.complex() is BatchComplex(stack). numpy and math on the data go through it,
        as a Stack itself is not an array (N81).

        >>> stack.complex().abs()
        """
        return BatchComplex(self)

    def transform(self) -> Batch:
        """Return a Batch view of this Stack's transform: its non-complex (y, x) grids
        (azi, rng, ele, ...) and every non-gridded variable, never the data
        (BatchComplex(stack)).

        Also `northing` and `easting`, each burst's own y/x as a vector.

        `azi` and `rng` are the BURST'S grid and restart in each one, so a
        plane across bursts written in them is a sawtooth; the map coordinates
        share a lattice.
        """
        import numpy as np
        out = {}
        # Batch(stack) keeps the transform and leaves the data grids out
        for key, ds in Batch(self).items():
            if 'y' not in ds.dims or 'x' not in ds.dims:
                out[key] = ds
                continue
            # ONE AXIS EACH, not a raster: a map coordinate is constant along
            # the other axis, so it is the coordinate vector and nothing is
            # broadcast, stored or shipped per block that a 1-D read gives.
            out[key] = ds.assign(
                northing=(('y',), np.asarray(ds.y.values, np.float32),
                          dict(ds.y.attrs)),
                easting=(('x',), np.asarray(ds.x.values, np.float32),
                         dict(ds.x.attrs)))
        return Batch(out)

    def incidence(self) -> Batch:
        """Incidence angle (radians) of each burst's grid at each pixel's own height (the transform's ele), from the
        stack's per-date numbers and the crop polygon (see Batch.incidence())."""
        return self.transform().incidence()

    def look(self) -> Batch:
        """Look azimuth (degrees, clockwise from geodetic north, satellite to target) of each burst's grid at each pixel's own
        height (the transform's ele), from the orbit geometry (see Batch.look())."""
        return self.transform().look()

    def troposphere_phase(self, baseline: 'str | None' = 'BPR', ref=None) -> Batch:
        """The tropospheric phase (radians) of each burst on the stack's grid and dates, the phase each date adds to the
        interferometric phase: corrected phase = phase minus this result. The sign is the library's: -4 pi / wavelength times
        the one-way delay of the model (a longer path is a negative phase).

        baseline : str or None, default 'BPR': per-date baseline whose |value| picks the reference date; None: absolute phase.
        ref : None, int, str or datetime: the date whose phase is subtracted at each pixel (NaN there gives NaN); None: each burst's own smallest |baseline| date (a burst without the baseline takes its first date), so the bursts may differ; ref= names one date for all bursts (a burst without it raises one ERROR line).

        The folder is the tropo= of load() (a VMF3 or ERA5 folder of the atmosphere model, the model is detected from it).
        The model's tables are computed once at this call and held for the stack (the dates and tiles of the bursts); each
        pixel takes the model's delay at its own ellipsoidal height (transform().ele) and incidence, by the kernel of the
        Stack (utils_atmosphere): the zenith delay bilinear at the pixel's lat/lon, the mapping at the pixel's incidence, for the
        date minus the reference. A pixel outside the model grid is NaN, with one warning per block.

        Lazy, as the Tiles reprojection: no phase is computed and no model file is read at the call beyond the held tables (ref=None reads only the per-date baseline, 1-D per burst); each chunk is computed when the result
        is. The chunks are the burst's (date, y, x) chunks. Returns a Batch of Datasets with the variable 'phase'
        (date, y, x), float32.

        >>> stack = Stack().load('gas.zarr', tropo='dem_tiles/VMF3_OP')
        >>> phase = stack.troposphere_phase()
        """
        return _atmosphere_phase(self, 'tropo', None, 'troposphere_phase', baseline=baseline, ref=ref)

    def ionosphere_phase(self, factor: float = 0.78, geocentric: bool = True, baseline: 'str | None' = 'BPR', ref=None) -> Batch:
        """The ionospheric phase (radians) of each burst on the stack's grid and dates, the phase each date adds to the
        interferometric phase: corrected phase = phase minus this result. The sign is the library's: +4 pi / wavelength
        times the one-way slant delay, the ionosphere shortens the phase path.

        The folder is the iono= of load() (a folder of the CODE IONEX maps). Each pixel takes the TEC at its pierce
        point: the point of the single-layer shell (R + H, 450 km above the earth sphere of the maps) where the line of
        sight from the satellite to the pixel crosses it (TN-200 step 2), from the burst's geometry at its terrain height
        (the midpoint of its ele actual_range). The TEC is bilinear over the 4 nodes around each pierce point, times the
        mapping at the pierce point (TN-200 Eq. 21 and 22, R = the pixel's distance from the earth centre) and factor. factor
        is the ionospheric factor of the mapping: the delay is factor times the slant TEC, so it scales the correction
        linearly. Proposed values: 0.78 for Sentinel-1
        (MPC 2025, at solar maximum; the default used here), 0.90 for Sentinel-1 with the older products (TN-200 and
        ETAD before 2025, which changes with the Sun's state), 0.76 for ALOS-2 (published), about 0.79 for NISAR (from
        the profile model's altitude slope anchored at Sentinel-1; not measured). The L-band accuracy is well below
        Sentinel-1's, so the single default 0.78 is used for all of them. A pixel outside the model grid, or an
        incidence outside [0, 90) degrees, raises one ERROR line.

        geocentric=True (default) maps each pixel at the geocentric zenith of the exact ray at its pierce point (ETAD ATBD
        2.7 Eq. 45-46). geocentric=False maps it at the ellipsoid-normal zenith of the pixel (FRM4SAR TN-200 Eq. 22), for
        verifications against TN-200; the pierce point is the same for both.

        baseline : str or None, default 'BPR': per-date baseline whose |value| picks the reference date; None: absolute phase.
        ref : None, int, str or datetime: the date whose phase is subtracted at each pixel (NaN there gives NaN); None: each burst's own smallest |baseline| date (a burst without the baseline takes its first date), so the bursts may differ; ref= names one date for all bursts (a burst without it raises one ERROR line).

        Lazy, as troposphere_phase(): the held tables are computed at the call; the chunks are the burst's (date, y, x) chunks.
        Returns a Batch of Datasets with the variable 'phase' (date, y, x), float32.

        >>> stack = Stack().load('gas.zarr', iono='dem_tiles')
        >>> phase = stack.ionosphere_phase(factor=0.78)
        """
        return _atmosphere_phase(self, 'iono', float(factor), 'ionosphere_phase', geocentric=bool(geocentric),
                                 baseline=baseline, ref=ref)

    def _elevation_phase_approximate(self) -> dict:
        """`{burst_id: value}` at the burst centre. See Batch."""
        return Batch._elevation_phase_approximate(self)

    def _incidence_approximate(self) -> dict:
        """`{burst_id: incidence (radians)}` at the burst centre. See Batch."""
        return Batch._incidence_approximate(self)

    def median(self, window):
        """A box MEDIAN filter over the grid, the robust twin of gaussian().

        `stack.singlelook().median(30)` replaces every pixel by the median of
        the samples in a 30 m box centred on it. Where gaussian() averages, this picks
        the middle sample, so a bright neighbour, a layover spike or a single
        decorrelated date cannot drag the answer.

        SPATIAL ONLY, as gaussian() is: it filters each (y, x) plane and
        touches no other dimension. NaN is skipped rather than propagated, so
        a prefiltered raster still answers from the samples each box has, and
        a box with nothing finite in it returns NaN.

        Parameters
        ----------
        window : float or tuple of float
            The box side in METRES on the ground, one number for a square or
            (y, x). Divided by each burst's posting here, rounded to the
            nearest pixel count and bumped up to odd, because the box is
            centred on its pixel.

        Returns
        -------
        Stack
            The filtered stack, same grid, same chunking, same variables. The
            data grids are filtered (N81: BatchComplex(stack)); the transform
            grids and the non-gridded variables ride along untouched.

        Examples
        --------
        >>> model = stack.singlelook().median(30).fit1d()
        >>> flat = stack.singlelook().median((30, 30))   # the axes apart
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        from . import utils_xarray
        from .utils_dask import overlap_check

        # THE DATA, NOT THE TRANSFORM (N81): azi, ele and rng were median-filtered too
        data = BatchComplex(self)
        BatchCore._require_lazy(data, 'median')

        out = {}
        for key, ds in data.items():
            wy, wx = utils_xarray.meters_to_pixels(
                window, utils_xarray.spacing_of(ds), minimum=3, odd=True,
                name='median() window')
            hy, hx = wy // 2, wx // 2
            new_vars = {}
            for var in ds.data_vars:
                arr = ds[var]
                # THE GRIDS AND NOTHING ELSE. The radar metadata rides beside
                # the data and is carried through, as gaussian() carries it.
                if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                    new_vars[var] = arr
                    continue
                # the chunks as they are: median2d loops over the dates of a block (no date axis in the halo)
                dsk = arr.data
                depth = ({0: 0, 1: hy, 2: hx} if arr.ndim == 3
                         else {0: hy, 1: hx})
                filtered = da.map_overlap(
                    utils_median.median2d, dsk,
                    depth=depth, boundary='none', trim=True, dtype=dsk.dtype,
                    window_y=wy, window_x=wx)
                # a halo larger than a block raises here (the output chunks would differ from the input's)
                filtered = overlap_check(filtered, dsk, 'median', (wy, wx), (hy, hx))
                new_vars[var] = xr.DataArray(filtered, dims=arr.dims,
                                             coords=arr.coords, name=var)
            o = xr.Dataset(new_vars, attrs=ds.attrs)
            out[key] = o
        graph_check('median', out, data)
        return self._restack(data, BatchComplex(out))

    def to_vtk(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK: ``BatchCore.to_vtk()`` of ``data`` (default: this stack) with its transform.

        Parameters
        ----------
        path : str
            Output directory/filename for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack; with ``overlay``, only the topography under the overlay.
        transform : Batch | None, optional
            Topography (``ele``). If ``None``, ``self.transform()``.
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery).
        mask : bool, optional
            If True, mask topography by valid data pixels.

        Examples
        --------
        stack.to_vtk('velocity', velocity, overlay=gmap)
        stack.to_vtk('gmap', overlay=gmap)
        """
        if data is None:
            target = Batch({}) if overlay is not None else self
        elif isinstance(data, (BatchCore, Stack)):
            target = data
        else:
            target = Batch(data)
        if transform is None:
            transform = self.transform()
        if data is None and overlay is not None and transform is None:
            raise ValueError("transform is required when data=None")
        BatchCore.to_vtk(target, path, transform=transform, overlay=overlay, mask=mask)

    def to_vtm(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to a seamless multiblock VTK tree: ``BatchCore.to_vtm()`` of ``data`` (default: this stack) with its transform.

        Parameters
        ----------
        path : str
            Output directory of the VTK tree.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack.
        transform : Batch | None, optional
            Topography (``ele``). If ``None``, ``self.transform()``.
        overlay : None
            Not built: raises. Use ``to_vtk()`` for an overlay.
        mask : bool, optional
            If True, z is NaN where the first pair has no data.

        Examples
        --------
        stack.to_vtm('velocity', velocity)
        """
        if data is None:
            target = self
        elif isinstance(data, (BatchCore, Stack)):
            target = data
        else:
            target = Batch(data)
        if transform is None:
            transform = self.transform()
        BatchCore.to_vtm(target, path, transform=transform, overlay=overlay, mask=mask)

    def to_vtks(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK from a Batch.

        Parameters
        ----------
        path : str
            Output directory for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Accepts any mapping convertible to ``Batch``.
        transform : Batch | None, optional
            Optional transform Batch providing topography (`ele`).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.
        """
        import os
        import numpy as np
        import xarray as xr
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        target = data if data is not None else self
        if not isinstance(target, (BatchCore, Stack)):
            target = Batch(target)
        tfm = transform
        if tfm is not None and not isinstance(tfm, (BatchCore, Stack)):
            tfm = Batch(tfm)

        if not target:
            return

        def _interp_to_grid(source: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
            if {'y', 'x'}.issubset(source.dims):
                return source.interp(y=target_da.y, x=target_da.x, method='linear')
            if {'lat', 'lon'}.issubset(source.dims):
                if {'lat', 'lon'}.issubset(target_da.coords):
                    return source.interp(lat=target_da.lat, lon=target_da.lon, method='linear')
                return source.rename({'lat': 'y', 'lon': 'x'}).interp(y=target_da.y, x=target_da.x, method='linear')
            return source

        def _format_pair(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"{utils_xarray.format_date(val[0])}_{utils_xarray.format_date(val[1])}"
            return utils_xarray.format_date(val)

        os.makedirs(path, exist_ok=True)

        # the one exported grid of each burst: its integer nodata is not drawn, ONE WARNING
        from .BatchCore import _skip_nodata, _warn_int_nodata
        _first = BatchCore._grids_of(next(iter(target.values())))
        if _first:
            _warn_int_nodata('to_vtks', dict([next((n, a.dtype) for n, a in _first.items())]))
        with tqdm(total=len(target), desc='Exporting VTK') as pbar:
            for burst, ds in target.items():
                if not ds.data_vars:
                    pbar.update(1)
                    continue

                # THE GRID is exported, and its own dims say whether there are pairs:
                # after mean('pair') the carried metadata still has its pairs, and
                # reading them off the Dataset wrote <burst>_19700101.vtk once per pair
                data_var = BatchCore._grid_vars(ds)[0]
                base_da = ds[data_var]

                if 'pair' in base_da.dims:
                    pair_coord = ds.coords.get('pair')
                    pair_values = pair_coord.values if pair_coord is not None else range(ds.sizes.get('pair', 0))
                    export_items = []
                    for i, pair_val in enumerate(pair_values):
                        ds_slice = ds.isel(pair=i)
                        if 'pair' in ds_slice.dims:
                            ds_slice = ds_slice.squeeze('pair', drop=True)
                        else:
                            ds_slice = ds_slice.squeeze(drop=True)
                        export_items.append((pair_val, ds_slice))
                else:
                    export_items = [(None, ds)]

                for pair_val, ds_item in export_items:
                    # an integer grid's nodata is not drawn, as to_dataset() sets it (_skip_nodata)
                    base_da_item = _skip_nodata(ds_item[data_var])
                    layers = [base_da_item.rename(data_var)]

                    if tfm is not None and burst in tfm:
                        tfm_ds = tfm[burst]
                        topo_da = tfm_ds.get('ele') if 'ele' in tfm_ds else tfm_ds.get('z') if 'z' in tfm_ds else None
                        if topo_da is not None:
                            topo_da = _interp_to_grid(topo_da, base_da_item)
                            if mask:
                                topo_da = topo_da.where(np.isfinite(base_da_item))
                            layers.append(topo_da.rename('z'))

                    if overlay is not None:
                        if not isinstance(overlay, xr.DataArray):
                            raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")

                        ov = overlay
                        if 'band' not in ov.dims:
                            ov = ov.expand_dims('band')
                        try:
                            ov = ov.sel(y=slice(float(base_da_item.y.min()), float(base_da_item.y.max())),
                                        x=slice(float(base_da_item.x.min()), float(base_da_item.x.max())))
                        except Exception:
                            try:
                                ov = ov.sel(lat=slice(float(base_da_item.lat.min()), float(base_da_item.lat.max())),
                                            lon=slice(float(base_da_item.lon.min()), float(base_da_item.lon.max())))
                            except Exception:
                                pass
                        ov = _interp_to_grid(ov, base_da_item)
                        layers.append(ov.rename('colors'))

                    ds_out = xr.merge(layers, compat='override', join='left')
                    vtk_grid = as_vtk(ds_out)

                    pair_suffix = ''
                    if pair_val is not None:
                        pair_suffix = f"_{_format_pair(pair_val)}"

                    filename = os.path.join(path, f"{burst}{pair_suffix}.vtk")

                    writer = vtkStructuredGridWriter()
                    writer.SetFileName(filename)
                    writer.SetInputData(vtk_grid)
                    writer.SetFileType(VTK_BINARY)
                    writer.Write()

                pbar.update(1)

    def compute(self, *batches: BatchCore) -> tuple:
        """Compute multiple Batch objects together efficiently.

        This method materializes multiple dependent Batch objects in a single
        dask graph execution, which is faster than computing them separately
        because shared computations are only performed once.

        When called without arguments, computes the Stack's own lazy data
        (equivalent to BatchCore.compute()).

        Stack serves as a unified interface - use empty Stack() for utility operations:
        - stack.compute(batch1, batch2) or Stack().compute(batch1, batch2)

        Parameters
        ----------
        *batches : BatchCore
            One or more Batch objects to compute together.
            If empty, computes the Stack itself.

        Returns
        -------
        tuple or Stack
            Tuple of computed Batch objects in the same order as input,
            or computed Stack when called without arguments.

        Examples
        --------
        >>> mintf, mcorr = stack.phasediff(pairs, wavelength=200)
        >>> mintf, mcorr = Stack().compute(mintf.downsample(20), mcorr.downsample(20))
        >>> stack_c = stack.compute()  # compute Stack itself
        """
        if not batches:
            return self._as_stack(BatchCore.compute(self._batch_view()))
        from .Batches import Batches
        return Batches(batches).compute()

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = None, debug=False, overwrite: bool = True):
        """Open or save a Batch/Batches snapshot.

        This is a convenience passthrough to Batches.snapshot(). Stack itself
        is never saved via snapshot() — use Stack.load()/save() for that.

        Parameters
        ----------
        *args : BatchCore or str
            Batch objects to save, or store path string.
        store : str, optional
            Path to the Zarr store (alternative to first positional arg).
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
        Batches or tuple
            Opened or saved Batch objects.

        Examples
        --------
        >>> # Open existing snapshot
        >>> intfcorr = stack.snapshot('intfcorr')
        >>> mintf, mcorr = stack.snapshot('mintf_corr')
        >>> # Save batches
        >>> mintf, mcorr = stack.snapshot(mintf, mcorr, store='mintf_corr')
        """
        from .Batches import Batches
        from . import utils_io

        # Handle case where first arg is store path
        if len(args) == 1 and isinstance(args[0], str):
            store = args[0]
            args = ()

        # If no batch args provided — save Stack (if non-empty) or open existing
        if len(args) == 0:
            if store is None:
                raise ValueError("store path is required to open snapshot")
            if len(self) > 0 and utils_io.write_wanted(store, storage_options, overwrite):
                # Non-empty Stack: save itself, then reopen
                utils_io.save(dict(self.items()), class_names=['Stack'], store=store, storage_options=storage_options,
                             caption=caption or 'Snapshotting...', debug=debug)
            return utils_io.open(store=store, storage_options=storage_options,
                                n_jobs=-1, debug=debug)

        # Save mode - args are batches
        return Batches(args).snapshot(store=store, storage_options=storage_options, caption=caption, debug=debug,
                                      overwrite=overwrite)

    def archive(self, *args, store: str | None = None, caption: str | None = None,
                compression: int = 6, debug=False):
        """Save or open an archive of the Stack or Batch objects as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Stack serves as a unified interface - use empty Stack() for utility operations:
        - stack.archive('path.zip') on non-empty Stack saves the Stack
        - Stack().archive('path.zip') on empty Stack opens existing archive
        - stack.archive(batch1, batch2, store='path.zip') saves batches

        Parameters
        ----------
        *args : BatchCore or str
            Batch objects to save, or store path string.
        store : str, optional
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
        Stack or tuple
            The saved Stack, or tuple of opened/saved Batch objects.

        Examples
        --------
        >>> # Save stack itself
        >>> stack.archive('mystack.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> stack.archive('mystack.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> stack.archive('gs://bucket/mystack.zip')
        >>> # Open existing archive
        >>> mintf, mcorr = Stack().archive('mintf_corr.zip')
        >>> # Save batches
        >>> mintf, mcorr = stack.archive(mintf, mcorr, store='mintf_corr.zip')
        """
        import zipfile
        import tempfile
        import os
        import fsspec
        import zarr
        from .Batches import Batches
        from . import utils_io

        # Handle case where first arg is store path
        if len(args) == 1 and isinstance(args[0], str):
            store = args[0]
            args = ()

        if store is None:
            raise ValueError("store path is required for archive")

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

        # If no batch args provided
        if len(args) == 0:
            # Empty Stack -> open mode, non-empty Stack -> save mode
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
                result = Batches().snapshot(store=zip_store, caption=caption or 'Opening archive...', debug=debug)
                zip_store.close()
                # Unwrap single Stack from tuple
                if len(result) == 1 and isinstance(result[0], Stack):
                    return result[0]
                return result
            # Save self - write to temp directory, then zip
            temp_dir = tempfile.mkdtemp()
            try:
                utils_io.save(dict(self.items()), class_names=['Stack'], store=temp_dir, storage_options=None,
                             caption=caption or 'Archiving...', debug=debug)
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
            return self

        # Save mode - args are batches
        return Batches(args).archive(store, caption=caption, compression=compression, debug=debug)

    def to_dataframe(self,
                     datas: dict[str, xr.Dataset | xr.DataArray] | None = None,
                     crs:str|None='auto',
                     attr_start:str='BPR',
                     debug:bool=False
                     ) -> pd.DataFrame:
        """
        Return a Pandas DataFrame for all Stack scenes.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing Stack scenes.

        Examples
        --------
        df = stack.to_dataframe()
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd
        import numpy as np

        if datas is not None and not isinstance(datas, dict):
            raise ValueError(f'ERROR: datas is not None or a dict: {type(datas)}')
    
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            crs = self.crs

        if datas is None:
            datas = self

        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in next(iter(datas.values())).data_vars]
        #print ('polarizations', polarizations)

        # the columns BY NAME, wherever the preprocessor stored them: the NISAR writers store lookdir and frequency
        # before BPR, where a position rule never finds them, and older stores ref_height after it. The columns
        # listed before keep their order, the stored one read backwards (S1 and NISAR stores differ there), and
        # the added ones follow their neighbour: lookdir flightDirection, band mission, frequency band (or mission)
        df_columns = {'fullBurstID', 'startTime', 'flightDirection', 'pathNumber', 'subswath', 'mission',
                      'beamModeType', 'geometry', 'BPR', attr_start}
        df_added = {'lookdir': ('flightDirection',), 'band': ('mission',), 'frequency': ('band', 'mission')}

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in datas.values():
            #print (data.id)
            per_date = [data_var for data_var in ds if ds[data_var].dims==('date',)]
            attrs = [attr for attr in per_date[::-1] if attr in df_columns]
            for attr, after in df_added.items():
                if attr in per_date:
                    anchor = next((a for a in after if a in attrs), None)
                    attrs.insert(attrs.index(anchor) + 1 if anchor is not None else len(attrs), attr)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs:
                    # Use isel + values to handle both numpy and dask arrays
                    value = ds[attr].isel(date=date_idx).values
                    # Compute if dask array
                    if hasattr(value, 'compute'):
                        value = value.compute()
                    # Extract scalar from 0-d array
                    if hasattr(value, 'item'):
                        value = value.item()
                    # Parse geometry WKT string
                    if attr == 'geometry':
                        processed_attr[attr] = wkt.loads(value)
                    else:
                        processed_attr[attr] = value
                processed_attrs.append(processed_attr)
                #print (processed_attr)
        df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        #del df['date']
        #df['polarization'] = ','.join(polarizations)
        # convert polarizations to strings like "VV,VH" to pevent confusing with tuples in the dataframe
        df = df.assign(polarization=','.join(map(str, polarizations)))
        # reorder columns to the same order as preprocessor uses
        pol = df.pop("polarization")
        # after flightDirection, and after the lookdir beside it
        df.insert(df.columns.get_loc('lookdir') + 1 if 'lookdir' in df.columns else 3, "polarization", pol)
        # round for human readability
        df['BPR'] = df['BPR'].round(1)

        group_col = 'fullBurstID'
        burst_col = 'startTime'
        #print ('df.columns[0]', df.columns[0])
        #print ('df.columns[:2][::-1].tolist()', df.columns[:2][::-1].tolist())
        df['startTime'] = pd.to_datetime(df['startTime'])
        #df['date'] = df['startTime'].dt.date.astype(str)
        df = df.sort_values(by=[group_col, burst_col]).set_index([group_col, burst_col])
        # move geometry to the end of the dataframe to be the most similar to insar_pygmtsar output
        df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]

        # Skip CRS transformation for Engineering CRS (radar coordinates mode)
        # since burst geometry is always in WGS84 from metadata
        if crs is not None:
            from pyproj import CRS as ProjCRS
            try:
                proj_crs = ProjCRS.from_user_input(crs)
                if proj_crs.type_name == 'Engineering CRS':
                    # Can't transform to engineering CRS, keep WGS84
                    return df
            except Exception:
                pass
            return df.to_crs(crs)
        return df

    @staticmethod
    def _load_zarr_array(zarr_path, group_path, name, storage_options=None):
        """
        Load a single zarr array as numpy with direct file reading.

        Reads one array from a zarr group, applies scale_factor and fill_value
        decoding. File handles are opened and closed within this call —
        no persistent descriptors.

        Uses fsspec for unified local/remote access and numcodecs for
        zstd decompression, bypassing zarr library overhead.

        Parameters
        ----------
        zarr_path : str
            Path to zarr store. Supports local and remote (fsspec) paths:
            - Local: /path/to/data.zarr
            - S3: s3://bucket/path/data.zarr
            - GCS: gs://bucket/path/data.zarr (requires gcsfs)
            - Azure: az://container/path/data.zarr (requires adlfs)
        group_path : str
            Relative path to group within zarr store. Empty string for root.
        name : str
            Array name within the group.
        storage_options : dict, optional
            Options passed to fsspec filesystem.

        Returns
        -------
        np.ndarray
            Float32 2D array with NaN for masked values.
        """
        import numpy as np
        import json
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/') if group_path else root

        arr_meta_path = f"{base_path}/{name}/zarr.json"
        with fs.open(arr_meta_path, 'r') as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        assert len(shape) >= 2, f"_load_zarr_array is for 2D+ delayed vars only, got {name} with shape {shape}"
        dtype = meta['data_type']
        attrs = meta.get('attributes', {})
        scale_factor = np.float64(attrs.get('scale_factor', 1.0))
        fill_value = attrs.get('_FillValue')

        # Single chunk only - for multi-chunk use _load_zarr_array_chunk
        chunk_suffix = '/'.join(['0'] * len(shape))
        chunk_path = f"{base_path}/{name}/c/{chunk_suffix}"
        codec = Zstd()
        with fs.open(chunk_path, 'rb') as f:
            raw = codec.decode(f.read())
        arr_int = np.frombuffer(raw, dtype=dtype).reshape(shape)

        arr_f32 = np.empty(shape, dtype=np.float32)
        np.multiply(arr_int, scale_factor, out=arr_f32, casting='unsafe')
        if fill_value is not None:
            np.putmask(arr_f32, arr_int == fill_value, np.nan)
        return arr_f32

    @staticmethod
    def _load_zarr_array_chunk(zarr_path, group_path, name, chunk_idx, chunk_shape,
                                disk_chunk_shape, scale_factor, fill_value, dtype, storage_options=None):
        """Load ONE chunk of a zarr array. Returns NaN array if chunk file doesn't exist."""
        import numpy as np
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        iy, ix = chunk_idx
        chunk_path = f"{base_path}/{name}/c/{iy}/{ix}"

        # Check if chunk exists - return NaN array if not (like native zarr)
        if not fs.exists(chunk_path):
            return np.full(chunk_shape, np.nan, dtype=np.float32)

        codec = Zstd()
        with fs.open(chunk_path, 'rb') as f:
            raw = codec.decode(f.read())
        # Zarr pads edge chunks to full disk_chunk_shape
        arr_full = np.frombuffer(raw, dtype=dtype).reshape(disk_chunk_shape)
        arr_chunk = arr_full[:chunk_shape[0], :chunk_shape[1]].copy()
        del arr_full

        arr_f32 = np.empty(chunk_shape, dtype=np.float32)
        np.multiply(arr_chunk, scale_factor, out=arr_f32, casting='unsafe')
        if fill_value is not None:
            np.putmask(arr_f32, arr_chunk == fill_value, np.nan)
        return arr_f32

    @staticmethod
    def _get_zarr_array_meta(zarr_path, group_path, name, storage_options=None):
        """Get zarr array metadata: shape, chunks, scale, fill_value, dtype."""
        import numpy as np
        import json
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        with fs.open(f"{base_path}/{name}/zarr.json", 'r') as f:
            meta = json.load(f)

        shape = tuple(meta['shape'])
        chunks = tuple(meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', shape))
        attrs = meta.get('attributes', {})
        scale_factor = np.float64(attrs.get('scale_factor', 1.0))
        fill_value = attrs.get('_FillValue')
        dtype = meta['data_type']

        return shape, chunks, scale_factor, fill_value, dtype

    @staticmethod
    def _load_zarr_complex_chunk(zarr_path, group_path, chunk_idx, chunk_shape,
                                  disk_chunk_shape, scale, fill_value, storage_options=None,
                                  re_dtype='int16'):
        """
        Load ONE zarr chunk of complex64 data. Returns NaN array if chunk files don't exist.

        Called separately for each chunk - one reader call per chunk.
        Handles edge chunks where disk_chunk_shape > chunk_shape (zarr pads edges).
        """
        import numpy as np
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')
        codec = Zstd()
        iy, ix = chunk_idx

        re_path = f"{base_path}/re/c/{iy}/{ix}"

        # Check if chunk exists - return NaN array if not (like native zarr)
        if not fs.exists(re_path):
            return np.full(chunk_shape, np.nan + 0j, dtype=np.complex64)

        # Read real part (disk has full chunk size, slice to logical size)
        with fs.open(re_path, 'rb') as f:
            raw = codec.decode(f.read())

        data = np.empty(chunk_shape, dtype=np.complex64)

        if re_dtype == 'float32':
            re_full = np.frombuffer(raw, dtype=np.float32).reshape(disk_chunk_shape)
            re_arr = re_full[:chunk_shape[0], :chunk_shape[1]]
            del re_full
            im_path = f"{base_path}/im/c/{iy}/{ix}"
            if fs.exists(im_path):
                with fs.open(im_path, 'rb') as f:
                    im_full = np.frombuffer(codec.decode(f.read()), dtype=np.float32).reshape(disk_chunk_shape)
                im_arr = im_full[:chunk_shape[0], :chunk_shape[1]]
                del im_full
                data.real[:] = re_arr
                data.imag[:] = im_arr
                del re_arr, im_arr
            else:
                data.real[:] = re_arr
                data.imag[:] = 0
                del re_arr
            return data
        else:
            # Integer (int16 or int32) with scale_factor
            int_dtype = np.dtype(re_dtype)
            re_full = np.frombuffer(raw, dtype=int_dtype).reshape(disk_chunk_shape)
            re_int = re_full[:chunk_shape[0], :chunk_shape[1]]
            del re_full

            im_path = f"{base_path}/im/c/{iy}/{ix}"
            # Amplitude-only mode: im doesn't exist → load re as amplitude, im=0
            if not fs.exists(im_path):
                if fill_value is not None:
                    mask = (re_int == fill_value)
                else:
                    mask = None
                np.multiply(re_int, scale, out=data.real, casting='unsafe')
                data.imag[:] = 0
                del re_int
            else:
                # Complex mode: load both re and im
                with fs.open(im_path, 'rb') as f:
                    im_full = np.frombuffer(codec.decode(f.read()), dtype=int_dtype).reshape(disk_chunk_shape)
                im_int = im_full[:chunk_shape[0], :chunk_shape[1]]
                del im_full
                if fill_value is not None:
                    mask = (re_int == fill_value) | (im_int == fill_value)
                else:
                    mask = None
                np.multiply(re_int, scale, out=data.real, casting='unsafe')
                del re_int
                np.multiply(im_int, scale, out=data.imag, casting='unsafe')
                del im_int

        if mask is not None and np.any(mask):
            np.putmask(data, mask, np.nan + 0j)

        return data

    @staticmethod
    def _load_zarr_complex(zarr_path, group_path, storage_options=None):
        """
        Load complex64 array from zarr - single chunk case only (S1 format).
        For multi-chunk NISAR, use _load_zarr_complex_chunk per chunk instead.
        """
        import numpy as np
        import json
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/') if group_path else root

        meta_path = f"{base_path}/re/zarr.json"
        with fs.open(meta_path, 'r') as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        re_dtype = meta.get('data_type', 'int16')
        scale = np.float64(meta['attributes'].get('scale_factor', 1.0))
        fill_value = meta['attributes'].get('_FillValue')

        codec = Zstd()
        data = np.empty(shape, dtype=np.complex64)

        with fs.open(f"{base_path}/re/c/0/0", 'rb') as f:
            raw = codec.decode(f.read())

        if re_dtype == 'float32':
            re_arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            im_path = f"{base_path}/im/c/0/0"
            if fs.exists(im_path):
                with fs.open(im_path, 'rb') as f:
                    im_arr = np.frombuffer(codec.decode(f.read()), dtype=np.float32).reshape(shape)
                data.real[:] = re_arr
                data.imag[:] = im_arr
                del re_arr, im_arr
            else:
                data.real[:] = re_arr
                data.imag[:] = 0
                del re_arr
            return data
        else:
            # Integer (int16 or int32) with scale_factor
            int_dtype = np.dtype(re_dtype)
            re_int = np.frombuffer(raw, dtype=int_dtype).reshape(shape)
            if fill_value is not None:
                mask = (re_int == fill_value)
            else:
                mask = None
            np.multiply(re_int, scale, out=data.real, casting='unsafe')
            del re_int

            # Amplitude-only mode: no 'im' directory → im=0
            im_path = f"{base_path}/im/c/0/0"
            if fs.exists(im_path):
                with fs.open(im_path, 'rb') as f:
                    im_int = np.frombuffer(codec.decode(f.read()), dtype=int_dtype).reshape(shape)
                if fill_value is not None:
                    mask |= (im_int == fill_value)
                np.multiply(im_int, scale, out=data.imag, casting='unsafe')
                del im_int
            else:
                data.imag[:] = 0

        if mask is not None and np.any(mask):
            np.putmask(data, mask, np.nan + 0j)

        return data

    @staticmethod
    def _get_zarr_slc_meta(zarr_path, group_path, storage_options=None):
        """Get SLC zarr metadata: shape, chunks, scale, fill_value."""
        import numpy as np
        import json
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        with fs.open(f"{base_path}/re/zarr.json", 'r') as f:
            meta = json.load(f)

        shape = tuple(meta['shape'])
        chunks = tuple(meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', shape))
        dtype = meta.get('data_type', 'int16')
        scale = np.float64(meta['attributes'].get('scale_factor', 1.0))
        fill_value = meta['attributes'].get('_FillValue')

        return shape, chunks, scale, fill_value, dtype

    @staticmethod
    def _zarr_meta_of(var):
        """
        (shape, chunks, scale, fill_value, dtype) of a stored zarr array, from its decoded variable in the consolidated
        tree: the encoding keeps the stored chunks, dtype, scale_factor and _FillValue, so no zarr.json is read.
        """
        import numpy as np
        enc = var.encoding
        shape = tuple(var.shape)
        chunks = tuple(enc.get('chunks') or shape)
        scale = np.float64(enc.get('scale_factor', 1.0))
        fill_value = enc.get('_FillValue')
        dtype = np.dtype(enc['dtype']).name
        return shape, chunks, scale, fill_value, dtype

    @staticmethod
    def _zarr_complex_source(zarr_path, group_path, meta, storage_options=None):
        """A burst's complex64 re/im for _zarr_dask(): (reader of disk chunk (iy, ix, chunk_shape), disk chunks).
        meta: (shape, chunks, scale, fill_value, dtype) of its re array, read by the caller."""
        import dask.array as da
        shape, zarr_chunks, scale, fill_value, re_dtype = meta
        disk = da.core.normalize_chunks(zarr_chunks, shape)
        if len(disk[0]) == 1 and len(disk[1]) == 1:
            # Single chunk: one reader for entire array
            return (lambda iy, ix, chunk_shape: (Stack._load_zarr_complex, (zarr_path, group_path, storage_options), {}),
                    disk)
        # Multi-chunk: one reader call per chunk
        return (lambda iy, ix, chunk_shape: (Stack._load_zarr_complex_chunk,
                                             (zarr_path, group_path, (iy, ix), chunk_shape, zarr_chunks,
                                              scale, fill_value, storage_options), {'re_dtype': re_dtype}),
                disk)

    @staticmethod
    def _zarr_array_source(zarr_path, group_path, name, meta, storage_options=None):
        """One float32 transform variable for _zarr_dask(): (reader of disk chunk (iy, ix, chunk_shape), disk chunks).
        meta: (shape, chunks, scale, fill_value, dtype) of the variable, read by the caller."""
        import dask.array as da
        shape, zarr_chunks, scale_factor, fill_value, dtype = meta
        disk = da.core.normalize_chunks(zarr_chunks, shape)
        if len(disk[0]) == 1 and len(disk[1]) == 1:
            # Single chunk: one reader for entire array
            return (lambda iy, ix, chunk_shape: (Stack._load_zarr_array, (zarr_path, group_path, name,
                                                                          storage_options), {}),
                    disk)
        # Multi-chunk: one reader per chunk
        return (lambda iy, ix, chunk_shape: (Stack._load_zarr_array_chunk,
                                             (zarr_path, group_path, name, (iy, ix), chunk_shape, zarr_chunks,
                                              scale_factor, fill_value, dtype, storage_options), {}),
                disk)

    @staticmethod
    def _load_chunks(disk_chunks, target_mb, floors=None):
        """The dask chunks ((y), (x)) of every grid of a loaded burst: the chunk2d logic on the disk chunks of its
        data, which output chunks never cross."""
        from .utils_dask import rechunk2d
        shape = (sum(disk_chunks[0]), sum(disk_chunks[1]))
        optimal = rechunk2d(shape, element_bytes=8, input_chunks=disk_chunks, merge=False, target_mb=target_mb, floors=floors)
        return tuple(optimal['y']), tuple(optimal['x'])

    @staticmethod
    def _load_rechunk(ds, chunks):
        """Every grid of a loaded burst in the dask chunks (1 date, chunks): _zarr_dask() gives them, so this only
        rechunks a grid a merge of polarizations with different dates changed."""
        rechunked = {}
        for var in ds.data_vars:
            arr = ds[var]
            if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                continue
            want = (((1,) * arr.shape[0],) if arr.ndim == 3 else ()) + tuple(chunks)
            if arr.data.chunks != want:
                rechunked[var] = arr.chunk(dict(zip(arr.dims, want)))
        return ds.assign(rechunked) if rechunked else ds

    @staticmethod
    def _zarr_dask(label, sources, chunks, dtype, lead):
        """
        Lazy array of zarr arrays read by disk chunk, in the dask chunks `chunks` ((y), (x)): (y, x) of one source,
        or (n, y, x) with one source per leading index (a date) when `lead`.

        sources: [(read, disk chunks)] of _zarr_complex_source() / _zarr_array_source().

        One reader task per disk chunk, and one task per output block that cuts it from the disk chunks it overlaps.
        """
        import uuid
        import numpy as np
        import dask.array as da
        from dask._task_spec import Task, TaskRef
        from dask.highlevelgraph import HighLevelGraph

        name = f'{label}-{uuid.uuid4().hex}'
        oy, ox = (np.cumsum((0,) + tuple(c)).tolist() for c in chunks)
        layer = {}
        for d, (read, disk) in enumerate(sources):
            rname = f'{label}-read-{uuid.uuid4().hex}'
            dy, dx = (np.cumsum((0,) + tuple(c)).tolist() for c in disk)
            assert dy[-1] == oy[-1] and dx[-1] == ox[-1], (label, disk, chunks)

            def overlap(edges, lo, hi):
                """[(disk chunk, start, stop within it)] of the disk chunks the range [lo, hi) overlaps."""
                first = int(np.searchsorted(edges, lo, 'right')) - 1
                last = int(np.searchsorted(edges, hi, 'left')) - 1
                return [(i, max(lo, edges[i]) - edges[i], min(hi, edges[i + 1]) - edges[i])
                        for i in range(first, last + 1)]
            for by in range(len(oy) - 1):
                ys = overlap(dy, oy[by], oy[by + 1])
                for bx in range(len(ox) - 1):
                    xs = overlap(dx, ox[bx], ox[bx + 1])
                    parts = []
                    for iy, _, _ in ys:
                        for ix, _, _ in xs:
                            key = (rname, iy, ix)
                            if key not in layer:
                                func, args, kwargs = read(iy, ix, (dy[iy + 1] - dy[iy], dx[ix + 1] - dx[ix]))
                                layer[key] = Task(key, func, *args, **kwargs)
                            parts.append(TaskRef(key))
                    key = (name, d, by, bx) if lead else (name, by, bx)
                    layout = (len(ys), len(xs), tuple((a, b) for _, a, b in ys), tuple((a, b) for _, a, b in xs),
                              bool(lead))
                    layer[key] = Task(key, _zarr_block, layout, *parts)
        out_chunks = (((1,) * len(sources),) if lead else ()) + tuple(tuple(c) for c in chunks)
        graph = HighLevelGraph.from_collections(name, layer, dependencies=())
        return da.Array(graph, name, chunks=out_chunks, dtype=dtype,
                        meta=np.empty((0,) * len(out_chunks), dtype=dtype))

    @staticmethod
    def _load_attrs(attrs: dict, where: str, notes: list, used: set) -> tuple[dict, dict]:
        """One date group's stored attributes, split into the scalars and the lists load() makes variables of.

        The scalars and the lists named in `used` (load()'s whitelist, with the names of attrs), except Conventions
        and spatial_ref, which load() handles itself. Numbers become float, as they always did; a JSON null is no
        value and is left out, so that date is filled like a date without the key. A list becomes an array of its
        own rank; a ragged one has no array shape and is an error naming it.
        """
        import numpy as np
        scalars, arrays = {}, {}
        for k, v in attrs.items():
            if k in ('Conventions', 'spatial_ref') or v is None or k not in used:
                continue
            if isinstance(v, (list, tuple)):
                try:
                    arrays[k] = np.array(v)
                except ValueError as ex:
                    raise ValueError(f'Stack.load: {where}: attribute {k!r} is a ragged list, which has no '
                                     f'array shape: {ex}') from None
            elif isinstance(v, str):
                scalars[k] = v
            elif isinstance(v, (int, float)):
                scalars[k] = float(v)
            else:
                notes.append(f'WARNING: Stack.load: {where}: attribute {k!r} is a {type(v).__name__}, neither a '
                             f'scalar nor a list, and is not loaded')
        return scalars, arrays

    @staticmethod
    def _load_date_vars(infos: list, where: str, notes: list) -> dict:
        """The per-date variables of one polarization's dates (sorted by date), from _load_attrs() of each.

        THE UNION OF THE KEYS over the dates, in the order they are first stored. A date that did not store a key
        gets no value -- NaN for a number, None for a string, which _merge_polarizations() writes as '' -- and
        never another date's: taking the first date's list wherever a date lacked it put the first orbit on every
        date, and a key only a later date stored was dropped. A list keeps its rank: (date, <key>_coef) as before
        for 1-D, (date, <key>_coef, <key>_coef1, ...) beyond. Shapes that differ between the dates cannot be one
        variable, and are an error naming the attribute and each date's shape.
        """
        import numpy as np
        import xarray as xr
        out = {}
        # 'burst' is set by the caller with the polarization replaced, 'polarization' is the variable name
        scalar_keys = list(dict.fromkeys(k for info in infos for k in info['scalar_attrs']
                                         if k not in ('burst', 'polarization')))
        array_keys = list(dict.fromkeys(k for info in infos for k in info['array_attrs']))
        both = [k for k in array_keys if k in scalar_keys]
        if both:
            raise ValueError(f'Stack.load: {where}: attribute(s) {both} are a scalar on some dates and a list on '
                             f'others; one variable along date needs one kind')
        for key in scalar_keys:
            vals = [info['scalar_attrs'].get(key) for info in infos]
            if all(v is None or isinstance(v, float) for v in vals):
                out[key] = xr.DataArray(np.array([np.nan if v is None else v for v in vals], dtype=np.float64),
                                        dims=['date'])
                continue
            if not all(v is None or isinstance(v, str) for v in vals):
                notes.append(f'WARNING: Stack.load: {where}: attribute {key!r} is a number on some dates and a string '
                             f'on others, and is loaded as strings')
            out[key] = xr.DataArray(vals, dims=['date'])
        for key in array_keys:
            arrs = [info['array_attrs'].get(key) for info in infos]
            present = [a for a in arrs if a is not None]
            if len({a.shape for a in present}) > 1:
                shapes = ', '.join(f"{str(info['date'])[:10]} {'absent' if a is None else a.shape}"
                                   for info, a in zip(infos, arrs))
                raise ValueError(f'Stack.load: {where}: attribute {key!r} has different shapes on different dates '
                                 f'({shapes}); one variable along date needs one shape')
            if len(present) < len(arrs):
                fill = (np.full(present[0].shape, np.nan) if present[0].dtype.kind in 'biuf'
                        else np.full(present[0].shape, None, dtype=object))
                arrs = [fill if a is None else a for a in arrs]
            try:
                stacked = np.stack(arrs)
            except (TypeError, ValueError) as ex:
                raise ValueError(f'Stack.load: {where}: attribute {key!r} cannot be combined over the dates: '
                                 f'{ex}') from None
            dims = ['date'] + [f'{key}_coef{i}' if i else f'{key}_coef' for i in range(stacked.ndim - 1)]
            out[key] = xr.DataArray(stacked, dims=dims)
        return out

    @staticmethod
    def _merge_polarizations(datas: list):
        """One burst Dataset from its per-polarization ones.

        A per-date variable holds ONE value per date; one the polarizations of a date store with different values
        is a merge conflict, and an error. The strings a date did not store are None through the merge, where None
        counts as missing, and '' after it: a store writes a string array, not an object array.
        """
        import xarray as xr
        ds = xr.merge(datas, compat='no_conflicts', combine_attrs='override')
        strings = {v: ds[v].fillna('').astype(str) for v in ds.data_vars
                   if ds[v].dtype == object
                   and all(s is None or isinstance(s, str) or (isinstance(s, float) and s != s)
                           for s in ds[v].values.ravel())}
        if strings:
            ds = ds.assign(strings)
        return ds

    def load(self, urls:str | list | pd.DataFrame, storage_options:dict[str, str]|None=None,
             debug:bool=False, attrs:list[str]|None=None, tropo:str|None=None, iono:str|None=None):
        """Load a store, a list of burst URLs or a DataFrame of them into this Stack, one Dataset per burst.

        Every stored burst attribute the stack functions read is loaded as a per-date variable (the whitelist
        _USED_SCALAR_ATTRS; every list it names). The argument attrs is a list of further stored attribute
        names, loaded the same way, so that a custom function can read them: each becomes a per-date variable
        with the stored values, NaN on a date that does not store it. None (the default) loads exactly the used
        names. A name that no loaded burst loads as a per-date variable is not a per-date parameter of the bursts
        (a data variable such as VV or azi, polarization, a null or a dict): it raises a ValueError naming it;
        attrs that is not a list raises a TypeError.

        The arguments tropo and iono are the folders of the atmosphere model fields (e.g. dem_tiles, where the
        ERA5, VMF3 and IONEX downloads were stored). Given, the model of each folder is detected from its VRT index,
        every scene time of the bursts is verified against it (utils_tropo.verify and utils_iono.verify: the stored
        epochs and their tile files; no model value is read here), and each burst keeps the absolute folder path
        as its attribute 'tropo' or 'iono'. A folder that holds no such model, or misses a scene time, raises one
        ERROR: line before the Stack is changed. None (the default) keeps nothing and checks nothing.

        Examples
        --------
        >>> stack = Stack().load('gas.zarr', tropo='dem_tiles/VMF3_OP', iono='dem_tiles')
        """
        if attrs is not None and (not isinstance(attrs, list) or not all(isinstance(a, str) for a in attrs)):
            raise TypeError(f'Stack.load: attrs must be a list of attribute names (str), not {type(attrs).__name__}')
        extra_attrs = list(attrs or [])
        import numpy as np
        import dask
        import dask.array as da
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        from shapely import wkt
        import os
        from insardev_toolkit import progressbar_joblib
        from insardev_toolkit.utils_pool import release_workers
        from tqdm.auto import tqdm
        import joblib
        import warnings
        # suppress the "Sending large graph of size …"
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module=r'distributed\.client',
            message=r'Sending large graph of size .*'
        )
        from distributed import get_client, WorkerPlugin
        class IgnoreDaskDivide(WorkerPlugin):
            def setup(self, worker):
                # suppress the "RuntimeWarning: invalid value encountered in divide"
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r'dask\._task_spec'
                )
        client = get_client()
        client.register_plugin(IgnoreDaskDivide(), name='ignore_divide')

        def _print_notes(notes):
            """Prints the notes of a load once each, in the order they were made."""
            for note in dict.fromkeys(notes):
                print(note, flush=True)

        # The stored burst attributes the stack functions read: the scalars, and the lists named here (a list is
        # loaded only when it is named; the others stay in the store). Each name carries its reader. The names of
        # attrs join this set for one call. Alignment params (sub_int_*, stretch_*, a_stretch_*, ashift, rshift) are
        # excluded - they're only needed for debugging alignment quality in insardev_pygmtsar
        _USED_SCALAR_ATTRS = {
            # Mission/acquisition metadata
            'startTime',           # reader: BatchCore._start_from (subscript)
            'polarization',        # reader: BatchCore.to_dataframe; also the name of the complex variable
            'burst',               # reader: BatchCore.to_dataframe (name list)
            'flightDirection',     # reader: BatchCore.to_dataframe (name list)
            'pathNumber',          # reader: BatchCore.to_dataframe (name list)
            'subswath',            # reader: BatchCore.to_dataframe (name list)
            'mission',             # reader: BatchCore.to_dataframe (name list)
            'beamModeType',        # reader: BatchCore.to_dataframe (name list)
            'fullBurstID',         # reader: BatchCore.to_dataframe (name list)
            'geometry',            # reader: Batch._fit3d_select_ps (subscript)
            'lookdir',             # reader: utils_geometry.burst_geometry (subscript)
            'band',                # display: the columns of Stack.to_dataframe (a fixed set of names)
            'frequency',           # display: the columns of Stack.to_dataframe; tells the two NISAR bands apart
            # Radar parameters for incidence, elevation, LOS calculations
            'radar_wavelength',    # reader: Batch.BatchComplex._fit3d_setup (subscript)
            'near_range',          # reader: Batch._centre_geometry_approximate (string)
            'SC_height_start',     # reader: Batch._centre_geometry_approximate (string)
            'SC_height_end',       # reader: Batch._centre_geometry_approximate (string)
            'earth_radius',        # reader: Batch._centre_geometry_approximate (string)
            'rng_samp_rate',       # reader: Batch._centre_geometry_approximate (string)
            'num_lines',           # reader: Batch._centre_geometry_approximate (string)
            'num_rng_bins',        # reader: Batch._centre_geometry_approximate; the far edge of the crop polygon
            # Baseline
            'BPR',                 # reader: Batch.filter (subscript)
            # the GMTSAR baseline triples per date (m, deg), pairs() gives the pair's by vector difference
            'baseline_start',      # reader: Batch._pair_geometry (subscript)
            'baseline_center',     # reader: utils_geometry TRIPLES, read by Batch._stack_geometry
            'baseline_end',        # reader: utils_geometry TRIPLES, read by Batch._stack_geometry
            'alpha_start',         # reader: utils_geometry TRIPLES, read by Batch._stack_geometry
            'alpha_center',        # reader: utils_geometry TRIPLES, read by Batch._stack_geometry
            'alpha_end',           # reader: utils_geometry TRIPLES, read by Batch._stack_geometry
            # Reference height for elevation computation
            'ref_height',          # reader: Batch._stack_geometry (subscript)
            # The gain formula of the azimuth alignment (align3d, G4w U2; reports/g4p_U1_gain_source.md section 2):
            # no reader in the stack files yet, the preprocessors read them too
            'PRF',                 # reader: the gain formula (align3d, planned)
            'SC_vel',              # reader: the gain formula (align3d, planned)
            'azimuthSteeringRate', # reader: the gain formula (align3d, planned)
            'azimuthFmRateT0',     # reader: the gain formula (align3d, planned)
            'azimuthFmRatePolynomial',  # a list: reader: the gain formula (align3d, planned)
        }
        # the names of attrs (one call): loaded beside the whitelist, the same rule
        _used_attrs = _USED_SCALAR_ATTRS | set(extra_attrs)

        def refuse_absent(datasets):
            """ValueError naming each attrs name that no loaded burst loads as a variable along date.

            A burst attribute is loaded as a variable with dims ('date', ...) (a number, a string or a list); a data
            variable of the burst (the image, azi, rng, ele: dims ('date', 'y', 'x') or ('y', 'x')) is not an attribute.
            A name stored only as a null, a dict, or one load() handles itself (spatial_ref, Conventions) loads no variable.
            """
            if not extra_attrs:
                return
            loaded = set()
            for dataset in datasets:
                for name in dataset.data_vars:
                    dims = dataset[name].dims
                    if 'date' in dims and 'y' not in dims and 'x' not in dims:
                        loaded.add(name)
            absent = [a for a in extra_attrs if a not in loaded]
            if absent:
                raise ValueError(f'Stack.load: attrs {absent} are not per-date parameters of the bursts')

        # Resolve dask chunk budget in main process (joblib loky workers
        # don't inherit dask.config, so get_dask_chunk_size_mb() would
        # return the default 128 MB regardless of user config).
        from .utils_dask import get_dask_chunk_size_mb, warn_floor
        _load_target_mb = get_dask_chunk_size_mb()
        floors = []  # the fixed smallest tiles of the call's grids: one WARNING after each note print

        def store_open_group_delayed(zarr_path, tree, group):
            """
            Open a fullBurstID group using delayed loading for complex data.

            The group is taken from the DataTree of the consolidated metadata that the caller opened once: no
            metadata file is read per group or per array. File handles are opened and closed cleanly for each
            load operation.
            """
            import rioxarray
            grp = tree[group]

            # Get burst subgroups (excluding transform)
            burst_keys = [k for k in grp.children if k != 'transform']
            if not burst_keys:
                # a snapshot (Stack.snapshot) keeps the stacked variables under the group: reopen it with snapshot()
                raise ValueError(f"ERROR: Stack.load('{zarr_path}'): a snapshot holds no burst groups; "
                                 f"reopen it with Stack().snapshot('{zarr_path}')")

            # Collect metadata for all bursts (no 2D data loading)
            burst_infos = []
            # the warnings, printed by the caller: this runs in a loky worker, whose output nobody sees
            notes = []
            spatial_ref = None
            for burst_key in burst_keys:
                burst_path = f"{group}/{burst_key}"

                # the burst's metadata from the consolidated tree (a copy: the caller's attributes are its own)
                ds = grp[burst_key].to_dataset(inherit=False).copy(deep=False)
                meta = Stack._zarr_meta_of(ds['re'])
                shape = ds['re'].shape
                date = np.datetime64(ds.attrs['startTime'], 's').astype('datetime64[ns]')
                polarization = ds.attrs['polarization']
                burst_name = ds.attrs['burst']

                # Capture spatial_ref from reference burst (BPR=0)
                if spatial_ref is None and ds.attrs.get('BPR', 1) == 0:
                    spatial_ref = ds.attrs.get('spatial_ref')

                # Extract scalar attrs (only whitelisted keys used by insardev) and every list attr
                scalar_attrs, array_attrs = Stack._load_attrs(ds.attrs, burst_path, notes, _used_attrs)

                # Store only what we need (not the full dataset!)
                burst_infos.append({
                    'polarization': polarization,
                    'burst_path': burst_path,
                    'burst_name': burst_name.replace(polarization, 'XX'),
                    'shape': shape,
                    'meta': meta,
                    'date': date,
                    'y': ds.y.values,
                    'x': ds.x.values,
                    'scalar_attrs': scalar_attrs,
                    'array_attrs': array_attrs,
                })
                ds.close()  # Close xarray dataset after extracting metadata

            # Group by polarization and sort by date
            polarizations = np.unique([info['polarization'] for info in burst_infos])
            pol_sources = {}
            for polarization in polarizations:
                pol_infos = sorted(
                    [info for info in burst_infos if info['polarization'] == polarization],
                    key=lambda x: x['date']
                )
                # one reader task per zarr disk chunk of each date
                pol_sources[polarization] = (pol_infos, [Stack._zarr_complex_source(zarr_path, info['burst_path'],
                                                                                   info['meta'], storage_options)
                                                         for info in pol_infos])
            # the dask chunks of every grid: chunk2d logic on the disk chunks of the first polarization's data
            chunks = Stack._load_chunks(pol_sources[polarizations[0]][1][0][1], _load_target_mb, floors)

            datas = []
            for polarization in polarizations:
                pol_infos, sources = pol_sources[polarization]
                # Stack all dates: (n_dates, y, x), in the dask chunks (1, chunks)
                stacked = Stack._zarr_dask(f'load-{polarization}', sources, chunks, np.complex64, lead=True)

                # Create xarray DataArray
                dates = [info['date'] for info in pol_infos]
                data_arr = xr.DataArray(
                    stacked,
                    dims=['date', 'y', 'x'],
                    coords={
                        'date': np.array(dates),
                        'y': pol_infos[0]['y'],
                        'x': pol_infos[0]['x'],
                    },
                )

                # Create dataset with polarization as variable name
                data_ds = xr.Dataset({polarization: data_arr})

                # Add scalar metadata as variables along date dimension
                data_ds['burst'] = xr.DataArray([info['burst_name'] for info in pol_infos], dims=['date'])
                # the loaded attributes, one variable along date each
                data_ds = data_ds.assign(Stack._load_date_vars(pol_infos, f'{group} {polarization}', notes))

                datas.append(data_ds)

            # Merge polarizations
            ds = Stack._merge_polarizations(datas)
            del datas

            # Load transform: zarr handles metadata/coords, custom reader for 2D chunks
            transform = grp['transform'].to_dataset(inherit=False).copy(deep=False)

            # Coords eagerly (small 1D arrays)
            ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)
            # the coordinates keep their attributes, actual_range among them
            ds.x.attrs.update(transform.x.attrs)
            ds.y.attrs.update(transform.y.attrs)

            # 2D vars as lazy dask arrays via custom reader (no persistent file descriptors)
            # One reader call per chunk for memory efficiency, in the dask chunks of the data
            transform_path = f"{group}/transform"
            for var in transform.data_vars:
                arr = Stack._zarr_dask(f'load-{var}', [Stack._zarr_array_source(
                    zarr_path, transform_path, var, Stack._zarr_meta_of(transform[var]), storage_options)],
                                       chunks, np.float32, lead=False)

                # the variable's attributes travel with it: actual_range is
                # how far it reaches, and reading it beats scanning the raster
                ds[var] = xr.DataArray(arr, dims=['y', 'x'],
                                       attrs=dict(transform[var].attrs))

            # Set spatial_ref
            if spatial_ref is None:
                spatial_ref = transform.attrs.get('spatial_ref')
            if spatial_ref is None:
                raise KeyError('spatial_ref')
            ds.attrs['spatial_ref'] = spatial_ref
            ds.rio.write_crs(spatial_ref, inplace=True)

            # Close the transform (metadata only, no lazy data references)
            transform.close()

            return group, Stack._load_rechunk(ds, chunks), notes

        # A LOCAL PATH IS THE CALLER'S. The chunks are read by the workers, and a
        # process worker keeps the directory it was started in, so a relative
        # path resolves somewhere else -- where every tile is missing, which a
        # zarr store legitimately reads as NaN. Made absolute here, once, where
        # the caller stands; URLs with a protocol are left as they are.
        def _local_abspath(u):
            u = os.path.expanduser(str(u))
            return u if '://' in u else os.path.abspath(u)

        # The atmosphere model folders (tropo=, iono=): the model of each is detected and every scene time of the
        # bursts verified before anything is loaded into the Stack (the check reads the VRT indexes only); the
        # absolute folder paths are kept as burst attributes, where every operation of the Stack keeps them.
        atmosphere = {}
        if tropo is not None:
            atmosphere['tropo'] = _local_abspath(tropo)
        if iono is not None:
            atmosphere['iono'] = _local_abspath(iono)

        def check_atmosphere(datasets):
            """Raises one ERROR: line when a folder holds no model of its kind or misses a scene time of the bursts."""
            if not atmosphere:
                return
            from insardev_toolkit import utils_tropo, utils_iono
            times = sorted({pd.Timestamp(d) for ds in datasets for d in ds.date.values})
            if 'tropo' in atmosphere:
                utils_tropo.model(atmosphere['tropo'])
                utils_tropo.verify(atmosphere['tropo'], times)
            if 'iono' in atmosphere:
                utils_iono.model(atmosphere['iono'])
                utils_iono.verify(atmosphere['iono'], times)

        def keep_atmosphere(datasets):
            """The folder paths as attributes of each burst Dataset (nothing when tropo and iono are None)."""
            for ds in datasets:
                ds.attrs.update(atmosphere)

        if isinstance(urls, str):
            # note: isinstance(urls, zarr.storage.ZipStore) can be loaded too but it is less efficient
            urls = _local_abspath(urls)
            zarr_path = urls  # Store for delayed loading
            # one read of the root's consolidated metadata: every group is taken from this tree in memory
            tree = xr.open_datatree(urls, engine='zarr', consolidated=True, zarr_format=3, chunks=None)
            groups = list(tree.children)
            dss = [store_open_group_delayed(zarr_path, tree, group)
                   for group in tqdm(groups, desc='Loading Dataset...'.ljust(25))]
            del tree
            # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
            release_workers()
            _print_notes([note for _, _, notes in dss for note in notes])
            warn_floor('load', floors)
            # list of key - dataset converted to dict and appended to the existing dict
            refuse_absent([ds for _, ds, _ in dss])
            check_atmosphere([ds for _, ds, _ in dss])
            keep_atmosphere([ds for _, ds, _ in dss])
            self.update((group, ds) for group, ds, _ in dss)
        # elif isinstance(urls, FsspecStore):
        #     root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
        #     dss = []
        #     for group in tqdm(list(root.group_keys()), desc='Loading Store'):
        #         dss.append(store_open_group(root, group))
        #     self.dss = dict(dss)
        #     del dss
        elif isinstance(urls, list) or isinstance(urls, pd.DataFrame):
            # load bursts and transform specified by URLs
            # this allows to load from multiple locations with precise control of the data
            if isinstance(urls, list):
                print ('NOTE: urls is a list, using it as is.')
                df = pd.DataFrame(urls, columns=['url'])
                df['fullBurstID'] = df['url'].str.rsplit('/', n=2).str[1]
                df['burst'] = df["url"].str.rsplit("/", n=2).str[2]
                urls = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
                print (urls.head())
            elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2 and len(urls.columns) == 1:
                print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
                #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
                #groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
            else:
                raise ValueError(f'ERROR: urls is not a list, or Pandas Dataframe with multiindex: {type(urls)}')
            urls = urls.assign(**{urls.columns[0]: [_local_abspath(u) for u in urls.iloc[:, 0].values]})

            dss = {}
            notes_all = []
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets...'.ljust(25)):
                df = urls[urls.index.get_level_values(0) == fullBurstID]
                burst_urls = df[df.index.get_level_values(1) != 'transform'].iloc[:,0].values
                transform_url = df[df.index.get_level_values(1) == 'transform'].iloc[:,0].values[0]

                # Read burst metadata eagerly from each URL (attrs, shape, coords)
                burst_infos = []
                notes = []
                spatial_ref = None
                for burst_url in burst_urls:
                    bds = xr.open_zarr(burst_url, consolidated=True, zarr_format=3,
                                       storage_options=storage_options)
                    shape = (bds.sizes['y'], bds.sizes['x'])
                    date = np.datetime64(bds.attrs['startTime'], 's').astype('datetime64[ns]')
                    polarization = bds.attrs['polarization']
                    burst_name = bds.attrs['burst']

                    if spatial_ref is None and bds.attrs.get('BPR', 1) == 0:
                        spatial_ref = bds.attrs.get('spatial_ref')

                    # Extract scalar attrs (only whitelisted keys used by insardev) and every list attr
                    scalar_attrs, array_attrs = Stack._load_attrs(bds.attrs, burst_url, notes, _used_attrs)

                    burst_infos.append({
                        'url': burst_url,
                        'polarization': polarization,
                        'burst_name': burst_name.replace(polarization, 'XX'),
                        'shape': shape,
                        'date': date,
                        'y': bds.y.values,
                        'x': bds.x.values,
                        'scalar_attrs': scalar_attrs,
                        'array_attrs': array_attrs,
                    })
                    bds.close()

                # Build dataset same as primary path: one reader task per zarr disk chunk of each date
                polarizations = np.unique([info['polarization'] for info in burst_infos])
                pol_sources = {}
                for polarization in polarizations:
                    pol_infos = sorted(
                        [info for info in burst_infos if info['polarization'] == polarization],
                        key=lambda x: x['date']
                    )
                    pol_sources[polarization] = (pol_infos, [Stack._zarr_complex_source(
                        info['url'], '', Stack._get_zarr_slc_meta(info['url'], '', storage_options), storage_options)
                                                             for info in pol_infos])
                # the dask chunks of every grid: chunk2d logic on the disk chunks of the first polarization's data
                chunks = Stack._load_chunks(pol_sources[polarizations[0]][1][0][1], _load_target_mb, floors)
                datas = []
                for polarization in polarizations:
                    pol_infos, sources = pol_sources[polarization]
                    stacked = Stack._zarr_dask(f'load-{polarization}', sources, chunks, np.complex64, lead=True)
                    dates = [info['date'] for info in pol_infos]
                    data_arr = xr.DataArray(
                        stacked,
                        dims=['date', 'y', 'x'],
                        coords={
                            'date': np.array(dates),
                            'y': pol_infos[0]['y'],
                            'x': pol_infos[0]['x'],
                        },
                    )
                    data_ds = xr.Dataset({polarization: data_arr})
                    data_ds['burst'] = xr.DataArray([info['burst_name'] for info in pol_infos], dims=['date'])
                    data_ds = data_ds.assign(Stack._load_date_vars(pol_infos, f'{fullBurstID} {polarization}',
                                                                   notes))

                    datas.append(data_ds)

                # Merge polarizations
                ds = Stack._merge_polarizations(datas)
                del datas

                # Load transform: zarr for metadata/coords, custom reader for 2D
                transform = xr.open_zarr(transform_url, consolidated=True, zarr_format=3,
                                         storage_options=storage_options)
                ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)
                ds.x.attrs.update(transform.x.attrs)
                ds.y.attrs.update(transform.y.attrs)
                for var in transform.data_vars:
                    # one reader task per zarr disk chunk, as the store path does
                    ds[var] = xr.DataArray(
                        Stack._zarr_dask(f'load-{var}', [Stack._zarr_array_source(
                            transform_url, '', var, Stack._get_zarr_array_meta(transform_url, '', var, storage_options),
                            storage_options)],
                                         chunks, np.float32, lead=False),
                        dims=['y', 'x'], attrs=dict(transform[var].attrs)
                    )

                if spatial_ref is None:
                    spatial_ref = transform.attrs.get('spatial_ref')
                if spatial_ref is None:
                    raise KeyError('spatial_ref')
                ds.attrs['spatial_ref'] = spatial_ref
                ds.rio.write_crs(spatial_ref, inplace=True)
                transform.close()
                notes_all.extend(notes)

                dss[fullBurstID] = Stack._load_rechunk(ds, chunks)
            # one print for the call
            _print_notes(notes_all)
            warn_floor('load', floors)

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            refuse_absent(dss.values())
            check_atmosphere(list(dss.values()))
            keep_atmosphere(dss.values())
            self.update(dss)
        else:
            raise ValueError(f'ERROR: urls must be a store path (str), a list of URLs, or a Pandas DataFrame '
                             f'with (fullBurstID, burst) MultiIndex: {type(urls)}')

        # Check for duplicate dates in each burst
        for key, ds in self.items():
            dates = ds.date.values
            unique, counts = np.unique(dates, return_counts=True)
            if (counts > 1).any():
                duplicates = unique[counts > 1]
                raise ValueError(f'Burst {key} contains duplicate dates: {duplicates}. '
                                 'This may be caused by corrupted data or library issues. '
                                 'Try restarting the runtime and reloading the data.')

        # chunk2d applied: spatial dims rechunked to dask budget, dim0=1.
        # User can override with .chunk2d(budget) or .chunk1d(budget) after load().

        return self

    def align(self,
              ref: int | str = 0,
              polarization: str | None = None,
              debug: bool = False,
              return_residuals: bool = False):
        """
        DATEWISE alignment: align the burst phases of each date of a complex SLC
        stack (the 'date' dimension) using interferometric double differences (ESD).

        The PAIRWISE alignment of interferograms (the 'pair' dimension: real,
        wrapped or complex) is BatchCore.align(), called on the interferograms.

        Not lazy: the burst overlaps of the stack are computed when this is
        called (one WARNING says so), and later changes to the input stack do
        not reach the corrections. Every burst's output carries a 'residual'
        variable (date,): the per-date residual after the correction, rad, 0
        for the reference date. One WARNING lists the dates whose residual is
        larger after the correction than before it; skip them with e.g.
        stack.sel(date=stack.residual < 0.5).

        For Sentinel-1 TOPS, adjacent bursts observe overlap regions at different
        azimuth squint angles, so single-date SLC cross-products have zero coherence.
        Instead, this method uses interferometric double differences: for each repeat
        date, it forms interferograms (ref × conj(rep)) per burst, then computes the
        double difference between adjacent bursts' interferograms to measure the
        burst-to-burst phase jump. These jumps are decomposed into per-date, per-burst
        corrections via global least-squares, then applied to the SLC data.

        The reference date is assumed to have zero burst-to-burst phase offsets
        (it defines the coregistration geometry).

        Parameters
        ----------
        ref : int or str, optional
            Reference date index or date string. Default is 0 (first date).
            The reference date gets zero correction.
        polarization : str, optional
            Polarization to use for offset estimation. Auto-detected if
            only one complex variable exists, otherwise defaults to 'VV'.
            Corrections are applied to all complex variables.
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return per-date residuals (rad). Default is False.

        Returns
        -------
        Stack or tuple
            If return_residuals is False:
                Phase-corrected Stack, each burst with the 'residual' variable (date,).
            If return_residuals is True:
                (corrected_stack, residuals) where residuals is list[float].

        Examples
        --------
        >>> # Align burst phases before interferogram formation
        >>> stack_aligned = stack.align()
        >>> intf, corr = stack_aligned.pairs(baseline).interferogram(wavelength=30)
        """
        import numpy as np
        import xarray as xr
        from . import utils_align

        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Auto-detect polarization (must be complex)
        sample_ds = self[ids[0]]
        available_pols = [v for v in sample_ds.data_vars
                         if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims
                         and np.issubdtype(sample_ds[v].dtype, np.complexfloating)]
        if polarization is None:
            if not available_pols:
                raise ValueError("No complex variables found in Stack")
            polarization = available_pols[0]

        # Get dates
        sample_da = sample_ds[polarization]
        if 'date' not in sample_da.dims:
            raise ValueError("Stack.align() requires complex SLC data with date dimension")
        n_dates = sample_da.sizes['date']
        dates = sample_da.coords['date'].values

        # Resolve reference date index
        if isinstance(ref, str):
            ref_idx = list(dates).index(np.datetime64(ref))
        else:
            ref_idx = int(ref)

        def with_residual(stack, residuals):
            """Every burst with the per-date 'residual' variable (rad, float32); the grids untouched."""
            res = np.asarray(residuals, dtype=np.float32)
            return type(stack)({bid: ds.assign(residual=xr.DataArray(res, dims=('date',)))
                                for bid, ds in stack.items()})

        if n_dates < 2:
            if debug:
                print('align(): need at least 2 dates for double-difference', flush=True)
            aligned = with_residual(self, [0.0] * n_dates)
            graph_check('align', aligned._data, self._data)
            return (aligned, [0.0]) if return_residuals else aligned

        if debug:
            print(f'align(): {n_bursts} bursts, {n_dates} dates, ref=date[{ref_idx}], pol={polarization}', flush=True)

        # Collect burst extents
        extents = {}
        for bid in ids:
            da = self[bid][polarization]
            extents[bid] = utils_align.extent(da.coords['y'].values, da.coords['x'].values)

        # Find overlapping burst pairs
        overlap_pairs = [(ids[i], ids[j]) for i, j in utils_align.overlap_pairs([extents[bid] for bid in ids])]

        if not overlap_pairs:
            if debug:
                print('No overlapping bursts found', flush=True)
            aligned = with_residual(self, [0.0] * n_dates)
            graph_check('align', aligned._data, self._data)
            return (aligned, [0.0] * n_dates) if return_residuals else aligned

        if debug:
            print(f'Found {len(overlap_pairs)} overlapping burst pairs', flush=True)

        # For each repeat date, compute double-difference at each overlap:
        #   dd = intf_burst1 × conj(intf_burst2)
        #   where intf_burst = burst[ref_date] × conj(burst[rep_date])
        # Phase(dd) = burst_jump(ref_date) - burst_jump(rep_date) ≈ -burst_jump(rep_date)
        # since ref_date is the coregistration reference (jump ≈ 0).
        import dask
        rep_dates = [d for d in range(n_dates) if d != ref_idx]

        # Pre-compute overlap y,x ranges for each pair (avoid loading full bursts)
        overlap_slices = {}
        for id1, id2 in overlap_pairs:
            y1 = self[id1][polarization].coords['y'].values
            y2 = self[id2][polarization].coords['y'].values
            x1 = self[id1][polarization].coords['x'].values
            x2 = self[id2][polarization].coords['x'].values
            overlap_slices[(id1, id2)] = utils_align.overlap_window(y1, y2, x1, x2)

        n_total_dds = len(rep_dates) * len(overlap_pairs)
        if debug:
            print(f'Computing {n_total_dds} double differences...', flush=True)

        # Build lazy dask graphs that reduce each overlap DD to two scalars (complex sum
        # and valid count). dask.compute() schedules them in parallel across workers;
        # each worker materializes only one overlap region at a time, and only the
        # scalar results are returned — full DD arrays are never collected to the client.
        dd_keys = []    # (d_rep, k) for result indexing
        dd_sums = []    # lazy scalar: nansum of DD complex values
        dd_counts = []  # lazy scalar: count of finite pixels
        for d_rep in rep_dates:
            for k, (id1, id2) in enumerate(overlap_pairs):
                y_sl, x_sl = overlap_slices[(id1, id2)]
                da1_ov = self[id1][polarization].sel(y=y_sl, x=x_sl)
                da2_ov = self[id2][polarization].sel(y=y_sl, x=x_sl)
                dd_sum, dd_count = utils_align.double_difference_sums(da1_ov.isel(date=ref_idx), da1_ov.isel(date=d_rep),
                                                                      da2_ov.isel(date=ref_idx), da2_ov.isel(date=d_rep))
                dd_sums.append(dd_sum)
                dd_counts.append(dd_count)
                dd_keys.append((d_rep, k))

        # ONE OVERLAP PAIR PER DASK WORKER in each dask.compute(), its double differences of every repeat date, returning
        # only scalars: a pair needs the overlap strips of its two bursts. One call for all pairs held most of the stack in
        # the workers at once, and one call per pair left all the workers but one idle (Türkiye 2023, 76 bursts: 2.2 s
        # against 15.5 s). Batches of the worker count (user, 2026-10-07: "when we have say 8 workers its reasonable to
        # process 8 at once (overlaps differ by size but maybe we can ignore it)"); without a Client, one pair at a time.
        print('WARNING: Stack.align() is not lazy: it computes the burst overlaps of the stack now; '
              'later changes to the input stack do not reach its corrections.', flush=True)
        from distributed import get_client
        try:
            n_workers = max(len(get_client().nthreads()), 1)
        except ValueError:
            n_workers = 1
        n = len(dd_keys)
        sums, counts = [None] * n, [None] * n
        for first in range(0, len(overlap_pairs), n_workers):
            batch = set(range(first, min(first + n_workers, len(overlap_pairs))))
            idx = [i for i, (_, kk) in enumerate(dd_keys) if kk in batch]
            values = dask.compute(*[dd_sums[i] for i in idx], *[dd_counts[i] for i in idx])
            for j, i in enumerate(idx):
                sums[i], counts[i] = values[j], values[len(idx) + j]

        dd_stats = {}
        for i, key in enumerate(dd_keys):
            dd_stats[key] = utils_align.double_difference(sums[i], counts[i])

        # Solve per-date global least-squares
        corrections = np.zeros((n_bursts, n_dates))

        for d_rep in rep_dates:
            rows_data = []
            for k, (id1, id2) in enumerate(overlap_pairs):
                stat = dd_stats[(d_rep, k)]
                if stat is None:
                    continue

                dd_phase, cnt = stat
                i, j = id_to_idx[id1], id_to_idx[id2]
                rows_data.append((i, j, dd_phase, cnt))

                if debug:
                    print(f'  dd {id1}-{id2} date[{d_rep}]: phase={dd_phase:.4f} rad, '
                          f'count={cnt}, weight={utils_align.edge_weight(cnt):.0f}', flush=True)

            if not rows_data:
                continue

            corrections[:, d_rep] = utils_align.solve_offsets(rows_data, n_bursts)

        if debug:
            for line in utils_align.correction_lines(ids, corrections):
                print(line, flush=True)

        # Apply corrections: multiply SLC by exp(-i * phi) per date
        # Use numpy broadcasting to avoid xarray coordinate alignment issues
        result = {}
        for bid in ids:
            ds = self[bid]
            bidx = id_to_idx[bid]
            phase_corr = corrections[bidx, :]  # shape (n_dates,)
            # Build correction array matching the date dimension position
            corr_arr = utils_align.phase_factor(phase_corr)

            new_vars = {}
            for var in ds.data_vars:
                da = ds[var]
                if np.issubdtype(da.dtype, np.complexfloating) and 'date' in da.dims:
                    # Reshape corr to broadcast: (n_dates, 1, 1, ...) matching date dim position
                    date_axis = list(da.dims).index('date')
                    shape = [1] * da.ndim
                    shape[date_axis] = n_dates
                    new_vars[var] = da * corr_arr.reshape(shape)
                else:
                    new_vars[var] = da
            result[bid] = xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)

        # Per-date residuals: the weighted mean |double difference| over the overlaps,
        # after the correction (returned and stored) and before it (for the warning)
        residuals = [0.0] * n_dates
        residuals_before = [0.0] * n_dates
        for d_rep in rep_dates:
            rows_data = [(id_to_idx[id1], id_to_idx[id2], dd_stats[(d_rep, k)][0], dd_stats[(d_rep, k)][1])
                         for k, (id1, id2) in enumerate(overlap_pairs) if dd_stats[(d_rep, k)] is not None]
            stats = utils_align.date_residual(rows_data, corrections[:, d_rep])
            if stats is not None:
                residuals[d_rep], residuals_before[d_rep] = stats

        aligned = with_residual(type(self)(result), residuals)

        # at the precision shown: a rise below 0.001 rad is not reported
        rose = utils_align.rose_dates(residuals, residuals_before, rep_dates)
        if rose:
            print(f'WARNING: Stack.align(): the residual rose on {len(rose)} date(s): '
                  + ', '.join(f'{str(dates[d])[:10]} {residuals_before[d]:.3f} -> {residuals[d]:.3f}' for d in rose)
                  + ' rad. Skip them with stack.sel(date=stack.residual < t).', flush=True)

        graph_check('align', aligned._data, self._data)
        if return_residuals:
            return aligned, residuals

        return aligned

    def pairs(self,
              pairs: list[tuple[str|int|None, str|int|None]] | np.ndarray | pd.DataFrame,
              baseline: str | None = 'BPR',
              ref=None
              ) -> Batches:
        """
        Select SLC data organized by interferometric pairs.

        Returns reference and repeat SLC data as a Batches, ready for
        interferogram computation via multiplication.

        Parameters
        ----------
        pairs : list, np.ndarray, or pd.DataFrame
            Pairs of dates as [(ref1, rep1), (ref2, rep2), ...].
            Dates can be indices (int) or dates (string, datetime64,
            Timestamp), matched to the day. [] pairs every date of each
            burst with its reference (the reference with itself included);
            a None inside a pair is the burst's reference date.
        baseline : str or None, default 'BPR'
            Per-date baseline whose smallest |value| picks each burst's
            reference date; None, or a burst lacking it: its first date.
        ref : None, int, str or datetime, default None
            The reference date of every burst, instead of the baseline pick
            (None: each burst's own pick).

        Returns
        -------
        Batches
            Batches containing [BatchComplex(ref), BatchComplex(rep)]
            with 'pair' dimension and ref/rep date coordinates. The pair
            coordinates BPR (rep - ref) and, when the store has them, the
            baseline triples baseline_start/center/end (m) and
            alpha_start/center/end (deg): the baseline vector rep - ref.

        Examples
        --------
        # Get paired SLC data
        ref, rep = stack.pairs(baseline.tolist())

        # Every date with its burst's reference (the reference with itself)
        ref, rep = stack.pairs([])

        # Manual phase difference
        phasediff = ref * rep.conj()

        # With filtering
        intf = (ref * rep.conj()).gaussian(wavelength=30).angle()
        """
        import numpy as np
        import xarray as xr

        keys = list(self.keys())
        bursts = {k: self[k] for k in keys}
        dates0 = np.asarray(bursts[keys[0]].coords['date'].values)

        # a date is its position in the first burst's dates, matched to the day; an index goes to isel() as given
        # (converting the whole array first turned dates into a string or datetime64 array, which isel() rejects)
        def _index(d, row):
            try:
                return Batch._ref_index(d, dates0)
            except (KeyError, TypeError) as e:
                raise type(e)(f'pairs(): {d!r} in pair {list(row)}: {e.args[0]}') from e

        # THE REFERENCE of each burst for a None and for []: _atmosphere_reference, the rule of the atmosphere phases.
        # ref given: that date in every burst (checked here); else each burst's smallest |baseline|, computed only when a
        # pair needs it; baseline None without ref: the first date
        refs = None if ref is None else _atmosphere_reference(self, baseline, ref, 'pairs')

        def reference(k):
            nonlocal refs
            if refs is None:
                refs = _atmosphere_reference(self, baseline, None, 'pairs')
            return 0 if refs is None else refs[k]

        # EACH BURST its own index pairs (N121): a None or [] names the burst's own reference
        idx = {k: utils_baseline.pair_indices(pairs, bursts[k].sizes['date'], lambda k=k: reference(k), _index)
               for k in keys}

        # Check for duplicate pairs
        for k in keys:
            unique, counts = np.unique(idx[k], axis=0, return_counts=True)
            if (counts > 1).any():
                duplicates = unique[counts > 1]
                raise ValueError(f'Input pairs contain duplicates: {duplicates.tolist()}')

        # Rename date->pair and reset to integer index, each burst by its own indices. The per-date 'residual' of
        # Stack.align() is dropped: carried as the reference date's value, it would
        # read as the pairwise 'residual' of BatchCore.align()
        def select(col):
            return Stack({k: bursts[k].isel(date=idx[k][:, col]).rename(date='pair')
                          .drop_vars('residual', errors='ignore')
                          .assign_coords(pair=np.arange(len(idx[k]))) for k in keys})

        data1 = select(0)
        data2 = select(1)

        # BPR differences aligned with pair dimension: BPR(rep) - BPR(ref)
        # Keep as per-burst dict structure (each burst has its own BPR)
        #
        # DataArray arithmetic, not Dataset: a Dataset operation applies to the
        # grids only, so `data2[['BPR']] - data1[['BPR']]` would carry data2's
        # BPR through unchanged and every baseline would come back 0. A Batch of
        # these DataArrays, as they are (N81): assign_coords() takes it directly.
        bpr = Batch({k: data2[k]['BPR'] - data1[k]['BPR'] for k in data1.keys()})

        # THE GMTSAR BASELINE TRIPLES the same way, rep - ref, as VECTORS: each date's baseline (length m, angle to
        # the horizontal deg, at the reference scene start, centre and end) to its horizontal and vertical
        # components, the difference, back to length and angle -- the baseline between the two dates. Not the
        # difference of the lengths or of the angles. A store without them (made before them) gets none
        triples = {}
        for k in data1.keys():
            if not all(f'{p}_{n}' in data1[k].data_vars for p in ('baseline', 'alpha')
                       for n in ('start', 'center', 'end')):
                continue
            vec = {}
            for n in ('start', 'center', 'end'):
                vec[n] = utils_baseline.baseline_difference(
                    data1[k][f'baseline_{n}'].values, data1[k][f'alpha_{n}'].values,
                    data2[k][f'baseline_{n}'].values, data2[k][f'alpha_{n}'].values)
            triples[k] = {f'baseline_{n}': ('pair', vec[n][0]) for n in vec}
            triples[k].update({f'alpha_{n}': ('pair', vec[n][1]) for n in vec})

        # EACH BURST ITS OWN ref/rep (N121): a burst's dates are its own startTimes, seconds apart
        # between bursts. The first burst's for every burst broke each per-burst match of pairs to
        # dates (rmse() of a per-date solution raised KeyError on multi-burst stores)
        ref_values = {k: np.asarray(bursts[k].coords['date'].values)[idx[k][:, 0]] for k in keys}
        rep_values = {k: np.asarray(bursts[k].coords['date'].values)[idx[k][:, 1]] for k in keys}

        def add_pair_coords(batch):
            out = {}
            for k, ds in batch.items():
                # A PAIR'S startTime IS THE EARLIER OF ITS TWO DATES (decided 2026-10-01), the burst's
                # own, on both sides: whichever leads a product, it carries the time bursts are ordered by
                if 'startTime' in ds.data_vars:
                    earlier = xr.DataArray(ref_values[k] <= rep_values[k], dims='pair')
                    ds = ds.assign(startTime=xr.where(earlier, data1[k]['startTime'], data2[k]['startTime']))
                out[k] = ds.assign_coords(ref=('pair', ref_values[k]), rep=('pair', rep_values[k]),
                                          **triples.get(k, {}))
            # BPR per burst from its Batch, as before
            return type(batch)(out).assign_coords(BPR=('pair', bpr))

        ref_batch = add_pair_coords(BatchComplex(data1))
        rep_batch = add_pair_coords(BatchComplex(data2))

        graph_check('pairs', [ref_batch, rep_batch], self._data)
        return Batches([ref_batch, rep_batch])

    def elevation(self, phase: Batch | float | list | "np.ndarray", baseline: float | None = None, transform: Batch | None = None) -> "Batch | float | np.ndarray":
        """Compute elevation (meters) from unwrapped phase grids, or the height of a scalar phase.

        Grids: stack.elevation(phase) is phase.elevation(stack) -- the stack gives each date's
        baseline triple, the reference date's numbers, the crop polygon, the CRS and the `ele`
        actual_range attribute, and no raster is read (see Batch.elevation()).

        Parameters
        ----------
        phase : Batch | float | list | np.ndarray
            Unwrapped phase grids (e.g., output of unwrap2d()), or a scalar/array
            of phase values for quick height-per-fringe calculations.
        baseline : float | None, optional
            Perpendicular baseline in meters, for a scalar/array phase only (required there):
            the height of ambiguity is stack.elevation(2*np.pi, baseline).
        transform : Batch | None, optional
            For a scalar/array phase only: the transform whose burst-centre geometry converts it.
            If ``None``, defaults to ``self.transform()``.

        Returns
        -------
        Batch | float | np.ndarray
            Elevation grids as float32 datasets, or scalar/array if input was scalar/array: the signed height
            difference the phase difference maps to, without the reference height the grids carry.
        """
        import xarray as xr
        import numpy as np

        # Handle scalar/list/array input for quick calculations
        if not isinstance(phase, (BatchCore, Stack)):
            if baseline is None:
                raise ValueError("baseline is required when phase is a scalar/list/array")
            if transform is None:
                transform = self.transform()

            # Get average parameters from first burst
            first_key = next(iter(transform.keys()))
            tfm = transform[first_key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    # .values, not .item(): the numbers of a lazily opened store are dask arrays, which have no item()
                    if var.ndim == 0:
                        return float(var.values)
                    # For per-date variables, return mean value
                    return float(var.mean().values)
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            if wavelength is None:
                raise KeyError(f"Missing radar_wavelength in transform for burst {first_key}")

            # ONE geometry, from elevation_phase(): radians per metre of height
            # per metre of baseline. This used to average SC_height_start/end as
            # a "slant range" and multiply by cos(incidence) -- the satellite
            # height where the SLANT RANGE belongs, the cosine where the SINE
            # belongs. Two errors that nearly cancel, leaving every height
            # biased. GMTSAR uses slant range and sin(incidence) in both
            # sbas.c and phase2topo.c; see Batch.elevation_phase().
            # the burst-centre value: this branch takes a SCALAR phase, which
            # has no pixel to be converted at
            import numpy as _np
            from . import utils_geometry as ug
            fac = transform._elevation_phase_approximate()[first_key]

            # phi = -fac * B_perp * dh  ->  dh = -phi / (fac * B_perp): the height of the phase DIFFERENCE, with no
            # reference height (ref_height belongs to the unwrapped grids only); utils_geometry.elevation_approx()
            return ug.elevation_approx(phase, fac, baseline)

        # GRIDS TAKE THE STACK ITSELF: their baselines are the triples, their geometry the stack's numbers
        if baseline is not None or transform is not None:
            raise ValueError('ERROR: elevation() of grids takes no baseline or transform: the stack gives them. '
                             'Use stack.elevation(phase).')
        # ONE conversion, Batch.elevation(): the grids are converted and every other variable (the burst names
        # the unwrapped phase carries) passes through (N81); a Batch of DataArrays is converted directly. This
        # loop used to be a copy without the grid filter, and raised TypeError on the burst names.
        return Batch.elevation(phase, self)

    def displacement_los(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Delegates like incidence(): the conversion is a scalar scaling by the
        mission wavelength and needs no radar geometry, so the implementation
        lives on Batch and Stack only supplies the default transform.
        """
        self._unwrapped_phase(phase, 'displacement_los')
        # Batch.displacement_los() has no None default: resolve it here.
        if transform is None:
            transform = self.transform()
        return phase.displacement_los(transform)

    @staticmethod
    def _unwrapped_phase(phase, what):
        """The phase the displacement conversions take: unwrapped, a Batch."""
        if isinstance(phase, BatchWrap):
            raise TypeError(
                f'{what}() requires unwrapped phase (Batch), got BatchWrap. '
                'Use unwrap3d() or unwrap2d() first to unwrap the phase.'
            )
        if not isinstance(phase, Batch):
            raise TypeError(
                f'phase must be Batch (unwrapped phase), got {type(phase).__name__}.'
            )

    def _displacement_component(self, phase: Batch, transform: Batch = None, func=None, suffix: str = '') -> Batch:
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin).

        Delegates like displacement_los(): ONE conversion, Batch._displacement_component(transform, func), which converts the
        grids, passes every other variable through (N81) and takes a Batch of DataArrays directly, so
        stack.displacement_vertical(phase) is phase.displacement_vertical(stack). This used to be a copy without
        the grid filter, and raised TypeError on the burst names an unwrapped phase carries."""
        self._unwrapped_phase(phase, 'displacement_los')
        # the stack itself is the geometry source when no transform is given; the Batch method computes the
        # incidence on the phase's own grid, reading no raster
        if transform is None:
            transform = self
        return phase._displacement_component(transform, func=func, suffix=suffix)

    def displacement_vertical(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute vertical displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute east-west displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.sin, suffix='eastwest')

    def align_elevation(self, **kwargs) -> "Stack":
        """Deprecated: elevation is now consistent across bursts at transform time.

        Since compute_transform_inverse() uses per-point local geocentric
        radius (R_local) instead of constant earth_radius, elevation values
        are inherently consistent across bursts with no post-hoc correction
        needed.

        Returns the Stack unchanged.
        """
        import warnings
        warnings.warn(
            "align_elevation() is deprecated and has no effect. "
            "Elevation is now consistent across bursts at transform time "
            "(per-point R_local replaces constant earth_radius).",
            DeprecationWarning,
            stacklevel=2,
        )
        return type(self)(dict(self))

    def baseline(self, days: int | None = None, meters: float | None = None,
                 invert: bool = False,
                 min_connections: int = 2, max_connections: int | None = None,
                 cleanup: bool = True) -> "Baseline":
        """Generate baseline pairs table from the Stack.

        Creates a Baseline DataFrame containing all valid interferometric pairs
        with their temporal and spatial baselines. Use ``.filter()`` on the
        result to exclude specific dates or pairs.

        .. deprecated::
            ``days`` and ``meters`` parameters are deprecated. Generate the
            full network with ``stack.baseline()`` then use
            ``baseline.filter(days=..., meters=...)`` to filter. This ensures
            you preview the full network first to spot extreme baselines or
            missing dates.

        Parameters
        ----------
        days : int, optional
            *Deprecated.* Maximum temporal separation in days.
            Use ``baseline.filter(days=...)`` instead.
        meters : float, optional
            *Deprecated.* Maximum perpendicular baseline difference in meters.
            Use ``baseline.filter(meters=...)`` instead.
        invert : bool, optional
            If True, invert reference and repeat dates. Default is False.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        max_connections : int, optional
            Maximum incoming and outgoing pairs per date. When set,
            iterates over dates chronologically and for each date limits
            both outgoing (date as ref) and incoming (date as rep) pairs
            to max_connections, dropping the longest-duration ones first.
            Applied before cleanup. Default None (no limit).
        cleanup : bool, optional
            If True (default), iteratively remove hanging dates and dates
            connected only to predecessors or only to successors.
            Set to False to keep the raw network for testing.

        Returns
        -------
        Baseline
            DataFrame subclass with columns: ref, rep, ref_baseline, rep_baseline,
            pair, baseline, duration. Has custom plot() and hist() methods.

        Examples
        --------
        >>> bl = stack.baseline()
        >>> bl.plot()  # preview full network first
        >>> bl = bl.filter(days=48, meters=100)  # then filter
        """
        import pandas as pd
        from .Baseline import Baseline

        # Get baseline table: date -> BPR
        # Extract BPR per date from first burst (all bursts have same dates)
        if not self:
            return Baseline()

        first_key = next(iter(self.keys()))
        first_ds = self[first_key]

        if 'date' not in first_ds.dims:
            raise ValueError("Stack must have 'date' dimension to compute baselines")

        # Normalize to date only (no time component)
        dates = pd.DatetimeIndex(first_ds.coords['date'].values).normalize()

        # Get BPR values - they are stored as a data variable along date dimension
        if 'BPR' in first_ds.data_vars:
            bpr_values = first_ds['BPR'].values
        elif 'BPR' in first_ds.coords:
            bpr_values = first_ds.coords['BPR'].values
        else:
            raise ValueError("Stack must have 'BPR' variable to compute baselines")

        df = utils_baseline.network_pairs(dates, bpr_values, days=days, meters=meters, invert=invert,
                                          max_connections=max_connections, min_connections=min_connections,
                                          cleanup=cleanup)
        return Baseline(df, burst_id=first_key, dates=dates)

    # THE INHERITED NAMES OUTSIDE THE REGISTRY (GKB U4): each is written here, as a plain forward to the stack's data (_data_op)
    # or as a wrapper with its own checks. The views of the held parts: crs, data, to_dataset(), map(), pipe().
    @property
    def crs(self):
        return self._batch_view().crs

    @property
    def data(self):
        return self._batch_view().data

    def to_dataset(self, *args, **kwargs):
        return self._batch_view().to_dataset(*args, **kwargs)
    to_dataset.__doc__ = BatchCore.to_dataset.__doc__

    def map(self, func, *args, **kwargs):
        return self._as_stack(self._batch_view().map(func, *args, **kwargs))
    map.__doc__ = BatchCore.map.__doc__

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)
    pipe.__doc__ = BatchCore.pipe.__doc__

    def map_da(self, *args, **kwargs):
        return self._data_op('map_da', args, kwargs)
    map_da.__doc__ = BatchComplex.map_da.__doc__

    def astype(self, *args, **kwargs):
        return self._data_op('astype', args, kwargs)
    astype.__doc__ = BatchComplex.astype.__doc__

    def abs(self, *args, **kwargs):
        return self._data_op('abs', args, kwargs)
    abs.__doc__ = BatchComplex.abs.__doc__

    def log10(self, *args, **kwargs):
        return self._data_op('log10', args, kwargs)
    log10.__doc__ = BatchComplex.log10.__doc__

    def multiply(self, *args, **kwargs):
        return self._data_op('multiply', args, kwargs)
    multiply.__doc__ = BatchComplex.multiply.__doc__

    def divide(self, *args, **kwargs):
        return self._data_op('divide', args, kwargs)
    divide.__doc__ = BatchComplex.divide.__doc__

    def clip(self, *args, **kwargs):
        return self._data_op('clip', args, kwargs)
    clip.__doc__ = BatchComplex.clip.__doc__

    def fillna(self, *args, **kwargs):
        return self._data_op('fillna', args, kwargs)
    fillna.__doc__ = BatchComplex.fillna.__doc__

    def power(self, *args, **kwargs):
        return self._data_op('power', args, kwargs)
    power.__doc__ = BatchComplex.power.__doc__

    def conj(self, *args, **kwargs):
        return self._data_op('conj', args, kwargs)
    conj.__doc__ = BatchComplex.conj.__doc__

    def angle(self, *args, **kwargs):
        return self._data_op('angle', args, kwargs)
    angle.__doc__ = BatchComplex.angle.__doc__

    def mask(self, *args, **kwargs):
        return self._data_op('mask', args, kwargs)
    mask.__doc__ = BatchComplex.mask.__doc__

    def combine_first(self, *args, **kwargs):
        return self._data_op('combine_first', args, kwargs)
    combine_first.__doc__ = BatchComplex.combine_first.__doc__

    def mean(self, *args, **kwargs):
        return self._data_op('mean', args, kwargs)
    mean.__doc__ = BatchComplex.mean.__doc__

    def gaussian(self, *args, **kwargs):
        return self._data_op('gaussian', args, kwargs)
    gaussian.__doc__ = BatchComplex.gaussian.__doc__

    def singlelook(self, *args, **kwargs):
        return self._data_op('singlelook', args, kwargs)
    singlelook.__doc__ = BatchComplex.singlelook.__doc__

    def multilook(self, *args, **kwargs):
        return self._data_op('multilook', args, kwargs)
    multilook.__doc__ = BatchComplex.multilook.__doc__

    def goldstein(self, *args, **kwargs):
        return self._data_op('goldstein', args, kwargs)
    goldstein.__doc__ = BatchComplex.goldstein.__doc__

    def dissolve(self, *args, **kwargs):
        return self._data_op('dissolve', args, kwargs)
    dissolve.__doc__ = BatchComplex.dissolve.__doc__

    def polyval(self, *args, **kwargs):
        return self._data_op('polyval', args, kwargs)
    polyval.__doc__ = BatchComplex.polyval.__doc__

    def coarsen(self, *args, **kwargs):
        return self._data_op('coarsen', args, kwargs)
    coarsen.__doc__ = BatchComplex.coarsen.__doc__

    def downsample(self, *args, **kwargs):
        return self._data_op('downsample', args, kwargs)
    downsample.__doc__ = BatchComplex.downsample.__doc__

    def interp(self, *args, **kwargs):
        return self._data_op('interp', args, kwargs)
    interp.__doc__ = BatchComplex.interp.__doc__

    def unwrap3d(self, *args, **kwargs):
        return self._data_op('unwrap3d', args, kwargs)
    unwrap3d.__doc__ = BatchComplex.unwrap3d.__doc__

    def sum(self, *args, **kwargs):
        return self._data_op('sum', args, kwargs)
    sum.__doc__ = BatchComplex.sum.__doc__

    def min(self, *args, **kwargs):
        return self._data_op('min', args, kwargs)
    min.__doc__ = BatchComplex.min.__doc__

    def max(self, *args, **kwargs):
        return self._data_op('max', args, kwargs)
    max.__doc__ = BatchComplex.max.__doc__

    def std(self, *args, **kwargs):
        return self._data_op('std', args, kwargs)
    std.__doc__ = BatchComplex.std.__doc__

    def var(self, *args, **kwargs):
        return self._data_op('var', args, kwargs)
    var.__doc__ = BatchComplex.var.__doc__

    def sqrt(self, *args, **kwargs):
        return self._data_op('sqrt', args, kwargs)
    sqrt.__doc__ = BatchComplex.sqrt.__doc__

    def square(self, *args, **kwargs):
        return self._data_op('square', args, kwargs)
    square.__doc__ = BatchComplex.square.__doc__

    def isfinite(self, *args, **kwargs):
        return self._data_op('isfinite', args, kwargs)
    isfinite.__doc__ = BatchComplex.isfinite.__doc__

    def threshold(self, *args, **kwargs):
        return self._data_op('threshold', args, kwargs)
    threshold.__doc__ = BatchComplex.threshold.__doc__

    def residuals(self, *args, **kwargs):
        return self._data_op('residuals', args, kwargs)
    residuals.__doc__ = BatchComplex.residuals.__doc__

    def rmse(self, *args, **kwargs):
        return self._data_op('rmse', args, kwargs)
    rmse.__doc__ = BatchComplex.rmse.__doc__

    chunks = property(lambda self: self._batch_view().chunks, doc=BatchCore.chunks.__doc__)
    dims = property(lambda self: self._batch_view().dims, doc=BatchCore.dims.__doc__)
    is_lazy = property(lambda self: self._batch_view().is_lazy, doc=BatchCore.is_lazy.__doc__)
    spacing = property(lambda self: self._batch_view().spacing, doc=BatchCore.spacing.__doc__)

    def from_dataset(self, data, **kwargs):
        return self._batch_view().from_dataset(data, **kwargs)
    from_dataset.__doc__ = BatchCore.from_dataset.__doc__

    def merge(self, other):
        return self._batch_view().merge(other._batch_view() if isinstance(other, Stack) else other)
    merge.__doc__ = BatchCore.merge.__doc__

    def save(self, store: str, storage_options: dict[str, str] | None = None, caption: str | None = 'Saving...', debug=False):
        return utils_io.save(dict(self.items()), class_names=['Stack'], store=store, storage_options=storage_options,
                             caption=caption, debug=debug)
    save.__doc__ = BatchCore.save.__doc__

    def open(self, store: str, storage_options: dict[str, str] | None = None, n_jobs: int = -1, debug=False):
        return self._batch_view().open(store, storage_options=storage_options, n_jobs=n_jobs, debug=debug)

    def to_dict(self) -> dict:
        return self._batch_view().to_dict()
    to_dict.__doc__ = BatchCore.to_dict.__doc__

    def to_geopackage(self, *args, **kwargs):
        return self._batch_view().to_geopackage(*args, **kwargs)
    to_geopackage.__doc__ = BatchCore.to_geopackage.__doc__

    def to_geojson(self, *args, **kwargs):
        return self._batch_view().to_geojson(*args, **kwargs)
    to_geojson.__doc__ = BatchCore.to_geojson.__doc__

    # THE REGISTRY (GKB U4): each name of _ON_DATA forwards to BatchComplex(self) on the stack's data and restacks the result;
    # each name of PART_SELECTORS runs on both parts (_on_parts); each name of _ON_MERGED runs once on the merged bursts (_on_merged).
    # The loops run once, at the end of the class body.
    for _name in _ON_DATA:
        locals()[_name] = _on_data(_name)
    for _name in PART_SELECTORS:
        locals()[_name] = _on_parts(_name)
    for _name in _ON_MERGED:
        locals()[_name] = _on_merged(_name)
    del _name
