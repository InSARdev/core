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
from .utils_torch import serialize_gpu
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect




def _trend2d_accumulate_for_dask(data_blk, *transform_blk, stats=None, **kwargs):
    """Module-level entry point for the trend2d() accumulator.

    Module level so dask ships it by name; blockwise hands the transform
    rasters over as separate positional arguments.
    """
    from . import utils_detrend
    return utils_detrend.trend2d_accumulate(data_blk, tuple(transform_blk),
                                            stats, **kwargs)


def _trend2d_accumulate_half_for_dask(data_blk, *args, stats=None,
                                      n_vars=0, **kwargs):
    """One checkerboard half; the other is the full total minus this one."""
    from . import utils_detrend
    return utils_detrend.trend2d_accumulate(
        data_blk, tuple(args[:n_vars]), stats,
        coords=(args[n_vars], args[n_vars + 1]), **kwargs)


def _trend2d_finalize_for_dask(total, half=None, dates=None, *, stats=None,
                               cells=0, k=0, axes=(), label=''):
    """One date block of accumulators -> its coefficients.

    Columns are the k gradients, the constant, why the date failed if it
    did (0 solved, 1 no pixels, 2 the trend walked out of `range`, 3 the
    covariate is degenerate on this date, 4 the iteration did not converge),
    the coherence at the solution, the coherence at ZERO trend (their
    difference is what the trend bought), the sample count, and the k
    reaches -- the half-power width of each variable's own sampling.
    """
    import numpy as np
    from . import utils_detrend
    stats = np.asarray(stats, np.float64).ravel()
    g, c, coh, coh0, det, why, lim, err = utils_detrend.trend2d_fit(
        np.asarray(total), cells, k, axes=axes,
        half=(np.asarray(half) if half is not None else None))
    span = np.maximum(stats[k:2 * k], 1e-30)[None, :]
    # the transform answers in turns across the extent; the plane wants a rate
    g = g * (2 * np.pi) / span
    if not det.all():
        # NAME THEM. A date that comes back NaN takes every pixel of the stack
        # with it downstream, so the message has to say which date and what
        # stopped it rather than how many fell out of one block.
        _reason = {1: 'no pixels',
                   2: 'the trend walked out of `range` -- widen it',
                   3: 'degenerate covariate on this date',
                   4: 'did not converge -- an unconverged number is a '
                      'plausible wrong value, so it is not returned'}
        _d = (np.asarray(dates).ravel() if dates is not None
              else np.arange(det.size))
        for _i in np.flatnonzero(~det):
            _nm = (str(np.asarray(_d[_i], dtype='datetime64[D]'))
                   if dates is not None else f'index {_d[_i]}')
            print(f"trend2d('{label}'): {_nm} did not resolve and comes back "
                  f"NaN -- {_reason.get(int(why[_i]), 'unknown')} "
                  f"(coherence {coh[_i]:.3f})", flush=True)
    M = int(cells) + 2 * utils_detrend.TREND2D_W
    n = np.asarray(total)[:, 4 * (M ** k):4 * (M ** k) + 1]
    return np.concatenate(
        [g, c[:, None], why[:, None].astype(np.float64), coh[:, None],
         coh0[:, None], n, lim * (2 * np.pi) / span,
         err * (2 * np.pi) / span], axis=1)


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




def _fit3d_scan_for_dask(block, owned, origin, cell_origin, kw, threads,
                         token=None):
    """PASS 1: the cascade's dense scan, one chunk, read inside the task.

    What leaves is what the plan always specified: the DS-candidate RANK
    RASTER for the owned pixels, and the PS-candidate WINNER GRID -- one
    candidate per independence cell, its series and the pixel it came from.
    The winner grid is the raster one pyramid level up, so the whole scene's
    candidates weigh megabytes and the PS test can be asked ONCE, over all of
    them, instead of once per chunk against whatever that chunk happened to
    contain.

    `cell_origin` names where this chunk's winner grid sits on the scene's own
    cell lattice, so the level-1 stage can lay the parts side by side.
    """
    import numpy as np
    from . import utils_arcs
    if isinstance(block, _Fit3dSlice):
        block = block.read(threads)
    # PARTIAL PER-DATE COVERAGE IS NORMAL AND TOLERATED. A pixel valid on some
    # dates and NaN on others is not a broken stack: the burst footprint drifts
    # sub-pixel between acquisitions, so a thin strip along the slanted edge is
    # imaged by only some dates. The pipeline handles it -- `_cascade_ps` only
    # ever uses pixels valid on EVERY date as candidates, so an edge pixel with
    # gaps simply is not a candidate, while the fully covered interior forms
    # the network. An earlier hard check here raised on that edge strip and
    # failed the whole fit; it was wrong. The one fatal case -- a date empty
    # over the WHOLE scene, so no all-dates pixel exists anywhere -- surfaces
    # as an empty network in `_fit3d_select_ps`, which raises there and names
    # the absent dates from their real coverage.
    wy, wx, _pey, _pex = kw['window']
    rank, W, wiy, wix = utils_arcs._cascade_pass1(
        block, owned, origin, wy, wx, tuple(kw['cell']),
        float(kw['threshold']), int(kw['min_agreeing']), threads=threads)
    # SCENE PIXELS ALREADY: the origin is handed to the scan in scene
    # coordinates, so the cells are cut on the SCENE lattice and the winner
    # positions come back on it -- one lattice for every burst, which is what
    # lets the level-1 stage lay the parts side by side exactly.
    return rank, W, wiy, wix


def _fit3d_select_ps(parts, kw, threads):
    """The PS test over the candidates in `parts`, on ONE scene lattice.

    Each part is `((W, iy, ix), (cell_y0, cell_x0))`; the grids are laid
    side by side by cell index and tested TOGETHER, a later part painting
    over an earlier one where both hold a winner. Returns None when no part
    holds a candidate, else a dict:
      g, wiy, wix : the test's coherence per lattice cell and the winner's
          scene pixel (-1 where none);
      U, iy, ix : the CERTIFIED winners' unit phasors and scene pixels, the
          network's input;
      cands, lat : how many candidates were tested and the lattice shape.
    Split out so the SELECTION can be varied -- all candidates together, or
    each chunk's alone -- while the network solve that follows stays the same.
    """
    import numpy as np
    from . import utils_arcs
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    n = parts[0][0][0].shape[0]
    NY = max(int(p[1][0]) + p[0][0].shape[1] for p in parts)
    NX = max(int(p[1][1]) + p[0][0].shape[2] for p in parts)
    Wser = np.full((n, NY, NX), np.nan, np.complex64)
    wiy = np.full((NY, NX), -1, np.int32)
    wix = np.full((NY, NX), -1, np.int32)
    for (W, iy_, ix_), (cy0, cx0) in parts:
        sy = slice(int(cy0), int(cy0) + W.shape[1])
        sx = slice(int(cx0), int(cx0) + W.shape[2])
        take = np.asarray(iy_) >= 0
        Wser[:, sy, sx] = np.where(take[None], W, Wser[:, sy, sx])
        wiy[sy, sx] = np.where(take, iy_, wiy[sy, sx])
        wix[sy, sx] = np.where(take, ix_, wix[sy, sx])

    # THE ONE FATAL DISAGREEMENT: no candidate is valid on EVERY date, so the
    # all-dates test that PS and the network both need leaves fewer than two.
    # Partial edge coverage does not cause this -- the interior is imaged by
    # every date and its candidates are all-dates valid; only a date that is
    # empty across essentially the WHOLE candidate footprint empties the
    # intersection. When that happens the fit would return a model of NaN with
    # no error, so it is caught here and the ABSENT dates are named -- those
    # far below the coverage the rest reach, not merely short of a stray edge
    # pixel.
    _amp = np.abs(Wser)
    _good = np.isfinite(_amp)
    _have = wiy >= 0
    if _have.any() and int((_good.all(axis=0) & _have).sum()) < 2:
        _cov = _good[:, _have].mean(axis=1)
        _cmax = float(_cov.max())
        _bad = np.flatnonzero(_cov < 0.5 * _cmax)
        if len(_bad) and _cmax > 0.0:
            _dvs = np.asarray(kw['date_values'])
            _nm = ', '.join(f'{str(_dvs[i])[:10]} ({100*_cov[i]:.0f}%)'
                            for i in _bad[:8])
            raise ValueError(
                f'fit3d(): no scatterer is coherent on every date -- '
                f'{len(_bad)} of {len(_cov)} dates cover under half of what '
                f'the rest do: {_nm}'
                + (' ...' if len(_bad) > 8 else ''))
    del _amp, _good
    wy, wx, pey, pex = kw['window']
    t, ele2phase, meter2rad, _car = utils_arcs._3d_fit_frame(
        kw['date_values'], kw['bperp'], kw['geometry'], n)
    g, _dh, _dv, _arcs = utils_arcs._cascade_ps(
        Wser, wiy, wix, ele2phase, t, meter2rad, wy, wx, pey, pex,
        float(kw['threshold']), int(kw['min_agreeing']),
        max_dh=kw['max_dh'], max_dv=kw['max_dv'], step_dh=kw['step_dh'],
        step_dv=kw['step_dv'], iterations=int(kw['iterations']),
        threads=threads, budget=kw['budget'],
        # the winner grid is built on the PS lattice in _cascade_pass1;
        # reach and the short-arc exclusion count in it
        pcell=utils_arcs._3d_ps_lattice(kw['cell']))
    out = dict(g=np.asarray(g, np.float32), wiy=wiy, wix=wix,
               cands=int(np.count_nonzero(wiy >= 0)), lat=(NY, NX))
    sel = np.isfinite(g) & (g >= float(kw['threshold'])) & (wiy >= 0)
    if not int(sel.sum()):
        out.update(U=np.zeros((n, 0), np.complex64),
                   iy=np.zeros(0, np.int64), ix=np.zeros(0, np.int64))
        return out
    W = Wser[:, sel]
    a = np.abs(W)
    with np.errstate(invalid='ignore', divide='ignore'):
        U = np.ascontiguousarray(
            np.where(a > 0, W / np.where(a > 0, a, 1), 0).astype(np.complex64))
    out.update(U=U, iy=wiy[sel].astype(np.int64), ix=wix[sel].astype(np.int64))
    return out


def _fit3d_ps_raster_for_dask(test, oy, ox, ny, nx):
    """The PS test written back onto ONE chunk of a burst's pixel grid.

    The test answers per lattice cell; the winner knows which pixel it is,
    so its coherence goes back to that pixel, NaN elsewhere. EVERY winner
    inside the chunk is written, whichever burst it was scanned from --
    the rule fit3d() writes its nodes by, so a burst whose grid overlaps
    another's shows the same points the model will carry there.
    """
    import numpy as np
    out = np.full((int(ny), int(nx)), np.nan, np.float32)
    if test is None:
        return out
    g = np.asarray(test['g']).ravel()
    yy = np.asarray(test['wiy']).ravel()
    xx = np.asarray(test['wix']).ravel()
    m = (yy >= 0) & (xx >= 0) & np.isfinite(g)
    yy = yy[m] - int(oy)
    xx = xx[m] - int(ox)
    ok = (yy >= 0) & (yy < int(ny)) & (xx >= 0) & (xx < int(nx))
    out[yy[ok], xx[ok]] = g[m][ok]
    return out


def _fit3d_level1_for_dask(parts, kw, threads):
    """THE PS TEST, then the one network over whatever it certified.

    The winner grids are laid on one cell lattice -- the bursts share a ground
    grid, so a chunk's grid sits a whole number of cells from the scene's
    origin -- and the long-arc test runs on the result. A candidate is
    therefore judged against every candidate within the PS extent, not against
    the ones that happen to share its chunk: at this chunking a chunk is a
    fraction of that extent wide, so the chunk-local test was asking a question
    the data in that chunk could not answer.

    THE LATER BURST WINS THE OVERLAP, as everywhere else here.
    """
    import numpy as np
    import time
    from . import utils_arcs
    _mark = time.monotonic()
    got = _fit3d_select_ps(parts, kw, threads)
    if got is not None and got['U'].shape[1] == 0:
        got = None
    cands = None if got is None else got['cands']
    lat = None if got is None else got['lat']
    # NO NETWORK IS FATAL, AND SAID SO HERE. This stage answers for the WHOLE
    # scene: with no certified PS there is no datum, so every later level has
    # nothing to attach to and every pixel comes back NaN. Returning None let
    # the run continue -- an hour of DS work writing NaN into planes nothing
    # anchors, ending in a model that is empty with no error anywhere. A block
    # holding no nodes is ordinary and still handled below; a SCENE holding
    # none is a stop.
    if got is None:
        raise ValueError(
            'fit3d(): the PS test certified no scatterer anywhere in the '
            f'scene{kw.get("tag", "")} at threshold={float(kw["threshold"]):g}. '
            'Nothing can be '
            'attached without a network, so the fit would return an empty '
            'model. Lower `threshold`, widen the PS extent (window[2:]), or '
            'check that the stack carries coherent scatterers.')
    U, iy, ix = got['U'], got['iy'], got['ix']
    if kw['debug']:
        # LEVEL 0, which is what this stage produces: the PS test and the
        # network solved over what it certified. It is the union driver's
        # FIRST pass, which is all the function's name says -- printing that
        # ordinal as the level labelled the PS network as level 1 and left
        # `level=0` reporting a level it had been asked not to run.
        print(f'DEBUG: level 0{kw.get("tag", "")}  {cands:,} candidates'
              f' on a {lat[0]} x {lat[1]} cell lattice'
              + f' -> {U.shape[1]:,} PS'
              + f'   {time.monotonic() - _mark:.1f}s', flush=True)
    if U.shape[1] < 2:
        raise ValueError(
            f'fit3d(): the PS test certified {U.shape[1]} scatterer(s) in the '
            f'whole scene{kw.get("tag", "")} at '
            f'threshold={float(kw["threshold"]):g}; a network '
            'needs at least two. The fit would return an empty model. Lower '
            '`threshold`, widen the PS extent (window[2:]), or check the '
            'stack.')
    return utils_arcs._3d_ps_network(
        U, iy, ix, kw['date_values'], bperp=kw['bperp'], window=kw['window'],
        threshold=float(kw['threshold']), geometry=kw['geometry'],
        budget=kw['budget'], consensus=kw['consensus'],
        iterations=int(kw['iterations']), max_dh=kw['max_dh'],
        max_dv=kw['max_dv'], step_dh=kw['step_dh'], step_dv=kw['step_dv'],
        max_seasonal=kw['max_seasonal'],
        err_dh=kw.get('err_dh', 5.0), err_dv=kw.get('err_dv', 1.0),
        threads=threads, debug=kw['debug'])


def _fit3d_attach_for_dask(block, part, net, kw, origin, threads, token=None,
                           emit_nodes=False):
    """PASS 2 of the union: the shared network written onto this block's grid.

    The block is read a SECOND time. That is the price of solving one network:
    pass 1 could not keep the pixels -- keeping them is what overloaded the
    host -- so the series come back from the store while the network does not.
    A node's model reaches this block only if the node stands in it; the datum
    it carries is the one the whole scene agreed on.
    """
    import numpy as np
    from . import utils_arcs
    if isinstance(block, _Fit3dSlice):
        block = block.read(threads)
    n, ny, nx = block.shape
    q = part
    # THE NODES WITHIN THIS BLOCK'S PS EXTENT, in the block's own index space.
    # Nodes outside the block are partners and nothing else -- the extent is a
    # full box centred on a pixel, so a candidate here reaches half of it
    # either side and no further. That bound is what lets the attachment go on
    # ranking against every node it is handed: a block is narrower than the
    # extent, so inside one burst the two rules already agreed, and across the
    # seam this is the reach the window always claimed. Handing it the whole
    # scene instead would rank a candidate against nodes its own window
    # refuses, which is neither cheaper nor right.
    nodes = None
    if net is not None and len(np.asarray(net['iy'])):
        _iy = np.asarray(net['iy']) - int(origin[0])
        _ix = np.asarray(net['ix']) - int(origin[1])
        _ry = max(int(kw['window'][2]) // 2, 1)
        _rx = max(int(kw['window'][3]) // 2, 1)
        m = ((_iy >= -_ry) & (_iy < ny + _ry)
             & (_ix >= -_rx) & (_ix < nx + _rx))
        if m.any():
            nodes = dict(iy=_iy[m], ix=_ix[m],
                         U=np.ascontiguousarray(net['U'][:, m]),
                         label=np.asarray(net['label'])[m],
                         vel=np.asarray(net['vel'])[m],
                         hgt=np.asarray(net['hgt'])[m],
                         coh=np.asarray(net['coh'])[m],
                         sea=np.asarray(net['sea'])[m])
            if kw['debug']:
                # COUNTED HERE, REPORTED ONCE. One line per block said the
                # same thing every time: the PS extent is wider than a block
                # by design, so nearly the whole network clears the filter and
                # only the owned count varies. The level's reducer prints the
                # spread over the blocks instead.
                _inb = ((nodes['iy'] >= 0) & (nodes['iy'] < ny)
                        & (nodes['ix'] >= 0) & (nodes['ix'] < nx))
                _offered, _ownedn = int(m.sum()), int(_inb.sum())
    _own = {}
    l, v, h, sa, cg, lv = utils_arcs._3d_ps_attach(
        block, q, nodes, kw['date_values'], out_stats=_own,
        spacing=kw['spacing'],
        bperp=kw['bperp'], window=kw['window'], threshold=kw['threshold'],
        cell=kw['cell'], geometry=kw['geometry'], budget=kw['budget'],
        level=kw['level'], max_dh=kw['max_dh'], max_dv=kw['max_dv'],
        step_dh=kw['step_dh'], step_dv=kw['step_dv'],
        consensus=kw['consensus'], iterations=kw['iterations'],
        err_dh=kw.get('err_dh', 5.0), err_dv=kw.get('err_dv', 1.0),
        threads=threads, debug=kw['debug'])
    # ONE CONVENTION ACROSS EVERY FIT, whichever driver called this
    v = -v
    sa = -sa
    out = np.concatenate(
        [l[None].astype(np.complex64), v[None].astype(np.complex64),
         h[None].astype(np.complex64), sa[None].astype(np.complex64),
         cg[None].astype(np.complex64), lv[None].astype(np.complex64)],
        axis=0)
    if not emit_nodes:
        return out
    # THE LEVEL-1 NODES, IN SCENE COORDINATES. Level 2 is a separate stage over
    # these; a chunk needs the ones its neighbours own as well as its own, so
    # they leave here in a frame every chunk shares.
    # THIS CALL'S OWN NODES, WITH NO FALLBACK. `stats` outlives the task --
    # dask reuses its worker threads, so the dict still holds whatever the
    # previous block on this thread wrote. Falling back to it when this call
    # produced nothing is not a safe default: the stale coordinates get
    # re-based to THIS block's origin and shipped as this block's level-1
    # nodes, so level 2 attaches to pixels that were never solved, carrying
    # another block's velocities and another datum. An empty table is the
    # correct answer for a block that attached nothing.
    _st = _own
    import numpy as _np
    if int(_st.get('ds_attached', 0)) and 'ds_iy' in _st:
        _nodes = dict(
            iy=_np.asarray(_st['ds_iy'], _np.int64) + int(origin[0]),
            ix=_np.asarray(_st['ds_ix'], _np.int64) + int(origin[1]),
            vel=_np.asarray(_st['ds_velocity_rad_yr'], float),
            hgt=_np.asarray(_st['ds_height_rad'], float),
            sea=_np.asarray(_st['ds_seasonal_rad']),
            label=_np.asarray(_st['ds_label']),
            gamma=(_np.asarray(_st['ds_gamma'], float)
                   if _st.get('ds_gamma') is not None else None),
            level=_np.ones(len(_st['ds_iy']), _np.int16))
    else:
        _nodes = dict(iy=_np.zeros(0, _np.int64), ix=_np.zeros(0, _np.int64),
                      vel=_np.zeros(0), hgt=_np.zeros(0),
                      sea=_np.zeros(0, _np.complex64),
                      label=_np.zeros(0, _np.int8), gamma=None)
    _nodes['_stats'] = {k: v for k, v in _st.items() if k.startswith('lvl_')}
    if kw['debug'] and nodes is not None:
        _nodes['_stats']['lvl_net_total'] = int(len(_np.asarray(net['iy'])))
        _nodes['_stats']['lvl_net_offered'] = _offered
        _nodes['_stats']['lvl_net_owned'] = _ownedn
    return out, _nodes


def _fit3d_keep(value, *deps):
    """Return `value`, having waited for `deps`. The graph has no other way
    to say "run this too" without making it a real input of something."""
    return value


def _fit3d_level_report(level_id, infos, debug=False, tag=''):
    """ONE REPORT PER LEVEL, not one per chunk.

    A chunk's numbers describe a chunk. With dozens of blocks the per-chunk
    lines are unreadable and, worse, unusable: a level's yield is the sum over
    its blocks and its error distribution is the pooled samples, neither of
    which a reader can recover from a scrolling list. The blocks of a level are
    all finished by the time this runs -- that is what the merge barrier is for
    -- so this is the first point where the level can be described at all.

    Returns the level's node tables, so it sits on the path the next level
    already depends on rather than being a side branch that has to be kept
    alive artificially.

    Under union=False every chunk is its own level and calls this with its
    one table and a `tag` naming the chunk, so the lines say what they
    describe.
    """
    import numpy as np
    tabs = [i for i in infos if i is not None]
    if not debug:
        return tabs
    st = [t.get('_stats') or {} for t in tabs]
    st = [x for x in st if x]
    if not st:
        return tabs
    tot = lambda k: sum(int(x.get(k, 0)) for x in st)

    # THE BLOCKS ALREADY REDUCED THESE. What arrives per key is a handful of
    # scalars per block (see `_lvl_stat`): counts and threshold tallies add up
    # exactly, extremes are the extreme of the blocks' own, and a percentile
    # is reported as the RANGE across blocks -- a median of medians is not the
    # median, so it is not printed as one.
    def agg(k):
        xs = [x[k] for x in st
              if isinstance(x.get(k), dict) and int(x[k].get('n', 0))]
        if not xs:
            return None
        a = dict(n=sum(int(x['n']) for x in xs),
                 blocks=len(xs),
                 min=min(x['min'] for x in xs),
                 max=max(x['max'] for x in xs),
                 p50lo=min(x['p50'] for x in xs),
                 p50hi=max(x['p50'] for x in xs),
                 p90hi=max(x['p90'] for x in xs),
                 p99hi=max(x['p99'] for x in xs))
        for key in set().union(*(set(x) for x in xs)):
            if key[:2] in ('le', 'ge', 'gt'):
                a[key] = sum(int(x.get(key, 0)) for x in xs)
        a['_thr'] = xs[0].get('_thr') or []
        return a
    def over(a):
        """(count, bound) for the single `gt` tally a key carries."""
        for op, x in a.get('_thr', []):
            if op == 'gt':
                return a.get(f'gt{x:g}', 0), x
        return 0, float('nan')
    pct = lambda a, key: 100.0 * a.get(key, 0) / max(a['n'], 1)
    rng = lambda a: (f"{a['p50lo']:.3g}" if a['p50lo'] == a['p50hi']
                     else f"{a['p50lo']:.3g}..{a['p50hi']:.3g}")

    left, att = tot('lvl_left'), tot('lvl_attached')
    print(f'DEBUG: LEVEL {level_id} over {len(st)} block(s){tag}: '
          f'{tot("lvl_cands"):,} candidates x {tot("lvl_fixed"):,} fixed nodes '
          f'-> {tot("lvl_arcs"):,} arcs; {left:,} left, {att:,} attached '
          f'({100.0 * att / max(left, 1):.1f}%)', flush=True)
    _tn = [int(x['lvl_net_total']) for x in st if 'lvl_net_total' in x]
    if _tn:
        # THE ONE FACT WORTH A LINE. The PS extent is wider than a block by
        # design, so every block is normally offered the whole network and
        # saying so 48 times says nothing. It matters only when the extent
        # STOPS reaching -- then some block was offered fewer, and that is
        # what gets printed.
        _net_n = max(_tn)
        _short = min((int(x.get('lvl_net_offered', 0)) for x in st
                      if 'lvl_net_total' in x), default=_net_n)
        print(f'DEBUG:   network {_net_n:,} nodes, '
              + ('all of them within the PS extent of every block'
                 if _short >= _net_n
                 else f'as few as {_short:,} within the PS extent of some '
                      f'block -- the extent no longer spans the scene'),
              flush=True)
    _no, _few = tot('lvl_no_consensus'), tot('lvl_too_few')
    if _no or _few:
        print(f'DEBUG:   of the {_no:,} not attached: {_few:,} had too few '
              f'partners, {_no - _few:,} had enough and disagreed', flush=True)
        _xc, _sd = tot('lvl_multi_comp'), tot('lvl_straddled')
        _xv = tot('lvl_cross_votes')
        print(f'DEBUG:   {_xc:,} candidates saw more than one component; '
              f'{_sd:,} straddled one, {_xv:,} votes crossed', flush=True)
    _ps = tot('lvl_ps')
    if _ps:
        print(f'DEBUG:   {_ps:,} PS + {att:,} DS = {_ps + att:,} measured '
              f'pixels after this level', flush=True)
    _a = agg('lvl_gamma')
    if _a:
        print(f'DEBUG:   attaching arc gamma p50 per block {rng(_a)}',
              flush=True)
    _vet, _sin = tot('lvl_vetted'), tot('lvl_solve_in')
    if _sin:
        print(f'DEBUG:   funnel: consensus vetted {_vet:,} arcs; solve was '
              f'handed {_sin:,} equations', flush=True)
    _pc, _pv = agg('lvl_pcells'), agg('lvl_pvotes')
    if _pc:
        print(f'DEBUG:   partner INDEPENDENCE: votes p50 per block '
              f'{rng(_pv) if _pv else "-"}, but distinct independence cells '
              f'p50 {rng(_pc)}; all partners in ONE cell: '
              f'{pct(_pc, "le1"):.1f}%, in <=2 cells: {pct(_pc, "le2"):.1f}%',
              flush=True)
    _pa2 = agg('lvl_parcm')
    if _pa2:
        print(f'DEBUG:   partner DISTANCE (mean over the voting arcs): p50 '
              f'per block {rng(_pa2)} worst-block p90 {_pa2["p90hi"]:.0f} p99 '
              f'{_pa2["p99hi"]:.0f} min {_pa2["min"]:.0f} max '
              f'{_pa2["max"]:.0f} m', flush=True)
    _oc = agg('lvl_offcentre')
    if _oc:
        _n_oc, _at_oc = over(_oc)
        print(f'DEBUG:   arcs entering the solve, vs the pixel\'s consensus '
              f'centre: p50 per block {rng(_oc)} worst-block p90 '
              f'{_oc["p90hi"]:.3f} p99 {_oc["p99hi"]:.3f} mm/yr; over '
              f'{_at_oc:g} mm/yr (err_dv) {_n_oc:,} of {_oc["n"]:,} '
              f'({100.0 * _n_oc / max(_oc["n"], 1):.1f}%)', flush=True)
    par = agg('lvl_partners')
    if par:
        kk = max((int(x.get('lvl_kk', 0)) for x in st), default=0)
        print(f'DEBUG:   shortlist k={kk}: partners found p50 per block '
              f'{rng(par)} min {int(par["min"])} max {int(par["max"])}',
              flush=True)
    av, ah = agg('lvl_clo_arc_v'), agg('lvl_clo_arc_h')
    dv, dh = agg('lvl_clo_ds_v'), agg('lvl_clo_ds_h')
    if av:
        print(f'DEBUG:   closure over {av["n"]:,} VOTING partners:', flush=True)
        _h = (f'   height p50 {rng(ah)} worst-block p90 {ah["p90hi"]:.2f} m'
              if ah else '')
        print(f'DEBUG:     per arc   rate p50 {rng(av)} worst-block p90 '
              f'{av["p90hi"]:.3f} mm/yr{_h}', flush=True)
    if dv:
        _n_dv, _at_dv = over(dv)
        _h = (f'   height p50 {rng(dh)} max {dh["max"]:.2f} m' if dh else '')
        print(f'DEBUG:     per DS    rate p50 {rng(dv)} max {dv["max"]:.3f} '
              f'mm/yr{_h}   over {_at_dv:g} mm/yr (err_dv): '
              f'{_n_dv:,} of {dv["n"]:,}', flush=True)
    return tabs


def _fit3d_ds_attach_for_dask(block, part, net, planes, tables, kw, origin,
                              owned, threads, token=None, emit_nodes=False,
                              level_id=2):
    """LEVEL 2, as its own stage: DS hung off the FIXED level-1 DS.

    Level 1 has already run everywhere and its values are settled. This stage
    READS them -- it never recomputes them -- which is what lets a chunk use
    the nodes its neighbours own: the tables arrive in scene coordinates and
    each chunk keeps the ones within ONE DS WINDOW of its bounds, the whole
    reach of a level-2 candidate.

    THE BLOCK IS READ WIDER THAN IT IS WRITTEN. A partner needs its phasor
    series to be fitted against, and a node another chunk owns has none inside
    this chunk's own rectangle -- so the read carries a DS-window halo while
    the CANDIDATES stay owned-only. Nothing in the halo is attached here; it
    exists to be attached to. What leaves is the owned rectangle alone.
    """
    import numpy as np
    from . import utils_arcs
    if isinstance(block, _Fit3dSlice):
        block = block.read(threads)
    S = np.ascontiguousarray(block, dtype=np.complex64)
    n, ny, nx = S.shape
    wy, wx, _pey, _pex = kw['window']
    oy0, ox0 = int(origin[0]), int(origin[1])
    y0, y1, x0, x1 = [int(v) for v in owned]
    # the PS layer, in the haloed frame -- only to keep nodes out of the
    # candidate set, as level 1 does. A scene with no network at all is the
    # same as one with no nodes near this block: nothing to exclude.
    if net is not None and len(np.asarray(net['iy'])):
        _iy = np.asarray(net['iy']) - oy0
        _ix = np.asarray(net['ix']) - ox0
        _in = (_iy >= 0) & (_iy < ny) & (_ix >= 0) & (_ix < nx)
        _oy, _ox = _iy[_in], _ix[_in]
    else:
        _oy = _ox = np.zeros(0, dtype=np.int64)
    # CANDIDATES ONLY WHERE THIS CHUNK ANSWERS. The rank raster covers the
    # owned rectangle; the halo is left NaN, so no candidate can arise there.
    q = np.asarray(part)
    cand_ds = np.zeros((ny, nx), dtype=bool)
    cand_ds[y0:y1, x0:x1] = np.isfinite(q) & (q >= float(kw['threshold']))
    if len(_oy):
        cand_ds[_oy, _ox] = False
    # the fixed layer: every table, kept to within one DS window of this block
    _acc = {k: [] for k in ('iy', 'ix', 'vel', 'hgt', 'sea', 'label', 'gamma',
                            'level')}
    # THE PS NETWORK IS A PARTNER POOL TOO -- the best anchors a candidate can
    # have, level 0 -- not only an exclusion mask
    if net is not None and len(np.asarray(net['iy'])):
        _gy0 = np.asarray(net['iy']) - oy0
        _gx0 = np.asarray(net['ix']) - ox0
        _m0 = (_gy0 >= 0) & (_gy0 < ny) & (_gx0 >= 0) & (_gx0 < nx)
        _m0 &= (np.isfinite(np.asarray(net['vel'], dtype=float))
                & np.isfinite(np.asarray(net['hgt'], dtype=float)))
        if _m0.any():
            _acc['iy'].append(_gy0[_m0]); _acc['ix'].append(_gx0[_m0])
            for k in ('vel', 'hgt', 'sea', 'label'):
                _acc[k].append(np.asarray(net[k])[_m0])
            _acc['gamma'].append(np.asarray(net['gamma'], dtype=float)[_m0]
                                 if net.get('gamma') is not None
                                 else np.zeros(int(_m0.sum())))
            _acc['level'].append(np.zeros(int(_m0.sum()), np.int16))
    # a level arrives as ONE entry holding its blocks' tables, because the
    # level's reducer sits between the levels; older callers pass them flat
    _flat = []
    for _t in tables:
        if isinstance(_t, (list, tuple)):
            _flat.extend(x for x in _t if x is not None)
        elif _t is not None:
            _flat.append(_t)
    for _tb in _flat:
        if _tb is None or not len(_tb['iy']):
            continue
        gy, gx = np.asarray(_tb['iy']) - oy0, np.asarray(_tb['ix']) - ox0
        m = (gy >= 0) & (gy < ny) & (gx >= 0) & (gx < nx)
        # FINITE ROWS ONLY. A consensus winner the anchor gate refused has no
        # value: as a fixed row it cannot vote or anchor anything, and marking
        # its pixel done would bar it from THIS level -- the level that exists
        # to retry what the previous one could not hold.
        m &= (np.isfinite(np.asarray(_tb['vel'], dtype=float))
              & np.isfinite(np.asarray(_tb['hgt'], dtype=float)))
        if not m.any():
            continue
        _acc['iy'].append(gy[m]); _acc['ix'].append(gx[m])
        for k in ('vel', 'hgt', 'sea', 'label'):
            _acc[k].append(np.asarray(_tb[k])[m])
        _acc['gamma'].append(np.asarray(_tb['gamma'])[m]
                             if _tb.get('gamma') is not None
                             else np.zeros(int(m.sum())))
        _acc['level'].append(np.asarray(_tb['level'])[m].astype(np.int16)
                             if _tb.get('level') is not None
                             else np.ones(int(m.sum()), np.int16))
    if not _acc['iy']:
        # NOTHING IN REACH, AND STILL TWO VALUES. `nout` is fixed when the
        # graph is built, so an early return that hands back one array makes
        # the caller's second slot a RASTER ROW instead of a node table, and
        # the next level reads `_tb['iy']` off it. A block with no fixed layer
        # near it is ordinary -- it just adds nothing.
        _none = dict(iy=np.zeros(0, int), ix=np.zeros(0, int),
                     vel=np.zeros(0, float), hgt=np.zeros(0, float),
                     sea=np.zeros(0, np.complex64),
                     label=np.zeros(0, np.int8), gamma=np.zeros(0, float),
                     level=np.zeros(0, np.int16), _stats={})
        return (np.asarray(planes), _none) if emit_nodes \
            else np.asarray(planes)
    ds_nodes = {k: np.concatenate(v) for k, v in _acc.items()}
    t, ele2phase, meter2rad, _car = utils_arcs._3d_fit_frame(
        kw['date_values'], kw['bperp'], kw['geometry'], n)
    _ma = utils_arcs._3d_consensus(kw['consensus'])
    _ii = max(1, int(kw['iterations']))
    _err_h = float(kw.get('err_dh', 5.0)) * meter2rad
    _err_v = float(kw.get('err_dv', 1.0)) * meter2rad / 1e3
    # level 1's planes cover the owned rectangle; lift them into the haloed
    # frame so this stage writes beside them, then hand back the owned part
    P = np.asarray(planes)
    lab_out = np.full((ny, nx), -1, dtype=np.int8)
    vel_out = np.full((ny, nx), np.nan, dtype=np.float32)
    hgt_out = np.full((ny, nx), np.nan, dtype=np.float32)
    sea_out = np.full((ny, nx), np.nan + 1j * np.nan, dtype=np.complex64)
    coh_out = np.full((ny, nx), np.nan, dtype=np.float32)
    lvl_out = np.full((ny, nx), -1, dtype=np.int8)
    lab_out[y0:y1, x0:x1] = P[0].real.astype(np.int8)
    vel_out[y0:y1, x0:x1] = (-P[1].real).astype(np.float32)   # stored negated
    hgt_out[y0:y1, x0:x1] = P[2].real.astype(np.float32)
    sea_out[y0:y1, x0:x1] = (-P[3]).astype(np.complex64)
    coh_out[y0:y1, x0:x1] = P[4].real.astype(np.float32)
    lvl_out[y0:y1, x0:x1] = P[5].real.astype(np.int8)
    # A CLEAN DICT FOR THIS LEVEL. Seeding from the thread's stats carries
    # the PREVIOUS level's `lvl_*` keys into this level's report -- level 2
    # was printing level 1's rejection counts and gamma. Nothing in
    # `_3d_ds_attach` needs a pre-existing key it does not write itself.
    _st = {'ds_attached': int(len(ds_nodes['iy']))}
    utils_arcs._fit_stats.reset(_st)
    utils_arcs._3d_ds_attach(
        S, cand_ds, ds_nodes, _oy, _ox, lab_out, vel_out, hgt_out, sea_out,
        coh_out, lvl_out, int(level_id), ele2phase, t, meter2rad,
        ny=ny, nx=nx, wy=wy, wx=wx,
        cell=tuple(kw['cell']), budget=kw['budget'],
        threshold=float(kw['threshold']), level=2,
        max_dh=kw['max_dh'], max_dv=kw['max_dv'], step_dh=kw['step_dh'],
        step_dv=kw['step_dv'], iterations=int(kw['iterations']),
        _ma=_ma, _ii=_ii, _err_h=_err_h, _err_v=_err_v,
        _nth=max(1, int(threads) if threads else 1), _st=_st,
        spacing=kw['spacing'], debug=bool(kw['debug']))
    out = np.concatenate(
        [lab_out[None, y0:y1, x0:x1].astype(np.complex64),
         (-vel_out)[None, y0:y1, x0:x1].astype(np.complex64),
         hgt_out[None, y0:y1, x0:x1].astype(np.complex64),
         (-sea_out)[None, y0:y1, x0:x1].astype(np.complex64),
         coh_out[None, y0:y1, x0:x1].astype(np.complex64),
         lvl_out[None, y0:y1, x0:x1].astype(np.complex64)], axis=0)
    if not emit_nodes:
        return out
    # WHAT THIS LEVEL ADDED, for the next one to stand on. Found by asking
    # which owned pixels became finite, so it does not depend on the kernel's
    # bookkeeping, and returned in SCENE coordinates because the next level's
    # chunks each want a different part of it.
    _was = np.isfinite(np.asarray(planes)[1].real)
    _now = np.isfinite(vel_out[y0:y1, x0:x1])
    _ny2, _nx2 = np.nonzero(_now & ~_was)
    _gy, _gx = _ny2 + y0, _nx2 + x0
    _new = dict(iy=_gy + oy0, ix=_gx + ox0,
                vel=vel_out[_gy, _gx].astype(float),
                hgt=hgt_out[_gy, _gx].astype(float),
                sea=sea_out[_gy, _gx],
                label=lab_out[_gy, _gx],
                gamma=coh_out[_gy, _gx].astype(float),
                level=lvl_out[_gy, _gx].astype(np.int16))
    # this block's contribution to the LEVEL's report, carried on the table
    # that already travels to the caller rather than on a second channel
    _new['_stats'] = {k: v for k, v in _st.items() if k.startswith('lvl_')}
    return out, _new

def _fit3d_model(ds, da_xr, both, date_values, bp):
    """The six planes of one burst as the model dataset, variables named by
    QUANTITY alone. `both` is (6, y, x) complex64: label, velocity, height,
    seasonal, coherence, level -- the planes every stage writes.

    rmse is sqrt(-2 ln gamma), derived here rather than carried as a seventh
    plane: the two are exact inverses, so shipping both through the graph
    would move the same information twice. The `date` coordinate is the
    MASTER, where B_perp is smallest -- where the fit zeroes its t -- so
    predict() reads the origin instead of reconstructing it.
    """
    import numpy as np
    import xarray as xr
    import dask.array as da
    lb = both[0].real.astype(np.int8)
    vv = both[1].real.astype(np.float32)
    hh_ = both[2].real.astype(np.float32)
    sa_ = both[3].astype(np.complex64)
    cg_ = both[4].real.astype(np.float32)
    lv_ = both[5].real.astype(np.int8)
    rr = da.sqrt(da.maximum(
        -2.0 * da.log(da.clip(cg_, 1e-9, 1.0)), 0.0)).astype(np.float32)
    coords = {k_: v for k_, v in da_xr.coords.items()
              if k_ in ('y', 'x', 'spatial_ref')}
    mvars = {}
    for nm_, arr_ in (('velocity', vv), ('height', hh_),
                      ('seasonal', sa_), ('coherence', cg_),
                      ('rmse', rr), ('conncomp', lb),
                      ('level', lv_)):
        mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
    mds = xr.Dataset(mvars, attrs=ds.attrs)
    _dd = (np.asarray(date_values).astype('datetime64[D]')
           .astype(np.float64))
    _b3 = (np.zeros_like(_dd) if bp is None
           else np.asarray(bp, float).ravel())
    if _b3.shape != _dd.shape:
        _b3 = np.zeros_like(_dd)
    mds = mds.assign_coords(
        date=np.datetime64(int(_dd[int(np.argmin(np.abs(_b3)))]), 'D'))
    if 'spatial_ref' in ds.coords:
        mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
    return mds


def _apply_goldstein_2d_for_dask(phase_block, corr_block, psize=32, threshold=0.5, device='cpu'):
    """Module-level function for Goldstein filter map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles both 2D (y, x) and 3D (1, y, x) blocks - _goldstein() handles squeeze/unsqueeze.

    Parameters
    ----------
    phase_block : np.ndarray
        Complex array from dask, shape (y, x) or (1, y, x)
    corr_block : np.ndarray
        Real array from dask, shape (y, x) or (1, y, x)
    psize : int or dict
        Patch size for the filter
    threshold : float
        Minimum fraction of valid pixels
    device : str
        PyTorch device

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input
    """
    # _goldstein handles (1, y, x) -> squeeze -> process -> unsqueeze
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize,
                                   threshold=threshold, device=device)


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
                # and the old `len(dims) == 1` test dropped them, taking
                # `incidence`, `d_drho_dh`, `baseline_model` and the orbit
                # polynomials out of the transform view that exists to carry them
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
        used for correlation in [0,1] range
        """
        return BatchUnit(super().clip(min=min, max=max, **kwargs))

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
        >>> model = stack.fit3d()
        >>> # modelled displacement per date, unwrapped
        >>> disp  = stack.predict(model=model).displacement_los(stack.transform())
        >>> # remove the modelled ground motion, keeping topography
        >>> ground = stack * stack.predict(model=model).iexp(-1)
        >>> # remove the whole model, topography included
        >>> resid  = stack * stack.predict(model=model, baseline='BPR').iexp(-1)
        """
        import numpy as np
        import xarray as xr
        import dask
        from insardev_toolkit import progressbar

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
        progressbar([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

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
    def _solve_spd(A, b, eps=1e-12):
        """Batched small symmetric-positive-definite solve, elementwise Cholesky.

        A: (n, k, k), b: (n, k) -> (n, k). Singular systems yield NaN rather than a
        plausible wrong value; those pixels are masked by valid_count downstream.
        """
        import torch
        n_cols = A.shape[-1]
        L = [[None] * n_cols for _ in range(n_cols)]
        bad = None
        for i in range(n_cols):
            for j in range(i + 1):
                acc = A[:, i, j].clone()
                for k in range(j):
                    acc = acc - L[i][k] * L[j][k]
                if i == j:
                    neg = acc <= eps
                    bad = neg if bad is None else (bad | neg)
                    L[i][j] = torch.sqrt(acc.clamp_min(eps))
                else:
                    L[i][j] = acc / L[j][j]
        y = []
        for i in range(n_cols):
            acc = b[:, i].clone()
            for k in range(i):
                acc = acc - L[i][k] * y[k]
            y.append(acc / L[i][i])
        x = [None] * n_cols
        for i in reversed(range(n_cols)):
            acc = y[i].clone()
            for k in range(i + 1, n_cols):
                acc = acc - L[k][i] * x[k]
            x[i] = acc / L[i][i]
        out = torch.stack(x, dim=1)
        if bad is not None:
            out = torch.where(bad.unsqueeze(1), torch.full_like(out, float('nan')), out)
        return out

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

    @staticmethod
    @serialize_gpu
    def _fit1d_pairs_torch(data, A, weight=None, device='auto',
                           dh_prior=None, dh_col=None):
        """Weighted least squares of a per-pixel model on per-PAIR unwrapped phase.

        A is (n_pairs, k) and IDENTICAL for every pixel -- only the weights vary
        with the data -- so the normal equations are two matrix products rather
        than a loop:

            AtWA = (outer(a_p, a_p) flattened)^T @ w      (k*k, N)
            AtWy = A^T @ (w * y)                         (k, N)

        and the whole fit is those two GEMMs plus a batched k x k solve. There is
        no lattice and no refinement: unwrapped phase makes the objective convex,
        which is the entire reason this is cheaper than the wrapped route it
        replaces.

        NO 2 PI AMBIGUITY REFIT. Removing integer cycles by
        round(residual / 2 pi) is exact when the residual is dominated by an
        unwrapping error and destructive when it is dominated by noise, and
        which one holds is decided by the coherence of the stack, not by the
        estimator. At low coherence it chases noise past pi rather than cycles.
        Unwrapping errors belong to the unwrapper and to phase closure over
        triplets, which sees them without a model in the way.

        Columns are scaled to unit norm before the solve and the solution scaled
        back: dt is O(1) years while ele2phase is O(1e-4), and a Cholesky on the
        raw normal equations of two columns eight orders of magnitude apart is
        not a solve, it is a coin toss.

        Returns (theta (k, N), gamma (N,), rmse (N,)).
        """
        import torch
        import numpy as np

        dev = Batch._get_torch_device(device)
        shape = data.shape
        n_p = shape[0]
        n_pix = int(np.prod(shape[1:]))
        y = torch.from_numpy(np.ascontiguousarray(
            data.reshape(n_p, -1), dtype=np.float32)).to(dev)
        Am = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).to(dev)
        k = int(Am.shape[1])

        nan_mask = torch.isnan(y)
        valid_count = (~nan_mask).sum(dim=0)
        if weight is None:
            w = (~nan_mask).float()
        else:
            g = np.asarray(weight, np.float32)
            if g.ndim == 3:
                g = g.reshape(g.shape[0], -1)
            g = torch.from_numpy(np.ascontiguousarray(g)).to(dev)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 0.999)
            # gamma^2, BOUNDED. The Cramer-Rao weight gamma^2/(1-gamma^2) is
            # the right one for a well-observed pixel and the wrong one here: it
            # is unbounded, so at low correlation it spans orders of magnitude
            # within a pixel and the rate is decided by that pixel's single best
            # pair. Clamping does not fix it, because the damage is the RATIO
            # between pairs, not the maximum. Weighting earns its keep where
            # correlation is high and uniform; where it is low, leaving it out
            # does better, which is why the default is None rather than the
            # correlation the caller happens to have.
            w = (~nan_mask).float() * (g * g)
        y = torch.where(nan_mask, torch.zeros_like(y), y)
        finite = (~nan_mask).float()

        scale = Am.norm(dim=0).clamp(min=1e-30)
        An = Am / scale
        # A PRIOR ON THE HEIGHT, not a robust loss. Adding the DEM column cost
        # held-out accuracy at EVERY quantile alike, which is variance, not
        # outliers. L1-IRLS and Huber were tried and both came back worse,
        # confirming there is no tail to trim.
        #
        # What a poorly-observed nuisance parameter needs is shrinkage. DEM
        # error is bounded by the DEM, not by this stack, so sigma_dh =
        # max_dh/3 is a property of the terrain model rather than a number to
        # tune, and
        #
        #     lambda = sigma_phi^2 / (meter2rad * sigma_dh)^2
        #
        # is the Bayesian weight it implies. It adapts by construction: where
        # the baselines resolve the height the prior is inert, and where they do
        # not it shrinks the column away.
        T = (An[:, :, None] * An[:, None, :]).reshape(n_p, k * k)
        # ALL PAIRS OR NOTHING. A pixel the unwrapper dropped from some pairs
        # is a degraded network, not a smaller one: the surviving subset is
        # whatever happened to stay coherent, so the model is fitted to a
        # different experiment at every such pixel and the result is not
        # comparable with its neighbours.
        #
        # The matrix is still tested below: linalg.solve raises on the WHOLE
        # batch when one element is singular, and a full pair set can still be
        # singular if every weight is zero.
        solvable = (valid_count == n_p) & (w.sum(dim=0) > 0)
        eye = torch.eye(k, device=dev, dtype=torch.float32).expand(n_pix, k, k)

        # WHICH column carries the height. It is index -1 only when the
        # seasonal is off; with the annual in the model the last column is its
        # sine, and shrinking that instead would leave the height untouched.
        _hc = k - 1 if dh_col is None else int(dh_col)
        _use_prior = bool(dh_prior) and 0 <= _hc < k
        lam_h = None
        # L1 / IRLS, the same reweighting lstsq() applies to the same input:
        # unwrapping error arrives as a whole cycle on one pair, which least
        # squares spreads over every parameter.
        w0 = w
        n_irls = 5
        eps_irls = 0.1
        for _pass in range(2 if _use_prior else 1):
          for _it in range(n_irls):
            AtWA = (T.T @ w).T.reshape(n_pix, k, k)
            AtWy = (An.T @ (w * y)).T
            if lam_h is not None:
                AtWA = AtWA.clone()
                AtWA[:, _hc, _hc] = AtWA[:, _hc, _hc] + lam_h
            AtWA_safe = torch.where(solvable[:, None, None], AtWA, eye)
            ok = solvable
            if dev.type == 'mps':
                # _solve_spd returns NaN for a singular element instead of
                # raising, so it needs no pre-test
                th = Batch._solve_spd(AtWA_safe, AtWy)
            else:
                # cholesky_ex REPORTS failure in `info` rather than raising, and
                # succeeding is exactly the property linalg.solve requires here,
                # so the un-solvable elements are swapped for the identity
                # before the solver ever sees them
                _, info = torch.linalg.cholesky_ex(AtWA_safe)
                ok = solvable & (info == 0)
                AtWA_safe = torch.where(ok[:, None, None], AtWA_safe, eye)
                th = torch.linalg.solve(AtWA_safe, AtWy.unsqueeze(-1)).squeeze(-1)
            ok = ok & torch.isfinite(th).all(dim=1)
            th = torch.where(ok[:, None], th, torch.full_like(th, 0.0))
            resid = (y - An @ th.T) * finite
            if _it + 1 < n_irls:
                # reweight and go round again; the last pass keeps the weights it
                # converged with so `resid` below is the one the model reports
                w = w0 / torch.clamp(torch.abs(resid), min=eps_irls)
                w = torch.where(finite > 0, w, torch.zeros_like(w))
            if _use_prior and _pass == 0:
                # sigma_phi per pixel from the unregularised pass, then the
                # Bayesian weight the height prior implies. The prior lives on
                # the NORMALISED column, so it means the same thing whatever
                # the baselines span.
                dof = (valid_count - k).clamp(min=1).float()
                s2 = (resid ** 2).sum(dim=0) / dof
                # An = A/s and theta_n = s*theta, so a penalty lambda on the
                # UNNORMALISED height becomes lambda/s^2 on the normalised one
                lam_h = s2 / float(dh_prior) ** 2 / (scale[_hc] ** 2)
                w = w0          # second pass re-runs IRLS from the base weights

        # LINEAR RESIDUALS, not circular. gamma used to be the resultant
        # |sum w e^{i r}| / sum w, which is right for wrapped phase and wrong
        # here: this phase is UNWRAPPED, so a residual of 2 pi is a whole cycle
        # of error and cos(2 pi) = 1 counts it as a perfect sample. With cycle
        # slips present it reports near-perfect agreement while the true
        # residual RMS is large.
        #
        # Unwrapping error is the dominant error mode in multilooked pairs, so
        # a quality number that cannot see it is worse than none: it certifies
        # exactly the pixels a caller most needs to drop. rmse is now the
        # weighted RMS of the residual itself, inflated by n/(n-k) for the
        # parameters spent, and coherence is its exact inverse transform
        # exp(-rmse^2/2) -- the same relation fit3d's pair satisfies, so the
        # two remain comparable while both now respond to a cycle slip.
        # The estimate is robust; the diagnostic must not be. rmse/coherence are
        # computed from the BASE weights w0, never the IRLS weights: those were
        # chosen to suppress the samples that fit worst, so scoring against them
        # would report the fit the reweighting engineered.
        wsum = w0.sum(dim=0)
        infl = valid_count.float() / (valid_count - k).clamp(min=1).float()
        mse = (w0 * resid * resid).sum(dim=0) / wsum.clamp(min=1e-12)
        rmse = torch.sqrt(torch.clamp(mse, min=0.0) * infl)
        gam = torch.exp(-0.5 * rmse * rmse).clamp(0.0, 1.0)

        # A box bound cannot help a convex fit -- it can only distort it. On
        # unwrapped phase the optimum is unique, so a constrained answer is a
        # boundary point and the residual is forced into the other parameters.
        # The bound therefore REJECTS here (see the caller).

        nanv = torch.full_like(gam, float('nan'))
        gam = torch.where(ok, gam, nanv)
        rmse = torch.where(ok, rmse, nanv)
        th = torch.where(ok[:, None], th / scale[None, :],
                         torch.full_like(th, float('nan')))

        out = (th.T.cpu().numpy(), gam.cpu().numpy(), rmse.cpu().numpy())
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()
        return out

    def trend2d(self, transform: 'BatchCore | None' = None, weight: 'BatchUnit | None' = None,
                degree: int = 1, device: str = 'auto', detrend: bool = False,
                extrapolate: bool = False, debug: bool = False) -> 'BatchCore':
        """
        Compute 2D polynomial trend (ramp) from data.

        Two modes:
        - Complex (BatchComplex): unit-circle fitting, returns BatchComplex
        - Real (Batch): standard polynomial, returns Batch

        Parameters
        ----------
        transform : BatchCore or None
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
            If None, uses y,x grid coordinates as regressors.
        weight : BatchUnit or None
            Optional weight for the fitting (typically correlation).
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
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
        >>> # With radar coordinates
        >>> trend = phase.trend2d(stack.transform(), weight=corr)
        >>> # With y,x grid coordinates (no transform needed)
        >>> trend = phase.trend2d(weight=corr)
        >>> # Complex interferogram
        >>> trend = intf_complex.trend2d(stack.transform(), weight=corr)
        >>> detrended = intf_complex * trend.conj()
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from . import utils_detrend
        from .Batch import Batch, BatchComplex

        phase = self

        # Validate lazy data
        BatchCore._require_lazy(phase, 'trend2d')

        # Auto-detect device
        resolved = BatchCore._get_torch_device(device, debug=debug)
        device = resolved.type

        if debug:
            print(f"DEBUG: using device={device}")

        if device == 'mps' and degree >= 3:
            print(f"NOTE: MPS has float32 precision issues for degree>={degree}. Use device='cpu' for better accuracy.")

        is_complex = isinstance(phase, BatchComplex)

        # Unify transform keys to phase
        if transform is not None:
            transform = transform.sel(phase)

        result = {}
        for key in phase.keys():
            ds = phase[key]

            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            phase_da_ref = ds[pols[0]]
            phase_shape = phase_da_ref.shape[-2:]
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
                weight_ds = weight[key]
                weight_pols = [v for v in weight_ds.data_vars
                              if 'y' in weight_ds[v].dims and 'x' in weight_ds[v].dims]
                if weight_pols:
                    weight_da_ref = weight_ds[weight_pols[0]]
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
                phase_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

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

                # Phase 0: Compute global feature standardization (pair-independent)
                feature_mean, feature_std = utils_detrend._compute_feature_stats(
                    var_dask_list, degree)
                n_poly = len(feature_mean)
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

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        if is_complex:
            return BatchComplex(result)
        return Batch(result)

    def fit1d(self, weight=None, baseline: str = 'BPR',
              max_dh: float = 30.0,
              max_seasonal: float = 0.0, device: str = 'auto',
              debug: bool = False) -> 'Batch':
        """
        Full per-pixel model on UNWRAPPED per-PAIR phase -- NO network.

        The unwrapped twin of BatchComplex.fit1d(): the same model and the same
        output, fitted by weighted least squares instead of by coherence
        maximisation, because unwrapped phase makes the objective convex. Returns
        the MODEL ONLY, named and scaled as fit3d() names and scales it, so
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
            gamma^2/(1-gamma^2), the phase-precision weight.
        baseline : str
            Per-pair perpendicular baseline coordinate, default 'BPR'. Without
            it the height column is dropped and `height` comes back NaN -- which
            is only right when the topographic phase is already removed.
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
            The model, named and scaled exactly as fit3d() and
            BatchComplex.fit1d():

              `velocity`   rad/yr
              `height`     rad per unit ele2phase, NaN with no baseline
              `seasonal`   complex rad, 0 when max_seasonal=0
              `coherence`  |sum w exp(i r)| / sum w about the reported model
              `rmse`       sqrt(-2 ln coherence), radians, n/(n-k) inflated

            plus a scalar `date` coordinate: the epoch the model is
            referenced to, which is what every fit here carries so predict()
            never has to guess. A pairs batch holds only baseline DIFFERENCES,
            so the master that fit3d() anchors on is not recoverable from it and
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

        BatchCore._require_lazy(self, 'fit1d')
        model_result = {}
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
                if baseline in src:
                    bp = np.asarray(src[baseline].values, dtype=float).ravel()
                    break
            e2p = None
            if bp is not None and bp.shape == dt.shape:
                # FROM THIS BATCH. A transform= argument used to select the
                # source of this scalar; every step of the pair pipeline carries
                # the metadata it reads, so it never supplied anything missing,
                # and the value only rescales `height` inversely and cancels in
                # the product.
                _fac = Batch._elevation_phase_approximate(self)[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)
            if e2p is None:
                print(f"fit1d(): no {baseline!r} for '{key}' -- the height "
                      "column is dropped and the topographic phase stays in the "
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
                w_dask = weight[key][pol].transpose('pair', ...).chunk(
                    {'pair': -1, 'y': cy, 'x': cx}).data
                if w_dask.chunks != d_dask.chunks:
                    w_dask = w_dask.rechunk(d_dask.chunks)

            # max_dh is a 3-sigma bound on a DEM error, so it also names the
            # PRIOR the height is shrunk under. One number, both jobs: what the
            # caller will accept, and what the terrain model is known to be.
            _dh_prior = meter2rad * float(max_dh) / 3.0 if has_h else None

            def _blk(b, wb=None, _A=A, _k=k, _hh=has_h, _se=fit_seasonal,
                     _dev=device, _dbg=bool(debug), _m2r=meter2rad,
                     _mh=float(max_dh),
                     _ms=float(max_seasonal), _dp=_dh_prior):
                shp = b.shape[1:]
                th, gam, rms = Batch._fit1d_pairs_torch(
                    b, _A, weight=wb, device=_dev, dh_prior=_dp,
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
                    # value rather than an absence -- the same convention fit3d
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
            Output of fit1d() or fit3d(): `velocity`, `height`, `seasonal`.
        baseline : str
            Per-pair or per-date perpendicular baseline, default 'BPR'. None,
            or absent, drops the height term.

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

        out = {}
        for key in self.keys():
            ds = self[key]
            if key not in model:
                raise KeyError(f"predict(): the model has no burst '{key}'.")
            mds = model[key]
            missing = [v for v in ('velocity', 'height', 'seasonal') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit1d() or fit3d().")

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
            e2p = None
            n_obs = da_.sizes['pair' if per_pair else 'date']
            if bp is not None and bp.shape == (n_obs,):
                _fac = Batch._elevation_phase_approximate(self)[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)

            import dask.array as da
            vel = mds['velocity'].data
            hgt = mds['height'].data
            sea = mds['seasonal'].data
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
        Per-pair displacement from a fit1d()/fit3d() model MIXED with the raw signal.

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
            Output of fit1d() or fit3d(): `velocity`, `height`, `seasonal`.
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

        `4 pi / (lambda R sin(incidence))`, per pixel. Multiply by B_perp and a
        height to get phase; divide a phase by it and B_perp to get a height.
        One definition of the geometry, so the forward and inverse conversions
        cannot drift apart.

        SLANT RANGE AND THE SINE OF THE INCIDENCE ANGLE, matching GMTSAR.
        `sbas.c:186` computes `scale = 4 pi / wl / rng / sin(theta)` with theta
        documented as "incidence angle of the radar wave", and `phase2topo.c`
        returns `topo = res * rho * c * sint / ret` where `c = re + height` is
        the satellite distance from the Earth CENTRE -- by the law of sines
        `c*sint/ret` is sin(incidence), so that is `res * rho * sin(incidence)`,
        the same relation inverted, and a numerical check agrees. Every `cos()`
        in GMTSAR's geometry is
        baseline projection (`bperp.c:151`, `B*cos(theta-alpha)`) or a look-
        vector rotation, never elevation from phase.

        This exists because `elevation()` inlined `SC_height * cos(incidence)`:
        the satellite HEIGHT (702 227 m here) where the slant range
        (856 485 - 860 637 m) belongs, and the cosine where the sine belongs.
        The errors nearly cancel -- 0.818 * 1.306 = 1.068 -- so the result read
        as plausible while biasing every height by +6.8%.

        Call on a transform batch (`stack.transform()`), which carries `rng`,
        `near_range`, `rng_samp_rate` and what `incidence()` needs.
        """
        import numpy as np

        c_light = 299792458.0
        inc_batch = self.incidence()
        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            wavelength = tfm['radar_wavelength']
            wavelength = float(np.asarray(wavelength.values).ravel()[0]) if hasattr(wavelength, 'values') else float(wavelength)
            # .values, NOT .item(): a Stack is lazy, and .item() raises
            # NotImplementedError on a dask array -- which made the pixelwise
            # geometry unreachable from any normal Stack while the scene-centre
            # approximation, which reads .values, always worked. Same idiom as
            # _elevation_phase_approximate() so the two cannot drift.
            near_range = float(np.asarray(tfm['near_range'].values).mean())
            rng_samp_rate = float(np.asarray(tfm['rng_samp_rate'].values).mean())
            slant_range = near_range + tfm['rng'] * (c_light / (2.0 * rng_samp_rate))
            incidence = inc_batch[key]['incidence']
            fac = (4.0 * np.pi / wavelength) / (slant_range * xr.ufuncs.sin(incidence))
            res = xr.Dataset({'elevation_phase': fac.astype('float32')})
            res.attrs = tfm.attrs
            if 'spatial_ref' in tfm.coords:
                res = res.assign_coords(spatial_ref=tfm.spatial_ref)
            out[key] = res
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
        by construction -- (0.5, 0.5) to (num_lines-0.5, num_rng_bins-0.5).
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

        c_light = 299792458.0
        out: dict = {}
        for key, ds in self.items():
            g = lambda n: float(np.asarray(ds[n].values).ravel().mean())
            wavelength = g('radar_wavelength')
            near_range = g('near_range')
            rng_samp_rate = g('rng_samp_rate')
            earth_radius = g('earth_radius')
            num_lines = g('num_lines')
            num_rng_bins = g('num_rng_bins')
            sc_height = 0.5 * (g('SC_height_start') + g('SC_height_end'))

            rng_c = num_rng_bins / 2.0
            try:
                from shapely import wkt as _wkt
                from pyproj import Transformer as _Tr
                poly = _wkt.loads(str(np.asarray(ds['geometry'].values).ravel()[0]))
                lon, lat = (np.asarray(v) for v in poly.exterior.coords.xy)
                crs = ds.rio.crs if hasattr(ds, 'rio') else None
                px, py = _Tr.from_crs('EPSG:4326', crs, always_xy=True).transform(
                    lon[:4], lat[:4])
                a_c = np.array([0.5, 0.5, num_lines - 0.5, num_lines - 0.5])
                r_c = np.array([0.5, num_rng_bins - 0.5,
                                num_rng_bins - 0.5, 0.5])
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
            except Exception:
                # no `geometry`, no CRS, or a degenerate polygon: the burst
                # centre still answers, just less well on a cropped stack
                pass

            slant_range = near_range + rng_c * (c_light / (2.0 * rng_samp_rate))
            ground_dist = earth_radius + (
                float(np.asarray(ds['ref_height'].values).ravel().mean())
                if 'ref_height' in ds else 0.0)
            sat_dist = earth_radius + sc_height
            cos_earth = np.clip(
                (ground_dist ** 2 + sat_dist ** 2 - slant_range ** 2)
                / (2.0 * ground_dist * sat_dist), -1.0, 1.0)
            sin_inc = np.clip(
                sat_dist * np.sin(np.arccos(cos_earth)) / slant_range, -1.0, 1.0)
            out[key] = (4.0 * np.pi / wavelength) / (slant_range * sin_inc)
        return out

    def incidence(self) -> "Batch":
        """Compute incidence angle from azi, rng, ele, and radar geometry parameters.

        Uses spherical Earth geometry with per-pixel satellite height interpolation
        and terrain elevation correction. Matches GMTSAR look vector results within ~0.07%.

        Required vars: azi, rng, ele, near_range, SC_height_start, SC_height_end,
                       earth_radius, rng_samp_rate, num_lines
        """
        import numpy as np
        import rioxarray  # for .rio accessor

        c = 299792458.0  # speed of light

        # Get CRS from input batch
        crs = self.crs

        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            # Get scalar parameters (mean if per-date)
            # .values, NOT .item() -- see elevation_phase(): a Stack is lazy and
            # .item() raises on a dask array, which put incidence() and everything
            # built on it out of reach of any stack that had not been computed.
            _g = lambda n: float(np.asarray(tfm[n].values).mean())
            near_range = _g('near_range')
            SC_height_start = _g('SC_height_start')
            SC_height_end = _g('SC_height_end')
            earth_radius = _g('earth_radius')
            rng_samp_rate = _g('rng_samp_rate')
            num_lines = _g('num_lines')

            # Get per-pixel coordinates
            azi = tfm['azi']
            rng = tfm['rng']
            ele = tfm['ele']

            # Compute slant range to the actual elevated ground point
            range_pixel_size = c / (2 * rng_samp_rate)
            slant_range = near_range + rng * range_pixel_size

            # Interpolate satellite height based on azimuth position
            SC_height = SC_height_start + (SC_height_end - SC_height_start) * azi / (num_lines - 1)

            # Ground at earth_radius + ele from Earth center
            ground_dist = earth_radius + ele

            # Satellite at earth_radius + SC_height from Earth center
            sat_dist = earth_radius + SC_height

            # Spherical Earth geometry: law of cosines + law of sines
            cos_earth = (ground_dist**2 + sat_dist**2 - slant_range**2) / (2 * ground_dist * sat_dist)
            cos_earth = xr.where(cos_earth > 1, 1, xr.where(cos_earth < -1, -1, cos_earth))
            earth_angle = xr.ufuncs.arccos(cos_earth)
            sin_inc = sat_dist * xr.ufuncs.sin(earth_angle) / slant_range
            sin_inc = xr.where(sin_inc > 1, 1, xr.where(sin_inc < -1, -1, sin_inc))
            incidence = xr.ufuncs.arcsin(sin_inc).astype('float32')

            result_ds = xr.Dataset({"incidence": incidence})
            result_ds.attrs = tfm.attrs
            # Preserve CRS
            if crs is not None:
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
            for var_name, data in phase_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    disp_vars[var_name] = data
                    continue
                disp = (data * scale).astype('float32')
                disp_vars[var_name] = disp
            out[key] = xr.Dataset(disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, transform: 'Batch | Stack', func, suffix: str = '') -> 'Batch':
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        # Decimate transform to match phase resolution for efficiency
        transform = Batch({k: transform[k].reindex(y=self[k].y, x=self[k].x, method='nearest')
                           for k in self.keys() if k in transform})

        los_batch = self.displacement_los(transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    comp_vars[var_name] = data
                    continue
                # align incidence to data grid
                incidence = inc_da.reindex_like(data, method='nearest')
                # the geometry is bent to the data's chunks, never the other way round
                if data.chunks is not None and incidence.chunks is not None:
                    _ch = tuple(data.chunks[-2:][('y', 'x').index(a)] for a in incidence.dims)
                    if incidence.chunks != _ch:
                        incidence = incidence.chunk(dict(zip(incidence.dims, _ch)))

                comp = (data / func(incidence)).astype('float32')

                if len(los_ds.data_vars) == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            out[key] = xr.Dataset(comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        return Batch(out)

    def displacement_vertical(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute vertical displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

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
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

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

    def elevation(self, transform: 'Batch | Stack', baseline: float | None = None) -> 'Batch':
        """Compute elevation (meters) from unwrapped phase grids.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch containing look vectors for incidence calculation,
            or a Stack (will call .transform() internally).
        baseline : float | None, optional
            Perpendicular baseline in meters. If None, uses burst-specific BPR
            from phase coordinates.

        Returns
        -------
        Batch
            Elevation grids as float32 datasets.

        Examples
        --------
        >>> elev = unwrapped.elevation(stack)
        >>> elev = unwrapped.elevation(stack.transform())
        """
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        ep_batch = transform.elevation_phase()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in self.items():
            if key not in ep_batch:
                raise KeyError(f'Missing geometry for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            if wavelength is None:
                raise KeyError(f"Missing radar_wavelength in transform for burst {key}")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            # ONE geometry, from elevation_phase(): radians per metre of height
            # per metre of baseline. This block used to inline
            # `SC_height * cos(incidence)` -- the satellite height where the
            # SLANT RANGE belongs and the cosine where the SINE belongs. The two
            # errors nearly cancel (0.818 * 1.306 = 1.068), so the answer looked
            # reasonable while every height came out 6.8% high. GMTSAR's
            # phase2topo.c and sbas.c both use slant range and sin(incidence);
            # see elevation_phase() for the references and the numbers.
            fac_da = ep_batch[key]['elevation_phase']

            ref_height = _scalar_from_ds(tfm, 'ref_height') or 0.0

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    elev_vars[var_name] = data
                    continue
                fac = fac_da.reindex_like(data, method='nearest')
                # the geometry is bent to the data's chunks, never the other way round
                if data.chunks is not None and fac.chunks is not None:
                    _ch = tuple(data.chunks[-2:][('y', 'x').index(a)] for a in fac.dims)
                    if fac.chunks != _ch:
                        fac = fac.chunk(dict(zip(fac.dims, _ch)))

                # phi = fac * B_perp * dh  ->  dh = phi / (fac * B_perp)
                elev = ref_height - data / (fac * bpr)
                elev_vars[var_name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def stl(self, freq: str = 'W', periods: int = 52, robust: bool = False) -> 'Batch':
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Decomposes time series into trend, seasonal, and residual components.
        The Batch must have a 'date' dimension.

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
            Batch containing 'trend', 'seasonal', and 'resid' variables for each polarization.

        Examples
        --------
        >>> model = (unwrapped - unwrapped.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> stl_result = displacement.stl(freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, resid components

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
            wrapped = {k: self.wrap(v) for k, v in (mapping or {}).items()}
            dict.__init__(self, wrapped)

    @staticmethod
    def wrap(data):
        """Wrap the (y, x) planes to [-pi, pi]; leave everything else alone.

        Wrapping the whole Dataset would fold the radar metadata too --
        near_range 800000 m comes back as 2.3 rad, silently -- so only variables
        that actually carry phase are wrapped.
        """
        if isinstance(data, xr.Dataset):
            out = data.copy()
            for v in data.data_vars:
                da_ = data[v]
                if da_.ndim >= 2 and tuple(da_.dims[-2:]) == ('y', 'x'):
                    out[v] = np.mod(da_ + np.pi, 2 * np.pi) - np.pi
            out.attrs = data.attrs
            return out
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def trend2d(self, *args, **kwargs):
        raise TypeError(
            "trend2d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    def __add__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: Batch):
        import xarray as xr
        import operator as _operator
        # SUBTRACT THE GRIDS, CARRY THE REST. A whole-Dataset `ds - val` reaches
        # the burst ids and the radar geometry too: numpy refuses to subtract
        # <U43 strings, so align() raised, and radar_wavelength minus itself is
        # 0, so meter2rad became infinite where it did not. _binary_vars is the
        # one place that rule lives.
        _sub_grids = lambda d, v: BatchCore._binary_vars(d, v, _operator.sub)
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    # Get a spatial variable (with y, x dims) to check for pair dimension
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)
                    first_elem = val[0]

                    if isinstance(first_elem, (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        result[k] = _sub_grids(ds, self[[k]].polyval({k: val})[k])
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        # Use da.stack for dask 0-d arrays to avoid triggering .compute()
                        if any(hasattr(v, 'dask') for v in val):
                            import dask.array as _da
                            offsets = xr.DataArray(_da.stack(val), dims=['pair'])
                        else:
                            offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = _sub_grids(ds, offsets)
                    elif len(val) == 1:
                        # Single value wrapped in list: [offset]
                        result[k] = _sub_grids(ds, val[0])
                    else:
                        # Single pair degree=1: [ramp, offset]
                        result[k] = _sub_grids(ds, self[[k]].polyval({k: val})[k])
                elif isinstance(val, (int, float)) \
                        or (hasattr(val, 'ndim') and val.ndim == 0):
                    # Scalar subtraction (concrete or dask 0-d array)
                    result[k] = _sub_grids(ds, val)
                else:
                    result[k] = _sub_grids(ds, val)
        return type(self)(result)

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] * other[k] if k in other else self[k] for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] / other[k] if k in other else self[k] for k in keys})

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
        return BatchComplex(self.map_da(lambda da, **kw: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
        """
        #print ('wrap _agg')
        import inspect
        import xarray as xr
        import pandas as pd
        out = {}
        for key, obj in self.items():
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
                    R = xr.ufuncs.abs(obj.mean(**kwargs))
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    agg_result = xr.ufuncs.angle(agg_result)
            
            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')
            
        #print ('wrap _agg self.chunks', self.chunks)
        #return type(self)(out).chunk(self.chunks)
        #print ('wrap _agg self.chunks', self.chunks)
        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        #print ('wrap chunks', chunks)
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

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
        #print ('wrap coarsen')
        chunks = self.chunks
        #print ('self.chunks', chunks)
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics
            ds2 = xr.ufuncs.exp(1j * ds.astype('float32'))
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds2 = ds2.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
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
        """
        from .Batch import Batch
        import xarray as xr

        keep_attrs = kwargs.pop('keep_attrs', False)
        data_vars = next(iter(self.values())).data_vars

        # build two Batches of the real sin and cos components and filter them
        sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
        cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

        # compute wrapped phase using arctan2
        out: dict[str, xr.Dataset] = {}
        for k in self.keys():
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
                 union: bool = False, device: str = 'auto',
                 debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

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
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = phase.unwrap2d(weight=corr)  # With weights
        """
        from .Stack import Stack

        return Stack.unwrap2d(Stack(), self, weight=weight,
                                       conncomp=conncomp, conncomp_size=conncomp_size,
                                       conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                                       conncomp_linkcount=conncomp_linkcount, union=union,
                                       device=device, debug=debug, **kwargs)

    def unwrap2d_chunk(self, weight: 'BatchUnit | None' = None, overlap=None,
                       device: str = 'auto', debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Unlike unwrap2d() which requires a single spatial chunk (global unwrapping),
        this method unwraps each spatial chunk independently with overlap margins.
        Suitable for large rasters where global unwrapping would exceed memory.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size (0.25 = 25%).
            Int = pixels. Tuple (y, x) for different overlap per axis. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batch
            Batch of unwrapped phase.
        """
        from .Stack import Stack

        return Stack.unwrap2d_chunk(Stack(), self, weight=weight,
                                              overlap=overlap, device=device,
                                              debug=debug, **kwargs)

    def unwrap2d_irls(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
                      max_iter: int = 50, tol: float = 1e-2, cg_max_iter: int = 10,
                      cg_tol: float = 1e-3, epsilon: float = 1e-2,
                      conncomp_size: int = 30, semaphore: int = 8, debug: bool = False) -> 'Batches':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

        This is the core unwrapping algorithm. Disconnected components are
        unwrapped independently and aligned using per-component circular mean.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        max_iter : int
            Maximum IRLS iterations. Default 50.
        tol : float
            Convergence tolerance. Default 1e-2.
        cg_max_iter : int
            Maximum conjugate gradient iterations. Default 10.
        cg_tol : float
            Conjugate gradient tolerance. Default 1e-3.
        epsilon : float
            Smoothing parameter for L1 approximation. Default 1e-2.
        conncomp_size : int
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default 30.
        semaphore : int
            Maximum concurrent CPU IRLS tasks per process. Default 8.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, BatchUnit):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: BatchUnit of component labels (uint16, 0=invalid, 1=largest, ...)

        Notes
        -----
        Uses a novel DCT+IRLS algorithm that combines DCT efficiency with IRLS
        robustness. See `utils_unwrap2d.irls_unwrap_2d` for algorithm details
        and references.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped, conncomp = phase.unwrap2d_irls(weight=corr)
        """
        from .Stack import Stack

        return Stack.unwrap2d_irls(Stack(), self, weight=weight,
                                            device=device, max_iter=max_iter, tol=tol,
                                            cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                                            epsilon=epsilon, conncomp_size=conncomp_size,
                                            semaphore=semaphore, debug=debug)

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
        dict.__init__(self, mapping or {})

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
    def fit1d(self, baseline: str = 'BPR',
              max_dh: float = 200.0, max_dv: float = 25.0,
              step_dh: float = 4.0, step_dv: float = 2.0,
              max_seasonal: 'float | None' = None,
              budget: 'str | None' = None) -> 'Batch':
        """
        Full per-pixel model on the per-date complex stack -- NO network.

        The 1d twin of fit3d(): the SAME `_3d_arc_fit` kernel and the SAME
        model, fitted on each pixel's own time series instead of on arcs
        between neighbours. 1d is the time axis alone; 3d adds the two spatial
        ones. Returns the MODEL ONLY, named and scaled exactly as fit3d()
        names and scales it, so predict(model) is the single inverse for both.

        Velocity is the ROTATION RATE of the per-date phase vectors, so a pixel
        time series is exactly the object `_3d_arc_fit` already solves: a
        zero-centred lattice over (height error, rate) followed by a
        majorise-minimise refinement, with the constant scatterer phase
        profiled out by rotation and never estimated. Nothing is wrapped or
        unwrapped, and no reference date is needed.

        WHY THIS REPLACED THE MOVING-WINDOW ESTIMATOR. The previous version
        reported the constant term of a {1, cos, sin} fit to per-window rates,
        each estimated over a short window of few samples searching +-pi/dt.
        That path holds up only at high coherence and degrades where real
        pixels live: its neighbourhood disagreement came out FLAT with radius,
        which is what a noise field looks like, while this fit's grows with the
        box, like a real field. It was also far worse conditioned and more
        expensive.

        WHAT IT GIVES UP. An annual term of amplitude A rad leaves the model
        misspecified, and the coherence of the true rate is |J0(A)| while a
        sideband one cycle/yr away gets |J1(A)|; they cross at A = 1.435 rad,
        above which the sideband is genuinely the higher maximum and NO
        coherence-maximising estimator returns the truth. Whether a stack
        reaches that amplitude is a property of the stack, not of the fit;
        where it does, the rate is not identifiable from one pixel's phase
        alone, and the annual term belongs in the model.

        Parameters
        ----------
        baseline : str
            Variable holding the perpendicular baseline per date. With it the
            per-pixel DEM error is solved jointly with the rate, which matters:
            they are NOT separable one at a time, because the perpendicular
            baseline is not a smooth function of time. Without it (absent
            variable, or None) the height term is not estimated at all and the
            rate carries whatever the DEM error contributes.
        max_dh, max_dv : float
            Largest height error (m) and rate (mm/yr) to admit. A pixel solving
            outside them returns NaN rather than a plausible wrong number. The
            search runs wider than they say, so max_dv=100 detects 99 mm/yr on
            its merits and never against a boundary.
        step_dh, step_dv : float
            Lattice steps. They choose which basin is found, not the accuracy --
            the refinement is continuous and absorbs the quantisation.
        max_seasonal : float
            Largest annual amplitude to admit, in mm of LOS (HALF amplitude, so
            60 means a 120 mm peak-to-peak swing). 0 (default) leaves the annual
            term out of the model entirely.

            It is not a refinement: an annual term of amplitude A radians leaves
            coherence |J0(A)| at the true rate and |J1(A)| one cycle/yr away,
            and they cross at A = 1.435 rad. Above that the sideband IS the
            higher maximum, so a {height, rate} fit returns the sideband rather
            than the truth; with the term in the model the rate returns to its
            no-seasonal accuracy.

            It costs search time, and a little accuracy when there is no annual
            signal at all, so it is cheap to leave on. Large amplitudes are
            only partly recovered, but they fail LOUDLY -- NaN rather than
            silent wrong rates. Where a stack carries no seasonal signal the
            default 0 is right; zones with a real one are what this is for.
            ON ARCS, KEEP IT SMALL. A seasonal signal is long-wavelength, so
            an arc -- two pixels tens of metres apart -- sees only the small
            residue that does not cancel in the difference. A large
            max_seasonal there is wrong twice over, since it searches thousands
            of lattice points for an amplitude that cannot be present.

            Small, it earns its keep: marginal arcs are rescued and nodes
            isolated at any threshold join the network. Set too small, the arcs
            are rescued but fitted poorly, so the amplitude does need room to
            move.

            Judge any gain against a MATCHED-gamma null, not a raw one: two free
            parameters always raise gamma, and pure-noise arcs sit low enough
            that there is far more room to climb there than at a real arc, so an
            unmatched comparison understates the real gain.

            Whether the atmosphere is itself seasonal is a property of the
            stack and has to be checked there, against a permuted-date null
            rather than by eye. On a stack with genuinely seasonal delay the
            annual term would absorb it, and per pixel the two are not
            separable.

            What it does fix, where a real seasonal signal exists, is the
            contamination of dh and dv by leaving it out: an unmodelled annual
            term biases the height and can push the rate onto a whole sideband,
            while modelling it returns both to their clean values.
        budget : str or None
            Memory budget for the lattice product, e.g. '512MB'.

        Returns
        -------
        Batch
            ONE dataset of model parameters, named by quantity, identical in
            name, unit and convention to fit3d()'s:

              `velocity`   rad/yr
              `height`     rad per unit ele2phase
              `seasonal`   complex rad, the fitted annual
              `coherence`  gamma, the resultant length the fit maximised
              `rmse`       radians, circular deviation about that same model

            NO `conncomp`. fit3d() carries one because its network solves in
            connected components; every pixel here is solved alone, so there is
            no component to report and none is invented. predict() never reads
            it -- it is pixelwise.

            HEIGHT AND SEASONAL COST NOTHING. This is not a richer fit, it is
            the same fit reporting what it already had: the kernel must solve
            height jointly with rate (they do not separate) and the annual sits
            in the same objective, and the previous version bound two of the
            four returned values to `_dh` and `_sa` and dropped them. Same
            lattice, same 16 refinements, same runtime, four parameters out
            instead of one.

            THE RMSE IS EXACT. gamma is the resultant length of the residual
            about the model that was actually REPORTED -- it is the objective
            the fit maximised -- so

                sigma = sqrt(-2 ln gamma)

            is self-consistent by construction, equals the RMS for small
            residuals, and has no ceiling as the phase decorrelates. It is
            inflated by n/(n-p) for the parameters spent, and p now counts the
            annual's two whenever max_seasonal is non-zero -- it did not while
            the annual was fitted and discarded, which under-reported sigma on
            every pixel.

        Examples
        --------
        >>> model = stack.fit1d()
        >>> noise = stack * stack.predict(model, baseline='BPR').iexp()
        """
        import dask.array as da
        import numpy as np
        from .BatchCore import _parse_budget
        import xarray as xr
        from .Batch import Batch

        BatchCore._require_lazy(self, 'fit1d')

        from .utils_dask import get_dask_chunk_size_mb
        budget_mb = (_parse_budget(budget) if budget is not None
                     else get_dask_chunk_size_mb())

        # DELEGATE BY STACK TYPE. The same call fits the same model whether the
        # samples are dates or pairs; only the design columns differ, so the
        # caller writes fit1d() either way and never selects a variant by hand.
        # THE ANNUAL'S DEFAULT FOLLOWS THE STACK, because the model does. On
        # dates the kernel's cos(2 pi t) IS the basis, so 5 mm is the useful
        # default. On pairs the annual is a DIFFERENCE of two epochs, which is
        # the kernel's basis at the mean epoch rotated 90 degrees and scaled by
        # 2 sin(pi dt) -- a PER-PAIR scale no single t can carry. So pairs
        # default to no annual and raise only if one is actually asked for.
        _pairs = any('pair' in ds[v].dims
                     for ds in self.values() for v in ds.data_vars)
        if max_seasonal is None:
            max_seasonal = 5.0
        if _pairs:
            # the split is kept so a pair-domain fit has a home when one works
            raise NotImplementedError(
                'fit1d() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')

        model_result = {}
        for burst_id, ds in self.items():

            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                    and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                raise TypeError(
                    f'fit1d() found no complex (date, y, x) variables in '
                    f'burst {burst_id}')
            # ONE polarisation, exactly as fit3d(): the model variables are
            # named by quantity alone so predict() can look them up directly,
            # and two polarisations would collide on those names.
            if len(pols) > 1:
                raise ValueError(
                    f"fit1d() fits ONE polarisation; burst '{burst_id}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two "
                    "polarisations would collide. Select one first, e.g. "
                    "batch[['VV']].")
            pol = pols[0]

            # ele2phase = B_perp / (R sin theta), one value per burst: it
            # varies about a percent across it, which keeps the fit a matmul
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
                    _fac = Batch._elevation_phase_approximate(self)[burst_id]
                    ele2phase = bp / (meter2rad / _fac)
            if meter2rad is None:
                raise TypeError(
                    f'fit1d() needs radar_wavelength in burst {burst_id} to '
                    f'turn a rotation rate into a velocity')

            # t = 0 AT THE MASTER, where B_perp is smallest -- the same origin
            # _3d_fit_frame and predict() use. Rate and height do not care
            # (a shift in t adds a constant and the constant is profiled out),
            # but the annual does: car = exp(2j*pi*t) rotates by
            # exp(2j*pi*delta), so a model fitted on one origin and removed on
            # another leaves a residual annual of 2|sin(pi*delta)|*|seasonal|.
            # This used to run from dates[0], which was invisible only because
            # the seasonal was discarded before anyone could subtract it.
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
                           _se=float(max_seasonal), _bm=int(budget_mb)):
                from .utils_arcs import _3d_arc_fit
                S = np.asarray(block)
                nd = S.shape[0]
                shape = S.shape[1:]
                Z = np.ascontiguousarray(S.reshape(nd, -1))
                if Z.shape[1] == 0:
                    e = np.empty(0, np.complex64).reshape(shape)
                    return np.stack([e, e, e, e, e], axis=0)
                # FOUR values, all of them kept. _3d_arc_fit already returns
                # rad/yr and rad per unit ele2phase: the library works in phase
                # and displacement_los() is the one place a length is made.
                gam, hgt, vel, sea = _3d_arc_fit(Z, _h, _t, _m,
                                                 _mh, _mv, _sh, _sv, _bm, _se)
                # ONE CONVENTION ACROSS EVERY FIT: displacement_los() must turn
                # this model into a negative rate where the ground subsides,
                # whichever fit produced it. _3d_arc_fit solves the per-DATE
                # phase, and a pair runs opposite to it -- an SLC phase is
                # -(4pi/lambda)r, so ref*conj(rep) carries +m2r*dr while a date
                # series carries -m2r*dr. displacement_los()'s -lambda/4pi is
                # derived for the pair, so the pair sense is the one the library
                # converts. Returning the date sense reports subsidence as uplift.
                #
                # HEIGHT IS NOT NEGATED. Its per-date term +hgt*e2p_d
                # differences to -hgt*e2p_pair, which already matches the pair
                # convention, and it is why a global sign flip on the
                # prediction does not work.
                vel = -vel
                sea = -sea
                # circular deviation about the REPORTED model, inflated for the
                # parameters the fit spent: phi0 and rate always, height when a
                # baseline was given, and the annual's real and imaginary parts
                # whenever it is in the model.
                nok = np.maximum((np.abs(Z) > 0).sum(axis=0), 1)
                npar = 2 + (0 if _h is None else 1) + (2 if _se > 0 else 0)
                infl = nok / np.maximum(nok - npar, 1)
                Rres = np.clip(gam.astype(np.float64), 1e-9, 1.0)
                rms = np.sqrt(np.maximum(-2.0 * np.log(Rres), 0.0) * infl)
                rms = np.where(np.isfinite(gam), rms, np.nan)
                # complex64 carries the real planes without rounding, so one
                # dtype ships all five and the seasonal needs no second pass
                return np.stack([vel.reshape(shape).astype(np.complex64),
                                 hgt.reshape(shape).astype(np.complex64),
                                 sea.reshape(shape).astype(np.complex64),
                                 gam.reshape(shape).astype(np.complex64),
                                 rms.reshape(shape).astype(np.complex64)],
                                axis=0)

            stacked = da.blockwise(
                _fit_block, 'nyx', data_dask, 'dyx',
                new_axes={'n': 5}, concatenate=True, dtype=np.complex64,
                meta=np.empty((0, 0, 0), dtype=np.complex64),
                name='fit1d_arcfit')

            coords = {kk: vv for kk, vv in data_da.coords.items()
                      if kk in ('y', 'x', 'spatial_ref')}
            mvars = {}
            for nm_, arr_ in (('velocity', stacked[0].real.astype(np.float32)),
                              ('height', stacked[1].real.astype(np.float32)),
                              ('seasonal', stacked[2].astype(np.complex64)),
                              ('coherence', stacked[3].real.astype(np.float32)),
                              ('rmse', stacked[4].real.astype(np.float32))):
                mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            # the epoch the model is referenced to, named as every other date
            # in this library is named
            mds = mds.assign_coords(date=np.datetime64(int(dday[_master]), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[burst_id] = mds
        return Batch(model_result)

    def predict(self, model, baseline: 'str | None' = 'BPR',
                ref=None) -> 'Batch':
        """
        Predicted per-date phase from a fit3d() or fit1d() model.

        Reconstructs, for every date in this stack,

            phi_d = velocity * t_d + height * ele2phase_d + Re(car_d * conj(seasonal))

        with `car_d = exp(2j*pi*t_d)`, `t_d` in years from the first date, and
        `ele2phase_d = BPR_d / median(R sin(incidence))` -- the same geometry
        fit3d() fitted against, so the two are exact inverses.

        The scatterer's constant phase is NOT part of the model: fit3d() profiles
        it out by rotation rather than gauging it to an epoch, so the prediction
        is correct up to one constant per pixel. That is what you want for
        removal -- `stack * predict(model).iexp(sign=1)` cancels the modelled
        part and leaves the constant, which no interferometric measurement
        determines. SIGN=1, not the default: the prediction is published as
        master*conj(date), the same radians displacement_los() converts, and
        rotating it out of the stack therefore runs the other way.

        Parameters
        ----------
        model : Batch
            Output of fit3d() or fit1d(): variables `velocity`, `height`,
            `seasonal`
            (unprefixed, one polarisation).
        baseline : str or None
            Per-date perpendicular baseline variable. DEFAULT None, i.e. the
            topographic term is NOT projected onto dates, so the prediction is
            rate + seasonal only -- the object that matches phase which has
            already had topography removed. Pass 'BPR' to include it and predict
            the raw phase instead. Verified: predict('BPR') / predict(None) is
            exactly exp(1j*ele2phase*height) to 4e-07 rad.

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
        >>> model  = stack.fit3d()
        >>> ground = stack * stack.predict(model=model).conj()
        >>> resid  = stack * stack.predict(model=model, baseline='BPR').conj()
        >>> # modelled displacement per date, unwrapped:
        >>> disp = (stack.predict(model=model, phasor=False)
        ...              .displacement_los(stack.transform()))
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        # A trend2d() MODEL is a different animal from a fit model: no
        # velocity or height, but per-date coefficients over the stack's own
        # covariates. Recognised by what it carries, evaluated per chunk.
        if all('trend2d_vars' in model[k].attrs for k in model):
            return self._trend2d_predict(model)
        out = {}
        for key, ds in self.items():
            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if len(pols) > 1:
                raise ValueError(
                    f"predict() takes ONE polarisation, burst '{key}' carries "
                    f"{len(pols)}: {pols}. fit3d() has the same rule.")
            if not pols:
                raise TypeError(
                    f"predict() found no complex (date, y, x) variables in '{key}'.")
            pol = pols[0]
            da_xr = ds[pol]
            if da_xr.dims[0] != 'date':
                da_xr = da_xr.transpose('date', ...)

            mds = model[key]
            missing = [v for v in ('velocity', 'height', 'seasonal') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit3d().")

            # TIME ORIGIN MUST MATCH THE FIT. utils_arcs zeroes t at the MASTER
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
                # to point the geometry somewhere else, but fit1d()/fit3d() take
                # theirs from the stack they are given and cannot be redirected,
                # so another source could only make the prediction disagree with
                # the model it inverts. Nothing physical rode on the choice
                # either: the scalar rescales `height` inversely and cancels in
                # the product.
                _fac = Batch._elevation_phase_approximate(self)[key]
                bp = np.asarray(ds[baseline].values, dtype=float)
                while bp.ndim > 1:
                    bp = np.nanmean(bp, axis=-1)
                _lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
                ele2phase = bp / ((4.0 * np.pi / _lam) / _fac)

            vel = mds['velocity'].data
            hgt = mds['height'].data
            sea = mds['seasonal'].data
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
                #     residual = stack * predict(model).iexp(sign=+1)
                # iexp() defaults to exp(-1j*phase), which is the removal
                # direction for the raw SLC phase this no longer returns.
                phi = vel * float(t[i])
                if ele2phase is not None:
                    # height is NaN where no baseline was available; a NaN would
                    # poison the whole date, so contribute only where it solved
                    phi = phi - da.nan_to_num(hgt) * float(ele2phase[i])
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
            pds = xr.Dataset({pol: xr.DataArray(pred, dims=('date', 'y', 'x'),
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
        Per-date displacement from a fit1d()/fit3d() model MIXED with the raw signal.

        Returns the model's displacement carrying the residual the data actually
        has, instead of the smooth curve predict() draws through it:

            resid = stack * predict(model, baseline).iexp(sign=1)
            mix   = predict(model, baseline=None) - angle(resid * conj(mean_date(resid)))

        NOTHING IS UNWRAPPED HERE, and that is the point rather than an omission.
        fit1d()/fit3d() fit in the complex domain, predict() returns radians that
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
            Output of fit1d() or fit3d(): variables `velocity`, `height`,
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
        # The height term's metres-to-radians scale comes from this stack on both
        # sides, as it does in fit1d()/fit3d(), so it cancels in the product.
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

    def trend2d(self, vars, union: bool = False,
                range: float = None,
                debug: bool = False) -> 'Batch':
        """
        Spatial trend of the complex phase, PER DATE, as a unit-magnitude
        phasor: `phi_d = sum_i g_di * v_i + k_d`, so removing it is a rotation.

        >>> trend = stack.where(stack.adi() < 0.25).trend2d(
        ...     stack.transform()[['northing','easting','ele']])
        >>> flat  = stack.detrend2d(trend)

        IT RUNS ON THE RAW STACK, where detrending belongs. The scatterer phase
        is still there, and being constant in time is what cancels it: one
        multiplication by the epoch the baselines are measured from, where BPR
        is zero. Its own plane is then common to every date, so no velocity
        depends on it.

        NOTHING IS UNWRAPPED, AND NOTHING IS SEARCHED. Maximising
        `sum cos(phi - g.v - k)` is a bounded-influence regression whose score
        is the SINE of the residual, and it is SOLVED for, by an ascent from
        zero. One pass over the data; the accumulator is read at whatever
        gradient the iteration asks for, not at lattice nodes.

        THE OLD GLOBAL ARGMAX WAS THE BUG. A variable's own distribution has a
        transform -- what a perfectly coherent, TREND-FREE date would score --
        and real topography puts big far lobes in it, because the pixels crowd
        into a fraction of the elevation range. Taking the largest peak over a
        wide band then answered with the elevation histogram rather than the
        phase whenever a date was weak. The ascent from zero follows the
        objective instead, to the stationary point CONNECTED to zero: strong
        trends on a connected slope are still reached, well past the
        half-power width, but a trend separated from zero by a null of a
        NEAR-UNIFORM sampling -- a multi-cycle ramp in `northing` -- is not,
        and comes back as the small stationary point near zero. Fit map
        ramps with Batch.trend2d(degree=...) on real phase, not here; this
        estimator is
        for covariates whose sampling has structure, elevation above all.

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
              trend removed;
          gain -- their difference. Fitting noise alone gains up to ~0.07,
              so gain above that means a trend was found;
          slope_<var> -- the fitted slope, radians per unit of the variable;
          resolution_<var> -- how far apart two slopes must be for this
              variable's sampling to tell them apart, same units: the
              half-power width of the sampling's own transform. A fitted
              slope beyond it is real signal but its value is lobe-ambiguous;
          stderr_<var> -- one-sigma of the fitted slope, same units, measured
              as half the disagreement of two checkerboard halves of the
              scene (one degree of freedom: honest scale, noisy itself);
          pixels -- samples fitted.
        NaN only when there is no fit at all: no pixels, a degenerate
        covariate, a trend walking out of `range`, or no convergence.
        THIS ONLY FITS; detrend2d() applies.

        Parameters
        ----------
        vars : Batch
            `stack.transform()[['northing','easting','ele']]` at this stack's
            posting; each variable becomes one gradient, as a raster or, like
            the map coordinates, a vector along one axis.

            TAKE THEM FROM THE UNFILTERED STACK: where() masks the geometry
            too. And mind the frame -- `azi` and `rng` restart at every burst,
            so one plane cannot be written in them across bursts; `northing`
            and `easting` are the same grid for all of them.
        union : bool
            False (default) fits each burst on its own pixels, so overlapping
            bursts can disagree over the ground they share. True adds every
            burst's accumulators into one fit per date; nothing is merged or
            resampled, a sum over pixels not caring where they came from.
        range : float
            How much of gradient space the accumulator can represent, radians
            across each variable's extent. Default None self-sizes: 128
            cycles at one variable -- an order of magnitude beyond any
            physical trend, for a grid of about a thousand numbers -- and 16
            and 8 cycles at two and three, where the grid is cells**k and
            width costs real memory. The answer does not depend on it (the
            same fit on a grid twice the size returns the same numbers); it
            only has to be big enough, and the fit says so if it ever is not.
            Leave it alone.
        debug : bool
            Print each date's turn across every variable, its coherence and
            its reach, and name the dates that did not resolve.

        Returns
        -------
        BatchComplex
            A model dataset per burst, nothing of the stack in it:
            The MODEL, per date, in scipy's terms and in float64:
            'intercept' (radians where every covariate is zero, relative to
            the reference date), per covariate 'slope_<var>' (radians per
            unit of it), 'resolution_<var>' (how far apart two slopes must be
            for this sampling to tell them apart, same units), 'stderr_<var>'
            (the slope's one-sigma, same units), and 'coherence', 'coherence0',
            'gain', 'pixels'; the covariate names ride as attributes. No
            raster: `stack.predict(trend)` evaluates the phase per chunk when
            asked, `stack.detrend2d(trend)` removes it.
        """
        import numpy as np
        import builtins as _builtins
        import xarray as xr
        import dask as _dask
        import dask.array as da
        from . import utils_detrend

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
                    f"stack[['{pols[0]}']]), fit it, and detrend2d() applies "
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
            preps.append({'key': key, 'pol': pols[0], 'ds': ds,
                          'data_da': data_da, 'data_ref': data_dask * _R,
                          'vars_dask': vars_dask, 'var_names': var_names,
                          'nd': nd, 'k': k, 'stats': stats,
                          'vars_dims': vars_dims, 'iref': _iref})

        if not preps:
            return BatchComplex({})
        k = preps[0]['k']
        # THE GRID SIZES ITSELF. `range` is only how much of gradient space
        # the accumulator can represent, and the cost of representing more is
        # a longer vector -- linear in cells at one variable -- so the default
        # is deliberately absurd: 128 cycles across the covariate's extent at
        # k=1, an order of magnitude beyond any physical trend, for a grid of
        # ~a thousand numbers per date. Only at two and three variables does
        # width cost real memory (the grid is cells**k), so there the default
        # falls back to 16 and 8 cycles and the one message that can ask for
        # more still exists. Nobody should ever need to set this.
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
            _acc0 = []
            for p in grp:
                _args = []
                for var_dask, _d in zip(p['vars_dask'], p['vars_dims']):
                    _args += [var_dask, ''.join(_d)]
                _w = utils_detrend.trend2d_width(_cells, k, len(_axes))
                _acc.append(da.blockwise(
                    _trend2d_accumulate_for_dask, 'dyxf',
                    p['data_ref'], 'dyx', *_args,
                    stats=_stats, cells=_cells,
                    dims=[''.join(_d) for _d in p['vars_dims']],
                    adjust_chunks={'y': 1, 'x': 1},
                    new_axes={'f': _w},
                    dtype=np.float64,
                    meta=np.empty((0, 0, 0, 0), np.float64)
                ).sum(axis=(1, 2)))
                # ONE CHECKERBOARD HALF of the same sums (the other half is
                # total minus this one): two independent coarse pixel sets,
                # whose disagreement prices the estimate per date
                _yc = np.asarray(p['data_da'].coords['y'].values, float)
                _xc = np.asarray(p['data_da'].coords['x'].values, float)
                _yd = da.from_array(_yc, chunks=p['data_ref'].chunks[1])
                _xd = da.from_array(_xc, chunks=p['data_ref'].chunks[2])
                _acc0.append(da.blockwise(
                    _trend2d_accumulate_half_for_dask, 'dyxf',
                    p['data_ref'], 'dyx', *_args, _yd, 'y', _xd, 'x',
                    stats=_stats, cells=_cells, n_vars=k,
                    dims=[''.join(_d) for _d in p['vars_dims']],
                    checker=0,
                    extent=(float(_yc.min()), float(_yc.max()),
                            float(_xc.min()), float(_xc.max())),
                    adjust_chunks={'y': 1, 'x': 1},
                    new_axes={'f': _w},
                    dtype=np.float64,
                    meta=np.empty((0, 0, 0, 0), np.float64)
                ).sum(axis=(1, 2)))
            _dts = da.from_array(
                np.asarray(grp[0]['data_da'].coords['date'].values)
                .astype('datetime64[D]').astype(np.int64),
                chunks=grp[0]['data_da'].data.chunks[0])
            _coef = da.blockwise(
                _trend2d_finalize_for_dask, 'dc',
                sum(_acc[1:], _acc[0]), 'df',
                sum(_acc0[1:], _acc0[0]), 'df',
                _dts, 'd',
                concatenate=True, stats=_stats, cells=_cells, k=k,
                axes=_axes,
                label=grp[0]['key'] if not union else 'union',
                new_axes={'c': 3 * k + 5},
                dtype=np.float64, meta=np.empty((0, 0), np.float64))

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
                print(f"trend2d('{p['key']}'): {p['nd']} dates, {k} "
                      f"variables {p['var_names']}, referenced to date "
                      f"{p['iref']}"
                      + (" [one fit for every burst]" if union else ""),
                      flush=True)
                hdr = "    date " + " ".join(f"{v:>12s}" for v in p['var_names'])
                print(hdr + f"{'coh':>9s} {'pixels':>12s}  "
                      + " ".join(f"{'resolution ' + v:>13s}"
                                 for v in p['var_names'])
                      + "   [rad across the variable's span]", flush=True)
                _tag = {1: 'no pixels', 2: 'walked out of `range`',
                        3: 'degenerate covariate', 4: 'did not converge'}
                for d in _builtins.range(p['nd']):
                    _coh = _cf[d, k + 2]
                    _reach = " ".join(
                        f"{_cf[d, k + 5 + i] * _span[i]:13.4f}"
                        for i in _builtins.range(k))
                    if not _det[d]:
                        print(f"    {d:4d} " + " ".join(f"{chr(45) * 2:>12s}"
                              for _ in _builtins.range(k))
                              + f" {_coh:8.5f} {int(_cf[d, k + 4]):12,d}  "
                              f"{_reach}   {_tag[int(_cf[d, k + 1])]}",
                              flush=True)
                        continue
                    row = " ".join(f"{_cf[d, i] * _span[i]:12.4f}"
                                   for i in _builtins.range(k))
                    print(f"    {d:4d} {row} {_coh:8.5f} "
                          f"{int(_cf[d, k + 4]):12,d}  {_reach}", flush=True)
                if union:
                    debug = False        # the table is the same for every burst

            ds = p['ds']
            # THE MODEL IS THE PRODUCT, like fit3d(): the fit's own numbers
            # and nothing of the stack. A date's trend is
            # `intercept_d + sum_i slope_di * v_i` -- a handful of numbers
            # per date over the stack's own covariates. Evaluating it here
            # into a (date, y, x) phasor made a raster the size of the
            # stack, which compute() then persisted across the cluster and
            # every block read had to fetch pieces of. predict() evaluates
            # it lazily where it is asked for; detrend2d() applies it so.
            # SCIPY'S NAMES AND CONVENTIONS, IN FLOAT64: `slope_<var>` is the
            # gradient per unit of the covariate and `intercept` the phase
            # where every covariate is zero, as linregress reports them, so
            # the model reads without any centre or span beside it. The fit
            # itself is centred for conditioning; the constant is moved to
            # zero here, in float64, where the shift costs nothing.
            o = xr.Dataset(coords={'date': np.asarray(
                p['data_da'].coords['date'].values)}, attrs=dict(ds.attrs))
            _icpt = _coef[:, k].astype(np.float64)
            for i in _builtins.range(k):
                _icpt = _icpt - _coef[:, i].astype(np.float64) * np.float64(_stats[i])
            o['intercept'] = xr.DataArray(_icpt, dims=('date',))
            o.attrs['trend2d_vars'] = list(p['var_names'])
            o.attrs['trend2d_dims'] = [''.join(_d) for _d in p['vars_dims']]
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
            for i, var in enumerate(p['var_names']):
                # per unit of the covariate, all three, so they read against
                # each other: the slope, how far apart two slopes must be for
                # this sampling to tell them apart, and the slope's one-sigma
                o[f'slope_{var}'] = xr.DataArray(
                    _coef[:, i].astype(np.float64), dims=('date',))
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
        expression `intercept_d + sum_i slope_di * v_i` over the covariate
        rasters, evaluated in float64 and handed on as float32 phase.

        THE COVARIATES ARE THE STACK'S OWN. The model names them; each is
        taken from this stack's variables, or from its map coordinates for
        `northing`/`easting`, exactly as transform() exposes them, so nothing
        has to be carried along with the model. `vars` overrides that with a
        Batch of covariates at this posting when the stack does not hold
        them.

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
            grids = [v for v in ds.data_vars
                     if ds[v].dtype.kind == 'c' and 'date' in ds[v].dims
                     and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not grids:
                raise TypeError(
                    f"predict() found no (date, y, x) complex variable in "
                    f"'{key}' to evaluate the trend on.")
            ref = ds[grids[0]].transpose('date', 'y', 'x')
            data = ref.data
            _md = np.asarray(mds.coords['date'].values)
            _sd = np.asarray(ds.coords['date'].values)
            if _md.shape != _sd.shape or not np.array_equal(_md, _sd):
                raise ValueError(
                    f"predict(): the trend2d() model of '{key}' holds "
                    f"{len(_md)} dates, this stack {len(_sd)}; they must be "
                    f"the same acquisitions in the same order.")
            # the model per date, as CONSTANTS in the graph; the phase is
            # evaluated in float64 -- an intercept at zero and a slope times
            # a covariate in the thousands cancel to a few radians -- and
            # handed on as float32, the precision the raster always had
            _dc = data.chunks[0]
            kd = da.from_array(np.asarray(mds['intercept'].values, np.float64),
                               chunks=(_dc,))
            phi = kd[:, None, None]
            src = vars[key] if vars is not None else ds
            for i, (v, d) in enumerate(zip(names, dims)):
                if v in src.data_vars:
                    cov = src[v]
                elif v == 'northing' and 'y' in src.coords:
                    cov = xr.DataArray(np.asarray(src.y.values, np.float32),
                                       dims=('y',))
                elif v == 'easting' and 'x' in src.coords:
                    cov = xr.DataArray(np.asarray(src.x.values, np.float32),
                                       dims=('x',))
                else:
                    raise KeyError(
                        f"predict(): the trend2d() model needs covariate "
                        f"'{v}', which '{key}' does not carry. Pass "
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
                    np.asarray(mds[f'slope_{v}'].values, np.float64),
                    chunks=(_dc,))
                _v = cd
                if _d == ('y',):
                    _v = _v[:, None]
                elif _d == ('x',):
                    _v = _v[None, :]
                phi = phi + gd[:, None, None] * _v[None]
            coords = {k_: v_ for k_, v_ in ref.coords.items()
                      if k_ in ('date', 'y', 'x', 'spatial_ref')}
            out[key] = xr.Dataset({'phase': xr.DataArray(
                phi.astype(np.float32), dims=('date', 'y', 'x'),
                coords=coords)}, attrs=dict(ds.attrs))
        return Batch(out)

    def detrend2d(self, trend: 'Batch', vars=None) -> 'BatchComplex':
        """
        Remove a trend2d() model from EVERY pixel: `self * exp(-1j * phase)`.

        The phase is predict(trend): the model's per-date coefficients over
        the stack's own covariates, evaluated per chunk -- nothing of raster
        size is built, persisted or fetched. One trend, every polarisation:
        the atmosphere is the same for each complex raster.

        Parameters
        ----------
        trend : Batch
            What trend2d() returned for this stack.
        vars : Batch or None
            Covariates at this posting, only when the stack does not carry
            the ones the model names.

        Returns
        -------
        BatchComplex
            The stack with the trend rotated out of every polarisation,
            everything else untouched.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        for key in self.keys():
            if key not in trend or 'trend2d_vars' not in trend[key].attrs:
                raise ValueError(
                    f"detrend2d(): no trend2d() model for burst '{key}' -- "
                    f"pass what trend2d() returned for this stack.")
        pred = self._trend2d_predict(trend, vars=vars)
        res = {}
        for key, ds in self.items():
            phi = pred[key]['phase']
            # THE SAME NUMBERS THE RASTER USED TO HOLD: float32 phase into a
            # complex64 exponential, rotated out
            rot = xr.DataArray(
                da.exp(np.complex64(-1j) * phi.data.astype(np.complex64)),
                dims=phi.dims, coords=phi.coords)
            upd = {v: ds[v] * rot for v in ds.data_vars
                   if ds[v].dtype.kind == 'c' and 'y' in ds[v].dims
                   and 'x' in ds[v].dims and 'date' in ds[v].dims}
            res[key] = ds.assign(upd)
        return type(self)(res)

    def fit3d(self, threshold: float = 0.5, window: tuple = (32, 128),
                cell: tuple = (2, 8),
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
          level 2+ the DS attached so far are offered to whatever is still
                   unresolved as VOTERS only, inside the DS WINDOW: a pixel
                   with too few PS arcs to fill the quorum alone may complete
                   it with DS, but its value is solved from its PS arcs and
                   nothing else -- only the PS hold the datum

        Returns ONE dataset carrying the solve, its variables named
        by quantity:

          `velocity`   rad/yr
          `height`     rad per unit ele2phase
          `seasonal`   complex rad, the fitted annual
          `coherence`  arc coherence tying the pixel to the network
          `rmse`       sqrt(-2 ln coherence), rad
          `conncomp`   int8, -1 nodata, 0 the largest component
          `level`      int8, WHICH CASCADE STEP PLACED THE PIXEL: 0 a PS
                       network node, 1 a DS attached to that network, n a DS
                       whose quorum needed DS of levels below n, -1 nothing solved

        all NaN (or -1) where nothing was solved. Names rather than positions,
        so a caller never counts commas and adding a quantity moves nothing.
        There is NO polarisation prefix: fit3d() takes exactly one polarisation
        and raises otherwise, so nothing needs disambiguating and predict() can
        look the names up directly.

        RADIANS OUT, as everywhere else here. `displacement_los()` stays the
        single place a length is produced. `coherence` is the arc quality that
        ties each pixel in -- the mean over its own arcs for a node, the
        attaching arc for a densified DS -- and `rmse` is its exact inverse
        transform, carried for convenience rather than as new information.

        VELOCITY IS A RASTER, not a stats entry. The kernels' stats live in
        `utils_arcs._fit_stats`, a per-thread object written by whichever
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
        window : tuple of int
            `(wy, wx)` is the DS window in pixels, centred on each pixel: the
            neighbourhood the short arc test measures over. `(wy, wx, py, px)`
            sets the PS extent apart from it, otherwise the extent is derived
            from the window.

            The PS extent is the reach of everything that tests against a NODE
            -- the long arcs that prove a PS, the network arcs that carry the
            datum, and the DS attachment. It is therefore the caller's cost
            dial as well: widening it grows the candidate partners per pixel
            and the arc fits with them.
        baseline : str
            Variable holding the perpendicular baseline per date.
        iterations : int
            Refinement passes per arc, for the arcs that reach the final fit. The per-arc search is a lattice followed
            by a majorise-minimise refinement, and the refinement's step
            contracts by exactly `(1 - gamma)` per pass -- so the useful count
            follows from `threshold`, not from taste. At a 0.4 gate the
            contraction is 0.6 and eight passes leave 0.6**8, under two percent
            of a lattice cell: 0.07 m at `step_dh=4`, 0.03 mm/yr at
            `step_dv=2`. Refining far below the step it sits inside buys
            nothing.

            It is also the larger half of the attachment's cost, since stage 1
            is one product over the box while this runs on every arc this many
            times. A lower `threshold` contracts more slowly and wants more
            passes; a higher one wants fewer.
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
            How far to carry the solve, 0 to 2. Each level uses the one below
            it as its references, so they are cumulative rather than
            alternative.

            0   the PS network only. Nodes carry values; every other pixel is
                NaN. This is the answer when a caller wants only what the PS
                test certified.
            1   plus DS attached to PS, each by `consensus` agreeing PS
                partners inside the PS extent.
            2   plus DS whose quorum needed DS voters: a pixel can hold one
                to four coherent PS arcs and still fail level 1 where the PS
                are too sparse to field `consensus` -- a property of the
                ground, not of the pixel. The DS attached so far complete its
                vote; its value still comes from its PS arcs alone, so a
                higher level adds a few pixels and never inherits DS error.

            The default is 1. Level 2 costs the most arcs by far and is much
            the slowest stage, and every reference it uses is itself one hop
            from the network, so it adds coverage at a somewhat higher error
            rate. That is a trade worth making deliberately rather than by
            default.

            There is no level 3. Each hop's error adds, and level-2 pixels are
            certified by DS rather than by PS, so the argument that justifies
            level 2 does not survive another step.

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
            Largest annual amplitude to admit, in mm of LOS (HALF amplitude, so
            60 means a 120 mm peak-to-peak swing). 0 (default) leaves the annual
            term out of the model entirely.

            It is not a refinement: an annual term of amplitude A radians leaves
            coherence |J0(A)| at the true rate and |J1(A)| one cycle/yr away,
            and they cross at A = 1.435 rad. Above that the sideband IS the
            higher maximum, so a {height, rate} fit returns the sideband rather
            than the truth; with the term in the model the rate returns to its
            no-seasonal accuracy.

            It costs search time, and a little accuracy when there is no annual
            signal at all, so it is cheap to leave on. Large amplitudes are
            only partly recovered, but they fail LOUDLY -- NaN rather than
            silent wrong rates. Where a stack carries no seasonal signal the
            default 0 is right; zones with a real one are what this is for.
            ON ARCS, KEEP IT SMALL. A seasonal signal is long-wavelength, so
            an arc -- two pixels tens of metres apart -- sees only the small
            residue that does not cancel in the difference. A large
            max_seasonal there is wrong twice over, since it searches thousands
            of lattice points for an amplitude that cannot be present.

            Small, it earns its keep: marginal arcs are rescued and nodes
            isolated at any threshold join the network. Set too small, the arcs
            are rescued but fitted poorly, so the amplitude does need room to
            move.

            Judge any gain against a MATCHED-gamma null, not a raw one: two free
            parameters always raise gamma, and pure-noise arcs sit low enough
            that there is far more room to climb there than at a real arc, so an
            unmatched comparison understates the real gain.

            Whether the atmosphere is itself seasonal is a property of the
            stack and has to be checked there, against a permuted-date null
            rather than by eye. On a stack with genuinely seasonal delay the
            annual term would absorb it, and per pixel the two are not
            separable.

            What it does fix, where a real seasonal signal exists, is the
            contamination of dh and dv by leaving it out: an unmodelled annual
            term biases the height and can push the rate onto a whole sideband,
            while modelling it returns both to their clean values.

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
            `model = stack.fit3d(...)`, then `stack.predict(model=model)`.

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
        >>> model = stack.fit3d()
        >>> predicted = stack.predict(model=model)
        >>> model['velocity'].plot(cmap='turbo')   # rad/yr
        >>> good = model.where(model['conncomp'] == 0)
        >>> main = velocity.where(labels == 0)   # one datum, comparable
        """
        # DELEGATE BY STACK TYPE, exactly as fit1d() does: pairs take the
        # pair branch (not implemented), dates take the PS network below.
        if any('pair' in ds[v].dims
               for ds in self.values() for v in ds.data_vars):
            # `window` is the PS network's (32, 128) tuple on the date path; on the
            # pair path it is the side of the box the covariance is estimated over, so
            # a scalar. A tuple is reduced to its smallest side rather than refused.
            # the split is kept so a pair-domain fit has a home when one works
            raise NotImplementedError(
                'fit3d() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')
        # validated HERE, not only in the kernel: fit3d() returns a lazy Batch,
        # so a bad value would otherwise surface at compute time far from where
        # it was written
        from . import utils_arcs as _ua
        _ua._3d_consensus(consensus)
        if int(level) < 0:
            raise ValueError(f'level must be >= 0; got {level!r}')
        _kw = dict(threshold=threshold, window=window, cell=cell,
                   baseline=baseline, level=level,
                   max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
                   step_dv=step_dv, max_seasonal=max_seasonal,
                   consensus=consensus, err_dh=err_dh, err_dv=err_dv,
                   iterations=iterations, debug=debug)
        if union or len(self) == 1:
            return self._fit3d_union(**_kw)
        # union=False: EACH BURST ITS OWN SCENE, as trend2d() fits each burst
        # on its own pixels. The same driver runs once per burst -- there is
        # no second code path -- and the bursts share ONE chain, so the
        # cluster never holds more heavy tasks at once than a single scene
        # would. Where bursts overlap the two models disagree over the shared
        # ground, exactly as two per-burst trends do; union=True is the way
        # to ask for one answer there.
        from .Batch import Batch
        chain = _Fit3dChain()
        out = {}
        for k in self.keys():
            out.update(type(self)({k: self[k]})._fit3d_union(
                chain=chain, tag=f' [{k}]', **_kw).items())
        return Batch(out)



    def _fit3d_setup(self, threshold, window, cell, baseline, level,
                     max_dh, max_dv, step_dh, step_dv, max_seasonal,
                     consensus, err_dh, err_dv, iterations, debug,
                     chain=None, tag=''):
        """WHAT EVERY fit3d() DRIVER SHARES: the polarisation, the windows,
        the cluster's shape, each burst's frame and the kwargs the stage
        functions take, and the scene lattice the bursts sit on.

        One place, so the union driver and the per-chunk driver cannot read
        the same stack two ways. Returns a dict; `_fit3d_scan` adds the
        pass-1 graph to it.
        """
        import os as _os
        import numpy as np
        import xarray as xr
        import dask as _dask
        import dask.array as da
        from .Batch import Batch
        from . import utils_arcs
        from .utils_dask import get_dask_chunk_size_mb

        _pols = [v for ds in self.values() for v in ds.data_vars
                 if ds[v].dtype.kind == 'c' and 'y' in ds[v].dims]
        if len(set(_pols)) != 1:
            raise ValueError(f"fit3d() takes ONE polarisation, this stack "
                             f"carries {sorted(set(_pols))}.")
        pol = sorted(set(_pols))[0]
        # THE INPUT CHUNKS STATE THE SIZE. arcs() takes the same number the
        # same way: a caller who wants the work blocked differently rechunks
        # the stack, and a second knob saying the same thing could only
        # contradict it.
        budget_mb = get_dask_chunk_size_mb()
        wy, wx, pey, pex = utils_arcs._3d_windows(window)
        utils_arcs._3d_check_window_cell(wy, wx, cell, 'fit3d')

        # THE CLUSTER STATES THE SHAPE, as the per-burst path reads it
        _slots = 1
        _cores = max(1, _os.process_cpu_count() or 1)
        try:
            from dask.distributed import get_client as _gc
            _winfo = _gc().scheduler_info().get('workers', {})
            if _winfo:
                _slots = len(_winfo)
                _decl = [w.get('resources', {}).get('cpu') for w in
                         _winfo.values()]
                _decl = [d for d in _decl if d]
                if _decl:
                    _cores = int(max(_decl))
        except (ValueError, ImportError):
            pass
        _threads = max(1, _cores // max(1, _slots))
        _width = _slots
        if chain is None:
            chain = _Fit3dChain(_width)
        elif chain.width is None:
            chain.width = int(_width)

        # EARLIEST BURST FIRST, so the later one wins the seam it shares
        _keys = sorted(self.keys(), key=lambda k: np.asarray(
            self[k].coords['date'].values).min())
        _dss = [self[k] for k in _keys]

        # EVERY CHUNK ANSWERS TO ITS OWN BURST'S GEOMETRY. Averaging the
        # wavelength, the elevation phase and the baseline across bursts
        # describes no acquisition that ever happened: each is recorded per
        # burst and is exactly known, so a chunk is fitted with the numbers its
        # own burst carries. The network is the one stage that spans bursts,
        # and it takes the EARLIEST burst's frame -- a real geometry rather
        # than a mean of several -- which is also the burst the datum rests on.
        def _scalar(v):
            a = np.asarray(v, dtype=float).ravel()
            return float(a[0]) if a.size == 1 else float(np.mean(a))
        _ep = Batch._elevation_phase_approximate(self)

        def _frame(key, ds):
            lam_ = _scalar(ds['radar_wavelength'].values)
            bp_ = (np.asarray(ds[baseline].values, float).ravel()
                   if baseline and baseline in ds.data_vars else None)
            yv_ = np.asarray(ds['y'].values, dtype=float)
            xv_ = np.asarray(ds['x'].values, dtype=float)
            return dict(
                date_values=np.asarray(ds.coords['date'].values),
                spacing=(abs(float(yv_[1] - yv_[0])) if yv_.size > 1 else 1.0,
                         abs(float(xv_[1] - xv_[0])) if xv_.size > 1 else 1.0),
                bperp=bp_,
                geometry=(lam_, (4.0 * np.pi / lam_) / _ep[key]))

        # THE SCENE LATTICE. The bursts are geocoded on one ground grid, so a
        # node's place in the scene is its place in its own burst plus a whole
        # number of pixels -- no resampling, and nothing to interpolate.
        yv0 = np.asarray(_dss[0]['y'].values, dtype=float)
        xv0 = np.asarray(_dss[0]['x'].values, dtype=float)
        dy = float(yv0[1] - yv0[0]) if yv0.size > 1 else 1.0
        dx = float(xv0[1] - xv0[0]) if xv0.size > 1 else 1.0
        spacing = (abs(dy), abs(dx))
        _y0s = [float(np.asarray(d['y'].values, dtype=float)[0]) for d in _dss]
        _x0s = [float(np.asarray(d['x'].values, dtype=float)[0]) for d in _dss]
        y_org = max(_y0s) if dy < 0 else min(_y0s)
        x_org = max(_x0s) if dx < 0 else min(_x0s)

        # the cascade gates a DS candidate on how many in-window arcs it
        # holds, which is the same count the rest of the solve answers to
        _ma_ds = utils_arcs._3d_consensus(consensus)
        _common = dict(
            window=(wy, wx, pey, pex), threshold=float(threshold),
            min_agreeing=int(_ma_ds),
            cell=tuple(cell), budget=budget_mb,
            level=int(level), max_dh=float(max_dh), max_dv=float(max_dv),
            step_dh=float(step_dh), step_dv=float(step_dv),
            max_seasonal=float(max_seasonal),
            consensus=int(_ma_ds),
            err_dh=float(err_dh), err_dv=float(err_dv),
            iterations=int(iterations), debug=bool(debug), tag=str(tag))
        _kw_of = {k: dict(_common, **_frame(k, ds))
                  for k, ds in zip(_keys, _dss)}
        # the network spans the bursts; it answers to the earliest one's
        _kw_net = _kw_of[_keys[0]]
        date_values = _kw_net['date_values']
        bp = _kw_net['bperp']
        return dict(pol=pol, keys=_keys, dss=_dss, kw_of=_kw_of, kw_net=_kw_net,
                    cores=_cores, threads=_threads, width=_width,
                    chain=chain, tag=str(tag),
                    window=(wy, wx, pey, pex), origin=(y_org, x_org),
                    step=(dy, dx), date_values=date_values, bperp=bp,
                    budget=budget_mb)

    def _fit3d_scan(self, threshold, window, cell, baseline, level,
                    max_dh, max_dv, step_dh, step_dv, max_seasonal,
                    consensus, err_dh, err_dv, iterations, debug,
                    chain=None, tag=''):
        """PASS 1, SHARED: every chunk scanned on the scene lattice.

        This is the first half of fit3d()'s union driver, and the whole of
        what arcs() needs: the graph that reads each chunk with its halo,
        scans it for the DS rank raster and the PS-candidate winner grid, and
        names where each winner grid sits on ONE scene lattice. fit3d() goes
        on to the PS test, the network and the levels; arcs() stops at the
        PS test and returns the two rasters. ONE driver, so the debug tool
        sees exactly the candidates the delivery sees -- a second copy of
        this loop drifted in every detail it was not kept up with.

        Returns a dict: `keys` (earliest burst first), `dss`, `pol`, `grid`
        (per key: rows, cols, the array, the dataset), `blocks` (one entry
        per chunk, see the loop), `kw_of` (per key), `kw_net`, `cores`,
        `threads`, `width`, `date_values`, `bperp`.
        """
        import numpy as np
        import dask as _dask
        import dask.array as da
        from . import utils_arcs
        _su = self._fit3d_setup(threshold, window, cell, baseline, level,
                                max_dh, max_dv, step_dh, step_dv, max_seasonal,
                                consensus, err_dh, err_dv, iterations, debug,
                                chain=chain, tag=tag)
        pol, _keys, _dss = _su['pol'], _su['keys'], _su['dss']
        _kw_of, _kw_net = _su['kw_of'], _su['kw_net']
        _cores, _threads, _width = _su['cores'], _su['threads'], _su['width']
        chain = _su['chain']
        wy, wx, _pey, _pex = _su['window']
        y_org, x_org = _su['origin']
        dy, dx = _su['step']
        date_values, bp = _su['date_values'], _su['bperp']

        # ---- PASS 1: the cascade scan, one chunk at a time ---------------
        # Each chunk is read WITH A FULL DS WINDOW OF HALO, as arcs() reads
        # it: the owned pixels' own windows have to be complete, and the cells
        # the chunk owns must be able to see every pixel that could win them.
        # What comes back is the owned rank raster and the chunk's winner
        # grid -- one candidate per independence cell.
        _blocks = []
        _grid = {}
        _hy2, _hx2 = wy // 2, wx // 2
        for key, ds in zip(_keys, _dss):
            da_xr = ds[pol]
            if da_xr.dims[0] != 'date':
                da_xr = da_xr.transpose('date', ...)
            # NOT RECHUNKED HERE. Every window below is read through
            # `_Fit3dSlice`, which assembles the blocks it is given inside the
            # fit task; forcing the date axis into one chunk first only adds a
            # merge layer to the graph and materialises the window twice. The
            # caller's chunking is the caller's to state.
            dsk = da_xr.data
            _ny, _nx = dsk.shape[1], dsk.shape[2]
            yv = np.asarray(ds['y'].values, dtype=float)
            xv = np.asarray(ds['x'].values, dtype=float)
            boy = int(round((float(yv[0]) - y_org) / dy)) if yv.size else 0
            box = int(round((float(xv[0]) - x_org) / dx)) if xv.size else 0
            _cy, _cx = dsk.chunks[1], dsk.chunks[2]
            _y0 = np.r_[0, np.cumsum(_cy)][:-1]
            _x0 = np.r_[0, np.cumsum(_cx)][:-1]
            _grid[key] = (len(_cy), len(_cx), da_xr, ds)
            for _i in range(len(_cy)):
                gy0, gy1 = int(_y0[_i]), int(_y0[_i]) + int(_cy[_i])
                ya, yb = max(0, gy0 - wy), min(_ny, gy1 + wy)
                for _j in range(len(_cx)):
                    gx0, gx1 = int(_x0[_j]), int(_x0[_j]) + int(_cx[_j])
                    xa, xb = max(0, gx0 - wx), min(_nx, gx1 + wx)
                    _gate = chain.gate()
                    # ONE LATTICE, THE SCENE'S. The scan cuts its cells on
                    # the lattice of the origin it is handed, and the origin
                    # is handed in scene pixels below -- so the part's first
                    # cell is exact integer arithmetic, never a rounded
                    # burst-lattice reconstruction: a burst whose origin is
                    # not a multiple of the half-window would land its whole
                    # grid up to half a cell off, duplicating nodes at seams.
                    # ON THE PS LATTICE: _cascade_pass1 lays its winner grid
                    # on `_3d_ps_lattice(cell)` and names the first cell as
                    # ceil(scene_origin / lattice); the merge in _fit3d_select_ps
                    # places the grid by THIS cell index, so any other unit
                    # here puts every block at the wrong offset and lets the
                    # next block overwrite part of the previous one -- a hole
                    # through the middle of the scene and thinned chunk edges
                    _pl = utils_arcs._3d_ps_lattice(cell)
                    _cell = (-(-(boy + gy0) // _pl[0]),
                             -(-(box + gx0) // _pl[1]))
                    _owned = (gy0 - ya, gy1 - ya, gx0 - xa, gx1 - xa)
                    # nout: the rank raster and the winner grid leave the scan
                    # as SEPARATE keys, so the one level-1 task depends on the
                    # winners alone. Returned as one value they travel
                    # together, and every block's raster would be shipped to
                    # the worker that runs the PS test and never read there.
                    _part = _dask.delayed(_fit3d_scan_for_dask, nout=4)(
                        _Fit3dSlice(dsk[:, ya:yb, xa:xb]), _owned,
                        (boy + ya, box + xa),
                        _cell, _kw_of[key], _threads, _gate)
                    chain.push(_part[2])
                    _sub = dsk[:, gy0:gy1, gx0:gx1]
                    # LEVEL 2 READS WIDER THAN IT WRITES. Its partners are
                    # level-1 nodes another chunk owns and solved; the values
                    # come from that chunk's table, but the ARC still needs
                    # their phasor series, and only a haloed read has them.
                    # Candidates stay owned-only, so nothing in the halo is
                    # ever attached here -- it is there to be attached TO.
                    #
                    # ITS OWN, NARROWER SLICE. The scan above needs a FULL
                    # window per side -- a winner cell reaches half a window
                    # past the owned edge and gating those pixels needs their
                    # windows whole -- but level 2 only reaches +-wy//2, the
                    # window box. Sharing the scan's slice made every level
                    # read twice the halo it uses, on every block of every
                    # level, which is pure memory.
                    _hy, _hx = max(wy // 2, 1), max(wx // 2, 1)
                    _ya2, _yb2 = max(0, gy0 - _hy), min(_ny, gy1 + _hy)
                    _xa2, _xb2 = max(0, gx0 - _hx), min(_nx, gx1 + _hx)
                    _own2 = (gy0 - _ya2, gy1 - _ya2, gx0 - _xa2, gx1 - _xa2)
                    _blocks.append((key, _sub, (gy0 + boy, gx0 + box),
                                    int(_cy[_i]), int(_cx[_j]), _part,
                                    _i, _j, _cell,
                                    dsk[:, _ya2:_yb2, _xa2:_xb2],
                                    (_ya2 + boy, _xa2 + box), _own2))

        return dict(_su, grid=_grid, blocks=_blocks)

    def _fit3d_union(self, threshold, window, cell, baseline, level,
                     max_dh, max_dv, step_dh, step_dv, max_seasonal,
                     consensus, err_dh, err_dv, iterations, debug,
                     chain=None, tag=''):
        """One network over the union of the bursts, returned on the burst grid.

        A node's partners are whatever lies inside the window, and a burst edge
        is not a fact about the ground: solved per burst, a node near the seam
        reaches only half its neighbourhood and the two bursts answer the same
        question from different networks. Merged, the network crosses the seam.

        ONLY THE NODES ARE MERGED, NEVER THE RASTERS. Each block is scanned
        where it is stored and yields a node table -- some thousands of phasor
        columns -- and those tables are all the shared solve ever sees. Merging
        the stacks instead, into one array over the scene, makes every block as
        wide as the scene: the chunking the caller asked for stops applying,
        and the host pays tens of gigabytes to carry a network that weighs
        megabytes. arcs() unions its winner grids the same way, for the same
        reason.

        Only the SOLVE is unioned. The model comes back on each burst's own
        grid, carrying only that burst's pixels, so nothing downstream sees a
        different geometry than it handed in.
        """
        import numpy as np
        import xarray as xr
        import dask as _dask
        import dask.array as da
        from .Batch import Batch
        _s = self._fit3d_scan(threshold, window, cell, baseline, level,
                              max_dh, max_dv, step_dh, step_dv, max_seasonal,
                              consensus, err_dh, err_dv, iterations, debug,
                              chain=chain, tag=tag)
        _keys, _grid, _blocks = _s['keys'], _s['grid'], _s['blocks']
        chain = _s['chain']
        _kw_of, _kw_net = _s['kw_of'], _s['kw_net']
        _cores, _width = _s['cores'], _s['width']
        date_values, bp = _s['date_values'], _s['bperp']
        # ---- LEVEL 1: the PS test over the WHOLE scene's candidates ------
        # The winner grids are merged onto one cell lattice and tested
        # together. A chunk here is a fraction of the PS extent wide, so a
        # chunk-local test asks each candidate about partners that mostly do
        # not exist in its chunk; on the merged grid it is asked about all of
        # them. The grid is the raster one pyramid level up, so the scene's
        # whole candidate set is megabytes and this is one in-memory task.
        _net = _dask.delayed(_fit3d_level1_for_dask)(
            [((b[5][1], b[5][2], b[5][3]), b[8]) for b in _blocks], _kw_net,
            _cores)

        # ---- PASS 2: the shared network written onto each burst's grid ----
        # THE THREAD BUDGET IS PER CONCURRENT TASK, NOT PER SLOT. This pass
        # submits one task per block and the gate runs at most `_width` of them
        # at once, so when there are fewer blocks than slots the cores divided
        # among the slots are divided among tasks that do not exist and the
        # rest of the machine idles. Attachment is the fitting-heavy stage, so
        # that idleness is the wall clock.
        _conc2 = max(1, min(int(_width) if _width else len(_blocks),
                            len(_blocks)))
        _threads2 = max(1, _cores // _conc2)
        _cells = {}
        _l2in, _l2tab = [], []
        for (key, _sub, _origin, _ny, _nx, _part, _i, _j, _c,
             _hsub, _horg, _hown) in _blocks:
            _kwb = _kw_of[key]
            _gate = chain.gate()
            # LEVEL 1 FIRST, EVERYWHERE. Its nodes are the fixed layer the
            # next stage stands on, and a chunk needs the ones its neighbours
            # own, so they have to be finished before level 2 starts. The
            # table is a few megabytes; the planes stay put. It emits even at
            # level=1, where nothing consumes the nodes, because the table is
            # also how the level's numbers reach its report.
            # THE CALLER'S LEVEL, CAPPED AT ONE. This stage IS level 1, and
            # levels >= 2 are the driver's own pass below, over the finished
            # tables. Capping instead of hardcoding keeps `level=0` meaning
            # what it documents: the PS network alone, every other pixel
            # NaN. Hardcoded to 1, union=True attached DS even when the
            # caller asked for none.
            _o = _dask.delayed(_fit3d_attach_for_dask, nout=2)(
                _Fit3dSlice(_sub), _part[0], _net,
                dict(_kwb, level=min(int(level), 1)),
                _origin, _threads2, _gate, emit_nodes=True)
            chain.push(_o[0])
            _l2tab.append(_o[1])
            if int(level) >= 2:
                _l2in.append((key, _hsub, _horg, _hown, _ny, _nx, _part,
                              _i, _j, _o[0]))
            else:
                _cells[(key, _i, _j)] = da.from_delayed(
                    _o[0], shape=(6, _ny, _nx), dtype=np.complex64)

        # ---- PASS 3: LEVEL 2, over the FINISHED level-1 nodes ------------
        # Level 1 is complete before this starts, so its values are fixed input
        # and are never recomputed here. Each chunk keeps the nodes within one
        # DS window of its own bounds -- the whole reach of a level-2 candidate
        # -- and then solves its own local clusters alone. Every cluster is
        # anchored to that fixed layer, so one split by a seam comes out
        # consistent on both sides and no chunk has to agree with another.
        # ONE PASS PER LEVEL, WITH A MERGE BETWEEN THEM. Each level stands on
        # the COMPLETE previous network, not on the part of it its own chunk
        # happened to find: the tables from every chunk are gathered before the
        # next level starts, so a candidate near a seam is offered the nodes its
        # neighbour attached in the round before. Repeating inside a chunk
        # instead would starve exactly those pixels, round after round.
        # THE LEVEL-1 REPORT, once every block of it has finished. At
        # level=1 nothing consumes the tables, so the report is hung off a
        # plane -- the only way to tell the graph to run it at all.
        _rep1 = _dask.delayed(_fit3d_level_report)(1, _l2tab, bool(debug),
                                                   str(tag))
        if int(level) < 2:
            _kk0 = next(iter(_cells))
            _ny0, _nx0 = _cells[_kk0].shape[1], _cells[_kk0].shape[2]
            _cells[_kk0] = da.from_delayed(
                _dask.delayed(_fit3d_keep)(_cells[_kk0], _rep1),
                shape=(6, _ny0, _nx0), dtype=np.complex64)
        _cur, _tabs = _l2in, ([_rep1] if int(level) >= 2 else [])
        for _lv in range(2, int(level) + 1):
            _nxt, _new3 = [], []
            _last = (_lv == int(level))
            for (key, _hsub, _horg, _hown, _ny, _nx, _part, _i, _j, _pl) \
                    in _cur:
                _g3 = chain.gate()
                # EVERY LEVEL EMITS, including the last: the table is the
                # channel its report travels on, and the last level deserves
                # a report as much as the others. Only `_tabs` stops growing.
                _o3 = _dask.delayed(_fit3d_ds_attach_for_dask, nout=2)(
                    _Fit3dSlice(_hsub), _part[0], _net, _pl, _tabs,
                    _kw_of[key], _horg, _hown, _threads2, _g3,
                    emit_nodes=True, level_id=_lv)
                _pl2 = _o3[0]
                chain.push(_pl2)
                _new3.append(_o3[1])
                _nxt.append((key, _hsub, _horg, _hown, _ny, _nx, _part,
                             _i, _j, _pl2))
                _cells[(key, _i, _j)] = da.from_delayed(
                    _pl2, shape=(6, _ny, _nx), dtype=np.complex64)
            # THE MERGE, AFTER THE WHOLE LEVEL AND NOT DURING IT. Extending
            # the table inside the block loop would hand block k the nodes
            # blocks 1..k-1 attached in the SAME level -- a partial network
            # that depends on the order the blocks were built, and one that
            # makes every block wait for the block before it. A level stands
            # on the COMPLETE level before it or it is not a level.
            _cur = _nxt
            # ONE REPORT FOR THE LEVEL, on the path the next level already
            # waits for. The last level has nothing after it, so its report is
            # hung off a plane instead -- otherwise it would never run.
            _rep = _dask.delayed(_fit3d_level_report)(_lv, _new3,
                                                      bool(debug), str(tag))
            if not _last:
                _tabs = _tabs + [_rep]
            elif _nxt:
                _k0, _s0, _o0, _w0, _n0, _x0, _p0, _i0, _j0, _q0 = _nxt[0]
                _cells[(_k0, _i0, _j0)] = da.from_delayed(
                    _dask.delayed(_fit3d_keep)(_q0, _rep),
                    shape=(6, _n0, _x0), dtype=np.complex64)

        model_result = {}
        for key in _keys:
            _nr, _nc, da_xr, ds = _grid[key]
            both = da.concatenate(
                [da.concatenate([_cells[(key, i, j)] for j in range(_nc)],
                                axis=2) for i in range(_nr)], axis=1)
            model_result[key] = _fit3d_model(ds, da_xr, both, date_values, bp)
        return Batch(model_result)

    def _fit3d_candidates(self, threshold, window, cell, baseline,
                          consensus, iterations, union, debug):
        """The DS rank raster and the PS test, per burst, BEFORE any network.

        This is what arcs() returns: pass 1 exactly as fit3d() runs it, then
        the PS test over the merged winner grid, written back to each burst's
        pixels. `union=False` scans each burst as its own scene; `union=True`
        tests the bursts together on one lattice, the later burst's winner
        taking a cell both hold, and writes every winner inside a burst's
        grid onto it, as fit3d() writes its nodes.
        """
        import numpy as np
        import xarray as xr
        import dask as _dask
        import dask.array as da
        from .Batch import Batch, Batches
        groups = ([self] if union else
                  [type(self)({k: self[k]}) for k in self.keys()])
        ds_out, ps_out = {}, {}
        chain = _Fit3dChain()
        for grp in groups:
            _s = grp._fit3d_scan(threshold, window, cell, baseline, 0,
                                 25.0, 25.0, 8.0, 2.0, 0.0, consensus,
                                 4.0, 1.0, iterations, debug, chain=chain,
                                 tag='' if union else f' [{list(grp.keys())[0]}]')
            _keys, _grid, _blocks = _s['keys'], _s['grid'], _s['blocks']
            _test = _dask.delayed(_fit3d_select_ps)(
                [((b[5][1], b[5][2], b[5][3]), b[8]) for b in _blocks],
                _s['kw_net'], _s['cores'])
            for key in _keys:
                _nr, _nc, da_xr, ds = _grid[key]
                rank = [[None] * _nc for _ in range(_nr)]
                ps = [[None] * _nc for _ in range(_nr)]
                for b in _blocks:
                    if b[0] != key:
                        continue
                    _, _sub, _origin, _ny, _nx, _part, _i, _j = b[:8]
                    rank[_i][_j] = da.from_delayed(
                        _part[0], shape=(_ny, _nx), dtype=np.float32)
                    ps[_i][_j] = da.from_delayed(
                        _dask.delayed(_fit3d_ps_raster_for_dask)(
                            _test, int(_origin[0]), int(_origin[1]),
                            int(_ny), int(_nx)),
                        shape=(_ny, _nx), dtype=np.float32)
                coords = {k_: v for k_, v in da_xr.coords.items()
                          if k_ in ('y', 'x', 'spatial_ref')}
                pol = _s['pol']
                for store, rows in ((ds_out, rank), (ps_out, ps)):
                    o = xr.Dataset({pol: xr.DataArray(
                        da.block(rows), dims=('y', 'x'), coords=coords,
                        name=pol)})
                    if 'spatial_ref' in ds.coords:
                        o = o.assign_coords(spatial_ref=ds.spatial_ref)
                    store[key] = o
        return Batches((Batch(ds_out), Batch(ps_out)))

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

    def real(self, **kwargs):
        """
        Return the real part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            # ds.map() applies the lambda to each DataArray in the Dataset
            ds_real = ds.map(lambda da: da.real, **kwargs)
            out[key] = ds_real
        return Batch(out)

    def imag(self, **kwargs):
        """
        Return the imaginary part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            ds_imag = ds.map(lambda da: da.imag, **kwargs)
            out[key] = ds_imag
        return Batch(out)

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

        BatchCore._require_lazy(self, 'threshold')

        results = {}
        for burst_id, burst_ds in self.items():
            burst_weight = weight[burst_id] if weight is not None else None
            filtered_vars = {}
            for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                data_da = burst_ds[pol]
                weight_da = burst_weight[pol] if burst_weight is not None else None

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

            filtered_ds = burst_ds.assign(filtered_vars)
            results[burst_id] = filtered_ds

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

        # Select, rename date→pair, and assign pair coords matching the caller
        n_pairs = len(ref_idx)
        pair_coords = np.arange(n_pairs)
        screen_ref = self.isel(date=ref_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))
        screen_rep = self.isel(date=rep_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))

        return screen_ref, screen_rep

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            # select only the vars whose dtype is complex
            complex_vars = [
                var for var in ds.data_vars
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray. The 1D radar
            # metadata rides along afterwards -- angle() of near_range is
            # meaningless, but DROPPING it strands every downstream unit
            # conversion, which is how radar_wavelength went missing.
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            meta = [v for v in ds.data_vars
                    if v not in complex_vars and ds[v].ndim <= 1]
            if meta:
                ds_phase = ds_phase.assign({v: ds[v] for v in meta})
            ds_phase.attrs = ds.attrs
            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def unwrap2d(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d(*args, **kwargs)

    def unwrap2d_irls(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d_irls(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot complex phase as wrapped phase via .angle() conversion.
        """
        return self.angle().plot(*args, **kwargs
        )

    @staticmethod
    @serialize_gpu
    def _goldstein(phase_np, corr_np, psize=32, threshold=0.5, device='auto'):
        """
        Apply Goldstein adaptive filter.

        Uses loop-based processing for CPU (constant memory) and
        PyTorch unfold/fold for GPU (vectorized).

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
        device : str, optional
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        np.ndarray
            Filtered complex array with same shape as input.
        """
        import numpy as np
        from .BatchCore import BatchCore
        from .utils_goldstein import goldstein_numpy, goldstein_pytorch

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

        # Ensure correct dtypes (goldstein functions require complex64/float32)
        if phase_np.dtype != np.complex64:
            phase_np = phase_np.astype(np.complex64)
        if corr_np.dtype != np.float32:
            corr_np = corr_np.astype(np.float32)

        # Dispatch based on device
        dev = BatchCore._get_torch_device(device)

        if dev.type == 'cpu':
            result = goldstein_numpy(phase_np, corr_np, psize_y, psize_x, threshold=threshold)
        else:
            import torch
            result = goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, dev, threshold=threshold)
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

        if squeeze:
            result = result[np.newaxis, ...]
        return result

    def goldstein(self, corr: BatchUnit, window: int | dict[str, int] = 32, threshold: float = 0.5,
                  device: str = 'auto', debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.

        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        window : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        threshold : float, optional
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
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

        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")

        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")

        # Validate lazy data
        BatchCore._require_lazy(self, 'goldstein')

        if isinstance(window, int):
            window = {'y': window, 'x': window}
        elif isinstance(window, (tuple, list)):
            window = {'y': window[0], 'x': window[1]}

        # Resolve device ONCE here, not in every task
        if device == 'auto':
            resolved_device = BatchCore._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            corr_ds = corr[k]
            filtered_vars = {}

            # Process each complex data variable in the dataset
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    corr_da = corr_ds[var_name]
                    phase_dask = var_data.data
                    corr_dask = corr_da.data

                    # Require first dimension chunked as 1 (avoid hidden rechunking overhead)
                    chunks = phase_dask.chunks
                    if var_data.ndim == 3 and chunks[0][0] != 1:
                        raise ValueError(
                            f"goldstein() requires first dimension chunked as 1, got chunks {chunks[0]}. "
                            f"Data should already have pair=1 chunks from load()."
                        )

                    # Calculate overlap depth: window//2 + 2 (PyGMTSAR formula)
                    depth_y = window['y'] // 2 + 2
                    depth_x = window['x'] // 2 + 2

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
                        device=device,
                    )

                    filtered_vars[var_name] = xr.DataArray(
                        filtered_dask,
                        dims=var_data.dims,
                        coords=var_data.coords
                    )
                else:
                    filtered_vars[var_name] = var_data

            # Create a new dataset with the filtered variables
            result[k] = xr.Dataset(
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
                 debug: bool = False, **kwargs):
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
                                       debug=debug, wrapper=Batches)

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
            out = {}
            for key in b:
                corr_ds = corr[key]
                corr_var = next(v for v in corr_ds.data_vars if 'y' in corr_ds[v].dims)
                corr_da = corr_ds[corr_var]
                mask = corr_da.mean('pair') >= threshold if 'pair' in corr_da.dims else corr_da >= threshold
                out[key] = b[key].where(mask)
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

    def goldstein(self, window: int | list[int, int] = 32, threshold: float = 0.5, device: str = 'auto'):
        """Apply Goldstein filter to phase using correlation as weight.

        Expects Batches with [BatchComplex (phase), BatchUnit (correlation)].

        Parameters
        ----------
        window : int or list[int, int]
            Goldstein filter patch size, default 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with Goldstein-filtered phase and unchanged correlation.

        Examples
        --------
        >>> phase, corr = stack.phasediff(pairs, wavelength=30).goldstein(32).angle()
        """
        if len(self) < 2:
            raise ValueError("goldstein() requires Batches with at least 2 elements: [phase, correlation]")

        phase, corr = self[0], self[1]

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"First element must be BatchComplex, got {type(phase).__name__}")
        if not isinstance(corr, BatchUnit):
            raise TypeError(f"Second element must be BatchUnit, got {type(corr).__name__}")

        filtered_phase = phase.goldstein(corr, window, threshold=threshold, device=device)
        return Batches([filtered_phase, corr] + list(self[2:]))

    def interferogram(self,
                  weight: 'BatchUnit | None' = None,
                  phase: 'BatchComplex | None' = None,
                  wavelength: float | None = None,
                  gaussian_threshold: float = 0.5,
                  device: str = 'auto') -> 'Batches':
        """
        Compute phase difference from paired SLC data.

        Expects Batches from pairs() with [ref, rep] BatchComplex objects.

        Parameters
        ----------
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking.
        phase : BatchComplex or None
            Optional phase to subtract (e.g., topographic phase).
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with [phase, correlation].

        Examples
        --------
        >>> ref, rep = stack.pairs(baseline.tolist())
        >>> phase, corr = ref.interferogram(rep, wavelength=30)
        >>> # Or chained:
        >>> phase, corr = stack.pairs(baseline.tolist()).interferogram(wavelength=30)
        """
        if len(self) != 2:
            raise ValueError("interferogram() requires Batches with exactly 2 elements: [ref, rep]")

        ref, rep = self[0], self[1]

        if not isinstance(ref, BatchComplex) or not isinstance(rep, BatchComplex):
            raise TypeError("Both elements must be BatchComplex")

        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(stack.from_dataset(data)) to convert a single DataArray.'
            )

        intf = ref * rep.conj()
        if phase is not None:
            if isinstance(phase, BatchComplex):
                intf = intf * phase
            else:
                intf = intf * phase.iexp(-1)

        if wavelength is not None:
            intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
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

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        import numpy as np
        from insardev_toolkit.progressbar import progressbar

        # Get all burst keys (should be same across all batches)
        keys = list(self[0].keys())
        n_batches = len(self)

        # Save input chunk structure per batch per burst
        all_input_chunks = []  # list of {burst_key: {var_name: chunks_dict}}
        for batch in self:
            batch_chunks = {}
            for key, ds in batch.items():
                ic = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if hasattr(arr.data, 'chunks'):
                        ic[var_name] = dict(zip(arr.dims, arr.data.chunks))
                batch_chunks[key] = ic
            all_input_chunks.append(batch_chunks)

        # Persist all batches at once — single scheduler submission
        # progressbar extracts futures and blocks until completion
        all_dicts = [dict(batch) for batch in self]
        all_results = list(dask.persist(*all_dicts))
        progressbar(all_results, desc='Computing bursts'.ljust(25))

        # Finalize: materialize coordinates and rechunk to match input
        computed_batches = []
        for bi in range(n_batches):
            result = all_results[bi]
            computed = {}
            for key, ds in result.items():
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)
                input_chunks = all_input_chunks[bi][key]
                rechunked_vars = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if var_name in input_chunks:
                        chunks = input_chunks[var_name]
                        if isinstance(arr.data, np.ndarray):
                            arr = arr.chunk(chunks)
                        elif hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks:
                            arr = arr.chunk(chunks)
                        rechunked_vars[var_name] = arr
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                computed[key] = ds
            computed_batches.append(type(self[bi])(computed))
        return Batches(computed_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, union=False,
                 device='auto', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

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
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = Batches([phase, corr]).unwrap2d()  # With weights
        """
        if len(self) < 1:
            raise ValueError("unwrap2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, union=union,
                              device=device, debug=debug, **kwargs)

    def unwrap2d_chunk(self, overlap=None, device='auto', debug=False, **kwargs):
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Unlike unwrap2d() which requires a single spatial chunk, this method
        unwraps each spatial chunk independently with overlap margins.

        Parameters
        ----------
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batches
            Batches with [unwrapped_phase, weight] preserving original types.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline).interferogram(wavelength=30).angle()
        >>> unwrapped, corr = phase.chunk2d('128MiB').unwrap2d_chunk()
        """
        if len(self) < 1:
            raise ValueError("unwrap2d_chunk() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        unwrapped = phase.unwrap2d_chunk(weight=weight, overlap=overlap,
                                          device=device, debug=debug, **kwargs)

        elements = [unwrapped] + list(self[1:])
        return Batches(elements)



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

        Applies Batch.displacement_los() to the first element.

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

        Pixels with cstd >= threshold are set to NaN. Uses correlation
        weights from the second element if available.

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
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

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
            The model: velocity, height, seasonal, coherence, rmse. Identical
            in name and unit to fit3d()'s, so predict() consumes either.
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None
        # BatchComplex.fit1d() normalises every sample to a unit phasor, so a
        # magnitude weight has nothing to act on and it takes none -- complex
        # data carries no correlation. Only the unwrapped fit is offered one.
        if weight is None or isinstance(phase, BatchComplex):
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
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        rmse_result = phase.rmse(solution, weight=weight)

        if weight is not None:
            # Reduce weight to (y, x) — detect temporal dimension
            w_sample_ds = next(iter(weight.values()))
            w_spatial = [v for v in w_sample_ds.data_vars if 'y' in w_sample_ds[v].dims]
            tdim = next((d for d in ('pair', 'date')
                         if w_spatial and d in w_sample_ds[w_spatial[0]].dims), None)
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
            Batch containing 'trend', 'seasonal', and 'resid' variables.

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
