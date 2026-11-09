# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------

"""
Dask utility functions for memory-efficient chunking.

Chunk size is controlled via Dask configuration:

    # Uses Dask config (default 128MB)
    psf = stack.psfunction()

    # Or set globally
    import dask.config
    dask.config.set({'array.chunk-size': '256MiB'})
    psf = stack.psfunction()  # Now uses 256MB
"""

import dask.config
from dask._task_spec import Task as _Task
import weakref as _weakref


def get_dask_chunk_size_mb() -> float:
    """
    Get the Dask chunk size in MiB from the 'array.chunk-size' configuration.

    A string is read with its unit ('128MiB', '256MB', '0.5MiB', '4 KiB'; MB and GB are read as MiB and GiB);
    a number or a string of digits is bytes. A value that cannot be read raises ValueError.

    Returns
    -------
    float
        Dask chunk size in MiB.

    >>> get_dask_chunk_size_mb()
    128.0
    """
    import re
    value = dask.config.get('array.chunk-size', '128MiB')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 2 ** 20
    m = re.match(r'^(\d+(?:\.\d*)?|\.\d+)\s*(GIB|GB|MIB|MB|KIB|KB|B)?$', str(value).strip().upper())
    if m is None:
        raise ValueError(f"Cannot read dask's array.chunk-size {value!r}. Use e.g. '128MiB', '256MB', '1GiB' or a number of bytes.")
    units = {'GIB': 1024.0, 'GB': 1024.0, 'MIB': 1.0, 'MB': 1.0, 'KIB': 1 / 1024, 'KB': 1 / 1024, 'B': 1 / 2 ** 20}
    return float(m.group(1)) * units[m.group(2) or 'B']


def cluster_cores():
    """
    Get the worker count and the CPU cores of the current Dask cluster.

    The cores are the 'cpu' resource the workers declare, the largest one, as the cluster
    states them (Client(resources={'cpu': N})). os.process_cpu_count() is asked only when
    no worker declares cores.

    Returns
    -------
    tuple of (int, int)
        (workers, cores): 1 worker without a Dask client or without workers.

    Examples
    --------
    >>> workers, cores = cluster_cores()
    """
    import os as _os
    _slots, _cores = 1, None
    try:
        from dask.distributed import get_client as _gc
        _winfo = _gc().scheduler_info().get('workers', {})
        if _winfo:
            _slots = len(_winfo)
            _decl = [w.get('resources', {}).get('cpu') for w in _winfo.values()]
            _decl = [d for d in _decl if d]
            if _decl:
                _cores = int(max(_decl))
    except (ValueError, ImportError):
        pass
    # THE DECLARED CORES FIRST, the process's own count only without them:
    # os.process_cpu_count() is new in Python 3.13
    if _cores is None:
        if not hasattr(_os, 'process_cpu_count'):
            raise RuntimeError(
                "Cannot count the CPU cores. Use Python 3.13+, or declare "
                "them in the Dask cluster: Client(..., resources={'cpu': N}).")
        _cores = max(1, _os.process_cpu_count() or 1)
    return _slots, _cores


def _compute_chunk_score(
    cy: int, cx: int,
    y_size: int, x_size: int,
    pixel_budget: int,
) -> float:
    """
    Compute objective score for a chunk configuration (lower is better).

    Goal: Find chunks that:
    1. Are close to budget (80-120%)
    2. Have uniform sizes (min chunk >= 50% of max chunk)
    3. Have balanced chunk counts (2x2 better than 4x1)
    4. Are near-square
    """
    import math

    # Chunk counts
    n_chunks_y = math.ceil(y_size / cy)
    n_chunks_x = math.ceil(x_size / cx)

    # Remainders
    has_y_remainder = (y_size % cy) != 0
    has_x_remainder = (x_size % cx) != 0
    ry = y_size % cy if has_y_remainder else cy
    rx = x_size % cx if has_x_remainder else cx

    # Chunk pixel counts
    full_chunk = cy * cx
    y_edge_chunk = ry * cx if has_y_remainder else full_chunk
    x_edge_chunk = cy * rx if has_x_remainder else full_chunk
    corner_chunk = ry * rx if (has_y_remainder and has_x_remainder) else full_chunk

    min_chunk_pixels = min(full_chunk, y_edge_chunk, x_edge_chunk, corner_chunk)

    # Uniformity ratio (1.0 = all chunks same size)
    uniformity = min_chunk_pixels / full_chunk

    # === PENALTIES ===

    # P1: Uniformity - want min chunk >= 50% of full chunk
    if uniformity >= 0.5:
        uniformity_penalty = (1.0 - uniformity)  # 0 at 100%, 0.5 at 50%
    else:
        # High penalty below 50%
        uniformity_penalty = 0.5 + (0.5 - uniformity) * 10

    # P2: Budget utilization - want 80-120% of budget
    budget_ratio = full_chunk / pixel_budget
    if 0.8 <= budget_ratio <= 1.2:
        budget_penalty = abs(1.0 - budget_ratio)  # 0 at 100%, 0.2 at edges
    elif budget_ratio < 0.8:
        budget_penalty = 0.2 + (0.8 - budget_ratio) * 2
    else:
        budget_penalty = 0.2 + (budget_ratio - 1.2) * 5  # Harder penalty for over-budget

    # P3: Chunk count imbalance
    if n_chunks_y > 0 and n_chunks_x > 0:
        count_ratio = max(n_chunks_y, n_chunks_x) / min(n_chunks_y, n_chunks_x)
        count_penalty = max(0, (count_ratio - 1) * 0.1)
    else:
        count_penalty = 0

    # P4: Aspect ratio - prefer square
    aspect_ratio = max(cy / cx, cx / cy)
    aspect_penalty = max(0, (aspect_ratio - 1) * 0.05)

    # === WEIGHTS ===
    w_uniformity = 100   # Most important - no tiny corners
    w_budget = 10        # Budget utilization
    w_count = 1          # Chunk count balance
    w_aspect = 1         # Aspect ratio

    score = (
        w_uniformity * uniformity_penalty +
        w_budget * budget_penalty +
        w_count * count_penalty +
        w_aspect * aspect_penalty
    )

    return score


def rechunk2d(
    shape: tuple,
    element_bytes: int,
    input_chunks: tuple = None,
    target_mb: float = None,
    min_chunk: int = 64,
    max_chunk: int = 16384,
    merge: bool = False,
    floors: list = None,
) -> dict:
    """
    Compute optimal chunk sizes aligned to input chunk boundaries.

    When input_chunks is provided, output chunk boundaries never cross
    input chunk boundaries.  Each input chunk is individually split or
    kept whole to produce output chunks close to the memory budget.
    When merge=True, adjacent small chunks may be merged.
    Returns chunk tuples per dim.

    When input_chunks is not provided, falls back to computing a single
    optimal chunk size per dim (legacy behaviour).

    Parameters
    ----------
    shape : tuple
        Array shape: (y_size, x_size) for 2D or (n, y_size, x_size) for 3D.
    element_bytes : int
        Bytes per element (e.g., 8 for complex64, 4 for float32).
    input_chunks : tuple of tuples, optional
        Input chunk sizes per spatial dim: ((cy0, cy1, ...), (cx0, cx1, ...)).
    target_mb : float, optional
        Target memory per chunk in MiB. If None, uses dask config ('array.chunk-size').
    min_chunk : int
        Minimum chunk size (default 64).
    max_chunk : int
        Maximum chunk size (default 16384).
    merge : bool
        If True, adjacent small input chunks may be merged into larger
        output tiles.  If False (default), each input chunk is kept
        whole or split — never merged with neighbors.  False avoids
        cross-chunk rechunk graphs that explode in layered pipelines.
    floors : list, optional
        The fixed smallest tile is appended when no tile of min_chunk x min_chunk px fits 1.2 times the budget
        (26.67 KiB per date at 8 B per pixel): the largest tile (cy, cx) is appended. The caller prints the
        WARNING (warn_floor); rechunk2d prints nothing.

    Returns
    -------
    dict
        When input_chunks is provided:
            {'y': (cy0, cy1, ...), 'x': (cx0, cx1, ...)} — chunk tuples.
        When input_chunks is not provided:
            {'y': cy, 'x': cx} — single int per dim.
        For 3D input, also includes {0: 1} (legacy) or {0: -1}.
    """
    import math

    # Handle 3D arrays
    is_3d = len(shape) == 3
    if is_3d:
        n_dim, y_size, x_size = shape
    else:
        y_size, x_size = shape

    # Get budget
    if target_mb is None:
        target_mb = get_dask_chunk_size_mb()

    pixel_budget = (target_mb * 1024 * 1024) // element_bytes

    if input_chunks is not None:
        y_in, x_in = input_chunks
        max_cy_in = max(y_in)
        max_cx_in = max(x_in)

        # Split (and optionally merge) input chunks, aligned to boundaries
        def split_dim(in_chunks, target_size, allow_merge=merge):
            out_chunks = []
            accum = 0
            for c in in_chunks:
                if c >= target_size:
                    # Flush any accumulated small chunks first
                    if accum > 0:
                        out_chunks.append(accum)
                        accum = 0
                    # Split large chunk into n roughly equal sub-tiles
                    n = max(1, math.ceil(c / target_size))
                    base = c // n
                    remainder = c % n
                    for i in range(n):
                        out_chunks.append(base + (1 if i < remainder else 0))
                elif allow_merge:
                    # Merge adjacent small chunks until reaching target
                    if accum + c > target_size and accum > 0:
                        out_chunks.append(accum)
                        accum = c
                    else:
                        accum += c
                else:
                    # Keep each input chunk as-is (no merging)
                    out_chunks.append(c)
            if accum > 0:
                out_chunks.append(accum)
            return tuple(out_chunks)

        # Try multiple strategies and pick the one with fewest output tiles.
        # Each strategy accounts for both dims: keeping one dim's chunks
        # whole lets the other dim use a larger target (= fewer tiles).
        candidates = []

        # Strategy A: keep y input chunks whole, adjust x
        x_target_a = pixel_budget // max_cy_in
        if x_target_a >= min_chunk:
            yg = split_dim(y_in, max_cy_in)
            xg = split_dim(x_in, x_target_a)
            candidates.append((yg, xg))

        # Strategy B: keep x input chunks whole, adjust y
        y_target_b = pixel_budget // max_cx_in
        if y_target_b >= min_chunk:
            yg = split_dim(y_in, y_target_b)
            xg = split_dim(x_in, max_cx_in)
            candidates.append((yg, xg))

        # Strategy C: balanced split/merge
        c_floor = []
        target_c = _find_optimal_chunk_pair(
            y_size, x_size, pixel_budget, element_bytes, min_chunk, max_chunk, floors=c_floor)
        yg = split_dim(y_in, target_c['y'])
        xg = split_dim(x_in, target_c['x'])
        candidates.append((yg, xg))
        c_tiles = (yg, xg)

        # Pick strategy with fewest output tiles
        best_yg, best_xg = min(candidates, key=lambda c: len(c[0]) * len(c[1]))
        if floors is not None and c_floor and (best_yg, best_xg) == c_tiles:
            floors.append((max(best_yg), max(best_xg)))

        result = {'y': best_yg, 'x': best_xg}
        if is_3d:
            result[0] = 1
        return result

    # Legacy path: no input_chunks — return single int per dim
    # Edge case: array smaller than budget
    if y_size * x_size <= pixel_budget:
        result = {'y': y_size, 'x': x_size}
        if is_3d:
            result[0] = 1
        return result

    result = _find_optimal_chunk_pair(
        y_size, x_size, pixel_budget, element_bytes, min_chunk, max_chunk, floors=floors)
    if is_3d:
        result[0] = 1
    return result


def warn_floor(who: str, floors) -> None:
    """
    One WARNING line per call when rechunk2d made its fixed smallest tile (the budget per date is too small for a 64 x 64 px
    tile: 1.2 x the budget is below 4096 px): the largest tile of floors is named. warn_floor('chunk2d', [(35, 62)])
    """
    if floors:
        cy, cx = max(floors, key=lambda p: p[0] * p[1])
        print(f"WARNING: {who}(): the budget per date is too small for a 64 x 64 px tile (1.2 x the budget is below 4096 px): "
              f"the largest floor tile made is {cy} x {cx} px", flush=True)


def _find_optimal_chunk_pair(
    y_size: int, x_size: int,
    pixel_budget: int, element_bytes: int,
    min_chunk: int = 64, max_chunk: int = 16384,
    floors: list = None,
) -> dict:
    """Find optimal (cy, cx) chunk pair for a given array shape and budget; the fallback tile is appended to floors."""
    max_pixels = int(1.2 * pixel_budget)

    def get_candidates(dim_size):
        candidates = set()
        for d in range(1, int(dim_size**0.5) + 1):
            if dim_size % d == 0:
                if min_chunk <= d <= min(max_chunk, dim_size):
                    candidates.add(d)
                comp = dim_size // d
                if min_chunk <= comp <= min(max_chunk, dim_size):
                    candidates.add(comp)
        for c in range(min_chunk, min(dim_size, max_chunk) + 1, 64):
            candidates.add(c)
        if dim_size <= max_chunk:
            candidates.add(dim_size)
        return sorted(candidates, reverse=True)

    valid_y = get_candidates(y_size)
    valid_x = get_candidates(x_size)

    input_aspect = max(y_size / x_size, x_size / y_size)
    max_aspect = max(4, min(input_aspect, 16))

    # THE BUDGET COMES FIRST (N148). A chunk within 80-120 % of the budget wins over any uniform one below it: the last
    # chunk of a dimension may be partial, as no size divides a prime evenly and the uniform choice then fell to the 64-px
    # floor, 25 times below the budget. Among the chunks within the budget window the score decides (uniformity,
    # counts, shape); when the aspect limit leaves none, the limit goes: the budget is the contract, the shape is a taste.
    # An array that cannot fill the budget (the 16384 cap, a thin array) gets the chunk closest to it.
    near = 0.8 * pixel_budget
    pairs = [(cy, cx) for cy in valid_y for cx in valid_x if cy * cx <= max_pixels]
    near_pairs = [p for p in pairs if p[0] * p[1] >= near]
    pool = [p for p in near_pairs if max(p[0] / p[1], p[1] / p[0]) <= max_aspect] or near_pairs
    best_result = None
    if pool:
        cy, cx = min(pool, key=lambda p: _compute_chunk_score(p[0], p[1], y_size, x_size, pixel_budget))
        best_result = {'y': cy, 'x': cx}
    elif pairs:
        cy, cx = max(pairs, key=lambda p: (p[0] * p[1], -_compute_chunk_score(p[0], p[1], y_size, x_size, pixel_budget)))
        best_result = {'y': cy, 'x': cx}

    if best_result is None:
        cy = min(y_size, max_chunk)
        cx = min(x_size, max_chunk)
        while cy * cx > max_pixels and cy > min_chunk:
            cy //= 2
        while cy * cx > max_pixels and cx > min_chunk:
            cx //= 2
        best_result = {'y': cy, 'x': cx}
        if floors is not None:
            floors.append((cy, cx))

    return best_result


def restore_chunks(data, original_chunks, debug: bool = False):
    """
    Restore data to original chunk structure.

    Parameters
    ----------
    data : dask.array.Array or xarray.DataArray
        Data to rechunk.
    original_chunks : tuple or dict
        Original chunk structure to restore.
    debug : bool
        Print debug information.

    Returns
    -------
    same type as input
        Data rechunked to original structure.
    """
    import dask.array as da
    import xarray as xr

    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        if isinstance(original_chunks, dict):
            if debug:
                print(f"restore_chunks: restoring to {original_chunks}")
            return data.chunk(original_chunks)
        else:
            # Convert tuple chunks to dict using dimension names
            dims = data.dims
            # Handle case where original_chunks is tuple of tuples
            chunk_dict = {}
            for i, dim in enumerate(dims):
                if i < len(original_chunks):
                    chunks = original_chunks[i]
                    if isinstance(chunks, tuple) and len(chunks) > 0:
                        # Use first chunk size as representative
                        chunk_dict[dim] = chunks[0]
                    else:
                        chunk_dict[dim] = -1  # Full dimension
            if debug:
                print(f"restore_chunks: restoring to {chunk_dict}")
            return data.chunk(chunk_dict)
    else:
        if debug:
            print(f"restore_chunks: restoring to {original_chunks}")
        return data.rechunk(original_chunks)


def _held_chunk(block):
    """Identity task that reads one persisted chunk (see hold_persisted)."""
    return block


def culled_array_optimize(dsk, keys, **kwargs):
    """
    dask's array optimization, then the graph culled to the requested keys (set as dask's array_optimize when insardev
    is imported).

    THE OPTIMIZED GRAPH CARRIES TASKS NO REQUESTED KEY DEPENDS ON: the low-level fusion and the slice rewriting after
    it leave them (117 of 1134 in a point series of GoldenValley's mix()), and the scheduler does not cull a graph that
    comes as an expression. When one of them was a key a graph before still registered -- a da.stack block of
    predict(), plain there and an alias of a fused task here -- the scheduler linked it to a dependency it never
    created, and the graph update failed with KeyError, now and then: the orphans depend on the order dask fuses in.
    Culled, every key of the graph is one a requested key needs, and the scheduler walks them all (dask issue 9888).
    """
    from dask.array.optimization import optimize
    from dask.core import flatten
    from dask._task_spec import cull
    out = optimize(dsk, keys, **kwargs)
    return cull(out, list(flatten(keys))) if isinstance(out, dict) else out


def hold_persisted(arr):
    """
    Keep the cluster data of a persisted dask array alive in every graph built from it.

    After a distributed persist the array graph is {key: Future}. Dask drops these
    entries when it converts a graph to tasks (dask.delayed of the array, to_delayed(),
    optimize), so a product built that way names the chunks by key only and holds no
    Future. Once the persisted array is freed, the scheduler releases the chunks and
    refuses the product ("lost dependencies"). Here each chunk is read by a one-line task
    that takes its Future as an argument; the argument survives any conversion, so the
    chunk stays on the cluster while any product still needs it. The {key: Future} layer
    is kept below it for futures_of() and for checks that look for Futures in the graph.

    Parameters
    ----------
    arr : dask.array.Array
        Result of a persist.

    Returns
    -------
    dask.array.Array
        Same chunks and data. An array without Futures (local scheduler) is returned as is.
    """
    import dask.array as da
    from dask.base import tokenize
    from dask.core import flatten
    from dask.highlevelgraph import HighLevelGraph
    from distributed import Future

    graph = arr.__dask_graph__()
    futures = {k: graph[k] for k in flatten(arr.__dask_keys__())}
    if not all(isinstance(f, Future) for f in futures.values()):
        return arr
    name = 'held-' + tokenize(arr.name)
    layer = {}
    for k, f in futures.items():
        k2 = (name,) + tuple(k[1:])
        layer[k2] = _Task(k2, _held_chunk, f)
    hlg = HighLevelGraph({arr.name: futures, name: layer}, {arr.name: set(), name: {arr.name}})
    return da.Array(hlg, name, chunks=arr.chunks, dtype=arr.dtype, meta=arr._meta)


def _scheduler_knows(keys, dask_scheduler=None):
    return all(k in dask_scheduler.tasks for k in keys)


def _drop_refused(client, futures):
    """
    Detach the futures of a graph the scheduler refused from the client, before raising.

    The raised error holds these futures through its traceback (and a cancelled future's
    own CancelledError, which the client keeps, holds them through the traceback raising it
    gave it). A future state left in client.futures is shared by the next persist of the
    same keys -- the recompute from the source the error asks for -- which then starts as
    'cancelled' and was refused falsely before the scheduler had even seen the new graph.
    The scheduler has already released the refused keys, so nothing is sent to it; the
    futures are released, so a kept traceback holds no cluster data either.
    """
    for f in futures:
        client.futures.pop(f.key, None)
        f.release()


def _refusal_reason(future):
    """The text of a cancelled future's CancelledError, raised and dropped here so that
    it keeps no traceback (and so no frame or future) alive."""
    reason = 'cancelled'
    try:
        future.result()
    except Exception as e:
        reason = str(e)
        e.__traceback__ = None
    return reason


def progress_persisted(persisted, desc):
    """
    Progress bar for a distributed persist; raise when the scheduler refuses the graph.

    The progress bar waits until the scheduler has the keys it tracks. A graph that needs
    data the cluster no longer holds is refused ("lost dependencies"), its keys never
    appear, and the bar waited forever. So first wait until the scheduler has all the
    keys or the futures report the refusal. The refused futures are detached from the
    client before the error is raised (_drop_refused), so the recompute from the source
    works on the first attempt.

    Parameters
    ----------
    persisted : object
        Result of dask.persist() (collections or containers of them).
    desc : str
        Progress bar label.

    Raises
    ------
    RuntimeError
        The graph needs data the cluster no longer holds.
    """
    import time
    from distributed.client import futures_of
    from insardev_toolkit.progressbar import progressbar

    futures = futures_of(persisted)
    if futures:
        client = futures[0].client
        keys = [f.key for f in futures]
        delay = 0.05
        while not client.run_on_scheduler(_scheduler_knows, keys):
            refused = next((f for f in futures if f.status in ('error', 'cancelled')), None)
            if refused is not None:
                _drop_refused(client, futures)
                if refused.status == 'error':
                    refused.result()
                reason = _refusal_reason(refused)
                raise RuntimeError(f'Input data is no longer on the cluster ({reason}). '
                                   'Recompute it from its source.')
            time.sleep(delay)
            delay = min(2 * delay, 1.0)
    progressbar(persisted, desc=desc)


_RESTART_SETTING = 'distributed.deploy.lost-worker-timeout'
# Client -> (monotonic end of its last restart, workers registered before it). Weak: a closed Client drops out.
_restart_last = _weakref.WeakKeyDictionary()


def _worker_count(client):
    return len(client.scheduler_info(n_workers=-1)['workers'])


def _wait_before_restart(client, timeout):
    """
    The wait of a restart of `client` that follows another one (see install_safe_restart). Waits only while less than the
    lost-worker timeout has passed since the previous restart returned; prints one line when it waits.
    """
    import time
    from dask.typing import no_default
    from dask.utils import parse_timedelta
    previous = _restart_last.get(client)
    if previous is None:
        return
    end, workers = previous
    setting = dask.config.get(_RESTART_SETTING)
    delay = parse_timedelta(setting, 's')
    elapsed = time.monotonic() - end
    if elapsed >= delay:
        return
    try:
        # the bound of distributed's own restart: the caller's timeout, else 4 x the connect timeout as Client.restart
        bound = (parse_timedelta(dask.config.get('distributed.comm.timeouts.connect'), 's') * 4
                 if timeout is no_default else parse_timedelta(timeout, 's'))
    except Exception:
        return                      # the original restart raises the error of its timeout argument itself
    print(f'insardev: client.restart() waits {delay - elapsed:.1f} s for the lost-worker timers of the previous restart '
          f'({_RESTART_SETTING} = {setting})', flush=True)
    if _worker_count(client) < workers:
        client.wait_for_workers(workers, timeout=bound)     # raises TimeoutError as restart does when the workers stay away
    rest = end + delay - time.monotonic()
    if rest > 0:
        time.sleep(rest)


def _restart_bound(timeout):
    """
    The bound of a restart: the caller's timeout, else 4 x the connect timeout (as the wait above). Parsed before any change,
    so a bad value raises here, as the original restart raises it.
    """
    from dask.typing import no_default
    from dask.utils import parse_timedelta
    if timeout is no_default:
        timeout = parse_timedelta(dask.config.get('distributed.comm.timeouts.connect'), 's') * 4
    return parse_timedelta(timeout, 's')


def _worker_names(client):
    """The names of the workers the scheduler has registered, as strings."""
    return {str(w['name']) for w in client.scheduler_info(n_workers=-1)['workers'].values()}


def _managed_by(client, cluster):
    """
    True when every registered worker is a spec entry of the cluster, and every spec entry is one of the cluster's workers. A
    worker the scheduler has lost may still be a cluster worker until its lost-worker timer closes it (deploy/spec.py 398-419); it
    is then a worker the cluster holds but the scheduler does not, and the restart drops its entry.
    """
    registered = _worker_names(client)
    spec = {str(name) for name in list(cluster.worker_spec)}
    names = {str(name) for name in list(cluster.workers)}
    return bool(registered) and registered <= spec <= names


def _release_on_scheduler(dask_scheduler=None, stimulus_id=None):
    """
    Runs on the scheduler (Client.run_on_scheduler): the part of Scheduler.restart before restart_workers (scheduler.py 6446-6462):
    keys released, task state cleared, plugins told. A copy of that body; drift is caught by the keys_freed rows.
    """
    import logging
    import time
    if stimulus_id is None:
        stimulus_id = f'client-restart-{time.time()}'
    for cs in dask_scheduler.clients.values():
        dask_scheduler.client_releases_keys(keys=[ts.key for ts in cs.wants_what], client=cs.client_key, stimulus_id=stimulus_id)
    dask_scheduler._clear_task_state()
    if dask_scheduler.tasks:
        raise RuntimeError('restart: the scheduler still holds tasks after the release')
    dask_scheduler.report({'op': 'restart'})
    for plugin in list(dask_scheduler.plugins.values()):
        try:
            plugin.restart(dask_scheduler)
        except Exception as e:
            logging.getLogger('distributed.scheduler').exception(e)


def _wait_gone(client, cluster, names, bound):
    """
    Waits, bounded, until none of `names` (strings) is a worker of `cluster` or registered with the scheduler. SpecCluster deletes
    a worker after its close (deploy/spec.py 366-368); the scheduler drops it when it has closed.
    """
    import time
    deadline = time.monotonic() + bound
    delay = 0.05
    while True:
        left = names & ({str(name) for name in cluster.workers} | _worker_names(client))
        if not left:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'workers {sorted(left)} still open {bound} s after the restart started')
        time.sleep(delay)
        delay = min(2 * delay, 0.5)


def _wait_recorded(cluster, names, bound):
    """
    Waits, bounded, until every name of `names` (strings) is a worker of `cluster`. SpecCluster records a new worker only after all
    the new workers have started (deploy/spec.py 387-390), which is later than the scheduler's registration that wait_for_workers counts.
    """
    import time
    deadline = time.monotonic() + bound
    while not names <= {str(name) for name in list(cluster.workers)}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f'workers {sorted(names)} not recorded by the cluster {bound} s after the restart started')
        time.sleep(0.05)


def _replace_workers(client, cluster, bound, wait_for_workers):
    """
    Variant V1: the spec entries of the registered workers are replaced by the same number of entries with fresh string names and
    the cluster is scaled to them. An entry of a worker the scheduler has lost is dropped: a restart restarts the registered workers
    only, as Client.restart does. SpecCluster closes the old workers (and deletes them) before it opens the new ones, in one pass.
    The retired integer names are never allocated again: SpecCluster allocates integer names from _i (deploy/spec.py 541-565), which
    is advanced past them.
    """
    import uuid
    registered = _worker_names(client)
    specs = [dict(spec) for name, spec in cluster.worker_spec.items() if str(name) in registered]
    retired = set(cluster.worker_spec) | set(cluster.workers)
    fresh = {f'r{uuid.uuid4().hex[:12]}-{i}': spec for i, spec in enumerate(specs)}
    integers = [name for name in retired if isinstance(name, int)]
    if integers:
        cluster._i = max(cluster._i, max(integers) + 1)
    cluster.worker_spec.clear()
    cluster.worker_spec.update(fresh)
    cluster.scale(len(fresh))
    _wait_gone(client, cluster, {str(name) for name in retired}, bound)
    if wait_for_workers:
        cluster.wait_for_workers(len(fresh), timeout=bound)
        _wait_recorded(cluster, set(fresh), bound)


def _missing_private_names():
    """
    The private distributed names the safe-restart hook calls that are missing or not accessible, as a list of their names (empty when
    all are there). Scheduler._clear_task_state is read on the class (it is inherited from the scheduler state). SpecCluster._i is an
    instance attribute that SpecCluster.__init__ assigns, and no instance exists at installation: it is accessible when the class has no
    entry of that name that raises on read, and the __init__ code assigns it.
    """
    from distributed.scheduler import Scheduler
    from distributed.deploy.spec import SpecCluster
    missing = []
    try:
        getattr(Scheduler, '_clear_task_state')
    except AttributeError:
        missing.append('Scheduler._clear_task_state')
    init_code = getattr(SpecCluster.__init__, '__code__', None)
    try:
        if '_i' in vars(SpecCluster):
            getattr(SpecCluster, '_i')
        elif init_code is None or '_i' not in init_code.co_names:
            missing.append('SpecCluster._i')
    except AttributeError:
        missing.append('SpecCluster._i')
    return missing


def install_safe_restart():
    """
    Replace the workers of a SpecCluster on restart instead of restarting them under their names (BUGS N194; GRS6 variant V1, C5).

    When the client has a SpecCluster in this process and every registered worker is one of the cluster's workers (_managed_by), the wrapper:
      1. releases the keys and clears the task state on the scheduler (the body of Scheduler.restart before restart_workers);
      2. replaces the spec entries of the registered workers by fresh string names (the entry of a worker the scheduler has lost is
         dropped) and scales the cluster to them: the old workers are closed and deleted, the new ones are opened
         (SpecCluster._correct_state_internal, deploy/spec.py 352-396). The retired integer names are never allocated again:
         SpecCluster._i is advanced past them, so a later scale() cannot take a name whose lost-worker timer
         (deploy/spec.py 398-419) is still pending;
      3. waits, bounded by the restart timeout, until the old workers are gone from the cluster and from the scheduler; with
         wait_for_workers it then waits until the new workers are registered and recorded by the cluster, as Client.restart
         does, and not otherwise.
    A fresh name matches no pending lost-worker timer, so this path does not wait between restarts.
    Otherwise (no SpecCluster, a registered worker that is not the cluster's, or an asynchronous client) it waits as before
    (_wait_before_restart) and
    calls the original restart. Errors of the restart pass through unchanged; KeyboardInterrupt is not caught.

    The private names it calls are Scheduler._clear_task_state (_release_on_scheduler) and SpecCluster._i (_replace_workers). At
    installation they are checked to exist and to be accessible (_missing_private_names); if one is missing, nothing is installed and
    one WARNING line names it: Client.restart stays dask's own. Every other name is public. Idempotent: a second call wraps nothing.
    """
    import functools
    import time
    from distributed import Client
    from distributed.deploy.spec import SpecCluster
    from dask.typing import no_default
    if getattr(Client.restart, '__insardev_safe_restart__', False):
        return
    missing = _missing_private_names()
    if missing:
        print(f"WARNING: install_safe_restart skipped, Client.restart is dask's own: private dask name(s) missing or not accessible "
              f"in this distributed version: {', '.join(missing)}", flush=True)
        return
    original = Client.restart

    @functools.wraps(original)
    def restart(self, timeout=no_default, wait_for_workers=True):
        if self.asynchronous:
            return original(self, timeout=timeout, wait_for_workers=wait_for_workers)
        cluster = getattr(self, 'cluster', None)
        if not isinstance(cluster, SpecCluster) or not _managed_by(self, cluster):
            _wait_before_restart(self, timeout)
            workers = _worker_count(self)
            try:
                return original(self, timeout=timeout, wait_for_workers=wait_for_workers)
            finally:
                _restart_last[self] = (time.monotonic(), workers)
        bound = _restart_bound(timeout)
        workers = _worker_count(self)
        try:
            self.run_on_scheduler(_release_on_scheduler, stimulus_id=f'client-restart-{time.time()}')
            _replace_workers(self, cluster, bound, wait_for_workers)
        finally:
            _restart_last[self] = (time.monotonic(), workers)

    restart.__insardev_safe_restart__ = True
    Client.restart = restart


def overlap_check(result, source, name, window, halo):
    """The map_overlap `result` of `source` keeps the input's chunks, or raises one line. A halo larger than a chunk makes
    dask rechunk the input (ensure_minimum_chunksize), and the output then has other chunks than the input the user gave:
    the ValueError names the function, the smallest input chunk, the window and the halo it needs.

    Examples
    --------
    >>> filtered = overlap_check(filtered, dsk, 'median', (11, 11), (5, 5))
    """
    if result.chunks == source.chunks:
        return result
    smallest = min(min(c) for c in source.chunks[-2:])
    raise ValueError(
        f'{name}(): window {tuple(int(v) for v in window)} px needs a halo of {tuple(int(v) for v in halo)} px, '
        f'larger than the smallest input chunk of {smallest} px: give the input blocks of at least {max(int(v) for v in halo)} px')


def union_check(chunks, who):
    """Rule (3) of the union drivers: a union=True function runs one task per input date chunk, so a date axis that is
    ONE chunk of more than one date would make one task join every date of every burst: that case raises one line.
    One date is one chunk in every layout, so it never raises. `chunks` is the date chunk sizes of the input, `who` the
    public name of the function.

    Examples
    --------
    >>> union_check((10, 10), 'trend2d')  # two date chunks: no raise
    >>> union_check((1,), 'unwrap2d')  # one date: no raise
    """
    if len(chunks) == 1 and chunks[0] > 1:
        raise ValueError(f"{who}(union=True): the date axis is one chunk, so one task would join all {chunks[0]} "
                         f"dates of every burst -- chunk the stack by date, one or more dates per chunk.")


def coarsen_chunks(blocks, window):
    """Block sizes of one dimension for coarsen() by `window`: the block boundaries moved up to whole windows
    (the first block starts at the window grid), or None when no boundary is left inside the dimension."""
    import numpy as np
    n, w = int(sum(blocks)), int(window)
    inner = sorted({-(-int(b) // w) * w for b in np.cumsum(blocks)[:-1]} - {0})
    inner = [b for b in inner if b < n]
    if inner:
        return tuple(int(s) for s in np.diff([0] + inner + [n]))
    return None


# the graph rule (the user, 2026-11-01): one WARNING line when a call makes more than GRAPH_CHUNK_LIMIT output chunks of one
# variable of one burst, when it adds more than GRAPH_TASK_LIMIT lazy tasks, or when its input already carries more than
# GRAPH_TASK_LIMIT (about 2000 chunks x 20 tasks per chunk: the user expects 10 to 20 tasks per chunk)
GRAPH_CHUNK_LIMIT = 2000
GRAPH_TASK_LIMIT = 40000


def _lazy_arrays(obj):
    """The dask arrays of a dask array, an xarray object, a list or a dict of them (other objects have none)."""
    import dask.array as da
    import xarray as xr
    if isinstance(obj, (list, tuple)):
        return [a for o in obj for a in _lazy_arrays(o)]
    if isinstance(obj, dict):
        return [a for o in obj.values() for a in _lazy_arrays(o)]
    if isinstance(obj, xr.Dataset):
        return [v.data for v in obj.data_vars.values() if isinstance(v.data, da.Array)]
    if isinstance(obj, xr.DataArray):
        obj = obj.data
    return [obj] if isinstance(obj, da.Array) else []


def _graph_layers(arrays):
    """{layer name: layer} of the lazy graphs of the arrays; a plain (materialised) graph counts as one layer."""
    from dask.highlevelgraph import HighLevelGraph
    layers = {}
    for arr in arrays:
        graph = arr.__dask_graph__()
        if isinstance(graph, HighLevelGraph):
            layers.update(graph.layers)
        else:
            layers.setdefault(arr.name, graph)
    return layers


def graph_check(name, result, source):
    """
    Warn when a call makes too many lazy graph objects: the graph rule (GRAPH_CHUNK_LIMIT, GRAPH_TASK_LIMIT).

    Counts the lazy graph only (the layers by length, nothing is optimised). Prints one WARNING line per rule that fires:
    more than GRAPH_CHUNK_LIMIT output chunks of one variable of one burst (the largest is named), more than GRAPH_TASK_LIMIT
    lazy tasks added to the input's graph by the call, and more than GRAPH_TASK_LIMIT lazy tasks that the input already
    carries (the line suggests a snapshot). Otherwise nothing.
    The count line names chunk2d() when the input already has as many chunks.

    Parameters
    ----------
    name : str
        Function name for the message.
    result : dask.array.Array, xarray.DataArray, xarray.Dataset, list or dict of them
        Output of the function.
    source : dask.array.Array, xarray.DataArray, xarray.Dataset, list or dict of them
        Input of the function.

    Returns
    -------
    dict or None
        added, output_chunks, added_per_chunk, input_per_chunk, input_tasks, max_chunks; None when the result has no dask array.

    Examples
    --------
    >>> graph_check('lee', result, source)
    """
    import math

    def named(obj, burst=None, var=None):
        """(label, dask array) of each lazy array of obj: a variable of a dataset, named by its burst (a dict key)."""
        import dask.array as da
        import xarray as xr
        if isinstance(obj, (list, tuple)):
            return [p for o in obj for p in named(o, burst, var)]
        if isinstance(obj, dict):
            return [p for k, o in obj.items() for p in named(o, str(k), var)]
        if isinstance(obj, xr.Dataset):
            return [p for k, v in obj.data_vars.items() for p in named(v.data, burst, str(k))]
        if isinstance(obj, xr.DataArray):
            return named(obj.data, burst, var or (None if obj.name is None else str(obj.name)))
        if isinstance(obj, da.Array):
            return [(' in '.join(p for p in (var, burst) if p) or 'array', obj)]
        return []

    res_arrays = _lazy_arrays(result)
    if not res_arrays:
        return None
    src_arrays = _lazy_arrays(source)
    src_layers = _graph_layers(src_arrays)
    res_layers = _graph_layers(res_arrays)
    added = sum(len(layer) for key, layer in res_layers.items() if key not in src_layers)
    out_chunks = sum(math.prod(a.numblocks) for a in res_arrays)
    in_total = sum(len(layer) for layer in src_layers.values())
    in_chunks = sum(math.prod(a.numblocks) for a in src_arrays) or 1
    counts = [(math.prod(a.numblocks), label) for label, a in named(result)]
    over = sorted((n, label) for n, label in counts if n > GRAPH_CHUNK_LIMIT)
    info = {'added': added, 'output_chunks': out_chunks,
            'added_per_chunk': added / max(out_chunks, 1), 'input_per_chunk': in_total / in_chunks,
            'input_tasks': in_total, 'max_chunks': max((n for n, _ in counts), default=0)}
    if over:
        n, label = over[-1]
        more = f' ({len(over) - 1} more arrays over the limit)' if len(over) > 1 else ''
        kept = n <= max((math.prod(a.numblocks) for a in src_arrays), default=0)
        advice = (': the input has as many: re-chunk it with a larger chunk2d() budget' if kept
                  else ': use a larger budget or keep the stored chunks')
        print(f"WARNING: {name}(): {n} output chunks of {label} (more than {GRAPH_CHUNK_LIMIT}){advice}{more}", flush=True)
    if added > GRAPH_TASK_LIMIT:
        print(f"WARNING: {name}(): {added} lazy tasks added ({info['added_per_chunk']:.0f} per output chunk, {out_chunks} "
              f"output chunks), more than {GRAPH_TASK_LIMIT}: the graph grows with the dates and pairs of the call", flush=True)
    if in_total > GRAPH_TASK_LIMIT:
        print(f"WARNING: {name}(): the input carries {in_total} lazy tasks, more than {GRAPH_TASK_LIMIT}: take a snapshot of "
              f"the input first (Stack.snapshot()), so the call does not carry its graph", flush=True)
    return info


# ----------------------------------------------------------------------------
# Inputs of a multi-input function (GMT U2, user ruling 2026-11-02). The PRIMARY inputs (the data: every polarisation)
# set the output's chunks and are never rechunked. A SECONDARY input (a weight, a phase to remove, a covariate) takes the
# primary's (y, x) chunks, rechunked with one WARNING per call; other date chunks of a secondary raise one line.

def _sizes(chunks) -> str:
    """The chunk sizes as text: _sizes((32, 32)) is '(32, 32)'; more than 8 blocks give their count."""
    return str(tuple(chunks)) if len(chunks) <= 8 else f'{len(chunks)} blocks'


def check_primary_chunks(arrays: dict, who: str) -> None:
    """Raises one line when the primary inputs (name: dask array of the same dims) are chunked differently:
    check_primary_chunks({'VV': vv, 'VH': vh}, 'detrend')."""
    names = list(arrays)
    for n in names[1:]:
        if arrays[n].chunks != arrays[names[0]].chunks:
            raise ValueError(f"ERROR: {who}(): the primary inputs '{names[0]}' and '{n}' are chunked differently; "
                             f"all polarisations are chunked the same")


def align_secondary(primary, secondary, who: str, name: str, check_dates: bool = True, dims=None):
    """A secondary input on the primary's chunks: returns (array, changed). Each of its dims takes the primary's chunks
    of that dim ('date' first, 'y' and 'x' last), so a 1-D covariate over 'x' gets the x chunks only. Its (y, x) chunks
    go to the primary's (changed=True: the caller warns once per call with warn_secondary); other date chunks raise
    one line, unless check_dates is False (the caller rebuilt the dates, so the secondary keeps its own date chunks).
    dims: the secondary's dims (default ('date', 'y', 'x') for 3-D and ('y', 'x') for 2-D; a 1-D one must name its dim).
    data, changed = align_secondary(primary, phase, 'detrend', 'phase')."""
    dims = tuple(dims) if dims is not None else {3: ('date', 'y', 'x'), 2: ('y', 'x')}.get(secondary.ndim)
    if dims is None or len(dims) != secondary.ndim:
        raise ValueError(f"align_secondary(): name the dims of the {secondary.ndim}-D secondary '{name}' (dims=...)")
    axis = {'date': 0, 'y': -2, 'x': -1}
    pc = primary.chunks
    target = []
    for d, c in zip(dims, secondary.chunks):
        if d == 'date':
            if check_dates and c != pc[0]:
                raise ValueError(f"ERROR: {who}(): the date chunks of the {name} {_sizes(c)} differ from the data's "
                                 f"{_sizes(pc[0])}: a secondary input takes the data's date chunks")
            target.append(pc[0] if check_dates else c)
        else:
            target.append(pc[axis[d]])
    target = tuple(target)
    if secondary.chunks == target:
        return secondary, False
    return secondary.rechunk(target), True


def warn_secondary(who: str, names) -> None:
    """ONE WARNING line per call for the secondary inputs rechunked in it (names: 'phase', "covariate 'ele'", ...):
    warn_secondary('detrend', ['phase'])."""
    names = list(dict.fromkeys(names))
    if names:
        print(f"WARNING: {who}(): {', '.join(names)} rechunked to the data's (y, x) chunks; the data's chunks are the output's.",
              flush=True)
