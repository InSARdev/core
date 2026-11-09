"""Residual statistics of the batch classes: time in years, the RMSE of a model, the RMS of residuals, and their fused forms."""
import numpy as np


NANOSECONDS_PER_YEAR = np.float64(365.25 * 24 * 60 * 60 * 1e9)


def years_between(start, stop):
    """Time from start to stop in years, float32; both are int64 nanoseconds (arrays or one value)."""
    return ((stop.astype(np.float64) - start.astype(np.float64)) / NANOSECONDS_PER_YEAR).astype(np.float32)


def phase_error(obs, pred):
    """Angle of obs against the model pred: angle(obs * conj(exp(1j * pred)))."""
    return np.angle(obs * np.conj(np.exp(1j * pred)))


def rmse_block(obs, vel, weight=None, *, dt, circular=False):
    """RMSE of one block over its pairs (axis 0): obs (p, y, x) against vel (y, x) * dt (p,) years.

    circular: obs are phases, residuals by phase_error(); otherwise values. weight (p, y, x) or None.
    Returns float32 (y, x).
    """
    P = obs.shape[0]
    sh = obs.shape[1:]
    accum = np.zeros(sh, dtype=np.float64)
    if weight is not None:
        w_sum = np.zeros(sh, dtype=np.float64)
    for p in range(P):
        pred = vel * dt[p]
        if circular:
            resid = phase_error(obs[p], pred)
        else:
            resid = obs[p].astype(np.float32) - pred
        if weight is not None:
            w = weight[p].astype(np.float64)
            accum += w * resid.astype(np.float64) ** 2
            w_sum += w
        else:
            accum += resid.astype(np.float64) ** 2
    denom = w_sum if weight is not None else np.float64(P)
    return np.sqrt(accum / denom).astype(np.float32)


def residual_block(obs, sol, *, mode):
    """The residual of one block: 'circular' (the angle of obs against the phase sol), 'wrap' (the difference wrapped to
    [-pi, pi)) or 'plain' (the difference). Float32 for phases."""
    if mode == 'circular':
        return phase_error(obs, sol).astype(np.float32)
    if mode == 'wrap':
        from .utils_circular import wrap_phase
        # NaN is kept out of the wrap: wrap_phase warns on NaN (the finite residuals are wrapped as before)
        d = obs - sol
        ok = ~np.isnan(d)
        return np.where(ok, wrap_phase(np.where(ok, d, 0)), np.nan).astype(d.dtype)
    return obs - sol


def rmse_pairs_block(obs, sol, weight=None, *, mode):
    """RMSE over the pairs (axis 0) of one block: the residual, the weight and the weighted rms in one pass. NaN residuals and
    NaN weights are skipped as xarray's sum and mean skip them. Float32 (y, x)."""
    e2 = residual_block(obs, sol, mode=mode) ** 2
    ok = ~np.isnan(e2)
    if weight is None:
        num = np.where(ok, e2, 0).sum(0, dtype=np.float32)
        den = ok.sum(0).astype(np.float32)
    else:
        wk = ~np.isnan(weight)
        num = np.where(ok & wk, weight * e2, 0).sum(0, dtype=np.float32)
        den = np.where(wk, weight, 0).sum(0, dtype=np.float32)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den != 0)
    return np.sqrt(out).astype(np.float32)


def _align_yx(named, moved):
    """The dask operands on the chunks of the first dask one (the primary: the observed). A secondary is put on them by
    utils_dask.align_secondary (GMT U3): its (y, x) chunks are rechunked (name appended to `moved`, one WARNING per call in
    BatchCore.rmse); its pair chunks must equal the primary's, else one line RAISES. None when no operand is dask; numpy
    operands are returned as they are."""
    import dask.array as da
    from .utils_dask import align_secondary
    ref = next((a for _, a in named if isinstance(a, da.Array)), None)
    if ref is None:
        return None
    out = []
    for name, a in named:
        if isinstance(a, da.Array):
            a, changed = align_secondary(ref, a, 'rmse', name, check_dates=ref.ndim == 3 and a.ndim == 3)
            if changed:
                moved.append(name)
        out.append(a)
    return out


def align_da(primary, secondary, name, moved):
    """An xarray secondary of the date-based rmse branch on the primary's (y, x) chunks (GMT U3, utils_dask.align_secondary):
    the secondary is transposed to the primary's dims order. Its temporal chunks are compared only when it has the primary's
    temporal dim (pair or date) and must equal them, else one line RAISES; a secondary with another temporal dim (the date-based
    solution, read by sel per pair) keeps its own. `name` is appended to `moved` when it is rechunked. Numpy or a grid-less
    primary: returned as it is.
    weight = align_da(obs_da, weight, 'weight', moved)"""
    import dask.array as da
    from .utils_dask import align_secondary
    if not (isinstance(primary.data, da.Array) and isinstance(secondary.data, da.Array)):
        return secondary
    if 'y' not in primary.dims or 'x' not in primary.dims or 'y' not in secondary.dims or 'x' not in secondary.dims:
        return secondary
    tdim = next((d for d in ('pair', 'date') if d in primary.dims), None)
    lead_p = [tdim] if tdim else []
    lead_s = [d for d in secondary.dims if d not in ('y', 'x')]
    if len(lead_s) != len(lead_p):
        return secondary
    pr = primary.transpose(*(lead_p + ['y', 'x']))
    sec = secondary.transpose(*(lead_s + ['y', 'x']))
    new, changed = align_secondary(pr.data, sec.data, 'rmse', name, check_dates=bool(lead_s) and lead_s[0] == tdim)
    if not changed:
        return sec
    moved.append(name)
    return sec.copy(data=new)


def _blocks(a, chunks):
    """A dask operand as it is; a numpy one cut to `chunks` (the chunking of the dask operands)."""
    import dask.array as da
    return a if isinstance(a, da.Array) else da.from_array(a, chunks=chunks)


def rmse_pairs(obs, sol, weight=None, *, mode, moved=None):
    """Pair-based RMSE of (pair, y, x) operands in ONE blockwise task per (y, x) chunk: the observed, the solution and the weight
    join inside it (rmse_pairs_block). The secondaries (solution, weight) are put on the observed's chunks (_align_yx: other (y, x)
    chunks are rechunked, other pair chunks RAISE one line); each operand's pairs join in the task under its own index letter.
    Numpy operands run as one block."""
    import dask.array as da
    import functools
    moved = [] if moved is None else moved
    ops = _align_yx([('observed', obs), ('solution', sol), ('weight', weight)], moved)
    if ops is None:
        return rmse_pairs_block(obs, sol, weight, mode=mode)
    obs, sol, weight = ops
    yx = next(a.chunks[-2:] for a in ops if isinstance(a, da.Array))
    args = []
    for a, letter in zip((obs, sol, weight), ('pyx', 'qyx', 'ryx')):
        if a is not None:
            args += [_blocks(a, ((a.shape[0],),) + yx), letter]
    # the (y, x) chunks are equal (_align_yx), so dask's alignment has nothing to do: align_arrays=False. Its chunk-count
    # warning multiplies the pair chunk counts of the operands (a count, not a rechunk) and would fire on every pair-chunked call
    return da.blockwise(functools.partial(rmse_pairs_block, mode=mode), 'yx', *args,
                        concatenate=True, align_arrays=False, dtype=np.float32, meta=np.empty((0, 0), dtype=np.float32))


def rmse_velocity(obs, vel, weight=None, *, dt, circular, moved=None):
    """RMSE of pairs against vel * dt (years per pair) in ONE blockwise task per (y, x) chunk (rmse_block). The observed, the
    weight and the velocity are put on the observed's chunks (_align_yx: other (y, x) chunks are rechunked, GMT U3; other pair
    chunks of the weight RAISE one line)."""
    import dask.array as da
    import functools
    moved = [] if moved is None else moved
    ops = _align_yx([('observed', obs), ('weight', weight), ('velocity', vel)], moved)
    if ops is None:
        return rmse_block(obs, vel, weight, dt=dt, circular=circular)
    obs, weight, vel = ops
    yx = next(a.chunks[-2:] for a in ops if isinstance(a, da.Array))
    args = [_blocks(obs, ((obs.shape[0],),) + yx), 'pyx', _blocks(vel, yx), 'yx']
    if weight is not None:
        args += [_blocks(weight, ((weight.shape[0],),) + yx), 'qyx']
    # the (y, x) chunks are equal (_align_yx): align_arrays=False, as in rmse_pairs
    return da.blockwise(functools.partial(rmse_block, dt=dt, circular=circular), 'yx', *args,
                        concatenate=True, align_arrays=False, dtype=np.float32, meta=np.empty((0, 0), dtype=np.float32))


def rms(err, dim, weight=None):
    """RMS of residuals err over dim, weighted by weight when given (BatchCore.rmse)."""
    err_sq = err ** 2
    if weight is not None:
        return np.sqrt((weight * err_sq).sum(dim) / weight.sum(dim))
    return np.sqrt(err_sq.mean(dim))


def mean_block(weight):
    """Mean over the first axis of one block (NaN skipped, as xarray's mean): (y, x)."""
    dt = np.result_type(weight.dtype, np.float32)
    ok = ~np.isnan(weight)
    num = np.where(ok, weight, 0).sum(0, dtype=dt)
    cnt = ok.sum(0).astype(dt)
    out = np.full(num.shape, np.nan, dtype=dt)
    np.divide(num, cnt, out=out, where=cnt != 0)
    return out


def mean_pairs(weight):
    """Mean over the first axis of a (pair, y, x) array, NaN skipped: (y, x) in ONE blockwise task per (y, x) chunk (mean_block)."""
    import dask.array as da
    if not isinstance(weight, da.Array):
        return mean_block(weight)
    dt = np.result_type(weight.dtype, np.float32)
    return da.blockwise(mean_block, 'yx', weight, 'pyx', concatenate=True, dtype=dt, meta=np.empty((0, 0), dtype=dt))
