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
from .BatchCore import BatchCore, _RULE_OPERANDS
import numpy as np
import xarray as xr
from .utils_circular import wrap_phase, circular_aggregate, coarsen_phasors, resultant_length, _std_of_R
from .Batch import Batch
from . import utils_unwrap2d
from typing import TYPE_CHECKING
from .utils_dask import graph_check
if TYPE_CHECKING:
    from .Stack import Stack


class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None, wrap: bool = True):
        from .BatchComplex import BatchComplex
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
                    out[v] = wrap_phase(da_)
            out.attrs = data.attrs
            return out
        return wrap_phase(data)

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
        if not BatchCore._is_operand(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _RULE_OPERANDS):
            return self._result({k: BatchCore._binary_vars(v, other, _operator.add) for k, v in self.items()})
        keys = self.keys()
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.add) if k in other else self[k])
                           for k in keys})

    def __sub__(self, other: Batch):
        if not BatchCore._is_operand(other):
            return NotImplemented
        import xarray as xr
        import operator as _operator
        # SUBTRACT THE GRIDS, CARRY THE REST. A whole-Dataset `ds - val` reaches
        # the burst ids and the radar geometry too: numpy refuses to subtract
        # <U43 strings, so align() raised, and radar_wavelength minus itself is
        # 0, so meter2rad became infinite where it did not. _binary_vars is the
        # one place that rule lives.
        _sub_grids = lambda d, v: BatchCore._binary_vars(d, v, _operator.sub)
        if isinstance(other, _RULE_OPERANDS):
            return self._result({k: _sub_grids(v, other) for k, v in self.items()})
        # the per-pair coefficients of align() go through the DataArray polyval() correction
        return self._result({k: (self._sub_coeffs(k, self[k], other[k]) if k in other else self[k])
                           for k in self.keys()})

    def __mul__(self, other: Batch):
        if not BatchCore._is_operand(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _RULE_OPERANDS):
            return self._result({k: BatchCore._binary_vars(v, other, _operator.mul) for k, v in self.items()})
        keys = self.keys()
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.mul) if k in other else self[k])
                           for k in keys})

    def __rmul__(self, other):
        if not isinstance(other, _RULE_OPERANDS):
            return NotImplemented
        # scalar * batch  → map scalar * each dataset
        return self._result({k: BatchCore._binary_vars(v, other, lambda a, b: b * a) for k, v in self.items()})

    def __truediv__(self, other: Batch):
        if not BatchCore._is_operand(other):
            return NotImplemented
        import operator as _operator
        if isinstance(other, _RULE_OPERANDS):
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
        from .BatchComplex import BatchComplex
        # the phasors are built as a plain dict: rebuilding a BatchWrap first
        # wrapped the complex values, which numpy refuses (interferogram(phase=BatchWrap))
        return BatchComplex(self._map_grids(lambda da, **kw: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after: the phase
        reductions (mean, sum, median, min, max). The spread is not phase: see std() and var().

        THE OPERATOR RULE: the grids are aggregated, every other variable passes
        through unchanged (the string metadata raised in astype() before).
        """
        import xarray as xr
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
            # the aggregation in the complex domain, the float32 wrapped phase back (utils_circular.circular_aggregate)
            out[key] = circular_aggregate(obj, name, dim, tuple(carry), **kwargs)
            for v, da_ in carry.items():
                out[key][v] = da_

        # each burst back in its own chunks, along the dims its result still has. The angle is in (-pi, pi] already: not wrapped again
        return self._restored(out)

    def std(self, dim=None, **kwargs) -> Batch:
        """
        The circular standard deviation sqrt(-2 ln R) over `dim` in radians, R the mean resultant length of the phase
        (utils_circular.resultant_length). Not wrapped: a spread is not a phase. Returns a plain Batch.
        """
        return self._spread(_std_of_R, dim, **kwargs)

    def var(self, dim=None, **kwargs) -> Batch:
        """
        The circular variance 1 - R over `dim`, R the mean resultant length of the phase, in [0, 1]. Not wrapped: a
        spread is not a phase. Returns a plain Batch.
        """
        return self._spread(lambda R: 1 - R, dim, **kwargs)

    def _spread(self, spread, dim=None, **kwargs) -> Batch:
        """
        The spread `spread(R)` of each burst, R from utils_circular.resultant_length: the metadata carried as _agg carries it,
        each burst back in its own chunks (BatchCore._restored), as a plain Batch.
        """
        import xarray as xr
        out = {}
        for key, obj in self.items():
            carry = {}
            if isinstance(obj, xr.Dataset):
                grids = BatchCore._grid_vars(obj)
                carry = {v: obj[v] for v in obj.data_vars if v not in grids}
                obj = obj[grids]
            elif hasattr(obj, 'obj') and isinstance(obj.obj, xr.Dataset):
                carry = {v: obj.obj[v] for v in obj.obj.data_vars if not BatchCore._is_grid(obj.obj[v])}
            out[key] = spread(resultant_length(obj, dim, tuple(carry), **kwargs)).astype('float32')
            for v, da_ in carry.items():
                out[key][v] = da_
        return Batch(self._restored(out))

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
            # the complex phasors of the grids, by the operator rule, and every other variable carried
            if isinstance(ds, xr.Dataset):
                grids = BatchCore._grid_vars(ds)
                meta = {v: ds[v] for v in ds.data_vars if v not in grids}
                ds2 = coarsen_phasors(ds[grids], meta, window)
            else:
                ds2 = coarsen_phasors(ds, {}, window)
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
        from .utils_gaussian import phase_of_sincos
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
                phase = xr.apply_ufunc(phase_of_sincos, sin[k], cos[k], dask='allowed').rename(src.name)
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
                phase = xr.apply_ufunc(phase_of_sincos, sin[k][var], cos[k][var], dask='allowed')
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
        # Validate parameters
        if not conncomp and conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )
        # a BatchUnit (N81), whatever the phase: a DataArray one weights every grid, a
        # Dataset one names every grid of a batch phase
        weight = BatchCore._weight(weight, self)

        if union:
            return BatchWrap._unwrap2d_union(
                self, weight, conncomp=conncomp, conncomp_size=conncomp_size,
                conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                conncomp_linkcount=conncomp_linkcount,
                debug=debug, **kwargs)

        # Use IRLS unwrapping - always returns (unwrapped, conncomp)
        # Pass conncomp_size to filter small components during IRLS
        unwrapped, conncomp_labels = self.unwrap2d_irls(
            weight, conncomp_size=conncomp_size, debug=debug, **kwargs
        )

        if conncomp:
            # Return size-ordered conncomp labels (already computed by irls)
            result = unwrapped, conncomp_labels
        else:
            # Link components using pre-computed conncomp from IRLS
            result = unwrapped._link_components(
                conncomp_labels=conncomp_labels,
                conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
                conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
                debug=debug
            )
        graph_check('unwrap2d', result, self)
        return result

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
        import dask
        import dask.array as da
        import xarray as xr
        from .Batch import Batch
        from .BatchWrap import BatchWrap
        from .BatchUnit import BatchUnit

        phase = self
        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        from .BatchCore import BatchCore
        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, phase)

        # Validate lazy data
        BatchCore._require_lazy(phase, 'unwrap2d')

        # every unwrapping path builds its graph here, so this prints once per call
        unknown = sorted(set(kwargs) - set(utils_unwrap2d._OLD_IRLS))
        if unknown:
            raise TypeError(f'unwrap2d_irls() got unexpected keyword arguments: {", ".join(unknown)}')
        utils_unwrap2d.old_irls_warning(kwargs, 'unwrap2d()')

        # THE CORES OVER THE RASTERS THAT RUN AT ONCE: a raster is one pair of one grid of one burst
        rasters = sum(da_.sizes['pair'] if 'pair' in da_.dims else 1
                      for key in phase.keys() for da_ in BatchCore._grids_of(phase[key]).values())
        threads = utils_unwrap2d.unwrap_threads(rasters)
        if debug:
            print(f'DEBUG unwrap2d: {rasters} rasters, {threads} numba threads each')

        # Process each burst in the batch
        unwrap_result = {}
        conncomp_result = {}

        burst_idx = 0
        for key in phase.keys():
            phase_ds = phase[key]
            weight_ds = weight[key] if weight is not None and key in weight else None

            if debug:
                print(f'\nProcessing burst {burst_idx}: {key}')
            burst_idx += 1

            # Get data variables (typically polarization like 'VV'), with y/x dims - excludes converted attributes;
            # a DataArray burst is its own one (N81)
            grids = BatchCore._grids_of(phase_ds)

            unwrap_vars = {}
            conncomp_vars = {}

            for var, phase_da in grids.items():
                weight_da = BatchCore._weight_of(weight_ds, var)

                # Save original spatial chunks to restore after unwrapping
                orig_y_chunks = None
                orig_x_chunks = None
                if hasattr(phase_da.data, 'chunks'):
                    orig_y_chunks = phase_da.data.chunks[-2]
                    orig_x_chunks = phase_da.data.chunks[-1]

                # Ensure pair dimension is chunked as 1 and spatial dims are single chunk
                rechunk_dict = {}
                if 'pair' in phase_da.dims:
                    chunks = phase_da.data.chunks
                    if chunks[0][0] != 1:
                        rechunk_dict['pair'] = 1
                if hasattr(phase_da.data, 'chunks'):
                    chunks = phase_da.data.chunks
                    if len(chunks[-2]) > 1:
                        rechunk_dict['y'] = -1
                    if len(chunks[-1]) > 1:
                        rechunk_dict['x'] = -1
                if rechunk_dict:
                    phase_da = phase_da.chunk(rechunk_dict)
                    if weight_da is not None:
                        weight_da = weight_da.chunk(rechunk_dict)

                # Save non-dimension coords along pair (ref, rep, BPR) for output
                pair_coords = {}
                n_pairs = None
                if 'pair' in phase_da.dims:
                    n_pairs = phase_da.sizes['pair']
                    for k, v in phase_da.coords.items():
                        if k != 'pair' and hasattr(v, 'dims') and v.dims == ('pair',):
                            vals = v.values if hasattr(v, 'values') else v
                            pair_coords[k] = ('pair', vals)
                    # Drop pair coordinate - use positional indexing only
                    if 'pair' in phase_da.indexes:
                        phase_da = phase_da.reset_index('pair', drop=True)
                    if weight_da is not None and 'pair' in weight_da.indexes:
                        weight_da = weight_da.reset_index('pair', drop=True)

                phase_dask = phase_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                lead = 'pair' in phase_da.dims
                params = (float(accuracy), int(max_iter), float(epsilon), int(conncomp_size), int(threads), bool(debug), lead)
                inds, out_ind = ('pyx', 'pcyx') if lead else ('yx', 'cyx')
                weight_arg = (weight_dask, inds) if weight_dask is not None else (None, None)
                # ONE TASK A PAIR on its raster's one block, y and x not contracted: dask fuses the inputs' chains into
                # the task. A layer that reads the blocks in the task (as fit1d_network() reads them) saves one transient copy
                # but stops that fusion, and two graphs sharing the inputs then hold their keys with different specs
                # (the scheduler's 'different run_spec' warning)
                result_dask = dask.array.blockwise(
                    utils_unwrap2d.irls_unwrap_block, out_ind, phase_dask, inds, *weight_arg, params, None,
                    dtype=np.float32, meta=np.empty((0,) * len(out_ind), dtype=np.float32), new_axes={'c': 2})
                unwrapped_dask = result_dask[:, 0] if lead else result_dask[0]
                conncomp_dask = (result_dask[:, 1] if lead else result_dask[1]).astype(np.uint16)

                # Build xarray results with proper dimensions
                # unwrapped_dask and conncomp_dask already have (n_pairs, y, x) shape
                result_dims = ('pair', 'y', 'x') if 'pair' in phase_da.dims else ('y', 'x')

                unwrap_da = xr.DataArray(
                    unwrapped_dask,
                    dims=result_dims,
                    coords={'y': phase_da.y, 'x': phase_da.x}
                )
                unwrap_da.attrs['units'] = 'radians'

                conncomp_da = xr.DataArray(
                    conncomp_dask,
                    dims=result_dims,
                    coords={'y': phase_da.y, 'x': phase_da.x}
                )

                # Assign non-dimension coords (ref, rep, BPR) - pair uses positional indexing
                if pair_coords:
                    unwrap_da = unwrap_da.assign_coords(**pair_coords)
                    conncomp_da = conncomp_da.assign_coords(**pair_coords)

                # Restore original spatial chunks (split is cheap)
                if orig_y_chunks is not None and (len(orig_y_chunks) > 1 or len(orig_x_chunks) > 1):
                    unwrap_da = unwrap_da.chunk({'y': orig_y_chunks, 'x': orig_x_chunks})
                    conncomp_da = conncomp_da.chunk({'y': orig_y_chunks, 'x': orig_x_chunks})

                unwrap_vars[var] = unwrap_da
                conncomp_vars[var] = conncomp_da

            if isinstance(phase_ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81): no metadata to carry
                unwrap_result[key] = BatchCore._form(phase_ds, unwrap_vars)
                conncomp_result[key] = BatchCore._form(phase_ds, conncomp_vars)
            else:
                # Preserve dataset attributes (subswath, pathNumber, etc.)
                unwrap_result[key] = xr.Dataset(Stack._carry_meta(phase_ds, unwrap_vars),
                                                attrs=phase_ds.attrs)
                conncomp_result[key] = xr.Dataset(Stack._carry_meta(phase_ds, conncomp_vars),
                                                  attrs=phase_ds.attrs)
            # Preserve CRS from input dataset
            if phase_ds.rio.crs is not None:
                unwrap_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)
                conncomp_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)

        from .Batches import Batches
        # the labels are integers, not units: a plain Batch (N81)
        return Batches((Batch(unwrap_result), Batch(conncomp_result)))

    @staticmethod
    def _unwrap2d_union(phase, weight=None, conncomp=False,
                        conncomp_size=1000, conncomp_gap=None,
                        conncomp_linksize=5, conncomp_linkcount=30,
                        debug=False, **kwargs):
        import numpy as np
        import xarray as xr
        from .BatchWrap import BatchWrap
        from .BatchCore import BatchCore
        from .Stack import Stack

        _batch_in = isinstance(phase, (BatchCore, Stack))
        # a Stack is read as its BatchComplex view (the same bursts, the merged per-burst Datasets): the rename and the
        # merge below act on that view; the result is the Batch from_dataset() gives, as for any batch phase (GKB U5)
        if isinstance(phase, Stack):
            phase = phase._batch_view()
        # a BatchUnit (N81), the shared check: any other weight raised only at the solve, or not at all
        weight = BatchCore._weight(weight, phase if _batch_in else None)
        # A BATCH OF DATAARRAYS merges to one DataArray and splits back to DataArrays (N81),
        # and a DataArray weight merges as one DataArray that weights every grid: the
        # variable of the Dataset to_dataset() returns
        # rule (3): the pair (date) chunks of the bursts AS GIVEN. to_dataset() below keeps them (the merged view has the
        # input's pair chunks), and the solve is one task per pair chunk; one chunk of many dates raises (union_check)
        if _batch_in:
            from .utils_dask import union_check
            for _b in phase.values():
                # the grids the solve reads, each with its own chunks: a Dataset's chunksizes raises when its variables
                # differ on a dimension, which is not the single-block case (notebook_chains T1)
                for _g in (_b.data_vars.values() if isinstance(_b, xr.Dataset) else (_b,)):
                    if 'y' in _g.dims and 'x' in _g.dims:
                        for _dim in ('pair', 'date'):
                            if _dim in _g.chunksizes:
                                union_check(_g.chunksizes[_dim], 'unwrap2d')
        _first = next(iter(phase.values()), None) if _batch_in else None
        _name = _first.name if isinstance(_first, xr.DataArray) else None
        # an unnamed phase is named for the merge only, as an unnamed weight is: to_dataset()
        # merges by name; the result is unnamed again, as unwrap2d() gives it per burst
        _unnamed = isinstance(_first, xr.DataArray) and _name is None
        if _unnamed:
            _name = 'phase'
            phase = phase._view({k: v.rename(_name) for k, v in phase.items()})
        _wfirst = next(iter(weight.values()), None) if weight is not None else None
        phase_ds = (phase.to_dataset(polarization=_name)[_name] if _name is not None
                    else phase.to_dataset()) if _batch_in else phase
        if weight is None:
            weight_ds = None
        elif isinstance(_wfirst, xr.DataArray):
            # an unnamed weight is named for the merge only: to_dataset() merges by name
            _wname = _wfirst.name if _wfirst.name is not None else 'weight'
            if _wfirst.name is None:
                weight = weight._view({k: v.rename(_wname) for k, v in weight.items()})
            weight_ds = weight.to_dataset(polarization=_wname)[_wname]
        elif _unnamed:
            # a Dataset weight weights an unnamed phase by its one grid (_weight_of)
            weight_ds = BatchCore._weight_of(weight.to_dataset(), None)
        else:
            weight_ds = weight.to_dataset(polarization=_name)[_name] if _name is not None else weight.to_dataset()
        if not (isinstance(phase_ds, xr.Dataset) or (_batch_in and isinstance(phase_ds, xr.DataArray))):
            raise TypeError(f"phase must be a Batch or xr.Dataset, got "
                            f"{type(phase).__name__}")

        if _batch_in:
            # THE GRIDS ONLY into the solve: from_dataset() below gives each burst its own metadata back.
            # The merged metadata went through the solve and came back in place of the burst's own
            def _grids(ds_):
                # a merged DataArray is its own one grid (N81)
                if isinstance(ds_, xr.DataArray):
                    return ds_
                return ds_[[v for v in ds_.data_vars if 'y' in ds_[v].dims and 'x' in ds_[v].dims]]
            phase_ds = _grids(phase_ds)
            weight_ds = _grids(weight_ds) if weight_ds is not None else None

        # THE SOLVE: one task per input pair chunk (rule (3)). dask concatenates the YX blocks of a chunk into its task,
        # the task loops over the pairs of the chunk in numpy, and the output keeps the input's pair chunks: nothing is
        # rechunked here. The merged raster is the single burst of this solve.
        import dask.array as da
        accuracy = kwargs.pop('accuracy', 2e-3)
        max_iter = kwargs.pop('max_iter', 100)
        epsilon = kwargs.pop('epsilon', 1e-2)
        unknown = sorted(set(kwargs) - set(utils_unwrap2d._OLD_IRLS))
        if unknown:
            raise TypeError(f'unwrap2d_irls() got unexpected keyword arguments: {", ".join(unknown)}')
        utils_unwrap2d.old_irls_warning(kwargs, 'unwrap2d()')
        # the phase is wrapped as BatchWrap wraps it (its constructor, the float check included)
        phase_ds = BatchWrap({'': phase_ds})['']
        grids = BatchCore._grids_of(phase_ds)
        rasters = sum(da_.sizes['pair'] if 'pair' in da_.dims else 1 for da_ in grids.values())
        threads = utils_unwrap2d.unwrap_threads(rasters)
        if debug:
            print(f'DEBUG unwrap2d: {rasters} rasters, {threads} numba threads each')
        link = dict(conncomp_size=conncomp_size, conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                    conncomp_linkcount=conncomp_linkcount, debug=debug)

        def unwrap2d_union_block(phase_b, weight_b, params):
            """The pairs of one input chunk, with its YX blocks joined: each pair is solved and, if asked, linked as one raster."""
            accuracy, max_iter, epsilon, conncomp_size, threads, debug, conncomp, link, lead = params
            if not lead:
                phase_b = phase_b[np.newaxis]
                weight_b = None if weight_b is None else weight_b[np.newaxis]
            out = np.empty((phase_b.shape[0], 2) + phase_b.shape[1:], dtype=np.float32)
            for p in range(phase_b.shape[0]):
                w = None if weight_b is None else weight_b[p:p + 1]
                res = utils_unwrap2d.irls_unwrap_block(phase_b[p:p + 1], w, (accuracy, max_iter, epsilon, conncomp_size,
                                                                            threads, debug, True))[0]
                unwrapped, labels = res[0], res[1]
                if not conncomp:
                    unwrapped = utils_unwrap2d.link_components_block(
                        unwrapped[np.newaxis], labels[np.newaxis].astype(np.uint16), **link)[0]
                out[p, 0], out[p, 1] = unwrapped, labels
            return out if lead else out[0]

        from .utils_dask import align_secondary, warn_secondary
        unwrap_vars, conncomp_vars = {}, {}
        moved = []
        for var, phase_da in grids.items():
            weight_da = BatchCore._weight_of(weight_ds, var)
            lead = 'pair' in phase_da.dims
            # a SECONDARY weight (GMT U3): its pair (date) chunks must be the phase's, or one line raises (as the phase's date
            # chunks raise in union_check); its (y, x) chunks go to the phase's, one WARNING per call (align_secondary)
            w_data = None
            if weight_da is not None:
                for dim in ('pair', 'date'):
                    p_ch, w_ch = phase_da.chunksizes.get(dim), weight_da.chunksizes.get(dim)
                    if p_ch is not None and w_ch is not None and p_ch != w_ch:
                        raise ValueError(f"unwrap2d(union=True): the weight's {dim} chunks {w_ch} differ from the phase's "
                                         f"{p_ch}: chunk the weight as the phase")
                w_data, changed = align_secondary(phase_da.data, weight_da.data, 'unwrap2d', 'weight', check_dates=False)
                if changed:
                    moved.append('weight')
            inds, out_ind = ('pyx', 'pcYX') if lead else ('yx', 'cYX')
            weight_arg = (da.asarray(w_data), inds) if weight_da is not None else (None, None)
            params = (float(accuracy), int(max_iter), float(epsilon), int(conncomp_size), int(threads), bool(debug),
                      bool(conncomp), link, lead)
            solved = da.blockwise(
                unwrap2d_union_block, out_ind, da.asarray(phase_da.data), inds, *weight_arg, params, None,
                concatenate=True, new_axes={'c': 2, 'Y': phase_da.sizes['y'], 'X': phase_da.sizes['x']},
                dtype=np.float32, meta=np.empty((0,) * len(out_ind), dtype=np.float32))
            unwrapped_dask = solved[:, 0] if lead else solved[0]
            conncomp_dask = (solved[:, 1] if lead else solved[1]).astype(np.uint16)
            # the coordinates of the pair chunk: the same as unwrap2d_irls() gives its results
            base = phase_da.reset_index('pair', drop=True) if lead and 'pair' in phase_da.indexes else phase_da
            pair_coords = ({k: ('pair', v.values) for k, v in phase_da.coords.items()
                            if k != 'pair' and v.dims == ('pair',)} if lead else {})
            dims = ('pair', 'y', 'x') if lead else ('y', 'x')
            unwrap_da = xr.DataArray(unwrapped_dask, dims=dims, coords={'y': base.y, 'x': base.x})
            if conncomp:
                # the linked result (conncomp=False) has no attrs, as _link_components() gives it
                unwrap_da.attrs['units'] = 'radians'
            conncomp_da = xr.DataArray(conncomp_dask, dims=dims, coords={'y': base.y, 'x': base.x})
            if pair_coords:
                unwrap_da = unwrap_da.assign_coords(**pair_coords)
                conncomp_da = conncomp_da.assign_coords(**pair_coords)
            unwrap_vars[var] = unwrap_da
            conncomp_vars[var] = conncomp_da
        warn_secondary('unwrap2d', moved)
        if isinstance(phase_ds, xr.DataArray):
            unwrap_obj = BatchCore._form(phase_ds, unwrap_vars)
            conncomp_obj = BatchCore._form(phase_ds, conncomp_vars)
        else:
            unwrap_obj = xr.Dataset(Stack._carry_meta(phase_ds, unwrap_vars), attrs=phase_ds.attrs)
            conncomp_obj = xr.Dataset(Stack._carry_meta(phase_ds, conncomp_vars), attrs=phase_ds.attrs)
        if phase_ds.rio.crs is not None:
            unwrap_obj.rio.write_crs(phase_ds.rio.crs, inplace=True)
            conncomp_obj.rio.write_crs(phase_ds.rio.crs, inplace=True)
        merged = [unwrap_obj, conncomp_obj] if conncomp else [unwrap_obj]

        if not _batch_in:
            graph_check('_unwrap2d_union', merged, phase)
            return tuple(merged) if conncomp else merged[0]

        # keep only each burst's own pixels; overlaps were filled from neighbours
        valid = phase.map_da(lambda da: da.notnull())
        out = [phase.from_dataset(m).where(valid) for m in merged]
        if _unnamed:
            out = [o._view({k: v.rename(None) for k, v in o.items()}) for o in out]
        graph_check('_unwrap2d_union', out, phase)
        return tuple(out) if conncomp else out[0]
