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
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from . import utils_complex
from . import utils_rotate
from . import utils_xarray
from .Batch import Batch
from .BatchWrap import BatchWrap
from .utils_detrend import _trend2d_accumulate_for_dask, _trend2d_accumulate_partial_for_dask, _trend2d_finalize_for_dask
from typing import TYPE_CHECKING
from .utils_dask import graph_check
if TYPE_CHECKING:
    from .Stack import Stack
    from .BatchUnit import BatchUnit


def _apply_goldstein_2d_for_dask(phase_block, corr_block, psize=32, threshold=0.5, halo=None, block_info=None):
    """Module-level function for Goldstein filter map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles 2D (y, x) blocks and 3D (pair, y, x) blocks, whose pairs are filtered one by one.

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
    if phase_block.ndim == 2:
        return BatchComplex._goldstein(phase_block, corr_block, psize=psize, threshold=threshold, skip=skip)
    # the pairs of the block are looped here, one 2D filter each; the block keeps its chunks
    if phase_block.shape[0] == 0:
        return np.zeros(phase_block.shape, dtype=np.complex64)
    return np.stack([BatchComplex._goldstein(p, c, psize=psize, threshold=threshold, skip=skip)
                     for p, c in zip(phase_block, corr_block)], axis=0)


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
        from . import utils_fit1d

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
                    bp = utils_fit1d.date_baseline(ds[baseline].values, len(dates))
                    # elevation_phase() = 4 pi / (lambda R sin(inc))
                    if _ep_all is None:
                        _ep_all = Batch._elevation_phase_approximate(self)
                    _fac = _ep_all[burst_id]
                    ele2phase = utils_fit1d.ele2phase(bp, lam_, _fac)
            if meter2rad is None:
                raise TypeError(
                    f'fit1d() needs radar_wavelength in burst {burst_id} to '
                    f'turn a rotation rate into a velocity')

            # t = 0 at the master (utils_fit1d.master_time_origin: smallest |B_perp|,
            # else the first date); it is written to the model's `date`, which
            # predict() reads as its origin
            tyr, _master = utils_fit1d.master_time_origin(dday, bp)

            data_da = ds[pol]
            if data_da.dims[0] != 'date':
                data_da = data_da.transpose('date', ...)
            data_dask = data_da.data

            # the per-block kernel is utils_fit1d.fit1d_dates_block; the defaults bind
            # this burst's values now, the graph runs later
            def _fit_block(block, _h=ele2phase, _t=tyr, _m=meter2rad,
                           _mh=max_dh, _mv=max_dv,
                           _sh=step_dh, _sv=step_dv,
                           _thr=threshold, _adaptive=adaptive):
                return utils_fit1d.fit1d_dates_block(block, _h, _t, _m, _mh, _mv,
                                                     _sh, _sv, _thr, _adaptive)

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
        graph_check('fit1d', model_result, self)
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
        from . import utils_fit1d
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
            # THE NOTE MARKS A REAL COMPUTATION: a number the model needs still holds work to compute. A read of a store
            # or of a numpy array, a selection of either, and the pieces the cluster holds are a conversion (ruling 2026-10-14)
            import dask
            from dask.highlevelgraph import MaterializedLayer
            from distributed import Future
            _reads = ('held-', 'array-', 'original-array-', 'from-zarr-', 'original-from-zarr-', 'open_dataset-',
                      'original-open_dataset-', 'getitem-', 'original-getitem-')
            if any(dask.is_dask_collection(model[k][n].data) and any(
                       not name.startswith(_reads) and not (isinstance(lay, MaterializedLayer) and any(
                           isinstance(x, Future) for x in lay.mapping.values()))
                       for name, lay in model[k][n].data.dask.layers.items())
                   for k in self.keys() if k in model
                   for n in ['intercept'] + [f'slope_{v}' for v in model[k].attrs['trend2d_vars']]
                   if n in model[k].data_vars):
                print("NOTE: BatchComplex.predict() computes the trend2d() model's numbers eagerly.")
            out = self._trend2d_predict(model, vars=vars)
            graph_check('predict', out, self)
            return out
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
            t = utils_fit1d.model_time(dday, t0)

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
                ele2phase = utils_fit1d.elevation_to_phase(bp, _lam, _fac)

            vel = mds['velocity'].data
            hgt = mds['height'].data
            # OPTIONAL: fit1d_network() fits an annual term and fit1d() does not, and
            # a model without the variable HAS none rather than an unknown one.
            sea = mds['seasonal'].data if 'seasonal' in mds else None
            pred = utils_fit1d.predict_dates(vel, hgt, sea, t, ele2phase)
            if ref is not None:
                # WHICH ACQUISITION READS ZERO -- see the docstring. Every
                # difference is invariant to this, so it cannot change a
                # removal or a rate, only which plane sits at zero.
                _i = Batch._ref_index(ref, ds.coords['date'].values)
                pred = utils_fit1d.reference_plane(pred, _i)

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
        graph_check('predict', out, self)
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
        from . import utils_fit1d
        out = {}
        for key in disp.keys():
            vars_ = BatchCore._vars_of(disp[key])
            pol = next(iter(vars_))
            mixed = utils_fit1d.mix_dates(BatchCore._vars_of(self[key])[pol].data,
                                          BatchCore._vars_of(full[key])[pol].data,
                                          vars_[pol].data)
            out[key] = BatchCore._form(disp[key], {pol: vars_[pol].copy(data=mixed)},
                                       attrs=disp[key].attrs)
        graph_check('mix', out, self)
        return Batch(out)

    def trend2d(self, *vars, transform=None, range: float = None,
                union: bool = False, debug: bool = False) -> 'Batch':
        """
        Spatial trend of the complex phase, PER DATE, as a unit-magnitude
        phasor: `phi_d = sum_i g_di * (v_i - z_i) + k_d`, or `phi_d = k_d`
        alone when no covariate is given, so removing it is a rotation.

        >>> ps_mask = stack.psfunction() > 1
        >>> trend = stack.where(ps_mask).trend2d('northing', 'easting', 'ele')
        >>> trend = stack.where(ps_mask).trend2d(
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
            'gain', 'pixels'; the covariate names, their scaling ('trend2d_stats')
            and the grid ('trend2d_cells') ride as attributes. With no
            covariate only 'intercept', 'stderr_intercept', 'coherence',
            'coherence0', 'gain' and 'pixels', with no covariate named. No
            raster: `stack.predict(trend)` evaluates the phase per chunk when
            asked, `stack.detrend(trend)` removes it. Each variable is one chunk
            over all its dates.

        Raises
        ------
        ValueError
            A covariate's values lie outside its stored range (one ERROR line names it).
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
                if tuple(var_dask.shape) != _sh:
                    raise ValueError(
                        f"trend2d(): '{var}' is {tuple(var_dask.shape)} over "
                        f"{_d} but the phase is {tuple(data_dask.shape[-2:])} "
                        f"over ('y', 'x') for '{key}'.")
                if var_dask.chunks != _ch:
                    # no rechunk: each data block joins the covariate's own blocks it overlaps
                    var_dask = utils_detrend.trend2d_cover(var_dask, _ch)
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
            else:
                _iref = int(np.argmin(np.abs(_b)))
            # just the conjugate: samples are normalised to unit phasors
            # after this, so the reference's magnitude cancels; a single
            # date has no reference and keeps its phase as it is
            _R = None if nd < 2 else data_da.isel(date=_iref).data

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
                          'data_da': data_da, 'data_ref': utils_detrend.trend2d_referenced(data_dask, _R),
                          'vars_dask': vars_dask, 'var_names': var_names,
                          'nd': nd, 'k': k, 'stats': stats,
                          'zero': _zeros,
                          'vars_dims': vars_dims, 'iref': _iref})

        if not preps:
            empty = BatchComplex({})
            graph_check('trend2d', empty, self)
            return empty
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
            range = utils_detrend.trend2d_default_range(k)
        _cells = utils_detrend.trend2d_cells(range)
        # axis-vector covariates (northing/easting) get a ramp start from
        # their marginal profiles -- their sampling is near-uniform, so the
        # 1-D scan is clean and a multi-cycle ramp becomes reachable
        _axes = tuple(i for i, _d in enumerate(preps[0]['vars_dims'])
                      if len(_d) == 1)[:2]
        if any(p['k'] != k for p in preps):
            raise ValueError("trend2d() got a different number of variables "
                             "for different bursts.")

        if union:
            # rule (3): one task per date chunk of the bursts AS GIVEN; one date chunk of many dates raises here
            from .utils_dask import union_check
            for p in preps:
                union_check(p['data_ref'].chunks[0], 'trend2d')
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
            _stats = utils_detrend.trend2d_stats([p['stats'] for p in grp], k)
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
                _extent = (float(_yc.min()), float(_yc.max()),
                           float(_xc.min()), float(_xc.max()))
                if union:
                    # rule (3): ONE accumulator task per date chunk of this burst, which joins that
                    # chunk's YX blocks as numpy inside (dask concatenates the blocks along the
                    # contracted y and x). Nothing is rechunked and no chunk is split.
                    _acc.append(da.blockwise(
                        _trend2d_accumulate_for_dask, 'df',
                        _ref, 'dyx', *_args, _yd, 'y', _xd, 'x',
                        stats=_stats, cells=_cells, n_vars=k,
                        names=tuple(p['var_names']),
                        dims=[''.join(_d) for _d in p['vars_dims']],
                        extent=_extent,
                        new_axes={'f': _w},
                        concatenate=True, dtype=np.float64,
                        meta=np.empty((0, 0), np.float64)))
                else:
                    # union=False: one partial per YX block, summed over the blocks; the blocks
                    # are NOT joined (the per-burst fit keeps its own summation order)
                    _acc.append(da.blockwise(
                        _trend2d_accumulate_partial_for_dask, 'dyxf',
                        _ref, 'dyx', *_args, _yd, 'y', _xd, 'x',
                        stats=_stats, cells=_cells, n_vars=k,
                        names=tuple(p['var_names']),
                        dims=[''.join(_d) for _d in p['vars_dims']],
                        extent=_extent,
                        new_axes={'f': _w}, adjust_chunks={'y': 1, 'x': 1},
                        concatenate=True, dtype=np.float64,
                        meta=np.empty((0, 0, 0, 0), np.float64)).sum(axis=(1, 2)))
            # THE DATES KEEP THE CALLER'S CHUNKS: each date chunk's accumulator
            # is solved in its own task, which fits the dates it holds one by one;
            # the model joins them below.
            _total = sum(_acc[1:], _acc[0])
            _dts = da.from_array(
                np.asarray(grp[0]['data_da'].coords['date'].values)
                .astype('datetime64[D]').astype(np.int64), chunks=_total.chunks[0])
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
            # THE MODEL IS ONE CHUNK OVER ITS DATES (merged by design, as polynomial(); the user 2026-11-03)
            _coef = utils_detrend.trend2d_model_join(_coef)
            if debug:
                # the table needs the numbers, so they are computed here, ONCE,
                # and the model carries them: computing it runs nothing again.
                # The coefficients are per date: one chunk holds them all.
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
                utils_detrend.trend2d_table(_coef, _stats, k, p['var_names'], degree, p['key'], p['nd'], p['iref'], union)
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
            _icpt, _slopes, _zeros = utils_detrend.trend2d_recenter(
                _coef, _names, p['zero'], _stats, k)
            o['intercept'] = xr.DataArray(_icpt, dims=('date',))
            # a constant names no covariate, so predict() needs none
            o.attrs['trend2d_vars'] = list(p['var_names']) if degree else []
            o.attrs['trend2d_dims'] = ([''.join(_d) for _d in p['vars_dims']]
                                       if degree else [])
            # aligned with trend2d_vars: where each variable contributes
            # nothing, and for a square the point it is squared about
            o.attrs['trend2d_zero'] = _zeros
            o.attrs['trend2d_ref'] = int(p['iref'])
            # the scaling each covariate was fitted with (centre, then extent)
            # and the grid: what predict() checks its covariates against
            o.attrs['trend2d_cells'] = int(_cells)
            o.attrs['trend2d_stats'] = [float(x) for x in _stats] if degree else []
            _mod = utils_detrend.trend2d_model_arrays(_coef, k, len(_names))
            o['coherence'] = xr.DataArray(_mod['coherence'], dims=('date',))
            o['coherence0'] = xr.DataArray(_mod['coherence0'], dims=('date',))
            # what removing the trend bought; ~0.07 is reachable by fitting
            # noise alone, so above that a trend was genuinely found
            o['gain'] = xr.DataArray(_mod['gain'], dims=('date',))
            o['pixels'] = xr.DataArray(_mod['pixels'], dims=('date',))
            if not degree:
                # the constant's one-sigma, radians: half the disagreement of
                # the two checkerboard halves' constants
                o['stderr_intercept'] = xr.DataArray(
                    _mod['stderr_intercept'], dims=('date',))
            for i, var in enumerate(p['var_names'] if degree else []):
                # per unit of the covariate, all three, so they read against
                # each other: the slope, how far apart two slopes must be for
                # this sampling to tell them apart, and the slope's one-sigma
                o[f'slope_{var}'] = xr.DataArray(_slopes[i], dims=('date',))
                o[f'resolution_{var}'] = xr.DataArray(
                    _mod['resolution'][i], dims=('date',))
                o[f'stderr_{var}'] = xr.DataArray(
                    _mod['stderr'][i], dims=('date',))
            if 'spatial_ref' in ds.coords:
                o = o.assign_coords(spatial_ref=ds['spatial_ref'].drop_vars(
                    list(ds['spatial_ref'].coords), errors='ignore'))
            out[p['key']] = o
        graph_check('trend2d', out, self)
        return Batch(out)

    def _trend2d_predict(self, model, vars=None, who='predict()') -> 'Batch':
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
        from . import utils_detrend
        out = {}
        from . import utils_dask
        moved = []   # the covariates rechunked in this call
        for key, ds in self.items():
            if key not in model:
                raise KeyError(f"predict(): the model has no burst '{key}'.")
            mds = model[key]
            names = list(mds.attrs['trend2d_vars'])
            dims = list(mds.attrs['trend2d_dims'])
            # the fit's scaling and grid: the covariates are checked against them;
            # a model fitted before they were stored is evaluated unchecked, with a warning
            _stats = mds.attrs.get('trend2d_stats')
            _cells = mds.attrs.get('trend2d_cells')
            _checked = (_stats is not None and _cells is not None
                        and len(_stats) == 2 * len(names))
            if names and not _checked:
                import warnings
                warnings.warn(
                    f"{who}: the trend2d() model of '{key}' carries no usable "
                    f"trend2d_stats or trend2d_cells (fitted before they were "
                    f"stored), so its covariates are not checked against its range",
                    stacklevel=2)
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
            _num = dict(zip(_need, dask.compute(*[mds[n].data for n in _need])))
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

            # the fit's (centre, extent) per covariate, its grid and the covariate count
            _scale = (_stats, int(_cells), len(names)) if _checked else None

            terms = []
            for i, (v, d) in enumerate(zip(names, dims)):
                _z = float(_zero[i]) if i < len(_zero) else 0.0
                if v.endswith('²'):
                    # A SQUARE IS REBUILT, NEVER LOOKED UP: the model records
                    # the point it is squared about, so a crop with another
                    # extent still evaluates the model that was fitted
                    _bc = _plain(v[:-1])
                    cov = (None if _bc is None else _bc.copy(
                        data=utils_detrend.trend2d_covariate(
                            _bc.data, True, _z, _scale, i, v, who)))
                else:
                    cov = _plain(v)
                    if cov is not None:
                        cov = cov.copy(data=utils_detrend.trend2d_covariate(
                            cov.data, False, _z, _scale, i, v, who))
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
                terms.append((cov.data, _d, _num[f'slope_{v}'], v))
            # THE PHASE over the data's chunks, in float64: each covariate is broadcast over the dates before it meets
            # the model's per-date slope. Multiplied as (dates, 1, 1) x (1, y, x), dask unifies the date chunks with the
            # block chunks: a PerformanceWarning and a rechunk of the dates, the factor being the block count.
            phi = da.from_array(np.asarray(_num['intercept'], np.float64), chunks=(_dc,))[:, None, None]
            for cov, _d, slope, v in terms:
                _ch = tuple(data.chunks[1 + 'yx'.index(a_)] for a_ in _d)
                cd = cov if isinstance(cov, da.Array) else da.from_array(np.asarray(cov, np.float32), chunks=_ch)
                cd = cd.astype(np.float64)
                # a covariate is a SECONDARY input (GMT U2): its (y, x) chunks go to the data's, one WARNING per call
                cd, changed = utils_dask.align_secondary(data, cd, who.rstrip('()'), f"covariate '{v}'", dims=_d)
                if changed:
                    moved.append(f"covariate '{v}'")
                cd = da.broadcast_to(cd[(None,) + tuple(slice(None) if a_ in _d else None for a_ in ('y', 'x'))],
                                     data.shape, chunks=data.chunks)
                gd = da.from_array(np.asarray(slope, np.float64), chunks=(_dc,))
                phi = phi + gd[:, None, None] * cd
            # a constant does not vary along the raster: the phase is broadcast to the raster it is applied to
            if tuple(phi.shape) != tuple(data.shape):
                phi = da.broadcast_to(phi, data.shape, chunks=data.chunks)
            phi = phi.astype(np.float32)
            coords = {k_: v_ for k_, v_ in ref.coords.items()
                      if k_ in ('date', 'y', 'x', 'spatial_ref')}
            # a DataArray in, a DataArray out (N81): the trend's 'phase'
            out[key] = BatchCore._form(ds, {'phase': xr.DataArray(
                phi, dims=('date', 'y', 'x'),
                coords=coords)}, attrs=dict(ds.attrs))
        utils_dask.warn_secondary(who.rstrip('()'), moved)
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
        Remove a trend from EVERY polarisation: `self * exp(-1j * phase)`, with the
        phase of `trend`, one of three inputs:

        - a phase Batch: stack.troposphere_phase() + stack.ionosphere_phase(), or
          predict() of a model; one real (date, y, x) variable per burst;
        - a trend2d() model, evaluated on the stack's own covariates;
        - an insardev_polsar trend3d() field, removed by its detrend3d().

        detrend(stack.troposphere_phase() + stack.ionosphere_phase()) applies the
        atmosphere correction to every polarisation, before any polarisation optimisation. The default
        phase is referenced to the reference date, so the correction subtracts that
        date's phase from every date.

        Parameters
        ----------
        trend : Batch
            One of the three inputs. A phase keeps the stack's dates and order; its NaN is applied as it is; a date it lacks stays uncorrected.
            Its (y, x) chunks may differ from the stack's: it is rechunked to them, with one WARNING.
            A trend2d() model returns its own dates.
        vars : Batch or None
            Covariates of a trend2d() model, when the stack does not carry them.

        Raises
        ------
        TypeError
            A phase that is not real float (complex, boolean, integer, datetime, timedelta, object, string), one with two gridded variables or none; `vars` with a phase.
        ValueError
            A phase on another (y, x) grid, or with a date the stack lacks; a phase with other date chunks than the
            stack's; polarisations chunked differently.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        from . import utils_detrend
        # a trend3d() FIELD, recognised by what it carries: insardev_polsar's detrend3d() removes it
        if BatchComplex._trend3d_field(trend, self.keys()):
            if vars is not None:
                raise TypeError("detrend(): `vars` are the covariates of a trend2d() "
                                "model; a trend3d() field names none.")
            return self._trend3d_method('detrend3d', 'detrend')(self, trend)

        def phase_of(key, ds):
            """The phase of one burst of a phase Batch, as (date, y, x): one real gridded
            variable on the stack's own (y, x) grid, with dates the stack holds."""
            if key not in trend:
                raise ValueError(f"ERROR: detrend(): the trend has no burst '{key}'.")
            vs = BatchCore._vars_of(trend[key])
            grids = [n for n, a in vs.items() if 'y' in a.dims and 'x' in a.dims]
            if len(grids) > 1:
                raise TypeError(f"ERROR: detrend(): the trend of burst '{key}' holds {len(grids)} gridded "
                                f"variables ({', '.join(map(str, grids))}); a phase is ONE real variable.")
            if not grids:
                raise TypeError(f"ERROR: detrend(): the trend of burst '{key}' holds no gridded variable "
                                f"({', '.join(map(str, vs)) or 'none'}); a phase holds one.")
            a = vs[grids[0]]
            if a.dtype.kind == 'c':
                raise TypeError(f"ERROR: detrend(): the phase of burst '{key}' is {a.dtype} {tuple(a.dims)}; "
                                f"a phase is one real (date, y, x) variable.")
            if a.dtype.kind != 'f':
                raise TypeError(f"ERROR: detrend(): the phase of burst '{key}' is {a.dtype} {tuple(a.dims)}; "
                                f"a phase is one real float (date, y, x) variable.")
            if set(a.dims) != {'date', 'y', 'x'}:
                raise ValueError(f"ERROR: detrend(): the phase of burst '{key}' is {a.dtype} {tuple(a.dims)}; "
                                 f"a phase is one real (date, y, x) variable.")
            if not (np.array_equal(a['y'].values, ds['y'].values) and np.array_equal(a['x'].values, ds['x'].values)):
                raise ValueError(f"ERROR: detrend(): the phase of burst '{key}' is not on the stack's (y, x) grid.")
            dates = np.asarray(a['date'].values)
            miss = dates[~np.isin(dates, np.asarray(ds['date'].values))]
            if miss.size:
                raise ValueError(f"ERROR: detrend(): the phase of burst '{key}' holds {miss.size} date(s) "
                                 f"the stack does not: {[str(d) for d in miss[:5]]}.")
            return a.transpose('date', 'y', 'x')

        # a trend2d() MODEL is recognised by what it carries (its trend2d_* attributes); a burst without trend2d_vars is named
        is_model = any(key in trend and any(str(a).startswith('trend2d_') for a in trend[key].attrs)
                       for key in self.keys())
        if is_model:
            for key in self.keys():
                if key not in trend or 'trend2d_vars' not in trend[key].attrs:
                    raise ValueError(f"ERROR: detrend(): no trend2d() model for burst '{key}' -- "
                                     f"pass what trend2d() returned for this stack.")
            # the NOTE marks a real computation, as in predict(): see there
            import dask
            from dask.highlevelgraph import MaterializedLayer
            from distributed import Future
            _reads = ('held-', 'array-', 'original-array-', 'from-zarr-', 'original-from-zarr-', 'open_dataset-',
                      'original-open_dataset-', 'getitem-', 'original-getitem-')
            if any(dask.is_dask_collection(trend[k][n].data) and any(
                       not name.startswith(_reads) and not (isinstance(lay, MaterializedLayer) and any(
                           isinstance(x, Future) for x in lay.mapping.values()))
                       for name, lay in trend[k][n].data.dask.layers.items())
                   for k in self.keys() if k in trend
                   for n in ['intercept'] + [f'slope_{v}' for v in trend[k].attrs['trend2d_vars']]
                   if n in trend[k].data_vars):
                print("NOTE: BatchComplex.detrend() computes the trend2d() model's numbers eagerly.")
            pred = self._trend2d_predict(trend, vars=vars, who='detrend()')
            phase = {key: BatchCore._vars_of(pred[key])['phase'] for key in self.keys()}
        else:
            if vars is not None:
                raise TypeError(f"ERROR: detrend(): `vars` ({type(vars).__name__}) came with a phase; "
                                "`vars` are the covariates of a trend2d() model only.")
            phase = {key: phase_of(key, ds) for key, ds in self.items()}
        res = {}
        from . import utils_dask
        moved = []   # the phases rechunked in this call: one WARNING at its end
        for key, ds in self.items():
            phi = phase[key]
            _pd = np.asarray(phi.coords['date'].values)
            _sd = np.asarray(ds.coords['date'].values)
            keep = None
            # THE PRIMARY inputs: the complex (date, y, x) variables of the burst, chunked alike (every polarisation)
            prim = {v: a.data for v, a in BatchCore._vars_of(ds).items()
                    if a.dtype.kind == 'c' and set(a.dims) == {'date', 'y', 'x'} and isinstance(a.data, da.Array)}
            utils_dask.check_primary_chunks(prim, 'detrend')
            # THE SECONDARY input, the phase (a model's chunks are the data's by construction): its (y, x) chunks go to
            # the data's, one WARNING per call. Its DATE chunks are the data's, never rechunked (GMT U2, GMT U2c): on the
            # stack's dates, or on the dates the phase has when it lacks some; other date chunks raise one line
            if prim and not is_model and isinstance(phi.data, da.Array):
                prim0 = next(iter(prim.values()))
                same = _pd.shape == _sd.shape and np.array_equal(_pd, _sd)
                if not same:
                    index = {d: i for i, d in enumerate(_sd)}
                    pos = np.array([index[d] for d in _pd if d in index], dtype=int)
                    # the data's date blocks restricted to the phase's dates, empty blocks dropped (no data read)
                    blk = np.searchsorted(np.cumsum(prim0.chunks[0]), pos, side='right')
                    want = tuple(int(n) for n in np.bincount(blk, minlength=len(prim0.chunks[0])) if n > 0)
                    if tuple(phi.data.chunks[0]) != tuple(want):
                        raise ValueError(f"ERROR: detrend(): the date chunks of the phase {utils_dask._sizes(phi.data.chunks[0])} "
                                         f"differ from the data's {utils_dask._sizes(want)} on the phase's dates: a secondary "
                                         "input takes the data's date chunks")
                pdata, changed = utils_dask.align_secondary(prim0, phi.data, 'detrend', 'phase', check_dates=same)
                if changed:
                    phi = phi.copy(data=pdata)
                    moved.append('phase')
            if is_model:
                if _pd.shape != _sd.shape or not np.array_equal(_pd, _sd):
                    # A date the trend does not carry is DROPPED, never passed through
                    # untouched -- a stack mixing detrended and raw dates is not a stack.
                    ds = ds.sel(date=_pd)
            elif _pd.shape != _sd.shape or not np.array_equal(_pd, _sd):
                # a phase keeps the stack's dates in the stack's order, matched by label;
                # a date the phase lacks is not rotated (keep is False there)
                keep = np.isin(_sd, _pd)
                phi = phi.reindex(date=_sd, fill_value=0)
            # THE SAME NUMBERS THE RASTER USED TO HOLD: float32 phase into a
            # complex64 exponential, rotated out of every complex (date, y, x) variable
            rot = utils_detrend.phase_rotation(phi.data)
            # a phase with the data's chunks and dates: exp(-1j * phase) and the product in ONE blockwise layer per variable
            fused = keep is None and not is_model and isinstance(phi.data, da.Array)

            def rotate(d, p):
                """The block d times exp(-1j * p), the numbers of utils_detrend.phase_rotate, in one numba pass (utils_rotate)."""
                return utils_rotate.rotate_block(d, p)

            upd = {}
            for v, a in BatchCore._vars_of(ds).items():
                if a.dtype.kind == 'c' and 'y' in a.dims and 'x' in a.dims and 'date' in a.dims:
                    if fused and a.chunks is not None and tuple(a.dims) == ('date', 'y', 'x'):
                        # the chunks match (checked above): no rechunk, no unify; the unused rot adds no layer
                        upd[v] = a.copy(data=da.map_blocks(rotate, a.data, phi.data,
                                                           dtype=np.result_type(a.dtype, np.complex64))).rename(None)
                        continue
                    # the stack's own chunks, whatever the phase's
                    r = xr.DataArray(rot, dims=phi.dims).transpose(*a.dims)
                    if a.chunks is not None:
                        r = xr.DataArray(r.data.rechunk(a.chunks), dims=a.dims)
                    # the dates the phase lacks keep the stack's values, bitwise (the mask is in utils)
                    upd[v] = a.copy(data=utils_detrend.phase_rotate(a.data, r.data, keep, a.dims.index('date'))).rename(None)
                    if keep is not None and not keep.all():
                        # the masked result has the date dim first, as xr.where gave it
                        upd[v] = upd[v].transpose('date', *[d for d in a.dims if d != 'date'])
            if isinstance(ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81)
                res[key] = BatchCore._form(ds, upd)
                continue
            res[key] = ds.assign(upd)
        utils_dask.warn_secondary('detrend', moved)
        out = type(self)(res)
        # the graph rule: every dask input of the call (the trend2d model's graph included) is not added
        graph_check('detrend', out, [self, trend, vars])
        return out

    def detrend2d(self, trend: 'Batch', vars=None) -> 'BatchComplex':
        """Alias of detrend(), for code that names the dims: prints a WARNING and calls
        detrend(trend, vars), which reads what the trend is: a phase, a trend2d() model or a trend3d() field
        (user, 2026-10-01)."""
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

          nodes    every pixel the arc test certifies; nothing is thinned, since
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
        from .Batch import Batch
        # the baselines and the radar geometry are the Dataset's (N81)
        BatchCore._needs_dataset(self, 'fit1d_network', ('BPR',) + BatchCore._GEOMETRY)
        # DELEGATE BY STACK TYPE, exactly as fit1d() does: pairs take the
        # pair branch (not implemented), dates take the PS network below.
        if any('pair' in ds[v].dims
               for ds in self.values() for v in ds.data_vars):
            # `window` is the PS network's (DS, PS extent) pair in metres on
            # the date path; on the pair path it is the side of the box the
            # covariance is estimated over, so a scalar.
            # the split is kept so a pair-domain fit has a home when one works
            raise NotImplementedError(
                'fit1d_network() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')
        # validated HERE, not only in the kernel: fit1d_network() returns a lazy Batch,
        # so a bad value would otherwise surface at compute time far from where
        # it was written
        utils_fit1d._3d_consensus(consensus)
        if int(level) < 0:
            raise ValueError(f'level must be >= 0; got {level!r}')
        # LEVEL 2+ IS DROPPED (the user, 2026-10-24): it added about 30 % more DS of
        # clearly lower accuracy at a high cost in time and memory
        if int(level) >= 2:
            print(f'WARNING: fit1d_network() level={level} is dropped; level 1 is used.')
            level = 1
        _kw = dict(threshold=threshold, window=window, cell=cell,
                   baseline=baseline, level=level,
                   max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
                   step_dv=step_dv, max_seasonal=max_seasonal,
                   consensus=consensus, err_dh=err_dh, err_dv=err_dv,
                   iterations=iterations, debug=debug)
        if union or len(self) == 1:
            return utils_fit1d._fit3d_union(self, union=union, **_kw)
        # union=False: EACH BURST ITS OWN SCENE, as trend2d() fits each burst
        # on its own pixels. The same driver runs once per burst -- there is
        # no second code path -- and the bursts share ONE chain, so the
        # cluster never holds more heavy tasks at once than a single scene
        # would. Where bursts overlap the two models disagree over the shared
        # ground, exactly as two per-burst trends do; union=True is the way
        # to ask for one answer there.
        chain = utils_fit1d._Fit3dChain()
        # ONCE FOR THE CALL, not once per burst: a missing ref_height warns
        # once, naming every burst. Each burst's value is its own either way.
        ep = Batch._elevation_phase_approximate(self)
        out = {}
        for k in self.keys():
            out.update(utils_fit1d._fit3d_union(type(self)({k: self[k]}),
                chain=chain, tag=f' [{k}]', ep=ep, **_kw).items())
        graph_check('fit1d_network', out, self)
        return Batch(out)

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
        return Batch(self.map_da(lambda da: utils_complex.intensity(da), **kwargs))

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

        graph_check('threshold', results, self)
        return BatchComplex(results)

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
        _graph_out = utils_complex.interferogram(self, self.isel(date=index))
        graph_check('singlelook', _graph_out, self)
        return _graph_out

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

    def pairs(self, pairs, baseline: 'str | None' = 'BPR', ref=None):
        """Select date pairs from per-date data, returning ref and rep stacks.

        Parameters
        ----------
        pairs : array-like (n_pairs, 2)
            Pairs as [[ref_date, rep_date], ...]. Dates as datetime64 or indices.
            [] pairs every date of each burst with its reference (the reference
            with itself included); a None inside a pair is the burst's reference date.
        baseline : str or None, default 'BPR'
            Per-date baseline whose smallest |value| picks each burst's reference
            date; None, or a burst lacking it: its first date.
        ref : None, int, str or datetime, default None
            The reference date of every burst, instead of the baseline pick.

        Returns
        -------
        tuple (ref, rep)
            Two BatchComplex with 'pair' dimension instead of 'date'.

        Examples
        --------
        ref, rep = bc.pairs([])   # every date with its burst's reference
        """
        import numpy as np
        from .Stack import _atmosphere_reference
        from . import utils_baseline

        keys = list(self.keys())
        # Map dates to integer indices (match by day to handle precision differences)
        date_days = np.array(self[keys[0]].coords['date'].values, dtype='datetime64[D]')
        date_to_idx = {d: i for i, d in enumerate(date_days)}

        def _index(d, row):
            return date_to_idx[np.datetime64(d, 'D')]

        # THE REFERENCE of each burst: _atmosphere_reference, as Stack.pairs() takes it
        refs = None if ref is None else _atmosphere_reference(self, baseline, ref, 'pairs')

        def reference(k):
            nonlocal refs
            if refs is None:
                refs = _atmosphere_reference(self, baseline, None, 'pairs')
            return 0 if refs is None else refs[k]

        # each burst its own index pairs: a None or [] names the burst's own reference
        idx = {k: utils_baseline.pair_indices(pairs, ds.sizes['date'], lambda k=k: reference(k), _index)
               for k, ds in self.items()}

        # Select, rename date→pair, and assign pair coords matching the caller.
        # The per-date 'residual' of Stack.align() is dropped, as Stack.pairs() does
        def select(col):
            return type(self)({k: ds.isel(date=idx[k][:, col]).rename(date='pair')
                               .drop_vars('residual', errors='ignore')
                               .assign_coords(pair=np.arange(len(idx[k])))
                               for k, ds in self.items()})

        ref_batch, rep_batch = select(0), select(1)
        graph_check('pairs', [ref_batch, rep_batch], self)
        return ref_batch, rep_batch

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


    @staticmethod
    def _unwrap_joined(x, kernel, token):
        """The date chunks of x joined per YX block: one task per YX block joins its date blocks (numpy inside) and
        unwraps the whole span of dates, one task per output block takes the dates of its own chunk. The output keeps
        the chunks of x; the dates are the unknowns, so the join is the function's design, not a rechunk."""
        import dask.array as da
        import numpy as np
        from dask.base import tokenize
        from dask.highlevelgraph import HighLevelGraph

        def join(blocks):
            return kernel(np.concatenate(blocks, axis=0))

        def take(stack, start, stop):
            return stack[start:stop]

        starts = np.cumsum((0,) + x.chunks[0]).tolist()
        # the kernel is in the key: two kernels on the same x and token are two computations (the token of the API
        # determines the kernel, the helper does not see it)
        join_name = 'unwrap3djoin-' + tokenize(x.name, token, kernel)
        out_name = 'unwrap3d-' + tokenize(x.name, token, kernel)
        dsk = {}
        for by in range(x.numblocks[1]):
            for bx in range(x.numblocks[2]):
                jk = (join_name, by, bx)
                dsk[jk] = (join, [(x.name, j, by, bx) for j in range(x.numblocks[0])])
                for j in range(x.numblocks[0]):
                    dsk[(out_name, j, by, bx)] = (take, jk, starts[j], starts[j + 1])
        graph = HighLevelGraph.from_collections(out_name, dsk, dependencies=[x])
        return da.Array(graph, out_name, chunks=x.chunks, dtype=np.float32,
                        meta=np.empty((0, 0, 0), dtype=np.float32))

    def unwrap3d(self, duration: float = 90.0, max_iter: int = 40,
                 search: int = 2, short_days: float = 40.0,
                 n_short: int = 6, min_disagree: int = 2,
                 min_dates: int = 3, n_trend: int = 3) -> 'Batch':
        """
        Unwrapped per-date phase DIRECTLY from the complex scenes.

        Replaces pairs() -> interferogram() -> unwrap2d() -> lstsq(). Single-look
        pair phase is exactly phi_ref - phi_rep, so the pair ambiguities are not
        free: k_ij = n_i - n_j. The network has one unknown integer per DATE
        instead of one per pair, every triplet closes by construction, and the
        dates are the unknowns, so no network inversion follows.

        Requires closure-exact input -- single-look scenes carrying per-date
        corrections only.
        Any per-PAIR correction upstream breaks closure and makes the per-date
        form invalid -- anything fitted or filtered on the interferograms rather
        than on the dates, goldstein() for instance, since what it removes from
        pair (i,j) is not the difference of anything it removed from i and j.

        Dates that cannot be reconciled with their neighbours are dropped to
        NaN and bridged with the longer interval to the next good date, so one
        bad acquisition does not corrupt everything after it.

        Rates outside the range the sampling can represent, +-lambda/(4 dt_min)
        taken over each pixel's surviving dates, come back as NaN rather than
        as a folded value: wrapped phase pins the rate only modulo 2*pi/dt_min
        and the siblings fit the observations exactly, so a solution is only
        knowable when its siblings are outside what the acquisition plan can
        express. No user prior is involved -- dt_min decides it.

        Parameters
        ----------
        duration : float
            Temporal baseline of the date pairs used as smoothness
            constraints. Default 90.
        max_iter : int
            Coordinate-descent sweeps over the integers. Default 40.
        search : int
            Integer offsets tried per date per sweep, +-search. Default 2.
        short_days, n_short, min_disagree :
            The recoverability test. Each of the up-to-n_short partners within
            short_days predicts the date's integer from its wrapped
            difference, which is right whenever the motion over that interval
            stays inside a half cycle; the date is dropped when at least
            min_disagree of them dissent from the majority. This assumes
            nothing about the shape of the trajectory, unlike a smoothness or
            curvature test, so an accelerating site is not read as defective.
            Widening short_days admits intervals where real motion can exceed
            a half cycle and drops more.
        min_dates : int
            Floor on surviving dates per pixel (default 3). Closure is
            structural here, so a resolved triplet is usable; there is no
            sample-size argument for a larger floor. Dropping stops at this
            floor, and since at most one date goes per pass it also bounds the
            work -- there is no separate max_drop to contradict it.
        n_trend : int
            Trend refinement passes (default 3). The rate is seeded from the
            shortest available intervals -- long gaps are excluded because a
            wrapped increment over dt only resolves rates below
            lambda/(4 dt), so a 120-day winter gap aliases anything above
            42 mm/yr while a 12-day one holds to 422 -- then re-fit over the
            whole unwrapped span and re-solved. Without this the smoothness
            objective drags fast pixels toward zero: a synthetic -150 mm/yr
            site was recovered as +19.

        Returns
        -------
        Batch
            Unwrapped per-date phase in radians; use displacement_los().
            The date chunks of one YX block are joined inside its task; the
            output keeps the chunks of the input.

        Examples
        --------
        >>> disp = (stack.chunk1d('0.5GB').fit1d_network()
        ...              .unwrap3d()
        ...              .displacement_los(stack.transform()))
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from .Batch import Batch
        from . import utils_unwrap3d

        BatchCore._require_lazy(self, 'unwrap3d')

        result = {}
        for key in self.keys():
            ds = self[key]
            # a Dataset's variables, or a DataArray as its own one (N81)
            vars_ = BatchCore._vars_of(ds)
            for a in vars_.values():
                if 'pair' in a.dims:
                    raise TypeError(
                        'unwrap3d() operates on the per-DATE stack, not on '
                        'per-pair data. It replaces the per-pair unwrap + lstsq route.')
            pols = [v for v, a in vars_.items()
                    if a.dtype.kind == 'c' and 'date' in a.dims
                    and 'y' in a.dims and 'x' in a.dims]
            if not pols:
                raise TypeError(
                    f'unwrap3d() found no complex (date, y, x) variables in '
                    f'burst {key}')
            date_values = np.asarray(ds.coords['date'].values)
            out_vars = {}
            for pol in pols:
                da_xr = vars_[pol]
                if da_xr.dims[0] != 'date':
                    da_xr = da_xr.transpose('date', ...)
                params = (float(duration), int(max_iter), int(search), float(short_days),
                          int(n_short), int(min_disagree), int(min_dates), int(n_trend))

                def _make(_d, _bd, _mi, _se, _sd, _ns, _mdis, _mdt, _ntr):
                    def kernel(block):
                        return utils_unwrap3d.unwrap3d_dates_array(
                            block, _d, duration=_bd, max_iter=_mi,
                            search=_se, short_days=_sd, n_short=_ns,
                            min_disagree=_mdis, min_dates=_mdt,
                            n_trend=_ntr)
                    return kernel

                kernel = _make(date_values, *params)
                if len(da_xr.data.chunks[0]) == 1:
                    out = da.map_blocks(
                        kernel, da_xr.data, dtype=np.float32,
                        meta=np.empty((0, 0, 0), dtype=np.float32))
                else:
                    out = BatchComplex._unwrap_joined(da_xr.data, kernel, (date_values,) + params)
                out_vars[pol] = xr.DataArray(out, dims=da_xr.dims,
                                             coords=da_xr.coords, name=pol)
            if isinstance(ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81): its coordinates are its own
                result[key] = BatchCore._form(ds, out_vars)
                continue
            new_ds = xr.Dataset(out_vars, attrs=ds.attrs)
            if 'spatial_ref' in ds.coords:
                new_ds = new_ds.assign_coords(spatial_ref=ds.spatial_ref)
            result[key] = new_ds
        graph_check('unwrap3d', result, self)
        return Batch(result)

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
            So does a block of the input smaller than the patch: the output
            keeps the input's chunks, and the halo would rechunk them. A
            correlation in other (y, x) chunks is rechunked to the phase's, with one
            WARNING; other date chunks, or polarisations chunked differently, raise
            one line at the call; a (y, x) correlation of all dates is broadcast over the dates.
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
        from .utils_dask import overlap_check

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
        from . import utils_dask
        moved = []   # the correlations rechunked in this call
        for k in self.keys():
            ds = self[k]
            corr_ds = corr[k]
            filtered_vars = {}
            window = windows[k]

            # the PRIMARY inputs: the complex variables of the burst, chunked alike (every polarisation; one line if not)
            utils_dask.check_primary_chunks({v: a.data for v, a in BatchCore._vars_of(ds).items()
                                             if a.dtype.kind == 'c' and isinstance(a.data, da.Array)}, 'goldstein')
            # Process each complex data variable in the dataset; a DataArray burst is
            # its own one (N81)
            for var_name, var_data in BatchCore._vars_of(ds).items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    corr_da = BatchCore._weight_of(corr_ds, var_name)
                    phase_dask = var_data.data
                    corr_dask = corr_da.data

                    # THE WEIGHT is a SECONDARY input (GMT U2): its (y, x) chunks go to the phase's, one WARNING per call;
                    # other date chunks raise one line. A (y, x) weight of all dates is broadcast over the dates lazily.
                    corr_dask, changed = utils_dask.align_secondary(phase_dask, corr_dask, 'goldstein', 'correlation')
                    if changed:
                        moved.append('correlation')
                    if phase_dask.ndim == 3 and corr_dask.ndim == 2:
                        corr_dask = da.broadcast_to(corr_dask, phase_dask.shape, chunks=phase_dask.chunks)

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
                    # the output keeps the input's chunks: dask rechunks the input when the halo is larger than a block,
                    # and the shared check raises one line then
                    filtered_dask = overlap_check(filtered_dask, phase_dask, 'goldstein',
                                                  (depth_y, depth_x), (depth_y, depth_x))

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

        utils_dask.warn_secondary('goldstein', moved)
        graph_check('goldstein', result, self)
        return type(self)(result)
