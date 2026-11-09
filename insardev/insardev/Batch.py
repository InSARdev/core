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
from .BatchCore import BatchCore, _SCALARS, _RULE_OPERANDS
import numpy as np
import xarray as xr
from . import utils_xarray
from .utils_geometry import _horn_gradient_2d_for_dask, _ref_mode
from typing import TYPE_CHECKING
from .utils_dask import graph_check, warn_floor
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect


def _horn_gradient_block(block, dy=1.0, dx=1.0, block_info=None):
    """The Horn gradient of a DEM block of a map_overlap with boundary 'none' (no boundary=nan layers, which cost dask about
    77 tasks per chunk). A side with a neighbour arrives with its one-pixel halo; the raster's own edge arrives without one
    and is padded with NaN here, the values the NaN boundary padded. The block leaves with the shape it arrived with, so
    dask's trim takes the halo back off the interior sides only. The dates of a (n, y, x) block are looped in numpy, one
    _horn_gradient_2d_for_dask per date. Module level so dask ships it by name."""
    import numpy as np
    if block_info is None:  # dask's meta inference calls the kernel without block_info: one block, both edges
        loc, nch = (0, 0), (1, 1)
    else:
        loc, nch = block_info[0]['chunk-location'], block_info[0]['num-chunks']
    top, bottom = loc[-2] > 0, loc[-2] < nch[-2] - 1
    left, right = loc[-1] > 0, loc[-1] < nch[-1] - 1
    z = np.asarray(block, dtype=np.float32)
    pad = [(0, 0)] * (z.ndim - 2) + [(0 if top else 1, 0 if bottom else 1), (0 if left else 1, 0 if right else 1)]
    z = np.pad(z, pad, mode='constant', constant_values=np.nan)
    if z.ndim == 2:
        g = _horn_gradient_2d_for_dask(z, dy, dx)
    else:
        g = np.stack([_horn_gradient_2d_for_dask(z[i], dy, dx) for i in range(z.shape[0])])
    return g[..., slice(None if top else 1, None if bottom else -1), slice(None if left else 1, None if right else -1)]


def _slope_block(grad):
    """Slope (degrees, float32) of a block of the complex Horn gradient."""
    import numpy as np
    return np.rad2deg(np.arctan(np.abs(grad))).astype('float32')


def _aspect_block(grad):
    """Aspect (degrees, float32) of a block of the complex Horn gradient: the DOWNSLOPE direction clockwise from north, NaN
    where the ground is level. The gradient is negated because aspect names where the surface falls."""
    import numpy as np
    mag = np.abs(grad)
    aspect = (np.rad2deg(np.arctan2(-grad.real, -grad.imag)) % 360.0)
    return np.where(mag > 0, aspect, np.nan).astype('float32')


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


def _crs_wkt(ds) -> str:
    """The CRS of a burst Dataset or DataArray as WKT."""
    return ds.rio.crs.to_wkt()


def _ele_range(ds) -> tuple | None:
    """The (lo, hi) of the `ele` actual_range ATTRIBUTE (never the raster), None without a valid one.

    >>> _ele_range(ds)"""
    if not isinstance(ds, xr.Dataset) or 'ele' not in ds.variables:
        return None
    rng = ds['ele'].attrs.get('actual_range')
    if rng is None or len(rng) != 2 or not np.all(np.isfinite(np.asarray(rng, dtype=np.float64))):
        return None
    return float(rng[0]), float(rng[1])


def _ele_midpoint(ds, key, where) -> float | None:
    """The midpoint of the `ele` actual_range ATTRIBUTE (never the raster), None without it."""
    rng = _ele_range(ds)
    return None if rng is None else 0.5 * (rng[0] + rng[1])


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


def _geometry_grid(block, geo, y, x, chunks, field=None, **kw):
    """A lazy (y, x) float32 grid of a geometry block function on the coordinates y, x in the chunks (cy, cx).

    The blocks get their GLOBAL pixel indices (arange of the whole grid, chunked like y and x) and the whole axes:
    the geometry nodes are the global ones, so the result does not depend on the chunks. field: an optional (y, x)
    array on the grid (the transform's ele), passed to the block with the chunks of the grid; kw: constant keyword
    arguments of the block."""
    import dask.array as da
    gy = da.from_array(np.arange(len(y)), chunks=(chunks[0],))
    gx = da.from_array(np.arange(len(x)), chunks=(chunks[1],))
    axes = dict(geo=geo, ycoord=np.asarray(y, np.float64), xcoord=np.asarray(x, np.float64), **kw)
    if field is None:
        return da.blockwise(block, 'yx', gy, 'y', gx, 'x', dtype=np.float32, **axes)
    if not isinstance(field, da.Array):
        field = da.from_array(np.asarray(field), chunks=(chunks[0], chunks[1]))
    field = field.rechunk((chunks[0], chunks[1]))
    return da.blockwise(block, 'yx', gy, 'y', gx, 'x', field, 'yx', dtype=np.float32, **axes)


def _grid_chunks(ds):
    """The (y, x) chunks of a burst's first lazy grid, one chunk without any."""
    for v in BatchCore._vars_of(ds).values():
        if BatchCore._is_grid(v) and v.chunks is not None:
            d = dict(zip(v.dims, v.chunks))
            return d['y'], d['x']
    return (ds.sizes['y'],), (ds.sizes['x'],)


def _elevation_grid(data, geos, pairs, joint, floors=None):
    """elevation() of one phase grid, (pair, y, x) or one pair's (y, x): lazy, the phase and its y, x only."""
    import dask.array as da
    from . import utils_geometry as ug
    arr = data.data[None] if data.dims == ('y', 'x') else data.data
    if not isinstance(arr, da.Array):
        # a computed (numpy) phase: one pair per chunk, and (y, x) in the library chunk budget, not the full grid
        from .utils_dask import rechunk2d
        c = rechunk2d(arr.shape[-2:], arr.dtype.itemsize, floors=floors)
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
        from .BatchUnit import BatchUnit
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
        from . import utils_rgb
        return utils_rgb.rgb_composite(copol, xpol, gamma=gamma, brightness=brightness, quantile=quantile)

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
        from .Batch import Batch
        from .BatchComplex import BatchComplex
        from .utils_dask import align_secondary, warn_secondary

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
        moved = []
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

                # Build variable arrays for the fit
                phase_spatial_chunks = phase_dask.chunks[-2:]
                if trans_ds is not None:
                    var_dask_list = []
                    for v in var_names:
                        # a covariate is a SECONDARY input (GMT U3): its (y, x) chunks go to the phase's, one WARNING per call
                        var_dask, changed = align_secondary(phase_dask, trans_ds[v].data, 'trend2d', f"covariate '{v}'",
                                                            check_dates=False)
                        if changed:
                            moved.append(f"covariate '{v}'")
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

                weight_dask = None
                if weight_da is not None:
                    weight_dask = weight_da.data
                    if squeeze_pair:
                        weight_dask = weight_dask[np.newaxis, ...]
                    # a SECONDARY weight (GMT U3): other date chunks raise one line, other (y, x) chunks go to the phase's
                    weight_dask, changed = align_secondary(phase_dask, weight_dask, 'trend2d', 'weight')
                    if changed:
                        moved.append('weight')

                result_dask = utils_detrend.trend2d_poly_dask(
                    phase_dask, weight_dask, var_dask_list, degree, is_complex,
                    detrend, extrapolate, debug=debug, label=f'{key}/{pol}')
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

        warn_secondary('trend2d', moved)
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

            The model keeps the data's (y, x) chunks: one task per block holds all its pairs; chunk2d() sizes the blocks.

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
        import dask.array as da
        from . import utils_fit1d
        from .utils_dask import align_secondary, warn_secondary

        # the radar geometry is the Dataset's (N81): a DataArray batch raises here;
        # baseline=None reads radar_wavelength alone (the pair baselines are coordinates)
        BatchCore._needs_dataset(self, 'fit1d', BatchCore._GEOMETRY if baseline else ('radar_wavelength',))
        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, self)
        BatchCore._require_lazy(self, 'fit1d')
        model_result = {}
        moved = []
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
            # the median acquisition origin (utils_fit1d.pair_time_origin): the master
            # is not recoverable from differences
            t0_day, t_ref, t_rep = utils_fit1d.pair_time_origin(rd, pd_)

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
            if bp is not None and bp.shape != rd.shape:
                raise ValueError(
                    f"fit1d(): {baseline!r} has {bp.shape[0]} value(s) for '{key}' "
                    f"but the batch has {rd.shape[0]} pair(s). Per-PAIR baselines "
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
                e2p = utils_fit1d.ele2phase(bp, lam, _fac)
            if e2p is None:
                print(f"fit1d(): baseline=None for '{key}' -- the height column "
                      "is dropped and the topographic phase stays in the "
                      "residual.", flush=True)

            # THE PAIR CONVENTION, which is this method's own -- pairs are not
            # a per-date stack and this fit is not BatchComplex.fit1d.
            #
            #   phi_p = +velocity*dt - height*ele2phase_p + annual
            #
            # `velocity` is the rate of LOS displacement expressed in radians
            # (utils_fit1d.fit1d_pair_design): the sense lstsq() works in, and the
            # sense displacement_los() converts. `height` keeps the other sign: it
            # is DEM error, not a displacement, and it is RETURNED IN RADIANS like
            # everything else -- radians per unit ele2phase -- and the metres only
            # ever appear in the diagnostic printed below.
            has_h = e2p is not None
            fit_seasonal = bool(max_seasonal and max_seasonal > 0)
            A = utils_fit1d.fit1d_pair_design(e2p, t_ref, t_rep, fit_seasonal)
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
            s_v, s_h, vif = utils_fit1d.fit1d_precision(A, has_h, fit_seasonal, meter2rad)
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

            # the data's chunks as given: no tiles of the library's own (the user's chunking decides)
            d_in = da_.data.chunks
            d_dask = da_.data
            w_dask = None
            if weight is not None:
                # a SECONDARY weight: other pair (date) chunks raise one line; its other (y, x) chunks go to the data's, one WARNING per call
                w_in = BatchCore._weight_of(weight[key], pol).transpose('pair', ...)
                if w_in.data.chunks[0] != d_in[0]:
                    from .utils_dask import _sizes
                    raise ValueError(f"ERROR: fit1d(): the date chunks of the weight {_sizes(w_in.data.chunks[0])} differ "
                                     f"from the data's {_sizes(d_in[0])}: a secondary input takes the data's date chunks")
                w_dask, changed = align_secondary(d_dask, w_in.data, 'fit1d', 'weight')
                if changed:
                    moved.append('weight')

            # the per-block kernel is utils_fit1d.fit1d_pairs_planes; the defaults
            # bind this key's values now, the graph runs later
            def _blk(b, wb=None, _A=A, _hh=has_h, _se=fit_seasonal, _m2r=meter2rad,
                     _mh=max_dh, _ms=max_seasonal):
                return utils_fit1d.fit1d_pairs_planes(b, _A, wb, _hh, _se, _mh, _ms, _m2r)

            # ONE TASK PER (y, x) BLOCK OF THE DATA, its pairs contracted inside (as BatchComplex.fit1d): the model keeps the data's (y, x) chunks
            args = (d_dask, 'pyx') if w_dask is None else (d_dask, 'pyx', w_dask, 'pyx')
            res = da.blockwise(_blk, 'nyx', *args, new_axes={'n': 5}, concatenate=True, dtype=np.complex64,
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
        warn_secondary('fit1d', moved)
        graph_check('fit1d', model_result, self)
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
        from . import utils_fit1d

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
                t_ref = utils_fit1d.model_time(rd, t0)
                t_rep = utils_fit1d.model_time(pd_, t0)
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
                t_d = utils_fit1d.model_time(dday, t0)

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
                e2p = utils_fit1d.elevation_to_phase(bp, lam, _fac)

            vel = mds['velocity'].data
            hgt = mds['height'].data
            # OPTIONAL: fit1d_network() fits an annual term and fit1d() does not, and
            # a model without the variable HAS none rather than an unknown one.
            sea = mds['seasonal'].data if 'seasonal' in mds else None
            if per_pair:
                pred = utils_fit1d.predict_pairs(vel, hgt, sea, t_ref, t_rep, e2p)
            else:
                pred = utils_fit1d.predict_dates(vel, hgt, sea, t_d, e2p)
            if ref is not None:
                if per_pair:
                    raise TypeError(
                        "ref re-references a per-DATE series, and a pair is "
                        "already a difference in which the reference cancels. "
                        "Call it on the per-date stack instead.")
                _i = Batch._ref_index(ref, da_.coords['date'].values)
                pred = utils_fit1d.reference_plane(pred, _i)

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
        graph_check('predict', out, self)
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
        from . import utils_fit1d
        full = self.predict(model, baseline=baseline)
        disp = (full if baseline is None
                else self.predict(model, baseline=None))
        out = {}
        for key in disp.keys():
            vars_ = BatchCore._vars_of(disp[key])
            pol = next(iter(vars_))
            mixed = utils_fit1d.mix_pairs(BatchCore._vars_of(self[key])[pol].data,
                                          BatchCore._vars_of(full[key])[pol].data,
                                          vars_[pol].data)
            res = BatchCore._form(disp[key], {pol: vars_[pol].copy(data=mixed)},
                                  attrs=disp[key].attrs)
            # the coordinates the operands carry and the disp does not (BPR, the baselines and
            # the alphas of self), as the operator rule merges them
            for src in (self[key], full[key]):
                for c in src.coords:
                    if c not in res.coords:
                        res = res.assign_coords({c: src.coords[c]})
            out[key] = res
        graph_check('mix', out, self)
        return Batch(out)


    def elevation_phase(self) -> "Batch":
        """Radians of phase per metre of elevation per metre of perpendicular baseline.

        `4 pi kN / (lambda R sin(incidence))` per pixel on the WGS84 ellipsoid, at the
        terrain height (the midpoint of the `ele` actual_range attribute): the sensitivity
        elevation() divides by in its one-division form, where the per-pixel
        perpendicular baseline of a pair comes from the baseline triples. Multiply by
        B_perp and a height to get phase; divide a phase by it and B_perp to get a height.

        SLANT RANGE AND THE SINE OF THE INCIDENCE ANGLE, matching GMTSAR (`sbas.c:186`,
        `phase2topo.c`); kN is the ellipsoid normal's factor, 1 on a sphere.

        READS NO RASTER: x, y and the reference date's numbers only.
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
        from . import utils_geometry as ug
        return {key: ug.elevation_phase_factor(wavelength, slant_range, sin_inc)
                for key, (wavelength, slant_range, sin_inc)
                in Batch._centre_geometry_approximate(self, 'scene-centre elevation_phase()').items()}

    def _incidence_approximate(self) -> dict:
        """The incidence angle (radians) at the centre of the PROCESSED area, `{burst_id: value}`: the
        counterpart of `_elevation_phase_approximate()`, from the same metadata-only centre and the same
        sphere (arcsin of its sin(incidence)), equally approximate and reading NO radar grids. The
        displacement conversions of a burst without (y, x) -- a point's series, an area mean -- use it,
        one value per burst; gridded data take the per-pixel incidence()."""
        from . import utils_geometry as ug
        return {key: ug.incidence_from_sine(sin_inc)
                for key, (_, _, sin_inc) in Batch._centre_geometry_approximate(self, 'scene-centre incidence()').items()}

    def _centre_geometry_approximate(self, where) -> dict:
        """`{burst_id: (wavelength, slant range, sin(incidence))}` at the centre of the PROCESSED area, on the
        sphere of earth_radius at ref_height: the shared geometry of `_elevation_phase_approximate()` (see there
        for where the centre comes from) and `_incidence_approximate()`. Metadata only, NO radar grids."""
        import numpy as np
        from shapely import wkt as _wkt
        from pyproj import Transformer as _Tr
        from . import utils_geometry as ug

        out: dict = {}
        ref_heights = _ref_heights(self, where)
        for key, ds in self.items():
            g = lambda n: float(np.asarray(ds[n].values).ravel().mean())
            wavelength = g('radar_wavelength')

            # The stack's burst `geometry` and CRS are the stored numbers of every real product: there is no
            # fallback to a burst centre (N138: none of 66 real calls on the pygmtsar and ecef stores reached one)
            poly = _wkt.loads(str(np.asarray(ds['geometry'].values).ravel()[0]))
            lon, lat = (np.asarray(v) for v in poly.exterior.coords.xy)
            px, py = _Tr.from_crs('EPSG:4326', ds.rio.crs, always_xy=True).transform(lon[:4], lat[:4])
            xc = 0.5 * (float(ds.x.min()) + float(ds.x.max()))
            yc = 0.5 * (float(ds.y.min()) + float(ds.y.max()))
            slant_range, sin_inc = ug.centre_geometry(
                px, py, xc, yc, g('num_lines'), g('num_rng_bins'), g('near_range'), g('rng_samp_rate'),
                0.5 * (g('SC_height_start') + g('SC_height_end')), ref_heights[key], g('earth_radius'))
            out[key] = (wavelength, slant_range, sin_inc)
        return out

    def incidence(self) -> "Batch":
        """Incidence angle (radians) on the WGS84 ellipsoid at each pixel's OWN height, the transform's `ele`.

        The satellite track is rebuilt from the REFERENCE date's numbers (near_range, rng_samp_rate, earth_radius,
        SC_height_start/end, num_lines, num_rng_bins, lookdir) and the exact crop polygon `geometry`, and evaluated on the
        absolute lattice nodes every 225 m at three heights, the ends and the midpoint of the `ele` actual_range attribute:
        bilinear between the nodes, a parabola in each pixel's own ele (read lazily, chunk by chunk). Measured against the
        same track at each pixel's own height without nodes: at most 0.000008 deg on 13 bursts with up to 2409 m of relief
        (the nodes' own term); against the precise orbit: 0.0005 deg (Gastein). A NaN ele gives NaN (no data there).
        Without the actual_range the height is 0 (one WARNING per call).

        Call it on the stack (`stack.incidence()`) or its transform() view.
        """
        from . import utils_geometry as ug
        BatchCore._needs_dataset(self, 'incidence', ug.SCALARS, 'Call it on the stack: stack.incidence().')
        crs = self.crs
        out: dict[str, xr.Dataset] = {}
        no_range: list = []
        for key, ds in self.items():
            geo = _stack_geometry(ds, key, 'incidence()', 'Call it on the stack: stack.incidence().', no_range)[0]
            chunks = _grid_chunks(ds)
            # each pixel's own height: the transform's ele and the ends of its actual_range (none: the constant height h_attr)
            kw = dict(field=ds['ele'].data, heights=_ele_range(ds) or (geo['h_attr'],) * 2) \
                if 'ele' in ds.variables else {}
            inc = _geometry_grid(ug.incidence_block, geo, ds.y.values, ds.x.values, chunks, **kw)
            result_ds = xr.Dataset({'incidence': xr.DataArray(inc, dims=('y', 'x'), coords={'y': ds.y, 'x': ds.x})})
            result_ds.attrs = ds.attrs
            result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        _warn_no_actual_range('incidence()', no_range)
        return Batch(out)

    def look(self) -> "Batch":
        """Look azimuth (degrees, clockwise from geodetic north, satellite to target, in [0, 360)) per pixel of the grid, at each
        pixel's OWN height, the transform's `ele`, as incidence().

        The satellite position on each pixel's zero-Doppler line comes from the orbit numbers of the REFERENCE date and the exact
        crop polygon, evaluated on the absolute lattice nodes every 225 m at three heights, the ends and the midpoint of the `ele`
        actual_range attribute: bilinear in its sine and cosine between the nodes, a parabola in each pixel's own ele (read
        lazily, chunk by chunk; no wrap at 0 and 360). Measured against the exact look at each pixel's own height: at most
        0.000017 deg (the midpoint it replaces: up to 0.00026 deg). A NaN ele gives NaN. Without the actual_range the height is 0
        (one WARNING per call). It is the orbit geometry, not the polygon heading. A decimated transform() gives the look on its
        own decimated grid.

        Call it on the stack (`stack.look()`) or its transform() view.
        """
        from . import utils_geometry as ug
        BatchCore._needs_dataset(self, 'look', ug.SCALARS, 'Call it on the stack: stack.look().')
        crs = self.crs
        out: dict[str, xr.Dataset] = {}
        no_range: list = []
        for key, ds in self.items():
            geo = _stack_geometry(ds, key, 'look()', 'Call it on the stack: stack.look().', no_range)[0]
            chunks = _grid_chunks(ds)
            # each pixel's own height: the transform's ele and the ends of its actual_range (none: the constant height h_attr)
            kw = dict(field=ds['ele'].data, heights=_ele_range(ds) or (geo['h_attr'],) * 2) \
                if 'ele' in ds.variables else {}
            look = _geometry_grid(ug.look_block, geo, ds.y.values, ds.x.values, chunks, **kw)
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
        from .utils_dask import overlap_check

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
            # that is the halo. No boundary=nan: the raster's own border has no
            # neighbour and the kernel pads it with NaN (_horn_gradient_block).
            # Float is taken in the kernel, as the values were cast before.
            dsk = z.data
            if not isinstance(dsk, da.Array):
                dsk = da.from_array(dsk, chunks=dsk.shape)
            depth = ({0: 0, 1: 1, 2: 1} if z.ndim == 3 else {0: 1, 1: 1})
            grad = da.map_overlap(
                _horn_gradient_block, dsk,
                depth=depth, boundary='none', trim=True, dtype=np.complex64,
                dy=dy, dx=dx)
            # the chunks as they are: a halo of 1 px fits every block, and a larger one raises (utils_dask)
            grad = overlap_check(grad, dsk, 'aspect', (3, 3), (1, 1))

            # slope and aspect of the gradient, one task per chunk each
            slope = grad.map_blocks(_slope_block, dtype=np.float32)
            aspect = grad.map_blocks(_aspect_block, dtype=np.float32)

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
        graph_check('aspect', out, self)
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
        from .BatchComplex import BatchComplex
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

        from . import utils_geometry as ug

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
                disp_vars[var_name] = ug.phase_to_los(data, wavelength)
            # a DataArray in, a DataArray out (N81)
            out[key] = BatchCore._form(phase_ds, disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, transform: 'Batch | Stack', func, suffix: str = '') -> 'Batch':
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin).

        The incidence is computed ON THE PHASE'S OWN GRID (its y, x and chunks) from the geometry source -- the
        stack or its transform() view -- with no raster read and nothing reindexed (the constant-height incidence_block).

        A DATAARRAY WITHOUT (y, x) -- a point's series, an area mean -- is converted PER BURST by the incidence at
        the centre of that source's processed area (_incidence_approximate(), the approximate burst-centre path of
        the scalar height per fringe): one value per burst, the per-burst sensitivity. A Dataset without any grid
        raises in displacement_los()."""
        import xarray as xr
        from . import utils_geometry as ug
        from .Stack import Stack
        # a Stack is its transform() here (GKB U5): the grids are read through the Batch API, never as a dict
        if isinstance(transform, Stack):
            transform = transform.transform()

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
                comp = ug.los_component(los_ds, centre[key], func)
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

                comp = ug.los_component(data, incidence, func)

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
        floors: list = []

        for key, phase_ds in self.items():
            grids = {n: d for n, d in BatchCore._vars_of(phase_ds).items() if 'y' in d.dims and 'x' in d.dims}
            if joint:
                if key not in stack:
                    raise KeyError(f'Missing geometry for key: {key}')
                geo, bh, bv, dates = _stack_geometry(stack[key], key, 'elevation()',
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
                elev_vars[var_name] = _elevation_grid(data, geos, pairs, joint, floors)

            # a DataArray in, a DataArray out (N81)
            out[key] = BatchCore._form(phase_ds, elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        warn_floor('elevation', floors)

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

    def _link_components(self, conncomp_labels=None, conncomp_size=100, conncomp_gap=None,
                         conncomp_linksize=5, conncomp_linkcount=30, debug=False):
        """
        Link disconnected components in unwrapped phase by finding optimal 2π offsets.

        Parameters
        ----------
        conncomp_labels : Batch or None
            Optional pre-computed connected component labels from IRLS.
            Labels should be size-ordered (1=largest, 2=second, etc.).
        conncomp_size : int
            Minimum component size to process.
        conncomp_gap : int or None
            Maximum pixel distance for connections.
        conncomp_linksize : int
            Number of pixels for offset estimation.
        conncomp_linkcount : int
            Maximum neighbor components to consider.
        debug : bool
            If True, print diagnostic information.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.
        """
        import xarray as xr
        import dask.array
        from functools import partial
        from . import utils_unwrap2d
        from .Stack import Stack

        def whole_pair(arr):
            """One block per pair, the whole y and x in one block (the leading axis of a 3D array is the pair)."""
            chunks = {arr.ndim - 2: -1, arr.ndim - 1: -1}
            if arr.ndim == 3:
                chunks[0] = 1
            return arr.rechunk(chunks)

        # Process each dataset in the batch
        result = {}
        for key in self.keys():
            ds = self[key]
            # Get corresponding conncomp dataset if available
            conncomp_ds = conncomp_labels[key] if conncomp_labels is not None and key in conncomp_labels else None
            # THE GRIDS ONLY, as unwrap2d_irls() takes them; the metadata rides along through _carry_meta().
            # Linked too, the metadata came back as float32 arrays: the strings ones that fail on compute,
            # most numbers NaN. A DataArray burst is its own one grid (N81)
            grids = BatchCore._grids_of(ds)
            conncomp_grids = BatchCore._vars_of(conncomp_ds) if conncomp_ds is not None else {}

            linked_vars = {}
            for var, data_arr in grids.items():
                # Use da.blockwise for efficient dask integration
                dask_data = dask.array.asarray(data_arr.data)
                dim_str = ''.join(chr(ord('a') + i) for i in range(dask_data.ndim))
                # THE WHOLE PAIR IN ONE BLOCK (N183): the components are sized, filtered and offset on one raster,
                # as unwrap2d_irls() solves them, so the linked phase does not depend on the block grid of the input.
                # The result goes back to the input block grid below (lazily); one task holds one whole pair.
                orig_chunks = dask_data.chunks
                whole_phase = whole_pair(dask_data)

                # Provide meta to avoid calling the linking during graph construction
                meta = np.empty((0,) * dask_data.ndim, dtype=np.float32)
                link = partial(utils_unwrap2d.link_components_block, conncomp_size=conncomp_size,
                               conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                               conncomp_linkcount=conncomp_linkcount, debug=debug)

                if var in conncomp_grids:
                    # Pass conncomp to the linking, on the same whole-pair blocks as the phase
                    conncomp_dask = whole_pair(dask.array.asarray(conncomp_grids[var].data))
                    result_dask = dask.array.blockwise(
                        link, dim_str,
                        whole_phase, dim_str,
                        conncomp_dask, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )
                else:
                    # No conncomp - will recompute inside the linking
                    result_dask = dask.array.blockwise(
                        link, dim_str,
                        whole_phase, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )
                # back to the input block grid
                result_dask = result_dask.rechunk(orig_chunks)

                linked_da = xr.DataArray(
                    result_dask,
                    dims=data_arr.dims,
                    coords=data_arr.coords
                )
                linked_vars[var] = linked_da

            if isinstance(ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81): no metadata to carry
                result[key] = BatchCore._form(ds, linked_vars)
                continue
            result[key] = xr.Dataset(Stack._carry_meta(ds, linked_vars),
                                     coords=ds.coords, attrs=ds.attrs)

        return Batch(result)

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
        # Validate parameters
        if conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )

        # the linking of this batch, with no labels given: it labels the components itself
        return self._link_components(
            conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
            conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
            debug=debug
        )
