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
from . import utils_io,  utils_xarray
import operator
from types import FunctionType
import numpy as np
import xarray as xr
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex, BatchVar
    from .Stack import Stack
    import rasterio as rio
    import pandas as pd
    import matplotlib

# the scalar operands of the Batch operators (a complex scalar included: ph * 1j)
_SCALARS = (int, float, complex, np.number)


def _parse_budget(budget):
    """Parse budget string like '128MiB', '256MB', '1GiB' to integer MB."""
    import re
    s = budget.strip().upper()
    m = re.match(r'^([\d.]+)\s*(MIB|MB|GIB|GB|KIB|KB)$', s)
    if not m:
        raise ValueError(f"Cannot parse budget '{budget}'. Use e.g. '128MiB', '256MB', '1GiB'.")
    val = float(m.group(1))
    unit = m.group(2)
    if unit in ('GIB', 'GB'):
        return int(val * 1024)
    elif unit in ('MIB', 'MB'):
        return int(val)
    elif unit in ('KIB', 'KB'):
        return max(1, int(val / 1024))
    return int(val)


def _nodata_of(dtype):
    """THE NODATA VALUE OF A MERGED GRID, by its dtype (to_dataset()): NaN for float and
    complex grids, -1 for signed integers (fit3d level and conncomp), 0 for unsigned
    integers (unwrap2d conncomp: 0 is no component) and False for bool. Not the dtype's
    maximum: 0 and -1 are the values a plot and a colormap handle easily."""
    dtype = np.dtype(dtype)
    if dtype.kind in 'fc':
        return np.nan
    if dtype.kind == 'i':
        return -1
    if dtype.kind == 'u':
        return 0
    if dtype.kind == 'b':
        return False
    raise TypeError(f'ERROR: to_dataset() cannot merge a {dtype} grid. Convert it to a number first.')


def _warn_int_nodata(who, dtypes):
    """ONE WARNING per call naming the integer and bool grids of {name: dtype}, whose
    nodata is set by the dtype (_nodata_of), for one burst or many (decided 2026-09-30:
    "show warning when such datatype appear - so user is aware")."""
    ints = [f'{name} {np.dtype(dt)} {_nodata_of(dt)}' for name, dt in dtypes.items() if np.dtype(dt).kind in 'biu']
    if ints:
        print(f"WARNING: {who}(): integer grids with nodata by dtype: {', '.join(ints)}. "
              f"Replace the nodata value or convert the dtype first if that is wrong.")


def _skip_nodata(value):
    """An integer or bool grid's nodata (_nodata_of: -1, 0, False) as NaN, the way the
    exports skip it as to_dataset() does (decided 2026-10-01): to_geojson() leaves it
    out, plot() and to_vtk() do not draw it. Float and complex grids as they are."""
    def skip(a):
        return a.where(a != _nodata_of(a.dtype)) if a.dtype.kind in 'biu' else a
    if isinstance(value, xr.DataArray):
        return skip(value)
    return value.assign({v: skip(value[v]) for v in value.data_vars if value[v].dtype.kind in 'biu'})


def _merge_tiles_for_dask(tiles, offsets, out_shape, fill_dtype, nodata=np.nan):
    """
    Module-level function for merging tiles in to_dataset().

    Defined at module level to avoid dask serialization issues with nested functions.
    When using dask distributed, closures with nested functions may not serialize correctly,
    causing random/incorrect behavior on workers.

    THE OVERLAP RULE: the tiles come in the bursts' acquisition order and each one is laid
    over the ones before it, so the MOST RECENT burst's valid value wins; its nodata
    (_nodata_of: NaN, or -1, 0, False for an integer or bool grid) is transparent and
    never hides an earlier burst's valid value; a pixel no tile covers stays nodata.

    Args:
        tiles: list of 3D arrays (1, ny, nx), the tiles of _tile_for_dask(), earliest burst first
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


def _tile_for_dask(block, s, y0, y1, x0, x1):
    """One tile of to_dataset(): rows [y0, y1) and columns [x0, x1) of one input block, as (1, ny, nx) --
    slice s of a 3-D block, or the 2-D block itself (s None)."""
    if s is None:
        return block[np.newaxis, y0:y1, x0:x1]
    return block[s:s + 1, y0:y1, x0:x1]


def _merge_parts_for_dask(offsets, out_shape, fill_dtype, nodata, *tiles):
    """One output block of to_dataset(): _merge_tiles_for_dask() of its tiles, as (1, ny, nx)."""
    return _merge_tiles_for_dask(tiles, offsets, out_shape, fill_dtype, nodata)[np.newaxis]


def _dissolve_pol_for_dask(da_current, das_others, wrap, extend, weight):
    """
    Module-level function for dissolving one polarization in dissolve().

    Defined at module level to avoid dask serialization issues with nested functions.
    When using dask distributed, closures with nested functions may not serialize correctly,
    causing random/incorrect behavior on workers.

    Args:
        da_current: xarray DataArray of current burst
        das_others: list of xarray DataArrays from overlapping bursts
        wrap: bool, True for wrapped phase (circular mean)
        extend: bool, True to fill NaN areas from overlapping bursts
        weight: float or None, weight of current burst

    Returns:
        numpy array with dissolved values
    """
    import warnings

    # Use exact coordinates from da_current - do NOT modify them
    ys = da_current.y.values
    xs = da_current.x.values
    n_others = len(das_others)

    if weight is None:
        w_current, w_other = 1.0, 1.0
    else:
        w_current = weight
        w_other = (1.0 - weight) / n_others if n_others > 0 else 0.0

    # Reindex das_others to match da_current coordinates
    # Grids are consistent with exactly matched coordinates in overlap areas
    das_reindexed = []
    for d in das_others:
        das_reindexed.append(d.reindex(y=ys, x=xs, fill_value=np.nan))

    current_vals = da_current.values
    current_valid = np.isfinite(current_vals)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        if wrap:
            weighted_sum = np.where(current_valid, np.exp(1j * current_vals).astype(np.complex64) * w_current, np.complex64(0))
            weight_sum = np.where(current_valid, w_current, 0.0)
            for d in das_reindexed:
                vals = d.values
                valid = np.isfinite(vals)
                weighted_sum += np.where(valid, np.exp(1j * vals).astype(np.complex64) * w_other, np.complex64(0))
                weight_sum += np.where(valid, w_other, 0.0)
            valid_weights = weight_sum > 0
            normalized = np.divide(weighted_sum, weight_sum, out=np.zeros_like(weighted_sum), where=valid_weights)
            out = np.where(valid_weights, np.arctan2(normalized.imag, normalized.real), np.nan)
        else:
            weighted_sum = np.where(current_valid, current_vals * w_current, 0.0)
            weight_sum = np.where(current_valid, w_current, 0.0)
            for d in das_reindexed:
                vals = d.values
                valid = np.isfinite(vals)
                weighted_sum += np.where(valid, vals * w_other, 0.0)
                weight_sum += np.where(valid, w_other, 0.0)
            out = np.divide(weighted_sum, weight_sum, out=np.full_like(weighted_sum, np.nan), where=weight_sum > 0)

        if not extend:
            out = np.where(current_valid, out, np.nan)

    return out.astype(da_current.dtype)




def _dissolve_raw_for_dask(current_arr, current_y, current_x,
                            others_arrs, others_ys, others_xs,
                            wrap, extend, weight):
    """
    Dissolve using raw numpy arrays + coordinates.

    Receives raw arrays (dask resolves them to numpy before calling) and
    numpy coordinate arrays. Reconstructs minimal xarray DataArrays for
    the interp-based coordinate matching, then delegates to _dissolve_pol_for_dask.

    For 3D arrays (pair, y, x), iterates over first dim.
    """
    import xarray as xr

    if current_arr.ndim > 2:
        n_stack = current_arr.shape[0]
        slices = []
        for i in range(n_stack):
            da_c = xr.DataArray(current_arr[i], dims=['y', 'x'],
                                coords={'y': current_y, 'x': current_x})
            das_o = [xr.DataArray(arr[i], dims=['y', 'x'],
                                  coords={'y': y, 'x': x})
                     for arr, y, x in zip(others_arrs, others_ys, others_xs)]
            slices.append(_dissolve_pol_for_dask(da_c, das_o, wrap, extend, weight))
        return np.stack(slices, axis=0)
    else:
        da_c = xr.DataArray(current_arr, dims=['y', 'x'],
                            coords={'y': current_y, 'x': current_x})
        das_o = [xr.DataArray(arr, dims=['y', 'x'],
                              coords={'y': y, 'x': x})
                 for arr, y, x in zip(others_arrs, others_ys, others_xs)]
        return _dissolve_pol_for_dask(da_c, das_o, wrap, extend, weight)




def _apply_gaussian_2d_for_dask(block, weight_block=None, sigmas=None, threshold=0.5,
                                 device='cpu', pixel_sizes=None, out_dtype=np.float32):
    """
    Module-level function for gaussian map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles both 2D (y, x) and 3D (1, y, x) blocks - gaussian_numpy() handles squeeze/unsqueeze.

    Parameters
    ----------
    block : np.ndarray
        Data block from dask, shape (y, x) or (1, y, x)
    weight_block : np.ndarray or None
        Weight block from dask, shape (y, x) or (1, y, x) or None
    sigmas : tuple
        Gaussian sigmas (sigma_y, sigma_x)
    threshold : float
        Threshold for weighted convolution
    device : str
        PyTorch device
    pixel_sizes : tuple
        Pixel sizes (dy, dx) in meters
    out_dtype : np.dtype
        Output dtype
    """
    from .utils_gaussian import gaussian_numpy
    # gaussian_numpy handles (1, y, x) -> squeeze -> process -> unsqueeze
    return gaussian_numpy(block, weight_block, sigma=sigmas, truncate=4.0, threshold=threshold,
                          device=device, pixel_sizes=pixel_sizes).astype(out_dtype)


@serialize_gpu
def _neighbors_kernel_2d_for_dask(data_chunk, window_y, window_x, half_y, half_x, device):
    """Count valid neighbors for 2D data.

    Defined at module level to avoid dask serialization issues with nested functions.
    Closures capturing variables can cause memory explosions in dask workers.

    Parameters
    ----------
    device : str
        Device string ('cpu', 'cuda', 'mps') - converted to torch.device internally.
    """
    import torch
    import numpy as np

    # Convert string to torch.device
    dev = torch.device(device)

    H, W = data_chunk.shape
    ny, nx = H - 2 * half_y, W - 2 * half_x
    n_total_neighbors = window_y * window_x - 1

    if ny <= 0 or nx <= 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    data_t = torch.from_numpy(data_chunk.astype(np.float32)).to(dev)
    valid = torch.isfinite(data_t).float()

    # Neighbor mask (exclude center)
    center_idx = (window_y // 2) * window_x + (window_x // 2)
    neighbor_mask = torch.ones(window_y * window_x, dtype=torch.bool, device=dev)
    neighbor_mask[center_idx] = False

    # Unfold to get windows
    window_valid = valid.unfold(0, window_y, 1).unfold(1, window_x, 1)
    neighbors_valid = window_valid.reshape(ny, nx, -1)[:, :, neighbor_mask]
    count = neighbors_valid.sum(dim=-1)

    # Set zero neighbors to NaN (isolated/invalid pixels)
    count = torch.where(count > 0, count, torch.full_like(count, float('nan')))

    # Pad back to full size
    result = torch.full((H, W), float('nan'), device=dev)
    result[half_y:H - half_y, half_x:W - half_x] = count

    output = result.cpu().numpy()

    del data_t, valid, window_valid, neighbors_valid, result
    if dev.type == 'mps':
        torch.mps.empty_cache()
    elif dev.type == 'cuda':
        torch.cuda.empty_cache()

    return output


class BatchCore(dict):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores real values (correlation and unwrapped phase).
    
    Examples:
    intfs60_detrend = Batch(intfs60) - Batch(intfs60_trend)

    dss = intfs60_detrend.sel(['106_226487_IW2','106_226488_IW2','106_226489_IW2','106_226490_IW2','106_226491_IW2'])
    dss_fixed = dss + {'106_226490_IW2': 2.6, '106_226491_IW2': 3})

    intfs60_detrend.isel(1)
    intfs60_detrend.isel([0, 2])
    intfs60_detrend.isel(slice(1, None))

    THE OPERATOR RULE (N81). A burst Dataset is a container of GRIDS, the
    variables carrying both y and x. Every operator -- arithmetic with a
    scalar, a DataArray or a Dataset in either order, comparisons, unary
    operators, numpy ufuncs, reductions, where() and the map helpers -- acts on
    the grids only, and every other variable (numeric or string, any dims)
    passes through unchanged. A Dataset without grids raises and names the
    variable to select. A Batch of DataArrays takes every operator directly;
    x.name and x['name'] both give it, as in xarray. A (y, x) grid keeps the
    batch's class (w['VV'] is a BatchWrap); any other variable or coordinate
    (BPR, near_range, residual, ref, y, ...) is a plain Batch. Class
    conversion (BatchWrap wraps) happens on explicit construction only,
    never on a selection or a mask.

    THE CLASS HOLDS ITS VALUES: BatchWrap(x) wraps float phase and
    BatchUnit(x) holds float units, and both raise for bool, integer or
    complex grids; BatchComplex(x) takes any dtype. A result the class does
    not keep -- a comparison, ~, np.isfinite() of a BatchWrap or a BatchUnit,
    any boolean result of a BatchComplex -- is a plain Batch (_as_class).

    A MASK MATCHES BY NAME (where(), mask()): a Dataset mask masks each grid
    by its grid of the same name, and a grid it lacks raises; a DataArray
    mask masks every grid. Non-grid variables pass through.

    A WEIGHT IS A BATCHUNIT: of Datasets, each grid weighted by its grid of
    the same name (a grid it lacks raises), or of one (y, x) DataArray that
    weights every grid. Anything else raises (_weight).

    A BATCH OF DATAARRAYS TAKES THE METHODS DIRECTLY: a DataArray in, a
    DataArray out (or the method's own named outputs). A method that needs
    the Dataset's non-grid variables (radar_wavelength, BPR, the radar
    geometry) raises for it and names them. Only the methods about named
    Dataset structure (plot, export, save, assign, merge, align) see it as
    one-variable Datasets (_DA_AS_DATASETS).
    """

    class CoordCollection:
        def __init__(self, ds):
            self._ds = ds
        def __getitem__(self, key):
            return self._ds.coords[key]
        def __contains__(self, key):
            return key in self._ds.coords
        def get(self, key, default=None):
            return self._ds.coords.get(key, default)
        def keys(self):
            return self._ds.coords.keys()
        def values(self):
            return self._ds.coords.values()
        def items(self):
            return self._ds.coords.items()

    @staticmethod
    def _get_torch_device(device='auto', debug=False):
        """Get PyTorch device. Delegates to utils_torch to avoid circular imports."""
        from .utils_torch import get_torch_device
        return get_torch_device(device, debug)

    @property
    def is_lazy(self) -> bool:
        """
        Check if batch data is lazy (dask arrays).

        Only GRIDDED variables count -- those carrying y and x. chunk1d() and
        chunk2d() rechunk exactly those and leave per-date metadata (BPR,
        startTime, fullBurstID, ...) alone, so requiring the metadata to be
        dask too would make the check unsatisfiable however the user chunks.
        Every operation that asks for lazy data works on the gridded arrays.

        Returns True if all gridded variables are dask arrays (lazy/deferred
        computation). Returns False if any has been computed to numpy.

        Returns
        -------
        bool
            True if data is lazy (dask), False if materialized (numpy).

        Examples
        --------
        >>> if phase.is_lazy:
        ...     phase = phase.compute()
        >>> assert not phase.is_lazy  # Now it's computed
        """
        import dask.array as da

        for key, ds in self.items():
            # a Dataset's variables, or a DataArray as its own one (N81)
            for var, arr in BatchCore._vars_of(ds).items():
                if 'y' not in arr.dims or 'x' not in arr.dims:
                    continue
                if not isinstance(arr.data, da.Array):
                    return False
            break  # Only check first burst
        return True

    @staticmethod
    def _require_lazy(batch, func_name: str):
        """
        Require that batch data is lazy (dask arrays) for memory-efficient processing.

        Parameters
        ----------
        batch : BatchCore
            Batch to validate.
        func_name : str
            Name of the calling function for error messages.

        Raises
        ------
        TypeError
            If data is not a dask array (e.g., numpy array from .compute(load=True)).
        """
        if not batch.is_lazy:
            # name the OFFENDING variable: reporting the first one prints the
            # type of a perfectly lazy array and hides which is materialised
            import dask.array as da
            offenders = []
            for key, ds in batch.items():
                for var, arr in BatchCore._vars_of(ds).items():
                    if 'y' not in arr.dims or 'x' not in arr.dims:
                        continue
                    if not isinstance(arr.data, da.Array):
                        offenders.append(f"{var} ({type(arr.data).__name__})")
                break
            raise TypeError(
                f"{func_name}() requires lazy (dask) data; these gridded "
                f"variables are materialised: {', '.join(offenders) or 'unknown'}. "
                f"Use .chunk('auto') to convert to dask arrays before calling {func_name}()."
            )

    @staticmethod
    def _gaussian(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5, device='auto',
                  pixel_sizes=None, resolution=67.0):
        """2D Gaussian convolution. See utils_gaussian.gaussian_numpy for full docs."""
        from .utils_gaussian import gaussian_numpy
        return gaussian_numpy(data_np, weight_np, sigma, truncate, threshold, device, pixel_sizes, resolution)

    def __init__(self, mapping: Mapping[str, xr.Dataset] | Stack | BatchComplex | None = None):
        from .Stack import Stack
        from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
        #print('BatchCore __init__', 0 if mapping is None else len(mapping))
        # Batch/etc. initialization won't filter out the data when it's a child class of BatchCore
        if isinstance(mapping, (Stack, BatchComplex)) and not isinstance(self, (Batch, BatchWrap, BatchUnit, BatchComplex)):
            real_dict = {}
            for key, ds in mapping.items():
                # pick only the data_vars whose dtype is not complex
                real_vars = [v for v in ds.data_vars if ds[v].dtype.kind != 'c' and tuple(ds[v].dims) == ('y', 'x')]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('BatchCore __init__ mapping', mapping or {}, '\n')
        super().__init__(mapping or {})

    def from_dataset(self, data: xr.Dataset | xr.DataArray, **kwargs) -> Batch:
        """
        Create a Batch by selecting each burst's coordinates from a merged Dataset.

        The input Dataset should have been created via to_dataset() or have
        coordinates that are supersets of each burst's coordinates.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data to split back into per-burst datasets. A Batch of
            DataArrays takes its own variable from a merged Dataset (the one
            x['VV'].to_dataset() returns) and gives DataArrays back; a merged
            DataArray (merged['VV']) splits into DataArrays.

        Returns
        -------
        Batch
            A new Batch with the same keys as self, each containing the
            selected subset of the Dataset (or of the DataArray).

        Examples
        --------
        # Round-trip: merge, process, split
        merged = batch.to_dataset()
        processed = some_processing(merged)
        result = batch.from_dataset(processed)
        """
        from .Batch import Batch
        from .utils_dask import rechunk2d

        # Validate input type: a merged Dataset, or a merged DataArray (merged['VV']), which
        # comes back as a Batch of DataArrays (N81)
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"data must be xr.Dataset or xr.DataArray, got {type(data).__name__}")
        # A BATCH OF DATAARRAYS takes the merged Dataset's ONE variable (N81): to_dataset()
        # always returns a Dataset, and the round trip gives DataArrays back. A DataArray is
        # one variable, so its name is never compared; a Dataset of several RAISES, to be picked
        first = next(iter(dict.values(self)), None)
        if isinstance(data, xr.Dataset) and isinstance(first, xr.DataArray):
            data = BatchCore._mask_of_dataarray(data, first, 'from_dataset', what='data')

        def rechunk_like(arr, src):
            """`arr` in the chunks of this burst's `src`, where they differ."""
            if src is None or not hasattr(src.data, 'chunks'):
                return None
            chunks = dict(zip(src.dims, src.data.chunks))
            if isinstance(arr.data, np.ndarray) or \
               (hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks):
                return arr.chunk(chunks)
            return None

        out = {}
        for key, ds in dict.items(self):
            # Select burst's spatial extent - coordinates match exactly
            selected = data.sel(y=ds.y, x=ds.x)

            # Align non-spatial dims (pair, date) if needed
            for dim in data.dims:
                if dim not in ('y', 'x') and dim in ds.dims:
                    # Select matching size and assign burst's coordinates
                    selected = selected.isel({dim: slice(0, ds.sizes[dim])})
                    selected = selected.assign_coords({dim: ds.coords[dim]})

            # Rechunk to match self's chunk structure per burst: a burst's variables, or a
            # DataArray burst as its own one variable (N81)
            src_vars = BatchCore._vars_of(ds)
            if isinstance(selected, xr.DataArray):
                src = src_vars.get(selected.name, ds if isinstance(ds, xr.DataArray) else None)
                rechunked = rechunk_like(selected, src)
                if rechunked is not None:
                    selected = rechunked
            else:
                rechunked_vars = {}
                for var_name in selected.data_vars:
                    rechunked = rechunk_like(selected[var_name], src_vars.get(var_name))
                    if rechunked is not None:
                        rechunked_vars[var_name] = rechunked
                if rechunked_vars:
                    selected = selected.assign(rechunked_vars)

            # RESTORE THE PER-BURST METADATA FROM SELF. to_dataset() merges the
            # bursts into ONE raster, and only the grids can survive that: the
            # 1D geometry is per burst and per date, so a merged raster has
            # nowhere to put it. This Batch still holds it, and the split back
            # is where it belongs -- nothing else in the round trip knows both
            # halves.
            #
            # Without this an unwrapped phase comes back carrying no
            # radar_wavelength, near_range, earth_radius, SC_height_start,
            # rng_samp_rate or BPR, and nothing notices until a fit asks for
            # them: `KeyError: radar_wavelength` out of fit1d(), several cells
            # after the step that actually dropped it.
            #
            # Only what the merged Dataset did NOT bring back is restored, so
            # anything the processing computed keeps precedence over the
            # original, and only where the dims still line up.
            # A DataArray holds no metadata variables: only a Dataset burst carries them,
            # into a Dataset result (N81).
            both = isinstance(ds, xr.Dataset) and isinstance(selected, xr.Dataset)
            carry = {v: ds[v] for v in ds.data_vars
                     if v not in selected.data_vars
                     and not ('y' in ds[v].dims and 'x' in ds[v].dims)
                     and all(d in selected.sizes
                             and selected.sizes[d] == ds.sizes[d]
                             for d in ds[v].dims)} if both else {}
            if carry:
                selected = selected.assign(carry)
            carry_coords = {c: ds.coords[c] for c in ds.coords
                            if c not in selected.coords
                            and all(d in selected.sizes
                                    and selected.sizes[d] == ds.sizes[d]
                                    for d in ds.coords[c].dims)}
            if carry_coords:
                selected = selected.assign_coords(carry_coords)
            if type(ds) is type(selected):
                for a_, v_ in ds.attrs.items():
                    selected.attrs.setdefault(a_, v_)
            out[key] = selected
        return Batch(out)

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     n = len(self)
    #     if n <= 1:
    #         # delegate to the underlying dict repr
    #         return dict.__repr__(self)
    #     sample = next(iter(self.values()))
    #     if not 'date' in sample and not 'pair' in sample:
    #         return f'{self.__class__.__name__} object containing {len(self)} items'
    #     sample_len = f'{len(sample.date)} date' if 'date' in sample else f'{len(sample.pair)} pair'
    #     keys = list(self.keys())
    #     return f'{self.__class__.__name__} object containing {len(self)} items for {sample_len} ({keys[0]} ... {keys[-1]})'

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     sample = next(iter(self.values()))  # pick any dataset
    #     # figure out which stack coord we have
    #     if 'date' in sample.coords:
    #         count = sample.coords['date'].size
    #         axis_name = 'date'
    #     elif 'pair' in sample.coords:
    #         count = sample.coords['pair'].size
    #         axis_name = 'pair'
    #     else:
    #         # fallback if neither coord is present
    #         return f"{self.__class__.__name__} containing {len(self)} items"
    #     keys = list(self.keys())
    #     return (
    #         f"{self.__class__.__name__} containing {len(self)} items "
    #         f"for {count} {axis_name} "
    #         f"({keys[0]} … {keys[-1]})"
    #     )

    def __repr__(self):
        # empty case
        if not self:
            return f"{self.__class__.__name__}(empty)"

        n = len(self)
        # single‐item: show the actual Dataset repr
        if n == 1:
            key, ds = next(iter(self.items()))
            return f"{self.__class__.__name__}['{key}']:\n{ds!r}"

        # multi‐item: show summary
        sample = next(iter(self.values()))
        
        # Handle CoordCollection objects
        if isinstance(sample, self.CoordCollection):
            keys = list(self.keys())
            return f"{self.__class__.__name__} coords containing {n} items ({keys[0]} … {keys[-1]})"
        
        if 'date' in sample.coords:
            count = sample.coords['date'].size
            axis = 'date'
        elif 'pair' in sample.coords:
            count = sample.coords['pair'].size
            axis = 'pair'
        else:
            return f"{self.__class__.__name__} containing {n} items"

        keys = list(self.keys())
        return (
            f"{self.__class__.__name__} containing {n} items "
            f"for {count} {axis} "
            f"({keys[0]} … {keys[-1]})"
        )

    def __or__(self, other):
        # Batch | Mapping
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(self)
        merged.update(other)
        return type(self)(merged)

    def __ror__(self, other):
        # Mapping | Batch
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(other)
        merged.update(self)
        return type(self)(merged)

    @property
    def data(self) -> xr.Dataset:
        """
        Return the single Dataset in this Batch.

        Raises
        ------
        ValueError
            if the Batch has zero or more than one item.
        """
        n = len(self)
        if n != 1:
            raise ValueError(f'Batch.data is only available for single-item Batches, but this Batch has {n} items')
        # return the only Dataset
        return next(iter(self.values()))

    # @property
    # def chunks(self) -> tuple[int, int, int]:
    #     sample = next(iter(self.values()))
    #     # for DatasetCoarsen extract the original Dataset
    #     if hasattr(sample, 'obj'):
    #         sample = sample.obj
    #     data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]

    #     if sample[data_var].chunks is None:
    #         print ('WARNING: Batch.chunks undefined, i.e. the data is not lazy and parallel chunks processing is not possible.')
    #         return (1, -1, -1)
    #     else:
    #         return tuple(chunks[0] for chunks in sample[data_var].chunks)

    @property
    def crs(self) -> rio.crs.CRS:
        return next(iter(self.values())).rio.crs

    @property
    def chunks(self) -> dict[str, int]:
        try:
            sample = next(iter(self.values()))
        except StopIteration:
            return {}

        # for DatasetCoarsen extract the original Dataset
        if hasattr(sample, 'obj'):
            sample = sample.obj
        if isinstance(sample, xr.DataArray):
            # a Batch of DataArrays: the chunks of the DataArray itself
            arr = sample
        else:
            data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]
            arr = sample[data_var]

        chunks = arr.chunks
        #print ('chunks', chunks)
        if chunks is None:
            # Data is not lazy (numpy arrays) - return empty dict silently
            # Use batch.is_lazy to check if data is lazy before calling .chunks
            return {}

        # build dict of first‐chunk sizes, one chunk means chunk size 1 or -1
        return {dim: sizes[0] if len(sizes) > 1 else (1 if sizes[0] == 1 else -1) for dim, sizes in zip(arr.dims, chunks)}

    def __getitem__(self, key):
        """
        Access coordinates, data variables, or datasets in the batch.
        
        Parameters
        ----------
        key : str, list, or tuple
            If str: access coordinate or data variable across all datasets
            If list/tuple: select subset of datasets
            
        Returns
        -------
        Batch
            Batch of the requested coordinate/variable or selected datasets

        Raises
        ------
        KeyError
            if the key is neither a burst nor a name any burst carries.
        """
        # Handle list/tuple keys for dataset selection
        if isinstance(key, (list, tuple)):
            out = {}
            for burst_id, ds in self.items():
                miss = [k for k in key
                        if isinstance(k, str) and k.endswith('²')
                        and k not in getattr(ds, 'data_vars', ())]
                if miss:
                    ds = ds.assign({k: self._square_of(ds, k, burst_id)
                                    for k in miss})
                out[burst_id] = ds[list(key)]
            return self._view(out)

        # Try to access as a dataset key first
        if dict.__contains__(self, key):
            return super().__getitem__(key)
        # If not a dataset key, try to access as coordinate/variable
        return self._select_var(key)

    def _select_var(self, key):
        """The variable or coordinate `key` of every burst that carries it: a Batch
        of DataArrays, as xarray's ds[key] and ds.key both give (N81).

        THE CLASS IS THE GRIDS' (N81, V2). A (y, x) grid keeps this batch's
        class: w['VV'] is wrapped phase and stays a BatchWrap, corr['VV'] a
        BatchUnit, intf['VV'] a BatchComplex. Anything else -- BPR,
        near_range, residual, ref, rep, y, x -- is not what the class
        describes and comes back as a plain Batch.

        A SELECTION IS NOT A CONSTRUCTION either way: no constructor runs
        (_view), so the values are the stored ones. A BatchWrap's w.BPR used
        to keep the class, and the next isel() or compute() rebuilt it through
        BatchWrap(...) and wrapped it into [-pi, pi].
        """
        subset = {
            k: ds[key] if not isinstance(ds, self.CoordCollection) else ds._ds.coords[key]
            for k, ds in self.items()
            if (isinstance(ds, self.CoordCollection) and key in ds._ds.coords) or
               (not isinstance(ds, self.CoordCollection)
                and (key in ds.coords or key in getattr(ds, 'data_vars', ())))
        }
        # NEITHER A BURST NOR A NAME ANY BURST CARRIES: that is a miss, and
        # a miss raises -- as __getattr__ raises AttributeError and
        # Stack.__getitem__ raises here. An empty Batch made a typo, and a
        # burst a selection dropped, read as a result that simply held
        # nothing. A name some bursts carry still returns those bursts.
        if not subset:
            raise KeyError(key)
        if all(BatchCore._is_grid(v) for v in subset.values()):
            return self._view(subset)
        from .Batch import Batch
        return BatchCore._view_as(Batch, subset)

    def _view(self, mapping):
        """This batch's class over `mapping`, WITHOUT the conversion its constructor
        applies (BatchWrap wraps to [-pi, pi]).

        Class conversion happens on EXPLICIT construction only -- BatchWrap(x)
        -- never on a selection or a mask taken from data that already is of
        this class (N81): wrapping a boolean mask made it floats, and sel()
        then read the floats as labels and kept the wrong pairs.
        """
        return BatchCore._view_as(type(self), mapping)

    @staticmethod
    def _view_as(klass, mapping):
        """`klass` over `mapping` without running its constructor (see _view)."""
        out = dict.__new__(klass)
        dict.__init__(out, mapping)
        return out

    @staticmethod
    def _named_dtypes(mapping):
        """(name, dtype) of every grid of every Dataset in `mapping`, and of every
        DataArray; anything else (a coarsen() window) has none."""
        out = []
        for v in (dict.values(mapping) if isinstance(mapping, dict) else mapping.values()):
            if isinstance(v, xr.DataArray):
                out.append((v.name, v.dtype))
            elif isinstance(v, xr.Dataset):
                out += [(n, v[n].dtype) for n in v.data_vars if 'y' in v[n].dims and 'x' in v[n].dims]
        return out

    @classmethod
    def _keeps(cls, dtype) -> bool:
        """Whether a result of this dtype keeps the class (_as_class)."""
        return True

    @staticmethod
    def _require_float(mapping, what):
        """THE CLASS HOLDS ITS VALUES (N81): BatchWrap and BatchUnit hold float grids
        only. `what` names the class and its values in the error."""
        for name, dt in BatchCore._named_dtypes(mapping or {}):
            if dt.kind != 'f':
                raise TypeError(f'ERROR: {what}, got {dt} {name}. Use Batch(x).')

    @staticmethod
    def _as_class(klass, mapping, view=False):
        """`klass` over an operation's result when the class keeps its values,
        otherwise a plain Batch (N81).

        A BatchWrap holds wrapped float phase and a BatchUnit float units: a
        comparison, ~, np.isfinite() or astype(bool) of either is a mask, not a
        phase or a unit, and as a BatchWrap it was wrapped into floats, which
        sel() read as labels. A BatchComplex keeps any result but a boolean one.
        view=True skips the constructor (_view): a mask is not a construction.
        """
        if all(klass._keeps(dt) for _, dt in BatchCore._named_dtypes(mapping)):
            return BatchCore._view_as(klass, mapping) if view else klass(mapping)
        from .Batch import Batch
        return BatchCore._view_as(Batch, mapping)

    def _result(self, mapping, view=False):
        """This batch's class over an operation's result, or a plain Batch (_as_class)."""
        return BatchCore._as_class(type(self), mapping, view)

    @staticmethod
    def _square_of(ds, name, key=''):
        """`<var>²` built on demand from `<var>`: a VIRTUAL VARIABLE, no
        store holds it.

        It is `(v - centre)²` about the midpoint of the variable's own
        actual_range, not the raw square. `span{1, v, (v-c)²}` is
        `span{1, v, v²}` for any c, so the fit is the same either way
        while the numbers are not: the raw square of a map coordinate is a
        straight line to float32. The centre travels in the variable's own
        attrs, so whatever evaluates the model later rebuilds exactly this
        and not a square about some other point.
        """
        base = name[:-1]
        if base not in getattr(ds, 'data_vars', ()) and base not in ds.coords:
            raise KeyError(
                f"'{name}' is the square of '{base}', which "
                f"{'burst ' + repr(key) if key else 'this dataset'} does not "
                f"carry. Nothing stores a squared variable; it is built from "
                f"its base, so the base has to be there.")
        v = ds[base]
        ar = v.attrs.get('actual_range')
        if ar is None:
            raise ValueError(
                f"'{base}' carries no actual_range, so the centre of "
                f"'{name}' is unknown. Squaring about a measured midpoint "
                f"instead would make the variable depend on which crop it was "
                f"built from -- write the attribute.")
        c = 0.5 * (float(ar[0]) + float(ar[1]))
        q = ((v.astype('float64') - c) ** 2).astype('float32')
        q.attrs = {k: val for k, val in v.attrs.items() if k != 'actual_range'}
        q.attrs['square_of'] = base
        q.attrs['square_centre'] = c
        _hi = (max(abs(float(ar[0]) - c), abs(float(ar[1]) - c))) ** 2
        q.attrs['actual_range'] = [0.0, float(_hi)]
        return q

    def __getattr__(self, name: str):
        """Attribute-style access to a variable or coordinate: `x.name` is
        `x['name']`, a Batch of DataArrays, as xarray's ds.name is ds['name']
        (N81). Only names that are not methods reach here, as in xarray."""
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if not self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            return self._select_var(name)
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    # A BATCH OF DATAARRAYS TAKES EVERY CORE METHOD DIRECTLY (N81). A method on the
    # grids alone works variable by variable (_vars_of): a DataArray in, a DataArray
    # out, or the method's own named outputs (aspect(), stl()). A method that needs
    # the Dataset's non-grid variables -- radar_wavelength, BPR, the radar geometry --
    # raises for a DataArray at its start (_needs_dataset): the DataArray does not
    # carry them, and a fit without them is silently another fit.
    # ONLY THESE, ABOUT NAMED DATASET STRUCTURE BY DESIGN, see a Batch of DataArrays
    # as ONE-VARIABLE DATASETS named after the DataArray (what x[['name']] holds):
    # plot and export by name, the store layout, Dataset structure (assign,
    # rename_vars, merge) and align(), whose output adds the 'residual' variable.
    # So do the modules' own methods, which core does not define.
    _DA_AS_DATASETS = frozenset({
        'plot', 'plot2', 'rgb', 'to_dataframe', 'to_geojson', 'to_geopackage', 'to_vtk',
        'to_vtks', 'save', 'snapshot', 'assign', 'rename_vars', 'merge', 'align',
    })

    def __getattribute__(self, name):
        if name[0] != '_':
            first = next(iter(dict.values(self)), None)
            if isinstance(first, xr.DataArray):
                # a method or property of the Batch classes (the modules' ones included)
                for klass in type(self).__mro__:
                    if klass is dict:
                        break
                    attr = klass.__dict__.get(name)
                    if attr is not None:
                        if isinstance(attr, (FunctionType, property)) and \
                                (name in BatchCore._DA_AS_DATASETS or not BatchCore._is_core(attr)):
                            return getattr(BatchCore._as_datasets(self, name), name)
                        break
        return dict.__getattribute__(self, name)

    @staticmethod
    def _is_core(attr) -> bool:
        """Whether a method is core insardev's: defined here, or a module's wrapper of a
        core method (its __wrapped__), which then takes what the core method takes."""
        fn = attr.fget if isinstance(attr, property) else attr
        while fn is not None:
            if (getattr(fn, '__module__', None) or '').split('.')[0] == 'insardev':
                return True
            fn = getattr(fn, '__wrapped__', None)
        return False

    # the per-date radar geometry the scene-centre and pixelwise conversions read
    _GEOMETRY = ('radar_wavelength', 'near_range', 'rng_samp_rate', 'earth_radius', 'SC_height_start',
                 'SC_height_end', 'num_lines', 'num_rng_bins')

    @staticmethod
    def _needs_dataset(batch, what, names, action='Call it on the Dataset batch x.'):
        """A method that needs the Dataset's non-grid variables `names` RAISES for a Batch
        of DataArrays (N81): a DataArray does not carry them, and without them the result
        is another one, silently -- a fit without its baselines, a trend without its epoch."""
        first = next(iter(dict.values(batch)), None)
        if isinstance(first, xr.DataArray):
            raise TypeError(f"ERROR: {what}() needs the Dataset ({', '.join(names)}), which "
                            f"x['{first.name}'] does not carry. {action}")

    @staticmethod
    def _needs_vars(batch, what, names, action='Call it on the Dataset batch x.'):
        """_needs_dataset(), and a Dataset burst without any of `names` RAISES too, naming
        them (N81): x[['VV']] drops them as x['VV'] does, and the result is then another
        one, silently -- no height term, or another reference date."""
        BatchCore._needs_dataset(batch, what, names, action)
        for key, ds in dict.items(batch):
            missing = [n for n in names if n not in ds.variables] if isinstance(ds, xr.Dataset) else []
            if missing:
                raise KeyError(f"ERROR: {what}() needs {', '.join(missing)}, which burst {key} "
                               f"does not carry. {action}")

    @staticmethod
    def _as_datasets(batch, what):
        """A Batch of DataArrays as one-variable Datasets named after each DataArray, for `what`()
        (_DA_AS_DATASETS and the modules' methods only)."""
        out = {}
        for k, v in dict.items(batch):
            if isinstance(v, xr.DataArray):
                if v.name is None:
                    raise TypeError(f"ERROR: {what}() needs a named DataArray batch. Name it: x.rename('name').")
                if v.name in v.coords:
                    # a coordinate's own DataArray (x.BPR on pair products, x.pair) carries
                    # that coordinate, and xarray refuses a Dataset of it: the variable is
                    # the coordinate's values, so the coordinate itself is dropped
                    v = v.drop_vars(v.name)
                v = v.to_dataset()
            out[k] = v
        return batch._view(out)

    @staticmethod
    def _is_dataarray_batch(batch) -> bool:
        """A Batch of DataArrays (N81): processed as xarray processes a DataArray, with or
        without (y, x) -- only a Batch of Datasets applies its operators to the grids only."""
        return isinstance(next(iter(dict.values(batch)), None), xr.DataArray)

    @staticmethod
    def _is_grid(da) -> bool:
        """A GRIDDED variable carries both y and x (N81)."""
        return 'y' in da.dims and 'x' in da.dims

    @staticmethod
    def _vars_of(value) -> dict:
        """{name: DataArray} of one burst: a Dataset's variables, or a DataArray as its own
        one variable. A method that works variable by variable takes a Batch of DataArrays
        directly through it (N81), with no Dataset built and no name required."""
        if isinstance(value, xr.DataArray):
            return {value.name: value}
        return {v: value[v] for v in value.data_vars}

    @staticmethod
    def _acquisition_order(batch, who='to_dataset') -> list:
        """THE BURSTS IN ACQUISITION ORDER, earliest first: the one order of every merge
        of bursts, to_dataset() and fit3d()'s seams, where the MOST RECENT burst wins.

        Read from each burst's OWN startTime variable (decided 2026-10-01), the one
        a stack loads per date and pairs() gives per pair (the earlier of its two
        dates): a burst comes at its earliest startTime. S1 stores keep its fraction
        (since 2026-10-01: the subswaths of one burst number start within a second);
        equal times, in older whole-second stores, keep the fullBurstID order. NISAR
        keeps whole seconds, its scenes being about 35 s apart. A batch without startTime -- a
        transform grid or a Batch of DataArrays, by design (user, 2026-10-01) -- is
        ordered by the fullBurstID numbers: within a track they count the
        acquisition cycles along the orbit, and the subswaths of a cycle are
        acquired IW1, IW2, IW3. Bursts only some of which carry startTime, or
        names that are not fullBurstIDs, are ordered so with a WARNING."""
        import re
        import dask
        keys = list(batch.keys())

        def natural(key):
            return [(0, int(p), '') if p.isdigit() else (1, 0, p) for p in re.split(r'(\d+)', str(key)) if p]
        by_name = sorted(keys, key=natural)
        if len(keys) < 2:
            return by_name
        timed = [isinstance(v, xr.Dataset) and 'startTime' in v.data_vars for v in (dict.__getitem__(batch, k) for k in keys)]
        if all(timed):
            # an opened store's startTime is lazy: every burst's in ONE compute
            starts = dask.compute(*[dict.__getitem__(batch, k)['startTime'].data for k in keys])
            first = {k: np.min(np.asarray(s, dtype='datetime64[us]')) for k, s in zip(keys, starts)}
            rank = {k: i for i, k in enumerate(by_name)}
            return sorted(keys, key=lambda k: (first[k], rank[k]))
        other = [str(k) for k in keys if not re.fullmatch(r'\d+_\d+(_[A-Z]+\d+)?', str(k))]
        if other:
            print(f"WARNING: {who}(): {', '.join(other[:3])}{', ...' if len(other) > 3 else ''} "
                  f"{'is not a fullBurstID' if len(other) == 1 else 'are not fullBurstIDs'}: bursts ordered by name.")
        elif any(timed):
            print(f"WARNING: {who}(): {timed.count(False)} of {len(keys)} bursts carry no startTime: "
                  f"bursts ordered by fullBurstID.")
        return by_name

    def _start_from(self, source):
        """THIS BATCH WITH EACH BURST'S startTime FROM `source`: the library's own selections
        of grids ahead of a merge (plot, to_vtk) drop it, and it is the burst order of the
        merge (_acquisition_order). The class and the grids stay as they are; a burst or a
        source without startTime is left as it is."""
        out = {}
        for k, v in dict.items(self):
            s = dict.get(source, k)
            if isinstance(v, xr.Dataset) and isinstance(s, xr.Dataset) and 'startTime' in s.data_vars:
                v = v.assign(startTime=s['startTime'])
            out[k] = v
        return self._view(out)

    @staticmethod
    def _grids_of(value) -> dict:
        """{name: DataArray} of the (y, x) grids of one burst (_vars_of, grids only)."""
        return {n: a for n, a in BatchCore._vars_of(value).items() if BatchCore._is_grid(a)}

    @staticmethod
    def _form(value, out_vars, **dataset_kwargs):
        """A burst's result in its input's form (N81): a DataArray burst gives its one
        output variable back as a DataArray, named as the output is; a Dataset burst gives
        xr.Dataset(out_vars, **dataset_kwargs). No Dataset is built for a DataArray.
        A DataArray with no output -- no (y, x) grid, or not one the method takes -- RAISES."""
        if isinstance(value, xr.DataArray):
            if not out_vars:
                if not BatchCore._is_grid(value):
                    raise BatchCore._no_grid_error(value)
                raise TypeError(f"ERROR: x['{value.name}'] ({value.dtype}, dims {tuple(value.dims)}) "
                                f"is not a grid this method takes.")
            (name, res), = out_vars.items()
            return res.rename(name)
        return xr.Dataset(out_vars, **dataset_kwargs)

    @staticmethod
    def _no_grid_error(value, who='x'):
        """THE ERROR FOR A BURST WITHOUT ANY (y, x) GRID (N81), _form's text: a DataArray
        names itself and its dims, a Dataset lists its variables. `who` names the argument."""
        if isinstance(value, xr.DataArray):
            return TypeError(f"ERROR: {who}['{value.name}'] has no (y, x) grid, dims {tuple(value.dims)}.")
        names = [str(v) for v in value.data_vars]
        listed = ', '.join(names[:8]) + (', ...' if len(names) > 8 else '')
        return TypeError(f"ERROR: {who} has no (y, x) grid ({listed}).")

    @staticmethod
    def _weight_of(w, var):
        """The weight grid of one burst for the data grid `var`: a DataArray weight weights
        every grid as it is (N81); a Dataset weight gives the grid of the same name for a
        Dataset's grid (_weight() matched the names), or its one grid for a DataArray, whose
        name is never compared (_weight() refused a weight with several)."""
        if w is None or isinstance(w, xr.DataArray):
            return w
        if var in w.data_vars:
            return w[var]
        grids = [n for n in w.data_vars if BatchCore._is_grid(w[n])]
        if len(grids) != 1:
            listed = ', '.join(str(n) for n in grids[:8]) + (', ...' if len(grids) > 8 else '')
            raise TypeError(f"ERROR: the weight has {len(grids)} grids ({listed}) for one DataArray. "
                            f"Pick one: weight['{grids[0] if grids else 'name'}'].")
        return w[grids[0]]

    @staticmethod
    def _grid_vars(ds) -> list:
        """The gridded variables of a Dataset; a Dataset without any RAISES (N81)."""
        grids = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
        if not grids:
            names = [str(v) for v in ds.data_vars]
            if not names:
                raise TypeError('ERROR: no (y, x) variables to operate on: the Dataset has none. '
                                "Select a variable: x['name'].")
            pick = next((v for v in names if ds[v].dtype.kind in 'biufc'), names[0])
            listed = ', '.join(names[:8]) + (', ...' if len(names) > 8 else '')
            raise TypeError(f"ERROR: no (y, x) variables to operate on ({listed}). Select one: x['{pick}'].")
        return grids

    @staticmethod
    def _mask_names(mgrids, names, how):
        """THE MASK RULE (N81): a Dataset mask masks each grid `names` holds by its
        grid of the same name (`mgrids`); a grid it lacks RAISES. There is no
        case where one mask grid masks every grid: that is a DataArray mask."""
        for n in names:
            if n not in mgrids:
                raise ValueError(f"ERROR: mask has no variable {n}. Use a DataArray mask to mask every "
                                 f"variable: x.{how}(mask['{mgrids[0]}'])")

    @staticmethod
    def _mask_of_dataarray(mask, da, how, what='mask'):
        """THE VARIABLE OF A DATASET FOR A DATAARRAY (N81): a DataArray is one variable by
        design and compares to a DataArray or to a Dataset of one variable, so its name
        is NEVER compared -- that is a Dataset's rule, which has many. A Dataset (`what`:
        a mask, a merged Dataset) gives its one variable (its one (y, x) grid for a
        gridded DataArray, whatever else rides along); several are ambiguous and RAISE."""
        names = [str(v) for v in mask.data_vars]
        if BatchCore._is_grid(da):
            names = [v for v in names if BatchCore._is_grid(mask[v])] or names
        if len(names) == 1:
            return mask[names[0]]
        if not names:
            raise ValueError(f"ERROR: the {what} has no variables for x.{how}().")
        listed = ', '.join(names[:8]) + (', ...' if len(names) > 8 else '')
        raise ValueError(f"ERROR: the {what} has {len(names)} variables ({listed}) for one DataArray. "
                         f"Pick one: x.{how}({what}['{names[0]}'])")

    @staticmethod
    def _no_grid(what, value):
        """THE ERROR FOR A MASK OR A WEIGHT WITHOUT (y, x) on Dataset data (N81,
        decided): it RAISES, never broadcasts -- such a mask is commonly a reduction,
        arr.mean(), that dropped y and x, and broadcast it is very hard to find."""
        if isinstance(value, xr.Dataset):
            names = list(value.data_vars)
            value = value[next((v for v in names if value[v].dtype.kind in 'biufc'), names[0])] if names else None
        name = getattr(value, 'name', None)
        dims = tuple(getattr(value, 'dims', ()))
        return TypeError(f"ERROR: {what} '{name}' has no (y, x) grid, dims {dims}: "
                         f"a reduction such as .mean() may have dropped y and x.")

    @staticmethod
    def _mask_grid(mask):
        """THE MASK RULE (N81): a DataArray mask masks every grid of a Dataset; a
        mask without (y, x) RAISES there, as a Dataset mask without grids does. (A
        Batch of DataArrays takes its mask directly, as in xarray.)"""
        if isinstance(mask, xr.DataArray) and not BatchCore._is_grid(mask):
            raise BatchCore._no_grid('mask', mask)
        return mask

    @staticmethod
    def _mask_dataset(mask):
        """A Dataset mask needs (y, x) grids: one without any RAISES (_no_grid)."""
        if not any(BatchCore._is_grid(mask[v]) for v in mask.data_vars):
            raise BatchCore._no_grid('mask', mask)
        return BatchCore._grid_vars(mask)

    @staticmethod
    def _weight(weight, data=None, name='weight', required=False, by_name=True):
        """A WEIGHT IS A BATCHUNIT (N81), checked where a function starts.

        None is no weight. Anything but a BatchUnit RAISES: the Batches methods
        took their second element only when it was a BatchUnit and dropped any
        other without a word, and the rest took any dict and failed later, or
        not at all. Given `data` (the batch the weight applies to), every burst
        of it needs a weight -- one missing was skipped, unweighted -- and a
        BatchUnit of (y, x) DataArrays, BatchUnit(corr['VV']) or corr['VV'],
        weights every grid of its burst. The weight comes back as it is, never
        converted: _weight_of(weight[key], var) gives the grid for `var`.

        A DATASET WEIGHT MATCHES BY NAME, as a Dataset mask does: each grid of
        `data` is weighted by the weight's grid of the same name, and a grid it
        lacks RAISES -- gaussian() and unwrap2d_snaphu() smoothed such a grid
        unweighted and rmse() took the weight's first grid instead, without a
        word. by_name=False skips the match for a caller that weights every grid
        by one grid of the weight (Batches.coherent()). required=True raises for
        None too, where the weight is not optional (goldstein()). A Batch of
        DATAARRAYS is one variable per burst by design: no name is compared, it
        takes the Dataset weight's one grid, and a weight with several raises,
        to be picked (weight['VV']) -- as a DataArray takes a Dataset mask.
        """
        import xarray as xr
        from .Batch import BatchUnit
        if weight is None and not required:
            return None
        if not isinstance(weight, BatchUnit):
            # a boolean or integer weight is not a unit and BatchUnit() refuses it:
            # the hint converts it, instead of sending the caller back and forth
            m = weight if isinstance(weight, dict) else (
                {'': weight} if isinstance(weight, (xr.Dataset, xr.DataArray)) else {})
            cast = any(dt.kind != 'f' for _, dt in BatchCore._named_dtypes(m))
            hint = f"BatchUnit({name}.astype('float32'))" if cast else f'BatchUnit({name})'
            raise TypeError(f'ERROR: {name} must be a BatchUnit, got {type(weight).__name__}. Use {hint}.')
        if data is None:
            return weight
        missing = [k for k in data.keys() if not dict.__contains__(weight, k)]
        if missing:
            more = f' and {len(missing) - 1} more' if len(missing) > 1 else ''
            raise ValueError(f'ERROR: {name} has no burst {missing[0]}{more}. '
                             f'Give a {name} for every burst.')
        for k in data.keys():
            wv = dict.get(weight, k)
            # a Dataset weight without (y, x) grids RAISES (N81, decided): it broadcast, or
            # failed deep in the kernel on a shape
            if isinstance(wv, xr.Dataset) and not any(BatchCore._is_grid(wv[n]) for n in wv.data_vars):
                raise BatchCore._no_grid(name, wv)
        if by_name:
            for k in data.keys():
                wv = dict.get(weight, k)
                if not isinstance(wv, xr.Dataset):
                    continue
                d = dict.get(data, k)
                if isinstance(d, xr.Dataset):
                    names = [n for n in d.data_vars if BatchCore._is_grid(d[n])]
                else:
                    # A DATAARRAY IS ONE VARIABLE by design: no name is compared, it takes the
                    # weight's one grid (_weight_of), and a weight with several is ambiguous
                    names = []
                    grids = [str(n) for n in wv.data_vars if BatchCore._is_grid(wv[n])]
                    if isinstance(d, xr.DataArray) and BatchCore._is_grid(d) and len(grids) > 1:
                        listed = ', '.join(grids[:8]) + (', ...' if len(grids) > 8 else '')
                        raise ValueError(f"ERROR: {name} has {len(grids)} grids ({listed}) for one DataArray. "
                                         f"Pick one: {name}['{grids[0]}'].")
                lack = [n for n in names if n not in wv.data_vars]
                if lack:
                    have = [n for n in wv.data_vars if BatchCore._is_grid(wv[n])]
                    form = f"{name}['{have[0]}']" if have else f"{name}['name']"
                    raise ValueError(f'ERROR: {name} has no variable {lack[0]}. Use a DataArray {name} '
                                     f'to weight every variable: {form}.')
                flat = [n for n in names if not BatchCore._is_grid(wv[n])]
                if flat:
                    raise BatchCore._no_grid(name, wv[flat[0]])
        for v in dict.values(weight):
            # a DataArray weight without (y, x) RAISES too; one with them is taken as it
            # is, no Dataset built: _weight_of() gives it for every grid
            if isinstance(v, xr.DataArray) and not BatchCore._is_grid(v):
                raise BatchCore._no_grid(name, v)
        return weight

    @staticmethod
    def _binary_vars(ds, other, op):
        """`op(ds, other)` by THE OPERATOR RULE (N81), for every operator on a burst.

        A Dataset is a container of GRIDS: an operator acts on its gridded
        variables -- those carrying both y and x -- and every other variable
        (numeric or string, any dims: burst ids, radar metadata, per-date
        polynomials, a 'residual') passes through from the Dataset unchanged,
        always. There is no second mode: a Dataset with no gridded variable
        raises, naming its variables, and the caller selects the one it means
        as a DataArray, x['HH'] / 2, where the operation is unambiguous.

        A DataArray takes the operator directly. Against a Dataset it applies to
        each gridded variable of the Dataset, and the Dataset's other variables
        pass through, in either order.

        Dataset op Dataset pairs the two by gridded variable name, as xarray
        does; with no name in common it raises instead of returning nothing.
        A field that applies to every polarisation is a DataArray, x * w['VV'].
        """
        if isinstance(ds, xr.DataArray):
            if isinstance(other, xr.Dataset):
                return BatchCore._binary_vars(other, ds, lambda a, b: op(b, a))
            return op(ds, other)
        grids = BatchCore._grid_vars(ds)
        rhs = other
        if isinstance(other, xr.Dataset):
            ogrids = BatchCore._grid_vars(other)
            if not set(grids) & set(ogrids):
                raise TypeError(f"ERROR: no common (y, x) variables to operate on: {[str(v) for v in grids]} "
                                f"and {[str(v) for v in ogrids]}. Select one: y['{ogrids[0]}'].")
            rhs = other[ogrids]
        res = op(ds[grids], rhs)
        for v in ds.data_vars:
            if v not in grids:
                res[v] = ds[v]
        res.attrs = ds.attrs
        return res

    def __add__(self, other):
        # scalar + batch. Routed through _binary_vars like the Dataset case:
        # a bare `v + other` hits the string metadata and raises UFuncTypeError
        if isinstance(other, _SCALARS):
            import operator as _operator
            return self._result({k: BatchCore._binary_vars(v, other, _operator.add)
                               for k, v in self.items()})
        keys = self.keys()
        import operator as _operator
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.add)
                              if k in other else self[k]) for k in keys})

    def __radd__(self, other):
        # scalar + batch → same as batch + scalar
        return self.__add__(other)

    def _sub_coeffs(self, k, ds, val):
        """`ds - val` for one burst, where `val` may be per-pair polynomial
        coefficients from align(): [[ramp, off], ...] (degree 1), [off, ...]
        (degree 0) or [ramp, offset] (one pair). The correction is a DataArray
        (polyval), so it subtracts from every grid by the operator rule."""
        import operator as _operator
        if not isinstance(val, (list, tuple)) or len(val) == 0:
            return BatchCore._binary_vars(ds, val, _operator.sub)
        sample_da = ds if isinstance(ds, xr.DataArray) else ds[BatchCore._grid_vars(ds)[0]]
        has_pair_dim = 'pair' in sample_da.dims
        n_pairs = sample_da.sizes.get('pair', 1)
        if isinstance(val[0], (list, tuple)):
            # multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
            return BatchCore._binary_vars(ds, self._view({k: ds}).polyval({k: val})[k], _operator.sub)
        if has_pair_dim and len(val) == n_pairs:
            # multi-pair degree=0: [off0, off1, ...], concrete scalars or dask 0-d arrays
            if any(hasattr(v, 'dask') for v in val):
                import dask.array as _da
                offsets = xr.DataArray(_da.stack(val), dims=['pair'])
            else:
                offsets = xr.DataArray(val, dims=['pair'])
            return BatchCore._binary_vars(ds, offsets, _operator.sub)
        if len(val) == 1:
            # a single value wrapped in a list: [offset]
            return BatchCore._binary_vars(ds, val[0], _operator.sub)
        # single pair degree=1: [ramp, offset]
        return BatchCore._binary_vars(ds, self._view({k: ds}).polyval({k: val})[k], _operator.sub)

    def __sub__(self, other):
        # batch - scalar
        if isinstance(other, _SCALARS):
            import operator as _operator
            return self._result({k: BatchCore._binary_vars(v, other, _operator.sub)
                               for k, v in self.items()})
        # THE LISTS SUBTRACT FROM THE GRIDS, as every operator does: a
        # whole-Dataset `ds - offsets` reached the per-pair metadata too, so
        # align() on an unwrapped phase shifted radar_wavelength by the burst's
        # offset, for fit1d() and predict() to read, and raised TypeError on the
        # burst strings. _binary_vars is the one place that rule lives.
        return self._result({k: (self._sub_coeffs(k, self[k], other[k]) if k in other else self[k])
                           for k in self.keys()})

    def __rsub__(self, other):
        # scalar - batch: the operand order is flipped, the metadata rule is not
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(v, other, lambda a, b: b - a)
                               for k, v in self.items()})
        return NotImplemented

    def __mul__(self, other):
        # batch * scalar
        if isinstance(other, _SCALARS):
            import operator as _operator
            return self._result({k: BatchCore._binary_vars(v, other, _operator.mul)
                               for k, v in self.items()})
        keys = self.keys()
        import operator as _operator
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.mul)
                              if k in other else self[k]) for k in keys})

    def __rmul__(self, other):
        # scalar * batch
        return self._result({k: BatchCore._binary_vars(v, other, lambda a, b: b * a)
                           for k, v in self.items()})

    def __truediv__(self, other):
        # batch / scalar
        if isinstance(other, _SCALARS):
            import operator as _operator
            return self._result({k: BatchCore._binary_vars(v, other, _operator.truediv)
                               for k, v in self.items()})
        keys = self.keys()
        import operator as _operator
        return self._result({k: (BatchCore._binary_vars(self[k], other[k], _operator.truediv)
                              if k in other else self[k]) for k in keys})

    def __rtruediv__(self, other):
        # scalar / batch: flipped operands, same metadata rule
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(v, other, lambda a, b: b / a)
                               for k, v in self.items()})
        return NotImplemented

    def __neg__(self):
        # -batch: the grids negated, the rest carried
        return self._result({k: BatchCore._binary_vars(v, None, lambda a, _: -a) for k, v in self.items()})

    def __abs__(self):
        # abs(batch) is batch.abs(): the grids' magnitude, the rest carried (a
        # BatchComplex gives a real Batch); Python had no abs() for a batch at all
        return self.abs()

    def _binop(self, other, op):
        """
        generic helper for any binary operator `op(ds, other)` or `op(ds, other_ds)`

        A comparison or a logical operator follows the operator rule
        (_binary_vars): it acts on the grids and carries the metadata through --
        numpy raises UFuncTypeError comparing the burst strings with a number,
        so `corr >= 0.3` failed on any Batch carrying them. A MASK IS NOT A
        CONSTRUCTION, and not a phase or a unit either: a boolean result of any
        class is a plain Batch (_as_class), never wrapped into floats.
        """
        if isinstance(other, _SCALARS):
            return self._result({k: BatchCore._binary_vars(ds, other, op) for k, ds in self.items()}, view=True)
        elif isinstance(other, BatchCore):
            return self._result({k: BatchCore._binary_vars(self[k], other[k], op) for k in self if k in other},
                                view=True)
        else:
            return NotImplemented

    def __gt__(self, other):   return self._binop(other, operator.gt)
    def __lt__(self, other):   return self._binop(other, operator.lt)
    def __ge__(self, other):   return self._binop(other, operator.ge)
    def __le__(self, other):   return self._binop(other, operator.le)
    def __eq__(self, other):   return self._binop(other, operator.eq)
    def __ne__(self, other):   return self._binop(other, operator.ne)
    def __and__(self, other):  return self._binop(other, operator.and_)
    def __or__(self, other):   return self._binop(other, operator.or_)
    # the mask a comparison made carries its metadata unchanged, and numpy has no ~ for a string or a float
    def __invert__(self):      return self._result({k: BatchCore._binary_vars(v, None, lambda a, _: ~a)
                                                    for k, v in self.items()}, view=True)

    # reversed ops
    __rgt__ = __gt__
    __rlt__ = __lt__
    __rand__ = __and__
    __ror__ = __or__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Support numpy ufuncs on Batch objects, e.g.:
        - np.exp(-1j * intfs)
        - np.isfinite(weight)
        - np.abs(batch)
        """
        # only handle the normal call
        if method != "__call__":
            return NotImplemented

        # find the first Batch among inputs
        batch = next((x for x in inputs if isinstance(x, BatchCore)), None)
        if batch is None:
            return NotImplemented

        # THE OPERATOR RULE (_binary_vars): a Dataset argument gives the ufunc its
        # grids, and the first Dataset's other variables pass through; a
        # DataArray is taken directly
        result = {}
        for k in batch.keys():
            # build the argument list for this key
            args = [
                inp[k] if isinstance(inp, BatchCore) else inp
                for inp in inputs
            ]
            carrier = next((a for a in args if isinstance(a, xr.Dataset)), None)
            args = [a[BatchCore._grid_vars(a)] if isinstance(a, xr.Dataset) else a for a in args]
            res = ufunc(*args, **kwargs)
            if carrier is not None and isinstance(res, xr.Dataset):
                for v in carrier.data_vars:
                    if not BatchCore._is_grid(carrier[v]):
                        res[v] = carrier[v]
                res.attrs = carrier.attrs
            result[k] = res

        return self._result(result)

    # def iexp(self):
    #     """
    #     np.exp(-1j * intfs)
    #     """
    #     import numpy as np
    #     return np.exp(1j * self)

    # def conj(self) -> BatchWrap:
    #     """
    #     Return a new BatchWrap in which each complex dataset has been
    #     replaced with its complex conjugate.

    #     Example:
    #     intfs.iexp().conj() for np.exp(-1j * intfs)
    #     """
    #     return type(self)({
    #         k: ds.conj()
    #         for k, ds in self.items()
    #     })

    def _map_grids(self, func, _numeric_only=True, **kwargs) -> dict:
        """func(DataArray) -> DataArray on the GRIDS of every burst, by the operator
        rule (_binary_vars): every other variable -- numeric or string, any dims
        -- passes through unchanged, a Dataset without grids raises and a
        DataArray takes func directly. A plain dict, for the caller's class.
        _numeric_only=False applies func to every grid whatever its dtype (a
        boolean mask too), as astype() needs."""
        def apply(ds):
            if isinstance(ds, xr.DataArray):
                return func(ds, **kwargs)
            grids = BatchCore._grid_vars(ds)
            result_vars = {}
            for var in ds.data_vars:
                da = ds[var]
                # the grids only, and of those the numeric ones, as before
                if var in grids and (not _numeric_only or np.issubdtype(da.dtype, np.number)):
                    result_vars[var] = func(da, **kwargs)
                else:
                    result_vars[var] = da
            result = xr.Dataset(result_vars)
            result.attrs = ds.attrs
            return result

        return {k: apply(ds) for k, ds in self.items()}

    def map_da(self, func, **kwargs):
        """Apply func(DataArray) → DataArray to the gridded variables of every
        dataset, or to every DataArray of a Batch of DataArrays.

        THE OPERATOR RULE: the (y, x) variables are the data; every other
        variable passes through unchanged -- the radar metadata was converted
        before, clip() set near_range to 1.0 and ** squared radar_wavelength.
        A Dataset without (y, x) variables raises.
        """
        return self._result(self._map_grids(func, **kwargs))

    def astype(self, dtype, **kwargs):
        # EVERY grid is converted, whatever its dtype: the numeric-only map helpers
        # pass a boolean grid through, so (ph > 0).astype('float32') stayed boolean
        # while the DataArray form (ph['VV'] > 0).astype('float32') converted
        return self._result(self._map_grids(lambda da: da.astype(dtype), _numeric_only=False, **kwargs))
    
    def abs(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs)

    def square(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.square(da), **kwargs)

    def sqrt(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.sqrt(da), **kwargs)

    def log10(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.log10(da), **kwargs)

    def multiply(self, value, **kwargs):
        return self.map_da(lambda da: da * value, **kwargs)

    def divide(self, value, **kwargs):
        return self.map_da(lambda da: da / value, **kwargs)

    def clip(self, min=None, max=None, **kwargs):
        return self.map_da(lambda da: da.clip(min=min, max=max), **kwargs)

    def isfinite(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.isfinite(da), **kwargs)

    def fillna(self, value=0, **kwargs):
        """Replace NaN with `value`, e.g. `velocity.fillna(0)`.

        The twin of `where()`: that one makes holes, this one closes them. It
        is what a plot or an export wants, where NaN is drawn as nothing and a
        reader cannot tell an unsolved pixel from a missing one -- and nothing
        the analysis wants, since a filled pixel then carries a number no
        measurement produced.

        NUMERIC VARIABLES ONLY, as every elementwise method here works:
        map_da() passes strings and objects through untouched, so the burst id
        and the rest of the metadata that rides along in each dataset are not
        candidates for filling.
        """
        return self.map_da(lambda da: da.fillna(value), **kwargs)

    # def where(self, cond, other=0):
    #     # cond can be a BatchWrap of booleans
    #     if isinstance(cond, BatchWrap):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return self.map_da(lambda da: da.where(cond, other), keep_attrs=True)

    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch‐wise .where:
        
    #     - if `cond` is a Batch (or BatchWrap) with exactly the same keys:
    #         * when other==0 → do ds * mask  (very fast, no alignment)
    #         * otherwise   → ds.where(mask, other, **kwargs)
    #     - else:
    #         broadcast a single mask or scalar/DataArray
    #         into every var via `map_da(lambda da: da.where(cond, other, **kwargs))`.
    #     """
    #     # per‐burst mask
    #     if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
    #         print ('X')
    #         out = {}
    #         for k, ds in self.items():
    #             mask = cond[k]
    #             # if mask coords don't exactly match ds, you can
    #             # uncomment the next line to reindex first:
    #             # mask = mask.reindex_like(ds, method='nearest')
                
    #             if other == 0:
    #                 # blaze past .where with a simple multiply
    #                 out[k] = ds * mask
    #             else:
    #                 out[k] = ds.where(mask, other, **kwargs)
    #         return type(self)(out)

    #     # single‐mask/scalar-broadcast case:
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)


    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch-wise .where: if cond is another Batch with exactly the same keys,
    #     do each ds.where(mask, other), otherwise fall back to per-DataArray broadcast.
    #     """
    #     # 1) fast path: cond is a Batch with the same bursts
    #     if isinstance(cond, Batch) and set(cond.keys()) == set(self.keys()):
    #         return type(self)({
    #             k: ds.where(cond[k], other, **kwargs)
    #             for k, ds in self.items()
    #         })

    #     # 2) broadcast a single mask/scalar to every var
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def where(self, cond, other=np.nan, **kwargs):
        """
        Batch-wise .where: keep the values where `cond` is True, `other` elsewhere.

        `cond` is a Batch with the same bursts (a mask per burst, reindexed to
        each grid's coordinates), or a scalar or a DataArray applied to every
        grid by map_da().

        keep_attrs=True argument can be used to preserve attributes of the original data.

        Examples
        --------
        >>> ph.where(corr > 0.3)          # VV by corr's VV (and VH by VH)
        >>> ph.where(corr['VV'] > 0.3)    # every grid by corr's VV

        THE MASK RULE (N81): only the (y, x) variables are masked, everything
        else passes through, and the mask's own metadata plays no part. A
        DATASET mask (corr > 0.3) masks each grid by its grid of the SAME NAME
        -- VV by VV, VH by VH -- and a grid it lacks raises; a DATAARRAY mask
        (corr['VV'] > 0.3) masks every grid. A Dataset, or a mask for Datasets,
        without (y, x) variables raises for a Batch of Datasets, whose
        operators apply to the (y, x) grids only, as in mask(). A BATCH OF
        DATAARRAYS NEEDS NO GRID: it takes its mask directly, as in xarray --
        a DataArray mask as it is, a Dataset mask's ONE variable -- with or
        without (y, x) (w['BPR'].where(...)). A DataArray is one variable by
        design, so no name is compared: ele.where(adi < 0.4) takes adi's VV;
        a mask with several variables raises, to be picked (mask['VV']).
        """
        # detect same key Batch-like mask
        if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
            out = {}
            for k, ds in self.items():
                mask_obj = cond[k]
                if isinstance(mask_obj, xr.Dataset) and isinstance(ds, xr.DataArray):
                    # a DataArray takes the mask's one variable directly, grid or not: no name compared
                    mask_da = BatchCore._mask_of_dataarray(mask_obj, ds, 'where')
                elif isinstance(mask_obj, xr.Dataset):
                    mgrids = BatchCore._mask_dataset(mask_obj)
                    names = BatchCore._grid_vars(ds)
                    BatchCore._mask_names(mgrids, names, 'where')
                    # every grid by the mask grid of its own name
                    new_ds = ds.copy()
                    for var in names:
                        mask_da = mask_obj[var]
                        extra_dims = set(mask_da.dims) - set(ds[var].dims)
                        if extra_dims:
                            raise ValueError(
                                f"where() mask has extra dimensions {extra_dims} not in data. "
                                f"Reduce the mask first, e.g. mask.mean() or mask.min() to collapse extra dims."
                            )
                        mask_da = mask_da.reindex_like(ds[var], method='nearest')
                        new_ds[var] = ds[var].where(mask_da, other, **kwargs)
                    out[k] = new_ds
                    continue
                elif isinstance(ds, xr.Dataset):
                    # a DataArray mask masks every grid of a Dataset
                    mask_da = BatchCore._mask_grid(mask_obj)
                else:
                    # a DataArray takes its mask directly, as in xarray (w['BPR'].where(...))
                    mask_da = mask_obj

                # Align mask to data coordinates (handles different x/y grids)
                # Get reference DataArray from ds for alignment (use spatial variable)
                if isinstance(ds, xr.Dataset):
                    ref_da = ds[BatchCore._grid_vars(ds)[0]]
                else:
                    ref_da = ds

                # Deny broadcasting: mask must not add dimensions to data
                extra_dims = set(mask_da.dims) - set(ref_da.dims)
                if extra_dims:
                    raise ValueError(
                        f"where() mask has extra dimensions {extra_dims} not in data. "
                        f"Reduce the mask first, e.g. mask.mean() or mask.min() to collapse extra dims."
                    )

                mask_da = mask_da.reindex_like(ref_da, method='nearest')

                if isinstance(ds, xr.Dataset):
                    # Mask ONLY spatial variables. Dataset.where broadcasts the
                    # (y, x) mask into EVERY variable, silently inflating
                    # non-spatial ones -- e.g. BPR (date,) becomes a
                    # (date, y, x) NaN cube, which then breaks pairs() when it
                    # assigns BPR as a per-pair coordinate.
                    new_ds = ds.copy()
                    for var in ds.data_vars:
                        if 'y' in ds[var].dims and 'x' in ds[var].dims:
                            new_ds[var] = ds[var].where(mask_da, other, **kwargs)
                    out[k] = new_ds
                else:
                    out[k] = ds.where(mask_da, other, **kwargs)
            return type(self)(out)

        # A PLAIN XARRAY MASK without (y, x) RAISES for a Batch of Datasets, as mask() raises
        # for it (N81, decided): their operators apply to the (y, x) grids only, and such a
        # mask is commonly a reduction, arr.mean(), that dropped y and x, and broadcast it is
        # very hard to find. A Batch of DataArrays takes it directly, as xarray does
        if not BatchCore._is_dataarray_batch(self):
            if isinstance(cond, xr.DataArray):
                BatchCore._mask_grid(cond)
            elif isinstance(cond, xr.Dataset):
                BatchCore._mask_dataset(cond)
        elif isinstance(cond, xr.Dataset):
            # a Dataset mask gives each DataArray its one variable, grid or not, as mask() takes
            # it: no name compared (_mask_of_dataarray); several variables RAISE
            for da_ in self.values():
                BatchCore._mask_of_dataarray(cond, da_, 'where')
            return self.map_da(lambda da: da.where(BatchCore._mask_of_dataarray(cond, da, 'where'),
                                                   other, **kwargs), **kwargs)
        # fallback: single scalar or DataArray broadcast
        # DataArray case seems not usefull because Batch datasets differ in shape
        return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def combine_first(self, other: 'BatchCore') -> 'BatchCore':
        """
        Combine two Batches, using values from self where valid, filling with other.

        For each pixel: use self's value if finite, otherwise use other's value.
        This is useful for merging results processed with different parameters
        (e.g., dense vs sparse regions) on the same grid.

        Parameters
        ----------
        other : BatchCore
            Batch to fill gaps from. Must have same keys and same grid as self.

        Returns
        -------
        BatchCore
            Combined result with same type as self.

        Raises
        ------
        ValueError
            If keys don't match or grids differ between self and other.

        Examples
        --------
        >>> # Process dense and sparse regions separately (same grid)
        >>> sim_dense = S_sparse.where(dense_mask['VV']).similarity(...)
        >>> sim_sparse = S_sparse.where(sparse_mask['VV']).similarity(...)
        >>> # Merge: use dense where available, fill with sparse
        >>> sim_merged = sim_dense.combine_first(sim_sparse)
        """
        if set(self.keys()) != set(other.keys()):
            raise ValueError(
                f"combine_first: keys must match. "
                f"self has {set(self.keys())}, other has {set(other.keys())}"
            )

        out = {}
        for k, ds_self in self.items():
            ds_other = other[k]

            if isinstance(ds_self, xr.Dataset):
                combined_vars = {}
                for var in ds_self.data_vars:
                    da_self = ds_self[var]
                    if var in ds_other.data_vars:
                        da_other = ds_other[var]
                        # Check grids match
                        if da_self.shape != da_other.shape:
                            raise ValueError(
                                f"combine_first: grid shapes must match for '{k}/{var}'. "
                                f"self has {da_self.shape}, other has {da_other.shape}"
                            )
                        # Use xarray's combine_first
                        combined_vars[var] = da_self.combine_first(da_other)
                    else:
                        combined_vars[var] = da_self
                out[k] = xr.Dataset(combined_vars, attrs=ds_self.attrs)
            else:
                # DataArray case
                if ds_self.shape != ds_other.shape:
                    raise ValueError(
                        f"combine_first: grid shapes must match for '{k}'. "
                        f"self has {ds_self.shape}, other has {ds_other.shape}"
                    )
                out[k] = ds_self.combine_first(ds_other)

        return type(self)(out)

    def mask(self, mask, other=np.nan):
        """
        Apply a mask to each burst.

        This is memory-efficient for large masks (e.g., landmask, DEM) as it
        reindexes the mask to each burst's coordinates using nearest-neighbor
        interpolation instead of broadcasting the full mask.

        Parameters
        ----------
        mask : Batch, xr.DataArray, xr.Dataset, or GeoDataFrame
            The mask to apply. Can be:
            - Batch with the same bursts (adi < 0.4): each burst by its own
              mask, by the rules below, as where() takes it
            - xr.DataArray: boolean (y, x) mask reindexed to each burst's
              coordinates; it masks every grid
            - xr.Dataset: its (y, x) variables mask the grids of their own
              names, as where() takes them, and a grid without one raises
              (mask it with a DataArray: x.mask(land['VV'])); the other
              variables play no part
            - GeoDataFrame: polygon(s) to mask by - pixels inside polygons are kept
            A Batch of DataArrays needs no (y, x) grid: it takes the mask
            directly, as where() does -- a Dataset mask's one variable, no
            name compared -- with or without (y, x).
        other : scalar, optional
            Value to use for masked elements. Default is np.nan.

        Returns
        -------
        Batch
            Masked batch with same type as self.

        Examples
        --------
        # Apply binary land mask, downloaded as tiles with a VRT index by Tiles().download_landmask(AOI, 'land.vrt')
        from insardev_toolkit import Tiles
        land = np.isfinite(Tiles().open('land.vrt').rio.reproject(intf.crs))
        masked_intf = intf.mask(land)

        # Mask by AOI polygon
        AOI = gpd.read_file('aoi.geojson')
        masked_velocity = velocity.mask(AOI)
        """
        import geopandas as gpd
        from shapely import Geometry
        import rioxarray

        # Handle GeoDataFrame/GeoSeries/Geometry masking
        if isinstance(mask, (gpd.GeoDataFrame, gpd.GeoSeries, Geometry)):
            # Extract geometry for rio.clip
            if isinstance(mask, gpd.GeoDataFrame):
                geom = mask.geometry
                mask_crs = mask.crs
            elif isinstance(mask, gpd.GeoSeries):
                geom = mask
                mask_crs = mask.crs
            else:
                # Shapely Geometry - no CRS info
                geom = [mask]
                mask_crs = None

            # Reproject to batch CRS if needed
            crs = self.crs
            if crs is not None and mask_crs is not None and mask_crs != crs:
                geom = gpd.GeoSeries(geom, crs=mask_crs).to_crs(crs)

            out = {}
            for key, ds in self.items():
                out[key] = ds.rio.clip(geom, all_touched=False)
            return type(self)(out)

        # A BATCH MASK (adi < 0.4), as where() takes it: each burst by the mask of the same
        # burst, under the rules below; a mask whose bursts are not the data's RAISES
        if isinstance(mask, BatchCore):
            lack = [k for k in self.keys() if not dict.__contains__(mask, k)]
            extra = [k for k in mask.keys() if not dict.__contains__(self, k)]
            if lack or extra:
                raise ValueError(f"ERROR: the mask has no burst {lack[0]}." if lack else
                                 f"ERROR: the mask has burst {extra[0]}, which the data lacks.")
            out = {}
            for key, ds in self.items():
                out[key] = dict.__getitem__(BatchCore.mask(type(self)({key: ds}), dict.__getitem__(mask, key),
                                                           other), key)
            return type(self)(out)

        # Handle xarray mask by THE MASK RULE (N81), as where() takes it: a Dataset
        # mask masks each grid by its grid of the same name, and a grid it lacks
        # raises; a DataArray mask masks every grid; the mask's other variables
        # play no part (its first variable was taken before, and a metadata
        # variable first raised "must be ... of boolean type" on a valid mask)
        # -- for a Batch of Datasets, whose mask needs (y, x). A BATCH OF DATAARRAYS NEEDS
        # NO GRID (N81): it takes its mask directly, as where() does, with or without (y, x),
        # a Dataset mask's one variable: a DataArray is one variable, no name is compared
        das = BatchCore._is_dataarray_batch(self)
        by_name = isinstance(mask, xr.Dataset)
        # the mask's variables, for the error that names what it lacks
        mask_vars = [str(v) for v in mask.data_vars] if by_name else []
        if by_name and das:
            # a DataArray takes the mask's one variable, no name compared (_mask_of_dataarray)
            masks = {ds.name: BatchCore._mask_of_dataarray(mask, ds, 'mask') for ds in self.values()}
        elif by_name:
            masks = {v: mask[v] for v in BatchCore._mask_dataset(mask)}
        else:
            masks = {None: mask if das else BatchCore._mask_grid(mask)}

        def _on(m, ds):
            """the mask on the burst's own y and x, as far as both have them"""
            dims = {d: ds[d] for d in ('y', 'x') if d in m.dims and d in ds.dims}
            return m.reindex(dims, method='nearest') if dims else m

        for m in masks.values():
            if not np.issubdtype(m.dtype, np.bool_):
                raise ValueError('Batch.mask: mask must be a Dataset or DataArray of boolean type, or a GeoDataFrame')

        # auto-chunk if not already chunked to avoid high memory usage
        masks = {v: (m if m.chunks else m.chunk('auto')) for v, m in masks.items()}
        mask = masks.get(None)

        out = {}
        for key, ds in self.items():
            if by_name:
                if isinstance(ds, xr.DataArray):
                    out[key] = ds.where(_on(masks[ds.name], ds), other)
                    continue
                names = BatchCore._grid_vars(ds)
                BatchCore._mask_names(list(masks) or mask_vars or ['name'], names, 'mask')
                masked = ds.copy()
                for v in names:
                    masked[v] = ds[v].where(masks[v].reindex(y=ds.y, x=ds.x, method='nearest'), other)
                out[key] = masked
                continue
            if isinstance(ds, xr.DataArray):
                out[key] = ds.where(_on(mask, ds), other)
                continue
            # the fastest way to align mask to burst coordinates
            mask_burst = mask.reindex(y=ds.y, x=ds.x, method='nearest')
            # THE GRIDS ONLY, as downsample() does it. Dataset.where() broadcasts
            # EVERY variable against the mask's (y, x), so the 1-D radar metadata
            # that rides along -- `burst` is a STRING -- comes back as a (y, x)
            # raster of strings. Nothing downstream survives that: plot() takes
            # any variable ending in (y, x) for a polarization and dies on it
            # with "can only concatenate str to str", and a mask says nothing
            # about the burst a grid was measured in anyway.
            _grids = BatchCore._grid_vars(ds)
            _meta = [v for v in ds.data_vars if v not in _grids]
            # preserve original chunking structure for lazy computation
            masked = ds[_grids].where(mask_burst, other)
            out[key] = masked.assign({v: ds[v] for v in _meta}) if _meta else masked
        return type(self)(out)



    @staticmethod
    def _resolve_stride_static(stride, win_y, win_x):
        if stride is None:
            return max(1, win_y // 2), max(1, win_x // 2)
        if isinstance(stride, (tuple, list)):
            return max(1, int(stride[0])), max(1, int(stride[1]))
        return max(1, int(stride)), max(1, int(stride))


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

        Examples
        --------
        >>> disp = (stack.chunk1d('0.5GB').fit3d()
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
                if len(da_xr.data.chunks[0]) != 1:
                    raise ValueError(
                        "unwrap3d() requires the date dimension in one chunk. "
                        "Use chunk1d() first.")

                def _make(_d, _bd, _mi, _se, _sd, _ns, _mdis, _mdt, _ntr):
                    def kernel(block):
                        return utils_unwrap3d.unwrap3d_dates_array(
                            block, _d, duration=_bd, max_iter=_mi,
                            search=_se, short_days=_sd, n_short=_ns,
                            min_disagree=_mdis, min_dates=_mdt,
                            n_trend=_ntr)
                    return kernel

                out = da.map_blocks(
                    _make(date_values, float(duration), int(max_iter),
                          int(search), float(short_days),
                          int(n_short), int(min_disagree), int(min_dates),
                          int(n_trend)),
                    da_xr.data, dtype=np.float32,
                    meta=np.empty((0, 0, 0), dtype=np.float32))
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
        return Batch(result)



    # Backward compatibility alias
    def neighbors(
        self,
        window: 'float | tuple' = 40,
        neighbors: tuple | None = None,
        device: str = 'auto'
    ) -> 'Batch':
        """
        Count valid (non-NaN) neighbors per pixel within a spatial window.

        Works on any 2D spatial data (y, x). For each pixel, counts how many
        neighbors in the window have finite values.

        Parameters
        ----------
        window : float or tuple of float
            Window size in METRES on the ground, one number for a square
            or (y, x). Rounded up to an odd number of pixels per axis, since
            the count is centred on a pixel.
        neighbors : tuple of int or None
            If provided, filter output: (min, max)
            - Pixels with count < min: set to NaN
            - Pixels with count > max: clipped to max
            If None, return raw counts.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'

        Returns
        -------
        Batch
            Valid neighbor count per pixel (float, NaN at borders)

        Examples
        --------
        >>> # Count valid neighbours of a sparse selection
        >>> adi = S_opt.adi()
        >>> sparse = adi.where(adi < 0.5)
        >>> nbrs = sparse.neighbors(window=120)
        >>> dense_mask = nbrs >= 10
        """
        import torch
        import dask.array as da

        # METRES IN, PIXELS PER BURST: the window is a ground size, and the
        # count it becomes is rounded UP to odd, because the kernel is centred
        # on a pixel and a metre request cannot choose parity.
        from . import utils_xarray
        import functools

        # Resolve device once and convert to string for clean serialization
        resolved = BatchCore._get_torch_device(device)
        device_str = resolved.type  # 'cpu', 'cuda', or 'mps'

        results = {}

        for burst_id, ds in self.items():
            count_vars = {}
            window_y, window_x = utils_xarray.meters_to_pixels(
                window, utils_xarray.spacing_of(ds), minimum=3, odd=True,
                name='neighbors() window')
            half_y, half_x = window_y // 2, window_x // 2

            # Validate neighbors if provided, against the count this grid gives
            if neighbors is not None:
                neighbors_min, neighbors_max = neighbors
                max_possible = window_y * window_x - 1
                if neighbors_max > max_possible:
                    raise ValueError(
                        f"neighbors max={neighbors_max} exceeds maximum for window ({window_y}, {window_x}): {max_possible}"
                    )
                if neighbors_min > neighbors_max:
                    raise ValueError(
                        f"neighbors min={neighbors_min} cannot exceed max={neighbors_max}"
                    )

            # Use functools.partial with module-level function to avoid closure
            # Closures capturing variables can cause memory explosions in dask workers
            neighbors_func = functools.partial(
                _neighbors_kernel_2d_for_dask,
                window_y=window_y,
                window_x=window_x,
                half_y=half_y,
                half_x=half_x,
                device=device_str
            )

            # a Dataset's variables, or a DataArray as its own one (N81)
            for var_name, data in BatchCore._vars_of(ds).items():
                # Skip non-spatial variables
                if 'y' not in data.dims or 'x' not in data.dims:
                    continue

                # Handle 2D data only
                if data.ndim != 2:
                    continue

                # Use map_overlap on input chunks as-is
                count_da = da.map_overlap(
                    neighbors_func,
                    data.data,
                    depth={0: half_y, 1: half_x},
                    boundary='none',
                    trim=True,
                    dtype=np.float32,
                )

                # Apply neighbors filtering if provided
                if neighbors is not None:
                    neighbors_min, neighbors_max = neighbors
                    count_da = da.clip(count_da, 0, neighbors_max)
                    count_da = da.where(count_da >= neighbors_min, count_da, np.nan)

                count_xr = xr.DataArray(
                    count_da,
                    dims=['y', 'x'],
                    coords={'y': data.y, 'x': data.x},
                    name=var_name
                )

                count_vars[var_name] = count_xr

            if count_vars:
                # a DataArray in, a DataArray out (N81)
                results[burst_id] = BatchCore._form(ds, count_vars)

        from .Batch import Batch
        return Batch(results)

    def crop(self, geometry):
        """
        Crop each burst to the bounding rectangle of a geometry.

        Unlike mask() which clips to the exact geometry shape, crop() selects
        the bounding box (rectangular extent) of the geometry.

        Parameters
        ----------
        geometry : GeoDataFrame, GeoSeries, or Shapely Geometry
            Geometry whose bounding box defines the crop extent.

        Returns
        -------
        Batch
            Cropped batch with same type as self.

        Examples
        --------
        # Crop to AOI bounding box
        cropped = velocity.crop(AOI.buffer(500))

        # Compare with mask (exact geometry)
        masked = velocity.mask(AOI.buffer(500))  # clips to exact shape
        cropped = velocity.crop(AOI.buffer(500))  # crops to bounding rectangle
        """
        import geopandas as gpd
        from shapely import Geometry

        # Extract bounds from geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            bounds = geometry.total_bounds  # (minx, miny, maxx, maxy)
            geom_crs = geometry.crs
        elif isinstance(geometry, gpd.GeoSeries):
            bounds = geometry.total_bounds
            geom_crs = geometry.crs
        elif isinstance(geometry, Geometry):
            bounds = geometry.bounds  # (minx, miny, maxx, maxy)
            geom_crs = None
        else:
            raise TypeError(f"geometry must be GeoDataFrame, GeoSeries, or Shapely Geometry, got {type(geometry).__name__}")

        minx, miny, maxx, maxy = bounds

        # Reproject bounds to batch CRS if needed
        crs = self.crs
        if crs is not None and geom_crs is not None and geom_crs != crs:
            from shapely.geometry import box
            bbox = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=geom_crs).to_crs(crs)
            minx, miny, maxx, maxy = bbox.total_bounds

        out = {}
        for key, ds in self.items():
            # Determine coordinate order (ascending or descending)
            y_asc = len(ds.y) < 2 or float(ds.y[1]) > float(ds.y[0])
            x_asc = len(ds.x) < 2 or float(ds.x[1]) > float(ds.x[0])

            # Create slices based on coordinate order
            y_slice = slice(miny, maxy) if y_asc else slice(maxy, miny)
            x_slice = slice(minx, maxx) if x_asc else slice(maxx, minx)

            clipped = ds.sel(y=y_slice, x=x_slice)
            if clipped.y.size > 0 and clipped.x.size > 0:
                out[key] = clipped

        return type(self)(out)

    def __pow__(self, exponent, **kwargs):
        return self.map_da(lambda da: da**exponent, **kwargs)

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        return self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs)

    # def abs(self):
    #     """ element-wise absolute value """
    #     return type(self)({k: ds.map(lambda da: da.abs()) for k, ds in self.items()})

    # def sqrt(self):
    #     """ element-wise square-root """
    #     return type(self)({k: ds.map(lambda da: da.sqrt()) for k, ds in self.items()})

    # def square(self):
    #     """ element-wise square """
    #     return type(self)({k: ds.map(lambda da: da**2) for k, ds in self.items()})

    # def clip(self, min_, max_):
    #     """ element-wise clip to [min_, max_] """
    #     return type(self)({k: ds.map(lambda da: da.clip(min_, max_)) for k, ds in self.items()})

    # def where(self, cond, other=np.nan):
    #     """
    #     like xarray.where: keep ds where cond is True, else fill with other.
    #     `cond` may be a scalar, a DataArray, or another Batch with the same keys.
    #     """
    #     if isinstance(cond, Batch):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return type(self)({
    #             k: ds.where(cond, other)
    #             for k, ds in self.items()
    #         })

    # def isfinite(self):
    #     """ element-wise finite mask """
    #     return type(self)({k: ds.map(lambda da: np.isfinite(da)) for k, ds in self.items()})

    # def sel(self, keys: dict|list|str):
    #     if isinstance(keys, str):
    #         keys = [keys]
    #     return type(self)({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

    @staticmethod
    def _burst_indexer(value, key, dim, func='sel'):
        """One burst's indexer out of a per-burst object.

        An indexer's value may be a Batch -- what `batch.coherence > 0.5` or
        any other batch expression returns -- and then EACH BURST IS INDEXED
        BY ITS OWN: the mask a burst was measured on is the mask it is
        selected by. A burst carrying a different number of dates, or a
        different verdict on the same date, is the normal case, and one
        shared mask cannot express it. Anything else is passed through
        untouched, the same indexer for every burst.
        """
        if not isinstance(value, BatchCore):
            return value
        if key not in value:
            raise KeyError(
                f"{func}(): the '{dim}' indexer has no burst '{key}'.")
        v = value[key]
        if isinstance(v, xr.Dataset):
            names = list(v.data_vars)
            if len(names) != 1:
                raise ValueError(
                    f"{func}(): the '{dim}' indexer of burst '{key}' carries "
                    f"{len(names)} variables {names}, it must carry one.")
            v = v[names[0]]
        return v

    @staticmethod
    def _drop_empty(mapping):
        """Bursts left holding no samples are DROPPED, not returned empty.

        A selection that misses a burst says so by the burst's ABSENCE: a
        batch whose count changed is visible at a glance, where a burst
        carrying a zero-length dimension reads as data right up until
        something computes over its pixels -- save(), coarsen(), chunk2d()
        and mask() all raise on one. crop() has always dropped them.

        A dimension indexed down to nothing is the only way a selector makes
        one, so this never removes a burst the caller still holds data for.
        """
        return {k: ds for k, ds in mapping.items()
                if not any(n == 0 for n in getattr(ds, 'sizes', {}).values())}

    def sel(self, keys: dict|list|str|pd.DataFrame|None = None, **indexers):
        """
        Select data by burst keys or coordinate values.

        Parameters
        ----------
        keys : str, list, dict, DataFrame, or None
            - str: Single burst key to select
            - list: List of burst keys to select
            - dict/Batch: Align dimensions between batches
            - DataFrame: Complex filtering by dates/polarizations
            - None: Use only keyword indexers
        **indexers : slice, value, mask or Batch
            Coordinate-based selection applied to each dataset.
            Example: x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000)
            A Batch indexes EACH BURST BY ITS OWN, so a per-burst boolean
            mask selects per burst: date=trend.coherence > 0.5

        Returns
        -------
        Batch
            New Batch with selected data. A burst whose selection came back
            EMPTY IS DROPPED, so the batch's own count reports what the
            window reached, as crop() does.

        Examples
        --------
        Select by burst keys:
        >>> subset = batch.sel(['burst1', 'burst2'])

        Select by spatial coordinates:
        >>> subset = batch.sel(x=slice(650_000, 700_000))
        >>> subset = batch.sel(x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000))

        Combine both:
        >>> subset = batch.sel(['burst1'], x=slice(650_000, 700_000))

        Keep the dates a per-burst mask marks, each burst by its own:
        >>> trend = trend.sel(date=trend.coherence > 0.5)
        """
        import pandas as pd
        import numpy as np

        # Handle coordinate-based selection via keyword indexers
        if indexers:
            result = self if keys is None else self
            # First apply key selection if provided
            if keys is not None:
                result = result.sel(keys)

            # Convert slices to index-based selection (fast, order-agnostic)
            def select_with_slices(ds, indexers, key):
                for dim, idx in indexers.items():
                    if dim not in ds.coords:
                        continue
                    idx = self._burst_indexer(idx, key, dim)
                    if isinstance(idx, slice):
                        coord_vals = ds.coords[dim].values
                        # Get bounds from slice, use coord min/max as defaults
                        start = idx.start if idx.start is not None else coord_vals.min()
                        stop = idx.stop if idx.stop is not None else coord_vals.max()
                        min_val, max_val = min(start, stop), max(start, stop)
                        # Find indices within range (order-agnostic)
                        mask = (coord_vals >= min_val) & (coord_vals <= max_val)
                        indices = np.where(mask)[0]
                        if len(indices) > 0:
                            # Use isel with index slice (fast)
                            ds = ds.isel({dim: slice(indices[0], indices[-1] + 1)})
                        else:
                            # No matching coordinates - return empty
                            ds = ds.isel({dim: slice(0, 0)})
                    else:
                        # Non-slice indexer (exact value, list, etc.)
                        ds = ds.sel({dim: idx})
                return ds

            # EVERY BURST IS SELECTED ON ITS OWN COORDINATES, and nothing
            # here reindexes one onto another's grid. Bursts carry independent
            # extents and independent lattice phases: one overlapping the
            # window part way returns that part, and one that does not reach
            # the window returns the EMPTY selection -- its own coordinates,
            # zero samples. Reindexing it onto whichever burst happened to be
            # selected first invented a raster at coordinates that burst never
            # had, and an all-NaN raster reads downstream like observed
            # no-data rather than like nothing at all.
            return type(result)(self._drop_empty(
                {k: select_with_slices(ds, indexers, k)
                 for k, ds in result.items()}))

        # Original key-based selection logic
        if keys is None:
            return self

        if not isinstance(keys, pd.DataFrame):
            if isinstance(keys, str):
                keys = [keys]
            if isinstance(keys, list):
                return type(self)({k: self[k] for k in keys})

            # keys is dict-like (e.g., BatchWrap, BatchUnit)
            # Select matching burst IDs and align dimensions (like 'pair') per key
            result = {}
            for k in keys.keys():
                if k not in self:
                    continue
                ds = self[k]
                other_ds = keys[k]

                # Align 'pair' dimension if both have it - use minimum size (positional indexing)
                if hasattr(other_ds, 'dims') and 'pair' in getattr(other_ds, 'dims', []):
                    if hasattr(ds, 'dims') and 'pair' in ds.dims:
                        n_pairs = min(ds.sizes['pair'], other_ds.sizes['pair'])
                        if n_pairs < ds.sizes['pair']:
                            ds = ds.isel(pair=slice(n_pairs))

                result[k] = ds
            return type(self)(result)

        dss = {}
        # iterate all burst groups (fullBurstID is the first index level)
        for id in keys.index.get_level_values(0).unique():
            if id not in self:
                continue
            # select all records for the current burst group
            records = keys[keys.index.get_level_values(0)==id]
            ds = self[id]
            
            # Detect dimension type: date for Stack-like, pair for Batch-like
            if 'date' in ds.dims:
                # Stack-like: filter by dates
                dates = records.startTime.values.astype(str)
                ds = ds.sel(date=dates)
            # For pair-based data, we just select the burst if it exists
            # (pair filtering is handled elsewhere or not needed for simple selection)
            
            # filter polarizations
            pols = records.polarization.unique()
            if len(pols) > 1:
                raise ValueError(f'ERROR: Inconsistent polarizations found for the same burst: {id}')
            elif len(pols) == 0:
                raise ValueError(f'ERROR: No polarizations found for the burst: {id}')
            pols = pols[0]
            if ',' in pols:
                pols = pols.split(',')
            if isinstance(pols, str):
                pols = [pols]
            count = 0
            if np.unique(pols).size < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not unique.')
            if len([pol for pol in pols if pol in ds.data_vars]) < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not available in the dataset: {id}')
            for pol in [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in ds.data_vars]:
                if pol not in pols:
                    ds = ds.drop(pol)
                else:
                    count += 1
            if count == 0:
                raise ValueError(f'ERROR: No valid polarizations found for the burst: {id}')
            dss[id] = ds
        return type(self)(dss)

    # def isel(self, indices):
    #     """Select by integer locations (like xarray .isel)."""
    #     import numpy as np

    #     keys = list(self.keys())
    #     # allow a single integer, a list of ints, or a slice
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     selected = {keys[i]: self[keys[i]] for i in idxs }
    #     return type(self)(selected)

    # def isel(self, indices=None, **indexers):
    #     """
    #     Select by integer locations, either by a single positional index/slice
    #     (applied over the *keys* of the batch) OR by keyword dimension selectors
    #     (delegated to each xarray.Dataset.isel).
    #     """
    #     # xarray‐style keyword isel
    #     if indexers:
    #         return type(self)({
    #             k: ds.isel(**indexers)
    #             for k, ds in self.items()
    #         })

    #     # positional isel over the batch keys (old behavior)
    #     import numpy as np
    #     keys = list(self.keys())
    #     if indices is None:
    #         return type(self)(dict(self))  # no selection
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     return type(self)({
    #         keys[i]: self[keys[i]]
    #         for i in idxs
    #     })

    def isel(self, indices=None, **indexers):
        """
        Select by integer locations, either by:
        keyword dimension selectors (delegated to each xarray.Dataset.isel)
        a single positional index/slice/list over the *keys* of the batch
        (NEW) a single dict positional argument of dimension indexers

        A burst indexed down to nothing is DROPPED, not returned empty.
        """
        import numpy as np

        # dict as a keyword indexers
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray‐style keyword isel (including dict-via-positional)
        if indexers:
            return type(self)(self._drop_empty({
                k: ds.isel(**{d: self._burst_indexer(v, k, d, 'isel')
                              for d, v in indexers.items()})
                for k, ds in self.items()
            }))

        # fallback: positional isel over the batch keys (old behavior)
        keys = list(self.keys())
        if indices is None:
            # no selection, cast to dict to prevent special logic in the class constructor
            return type(self)(dict(self))
        if isinstance(indices, (int, np.integer)):
            idxs = [indices]
        elif isinstance(indices, slice):
            idxs = list(range(*indices.indices(len(keys))))
        else:
            idxs = list(indices)

        return type(self)({
            keys[i]: self[keys[i]]
            for i in idxs
        })

    def drop_sel(self, keys: dict|list|str|None = None, errors='raise', **indexers):
        """
        Drop bursts by key and/or drop coordinate labels from every dataset.

        The inverse of sel(): burst keys listed here are removed from the batch,
        and dimension indexers are passed to each xarray.Dataset.drop_sel(), which
        drops the named labels (not slices) along that dimension.

        Parameters
        ----------
        keys : str, list, dict, or None
            - str: Single burst key to drop
            - list: Burst keys to drop
            - dict/Batch: Drop the bursts named by its keys
            - None: Use only keyword indexers
        errors : {'raise', 'ignore'}
            'raise' reports burst keys or coordinate labels that are not present,
            'ignore' silently skips them.
        **indexers : label, list of labels, mask or Batch
            Coordinate labels dropped from each dataset.
            Example: date=['2021-01-01'], pair=[('2021-01-01', '2021-01-13')]
            A boolean mask marks WHAT TO DROP, and a Batch of masks drops
            per burst: date=trend.coherence < 0.5

        Returns
        -------
        Batch
            New Batch without the dropped bursts and labels. A burst left
            holding nothing is dropped too.

        Examples
        --------
        Drop bursts by key:
        >>> subset = batch.drop_sel('burst1')
        >>> subset = batch.drop_sel(['burst1', 'burst2'])

        Drop dates from every burst:
        >>> subset = stack.drop_sel(date=['2021-01-01', '2021-01-13'])

        Combine both:
        >>> subset = stack.drop_sel('burst1', date='2021-01-01')

        Drop the dates a per-burst mask marks, each burst by its own:
        >>> trend = trend.drop_sel(date=trend.coherence < 0.5)
        """
        if keys is None and not indexers:
            # no selection, cast to dict to prevent special logic in the class constructor
            return type(self)(dict(self))

        result = self
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, Mapping):
                keys = list(keys.keys())
            keys = list(keys)
            if errors == 'raise':
                missing = [k for k in keys if k not in result]
                if missing:
                    raise KeyError(f'ERROR: bursts are not available in the batch: {missing}')
            dropped = set(keys)
            result = type(self)({k: ds for k, ds in result.items() if k not in dropped})

        if indexers:
            out = {}
            for k, ds in result.items():
                idx = {}
                for dim, v in indexers.items():
                    v = self._burst_indexer(v, k, dim, 'drop_sel')
                    _v = np.asarray(v.values if isinstance(v, (xr.DataArray, xr.Variable)) else v)
                    if _v.dtype == bool and _v.ndim == 1:
                        # a MASK MARKS WHAT TO DROP here, the mirror of sel():
                        # the labels it stands for, so xarray sees labels
                        v = np.asarray(ds.coords[dim].values)[_v]
                    idx[dim] = v
                out[k] = ds.drop_sel(idx, errors=errors)
            result = type(self)(self._drop_empty(out))

        return result

    def drop_isel(self, indices=None, **indexers):
        """
        Drop by integer locations, either by:
        keyword dimension indexers (delegated to each xarray.Dataset.drop_isel)
        a single positional index/slice/list over the *keys* of the batch
        a single dict positional argument of dimension indexers

        Like isel(), dimension indexers take precedence: when they are given the
        positional argument is not applied to the batch keys. A burst left
        holding nothing is dropped.

        Examples
        --------
        Drop bursts by position:
        >>> subset = batch.drop_isel(0)
        >>> subset = batch.drop_isel([0, -1])
        >>> subset = batch.drop_isel(slice(2, None))

        Drop dates from every burst:
        >>> subset = stack.drop_isel(date=[0, 1])

        A boolean mask marks what to drop, a Batch of masks drops per burst:
        >>> trend = trend.drop_isel(date=trend.coherence < 0.5)
        """
        import numpy as np

        # dict as a keyword indexers
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray-style keyword drop_isel (including dict-via-positional)
        if indexers:
            out = {}
            for k, ds in self.items():
                idx = {}
                for dim, v in indexers.items():
                    v = self._burst_indexer(v, k, dim, 'drop_isel')
                    _v = np.asarray(v.values if isinstance(v, (xr.DataArray, xr.Variable)) else v)
                    if _v.dtype == bool and _v.ndim == 1:
                        # a MASK MARKS WHAT TO DROP: the positions it stands for
                        v = np.flatnonzero(_v)
                    idx[dim] = v
                out[k] = ds.drop_isel(**idx)
            return type(self)(self._drop_empty(out))

        # fallback: positional drop over the batch keys
        keys = list(self.keys())
        if indices is None:
            # no selection, cast to dict to prevent special logic in the class constructor
            return type(self)(dict(self))
        if isinstance(indices, (int, np.integer)):
            idxs = [indices]
        elif isinstance(indices, slice):
            idxs = list(range(*indices.indices(len(keys))))
        else:
            idxs = list(indices)

        dropped = {keys[i] for i in idxs}
        return type(self)({
            k: ds for k, ds in self.items() if k not in dropped
        })

    @property
    def dims(self):
        return {k: self[k].dims for k in self.keys()}
    
    @property
    def coords(self):
        """Return a Batch of Coordinates for each dataset."""
        return type(self)({k: ds.coords.to_dataset() for k, ds in self.items()})

    def assign_coords(self, coords=None, **coords_kwargs):
        """
        Assign new coordinates to each dataset in the batch.
        Works like xarray.Dataset.assign_coords but handles batch operations.
        
        Parameters
        ----------
        coords : dict-like or Batch, optional
            Dictionary of coordinates to assign or Batch of coordinates
        **coords_kwargs : optional
            Coordinates to assign, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with assigned coordinates
        """
        if coords is None:
            coords = {}
        coords = dict(coords, **coords_kwargs)
        
        # Check if any coord is a BatchCore - if so, we need per-burst assignment
        batch_coords = {name: coord for name, coord in coords.items() 
                       if isinstance(coord, tuple) and len(coord) == 2 
                       and isinstance(coord[1], BatchCore)}
        
        if batch_coords:
            # Per-burst coordinate assignment
            result = {}
            for key, ds in self.items():
                ds_coords = {}
                for name, coord in coords.items():
                    if name in batch_coords:
                        dims, batch = coord
                        # Get this burst's values from the batch: its first variable, or a
                        # DataArray burst itself (N81)
                        data = next(iter(BatchCore._vars_of(batch[key]).values()))
                        # Compute lazy arrays - coordinates should never be lazy
                        values = data.compute().values if hasattr(data.data, 'compute') else data.values
                        ds_coords[name] = (dims, values)
                    else:
                        ds_coords[name] = coord
                result[key] = ds.assign_coords(ds_coords)
            return type(self)(result)
        
        def process_coord(coord):
            if not isinstance(coord, tuple) or len(coord) != 2:
                return coord
                
            dims, data = coord
            
            # Handle DataArray directly
            if isinstance(data, xr.DataArray):
                values = data.values
                return xr.DataArray(values if data.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle BatchComplex
            if isinstance(data, type(self)):
                first_ds = next(iter(data.values()))
                if isinstance(first_ds, xr.DataArray):
                    values = first_ds.values
                    return xr.DataArray(values if first_ds.ndim > 0 else np.array([values]), dims=dims)
                elif isinstance(first_ds, xr.Dataset):
                    coord_name = first_ds.dims[0]
                    values = first_ds.coords[coord_name].values
                    return xr.DataArray(values if not np.isscalar(values) else np.array([values]), dims=dims)
            
            # Handle objects with values attribute
            if hasattr(data, 'values'):
                values = data.values() if callable(data.values) else data.values
                if hasattr(values, '__iter__'):
                    values = next(iter(values))
                    if isinstance(values, xr.DataArray):
                        values = values.values
                values = np.asarray(values)
                return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle array-like inputs
            values = np.asarray(data)
            return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
        
        # Get target dimension size from first dataset
        first_ds = next(iter(self.values()))
        target_size = first_ds.dims[list(coords.values())[0][0]]
        
        # Process coordinates
        processed_coords = {name: process_coord(coord) for name, coord in coords.items()}
        
        # Ensure consistent dimension sizes
        for name, coord in processed_coords.items():
            if coord.size != target_size:
                if coord.size == 1 and target_size == 2:
                    processed_coords[name] = xr.DataArray([coord.values[0], coord.values[0]], dims=coord.dims)
                else:
                    raise ValueError(f"Coordinate {name} has size {coord.size} but expected size {target_size}")
        
        return type(self)({
            k: ds.assign_coords(processed_coords)
            for k, ds in self.items()
        })

    def set_index(self, indexes=None, **indexes_kwargs):
        """
        Set Dataset index(es) for each dataset in the batch.
        Works like xarray.Dataset.set_index but handles batch operations.
        
        Parameters
        ----------
        indexes : dict-like or Batch, optional
            Dictionary of indexes to set or Batch of indexes
        **indexes_kwargs : optional
            Indexes to set, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with set indexes
        """
        if indexes is None:
            indexes = {}
        indexes = dict(indexes, **indexes_kwargs)
        
        # Handle both dict and Batch inputs
        if isinstance(indexes, type(self)):
            return type(self)({
                k: ds.set_index(indexes[k])
                for k, ds in self.items()
                if k in indexes
            })
        else:
            return type(self)({
                k: ds.set_index(indexes)
                for k, ds in self.items()
            })

    def expand_dims(self, *args, **kw):
        return type(self)({k: ds.expand_dims(*args, **kw) for k, ds in self.items()})

    @staticmethod
    def _assign_value(name, value, key, ds):
        """Resolve one assign() value for one burst."""
        if callable(value):
            value = value(ds)
        # a Batch (or any mapping keyed by burst id) carries a different value
        # per burst -- that is what batch arithmetic returns
        if isinstance(value, Mapping):
            if key not in value:
                raise KeyError(f"assign(): value for '{name}' has no entry for burst '{key}'")
            value = value[key]
        if isinstance(value, xr.Dataset):
            # BATCH ARITHMETIC RETURNS DATASETS, NOT DATAARRAYS. Their single
            # variable is named for the source (a polarization, or the variable
            # the caller selected), never for the assignment target, so unwrap
            # it here; assigning the Dataset itself would add its own name
            # instead of `name`.
            names = list(value.data_vars)
            if name in names:
                value = value[name]
            elif len(names) == 1:
                value = value[names[0]]
            else:
                raise ValueError(
                    f"assign(): value for '{name}' has {len(names)} variables {names} "
                    f"for burst '{key}'; select one, e.g. batch[['{names[0]}']]"
                )
        return value

    def assign(self, variables=None, **variables_kwargs):
        """
        Add or replace data variables in every burst, returning a new Batch.

        Works like xarray.Dataset.assign, with one addition: a value may be a
        Batch (or any mapping keyed by burst id), and then each burst takes its
        own entry from it. Batch arithmetic returns exactly that, so the result
        of an expression over this Batch can be assigned straight back.

        The original Batch is not modified, and neither are its Datasets.

        Parameters
        ----------
        variables : dict-like, optional
            Mapping of variable name to value. A value may be a Batch or
            mapping keyed by burst id, an xr.Dataset holding a single variable,
            an xr.DataArray, a (dims, data) tuple, a scalar, or a callable
            invoked with each burst's Dataset.
        **variables_kwargs : optional
            The same, as keyword arguments. These take precedence over
            `variables` on a name collision, as in xarray.

        Returns
        -------
        BatchCore
            A new Batch of the same class, with the variables added or
            replaced. Names already present are overwritten.

        Examples
        --------
        >>> model = model.assign(velocity=model[['velocity']] - DATUM_OFFSET)
        >>> model = model.assign(velocity=velocity_los['velocity'], rmse=rmse)
        >>> model = model.assign(velocity_mm=lambda ds: -4.4138 * ds.velocity)
        """
        if variables is None:
            variables = {}
        variables = dict(variables, **variables_kwargs)
        if not variables:
            return type(self)(dict(self))

        return type(self)({
            key: ds.assign({name: self._assign_value(name, value, key, ds)
                            for name, value in variables.items()})
            for key, ds in self.items()
        })

    def drop_vars(self, names, errors='raise'):
        """Return a new Batch with those data-vars removed from each dataset."""
        if isinstance(names, str):
            names = [names]
        return type(self)({
            k: ds.drop_vars(names, errors=errors)
            for k, ds in self.items()
        })

    def rename_vars(self, **kw):
        return type(self)({k: ds.rename_vars(**kw) for k, ds in self.items()})
    
    def rename(self, *args, **kw):
        """xarray's rename per burst; a Batch of DataArrays takes a new name, x.rename('phase')."""
        return type(self)({k: ds.rename(*args, **kw) for k, ds in self.items()})

    def merge(self, other: 'BatchCore') -> 'Batch':
        """Merge variables from another Batch into this one (per burst xr.merge).

        Both Batches must share the same burst keys and compatible coordinates.
        Use rename() first to avoid variable name conflicts.
        Always returns Batch (plain real-valued) since mixed types should not
        support specialized operations (e.g., wrapped phase arithmetic).

        Parameters
        ----------
        other : BatchCore
            Batch with additional variables to merge.

        Returns
        -------
        Batch
            Merged batch containing variables from both.

        Examples
        --------
        >>> corr_mean = mcorr.mean('pair').rename(VV='VV_cor')
        >>> combined = corr_mean.merge(rmse_mm.rename(VV='VV_rmse_mm'))
        >>> df = combined.to_dataframe()
        """
        import xarray as xr
        from .Batch import Batch
        # 'no_conflicts' STATED, and deliberately not the 'override' xarray is
        # moving to. This merges two products a caller built separately, so a
        # variable carrying the same name in both is a mistake worth hearing
        # about -- 'override' would silently keep the first and drop the other.
        return Batch({k: xr.merge([ds, other[k]], compat='no_conflicts')
                      for k, ds in self.items() if k in other})

    def reindex(self, **kw):
        return type(self)({k: ds.reindex(**kw) for k, ds in self.items()})

    def interp(self, **kw):
        return type(self)({k: ds.interp(**kw) for k, ds in self.items()})

    def interp_like(self, other: Batch, **interp_kwargs):
        """Regrid each Dataset onto the coords of the *corresponding* Dataset in `other`."""
        return type(self)({k: ds.interp_like(other[k], **interp_kwargs) for k, ds in self.items() if k in other})

    def reindex_like(self, other: Batch, **reindex_kwargs):
        return type(self)({k: ds.reindex_like(other[k], **reindex_kwargs) for k, ds in self.items() if k in other})

    def transpose(self, *dims, **kw):
        return type(self)({k: ds.transpose(*dims, **kw) for k, ds in self.items()})

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Internal helper for aggregation methods.
        If the target object's .<name>() accepts a `dim=` arg, we pass dim, otherwise we just call it without.
        """
        import inspect
        out = {}
        for key, obj in self.items():
            # THE OPERATOR RULE: the grids are reduced, every other variable
            # passes through unchanged -- a per-pair BPR keeps its pairs after
            # mean('pair'), and a burst id is not averaged
            carry = {}
            src = obj
            if isinstance(obj, xr.Dataset):
                grids = BatchCore._grid_vars(obj)
                carry = {v: obj[v] for v in obj.data_vars if v not in grids}
                src = obj[grids]
            fn = getattr(src, name)
            sig = inspect.signature(fn)
            if "dim" in sig.parameters:
                out[key] = fn(dim=dim, **kwargs)
            else:
                out[key] = fn(**kwargs)
            for v, da_ in carry.items():
                out[key][v] = da_
            # Preserve attrs (xarray aggregations drop them by default)
            if hasattr(obj, 'attrs') and hasattr(out[key], 'attrs'):
                out[key].attrs = obj.attrs

        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

    def mean(self, dim=None, **kwargs):
        return self._agg("mean", dim=dim, **kwargs)

    def sum(self, dim=None, **kwargs):
        return self._agg("sum", dim=dim, **kwargs)

    def min(self, dim=None, **kwargs):
        return self._agg("min", dim=dim, **kwargs)

    def max(self, dim=None, **kwargs):
        return self._agg("max", dim=dim, **kwargs)

    def median(self, dim=None, **kwargs):
        return self._agg("median", dim=dim, **kwargs)

    def std(self, dim=None, **kwargs):
        return self._agg("std", dim=dim, **kwargs)

    def var(self, dim=None, **kwargs):
        return self._agg("var", dim=dim, **kwargs)

    def rmse(self, solution, weight=None):
        """RMSE: self (pairs/dates) vs solution (pairs, dates, or velocity).

        A Batch of Datasets compares its (y, x) grids; a Batch of DataArrays
        is compared as it is, with or without (y, x), broadcast against the
        solution's grid as xarray broadcasts it. The solution needs a (y, x)
        grid: it gives the form and the pixels of the result.

        Parameters
        ----------
        solution : Batch or BatchWrap
            If pair-based: direct comparison.
            If date-based: pairs reconstructed as sol[rep] - sol[ref].
            If spatial-only (y, x): interpreted as yearly rate and
                reconstructed per pair (velocity * dt_years) or per date.
        weight : BatchUnit, optional
            Per-pair/date weights (e.g., correlation).

        Returns
        -------
        Batch
            Per-pixel RMSE on (y, x) grid.
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        from .Batch import Batch, BatchComplex

        # a BatchUnit (N81), for every burst this computes: those of both self and solution
        weight = BatchCore._weight(weight, {k: dict.__getitem__(self, k) for k in self if k in solution})

        nanoseconds_per_year = np.float64(365.25 * 24 * 60 * 60 * 1e9)

        # Detect solution type: pair-based, date-based, or velocity (spatial-only).
        # Every burst's variables through _vars_of: a DataArray is its own one (N81)
        vars_of = BatchCore._vars_of

        def obs_vars_of(value):
            """The observed variables compared: a Dataset's grids (its operators apply to the
            grids only), a DataArray AS IT IS, with or without (y, x) -- broadcast against the
            solution's grid, as xarray does (N81: a Batch of DataArrays needs no grid)."""
            return {v: a for v, a in vars_of(value).items() if isinstance(value, xr.DataArray) or 'y' in a.dims}
        # THE SOLUTION NEEDS A (y, x) GRID: it gives the form (pairs, dates or a rate) and the
        # pixels of the result -- _form's short error, not an IndexError below
        sol_first = next(iter(solution.values()))
        sol_sample = vars_of(sol_first)
        spatial_vars = [v for v, a in sol_sample.items() if 'y' in a.dims]
        if not spatial_vars:
            raise BatchCore._no_grid_error(sol_first, 'solution')
        sample_sol_dims = sol_sample[spatial_vars[0]].dims
        is_date_based = 'date' in sample_sol_dims
        is_pair_based = 'pair' in sample_sol_dims
        is_velocity = not is_date_based and not is_pair_based
        is_complex = isinstance(self, BatchComplex)

        if is_velocity:
            # Fused kernel: accumulate per-pair residuals, O(y_chunk * x_chunk) memory
            out = {}
            for key in self:
                if key not in solution:
                    continue
                obs_ds = self[key]
                obs_vars = vars_of(obs_ds)
                sol_vars = vars_of(solution[key])
                rmse_vars = {}
                for var in obs_vars_of(obs_ds):
                    sol_var = var if var in sol_vars else spatial_vars[0]
                    if sol_var not in sol_vars:
                        continue
                    obs_da = obs_vars[var]
                    vel_da = sol_vars[sol_var]
                    if not BatchCore._is_grid(obs_da):
                        # a DataArray without (y, x): on the rate's grid, as xarray broadcasts it
                        obs_da = obs_da.broadcast_like(vel_da).transpose(..., *vel_da.dims)

                    if 'pair' in obs_da.dims:
                        tdim = 'pair'
                        refs = obs_ds['ref'].values.astype('datetime64[ns]').astype(np.int64)
                        reps = obs_ds['rep'].values.astype('datetime64[ns]').astype(np.int64)
                        dt_np = ((reps.astype(np.float64) - refs.astype(np.float64))
                                 / nanoseconds_per_year).astype(np.float32)
                    elif 'date' in obs_da.dims:
                        tdim = 'date'
                        dates = obs_da.coords['date'].values.astype('datetime64[ns]').astype(np.int64)
                        dt_np = ((dates.astype(np.float64) - np.float64(dates[0]))
                                 / nanoseconds_per_year).astype(np.float32)
                    else:
                        continue

                    if obs_da.dims[0] != tdim:
                        obs_da = obs_da.transpose(tdim, ...)

                    obs_dask = obs_da.data
                    vel_data = vel_da.data
                    if not isinstance(vel_data, da.Array):
                        vel_data = da.from_array(vel_da.values, chunks=obs_dask.chunks[1:])

                    w_dask = None
                    if weight is not None and key in weight:
                        # the grid of the same name: _weight() raised for one it lacks
                        w_da = BatchCore._weight_of(weight[key], var)
                        if w_da.dims[0] != tdim:
                            w_da = w_da.transpose(tdim, ...)
                        w_dask = w_da.data

                    def _kernel(obs_blk, vel_blk, *args,
                                _dt=dt_np, _cplx=is_complex):
                        weight_blk = args[0] if args else None
                        P = obs_blk.shape[0]
                        sh = obs_blk.shape[1:]
                        accum = np.zeros(sh, dtype=np.float64)
                        if weight_blk is not None:
                            w_sum = np.zeros(sh, dtype=np.float64)
                        for p in range(P):
                            pred = vel_blk * _dt[p]
                            if _cplx:
                                resid = np.angle(
                                    obs_blk[p] * np.conj(np.exp(1j * pred)))
                            else:
                                resid = obs_blk[p].astype(np.float32) - pred
                            if weight_blk is not None:
                                w = weight_blk[p].astype(np.float64)
                                accum += w * resid.astype(np.float64) ** 2
                                w_sum += w
                            else:
                                accum += resid.astype(np.float64) ** 2
                        denom = w_sum if weight_blk is not None else np.float64(P)
                        return np.sqrt(accum / denom).astype(np.float32)

                    bw_args = [obs_dask, 'pyx', vel_data, 'yx']
                    if w_dask is not None:
                        bw_args += [w_dask, 'pyx']
                    rmse_dask = da.blockwise(
                        _kernel, 'yx', *bw_args,
                        concatenate=True,
                        dtype=np.float32,
                        meta=np.empty((0, 0), dtype=np.float32))

                    coords = {k: v for k, v in obs_da.coords.items()
                              if k in ('y', 'x', 'spatial_ref')}
                    rmse_vars[var] = xr.DataArray(
                        rmse_dask, dims=('y', 'x'), coords=coords)

                if isinstance(obs_ds, xr.DataArray) and not rmse_vars:
                    continue
                # a DataArray in, a DataArray out (N81)
                out[key] = BatchCore._form(obs_ds, rmse_vars, attrs=obs_ds.attrs)
            return Batch(out)

        # --- Non-velocity: date-based or pair-based solution ---
        if is_date_based:
            # Reconstruct pairs from date-based solution
            recon = {}
            for key in self:
                if key not in solution:
                    continue
                obs_ds = self[key]
                obs_vars = vars_of(obs_ds)
                sol_vars = vars_of(solution[key])
                recon_vars = {}
                for var, obs_da in obs_vars_of(obs_ds).items():
                    if var not in sol_vars:
                        continue
                    refs = obs_da.coords['ref'].values
                    reps = obs_da.coords['rep'].values
                    recon_list = []
                    for p in range(len(refs)):
                        recon_list.append(
                            sol_vars[var].sel(date=reps[p]) - sol_vars[var].sel(date=refs[p])
                        )
                    recon_vars[var] = xr.concat(recon_list, dim='pair')
                if isinstance(obs_ds, xr.DataArray) and not recon_vars:
                    continue
                recon[key] = BatchCore._form(obs_ds, recon_vars)
            solution_pairs = Batch(recon)
        else:
            solution_pairs = solution

        # Compute error — angle-based for complex phase, direct for real
        if is_complex:
            error_dict = {}
            for key in self:
                if key not in solution_pairs:
                    continue
                obs_ds = self[key]
                obs_vars = vars_of(obs_ds)
                sol_vars = vars_of(solution_pairs[key])
                err_vars = {}
                for var in obs_vars_of(obs_ds):
                    sol_var = var if var in sol_vars else next(
                        (v for v, a in sol_vars.items() if 'y' in a.dims), None)
                    if sol_var is None:
                        continue
                    err_vars[var] = xr.apply_ufunc(
                        lambda obs, pred: np.angle(
                            obs * np.conj(np.exp(1j * pred))
                        ).astype(np.float32),
                        obs_vars[var], sol_vars[sol_var],
                        dask='parallelized', output_dtypes=[np.float32])
                if isinstance(obs_ds, xr.DataArray) and not err_vars:
                    continue
                error_dict[key] = BatchCore._form(obs_ds, err_vars)
            error = Batch(error_dict)
        else:
            error = self - solution_pairs

        # Compute per-pixel RMSE across temporal dimension (pair or date)
        out = {}
        for key in error:
            err_ds = error[key]
            rmse_vars = {}
            for var, err_da in vars_of(err_ds).items():
                if 'y' not in err_da.dims:
                    continue
                tdim = next((d for d in ('pair', 'date') if d in err_da.dims), None)
                if tdim is None:
                    continue
                err_sq = err_da ** 2
                if weight is not None and key in weight:
                    # the grid of the same name: _weight() raised for one it lacks
                    w_da = BatchCore._weight_of(weight[key], var)
                    rmse_val = np.sqrt((w_da * err_sq).sum(tdim) / w_da.sum(tdim))
                else:
                    rmse_val = np.sqrt(err_sq.mean(tdim))
                rmse_vars[var] = rmse_val.astype('float32')
            if isinstance(err_ds, xr.DataArray) and not rmse_vars:
                continue
            # a DataArray in, a DataArray out (N81)
            out[key] = BatchCore._form(err_ds, rmse_vars, attrs=self[key].attrs)

        return Batch(out)

    def polyval(self, coeffs: dict[str, list | xr.DataArray], dim: str = 'x') -> BatchCore:
        """
        Evaluate polynomial coefficients for each burst.

        Applies xarray.polyval to evaluate polynomial corrections at each position
        along the specified dimension. Designed to work with polynomial coefficients
        returned by Stack.burst_polyfit.

        Parameters
        ----------
        coeffs : dict[str, list | xr.DataArray]
            Polynomial coefficients per burst. Can be either:
            - list[float]: [ramp, offset] for linear polynomial ramp*x + offset (single pair)
            - list[list[float]]: [[ramp0, offset0], [ramp1, offset1], ...] (multiple pairs)
            - list[float]: [offset0, offset1, ...] for degree=0 with multiple pairs
            - xr.DataArray: with 'degree' dimension [1, 0] (xarray.polyfit format)
            Following xarray.polyfit convention: highest degree first.
        dim : str, optional
            Coordinate dimension to evaluate polynomial on. Default is 'x' for range
            direction corrections.

        Returns
        -------
        BatchCore (or subclass)
            A Batch of DataArrays, of this batch's class unconverted: the
            polynomial evaluated along `dim` (per pair when the coefficients are
            per pair), named after the first gridded variable. A DataArray, so
            it applies to every grid of the batch it is combined with (N81).

        Examples
        --------
        >>> # Single pair: Estimate offsets and ramps
        >>> coeffs = Stack.burst_polyfit(intfs, degree=1)
        >>> corrections = intfs.polyval(coeffs, dim='x')
        >>> intfs_aligned = intfs - corrections

        >>> # Multiple pairs: coefficients are lists per pair
        >>> offsets = Stack.burst_polyfit(intfs_multi, degree=0)
        >>> # offsets = {'burst1': [off0, off1], 'burst2': [off0, off1]}
        >>> intfs_aligned = intfs_multi - offsets

        See Also
        --------
        xarray.polyval : Underlying polynomial evaluation function
        Stack.burst_polyfit : Function that produces compatible coefficients
        """
        result = {}
        for bid, ds in self.items():
            # Get a spatial variable (with y, x dims)
            sample_var = ds.name if isinstance(ds, xr.DataArray) else BatchCore._grid_vars(ds)[0]
            sample_da = ds if isinstance(ds, xr.DataArray) else ds[sample_var]

            if bid not in coeffs:
                # No coefficients for this burst - zero correction
                result[bid] = xr.zeros_like(sample_da).rename(sample_var)
                continue

            # Get coordinate for evaluation
            coord = ds.coords[dim]

            coeff = coeffs[bid]

            # Check if we have per-pair coefficients (list of lists or list of scalars for multiple pairs)
            has_pair_dim = 'pair' in sample_da.dims
            n_pairs = sample_da.sizes.get('pair', 1)

            if isinstance(coeff, (list, tuple)) and len(coeff) > 0:
                first_elem = coeff[0]

                # Detect format:
                # - Single pair degree=1: [ramp, offset] where both are scalars
                # - Single pair degree=0: scalar (but wrapped in list by caller)
                # - Multi pair degree=0: [off0, off1, ...] list of scalars
                # - Multi pair degree=1: [[ramp0, off0], [ramp1, off1], ...] list of lists

                if isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                    corrections = []
                    for pair_coeff in coeff:
                        corr = pair_coeff[0] * coord + pair_coeff[1]
                        corrections.append(corr)
                    # Stack along pair dimension
                    correction = xr.concat(corrections, dim='pair')

                elif has_pair_dim and len(coeff) == n_pairs and not isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=0: [off0, off1, ...] - all scalars matching pair count
                    # Check if it looks like [ramp, offset] for single pair (2 elements, no pair dim wouldn't reach here)
                    if len(coeff) == 2 and not has_pair_dim:
                        # Single pair degree=1: [ramp, offset]
                        correction = coeff[0] * coord + coeff[1]
                    else:
                        # Multi-pair degree=0
                        corrections = [xr.full_like(coord, off, dtype=float) for off in coeff]
                        correction = xr.concat(corrections, dim='pair')

                else:
                    # Single pair degree=1: [ramp, offset]
                    correction = coeff[0] * coord + coeff[1]

            elif isinstance(coeff, xr.DataArray):
                # General case using xr.polyval
                correction = xr.polyval(coord, coeff)
            else:
                # Single scalar (degree=0, single pair)
                correction = xr.full_like(coord, float(coeff), dtype=float)

            result[bid] = correction.rename(sample_var)

        # a correction is a polynomial, never re-converted: a BatchWrap does not wrap it
        return self._view(result)

    # def coarsen(self, window: dict[str,int], **kwargs):
    #     """
    #     intfs.coarsen({'y':2, 'x':8}, boundary='trim').mean().isel(0)
    #     """
    #     return type(self)({
    #         k: ds.coarsen(window, **kwargs)
    #         for k, ds in self.items()
    #     })

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
        chunks = self.chunks
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds = ds.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds = ds.isel({dim: slice(start, None)})
            # coarsen and revert original chunks
            out[key] = ds.coarsen(window, **kwargs)

        return type(self)(out)

    def chunk(self, chunks, p2p=False):
        """
        Rechunk the data in each burst dataset.

        Parameters
        ----------
        chunks : dict, int or 'auto'
            Chunk specification. If 'auto', uses chunk size 1 for first dimension
            (date/pair) and uniform chunking for spatial dimensions (y, x). An int
            is one size for every dim, as in xarray: -1 is one chunk.
        p2p : bool
            Use P2P (peer-to-peer) rechunk for constant-memory rechunking.
            Creates a materialization barrier that breaks shared upstream
            dependencies. Useful when switching from space mode (pair=1)
            to time mode (pair=-1) after operations with shared task graphs
            like interferogram(). Default False.

        Returns
        -------
        BatchCore
            New batch with rechunked data.

        Examples
        --------
        >>> # Explicit chunks
        >>> batch.chunk({'y': 2048, 'x': 2048})

        >>> # Auto: date=1, y/x=uniform based on dask.config['array.chunk-size']
        >>> batch.chunk('auto')

        >>> # P2P rechunk: space→time mode with materialization barrier
        >>> batch.chunk({'pair': -1, 'y': 256, 'x': 256}, p2p=True)
        """
        import dask
        from .utils_dask import rechunk2d

        # P2P rechunk context: constant memory, acts as materialization barrier
        if p2p:
            ctx = dask.config.set({
                "array.rechunk.method": "p2p",
                "optimization.fuse.active": False,
            })
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        def rechunk(arr, dims):
            """one spatial variable rechunked; `dims` are the dims of its burst"""
            if chunks == 'auto':
                # Use rechunk2d for uniform chunk sizes
                y_size, x_size = arr.shape[-2], arr.shape[-1]
                element_bytes = arr.dtype.itemsize
                in_chunks = (arr.data.chunks[-2], arr.data.chunks[-1]) if hasattr(arr.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes, input_chunks=in_chunks)
                if arr.ndim == 3:
                    var_chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                else:
                    var_chunks = {'y': optimal['y'], 'x': optimal['x']}
            elif not isinstance(chunks, dict):
                # one size for every dim, as xarray's .chunk(-1): the dict branch
                # below failed on it with "'int' object is not a mapping"
                var_chunks = {d: chunks for d in arr.dims}
            else:
                # Explicit chunks - add first dim=1 for 3D
                if arr.ndim == 3:
                    var_chunks = {arr.dims[0]: 1, **chunks}
                else:
                    var_chunks = chunks
                # a dim of the burst this variable lacks -- the 'pair' of the
                # metadata a reduction carried past a (y, x) grid -- is not its to chunk
                var_chunks = {d: c for d, c in var_chunks.items() if d in arr.dims or d not in dims}
            return arr.chunk(var_chunks)

        with ctx:
            # Only chunk spatial variables (y, x dims), leave non-spatial as-is
            result = {}
            for k, ds in self.items():
                if isinstance(ds, xr.DataArray):
                    # a Batch of DataArrays: the DataArray itself, when it is spatial
                    spatial = ds.ndim in (2, 3) and ds.dims[-2:] == ('y', 'x')
                    result[k] = rechunk(ds, ds.dims) if spatial else ds
                    continue
                rechunked_vars = {}
                for var in ds.data_vars:
                    arr = ds[var]
                    # Only touch spatial variables
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
                    rechunked_vars[var] = rechunk(arr, ds.dims)
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def chunk2d(self, budget=None, chunks=1, p2p=False):
        """
        Rechunk for pair-based (2D) processing: dim-0=chunks, spatial dims sized to budget.

        Computes optimal spatial chunk sizes so that each 2D slice (one pair/date)
        fits within the specified memory budget.

        Parameters
        ----------
        budget : str or None
            Memory budget per chunk, e.g. '128MiB', '256MB', '1GiB'.
            If None (default), uses dask.config['array.chunk-size'].
        chunks : int
            Chunk size for the first (pair/date) dimension. Default 1.
            Use -1 to merge all into one chunk.
        p2p : bool
            Use P2P rechunk for constant-memory rechunking. Default False.

        Returns
        -------
        BatchCore
            New batch with rechunked data.

        Examples
        --------
        >>> stack = Stack().load(zarr_path).chunk2d('128MiB')
        >>> phase, corr = stack.phasediff_multilook(pairs, wavelength=200)
        >>> # Partial merge for faster snapshot + read:
        >>> intfcorr2d.chunk2d('5MB', 20).snapshot('intfcorr2d')
        """
        import dask
        from .utils_dask import rechunk2d

        target_mb = _parse_budget(budget) if budget is not None else None

        ctx = dask.config.set({
            "array.rechunk.method": "p2p" if p2p else "tasks",
            **({"optimization.fuse.active": False} if p2p else {}),
        })

        with ctx:
            result = {}
            for k, ds in self.items():
                # Compute spatial chunks once per burst using 8 bytes (complex64)
                # so all variables get identical spatial chunks.
                # y, x ARE THE SPATIAL AXES AND WHAT LEADS THEM -- date or pair --
                # IS THE STACK, on whichever variable carries it: the stack
                # length is read off every (stack, y, x) variable and the
                # spatial chunks off a stack variable when there is one, picked
                # by name -- never off whichever variable happens to come first
                # (a DataArray is its own one variable, N81)
                vars_ = BatchCore._vars_of(ds)
                rasters = [v for v, a in vars_.items()
                           if a.ndim in (2, 3) and a.dims[-2:] == ('y', 'x')]
                stacks = [v for v in rasters if vars_[v].ndim == 3]
                n_stack = max((vars_[v].shape[0] for v in stacks), default=0)
                picked = sorted(stacks or rasters, key=str)
                sample = vars_[picked[0]] if picked else None
                if sample is None:
                    result[k] = ds
                    continue
                # Scale budget by chunks so total chunk memory stays within budget
                dim0 = chunks if chunks != -1 else (n_stack if n_stack > 0 else 1)
                per_slice_mb = (target_mb / dim0) if target_mb is not None else None
                y_size, x_size = sample.shape[-2], sample.shape[-1]
                in_chunks = (sample.data.chunks[-2], sample.data.chunks[-1]) if hasattr(sample.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                   input_chunks=in_chunks, target_mb=per_slice_mb, merge=True)
                rechunked_vars = {}
                for var, arr in vars_.items():
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
                    if arr.ndim == 3:
                        var_chunks = {arr.dims[0]: chunks, 'y': optimal['y'], 'x': optimal['x']}
                    else:
                        var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                    rechunked = arr.chunk(var_chunks)
                    if hasattr(rechunked.data, 'dask'):
                        # Full fusion only when graph has enough layers for linear chains.
                        # Rechunk-only graphs (≤3 layers) have no fusible chains —
                        # skip expensive ensure_dict + fuse_linear (22s on 1M keys, 0% reduction).
                        n_layers = len(rechunked.data.__dask_graph__().layers)
                        with dask.config.set({'optimization.fuse.active': n_layers > 3}):
                            (rechunked.data,) = dask.optimize(rechunked.data)
                    rechunked_vars[var] = rechunked
                if isinstance(ds, xr.DataArray):
                    ds = rechunked_vars[ds.name]
                elif rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def chunk1d(self, budget=None, chunks=-1, p2p=False):
        """
        Rechunk for date-based (1D) processing: dim-0=chunks, spatial dims sized to budget.

        Computes optimal spatial chunk sizes so that the date/pair stack
        fits within the specified memory budget per spatial tile.

        Parameters
        ----------
        budget : str
            Memory budget per chunk, e.g. '128MiB', '256MB', '1GiB'.
        chunks : int
            Chunk size for the first (pair/date) dimension. Default -1 (all).
            Use e.g. 20 for partial merge — much faster rechunk/snapshot
            while still grouping pairs for efficient sequential reads.
        p2p : bool
            Use P2P rechunk for constant-memory rechunking. Default False.

        Returns
        -------
        BatchCore
            New batch with rechunked data.

        Examples
        --------
        >>> data = Stack().snapshot('detrend').chunk1d('128MiB')
        >>> model = data.fit1d(weight=corr)
        >>> # Partial merge for faster snapshot:
        >>> intfcorr2d.chunk1d('1GB', 20).snapshot('intfcorr')
        """
        import dask
        from .utils_dask import rechunk2d

        target_mb = _parse_budget(budget) if budget is not None else None

        ctx = dask.config.set({
            "array.rechunk.method": "p2p" if p2p else "tasks",
            **({"optimization.fuse.active": False} if p2p else {}),
        })

        with ctx:
            result = {}
            for k, ds in self.items():
                # Find the largest dim-0 among 3D variables (a DataArray is its own one, N81)
                vars_ = BatchCore._vars_of(ds)
                n_stack = 0
                sample = None
                for var, arr in vars_.items():
                    if arr.ndim == 3 and arr.dims[-2:] == ('y', 'x'):
                        if arr.shape[0] > n_stack:
                            n_stack = arr.shape[0]
                            sample = arr
                if sample is None:
                    result[k] = ds
                    continue
                # Resolve chunks: -1 means all pairs/dates in one chunk
                dim0 = n_stack if chunks == -1 else chunks
                # Divide budget by dim0 to get per-slice budget,
                # then use rechunk2d (a 2D function) with per-slice element_bytes=8.
                per_slice_mb = (target_mb / dim0) if target_mb is not None else None
                y_size, x_size = sample.shape[1], sample.shape[2]
                in_chunks = (sample.data.chunks[1], sample.data.chunks[2]) if hasattr(sample.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                   input_chunks=in_chunks, target_mb=per_slice_mb, merge=True)
                rechunked_vars = {}
                for var, arr in vars_.items():
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
                    if arr.ndim == 3:
                        var_chunks = {arr.dims[0]: chunks, 'y': optimal['y'], 'x': optimal['x']}
                    else:
                        var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                    rechunked = arr.chunk(var_chunks)
                    if hasattr(rechunked.data, 'dask'):
                        n_layers = len(rechunked.data.__dask_graph__().layers)
                        with dask.config.set({'optimization.fuse.active': n_layers > 3}):
                            (rechunked.data,) = dask.optimize(rechunked.data)
                    rechunked_vars[var] = rechunked
                if isinstance(ds, xr.DataArray):
                    ds = rechunked_vars[ds.name]
                elif rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def map(self, func, *args, **kwargs):
        # a result the class does not keep (a mask of a BatchWrap) is a plain Batch
        return self._result({k: func(ds, *args, **kwargs) for k, ds in self.items()})

    def to_dict(self) -> dict:
        """
        Extract data variables as a dictionary of {burst_key: {var_name: values}}.

        Returns
        -------
        dict
            Dictionary mapping burst keys to dictionaries of variable names to numpy arrays.

        Examples
        --------
        >>> # the per-date BPR, a Batch of DataArrays (stack.BPR is stack['BPR'])
        >>> stack.BPR.to_dict()
        {'123_262885_IW2': {'BPR': array([   0.  , -166.47])},
         '123_262886_IW2': {'BPR': array([   0.  , -166.34])},
         '123_262887_IW2': {'BPR': array([   0.  , -166.2 ])}}
        """
        result = {}
        for key, ds in self.items():
            # a DataArray is its own one variable (N81)
            result[key] = {var: arr.values for var, arr in BatchCore._vars_of(ds).items()}
        return result

    def compute(self):
        """
        Compute lazy data in the batch.

        Persists all bursts at once via dask.persist(). Data stays in
        distributed worker memory (not pulled to client), letting the scheduler
        optimize across the full graph. Rechunks results to match input chunk
        structure. For memory-constrained sequential processing, use snapshot().
        Lazy products of the result (e.g. dissolve()) keep its data in worker
        memory while they exist, so chains like x.compute().dissolve().compute()
        work without holding the intermediate result.

        A Batch of DataArrays is computed as the DataArrays it holds (N81): no
        Dataset, no temporary name. A batch with nothing lazy comes back as it is.

        Returns
        -------
        BatchCore
            New batch with computed data, rechunked to match input.

        Raises
        ------
        RuntimeError
            The batch needs data the cluster no longer holds (e.g. after client.restart()).
        """
        import dask
        from .utils_dask import progress_persisted

        # NOTHING LAZY, NOTHING TO COMPUTE: the batch as it is (no constructor rerun)
        if not BatchCore._is_lazy_any(self):
            return self

        # Save input chunk structure per burst
        all_input_chunks = {key: BatchCore._input_chunks(v) for key, v in self.items()}

        # Persist all bursts at once — single scheduler submission
        # progress_persisted extracts futures and blocks until completion
        result = dask.persist(dict(self))[0]
        progress_persisted(result, desc='Computing Batch...'.ljust(25))

        # Finalize: materialize coordinates, hold the futures, rechunk to match input
        return type(self)({key: BatchCore._persisted(v, all_input_chunks[key]) for key, v in result.items()})

    @staticmethod
    def _is_lazy_any(batch) -> bool:
        """Whether any burst of `batch` holds dask data (a variable or a coordinate), for compute()."""
        import dask
        return any(dask.is_dask_collection(v) for v in dict.values(batch))

    @staticmethod
    def _input_chunks(value) -> dict:
        """The chunks of every dask variable of one burst, for compute() to restore:
        {name: {dim: chunks}} of a Dataset, {None: {dim: chunks}} of a DataArray."""
        if isinstance(value, xr.DataArray):
            return {None: dict(zip(value.dims, value.data.chunks))} if hasattr(value.data, 'chunks') else {}
        return {v: dict(zip(value[v].dims, value[v].data.chunks))
                for v in value.data_vars if hasattr(value[v].data, 'chunks')}

    @staticmethod
    def _persisted(value, input_chunks):
        """One burst of a persist, a Dataset or a DataArray taken as it is (N81): the
        lazy coordinates materialised, every dask variable holding its cluster data
        (hold_persisted, N111) and put back on its input chunks (_input_chunks)."""
        import dask.array as da
        from .utils_dask import hold_persisted

        new_coords = {name: (coord.dims, coord.compute().values) for name, coord in value.coords.items()
                      if hasattr(coord, 'data') and hasattr(coord.data, 'compute')}
        if new_coords:
            value = value.assign_coords(new_coords)

        def finish(arr, chunks):
            """(the variable held and rechunked, whether it changed)"""
            # products of this batch keep its cluster data alive, even after the batch is freed
            held = isinstance(arr.data, da.Array)
            if held:
                arr = arr.copy(data=hold_persisted(arr.data))
            if chunks is not None:
                if isinstance(arr.data, np.ndarray):
                    arr = arr.chunk(chunks)
                elif hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks:
                    arr = arr.chunk(chunks)
            return arr, held or chunks is not None

        if isinstance(value, xr.DataArray):
            return finish(value, input_chunks.get(None))[0]
        rechunked_vars = {}
        for var_name in value.data_vars:
            arr, changed = finish(value[var_name], input_chunks.get(var_name))
            if changed:
                rechunked_vars[var_name] = arr
        return value.assign(rechunked_vars) if rechunked_vars else value

    def to_dataframe(self,
                     crs: str | int | None = 'auto',
                     debug: bool = False) -> pd.DataFrame:
        """
        Return a Pandas/GeoPandas DataFrame for all Batch scenes.
        
        Extracts attributes from each Dataset in the Batch (from .attrs or dim-indexed data vars)
        and combines them into a single DataFrame, matching the Stack.to_dataframe format
        with additional ref/rep columns for pair information.

        Parameters
        ----------
        crs : str | int | None, optional
            Coordinate reference system for the output GeoDataFrame.
            If 'auto', uses CRS from the data. If None, returns without CRS conversion.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            The DataFrame containing Batch scenes with their attributes.
            Index is (fullBurstID, burst) when the Datasets carry both in .attrs, else the default
            RangeIndex; Stack.to_dataframe() indexes by (fullBurstID, startTime).
            For pair-based data, ref and rep columns are added after the index.

        Examples
        --------
        >>> df = batch.to_dataframe()
        >>> df = batch.to_dataframe(crs=4326)
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd

        if not self:
            return pd.DataFrame()

        # Detect native CRS from data
        sample = next(iter(self.values()))
        native_crs = self.crs
        if native_crs is None:
            raise ValueError('Batch has no CRS. Check the processing pipeline that produced this Batch.')
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            crs = native_crs

        # Detect spatial data variables: the GRIDS (the operator rule), not the
        # per-pair or per-date metadata carried beside them
        spatial_vars = [v for v in sample.data_vars if BatchCore._is_grid(sample[v])]
        ndims = {sample[v].ndim for v in spatial_vars}
        if len(ndims) > 1:
            raise ValueError(f'Mixed 2D and 3D variables not supported: {{{", ".join(f"{v}: {sample[v].ndim}D" for v in spatial_vars)}}}')

        # Detect dimension: 'date' for BatchComplex, 'pair' for others, None for spatial-only.
        # Read off the grids: after mean('pair') the metadata still carries its pairs.
        grid_dims = set().union(*(sample[v].dims for v in spatial_vars)) if spatial_vars else set(sample.dims)
        if 'date' in grid_dims:
            dim = 'date'
        elif 'pair' in grid_dims:
            dim = 'pair'
        else:
            dim = None

        # Define the attribute order matching Stack.to_dataframe
        attr_order = ['fullBurstID', 'burst', 'startTime', 'polarization', 'flightDirection',
                      'pathNumber', 'subswath', 'mission', 'beamModeType', 'BPR']

        # Spatial-only data (e.g., RMSE, elevation): one row per pixel with data values
        if dim is None:
            frames = []
            for key, ds in self.items():
                # Get spatial data variables (the grids)
                spatial_vars = [v for v in ds.data_vars if BatchCore._is_grid(ds[v])]
                if not spatial_vars:
                    continue
                df_burst = ds[spatial_vars].to_dataframe().reset_index()
                # Drop all-NaN rows
                df_burst = df_burst.dropna(subset=spatial_vars, how='all')
                # Add burst metadata from attrs
                for attr_name in attr_order:
                    if attr_name in ds.attrs:
                        value = ds.attrs[attr_name]
                        if attr_name == 'startTime':
                            value = pd.Timestamp(value)
                        df_burst[attr_name] = value
                frames.append(df_burst)

            if not frames:
                return pd.DataFrame()
            df = pd.concat(frames, ignore_index=True)

            # Create Point geometry in data's native CRS
            df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
            df = gpd.GeoDataFrame(df, crs=native_crs)

            # Reorder: burst metadata first, then data, then geometry
            meta_cols = [c for c in attr_order if c in df.columns]
            data_cols = ['y', 'x'] + spatial_vars
            ordered = meta_cols + data_cols + ['geometry']
            df = df[[c for c in ordered if c in df.columns]]

            if 'fullBurstID' in df.columns and 'burst' in df.columns:
                df = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])

            if crs is not None and crs != native_crs:
                df = df.to_crs(crs)
            return df

        # Date/pair-based data: one row per pixel per date/pair with data values
        spatial_vars = [v for v in sample.data_vars
                       if 'y' in sample[v].dims and 'x' in sample[v].dims]
        frames = []
        for key, ds in self.items():
            for idx in range(ds.dims[dim]):
                # Extract 2D slice for this date/pair
                ds_slice = ds[spatial_vars].isel({dim: idx})
                df_slice = ds_slice.to_dataframe().reset_index()
                # Drop all-NaN rows
                df_slice = df_slice.dropna(subset=spatial_vars, how='all')

                # Add date/pair info
                if dim == 'pair':
                    if 'ref' in ds.coords:
                        df_slice['ref'] = pd.Timestamp(ds['ref'].values[idx])
                    if 'rep' in ds.coords:
                        df_slice['rep'] = pd.Timestamp(ds['rep'].values[idx])
                elif dim == 'date':
                    df_slice['date'] = pd.Timestamp(ds[dim].values[idx])

                # Add burst metadata from attrs
                for attr_name in attr_order:
                    if attr_name in ds.attrs:
                        value = ds.attrs[attr_name]
                        if attr_name == 'startTime':
                            value = pd.Timestamp(value)
                        df_slice[attr_name] = value

                frames.append(df_slice)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

        # Create Point geometry in data's native CRS
        df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
        df = gpd.GeoDataFrame(df, crs=native_crs)

        # Reorder columns: burst metadata, date/pair info, coordinates, data, geometry
        if dim == 'pair':
            time_cols = ['ref', 'rep']
        else:
            time_cols = ['date']
        meta_cols = [c for c in attr_order if c in df.columns]
        time_cols = [c for c in time_cols if c in df.columns]
        data_cols = ['y', 'x'] + spatial_vars
        ordered = meta_cols + time_cols + data_cols + ['geometry']
        df = df[[c for c in ordered if c in df.columns]]

        if 'fullBurstID' in df.columns and 'burst' in df.columns:
            df = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])

        if crs is not None and crs != native_crs:
            df = df.to_crs(crs)
        return df

    @property
    def spacing(self) -> tuple[float, float]:
        """Return the (y, x) grid spacing."""
        sample = next(iter(self.values()))
        return sample.y.diff('y').item(0), sample.x.diff('x').item(0)
    
    def downsample(self, new_spacing: tuple[float, float] | float | int, debug: bool = False):
        """
        Update the Batch data onto a grid with the given (y, x) spacing.
        Like to coarsening but with cell size in meters instead of pixels:
        intfs.downsample(60)
        intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()

        If the requested spacing rounds to one pixel per axis -- the data is
        already at, or coarser than, it -- returns the input unchanged.
        """
        if isinstance(new_spacing, (int, float)):
            new_spacing = (new_spacing, new_spacing)
        dy, dx = self.spacing
        yscale, xscale = max(1, int(np.round(new_spacing[0]/dy))), max(1, int(np.round(new_spacing[1]/dx)))
        # If both scale factors are 1, no downsampling needed - return as is
        if yscale == 1 and xscale == 1:
            return self
        if debug:
            print (f'DEBUG: cell size in meters: y={dy:.1f}, x={dx:.1f} -> y={new_spacing[0]:.1f}, x={new_spacing[1]:.1f}')

        # Compute output chunk budget: input first chunk × 8 bytes / coarsen factors.
        # This preserves the same spatial granularity (chunk count) after coarsening.
        from .utils_dask import rechunk2d
        sample_ds = next(iter(self.values()))
        if hasattr(sample_ds, 'obj'):
            sample_ds = sample_ds.obj
        output_budget_mb = None
        # a Dataset's variables, or a DataArray as its own one (N81)
        for arr in BatchCore._vars_of(sample_ds).values():
            if arr.ndim >= 2 and arr.dims[-2:] == ('y', 'x') and hasattr(arr.data, 'chunks'):
                cy0 = arr.data.chunks[-2][0]
                cx0 = arr.data.chunks[-1][0]
                output_budget_mb = cy0 * cx0 * 8 / (1024 * 1024) / yscale / xscale
                break

        # Coarsen the (y, x) planes only. .mean() over the whole Dataset would
        # also hit the 1D radar metadata -- and 'burst' is a STRING, so it dies
        # with "could not convert string to float". The metadata is re-attached
        # unchanged afterwards: downsampling a grid says nothing about the
        # wavelength that grid was measured at.
        meta_keys = {}
        spatial = {}
        for key, ds in self.items():
            if isinstance(ds, xr.DataArray):
                # a DataArray is its own grid and carries no metadata (N81)
                meta_keys[key] = None
                spatial[key] = ds
                continue
            m = [v for v in ds.data_vars
                 if not (ds[v].ndim >= 2 and tuple(ds[v].dims[-2:]) == ('y', 'x'))]
            meta_keys[key] = ds[m] if m else None
            spatial[key] = ds[[v for v in ds.data_vars if v not in m]]
        result = type(self)(spatial).coarsen(
            {'y': yscale, 'x': xscale}, boundary='trim').mean()
        for key in list(result.keys()):
            md = meta_keys.get(key)
            if md is not None and len(md.data_vars):
                attrs = result[key].attrs
                result[key] = result[key].assign(
                    {v: md[v] for v in md.data_vars})
                result[key].attrs = attrs

        # Rechunk output to preserve input spatial granularity
        if output_budget_mb is not None:
            sample_ds = next(iter(result.values()))
            for arr in BatchCore._vars_of(sample_ds).values():
                if arr.ndim >= 2 and arr.dims[-2:] == ('y', 'x') and hasattr(arr.data, 'chunks'):
                    y_size, x_size = arr.shape[-2], arr.shape[-1]
                    optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                       target_mb=output_budget_mb)
                    result = result.chunk({'y': optimal['y'], 'x': optimal['x']})
                    break

        return result

    def save(self, store: str, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Saving...', debug=False):
        return utils_io.save(self, store=store, storage_options=storage_options, caption=caption, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, n_jobs=n_jobs, debug=debug)
        if not isinstance(data, dict):
            raise ValueError(f'ERROR: open() returns multiple datasets, you need to use Stack class to open them.')
        return data
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Snapshotting...',
                debug=False, **kwargs):
        # Only save if this batch has data; otherwise just open existing store
        if len(self) > 0:
            utils_io.save(self, store=store, storage_options=storage_options, caption=caption,
                         debug=debug)
        return utils_io.open(store=store, storage_options=storage_options,
                            n_jobs=-1, debug=debug)

    def to_dataset(self, polarization=None, chunks='auto', compute: bool = False, debug: bool = False):
        """
        Merge multiple burst DataArrays into a single unified grid.

        This function efficiently combines bursts using dask operations for lazy
        evaluation. For each output chunk, it selects the minimal set of input
        bursts needed and lays them in ACQUISITION ORDER (_acquisition_order(),
        by each burst's startTime): where bursts overlap, the MOST RECENT burst's
        valid value wins; a burst's nodata never hides an earlier burst's valid
        value; pixels no burst covers hold the nodata value.

        THE NODATA VALUE IS SET BY THE DTYPE: NaN for float and complex grids,
        -1 for signed integers (fit3d level and conncomp), 0 for unsigned
        integers (unwrap2d conncomp) and False for bool; every grid keeps its
        own dtype. Integer or bool grids, of one burst or many, print one
        WARNING naming them: replace the nodata value or convert the dtype
        first where that value is wrong. plot(), to_geojson() and to_vtk()
        skip the same nodata.

        For best results with overlapping bursts, call .dissolve() first to average
        values in overlap regions, then call .to_dataset() to merge into a single grid.

        Parameters
        ----------
        polarization : str, optional
            Specific (y, x) variable to merge. If None, merges every (y, x) variable.
        chunks : str, int, or tuple, optional
            Spatial chunk size for processing. Options:
            - 'auto' (default): automatically determine chunk size based on memory
            - int: use same chunk size for both y and x dimensions
            - tuple (y_chunk, x_chunk): explicit chunk sizes for each dimension
            Note: Only 2D spatial chunks are supported. The stack dimension
            (date/pair) is always processed one slice at a time.
        compute : bool, optional
            Whether to compute the result immediately. Default is False (lazy).
        debug : bool, optional
            Print debugging/profiling information. Default is False.

        Returns
        -------
        xr.Dataset
            Merged data on a unified grid: ALWAYS a Dataset (N81), for one burst
            or many and for a Batch of Datasets or of DataArrays, holding the
            (y, x) variables only -- every one, or the one `polarization` names.
            A burst's other variables have no place on a merged grid. Every
            grid keeps its dtype, for one burst or many.

        Raises
        ------
        ValueError
            Nothing to merge: the batch has no bursts, or every burst is empty
            (e.g. after a spatial sel()). Empty bursts next to others are skipped.

        Examples
        --------
        >>> # Merge bursts into single grid (the most recent burst wins overlaps)
        >>> merged = batch.to_dataset()
        >>> vv = batch.to_dataset(polarization='VV')['VV']
        >>>
        >>> # With explicit chunk size for memory-constrained environments
        >>> merged = batch.to_dataset(chunks=1024)
        >>>
        >>> # With different y/x chunk sizes
        >>> merged = batch.to_dataset(chunks=(512, 2048))
        >>>
        >>> # For smooth overlaps, dissolve first then merge
        >>> merged = batch.dissolve().to_dataset()
        >>>
        >>> # Debug mode to see chunk sizes and burst distribution
        >>> merged = batch.to_dataset(debug=True)
        """
        import xarray as xr
        import numpy as np
        import dask
        import dask.array as da
        from insardev_toolkit import datagrid

        # NOTHING TO MERGE RAISES (decided): a None result is no input to any next step
        if not len(self):
            raise ValueError('ERROR: to_dataset(): the batch has no bursts. Nothing to merge.')

        sample = next(iter(self.values()))
        # A BATCH OF DATAARRAYS merges as it is (N81): each burst's grid is the DataArray
        # itself, named as the DataArray in the Dataset returned, as for any batch
        if isinstance(sample, xr.DataArray) and sample.name is None:
            raise TypeError("ERROR: to_dataset() needs a named DataArray batch. Name it: x.rename('name').")
        # THE (y, x) VARIABLES ONLY, the same for one burst or many: a burst's metadata
        # (BPR, radar_wavelength, burst strings) has no place on a merged grid
        grids = list(BatchCore._grids_of(sample))
        if not grids:
            names = ', '.join(str(v) for v in BatchCore._vars_of(sample))
            raise TypeError(f'ERROR: to_dataset(): the batch has no (y, x) grid ({names}).')
        if polarization is None:
            polarizations = grids
        elif polarization in grids:
            polarizations = [polarization]
        else:
            raise KeyError(f"ERROR: to_dataset(): no (y, x) variable '{polarization}'; the batch has "
                           f"{', '.join(str(v) for v in grids)}.")
        # every burst empty (e.g. after a spatial sel()) RAISES too; an empty burst next
        # to others is skipped below
        if not any(v.sizes.get('y', 0) and v.sizes.get('x', 0) for v in self.values()):
            raise ValueError(f"ERROR: to_dataset(): {'the burst is' if len(self) == 1 else 'every burst is'} "
                             f"empty (0 pixels), e.g. after a spatial sel(). Nothing to merge.")

        if len(self) == 1:
            if debug:
                print(f"=== to_dataset() debug ===")
                print(f"Single burst - returning directly (no merge needed)")
                print(f"Burst key: {next(iter(self.keys()))}")
                # Get spatial vars
                for var, da in list(BatchCore._grids_of(sample).items())[:3]:  # Show first 3 spatial vars
                    print(f"  {var}: shape={da.shape}, dtype={da.dtype}")
            if isinstance(sample, xr.DataArray):
                # the Dataset of the one grid, named as the DataArray
                sample = xr.Dataset({sample.name: sample})
            else:
                sample = sample[polarizations]
            # the nodata by dtype holds for one burst too: the exports skip it (_skip_nodata)
            _warn_int_nodata('to_dataset', {v: sample[v].dtype for v in sample.data_vars})
            if compute:
                from .utils_dask import progress_persisted
                progress_persisted(sample := sample.persist(), desc=f'Compute Dataset'.ljust(25))
                return sample
            return sample

        # Build data dictionary: {pol: [burst0_data, burst1_data, ...]}, THE EARLIEST BURST
        # FIRST: laid in this order, the most recent burst wins the overlaps
        burst_keys = BatchCore._acquisition_order(self)
        datas_by_pol = {pol: [self[k] if isinstance(self[k], xr.DataArray) and self[k].name == pol
                              else self[k][pol] for k in burst_keys] for pol in polarizations}

        # Get grid info from first polarization (all pols have same grid)
        first_pol = polarizations[0]
        first_datas = datas_by_pol[first_pol]

        # Handle stack dimension - preserve original coordinate values. EACH VARIABLE ITS
        # OWN (N81): a Stack's grids differ -- VV (date, y, x), ele (y, x) -- and the first
        # one's stack dimension indexed a 2-D grid as 3-D; a 2-D grid is one 'fake' slice
        # (read as its own 2-D blocks below: no expand_dims layer on the input)
        stack_of = {}
        for pol in polarizations:
            pdims = datas_by_pol[pol][0].dims
            if len(pdims) > 2:
                stack_of[pol] = (pdims[0], datas_by_pol[pol][0][pdims[0]].values)
            else:
                stack_of[pol] = ('fake', [0])
        first_datas = datas_by_pol[first_pol]
        stackvar, stackval = stack_of[first_pol]

        n_stack = len(stackval)

        # Filter out empty bursts (e.g., after spatial .sel() subsetting)
        # (not all of them: that raised above)
        nonempty_indices = [i for i, ds in enumerate(first_datas) if ds.y.size > 0 and ds.x.size > 0]
        if len(nonempty_indices) < len(first_datas):
            for pol in polarizations:
                datas_by_pol[pol] = [datas_by_pol[pol][i] for i in nonempty_indices]
            first_datas = datas_by_pol[first_pol]

        # Ensure coordinates are concrete values (important for data from delayed computations)
        dy = float(first_datas[0].y.diff('y').values[0])  # Signed spacing
        dx = float(first_datas[0].x.diff('x').values[0])

        # Find global min/max across all bursts
        # Use explicit float conversion to handle any lazy coordinate types
        y_min = min(float(np.asarray(ds.y.values).min()) for ds in first_datas)
        y_max = max(float(np.asarray(ds.y.values).max()) for ds in first_datas)
        x_min = min(float(np.asarray(ds.x.values).min()) for ds in first_datas)
        x_max = max(float(np.asarray(ds.x.values).max()) for ds in first_datas)

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

        fill_dtype = first_datas[0].dtype

        # Determine chunk sizes for spatial dimensions
        # Stack dimension is always processed one slice at a time (chunked to 1)
        if chunks == 'auto':
            # Use rechunk2d for uniform chunk sizes
            # Use 16 bytes to account for memory overhead (output + overlapping inputs)
            from .utils_dask import rechunk2d
            # to_dataset creates a new output grid — no input chunks to align to
            optimal = rechunk2d((ys.size, xs.size), element_bytes=16)
            y_chunk_size = optimal['y']
            x_chunk_size = optimal['x']
        elif isinstance(chunks, (int, np.integer)):
            # Single int: use same size for both dimensions
            y_chunk_size = min(int(chunks), ys.size)
            x_chunk_size = min(int(chunks), xs.size)
        elif isinstance(chunks, (tuple, list)) and len(chunks) == 2:
            # Tuple/list of (y_chunk, x_chunk)
            y_chunk_size = min(int(chunks[0]), ys.size)
            x_chunk_size = min(int(chunks[1]), xs.size)
        else:
            raise ValueError(
                f"chunks must be 'auto', int, or 2-tuple (y, x), got {type(chunks).__name__}: {chunks}. "
                "Note: 3D chunks are not supported; stack dimension is always processed per-slice."
            )

        # Number of chunks in each dimension
        n_y_chunks = (ys.size + y_chunk_size - 1) // y_chunk_size
        n_x_chunks = (xs.size + x_chunk_size - 1) // x_chunk_size

        # Extract extents and build spatial index for O(1) chunk lookup
        burst_info = []
        # chunk_index[(yi, xi)] = list of (burst_idx, coverage) sorted by coverage desc
        from collections import defaultdict
        chunk_index = defaultdict(list)

        # Index formula: idx = (coord - y_first) / dy
        # Works for both increasing (dy > 0) and decreasing (dy < 0) coords
        for burst_idx, d in enumerate(first_datas):
            # Ensure coordinates are concrete numpy arrays (not dask or other lazy types)
            # This is important when data comes from delayed computations like dissolve()
            burst_ys = np.asarray(d.y.values, dtype=np.float64)
            burst_xs = np.asarray(d.x.values, dtype=np.float64)
            info = {
                'y_coords': burst_ys,
                'x_coords': burst_xs,
            }
            burst_info.append(info)

            # Convert burst first/last coordinates to global array indices
            burst_y_start_idx = int(round((burst_ys[0] - y_first) / dy))
            burst_y_end_idx = int(round((burst_ys[-1] - y_first) / dy)) + 1
            burst_x_start_idx = int(round((burst_xs[0] - x_first) / dx))
            burst_x_end_idx = int(round((burst_xs[-1] - x_first) / dx)) + 1

            # Which output chunks does this burst overlap? (using array indices)
            y_chunk_start = max(0, burst_y_start_idx // y_chunk_size)
            y_chunk_end = min(n_y_chunks, (burst_y_end_idx + y_chunk_size - 1) // y_chunk_size)
            x_chunk_start = max(0, burst_x_start_idx // x_chunk_size)
            x_chunk_end = min(n_x_chunks, (burst_x_end_idx + x_chunk_size - 1) // x_chunk_size)

            # Register this burst for all overlapping chunks
            for yi in range(y_chunk_start, y_chunk_end):
                for xi in range(x_chunk_start, x_chunk_end):
                    # Compute coverage (number of pixels in overlap)
                    chunk_y0 = yi * y_chunk_size
                    chunk_y1 = min(chunk_y0 + y_chunk_size, ys.size)
                    chunk_x0 = xi * x_chunk_size
                    chunk_x1 = min(chunk_x0 + x_chunk_size, xs.size)

                    # Overlap in array index space
                    ov_y0 = max(chunk_y0, burst_y_start_idx)
                    ov_y1 = min(chunk_y1, burst_y_end_idx)
                    ov_x0 = max(chunk_x0, burst_x_start_idx)
                    ov_x1 = min(chunk_x1, burst_x_end_idx)

                    coverage = max(0, ov_y1 - ov_y0) * max(0, ov_x1 - ov_x0)
                    if coverage > 0:
                        chunk_index[(yi, xi)].append((burst_idx, coverage))

        # Each chunk's bursts in acquisition order (burst_idx follows it): the most recent
        # is laid last and wins the overlaps
        for key in chunk_index:
            chunk_index[key].sort()

        # Debug output
        if debug:
            print(f"=== to_dataset() debug ===")
            print(f"Number of bursts: {len(first_datas)}")
            print(f"Output grid: y={ys.size}, x={xs.size} (total {ys.size * xs.size:,} pixels)")
            print(f"Chunk size: y={y_chunk_size}, x={x_chunk_size} ({y_chunk_size * x_chunk_size:,} pixels/chunk)")
            print(f"Number of chunks: {n_y_chunks} y × {n_x_chunks} x = {n_y_chunks * n_x_chunks} total")
            print(f"Stack dimension: {stackvar}={n_stack} slices")
            print(f"Data type: {fill_dtype} ({np.dtype(fill_dtype).itemsize} bytes/pixel)")

            # Memory estimate per chunk
            chunk_mem_mb = y_chunk_size * x_chunk_size * np.dtype(fill_dtype).itemsize / 1024 / 1024
            print(f"Memory per chunk: ~{chunk_mem_mb:.1f} MB")

            # Burst distribution across chunks
            bursts_per_chunk = [len(chunk_index.get((yi, xi), []))
                               for yi in range(n_y_chunks) for xi in range(n_x_chunks)]
            if bursts_per_chunk:
                # Show distribution histogram
                from collections import Counter
                dist = Counter(bursts_per_chunk)
                dist_str = ", ".join(f"{k} bursts: {v}" for k, v in sorted(dist.items()))
                print(f"Chunk distribution: {dist_str}")

            # Burst extents
            print(f"Burst extents:")
            for idx, info in enumerate(burst_info):
                by = info['y_coords']
                bx = info['x_coords']
                print(f"  [{idx}] y=[{by.min():.1f}, {by.max():.1f}] "
                      f"x=[{bx.min():.1f}, {bx.max():.1f}] ({len(by)}×{len(bx)} pixels)")

        # Build result for each polarization: ONE TASK PER OUTPUT BLOCK, and one per tile it
        # reads, on the inputs' own blocks, that dask never fuses (_UnfusedTask) -- the same
        # keys, naming the same tasks, in every graph (as Stack._zarr_dask()). Read through
        # to_delayed(optimize_graph=False), an input was not Blockwise-fused in a
        # to_dataset() graph and was in every other: the interferogram product ref *
        # conj(rep) was mul(getitem, conjugate) here and one fused task of the two getitems
        # elsewhere. The scheduler, still holding that key among the inputs of a computed
        # result, kept its old dependencies (dask issue 9888), and to_vtk() failed now and
        # then with KeyError or 'missing keys'. Now an input's blocks are read as they are
        # computed anywhere, and nothing here fuses into them.
        # The merge functions are module level (_tile_for_dask, _merge_parts_for_dask).
        from dask.base import tokenize
        from dask._task_spec import TaskRef
        from dask.highlevelgraph import HighLevelGraph
        from .utils_dask import _UnfusedTask

        def _overlap(edges, lo, hi):
            """[(chunk, start, stop within it)] of the chunks the range [lo, hi) overlaps."""
            first = int(np.searchsorted(edges, lo, 'right')) - 1
            last = int(np.searchsorted(edges, hi, 'left')) - 1
            return [(i, max(lo, edges[i]) - edges[i], min(hi, edges[i + 1]) - edges[i])
                    for i in range(first, last + 1)]

        y_sizes = tuple(min(y_chunk_size, ys.size - yi * y_chunk_size) for yi in range(n_y_chunks))
        x_sizes = tuple(min(x_chunk_size, xs.size - xi * x_chunk_size) for xi in range(n_x_chunks))
        results = {}
        # AN INTEGER OR BOOLEAN GRID KEEPS ITS DTYPE (decided): no NaN, so its nodata value
        # is set by the dtype (_nodata_of: -1 signed, 0 unsigned, False bool), transparent
        # in overlaps and filling the pixels no burst covers -- ONE WARNING names them all
        nodata_of = {pol: _nodata_of(datas_by_pol[pol][0].dtype) for pol in polarizations}
        _warn_int_nodata('to_dataset', {pol: datas_by_pol[pol][0].dtype for pol in polarizations})
        for pol in polarizations:
            datas = datas_by_pol[pol]
            # this variable's own stack dimension and dtype (a Stack's grids differ)
            stackvar, stackval = stack_of[pol]
            n_stack = len(stackval)
            fill_dtype = np.dtype(datas[0].dtype)
            nodata = nodata_of[pol]

            # each burst's grid as it is: dask (numpy as one block), 3-D (stack, y, x) or 2-D
            arrays = [d.data if isinstance(d.data, da.Array) else da.from_array(d.data, chunks=d.data.shape)
                      for d in datas]
            edges = [[np.cumsum((0,) + tuple(c)).tolist() for c in a.chunks] for a in arrays]
            # the same merge of the same inputs on the same grid is the same keys
            token = tokenize(pol, stackvar, n_stack, ys, xs, y_chunk_size, x_chunk_size, str(fill_dtype), str(nodata),
                             [(a.name, a.chunks, info['y_coords'], info['x_coords'])
                              for a, info in zip(arrays, burst_info)])
            name = f'to_dataset-{token}'
            tname = f'to_dataset-tile-{token}'
            layer = {}

            for s_idx in range(n_stack):
                for yi in range(n_y_chunks):
                    yb0 = yi * y_chunk_size
                    yb1 = min(yb0 + y_chunk_size, ys.size)
                    for xi in range(n_x_chunks):
                        xb0 = xi * x_chunk_size
                        xb1 = min(xb0 + x_chunk_size, xs.size)

                        # O(1) lookup of overlapping bursts via spatial index
                        overlapping = chunk_index.get((yi, xi), [])

                        out_shape = (yb1 - yb0, xb1 - xb0)

                        # Output chunk coordinates
                        out_ys = ys[yb0:yb1]
                        out_xs = xs[xb0:xb1]

                        # the tiles (a tile over several input blocks: one part per block) and
                        # their offsets within this chunk; none gives a NaN block
                        parts = []
                        offsets = []

                        # Output chunk coordinate bounds (with small tolerance for floating point)
                        # Use tiny tolerance (1e-6 * spacing) to handle floating point precision
                        # Apply tolerance to expand bounds (not shrink), regardless of coord direction
                        tol_y = abs(dy) * 1e-6
                        tol_x = abs(dx) * 1e-6
                        y_lo = min(out_ys[0], out_ys[-1]) - tol_y
                        y_hi = max(out_ys[0], out_ys[-1]) + tol_y
                        x_lo = min(out_xs[0], out_xs[-1]) - tol_x
                        x_hi = max(out_xs[0], out_xs[-1]) + tol_x

                        for burst_idx, _ in overlapping:
                            info = burst_info[burst_idx]
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

                                # the tile, cut from each input block it overlaps
                                arr = arrays[burst_idx]
                                ey, ex = edges[burst_idx][-2], edges[burst_idx][-1]
                                if arr.ndim == 3:
                                    es = edges[burst_idx][0]
                                    sc = int(np.searchsorted(es, s_idx, 'right')) - 1
                                    lead, s_local = (sc,), s_idx - es[sc]
                                else:
                                    lead, s_local = (), None
                                for iy, py0, py1 in _overlap(ey, by0, by1):
                                    for ix, px0, px1 in _overlap(ex, bx0, bx1):
                                        key = (tname, s_idx, yi, xi, len(parts))
                                        layer[key] = _UnfusedTask(key, _tile_for_dask,
                                                                  TaskRef((arr.name,) + lead + (iy, ix)),
                                                                  s_local, py0, py1, px0, px1)
                                        parts.append(TaskRef(key))
                                        offsets.append((y_off + ey[iy] + py0 - by0, x_off + ex[ix] + px0 - bx0))

                        key = (name, s_idx, yi, xi)
                        layer[key] = _UnfusedTask(key, _merge_parts_for_dask, tuple(offsets), out_shape,
                                                  fill_dtype, nodata, *parts)

            graph = HighLevelGraph.from_collections(name, layer, dependencies=arrays)
            data = da.Array(graph, name, chunks=((1,) * n_stack, y_sizes, x_sizes), dtype=fill_dtype,
                            meta=np.empty((0, 0, 0), dtype=fill_dtype))

            result = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
                .rename(pol)\
                .assign_attrs(datas[0].attrs)
            # Preserve ref/rep coordinates along pair dimension
            if stackvar == 'pair':
                if 'ref' in datas[0].coords:
                    result = result.assign_coords(ref=(stackvar, datas[0].coords['ref'].values))
                if 'rep' in datas[0].coords:
                    result = result.assign_coords(rep=(stackvar, datas[0].coords['rep'].values))
            result = datagrid.spatial_ref(result, datas)
            if stackvar == 'fake':
                # a 2-D grid: no scalar 'fake' coordinate left on it
                result = result.isel({stackvar: 0}, drop=True)
            results[pol] = result

        # ALWAYS A DATASET (N81), of the one polarization requested or of all of them
        #
        # BOTH KWARGS STATED, not left to the default. Every DataArray here
        # was built from the SAME `ys` and `xs` (each with its own stack
        # dimension) -- the grid is
        # read once from the first polarization, above -- and the names are
        # the polarizations, so nothing overlaps and nothing needs
        # reconciling. `compat='override'` says exactly that and skips the
        # comparison; it is also the default xarray is moving to, so the
        # result cannot change under us. `join='exact'` turns the shared
        # grid from an assumption into a check: if two polarizations ever
        # arrive on different axes this raises, where the default outer
        # join would quietly pad the union with NaN.
        output = xr.merge(list(results.values()),
                          compat='override', join='exact')

        if compute:
            from .utils_dask import progress_persisted
            progress_persisted(output := output.persist(), desc=f'Computing Dataset...'.ljust(25))
        return output

    @staticmethod
    def _gpkg_block(blocks, yv, xv, decimals):
        """One spatial block reduced to its FINITE pixels, where it lives.

        `blocks` are the block's OWN arrays -- nothing of the scene's graph
        travels with them -- so what comes back is the sparse points rather
        than the raster they came from.
        """
        import numpy as np
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

    def to_geopackage(self, filename: str, crs: str | int | None = None,
                      decimals: int = 3, overwrite: bool = True,
                      debug: bool = False) -> None:
        """
        Write the batch to a GeoPackage, ONE LAYER PER BURST named for it.

        Every pixel holding a value becomes a point in its burst's layer,
        carrying the batch's variables as attributes. Pixels where every
        variable is NaN are not written -- a fit3d() model is mostly unsolved,
        and those rows would multiply the file by the coverage it does not
        have. A third dimension becomes COLUMNS, one per date or pair, which
        is how a time series is carried in a vector table. A complex variable
        becomes `<name>_amp` and `<name>_phase` (degrees) -- the form the
        quantity is read in, and lossless, where `real`/`imag` are the fit's
        own basis and neither means anything alone.

        MATERIALISED BLOCK BY BLOCK, NOT ALL AT ONCE. `compute()` persists the
        whole batch into cluster memory, which is right when the result is a
        raster the caller keeps; here the result is a FILE, and a scene of a
        few hundred million pixels never has to exist in memory. Each block is
        reduced to its finite points ON THE WORKER -- the same shape as
        `snapshot()`'s batched writes -- and only those points travel. Blocks
        go out `n_workers` at a time, so the cluster stays busy while the peak
        stays at one batch of sparse tables.

        THE BLOCKS ARE PASSED AS DELAYED BLOCKS, not as the dataset sliced
        inside the task: handing a dask-backed object to every task copies the
        whole scene's graph into every one of them, and the task would then
        have to compute inside a worker.

        THE WRITE IS SERIAL because a GeoPackage is a SQLite file with exactly
        one writer. Parallelism belongs on the extraction, which is where the
        work is; appending the tables is IO the cluster cannot help with.

        Parameters
        ----------
        filename : str
            Path to the output GeoPackage. Created if absent. The `.gpkg`
            extension is added when missing -- GDAL writes the file either
            way but warns that a nameless extension does not conform, and
            nothing else opens it by that name.
        crs : str | int | None, optional
            Reproject the points to this CRS. Default None keeps the batch's own.
        decimals : int, optional
            Round float attributes to this many decimals. Default 3 --
            millimetre precision for velocities in mm/yr and heights in
            metres, which is already below what the estimates resolve.
        overwrite : bool, optional
            Replace an existing file. Default True. When False the layers are
            added to whatever is already there.
        debug : bool, optional
            Print per-burst counts and timings.

        Returns
        -------
        None
            Nothing, so a notebook cell ending on this call stays silent
            instead of echoing the filename back.

        Examples
        --------
        >>> model.to_geopackage('model.gpkg')
        >>> model.to_geopackage('model')            # writes model.gpkg
        >>> model.to_geopackage('model_wgs84.gpkg', crs=4326)
        """
        import os
        import time
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import dask
        import dask.array as da
        from tqdm.auto import tqdm

        if not self:
            raise ValueError('to_geopackage(): the batch is empty')
        if not filename.lower().endswith('.gpkg'):
            filename = filename + '.gpkg'
        native_crs = self.crs
        if native_crs is None:
            raise ValueError('to_geopackage(): the batch has no CRS. Check the '
                             'pipeline that produced it.')
        if overwrite and os.path.exists(filename):
            os.remove(filename)

        try:
            from dask.distributed import get_client
            n_workers = max(len(get_client().nthreads()), 1)
        except (ValueError, ImportError):
            n_workers = 1

        written = 0
        pbar = tqdm(desc='GeoPackage...'.ljust(25), total=len(self))
        for key, ds in self.items():
            t0 = time.monotonic()
            names = [v for v in ds.data_vars
                     if ds[v].ndim >= 2 and ds[v].dims[-2:] == ('y', 'x')]
            if not names:
                pbar.update(1)
                continue
            dim = next((d for d in ('date', 'pair')
                        if d in ds[names[0]].dims), None)
            labels = []
            if dim is not None:
                vals = np.asarray(ds[dim].values)
                labels = [np.datetime_as_string(v, unit='D').replace('-', '')
                          if np.issubdtype(vals.dtype, np.datetime64)
                          else str(v) for v in vals]

            yv = np.asarray(ds['y'].values)
            xv = np.asarray(ds['x'].values)
            # THE DATA'S OWN CHUNKING is the block grid: a chunk is a unit the
            # pipeline already sized for memory. A dimension that is not
            # chunked is one block.
            dl, grid = {}, None
            for name in names:
                arr = ds[name]
                d = arr.data
                if not hasattr(d, 'to_delayed'):
                    d = da.from_array(np.asarray(d), chunks=d.shape)
                if dim is not None and dim in arr.dims:
                    d = d.rechunk({arr.dims.index(dim): -1})
                dl[name] = d.to_delayed()
                g = dl[name].shape[-2:]
                grid = g if grid is None else grid
                if g != grid:
                    raise ValueError(
                        f'to_geopackage(): {key!r} variable {name!r} is chunked '
                        f'{g} against {grid} for {names[0]!r}; rechunk the batch '
                        'so its variables share one block grid.')
            ychunks = ds[names[0]].chunks
            if ychunks is None:
                ysz, xsz = [len(yv)], [len(xv)]
            else:
                dims = ds[names[0]].dims
                ysz = list(ychunks[dims.index('y')])
                xsz = list(ychunks[dims.index('x')])
            yb = np.r_[0, np.cumsum(ysz)]
            xb = np.r_[0, np.cumsum(xsz)]

            mode, rows = ('w' if overwrite else 'a'), 0
            jobs = [(i, j) for i in range(len(ysz)) for j in range(len(xsz))]
            for k in range(0, len(jobs), n_workers):
                tasks = []
                for i, j in jobs[k:k + n_workers]:
                    blocks = {n: (v[i, j] if v.ndim == 2 else v[0, i, j])
                              for n, v in dl.items()}
                    tasks.append(dask.delayed(BatchCore._gpkg_block)(
                        blocks, yv[yb[i]:yb[i + 1]], xv[xb[j]:xb[j + 1]],
                        decimals))
                for out in dask.compute(*tasks):
                    if out is None:
                        continue
                    cols = {}
                    for c, v in out.items():
                        if c in ('y', 'x'):
                            continue
                        # the trailing #i is the date/pair index, named here
                        # where the labels are known
                        if '#' in c and labels:
                            base, idx = c.rsplit('#', 1)
                            suf = idx.split('_', 1)
                            lab = labels[int(suf[0])]
                            c = (f'{base}_{lab}' if len(names) > 1 else lab)
                            if len(suf) > 1:
                                c = f'{c}_{suf[1]}'
                        cols[c] = v
                    gdf = gpd.GeoDataFrame(
                        pd.DataFrame(cols),
                        geometry=gpd.points_from_xy(out['x'], out['y']),
                        crs=native_crs)
                    if crs is not None:
                        gdf = gdf.to_crs(crs)
                    # ONE WRITER AT A TIME: append as the blocks arrive, so the
                    # file grows and the client never holds the whole layer
                    gdf.to_file(filename, layer=str(key), driver='GPKG',
                                mode=mode)
                    mode = 'a'
                    rows += len(gdf)
                    del gdf, cols
            written += rows
            if debug:
                print(f'DEBUG: {key}  {rows:,} points from {len(jobs)} block(s)'
                      f'   {time.monotonic() - t0:.1f}s', flush=True)
            pbar.update(1)
        pbar.close()
        if debug:
            print(f'DEBUG: {filename}: {len(self)} layer(s), {written:,} points',
                  flush=True)

    def to_geojson(self, filename: str = None, crs: str = None, decimals: int = 3) -> str:
        """
        Convert batch data to GeoJSON with pixel rectangles.

        Creates a GeoJSON FeatureCollection where each pixel is represented
        as a polygon rectangle. All data variables (VV, VH, etc.) are preserved
        as properties in each feature. Coordinates are rounded to 6 digits.

        Parameters
        ----------
        filename : str, optional
            Path to save the GeoJSON file. If None (default), returns the
            GeoJSON string.
        crs : str, optional
            Target CRS (e.g., 'EPSG:4326'). If None (default), uses the data's
            original CRS.
        decimals : int, optional
            Number of decimal places for rounding values. Default is 3.

        Returns
        -------
        str or None
            GeoJSON string if filename is None, otherwise None (saves to file).

        Examples
        --------
        Save to file in WGS84:

        >>> velocity.to_geojson('velocity.geojson', crs='EPSG:4326')

        Save in original CRS:

        >>> velocity.to_geojson('velocity.geojson')

        Read from file:

        >>> import geopandas as gpd
        >>> gdf = gpd.read_file('velocity.geojson')

        Or get as string:

        >>> geojson_str = velocity.to_geojson()

        Or create GeoDataFrame from string:

        >>> import geopandas as gpd
        >>> gdf = gpd.read_file(velocity.to_geojson(), driver='GeoJSON')

        Or parse to dict:

        >>> import json
        >>> geojson = json.loads(velocity.to_geojson())
        """
        import geopandas as gpd
        import shapely.geometry

        # Merge to single dataset; an integer grid's nodata is left out, as to_dataset() sets it
        ds = _skip_nodata(self.to_dataset())

        # Get spatial data variables (with y, x dims) - excludes converted attributes
        data_vars = [v for v in ds.data_vars
                    if 'y' in ds[v].dims and 'x' in ds[v].dims]

        # Convert to dataframe and drop NaN rows
        df = ds.to_dataframe().dropna().reset_index()

        if df.empty:
            return None

        # Get pixel spacing
        dy, dx = self.spacing

        # Create rectangles in projected coordinates
        def point_to_rectangle(row, half_y, half_x):
            return shapely.geometry.Polygon([
                (row.x - half_x, row.y - half_y),
                (row.x + half_x, row.y - half_y),
                (row.x + half_x, row.y + half_y),
                (row.x - half_x, row.y + half_y)
            ])

        # Build GeoDataFrame with rectangles
        gdf = gpd.GeoDataFrame(
            df[['y', 'x'] + data_vars],
            geometry=[point_to_rectangle(row, abs(dy)/2, abs(dx)/2) for _, row in df.iterrows()],
            crs=self.crs
        )

        # Drop projected x, y columns and ensure column names are plain strings
        gdf = gdf.drop(columns=['x', 'y'])
        gdf.columns = [str(c) for c in gdf.columns]

        # Reproject if CRS specified, round values to 3 digits and coordinates to 6 digits
        if crs is not None:
            gdf = gdf.to_crs(crs)
        for col in data_vars:
            gdf[col] = gdf[col].astype(float).round(decimals)
        gdf.geometry = shapely.set_precision(gdf.geometry, grid_size=1e-6)

        # Add CRS for GIS software compatibility
        crs_urn = gdf.crs.to_string().replace(':', '::')
        crs_str = f'"crs": {{"type": "name", "properties": {{"name": "urn:ogc:def:crs:{crs_urn}"}}}}, '
        geojson = gdf.to_json(drop_id=True).replace('"type": "FeatureCollection", ', '"type": "FeatureCollection", ' + crs_str)

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(geojson)
            return
        return geojson

    def to_vtk(self, path: str, transform: 'BatchCore | Stack | None' = None,
               overlay: "xr.DataArray | None" = None, mask: bool = True):
        """Export to VTK.

        Merges bursts using to_dataset() and exports one VTK file per data variable
        (e.g., VV.vtk). Within each file, pairs become separate VTK arrays named
        by date (e.g., 20190708_20190702).

        Parameters
        ----------
        path : str
            Output directory/filename for VTK files.
        transform : BatchCore, Stack, or None, optional
            Optional transform Batch providing topography (``ele`` or ``z``),
            or a Stack (will call .transform() internally).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.

        Examples
        --------
        >>> velocity.to_vtk('velocity', transform=stack)
        >>> velocity.to_vtk('velocity', transform=stack.transform())
        >>> velocity.to_vtk('velocity', transform=stack, overlay=gmap)
        """
        import os
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk
        from .Batch import Batch
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        # Handle overlay-only case (export just overlay on topography)
        if not self and overlay is not None and transform is not None:
            tfm = transform if isinstance(transform, BatchCore) else Batch(transform)
            topo_merged = tfm[['ele']]._start_from(tfm).to_dataset()
            topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
            if topo_da is None:
                raise ValueError("transform must contain 'ele' variable")

            ov = overlay
            if 'band' not in ov.dims:
                ov = ov.expand_dims('band')
            topo_da = topo_da.interp(y=ov.y, x=ov.x, method='linear')

            if mask:
                topo_da = topo_da.where(np.isfinite(ov.isel(band=0)))

            layers = [topo_da.rename('z'), ov.rename('colors')]
            ds_out = xr.merge(layers, compat='override', join='left')
            vtk_grid = as_vtk(ds_out)

            if path.endswith('.vtk'):
                filename = path
            else:
                filename = f'{path}.vtk'
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

            writer = vtkStructuredGridWriter()
            writer.SetFileName(filename)
            writer.SetInputData(vtk_grid)
            writer.SetFileType(VTK_BINARY)
            writer.Write()
            return

        if not self:
            return

        tfm = transform if transform is None or isinstance(transform, BatchCore) else Batch(transform)

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(da, idx):
            if 'ref' in da.coords and 'rep' in da.coords:
                ref_val = da.coords['ref'].values[idx]
                rep_val = da.coords['rep'].values[idx]
                return f"{_format_dt(ref_val)}_{_format_dt(rep_val)}"
            return str(idx)

        os.makedirs(path, exist_ok=True)

        # Merge bursts into unified dataset(s) per variable.
        # Compute eagerly — VTK export needs all data in memory anyway,
        # and computing here avoids dask graph issues (stale rechunk keys
        # when downsample/coarsen layers are combined with to_dataset mosaic).
        # the (y, x) variables only, a Dataset for one burst or many (to_dataset(), N81)
        merged = _skip_nodata(self.to_dataset(compute=True))

        # Get transform elevation merged via to_dataset()
        topo_merged = None
        if tfm is not None:
            # Decimate each burst's transform to match corresponding input burst
            def _nearest_indices(source_coords, target_coords):
                descending = len(source_coords) > 1 and source_coords[0] > source_coords[-1]
                if descending:
                    source_coords = source_coords[::-1]
                indices = np.searchsorted(source_coords, target_coords)
                indices = np.clip(indices, 0, len(source_coords) - 1)
                prev_indices = np.clip(indices - 1, 0, len(source_coords) - 1)
                prev_diff = np.abs(source_coords[prev_indices] - target_coords)
                curr_diff = np.abs(source_coords[indices] - target_coords)
                indices = np.where(prev_diff < curr_diff, prev_indices, indices)
                if descending:
                    indices = len(source_coords) - 1 - indices
                return indices

            decimated = {}
            for k in self.keys():
                if k not in tfm:
                    continue
                tfm_ds = tfm[k][['ele'] + (['startTime'] if 'startTime' in tfm[k].data_vars else [])]
                tgt_ds = self[k]
                y_idx = _nearest_indices(tfm_ds.y.values, tgt_ds.y.values)
                x_idx = _nearest_indices(tfm_ds.x.values, tgt_ds.x.values)
                selected = tfm_ds.isel(y=y_idx, x=x_idx)
                selected = selected.assign_coords(y=tgt_ds.y, x=tgt_ds.x)
                decimated[k] = selected
            # a transform without any of the bursts RAISES: it gave no topography, silently
            if not decimated:
                raise ValueError(f"ERROR: to_vtk(): the transform has none of the bursts "
                                 f"({', '.join(list(self.keys())[:3])}). Pass their transform.")
            topo_merged = Batch(decimated).to_dataset(compute=True)

        data_vars = list(merged.data_vars)

        with tqdm(total=len(data_vars), desc='Exporting VTK') as pbar:
            for data_var in data_vars:
                da = merged[data_var]

                if 'pair' in da.dims:
                    n_pairs = da.sizes['pair']
                    export_items = []
                    for i in range(n_pairs):
                        da_slice = da.isel(pair=i)
                        if 'pair' in da_slice.dims:
                            da_slice = da_slice.squeeze('pair', drop=True)
                        pair_label = _format_pair(da, i)
                        export_items.append((pair_label, da_slice))
                else:
                    export_items = [(None, da)]

                if not export_items:
                    pbar.update(1)
                    continue

                ref_da = export_items[0][1]
                layers = []

                ov = None
                if overlay is not None:
                    if not isinstance(overlay, xr.DataArray):
                        raise TypeError("overlay must be an xarray.DataArray")
                    ov = overlay
                    if 'band' not in ov.dims:
                        ov = ov.expand_dims('band')
                    y_min, y_max = float(ref_da.y.min()), float(ref_da.y.max())
                    x_min, x_max = float(ref_da.x.min()), float(ref_da.x.max())
                    try:
                        ov_y_asc = len(ov.y) < 2 or float(ov.y[1]) > float(ov.y[0])
                        ov_x_asc = len(ov.x) < 2 or float(ov.x[1]) > float(ov.x[0])
                        y_slice = slice(y_min, y_max) if ov_y_asc else slice(y_max, y_min)
                        x_slice = slice(x_min, x_max) if ov_x_asc else slice(x_max, x_min)
                        ov = ov.sel(y=y_slice, x=x_slice)
                    except Exception:
                        pass
                    if ov.size == 0:
                        ov = None

                target_y = ov.y if ov is not None else ref_da.y
                target_x = ov.x if ov is not None else ref_da.x

                if topo_merged is not None:
                    topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
                    if topo_da is not None:
                        topo_da = topo_da.interp(y=target_y, x=target_x, method='linear')
                        if mask:
                            ref_for_mask = ref_da.interp(y=target_y, x=target_x, method='nearest') if ov is not None else ref_da
                            topo_da = topo_da.where(np.isfinite(ref_for_mask))
                        layers.append(topo_da.rename('z'))

                if ov is not None:
                    layers.append(ov.rename('colors'))

                for pair_label, da_item in export_items:
                    var_name = pair_label if pair_label is not None else data_var
                    if ov is not None:
                        da_item = da_item.interp(y=target_y, x=target_x, method='linear')
                    layers.append(da_item.rename(var_name))

                ds_out = xr.merge(layers, compat='override', join='left')
                vtk_grid = as_vtk(ds_out)

                filename = os.path.join(path, f"{data_var}.vtk")

                writer = vtkStructuredGridWriter()
                writer.SetFileName(filename)
                writer.SetInputData(vtk_grid)
                writer.SetFileType(VTK_BINARY)
                writer.Write()

                pbar.update(1)

    def to_vtks(self, path: str, transform: 'BatchCore | Stack | None' = None,
                overlay: "xr.DataArray | None" = None, mask: bool = True):
        """Export to VTK per-burst (separate file per burst).

        Parameters
        ----------
        path : str
            Output directory for VTK files.
        transform : BatchCore, Stack, or None, optional
            Optional transform Batch providing topography (`ele`),
            or a Stack (will call .transform() internally).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.

        Examples
        --------
        >>> velocity.to_vtks('vtk', transform=stack)
        >>> velocity.to_vtks('vtk', transform=stack.transform())
        """
        import os
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk
        from .Batch import Batch
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        tfm = transform if transform is None or isinstance(transform, BatchCore) else Batch(transform)

        if not self:
            return

        def _interp_to_grid(source: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
            if {'y', 'x'}.issubset(source.dims):
                return source.interp(y=target_da.y, x=target_da.x, method='linear')
            if {'lat', 'lon'}.issubset(source.dims):
                if {'lat', 'lon'}.issubset(target_da.coords):
                    return source.interp(lat=target_da.lat, lon=target_da.lon, method='linear')
                return source.rename({'lat': 'y', 'lon': 'x'}).interp(y=target_da.y, x=target_da.x, method='linear')
            return source

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"{_format_dt(val[0])}_{_format_dt(val[1])}"
            return _format_dt(val)

        os.makedirs(path, exist_ok=True)

        # the one exported grid of each burst: its integer nodata is not drawn, ONE WARNING
        _first = BatchCore._grids_of(next(iter(self.values())))
        if _first:
            _warn_int_nodata('to_vtks', dict([next((n, a.dtype) for n, a in _first.items())]))
        with tqdm(total=len(self), desc='Exporting VTK') as pbar:
            for burst, ds in self.items():
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
                            raise TypeError("overlay must be an xarray.DataArray")

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

    def plot(self,
            cmap: matplotlib.colors.Colormap | str | None = 'viridis',
            alpha: float = 0.7,
            vmin: float | None = None,
            vmax: float | None = None,
            quantile: float | None = None,
            symmetrical: bool = False,
            caption: str = '',
            cols: int = 4,
            rows: int = 4,
            size: float = 4,
            nbins: int = 5,
            aspect: float = 1.02,
            y: float = 1.05,
            flip: bool = False,
            extent: tuple[int, int] = (8000, 4000),
            composite: bool = False,
            gamma: float = 1.0,
            brightness: float = 2.0,
            ):
        """
        Plot batch data as images.

        Parameters
        ----------
        cmap : str or Colormap, optional
            Colormap for single-polarization plots. Default 'viridis'.
        alpha : float, optional
            Transparency. Default 0.7.
        vmin, vmax : float, optional
            Value range for colormap. Mutually exclusive with quantile.
        quantile : float or list, optional
            Quantile(s) for automatic range, e.g., [0.02, 0.98].
        symmetrical : bool, optional
            Center colormap at zero. Default False.
        caption : str, optional
            Title caption.
        cols, rows : int, optional
            Max columns/rows for subplots. Default 4.
        size : float, optional
            Figure size multiplier. Default 4.
        nbins : int, optional
            Number of axis tick bins. Default 5.
        aspect : float, optional
            Figure aspect ratio. Default 1.02.
        y : float, optional
            Suptitle y position. Default 1.05.
        flip : bool, optional
            Flip y-axis (north up). Default False.
        extent : tuple, optional
            Target display extent in pixels. Default (8000, 4000).
        composite : bool, optional
            Enable RGB composite mode for dual-pol data. Default False.
            R=co-pol, G=cross-pol, B=co-pol produces:
            - Magenta/pink: surface scattering (high co-pol)
            - Green: volume scattering (vegetation, high cross-pol)
            - White/gray: mixed scattering
            Requires exactly 2 polarizations (e.g., HH+HV or VV+VH).
            For best results: backscatter(decibels=False).lee().
        gamma : float, optional
            Gamma correction for composite tone curve. Default 1.0.
            Values > 1 brighten dark areas, < 1 increase contrast.
            Note: gamma changes color ratios; use brightness for uniform scaling.
        brightness : float, optional
            Linear brightness multiplier for composite mode. Default 2.0.
            Values > 1 brighten the image, < 1 darken it.
            Preserves color ratios (unlike gamma).

        Returns
        -------
        list
            List of FacetGrid (non-composite) or Figure (composite).
            Always returns a list for consistent handling.

        Examples
        --------
        >>> # Single polarization plot
        >>> stack[['VV']].backscatter().plot(quantile=[0.02, 0.98])

        >>> # Dual-pol RGB composite with Lee filter (recommended)
        >>> stack[['HH', 'HV']].backscatter(decibels=False).lee().plot(
        ...     composite=True)  # default brightness=2.0

        >>> # Adjust brightness while preserving colors
        >>> stack[['HH', 'HV']].backscatter(decibels=False).lee().plot(
        ...     composite=True, brightness=2.5)
        """
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.ticker as mticker
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as plt
        from .Batch import BatchWrap

        # no data means no plot and no error
        if not len(self):
            return

        wrap = True if type(self) == BatchWrap else False

        # use outer variables
        def plot_polarization(polarization):
            stackvar = list(sample[polarization].dims)[0] if len(sample[polarization].dims) > 2 else None

            # Calculate decimation factors from batch extent (without materializing full grid)
            batch = self[[polarization]]._start_from(self)
            if stackvar is not None:
                batch = batch.isel({stackvar: slice(0, rows*cols)})
            # Estimate merged grid size from coordinate ranges
            y_coords = np.concatenate([np.asarray(ds[polarization].y) for ds in batch.values()])
            x_coords = np.concatenate([np.asarray(ds[polarization].x) for ds in batch.values()])
            dy, dx = self.spacing
            size_y = int((y_coords.max() - y_coords.min()) / abs(dy)) + 1
            size_x = int((x_coords.max() - x_coords.min()) / abs(dx)) + 1
            factor_y = max(1, int(np.round(size_y / (extent[1] / rows))))
            factor_x = max(1, int(np.round(size_x / (extent[0] / cols))))

            # Decimate batches BEFORE to_dataset() - much more memory efficient
            batch_decimated = batch.isel(y=slice(None, None, factor_y), x=slice(None, None, factor_x))
            # an integer grid's nodata is not drawn, as to_dataset() sets it
            da = _skip_nodata(batch_decimated.to_dataset()[polarization])
            if stackvar is None:
                stackvar = 'fake'
                da = da.expand_dims({stackvar: [0]})

            # materialize for all the calculations and plotting
            from .utils_dask import progress_persisted
            progress_persisted(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

            # calculate min, max when needed
            if quantile is not None:
                q = np.nanquantile(da.values, quantile)
                # Handle edge cases: all NaN data returns scalar, empty data, etc.
                if np.ndim(q) == 0:
                    _vmin = _vmax = float(q)
                else:
                    _vmin, _vmax = q[0], q[-1]
            else:
                _vmin, _vmax = vmin, vmax
            # define symmetrical boundaries
            if symmetrical is True and _vmax > 0:
                minmax = max(abs(_vmin), _vmax)
                _vmin = -minmax
                _vmax =  minmax
            
            # note: multi-plots ineffective for linked lazy data
            # Convert coordinates to kilometers for cleaner display
            da_plot = (self.wrap(da) if wrap else da)
            fg = da_plot.plot.imshow(
                col=stackvar,
                col_wrap=min(cols, da[stackvar].size), size=size, aspect=aspect,
                vmin=_vmin, vmax=_vmax,
                cmap=cmap, alpha=alpha,
                interpolation='none',
                cbar_kwargs={'label': caption or polarization},
            )
            fg.set_axis_labels('easting [km]', 'northing [km]')
            fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
            fg.fig.suptitle(f'{polarization} {caption or ""}'.strip(), y=y)

            # fg is the FacetGrid returned by xarray.plot.imshow
            # Get original limits from first axis before any modifications
            if flip:
                first_ax = fg.axs.flatten()[0]
                orig_xlim = first_ax.get_xlim()
                orig_ylim = first_ax.get_ylim()
                # Ensure we flip to reversed order (max, min)
                flipped_xlim = (max(orig_xlim), min(orig_xlim))
                flipped_ylim = (max(orig_ylim), min(orig_ylim))

            for idx, ax in enumerate(fg.axs.flatten()):
                # flip axes if requested (force consistent flipped limits)
                if flip:
                    ax.set_xlim(flipped_xlim)
                    ax.set_ylim(flipped_ylim)
                # format tick labels in km
                km_formatter = FuncFormatter(lambda v, _: f'{v/1000:.0f}')
                ax.xaxis.set_major_formatter(km_formatter)
                ax.yaxis.set_major_formatter(km_formatter)
                if stackvar == 'fake':
                    # remove 'fake = 0' title
                    ax.set_title('')
                elif stackvar in ('pair', 'date') and idx < da[stackvar].size:
                    # Format pair/date titles nicely
                    if stackvar == 'pair':
                        # Get ref/rep from non-dimension coordinates
                        if 'ref' in da.coords and 'rep' in da.coords:
                            ref_val = da.coords['ref'].values[idx]
                            rep_val = da.coords['rep'].values[idx]
                            ref_str = pd.Timestamp(ref_val).strftime('%Y-%m-%d')
                            rep_str = pd.Timestamp(rep_val).strftime('%Y-%m-%d')
                            ax.set_title(f'{ref_str} {rep_str}')
                        else:
                            ax.set_title(f'pair={idx}')
                    elif stackvar == 'date':
                        coord_val = da[stackvar].values[idx]
                        if hasattr(coord_val, 'strftime'):
                            ax.set_title(f'date={coord_val.strftime("%Y-%m-%d")}')
                        else:
                            ax.set_title(f'date={pd.Timestamp(coord_val).strftime("%Y-%m-%d")}')

            return fg

        def plot_composite(pol1, pol2):
            """
            Plot RGB composite using ASF HyP3 decomposition formula.

            Based on: https://github.com/ASFHyP3/hyp3-lib/blob/develop/docs/rgb_decomposition.md
            - Red: Surface scattering (co-pol dominant areas)
            - Green: Volume scattering (cross-pol, vegetation/ice)
            - Blue: Surface scattering with low volume
            """
            # Get stack variable from first polarization
            stackvar = list(sample[pol1].dims)[0] if len(sample[pol1].dims) > 2 else None

            # Calculate decimation factors
            batch = self[[pol1, pol2]]._start_from(self)
            if stackvar is not None:
                batch = batch.isel({stackvar: slice(0, rows*cols)})
            y_coords = np.concatenate([np.asarray(ds[pol1].y) for ds in batch.values()])
            x_coords = np.concatenate([np.asarray(ds[pol1].x) for ds in batch.values()])
            dy, dx = self.spacing
            size_y = int((y_coords.max() - y_coords.min()) / abs(dy)) + 1
            size_x = int((x_coords.max() - x_coords.min()) / abs(dx)) + 1
            factor_y = max(1, int(np.round(size_y / (extent[1] / rows))))
            factor_x = max(1, int(np.round(size_x / (extent[0] / cols))))

            # Decimate and convert to dataset
            batch_decimated = batch.isel(y=slice(None, None, factor_y), x=slice(None, None, factor_x))
            ds_merged = batch_decimated.to_dataset()
            da_copol = ds_merged[pol1]   # Co-pol (HH or VV)
            da_xpol = ds_merged[pol2]    # Cross-pol (HV or VH)

            if stackvar is None:
                stackvar = 'fake'
                da_copol = da_copol.expand_dims({stackvar: [0]})
                da_xpol = da_xpol.expand_dims({stackvar: [0]})

            # Materialize both polarizations together
            import dask
            da_copol, da_xpol = dask.persist(da_copol, da_xpol)
            from .utils_dask import progress_persisted
            progress_persisted([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

            # Compute RGB using shared method from Batch
            from .Batch import Batch
            copol = da_copol.values
            xpol = da_xpol.values
            rgb_float = Batch._compute_rgb(copol, xpol, gamma=gamma,
                                           brightness=brightness, quantile=quantile)

            # Add alpha channel for transparent NaN pixels
            nan_mask = ~np.isfinite(copol) | ~np.isfinite(xpol)
            alpha_channel = np.where(nan_mask, 0.0, 1.0)
            rgba_array = np.concatenate([rgb_float, alpha_channel[..., np.newaxis]], axis=-1)

            # Create figure with subplots
            n_panels = rgba_array.shape[0]
            n_cols = min(cols, n_panels)
            n_rows = int(np.ceil(n_panels / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(size * n_cols * aspect, size * n_rows),
                                     squeeze=False)
            axes = axes.flatten()

            # Get coordinate extent for imshow
            y_min, y_max = float(da_copol.y.min()), float(da_copol.y.max())
            x_min, x_max = float(da_copol.x.min()), float(da_copol.x.max())
            img_extent = [x_min, x_max, y_max, y_min] if flip else [x_min, x_max, y_min, y_max]

            km_formatter = FuncFormatter(lambda v, _: f'{v/1000:.0f}')

            for idx in range(len(axes)):
                ax = axes[idx]
                if idx < n_panels:
                    ax.imshow(rgba_array[idx], extent=img_extent, aspect='auto',
                              origin='upper' if flip else 'lower', alpha=alpha)
                    ax.xaxis.set_major_formatter(km_formatter)
                    ax.yaxis.set_major_formatter(km_formatter)
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins))
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins))
                    ax.set_xlabel('easting [km]')
                    ax.set_ylabel('northing [km]')

                    # Set title
                    if stackvar == 'fake':
                        ax.set_title('')
                    elif stackvar in ('pair', 'date') and idx < da_copol[stackvar].size:
                        if stackvar == 'pair':
                            if 'ref' in da_copol.coords and 'rep' in da_copol.coords:
                                ref_val = da_copol.coords['ref'].values[idx]
                                rep_val = da_copol.coords['rep'].values[idx]
                                ref_str = pd.Timestamp(ref_val).strftime('%Y-%m-%d')
                                rep_str = pd.Timestamp(rep_val).strftime('%Y-%m-%d')
                                ax.set_title(f'{ref_str} {rep_str}')
                            else:
                                ax.set_title(f'pair={idx}')
                        elif stackvar == 'date':
                            coord_val = da_copol[stackvar].values[idx]
                            if hasattr(coord_val, 'strftime'):
                                ax.set_title(f'date={coord_val.strftime("%Y-%m-%d")}')
                            else:
                                ax.set_title(f'date={pd.Timestamp(coord_val).strftime("%Y-%m-%d")}')
                else:
                    ax.set_visible(False)

            # Add suptitle with RGB assignment
            fig.suptitle(f'{caption or "RGB Composite"} (R={pol1}, G={pol2}, B={pol1})', y=y)
            plt.tight_layout()

            return fig

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        sample = next(iter(self.values()))
        # Only plot spatial variables (with y, x dims), skip 1D coords like ref/rep/BPR
        polarizations = [v for v in sample.data_vars
                         if sample[v].ndim >= 2 and sample[v].dims[-2:] == ('y', 'x')]
        #print ('polarizations', polarizations)

        # Handle composite mode
        if composite:
            if len(polarizations) < 2:
                raise ValueError(f"Composite mode requires exactly 2 polarizations, found {len(polarizations)}: {polarizations}")
            if len(polarizations) > 2:
                raise ValueError(f"Composite mode requires exactly 2 polarizations, found {len(polarizations)}: {polarizations}. "
                               "Pre-select polarizations using batch[[pol1, pol2]].plot(composite=True)")

            pol1, pol2 = polarizations[0], polarizations[1]
            fg = plot_composite(pol1, pol2)
            return [fg]

        # process polarizations one by one
        fgs = []
        for pol in polarizations:
            fg = plot_polarization(polarization=pol)
            fgs.append(fg)
        return fgs

    def gaussian(
        self,
        wavelength: float,
        weight: BatchUnit | None = None,
        threshold: float = 0.5,
        device: str = 'auto',
        debug: bool = False
    ) -> Batch:
        """
        2D (yx) Gaussian kernel smoothing (multilook) on each dataset in this Batch.

        Parameters
        ----------
        wavelength : float
            The filter, in metres: the sigma follows from it by the 5.3 cutoff
            formula. FIRST AND REQUIRED, as multilook() takes it -- it is the
            only argument that decides what this does, and it used to sit
            second behind an optional weight, so `gaussian(200)` bound 200 to
            the weight and every caller had to write `wavelength=`. Omitted, it
            once left sigma None and the kernel returned the input untouched:
            a filter that silently did nothing, by default.
        weight : BatchUnit or None
            A Batch of 2D DataArrays, one per key, matching this Batch's keys.
            If None, no weighting is applied.
        threshold : float
            Drop-off threshold for the kernel.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool
            Print sigma values if True.

        Returns
        -------
        Batch
            A new Batch with the same keys, each smoothed by its corresponding weight.
        """
        import xarray as xr
        import numpy as np
        from .Batch import BatchUnit
        # constant 5.3 defines half-gain at filter_wavelength
        cutoff = 5.3

        # A WEIGHT WHERE THE WAVELENGTH GOES is the old argument order, and it
        # would otherwise reach the `wavelength <= 0` test and fail there with
        # something unrelated to what the caller wrote.
        if isinstance(wavelength, BatchUnit):
            raise TypeError(
                'gaussian() takes the wavelength first now, as multilook() does. '
                'Pass the weight by name: gaussian(wavelength, weight=...)')

        # validate weight if provided: a BatchUnit (N81), a DataArray one weighting every grid,
        # a Dataset one naming every grid; a burst or a grid it lacks raises (_weight)
        weight = BatchCore._weight(weight, self)

        # Validate lazy data
        BatchCore._require_lazy(self, 'gaussian')

        # precompute pixel sizes for decimation
        dy, dx = self.spacing

        # REQUIRED, NOT OPTIONAL. It used to accept None and pass sigma=None to
        # the kernel, whose first line is `if sigma is None: return data_np` --
        # so the filter returned its input untouched and said nothing.
        if wavelength is None or wavelength <= 0:
            raise ValueError(
                f'gaussian() needs a positive wavelength in metres, got {wavelength!r}')
        sig_y = wavelength / (dy * cutoff)
        sig_x = wavelength / (dx * cutoff)
        if debug:
            print(f'DEBUG: multilooking sigmas ({sig_y:.2f}, {sig_x:.2f}), wavelength {wavelength:.1f}')
        sigmas = (sig_y, sig_x)

        import dask.array as da

        # Resolve device ONCE here, not in every task
        # This avoids repeated get_client()/scheduler_info() calls inside workers
        if device == 'auto':
            resolved_device = BatchCore._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

        out = {}
        # loop over each key
        for key, ds in self.items():
            # weight is BatchUnit (dict of Datasets) - get Dataset for this burst
            w = weight[key] if weight is not None else None

            new_vars = {}
            # a Dataset's variables, or a DataArray as its own one (N81)
            for var, data_arr in BatchCore._vars_of(ds).items():
                # Non-spatial variables are CARRIED, not dropped. They are the
                # radar metadata -- radar_wavelength, near_range, earth_radius,
                # SC_height_start, rng_samp_rate, BPR -- and dropping them here
                # stranded every downstream unit conversion and ele2phase build.
                # Smoothing does not apply to them; passing them through does.
                if not (data_arr.ndim in (2, 3) and data_arr.dims[-2:] == ('y', 'x')):
                    new_vars[var] = data_arr
                    continue

                is_complex = np.issubdtype(data_arr.dtype, np.complexfloating)
                out_dtype = np.complex64 if is_complex else np.float32

                # Get weight dask array for this variable: the grid of the same name (_weight)
                weight_dask = BatchCore._weight_of(w, var).data if w is not None else None

                # Ensure first dimension chunked as 1 for per-item spatial processing
                dask_data = data_arr.data
                if data_arr.ndim == 3 and dask_data.chunks[0][0] != 1:
                    dask_data = dask_data.rechunk({0: 1})

                # Calculate overlap depth from sigmas (truncate=4.0 is used in gaussian_numpy)
                truncate = 4.0
                depth_y = int(np.ceil(sigmas[0] * truncate)) if sigmas is not None else 0
                depth_x = int(np.ceil(sigmas[1] * truncate)) if sigmas is not None else 0

                if debug:
                    if data_arr.ndim == 3:
                        _nc = (len(dask_data.chunks[1]), len(dask_data.chunks[2]))
                    else:
                        _nc = (len(dask_data.chunks[0]), len(dask_data.chunks[1]))
                    print(f'DEBUG: gaussian depth=({depth_y}, {depth_x}), n_chunks={_nc}')

                if data_arr.ndim == 3:
                    depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                    depth_2d = {0: depth_y, 1: depth_x}
                    if weight_dask is not None and weight_dask.ndim == 2:
                        # 3D data with 2D weight: loop approach (process each date separately)
                        if weight_dask.chunks != dask_data[0].chunks:
                            raise ValueError(
                                f"gaussian() weight chunks {weight_dask.chunks} "
                                f"must match data chunks {dask_data[0].chunks}")
                        slices = []
                        for i in range(dask_data.shape[0]):
                            result_slice = da.map_overlap(
                                _apply_gaussian_2d_for_dask,
                                dask_data[i], weight_dask,
                                depth=depth_2d, boundary='none',
                                dtype=out_dtype,
                                sigmas=sigmas, threshold=threshold,
                                device=device, pixel_sizes=(dy, dx),
                                out_dtype=out_dtype)
                            slices.append(result_slice)
                        result_dask = da.stack(slices, axis=0)
                    elif weight_dask is not None:
                        # 3D data with 3D weight: require matching shape/chunks
                        if weight_dask.shape != dask_data.shape:
                            raise ValueError(
                                f"gaussian() weight shape {weight_dask.shape} "
                                f"must match data shape {dask_data.shape}")
                        if weight_dask.chunks != dask_data.chunks:
                            raise ValueError(
                                f"gaussian() weight chunks {weight_dask.chunks} "
                                f"must match data chunks {dask_data.chunks}")
                        result_dask = da.map_overlap(
                            _apply_gaussian_2d_for_dask,
                            dask_data, weight_dask,
                            depth=depth_3d, boundary='none',
                            dtype=out_dtype,
                            sigmas=sigmas, threshold=threshold,
                            device=device, pixel_sizes=(dy, dx),
                            out_dtype=out_dtype)
                    else:
                        # No weight: single map_overlap on 3D data
                        result_dask = da.map_overlap(
                            _apply_gaussian_2d_for_dask,
                            dask_data,
                            depth=depth_3d, boundary='none',
                            dtype=out_dtype, weight_block=None,
                            sigmas=sigmas, threshold=threshold,
                            device=device, pixel_sizes=(dy, dx),
                            out_dtype=out_dtype)
                else:
                    # 2D data (y, x)
                    depth_2d = {0: depth_y, 1: depth_x}
                    if weight_dask is not None:
                        if weight_dask.chunks != dask_data.chunks:
                            raise ValueError(
                                f"gaussian() weight chunks {weight_dask.chunks} "
                                f"must match data chunks {dask_data.chunks}")
                        result_dask = da.map_overlap(
                            _apply_gaussian_2d_for_dask,
                            dask_data, weight_dask,
                            depth=depth_2d, boundary='none',
                            dtype=out_dtype,
                            sigmas=sigmas, threshold=threshold,
                            device=device, pixel_sizes=(dy, dx),
                            out_dtype=out_dtype)
                    else:
                        result_dask = da.map_overlap(
                            _apply_gaussian_2d_for_dask,
                            dask_data,
                            depth=depth_2d, boundary='none',
                            dtype=out_dtype, weight_block=None,
                            sigmas=sigmas, threshold=threshold,
                            device=device, pixel_sizes=(dy, dx),
                            out_dtype=out_dtype)

                new_vars[var] = xr.DataArray(
                    result_dask,
                    dims=data_arr.dims,
                    coords=data_arr.coords
                )

            if isinstance(ds, xr.DataArray):
                # a DataArray in, a DataArray out (N81)
                res = new_vars[ds.name].rename(ds.name)
                res.attrs = ds.attrs
                out[key] = res
                continue
            new_ds = xr.Dataset(new_vars)
            new_ds.attrs = ds.attrs
            out[key] = new_ds

        return type(self)(out)

    @staticmethod
    def _overlap_residual(diff, circular: bool):
        """One burst overlap's term of residuals(): (|median phase difference|, valid pixel count, median).

        `diff` is the overlap's phase difference, burst 2 minus burst 1; `circular` wraps it and its median to
        [-pi, pi) first. None when no pixel is valid. align() takes the same term from the overlaps its solve
        already holds, so its 'residual' is this measure without a second pass over the grids.
        """
        wrap = (lambda x: (x + np.pi) % (2*np.pi) - np.pi) if circular else (lambda x: x)
        valid = np.asarray(diff).ravel()
        valid = valid[np.isfinite(valid)]
        if len(valid) == 0:
            return None
        median_diff = np.median(wrap(valid))
        return np.abs(wrap(median_diff)), len(valid), median_diff

    @staticmethod
    def _residual_mean(terms, n_pairs: int) -> list:
        """residuals() per pair from (pair index, |median|, weight) overlap terms, in overlap order: the weighted
        mean rounded to 3 decimals, 0.0 for a pair without a valid overlap."""
        sums = [0.0] * n_pairs
        weights = [0.0] * n_pairs
        for p, value, weight in terms:
            sums[p] += value * weight
            weights[p] += weight
        return [0.0 if weights[p] == 0 else round(sums[p] / weights[p], 3) for p in range(n_pairs)]

    def residuals(self, polarization: str | None = None, debug: bool = False) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with align(),
        these median differences should be close to zero.

        Parameters
        ----------
        polarization : str, optional
            Polarization to use for residual computation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
        debug : bool, optional
            Print debug information for each overlap. Default is False.

        Returns
        -------
        float or list[float]
            Single pair: Weighted mean absolute median phase discrepancy in radians.
            Multiple pairs: List of discrepancies, one per pair.
            0.0 = perfect alignment, π = maximum discrepancy (for wrapped phase).

            Practical interpretation:

            - < 0.1 rad: Excellent alignment
            - 0.1 - 0.5 rad: Good alignment
            - 0.5 - 1.0 rad: Moderate misalignment
            - > 1.0 rad: Poor alignment

        Examples
        --------
        >>> # Compare before and after alignment
        >>> before = intfs.residuals()
        >>> aligned = intfs.align()
        >>> after = aligned.residuals()
        >>> print(f'Discrepancy reduced from {before} to {after}')
        """
        from .Batch import Batch, BatchWrap
        import dask

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"residuals() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        # Collect burst extents and detect pair dimension
        ids = sorted(self.keys())

        # Auto-detect polarization if not specified
        # (a Dataset's variables, or a DataArray as its own one, N81)
        sample_vars = BatchCore._vars_of(self[ids[0]])
        # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
        available_pols = [v for v, a in sample_vars.items()
                         if 'y' in a.dims and 'x' in a.dims]
        if polarization is None:
            polarization = available_pols[0]
        if polarization not in available_pols:
            raise ValueError(f"Polarization '{polarization}' not found. Available: {available_pols}")

        sample_da = sample_vars[polarization]
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        # Extract pathNumber and subswath from burst ID (format: "123_262883_IW2")
        burst_subswath = {}
        burst_track = {}  # pathNumber + subswath for detailed debug output
        for bid in ids:
            parts = bid.split('_')
            if len(parts) < 3:
                raise ValueError(f"Burst '{bid}' has invalid format, expected 'pathNumber_burstNumber_subswath'")
            path_num = parts[0]
            subswath = parts[2]
            burst_subswath[bid] = subswath
            burst_track[bid] = f"{path_num}{subswath}"

        extents = {}
        for bid in ids:
            ds = self[bid]
            da = BatchCore._vars_of(ds)[polarization]
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            # Get coordinates from Dataset if not on DataArray
            y_coords = da.coords['y'].values if 'y' in da.coords else ds.coords['y'].values
            x_coords = da.coords['x'].values if 'x' in da.coords else ds.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        # Find all overlapping burst pairs
        overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                e2 = extents[id2]
                if extents_overlap(e1, e2):
                    overlap_pairs.append((id1, id2))

        if not overlap_pairs:
            return [0.0] * n_pairs if has_pair_dim else 0.0

        if debug:
            print(f'residuals: found {len(overlap_pairs)} overlap pairs, {n_pairs} pair(s)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in overlap_pairs:
            i1 = BatchCore._vars_of(self[id1])[polarization]
            i2 = BatchCore._vars_of(self[id2])[polarization]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff)

        # Compute all phase differences at once - dask schedules efficiently
        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        for (id1, id2, pair_idx), phase_diff in zip(jobs, computed_diffs):
            term = BatchCore._overlap_residual(phase_diff.values, use_circular)
            if term is None:
                continue
            abs_discrepancy, weight, median_diff = term
            results.append((pair_idx, abs_discrepancy, weight, id1, id2, median_diff))

        # Per-subswath tracking for debug
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
        per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            per_overlap_discrepancies[pair_idx].append(abs_discrepancy)

            # Extract track info for debug stats
            if debug:
                track1 = burst_track[id1]
                track2 = burst_track[id2]

                # Categorize: same track or cross-track
                if track1 == track2:
                    track_key = track1
                else:
                    track_key = f'{track1}-{track2}'

                key = (track_key, pair_idx)
                if key not in subswath_stats:
                    subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0, 'values': []}
                subswath_stats[key]['sum'] += abs_discrepancy * weight
                subswath_stats[key]['weight'] += weight
                subswath_stats[key]['count'] += 1
                subswath_stats[key]['values'].append(abs_discrepancy)

        discrepancies = BatchCore._residual_mean([(r[0], r[1], r[2]) for r in results], n_pairs)

        if debug:
            # Compute std for overall discrepancy
            for p in range(n_pairs):
                vals = per_overlap_discrepancies[p]
                if len(vals) > 1:
                    std = np.std(vals)
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} ± {std:.3f} rad ({len(vals)} overlaps)', flush=True)
                else:
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} rad ({len(vals)} overlaps)', flush=True)

            # Print per-track stats (only for pair_idx=0 to avoid clutter)
            print('Per-track discrepancy (pair 0):', flush=True)
            for (track, pair_idx), stats in sorted(subswath_stats.items()):
                if pair_idx == 0 and stats['weight'] > 0:
                    track_disc = stats['sum'] / stats['weight']
                    vals = stats['values']
                    if len(vals) > 1:
                        track_std = np.std(vals)
                        print(f'  {track}: {track_disc:.3f} ± {track_std:.3f} rad ({stats["count"]} overlaps)', flush=True)
                    else:
                        print(f'  {track}: {track_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)

        # Return single value for single pair, list for multiple
        if n_pairs == 1 and not has_pair_dim:
            return discrepancies[0]
        return discrepancies

    def _align_coeffs(self,
            degree: int = 0,
            method: str = 'median',
            polarization: str | None = None,
            debug: bool = False,
            return_residuals: bool = False,
            lazy_residual: bool = False):
        """
        Estimate per-burst polynomial coefficients using overlap-based least-squares.

        Fits polynomial corrections (offset or offset+ramp) to each burst by analyzing
        phase differences in overlapping regions. Uses global least-squares optimization
        to find consistent coefficients across all bursts.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree:
            - 0 (default): Estimate offsets only.
            - 1: Estimate linear ramp (in x/range direction).
        method : str, optional
            Estimation method: 'median' (robust) or 'mean' (faster).
        polarization : str, optional
            Polarization to use for coefficient estimation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return input residuals (before correction). Default is False.
        lazy_residual : bool, optional
            degree=0 only. If True, also return the residual AFTER subtracting the offsets, by the measure of
            residuals(), as a lazy float32 dask array (pair,) (0-d without a pair dimension). The solve task
            takes it from the overlaps it already holds: no second pass over the grids. Default is False.

        Returns
        -------
        dict or tuple
            If return_residuals is False:
                For single pair (no pair dimension):
                    degree=0: {burst_id: offset}
                    degree=1: {burst_id: [ramp, intercept]}
                For multiple pairs:
                    degree=0: {burst_id: [offset_pair0, offset_pair1, ...]}
                    degree=1: {burst_id: [[ramp0, intercept0], [ramp1, intercept1], ...]}
            If return_residuals is True:
                (coefficients_dict, residuals) where residuals is float or list[float]
            If lazy_residual is True, the lazy residual after the correction is appended last:
                (coefficients_dict, residual) or (coefficients_dict, residuals, residual)

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = intfs._align_coeffs(degree=0)
        >>> intfs1 = intfs - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = intfs1._align_coeffs(degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = intfs2._align_coeffs(degree=0)
        >>> # Combine coefficients (for single pair)
        >>> coeffs = {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
        >>> aligned = intfs - intfs.polyval(coeffs)
        """
        from .Batch import Batch, BatchWrap
        import dask
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"_align_coeffs() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")
        if lazy_residual and degree != 0:
            raise ValueError(f'_align_coeffs(): lazy_residual needs degree=0, got degree={degree}')

        # Constants
        MIN_OVERLAP_PIXELS = 50
        MIN_ROW_PIXELS = 10
        MIN_VALID_ROWS = 5
        MIN_INLIER_SAMPLES = 10
        MAD_OUTLIER_THRESHOLD = 2.5
        OUTPUT_PRECISION = 3
        RAMP_PRECISION = 9

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        def phase_diff(a, center):
            """Circular or linear difference from center."""
            if use_circular:
                return maybe_wrap(a - center)
            return a - center

        def phase_mean(a):
            """Circular or linear mean."""
            if use_circular:
                return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
            return np.mean(a)

        def phase_mad(a, center):
            """Circular or linear MAD."""
            return np.median(np.abs(phase_diff(a, center)))

        # Collect burst extents and x-centers
        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Auto-detect polarization if not specified
        sample_ds = self[ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
        available_pols = [v for v in sample_ds.data_vars
                         if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
        if polarization is None:
            polarization = available_pols[0]
        if polarization not in available_pols:
            raise ValueError(f"Polarization '{polarization}' not found. Available: {available_pols}")

        # Detect number of pairs
        sample_da = sample_ds[polarization]
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        if debug:
            print(f'_align_coeffs(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s), pol={polarization}', flush=True)

        # Extract pathNumber and subswath from burst ID (format: "123_262883_IW2")
        # Used to skip same-path different-subswath overlaps (small x-extent, diagonal connection)
        # but allow cross-path overlaps which can have large x-extent with significant iono ramps
        burst_path = {}  # pathNumber (e.g., '33')
        burst_subswath = {}  # subswath (e.g., 'IW3')
        for bid in ids:
            if degree == 1:
                parts = bid.split('_')
                if len(parts) < 3:
                    raise ValueError(f"Burst '{bid}' has invalid format, expected 'pathNumber_burstNumber_subswath'")
                burst_path[bid] = parts[0]
                burst_subswath[bid] = parts[2]

        extents = {}
        x_centers = {}

        for bid in ids:
            ds = self[bid]
            da = ds[polarization]
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            # Get coordinates from Dataset if not on DataArray
            y_coords = da.coords['y'].values if 'y' in da.coords else ds.coords['y'].values
            x_coords = da.coords['x'].values if 'x' in da.coords else ds.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())
            x_centers[bid] = float(np.mean(x_coords))

        # Detect coordinate ordering for .sel() slicing
        _sample_y = self[ids[0]].coords['y'].values
        _y_descending = len(_sample_y) > 1 and _sample_y[0] > _sample_y[-1]

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        def process_phase_diff(diff_np, x_coords, id1, id2, pair_idx):
            """Process overlap numpy array to extract offset and optionally ramp.

            Parameters
            ----------
            diff_np : numpy.ndarray
                2D array of phase differences in the overlap region.
            x_coords : numpy.ndarray
                1D array of x coordinate values for columns.
            """
            all_valid = diff_np.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]

            if len(all_valid) < MIN_OVERLAP_PIXELS:
                return None

            if diff_np.ndim < 2:
                return None

            # Row-wise processing
            row_phases = []
            row_x_centroids = []
            row_weights = []

            for y_idx in range(diff_np.shape[0]):
                row = diff_np[y_idx, :]
                valid_mask = np.isfinite(row)
                n_valid = np.sum(valid_mask)
                if n_valid >= MIN_ROW_PIXELS:
                    x_valid = x_coords[valid_mask]
                    phase_valid = row[valid_mask]
                    # Unwrap for row mean computation (needed for both circular and linear)
                    if use_circular:
                        phase_unwrapped = np.unwrap(phase_valid)
                        row_mean = maybe_wrap(np.mean(phase_unwrapped))
                    else:
                        row_mean = np.mean(phase_valid)
                    row_phases.append(row_mean)
                    row_x_centroids.append(np.mean(x_valid))
                    row_weights.append(n_valid)

            if len(row_phases) < MIN_VALID_ROWS:
                return None

            a = np.array(row_phases)
            x_row = np.array(row_x_centroids)
            weights = np.array(row_weights)
            a = maybe_wrap(a)

            # Outlier rejection
            if method == 'median':
                offset_initial = np.median(a)
                mad = phase_mad(a, offset_initial)
                if mad > 0:
                    inliers = np.abs(phase_diff(a, offset_initial)) <= MAD_OUTLIER_THRESHOLD * mad
                    if np.sum(inliers) >= MIN_INLIER_SAMPLES:
                        a = a[inliers]
                        x_row = x_row[inliers]
                        weights = weights[inliers]

            n_valid = int(np.sum(weights))
            x_centroid = float(np.average(x_row, weights=weights))

            # Compute offset
            if method == 'median':
                sorted_idx = np.argsort(a)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                offset = a[sorted_idx[median_idx]]
            else:
                offset = phase_mean(a)

            # Compute ramp if degree=1
            ramp_val = None
            if degree == 1 and len(a) >= MIN_VALID_ROWS:
                x_centered = x_row - x_centroid
                x_range = np.max(x_row) - np.min(x_row)
                if x_range > 100:
                    residuals = a - offset
                    Swxx = np.sum(weights * x_centered**2)
                    Swxr = np.sum(weights * x_centered * residuals)
                    if Swxx > 1e-10:
                        ramp_val = Swxr / Swxx

            return (id1, id2, pair_idx, maybe_wrap(offset), ramp_val, x_centroid, n_valid)

        # Find overlapping burst pairs
        # For degree=1 (ramp), skip same-path cross-subswath overlaps (small x-extent, diagonal)
        # but allow cross-path overlaps which have large x-extent with significant iono ramps
        # For degree=0 (offset), use all overlaps including cross-subswath
        all_overlap_pairs = []
        cross_subswath_skipped = 0
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if extents_overlap(e1, extents[id2]):
                    if degree == 1:
                        # For ramp estimation, skip same-path cross-subswath overlaps (diagonal, small x-extent)
                        # but allow cross-path overlaps - they have large x-extent with iono ramp differences
                        path1, path2 = burst_path[id1], burst_path[id2]
                        sw1, sw2 = burst_subswath[id1], burst_subswath[id2]
                        if path1 == path2 and sw1 != sw2:
                            # Same path, different subswath: diagonal overlap, skip
                            cross_subswath_skipped += 1
                            continue
                        # Same path + same subswath (along-track) or different paths: allow
                    all_overlap_pairs.append((id1, id2))

        if debug:
            print(f'Found {len(all_overlap_pairs)} overlapping burst pairs', flush=True)
            if degree == 1 and cross_subswath_skipped > 0:
                print(f'  (skipped {cross_subswath_skipped} same-path cross-subswath pairs for ramp estimation)', flush=True)

        # Pass raw burst data arrays (not pre-computed diffs) to a single
        # delayed task. This creates N_bursts graph dependencies instead of
        # N_overlaps*3 layers from xarray diff operations, keeping the graph
        # minimal for downstream dissolve().
        import dask.array as _da

        # Collect burst data + coordinates (coordinates are numpy, not dask)
        burst_data = [self[bid][polarization].data for bid in ids]
        burst_y = [self[bid][polarization].y.values for bid in ids]
        burst_x = [self[bid][polarization].x.values for bid in ids]

        if debug:
            print(f'Building lazy graph for {len(all_overlap_pairs)} overlap pairs, {len(ids)} bursts...', flush=True)

        # Single delayed task: receives resolved burst numpy arrays,
        # computes overlaps + diffs internally, then solves.
        def _align_coeffs_all(*burst_data_arrays):
            import xarray as xr
            from scipy import sparse as _sparse
            from scipy.sparse.linalg import lsqr as _lsqr
            from scipy.sparse.csgraph import connected_components as _cc

            def _overlap_diff(i1_idx, i2_idx, pair_idx):
                """The overlap's phase difference, burst 2 minus burst 1, on the common coordinates."""
                d1 = np.asarray(burst_data_arrays[i1_idx])
                d2 = np.asarray(burst_data_arrays[i2_idx])
                d1_p = d1[pair_idx] if has_pair_dim else d1
                d2_p = d2[pair_idx] if has_pair_dim else d2

                # Build xarray DataArrays for coordinate-aware overlap
                da1 = xr.DataArray(d1_p, dims=['y', 'x'],
                                   coords={'y': burst_y[i1_idx],
                                           'x': burst_x[i1_idx]})
                da2 = xr.DataArray(d2_p, dims=['y', 'x'],
                                   coords={'y': burst_y[i2_idx],
                                           'x': burst_x[i2_idx]})
                return da2 - da1

            # Compute overlap diffs and process statistics
            _pbp = {p: [] for p in range(n_pairs)}
            for id1, id2 in all_overlap_pairs:
                i1_idx = id_to_idx[id1]
                i2_idx = id_to_idx[id2]

                for pair_idx in range(n_pairs):
                    diff = _overlap_diff(i1_idx, i2_idx, pair_idx)
                    stat = process_phase_diff(diff.values,
                                              diff.coords['x'].values,
                                              id1, id2, pair_idx)
                    if stat is None:
                        continue
                    _id1s, _id2s, _pidxs, _off, _rv, _xcent, _nu = stat
                    _w = np.sqrt(_nu)
                    if degree == 0:
                        _pbp[_pidxs].append((_id1s, _id2s, _off, _w))
                    else:
                        if _rv is not None:
                            _pbp[_pidxs].append((_id1s, _id2s, _off, _rv, _xcent, _w))

            def _solve_one(pidx):
                pairs = _pbp[pidx]
                if len(pairs) == 0:
                    if degree == 0:
                        return {bid: np.float32(0.0) for bid in ids}
                    else:
                        return {bid: [np.float32(0.0), np.float32(0.0)] for bid in ids}

                adj = _sparse.lil_matrix((n_bursts, n_bursts))
                for p in pairs:
                    adj[id_to_idx[p[0]], id_to_idx[p[1]]] = 1
                    adj[id_to_idx[p[1]], id_to_idx[p[0]]] = 1
                n_comp_all, labels = _cc(adj.tocsr(), directed=False)

                if degree == 0:
                    out = {}
                    for comp in range(n_comp_all):
                        ci = np.where(labels == comp)[0]
                        cids = [ids[ii] for ii in ci]
                        cmap = {bid: ii for ii, bid in enumerate(cids)}
                        nc = len(cids)
                        if nc == 1:
                            out[cids[0]] = np.float32(0.0)
                            continue
                        cp = [(a, b, o, w) for a, b, o, w in pairs
                              if a in cmap and b in cmap]
                        if not cp:
                            for bid in cids:
                                out[bid] = np.float32(0.0)
                            continue
                        ncp = len(cp)
                        Am = _sparse.lil_matrix((ncp + 1, nc))
                        bv = np.zeros(ncp + 1)
                        Wv = np.zeros(ncp + 1)
                        for kk, (a, b, o, w) in enumerate(cp):
                            Am[kk, cmap[a]] = -1
                            Am[kk, cmap[b]] = +1
                            bv[kk] = o
                            Wv[kk] = w
                        cw = np.sum(Wv[:-1]) * 100 if np.sum(Wv[:-1]) > 0 else 1e6
                        Am[ncp, 0] = 1
                        Wv[ncp] = cw
                        sqW = np.sqrt(Wv)
                        res = _lsqr(_sparse.diags(sqW) @ Am.tocsr(), sqW * bv)
                        for ii, bid in enumerate(cids):
                            out[bid] = np.float32(round(float(maybe_wrap(res[0][ii])),
                                                        OUTPUT_PRECISION))
                    return out
                else:
                    out = {}
                    for comp in range(n_comp_all):
                        ci = np.where(labels == comp)[0]
                        cids = [ids[ii] for ii in ci]
                        cmap = {bid: ii for ii, bid in enumerate(cids)}
                        nc = len(cids)
                        if nc == 1:
                            out[cids[0]] = [np.float32(0.0), np.float32(0.0)]
                            continue
                        cp = [(a, b, o, r, xc, w) for a, b, o, r, xc, w in pairs
                              if a in cmap and b in cmap]
                        if not cp:
                            for bid in cids:
                                out[bid] = [np.float32(0.0), np.float32(0.0)]
                            continue
                        ncp = len(cp)
                        Am = _sparse.lil_matrix((ncp + 1, nc))
                        bv = np.zeros(ncp + 1)
                        Wv = np.zeros(ncp + 1)
                        for kk, (a, b, o, rd, xc, w) in enumerate(cp):
                            Am[kk, cmap[a]] = -1
                            Am[kk, cmap[b]] = +1
                            bv[kk] = rd
                            Wv[kk] = w
                        cw = np.sum(Wv[:-1]) * 100 if np.sum(Wv[:-1]) > 0 else 1e6
                        Am[ncp, 0] = 1
                        Wv[ncp] = cw
                        sqW = np.sqrt(Wv)
                        res = _lsqr(_sparse.diags(sqW) @ Am.tocsr(), sqW * bv)
                        for ii, bid in enumerate(cids):
                            ramp = np.float32(round(float(res[0][ii]), RAMP_PRECISION))
                            intercept = np.float32(round(-ramp * x_centers[bid],
                                                         OUTPUT_PRECISION))
                            out[bid] = [ramp, intercept]
                    return out

            rpp = [_solve_one(p) for p in range(n_pairs)]

            # Format output
            if n_pairs == 1 and not has_pair_dim:
                offsets = rpp[0]
            else:
                offsets = {bid: [rpp[p][bid] for p in range(n_pairs)] for bid in ids}

            # Residuals (if requested)
            residuals = None
            if return_residuals:
                disc = []
                for p in range(n_pairs):
                    pp = _pbp[p]
                    if not pp:
                        disc.append(0.0)
                        continue
                    if degree == 0:
                        offs = [abs(maybe_wrap(t[2])) for t in pp]
                        ws = [t[3] for t in pp]
                    else:
                        offs = [abs(maybe_wrap(t[2])) for t in pp]
                        ws = [t[5] for t in pp]
                    tw = sum(ws)
                    disc.append(round(sum(o * w for o, w in zip(offs, ws)) / tw, 3)
                                if tw > 0 else 0.0)
                residuals = disc[0] if (n_pairs == 1 and not has_pair_dim) else disc

            # Residual after the correction (degree 0): residuals()' measure on the
            # overlaps this task already holds, each shifted by its offset difference
            residual_after = None
            if lazy_residual:
                terms = []
                for id1, id2 in all_overlap_pairs:
                    for pair_idx in range(n_pairs):
                        diff = _overlap_diff(id_to_idx[id1], id_to_idx[id2], pair_idx)
                        shift = rpp[pair_idx][id2] - rpp[pair_idx][id1]
                        term = BatchCore._overlap_residual(diff.values - shift, use_circular)
                        if term is not None:
                            terms.append((pair_idx, term[0], term[1]))
                disc = BatchCore._residual_mean(terms, n_pairs)
                residual_after = np.asarray(disc if has_pair_dim else disc[0], dtype=np.float32)

            return {'offsets': offsets, 'residuals': residuals, 'residual_after': residual_after}

        # Single delayed call — dask resolves burst data arrays before calling.
        # Graph has ~N_bursts layers (not ~N_overlaps*3 from xarray diffs).
        # A UNIQUE KEY PER CALL (pure=False, N81): the arrays reach the task through
        # finalize keys that dask names anew in every graph, so a pure key named the
        # same task over different dependencies in two consecutive computes. When the
        # second graph reached the scheduler before the first one's release, the
        # scheduler kept the stale task (dask issue 9888) and the compute failed with
        # KeyError ('held-sub-sub-...') or an AssertionError on a TaskState.
        solve_result = dask.delayed(_align_coeffs_all, pure=False)(*burst_data)

        # Extract per-burst dask 0-d arrays from delayed solve result
        offsets_part = solve_result['offsets']

        if n_pairs == 1 and not has_pair_dim:
            if degree == 0:
                coeffs = {bid: _da.from_delayed(offsets_part[bid],
                          shape=(), dtype=np.float32) for bid in ids}
            else:
                coeffs = {bid: [
                    _da.from_delayed(offsets_part[bid][0], shape=(), dtype=np.float32),
                    _da.from_delayed(offsets_part[bid][1], shape=(), dtype=np.float32),
                ] for bid in ids}
        else:
            if degree == 0:
                coeffs = {bid: [
                    _da.from_delayed(offsets_part[bid][p], shape=(), dtype=np.float32)
                    for p in range(n_pairs)
                ] for bid in ids}
            else:
                coeffs = {bid: [
                    [_da.from_delayed(offsets_part[bid][p][0], shape=(), dtype=np.float32),
                     _da.from_delayed(offsets_part[bid][p][1], shape=(), dtype=np.float32)]
                    for p in range(n_pairs)
                ] for bid in ids}

        out = (coeffs,)
        if return_residuals:
            # Residuals require concrete values — triggers the solve chain
            print('_align_coeffs(return_residuals=True): computing residuals breaks lazy chain, use for diagnostics only', flush=True)
            residuals_out = solve_result['residuals'].compute()
            if debug:
                print(f'Input residuals: {residuals_out}', flush=True)
            out += (residuals_out,)
        if lazy_residual:
            out += (_da.from_delayed(solve_result['residual_after'],
                                     shape=(n_pairs,) if has_pair_dim else (), dtype=np.float32),)

        return out[0] if len(out) == 1 else out

    def align(self,
              degree: int = 0,
              method: str = 'median',
              polarization: str | None = None,
              debug: bool = False,
              return_residuals: bool = False):
        """
        PAIRWISE alignment: align the bursts of each interferogram (the 'pair'
        dimension) by removing phase offsets and optionally ionospheric ramps.

        The DATEWISE alignment of a complex SLC stack (the 'date' dimension)
        is Stack.align(); a complex input without a 'pair' dimension raises.

        Input: unwrapped phase (Batch), wrapped phase (BatchWrap) or a complex
        interferogram (BatchComplex with a 'pair' dimension, e.g. before angle()).
        For complex input the coefficients are estimated on its phase, the angle
        computed lazily, with the wrapped-phase statistics, and applied as a
        phase rotation, multiplying by exp(-1j * correction): the magnitude is
        unchanged and the result is a BatchComplex.

        Uses a multi-step approach for optimal alignment:
        - degree=0: Single-step offset correction
        - degree=1: 3-step correction (offset → ramp → re-offset) for ionospheric ramp removal

        The 3-step approach produces consistent fringes across bursts by removing
        per-track ionospheric ramps, which is essential for deformation analysis.

        Lazy: nothing is computed at call time (return_residuals and debug excepted).
        Every burst's output carries a 'residual' variable (pair,): the residual
        after the alignment per pair, the measure of residuals(), lazy, taken
        from the overlaps the solve already holds. No warning is printed for a
        pair whose residual rose; select on the variable instead, e.g.
        aligned.sel(pair=aligned.residual < 0.5).

        Parameters
        ----------
        degree : int, optional
            Correction degree:
            - 0 (default): Offset-only correction (faster, good overlap alignment)
            - 1: Offset + linear ramp correction (better fringe continuity)
        method : str, optional
            Estimation method: 'median' (robust, default) or 'mean' (faster).
        polarization : str, optional
            Polarization to use for coefficient estimation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
            Corrections are applied to all polarizations since phase offsets
            are the same for all polarizations (same geometry).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return final residuals. Default is False.

        Returns
        -------
        BatchCore or tuple
            If return_residuals is False:
                Aligned interferograms with phase corrections applied, of the
                input's class, each burst with the 'residual' variable (pair,).
            If return_residuals is True:
                (aligned_intfs, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # Simple offset-only alignment (default)
        >>> aligned = intfs.align()
        >>>
        >>> # Alignment with ramp correction
        >>> aligned = intfs.align(degree=1)
        >>>
        >>> # Use VH polarization for estimation
        >>> aligned = intfs.align(polarization='VH')
        >>>
        >>> # With coherence filtering
        >>> aligned = intfs.where(corr >= 0.3).align()
        >>>
        >>> # Complex interferogram: the phase rotates, the magnitude stays
        >>> aligned = (ref * rep.conj()).align()
        >>>
        >>> # Skip the pairs the alignment left inconsistent
        >>> aligned = aligned.sel(pair=aligned.residual < 0.5)
        >>>
        >>> # Get alignment quality with result
        >>> aligned, res = intfs.align(return_residuals=True)
        >>> print('Residuals:', res)

        Notes
        -----
        For degree=1, the function performs:
        1. Estimate and remove offsets
        2. Estimate and remove ramps (using along-track and cross-path overlaps)
        3. Re-estimate offsets on ramp-corrected data
        4. Combine into final [ramp, offset] coefficients

        Ramp estimation uses:
        - Same-path, same-subswath overlaps (along-track, y-direction)
        - Cross-path overlaps (can have large x-extent with significant iono ramps)

        It skips same-path, cross-subswath overlaps (diagonal, small x-extent).

        This 3-step approach achieves better fringe continuity than single-step
        methods because it separates the offset and ramp estimation, avoiding
        cross-contamination between the two.
        """
        from .Batch import Batch, BatchWrap, BatchComplex

        # Validate class type
        is_complex = isinstance(self, BatchComplex)
        if is_complex:
            # pairwise only: a complex stack of dates is aligned by Stack.align()
            sample_ds = next(iter(self.values()))
            grids = [v for v in sample_ds.data_vars if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
            if not grids or 'pair' not in sample_ds[grids[0]].dims:
                raise ValueError("align() is pairwise: this complex data has no 'pair' dimension. "
                                 "For a date stack, use Stack.align().")
        elif not isinstance(self, (Batch, BatchWrap)):
            raise TypeError(f"align() only works with Batch (unwrapped), BatchWrap (wrapped) or complex "
                            f"interferograms (BatchComplex with a 'pair' dimension), not {type(self).__name__}")

        # the phase the coefficients are estimated on: a complex input's angle, lazy
        phase = self.angle() if is_complex else self
        # the phase of a result, for its residuals()
        phase_of = (lambda b: b.angle()) if is_complex else (lambda b: b)

        # Auto-detect polarization if not specified
        if polarization is None:
            ids = list(phase.keys())
            sample_ds = phase[ids[0]]
            # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
            available_pols = [v for v in sample_ds.data_vars
                             if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
            polarization = available_pols[0]

        def rotate(coeffs):
            """The complex input times exp(-1j * polyval(coeffs)) on every complex grid: the phase is corrected
            as the real-valued branches subtract it, the magnitude is unchanged."""
            # polyval() returns DataArrays: the rotation is a DataArray per burst
            rot = phase.polyval(coeffs).iexp(sign=-1)
            out = {}
            for k, ds in self.items():
                if k not in rot:
                    out[k] = ds
                    continue
                r = rot[k].astype(ds[polarization].dtype)
                out[k] = BatchCore._binary_vars(ds, r, operator.mul)
            return type(self)(out)

        def with_residual(batch, residual):
            """Every burst of `batch` with the lazy 'residual' variable; the grids untouched (no re-wrap)."""
            dims = ('pair',) if residual.ndim else ()
            out = {k: ds.assign(residual=xr.DataArray(residual, dims=dims)) for k, ds in batch.items()}
            return BatchWrap(out, wrap=False) if isinstance(batch, BatchWrap) else type(batch)(out)

        if degree == 0:
            # Single-step offset correction
            if debug:
                print('align(degree=0): single-step offset correction', flush=True)
                res_in = phase.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            offsets, residual = phase._align_coeffs(degree=0, method=method, polarization=polarization, debug=debug,
                                                    lazy_residual=True)
            if is_complex:
                # the offsets as [[0, offset], ...] per pair: polyval evaluates them lazily
                aligned = rotate({b: [[0.0, o] for o in offsets[b]] for b in offsets})
            else:
                aligned = self - offsets

            if debug or return_residuals:
                res_out = phase_of(aligned).residuals(polarization=polarization)
                if debug:
                    print(f'Output residuals: {res_out}', flush=True)

            aligned = with_residual(aligned, residual)
            if return_residuals:
                return aligned, res_out
            return aligned

        elif degree == 1:
            # 3-step offset-ramp-offset correction
            if debug:
                print('align(degree=1): 3-step offset-ramp-offset correction', flush=True)
                res_in = phase.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            # Step 1: Estimate offsets
            if debug:
                print('\nStep 1: Estimate offsets...', flush=True)
            offsets1 = phase._align_coeffs(degree=0, method=method, polarization=polarization, debug=debug)
            intfs1 = phase - offsets1
            if debug:
                res1 = intfs1.residuals(polarization=polarization)
                print(f'Residuals after step 1: {res1}', flush=True)

            # Step 2: Estimate ramps (uses same-track overlaps only)
            if debug:
                print('\nStep 2: Estimate ramps...', flush=True)
            ramps = intfs1._align_coeffs(degree=1, method=method, polarization=polarization, debug=debug)
            intfs2 = intfs1 - intfs1.polyval(ramps)
            if debug:
                res2 = intfs2.residuals(polarization=polarization)
                print(f'Residuals after step 2: {res2}', flush=True)

            # Step 3: Re-estimate offsets
            if debug:
                print('\nStep 3: Re-estimate offsets...', flush=True)
            offsets2, residual = intfs2._align_coeffs(degree=0, method=method, polarization=polarization,
                                                      debug=debug, lazy_residual=True)

            # Combine coefficients: [ramp, offset1 + ramp_intercept + offset2]
            # Detect if multi-pair
            sample_bid = list(offsets1.keys())[0]
            is_multi_pair = isinstance(offsets1[sample_bid], list)

            if is_multi_pair:
                n_pairs = len(offsets1[sample_bid])
                coeffs = {
                    b: [[ramps[b][p][0], ramps[b][p][1] + offsets1[b][p] + offsets2[b][p]]
                        for p in range(n_pairs)]
                    for b in offsets1
                }
            else:
                coeffs = {
                    b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]]
                    for b in offsets1
                }

            aligned = rotate(coeffs) if is_complex else self - self.polyval(coeffs)

            if debug or return_residuals:
                res_out = phase_of(aligned).residuals(polarization=polarization)
                if debug:
                    print(f'Final residuals: {res_out}', flush=True)

            aligned = with_residual(aligned, residual)
            if return_residuals:
                return aligned, res_out
            return aligned

        else:
            raise ValueError(f"degree must be 0 or 1, got {degree}")

    def dissolve(self, extend: bool = False, weight: float = None, debug: bool = False):
        """
        Dissolve burst boundaries by averaging overlapping regions.

        For each burst, this method computes a merged product covering that burst's
        extent, averaging values from all overlapping bursts.

        For wrapped phase data (BatchWrap), circular mean is used.
        For unwrapped phase or other data (Batch, BatchUnit), arithmetic mean is used.

        Parameters
        ----------
        extend : bool, optional
            If True, NaN areas in current burst can be filled by overlapping
            bursts. Good for unwrapping consistency between bursts.
            If False (default), only pixels valid in the current burst are kept (NaN areas remain NaN).
            Better for performance when you don't want to process same pixels in multiple bursts.
        weight : float, optional
            Normalized weight of the current burst in range [0, 1]. Default is None.
            weight=None: equal weights for all bursts (simple average)
            weight=1: only current burst used, overlapping bursts ignored
            weight=0: only overlapping bursts used, current burst ignored
            weight=0.5: current burst has same weight as sum of all overlapping bursts
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        BatchCore
            New batch with dissolved (averaged) overlap regions (lazy).

        Examples
        --------
        >>> # Dissolve with equal weights (default)
        >>> intfs_dissolved = intfs.dissolve()
        >>>
        >>> # Dissolve without extension (keep original burst footprint)
        >>> intfs_dissolved = intfs.dissolve(extend=False)
        >>>
        >>> # Dissolve with current burst having 70% weight
        >>> intfs_dissolved = intfs.dissolve(weight=0.7)

        Notes
        -----
        - For BatchWrap (wrapped phase): uses circular mean via exp(1j*phase)
        - For Batch/BatchUnit (unwrapped phase, correlation): uses arithmetic mean
        - Returns lazy data, processes per burst replacing polarization variables
        """
        import warnings
        import dask
        import dask.array as da
        from shapely import box, STRtree
        from .Batch import BatchWrap

        if len(self) <= 1:
            return type(self)(self)

        wrap = isinstance(self, BatchWrap)
        burst_ids = list(self.keys())
        # a Dataset's variables, or a DataArray as its own one (N81)
        vars_of = {bid: BatchCore._vars_of(self[bid]) for bid in burst_ids}
        sample = vars_of[burst_ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes
        polarizations = [v for v, a in sample.items()
                        if 'y' in a.dims and 'x' in a.dims]

        if debug:
            import time
            t0 = time.time()
            print(f'dissolve: {len(self)} bursts, wrap={wrap}, extend={extend}, weight={weight}', flush=True)

        # Build STRtree for fast spatial queries
        first_pol = polarizations[0]
        burst_extents = tuple(
            (float(vars_of[bid][first_pol].y.min()), float(vars_of[bid][first_pol].y.max()),
             float(vars_of[bid][first_pol].x.min()), float(vars_of[bid][first_pol].x.max()))
            for bid in burst_ids
        )
        burst_boxes = [box(xmin, ymin, xmax, ymax) for ymin, ymax, xmin, xmax in burst_extents]
        tree = STRtree(burst_boxes)

        overlapping_map = {
            burst_idx: tuple(int(idx) for idx in tree.query(burst_boxes[burst_idx]) if idx != burst_idx)
            for burst_idx in range(len(burst_ids))
        }

        if debug:
            total_overlaps = sum(len(v) for v in overlapping_map.values())
            print(f'dissolve: STRtree found {total_overlaps} burst overlaps', flush=True)

        # Build output — one dask.delayed task per burst per pol.
        # Pass raw dask arrays (not xarray DataArrays) to avoid expensive
        # xarray __dask_graph__() calls during dask.delayed graph construction.
        # The _dissolve_raw_for_dask function receives numpy arrays (dask resolves
        # them) and reconstructs minimal xarray DataArrays for coord matching.
        output = {}
        for burst_idx, bid in enumerate(burst_ids):
            overlapping_indices = overlapping_map[burst_idx]
            ds_current = self[bid]

            if not overlapping_indices:
                output[bid] = ds_current
                continue

            ds_others = [vars_of[burst_ids[idx]] for idx in overlapping_indices]

            # a DataArray in, a DataArray out (N81)
            is_array = isinstance(ds_current, xr.DataArray)
            new_ds = None if is_array else ds_current.copy()
            for pol in polarizations:
                da_current = vars_of[bid][pol]
                das_others = [ds[pol] for ds in ds_others]

                # Extract raw arrays and numpy coordinates.
                # Raw dask arrays have O(1) __dask_graph__() (direct attribute),
                # vs xarray DataArrays which create temp Dataset each call.
                current_arr = da_current.data
                current_y = da_current.y.values
                current_x = da_current.x.values

                if not isinstance(current_arr, da.Array):
                    current_arr = da.from_array(current_arr, chunks=current_arr.shape)

                others_arrs = []
                others_ys = []
                others_xs = []
                for d in das_others:
                    arr = d.data
                    if not isinstance(arr, da.Array):
                        arr = da.from_array(arr, chunks=arr.shape)
                    others_arrs.append(arr)
                    others_ys.append(d.y.values)
                    others_xs.append(d.x.values)

                # a unique key per call (pure=False): the arrays arrive through finalize
                # keys named anew in every graph, see _align_coeffs (N81)
                delayed_result = dask.delayed(_dissolve_raw_for_dask, pure=False)(
                    current_arr, current_y, current_x,
                    others_arrs, others_ys, others_xs,
                    wrap, extend, weight
                )
                delayed_array = da.from_delayed(
                    delayed_result,
                    shape=da_current.shape,
                    dtype=da_current.dtype
                )
                if is_array:
                    new_ds = da_current.copy(data=delayed_array)
                else:
                    new_ds[pol] = da_current.copy(data=delayed_array)

            output[bid] = new_ds

        if debug:
            print(f'dissolve: preparation done in {time.time() - t0:.1f}s', flush=True)

        return type(self)(output)
