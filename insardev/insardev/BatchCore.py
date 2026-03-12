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
from . import utils_io,  utils_xarray
import operator
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


def _merge_tiles_for_dask(tiles, offsets, out_shape, fill_dtype):
    """
    Module-level function for merging tiles in to_dataset().

    Defined at module level to avoid dask serialization issues with nested functions.
    When using dask distributed, closures with nested functions may not serialize correctly,
    causing random/incorrect behavior on workers.

    Args:
        tiles: list of dask delayed objects that will compute to 3D arrays (1, ny, nx)
        offsets: list of (y_offset, x_offset) tuples for each tile
        out_shape: (ny, nx) output shape
        fill_dtype: output dtype

    Returns:
        merged 2D numpy array
    """
    out = np.full(out_shape, np.nan, dtype=fill_dtype)

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

            # In-place fmin: min of existing and new, NaN treated as missing
            view = out[y0c:y1c, x0c:x1c]
            np.fmin(view, tile[ty0:ty1, tx0:tx1], out=view)

    return out


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


def _dissolve_pol_3d_for_dask(da_slice, das_others_slice, wrap, extend, weight):
    """
    Module-level wrapper for 3D dissolve returning shape (1, y, x).

    Defined at module level to avoid dask serialization issues with nested functions.
    """
    return _dissolve_pol_for_dask(da_slice, das_others_slice, wrap, extend, weight)[np.newaxis, ...]


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


def _apply_gaussian_for_dask(block, weight_block, sigmas, threshold, device, pixel_sizes, out_dtype):
    """
    Module-level function for gaussian blockwise operation (DEPRECATED - use _apply_gaussian_2d_for_dask).

    Defined at module level to avoid dask serialization issues with nested functions.
    Weight is passed as a separate dask array to blockwise (not via partial) to avoid
    serializing large arrays with each task - which causes memory explosions.

    Parameters
    ----------
    block : np.ndarray
        Data block from dask, shape (1, y, x) or (y, x)
    weight_block : np.ndarray or None
        Weight block from dask, shape (y, x) or None if no weight
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
    return gaussian_numpy(block, weight_block, sigma=sigmas, truncate=4.0, threshold=threshold,
                          device=device, pixel_sizes=pixel_sizes).astype(out_dtype)


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


def _trend2d_chunk_kernel(phase_block, *args, var_count=0,
                            degree=1, detrend=False, has_weight=False):
    """
    Per-tile trend2d kernel for da.map_overlap / da.blockwise.

    Accepts 2D (y, x) or 3D (1, y, x) phase blocks.
    Uses pure-numpy polynomial fitting (same as trend2d)
    to avoid torch overhead per tile.

    For map_overlap: all inputs are 2D (y, x).
    For blockwise: phase is 3D (1, y, x), vars are 2D (y, x).
    """
    from .utils_detrend import _build_poly_features

    is_3d = phase_block.ndim == 3
    if is_3d:
        # 3D (1, y, x) → extract 2D
        phase_2d = phase_block[0]
    else:
        phase_2d = phase_block

    # All-NaN tile: nothing to fit, return input as-is
    if not np.any(np.isfinite(phase_2d)):
        return phase_block

    # Unpack positional args: var_count transform arrays, then optional weight.
    # Vars may be 2D (blockwise 'yx') or 3D (map_overlap broadcast).
    var_blocks = []
    for i in range(var_count):
        v = args[i]
        var_blocks.append(v[0] if v.ndim == 3 else v)
    if has_weight:
        w = args[var_count]
        weight_2d = w[0] if w.ndim == 3 else w
    else:
        weight_2d = None

    ny, nx = phase_2d.shape
    n_pixels = ny * nx
    is_complex = np.iscomplexobj(phase_2d)

    # Build fit mask and flatten variables
    fit_mask = np.ones(n_pixels, dtype=bool)
    var_flat_list = []
    for v in var_blocks:
        v_flat = v.ravel().astype(np.float64)
        fit_mask &= np.isfinite(v_flat)
        var_flat_list.append(np.nan_to_num(v_flat, nan=0.0))

    n_valid_fit = fit_mask.sum()
    if n_valid_fit < 10:
        return phase_block

    # Build polynomial features for standardization stats (only on fit-valid pixels)
    X_poly_valid = _build_poly_features([v[fit_mask] for v in var_flat_list],
                                         int(n_valid_fit), degree)
    n_poly = X_poly_valid.shape[1]
    n_feat = n_poly + 1  # +1 for bias
    feature_mean = X_poly_valid.mean(axis=0)
    feature_std = X_poly_valid.std(axis=0) + 1e-10
    del X_poly_valid

    # Phase + weight flattened
    phase_flat = phase_2d.ravel()
    if is_complex:
        valid = np.isfinite(phase_flat) & (phase_flat != 0) & fit_mask
    else:
        valid = np.isfinite(phase_flat) & fit_mask
    if weight_2d is not None:
        w_flat = weight_2d.ravel()
        valid &= np.isfinite(w_flat)
    else:
        w_flat = None

    n_valid = valid.sum()
    if n_valid < n_feat:
        return phase_block

    # Pixel batching: keep A_std_batch + WA under half dask chunk budget
    from .utils_dask import get_dask_chunk_size_mb
    _budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
    batch_size = max(1024, _budget // max(1, 2 * n_feat * 8))
    n_batches = (n_pixels + batch_size - 1) // batch_size

    # Phase 1: Accumulate normal equations in batches
    AtWA = np.zeros((n_feat, n_feat), dtype=np.float64)
    cdtype = np.complex128 if is_complex else np.float64
    AtWb = np.zeros(n_feat, dtype=cdtype)

    for bi in range(n_batches):
        s = bi * batch_size
        e = min((bi + 1) * batch_size, n_pixels)

        valid_b = valid[s:e]
        n_v = int(valid_b.sum())
        if n_v == 0:
            continue

        # Build A_std for batch
        var_b = [v[s:e] for v in var_flat_list]
        X_poly_b = _build_poly_features(var_b, e - s, degree)
        A_std_b = np.concatenate([
            (X_poly_b - feature_mean) / feature_std,
            np.ones((e - s, 1), dtype=np.float64)
        ], axis=1)

        A_v = A_std_b[valid_b]
        if w_flat is not None:
            sqrt_w = np.sqrt(np.clip(w_flat[s:e][valid_b], 0, None))
            WA_v = A_v * sqrt_w[:, None]
        else:
            WA_v = A_v

        AtWA += WA_v.T @ WA_v

        p_v = phase_flat[s:e][valid_b]
        if is_complex:
            p_abs = np.abs(p_v)
            with np.errstate(invalid='ignore', divide='ignore'):
                b_v = np.where(p_abs > 0, p_v / p_abs, 0 + 0j)
            b_v = np.nan_to_num(b_v, nan=0.0)
        else:
            b_v = np.nan_to_num(p_v, nan=0.0).astype(np.float64)

        if w_flat is not None:
            Wb_v = sqrt_w * b_v
        else:
            Wb_v = b_v
        AtWb += WA_v.T.astype(cdtype) @ Wb_v

    # Solve
    reg = 1e-10 * np.eye(n_feat, dtype=np.float64)
    try:
        coeffs = np.linalg.solve(AtWA.astype(cdtype) + reg, AtWb)
    except np.linalg.LinAlgError:
        return phase_block

    # Phase 2: Apply trend in batches
    if detrend:
        out_2d = phase_2d.copy()
    else:
        out_2d = np.empty((ny, nx), dtype=np.complex64 if is_complex else np.float32)

    for bi in range(n_batches):
        s = bi * batch_size
        e = min((bi + 1) * batch_size, n_pixels)
        sy, sx = divmod(s, nx)
        ey, ex = divmod(e, nx)

        var_b = [v[s:e] for v in var_flat_list]
        X_poly_b = _build_poly_features(var_b, e - s, degree)
        A_std_b = np.concatenate([
            (X_poly_b - feature_mean) / feature_std,
            np.ones((e - s, 1), dtype=np.float64)
        ], axis=1)

        trend_b = A_std_b.astype(cdtype) @ coeffs
        if is_complex:
            trend_abs = np.abs(trend_b)
            with np.errstate(invalid='ignore', divide='ignore'):
                trend_b = np.where(trend_abs > 0, trend_b / trend_abs, 0)
            trend_b = np.asarray(trend_b, dtype=np.complex64)
            trend_b[~np.isfinite(trend_b)] = 0
        else:
            trend_b = np.asarray(trend_b, dtype=np.float32)

        out_flat = out_2d.ravel()
        if detrend:
            if is_complex:
                out_flat[s:e] = phase_flat[s:e] * np.conj(trend_b)
            else:
                out_flat[s:e] = phase_flat[s:e] - trend_b
        else:
            out_flat[s:e] = trend_b

    if is_3d:
        return out_2d[np.newaxis, ...]
    return out_2d


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

        Returns True if all data variables are dask arrays (lazy/deferred computation).
        Returns False if data has been computed to numpy arrays.

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
            for var in ds.data_vars:
                if not isinstance(ds[var].data, da.Array):
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
            # Get type name for error message
            for key, ds in batch.items():
                for var in ds.data_vars:
                    data_type = type(ds[var].data).__name__
                    break
                break
            raise TypeError(
                f"{func_name}() requires lazy (dask) data, got {data_type}. "
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

    def from_dataset(self, data: xr.Dataset, **kwargs) -> Batch:
        """
        Create a Batch by selecting each burst's coordinates from a merged Dataset.

        The input Dataset should have been created via to_dataset() or have
        coordinates that are supersets of each burst's coordinates.

        Parameters
        ----------
        data : xr.Dataset
            The input data to split back into per-burst datasets.

        Returns
        -------
        Batch
            A new Batch with the same keys as self, each containing the
            selected subset of the Dataset.

        Examples
        --------
        # Round-trip: merge, process, split
        merged = batch.to_dataset()
        processed = some_processing(merged)
        result = batch.from_dataset(processed)
        """
        from .Batch import Batch
        from .utils_dask import rechunk2d

        # Validate input type
        if not isinstance(data, xr.Dataset):
            raise TypeError(f"data must be xr.Dataset, got {type(data).__name__}")

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

            # Rechunk to match self's chunk structure per burst
            rechunked_vars = {}
            for var_name in selected.data_vars:
                arr = selected[var_name]
                src = ds[var_name] if var_name in ds.data_vars else None
                if src is not None and hasattr(src.data, 'chunks'):
                    chunks = dict(zip(src.dims, src.data.chunks))
                    if isinstance(arr.data, np.ndarray) or \
                       (hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks):
                        rechunked_vars[var_name] = arr.chunk(chunks)
            if rechunked_vars:
                selected = selected.assign(rechunked_vars)
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
        data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]

        chunks = sample[data_var].chunks
        #print ('chunks', chunks)
        if chunks is None:
            # Data is not lazy (numpy arrays) - return empty dict silently
            # Use batch.is_lazy to check if data is lazy before calling .chunks
            return {}

        # build dict of first‐chunk sizes, one chunk means chunk size 1 or -1
        return {dim: sizes[0] if len(sizes) > 1 else (1 if sizes[0] == 1 else -1) for dim, sizes in zip(sample[data_var].dims, chunks)}

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
        """
        # Handle list/tuple keys for dataset selection
        if isinstance(key, (list, tuple)):
            return type(self)({
                burst_id: ds[key]
                for burst_id, ds in self.items()
            })
            
        # Try to access as a dataset key first
        try:
            return super().__getitem__(key)
        except KeyError:
            # If not a dataset key, try to access as coordinate/variable
            return type(self)({
                k: ds[key] if not isinstance(ds, self.CoordCollection) else ds._ds.coords[key]
                for k, ds in self.items()
                if (isinstance(ds, self.CoordCollection) and key in ds._ds.coords) or 
                   (not isinstance(ds, self.CoordCollection) and (key in ds.coords or key in ds.data_vars))
            })

    def __getattr__(self, name: str):
        """Attribute-style access to coords or data variables (e.g., batch.ele)."""
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if not self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def _extract(ds):
            # CoordCollection wrapper
            if isinstance(ds, self.CoordCollection):
                if name in ds._ds.coords:
                    return ds._ds.coords.to_dataset()[[name]]
                return None

            # Dataset: prefer data_vars, then coords
            if hasattr(ds, 'data_vars'):
                if name in ds.data_vars:
                    return ds[[name]]
                if name in ds.coords:
                    return ds.coords.to_dataset()[[name]]
                return None

            # DataArray fallback: match by name
            if hasattr(ds, 'name') and ds.name == name:
                return ds.to_dataset()
            if hasattr(ds, 'coords') and name in ds.coords:
                return ds.coords.to_dataset()[[name]]
            return None

        subset = {k: out for k, ds in self.items() if (out := _extract(ds)) is not None}
        if subset:
            return type(self)(subset)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __add__(self, other):
        # scalar + batch → map scalar + each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v + other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __radd__(self, other):
        # scalar + batch → same as batch + scalar
        return self.__add__(other)

    def __sub__(self, other):
        # scalar - batch → map scalar - each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v - other for k, v in self.items()})
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)):
                    # Get a spatial variable (with y, x dims) to check for pair dimension
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)

                    if len(val) > 0 and isinstance(val[0], (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        # Use polyval for this case
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        # Handle both concrete scalars and dask 0-d arrays
                        if any(hasattr(v, 'dask') for v in val):
                            import dask.array as _da
                            offsets = xr.DataArray(_da.stack(val), dims=['pair'])
                        else:
                            offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = ds - offsets
                    else:
                        # Single pair degree=1: [ramp, offset] or other
                        result[k] = ds - val
                elif isinstance(val, (int, float, np.floating, np.integer)) \
                        or (hasattr(val, 'ndim') and val.ndim == 0):
                    # Scalar subtraction (concrete or dask 0-d array):
                    # only apply to spatial variables (y, x dims)
                    new_ds = ds.copy()
                    for var in ds.data_vars:
                        if 'y' in ds[var].dims and 'x' in ds[var].dims:
                            new_ds[var] = ds[var] - val
                    result[k] = new_ds
                else:
                    result[k] = ds - val
        return type(self)(result)

    def __rsub__(self, other):
        # scalar - batch
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: other - v for k, v in self.items()})
        return NotImplemented

    def __mul__(self, other):
        # scalar * batch → map scalar * each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v * other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] * other[k] if k in other else self[k]) for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other):
        # batch / scalar → map each dataset / scalar
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v / other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] / other[k] if k in other else self[k]) for k in keys})

    def __rtruediv__(self, other):
        # scalar / batch
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: other / v for k, v in self.items()})
        return NotImplemented

    def __neg__(self):
        # -batch → negate each dataset
        return type(self)({k: -v for k, v in self.items()})

    def _binop(self, other, op):
        """
        generic helper for any binary operator `op(ds, other)` or `op(ds, other_ds)`
        """
        if isinstance(other, (int, float)):
            return type(self)({k: op(ds, other) for k, ds in self.items()})
        elif isinstance(other, BatchCore):
            common = set(self) & set(other)
            return type(self)({k: op(self[k], other[k]) for k in common})
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
    def __invert__(self):      return type(self)({k: ~v for k, v in self.items()})

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

        result = {}
        for k in batch.keys():
            # build the argument list for this key
            args = [
                inp[k] if isinstance(inp, BatchCore) else inp
                for inp in inputs
            ]
            result[k] = ufunc(*args, **kwargs)

        return type(self)(result)

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

    def map_da(self, func, **kwargs):
        """Apply func(DataArray) → DataArray to every numeric var in every dataset.

        Non-numeric variables (strings, objects) are passed through unchanged.
        """
        def apply_to_numeric(ds):
            result_vars = {}
            for var in ds.data_vars:
                da = ds[var]
                # Skip non-numeric dtypes (strings, objects, etc.)
                if not np.issubdtype(da.dtype, np.number) and not np.issubdtype(da.dtype, np.complexfloating):
                    result_vars[var] = da
                else:
                    result_vars[var] = func(da, **kwargs)
            result = xr.Dataset(result_vars)
            result.attrs = ds.attrs
            return result

        return type(self)({k: apply_to_numeric(ds) for k, ds in self.items()})

    def astype(self, dtype, **kwargs):
        return self.map_da(lambda da: da.astype(dtype), **kwargs)
    
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
        Fast batch-wise .where:

        If `cond` is a Batch (or subclass) with exactly the same keys,
           and each cond[k] is a 1-variable Dataset or a DataArray,
           we extract the single DataArray mask and do either:
             - other==0 → simple multiply ds * mask_da
             - else       → ds.where(mask_da, other, **kwargs)

        Otherwise fall back to per-DataArray map_da (slower).

        keep_attrs=True argument can be used to preserve attributes of the original data.
        """
        # detect same key Batch-like mask
        if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
            out = {}
            for k, ds in self.items():
                mask_obj = cond[k]
                # extract DataArray from a 1-var Dataset or use it direct
                if isinstance(mask_obj, xr.Dataset):
                    mask_vars = list(mask_obj.data_vars)
                    if isinstance(ds, xr.Dataset):
                        data_vars = list(ds.data_vars)
                        # Multi-var case: apply each mask var to corresponding data var
                        if set(mask_vars) == set(data_vars) or set(mask_vars) >= set(data_vars):
                            new_ds = ds.copy()
                            for var in data_vars:
                                if var in mask_vars:
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
                        # Single mask var case
                        elif len(mask_vars) == 1:
                            mask_da = mask_obj[mask_vars[0]]
                        else:
                            raise ValueError(
                                f"Batch.where: mask vars {mask_vars} don't match data vars {data_vars} for '{k}'"
                            )
                    else:
                        if len(mask_vars) != 1:
                            raise ValueError(f"Batch.where: expected 1 var in mask for '{k}', got {mask_vars}")
                        mask_da = mask_obj[mask_vars[0]]
                else:
                    mask_da = mask_obj

                # Align mask to data coordinates (handles different x/y grids)
                # Get reference DataArray from ds for alignment (use spatial variable)
                if isinstance(ds, xr.Dataset):
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    ref_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    ref_da = ds[ref_var]
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

                out[k] = ds.where(mask_da, other, **kwargs)
            return type(self)(out)

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
        >>> sim_dense = S_sparse.where(dense_mask).similarity(...)
        >>> sim_sparse = S_sparse.where(sparse_mask).similarity(...)
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
        mask : xr.DataArray, xr.Dataset, or GeoDataFrame
            The mask to apply. Can be:
            - xr.DataArray: boolean mask reindexed to each burst's coordinates
            - xr.Dataset: first data variable used as mask, reindexed per burst
            - GeoDataFrame: polygon(s) to mask by - pixels inside polygons are kept
        other : scalar, optional
            Value to use for masked elements. Default is np.nan.

        Returns
        -------
        Batch
            Masked batch with same type as self.

        Examples
        --------
        # Apply binary landmask
        land = np.isfinite(xr.open_dataarray('land.nc').rio.reproject(intf.crs))
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

        # Handle xarray mask
        if isinstance(mask, xr.Dataset):
            mask = next(iter(mask.data_vars.values()))

        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError('Batch.mask: mask must be a Dataset or DataArray of boolean type, or a GeoDataFrame')

        # auto-chunk if not already chunked to avoid high memory usage
        if not mask.chunks:
            mask = mask.chunk('auto')

        out = {}
        for key, ds in self.items():
            # the fastest way to align mask to burst coordinates
            mask_burst = mask.reindex(y=ds.y, x=ds.x, method='nearest')
            # preserve original chunking structure for lazy computation
            out[key] = ds.where(mask_burst, other)
        return type(self)(out)

    def trend2d(self, transform: 'BatchCore', weight: 'BatchUnit | None' = None,
                degree: int = 1, device: str = 'auto', detrend: bool = False,
                debug: bool = False) -> 'BatchCore':
        """
        Compute 2D polynomial trend (ramp) from data.

        Two modes:
        - Complex (BatchComplex): unit-circle fitting, returns BatchComplex
        - Real (Batch): standard polynomial, returns Batch

        Parameters
        ----------
        transform : BatchCore
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
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
        >>> # Real phase
        >>> trend = phase.trend2d(stack.transform(), weight=corr)
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

        if transform is None:
            raise ValueError("transform is required for trend2d. Use stack.transform()[['azi','rng','ele']] or stack.transform()[['azi','rng']].")

        # Unify transform keys to phase
        transform = transform.sel(phase)

        result = {}
        for key in phase.keys():
            ds = phase[key]

            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            trans_ds = transform[key]
            var_names = [v for v in trans_ds.data_vars
                        if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]

            # Check that transform and weight resolutions match phase resolution
            phase_da_ref = ds[pols[0]]
            phase_shape = phase_da_ref.shape[-2:]
            phase_dy = float(phase_da_ref.y.diff('y')[0])
            phase_dx = float(phase_da_ref.x.diff('x')[0])

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

                # Rechunk transform to match phase spatial chunks
                phase_spatial_chunks = phase_dask.chunks[-2:]
                var_dask_list = []
                for v in var_names:
                    var_dask = trans_ds[v].data
                    if var_dask.chunks != phase_spatial_chunks:
                        var_dask = var_dask.rechunk(phase_spatial_chunks)
                    var_dask_list.append(var_dask)

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
                                  degree, is_complex, detrend_mode):
                    def fn(*args):
                        phase_c = args[0]
                        coeffs_c = args[1]
                        var_cs = args[2:2 + n_vars]
                        return utils_detrend._apply_chunk(
                            phase_c, coeffs_c, var_cs,
                            feature_mean, feature_std,
                            degree, is_complex, detrend_mode)
                    return fn

                apply_fn = make_apply_fn(
                    n_vars, feature_mean, feature_std,
                    degree, is_complex, detrend)

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

    def trend2d_chunk(self, transform: 'BatchCore', weight: 'BatchUnit | None' = None,
                       degree: int = 1, overlap: 'float | int | tuple | None' = None,
                       device: str = 'auto', detrend: bool = False,
                       debug: bool = False) -> 'BatchCore':
        """
        Compute 2D polynomial trend using overlapping-tile local fitting.

        Fits independent local polynomials per overlapping tile using
        da.map_overlap. Each tile sees neighbors via overlap for well-conditioned
        fits at tile edges. No blending needed — dask trims overlap margins.

        Parameters
        ----------
        transform : BatchCore
            Coordinate transform from stack.transform() containing 'azi' and 'rng'.
        weight : BatchUnit or None
            Optional weight for the fitting (typically correlation).
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        overlap : float, int, or tuple
            Overlap size. Float values are fractions of chunk size (e.g. 0.25 = 25%).
            Int values are pixels. Tuple (overlap_y, overlap_x) allows different
            overlap per axis. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        detrend : bool
            If True, return detrended data instead of the trend surface.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch or BatchComplex
            Trend surface or detrended data (same type as input).

        Examples
        --------
        >>> # Local polynomial trend removal
        >>> trend = phase.trend2d_chunk(stack.transform(), weight=corr)
        >>> # Detrend in one step (fused, memory-efficient)
        >>> detrended = intf.trend2d_chunk(transform, weight=corr, detrend=True)
        """
        import dask
        import dask.array as da
        import numpy as np
        import xarray as xr
        from .Batch import Batch, BatchComplex

        phase = self

        # Validate lazy data
        BatchCore._require_lazy(phase, 'trend2d_chunk')

        # Auto-detect device
        resolved = BatchCore._get_torch_device(device, debug=debug)
        device = resolved.type

        if debug:
            print(f"DEBUG: trend2d_chunk using device={device}")

        is_complex = isinstance(phase, BatchComplex)

        if transform is None:
            raise ValueError("transform is required for trend2d_chunk.")

        # Unify transform keys to phase
        transform = transform.sel(phase)

        # Parse overlap into (ov_y, ov_x) — each is float or int
        if overlap is None:
            ov_y, ov_x = 0, 0
        elif isinstance(overlap, tuple):
            ov_y, ov_x = overlap
        else:
            ov_y, ov_x = overlap, overlap

        result = {}
        for key in phase.keys():
            ds = phase[key]

            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            trans_ds = transform[key]
            var_names = [v for v in trans_ds.data_vars
                        if 'y' in trans_ds[v].dims and 'x' in trans_ds[v].dims]

            # Check that transform shape matches phase
            phase_da_ref = ds[pols[0]]
            phase_shape = phase_da_ref.shape[-2:]

            trans_da_ref = trans_ds[var_names[0]]
            trans_shape = trans_da_ref.shape
            if phase_shape != trans_shape:
                phase_dy = float(phase_da_ref.y.diff('y')[0])
                phase_dx = float(phase_da_ref.x.diff('x')[0])
                trans_dy = float(trans_da_ref.y.diff('y')[0])
                trans_dx = float(trans_da_ref.x.diff('x')[0])
                raise ValueError(
                    f"Transform shape {trans_shape} does not match phase shape {phase_shape}. "
                    f"Phase spacing: dy={phase_dy:.1f}, dx={phase_dx:.1f}. "
                    f"Transform spacing: dy={trans_dy:.1f}, dx={trans_dx:.1f}. "
                    f"Use stack.transform()[['azi','rng','ele']].downsample(N) to match."
                )

            if weight is not None:
                weight_ds = weight[key]
                weight_pols = [v for v in weight_ds.data_vars
                              if 'y' in weight_ds[v].dims and 'x' in weight_ds[v].dims]
                if weight_pols:
                    weight_da_ref = weight_ds[weight_pols[0]]
                    weight_shape = weight_da_ref.shape[-2:]
                    if phase_shape != weight_shape:
                        raise ValueError(
                            f"Weight shape {weight_shape} does not match phase shape {phase_shape}."
                        )

            if debug:
                print(f"DEBUG {key}: variables={var_names}")

            result_ds = {}
            for pol in pols:
                phase_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                phase_dask = phase_da.data
                has_weight = weight_da is not None
                out_dtype = phase_da.dtype if is_complex else np.float32

                # Prepare transform vars (2D) — rechunk to match phase spatial chunks
                phase_spatial_chunks = phase_dask.chunks[-2:]
                var_dask_list = []
                for v in var_names:
                    var_dask = trans_ds[v].data
                    if var_dask.chunks != phase_spatial_chunks:
                        var_dask = var_dask.rechunk(phase_spatial_chunks)
                    var_dask_list.append(var_dask)
                n_vars = len(var_dask_list)

                # Prepare weight — rechunk to match phase if needed
                if has_weight:
                    weight_dask = weight_da.data
                    if weight_dask.chunks != phase_dask.chunks:
                        weight_dask = weight_dask.rechunk(phase_dask.chunks)

                # Compute overlap depth from chunk sizes
                cy0 = phase_dask.chunks[-2][0]
                cx0 = phase_dask.chunks[-1][0]
                depth_y = int(ov_y * cy0) if isinstance(ov_y, float) else int(ov_y)
                depth_x = int(ov_x * cx0) if isinstance(ov_x, float) else int(ov_x)
                if overlap is not None and overlap != 0 and overlap != (0, 0):
                    depth_y = max(1, depth_y)
                    depth_x = max(1, depth_x)

                if debug:
                    print(f"DEBUG {key}/{pol}: depth=({depth_y},{depth_x}), "
                          f"chunks={phase_dask.chunksize}")

                # Build kernel function for blockwise/map_overlap
                def _make_kernel(nv, deg, det, hw):
                    def kernel(phase_block, *args):
                        var_arrays = args[:nv]
                        weight_block = args[nv] if hw else None
                        # Loop over pairs in dim 0
                        results = []
                        for pidx in range(phase_block.shape[0]):
                            p = phase_block[pidx:pidx+1]
                            kargs = list(var_arrays)
                            if hw:
                                kargs.append(weight_block[pidx:pidx+1])
                            r = _trend2d_chunk_kernel(
                                p, *kargs,
                                var_count=nv, degree=deg,
                                detrend=det, has_weight=hw)
                            results.append(r)
                        return np.concatenate(results, axis=0)
                    return kernel
                kernel = _make_kernel(n_vars, degree, detrend, has_weight)

                if depth_y > 0 or depth_x > 0:
                    # map_overlap path — needs all inputs to have same spatial chunks
                    depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                    depth_2d = {0: depth_y, 1: depth_x}

                    # Wrap kernel for map_overlap: unpack positional args
                    def _make_overlap_fn(kern, nv, hw):
                        def fn(phase_block, *args):
                            # map_overlap passes blocks with overlap margins
                            return kern(phase_block, *args)
                        return fn
                    overlap_fn = _make_overlap_fn(kernel, n_vars, has_weight)

                    # Build args list for map_overlap
                    overlap_args = []
                    for v_dask in var_dask_list:
                        overlap_args.append(v_dask)
                    if has_weight:
                        overlap_args.append(weight_dask)

                    result_dask = da.map_overlap(
                        overlap_fn,
                        phase_dask, *overlap_args,
                        depth=[depth_3d] + [depth_2d] * n_vars +
                              ([depth_3d] if has_weight else []),
                        boundary='none',
                        dtype=out_dtype,
                    )
                else:
                    # blockwise path — no overlap, same output chunks as input
                    blockwise_args = [phase_dask, 'pyx']
                    for v_dask in var_dask_list:
                        blockwise_args.extend([v_dask, 'yx'])
                    if has_weight:
                        blockwise_args.extend([weight_dask, 'pyx'])

                    result_dask = da.blockwise(
                        kernel, 'pyx',
                        *blockwise_args,
                        concatenate=True,
                        dtype=out_dtype,
                        meta=np.empty((0, 0, 0), dtype=out_dtype),
                    )

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

    @classmethod
    def trend2d_dataset(cls, phase, weight=None, transform=None, degree=1, device='auto', debug=False):
        """
        Compute 2D trend from phase Dataset using PyTorch (GPU-accelerated).

        Convenience wrapper around trend2d() for working with merged datasets
        instead of per-burst batches.

        Parameters
        ----------
        phase : xr.Dataset
            Phase dataset.
        weight : xr.Dataset, optional
            Correlation/weight dataset.
        transform : xr.Dataset, optional
            Transform dataset with variables to use as regressors.
        degree : int, optional
            Polynomial degree (default 1).
        device : str, optional
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        xr.Dataset
            Trend dataset with same structure as input phase.

        Examples
        --------
        >>> trend = BatchCore.trend2d_dataset(phase_ds, corr_ds, transform_ds, degree=1)
        """
        import xarray as xr
        from .Batch import Batch, BatchUnit

        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")
        if transform is not None and not isinstance(transform, xr.Dataset):
            raise TypeError(f"transform must be xr.Dataset, got {type(transform).__name__}")

        # Rechunk to single spatial chunk if needed
        needs_rechunk = False
        for var_name in phase.data_vars:
            var_data = phase[var_name]
            if hasattr(var_data.data, 'chunks'):
                chunks = var_data.data.chunks
                y_chunks = chunks[-2] if len(chunks) >= 2 else chunks[0]
                x_chunks = chunks[-1]
                if len(y_chunks) > 1 or len(x_chunks) > 1:
                    print(f"NOTE: trend2d_dataset() rechunking to single spatial chunk "
                          f"(from y: {len(y_chunks)} chunks, x: {len(x_chunks)} chunks)")
                    needs_rechunk = True
                break

        if needs_rechunk:
            phase = phase.chunk({'y': -1, 'x': -1})
            if weight is not None:
                weight = weight.chunk({'y': -1, 'x': -1})
            if transform is not None:
                transform = transform.chunk({'y': -1, 'x': -1})

        # Wrap datasets in single-key Batches for trend2d()
        dummy_key = '__dataset__'
        phase_batch = Batch({dummy_key: phase})
        weight_batch = BatchUnit({dummy_key: weight}) if weight is not None else None
        transform_batch = Batch({dummy_key: transform}) if transform is not None else None

        # Call trend2d on the batch
        trend_batch = phase_batch.trend2d(
            transform=transform_batch,
            weight=weight_batch,
            degree=degree,
            device=device,
            debug=debug
        )

        output = trend_batch[dummy_key]
        output.attrs = phase.attrs
        return output

    def trend1d(self, weight: 'BatchUnit | None' = None, baseline: str = 'BPR',
                degree: int = 1, device: str = 'auto', detrend: bool = False,
                intercept: bool = False, slope: bool = True,
                debug: bool = False) -> 'Batch':
        """
        Fit 1D polynomial trend along perpendicular baseline at each (y, x) pixel.

        Two modes:
        - Complex (BatchComplex): unit-circle fitting, returns BatchComplex
        - Real (Batch): standard polynomial, returns Batch

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the fitting (typically correlation).
        baseline : str
            Variable name to regress against (default 'BPR' for perpendicular baseline).
        degree : int
            Polynomial degree (1=linear, 2=quadratic). Default 1.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        detrend : bool
            If True, return detrended data instead of the trend surface.
            Fuses fit+subtract into one blockwise call to avoid double-referencing
            the input data in the dask graph.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch or BatchComplex
            Fitted trend values, same type and shape as input data.

        Examples
        --------
        >>> trend = intf.trend1d(weight=corr)
        >>> detrended = intf * trend.conj()  # complex
        >>> detrended = phase - trend        # real
        """
        import dask
        import dask.array as da
        import numpy as np
        import xarray as xr
        import pandas as pd
        from . import utils_detrend
        from .Batch import Batch, BatchComplex

        data = self

        # Validate lazy data
        BatchCore._require_lazy(data, 'trend1d')

        # Auto-detect device
        resolved = BatchCore._get_torch_device(device, debug=debug)
        device = resolved.type

        if debug:
            print(f"DEBUG: using device={device}")

        is_complex = isinstance(data, BatchComplex)

        result = {}
        for key in data.keys():
            ds = data[key]

            pols = [v for v in ds.data_vars
                   if 'y' in ds[v].dims and 'x' in ds[v].dims]

            if not pols:
                result[key] = ds
                continue

            ref_da = ds[pols[0]]

            # Detect stack dimension (pair or date)
            stack_dims = [d for d in ['pair', 'date'] if d in ref_da.dims]
            if not stack_dims:
                raise ValueError(f"Data must have 'pair' or 'date' dimension, got: {ref_da.dims}")
            stack_dim = stack_dims[0]

            # Determine baseline_values for regression
            if hasattr(baseline, 'keys') and callable(baseline.keys):
                baseline_ds = baseline[key]
                if hasattr(baseline_ds, 'data_vars') and len(baseline_ds.data_vars) > 0:
                    baseline_var = list(baseline_ds.data_vars)[0]
                    baseline_values = np.asarray(baseline_ds[baseline_var].values, dtype=np.float64)
                else:
                    baseline_values = np.asarray(baseline_ds.values, dtype=np.float64)
            elif isinstance(baseline, str):
                if baseline in ds.data_vars:
                    baseline_values = np.asarray(ds[baseline].values, dtype=np.float64)
                elif baseline in ref_da.coords:
                    baseline_coord = ref_da.coords[baseline]
                    if np.issubdtype(baseline_coord.dtype, np.datetime64):
                        dates = pd.to_datetime(baseline_coord.values)
                        baseline_values = (dates - dates.min()).total_seconds().values / 86400.0
                    else:
                        baseline_values = baseline_coord.values.astype(np.float64)
                else:
                    raise ValueError(f"'{baseline}' not found in data variables or coordinates")
            elif isinstance(baseline, (xr.DataArray, pd.DataFrame, pd.Series)):
                baseline_values = np.asarray(baseline.values, dtype=np.float64)
            else:
                baseline_values = np.asarray(baseline, dtype=np.float64)

            if debug:
                print(f"DEBUG {key}: stack_dim={stack_dim}, n_samples={len(baseline_values)}, baseline range=[{baseline_values.min():.2f}, {baseline_values.max():.2f}]")

            out_dtype = np.complex64 if is_complex else np.float32

            result_ds = {}
            for pol in pols:
                data_da = ds[pol]
                weight_da = weight[key][pol] if weight is not None else None

                if data_da.dims[0] != stack_dim:
                    data_da = data_da.transpose(stack_dim, ...)
                    if weight_da is not None:
                        weight_da = weight_da.transpose(stack_dim, ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                n_stack = data_dask.shape[0]

                def _trend1d_block(data_block, weight_block=None,
                                   _bv=baseline_values, _dev=device,
                                   _deg=degree, _detrend=detrend, _dtype=out_dtype,
                                   _intercept=intercept, _slope=slope):
                    import torch
                    trend = utils_detrend.trend1d_array(
                        [data_block], _bv,
                        [weight_block] if weight_block is not None else None,
                        torch.device(_dev), _deg,
                        intercept=_intercept, slope=_slope,
                    )
                    if _detrend:
                        if np.iscomplexobj(data_block):
                            return (data_block * np.conj(trend)).astype(_dtype)
                        else:
                            return (data_block - trend).astype(_dtype)
                    return trend.astype(_dtype)

                if weight_dask is not None:
                    result_dask = da.blockwise(
                        _trend1d_block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_stack},
                        concatenate=True,
                        dtype=out_dtype,
                        meta=np.empty((0, 0, 0), dtype=out_dtype),
                    )
                else:
                    result_dask = da.blockwise(
                        _trend1d_block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_stack},
                        concatenate=True,
                        dtype=out_dtype,
                        meta=np.empty((0, 0, 0), dtype=out_dtype),
                    )

                fit_da = xr.DataArray(
                    result_dask,
                    dims=data_da.dims,
                    coords=data_da.coords
                )

                result_ds[pol] = fit_da

            result[key] = xr.Dataset(result_ds, attrs=ds.attrs)

        if is_complex:
            return BatchComplex(result)
        return Batch(result)

    # Backward compatibility alias
    regression1d_baseline = trend1d

    def trend1d_pairs(self, weight: 'BatchUnit | None' = None, degree: int = 1,
                      device: str = 'auto', detrend: bool = False,
                      debug: bool = False) -> 'Batch':
        """
        Fit 1D polynomial trend along temporal pairs for each date.

        For each date, gathers all pairs sharing that date, fits a polynomial
        to phase vs temporal baseline, and evaluates the model. Uses all pairs
        (no temporal filtering) to maximize the fit constraint.

        Two modes:
        - Complex input (BatchComplex): unit-circle fitting per date,
          reconstruct pair trend as model[ref] * conj(model[rep]).
          Returns BatchComplex with unit-magnitude complex trend.
        - Real input (Batch): standard polynomial fit per date,
          reconstruct as model[ref] - model[rep]. Returns Batch.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the fitting (typically correlation).
        degree : int
            Polynomial degree (0=mean, 1=linear). Default 1.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        detrend : bool
            If True, return detrended data instead of the trend surface.
            Fuses fit+subtract into one blockwise call to avoid double-referencing
            the input data in the dask graph.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch or BatchComplex
            Fitted trend values, same shape as input data.

        Examples
        --------
        >>> trend = intf.trend1d_pairs(degree=1)
        >>> detrended = intf * trend.conj()  # complex
        >>> detrended = phase - trend        # real
        """
        import dask.array as da
        import numpy as np
        import xarray as xr
        import pandas as pd
        from . import utils_detrend
        from .Batch import Batch, BatchComplex

        data = self

        # Validate lazy data
        BatchCore._require_lazy(data, 'trend1d_pairs')

        # Auto-detect device
        resolved = BatchCore._get_torch_device(device, debug=debug)
        device = resolved.type

        if debug:
            print(f"DEBUG: trend1d_pairs using device={device}")

        is_complex = isinstance(data, BatchComplex)

        if not isinstance(data, (Batch, BatchComplex)):
            raise TypeError(f"trend1d_pairs() requires Batch or BatchComplex, got {type(data).__name__}")

        out_dtype = np.complex64 if is_complex else np.float32

        results = {}
        for burst_id in data.keys():
            burst_ds = data[burst_id]
            burst_weight = weight[burst_id] if weight is not None else None

            result_ds = {}
            for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                data_da = burst_ds[pol]
                weight_da = burst_weight[pol] if burst_weight is not None else None

                if debug:
                    n_pairs = len(data_da.pair)
                    n_dates = len(set(burst_ds['ref'].values.tolist() + burst_ds['rep'].values.tolist()))
                    print(f"DEBUG {burst_id}: {n_dates} dates, {n_pairs} pairs")

                # Get ref/rep as int64 nanoseconds (avoids datetime64 serialization issues)
                ref_values = burst_ds['ref'].values.astype('datetime64[ns]').astype(np.int64)
                rep_values = burst_ds['rep'].values.astype('datetime64[ns]').astype(np.int64)

                if data_da.dims[0] != 'pair':
                    data_da = data_da.transpose('pair', ...)
                    if weight_da is not None:
                        weight_da = weight_da.transpose('pair', ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                n_pairs_val = len(ref_values)

                def _trend1d_pairs_block(data_block, weight_block=None,
                                         _ref=ref_values, _rep=rep_values,
                                         _dev=str(device), _deg=degree,
                                         _detrend=detrend,
                                         _dtype=out_dtype):
                    import torch
                    trend = utils_detrend.trend1d_pairs_array(
                        [data_block],
                        [weight_block] if weight_block is not None else None,
                        _ref, _rep, torch.device(_dev), _deg
                    )
                    if _detrend:
                        if np.iscomplexobj(data_block):
                            return (data_block * np.conj(trend)).astype(_dtype)
                        else:
                            return (data_block - trend).astype(_dtype)
                    return trend.astype(_dtype)

                if weight_dask is not None:
                    result_dask = da.blockwise(
                        _trend1d_pairs_block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=out_dtype,
                        meta=np.empty((0, 0, 0), dtype=out_dtype),
                    )
                else:
                    result_dask = da.blockwise(
                        _trend1d_pairs_block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=out_dtype,
                        meta=np.empty((0, 0, 0), dtype=out_dtype),
                    )

                trend_da = xr.DataArray(
                    result_dask,
                    dims=data_da.dims,
                    coords=data_da.coords
                )
                result_ds[pol] = trend_da

            results[burst_id] = xr.Dataset(result_ds, attrs=burst_ds.attrs)

        if is_complex:
            return BatchComplex(results)
        return Batch(results)

    def regression1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("regression1d_pairs() is removed. Use trend1d_pairs() instead.")

    def neighbors(
        self,
        window: tuple = (5, 5),
        neighbors: tuple | None = None,
        device: str = 'auto'
    ) -> 'Batch':
        """
        Count valid (non-NaN) neighbors per pixel within a spatial window.

        Works on any 2D spatial data (y, x). For each pixel, counts how many
        neighbors in the window have finite values.

        Parameters
        ----------
        window : tuple of int
            Window size (y, x). Must be odd numbers.
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
        >>> # Count neighbors on similarity result
        >>> sim = S_opt.similarity(window=(5, 5), neighbors=(5, 5))
        >>> sparse_sim = sim.where(sim < 0.5)
        >>> nbrs = sparse_sim.neighbors(window=(15, 15))
        >>> dense_mask = nbrs >= 10
        """
        import torch
        import dask.array as da

        window_y, window_x = window

        # Validate window sizes are odd
        if window_y % 2 == 0 or window_x % 2 == 0:
            raise ValueError(f"Window sizes must be odd, got ({window_y}, {window_x})")

        # Validate neighbors if provided
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

        # Resolve device once and convert to string for clean serialization
        resolved = BatchCore._get_torch_device(device)
        device_str = resolved.type  # 'cpu', 'cuda', or 'mps'
        half_y, half_x = window_y // 2, window_x // 2

        # Use functools.partial with module-level function to avoid closure
        # Closures capturing variables can cause memory explosions in dask workers
        import functools
        neighbors_func = functools.partial(
            _neighbors_kernel_2d_for_dask,
            window_y=window_y,
            window_x=window_x,
            half_y=half_y,
            half_x=half_x,
            device=device_str
        )

        results = {}

        for burst_id, ds in self.items():
            count_vars = {}

            for var_name in ds.data_vars:
                data = ds[var_name]

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
                results[burst_id] = xr.Dataset(count_vars)

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
        **indexers : slice or value
            Coordinate-based selection applied to each dataset.
            Example: x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000)

        Returns
        -------
        Batch
            New Batch with selected data.

        Examples
        --------
        Select by burst keys:
        >>> subset = batch.sel(['burst1', 'burst2'])

        Select by spatial coordinates:
        >>> subset = batch.sel(x=slice(650_000, 700_000))
        >>> subset = batch.sel(x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000))

        Combine both:
        >>> subset = batch.sel(['burst1'], x=slice(650_000, 700_000))
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
            def select_with_slices(ds, indexers):
                for dim, idx in indexers.items():
                    if dim not in ds.coords:
                        continue
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

            # Apply coordinate selection to each dataset
            # Bursts outside the range get NaN-filled with the target coordinates
            out = {}
            target_coords = {}  # Will store target y/x from first non-empty result

            for k, ds in result.items():
                selected = select_with_slices(ds, indexers)
                if all(selected.sizes[d] > 0 for d in ('y', 'x') if d in selected.sizes):
                    out[k] = selected
                    # Capture target coordinates from first non-empty result
                    if not target_coords:
                        for dim in ('y', 'x'):
                            if dim in selected.coords:
                                target_coords[dim] = selected.coords[dim].values

            # If no burst had data, compute target coords from slice bounds and spacing
            if not target_coords and result:
                sample_ds = next(iter(result.values()))
                for dim in ('y', 'x'):
                    if dim in indexers and isinstance(indexers[dim], slice) and dim in sample_ds.coords:
                        coord_vals = sample_ds.coords[dim].values
                        # Get spacing from original data
                        spacing = abs(coord_vals[1] - coord_vals[0]) if len(coord_vals) > 1 else 1
                        # Get bounds from slice
                        start = indexers[dim].start if indexers[dim].start is not None else coord_vals.min()
                        stop = indexers[dim].stop if indexers[dim].stop is not None else coord_vals.max()
                        min_val, max_val = min(start, stop), max(start, stop)
                        # Create target coordinates
                        is_descending = len(coord_vals) > 1 and coord_vals[0] > coord_vals[-1]
                        if is_descending:
                            target_coords[dim] = np.arange(max_val, min_val - spacing/2, -spacing)
                        else:
                            target_coords[dim] = np.arange(min_val, max_val + spacing/2, spacing)

            # Fill empty bursts with NaN using target coordinates
            if target_coords:
                for k, ds in result.items():
                    if k not in out:
                        # Reindex to target coords with NaN fill
                        out[k] = ds.reindex(**target_coords, fill_value=np.nan)

            return type(result)(out)

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
        """
        import numpy as np

        # dict as a keyword indexers
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray‐style keyword isel (including dict-via-positional)
        if indexers:
            return type(self)({
                k: ds.isel(**indexers)
                for k, ds in self.items()
            })

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
                        # Get this burst's values from the batch
                        burst_ds = batch[key]
                        var_name = next(iter(burst_ds.data_vars))
                        data = burst_ds[var_name]
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

    def drop_vars(self, names):
        """Return a new Batch with those data-vars removed from each dataset."""
        if isinstance(names, str):
            names = [names]
        return type(self)({
            k: ds.drop_vars(names)
            for k, ds in self.items()
        })

    def rename_vars(self, **kw):
        return type(self)({k: ds.rename_vars(**kw) for k, ds in self.items()})
    
    def rename(self, **kw):
        return type(self)({k: ds.rename(**kw) for k, ds in self.items()})

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
        return Batch({k: xr.merge([ds, other[k]]) for k, ds in self.items() if k in other})

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
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            if "dim" in sig.parameters:
                out[key] = fn(dim=dim, **kwargs)
            else:
                out[key] = fn(**kwargs)
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
        """RMSE: self (pairs) vs solution (pairs or dates).

        Parameters
        ----------
        solution : Batch or BatchWrap
            If pair-based: direct comparison.
            If date-based: pairs reconstructed as sol[rep] - sol[ref].
        weight : BatchUnit, optional
            Per-pair weights (e.g., correlation).

        Returns
        -------
        Batch
            Per-pixel RMSE on (y, x) grid.
        """
        import numpy as np
        import xarray as xr
        from .Batch import Batch

        # Detect pair-based vs date-based solution
        sol_sample_ds = next(iter(solution.values()))
        spatial_vars = [v for v in sol_sample_ds.data_vars if 'y' in sol_sample_ds[v].dims]
        is_date_based = 'date' in sol_sample_ds[spatial_vars[0]].dims

        if is_date_based:
            # Reconstruct pairs from date-based solution
            recon = {}
            for key in self:
                if key not in solution:
                    continue
                obs_ds = self[key]
                sol_ds = solution[key]
                recon_vars = {}
                for var in obs_ds.data_vars:
                    if var not in sol_ds.data_vars or 'y' not in obs_ds[var].dims:
                        continue
                    refs = obs_ds[var].coords['ref'].values
                    reps = obs_ds[var].coords['rep'].values
                    recon_list = []
                    for p in range(len(refs)):
                        recon_list.append(
                            sol_ds[var].sel(date=reps[p]) - sol_ds[var].sel(date=refs[p])
                        )
                    recon_vars[var] = xr.concat(recon_list, dim='pair')
                recon[key] = xr.Dataset(recon_vars)
            solution_pairs = Batch(recon)
        else:
            solution_pairs = solution

        # Compute error using batch subtraction (handles wrapping for BatchWrap)
        error = self - solution_pairs

        # Compute per-pixel RMSE across pairs
        out = {}
        for key in error:
            err_ds = error[key]
            rmse_vars = {}
            for var in err_ds.data_vars:
                if 'y' not in err_ds[var].dims or 'pair' not in err_ds[var].dims:
                    continue
                err_sq = err_ds[var] ** 2
                if weight is not None and key in weight:
                    w_ds = weight[key]
                    w_da = w_ds[var] if var in w_ds.data_vars else w_ds[next(
                        v for v in w_ds.data_vars if 'y' in w_ds[v].dims)]
                    rmse_val = np.sqrt((w_da * err_sq).sum('pair') / w_da.sum('pair'))
                else:
                    rmse_val = np.sqrt(err_sq.mean('pair'))
                rmse_vars[var] = rmse_val.astype('float32')
            out[key] = xr.Dataset(rmse_vars, attrs=self[key].attrs)

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
            New batch with polynomial evaluated at each position. The result has
            the same structure as self, with polynomial values broadcast to match
            each dataset's shape.

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
            spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
            sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]

            if bid not in coeffs:
                # No coefficients for this burst - zero correction
                result[bid] = xr.zeros_like(ds[sample_var]).to_dataset(name=sample_var)
                continue

            # Get coordinate for evaluation
            coord = ds.coords[dim]
            sample_da = ds[sample_var]

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

            result[bid] = correction.to_dataset(name=sample_var)

        return type(self)(result)

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
        chunks : dict or 'auto'
            Chunk specification. If 'auto', uses chunk size 1 for first dimension
            (date/pair) and uniform chunking for spatial dimensions (y, x).
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

        with ctx:
            # Only chunk spatial variables (y, x dims), leave non-spatial as-is
            result = {}
            for k, ds in self.items():
                rechunked_vars = {}
                for var in ds.data_vars:
                    arr = ds[var]
                    # Only touch spatial variables
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
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
                    else:
                        # Explicit chunks - add first dim=1 for 3D
                        if arr.ndim == 3:
                            var_chunks = {arr.dims[0]: 1, **chunks}
                        else:
                            var_chunks = chunks
                    rechunked_vars[var] = arr.chunk(var_chunks)
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def chunk2d(self, budget=None, p2p=False):
        """
        Rechunk for pair-based (2D) processing: dim-0=1, spatial dims sized to budget.

        Computes optimal spatial chunk sizes so that each 2D slice (one pair/date)
        fits within the specified memory budget.

        Parameters
        ----------
        budget : str or None
            Memory budget per chunk, e.g. '128MiB', '256MB', '1GiB'.
            If None (default), uses dask.config['array.chunk-size'].
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
        """
        import dask
        from .utils_dask import rechunk2d

        target_mb = _parse_budget(budget) if budget is not None else None

        if p2p:
            ctx = dask.config.set({
                "array.rechunk.method": "p2p",
                "optimization.fuse.active": False,
            })
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            result = {}
            for k, ds in self.items():
                # Compute spatial chunks once per burst using 8 bytes (complex64)
                # so all variables get identical spatial chunks.
                sample = None
                for var in ds.data_vars:
                    arr = ds[var]
                    if arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x'):
                        sample = arr
                        break
                if sample is None:
                    result[k] = ds
                    continue
                y_size, x_size = sample.shape[-2], sample.shape[-1]
                in_chunks = (sample.data.chunks[-2], sample.data.chunks[-1]) if hasattr(sample.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                   input_chunks=in_chunks, target_mb=target_mb, merge=True)
                rechunked_vars = {}
                for var in ds.data_vars:
                    arr = ds[var]
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
                    if arr.ndim == 3:
                        var_chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                    else:
                        var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                    rechunked_vars[var] = arr.chunk(var_chunks)
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def chunk1d(self, budget=None, p2p=False):
        """
        Rechunk for date-based (1D) processing: dim-0=-1, spatial dims sized to budget.

        Computes optimal spatial chunk sizes so that the full date/pair stack
        fits within the specified memory budget per spatial tile.

        Parameters
        ----------
        budget : str
            Memory budget per chunk, e.g. '128MiB', '256MB', '1GiB'.
        p2p : bool
            Use P2P rechunk for constant-memory rechunking. Default False.

        Returns
        -------
        BatchCore
            New batch with rechunked data.

        Examples
        --------
        >>> data = Stack().snapshot('detrend2d_chunk').chunk1d('128MiB')
        >>> displacement = data.detrend1d()
        """
        import dask
        from .utils_dask import rechunk2d

        target_mb = _parse_budget(budget) if budget is not None else None

        if p2p:
            ctx = dask.config.set({
                "array.rechunk.method": "p2p",
                "optimization.fuse.active": False,
            })
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            result = {}
            for k, ds in self.items():
                # Find the largest dim-0 among 3D variables
                n_stack = 0
                sample = None
                for var in ds.data_vars:
                    arr = ds[var]
                    if arr.ndim == 3 and arr.dims[-2:] == ('y', 'x'):
                        if arr.shape[0] > n_stack:
                            n_stack = arr.shape[0]
                            sample = arr
                if sample is None:
                    result[k] = ds
                    continue
                # Divide budget by n_stack to get per-slice budget,
                # then use rechunk2d (a 2D function) with per-slice element_bytes=8.
                per_slice_mb = (target_mb / n_stack) if target_mb is not None else None
                y_size, x_size = sample.shape[1], sample.shape[2]
                in_chunks = (sample.data.chunks[1], sample.data.chunks[2]) if hasattr(sample.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                   input_chunks=in_chunks, target_mb=per_slice_mb, merge=True)
                rechunked_vars = {}
                for var in ds.data_vars:
                    arr = ds[var]
                    if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                        continue
                    if arr.ndim == 3:
                        var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                    else:
                        var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                    rechunked_vars[var] = arr.chunk(var_chunks)
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                result[k] = ds
        return type(self)(result)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def map(self, func, *args, **kwargs):
        return type(self)({k: func(ds, *args, **kwargs) for k, ds in self.items()})

    def to_dict(self) -> dict:
        """
        Extract data variables as a dictionary of {burst_key: {var_name: values}}.

        Returns
        -------
        dict
            Dictionary mapping burst keys to dictionaries of variable names to numpy arrays.

        Examples
        --------
        >>> bpr = stack.BPR(stack.isel(date=[1]), stack.isel(date=[0]))
        >>> bpr.to_dict()
        {'123_262885_IW2': {'BPR': array([-158.75])},
         '123_262886_IW2': {'BPR': array([-158.81])},
         '123_262887_IW2': {'BPR': array([-158.87])}}
        """
        result = {}
        for key, ds in self.items():
            result[key] = {var: ds[var].values for var in ds.data_vars}
        return result

    def compute(self):
        """
        Compute lazy data in the batch.

        Persists all bursts at once via dask.persist(). Data stays in
        distributed worker memory (not pulled to client), letting the scheduler
        optimize across the full graph. Rechunks results to match input chunk
        structure. For memory-constrained sequential processing, use snapshot().

        Returns
        -------
        BatchCore
            New batch with computed data, rechunked to match input.
        """
        import dask
        from insardev_toolkit.progressbar import progressbar

        # Save input chunk structure per burst
        all_input_chunks = {}
        for key, ds in self.items():
            ic = {}
            for var_name in ds.data_vars:
                arr = ds[var_name]
                if hasattr(arr.data, 'chunks'):
                    ic[var_name] = dict(zip(arr.dims, arr.data.chunks))
            all_input_chunks[key] = ic

        # Persist all bursts at once — single scheduler submission
        # progressbar extracts futures and blocks until completion
        progressbar(result := dask.persist(dict(self))[0], desc='Computing Batch...'.ljust(25))

        # Finalize: materialize coordinates and rechunk to match input
        computed = {}
        for key, ds in result.items():
            new_coords = {}
            for name, coord in ds.coords.items():
                if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                    new_coords[name] = (coord.dims, coord.compute().values)
            if new_coords:
                ds = ds.assign_coords(new_coords)
            input_chunks = all_input_chunks[key]
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
        return type(self)(computed)

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
            Index is (fullBurstID, burst) matching Stack.to_dataframe.
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

        # Detect spatial data variables (skip 1D/0D vars like converted attributes)
        spatial_vars = [v for v in sample.data_vars if sample[v].ndim >= 2]
        ndims = {sample[v].ndim for v in spatial_vars}
        if len(ndims) > 1:
            raise ValueError(f'Mixed 2D and 3D variables not supported: {{{", ".join(f"{v}: {sample[v].ndim}D" for v in spatial_vars)}}}')

        # Detect dimension: 'date' for BatchComplex, 'pair' for others, None for spatial-only
        if 'date' in sample.dims:
            dim = 'date'
        elif 'pair' in sample.dims:
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
                # Get spatial data variables (skip 1D/0D vars)
                spatial_vars = [v for v in ds.data_vars if ds[v].ndim >= 2]
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

        If the data is already at or finer than the requested spacing,
        returns the input unchanged.
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
        for v in sample_ds.data_vars:
            arr = sample_ds[v]
            if arr.ndim >= 2 and arr.dims[-2:] == ('y', 'x') and hasattr(arr.data, 'chunks'):
                cy0 = arr.data.chunks[-2][0]
                cx0 = arr.data.chunks[-1][0]
                output_budget_mb = cy0 * cx0 * 8 / (1024 * 1024) / yscale / xscale
                break

        result = self.coarsen({'y': yscale, 'x': xscale}, boundary='trim').mean()

        # Rechunk output to preserve input spatial granularity
        if output_budget_mb is not None:
            sample_ds = next(iter(result.values()))
            for v in sample_ds.data_vars:
                arr = sample_ds[v]
                if arr.ndim >= 2 and arr.dims[-2:] == ('y', 'x') and hasattr(arr.data, 'chunks'):
                    y_size, x_size = arr.shape[-2], arr.shape[-1]
                    optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                       target_mb=output_budget_mb)
                    result = result.chunk({'y': optimal['y'], 'x': optimal['x']})
                    break

        return result

    def save(self, store: str, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Saving...', n_bursts: int = 2, debug=False):
        return utils_io.save(self, store=store, storage_options=storage_options, compat=False, caption=caption, n_bursts=n_bursts, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, compat=False, n_jobs=n_jobs, debug=debug)
        if not isinstance(data, dict):
            raise ValueError(f'ERROR: open() returns multiple datasets, you need to use Stack class to open them.')
        return data
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Snapshotting...',
                n_bursts: int = 2, debug=False, **kwargs):
        # Only save if this batch has data; otherwise just open existing store
        if len(self) > 0:
            utils_io.save(self, store=store, storage_options=storage_options, caption=caption, n_bursts=n_bursts, debug=debug)
        return utils_io.open(store=store, storage_options=storage_options, compat=False,
                            n_jobs=-1, debug=debug)

    def to_dataset(self, polarization=None, chunks='auto', compute: bool = False, debug: bool = False):
        """
        Merge multiple burst DataArrays into a single unified grid.

        This function efficiently combines bursts using dask delayed operations
        for lazy evaluation. For each output chunk, it selects the minimal set
        of input bursts needed and combines their data using forward-fill.

        For best results with overlapping bursts, call .dissolve() first to average
        values in overlap regions, then call .to_dataset() to merge into a single grid.

        Parameters
        ----------
        polarization : str, optional
            Specific polarization to process. If None, processes all polarizations.
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
        xr.DataArray or xr.Dataset
            Merged data on a unified grid.

        Examples
        --------
        >>> # Merge bursts into single grid (fast, uses ffill for overlaps)
        >>> merged = batch.to_dataset()
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
        from insardev_toolkit import progressbar, datagrid

        if not len(self):
            return None

        sample = next(iter(self.values()))
        if len(self) == 1:
            if debug:
                print(f"=== to_dataset() debug ===")
                print(f"Single burst - returning directly (no merge needed)")
                print(f"Burst key: {next(iter(self.keys()))}")
                # Get spatial vars
                spatial_vars = [v for v in sample.data_vars if 'y' in sample[v].dims and 'x' in sample[v].dims]
                for var in spatial_vars[:3]:  # Show first 3 spatial vars
                    da = sample[var]
                    print(f"  {var}: shape={da.shape}, dtype={da.dtype}")
            if compute:
                progressbar(sample := sample.persist(), desc=f'Compute Dataset'.ljust(25))
                return sample
            return sample

        # Determine which polarizations to process
        if polarization is None:
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            polarizations = [v for v in sample.data_vars
                            if 'y' in sample[v].dims and 'x' in sample[v].dims]
        else:
            polarizations = [polarization]

        # Build data dictionary: {pol: [burst0_data, burst1_data, ...]}
        burst_keys = list(self.keys())
        datas_by_pol = {pol: [self[k][pol] for k in burst_keys] for pol in polarizations}

        # Get grid info from first polarization (all pols have same grid)
        first_pol = polarizations[0]
        first_datas = datas_by_pol[first_pol]

        # Get signed spacing from first burst
        dims = first_datas[0].dims
        stackvar = list(dims)[0] if len(dims) > 2 else None

        # Handle stack dimension - preserve original coordinate values
        if stackvar is not None:
            stackval = first_datas[0][stackvar].values
        else:
            stackvar = 'fake'
            stackval = [0]
            # Expand dims for all polarizations
            for pol in polarizations:
                datas_by_pol[pol] = [d.expand_dims({stackvar: [0]}) for d in datas_by_pol[pol]]
            first_datas = datas_by_pol[first_pol]

        n_stack = len(stackval)

        # Filter out empty bursts (e.g., after spatial .sel() subsetting)
        nonempty_indices = [i for i, ds in enumerate(first_datas) if ds.y.size > 0 and ds.x.size > 0]
        if not nonempty_indices:
            return None
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

        # Sort each chunk's burst list by coverage (descending)
        for key in chunk_index:
            chunk_index[key].sort(key=lambda x: -x[1])

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

        # Build result for each polarization
        # Note: merge function is defined at module level (_merge_tiles_for_dask)
        # to avoid dask serialization issues with nested function closures
        results = {}
        for pol in polarizations:
            datas = datas_by_pol[pol]

            # Ensure data is dask and rechunk stack dim to 1 (keep spatial chunks)
            def ensure_dask_rechunked(d):
                arr = d.data
                if not isinstance(arr, da.Array):
                    arr = da.from_array(arr, chunks=arr.shape)
                return arr.rechunk({0: 1})

            datas_rechunked = [ensure_dask_rechunked(d) for d in datas]

            # Build output blocks for each stack slice separately
            stack_mosaics = []
            for s_idx in range(n_stack):
                # Build 2D mosaic for this stack slice
                blocks_rows = []
                for yi in range(n_y_chunks):
                    yb0 = yi * y_chunk_size
                    yb1 = min(yb0 + y_chunk_size, ys.size)
                    blocks_row = []
                    for xi in range(n_x_chunks):
                        xb0 = xi * x_chunk_size
                        xb1 = min(xb0 + x_chunk_size, xs.size)

                        # O(1) lookup of overlapping bursts via spatial index
                        overlapping = chunk_index.get((yi, xi), [])

                        out_shape = (yb1 - yb0, xb1 - xb0)

                        # Output chunk coordinates
                        out_ys = ys[yb0:yb1]
                        out_xs = xs[xb0:xb1]

                        if len(overlapping) == 0:
                            # No data - create NaN block
                            block = da.full(out_shape, np.nan, dtype=fill_dtype)
                        else:
                            # Collect delayed tile references and their offsets within this chunk
                            tiles_delayed = []
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
                                    by0, by1 = idx_y[0], idx_y[-1] + 1
                                    bx0, bx1 = idx_x[0], idx_x[-1] + 1

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

                                    # Slice the dask array to get this tile
                                    # Then rechunk to single spatial chunk for the merge function
                                    tile_slice = datas_rechunked[burst_idx][s_idx:s_idx+1, by0:by1, bx0:bx1].rechunk({1: -1, 2: -1})
                                    # Convert to delayed - dask will auto-compute when merge is called
                                    tile_delayed = tile_slice.to_delayed(optimize_graph=False).ravel()[0]
                                    tiles_delayed.append(tile_delayed)
                                    offsets.append((y_off, x_off))

                            if len(tiles_delayed) == 0:
                                block = da.full(out_shape, np.nan, dtype=fill_dtype)
                            else:
                                # Merge tiles - dask auto-computes delayed tiles before calling
                                # Tiles arrive as 3D (1, ny, nx), squeeze to 2D in merge
                                # IMPORTANT: Use standalone function instead of lambda with closure
                                # to avoid serialization issues with nested functions in dask distributed
                                # IMPORTANT: Convert to tuples (immutable) before passing to delayed
                                # This ensures proper serialization in distributed environments
                                # and prevents any race conditions with list mutations
                                block = da.from_delayed(
                                    dask.delayed(_merge_tiles_for_dask, pure=True)(
                                        tuple(tiles_delayed), tuple(offsets), out_shape, fill_dtype
                                    ),
                                    shape=out_shape,
                                    dtype=fill_dtype
                                )

                        blocks_row.append(block)
                    blocks_rows.append(blocks_row)

                # Assemble 2D mosaic for this stack slice
                mosaic_2d = da.block(blocks_rows)
                stack_mosaics.append(mosaic_2d[np.newaxis, :, :])

            # Stack all slices along axis 0
            data = da.concatenate(stack_mosaics, axis=0)

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
                result = result.isel({stackvar: 0})
            results[pol] = result

        # Return DataArray if single polarization was requested, Dataset otherwise
        if polarization is not None:
            # Single polarization explicitly requested - return DataArray
            output = results[polarization]
        else:
            # All polarizations - return Dataset (even if only one)
            output = xr.merge(list(results.values()))

        if compute:
            progressbar(output := output.persist(), desc=f'Computing Dataset...'.ljust(25))
        return output

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

        # Merge to single dataset
        ds = self.to_dataset()

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
            topo_merged = tfm[['ele']].to_dataset()
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
        merged = self.to_dataset(compute=True)
        if isinstance(merged, xr.DataArray):
            merged = merged.to_dataset()

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
                tfm_ds = tfm[k][['ele']]
                tgt_ds = self[k]
                y_idx = _nearest_indices(tfm_ds.y.values, tgt_ds.y.values)
                x_idx = _nearest_indices(tfm_ds.x.values, tgt_ds.x.values)
                selected = tfm_ds.isel(y=y_idx, x=x_idx)
                selected = selected.assign_coords(y=tgt_ds.y, x=tgt_ds.x)
                decimated[k] = selected
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

        with tqdm(total=len(self), desc='Exporting VTK') as pbar:
            for burst, ds in self.items():
                if not ds.data_vars:
                    pbar.update(1)
                    continue

                data_var = next(iter(ds.data_vars))
                base_da = ds[data_var]

                if 'pair' in ds.dims:
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
                    base_da_item = ds_item[data_var]
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
        from insardev_toolkit import progressbar

        # no data means no plot and no error
        if not len(self):
            return

        wrap = True if type(self) == BatchWrap else False

        # use outer variables
        def plot_polarization(polarization):
            stackvar = list(sample[polarization].dims)[0] if len(sample[polarization].dims) > 2 else None

            # Calculate decimation factors from batch extent (without materializing full grid)
            batch = self[[polarization]]
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
            da = batch_decimated.to_dataset()[polarization]
            if stackvar is None:
                stackvar = 'fake'
                da = da.expand_dims({stackvar: [0]})

            # materialize for all the calculations and plotting
            progressbar(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

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
            fg.fig.suptitle(f'{polarization} {caption or ''}'.strip(), y=y)

            # fg is the FacetGrid returned by xarray.plot.imshow
            # Get original limits from first axis before any modifications
            if flip:
                first_ax = fg.axes.flatten()[0]
                orig_xlim = first_ax.get_xlim()
                orig_ylim = first_ax.get_ylim()
                # Ensure we flip to reversed order (max, min)
                flipped_xlim = (max(orig_xlim), min(orig_xlim))
                flipped_ylim = (max(orig_ylim), min(orig_ylim))

            for idx, ax in enumerate(fg.axes.flatten()):
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
            batch = self[[pol1, pol2]]
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
            progressbar([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

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
        # find all variables in the first dataset related to polarizations
        # TODO
        #polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
        polarizations = list(sample.data_vars)
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
        weight: BatchUnit | None = None,
        wavelength: float | None = None,
        threshold: float = 0.5,
        device: str = 'auto',
        debug: bool = False
    ) -> Batch:
        """
        2D (yx) Gaussian kernel smoothing (multilook) on each dataset in this Batch.

        Parameters
        ----------
        weight : BatchUnit or None
            A Batch of 2D DataArrays, one per key, matching this Batch's keys.
            If None, no weighting is applied.
        wavelength : float or None
            Gaussian sigma via 5.3 cutoff formula. Must be positive if provided.
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

        # validate weight if provided
        if weight is not None:
            if not isinstance(weight, BatchUnit) or set(weight.keys()) != set(self.keys()):
                raise ValueError('`weight` must be a BatchUnit with the same keys as `self`')

        # Validate lazy data
        BatchCore._require_lazy(self, 'gaussian')

        # precompute pixel sizes for decimation
        dy, dx = self.spacing

        # validate wavelength if provided
        if wavelength is not None:
            if wavelength <= 0:
                raise ValueError('wavelength must be positive')
            sig_y = wavelength / (dy * cutoff)
            sig_x = wavelength / (dx * cutoff)
            if debug:
                print(f'DEBUG: multilooking sigmas ({sig_y:.2f}, {sig_x:.2f}), wavelength {wavelength:.1f}')
            sigmas = (sig_y, sig_x)
        else:
            sigmas = None

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
            for var in ds.data_vars:
                data_arr = ds[var]
                # Skip non-spatial variables
                if not (data_arr.ndim in (2, 3) and data_arr.dims[-2:] == ('y', 'x')):
                    continue

                is_complex = np.issubdtype(data_arr.dtype, np.complexfloating)
                out_dtype = np.complex64 if is_complex else np.float32

                # Get weight dask array for this variable
                weight_dask = w[var].data if w is not None and var in w.data_vars else None

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

            new_ds = xr.Dataset(new_vars)
            new_ds.attrs = ds.attrs
            out[key] = new_ds

        return type(self)(out)

    def residuals(self, polarization: str | None = None, debug: bool = False) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with fit(),
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

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        # Collect burst extents and detect pair dimension
        ids = sorted(self.keys())

        # Auto-detect polarization if not specified
        sample_ds = self[ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
        available_pols = [v for v in sample_ds.data_vars
                         if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
        if polarization is None:
            polarization = available_pols[0]
        if polarization not in available_pols:
            raise ValueError(f"Polarization '{polarization}' not found. Available: {available_pols}")

        sample_da = sample_ds[polarization]
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
            da = ds[polarization]
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
            i1 = self[id1][polarization]
            i2 = self[id2][polarization]

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
            valid = phase_diff.values.ravel()
            valid = valid[np.isfinite(valid)]

            if len(valid) == 0:
                continue

            valid = maybe_wrap(valid)
            median_diff = np.median(valid)
            abs_discrepancy = np.abs(maybe_wrap(median_diff))
            weight = len(valid)

            results.append((pair_idx, abs_discrepancy, weight, id1, id2, median_diff))

        # Aggregate results per pair and per subswath
        total_weights = [0.0] * n_pairs
        weighted_sums = [0.0] * n_pairs

        # Per-subswath tracking for debug
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
        per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            weighted_sums[pair_idx] += abs_discrepancy * weight
            total_weights[pair_idx] += weight
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

        discrepancies = []
        for p in range(n_pairs):
            if total_weights[p] == 0:
                discrepancies.append(0.0)
            else:
                discrepancies.append(round(weighted_sums[p] / total_weights[p], 3))

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

    def fit(self,
            degree: int = 0,
            method: str = 'median',
            polarization: str | None = None,
            debug: bool = False,
            return_residuals: bool = False):
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

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = intfs.fit(degree=0)
        >>> intfs1 = intfs - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = intfs1.fit(degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = intfs2.fit(degree=0)
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
            raise TypeError(f"fit() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

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
            print(f'fit(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s), pol={polarization}', flush=True)

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
        def _fit_all(*burst_data_arrays):
            import xarray as xr
            from scipy import sparse as _sparse
            from scipy.sparse.linalg import lsqr as _lsqr
            from scipy.sparse.csgraph import connected_components as _cc

            # Compute overlap diffs and process statistics
            _pbp = {p: [] for p in range(n_pairs)}
            for id1, id2 in all_overlap_pairs:
                i1_idx = id_to_idx[id1]
                i2_idx = id_to_idx[id2]
                d1 = np.asarray(burst_data_arrays[i1_idx])
                d2 = np.asarray(burst_data_arrays[i2_idx])

                for pair_idx in range(n_pairs):
                    d1_p = d1[pair_idx] if has_pair_dim else d1
                    d2_p = d2[pair_idx] if has_pair_dim else d2

                    # Build xarray DataArrays for coordinate-aware overlap
                    da1 = xr.DataArray(d1_p, dims=['y', 'x'],
                                       coords={'y': burst_y[i1_idx],
                                               'x': burst_x[i1_idx]})
                    da2 = xr.DataArray(d2_p, dims=['y', 'x'],
                                       coords={'y': burst_y[i2_idx],
                                               'x': burst_x[i2_idx]})
                    diff = da2 - da1
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

            return {'offsets': offsets, 'residuals': residuals}

        # Single delayed call — dask resolves burst data arrays before calling.
        # Graph has ~N_bursts layers (not ~N_overlaps*3 from xarray diffs).
        solve_result = dask.delayed(_fit_all, pure=True)(*burst_data)

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

        if return_residuals:
            # Residuals require concrete values — triggers the solve chain
            print('fit(return_residuals=True): computing residuals breaks lazy chain, use for diagnostics only', flush=True)
            residuals_out = solve_result['residuals'].compute()
            if debug:
                print(f'Input residuals: {residuals_out}', flush=True)
            return (coeffs, residuals_out)

        return coeffs

    def align(self,
              degree: int = 0,
              method: str = 'median',
              polarization: str | None = None,
              debug: bool = False,
              return_residuals: bool = False):
        """
        Align burst interferograms by removing phase offsets and optionally ionospheric ramps.

        Uses a multi-step approach for optimal alignment:
        - degree=0: Single-step offset correction
        - degree=1: 3-step correction (offset → ramp → re-offset) for ionospheric ramp removal

        The 3-step approach produces consistent fringes across bursts by removing
        per-track ionospheric ramps, which is essential for deformation analysis.

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
                Aligned interferograms with phase corrections applied.
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
        from .Batch import Batch, BatchWrap

        # Validate class type
        if not isinstance(self, (Batch, BatchWrap)):
            raise TypeError(f"align() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        # Auto-detect polarization if not specified
        if polarization is None:
            ids = list(self.keys())
            sample_ds = self[ids[0]]
            # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
            available_pols = [v for v in sample_ds.data_vars
                             if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
            polarization = available_pols[0]

        if degree == 0:
            # Single-step offset correction
            if debug:
                print('align(degree=0): single-step offset correction', flush=True)
                res_in = self.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            offsets = self.fit(degree=0, method=method, polarization=polarization, debug=debug)
            aligned = self - offsets

            if debug or return_residuals:
                res_out = aligned.residuals(polarization=polarization)
                if debug:
                    print(f'Output residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        elif degree == 1:
            # 3-step offset-ramp-offset correction
            if debug:
                print('align(degree=1): 3-step offset-ramp-offset correction', flush=True)
                res_in = self.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            # Step 1: Estimate offsets
            if debug:
                print('\nStep 1: Estimate offsets...', flush=True)
            offsets1 = self.fit(degree=0, method=method, polarization=polarization, debug=debug)
            intfs1 = self - offsets1
            if debug:
                res1 = intfs1.residuals(polarization=polarization)
                print(f'Residuals after step 1: {res1}', flush=True)

            # Step 2: Estimate ramps (uses same-track overlaps only)
            if debug:
                print('\nStep 2: Estimate ramps...', flush=True)
            ramps = intfs1.fit(degree=1, method=method, polarization=polarization, debug=debug)
            intfs2 = intfs1 - intfs1.polyval(ramps)
            if debug:
                res2 = intfs2.residuals(polarization=polarization)
                print(f'Residuals after step 2: {res2}', flush=True)

            # Step 3: Re-estimate offsets
            if debug:
                print('\nStep 3: Re-estimate offsets...', flush=True)
            offsets2 = intfs2.fit(degree=0, method=method, polarization=polarization, debug=debug)

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

            aligned = self - self.polyval(coeffs)

            if debug or return_residuals:
                res_out = aligned.residuals(polarization=polarization)
                if debug:
                    print(f'Final residuals: {res_out}', flush=True)

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
        sample = self[burst_ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes
        polarizations = [v for v in sample.data_vars
                        if 'y' in sample[v].dims and 'x' in sample[v].dims]

        if debug:
            import time
            t0 = time.time()
            print(f'dissolve: {len(self)} bursts, wrap={wrap}, extend={extend}, weight={weight}', flush=True)

        # Build STRtree for fast spatial queries
        first_pol = polarizations[0]
        burst_extents = tuple(
            (float(self[bid][first_pol].y.min()), float(self[bid][first_pol].y.max()),
             float(self[bid][first_pol].x.min()), float(self[bid][first_pol].x.max()))
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

            ds_others = [self[burst_ids[idx]] for idx in overlapping_indices]

            new_ds = ds_current.copy()
            for pol in polarizations:
                da_current = ds_current[pol]
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

                delayed_result = dask.delayed(_dissolve_raw_for_dask, pure=True)(
                    current_arr, current_y, current_x,
                    others_arrs, others_ys, others_xs,
                    wrap, extend, weight
                )
                delayed_array = da.from_delayed(
                    delayed_result,
                    shape=da_current.shape,
                    dtype=da_current.dtype
                )
                new_ds[pol] = da_current.copy(data=delayed_array)

            output[bid] = new_ds

        if debug:
            print(f'dissolve: preparation done in {time.time() - t0:.1f}s', flush=True)

        return type(self)(output)
