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
from .utils_torch import serialize_gpu, GPU_LOCK
from .BatchCore import BatchCore
from . import utils_unwrap2d
import numpy as np
import rioxarray
import threading
from contextlib import nullcontext
from . import utils_stl
from insardev_toolkit import progressbar
from .utils_vtk import as_vtk as _as_vtk
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex, Batches
from . import utils_io
from . import utils_xarray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import rasterio as rio
    import pandas as pd
    import xarray as xr

# GPU work serializes on the ONE process-wide lock in utils_torch;
# CPU work still uses a semaphore to cap parallelism (see below).

_irls_cpu_semaphores = {}

_irls_cpu_semaphores_lock = threading.Lock()

def _get_irls_semaphore(value):
    """Get or create a per-process CPU semaphore with the given value."""
    with _irls_cpu_semaphores_lock:
        if value not in _irls_cpu_semaphores:
            _irls_cpu_semaphores[value] = threading.Semaphore(value)
        return _irls_cpu_semaphores[value]

def _irls_process_chunk_with_tuple(phase_chunk, weight_chunk, params_tuple):
    """Module-level function for IRLS processing with params as tuple.

    Returns tuple of (unwrapped, conncomp) arrays, each with shape (1, y, x).
    """
    # Unpack params tuple
    device, max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug, semaphore = params_tuple

    if debug:
        print(f'DEBUG _irls_process_chunk: phase_chunk.shape={phase_chunk.shape}, device={device!r}')

    # Squeeze to 2D for processing - handle both (1, y, x) and (y, x) input
    if phase_chunk.ndim == 3:
        phase_2d = phase_chunk[0]
        weight_2d = weight_chunk[0] if weight_chunk is not None else None
    elif phase_chunk.ndim == 2:
        phase_2d = phase_chunk
        weight_2d = weight_chunk
    else:
        raise ValueError(f"Expected 2D or 3D phase_chunk, got shape {phase_chunk.shape}")

    if debug:
        print(f'DEBUG _irls_process_chunk: phase_2d.shape={phase_2d.shape}')

    if np.all(np.isnan(phase_2d)):
        if debug:
            print('  All NaN, skipping')
        unwrapped = phase_2d.astype(np.float32)
        conncomp = np.zeros_like(phase_2d, dtype=np.uint16)
        return unwrapped[np.newaxis, ...], conncomp[np.newaxis, ...]

    sem = _get_irls_semaphore(semaphore) if str(device) == 'cpu' else GPU_LOCK
    with sem:
        unwrapped, conncomp = utils_unwrap2d.irls_unwrap_2d(
            phase_2d, weight=weight_2d, device=device,
            max_iter=max_iter, tol=tol, cg_max_iter=cg_max_iter,
            cg_tol=cg_tol, epsilon=epsilon, conncomp_size=conncomp_size, debug=debug
        )
    # Add pair dim back: (y, x) -> (1, y, x)
    return unwrapped[np.newaxis, ...], conncomp[np.newaxis, ...]



def _irls_process_no_weight_conncomp(phase_chunk, params_tuple):
    """Process phase chunk without weight. Returns stacked (1, 2, y, x)."""
    unwrapped, conncomp = _irls_process_chunk_with_tuple(phase_chunk, None, params_tuple)
    # Stack: (1, y, x) + (1, y, x) -> (1, 2, y, x) for blockwise with 'pcyx' output
    stacked = np.stack([unwrapped[0].astype(np.float32), conncomp[0].astype(np.float32)], axis=0)
    return stacked[np.newaxis, ...]  # (1, 2, y, x)

def _irls_process_with_weight_conncomp(phase_chunk, weight_chunk, params_tuple):
    """Process phase chunk with weight. Returns stacked (1, 2, y, x)."""
    unwrapped, conncomp = _irls_process_chunk_with_tuple(phase_chunk, weight_chunk, params_tuple)
    # Stack: (1, y, x) + (1, y, x) -> (1, 2, y, x) for blockwise with 'pcyx' output
    stacked = np.stack([unwrapped[0].astype(np.float32), conncomp[0].astype(np.float32)], axis=0)
    return stacked[np.newaxis, ...]  # (1, 2, y, x)

class Stack(BatchComplex):

    _STRUCTURE_4CONN = utils_unwrap2d.STRUCTURE_4CONN

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

    def _reorder_conncomp_by_size(self, conncomp_labels):
        """
        Reorder connected component labels by size (largest=1, smallest=max).

        Parameters
        ----------
        conncomp_labels : BatchUnit
            Batch of connected component labels.

        Returns
        -------
        BatchUnit
            Batch with reordered labels (1=largest, 2=second largest, etc.).
        """
        import xarray as xr
        import dask.array
        from .Batch import BatchUnit

        def _reorder_2d(labels_2d):
            """Reorder labels in a single 2D array."""
            # Handle (1, y, x) arrays from blockwise
            squeeze = False
            if labels_2d.ndim == 3 and labels_2d.shape[0] == 1:
                labels_2d = labels_2d[0]
                squeeze = True

            # Get unique labels (excluding 0 and NaN)
            valid_mask = ~np.isnan(labels_2d) & (labels_2d > 0)
            if not np.any(valid_mask):
                result = labels_2d.astype(np.float32)
                return result[np.newaxis, ...] if squeeze else result

            unique_labels = np.unique(labels_2d[valid_mask])
            if len(unique_labels) == 0:
                result = labels_2d.astype(np.float32)
                return result[np.newaxis, ...] if squeeze else result

            # Count pixels per label
            sizes = []
            for label in unique_labels:
                sizes.append(np.sum(labels_2d == label))

            # Sort by size (descending) and create mapping
            sorted_indices = np.argsort(sizes)[::-1]
            label_mapping = {}
            for new_label, idx in enumerate(sorted_indices, start=1):
                old_label = unique_labels[idx]
                label_mapping[old_label] = new_label

            # Apply mapping
            result = np.zeros_like(labels_2d)
            result[~valid_mask] = np.nan
            for old_label, new_label in label_mapping.items():
                result[labels_2d == old_label] = new_label

            result = result.astype(np.float32)
            return result[np.newaxis, ...] if squeeze else result

        # Process each dataset in the batch
        result = {}
        for key in conncomp_labels.keys():
            ds = conncomp_labels[key]
            data_vars = list(ds.data_vars)

            reordered_vars = {}
            for var in data_vars:
                data_arr = ds[var]

                # Use da.blockwise for efficient dask integration
                dask_data = data_arr.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(dask_data.ndim))

                # Provide meta to avoid calling _reorder_2d during graph construction
                meta = np.empty((0,) * dask_data.ndim, dtype=np.float32)
                result_dask = dask.array.blockwise(
                    _reorder_2d, dim_str,
                    dask_data, dim_str,
                    dtype=np.float32,
                    meta=meta,
                )

                reordered_da = xr.DataArray(
                    result_dask,
                    dims=data_arr.dims,
                    coords=data_arr.coords
                )
                reordered_vars[var] = reordered_da

            result[key] = xr.Dataset(Stack._carry_meta(ds, reordered_vars),
                                     coords=ds.coords, attrs=ds.attrs)

        return BatchUnit(result)

    def _compute_conncomp_labels(self, phase):
        """
        Compute connected component labels from phase data.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets.

        Returns
        -------
        BatchUnit
            Batch of connected component labels (int32).
        """
        import dask
        import dask.array as da
        import xarray as xr
        from .Batch import BatchUnit

        result = {}
        for key in phase.keys():
            phase_ds = phase[key]
            data_vars = [v for v in phase_ds.data_vars
                        if 'y' in phase_ds[v].dims and 'x' in phase_ds[v].dims]

            label_vars = {}
            for var in data_vars:
                phase_da = phase_ds[var]

                def compute_labels(phase_chunk):
                    """Compute connected components for a chunk."""
                    if phase_chunk.ndim == 3:
                        # (pair, y, x) -> process each pair
                        result = np.zeros_like(phase_chunk, dtype=np.int32)
                        for i in range(phase_chunk.shape[0]):
                            result[i] = utils_unwrap2d.conncomp_2d(phase_chunk[i])
                        return result
                    else:
                        return utils_unwrap2d.conncomp_2d(phase_chunk).astype(np.int32)

                dask_data = phase_da.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(dask_data.ndim))
                meta = np.empty((0,) * dask_data.ndim, dtype=np.int32)

                result_dask = da.blockwise(
                    compute_labels, dim_str,
                    dask_data, dim_str,
                    dtype=np.int32,
                    meta=meta,
                )

                label_da = xr.DataArray(
                    result_dask,
                    dims=phase_da.dims,
                    coords=phase_da.coords
                )
                label_vars[var] = label_da

            result[key] = xr.Dataset(Stack._carry_meta(phase_ds, label_vars),
                                     coords=phase_ds.coords, attrs=phase_ds.attrs)

        return BatchUnit(result)

    def _link_components(self, unwrapped, conncomp_labels=None, conncomp_size=100, conncomp_gap=None,
                         conncomp_linksize=5, conncomp_linkcount=30, debug=False):
        """
        Link disconnected components in unwrapped phase by finding optimal 2π offsets.

        Parameters
        ----------
        unwrapped : Batch
            Batch of unwrapped phase datasets.
        conncomp_labels : BatchUnit or None
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
        from scipy import ndimage
        from .Batch import Batch

        def _link_2d(phase_2d, conncomp_2d=None):
            """Link components in a single 2D array."""
            import time

            # Handle (1, y, x) arrays from blockwise
            squeeze = False
            if phase_2d.ndim == 3 and phase_2d.shape[0] == 1:
                phase_2d = phase_2d[0]
                squeeze = True
            if conncomp_2d is not None and conncomp_2d.ndim == 3 and conncomp_2d.shape[0] == 1:
                conncomp_2d = conncomp_2d[0]

            # Find connected components - use pre-computed or compute fresh
            valid_mask = ~np.isnan(phase_2d)
            if not np.any(valid_mask):
                return phase_2d[np.newaxis, ...] if squeeze else phase_2d

            min_size = max(conncomp_size, 4)
            if debug:
                print(f'  Component filtering: conncomp_size={conncomp_size}, min_size={min_size}')

            if conncomp_2d is not None:
                # Use pre-computed conncomp from IRLS (size-ordered: 1=largest, 2=second, etc.)
                labeled = conncomp_2d.astype(np.int32)
                n_total = int(labeled.max())
                sizes = np.bincount(labeled.ravel(), minlength=n_total + 1)
                slices = ndimage.find_objects(labeled)
                # Build components list - already size-ordered, just filter by min_size
                components = [
                    {'label': i, 'size': sizes[i], 'slices': slices[i - 1] if i > 0 and i <= len(slices) else None}
                    for i in range(1, n_total + 1)
                    if sizes[i] >= min_size and (i <= len(slices) and slices[i - 1] is not None)
                ]
                if debug:
                    print(f'  Using pre-computed conncomp: {n_total} components, {len(components)} after size filter')
            else:
                labeled, components, n_total, sizes = utils_unwrap2d.get_connected_components(valid_mask, min_size)

            # Filter small components BEFORE linking - set to NaN
            linked_labels = np.array([comp['label'] for comp in components])
            is_linked = np.isin(labeled, linked_labels) | (labeled == 0)
            phase_2d = phase_2d.copy()
            phase_2d[~is_linked] = np.nan

            if debug:
                n_filtered = n_total - len(components)
                print(f'  Filtered {n_filtered} small components (< {min_size} pixels)')

            if len(components) < 2:
                return phase_2d[np.newaxis, ...] if squeeze else phase_2d

            if debug:
                gap_str = 'unlimited' if conncomp_gap is None else str(conncomp_gap)
                print(f'  Linking {len(components)} components (conncomp_gap={gap_str})...')
                t0 = time.time()

            # Find connections using labeled array (memory efficient - no boolean masks)
            connections = utils_unwrap2d.find_component_connections_fast(
                labeled, phase_2d, components,
                conncomp_gap=conncomp_gap, max_neighbors=conncomp_linkcount,
                n_neighbors=conncomp_linksize
            )

            if debug:
                print(f'    Found {len(connections)} connections')

            if len(connections) == 0:
                return phase_2d[np.newaxis, ...] if squeeze else phase_2d

            # Apply ILP to find optimal offsets (using labeled array, memory efficient)
            result = utils_unwrap2d.connect_components_ilp_fast(
                phase_2d, labeled, components, connections,
                max_time=60.0, debug=debug
            )

            if debug:
                elapsed = time.time() - t0
                print(f'  Component linking done ({elapsed:.2f}s)')

            return result[np.newaxis, ...] if squeeze else result

        # Process each dataset in the batch
        result = {}
        for key in unwrapped.keys():
            ds = unwrapped[key]
            # Get corresponding conncomp dataset if available
            conncomp_ds = conncomp_labels[key] if conncomp_labels is not None and key in conncomp_labels else None
            data_vars = list(ds.data_vars)

            linked_vars = {}
            for var in data_vars:
                data_arr = ds[var]

                # Use da.blockwise for efficient dask integration
                dask_data = data_arr.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(dask_data.ndim))

                # Provide meta to avoid calling _link_2d during graph construction
                meta = np.empty((0,) * dask_data.ndim, dtype=np.float32)

                if conncomp_ds is not None and var in conncomp_ds:
                    # Pass conncomp to _link_2d
                    conncomp_dask = conncomp_ds[var].data
                    result_dask = dask.array.blockwise(
                        _link_2d, dim_str,
                        dask_data, dim_str,
                        conncomp_dask, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )
                else:
                    # No conncomp - will recompute inside _link_2d
                    result_dask = dask.array.blockwise(
                        lambda phase: _link_2d(phase, None), dim_str,
                        dask_data, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )

                linked_da = xr.DataArray(
                    result_dask,
                    dims=data_arr.dims,
                    coords=data_arr.coords
                )
                linked_vars[var] = linked_da

            result[key] = xr.Dataset(Stack._carry_meta(ds, linked_vars),
                                     coords=ds.coords, attrs=ds.attrs)

        return Batch(result)

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
        >>> linked = unwrapped.unwrap2d_link(conncomp_size=10_000, debug=True)

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
        from .Batch import Batch

        # Validate parameters
        if conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )

        # Link components using the existing method
        return self._link_components(
            phase, conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
            conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
            debug=debug
        )

    def unwrap2d(self, phase, weight=None, conncomp=False,
                conncomp_size=1_000, conncomp_gap=None,
                conncomp_linksize=5, conncomp_linkcount=30, union=False,
                device='auto', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        .. deprecated::
            Use ``phase.unwrap2d(weight=corr)`` on BatchWrap instead.
            This Stack method will be removed in a future version.

        Uses Iteratively Reweighted Least Squares with DCT-based preconditioner.
        GPU-accelerated using PyTorch (MPS on Apple Silicon, CUDA on NVIDIA,
        or CPU fallback).

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
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', 'cpu', or 'tpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            If True, print diagnostic information. Default is False.
        **kwargs
            Additional arguments passed to unwrap2d_irls:
            max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase (components linked).
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp)
            where conncomp labels are ordered by size (1=largest).

        Notes
        -----
        GPU/TPU acceleration:
        - tpu on Google Cloud TPU (experimental, requires torch_xla)
        - cuda on NVIDIA GPUs and AMD GPUs (via ROCm)
        - mps on Apple Silicon (M1/M2/M3/M4)
        - cpu fallback otherwise

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

        Force CPU processing:
        >>> unwrapped = stack.unwrap2d(intfs, corr, device='cpu')
        """
        # Validate parameters
        if not conncomp and conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )

        if union:
            return self._unwrap2d_union(
                phase, weight, conncomp=conncomp, conncomp_size=conncomp_size,
                conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                conncomp_linkcount=conncomp_linkcount, device=device,
                debug=debug, **kwargs)

        # Use IRLS unwrapping - always returns (unwrapped, conncomp)
        # Pass conncomp_size to filter small components during IRLS
        unwrapped, conncomp_labels = self.unwrap2d_irls(
            phase, weight, device=device, conncomp_size=conncomp_size, debug=debug, **kwargs
        )

        if conncomp:
            # Return size-ordered conncomp labels (already computed by irls)
            return unwrapped, conncomp_labels
        else:
            # Link components using pre-computed conncomp from IRLS
            unwrapped = self._link_components(
                unwrapped, conncomp_labels=conncomp_labels,
                conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
                conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
                debug=debug
            )
            return unwrapped

    def _unwrap2d_union(self, phase, weight=None, conncomp=False,
                        conncomp_size=1000, conncomp_gap=None,
                        conncomp_linksize=5, conncomp_linkcount=30,
                        device='auto', debug=False, **kwargs):
        """One solve over the union of the bursts, returned on the burst grid.

        Unwrapping needs one raster, so the bursts are merged internally; the
        caller passes a Batch and gets a Batch back.
        """
        import numpy as np
        import xarray as xr
        from .Batch import BatchWrap, BatchUnit
        from .BatchCore import BatchCore

        _batch_in = isinstance(phase, BatchCore)
        phase_ds = phase.to_dataset() if _batch_in else phase
        weight_ds = (weight.to_dataset()
                     if isinstance(weight, BatchCore) else weight)
        if not isinstance(phase_ds, xr.Dataset):
            raise TypeError(f"phase must be a Batch or xr.Dataset, got "
                            f"{type(phase).__name__}")
        if weight_ds is not None and not isinstance(weight_ds, xr.Dataset):
            raise TypeError(f"weight must be a Batch or xr.Dataset, got "
                            f"{type(weight).__name__}")

        # the solve is one raster; this bends the merged view, not the caller's
        _spatial = {d: -1 for d in ('y', 'x') if d in phase_ds.dims}
        phase_ds = phase_ds.chunk(_spatial)
        if weight_ds is not None:
            weight_ds = weight_ds.chunk(_spatial)

        # the merged raster IS the single burst of this solve
        result = self.unwrap2d(
            BatchWrap({'': phase_ds}),
            BatchUnit({'': weight_ds}) if weight_ds is not None else None,
            conncomp=conncomp, conncomp_size=conncomp_size,
            conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
            conncomp_linkcount=conncomp_linkcount, union=False,
            device=device, debug=debug, **kwargs)
        merged = ([r[''] for r in result] if conncomp else [result['']])

        if not _batch_in:
            return tuple(merged) if conncomp else merged[0]

        # keep only each burst's own pixels; overlaps were filled from neighbours
        valid = phase.map_da(lambda da: da.notnull())
        out = [phase.from_dataset(m).where(valid) for m in merged]
        return tuple(out) if conncomp else out[0]

    def unwrap2d_irls(self, phase, weight=None, device='auto',
                      max_iter=50, tol=1e-2, cg_max_iter=10, cg_tol=1e-3, epsilon=1e-2,
                      conncomp_size=30, semaphore=8, debug=False):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        This algorithm provides high-quality unwrapping with L¹ norm that
        preserves discontinuities, and supports quality weighting from
        correlation data. GPU-accelerated using PyTorch (MPS on Apple Silicon,
        CUDA on NVIDIA, or CPU fallback).

        Uses GPU-accelerated DCT as initial solution, then refines it through
        weighted IRLS iterations. This handles phase residues properly by
        down-weighting inconsistent regions based on correlation.

        Disconnected components (separated by NaN regions) are unwrapped
        independently - the algorithm naturally handles this through edge
        weight zeroing where either adjacent pixel is invalid.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. Higher values indicate
            more reliable phase measurements.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', 'cpu', or 'tpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        max_iter : int, optional
            Maximum IRLS iterations. Default is 50.
        tol : float, optional
            Convergence tolerance for relative change. Default is 1e-2.
        cg_max_iter : int, optional
            Maximum conjugate gradient iterations per IRLS step. Default is 10.
        cg_tol : float, optional
            Conjugate gradient convergence tolerance. Default is 1e-3.
        epsilon : float, optional
            Smoothing parameter for L¹ approximation. Larger values improve
            numerical stability but reduce L¹ approximation quality. Default is 1e-2.
        conncomp_size : int, optional
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default is 30.
        semaphore : int, optional
            Maximum concurrent CPU IRLS tasks per process. Limits PyTorch thread
            contention in high-thread configs (e.g. 1w16t). Transparent when
            threads_per_worker < semaphore. Default is 8.
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, BatchUnit):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: BatchUnit of component labels (uint16, 0=invalid, 1=largest, 2=second, ...)

        Notes
        -----
        - GPU/TPU-accelerated using PyTorch (TPU via XLA, CUDA/ROCm, MPS, or CPU)
        - L¹ norm preserves discontinuities better than L² (DCT alone)
        - Correlation weighting handles phase residues properly
        - Provides consistent results across multi-burst data

        **Algorithm**: Uses a novel DCT+IRLS combination. See
        `utils_unwrap2d.irls_unwrap_2d` for algorithm details and references.
        """
        import dask
        import dask.array as da
        import torch
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        assert weight is None or isinstance(weight, BatchUnit), 'ERROR: weight should be a BatchUnit object'

        # Validate lazy data
        from .BatchCore import BatchCore
        BatchCore._require_lazy(phase, 'unwrap2d')

        # Resolve device using shared helper (handles Dask cluster resources)
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string
        device_name = device.upper()

        if debug:
            print(f'Using device: {device_name}')

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

            # Get data variables (typically polarization like 'VV'), with y/x dims - excludes converted attributes
            data_vars = [v for v in phase_ds.data_vars
                        if 'y' in phase_ds[v].dims and 'x' in phase_ds[v].dims]

            unwrap_vars = {}
            conncomp_vars = {}

            for var in data_vars:
                phase_da = phase_ds[var]
                weight_da = weight_ds[var] if weight_ds is not None else None

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

                # Use da.blockwise for efficient dask integration
                # With chunk={'pair': 1}, dask splits (n_pairs, y, x) into n_pairs chunks of (1, y, x)

                # Use da.blockwise for efficient chunk processing
                phase_dask = phase_da.data
                weight_dask = weight_da.data if weight_da is not None else None

                # Provide meta to avoid calling process_wrapper during graph construction
                meta = np.empty((0, 0, 0), dtype=np.float32)

                # Create params tuple - simple immutable structure for reliable serialization
                # Use str() to create a fresh string copy
                params_tuple = (str(device), max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug, semaphore)

                if debug:
                    print(f'DEBUG creating params tuple with device={params_tuple[0]!r}, semaphore={semaphore}')

                # Use conncomp versions that return stacked (1, 2, y, x) per chunk
                # Output shape is (n_pairs, 2, y, x)
                meta_stacked = np.empty((0, 0, 0, 0), dtype=np.float32)

                if weight_dask is None:
                    # Pass params tuple as scalar through blockwise
                    result_dask = dask.array.blockwise(
                        _irls_process_no_weight_conncomp, 'pcyx',
                        phase_dask, 'pyx',
                        params_tuple, None,  # None index means scalar (not broadcasted)
                        dtype=np.float32,
                        meta=meta_stacked,
                        new_axes={'c': 2},  # Output has new axis of size 2
                    )
                else:
                    result_dask = dask.array.blockwise(
                        _irls_process_with_weight_conncomp, 'pcyx',
                        phase_dask, 'pyx',
                        weight_dask, 'pyx',
                        params_tuple, None,
                        dtype=np.float32,
                        meta=meta_stacked,
                        new_axes={'c': 2},
                    )

                # Split stacked result: (n_pairs, 2, y, x) -> unwrapped (n_pairs, y, x), conncomp (n_pairs, y, x)
                unwrapped_dask = result_dask[:, 0, :, :]  # First channel is unwrapped phase
                conncomp_dask = result_dask[:, 1, :, :].astype(np.uint16)  # Second channel is conncomp

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

            # Preserve dataset attributes (subswath, pathNumber, etc.)
            unwrap_result[key] = xr.Dataset(Stack._carry_meta(phase_ds, unwrap_vars),
                                            attrs=phase_ds.attrs)
            conncomp_result[key] = xr.Dataset(Stack._carry_meta(phase_ds, conncomp_vars),
                                              attrs=phase_ds.attrs)
            # Preserve CRS from input dataset
            if phase_ds.rio.crs is not None:
                unwrap_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)
                conncomp_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)

        from .Batch import BatchUnit, Batches
        return Batches((Batch(unwrap_result), BatchUnit(conncomp_result)))

    @staticmethod
    def _detect_discontinuity_hough(phase, grad_threshold=2.0, mask_width=3, debug=False):
        """
        Detect discontinuity using Hough transform - masks the full detected line.

        Unlike hough_focal which tries to find the fault segment, this method
        masks the entire Hough line across the image. Use this when you want
        to see if the line would split the image.

        Parameters
        ----------
        phase : np.ndarray
            2D wrapped phase array
        grad_threshold : float
            Gradient threshold for Hough detection
        mask_width : int
            Half-width of mask
        debug : bool
            Print debug info

        Returns
        -------
        mask : np.ndarray
            Boolean mask
        info : dict
            Detection info including 'splits_image' flag
        """
        import cv2
        from scipy import ndimage

        phase = np.asarray(phase)
        height, width = phase.shape

        # Compute gradient magnitude using wrap-aware gradient
        dx, dy = utils_unwrap2d.wrapped_gradient(phase)
        grad_mag = np.sqrt(dx**2 + dy**2)

        # Binary edge image
        edges = (grad_mag > grad_threshold).astype(np.uint8) * 255
        nan_mask = np.isnan(phase)
        edges[nan_mask] = 0

        if not np.any(edges):
            return np.zeros((height, width), dtype=bool), {
                'angle': None, 'n_masked': 0, 'splits_image': False, 'n_components': 1
            }

        # Hough transform using OpenCV
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)

        if lines is None or len(lines) == 0:
            return np.zeros((height, width), dtype=bool), {
                'angle': None, 'n_masked': 0, 'splits_image': False, 'n_components': 1
            }

        # Get strongest line
        rho, theta = lines[0][0]

        if debug:
            print(f'  Hough line: rho={rho:.1f}, theta={np.rad2deg(theta):.1f}°')

        angle = theta
        dist = rho

        # Create mask along the FULL line (not just high-gradient segment)
        # OpenCV Hough: x*cos(theta) + y*sin(theta) = rho
        mask = np.zeros((height, width), dtype=bool)

        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        # Sample points along the line across the entire image
        if abs(sin_t) > abs(cos_t):
            # Line is more horizontal - iterate over x
            for x in range(width):
                y = int((dist - x * cos_t) / sin_t)
                if 0 <= y < height:
                    for ddy in range(-mask_width, mask_width + 1):
                        for ddx in range(-mask_width, mask_width + 1):
                            yy, xx = y + ddy, x + ddx
                            if 0 <= yy < height and 0 <= xx < width:
                                mask[yy, xx] = True
        else:
            # Line is more vertical - iterate over y
            for y in range(height):
                x = int((dist - y * sin_t) / cos_t)
                if 0 <= x < width:
                    for ddy in range(-mask_width, mask_width + 1):
                        for ddx in range(-mask_width, mask_width + 1):
                            yy, xx = y + ddy, x + ddx
                            if 0 <= yy < height and 0 <= xx < width:
                                mask[yy, xx] = True

        n_masked = np.sum(mask)

        # Check if mask splits image
        valid_region = ~mask
        labeled, n_components = ndimage.label(valid_region)
        splits_image = n_components > 1

        if debug:
            print(f'  Mask: {n_masked} pixels')
            if splits_image:
                print(f'  WARNING: Mask splits image into {n_components} disconnected regions!')

        info = {
            'angle': angle,
            'rho': dist,
            'n_masked': n_masked,
            'splits_image': splits_image,
            'n_components': n_components
        }

        return mask, info

    def unwrap2d_dataset_mask(self, phase, method='hough_focal', grad_threshold=2.0,
                               mask_width=3, debug=False):
        """
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

    @staticmethod
    def _snaphu_unwrap_array(phase_arr, corr_arr=None, defomax=0, debug=False):
        """
        Unwrap a single 2D phase array using SNAPHU via pipes.

        Parameters
        ----------
        phase_arr : np.ndarray
            2D wrapped phase array in radians.
        corr_arr : np.ndarray, optional
            2D correlation array (0-1). If None, uniform weight is used.
        defomax : float, optional
            Maximum expected deformation in cycles. 0 = smooth mode. Default 0.
        debug : bool, optional
            Print debug info. Default False.

        Returns
        -------
        unwrap_arr : np.ndarray
            2D unwrapped phase array.
        """
        import subprocess
        import os
        import tempfile

        nrow, ncol = phase_arr.shape

        # Build SNAPHU config
        # Note: NPROC only applies to tiled mode, non-tiled uses 1 CPU
        conf = f"""
INFILEFORMAT   FLOAT_DATA
OUTFILEFORMAT  FLOAT_DATA
CORRFILEFORMAT FLOAT_DATA
ALTITUDE       693000.0
EARTHRADIUS    6378000.0
NEARRANGE      831000
DR             18.4
DA             28.2
RANGERES       28
AZRES          44
LAMBDA         0.0554658
NLOOKSRANGE    1
NLOOKSAZ       1
NPROC          1
DEFOMAX_CYCLE  {defomax}
"""

        # Create temp directory for SNAPHU files
        with tempfile.TemporaryDirectory(prefix='snaphu_') as tmpdir:
            phase_file = os.path.join(tmpdir, 'phase.bin')
            mask_file = os.path.join(tmpdir, 'mask.bin')
            corr_file = os.path.join(tmpdir, 'corr.bin')
            unwrap_file = os.path.join(tmpdir, 'unwrap.bin')

            # Write phase (NaN -> 0)
            phase_filled = np.where(np.isnan(phase_arr), 0, phase_arr).astype(np.float32)
            phase_filled.tofile(phase_file)

            # Write mask (valid=1, NaN=0)
            mask = np.where(np.isnan(phase_arr), 0, 1).astype(np.uint8)
            mask.tofile(mask_file)

            # Build command
            argv = ['snaphu', phase_file, str(ncol), '-M', mask_file,
                    '-f', '/dev/stdin', '-o', unwrap_file, '-d']

            # Add correlation if provided
            if corr_arr is not None:
                corr_filled = np.where(np.isnan(corr_arr), 0, corr_arr).astype(np.float32)
                corr_filled.tofile(corr_file)
                argv.extend(['-c', corr_file])

            if debug:
                argv.append('-v')
                print(f'DEBUG: snaphu argv: {argv}')

            # Run SNAPHU
            p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, encoding='utf8')
            stdout, stderr = p.communicate(input=conf)

            if debug and stderr:
                print(f'DEBUG: snaphu stderr:\n{stderr}')

            # Read output
            if os.path.exists(unwrap_file):
                unwrap_arr = np.fromfile(unwrap_file, dtype=np.float32).reshape(nrow, ncol)
                # Restore NaN mask
                unwrap_arr = np.where(np.isnan(phase_arr), np.nan, unwrap_arr)
                # Remove mean (same as IRLS does for consistency)
                valid_mask = ~np.isnan(unwrap_arr)
                if np.any(valid_mask):
                    unwrap_arr[valid_mask] -= np.nanmean(unwrap_arr)
            else:
                if debug:
                    print(f'DEBUG: SNAPHU failed, output file not found')
                unwrap_arr = np.full((nrow, ncol), np.nan, dtype=np.float32)

        return unwrap_arr

    def unwrap2d_snaphu(self, phase, corr=None, defomax=0, debug=False):
        """
        Unwrap phase using SNAPHU algorithm.

        Simplified SNAPHU wrapper for single-burst processing.
        No tiling needed as bursts are small enough.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        corr : BatchUnit, optional
            Batch of correlation values. If None, uniform weight is used.
        defomax : float, optional
            Maximum expected deformation in cycles per pixel.
            0 = smooth mode (default), good for atmospheric/orbital signals.
            Use higher values (e.g., 1.2) for deformation with discontinuities.
        debug : bool, optional
            Print debug information. Default False.

        Returns
        -------
        Batch
            Batch of unwrapped phase datasets (lazy).

        Examples
        --------
        >>> intf, corr = stack.phasediff_multilook(pairs, wavelength=30)
        >>> unwrapped = stack.unwrap2d_snaphu(intf, corr).compute()
        """
        import dask
        import dask.array
        import xarray as xr
        from .Batch import Batch

        results = {}

        for key in phase.keys():
            phase_ds = phase[key]
            corr_ds = corr[key] if corr is not None else None

            result_vars = {}
            for var in phase_ds.data_vars:
                if var == 'spatial_ref':
                    continue

                phase_da = phase_ds[var]
                corr_da = corr_ds[var] if corr_ds is not None and var in corr_ds else None

                # Ensure data is chunked for lazy processing (1 chunk per pair)
                if 'pair' in phase_da.dims:
                    if not isinstance(phase_da.data, dask.array.Array):
                        phase_da = phase_da.chunk({'pair': 1})
                    if corr_da is not None and not isinstance(corr_da.data, dask.array.Array):
                        corr_da = corr_da.chunk({'pair': 1})

                # Save non-dimension coords along pair (ref, rep, BPR) for output
                pair_coords = {}
                if 'pair' in phase_da.dims:
                    for k, v in phase_da.coords.items():
                        if k != 'pair' and hasattr(v, 'dims') and v.dims == ('pair',):
                            vals = v.values if hasattr(v, 'values') else v
                            pair_coords[k] = ('pair', vals)

                # Create wrapper that captures defomax and debug
                def make_wrapper(defomax_val, debug_val):
                    def process_wrapper(phase_chunk, corr_chunk=None):
                        """Process single pair chunk: (1, y, x) -> (1, y, x)"""
                        phase_2d = phase_chunk[0]
                        corr_2d = corr_chunk[0] if corr_chunk is not None else None
                        unwrap_2d = Stack._snaphu_unwrap_array(
                            phase_2d, corr_2d, defomax=defomax_val, debug=debug_val
                        )
                        return unwrap_2d[np.newaxis, ...].astype(np.float32)
                    return process_wrapper

                wrapper = make_wrapper(defomax, debug)

                # Use da.blockwise for efficient dask integration
                phase_dask = phase_da.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(phase_dask.ndim))

                # Provide meta to avoid calling wrapper during graph construction
                meta = np.empty((0,) * phase_dask.ndim, dtype=np.float32)

                if corr_da is None:
                    result_dask = dask.array.blockwise(
                        wrapper, dim_str,
                        phase_dask, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )
                else:
                    corr_dask = corr_da.data
                    def wrapper_with_corr(phase_chunk, corr_chunk):
                        return wrapper(phase_chunk, corr_chunk)
                    result_dask = dask.array.blockwise(
                        wrapper_with_corr, dim_str,
                        phase_dask, dim_str,
                        corr_dask, dim_str,
                        dtype=np.float32,
                        meta=meta,
                    )

                unwrap_da = xr.DataArray(
                    result_dask,
                    dims=phase_da.dims,
                    coords=phase_da.coords
                )

                # Restore pair coords
                if pair_coords:
                    unwrap_da = unwrap_da.assign_coords(**pair_coords)

                result_vars[var] = unwrap_da

            result_ds = xr.Dataset(Stack._carry_meta(phase_ds, result_vars))
            result_ds.attrs = phase_ds.attrs
            results[key] = result_ds

        return Batch(results)

    import numpy as np

    import xarray as xr

    import pandas as pd

    def _get_pairs(self, pairs, dates=False):
        """
        Get pairs as DataFrame and optionally dates array.

        Parameters
        ----------
        pairs : np.ndarray, optional
            An array of pairs. If None, all pairs are considered. Default is None.
        dates : bool, optional
            Whether to return dates array. Default is False.
        name : str, optional
            The name of the phase filter. Default is 'phasefilt'.

        Returns
        -------
        pd.DataFrame or tuple
            A DataFrame of pairs. If dates is True, also returns an array of dates.
        """
        import xarray as xr
        import pandas as pd
        import numpy as np
        from glob import glob

        if isinstance(pairs, pd.DataFrame):
            # workaround for baseline_pairs() output
            pairs = pairs.rename(columns={'ref_date': 'ref', 'rep_date': 'rep'})
        elif isinstance(pairs, (xr.DataArray, xr.Dataset)):
            # pairs = pd.DataFrame({
#                 'ref': pairs.coords['ref'].values,
#                 'rep': pairs.coords['rep'].values
#             })
            refs = pairs.coords['ref'].values
            reps = pairs.coords['rep'].values
            pairs = pd.DataFrame({
                'ref': refs if isinstance(refs, np.ndarray) else [refs],
                'rep': reps if isinstance(reps, np.ndarray) else [reps]
            })
        else:
            # Convert numpy array to DataFrame
            # in case of 1d array with 2 items convert to a single pair
            pairs_2d = [pairs] if np.asarray(pairs).shape == (2,) else pairs
            pairs = pd.DataFrame(pairs_2d, columns=['ref', 'rep'])

        # Convert ref and rep columns to datetime format
        pairs['ref'] = pd.to_datetime(pairs['ref'])
        pairs['rep'] = pd.to_datetime(pairs['rep'])
        pairs['pair'] = [f'{ref} {rep}' for ref, rep in zip(pairs['ref'].dt.date, pairs['rep'].dt.date)]
        # Calculate the duration in days and add it as a new column
        #pairs['duration'] = (pairs['rep'] - pairs['ref']).dt.days

        if dates:
            # pairs is DataFrame
            dates = np.unique(pairs[['ref', 'rep']].astype(str).values.flatten())
            return (pairs, dates)
        return pairs

    def stl(self, data, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL) on Batch data.

        Decomposes time series into trend, seasonal, and residual components.
        The input Batch must have a 'date' dimension.

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
            Batch containing 'trend', 'seasonal', and 'resid' variables for each polarization.

        Examples
        --------
        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> displacement = model.displacement_los(stack.transform())
        >>> stl_result = stack.stl(displacement, freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, resid components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        import xarray as xr
        from .BatchCore import BatchCore

        # Validate input
        if not isinstance(data, dict):
            raise TypeError(f"data must be a Batch, got {type(data).__name__}")

        # Validate lazy data
        BatchCore._require_lazy(data, 'stl')

        sample_ds = next(iter(data.values()))
        if 'date' not in sample_ds.dims:
            raise ValueError("Input Batch must have 'date' dimension for STL decomposition")

        # Get polarizations from the first dataset (spatial, with y/x dims) - excludes converted attributes
        polarizations = [v for v in sample_ds.data_vars
                        if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]

        results = {}
        for key, ds in data.items():
            result_vars = {}
            for pol in polarizations:
                if pol not in ds.data_vars:
                    continue
                da = ds[pol]
                # Apply STL decomposition
                stl_ds = self._stl(da, freq=freq, periods=periods, robust=robust)
                # Rename variables to include polarization
                for var in ['trend', 'seasonal', 'resid']:
                    result_vars[f'{pol}_{var}'] = stl_ds[var]

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = ds.attrs
            # Preserve CRS if available
            if hasattr(ds, 'rio') and ds.rio.crs is not None:
                import rioxarray
                result_ds = result_ds.rio.write_crs(ds.rio.crs)
            results[key] = result_ds

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

        # Default chunk sizes if not set on Stack
        netcdf_chunksize = getattr(self, 'netcdf_chunksize', 512)
        chunksize1d = getattr(self, 'chunksize1d', 10000)

        if not isinstance(data, xr.DataArray):
            raise Exception('Invalid input: The "data" parameter should be of type xarray.DataArray.')

        dt, dt_periodic = utils_stl.stl_periodic(data.date, freq)
        n_dates_out = len(dt_periodic)
        n_dates_in = data.date.size

        data_dask = data.data

        def _stl_block(data_block, _dt=dt, _dt_periodic=dt_periodic,
                       _n_dates_out=n_dates_out, _periods=periods, _robust=robust):
            import math
            from .utils_dask import get_dask_chunk_size_mb
            ny, nx = data_block.shape[1], data_block.shape[2]
            n_dates_in_local = data_block.shape[0]
            result = np.empty((3, _n_dates_out, ny, nx), dtype=np.float32)
            vec_stl = np.vectorize(
                lambda ts: utils_stl.stl1d(ts, _dt, _dt_periodic, _periods, _robust),
                signature='(n)->(m),(m),(m)'
            )
            per_pixel_bytes = (n_dates_in_local + 3 * _n_dates_out) * 4
            budget_bytes = int(get_dask_chunk_size_mb() * 1024 * 1024)
            max_sub_pixels = max(256, budget_bytes // max(1, per_pixel_bytes))
            sub_side = int(math.sqrt(max_sub_pixels))
            sub_h = min(sub_side, ny)
            sub_w = min(sub_side, nx)
            for ty0 in range(0, ny, sub_h):
                ty1 = min(ty0 + sub_h, ny)
                for tx0 in range(0, nx, sub_w):
                    tx1 = min(tx0 + sub_w, nx)
                    tile = data_block[:, ty0:ty1, tx0:tx1]
                    tile_t = tile.transpose(1, 2, 0)
                    del tile
                    block = np.asarray(vec_stl(tile_t))
                    del tile_t
                    result[:, :, ty0:ty1, tx0:tx1] = block.transpose(0, 3, 1, 2)
                    del block
            del vec_stl
            return result

        models = dask.array.blockwise(
            _stl_block, 'cdyx',
            data_dask, 'pyx',
            new_axes={'c': 3, 'd': n_dates_out},
            concatenate=True,
            dtype=np.float32,
            meta=np.empty((0, 0, 0, 0), dtype=np.float32),
        )

        coords = {'date': dt_periodic.astype('datetime64[ns]'), 'y': data.y, 'x': data.x}

        # transform to separate variables
        varnames = ['trend', 'seasonal', 'resid']
        keys_vars = {}
        for varidx, varname in enumerate(varnames):
            var_data = models[varidx]
            keys_vars[varname] = xr.DataArray(var_data, coords=coords)
        model = xr.Dataset({**keys_vars})
        del models

        return model

    @staticmethod
    @serialize_gpu
    def _psfunction_torch(amplitudes, device='auto', debug=False):
        """
        Compute PS function using PyTorch for GPU acceleration.

        Parameters
        ----------
        amplitudes : np.ndarray
            3D array of shape (n_dates, height, width) containing amplitude values.
        device : str
            Device to use ('auto', 'mps', 'cuda', 'cpu').
        debug : bool
            Print debug information.

        Returns
        -------
        np.ndarray
            2D array of shape (height, width) containing PS function values.
        """
        import torch
        import numpy as np

        # Select device using shared helper
        dev = Stack._get_torch_device(device)

        if debug:
            print(f'DEBUG: _psfunction_torch using device={dev}')

        n_dates, height, width = amplitudes.shape

        # Convert to intensity (|z|^2) and move to device
        intensity = torch.from_numpy(amplitudes.astype(np.float32) ** 2).to(dev)

        # Compute mean intensity per date for normalization
        # Shape: (n_dates,)
        mean_intensity_per_date = intensity.nanmean(dim=(1, 2))

        # Global mean across all dates
        global_mean = mean_intensity_per_date.nanmean()

        # Normalize each date: intensity * (global_mean / date_mean)
        # Reshape for broadcasting: (n_dates, 1, 1)
        norm_factor = global_mean / mean_intensity_per_date
        norm_factor = norm_factor.view(n_dates, 1, 1)
        intensity_norm = intensity * norm_factor

        # Compute mean and std across date dimension
        # Use masked operations to handle NaN
        mean_amp = intensity_norm.nanmean(dim=0)  # (height, width)

        # PyTorch nanstd - compute manually since not built-in
        # std = sqrt(E[(x - mean)^2])
        diff = intensity_norm - mean_amp.unsqueeze(0)
        # Count non-NaN values
        valid_mask = ~torch.isnan(intensity_norm)
        n_valid = valid_mask.sum(dim=0).float()
        n_valid = torch.clamp(n_valid, min=1)  # Avoid division by zero

        # Variance with Bessel's correction (n-1)
        variance = torch.where(
            valid_mask, diff ** 2, torch.tensor(0.0, device=dev)
        ).sum(dim=0) / (n_valid - 1).clamp(min=1)
        std_amp = torch.sqrt(variance)

        # PS function: mean / (2 * std)
        psf = mean_amp / (2 * std_amp)

        # Handle invalid values (inf, nan)
        psf = torch.where(torch.isfinite(psf), psf, torch.tensor(float('nan'), device=dev))

        # Move back to CPU and convert to numpy
        result = psf.cpu().numpy()

        # Cleanup GPU memory
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    def psfunction(self, device='auto', allow_rechunk=True, debug=False):
        """
        Compute PS (Persistent Scatterer) function for weighting in single-look processing.

        The PS function identifies stable scatterers by computing the ratio of mean
        amplitude to amplitude standard deviation across the temporal stack:

            psfunction = mean_intensity / (2 * std_intensity)

        Higher values indicate more stable scatterers (consistent backscatter).

        Parameters
        ----------
        device : str
            Device for PyTorch computation ('auto', 'mps', 'cuda', 'cpu').
        allow_rechunk : bool
            If True (default), automatically rechunk input data to fit within Dask chunk size
            (dask.config['array.chunk-size']) using aligned chunk boundaries (//2, //4, etc.)
            to prevent dask slowdowns. Output is rechunked back to original spatial chunks.
        debug : bool
            Print debug information.

        Returns
        -------
        BatchUnit
            Per-burst PS function values for use as weight in phasediff_multilook() (lazy).

        Examples
        --------
        # Use PS function as weight for single-look interferograms
        psf = stack.psfunction()
        sintf, scorr = stack.phasediff_multilook(pairs, wavelength=30, weight=psf)

        # Control memory via Dask config
        import dask.config
        dask.config.set({'array.chunk-size': '256MiB'})
        psf = stack.psfunction()

        # Disable automatic rechunking (manual chunk control)
        psf = stack.chunk({'y': 2048, 'x': 2048}).psfunction(allow_rechunk=False)
        """
        import dask.array
        import numpy as np
        import torch
        import xarray as xr
        from .Batch import BatchUnit
        from .utils_dask import rechunk3d, restore_chunks, get_dask_chunk_size_mb

        # Auto-detect device based on Dask cluster resources and hardware
        # Convert to string once to avoid serialization issues and repeated resolution
        resolved = Stack._get_torch_device(device, debug=debug)
        device = resolved.type  # 'cpu', 'cuda', or 'mps' as string

        # Get Dask chunk size from config
        dask_chunk_mb = get_dask_chunk_size_mb()

        if debug:
            print(f"DEBUG: psfunction using device={device}, allow_rechunk={allow_rechunk}, dask.config['array.chunk-size']={dask_chunk_mb} MB")

        results = {}
        note_printed = False  # Only print NOTE once for first burst
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

            # Merge dates dim, keep input spatial chunks as-is.
            slc_data = slc_data.chunk({'date': -1})

            # Create wrapper that captures device and debug
            def make_wrapper(dev, dbg):
                def process_wrapper(slc_chunk):
                    """Process spatial chunk: (chunk_y, chunk_x, n_dates) -> (chunk_y, chunk_x)

                    Note: input_core_dims=[['date']] moves date to last axis.
                    """
                    # Transpose to (n_dates, chunk_y, chunk_x) for _psfunction_torch
                    slc_transposed = np.moveaxis(slc_chunk, -1, 0)
                    # Compute amplitude |z|
                    amplitudes = np.abs(slc_transposed)
                    # Compute PS function using PyTorch
                    psf_values = Stack._psfunction_torch(amplitudes, device=dev, debug=dbg)
                    return psf_values.astype(np.float32)
                return process_wrapper

            wrapper = make_wrapper(device, debug)

            # Use xr.apply_ufunc with dask='parallelized' for lazy execution
            # Core dim is 'date' (reduction), chunked dims are y, x
            # Note: input_core_dims moves 'date' to last axis, wrapper transposes back
            # Provide explicit meta to avoid ComplexWarning when dask infers
            # output type from complex input (we intentionally convert to real)
            psf_da = xr.apply_ufunc(
                wrapper,
                slc_data,
                input_core_dims=[['date']],
                output_core_dims=[[]],
                dask='parallelized',
                dask_gufunc_kwargs={'meta': np.array((), dtype=np.float32)},
            )

            # Assign name to match SLC variable
            psf_da.name = var_name

            results[key] = xr.Dataset({var_name: psf_da})

        return BatchUnit(results)

    @staticmethod
    def as_vtk(dataset):
        # Wrapper retained for backward compatibility
        return _as_vtk(dataset)

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
        colormap = matplotlib.cm.get_cmap(cmap, n)
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
        #print('Stack __init__', 0 if mapping is None else len(mapping))
        super().__init__(mapping)

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
        dtypes = set()
        has_temporal_dim = False
        for ds in subset.values():
            for var in ds.data_vars:
                dtypes.add(ds[var].dtype)
                # Check if variable has temporal dimension (date or pair)
                if 'date' in ds[var].dims or 'pair' in ds[var].dims:
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
        """Access by key or variable list."""
        # Handle variable selection like sbas[['ele']] or sbas[['VV', 'VH']]
        if isinstance(key, list):
            if len(key) == 1 and self:
                var = key[0]
                # Build a Batch containing only the requested variable for each scene
                subset = {k: ds[[var]] for k, ds in self.items() if var in ds.data_vars}
                if not subset:
                    raise KeyError(var)
            else:
                # Multiple variables
                subset = {k: ds[key] for k, ds in self.items()}
            # Return appropriate Batch type if spatial-only or complex, else Stack
            batch_type = self._batch_type_for_subset(subset)
            if batch_type is not None:
                return batch_type(subset)
            # Has temporal dimension but not complex - return Stack
            return type(self)(subset)
        return dict.__getitem__(self, key)

    def __getattr__(self, name: str):
        """
        Access variables (e.g., 'ele') from the Stack as Batch.

        This allows accessing variables stored in burst datasets:
            sbas.ele  -> BatchVar containing elevation data
        """
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self:
            sample = next(iter(self.values()), None)
            if sample is not None and hasattr(sample, 'data_vars'):
                if name in sample.data_vars or name in sample.coords:
                    subset = {k: ds[[name]] for k, ds in self.items() if name in ds.data_vars or name in ds.coords}
                    if subset:
                        batch_type = self._batch_type_for_subset(subset)
                        # For single attribute access, default to Batch if not BatchComplex
                        if batch_type is None:
                            batch_type = Batch
                        return batch_type(subset)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def transform(self) -> Batch:
        """Return a Batch view of this Stack (including 1D/2D non-complex vars).

        Also `northing` and `easting`, each burst's own y/x as a vector.

        `azi` and `rng` are the BURST'S grid and restart in each one, so a
        plane across bursts written in them is a sawtooth; the map coordinates
        share a lattice.
        """
        import numpy as np
        out = {}
        for key, ds in self.items():
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
        """Compute incidence angle for each burst via linear polynomial fit."""
        return self.transform().incidence()

    def _elevation_phase_approximate(self) -> dict:
        """`{burst_id: value}` at the burst centre. See Batch."""
        return Batch._elevation_phase_approximate(self)

    def optimize2(self, angle_coarse: float = 15, angle_fine: float = 5,
                  window: 'float | tuple | None' = 40, device: str = 'auto') -> "Stack":
        """
        Polarimetric optimization of amplitude and phase for dual-pol data.

        NOTE: Requires insardev_polsar extension.

        AMPLITUDE: the VV/VH combination minimizing ADI over the stack.
        PHASE:     the VV/VH combination maximizing coherence over the chain of
                   consecutive acquisitions (each date with the previous and next).
                   Needs no temporal parameter: the chain follows the real dates,
                   so gaps and uneven sampling are covered by construction.
                   window=None skips it and keeps the original co-pol phase,
                   the previous behaviour.

        Both mechanisms are one per pixel, shared across dates (Equal Scattering
        Mechanism), so every interferogram is a difference of per-date phases and
        triplet closure is exactly zero — unlike per-pair optimization
        (interferogram2), which does not close.

        Parameters
        ----------
        angle_coarse : float
            Coarse grid step in degrees. Default 15°.
        angle_fine : float
            Fine grid step in degrees. Default 5°.
        window : float or tuple or None
            Spatial window in METRES for the coherence estimate used by the
            phase search. None (default) skips it and keeps the original co-pol
            phase — unchanged behaviour. Pass e.g. (3, 12) to optimize the phase.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Stack
            Optimized Stack with VV phase and optimized VV+VH amplitude.
            Crosspol variable is dropped; all other variables preserved.

        Examples
        --------
        >>> import insardev_polsar
        >>> stack_opt = stack.optimize2()
        >>> adi = stack_opt.adi()
        >>> ps_mask = adi[['VV']] < 0.5
        >>> mintf, mcorr = stack_opt.pairs(baseline.tolist()).interferogram(wavelength=30)

        Notes
        -----
        Use optimize2() to feed the result into standard interferometric
        pipelines (phasediff, unwrap, etc.). Use stack.adi() after
        optimize2().compute() to get ADI values.
        """
        # Detect polarizations
        sample_ds = next(iter(self.values()))
        if 'VV' in sample_ds.data_vars and 'VH' in sample_ds.data_vars:
            pols = ['VV', 'VH']
        elif 'HH' in sample_ds.data_vars and 'HV' in sample_ds.data_vars:
            pols = ['HH', 'HV']
        else:
            raise ValueError("Dual-pol data required (VV+VH or HH+HV)")

        # Import implementation from insardev_polsar
        try:
            from insardev_polsar.adi2 import optimize2 as _optimize2_impl
        except ImportError:
            raise ImportError("optimize2() requires insardev_polsar extension. Install it first.")

        # Get dual-pol subset as BatchComplex
        batch_complex = self[pols]

        # Call internal optimize2 implementation
        s_opt_batch = _optimize2_impl(batch_complex, angle_coarse=angle_coarse,
                                      angle_fine=angle_fine, window=window,
                                      device=device)

        # Merge S_opt back into original stack structure (preserves BPR, etc.)
        output_pol = pols[0]  # VV or HH
        s_opt_dict = {}
        for burst_id, orig_ds in self.items():
            s_opt_ds = s_opt_batch[burst_id]
            # Drop original pols, add optimized output
            merged = orig_ds.drop_vars(pols).assign({output_pol: s_opt_ds[output_pol]})
            s_opt_dict[burst_id] = merged

        return type(self)(s_opt_dict)

    def adi2(self,
             angle_coarse: float = 15,
             angle_fine: float = 5,
             device: str = 'auto') -> Batch:
        """
        Dual-pol ADI: optimize2() + adi() in one call.

        NOTE: Requires insardev_polsar extension.

        Finds optimal VV/VH amplitude combination that minimizes ADI,
        then computes ADI on the optimized amplitudes.

        Parameters
        ----------
        angle_coarse : float
            Coarse grid step in degrees. Default 15.
        angle_fine : float
            Fine grid step in degrees. Default 5.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batch
            ADI values computed on polarimetrically optimized amplitudes.

        Examples
        --------
        >>> import insardev_polsar
        >>> adi = stack.adi2()
        >>> ps_mask = adi[['VV']] < 0.4
        """
        sample_ds = next(iter(self.values()))
        if 'VV' in sample_ds.data_vars and 'VH' in sample_ds.data_vars:
            pols = ['VV', 'VH']
        elif 'HH' in sample_ds.data_vars and 'HV' in sample_ds.data_vars:
            pols = ['HH', 'HV']
        else:
            raise ValueError("Dual-pol data required (VV+VH or HH+HV)")

        batch_complex = self[pols]
        return batch_complex.adi2(angle_coarse, angle_fine, device)

    def adi(self, device: str = 'auto') -> Batch:
        """
        Compute Amplitude Dispersion Index (ADI) for calibrated σ₀ data.

        Wrapper that calls BatchComplex.adi() on complex variables.

        Parameters
        ----------
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batch
            ADI values for each polarization variable present.

        Examples
        --------
        >>> adi = stack.adi()
        >>> ps_mask = adi < 0.25
        """
        # Get complex variables
        sample_ds = next(iter(self.values()))
        complex_vars = [v for v in sample_ds.data_vars
                        if sample_ds[v].dtype.kind == 'c' and 'date' in sample_ds[v].dims]

        if not complex_vars:
            raise ValueError("No complex time-series data found")

        # Get as BatchComplex and call adi()
        batch_complex = self[complex_vars]
        return batch_complex.adi(device)

    def neighbors(
        self,
        window: 'float | tuple' = 40,
        neighbors: tuple | None = None,
        valid_threshold: float = 0.5,
        device: str = 'auto'
    ) -> Batch:
        """
        Count valid neighbors per pixel within a spatial window.

        NOTE: Requires insardev_polsar extension.

        Wrapper that calls BatchComplex.neighbors() on complex variables.
        Useful for estimating pixel density before running similarity().

        Parameters
        ----------
        window : float or tuple of float
            Window size in METRES on the ground, one number for a square
            or (y, x). Rounded up to an odd number of pixels per axis.
        neighbors : tuple of int or None
            If provided, filter output: (min, max)
            - Pixels with count < min: set to NaN
            - Pixels with count > max: clipped to max
        valid_threshold : float
            Minimum fraction of dates with valid data.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'

        Returns
        -------
        Batch
            Valid neighbor count per pixel

        Examples
        --------
        >>> counts = stack.neighbors(window=120)
        >>> dense_mask = counts >= 10
        """
        # Get complex variables
        sample_ds = next(iter(self.values()))
        complex_vars = [v for v in sample_ds.data_vars
                        if sample_ds[v].dtype.kind == 'c' and 'date' in sample_ds[v].dims]

        if not complex_vars:
            raise ValueError("No complex time-series data found")

        # Get as BatchComplex and call neighbors()
        batch_complex = self[complex_vars]
        return batch_complex.neighbors(window, neighbors, valid_threshold, device)

    def to_vtk(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK.

        Merges bursts using to_dataset() and exports one VTK file per data variable
        (e.g., VV.vtk). Within each file, pairs become separate VTK arrays named
        by date (e.g., 20190708_20190702).

        Parameters
        ----------
        path : str
            Output directory/filename for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Use ``data=None`` with
            ``overlay`` to export just topography with image overlay.
        transform : Batch | None, optional
            Optional transform Batch providing topography (``ele`` or ``z``).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.

        Examples
        --------
        # Export data with overlay
        stack.to_vtk('velocity', velocity, overlay=gmap)

        # Export just topography with image overlay (like PyGMTSAR export_vtk)
        stack.to_vtk('gmap', data=None, overlay=gmap)
        """
        import os
        import numpy as np
        import xarray as xr
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        # Handle data=None case (export just overlay on topography)
        if data is None and overlay is not None:
            tfm = transform if transform is not None else self.transform()
            if tfm is not None and not isinstance(tfm, BatchCore):
                tfm = Batch(tfm)

            if tfm is None:
                raise ValueError("transform is required when data=None")

            # Get topography at native resolution
            topo_merged = tfm[['ele']].to_dataset()
            topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
            if topo_da is None:
                raise ValueError("transform must contain 'ele' variable")

            # Interpolate elevation to overlay grid (preserves image quality)
            ov = overlay
            if 'band' not in ov.dims:
                ov = ov.expand_dims('band')
            # Interpolate elevation to match overlay coordinates
            topo_da = topo_da.interp(y=ov.y, x=ov.x, method='linear')

            if mask:
                # Mask by finite values in overlay
                topo_da = topo_da.where(np.isfinite(ov.isel(band=0)))

            # Build output dataset
            layers = [topo_da.rename('z'), ov.rename('colors')]
            ds_out = xr.merge(layers, compat='override', join='left')
            vtk_grid = as_vtk(ds_out)

            # Determine output filename
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

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        # Default to self.transform() when called on Stack and transform not provided
        tfm_is_default = transform is None
        tfm = transform if transform is not None else self.transform()
        if tfm is not None and not isinstance(tfm, BatchCore):
            tfm = Batch(tfm)

        if not target:
            return

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(da, idx):
            """Format pair label from ref/rep coordinates at given index."""
            if 'ref' in da.coords and 'rep' in da.coords:
                ref_val = da.coords['ref'].values[idx]
                rep_val = da.coords['rep'].values[idx]
                return f"{_format_dt(ref_val)}_{_format_dt(rep_val)}"
            return str(idx)

        os.makedirs(path, exist_ok=True)

        # Merge bursts into unified dataset(s) per variable
        merged = target.to_dataset()
        if isinstance(merged, xr.DataArray):
            merged = merged.to_dataset()

        # Get transform elevation merged via to_dataset()
        # Decimate default transform to match input batch resolution for efficiency
        topo_merged = None
        if tfm is not None:
            if tfm_is_default:
                # Decimate each burst's transform to match corresponding input burst
                # Use index-based nearest neighbor selection (much faster than reindex)
                def _nearest_indices(source_coords, target_coords):
                    """Find indices in source_coords nearest to target_coords."""
                    # Handle descending coordinates (e.g., y going north to south)
                    descending = len(source_coords) > 1 and source_coords[0] > source_coords[-1]
                    if descending:
                        source_coords = source_coords[::-1]
                    indices = np.searchsorted(source_coords, target_coords)
                    indices = np.clip(indices, 0, len(source_coords) - 1)
                    # Check if previous index is closer
                    prev_indices = np.clip(indices - 1, 0, len(source_coords) - 1)
                    prev_diff = np.abs(source_coords[prev_indices] - target_coords)
                    curr_diff = np.abs(source_coords[indices] - target_coords)
                    indices = np.where(prev_diff < curr_diff, prev_indices, indices)
                    # Convert back to original order if descending
                    if descending:
                        indices = len(source_coords) - 1 - indices
                    return indices

                decimated = {}
                for k in target.keys():
                    if k not in tfm:
                        continue
                    tfm_ds = tfm[k][['ele']]
                    tgt_ds = target[k]
                    # Find nearest indices for y and x coordinates
                    y_idx = _nearest_indices(tfm_ds.y.values, tgt_ds.y.values)
                    x_idx = _nearest_indices(tfm_ds.x.values, tgt_ds.x.values)
                    # Select using indices and assign target coordinates
                    selected = tfm_ds.isel(y=y_idx, x=x_idx)
                    selected = selected.assign_coords(y=tgt_ds.y, x=tgt_ds.x)
                    decimated[k] = selected
                topo_merged = Batch(decimated).to_dataset()
            else:
                # User-provided transform: use as-is
                topo_merged = tfm[['ele']].to_dataset()

        # Group by data variable (polarization)
        data_vars = list(merged.data_vars)

        with tqdm(total=len(data_vars), desc='Exporting VTK') as pbar:
            for data_var in data_vars:
                da = merged[data_var]

                # Handle pair dimension
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

                # Determine target grid - use overlay grid if provided (preserves image quality)
                ov = None
                if overlay is not None:
                    if not isinstance(overlay, xr.DataArray):
                        raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")
                    ov = overlay
                    if 'band' not in ov.dims:
                        ov = ov.expand_dims('band')
                    # Select overlay region matching data extent
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

                # Use overlay grid or data grid as target
                target_y = ov.y if ov is not None else ref_da.y
                target_x = ov.x if ov is not None else ref_da.x

                # Add topography from transform
                if topo_merged is not None:
                    topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
                    if topo_da is not None:
                        topo_da = topo_da.interp(y=target_y, x=target_x, method='linear')
                        if mask:
                            ref_for_mask = ref_da.interp(y=target_y, x=target_x, method='nearest') if ov is not None else ref_da
                            topo_da = topo_da.where(np.isfinite(ref_for_mask))
                        layers.append(topo_da.rename('z'))

                # Add overlay at native resolution
                if ov is not None:
                    layers.append(ov.rename('colors'))

                # Add data arrays - interpolate to target grid
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
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        tfm = transform
        if tfm is not None and not isinstance(tfm, BatchCore):
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

        with tqdm(total=len(target), desc='Exporting VTK') as pbar:
            for burst, ds in target.items():
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
        >>> stack_opt = stack.optimize2().compute()  # compute Stack itself
        """
        if not batches:
            return BatchCore.compute(self)
        from .Batch import Batches
        return Batches(batches).compute()

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = None, debug=False):
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
        from .Batch import Batches
        from . import utils_io

        # Handle case where first arg is store path
        if len(args) == 1 and isinstance(args[0], str):
            store = args[0]
            args = ()

        # If no batch args provided — save Stack (if non-empty) or open existing
        if len(args) == 0:
            if store is None:
                raise ValueError("store path is required to open snapshot")
            if len(self) > 0:
                # Non-empty Stack: save itself, then reopen
                utils_io.save(self, store=store, storage_options=storage_options,
                             caption=caption or 'Snapshotting...', debug=debug)
                return utils_io.open(store=store, storage_options=storage_options,
                                     n_jobs=-1, debug=debug)
            return utils_io.open(store=store, storage_options=storage_options,
                                n_jobs=-1, debug=debug)

        # Save mode - args are batches
        return Batches(args).snapshot(store=store, storage_options=storage_options, caption=caption, debug=debug)

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
        from .Batch import Batches
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
                utils_io.save(self, store=temp_dir, storage_options=None,
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

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in datas.values():
            #print (data.id)
            attrs = [data_var for data_var in ds if ds[data_var].dims==('date',)][::-1]
            attr_start_idx = attrs.index(attr_start)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs[:attr_start_idx+1]:
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
        df.insert(3, "polarization", pol)
        # round for human readability
        df['BPR'] = df['BPR'].round(1)

        group_col = df.columns[0]
        burst_col = df.columns[1]
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
        scale_factor = np.float32(attrs.get('scale_factor', 1.0))
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
        scale_factor = np.float32(attrs.get('scale_factor', 1.0))
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
        scale = np.float32(meta['attributes'].get('scale_factor', 1.0))
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
        scale = np.float32(meta['attributes'].get('scale_factor', 1.0))
        fill_value = meta['attributes'].get('_FillValue')

        return shape, chunks, scale, fill_value, dtype

    def load(self, urls:str | list | dict[str, str], storage_options:dict[str, str]|None=None,
             debug:bool=False):
        import numpy as np
        import dask
        import dask.array as da
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        import zarr
        from shapely import wkt
        import os
        from insardev_toolkit import progressbar_joblib
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

        # Whitelist of scalar attrs actually used by insardev processing
        # Alignment params (sub_int_*, stretch_*, a_stretch_*, ashift, rshift) are
        # excluded - they're only needed for debugging alignment quality in insardev_pygmtsar
        _USED_SCALAR_ATTRS = {
            # Mission/acquisition metadata
            'startTime', 'polarization', 'burst', 'flightDirection',
            'pathNumber', 'subswath', 'mission', 'beamModeType',
            'fullBurstID', 'geometry',  # Used in to_dataframe()
            # Radar parameters for incidence, elevation, LOS calculations
            'radar_wavelength', 'near_range',
            'SC_height_start', 'SC_height_end', 'earth_radius',
            'rng_samp_rate', 'num_lines',
            'num_rng_bins',  # insardev_ecef: bilinear interpolation of incidence/elevation corners
            # Baseline
            'BPR', 'BPT', 'B_perpendicular', 'B_parallel',
            # Reference height for elevation computation
            'ref_height',
        }

        # Resolve dask chunk budget in main process (joblib loky workers
        # don't inherit dask.config, so get_dask_chunk_size_mb() would
        # return the default 128 MB regardless of user config).
        from .utils_dask import get_dask_chunk_size_mb
        _load_target_mb = get_dask_chunk_size_mb()

        def store_open_group_delayed(zarr_path, group):
            """
            Open a fullBurstID group using delayed loading for complex data.

            This avoids the complex xarray concat graph by creating simple
            dask.delayed tasks for each date's data. File handles are opened
            and closed cleanly for each load operation.
            """
            import rioxarray
            root = zarr.open_consolidated(zarr_path, zarr_format=3, mode='r')
            grp = root[group]

            # Get burst subgroups (excluding transform)
            burst_keys = [k for k in grp.keys() if k != 'transform']

            # Collect metadata for all bursts (no 2D data loading)
            burst_infos = []
            spatial_ref = None
            for burst_key in burst_keys:
                burst_grp = grp[burst_key]
                burst_path = f"{group}/{burst_key}"

                # Open to get metadata only
                ds = xr.open_zarr(burst_grp.store, group=burst_grp.path,
                                  consolidated=True, zarr_format=3)

                shape = burst_grp['re'].shape
                date = np.datetime64(ds.attrs['startTime'], 's')
                polarization = ds.attrs['polarization']
                burst_name = ds.attrs['burst']

                # Capture spatial_ref from reference burst (BPR=0)
                if spatial_ref is None and ds.attrs.get('BPR', 1) == 0:
                    spatial_ref = ds.attrs.get('spatial_ref')

                # Extract scalar attrs (only whitelisted keys used by insardev)
                scalar_attrs = {}
                array_attrs = {}
                skip_attrs = {'Conventions', 'spatial_ref'}
                for k, v in ds.attrs.items():
                    if k in skip_attrs:
                        continue
                    if isinstance(v, (list, tuple)):
                        array_attrs[k] = np.array(v)
                    elif k in _USED_SCALAR_ATTRS:
                        # Only include whitelisted scalar attrs
                        if isinstance(v, str):
                            scalar_attrs[k] = v
                        else:
                            scalar_attrs[k] = float(v) if isinstance(v, (int, float)) else v

                # Store only what we need (not the full dataset!)
                burst_infos.append({
                    'polarization': polarization,
                    'burst_path': burst_path,
                    'burst_name': burst_name.replace(polarization, 'XX'),
                    'shape': shape,
                    'date': date,
                    'y': ds.y.values,
                    'x': ds.x.values,
                    'scalar_attrs': scalar_attrs,
                    'array_attrs': array_attrs,
                })
                ds.close()  # Close xarray dataset after extracting metadata

            # Group by polarization and sort by date
            polarizations = np.unique([info['polarization'] for info in burst_infos])

            datas = []
            for polarization in polarizations:
                pol_infos = sorted(
                    [info for info in burst_infos if info['polarization'] == polarization],
                    key=lambda x: x['date']
                )

                # Create delayed dask arrays for each date
                delayed_arrays = []
                for info in pol_infos:
                    shape = info['shape']
                    # Get chunk metadata
                    _, zarr_chunks, scale, fill_value, re_dtype = Stack._get_zarr_slc_meta(
                        zarr_path, info['burst_path'], storage_options
                    )
                    n_chunks_y = (shape[0] + zarr_chunks[0] - 1) // zarr_chunks[0]
                    n_chunks_x = (shape[1] + zarr_chunks[1] - 1) // zarr_chunks[1]

                    if n_chunks_y == 1 and n_chunks_x == 1:
                        # Single chunk (S1): one reader for entire array
                        delayed_load = dask.delayed(Stack._load_zarr_complex)(
                            zarr_path, info['burst_path'], storage_options
                        )
                        arr = da.from_delayed(delayed_load, shape=shape, dtype=np.complex64)
                    else:
                        # Multi-chunk (NISAR): one reader call per chunk
                        chunk_rows = []
                        for iy in range(n_chunks_y):
                            chunk_cols = []
                            for ix in range(n_chunks_x):
                                # Logical chunk shape (may be smaller at edges)
                                y0, y1 = iy * zarr_chunks[0], min((iy + 1) * zarr_chunks[0], shape[0])
                                x0, x1 = ix * zarr_chunks[1], min((ix + 1) * zarr_chunks[1], shape[1])
                                chunk_shape = (y1 - y0, x1 - x0)

                                delayed_chunk = dask.delayed(Stack._load_zarr_complex_chunk)(
                                    zarr_path, info['burst_path'], (iy, ix),
                                    chunk_shape, zarr_chunks, scale, fill_value, storage_options,
                                    re_dtype=re_dtype
                                )
                                chunk_arr = da.from_delayed(delayed_chunk, shape=chunk_shape, dtype=np.complex64)
                                chunk_cols.append(chunk_arr)
                            chunk_rows.append(chunk_cols)
                        arr = da.block(chunk_rows)

                    arr = arr[np.newaxis, :, :]  # Add date dim: (y, x) -> (1, y, x)
                    delayed_arrays.append(arr)

                # Stack all dates: (n_dates, y, x)
                stacked = da.concatenate(delayed_arrays, axis=0)

                # Zarr disk chunks are rechunked to dask budget via chunk2d logic below.

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

                # Add all scalar attrs as variables (replicated per date)
                # Preserve original order from first burst (to_dataframe expects specific order)
                # Exclude 'burst' (handled above with XX replacement) and 'polarization' (per-pol)
                excluded_keys = {'burst', 'polarization'}
                all_scalar_keys = [k for k in pol_infos[0]['scalar_attrs'].keys() if k not in excluded_keys]
                # Add any keys from other dates that might be missing
                for info in pol_infos[1:]:
                    for k in info['scalar_attrs'].keys():
                        if k not in all_scalar_keys and k not in excluded_keys:
                            all_scalar_keys.append(k)
                for key in all_scalar_keys:
                    vals = [info['scalar_attrs'].get(key, np.nan) for info in pol_infos]
                    # Check if all values are numeric
                    if all(isinstance(v, (int, float, np.number)) for v in vals):
                        data_ds[key] = xr.DataArray(np.array(vals, dtype=np.float64), dims=['date'])
                    else:
                        # String values
                        data_ds[key] = xr.DataArray(vals, dims=['date'])

                # Add array attrs (e.g., polynomial coefficients) - take from first burst
                first_info = pol_infos[0]
                for key, arr in first_info['array_attrs'].items():
                    if arr.ndim == 1:
                        # Stack arrays from all dates: (n_dates, n_coef)
                        stacked = np.stack([info['array_attrs'].get(key, arr) for info in pol_infos])
                        data_ds[key] = xr.DataArray(stacked, dims=['date', f'{key}_coef'])

                datas.append(data_ds)

            # Merge polarizations
            ds = xr.merge(datas, compat='no_conflicts', combine_attrs='override')
            del datas

            # Load transform: zarr handles metadata/coords, custom reader for 2D chunks
            grp_transform = grp['transform']
            transform = xr.open_zarr(grp_transform.store, group=grp_transform.path,
                                     consolidated=True, zarr_format=3)

            # Coords eagerly (small 1D arrays)
            ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)
            # the coordinates keep their attributes, actual_range among them
            ds.x.attrs.update(transform.x.attrs)
            ds.y.attrs.update(transform.y.attrs)

            # 2D vars as lazy dask arrays via custom reader (no persistent file descriptors)
            # One reader call per chunk for memory efficiency
            # Transform 2D vars loaded with zarr disk chunks, rechunked via chunk2d logic below.
            transform_path = f"{group}/transform"
            for var in transform.data_vars:
                shape, zarr_chunks, scale_factor, fill_value, dtype = Stack._get_zarr_array_meta(
                    zarr_path, transform_path, var, storage_options
                )
                n_chunks_y = (shape[0] + zarr_chunks[0] - 1) // zarr_chunks[0]
                n_chunks_x = (shape[1] + zarr_chunks[1] - 1) // zarr_chunks[1]

                if n_chunks_y == 1 and n_chunks_x == 1:
                    # Single chunk: one reader for entire array
                    delayed_load = dask.delayed(Stack._load_zarr_array)(
                        zarr_path, transform_path, var, storage_options
                    )
                    arr = da.from_delayed(delayed_load, shape=shape, dtype=np.float32)
                else:
                    # Multi-chunk: one reader per chunk
                    chunk_rows = []
                    for iy in range(n_chunks_y):
                        chunk_cols = []
                        for ix in range(n_chunks_x):
                            y0, y1 = iy * zarr_chunks[0], min((iy + 1) * zarr_chunks[0], shape[0])
                            x0, x1 = ix * zarr_chunks[1], min((ix + 1) * zarr_chunks[1], shape[1])
                            chunk_shape = (y1 - y0, x1 - x0)

                            delayed_chunk = dask.delayed(Stack._load_zarr_array_chunk)(
                                zarr_path, transform_path, var, (iy, ix),
                                chunk_shape, zarr_chunks, scale_factor, fill_value, dtype, storage_options
                            )
                            chunk_arr = da.from_delayed(delayed_chunk, shape=chunk_shape, dtype=np.float32)
                            chunk_cols.append(chunk_arr)
                        chunk_rows.append(chunk_cols)
                    arr = da.block(chunk_rows)

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

            # Close zarr resources (metadata only, no lazy data references)
            transform.close()
            root.store.close()

            # Apply chunk2d logic: rechunk spatial dims to optimal sizes for budget
            from .utils_dask import rechunk2d
            sample = None
            for var in ds.data_vars:
                arr = ds[var]
                if arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x'):
                    sample = arr
                    break
            if sample is not None:
                y_size, x_size = sample.shape[-2], sample.shape[-1]
                in_chunks = (sample.data.chunks[-2], sample.data.chunks[-1]) if hasattr(sample.data, 'chunks') else None
                optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                   input_chunks=in_chunks, merge=False,
                                   target_mb=_load_target_mb)
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

            return group, ds

        if isinstance(urls, str):
            # note: isinstance(urls, zarr.storage.ZipStore) can be loaded too but it is less efficient
            urls = os.path.expanduser(urls)
            zarr_path = urls  # Store for delayed loading
            root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
            groups = list(root.group_keys())
            del root  # Close the root - we'll reopen in each group loader

            # Use the new delayed loading approach
            with progressbar_joblib.progressbar_joblib(tqdm(desc='Loading Dataset...'.ljust(25), total=len(groups))) as progress_bar:
                dss = joblib.Parallel(n_jobs=-1, backend='loky')\
                    (joblib.delayed(store_open_group_delayed)(zarr_path, group) for group in groups)
            # list of key - dataset converted to dict and appended to the existing dict
            self.update(dss)
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

            dss = {}
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets...'.ljust(25)):
                df = urls[urls.index.get_level_values(0) == fullBurstID]
                burst_urls = df[df.index.get_level_values(1) != 'transform'].iloc[:,0].values
                transform_url = df[df.index.get_level_values(1) == 'transform'].iloc[:,0].values[0]

                # Read burst metadata eagerly from each URL (attrs, shape, coords)
                burst_infos = []
                spatial_ref = None
                for burst_url in burst_urls:
                    bds = xr.open_zarr(burst_url, consolidated=True, zarr_format=3,
                                       storage_options=storage_options)
                    shape = (bds.dims['y'], bds.dims['x'])
                    date = np.datetime64(bds.attrs['startTime'], 's')
                    polarization = bds.attrs['polarization']
                    burst_name = bds.attrs['burst']

                    if spatial_ref is None and bds.attrs.get('BPR', 1) == 0:
                        spatial_ref = bds.attrs.get('spatial_ref')

                    # Extract scalar attrs (only whitelisted keys used by insardev)
                    scalar_attrs = {}
                    array_attrs = {}
                    for k, v in bds.attrs.items():
                        if k in {'Conventions', 'spatial_ref'}:
                            continue
                        if isinstance(v, (list, tuple)):
                            array_attrs[k] = np.array(v)
                        elif k in _USED_SCALAR_ATTRS:
                            # Only include whitelisted scalar attrs
                            if isinstance(v, str):
                                scalar_attrs[k] = v
                            else:
                                scalar_attrs[k] = float(v) if isinstance(v, (int, float)) else v

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

                # Build dataset same as primary path: delayed complex arrays
                polarizations = np.unique([info['polarization'] for info in burst_infos])
                datas = []
                for polarization in polarizations:
                    pol_infos = sorted(
                        [info for info in burst_infos if info['polarization'] == polarization],
                        key=lambda x: x['date']
                    )

                    delayed_arrays = []
                    for info in pol_infos:
                        delayed_load = dask.delayed(Stack._load_zarr_complex)(
                            info['url'], '', storage_options
                        )
                        arr = da.from_delayed(delayed_load, shape=info['shape'], dtype=np.complex64)
                        arr = arr[np.newaxis, :, :]
                        delayed_arrays.append(arr)

                    stacked = da.concatenate(delayed_arrays, axis=0)
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

                    # Exclude 'burst' (handled above with XX replacement) and 'polarization' (per-pol)
                    excluded_keys = {'burst', 'polarization'}
                    all_scalar_keys = [k for k in pol_infos[0]['scalar_attrs'].keys() if k not in excluded_keys]
                    for info in pol_infos[1:]:
                        for k in info['scalar_attrs'].keys():
                            if k not in all_scalar_keys and k not in excluded_keys:
                                all_scalar_keys.append(k)
                    for key in all_scalar_keys:
                        vals = [info['scalar_attrs'].get(key, np.nan) for info in pol_infos]
                        if all(isinstance(v, (int, float, np.number)) for v in vals):
                            data_ds[key] = xr.DataArray(np.array(vals, dtype=np.float64), dims=['date'])
                        else:
                            data_ds[key] = xr.DataArray(vals, dims=['date'])

                    first_info = pol_infos[0]
                    for key, arr in first_info['array_attrs'].items():
                        if arr.ndim == 1:
                            stacked_arr = np.stack([info['array_attrs'].get(key, arr) for info in pol_infos])
                            data_ds[key] = xr.DataArray(stacked_arr, dims=['date', f'{key}_coef'])

                    datas.append(data_ds)

                # Merge polarizations
                ds = xr.merge(datas, compat='no_conflicts', combine_attrs='override')
                del datas

                # Load transform: zarr for metadata/coords, custom reader for 2D
                transform = xr.open_zarr(transform_url, consolidated=True, zarr_format=3,
                                         storage_options=storage_options)
                ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)
                ds.x.attrs.update(transform.x.attrs)
                ds.y.attrs.update(transform.y.attrs)
                for var in transform.data_vars:
                    shape = transform[var].shape
                    delayed_load = dask.delayed(Stack._load_zarr_array)(
                        transform_url, '', var, storage_options
                    )
                    ds[var] = xr.DataArray(
                        da.from_delayed(delayed_load, shape=shape, dtype=np.float32),
                        dims=['y', 'x'], attrs=dict(transform[var].attrs)
                    )

                if spatial_ref is None:
                    spatial_ref = transform.attrs.get('spatial_ref')
                if spatial_ref is None:
                    raise KeyError('spatial_ref')
                ds.attrs['spatial_ref'] = spatial_ref
                ds.rio.write_crs(spatial_ref, inplace=True)
                transform.close()

                # Apply chunk2d logic: rechunk spatial dims to optimal sizes for budget
                from .utils_dask import rechunk2d
                sample = None
                for var in ds.data_vars:
                    arr = ds[var]
                    if arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x'):
                        sample = arr
                        break
                if sample is not None:
                    y_size, x_size = sample.shape[-2], sample.shape[-1]
                    in_chunks = (sample.data.chunks[-2], sample.data.chunks[-1]) if hasattr(sample.data, 'chunks') else None
                    optimal = rechunk2d((y_size, x_size), element_bytes=8,
                                       input_chunks=in_chunks, merge=True)
                    rechunked_vars = {}
                    for var_name in ds.data_vars:
                        arr = ds[var_name]
                        if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                            continue
                        if arr.ndim == 3:
                            var_chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                        else:
                            var_chunks = {'y': optimal['y'], 'x': optimal['x']}
                        rechunked_vars[var_name] = arr.chunk(var_chunks)
                    if rechunked_vars:
                        ds = ds.assign(rechunked_vars)

                dss[fullBurstID] = ds

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            self.update(dss)

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
        Align burst phases using interferometric double differences (ESD).

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
                Phase-corrected Stack.
            If return_residuals is True:
                (corrected_stack, residuals) where residuals is list[float].

        Examples
        --------
        >>> # Align burst phases before interferogram formation
        >>> stack_aligned = stack.align()
        >>> phase, corr = stack_aligned.pairs(baseline).interferogram(wavelength=30)
        """
        import numpy as np
        import xarray as xr
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        MIN_OVERLAP_PIXELS = 50

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

        if n_dates < 2:
            if debug:
                print('align(): need at least 2 dates for double-difference', flush=True)
            return (self, [0.0]) if return_residuals else self

        if debug:
            print(f'align(): {n_bursts} bursts, {n_dates} dates, ref=date[{ref_idx}], pol={polarization}', flush=True)

        # Collect burst extents
        extents = {}
        for bid in ids:
            da = self[bid][polarization]
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            return not (y1_max < y2_min or y2_max < y1_min) and not (x1_max < x2_min or x2_max < x1_min)

        # Find overlapping burst pairs
        overlap_pairs = []
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids[i+1:], i+1):
                if extents_overlap(extents[id1], extents[id2]):
                    overlap_pairs.append((id1, id2))

        if not overlap_pairs:
            if debug:
                print('No overlapping bursts found', flush=True)
            return (self, [0.0] * n_dates) if return_residuals else self

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
            overlap_slices[(id1, id2)] = (
                slice(max(y1.min(), y2.min()), min(y1.max(), y2.max())),
                slice(max(x1.min(), x2.min()), min(x1.max(), x2.max())),
            )

        n_total_dds = len(rep_dates) * len(overlap_pairs)
        if debug:
            print(f'Computing {n_total_dds} double differences...', flush=True)

        # Build lazy dask graphs that reduce each overlap DD to two scalars (complex sum
        # and valid count). dask.compute() schedules them in parallel across workers;
        # each worker materializes only one overlap region at a time, and only the
        # scalar results are returned — full DD arrays are never collected to the client.
        import dask.array as da_module
        dd_keys = []    # (d_rep, k) for result indexing
        dd_sums = []    # lazy scalar: nansum of DD complex values
        dd_counts = []  # lazy scalar: count of finite pixels
        for d_rep in rep_dates:
            for k, (id1, id2) in enumerate(overlap_pairs):
                y_sl, x_sl = overlap_slices[(id1, id2)]
                da1_ov = self[id1][polarization].sel(y=y_sl, x=x_sl)
                da2_ov = self[id2][polarization].sel(y=y_sl, x=x_sl)
                intf1 = da1_ov.isel(date=ref_idx) * da1_ov.isel(date=d_rep).conj()
                intf2 = da2_ov.isel(date=ref_idx) * da2_ov.isel(date=d_rep).conj()
                dd = intf1 * intf2.conj()  # lazy (y, x) overlap array
                # Reduce to two scalars within the dask graph
                dd_dask = dd.data  # underlying dask array
                dd_sums.append(da_module.nansum(dd_dask))
                dd_counts.append(da_module.sum(da_module.isfinite(dd_dask)))
                dd_keys.append((d_rep, k))

        # Single dask.compute() call — parallel across workers, returns only scalars
        all_scalars = dask.compute(*dd_sums, *dd_counts)
        n = len(dd_keys)
        sums = all_scalars[:n]
        counts = all_scalars[n:]

        dd_stats = {}
        for i, key in enumerate(dd_keys):
            cnt = int(counts[i])
            if cnt >= MIN_OVERLAP_PIXELS:
                dd_stats[key] = (float(np.angle(sums[i] / cnt)), cnt)
            else:
                dd_stats[key] = None

        # Solve per-date global least-squares
        corrections = np.zeros((n_bursts, n_dates))

        for d_rep in rep_dates:
            rows_data = []
            for k, (id1, id2) in enumerate(overlap_pairs):
                stat = dd_stats[(d_rep, k)]
                if stat is None:
                    continue

                dd_phase, cnt = stat
                weight = np.sqrt(float(cnt))
                i, j = id_to_idx[id1], id_to_idx[id2]
                # dd_phase = jump_ref - jump_rep ≈ -jump_rep
                # correction_i - correction_j = -dd_phase
                rows_data.append((i, j, -dd_phase, weight))

                if debug:
                    print(f'  dd {id1}-{id2} date[{d_rep}]: phase={dd_phase:.4f} rad, '
                          f'count={cnt}, weight={weight:.0f}', flush=True)

            if not rows_data:
                continue

            n_edges = len(rows_data)
            A = sparse.lil_matrix((n_edges, n_bursts))
            b = np.zeros(n_edges)
            W = np.zeros(n_edges)

            for r, (i, j, offset, weight) in enumerate(rows_data):
                A[r, i] = 1
                A[r, j] = -1
                b[r] = offset
                W[r] = weight

            A = A.tocsr()

            # Connected components for per-component constraints
            adjacency = sparse.lil_matrix((n_bursts, n_bursts))
            for (i, j, _, _) in rows_data:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
            n_comp, labels = connected_components(adjacency.tocsr(), directed=False)

            # Constraint: first burst in each component = 0
            constraint_weight = np.sum(W) * 100 if np.sum(W) > 0 else 1e6
            constraints = []
            for comp in range(n_comp):
                members = np.where(labels == comp)[0]
                row = sparse.lil_matrix((1, n_bursts))
                row[0, members[0]] = 1.0
                constraints.append(row.tocsr())

            A_constrained = sparse.vstack([A] + constraints)
            b_constrained = np.concatenate([b, np.zeros(n_comp)])
            W_constrained = np.concatenate([W, np.full(n_comp, constraint_weight)])

            sqrt_W = np.sqrt(W_constrained)
            result = lsqr(sparse.diags(sqrt_W) @ A_constrained.tocsr(), sqrt_W * b_constrained)
            corrections[:, d_rep] = result[0]

        if debug:
            for i, bid in enumerate(ids):
                corr = corrections[i, :]
                if np.any(corr != 0):
                    print(f'  {bid}: corrections = {[f"{c:.4f}" for c in corr]} rad', flush=True)

        # Apply corrections: multiply SLC by exp(-i * phi) per date
        # Use numpy broadcasting to avoid xarray coordinate alignment issues
        result = {}
        for bid in ids:
            ds = self[bid]
            bidx = id_to_idx[bid]
            phase_corr = corrections[bidx, :]  # shape (n_dates,)
            # Build correction array matching the date dimension position
            corr_arr = np.exp(-1j * phase_corr).astype(np.complex64)

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

        aligned = type(self)(result)

        if return_residuals:
            residuals = [0.0] * n_dates
            for d_rep in rep_dates:
                abs_discrepancies = []
                weights = []
                for k, (id1, id2) in enumerate(overlap_pairs):
                    stat = dd_stats[(d_rep, k)]
                    if stat is None:
                        continue
                    dd_phase, cnt = stat
                    i, j = id_to_idx[id1], id_to_idx[id2]
                    corrected = -dd_phase - (corrections[i, d_rep] - corrections[j, d_rep])
                    corrected = (corrected + np.pi) % (2*np.pi) - np.pi
                    abs_discrepancies.append(abs(corrected))
                    weights.append(float(cnt))
                if abs_discrepancies:
                    residuals[d_rep] = float(np.average(abs_discrepancies, weights=weights))
            return aligned, residuals

        return aligned

    def pairs(self,
              pairs: list[tuple[str|int, str|int]] | np.ndarray | pd.DataFrame
              ) -> Batches:
        """
        Select SLC data organized by interferometric pairs.

        Returns reference and repeat SLC data as a Batches, ready for
        interferogram computation via multiplication.

        Parameters
        ----------
        pairs : list, np.ndarray, or pd.DataFrame
            Pairs of dates as [(ref1, rep1), (ref2, rep2), ...].
            Dates can be indices (int) or date strings.

        Returns
        -------
        Batches
            Batches containing [BatchComplex(ref), BatchComplex(rep)]
            with 'pair' dimension and ref/rep date coordinates.

        Examples
        --------
        # Get paired SLC data
        ref, rep = stack.pairs(baseline.tolist())

        # Manual phase difference
        phasediff = ref * rep.conj()

        # With filtering
        intf = (ref * rep.conj()).gaussian(wavelength=30).angle()
        """
        import numpy as np

        pairs = np.array(pairs if isinstance(pairs[0], (list, tuple, np.ndarray)) else [pairs])

        # Check for duplicate pairs
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        if (counts > 1).any():
            duplicates = unique[counts > 1]
            raise ValueError(f'Input pairs contain duplicates: {duplicates.tolist()}')

        ref_dates = pairs[:, 0]
        rep_dates = pairs[:, 1]
        n_pairs = len(ref_dates)

        # Rename date->pair and reset to integer index
        data1 = self.isel(date=ref_dates).rename(date='pair').map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))
        data2 = self.isel(date=rep_dates).rename(date='pair').map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))

        # BPR differences aligned with pair dimension: BPR(rep) - BPR(ref)
        # Keep as per-burst dict structure (each burst has its own BPR)
        #
        # DataArray arithmetic, not Dataset: a Dataset operation applies to the
        # grids only, so `data2[['BPR']] - data1[['BPR']]` would carry data2's
        # BPR through unchanged and every baseline would come back 0.
        bpr = Batch({k: (data2[k]['BPR'] - data1[k]['BPR']).to_dataset(name='BPR')
                     for k in data1.keys()})

        # Store original datetime values for ref/rep (already materialized via .values)
        ref_values = self.isel(date=ref_dates).coords['date'].values
        rep_values = self.isel(date=rep_dates).coords['date'].values

        def add_pair_coords(batch):
            # Add ref/rep/BPR as non-dimension coordinates along pair dimension
            return batch.assign_coords(
                ref=('pair', ref_values),
                rep=('pair', rep_values),
                BPR=('pair', bpr)
            )

        ref_batch = add_pair_coords(BatchComplex(data1))
        rep_batch = add_pair_coords(BatchComplex(data2))

        return Batches([ref_batch, rep_batch])

    def elevation(self, phase: Batch | float | list | "np.ndarray", baseline: float | None = None, transform: Batch | None = None) -> "Batch | float | np.ndarray":
        """Compute elevation (meters) from unwrapped phase grids in radar coordinates.

        Parameters
        ----------
        phase : Batch | float | list | np.ndarray
            Unwrapped phase grids (e.g., output of unwrap2d()), or a scalar/array
            of phase values for quick height-per-fringe calculations.
        baseline : float | None, optional
            Perpendicular baseline in meters. Required when phase is scalar/array.
            If ``None`` and phase is Batch, use the burst-specific ``BPR``.
        transform : Batch | None, optional
            Transform batch containing look vectors; its `.incidence()` is used.
            If ``None``, defaults to ``self.transform()``.

        Returns
        -------
        Batch | float | np.ndarray
            Elevation grids as float32 datasets, or scalar/array if input was scalar/array.
        """
        import xarray as xr
        import numpy as np

        # Handle scalar/list/array input for quick calculations
        if not isinstance(phase, BatchCore):
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
                    if var.ndim == 0:
                        return float(var.item())
                    # For per-date variables, return mean value
                    return float(var.mean().item())
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
            fac = transform._elevation_phase_approximate()[first_key]

            # Convert input to numpy array
            phase_arr = np.asarray(phase)
            is_scalar = phase_arr.ndim == 0

            ref_height = _scalar_from_ds(tfm, 'ref_height') or 0.0

            # phi = fac * B_perp * dh  ->  dh = phi / (fac * B_perp)
            elev = ref_height - phase_arr / (fac * baseline)

            # Return same type as input, rounded to 3 decimals
            if is_scalar:
                return round(float(elev), 3)
            return np.round(elev, 3)

        # Default to self.transform() when called on Stack and transform not provided
        if transform is None:
            transform = self.transform()
        ep_batch = transform.elevation_phase()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in phase.items():
            if key not in ep_batch:
                raise KeyError(f'Missing geometry for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    # For per-date variables, return mean value
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            if wavelength is None:
                raise KeyError(f"Missing radar_wavelength in transform for burst {key}")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                # Use BPR as DataArray to broadcast correctly across pairs
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            fac_da = ep_batch[key]['elevation_phase']

            ref_height = _scalar_from_ds(tfm, 'ref_height') or 0.0

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                fac_da_i = fac_da.reindex_like(data, method='nearest')
                # the geometry is bent to the data's chunks, never the other way round
                if data.chunks is not None and fac_da_i.chunks is not None:
                    _ch = tuple(data.chunks[-2:][('y', 'x').index(a)] for a in fac_da_i.dims)
                    if fac_da_i.chunks != _ch:
                        fac_da_i = fac_da_i.chunk(dict(zip(fac_da_i.dims, _ch)))

                # ONE geometry, from elevation_phase(). This inlined
                # SC_height * cos(incidence): the satellite height where the
                # slant range belongs and the cosine where the sine belongs.
                # phi = fac * B_perp * dh  ->  dh = phi / (fac * B_perp)
                elev = ref_height - data / (fac_da_i * bpr)
                elev_vars[var_name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def displacement_los(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Delegates like incidence(): the conversion is a scalar scaling by the
        mission wavelength and needs no radar geometry, so the implementation
        lives on Batch and Stack only supplies the default transform.
        """
        if isinstance(phase, BatchWrap):
            raise TypeError(
                'displacement_los() requires unwrapped phase (Batch), got BatchWrap. '
                'Use unwrap3d() or unwrap2d() first to unwrap the phase.'
            )
        if not isinstance(phase, Batch):
            raise TypeError(
                f'phase must be Batch (unwrapped phase), got {type(phase).__name__}.'
            )
        # Batch.displacement_los() has no None default: resolve it here.
        if transform is None:
            transform = self.transform()
        return phase.displacement_los(transform)

    def _displacement_component(self, phase: Batch, transform: Batch = None, func=None, suffix: str = '') -> Batch:
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np

        # Default to self.transform() when not provided, decimated to match phase resolution
        tfm_is_default = transform is None
        if transform is None:
            transform = self.transform()

        # Decimate default transform to match input phase resolution for efficiency
        if tfm_is_default and transform is not None:
            transform = Batch({k: transform[k].reindex(y=phase[k].y, x=phase[k].x, method='nearest')
                               for k in phase.keys() if k in transform})

        los_batch = self.displacement_los(phase, transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
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
        import numpy as np
        import pandas as pd
        from .Baseline import Baseline

        if days is None:
            days = int(1e6)

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

        # Build baseline table
        tbl = pd.DataFrame({'date': dates, 'BPR': bpr_values}).set_index('date')

        # Generate pairs
        data = []
        for line1 in tbl.itertuples():
            for line2 in tbl.itertuples():
                if not (line1.Index < line2.Index and (line2.Index - line1.Index).days <= days):
                    continue
                if meters is not None and not (abs(line1.BPR - line2.BPR) <= meters):
                    continue

                if not invert:
                    data.append({
                        'ref': line1.Index,
                        'rep': line2.Index,
                        'ref_baseline': np.round(line1.BPR, 2),
                        'rep_baseline': np.round(line2.BPR, 2)
                    })
                else:
                    data.append({
                        'ref': line2.Index,
                        'rep': line1.Index,
                        'ref_baseline': np.round(line2.BPR, 2),
                        'rep_baseline': np.round(line1.BPR, 2)
                    })

        if not data:
            raise ValueError("No valid baseline pairs found. "
                             "Try increasing 'days' or 'meters'.")

        df = pd.DataFrame(data).sort_values(['ref', 'rep']).reset_index(drop=True)

        if max_connections is not None:
            _dur = (df['rep'] - df['ref']).dt.days
            keep = pd.Series(True, index=df.index)
            all_dates = sorted(set(df['ref']) | set(df['rep']))
            for date in all_dates:
                # Outgoing pairs (date as ref)
                ref_mask = (df['ref'] == date) & keep
                if ref_mask.sum() > max_connections:
                    ref_dur = _dur[ref_mask].sort_values()
                    keep[ref_dur.index[max_connections:]] = False
                # Incoming pairs (date as rep)
                rep_mask = (df['rep'] == date) & keep
                if rep_mask.sum() > max_connections:
                    rep_dur = _dur[rep_mask].sort_values()
                    keep[rep_dur.index[max_connections:]] = False
            df = df[keep].reset_index(drop=True)

        if cleanup:
            from .Baseline import _cleanup_network
            df = _cleanup_network(df, min_connections=min_connections)

        if len(df) == 0:
            raise ValueError("No valid baseline pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")

        df = df.reset_index(drop=True)
        df = df.assign(
            pair=[f'{ref.date()} {rep.date()}' for ref, rep in zip(df['ref'], df['rep'])],
            baseline=df['rep_baseline'] - df['ref_baseline'],
            duration=(df['rep'] - df['ref']).dt.days
        )

        return Baseline(df, burst_id=first_key, dates=dates)
