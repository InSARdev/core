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
from .Stack_unwrap1d import Stack_unwrap1d
from . import utils_unwrap2d
import numpy as np


def _irls_process_chunk_with_tuple(phase_chunk, weight_chunk, params_tuple):
    """Module-level function for IRLS processing with params as tuple.

    Returns tuple of (unwrapped, conncomp) arrays, each with shape (1, y, x).
    """
    # Unpack params tuple: (device, max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug)
    device, max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug = params_tuple

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

    # Call utils function directly - no object references
    unwrapped, conncomp = utils_unwrap2d.irls_unwrap_2d(
        phase_2d, weight=weight_2d, device=device,
        max_iter=max_iter, tol=tol, cg_max_iter=cg_max_iter,
        cg_tol=cg_tol, epsilon=epsilon, conncomp_size=conncomp_size, debug=debug
    )
    # Add pair dim back: (y, x) -> (1, y, x)
    return unwrapped[np.newaxis, ...], conncomp[np.newaxis, ...]


def _irls_process_no_weight(phase_chunk, params_tuple):
    """Process phase chunk without weight. Returns only unwrapped phase."""
    unwrapped, _ = _irls_process_chunk_with_tuple(phase_chunk, None, params_tuple)
    return unwrapped


def _irls_process_with_weight(phase_chunk, weight_chunk, params_tuple):
    """Process phase chunk with weight. Returns only unwrapped phase."""
    unwrapped, _ = _irls_process_chunk_with_tuple(phase_chunk, weight_chunk, params_tuple)
    return unwrapped


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


class Stack_unwrap2d(Stack_unwrap1d):
    """2D phase unwrapping using GPU-accelerated IRLS algorithm with DCT initialization."""

    # 4-connectivity structure for scipy.ndimage.label (no diagonals)
    _STRUCTURE_4CONN = utils_unwrap2d.STRUCTURE_4CONN

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

            result[key] = xr.Dataset(reordered_vars, coords=ds.coords, attrs=ds.attrs)

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

            result[key] = xr.Dataset(label_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

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

            result[key] = xr.Dataset(linked_vars, coords=ds.coords, attrs=ds.attrs)

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
                conncomp_linksize=5, conncomp_linkcount=30, device='auto', debug=False, **kwargs):
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

    def unwrap2d_dataset(self, phase, weight=None, conncomp=False,
                         conncomp_size=1000, conncomp_gap=None,
                         conncomp_linksize=5, conncomp_linkcount=30, device='auto', debug=False, **kwargs):
        """
        Unwrap a single phase Dataset using GPU-accelerated IRLS algorithm.

        Convenience wrapper around unwrap2d() for working with merged datasets
        instead of per-burst batches. Useful when you have already dissolved
        and merged your data.

        Parameters
        ----------
        phase : xr.Dataset
            Wrapped phase dataset (from intf.align().dissolve().to_dataset()).
        weight : xr.Dataset, optional
            Correlation values for weighting (from corr.dissolve().to_dataset()).
        conncomp : bool, optional
            If False (default), link disconnected components.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int, optional
            Minimum pixels for a connected component. Default is 1000.
        conncomp_gap : int or None, optional
            Maximum pixel distance between components. Default is None.
        conncomp_linksize : int, optional
            Pixels for offset estimation. Default is 5.
        conncomp_linkcount : int, optional
            Maximum neighbor components to consider. Default is 30.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', 'cpu', or 'tpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print diagnostic information. Default is False.
        **kwargs
            Additional arguments passed to unwrap2d_irls.

        Returns
        -------
        xr.Dataset or tuple
            If conncomp is False: Unwrapped phase Dataset.
            If conncomp is True: tuple of (unwrapped Dataset, conncomp Dataset).

        Examples
        --------
        Basic usage with merged datasets:
        >>> intf_ds = intf.align().dissolve().compute().to_dataset()
        >>> corr_ds = corr.dissolve().compute().to_dataset()
        >>> unwrapped = stack.unwrap2d_dataset(intf_ds, corr_ds)

        Get connected components:
        >>> unwrapped, conncomp = stack.unwrap2d_dataset(intf_ds, corr_ds, conncomp=True)

        Convert back to per-burst Batch:
        >>> phase_batch = intf.from_dataset(unwrapped)
        """
        import xarray as xr
        from .Batch import BatchWrap, BatchUnit

        # Validate input types
        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")

        # Check for spatial chunking and rechunk if needed
        # Phase unwrapping is a global operation that needs single spatial chunk
        needs_rechunk = False
        for var_name in phase.data_vars:
            var_data = phase[var_name]
            if hasattr(var_data.data, 'chunks'):
                chunks = var_data.data.chunks
                # For 2D or 3D data, check spatial dimensions (last two)
                y_chunks = chunks[-2] if len(chunks) >= 2 else chunks[0]
                x_chunks = chunks[-1]
                if len(y_chunks) > 1 or len(x_chunks) > 1:
                    print(f"NOTE: unwrap2d_dataset() rechunking to single spatial chunk "
                          f"(from y: {len(y_chunks)} chunks, x: {len(x_chunks)} chunks)")
                    needs_rechunk = True
                break  # Only check first variable

        if needs_rechunk:
            # Rechunk to single spatial chunk (keep first dim if present)
            phase = phase.chunk({'y': -1, 'x': -1})
            if weight is not None:
                weight = weight.chunk({'y': -1, 'x': -1})

        # Wrap datasets in temporary batches with empty key
        intf_batch = BatchWrap({'': phase})
        corr_batch = BatchUnit({'': weight}) if weight is not None else None

        # Call unwrap2d
        result = self.unwrap2d(
            intf_batch, corr_batch, conncomp=conncomp,
            conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
            conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
            device=device, debug=debug, **kwargs
        )

        # Extract and return the dataset(s)
        if conncomp:
            unwrapped, conncomp_labels = result
            return unwrapped[''], conncomp_labels['']
        else:
            return result['']

    # =========================================================================
    # GPU-Accelerated Phase Unwrapping Methods (PyTorch)
    # =========================================================================

    def unwrap2d_irls(self, phase, weight=None, device='auto',
                      max_iter=50, tol=1e-2, cg_max_iter=10, cg_tol=1e-3, epsilon=1e-2,
                      conncomp_size=30, debug=False):
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
        resolved = Stack_unwrap2d._get_torch_device(device, debug=debug)
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

                # Ensure pair dimension is chunked as 1
                if 'pair' in phase_da.dims:
                    chunks = phase_da.data.chunks
                    if chunks[0][0] != 1:
                        phase_da = phase_da.chunk({'pair': 1})
                    if weight_da is not None:
                        weight_chunks = weight_da.data.chunks
                        if weight_chunks[0][0] != 1:
                            weight_da = weight_da.chunk({'pair': 1})

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

                # Resource annotation limits concurrent unwrap2d operations
                # Workers must be configured with resources={'unwrap2d': N, 'gpu': M} to limit concurrency
                use_gpu = device != 'cpu'
                task_resources = {'unwrap2d': 1, 'gpu': 1} if use_gpu else {'unwrap2d': 1}
                # Provide meta to avoid calling process_wrapper during graph construction
                meta = np.empty((0, 0, 0), dtype=np.float32)

                # Create params tuple - simple immutable structure for reliable serialization
                # Use str() to create a fresh string copy
                params_tuple = (str(device), max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug)

                if debug:
                    print(f'DEBUG creating params tuple with device={params_tuple[0]!r}')

                # Use conncomp versions that return stacked (1, 2, y, x) per chunk
                # Output shape is (n_pairs, 2, y, x)
                meta_stacked = np.empty((0, 0, 0, 0), dtype=np.float32)

                with dask.annotate(resources=task_resources):
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

                unwrap_vars[var] = unwrap_da
                conncomp_vars[var] = conncomp_da

            # Preserve dataset attributes (subswath, pathNumber, etc.)
            unwrap_result[key] = xr.Dataset(unwrap_vars, attrs=phase_ds.attrs)
            conncomp_result[key] = xr.Dataset(conncomp_vars, attrs=phase_ds.attrs)
            # Preserve CRS from input dataset
            if phase_ds.rio.crs is not None:
                unwrap_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)
                conncomp_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)

        from .Batch import BatchUnit, Batches
        return Batches((Batch(unwrap_result), BatchUnit(conncomp_result)))

    def _process_irls_slice(self, phase_np, weight_np, device,
                            max_iter, tol, cg_max_iter, cg_tol, epsilon, conncomp_size, debug):
        """Process a single 2D phase slice with IRLS unwrapping.

        Unwraps the full raster directly - disconnected components (separated
        by NaN regions) are handled automatically through edge weight zeroing.

        Returns only unwrapped phase (discards conncomp).
        """
        if debug:
            print(f'DEBUG _process_irls_slice: received device={device}')

        if np.all(np.isnan(phase_np)):
            if debug:
                print('  All NaN, skipping')
            return phase_np.astype(np.float32)

        # Unwrap using IRLS directly on full raster
        unwrapped, _ = utils_unwrap2d.irls_unwrap_2d(
            phase_np, weight=weight_np, device=device,
            max_iter=max_iter, tol=tol, cg_max_iter=cg_max_iter,
            cg_tol=cg_tol, epsilon=epsilon, conncomp_size=conncomp_size, debug=debug
        )
        return unwrapped

    # =========================================================================
    # Discontinuity Detection for Phase Unwrapping
    # =========================================================================

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
        >>> unwrapped = stack.unwrap2d_dataset(intf_masked, corr_masked)

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
                        unwrap_2d = Stack_unwrap2d._snaphu_unwrap_array(
                            phase_2d, corr_2d, defomax=defomax_val, debug=debug_val
                        )
                        return unwrap_2d[np.newaxis, ...].astype(np.float32)
                    return process_wrapper

                wrapper = make_wrapper(defomax, debug)

                # Use da.blockwise for efficient dask integration
                phase_dask = phase_da.data
                dim_str = ''.join(chr(ord('a') + i) for i in range(phase_dask.ndim))

                # Resource annotation limits concurrent snaphu operations
                # Workers must be configured with resources={'snaphu': N} to limit concurrency
                task_resources = {'snaphu': 1}
                # Provide meta to avoid calling wrapper during graph construction
                meta = np.empty((0,) * phase_dask.ndim, dtype=np.float32)

                with dask.annotate(resources=task_resources):
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

            result_ds = xr.Dataset(result_vars)
            result_ds.attrs = phase_ds.attrs
            results[key] = result_ds

        return Batch(results)
