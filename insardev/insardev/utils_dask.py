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


def get_dask_chunk_size_mb() -> int:
    """
    Get Dask chunk size from configuration in megabytes.

    Reads from dask.config 'array.chunk-size' (e.g., '128MiB', '256MB').
    Returns 128 MB as default if not configured.

    Returns
    -------
    int
        Dask chunk size in megabytes.
    """
    chunk_size_str = dask.config.get('array.chunk-size', '128MiB')

    # Parse string like '128MiB', '256MB', '1GiB'
    chunk_size_str = chunk_size_str.strip().upper()

    if chunk_size_str.endswith('GIB') or chunk_size_str.endswith('GB'):
        suffix = 'GIB' if 'GIB' in chunk_size_str else 'GB'
        value = float(chunk_size_str.replace(suffix, '').strip())
        return int(value * 1024)
    elif chunk_size_str.endswith('MIB') or chunk_size_str.endswith('MB'):
        suffix = 'MIB' if 'MIB' in chunk_size_str else 'MB'
        value = float(chunk_size_str.replace(suffix, '').strip())
        return int(value)
    elif chunk_size_str.endswith('KIB') or chunk_size_str.endswith('KB'):
        suffix = 'KIB' if 'KIB' in chunk_size_str else 'KB'
        value = float(chunk_size_str.replace(suffix, '').strip())
        return max(1, int(value / 1024))
    else:
        # Assume bytes
        try:
            return max(1, int(float(chunk_size_str) / (1024 * 1024)))
        except ValueError:
            return 128  # Default fallback


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
                "fit3d() cannot count the CPU cores. Use Python 3.13+, or "
                "declare the cores in the Dask cluster init: Client(..., "
                "resources={'cpu': N}) with N the machine's CPU cores.")
        _cores = max(1, _os.process_cpu_count() or 1)
    return _slots, _cores


def get_aligned_chunk_size(original_chunk: int, target_bytes: int, element_bytes: int,
                           other_dims_size: int = 1, min_chunk: int = 256) -> int:
    """
    Calculate aligned chunk size that divides evenly into original chunk.

    Uses power-of-2 divisions (//2, //4, //8, ...) to ensure new chunks align
    perfectly with original chunk boundaries, avoiding inefficient overlapping
    chunks that degrade dask performance.

    Parameters
    ----------
    original_chunk : int
        Original chunk size in pixels for this dimension.
    target_bytes : int
        Target memory per chunk in bytes.
    element_bytes : int
        Bytes per element (e.g., 8 for complex64, 4 for float32).
    other_dims_size : int
        Product of sizes of other dimensions in the chunk (e.g., n_dates * other_spatial).
    min_chunk : int
        Minimum chunk size (default 256).

    Returns
    -------
    int
        Aligned chunk size that divides evenly into original_chunk.

    Examples
    --------
    >>> # Original 4096 chunk, need smaller for memory
    >>> get_aligned_chunk_size(4096, 256*1024*1024, 8, other_dims_size=5*4096)
    2048  # Returns 2048 (4096//2) if it fits, else 1024 (4096//4), etc.
    """
    # Calculate maximum chunk size that fits in target memory
    # chunk_y * other_dims_size * element_bytes <= target_bytes
    max_chunk = target_bytes // (other_dims_size * element_bytes)
    max_chunk = max(min_chunk, max_chunk)

    # If original chunk already fits, use it
    if original_chunk <= max_chunk:
        return original_chunk

    # Find largest power-of-2 division that fits
    # Try //2, //4, //8, //16, ... until it fits or hits min_chunk
    divisor = 2
    while True:
        candidate = original_chunk // divisor
        if candidate <= max_chunk:
            return max(min_chunk, candidate)
        if candidate <= min_chunk:
            return min_chunk
        divisor *= 2


def subdivide_chunk(chunk_size: int, n_parts: int) -> tuple:
    """
    Subdivide a single chunk into n roughly equal parts.

    Parameters
    ----------
    chunk_size : int
        Original chunk size.
    n_parts : int
        Number of parts to split into.

    Returns
    -------
    tuple
        Tuple of part sizes that sum to chunk_size.

    Examples
    --------
    >>> subdivide_chunk(101, 2)
    (50, 51)
    >>> subdivide_chunk(301, 3)
    (100, 100, 101)
    >>> subdivide_chunk(100, 4)
    (25, 25, 25, 25)
    """
    if n_parts <= 1:
        return (chunk_size,)

    base_size = chunk_size // n_parts
    remainder = chunk_size % n_parts

    # Distribute remainder across parts: first 'remainder' parts get +1
    parts = []
    for i in range(n_parts):
        if i < remainder:
            parts.append(base_size + 1)
        else:
            parts.append(base_size)

    return tuple(parts)


def subdivide_chunks_dim(chunks: tuple, n_parts: int) -> tuple:
    """
    Subdivide all chunks in one dimension into n parts each.

    Parameters
    ----------
    chunks : tuple
        Original chunk sizes for one dimension, e.g., (5001, 4003, 5001).
    n_parts : int
        Number of parts to split each chunk into.

    Returns
    -------
    tuple
        New chunk sizes, e.g., (2500, 2501, 2001, 2002, 2500, 2501).

    Examples
    --------
    >>> subdivide_chunks_dim((100, 80), 2)
    (50, 50, 40, 40)
    >>> subdivide_chunks_dim((101, 99), 2)
    (51, 50, 50, 49)
    """
    result = []
    for chunk in chunks:
        result.extend(subdivide_chunk(chunk, n_parts))
    return tuple(result)


def compute_subdivided_chunks(
    original_chunks: tuple,
    target_bytes: int,
    element_bytes: int,
    min_chunk: int = 64,
) -> tuple:
    """
    Compute subdivided chunks that fit memory budget without crossing original boundaries.

    Each original chunk is split into roughly equal parts. The subdivision factor
    is chosen to make chunks fit within target_bytes while keeping chunks above min_chunk.

    Parameters
    ----------
    original_chunks : tuple
        Original chunk sizes, e.g., ((5001, 4003), (1921, 1921)) for 2D.
    target_bytes : int
        Target memory per chunk in bytes.
    element_bytes : int
        Bytes per element.
    min_chunk : int
        Minimum chunk size per dimension.

    Returns
    -------
    tuple
        New chunk sizes for each dimension.

    Examples
    --------
    >>> # Original chunks 5001×1921, target 16MB for float32
    >>> compute_subdivided_chunks(((5001,), (1921,)), 16*1024*1024, 4)
    ((2500, 2501), (960, 961))  # approximately
    """
    ndim = len(original_chunks)
    pixel_budget = target_bytes // element_bytes

    # Get max chunk size per dimension
    max_y = max(original_chunks[0]) if original_chunks[0] else 1
    max_x = max(original_chunks[1]) if ndim > 1 and original_chunks[1] else 1

    # If already fits, no subdivision needed
    if max_y * max_x <= pixel_budget:
        return original_chunks

    # Find subdivision factor that fits budget
    # Try subdividing both dimensions equally first
    for n in range(1, 32):  # Up to 32x subdivision
        new_y = (max_y + n - 1) // n  # Ceiling division
        new_x = (max_x + n - 1) // n

        if new_y * new_x <= pixel_budget and new_y >= min_chunk and new_x >= min_chunk:
            # Found good subdivision, apply to all chunks
            new_chunks_y = subdivide_chunks_dim(original_chunks[0], n)
            new_chunks_x = subdivide_chunks_dim(original_chunks[1], n) if ndim > 1 else ()
            if ndim == 2:
                return (new_chunks_y, new_chunks_x)
            elif ndim == 3:
                return (original_chunks[0], new_chunks_y, new_chunks_x)

    # If equal subdivision doesn't work, try asymmetric
    for n_y in range(1, 32):
        for n_x in range(1, 32):
            new_y = (max_y + n_y - 1) // n_y
            new_x = (max_x + n_x - 1) // n_x

            if new_y * new_x <= pixel_budget and new_y >= min_chunk and new_x >= min_chunk:
                new_chunks_y = subdivide_chunks_dim(original_chunks[0], n_y)
                new_chunks_x = subdivide_chunks_dim(original_chunks[1], n_x) if ndim > 1 else ()
                if ndim == 2:
                    return (new_chunks_y, new_chunks_x)
                elif ndim == 3:
                    return (original_chunks[0], new_chunks_y, new_chunks_x)

    # Fallback: just return original (shouldn't happen with reasonable inputs)
    return original_chunks


def compute_aligned_chunks_2d(shape: tuple, original_chunks: tuple, target_bytes: int,
                               element_bytes: int, min_chunk: int = 64) -> tuple:
    """
    Compute aligned chunk sizes for 2D spatial data.

    Subdivides each original chunk into roughly equal parts that fit within
    the memory budget. Never crosses original chunk boundaries.

    Parameters
    ----------
    shape : tuple
        Array shape (y, x).
    original_chunks : tuple
        Original chunk sizes ((y_chunks...), (x_chunks...)).
    target_bytes : int
        Target memory per chunk in bytes.
    element_bytes : int
        Bytes per element.
    min_chunk : int
        Minimum chunk size.

    Returns
    -------
    tuple
        New chunk sizes as tuples for each dimension.
    """
    return compute_subdivided_chunks(original_chunks, target_bytes, element_bytes, min_chunk)




def compute_aligned_chunks_3d(shape: tuple, original_chunks: tuple, target_bytes: int,
                               element_bytes: int, min_chunk: int = 64,
                               keep_first_dim: bool = True) -> tuple:
    """
    Compute aligned chunk sizes for 3D data (e.g., date/pair, y, x).

    Subdivides each original spatial chunk into roughly equal parts that fit within
    the memory budget. Never crosses original chunk boundaries.

    Parameters
    ----------
    shape : tuple
        Array shape (n, y, x) where n is dates or pairs.
    original_chunks : tuple
        Original chunk sizes ((n_chunks...), (y_chunks...), (x_chunks...)).
    target_bytes : int
        Target memory per chunk in bytes.
    element_bytes : int
        Bytes per element.
    min_chunk : int
        Minimum spatial chunk size.
    keep_first_dim : bool
        If True, keep first dimension as single chunk (all dates/pairs together).
        If False, also chunk first dimension.

    Returns
    -------
    tuple
        New chunk sizes as tuples for each dimension.
    """
    n_dim = shape[0]

    if keep_first_dim:
        n_chunks = (n_dim,)
        # Adjust target for all dates in one chunk
        spatial_target_bytes = target_bytes // n_dim
    else:
        n_chunks = original_chunks[0]
        # Use first n_chunk as representative
        n_chunk = n_chunks[0] if n_chunks else n_dim
        spatial_target_bytes = target_bytes // n_chunk

    # Subdivide spatial dimensions
    spatial_chunks = (original_chunks[1], original_chunks[2])
    new_spatial = compute_subdivided_chunks(spatial_chunks, spatial_target_bytes, element_bytes, min_chunk)

    return (n_chunks, new_spatial[0], new_spatial[1])


def validate_chunks_aligned(original_chunks: tuple, new_chunks: tuple) -> bool:
    """
    Validate that new chunks don't cross original chunk boundaries.

    Parameters
    ----------
    original_chunks : tuple
        Original chunk sizes per dimension.
    new_chunks : tuple
        New chunk sizes per dimension.

    Returns
    -------
    bool
        True if aligned, False if any new chunk crosses original boundary.

    Raises
    ------
    ValueError
        If new chunks cross original boundaries.
    """
    for dim, (orig_dim, new_dim) in enumerate(zip(original_chunks, new_chunks)):
        # Build boundary positions for original chunks
        orig_boundaries = set()
        pos = 0
        for chunk in orig_dim:
            pos += chunk
            orig_boundaries.add(pos)

        # Check that all new chunk boundaries align with original
        pos = 0
        for chunk in new_dim:
            pos += chunk
            # Every new boundary must be at an original boundary
            # Exception: intermediate boundaries within an original chunk are fine
            # What we need to check: the cumulative sum of new chunks must hit
            # every original boundary exactly

        # Alternative check: sum of new chunks between original boundaries
        # must equal original chunk size
        orig_pos = 0
        new_idx = 0
        new_pos = 0

        for orig_chunk in orig_dim:
            orig_end = orig_pos + orig_chunk
            # Sum new chunks until we reach orig_end
            sum_new = 0
            while new_idx < len(new_dim) and new_pos < orig_end:
                sum_new += new_dim[new_idx]
                new_pos += new_dim[new_idx]
                new_idx += 1

            if new_pos != orig_end:
                raise ValueError(
                    f"New chunks cross original boundary in dimension {dim}: "
                    f"original chunk ends at {orig_end}, but new chunks end at {new_pos}. "
                    f"Original: {orig_dim}, New: {new_dim}"
                )
            orig_pos = orig_end

    return True


def rechunk3d(data, target_mb: int = None, min_chunk: int = 64,
              keep_first_dim: bool = True, debug: bool = False):
    """
    Rechunk dask array to fit within Dask chunk size without crossing original boundaries.

    Subdivides each original chunk into roughly equal parts. The new chunks never
    cross original chunk boundaries, ensuring efficient dask operations.

    If original chunks already fit within target, returns data unchanged to allow
    users to apply custom rechunking via .chunk() if needed.

    Parameters
    ----------
    data : dask.array.Array or xarray.DataArray
        Input data to rechunk.
    target_mb : int, optional
        Target memory per chunk in megabytes. If None (default), uses Dask config
        'array.chunk-size' setting (typically 128MB).
    min_chunk : int
        Minimum spatial chunk size (default 64).
    keep_first_dim : bool
        For 3D data, keep first dimension (dates/pairs) as single chunk.
    debug : bool
        Print debug information.

    Returns
    -------
    same type as input
        Rechunked data with subdivided chunk boundaries, or original data if
        no rechunking needed.

    Raises
    ------
    ValueError
        If computed new chunks would cross original chunk boundaries (indicates bug).

    Examples
    --------
    >>> # Rechunk using Dask config chunk size
    >>> slc_rechunked = rechunk3d(slc_data, keep_first_dim=True)

    >>> # Original: ((5001, 4003), (1921, 1921))
    >>> # New:      ((2500, 2501, 2001, 2002), (960, 961, 960, 961))

    >>> # If chunks already fit, returns unchanged (allows custom rechunking)
    >>> small_data = rechunk3d(data, target_mb=256)  # returns as-is if fits
    >>> custom = small_data.chunk({'y': 1000, 'x': 1000})  # user can customize
    """
    import dask.array as da
    import xarray as xr

    # Handle xarray DataArray
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        dask_data = data.data
        if not isinstance(dask_data, da.Array):
            return data
    else:
        dask_data = data

    if not isinstance(dask_data, da.Array):
        return data

    # Get Dask chunk size from config if not specified
    if target_mb is None:
        target_mb = get_dask_chunk_size_mb()

    target_bytes = target_mb * 1024 * 1024
    element_bytes = dask_data.dtype.itemsize
    shape = dask_data.shape
    original_chunks = dask_data.chunks
    ndim = dask_data.ndim

    # Check if rechunking is actually needed
    current_chunk_bytes = 1
    for i, dim_chunks in enumerate(original_chunks):
        if ndim == 3 and keep_first_dim and i == 0:
            current_chunk_bytes *= shape[0]
        else:
            current_chunk_bytes *= max(dim_chunks) if dim_chunks else 1
    current_chunk_bytes *= element_bytes

    if current_chunk_bytes <= target_bytes:
        if debug:
            print(f"rechunk3d: effective chunks ({current_chunk_bytes/1e6:.1f} MB) "
                  f"already fit target ({target_mb} MB), returning unchanged")
        # Return data as-is to allow user custom rechunking
        # Only exception: for 3D with keep_first_dim, ensure first dim is merged
        if ndim == 3 and keep_first_dim and original_chunks[0] != (shape[0],):
            new_chunks = {0: shape[0]}
            if is_xarray:
                return data.chunk(new_chunks)
            else:
                return dask_data.rechunk(new_chunks)
        return data

    # Compute new chunks
    if ndim == 2:
        new_chunk_sizes = compute_aligned_chunks_2d(
            shape, original_chunks, target_bytes, element_bytes, min_chunk
        )
    elif ndim == 3:
        new_chunk_sizes = compute_aligned_chunks_3d(
            shape, original_chunks, target_bytes, element_bytes, min_chunk, keep_first_dim
        )
    else:
        if debug:
            print(f"rechunk3d: unsupported ndim={ndim}, returning unchanged")
        return data

    # Validate that new chunks don't cross original boundaries
    validate_chunks_aligned(original_chunks, new_chunk_sizes)

    if debug:
        print(f"rechunk3d: {shape}, dtype={dask_data.dtype}")
        print(f"  original chunks: {original_chunks}")
        print(f"  original chunk size: {current_chunk_bytes/1e6:.1f} MB")
        print(f"  target: {target_mb} MB")
        print(f"  new chunks: {new_chunk_sizes}")
        max_new_chunk = 1
        for dim_chunks in new_chunk_sizes:
            max_new_chunk *= max(dim_chunks) if dim_chunks else 1
        print(f"  max new chunk size: {max_new_chunk * element_bytes / 1e6:.1f} MB")

    if is_xarray:
        dims = data.dims
        chunk_dict = {dims[i]: new_chunk_sizes[i] for i in range(len(new_chunk_sizes))}
        return data.chunk(chunk_dict)
    else:
        return dask_data.rechunk(new_chunk_sizes)


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
    target_mb : int, optional
        Target memory per chunk in MB. If None, uses dask config.
    min_chunk : int
        Minimum chunk size (default 64).
    max_chunk : int
        Maximum chunk size (default 16384).
    merge : bool
        If True, adjacent small input chunks may be merged into larger
        output tiles.  If False (default), each input chunk is kept
        whole or split — never merged with neighbors.  False avoids
        cross-chunk rechunk graphs that explode in layered pipelines.

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
        target_c = _find_optimal_chunk_pair(
            y_size, x_size, pixel_budget, element_bytes, min_chunk, max_chunk)
        yg = split_dim(y_in, target_c['y'])
        xg = split_dim(x_in, target_c['x'])
        candidates.append((yg, xg))

        # Pick strategy with fewest output tiles
        best_yg, best_xg = min(candidates, key=lambda c: len(c[0]) * len(c[1]))

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
        y_size, x_size, pixel_budget, element_bytes, min_chunk, max_chunk)
    if is_3d:
        result[0] = 1
    return result


def _find_optimal_chunk_pair(
    y_size: int, x_size: int,
    pixel_budget: int, element_bytes: int,
    min_chunk: int = 64, max_chunk: int = 16384,
) -> dict:
    """Find optimal (cy, cx) chunk pair for a given array shape and budget."""
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
