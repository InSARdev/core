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

    best_score = float('inf')
    best_result = None

    for cy in valid_y:
        for cx in valid_x:
            if cy * cx > max_pixels:
                continue
            aspect = max(cy / cx, cx / cy)
            if aspect > max_aspect:
                continue
            score = _compute_chunk_score(cy, cx, y_size, x_size, pixel_budget)
            if score < best_score:
                best_score = score
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


class _HeldTask(_Task):
    """The task of one held chunk (hold_persisted). It is NEVER FUSED into the chain that
    reads it: fused, it went into a different chain in each graph (one reader in an angle,
    two in an align), so the same downstream key -- the key of data another compute still
    holds -- became an alias of a different fused task in the next graph. The scheduler
    keeps the known key's old dependencies (dask issue 9888), and the compute failed with
    KeyError ('held-sub-sub-...'), an AssertionError on a TaskState, or hung. Unfused, a
    chain built on computed data starts above this task in every graph, as it did on the
    Futures before the hold."""
    __slots__ = ()

    @property
    def block_fusion(self) -> bool:
        return True


class _UnfusedTask(_Task):
    """A task that dask never fuses with its neighbours, so its key names the same task in every graph
    (Stack._zarr_dask: the zarr readers of Stack.load(); see _HeldTask for the same failure)."""
    __slots__ = ()

    @property
    def block_fusion(self) -> bool:
        return True


def _unfused_block(block):
    """Identity task that reads one block of a shared array (see unfused)."""
    return block


def unfused(arr):
    """
    The same dask array, each block read by a task dask never fuses (_UnfusedTask).

    FOR AN ARRAY TWO RESULTS SHARE: the Gaussian-filtered interferogram, which both the
    phase and the correlation of interferogram() read. In a graph of the phase alone,
    the filter was fused into the phase's chain and the phase's key became an alias of
    that fused task; in a graph of both, the filter stayed apart and the same key was
    one task over it. The scheduler, still holding the key from one computation (the
    unwrapped phase), kept its old definition for the next (dask issue 9888, 'different
    run_spec' on 'sub-...'). The same for the interferogram product under the filter:
    one block spans every pair, and a graph of one pair fused it into that pair's chain.
    Read through here, the consumers build on the same unfused keys in every graph, so
    every graph defines them the same way.

    Parameters
    ----------
    arr : dask.array.Array
        Any dask array; anything else is returned as is.

    Returns
    -------
    dask.array.Array
        Same chunks, dtype and data.
    """
    import dask.array as da
    from dask.base import tokenize
    from dask.core import flatten
    from dask._task_spec import TaskRef
    from dask.highlevelgraph import HighLevelGraph
    if not isinstance(arr, da.Array):
        return arr
    name = f'unfused-{tokenize(arr.name)}'
    layer = {}
    for key in flatten(arr.__dask_keys__()):
        out = (name,) + tuple(key[1:])
        layer[out] = _UnfusedTask(out, _unfused_block, TaskRef(key))
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[arr])
    return da.Array(graph, name, chunks=arr.chunks, dtype=arr.dtype, meta=arr._meta)


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
    The reading task is never fused (_HeldTask), so every graph gives the keys built on it
    the same task.

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
        layer[k2] = _HeldTask(k2, _held_chunk, f)
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
