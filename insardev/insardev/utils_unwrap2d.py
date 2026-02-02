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
Static utility functions for 2D phase unwrapping.

These functions contain the core algorithms for 2D phase unwrapping,
component detection, and component linking.
"""
import numpy as np

# 4-connectivity structure for scipy.ndimage.label (no diagonals)
STRUCTURE_4CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def get_connected_components(valid_mask_2d, min_size=4):
    """
    Find connected components with bounding boxes using scipy.ndimage.

    Parameters
    ----------
    valid_mask_2d : np.ndarray
        2D boolean array where True indicates valid pixels.
    min_size : int, optional
        Minimum component size to include in results. Default is 4.

    Returns
    -------
    labeled_array : np.ndarray
        2D int32 array with component labels (0 = invalid, 1+ = component labels).
    components : list of dict
        List of component info dicts sorted by size (largest first), each with:
        - 'label': int, the component label in labeled_array
        - 'size': int, number of pixels in the component
        - 'slices': tuple of slices for bounding box
    n_total : int
        Total number of components found (before min_size filtering).
    sizes : np.ndarray
        Array of component sizes indexed by label (sizes[0] = 0, sizes[1] = size of label 1, etc.)
    """
    from scipy import ndimage

    labeled_array, n_total = ndimage.label(valid_mask_2d, structure=STRUCTURE_4CONN)

    if n_total == 0:
        return labeled_array, [], 0, np.array([0])

    # Get sizes and bounding boxes efficiently
    sizes = np.bincount(labeled_array.ravel(), minlength=n_total + 1)
    slices = ndimage.find_objects(labeled_array)

    # Build component list sorted by size (largest first), filtering by min_size
    components = [
        {'label': i + 1, 'size': sizes[i + 1], 'slices': slices[i]}
        for i in np.argsort(sizes[1:])[::-1]
        if sizes[i + 1] >= min_size and slices[i] is not None
    ]

    return labeled_array, components, n_total, sizes


def print_component_stats_debug(method_name, shape, n_valid, n_components, sizes):
    """Print debug statistics about connected components."""
    comp_sizes = sizes[1:n_components + 1] if n_components > 0 else []
    sorted_sizes = np.sort(comp_sizes)[::-1]
    n_tiny = np.sum(comp_sizes < 10) if len(comp_sizes) > 0 else 0

    print(f'{method_name}: {shape} grid, {n_valid} valid pixels, {n_components} components')
    if n_components <= 10:
        print(f'  Component sizes: {list(sorted_sizes)}')
    else:
        print(f'  Largest 5: {list(sorted_sizes[:5])}, smallest 5: {list(sorted_sizes[-5:])}, tiny(<10px): {n_tiny}')


def find_connected_components(valid_mask_2d, min_size=None):
    """
    Find connected components in a 2D valid mask using 4-connectivity.

    Parameters
    ----------
    valid_mask_2d : np.ndarray
        2D boolean array where True indicates valid pixels.
    min_size : int, optional
        Minimum component size to include. If None, all components are returned.

    Returns
    -------
    list of np.ndarray
        List of boolean masks, one per connected component (sorted by size, largest first).
    """
    labeled_array, components, n_total, _ = get_connected_components(
        valid_mask_2d, min_size=min_size or 1
    )
    return [(labeled_array == c['label']) for c in components]


def line_crosses_mask(p1, p2, mask):
    """
    Check if the line segment from p1 to p2 crosses any True pixels in mask.

    Uses Bresenham-like sampling along the line.

    Parameters
    ----------
    p1, p2 : tuple
        (row, col) endpoints of the line segment.
    mask : np.ndarray
        2D boolean array to check against.

    Returns
    -------
    bool
        True if the line crosses any True pixels in mask.
    """
    r1, c1 = p1
    r2, c2 = p2

    # Number of steps (at least the max of row/col difference)
    n_steps = max(abs(r2 - r1), abs(c2 - c1), 1)

    for step in range(1, n_steps):  # Skip endpoints
        t = step / n_steps
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))

        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            if mask[r, c]:
                return True

    return False


def find_component_connections(components, conncomp_gap=None, max_neighbors=30):
    """
    Find direct connections between components based on minimum distance.

    Uses a size-weighted approach: prioritizes connections to larger components
    to ensure small components connect to the main component network.

    Parameters
    ----------
    components : list of np.ndarray
        List of boolean masks, one per connected component.
    conncomp_gap : int or None, optional
        Maximum pixel distance to consider components as connectable.
        If None (default), no distance limit is applied.
    max_neighbors : int, optional
        Maximum number of nearest neighbors to check for each component.
        Default is 30.

    Returns
    -------
    list of tuple
        List of (comp_i, comp_j, closest_i, closest_j, distance) where:
        - comp_i, comp_j: component indices
        - closest_i: (row, col) of closest pixel in component i
        - closest_j: (row, col) of closest pixel in component j
        - distance: Euclidean distance between closest pixels
    """
    n_comps = len(components)
    if n_comps < 2:
        return []

    # Create combined mask of all components for intersection checking
    all_comps_mask = np.zeros_like(components[0], dtype=bool)
    for comp_mask in components:
        all_comps_mask |= comp_mask

    # Get pixel coordinates, centroids, and sizes for each component
    comp_coords = []
    centroids = []
    comp_sizes = []
    for comp_mask in components:
        rows, cols = np.where(comp_mask)
        comp_coords.append(np.column_stack([rows, cols]))
        centroids.append((np.mean(rows), np.mean(cols)))
        comp_sizes.append(len(rows))

    centroids = np.array(centroids)
    comp_sizes = np.array(comp_sizes)

    # Sort components by size (largest first) for prioritized connection
    size_order = np.argsort(comp_sizes)[::-1]

    # For each component, find candidate neighbors using size-weighted scoring
    # Score = size_weight / (distance + 1), prefer larger and closer components
    candidate_pairs = set()
    for i in range(n_comps):
        # Compute distances from this centroid to all others
        dists = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        dists[i] = np.inf  # Exclude self

        # Size-weighted score: larger components get higher priority
        # Use log(size) to avoid extreme weighting
        size_weights = np.log1p(comp_sizes)
        scores = size_weights / (dists + 1)
        scores[i] = -np.inf  # Exclude self

        # Get indices of best candidates (highest scores)
        n_neighbors = min(max_neighbors, n_comps - 1)
        best_candidates = np.argpartition(scores, -n_neighbors)[-n_neighbors:]

        for j in best_candidates:
            if scores[j] > -np.inf:
                # Add as sorted tuple to avoid duplicates
                pair = (min(i, j), max(i, j))
                candidate_pairs.add(pair)

    # Also ensure every component considers connecting to the largest components
    # This guarantees small isolated components can reach the main network
    n_largest = min(5, n_comps)
    largest_indices = size_order[:n_largest]
    for i in range(n_comps):
        if i not in largest_indices:
            for j in largest_indices:
                pair = (min(i, j), max(i, j))
                candidate_pairs.add(pair)

    connections = []

    # Check only candidate pairs
    for i, j in candidate_pairs:
        coords_i = comp_coords[i]
        coords_j = comp_coords[j]

        # For large components, subsample to speed up initial search
        max_sample = 500  # Reduced from 1000 to save memory
        if len(coords_i) > max_sample:
            idx_i = np.random.choice(len(coords_i), max_sample, replace=False)
            sample_i = coords_i[idx_i]
        else:
            sample_i = coords_i

        if len(coords_j) > max_sample:
            idx_j = np.random.choice(len(coords_j), max_sample, replace=False)
            sample_j = coords_j[idx_j]
        else:
            sample_j = coords_j

        # Compute pairwise distances efficiently using scipy
        from scipy.spatial.distance import cdist
        dists = cdist(sample_i, sample_j, metric='euclidean')

        min_dist = np.min(dists)

        # Check distance limit if specified
        if conncomp_gap is not None and min_dist > conncomp_gap:
            continue

        # Find the actual closest pair (in full set if we subsampled)
        search_radius = max(min_dist * 2, 100)  # Search radius for refinement

        if len(coords_i) > max_sample or len(coords_j) > max_sample:
            # Refine: find closest in full set near the approximate closest
            approx_idx = np.unravel_index(np.argmin(dists), dists.shape)
            approx_i = sample_i[approx_idx[0]]
            approx_j = sample_j[approx_idx[1]]

            # Search in neighborhood
            dist_to_approx_i = np.sqrt(np.sum((coords_i - approx_i) ** 2, axis=1))
            near_i = coords_i[dist_to_approx_i < search_radius]

            dist_to_approx_j = np.sqrt(np.sum((coords_j - approx_j) ** 2, axis=1))
            near_j = coords_j[dist_to_approx_j < search_radius]

            if len(near_i) == 0 or len(near_j) == 0:
                continue

            # Limit to avoid memory explosion with large refinement sets
            max_refine = 1000
            if len(near_i) > max_refine:
                sort_idx = np.argsort(dist_to_approx_i[dist_to_approx_i < search_radius])[:max_refine]
                near_i = near_i[sort_idx]
            if len(near_j) > max_refine:
                sort_idx = np.argsort(dist_to_approx_j[dist_to_approx_j < search_radius])[:max_refine]
                near_j = near_j[sort_idx]

            dists = cdist(near_i, near_j, metric='euclidean')

            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            closest_i = tuple(near_i[min_idx[0]])
            closest_j = tuple(near_j[min_idx[1]])
            min_dist = dists[min_idx]
        else:
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            closest_i = tuple(coords_i[min_idx[0]])
            closest_j = tuple(coords_j[min_idx[1]])

        # Check if connection is direct (doesn't cross other components)
        # Instead of creating a full mask copy, check crossing inline
        def crosses_other_components(p1, p2, skip_i, skip_j):
            """Check if line crosses any component other than skip_i and skip_j."""
            r1, c1 = p1
            r2, c2 = p2
            n_steps = max(abs(r2 - r1), abs(c2 - c1), 1)
            for step in range(1, n_steps):
                t = step / n_steps
                r = int(round(r1 + t * (r2 - r1)))
                c = int(round(c1 + t * (c2 - c1)))
                if 0 <= r < all_comps_mask.shape[0] and 0 <= c < all_comps_mask.shape[1]:
                    if all_comps_mask[r, c] and not components[skip_i][r, c] and not components[skip_j][r, c]:
                        return True
            return False

        if crosses_other_components(closest_i, closest_j, i, j):
            # Connection crosses another component - skip it
            continue

        connections.append((i, j, closest_i, closest_j, min_dist))

    return connections


def estimate_component_offset(unwrapped, comp_mask_i, comp_mask_j, closest_i, closest_j, n_neighbors=5):
    """
    Estimate the integer 2π offset between two components.

    Uses N pixels closest to the connection point on each side,
    which includes interior pixels (less noisy than border-only).
    Uses median for robustness to outliers.

    Parameters
    ----------
    unwrapped : np.ndarray
        2D array of unwrapped phase values.
    comp_mask_i, comp_mask_j : np.ndarray
        Boolean masks for the two components.
    closest_i, closest_j : tuple
        (row, col) of the closest pixels defining the connection.
    n_neighbors : int, optional
        Number of pixels to use on each side. Default is 5.

    Returns
    -------
    tuple
        (k_offset, confidence) where:
        - k_offset: integer number of 2π cycles to add to component j
        - confidence: measure of how reliable the estimate is (0-1)
    """
    # Get coordinates of pixels in each component
    rows_i, cols_i = np.where(comp_mask_i)
    rows_j, cols_j = np.where(comp_mask_j)

    # Find N closest pixels to the connection point on each side
    dist_i = np.sqrt((rows_i - closest_i[0])**2 + (cols_i - closest_i[1])**2)
    dist_j = np.sqrt((rows_j - closest_j[0])**2 + (cols_j - closest_j[1])**2)

    # Get indices of N closest pixels
    n_i = min(n_neighbors, len(dist_i))
    n_j = min(n_neighbors, len(dist_j))

    if n_i < 3 or n_j < 3:
        return 0, 0.0

    idx_i = np.argpartition(dist_i, n_i - 1)[:n_i]
    idx_j = np.argpartition(dist_j, n_j - 1)[:n_j]

    # Get phase values at these pixels
    phase_i = unwrapped[rows_i[idx_i], cols_i[idx_i]]
    phase_j = unwrapped[rows_j[idx_j], cols_j[idx_j]]

    # Filter out NaN values
    valid_i = ~np.isnan(phase_i)
    valid_j = ~np.isnan(phase_j)

    if np.sum(valid_i) < 3 or np.sum(valid_j) < 3:
        return 0, 0.0

    # Use median for robustness to outliers
    median_phase_i = np.median(phase_i[valid_i])
    median_phase_j = np.median(phase_j[valid_j])

    # Phase difference and integer offset
    delta_phase = median_phase_i - median_phase_j
    k_offset = int(np.round(delta_phase / (2 * np.pi)))

    # Confidence based on:
    # 1. How close the fractional part is to an integer
    # 2. Standard deviation of the phase values (lower = more confident)
    fractional = (delta_phase / (2 * np.pi)) - k_offset
    frac_confidence = 1.0 - 2 * abs(fractional)  # 1.0 if exactly integer, 0.0 if halfway

    # Check consistency: std of phase values should be small relative to 2π
    std_i = np.std(phase_i[valid_i])
    std_j = np.std(phase_j[valid_j])
    std_confidence = max(0, 1.0 - (std_i + std_j) / (2 * np.pi))

    confidence = frac_confidence * std_confidence

    return k_offset, confidence


def connect_components_ilp(unwrapped, components, connections, n_neighbors=5, max_time=60.0, debug=False):
    """
    Connect separately-unwrapped components using ILP optimization.

    Finds optimal integer 2π offsets for each component to minimize
    phase discontinuities at connection points.

    Parameters
    ----------
    unwrapped : np.ndarray
        2D array with separately unwrapped components.
    components : list of np.ndarray
        List of boolean masks for each component.
    connections : list of tuple
        Output from find_component_connections.
    n_neighbors : int, optional
        Number of pixels to use for offset estimation at each connection.
        Default is 50.
    max_time : float, optional
        Maximum solver time in seconds. Default is 60.
    debug : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    np.ndarray
        2D array with connected unwrapped phase.
    """
    from ortools.sat.python import cp_model

    n_comps = len(components)
    if n_comps < 2 or len(connections) == 0:
        return unwrapped.copy()

    # Estimate offsets and weights for each connection
    edge_data = []
    for comp_i, comp_j, closest_i, closest_j, distance in connections:
        k_offset, confidence = estimate_component_offset(
            unwrapped, components[comp_i], components[comp_j],
            closest_i, closest_j, n_neighbors=n_neighbors
        )
        # Weight by confidence and inverse distance
        weight = confidence / (distance + 1.0)
        edge_data.append((comp_i, comp_j, k_offset, weight))

        if debug:
            print(f'  Connection {comp_i}-{comp_j}: dist={distance:.1f}, '
                  f'k_offset={k_offset}, confidence={confidence:.3f}')

    # Build ILP model
    model = cp_model.CpModel()

    # Variables: k_i = integer offset for component i
    # Range: reasonable bounds (±100 cycles should be enough)
    k_vars = [model.NewIntVar(-100, 100, f'k_{i}') for i in range(n_comps)]

    # Fix component 0 as reference
    model.Add(k_vars[0] == 0)

    # Objective: minimize weighted sum of |measured_offset - (k_i - k_j)|
    # We use absolute value linearization: |x| = max(x, -x)
    scale = 1000  # Scale weights to integers for CP-SAT
    abs_vars = []

    for idx, (comp_i, comp_j, k_offset, weight) in enumerate(edge_data):
        # k_offset = round((phase_i - phase_j) / 2π)
        # To align: phase_j + k_offset*2π ≈ phase_i
        # After offsets: (phase_i + k_i*2π) ≈ (phase_j + k_j*2π)
        # So: phase_i - phase_j ≈ (k_j - k_i)*2π
        # Thus: k_offset ≈ k_j - k_i
        # Minimize: |k_offset - (k_j - k_i)| = |k_offset - k_j + k_i|

        # Create auxiliary variable for the difference
        diff_var = model.NewIntVar(-200, 200, f'diff_{idx}')
        model.Add(diff_var == k_offset - k_vars[comp_j] + k_vars[comp_i])

        # Absolute value
        abs_var = model.NewIntVar(0, 200, f'abs_{idx}')
        model.AddAbsEquality(abs_var, diff_var)

        abs_vars.append((abs_var, int(weight * scale)))

    # Objective: minimize weighted sum of absolute differences
    model.Minimize(sum(w * v for v, w in abs_vars))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if debug:
            print(f'  ILP solver failed with status {status}')
        return unwrapped.copy()

    # Extract solution
    k_offsets = [solver.Value(k_vars[i]) for i in range(n_comps)]

    if debug:
        print(f'  ILP solution: k_offsets = {k_offsets}')

    # Apply offsets to create connected result
    result = unwrapped.copy()
    for i, comp_mask in enumerate(components):
        if k_offsets[i] != 0:
            result[comp_mask] += k_offsets[i] * 2 * np.pi

    return result


def conncomp_2d(phase):
    """
    Compute connected components for a 2D phase array.

    Parameters
    ----------
    phase : np.ndarray
        2D array of phase values (NaN indicates invalid pixels).

    Returns
    -------
    np.ndarray
        2D array of connected component labels (0 for invalid pixels).
    """
    from scipy.ndimage import label

    valid_mask = ~np.isnan(phase)
    labeled_array, num_features = label(valid_mask)
    return labeled_array.astype(np.int32)


def wrap(phase):
    """Wrap phase to [-π, π]."""
    return np.arctan2(np.sin(phase), np.cos(phase))


def wrapped_gradient(phase):
    """Compute wrapped phase gradients (handling phase wrapping)."""
    dx = np.zeros_like(phase)
    dx[:, :-1] = phase[:, 1:] - phase[:, :-1]
    dx = wrap(dx)

    dy = np.zeros_like(phase)
    dy[:-1, :] = phase[1:, :] - phase[:-1, :]
    dy = wrap(dy)

    return dx, dy


def irls_unwrap_2d(phase, weight=None, device='auto', max_iter=50, tol=1e-3,
                   cg_max_iter=20, cg_tol=1e-4, epsilon=1e-2, debug=False):
    """
    Unwrap 2D phase using GPU/TPU-accelerated Iteratively Reweighted Least Squares (L¹ norm).

    This algorithm solves the L¹ phase unwrapping problem:
        min Σ w_ij |∇φ_ij - wrap(∇ψ_ij)|

    by iteratively solving weighted L² problems using preconditioned
    conjugate gradient. Accelerated using PyTorch (TPU via XLA, CUDA on NVIDIA/AMD,
    MPS on Apple Silicon, or CPU fallback).

    Parameters
    ----------
    phase : np.ndarray
        2D array of wrapped phase values in radians.
    weight : np.ndarray, optional
        2D array of quality weights (e.g., correlation). Higher values
        indicate more reliable phase. If None, uniform weights are used.
    device : str or torch.device, optional
        PyTorch device: 'auto' (default), 'cuda', 'mps', 'cpu', or 'tpu'.
        'auto' uses GPU if Dask client has resources={'gpu': 1}.
    max_iter : int, optional
        Maximum IRLS iterations. Default is 50.
    tol : float, optional
        Convergence tolerance for relative change in solution. Default is 1e-3.
    cg_max_iter : int, optional
        Maximum conjugate gradient iterations per IRLS step. Default is 20.
    cg_tol : float, optional
        Conjugate gradient convergence tolerance. Default is 1e-4.
    epsilon : float, optional
        Smoothing parameter for L¹ approximation. Default is 1e-2.
    debug : bool, optional
        If True, print convergence information. Default is False.

    Returns
    -------
    np.ndarray
        2D array of unwrapped phase values.

    Notes
    -----
    The algorithm uses GPU-accelerated DCT as initial solution (same as
    standalone DCT method), then refines it through weighted iterations.
    The DCT initialization provides a good starting point, and the IRLS
    iterations correct for residue-induced errors using correlation weights.

    Based on: Dubois-Taine et al., "Iteratively Reweighted Least Squares
    for Phase Unwrapping", arXiv:2401.09961 (2024).

    Achieves 10-20x speedup over SNAPHU on GPU/TPU.

    Default parameters ensure robust convergence for InSAR processing:
    - max_iter=50, cg_max_iter=20: Ensures full convergence, avoiding
      phase discontinuities from early termination.
    - tol=1e-3, cg_tol=1e-4: Tight tolerances for accurate results.
    - epsilon=1e-2: Balances L¹ approximation quality with stability.

    For faster processing on clean data (may have artifacts on noisy data):
    >>> irls_unwrap_2d(phase, weight, max_iter=20, tol=1e-2)
    """
    import torch
    from torch_dct import dct_2d, idct_2d
    import time
    from .BatchCore import BatchCore

    # Validate and set device using shared helper
    if isinstance(device, str):
        if device == 'tpu':
            # TPU uses torch_xla device
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        else:
            device = BatchCore._get_torch_device(device, debug=debug)

    device_name = str(device)
    height, width = phase.shape

    # Use float32 for all devices - sufficient for phase unwrapping
    dtype = torch.float32
    np_dtype = np.float32

    _t_start = time.time()

    # Handle NaN values
    nan_mask = np.isnan(phase)
    if np.all(nan_mask):
        return np.full_like(phase, np.nan)

    # Create valid mask for computation
    valid_mask = ~nan_mask

    # Fill NaN with 0 for computation (avoid np.where temp array)
    phase_filled = phase.copy()
    np.copyto(phase_filled, 0.0, where=nan_mask)

    # Prepare weights (avoid np.where temp arrays)
    if weight is not None:
        # Handle NaN in weights (both from phase nan_mask AND from weight itself)
        weight_filled = weight.copy()
        np.copyto(weight_filled, 0.0, where=np.isnan(weight))
        np.copyto(weight_filled, 0.0, where=nan_mask)
        np.clip(weight_filled, 0.01, 1.0, out=weight_filled)
    else:
        # 1 where valid, 0 where NaN (avoid ~nan_mask temp bool array)
        weight_filled = np.subtract(1.0, nan_mask, dtype=np.float32)

    # Save input circular mean for later restoration (unwrapping removes DC component)
    # Use circular mean for wrapped phase: atan2(mean(sin), mean(cos))
    if valid_mask.any():
        phase_valid = phase[valid_mask]
        input_mean = np.arctan2(np.mean(np.sin(phase_valid)), np.mean(np.cos(phase_valid)))
    else:
        input_mean = 0.0

    # Convert to torch tensors
    phi = torch.from_numpy(phase_filled.astype(np_dtype)).to(device)
    w = torch.from_numpy(weight_filled.astype(np_dtype)).to(device)
    valid = torch.from_numpy(valid_mask).to(device)

    # Compute wrapped phase differences (target gradients)
    dx_target = torch.zeros_like(phi)
    dy_target = torch.zeros_like(phi)
    dx_target[:, :-1] = phi[:, 1:] - phi[:, :-1]
    dy_target[:-1, :] = phi[1:, :] - phi[:-1, :]

    # Wrap to [-π, π]
    dx_target = torch.atan2(torch.sin(dx_target), torch.cos(dx_target))
    dy_target = torch.atan2(torch.sin(dy_target), torch.cos(dy_target))

    # Edge weights (average of adjacent pixel weights)
    wx = torch.zeros_like(w)
    wy = torch.zeros_like(w)
    wx[:, :-1] = (w[:, :-1] + w[:, 1:]) / 2
    wy[:-1, :] = (w[:-1, :] + w[1:, :]) / 2

    # Zero out weights at invalid edges
    wx[:, :-1] *= (valid[:, :-1] & valid[:, 1:]).to(dtype)
    wy[:-1, :] *= (valid[:-1, :] & valid[1:, :]).to(dtype)

    # Precompute DCT eigenvalues for preconditioner
    # IMPORTANT: Trigonometric functions require float64 precision!
    # Small errors in cos() accumulate through thousands of preconditioner
    # applications in the CG solver. Using float32 on GPU for eigenvalues
    # causes residuals to degrade from ~0.4 rad to ~1.4 rad at 6400x6400.
    # Compute on CPU with float64, then convert to float32 and transfer.
    i_idx = torch.arange(height, dtype=torch.float64)
    j_idx = torch.arange(width, dtype=torch.float64)
    cos_i = torch.cos(torch.pi * i_idx / height)
    cos_j = torch.cos(torch.pi * j_idx / width)
    eigenvalues = (2 * cos_i.unsqueeze(1) + 2 * cos_j.unsqueeze(0) - 4).to(dtype).to(device)
    eigenvalues[0, 0] = 1.0  # Avoid division by zero

    # Pre-allocate buffers for gradient computation (avoid repeated allocation)
    _dx_buf = torch.zeros_like(phi)
    _dy_buf = torch.zeros_like(phi)
    _result_buf = torch.zeros_like(phi)

    def apply_laplacian(x, wx_irls, wy_irls):
        """Apply weighted Laplacian operator: -∇·(w·∇x)"""
        # Forward differences (reuse buffers)
        _dx_buf.zero_()
        _dy_buf.zero_()
        _dx_buf[:, :-1] = x[:, 1:] - x[:, :-1]
        _dy_buf[:-1, :] = x[1:, :] - x[:-1, :]

        # Weight the gradients in-place
        _dx_buf.mul_(wx_irls)
        _dy_buf.mul_(wy_irls)

        # Backward differences (divergence)
        _result_buf.zero_()
        _result_buf[:, 1:].sub_(_dx_buf[:, :-1])
        _result_buf[:, :-1].add_(_dx_buf[:, :-1])
        _result_buf[1:, :].sub_(_dy_buf[:-1, :])
        _result_buf[:-1, :].add_(_dy_buf[:-1, :])

        return _result_buf.neg()

    def apply_preconditioner(r):
        """Apply DCT-based preconditioner (approximate inverse Laplacian)"""
        r_dct = dct_2d(r)
        r_dct.div_(-eigenvalues + 1e-10)
        r_dct[0, 0] = 0.0
        return idct_2d(r_dct)

    def conjugate_gradient(b, wx_irls, wy_irls, x0, max_iter_cg, tol_cg):
        """Preconditioned conjugate gradient solver for IRLS."""
        x = x0  # No clone - modify in place, caller provides buffer
        r = b - apply_laplacian(x, wx_irls, wy_irls)

        # Check for NaN in initial residual
        if not torch.isfinite(r).all():
            return x0

        z = apply_preconditioner(r)
        p = z.clone()  # Need clone here - p gets modified
        rz = torch.sum(r * z)

        for i in range(max_iter_cg):
            Ap = apply_laplacian(p, wx_irls, wy_irls).clone()  # Need clone - buffer reused
            pAp = torch.sum(p * Ap)
            if pAp.abs() < 1e-15 or not torch.isfinite(pAp):
                break
            alpha = rz / pAp

            # Clamp alpha to prevent explosion
            alpha = torch.clamp(alpha, -1e6, 1e6)
            alpha_val = alpha.item()

            # Update x
            x.add_(p, alpha=alpha_val)

            r.sub_(Ap, alpha=alpha_val)

            # Check for numerical issues mid-iteration
            if not torch.isfinite(x).all():
                break

            r_norm = torch.sqrt(torch.sum(r * r))
            if r_norm < tol_cg or not torch.isfinite(r_norm):
                break

            z = apply_preconditioner(r)
            rz_new = torch.sum(r * z)
            if rz.abs() < 1e-15:
                break
            beta = rz_new / rz
            beta = torch.clamp(beta, -1e6, 1e6)
            p.mul_(beta.item()).add_(z)
            rz = rz_new

        return x

    # Initialize with DCT solution (L² result)
    rho = torch.zeros_like(phi)
    rho[:, 1:] += wx[:, :-1] * dx_target[:, :-1]
    rho[:, :-1] -= wx[:, :-1] * dx_target[:, :-1]
    rho[1:, :] += wy[:-1, :] * dy_target[:-1, :]
    rho[:-1, :] -= wy[:-1, :] * dy_target[:-1, :]

    rho_dct = dct_2d(rho)
    u_dct = rho_dct / (-eigenvalues + 1e-10)
    u_dct[0, 0] = 0.0
    u = idct_2d(u_dct)

    # Check DCT initialization for NaN/inf
    if not torch.isfinite(u).all():
        if debug:
            nan_count = (~torch.isfinite(u)).sum().item()
            print(f'  DCT init produced {nan_count} NaN/inf values, filling with zeros')
        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))

    # Re-center DCT result to improve float32 precision
    # This keeps values near zero where float32 has best precision
    if valid.any():
        u_mean = u[valid].mean()
        u.sub_(u_mean)

    _t_init = time.time()

    if debug:
        # Handle case where weights might still have issues
        w_valid = w[valid]
        if w_valid.numel() > 0:
            w_min, w_max = w_valid.min().item(), w_valid.max().item()
        else:
            w_min, w_max = 0.0, 0.0
        print(f'  Input: {height}x{width}, valid pixels: {valid_mask.sum().item()}, '
              f'weight range: [{w_min:.3f}, {w_max:.3f}]')
        print(f'  DCT init range: [{u.min().item():.2f}, {u.max().item():.2f}]')

    # IRLS iterations - keep track of last good solution
    u_best = u.clone()
    best_residual = float('inf')

    # Pre-allocate buffers for IRLS loop
    dx_u = torch.zeros_like(u)
    dy_u = torch.zeros_like(u)
    rx = torch.zeros_like(u)
    ry = torch.zeros_like(u)
    wx_irls = torch.zeros_like(u)
    wy_irls = torch.zeros_like(u)
    b = torch.zeros_like(u)
    u_prev = torch.zeros_like(u)
    eps_sq = epsilon * epsilon

    for iteration in range(max_iter):
        u_prev.copy_(u)

        # Re-center u to prevent numerical drift (important for float32)
        # Phase unwrapping only cares about gradients, so mean is arbitrary
        if valid.any():
            u_mean = u[valid].mean()
            u.sub_(u_mean)

        # Compute current gradients (reuse buffers)
        dx_u.zero_()
        dy_u.zero_()
        dx_u[:, :-1] = u[:, 1:] - u[:, :-1]
        dy_u[:-1, :] = u[1:, :] - u[:-1, :]

        # Compute residuals in-place
        torch.sub(dx_u, dx_target, out=rx)
        torch.sub(dy_u, dy_target, out=ry)

        # Track best solution by residual magnitude
        current_residual = (torch.sum(rx * rx) + torch.sum(ry * ry)).item()
        if current_residual < best_residual and torch.isfinite(u).all():
            best_residual = current_residual
            u_best.copy_(u)

        # Update IRLS weights: w_irls = w / sqrt(r² + ε²)
        # In-place operations
        torch.addcmul(torch.full_like(rx, eps_sq), rx, rx, out=wx_irls)
        wx_irls.sqrt_()
        torch.div(wx, wx_irls, out=wx_irls)

        torch.addcmul(torch.full_like(ry, eps_sq), ry, ry, out=wy_irls)
        wy_irls.sqrt_()
        torch.div(wy, wy_irls, out=wy_irls)

        # Clamp weights in-place
        wx_irls.clamp_(min=1e-6, max=1e6)
        wy_irls.clamp_(min=1e-6, max=1e6)

        # Compute right-hand side: -∇·(w_irls · ∇target)
        b.zero_()
        b[:, 1:].addcmul_(wx_irls[:, :-1], dx_target[:, :-1])
        b[:, :-1].addcmul_(wx_irls[:, :-1], dx_target[:, :-1], value=-1)
        b[1:, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :])
        b[:-1, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :], value=-1)

        # Solve weighted Laplacian system using CG
        u = conjugate_gradient(b, wx_irls, wy_irls, u, cg_max_iter, cg_tol)

        # Check for numerical issues during iteration
        if not torch.isfinite(u).all():
            if debug:
                print(f'  IRLS iter {iteration}: NaN/inf detected, reverting to best solution')
            u = u_best.clone()
            break

        # Check convergence (in-place computation)
        diff = torch.norm(u - u_prev)
        norm_u = torch.norm(u) + 1e-10
        rel_change = (diff / norm_u).item()

        if debug and iteration % 5 == 0:
            print(f'  IRLS iter {iteration}: rel_change = {rel_change:.2e}, residual = {current_residual:.2e}')

        if rel_change < tol:
            if debug:
                print(f'  IRLS converged at iteration {iteration}')
            break

    # Use best solution found during iterations if current is invalid
    if not torch.isfinite(u).all():
        if debug:
            print(f'  Final solution has NaN/inf, reverting to best solution')
        u = u_best

    _t_end = time.time()

    # Convert back to numpy
    unwrapped = u.cpu().numpy().astype(np.float32)

    # Check for NaN/inf from numerical issues - return NaN array (no hidden fallback)
    if not np.isfinite(unwrapped[~nan_mask]).all():
        import warnings
        nan_count = np.sum(~np.isfinite(unwrapped[~nan_mask]))
        total_valid = np.sum(~nan_mask)
        warnings.warn(f'IRLS produced {nan_count}/{total_valid} NaN/inf values - returning NaN for this component', RuntimeWarning)
        return np.full_like(phase, np.nan, dtype=np.float32)

    # Restore NaN values
    unwrapped[nan_mask] = np.nan

    # Validate and correct: rewrapped phase should match input
    valid_mask = ~nan_mask
    if np.any(valid_mask):
        diff = unwrapped[valid_mask] - phase[valid_mask]
        k_values = np.round(diff / (2 * np.pi))
        k_median = np.median(k_values)
        unwrapped[valid_mask] = unwrapped[valid_mask] - k_median * 2 * np.pi

    # Restore input circular mean (unwrapping removes DC component)
    # Wrap the output to compute its circular mean, then adjust
    if np.any(valid_mask):
        unwrapped_valid = unwrapped[valid_mask]
        # Wrap to [-π, π] for circular mean computation
        wrapped_output = np.mod(unwrapped_valid + np.pi, 2 * np.pi) - np.pi
        output_mean = np.arctan2(np.mean(np.sin(wrapped_output)), np.mean(np.cos(wrapped_output)))
        # Adjust by the difference in circular means
        unwrapped[valid_mask] = unwrapped_valid + (input_mean - output_mean)

    if debug:
        print(f'TIMING irls_unwrap_2d ({height}x{width}) on {device_name}:')
        print(f'  init (DCT):   {(_t_init - _t_start)*1000:.1f} ms')
        print(f'  IRLS iters:   {(_t_end - _t_init)*1000:.1f} ms ({iteration+1} iterations)')
        print(f'  TOTAL:        {(_t_end - _t_start)*1000:.1f} ms')

    return unwrapped


def detect_discontinuity_hough_focal(phase, grad_threshold=2.0, mask_width=3, debug=False):
    """
    Detect discontinuity using Hough transform with focal point (tip) detection.

    Finds the fault direction via Hough transform, then locates the tip
    by finding where gradient magnitude drops along the line.

    Parameters
    ----------
    phase : np.ndarray
        2D wrapped phase array.
    grad_threshold : float
        Gradient magnitude threshold for edge detection (radians).
        Default 2.0 rad ≈ 0.64π.
    mask_width : int
        Half-width of the mask along the detected line. Default 3.
    debug : bool
        Print diagnostic information.

    Returns
    -------
    mask : np.ndarray
        Boolean mask, True = discontinuity pixels to be masked as NaN.
    info : dict
        Detection info: focal_point, angle, n_masked.
    """
    import cv2

    height, width = phase.shape

    # Compute gradient magnitude
    dx, dy = wrapped_gradient(phase)
    grad_mag = np.sqrt(dx**2 + dy**2)

    # Create binary edge image
    edges = (grad_mag > grad_threshold).astype(np.uint8) * 255
    nan_mask = np.isnan(phase)
    edges[nan_mask] = 0

    # Standard Hough transform to find dominant line direction
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    if lines is None or len(lines) == 0:
        if debug:
            print('  No lines detected by Hough transform')
        return np.zeros((height, width), dtype=bool), {'focal_point': None, 'angle': None, 'n_masked': 0}

    # Get dominant line parameters
    rho, theta = lines[0][0]
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if debug:
        print(f'  Hough detected line: rho={rho:.1f}, theta={np.rad2deg(theta):.1f}°')

    # Sample points along the detected line
    if abs(sin_t) > abs(cos_t):
        x_samples = np.arange(0, width, 5)
        y_samples = ((rho - x_samples * cos_t) / sin_t).astype(int)
        valid = (y_samples >= 0) & (y_samples < height)
        x_samples, y_samples = x_samples[valid], y_samples[valid]
    else:
        y_samples = np.arange(0, height, 5)
        x_samples = ((rho - y_samples * sin_t) / cos_t).astype(int)
        valid = (x_samples >= 0) & (x_samples < width)
        x_samples, y_samples = x_samples[valid], y_samples[valid]

    if len(x_samples) == 0:
        return np.zeros((height, width), dtype=bool), {'focal_point': None, 'angle': None, 'n_masked': 0}

    # Compute gradient magnitude along the line
    grad_along_line = []
    for x, y in zip(x_samples, y_samples):
        y_min, y_max = max(0, y-5), min(height, y+5)
        x_min, x_max = max(0, x-5), min(width, x+5)
        window_grad = grad_mag[y_min:y_max, x_min:x_max]
        grad_along_line.append(np.nanmean(window_grad))

    grad_along_line = np.array(grad_along_line)

    # Find segment of line with high gradient (the fault)
    threshold = grad_threshold * 0.5
    high_grad = grad_along_line > threshold

    if not np.any(high_grad):
        return np.zeros((height, width), dtype=bool), {'focal_point': None, 'angle': None, 'n_masked': 0}

    # Find the extent of high-gradient region (the fault segment)
    high_indices = np.where(high_grad)[0]
    start_idx = high_indices[0]
    end_idx = high_indices[-1]

    # Get start and end points of the fault segment
    start_y, start_x = int(y_samples[start_idx]), int(x_samples[start_idx])
    end_y, end_x = int(y_samples[end_idx]), int(x_samples[end_idx])

    # Apply edge margin constraint - don't let mask touch image boundaries
    # This prevents disconnecting the two sides of the fault
    edge_margin = int(min(height, width) * 0.05)  # 5% margin
    if start_x < edge_margin or start_y < edge_margin or \
       start_x >= width - edge_margin or start_y >= height - edge_margin:
        # Find first point that's inside the margin
        for idx in high_indices:
            y, x = int(y_samples[idx]), int(x_samples[idx])
            if edge_margin <= x < width - edge_margin and edge_margin <= y < height - edge_margin:
                start_idx = idx
                start_y, start_x = y, x
                break

    if end_x < edge_margin or end_y < edge_margin or \
       end_x >= width - edge_margin or end_y >= height - edge_margin:
        # Find last point that's inside the margin
        for idx in reversed(high_indices):
            y, x = int(y_samples[idx]), int(x_samples[idx])
            if edge_margin <= x < width - edge_margin and edge_margin <= y < height - edge_margin:
                end_idx = idx
                end_y, end_x = y, x
                break

    if debug:
        print(f'  Fault segment: ({start_y}, {start_x}) to ({end_y}, {end_x})')

    # Create mask along the entire fault segment
    mask = np.zeros((height, width), dtype=bool)

    # Mask all points along the line segment from start to end
    n_points = max(abs(end_x - start_x), abs(end_y - start_y)) * 2
    if n_points > 0:
        for i in range(n_points + 1):
            t = i / n_points
            py = int(start_y + t * (end_y - start_y))
            px = int(start_x + t * (end_x - start_x))

            if 0 <= py < height and 0 <= px < width:
                for ddy in range(-mask_width, mask_width + 1):
                    for ddx in range(-mask_width, mask_width + 1):
                        yy, xx = py + ddy, px + ddx
                        if 0 <= yy < height and 0 <= xx < width:
                            mask[yy, xx] = True

    n_masked = np.sum(mask)
    if debug:
        print(f'  Mask: {n_masked} pixels')

    # Check if mask splits image into disconnected regions
    from scipy import ndimage
    valid_region = ~mask
    labeled, n_components = ndimage.label(valid_region)
    splits_image = n_components > 1

    if debug and splits_image:
        print(f'  WARNING: Mask splits image into {n_components} disconnected regions!')

    info = {
        'start_point': (start_y, start_x),
        'end_point': (end_y, end_x),
        'angle': np.rad2deg(theta),
        'n_masked': n_masked,
        'splits_image': splits_image
    }

    return mask, info
