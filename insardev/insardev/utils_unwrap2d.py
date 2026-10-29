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
# Snaphu for comparison (dev-time only; the library has no snaphu function, BUGS N158).
# To unwrap a 2-D array with snaphu eagerly, write the wrapped phase to a binary file,
# run the snaphu binary on it by subprocess with the snaphu command-line options,
# and read the unwrapped output back into a NumPy array. Nothing is lazy in this path.
import math
import numpy as np
import numba as nb
import rocket_fft  # noqa: F401  (the scipy.fft overloads inside numba)
import scipy.fft

# 4-connectivity structure for scipy.ndimage.label (no diagonals)
STRUCTURE_4CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@nb.njit(parallel=False, cache=True)
def _accum_sincos_count(labeled_flat, phase_flat, unwrapped_flat, n_labels):
    """
    Single-pass accumulation of sin/cos sums and pixel counts per component.

    ~6x faster than separate np.bincount calls because:
    - Single pass over data (better cache locality)
    - sin/cos computed inline (no intermediate arrays)
    - Counts computed simultaneously (replaces cv2.connectedComponentsWithStats)

    Parameters
    ----------
    labeled_flat : np.ndarray (int32)
        Flattened label array from cv2.connectedComponents.
        Label 0 = invalid pixels, labels 1..n_labels-1 = valid components.
    phase_flat : np.ndarray (float32)
        Flattened input (wrapped) phase array.
    unwrapped_flat : np.ndarray (float32)
        Flattened unwrapped phase array.
    n_labels : int
        Number of labels including background (0).

    Returns
    -------
    sin_in, cos_in, sin_out, cos_out, counts : tuple of np.ndarray
        Accumulated sin/cos sums (float64) and pixel counts (int64) per label.
    """
    sin_in = np.zeros(n_labels, np.float64)
    cos_in = np.zeros(n_labels, np.float64)
    sin_out = np.zeros(n_labels, np.float64)
    cos_out = np.zeros(n_labels, np.float64)
    counts = np.zeros(n_labels, np.int64)
    for i in range(labeled_flat.size):
        k = labeled_flat[i]
        if k > 0:  # Skip invalid pixels (label 0)
            p_in = phase_flat[i]
            p_out = unwrapped_flat[i]
            sin_in[k] += np.sin(p_in)
            cos_in[k] += np.cos(p_in)
            sin_out[k] += np.sin(p_out)
            cos_out[k] += np.cos(p_out)
            counts[k] += 1
    return sin_in, cos_in, sin_out, cos_out, counts


@nb.njit(parallel=False, cache=True)
def _apply_adjustment(unwrapped_flat, labeled_flat, adjustment):
    """Apply per-component adjustment to unwrapped phase (in-place)."""
    for i in range(labeled_flat.size):
        k = labeled_flat[i]
        if k > 0:
            unwrapped_flat[i] += adjustment[k]


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








def find_component_connections_fast(labeled_array, phase, components,
                                     conncomp_gap=None, max_neighbors=30, n_neighbors=5):
    """
    Find connections between components using vectorized boundaries and STRtree.

    Uses OpenCV contours to vectorize components, extracts boundaries (not polygons),
    and uses Shapely STRtree for efficient spatial queries.

    Parameters
    ----------
    labeled_array : np.ndarray
        2D int32 array with component labels (0=invalid, 1+=components).
    phase : np.ndarray
        2D float32 array of unwrapped phase.
    components : list of dict
        Component info with 'label', 'size', 'slices' (from get_connected_components).
    conncomp_gap : int or None
        Maximum gap in pixels. None = unlimited.
    max_neighbors : int
        Maximum connections per component.
    n_neighbors : int
        Window size for phase offset estimation (n_neighbors x n_neighbors).

    Returns
    -------
    list of tuple
        (comp_i, comp_j, closest_i, closest_j, distance, k_offset, confidence)
    """
    import cv2
    from shapely.geometry import Polygon, LinearRing, LineString, box
    from shapely import make_valid
    from shapely.strtree import STRtree
    from shapely.ops import nearest_points

    n_comps = len(components)
    if n_comps < 2:
        return []

    shape = labeled_array.shape

    # Step 1: Vectorize components and extract BOUNDARIES (not polygons)
    boundary_list = []  # List of boundary geometries (LinearRing or MultiLineString)
    boundary_to_comp = []  # Map boundary index to component index

    for comp_idx, comp in enumerate(components):
        sl = comp['slices']
        label = comp['label']

        sub_labeled = labeled_array[sl]
        comp_mask = (sub_labeled == label).astype(np.uint8)

        # Get ALL contours (external and holes)
        contours, hierarchy = cv2.findContours(comp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 3:
                pts = contour.squeeze()
                if pts.ndim == 2 and len(pts) >= 3:
                    pts = pts + np.array([sl[1].start, sl[0].start])
                    try:
                        poly = Polygon(pts)
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        # Extract boundaries from valid polygon/multipolygon
                        if poly.geom_type == 'Polygon':
                            if poly.boundary.length > 0:
                                boundary_list.append(poly.boundary)
                                boundary_to_comp.append(comp_idx)
                        elif poly.geom_type == 'MultiPolygon':
                            for g in poly.geoms:
                                if g.boundary.length > 0:
                                    boundary_list.append(g.boundary)
                                    boundary_to_comp.append(comp_idx)
                        elif poly.geom_type == 'GeometryCollection':
                            for g in poly.geoms:
                                if hasattr(g, 'boundary') and g.boundary.length > 0:
                                    boundary_list.append(g.boundary)
                                    boundary_to_comp.append(comp_idx)
                    except Exception:
                        pass

    n_boundaries = len(boundary_list)
    if n_boundaries < 2:
        return []

    # Step 2: Build STRtree of boundaries
    strtree = STRtree(boundary_list)
    centroids = [b.centroid for b in boundary_list]

    # Step 3: Build polygon-level STRtree for efficient component neighbor search
    # Group boundaries by component and build per-component structures
    comp_polygons = []  # One polygon per component (union of all boundaries)
    comp_boundary_trees = []  # STRtree of boundaries for each component
    comp_boundary_lists = []  # List of boundary indices for each component

    for comp_idx in range(n_comps):
        # Get all boundaries for this component
        comp_boundaries = [i for i, c in enumerate(boundary_to_comp) if c == comp_idx]
        comp_boundary_lists.append(comp_boundaries)

        # Build boundary STRtree for this component
        if comp_boundaries:
            comp_geoms = [boundary_list[i] for i in comp_boundaries]
            comp_boundary_trees.append(STRtree(comp_geoms))
        else:
            comp_boundary_trees.append(None)

        # Create component polygon (convex hull of all boundaries for bbox search)
        if comp_boundaries:
            from shapely.ops import unary_union
            all_bounds = unary_union([boundary_list[i] for i in comp_boundaries])
            comp_polygons.append(all_bounds.convex_hull)
        else:
            comp_polygons.append(None)

    # Build polygon-level STRtree (one entry per component)
    valid_comp_indices = [i for i in range(n_comps) if comp_polygons[i] is not None]
    valid_comp_polys = [comp_polygons[i] for i in valid_comp_indices]
    poly_strtree = STRtree(valid_comp_polys)

    # Step 4: For each component, find nearest neighbor components using polygon tree
    candidate_pairs = set()

    for comp_i in range(n_comps):
        if comp_polygons[comp_i] is None:
            continue

        # Query polygon tree for nearest components
        poly_i = comp_polygons[comp_i]
        # Get all components sorted by distance
        all_indices = poly_strtree.query(poly_i.buffer(max(shape[0], shape[1])))
        neighbors_with_dist = []
        for idx in all_indices:
            comp_j = valid_comp_indices[idx]
            if comp_j != comp_i:
                d = poly_i.distance(comp_polygons[comp_j])
                neighbors_with_dist.append((comp_j, d))

        neighbors_with_dist.sort(key=lambda x: x[1])

        # Take max_neighbors nearest components
        for comp_j, _ in neighbors_with_dist[:max_neighbors]:
            # Find nearest boundary pair between comp_i and comp_j
            if not comp_boundary_lists[comp_i] or not comp_boundary_lists[comp_j]:
                continue

            # Use boundary trees to find closest pair
            best_pair = None
            best_dist = float('inf')

            for bi in comp_boundary_lists[comp_i]:
                boundary_i = boundary_list[bi]
                # Query comp_j's boundary tree for nearest
                if comp_boundary_trees[comp_j] is not None:
                    nearest_idx = comp_boundary_trees[comp_j].nearest(boundary_i)
                    bj = comp_boundary_lists[comp_j][nearest_idx]
                    d = boundary_i.distance(boundary_list[bj])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (bi, bj)

            if best_pair:
                candidate_pairs.add((min(best_pair), max(best_pair)))

    # Step 5: Compute nearest_points and filter by polygon intersection
    valid_pairs = []

    for i, j in candidate_pairs:
        try:
            p1, p2 = nearest_points(boundary_list[i], boundary_list[j])
            dist = p1.distance(p2)

            if conncomp_gap is not None and dist > conncomp_gap:
                continue

            line = LineString([p1, p2])

            # Check intersection with other component polygons (not boundaries)
            intersecting = poly_strtree.query(line, predicate='intersects')
            # Filter to components other than i or j's component
            comp_i = boundary_to_comp[i]
            comp_j = boundary_to_comp[j]
            # Convert poly_strtree indices to component indices
            blocking = [valid_comp_indices[k] for k in intersecting
                       if valid_comp_indices[k] != comp_i and valid_comp_indices[k] != comp_j]

            if len(blocking) == 0:
                valid_pairs.append((i, j, dist, p1, p2))
        except Exception:
            pass

    # Sort by distance
    valid_pairs.sort(key=lambda x: x[2])

    # Step 5: Apply max_neighbors limit and compute phase offsets
    connections = []
    conn_count = {i: 0 for i in range(n_comps)}
    half = n_neighbors // 2

    for i, j, dist, p1, p2 in valid_pairs:
        comp_i = boundary_to_comp[i]
        comp_j = boundary_to_comp[j]

        if conn_count[comp_i] >= max_neighbors or conn_count[comp_j] >= max_neighbors:
            continue

        label_i = components[comp_i]['label']
        label_j = components[comp_j]['label']

        # Connection points (shapely x,y = col,row)
        cx1, cy1 = int(round(p1.x)), int(round(p1.y))
        cx2, cy2 = int(round(p2.x)), int(round(p2.y))
        p_i = (cy1, cx1)
        p_j = (cy2, cx2)

        # Estimate phase offset using square windows filtered by component label
        y0, y1 = max(0, cy1-half), min(shape[0], cy1+half+1)
        x0, x1 = max(0, cx1-half), min(shape[1], cx1+half+1)
        window_phase_i = phase[y0:y1, x0:x1]
        window_label_i = labeled_array[y0:y1, x0:x1]
        vals_i = window_phase_i[window_label_i == label_i]

        y0, y1 = max(0, cy2-half), min(shape[0], cy2+half+1)
        x0, x1 = max(0, cx2-half), min(shape[1], cx2+half+1)
        window_phase_j = phase[y0:y1, x0:x1]
        window_label_j = labeled_array[y0:y1, x0:x1]
        vals_j = window_phase_j[window_label_j == label_j]

        if len(vals_i) == 0 or len(vals_j) == 0:
            continue

        mean_i = np.nanmean(vals_i)
        mean_j = np.nanmean(vals_j)

        if np.isnan(mean_i) or np.isnan(mean_j):
            continue

        # Compute phase offset
        delta = mean_i - mean_j
        k_offset = int(np.round(delta / (2 * np.pi)))

        # Confidence based on fractional part
        fractional = (delta / (2 * np.pi)) - k_offset
        confidence = max(0, 1.0 - 2 * abs(fractional))

        connections.append((comp_i, comp_j, p_i, p_j, dist, k_offset, confidence, delta))
        conn_count[comp_i] += 1
        conn_count[comp_j] += 1

    return connections






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




def connect_components_ilp_fast(unwrapped, labeled_array, components, connections, max_time=60.0, debug=False):
    """
    Connect components using ILP optimization with labeled array (memory efficient).

    Parameters
    ----------
    unwrapped : np.ndarray
        2D array with separately unwrapped components.
    labeled_array : np.ndarray
        2D int32 array with component labels (0=invalid, 1+=components).
    components : list of dict
        Component info with 'label', 'size', 'slices' from get_connected_components.
    connections : list of tuple
        Output from find_component_connections_fast:
        (comp_i, comp_j, p_i, p_j, distance, k_offset, confidence)
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

    # Extract edge data from connections (already computed by find_component_connections_fast)
    # Weight by sqrt of component sizes so large components anchor small ones
    edge_data = []
    for conn in connections:
        comp_i, comp_j, p_i, p_j, distance, k_offset, confidence, delta = conn
        # Size weight: sqrt of minimum size ensures small components align to large ones
        size_i = components[comp_i]['size']
        size_j = components[comp_j]['size']
        size_weight = np.sqrt(min(size_i, size_j))
        # Use sqrt(distance) for gentler decay - raw distance is too aggressive
        weight = confidence * size_weight / (np.sqrt(distance) + 1.0)
        edge_data.append((comp_i, comp_j, k_offset, weight))

        if debug:
            print(f'  Connection {comp_i}-{comp_j}: dist={distance:.1f}, '
                  f'delta={delta:.3f}, k_offset={k_offset}, confidence={confidence:.3f}, '
                  f'sizes=({size_i}, {size_j})')

    # Build ILP model
    model = cp_model.CpModel()

    # Variables: k_i = integer offset for component i
    k_vars = [model.NewIntVar(-100, 100, f'k_{i}') for i in range(n_comps)]

    # Fix component 0 as reference (largest component)
    model.Add(k_vars[0] == 0)

    # Objective: minimize weighted sum of |(k_i - k_j) - k_offset|
    # k_offset = round((phase_j - phase_i) / (2π)), so we want k_i - k_j = k_offset
    scale = 1000
    abs_vars = []

    for idx, (comp_i, comp_j, k_offset, weight) in enumerate(edge_data):
        diff_var = model.NewIntVar(-200, 200, f'diff_{idx}')
        # diff = (k_i - k_j) - k_offset, should be 0 for perfect alignment
        model.Add(diff_var == k_vars[comp_i] - k_vars[comp_j] - k_offset)

        abs_var = model.NewIntVar(0, 200, f'abs_{idx}')
        model.AddAbsEquality(abs_var, diff_var)

        abs_vars.append((abs_var, int(weight * scale)))

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

    # Find components reachable from reference (component 0) using BFS
    from collections import deque
    adj = {i: set() for i in range(n_comps)}
    for comp_i, comp_j, k_offset, weight in edge_data:
        adj[comp_i].add(comp_j)
        adj[comp_j].add(comp_i)

    # Debug: check which components have any connections
    if debug:
        connected_comps = set()
        for comp_i, comp_j, k_offset, weight in edge_data:
            connected_comps.add(comp_i)
            connected_comps.add(comp_j)
        no_connections = [i for i in range(n_comps) if i not in connected_comps]
        print(f'  Components with connections: {len(connected_comps)}/{n_comps}')
        if no_connections:
            print(f'  Components with NO connections: {no_connections}')

    reachable = set([0])
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Set k=0 for unreachable components (they have no constraint, solver picks boundary)
    for i in range(n_comps):
        if i not in reachable:
            k_offsets[i] = 0

    if debug:
        n_unreachable = n_comps - len(reachable)
        print(f'  Reachable from comp 0: {len(reachable)}/{n_comps}')
        if n_unreachable > 0:
            unreachable = [i for i in range(n_comps) if i not in reachable]
            print(f'  Unreachable components: {unreachable[:20]}{"..." if len(unreachable) > 20 else ""}')

    # Apply offsets using labeled_array (memory efficient - no boolean masks)
    result = unwrapped.copy()
    n_adjusted = 0
    for i, comp in enumerate(components):
        if k_offsets[i] != 0:
            label = comp['label']
            result[labeled_array == label] += k_offsets[i] * 2 * np.pi
            n_adjusted += 1

    if debug:
        # Summary of k_offsets applied
        nonzero_k = [(i, k_offsets[i]) for i in range(n_comps) if k_offsets[i] != 0]
        print(f'  Applied 2π offsets to {n_adjusted}/{n_comps} components')
        if nonzero_k:
            # Show first 10 offsets
            for i, k in nonzero_k[:10]:
                print(f'    comp {i}: k={k:+d} ({k*2*np.pi:+.2f} rad)')
            if len(nonzero_k) > 10:
                print(f'    ... and {len(nonzero_k)-10} more')
        # Show histogram of k values
        k_values = [k for _, k in nonzero_k]
        if k_values:
            from collections import Counter
            k_counts = Counter(k_values)
            print(f'  k value distribution: {dict(sorted(k_counts.items()))}')

    return result


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


# ---------------------------------------------------------------------------------------------------------------------
# The 2D unwrapping engine: L1 IRLS. Each step solves the weighted Laplacian system by conjugate gradients with an
# aggregation multigrid preconditioner; the start is the unweighted least-squares solution by the 2D DCT (rocket-fft).
# The kernels take a row range [r0, r1) (a column range for the column DCTs) and run serially without the GIL; _Blocks
# runs them over fixed blocks of _BLOCK rows on a thread pool, so every result, sums included, is the same for any
# number of threads. Kept per raster (float32): the filled phase and the pixel weights (the targets and the edge weights
# are computed from them where needed), the IRLS weights, u, its best iterate and the CG's r, z, p, plus the multigrid
# levels (a third of a raster more): about 47 bytes a pixel.
# ---------------------------------------------------------------------------------------------------------------------
_BLOCK = 128                        # rows of one block: the unit of work and of every partial sum
# NO fastmath: with reassoc or contract, a vectorised loop and its scalar fallback (chosen at run time, e.g. by the
# overlap checks on the arrays' addresses) round differently, and the result changes from call to call. The sums keep
# four fixed accumulators instead: the same order on every path, and still several additions in flight.
_CG_REL, _CG_MAX = 0.3, 40          # a CG solve ends at 0.3 of its starting residual, or after 40 iterations
_EPS_START, _EPS_STEPS = 0.3, 6     # epsilon falls geometrically from 0.3 to `epsilon` over the first 6 steps
_MG_ALPHA, _MG_MIN, _MG_COARSE = 1.5, 24, 40   # coarse correction scale, coarsest side, coarsest-level sweeps
_MG_SERIAL = 1 << 18                # levels below this many pixels run inline (too short for a thread hand-off)
_OLD_IRLS = ('tol', 'cg_max_iter', 'cg_tol', 'device', 'semaphore')


@nb.njit(nogil=True, cache=True, inline='always')
def _wrap(d):
    return d - 2.0 * math.pi * math.floor(d / (2.0 * math.pi) + 0.5)


@nb.njit(nogil=True, cache=True, inline='always')
def _we(a, b):
    """The edge weight: the mean of the two pixel weights, 0 when either pixel is invalid (weight 0)."""
    return (a + b) / np.float32(2.0) if (a > 0 and b > 0) else np.float32(0.0)


@nb.njit(nogil=True, cache=True)
def _lap(x, wx, wy, i, j, H, W):
    """(A x)[i, j] of the weighted Neumann Laplacian A = -div(w grad), at any pixel."""
    xc = x[i, j]
    v = 0.0
    if j > 0:
        v += np.float64(wx[i, j - 1]) * (xc - x[i, j - 1])
    if j < W - 1:
        v -= np.float64(wx[i, j]) * (x[i, j + 1] - xc)
    if i > 0:
        v += np.float64(wy[i - 1, j]) * (xc - x[i - 1, j])
    if i < H - 1:
        v -= np.float64(wy[i, j]) * (x[i + 1, j] - xc)
    return v


@nb.njit(nogil=True, cache=True, inline='always')
def _lap_in(x, wx, wy, i, j):
    """(A x)[i, j] at an interior pixel, float32, no branch."""
    xc = x[i, j]
    return (wx[i, j - 1] * (xc - x[i, j - 1]) - wx[i, j] * (x[i, j + 1] - xc)
            + wy[i - 1, j] * (xc - x[i - 1, j]) - wy[i, j] * (x[i + 1, j] - xc))


@nb.njit(nogil=True, cache=True)
def _rhs(wx, wy, phi, i, j, H, W):
    """b[i, j] = -div(w g), the targets g = wrap(grad phi) on the fly."""
    v = 0.0
    if j > 0:
        v += np.float64(wx[i, j - 1]) * _wrap(np.float64(phi[i, j]) - phi[i, j - 1])
    if j < W - 1:
        v -= np.float64(wx[i, j]) * _wrap(np.float64(phi[i, j + 1]) - phi[i, j])
    if i > 0:
        v += np.float64(wy[i - 1, j]) * _wrap(np.float64(phi[i, j]) - phi[i - 1, j])
    if i < H - 1:
        v -= np.float64(wy[i, j]) * _wrap(np.float64(phi[i + 1, j]) - phi[i, j])
    return v


@nb.njit(nogil=True, cache=True)
def _rhs_base(w, phi, out, r0, r1):
    """out = -div(w_edge g): the right-hand side of the start, edge weights from the pixel weights."""
    H, W = out.shape
    for i in range(r0, r1):
        for j in range(W):
            v = 0.0
            if j > 0:
                v += np.float64(_we(w[i, j - 1], w[i, j])) * _wrap(np.float64(phi[i, j]) - phi[i, j - 1])
            if j < W - 1:
                v -= np.float64(_we(w[i, j], w[i, j + 1])) * _wrap(np.float64(phi[i, j + 1]) - phi[i, j])
            if i > 0:
                v += np.float64(_we(w[i - 1, j], w[i, j])) * _wrap(np.float64(phi[i, j]) - phi[i - 1, j])
            if i < H - 1:
                v -= np.float64(_we(w[i, j], w[i + 1, j])) * _wrap(np.float64(phi[i + 1, j]) - phi[i, j])
            out[i, j] = v


@nb.njit(nogil=True, cache=True)
def _irls_weights(u, phi, w, wxi, wyi, eps_sq, r0, r1):
    """wxi = w_edge / sqrt(r^2 + eps^2) clamped to [1e-6, 1e6], r = grad u - wrap(grad phi)."""
    H, W = u.shape
    for i in range(r0, r1):
        for j in range(W):
            if j < W - 1:
                rx = np.float32(np.float64(u[i, j + 1] - u[i, j]) - _wrap(np.float64(phi[i, j + 1]) - phi[i, j]))
                ax = _we(w[i, j], w[i, j + 1])
            else:
                rx = np.float32(0.0)
                ax = np.float32(0.0)
            if i < H - 1:
                ry = np.float32(np.float64(u[i + 1, j] - u[i, j]) - _wrap(np.float64(phi[i + 1, j]) - phi[i, j]))
                ay = _we(w[i, j], w[i + 1, j])
            else:
                ry = np.float32(0.0)
                ay = np.float32(0.0)
            wxi[i, j] = min(max(ax / np.sqrt(rx * rx + eps_sq), np.float32(1e-6)), np.float32(1e6))
            wyi[i, j] = min(max(ay / np.sqrt(ry * ry + eps_sq), np.float32(1e-6)), np.float32(1e6))


@nb.njit(nogil=True, cache=True)
def _l1(u, phi, w, r0, r1):
    """sum over the valid edges of w_edge |grad u - wrap(grad phi)|: the objective, and the sum of u (finiteness)."""
    H, W = u.shape
    s = 0.0
    su = 0.0
    for i in range(r0, r1):
        for j in range(W):
            su += u[i, j]
            if j < W - 1:
                a = _we(w[i, j], w[i, j + 1])
                if a > 0:
                    s += np.float64(a) * abs(np.float64(u[i, j + 1] - u[i, j]) - _wrap(np.float64(phi[i, j + 1]) - phi[i, j]))
            if i < H - 1:
                a = _we(w[i, j], w[i + 1, j])
                if a > 0:
                    s += np.float64(a) * abs(np.float64(u[i + 1, j] - u[i, j]) - _wrap(np.float64(phi[i + 1, j]) - phi[i, j]))
    return s, su


@nb.njit(nogil=True, cache=True, inline='always')
def _cg_r(x, r, wxi, wyi, phi, i, j):
    """r[i, j] = b - (A x) at an interior pixel, b = -div(w_irls g) from the targets on the fly; returns it."""
    gl = np.float32(_wrap(np.float64(phi[i, j]) - phi[i, j - 1]))
    gr = np.float32(_wrap(np.float64(phi[i, j + 1]) - phi[i, j]))
    gu = np.float32(_wrap(np.float64(phi[i, j]) - phi[i - 1, j]))
    gd = np.float32(_wrap(np.float64(phi[i + 1, j]) - phi[i, j]))
    rv = (wxi[i, j - 1] * gl - wxi[i, j] * gr + wyi[i - 1, j] * gu - wyi[i, j] * gd) - _lap_in(x, wxi, wyi, i, j)
    r[i, j] = rv
    return rv


@nb.njit(nogil=True, cache=True)
def _cg_start(x, r, wxi, wyi, phi, r0, r1):
    """r = b - A x, b = -div(w_irls g); returns sum r^2 and sum r (a NaN or an infinity carries into it)."""
    H, W = x.shape
    s2 = 0.0
    s = 0.0
    for i in range(r0, r1):
        if 0 < i < H - 1 and W > 2:
            for j in (0, W - 1):
                v = _rhs(wxi, wyi, phi, i, j, H, W) - _lap(x, wxi, wyi, i, j, H, W)
                r[i, j] = v
                s2 += v * v
                s += v
            a0 = a1 = a2 = a3 = 0.0
            c0 = c1 = c2 = c3 = 0.0
            j = 1
            while j + 3 < W - 1:
                v0 = np.float64(_cg_r(x, r, wxi, wyi, phi, i, j))
                v1 = np.float64(_cg_r(x, r, wxi, wyi, phi, i, j + 1))
                v2 = np.float64(_cg_r(x, r, wxi, wyi, phi, i, j + 2))
                v3 = np.float64(_cg_r(x, r, wxi, wyi, phi, i, j + 3))
                a0 += v0 * v0
                a1 += v1 * v1
                a2 += v2 * v2
                a3 += v3 * v3
                c0 += v0
                c1 += v1
                c2 += v2
                c3 += v3
                j += 4
            while j < W - 1:
                v0 = np.float64(_cg_r(x, r, wxi, wyi, phi, i, j))
                a0 += v0 * v0
                c0 += v0
                j += 1
            s2 += (a0 + a1) + (a2 + a3)
            s += (c0 + c1) + (c2 + c3)
        else:
            for j in range(W):
                v = _rhs(wxi, wyi, phi, i, j, H, W) - _lap(x, wxi, wyi, i, j, H, W)
                r[i, j] = v
                s2 += v * v
                s += v
    return s2, s


@nb.njit(nogil=True, cache=True)
def _p_ap(p, wxi, wyi, r0, r1):
    """sum p (A p)."""
    H, W = p.shape
    s = 0.0
    for i in range(r0, r1):
        if 0 < i < H - 1 and W > 2:
            s += np.float64(p[i, 0]) * _lap(p, wxi, wyi, i, 0, H, W)
            a0 = a1 = a2 = a3 = 0.0
            j = 1
            while j + 3 < W - 1:
                a0 += np.float64(p[i, j] * _lap_in(p, wxi, wyi, i, j))
                a1 += np.float64(p[i, j + 1] * _lap_in(p, wxi, wyi, i, j + 1))
                a2 += np.float64(p[i, j + 2] * _lap_in(p, wxi, wyi, i, j + 2))
                a3 += np.float64(p[i, j + 3] * _lap_in(p, wxi, wyi, i, j + 3))
                j += 4
            while j < W - 1:
                a0 += np.float64(p[i, j] * _lap_in(p, wxi, wyi, i, j))
                j += 1
            s += ((a0 + a1) + (a2 + a3)) + np.float64(p[i, W - 1]) * _lap(p, wxi, wyi, i, W - 1, H, W)
        else:
            for j in range(W):
                s += np.float64(p[i, j]) * _lap(p, wxi, wyi, i, j, H, W)
    return s


@nb.njit(nogil=True, cache=True, inline='always')
def _cg_upd(x, r, p, wxi, wyi, a, i, j):
    """x[i, j] += a p, r[i, j] -= a (A p) at an interior pixel; returns (r, x)."""
    xv = x[i, j] + a * p[i, j]
    x[i, j] = xv
    rv = r[i, j] - a * _lap_in(p, wxi, wyi, i, j)
    r[i, j] = rv
    return rv, xv


@nb.njit(nogil=True, cache=True)
def _cg_step(x, r, p, wxi, wyi, alpha, r0, r1):
    """x += alpha p, r -= alpha A p (A p recomputed); returns sum r^2 and sum x (finiteness). The rows of x and r change
    while the stencil reads p only, so blocks do not interfere."""
    H, W = x.shape
    a = np.float32(alpha)
    s = 0.0
    sx = 0.0
    for i in range(r0, r1):
        if 0 < i < H - 1 and W > 2:
            a0 = a1 = a2 = a3 = 0.0
            b0 = b1 = b2 = b3 = 0.0
            j = 1
            while j + 3 < W - 1:
                rv, xv = _cg_upd(x, r, p, wxi, wyi, a, i, j)
                a0 += np.float64(rv * rv)
                b0 += np.float64(xv)
                rv, xv = _cg_upd(x, r, p, wxi, wyi, a, i, j + 1)
                a1 += np.float64(rv * rv)
                b1 += np.float64(xv)
                rv, xv = _cg_upd(x, r, p, wxi, wyi, a, i, j + 2)
                a2 += np.float64(rv * rv)
                b2 += np.float64(xv)
                rv, xv = _cg_upd(x, r, p, wxi, wyi, a, i, j + 3)
                a3 += np.float64(rv * rv)
                b3 += np.float64(xv)
                j += 4
            while j < W - 1:
                rv, xv = _cg_upd(x, r, p, wxi, wyi, a, i, j)
                a0 += np.float64(rv * rv)
                b0 += np.float64(xv)
                j += 1
            s += (a0 + a1) + (a2 + a3)
            sx += (b0 + b1) + (b2 + b3)
            for j in (0, W - 1):
                xv = x[i, j] + a * p[i, j]
                x[i, j] = xv
                rv = r[i, j] - np.float32(alpha * _lap(p, wxi, wyi, i, j, H, W))
                r[i, j] = rv
                s += np.float64(rv) * rv
                sx += xv
        else:
            for j in range(W):
                xv = x[i, j] + a * p[i, j]
                x[i, j] = xv
                rv = r[i, j] - np.float32(alpha * _lap(p, wxi, wyi, i, j, H, W))
                r[i, j] = rv
                s += np.float64(rv) * rv
                sx += xv
    return s, sx


@nb.njit(nogil=True, cache=True)
def _dot(a, b, r0, r1):
    W = a.shape[1]
    s = 0.0
    for i in range(r0, r1):
        a0 = a1 = a2 = a3 = 0.0
        j = 0
        while j + 3 < W:
            a0 += np.float64(a[i, j]) * b[i, j]
            a1 += np.float64(a[i, j + 1]) * b[i, j + 1]
            a2 += np.float64(a[i, j + 2]) * b[i, j + 2]
            a3 += np.float64(a[i, j + 3]) * b[i, j + 3]
            j += 4
        while j < W:
            a0 += np.float64(a[i, j]) * b[i, j]
            j += 1
        s += (a0 + a1) + (a2 + a3)
    return s


@nb.njit(nogil=True, cache=True)
def _p_update(p, z, beta, r0, r1):
    """p = z + beta p."""
    for i in range(r0, r1):
        for j in range(p.shape[1]):
            p[i, j] = z[i, j] + np.float32(beta) * p[i, j]


@nb.njit(nogil=True, cache=True)
def _masked_sum(u, w, r0, r1):
    """the sum of u over the valid pixels (weight > 0) and their count."""
    s = 0.0
    n = 0
    for i in range(r0, r1):
        for j in range(u.shape[1]):
            if w[i, j] > 0:
                s += u[i, j]
                n += 1
    return s, n


@nb.njit(nogil=True, cache=True)
def _shift(u, m, r0, r1):
    for i in range(r0, r1):
        for j in range(u.shape[1]):
            u[i, j] -= np.float32(m)


@nb.njit(nogil=True, cache=True)
def _copy(dst, src, r0, r1):
    for i in range(r0, r1):
        for j in range(dst.shape[1]):
            dst[i, j] = src[i, j]


@nb.njit(nogil=True, cache=True)
def _zero(x, r0, r1):
    for i in range(r0, r1):
        for j in range(x.shape[1]):
            x[i, j] = 0.0


@nb.njit(nogil=True, cache=True)
def _dct_rows(x, out, r0, r1):
    out[r0:r1] = scipy.fft.dct(x[r0:r1], 2, None, 1, None, False, 1)


@nb.njit(nogil=True, cache=True)
def _dct_cols(x, out, c0, c1):
    out[:, c0:c1] = scipy.fft.dct(x[:, c0:c1], 2, None, 0, None, False, 1)


@nb.njit(nogil=True, cache=True)
def _idct_rows(x, out, r0, r1):
    out[r0:r1] = scipy.fft.idct(x[r0:r1], 2, None, 1, None, False, 1)


@nb.njit(nogil=True, cache=True)
def _idct_cols(x, out, c0, c1):
    out[:, c0:c1] = scipy.fft.idct(x[:, c0:c1], 2, None, 0, None, False, 1)


@nb.njit(nogil=True, cache=True)
def _eig_scale(t, ey, ex, r0, r1):
    """t /= -(ey[i] + ex[j]), the mean mode set to 0; float64 eigenvalues ey = -4 sin^2(pi i / 2H): no cancellation."""
    for i in range(r0, r1):
        for j in range(t.shape[1]):
            t[i, j] = -(t[i, j] / (ey[i] + ex[j])) if (i > 0 or j > 0) else 0.0


@nb.njit(nogil=True, cache=True)
def _coarsen(wx, wy):
    """The coarse edge weights of 2 x 2 aggregation: the Galerkin operator of a piecewise constant prolongation is again
    a weighted 5-point Laplacian whose edges are the sums of the fine edges crossing between two aggregates."""
    H, W = wx.shape
    H1, W1 = (H + 1) // 2, (W + 1) // 2
    wx1 = np.zeros((H1, W1), np.float32)
    wy1 = np.zeros((H1, W1), np.float32)
    for I in range(H1):
        for J in range(W1):
            if J < W1 - 1:
                j = 2 * J + 1
                s = np.float32(0.0)
                for i in (2 * I, 2 * I + 1):
                    if i < H and j < W - 1:
                        s += wx[i, j]
                wx1[I, J] = s
            if I < H1 - 1:
                i = 2 * I + 1
                s = np.float32(0.0)
                for j in (2 * J, 2 * J + 1):
                    if j < W and i < H - 1:
                        s += wy[i, j]
                wy1[I, J] = s
    return wx1, wy1


@nb.njit(nogil=True, cache=True)
def _gs_rows(x, b, wx, wy, color, r0, r1):
    """One red (0) or black (1) Gauss-Seidel half-sweep of A x = b over the rows r0..r1: a pixel of one colour reads the
    other colour only, so the blocks of a half-sweep are independent."""
    H, W = x.shape
    for i in range(r0, r1):
        j0 = (i + color) % 2
        if 0 < i < H - 1 and W > 2:
            j = j0
            if j == 0:
                d = np.float64(wx[i, 0]) + wy[i - 1, 0] + wy[i, 0]
                if d > 0:
                    x[i, 0] = (np.float64(b[i, 0]) + np.float64(wx[i, 0]) * x[i, 1] + np.float64(wy[i - 1, 0]) * x[i - 1, 0]
                               + np.float64(wy[i, 0]) * x[i + 1, 0]) / d
                j = 2
            while j < W - 1:
                a0, a1, a2, a3 = wx[i, j - 1], wx[i, j], wy[i - 1, j], wy[i, j]
                d = np.float64(a0) + a1 + a2 + a3
                if d > 0:
                    x[i, j] = (np.float64(b[i, j]) + np.float64(a0) * x[i, j - 1] + np.float64(a1) * x[i, j + 1]
                               + np.float64(a2) * x[i - 1, j] + np.float64(a3) * x[i + 1, j]) / d
                j += 2
            if j == W - 1:
                d = np.float64(wx[i, j - 1]) + wy[i - 1, j] + wy[i, j]
                if d > 0:
                    x[i, j] = (np.float64(b[i, j]) + np.float64(wx[i, j - 1]) * x[i, j - 1] + np.float64(wy[i - 1, j]) * x[i - 1, j]
                               + np.float64(wy[i, j]) * x[i + 1, j]) / d
        else:
            for j in range(j0, W, 2):
                s = np.float64(b[i, j])
                d = 0.0
                if j > 0:
                    s += np.float64(wx[i, j - 1]) * x[i, j - 1]
                    d += wx[i, j - 1]
                if j < W - 1:
                    s += np.float64(wx[i, j]) * x[i, j + 1]
                    d += wx[i, j]
                if i > 0:
                    s += np.float64(wy[i - 1, j]) * x[i - 1, j]
                    d += wy[i - 1, j]
                if i < H - 1:
                    s += np.float64(wy[i, j]) * x[i + 1, j]
                    d += wy[i, j]
                if d > 0:
                    x[i, j] = s / d


@nb.njit(nogil=True, cache=True)
def _res_rows(x, b, wx, wy, out, r0, r1):
    """out = b - A x."""
    H, W = x.shape
    for i in range(r0, r1):
        if 0 < i < H - 1 and W > 2:
            out[i, 0] = b[i, 0] - _lap(x, wx, wy, i, 0, H, W)
            for j in range(1, W - 1):
                out[i, j] = b[i, j] - _lap_in(x, wx, wy, i, j)
            out[i, W - 1] = b[i, W - 1] - _lap(x, wx, wy, i, W - 1, H, W)
        else:
            for j in range(W):
                out[i, j] = b[i, j] - _lap(x, wx, wy, i, j, H, W)


@nb.njit(nogil=True, cache=True)
def _restrict_rows(res, rc, R0, R1):
    """rc = the 2 x 2 sums of res (the transpose of the piecewise constant prolongation), coarse rows R0..R1."""
    H, W = res.shape
    for I in range(R0, R1):
        i0 = 2 * I
        i1 = i0 + 1
        for J in range(rc.shape[1]):
            j0 = 2 * J
            j1 = j0 + 1
            s = np.float64(res[i0, j0])
            if j1 < W:
                s += res[i0, j1]
            if i1 < H:
                s += res[i1, j0]
                if j1 < W:
                    s += res[i1, j1]
            rc[I, J] = s


@nb.njit(nogil=True, cache=True)
def _prolong_rows(xc, x, alpha, r0, r1):
    """x += alpha P xc: each coarse value into its 2 x 2 aggregate."""
    a = np.float32(alpha)
    for i in range(r0, r1):
        I = i // 2
        for j in range(x.shape[1]):
            x[i, j] += a * xc[I, j // 2]


@nb.njit(nogil=True, cache=True)
def _sum_rows(x, r0, r1):
    W = x.shape[1]
    s = 0.0
    for i in range(r0, r1):
        a0 = a1 = a2 = a3 = 0.0
        j = 0
        while j + 3 < W:
            a0 += x[i, j]
            a1 += x[i, j + 1]
            a2 += x[i, j + 2]
            a3 += x[i, j + 3]
            j += 4
        while j < W:
            a0 += x[i, j]
            j += 1
        s += (a0 + a1) + (a2 + a3)
    return s


def _cores():
    """The CPU cores this process may use: os.process_cpu_count() (Python 3.13+), else the affinity mask, else
    os.cpu_count()."""
    import os
    count = getattr(os, 'process_cpu_count', None)
    if count is not None:
        return max(1, count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def unwrap_threads(rasters):
    """
    Numba threads for each of `rasters` rasters unwrapped together: the cores over the rasters that run at once.

    The cores are the Dask cluster's declared 'cpu' resource, else this process's own count (as fit1d_network() counts them);
    at most the cluster's worker threads (without a cluster, the cores) run at once. One raster takes every core, 8
    take 2 each on 16 cores, 16 and more take 1. The result does not depend on it.

    Examples
    --------
    >>> unwrap_threads(1)
    """
    cores, slots = None, None
    try:
        from dask.distributed import get_client
        workers = get_client().scheduler_info().get('workers', {})
        if workers:
            slots = sum(int(w.get('nthreads', 1)) for w in workers.values())
            declared = [w.get('resources', {}).get('cpu') for w in workers.values()]
            declared = [d for d in declared if d]
            if declared:
                cores = int(max(declared))
    except (ValueError, ImportError):
        pass
    if cores is None:
        cores = _cores()
    if slots is None:
        slots = cores
    return max(1, cores // max(1, min(int(rasters), slots)))


class _Blocks:
    """Fixed blocks of _BLOCK rows (and of _BLOCK columns for the column transforms) over a pool of `threads` Python
    threads; results come back in block order, so a sum of them does not depend on the number of threads."""

    def __init__(self, H, W, threads):
        import concurrent.futures
        self.threads = max(1, int(threads))
        self.rows = [(a, min(a + _BLOCK, H)) for a in range(0, H, _BLOCK)]
        self.cols = [(a, min(a + _BLOCK, W)) for a in range(0, W, _BLOCK)]
        self.ex = concurrent.futures.ThreadPoolExecutor(self.threads) if self.threads > 1 else None

    def run(self, fn, args, blocks, inline=False):
        if self.ex is None or inline:
            return [fn(*args, a, b) for a, b in blocks]
        return list(self.ex.map(lambda ab: fn(*args, ab[0], ab[1]), blocks))

    def map(self, fn, *args):
        return self.run(fn, args, self.rows)

    def map_cols(self, fn, *args):
        return self.run(fn, args, self.cols)

    def close(self):
        if self.ex is not None:
            self.ex.shutdown()


class _Multigrid:
    """The aggregation multigrid V-cycle of one IRLS step's system, z = M^-1 r: red-black Gauss-Seidel (red then black
    before the coarse correction, black then red after: symmetric, so it preconditions CG), 2 x 2 restriction and
    injection, the coarse correction scaled by _MG_ALPHA, the coarsest level by _MG_COARSE symmetric sweeps, and the
    mean of z removed (the constant null space)."""

    def __init__(self, wx, wy, blocks):
        self.levels = [(wx, wy)]
        while min(self.levels[-1][0].shape) > _MG_MIN:
            self.levels.append(_coarsen(*self.levels[-1]))
        self.blocks = blocks
        self.x = [None] + [np.zeros(l[0].shape, np.float32) for l in self.levels[1:]]
        self.b = [None] + [np.zeros(l[0].shape, np.float32) for l in self.levels[1:]]
        self.res = [np.zeros(l[0].shape, np.float32) for l in self.levels[:-1]]

    def _run(self, k, fn, args, n, step=_BLOCK):
        inline = self.levels[k][0].size < _MG_SERIAL
        return self.blocks.run(fn, args, [(a, min(a + step, n)) for a in range(0, n, step)], inline)

    def _cycle(self, k, b, x):
        wx, wy = self.levels[k]
        H = x.shape[0]
        self._run(k, _zero, (x,), H)
        if k == len(self.levels) - 1:
            for _ in range(_MG_COARSE):
                for c in (0, 1, 1, 0):
                    self._run(k, _gs_rows, (x, b, wx, wy, c), H)
            return
        for c in (0, 1):
            self._run(k, _gs_rows, (x, b, wx, wy, c), H)
        self._run(k, _res_rows, (x, b, wx, wy, self.res[k]), H)
        bc, xc = self.b[k + 1], self.x[k + 1]
        self._run(k, _restrict_rows, (self.res[k], bc), bc.shape[0], _BLOCK // 2)
        self._cycle(k + 1, bc, xc)
        self._run(k, _prolong_rows, (xc, x, _MG_ALPHA), H)
        for c in (1, 0):
            self._run(k, _gs_rows, (x, b, wx, wy, c), H)

    def apply(self, r, z):
        self._cycle(0, r, z)
        m = sum(self._run(0, _sum_rows, (z,), z.shape[0])) / z.size
        self._run(0, _shift, (z, m), z.shape[0])


def _raster(phase, weight):
    """The valid bounding box of a raster given as a 2D array or as nested blocks ([rows][columns], dask's
    concatenate=False form; a block may carry a leading axis of 1), and its filled float32 phase (NaN 0), NaN mask and
    pixel weights (clipped to [0.01, 1], 0 where the phase is NaN; 1 without a weight), assembled block by block: no
    merged copy of the raster."""
    def grid(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return [[x]]
        return x if isinstance(x[0], (list, tuple)) else [x]
    pb, wb = grid(phase), grid(weight)

    def two_d(a):
        a = np.asarray(a)
        return a.reshape(a.shape[-2:]) if a.ndim > 2 else a
    heights = [two_d(row[0]).shape[0] for row in pb]
    widths = [two_d(b).shape[1] for b in pb[0]]
    y0s, x0s = np.cumsum([0] + heights), np.cumsum([0] + widths)
    shape = (int(y0s[-1]), int(x0s[-1]))
    rows_ok, cols_ok = np.zeros(shape[0], bool), np.zeros(shape[1], bool)
    for bi, row in enumerate(pb):
        for bj, blk in enumerate(row):
            ok = ~np.isnan(two_d(blk))
            rows_ok[y0s[bi]:y0s[bi + 1]] |= ok.any(axis=1)
            cols_ok[x0s[bj]:x0s[bj + 1]] |= ok.any(axis=0)
    if not rows_ok.any():
        return shape, None, None, None, None
    r0, r1 = int(np.argmax(rows_ok)), shape[0] - int(np.argmax(rows_ok[::-1]))
    c0, c1 = int(np.argmax(cols_ok)), shape[1] - int(np.argmax(cols_ok[::-1]))
    H, W = r1 - r0, c1 - c0
    phi, w = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    nan_mask = np.ones((H, W), bool)
    for bi, row in enumerate(pb):
        for bj, blk in enumerate(row):
            ya, yb = max(y0s[bi], r0), min(y0s[bi + 1], r1)
            xa, xb = max(x0s[bj], c0), min(x0s[bj + 1], c1)
            if ya >= yb or xa >= xb:
                continue
            src = two_d(blk)[ya - y0s[bi]:yb - y0s[bi], xa - x0s[bj]:xb - x0s[bj]]
            bad = np.isnan(src)
            dst = (slice(ya - r0, yb - r0), slice(xa - c0, xb - c0))
            phi[dst] = np.where(bad, 0.0, src)
            nan_mask[dst] = bad
            if wb is None:
                w[dst] = np.where(bad, 0.0, 1.0)
            else:
                ws = two_d(wb[bi][bj])[ya - y0s[bi]:yb - y0s[bi], xa - x0s[bj]:xb - x0s[bj]]
                w[dst] = np.where(bad, 0.0, np.clip(np.nan_to_num(ws, nan=0.0), 0.01, 1.0))
    return shape, (r0, r1, c0, c1), phi, nan_mask, w


def _irls(phi, w, accuracy, max_iter, epsilon, threads, debug):
    """The solve on the filled raster: the unwrapped float32 solution (the valid pixels' values; any value elsewhere)."""
    import time
    t0 = time.time()
    H, W = phi.shape
    blocks = _Blocks(H, W, threads)
    try:
        wxi, wyi, u, u_best, r, z, p = (np.empty((H, W), np.float32) for _ in range(7))
        # the start: the unweighted Neumann least-squares solution by the 2D DCT, in place on u
        ey = -4.0 * np.sin(np.pi * np.arange(H) / (2 * H)) ** 2
        ex = -4.0 * np.sin(np.pi * np.arange(W) / (2 * W)) ** 2
        blocks.map(_rhs_base, w, phi, r)
        blocks.map(_dct_rows, r, u)
        blocks.map_cols(_dct_cols, u, u)
        blocks.map(_eig_scale, u, ey, ex)
        blocks.map_cols(_idct_cols, u, u)
        blocks.map(_idct_rows, u, u)
        np.nan_to_num(u, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts = blocks.map(_masked_sum, u, w)
        count = sum(n for _, n in parts)
        l1_prev, l1_best, cg_total, step, reached = None, None, 0, -1, False
        for step in range(max_iter):
            if count > 0:
                parts = blocks.map(_masked_sum, u, w)
                blocks.map(_shift, u, sum(s for s, _ in parts) / count)
            parts = blocks.map(_l1, u, phi, w)
            l1 = sum(s for s, _ in parts)
            if not math.isfinite(sum(su for _, su in parts)):
                blocks.map(_copy, u, u_best)
                break
            if l1_best is None or l1 < l1_best:
                l1_best = l1
                blocks.map(_copy, u_best, u)
            # THE STOP: the objective's relative decrease over one step, once epsilon is at its target
            if l1_prev is not None and step > _EPS_STEPS and (l1_prev - l1) <= accuracy * l1:
                reached = True
                break
            l1_prev = l1
            eps = epsilon * (_EPS_START / epsilon) ** max(0.0, 1.0 - step / _EPS_STEPS) if _EPS_START > epsilon else epsilon
            blocks.map(_irls_weights, u, phi, w, wxi, wyi, np.float32(eps * eps))
            mg = _Multigrid(wxi, wyi, blocks)
            parts = blocks.map(_cg_start, u, r, wxi, wyi, phi)
            r0n = math.sqrt(sum(s2 for s2, _ in parts))
            if not math.isfinite(sum(s for _, s in parts)) or r0n == 0.0:
                continue
            mg.apply(r, z)
            blocks.map(_copy, p, z)
            rz = sum(blocks.map(_dot, r, z))
            for k in range(_CG_MAX):
                cg_total += 1
                pap = sum(blocks.map(_p_ap, p, wxi, wyi))
                if not (abs(pap) > 1e-30) or not math.isfinite(pap):
                    break
                alpha = min(max(rz / pap, -1e6), 1e6)
                parts = blocks.map(_cg_step, u, r, p, wxi, wyi, alpha)
                rn = math.sqrt(sum(s for s, _ in parts))
                if not math.isfinite(sum(sx for _, sx in parts)) or not math.isfinite(rn) or rn <= _CG_REL * r0n:
                    break
                mg.apply(r, z)
                rz_new = sum(blocks.map(_dot, r, z))
                if not (abs(rz) > 1e-30):
                    break
                blocks.map(_p_update, p, z, min(max(rz_new / rz, -1e6), 1e6))
                rz = rz_new
            del mg
            if debug:
                print(f'  unwrap step {step}: L1 {l1:.6g}, eps {eps:.3g}, CG {k + 1}', flush=True)
        else:
            parts = blocks.map(_l1, u, phi, w)
            if math.isfinite(sum(su for _, su in parts)) and sum(s for s, _ in parts) <= l1_best:
                l1_best = sum(s for s, _ in parts)
                blocks.map(_copy, u_best, u)
        result = u_best
    finally:
        blocks.close()
    if debug:
        print(f'  unwrap {H}x{W}: {step + 1} steps, {cg_total} CG iterations, L1 {l1_best:.6g}, '
              f'{"accuracy reached" if reached else "max_iter reached"}, {time.time() - t0:.2f} s', flush=True)
    return result, reached, step + 1


def _postprocess(unwrapped, phi, nan_mask, conncomp_size):
    """The 2 pi cycle of the median pixel, the input's circular mean per 4-connected component, the labels by size
    (0 invalid, 1 the largest)."""
    import cv2
    unwrapped[nan_mask] = np.nan
    valid = ~nan_mask
    conncomp = np.zeros(unwrapped.shape, dtype=np.uint16)
    if not np.any(valid):
        return unwrapped, conncomp
    k = np.round((unwrapped[valid] - phi[valid]) / (2 * np.pi))
    unwrapped[valid] -= np.median(k) * 2 * np.pi
    n_labels, labeled = cv2.connectedComponents(valid.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    labeled_flat = np.ascontiguousarray(labeled.ravel(), dtype=np.int32)
    unwrapped_flat = unwrapped.ravel()
    sin_in, cos_in, sin_out, cos_out, count = _accum_sincos_count(labeled_flat, np.ascontiguousarray(phi.ravel()),
                                                                  unwrapped_flat.astype(np.float32), n_labels)
    cf = np.maximum(count.astype(np.float64), 1)
    d = np.arctan2(sin_in / cf, cos_in / cf) - np.arctan2(sin_out / cf, cos_out / cf)
    adjustment = np.arctan2(np.sin(d), np.cos(d))
    adjustment[0] = 0
    _apply_adjustment(unwrapped_flat, labeled_flat, adjustment)
    sizes = count[1:n_labels]
    old_to_new = np.zeros(n_labels, dtype=np.uint16)
    new_label = 1
    for old in np.argsort(sizes)[::-1]:
        if sizes[old] >= conncomp_size:
            old_to_new[old + 1] = new_label
            new_label += 1
    return unwrapped, old_to_new[labeled].astype(np.uint16)


def old_irls_warning(given, where):
    """ONE warning naming the parameters of the former solver that `where` accepts and ignores."""
    if given:
        print(f'WARNING: {where} ignores {", ".join(sorted(given))}: the solver stops at `accuracy` '
              f'(max_iter is its safety cap).', flush=True)


def irls_unwrap_2d(phase, weight=None, accuracy=2e-3, max_iter=100, epsilon=1e-2, conncomp_size=30, threads=None,
                   debug=False, **kwargs):
    """
    Unwrap 2D phase: min sum w |grad u - wrap(grad phase)| (L1) by iteratively reweighted least squares.

    Each step solves its weighted least-squares system by conjugate gradients preconditioned with an aggregation
    multigrid built from that step's weights; the start is the unweighted least-squares solution by the 2D DCT, and
    epsilon (the L1 smoothing) falls from 0.3 to `epsilon` over the first 6 steps. The solve stops when one step lowers
    the L1 objective by less than `accuracy` of it: 2e-3 (the default) reaches the former strong set's result in less
    time than the former relaxed set took; 5e-4 is near the optimum and takes about 1.9 times longer. Then the 2 pi
    cycle of the median pixel is the input's, and every 4-connected component gets the input's circular mean.

    Parameters
    ----------
    phase : np.ndarray or nested list of np.ndarray
        Wrapped phase (radians), 2D, NaN where invalid; or its blocks as dask's concatenate=False passes them.
    weight : like phase, optional
        Quality weight (correlation), clipped to [0.01, 1]; uniform without it.
    accuracy : float, optional
        The stop: the relative decrease of the L1 objective over one step. Default 2e-3.
    max_iter : int, optional
        The safety cap on the steps; reaching it before `accuracy` prints one WARNING. Default 100.
    epsilon : float, optional
        The L1 smoothing at the end of the continuation. Default 1e-2.
    conncomp_size : int, optional
        Components smaller than this many pixels get the label 0. Default 30.
    threads : int, optional
        Numba threads; the result is the same for any value. Default: the cores (unwrap_threads(1)).
    debug : bool, optional
        Print every step's objective.
    **kwargs
        The former solver's tol, cg_max_iter, cg_tol, device, semaphore: accepted and ignored with one WARNING.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The unwrapped phase (float32, NaN where invalid) and the component labels (uint16, 0 invalid, 1 the largest).

    Examples
    --------
    >>> unwrapped, conncomp = irls_unwrap_2d(phase, corr)
    >>> unwrapped, conncomp = irls_unwrap_2d(phase, corr, accuracy=5e-4)
    """
    unknown = sorted(set(kwargs) - set(_OLD_IRLS))
    if unknown:
        raise TypeError(f'irls_unwrap_2d() got unexpected keyword arguments: {", ".join(unknown)}')
    old_irls_warning(kwargs, 'irls_unwrap_2d()')
    shape, box, phi, nan_mask, w = _raster(phase, weight)
    if box is None:
        return np.full(shape, np.nan, np.float32), np.zeros(shape, np.uint16)
    r0, r1, c0, c1 = box
    u, reached, steps = _irls(phi, w, float(accuracy), int(max_iter), float(epsilon),
                              unwrap_threads(1) if threads is None else int(threads), debug)
    if not reached:
        print(f'WARNING: unwrapping stopped at max_iter={max_iter} before accuracy={accuracy:g}.', flush=True)
    del w
    unwrapped, conncomp = _postprocess(u, phi, nan_mask, conncomp_size)
    if box == (0, shape[0], 0, shape[1]):
        return unwrapped, conncomp
    out, cc = np.full(shape, np.nan, np.float32), np.zeros(shape, np.uint16)
    out[r0:r1, c0:c1] = unwrapped
    cc[r0:r1, c0:c1] = conncomp
    return out, cc


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


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _l = np.array([0, 1, 1], dtype=np.int32)
    _p = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    _u = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    _adj = np.array([0.0, 0.0], dtype=np.float64)
    _accum_sincos_count(_l, _p, _u, 2)
    _apply_adjustment(_u, _l, _adj)
    # the engine's kernels, on a small raster with an invalid corner (every branch); no warning
    _ph = np.linspace(-3.0, 3.0, 40 * 48, dtype=np.float32).reshape(40, 48)
    _ph[:3, :5] = np.nan
    _shape, _box, _phi, _nan, _w = _raster(_ph, np.full((40, 48), 0.5, np.float32))
    _uw, _, _ = _irls(_phi, _w, 2e-3, 2, 1e-2, 1, False)
    _postprocess(_uw, _phi, _nan, 30)

_warmup_numba_cache()


