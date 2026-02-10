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
import numba as nb

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


def _line_crosses_other_labels(labeled_array, p1, p2, label1, label2):
    """Check if line from p1 to p2 crosses labels other than label1/label2."""
    r1, c1 = p1
    r2, c2 = p2
    dr, dc = r2 - r1, c2 - c1
    n_steps = max(abs(dr), abs(dc), 1)

    for step in range(1, n_steps):
        t = step / n_steps
        r = int(round(r1 + t * dr))
        c = int(round(c1 + t * dc))

        label_at = labeled_array[r, c]
        if label_at > 0 and label_at != label1 and label_at != label2:
            return True
    return False


def _estimate_offset_fast(phase, labeled_array, label_i, label_j, p_i, p_j, n_neighbors=5):
    """Estimate phase offset between components at connection point."""
    # Sample pixels near connection point
    r_i, c_i = p_i
    r_j, c_j = p_j

    # Search window around connection points
    window = max(n_neighbors * 2, 10)

    # Get nearby pixels from component i
    r_min_i = max(0, r_i - window)
    r_max_i = min(labeled_array.shape[0], r_i + window + 1)
    c_min_i = max(0, c_i - window)
    c_max_i = min(labeled_array.shape[1], c_i + window + 1)

    sub_label_i = labeled_array[r_min_i:r_max_i, c_min_i:c_max_i]
    sub_phase_i = phase[r_min_i:r_max_i, c_min_i:c_max_i]
    mask_i = (sub_label_i == label_i) & ~np.isnan(sub_phase_i)

    # Get nearby pixels from component j
    r_min_j = max(0, r_j - window)
    r_max_j = min(labeled_array.shape[0], r_j + window + 1)
    c_min_j = max(0, c_j - window)
    c_max_j = min(labeled_array.shape[1], c_j + window + 1)

    sub_label_j = labeled_array[r_min_j:r_max_j, c_min_j:c_max_j]
    sub_phase_j = phase[r_min_j:r_max_j, c_min_j:c_max_j]
    mask_j = (sub_label_j == label_j) & ~np.isnan(sub_phase_j)

    if mask_i.sum() < 3 or mask_j.sum() < 3:
        return 0, 0.0

    # Get phase values
    phase_vals_i = sub_phase_i[mask_i]
    phase_vals_j = sub_phase_j[mask_j]

    # Use median for robustness
    median_i = np.median(phase_vals_i[:n_neighbors] if len(phase_vals_i) > n_neighbors else phase_vals_i)
    median_j = np.median(phase_vals_j[:n_neighbors] if len(phase_vals_j) > n_neighbors else phase_vals_j)

    delta = median_i - median_j
    k_offset = int(np.round(delta / (2 * np.pi)))

    # Confidence based on fractional part
    fractional = (delta / (2 * np.pi)) - k_offset
    confidence = 1.0 - 2 * abs(fractional)

    return k_offset, max(0, confidence)


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
                   cg_max_iter=20, cg_tol=1e-4, epsilon=1e-2, conncomp_size=30, debug=False):
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
    conncomp_size : int, optional
        Minimum connected component size in pixels. Components smaller than this
        are marked invalid (label 0). Default is 30. Set to 1 to keep all components.
    debug : bool, optional
        If True, print convergence information. Default is False.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - unwrapped: 2D float32 array of unwrapped phase values
        - conncomp: 2D uint16 array of component labels (0=invalid, 1=largest, 2=second, ...)

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
    import time
    from .BatchCore import BatchCore

    # Validate input is 2D
    if debug:
        print(f"DEBUG irls_unwrap_2d: input phase.shape={phase.shape}, weight.shape={weight.shape if weight is not None else None}")
    if phase.ndim != 2:
        raise ValueError(f"irls_unwrap_2d expects 2D phase array, got shape {phase.shape}")

    # Validate and set device
    if isinstance(device, str):
        if device == 'tpu':
            # TPU uses torch_xla device
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        elif device == 'auto':
            # Only call helper for auto-detection (respects Dask cluster resources)
            device = BatchCore._get_torch_device(device, debug=debug)
        else:
            # Explicit device ('cpu', 'cuda', 'mps') - use directly without re-detection
            if debug:
                print(f"DEBUG: using device={device}")
            device = torch.device(device)

    device_name = str(device)
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]

    # Use float32 for all devices - sufficient for phase unwrapping
    dtype = torch.float32
    np_dtype = np.float32

    _t_start = time.time()
    original_shape = phase.shape

    # Handle NaN values
    nan_mask = np.isnan(phase)
    if np.all(nan_mask):
        return np.full_like(phase, np.nan), np.zeros(phase.shape, dtype=np.uint16)

    # Find bounding box of valid pixels efficiently (only creates small 1D arrays)
    valid_rows = np.any(~nan_mask, axis=1)  # shape: (height,)
    valid_cols = np.any(~nan_mask, axis=0)  # shape: (width,)
    r0, r1 = np.argmax(valid_rows), len(valid_rows) - np.argmax(valid_rows[::-1])
    c0, c1 = np.argmax(valid_cols), len(valid_cols) - np.argmax(valid_cols[::-1])
    del valid_rows, valid_cols

    # Crop to bounding box if it saves significant space
    crop_applied = False
    if (r1 - r0) < phase.shape[0] or (c1 - c0) < phase.shape[1]:
        crop_applied = True
        phase = phase[r0:r1, c0:c1]  # view, no copy yet
        nan_mask = nan_mask[r0:r1, c0:c1]  # view
        if weight is not None:
            weight = weight[r0:r1, c0:c1]  # view
        if debug:
            print(f'  Cropped to bbox [{r0}:{r1}, {c0}:{c1}] = {r1-r0}x{c1-c0} '
                  f'(from {original_shape[0]}x{original_shape[1]})', flush=True)

    # Get dimensions after cropping
    height, width = phase.shape

    # Create valid mask for computation
    valid_mask = ~nan_mask

    # Fill NaN with 0 for computation (avoid np.where temp array)
    phase_filled = phase.copy()  # now copies only cropped region
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

    # Note: circular mean restoration is now done per-component at the end
    # to automatically align disconnected components

    # Convert to torch tensors
    phi = torch.from_numpy(phase_filled.astype(np_dtype)).to(device)
    w = torch.from_numpy(weight_filled.astype(np_dtype)).to(device)
    valid = torch.from_numpy(valid_mask).to(device)

    # Free numpy arrays - no longer needed
    del phase_filled, weight_filled

    # Compute wrapped phase differences (target gradients)
    dx_target = torch.zeros_like(phi)
    dy_target = torch.zeros_like(phi)
    dx_target[:, :-1] = phi[:, 1:] - phi[:, :-1]
    dy_target[:-1, :] = phi[1:, :] - phi[:-1, :]

    # Wrap to [-π, π] - in-place to avoid temporaries
    torch.atan2(torch.sin(dx_target), torch.cos(dx_target), out=dx_target)
    torch.atan2(torch.sin(dy_target), torch.cos(dy_target), out=dy_target)

    # phi no longer needed after target gradients computed
    del phi

    # Edge weights (average of adjacent pixel weights)
    wx = torch.zeros_like(w)
    wy = torch.zeros_like(w)
    wx[:, :-1] = (w[:, :-1] + w[:, 1:]) / 2
    wy[:-1, :] = (w[:-1, :] + w[1:, :]) / 2

    # Zero out weights at invalid edges
    wx[:, :-1] *= (valid[:, :-1] & valid[:, 1:]).to(dtype)
    wy[:-1, :] *= (valid[:-1, :] & valid[1:, :]).to(dtype)

    # w no longer needed after edge weights computed
    del w

    # Precompute DCT eigenvalues for preconditioner
    # IMPORTANT: Trigonometric functions require float64 precision!
    # Small errors in cos() accumulate through thousands of preconditioner
    # applications in the CG solver. Compute on CPU with float64, then transfer.
    i_idx = torch.arange(height, dtype=torch.float64)
    j_idx = torch.arange(width, dtype=torch.float64)
    cos_i = torch.cos(torch.pi * i_idx / height)
    cos_j = torch.cos(torch.pi * j_idx / width)
    eigenvalues = (2 * cos_i.unsqueeze(1) + 2 * cos_j.unsqueeze(0) - 4).to(dtype).to(device)
    eigenvalues[0, 0] = 1.0  # Avoid division by zero
    del i_idx, j_idx, cos_i, cos_j
    # Pre-allocate buffers - MINIMIZED count to reduce memory
    # Laplacian/gradient buffers (reused for dx_u, dy_u in IRLS)
    _dx_buf = torch.zeros((height, width), dtype=dtype, device=device)
    _dy_buf = torch.zeros((height, width), dtype=dtype, device=device)
    _result_buf = torch.zeros((height, width), dtype=dtype, device=device)

    # CG solver buffers (_cg_Ap eliminated - reuse _result_buf from apply_laplacian)
    _cg_r = torch.zeros((height, width), dtype=dtype, device=device)
    _cg_z = torch.zeros((height, width), dtype=dtype, device=device)
    _cg_p = torch.zeros((height, width), dtype=dtype, device=device)

    # IRLS buffers - rx, ry hold residuals needed for IRLS weights
    rx = torch.zeros((height, width), dtype=dtype, device=device)
    ry = torch.zeros((height, width), dtype=dtype, device=device)
    wx_irls = torch.zeros((height, width), dtype=dtype, device=device)
    wy_irls = torch.zeros((height, width), dtype=dtype, device=device)
    b = torch.zeros((height, width), dtype=dtype, device=device)
    u_prev = torch.zeros((height, width), dtype=dtype, device=device)

    # Use scalar for epsilon squared (saves 1.6GB vs full tensor)
    eps_sq = epsilon * epsilon

    # DCT work buffer
    _dct_work = torch.zeros((height, width), dtype=dtype, device=device)

    # Adaptive DCT chunking based on 2D pixels per chunk:
    # - Row DCT chunk: dct_chunk rows × width cols
    # - Col DCT chunk: height rows × dct_chunk cols
    # - CPU: no chunking (fastest)
    # - GPU/MPS: ~10M pixels per chunk (optimal for parallelism)
    if device_type == 'cpu':
        target_pixels = 1_000_000_000  # 1B = no chunking
    else:
        target_pixels = 10_000_000  # 10M pixels per chunk

    total_pixels = height * width
    if total_pixels < 2 * target_pixels:
        # Small array - no chunking needed
        dct_chunk = max(height, width)
    else:
        # Large array - chunk to target_pixels per operation
        dct_chunk = target_pixels // max(height, width)

    from torch_dct import dct as torch_dct_fn, idct as torch_idct_fn
    _dct_col_buf = torch.zeros((min(dct_chunk, width), height), dtype=dtype, device=device)

    def dct_2d_chunked(x, out):
        """2D DCT with adaptive chunking."""
        H, W = x.shape
        # Row DCT in chunks
        for i in range(0, H, dct_chunk):
            i_end = min(i + dct_chunk, H)
            out[i:i_end] = torch_dct_fn(x[i:i_end])
        # Column DCT via transpose buffer
        for j in range(0, W, dct_chunk):
            j_end = min(j + dct_chunk, W)
            cw = j_end - j
            _dct_col_buf[:cw, :].copy_(out[:, j:j_end].T)
            _dct_col_buf[:cw, :] = torch_dct_fn(_dct_col_buf[:cw, :])
            out[:, j:j_end].copy_(_dct_col_buf[:cw, :].T)

    def idct_2d_chunked(x, out):
        """2D IDCT with adaptive chunking."""
        H, W = x.shape
        # Column IDCT first
        for j in range(0, W, dct_chunk):
            j_end = min(j + dct_chunk, W)
            cw = j_end - j
            _dct_col_buf[:cw, :].copy_(x[:, j:j_end].T)
            _dct_col_buf[:cw, :] = torch_idct_fn(_dct_col_buf[:cw, :])
            out[:, j:j_end].copy_(_dct_col_buf[:cw, :].T)
        # Row IDCT
        for i in range(0, H, dct_chunk):
            i_end = min(i + dct_chunk, H)
            out[i:i_end] = torch_idct_fn(out[i:i_end])

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

        # In-place negation to avoid creating new tensor
        _result_buf.neg_()
        return _result_buf

    def apply_preconditioner(r, out):
        """Apply DCT-based preconditioner (approximate inverse Laplacian) into out buffer."""
        with torch.no_grad():
            # DCT-2D using chunked implementation
            dct_2d_chunked(r, _dct_work)

            # Divide by negative eigenvalues
            _dct_work.div_(eigenvalues)
            _dct_work.neg_()
            _dct_work[0, 0] = 0.0

            # IDCT-2D using chunked implementation
            idct_2d_chunked(_dct_work, out)
        return out

    def conjugate_gradient(b, wx_irls, wy_irls, x0, max_iter_cg, tol_cg):
        """Preconditioned conjugate gradient solver for IRLS using pre-allocated buffers.

        Returns (x, n_iters) where n_iters is the number of CG iterations performed.
        """
        x = x0  # No clone - modify in place, caller provides buffer
        n_iters = 0

        # r = b - A*x (use _cg_r buffer)
        _cg_r.copy_(b)
        _cg_r.sub_(apply_laplacian(x, wx_irls, wy_irls))

        # Check for NaN in initial residual
        if not torch.isfinite(_cg_r).all():
            return x0, 0

        # z = preconditioner(r) (use _cg_z buffer)
        apply_preconditioner(_cg_r, _cg_z)

        # p = z.clone() (use _cg_p buffer)
        _cg_p.copy_(_cg_z)

        # Use dot product to avoid creating height*width temporary tensor
        rz = torch.dot(_cg_r.view(-1), _cg_z.view(-1))

        for i in range(max_iter_cg):
            n_iters = i + 1
            # Ap = A*p (apply_laplacian returns _result_buf directly - no copy needed)
            Ap = apply_laplacian(_cg_p, wx_irls, wy_irls)

            # Use dot product to avoid creating height*width temporary tensor
            pAp = torch.dot(_cg_p.view(-1), Ap.view(-1))
            if pAp.abs() < 1e-15 or not torch.isfinite(pAp):
                break
            alpha = rz / pAp

            # Clamp alpha to prevent explosion
            alpha = torch.clamp(alpha, -1e6, 1e6)
            alpha_val = alpha.item()

            # Update x in-place
            x.add_(_cg_p, alpha=alpha_val)

            # Update r in-place (Ap is _result_buf, will be overwritten next iteration)
            _cg_r.sub_(Ap, alpha=alpha_val)

            # Check for numerical issues mid-iteration
            if not torch.isfinite(x).all():
                break

            # norm() is efficient and doesn't create intermediate tensor
            r_norm = _cg_r.norm()
            if r_norm < tol_cg or not torch.isfinite(r_norm):
                break

            # z = preconditioner(r) (reuse _cg_z buffer)
            apply_preconditioner(_cg_r, _cg_z)

            # Use dot product to avoid creating height*width temporary tensor
            rz_new = torch.dot(_cg_r.view(-1), _cg_z.view(-1))
            if rz.abs() < 1e-15:
                break
            beta = rz_new / rz
            beta = torch.clamp(beta, -1e6, 1e6)

            # p = z + beta*p (in-place)
            _cg_p.mul_(beta.item()).add_(_cg_z)
            rz = rz_new

        return x, n_iters

    # Initialize with DCT solution (L² result)
    # Use _result_buf temporarily for rho to avoid extra allocation
    rho = _result_buf
    rho.zero_()
    rho[:, 1:] += wx[:, :-1] * dx_target[:, :-1]
    rho[:, :-1] -= wx[:, :-1] * dx_target[:, :-1]
    rho[1:, :] += wy[:-1, :] * dy_target[:-1, :]
    rho[:-1, :] -= wy[:-1, :] * dy_target[:-1, :]

    # DCT initialization using chunked implementation
    dct_2d_chunked(rho, _dct_work)

    # Divide by negative eigenvalues
    _dct_work.div_(eigenvalues)
    _dct_work.neg_()
    _dct_work[0, 0] = 0.0

    # IDCT to get initial solution (use rho buffer as temporary output, then clone)
    idct_2d_chunked(_dct_work, rho)
    u = rho.clone()

    # Check DCT initialization for NaN/inf - fix in-place to avoid temp allocation
    if not torch.isfinite(u).all():
        if debug:
            nan_count = (~torch.isfinite(u)).sum().item()
            print(f'  DCT init produced {nan_count} NaN/inf values, filling with zeros')
        u.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-compute valid mask as float and count for efficient mean computation
    # (avoids creating 1GB+ temporary from u[valid] indexing on each iteration)
    valid_float = valid.to(dtype)
    valid_count = valid_float.sum()

    # Re-center DCT result to improve float32 precision
    # This keeps values near zero where float32 has best precision
    if valid_count > 0:
        u_mean = (u * valid_float).sum() / valid_count
        u.sub_(u_mean)

    _t_init = time.time()

    if debug:
        # Handle case where weights might still have issues
        edge_valid = valid[:, :-1] & valid[:, 1:]
        valid_wx = wx[:, :-1][edge_valid]
        if valid_wx.numel() > 0:
            w_min, w_max = valid_wx.min().item(), valid_wx.max().item()
        else:
            w_min, w_max = 0.0, 0.0
        print(f'  Input: {height}x{width}, valid pixels: {valid_mask.sum().item()}, '
              f'weight range: [{w_min:.3f}, {w_max:.3f}]')
        print(f'  DCT init range: [{u.min().item():.2f}, {u.max().item():.2f}]')

    # IRLS iterations - keep track of last good solution
    u_best = u.clone()
    best_residual = float('inf')

    for iteration in range(max_iter):
        u_prev.copy_(u)

        # Re-center u to prevent numerical drift (important for float32)
        # Phase unwrapping only cares about gradients, so mean is arbitrary
        # Use pre-computed valid_float to avoid creating 1GB+ temporary from u[valid]
        if valid_count > 0:
            u_mean = (u * valid_float).sum() / valid_count
            u.sub_(u_mean)

        # Compute current gradients into rx, ry directly (will subtract target next)
        rx.zero_()
        ry.zero_()
        rx[:, :-1] = u[:, 1:] - u[:, :-1]
        ry[:-1, :] = u[1:, :] - u[:-1, :]

        # Compute residuals in-place: rx = grad_x(u) - dx_target
        rx.sub_(dx_target)
        ry.sub_(dy_target)

        # Track best solution by residual magnitude
        # Use dot product to avoid creating height*width temporary tensors from pow(2)
        current_residual = (torch.dot(rx.view(-1), rx.view(-1)) + torch.dot(ry.view(-1), ry.view(-1))).item()
        if current_residual < best_residual and torch.isfinite(u).all():
            best_residual = current_residual
            u_best.copy_(u)

        # Update IRLS weights: w_irls = w / sqrt(r² + ε²)
        # Use scalar eps_sq to avoid 1.6GB tensor
        torch.mul(rx, rx, out=wx_irls)  # wx_irls = rx²
        wx_irls.add_(eps_sq)             # wx_irls = rx² + ε²
        wx_irls.sqrt_()                  # wx_irls = sqrt(rx² + ε²)
        torch.div(wx, wx_irls, out=wx_irls)  # wx_irls = wx / sqrt(...)

        torch.mul(ry, ry, out=wy_irls)  # wy_irls = ry²
        wy_irls.add_(eps_sq)             # wy_irls = ry² + ε²
        wy_irls.sqrt_()                  # wy_irls = sqrt(ry² + ε²)
        torch.div(wy, wy_irls, out=wy_irls)  # wy_irls = wy / sqrt(...)

        # Clamp weights in-place
        wx_irls.clamp_(min=1e-6, max=1e6)
        wy_irls.clamp_(min=1e-6, max=1e6)

        # Compute right-hand side: -∇·(w_irls · ∇target)
        b.zero_()
        b[:, 1:].addcmul_(wx_irls[:, :-1], dx_target[:, :-1])
        b[:, :-1].addcmul_(wx_irls[:, :-1], dx_target[:, :-1], value=-1)
        b[1:, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :])
        b[:-1, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :], value=-1)

        # Free rx, ry before CG - they're not needed during CG solve
        # This reduces peak memory during preconditioner calls by 3.2 GB
        del rx, ry
        if device_type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif device_type == 'cuda':
            torch.cuda.empty_cache()

        # Solve weighted Laplacian system using CG
        u, cg_iters = conjugate_gradient(b, wx_irls, wy_irls, u, cg_max_iter, cg_tol)

        # Recreate rx, ry for next iteration
        rx = torch.zeros((height, width), dtype=dtype, device=device)
        ry = torch.zeros((height, width), dtype=dtype, device=device)

        # Check for numerical issues during iteration
        if not torch.isfinite(u).all():
            if debug:
                print(f'  IRLS iter {iteration}: NaN/inf detected, reverting to best solution')
            u.copy_(u_best)
            break

        # Check convergence using pre-allocated buffer
        u_prev.sub_(u)  # u_prev now holds diff
        diff = torch.norm(u_prev)
        norm_u = torch.norm(u) + 1e-10
        rel_change = (diff / norm_u).item()

        if debug and iteration % 5 == 0:
            print(f'  IRLS iter {iteration}: rel_change = {rel_change:.2e}, residual = {current_residual:.2e}, cg_iters = {cg_iters}')

        # Aggressive memory cleanup to prevent accumulation from DCT temporaries
        # torch_dct creates ~15GB of intermediates per DCT/IDCT pair
        if device_type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif device_type == 'cuda':
            torch.cuda.empty_cache()

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

    # Convert back to numpy before cleanup
    unwrapped = u.cpu().numpy().astype(np.float32)

    # Explicit cleanup of all GPU tensors to free memory
    del u, u_best, u_prev
    del dx_target, dy_target, wx, wy, eigenvalues
    del _dx_buf, _dy_buf, _result_buf, _dct_work
    del _cg_r, _cg_z, _cg_p
    del rx, ry, wx_irls, wy_irls, b
    del valid, valid_float, valid_count

    # Force synchronization and clear GPU cache
    if device_type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device_type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Force Python garbage collection
    import gc
    gc.collect()

    # Check for NaN/inf from numerical issues - return NaN array (no hidden fallback)
    if not np.isfinite(unwrapped[~nan_mask]).all():
        import warnings
        nan_count = np.sum(~np.isfinite(unwrapped[~nan_mask]))
        total_valid = np.sum(~nan_mask)
        warnings.warn(f'IRLS produced {nan_count}/{total_valid} NaN/inf values - returning NaN for this component', RuntimeWarning)
        return np.full(original_shape, np.nan, dtype=np.float32), np.zeros(original_shape, dtype=np.uint16)

    # Restore NaN values
    unwrapped[nan_mask] = np.nan

    # Validate and correct: rewrapped phase should match input
    valid_mask = ~nan_mask
    if np.any(valid_mask):
        diff = unwrapped[valid_mask] - phase[valid_mask]
        k_values = np.round(diff / (2 * np.pi))
        k_median = np.median(k_values)
        unwrapped[valid_mask] = unwrapped[valid_mask] - k_median * 2 * np.pi

    # Restore input circular mean PER COMPONENT (not global) - VECTORIZED
    # Each disconnected component should be aligned to its own input circular mean
    # This automatically aligns components with sub-π offsets without needing ILP
    _t_align_start = time.time()
    n_components = 0
    conncomp = np.zeros(unwrapped.shape, dtype=np.uint16)

    if np.any(valid_mask):
        # Use cv2.connectedComponents (faster than WithStats - counts come from numba)
        import cv2
        _t0 = time.time()
        n_labels, labeled = cv2.connectedComponents(
            valid_mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S
        )
        n_components = n_labels - 1  # cv2 counts background as label 0
        _t1 = time.time()

        # Prepare contiguous arrays for numba
        labeled_flat = np.ascontiguousarray(labeled.ravel(), dtype=np.int32)
        phase_flat = np.ascontiguousarray(phase.ravel(), dtype=np.float32)
        unwrapped_flat = unwrapped.ravel()  # Will modify in-place
        _t2 = time.time()

        # Single-pass numba accumulation: sin/cos sums AND counts (6x faster than bincounts)
        sin_sum, cos_sum, sin_sum_out, cos_sum_out, count_per_label = _accum_sincos_count(
            labeled_flat, phase_flat, unwrapped_flat.astype(np.float32), n_labels
        )
        _t2b = time.time()

        # Circular mean per component: atan2(mean(sin), mean(cos))
        count_float = count_per_label.astype(np.float64)
        count_float[count_float == 0] = 1  # Avoid division by zero
        input_mean = np.arctan2(sin_sum / count_float, cos_sum / count_float)
        output_mean = np.arctan2(sin_sum_out / count_float, cos_sum_out / count_float)
        _t3 = time.time()

        # Adjustment per label - wrap to [-π, π] to avoid adding/removing full cycles
        diff = input_mean - output_mean
        adjustment = np.arctan2(np.sin(diff), np.cos(diff))  # wrap to [-π, π]
        adjustment[0] = 0
        # Apply adjustment using numba (faster than numpy fancy indexing)
        _apply_adjustment(unwrapped_flat, labeled_flat, adjustment)
        _t4 = time.time()

        # Relabel by size: 0=invalid, 1=largest, 2=second largest, etc.
        # Filter out components smaller than conncomp_size
        sizes_1_to_n = count_per_label[1:n_labels]  # sizes for labels 1..n_labels-1
        # argsort descending: largest first
        sorted_indices = np.argsort(sizes_1_to_n)[::-1]  # indices 0..n-1, sorted by size desc
        # Build mapping: old_label -> new_label (0 for small components)
        old_to_new = np.zeros(n_labels, dtype=np.uint16)
        new_label = 1
        for old_idx in sorted_indices:
            old_label = old_idx + 1  # labels are 1-indexed
            if sizes_1_to_n[old_idx] >= conncomp_size:
                old_to_new[old_label] = new_label
                new_label += 1
            # else: old_to_new[old_label] stays 0 (invalid)
        n_components = new_label - 1  # Update to count only kept components
        _t5 = time.time()

        if debug:
            n_filtered = np.sum(sizes_1_to_n < conncomp_size)
            print(f'  align timing: label={(_t1-_t0)*1000:.0f}ms, prep={(_t2-_t1)*1000:.0f}ms, '
                  f'accum={(_t2b-_t2)*1000:.0f}ms, mean={(_t3-_t2b)*1000:.0f}ms, '
                  f'adjust={(_t4-_t3)*1000:.0f}ms, relabel={(_t5-_t4)*1000:.0f}ms')
            print(f'  conncomp: {n_components} kept, {n_filtered} filtered (size<{conncomp_size})')
        # Apply relabeling vectorized
        conncomp = old_to_new[labeled].astype(np.uint16)

    _t_align_end = time.time()

    # Place cropped result back into full-size output if cropping was applied
    if crop_applied:
        result = np.full(original_shape, np.nan, dtype=np.float32)
        result[r0:r1, c0:c1] = unwrapped
        unwrapped = result
        conncomp_full = np.zeros(original_shape, dtype=np.uint16)
        conncomp_full[r0:r1, c0:c1] = conncomp
        conncomp = conncomp_full

    if debug:
        print(f'TIMING irls_unwrap_2d ({height}x{width}) on {device_name}:')
        print(f'  init (DCT):   {(_t_init - _t_start)*1000:.1f} ms')
        print(f'  IRLS iters:   {(_t_end - _t_init)*1000:.1f} ms ({iteration+1} iterations)')
        print(f'  align comps:  {(_t_align_end - _t_align_start)*1000:.1f} ms ({n_components} components)')
        print(f'  TOTAL:        {(_t_align_end - _t_start)*1000:.1f} ms')

    return unwrapped, conncomp


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
