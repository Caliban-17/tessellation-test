# -*- coding: utf-8 -*-
# WARNING: This file now contains the 2D TOROIDAL tessellation logic,
# despite the original 'tessellation.py' name.

import numpy as np
try:
    from scipy.spatial import Voronoi as ScipyVoronoi
    from scipy.spatial import QhullError
except ImportError:
    print("ERROR: SciPy library not found or import error. Please install it (`pip install scipy`)")
    raise

# --- Use absolute imports consistently ---
# Assumes package 'tessellation_test' is installed or PYTHONPATH is set correctly
try:
    from utils.geometry import ( # Assumes utils is at root
        toroidal_distance_sq, polygon_area, polygon_centroid, wrap_point,
        generate_ghost_points, clip_polygon_to_boundary
    )
except ImportError as e:
    print(f"ERROR: Could not import from root 'utils' directory: {e}. "
          "Ensure 'utils' is in the project root and PYTHONPATH includes the project root.")
    raise
# --- End Imports ---

try:
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon
    SHAPELY_AVAILABLE_FOR_UNION = True
except ImportError:
    SHAPELY_AVAILABLE_FOR_UNION = False


def generate_voronoi_regions_toroidal(points, width, height, epsilon=1e-9):
    """
    Generates Voronoi regions for points on a 2D torus [0, W] x [0, H].
    Regions pieces are merged using shapely.ops.unary_union to handle overlaps.
    Adds a small epsilon offset to points near boundaries for robustness.

    Args:
        points (np.ndarray): Array of shape (N, 2) of generator points in [0, W]x[0, H].
        width (float): Width of the torus domain.
        height (float): Height of the torus domain.
        epsilon (float): Small offset to nudge points away from exact boundaries.

    Returns:
        list: A list of lists of numpy arrays. Each inner list corresponds to an
              original point. Each numpy array within the inner list contains the
              vertices (M, 2) of a merged polygon piece making up the Voronoi region
              clipped to the boundary [0, width] x [0, height]. Usually one piece
              after merging, but can be multiple if the merged region is disjoint.
              Returns None if Voronoi calculation fails or input is invalid.
    """
    # Input validation
    if points is None: return None
    points_orig = np.asarray(points).copy()
    if points_orig.ndim != 2 or points_orig.shape[1] != 2: return None
    N_original = points_orig.shape[0]
    if N_original < 4: return None
    if width <= 0 or height <= 0: return None

    # Preprocessing & Degeneracy Checks
    points_offset = points_orig.copy()
    points_offset[:, 0] = np.where(np.isclose(points_offset[:, 0], 0.0), epsilon, points_offset[:, 0])
    points_offset[:, 0] = np.where(np.isclose(points_offset[:, 0], width), width - epsilon, points_offset[:, 0])
    points_offset[:, 1] = np.where(np.isclose(points_offset[:, 1], 0.0), epsilon, points_offset[:, 1])
    points_offset[:, 1] = np.where(np.isclose(points_offset[:, 1], height), height - epsilon, points_offset[:, 1])
    unique_offset_points, _ = np.unique(np.round(points_offset, decimals=9), axis=0, return_index=True)
    if len(unique_offset_points) < N_original: return None
    if len(unique_offset_points) >= 2:
        centered_points = unique_offset_points - np.mean(unique_offset_points, axis=0)
        rank = np.linalg.matrix_rank(centered_points, tol=1e-8)
        if rank < 2: return None

    try:
        all_points, original_indices = generate_ghost_points(points_offset, width, height)
        vor = ScipyVoronoi(all_points)
    except (QhullError, Exception):
        return None

    # Process regions and clip
    raw_clipped_regions = [[] for _ in range(N_original)]
    processed_voronoi_regions = {}
    point_region_map = vor.point_region
    for point_idx_all, region_idx in enumerate(point_region_map):
        if point_idx_all >= len(original_indices): continue
        original_idx = original_indices[point_idx_all]
        if region_idx == -1 or region_idx < 0 or region_idx >= len(vor.regions): continue
        if region_idx not in processed_voronoi_regions:
            region_vertex_indices = vor.regions[region_idx]
            if not region_vertex_indices:
                processed_voronoi_regions[region_idx] = []
                continue
            finite_vertex_indices = [idx for idx in region_vertex_indices if idx != -1]
            if len(finite_vertex_indices) < 3 or any(idx < 0 or idx >= len(vor.vertices) for idx in finite_vertex_indices):
                processed_voronoi_regions[region_idx] = []
                continue
            polygon_vertices = vor.vertices[finite_vertex_indices]
            clipped_pieces = clip_polygon_to_boundary(polygon_vertices, width, height)
            processed_voronoi_regions[region_idx] = clipped_pieces
        clipped_pieces = processed_voronoi_regions[region_idx]
        if clipped_pieces and 0 <= original_idx < N_original:
            raw_clipped_regions[original_idx].extend(clipped_pieces)

    # Merge pieces
    if not SHAPELY_AVAILABLE_FOR_UNION:
        final_regions = raw_clipped_regions
    else:
        merged_regions = []
        for i, pieces in enumerate(raw_clipped_regions):
            if not pieces: merged_regions.append([]); continue
            try:
                shapely_polygons = []
                for p_verts in pieces:
                     if p_verts is not None and len(p_verts) >= 3:
                          poly = Polygon([tuple(p) for p in p_verts])
                          if not poly.is_valid: poly = poly.buffer(0)
                          if poly.is_valid and not poly.is_empty: shapely_polygons.append(poly)
                if not shapely_polygons: merged_regions.append([]); continue
                merged_geom = unary_union([p.buffer(epsilon*10) for p in shapely_polygons]).buffer(-epsilon*10)
                output_merged_pieces = []
                if merged_geom.is_empty: pass
                elif isinstance(merged_geom, Polygon):
                    if merged_geom.is_valid and not merged_geom.is_empty:
                         verts = np.array(merged_geom.exterior.coords)[:-1]
                         if len(verts) >= 3: output_merged_pieces.append(verts)
                elif isinstance(merged_geom, MultiPolygon):
                    for poly in merged_geom.geoms:
                         if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                              verts = np.array(poly.exterior.coords)[:-1]
                              if len(verts) >= 3: output_merged_pieces.append(verts)
                merged_regions.append(output_merged_pieces)
            except Exception:
                merged_regions.append(pieces) # Fallback
        final_regions = merged_regions

    if len(final_regions) != N_original: return None
    return final_regions


def calculate_energy_2d(regions_data, points, width, height,
                        lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                        target_area_func=None, point_weights=None,
                        lambda_min_area=0.0, min_area_threshold=0.0): # <-- ADDED new parameters
    """
    Calculates the total energy of the 2D toroidal tessellation.
    Includes optional minimum area penalty.
    """
    total_energy = 0.0
    energy_components = {'area': 0.0, 'centroid': 0.0, 'angle': 0.0, 'min_area': 0.0} # Added min_area
    num_valid_regions = 0
    num_generators = len(points)
    target_total_area = width * height

    # Weights Preprocessing
    use_weights = False; sum_weights = 0.0
    if target_area_func is None and point_weights is not None:
        point_weights = np.asarray(point_weights)
        if point_weights.shape == (num_generators,) and np.all(point_weights > 0):
            sum_weights = np.sum(point_weights)
            if sum_weights > 1e-9: use_weights = True

    if regions_data is None or len(regions_data) != len(points):
         return np.inf, energy_components

    calculated_areas = [] # Store areas to calculate min_area penalty later

    for i, region_pieces in enumerate(regions_data):
        # Calculate area first for checks and storage
        current_total_area = sum(polygon_area(piece) for piece in region_pieces) if region_pieces else 0.0
        calculated_areas.append(current_total_area)

        if not region_pieces or current_total_area < 1e-12: continue # Skip if no pieces or zero area

        generator_point = points[i]
        num_valid_regions += 1

        # Area Term
        target_area = 1e-9
        if target_area_func: target_area = target_area_func(generator_point)
        elif use_weights: target_area = target_total_area * point_weights[i] / sum_weights
        elif num_generators > 0: target_area = target_total_area / num_generators
        if target_area <= 0: target_area = 1e-9 # Ensure positive
        area_diff_sq = (current_total_area - target_area)**2
        energy_components['area'] += lambda_area * area_diff_sq

        # Centroid Term
        overall_centroid_x, overall_centroid_y, total_weight = 0.0, 0.0, 0.0
        for piece in region_pieces:
             piece_area = polygon_area(piece); piece_centroid = polygon_centroid(piece)
             if piece_area > 1e-12 and piece_centroid is not None:
                 overall_centroid_x += piece_centroid[0] * piece_area; overall_centroid_y += piece_centroid[1] * piece_area; total_weight += piece_area
        if total_weight > 1e-12:
            region_centroid = np.array([overall_centroid_x / total_weight, overall_centroid_y / total_weight])
            centroid_dist_sq = toroidal_distance_sq(generator_point, region_centroid, width, height)
            energy_components['centroid'] += lambda_centroid * centroid_dist_sq

        # Angle Term
        angle_penalty = 0; small_angle_threshold_rad = np.deg2rad(20)
        for piece in region_pieces:
             verts = piece; n_verts = len(verts)
             if n_verts < 3: continue
             for j in range(n_verts):
                 p_prev, p_curr, p_next = verts[j-1], verts[j], verts[(j+1) % n_verts]; v1, v2 = p_prev - p_curr, p_next - p_curr
                 norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                 if norm1 > 1e-12 and norm2 > 1e-12:
                     cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0); angle = np.arccos(cos_theta)
                     if 0 < angle < small_angle_threshold_rad: angle_penalty += (small_angle_threshold_rad - angle)**2
        energy_components['angle'] += lambda_angle * angle_penalty

    # Minimum Area Penalty Term - Calculated after loop using stored areas
    if lambda_min_area > 0 and min_area_threshold > 0:
        for area in calculated_areas:
             # Only apply penalty if area is positive but below threshold
            if 0 < area < min_area_threshold:
                penalty = (min_area_threshold - area)**2
                energy_components['min_area'] += lambda_min_area * penalty

    total_energy = sum(energy_components.values())
    return total_energy, energy_components


def calculate_gradient_2d(points, width, height,
                          lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                          target_area_func=None, point_weights=None,
                          lambda_min_area=0.0, min_area_threshold=0.0, # <-- ADDED new parameters
                          delta=1e-6):
    """
    Calculates the gradient of the 2D energy function using finite differences.
    Passes relevant parameters down to the energy calculation.
    """
    n_points = points.shape[0]
    gradient = np.zeros_like(points, dtype=float)
    points_for_grad = points.copy()

    # Calculate base energy, passing all relevant parameters
    regions_base = generate_voronoi_regions_toroidal(points_for_grad, width, height)
    if regions_base is None: return gradient

    # Pass all parameters to base energy calculation
    energy_base, _ = calculate_energy_2d(
        regions_base, points_for_grad, width, height,
        lambda_area, lambda_centroid, lambda_angle,
        target_area_func, point_weights,
        lambda_min_area, min_area_threshold # Pass new params
    )
    if not np.isfinite(energy_base): return gradient

    for i in range(n_points):
        for j in range(2):
            points_perturbed = points_for_grad.copy()
            points_perturbed[i, j] += delta

            regions_perturbed = generate_voronoi_regions_toroidal(points_perturbed, width, height)
            if regions_perturbed is None: gradient[i, j] = 0.0; continue

            # Calculate perturbed energy, passing all relevant parameters
            energy_perturbed, _ = calculate_energy_2d(
                regions_perturbed, points_perturbed, width, height,
                lambda_area, lambda_centroid, lambda_angle,
                target_area_func, point_weights,
                lambda_min_area, min_area_threshold # Pass new params
            )
            if not np.isfinite(energy_perturbed): gradient[i, j] = 0.0; continue

            gradient[i, j] = (energy_perturbed - energy_base) / delta
    return gradient


def optimize_tessellation_2d(initial_points, width, height, iterations=50, learning_rate=0.1,
                             lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                             target_area_func=None, point_weights=None,
                             lambda_min_area=0.0, min_area_threshold=0.0, # <-- ADDED new parameters
                             verbose=False):
    """
    Optimizes 2D toroidal tessellation using gradient descent.
    Can include minimum area constraint.
    """
    points = initial_points.copy()
    points[:, 0] = np.mod(points[:, 0], width)
    points[:, 1] = np.mod(points[:, 1], height)
    N = len(points)

    # Validate weights
    valid_weights = None
    if target_area_func is None and point_weights is not None:
        point_weights_arr = np.asarray(point_weights)
        if point_weights_arr.shape == (N,) and np.all(point_weights_arr > 0):
            if abs(np.sum(point_weights_arr)) > 1e-9: valid_weights = point_weights_arr

    history = []
    last_successful_regions = None
    points_before_fail = points.copy()
    last_i = 0

    if verbose:
        print(f"Starting 2D optimization: LR={learning_rate}, Iter={iterations}")
        print(f"Lambdas: Area={lambda_area}, Centroid={lambda_centroid}, Angle={lambda_angle}, MinArea={lambda_min_area}")
        if target_area_func: print("Using target area function.")
        elif valid_weights is not None: print(f"Using point weights (Sum: {np.sum(valid_weights):.2f})")
        else: print("Using uniform target area.")
        if lambda_min_area > 0: print(f"Using Min Area Threshold: {min_area_threshold:.4f}")

    for i in range(iterations):
        last_i = i
        regions_current = generate_voronoi_regions_toroidal(points, width, height)
        if regions_current is None:
            if verbose: print(f"Iter {i+1}/{iterations}: Failed Voronoi gen. Stopping.")
            return last_successful_regions, points_before_fail, history
        last_successful_regions = regions_current
        points_before_fail = points.copy()

        # Calculate energy, passing all parameters
        current_energy, energy_components = calculate_energy_2d(
            regions_current, points, width, height,
            lambda_area, lambda_centroid, lambda_angle,
            target_area_func, valid_weights,
            lambda_min_area, min_area_threshold # Pass new params
        )
        history.append(current_energy)

        if verbose:
             valid_comps = {k: v for k, v in energy_components.items() if np.isfinite(v)}
             comp_str_parts = [f"{k.capitalize()}: {v:.4f}" for k, v in valid_comps.items() if k != 'min_area']
             # Only show MinAreaPen if it's active and non-zero
             if 'min_area' in valid_comps and valid_comps['min_area'] > 1e-9 and lambda_min_area > 0:
                 comp_str_parts.append(f"MinAreaPen: {valid_comps['min_area']:.4f}")
             comp_str = ", ".join(comp_str_parts)
             energy_str = f"{current_energy:.4f}" if np.isfinite(current_energy) else f"{current_energy}"
             print(f"Iter {i+1}/{iterations}: Energy={energy_str} ({comp_str})")


        if not np.isfinite(current_energy):
             if verbose: print(f"Iter {i+1}/{iterations}: Energy non-finite. Stopping.")
             return last_successful_regions, points_before_fail, history

        # Calculate gradient, passing all parameters
        grad = calculate_gradient_2d(
            points, width, height,
            lambda_area, lambda_centroid, lambda_angle,
            target_area_func, valid_weights,
            lambda_min_area, min_area_threshold, # Pass new params
            delta=1e-6
        )

        grad_norm = np.linalg.norm(grad)
        if not np.isfinite(grad_norm):
             if verbose: print(f"Iter {i+1}/{iterations}: Gradient non-finite. Stopping.")
             return last_successful_regions, points_before_fail, history
        if grad_norm < 1e-9:
             if verbose: print(f"Iter {i+1}/{iterations}: Gradient norm near zero ({grad_norm:.2e}). Converged/Stuck.")
             break

        points = points - learning_rate * grad
        points[:, 0] = np.mod(points[:, 0], width)
        points[:, 1] = np.mod(points[:, 1], height)

    final_regions = generate_voronoi_regions_toroidal(points, width, height)
    if final_regions is None:
        final_regions = last_successful_regions

    if verbose: print(f"Optimization finished after {last_i+1} iterations.")
    return final_regions, points, history