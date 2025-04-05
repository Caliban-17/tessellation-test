import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi # Rename to avoid conflict
from utils.geometry import (
    toroidal_distance_sq, polygon_area, polygon_centroid, wrap_point,
    generate_ghost_points, clip_polygon_to_boundary
)

# Note: Voronoi calculation and clipping require Shapely library

def generate_voronoi_regions_toroidal(points, width, height):
    """
    Generates Voronoi regions for points on a 2D torus [0, W] x [0, H].

    Args:
        points (np.ndarray): Array of shape (N, 2) of generator points in [0, W]x[0, H].
        width (float): Width of the torus domain.
        height (float): Height of the torus domain.

    Returns:
        list: A list of lists of numpy arrays. Each inner list corresponds to an
              original point. Each numpy array within the inner list contains the
              vertices (N, 2) of a polygon piece making up the Voronoi region
              clipped to the boundary [0, width] x [0, height].
              Returns None if Voronoi calculation fails.
    """
    if points is None or len(points) < 1:
        print("Error: No points provided for Voronoi generation.")
        return None
    if width <= 0 or height <= 0:
        print("Error: Width and height must be positive.")
        return None

    N_original = points.shape[0]

    # Ensure points are within the primary domain (optional, depends on caller)
    # points[:, 0] = np.mod(points[:, 0], width)
    # points[:, 1] = np.mod(points[:, 1], height)

    try:
        # 1. Generate ghost points for toroidal wrapping
        all_points, original_indices = generate_ghost_points(points, width, height)

        # 2. Compute standard Voronoi diagram on the 3x3 grid of points
        # Add buffer points far away if needed by qhull to prevent errors with few points
        # buffer_dist = max(width, height) * 2
        # buffer_points = np.array([
        #     [-buffer_dist, -buffer_dist], [buffer_dist, -buffer_dist],
        #     [-buffer_dist, buffer_dist], [buffer_dist, buffer_dist]
        # ]) * 5 # Even further
        # combined_points_for_voronoi = np.vstack((all_points, buffer_points))

        vor = ScipyVoronoi(all_points) # Use all_points directly

    except Exception as e:
        print(f"Error during Scipy Voronoi calculation: {e}")
        # Common issue: QhullError if points are degenerate (collinear, coincident)
        # Check for duplicate points:
        unique_points, counts = np.unique(points, axis=0, return_counts=True)
        if len(unique_points) < len(points):
             print(f"Warning: Duplicate points detected ({len(points) - len(unique_points)} duplicates). Voronoi may fail or be ill-defined.")
        # Check for collinear points? Harder.
        return None # Indicate failure

    # 3. Process regions and clip them to the boundary [0, W] x [0, H]
    clipped_regions_all_points = [[] for _ in range(len(all_points))] # Store potentially multi-piece regions

    # Map regions from Voronoi object back to the generating points
    point_to_region_map = vor.point_region # Index is point index, value is region index

    for point_idx, region_idx in enumerate(point_to_region_map):
        if region_idx == -1: # Point is outside the Voronoi diagram (shouldn't happen for valid input?)
            continue

        region_vertex_indices = vor.regions[region_idx]

        if not region_vertex_indices or -1 in region_vertex_indices:
             # Region is infinite (extends to boundary of Voronoi diagram bounds)
             # Clipping should handle this, but we need finite vertices.
             # Reconstruct using ridges? Complex.
             # Easier: Clip the polygon defined by finite vertices. SciPy usually gives finite vertices.
             finite_vertex_indices = [idx for idx in region_vertex_indices if idx != -1]
             if len(finite_vertex_indices) < 3:
                 continue # Not enough vertices for a polygon
             polygon_vertices = vor.vertices[finite_vertex_indices]

        else:
            # Region is finite
            polygon_vertices = vor.vertices[region_vertex_indices]

        if len(polygon_vertices) < 3:
            continue # Skip degenerate regions

        # Clip this polygon to the boundary [0, W] x [0, H]
        clipped_pieces = clip_polygon_to_boundary(polygon_vertices, width, height)

        # Store the clipped pieces associated with this point index (from all_points)
        clipped_regions_all_points[point_idx] = clipped_pieces


    # 4. Consolidate clipped regions for the *original* points
    # A region for an original point can be made of pieces from its 9 ghost copies.
    final_regions = [[] for _ in range(N_original)]
    for all_points_idx, original_idx in enumerate(original_indices):
         # Add the clipped pieces associated with this ghost point instance
         # to the list for the corresponding original point.
         final_regions[original_idx].extend(clipped_regions_all_points[all_points_idx])

    # Optional: Merge contiguous polygon pieces for the same original point using shapely?
    # For now, return list of pieces per original point. The visualization layer can draw them all.

    # Sanity check: ensure output list has correct length
    if len(final_regions) != N_original:
         print(f"Warning: Mismatch in final region count ({len(final_regions)}) vs original points ({N_original}).")
         # Fallback or error? Return None for safety.
         return None

    return final_regions


def calculate_energy_2d(regions_data, points, width, height,
                        lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                        target_area_func=None):
    """
    Calculates the total energy of the 2D toroidal tessellation.

    Args:
        regions_data (list): Output from generate_voronoi_regions_toroidal.
                             List (size N) of lists of polygon vertex arrays.
        points (np.ndarray): The generator points (N, 2).
        width, height (float): Domain dimensions.
        lambda_area, lambda_centroid, lambda_angle: Weights.
        target_area_func (callable, optional): func(point_xy) -> target_area.

    Returns:
        float: Total energy.
        dict: Dictionary of energy components.
    """
    total_energy = 0.0
    energy_components = {'area': 0.0, 'centroid': 0.0, 'angle': 0.0}
    num_valid_regions = 0
    num_generators = len(points)
    target_total_area = width * height

    if regions_data is None:
         return np.inf, energy_components # Voronoi failed

    for i, region_pieces in enumerate(regions_data):
        if not region_pieces: continue # Skip if point has no associated region pieces after clipping

        generator_point = points[i]
        current_total_area = sum(polygon_area(piece) for piece in region_pieces)

        if current_total_area < 1e-12: continue # Skip zero-area regions

        num_valid_regions += 1

        # --- Area Term ---
        if target_area_func:
            target_area = target_area_func(generator_point)
            if target_area <= 0:
                 target_area = target_total_area / num_generators # Fallback
        else:
            target_area = target_total_area / num_generators # Default uniform

        area_diff_sq = (current_total_area - target_area)**2
        energy_components['area'] += lambda_area * area_diff_sq

        # --- Centroid Term ---
        # Calculate overall centroid of potentially multiple pieces (weighted average)
        overall_centroid_x, overall_centroid_y = 0.0, 0.0
        total_weight = 0.0
        valid_centroid_calc = True
        for piece in region_pieces:
             piece_area = polygon_area(piece)
             piece_centroid = polygon_centroid(piece)
             if piece_area > 1e-12 and piece_centroid is not None:
                 overall_centroid_x += piece_centroid[0] * piece_area
                 overall_centroid_y += piece_centroid[1] * piece_area
                 total_weight += piece_area
             else:
                 # Handle degenerate pieces if necessary
                 pass

        if total_weight > 1e-12:
            region_centroid = np.array([overall_centroid_x / total_weight, overall_centroid_y / total_weight])
            # Use TOROIDAL distance squared for energy penalty
            centroid_dist_sq = toroidal_distance_sq(generator_point, region_centroid, width, height)
            energy_components['centroid'] += lambda_centroid * centroid_dist_sq
        else:
            # Cannot calculate valid centroid if total area of pieces is zero
            energy_components['centroid'] += 0 # Or assign a penalty?

        # --- Angle Term (Simpler 2D version) ---
        angle_penalty = 0
        small_angle_threshold_rad = np.deg2rad(20) # Penalize angles < 20 degrees
        for piece in region_pieces:
             verts = piece
             n_verts = len(verts)
             if n_verts < 3: continue
             for j in range(n_verts):
                 p_prev = verts[j-1]
                 p_curr = verts[j]
                 p_next = verts[(j+1) % n_verts]
                 # Vectors along edges meeting at p_curr
                 v1 = p_prev - p_curr
                 v2 = p_next - p_curr
                 norm1 = np.linalg.norm(v1)
                 norm2 = np.linalg.norm(v2)
                 if norm1 > 1e-12 and norm2 > 1e-12:
                     cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                     cos_theta = np.clip(cos_theta, -1.0, 1.0)
                     angle = np.arccos(cos_theta)
                     if 0 < angle < small_angle_threshold_rad:
                         angle_penalty += (small_angle_threshold_rad - angle)**2
                 # else: Handle collinear/coincident points - angle is 0 or pi, likely not penalized


        energy_components['angle'] += lambda_angle * angle_penalty


    total_energy = sum(energy_components.values())
    # Optional normalization by num_valid_regions here

    return total_energy, energy_components


def calculate_gradient_2d(points, width, height,
                          lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                          target_area_func=None, delta=1e-6):
    """
    Calculates the gradient of the 2D energy function using finite differences.

    Args:
        points (np.ndarray): Current generator points (N, 2).
        width, height (float): Domain dimensions.
        lambda_area, lambda_centroid, lambda_angle: Weights.
        target_area_func (callable, optional): Target area function.
        delta (float): Step size for finite differences.

    Returns:
        np.ndarray: Gradient vector of shape (N, 2). Returns zeros if fails.
    """
    n_points = points.shape[0]
    gradient = np.zeros_like(points, dtype=float)

    # Calculate base energy
    regions_base = generate_voronoi_regions_toroidal(points, width, height)
    if regions_base is None:
        print("Warning: Base Voronoi failed in gradient calculation.")
        return gradient # Zero gradient

    energy_base, _ = calculate_energy_2d(regions_base, points, width, height,
                                         lambda_area, lambda_centroid, lambda_angle,
                                         target_area_func)

    if not np.isfinite(energy_base):
        print(f"Warning: Base energy non-finite ({energy_base}). Zero gradient.")
        return gradient

    for i in range(n_points):
        for j in range(2): # Iterate x and y dimensions
            points_perturbed = points.copy()
            points_perturbed[i, j] += delta
            # Wrap perturbed point back into domain? Optional, toroidal distance handles it.
            # points_perturbed[i] = wrap_point(points_perturbed[i], width, height)

            # Calculate perturbed energy
            regions_perturbed = generate_voronoi_regions_toroidal(points_perturbed, width, height)
            if regions_perturbed is None:
                 print(f"Warning: Perturbed Voronoi failed (Point {i}, Dim {j}). Grad comp=0.")
                 gradient[i, j] = 0.0
                 continue

            energy_perturbed, _ = calculate_energy_2d(regions_perturbed, points_perturbed, width, height,
                                                     lambda_area, lambda_centroid, lambda_angle,
                                                     target_area_func)

            if not np.isfinite(energy_perturbed):
                 print(f"Warning: Perturbed energy non-finite (P {i}, D {j}). Grad comp=0.")
                 gradient[i, j] = 0.0
                 continue

            # Forward difference
            gradient[i, j] = (energy_perturbed - energy_base) / delta

    return gradient


def optimize_tessellation_2d(initial_points, width, height, iterations=50, learning_rate=0.1,
                             lambda_area=1.0, lambda_centroid=0.1, lambda_angle=0.01,
                             target_area_func=None, verbose=False):
    """
    Optimizes 2D toroidal tessellation using gradient descent.

    Args:
        initial_points (np.ndarray): Starting points (N, 2) in [0,W]x[0,H].
        width, height (float): Domain dimensions.
        iterations (int): Number of steps.
        learning_rate (float): Step size.
        lambda_area, lambda_centroid, lambda_angle: Weights.
        target_area_func (callable, optional): Target area function.
        verbose (bool): Print progress.

    Returns:
        tuple: (final_regions, final_points, history)
            - final_regions (list): Result from generate_voronoi_regions_toroidal.
            - final_points (np.ndarray): Optimized points (N, 2).
            - history (list): Energy values over iterations.
    """
    points = initial_points.copy()
    # Ensure points start within domain
    points[:, 0] = np.mod(points[:, 0], width)
    points[:, 1] = np.mod(points[:, 1], height)

    history = []
    last_successful_regions = None
    points_before_fail = points.copy()

    print(f"Starting 2D optimization: LR={learning_rate}, Iter={iterations}")
    print(f"Lambdas: Area={lambda_area}, Centroid={lambda_centroid}, Angle={lambda_angle}")

    for i in range(iterations):
        # 1. Generate current regions (needed for energy calculation if logging)
        regions_current = generate_voronoi_regions_toroidal(points, width, height)
        if regions_current is None:
            print(f"Iter {i+1}/{iterations}: Failed Voronoi gen. Stopping.")
            return last_successful_regions, points_before_fail, history
        last_successful_regions = regions_current
        points_before_fail = points.copy()

        # 2. Calculate energy (for history/logging)
        current_energy, energy_components = calculate_energy_2d(
            regions_current, points, width, height,
            lambda_area, lambda_centroid, lambda_angle, target_area_func
        )
        history.append(current_energy)

        if verbose:
             comp_str = ", ".join([f"{k.capitalize()}: {v:.4f}" for k, v in energy_components.items()])
             print(f"Iter {i+1}/{iterations}: Energy={current_energy:.4f} ({comp_str})")

        if not np.isfinite(current_energy):
             print(f"Iter {i+1}/{iterations}: Energy non-finite. Stopping.")
             return last_successful_regions, points_before_fail, history

        # 3. Calculate gradient
        grad = calculate_gradient_2d(
            points, width, height, lambda_area, lambda_centroid, lambda_angle,
            target_area_func, delta=1e-6
        )

        grad_norm = np.linalg.norm(grad)
        if not np.isfinite(grad_norm):
             print(f"Iter {i+1}/{iterations}: Gradient non-finite. Stopping.")
             return last_successful_regions, points_before_fail, history
        if grad_norm < 1e-8:
             print(f"Iter {i+1}/{iterations}: Gradient norm near zero ({grad_norm:.2e}). Converged/Stuck.")
             break

        # 4. Update points
        points = points - learning_rate * grad

        # 5. Wrap points back into the fundamental domain [0, W] x [0, H]
        points[:, 0] = np.mod(points[:, 0], width)
        points[:, 1] = np.mod(points[:, 1], height)

    # Final regions calculation
    final_regions = generate_voronoi_regions_toroidal(points, width, height)
    if final_regions is None:
        print("Warning: Failed to generate final regions after optimization.")
        final_regions = last_successful_regions # Return last good one

    print(f"Optimization finished after {i+1} iterations.")
    return final_regions, points, history