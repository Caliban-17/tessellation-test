import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from shapely.geometry import Polygon

# Parameters (fine-tune as needed)
λ1, λ2, λ3, λ4, λ5, λ7 = 1, 1, 1, 1, 1, 1
A0, alpha = 0.03, 10       # Radial size gradient (larger center tiles)
A_min, A_max = 0.005, 0.1  # Encourage variety in tile sizes explicitly

def generate_voronoi(points, domain_size=(1.0, 1.0)):
    """
    Generate a Voronoi diagram from a set of points.
    
    Parameters:
        points: Array of points
        domain_size: Size of the domain (width, height)
        
    Returns:
        Voronoi diagram
    """
    points_wrapped = points.copy() % domain_size
    return Voronoi(points_wrapped, qhull_options='QJ')

def voronoi_polygons(vor, domain_size=(1.0, 1.0)):
    """
    Extract polygon vertices from Voronoi diagram.
    
    Parameters:
        vor: Voronoi diagram object from scipy.spatial
        domain_size: Size of the domain (width, height)
        
    Returns:
        List of polygons, where each polygon is an array of vertex coordinates
    """
    polygons = []
    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 not in vertices and vertices:
            polygon = np.array([vor.vertices[i] for i in vertices])
            # Ensure vertices are within domain
            polygon = np.clip(polygon, 0, domain_size)
            polygons.append(polygon)
    return polygons

def polygon_area(poly):
    """
    Calculate the area of a polygon.
    
    Parameters:
        poly: Array of vertex coordinates
        
    Returns:
        Area of the polygon
    """
    # Explicitly ensure polygon is closed
    if len(poly) < 3:
        return 0.0  # Polygon with fewer than 3 vertices is invalid
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    try:
        shapely_poly = Polygon(poly)
        if not shapely_poly.is_valid:
            return 0.0
        return shapely_poly.area
    except Exception:
        return 0.0

def target_area(r, A0=A0, alpha=alpha):
    """
    Calculate the target area for a polygon based on distance from center.
    
    Parameters:
        r: Distance from center
        A0: Base area
        alpha: Radial gradient factor
        
    Returns:
        Target area
    """
    return A0 / (1 + alpha * r**2)

def angle_penalty(poly):
    """
    Calculate a penalty based on the angles in a polygon.
    
    Parameters:
        poly: Array of vertex coordinates
        
    Returns:
        Angle penalty value
    """
    angles = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        ba, bc = a - b, c - b
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm == 0 or bc_norm == 0:
            continue
        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        angle = np.arccos(np.clip(cosine_angle, -1, 1))
        angles.append(angle)
    
    if not angles:
        return 0.0
        
    # Avoid division by zero
    angles = np.array(angles)
    angles = np.maximum(angles, 1e-6)
    return np.sum(1 / angles)

def boundary_stability(poly_i, other_polys):
    """
    Calculate boundary stability between a polygon and others.
    
    Parameters:
        poly_i: The polygon to check
        other_polys: List of other polygons
        
    Returns:
        Boundary stability measure
    """
    if not other_polys:
        return 0.0
        
    min_distances = []
    for poly_j in other_polys:
        if np.array_equal(poly_i, poly_j):
            continue
        dist = distance_matrix(poly_i, poly_j)
        min_distances.append(dist.min())
    
    if not min_distances:
        return 0.0
        
    return min(min_distances) ** 2

def size_variety_penalty(area, A_min=A_min, A_max=A_max):
    """
    Calculate penalty for polygon size outside desired range.
    
    Parameters:
        area: Polygon area
        A_min: Minimum desired area
        A_max: Maximum desired area
        
    Returns:
        Size variety penalty
    """
    if A_min <= area <= A_max:
        return 0  # No penalty if within the desired range (maximizing variety explicitly)
    return min((area - A_min)**2, (area - A_max)**2)

def centroid_gradient(poly, centroid_target):
    """
    Calculate gradient towards a target centroid.
    
    Parameters:
        poly: Array of vertex coordinates
        centroid_target: Target centroid coordinates
        
    Returns:
        Gradient
    """
    centroid = np.mean(poly, axis=0)
    return centroid - centroid_target

def compute_total_gradient(poly, all_polys, center=(0.5, 0.5)):
    """
    Compute the total gradient for a polygon.
    
    Parameters:
        poly: Array of vertex coordinates
        all_polys: List of all polygons
        center: Center point for radial gradient
        
    Returns:
        Total gradient for polygon vertices
    """
    centroid = np.mean(poly, axis=0)
    r = np.linalg.norm(centroid - center)
    area = polygon_area(poly)

    grad = np.zeros_like(poly)

    # Area gradient: moves vertices toward or away from centroid gently
    area_diff = (area - target_area(r))
    grad += λ1 * area_diff * (poly - centroid)

    # Overlap and gap prevention (constrained)
    other_polys = [p for p in all_polys if not np.array_equal(p, poly)]
    for neighbor in other_polys:
        dist = distance_matrix(poly, neighbor)
        min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
        direction = poly[min_dist_idx[0]] - neighbor[min_dist_idx[1]]
        distance = np.linalg.norm(direction)
        if distance > 0:
            grad[min_dist_idx[0]] += λ2 * (direction / distance) * np.exp(-distance)

    # Angle penalty (small adjustment)
    angle_pen = angle_penalty(poly)
    grad += λ3 * angle_pen * (poly - centroid)

    # Area variety penalty (gentle enforcement)
    variety_pen = size_variety_penalty(area)
    grad += λ4 * variety_pen * (poly - centroid)

    # Boundary stability (small correction)
    boundary_pen = boundary_stability(poly, other_polys)
    grad += λ5 * boundary_pen * (poly - centroid)

    # Shape regularity (centroid pull gently)
    grad += λ7 * (poly - centroid) * 0.1

    # Normalize gradient to prevent extreme moves
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    mask = norm == 0
    norm[mask] = 1
    grad_normalized = grad / norm

    return grad_normalized

def update_polygon(poly, grad_normalized, learning_rate=0.0001):
    """
    Update polygon vertices using gradient descent.
    
    Parameters:
        poly: Array of vertex coordinates
        grad_normalized: Normalized gradient
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Updated polygon
    """
    # Reduce learning rate significantly
    return poly - learning_rate * grad_normalized

def compute_energy_gradient(vertex, polygon, center=(0.5, 0.5), learning_rate=0.01):
    """
    Compute the energy gradient for a single vertex.
    
    Parameters:
        vertex: The vertex to compute the gradient for
        polygon: The polygon containing the vertex
        center: The center point for radial gradient
        learning_rate: The learning rate for gradient descent
        
    Returns:
        The computed gradient
    """
    centroid = np.mean(polygon, axis=0)
    r = np.linalg.norm(centroid - center)
    area = polygon_area(polygon)
    
    # Simplified gradient computation for a single vertex
    gradient = np.zeros_like(vertex)
    
    # Area gradient: moves vertex toward or away from centroid gently
    area_diff = (area - target_area(r))
    gradient += λ1 * area_diff * (vertex - centroid)
    
    # Angle penalty component
    angles = []
    vertex_idx = None
    for i, p in enumerate(polygon):
        if np.array_equal(p, vertex):
            vertex_idx = i
            break
            
    if vertex_idx is not None:
        n = len(polygon)
        a, b, c = polygon[(vertex_idx - 1) % n], vertex, polygon[(vertex_idx + 1) % n]
        ba, bc = a - b, c - b
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm > 0 and bc_norm > 0:  # Avoid division by zero
            cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
            angle = np.arccos(np.clip(cosine_angle, -1, 1))
            if angle > 0:  # Avoid division by zero
                gradient += λ3 * (1 / angle) * (vertex - centroid)
    
    # Size variety penalty
    variety_pen = size_variety_penalty(area)
    gradient += λ4 * variety_pen * (vertex - centroid)
    
    # Normalize gradient to prevent extreme moves
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient = gradient / norm
    
    return gradient

def update_vertex(vertex, gradient, learning_rate=0.01):
    """
    Update a vertex position using the computed gradient.
    
    Parameters:
        vertex: The vertex to update
        gradient: The computed gradient
        learning_rate: The learning rate for the update
        
    Returns:
        The updated vertex position
    """
    updated_vertex = vertex - learning_rate * gradient
    # Ensure the vertex stays within bounds
    updated_vertex = np.clip(updated_vertex, 0, 1)
    return updated_vertex