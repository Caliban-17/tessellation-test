import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from shapely.geometry import Polygon

# Parameters (fine-tune as needed)
λ1, λ2, λ3, λ4, λ5, λ7 = 1, 1, 1, 1, 1, 1
A0, alpha = 0.03, 10       # Radial size gradient (larger center tiles)
A_min, A_max = 0.005, 0.1  # Encourage variety in tile sizes explicitly

def generate_voronoi(points, domain_size=(1.0, 1.0)):
    points_wrapped = points % domain_size
    return Voronoi(points_wrapped, qhull_options='QJ')

def voronoi_polygons(vor):
    polygons = []
    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 not in vertices and vertices:
            polygon = np.array([vor.vertices[i] for i in vertices])
            polygons.append(polygon)
    return polygons

def polygon_area(poly):
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
    return A0 / (1 + alpha * r**2)

def angle_penalty(poly):
    angles = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1, 1))
        angles.append(angle)
    return np.sum(1 / np.array(angles))

def boundary_stability(poly_i, other_polys):
    min_dist = min(
        distance_matrix(poly_i, poly_j).min()
        for poly_j in other_polys if not np.array_equal(poly_i, poly_j)
    )
    return min_dist**2

def size_variety_penalty(area, A_min=A_min, A_max=A_max):
    if A_min <= area <= A_max:
        return 0  # No penalty if within the desired range (maximizing variety explicitly)
    return min((area - A_min)**2, (area - A_max)**2)

def centroid_gradient(poly, centroid_target):
    centroid = np.mean(poly, axis=0)
    return centroid - centroid_target

def compute_total_gradient(poly, all_polys, center=(0.5, 0.5)):
    centroid = np.mean(poly, axis=0)
    r = np.linalg.norm(centroid - center)
    area = polygon_area(poly)

    grad = np.zeros_like(poly)

    # Area gradient: moves vertices toward or away from centroid gently
    area_diff = (area - target_area(r))
    grad += λ1 * area_diff * (poly - centroid)

    # Overlap and gap prevention (constrained)
    for neighbor in all_polys:
        if not np.array_equal(poly, neighbor):
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
    boundary_pen = boundary_stability(poly, all_polys)
    grad += λ5 * boundary_pen * (poly - centroid)

    # Shape regularity (centroid pull gently)
    grad += λ7 * (poly - centroid) * 0.1

    # Normalize gradient to prevent extreme moves
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    norm[norm == 0] = 1
    grad_normalized = grad / norm

    return grad_normalized
def update_polygon(poly, grad_normalised, learning_rate=0.0001):
    # Reduce learning rate significantly
    return poly - learning_rate * grad_normalised