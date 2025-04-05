import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection, LineString, box

def toroidal_distance_sq(p1, p2, width, height):
    """
    Calculates the squared shortest distance between two points on a 2D torus.
    Using squared distance avoids sqrt for efficiency, often sufficient for comparisons/gradients.

    Args:
        p1 (np.ndarray): First point (x, y).
        p2 (np.ndarray): Second point (x, y).
        width (float): Width of the torus domain.
        height (float): Height of the torus domain.

    Returns:
        float: Squared toroidal distance.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    # Account for wrap-around
    delta_x = min(abs(dx), width - abs(dx))
    delta_y = min(abs(dy), height - abs(dy))

    return delta_x**2 + delta_y**2

def toroidal_distance(p1, p2, width, height):
    """Calculates the shortest distance between two points on a 2D torus."""
    return np.sqrt(toroidal_distance_sq(p1, p2, width, height))

def polygon_area(vertices):
    """
    Calculates the area of a 2D polygon using the Shoelace formula.
    Assumes vertices are ordered (clockwise or counter-clockwise).

    Args:
        vertices (np.ndarray): Array of shape (N, 2) representing N ordered vertices.

    Returns:
        float: Area of the polygon (always positive).
    """
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    # Apply Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def polygon_centroid(vertices):
    """
    Calculates the centroid of a 2D polygon.

    Args:
        vertices (np.ndarray): Array of shape (N, 2) representing N ordered vertices.

    Returns:
        np.ndarray: Centroid coordinates (x, y), or None if area is zero.
    """
    if len(vertices) < 3:
        return None
    area_val = polygon_area(vertices)
    if abs(area_val) < 1e-12: # Avoid division by zero for degenerate polygons
        # Return geometric mean for degenerate cases? Or center of bbox?
        return np.mean(vertices, axis=0) # Simple mean as fallback

    x = vertices[:, 0]
    y = vertices[:, 1]
    # Shifted coordinates for formula stability with large coordinates? Not strictly needed here.
    x_shifted = x #- np.mean(x)
    y_shifted = y #- np.mean(y)

    # Apply centroid formula components
    a = np.dot(x_shifted, np.roll(y_shifted, 1)) - np.dot(y_shifted, np.roll(x_shifted, 1))
    cx_term = (x_shifted + np.roll(x_shifted, 1)) * a
    cy_term = (y_shifted + np.roll(y_shifted, 1)) * a

    centroid_x = np.sum(cx_term) / (6.0 * area_val)
    centroid_y = np.sum(cy_term) / (6.0 * area_val)

    # Add back mean if subtracted earlier
    # centroid_x += np.mean(x)
    # centroid_y += np.mean(y)

    # Handle potential sign issue from Shoelace area calculation / vertex order
    # Recalculate area with consistent order if needed, but centroid formula handles it.
    # Let's recheck the standard formula:
    signed_area = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(len(vertices)):
        x_i = vertices[i, 0]
        y_i = vertices[i, 1]
        x_ip1 = vertices[(i + 1) % len(vertices), 0]
        y_ip1 = vertices[(i + 1) % len(vertices), 1]
        cross_term = (x_i * y_ip1 - x_ip1 * y_i)
        signed_area += cross_term
        Cx += (x_i + x_ip1) * cross_term
        Cy += (y_i + y_ip1) * cross_term

    if abs(signed_area) < 1e-12:
        return np.mean(vertices, axis=0) # Fallback for zero area

    signed_area *= 0.5
    Cx /= (6.0 * signed_area)
    Cy /= (6.0 * signed_area)

    return np.array([Cx, Cy])


def wrap_point(point, width, height):
    """Wraps a point coordinates to stay within the [0, width) x [0, height) domain."""
    return np.array([point[0] % width, point[1] % height])

def generate_ghost_points(points, width, height):
    """
    Generates 8 'ghost' copies of each point, shifted for the toroidal topology.

    Args:
        points (np.ndarray): Array of shape (N, 2) of original points.
        width (float): Width of the domain.
        height (float): Height of the domain.

    Returns:
        np.ndarray: Array of shape (9N, 2) containing original and ghost points.
        np.ndarray: Array of shape (N,) mapping original point index for each row in output.
                    (Useful for associating Voronoi regions back to original points)
    """
    N = points.shape[0]
    all_points = np.zeros(((3 * 3 * N), 2))
    original_indices = np.zeros(9 * N, dtype=int)

    idx = 0
    for i, p in enumerate(points):
        for dx_factor in [-1, 0, 1]:
            for dy_factor in [-1, 0, 1]:
                all_points[idx, 0] = p[0] + dx_factor * width
                all_points[idx, 1] = p[1] + dy_factor * height
                original_indices[idx] = i
                idx += 1

    return all_points, original_indices

def clip_polygon_to_boundary(polygon_vertices, width, height):
    """
    Clips a polygon (defined by vertices) to the rectangular boundary [0, W] x [0, H].
    Requires Shapely.

    Args:
        polygon_vertices (list or np.ndarray): List of (x, y) vertex coordinates.
        width (float): Width of the boundary box.
        height (float): Height of the boundary box.

    Returns:
        list: A list of numpy arrays, where each array contains the vertices
              of a clipped polygon piece. Returns an empty list if the polygon
              is outside the boundary or clipping fails. Returns a list with
              one element if the polygon is fully contained or clipped cleanly.
              Can return multiple polygons if the original polygon crosses the
              boundary multiple times in a complex way after Voronoi generation.
    """
    if polygon_vertices is None or len(polygon_vertices) < 3:
        return []

    try:
        # Define the clipping boundary
        boundary = box(0, 0, width, height)

        # Create a Shapely polygon
        # Add closing vertex if not present (shapely doesn't strictly require it)
        if not np.allclose(polygon_vertices[0], polygon_vertices[-1]):
             poly_verts_closed = list(polygon_vertices) + [polygon_vertices[0]]
        else:
             poly_verts_closed = list(polygon_vertices)

        # Check for sufficient vertices and validity before creating polygon
        if len(poly_verts_closed) < 4: # Need at least 3 unique points for a polygon (4 verts closed)
             # print("Warning: Not enough vertices for Shapely polygon.")
             return []

        # Attempt to create the Shapely polygon
        polygon = Polygon(poly_verts_closed)

        # Check if the polygon is valid (e.g., not self-intersecting in problematic ways)
        if not polygon.is_valid:
            # Try to buffer slightly to fix minor validity issues
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                # print(f"Warning: Invalid polygon geometry encountered, cannot clip. Vertices: {polygon_vertices}")
                return [] # Cannot proceed with invalid geometry


        # Perform the intersection
        clipped_geom = polygon.intersection(boundary)

        # Process the result (can be Polygon, MultiPolygon, or empty)
        output_polygons = []
        if clipped_geom.is_empty:
            return []
        elif isinstance(clipped_geom, Polygon):
            # Extract vertices, ensuring correct format (list of points)
            verts = np.array(clipped_geom.exterior.coords)
            output_polygons.append(verts[:-1]) # Exclude duplicate closing vertex
        elif isinstance(clipped_geom, (MultiPolygon, GeometryCollection)):
            # Handle multiple disjoint pieces resulting from clipping
            for geom in clipped_geom.geoms:
                 if isinstance(geom, Polygon) and not geom.is_empty:
                     verts = np.array(geom.exterior.coords)
                     output_polygons.append(verts[:-1]) # Exclude duplicate closing vertex
        elif isinstance(clipped_geom, (LineString, Point)):
             # Intersection resulted in lower dimension object (e.g., only touches boundary)
             # print("Warning: Polygon intersection resulted in LineString or Point.")
             return []
        else:
            # print(f"Warning: Unexpected geometry type after clipping: {type(clipped_geom)}")
            return []

        return output_polygons

    except Exception as e:
        # print(f"Error during polygon clipping: {e}. Polygon vertices: {polygon_vertices}")
        return [] # Return empty list on error