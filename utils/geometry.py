import numpy as np
# Ensure shapely is imported correctly
try:
    from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection, LineString, box
    from shapely.errors import GEOSException
    SHAPELY_AVAILABLE = True
except ImportError:
    print("ERROR: Shapely library not found. Please install it (`pip install shapely`)")
    SHAPELY_AVAILABLE = False
    # Optionally raise the error or exit if shapely is critical
    # raise

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
    # Ensure inputs are numpy arrays
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

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
    vertices = np.asarray(vertices) # Ensure numpy array
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
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
        np.ndarray: Centroid coordinates (x, y), or None if area is zero or invalid input.
    """
    vertices = np.asarray(vertices) # Ensure numpy array
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
        return None # Invalid input

    signed_area = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(vertices)
    for i in range(n):
        x_i = vertices[i, 0]
        y_i = vertices[i, 1]
        x_ip1 = vertices[(i + 1) % n, 0]
        y_ip1 = vertices[(i + 1) % n, 1]
        cross_term = (x_i * y_ip1 - x_ip1 * y_i)
        signed_area += cross_term
        Cx += (x_i + x_ip1) * cross_term
        Cy += (y_i + y_ip1) * cross_term

    if abs(signed_area) < 1e-12:
        # For zero area, return geometric mean as fallback (might be line or point)
        return np.mean(vertices, axis=0)

    signed_area *= 0.5
    # Avoid division by zero if signed_area is somehow still zero after check
    if abs(signed_area) < 1e-12:
        return np.mean(vertices, axis=0)

    Cx /= (6.0 * signed_area)
    Cy /= (6.0 * signed_area)

    return np.array([Cx, Cy])


def wrap_point(point, width, height):
    """Wraps a point coordinates to stay within the [0, width) x [0, height) domain."""
    point = np.asarray(point)
    # Use fmod for floating point modulo which handles negatives correctly for wrapping
    # x = np.fmod(point[0], width)
    # y = np.fmod(point[1], height)
    # return np.array([x + width if x < 0 else x, y + height if y < 0 else y])
    # Simpler: standard modulo works correctly for positive width/height
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
        np.ndarray: Array of shape (9N,) mapping original point index for each row in output.
    """
    points = np.asarray(points)
    N = points.shape[0]
    # Pre-allocate array
    all_points = np.zeros(((9 * N), 2))
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
    Requires Shapely. Returns empty list for invalid input polygons.

    Args:
        polygon_vertices (list or np.ndarray): List of (x, y) vertex coordinates.
        width (float): Width of the boundary box.
        height (float): Height of the boundary box.

    Returns:
        list: A list of numpy arrays, where each array contains the vertices
              of a clipped polygon piece.
    """
    if not SHAPELY_AVAILABLE:
        # print("Warning: Shapely not available, cannot perform polygon clipping.")
        # Return unclipped vertices? Or empty? Empty is safer.
        return []

    # Check for None or insufficient vertices explicitly
    if polygon_vertices is None:
        return []
    # Ensure numpy array and check shape
    polygon_vertices = np.asarray(polygon_vertices)
    if polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2 or polygon_vertices.shape[0] < 3:
        return []

    try:
        # Define the clipping boundary
        boundary = box(0, 0, width, height, ccw=True) # Ensure CCW boundary

        # Create a Shapely polygon from vertices
        poly_verts_list = [tuple(p) for p in polygon_vertices]
        polygon = Polygon(poly_verts_list)

        # FIX: Check validity *before* buffering or intersection.
        if not polygon.is_valid:
            # Optionally try buffering to fix minor issues? Risky for complex invalidity.
            # polygon = polygon.buffer(0)
            # if not polygon.is_valid:
            #     # print(f"Warning: Invalid polygon geometry, cannot clip. Vertices (first 5): {polygon_vertices[:5]}")
            return [] # Return empty list for invalid input geometry

        # Perform the intersection (use buffer(0) on boundary too for robustness?)
        clipped_geom = polygon.intersection(boundary.buffer(0))

        # Process the result
        output_polygons = []
        if clipped_geom.is_empty:
            pass # Return []
        elif isinstance(clipped_geom, Polygon):
            if clipped_geom.is_valid and not clipped_geom.is_empty and clipped_geom.area > 1e-12: # Check area
                 verts = np.array(clipped_geom.exterior.coords)
                 if len(verts) > 1 and np.allclose(verts[0], verts[-1]): verts = verts[:-1]
                 if len(verts) >= 3: output_polygons.append(verts)
        elif isinstance(clipped_geom, (MultiPolygon, GeometryCollection)):
            for geom in clipped_geom.geoms:
                 if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty and geom.area > 1e-12: # Check area
                     verts = np.array(geom.exterior.coords)
                     if len(verts) > 1 and np.allclose(verts[0], verts[-1]): verts = verts[:-1]
                     if len(verts) >= 3: output_polygons.append(verts)

        return output_polygons

    except (GEOSException, Exception) as e:
        # print(f"Error during polygon clipping (Shapely/GEOS): {e}. Polygon vertices (first 5): {polygon_vertices[:5]}")
        return []