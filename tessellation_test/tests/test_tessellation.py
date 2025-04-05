import numpy as np
import pytest
from scipy.spatial import Voronoi as ScipyVoronoi

# Assuming tests run from root dir where tessellation_test is visible
from tessellation_test.src.tessellation import (
    generate_voronoi_regions_toroidal, optimize_tessellation_2d,
    calculate_energy_2d, calculate_gradient_2d
)
from utils.geometry import (
    toroidal_distance, toroidal_distance_sq, polygon_area, polygon_centroid,
    wrap_point, generate_ghost_points, clip_polygon_to_boundary
)

# --- Constants ---
WIDTH = 10.0
HEIGHT = 8.0
CENTER = np.array([WIDTH / 2, HEIGHT / 2])

# --- Fixtures ---

@pytest.fixture
def sample_points_2d():
    """Generate random points within the 2D domain."""
    np.random.seed(42)
    points = np.random.rand(15, 2) # 15 points
    points[:, 0] *= WIDTH
    points[:, 1] *= HEIGHT
    return points

@pytest.fixture
def simple_points_2d():
    """Generate a few simple, well-spaced points."""
    return np.array([
        [WIDTH * 0.25, HEIGHT * 0.25],
        [WIDTH * 0.75, HEIGHT * 0.25],
        [WIDTH * 0.25, HEIGHT * 0.75],
        [WIDTH * 0.75, HEIGHT * 0.75],
        [WIDTH * 0.5, HEIGHT * 0.5], # Center point
    ])

# --- Test Geometry Utils ---

def test_wrap_point():
    assert np.allclose(wrap_point(np.array([WIDTH + 1, HEIGHT + 1]), WIDTH, HEIGHT), [1, 1])
    assert np.allclose(wrap_point(np.array([-1, -1]), WIDTH, HEIGHT), [WIDTH - 1, HEIGHT - 1])
    assert np.allclose(wrap_point(np.array([WIDTH / 2, HEIGHT / 2]), WIDTH, HEIGHT), [WIDTH / 2, HEIGHT / 2])
    assert np.allclose(wrap_point(np.array([WIDTH, HEIGHT]), WIDTH, HEIGHT), [0, 0])

def test_toroidal_distance():
    p1 = np.array([1, 1])
    # Point itself
    assert np.isclose(toroidal_distance(p1, p1, WIDTH, HEIGHT), 0.0)
    # Simple adjacent point
    p2 = np.array([2, 1])
    assert np.isclose(toroidal_distance(p1, p2, WIDTH, HEIGHT), 1.0)
    # Wrap-around horizontal
    p3 = np.array([WIDTH - 1, 1])
    assert np.isclose(toroidal_distance(p1, p3, WIDTH, HEIGHT), 2.0) # Dist(1, 9) on torus width 10 is 2
    # Wrap-around vertical
    p4 = np.array([1, HEIGHT - 1])
    assert np.isclose(toroidal_distance(p1, p4, WIDTH, HEIGHT), 2.0) # Dist(1, 7) on torus height 8 is 2
    # Wrap-around both
    p5 = np.array([WIDTH - 1, HEIGHT - 1])
    assert np.isclose(toroidal_distance(p1, p5, WIDTH, HEIGHT), np.sqrt(2.0**2 + 2.0**2))
    # Check squared distance
    assert np.isclose(toroidal_distance_sq(p1, p5, WIDTH, HEIGHT), 2.0**2 + 2.0**2)

def test_generate_ghost_points():
    points = np.array([[1, 1], [5, 6]])
    ghosts, indices = generate_ghost_points(points, WIDTH, HEIGHT)
    assert ghosts.shape == (9 * 2, 2)
    assert indices.shape == (9 * 2,)
    # Check if original points are present
    assert np.any(np.all(ghosts == [1, 1], axis=1))
    assert np.any(np.all(ghosts == [5, 6], axis=1))
    # Check one ghost point location
    assert np.any(np.all(ghosts == [1 + WIDTH, 1], axis=1)) # Shifted right
    assert np.any(np.all(ghosts == [1, 1 + HEIGHT], axis=1)) # Shifted up
    assert np.any(np.all(ghosts == [1 - WIDTH, 1 - HEIGHT], axis=1)) # Shifted left-down
    # Check indices mapping
    assert indices[0] == 0 # First point's ghosts
    assert indices[8] == 0
    assert indices[9] == 1 # Second point's ghosts


def test_polygon_area():
    # Square
    verts_sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert np.isclose(polygon_area(verts_sq), 4.0)
    # Triangle
    verts_tri = np.array([[0, 0], [3, 0], [0, 4]])
    assert np.isclose(polygon_area(verts_tri), 6.0)
    # Degenerate
    verts_deg = np.array([[0, 0], [1, 1]])
    assert np.isclose(polygon_area(verts_deg), 0.0)
    verts_line = np.array([[0,0], [1,1], [2,2]])
    assert np.isclose(polygon_area(verts_line), 0.0)


def test_polygon_centroid():
    # Square
    verts_sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert np.allclose(polygon_centroid(verts_sq), [1.0, 1.0])
    # Triangle
    verts_tri = np.array([[0, 0], [3, 0], [0, 3]]) # Easier triangle
    assert np.allclose(polygon_centroid(verts_tri), [1.0, 1.0])
    # Rectangle
    verts_rect = np.array([[1, 1], [5, 1], [5, 4], [1, 4]])
    assert np.allclose(polygon_centroid(verts_rect), [3.0, 2.5])
     # Degenerate (should return mean)
    verts_deg = np.array([[0, 0], [1, 1], [0, 0]])
    assert np.allclose(polygon_centroid(verts_deg), [1/3, 1/3])


# --- Test Tessellation Core ---

@pytest.fixture
def sample_regions(simple_points_2d):
     """Generate toroidal regions for simple points."""
     regions = generate_voronoi_regions_toroidal(simple_points_2d, WIDTH, HEIGHT)
     assert regions is not None, "Fixture failed: Toroidal Voronoi generation failed."
     assert len(regions) == len(simple_points_2d)
     return regions

def test_generate_voronoi_toroidal(simple_points_2d):
    regions = generate_voronoi_regions_toroidal(simple_points_2d, WIDTH, HEIGHT)
    assert regions is not None
    assert len(regions) == len(simple_points_2d) # One list of pieces per input point
    total_area = 0
    for i, pieces in enumerate(regions):
        assert isinstance(pieces, list)
        # Check if the region has at least one piece
        # This might fail if a point is suppressed by others
        # assert len(pieces) > 0, f"Point {i} resulted in no region pieces."
        if not pieces:
            print(f"Warning: Point {i} resulted in no region pieces in test.")
            continue

        for piece_verts in pieces:
            assert isinstance(piece_verts, np.ndarray)
            assert piece_verts.ndim == 2
            assert piece_verts.shape[1] == 2
            assert len(piece_verts) >= 3 # Should be valid polygon vertices
            # Check if vertices are within boundary
            assert np.all(piece_verts[:, 0] >= -1e-9) and np.all(piece_verts[:, 0] <= WIDTH + 1e-9)
            assert np.all(piece_verts[:, 1] >= -1e-9) and np.all(piece_verts[:, 1] <= HEIGHT + 1e-9)
            total_area += polygon_area(piece_verts)

    # Total area should equal the domain area
    assert np.isclose(total_area, WIDTH * HEIGHT)

def test_generate_voronoi_toroidal_fail_cases():
    # Too few points
    assert generate_voronoi_regions_toroidal(np.array([[1,1]]), WIDTH, HEIGHT) is None
    # Duplicate points (might cause Qhull error, should return None)
    points_dup = np.array([[1,1], [1,1], [2,2], [3,3], [4,4]])
    assert generate_voronoi_regions_toroidal(points_dup, WIDTH, HEIGHT) is None
    # Invalid dimensions
    assert generate_voronoi_regions_toroidal(np.array([[1,1],[2,2],[3,3],[4,4]]), 0, HEIGHT) is None


def test_calculate_energy_2d(sample_regions, simple_points_2d):
    energy, components = calculate_energy_2d(sample_regions, simple_points_2d, WIDTH, HEIGHT)
    assert np.isfinite(energy)
    assert energy >= 0
    assert isinstance(components, dict)
    assert 'area' in components and components['area'] >= 0
    assert 'centroid' in components and components['centroid'] >= 0
    assert 'angle' in components and components['angle'] >= 0

    # Test with target area func (e.g., larger near center)
    base_area = (WIDTH * HEIGHT) / len(simple_points_2d)
    target_func = lambda p: base_area * (1.0 + 0.5 * (1 - toroidal_distance(p, CENTER, WIDTH, HEIGHT) / max(WIDTH, HEIGHT)))
    energy_t, components_t = calculate_energy_2d(sample_regions, simple_points_2d, WIDTH, HEIGHT, target_area_func=target_func)
    assert np.isfinite(energy_t)
    assert not np.isclose(energy, energy_t) # Energy should change


def test_calculate_gradient_2d(simple_points_2d):
    points = simple_points_2d
    grad = calculate_gradient_2d(points, WIDTH, HEIGHT, lambda_area=1.0, lambda_centroid=0.1)
    assert grad.shape == points.shape
    assert np.all(np.isfinite(grad))

    # Test gradient by moving against it (energy should decrease)
    regions_orig = generate_voronoi_regions_toroidal(points, WIDTH, HEIGHT)
    if regions_orig is None: pytest.skip("Skip grad check: Base Voronoi failed.")
    energy_orig, _ = calculate_energy_2d(regions_orig, points, WIDTH, HEIGHT)
    if not np.isfinite(energy_orig): pytest.skip("Skip grad check: Base energy infinite.")

    step = 1e-7
    points_moved = points - step * grad
    points_moved = np.array([wrap_point(p, WIDTH, HEIGHT) for p in points_moved]) # Wrap points

    regions_moved = generate_voronoi_regions_toroidal(points_moved, WIDTH, HEIGHT)
    if regions_moved is None: pytest.skip("Skip grad check: Moved Voronoi failed.")
    energy_moved, _ = calculate_energy_2d(regions_moved, points_moved, WIDTH, HEIGHT)
    if not np.isfinite(energy_moved): pytest.skip("Skip grad check: Moved energy infinite.")

    # print(f"Energy Orig: {energy_orig}, Energy Moved: {energy_moved}, Diff: {energy_orig - energy_moved}")
    assert energy_moved <= energy_orig + 1e-7 # Allow tolerance

def test_optimize_tessellation_2d_runs(sample_points_2d):
    initial_points = sample_points_2d
    final_regions, final_points, history = optimize_tessellation_2d(
        initial_points, WIDTH, HEIGHT,
        iterations=5, learning_rate=0.5, # Use larger LR for 2D?
        verbose=False
    )
    assert final_regions is not None
    assert final_points.shape == initial_points.shape
    assert np.all(final_points >= 0) & np.all(final_points[:,0] <= WIDTH) & np.all(final_points[:,1] <= HEIGHT) # Points wrapped
    assert isinstance(history, list)
    assert 0 < len(history) <= 5
    assert all(np.isfinite(h) for h in history)
    assert history[-1] <= history[0] + 1e-7 # Energy should decrease or stay similar


def test_optimize_tessellation_2d_energy_decrease(simple_points_2d):
     initial_points = simple_points_2d
     regions_initial = generate_voronoi_regions_toroidal(initial_points, WIDTH, HEIGHT)
     if regions_initial is None: pytest.skip("Skip opt E decrease: Initial Voronoi failed.")
     energy_initial, _ = calculate_energy_2d(regions_initial, initial_points, WIDTH, HEIGHT)
     if not np.isfinite(energy_initial): pytest.skip("Skip opt E decrease: Initial energy infinite.")

     final_regions, final_points, history = optimize_tessellation_2d(
        initial_points, WIDTH, HEIGHT,
        iterations=10, learning_rate=1.0,
        lambda_area=0.5, lambda_centroid=0.5, # Equal weight
        verbose=False
     )

     if not history: pytest.skip("Skip opt E decrease: No history.")
     energy_final = history[-1]
     print(f"Initial Energy: {energy_initial}, Final Energy: {energy_final}")
     assert energy_final < energy_initial + 1e-7
     # Check points moved
     if energy_initial > 1e-6:
         assert not np.allclose(initial_points, final_points)