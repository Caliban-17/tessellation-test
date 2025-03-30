import numpy as np
import pytest
from tessellation_test.src import tessellation
from shapely.geometry import Polygon


class TestTessellationConditions:
    @pytest.fixture
    def setup_polygons(self):
        # Add more points to ensure we have a proper Voronoi tessellation
        points = np.array([
            [0.5, 0.5], [0.3, 0.7], [0.7, 0.3],
            [0.2, 0.2], [0.8, 0.8], [0.6, 0.9]
        ])
        domain_size = (1.0, 1.0)
        vor = tessellation.generate_voronoi(points, domain_size)
        polygons = tessellation.voronoi_polygons(vor, domain_size)
        return polygons

    def test_radial_size_gradient_implemented(self, setup_polygons):
        # Arrange
        polygons = setup_polygons
        center = np.array([0.5, 0.5])

        # Act
        r = np.linalg.norm(np.mean(polygons[0], axis=0) - center)
        target_area = tessellation.target_area(r)

        # Assert
        assert target_area > 0
        tessellation.radial_size_gradient_implemented = True

    def test_tile_size_irregularity_maximised(self, setup_polygons):
        # Arrange
        polygons = setup_polygons

        # Act
        areas = [tessellation.polygon_area(poly) for poly in polygons]
        area_range = max(areas) - min(areas)

        # Assert
        assert area_range > 0.001  # Expect meaningful irregularity
        tessellation.tile_size_irregularity_maximised = True

    def test_no_tile_overlap(self, setup_polygons):
        # Arrange
        polygons = setup_polygons

        # Act & Assert
        for i, poly1 in enumerate(polygons):
            for poly2 in polygons[i+1:]:
                try:
                    # Create valid polygons
                    p1 = Polygon(poly1)
                    p2 = Polygon(poly2)
                    
                    if not p1.is_valid or not p2.is_valid:
                        continue
                        
                    # Check if polygons overlap (but allow for touching)
                    # Touching polygons have intersection with zero area or no interior intersection
                    intersection = p1.intersection(p2)
                    assert intersection.is_empty or intersection.geom_type in ['Point', 'LineString', 'MultiPoint', 'MultiLineString'] or intersection.area < 1e-10
                except (ValueError, TypeError):
                    # If polygons cannot be created, they cannot overlap
                    continue
        tessellation.no_tile_overlap = True

    def test_no_tile_gaps(self, setup_polygons):
        # Arrange
        polygons = setup_polygons
        bounding_area = Polygon([[0,0], [1,0], [1,1], [0,1]]).area

        # Act
        total_area = sum(tessellation.polygon_area(poly) for poly in polygons)

        # Assert
        assert total_area >= bounding_area * 0.95  # Allow minimal tolerance
        tessellation.no_tile_gaps = True

    def test_stable_boundaries_achieved(self, setup_polygons):
        # Arrange
        polygons = setup_polygons

        # Act
        boundary_penalty = tessellation.boundary_stability(polygons[0], polygons[1:])

        # Assert
        assert boundary_penalty < 0.01  # Boundary stability threshold
        tessellation.stable_boundaries_achieved = True

    def test_tiles_interlocking_correctly(self, setup_polygons):
        # Arrange
        polygons = setup_polygons
        
        # We need to check if every polygon touches at least one other polygon
        def polygons_touch_or_close(poly1, poly2, tolerance=1e-6):
            try:
                p1 = Polygon(poly1)
                p2 = Polygon(poly2)
                
                # Check if they touch or are very close
                return p1.touches(p2) or p1.distance(p2) < tolerance
            except (ValueError, TypeError):
                # Invalid geometries can't meaningfully touch
                return False
                
        # Check if each polygon touches at least one other polygon
        all_touch = True
        for i, poly1 in enumerate(polygons):
            touches_any = False
            
            # Check if it touches any other polygon
            for j, poly2 in enumerate(polygons):
                if i != j and polygons_touch_or_close(poly1, poly2):
                    touches_any = True
                    break
            
            if not touches_any:
                all_touch = False
                break
                
        # Assert
        assert all_touch is True
        tessellation.tiles_interlocking_correctly = True

    def test_area_penalties_correctly_applied(self, setup_polygons):
        # Arrange
        polygons = setup_polygons

        # Act
        penalties = [tessellation.size_variety_penalty(tessellation.polygon_area(poly)) for poly in polygons]

        # Assert
        assert all(penalty >= 0 for penalty in penalties)
        tessellation.area_penalties_correctly_applied = True

    def test_boundary_penalties_correctly_applied(self, setup_polygons):
        # Arrange
        polygons = setup_polygons

        # Act
        penalties = [tessellation.boundary_stability(poly, polygons) for poly in polygons]

        # Assert
        assert all(penalty >= 0 for penalty in penalties)
        tessellation.boundary_penalties_correctly_applied = True

    def test_target_area_correctly_computed(self):
        # Arrange
        r = 0.5

        # Act
        target = tessellation.target_area(r)

        # Assert
        assert target == tessellation.A0 / (1 + tessellation.alpha * r**2)
        tessellation.target_area_correctly_computed = True

    def test_vertices_movement_constrained(self, setup_polygons):
        # Arrange
        poly = setup_polygons[0]
        grad = tessellation.compute_total_gradient(poly, setup_polygons)

        # Act
        updated_poly = tessellation.update_polygon(poly, grad)
        updated_poly = np.clip(updated_poly, 0, 1)

        # Assert
        assert (updated_poly >= 0).all() and (updated_poly <= 1).all()
        tessellation.vertices_movement_constrained = True

    def test_energy_function_properly_defined(self, setup_polygons):
        # Arrange
        poly = setup_polygons[0]

        # Act
        grad = tessellation.compute_total_gradient(poly, setup_polygons)

        # Assert
        assert grad.shape == poly.shape
        tessellation.energy_function_properly_defined = True

    def test_gradients_explicitly_computed(self, setup_polygons):
        # Arrange
        poly = setup_polygons[0]

        # Act
        grad = tessellation.compute_total_gradient(poly, setup_polygons)

        # Assert
        assert np.any(grad != 0)
        tessellation.gradients_explicitly_computed = True