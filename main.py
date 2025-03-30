"""
Main entry point for the tessellation algorithm demonstration.
"""

import numpy as np
from tessellation_test.src import tessellation


def main():
    # Define a simple polygon (triangle) for demonstration.
    polygon = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ])

    # Choose a vertex to update (e.g., the first vertex)
    vertex = polygon[0]
    print("Initial vertex:", vertex)

    # Compute the energy gradient for the vertex.
    gradient = tessellation.compute_energy_gradient(vertex, polygon)
    print("Computed gradient:", gradient)

    # Update the vertex position using the computed gradient.
    updated_vertex = tessellation.update_vertex(vertex, gradient)
    print("Updated vertex:", updated_vertex)

    # Demonstrate full tessellation optimization
    print("\nDemonstrating full tessellation optimization:")
    # Generate random points
    np.random.seed(42)
    num_points = 20
    points = np.random.rand(num_points, 2)
    
    # Generate initial Voronoi tessellation
    vor = tessellation.generate_voronoi(points)
    polygons = tessellation.voronoi_polygons(vor)
    
    print(f"Initial tessellation with {len(polygons)} polygons")
    
    # Run a few optimization iterations
    iterations = 10
    for i in range(iterations):
        updated_polygons = []
        for poly in polygons:
            grad = tessellation.compute_total_gradient(poly, polygons)
            updated_poly = tessellation.update_polygon(poly, grad)
            updated_poly = np.clip(updated_poly, 0, 1)
            updated_polygons.append(updated_poly)
        polygons = updated_polygons
        
        # Print progress
        if (i+1) % 5 == 0:
            print(f"Completed {i+1} iterations")
    
    print("Optimization complete")
    
    # Calculate and print some metrics
    areas = [tessellation.polygon_area(poly) for poly in polygons]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    print(f"Average tile area: {avg_area:.6f}")
    print(f"Min tile area: {min_area:.6f}")
    print(f"Max tile area: {max_area:.6f}")
    print(f"Area ratio (max/min): {max_area/min_area if min_area > 0 else 'N/A'}")


if __name__ == "__main__":
    main()