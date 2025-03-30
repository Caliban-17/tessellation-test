"""
Main entry point for the tessellation algorithm demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tessellation_test.src import tessellation


def plot_2d_tessellation(polygons):
    """Plot a 2D projection of the tessellation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    for polygon in polygons:
        ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("2D Projection of Tessellation")
    plt.savefig("tessellation_2d.png")
    plt.close()

def plot_3d_tessellation(points, zoom_factor=0.5):
    """Plot the tessellation on a 3D sphere."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert 2D points to 3D points on sphere
    sphere_points = tessellation.points_to_sphere(points, zoom_factor=zoom_factor)
    
    # Plot the sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Plot the points
    ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
               c='r', s=50)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_title("3D Spherical Tessellation")
    
    # Rotate view to show central part of sphere
    ax.view_init(elev=30, azim=45)
    
    plt.savefig("tessellation_3d.png")
    plt.close()

def main():
    print("Tessellation Test - Spherical Approach")
    print("--------------------------------------")
    
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
    
    # Check if tiles are interlocking
    interlock_count = 0
    for i, poly1 in enumerate(polygons):
        for j, poly2 in enumerate(polygons):
            if i != j:
                try:
                    from shapely.geometry import Polygon
                    p1 = Polygon(poly1)
                    p2 = Polygon(poly2)
                    if p1.touches(p2) or p1.distance(p2) < 1e-6:
                        interlock_count += 1
                        break
                except:
                    pass
    
    print(f"Number of interlocking tiles: {interlock_count} out of {len(polygons)}")
    
    # Plot the results
    print("\nGenerating visualizations...")
    plot_2d_tessellation(polygons)
    
    # Convert polygons to points for 3D visualization
    points_array = np.array([np.mean(poly, axis=0) for poly in polygons])
    plot_3d_tessellation(points_array)
    
    print("Visualizations saved as tessellation_2d.png and tessellation_3d.png")
    
    print("\nRun 'streamlit run tessellation_test/streamlit_app/app.py' for interactive visualization.")


if __name__ == "__main__":
    main()