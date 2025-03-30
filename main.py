"""
Main entry point for the spherical tessellation algorithm demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tessellation_test.src.tessellation import (
    generate_voronoi, optimize_tessellation, spherical_polygon_area,
    get_region_centroids, spherical_distance
)


def plot_3d_tessellation(regions, filename="tessellation_3d.png", colormap='viridis'):
    """
    Plot the spherical Voronoi tessellation in 3D.
    
    Parameters:
        regions: List of region vertices
        filename: Output filename for the image
        colormap: Matplotlib colormap name
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    
    # Calculate areas for color mapping
    areas = [spherical_polygon_area(region) for region in regions]
    area_min, area_max = min(areas), max(areas)
    
    # Normalize areas for color mapping
    norm = mcolors.Normalize(vmin=area_min, vmax=area_max)
    
    # Plot the unit sphere wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Plot each region
    for i, region in enumerate(regions):
        # Compute region face color based on its area
        area = areas[i]
        color = cmap(norm(area))
        
        # Create triangles from the region vertices
        n = len(region)
        triangles = []
        centroid = np.mean(region, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Project to sphere
        
        for j in range(n):
            triangles.append([centroid, region[j], region[(j+1) % n]])
        
        # Plot each triangle
        for tri in triangles:
            tri_array = np.array(tri)
            ax.plot_trisurf(tri_array[:, 0], tri_array[:, 1], tri_array[:, 2], 
                           color=color, alpha=0.7, shade=True)
        
        # Plot the region edges
        for j in range(n):
            edge = np.array([region[j], region[(j+1) % n]])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'k-', lw=0.5, alpha=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Region Area')
    
    # Set equal aspect ratio and labels
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Spherical Voronoi Tessellation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the initial view angle
    ax.view_init(elev=30, azim=45)
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_gradient_visualization(regions, filename="tessellation_gradient.png", colormap='Paired'):
    """
    Plot the spherical tessellation with colors based on distance from a reference point.
    
    Parameters:
        regions: List of region vertices
        filename: Output filename for the image
        colormap: Matplotlib colormap name
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reference point (north pole)
    ref_point = np.array([0, 0, 1.0])
    
    # Get centroids
    centroids = [np.mean(region, axis=0) / np.linalg.norm(np.mean(region, axis=0)) for region in regions]
    
    # Calculate distances from reference point
    distances = [spherical_distance(centroid, ref_point) for centroid in centroids]
    min_dist, max_dist = min(distances), max(distances)
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=min_dist, vmax=max_dist)
    
    # Plot the unit sphere wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Plot each region with color based on distance
    for i, region in enumerate(regions):
        distance = distances[i]
        color = cmap(norm(distance))
        
        # Create triangles from the region vertices
        n = len(region)
        triangles = []
        centroid = centroids[i]
        
        for j in range(n):
            triangles.append([centroid, region[j], region[(j+1) % n]])
        
        # Plot each triangle
        for tri in triangles:
            tri_array = np.array(tri)
            ax.plot_trisurf(tri_array[:, 0], tri_array[:, 1], tri_array[:, 2], 
                           color=color, alpha=0.7, shade=True)
        
        # Plot the region edges
        for j in range(n):
            edge = np.array([region[j], region[(j+1) % n]])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'k-', lw=0.5, alpha=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Angular Distance from North Pole')
    
    # Set equal aspect ratio and labels
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Spherical Voronoi Tessellation with Distance Gradient')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the initial view angle
    ax.view_init(elev=30, azim=45)
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Spherical Tessellation - Direct 3D Approach")
    print("-------------------------------------------")
    
    # Generate initial tessellation
    print("Generating spherical Voronoi tessellation...")
    num_points = 30
    random_seed = 42
    
    vor = generate_voronoi(num_points=num_points, random_seed=random_seed)
    
    # Extract regions
    initial_regions = []
    for region in vor.regions:
        if region:
            initial_regions.append(vor.vertices[region])
    
    print(f"Generated tessellation with {len(initial_regions)} regions")
    
    # Plot initial tessellation
    print("Creating initial visualization...")
    plot_3d_tessellation(initial_regions, "initial_tessellation.png")
    print("Initial tessellation saved as 'initial_tessellation.png'")
    
    # Optimize tessellation
    print("\nOptimizing tessellation...")
    iterations = 50
    learning_rate = 0.001
    
    optimized_regions = optimize_tessellation(vor, iterations=iterations, learning_rate=learning_rate)
    print(f"Completed {iterations} optimization iterations")
    
    # Plot optimized tessellation
    print("Creating optimized visualization...")
    plot_3d_tessellation(optimized_regions, "optimized_tessellation.png")
    print("Optimized tessellation saved as 'optimized_tessellation.png'")
    
    # Plot gradient visualization
    print("Creating gradient visualization...")
    plot_3d_gradient_visualization(optimized_regions)
    print("Gradient visualization saved as 'tessellation_gradient.png'")
    
    # Calculate and print metrics
    print("\nTessellation Metrics:")
    areas = [spherical_polygon_area(region) for region in optimized_regions]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    print(f"Average region area: {avg_area:.6f}")
    print(f"Minimum region area: {min_area:.6f}")
    print(f"Maximum region area: {max_area:.6f}")
    print(f"Area ratio (max/min): {max_area/min_area if min_area > 0 else 'N/A'}")
    
    # Check region connectivity
    print("\nChecking region connectivity...")
    connected_count = 0
    for i, region1 in enumerate(optimized_regions):
        for j, region2 in enumerate(optimized_regions):
            if i >= j:  # Skip self and already checked pairs
                continue
                
            # Calculate minimum distance between regions
            min_distance = float('inf')
            for v1 in region1:
                for v2 in region2:
                    dist = np.linalg.norm(v1 - v2)
                    min_distance = min(min_distance, dist)
            
            # If regions are very close, consider them connected
            if min_distance < 0.01:
                connected_count += 1
    
    print(f"Number of connected region pairs: {connected_count}")
    print(f"Connectivity ratio: {connected_count / (len(optimized_regions) * (len(optimized_regions) - 1) / 2):.2f}")
    
    print("\nRun 'streamlit run tessellation_test/streamlit_app/app.py' for interactive visualization.")


if __name__ == "__main__":
    main()