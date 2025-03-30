import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tessellation_test.src.tessellation import (
    generate_voronoi, optimize_tessellation, spherical_polygon_area, 
    spherical_distance, get_region_centroids
)

def plot_spherical_voronoi(ax, regions, colormap='viridis'):
    """
    Plot the spherical Voronoi regions in 3D.
    
    Parameters:
        ax: Matplotlib 3D axes
        regions: List of region vertices
        colormap: Matplotlib colormap name
    """
    # Plot the unit sphere wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    
    # Calculate areas for color mapping
    areas = [spherical_polygon_area(region) for region in regions]
    area_min, area_max = min(areas), max(areas)
    
    # Normalize areas for color mapping
    norm = mcolors.Normalize(vmin=area_min, vmax=area_max)
    
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
        
        # Optionally, plot the region edges
        for j in range(n):
            edge = np.array([region[j], region[(j+1) % n]])
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'k-', lw=0.5, alpha=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Region Area')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Spherical Voronoi Tessellation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_spherical_regions_with_gradients(ax, regions, colormap='Paired'):
    """
    Plot regions with color gradient based on angular distance from reference point.
    
    Parameters:
        ax: Matplotlib 3D axes
        regions: List of region vertices
        colormap: Matplotlib colormap name
    """
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

def main():
    st.title("Spherical Voronoi Tessellation")
    st.write("""
    This application demonstrates a Voronoi tessellation directly on the surface of a 3D sphere.
    The tiles form the outer crust of the sphere, with no projection to 2D.
    """)
    
    # Create sidebar for parameters
    st.sidebar.header("Parameters")
    num_points = st.sidebar.slider("Number of Points", 10, 100, 30)
    iterations = st.sidebar.slider("Optimization Iterations", 0, 200, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    # Random seed for reproducibility
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Visualization options
    st.sidebar.header("Visualization")
    colormap = st.sidebar.selectbox(
        "Color Scheme", 
        ["viridis", "plasma", "inferno", "magma", "cividis", "Paired", "Set3", "tab10"]
    )
    view_mode = st.sidebar.radio(
        "View Type", 
        ["Area-based Coloring", "Distance-based Gradient"]
    )
    
    # Generate initial Voronoi tessellation
    vor = generate_voronoi(num_points=num_points, random_seed=seed)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Optimize button
    if iterations > 0 and st.button("Optimize Tessellation"):
        status_text.text("Optimizing tessellation...")
        
        # Extract initial regions
        initial_regions = []
        for region in vor.regions:
            if region:
                initial_regions.append(vor.vertices[region])
        
        # Initialize regions
        regions = initial_regions.copy()
        
        # Run optimization
        for i in range(iterations):
            progress_bar.progress((i + 1) / iterations)
            
            updated_regions = []
            for vertices in regions:
                # Compute gradient and update vertices
                grad = calculate_gradient(vertices, regions)
                updated_vertices = update_vertices(vertices, grad, learning_rate)
                updated_regions.append(updated_vertices)
            
            regions = updated_regions
            status_text.text(f"Iteration {i+1}/{iterations}")
        
        optimized_regions = regions
        status_text.text("Optimization complete!")
    else:
        # Extract regions for unoptimized visualization
        optimized_regions = []
        for region in vor.regions:
            if region:
                optimized_regions.append(vor.vertices[region])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot spherical Voronoi with appropriate coloring
    if view_mode == "Area-based Coloring":
        plot_spherical_voronoi(ax, optimized_regions, colormap)
    else:
        plot_spherical_regions_with_gradients(ax, optimized_regions, colormap)
    
    # Set the initial view angle
    ax.view_init(elev=30, azim=45)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display tessellation metrics
    st.subheader("Tessellation Metrics")
    
    # Calculate areas
    areas = [spherical_polygon_area(region) for region in optimized_regions]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Area", f"{avg_area:.6f}")
    with col2:
        st.metric("Min Area", f"{min_area:.6f}")
    with col3:
        st.metric("Max Area", f"{max_area:.6f}")
    
    st.metric("Area Ratio (max/min)", f"{max_area/min_area if min_area > 0 else 'N/A'}")
    
    # Information about the spherical tiling approach
    st.subheader("About Spherical Tiling")
    st.write("""
    This visualization demonstrates a true spherical Voronoi tessellation where:
    
    - The tiles exist directly on the sphere's surface (not projected to 2D)
    - Each tile is a polygon on the curved surface of the sphere
    - The collection of tiles form the complete outer crust of the sphere
    - Tile sizes follow a radial gradient from a reference point (like the north pole)
    - The optimization maintains proper spacing and interlocking of tiles
    
    The spherical approach naturally handles tile connectivity without needing special
    boundary conditions, because a sphere has no boundaries.
    """)

if __name__ == "__main__":
    main()