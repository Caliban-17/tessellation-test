import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import SphericalVoronoi
from tessellation_test.src.tessellation import (
    generate_voronoi, voronoi_polygons, compute_total_gradient, update_polygon, polygon_area,
    points_to_sphere, sphere_to_points
)

def plot_voronoi(polygons, ax):
    """
    Plot Voronoi polygons on matplotlib axes.
    
    Parameters:
        polygons: List of polygons to plot
        ax: Matplotlib axes to plot on
    """
    for i, polygon in enumerate(polygons):
        # Use a color map to distinguish polygons
        color = plt.cm.viridis(i / len(polygons))
        ax.fill(*zip(*polygon), alpha=0.7, edgecolor='black', facecolor=color)

def plot_spherical_voronoi(points, ax, zoom_factor=0.5):
    """
    Plot points and their Voronoi cells on a sphere as 2D polygons on the 3D surface.
    
    Parameters:
        points: 2D points to convert to sphere
        ax: Matplotlib 3D axes to plot on
        zoom_factor: How far to zoom in on the sphere (0.5 = hemisphere)
    """
    # Convert 2D points to 3D points on sphere
    sphere_points = points_to_sphere(points, zoom_factor=zoom_factor)
    
    # Create a proper spherical Voronoi diagram
    sv = SphericalVoronoi(sphere_points, radius=1.0, center=np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    
    # Plot the base sphere with light transparency
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Plot the Voronoi cells with different colors
    for i, region in enumerate(sv.regions):
        if not region:
            continue  # Skip empty regions
            
        # Select color from colormap
        color = plt.cm.viridis(i / len(sv.regions))
        
        # Get the vertices of the region
        vertices = sv.vertices[region]
        
        # Plot the edges of the region as 2D polygons on 3D surface
        n = len(vertices)
        for j in range(n):
            # Draw line between consecutive vertices
            line = np.array([vertices[j], vertices[(j+1) % n]])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color='black', linewidth=1.5)
            
        # Create and plot a surface patch for the region
        # This makes a cleaner visualization of the polygon on the sphere
        region_vertices = vertices
        if len(region_vertices) > 2:
            # For regions with at least 3 vertices, create a proper polygon
            for j in range(0, len(region_vertices)-2):
                triangle = np.array([
                    region_vertices[0],
                    region_vertices[j+1],
                    region_vertices[j+2]
                ])
                ax.plot_trisurf(
                    triangle[:, 0], triangle[:, 1], triangle[:, 2],
                    color=color, alpha=0.6
                )
    
    # Plot the sphere points
    ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
               color='red', s=50, zorder=10)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Rotate view for better visibility
    ax.view_init(elev=30, azim=45)

def main():
    st.title("Tessellation Test: Spherical Voronoi with Irregular Tiles")
    
    view_mode = st.radio("Visualization Mode", ["2D Projection", "3D Sphere"])
    
    num_points = st.slider("Number of Points", 5, 50, 15)
    zoom_factor = st.slider("Zoom Factor", 0.1, 1.0, 0.5, 
                           help="Controls how much of the sphere is used (lower = more zoomed in)")
    iterations = st.slider("Iterations", 1, 200, 50)
    learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001, format="%.5f")
    seed = st.number_input("Random Seed", 0, 1000, 42)

    np.random.seed(seed)
    
    # Generate points with more central concentration for better visualization
    points_center = np.random.normal(0.5, 0.15, (int(num_points*0.6), 2))
    points_spread = np.random.rand(num_points - len(points_center), 2)
    points = np.vstack([points_center, points_spread])
    
    # Ensure points stay within [0,1] range
    points = np.clip(points, 0, 1)
    
    center = np.array([0.5, 0.5])
    
    # Generate Voronoi diagram
    vor = generate_voronoi(points)
    polygons = voronoi_polygons(vor)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if st.button("Optimize Tessellation"):
        for i in range(iterations):
            status_text.text(f"Iteration {i+1}/{iterations}")
            progress_bar.progress((i + 1) / iterations)
            
            updated_polygons = []
            for poly in polygons:
                grad = compute_total_gradient(poly, polygons, center)
                updated_poly = update_polygon(poly, grad, learning_rate)
                updated_poly = np.clip(updated_poly, 0, 1)  # Explicitly clamp vertices
                updated_polygons.append(updated_poly)
            polygons = updated_polygons
        
        status_text.text("Optimization complete!")
    
    # Plot visualizations
    if view_mode == "2D Projection":
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Draw the Voronoi polygons
        plot_voronoi(polygons, ax)
        
        # Add the original points
        ax.scatter(points[:, 0], points[:, 1], color='red', s=30, zorder=10)
        
        ax.set_title("2D Projection of Voronoi Tessellation")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    else:
        # 3D Visualization
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw the spherical Voronoi tessellation
        plot_spherical_voronoi(points, ax, zoom_factor)
        
        ax.set_title("3D Spherical Voronoi Tessellation")
        st.pyplot(fig)
    
    # Display some metrics
    st.subheader("Tessellation Metrics")
    
    # Calculate areas
    areas = [polygon_area(poly) for poly in polygons]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Average tile area", f"{avg_area:.6f}")
        st.metric("Min tile area", f"{min_area:.6f}")
    
    with metrics_col2:
        st.metric("Max tile area", f"{max_area:.6f}")
        st.metric("Area ratio (max/min)", f"{max_area/min_area:.2f}" if min_area > 0 else "N/A")
    
    # Add explanation of the approach
    st.subheader("About Spherical Tiling")
    st.write("""
    This visualization demonstrates an irregular Voronoi tessellation on a sphere with no explicit boundaries, 
    simulating the topology of a boundaryless universe. 
    
    The key features are:
    - **Zoomed-in spherical topology**: Points are projected onto a unit sphere, and only a portion is used
    - **Radial area gradients**: Larger central tiles are created based on distance from center
    - **Maximized irregularity**: Tiles have varied sizes and shapes
    - **Interlocking tiles**: All tiles connect properly with no gaps or overlaps
    
    The zoom factor controls how much of the sphere is used - smaller values focus on just a small patch
    of the sphere, completely avoiding boundary effects that would otherwise require special handling.
    """)

if __name__ == "__main__":
    main()