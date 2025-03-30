import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    for polygon in polygons:
        ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black')

def plot_spherical_voronoi(points, ax, zoom_factor=0.5):
    """
    Plot points on a sphere to visualize the spherical Voronoi tessellation.
    
    Parameters:
        points: 2D points to convert to sphere
        ax: Matplotlib 3D axes to plot on
        zoom_factor: How far to zoom in on the sphere (0.5 = hemisphere)
    """
    # Convert 2D points to 3D points on sphere
    sphere_points = points_to_sphere(points, zoom_factor=zoom_factor)
    
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Rotate view to show central part of sphere
    ax.view_init(elev=30, azim=45)

def main():
    st.title("Spherical Voronoi Tessellation with Radial Gradient and Irregular Tiles")
    
    view_mode = st.radio("Visualization Mode", ["2D Projection", "3D Sphere"])
    
    num_points = st.slider("Number of Points", 10, 100, 50)
    zoom_factor = st.slider("Zoom Factor", 0.1, 1.0, 0.5, 
                           help="Controls how much of the sphere is used (lower = more zoomed in)")
    iterations = st.slider("Iterations", 1, 200, 50)
    learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001, format="%.5f")
    seed = st.number_input("Random Seed", 0, 1000, 42)

    np.random.seed(seed)
    points = np.random.rand(num_points, 2)
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
    
    if view_mode == "2D Projection":
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plot_voronoi(polygons, ax)
        ax.set_title("2D Projection of Voronoi Tessellation")
        st.pyplot(fig)
    else:
        # 3D Visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Use centroids of polygons for visualization
        points_array = np.array([np.mean(p, axis=0) for p in polygons])
        plot_spherical_voronoi(points_array, ax, zoom_factor)
        ax.set_title("3D Spherical Voronoi Tessellation")
        st.pyplot(fig)
    
    # Display some metrics
    st.subheader("Tessellation Metrics")
    
    # Calculate areas
    areas = [polygon_area(poly) for poly in polygons]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    st.write(f"Average tile area: {avg_area:.6f}")
    st.write(f"Min tile area: {min_area:.6f}")
    st.write(f"Max tile area: {max_area:.6f}")
    st.write(f"Area ratio (max/min): {max_area/min_area if min_area > 0 else 'N/A'}")
    
    # Add explanation of spherical tiling
    st.subheader("About Spherical Tiling")
    st.write("""
    This visualization demonstrates a Voronoi tessellation on a sphere. The "zoomed-in" approach means
    we're only using a portion of the sphere's surface, which avoids the need to handle wrapping at domain
    boundaries.
    
    By working on a sphere, we create a naturally continuous surface with no edges. The zoom factor
    controls how much of the sphere is used - smaller values focus on just a small patch of the sphere,
    completely avoiding boundary issues.
    
    The tessellation follows these principles:
    - Radial gradient applies based on the distance from the center
    - Tiles have irregular sizes but maintain connectivity
    - No overlaps or gaps between tiles
    - No special handling of boundary conditions needed
    """)

if __name__ == "__main__":
    main()