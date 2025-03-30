import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tessellation_test.src.tessellation import (
    generate_voronoi, voronoi_polygons, compute_total_gradient, update_polygon, polygon_area
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

def main():
    st.title("Optimized Voronoi Tessellation with Radial Gradient and Irregular Tiles")

    num_points = st.slider("Number of Points", 10, 100, 50)
    iterations = st.slider("Iterations", 1, 200, 50)
    learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001, format="%.5f")
    seed = st.number_input("Random Seed", 0, 1000, 42)

    np.random.seed(seed)
    points = np.random.rand(num_points, 2)
    center = np.array([0.5, 0.5])

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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plot_voronoi(polygons, ax)
    ax.set_title("Voronoi Tessellation")
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

if __name__ == "__main__":
    main()