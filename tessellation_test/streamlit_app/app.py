import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tessellation_test.src.tessellation import (
    generate_voronoi, voronoi_polygons, compute_total_gradient, update_polygon
)

def plot_voronoi(polygons, ax):
    for polygon in polygons:
        ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black')

def main():
    st.title("Optimized Voronoi Tessellation with Radial Gradient and Irregular Tiles")

    num_points = st.slider("Number of Points", 10, 100, 50)
    iterations = st.slider("Iterations", 1, 200, 50)

    np.random.seed(42)
    points = np.random.rand(num_points, 2)
    center = np.array([0.5, 0.5])

    vor = generate_voronoi(points)
    polygons = voronoi_polygons(vor)

    for _ in range(iterations):
        updated_polygons = []
        for poly in polygons:
                grad = compute_total_gradient(poly, polygons, center)
                updated_poly = update_polygon(poly, grad)
                updated_poly = np.clip(updated_poly, 0, 1)  # Explicitly clamp vertices
                updated_polygons.append(updated_poly)
        polygons = updated_polygons

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    plot_voronoi(polygons, ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()