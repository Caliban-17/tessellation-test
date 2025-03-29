"""
Main entry point for the tessellation algorithm demonstration.
"""

import numpy as np
from tessellation_test.src import tessellation as tessellation


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


if __name__ == "__main__":
    main()