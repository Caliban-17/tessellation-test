"""
Main entry point for 2D Toroidal Tessellation demonstration.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# Assumes script is run from project root
from tessellation_test.src.tessellation import generate_voronoi_regions_toroidal, optimize_tessellation_2d
from utils.geometry import polygon_area, toroidal_distance

# --- Domain Parameters ---
WIDTH = 10.0
HEIGHT = 8.0
CENTER = np.array([WIDTH / 2, HEIGHT / 2])

def plot_2d_tessellation(regions_data, points, width, height, filename="tessellation_2d.png", colormap='viridis', title_suffix="", show_points=True, show_tiling=True):
    """
    Plot the 2D toroidal Voronoi tessellation.

    Args:
        regions_data (list): Output from generate_voronoi_regions_toroidal.
        points (np.ndarray): The generator points (N, 2).
        width, height (float): Domain dimensions.
        filename (str): Output filename.
        colormap (str): Matplotlib colormap name.
        title_suffix (str): Suffix for plot title.
        show_points (bool): Whether to plot generator points.
        show_tiling (bool): Whether to show adjacent tiles to visualize wrapping.
    """
    if not regions_data:
        print("Warning: No regions data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10 * height / width)) # Adjust aspect ratio
    ax.set_aspect('equal', adjustable='box')

    try:
        cmap = plt.get_cmap(colormap)
    except ValueError:
        print(f"Warning: Invalid colormap '{colormap}', using 'viridis'.")
        cmap = plt.get_cmap('viridis')


    patches = []
    colors = []
    areas = []

    for i, region_pieces in enumerate(regions_data):
        current_total_area = sum(polygon_area(piece) for piece in region_pieces)
        if current_total_area > 1e-9:
             areas.append(current_total_area)
             for piece_verts in region_pieces:
                  polygon = MplPolygon(piece_verts, closed=True)
                  patches.append(polygon)
                  # Store area corresponding to this patch for coloring
                  # All pieces of a region get same color based on total area
                  colors.append(current_total_area)


    if not patches:
         print("Warning: No valid patches created for plotting.")
         plt.close(fig)
         return

    # Normalize colors based on area
    norm = mcolors.Normalize(vmin=min(areas) if areas else 0, vmax=max(areas) if areas else 1)
    patch_colors = cmap(norm(colors)) # Map normalized areas to colors

    collection = PatchCollection(patches, facecolors=patch_colors, edgecolors='black', lw=0.5, alpha=0.8)
    ax.add_collection(collection)

    # --- Optional: Show Tiling ---
    if show_tiling:
        # Add collections for neighboring tiles to show wrapping
        for dx_factor in [-1, 1]:
            offset = np.array([dx_factor * width, 0])
            shifted_patches = [MplPolygon(p.get_xy() + offset, closed=True) for p in patches]
            shifted_collection = PatchCollection(shifted_patches, facecolors=patch_colors, edgecolors='gray', lw=0.3, alpha=0.4)
            ax.add_collection(shifted_collection)
        for dy_factor in [-1, 1]:
            offset = np.array([0, dy_factor * height])
            shifted_patches = [MplPolygon(p.get_xy() + offset, closed=True) for p in patches]
            shifted_collection = PatchCollection(shifted_patches, facecolors=patch_colors, edgecolors='gray', lw=0.3, alpha=0.4)
            ax.add_collection(shifted_collection)
        # Corner tiles
        for dx_factor in [-1, 1]:
             for dy_factor in [-1, 1]:
                 offset = np.array([dx_factor * width, dy_factor * height])
                 shifted_patches = [MplPolygon(p.get_xy() + offset, closed=True) for p in patches]
                 shifted_collection = PatchCollection(shifted_patches, facecolors=patch_colors, edgecolors='gray', lw=0.3, alpha=0.2)
                 ax.add_collection(shifted_collection)

    # Plot generator points if requested
    if show_points and points is not None:
         ax.scatter(points[:, 0], points[:, 1], c='red', s=10, zorder=5, label='Generators')
         if show_tiling: # Also plot ghost points
             ghost_x, ghost_y = [], []
             for dx_factor in [-1, 1]: ghost_x.extend(points[:, 0] + dx_factor * width)
             for dy_factor in [-1, 1]: ghost_y.extend(points[:, 1] + dy_factor * height)
             for dx_factor in [-1, 0, 1]:
                 for dy_factor in [-1, 0, 1]:
                     if dx_factor == 0 and dy_factor == 0: continue
                     ax.scatter(points[:, 0] + dx_factor*width, points[:, 1] + dy_factor*height, c='red', s=5, alpha=0.3, zorder=4)


    # Set plot limits
    plot_margin_factor = 1.1 if not show_tiling else 1.5 # Larger margin if showing tiling
    ax.set_xlim(-width * (plot_margin_factor-1) , width * plot_margin_factor)
    ax.set_ylim(-height* (plot_margin_factor-1) , height* plot_margin_factor)
    if not show_tiling:
         ax.set_xlim(0, width)
         ax.set_ylim(0, height)


    # Draw main boundary rectangle
    ax.add_patch(MplPolygon([[0,0], [width,0], [width,height], [0,height]], closed=True, fill=False, edgecolor='blue', lw=1.5, linestyle='--'))

    ax.set_title(f'2D Toroidal Voronoi Tessellation {title_suffix}'.strip())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(areas)
    plt.colorbar(sm, ax=ax, label='Region Area', shrink=0.6)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close(fig)


def main():
    print("2D Toroidal Tessellation Demonstration")
    print("--------------------------------------")

    # Parameters
    num_points = 35
    random_seed = 42
    iterations = 60
    learning_rate = 0.8 # May need tuning for 2D
    lambda_area = 1.0
    lambda_centroid = 0.2
    lambda_angle = 0.005 # Angle penalties might be less critical in 2D

    # Target area function: e.g., larger near center
    target_total_area = WIDTH * HEIGHT
    base_area = target_total_area / num_points
    max_dist = np.sqrt((WIDTH/2)**2 + (HEIGHT/2)**2) # Max distance from center
    # Make larger away from center
    target_area_func = lambda p: base_area * (1 + 0.8 * (toroidal_distance(p, CENTER, WIDTH, HEIGHT) / max_dist))


    # Generate initial points
    print(f"Generating {num_points} initial points (Seed={random_seed})...")
    np.random.seed(random_seed)
    points = np.random.rand(num_points, 2)
    points[:, 0] *= WIDTH
    points[:, 1] *= HEIGHT

    # Initial Tessellation
    print("Generating initial tessellation...")
    initial_regions = generate_voronoi_regions_toroidal(points, WIDTH, HEIGHT)
    if initial_regions:
        plot_2d_tessellation(initial_regions, points, WIDTH, HEIGHT, "initial_tessellation_2d.png", title_suffix="(Initial)")
    else:
        print("Failed to generate initial tessellation.")
        return # Stop if initial failed

    # Optimize Tessellation
    print(f"\nOptimizing tessellation ({iterations} iterations, lr={learning_rate})...")
    optimized_regions, final_points, history = optimize_tessellation_2d(
        points, WIDTH, HEIGHT,
        iterations=iterations, learning_rate=learning_rate,
        lambda_area=lambda_area, lambda_centroid=lambda_centroid, lambda_angle=lambda_angle,
        target_area_func=target_area_func, verbose=True
    )

    # Plot Optimized
    if optimized_regions:
        print("Plotting optimized tessellation...")
        plot_2d_tessellation(optimized_regions, final_points, WIDTH, HEIGHT, "optimized_tessellation_2d.png", title_suffix="(Optimized)")
    else:
        print("Optimization failed to produce valid regions.")

    # Metrics
    if optimized_regions:
        print("\nOptimized Tessellation Metrics:")
        areas = [sum(polygon_area(p) for p in pieces) for pieces in optimized_regions if pieces]
        if areas:
             total_area = np.sum(areas)
             print(f"Total area covered: {total_area:.4f} (Expected ~{WIDTH*HEIGHT:.4f})")
             print(f"Average region area: {np.mean(areas):.6f}")
             print(f"Std Dev of area:   {np.std(areas):.6f}")
             min_a, max_a = np.min(areas), np.max(areas)
             print(f"Minimum region area: {min_a:.6f}")
             print(f"Maximum region area: {max_a:.6f}")
             print(f"Area ratio (max/min): {max_a/min_a if min_a > 1e-9 else 'N/A'}")
        else:
             print("No valid regions to calculate metrics.")


if __name__ == "__main__":
    main()