import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import sys
import os
import time

# --- Path Setup ---
try:
    # Assume script is run from project root: streamlit run tessellation_test/streamlit_app/app_2d.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from tessellation_test.src.tessellation import generate_voronoi_regions_toroidal, optimize_tessellation_2d
    from utils.geometry import polygon_area, toroidal_distance
except ImportError as e:
    st.error(f"""
    Failed to import project modules.
    Ensure the app is run from the project root directory:
    `streamlit run tessellation_test/streamlit_app/app_2d.py`
    Or ensure the 'tessellation_test' package is installed (e.g., `pip install -e .` from root).
    Error details: {e}
    """)
    st.stop()

# --- Constants ---
# Make domain size configurable? For now, fixed.
WIDTH = 10.0
HEIGHT = 8.0
CENTER = np.array([WIDTH / 2, HEIGHT / 2])
MAX_DIST = np.sqrt((WIDTH/2)**2 + (HEIGHT/2)**2)

# --- Plotting Function (for Streamlit) ---
# (Plotting function remains unchanged from previous version)
def plot_2d_tessellation_st(regions_data, points, width, height, colormap='viridis', title="2D Toroidal Tessellation", show_points=True, show_tiling=False):
    """Plot 2D toroidal tessellation for Streamlit."""
    if not regions_data:
        st.warning("No valid regions data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 8 * height / width)) # Adjust size for Streamlit
    ax.set_aspect('equal', adjustable='box')

    try: cmap = plt.get_cmap(colormap)
    except ValueError: cmap = plt.get_cmap('viridis')

    patches = []
    colors = [] # Based on total area of region
    areas = []  # Store total area per region for color mapping

    for i, region_pieces in enumerate(regions_data):
        # Calculate total area only once per region
        current_total_area = sum(polygon_area(piece) for piece in region_pieces)
        if current_total_area > 1e-9:
            areas.append(current_total_area)
            # Add all pieces for this region to the patch list
            for piece_verts in region_pieces:
                 polygon = MplPolygon(piece_verts, closed=True)
                 patches.append(polygon)
                 # Associate the total area with each piece for consistent coloring
                 colors.append(current_total_area)

    if not patches: return None # No valid polygons

    # Normalize colors based on area
    vmin = min(areas) if areas else 0
    vmax = max(areas) if areas else 1
    if vmin == vmax: # Handle case where all areas are the same
        norm = mcolors.Normalize(vmin=vmin - 0.1 if vmin == 0 else vmin*0.9, vmax=vmax + 0.1 if vmax == 0 else vmax*1.1)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    patch_colors = cmap(norm(colors)) # Get actual colors

    collection = PatchCollection(patches, facecolors=patch_colors, edgecolors='black', lw=0.5, alpha=0.8)
    ax.add_collection(collection)

    if show_tiling:
        # Simplified tiling for Streamlit display (just immediate neighbours)
        for dx_factor in [-1, 1]:
             offset = np.array([dx_factor * width, 0])
             shifted_patches = [MplPolygon(p.get_xy() + offset, closed=True) for p in patches]
             # Re-use patch_colors as the order matches
             shifted_collection = PatchCollection(shifted_patches, facecolors=patch_colors, edgecolors='gray', lw=0.3, alpha=0.4)
             ax.add_collection(shifted_collection)
        for dy_factor in [-1, 1]:
             offset = np.array([0, dy_factor * height])
             shifted_patches = [MplPolygon(p.get_xy() + offset, closed=True) for p in patches]
             shifted_collection = PatchCollection(shifted_patches, facecolors=patch_colors, edgecolors='gray', lw=0.3, alpha=0.4)
             ax.add_collection(shifted_collection)

    if show_points and points is not None:
         ax.scatter(points[:, 0], points[:, 1], c='red', s=10, zorder=5)
         # Optionally show ghost points if tiling is on
         if show_tiling:
              for dx_factor in [-1, 0, 1]:
                 for dy_factor in [-1, 0, 1]:
                     if dx_factor == 0 and dy_factor == 0: continue
                     ax.scatter(points[:, 0] + dx_factor*width, points[:, 1] + dy_factor*height, c='red', s=5, alpha=0.3, zorder=4)

    # Set plot limits
    if show_tiling:
        ax.set_xlim(-0.5*width, 1.5*width)
        ax.set_ylim(-0.5*height, 1.5*height)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

    ax.add_patch(MplPolygon([[0,0], [width,0], [width,height], [0,height]], closed=True, fill=False, edgecolor='blue', lw=1, linestyle='--'))
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array(areas) # Use actual areas array here
    fig.colorbar(sm, ax=ax, label='Region Area', shrink=0.7)

    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🌀 Interactive 2D Toroidal Tessellation")
st.markdown("Generates and optimizes Voronoi tessellations on a rectangle with wrapped boundaries.")

# --- Parameters Sidebar ---
st.sidebar.header("Simulation Parameters")
# Use session state
if 'num_points_2d' not in st.session_state: st.session_state.num_points_2d = 35
if 'random_seed_2d' not in st.session_state: st.session_state.random_seed_2d = 42
if 'run_opt_2d' not in st.session_state: st.session_state.run_opt_2d = True
if 'iters_2d' not in st.session_state: st.session_state.iters_2d = 50
if 'lr_2d' not in st.session_state: st.session_state.lr_2d = 0.8 # Keep default if added to options
if 'lambda_area_2d' not in st.session_state: st.session_state.lambda_area_2d = 1.0
if 'lambda_cent_2d' not in st.session_state: st.session_state.lambda_cent_2d = 0.2
if 'lambda_angle_2d' not in st.session_state: st.session_state.lambda_angle_2d = 0.005
if 'target_choice_2d' not in st.session_state: st.session_state.target_choice_2d = "Smaller near Center" # Default
if 'cmap_2d' not in st.session_state: st.session_state.cmap_2d = 'plasma' # Change default cmap
if 'show_pts_2d' not in st.session_state: st.session_state.show_pts_2d = True
if 'show_tile_2d' not in st.session_state: st.session_state.show_tile_2d = False


st.session_state.num_points_2d = st.sidebar.slider("Number of Points", 5, 200, st.session_state.num_points_2d, 5, key="num_points_slider_2d")
st.session_state.random_seed_2d = st.sidebar.number_input("Random Seed", value=st.session_state.random_seed_2d, key="seed_input_2d")

# Optimization settings...
st.sidebar.header("Optimization Settings")
st.session_state.run_opt_2d = st.sidebar.checkbox("Run Optimization", st.session_state.run_opt_2d, key="run_opt_cb_2d")
st.session_state.iters_2d = st.sidebar.slider("Iterations", 0, 200, st.session_state.iters_2d, 5, disabled=not st.session_state.run_opt_2d, key="iters_slider_2d")

# FIX: Add 0.8 to options and sort them
lr_options = sorted([0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 2.0])
# Ensure default value is actually in the options
default_lr = st.session_state.lr_2d if st.session_state.lr_2d in lr_options else 0.5 # Fallback if current state value is invalid
st.session_state.lr_2d = st.sidebar.select_slider(
    "Learning Rate",
    options=lr_options,
    value=default_lr, # Use corrected default
    disabled=not st.session_state.run_opt_2d,
    key="lr_slider_2d"
)


st.sidebar.subheader("Energy Weights (Lambdas)")
st.session_state.lambda_area_2d = st.sidebar.slider("λ Area", 0.0, 5.0, st.session_state.lambda_area_2d, 0.1, disabled=not st.session_state.run_opt_2d, key="l_area_2d")
st.session_state.lambda_cent_2d = st.sidebar.slider("λ Centroid", 0.0, 5.0, st.session_state.lambda_cent_2d, 0.1, disabled=not st.session_state.run_opt_2d, key="l_cent_2d")
st.session_state.lambda_angle_2d = st.sidebar.slider("λ Angle", 0.0, 0.1, st.session_state.lambda_angle_2d, 0.001, disabled=not st.session_state.run_opt_2d, key="l_angle_2d")

st.sidebar.subheader("Target Area Function")
target_options_2d = ["Uniform Area", "Larger near Center", "Smaller near Center"]
default_target_index = target_options_2d.index(st.session_state.target_choice_2d) if st.session_state.target_choice_2d in target_options_2d else 2 # Fallback index
st.session_state.target_choice_2d = st.sidebar.selectbox(
    "Target Area Profile",
    target_options_2d,
    index=default_target_index,
    disabled=not st.session_state.run_opt_2d,
    key="target_sel_2d"
)

st.sidebar.header("Visualization")
available_colormaps = sorted([m for m in plt.colormaps() if not m.endswith("_r")])
default_cmap_index = available_colormaps.index(st.session_state.cmap_2d) if st.session_state.cmap_2d in available_colormaps else 10 # Fallback index
st.session_state.cmap_2d = st.sidebar.selectbox("Colormap", available_colormaps, index=default_cmap_index, key="cmap_sel_2d")
st.session_state.show_pts_2d = st.sidebar.checkbox("Show Points", st.session_state.show_pts_2d, key="showpts_cb_2d")
st.session_state.show_tile_2d = st.sidebar.checkbox("Show Tiling (Wrap)", st.session_state.show_tile_2d, key="showtile_cb_2d")

# --- Simulation Execution ---
# Initialize session state for results if they don't exist
if 'points_init_2d' not in st.session_state: st.session_state.points_init_2d = None
if 'regions_init_2d' not in st.session_state: st.session_state.regions_init_2d = None
if 'points_opt_2d' not in st.session_state: st.session_state.points_opt_2d = None
if 'regions_opt_2d' not in st.session_state: st.session_state.regions_opt_2d = None
if 'history_2d' not in st.session_state: st.session_state.history_2d = None
if 'metrics_2d' not in st.session_state: st.session_state.metrics_2d = None

# Button to trigger simulation
if st.sidebar.button("🚀 Run Simulation", key="run_button_2d"):
    # Clear previous results before running
    st.session_state.regions_init_2d = None
    st.session_state.regions_opt_2d = None
    st.session_state.history_2d = None
    st.session_state.metrics_2d = None
    st.session_state.points_init_2d = None
    st.session_state.points_opt_2d = None


    # --- Define Target Area Function ---
    num_points = st.session_state.num_points_2d
    target_total_area = WIDTH * HEIGHT
    base_area = target_total_area / num_points if num_points > 0 else target_total_area
    target_choice = st.session_state.target_choice_2d
    run_opt = st.session_state.run_opt_2d
    target_func = None # Initialize
    target_desc = "Uniform Area"

    if run_opt:
        factor = 0.8 # Strength of gradient
        if target_choice == "Uniform Area":
            target_func = lambda p: base_area
        elif target_choice == "Larger near Center": # Smaller away from center
            # FIX: Calculate norm_dist inside the lambda
            target_func = lambda p: base_area * (1 + factor * (1 - (toroidal_distance(p, CENTER, WIDTH, HEIGHT) / MAX_DIST if MAX_DIST > 0 else 0)))
            target_desc = "Larger near Center"
        elif target_choice == "Smaller near Center": # Larger away from center
            # FIX: Calculate norm_dist inside the lambda
            target_func = lambda p: base_area * (1 + factor * (toroidal_distance(p, CENTER, WIDTH, HEIGHT) / MAX_DIST if MAX_DIST > 0 else 0))
            target_desc = "Smaller near Center"
    else:
        target_func = lambda p: base_area # Uniform if not optimizing


    # --- Generate Initial Points ---
    st.info(f"Generating {num_points} initial points (Seed: {st.session_state.random_seed_2d})...")
    time_start_gen = time.time()
    np.random.seed(st.session_state.random_seed_2d)
    points = np.random.rand(num_points, 2)
    points[:, 0] *= WIDTH
    points[:, 1] *= HEIGHT
    st.session_state.points_init_2d = points # Store initial points

    # --- Initial Tessellation ---
    status_placeholder = st.empty()
    with status_placeholder.container(): # Use container for spinner message
        with st.spinner("Calculating initial tessellation..."):
            regions_init = generate_voronoi_regions_toroidal(points, WIDTH, HEIGHT)
    status_placeholder.empty() # Clear message
    st.session_state.regions_init_2d = regions_init
    gen_time = time.time() - time_start_gen

    if regions_init:
         # Check if any region actually got generated (sometimes clipping fails)
         if any(st.session_state.regions_init_2d):
              st.success(f"Generated initial tessellation in {gen_time:.2f}s.")
         else:
              st.warning(f"Initial Voronoi calculation completed in {gen_time:.2f}s, but resulted in empty regions (check for point degeneracy or clipping issues).")
              # Treat as failure for downstream steps
              st.session_state.regions_init_2d = None
              st.stop()
    else:
         st.error(f"Failed to generate initial tessellation in {gen_time:.2f}s.")
         st.stop() # Stop if initial generation failed

    # --- Optimization ---
    if run_opt:
        st.info(f"Optimizing ({st.session_state.iters_2d} iterations, LR={st.session_state.lr_2d}, Target: {target_desc})...")
        time_start_opt = time.time()
        with st.spinner(f"Optimizing ({st.session_state.iters_2d} iterations)..."):
            regions_opt, points_opt, history = optimize_tessellation_2d(
                st.session_state.points_init_2d, WIDTH, HEIGHT,
                iterations=st.session_state.iters_2d,
                learning_rate=st.session_state.lr_2d,
                lambda_area=st.session_state.lambda_area_2d,
                lambda_centroid=st.session_state.lambda_cent_2d,
                lambda_angle=st.session_state.lambda_angle_2d,
                target_area_func=target_func,
                verbose=False # Disable console spam in Streamlit
            )
        opt_time = time.time() - time_start_opt
        st.success(f"Optimization complete in {opt_time:.2f}s.")
        st.session_state.regions_opt_2d = regions_opt
        st.session_state.points_opt_2d = points_opt
        st.session_state.history_2d = history
    else:
        st.info("Optimization skipped.")
        st.session_state.regions_opt_2d = regions_init # Show initial if no opt
        st.session_state.points_opt_2d = points
        st.session_state.history_2d = None


    # --- Calculate Metrics (for the displayed 'optimized' state) ---
    metrics_data = {}
    final_regions_for_metrics = st.session_state.regions_opt_2d if st.session_state.regions_opt_2d else st.session_state.regions_init_2d
    if final_regions_for_metrics:
        areas = [sum(polygon_area(p) for p in pieces) for pieces in final_regions_for_metrics if pieces] # Sum area of pieces per region
        if areas:
            metrics_data['Total Area'] = f"{np.sum(areas):.4f} / {WIDTH*HEIGHT:.4f}"
            metrics_data['Num Regions'] = len(areas)
            metrics_data['Avg Area'] = f"{np.mean(areas):.6f}"
            metrics_data['Std Dev Area'] = f"{np.std(areas):.6f}"
            min_a, max_a = np.min(areas), np.max(areas)
            metrics_data['Min Area'] = f"{min_a:.6f}"
            metrics_data['Max Area'] = f"{max_a:.6f}"
            metrics_data['Area Ratio (Max/Min)'] = f"{max_a/min_a:.3f}" if min_a > 1e-9 else "N/A"
        st.session_state.metrics_2d = metrics_data


# --- Display Results ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Initial Tessellation")
    # Use .get to safely access session state that might not be set yet
    if st.session_state.get('regions_init_2d'):
        fig_init = plot_2d_tessellation_st(
            st.session_state.regions_init_2d,
            st.session_state.get('points_init_2d'), # Use .get for safety
            WIDTH, HEIGHT,
            colormap=st.session_state.cmap_2d,
            title="Initial",
            show_points=st.session_state.show_pts_2d,
            show_tiling=st.session_state.show_tile_2d
        )
        if fig_init: st.pyplot(fig_init)
        else: st.warning("Could not plot initial tessellation.")
    else:
        st.info("Click 'Run Simulation' to generate.")

with col2:
    st.subheader("Final Tessellation")
    if st.session_state.get('regions_opt_2d'):
        fig_opt = plot_2d_tessellation_st(
            st.session_state.regions_opt_2d,
            st.session_state.get('points_opt_2d'),
            WIDTH, HEIGHT,
            colormap=st.session_state.cmap_2d,
            title="Final" + (" (Optimized)" if st.session_state.get('run_opt_2d') else " (Initial)"),
            show_points=st.session_state.show_pts_2d,
            show_tiling=st.session_state.show_tile_2d
        )
        if fig_opt: st.pyplot(fig_opt)
        else: st.warning("Could not plot final tessellation.")
    elif st.session_state.get('run_button_2d'): # Only show warning if simulation was run
         st.warning("No final results available (Optimization might have failed or produced empty regions).")
    else:
         st.info("Click 'Run Simulation' to generate.")


# --- History and Metrics ---
# Make sure history exists and opt was run
if st.session_state.get('run_opt_2d') and st.session_state.get('history_2d'):
    st.subheader("Optimization Energy History")
    hist = st.session_state.history_2d
    if hist: # Check if history list is not empty
         fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
         ax_hist.plot(range(1, len(hist) + 1), hist)
         ax_hist.set_xlabel("Iteration"); ax_hist.set_ylabel("Total Energy")
         ax_hist.set_title("Energy Convergence"); ax_hist.grid(True)
         # Use symlog if energy can be negative or zero, otherwise log
         # Check min energy value
         min_energy = min(hist) if hist else 0
         if min_energy > 1e-9: # Use log if energy stays positive
              ax_hist.set_yscale('log')
         else: # Use linear or symlog otherwise
              ax_hist.set_yscale('linear') # Safest default if values near zero

         ax_hist.set_xlim(left=1)
         st.pyplot(fig_hist)
    else:
         st.info("No optimization history recorded.")

# Display Metrics
st.subheader("Final Tessellation Metrics")
if st.session_state.get('metrics_2d'):
    metrics_data = st.session_state.metrics_2d
    num_metrics = len(metrics_data)
    cols = st.columns(min(num_metrics, 4)) # Max 4 columns
    col_idx = 0
    for key, value in metrics_data.items():
         cols[col_idx % 4].metric(key, value)
         col_idx += 1
elif st.session_state.get('run_button_2d'): # Only show if sim was run
     st.info("No metrics calculated (possibly due to empty regions).")


st.sidebar.markdown("---")
st.sidebar.markdown("2D Toroidal Version")