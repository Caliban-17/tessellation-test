# Tessellation Test

## 📐 Overview

**Tessellation Test** is an experimental Python project designed to generate and optimize **irregular Voronoi tessellations** with no explicit boundaries — simulating the topology of a **boundaryless universe**. The algorithm emphasizes:

- 📊 Radial area gradients (larger central tiles)
- 🔀 Maximized irregularity in tile sizes
- 🧩 Interlocking tiles without gaps or overlaps
- 🔍 Spherical topology for seamless continuity

The engine uses a mathematically grounded energy function based on multivariate calculus and gradient descent to achieve aesthetic and structural balance.

## 🆕 Direct Spherical Approach

The latest version implements a **true spherical tessellation** to naturally achieve the boundaryless property:

- 🌐 Points are generated directly on a unit sphere surface
- 🔄 Spherical Voronoi tessellation is computed using SciPy's SphericalVoronoi
- 📊 Region areas are calculated using spherical geometry principles
- 🧮 Optimization occurs directly on the sphere surface

This approach completely eliminates boundary conditions since a sphere has no boundaries. The tessellation forms a continuous outer crust with regions that interlock perfectly.

---

## 🧠 Mathematical Formulation

The tessellation is optimized by minimizing a composite energy function over region vertices:

\[
\frac{\partial}{\partial v_{i,j}}\Biggl[
\lambda_1(A(F_i)-A_{\text{target}}(\|\bar{v}_i\|))^2
+ \lambda_2\sum_{F_j \sim F_i} \delta(e_{ij})^2
+ \lambda_3\sum_{\theta \in \Theta_i} \frac{1}{\theta}
+ \lambda_4 \phi(A(F_i))
+ \lambda_5 d(\partial F_i, \partial(\cup_{j\neq i}F_j))^2
+ \lambda_7\left\|\nabla c(F_i)\right\|^2
\Biggr] = \mathbf{0}
\]

Each λ-term corresponds to a geometrically or topologically meaningful constraint:
- λ₁: Area difference from target (based on angular distance)
- λ₂: Boundary stability between adjacent regions
- λ₃: Angular penalty to maintain well-formed spherical polygons
- λ₄: Size variety penalty to enforce desired region size range
- λ₅: Boundary stability with all other regions
- λ₇: Centroid gradient for shape regularity

---

## 🔧 Features

- ✅ Radial size gradient implementation (larger tiles toward reference point)
- ✅ Spherical polygon area calculation and optimization
- ✅ Gapless and overlap-free tessellation on sphere surface
- ✅ Stable region boundaries with proper interlocking
- ✅ Tangent-space gradient computation for vertex movements
- ✅ Modular penalty functions (angle, size, boundary)
- ✅ Comprehensive test suite (12 passing tests)
- ✅ True 3D visualization with region coloring options

---

## 📊 Visualization

The project includes multiple visualization methods:

- 3D visualization showing the complete spherical tessellation
- Color-coded regions based on area or distance from reference point
- Interactive Streamlit app with adjustable parameters:
  - Number of points
  - Optimization iterations
  - Learning rate
  - Random seed
  - Color scheme selection
  - View mode options

---

## 🚀 Installation

```bash
git clone https://github.com/Caliban-17/tessellation-test.git
cd tessellation-test
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📚 Usage

To run the main demonstration:

```bash
python main.py
```

This will generate three visualizations:
- `initial_tessellation.png`: The unoptimized spherical tessellation
- `optimized_tessellation.png`: The result after optimization
- `tessellation_gradient.png`: Visualization with distance-based gradient coloring

To run the interactive Streamlit app:

```bash
streamlit run tessellation_test/streamlit_app/app.py
```

To run the tests:

```bash
python -m pytest tessellation_test/tests/ -v
```

To visualize the project's directory structure:

```bash
python tessellation_test/visualize_tree.py
```

## 🧪 Technical Details

The implementation uses:
- SciPy's SphericalVoronoi for generating the initial tessellation
- Spherical geometry for area calculations and angular distances
- Tangent-space gradient projection to keep vertices on the sphere
- Custom energy functions with tunable λ parameters
- Gradient descent optimization with normalization
- Matplotlib's 3D plotting for visualization
- Streamlit for interactive exploration
- Comprehensive test suite verifying mathematical properties