# Tessellation Test

## 📐 Overview

**Tessellation Test** is an experimental Python project designed to generate and optimize **irregular Voronoi tessellations** with no explicit boundaries — simulating the topology of a **boundaryless universe**. The algorithm emphasizes:

- 📊 Radial area gradients (larger central tiles)
- 🔀 Maximized irregularity in tile sizes
- 🧩 Interlocking tiles without gaps or overlaps
- 🔍 Zoomed-in spherical topology for seamless continuity

The engine uses a mathematically grounded energy function based on multivariate calculus and gradient descent to achieve aesthetic and structural balance.

## 🆕 Zoomed-in Spherical Approach

The latest version implements a **zoomed-in spherical tessellation** to naturally achieve the boundaryless property:

- 🌐 Points are projected onto a unit sphere
- 🔍 Only a portion of the sphere is used (zoomed-in view)
- 🔄 Voronoi tessellation is computed on this portion
- 📈 Results are projected back to 2D using an equal-area projection

This approach eliminates the need to handle complex boundary conditions since we're not using the entire sphere surface - we're only working with a small patch of it, far from any boundary.

---

## 🧠 Mathematical Formulation

The tessellation is optimized by minimizing a composite energy function over polygon vertices:

\[
\frac{\partial}{\partial v_{i,j}}\Biggl[
\lambda_1(A(F_i)-A_{\text{target}}(\|\bar{v}_i\|))^2
+ \lambda_2\sum_{F_j \sim F_i} \delta(e_{ij})^2
+ \lambda_3\sum_{\theta \in \Theta_i} \frac{1}{\theta}
+ \lambda_4 \phi(A(F_i))
+ \lambda_5 d(\partial F_i, \partial(\cup_{j\neq i}F_j))^2
+ \lambda_6\left\|\frac{\partial v_{i,j}(t)}{\partial t}\right\|^2
+ \lambda_7\left\|\nabla c(F_i)\right\|^2
\Biggr] = \mathbf{0}
\]

Each λ-term corresponds to a geometrically or topologically meaningful constraint.

---

## 🔧 Features

- ✅ Radial size gradient implementation
- ✅ Polygonal area irregularity
- ✅ Gapless and overlap-free tessellation
- ✅ Stable polygonal boundaries
- ✅ Gradient-constrained vertex movements
- ✅ Modular penalty functions (angle, size, boundary, etc.)
- ✅ Fully tested (12 passing tests)
- ✅ Zoomed-in spherical topology for true boundaryless tiling

---

## 📊 Visualization

The project includes both 2D and 3D visualizations:

- 2D projection showing the tessellation in a planar representation
- 3D visualization showing the points on a sphere
- Interactive Streamlit app with zoom factor control

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

To run the interactive Streamlit app:

```bash
streamlit run tessellation_test/streamlit_app/app.py
```

To run the tests:

```bash
python -m pytest tessellation_test/tests/ -v
```

## 🔍 Zooming Explained

The zoom factor (default 0.5) controls how much of the sphere is used:
- Lower values (e.g., 0.2) focus on a very small patch, completely avoiding boundary effects
- Higher values (e.g., 0.8) use more of the sphere but may introduce more distortion
- The optimal setting balances coverage area with natural tessellation properties

For best results, keep the zoom factor between 0.3 and 0.7.