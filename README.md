# Tessellation Test

## 📐 Overview

**Tessellation Test** is an experimental Python project designed to generate and optimize **irregular Voronoi tessellations** with no explicit boundaries — simulating the topology of a **boundaryless universe**. The algorithm emphasizes:

- 📊 Radial area gradients (larger central tiles)
- 🔀 Maximized irregularity in tile sizes
- 🧩 Interlocking tiles without gaps or overlaps
- 🔁 Periodic boundary conditions for seamless continuity

The engine uses a mathematically grounded energy function based on multivariate calculus and gradient descent to achieve aesthetic and structural balance.

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
- ✅ Fully tested (12 pytest assertions)

---

## 🚀 Installation

```bash
git clone https://github.com/Caliban-17/tessellation-test.git
cd tessellation-test
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
