## <span style="font-size: 20px;">LU Decomposition</span>

LU decomposition factors a square matrix into a product of a lower triangular matrix $L$ and an upper triangular matrix $U$. It is the algorithmic backbone of Gaussian elimination and the most common method for solving linear systems in practice.

---

## The Factorization

Given a square matrix $A$, find:

$$
A = LU
$$

where $L$ is lower triangular with ones on the diagonal, and $U$ is upper triangular:

$$
L = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ l_{21} & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & 1 \end{pmatrix}, \quad
U = \begin{pmatrix} u_{11} & u_{12} & \cdots & u_{1n} \\ 0 & u_{22} & \cdots & u_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & u_{nn} \end{pmatrix}
$$

The convention of placing ones on the diagonal of $L$ is called the **Doolittle form**. The alternative **Crout form** puts ones on the diagonal of $U$ instead.

---

## Connection to Gaussian Elimination

LU decomposition IS Gaussian elimination, just organized as a matrix factorization. Each step of elimination multiplies by an elementary lower triangular matrix $E_k$ that zeros out entries below the pivot. The product $E_{n-1} \cdots E_1 A = U$ gives the upper triangular form, and $L = E_1^{-1} \cdots E_{n-1}^{-1}$ collects the multipliers.

The beautiful part: each $E_k^{-1}$ just negates the sub-diagonal entries, and their product slots them neatly into $L$. The entries $l_{ij}$ (for $i > j$) are exactly the elimination multipliers $l_{ij} = a_{ij}^{(j)} / a_{jj}^{(j)}$ used during row reduction.

---

## Solving Linear Systems

To solve $Ax = b$ given $A = LU$:

1. **Forward substitution**: Solve $Ly = b$ for $y$ (top to bottom, $O(n^2)$)
2. **Backward substitution**: Solve $Ux = y$ for $x$ (bottom to top, $O(n^2)$)

The key advantage: the $O(n^3)$ factorization is done once, then each new right-hand side $b$ costs only $O(n^2)$. This is critical when solving the same system for many different $b$ vectors - common in simulation, control, and optimization.

---

## Partial Pivoting

Plain LU factorization can fail (zero pivot) or be numerically unstable (small pivot causing large multipliers). **Partial pivoting** swaps rows at each step to put the largest element in the pivot position:

$$
PA = LU
$$

where $P$ is a permutation matrix recording all row swaps. This is what all practical implementations use. Complete pivoting (swapping both rows and columns) gives even better stability but is rarely needed and costs more.

---

## Computational Cost

- **Factorization**: $\frac{2}{3}n^3$ floating point operations (same as Gaussian elimination)
- **Each solve** (given L, U): $O(n^2)$ operations
- **Determinant**: $\det(A) = \det(P)^{-1} \cdot \prod_i u_{ii}$ where $\det(P) = \pm 1$ depending on the number of row swaps
- **Matrix inverse**: Solve $AX = I$ column by column using the same LU factorization - $n$ solves at $O(n^2)$ each, total $O(n^3)$

---

## When to Use LU

| Scenario | Recommended Method |
|----------|-------------------|
| Single solve $Ax = b$ | LU or QR |
| Multiple solves, same $A$ | LU (factor once, solve many) |
| Least squares ($m > n$) | QR or SVD |
| Symmetric positive definite | Cholesky ($A = LL^T$, twice as fast) |
| Large sparse systems | Iterative methods (CG, GMRES) |

---

## Variants

- **LDU decomposition**: $A = LDU$ where $D$ is diagonal, useful for symmetric matrices where $A = LDL^T$
- **Cholesky**: $A = LL^T$ for symmetric positive definite matrices (half the flops, guaranteed stable without pivoting)
- **Block LU**: For large matrices, operates on submatrix blocks to exploit cache locality and enable parallelism
- **Incomplete LU (ILU)**: Approximate factorization used as a preconditioner for iterative methods on sparse systems

---

## SciPy Implementation

```python
import scipy.linalg as la

# Full factorization
P, L, U = la.lu(A)

# Compact form for solving (more efficient)
lu, piv = la.lu_factor(A)
x = la.lu_solve((lu, piv), b)

# Determinant via LU
det_A = np.prod(np.diag(U)) * np.linalg.det(P)
```

---

## Applications

- **Solving linear systems**: The default direct method in most numerical libraries (LAPACK, NumPy, MATLAB)
- **Computing determinants**: Product of pivots from $U$, adjusted for permutation sign
- **Matrix inversion**: Solve $n$ linear systems with identity columns as right-hand sides
- **Simulation**: Time-stepping in PDEs with constant coefficient matrix (factor once, solve at each timestep)
- **Sparse direct solvers**: Libraries like UMFPACK and SuperLU use LU with fill-reducing orderings