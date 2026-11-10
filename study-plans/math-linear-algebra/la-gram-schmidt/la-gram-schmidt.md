## <span style="font-size: 20px;">The Gram-Schmidt Process</span>

The Gram-Schmidt process transforms a set of linearly independent vectors into an orthonormal set that spans the same subspace. It is one of the most important constructive algorithms in linear algebra, underlying QR decomposition and many iterative methods.

---

## The Algorithm

Given linearly independent vectors $\{v_1, v_2, \ldots, v_n\}$, produce orthogonal vectors $\{u_1, u_2, \ldots, u_n\}$:

$$
u_1 = v_1
$$

$$
u_k = v_k - \sum_{j=1}^{k-1} \text{proj}_{u_j} v_k = v_k - \sum_{j=1}^{k-1} \frac{v_k \cdot u_j}{u_j \cdot u_j} u_j
$$

Then normalize to get orthonormal vectors:

$$
e_k = \frac{u_k}{\|u_k\|}
$$

The result is a set of orthonormal vectors $\{e_1, e_2, \ldots, e_n\}$ where $e_i \cdot e_j = 0$ for $i \neq j$ and $\|e_i\| = 1$.

---

## Why It Works

At each step, $u_k$ is constructed by taking $v_k$ and subtracting its projections onto all previously computed orthogonal vectors. What remains is the component of $v_k$ that is perpendicular to the subspace spanned by $\{u_1, \ldots, u_{k-1}\}$.

By induction: $u_1$ is trivially orthogonal to nothing. If $u_1, \ldots, u_{k-1}$ are mutually orthogonal, then $u_k$ has had all components in those directions removed, so $u_k \cdot u_j = 0$ for all $j < k$.

Crucially, the span is preserved at each step: $\text{span}(e_1, \ldots, e_k) = \text{span}(v_1, \ldots, v_k)$ for every $k$. The process repackages the same subspace into an orthonormal basis.

---

## Connection to QR Decomposition

The Gram-Schmidt process directly produces the QR factorization. If $A = [v_1 | v_2 | \ldots | v_n]$ and $Q = [e_1 | e_2 | \ldots | e_n]$, then:

$$
A = QR
$$

where $Q$ is orthogonal ($Q^TQ = I$) and $R$ is upper triangular. The entries of $R$ are the projection coefficients:

$$
r_{jk} = e_j \cdot v_k \quad (j \leq k), \qquad r_{jk} = 0 \quad (j > k)
$$

The diagonal entries $r_{kk} = \|u_k\|$ are the norms of the orthogonal vectors before normalization. QR decomposition is the preferred method for solving least-squares problems due to its numerical stability compared to the normal equations.

---

## Classical vs. Modified Gram-Schmidt

**Classical Gram-Schmidt (CGS)** computes all projections using the original vectors:

$$
u_k = v_k - \sum_{j=1}^{k-1} \frac{v_k \cdot u_j}{u_j \cdot u_j} u_j
$$

**Modified Gram-Schmidt (MGS)** subtracts projections one at a time, updating the intermediate result after each subtraction:

$$
u_k^{(1)} = v_k - \text{proj}_{u_1} v_k, \quad u_k^{(2)} = u_k^{(1)} - \text{proj}_{u_2} u_k^{(1)}, \quad \ldots
$$

Mathematically identical, but MGS is much more numerically stable. In floating point arithmetic, CGS can produce vectors that are far from orthogonal when the input vectors are nearly dependent, while MGS maintains better orthogonality. The difference matters in practice: for ill-conditioned matrices, CGS may produce $Q$ where $\|Q^TQ - I\|$ is large, while MGS keeps this error small.

For even better stability, Householder reflections are used (this is what NumPy's QR internally uses).

---

## Detecting Linear Dependence

If $v_k$ is a linear combination of the previous vectors, then after subtracting all projections, $u_k = 0$ (or numerically very small). This is how Gram-Schmidt reveals the rank of a set of vectors: the number of non-zero $u_k$ vectors equals the rank. In practice, you check if $\|u_k\|$ falls below a threshold relative to $\|v_k\|$.

---

## Worked Example

Consider $v_1 = (1, 1, 0)^T$ and $v_2 = (1, 0, 1)^T$:

1. $u_1 = v_1 = (1, 1, 0)^T$
2. $\text{proj}_{u_1} v_2 = \frac{v_2 \cdot u_1}{u_1 \cdot u_1} u_1 = \frac{1}{2}(1, 1, 0)^T$
3. $u_2 = v_2 - \text{proj}_{u_1} v_2 = (1/2, -1/2, 1)^T$
4. Normalize: $e_1 = (1/\sqrt{2}, 1/\sqrt{2}, 0)^T$, $e_2 = (1/\sqrt{6}, -1/\sqrt{6}, 2/\sqrt{6})^T$

Verify: $e_1 \cdot e_2 = 1/\sqrt{12} - 1/\sqrt{12} + 0 = 0$.

---

## Applications

- **QR factorization**: The primary algorithm for thin QR in many implementations
- **Orthogonal basis construction**: Needed for orthogonal projections onto subspaces
- **Eigenvalue algorithms**: QR iteration uses repeated QR factorizations
- **Krylov methods**: Arnoldi and Lanczos iterations use Gram-Schmidt to build orthogonal bases for iterative eigensolvers and linear system solvers
- **Numerical linear algebra**: Solving $Ax = b$ via QR is more stable than via normal equations

---

## NumPy Implementation

```python
def gram_schmidt(V):
    """V: columns are input vectors. Returns Q (orthonormal)."""
    n = V.shape[1]
    Q = np.zeros_like(V, dtype=float)
    for k in range(n):
        u = V[:, k].astype(float)
        for j in range(k):
            u -= np.dot(Q[:, j], V[:, k]) * Q[:, j]
        Q[:, k] = u / np.linalg.norm(u)
    return Q
```