## <span style="font-size: 20px;">Singular Value Decomposition (SVD)</span>

The SVD is arguably the most important matrix decomposition in applied mathematics. Unlike eigendecomposition, it exists for any matrix - not just square ones - and reveals the fundamental geometric action of a linear transformation.

---

## The Factorization

Any $m \times n$ matrix $A$ can be decomposed as:

$$
A = U \Sigma V^T
$$

- $U$: $m \times m$ orthogonal matrix (columns are the left singular vectors)
- $\Sigma$: $m \times n$ diagonal matrix (singular values on diagonal, zeros elsewhere)
- $V^T$: $n \times n$ orthogonal matrix (rows are the right singular vectors)

The singular values are non-negative and conventionally sorted in descending order: $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$.

---

## Geometric Interpretation

The SVD decomposes the action of $A$ into three steps:

1. **Rotate/reflect** in the input space ($V^T$): align with the right singular vectors
2. **Scale** along each axis ($\Sigma$): stretch or shrink by each singular value
3. **Rotate/reflect** in the output space ($U$): align with the left singular vectors

Any linear transformation is just a rotation, a scaling, and another rotation. The singular values tell you how much stretching happens along each principal axis. This geometric picture makes SVD invaluable for understanding what a matrix "does" to vectors.

---

## The Four Fundamental Subspaces

SVD reveals all four fundamental subspaces of $A$ (assuming rank $r$):

| Subspace | From SVD | Dimension |
|----------|----------|-----------|
| Column space (range) | First $r$ columns of $U$ | $r$ |
| Left null space | Last $m - r$ columns of $U$ | $m - r$ |
| Row space | First $r$ columns of $V$ | $r$ |
| Null space (kernel) | Last $n - r$ columns of $V$ | $n - r$ |

These four subspaces and their relationships form the complete picture of a linear transformation.

---

## Singular Values and Eigenvalues

The singular values of $A$ are the square roots of the eigenvalues of $A^T A$ (or equivalently $AA^T$):

$$
\sigma_i = \sqrt{\lambda_i(A^T A)}
$$

The columns of $V$ are eigenvectors of $A^T A$, and the columns of $U$ are eigenvectors of $AA^T$. This connects SVD to eigendecomposition while working for any matrix shape. For symmetric positive definite matrices, the singular values equal the eigenvalues.

---

## Economy (Compact) SVD

The full SVD has $U$ as $m \times m$ and $V$ as $n \times n$, but many columns may correspond to zero singular values. The **economy SVD** keeps only the first $\min(m, n)$ components:

- If $m > n$: $U$ becomes $m \times n$, $\Sigma$ becomes $n \times n$, $V^T$ stays $n \times n$
- If $m < n$: $U$ stays $m \times m$, $\Sigma$ becomes $m \times m$, $V^T$ becomes $m \times n$

The **truncated SVD** goes further, keeping only the top $k$ singular values:

$$
A_k = U_k \Sigma_k V_k^T
$$

where $U_k$ is $m \times k$, $\Sigma_k$ is $k \times k$, and $V_k^T$ is $k \times n$. This gives the best rank-$k$ approximation to $A$ (Eckart-Young theorem).

---

## Key Properties

- Exists for **any** matrix (any shape, any rank, real or complex)
- Singular values are always real and non-negative
- $\|A\|_2 = \sigma_1$ (spectral/operator norm: maximum stretching factor)
- $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ (Frobenius norm)
- $\text{rank}(A) = $ number of nonzero singular values
- $\text{cond}(A) = \sigma_1 / \sigma_r$ (condition number: ratio of largest to smallest nonzero singular value)

---

## Computing SVD

The standard algorithm uses a two-phase approach: first bidiagonalize $A$ using Householder reflections (cost $O(mn^2)$ for $m \geq n$), then iteratively diagonalize using the QR algorithm adapted for bidiagonal matrices. The total cost is $O(mn \min(m,n))$.

---

## NumPy Implementation

```python
# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)

# Economy SVD (smaller U or Vt)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruct: A = U @ np.diag(s) @ Vt
```

Note: NumPy returns $s$ as a 1D array of singular values, not as the full diagonal matrix $\Sigma$. Use `np.diag(s)` to construct the matrix form.

---

## Applications

- **Pseudoinverse**: $A^+ = V \Sigma^+ U^T$ (invert nonzero singular values)
- **Rank determination**: Count singular values above a numerical threshold
- **Low-rank approximation**: Truncate to top $k$ singular values (Eckart-Young theorem)
- **PCA**: SVD of centered data matrix directly gives principal components
- **Latent Semantic Analysis**: SVD of term-document matrix reveals latent topics
- **Image compression**: Keep top $k$ components per color channel
- **Recommender systems**: Matrix factorization via truncated SVD