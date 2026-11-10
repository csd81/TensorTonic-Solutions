## <span style="font-size: 20px;">Low-Rank Matrix Approximation</span>

Low-rank approximation is one of the most powerful ideas in applied linear algebra: approximate a large matrix by one with much lower rank, capturing the essential structure while discarding noise or unimportant details. It is the foundation of dimensionality reduction, data compression, and collaborative filtering.

---

## The Eckart-Young-Mirsky Theorem

The best rank-$k$ approximation to a matrix $A$ (in both Frobenius and spectral norms) is given by truncating the SVD:

$$
A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T
$$

where $U_k$, $\Sigma_k$, $V_k^T$ are the top-$k$ components of the SVD. No other rank-$k$ matrix is closer to $A$. This is a remarkable optimality result - among all possible rank-$k$ matrices, SVD truncation is provably the best.

---

## Approximation Error

The error of the rank-$k$ approximation is completely determined by the discarded singular values:

$$
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$

$$
\|A - A_k\|_2 = \sigma_{k+1}
$$

If the singular values decay rapidly (which they often do in practice), even a very low rank gives an excellent approximation. The ratio of captured energy is $\sum_{i=1}^{k} \sigma_i^2 / \sum_{i=1}^{r} \sigma_i^2$, and this is exactly the "explained variance ratio" used in PCA.

---

## Image Compression Example

An $m \times n$ grayscale image stored as a matrix requires $mn$ values. The rank-$k$ SVD approximation stores:

- $U_k$: $m \times k$ values
- $\sigma_1, \ldots, \sigma_k$: $k$ values
- $V_k^T$: $k \times n$ values

Total: $k(m + n + 1)$ values. For a $1000 \times 1000$ image with $k = 50$, that is $100{,}050$ values instead of $1{,}000{,}000$ - roughly 10x compression while retaining the dominant visual features. The first few singular components capture large-scale structure (brightness, gradients), while later ones capture fine detail and texture.

---

## Data Denoising

In many applications, the true signal is low-rank and noise adds full-rank perturbation. The noisy observation is:

$$
A_{\text{observed}} = A_{\text{signal}} + A_{\text{noise}}
$$

Since noise spreads across all singular values (roughly uniformly for Gaussian noise) while signal concentrates in the top singular values, truncating the SVD effectively separates signal from noise. The challenge is choosing the right $k$ - methods include:

- **Elbow rule**: Plot singular values and look for a sharp drop
- **Cross-validation**: Hold out entries and minimize reconstruction error
- **Random matrix theory**: For Gaussian noise of variance $\sigma^2$, threshold at $\sigma \sqrt{\max(m,n)}$ (Marchenko-Pastur law)
- **Gavish-Donoho threshold**: An asymptotically optimal threshold for the spiked model

---

## Connection to PCA

PCA on a dataset $X$ (centered, samples as rows) finds directions of maximum variance. These are exactly the right singular vectors of $X$, and the variance explained by each component is $\sigma_i^2 / (n-1)$. Keeping the top $k$ principal components is mathematically identical to computing the rank-$k$ SVD approximation of the centered data matrix. The "scree plot" in PCA is just a plot of the singular values.

---

## Recommender Systems

User-item rating matrices are approximately low-rank: a few latent factors (genre preferences, item popularity, user demographics) explain most ratings. Matrix factorization methods decompose the (sparse, incomplete) rating matrix:

$$
R \approx U V^T
$$

where $U$ ($\text{users} \times k$) captures user factors and $V$ ($\text{items} \times k$) captures item factors. This is a constrained low-rank approximation problem. SVD provides the starting point for many algorithms (Simon Funk's SGD-based approach, Alternating Least Squares). The Netflix Prize famously demonstrated the power of these techniques.

---

## Computational Savings

For an $m \times n$ matrix of rank $r$, storing the full matrix costs $O(mn)$. The rank-$k$ approximation costs:

$$
O(k(m + n))
$$

Matrix-vector products drop from $O(mn)$ to $O(k(m + n))$. For $k \ll \min(m, n)$, the savings are dramatic. Randomized SVD algorithms (Halko, Martinsson, Tropp 2011) can compute the top-$k$ SVD in $O(mn \log k)$ time instead of $O(mn \min(m,n))$ for the full SVD, making low-rank approximation practical for very large matrices.

---

## NumPy Implementation

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
k = 10  # desired rank
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Approximation error
error = np.sqrt(np.sum(s[k:]**2))

# Explained variance ratio
explained = np.cumsum(s**2) / np.sum(s**2)
```

---

## Choosing the Rank $k$

- **Explained variance ratio**: Keep enough components to capture 95% (or 99%) of $\sum \sigma_i^2$
- **Elbow method**: Plot singular values vs index, look for a sharp drop
- **Cross-validation**: Hold out entries, minimize reconstruction error on held-out data
- **Task performance**: Choose $k$ that maximizes downstream task accuracy (e.g., classification or recommendation quality)