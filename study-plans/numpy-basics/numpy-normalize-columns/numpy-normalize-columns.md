# <span style="font-size: 20px;">Normalize Columns</span>

<span style="font-size: 14px;">Column normalization (also called feature scaling) rescales each column of a data matrix so that all features contribute equally to downstream algorithms. Without normalization, features with large magnitudes dominate distance-based methods (k-NN, SVM, k-means) and slow down gradient-based optimization (neural networks, logistic regression). The two most common techniques are min-max scaling and z-score standardization.</span>

---

## <span style="font-size: 16px;">Min-Max Normalization</span>

<span style="font-size: 14px;">Rescales each column to the range $[0, 1]$:</span>

$$
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

```python
col_min = arr.min(axis=0)
col_max = arr.max(axis=0)
normalized = (arr - col_min) / (col_max - col_min)
```

<span style="font-size: 14px;">After normalization, every column has minimum 0 and maximum 1. The original distribution shape is preserved; only the scale changes.</span>

### <span style="font-size: 14px;">When to Use</span>

* <span style="font-size: 14px;">Features must be in a bounded range (e.g., neural network inputs with sigmoid output)</span>
* <span style="font-size: 14px;">Algorithms sensitive to magnitude (k-NN, SVM with RBF kernel)</span>
* <span style="font-size: 14px;">Image pixel values (0-255 -> 0-1)</span>

### <span style="font-size: 14px;">Weakness</span>

<span style="font-size: 14px;">Outliers compress the majority of values into a narrow range. If the max is 1000 but 99% of values are below 10, normalization maps the majority to $[0, 0.01]$.</span>

---

## <span style="font-size: 16px;">Z-Score Standardization</span>

<span style="font-size: 14px;">Centers each column to mean 0 and standard deviation 1:</span>

$$
z = \frac{x - \mu}{\sigma}
$$

```python
col_mean = arr.mean(axis=0)
col_std = arr.std(axis=0)
standardized = (arr - col_mean) / col_std
```

### <span style="font-size: 14px;">When to Use</span>

* <span style="font-size: 14px;">Data is approximately Gaussian</span>
* <span style="font-size: 14px;">Algorithms assume centered data (PCA, linear regression)</span>
* <span style="font-size: 14px;">Gradient-based optimization (standardized features improve conditioning)</span>

---

## <span style="font-size: 16px;">Broadcasting in Normalization</span>

<span style="font-size: 14px;">The key to vectorized normalization is broadcasting. For a $(m, n)$ array:</span>

```python
arr.shape         # (m, n)
arr.mean(axis=0)  # shape (n,) - one mean per column
```

<span style="font-size: 14px;">When you subtract a $(n,)$ array from a $(m, n)$ array, NumPy broadcasts the 1D array along axis 0, subtracting the column mean from every row. This happens automatically without explicit loops.</span>

<span style="font-size: 14px;">If you use `keepdims=True`:</span>

```python
arr.mean(axis=0, keepdims=True)  # shape (1, n)
```

<span style="font-size: 14px;">Broadcasting works the same way: the $(1, n)$ array is broadcast across all $m$ rows.</span>

---

## <span style="font-size: 16px;">Handling Zero Standard Deviation</span>

<span style="font-size: 14px;">This problem excludes constant columns. More generally, a constant column has zero standard deviation and requires a defined zero-denominator policy:</span>

```python
col_std = arr.std(axis=0)
col_std[col_std == 0] = 1.0  # avoid division by zero
standardized = (arr - arr.mean(axis=0)) / col_std
```

<span style="font-size: 14px;">Setting $\sigma = 1$ for constant columns produces an all-zero column, which correctly represents "this feature carries no information."</span>

---

## <span style="font-size: 16px;">L2 Normalization (Unit Vectors)</span>

<span style="font-size: 14px;">Normalize each column (or row) to unit length:</span>

$$
x_{\text{unit}} = \frac{x}{\|x\|_2}
$$

```python
norms = np.linalg.norm(arr, axis=0)  # L2 norm per column
normalized = arr / norms
```

<span style="font-size: 14px;">L2 normalization is used in cosine similarity computations, text embeddings, and anywhere direction matters more than magnitude.</span>

---

## <span style="font-size: 16px;">Row vs. Column Normalization</span>

<span style="font-size: 14px;">The axis parameter determines the direction:</span>

| Goal | axis for stat | Broadcast direction |
|------|--------------|-------------------|
| Normalize columns (features) | `axis=0` | Along rows |
| Normalize rows (samples) | `axis=1, keepdims=True` | Along columns |

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Division by zero**: Constant columns need a defined zero-denominator policy in general; this problem guarantees they are absent.</span>
* <span style="font-size: 14px;">**Axis confusion**: `axis=0` computes stats along rows (per column). Mixing this up normalizes the wrong direction.</span>
* <span style="font-size: 14px;">**Fitting on test data**: In ML, compute normalization parameters (mean, std) from training data only. Applying test data's own statistics causes data leakage.</span>
* <span style="font-size: 14px;">**Integer arrays**: True division produces floating-point values. Converting inputs to float64 first fixes the precision of intermediate calculations.</span>