# <span style="font-size: 20px;">Norm-Gated Linear Transform</span>

<span style="font-size: 14px;">A norm-gated linear transform applies a linear transformation $Y = XW$ and then gates (scales) each row of the output based on the L2 norm of the corresponding transformed row. Rows with transformed norms below a threshold are zeroed out, implementing a form of sparse activation. This pattern appears in attention mechanisms, gating networks, and feature selection layers where transformed magnitude determines retention.</span>

---

## <span style="font-size: 16px;">Mathematical Definition</span>

<span style="font-size: 14px;">Given input $X \in \mathbb{R}^{n \times d}$, weight matrix $W \in \mathbb{R}^{d \times k}$, and threshold $\tau$:</span>

$$
Y = XW
$$

$$
g_i = \begin{cases} 1 & \text{if } \|Y_i\|_2 \geq \tau \\ 0 & \text{if } \|Y_i\|_2 < \tau \end{cases}
$$

$$
Z_i = g_i \cdot Y_i
$$

<span style="font-size: 14px;">Here, i indexes rows, Y is the transformed matrix, g is the binary gate, Z is the gated output, and the positive threshold is τ. The dimensions n, d, and k are the row count, input width, and output width.</span>

---

## <span style="font-size: 16px;">Computing Row Norms</span>

```python
Y = X @ W
norms = np.linalg.norm(Y, axis=1)  # shape (n,)
```

<span style="font-size: 14px;">The L2 norm is computed from each transformed row:</span>

$$
\|Y_i\|_2 = \sqrt{\sum_{j=1}^{k} Y_{ij}^2}
$$

### <span style="font-size: 14px;">Alternative: Manual Computation</span>

```python
norms = np.sqrt(np.sum(Y**2, axis=1))
```

<span style="font-size: 14px;">Both approaches produce the same result. `np.linalg.norm` is preferred for readability and handles edge cases (like zero vectors) correctly.</span>

---

## <span style="font-size: 16px;">Matrix Multiplication</span>

```python
Y = X @ W          # shape (n, k)
# or equivalently:
Y = np.dot(X, W)
Y = np.matmul(X, W)
```

<span style="font-size: 14px;">The `@` operator is the most readable. For 2D arrays, all three forms are equivalent.</span>

---

## <span style="font-size: 16px;">Creating the Gate Mask</span>

```python
gate = (norms >= threshold).astype(np.float64)  # shape (n,)
# or as boolean:
gate = norms >= threshold  # shape (n,), dtype bool
```

<span style="font-size: 14px;">The gate is 1 for rows with sufficient norm and 0 otherwise.</span>

---

## <span style="font-size: 16px;">Applying the Gate</span>

```python
Z = Y * gate[:, np.newaxis]  # broadcast (n, 1) * (n, k)
```

<span style="font-size: 14px;">The `[:, np.newaxis]` reshape is necessary to broadcast the $(n,)$ gate vector across the $(n, k)$ output matrix. Each row of $Y$ is multiplied by its corresponding gate value (0 or 1).</span>

---

## <span style="font-size: 16px;">Complete Implementation</span>

```python
def norm_gated_linear(X, W, threshold):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Y = X @ W
    norms = np.linalg.norm(Y, axis=1)
    gate = (norms >= threshold).astype(Y.dtype)
    return Y * gate[:, np.newaxis]
```

---

## <span style="font-size: 16px;">Soft Gating Variant</span>

<span style="font-size: 14px;">Outside this problem, a continuous alternative uses the transformed-row norm as a scaling factor:</span>

$$
Z_i = \frac{\|Y_i\|_2}{\max(\|Y_i\|_2, \tau)} \cdot Y_i
$$

```python
soft_gate = norms / np.maximum(norms, threshold)
Z = Y * soft_gate[:, np.newaxis]
```

<span style="font-size: 14px;">This smoothly attenuates small-norm rows rather than completely zeroing them out, although this soft gate can still have a derivative discontinuity at the threshold.</span>

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Attention mechanisms**: Gate query vectors by their norm to suppress low-confidence queries</span>
* <span style="font-size: 14px;">**Sparse activation**: Zero out outputs for inputs with low signal-to-noise ratio</span>
* <span style="font-size: 14px;">**Feature selection**: Only propagate signals from features with sufficient magnitude</span>
* <span style="font-size: 14px;">**Robust inference**: Reject noisy or corrupted inputs that have abnormally low/high norms</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting newaxis on gate**: Multiplying Y by the one-dimensional gate without reshaping tries to broadcast $(n, k)$ and $(n,)$, which scales columns when the dimensions happen to match and otherwise may fail to broadcast.</span>
* <span style="font-size: 14px;">**Threshold too high**: If the threshold is larger than most row norms, the output is mostly zeros.</span>
* <span style="font-size: 14px;">**Gradient flow**: The hard gate is locally constant away from the threshold and discontinuous at the threshold, so its derivative there is undefined.</span>
* <span style="font-size: 14px;">**dtype**: Multiplying a boolean mask by a float64 array produces float64 output. Casting the mask to float is optional.</span>