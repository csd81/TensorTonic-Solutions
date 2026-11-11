# <span style="font-size: 20px;">Column Scaling</span>

<span style="font-size: 14px;">Column scaling multiplies each column of a matrix by a corresponding scalar. This is the fundamental operation in feature scaling, where each feature (column) is multiplied by a normalization factor. Unlike row scaling, column scaling does not require a reshape because NumPy's broadcasting naturally aligns 1D arrays with the last dimension of 2D arrays.</span>

---

## <span style="font-size: 16px;">The Natural Broadcasting Pattern</span>

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)
w = np.array([10, 100, 1000])  # shape (3,)

result = A * w
# [[10, 200, 3000],
#  [40, 500, 6000]]
```

<span style="font-size: 14px;">No reshape is needed. NumPy broadcasts the $(3,)$ array along axis 0, multiplying each column by the corresponding weight. This works because broadcasting aligns from the right: shapes $(2, 3)$ and $(3,)$ are compatible.</span>

$$
A \odot w = \begin{pmatrix} a_{00} w_0 & a_{01} w_1 & a_{02} w_2 \\ a_{10} w_0 & a_{11} w_1 & a_{12} w_2 \end{pmatrix}
$$

---

## <span style="font-size: 16px;">Equivalent Matrix Operation</span>

<span style="font-size: 14px;">Column scaling is equivalent to right-multiplication by a diagonal matrix:</span>

$$
A \cdot \text{diag}(w)
$$

```python
A @ np.diag(w)   # correct but creates full (n, n) matrix
A * w            # same result, no temporary matrix
```

---

## <span style="font-size: 16px;">Feature Scaling</span>

<span style="font-size: 14px;">The most common application is normalizing features in a data matrix $X$ of shape $(m, n)$ where rows are samples and columns are features:</span>

### <span style="font-size: 14px;">Min-Max Scaling</span>

```python
col_min = X.min(axis=0)   # (n,)
col_max = X.max(axis=0)   # (n,)
X_scaled = (X - col_min) / (col_max - col_min)
```

### <span style="font-size: 14px;">Z-Score Standardization</span>

```python
col_mean = X.mean(axis=0)  # (n,)
col_std = X.std(axis=0)    # (n,)
X_standard = (X - col_mean) / col_std
```

<span style="font-size: 14px;">Both patterns work because subtracting and dividing by $(n,)$ arrays broadcasts naturally along axis 0.</span>

---

## <span style="font-size: 16px;">Row Scaling vs. Column Scaling</span>

| Operation | Weight shape | Reshape needed | Broadcasting |
|-----------|-------------|---------------|-------------|
| Column scaling | $(n,)$ | No | Natural (right-aligned) |
| Row scaling | $(m,)$ | Yes: $(m, 1)$ | Requires `[:, None]` |

<span style="font-size: 14px;">The asymmetry exists because NumPy broadcasts from the right. A $(3,)$ vector aligns with the last axis (columns) automatically, but aligning with the first axis (rows) requires explicit reshaping.</span>

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Feature weighting**: Weight features by importance scores</span>
* <span style="font-size: 14px;">**Inverse standardization**: Multiply by std and add mean to reverse z-score normalization</span>
* <span style="font-size: 14px;">**Unit conversion**: Convert each column from one unit to another with column-specific factors</span>
* <span style="font-size: 14px;">**Diagonal preconditioning**: Scale columns of a matrix to improve the condition number for linear solvers</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Confusing row and column scaling**: Column scaling uses a $(n,)$ vector naturally. If you accidentally use a $(m,)$ vector, broadcasting may produce wrong results or a shape error.</span>
* <span style="font-size: 14px;">**Division by zero in std**: Constant columns have $\sigma = 0$. Handle with `col_std[col_std == 0] = 1` before dividing.</span>
* <span style="font-size: 14px;">**Applying test statistics**: In ML, always compute scaling parameters from training data only, not from the test set.</span>