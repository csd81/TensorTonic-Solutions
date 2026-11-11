# <span style="font-size: 20px;">Aggregation Functions</span>

<span style="font-size: 14px;">Aggregation reduces an array (or a specific axis of an array) to summary statistics: sum, mean, standard deviation, min, max, and more. These operations are the building blocks of data analysis and feature engineering. NumPy implements them as compiled C functions that process entire arrays without Python loop overhead, making them orders of magnitude faster than manual iteration.</span>

---

## <span style="font-size: 16px;">Core Aggregation Functions</span>

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(a)      # 21 (sum of all elements)
np.mean(a)     # 3.5
np.std(a)      # standard deviation (ddof=0 by default)
np.var(a)      # variance
np.min(a)      # 1
np.max(a)      # 6
np.prod(a)     # 720 (product of all elements)
np.median(a)   # 3.5
```

### <span style="font-size: 14px;">Method vs. Function Syntax</span>

```python
a.sum()         # equivalent to np.sum(a)
a.mean(axis=0)  # equivalent to np.mean(a, axis=0)
```

<span style="font-size: 14px;">Both forms are identical. The method syntax is more concise for simple cases.</span>

---

## <span style="font-size: 16px;">The axis Parameter</span>

<span style="font-size: 14px;">The `axis` parameter controls which dimension is collapsed:</span>

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
# shape: (2, 3)

np.sum(a, axis=0)   # [5, 7, 9] - sum along rows, result shape (3,)
np.sum(a, axis=1)   # [6, 15] - sum along columns, result shape (2,)
np.sum(a)            # 21 - sum all elements (no axis)
```

<span style="font-size: 14px;">Think of axis as "the dimension that disappears." `axis=0` collapses the row dimension, leaving one value per column. `axis=1` collapses the column dimension, leaving one value per row.</span>

### <span style="font-size: 14px;">Shape After Aggregation</span>

<span style="font-size: 14px;">For an array of shape $(m, n)$:</span>

* <span style="font-size: 14px;">`axis=0`: result shape is $(n,)$</span>
* <span style="font-size: 14px;">`axis=1`: result shape is $(m,)$</span>
* <span style="font-size: 14px;">`axis=None` (default): scalar result</span>

---

## <span style="font-size: 16px;">keepdims Parameter</span>

<span style="font-size: 14px;">By default, aggregation removes the collapsed axis. `keepdims=True` preserves it as length 1:</span>

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(a, axis=1)              # shape (2,)
np.sum(a, axis=1, keepdims=True)  # shape (2, 1)
```

<span style="font-size: 14px;">This is essential for broadcasting. To normalize each row by its sum:</span>

```python
row_sums = a.sum(axis=1, keepdims=True)  # shape (2, 1)
normalized = a / row_sums                 # shape (2, 3) via broadcasting
```

<span style="font-size: 14px;">Without `keepdims`, the division would fail or produce wrong results because a $(2,)$ array cannot broadcast correctly with a $(2, 3)$ array along axis 1.</span>

---

## <span style="font-size: 16px;">NaN-Safe Aggregations</span>

<span style="font-size: 14px;">Standard functions propagate NaN:</span>

```python
np.mean([1, 2, np.nan, 4])  # nan
```

<span style="font-size: 14px;">Use the `nan`-prefixed versions to skip NaN values:</span>

```python
np.nanmean([1, 2, np.nan, 4])   # 2.333
np.nansum([1, 2, np.nan, 4])    # 7.0
np.nanstd([1, 2, np.nan, 4])    # standard deviation ignoring NaN
np.nanmin([1, 2, np.nan, 4])    # 1.0
np.nanmax([1, 2, np.nan, 4])    # 4.0
```

---

## <span style="font-size: 16px;">Argmin and Argmax</span>

<span style="font-size: 14px;">Find the index of the minimum/maximum value:</span>

```python
a = np.array([30, 10, 50, 20])
np.argmin(a)           # 1 (index of 10)
np.argmax(a)           # 2 (index of 50)

# Along an axis
arr = np.array([[1, 9], [8, 2]])
np.argmax(arr, axis=0)  # [1, 0] - row index of max in each column
np.argmax(arr, axis=1)  # [1, 0] - column index of max in each row
```

---

## <span style="font-size: 16px;">Cumulative Operations</span>

```python
np.cumsum([1, 2, 3, 4])    # [1, 3, 6, 10]
np.cumprod([1, 2, 3, 4])   # [1, 2, 6, 24]
```

<span style="font-size: 14px;">Cumulative operations are useful for running totals, CDF computation, and prefix sums.</span>

---

## <span style="font-size: 16px;">Stacking Aggregation Results</span>

<span style="font-size: 14px;">A common pattern is to compute multiple statistics and stack them:</span>

```python
stats = np.array([
    np.mean(a, axis=0),
    np.std(a, axis=0),
    np.min(a, axis=0),
    np.max(a, axis=0)
])  # shape (4, n_cols)
```

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**axis confusion**: `axis=0` collapses rows, giving per-column results. This is counterintuitive for newcomers.</span>
* <span style="font-size: 14px;">**NaN propagation**: A NaN affects the reduction slice containing it; unrelated rows or columns are unchanged. NaN-aware reductions can skip missing values.</span>
* <span style="font-size: 14px;">**std ddof default**: NumPy's `std()` uses `ddof=0` (population std). Pandas uses `ddof=1` (sample std). This causes different results on the same data.</span>
* <span style="font-size: 14px;">**Integer overflow**: Summing large integer arrays can overflow. Use `dtype=np.int64` or `dtype=np.float64` for large sums.</span>