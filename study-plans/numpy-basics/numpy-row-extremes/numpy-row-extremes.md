# <span style="font-size: 20px;">Row Extremes</span>

<span style="font-size: 14px;">Finding the minimum and maximum values in each row of a matrix, along with their positions (column indices), is a common operation in data analysis and machine learning. It combines aggregation (`min`/`max`) with index-finding (`argmin`/`argmax`) and stacking. Understanding how to perform these operations along specific axes and combine the results efficiently is essential for feature extraction and result interpretation.</span>

---

## <span style="font-size: 16px;">Per-Row Min and Max</span>

```python
arr = np.array([[5, 2, 8],
                [1, 7, 3],
                [9, 4, 6]])

row_min = arr.min(axis=1)   # [2, 1, 4]
row_max = arr.max(axis=1)   # [8, 7, 9]
```

<span style="font-size: 14px;">`axis=1` collapses the column dimension, computing the statistic across each row.</span>

---

## <span style="font-size: 16px;">Per-Row Argmin and Argmax</span>

```python
col_of_min = arr.argmin(axis=1)  # [1, 0, 1] - column indices of minima
col_of_max = arr.argmax(axis=1)  # [2, 1, 0] - column indices of maxima
```

<span style="font-size: 14px;">`argmin` and `argmax` return the index of the extreme value along the specified axis. For axis=1, they return column indices.</span>

---

## <span style="font-size: 16px;">Stacking Results</span>

Combine the results in the required order: maximum values, maximum column indices, minimum values, and minimum column indices. Each output column corresponds to one original row.

```python
result = np.stack([row_max, col_of_max, row_min, col_of_min]).astype(np.float64)
# shape (4, 3) for the three-row example
```

---

## <span style="font-size: 16px;">Extracting Values at Argmax Indices</span>

<span style="font-size: 14px;">To use argmax indices to select from another array:</span>

```python
idx = arr.argmax(axis=1)  # [2, 1, 0]
# Extract the max value from each row (should match arr.max(axis=1)):
values = arr[np.arange(arr.shape[0]), idx]
```

<span style="font-size: 14px;">The `np.arange` provides row indices paired with the column indices from argmax.</span>

---

## <span style="font-size: 16px;">np.ptp() for Range</span>

```python
np.ptp(arr, axis=1)  # peak-to-peak: max - min per row
# [6, 6, 5]
```

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Classification confidence**: For a probability matrix, `argmax(axis=1)` gives predicted classes and `max(axis=1)` gives confidence scores.</span>
* <span style="font-size: 14px;">**Outlier detection**: Rows where `max - min` (range) is unusually large may contain outliers.</span>
* <span style="font-size: 14px;">**Feature importance**: For each sample, which feature has the largest/smallest value?</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**axis confusion**: `axis=1` gives per-row results; `axis=0` gives per-column results. Mixing these up is a frequent error.</span>
* <span style="font-size: 14px;">**Argmax with ties**: When multiple elements have the same max value, `argmax` returns the first occurrence.</span>
* <span style="font-size: 14px;">**NaN handling**: `np.max` returns NaN if any element is NaN. Use `np.nanmax` to ignore NaN.</span>