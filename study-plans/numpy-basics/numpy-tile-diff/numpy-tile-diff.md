# <span style="font-size: 20px;">Tile and Diff</span>

<span style="font-size: 14px;">Tiling replicates an array along specified axes, while differencing computes the change between consecutive elements. Together, they are useful for creating repeated patterns, computing discrete derivatives, and building structured matrices. NumPy's `np.tile()` and `np.diff()` provide these operations as efficient vectorized functions.</span>

---

## <span style="font-size: 16px;">np.tile()</span>

<span style="font-size: 14px;">`np.tile(A, reps)` constructs an array by repeating $A$ the specified number of times:</span>

```python
a = np.array([1, 2, 3])

np.tile(a, 2)        # [1, 2, 3, 1, 2, 3]
np.tile(a, (2, 1))   # [[1, 2, 3], [1, 2, 3]]
np.tile(a, (2, 3))   # [[1,2,3,1,2,3,1,2,3], [1,2,3,1,2,3,1,2,3]]
```

### <span style="font-size: 14px;">2D Tiling</span>

```python
arr = np.array([[1, 2], [3, 4]])
np.tile(arr, (3, 2))
# [[1,2,1,2],
#  [3,4,3,4],
#  [1,2,1,2],
#  [3,4,3,4],
#  [1,2,1,2],
#  [3,4,3,4]]
```

<span style="font-size: 14px;">The `reps` tuple $(3, 2)$ means "3 copies along axis 0, 2 copies along axis 1."</span>

### <span style="font-size: 14px;">tile vs. repeat</span>

* <span style="font-size: 14px;">`np.tile([1,2,3], 2)` = $[1,2,3,1,2,3]$ (whole array repeated)</span>
* <span style="font-size: 14px;">`np.repeat([1,2,3], 2)` = $[1,1,2,2,3,3]$ (each element repeated)</span>

---

## <span style="font-size: 16px;">np.diff()</span>

<span style="font-size: 14px;">`np.diff(a)` computes the first-order finite difference: the difference between consecutive elements:</span>

```python
a = np.array([1, 4, 9, 16, 25])
np.diff(a)     # [3, 5, 7, 9] - length n-1
np.diff(a, n=2)  # [2, 2, 2] - second differences (diff of diff)
```

<span style="font-size: 14px;">The result has one fewer element along the differenced axis. For a time series $[x_0, x_1, ..., x_n]$, the first differences are $[x_1 - x_0, x_2 - x_1, ..., x_n - x_{n-1}]$.</span>

### <span style="font-size: 14px;">2D Differences</span>

```python
arr = np.array([[1, 3, 6],
                [10, 15, 21]])

np.diff(arr, axis=1)  # row-wise: [[2, 3], [5, 6]]
np.diff(arr, axis=0)  # column-wise: [[9, 12, 15]]
```

---

## <span style="font-size: 16px;">Combining Tile and Diff</span>

<span style="font-size: 14px;">For this problem, repeat the entire matrix vertically, then compare consecutive rows and append one final zero row:</span>

```python
a = np.array([[1, 2], [3, 5]], dtype=np.float64)
tiled = np.tile(a, (3, 1))
diffs = np.diff(tiled, axis=0)
result = np.stack([tiled, np.pad(diffs, ((0, 1), (0, 0)))])
```

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Discrete derivative**: Consecutive differences must be divided by the sampling interval to approximate a derivative; np.diff alone computes only the numerator</span>
* <span style="font-size: 14px;">**Stationarity test**: If second differences are approximately constant, the series is approximately quadratic</span>
* <span style="font-size: 14px;">**Pattern repetition**: Tiling creates periodic signals, repeated masks, or repeated grids of blocks; it does not construct block-diagonal matrices by itself</span>
* <span style="font-size: 14px;">**Data augmentation**: Tile templates to create repeated patterns for training data</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Length reduction**: `np.diff` reduces array length by 1 (or by $n$ for $n$-th order differences). Account for this in downstream operations.</span>
* <span style="font-size: 14px;">**tile vs. broadcast**: For many operations, broadcasting achieves the same result as tiling without allocating memory for the repeated copies. Use tile only when you need the actual repeated data.</span>
* <span style="font-size: 14px;">**axis in diff**: Default axis is the last axis ($-1$). Specify `axis=0` explicitly for column-wise differences on 2D arrays.</span>