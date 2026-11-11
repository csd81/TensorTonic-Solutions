# <span style="font-size: 20px;">Sort and Argsort</span>

<span style="font-size: 14px;">Sorting arrays and finding the indices that would sort them are fundamental operations in data analysis. `np.sort()` returns a sorted copy, while `np.argsort()` returns the index permutation that would produce the sorted order. Together, they enable ranking, top-k selection, inverse permutations, and sorting one array by the order of another.</span>

---

## <span style="font-size: 16px;">np.sort()</span>

```python
a = np.array([30, 10, 50, 20, 40])
np.sort(a)  # [10, 20, 30, 40, 50]
```

<span style="font-size: 14px;">`np.sort()` returns a new sorted array. The original is unchanged. For in-place sorting:</span>

```python
a.sort()  # modifies a in-place
```

### <span style="font-size: 14px;">Sorting Along an Axis</span>

```python
arr = np.array([[3, 1, 2],
                [6, 4, 5]])

np.sort(arr, axis=1)  # sort each row: [[1,2,3], [4,5,6]]
np.sort(arr, axis=0)  # sort each column: [[3,1,2], [6,4,5]]
```

---

## <span style="font-size: 16px;">np.argsort()</span>

<span style="font-size: 14px;">Returns the indices that would sort the array:</span>

```python
a = np.array([30, 10, 50, 20])
idx = np.argsort(a)  # [1, 3, 0, 2]
a[idx]               # [10, 20, 30, 50] - sorted
```

<span style="font-size: 14px;">`argsort` is more useful than `sort` because the index array can be used to reorder other arrays in the same way:</span>

```python
names = np.array(['c', 'a', 'd', 'b'])
values = np.array([30, 10, 50, 20])
order = np.argsort(values)
names[order]  # ['a', 'b', 'c', 'd'] - sorted by values
```

---

## <span style="font-size: 16px;">Descending Sort</span>

<span style="font-size: 14px;">NumPy sorts ascending by default. For descending order:</span>

```python
np.sort(a)[::-1]        # sort ascending, then reverse
a[np.argsort(-a)]       # argsort of negated values
a[np.argsort(a)[::-1]]  # reverse the argsort indices
```

---

## <span style="font-size: 16px;">Sorting Algorithms</span>

```python
np.sort(a, kind='quicksort')   # default, O(n log n) average
np.sort(a, kind='mergesort')   # stable, O(n log n) guaranteed
np.sort(a, kind='heapsort')    # O(n log n), but slow in practice
np.sort(a, kind='stable')      # preserves the order of equal values
```

<span style="font-size: 14px;">Stable sorting preserves the relative order of equal elements. This matters when sorting records by one field while preserving a previous sort order on another field.</span>

---

## <span style="font-size: 16px;">Top-K Selection</span>

<span style="font-size: 14px;">For finding the k smallest or largest elements without full sorting:</span>

```python
# np.partition: O(n) partial sort
k = 3
np.partition(a, k)[:k]    # k smallest elements (not sorted)
np.partition(a, -k)[-k:]  # k largest elements (not sorted)

# np.argpartition: indices of k smallest/largest
idx = np.argpartition(a, k)[:k]
```

<span style="font-size: 14px;">Partition is $O(n)$ compared to sort's $O(n \log n)$, making it much faster for large arrays when you only need the top/bottom k.</span>

---

## <span style="font-size: 16px;">Ranking</span>

<span style="font-size: 14px;">`argsort` of `argsort` gives the rank of each element:</span>

```python
a = np.array([30, 10, 50, 20])
ranks = np.argsort(np.argsort(a))  # [2, 0, 3, 1]
# 30 has rank 2, 10 has rank 0, 50 has rank 3, 20 has rank 1
```

---

## <span style="font-size: 16px;">np.lexsort() for Multi-Key Sorting</span>

```python
names = np.array(['b', 'a', 'a', 'b'])
ages = np.array([30, 25, 30, 25])

# Sort by name first, then by age (last key is primary)
order = np.lexsort((ages, names))
# Sorts primarily by names, breaks ties with ages
```

<span style="font-size: 14px;">Note: `lexsort` sorts by the last key first (reversed from intuition). The last array in the tuple is the primary sort key.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**In-place vs. copy**: `np.sort(a)` returns a copy; `a.sort()` sorts in-place. They look similar but behave differently.</span>
* <span style="font-size: 14px;">**lexsort key order**: The last key is the primary sort key, which is counterintuitive.</span>
* <span style="font-size: 14px;">**argsort on multidimensional**: `np.argsort(arr, axis=1)` returns indices per row, not global indices. You need `np.take_along_axis` to apply them.</span>
* <span style="font-size: 14px;">**NaN in sorting**: NaN values sort to the end with `np.sort`. This may or may not be desired.</span>