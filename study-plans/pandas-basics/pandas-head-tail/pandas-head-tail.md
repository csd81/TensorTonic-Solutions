# <span style="font-size: 20px;">Head and Tail Operations</span>

<span style="font-size: 14px;">When working with datasets that have thousands or millions of rows, you never want to print the entire DataFrame. The `head()` and `tail()` methods let you peek at the beginning and end of the data, respectively. These are among the most frequently called methods in exploratory data analysis, and understanding their behavior in edge cases prevents subtle bugs.</span>

---

## <span style="font-size: 16px;">The head() Method</span>

<span style="font-size: 14px;">`df.head(n)` returns the first $n$ rows of the DataFrame as a new DataFrame:</span>

```python
df.head()    # first 5 rows (default)
df.head(10)  # first 10 rows
df.head(1)   # first row as a single-row DataFrame
```

<span style="font-size: 14px;">The default value of $n$ is 5, which is usually enough to verify column names, data types, and general formatting. Copying behavior depends on the pandas version and copy-on-write settings. Use an explicit copy before editing a selection independently.</span>

### <span style="font-size: 14px;">Negative Indexing with head()</span>

<span style="font-size: 14px;">A lesser-known feature is that `head()` accepts negative values. `df.head(-k)` returns all rows except the last $k$:</span>

```python
df.head(-3)  # all rows except the last 3
```

<span style="font-size: 14px;">This mirrors Python's slice notation: `df.head(-3)` is equivalent to `df.iloc[:-3]`. It is useful when the last few rows contain summary statistics or footer metadata that you want to exclude.</span>

---

## <span style="font-size: 16px;">The tail() Method</span>

<span style="font-size: 14px;">`df.tail(n)` returns the last $n$ rows:</span>

```python
df.tail()    # last 5 rows (default)
df.tail(10)  # last 10 rows
```

<span style="font-size: 14px;">Tail is the mirror image of head: it counts from the end of the DataFrame. Similarly, `df.tail(-k)` returns all rows except the first $k$:</span>

```python
df.tail(-2)  # all rows except the first 2
```

<span style="font-size: 14px;">This is equivalent to `df.iloc[2:]` and is useful when the first few rows contain header metadata or column descriptions that were accidentally loaded as data rows.</span>

---

## <span style="font-size: 16px;">Return Type and Index Preservation</span>

<span style="font-size: 14px;">Both methods return a DataFrame, not a list or array. The original index is preserved:</span>

```python
df = pd.DataFrame({'x': range(100)})
result = df.tail(3)
print(result.index)  # Int64Index([97, 98, 99])
```

<span style="font-size: 14px;">This means row labels from the original DataFrame carry over. If you need a fresh 0-based index, chain `.reset_index(drop=True)`:</span>

```python
result = df.tail(3).reset_index(drop=True)
print(result.index)  # RangeIndex(start=0, stop=3)
```

---

## <span style="font-size: 16px;">Edge Cases</span>

### <span style="font-size: 14px;">n Greater Than Length</span>

<span style="font-size: 14px;">If $n$ exceeds the number of rows, both methods return the entire DataFrame without raising an error:</span>

```python
df = pd.DataFrame({'x': [1, 2, 3]})
df.head(100)  # returns all 3 rows, no error
```

<span style="font-size: 14px;">This is intentional and matches Python's slice behavior: `[1, 2, 3][:100]` returns `[1, 2, 3]`.</span>

### <span style="font-size: 14px;">n Equal to Zero</span>

<span style="font-size: 14px;">`df.head(0)` and `df.tail(0)` return an empty DataFrame with the same columns and dtypes but no rows. This is useful for extracting the schema:</span>

```python
schema = df.head(0)  # empty DataFrame with correct columns/dtypes
```

### <span style="font-size: 14px;">Empty DataFrames</span>

<span style="font-size: 14px;">Calling head or tail on an empty DataFrame returns another empty DataFrame. No error is raised.</span>

---

## <span style="font-size: 16px;">Head and Tail on Series</span>

<span style="font-size: 14px;">Both methods also work on Series objects, returning a Series:</span>

```python
df['salary'].head(3)  # first 3 values as a Series
df['salary'].tail(3)  # last 3 values as a Series
```

<span style="font-size: 14px;">This is useful for quick column-level inspection without selecting the entire DataFrame.</span>

---

## <span style="font-size: 16px;">Comparison with iloc Slicing</span>

<span style="font-size: 14px;">For positive n, head and tail select the same rows as these iloc slices:</span>

| Method | Equivalent iloc |
|--------|----------------|
| `df.head(n)` | `df.iloc[:n]` |
| `df.tail(n)` | `df.iloc[-n:]` |
| `df.head(-n)` | `df.iloc[:-n]` |
| `df.tail(-n)` | `df.iloc[n:]` |

<span style="font-size: 14px;">The methods are preferred for readability, but `iloc` is more flexible when you need both row and column slicing simultaneously.</span>

---

## <span style="font-size: 16px;">Practical Patterns</span>

### <span style="font-size: 14px;">Verifying Sort Order</span>

<span style="font-size: 14px;">After sorting a DataFrame, check both ends to confirm the sort worked:</span>

```python
df_sorted = df.sort_values('date')
print(df_sorted.head(1))  # earliest date
print(df_sorted.tail(1))  # latest date
```

### <span style="font-size: 14px;">Sampling vs. Head</span>

<span style="font-size: 14px;">`df.head(5)` always returns the same first 5 rows, which may not be representative if the data is sorted. For a random sample, use `df.sample(5)` instead. Head is for structure inspection; sample is for distribution inspection.</span>

### <span style="font-size: 14px;">Pipeline Debugging</span>

<span style="font-size: 14px;">Insert `.head()` calls in method chains to inspect intermediate results without running the full pipeline:</span>

```python
# Debug: what does the data look like after filtering?
(df
 .query('age > 30')
 .head()  # inspect here
 .groupby('department')
 .mean()
)
```

---

## <span style="font-size: 16px;">Performance</span>

<span style="font-size: 14px;">These methods select rows by position. Depending on the pandas version and copy-on-write settings, the selection may share data or be copied. Work and memory use can depend on the number of selected rows and columns.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Assuming head() shows representative data**: If the DataFrame is sorted, the first few rows may all be from one category. Use `sample()` for a representative view.</span>
* <span style="font-size: 14px;">**Forgetting index preservation**: After `tail(3)`, the index starts at the original position, not 0. This can cause unexpected behavior in index-based lookups.</span>
* <span style="font-size: 14px;">**Confusing head/tail with first/last**: GroupBy objects have `first()` and `last()` methods that return the first/last non-null value per group, which is a completely different operation.</span>