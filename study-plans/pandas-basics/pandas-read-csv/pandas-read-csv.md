# <span style="font-size: 20px;">Create DataFrame from Dict</span>

<span style="font-size: 14px;">The most fundamental way to create a pandas DataFrame is from a Python dictionary. This operation is the entry point for nearly every pandas tutorial and the building block for test data, configuration tables, and data transformations where you construct results programmatically. Understanding the constructor's behavior, the relationship between dictionaries and DataFrames, and how to extract structured information from the result is essential for productive pandas work.</span>

---

## <span style="font-size: 16px;">The DataFrame Constructor</span>

<span style="font-size: 14px;">The basic pattern is to pass a dictionary where keys become column names and values are lists (or arrays) of column data:</span>

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000.0, 60000.0, 70000.0]
}
df = pd.DataFrame(data)
```

<span style="font-size: 14px;">The resulting DataFrame has three columns with names matching the dictionary keys, and the row index defaults to `RangeIndex(start=0, stop=3, step=1)`.</span>

### <span style="font-size: 14px;">Value Length Requirement</span>

<span style="font-size: 14px;">All lists in the dictionary must have the same length. If they differ, pandas raises a `ValueError`:</span>

```python
pd.DataFrame({'a': [1, 2], 'b': [10, 20, 30]})
# ValueError: All arrays must be of the same length
```

<span style="font-size: 14px;">This constraint ensures the DataFrame remains rectangular. If your data has missing values, use `None` or `np.nan` as placeholders.</span>

---

## <span style="font-size: 16px;">Column Ordering</span>

<span style="font-size: 14px;">In Python 3.7+, dictionaries maintain insertion order. The DataFrame constructor preserves this order, so columns appear in the same sequence as the dictionary keys:</span>

```python
data = {'z': [1], 'a': [2], 'm': [3]}
df = pd.DataFrame(data)
df.columns  # Index(['z', 'a', 'm'])
```

<span style="font-size: 14px;">If you need a specific column order different from the dictionary, pass the `columns` parameter:</span>

```python
df = pd.DataFrame(data, columns=['a', 'm', 'z'])
```

---

## <span style="font-size: 16px;">Type Inference</span>

<span style="font-size: 14px;">The constructor infers the dtype of each column from its values:</span>

* <span style="font-size: 14px;">All integers: `int64`</span>
* <span style="font-size: 14px;">Any float or `None` mixed with integers: `float64`</span>
* <span style="font-size: 14px;">All strings: `object`</span>
* <span style="font-size: 14px;">Mixed types: `object`</span>

<span style="font-size: 14px;">You can override inference with the `dtype` parameter:</span>

```python
df = pd.DataFrame({'id': [1, 2, 3]}, dtype='float32')
```

---

## <span style="font-size: 16px;">Extracting Structural Information</span>

<span style="font-size: 14px;">After creating a DataFrame, you typically inspect its structure:</span>

```python
df.shape          # (n_rows, n_cols) tuple
df.columns        # Index of column names
df.dtypes         # Series of column dtypes
df.index          # Row index
df.values         # Underlying numpy array
df.to_dict()      # Convert back to dictionary
```

### <span style="font-size: 14px;">Converting to Lists and Dicts</span>

<span style="font-size: 14px;">Pandas provides multiple serialization formats:</span>

```python
df.to_dict('list')    # {'col': [values]}
df.to_dict('records') # [{'col': val}, {'col': val}]
df.to_dict('index')   # {idx: {'col': val}}
df.values.tolist()    # [[row1], [row2], ...]
```

<span style="font-size: 14px;">The `'records'` orientation is especially useful for JSON APIs, where each row becomes a separate JSON object.</span>

---

## <span style="font-size: 16px;">Alternative Constructors</span>

<span style="font-size: 14px;">Beyond dictionaries, DataFrames can be created from many sources:</span>

### <span style="font-size: 14px;">From a List of Dictionaries (Row-Oriented)</span>

```python
rows = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
]
df = pd.DataFrame(rows)
```

<span style="font-size: 14px;">Each dictionary represents a row. Missing keys become NaN. This format is common when reading JSON arrays or API responses.</span>

### <span style="font-size: 14px;">From a 2D NumPy Array</span>

```python
import numpy as np
arr = np.array([[1, 2], [3, 4], [5, 6]])
df = pd.DataFrame(arr, columns=['a', 'b'])
```

### <span style="font-size: 14px;">From a List of Lists</span>

```python
data = [[1, 'Alice'], [2, 'Bob']]
df = pd.DataFrame(data, columns=['id', 'name'])
```

---

## <span style="font-size: 16px;">Custom Index</span>

<span style="font-size: 14px;">By default, the constructor creates a `RangeIndex` (0, 1, 2, ...). You can specify a custom index:</span>

```python
df = pd.DataFrame(data, index=['row_a', 'row_b', 'row_c'])
```

<span style="font-size: 14px;">Or set the index after creation:</span>

```python
df = df.set_index('id')
```

<span style="font-size: 14px;">The index is fundamental to pandas' alignment behavior: operations between DataFrames automatically align on their index, which prevents mismatched-row bugs.</span>

---

## <span style="font-size: 16px;">Empty DataFrames</span>

<span style="font-size: 14px;">You can create an empty DataFrame with a predefined schema:</span>

```python
df = pd.DataFrame(columns=['name', 'age', 'salary'])
df.dtypes  # all object (no data to infer from)
```

<span style="font-size: 14px;">Or with explicit types:</span>

```python
df = pd.DataFrame({'name': pd.Series(dtype='str'),
                    'age': pd.Series(dtype='int64')})
```

<span style="font-size: 14px;">Empty DataFrames with defined schemas are useful as accumulators in loops or as templates for validation.</span>

---

## <span style="font-size: 16px;">Performance: Dict vs. Other Sources</span>

<span style="font-size: 14px;">Constructing a DataFrame once from complete column lists avoids repeatedly allocating and copying a growing table. Python lists are converted to arrays; do not assume the constructor shares their storage.</span>

```python
# Fast: column-oriented construction
df = pd.DataFrame({'a': list(range(1000000))})

# Slow: row-by-row construction
df = pd.DataFrame()
for i in range(1000000):
    df = pd.concat([df, pd.DataFrame({'a': [i]})])
```

<span style="font-size: 14px;">The row-by-row approach creates a new DataFrame at each iteration and copies all previous data. For large datasets, always accumulate data in lists first, then construct the DataFrame once.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Unequal list lengths**: All value lists must have the same length. Use None/NaN for missing values.</span>
* <span style="font-size: 14px;">**Scalar values without index**: $\texttt{pd.DataFrame(\{'a': 1\})}$ raises a ValueError. Either wrap in a list ($\texttt{\{'a': [1]\}}$) or provide an explicit index.</span>
* <span style="font-size: 14px;">**Mutating the source dict**: After construction, modifying the original lists does not affect the DataFrame (data is copied). However, if you pass numpy arrays, the DataFrame may share memory.</span>
* <span style="font-size: 14px;">**Object dtype for mixed data**: A column with `[1, 'two', 3.0]` becomes object dtype, and numeric operations will fail silently or raise errors.</span>