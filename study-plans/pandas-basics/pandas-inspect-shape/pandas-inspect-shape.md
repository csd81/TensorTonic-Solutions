# <span style="font-size: 20px;">Inspect DataFrame Shape</span>

<span style="font-size: 14px;">Every data analysis task begins with understanding the structure of the dataset. Before filtering, transforming, or modeling, you need to know how many observations you have, what columns exist, and what types the columns store. Pandas provides a set of attributes and methods that answer these questions instantly, and learning to use them reflexively is one of the most important habits in practical data science.</span>

---

## <span style="font-size: 16px;">The Shape Attribute</span>

<span style="font-size: 14px;">The most fundamental structural property of a DataFrame is its shape:</span>

```python
df.shape  # returns (n_rows, n_cols)
```

<span style="font-size: 14px;">This returns a plain Python tuple of two integers. The first element is the number of rows (observations), and the second is the number of columns (features). Unlike calling `len(df)`, which only returns the row count, `shape` gives both dimensions in one call.</span>

<span style="font-size: 14px;">Shape is an attribute, not a method, so you never write `df.shape()` with parentheses. This distinction matters because calling an attribute with parentheses raises a `TypeError`.</span>

### <span style="font-size: 14px;">Why Shape Matters</span>

* <span style="font-size: 14px;">**Data validation**: After a merge, join, or filter, checking shape confirms you have the expected number of rows. A common bug is an accidental many-to-many merge that silently inflates the row count.</span>
* <span style="font-size: 14px;">**Memory estimation**: A DataFrame with shape $(m, n)$ holding float64 values uses roughly $8 \times m \times n$ bytes. Knowing shape lets you estimate whether a dataset fits in memory before loading it entirely.</span>
* <span style="font-size: 14px;">**Debugging pipelines**: When a transformation produces unexpected output, printing shape at each step is the fastest way to locate where the pipeline diverged.</span>

---

## <span style="font-size: 16px;">Column Names</span>

<span style="font-size: 14px;">The `df.columns` attribute returns an `Index` object containing the column labels:</span>

```python
df.columns        # Index(['name', 'age', 'salary'], dtype='object')
df.columns.tolist()  # ['name', 'age', 'salary']
```

<span style="font-size: 14px;">The Index behaves like an immutable array. You can iterate over it, check membership with `in`, and convert it to a list with `.tolist()`. Checking whether a column exists before accessing it prevents `KeyError` exceptions:</span>

```python
if 'revenue' in df.columns:
    total = df['revenue'].sum()
```

### <span style="font-size: 14px;">Column Count Shortcut</span>

<span style="font-size: 14px;">While `df.shape[1]` gives the column count, you can also use `len(df.columns)`. Both are constant-time operations, but `shape[1]` is more idiomatic when you only need the number, while `df.columns` is used when you need the actual names.</span>

---

## <span style="font-size: 16px;">Data Types</span>

<span style="font-size: 14px;">The `df.dtypes` attribute returns a Series mapping each column name to its data type:</span>

```python
df.dtypes
# name      object
# age        int64
# salary   float64
```

<span style="font-size: 14px;">Understanding dtypes is critical because:</span>

* <span style="font-size: 14px;">**object dtype** usually means strings, but can also hide mixed types (integers and strings in the same column). This happens silently when CSV parsing encounters inconsistent data.</span>
* <span style="font-size: 14px;">**int64 vs float64**: A column that should be integer but contains missing values (`NaN`) will be upcast to float64, because `NaN` is a float. Pandas 1.0+ offers `Int64` (nullable integer) to avoid this.</span>
* <span style="font-size: 14px;">**datetime64**: Date columns loaded as strings will behave incorrectly in time-based operations until converted with `pd.to_datetime()`.</span>
* <span style="font-size: 14px;">**category**: Columns with a small number of unique values (like gender or country) can be converted to category dtype for significant memory savings and faster groupby operations.</span>

---

## <span style="font-size: 16px;">The size Attribute</span>

<span style="font-size: 14px;">`df.size` returns the total number of elements in the DataFrame:</span>

$$
\text{total cells} = \text{rows} \times \text{columns}
$$

<span style="font-size: 14px;">This counts every cell, including those containing `NaN`. It is rarely used directly in analysis, but it is useful for quick sanity checks: if you expect a 1000-row, 10-column dataset, size should be 10000.</span>

---

## <span style="font-size: 16px;">The info() Method</span>

<span style="font-size: 14px;">For a comprehensive overview, `df.info()` prints a summary that combines shape, dtypes, non-null counts, and memory usage:</span>

```python
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    1000 non-null   object
#  1   age     985 non-null    float64
#  2   salary  990 non-null    float64
# dtypes: float64(2), object(1)
# memory usage: 23.5 KB
```

<span style="font-size: 14px;">The non-null counts are especially valuable because they immediately reveal missing data. If a column shows 985 non-null out of 1000 rows, you know 15 values are missing before writing any explicit checks.</span>

### <span style="font-size: 14px;">Memory Usage</span>

<span style="font-size: 14px;">By default, `info()` shows a rough memory estimate. Pass `memory_usage='deep'` to get the true memory consumption, which accounts for the actual size of Python objects stored in object-dtype columns. The deep calculation is slower but accurate:</span>

```python
df.info(memory_usage='deep')
```

<span style="font-size: 14px;">For large datasets, this is essential for deciding whether to dowcast numeric types (e.g., float64 to float32) or convert strings to categories.</span>

---

## <span style="font-size: 16px;">The describe() Method</span>

<span style="font-size: 14px;">While `info()` shows structural metadata, `describe()` provides statistical summaries:</span>

```python
df.describe()
#              age       salary
# count   985.00      990.00
# mean     34.50    65000.00
# std      12.30    15000.00
# min      18.00    30000.00
# 25%      25.00    55000.00
# 50%      33.00    63000.00
# 75%      43.00    75000.00
# max      65.00   120000.00
```

<span style="font-size: 14px;">By default it only summarizes numeric columns. Pass `include='all'` to include object and category columns, which adds count, unique, top (most frequent value), and freq (frequency of the most frequent value).</span>

---

## <span style="font-size: 16px;">Practical Inspection Workflow</span>

<span style="font-size: 14px;">An experienced data scientist inspects a new dataset in a consistent order:</span>

1. <span style="font-size: 14px;">`df.shape` - how big is the data?</span>
2. <span style="font-size: 14px;">`df.columns.tolist()` - what features exist?</span>
3. <span style="font-size: 14px;">`df.dtypes` - are types correct?</span>
4. <span style="font-size: 14px;">`df.head()` - do the first few rows look reasonable?</span>
5. <span style="font-size: 14px;">`df.info()` - any missing values?</span>
6. <span style="font-size: 14px;">`df.describe()` - any outliers or suspicious distributions?</span>

<span style="font-size: 14px;">This six-step workflow catches the vast majority of data quality issues before any modeling code is written. Skipping this step is one of the most common causes of subtle bugs in data pipelines.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Calling shape as a method**: `df.shape()` raises `TypeError`. It is a property, not a function.</span>
* <span style="font-size: 14px;">**Confusing len(df) and df.size**: `len(df)` returns the row count; `df.size` returns rows times columns.</span>
* <span style="font-size: 14px;">**Ignoring object dtypes**: Object columns that should be numeric will silently fail in arithmetic. Always check dtypes after loading.</span>
* <span style="font-size: 14px;">**Not checking shape after merges**: A merge on non-unique keys can multiply your row count unexpectedly. Always compare `df.shape[0]` before and after.</span>