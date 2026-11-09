# <span style="font-size: 20px;">Data Types Overview</span>

<span style="font-size: 14px;">Every column in a pandas DataFrame has a data type (dtype) that determines how values are stored in memory, what operations are valid, and how much memory the column consumes. Choosing the right dtype is not just a performance concern: incorrect types cause silent bugs where operations produce wrong results instead of raising errors. A thorough understanding of the pandas type system is essential for writing correct data pipelines.</span>

---

## <span style="font-size: 16px;">Core Data Types</span>

<span style="font-size: 14px;">Pandas inherits most of its types from NumPy and adds several of its own extension types:</span>

### <span style="font-size: 14px;">Numeric Types</span>

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `int64` | 8 bytes | $-2^{63}$ to $2^{63}-1$ | Default integer type |
| `int32` | 4 bytes | $-2^{31}$ to $2^{31}-1$ | Memory optimization |
| `float64` | 8 bytes | $\pm 1.8 \times 10^{308}$ | Default float, supports NaN |
| `float32` | 4 bytes | $\pm 3.4 \times 10^{38}$ | Memory optimization |
| `bool` | 1 byte | True/False | Boolean flags |

<span style="font-size: 14px;">The default types (`int64`, `float64`) are chosen for safety: they handle most real-world values without overflow. Downcasting to smaller types saves memory but risks overflow if values exceed the type's range.</span>

### <span style="font-size: 14px;">String and Object Types</span>

* <span style="font-size: 14px;">**object**: The legacy type for strings. Stores Python objects, so each element is a full Python string with reference counting overhead. A column of 1 million strings uses far more memory than a numeric column of the same length.</span>
* <span style="font-size: 14px;">**StringDtype** (`pd.StringDtype()`): Pandas' native string type, introduced in version 1.0. Uses pyarrow or Python strings internally. Offers proper NA handling and string-specific methods.</span>

### <span style="font-size: 14px;">Datetime Types</span>

* <span style="font-size: 14px;">**datetime64[ns]**: Timestamps with nanosecond precision. Used for dates and timestamps. Created by `pd.to_datetime()`.</span>
* <span style="font-size: 14px;">**timedelta64[ns]**: Duration between two timestamps. Created by subtracting datetime columns.</span>
* <span style="font-size: 14px;">**period**: Time spans (e.g., "January 2024"). Created with pd.Period or by converting a DatetimeIndex with its to_period method.</span>

### <span style="font-size: 14px;">Categorical Type</span>

<span style="font-size: 14px;">`category` stores data as integer codes mapping to a fixed set of values. A column with 1 million rows but only 50 unique cities stores 50 strings plus 1 million small integers instead of 1 million full strings.</span>

---

## <span style="font-size: 16px;">Inspecting Data Types</span>

<span style="font-size: 14px;">Three main tools for checking types:</span>

```python
df.dtypes          # Series: column name -> dtype
df.dtypes.value_counts()  # count of each dtype
df.info()          # columns, dtypes, non-null counts, memory
```

<span style="font-size: 14px;">`df.dtypes` is the quickest check. `df.info()` is more comprehensive because it also shows non-null counts, which indicate missing data.</span>

---

## <span style="font-size: 16px;">The NaN and Missing Data Problem</span>

<span style="font-size: 14px;">One of the most important dtype interactions in pandas is the relationship between missing values and types:</span>

* <span style="font-size: 14px;">**int64 columns cannot hold NaN**: `NaN` is a float value. If an integer column contains missing data, pandas automatically upcasts it to float64. This means a column of IDs `[1, 2, NaN, 4]` becomes `[1.0, 2.0, NaN, 4.0]`.</span>
* <span style="font-size: 14px;">**Nullable integer types**: Pandas provides `Int64` (capital I), `Int32`, etc., which support `pd.NA` as a missing value indicator without converting to float.</span>
* <span style="font-size: 14px;">**Boolean with NaN**: Similarly, a boolean column with missing values becomes object type. Use `pd.BooleanDtype()` for nullable booleans.</span>

```python
# Standard int - upcasts to float on NaN
pd.Series([1, 2, None])  # dtype: float64

# Nullable int - stays integer
pd.Series([1, 2, None], dtype='Int64')  # dtype: Int64
```

---

## <span style="font-size: 16px;">Type Conversion</span>

<span style="font-size: 14px;">The primary method for converting types is `astype()`:</span>

```python
df['age'] = df['age'].astype('int32')
df['name'] = df['name'].astype('category')
df['flag'] = df['flag'].astype(bool)
```

<span style="font-size: 14px;">`astype()` raises an error if the conversion is impossible (e.g., converting "hello" to int). For safer conversions, use specialized functions:</span>

```python
pd.to_numeric(df['col'], errors='coerce')     # invalid -> NaN
pd.to_datetime(df['col'], errors='coerce')     # invalid -> NaT
```

<span style="font-size: 14px;">The `errors='coerce'` parameter converts unparseable values to NaN/NaT instead of raising an exception, which is essential when dealing with dirty data.</span>

---

## <span style="font-size: 16px;">Automatic Type Inference</span>

<span style="font-size: 14px;">When loading data from CSV, pandas infers types by scanning the values:</span>

* <span style="font-size: 14px;">Columns with all integers become `int64`</span>
* <span style="font-size: 14px;">Columns with any decimal become `float64`</span>
* <span style="font-size: 14px;">Columns with mixed types become `object`</span>
* <span style="font-size: 14px;">Columns matching date patterns remain `object` unless `parse_dates` is specified</span>

<span style="font-size: 14px;">This inference is imperfect. A column of ZIP codes like "02134" is loaded as integer 2134, losing the leading zero. Always specify dtypes explicitly for critical columns:</span>

```python
df = pd.read_csv('data.csv', dtype={'zip': str, 'id': 'Int64'})
```

---

## <span style="font-size: 16px;">Memory Optimization</span>

<span style="font-size: 14px;">Choosing appropriate types can dramatically reduce memory usage. A DataFrame with 10 million rows:</span>

| Column Type | Per-Value | 10M Values |
|-------------|-----------|------------|
| float64 | 8 bytes | 80 MB |
| float32 | 4 bytes | 40 MB |
| int64 | 8 bytes | 80 MB |
| int8 | 1 byte | 10 MB |
| category (100 unique) | ~1 byte | ~10 MB |
| object (strings) | ~50+ bytes | 500+ MB |

<span style="font-size: 14px;">`pd.to_numeric(df['col'], downcast='integer')` automatically selects the smallest integer type that fits the data. Similarly, `downcast='float'` converts to float32 when possible.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Silent float upcasting**: Integer columns with NaN silently become float64. This can break ID columns, foreign keys, and hash-based joins.</span>
* <span style="font-size: 14px;">**Object dtype hiding mixed types**: A column with dtype "object" might contain integers, strings, and None mixed together. Use `df['col'].apply(type).value_counts()` to check.</span>
* <span style="font-size: 14px;">**String comparison on numeric objects**: Comparing strings with an integer raises a TypeError. Comparing two strings is lexicographic, so convert numeric text before numeric comparisons.</span>
* <span style="font-size: 14px;">**Category ordering**: By default, categories are unordered. Comparison operators (`<`, `>`) raise errors unless you create an ordered category with `pd.CategoricalDtype(ordered=True)`.</span>