# <span style="font-size: 20px;">Column Selection</span>

<span style="font-size: 14px;">Selecting columns is the most basic operation in DataFrame manipulation. Pandas provides multiple syntaxes for column access, each with different trade-offs in readability, safety, and flexibility. Understanding when to use bracket notation versus dot notation, and how selection interacts with the DataFrame's internal data structures, prevents common bugs and performance issues.</span>

---

## <span style="font-size: 16px;">Bracket Notation</span>

<span style="font-size: 14px;">The primary way to select a single column is bracket notation with a string key:</span>

```python
series = df['column_name']
```

<span style="font-size: 14px;">This returns a `Series` object, which is a one-dimensional labeled array. The Series shares the same index as the parent DataFrame, and modifying it may or may not affect the original DataFrame depending on whether a copy or view was returned.</span>

### <span style="font-size: 14px;">Why Bracket Notation is Preferred</span>

* <span style="font-size: 14px;">**Works with any column name**: Column names with spaces, special characters, or names that conflict with DataFrame methods (like "count", "values", "shape") require bracket notation.</span>
* <span style="font-size: 14px;">**Unambiguous**: `df['mean']` always accesses the column named "mean". df.mean refers to the aggregation method rather than a column.</span>
* <span style="font-size: 14px;">**Works with variables**: When the column name is stored in a variable, bracket notation is the only option: `df[col_name]`.</span>

---

## <span style="font-size: 16px;">Dot Notation</span>

<span style="font-size: 14px;">Pandas also supports attribute-style access:</span>

```python
series = df.column_name
```

<span style="font-size: 14px;">This is purely a convenience shorthand. It works only when the column name:</span>

* <span style="font-size: 14px;">Is a valid Python identifier (no spaces, no starting with a digit)</span>
* <span style="font-size: 14px;">Does not conflict with an existing DataFrame attribute or method</span>

<span style="font-size: 14px;">Because of these restrictions, dot notation is generally discouraged in production code. In interactive exploration it saves keystrokes, but bracket notation is always safer.</span>

### <span style="font-size: 14px;">The Shadowing Problem</span>

<span style="font-size: 14px;">Consider a DataFrame with a column named "count":</span>

```python
df = pd.DataFrame({'count': [1, 2, 3], 'value': [10, 20, 30]})
df.count    # returns the count METHOD, not the column
df['count'] # returns the column Series
```

<span style="font-size: 14px;">This is a source of insidious bugs. The expression `df.count` silently returns a bound method instead of the data. No error is raised, but downstream operations fail in confusing ways.</span>

---

## <span style="font-size: 16px;">Series vs. DataFrame Return Types</span>

<span style="font-size: 14px;">A critical distinction in pandas is whether a selection returns a Series or a DataFrame:</span>

```python
df['name']     # Series
df[['name']]   # DataFrame with one column
```

<span style="font-size: 14px;">The difference is the double brackets. `df[['name']]` passes a list of column names, and selecting with a list always returns a DataFrame, even if the list has one element. This matters because many operations expect a DataFrame, not a Series:</span>

```python
# Both include the column name as a header by default.
df[['name']].to_csv('output.csv')
df['name'].to_csv('output.csv')
```

---

## <span style="font-size: 16px;">Extracting Values</span>

<span style="font-size: 14px;">Once you have a Series, there are several ways to extract the raw values:</span>

```python
df['age'].values        # numpy array
df['age'].tolist()      # Python list
df['age'].to_numpy()    # numpy array (preferred over .values)
```

<span style="font-size: 14px;">`.tolist()` is the most common for interoperability with non-pandas code. It returns a plain Python list. Lists are unhashable and can be serialized to JSON when all their values are JSON-compatible.</span>

<span style="font-size: 14px;">`.to_numpy()` is preferred over `.values` because it offers dtype control and handles extension types (nullable integers, strings) correctly.</span>

---

## <span style="font-size: 16px;">Column Selection by Data Type</span>

<span style="font-size: 14px;">Pandas provides `select_dtypes()` for selecting columns based on their data type:</span>

```python
numeric_df = df.select_dtypes(include='number')       # all numeric columns
text_df = df.select_dtypes(include='object')           # all string/object columns
non_numeric_df = df.select_dtypes(exclude='number')    # everything except numeric
```

<span style="font-size: 14px;">This is essential for preprocessing pipelines where numeric columns need scaling and categorical columns need encoding. Instead of manually listing column names, you let the types drive the selection.</span>

---

## <span style="font-size: 16px;">Column Selection with filter()</span>

<span style="font-size: 14px;">The `df.filter()` method selects columns by name pattern:</span>

```python
df.filter(like='price')    # columns containing 'price'
df.filter(regex='^feat_')  # columns matching regex
df.filter(items=['a','b']) # specific columns by name
```

<span style="font-size: 14px;">This is particularly useful when working with feature-engineered datasets where column names follow a pattern (e.g., "feat_1", "feat_2", ...).</span>

---

## <span style="font-size: 16px;">Chained Column Access</span>

<span style="font-size: 14px;">Column selection can be chained with other operations:</span>

```python
df['salary'].mean()                    # scalar
df['salary'].value_counts()            # frequency table
df['salary'].describe()                # summary statistics
df['salary'].plot.hist(bins=30)        # histogram
```

<span style="font-size: 14px;">Because selection returns a Series, all Series methods are immediately available. This makes pandas highly composable: you select, then operate, then select again.</span>

---

## <span style="font-size: 16px;">Views vs. Copies</span>

<span style="font-size: 14px;">Single-column selection may return a view (a reference to the original data) rather than an independent copy. This means modifying the selected Series can modify the original DataFrame:</span>

```python
s = df['age']
s.iloc[0] = 999  # may modify df['age'].iloc[0]
```

<span style="font-size: 14px;">Pandas issues a `SettingWithCopyWarning` when this ambiguity arises. To be safe, explicitly copy when you intend to modify:</span>

```python
s = df['age'].copy()
s.iloc[0] = 999  # df is untouched
```

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Using dot notation for column names that shadow methods**: `df.count`, `df.sum`, `df.mean` all return methods, not columns. Always use brackets.</span>
* <span style="font-size: 14px;">**Single vs. double brackets**: `df['col']` gives a Series; `df[['col']]` gives a DataFrame. Mixing these up causes type errors downstream.</span>
* <span style="font-size: 14px;">**Modifying a view**: Assigning to a selected Series without copying can corrupt the original DataFrame. Use `.copy()` when in doubt.</span>
* <span style="font-size: 14px;">**KeyError on misspelled columns**: Unlike dot notation (which raises `AttributeError`), bracket notation raises `KeyError`. Check `df.columns` to verify the exact column name.</span>