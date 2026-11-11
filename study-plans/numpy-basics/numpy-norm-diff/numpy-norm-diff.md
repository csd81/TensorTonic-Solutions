# <span style="font-size: 20px;">Normalized Difference</span>

<span style="font-size: 14px;">This problem compares absolute differences after clipping both arrays to the supplied interval and rescaling that interval to [0, 1]. The output lies in [0, 1]. NDVI, discussed separately below, uses a different normalized-ratio definition.</span>

---

## <span style="font-size: 16px;">Mathematical Definition</span>

<span style="font-size: 14px;">For two arrays $a$ and $b$ of the same shape:</span>

$$
\text{ND}(a,b)=\frac{|\operatorname{clip}(a,\mathrm{lo},\mathrm{hi})-\operatorname{clip}(b,\mathrm{lo},\mathrm{hi})|}{\mathrm{hi}-\mathrm{lo}}
$$

<span style="font-size: 14px;">Properties:</span>

* <span style="font-size: 14px;">Range: [0, 1] because both clipped values lie in the same interval</span>
* <span style="font-size: 14px;">The result is zero when the clipped values are equal</span>
* <span style="font-size: 14px;">The result is one when the clipped values are at opposite interval endpoints</span>
* <span style="font-size: 14px;">Swapping the two inputs leaves the absolute difference unchanged</span>
* <span style="font-size: 14px;">The interval width is positive, so two zero inputs do not cause division by zero</span>

---

## <span style="font-size: 16px;">np.clip() for Range Bounding</span>

<span style="font-size: 14px;">Before computing the normalized difference, inputs often need to be clipped to a valid range:</span>

```python
a_clipped = np.clip(a, lo, hi)
b_clipped = np.clip(b, lo, hi)
```

<span style="font-size: 14px;">`np.clip(x, lo, hi)` clamps every element:</span>

$$
\text{clip}(x_i) = \begin{cases} lo & \text{if } x_i < lo \\ x_i & \text{if } lo \leq x_i \leq hi \\ hi & \text{if } x_i > hi \end{cases}
$$

<span style="font-size: 14px;">Clipping is essential for removing outliers or ensuring physical constraints (e.g., reflectance values must be in $[0, 1]$).</span>

---

## Rescaling and Absolute Difference

The denominator is the positive interval width, not the sum of the inputs. Both arrays use the same lower and upper bounds. Subtracting the common lower bound cancels when the normalized arrays are compared.

```python
def norm_diff(a, b, lo, hi):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_scaled = (np.clip(a, lo, hi) - lo) / (hi - lo)
    b_scaled = (np.clip(b, lo, hi) - lo) / (hi - lo)
    return np.abs(a_scaled - b_scaled)
```

---

## <span style="font-size: 16px;">Applications</span>

### <span style="font-size: 14px;">NDVI (Vegetation Index)</span>

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
$$

<span style="font-size: 14px;">This ratio is a separate remote-sensing measure, not the operation requested here. NIR and Red denote near-infrared and red reflectance. Vegetation commonly has positive NDVI, while water commonly has negative NDVI.</span>

### <span style="font-size: 14px;">Feature Engineering</span>

<span style="font-size: 14px;">The ratio used by NDVI is unchanged when both inputs are multiplied by the same positive scale and its denominator is nonzero. The operation in this problem is unchanged only when the interval bounds are scaled along with the inputs.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Interval width**: The denominator is the upper bound minus the lower bound. The constraints guarantee it is positive.</span>
* <span style="font-size: 14px;">**Precision**: Float64 conversion fixes intermediate precision; ordinary true division does not perform integer truncation.</span>
* <span style="font-size: 14px;">**Clip order**: Reversed clipping bounds fill the result with the supplied upper clipping bound, rather than the intended clipped values.</span>
* <span style="font-size: 14px;">**Shapes**: This problem guarantees matching shapes. More generally, unequal shapes may broadcast to an unintended result or fail when incompatible.</span>