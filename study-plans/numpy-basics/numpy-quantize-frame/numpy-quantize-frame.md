# <span style="font-size: 20px;">Quantize and Frame</span>

<span style="font-size: 14px;">Quantization rounds continuous values to discrete levels, while framing surrounds an array with a border of constant values (typically zeros). These two operations, when combined, form a common preprocessing pipeline: discretize the data to a fixed precision, then add padding for boundary handling. NumPy provides `np.floor`, `np.ceil`, `np.round` for quantization and `np.pad` for framing.</span>

---

## <span style="font-size: 16px;">Rounding Modes</span>

<span style="font-size: 14px;">NumPy offers four distinct rounding behaviors:</span>

| Function | Direction | Example: 2.7 | Example: -2.3 |
|----------|-----------|--------------|---------------|
| `np.floor` | Toward $-\infty$ | 2.0 | -3.0 |
| `np.ceil` | Toward $+\infty$ | 3.0 | -2.0 |
| `np.trunc` | Toward 0 | 2.0 | -2.0 |
| `np.round` | Nearest (banker's) | 3.0 | -2.0 |

```python
x = np.array([2.3, 2.7, -1.5, -2.3])
np.floor(x)   # [ 2.,  2., -2., -3.]
np.ceil(x)    # [ 3.,  3., -1., -2.]
np.trunc(x)   # [ 2.,  2., -1., -2.]
np.round(x)   # [ 2.,  3., -2., -2.] - banker's rounding for 0.5
```

### <span style="font-size: 14px;">Banker's Rounding</span>

<span style="font-size: 14px;">`np.round` uses "round half to even" (banker's rounding): 0.5 rounds to the nearest even number. This prevents systematic bias:</span>

```python
np.round(0.5)   # 0.0 (rounds to even)
np.round(1.5)   # 2.0 (rounds to even)
np.round(2.5)   # 2.0 (rounds to even)
```

---

## <span style="font-size: 16px;">Quantization to a Grid</span>

<span style="font-size: 14px;">To quantize values to a specific step size:</span>

```python
step = 0.25
quantized = np.round(x / step) * step
# Rounds to nearest multiple of 0.25
```

<span style="font-size: 14px;">This is used in signal processing (bit-depth reduction), pricing (rounding to cents), and model compression (weight quantization).</span>

---

## <span style="font-size: 16px;">np.pad() for Framing</span>

```python
arr = np.array([[1, 2], [3, 4]])

np.pad(arr, 1, mode='constant', constant_values=0)
# [[0, 0, 0, 0],
#  [0, 1, 2, 0],
#  [0, 3, 4, 0],
#  [0, 0, 0, 0]]
```

### <span style="font-size: 14px;">Asymmetric Padding</span>

```python
np.pad(arr, ((1, 2), (3, 0)), mode='constant', constant_values=0)
# 1 row on top, 2 on bottom, 3 columns on left, 0 on right
```

### <span style="font-size: 14px;">Other Padding Modes</span>

```python
np.pad(arr, 1, mode='edge')      # replicate edge values
np.pad(arr, 1, mode='reflect')   # mirror reflection
np.pad(arr, 1, mode='wrap')      # periodic wrapping
```

---

## <span style="font-size: 16px;">Combined Pipeline</span>

<span style="font-size: 14px;">Apply three rounding modes and surround each with a zero border:</span>

```python
arr = np.asarray([[1.25, -2.75], [3.5, 4.125]], dtype=np.float64)
decimals = 1
pad_width = 1

layers = [np.round(arr, decimals), np.floor(arr), np.ceil(arr)]
results = [np.pad(layer, pad_width, mode="constant", constant_values=0.0)
           for layer in layers]
output = np.stack(results)  # shape (3, 4, 4)
```

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Image processing**: Quantize pixel values to reduce color depth; pad images for convolution borders</span>
* <span style="font-size: 14px;">**Neural network compression**: Quantize weights to 8-bit integers for faster inference</span>
* <span style="font-size: 14px;">**Signal processing**: Quantization simulates ADC (analog-to-digital conversion); padding enables FFT on non-power-of-2 lengths</span>
* <span style="font-size: 14px;">**Convolution**: Suitable zero-padding can preserve spatial dimensions for a stride-one convolution; the required width depends on the kernel and dilation</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Banker's rounding surprises**: `np.round(0.5) = 0` and `np.round(1.5) = 2` surprises users who expect "round half up."</span>
* <span style="font-size: 14px;">**Pad width format**: For asymmetric padding, `pad_width` takes `((before, after), ...)` per axis. Getting the nesting wrong causes shape errors.</span>
* <span style="font-size: 14px;">**Integer arrays after rounding**: `np.floor(arr)` returns float, not int. Use `np.floor(arr).astype(int)` for integers.</span>
* <span style="font-size: 14px;">**Padding increases size**: A $(m, n)$ array padded with width $p$ becomes $(m + 2p, n + 2p)$. Account for this in downstream shape calculations.</span>