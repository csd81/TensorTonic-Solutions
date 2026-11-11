# <span style="font-size: 20px;">Angle Features</span>

<span style="font-size: 14px;">Trigonometric functions - sine, cosine, and tangent - are the bridge between angular measurements and Cartesian coordinates. In machine learning and signal processing, converting angles to their sine and cosine components is a standard feature engineering technique because it preserves the circular nature of angular data. NumPy's trigonometric functions operate element-wise on arrays, processing millions of angles in a single vectorized call.</span>

---

## <span style="font-size: 16px;">Core Trigonometric Functions</span>

```python
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

np.sin(angles)   # [0.0, 0.5, 0.707, 0.866, 1.0]
np.cos(angles)   # [1.0, 0.866, 0.707, 0.5, 0.0]
np.tan(angles)   # approximately [0.0, 0.577, 1.0, 1.732, 1.633e16]
```

<span style="font-size: 14px;">All functions expect input in radians, not degrees. To convert:</span>

```python
np.radians(180)     # pi
np.degrees(np.pi)   # 180.0
np.deg2rad(90)      # pi/2
np.rad2deg(np.pi/4) # 45.0
```

---

## <span style="font-size: 16px;">Why Encode Angles as (sin, cos)?</span>

<span style="font-size: 14px;">Raw angles have a discontinuity: 359 degrees and 1 degree are numerically far apart (358) but geometrically close. Encoding as (sin, cos) eliminates this problem:</span>

* <span style="font-size: 14px;">$\sin(359°) = -0.017$, $\sin(1°) = 0.017$ - close</span>
* <span style="font-size: 14px;">$\cos(359°) = 0.9998$, $\cos(1°) = 0.9998$ - close</span>

<span style="font-size: 14px;">The Euclidean distance between the (sin, cos) representations correctly reflects the angular proximity. This encoding is used for:</span>

* <span style="font-size: 14px;">**Time features**: Hour of day (0-23), day of week (0-6), month of year (1-12)</span>
* <span style="font-size: 14px;">**Direction features**: Wind direction, heading, orientation</span>
* <span style="font-size: 14px;">**Phase features**: Signal phase, periodic patterns</span>

### <span style="font-size: 14px;">Encoding Formula</span>

<span style="font-size: 14px;">For a cyclic feature with period $T$ and value $x$:</span>

$$
s = \sin\left(\frac{2\pi x}{T}\right)
$$

$$
c = \cos\left(\frac{2\pi x}{T}\right)
$$

Here, s and c are the sine and cosine features.

```python
hour = np.array([0, 6, 12, 18, 23])
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

---

## <span style="font-size: 16px;">Stacking into Feature Arrays</span>

```python
angles = np.array([0.1, 0.5, 1.0, 2.0])
features = np.array([np.sin(angles), np.cos(angles), np.tan(angles)])
# shape (3, 4): each row is one feature type
```

<span style="font-size: 14px;">For a feature matrix where columns are features:</span>

```python
features = np.column_stack([np.sin(angles), np.cos(angles)])
# shape (4, 2)
```

---

## <span style="font-size: 16px;">Inverse Trigonometric Functions</span>

```python
np.arcsin(0.5)     # pi/6
np.arccos(0.5)     # pi/3
np.arctan(1.0)     # pi/4
np.arctan2(y, x)   # angle from x-axis to point (x, y), handles all quadrants
```

<span style="font-size: 14px;">`arctan2` is preferred over `arctan` because it correctly handles all four quadrants and avoids division by zero when $x = 0$.</span>

---

## <span style="font-size: 16px;">The Pythagorean Identity</span>

$$
\sin^2(\theta) + \cos^2(\theta) = 1
$$

<span style="font-size: 14px;">This identity means the (sin, cos) encoding always lies on the unit circle. It provides a natural normalization that ML algorithms can exploit.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Radians vs. degrees**: NumPy trig functions use radians. Passing degrees gives wrong results without conversion.</span>
* <span style="font-size: 14px;">**Tangent at $\pi/2$**: $\tan(\pi/2)$ is mathematically undefined. NumPy returns a very large number, not inf, due to floating-point imprecision.</span>
* <span style="font-size: 14px;">**Encoding only sin (not cos)**: Using only sin loses information. $\sin(30°) = \sin(150°) = 0.5$. Both sin and cos are needed to uniquely identify the angle.</span>
* <span style="font-size: 14px;">**Period mismatch**: For hour-of-day encoding, the period is 24, not 12. Using the wrong period creates aliases.</span>