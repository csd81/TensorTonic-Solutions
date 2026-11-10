## <span style="font-size: 20px;">Percentiles and Quantiles</span>

Percentiles partition ordered data into 100 equal parts, providing a complete picture of the distribution beyond simple measures of center and spread. The $p$-th percentile is the value below which $p\%$ of observations fall.

---

## Definitions

**Percentile:** The $p$-th percentile $P_p$ satisfies: at least $p\%$ of data $\leq P_p$ and at least $(100-p)\%$ of data $\geq P_p$.

**Quantile:** Quantiles are the general form. The $q$-th quantile (where $0 \leq q \leq 1$) equals the $(100q)$-th percentile. Common quantiles have special names:

| Name | Quantile | Percentile |
|------|----------|------------|
| Minimum | $Q_0$ | $P_0$ |
| First Quartile | $Q_{0.25}$ | $P_{25}$ |
| Median | $Q_{0.5}$ | $P_{50}$ |
| Third Quartile | $Q_{0.75}$ | $P_{75}$ |
| Maximum | $Q_1$ | $P_{100}$ |

---

## Linear Interpolation Method

When the percentile falls between two data points, linear interpolation provides a smooth estimate. For sorted data $x_{(1)} \leq x_{(2)} \leq \ldots \leq x_{(n)}$:

1. Compute the rank: $r = \frac{p}{100} \cdot (n - 1)$
2. Find the integer part $k = \lfloor r \rfloor$ and fractional part $f = r - k$
3. Interpolate: $P_p = x_{(k)} + f \cdot (x_{(k+1)} - x_{(k)})$

This is the default method used by `np.percentile()` (the "linear" method). Other methods exist (lower, higher, nearest, midpoint) but linear interpolation is most common.

---

## Interquartile Range (IQR)

The IQR measures the spread of the middle 50% of data:

$$\text{IQR} = Q_3 - Q_1 = P_{75} - P_{25}$$

The IQR is robust to outliers since it ignores the bottom 25% and top 25% of values.

**The 1.5*IQR Rule for Outlier Detection:**

A value $x$ is a suspected outlier if:

$$x < Q_1 - 1.5 \cdot \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \cdot \text{IQR}$$

Values beyond $3 \cdot \text{IQR}$ from the quartiles are considered extreme outliers. This rule forms the basis of box plot whiskers.

---

## Five-Number Summary and Box Plots

The five-number summary consists of: minimum, $Q_1$, median, $Q_3$, maximum. Box plots visualize this:

- The box spans $Q_1$ to $Q_3$ (the IQR)
- A line inside marks the median
- Whiskers extend to the most extreme non-outlier points
- Individual points beyond whiskers represent outliers

This provides a quick visual comparison of distributions across groups.

---

## Percentile Rank vs Percentile Score

These are inverse operations:
- **Percentile score:** Given a percentile $p$, find the value $P_p$
- **Percentile rank:** Given a value $x$, find what percentage of data falls below it

$$\text{Percentile Rank}(x) = \frac{\text{count of values} < x}{n} \times 100$$

---

## Applications in Machine Learning

**Data Cleaning:** The IQR rule identifies outliers without assumptions about distribution shape, unlike z-score methods that assume normality.

**Feature Engineering:** Percentile transforms (rank transforms) map any distribution to a uniform distribution. This is useful for features with heavy tails or non-linear relationships.

**Model Evaluation:** Percentile-based thresholds help set decision boundaries. For example, flagging the top 5% of fraud scores requires the 95th percentile as threshold.

**Robust Scaling:** Scaling by IQR ($\frac{x - \text{median}}{\text{IQR}}$) is more robust than standardization for data with outliers. This is implemented as `RobustScaler` in scikit-learn.

**Quantile Regression:** While ordinary regression predicts the conditional mean, quantile regression predicts conditional percentiles. This is valuable when you need to estimate prediction intervals or understand how different parts of the distribution respond to features.

---

## Common Pitfalls

- **Different interpolation methods give different answers:** NumPy supports several methods (linear, lower, higher, nearest, midpoint). The choice matters most for small datasets. Always specify the method explicitly when reproducibility matters.
- **Percentiles of discrete data:** For integer-valued data, interpolated percentiles may return non-integer values. Decide whether this is acceptable for your application or if you need a discrete quantile method.
- **IQR of zero:** If more than 50% of observations share the same value, the IQR can be zero. This breaks IQR-based outlier detection and robust scaling. Check for this edge case before applying these methods.