## <span style="font-size: 20px;">Skewness and Kurtosis</span>

Skewness and kurtosis are the **third and fourth standardized moments** of a distribution. They describe the shape of a distribution beyond what mean and variance capture - specifically, asymmetry and tail behavior.

---

## Skewness: Measuring Asymmetry

**Sample skewness** (Fisher's definition with bias correction):

$$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

where $s$ is the sample standard deviation (`ddof=1`). The key insight: cubing preserves sign. Values above the mean contribute positively; values below contribute negatively.

**Interpretation:**

- $g_1 > 0$ (right-skewed): The right tail is longer. Most values cluster left of the mean. Examples: income, house prices, waiting times.
- $g_1 < 0$ (left-skewed): The left tail is longer. Most values cluster right of the mean. Examples: age at retirement, exam scores with a ceiling.
- $g_1 \approx 0$ (symmetric): Tails are balanced. Example: heights, measurement errors.

**Rules of thumb:** $|g_1| < 0.5$ is approximately symmetric; $0.5 < |g_1| < 1$ is moderately skewed; $|g_1| > 1$ is highly skewed.

---

## Excess Kurtosis: Measuring Tail Weight

**Sample excess kurtosis** (with bias correction):

$$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

The "excess" means we subtract 3 so that the normal distribution has kurtosis 0 (raw kurtosis of the normal is 3).

Kurtosis is often misunderstood as "peakedness." It is primarily about **tail weight** - how much probability mass is in the tails relative to a normal distribution.

**Three regimes:**

- **Leptokurtic** ($g_2 > 0$): Heavier tails than normal. More extreme outliers are expected. Examples: financial returns, Student's $t$-distribution.
- **Mesokurtic** ($g_2 \approx 0$): Similar tails to normal. Example: the normal distribution itself.
- **Platykurtic** ($g_2 < 0$): Lighter tails than normal. Fewer extreme values. Example: uniform distribution ($g_2 = -1.2$).

---

## Why the Fourth Power?

The standardized fourth moment $\left(\frac{x_i - \bar{x}}{s}\right)^4$ amplifies extreme deviations dramatically. A value 3 standard deviations from the mean contributes $3^4 = 81$ times more than a value 1 standard deviation away. This makes kurtosis extremely sensitive to outliers and tail behavior.

---

## Connection to the Normal Distribution

The normal distribution serves as the reference:
- Skewness = 0 (perfectly symmetric)
- Excess kurtosis = 0 (by definition of "excess")

Many statistical tests assume normality. Checking skewness and kurtosis is a quick diagnostic:
- **Jarque-Bera test:** $JB = \frac{n}{6}\left(g_1^2 + \frac{g_2^2}{4}\right)$ tests whether skewness and kurtosis jointly match a normal distribution.

---

## Applications in Machine Learning

**Assumption Checking:** Linear regression, LDA, and Gaussian Naive Bayes assume normally distributed features. High skewness or kurtosis signals the need for transformations (log, Box-Cox, Yeo-Johnson).

**Feature Selection:** Features with extreme kurtosis may dominate distance-based models (KNN, SVM) due to occasional extreme values. Understanding tail behavior helps decide whether to clip, transform, or remove such features.

**Financial ML and Risk:** Excess kurtosis directly measures "tail risk" - the probability of extreme market moves. Models that ignore heavy tails (assuming normality) dramatically underestimate crash probabilities.

**Data Transformation:** Right-skewed features benefit from log or square-root transforms to reduce skewness. The goal is often to make features more Gaussian-like for models that assume normality.

**Anomaly and Novelty Detection:** Points in the extreme tails (as quantified by high kurtosis) are natural anomaly candidates. Leptokurtic features may generate more false positives in z-score-based anomaly detection because extreme values are more probable than the normal distribution predicts.

---

## Relationship Between Moments

The four standardized moments form a complete shape description:

| Moment | Measures | Symbol |
|--------|----------|--------|
| 1st (mean) | Location/center | $\mu$ |
| 2nd (variance) | Spread/dispersion | $\sigma^2$ |
| 3rd (skewness) | Asymmetry | $g_1$ |
| 4th (kurtosis) | Tail weight | $g_2$ |

Higher moments exist but are rarely used in practice because they require enormous sample sizes to estimate reliably. Even kurtosis requires large samples (typically $n > 100$) for stable estimates.