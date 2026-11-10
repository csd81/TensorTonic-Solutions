## <span style="font-size: 20px;">Pearson Correlation</span>

The Pearson correlation coefficient measures the **strength and direction of the linear relationship** between two variables. It is one of the most widely used statistics in data analysis and machine learning.

---

## Formula

For two variables $X$ and $Y$ with $n$ paired observations:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Equivalently, using covariance and standard deviations:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

where $\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$.

---

## Range and Interpretation

The correlation coefficient is bounded: $-1 \leq r \leq 1$.

| Value | Interpretation |
|-------|---------------|
| $r = 1$ | Perfect positive linear relationship |
| $0.7 < r < 1$ | Strong positive correlation |
| $0.3 < r < 0.7$ | Moderate positive correlation |
| $-0.3 < r < 0.3$ | Weak or no linear correlation |
| $-0.7 < r < -0.3$ | Moderate negative correlation |
| $-1 < r < -0.7$ | Strong negative correlation |
| $r = -1$ | Perfect negative linear relationship |

**Critical caveat:** $r \approx 0$ does not mean "no relationship" - it means no *linear* relationship. Two variables can have a perfect nonlinear relationship (e.g., $Y = X^2$) with $r = 0$.

---

## Correlation Matrix

For a dataset with $p$ features, the **correlation matrix** $R$ is a $p \times p$ symmetric matrix where entry $R_{ij} = r(X_i, X_j)$.

**Properties of the correlation matrix:**
- Symmetric: $R_{ij} = R_{ji}$
- Diagonal entries are 1: $R_{ii} = 1$ (each variable is perfectly correlated with itself)
- Positive semi-definite: all eigenvalues $\geq 0$
- Related to covariance matrix: $R = D^{-1} C D^{-1}$ where $C$ is the covariance matrix and $D$ is the diagonal matrix of standard deviations

**Computing efficiently:** Center the data ($X_c = X - \bar{X}$), compute covariance ($C = X_c^T X_c / (n-1)$), then normalize by standard deviations: $R_{ij} = C_{ij} / (s_i s_j)$.

---

## Correlation is Not Causation

This principle is fundamental. Correlation can arise from:
1. **Direct causation:** $X$ causes $Y$
2. **Reverse causation:** $Y$ causes $X$
3. **Confounding:** A third variable $Z$ causes both $X$ and $Y$
4. **Coincidence:** Spurious correlation in finite samples

Example: ice cream sales and drowning deaths are positively correlated. The confounder is temperature - hot weather increases both.

---

## Partial Correlation

Partial correlation measures the relationship between $X$ and $Y$ after removing the effect of confounding variables. For variables $X$, $Y$ conditioned on $Z$:

$$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}$$

This reveals whether an observed correlation is genuine or driven by a shared confounder.

---

## Applications in Machine Learning

**Feature Selection:** Highly correlated features ($|r| > 0.9$) provide redundant information. Dropping one of each correlated pair reduces dimensionality without losing predictive power.

**Multicollinearity Detection:** In linear regression, highly correlated predictors inflate standard errors and make coefficients unstable. The Variance Inflation Factor (VIF) is derived from correlations among predictors.

**Correlation-Based Clustering:** Grouping features by correlation structure reveals latent factors. This is related to PCA: the principal components are eigenvectors of the correlation (or covariance) matrix.

**Data Leakage Detection:** A feature with suspiciously high correlation with the target ($r > 0.95$) may indicate data leakage - information from the future or the target itself contaminating the feature.

**Dimensionality Reduction (PCA):** PCA finds directions of maximum variance by computing eigenvectors of the covariance (or correlation) matrix. Using the correlation matrix instead of covariance is equivalent to standardizing features first, which prevents high-variance features from dominating.

---

## Limitations and Alternatives

- **Pearson only captures linear relationships.** For monotonic but nonlinear relationships, use **Spearman's rank correlation**: $r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ where $d_i$ is the rank difference.
- **Sensitive to outliers:** A single extreme point can dramatically alter Pearson's $r$. Robust alternatives include Spearman's $r_s$ or Kendall's $\tau$.
- **Assumes continuous data:** For ordinal or categorical variables, use Spearman's rank correlation or Cramer's V respectively.
- **$r^2$ (coefficient of determination):** The squared correlation tells you the proportion of variance in one variable explained by a linear relationship with the other. For example, $r = 0.7$ means $r^2 = 0.49$, so only 49% of variance is explained.