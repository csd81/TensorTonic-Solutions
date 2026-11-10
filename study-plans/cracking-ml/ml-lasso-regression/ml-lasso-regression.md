# <span style="font-size: 20px;">Lasso Regression</span>

<span style="font-size: 14px;">Lasso (Least Absolute Shrinkage and Selection Operator) regression adds an L1 penalty to the ordinary least squares objective. Its distinctive property is that it can set coefficients exactly to zero, performing automatic feature selection.</span>

---

## <span style="font-size: 16px;">Motivation</span>

- <span style="font-size: 14px;">When many features are available, some may be irrelevant or redundant</span>
- <span style="font-size: 14px;">Ridge regression shrinks all coefficients but never eliminates any</span>
- <span style="font-size: 14px;">Lasso produces sparse models by zeroing out unimportant features, making the model more interpretable</span>

---

## <span style="font-size: 16px;">Objective Function</span>

The objective used in this problem is

$$
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(x_i^{\mathsf T}w+b-y_i)^2+\alpha\sum_{j=1}^{d}|w_j|.
$$

Here, n is the number of samples, d is the feature count, x with subscript i is feature row i, y with subscript i is its target, w is the weight vector, b is the separate unpenalized bias, and alpha is the L1 penalty strength. The indices i and j select samples and weights.

### The optimizer used here

The task uses finite batch subgradient updates from zero weights and bias. At a zero weight, it chooses a subgradient of zero for the absolute-value term. The supplied learning rate and epoch count therefore determine the returned parameters; they need not be the exact minimizer or contain exact zeros.

### Coordinate descent as an alternative

Coordinate descent optimizes one weight at a time and is a common Lasso solver, but it is not the update method assessed here. Holding the bias and other weights fixed, define the partial residual and column statistics as

$$
r_j=y-b\mathbf{1}-Xw+X_jw_j.
$$

$$
\rho_j=\frac{X_j^{\mathsf T}r_j}{n}.
$$

$$
z_j=\frac{X_j^{\mathsf T}X_j}{n}.
$$

For a nonzero feature column, the coordinate minimizer is

$$
w_j=\frac{\operatorname{sign}(\rho_j)\max(|\rho_j|-\alpha/2,0)}{z_j}.
$$

Here, X is the feature matrix, X with subscript j is its jth column, the bold one is an all-ones vector, r with subscript j excludes that column's contribution, rho is the mean residual correlation, and z is the mean squared column value. Soft thresholding creates exact zeros. The threshold is alpha divided by two because this objective uses mean squared error without a one-half factor.

---

## <span style="font-size: 16px;">Feature Standardization</span>

- <span style="font-size: 14px;">In applications, feature standardization makes the penalty comparable across differently scaled features. This task uses the supplied features directly; do not add standardization</span>
- <span style="font-size: 14px;">Each feature is transformed as:</span> $x_j' = \frac{x_j - \mu_j}{\sigma_j}$
- <span style="font-size: 14px;">Mean and standard deviation are computed from training data only</span>
- <span style="font-size: 14px;">The same transformation is applied to test data</span>

---

## <span style="font-size: 16px;">Effect of Alpha</span>

These statements describe the minimized Lasso objective. A fixed number of subgradient updates need not reach that optimum or produce exact zeros.

- <span style="font-size: 14px;">When</span> $\alpha \to 0$<span style="font-size: 14px;">, Lasso converges to the OLS solution</span>
- <span style="font-size: 14px;">As</span> $\alpha$ <span style="font-size: 14px;">increases, weaker features are zeroed out first</span>
- <span style="font-size: 14px;">When</span> $\alpha$ <span style="font-size: 14px;">is very large, all features are eliminated and predictions equal</span> $\bar{y}$

---

## <span style="font-size: 16px;">Lasso vs Ridge</span>

- <span style="font-size: 14px;">Ridge (L2): shrinks all coefficients but never zeros them out. Better when all features contribute.</span>
- <span style="font-size: 14px;">Lasso (L1): zeros out unimportant features. Better when only a few features truly matter.</span>
- <span style="font-size: 14px;">The geometric intuition: L1 constraint has corners at the axes where coefficients are exactly zero. L2 constraint is a smooth sphere with no corners.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Lasso vs Ridge?**</span>
  <span style="font-size: 14px;">A: Lasso (L1) drives coefficients to exactly zero, performing feature selection. Ridge (L2) shrinks coefficients but never to zero. Use Lasso when you believe only a few features matter; Ridge when all features contribute</span>

- <span style="font-size: 14px;">**Q: Why coordinate descent?**</span>
  <span style="font-size: 14px;">A: The L1 penalty has a kink at zero. Coordinate descent handles it with soft thresholding; the assessed algorithm instead chooses a specified subgradient and performs batch updates</span>

- <span style="font-size: 14px;">**Q: What is soft thresholding?**</span>
  <span style="font-size: 14px;">A: The operator $S(\rho, \alpha) = \text{sign}(\rho) \cdot \max(|\rho| - \alpha, 0)$ shrinks the correlation toward zero and clips it to exactly zero when its magnitude is below $\alpha$</span>

- <span style="font-size: 14px;">**Q: When does Lasso fail?**</span>
  <span style="font-size: 14px;">A: When features are highly correlated (grouped), Lasso tends to select one and zero out the rest arbitrarily. Elastic Net (L1+L2) handles this by encouraging correlated features to share similar coefficients</span>

- <span style="font-size: 14px;">**Q: How to choose alpha?**</span>
  <span style="font-size: 14px;">A: Use cross-validation over a log-scale grid. The regularization path (coefficients as a function of alpha) shows which features are selected at each penalty level</span>

- <span style="font-size: 14px;">**Q: Proximal gradient alternative?**</span>
  <span style="font-size: 14px;">A: Instead of coordinate descent, Lasso can be solved with proximal gradient descent (ISTA/FISTA), which applies soft thresholding to the entire gradient step at once</span>

---