# <span style="font-size: 20px;">Ridge Regression</span>

<span style="font-size: 14px;">Ridge regression is a regularized version of ordinary least squares (OLS) that adds an L2 penalty to the loss function. This penalty shrinks all coefficients toward zero, producing a more stable and generalizable model.</span>

---

## <span style="font-size: 16px;">Motivation</span>

- <span style="font-size: 14px;">OLS minimizes</span> $\|y - Xw\|^2$ <span style="font-size: 14px;">and can produce arbitrarily large coefficients when features are correlated or when</span> $d > n$
- <span style="font-size: 14px;">Large coefficients lead to high variance in predictions on new data</span>
- <span style="font-size: 14px;">Ridge adds a penalty proportional to the squared magnitude of the coefficients, trading a small increase in bias for a large decrease in variance</span>

---

## <span style="font-size: 16px;">Objective Function</span>

<span style="font-size: 14px;">The Ridge objective minimizes:</span>

$$
J(w) = \|y - Xw\|^2 + \alpha \|w_{1:}\|_2^2
$$

- $y \in \mathbb{R}^n$ <span style="font-size: 14px;">is the target vector</span>
- $X \in \mathbb{R}^{n \times (d+1)}$ <span style="font-size: 14px;">is the augmented feature matrix (with intercept column)</span>
- $w \in \mathbb{R}^{d+1}$ <span style="font-size: 14px;">is the weight vector including the intercept</span>
- $w_{1:}$ <span style="font-size: 14px;">denotes all weights except the intercept</span>
- $\alpha > 0$ <span style="font-size: 14px;">controls the regularization strength</span>

---

## <span style="font-size: 16px;">Closed-Form Solution</span>

<span style="font-size: 14px;">Setting the gradient to zero yields:</span>

$$
\nabla_w J = -2X^T(y - Xw) + 2\alpha I' w = 0
$$

<span style="font-size: 14px;">Solving for</span> $w$<span style="font-size: 14px;">:</span>

$$
w = (X^T X + \alpha I')^{-1} X^T y
$$

- $I'$ <span style="font-size: 14px;">is the identity matrix with 0 in the top-left entry (so the intercept is not regularized)</span>
- <span style="font-size: 14px;">The term</span> $\alpha I'$ <span style="font-size: 14px;">makes</span> $X^T X + \alpha I'$ <span style="font-size: 14px;">positive definite, guaranteeing invertibility</span>

---

## <span style="font-size: 16px;">Feature Standardization</span>

- <span style="font-size: 14px;">Features must be standardized before applying Ridge so the penalty treats all features equally</span>
- <span style="font-size: 14px;">Each feature is transformed as:</span> $x_j' = \frac{x_j - \mu_j}{\sigma_j}$
- <span style="font-size: 14px;">Mean</span> $\mu_j$ <span style="font-size: 14px;">and standard deviation</span> $\sigma_j$ <span style="font-size: 14px;">are computed from training data only</span>
- <span style="font-size: 14px;">The same transformation is applied to test data</span>
- <span style="font-size: 14px;">If a feature has zero variance, its standard deviation is set to 1</span>

---

## <span style="font-size: 16px;">Intercept Handling</span>

- <span style="font-size: 14px;">A column of ones is prepended to the standardized feature matrix</span>
- <span style="font-size: 14px;">The intercept is excluded from the penalty because it captures the mean of</span> $y$
- <span style="font-size: 14px;">Under strong regularization, all slope coefficients shrink to zero and predictions collapse to</span> $\bar{y}$<span style="font-size: 14px;">, which is the correct default behavior</span>

---

## <span style="font-size: 16px;">Effect of Alpha</span>

- <span style="font-size: 14px;">When</span> $\alpha \to 0$<span style="font-size: 14px;">, Ridge converges to the OLS solution</span>
- <span style="font-size: 14px;">When</span> $\alpha \to \infty$<span style="font-size: 14px;">, all slope coefficients shrink to zero</span>
- <span style="font-size: 14px;">Ridge never sets any coefficient exactly to zero (unlike Lasso)</span>
- <span style="font-size: 14px;">The optimal</span> $\alpha$ <span style="font-size: 14px;">is typically chosen via cross-validation</span>

---

## <span style="font-size: 16px;">Comparison with OLS</span>

- <span style="font-size: 14px;">OLS is unbiased but can have high variance when features are correlated</span>
- <span style="font-size: 14px;">Ridge introduces bias but substantially reduces variance</span>
- <span style="font-size: 14px;">The bias-variance tradeoff often results in lower test error for Ridge</span>
- <span style="font-size: 14px;">Ridge is especially useful when the number of features is large relative to the number of samples</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Ridge vs Lasso?**</span>
  <span style="font-size: 14px;">A: Ridge shrinks coefficients toward zero but never to exactly zero. Lasso (L1 penalty) can produce exactly zero coefficients, performing feature selection. Ridge is better when all features contribute; Lasso when only a few matter</span>

- <span style="font-size: 14px;">**Q: What happens at extreme alpha values?**</span>
  <span style="font-size: 14px;">A: At $\alpha = 0$ Ridge equals OLS. As $\alpha \to \infty$, all slope coefficients shrink to zero and predictions collapse to $\bar{y}$</span>

- <span style="font-size: 14px;">**Q: Why standardize features?**</span>
  <span style="font-size: 14px;">A: Without standardization, the L2 penalty penalizes larger-scale features less. A feature measured in millimeters would have its coefficient penalized 1000x more than the same feature measured in meters</span>

- <span style="font-size: 14px;">**Q: When does Ridge outperform OLS?**</span>
  <span style="font-size: 14px;">A: When features are correlated (multicollinearity), when $d > n$, or when the OLS estimate has high variance. Ridge trades bias for variance reduction</span>

- <span style="font-size: 14px;">**Q: How to choose alpha?**</span>
  <span style="font-size: 14px;">A: Use k-fold cross-validation. Typical practice is to search over a log-scale grid (e.g., $10^{-4}$ to $10^4$)</span>

- <span style="font-size: 14px;">**Q: Bayesian interpretation?**</span>
  <span style="font-size: 14px;">A: Ridge regression is equivalent to MAP estimation with a Gaussian prior on the weights: $w \sim \mathcal{N}(0, \frac{1}{\alpha} I)$</span>

---