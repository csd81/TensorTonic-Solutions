# <span style="font-size: 20px;">Bagging Classifier</span>

Bagging (Bootstrap Aggregating), introduced by Leo Breiman in 1996, is an ensemble method that reduces variance by training multiple models on different bootstrap samples and combining their predictions. It is particularly effective with high-variance base learners like decision trees.

---

## <span style="font-size: 16px;">The Variance Problem</span>

A single decision tree is a high-variance estimator: small changes in training data can produce very different trees. If we average $B$ independent identically distributed random variables, each with variance $\sigma^2$:

$$
\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} X_b\right) = \frac{\sigma^2}{B}
$$

Bagging approximates this by creating "different" training sets via bootstrapping and averaging the resulting models.

---

## <span style="font-size: 16px;">Bootstrap Sampling</span>

A bootstrap sample draws $n$ observations from the training set of size $n$ with replacement. Key properties:

- Each sample includes about $1 - (1 - 1/n)^n \approx 63.2\%$ of the original data
- The remaining $\approx 36.8\%$ are called out-of-bag (OOB) samples
- Some observations appear multiple times, others not at all
- This introduces diversity among the trained models

---

## <span style="font-size: 16px;">Bagging Algorithm</span>

For classification with $B$ base estimators:

- **Training**: for each $b = 1, \ldots, B$: draw a bootstrap sample of size $n$, train a decision tree $h_b$ on that sample
- **Prediction**: for a new point $x$, collect the votes $\{h_1(x), \ldots, h_B(x)\}$ and return the majority class:

$$
\hat{y}(x) = \text{mode}\left(\{h_b(x)\}_{b=1}^{B}\right)
$$

For regression, the prediction is the average: $\hat{y}(x) = \frac{1}{B}\sum_{b=1}^{B} h_b(x)$

---

## <span style="font-size: 16px;">Why Bagging Reduces Variance</span>

Consider $B$ estimators with pairwise correlation $\rho$ and individual variance $\sigma^2$. The variance of their average is:

$$
\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} h_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

- As $B \to \infty$, the second term vanishes, but $\rho\sigma^2$ remains
- Bagging reduces the second term but cannot eliminate correlation between trees
- This is why Random Forests further reduce $\rho$ by also subsampling features

---

## <span style="font-size: 16px;">Out-of-Bag (OOB) Error</span>

Since each bootstrap sample leaves out about 36.8% of observations, we can evaluate each tree on its OOB samples. For each training point $x_i$, we collect predictions only from trees that did not use $x_i$ in training:

$$
\hat{y}_{\text{OOB}}(x_i) = \text{mode}\left(\{h_b(x_i) : x_i \notin S_b\}\right)
$$

The OOB error provides an unbiased estimate of the generalization error without needing a separate validation set, similar to leave-one-out cross-validation.

---

## <span style="font-size: 16px;">Bagging vs. Boosting</span>

| Aspect | Bagging | Boosting |
|---|---|---|
| Training | Parallel (independent) | Sequential (dependent) |
| Sampling | Bootstrap (uniform) | Reweighted samples |
| Reduces | Variance | Bias (and variance) |
| Overfitting | Resistant | Can overfit |
| Base learners | Strong (deep trees) | Weak (shallow stumps) |

---

## <span style="font-size: 16px;">Computational Complexity</span>

- **Training**: $O(B \cdot n^2 \cdot d)$ for $B$ trees, each fit on $n$ samples with $d$ features
- **Prediction**: $O(B \cdot \text{depth})$ per test point
- Trees can be trained in parallel since they are independent

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- **Q: Why bootstrap instead of just splitting the data?**
  A: Splitting reduces the training set size for each model. Bootstrapping keeps the full $n$ samples per tree while still creating diversity through resampling.

- **Q: Does bagging reduce bias?**
  A: No. The bias of the ensemble is approximately equal to the bias of a single base learner. Bagging only reduces variance.

- **Q: How many trees are enough?**
  A: The error typically plateaus after 50-200 trees. OOB error can be monitored to determine when additional trees stop helping.

---