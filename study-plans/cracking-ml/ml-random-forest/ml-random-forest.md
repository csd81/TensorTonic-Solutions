# <span style="font-size: 20px;">Random Forest from Scratch</span>

Random Forest, introduced by Leo Breiman in 2001, extends bagging by adding random feature subsampling at each split. This additional randomization further decorrelates the trees, reducing ensemble variance beyond what bagging alone achieves.

---

## <span style="font-size: 16px;">From Bagging to Random Forest</span>

Recall the variance of an ensemble of $B$ estimators with pairwise correlation $\rho$:

$$
\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} h_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

Bagging reduces the second term by increasing $B$, but cannot eliminate $\rho\sigma^2$. Trees trained on bootstrap samples of the same data remain correlated because they tend to split on the same dominant features. Random Forest addresses this by restricting each split to a random subset of $m$ features, forcing trees to use different features and reducing $\rho$.

---

## <span style="font-size: 16px;">Feature Subsampling</span>

At each node (not just each tree), a random subset of $m$ features is selected, and the best split is found among only those features. Common choices:

- **Classification**: $m = \lfloor\sqrt{d}\rfloor$ (Breiman's recommendation)
- **Regression**: $m = \lfloor d/3 \rfloor$
- **Alternative**: $m = \lfloor\log_2 d\rfloor$

Smaller $m$ gives more randomization (lower correlation, higher individual tree variance). Larger $m$ gives less randomization (higher correlation, lower individual tree variance). When $m = d$, Random Forest reduces to standard bagging.

---

## <span style="font-size: 16px;">Algorithm</span>

For each tree $b = 1, \ldots, B$:

1. Draw a bootstrap sample of size $n$ with replacement
2. Build a CART tree on the bootstrap sample, but at each split: randomly select $m$ features, find the best Gini split among those $m$ features, split and recurse
3. Grow the tree to maximum depth (or until stopping criteria)

For prediction, aggregate all tree votes via majority vote (classification) or averaging (regression).

---

## <span style="font-size: 16px;">Key Properties</span>

- **Low bias**: individual trees are grown deep (low bias, high variance)
- **Low variance**: averaging decorrelated trees reduces variance
- **No overfitting with more trees**: unlike boosting, adding more trees to a random forest does not increase overfitting (error plateaus)
- **Feature importance**: the average Gini reduction across all trees provides a natural measure of feature importance
- **OOB error**: same as bagging, provides a free estimate of generalization error

---

## <span style="font-size: 16px;">Random Forest vs. Bagging vs. Boosting</span>

| Aspect | Bagging | Random Forest | Boosting |
|---|---|---|---|
| Feature selection | All features | Random subset per split | All features |
| Tree correlation | Moderate | Low | N/A (sequential) |
| Primary benefit | Reduce variance | Further reduce variance | Reduce bias |
| Risk of overfitting | Low | Low | Moderate |

---

## <span style="font-size: 16px;">Computational Complexity</span>

- **Training**: $O(B \cdot n \cdot m \cdot n)$ per level, where $m \ll d$ reduces the per-split cost compared to bagging
- **Prediction**: $O(B \cdot \text{depth})$ per test point
- Embarrassingly parallel: each tree can be built independently

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- **Q: Why is feature subsampling important?**
  A: Without it, all trees tend to split on the same strongest feature first, making them correlated. Subsampling forces diversity.

- **Q: How do you choose $m$?**
  A: Cross-validate. $\sqrt{d}$ is a good default for classification, $d/3$ for regression. Smaller $m$ is better when there are many irrelevant features.

- **Q: Can you add more trees to an existing random forest?**
  A: Yes, since each tree is independent. This is an advantage over boosting, where models are sequential.

---