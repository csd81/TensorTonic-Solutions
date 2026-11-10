# <span style="font-size: 20px;">Gradient Boosted Regressor</span>

Gradient Boosting, formalized by Friedman in 2001, is a powerful ensemble technique that builds models sequentially, with each new model fitting the negative gradient of the loss function with respect to the current predictions. For squared error loss, the negative gradient is simply the residual.

---

## <span style="font-size: 16px;">Core Idea</span>

Unlike bagging (which reduces variance by averaging independent models), boosting reduces bias by iteratively correcting errors. At each step, we fit a new model to the "mistakes" of the current ensemble:

$$
F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)
$$

where $h_t$ is a regression tree and $\eta$ is the learning rate (shrinkage). The key insight is that $h_t$ is fit to the negative gradient of the loss, not the original targets.

---

## <span style="font-size: 16px;">Squared Error Loss</span>

For squared error loss $L(y, F) = \frac{1}{2}(y - F)^2$:

$$
-\frac{\partial L}{\partial F} = y - F = r \quad \text{(the residual)}
$$

So for regression with squared loss, gradient boosting simply fits trees to residuals. This is the most intuitive case and the starting point for understanding gradient boosting.

---

## <span style="font-size: 16px;">Algorithm</span>

1. Initialize: $F_0(x) = \bar{y}$ (mean of targets, which minimizes total squared error)
2. For $t = 1, \ldots, T$:
   - Compute residuals: $r_i = y_i - F_{t-1}(x_i)$
   - Fit a regression tree $h_t$ to $(X, r)$ using MSE splits
   - Update: $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$
3. Final prediction: $F_T(x) = F_0 + \eta \sum_{t=1}^{T} h_t(x)$

---

## <span style="font-size: 16px;">Learning Rate (Shrinkage)</span>

The learning rate $\eta \in (0, 1]$ controls how much each tree contributes:

- **Small $\eta$** (e.g., 0.01-0.1): each tree makes a small correction. Requires more trees but generalizes better. This is called shrinkage.
- **Large $\eta$** (e.g., 0.5-1.0): each tree makes a large correction. Fewer trees needed but higher risk of overfitting.

Empirically, smaller learning rates with more trees almost always outperform larger rates with fewer trees, at the cost of computation time.

---

## <span style="font-size: 16px;">Tree Depth</span>

Unlike random forests, gradient boosting uses shallow trees (typically depth 3-6):

- **Depth 1** (stumps): captures only main effects, no interactions
- **Depth 2**: captures pairwise interactions
- **Depth $J$**: captures up to $J$-way interactions

Shallow trees act as weak learners (high bias, low variance). The sequential correction process gradually reduces bias while the learning rate controls variance.

---

## <span style="font-size: 16px;">General Gradient Boosting Framework</span>

The algorithm generalizes beyond squared error. For any differentiable loss $L(y, F)$:

$$
r_i^{(t)} = -\left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F = F_{t-1}(x_i)}
$$

| Loss | Negative Gradient | Use Case |
|---|---|---|
| Squared $\frac{1}{2}(y-F)^2$ | $y - F$ | Regression |
| Absolute $|y-F|$ | $\text{sign}(y-F)$ | Robust regression |
| Log loss | $y - \sigma(F)$ | Classification |

---

## <span style="font-size: 16px;">Computational Complexity</span>

- **Training**: $O(T \cdot n^2 \cdot d)$ for $T$ trees with brute-force split search
- **Prediction**: $O(T \cdot \text{depth})$ per test point
- Trees are sequential and cannot be parallelized (unlike bagging/random forests)

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- **Q: Why shallow trees?**
  A: Deep trees already have low bias. Boosting reduces bias, so starting with low-bias learners is redundant and leads to overfitting. Shallow trees leave room for boosting to improve.

- **Q: How does this relate to XGBoost?**
  A: XGBoost adds regularization terms to the objective, uses second-order Taylor expansion of the loss, and employs histogram-based split finding for speed.

- **Q: Can boosting overfit?**
  A: Yes, especially with too many trees or too high a learning rate. Monitor validation error and use early stopping.

---