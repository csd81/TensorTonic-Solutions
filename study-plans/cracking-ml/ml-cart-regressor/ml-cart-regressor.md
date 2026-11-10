# <span style="font-size: 20px;">Decision Tree Regressor</span>

CART (Classification and Regression Trees) can be adapted for regression by replacing Gini impurity with mean squared error (MSE) and predicting the mean target value at each leaf instead of the majority class.

---

## <span style="font-size: 16px;">MSE as Impurity</span>

For regression, node impurity is measured by the variance of target values:

$$
\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

where $\bar{y}$ is the mean of targets in set $S$. A node with identical targets has $\text{MSE} = 0$ (pure). Higher variance means the node's prediction (the mean) is a poor fit.

---

## <span style="font-size: 16px;">Variance Reduction</span>

The splitting criterion maximizes the weighted reduction in MSE:

$$
\Delta(S, j, t) = \text{MSE}(S) - \frac{|S_L|}{|S|} \text{MSE}(S_L) - \frac{|S_R|}{|S|} \text{MSE}(S_R)
$$

where $S_L = \{x \in S : x_j \leq t\}$ and $S_R = \{x \in S : x_j > t\}$. This is equivalent to minimizing the total weighted MSE of the two child nodes. The algorithm greedily selects the split with the largest variance reduction at each step.

---

## <span style="font-size: 16px;">Tree Construction</span>

The recursive algorithm mirrors the classification version:

- **Base cases** (create a leaf): depth reaches the maximum, fewer than the minimum samples remain, all target values are identical, or no split produces positive variance reduction
- **Recursive case**: find the best split by exhaustive search over all features and unique thresholds, partition the data, and recurse
- **Leaf prediction**: the mean of the target values that reach the leaf

Each internal node stores: the feature index, the threshold, and pointers to left and right children. Each leaf stores the predicted value (mean of targets).

---

## <span style="font-size: 16px;">Classification vs. Regression CART</span>

| Aspect | Classification | Regression |
|---|---|---|
| Impurity measure | Gini impurity $1 - \sum p_k^2$ | MSE $\frac{1}{n}\sum(y_i - \bar{y})^2$ |
| Leaf prediction | Majority class | Mean of targets |
| Output | Discrete class labels | Continuous values |

The tree structure, splitting procedure, and stopping criteria are identical. Only the impurity function and leaf prediction change.

---

## <span style="font-size: 16px;">Piecewise Constant Approximation</span>

A regression tree produces a piecewise constant function: the feature space is partitioned into rectangular regions (one per leaf), and the prediction within each region is constant (the mean of training targets in that region). With sufficient depth, the tree can approximate any continuous function to arbitrary precision, but deeper trees risk overfitting.

---

## <span style="font-size: 16px;">Bias-Variance Tradeoff</span>

- **Deep trees**: low bias (can fit complex patterns) but high variance (sensitive to training data)
- **Shallow trees**: high bias (crude approximation) but low variance (stable predictions)
- Controlling `max_depth` and `min_samples` trades off between underfitting and overfitting

Regression trees are the foundation of ensemble methods like gradient boosted trees, where many shallow trees (stumps or depth-limited trees) are combined to achieve both low bias and low variance.

---

## <span style="font-size: 16px;">Computational Complexity</span>

- **Training**: $O(n^2 \cdot d)$ worst case, where $n$ is the number of samples and $d$ is the number of features
- **Prediction**: $O(\text{depth})$ per sample

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- **Q: Why MSE instead of MAE?**
  A: MSE is differentiable and leads to a closed-form optimal prediction (the mean). MAE would require the median. MSE also penalizes large errors more heavily.

- **Q: How is this used in gradient boosting?**
  A: Each boosting iteration fits a regression tree to the negative gradient (residuals for squared loss). The tree partitions the feature space, and the leaf values are the mean residual in each region.

- **Q: Can regression trees handle multivariate outputs?**
  A: Yes, by computing MSE across all output dimensions. This is used in multi-output regression.

---