# <span style="font-size: 20px;">Decision Tree Classifier (CART)</span>

<span style="font-size: 14px;">CART (Classification and Regression Trees), introduced by Breiman et al. in 1984, is one of the most fundamental algorithms in machine learning. It builds a binary tree by recursively splitting the data along axis-aligned boundaries, selecting at each node the feature and threshold that best separate the classes according to an impurity criterion.</span>

---

## <span style="font-size: 16px;">Gini Impurity</span>

<span style="font-size: 14px;">Gini impurity measures how often a randomly chosen sample from a set would be misclassified if it were labeled according to the distribution of labels in that set:</span>

$$
G(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

<span style="font-size: 14px;">where</span> $p_k$ <span style="font-size: 14px;">is the fraction of samples in</span> $S$ <span style="font-size: 14px;">belonging to class</span> $k$<span style="font-size: 14px;">.</span>

- <span style="font-size: 14px;">When all samples belong to one class:</span> $G = 0$ <span style="font-size: 14px;">(pure node)</span>
- <span style="font-size: 14px;">For binary classification with equal proportions:</span> $G = 0.5$ <span style="font-size: 14px;">(maximum impurity)</span>
- <span style="font-size: 14px;">For</span> $K$ <span style="font-size: 14px;">equally represented classes:</span> $G = 1 - 1/K$

---

## <span style="font-size: 16px;">Splitting Criterion</span>

<span style="font-size: 14px;">At each node, CART considers every possible split of the form</span> $x_j \leq t$ <span style="font-size: 14px;">for each feature</span> $j$ <span style="font-size: 14px;">and each unique value</span> $t$ <span style="font-size: 14px;">of that feature in the current data. The split that maximizes the information gain is selected:</span>

$$
\text{Gain}(S, j, t) = G(S) - \frac{|S_L|}{|S|} G(S_L) - \frac{|S_R|}{|S|} G(S_R)
$$

<span style="font-size: 14px;">where</span> $S_L = \{x \in S : x_j \leq t\}$ <span style="font-size: 14px;">and</span> $S_R = \{x \in S : x_j > t\}$<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">The algorithm exhaustively searches over all features and thresholds to find the split with the largest gain. This greedy approach does not guarantee a globally optimal tree, but it is computationally tractable.</span>

---

## <span style="font-size: 16px;">Gini vs. Entropy</span>

<span style="font-size: 14px;">An alternative impurity measure is the entropy (information gain):</span>

$$
H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

- <span style="font-size: 14px;">Gini and entropy produce very similar trees in practice</span>
- <span style="font-size: 14px;">Gini is slightly faster to compute (no logarithms)</span>
- <span style="font-size: 14px;">Entropy tends to produce slightly more balanced trees</span>
- <span style="font-size: 14px;">CART uses Gini by default; ID3 and C4.5 use entropy</span>

---

## <span style="font-size: 16px;">Tree Construction</span>

<span style="font-size: 14px;">The recursive algorithm:</span>

- <span style="font-size: 14px;">**Base cases** (create a leaf): the node is pure (all labels identical), depth reaches the maximum, fewer than the minimum number of samples remain, or no split produces positive gain</span>
- <span style="font-size: 14px;">**Recursive case**: find the best split, partition the data into left and right subsets, and recurse on each</span>
- <span style="font-size: 14px;">**Leaf prediction**: the majority class among the samples that reach the leaf</span>

<span style="font-size: 14px;">Each internal node stores: the feature index, the threshold value, and pointers to the left and right children. Each leaf stores the predicted class label.</span>

---

## <span style="font-size: 16px;">Stopping Criteria and Pruning</span>

<span style="font-size: 14px;">Without constraints, a decision tree will grow until every leaf is pure, perfectly memorizing the training data. This leads to overfitting. Common controls:</span>

- <span style="font-size: 14px;">**Max depth**: limits how deep the tree can grow</span>
- <span style="font-size: 14px;">**Min samples split**: minimum number of samples required to attempt a split</span>
- <span style="font-size: 14px;">**Min samples leaf**: minimum number of samples in each leaf</span>
- <span style="font-size: 14px;">**Min impurity decrease**: require a minimum gain to split</span>

<span style="font-size: 14px;">**Pre-pruning** applies these constraints during construction. **Post-pruning** (cost-complexity pruning) grows the full tree first, then iteratively removes subtrees that do not improve cross-validated performance. CART uses cost-complexity pruning with the parameter</span> $\alpha$<span style="font-size: 14px;">:</span>

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

<span style="font-size: 14px;">where</span> $R(T)$ <span style="font-size: 14px;">is the misclassification rate and</span> $|T|$ <span style="font-size: 14px;">is the number of leaves.</span>

---

## <span style="font-size: 16px;">Prediction</span>

<span style="font-size: 14px;">To classify a new point, start at the root and at each internal node, go left if</span> $x_j \leq t$ <span style="font-size: 14px;">or right otherwise. Continue until reaching a leaf, which provides the predicted class. Prediction is</span> $O(\text{depth})$ <span style="font-size: 14px;">per point.</span>

---

## <span style="font-size: 16px;">Computational Complexity</span>

- <span style="font-size: 14px;">**Training**: at each node, evaluating all splits takes</span> $O(n \cdot d)$ <span style="font-size: 14px;">where</span> $n$ <span style="font-size: 14px;">is the number of samples and</span> $d$ <span style="font-size: 14px;">is the number of features. With</span> $O(n)$ <span style="font-size: 14px;">nodes in the worst case, total training is</span> $O(n^2 \cdot d)$
- <span style="font-size: 14px;">**Prediction**:</span> $O(\text{depth})$ <span style="font-size: 14px;">per sample, which is at most</span> $O(\log n)$ <span style="font-size: 14px;">for a balanced tree</span>

---

## <span style="font-size: 16px;">Advantages and Limitations</span>

<span style="font-size: 14px;">**Advantages:**</span>

- <span style="font-size: 14px;">Interpretable: the tree structure can be visualized and understood</span>
- <span style="font-size: 14px;">No feature scaling needed: splits are based on thresholds, not distances</span>
- <span style="font-size: 14px;">Handles mixed feature types and missing values naturally</span>
- <span style="font-size: 14px;">Non-parametric: makes no distributional assumptions</span>
- <span style="font-size: 14px;">Can capture nonlinear relationships and interactions</span>

<span style="font-size: 14px;">**Limitations:**</span>

- <span style="font-size: 14px;">High variance: small changes in data can produce very different trees</span>
- <span style="font-size: 14px;">Greedy construction: each split is locally optimal, not globally</span>
- <span style="font-size: 14px;">Axis-aligned splits: cannot efficiently represent diagonal decision boundaries</span>
- <span style="font-size: 14px;">Prone to overfitting without proper regularization</span>
- <span style="font-size: 14px;">Biased toward features with many unique values</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How do you handle continuous vs. categorical features?**</span>
  <span style="font-size: 14px;">A: For continuous features, try all unique values as thresholds. For categorical features with $m$ levels, try all $2^{m-1} - 1$ binary partitions (CART), or use one-hot encoding.</span>

- <span style="font-size: 14px;">**Q: Why are decision trees the building block of ensemble methods?**</span>
  <span style="font-size: 14px;">A: Their high variance makes them ideal for variance-reduction techniques like bagging and random forests. Their ability to fit complex boundaries makes them powerful weak learners for boosting.</span>

- <span style="font-size: 14px;">**Q: How does CART differ from ID3/C4.5?**</span>
  <span style="font-size: 14px;">A: CART always produces binary splits and uses Gini impurity. ID3 uses entropy and produces multi-way splits. C4.5 extends ID3 with gain ratio to reduce bias toward multi-valued features.</span>

- <span style="font-size: 14px;">**Q: What is the bias-variance tradeoff for trees?**</span>
  <span style="font-size: 14px;">A: Deep trees have low bias but high variance (overfit). Shallow trees have high bias but low variance (underfit). Ensembles like random forests reduce variance while keeping low bias.</span>

---