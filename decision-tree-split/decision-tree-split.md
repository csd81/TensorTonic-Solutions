## What Is a Decision Tree Split?

A decision tree split divides a node's samples into two or more groups based on a feature value. The goal is to create child nodes that are **purer** than the parent, meaning samples in each child are more homogeneous in their class labels.

At each internal node, the tree asks a question like "Is feature X greater than threshold T?" and routes samples left or right based on the answer.

---

## The Splitting Criterion

To find the best split, we need a measure of node **impurity**. Common measures:

**Gini impurity:**

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2
$$

**Entropy:**

$$
H(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

where $p_i$ is the proportion of samples in class $i$ and $C$ is the number of classes.

The best split maximizes the reduction in impurity.

---

## Information Gain

Information gain measures how much a split reduces impurity:

$$
\text{IG}(S, A) = \text{Impurity}(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} \text{Impurity}(S_v)
$$

where:
- $S$ is the set of samples at the parent node
- $A$ is the feature used for splitting
- $S_v$ is the subset of samples where feature $A$ has value $v$
- $|S_v|/|S|$ is the proportion of samples going to child $v$

We choose the split that maximizes information gain.

---

## Types of Splits

**Binary split (most common):**

For numeric features: "Is X <= threshold?"
- Left child: samples where $X \leq t$
- Right child: samples where $X > t$

For categorical features: "Is X in subset S?"
- Left child: samples where $X \in S$
- Right child: samples where $X \notin S$

**Multi-way split:**

For categorical features with $k$ values, create $k$ children, one for each value.

---

## Finding the Best Split for Numeric Features

**Step 1:** Sort samples by the feature value

**Step 2:** Consider split points between consecutive distinct values

**Step 3:** For each candidate threshold, compute information gain

**Step 4:** Choose the threshold with maximum information gain

**Example:**

Feature values (sorted): [1, 2, 2, 4, 5, 7]

Candidate thresholds: 1.5, 3, 4.5, 6 (midpoints between distinct values)

For each threshold, split the data and compute impurity reduction.

---

## Worked Example: Finding the Best Split

**Data:** 10 samples with feature X and binary label (+ or -)

- Sample 1: X = 2, Label = +
- Sample 2: X = 3, Label = +
- Sample 3: X = 4, Label = -
- Sample 4: X = 5, Label = +
- Sample 5: X = 6, Label = -
- Sample 6: X = 7, Label = -
- Sample 7: X = 8, Label = -
- Sample 8: X = 9, Label = +
- Sample 9: X = 10, Label = -
- Sample 10: X = 11, Label = -

**Parent node:** 4 positive, 6 negative

**Parent Gini:**
$p_+ = 0.4$, $p_- = 0.6$

$\text{Gini} = 1 - (0.4^2 + 0.6^2) = 1 - (0.16 + 0.36) = 0.48$

---

**Consider split at X = 5.5:**

Left child (X <= 5.5): samples 1, 2, 3, 4
- 3 positive, 1 negative
- $\text{Gini}_L = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375$

Right child (X > 5.5): samples 5, 6, 7, 8, 9, 10
- 1 positive, 5 negative
- $\text{Gini}_R = 1 - (0.167^2 + 0.833^2) = 1 - 0.722 = 0.278$

**Weighted average Gini:**
$\text{Gini}_{\text{children}} = \frac{4}{10}(0.375) + \frac{6}{10}(0.278) = 0.15 + 0.167 = 0.317$

**Gini gain:** $0.48 - 0.317 = 0.163$

---

**Consider split at X = 3.5:**

Left child (X <= 3.5): samples 1, 2
- 2 positive, 0 negative (pure!)
- $\text{Gini}_L = 0$

Right child (X > 3.5): samples 3, 4, 5, 6, 7, 8, 9, 10
- 2 positive, 6 negative
- $\text{Gini}_R = 1 - (0.25^2 + 0.75^2) = 0.375$

**Weighted average Gini:**
$\text{Gini}_{\text{children}} = \frac{2}{10}(0) + \frac{8}{10}(0.375) = 0 + 0.3 = 0.3$

**Gini gain:** $0.48 - 0.3 = 0.18$

**Better split!** Splitting at X = 3.5 gives higher Gini gain.

---

## Splitting Categorical Features

**Approach 1: One-vs-rest**

For each category value, create a binary split: "Is X = value?"

**Approach 2: Subset search**

Find the best subset of values for the left child. For $k$ categories, there are $2^{k-1} - 1$ possible subsets to consider.

**Approach 3: Binary encoding**

Convert categorical feature to multiple binary features, then split on each.

---

## Stopping Criteria

The tree stops splitting when:

**Maximum depth reached:** Tree has grown to specified depth limit

**Minimum samples:** Node has fewer than minimum required samples to split

**Pure node:** All samples have the same class (impurity = 0)

**No improvement:** No split improves impurity beyond a threshold

**Minimum samples per leaf:** Split would create a leaf with too few samples

---

## Greedy Splitting

Decision trees use a **greedy** approach:

1. At each node, find the locally optimal split
2. Do not reconsider previous splits
3. Do not look ahead to future splits

This is computationally efficient but may not find the globally optimal tree.

---

## Split Quality Metrics Comparison

**Gini impurity:**
- Ranges from 0 (pure) to 0.5 (binary, equal split)
- Faster to compute (no logarithms)
- Default in CART and scikit-learn

**Entropy:**
- Ranges from 0 (pure) to 1 (binary, equal split)
- Used in ID3 and C4.5
- Slightly more sensitive to class distribution

**Misclassification error:**
- $1 - \max(p_i)$
- Not recommended for tree building (not strictly concave)
- Used for pruning

In practice, Gini and entropy produce similar trees.

---

## Handling Missing Values

**During training:**
- Ignore missing values when computing split statistics
- Or use surrogate splits (backup features that approximate the primary split)

**During prediction:**
- Use surrogate splits
- Or send to the child with more samples
- Or assign probabilistically to both children

---

## Numerical Stability

When computing impurity:
- Avoid log(0) by using the convention $0 \log(0) = 0$
- Use stable summation for large numbers of classes
- Watch for floating-point precision issues with very small probabilities

---

## Computational Complexity

**For each node:**
- For each feature (d features total)
- Sort by feature value: $O(n \log n)$
- Scan through thresholds: $O(n)$
- Total per node: $O(d \cdot n \log n)$

**For the whole tree:**
- Depth $\log n$ (balanced) to $n$ (degenerate)
- Overall: $O(d \cdot n^2 \log n)$ worst case

**Optimizations:**
- Pre-sort features once
- Use histograms instead of exact splits (gradient boosting trees)

---

## Regression Tree Splits

For regression, use variance reduction instead of Gini/entropy:

**Parent variance:**

$$
\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

**Split criterion:** Minimize weighted child variance

$$
\text{Var reduction} = \text{Var}(S) - \frac{|S_L|}{|S|}\text{Var}(S_L) - \frac{|S_R|}{|S|}\text{Var}(S_R)
$$

The best split maximizes variance reduction.