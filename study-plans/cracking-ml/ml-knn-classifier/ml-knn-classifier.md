# <span style="font-size: 20px;">KNN Classifier</span>

<span style="font-size: 14px;">K-Nearest Neighbors is one of the simplest and most intuitive classification algorithms. It makes no assumptions about the underlying data distribution (non-parametric) and requires no training phase, making it a "lazy learning" method that stores the entire training set and defers computation to prediction time.</span>

---

## <span style="font-size: 16px;">Core Idea</span>

- <span style="font-size: 14px;">To classify a new point, find the</span> $k$ <span style="font-size: 14px;">training points closest to it</span>
- <span style="font-size: 14px;">Take a majority vote among their labels</span>
- <span style="font-size: 14px;">The decision boundary is implicitly defined by the training data and the value of</span> $k$
- <span style="font-size: 14px;">No model parameters are learned; the training data itself is the model</span>

---

## <span style="font-size: 16px;">Distance Metrics</span>

<span style="font-size: 14px;">The most common choice is Euclidean distance:</span>

$$
d(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}
$$

<span style="font-size: 14px;">Other common metrics include:</span>

- <span style="font-size: 14px;">**Manhattan distance**:</span> $d(x, x') = \sum_j |x_j - x'_j|$
- <span style="font-size: 14px;">**Minkowski distance** (generalization):</span> $d(x, x') = \left(\sum_j |x_j - x'_j|^p\right)^{1/p}$
- <span style="font-size: 14px;">**Cosine distance**: measures the angle between vectors rather than magnitude, useful for text data</span>

<span style="font-size: 14px;">The choice of distance metric depends on the data. Euclidean works well when features are on similar scales.</span>

---

## <span style="font-size: 16px;">Choosing K</span>

- $k = 1$<span style="font-size: 14px;">: assigns the label of the single nearest neighbor. Very sensitive to noise, produces complex (overfitting) decision boundaries</span>
- <span style="font-size: 14px;">Small</span> $k$<span style="font-size: 14px;">: low bias, high variance. Captures fine-grained patterns but is sensitive to outliers</span>
- <span style="font-size: 14px;">Large</span> $k$<span style="font-size: 14px;">: high bias, low variance. Smoother decision boundaries but may miss local structure</span>
- $k = n$<span style="font-size: 14px;">: always predicts the majority class in the training set (maximum bias)</span>
- <span style="font-size: 14px;">Odd</span> $k$ <span style="font-size: 14px;">is preferred for binary classification to avoid ties</span>
- <span style="font-size: 14px;">Common practice: use cross-validation to select</span> $k$<span style="font-size: 14px;">, typical values range from 3 to 15</span>

---

## <span style="font-size: 16px;">Feature Scaling</span>

- <span style="font-size: 14px;">KNN is sensitive to feature scales because the distance metric treats all dimensions equally</span>
- <span style="font-size: 14px;">A feature ranging from 0 to 1000 will dominate a feature ranging from 0 to 1</span>
- <span style="font-size: 14px;">Always standardize or normalize features before applying KNN</span>
- <span style="font-size: 14px;">Common approaches: z-score normalization (subtract mean, divide by std) or min-max scaling</span>

---

## <span style="font-size: 16px;">Computational Complexity</span>

- <span style="font-size: 14px;">**Training**: none (just store the data),</span> $O(1)$
- <span style="font-size: 14px;">**Prediction**: for each test point, compute distance to all</span> $n$ <span style="font-size: 14px;">training points, then find the</span> $k$ <span style="font-size: 14px;">smallest:</span> $O(nd + n \log k)$ <span style="font-size: 14px;">per query</span>
- <span style="font-size: 14px;">For large datasets, this is prohibitively slow. Approximate methods exist:</span>
  - <span style="font-size: 14px;">**KD-trees**: partition space for</span> $O(d \log n)$ <span style="font-size: 14px;">average query time (ineffective in high dimensions)</span>
  - <span style="font-size: 14px;">**Ball trees**: better than KD-trees for moderate dimensions</span>
  - <span style="font-size: 14px;">**Locality-sensitive hashing (LSH)**: approximate nearest neighbors in sublinear time</span>

---

## <span style="font-size: 16px;">Weighted KNN</span>

- <span style="font-size: 14px;">Standard KNN gives equal weight to all</span> $k$ <span style="font-size: 14px;">neighbors regardless of distance</span>
- <span style="font-size: 14px;">Weighted KNN assigns higher weight to closer neighbors:</span>

$$
w_i = \frac{1}{d(x, x_i)^2}
$$

- <span style="font-size: 14px;">The predicted label becomes the class with the highest total weight among neighbors</span>
- <span style="font-size: 14px;">This reduces the influence of distant points within the</span> $k$ <span style="font-size: 14px;">neighborhood</span>

---

## <span style="font-size: 16px;">Curse of Dimensionality</span>

- <span style="font-size: 14px;">In high dimensions, all points become approximately equidistant from each other</span>
- <span style="font-size: 14px;">The "nearest neighbor" becomes meaningless when the ratio of nearest to farthest distance approaches 1</span>
- <span style="font-size: 14px;">KNN degrades rapidly beyond roughly 20 dimensions without dimensionality reduction</span>
- <span style="font-size: 14px;">PCA or feature selection should be applied before KNN in high-dimensional settings</span>

---

## <span style="font-size: 16px;">KNN for Regression</span>

- <span style="font-size: 14px;">Instead of majority vote, average the target values of the</span> $k$ <span style="font-size: 14px;">nearest neighbors</span>
- <span style="font-size: 14px;">Weighted averaging (inverse distance weighting) often improves results</span>
- <span style="font-size: 14px;">Same computational tradeoffs and scaling concerns apply</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why is KNN called a lazy learner?**</span>
  <span style="font-size: 14px;">A: It performs no computation during training; it simply stores the data. All computation happens at prediction time when distances are calculated. This contrasts with eager learners like logistic regression that learn model parameters during training</span>

- <span style="font-size: 14px;">**Q: How do you handle ties?**</span>
  <span style="font-size: 14px;">A: For an equal vote count between classes, common strategies include: choosing the class with the nearest individual neighbor, choosing the smallest class label, or using an odd $k$ to prevent ties in binary classification</span>

- <span style="font-size: 14px;">**Q: Why does KNN fail in high dimensions?**</span>
  <span style="font-size: 14px;">A: The curse of dimensionality makes distances between points nearly uniform, so the concept of "nearest" loses meaning. The volume of a high-dimensional hypersphere grows exponentially, requiring exponentially more data to maintain neighbor density</span>

- <span style="font-size: 14px;">**Q: KNN vs parametric models?**</span>
  <span style="font-size: 14px;">A: KNN makes no assumptions about the data distribution (non-parametric), so it can model arbitrarily complex boundaries. However, it is slow at prediction, memory-intensive, and requires careful feature scaling. Parametric models are faster at inference and generalize better when their assumptions hold</span>

- <span style="font-size: 14px;">**Q: How to speed up KNN?**</span>
  <span style="font-size: 14px;">A: Use spatial data structures (KD-trees, ball trees) for exact neighbors in low dimensions, or approximate nearest neighbor methods (LSH, FAISS) for high dimensions. Reducing dimensionality with PCA also helps</span>

---