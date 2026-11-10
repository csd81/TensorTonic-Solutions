# <span style="font-size: 20px;">Feature Scaling & Normalization</span>

<span style="font-size: 14px;">Feature scaling is one of the most important preprocessing steps in machine learning. Raw features often have wildly different ranges - for instance, age (0-100) vs. income (0-1,000,000). Without scaling, features with large magnitudes dominate distance calculations and gradient updates, leading to poor model performance. Understanding when and how to apply each scaling method is a fundamental skill for any ML practitioner.</span>

---

## <span style="font-size: 16px;">Why Feature Scaling Matters</span>

<span style="font-size: 14px;">Many ML algorithms are sensitive to the scale of input features:</span>

- <span style="font-size: 14px;">**Gradient-based methods** (linear regression, neural networks, logistic regression): features with larger scales produce larger gradients, causing the optimizer to oscillate along those dimensions. Scaling ensures balanced gradient magnitudes across all features, leading to faster and more stable convergence.</span>
- <span style="font-size: 14px;">**Distance-based methods** (k-NN, k-means, SVM with RBF kernel): the Euclidean distance is dominated by features with the widest range. A feature ranging from 0 to 1000 contributes far more to the distance than one ranging from 0 to 1.</span>
- <span style="font-size: 14px;">**Regularization** (L1, L2): penalizes large weights. If features are on different scales, the regularization penalty is uneven - features with small magnitudes need large weights to contribute, and those weights get penalized more heavily.</span>
- <span style="font-size: 14px;">**Algorithms unaffected by scaling**: decision trees, random forests, and gradient boosted trees split on individual features independently, so they are invariant to monotonic transformations of features.</span>

---

## <span style="font-size: 16px;">Min-Max Scaling</span>

<span style="font-size: 14px;">Min-Max scaling linearly maps each feature to the interval</span> $[0, 1]$ <span style="font-size: 14px;">using the formula:</span>

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

<span style="font-size: 14px;">The minimum value maps to 0 and the maximum to 1, with all other values distributed proportionally between them.</span>

<span style="font-size: 14px;">**Strengths**: preserves the original distribution shape; bounded output is useful for algorithms that expect inputs in a fixed range (e.g., neural networks with sigmoid activations).</span>

<span style="font-size: 14px;">**Weaknesses**: highly sensitive to outliers. A single extreme value compresses all other values into a narrow band. For example, if most values are in [0, 100] but one outlier is 10,000, the scaled values for the majority cluster near 0.</span>

<span style="font-size: 14px;">**Edge case**: when a feature is constant (max equals min), the denominator is zero. The standard approach is to return 0 for all values in that feature column.</span>

---

## <span style="font-size: 16px;">Z-Score Standardization</span>

<span style="font-size: 14px;">Z-Score standardization (also called standard scaling) centers each feature at zero mean and scales to unit variance:</span>

$$
x' = \frac{x - \mu}{\sigma}
$$

<span style="font-size: 14px;">where</span> $\mu$ <span style="font-size: 14px;">is the feature mean and</span> $\sigma$ <span style="font-size: 14px;">is the population standard deviation (ddof=0).</span>

<span style="font-size: 14px;">The result tells you how many standard deviations each observation is from the mean. A value of +2.0 means the observation is two standard deviations above the mean.</span>

<span style="font-size: 14px;">**Strengths**: works well when the data is approximately Gaussian. Does not bound the output to a fixed range, which makes it more robust to outliers than min-max scaling.</span>

<span style="font-size: 14px;">**Weaknesses**: if the data is not normally distributed, the "zero mean, unit variance" property is less meaningful. Also, the unbounded output can be problematic for models expecting bounded inputs.</span>

<span style="font-size: 14px;">**Edge case**: when a feature has zero variance (all values identical), the standard deviation is zero. Return 0 for all values to avoid division by zero.</span>

---

## <span style="font-size: 16px;">Robust Scaling</span>

<span style="font-size: 14px;">Robust scaling uses statistics that are resistant to outliers - the median and the interquartile range (IQR):</span>

$$
x' = \frac{x - \text{median}}{Q_3 - Q_1}
$$

<span style="font-size: 14px;">where</span> $Q_1$ <span style="font-size: 14px;">is the 25th percentile and</span> $Q_3$ <span style="font-size: 14px;">is the 75th percentile. The IQR measures the spread of the middle 50% of the data.</span>

<span style="font-size: 14px;">**Strengths**: the median is not affected by extreme values (unlike the mean), and the IQR ignores the tails of the distribution. This makes robust scaling the best choice when the data contains significant outliers.</span>

<span style="font-size: 14px;">**Weaknesses**: does not produce data with a specific mean or variance. The output range is unbounded and depends on the distribution shape.</span>

<span style="font-size: 14px;">**Edge case**: when the IQR is zero (more than half the data is identical), set the denominator to 1.0 to avoid division by zero, which results in the scaled values being 0.</span>

---

## <span style="font-size: 16px;">L2 Normalization</span>

<span style="font-size: 14px;">L2 normalization scales each **row** (sample) so that its Euclidean norm equals 1:</span>

$$
\hat{x} = \frac{x}{\|x\|_2} = \frac{x}{\sqrt{\sum_j x_j^2}}
$$

<span style="font-size: 14px;">Unlike the previous methods that operate column-wise (per feature), L2 normalization operates row-wise (per sample). After normalization, each sample lies on the unit hypersphere.</span>

<span style="font-size: 14px;">**Use cases**: text classification with TF-IDF vectors, cosine similarity computations, and any setting where the direction of the feature vector matters more than its magnitude.</span>

<span style="font-size: 14px;">**Edge case**: if a row has zero norm (all zeros), return the zero vector unchanged to avoid division by zero.</span>

---

## <span style="font-size: 16px;">Choosing the Right Method</span>

<span style="font-size: 14px;">The choice depends on your data characteristics and the downstream algorithm:</span>

- <span style="font-size: 14px;">**Min-Max**: use when features need to be in a bounded range (e.g., image pixel values, neural network inputs). Avoid with outlier-heavy data.</span>
- <span style="font-size: 14px;">**Z-Score**: the default choice for most ML pipelines. Works well with linear models, SVMs, and neural networks. Assumes roughly Gaussian features.</span>
- <span style="font-size: 14px;">**Robust**: use when the data has outliers that you want to keep (not remove). Good for datasets with skewed distributions.</span>
- <span style="font-size: 14px;">**L2 Norm**: use when the angle (direction) of the feature vector matters, not its length. Standard for text and information retrieval.</span>

<span style="font-size: 14px;">A critical practice: always fit scaling parameters (min, max, mean, std, etc.) on the **training set only**, then apply the same transformation to the test set. Fitting on the entire dataset causes data leakage.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: When should you NOT scale features?**</span>
  <span style="font-size: 14px;">A: Tree-based models (decision trees, random forests, gradient boosted trees) are invariant to monotonic feature transformations because they split on thresholds. Scaling does not affect the split quality. Also, features that are already on a meaningful scale (e.g., binary indicators) should generally not be scaled.</span>

- <span style="font-size: 14px;">**Q: What happens if you fit the scaler on the test set?**</span>
  <span style="font-size: 14px;">A: This causes data leakage. The model indirectly gets information about the test distribution through the scaling parameters. For example, the test set's min and max influence the scaled training values, making cross-validation scores overly optimistic. Always fit on train, transform both train and test.</span>

- <span style="font-size: 14px;">**Q: How does feature scaling affect regularization?**</span>
  <span style="font-size: 14px;">A: Without scaling, features with large magnitudes naturally have small weights, while features with small magnitudes need large weights. L2 regularization penalizes large weights uniformly, so it disproportionately shrinks the weights of small-scale features. Scaling ensures equal treatment across all features.</span>

- <span style="font-size: 14px;">**Q: What is the difference between normalization and standardization?**</span>
  <span style="font-size: 14px;">A: These terms are used inconsistently in practice. Strictly, "normalization" rescales values to a bounded range (like [0,1] via min-max), while "standardization" transforms to zero mean and unit variance (z-score). "L2 normalization" scales vectors to unit norm. In interviews, always clarify which specific transformation is meant.</span>

- <span style="font-size: 14px;">**Q: Can you apply different scaling methods to different features?**</span>
  <span style="font-size: 14px;">A: Yes, and sometimes you should. For example, you might use robust scaling for features with outliers and z-score for well-behaved Gaussian features. Scikit-learn's ColumnTransformer lets you apply different transformers to different columns in a single pipeline.</span>

---