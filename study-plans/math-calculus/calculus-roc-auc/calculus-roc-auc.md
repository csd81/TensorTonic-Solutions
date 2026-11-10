# <span style="font-size: 20px;">Area Under the ROC Curve via Trapezoidal Rule</span>

## Binary Classification and Decision Thresholds

In binary classification, a model outputs a continuous score $s(x)$ for each input $x$. This score might be a predicted probability, a logit, or any real-valued quantity where higher values indicate greater confidence that the input belongs to the positive class. To convert a continuous score into a hard binary prediction, we apply a decision threshold $t$:

$$\hat{y} = \begin{cases} 1 & \text{if } s(x) \geq t \\ 0 & \text{if } s(x) < t \end{cases}
$$

The choice of threshold profoundly affects the classifier's behavior. A low threshold classifies many samples as positive, catching most true positives but also producing many false positives. A high threshold is conservative, producing few false positives but potentially missing many true positives. No single threshold tells the whole story of a classifier's quality - this motivates the ROC curve, which evaluates performance across all possible thresholds simultaneously.

## The Confusion Matrix

For a binary classifier at a given threshold, every prediction falls into one of four categories:

|                        | Predicted Positive    | Predicted Negative    |
|------------------------|-----------------------|-----------------------|
| **Actually Positive**  | True Positive (TP)    | False Negative (FN)   |
| **Actually Negative**  | False Positive (FP)   | True Negative (TN)    |

The total number of actual positives is $P = \text{TP} + \text{FN}$ and the total number of actual negatives is $N = \text{FP} + \text{TN}$. From these four counts, we derive several important rates.

**True Positive Rate (TPR)**, also called sensitivity or recall:

$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{P}$$

TPR measures the fraction of positive samples correctly identified.

**False Positive Rate (FPR)**, also called the fall-out rate or (1 - specificity):

$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{N}$$

FPR measures the fraction of negative samples incorrectly classified as positive.

**Specificity** is the complement of FPR:

$$\text{Specificity} = 1 - \text{FPR} = \frac{\text{TN}}{N}$$

A critical observation: TPR and FPR are computed over different populations. TPR uses only the actual positive samples as its denominator, while FPR uses only the actual negative samples. This independence from overall class balance is a key advantage of ROC analysis and explains why AUC is robust to imbalanced datasets.

## The Receiver Operating Characteristic (ROC) Curve

The ROC curve is a plot of TPR (y-axis) versus FPR (x-axis) as the decision threshold varies continuously from positive infinity to negative infinity.

### Construction Algorithm

Given $n$ samples with true labels $y_i \in \{0, 1\}$ and predicted scores $s_i \in \mathbb{R}$:

1. **Sort** the samples by score in descending order: $s_{\pi(1)} \geq s_{\pi(2)} \geq \cdots \geq s_{\pi(n)}$

2. **Initialize** at the origin: TPR = 0, FPR = 0. This corresponds to a threshold above all scores, where nothing is classified as positive: TP = 0, FP = 0.

3. **Sweep** through sorted samples. For each sample $i$ processed in descending score order, we effectively lower the threshold to just below $s_{\pi(i)}$, classifying sample $i$ as positive:
   - If $y_{\pi(i)} = 1$ (it is a true positive): increment TP by 1, so TPR increases by $1/P$
   - If $y_{\pi(i)} = 0$ (it is a false positive): increment FP by 1, so FPR increases by $1/N$
   - Record the new (FPR, TPR) point

4. **Terminate** at (1, 1): the threshold is below all scores, so everything is classified as positive: TPR = 1, FPR = 1.

This produces $n + 1$ points on the ROC curve (including the origin at step 0).

### Geometric Interpretation

Each point on the ROC curve corresponds to a specific operating threshold. Moving along the curve from (0, 0) to (1, 1) corresponds to progressively lowering the threshold from infinity to negative infinity, classifying more and more samples as positive.

- A **vertical segment** occurs when a true positive is encountered: TPR increases while FPR stays constant. This is desirable - we are catching a positive without incurring a false alarm.

- A **horizontal segment** occurs when a false positive is encountered: FPR increases while TPR stays constant. This is undesirable - we are adding a false alarm without catching any additional positives.

A perfect classifier encounters all true positives before any false positives (when sorted by score), so its ROC curve goes straight up from (0, 0) to (0, 1) and then right to (1, 1). This encloses the maximum possible area.

A random classifier (scores independent of labels) has positives and negatives interleaved randomly when sorted by score. In expectation, for every $1/P$ increase in TPR, there is a corresponding $1/N$ increase in FPR, and the curve follows the diagonal from (0, 0) to (1, 1).

An inverse classifier (lower scores for positives) encounters all false positives before true positives, producing a curve that goes right from (0, 0) to (1, 0) and then up to (1, 1). This encloses the minimum possible area.

### Properties of the ROC Curve

1. **Always starts at (0, 0)** (threshold above all scores) and **ends at (1, 1)** (threshold below all scores)

2. **Monotonically non-decreasing** in both coordinates: as we process more samples, both TP and FP can only increase or stay the same, so TPR and FPR are non-decreasing.

3. **Scale invariant**: the ROC curve depends only on the ranking (ordering) of the scores, not their absolute values. Multiplying all scores by a positive constant or adding a constant to all scores does not change the ROC curve.

4. **Threshold invariant**: the curve summarizes performance across all thresholds simultaneously, without privileging any single operating point.

## Area Under the ROC Curve (AUC)

The AUC condenses the entire ROC curve into a single scalar value:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$$

This is the area of the region bounded by the ROC curve above and the FPR axis below, integrated from FPR = 0 to FPR = 1.

### Key Properties

- **Range**: $0 \leq \text{AUC} \leq 1$
- **Perfect classifier**: AUC = 1 (the ROC curve covers the entire unit square above the diagonal)
- **Random classifier**: AUC = 1/2 (the ROC curve follows the diagonal, enclosing a triangle of area 1/2)
- **Inverse classifier**: AUC = 0 (the ROC curve hugs the bottom and right edges)

### Probabilistic Interpretation: The Concordance Probability

The AUC has a probabilistic interpretation. It equals the probability that a randomly chosen positive sample receives a higher score than a randomly chosen negative sample:

$$\text{AUC} = P(s(x^+) > s(x^-))$$

This is known as the **concordance probability** or the **c-statistic**. For $P$ positive samples and $N$ negative samples, there are $P \times N$ pairs, and:

$$\text{AUC} = \frac{\text{number of concordant pairs}}{P \times N}$$

where a concordant pair $(x^+, x^-)$ satisfies $s(x^+) > s(x^-)$, meaning the positive sample has a strictly higher score than the negative sample.

This interpretation makes AUC intuitive: if you randomly pick one positive and one negative sample, AUC is the probability the model correctly ranks the positive above the negative.

### Why AUC is Preferred Over Accuracy

Accuracy conflates two very different types of errors (false positives and false negatives) and weights them by class prevalence. In imbalanced datasets, a trivial classifier that always predicts the majority class can achieve a high accuracy while having zero discriminative ability.

AUC, by contrast, evaluates TPR and FPR separately over their respective populations. The trivial "always negative" classifier sits at the origin of ROC space, the worst possible starting point, and its AUC correctly reflects that it has no ranking ability.

More broadly, AUC separates error types and is invariant to class balance, making it a more reliable metric for comparing classifiers across different datasets and application domains.

## The Trapezoidal Rule: Review

The trapezoidal rule is a method for numerical integration that approximates the definite integral of a function by summing the areas of trapezoids formed between consecutive sample points.

For a function sampled at points $(x_0, y_0), (x_1, y_1), \ldots, (x_m, y_m)$ where $x_0 < x_1 < \cdots < x_m$:

$$\int_{x_0}^{x_m} f(x) \, dx \approx \sum_{i=1}^{m} \frac{(x_i - x_{i-1})(y_i + y_{i-1})}{2}$$

Each term computes the area of a trapezoid with:
- Width: $\Delta x_i = x_i - x_{i-1}$ (the horizontal distance between consecutive points)
- Parallel sides of height $y_{i-1}$ (left) and $y_i$ (right)
- Area = width $\times$ average height = $\Delta x_i \cdot (y_{i-1} + y_i) / 2$

For smooth functions, the trapezoidal rule has an approximation error of order $O(h^2)$ where $h$ is the step size. But for piecewise linear functions, the trapezoidal rule gives the exact integral because it uses linear interpolation between points, and the function is already piecewise linear.

### Why the Trapezoidal Rule is Exact for ROC Curves

The ROC curve constructed by processing one sample at a time consists entirely of horizontal segments (when a negative sample is processed) and vertical segments (when a positive sample is processed). Between any two consecutive ROC points, the curve is a straight line.

For a horizontal segment from $(x_i, y)$ to $(x_{i+1}, y)$:
$$\text{Trapezoid area} = (x_{i+1} - x_i) \cdot \frac{y + y}{2} = (x_{i+1} - x_i) \cdot y$$

This is the area of a rectangle, which is exactly correct for a horizontal line segment.

For a vertical segment from $(x, y_i)$ to $(x, y_{i+1})$:
$$\text{Trapezoid area} = (x - x) \cdot \frac{y_i + y_{i+1}}{2} = 0$$

Correct again: a vertical segment contributes no area under the curve (it has zero width).

Because the ROC curve is piecewise linear and the trapezoidal rule is exact for linear segments, the trapezoidal rule computes the exact AUC with no approximation error whatsoever.

## Computing AUC with the Trapezoidal Rule

Combining the ROC curve construction with the trapezoidal rule gives us a complete algorithm for computing AUC.

Given $n + 1$ ROC curve points $(\text{FPR}_0, \text{TPR}_0), (\text{FPR}_1, \text{TPR}_1), \ldots, (\text{FPR}_n, \text{TPR}_n)$:

$$\text{AUC} = \sum_{i=1}^{n} \frac{(\text{FPR}_i - \text{FPR}_{i-1})(\text{TPR}_i + \text{TPR}_{i-1})}{2}$$

### Complete Pseudocode

```
function AUC_ROC(y_true, y_scores):
    n = length(y_true)
    P = count(y_true == 1)      # total positives
    N = count(y_true == 0)      # total negatives

    # Sort indices by descending score (stable sort)
    indices = argsort(-y_scores, stable=True)

    # Initialize ROC curve at origin
    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0

    # Build ROC curve by processing each sample
    for i in indices:
        if y_true[i] == 1:
            tp += 1             # true positive found
        else:
            fp += 1             # false positive found
        fpr.append(fp / N)
        tpr.append(tp / P)

    # Trapezoidal integration
    auc = 0
    for i = 1 to n:
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return fpr, tpr, auc
```

### Computational Complexity

- **Sorting**: $O(n \log n)$ to sort the scores
- **ROC curve construction**: $O(n)$ to sweep through sorted samples
- **Trapezoidal integration**: $O(n)$ to sum the trapezoid areas
- **Overall**: $O(n \log n)$, dominated by the sorting step

The space complexity is $O(n)$ for storing the ROC curve points. For applications where only the AUC is needed (not the full ROC curve), the integration can be done incrementally in $O(1)$ additional space.

## ROC Curves and Cumulative Distribution Functions

The ROC curve has an elegant interpretation in terms of cumulative distribution functions. Let $F_1(t)$ and $F_0(t)$ be the CDFs of the scores for the positive and negative classes, respectively:

$$F_1(t) = P(s \leq t \mid y = 1), \quad F_0(t) = P(s \leq t \mid y = 0)$$

Then at threshold $t$:
- $\text{TPR}(t) = 1 - F_1(t)$: the fraction of positives scoring above $t$
- $\text{FPR}(t) = 1 - F_0(t)$: the fraction of negatives scoring above $t$

The ROC curve is the parametric curve $(\text{FPR}(t), \text{TPR}(t))$ as $t$ decreases from $+\infty$ to $-\infty$. In terms of the CDFs:

$$\text{TPR} = 1 - F_1(F_0^{-1}(1 - \text{FPR}))$$

For well-separated score distributions (positive scores much higher than negative scores), $F_1$ and $F_0$ have minimal overlap, and the ROC curve bulges toward the upper-left corner (AUC approaching 1). When the score distributions overlap completely (scores convey no information about class), the ROC curve approaches the diagonal (AUC approaching 1/2).

### Separation and Discrimination

The degree to which the positive and negative score distributions are separated determines the shape of the ROC curve. This can be formalized through the class-conditional densities $f_1(s)$ and $f_0(s)$, where:

$$\text{TPR}(t) = \int_t^{\infty} f_1(s) \, ds, \quad \text{FPR}(t) = \int_t^{\infty} f_0(s) \, ds$$

If $f_1$ and $f_0$ are Gaussian with equal variance $\sigma^2$ but different means $\mu_1 > \mu_0$, the ROC curve depends only on the standardized distance $d' = (\mu_1 - \mu_0)/\sigma$, known as the discriminability index. As $d' \to \infty$, the distributions separate completely and AUC approaches 1. As $d' \to 0$, the distributions overlap entirely and AUC approaches 1/2.

This Gaussian case is a useful theoretical benchmark: the ROC curve has a closed-form expression involving the normal CDF $\Phi$:

$$\text{TPR} = \Phi\left(\Phi^{-1}(\text{FPR}) + d'\right)$$

This relationship is linear in normal deviate space, which is the basis of the DET (Detection Error Trade-off) curve used in biometrics.

## Relationship to the Wilcoxon-Mann-Whitney Statistic

The concordance probability interpretation of AUC links it to the normalized Wilcoxon-Mann-Whitney U statistic from nonparametric statistics:

$$U = \sum_{i=1}^{P} \sum_{j=1}^{N} \mathbf{1}[s_i^+ > s_j^-], \quad \text{AUC} = \frac{U}{P \cdot N}$$

This connection has practical implications: hypothesis tests for comparing AUC values between two classifiers can be derived from the theory of U-statistics, providing confidence intervals and significance tests.

## Implementation Considerations

### Numerical Stability

The trapezoidal rule for ROC curves is inherently numerically stable because:

- All FPR and TPR values are rational numbers of the form $k/P$ or $k/N$ where $0 \leq k \leq \max(P, N)$
- Differences $\Delta\text{FPR}$ are either exactly 0 or exactly $1/N$ (when processing one sample at a time)
- Products are bounded: each trapezoid area is at most $1/N$
- There are no divisions by very small numbers, no logarithms, and no exponentials

This is in contrast to many other ML metrics (like cross-entropy loss or KL divergence) where numerical stability requires careful handling of near-zero values.

### Sorting Stability

Using a stable sort ensures that samples with identical scores maintain their relative order from the input. For strictly unique scores, any sorting algorithm produces the same result. But when ties exist, a stable sort makes the output deterministic and reproducible.

### Handling Tied Scores

When multiple samples share the same score, the order in which they are processed can affect the shape of the ROC curve (though not necessarily the AUC). The standard convention is to treat tied samples as a group: all samples with the same score are processed simultaneously, producing a single diagonal step rather than separate horizontal and vertical steps.

Formally, if at some score value $t$ there are $p$ positives and $q$ negatives, the ROC curve jumps from the current point by $(\Delta\text{FPR}, \Delta\text{TPR}) = (q/N, p/P)$ in a single step. The trapezoidal rule then computes the area of the resulting trapezoid, which correctly averages the two possible orderings (all positives first vs. all negatives first).

### Vectorized Computation

While the ROC curve construction uses a loop, the trapezoidal integration step can be vectorized:

```python
fpr_arr = np.array(fpr)
tpr_arr = np.array(tpr)
delta_fpr = np.diff(fpr_arr)
avg_tpr = (tpr_arr[:-1] + tpr_arr[1:]) / 2.0
auc = np.sum(delta_fpr * avg_tpr)
```

### Memory-Efficient Incremental Computation

If only the AUC value is needed (not the full ROC curve), the computation can be done incrementally without storing all the points:

```
auc = 0
prev_fpr = 0
prev_tpr = 0
for each sample in sorted order:
    if label == 1:
        new_tpr = prev_tpr + 1/P
        new_fpr = prev_fpr
    else:
        new_tpr = prev_tpr
        new_fpr = prev_fpr + 1/N
    auc += (new_fpr - prev_fpr) * (new_tpr + prev_tpr) / 2
    prev_fpr = new_fpr
    prev_tpr = new_tpr
```

This uses $O(1)$ additional memory beyond the input.

## Why AUC Matters in Machine Learning

AUC is one of the most widely used metrics for evaluating binary classifiers across diverse domains.

**Threshold independence**: Unlike accuracy, precision, recall, or F1-score, AUC does not require choosing a decision threshold. It evaluates the classifier's ability to rank positive samples above negative samples across all possible thresholds simultaneously. This is especially valuable during model selection, where the deployment threshold may not yet be determined.

**Class balance independence**: Because TPR and FPR are computed separately over the positive and negative populations, AUC is invariant to class imbalance. A dataset with a small fraction of positives yields the same AUC as a balanced dataset, given the same score distributions within each class. This makes AUC the go-to metric for fraud detection, medical diagnosis, anomaly detection, and other domains with severe class imbalance.

**Model comparison**: Two models can be compared by their AUC values without needing to agree on a threshold. The model with higher AUC has better ranking ability across all operating points.

**Calibration independence**: AUC depends only on the ranking of scores, not their calibration. A model that assigns barely distinguishable scores to the two classes still achieves perfect AUC if the ranking is correct. This can be either an advantage (robust to monotonic score transformations) or a limitation (does not assess probability quality).

**Limitations of AUC**: AUC can be misleading when the costs of false positives and false negatives are highly asymmetric, or when only a narrow operating region is relevant. A model might have high overall AUC but perform poorly in the specific FPR range that matters for deployment. In such cases, the precision-recall curve may be more informative.

## Connection to Other Evaluation Metrics

The ROC curve is one of several curve-based evaluation tools in ML:

**Precision-Recall curve**: Plots precision (TP / (TP + FP)) against recall (same as TPR). The area under the PR curve (AUPRC or average precision) is another popular summary metric, especially for highly imbalanced datasets where the positive class is rare. Unlike AUC-ROC, AUPRC is sensitive to class imbalance.

**Detection Error Trade-off (DET) curve**: Plots false negative rate (1 - TPR) against FPR on a normal deviate scale. Used primarily in biometrics and speaker verification.

**Calibration curve**: Plots predicted probability against observed frequency. Not directly an integral-based metric, but complements AUC by assessing probability quality rather than ranking quality.

All of these curve-based metrics share a common computational pattern: construct a curve by varying a threshold, then summarize the curve by computing an area (integral). The trapezoidal rule is the standard numerical integration method for all of them.
