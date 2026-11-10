# <span style="font-size: 20px;">AUC-ROC from Scratch</span>

<span style="font-size: 14px;">The ROC (Receiver Operating Characteristic) curve is a fundamental evaluation tool for binary classifiers. It plots TPR vs. FPR at every possible threshold, and the AUC (Area Under the Curve) summarizes discriminative ability in a single number.</span>

---

## <span style="font-size: 16px;">Definitions</span>

$$
\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}
$$

- <span style="font-size: 14px;">**TPR (Sensitivity/Recall)**: fraction of positives correctly identified</span>
- <span style="font-size: 14px;">**FPR (1 - Specificity)**: fraction of negatives incorrectly classified as positive</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Sort samples by predicted score (descending)</span>
2. <span style="font-size: 14px;">For each unique threshold value:</span>
   - <span style="font-size: 14px;">Classify points with score >= threshold as positive</span>
   - <span style="font-size: 14px;">Compute TPR and FPR</span>
   - <span style="font-size: 14px;">Record the (FPR, TPR) point</span>
3. <span style="font-size: 14px;">Connect the points to form the ROC curve</span>
4. <span style="font-size: 14px;">Compute the area using the trapezoidal rule</span>

---

## <span style="font-size: 16px;">Interpreting AUC</span>

- <span style="font-size: 14px;">**AUC = 1.0**: perfect classifier (all positives ranked above all negatives)</span>
- <span style="font-size: 14px;">**AUC = 0.5**: random classifier (no discriminative power)</span>
- <span style="font-size: 14px;">**AUC < 0.5**: worse than random (predictions are inverted)</span>
- <span style="font-size: 14px;">**AUC = 0.0**: perfectly wrong (all negatives ranked above all positives)</span>

<span style="font-size: 14px;">Probabilistic interpretation: AUC equals the probability that a randomly chosen positive is scored higher than a randomly chosen negative, plus half the probability that their scores are equal.</span>

---

## <span style="font-size: 16px;">ROC vs. Precision-Recall</span>

- <span style="font-size: 14px;">ROC curves are robust to class imbalance in the evaluation (FPR uses only negatives, TPR uses only positives)</span>
- <span style="font-size: 14px;">However, for highly imbalanced datasets, Precision-Recall curves give a more informative picture because precision is directly affected by the number of false positives relative to true positives</span>

---

## <span style="font-size: 16px;">Computational Complexity</span>

- <span style="font-size: 14px;">Sorting by predicted scores costs $O(n \log n)$, which dominates the total computation</span>
- <span style="font-size: 14px;">After sorting, a single pass through the ranked list computes all (FPR, TPR) pairs in $O(n)$</span>
- <span style="font-size: 14px;">The Wilcoxon-Mann-Whitney statistic gives AUC in $O(n \log n)$ without constructing the full curve</span>
- <span style="font-size: 14px;">For very large datasets, approximate AUC methods sample positive-negative pairs to reduce cost</span>
---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: When is AUC misleading?**</span>
  <span style="font-size: 14px;">A: When you care about a specific operating point (threshold), AUC averages over all thresholds. Also, in heavily imbalanced datasets, a high AUC can mask poor precision.</span>

- <span style="font-size: 14px;">**Q: Can AUC be computed without thresholds?**</span>
  <span style="font-size: 14px;">A: Yes, compare positive-negative pairs: award one for a higher positive score, one half for equal scores, and zero for a lower positive score, then average across all pairs.</span>

- <span style="font-size: 14px;">**Q: What about multi-class AUC?**</span>
  <span style="font-size: 14px;">A: Use one-vs-rest or one-vs-one AUC and average (macro or weighted).</span>

---