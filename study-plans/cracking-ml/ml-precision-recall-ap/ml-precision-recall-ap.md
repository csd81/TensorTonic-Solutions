# <span style="font-size: 20px;">Precision-Recall Curve & Average Precision</span>

<span style="font-size: 14px;">The Precision-Recall (PR) curve is especially useful for evaluating classifiers on imbalanced datasets, where the ROC curve can be overly optimistic. AP provides a single-number summary of the PR curve.</span>

---

## <span style="font-size: 16px;">Definitions</span>

$$
\begin{aligned}
\text{Precision} &= \frac{TP}{TP + FP} \\[4pt]
\text{Recall} &= \frac{TP}{TP + FN} = \frac{TP}{P}
\end{aligned}
$$

- <span style="font-size: 14px;">**Precision**: fraction of predicted positives that are truly positive</span>
- <span style="font-size: 14px;">**Recall (Sensitivity)**: fraction of actual positives that are correctly identified</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Sort samples by predicted score (descending)</span>
2. <span style="font-size: 14px;">For each unique threshold value:</span>
   - <span style="font-size: 14px;">Classify points with score >= threshold as positive</span>
   - <span style="font-size: 14px;">Compute Precision and Recall</span>
   - <span style="font-size: 14px;">Record the (Recall, Precision) point</span>
3. <span style="font-size: 14px;">Anchor the curve at (Recall=0, Precision=1)</span>
4. <span style="font-size: 14px;">Compute AP as the sum of (Recall_k - Recall_{k-1}) * Precision_k</span>

---

## <span style="font-size: 16px;">Interpreting AP</span>

- <span style="font-size: 14px;">**AP = 1.0**: perfect classifier (all positives ranked before all negatives)</span>
- <span style="font-size: 14px;">**AP = 0.5** (for balanced classes): roughly random performance</span>
- <span style="font-size: 14px;">**Baseline AP** equals the proportion of positives in the dataset</span>

<span style="font-size: 14px;">Unlike AUC-ROC, AP is directly affected by class imbalance, making it a stricter metric when positives are rare.</span>

---

## <span style="font-size: 16px;">PR vs. ROC</span>

- <span style="font-size: 14px;">ROC uses FPR (based on negatives count), which is stable under class imbalance</span>
- <span style="font-size: 14px;">PR uses Precision (based on predicted positives), which drops when false positives increase</span>
- <span style="font-size: 14px;">For imbalanced problems (rare positives), PR curves are more informative</span>

---

## <span style="font-size: 16px;">Practical Considerations</span>

- <span style="font-size: 14px;">AP is sensitive to the ranking of the first few predictions: a false positive at the top of the list causes a large precision drop that propagates through the entire curve</span>
- <span style="font-size: 14px;">When comparing models, always use the same positive class prevalence - AP baselines shift with class balance</span>
- <span style="font-size: 14px;">Interpolated vs. non-interpolated AP: some frameworks (e.g., PASCAL VOC) use 11-point interpolation, while others (scikit-learn) compute the exact area - be clear about which variant you are using</span>
- <span style="font-size: 14px;">For multi-label or object detection tasks, AP is computed per class and then averaged to get mAP</span>
---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why start at (0, 1)?**</span>
  <span style="font-size: 14px;">A: With a threshold above all scores, no predictions are made. By convention, precision is 1.0 and recall is 0.0 at this point.</span>

- <span style="font-size: 14px;">**Q: How do tied scores affect the curve?**</span>
  <span style="font-size: 14px;">A: Tied scores are grouped into a single threshold. All samples at that score are predicted positive simultaneously.</span>

- <span style="font-size: 14px;">**Q: What is the relationship between AP and mAP?**</span>
  <span style="font-size: 14px;">A: mAP (mean Average Precision) is the mean of AP values across multiple classes or queries, commonly used in object detection and information retrieval.</span>

---