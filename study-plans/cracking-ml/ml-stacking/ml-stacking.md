# <span style="font-size: 20px;">Stacking Ensemble</span>

Stacking (stacked generalization), introduced by Wolpert in 1992, is an ensemble method that trains a meta-learner to combine the predictions of multiple base models. Unlike bagging and boosting which use the same type of base learner, stacking typically combines diverse models.

---

## <span style="font-size: 16px;">Key Insight: Information Leakage</span>

A naive approach would be: train base models on the training set, get their predictions on the same training set, and train the meta-learner on those. This causes information leakage because the base models have seen the same data they are generating meta-features for, leading to overly optimistic meta-features and poor generalization.

The solution is cross-validation: generate meta-features using out-of-fold predictions so the meta-learner only sees base model predictions on data the base models have not been trained on.

---

## <span style="font-size: 16px;">Algorithm</span>

### <span style="font-size: 16px;">Training Phase</span>

1. Split training data into $K$ folds
2. For each fold $k$:
   - Train each base model on the remaining $K-1$ folds
   - Predict on fold $k$ to generate meta-features for those samples
3. Collect all out-of-fold predictions into an $(n, M)$ meta-feature matrix ($M$ base models)
4. Train the meta-learner on this matrix with the original labels

### <span style="font-size: 16px;">Prediction Phase</span>

1. Train each base model on all training data
2. Generate meta-features for test points using these fully-trained base models
3. Predict using the meta-learner

---

## <span style="font-size: 16px;">Base Model Diversity</span>

Stacking works best when base models are diverse and make different kinds of errors:

- **Decision stump**: fast, linear-like, high bias
- **KNN**: nonparametric, captures local structure
- Other options: logistic regression, naive Bayes, decision trees, SVM

The meta-learner learns which base model to trust in which region of the feature space.

---

## <span style="font-size: 16px;">Meta-Learner Choice</span>

Common meta-learners include:

- **Logistic regression**: simple, regularized, good default
- **Linear regression** (for regression stacking)
- **Ridge/elastic net**: prevents overfitting to base model predictions

A simple linear model is preferred as the meta-learner to avoid overfitting the small meta-feature space. Complex meta-learners risk memorizing the cross-validated predictions.

---

## <span style="font-size: 16px;">Stacking vs. Voting vs. Blending</span>

| Method | Description |
|---|---|
| Majority voting | Equal weight to each model |
| Weighted voting | Fixed weights based on validation accuracy |
| Blending | Like stacking but uses a single holdout set instead of CV |
| Stacking | Learned combination via cross-validated meta-features |

Stacking is the most flexible but also the most complex.

---

## <span style="font-size: 16px;">Practical Considerations</span>

- **Number of folds**: typically 3-5. More folds give more training data per fold but increase computation
- **Number of base models**: diminishing returns beyond 3-5 diverse models
- **Meta-features**: can include raw predictions, class probabilities, or both
- **Multi-level stacking**: stacking can be applied recursively (base -> level-1 -> level-2), but diminishing returns and overfitting risk increase

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- **Q: Why not train meta-learner on training set predictions?**
  A: Information leakage. Base models have seen the training data, so their training predictions are overly confident. CV-based predictions are honest estimates of out-of-sample performance.

- **Q: Can you use the original features alongside meta-features?**
  A: Yes, this is called "stacking with passthrough." It gives the meta-learner access to both raw features and base model predictions.

- **Q: How does this relate to Kaggle competitions?**
  A: Stacking is one of the most successful competition techniques. Top solutions often use multi-level stacking with many diverse base models.

---