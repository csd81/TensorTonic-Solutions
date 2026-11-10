# <span style="font-size: 20px;">AdaBoost from Scratch</span>

<span style="font-size: 14px;">AdaBoost (Adaptive Boosting), introduced by Freund and Schapire in 1997, is the first practical boosting algorithm. It combines many weak learners into a strong classifier by sequentially training each learner on reweighted data, focusing on previously misclassified examples.</span>

---

## <span style="font-size: 16px;">Boosting Intuition</span>

<span style="font-size: 14px;">A weak learner is a classifier that performs only slightly better than random guessing (error</span> $< 0.5$<span style="font-size: 14px;">). Boosting theory shows that combining many weak learners can produce an arbitrarily accurate strong learner. AdaBoost achieves this by:</span>

1. <span style="font-size: 14px;">Training a weak learner on weighted data</span>
2. <span style="font-size: 14px;">Increasing weights on misclassified points so the next learner focuses on them</span>
3. <span style="font-size: 14px;">Assigning each learner a vote proportional to its accuracy</span>

---

## <span style="font-size: 16px;">Decision Stumps</span>

<span style="font-size: 14px;">The most common weak learner for AdaBoost is the decision stump: a depth-1 decision tree that splits on a single feature. For each feature</span> $j$ <span style="font-size: 14px;">and threshold</span> $t$<span style="font-size: 14px;">, two polarities are considered:</span>

- <span style="font-size: 14px;">**Polarity +1**: predict</span> $+1$ <span style="font-size: 14px;">if</span> $x_j > t$<span style="font-size: 14px;">, else</span> $-1$
- <span style="font-size: 14px;">**Polarity -1**: predict</span> $+1$ <span style="font-size: 14px;">if</span> $x_j \leq t$<span style="font-size: 14px;">, else</span> $-1$

<span style="font-size: 14px;">The stump with the lowest weighted error is selected at each round.</span>

---

## <span style="font-size: 16px;">AdaBoost Algorithm</span>

<span style="font-size: 14px;">Initialize sample weights</span> $w_i = 1/n$ <span style="font-size: 14px;">for</span> $i = 1, \ldots, n$<span style="font-size: 14px;">. For each round</span> $t = 1, \ldots, T$<span style="font-size: 14px;">:</span>

1. <span style="font-size: 14px;">Train weak learner</span> $h_t$ <span style="font-size: 14px;">to minimize weighted error:</span> $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
2. <span style="font-size: 14px;">Compute learner weight:</span> $\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$
3. <span style="font-size: 14px;">Update sample weights:</span> $w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))$
4. <span style="font-size: 14px;">Normalize:</span> $w_i \leftarrow w_i / \sum_j w_j$

<span style="font-size: 14px;">Final prediction:</span>

$$
H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)
$$

---

## <span style="font-size: 16px;">Why the Weight Update Works</span>

<span style="font-size: 14px;">The update</span> $w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h_t(x_i))$ <span style="font-size: 14px;">has an elegant structure:</span>

- <span style="font-size: 14px;">If</span> $h_t(x_i) = y_i$ <span style="font-size: 14px;">(correct): the exponent is</span> $-\alpha_t$ <span style="font-size: 14px;">(negative), so</span> $w_i$ <span style="font-size: 14px;">decreases</span>
- <span style="font-size: 14px;">If</span> $h_t(x_i) \neq y_i$ <span style="font-size: 14px;">(wrong): the exponent is</span> $+\alpha_t$ <span style="font-size: 14px;">(positive), so</span> $w_i$ <span style="font-size: 14px;">increases</span>

<span style="font-size: 14px;">This forces subsequent learners to focus on the hardest examples. AdaBoost can be shown to minimize the exponential loss function:</span>

$$
L = \sum_{i=1}^{n} \exp\left(-y_i \sum_{t=1}^{T} \alpha_t h_t(x_i)\right)
$$

---

## <span style="font-size: 16px;">Learner Weight</span> $\alpha_t$

<span style="font-size: 14px;">The formula</span> $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ <span style="font-size: 14px;">has intuitive properties:</span>

- <span style="font-size: 14px;">If</span> $\epsilon_t = 0$ <span style="font-size: 14px;">(perfect):</span> $\alpha_t \to \infty$ <span style="font-size: 14px;">(this learner dominates)</span>
- <span style="font-size: 14px;">If</span> $\epsilon_t = 0.5$ <span style="font-size: 14px;">(random):</span> $\alpha_t = 0$ <span style="font-size: 14px;">(this learner is ignored)</span>
- <span style="font-size: 14px;">If</span> $\epsilon_t > 0.5$<span style="font-size: 14px;">:</span> $\alpha_t < 0$ <span style="font-size: 14px;">(predictions are flipped)</span>

<span style="font-size: 14px;">In practice, we clip</span> $\epsilon_t$ <span style="font-size: 14px;">away from 0 to avoid numerical issues.</span>

---

## <span style="font-size: 16px;">Training Error Bound</span>

<span style="font-size: 14px;">AdaBoost has a remarkable theoretical guarantee. If each weak learner achieves weighted error</span> $\epsilon_t \leq 1/2 - \gamma$ <span style="font-size: 14px;">for some</span> $\gamma > 0$<span style="font-size: 14px;">:</span>

$$
\text{Training error of } H \leq \exp(-2\gamma^2 T)
$$

<span style="font-size: 14px;">This decays exponentially with the number of rounds</span> $T$<span style="font-size: 14px;">.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How does AdaBoost differ from gradient boosting?**</span>
  <span style="font-size: 14px;">A: AdaBoost reweights samples; gradient boosting fits new models to residuals (negative gradients). AdaBoost is a special case of gradient boosting with exponential loss.</span>

- <span style="font-size: 14px;">**Q: What happens with noisy data?**</span>
  <span style="font-size: 14px;">A: AdaBoost can overfit to noise because it keeps increasing weights on misclassified (possibly noisy) points. Regularization techniques include limiting $T$ or using a learning rate shrinkage.</span>

- <span style="font-size: 14px;">**Q: Can AdaBoost handle multiclass?**</span>
  <span style="font-size: 14px;">A: Yes, via SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss function), which generalizes the binary algorithm.</span>

---