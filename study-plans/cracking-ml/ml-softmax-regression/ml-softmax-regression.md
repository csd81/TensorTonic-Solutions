# <span style="font-size: 20px;">Softmax Regression</span>

<span style="font-size: 14px;">Softmax regression is the natural extension of logistic regression from 2 classes to</span> $K$ <span style="font-size: 14px;">classes. It is the output layer of virtually every classification neural network and appears frequently in ML interviews as a standalone coding problem. Understanding softmax, cross-entropy, and the numerical stability tricks is essential.</span>

---

## <span style="font-size: 16px;">The Model</span>

<span style="font-size: 14px;">For</span> $n$ <span style="font-size: 14px;">samples with</span> $d$ <span style="font-size: 14px;">features and</span> $K$ <span style="font-size: 14px;">classes:</span>

$$
Z = XW + \mathbf{1} b^T \in \mathbb{R}^{n \times K}
$$

* $W \in \mathbb{R}^{d \times K}$ <span style="font-size: 14px;">is the weight matrix (one column per class)</span>
* $b \in \mathbb{R}^K$ <span style="font-size: 14px;">is the bias vector</span>
* $Z_{ik}$ <span style="font-size: 14px;">is the logit (unnormalized score) for sample</span> $i$ <span style="font-size: 14px;">and class</span> $k$

---

## <span style="font-size: 16px;">Softmax Function</span>

<span style="font-size: 14px;">The softmax converts logits to a valid probability distribution over classes:</span>

$$
P(y = k \mid x_i) = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}
$$

* <span style="font-size: 14px;">Output is always positive and sums to 1 across classes for each sample</span>
* <span style="font-size: 14px;">Amplifies the largest logit: the class with highest score gets most of the probability mass</span>
* <span style="font-size: 14px;">Reduces to the sigmoid when</span> $K = 2$

### <span style="font-size: 16px;">Numerical Stability</span>

<span style="font-size: 14px;">Computing</span> $e^{z_{ik}}$ <span style="font-size: 14px;">directly overflows for large logits. The standard trick is to subtract the row-wise maximum:</span>

$$
P(y = k \mid x_i) = \frac{e^{z_{ik} - m_i}}{\sum_{j=1}^{K} e^{z_{ij} - m_i}}, \quad m_i = \max_j z_{ij}
$$

<span style="font-size: 14px;">This does not change the result (the</span> $e^{-m_i}$ <span style="font-size: 14px;">cancels in numerator and denominator) but prevents overflow since the largest exponent is now 0.</span>

---

## <span style="font-size: 16px;">Cross-Entropy Loss</span>

<span style="font-size: 14px;">The loss measures how well the predicted distribution matches the true labels:</span>

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \log P(y = y_i \mid x_i)
$$

<span style="font-size: 14px;">This is equivalent to the negative log-likelihood under a categorical distribution. Using one-hot encoded labels</span> $Y \in \{0, 1\}^{n \times K}$<span style="font-size: 14px;">:</span>

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} Y_{ik} \log P_{ik}
$$

* <span style="font-size: 14px;">Only the probability assigned to the correct class contributes to the loss for each sample</span>
* <span style="font-size: 14px;">Confident correct predictions have near-zero loss; confident wrong predictions have very high loss</span>

---

## <span style="font-size: 16px;">Gradients</span>

<span style="font-size: 14px;">The gradients have the same elegant form as logistic regression:</span>

$$
\frac{\partial L}{\partial W} = \frac{1}{n} X^T (P - Y)
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \mathbf{1}^T (P - Y)
$$

<span style="font-size: 14px;">where</span> $P \in \mathbb{R}^{n \times K}$ <span style="font-size: 14px;">is the softmax output matrix and</span> $Y$ <span style="font-size: 14px;">is the one-hot label matrix. The error term</span> $(P - Y)$ <span style="font-size: 14px;">is the difference between predicted and true distributions.</span>

<span style="font-size: 14px;">This clean form arises because softmax + cross-entropy is a member of the exponential family, just like sigmoid + binary cross-entropy.</span>

---

## <span style="font-size: 16px;">Connection to Logistic Regression</span>

* <span style="font-size: 14px;">When</span> $K = 2$<span style="font-size: 14px;">, softmax regression is exactly equivalent to logistic regression</span>
* <span style="font-size: 14px;">The softmax reduces to the sigmoid:</span> $\frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \sigma(z_1 - z_0)$
* <span style="font-size: 14px;">The weight matrix</span> $W$ <span style="font-size: 14px;">generalizes the weight vector</span> $w$<span style="font-size: 14px;">; each class gets its own set of weights</span>
* <span style="font-size: 14px;">Cross-entropy generalizes binary cross-entropy to</span> $K$ <span style="font-size: 14px;">classes</span>

---

## <span style="font-size: 16px;">Overparameterization</span>

<span style="font-size: 14px;">Softmax regression has a redundant degree of freedom: adding the same vector to all columns of</span> $W$ <span style="font-size: 14px;">does not change the output. This means the model is overparameterized by one class. In practice this is harmless (gradient descent still converges), but it means:</span>

* <span style="font-size: 14px;">The solution is not unique (infinitely many equivalent weight matrices)</span>
* <span style="font-size: 14px;">Regularization (L2) breaks the symmetry and makes the solution unique</span>
* <span style="font-size: 14px;">Some implementations fix one class's weights to zero, reducing to</span> $K-1$ <span style="font-size: 14px;">independent weight vectors</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Q: Why not K separate binary classifiers?**</span>
  <span style="font-size: 14px;">A: Softmax produces calibrated probabilities that sum to 1; one-vs-rest does not guarantee this</span>

* <span style="font-size: 14px;">**Q: Numerical stability?**</span>
  <span style="font-size: 14px;">A: Subtract row max before exponentiating to prevent overflow; clip log input to prevent $\log(0)$</span>

* <span style="font-size: 14px;">**Q: How does temperature scaling work?**</span>
  <span style="font-size: 14px;">A: Divide logits by temperature $T$ before softmax: $T > 1$ makes probabilities softer, $T < 1$ makes them sharper</span>

* <span style="font-size: 14px;">**Q: Relation to neural networks?**</span>
  <span style="font-size: 14px;">A: The output layer of a classification network is softmax regression applied to the last hidden layer's features</span>

* <span style="font-size: 14px;">**Q: When does it fail?**</span>
  <span style="font-size: 14px;">A: When classes are not linearly separable; use deeper models or kernel methods</span>

* <span style="font-size: 14px;">**Q: Label smoothing?**</span>
  <span style="font-size: 14px;">A: Replace one-hot targets with $(1 - \epsilon)$ for the correct class and $\epsilon / (K-1)$ for others; prevents overconfident predictions</span>

---