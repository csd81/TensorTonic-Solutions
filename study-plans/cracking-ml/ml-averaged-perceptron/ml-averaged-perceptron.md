# <span style="font-size: 20px;">Averaged Perceptron</span>

<span style="font-size: 14px;">The perceptron is one of the oldest and simplest classification algorithms, dating back to Frank Rosenblatt's 1958 paper. It learns a linear decision boundary by making corrections whenever it misclassifies a training point. The averaged variant, introduced by Freund and Schapire, dramatically improves generalization by returning the average of all weight vectors visited during training rather than just the final one.</span>

---

## <span style="font-size: 16px;">The Basic Perceptron</span>

- <span style="font-size: 14px;">Maintains a weight vector</span> $w$ <span style="font-size: 14px;">and bias</span> $b$ <span style="font-size: 14px;">defining the hyperplane</span> $w \cdot x + b = 0$
- <span style="font-size: 14px;">Predicts</span> $+1$ <span style="font-size: 14px;">if</span> $w \cdot x + b > 0$<span style="font-size: 14px;">, else</span> $-1$
- <span style="font-size: 14px;">For each misclassified point (</span>$y_i(w \cdot x_i + b) \leq 0$<span style="font-size: 14px;">), updates:</span>

$$
w \leftarrow w + y_i \, x_i, \quad b \leftarrow b + y_i
$$

- <span style="font-size: 14px;">This pushes the decision boundary to correctly classify the misclassified point</span>
- <span style="font-size: 14px;">**Perceptron Convergence Theorem**: if the data is linearly separable, the perceptron converges in a finite number of updates</span>
- <span style="font-size: 14px;">If the data is not linearly separable, the perceptron will oscillate and never converge</span>

---

## <span style="font-size: 16px;">Why Average?</span>

- <span style="font-size: 14px;">The final weight vector of the basic perceptron is heavily influenced by the last few training points it saw</span>
- <span style="font-size: 14px;">It can overfit to recent corrections and perform poorly on unseen data</span>
- <span style="font-size: 14px;">Averaging smooths out these fluctuations, similar to how ensemble methods average multiple models</span>
- <span style="font-size: 14px;">The averaged perceptron achieves generalization bounds comparable to SVMs in practice</span>

---

## <span style="font-size: 16px;">Averaging Strategy</span>

<span style="font-size: 14px;">After processing each training point (whether or not an update occurs), add the current</span> $w$ <span style="font-size: 14px;">and</span> $b$ <span style="font-size: 14px;">to running sums. After training, divide by the total count</span> $T = n \times \text{epochs}$<span style="font-size: 14px;">:</span>

$$
\bar{w} = \frac{1}{T} \sum_{t=1}^{T} w_t, \quad \bar{b} = \frac{1}{T} \sum_{t=1}^{T} b_t
$$

<span style="font-size: 14px;">An efficient implementation uses a trick: instead of storing all weight vectors, maintain a counter</span> $c$ <span style="font-size: 14px;">of how many steps the current weight has survived unchanged, and accumulate</span> $c \cdot w$ <span style="font-size: 14px;">into the running sum when an update occurs.</span>

---

## <span style="font-size: 16px;">Online Learning</span>

- <span style="font-size: 14px;">The perceptron is an **online algorithm**: it processes one example at a time and updates immediately</span>
- <span style="font-size: 14px;">No matrix inversions, no gradient sums over the full dataset</span>
- <span style="font-size: 14px;">Update cost is</span> $O(d)$ <span style="font-size: 14px;">per example (just a vector addition)</span>
- <span style="font-size: 14px;">Memory efficient: only stores</span> $w$ <span style="font-size: 14px;">and the running sum</span>
- <span style="font-size: 14px;">Suitable for streaming data where examples arrive one at a time</span>

---

## <span style="font-size: 16px;">Connection to SVM</span>

- <span style="font-size: 14px;">Both the perceptron and SVM learn linear decision boundaries</span>
- <span style="font-size: 14px;">The perceptron finds any separating hyperplane; the SVM finds the maximum-margin one</span>
- <span style="font-size: 14px;">The margin</span> $y_i(w \cdot x_i + b)$ <span style="font-size: 14px;">in the perceptron update rule is the same quantity the SVM maximizes</span>
- <span style="font-size: 14px;">The averaged perceptron can be viewed as an approximation to the SVM solution</span>
- <span style="font-size: 14px;">In NLP, the averaged perceptron was historically the go-to classifier before deep learning (e.g., for POS tagging, named entity recognition)</span>

---

## <span style="font-size: 16px;">Margin and Separability</span>

- <span style="font-size: 14px;">The signed margin of point</span> $i$ <span style="font-size: 14px;">is</span> $y_i(w \cdot x_i + b)$
- <span style="font-size: 14px;">Positive margin means correct classification; negative means misclassification</span>
- <span style="font-size: 14px;">The perceptron only updates on points with non-positive margin (</span>$\leq 0$<span style="font-size: 14px;">)</span>
- <span style="font-size: 14px;">The geometric margin</span> $\gamma = y_i(w \cdot x_i + b) / \|w\|$ <span style="font-size: 14px;">measures the distance to the decision boundary</span>
- <span style="font-size: 14px;">The number of perceptron updates is bounded by</span> $R^2 / \gamma^2$<span style="font-size: 14px;">, where</span> $R$ <span style="font-size: 14px;">is the radius of the data and</span> $\gamma$ <span style="font-size: 14px;">is the margin of the best separating hyperplane</span>

---

## <span style="font-size: 16px;">Multiclass Extension</span>

- <span style="font-size: 14px;">Maintain a weight vector</span> $w_c$ <span style="font-size: 14px;">for each class</span> $c$
- <span style="font-size: 14px;">Predict</span> $\hat{y} = \arg\max_c \, w_c \cdot x$
- <span style="font-size: 14px;">If</span> $\hat{y} \neq y_i$<span style="font-size: 14px;">:</span> $w_{y_i} \leftarrow w_{y_i} + x_i$ <span style="font-size: 14px;">and</span> $w_{\hat{y}} \leftarrow w_{\hat{y}} - x_i$
- <span style="font-size: 14px;">This is the "one-vs-all" perceptron used extensively in structured prediction</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Perceptron vs logistic regression?**</span>
  <span style="font-size: 14px;">A: The perceptron makes hard predictions and uses a step-function loss (zero loss when correct, update when wrong). Logistic regression uses a smooth sigmoid and cross-entropy loss, producing probabilistic outputs. Logistic regression works on non-separable data; the perceptron does not converge without separability</span>

- <span style="font-size: 14px;">**Q: Why does averaging help?**</span>
  <span style="font-size: 14px;">A: The final perceptron weight vector is unstable, heavily influenced by the last few updates. Averaging acts like an implicit ensemble over all weight vectors seen during training, smoothing out noise and improving generalization. Theoretically, the averaged perceptron has tighter generalization bounds</span>

- <span style="font-size: 14px;">**Q: What if the data is not linearly separable?**</span>
  <span style="font-size: 14px;">A: The basic perceptron will cycle indefinitely without converging. The averaged perceptron still produces useful results because averaging dampens the oscillations. For guaranteed convergence on non-separable data, use the "pocket algorithm" (keep the best weight vector seen so far) or switch to logistic regression / SVM</span>

- <span style="font-size: 14px;">**Q: What is the learning rate in the perceptron?**</span>
  <span style="font-size: 14px;">A: The standard perceptron uses an implicit learning rate of 1. Some variants use $\eta$ to scale updates, but for the averaged perceptron, the averaging itself provides the necessary regularization</span>

- <span style="font-size: 14px;">**Q: Connection to neural networks?**</span>
  <span style="font-size: 14px;">A: The perceptron is a single-layer, single-neuron network with a step activation function. Stacking multiple perceptrons with non-linear activations gives multi-layer perceptrons (MLPs), which can learn non-linear boundaries</span>

---