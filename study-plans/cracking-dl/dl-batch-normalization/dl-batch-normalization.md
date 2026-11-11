# <span style="font-size: 20px;">Batch Normalization</span>

<span style="font-size: 14px;">Batch normalization (Ioffe & Szegedy, 2015) transformed deep learning by making it possible to train much deeper networks with higher learning rates. It normalizes the input to each layer across the batch dimension, reducing the internal covariate shift problem and acting as a regularizer.</span>

---

## <span style="font-size: 16px;">Why Batch Normalization Works</span>

<span style="font-size: 14px;">Without normalization, as parameters update during training, the distribution of each layer's inputs changes. This forces subsequent layers to continuously adapt to a moving target (internal covariate shift). Batch normalization fixes this by ensuring that each layer always receives inputs with zero mean and unit variance.</span>

<span style="font-size: 14px;">Practical benefits:</span>
* <span style="font-size: 14px;">**Higher learning rates**: normalized activations are less prone to exploding or vanishing, so larger steps are safe</span>
* <span style="font-size: 14px;">**Faster convergence**: smoother loss landscape reduces the number of iterations needed</span>
* <span style="font-size: 14px;">**Regularization**: the batch statistics introduce noise (each sample's normalization depends on which other samples are in the batch), acting like a mild regularizer</span>
* <span style="font-size: 14px;">**Reduces sensitivity to initialization**: batch norm re-normalizes after each layer, so poor weight initialization is corrected automatically</span>

---

## <span style="font-size: 16px;">The Algorithm</span>

<span style="font-size: 14px;">For a mini-batch</span> $\{x_1, \dots, x_N\}$ <span style="font-size: 14px;">of</span> $D$<span style="font-size: 14px;">-dimensional vectors:</span>

<span style="font-size: 14px;">**Step 1 - Batch statistics:**</span>

$$
\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \quad \sigma^2_j = \frac{1}{N}\sum_{i=1}^{N}(x_{ij} - \mu_j)^2
$$

<span style="font-size: 14px;">**Step 2 - Normalize:**</span>

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma^2_j + \epsilon}}
$$

<span style="font-size: 14px;">**Step 3 - Scale and shift:**</span>

$$
y_{ij} = \gamma_j \cdot \hat{x}_{ij} + \beta_j
$$

<span style="font-size: 14px;">The learnable parameters</span> $\gamma$ <span style="font-size: 14px;">and</span> $\beta$ <span style="font-size: 14px;">allow the network to undo the normalization if that is optimal. If</span> $\gamma = \sigma$ <span style="font-size: 14px;">and</span> $\beta = \mu$<span style="font-size: 14px;">, the output equals the input - so batch norm can learn the identity.</span>

---

## <span style="font-size: 16px;">Training vs Inference</span>

<span style="font-size: 14px;">During training, batch statistics are computed from the current mini-batch. During inference, there may be only one sample (batch size 1), so batch statistics are meaningless. Instead, inference uses running averages accumulated during training:</span>

$$
\bar{\mu} \leftarrow (1 - m) \cdot \bar{\mu} + m \cdot \mu_{\text{batch}}
$$

<span style="font-size: 14px;">where</span> $m$ <span style="font-size: 14px;">is the momentum (typically 0.1). The running statistics are exponential moving averages of all batch statistics seen during training. This ensures deterministic inference - the same input always produces the same output regardless of what other inputs are in the batch.</span>

---

## <span style="font-size: 16px;">Backward Pass</span>

<span style="font-size: 14px;">The gradient through batch norm is non-trivial because every output depends on every input (through the mean and variance). Given upstream gradient</span> $\partial L / \partial y$<span style="font-size: 14px;">:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial \gamma} &= \sum_{i=1}^{N} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i \\[6pt]
\frac{\partial L}{\partial \beta} &= \sum_{i=1}^{N} \frac{\partial L}{\partial y_i}
\end{aligned}
$$

<span style="font-size: 14px;">For the input gradient, let</span> $d\hat{x} = \frac{\partial L}{\partial y} \cdot \gamma$<span style="font-size: 14px;">:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial x_i} = \frac{1}{N \cdot s} \bigl(&N \cdot d\hat{x}_i - \sum_j d\hat{x}_j \\
&- \hat{x}_i \sum_j d\hat{x}_j \cdot \hat{x}_j \bigr)
\end{aligned}
$$

<span style="font-size: 14px;">where</span> $s = \sqrt{\sigma^2 + \epsilon}$<span style="font-size: 14px;">. This formula accounts for the fact that the mean and variance depend on all inputs, creating inter-sample gradient flow.</span>

---

## <span style="font-size: 16px;">Placement and Variants</span>

* <span style="font-size: 14px;">**Pre-activation vs post-activation**: the original paper places BN before the activation (Conv -> BN -> ReLU). Some architectures place it after (Conv -> ReLU -> BN). Pre-activation is more common</span>
* <span style="font-size: 14px;">**Layer Normalization**: normalizes across features instead of across the batch. Used in transformers because it works with variable batch sizes and sequence lengths</span>
* <span style="font-size: 14px;">**Group Normalization**: normalizes across groups of channels. Used when batch size is too small for reliable batch statistics (e.g., object detection)</span>
* <span style="font-size: 14px;">**Instance Normalization**: normalizes each sample independently. Used in style transfer</span>
* <span style="font-size: 14px;">**RMSNorm**: a simplified variant that only normalizes by root mean square (no mean subtraction). Used in LLaMA and other recent LLMs</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why does batch norm behave differently during training and inference?** During training, batch statistics provide the normalization while also introducing beneficial noise. During inference, we want deterministic predictions, so we use fixed running statistics. If you used batch statistics at inference with batch_size=1, the output would always be zero (mean subtracted, divided by zero variance)</span>
* <span style="font-size: 14px;">**Why use the population variance (1/N) instead of the sample variance (1/(N-1))?** The original paper uses 1/N (population variance) for the batch statistics during training. PyTorch's BatchNorm uses 1/(N-1) (Bessel's correction) for the running variance update but 1/N for the normalization. This subtlety sometimes appears in interviews</span>
* <span style="font-size: 14px;">**When does batch norm fail?** With very small batch sizes (batch_size=1 or 2), batch statistics are too noisy to be useful. This is why object detection and video models often use Group Normalization or Synchronized Batch Normalization across GPUs</span>
* <span style="font-size: 14px;">**Why not just use Layer Normalization everywhere?** Layer Norm normalizes per-sample (across features), while Batch Norm normalizes per-feature (across samples). For CNNs, Batch Norm is better because features at the same spatial position across different images should have consistent statistics. For transformers, Layer Norm is preferred because sequence lengths vary and batch norm across tokens is not meaningful</span>
* <span style="font-size: 14px;">**What is the role of gamma and beta?** They allow the network to learn to undo the normalization if needed. Without them, the representational power of the layer would be limited to zero-mean, unit-variance outputs. With them, the network can learn any mean and variance, but starts from a normalized baseline</span>

---