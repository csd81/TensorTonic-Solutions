# <span style="font-size: 20px;">Weight Initialization</span>

<span style="font-size: 14px;">Weight initialization determines the starting point for gradient-based optimization. A poorly initialized network can fail to train entirely: activations may saturate (vanishing gradients) or blow up (exploding gradients) before the first parameter update ever happens. Understanding initialization is fundamental to debugging training failures.</span>

---

## <span style="font-size: 16px;">Why Initialization Matters</span>

<span style="font-size: 14px;">Consider a deep network with</span> $L$ <span style="font-size: 14px;">layers and no activation functions:</span> $a^{(L)} = W^{(L)} W^{(L-1)} \cdots W^{(1)} x$<span style="font-size: 14px;">. If each</span> $W^{(l)}$ <span style="font-size: 14px;">has eigenvalues slightly greater than 1, the product grows exponentially. If slightly less than 1, it shrinks exponentially. With</span> $L = 50$ <span style="font-size: 14px;">layers, even a factor of 1.1 per layer gives</span> $1.1^{50} \approx 117$<span style="font-size: 14px;">, and 0.9 per layer gives</span> $0.9^{50} \approx 0.005$<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">The goal of proper initialization is to ensure that both forward-pass activations and backward-pass gradients maintain consistent magnitude across all layers.</span>

---

## <span style="font-size: 16px;">Variance Propagation Analysis</span>

<span style="font-size: 14px;">For a single layer</span> $z = Wx$ <span style="font-size: 14px;">(ignoring bias), each output element is:</span>

$$
z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} x_i
$$

<span style="font-size: 14px;">Assuming weights and inputs are independent, zero-mean:</span>

$$
\text{Var}(z_j) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

<span style="font-size: 14px;">To preserve variance (</span>$\text{Var}(z) = \text{Var}(x)$<span style="font-size: 14px;">), we need</span> $\text{Var}(w) = 1/n_{\text{in}}$<span style="font-size: 14px;">. This is the forward-pass constraint. The backward-pass constraint (preserving gradient variance) gives</span> $\text{Var}(w) = 1/n_{\text{out}}$<span style="font-size: 14px;">.</span>

---

## <span style="font-size: 16px;">Xavier (Glorot) Initialization</span>

<span style="font-size: 14px;">Xavier initialization compromises between forward and backward constraints:</span>

$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

* <span style="font-size: 14px;">**Normal variant**:</span> $W \sim \mathcal{N}(0, \sigma)$ <span style="font-size: 14px;">where</span> $\sigma = \sqrt{2/(n_{\text{in}} + n_{\text{out}})}$
* <span style="font-size: 14px;">**Uniform variant**:</span> $W \sim \mathcal{U}(-a, a)$ <span style="font-size: 14px;">where</span> $a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$

<span style="font-size: 14px;">The uniform bound comes from</span> $\text{Var}(\mathcal{U}(-a, a)) = a^2/3$<span style="font-size: 14px;">, so</span> $a^2/3 = 2/(n_{\text{in}} + n_{\text{out}})$ <span style="font-size: 14px;">gives</span> $a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">Xavier initialization assumes the activation function is approximately linear around zero. This holds for sigmoid and tanh (their derivatives near zero are close to 1), but fails for ReLU.</span>

---

## <span style="font-size: 16px;">Kaiming (He) Initialization</span>

<span style="font-size: 14px;">ReLU zeros out roughly half the activations, so the effective variance is halved at each layer. Kaiming initialization compensates by doubling the weight variance:</span>

$$
\text{Var}(w) = \frac{2}{n_{\text{in}}}
$$

<span style="font-size: 14px;">The factor of 2 comes from the identity</span> $E[\text{ReLU}(x)^2] = \text{Var}(x)/2$ <span style="font-size: 14px;">when</span> $x \sim \mathcal{N}(0, \sigma^2)$<span style="font-size: 14px;">.</span>

* <span style="font-size: 14px;">**Normal**:</span> $\sigma = \sqrt{2/n_{\text{in}}}$
* <span style="font-size: 14px;">**Uniform**:</span> $a = \sqrt{6/n_{\text{in}}}$

<span style="font-size: 14px;">Kaiming initialization only considers the forward pass (</span>$n_{\text{in}}$<span style="font-size: 14px;">). A backward-pass variant using</span> $n_{\text{out}}$ <span style="font-size: 14px;">exists but is rarely used in practice.</span>

---

## <span style="font-size: 16px;">Why Random Normal Fails</span>

<span style="font-size: 14px;">With</span> $W \sim \mathcal{N}(0, 1)$ <span style="font-size: 14px;">and</span> $n_{\text{in}} = 256$<span style="font-size: 14px;">:</span>

$$
\text{Var}(z_j) = 256 \cdot 1 \cdot \text{Var}(x) = 256 \cdot \text{Var}(x)
$$

<span style="font-size: 14px;">Each layer amplifies the variance by the fan-in. After 3 layers of width 256, the variance is multiplied by</span> $256^3 \approx 1.7 \times 10^7$<span style="font-size: 14px;">. With sigmoid or tanh, the activations saturate at their extremes (0/1 or -1/1), gradients approach zero, and learning stops. With ReLU, activations grow without bound, eventually causing numerical overflow.</span>

<span style="font-size: 14px;">Using</span> $\mathcal{N}(0, 0.01)$ <span style="font-size: 14px;">overcorrects: variance shrinks by</span> $0.0001 \cdot n_{\text{in}}$ <span style="font-size: 14px;">per layer, causing activations to collapse to zero. Xavier and Kaiming find the exact variance that preserves signal magnitude.</span>

---

## <span style="font-size: 16px;">Practical Guidelines</span>

* <span style="font-size: 14px;">**ReLU/LeakyReLU networks**: use Kaiming initialization (the PyTorch default for Conv2d and Linear layers)</span>
* <span style="font-size: 14px;">**Sigmoid/Tanh networks**: use Xavier initialization</span>
* <span style="font-size: 14px;">**Transformers**: Xavier for attention projections, Kaiming for FFN layers, scaled initialization for residual paths (divide by</span> $\sqrt{2L}$ <span style="font-size: 14px;">where</span> $L$ <span style="font-size: 14px;">is the number of layers)</span>
* <span style="font-size: 14px;">**Batch-normalized networks**: initialization matters less because BN re-normalizes activations at each layer. However, proper init still helps early training</span>
* <span style="font-size: 14px;">**Residual networks**: the skip connection adds the input back, so activations can grow as</span> $\sqrt{L}$<span style="font-size: 14px;">. Some architectures (GPT-2, T5) scale the residual branch by</span> $1/\sqrt{L}$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why does Kaiming use 2/fan_in but Xavier uses 2/(fan_in + fan_out)?** Xavier balances forward and backward variance preservation equally. Kaiming prioritizes forward-pass stability and adds the factor of 2 to compensate for ReLU's half-activation. For symmetric activations (tanh), the forward and backward constraints are equally important, so the harmonic mean is justified</span>
* <span style="font-size: 14px;">**Does initialization matter with batch normalization?** Less, but still yes. BN normalizes activations at each layer, so the steady-state is less sensitive to init. However, the first few gradient steps before BN statistics stabilize can diverge with very bad initialization. Empirically, networks with BN still train faster with proper init</span>
* <span style="font-size: 14px;">**How do you initialize the bias?** Biases are almost always initialized to zero. The weights break symmetry between neurons; biases do not need to. Exception: LSTM forget gates are often initialized with bias = 1 to encourage remembering early in training</span>
* <span style="font-size: 14px;">**What about residual networks?** With standard Kaiming init, the variance doubles at each residual block (signal + residual). Two solutions: (1) scale the residual branch by</span> $1/\sqrt{L}$ <span style="font-size: 14px;">(GPT-2 approach), or (2) initialize the last layer of each residual block to zero (Fixup initialization), so the network starts as an identity function</span>
* <span style="font-size: 14px;">**What happens if all weights are initialized to the same value?** All neurons in a layer compute the same output and receive the same gradient. They remain identical forever - this is the symmetry problem. Randomness in initialization is essential to break this symmetry</span>

---