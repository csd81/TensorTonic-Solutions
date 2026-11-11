# <span style="font-size: 20px;">Backpropagation (Manual)</span>

<span style="font-size: 14px;">Backpropagation is the algorithm that makes neural network training possible. It computes the gradient of the loss function with respect to every parameter in the network by applying the chain rule layer by layer from the output back to the input. This is the single most important algorithm in deep learning and the most frequently tested topic in DL interviews at top companies.</span>

---

## <span style="font-size: 16px;">The Chain Rule</span>

<span style="font-size: 14px;">The chain rule is the mathematical foundation of backpropagation. For a composition of functions</span> $L(f(g(x)))$<span style="font-size: 14px;">:</span>

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

<span style="font-size: 14px;">In a neural network, each layer is a function composition. The loss depends on the output, which depends on the last layer's weights, which depends on the previous layer's activations, and so on. The chain rule lets us decompose this dependency into local gradients at each layer and multiply them together.</span>

---

## <span style="font-size: 16px;">Deriving the Gradients</span>

<span style="font-size: 14px;">For MSE loss</span> $L = \frac{1}{2}\|a^{(L)} - y\|^2$<span style="font-size: 14px;">, the output gradient is:</span>

$$
\frac{\partial L}{\partial a^{(L)}} = a^{(L)} - y
$$

<span style="font-size: 14px;">Define the error signal</span> $\delta^{(l)}$ <span style="font-size: 14px;">as the gradient of the loss with respect to the pre-activation at layer</span> $l$<span style="font-size: 14px;">. For the output layer (identity activation):</span>

$$
\begin{aligned}
\delta^{(L)} &= \frac{\partial L}{\partial z^{(L)}} \\
&= \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \\
&= (a^{(L)} - y) \cdot 1
\end{aligned}
$$

<span style="font-size: 14px;">For hidden layers with ReLU:</span>

$$
\delta^{(l)} = \left((W^{(l+1)})^T \delta^{(l+1)}\right) \odot \mathbb{1}[z^{(l)} > 0]
$$

<span style="font-size: 14px;">where</span> $\odot$ <span style="font-size: 14px;">is element-wise multiplication and</span> $\mathbb{1}[z^{(l)} > 0]$ <span style="font-size: 14px;">is the ReLU derivative (1 where</span> $z > 0$<span style="font-size: 14px;">, 0 elsewhere).</span>

<span style="font-size: 14px;">The parameter gradients at each layer are:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T, \\
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
\end{aligned}
$$

---

## <span style="font-size: 16px;">The Algorithm Step by Step</span>

<span style="font-size: 14px;">1. **Forward pass**: compute and store</span> $z^{(l)}$ <span style="font-size: 14px;">and</span> $a^{(l)}$ <span style="font-size: 14px;">for every layer</span>
<span style="font-size: 14px;">2. **Initialize**: set</span> $\delta^{(L)} = a^{(L)} - y$
<span style="font-size: 14px;">3. **For</span> $l = L, L-1, \dots, 1$<span style="font-size: 14px;">:**</span>
<span style="font-size: 14px;">   a. Compute</span> $\partial L / \partial W^{(l)} = \delta^{(l)} (a^{(l-1)})^T$
<span style="font-size: 14px;">   b. Compute</span> $\partial L / \partial b^{(l)} = \delta^{(l)}$
<span style="font-size: 14px;">   c. If</span> $l > 1$<span style="font-size: 14px;">: propagate</span> $\delta^{(l-1)} = (W^{(l)})^T \delta^{(l)} \odot \text{ReLU}'(z^{(l-1)})$

<span style="font-size: 14px;">Notice that step 3c requires both</span> $W^{(l)}$ <span style="font-size: 14px;">(current weights) and</span> $z^{(l-1)}$ <span style="font-size: 14px;">(previous pre-activation). This is why the forward pass must store all intermediate values.</span>

---

## <span style="font-size: 16px;">Why Backpropagation Is Efficient</span>

<span style="font-size: 14px;">A naive approach would compute the gradient for each parameter independently by perturbing it and measuring the loss change (numerical differentiation). For</span> $P$ <span style="font-size: 14px;">parameters, this requires</span> $P$ <span style="font-size: 14px;">forward passes, giving</span> $O(P \cdot C_{\text{forward}})$ <span style="font-size: 14px;">total cost.</span>

<span style="font-size: 14px;">Backpropagation computes ALL parameter gradients in a single backward pass that costs roughly the same as one forward pass. The total cost is</span> $O(C_{\text{forward}})$<span style="font-size: 14px;">, independent of the number of parameters. This is because the chain rule decomposes into local operations at each layer, and each local gradient is reused for all parameters in that layer.</span>

<span style="font-size: 14px;">The tradeoff is memory: storing all intermediate activations requires</span> $O(\sum_l n_l)$ <span style="font-size: 14px;">memory. Gradient checkpointing trades some of this memory back for recomputation.</span>

---

## <span style="font-size: 16px;">Vanishing and Exploding Gradients</span>

<span style="font-size: 14px;">The gradient at early layers is a product of many terms:</span>

$$
\frac{\partial L}{\partial W^{(1)}} \propto \prod_{l=2}^{L} (W^{(l)})^T \cdot \text{diag}(f'(z^{(l-1)}))
$$

<span style="font-size: 14px;">If</span> $\|W^{(l)}\| < 1$ <span style="font-size: 14px;">for many layers, this product shrinks exponentially (vanishing gradients). If</span> $\|W^{(l)}\| > 1$<span style="font-size: 14px;">, it grows exponentially (exploding gradients).</span>

<span style="font-size: 14px;">**Solutions in practice:**</span>
* <span style="font-size: 14px;">**ReLU activation**: derivative is exactly 1 for positive inputs, avoiding the saturation region of sigmoid/tanh where the derivative approaches 0</span>
* <span style="font-size: 14px;">**Residual connections**: gradient flows through the skip connection unattenuated</span>
* <span style="font-size: 14px;">**Proper initialization** (Kaiming/He): sets initial weight variance so that activations and gradients maintain consistent magnitude across layers</span>
* <span style="font-size: 14px;">**Gradient clipping**: caps the gradient norm to prevent explosion</span>
* <span style="font-size: 14px;">**Batch/Layer normalization**: normalizes activations to prevent them from growing or shrinking across layers</span>

---

## <span style="font-size: 16px;">Numerical Gradient Checking</span>

<span style="font-size: 14px;">To verify a backpropagation implementation, compare the analytical gradient against a numerical approximation using the centered difference formula:</span>

$$
\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}
$$

<span style="font-size: 14px;">with</span> $\epsilon \approx 10^{-5}$<span style="font-size: 14px;">. The relative error between analytical and numerical gradients should be below</span> $10^{-5}$<span style="font-size: 14px;">:</span>

$$
\text{relative error} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, \epsilon)}
$$

<span style="font-size: 14px;">This is a standard debugging technique. In interviews, knowing how to gradient-check your implementation signals production-level rigor.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**What happens to the gradient when a ReLU neuron is dead (z <= 0)?** The gradient is exactly zero for that neuron and all parameters upstream of it. This means the neuron can never recover - it is permanently dead. This is the "dying ReLU" problem, addressed by LeakyReLU (small non-zero gradient for z < 0) or careful initialization</span>
* <span style="font-size: 14px;">**Why is the weight gradient an outer product?** Because</span> $z = Wa + b$ <span style="font-size: 14px;">is a linear function of</span> $W$<span style="font-size: 14px;">. The derivative</span> $\partial z_i / \partial W_{ij} = a_j$<span style="font-size: 14px;">, so</span> $\partial L / \partial W_{ij} = \delta_i \cdot a_j$<span style="font-size: 14px;">, which is exactly the</span> $(i, j)$ <span style="font-size: 14px;">entry of the outer product</span> $\delta \cdot a^T$
* <span style="font-size: 14px;">**How does backprop change for batch processing?** Each sample contributes a gradient, and the final gradient is the average over the batch. In matrix form,</span> $\partial L / \partial W^{(l)} = \frac{1}{B} \Delta^{(l)} (A^{(l-1)})^T$ <span style="font-size: 14px;">where</span> $B$ <span style="font-size: 14px;">is the batch size and columns represent different samples</span>
* <span style="font-size: 14px;">**What is the Jacobian and when does it matter?** The Jacobian</span> $J$ <span style="font-size: 14px;">of a vector-valued function is the matrix of all partial derivatives. For a layer mapping</span> $a^{(l-1)} \to a^{(l)}$<span style="font-size: 14px;">, the Jacobian has shape</span> $(n_l, n_{l-1})$<span style="font-size: 14px;">. Backprop multiplies by</span> $J^T$ <span style="font-size: 14px;">at each layer. When activations are element-wise (like ReLU), the Jacobian is diagonal, making the multiplication efficient</span>
* <span style="font-size: 14px;">**How does automatic differentiation differ from manual backprop?** Autograd systems (PyTorch, JAX) record a computational graph during the forward pass and automatically apply the chain rule in reverse. The math is identical to manual backprop, but the implementation generalizes to arbitrary computation graphs, not just sequential layers</span>

---