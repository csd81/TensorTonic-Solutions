<span style="font-size: 14px;">Derivatives are the engine of deep learning. Every call to `loss.backward()` computes activation derivatives across every layer and every sample in the batch. Understanding the analytical form of these derivatives is essential for debugging gradients, designing architectures, and reasoning about training dynamics.</span>

## <span style="font-size: 14px;">What Is a Derivative?</span>

<span style="font-size: 14px;">The derivative of a function $f$ at a point $x$ measures the instantaneous rate of change of $f$ with respect to $x$:</span>

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

<span style="font-size: 14px;">Geometrically, $f'(x)$ is the slope of the tangent line to $f$ at $x$. In the context of neural networks, the derivative tells us how sensitive the output of a neuron is to small changes in its input.</span>

### <span style="font-size: 14px;">Key Differentiation Rules</span>

<span style="font-size: 14px;">Several differentiation rules are used repeatedly when computing activation derivatives:</span>

* <span style="font-size: 14px;">**Power rule**: $\frac{d}{dx} x^n = n x^{n-1}$</span>
* <span style="font-size: 14px;">**Sum rule**: $(f + g)' = f' + g'$</span>
* <span style="font-size: 14px;">**Product rule**: $(f \cdot g)' = f' \cdot g + f \cdot g'$</span>
* <span style="font-size: 14px;">**Quotient rule**: $(f/g)' = (f'g - fg') / g^2$</span>
* <span style="font-size: 14px;">**Chain rule**: $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$</span>
* <span style="font-size: 14px;">**Exponential**: $\frac{d}{dx} e^x = e^x$</span>

<span style="font-size: 14px;">The chain rule is particularly important because neural networks are compositions of functions, and backpropagation is simply the systematic application of the chain rule from output to input.</span>

## <span style="font-size: 14px;">The Sigmoid Function and Its Derivative</span>

<span style="font-size: 14px;">The sigmoid function is defined as:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### <span style="font-size: 14px;">Deriving the Sigmoid Derivative</span>

<span style="font-size: 14px;">We can compute $\sigma'(x)$ using the quotient rule. Let $u = 1$ and $v = 1 + e^{-x}$:</span>

$$
\begin{aligned}
\sigma'(x) &= \frac{u'v - uv'}{v^2} \\
&= \frac{0 \cdot (1 + e^{-x}) - 1 \cdot (-e^{-x})}{(1 + e^{-x})^2} \\
&= \frac{e^{-x}}{(1 + e^{-x})^2}
\end{aligned}
$$

<span style="font-size: 14px;">Now we use a clever algebraic trick. Notice that:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad 1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}
$$

<span style="font-size: 14px;">Multiplying these together:</span>

$$
\sigma(x)(1 - \sigma(x)) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \frac{e^{-x}}{(1 + e^{-x})^2}
$$

<span style="font-size: 14px;">This is exactly $\sigma'(x)$! Therefore:</span>

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

<span style="font-size: 14px;">This elegant self-referential form is one of the most famous results in neural network mathematics. It means you can compute the derivative using only the function value itself - no need to recompute exponentials.</span>

### <span style="font-size: 14px;">Properties of the Sigmoid Derivative</span>

* <span style="font-size: 14px;">**Maximum**: $\sigma'(x)$ achieves its maximum at $x = 0$, where $\sigma(0) = 1/2$, giving $\sigma'(0) = \sigma(0)(1 - \sigma(0)) = 1/4$</span>
* <span style="font-size: 14px;">**Symmetry**: $\sigma'(x) = \sigma'(-x)$ (the derivative is symmetric about $x = 0$)</span>
* <span style="font-size: 14px;">**Shape**: bell-shaped curve, peaked at $x = 0$, decaying to 0 as $|x| \to \infty$</span>
* <span style="font-size: 14px;">**Bound**: $0 < \sigma'(x) \leq 1/4$ for all $x$</span>

<span style="font-size: 14px;">The fact that $\sigma'(x) \leq 1/4$ is the root cause of the vanishing gradient problem. When backpropagating through $L$ sigmoid layers, the gradient is multiplied by $\sigma'$ at each layer, giving a maximum contribution of $(1/4)^L$ from the sigmoid derivatives alone.</span>

### <span style="font-size: 14px;">Alternative Derivation Using the Chain Rule</span>

<span style="font-size: 14px;">We can also derive $\sigma'$ using the chain rule. Write $\sigma(x) = (1 + e^{-x})^{-1}$. Let $u = 1 + e^{-x}$:</span>

$$
\sigma'(x) = -1 \cdot u^{-2} \cdot u' = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}
$$

<span style="font-size: 14px;">which simplifies to $\sigma(x)(1 - \sigma(x))$ as before.</span>

## <span style="font-size: 14px;">The Hyperbolic Tangent and Its Derivative</span>

<span style="font-size: 14px;">The hyperbolic tangent function is defined as:</span>

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<span style="font-size: 14px;">It is closely related to sigmoid: $\tanh(x) = 2\sigma(2x) - 1$. The range of tanh is $(-1, 1)$, centered at 0, while sigmoid maps to $(0, 1)$.</span>

### <span style="font-size: 14px;">Deriving the Tanh Derivative</span>

<span style="font-size: 14px;">Using the quotient rule with $u = e^x - e^{-x}$ and $v = e^x + e^{-x}$:</span>

$$
u' = e^x + e^{-x}, \quad v' = e^x - e^{-x}
$$

$$
\tanh'(x) = \frac{u'v - uv'}{v^2} = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}
$$

<span style="font-size: 14px;">Expanding the numerator:</span>

$$
(e^x + e^{-x})^2 - (e^x - e^{-x})^2 = (e^{2x} + 2 + e^{-2x}) - (e^{2x} - 2 + e^{-2x}) = 4
$$

<span style="font-size: 14px;">So:</span>

$$
\tanh'(x) = \frac{4}{(e^x + e^{-x})^2}
$$

<span style="font-size: 14px;">Now, since $\tanh^2(x) = \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2}$:</span>

$$
\begin{aligned}
1 - \tanh^2(x) &= 1 - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2} \\
&= \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} \\
&= \frac{4}{(e^x + e^{-x})^2}
\end{aligned}
$$

<span style="font-size: 14px;">Therefore:</span>

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

<span style="font-size: 14px;">Like sigmoid, the tanh derivative can be expressed purely in terms of the function value.</span>

### <span style="font-size: 14px;">Properties of the Tanh Derivative</span>

* <span style="font-size: 14px;">**Maximum**: $\tanh'(0) = 1 - \tanh^2(0) = 1$</span>
* <span style="font-size: 14px;">**Symmetry**: $\tanh'(x) = \tanh'(-x)$ (even function)</span>
* <span style="font-size: 14px;">**Bound**: $0 < \tanh'(x) \leq 1$ for all $x$</span>
* <span style="font-size: 14px;">**Decay**: $\tanh'(x) \to 0$ as $|x| \to \infty$</span>

<span style="font-size: 14px;">The maximum gradient of $1$ (vs. $1/4$ for sigmoid) makes tanh better for gradient flow. However, tanh still saturates for large $|x|$, causing vanishing gradients in deep networks.</span>

### <span style="font-size: 14px;">Comparison: Sigmoid vs. Tanh Gradients</span>

<span style="font-size: 14px;">Both sigmoid and tanh suffer from saturation, but tanh has a key advantage: its maximum derivative is $1$ (at $x = 0$), compared to $1/4$ for sigmoid. Through $L$ layers, the best-case sigmoid derivative contribution decays as $(1/4)^L$, while tanh's best case remains $1^L = 1$. Tanh was preferred over sigmoid in earlier network designs for this reason, before ReLU replaced both.</span>

## <span style="font-size: 14px;">ReLU and Its Derivative</span>

<span style="font-size: 14px;">The Rectified Linear Unit is defined as:</span>

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

### <span style="font-size: 14px;">The ReLU Derivative</span>

<span style="font-size: 14px;">For $x > 0$: $\text{ReLU}(x) = x$, so $\text{ReLU}'(x) = 1$.</span>

<span style="font-size: 14px;">For $x < 0$: $\text{ReLU}(x) = 0$, so $\text{ReLU}'(x) = 0$.</span>

<span style="font-size: 14px;">At $x = 0$: the derivative is technically undefined (left derivative is 0, right derivative is 1). In practice, frameworks use a **subgradient convention**, typically setting $\text{ReLU}'(0) = 0$.</span>

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

<span style="font-size: 14px;">This can be written compactly as $\text{ReLU}'(x) = \mathbf{1}[x > 0]$, where $\mathbf{1}[\cdot]$ is the indicator function.</span>

### <span style="font-size: 14px;">Why ReLU Revolutionized Deep Learning</span>

<span style="font-size: 14px;">ReLU solved the vanishing gradient problem because its derivative is either 0 or 1 - never a small fraction that compounds across layers. For neurons with positive pre-activation, the gradient passes through unchanged (multiplied by 1). This enables training of much deeper networks.</span>

<span style="font-size: 14px;">The key advantages of ReLU:</span>

* <span style="font-size: 14px;">**No saturation for positive inputs**: unlike sigmoid/tanh, the gradient does not decay for large positive $x$</span>
* <span style="font-size: 14px;">**Computational efficiency**: computing $\max(0, x)$ and its derivative is much cheaper than computing exponentials</span>
* <span style="font-size: 14px;">**Sparsity**: approximately 50% of neurons output 0, creating sparse representations</span>

<span style="font-size: 14px;">The disadvantage is the **dying ReLU problem**: if a neuron's pre-activation is always negative (due to large negative biases or unlucky initialization), its gradient is always 0, and the neuron never updates. It is effectively "dead".</span>

## <span style="font-size: 14px;">Swish (SiLU) and Its Derivative</span>

<span style="font-size: 14px;">The Swish activation function (also called SiLU: Sigmoid Linear Unit) is defined as:</span>

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

<span style="font-size: 14px;">where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function.</span>

### <span style="font-size: 14px;">Deriving the Swish Derivative Using the Product Rule</span>

<span style="font-size: 14px;">Since $\text{Swish}(x) = x \cdot \sigma(x)$ is a product of two functions, we apply the product rule:</span>

$$
\text{Swish}'(x) = \frac{d}{dx}[x] \cdot \sigma(x) + x \cdot \frac{d}{dx}[\sigma(x)]
$$

$$
= 1 \cdot \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

$$
= \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

<span style="font-size: 14px;">Factoring out $\sigma(x)$:</span>

$$
\text{Swish}'(x) = \sigma(x)\left[1 + x(1 - \sigma(x))\right]
$$

<span style="font-size: 14px;">Or equivalently:</span>

$$
\begin{aligned}
\text{Swish}'(x) &= \sigma(x) + x \cdot \sigma(x) - x \cdot \sigma(x)^2 \\
&= \text{Swish}(x) \cdot \frac{1}{x} + \text{Swish}(x) \cdot (1 - \sigma(x))
\end{aligned}
$$

<span style="font-size: 14px;">The most practical form for implementation is:</span>

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))
$$

### <span style="font-size: 14px;">Properties of the Swish Derivative</span>

* <span style="font-size: 14px;">**At $x = 0$**: $\text{Swish}'(0) = \sigma(0) + 0 = 1/2$</span>
* <span style="font-size: 14px;">**For large positive $x$**: $\sigma(x) \to 1$ and $\sigma(x)(1-\sigma(x)) \to 0$, so $\text{Swish}'(x) \to 1$</span>
* <span style="font-size: 14px;">**For large negative $x$**: $\sigma(x) \to 0$ and $x \cdot \sigma(x) \to 0$, so $\text{Swish}'(x) \to 0$</span>
* <span style="font-size: 14px;">**Non-monotonic**: $\text{Swish}'(x)$ can be slightly negative for moderately negative $x$, meaning Swish is not monotonically increasing</span>
* <span style="font-size: 14px;">**Smooth**: unlike ReLU, the Swish derivative is a continuous, smooth function everywhere</span>

### <span style="font-size: 14px;">Why Swish Outperforms ReLU</span>

<span style="font-size: 14px;">Swish addresses several limitations of ReLU:</span>

* <span style="font-size: 14px;">**Smoothness**: Swish and its derivative are infinitely differentiable, enabling better optimization with higher-order methods</span>
* <span style="font-size: 14px;">**Non-zero gradients everywhere**: unlike ReLU which has zero gradient for $x < 0$, Swish has small but nonzero gradients for negative inputs, preventing dead neurons</span>
* <span style="font-size: 14px;">**Self-gating**: the sigmoid factor $\sigma(x)$ acts as a soft gate, automatically determining how much of the input to pass through based on the input value itself</span>

<span style="font-size: 14px;">Swish (under the name SiLU) is used in many modern architectures including EfficientNet, GPT variants, and diffusion models.</span>

## <span style="font-size: 14px;">The Product Rule in Detail</span>

<span style="font-size: 14px;">The product rule is central to deriving the Swish derivative, so let us review it carefully.</span>

<span style="font-size: 14px;">If $h(x) = f(x) \cdot g(x)$, then:</span>

$$
h'(x) = f'(x) \cdot g(x) + f(x) \cdot g'(x)
$$

### <span style="font-size: 14px;">Proof of the Product Rule</span>

<span style="font-size: 14px;">Starting from the definition of the derivative:</span>

$$
h'(x) = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}
$$

<span style="font-size: 14px;">Adding and subtracting $f(x+h)g(x)$ in the numerator:</span>

$$
= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}
$$

$$
= \lim_{h \to 0} \left[ f(x+h) \cdot \frac{g(x+h) - g(x)}{h} + g(x) \cdot \frac{f(x+h) - f(x)}{h} \right]
$$

$$
= f(x) \cdot g'(x) + g(x) \cdot f'(x)
$$

<span style="font-size: 14px;">The product rule extends to products of more than two functions. For three functions:</span>

$$
(fgh)' = f'gh + fg'h + fgh'
$$

<span style="font-size: 14px;">This generalization is relevant for more complex activations and for the chain rule through multi-layer networks.</span>

## <span style="font-size: 14px;">Derivatives in Backpropagation</span>

<span style="font-size: 14px;">During backpropagation, the derivative of the activation function appears as a multiplicative factor in the gradient computation. For a single neuron with pre-activation $z = w^T x + b$ and activation $a = f(z)$:</span>

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot f'(z)
$$

<span style="font-size: 14px;">The gradient with respect to the pre-activation is the incoming gradient (from the layer above) multiplied by the local derivative of the activation function. This is the core operation of backpropagation at each neuron.</span>

### <span style="font-size: 14px;">Gradient Flow Through Layers</span>

<span style="font-size: 14px;">In a network with $L$ layers, the gradient at the first layer involves a product of all activation derivatives:</span>

$$
\frac{\partial L}{\partial w_1} \propto \prod_{l=1}^{L} f'(z_l)
$$

<span style="font-size: 14px;">The behavior of this product depends critically on the activation function:</span>

* <span style="font-size: 14px;">**Sigmoid**: $f'(z) \leq 1/4$, so the product decays as $(1/4)^L$</span>
* <span style="font-size: 14px;">**Tanh**: $f'(z) \leq 1$, but typically $f'(z) < 1$ except at $z = 0$</span>
* <span style="font-size: 14px;">**ReLU**: $f'(z) \in \{0, 1\}$, so the product is either 0 or 1</span>
* <span style="font-size: 14px;">**Swish**: $f'(z)$ is close to 1 for positive $z$, with smooth transition</span>

## <span style="font-size: 14px;">Verification via Finite Differences</span>

<span style="font-size: 14px;">Any analytical derivative can be verified numerically using the **central difference** approximation:</span>

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

<span style="font-size: 14px;">for a small step size $h$. The central difference has error $O(h^2)$, much better than the one-sided forward or backward differences which have error $O(h)$.</span>

<span style="font-size: 14px;">This technique, called **gradient checking**, is the standard way to verify that a backpropagation implementation is correct. PyTorch provides `torch.autograd.gradcheck` which uses this approach.</span>

## <span style="font-size: 14px;">Summary of Activation Derivatives</span>

<span style="font-size: 14px;">A summary of the four activation functions and their derivatives:</span>

* <span style="font-size: 14px;">**Sigmoid**: $\sigma(x) = 1/(1+e^{-x})$, derivative $\sigma'(x) = \sigma(x)(1-\sigma(x))$, max gradient $1/4$</span>
* <span style="font-size: 14px;">**Tanh**: $\tanh(x) = (e^x-e^{-x})/(e^x+e^{-x})$, derivative $\tanh'(x) = 1 - \tanh^2(x)$, max gradient $1$</span>
* <span style="font-size: 14px;">**ReLU**: $\text{ReLU}(x) = \max(0,x)$, derivative $\text{ReLU}'(x) = \mathbf{1}[x>0]$, gradient is $0$ or $1$</span>
* <span style="font-size: 14px;">**Swish**: $\text{Swish}(x) = x\sigma(x)$, derivative $\text{Swish}'(x) = \sigma(x) + x\sigma(x)(1-\sigma(x))$, asymptotes to $1$</span>

## <span style="font-size: 14px;">Vectorized Implementation</span>

<span style="font-size: 14px;">In practice, activation derivatives are applied element-wise to arrays of pre-activations. NumPy's vectorized operations make this efficient:</span>

* <span style="font-size: 14px;">`np.exp(-x)` computes the element-wise exponential</span>
* <span style="font-size: 14px;">`np.where(x > 0, 1.0, 0.0)` implements the ReLU derivative indicator</span>
* <span style="font-size: 14px;">`np.tanh(x)` computes the element-wise hyperbolic tangent</span>

<span style="font-size: 14px;">When implementing these derivatives, avoid recomputing values unnecessarily. For sigmoid, compute $\sigma(x)$ once and reuse it for $\sigma(x)(1-\sigma(x))$. For Swish, compute $\sigma(x)$ once and use it in the product rule expression.</span>

## <span style="font-size: 14px;">Beyond Single-Variable Derivatives</span>

<span style="font-size: 14px;">While this problem focuses on single-variable derivatives of activation functions, the same derivatives appear as components in multivariable calculus when computing Jacobians and gradients of neural network layers. The element-wise nature of activation functions means that the Jacobian of an activation layer is a diagonal matrix with the activation derivatives on the diagonal:</span>

$$
J_{f} = \text{diag}(f'(z_1), f'(z_2), \ldots, f'(z_n))
$$

<span style="font-size: 14px;">This diagonal structure is what makes backpropagation through activation layers computationally cheap - it is just an element-wise multiplication rather than a full matrix multiplication.</span>
