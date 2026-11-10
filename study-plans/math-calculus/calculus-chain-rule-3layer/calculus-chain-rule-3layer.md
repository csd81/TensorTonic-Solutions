<span style="font-size: 14px;">The chain rule is the single most important concept in deep learning. Backpropagation - the algorithm that trains every neural network - is nothing more than the systematic application of the chain rule from the output layer back to the input. If you understand the chain rule applied to a 3-layer scalar network, you understand the core of backpropagation.</span>

## <span style="font-size: 14px;">The Chain Rule: Single Variable</span>

<span style="font-size: 14px;">If $y = f(g(x))$, then the derivative of $y$ with respect to $x$ is:</span>

$$
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

<span style="font-size: 14px;">In Leibniz notation, if $y = f(u)$ and $u = g(x)$:</span>

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

<span style="font-size: 14px;">The intuition is that small changes propagate multiplicatively through compositions. If $u$ changes by a factor of $g'(x)$ per unit change in $x$, and $y$ changes by a factor of $f'(u)$ per unit change in $u$, then $y$ changes by $f'(u) \cdot g'(x)$ per unit change in $x$.</span>

### <span style="font-size: 14px;">Proof of the Chain Rule</span>

<span style="font-size: 14px;">Starting from the definition of the derivative:</span>

$$
\frac{dy}{dx} = \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h}
$$

<span style="font-size: 14px;">Let $\Delta u = g(x+h) - g(x)$. For small $h$, $\Delta u \approx g'(x) \cdot h$. Multiplying and dividing by $\Delta u$ (when $\Delta u \neq 0$):</span>

$$
\frac{dy}{dx} = \lim_{h \to 0} \frac{f(g(x) + \Delta u) - f(g(x))}{\Delta u} \cdot \frac{\Delta u}{h}
$$

<span style="font-size: 14px;">As $h \to 0$, $\Delta u \to 0$ as well (since $g$ is continuous), so:</span>

$$
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

### <span style="font-size: 14px;">Extended Chain Rule</span>

<span style="font-size: 14px;">For a composition of three functions $y = f(g(h(x)))$:</span>

$$
\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)
$$

<span style="font-size: 14px;">And for $n$ functions $y = f_n \circ f_{n-1} \circ \cdots \circ f_1(x)$:</span>

$$
\frac{dy}{dx} = \prod_{i=1}^{n} f_i'\left(f_{i-1} \circ \cdots \circ f_1(x)\right)
$$

<span style="font-size: 14px;">where we evaluate each derivative at the appropriate intermediate value. This telescoping product is exactly what backpropagation computes.</span>

## <span style="font-size: 14px;">A 3-Layer Scalar Network</span>

<span style="font-size: 14px;">Consider the simplest possible "deep" network: three layers, each with a single scalar weight, using sigmoid activation:</span>

$$
y = \sigma(w_3 \cdot \sigma(w_2 \cdot \sigma(w_1 \cdot x)))
$$

<span style="font-size: 14px;">where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function and $w_1, w_2, w_3, x$ are all scalars.</span>

### <span style="font-size: 14px;">Forward Pass: Computing Intermediate Values</span>

<span style="font-size: 14px;">To apply the chain rule, we first define intermediate variables for each layer:</span>

<span style="font-size: 14px;">**Layer 1:**</span>

$$
z_1 = w_1 \cdot x \quad \text{(pre-activation)}
$$

$$
a_1 = \sigma(z_1) \quad \text{(activation)}
$$

<span style="font-size: 14px;">**Layer 2:**</span>

$$
z_2 = w_2 \cdot a_1 \quad \text{(pre-activation)}
$$

$$
a_2 = \sigma(z_2) \quad \text{(activation)}
$$

<span style="font-size: 14px;">**Layer 3 (output):**</span>

$$
z_3 = w_3 \cdot a_2 \quad \text{(pre-activation)}
$$

$$
y = \sigma(z_3) \quad \text{(output)}
$$

<span style="font-size: 14px;">This decomposition into pairs of linear (pre-activation) and nonlinear (activation) steps is the fundamental structure of every neural network layer.</span>

## <span style="font-size: 14px;">Applying the Chain Rule: $\partial y / \partial w_1$</span>

<span style="font-size: 14px;">To compute how the output $y$ changes with respect to the first-layer weight $w_1$, we trace the dependency path backwards through the network:</span>

$$
y \leftarrow z_3 \leftarrow a_2 \leftarrow z_2 \leftarrow a_1 \leftarrow z_1 \leftarrow w_1
$$

<span style="font-size: 14px;">Each arrow represents a functional dependency. The chain rule gives:</span>

$$
\begin{aligned}
\frac{\partial y}{\partial w_1} &= \frac{\partial y}{\partial z_3} \cdot \frac{\partial z_3}{\partial a_2} \\
&\phantom{=} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}
\end{aligned}
$$

<span style="font-size: 14px;">Now we compute each factor:</span>

### <span style="font-size: 14px;">Factor 1: $\partial y / \partial z_3$</span>

<span style="font-size: 14px;">Since $y = \sigma(z_3)$, and $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:</span>

$$
\frac{\partial y}{\partial z_3} = \sigma'(z_3) = y(1 - y)
$$

### <span style="font-size: 14px;">Factor 2: $\partial z_3 / \partial a_2$</span>

<span style="font-size: 14px;">Since $z_3 = w_3 \cdot a_2$:</span>

$$
\frac{\partial z_3}{\partial a_2} = w_3
$$

### <span style="font-size: 14px;">Factor 3: $\partial a_2 / \partial z_2$</span>

<span style="font-size: 14px;">Since $a_2 = \sigma(z_2)$:</span>

$$
\frac{\partial a_2}{\partial z_2} = \sigma'(z_2) = a_2(1 - a_2)
$$

### <span style="font-size: 14px;">Factor 4: $\partial z_2 / \partial a_1$</span>

<span style="font-size: 14px;">Since $z_2 = w_2 \cdot a_1$:</span>

$$
\frac{\partial z_2}{\partial a_1} = w_2
$$

### <span style="font-size: 14px;">Factor 5: $\partial a_1 / \partial z_1$</span>

<span style="font-size: 14px;">Since $a_1 = \sigma(z_1)$:</span>

$$
\frac{\partial a_1}{\partial z_1} = \sigma'(z_1) = a_1(1 - a_1)
$$

### <span style="font-size: 14px;">Factor 6: $\partial z_1 / \partial w_1$</span>

<span style="font-size: 14px;">Since $z_1 = w_1 \cdot x$:</span>

$$
\frac{\partial z_1}{\partial w_1} = x
$$

### <span style="font-size: 14px;">The Complete Gradient</span>

<span style="font-size: 14px;">Multiplying all six factors:</span>

$$
\frac{\partial y}{\partial w_1} = \sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2) \cdot w_2 \cdot \sigma'(z_1) \cdot x
$$

<span style="font-size: 14px;">Or equivalently:</span>

$$
\frac{\partial y}{\partial w_1} = x \cdot \prod_{l=1}^{3} \sigma'(z_l) \cdot \prod_{l=2}^{3} w_l
$$

<span style="font-size: 14px;">This formula reveals the structure of backpropagation: the gradient is a product of local derivatives (the $\sigma'(z_l)$ terms) and weight multipliers (the $w_l$ terms for upstream layers), starting from the input $x$.</span>

## <span style="font-size: 14px;">Why This IS Backpropagation</span>

<span style="font-size: 14px;">Backpropagation computes exactly this chain of products, but in an efficient order. Starting from the output:</span>

<span style="font-size: 14px;">**Step 1**: compute $\delta_3 = \sigma'(z_3)$ (the "error signal" at layer 3)</span>

<span style="font-size: 14px;">**Step 2**: propagate back: $\delta_2 = \delta_3 \cdot w_3 \cdot \sigma'(z_2)$ (multiply by weight and local derivative)</span>

<span style="font-size: 14px;">**Step 3**: propagate back: $\delta_1 = \delta_2 \cdot w_2 \cdot \sigma'(z_1)$</span>

<span style="font-size: 14px;">**Gradient**: $\partial y / \partial w_1 = \delta_1 \cdot x$</span>

<span style="font-size: 14px;">Each step multiplies the accumulated error signal by the local weight and the local activation derivative. This is the backward pass: it starts at the output and works back to the input, accumulating the chain rule product one factor at a time.</span>

### <span style="font-size: 14px;">Efficiency of Backpropagation</span>

<span style="font-size: 14px;">The key insight of backpropagation is that intermediate products are reused. When computing gradients for $w_2$ and $w_3$:</span>

$$
\frac{\partial y}{\partial w_2} = \sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2) \cdot a_1
$$

$$
\frac{\partial y}{\partial w_3} = \sigma'(z_3) \cdot a_2
$$

<span style="font-size: 14px;">The factor $\sigma'(z_3)$ (computed once) is shared across all three gradients. The factor $\sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2)$ (computed for $w_1$'s gradient) is also needed for $w_2$'s gradient. By computing the chain rule products from output to input and caching intermediate results, backpropagation computes all gradients in a single backward pass with the same computational cost as one forward pass.</span>

## <span style="font-size: 14px;">Vanishing Gradients in the Chain Rule</span>

<span style="font-size: 14px;">The gradient formula reveals why deep sigmoid networks are hard to train:</span>

$$
\frac{\partial y}{\partial w_1} = \sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2) \cdot w_2 \cdot \sigma'(z_1) \cdot x
$$

<span style="font-size: 14px;">The sigmoid derivative satisfies $\sigma'(z) \leq \frac{1}{4}$ for all $z$, with equality only at $z = 0$. Therefore, the product of $n$ sigmoid derivatives is bounded by:</span>

$$
\prod_{l=1}^{n} \sigma'(z_l) \leq \left(\frac{1}{4}\right)^n
$$

<span style="font-size: 14px;">This bound shrinks exponentially with depth $n$. Even in the best case (all pre-activations at $z = 0$), the gradient through $n$ sigmoid layers is at most $\left(\frac{1}{4}\right)^n$ times $|\prod_l w_l \cdot x|$. As $n$ grows, this factor decays to zero exponentially fast, meaning gradients for early-layer weights become vanishingly small relative to those for later layers.</span>

<span style="font-size: 14px;">This exponential decay of gradients with depth is the **vanishing gradient problem**. It is not a bug in backpropagation - it is an inherent mathematical consequence of the chain rule applied to saturating activation functions.</span>

## <span style="font-size: 14px;">Finite Difference Verification</span>

<span style="font-size: 14px;">Any analytical gradient can be verified numerically using finite differences. The **central difference** approximation for the gradient of a scalar function $f$ with respect to parameter $\theta$ is:</span>

$$
\frac{\partial f}{\partial \theta} \approx \frac{f(\theta + h) - f(\theta - h)}{2h}
$$

<span style="font-size: 14px;">for a suitably small step size $h$.</span>

### <span style="font-size: 14px;">Applying to Our Network</span>

<span style="font-size: 14px;">To verify $\partial y / \partial w_1$, we compute:</span>

$$
\frac{\partial y}{\partial w_1} \approx \frac{f(w_1 + h) - f(w_1 - h)}{2h}
$$

<span style="font-size: 14px;">where $f(w_1) = \sigma(w_3 \cdot \sigma(w_2 \cdot \sigma(w_1 \cdot x)))$ is the full network output as a function of $w_1$ (with $w_2, w_3, x$ held constant).</span>

<span style="font-size: 14px;">This requires two forward passes (one with $w_1 + h$ and one with $w_1 - h$), which is far more expensive than backpropagation for networks with many parameters. However, it serves as an invaluable correctness check during development.</span>

### <span style="font-size: 14px;">Error Analysis</span>

<span style="font-size: 14px;">The central difference has an error of order $O(h^2)$:</span>

$$
\frac{f(\theta + h) - f(\theta - h)}{2h} = f'(\theta) + \frac{h^2}{6} f'''(\theta) + O(h^4)
$$

<span style="font-size: 14px;">So the approximation improves quadratically as $h$ decreases. However, if $h$ is too small, floating-point cancellation errors dominate because $f(\theta + h)$ and $f(\theta - h)$ are nearly identical, and their difference loses significant digits. The optimal $h$ balances truncation error ($O(h^2)$) against cancellation error, which grows as $h$ shrinks.</span>

## <span style="font-size: 14px;">The Chain Rule for Other Parameters</span>

<span style="font-size: 14px;">The same chain rule logic applies to $w_2$ and $w_3$, with shorter chains:</span>

### <span style="font-size: 14px;">Gradient w.r.t. $w_2$</span>

$$
\begin{aligned}
\frac{\partial y}{\partial w_2} &= \frac{\partial y}{\partial z_3} \cdot \frac{\partial z_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2} \\
&= \sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2) \cdot a_1
\end{aligned}
$$

<span style="font-size: 14px;">This chain has only 4 factors (two sigmoid derivatives, one weight, one activation).</span>

### <span style="font-size: 14px;">Gradient w.r.t. $w_3$</span>

$$
\begin{aligned}
\frac{\partial y}{\partial w_3} &= \frac{\partial y}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_3} \\
&= \sigma'(z_3) \cdot a_2
\end{aligned}
$$

<span style="font-size: 14px;">This chain has only 2 factors. The gradient for the last layer is always the simplest, which is one reason why output layers train fastest.</span>

### <span style="font-size: 14px;">Gradient w.r.t. $x$</span>

$$
\frac{\partial y}{\partial x} = \sigma'(z_3) \cdot w_3 \cdot \sigma'(z_2) \cdot w_2 \cdot \sigma'(z_1) \cdot w_1
$$

<span style="font-size: 14px;">This is identical to $\partial y / \partial w_1$ except the last factor is $w_1$ instead of $x$.</span>

## <span style="font-size: 14px;">Connection to General Backpropagation</span>

<span style="font-size: 14px;">In a general neural network with layer operations $f_l$ (which include both the linear transformation and the activation):</span>

$$
y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
$$

<span style="font-size: 14px;">The gradient with respect to the parameters $\theta_l$ of layer $l$ is:</span>

$$
\begin{aligned}
\frac{\partial y}{\partial \theta_l} &= \left(\prod_{k=l+1}^{L} \frac{\partial f_k}{\partial f_{k-1}}\right) \cdot \frac{\partial f_l}{\partial \theta_l}
\end{aligned}
$$

<span style="font-size: 14px;">The term in parentheses is the product of Jacobians from layer $l+1$ to the output. In the scalar case, Jacobians reduce to ordinary derivatives, and this product becomes the product of scalars we computed above.</span>

### <span style="font-size: 14px;">Forward Mode vs. Reverse Mode</span>

<span style="font-size: 14px;">There are two ways to compute the chain rule product:</span>

* <span style="font-size: 14px;">**Forward mode**: compute from input to output. Start with $\partial z_1/\partial w_1 = x$, then accumulate factors left to right.</span>
* <span style="font-size: 14px;">**Reverse mode**: compute from output to input. Start with $\partial y/\partial z_3 = \sigma'(z_3)$, then accumulate factors right to left.</span>

<span style="font-size: 14px;">Both give the same answer. Reverse mode (backpropagation) is preferred because it computes gradients for ALL parameters in one pass, while forward mode requires a separate pass for each parameter.</span>

## <span style="font-size: 14px;">Numerical Stability Considerations</span>

<span style="font-size: 14px;">When computing the chain rule product numerically, several issues can arise:</span>

* <span style="font-size: 14px;">**Underflow**: if many sigmoid derivatives are very small (saturated neurons), their product may underflow to 0 in floating-point arithmetic</span>
* <span style="font-size: 14px;">**Precision loss**: multiplying many numbers less than 1 accumulates relative error</span>
* <span style="font-size: 14px;">**Log-space computation**: for very deep networks, it is sometimes better to work in log-space, computing $\log|\partial y / \partial w_1| = \sum_l \log|\text{factor}_l|$</span>

<span style="font-size: 14px;">In practice, modern architectures avoid these issues through architectural choices: ReLU activations (derivatives are 0 or 1), residual connections (additive gradient paths), and layer normalization (preventing pre-activations from saturating).</span>

## <span style="font-size: 14px;">The Importance of Intermediate Values</span>

<span style="font-size: 14px;">A critical implementation detail of backpropagation is that the forward pass must save all intermediate values ($z_1, a_1, z_2, a_2, z_3$) because the backward pass needs them to compute the derivatives. This is why neural network training requires more memory than inference: the activations from the forward pass must be retained until the backward pass uses them.</span>

<span style="font-size: 14px;">In our 3-layer network:</span>

* <span style="font-size: 14px;">$\sigma'(z_1) = a_1(1 - a_1)$ requires $a_1$</span>
* <span style="font-size: 14px;">$\sigma'(z_2) = a_2(1 - a_2)$ requires $a_2$</span>
* <span style="font-size: 14px;">$\sigma'(z_3) = y(1 - y)$ requires $y$</span>

<span style="font-size: 14px;">Techniques like gradient checkpointing trade computation for memory by recomputing some activations during the backward pass instead of storing them.</span>

## <span style="font-size: 14px;">Summary</span>

<span style="font-size: 14px;">The chain rule applied to a 3-layer scalar network gives:</span>

$$
\begin{aligned}
\frac{\partial y}{\partial w_1} &= \underbrace{\sigma'(z_3)}_{\text{layer 3}} \cdot \underbrace{w_3}_{\text{weight 3}} \cdot \underbrace{\sigma'(z_2)}_{\text{layer 2}} \\
&\quad \cdot \underbrace{w_2}_{\text{weight 2}} \cdot \underbrace{\sigma'(z_1)}_{\text{layer 1}} \cdot \underbrace{x}_{\text{input}}
\end{aligned}
$$

<span style="font-size: 14px;">This is backpropagation in its simplest form. Each factor has a clear meaning: activation derivatives control how much gradient passes through each layer, and weights scale the gradient as it propagates. If you can compute this for 3 layers, you understand the algorithm that trains every neural network.</span>
