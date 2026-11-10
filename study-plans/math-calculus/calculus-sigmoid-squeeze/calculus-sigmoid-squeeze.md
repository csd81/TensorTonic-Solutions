<span style="font-size: 14px;">The Squeeze Theorem (also called the Sandwich Theorem or Pinching Theorem) is one of the most powerful tools in calculus for establishing limits and bounds on functions. In machine learning, it provides the rigorous foundation for understanding why the sigmoid activation saturates for extreme inputs, which directly causes the vanishing gradient problem that shaped the evolution of neural network architectures.</span>

## <span style="font-size: 14px;">The Squeeze Theorem: Statement and Proof</span>

<span style="font-size: 14px;">The Squeeze Theorem states:</span>

<span style="font-size: 14px;">If $g(x) \leq f(x) \leq h(x)$ for all $x$ in some interval around $a$ (except possibly at $a$ itself), and</span>

$$
\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L
$$

<span style="font-size: 14px;">then</span>

$$
\lim_{x \to a} f(x) = L
$$

<span style="font-size: 14px;">The intuition is simple: if $f$ is trapped between two functions that both converge to the same value $L$, then $f$ has no choice but to converge to $L$ as well. The function is "squeezed" into the limit.</span>

### <span style="font-size: 14px;">Proof Sketch</span>

<span style="font-size: 14px;">For any $\epsilon > 0$, since $\lim_{x \to a} g(x) = L$, there exists $\delta_1 > 0$ such that $|g(x) - L| < \epsilon$ whenever $0 < |x - a| < \delta_1$. Similarly, since $\lim_{x \to a} h(x) = L$, there exists $\delta_2 > 0$ such that $|h(x) - L| < \epsilon$ whenever $0 < |x - a| < \delta_2$.</span>

<span style="font-size: 14px;">Let $\delta = \min(\delta_1, \delta_2)$. Then for $0 < |x - a| < \delta$:</span>

$$
L - \epsilon < g(x) \leq f(x) \leq h(x) < L + \epsilon
$$

<span style="font-size: 14px;">which gives $|f(x) - L| < \epsilon$. Since $\epsilon$ was arbitrary, $\lim_{x \to a} f(x) = L$.</span>

### <span style="font-size: 14px;">Classic Example: $\lim_{x \to 0} x \sin(1/x)$</span>

<span style="font-size: 14px;">The function $x \sin(1/x)$ oscillates wildly near $x = 0$, but we can bound it:</span>

$$
-|x| \leq x \sin(1/x) \leq |x|
$$

<span style="font-size: 14px;">Since $\lim_{x \to 0} (-|x|) = 0$ and $\lim_{x \to 0} |x| = 0$, the Squeeze Theorem gives $\lim_{x \to 0} x \sin(1/x) = 0$.</span>

### <span style="font-size: 14px;">Squeeze Theorem at Infinity</span>

<span style="font-size: 14px;">The theorem also applies to limits at infinity. If $g(x) \leq f(x) \leq h(x)$ for all sufficiently large $x$, and</span>

$$
\lim_{x \to \infty} g(x) = \lim_{x \to \infty} h(x) = L
$$

<span style="font-size: 14px;">then $\lim_{x \to \infty} f(x) = L$. The same holds for $x \to -\infty$. This version is particularly relevant for sigmoid analysis, where we care about the behavior for large positive and large negative inputs.</span>

## <span style="font-size: 14px;">The Sigmoid Function</span>

<span style="font-size: 14px;">The sigmoid (or logistic) function is defined as:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

<span style="font-size: 14px;">This function maps any real number to the interval $(0, 1)$, making it historically popular as an activation function in neural networks and as the link function in logistic regression.</span>

### <span style="font-size: 14px;">Equivalent Forms</span>

<span style="font-size: 14px;">The sigmoid has several equivalent representations that are useful in different contexts:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} = \frac{1}{2} + \frac{1}{2}\tanh\!\left(\frac{x}{2}\right)
$$

<span style="font-size: 14px;">The form $e^x / (e^x + 1)$ is sometimes more convenient algebraically. The tanh form shows that sigmoid is just a shifted and scaled version of the hyperbolic tangent.</span>

### <span style="font-size: 14px;">Key Properties</span>

<span style="font-size: 14px;">The sigmoid function has several important properties:</span>

* <span style="font-size: 14px;">**Range**: $0 < \sigma(x) < 1$ for all real $x$. The function never actually reaches 0 or 1.</span>
* <span style="font-size: 14px;">**Symmetry**: $\sigma(-x) = 1 - \sigma(x)$. The function is symmetric about the point $(0, 0.5)$.</span>
* <span style="font-size: 14px;">**Monotonicity**: $\sigma$ is strictly increasing. $\sigma'(x) > 0$ for all $x$.</span>
* <span style="font-size: 14px;">**Midpoint**: $\sigma(0) = 0.5$</span>
* <span style="font-size: 14px;">**Derivative**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, which has a beautiful self-referential form</span>
* <span style="font-size: 14px;">**Inflection point**: at $x = 0$, where the derivative is maximal: $\sigma'(0) = \frac{1}{4}$</span>

### <span style="font-size: 14px;">Proving the Range is $(0, 1)$</span>

<span style="font-size: 14px;">To rigorously show that $0 < \sigma(x) < 1$, note that:</span>

<span style="font-size: 14px;">Since $e^{-x} > 0$ for all real $x$, we have $1 + e^{-x} > 1$, and therefore:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}} < \frac{1}{1} = 1
$$

<span style="font-size: 14px;">Also, since $1 + e^{-x} > 0$ (both terms are positive), we have:</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}} > 0
$$

<span style="font-size: 14px;">So $\sigma(x) \in (0, 1)$ for all $x \in \mathbb{R}$. Note that the bounds 0 and 1 are never attained; they are approached asymptotically.</span>

## <span style="font-size: 14px;">Bounding the Sigmoid with the Squeeze Theorem</span>

<span style="font-size: 14px;">While we have shown that $0 < \sigma(x) < 1$, we can construct much tighter bounds that converge to the same limits as $\sigma$ for large $|x|$. These tighter bounds provide a constructive proof using the Squeeze Theorem.</span>

### <span style="font-size: 14px;">Upper Bound: $\sigma(x) \leq \min(1, e^x)$</span>

<span style="font-size: 14px;">For the upper bound, we consider two cases:</span>

<span style="font-size: 14px;">**Case 1: $x \geq 0$.** Since $\sigma(x) < 1$ always, and $e^x \geq 1$ when $x \geq 0$, the tighter bound is simply $\sigma(x) < 1$. So $\min(1, e^x) = 1 \geq \sigma(x)$.</span>

<span style="font-size: 14px;">**Case 2: $x < 0$.** Using the form $\sigma(x) = e^x / (1 + e^x)$, and noting that $1 + e^x > 1$ when $e^x > 0$:</span>

$$
\sigma(x) = \frac{e^x}{1 + e^x} < \frac{e^x}{1} = e^x
$$

<span style="font-size: 14px;">Since $e^x < 1$ when $x < 0$, we have $\min(1, e^x) = e^x \geq \sigma(x)$.</span>

<span style="font-size: 14px;">Combining both cases: $\sigma(x) \leq \min(1, e^x)$ for all $x$.</span>

### <span style="font-size: 14px;">Lower Bound: $\sigma(x) \geq \max(0, 1 - e^{-x})$</span>

<span style="font-size: 14px;">For the lower bound, again two cases:</span>

<span style="font-size: 14px;">**Case 1: $x < 0$.** Since $\sigma(x) > 0$ always, and $1 - e^{-x} < 0$ when $x < 0$ (because $-x > 0$ gives $e^{-x} > 1$), we have $\max(0, 1 - e^{-x}) = 0 \leq \sigma(x)$.</span>

<span style="font-size: 14px;">**Case 2: $x \geq 0$.** We need to show that $\sigma(x) \geq 1 - e^{-x}$ when $x \geq 0$. Setting $u = e^{-x}$ where $0 < u \leq 1$:</span>

$$
\begin{aligned}
\sigma(x) - (1 - e^{-x}) &= \frac{1}{1 + u} - (1 - u) \\
&= \frac{1 - (1 - u)(1 + u)}{1 + u} \\
&= \frac{1 - (1 - u^2)}{1 + u} = \frac{u^2}{1 + u} \geq 0
\end{aligned}
$$

<span style="font-size: 14px;">Since $u^2/(1+u) \geq 0$ for all $u \geq 0$, we have $\sigma(x) \geq 1 - e^{-x}$ when $x \geq 0$. Therefore $\sigma(x) \geq \max(0, 1 - e^{-x})$ for all $x$.</span>

### <span style="font-size: 14px;">The Complete Squeeze</span>

<span style="font-size: 14px;">Combining the bounds:</span>

$$
\max(0, 1 - e^{-x}) \leq \sigma(x) \leq \min(1, e^x)
$$

<span style="font-size: 14px;">This inequality holds for all real $x$. Now applying the Squeeze Theorem:</span>

<span style="font-size: 14px;">**As $x \to +\infty$:**</span>

$$
\lim_{x \to +\infty} \max(0, 1 - e^{-x}) = \lim_{x \to +\infty} (1 - e^{-x}) = 1
$$

$$
\lim_{x \to +\infty} \min(1, e^x) = 1
$$

<span style="font-size: 14px;">Both bounds converge to 1, so $\lim_{x \to +\infty} \sigma(x) = 1$.</span>

<span style="font-size: 14px;">**As $x \to -\infty$:**</span>

$$
\lim_{x \to -\infty} \max(0, 1 - e^{-x}) = 0
$$

$$
\lim_{x \to -\infty} \min(1, e^x) = \lim_{x \to -\infty} e^x = 0
$$

<span style="font-size: 14px;">Both bounds converge to 0, so $\lim_{x \to -\infty} \sigma(x) = 0$.</span>

## <span style="font-size: 14px;">How Tight Are the Bounds?</span>

<span style="font-size: 14px;">The gap between the upper and lower bounds decreases exponentially as $|x|$ increases. Let us quantify this:</span>

<span style="font-size: 14px;">**For $x \geq 0$:** the gap is $1 - (1 - e^{-x}) = e^{-x}$, which decays exponentially.</span>

<span style="font-size: 14px;">**For $x < 0$:** the gap is $e^x - 0 = e^x$, which also decays exponentially (toward 0 as $x \to -\infty$).</span>

<span style="font-size: 14px;">At $x = 0$, the gap is maximal: upper $= \min(1, 1) = 1$, lower $= \max(0, 0) = 0$, gap $= 1$. This makes sense because $\sigma(0)$ is in the middle of the range.</span>

<span style="font-size: 14px;">As $|x|$ increases, the gap $e^{-|x|}$ decays exponentially, so the bounds become extremely tight very quickly. For moderately large $|x|$, the bounds are essentially indistinguishable from the true sigmoid value.</span>

## <span style="font-size: 14px;">Saturation of the Sigmoid</span>

<span style="font-size: 14px;">A function is said to **saturate** when its output approaches a constant value and becomes insensitive to changes in input. For the sigmoid, saturation occurs in two regimes:</span>

* <span style="font-size: 14px;">**Positive saturation** ($x \gg 0$): $\sigma(x) \approx 1$</span>
* <span style="font-size: 14px;">**Negative saturation** ($x \ll 0$): $\sigma(x) \approx 0$</span>

<span style="font-size: 14px;">More precisely, for sufficiently large $|x|$, the distance from saturation is bounded by $e^{-|x|}$, which decays exponentially. The sigmoid output rapidly approaches its limiting value of 0 or 1.</span>

### <span style="font-size: 14px;">Rate of Saturation</span>

<span style="font-size: 14px;">The distance from saturation decays exponentially:</span>

* <span style="font-size: 14px;">For large positive $x$: $1 - \sigma(x) = \sigma(-x) = \frac{1}{1 + e^x} \approx e^{-x}$</span>
* <span style="font-size: 14px;">For large negative $x$: $\sigma(x) = \frac{1}{1 + e^{-x}} \approx e^x$</span>

<span style="font-size: 14px;">The exponential decay means that saturation happens very rapidly. For large enough $|x|$, the sigmoid is within machine precision of its limiting value for any fixed-precision floating-point format.</span>

## <span style="font-size: 14px;">The Vanishing Gradient Problem</span>

<span style="font-size: 14px;">The saturation of sigmoid has a direct and devastating consequence for training deep neural networks: the **vanishing gradient problem**.</span>

### <span style="font-size: 14px;">Sigmoid Derivative</span>

<span style="font-size: 14px;">The derivative of the sigmoid function is:</span>

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

<span style="font-size: 14px;">This derivative reaches its maximum at $x = 0$ where $\sigma'(0) = \frac{1}{4}$. For large $|x|$:</span>

* <span style="font-size: 14px;">When $\sigma(x) \approx 1$: $\sigma'(x) \approx 1 \cdot 0 = 0$</span>
* <span style="font-size: 14px;">When $\sigma(x) \approx 0$: $\sigma'(x) \approx 0 \cdot 1 = 0$</span>

<span style="font-size: 14px;">So when the sigmoid saturates, its gradient essentially vanishes. In a deep network with $L$ layers all using sigmoid activations, the gradient flowing back from the output to an early layer is the product of $L$ terms, each bounded by $\frac{1}{4}$:</span>

$$
\begin{aligned}
\frac{\partial \text{Loss}}{\partial w_1} &= \frac{\partial \text{Loss}}{\partial z_L} \cdot \prod_{l=1}^{L} \sigma'(z_l) \cdot w_l
\end{aligned}
$$

<span style="font-size: 14px;">Even if none of the sigmoids are fully saturated, the maximum product of $L$ sigmoid derivatives is $(1/4)^L$. Since $1/4 < 1$, this product decays exponentially with depth. The gradients become astronomically small, and learning effectively stops.</span>

### <span style="font-size: 14px;">Architectural Mitigations</span>

<span style="font-size: 14px;">The key architectural responses to the vanishing gradient problem include:</span>

* <span style="font-size: 14px;">**ReLU activation**: $\text{ReLU}(x) = \max(0, x)$ has gradient 1 for positive inputs, avoiding the multiplicative decay inherent in sigmoid</span>
* <span style="font-size: 14px;">**Residual connections**: provide a direct gradient pathway that bypasses the activation functions, so $\frac{\partial}{\partial x}(x + F(x)) = 1 + F'(x)$</span>
* <span style="font-size: 14px;">**Batch normalization**: prevents pre-activations from drifting into the saturated regime by centering and scaling</span>
* <span style="font-size: 14px;">**LSTM/GRU gates**: use sigmoid strategically with additive (rather than multiplicative) gradient paths</span>

## <span style="font-size: 14px;">Saturation Threshold Analysis</span>

<span style="font-size: 14px;">For a given tolerance $\epsilon$, the sigmoid output $\sigma(x)$ is considered **near saturation** if:</span>

$$
\min(\sigma(x), 1 - \sigma(x)) < \epsilon
$$

<span style="font-size: 14px;">This is equivalent to $\sigma(x) < \epsilon$ or $\sigma(x) > 1 - \epsilon$. Using the approximation $\sigma(x) \approx e^x$ for large negative $x$:</span>

$$
e^x < \epsilon \implies x < \ln(\epsilon)
$$

<span style="font-size: 14px;">Symmetrically, saturation at the upper end occurs for $x > -\ln(\epsilon)$. This provides a closed-form expression for the saturation threshold at any desired tolerance: the sigmoid is within $\epsilon$ of its limit whenever $|x| > |\ln(\epsilon)|$.</span>

## <span style="font-size: 14px;">Why Bounds Matter in Practice</span>

<span style="font-size: 14px;">Bounding functions are not just theoretical curiosities. They appear throughout machine learning in several ways:</span>

### <span style="font-size: 14px;">Numerical Stability</span>

<span style="font-size: 14px;">When computing $\sigma(x)$ for very large or very small $x$, direct evaluation of $1/(1 + e^{-x})$ can suffer from floating-point overflow. For large negative $x$, $e^{-x}$ overflows to infinity, giving an indeterminate result. Using the equivalent form $e^x / (1 + e^x)$ for negative $x$ avoids this issue, and the bounding analysis tells us exactly when this switch is necessary.</span>

### <span style="font-size: 14px;">Gradient Clipping and Stabilization</span>

<span style="font-size: 14px;">Knowing that sigmoid saturates exponentially fast allows frameworks to implement efficient gradient computation. Beyond a threshold determined by the machine precision, one can simply return 0 for $\sigma'(x)$ without any loss of numerical accuracy, since $\sigma(x)(1 - \sigma(x))$ is already below representable precision.</span>

### <span style="font-size: 14px;">Probabilistic Interpretation</span>

<span style="font-size: 14px;">In logistic regression and binary classification, sigmoid converts log-odds (logits) to probabilities. The saturation analysis tells us that logits with large magnitude represent "confident" predictions (probability exponentially close to 0 or 1). This has practical implications for calibration and uncertainty quantification.</span>

## <span style="font-size: 14px;">When Sigmoid Is Still Used</span>

<span style="font-size: 14px;">Despite the vanishing gradient problem, sigmoid remains important in modern architectures:</span>

* <span style="font-size: 14px;">**Binary classification output**: the final layer of a binary classifier typically uses sigmoid to produce a probability</span>
* <span style="font-size: 14px;">**Gate mechanisms**: LSTM and GRU cells use sigmoid gates to control information flow. The saturation here is actually desirable, as gates should be close to 0 or 1</span>
* <span style="font-size: 14px;">**Attention weights**: some attention mechanisms use sigmoid instead of softmax for independent, multi-label attention</span>
* <span style="font-size: 14px;">**Swish/SiLU**: $x \cdot \sigma(x)$ is a modern activation that uses sigmoid as a self-gating mechanism</span>

<span style="font-size: 14px;">In these contexts, the saturation behavior is either handled by careful initialization or is actually a feature rather than a bug.</span>

## <span style="font-size: 14px;">Summary of Key Results</span>

<span style="font-size: 14px;">The analysis of sigmoid boundedness via the Squeeze Theorem establishes:</span>

* <span style="font-size: 14px;">Lower bound: $\sigma(x) \geq \max(0, 1 - e^{-x})$</span>
* <span style="font-size: 14px;">Upper bound: $\sigma(x) \leq \min(1, e^x)$</span>
* <span style="font-size: 14px;">Both bounds converge to the same limit as $\sigma$ for $x \to \pm\infty$</span>
* <span style="font-size: 14px;">The gap between bounds decays as $e^{-|x|}$</span>
* <span style="font-size: 14px;">For $|x| > |\ln(\epsilon)|$, $\sigma(x)$ is within $\epsilon$ of 0 or 1 (saturation)</span>
* <span style="font-size: 14px;">Saturation causes the gradient $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ to vanish</span>
* <span style="font-size: 14px;">In deep networks, this gradient vanishing compounds exponentially with depth</span>
