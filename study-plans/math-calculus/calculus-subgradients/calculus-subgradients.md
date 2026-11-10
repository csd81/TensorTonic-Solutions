<span style="font-size: 14px;">Many of the most important functions in machine learning are not differentiable everywhere. The absolute value function $|x|$ has a corner at $x = 0$. The ReLU activation $\max(0, x)$ has a kink at the origin. The L1 penalty $|w|$ used in Lasso regression is non-smooth at zero - exactly the point where sparsity happens. To optimize objectives involving these functions, we need a generalization of the derivative that works at non-differentiable points. That generalization is the subgradient.</span>

## <span style="font-size: 14px;">Why Classical Derivatives Fail</span>

<span style="font-size: 14px;">The derivative of a function $f$ at a point $x_0$ is defined as the limit:</span>

$$
f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}
$$

<span style="font-size: 14px;">This limit must exist and be equal whether $h$ approaches from the left or the right. For $f(x) = |x|$ at $x_0 = 0$:</span>

$$
\lim_{h \to 0^+} \frac{|0 + h| - |0|}{h} = \lim_{h \to 0^+} \frac{h}{h} = 1
$$

$$
\lim_{h \to 0^-} \frac{|0 + h| - |0|}{h} = \lim_{h \to 0^-} \frac{-h}{h} = -1
$$

<span style="font-size: 14px;">The left and right limits disagree, so the derivative does not exist at $x = 0$. The function has a corner - a point where the slope changes abruptly. The same situation arises for ReLU at $x = 0$, where the left derivative is 0 and the right derivative is 1.</span>

<span style="font-size: 14px;">If we restrict ourselves to classical derivatives, we cannot perform gradient descent on any objective containing $|w|$, $\max(0, w)$, or similar non-smooth terms. Since L1 regularization, ReLU activations, and hinge loss are cornerstones of modern ML, we need a broader framework.</span>

## <span style="font-size: 14px;">The Subgradient: Definition</span>

<span style="font-size: 14px;">For a convex function $f: \mathbb{R} \to \mathbb{R}$, a scalar $g$ is called a subgradient of $f$ at $x_0$ if it satisfies the subgradient inequality:</span>

$$
f(x) \geq f(x_0) + g \cdot (x - x_0) \quad \text{for all } x
$$

<span style="font-size: 14px;">Geometrically, this says that the line $y = f(x_0) + g(x - x_0)$ lies below (or touches) the graph of $f$ everywhere. At a differentiable point, the only line that satisfies this is the tangent line, so the only subgradient is the ordinary derivative $f'(x_0)$. But at a non-differentiable point like a corner, there are many lines that lie below the graph, and each one gives a valid subgradient.</span>

### <span style="font-size: 14px;">The Subdifferential</span>

<span style="font-size: 14px;">The set of all subgradients of $f$ at $x_0$ is called the subdifferential, denoted $\partial f(x_0)$:</span>

$$
\partial f(x_0) = \{g \in \mathbb{R} : f(x) \geq f(x_0) + g(x - x_0), \, \forall x\}
$$

<span style="font-size: 14px;">For a convex function, the subdifferential is always a closed interval $[a, b]$ (possibly a single point when the function is differentiable). The endpoints $a$ and $b$ are the left and right derivatives:</span>

$$
a = \lim_{h \to 0^-} \frac{f(x_0 + h) - f(x_0)}{h}, \quad b = \lim_{h \to 0^+} \frac{f(x_0 + h) - f(x_0)}{h}
$$

<span style="font-size: 14px;">A necessary and sufficient condition for $x^*$ to be a minimizer of a convex function is that $0 \in \partial f(x^*)$. This generalizes the classical condition $f'(x^*) = 0$ to non-smooth functions.</span>

## <span style="font-size: 14px;">Subgradient of the Absolute Value</span>

<span style="font-size: 14px;">For $f(x) = |x|$:</span>

$$
\partial |x| = \begin{cases} \{-1\} & x < 0 \\ [-1, 1] & x = 0 \\ \{1\} & x > 0 \end{cases}
$$

<span style="font-size: 14px;">At $x \neq 0$, the function is differentiable and has a unique subgradient equal to the derivative: $\text{sign}(x)$. At $x = 0$, any value $g \in [-1, 1]$ satisfies the subgradient inequality. To verify: we need $|x| \geq 0 + g \cdot x$ for all $x$. For $x > 0$: $x \geq gx$, so $g \leq 1$. For $x < 0$: $-x \geq gx$, so $g \geq -1$. Thus $g \in [-1, 1]$.</span>

<span style="font-size: 14px;">The standard convention in optimization algorithms is to choose $g = 0$ at $x = 0$, which corresponds to the sign function with $\text{sign}(0) = 0$. This is also the "minimum norm subgradient" - the subgradient with the smallest absolute value.</span>

### <span style="font-size: 14px;">Connection to L1 Regularization</span>

<span style="font-size: 14px;">L1 regularization adds $\lambda |w|$ to the loss function, encouraging weights to be exactly zero (sparsity). The subgradient of $\lambda |w|$ with respect to $w$ is:</span>

$$
\partial(\lambda|w|) = \begin{cases} \{-\lambda\} & w < 0 \\ [-\lambda, \lambda] & w = 0 \\ \{\lambda\} & w > 0 \end{cases}
$$

<span style="font-size: 14px;">At $w = 0$, the subdifferential is the interval $[-\lambda, \lambda]$. A weight stays at zero as long as the data-dependent gradient falls within this interval - the L1 penalty "absorbs" small gradients, which is why Lasso produces sparse solutions. This is fundamentally a subgradient phenomenon: the non-smoothness of $|w|$ at zero creates a "dead zone" for gradients that would otherwise push $w$ away from zero.</span>

## <span style="font-size: 14px;">Subgradient of ReLU</span>

<span style="font-size: 14px;">For $f(x) = \max(0, x) = \text{ReLU}(x)$:</span>

$$
\partial \text{ReLU}(x) = \begin{cases} \{0\} & x < 0 \\ [0, 1] & x = 0 \\ \{1\} & x > 0 \end{cases}
$$

<span style="font-size: 14px;">At $x = 0$, any value $g \in [0, 1]$ is a valid subgradient. The standard convention in deep learning frameworks is to use $g = 0$ at the kink, which corresponds to the left derivative. Some implementations use $g = 1$ (the right derivative) or $g = 0.5$ (the average). In practice, the probability of hitting $x = 0$ exactly is negligible with floating-point arithmetic, so the choice rarely matters. The convention $g = 0$ has the computational advantage of matching the case $x < 0$, allowing a single threshold check: the subgradient is 1 if $x > 0$ and 0 otherwise.</span>

### <span style="font-size: 14px;">Practical Impact in Neural Networks</span>

<span style="font-size: 14px;">ReLU's non-differentiability at zero is one of the reasons the "dying ReLU" problem exists. When a neuron's pre-activation is negative, the gradient is exactly zero, and the neuron stops learning. If the pre-activation lands exactly at zero, the subgradient choice determines whether the neuron can recover. With $g = 0$, a neuron at the boundary gets no gradient and may stay dead. With $g = 0.5$, it gets a small push. In practice, Leaky ReLU and other variants avoid this issue entirely by ensuring a non-zero gradient for all inputs.</span>

## <span style="font-size: 14px;">Properties of Subgradients</span>

### <span style="font-size: 14px;">Sum Rule</span>

<span style="font-size: 14px;">If $f$ and $g$ are convex, then:</span>

$$
\partial(f + g)(x) \supseteq \partial f(x) + \partial g(x)
$$

<span style="font-size: 14px;">where the sum on the right is the Minkowski sum: $A + B = \{a + b : a \in A, b \in B\}$. For practical purposes, if $g_f \in \partial f(x)$ and $g_g \in \partial g(x)$, then $g_f + g_g \in \partial(f + g)(x)$. This is the subgradient analog of the sum rule for derivatives, and it is essential for computing subgradients of composite objectives like "data loss + regularization."</span>

### <span style="font-size: 14px;">Scaling Rule</span>

<span style="font-size: 14px;">For $\alpha > 0$: $\partial(\alpha f)(x) = \alpha \cdot \partial f(x)$. This is straightforward and directly mirrors the scaling rule for ordinary derivatives.</span>

### <span style="font-size: 14px;">Chain Rule (Limited)</span>

<span style="font-size: 14px;">Unlike classical derivatives, the chain rule for subgradients is more restrictive. If $f(x) = h(g(x))$ where $h$ is convex and non-decreasing and $g$ is convex, then a valid subgradient is $h'(g(x)) \cdot g'(x)$ (using any subgradient for each). However, the general chain rule for subgradients requires additional conditions. This limitation is one reason why proximal methods and ADMM are preferred over pure subgradient methods for structured non-smooth optimization.</span>

## <span style="font-size: 14px;">Subgradient Descent</span>

<span style="font-size: 14px;">Subgradient descent is the natural extension of gradient descent to non-smooth convex functions. At each iteration, we select any subgradient $g_t \in \partial f(x_t)$ and update:</span>

$$
x_{t+1} = x_t - \eta_t g_t
$$

<span style="font-size: 14px;">where $\eta_t$ is the step size (learning rate). The key difference from gradient descent is that subgradient descent is NOT guaranteed to decrease the objective at every step. At a non-differentiable point, the chosen subgradient may not be a descent direction. Despite this, the algorithm converges to the minimum under appropriate step-size conditions.</span>

### <span style="font-size: 14px;">Convergence Theory</span>

<span style="font-size: 14px;">For a convex function with bounded subgradients ($\|g\| \leq G$), subgradient descent with a fixed step size $\eta$ satisfies:</span>

$$
\min_{t=1,\ldots,T} f(x_t) - f(x^*) \leq \frac{\|x_1 - x^*\|^2}{2\eta T} + \frac{\eta G^2}{2}
$$

<span style="font-size: 14px;">Setting $\eta = \frac{\|x_1 - x^*\|}{G\sqrt{T}}$ gives a convergence rate of $O(1/\sqrt{T})$. This is slower than gradient descent's $O(1/T)$ rate for smooth functions, reflecting the fact that non-smooth functions provide less information at each step.</span>

<span style="font-size: 14px;">With diminishing step sizes satisfying $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (such as $\eta_t = c/\sqrt{t}$), the iterates converge to the optimal solution: $\lim_{t \to \infty} f(x_t) = f(x^*)$. A fixed step size, while simpler, only guarantees convergence to a neighborhood of the optimum.</span>

### <span style="font-size: 14px;">Why Not Descent?</span>

<span style="font-size: 14px;">Consider minimizing a non-smooth convex function like $f(x) = |x|$ using a fixed step size $\eta$. Starting from any $x_0 > 0$, the subgradient is $g = +1$, so the update gives $x_1 = x_0 - \eta$. If $\eta > x_0$, then $x_1 < 0$ and $f(x_1) = |x_1| = \eta - x_0 > x_0 = f(x_0)$: the objective increased. The next subgradient is $g = -1$, yielding $x_2 = x_1 + \eta = x_0$. The iterates oscillate forever without reaching the minimum. This pathology arises whenever the fixed step size exceeds the distance to the optimum, causing the method to overshoot the non-differentiable point.</span>

<span style="font-size: 14px;">This is why step size management is critical for subgradient methods. When $\eta < |x_0|$, the method moves closer to zero at each step and eventually reaches the optimum. The fundamental tension is that a large step size makes progress quickly when far from the optimum but causes oscillation near it, while a small step size converges reliably but slowly. Diminishing step sizes (satisfying the conditions in the convergence theory above) resolve this tension by using large steps initially and progressively smaller steps as the iterates approach the optimum.</span>

## <span style="font-size: 14px;">The Optimization Problem: $L(w) = |w - 3| + 0.5w^2$</span>

<span style="font-size: 14px;">This problem asks you to minimize a function that combines a non-smooth term $|w - 3|$ with a smooth quadratic term $0.5w^2$. This structure mirrors the elastic net objective in machine learning, where $|w - 3|$ plays the role of L1 regularization (shifted) and $0.5w^2$ plays the role of L2 regularization.</span>

### <span style="font-size: 14px;">Finding the Minimum Analytically</span>

<span style="font-size: 14px;">The subdifferential of $L(w) = |w - 3| + 0.5w^2$ is:</span>

$$
\partial L(w) = \partial|w - 3| + w = \begin{cases} \{-1 + w\} & w < 3 \\ [-1 + 3, 1 + 3] = [2, 4] & w = 3 \\ \{1 + w\} & w > 3 \end{cases}
$$

<span style="font-size: 14px;">The minimum occurs where $0 \in \partial L(w)$. For $w < 3$: $-1 + w = 0 \implies w = 1$. Since $1 < 3$, this is consistent. For $w > 3$: $1 + w = 0 \implies w = -1$, but $-1 > 3$ is a contradiction. At $w = 3$: $0 \notin [2, 4]$. Therefore, the unique minimum is at $w^* = 1$, and the minimum value is $L(w^*) = |w^* - 3| + 0.5(w^*)^2$.</span>

### <span style="font-size: 14px;">Subgradient Descent Dynamics</span>

<span style="font-size: 14px;">Starting from some $w_0$ and using a fixed learning rate $\eta$, the subgradient descent update is:</span>

$$
w_{t+1} = w_t - \eta \cdot g_t
$$

<span style="font-size: 14px;">where $g_t \in \partial L(w_t)$. For $w_t \neq 3$ (the typical case), $g_t = \text{sign}(w_t - 3) + w_t$.</span>

<span style="font-size: 14px;">In the region $w < 3$, the update becomes:</span>

$$
w_{t+1} = w_t - \eta(-1 + w_t) = w_t(1 - \eta) + \eta
$$

<span style="font-size: 14px;">This is a linear recurrence with fixed point $w^* = \eta / \eta = 1$. For $0 < \eta < 2$, the recurrence converges: the contraction factor $(1 - \eta)$ has absolute value less than 1, so $w_t \to 1$ geometrically. Specifically, the error at step $t$ satisfies $|w_t - 1| = |1 - \eta|^t |w_0 - 1|$, which decays exponentially. Smaller values of $|1 - \eta|$ yield faster convergence, with the optimal contraction occurring at $\eta = 1$ (where convergence is immediate in this region). For $\eta > 2$, the contraction factor exceeds 1 and the iterates diverge.</span>

<span style="font-size: 14px;">In the region $w > 3$, the gradient $1 + w$ is always positive (since $w > 3 > 0$), so the update always decreases $w$. The trajectory moves toward $w = 3$ from above, and once it crosses below 3, the dynamics switch to the $w < 3$ regime described above.</span>

## <span style="font-size: 14px;">Proximal Methods: A Better Alternative</span>

<span style="font-size: 14px;">While subgradient descent works for non-smooth optimization, it converges slowly ($O(1/\sqrt{T})$ vs. $O(1/T)$ for smooth problems). Proximal gradient methods offer a faster alternative when the objective has the form $f(x) = g(x) + h(x)$ where $g$ is smooth and $h$ is non-smooth but has a tractable "proximal operator."</span>

<span style="font-size: 14px;">The proximal operator of $h$ with step size $\eta$ is:</span>

$$
\text{prox}_{\eta h}(v) = \arg\min_x \left\{h(x) + \frac{1}{2\eta}\|x - v\|^2\right\}
$$

<span style="font-size: 14px;">For $h(x) = \lambda|x|$, the proximal operator is the soft-thresholding function:</span>

$$
\text{prox}_{\eta \lambda |\cdot|}(v) = \text{sign}(v) \cdot \max(|v| - \eta\lambda, 0)
$$

<span style="font-size: 14px;">This is the operation behind the ISTA (Iterative Shrinkage-Thresholding Algorithm) and FISTA algorithms used for Lasso regression. The proximal approach achieves $O(1/T)$ convergence (and $O(1/T^2)$ with acceleration), much faster than subgradient descent.</span>

<span style="font-size: 14px;">For our problem $L(w) = |w - 3| + 0.5w^2$, the proximal gradient method would first take a gradient step on the smooth part $0.5w^2$ and then apply the proximal operator for $|w - 3|$. This would converge faster than the pure subgradient approach, but understanding subgradient descent is a prerequisite for understanding why proximal methods are designed the way they are.</span>

## <span style="font-size: 14px;">Subgradients in Deep Learning</span>

<span style="font-size: 14px;">In practice, deep learning frameworks do not explicitly compute subgradients. Instead, they use automatic differentiation, which computes derivatives of each elementary operation. At non-differentiable points (like ReLU at 0), the framework uses a fixed convention - typically the left or right derivative. This is technically a subgradient (since the left and right derivatives at a convex kink are both valid subgradients), and it works well in practice because:</span>

<span style="font-size: 14px;">1. The probability of a continuous-valued input hitting a non-differentiable point exactly is measure-zero.</span>

<span style="font-size: 14px;">2. The loss function of a neural network is piecewise smooth (smooth within each "linear region" defined by ReLU activations), so gradient descent with subgradients at the kinks still converges.</span>

<span style="font-size: 14px;">3. Stochastic gradient noise typically dominates the subgradient ambiguity, so the choice at non-differentiable points has negligible effect on training.</span>

<span style="font-size: 14px;">However, understanding subgradients remains essential for implementing custom loss functions, designing regularization schemes, and analyzing convergence properties of training algorithms. The theoretical guarantees of SGD for non-smooth objectives rely on subgradient theory.</span>

## <span style="font-size: 14px;">Summary</span>

<span style="font-size: 14px;">Subgradients generalize derivatives to non-smooth convex functions. At a differentiable point, the subgradient is unique and equals the derivative. At a non-differentiable point, the subdifferential is an interval of valid subgradients. The optimality condition for convex minimization is $0 \in \partial f(x^*)$, which generalizes $f'(x^*) = 0$. Subgradient descent uses any subgradient as a descent direction, converging at a rate of $O(1/\sqrt{T})$ for convex functions. These concepts underpin L1 regularization, ReLU backpropagation, and the design of modern non-smooth optimization algorithms used throughout machine learning.</span>