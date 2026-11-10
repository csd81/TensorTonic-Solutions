# <span style="font-size: 20px;">Learning Rate Stability and Convergence Bounds</span>

## The Quadratic Model of Optimization

The simplest nontrivial optimization problem is minimizing a quadratic function. For a scalar weight $w$, consider the loss:

$$L(w) = \frac{1}{2} C w^2$$

where $C > 0$ is a constant that represents the curvature of the loss surface. This is the prototypical "bowl-shaped" loss that arises when you zoom in on any smooth loss function near its minimum. The Taylor expansion of any twice-differentiable loss around its minimum $w^*$ is:

$$L(w) \approx L(w^*) + \frac{1}{2} L''(w^*)(w - w^*)^2$$

So near the minimum, every smooth loss looks quadratic, and the curvature $C$ plays the role of $L''(w^*)$, the second derivative at the minimum.

Despite its simplicity, this quadratic model captures the fundamental relationship between learning rate and convergence. The analysis extends directly to multivariable problems through eigenvalue decomposition of the Hessian, making it the foundation of convergence theory for gradient-based optimization.

### Why the Quadratic Model Matters

In machine learning, the quadratic model is not merely an approximation. Several important loss functions are exactly quadratic:

- **Linear regression with MSE loss:** $L(w) = \|Xw - y\|^2$ is a quadratic function of $w$
- **The final phase of training:** Near convergence, any smooth loss is well-approximated by its quadratic Taylor expansion
- **Second-order methods:** Newton's method, natural gradient, and K-FAC all work by solving a quadratic approximation at each step

Understanding the quadratic case completely is equivalent to understanding the local convergence behavior of gradient descent on any smooth loss.

### The Gradient

The gradient of $L(w) = \frac{1}{2} C w^2$ is:

$$L'(w) = Cw$$

This is a linear function of $w$: the gradient points away from the origin (the minimum) with magnitude proportional to how far $w$ is from zero. The proportionality constant is $C$, the curvature. Higher curvature means steeper slopes and larger gradients at the same distance from the minimum.

## Gradient Descent on the Quadratic

### The Update Rule

Gradient descent with learning rate $\alpha > 0$ updates the weight as:

$$w_{t+1} = w_t - \alpha \cdot L'(w_t) = w_t - \alpha C w_t = (1 - \alpha C) w_t$$

This is a simple linear recurrence. The weight at each step is a fixed multiple of the weight at the previous step. The multiplier $(1 - \alpha C)$ is constant across all iterations because the gradient is linear.

### Closed-Form Solution

Since $w_{t+1} = (1 - \alpha C) w_t$, we can unroll the recurrence:

$$w_t = (1 - \alpha C)^t \cdot w_0$$

This gives us the exact value of $w$ at any iteration $t$ without having to simulate the trajectory step by step. The behavior is entirely determined by the base $(1 - \alpha C)$ raised to the power $t$.

### The Loss Trajectory

Substituting the closed-form solution into the loss:

$$L(w_t) = \frac{1}{2} C w_t^2 = \frac{1}{2} C (1 - \alpha C)^{2t} w_0^2 = (1 - \alpha C)^{2t} \cdot L(w_0)$$

The loss decays (or grows) geometrically with rate $(1 - \alpha C)^2$ per iteration. If $|1 - \alpha C| < 1$, the loss shrinks exponentially. If $|1 - \alpha C| > 1$, the loss grows exponentially - the optimizer diverges.

## The Contraction Mapping Perspective

### Contraction Maps

A function $g: \mathbb{R} \to \mathbb{R}$ is a contraction mapping if there exists a constant $\rho \in [0, 1)$ such that for all $x, y$:

$$|g(x) - g(y)| \leq \rho |x - y|$$

The constant $\rho$ is called the contraction factor. A contraction mapping brings points closer together by at least a factor of $\rho$ at each application.

### Gradient Descent as a Contraction

The gradient descent update defines a map:

$$g(w) = w - \alpha C w = (1 - \alpha C) w$$

The fixed point of this map is $w^* = 0$ (the minimum of the loss). For this linear map:

$$|g(w) - g(w^*)| = |g(w) - 0| = |1 - \alpha C| \cdot |w|= |1 - \alpha C| \cdot |w - w^*|$$

So gradient descent is a contraction mapping if and only if the contraction factor $\rho = |1 - \alpha C|$ satisfies $\rho < 1$.

### The Banach Fixed-Point Theorem

The Banach fixed-point theorem (also called the contraction mapping theorem) states: if $g$ is a contraction on a complete metric space, then $g$ has a unique fixed point, and the iteration $w_{t+1} = g(w_t)$ converges to this fixed point from any starting point $w_0$.

For gradient descent on the quadratic, this means: if $|1 - \alpha C| < 1$, then the iterates $w_t$ converge to $w^* = 0$ regardless of the initialization $w_0$. This is a global convergence guarantee - there is no dependence on how far away the initial point is.

### Rate of Convergence

The contraction factor $\rho = |1 - \alpha C|$ directly determines the convergence speed:

$$|w_t - w^*| = \rho^t |w_0 - w^*|$$

After $t$ iterations, the distance to the minimum has been reduced by a factor of $\rho^t$. To reduce the error by a factor of $\epsilon$, we need:

$$\rho^t \leq \epsilon \implies t \geq \frac{\ln(1/\epsilon)}{\ln(1/\rho)}$$

For $\rho = 0.5$, we need about $t \geq 10$ iterations to reduce the error by $1000\times$. For $\rho = 0.99$, we need about $t \geq 690$ iterations for the same reduction.

## Deriving the Convergence Bound

### The Stability Condition

Gradient descent converges if and only if $|1 - \alpha C| < 1$. Expanding this absolute value inequality:

$$-1 < 1 - \alpha C < 1$$

The right inequality gives:

$$1 - \alpha C < 1 \implies -\alpha C < 0 \implies \alpha C > 0$$

This is automatically satisfied since $\alpha > 0$ and $C > 0$.

The left inequality gives:

$$1 - \alpha C > -1 \implies 2 > \alpha C \implies \alpha < \frac{2}{C}$$

Therefore, the convergence condition is:

$$\boxed{0 < \alpha < \frac{2}{C}}$$

The maximum stable learning rate is $\alpha_{\max} = 2/C$. Any learning rate above this threshold causes divergence.

### Boundary Behavior

At the exact boundary $\alpha = 2/C$:

$$1 - \alpha C = 1 - 2 = -1$$

So $w_{t+1} = -w_t$, meaning the weight oscillates between $w_0$ and $-w_0$ forever without converging or diverging. The loss stays constant at $L(w_0)$. This is a marginally stable orbit.

### The Role of Curvature

The convergence bound $\alpha < 2/C$ reveals a fundamental principle: **the maximum stable learning rate is inversely proportional to the curvature**. High-curvature losses (steep, narrow bowls) require small learning rates. Low-curvature losses (shallow, wide bowls) can tolerate large learning rates.

This is the mathematical reason why:
- Learning rate tuning is problem-dependent
- Features with different scales cause training difficulties (different effective curvatures)
- Feature normalization and batch normalization improve training stability (they equalize curvatures)
- Second-order methods like Newton's method, which adapt to curvature, can use larger effective step sizes

## Three Regimes of Learning Rate Behavior

### Regime 1: Fast Monotone Convergence ($0 < \alpha < 1/C$)

When $\alpha < 1/C$, the multiplier $1 - \alpha C$ is between 0 and 1:

$$0 < 1 - \alpha C < 1$$

The weight shrinks monotonically toward zero. Each iterate has the same sign as $w_0$ and is strictly closer to zero than the previous one. The trajectory is a smooth, monotonically decreasing (in absolute value) sequence.

The loss decreases monotonically at each step. There is no oscillation or overshooting. This is the safest regime.

### Regime 2: Oscillatory Convergence ($1/C < \alpha < 2/C$)

When $1/C < \alpha < 2/C$, the multiplier $1 - \alpha C$ is between $-1$ and $0$:

$$-1 < 1 - \alpha C < 0$$

The weight alternates sign at each step: $w_0, -|r| w_0, |r|^2 w_0, -|r|^3 w_0, \ldots$ where $r = 1 - \alpha C \in (-1, 0)$. The magnitude $|w_t| = |r|^t |w_0|$ still decreases, so the iterates converge to zero, but they overshoot past the minimum at every step.

The loss still decreases overall (since $|r|^2 < 1$), but the weight itself oscillates. The closer $\alpha$ is to $2/C$, the larger the oscillations and the slower the convergence.

### Regime 3: Divergence ($\alpha > 2/C$)

When $\alpha > 2/C$, the multiplier $1 - \alpha C < -1$:

$$1 - \alpha C < -1$$

The weight alternates sign AND grows in magnitude at each step: $|w_{t+1}| = |1 - \alpha C| \cdot |w_t| > |w_t|$. The iterates diverge to $\pm \infty$ with alternating sign.

The loss grows exponentially: $L(w_t) = (1 - \alpha C)^{2t} L(w_0)$ with $(1 - \alpha C)^2 > 1$. The optimization has catastrophically failed.

### Summary Table

| Range of $\alpha$ | Multiplier $1 - \alpha C$ | Behavior | Convergence |
|---|---|---|---|
| $(0, 1/C)$ | $(0, 1)$ | Monotone decrease | Yes, no oscillation |
| $\alpha = 1/C$ | $0$ | Converges in 1 step | Optimal |
| $(1/C, 2/C)$ | $(-1, 0)$ | Oscillatory decrease | Yes, with oscillation |
| $\alpha = 2/C$ | $-1$ | Oscillation, no decay | Marginal stability |
| $(2/C, \infty)$ | $(-\infty, -1)$ | Oscillatory increase | Divergence |

## The Optimal Learning Rate

### Minimizing the Contraction Factor

The convergence speed is determined by $\rho = |1 - \alpha C|$. To converge as fast as possible, we minimize $\rho$ over $\alpha$:

$$\frac{d}{d\alpha} |1 - \alpha C| = 0$$

The function $1 - \alpha C$ is a decreasing linear function of $\alpha$. Its absolute value is minimized where $1 - \alpha C = 0$, giving:

$$\alpha^* = \frac{1}{C}$$

At this optimal learning rate, $1 - \alpha^* C = 0$, so $w_1 = 0 \cdot w_0 = 0$. Gradient descent converges in a single step. The contraction factor is $\rho = 0$.

### Why One-Step Convergence is Special to Quadratics

The one-step convergence at $\alpha = 1/C$ is a special property of quadratic functions, where the gradient is linear and perfectly predicts the location of the minimum. For nonlinear gradients (non-quadratic losses), no fixed learning rate achieves one-step convergence, and the optimal learning rate depends on the current position.

### Connection to Newton's Method

Newton's method for minimizing $L(w)$ uses the update:

$$w_{t+1} = w_t - \frac{L'(w_t)}{L''(w_t)}$$

For $L(w) = \frac{1}{2} C w^2$, this gives $w_{t+1} = w_t - \frac{Cw_t}{C} = 0$. Newton's method converges in one step for quadratics. Comparing with gradient descent: $w_{t+1} = w_t - \alpha C w_t$, we see that Newton's method implicitly uses $\alpha = 1/C = 1/L''(w)$, the inverse of the local curvature. This is exactly the optimal learning rate.

Newton's method can be viewed as gradient descent with an automatically chosen, curvature-adapted learning rate. This explains its fast convergence but also its computational cost (computing the Hessian $L''$ or its inverse).

## Multivariable Generalization

### The Multivariable Quadratic

For a weight vector $\mathbf{w} \in \mathbb{R}^n$, the quadratic loss is:

$$L(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top H \mathbf{w}$$

where $H$ is a symmetric positive definite matrix (the Hessian). The gradient is $\nabla L = H\mathbf{w}$, and the gradient descent update is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha H \mathbf{w}_t = (I - \alpha H)\mathbf{w}_t$$

### Eigenvalue Decomposition

Since $H$ is symmetric, it has real eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ with orthonormal eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$. Decomposing $\mathbf{w}_0$ in the eigenbasis:

$$\mathbf{w}_0 = \sum_{i=1}^n c_i \mathbf{v}_i$$

The gradient descent update becomes:

$$\mathbf{w}_t = \sum_{i=1}^n (1 - \alpha \lambda_i)^t c_i \mathbf{v}_i$$

Each eigendirection evolves independently with its own multiplier $(1 - \alpha \lambda_i)$.

### The Convergence Condition

For convergence, we need $|1 - \alpha \lambda_i| < 1$ for ALL eigenvalues. The binding constraints come from the smallest and largest eigenvalues:

$$|1 - \alpha \lambda_{\min}| < 1 \quad \text{and} \quad |1 - \alpha \lambda_{\max}| < 1$$

The first gives $\alpha > 0$ (always satisfied). The second gives $\alpha < 2/\lambda_{\max}$.

Therefore, the multivariable convergence bound is:

$$\boxed{\alpha < \frac{2}{\lambda_{\max}(H)}}$$

where $\lambda_{\max}$ is the largest eigenvalue of the Hessian. This generalizes the scalar result $\alpha < 2/C$ since, for the scalar quadratic, $C$ is the only eigenvalue.

### The Condition Number Problem

The convergence rate in the multivariable case is:

$$\rho = \max_i |1 - \alpha \lambda_i|$$

The optimal learning rate minimizes this maximum. It can be shown that the optimal $\alpha$ is:

$$\alpha^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}$$

and the resulting convergence rate is:

$$\rho^* = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}$$

where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number of $H$.

For well-conditioned problems ($\kappa \approx 1$), $\rho^* \approx 0$ and convergence is fast. For ill-conditioned problems ($\kappa \gg 1$), $\rho^* \approx 1$ and convergence is extremely slow. This is the fundamental limitation of gradient descent with a fixed learning rate.

### The Condition Number in Neural Networks

Neural network Hessians are typically ill-conditioned, with condition numbers ranging from $10^3$ to $10^8$ or more. This is why:

- Vanilla gradient descent is too slow for deep learning
- Adaptive methods (Adam, RMSProp) that estimate per-parameter curvature are preferred
- Preconditioning (batch normalization, weight normalization) improves conditioning
- Second-order methods (K-FAC, L-BFGS) that approximate the full Hessian can converge faster

## Lipschitz Continuity and Smoothness

### Lipschitz Continuous Gradients

A function $L$ has $C$-Lipschitz continuous gradients (or is $C$-smooth) if:

$$\|L'(w) - L'(v)\| \leq C \|w - v\| \quad \text{for all } w, v$$

The constant $C$ is the Lipschitz constant of the gradient. For the quadratic $L(w) = \frac{1}{2} C w^2$, the gradient $L'(w) = Cw$ is exactly $C$-Lipschitz:

$$|L'(w) - L'(v)| = |Cw - Cv| = C|w - v|$$

### The Descent Lemma

For any function with $C$-Lipschitz gradients, the following inequality holds (called the descent lemma or the quadratic upper bound):

$$L(w - \alpha L'(w)) \leq L(w) - \alpha \left(1 - \frac{\alpha C}{2}\right) |L'(w)|^2$$

If $\alpha < 2/C$, then $1 - \alpha C/2 > 0$, and the right side is strictly less than $L(w)$ whenever $L'(w) \neq 0$. This guarantees that each gradient descent step decreases the loss.

This is the general proof that gradient descent with $\alpha < 2/C$ is a descent method for any function with $C$-Lipschitz gradients - not just quadratics.

### Sufficient Decrease and the 1/C Learning Rate

The sufficient decrease per step is maximized when the coefficient $\alpha(1 - \alpha C/2)$ is maximized. Taking the derivative with respect to $\alpha$ and setting it to zero:

$$\frac{d}{d\alpha}\left[\alpha - \frac{\alpha^2 C}{2}\right] = 1 - \alpha C = 0 \implies \alpha = \frac{1}{C}$$

So $\alpha = 1/C$ maximizes the guaranteed decrease per step, consistent with our earlier finding that $\alpha = 1/C$ is optimal for the quadratic.

### Connection to the Convergence Bound

The descent lemma provides a weaker but more general bound than the exact quadratic analysis. For the quadratic, we showed convergence requires $\alpha < 2/C$ exactly. The descent lemma guarantees sufficient decrease for $\alpha < 2/C$ for any smooth function, not just quadratics. The bound is tight: there exist smooth functions (specifically, quadratics) where $\alpha = 2/C$ leads to non-convergence.

## Convergence Rates in Detail

### Linear Convergence

For the quadratic loss with $\alpha \in (0, 2/C)$, the loss converges linearly (also called geometrically):

$$L(w_t) = (1 - \alpha C)^{2t} L(w_0)$$

The per-step contraction of the loss is $(1 - \alpha C)^2$. "Linear convergence" means the logarithm of the error decreases linearly with the iteration count:

$$\log L(w_t) = 2t \log|1 - \alpha C| + \log L(w_0)$$

This is the best convergence rate achievable by gradient descent on smooth, strongly convex functions.

### Sublinear Convergence on General Convex Functions

For convex but not strongly convex functions (where the Hessian can have zero eigenvalues), gradient descent achieves only sublinear convergence:

$$L(w_t) - L(w^*) \leq \frac{C \|w_0 - w^*\|^2}{2t}$$

The error decreases as $O(1/t)$ rather than $O(\rho^t)$. This is fundamentally slower - linear convergence reaches machine precision in a logarithmic number of steps, while sublinear convergence requires a polynomial number.

### Convergence Rate vs Learning Rate

The contraction factor $\rho = |1 - \alpha C|$ varies with $\alpha$:

- At $\alpha = 0$: $\rho = 1$ (no progress)
- As $\alpha$ increases from $0$: $\rho$ decreases linearly
- At $\alpha = 1/C$: $\rho = 0$ (optimal, one-step convergence)
- As $\alpha$ increases past $1/C$: $\rho$ increases linearly (but iterates now oscillate)
- At $\alpha = 2/C$: $\rho = 1$ (marginal, no convergence)
- For $\alpha > 2/C$: $\rho > 1$ (divergence)

The convergence rate is symmetric around the optimal $\alpha = 1/C$: learning rates $1/C - \delta$ and $1/C + \delta$ give the same contraction factor $|\delta C|$ but different qualitative behavior (monotone vs oscillatory).

## Connection to Adaptive Learning Rate Methods

### The Problem with a Single Learning Rate

In multivariable optimization, using a single scalar learning rate $\alpha$ for all parameters is suboptimal because different parameters may have different curvatures. The convergence bound $\alpha < 2/\lambda_{\max}$ is set by the worst-case (highest curvature) direction, which means low-curvature directions converge unnecessarily slowly.

### Per-Parameter Learning Rates

The ideal approach would be to use a different learning rate for each eigendirection of the Hessian: $\alpha_i = 1/\lambda_i$. This would make gradient descent converge in one step along every direction simultaneously. However, computing the eigenvectors and eigenvalues of the Hessian is prohibitively expensive for neural networks.

### Adam and RMSProp

Adaptive methods like Adam and RMSProp approximate per-parameter curvature using running averages of squared gradients:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} g_t$$

The denominator $\sqrt{v_t}$ acts as an estimate of $\sqrt{\lambda_i}$ (the square root of the curvature in direction $i$), so the effective learning rate for parameter $i$ is approximately $\alpha / \sqrt{\lambda_i}$, which partially adapts to the local curvature.

For the quadratic model with curvature $C$, if the running average correctly estimates $g^2 \approx C^2 w^2$, then $\sqrt{v} \approx C|w|$, and the effective learning rate is $\alpha / (C|w|)$. This is not exactly $1/C$ (the optimal), but it moves in the right direction by reducing the learning rate when the curvature is high.

### Learning Rate Warmup

The convergence bound $\alpha < 2/C$ explains why learning rate warmup is beneficial. At the start of training, the loss landscape may have high curvature (large $C$), requiring a small learning rate. As training progresses and the iterates approach a flatter region, the effective curvature decreases, allowing larger learning rates. Warmup gradually increases $\alpha$ to avoid early divergence while still reaching a good learning rate for the later phase.

## Practical Implications for Neural Network Training

### Learning Rate as the Most Important Hyperparameter

The analysis shows that the learning rate must satisfy $\alpha < 2/\lambda_{\max}(H)$ for stability. In practice, the maximum eigenvalue of the Hessian is unknown and changes during training. This is why:

- Grid search or learning rate range tests are standard practice
- Learning rates are typically set conservatively (below the stability boundary)
- Learning rate schedules that decrease $\alpha$ over time accommodate changes in curvature

### Batch Size and Effective Learning Rate

When using mini-batch SGD with batch size $B$, the gradient estimate has variance proportional to $1/B$. The effective dynamics are similar to gradient descent with an effective learning rate that scales with batch size. The linear scaling rule states that when you multiply the batch size by $k$, you should also multiply the learning rate by $k$ to maintain similar convergence behavior. This is a direct consequence of the analysis: larger batch sizes reduce gradient noise, allowing larger learning rates within the stability bound.

### Loss Spikes and Divergence

Training loss spikes - sudden, dramatic increases in loss - are symptoms of the learning rate exceeding the local stability bound $2/\lambda_{\max}$. This can happen when the optimizer enters a region of high curvature. The standard remedy is reducing the learning rate or using gradient clipping (which effectively reduces the step size when gradients are large).

### The Edge of Stability

Recent research has identified a phenomenon called the "edge of stability" in neural network training. When using a fixed learning rate $\alpha$, the maximum eigenvalue of the Hessian $\lambda_{\max}$ evolves during training and tends to hover near $2/\alpha$ - the stability boundary. The training process self-organizes so that the effective dynamics are near the boundary between convergence and divergence. This produces oscillatory but non-divergent behavior that is qualitatively different from both the convergent and divergent regimes of the quadratic model, and is an active area of research.

## Gradient Descent Dynamics and Phase Portraits

### The Dynamical Systems View

Gradient descent defines a discrete dynamical system:

$$w_{t+1} = g(w_t) = (1 - \alpha C) w_t$$

The map $g$ has a single fixed point at $w^* = 0$. The stability of this fixed point is determined by the derivative of $g$ at the fixed point:

$$g'(w^*) = 1 - \alpha C$$

The fixed point is stable (attracting) if $|g'(w^*)| < 1$ and unstable (repelling) if $|g'(w^*)| > 1$. This recovers the convergence condition $|1 - \alpha C| < 1$.

### Continuous-Time Limit

As $\alpha \to 0$, the discrete gradient descent update approaches the continuous gradient flow ODE:

$$\frac{dw}{dt} = -Cw$$

The solution is $w(t) = e^{-Ct} w_0$, which always converges to zero for $C > 0$. There is no stability constraint in continuous time - divergence is purely an artifact of discretization (taking finite steps). This is why infinitesimally small learning rates always converge but may be impractically slow.

### Orbits and Periodic Points

At $\alpha = 2/C$, the map becomes $g(w) = -w$, creating a period-2 orbit: $w_0, -w_0, w_0, -w_0, \ldots$. For $\alpha$ slightly above $2/C$, this orbit becomes unstable and the iterates spiral outward. For $\alpha$ slightly below $2/C$, the iterates spiral inward.

In higher-dimensional nonlinear systems (like neural network training), more complex periodic orbits and chaotic dynamics can emerge for large learning rates. The quadratic model captures only the simplest case, but the principle is the same: exceeding the stability bound leads to qualitatively different, often pathological, dynamics.

## Implementation Considerations

### Computing the Trajectory

The gradient descent loop for the quadratic is straightforward:

```
trajectory = [w0]
w = w0
for i in range(n_iters):
    grad = curvature * w
    w = w - alpha * grad
    trajectory.append(w)
```

For each of the three learning rates ($0.5/C$, $1.9/C$, $2.1/C$), this produces a trajectory of $n + 1$ values.

### Numerical Overflow in the Divergent Case

For the divergent trajectory ($\alpha = 2.1/C$), the weight grows as $|w_t| = 1.1^t |w_0|$ (since $|1 - 2.1| = 1.1$). For moderate $n$ (say, 50 iterations), this gives $|w_{50}| \approx 117 |w_0|$, which is well within floating-point range. For very large $n$, the divergent trajectory may overflow to infinity, which is the correct mathematical behavior.

### Verifying the Convergence Bound

The convergence bound $\alpha_{\max} = 2/C$ can be verified numerically by checking that:
- The convergent trajectory ($\alpha = 0.5/C$) approaches zero
- The oscillatory trajectory ($\alpha = 1.9/C$) oscillates but approaches zero
- The divergent trajectory ($\alpha = 2.1/C$) grows without bound

The final values of each trajectory provide a clear empirical confirmation of the theoretical bound.

### The Three Learning Rate Choices

The specific choices $0.5/C$, $1.9/C$, and $2.1/C$ are designed to demonstrate each regime clearly:

- $\alpha = 0.5/C$: Multiplier is $1 - 0.5 = 0.5$. Fast monotone convergence. After $t$ steps, $|w_t| = 0.5^t |w_0|$, halving the distance each step.

- $\alpha = 1.9/C$: Multiplier is $1 - 1.9 = -0.9$. Slow oscillatory convergence. After $t$ steps, $|w_t| = 0.9^t |w_0|$, reducing by only $10\%$ each step, and alternating sign.

- $\alpha = 2.1/C$: Multiplier is $1 - 2.1 = -1.1$. Oscillatory divergence. After $t$ steps, $|w_t| = 1.1^t |w_0|$, growing by $10\%$ each step, and alternating sign.
