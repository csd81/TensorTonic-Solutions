# <span style="font-size: 20px;">Directional Derivative & Steepest Descent Direction</span>

## Scalar Fields and the Gradient Vector

A scalar field is a function that maps a vector to a scalar: $L : \mathbb{R}^d \to \mathbb{R}$. In machine learning, loss functions are scalar fields that take a parameter vector $\mathbf{w} \in \mathbb{R}^d$ and return a scalar loss value $L(\mathbf{w})$.

The gradient of $L$ at a point $\mathbf{w}$ is the vector of partial derivatives:

$$\nabla L(\mathbf{w}) = \begin{pmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \\ \frac{\partial L}{\partial w_d} \end{pmatrix}
$$

Each component $\frac{\partial L}{\partial w_j}$ measures the rate of change of $L$ when only $w_j$ is varied, holding all other components fixed. The gradient packages all these individual rates of change into a single vector that captures the complete first-order local behavior of $L$.

### The Gradient of a Quadratic Form

For this problem, we work with the specific loss function:

$$L(w_1, w_2) = w_1^2 + 3w_2^2$$

This is an elliptic paraboloid - a bowl-shaped surface with its minimum at the origin $(0, 0)$ where $L = 0$. The level curves (contours where $L$ is constant) are ellipses:

$$w_1^2 + 3w_2^2 = c \quad \text{for constant } c > 0$$

These ellipses are elongated along the $w_1$-axis because the coefficient of $w_2^2$ is larger (3 vs 1), meaning $L$ increases faster in the $w_2$ direction.

Computing the gradient:

$$\frac{\partial L}{\partial w_1} = 2w_1, \quad \frac{\partial L}{\partial w_2} = 6w_2$$

$$\nabla L(w_1, w_2) = \begin{pmatrix} 2w_1 \\ 6w_2 \end{pmatrix}
$$

At the point $(1, 1)$, for example, the gradient is $(2, 6)$. The larger second component reflects the fact that the loss surface is steeper in the $w_2$ direction.

### Gradient and Level Curves

A fundamental property of the gradient is that it is always perpendicular (normal) to the level curves at any point. This can be proven as follows: a level curve is defined by $L(\mathbf{w}) = c$. If $\mathbf{w}(t)$ is a curve lying on this level surface (so $L(\mathbf{w}(t)) = c$ for all $t$), then by the chain rule:

$$\frac{d}{dt}L(\mathbf{w}(t)) = \nabla L(\mathbf{w}(t)) \cdot \mathbf{w}'(t) = 0$$

Since $\mathbf{w}'(t)$ is tangent to the level curve, and the dot product with $\nabla L$ is zero, the gradient must be perpendicular to the level curve.

For our quadratic loss, the level curves are ellipses, and the gradient at any point is perpendicular to the ellipse at that point. This perpendicularity is what makes the gradient the direction of steepest change.

## The Directional Derivative

The partial derivatives $\frac{\partial L}{\partial w_j}$ measure rates of change along the coordinate axes. But what if we want to know how $L$ changes along an arbitrary direction?

### Definition

The directional derivative of $L$ at point $\mathbf{w}$ in the direction of a unit vector $\mathbf{v}$ (where $\|\mathbf{v}\| = 1$) is:

$$D_\mathbf{v} L(\mathbf{w}) = \lim_{t \to 0} \frac{L(\mathbf{w} + t\mathbf{v}) - L(\mathbf{w})}{t}$$

This measures the instantaneous rate of change of $L$ as we move from $\mathbf{w}$ in the direction $\mathbf{v}$. The requirement that $\|\mathbf{v}\| = 1$ ensures the rate of change is per unit distance (not influenced by the magnitude of the direction vector).

### Computation via the Gradient

For a differentiable function, the directional derivative equals the dot product of the gradient with the direction:

$$D_\mathbf{v} L(\mathbf{w}) = \nabla L(\mathbf{w}) \cdot \mathbf{v} = \sum_{j=1}^d \frac{\partial L}{\partial w_j} v_j$$

**Proof**: Using the first-order Taylor expansion:

$$L(\mathbf{w} + t\mathbf{v}) \approx L(\mathbf{w}) + t \nabla L(\mathbf{w}) \cdot \mathbf{v} + O(t^2)$$

Therefore:

$$
\begin{aligned}
D_\mathbf{v} L(\mathbf{w}) &= \lim_{t \to 0} \frac{L(\mathbf{w}) + t \nabla L(\mathbf{w}) \cdot \mathbf{v} + O(t^2) - L(\mathbf{w})}{t} \\
&= \nabla L(\mathbf{w}) \cdot \mathbf{v}
\end{aligned}
$$

This formula is extremely useful because it reduces the directional derivative computation to a simple dot product, avoiding the need to evaluate the limit definition.

### Interpreting the Sign

The directional derivative $D_\mathbf{v} L$ can be positive, negative, or zero:

- $D_\mathbf{v} L > 0$: the loss increases as we move in direction $\mathbf{v}$
- $D_\mathbf{v} L < 0$: the loss decreases as we move in direction $\mathbf{v}$
- $D_\mathbf{v} L = 0$: the loss is stationary to first order in direction $\mathbf{v}$ (we are moving along a level curve)

## The Steepest Descent Direction

The central question of optimization is: in which direction should we move to decrease the loss as rapidly as possible?

### The Optimization Problem

We want to find the unit vector $\mathbf{v}^*$ that minimizes the directional derivative:

$$\mathbf{v}^* = \arg\min_{\|\mathbf{v}\|=1} D_\mathbf{v} L(\mathbf{w}) = \arg\min_{\|\mathbf{v}\|=1} \nabla L(\mathbf{w}) \cdot \mathbf{v}$$

### Solution via the Cauchy-Schwarz Inequality

The Cauchy-Schwarz inequality states that for any vectors $\mathbf{a}$ and $\mathbf{b}$:

$$|\mathbf{a} \cdot \mathbf{b}| \leq \|\mathbf{a}\| \|\mathbf{b}\|$$

with equality if and only if $\mathbf{b} = \alpha \mathbf{a}$ for some scalar $\alpha$.

Applying this with $\mathbf{a} = \nabla L$ and $\mathbf{b} = \mathbf{v}$ (where $\|\mathbf{v}\| = 1$):

$$-\|\nabla L\| \leq \nabla L \cdot \mathbf{v} \leq \|\nabla L\|$$

The **lower bound** $-\|\nabla L\|$ is achieved when $\mathbf{v}$ is antiparallel to $\nabla L$:

$$\mathbf{v}^* = -\frac{\nabla L}{\|\nabla L\|}$$

The **upper bound** $+\|\nabla L\|$ is achieved when $\mathbf{v}$ is parallel to $\nabla L$:

$$\mathbf{v}_{\max} = +\frac{\nabla L}{\|\nabla L\|}$$

### The Steepest Descent Theorem

**Theorem**: The direction of steepest descent of $L$ at $\mathbf{w}$ is $\mathbf{v}^* = -\nabla L / \|\nabla L\|$, and the corresponding rate of decrease is $-\|\nabla L\|$.

**Proof**: We need to show that $\nabla L \cdot \mathbf{v}$ is minimized over all unit vectors when $\mathbf{v} = -\nabla L / \|\nabla L\|$.

Let $\mathbf{g} = \nabla L(\mathbf{w})$ and let $\theta$ be the angle between $\mathbf{g}$ and $\mathbf{v}$. Then:

$$D_\mathbf{v} L = \mathbf{g} \cdot \mathbf{v} = \|\mathbf{g}\| \|\mathbf{v}\| \cos\theta = \|\mathbf{g}\| \cos\theta$$

since $\|\mathbf{v}\| = 1$. This expression is minimized when $\cos\theta = -1$, i.e., $\theta = \pi$, meaning $\mathbf{v}$ points in the opposite direction of $\mathbf{g}$:

$$\mathbf{v}^* = -\frac{\mathbf{g}}{\|\mathbf{g}\|} = -\frac{\nabla L}{\|\nabla L\|}$$

The minimum directional derivative value is:

$$D_{\mathbf{v}^*} L = \|\nabla L\| \cdot (-1) = -\|\nabla L\|$$

This proves that the negative gradient direction is the direction of steepest descent, and the magnitude of the gradient equals the steepest rate of decrease.

### The Special Case: Zero Gradient

When $\nabla L = \mathbf{0}$ (the gradient vanishes), $D_\mathbf{v} L = 0$ for all directions $\mathbf{v}$. The loss is stationary to first order - there is no direction that produces a first-order decrease. Such points are called **critical points** (or stationary points). For our quadratic $L(w_1, w_2) = w_1^2 + 3w_2^2$, the only critical point is the origin $(0, 0)$, which is the global minimum.

## Why This Matters for Gradient Descent

Gradient descent uses the update rule:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

The steepest descent theorem provides the mathematical justification for this choice. By moving in the direction $-\nabla L$, gradient descent moves in the direction that reduces $L$ most rapidly (per unit step size).

The learning rate $\alpha$ controls the step length. The update can be rewritten as:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - (\alpha \|\nabla L\|) \cdot \frac{\nabla L}{\|\nabla L\|}$$

This decomposes the update into a step of length $\alpha \|\nabla L\|$ in the steepest descent direction $-\nabla L / \|\nabla L\|$.

### The Zigzag Problem

While the steepest descent direction is locally optimal, it can lead to globally inefficient paths. For our elliptic loss $L = w_1^2 + 3w_2^2$, the level curves are ellipses with a 3:1 aspect ratio. The gradient at most points does not point directly toward the minimum; instead, it points perpendicular to the (elliptical) contour, which causes gradient descent to zigzag.

Consider starting at $(3, 1)$. The gradient is $(6, 6)$, pointing at 45 degrees. But the minimum is at the origin, which is in the direction $(-3, -1)$, at about 18 degrees from the $-w_1$ axis. The steepest descent direction and the direction to the minimum are different because the loss landscape is anisotropic (not equally scaled in all directions).

This zigzag behavior is a well-known limitation of steepest descent and motivates advanced optimizers like momentum, Adam, and natural gradient, which account for the curvature of the loss landscape.

## Computing the Steepest Descent Direction in Practice

Given a loss $L(\mathbf{w})$ and a point $\mathbf{w}$, the procedure for finding the steepest descent direction is:

1. Compute the gradient $\nabla L(\mathbf{w})$
2. Compute the gradient norm $\|\nabla L\|$
3. The steepest descent direction is $\mathbf{v}^* = -\nabla L / \|\nabla L\|$
4. The directional derivative in this direction is $D_{\mathbf{v}^*} L = -\|\nabla L\|$

For any other unit vector $\mathbf{v}$, the directional derivative is $D_\mathbf{v} L = \nabla L \cdot \mathbf{v}$. By the Cauchy-Schwarz inequality, this satisfies:

$$-\|\nabla L\| \leq D_\mathbf{v} L \leq +\|\nabla L\|$$

The lower bound is achieved only in the steepest descent direction $\mathbf{v}^*$, and the upper bound only in the steepest ascent direction $-\mathbf{v}^*$. Moving along coordinate axes or arbitrary diagonal directions always produces directional derivatives strictly between these extremes (unless the gradient happens to align with that direction).

For the quadratic loss $L = w_1^2 + 3w_2^2$, the gradient $\nabla L = (2w_1, 6w_2)$ has a larger component in $w_2$ than $w_1$ (due to the coefficient $3$), so the steepest descent direction tilts more toward the $w_2$-axis than the $w_1$-axis at most points. This asymmetry reflects the anisotropy of the loss surface.

## Worked Example: Directional Derivative Along a Level Curve

Consider the same loss at $\mathbf{w} = (1, 1)$ where $L = 1 + 3 = 4$. The level curve $L = 4$ is the ellipse $w_1^2 + 3w_2^2 = 4$.

The tangent to this ellipse at $(1, 1)$ can be found by implicit differentiation:

$$2w_1 + 6w_2 \frac{dw_2}{dw_1} = 0 \implies \frac{dw_2}{dw_1} = -\frac{w_1}{3w_2} = -\frac{1}{3}$$

So the tangent direction is proportional to $(1, -1/3)$. Normalizing:

$$\|(1, -1/3)\| = \sqrt{1 + 1/9} = \sqrt{10/9} = \frac{\sqrt{10}}{3}$$

$$
\begin{aligned}
\mathbf{v}_{\text{tangent}} &= \frac{(1, -1/3)}{\sqrt{10}/3} \\
&= \frac{3}{\sqrt{10}}(1, -1/3) = \left(\frac{3}{\sqrt{10}}, -\frac{1}{\sqrt{10}}\right) \\
&\approx (0.9487, -0.3162)
\end{aligned}
$$

The directional derivative along this tangent:

$$D_{\mathbf{v}_{\text{tangent}}} L = 2 \cdot \frac{3}{\sqrt{10}} + 6 \cdot \left(-\frac{1}{\sqrt{10}}\right) = \frac{6 - 6}{\sqrt{10}} = 0$$

As expected, the directional derivative along a level curve tangent is zero - moving along the contour does not change the loss. This confirms that the gradient $(2, 6)$ is perpendicular to the tangent direction $(\frac{3}{\sqrt{10}}, -\frac{1}{\sqrt{10}})$:

$$(2, 6) \cdot \left(\frac{3}{\sqrt{10}}, -\frac{1}{\sqrt{10}}\right) = \frac{6 - 6}{\sqrt{10}} = 0 \quad \checkmark$$

## The Dot Product Perspective

The directional derivative formula $D_\mathbf{v} L = \nabla L \cdot \mathbf{v} = \|\nabla L\| \cos\theta$ reveals that the directional derivative depends only on the angle $\theta$ between the gradient and the direction:

- $\theta = 0$ (same direction as gradient): $D_\mathbf{v} L = +\|\nabla L\|$ (maximum increase)
- $\theta = \pi/2$ (perpendicular to gradient): $D_\mathbf{v} L = 0$ (along level curve)
- $\theta = \pi$ (opposite to gradient): $D_\mathbf{v} L = -\|\nabla L\|$ (maximum decrease)

This means the directional derivative varies as $\cos\theta$ around the unit circle of directions. It is a smooth, sinusoidal function of angle, with a single maximum (along $\nabla L$) and a single minimum (along $-\nabla L$).

For any direction making an acute angle with the negative gradient ($\theta > \pi/2$ from the gradient, equivalently $\theta < \pi/2$ from the negative gradient), the directional derivative is negative, meaning the loss decreases in that direction. The set of "descent directions" forms a half-space in direction space, with the level-curve tangent as the boundary.

## Second-Order Analysis: Is Steepest Descent Optimal?

The steepest descent direction is optimal among all directions for a single infinitesimal step. But for a finite step, the story is more nuanced.

The second-order Taylor expansion of $L$ around $\mathbf{w}$ is:

$$
\begin{aligned}
L(\mathbf{w} + \boldsymbol{\delta}) &\approx L(\mathbf{w}) + \nabla L^\top \boldsymbol{\delta} \\
&\quad + \frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H} \boldsymbol{\delta}
\end{aligned}
$$

where $\mathbf{H}$ is the Hessian matrix:

$$\mathbf{H} = \begin{pmatrix} \frac{\partial^2 L}{\partial w_1^2} & \frac{\partial^2 L}{\partial w_1 \partial w_2} \\ \frac{\partial^2 L}{\partial w_2 \partial w_1} & \frac{\partial^2 L}{\partial w_2^2} \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 6 \end{pmatrix}
$$

For our quadratic loss, the Hessian is constant (independent of $\mathbf{w}$), and the Taylor expansion is exact (no higher-order terms).

Newton's method accounts for the curvature by using the step $\boldsymbol{\delta} = -\mathbf{H}^{-1}\nabla L$:

$$\boldsymbol{\delta}_{\text{Newton}} = -\begin{pmatrix} 1/2 & 0 \\ 0 & 1/6 \end{pmatrix}\begin{pmatrix} 2w_1 \\ 6w_2 \end{pmatrix} = \begin{pmatrix} -w_1 \\ -w_2 \end{pmatrix}
$$

This step goes directly to the minimum $(0, 0)$ in a single iteration, regardless of the starting point. The steepest descent direction $-\nabla L = (-2w_1, -6w_2)$ would also reach the minimum eventually but typically takes many zigzagging steps.

The difference between steepest descent and Newton's method illustrates a key tension in optimization: steepest descent is easy to compute (just the gradient) but can be inefficient for ill-conditioned problems, while Newton's method accounts for curvature but requires computing and inverting the Hessian.

## Lagrange Multiplier Derivation

An alternative proof of the steepest descent theorem uses Lagrange multipliers. We want to minimize $f(\mathbf{v}) = \nabla L \cdot \mathbf{v}$ subject to the constraint $g(\mathbf{v}) = \|\mathbf{v}\|^2 - 1 = 0$.

The Lagrangian is:

$$\mathcal{L}(\mathbf{v}, \lambda) = \nabla L \cdot \mathbf{v} - \lambda(\|\mathbf{v}\|^2 - 1)$$

Setting $\nabla_\mathbf{v} \mathcal{L} = 0$:

$$\nabla L - 2\lambda \mathbf{v} = 0 \implies \mathbf{v} = \frac{\nabla L}{2\lambda}$$

Using the constraint $\|\mathbf{v}\| = 1$:

$$\left\|\frac{\nabla L}{2\lambda}\right\| = 1 \implies |2\lambda| = \|\nabla L\| \implies \lambda = \pm\frac{\|\nabla L\|}{2}$$

For $\lambda = +\|\nabla L\|/2$: $\mathbf{v} = +\nabla L / \|\nabla L\|$ (steepest ascent, the maximum)

For $\lambda = -\|\nabla L\|/2$: $\mathbf{v} = -\nabla L / \|\nabla L\|$ (steepest descent, the minimum)

This Lagrange multiplier approach gives the same result but demonstrates the connection between constrained optimization and the steepest descent theorem.

## Directional Derivatives in Higher Dimensions

While our problem uses $d = 2$, the theory extends to any dimension. In $\mathbb{R}^d$:

$$D_\mathbf{v} L = \nabla L \cdot \mathbf{v} = \sum_{j=1}^d \frac{\partial L}{\partial w_j} v_j$$

The steepest descent direction is always $-\nabla L / \|\nabla L\|$, and the steepest rate of decrease is always $-\|\nabla L\|$.

In high-dimensional spaces (deep learning uses $d$ in the millions or billions), the gradient lives in a $d$-dimensional space, and the "cone" of descent directions (those with $D_\mathbf{v} L < 0$) forms a half-space in $\mathbb{R}^d$. The fraction of random directions that are descent directions is always exactly 50%, regardless of dimension.

However, the fraction of directions that produce "significant" descent (say, at least half the steepest descent rate) decreases with dimension. In very high dimensions, most random directions are nearly perpendicular to the gradient, producing near-zero directional derivatives. The gradient direction becomes increasingly "special" as the dimensionality grows.

## Practical Implications for Optimization

Understanding directional derivatives and steepest descent connects to several practical concepts in ML optimization:

**Learning rate sensitivity**: The steepest descent rate $-\|\nabla L\|$ varies across the loss landscape. Near a minimum, $\|\nabla L\|$ is small, and large learning rates can overshoot. Far from the minimum, $\|\nabla L\|$ is large, and the same learning rate produces big steps.

**Gradient clipping**: If $\|\nabla L\|$ is very large (exploding gradients), the update $-\alpha \nabla L$ can be enormous and destabilize training. Gradient clipping rescales the gradient when $\|\nabla L\| > \text{threshold}$, effectively limiting the step size while preserving the steepest descent direction.

**Adaptive methods**: Optimizers like Adam and RMSProp modify the effective direction and step size for each parameter independently. They move in a direction that is NOT the steepest descent direction but is often more effective because they account for the curvature of the loss landscape.

**Momentum**: Instead of always following the steepest descent direction, momentum methods accumulate a running average of past gradients. This smooths out the zigzag behavior and can dramatically accelerate convergence on ill-conditioned problems.

All of these techniques are variations on the fundamental idea that the gradient tells us the direction and rate of steepest change, and intelligent optimization uses this information to navigate the loss landscape efficiently.
