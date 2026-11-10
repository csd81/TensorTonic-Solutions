# <span style="font-size: 20px;">Gradient Descent on a Non-Convex Loss</span>

## Convex vs Non-Convex Optimization

In convex optimization, every local minimum is a global minimum. This is the property that makes problems like linear regression and logistic regression "easy" - gradient descent is guaranteed to find the best solution regardless of initialization. The Hessian of a convex function is positive semi-definite everywhere, which means the loss surface has a single bowl shape with no ridges, valleys, or bumps to get trapped in.

Non-convex optimization is fundamentally harder. A non-convex loss function can have multiple local minima, local maxima, saddle points (in higher dimensions), and plateaus. Gradient descent can converge to any of these critical points depending on where it starts and how it moves. Most real-world machine learning problems - particularly neural network training - involve non-convex loss landscapes, which makes understanding non-convex optimization essential for practitioners.

The function $L(w) = w^4 - 3w^2 + 2$ is a simple one-dimensional non-convex function that exhibits the key phenomena: multiple local minima separated by a local maximum, with gradient descent converging to different solutions depending on initialization. Despite being far simpler than a neural network loss, it captures the essential challenges of non-convex optimization.

## Critical Points and the First Derivative

### Definition

A critical point (also called a stationary point) of a differentiable function $L(w)$ is a point $w^*$ where the gradient vanishes:

$$L'(w^*) = 0$$

At a critical point, the function has zero slope - it is neither increasing nor decreasing. Critical points are candidates for local minima and local maxima, but they could also be inflection points (in 1D) or saddle points (in higher dimensions).

### Finding Critical Points of Our Function

For $L(w) = w^4 - 3w^2 + 2$, the first derivative is:

$$L'(w) = 4w^3 - 6w$$

Setting this to zero:

$$4w^3 - 6w = 0$$

$$2w(2w^2 - 3) = 0$$

This gives us two factors. Either $w = 0$ or $2w^2 - 3 = 0$, which gives $w^2 = 3/2$, so $w = \pm\sqrt{3/2} = \pm\sqrt{6}/2 \approx \pm 1.2247$.

The three critical points, sorted in ascending order, are:

$$w_1^* = -\sqrt{3/2} \approx -1.2247, \quad w_2^* = 0, \quad w_3^* = \sqrt{3/2} \approx 1.2247$$

### Understanding the Factorization

The gradient $L'(w) = 2w(2w^2 - 3)$ is a product of a linear term $2w$ and a quadratic term $2w^2 - 3$. The linear term is zero at $w = 0$. The quadratic term is zero at $w = \pm\sqrt{3/2}$. Between the roots, the sign of $L'(w)$ changes, which tells us where the function is increasing vs decreasing:

- For $w < -\sqrt{3/2}$: $2w < 0$ and $2w^2 - 3 > 0$, so $L'(w) < 0$. The function is decreasing.
- For $-\sqrt{3/2} < w < 0$: $2w < 0$ and $2w^2 - 3 < 0$, so $L'(w) > 0$. The function is increasing.
- For $0 < w < \sqrt{3/2}$: $2w > 0$ and $2w^2 - 3 < 0$, so $L'(w) < 0$. The function is decreasing.
- For $w > \sqrt{3/2}$: $2w > 0$ and $2w^2 - 3 > 0$, so $L'(w) > 0$. The function is increasing.

This sign analysis confirms the critical point classification: the function decreases into $-\sqrt{3/2}$ and $\sqrt{3/2}$ (both minima), with a maximum at $w = 0$ separating them.

## The Second Derivative Test

### Theory

Once we find the critical points, we need to classify them. The second derivative test states:

- If $L''(w^*) > 0$: the function curves upward at $w^*$, so $w^*$ is a **local minimum**
- If $L''(w^*) < 0$: the function curves downward at $w^*$, so $w^*$ is a **local maximum**
- If $L''(w^*) = 0$: the test is inconclusive - $w^*$ could be an inflection point, a flat minimum, or a flat maximum

The second derivative measures the curvature of the function. Positive curvature means the function is concave up (bowl-shaped), and negative curvature means it is concave down (dome-shaped).

### Applying the Test

For $L(w) = w^4 - 3w^2 + 2$, the second derivative is:

$$L''(w) = 12w^2 - 6$$

Evaluating at each critical point:

**At $w = -\sqrt{3/2}$:**
$$L''(-\sqrt{3/2}) = 12 \cdot \frac{3}{2} - 6 = 18 - 6 = 12 > 0$$
Classification: **local minimum**

**At $w = 0$:**
$$L''(0) = 12 \cdot 0 - 6 = -6 < 0$$
Classification: **local maximum**

**At $w = \sqrt{3/2}$:**
$$L''(\sqrt{3/2}) = 12 \cdot \frac{3}{2} - 6 = 18 - 6 = 12 > 0$$
Classification: **local minimum**

### Function Values at Critical Points

The loss values at each critical point are:

$$L(-\sqrt{3/2}) = \left(\frac{3}{2}\right)^2 - 3 \cdot \frac{3}{2} + 2 = \frac{9}{4} - \frac{9}{2} + 2 = -\frac{1}{4} = -0.25$$

$$L(0) = 0 - 0 + 2 = 2$$

$$L(\sqrt{3/2}) = -0.25$$

Both minima have the same function value ($-0.25$), and the local maximum is at $L(0) = 2$. The energy barrier between the two minima is $2 - (-0.25) = 2.25$.

### A Note on Saddle Points

In one dimension, the second derivative test classifies critical points as local minima or local maxima (when the test is conclusive). True saddle points - points that are minima in some directions and maxima in others - require at least two dimensions. In higher dimensions, the Hessian matrix replaces the scalar second derivative, and saddle points occur when the Hessian has both positive and negative eigenvalues (i.e., is indefinite). For neural networks, saddle points are far more common than local maxima, and recent research suggests they are the primary obstacle to optimization, not local minima.

## Gradient Descent Algorithm

### The Update Rule

Gradient descent minimizes a function by iteratively taking steps in the direction of steepest descent (the negative gradient):

$$w_{t+1} = w_t - \alpha \cdot L'(w_t)$$

where $\alpha > 0$ is the learning rate (step size) and $L'(w_t)$ is the gradient evaluated at the current point.

For our function:

$$w_{t+1} = w_t - \alpha(4w_t^3 - 6w_t)$$

### Why Gradient Descent Works

The gradient $L'(w)$ points in the direction of steepest ascent. By moving in the opposite direction ($-L'(w)$), we take a step that locally decreases the function value. For sufficiently small $\alpha$, each step is guaranteed to reduce (or not increase) the function value. This is formalized by the sufficient decrease condition: if $\alpha$ is small enough, then $L(w_{t+1}) < L(w_t)$ whenever $L'(w_t) \neq 0$.

### The Trajectory

The trajectory is the sequence of points $[w_0, w_1, w_2, \ldots, w_T]$ visited during gradient descent. Starting from the initial point $w_0$, each subsequent point is determined by the update rule. The trajectory provides a complete history of the optimization process and reveals how the algorithm navigates the loss landscape.

For our non-convex function, the trajectory will show the algorithm sliding downhill from the initialization, possibly overshooting and oscillating before settling into a minimum. The key observation is that trajectories starting on different sides of the local maximum at $w = 0$ will converge to different minima.

## Initialization Dependence

### Basins of Attraction

Each local minimum has a basin of attraction: the set of initial points from which gradient descent converges to that minimum. For our function:

- **Left basin:** $w_0 < 0$ leads to convergence to $w^* = -\sqrt{3/2}$
- **Right basin:** $w_0 > 0$ leads to convergence to $w^* = +\sqrt{3/2}$
- **Boundary:** $w_0 = 0$ is exactly at the local maximum, where $L'(0) = 0$, so gradient descent stays at $w_0 = 0$ forever (an unstable equilibrium)

This is a simple example of a general phenomenon: the boundary between basins of attraction (called a separatrix) passes through saddle points and local maxima.

### Symmetry of the Loss

Our function $L(w) = w^4 - 3w^2 + 2$ is symmetric: $L(w) = L(-w)$. This means the two minima are mirror images of each other, and the basins of attraction are symmetric around $w = 0$. In real neural networks, the loss landscape has complex, high-dimensional symmetries (like permutation symmetry of neurons in a layer), leading to many equivalent minima.

### Near the Boundary

Starting very close to $w = 0$ but on the positive side is instructive. Near the local maximum, the gradient $L'(w) \approx -6w$ is small in magnitude, so gradient descent makes very slow initial progress. The negative gradient pushes $w$ away from zero (toward $+\sqrt{3/2}$), but the repulsive force is weak when $w$ is close to zero. The trajectory shows a slow exponential escape from the neighborhood of the unstable equilibrium, followed by rapid convergence once the algorithm enters the steep part of the basin.

### Sensitivity to Perturbation

At exactly $w_0 = 0$, gradient descent is stuck forever. But at $w_0 = 0 + \epsilon$ for any $\epsilon > 0$, it converges to the positive minimum, and at $w_0 = 0 - \epsilon$, it converges to the negative minimum. This extreme sensitivity to initial conditions near the boundary is characteristic of non-convex optimization and explains why stochastic methods (which add noise) can be more robust than deterministic gradient descent.

## The Role of Learning Rate

### Too Small

With a very small learning rate, gradient descent converges reliably but slowly. Each step moves only a tiny distance, requiring many iterations to reach the minimum. For our function, $\alpha = 0.001$ might require thousands of iterations.

### Just Right

A moderate learning rate balances speed and stability. For our function, $\alpha = 0.01$ works well: the algorithm converges within a few hundred iterations without oscillating.

### Too Large

A large learning rate can cause the algorithm to overshoot the minimum and oscillate, or even diverge. For a quartic function, the gradient grows as $w^3$, so points far from the origin have very large gradients. If $\alpha$ is too large, a step from $w = 2$ (where $L'(2) = 20$) could overshoot dramatically.

### The Stability Condition

For a quadratic function $L(w) = \frac{1}{2}cw^2$, gradient descent converges if and only if $\alpha < 2/c$. For our non-convex function, the local curvature varies: near the minima, $L''(\pm\sqrt{3/2}) = 12$, so the local stability condition is approximately $\alpha < 2/12 \approx 0.167$. Near the local maximum, $L''(0) = -6$, which does not impose a stability constraint (the maximum is repulsive).

## Connection to Neural Network Training

### Non-Convexity in Deep Learning

Neural network loss functions are non-convex due to the nonlinear activation functions and the overparameterization of the model. A two-layer network with ReLU activations already has a non-convex loss landscape, and deeper networks have increasingly complex loss surfaces with many critical points.

### Multiple Equivalent Minima

Like our toy function with two symmetric minima, neural networks typically have many global (or near-global) minima that achieve similar loss values but correspond to different weight configurations. These are often related by symmetry transformations (permuting neurons in a hidden layer, for example). The minimum that SGD finds depends on:

- **Initialization:** Random initialization breaks symmetry and determines the basin of attraction
- **Learning rate:** Affects which minima are reachable and how the trajectory explores the landscape
- **Mini-batch noise:** Stochastic gradient descent adds noise that can help escape shallow local minima
- **Momentum:** Helps the optimizer carry past small bumps in the loss surface

### The Loss Landscape Perspective

Recent research has shown that neural network loss landscapes, while non-convex, have favorable properties in the overparameterized regime: most local minima are approximately global minima (they have similar loss values), and saddle points (which can slow optimization) can be escaped with appropriate techniques like adding noise or using momentum. Our 1D example captures the "multiple minima" aspect but cannot capture the high-dimensional phenomena like saddle points and the blessing of dimensionality.

### Why Initialization Schemes Matter

The observation that different initializations lead to different solutions is the reason why initialization schemes like Xavier/Glorot initialization and He initialization are important in practice. These schemes set the initial weight scale based on the layer dimensions, ensuring that:

- Activations neither vanish nor explode through the layers
- Gradients flow properly during backpropagation
- The initial point is in a "good" basin of attraction

Poor initialization can lead to convergence to a bad local minimum, extremely slow convergence, or numerical instability.

## Global vs Local Optimality

### Definitions

A **global minimum** is a point $w^*$ such that $L(w^*) \leq L(w)$ for all $w$. A **local minimum** is a point where $L(w^*) \leq L(w)$ for all $w$ in some neighborhood of $w^*$.

For $L(w) = w^4 - 3w^2 + 2$, both $w = \pm\sqrt{3/2}$ are global minima (they achieve the same minimum value of $-0.25$, and $L(w) \to +\infty$ as $w \to \pm\infty$).

### The Challenge

In high-dimensional non-convex optimization, finding the global minimum is NP-hard in general. Gradient-based methods can only guarantee convergence to a local minimum (or more precisely, to a point where the gradient is approximately zero). The success of deep learning relies on the empirical observation that the local minima found by SGD are "good enough" - they generalize well to unseen data even if they are not globally optimal.

## Convergence Behavior

### Linear Convergence Near a Minimum

Near a local minimum $w^*$ with $L''(w^*) = c > 0$, the function is approximately quadratic: $L(w) \approx L(w^*) + \frac{c}{2}(w - w^*)^2$. Gradient descent on this quadratic converges linearly:

$$|w_{t+1} - w^*| = |1 - \alpha c| \cdot |w_t - w^*|$$

The convergence rate is $|1 - \alpha c|$, which is minimized when $\alpha = 1/c$ (giving convergence in one step for the quadratic approximation). For our function near $\pm\sqrt{3/2}$, $c = 12$, so the optimal local learning rate is $\alpha = 1/12 \approx 0.083$.

### Slow Escape from Unstable Equilibria

Near the local maximum at $w = 0$, the dynamics are repulsive: the gradient pushes the point away from zero, but the push is weak when $w$ is close to zero ($L'(w) \approx -6w$ near $w = 0$). This means starting near $w = 0$ results in an initial phase of slow exponential escape before the algorithm enters the steep part of the basin and converges quickly.

### Trajectory Shape

A typical trajectory from $w_0 = 2$ with $\alpha = 0.01$ would show:
1. Large initial steps (the gradient is large at $w = 2$: $L'(2) = 20$)
2. Rapid approach toward $\sqrt{3/2} \approx 1.2247$
3. Oscillation around the minimum (if steps overshoot)
4. Gradual convergence as the gradient becomes small near the minimum

## Practical Implementation

### Computing Critical Points

For this specific function, the critical points are computed analytically by solving $4w^3 - 6w = 0$. The roots are $w = 0$ and $w = \pm\sqrt{3/2}$. In code:

```
cp = sorted([-sqrt(3/2), 0.0, sqrt(3/2)])
```

### Classification

The second derivative $L''(w) = 12w^2 - 6$ is evaluated at each critical point. Positive means local minimum, negative means local maximum.

### Gradient Descent Loop

The gradient descent loop is straightforward:

```
trajectory = [w0]
w = w0
for i in range(n_iters):
    grad = 4*w**3 - 6*w
    w = w - lr * grad
    trajectory.append(w)
```

The trajectory has length $n\_iters + 1$ (including the initial point).

### Numerical Precision

Using `float64` arithmetic ensures that the trajectory is computed accurately even for many iterations. The critical points should be computed with `np.sqrt(1.5)` for maximum precision.
