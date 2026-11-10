# <span style="font-size: 20px;">Lasso Regression using ISTA</span>

## The Lasso Objective

Lasso adds an L1 penalty to least squares, promoting sparse solutions (some weights driven exactly to zero):

$$
J(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \lambda \|w\|_1
$$

The L1 term is non-differentiable at zero, so standard gradient descent cannot be applied directly.

## Proximal Gradient Descent

The objective splits into a smooth part $f(w) = \frac{1}{2n}\|Xw - y\|_2^2$ and a non-smooth part $g(w) = \lambda\|w\|_1$. ISTA (Iterative Shrinkage-Thresholding Algorithm) handles this by alternating a gradient step on $f$ with a proximal step for $g$:

1. Gradient step: $w^{1/2} = w_k - t \cdot \nabla f(w_k)$
2. Proximal step: $w_{k+1} = \text{prox}_{t \cdot g}(w^{1/2})$

## Soft-Thresholding

The proximal operator for the L1 norm is the soft-thresholding function, applied element-wise:

$$
S_\tau(x) = \text{sign}(x) \cdot \max(|x| - \tau, 0)
$$

This shrinks values toward zero and sets small values exactly to zero, which is how Lasso achieves sparsity.

## Step Size

The step size $t = 1/L$ guarantees convergence, where $L$ is the Lipschitz constant of $\nabla f$. For the least-squares gradient $\nabla f(w) = X^T(Xw - y)/n$, the Lipschitz constant is the largest eigenvalue of $X^TX/n$.

## Convergence

ISTA converges at rate $O(1/k)$ in objective value. Its accelerated variant FISTA (with Nesterov-style momentum) achieves $O(1/k^2)$, but standard ISTA is the foundation.