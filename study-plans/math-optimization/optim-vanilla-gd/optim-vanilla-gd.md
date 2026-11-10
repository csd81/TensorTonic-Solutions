# <span style="font-size: 20px;">Vanilla Gradient Descent</span>

## What is Gradient Descent?

Gradient descent is the foundational optimization algorithm behind nearly all of modern machine learning. It is a first-order iterative method for finding a local minimum of a differentiable function by taking repeated steps in the direction of steepest descent (the negative gradient).

The update rule for a parameter vector $w$ is:

$$
w_{t+1} = w_t - \alpha \nabla f(w_t)
$$

where $\alpha > 0$ is the learning rate and $\nabla f(w_t)$ is the gradient at the current point. This is applied repeatedly until convergence.

## The Learning Rate

The learning rate $\alpha$ controls step size:

- **Too small:** Many iterations needed, impractically slow convergence
- **Just right:** Steady progress, balanced speed and stability
- **Too large:** Overshooting, oscillation, or divergence

For a function with Lipschitz-continuous gradients (constant $L$), convergence requires:

$$
\alpha < \frac{2}{L}
$$

## The Function

We minimize $f(x, y) = x^2 + 3y^2$, a simple elliptic paraboloid. The Hessian is:

$$
H = \begin{pmatrix} 2 & 0 \\ 0 & 6 \end{pmatrix}
$$

The eigenvalues are 2 and 6. The condition number is $\kappa = 6/2 = 3$, meaning the curvature in y is 3 times steeper than in x. This creates elliptical contours and different convergence rates along each axis.

## Computing the Gradient

For $f(x, y) = x^2 + 3y^2$, applying the power rule:

$$
\frac{\partial f}{\partial x} = 2x \qquad \frac{\partial f}{\partial y} = 6y
$$

At the minimum $(0, 0)$, the gradient is exactly zero.

## The 2D Update Rule

In two dimensions, the update becomes a pair of simultaneous updates:

$$
x_{t+1} = x_t - \alpha \cdot 2x_t = (1 - 2\alpha) x_t
$$

$$
y_{t+1} = y_t - \alpha \cdot 6y_t = (1 - 6\alpha) y_t
$$

Both partial derivatives must be computed at the current point $(x_t, y_t)$ before either coordinate is updated.

## Convergence Analysis

Since $f$ is quadratic, the solution is exact: $x_t = (1 - 2\alpha)^t x_0$ and $y_t = (1 - 6\alpha)^t y_0$. Convergence requires both multipliers to have magnitude less than 1:

- $|1 - 2\alpha| < 1$ gives $\alpha < 1$ (x converges)
- $|1 - 6\alpha| < 1$ gives $\alpha < 1/3$ (y converges)

The binding constraint is the steeper direction: $\alpha < 1/3$. This illustrates a general principle: the maximum stable learning rate is $2/\lambda_{\max}$ where $\lambda_{\max}$ is the largest eigenvalue of the Hessian.

The optimal learning rate (fastest convergence for the slowest coordinate) balances both directions. For this function, $\alpha = 1/4$ gives multipliers $0.5$ and $-0.5$, so both coordinates halve in magnitude each step.

## Anisotropic Convergence

With learning rate $\alpha = 0.1$:

- x multiplier: $1 - 0.2 = 0.8$ (slow decay)
- y multiplier: $1 - 0.6 = 0.4$ (fast decay)

The y-coordinate converges much faster because its curvature is steeper. This creates the characteristic "narrow valley" behavior that motivates adaptive methods like Adam, which scale each coordinate's learning rate by the inverse of its gradient magnitude.

## Vanilla vs Advanced Gradient Descent

"Vanilla" GD uses only the current gradient with a fixed learning rate. Its limitations motivate all modern optimizers:

- **Momentum** adds exponential moving average of past gradients
- **AdaGrad** adapts learning rate per-parameter based on gradient history
- **RMSProp** fixes AdaGrad's vanishing learning rate with exponential moving average
- **Adam** combines momentum with RMSProp plus bias correction

Understanding vanilla GD thoroughly is essential before studying these methods, as every advanced optimizer is a direct modification of the vanilla update rule.