# <span style="font-size: 20px;">Gradient of Multivariate Linear Regression Loss</span>

## Linear Regression: The Foundation of Machine Learning

Linear regression is the simplest and most fundamental model in machine learning. Given a dataset of $n$ input-output pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where each $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector and $y_i \in \mathbb{R}$ is a scalar target, linear regression assumes the relationship:

$$\hat{y}_i = \mathbf{w}^\top \mathbf{x}_i = \sum_{j=1}^d w_j x_{ij}$$

where $\mathbf{w} \in \mathbb{R}^d$ is the weight vector (also called the parameter vector or coefficient vector). The model predicts $\hat{y}_i$ as a linear combination of the input features.

In matrix notation, we stack all inputs into a design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ where row $i$ is $\mathbf{x}_i^\top$, and all targets into a vector $\mathbf{y} \in \mathbb{R}^n$. The predictions for all $n$ samples are then:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

This compact notation replaces $n$ individual dot products with a single matrix-vector multiplication.

### Why Start Here?

Linear regression is the starting point for understanding gradients in ML because:
- The loss function is a simple quadratic in $\mathbf{w}$, so its gradient has a clean closed-form expression
- The gradient computation involves the core operations (matrix transposes, matrix-vector products) that appear in all deeper models
- It introduces the concept of comparing analytical gradients against numerical gradients, a debugging technique used throughout deep learning
- Every more complex model (logistic regression, neural networks, transformers) builds on these same gradient computations, just with additional nonlinearities

## The Mean Squared Error Loss Function

The most common loss function for linear regression is the sum of squared errors (SSE), also sometimes written as the mean squared error (MSE) with a $1/n$ factor:

$$L(\mathbf{w}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2$$

In matrix notation, defining the residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w}$:

$$
\begin{aligned}
L(\mathbf{w}) &= \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 \\
&= (\mathbf{y} - \mathbf{X}\mathbf{w})^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) \\
&= \mathbf{r}^\top \mathbf{r}
\end{aligned}
$$

The loss $L(\mathbf{w})$ is a scalar-valued function of the $d$-dimensional vector $\mathbf{w}$. It measures the total squared discrepancy between predictions and targets. The goal of training is to find $\mathbf{w}^*$ that minimizes $L$.

### Properties of the SSE Loss

The SSE loss has several important mathematical properties:

1. **Non-negativity**: $L(\mathbf{w}) \geq 0$ for all $\mathbf{w}$, since it is a sum of squares.

2. **Convexity**: $L(\mathbf{w})$ is a convex function of $\mathbf{w}$, which means any local minimum is also a global minimum. This follows from the fact that $L$ is a quadratic function with a positive semi-definite Hessian matrix (we will explore the Hessian in a later problem).

3. **Differentiability**: $L(\mathbf{w})$ is infinitely differentiable (smooth) everywhere, so the gradient exists at every point.

4. **Quadratic structure**: Expanding the loss, we get:

$$L(\mathbf{w}) = \mathbf{y}^\top\mathbf{y} - 2\mathbf{y}^\top\mathbf{X}\mathbf{w} + \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}$$

This is a quadratic form in $\mathbf{w}$, analogous to $f(x) = ax^2 + bx + c$ in one dimension. The matrix $\mathbf{X}^\top\mathbf{X}$ plays the role of $a$, the vector $-2\mathbf{X}^\top\mathbf{y}$ plays the role of $b$, and the scalar $\mathbf{y}^\top\mathbf{y}$ plays the role of $c$.

## The Gradient: Definition and Intuition

The gradient of a scalar-valued function $L : \mathbb{R}^d \to \mathbb{R}$ is the vector of partial derivatives:

$$\nabla L(\mathbf{w}) = \begin{pmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \\ \frac{\partial L}{\partial w_d} \end{pmatrix}
$$

Each component $\frac{\partial L}{\partial w_j}$ tells us how $L$ changes when we perturb $w_j$ by a small amount while keeping all other weights fixed. The gradient vector points in the direction of steepest increase of $L$, and its negative $-\nabla L$ points in the direction of steepest decrease.

### The Gradient as a Linear Approximation

The gradient enables a first-order Taylor approximation of $L$ near $\mathbf{w}$:

$$L(\mathbf{w} + \boldsymbol{\delta}) \approx L(\mathbf{w}) + \nabla L(\mathbf{w})^\top \boldsymbol{\delta}$$

This approximation is accurate when $\|\boldsymbol{\delta}\|$ is small. It tells us that the change in loss is approximately the dot product of the gradient with the perturbation direction. This is the mathematical foundation of gradient descent: moving in the direction $-\nabla L$ (appropriately scaled) reduces the loss.

## Deriving the Gradient of the SSE Loss

We derive $\nabla L$ two ways: component-wise (for clarity) and using matrix calculus (for elegance and efficiency).

### Method 1: Component-wise Derivation

The loss function is:

$$L(\mathbf{w}) = \sum_{i=1}^n (y_i - \sum_{j=1}^d w_j x_{ij})^2$$

To find $\frac{\partial L}{\partial w_k}$ for a specific component $k$, we apply the chain rule:

$$
\begin{aligned}
\frac{\partial L}{\partial w_k} &= \sum_{i=1}^n 2\!\left(y_i - \sum_{j=1}^d w_j x_{ij}\right) \\
&\quad \cdot \frac{\partial}{\partial w_k}\!\left(y_i - \sum_{j=1}^d w_j x_{ij}\right)
\end{aligned}
$$

The inner derivative is:

$$\frac{\partial}{\partial w_k}\left(y_i - \sum_{j=1}^d w_j x_{ij}\right) = -x_{ik}$$

because only the $j = k$ term in the sum depends on $w_k$. Substituting:

$$
\begin{aligned}
\frac{\partial L}{\partial w_k} &= \sum_{i=1}^n 2(y_i - \mathbf{w}^\top \mathbf{x}_i)(-x_{ik}) \\
&= -2\sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i) x_{ik}
\end{aligned}
$$

This can be recognized as $-2$ times the dot product of the $k$-th column of $\mathbf{X}$ with the residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w}$. Since the $k$-th column of $\mathbf{X}$ is the $k$-th row of $\mathbf{X}^\top$, stacking all components gives:

$$\nabla L(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

### Method 2: Matrix Calculus Derivation

Starting from the expanded quadratic form:

$$L(\mathbf{w}) = \mathbf{y}^\top\mathbf{y} - 2\mathbf{y}^\top\mathbf{X}\mathbf{w} + \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}$$

We apply the following matrix calculus rules:

**Rule 1**: For a constant vector $\mathbf{a}$, $\nabla_\mathbf{w}(\mathbf{a}^\top\mathbf{w}) = \mathbf{a}$

**Rule 2**: For a symmetric matrix $\mathbf{A}$, $\nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{A}\mathbf{w}) = 2\mathbf{A}\mathbf{w}$

**Rule 3**: The gradient of a constant (with respect to $\mathbf{w}$) is zero.

Applying these rules term by term:

- $\nabla_\mathbf{w}(\mathbf{y}^\top\mathbf{y}) = \mathbf{0}$ (constant, Rule 3)
- $\nabla_\mathbf{w}(-2\mathbf{y}^\top\mathbf{X}\mathbf{w}) = -2\mathbf{X}^\top\mathbf{y}$ (Rule 1, with $\mathbf{a} = \mathbf{X}^\top\mathbf{y}$, noting that $\mathbf{y}^\top\mathbf{X}\mathbf{w} = (\mathbf{X}^\top\mathbf{y})^\top\mathbf{w}$)
- $\nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}) = 2\mathbf{X}^\top\mathbf{X}\mathbf{w}$ (Rule 2, with $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$, which is symmetric)

Combining:

$$
\begin{aligned}
\nabla L(\mathbf{w}) &= \mathbf{0} - 2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\mathbf{w} \\
&= 2\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{X}^\top\mathbf{y}
\end{aligned}
$$

Factoring out:

$$\nabla L(\mathbf{w}) = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

Both methods yield the same result:

$$\boxed{\nabla L(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})}$$

### Interpreting the Gradient

The gradient $\nabla L = -2\mathbf{X}^\top \mathbf{r}$ has a clear interpretation. The residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w}$ measures the prediction errors for each sample. The matrix multiplication $\mathbf{X}^\top \mathbf{r}$ computes a weighted sum of feature vectors, weighted by the residuals:

$$(\mathbf{X}^\top \mathbf{r})_j = \sum_{i=1}^n x_{ij} r_i$$

Each component of $\mathbf{X}^\top \mathbf{r}$ measures the correlation between feature $j$ and the prediction errors. If feature $j$ is positively correlated with the residuals (the model is under-predicting for samples with high feature $j$ values), then $(\mathbf{X}^\top \mathbf{r})_j > 0$, and the gradient component $-2(\mathbf{X}^\top \mathbf{r})_j < 0$, indicating that $w_j$ should be increased. This is exactly what gradient descent does.

## The Normal Equations

Setting the gradient to zero gives the optimal weights:

$$\nabla L(\mathbf{w}^*) = \mathbf{0} \implies -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}^*) = \mathbf{0}$$

$$\implies \mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y}$$

These are the **normal equations**. If $\mathbf{X}^\top\mathbf{X}$ is invertible (which requires $\mathbf{X}$ to have full column rank, meaning the features are linearly independent), the closed-form solution is:

$$\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

This is the **ordinary least squares (OLS)** solution. In practice, we often use gradient descent instead of directly computing the inverse, especially when:
- $d$ is very large (inverting $\mathbf{X}^\top\mathbf{X}$ takes $O(d^3)$ time)
- We want to add regularization
- The data arrives in a streaming fashion (online learning)
- We are optimizing a loss function that is not quadratic (e.g., logistic regression, neural networks)

Even when the OLS solution is available, understanding the gradient is essential because the same gradient computation pattern appears in every model trained with gradient descent.

## Numerical Gradient via Finite Differences

The numerical gradient provides an independent verification of the analytical gradient. For each component $j$ of $\mathbf{w}$, we approximate $\frac{\partial L}{\partial w_j}$ using finite differences.

### Central Difference Formula

The central difference approximation is:

$$\frac{\partial L}{\partial w_j} \approx \frac{L(\mathbf{w} + h\mathbf{e}_j) - L(\mathbf{w} - h\mathbf{e}_j)}{2h}$$

where $\mathbf{e}_j$ is the $j$-th standard basis vector (all zeros except a 1 in position $j$) and $h$ is a small step size.

This approximation has error $O(h^2)$, which is much better than the forward difference ($O(h)$). For a suitably small step size, the central difference gives many digits of accuracy for smooth functions, which is more than enough to verify an analytical gradient.

### Why Central Differences?

The central difference formula has the error expansion:

$$
\begin{aligned}
\frac{L(\mathbf{w} + h\mathbf{e}_j) - L(\mathbf{w} - h\mathbf{e}_j)}{2h} &= \frac{\partial L}{\partial w_j} + \frac{h^2}{6}\frac{\partial^3 L}{\partial w_j^3} \\
&\quad + O(h^4)
\end{aligned}
$$

The leading error term is $O(h^2)$, meaning the error decreases quadratically as $h$ shrinks. Compare this to the forward difference:

$$\frac{L(\mathbf{w} + h\mathbf{e}_j) - L(\mathbf{w})}{h} = \frac{\partial L}{\partial w_j} + \frac{h}{2}\frac{\partial^2 L}{\partial w_j^2} + O(h^2)$$

which has only $O(h)$ accuracy. For the SSE loss, which is quadratic in $\mathbf{w}$, the third and higher derivatives are zero, so the central difference formula is actually exact (up to floating-point rounding errors) for any $h$.

### Choosing the Step Size

The optimal step size balances approximation error (which decreases with $h$) against floating-point rounding error (which increases as $h \to 0$ because $L(\mathbf{w} + h\mathbf{e}_j)$ and $L(\mathbf{w} - h\mathbf{e}_j)$ become very close, causing catastrophic cancellation).

For double-precision arithmetic, the optimal $h$ for central differences is approximately $h \approx \epsilon_{\text{mach}}^{1/3}$ where $\epsilon_{\text{mach}}$ is machine epsilon. This gives many digits of agreement between the analytical and numerical gradients.

### The Gradient Checking Algorithm

```
function numerical_gradient(L, w, h=1e-5):
    d = length(w)
    grad = zeros(d)
    for j = 1 to d:
        w_plus = copy(w)
        w_minus = copy(w)
        w_plus[j] += h
        w_minus[j] -= h
        grad[j] = (L(w_plus) - L(w_minus)) / (2 * h)
    return grad
```

This requires $2d$ evaluations of the loss function (two per parameter). For large $d$, this is expensive, which is why we use analytical gradients for training and only use numerical gradients for verification during development and debugging.

## Gradient Descent for Linear Regression

Given the analytical gradient $\nabla L(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$, the gradient descent update rule is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t) = \mathbf{w}_t + 2\alpha \mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}_t)$$

where $\alpha > 0$ is the learning rate. At each step, the weights are adjusted in the direction that reduces the loss.

### Convergence

For convex quadratic losses, gradient descent converges to the global minimum provided the learning rate satisfies:

$$0 < \alpha < \frac{1}{\lambda_{\max}(\mathbf{X}^\top\mathbf{X})}$$

where $\lambda_{\max}$ is the largest eigenvalue of $\mathbf{X}^\top\mathbf{X}$. If $\alpha$ is too large, the updates overshoot and the loss diverges.

The convergence rate depends on the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ of $\mathbf{X}^\top\mathbf{X}$. A well-conditioned problem ($\kappa$ close to 1) converges quickly, while an ill-conditioned problem ($\kappa \gg 1$) converges slowly because the loss landscape is elongated along certain directions.

## Matrix Calculus Rules

The derivation of the SSE gradient relies on a few key matrix calculus identities. These rules generalize the familiar single-variable derivative rules to vectors and matrices.

### Gradient of a Linear Function

For a constant vector $\mathbf{a} \in \mathbb{R}^d$:

$$\nabla_\mathbf{w}(\mathbf{a}^\top\mathbf{w}) = \nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{a}) = \mathbf{a}$$

This is the vector analogue of $\frac{d}{dx}(ax) = a$.

### Gradient of a Quadratic Form

For a symmetric matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$:

$$\nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{A}\mathbf{w}) = 2\mathbf{A}\mathbf{w}$$

This is the vector analogue of $\frac{d}{dx}(ax^2) = 2ax$.

For a general (not necessarily symmetric) matrix $\mathbf{A}$:

$$\nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{A}\mathbf{w}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{w}$$

When $\mathbf{A}$ is symmetric ($\mathbf{A} = \mathbf{A}^\top$), this reduces to $2\mathbf{A}\mathbf{w}$. In our case, $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$, which is always symmetric: $(\mathbf{X}^\top\mathbf{X})^\top = \mathbf{X}^\top(\mathbf{X}^\top)^\top = \mathbf{X}^\top\mathbf{X}$.

### Chain Rule for Vector Functions

If $\mathbf{g}(\mathbf{w}) = \mathbf{y} - \mathbf{X}\mathbf{w}$ and $L(\mathbf{w}) = \mathbf{g}(\mathbf{w})^\top\mathbf{g}(\mathbf{w}) = \|\mathbf{g}(\mathbf{w})\|^2$, then:

$$\nabla L = 2 \mathbf{J}_g^\top \mathbf{g}$$

where $\mathbf{J}_g = \frac{\partial \mathbf{g}}{\partial \mathbf{w}} = -\mathbf{X}$ is the Jacobian of $\mathbf{g}$. Substituting:

$$\nabla L = 2(-\mathbf{X})^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

This chain rule approach generalizes to any composition of differentiable functions and is the foundation of backpropagation in neural networks.

### Gradient of the Squared Norm

For a vector-valued function $\mathbf{f}(\mathbf{w})$:

$$\nabla_\mathbf{w}\|\mathbf{f}(\mathbf{w})\|^2 = 2\mathbf{J}_f^\top \mathbf{f}(\mathbf{w})$$

where $\mathbf{J}_f$ is the Jacobian matrix with entries $(\mathbf{J}_f)_{ij} = \frac{\partial f_i}{\partial w_j}$. When $\mathbf{f}(\mathbf{w}) = \mathbf{y} - \mathbf{X}\mathbf{w}$, the Jacobian is $-\mathbf{X}$ (constant), and we recover our gradient formula.

## Alternative Forms of the Gradient

The SSE gradient can be written in several equivalent forms, each useful in different contexts:

**Residual form** (most common):
$$\nabla L = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

**Expanded form** (useful for understanding the quadratic structure):
$$\nabla L = 2\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{X}^\top\mathbf{y}$$

**Component form** (useful for understanding per-feature contributions):
$$\frac{\partial L}{\partial w_j} = -2\sum_{i=1}^n x_{ij}(y_i - \hat{y}_i) = 2\sum_{i=1}^n x_{ij}(\hat{y}_i - y_i)$$

**Summation form** (useful for stochastic gradient descent):
$$\nabla L = \sum_{i=1}^n \nabla L_i \quad \text{where} \quad \nabla L_i = -2(y_i - \mathbf{w}^\top\mathbf{x}_i)\mathbf{x}_i$$

The summation form reveals that the total gradient is a sum of per-sample gradients. This decomposition is the basis of stochastic gradient descent (SGD), where we approximate the full gradient using a random subset (mini-batch) of samples.

## Connection to Gradient Descent in Practice

In real ML training, the gradient computation $\nabla L = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$ is the core operation at each step. The training loop looks like:

```
initialize w randomly
for t = 1, 2, ..., T:
    predictions = X @ w                    # O(nd)
    residuals = y - predictions             # O(n)
    gradient = -2 * X.T @ residuals         # O(nd)
    w = w - learning_rate * gradient        # O(d)
```

Each iteration costs $O(nd)$: two matrix-vector products. This is the same cost as computing $2d$ predictions, making gradient computation efficient relative to the model's forward pass.

For neural networks, the gradient computation is more complex (backpropagation through layers), but the fundamental pattern is the same: compute the forward pass, compute the residuals (or more generally, the loss gradient with respect to the output), then propagate gradients backward through the network.

### Why Not Just Solve the Normal Equations?

For linear regression, the closed-form solution $\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ exists. Why bother with gradient descent?

1. **Scalability**: Computing $(\mathbf{X}^\top\mathbf{X})^{-1}$ takes $O(d^3)$ time. For $d = 10^6$ (common in NLP), this is infeasible. Gradient descent requires only $O(nd)$ per step.

2. **Generalization**: The normal equation only works for linear regression with squared loss. Gradient descent works for any differentiable loss function and any differentiable model.

3. **Online learning**: Gradient descent can process data in batches, allowing it to handle streaming data or datasets too large to fit in memory.

4. **Regularization**: Adding L2 regularization modifies the gradient to $\nabla L = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda\mathbf{w}$, which is easy to implement in gradient descent. The normal equation solution becomes $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$, which still requires the matrix inverse.

## Geometric Interpretation

The SSE loss defines a paraboloid in the $d$-dimensional weight space. The gradient at any point $\mathbf{w}$ is perpendicular to the level surface (contour) of $L$ passing through $\mathbf{w}$ and points in the direction of steepest ascent.

For $d = 2$, the level surfaces are ellipses centered at $\mathbf{w}^*$ (the optimal weights). The axes of these ellipses are aligned with the eigenvectors of $\mathbf{X}^\top\mathbf{X}$, and their aspect ratio equals the square root of the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.

When $\kappa$ is large (elongated ellipses), the gradient at most points is nearly perpendicular to the direction toward $\mathbf{w}^*$, causing gradient descent to zigzag inefficiently. This is why preconditioning (e.g., Adam optimizer, natural gradient) can dramatically improve convergence by reshaping the loss landscape.

## Debugging with Gradient Checking

Gradient checking is a standard practice in ML development. The procedure is:

1. Compute the analytical gradient $\mathbf{g}_a = \nabla L(\mathbf{w})$ using your implemented formula
2. Compute the numerical gradient $\mathbf{g}_n$ using central differences
3. Check that they agree: $\|\mathbf{g}_a - \mathbf{g}_n\| / (\|\mathbf{g}_a\| + \|\mathbf{g}_n\|) < \epsilon$

The relative error formulation is important because it accounts for the scale of the gradient. A small absolute difference is negligible when the gradient is large but significant when the gradient itself is small.

For the SSE loss, which is quadratic, the analytical and numerical gradients should agree to near-machine precision. For non-quadratic losses (cross-entropy, huber, etc.), agreement within a few orders of magnitude of the truncation error is typically sufficient.

This debugging technique is universally applicable - it works for any differentiable loss function and any model architecture, including deep neural networks. The key insight is that the numerical gradient requires no mathematical derivation; it only requires the ability to evaluate the loss function at perturbed parameter values.
