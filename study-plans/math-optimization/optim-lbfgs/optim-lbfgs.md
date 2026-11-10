# <span style="font-size: 20px;">L-BFGS</span>

## From Newton to Quasi-Newton

Newton's method converges very fast but requires computing and storing the full $d \times d$ Hessian. For high-dimensional problems, this is impractical. Quasi-Newton methods approximate the Hessian using only gradient information.

## BFGS

BFGS (Broyden-Fletcher-Goldfarb-Shanno) builds an approximation $H_k$ of the inverse Hessian by accumulating information from successive gradient differences. At each step, it stores:

$$
s_k = w_{k+1} - w_k, \quad y_k = \nabla f(w_{k+1}) - \nabla f(w_k)
$$

The pair $(s_k, y_k)$ encodes curvature information: $s_k$ is the step taken, and $y_k$ is how the gradient changed over that step. The ratio $s_k^T y_k / y_k^T y_k$ approximates the inverse curvature along that direction.

## Limited Memory (L-BFGS)

Full BFGS stores a dense $d \times d$ matrix. L-BFGS stores only the last $m$ pairs $(s_k, y_k)$ and uses the "two-loop recursion" to implicitly compute $H_k \nabla f$ in $O(md)$ time and $O(md)$ memory, where $m$ is typically 5-20.

## Two-Loop Recursion

Given the current gradient $g$ and stored pairs $(s_i, y_i)$ for $i = k-m, \ldots, k-1$, the algorithm computes the search direction $d = -H_k g$:

**First loop** (backward through history):

$$
\begin{aligned}
&\text{For } i = k-1, \ldots, k-m: \\
&\quad \rho_i = \frac{1}{y_i^T s_i}, \quad \alpha_i = \rho_i \, s_i^T q \\
&\quad q \leftarrow q - \alpha_i \, y_i
\end{aligned}
$$

starting with $q = g$.

**Scale** by initial Hessian estimate:

$$
\gamma = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}, \quad r = \gamma \, q
$$

The scaling $\gamma$ adapts the step size to the local curvature. When no history exists, use $\gamma = 1$.

**Second loop** (forward through history):

$$
\text{For } i = k-m, \ldots, k-1: \quad \beta = \rho_i \, y_i^T r, \quad r \leftarrow r + s_i (\alpha_i - \beta)
$$

The result $r$ is the search direction $H_k g$, so the descent direction is $d = -r$.

## Backtracking Line Search

L-BFGS uses a line search to find an appropriate step size $\alpha$ along the direction $d$. The Armijo condition checks for sufficient decrease:

$$
f(w + \alpha d) \leq f(w) + c_1 \, \alpha \, \nabla f^T d
$$

where $c_1 = 10^{-4}$ is standard. Start with $\alpha = 1$ and halve it until the condition is met. Since $d$ is a descent direction, $\nabla f^T d < 0$, so the right-hand side is less than $f(w)$.

## Why L-BFGS Works So Well

On the Rosenbrock function, gradient descent takes thousands of steps because it cannot account for the curvature mismatch between the $x$ and $y$ directions. L-BFGS builds up curvature information from the $(s, y)$ pairs and adapts its step to the local geometry, typically converging in 10-30 iterations even from distant starting points.