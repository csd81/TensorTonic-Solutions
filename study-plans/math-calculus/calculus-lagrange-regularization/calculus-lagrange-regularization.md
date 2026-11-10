# <span style="font-size: 20px;">Lagrange Multipliers: Regularization as Constrained Optimization</span>

## The Two Faces of Regularization

In machine learning, regularization appears in two seemingly different forms:

**The penalty form (unconstrained):**

$$\min_{\mathbf{w}} \; L(\mathbf{w}) + \lambda \|\mathbf{w}\|^2$$

Here, a penalty term $\lambda \|\mathbf{w}\|^2$ is added to the loss, discouraging large weights. The hyperparameter $\lambda > 0$ controls the strength of regularization.

**The constraint form (constrained):**

$$\min_{\mathbf{w}} \; L(\mathbf{w}) \quad \text{subject to} \quad \|\mathbf{w}\|^2 \leq c$$

Here, the weights are explicitly restricted to lie within a ball of radius $\sqrt{c}$. The hyperparameter $c > 0$ controls the size of the feasible region.

These two formulations look different, but they are mathematically equivalent: for every $\lambda > 0$, there exists a $c > 0$ such that both problems have the same solution, and vice versa. This equivalence is one of the most important results in optimization theory, and it is proven using Lagrange multipliers and the Karush-Kuhn-Tucker (KKT) conditions.

## Constrained Optimization

### The General Problem

A constrained optimization problem has the form:

$$\min_{\mathbf{x}} \; f(\mathbf{x}) \quad \text{subject to} \quad g(\mathbf{x}) \leq 0$$

where $f$ is the objective function and $g$ is the constraint function. The set $\{\mathbf{x} : g(\mathbf{x}) \leq 0\}$ is called the feasible region.

The challenge is that the minimum of $f$ over all of $\mathbb{R}^n$ (the unconstrained minimum) may not satisfy the constraint. When the unconstrained minimum lies outside the feasible region, the constrained minimum must lie on the boundary $g(\mathbf{x}) = 0$.

### The Lagrangian

The Lagrangian function combines the objective and constraint into a single function:

$$\mathcal{L}(\mathbf{x}, \mu) = f(\mathbf{x}) + \mu \cdot g(\mathbf{x})$$

where $\mu \geq 0$ is the Lagrange multiplier (also called the dual variable). The Lagrange multiplier measures the "price" of the constraint: it tells you how much the optimal objective value would change if the constraint were relaxed slightly.

### The KKT Conditions

The Karush-Kuhn-Tucker (KKT) conditions are necessary conditions for optimality in constrained optimization. If $\mathbf{x}^*$ is a local minimum of the constrained problem with appropriate regularity conditions, then there exists a multiplier $\mu^* \geq 0$ such that:

**1. Stationarity:** The gradient of the Lagrangian vanishes:

$$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \mu^*) = \nabla f(\mathbf{x}^*) + \mu^* \nabla g(\mathbf{x}^*) = \mathbf{0}$$

This says that at the optimum, the gradient of the objective is a negative scalar multiple of the gradient of the constraint. Geometrically, the objective cannot be decreased without violating the constraint.

**2. Primal feasibility:** The solution satisfies the constraint:

$$g(\mathbf{x}^*) \leq 0$$

**3. Dual feasibility:** The multiplier is non-negative:

$$\mu^* \geq 0$$

**4. Complementary slackness:** Either the constraint is active or the multiplier is zero:

$$\mu^* \cdot g(\mathbf{x}^*) = 0$$

Complementary slackness is the key condition. It says:
- If $\mu^* > 0$, then $g(\mathbf{x}^*) = 0$ (the constraint is active/binding)
- If $g(\mathbf{x}^*) < 0$, then $\mu^* = 0$ (the constraint is slack and irrelevant)

### Sufficiency for Convex Problems

For convex problems (where $f$ is convex and $g$ is convex), the KKT conditions are both necessary and sufficient for global optimality. This is the case for our regularization problem, since both the quadratic loss and the norm constraint are convex.

## The Quadratic Loss Model

### Definition

Consider the quadratic loss function:

$$L(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top A \mathbf{w} - \mathbf{b}^\top \mathbf{w}$$

where $A$ is a $2 \times 2$ symmetric positive definite matrix and $\mathbf{b} \in \mathbb{R}^2$. This is the loss function for linear regression in 2D, where $A = \mathbf{X}^\top \mathbf{X}$ and $\mathbf{b} = \mathbf{X}^\top \mathbf{y}$.

### The Unregularized Solution

The gradient of $L$ is:

$$\nabla L(\mathbf{w}) = A\mathbf{w} - \mathbf{b}$$

Setting the gradient to zero gives the unregularized optimum:

$$A\mathbf{w}^* = \mathbf{b} \implies \mathbf{w}^* = A^{-1}\mathbf{b}$$

Since $A$ is positive definite, this is the unique global minimum.

### Why Regularize?

The unregularized solution $\mathbf{w}^* = A^{-1}\mathbf{b}$ may have undesirable properties:

- **Large weights:** If $A$ is ill-conditioned (eigenvalues spanning many orders of magnitude), the solution can have very large components, leading to overfitting.
- **Sensitivity to noise:** Small perturbations in $\mathbf{b}$ (due to noisy data) can cause large changes in $\mathbf{w}^*$ when $A$ is ill-conditioned.
- **Lack of generalization:** Overly large weights mean the model relies too heavily on specific features, reducing its ability to generalize.

Regularization addresses all three issues by either penalizing or constraining the norm of the weight vector.

## The Penalty Form (L2 Regularization)

### The Regularized Objective

Adding an L2 penalty gives:

$$\min_{\mathbf{w}} \; \frac{1}{2} \mathbf{w}^\top A \mathbf{w} - \mathbf{b}^\top \mathbf{w} + \lambda \|\mathbf{w}\|^2$$

where $\lambda > 0$. Rewriting $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w} = \mathbf{w}^\top I \mathbf{w}$:

$$
\begin{aligned}
 &= \frac{1}{2} \mathbf{w}^\top A \mathbf{w} + \lambda \mathbf{w}^\top I \mathbf{w} - \mathbf{b}^\top \mathbf{w} \\
&= \frac{1}{2} \mathbf{w}^\top (A + 2\lambda I) \mathbf{w} - \mathbf{b}^\top \mathbf{w}
\end{aligned}
$$

### The Regularized Solution

The gradient of the regularized objective is:

$$(A + 2\lambda I)\mathbf{w} - \mathbf{b} = \mathbf{0}$$

The solution is:

$$\mathbf{w}_\lambda = (A + 2\lambda I)^{-1} \mathbf{b}$$

Since $A$ is positive definite and $\lambda > 0$, the matrix $A + 2\lambda I$ is positive definite with eigenvalues shifted up by $2\lambda$. This means:

- The solution always exists and is unique
- The eigenvalues of $(A + 2\lambda I)^{-1}$ are smaller than those of $A^{-1}$, so $\|\mathbf{w}_\lambda\| < \|\mathbf{w}^*\|$
- The solution is less sensitive to noise in $\mathbf{b}$

### The Shrinkage Effect

The regularized solution can be understood through the eigendecomposition of $A$. If $A = Q \Lambda Q^\top$ with eigenvalues $\lambda_1, \lambda_2$ and eigenvectors $\mathbf{q}_1, \mathbf{q}_2$, then:

$$\mathbf{w}_\lambda = Q (\Lambda + 2\lambda I)^{-1} Q^\top \mathbf{b}$$

In the eigenbasis, each component of $\mathbf{b}$ is scaled by $1/(\lambda_i + 2\lambda)$ instead of $1/\lambda_i$. The shrinkage factor is:

$$\frac{\lambda_i}{\lambda_i + 2\lambda}$$

Components corresponding to small eigenvalues (high-variance directions) are shrunk the most. This is the mathematical basis of ridge regression's bias-variance tradeoff: regularization increases bias but reduces variance.

## The Constraint Form

### The Constrained Problem

The constrained formulation is:

$$\min_{\mathbf{w}} \; \frac{1}{2} \mathbf{w}^\top A \mathbf{w} - \mathbf{b}^\top \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|^2 \leq c$$

The constraint $\|\mathbf{w}\|^2 \leq c$ restricts the weights to lie within a ball of radius $\sqrt{c}$ centered at the origin.

### The Lagrangian

The constraint $g(\mathbf{w}) = \|\mathbf{w}\|^2 - c \leq 0$. The Lagrangian is:

$$\mathcal{L}(\mathbf{w}, \mu) = \frac{1}{2} \mathbf{w}^\top A \mathbf{w} - \mathbf{b}^\top \mathbf{w} + \mu (\|\mathbf{w}\|^2 - c)$$

### Applying KKT

**Stationarity:**

$$\nabla_{\mathbf{w}} \mathcal{L} = A\mathbf{w} - \mathbf{b} + 2\mu \mathbf{w} = (A + 2\mu I)\mathbf{w} - \mathbf{b} = \mathbf{0}$$

This gives:

$$\mathbf{w} = (A + 2\mu I)^{-1} \mathbf{b}$$

**Complementary slackness:**

$$\mu (\|\mathbf{w}\|^2 - c) = 0$$

Two cases arise:

**Case 1: $\mu = 0$ (constraint is slack).** Then $\mathbf{w} = A^{-1}\mathbf{b} = \mathbf{w}^*$, the unregularized solution. This holds when $\|\mathbf{w}^*\|^2 \leq c$, i.e., when the constraint is large enough to contain the unconstrained optimum.

**Case 2: $\mu > 0$ (constraint is active).** Then $\|\mathbf{w}\|^2 = c$ and $\mathbf{w} = (A + 2\mu I)^{-1}\mathbf{b}$. The multiplier $\mu$ is determined by the condition:

$$\|(A + 2\mu I)^{-1}\mathbf{b}\|^2 = c$$

### Solving for the Multiplier

The function $\phi(\mu) = \|(A + 2\mu I)^{-1}\mathbf{b}\|^2$ is strictly decreasing in $\mu$ for $\mu \geq 0$ (adding regularization always shrinks the solution). Therefore:

- $\phi(0) = \|A^{-1}\mathbf{b}\|^2 = \|\mathbf{w}^*\|^2$
- $\phi(\mu) \to 0$ as $\mu \to \infty$

If $c < \|\mathbf{w}^*\|^2$, there is a unique $\mu^* > 0$ satisfying $\phi(\mu^*) = c$. This $\mu^*$ can be found by bisection: since $\phi$ is continuous and strictly decreasing, we search for the root of $\phi(\mu) - c = 0$ in the interval $[0, M]$ where $M$ is large enough that $\phi(M) < c$.

## The Equivalence

### From Penalty to Constraint

Given $\lambda > 0$, the regularized solution is $\mathbf{w}_\lambda = (A + 2\lambda I)^{-1}\mathbf{b}$. Setting:

$$c = \|\mathbf{w}_\lambda\|^2$$

the constrained problem with this $c$ has the same solution $\mathbf{w}_\lambda$. The Lagrange multiplier of the constrained problem equals $\lambda$.

**Proof:** The KKT stationarity condition for the constrained problem is $(A + 2\mu I)\mathbf{w} = \mathbf{b}$, with solution $\mathbf{w} = (A + 2\mu I)^{-1}\mathbf{b}$. The complementary slackness condition requires $\|\mathbf{w}\|^2 = c = \|\mathbf{w}_\lambda\|^2$. Since $\phi(\mu) = \|(A + 2\mu I)^{-1}\mathbf{b}\|^2$ is strictly decreasing, the unique solution to $\phi(\mu) = c$ is $\mu = \lambda$. Therefore the constrained solution is $(A + 2\lambda I)^{-1}\mathbf{b} = \mathbf{w}_\lambda$.

### From Constraint to Penalty

Given $c > 0$ with $c < \|\mathbf{w}^*\|^2$, the constrained solution has an associated Lagrange multiplier $\mu^* > 0$. Setting $\lambda = \mu^*$, the regularized problem with this $\lambda$ has the same solution.

**Proof:** The constrained solution satisfies $(A + 2\mu^* I)\mathbf{w}_c = \mathbf{b}$. But this is exactly the optimality condition for the regularized problem with $\lambda = \mu^*$. Therefore $\mathbf{w}_c = (A + 2\mu^* I)^{-1}\mathbf{b} = \mathbf{w}_{\mu^*}$.

### The Bijection

The mapping $\lambda \mapsto c(\lambda) = \|\mathbf{w}_\lambda\|^2$ is a strictly decreasing bijection from $(0, \infty)$ to $(0, \|\mathbf{w}^*\|^2)$:

- As $\lambda \to 0^+$: $c(\lambda) \to \|\mathbf{w}^*\|^2$ (minimal regularization, constraint barely active)
- As $\lambda \to \infty$: $c(\lambda) \to 0$ (maximal regularization, weights shrunk to zero)

The inverse mapping $c \mapsto \lambda(c)$ is also strictly decreasing, and equals the Lagrange multiplier of the constrained problem.

## Geometric Interpretation

### Level Sets and Constraint Boundaries

The regularization equivalence has a beautiful geometric interpretation in 2D:

- The **loss level sets** $\{w : L(\mathbf{w}) = k\}$ are ellipses centered at $\mathbf{w}^* = A^{-1}\mathbf{b}$, with axes determined by the eigenvectors of $A$
- The **constraint boundary** $\|\mathbf{w}\|^2 = c$ is a circle of radius $\sqrt{c}$ centered at the origin
- The **constrained optimum** is the point on the circle where a loss level set is tangent to the circle

At the tangent point, the gradients of $L$ and the constraint are parallel:

$$\nabla L(\mathbf{w}) = -\mu \nabla g(\mathbf{w})$$

$$A\mathbf{w} - \mathbf{b} = -2\mu \mathbf{w}$$

This is exactly the KKT stationarity condition.

### The Penalty Interpretation

The regularized objective $L(\mathbf{w}) + \lambda \|\mathbf{w}\|^2$ has level sets that are ellipses centered at $\mathbf{w}_\lambda$ (which is closer to the origin than $\mathbf{w}^*$). The regularized minimum $\mathbf{w}_\lambda$ is the point where these modified level sets reach their innermost contour - which geometrically coincides with the tangent point of the original level sets on the constraint circle of radius $\|\mathbf{w}_\lambda\|$.

## Connection to Ridge Regression

### Ridge Regression

For linear regression with design matrix $X$ and target $\mathbf{y}$, the ridge regression problem is:

$$\min_{\mathbf{w}} \; \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$$

Setting $A = X^\top X$ and $\mathbf{b} = X^\top \mathbf{y}$, this is exactly our quadratic loss with L2 penalty. The solution is:

$$\mathbf{w}_\text{ridge} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$

Note the factor of $\lambda$ versus $2\lambda$ depends on the convention for the loss (whether the $1/2$ factor is included). In our formulation with $L(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top A \mathbf{w} - \mathbf{b}^\top \mathbf{w}$, the gradient gives $(A + 2\lambda I)\mathbf{w} = \mathbf{b}$.

### Tikhonov Regularization

Ridge regression is a special case of Tikhonov regularization, which uses the penalty $\|\Gamma \mathbf{w}\|^2$ for a general matrix $\Gamma$. When $\Gamma = \sqrt{\lambda} I$, this reduces to L2 regularization. More general choices of $\Gamma$ allow penalizing different weight components differently.

## Bayesian Interpretation

### Maximum A Posteriori Estimation

The equivalence between penalty and constraint has a Bayesian interpretation. Consider:

- **Likelihood:** $p(\mathbf{y} | \mathbf{w}) \propto \exp(-L(\mathbf{w}))$
- **Prior:** $p(\mathbf{w}) \propto \exp(-\lambda \|\mathbf{w}\|^2)$ (Gaussian prior with precision $2\lambda$)

The maximum a posteriori (MAP) estimate is:

$$\mathbf{w}_\text{MAP} = \arg\max_{\mathbf{w}} \; p(\mathbf{w} | \mathbf{y}) = \arg\min_{\mathbf{w}} \; L(\mathbf{w}) + \lambda \|\mathbf{w}\|^2$$

This is exactly the L2-regularized problem. The regularization strength $\lambda$ encodes the prior belief about the scale of the weights: larger $\lambda$ means a tighter prior (smaller expected weights).

### The Constraint as a Confidence Region

The constrained formulation $\|\mathbf{w}\|^2 \leq c$ can be interpreted as restricting the optimization to the highest-density region of the Gaussian prior. The level set $\|\mathbf{w}\|^2 = c$ defines an ellipsoid (a circle for the isotropic Gaussian prior) that contains a certain fraction of the prior probability mass. Optimizing the loss within this region is equivalent to finding the MAP estimate with the corresponding $\lambda$.

## Weight Decay

### Weight Decay in SGD

In practice, L2 regularization is often implemented as weight decay in the gradient update:

$$
\begin{aligned}
\mathbf{w}_{t+1} &= \mathbf{w}_t - \alpha (\nabla L(\mathbf{w}_t) + 2\lambda \mathbf{w}_t) \\
&= (1 - 2\alpha\lambda)\mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)
\end{aligned}
$$

The term $(1 - 2\alpha\lambda)\mathbf{w}_t$ shrinks the weights by a factor of $(1 - 2\alpha\lambda)$ at each step, hence the name "weight decay."

### Decoupled Weight Decay (AdamW)

For adaptive optimizers like Adam, L2 regularization and weight decay are NOT equivalent. Adding the $\lambda \|\mathbf{w}\|^2$ penalty to the loss modifies the gradient used by Adam's moment estimates, which changes the effective learning rate per parameter. Decoupled weight decay (AdamW) applies the decay directly to the weights, separate from the adaptive gradient:

$$\mathbf{w}_{t+1} = (1 - \alpha \lambda) \mathbf{w}_t - \alpha \cdot \text{Adam\_update}(\nabla L(\mathbf{w}_t))$$

This distinction between L2 regularization and decoupled weight decay is one of the most practically important consequences of the regularization-constraint equivalence.

## Numerical Methods for Finding the Lagrange Multiplier

### Bisection Method

For the constrained problem, finding the Lagrange multiplier $\mu$ requires solving:

$$\phi(\mu) = \|(A + 2\mu I)^{-1}\mathbf{b}\|^2 = c$$

Since $\phi$ is strictly decreasing, continuous, and satisfies $\phi(0) > c$ and $\phi(\mu) \to 0$ as $\mu \to \infty$, bisection is guaranteed to converge.

The algorithm:
1. Set $\mu_\text{lo} = 0$ and find $\mu_\text{hi}$ large enough that $\phi(\mu_\text{hi}) < c$
2. Repeat: set $\mu_\text{mid} = (\mu_\text{lo} + \mu_\text{hi})/2$
3. If $\phi(\mu_\text{mid}) > c$, set $\mu_\text{lo} = \mu_\text{mid}$; else set $\mu_\text{hi} = \mu_\text{mid}$
4. Stop when $|\mu_\text{hi} - \mu_\text{lo}|$ is below a tolerance

Each iteration halves the interval, so after $k$ iterations the error is at most $(\mu_\text{hi}^{(0)} - \mu_\text{lo}^{(0)}) / 2^k$.

### Newton's Method

A faster alternative is Newton's method on $\phi(\mu) - c = 0$. The derivative of $\phi$ with respect to $\mu$ can be computed using the matrix identity:

$$\frac{d}{d\mu} (A + 2\mu I)^{-1} = -2(A + 2\mu I)^{-2}$$

This gives:

$$\phi'(\mu) = -4 \mathbf{b}^\top (A + 2\mu I)^{-3} \mathbf{b}$$

Newton's update is:

$$\mu_{k+1} = \mu_k - \frac{\phi(\mu_k) - c}{\phi'(\mu_k)}$$

This converges quadratically but requires more computation per iteration.

### The Eigendecomposition Approach

For the 2D case, we can use the eigendecomposition of $A$ to solve for $\mu$ analytically (or semi-analytically). Let $A = Q \text{diag}(\sigma_1, \sigma_2) Q^\top$ and $\tilde{\mathbf{b}} = Q^\top \mathbf{b}$. Then:

$$\phi(\mu) = \frac{\tilde{b}_1^2}{(\sigma_1 + 2\mu)^2} + \frac{\tilde{b}_2^2}{(\sigma_2 + 2\mu)^2}$$

Setting $\phi(\mu) = c$ and letting $t = 2\mu$:

$$\frac{\tilde{b}_1^2}{(\sigma_1 + t)^2} + \frac{\tilde{b}_2^2}{(\sigma_2 + t)^2} = c$$

This is a polynomial equation in $t$ (degree 4 after clearing denominators) that can be solved numerically. However, bisection is simpler and equally reliable.

## Practical Implications

### Hyperparameter Interpretation

The equivalence gives two equivalent ways to think about regularization strength:
- **Penalty perspective ($\lambda$):** "How much should I penalize large weights?" Larger $\lambda$ means stronger penalty.
- **Constraint perspective ($c$):** "How large should I allow the weight norm to be?" Smaller $c$ means tighter constraint.

In practice, $\lambda$ is more commonly used because it is easier to implement (just add $\lambda \|\mathbf{w}\|^2$ to the loss). But the constraint perspective is more interpretable: "I want my model's weights to have total squared norm at most $c$."

### Cross-Validation and the Regularization Path

As $\lambda$ varies from 0 to $\infty$, the regularized solution traces a path from $\mathbf{w}^*$ (unregularized) to $\mathbf{0}$ (fully regularized). This is the regularization path. The equivalence tells us that this path also corresponds to shrinking the constraint radius from $\|\mathbf{w}^*\|$ to 0.

Cross-validation selects the $\lambda$ (or equivalently, the $c$) that optimizes validation performance. The regularization path can be computed efficiently for ridge regression because the solution has a closed form for each $\lambda$.

### Elastic Net and Other Regularizers

The equivalence generalizes to other regularizers:
- **L1 regularization (Lasso):** $\|\mathbf{w}\|_1 \leq c$ is equivalent to $L(\mathbf{w}) + \lambda \|\mathbf{w}\|_1$ for some $\lambda$. The constraint region is a diamond (L1 ball), and the constraint boundary has corners that promote sparsity.
- **Elastic net:** combines L1 and L2 penalties. The constraint region is a blend of the diamond and circle.
- **Group lasso:** uses group norms to encourage structured sparsity.

In all cases, the fundamental equivalence holds: penalty and constraint formulations are dual perspectives of the same optimization problem.

### Norm-Based Generalization Bounds

The constraint perspective is central to learning theory. PAC-Bayesian generalization bounds and Rademacher complexity bounds often involve the norm of the weight vector:

$$\text{Generalization gap} \leq O\left(\frac{\|\mathbf{w}\| \cdot \|X\|}{\sqrt{n}}\right)$$

This means that constraining the weight norm directly controls generalization. The equivalence tells us that L2 regularization implicitly provides this control.

## Implementation Considerations

### Solving the Linear System

For the quadratic loss, both the unregularized and regularized solutions require solving a linear system:

$$A\mathbf{w} = \mathbf{b} \quad \text{or} \quad (A + 2\lambda I)\mathbf{w} = \mathbf{b}$$

For 2D problems, this can be done with direct matrix inversion or Gaussian elimination. For large-scale problems, iterative methods (conjugate gradient) are more efficient.

### Verifying the Equivalence

The equivalence is verified by:
1. Solving the regularized problem to get $\mathbf{w}_\lambda$
2. Computing $c = \|\mathbf{w}_\lambda\|^2$
3. Solving the constrained problem with this $c$ using bisection
4. Checking that the constrained solution matches $\mathbf{w}_\lambda$
5. Checking that the Lagrange multiplier from bisection equals $\lambda$

The match should be within numerical tolerance (determined by the bisection convergence criterion).

### The Loss Function Convention

The factor of 2 in $(A + 2\lambda I)$ arises from the derivative of $\lambda \|\mathbf{w}\|^2 = \lambda \mathbf{w}^\top \mathbf{w}$. Different conventions in the literature may include or omit the $1/2$ factor in the loss or the penalty, leading to $(A + \lambda I)$ instead. The equivalence holds regardless of the convention; only the numerical relationship between $\lambda$ and $c$ changes.
