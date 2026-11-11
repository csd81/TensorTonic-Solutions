# Hessian-Vector Product

The gradient of a loss tells you the slope in every direction, and the Hessian tells you how that slope changes, which is the curvature of the loss. Many tools need curvature, from Newton-style optimizers to measuring the sharpness of a minimum, but the full Hessian is far too large to build for a real model. This page shows how JAX gets the product of the Hessian with any vector by combining its two kinds of automatic differentiation, without ever forming the matrix, which is what you implement in this problem.

## Curvature of a one-number function

Start with one parameter and $f(w) = w^4 / 4$. The first derivative is $f'(w) = w^3$ and the second derivative is $f''(w) = 3w^2$. At $w = 2$, the slope is $8$ and the curvature is $12$.

The curvature says how fast the slope changes when $w$ moves. If you nudge $w$ by a small amount $\epsilon$ in a direction $v$, the slope changes by about $\epsilon \cdot f''(w) \cdot v$. With $v = 0.5$ that change rate is $12 \cdot 0.5 = 6$. For one parameter the Hessian is just this single number, and the Hessian-vector product is $f''(w) \cdot v$.

## The Hessian is too big to build

With $D$ parameters, the gradient $\nabla f(w)$ is a vector of length $D$ and the Hessian $H(w)$ is the $D$ by $D$ matrix of second derivatives, with entry $H_{ij} = \partial^2 f / \partial w_i \partial w_j$. Row $i$ of the Hessian says how the $i$-th entry of the gradient changes as each parameter moves.

The size grows with the square of the parameter count. A model with one million parameters has $10^{12}$ Hessian entries, which in float32 is 4 terabytes. Most algorithms that use curvature never need the matrix itself, though. Conjugate gradient (an iterative solver for linear systems), the power iteration for the top eigenvalue, and many sharpness measures only ever ask for $H(w)\,v$ for some vector $v$ of length $D$, and that answer is just another vector of length $D$.

## Two ways to push derivatives through a program

JAX has two basic differentiation modes. To describe them, think of any function $g$ that maps a vector to a vector, and of its Jacobian $J$, the matrix of first derivatives of every output with respect to every input.

- **Forward mode.** jax.jvp takes a point and a tangent vector $v$ of input size and returns $g$ at the point together with $J v$, the rate at which the outputs change when the input moves along $v$. The name stands for Jacobian-vector product. Its cost is a small constant multiple of evaluating $g$ once.
- **Reverse mode.** jax.vjp takes a point and returns a function that maps a vector $u$ of output size to $u^\top J$, the vector-Jacobian product. jax.grad is the special case for a scalar output with $u = 1$, which gives the whole gradient in one backward pass.

Reverse mode is ideal for a scalar loss with many parameters, which is why training uses it. Forward mode is ideal when you want the derivative along one chosen direction.

## Differentiating the gradient along a direction

The trick is to notice that the gradient is itself a function. jax.grad turns $f$ into a function that maps $w$ to the vector $\nabla f(w)$, and the Jacobian of that function is exactly the Hessian. A Jacobian-vector product of the gradient function is therefore a Hessian-vector product.

$$
H(w)\,v = \frac{d}{d\epsilon} \nabla f(w + \epsilon v) \Big|_{\epsilon = 0}
$$

Forward mode computes this derivative exactly, without choosing a step size, by carrying the tangent $v$ through every operation of the gradient computation. The cost is a small multiple of one gradient evaluation, and the memory is a few vectors of length $D$. This combination is called forward-over-reverse, because forward mode is applied on top of a reverse-mode gradient.

Other combinations also work. Reverse-over-reverse takes the gradient of $\nabla f(w) \cdot v$, and reverse-over-forward swaps the order. Forward-over-reverse is the usual choice because forward mode stores no extra tape of intermediate values, so it adds the least memory on top of the gradient.

## Finite differences lose precision

A tempting shortcut is $\big(\nabla f(w + \epsilon v) - \nabla f(w - \epsilon v)\big) / (2\epsilon)$. It needs a step size, and no step size works well in float32. A large $\epsilon$ adds truncation error from the higher derivatives, and a small one subtracts two nearly equal gradients and loses most of their digits. Forward mode gives the exact derivative of the computed gradient, accurate to float32 rounding.

## The loss in this problem

The loss combines a logistic regression data term with a quartic penalty. For rows $x_n$ of the data matrix and labels $y_n \in \{0, 1\}$, the data term is the mean of $\log(1 + e^{x_n \cdot w}) - y_n\, x_n \cdot w$, and the penalty is $\frac{\alpha}{4} \sum_i w_i^4$ with strength $\alpha \ge 0$.

The two parts have different curvature. The data term curves only along the directions spanned by the data rows, weighted by how uncertain the model is about each example, which is largest when the predicted probability is near 0.5. The quartic penalty curves each weight on its own, with curvature $3\alpha w_i^2$, which is zero at $w_i = 0$ and grows quickly away from it. That second part is the one-number example from the start of this page, applied to every weight.

## A two-weight walk-through

Take $D = 2$, one example with features $x_1 = [1, 1]$ and label $y_1 = 1$, weights $w = [1, 0]$, direction $v = [1, 1]$ and $\alpha = 1$.

The logit is $x_1 \cdot w = 1$, so the predicted probability is $\sigma(1) = 0.731059$ and the uncertainty weight is $\sigma(1)(1 - \sigma(1)) = 0.196612$. The data term's Hessian is that weight times the outer product of $x_1$ with itself, a matrix whose entries all equal $0.196612$, and its product with $v$ is $0.196612 \cdot (x_1 \cdot v) = 0.196612 \cdot 2 = 0.393224$ in both entries. The penalty's curvature $3 w_i^2$ is 3 for the first weight and 0 for the second, so it contributes $[3 \cdot 1, 0 \cdot 1] = [3, 0]$. Adding the two parts gives the Hessian-vector product $[3.393224, 0.393224]$.

The forward-over-reverse call returns this vector as the tangent output. It also returns the gradient at $w$ as the primal output, which here is $[0.731059, -0.268941]$. The 2 by 2 matrix in step 2 appears only in this explanation, and the transform never stores it.

## Common mistakes to avoid

- **Building the full Hessian.** jax.hessian or an explicit $D$ by $D$ matrix gives the right numbers on tiny inputs but costs memory that grows with the square of the parameter count.
- **Applying forward mode to the loss instead of its gradient.** A jvp of the scalar loss gives the directional slope $\nabla f(w) \cdot v$, which is a single number, while the problem needs the Hessian times $v$.
- **Keeping the wrong output of jvp.** The first output is the gradient at $w$ and the second is the product with the tangent, which is the one this problem asks for.
- **Using finite differences.** Any step size trades truncation error against cancellation error, so the result is far less accurate than forward mode in float32.
- **Getting the quartic curvature wrong.** The penalty $\frac{\alpha}{4} w_i^4$ has second derivative $3\alpha w_i^2$, and changing the constant in front of the penalty changes every product with a nonzero weight.
- **Turning traced values into Python numbers.** The gradient function is traced by the forward-mode transform, so calling float or NumPy on its intermediate values fails.
