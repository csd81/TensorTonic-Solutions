# Value and Grad of a Loss

Training any model needs two numbers at every step, the loss that says how wrong the model is and the gradient that says which way to move each parameter. JAX gets both from a single transform called jax.value_and_grad, which takes an ordinary Python function and gives back a new function that returns the value and its gradient together. This page starts with a one-number example, explains how JAX treats a function and its derivative, and ends with the logistic regression loss you differentiate in this problem.

## A gradient is a new function

In PyTorch you call backward on a tensor and the gradients appear on the parameters as a side effect. JAX instead treats differentiation as an operation on functions, so jax.grad takes a function $f$ and returns another function that computes the derivative of $f$ at whatever point you give it.

Take $f(w) = (w - 3)^2$. Its derivative is $2(w - 3)$, so at $w = 1$ the value is $f(1) = 4$ and the derivative is $-4$. In JAX the derivative function is built once and then called at any point, and calling it at 1.0 returns $-4$. The negative sign says that increasing $w$ lowers the loss, which is why gradient descent moves $w$ upward towards 3.

Because the transform acts on a function, that function has to be a pure function of its inputs. It must compute its output only from its arguments, without reading or changing global state. Anything it depends on that is not an argument, such as the data, is treated as a constant during differentiation.

## Why the value and the gradient come together

Reverse-mode differentiation, the method behind jax.grad, works in two passes. The forward pass evaluates the function and records the intermediate results. The backward pass then walks those results in reverse order and accumulates the gradient using the chain rule.

The forward pass already produces the loss, so throwing it away and calling the loss function a second time just to log it doubles the forward work. jax.value_and_grad keeps it. It returns a pair, the value of the function and the gradient, from the same forward and backward pass. For $f(w) = (w - 3)^2$ at $w = 1$, the pair is $(4, -4)$.

Most JAX training loops use this pattern, since the loss is needed for logging and early stopping and the gradient is needed for the update. Any extra outputs, such as accuracy, can ride along through the has_aux option, which tells JAX that the function returns a pair of the loss and some auxiliary data that should not be differentiated.

## Parameters can be any nested container

Real models have many parameter arrays, and keeping them in separate arguments quickly becomes awkward. JAX lets you pass them as one nested container of arrays, such as a dict, a list or a tuple. JAX calls such a container a pytree, and the arrays at its ends are its leaves.

When the argument you differentiate is a pytree, the gradient is a pytree with exactly the same structure. If the parameters are a dict with keys w and b, the gradient is also a dict with keys w and b, and each leaf of the gradient has the same shape as the matching parameter. This makes the update step simple, because every gradient leaf lines up with the parameter it belongs to.

By default the transform differentiates with respect to the first argument only. The data arrays can still be passed as later arguments or captured from the surrounding function, and either way they receive no gradient.

## The logistic regression loss

Logistic regression predicts the probability that a label is 1. For a feature row $x_n$ of length $D$, a weight vector $w$ of length $D$ and a scalar bias $b$, the model first computes a logit $z_n = x_n \cdot w + b$. The predicted probability is the sigmoid of the logit, $\sigma(z_n) = 1 / (1 + e^{-z_n})$.

The loss for one example with label $y_n \in \{0, 1\}$ is the negative log probability of the correct label. Writing that out and simplifying gives a form that uses the logit directly.

$$
\ell_n = \log\big(1 + e^{z_n}\big) - y_n z_n
$$

When $y_n = 1$ this is $\log(1 + e^{-z_n})$, which is small when the logit is large and positive. When $y_n = 0$ it is $\log(1 + e^{z_n})$, which is small when the logit is large and negative. The training loss averages $\ell_n$ over the $N$ examples, so its size does not depend on the batch size.

The derivative of $\ell_n$ with respect to the logit is $\sigma(z_n) - y_n$, the gap between the predicted probability and the label. You never have to write this by hand, because the transform derives it from the loss function, but it explains what the gradient means.

## Why the naive formula overflows

The term $\log(1 + e^{z})$ looks harmless, but $e^{z}$ is computed first. The largest float32 number is about $3.4 \times 10^{38}$, and $e^{z}$ passes that once $z$ is above about 88.72. For $z = 89$ the exponential becomes infinity, the logarithm of infinity is infinity, and the backward pass then divides infinity by infinity and produces NaN.

The function $\log(1 + e^{z})$ has its own name, softplus. A stable version rewrites it as $\max(z, 0) + \log(1 + e^{-|z|})$, which never exponentiates a positive number. For $z = 100$ it returns 100 exactly in float32. JAX provides this as jax.nn.softplus, and jnp.logaddexp with a first argument of 0 gives the same result. Large logits are common when a model is confident, so the stable form matters in practice.

## Regularizing the weights and not the bias

An L2 penalty adds $\frac{\lambda}{2} \sum_i w_i^2$ to the loss, where $\lambda \ge 0$ controls its strength. It pulls the weights towards zero, which discourages the model from relying heavily on any one feature and reduces overfitting. Its gradient with respect to $w$ is simply $\lambda w$, which the transform adds for you once the term is part of the loss.

The bias is usually left out of the penalty. It only shifts every logit by the same amount, so it does not make the model more sensitive to any feature, and penalizing it would push predictions towards a probability of 0.5 for no benefit. In this problem the penalty applies to $w$ only, so the gradient for $b$ comes from the data term alone.

## A two-example walk-through

Take one feature, $w = [1.0]$, $b = 0$, two examples with features $2$ and $-1$, labels $1$ and $0$, and $\lambda = 0.5$.

The logits are $z_1 = 2$ and $z_2 = -1$, and their softplus values are $\log(1 + e^{2}) = 2.126928$ and $\log(1 + e^{-1}) = 0.313262$. Subtracting $y_n z_n$ gives per-example losses of $2.126928 - 1 \cdot 2 = 0.126928$ and $0.313262 - 0 = 0.313262$, whose mean is $0.220095$. The penalty adds $\frac{0.5}{2} \cdot 1^2 = 0.25$, so the loss is $0.470095$.

For the gradient, the sigmoids are $\sigma(2) = 0.880797$ and $\sigma(-1) = 0.268941$, so the gaps $\sigma(z_n) - y_n$ are $-0.119203$ and $0.268941$. Dividing by $N = 2$ gives $-0.059601$ and $0.134471$. The bias gradient is their sum, $0.074869$. The weight gradient multiplies each by its feature and adds the penalty term, $2 \cdot (-0.059601) + (-1) \cdot 0.134471 + 0.5 \cdot 1 = 0.246326$. One call to the transform returns the loss $0.470095$ and the dict of these two gradients.

## Common mistakes to avoid

- **Calling the loss again for the value.** Running the loss function once for the value and once inside jax.grad repeats the forward pass that jax.value_and_grad already performs.
- **Writing log of one plus exp directly.** The exponential overflows float32 above a logit of about 88.72, which turns the loss into infinity and the gradient into NaN.
- **Adding the bias to the penalty.** The L2 term covers $w$ only, so including $b$ changes both the loss and the bias gradient.
- **Summing instead of averaging.** The data term is a mean over examples, and a sum scales the loss and every gradient by $N$.
- **Converting traced values to Python numbers.** Calling float on an intermediate result inside the loss function breaks the trace that the transform builds.
- **Changing the params dict in place.** The caller still needs the original parameters for the update step, so build new values instead of assigning into the dict.
