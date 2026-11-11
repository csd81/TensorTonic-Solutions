# Straight-Through Estimator with Custom VJP

Quantization stores a model's numbers on a coarse grid, such as 8-bit or 4-bit integers, so the model uses less memory and runs faster at inference. Training a model to tolerate that grid, called quantization-aware training, runs the rounding inside the forward pass, but rounding has a derivative of zero almost everywhere and would stop all learning. The straight-through estimator replaces that useless derivative with a simple stand-in, and jax.custom_vjp is the JAX tool for attaching such a hand-written backward rule to a function, which is what you build in this problem.

## Rounding onto a grid

A quantizer has a step size called the scale and an integer range from qmin to qmax. To quantize a value $x$, divide by the scale, round to the nearest integer level, clip the level into the range, and multiply back by the scale.

$$
q(x) = s \cdot \operatorname{clip}\big(\operatorname{round}(x / s),\ q_{\min},\ q_{\max}\big)
$$

Here $s$ is the scale, and $q_{\min}$ and $q_{\max}$ are the smallest and largest allowed levels. A signed 4-bit grid uses levels from $-8$ to $7$, and an unsigned 8-bit grid uses levels from 0 to 255. This is often called fake quantization, because the result is stored as a float, yet it can only take values on the grid. The model sees exactly the numbers it will see after real quantization.

## Ties go to the even level

A value exactly halfway between two levels needs a rule. jnp.round follows the IEEE default, which rounds a half to the nearest even integer. So 0.5 becomes 0, 1.5 becomes 2, 2.5 becomes 2, and $-1.5$ becomes $-2$. The more familiar rule of adding 0.5 and taking the floor always rounds halves up, which gives 1 and 3 for 0.5 and 2.5. Rounding halves to even avoids a small upward bias when many values land exactly on halves, which happens often when the scale is a power of two.

## Why the true gradient is useless

The rounding function is a staircase. Between two steps it is flat, so its derivative is 0, and at each step it jumps, so its derivative does not exist. Automatic differentiation reports 0 everywhere, so any parameter that only reaches the loss through a quantizer receives a gradient of exactly 0 and never changes.

Clipping adds a second flat region. A value far below $q_{\min} \cdot s$ maps to the lowest level no matter how it moves, so the output does not depend on it at all, and its true derivative is 0 as well.

## Pretending rounding is the identity

The straight-through estimator (Bengio et al., 2013) keeps the rounding in the forward pass and replaces its derivative in the backward pass with the derivative of the identity, which is 1. The reasoning is that rounding changes a value by at most half a step, so on average the quantized output moves together with its input. The gradient flows straight through the rounding as if it were not there.

The clipped version used in practice keeps the clipping honest. Inside the range, where the quantizer tracks its input, the gradient passes through unchanged. Outside the range, where the quantizer is saturated, the gradient is set to 0. Writing $c$ for the gradient arriving from later layers, the rule is

$$
\frac{\partial L}{\partial x} = c \cdot \mathbb{1}\big[q_{\min} \le x / s \le q_{\max}\big]
$$

where $L$ is the loss and $\mathbb{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. Blocking the saturated entries stops the optimizer from pushing values ever further past the edge of the grid, where moving them changes nothing in the forward pass.

## Cotangents and the vector-Jacobian product

Reverse-mode differentiation passes one array backwards through each operation. That array is called the cotangent, and it has the same shape as the operation's output. For an operation with output $u = h(x)$, the cotangent is $c = \partial L / \partial u$, and the backward rule turns it into $\partial L / \partial x$, the cotangent for the input. This is the vector-Jacobian product.

jax.vjp exposes it directly. Given a function and an input, it returns the function's output and a pullback function. Calling the pullback with a cotangent returns a tuple with one cotangent per input. Calling jax.grad on the sum of $c$ times the output gives the same result, since the derivative of that sum with respect to the output is $c$.

## Attaching a custom backward rule

jax.custom_vjp wraps a function so that JAX uses your backward rule instead of differentiating the function's body. You supply two pieces with defvjp.

- **The forward rule.** It runs when the function is differentiated, returns the normal output, and also returns residuals, which are whatever values the backward rule will need later. For the clipped straight-through rule, the input itself is enough.
- **The backward rule.** It receives the residuals and the cotangent of the output and returns a tuple with one cotangent for each differentiable input.

When the function is called without differentiation, JAX runs the original body. Inside a jaxpr, the intermediate program that JAX builds when it traces a function, the wrapped function shows up as a single custom_vjp_call operation. Values such as the scale and the range can be captured from the enclosing function, so they are constants as far as differentiation is concerned. JAX also offers jax.custom_jvp, which defines a forward-mode rule instead, but custom_vjp is the natural fit when you want to dictate exactly what flows backwards.

## A four-value walk-through

Take a signed 3-bit grid with $s = 0.25$, $q_{\min} = -4$ and $q_{\max} = 3$, the inputs $x = [-1.3, 0.1, 0.375, 0.6]$, and the cotangent $c = [0.5, -1, 2, 1]$.

The ratios $x / s$ are $[-5.2, 0.4, 1.5, 2.4]$, and rounding them gives the levels $[-5, 0, 2, 2]$, where 1.5 is a tie that goes to the even level 2. Clipping the levels to the range turns $-5$ into $-4$, so the first value is saturated, and multiplying by the scale gives the quantized values $[-1, 0, 0.5, 0.5]$. The range test on the ratios fails only for $-5.2$, so the gradient is $[0, -1, 2, 1]$.

The true derivative would have given $[0, 0, 0, 0]$, so without the custom rule nothing upstream of this quantizer could learn.

## Where fake quantization sits in training

In quantization-aware training, fake quantizers are placed on the weights of each layer and often on its activations too. The forward pass therefore sees the same rounding errors that the deployed integer model will make, and the straight-through gradients let the full-precision weights adjust so that those errors hurt the loss as little as possible. After training, the weights are rounded one last time and stored as true integers together with their scale.

## Common mistakes to avoid

- **Differentiating the rounding directly.** Without a custom rule the gradient through the quantizer is exactly zero, so nothing before it can learn.
- **Faking the rule with stop_gradient.** An identity plus a stopped correction mimics the numbers, but it hides the rule inside ordinary operations instead of declaring it with jax.custom_vjp.
- **Excluding the ends of the range.** A value that lands exactly on $q_{\min}$ or $q_{\max}$ is still inside the grid, so its gradient must pass through.
- **Passing the gradient through saturated values.** An unclipped identity keeps pushing values that are already past the edge, even though moving them changes nothing.
- **Rounding halves upwards.** jnp.round sends halves to the even level, so adding 0.5 and taking the floor gives a different level for every tie.
- **Returning a bare array from the backward rule.** The backward rule must return a tuple with one entry per differentiable input, even when there is only one.
