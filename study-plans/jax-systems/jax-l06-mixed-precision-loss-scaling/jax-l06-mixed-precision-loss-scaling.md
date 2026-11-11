# Mixed Precision with Loss Scaling

Large models train faster and use less memory when most of the arithmetic runs in a 16-bit floating-point format instead of the usual 32-bit one. The catch is that 16-bit numbers are coarse, so a careful training step keeps a full-precision copy of the weights, multiplies the loss by a large factor before differentiating, and watches for gradients that overflow. This page builds that recipe from the number formats up, ending with the dynamic loss scale you implement in this problem.

## Three ways to store a number

A floating-point number has a sign bit, some exponent bits that set its size and some mantissa bits that set how finely values are spaced. The exponent bits fix the range and the mantissa bits fix the precision.

| Format | Exponent bits | Mantissa bits | Largest value | Spacing just above 1 |
|---|---|---|---|---|
| float32 | 8 | 23 | about 3.4e38 | about 1.19e-7 |
| float16 | 5 | 10 | 65504 | 0.0009765625 |
| bfloat16 | 8 | 7 | about 3.39e38 | 0.0078125 |

bfloat16 (brain float 16) keeps the 8 exponent bits of float32 and drops most of the mantissa, so it covers the same huge range with only about two to three significant decimal digits. float16 makes the opposite trade, with finer spacing but a tiny range.

The coarse spacing is easy to see. Casting 1.01 to bfloat16 gives 1.0078125, the nearest value on its grid, while float16 gives 1.009765625. Matrix multiplications tolerate this rounding well, because each output is a sum of many products and the errors partly cancel. That is why accelerators run the heavy matmuls of training in bfloat16 and accumulate the sums at higher precision.

## Why the weights stay in float32

Rounding becomes a real problem when a small update is added to a large weight. Take a weight equal to 1.0 and an update of 0.001. In float32 the sum is about 1.001. In bfloat16 the nearest representable neighbours of 1.0 are 0.99609375 and 1.0078125, so 1.0 plus 0.001 rounds straight back to 1.0 and the update vanishes. Over thousands of steps, a model stored in bfloat16 would simply stop learning from its small gradients.

Mixed precision solves this by keeping the master weights, the copy the optimizer reads and writes, in float32. At the start of each step the master weights are cast to bfloat16 for the forward pass, and the gradients come back to float32 before the update. In JAX this happens naturally. When a function casts float32 parameters to bfloat16 inside, jax.grad returns gradients with the dtype of the parameters it was given, float32, because the cast is part of the differentiated function and its derivative casts back.

The loss itself is also worth computing in float32. In this problem the predictions are made in bfloat16, then cast to float32, and the mean squared error is taken there, so the final reduction over the batch does not lose digits.

## Gradients that are too small to store

The original motivation for loss scaling comes from float16. Its smallest positive value is about 6e-8, and anything smaller rounds to zero. Gradients of a loss averaged over many examples are often tiny, so in a float16 backward pass a gradient of 2e-8 becomes exactly 0, and that parameter receives no update at all.

Multiplying the loss by a factor $S$ (the loss scale) fixes this. By the chain rule, the gradient of $S \cdot L$ is $S$ times the gradient of $L$, so every intermediate gradient in the backward pass is $S$ times larger. With $S = 1024$, the same gradient becomes about 2.05e-5, which float16 stores comfortably. Dividing by 1024 afterwards, in float32, recovers about 2.0023e-8, close to the original value.

$$
\nabla_\theta \left( S \cdot L \right) = S \, \nabla_\theta L, \qquad g = \frac{\nabla_\theta \left( S \cdot L \right)}{S}
$$

Here $\theta$ stands for the master parameters, $L$ is the loss and $g$ is the unscaled gradient used for the update. Choosing $S$ as a power of two makes the multiply and the divide exact, because they only change the exponent.

bfloat16 has the same exponent range as float32, so underflow is far rarer there, and bfloat16 training usually runs without any loss scaling. Loss scaling exists mainly for float16. This exercise runs the float16 recipe in bfloat16 so that every part of it can run on a CPU, which is why its overflow cases need extreme loss scales close to the bfloat16 maximum of about 3.39e38. With float16, a loss of 100 multiplied by a scale of 1024 already overflows, because 102400 is larger than the float16 maximum of 65504. The overflow check is still useful on its own, as a guard that skips any step that produced inf or NaN for whatever reason.

## When the scale is too large

A larger $S$ protects more small gradients, but there is a ceiling. Every format has a largest value, and anything beyond it becomes infinity. If the scaled loss or any scaled gradient passes that limit, the backward pass produces inf, and arithmetic on inf soon produces NaN (not a number). An update built from such a gradient would destroy the weights.

So after unscaling, the step checks every gradient entry with an is-finite test. The gradients are finite only when every entry of every parameter's gradient is a normal number. When the check fails, the step is skipped. The master weights stay exactly as they were, and the only thing that changes is the loss scale, which is lowered so the next step is less likely to overflow.

## The dynamic loss scale

Picking one fixed $S$ by hand is fragile, since the right value changes as training progresses. A dynamic loss scale adjusts itself with two pieces of state, the scale $S$ and a counter of consecutive good steps, and one setting, the growth interval $N$.

- **After a finite step.** The counter goes up by one. When it reaches $N$, the scale doubles and the counter returns to 0, which probes whether a larger scale is now safe.
- **After a non-finite step.** The scale halves, but never below 1.0, and the counter returns to 0.

Here is a trace with $S = 8$, $N = 2$ and the counter starting at 0, for five steps whose gradients are finite, finite, non-finite, finite and non-finite.

| Step | Gradients | Scale after | Counter after |
|---|---|---|---|
| 1 | finite | 8 | 1 |
| 2 | finite | 16 | 0 |
| 3 | non-finite | 8 | 0 |
| 4 | finite | 8 | 1 |
| 5 | non-finite | 4 | 0 |

The scale climbs slowly and backs off immediately, so it settles near the largest value that does not overflow. Each backing off costs one skipped update, which is cheap compared with corrupted weights. Real training uses growth intervals in the thousands, so doubling is rare.

## Writing the step so it compiles

A training step in JAX is normally wrapped in jax.jit, which traces the function once with abstract placeholder values called tracers and compiles the result. A tracer has a shape and a dtype but no concrete value while tracing, so Python cannot decide an if statement on it. Writing "if the gradients are finite" with a plain Python if raises an error under jit, because the answer is only known when the compiled program runs.

The JAX way is to compute both outcomes and select between them with a traced condition. The function jnp.where takes a boolean, a value for true and a value for false, and works elementwise or on scalars. The function lax.cond does the same for whole branches. Both turn the skip decision into data flow, so the compiled step can run every iteration without retracing. A Python value that never changes between calls, such as the growth interval here, can be marked static instead, and then ordinary Python comparisons on it are fine.

## Where this step sits in training

In a full training loop, the loss scale and the good-step counter are part of the training state, carried from one call of the step to the next alongside the master weights and the optimizer state. The step in this problem uses plain gradient descent for the update, but the same scaling, unscaling, checking and skipping wrap around any optimizer, such as Adam.

## Common mistakes to avoid

- **Updating with scaled gradients.** Forgetting to divide by the loss scale makes every update $S$ times too large, which wrecks training even when nothing overflows.
- **Skipping the loss scale entirely.** Differentiating the unscaled loss gives the same numbers on easy steps but never triggers the overflow check, so the dynamic scale loses its meaning.
- **Keeping the master weights in bfloat16.** Small updates round away against large weights, so learning silently stalls.
- **Using a Python if on the finite flag.** Under jax.jit the flag is a tracer, so a Python branch raises an error, while jnp.where or lax.cond select the result inside the compiled program.
- **Updating the weights on a non-finite step.** An update built from inf or NaN gradients makes the weights non-finite for every later step.
- **Forgetting to reset the counter.** The counter must return to 0 both after the scale doubles and after an overflow, otherwise the scale grows too early.
- **Letting the scale fall below 1.** Halving without a floor can shrink the scale towards zero after a run of bad steps, which brings back the underflow the scaling was meant to prevent.