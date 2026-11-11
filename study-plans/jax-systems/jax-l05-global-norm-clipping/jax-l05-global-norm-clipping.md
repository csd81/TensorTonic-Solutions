# Global Norm Gradient Clipping

Gradient clipping is a safety rail that stops one unusually large gradient from throwing a model's parameters far off course. The version used in almost every large training run measures the size of the whole gradient, across every array of the model, and shrinks all of it by one shared factor when that size crosses a limit. This page builds the idea from a single vector, explains why the scale must be shared, and then covers two JAX details that matter in practice, making the decision inside jax.jit and computing the norm in enough precision.

## Why one large gradient is dangerous

A training step moves every parameter by the learning rate times its gradient. Most of the time the gradients have a familiar size, and the learning rate is tuned for that size. Every so often a batch produces a gradient that is hundreds of times larger, for example because of an unusual example, a long sequence in a recurrent network, or a loss spike in a large transformer. One step with that gradient moves the parameters so far that the model may never recover (Pascanu et al., 2013, studied this for recurrent networks).

Lowering the learning rate for every step would slow down all the normal steps to protect against a few rare ones. Clipping only changes the rare steps. It leaves ordinary gradients alone and caps the length of the unusual ones.

## Clipping a single vector

Start with one gradient vector $g = [6, 8]$. Its L2 norm, the ordinary length of the vector, is $\sqrt{6^2 + 8^2} = 10$. Suppose the limit is $c = 2$. Dividing the vector by its own norm and multiplying by the limit gives $[1.2, 1.6]$, whose length is exactly 2.

The clipped vector points in the same direction as the original, because both entries were multiplied by the same number, $c / \lVert g \rVert = 0.2$. Only the length changed. A gradient points in the direction in which the loss falls fastest, so keeping the direction and reducing the length gives a shorter step in the same good direction.

## Treating the whole model as one vector

A model's gradient is a pytree with many leaves, such as a weight matrix and a bias for every layer. The global norm treats all of those numbers as if they were written out one after another in one long vector, and measures the length of that vector.

$$
\lVert g \rVert = \sqrt{\sum_{\ell} \sum_{i} g_{\ell,i}^{2}}
$$

Here $\ell$ runs over the leaves of the tree and $i$ runs over the entries of one leaf, so $g_{\ell,i}$ is a single number of the gradient. The inner sum is the squared norm of one leaf, and the global norm is the square root of the sum of those per-leaf squared norms.

Take the gradient {"w": [1, 2], "b": 2}. The squared norm of w is $1 + 4 = 5$ and the squared norm of b is 4, so the global norm is $\sqrt{9} = 3$. With a limit of $c = 1.5$, the shared scale is $1.5 / 3 = 0.5$, and the clipped gradient is {"w": [0.5, 1.0], "b": 1.0}, whose global norm is exactly 1.5.

## Why the scale must be shared

A tempting shortcut clips every leaf separately by its own norm. On the same example, w has norm $\sqrt{5} \approx 2.2361$, so it would be scaled by $1.5 / 2.2361 \approx 0.6708$ to become $[0.6708, 1.3416]$, and b has norm 2, so it would become 1.5.

Two things went wrong. Before clipping, b was twice as large as the first entry of w, and afterwards it is about 2.236 times as large, so the direction of the whole gradient changed. The total length is also wrong, since the global norm of the per-leaf result is about 2.1213, well above the limit of 1.5. Using one scale for every leaf keeps the ratios between all entries of the model, which is why the global version is the standard one.

## When nothing should change

The scale is the smaller of 1 and $c / \lVert g \rVert$. When the global norm is already at or below the limit, the scale is 1 and the gradient passes through untouched. Clipping never makes a gradient longer, so a run with healthy gradients behaves exactly as if clipping were switched off.

A gradient of all zeros has a global norm of 0, which is below any positive limit, so it also passes through unchanged. A formula that always divides by the norm would compute $0 / 0$ here and produce NaN, a special floating-point value meaning not a number, which would then spread to every parameter it touches. Choosing the scale of 1 whenever the norm is within the limit avoids that division in the case where it matters.

## Making the decision inside jax.jit

Clipping is part of the training step, and the training step almost always runs under jax.jit. When jit traces a function, the arguments are replaced by tracers. A tracer is a placeholder that knows the shape and dtype of an array but has no concrete numbers, because the function is being turned into a compiled program that will later run on many different inputs.

A Python if statement needs a real True or False at the moment it runs. Comparing the traced norm with the limit gives a traced boolean instead, so JAX raises an error explaining that a traced value cannot be converted to a Python bool. The same thing happens with float() on a traced value, or with a NumPy function that needs real numbers.

The fix is to express the choice as array arithmetic, which becomes part of the compiled program. A selection function such as jnp.where takes a traced condition and two candidate values and picks one elementwise when the program runs. An expression built from jnp.minimum gives the same scale here. Both candidates are computed, and the one that is not chosen is simply discarded. If the norm is 0, the unused candidate $c / 0$ evaluates to infinity, and the selection still returns the scale of 1.

## Computing the norm in enough precision

Many training setups keep gradients in bfloat16, a 16-bit format with only 8 bits of precision, which is about 2 to 3 significant decimal digits. A norm is a sum of many squares, and long sums are where low precision hurts most.

Suppose a leaf has 1,000 entries equal to 0.1 in bfloat16. The bfloat16 value closest to 0.1 is 0.10009765625, so each exact square is about 0.0100195 and the true sum is about 10.0195. If a running total is kept in bfloat16 and every square is added one term at a time, the total stops growing at 4. At that size, the gap between neighbouring bfloat16 numbers is 0.03125, so adding a term of about 0.01 is rounded away every time. Calling jnp.sum on a bfloat16 array of the squares does better on CPU, because XLA adds the terms at a wider precision, but it still rounds the result to bfloat16 and returns 10. Converting each leaf to float32 before squaring and summing keeps about 7 significant digits and gives 10.0195.

The scale computed from a float32 norm is itself float32. Multiplying a bfloat16 leaf by a float32 array promotes the result to float32, so a clipped leaf would quietly double its memory and no longer match the dtype of its parameter. Casting each clipped leaf back to its original dtype keeps the gradient tree in the same format it arrived in.

## Where clipping sits in a training step

Clipping happens after the gradients are computed, after they are averaged over microbatches or devices, and before the optimizer turns them into parameter updates. It must see the final averaged gradient, because that is the vector the optimizer will actually use.

Training code usually logs the global norm before clipping at every step. A sudden jump in that number is one of the earliest signs of an unstable run, and a norm that stays far above the limit for many steps means the limit, or the learning rate, needs attention. Limits around 1.0 are common for large language models.

## Common mistakes to avoid

- **Clipping each leaf by its own norm.** Separate scales change the direction of the whole gradient and leave the global norm above the limit.
- **Always rescaling to the limit.** Multiplying by $c / \lVert g \rVert$ even when the norm is below the limit makes small gradients larger.
- **Using a Python if on the norm.** Inside jax.jit the norm is a tracer with no concrete value, so the comparison cannot become a Python bool.
- **Summing the norms of the leaves.** The global norm is the square root of the sum of squared entries, while a sum of per-leaf norms is always at least as large.
- **Summing squares in bfloat16.** A bfloat16 sum is rounded to about 3 significant digits, and a running total added one term at a time stops growing once each new term is smaller than half the gap between neighbouring values.
- **Returning float32 leaves for bfloat16 gradients.** The float32 scale promotes each leaf, so the result needs a cast back to the leaf's own dtype.