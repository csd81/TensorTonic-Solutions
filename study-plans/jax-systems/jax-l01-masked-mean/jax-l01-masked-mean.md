# Masked Mean Without Boolean Indexing

Batches in machine learning often mix real data with padding. Sentences of different lengths are padded to one common length, and a mask records which positions are real. In NumPy you would pick the valid entries with boolean indexing and average them, and under the JAX compiler that approach fails. This page explains why, and builds a masked mean from operations whose output shapes never depend on the data.

## Boolean indexing makes the shape depend on the data

Take one row x = [2, 4, 6, 8] with the mask [True, True, False, False]. In NumPy, and in JAX outside of jit, x[mask] returns [2, 4], an array of shape (2,). With the mask [True, True, True, False], the same expression returns [2, 4, 6], of shape (3,).

The output length equals the number of True entries. That number is part of the mask's values, so the shape of the result cannot be known until the mask values are known.

## Why jit needs every shape in advance

The function jax.jit compiles by tracing. JAX calls your function with tracers, which are placeholders that carry a shape and a dtype but no values, and records the operations applied to them. A print statement inside a jitted function called with a float32 array of shape (2, 4) shows JitTracer(float32[2,4]) instead of numbers.

XLA, the compiler behind jit, plans the memory and the work for exact shapes, so every intermediate shape must follow from the input shapes alone. The expression x[mask] cannot be planned, and under jit it raises a NonConcreteBooleanIndexError. A Python if statement on a traced value fails for a related reason, since choosing a branch needs a concrete True or False, and it raises a TracerBoolConversionError.

Arguments marked as static work differently. An argument listed in the static_argnames option of jax.jit is passed as an ordinary Python value during tracing, so the function can use it to choose shapes or reduction axes. Each new value of a static argument triggers a new compilation. The axis in this problem is static, while x and mask are traced.

## Selecting instead of removing

The function jnp.where(mask, x, 0) keeps every position and replaces each invalid entry with zero. For the row above it returns [2, 4, 0, 0], which always has the shape of x. The sum of that array is 6, the sum of the mask is 2, because each True counts as 1, and the mean of the valid entries is $6 / 2 = 3$. The answer matches NumPy's x[mask].mean(), and no shape ever depended on the mask.

Written as a formula, let $m_j$ be 1 where the mask is True and 0 elsewhere, and let $s_j$ be the selected value at position $j$ along the axis.

$$
s_j = \begin{cases} x_j & \text{if } m_j = 1 \\ 0 & \text{otherwise} \end{cases} \qquad \bar{x} = \frac{\sum_j s_j}{\sum_j m_j}
$$

Here $\bar{x}$ is the masked mean of the slice. The numerator and the denominator are both plain reductions over a fixed axis, which is exactly what a compiler can plan.

## Padding values must not leak

Padding positions can hold anything. They might be zeros, copies of the last real value, or non-finite values left behind by earlier computations, such as infinity from a masked maximum. Multiplying by a mask of ones and zeros looks equivalent to selection, but in floating point infinity times zero is NaN. When the mask arrives as float32 data, for example from a data pipeline that stores masks as ones and zeros, x = [1, infinity] times the mask [1, 0] gives [1, NaN], and the NaN spreads into the sum. A bool mask converted to float32 inside a jitted function happens to be compiled into a select, so the same product stays finite there, while the identical code run outside jit gives [1, NaN]. Selecting with jnp.where states the intent directly and returns [1, 0] in every one of these cases, whatever the padding held.

## Slices with no valid entries

A slice whose mask is all False has a sum of 0 and a count of 0, and in floating point $0 / 0$ is NaN. The mean of nothing has no natural value, so a function has to choose one. This problem uses 0.0, which is also a common choice when pooling an empty sequence.

The fix uses selection again. Clamp the count so that the division never sees zero, then choose 0.0 wherever the true count was zero. Take the rows [5, 1] and [3, 7] with the masks [False, False] and [True, True]. The selected sums are [0, 10] and the counts are [0, 2], so the clamped counts are [1, 2], the divisions give [0, 5], and the final choice keeps [0, 5].

A Python if could not express this decision even outside jit, because one row is empty while the other is not. The choice has to be made separately for every slice, which is what an elementwise jnp.where does.

## Axes and shapes

A reduction along an axis removes that axis from the shape. Summing an array of shape (2, 3, 4) along axis 1 gives shape (2, 4), and summing along axis -1 gives shape (2, 3), because a negative axis counts from the end. The selected sum and the count must reduce along the same axis so that their entries line up one to one.

Summing a bool array gives an int32 count in JAX by default. Dividing a float32 sum by an int32 count gives float32, because JAX promotes the integer to the floating-point type, so the mean keeps the dtype of x.

## A two-dimensional example along axis 0

Reducing along axis 0 averages down each column instead of across each row. Take x with the rows [1, 4], [2, 5] and [6, 8], and a mask with the rows [True, False], [True, False] and [False, False]. Column 0 has two valid entries, 1 and 2, and column 1 has none. The selected sums along axis 0 are [3, 0] and the counts are [2, 0], so the result is [1.5, 0]. The 6 in column 0 and every entry of column 1 are padding, and none of them reaches the answer.

## Other reductions with a mask

The same idea extends to other reductions, as long as each padding entry is replaced by a value that cannot change the answer. For a masked maximum the neutral value is negative infinity, so selecting from [3, 9, 4] with the mask [True, False, True] and taking the maximum gives 4. Many jax.numpy reductions also accept a where argument. Called on the same values and mask, jnp.mean with that argument returns 3.5, and with an all-False mask it returns NaN, so an empty slice still needs its own choice of value.

## Where this pattern appears

Sentence encoders average token embeddings over the real tokens only, language model losses average per-token losses over the positions that are not padding, and evaluation code computes accuracy over padded batches. All of them reduce over a mask with a fixed shape, and the next problem in this section builds such padded batches and their masks.

## Common mistakes to avoid

- **Indexing with the mask.** The result of x[mask] has a shape that depends on the mask values, so it fails under jit.
- **Dividing by the length of the axis.** Averaging the selected array over the whole axis counts padding positions in the denominator.
- **Leaving empty slices unguarded.** The division 0/0 gives NaN for any slice with no valid entries.
- **Branching with a Python if.** Testing a traced value raises an error under jit and cannot decide separately for each slice.
- **Multiplying by a float mask that arrives as data.** Infinity or NaN in a padding position times zero gives NaN, which spreads into the sum.
- **Reducing the count along a different axis.** The totals and the counts then belong to different slices and no longer line up.