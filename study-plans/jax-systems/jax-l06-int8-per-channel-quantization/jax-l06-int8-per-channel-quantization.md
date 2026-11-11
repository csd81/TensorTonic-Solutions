# Int8 Matmul with Per-Channel Scales

Serving a large model is mostly a matter of reading weights from memory and multiplying them, so storing weights and activations as 8-bit integers instead of 32-bit floats cuts that memory to a quarter and lets the hardware use its fast integer multipliers. The price is that every number is snapped to one of a small set of levels, and the sums of integer products need more bits than the inputs have. This page builds the W8A8 recipe used for int8 inference, where W8 means 8-bit weights and A8 means 8-bit activations, starting from the rounding of a single number and ending with a matmul whose integer sums run in int32 under jax.jit.

## What an int8 code stores

Symmetric quantization represents a real number $v$ by an integer code $q$ and a positive float scale $s$, with $v \approx q \, s$. The code is stored as int8, a signed 8-bit integer type that holds the whole numbers from -128 to 127. Symmetric schemes use only -127 to 127, so the range is the same on both sides of zero and zero itself is the exact code 0.

The scale comes from the largest absolute value in the group of numbers that share it, called the absmax. Setting $s = \text{absmax} / 127$ sends the largest value to exactly 127 or -127 and spreads every other value over the levels in between. Take the group [0.5, -1.27, 0.3]. Its absmax is 1.27, so its scale is 0.01, and the three values divide to 50, -127 and about 30.000002, which round to the codes 50, -127 and 30. Each code is at most one half away from the true ratio, so each dequantized value $q \, s$ is at most half a scale away from the original.

A group whose values are all zero has absmax 0, which would make the scale 0 and the division undefined. Giving such a group the scale 1 keeps every code at 0 and every dequantized value at 0.

## One scale per output column

The weight matrix w has shape (D_in, D_out) and is applied as x times w, so column $j$ produces output channel $j$. Different channels often have very different ranges. A single scale for the whole matrix, called a per-tensor scale, is set by the largest value anywhere, and it crushes the small columns onto a few levels.

Take w with the columns [0.5, -1.27, 0.3] and [-0.023, 0.011, 0.041]. A per-tensor scale is 1.27 / 127 = 0.01, and the second column then rounds to the codes -2, 1 and 4, which dequantize to -0.02, 0.01 and 0.04. The first of those is about 13 percent off. With its own scale of 0.041 / 127, about 0.000323, the same column gets the codes -71, 34 and 127, which dequantize to about -0.022921, 0.010976 and 0.041, all within 0.4 percent of the originals. Giving each output column its own scale is what per-channel quantization means here. It costs one extra float per column and keeps every channel on the full set of levels.

## One scale per token for the activations

The activations x have shape (B, D_in), with one row per token. Weights are fixed after training, so their scales can be computed once, while activations change on every call, so their scales are computed on the fly from each row. With one scale per row, a token with unusually large values only coarsens its own codes and leaves the resolution of the other tokens in the batch alone.

In the running example, the row x = [1.2, 2.0, -0.9] has absmax 2.0, so its scale is 2 / 127, about 0.015748, and its codes are 76, 127 and -57.

## Why the scales factor out of the sum

Replacing each value by its dequantized form gives

$$
y_{bj} \approx \sum_{i} \left( q^x_{bi} \, s^x_b \right) \left( q^w_{ij} \, s^w_j \right) = s^x_b \, s^w_j \sum_{i} q^x_{bi} \, q^w_{ij},
$$

where $b$ indexes a row of x, $j$ indexes a column of w, $i$ runs over the shared inner dimension, $s^x_b$ is the scale of row $b$ and $s^w_j$ is the scale of column $j$. Neither scale depends on $i$, so both move outside the sum. What remains inside is a sum of products of integers, which an integer matmul computes directly, and the scales are applied once per output entry afterwards.

This is why the scales are attached to these particular groups. A scale for each input feature of x, one per column of x, would depend on $i$ and could not leave the sum, so the matmul could no longer run purely on integers. Per-row scales for x and per-column scales for w are the finest choices that keep it integer.

In the running example the first integer sum is 76 × 50 + 127 × (-127) + (-57) × 30 = -14039, and the second is -8317. Multiplying by 0.015748 and by the column scales 0.01 and 0.000323 gives y of about [-2.210866, -0.042284], close to the float product of x and w, which is about [-2.21, -0.0425]. With the per-tensor weight scale, the second entry would come out as about -0.039843 instead.

## Why the sum needs 32 bits

A product of two int8 codes can reach 127 × 127 = 16,129 in magnitude, which already does not fit in int8. A 16-bit integer (int16) reaches 32,767, so a sum of only three such products can overflow it. Integer overflow in JAX wraps around silently. For a row of eight codes equal to 127 multiplied by a column of eight codes equal to 127, the true sum is 129,032. An int8 result gives 8 and an int16 result gives -2,040, and no error is raised in either case.

A 32-bit integer (int32) reaches 2,147,483,647, enough for about 133,000 products of the largest size, which covers the inner dimension of any real layer. This is why the int8 matrix units on GPUs and TPUs multiply 8-bit inputs and accumulate in int32. With D_in up to 64 in this problem, a sum can reach 64 × 16,129 = 1,032,256 in magnitude.

## Telling JAX to accumulate in int32

By default, multiplying two int8 arrays with jnp.matmul returns an int8 result, which is the overflowing case above. Both jnp.matmul and jax.lax.dot_general accept an argument called preferred_element_type, and setting it to jnp.int32 asks for int32 accumulation and an int32 result while the inputs stay int8. The function jax.lax.dot_general is the general contraction that every JAX matmul lowers to. Its dimension_numbers argument names the axes that are summed and the axes that are batch axes, and a plain matrix product sums axis 1 of the first operand against axis 0 of the second.

The function jax.make_jaxpr shows the program that JAX records while tracing a function, called its jaxpr. For a correct solution traced on a (2, 3) input and a (3, 2) weight, the matmul appears as a single dot_general equation whose inputs have the types i8[2,3] and i8[3,2], whose output has the type i32[2,2], and whose parameters include preferred_element_type=int32. Under jax.jit this whole program compiles once per input shape. The rounding, the clipping and the choice of scale 1 for an all-zero group all have to be traced operations, which is why jnp.where takes the place of a Python if.

## Exact integers versus dequantized floats

A tempting shortcut is to dequantize both operands back to float32 and multiply them with an ordinary float matmul. In exact arithmetic that gives the same y, although floating-point rounding makes the two results differ. Every dequantized value $q \, s$ is rounded to float32, every product is rounded again, and the running sum is rounded at each step. The integer path has an exact sum, followed by one conversion to float32, which is exact for integers up to 16,777,216, and two rounded multiplies by the scales.

The difference is largest when big products cancel. Take one row of x and one column of w with 16 entries each, where the absolute values of the 16 products add up to 87,422 while their integer sum is -2. With the scales 0.01 and 0.003, the exact output is about -5.99999992e-05, and the int32 path gives -5.99999985e-05, a relative error of about 1.2e-8. The float32 matmul of the dequantized values gives about -6.00814819e-05, a relative error of about 0.00136. Over a batch of such inputs with D_in up to 64, the int32 path stayed within a relative error of about 1.6e-7 on every entry, while the dequantized float path was off by as much as 0.0034.

Speed and memory are the other reasons to stay in integers. A float matmul on dequantized values gives up the int8 matrix units entirely, and the float32 copies it builds are as large as the original weights.

## Common mistakes to avoid

- Calling jnp.matmul on int8 arrays without preferred_element_type gives int8 sums that wrap around as soon as a sum passes 127.
- Asking for int16 accumulation only moves the overflow to 32,767, which three of the largest products already exceed.
- Dequantizing to float32 and multiplying in float32 skips the integer matmul, and its results drift from the exact integer path when products cancel.
- Taking the weight absmax over axis 1 gives one scale per input row of w, and such a scale cannot be pulled out of the sum over that same axis.
- Quantizing x with one scale per input feature, over axis 0, runs into the same problem, because those scales also depend on the summed index.
- Using a single scale for the whole weight matrix collapses the small columns onto a handful of levels.
- Casting to int8 without rounding first truncates toward zero, which pulls every code one step closer to 0 about half of the time.
- Dividing by an absmax of zero produces NaN for an all-zero row or column, so those groups need the scale 1.