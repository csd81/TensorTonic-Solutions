# RMSNorm Forward Pass

RMSNorm is the normalization layer used in most modern language models, including Llama, Mistral and Gemma. It rescales each feature vector so its root mean square is about 1, then applies a learned per-feature scale. The heart of this problem is normalizing by the root mean square alone, without subtracting the mean, and keeping the arithmetic accurate when the input is stored at lower precision.

## Why transformers normalize

A deep transformer adds the output of every attention and MLP block back into a running residual stream. Without normalization, the size of the values in that stream can drift up or down from layer to layer, and the gradients drift with them. Training then becomes unstable or very slow.

A normalization layer fixes the scale of each token's feature vector before it enters a block. Every block then sees inputs of a predictable size, whatever happened in earlier layers.

## The RMSNorm formula

For one feature vector $x$ of width $d$, the root mean square is

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

where $\epsilon$ is a small positive constant. RMSNorm divides the vector by this value and multiplies elementwise by a learned scale vector $g$ of width $d$:

$$
\operatorname{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x)} \odot g
$$

The normalization runs independently over the last dimension. For a tensor of shape (batch, sequence, $d$), every token's vector gets its own RMS, and the batch and sequence dimensions are never mixed.

## What epsilon does

If every entry of $x$ were zero, the mean square would be zero and the division would be undefined. Adding $\epsilon$ inside the square root keeps the denominator positive, so the division is always defined. For ordinary vectors, $\epsilon$ is far smaller than the mean square and changes the result only in the last few decimal places.

The epsilon goes inside the square root, next to the mean square. Placing it outside, after the square root, gives slightly different numbers.

## How it differs from LayerNorm

LayerNorm, used in the original transformer and GPT-2, first subtracts the mean of the vector and then divides by its standard deviation. It also adds a learned bias after scaling.

RMSNorm drops both the mean subtraction and the bias. Zhang and Sennrich (2019) found that the rescaling does most of the stabilizing work and that recentering adds little. Removing it saves a reduction over the vector and a few operations per element, a saving that repeats in every layer for every token.

One useful property follows directly from the formula. Scaling the input by any positive constant $c$ leaves the output unchanged, apart from the tiny effect of $\epsilon$, because the RMS scales by the same $c$. The layer responds only to the direction of the vector, and its overall size has no effect on the output.

## A vector with a nonzero mean

Take $x = [2, -2, 1, 1]$, scale $g = [1, 1, 2, 2]$ and a very small $\epsilon$. The mean square is

$$
\frac{4 + 4 + 1 + 1}{4} = 2.5
$$

so the RMS is $\sqrt{2.5} \approx 1.581139$. Dividing gives the normalized vector $[1.264911, -1.264911, 0.632456, 0.632456]$, whose own mean square is 1.

Multiplying by $g$ doubles the last two entries, giving $[1.264911, -1.264911, 1.264911, 1.264911]$. The mean of $x$ was 0.5, and RMSNorm never subtracted it. LayerNorm would first subtract that mean of 0.5, so every entry would come out different.

## Why the arithmetic runs in float32

Models often store activations in bfloat16 to save memory. Bfloat16 keeps only about two to three significant decimal digits, and summing many squared values at that precision loses accuracy quickly, especially for wide vectors.

For this reason the reduction and the scaling are carried out in float32, and only the final result returns to the lower-precision format the rest of the model uses.

## Common mistakes to avoid

- **Subtracting the mean.** That turns RMSNorm into LayerNorm and changes every output.
- **Putting epsilon outside the square root.** The constant is added to the mean square before taking the root.
- **Normalizing over the wrong dimension.** Each vector along the last dimension gets its own RMS.
- **Using the sum instead of the mean.** Dividing by the root of the sum of squares leaves a factor of $\sqrt{d}$ in the result.
- **Doing the arithmetic in bfloat16.** Low-precision reductions lose accuracy, so the computation belongs in float32.
- **Forgetting the learned scale.** The normalized vector must still be multiplied elementwise by $g$.
