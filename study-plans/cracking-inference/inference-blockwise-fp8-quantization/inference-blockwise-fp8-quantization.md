# Block-Wise FP8 Scaling

The previous problems mapped values to integer codes. FP8 is a low-precision floating-point format, so its representable values and rounding behavior are different from INT8 or INT4.

This exercise intentionally isolates one part of FP8 quantization: divide a matrix into two-dimensional blocks, choose one scale per block, and scale values into a supplied finite range. It returns a floating-point proxy rather than performing a real FP8 hardware cast.

## What the simulation represents

For every block, the output relation is

$$
x\approx x_{scaled}s_{block}
$$

The scaled proxy is constrained to the interval from $-F_{max}$ to $F_{max}$, where $F_{max}$ is supplied by the caller.

A real FP8 conversion would also round values to the discrete numbers representable by a particular FP8 format. That rounding is absent here. The exercise tests block scaling, saturation, and reconstruction in ordinary deterministic PyTorch operations.

This boundary is important: the returned proxy remains floating point and should not be described as a genuine FP8 tensor.

## Dividing a matrix into blocks

The input is a matrix with $R$ rows and $C$ columns. A block covers up to $R_b$ rows and $C_b$ columns.

For a $5\times7$ matrix with row block size 2 and column block size 3, row intervals are $[0,2)$, $[2,4)$, and $[4,5)$. Column intervals are $[0,3)$, $[3,6)$, and $[6,7)$.

Combining them produces nine logical blocks. The blocks along the bottom and right edges are smaller, and the bottom-right block contains only one element.

The number of blocks is

$$
N_r=\left\lceil\frac{R}{R_b}\right\rceil,\qquad N_c=\left\lceil\frac{C}{C_b}\right\rceil
$$

The scale tensor therefore has shape $(N_r,N_c)$.

## Choosing a local scale

For block $(i,j)$, find its absolute maximum:

$$
a_{i,j}=\max_{x\text{ in block }(i,j)}|x|
$$

For a nonzero block, the required scale is

$$
s_{i,j}=\frac{a_{i,j}}{F_{max}}
$$

Dividing the block by this scale maps its largest magnitude to $F_{max}$. Positive and negative values share the same scale.

Every scale depends only on its own block. A large value in the upper-left block must not reduce precision in the lower-right block.

## A two-block example

Suppose a row is divided into two column blocks:

$$
[1,2,4\;|\;100,200,400]
$$

Let $F_{max}=8$. The first block has scale $4/8=0.5$, producing proxy values $[2,4,8]$.

The second block has scale $400/8=50$, also producing $[2,4,8]$. Each region uses the available range according to its own magnitudes.

One scale for the entire row would be 50, causing values in the first block to become tiny proxy values. Local scales keep different magnitude regions independent.

## Scaling and saturation

The scaled proxy for each block is

$$
X_{scaled}=\operatorname{clip}\left(\frac{X}{s_{i,j}},-F_{max},F_{max}\right)
$$

Clipping guarantees that no proxy magnitude exceeds the supplied limit. Under exact arithmetic, a scale derived from the same block maximum already places every value inside the range. Clipping still protects the declared boundary against floating-point effects.

There is no rounding step in this problem. Adding integer-style rounding would discard information that the deterministic FP8 proxy is expected to retain.

The value of $F_{max}$ must come from the input. Hardcoding a familiar FP8 limit fails when the tests supply another positive value.

## Reconstruction

Each block is reconstructed with its matching scale:

$$
\hat{X}_{block}=X_{scaled}s_{i,j}
$$

Because this simulation does not round to actual FP8 representable values, reconstruction of finite non-clipped inputs will often be very close to the original matrix.

The reconstruction still needs to be computed from the returned proxy. Copying the original input would hide mistakes in scale placement or saturation.

Proxy and reconstructed tensors have the same shape as the input. Each position uses the scale selected by its row-block and column-block indices.

## Partial edge blocks

When dimensions do not divide evenly, the last block along an axis uses only existing elements. Its maximum must not include padding, values from a neighboring block, or an assumed full-block shape.

For a $3\times5$ matrix with $2\times2$ blocks, the bottom-right block has shape $1\times1$. It still receives exactly one scale.

Block slicing should stop at the actual row and column counts. Both edges can be partial at the same time.

If a block size exceeds the corresponding matrix dimension, there is one block along that axis. Large block sizes are valid and simply produce coarser scaling granularity.

## Zero blocks

An all-zero block has absolute maximum zero. Dividing by the normal scale would be invalid, so its required scale is one.

Its proxy and reconstruction remain zero. The fallback applies only to that block; other blocks continue to use maximum-derived scales.

Testing zero at block granularity matters because a matrix may contain one zero region next to several nonzero regions.

## Compared with group-wise INT4

Group-wise INT4 divides each row into one-dimensional column groups and produces integer codes from $-7$ through $7$. This problem divides across both rows and columns and returns a floating proxy bounded by $F_{max}$.

Both approaches use local scales to isolate different magnitude regions. Their layouts, code spaces, and rounding rules are different.

The absence of rounding here does not mean real FP8 has unlimited precision. It reflects the narrower simulation contract chosen to remain portable without GPU-only FP8 support.

## Granularity and scale metadata

Smaller blocks adapt to local ranges more closely but require more scale values. Larger blocks use less metadata while allowing an outlier to influence more elements.

For block sizes $R_b$ and $C_b$, scale storage grows as

$$
O\left(\left\lceil\frac{R}{R_b}\right\rceil\left\lceil\frac{C}{C_b}\right\rceil\right)
$$

This exercise accepts the block sizes and does not choose them or model the byte format of scale metadata.

## Cost and memory

Every matrix element participates in one block maximum, scaling, clipping, and reconstruction. Total time is $O(RC)$.

The proxy and reconstruction each use $O(RC)$ elements. Block scales use $O(N_rN_c)$ elements.

The function performs no device-specific kernel and makes no performance claim about real FP8 execution. It only models the requested numerical transformation.

## Common mistakes to avoid

- Using one scale per row ignores the two-dimensional block layout.
- Including neighboring values in an edge block changes its local maximum.
- Applying INT-style rounding violates the floating proxy contract.
- Hardcoding an FP8 maximum ignores the supplied range limit.
- Returning an FP8 dtype makes correctness depend on unavailable hardware or runner support.
- Handling zero only for the complete matrix misses isolated zero blocks.
- Reconstructing every block with one global scale loses the block mapping.
- Assuming block sizes divide the matrix drops or mis-scales edge elements.

The core idea is a grid of local rulers. Each matrix block chooses a ruler that fits its own largest magnitude into the supplied range, and the same ruler converts the proxy back to floating-point values.
