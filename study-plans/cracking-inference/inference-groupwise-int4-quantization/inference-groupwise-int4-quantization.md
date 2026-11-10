# Group-Wise INT4 Quantization

Per-channel INT8 quantization gives each channel its own scale. Group-wise quantization makes the scale even more local by dividing each row into consecutive column groups.

This problem also reduces the code range from 8-bit to signed 4-bit values. Fewer codes mean coarser approximation, so local scales become especially useful for keeping one outlier from controlling an entire row.

## The signed INT4 range

The exercise uses codes from $-7$ through $7$. The range is symmetric around zero, just like the earlier INT8 problems.

For a group with absolute maximum $a_{r,g}$, the scale is

$$
s_{r,g}=\frac{a_{r,g}}{7}
$$

Each value in that row and group becomes

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{w}{s_{r,g}}\right),-7,7\right)
$$

Only fifteen integer levels are used: zero, seven positive levels, and seven negative levels.

## Storage dtype versus logical precision

PyTorch does not provide the ordinary packed signed 4-bit integer tensor required here. The function stores the codes in an INT8 tensor while restricting every value to $[-7,7]$.

This means the returned tensor demonstrates INT4 code values but does not pack two codes into one byte. Its physical tensor storage is still one byte per code.

Packing, bit manipulation, and specialized matrix-multiplication kernels are outside the contract. Correctness depends on code values, scales, and reconstruction.

## How groups are formed

For a weight matrix with $R$ rows and $C$ columns, each row is partitioned independently. Group $g$ starts at column $gG$, where $G$ is the group size.

With eight columns and group size three, every row has these groups:

- columns 0, 1, and 2,
- columns 3, 4, and 5,
- columns 6 and 7.

Groups are contiguous, begin at column zero, and never cross between rows.

The number of logical groups per row is

$$
N_g=\left\lceil\frac{C}{G}\right\rceil
$$

Every row has the same group boundaries because the matrix has a shared column count.

## Why local groups help

Consider one row:

$$
[1,2,7,\;0.1,0.2,0.7]
$$

With group size three, the first scale is $7/7=1$. Its values map exactly to codes $[1,2,7]$.

The second scale is $0.7/7=0.1$. Its values also map to $[1,2,7]$ even though their magnitudes are ten times smaller.

One scale for the full row would be one, causing $0.1$ and $0.2$ to round to zero. Separate groups preserve useful resolution for the smaller region.

An outlier affects only the group that contains it. Values in other groups keep scales based on their own ranges.

This locality is especially important with only fifteen available levels. When unrelated large and small values share one scale, the scale must cover the large values and the smaller ones may collapse onto the same few codes. Grouping gives those smaller regions a finer step without changing the logical four-bit range.

## The final partial group

When the column count is not divisible by group size, the final group contains only the remaining columns. It still receives an independent scale based only on those real values.

For five columns with group size two, the groups have widths two, two, and one. The final one-element group maps its nonzero value to code 7 or $-7$ because that value is its own absolute maximum.

Padding the group with invented zeros is unnecessary. Including values from an earlier group is wrong because it changes the maximum.

If the group size is larger than the complete row width, there is exactly one partial group containing the whole row.

## Full groups and reshape boundaries

Full groups all have width $G$, so their portion of the matrix can be viewed as rows by group count by group width. The group maximum is then taken across the final dimension.

The trailing partial group cannot be included in that fixed-width reshape. It is handled as its actual slice.

After quantization, full-group results return to the original row-column layout, and the partial result occupies the trailing columns. The final code and dequantized tensors must match the input shape exactly.

The group ordering is also part of the result. Scale column zero belongs to the first columns of every row, scale column one belongs to the next interval, and so on. Reordering scales by magnitude would destroy that positional relationship.

The implementation strategy may differ, but no column can be lost, duplicated, or moved to another group.

## Scale tensor shape

There is one scale for every row-group pair, so scales have shape

$$
(R,N_g)
$$

Scale entry $(r,g)$ belongs to row $r$ and the contiguous column interval for group $g$.

Dequantization must expand each scale across only its group’s columns:

$$
\hat{w}_{r,c}=q_{r,c}s_{r,g(c)}
$$

Using the correct scale values with the wrong group positions still produces a plausible-shaped but incorrect reconstruction.

## Zero groups

A group can be entirely zero even when other groups in the same row are nonzero. Its maximum is zero, so it receives fallback scale one.

All of its codes and reconstructed values remain zero. The fallback applies only to that group and must not alter neighboring group scales.

As with per-channel quantization, local granularity requires local zero handling.

## Granularity and metadata

Smaller groups adapt more closely to local magnitudes and can reduce rounding error. They also require more scale values.

Larger groups use less metadata, but each scale must cover a wider collection of weights. A single outlier then affects more values.

This exercise takes the group size as given. It does not choose an optimal group size, estimate model accuracy, or pack scales with weights.

Compared with per-channel INT8, group-wise INT4 uses fewer code levels and more scales within each row. Those two changes trade representation size and local accuracy in different directions.

## Cost and memory

Every weight participates in one maximum calculation, one quantization, and one reconstruction, so time is $O(RC)$.

Codes and reconstructed weights use $O(RC)$ elements. Scale metadata uses

$$
O\left(R\left\lceil\frac{C}{G}\right\rceil\right)
$$

values.

The returned INT8 code tensor does not realize packed 4-bit memory savings, because this problem is a deterministic simulation of the quantization values.

## Common mistakes to avoid

- Using the INT8 divisor 127 instead of the INT4 divisor 7 creates mostly unused codes.
- Allowing code $-8$ violates the specified symmetric range.
- Creating groups across flattened rows lets one row’s values affect another row’s scale.
- Dropping the final partial group loses trailing columns.
- Treating the partial group as though it had values from the previous group changes its scale.
- Applying one zero fallback to the whole matrix misses isolated zero groups.
- Returning packed bytes would change the required output shape and representation.
- Reconstructing with a row scale instead of the matching row-group scale removes group locality.

The useful mental model is a row divided into short rulers. Every ruler uses the same fifteen signed codes, but its step size is chosen from only the values covered by that group.
