# Int4 Weight Quantization

Int4 weight quantization is the last step of the EmbeddingGemma 2 recipe. Each backbone weight is stored as a 4-bit integer code plus a shared floating scale, and the paper releases two variants that differ only in how many weights share one scale. This problem builds both groupings with one symmetric scheme and makes the trade-off between them measurable.

## Sixteen levels and one scale

A signed 4-bit integer has sixteen values, from $-8$ to $7$. To store a real weight $w$ this way, you pick a scale $s$, store the integer nearest to $w / s$, and recover $\hat{w} = c \, s$ at inference time. The scale sets the grid spacing, so every weight in a group is rounded to a multiple of $s$, and the worst rounding error is $s / 2$.

The difficulty is that one scale has to cover the largest weight in its group. With the symmetric rule $s_g = \max_{w \in g} |w| / 7$, the largest magnitude lands exactly on code $7$ or $-7$, and everything else is placed on the same grid. If a single outlier is fifty times larger than its neighbors, the grid spacing grows fifty times, and the small weights all round to $0$. Transformer weight matrices do contain such outlier columns.

## Per-block against per-channel scales

Per-channel quantization, the Q4\_C variant, gives every output row one scale. It stores only $N_{\text{out}}$ scales for a matrix with shape $(N_{\text{out}}, N_{\text{in}})$, so its overhead is tiny, but an outlier anywhere in a row widens the grid for the entire row. Per-block quantization, the Q4\_0 variant, splits each row into runs of $k$ consecutive input columns and gives each run its own scale. An outlier then only coarsens the $k$ weights in its own block, at the price of storing $N_{\text{out}} N_{\text{in}} / k$ scales instead of $N_{\text{out}}$.

Table 1 of Schechter Vera et al., 2026 shows how close the two end up after quantization-aware training. On MTEB(Multilingual, v2) Mean(Task), the bfloat16 checkpoint scores 61.36, the per-block Q4\_0 checkpoint 60.96 and the per-channel Q4\_C checkpoint 60.84. Both stay within about half a point of the float model.

The paper names the two variants but does not publish their block size, their scale rule or their rounding. The exercise therefore chooses the format. It uses a symmetric absmax scale with divisor $7$, a clamp to the range $-8$ to $7$, round half to even, groups that run along the input dimension, and a zero scale with zero codes for any group whose largest magnitude is $0$. One consequence of the symmetric rule is that the code $-8$ is never produced, because $|w / s_g|$ never exceeds $7$. The clamp only matters when a different divisor or a numerical slip pushes a ratio past the range.

## Training the model to expect the grid

Rounding weights after training usually costs more quality than these numbers suggest, so EmbeddingGemma 2 uses quantization-aware training. A separate stage after model merging initializes a quantized copy from the float checkpoint and distills it from that checkpoint while the forward pass sees rounded weights. The vision encoder gets the same treatment with 8-bit weights and activations, written W8A8, while the audio encoder reuses the quantization from Gemma 4.

## Grouping one row two ways

Take the row $(0.3, -0.1, 1.0, -2.8)$. Per-channel quantization uses the peak $2.8$, so $s = 0.4$. The ratios are $0.75$, $-0.25$, $2.5$ and $-7$, which round to $1$, $0$, $2$ and $-7$, with $2.5$ going to the even code $2$. The dequantized row is $(0.4, 0, 0.8, -2.8)$, and the absolute errors sum to $0.1 + 0.1 + 0.2 + 0 = 0.4$.

Per-block quantization with $k = 2$ splits the row into $(0.3, -0.1)$ and $(1.0, -2.8)$. The first block has $s = 0.3 / 7 \approx 0.0429$, so its ratios are $7$ and about $-2.33$, giving codes $7$ and $-2$ and values $0.3$ and about $-0.0857$. The second block keeps $s = 0.4$ and the same codes as before. The total error falls to about $0.0143 + 0.2 = 0.2143$, almost half, because the small block no longer shares a grid with $-2.8$.

## Common mistakes to avoid

- **Dividing the peak by 8.** The largest magnitude then maps to a ratio of $8$, and the positive peak overflows the top code of $7$.
- **Using an asymmetric min and max range.** That scheme needs a stored zero point for every group, which the symmetric format here does not have.
- **Grouping along the output dimension.** The blocks in this exercise run across consecutive input columns of one row, which is the layout a matrix product reads.
- **Rounding with floor or with a plus one half shortcut.** Floor biases every weight downward, and adding one half breaks ties upward instead of toward the even code.
- **Using one scale for the whole tensor.** A single global peak lets the largest row decide the grid for every other row.
- **Dividing by a zero scale.** An all-zero group produces $0 / 0$ and leaves nonfinite values unless that group is handled before the division.
- **Returning the codes as floats.** The codes are integers, and keeping them in int8 is what makes the format 4-bit in spirit.