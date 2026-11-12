# Reconstruct a Column-Parallel MLP Layer

When a single weight matrix is too large or too slow for one GPU, tensor parallelism splits it across several GPUs that each compute part of the layer. Column parallelism, introduced for transformers in Megatron-LM (Shoeybi et al., 2019), gives each GPU a block of the output columns. This problem centres on why each GPU can compute its slice independently and how the slices fit back together into the full layer.

## Why split a layer across GPUs

The MLP weights in a large transformer can hold hundreds of millions of values per layer. Data parallelism copies the whole model to every GPU, which stops working once one copy no longer fits. Tensor parallelism instead divides individual matrices, so each GPU stores and multiplies only a fraction of every layer, and the way a matrix is divided decides how much the GPUs must communicate.

Each participating GPU is called a rank, numbered 0 to $R - 1$.

## Splitting the output columns

A linear layer computes $Y = XW + b$, where $X$ has shape $(B, D_{\text{in}})$ with batch size $B$ and input width $D_{\text{in}}$, $W$ has shape $(D_{\text{in}}, D_{\text{out}})$ and $b$ has length $D_{\text{out}}$.

Output column $c$ of $Y$ depends only on column $c$ of $W$ and entry $c$ of $b$, together with the whole input $X$. So if $W$ is cut into blocks of columns $W_0, W_1, \ldots, W_{R-1}$ and the bias is cut the same way, rank $r$ can compute

$$
Y_r = X W_r + b_r
$$

on its own, with no communication. Here $W_r$ has shape $(D_{\text{in}}, D_r)$, $b_r$ has length $D_r$ and is added to every row of the batch, and $Y_r$ is rank $r$'s block of output features. The shard widths $D_r$ add up to $D_{\text{out}}$. When there is no bias, $b_r$ is simply zero.

Placing the blocks side by side in rank order rebuilds the full output:

$$
Y = \operatorname{concat}(Y_0, \ldots, Y_{R-1})
$$

The blocks do not need equal widths. Rank order is what decides where each block's features go.

## Why columns pair well with an activation

In a transformer MLP, the first linear layer is followed by an elementwise activation such as GELU. Because the activation treats every feature separately, each rank can apply it to its own block without seeing the others.

Megatron-LM exploits this. The first MLP layer is split by columns, so each rank holds a block of hidden features after the activation. The second layer is split by rows to match. Rows of the second weight matrix correspond to hidden features, so rank $r$ holds the rows that match its hidden block and produces a full-width partial sum. A single all-reduce at the end adds the partial sums, so the whole MLP needs only one communication step in the forward pass.

This problem covers only the column-parallel linear projection and the reconstruction of its full output, which in a real system would be an all-gather along the feature axis.

## A two-rank example with unequal widths

Take one input row $X = [3, -1]$. Rank 0 holds a single output column and rank 1 holds two:

$$
W_0 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \ b_0 = [1], \qquad W_1 = \begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix}, \ b_1 = [0, 0]
$$

Rank 0 computes $Y_0 = [3 - 2] + [1] = [2]$. Rank 1 computes $Y_1 = [-1, 2] + [0, 0] = [-1, 2]$. Concatenating in rank order gives $Y = [2, -1, 2]$.

Stacking the shards side by side gives the full weight $W = \begin{bmatrix} 1 & 0 & 1 \\ 2 & 1 & 1 \end{bmatrix}$ and bias $[1, 0, 0]$, and computing $XW + b$ directly gives the same $[2, -1, 2]$. The split changes where the work happens without changing the result.

## Common mistakes to avoid

- **Concatenating along the batch axis.** The blocks are output features, so they join along the last dimension.
- **Mixing up rank order.** Rank $r$'s features must appear in position $r$ of the concatenation.
- **Adding the full bias on every rank.** Each rank adds only its own bias block, matching its columns.
- **Assuming equal shard widths.** Ranks may hold different numbers of output columns.
- **Applying GELU before reconstructing.** The activation belongs to the next stage of the MLP, after this projection.
