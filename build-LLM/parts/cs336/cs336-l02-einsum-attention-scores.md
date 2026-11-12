# Named-Dimension Batched Attention Scores

Attention scores measure how strongly each query position should look at each key position. Multi-head attention computes these scores separately for several heads, each working on its own slice of the model width. The heart of this problem is splitting the width into heads correctly and contracting only the right axis, so every head gets its own independent score matrix.

## How heads live inside the model width

Queries arrive with shape $(B, S_q, D)$, where $B$ is the batch size, $S_q$ is the number of query positions and $D$ is the model width. Keys have shape $(B, S_k, D)$ with $S_k$ key positions. With $H$ heads, each head has width

$$
d_h = \frac{D}{H}
$$

Each token's width-$D$ vector holds the $H$ heads as consecutive blocks. Head 0 owns coordinates 0 to $d_h - 1$, head 1 owns the next $d_h$ coordinates, and so on.

Splitting the last axis of size $D$ into two axes of sizes $H$ and $d_h$, in that order, exposes the heads without moving any values. Each head then has its own slice of width $d_h$ for every token, in both the queries and the keys.

The order of the two new axes matters. Splitting into $(d_h, H)$ instead would interleave the heads, assigning coordinates 0, $H$, $2H$ and so on to head 0. The shapes would still line up, but every head would see the wrong coordinates.

Inputs that come out of a transpose are not laid out contiguously in memory, so the split has to work on any stride pattern. A reshape always does, because it copies when it has to.

## What a score is

Within one head, write $q_i$ for the width-$d_h$ slice of query position $i$ and $k_j$ for the slice of key position $j$. Their score is the dot product divided by $\sqrt{d_h}$:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}
$$

A large score means the query and key point in a similar direction. In a full attention layer these scores later go through a softmax to become weights, and this problem stops just before that step.

## Why divide by the square root of the head width

If the coordinates of $q_i$ and $k_j$ are independent with mean 0 and variance 1, their dot product is a sum of $d_h$ terms, each with variance 1. The sum therefore has variance $d_h$ and a typical size of $\sqrt{d_h}$.

Without scaling, wider heads would produce larger scores, and a softmax over large scores pushes almost all the weight onto one key. Its gradients then become tiny and training stalls. Dividing by $\sqrt{d_h}$ keeps the scores at roughly unit scale whatever the head width.

The divisor is $\sqrt{d_h}$ because each head's dot product sums over only its own $d_h$ coordinates.

## Reading an einsum contraction

Einstein summation describes a tensor product by giving each axis a letter. Every letter that appears in the output is kept. A kept letter that appears in both inputs is matched elementwise, and a letter that appears in the inputs but not in the output is summed over.

Ordinary matrix multiplication is written $\texttt{ij,jk->ik}$. The letter j appears in both inputs but not in the output, so it is summed, while i and k each come from one input and stay in the result.

For attention scores, the batch, head, query position and key position letters all appear in the output. The head-width letter appears in both inputs but not in the output, so it is the one that gets summed. That single summed axis is exactly the dot product inside each head:

$$
A_{b,h,i,j} = \frac{1}{\sqrt{d_h}} \sum_{r=1}^{d_h} Q_{b,h,i,r} K_{b,h,j,r}
$$

Here $Q$ and $K$ are the queries and keys viewed with separate head and head-width axes, $r$ runs over coordinates within a head, and the output letters decide the final axis order, $(B, H, S_q, S_k)$. Query and key lengths can differ, for example during generation, when one new query attends to all earlier keys.

## A small two-head example

Take one batch, one query token and two key tokens, with $D = 4$ and $H = 2$, so $d_h = 2$. The full token vectors are

$$
x_q = [1, 1, 2, 0], \qquad x_{k_1} = [1, 0, 0, 1], \qquad x_{k_2} = [1, 1, 1, 1]
$$

Head 0 uses the first two coordinates. Its query slice is $[1, 1]$, so its dot products with the key slices $[1, 0]$ and $[1, 1]$ are 1 and 2.

Head 1 uses the last two coordinates. Its query slice is $[2, 0]$, so its dot products with $[0, 1]$ and $[1, 1]$ are 0 and 2.

Dividing by $\sqrt{2}$ gives the scores

$$
A_{\text{head } 0} = [0.707107, \ 1.414214], \qquad A_{\text{head } 1} = [0, \ 1.414214]
$$

The two heads disagree about the first key. That independence is the point of multiple heads, since each one can learn to look for a different kind of relationship.

## Common mistakes to avoid

- **Scaling by the full width.** The divisor is $\sqrt{d_h}$, because each head sums over only $d_h$ coordinates.
- **Splitting heads in the wrong order.** The width splits into $(H, d_h)$, since heads are consecutive blocks.
- **Summing over the wrong letter.** Only the head-width axis is contracted, and batch, head and both position axes stay in the output.
- **Returning the wrong axis order.** The result must be $(B, H, S_q, S_k)$, with heads before positions.
- **Assuming the input is contiguous.** Operations such as view can fail on some transposed layouts, while reshape always works.
- **Applying a softmax.** This problem returns the scaled scores before any softmax or mask.
