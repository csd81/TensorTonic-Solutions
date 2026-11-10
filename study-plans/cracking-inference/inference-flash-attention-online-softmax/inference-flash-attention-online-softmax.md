# FlashAttention with Online Softmax

Ordinary attention first builds every query-key score, stores the complete score matrix, applies softmax, and multiplies by the values. That description is mathematically clean, but the intermediate matrix becomes large when the sequence is long.

FlashAttention computes the same result without keeping the full score matrix. It processes small score tiles, carries forward a few statistics for each query row, and discards a tile after its contribution has been absorbed.

The word “exact” matters. This problem is not asking for approximate or sparse attention. Changing the block sizes may change the order of floating-point operations slightly, but it must not change the attention being computed.

## Why a naive block softmax is wrong

Suppose one query has scores $[2,4,1,3]$, processed as two blocks: $[2,4]$ and $[1,3]$.

If you apply softmax separately to each block, each block receives probability mass one. Combining those results would give the first half and second half equal total importance, even though score 4 should dominate score 1.

Softmax is normalized across all keys for a query. Blocking the computation must not split that global normalization into unrelated local normalizations.

The challenge is to process one block at a time while preserving the same numerator and denominator that a full softmax would produce.

## First understand numerically stable softmax

For scores $s_1,\ldots,s_n$, softmax can be written using their maximum $m$:

$$
p_j=\frac{e^{s_j-m}}{\sum_r e^{s_r-m}}
$$

Subtracting the same value from every score does not change the probabilities. Choosing the maximum keeps every exponent at most one and prevents large positive scores from overflowing.

To compute the final attention output, one row only needs three pieces of running state:

- $m$, the largest score seen so far,
- $\ell$, the sum of exponentials measured relative to $m$,
- $O$, the unnormalized weighted sum of value vectors measured relative to $m$.

After all key blocks have been processed, the attention output is $O/\ell$.

## The important rescaling idea

The reference maximum may change when a later block arrives. Earlier contributions were measured relative to the old maximum, so they must be converted to the new reference before new contributions are added.

Let the old maximum be $m$, and let $m'$ be the maximum after seeing the next block. Every old exponential is rescaled by

$$
\alpha=e^{m-m'}
$$

The running denominator and output accumulator both need this factor:

$$
\ell' = \alpha\ell + \sum_j e^{s_j-m'}
$$

$$
O' = \alpha O + \sum_j e^{s_j-m'}v_j
$$

The sums on the right contain only scores and values from the new block. Once they are added, $m'$, $\ell'$, and $O'$ summarize every block seen so far.

## A worked two-block example

Process scores $[1,2]$ first. Their maximum is 2, so the running denominator is

$$
\ell=e^{-1}+e^0\approx1.368
$$

Now the second block $[5,3]$ arrives. The new maximum is 5. The old denominator was measured relative to 2, so rescale it by

$$
e^{2-5}=e^{-3}\approx0.050
$$

The corrected old contribution is approximately $1.368\times0.050=0.068$. The new block contributes $e^0+e^{-2}\approx1.135$, giving a final denominator of approximately $1.203$.

If you skipped the rescaling, the first block would retain far too much weight because its exponentials would still use 2 as their reference while the second block uses 5.

The value accumulator $O$ follows the same correction. Whatever weighted value sum came from the first block must also be multiplied by $0.050$ before adding the second block’s weighted values.

## Turning online softmax into tiled attention

Choose a query block and keep it active while iterating through key/value blocks. Each pair produces a score tile rather than the full score matrix.

For every query row in that block:

1. compute scaled scores against the current key block,
2. apply any causal mask using absolute query and key positions,
3. update the running maximum,
4. rescale the old denominator and output accumulator,
5. add the current block’s exponential mass and weighted values.

After the final key block, divide each row’s accumulator by its denominator and write the completed query block to the output.

Each query row has its own $m$, $\ell$, and $O$. Rows cannot share a maximum because different queries can assign very different scores to the same keys.

## Causal masking across blocks

In causal attention, query position $q$ may not attend to key position $k$ when $k>q$. Block-local indices are not enough to test that rule.

For example, the first element of a key block beginning at position 4 has local index 0 but absolute index 4. A query at absolute position 2 must mask it. Use the block start offsets to compare absolute positions.

Masked scores behave like negative infinity. Some early key blocks may contain no valid key for a query row. That row should retain its previous running state until a valid score appears.

## Partial blocks are normal

Sequence lengths rarely divide every chosen block size exactly. If seven queries are processed in blocks of three, the last query block contains one row. The same applies independently to key blocks.

The online recurrence does not require full blocks. Slice each block up to the sequence boundary and derive its actual size from the slice. Padding a partial block with ordinary scores would introduce fake keys and change the output.

## Scope of this exercise

The original FlashAttention work is motivated by reducing data movement between GPU memory levels. Real kernels combine tiling, memory placement, and operation fusion.

This problem focuses on the algorithmic heart of the forward pass in PyTorch: tiled scores and online softmax. You are not being asked to write a CUDA or Triton kernel, reproduce a hardware-specific schedule, or benchmark kernel speed.

## Memory and computation

Dense attention stores a score matrix proportional to $BS_qS_k$. This implementation stores one score tile proportional to $B Q_b K_b$, plus row statistics and an output accumulator for the active query block.

The arithmetic remains proportional to $BS_qS_k(d_k+d_v)$. FlashAttention does not avoid the query-key comparisons. Its advantage comes from organizing exact attention so the full score matrix does not have to be materialized and repeatedly moved through memory.

## Common mistakes to avoid

- Applying softmax independently inside each key block changes the result.
- Rescaling $\ell$ but not $O$ leaves old values in the wrong numerical frame.
- Using one maximum for an entire query block mixes independent rows.
- Applying a causal mask with local block indices gives wrong cross-block visibility.
- Assuming block sizes divide sequence lengths loses final partial blocks.
- Dividing the accumulator before all key blocks have contributed breaks global normalization.

The core idea is bookkeeping, not approximation: retain exactly enough state to combine the next tile as though every score had been available at once.
