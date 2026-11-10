# Multi-Query Attention

Multi-query attention is easiest to understand as one careful change to multi-head attention.

Multi-head attention gives every head its own queries, keys, and values. Multi-query attention keeps the many query heads, but all of them share one set of keys and one set of values.

That sentence contains the whole idea. The rest of the theory explains why this sharing is useful and how to implement it without accidentally turning it back into ordinary multi-head attention.

## Why keep many queries but share keys and values?

A query describes what the current head wants to find. Keeping several query heads preserves several ways of asking questions about the same context.

Keys and values play a different role. They describe the records being searched and the content those records return. MQA asks whether every query head truly needs its own private copy of that searchable memory.

Imagine several researchers using the same library. Each researcher can arrive with a different question, but they can all use the same catalog and the same books. In this analogy, query heads are the researchers, keys are the catalog entries, and values are the book contents.

The analogy is not a claim that the heads understand literal questions or books. It highlights the structural decision: requests remain separate while the searchable memory is shared.

## Compare MHA and MQA directly

Suppose there are four attention heads.

In multi-head attention, the structure is

- four query heads,
- four key heads,
- four value heads.

In multi-query attention, it becomes

- four query heads,
- one shared key head,
- one shared value head.

Each query head still computes its own attention weights. Query head 0 and query head 3 can focus on different tokens because their query vectors differ, even though both are compared with the same keys.

Sharing keys does not make the heads produce identical scores. A dot product depends on both sides of the comparison. The key can stay fixed while different queries produce different relevance judgments.

## How the projections reveal the design

Let the model width be $d_{\mathrm{model}}$ and let there be $h_q$ query heads. One query head has width

$$
d_k = \frac{d_{\mathrm{model}}}{h_q}
$$

The query projection still produces the full model width because it contains all $h_q$ query heads:

$$
Q = XW_q
$$

The key and value projections produce only one head each:

$$
K = XW_k,
\qquad
V = XW_v
$$

This difference is visible in the weights. $W_q$ maps model width to model width, while $W_k$ and $W_v$ map model width to only $d_k$ features.

If you project keys and values to the full model width and then split them into $h_q$ heads, you have implemented MHA, not MQA.

## How one K/V head serves every query head

After projection, queries are separated into $h_q$ heads. Keys and values are not split because they already contain exactly one head.

For query head $i$, the calculation is

$$
H_i
=
\operatorname{softmax}\left(
\frac{Q_iK^{\mathsf T}}{\sqrt{d_k}} + M
\right)V
$$

The same $K$ and $V$ appear for every $i$. Only $Q_i$ changes.

In a tensor implementation, give $K$ and $V$ a head dimension of size one. That dimension can broadcast across the $h_q$ query heads. Broadcasting does not create new learned K/V heads. It simply lets each query head use the same data.

You could physically copy the shared tensors, but that would hide the idea and waste memory. A size-one head axis states the intended sharing directly.

## A small example of different questions sharing memory

Suppose the shared keys describe three tokens. Query head 1 compares with them and produces attention weights

$$
[0.70,0.20,0.10]
$$

Query head 2 uses the same keys but has a different query, producing

$$
[0.10,0.25,0.65]
$$

Both heads retrieve from the same three values. The first output takes most of its content from value 1, while the second takes most from value 3. The shared memory has not prevented the heads from making different choices.

After every query head retrieves its weighted value mixture, the head outputs are concatenated. Since there are $h_q$ outputs of width $d_k$, the model width is restored. The final matrix $W_o$ then mixes information across those query heads.

## Why MQA matters during inference

During autoregressive generation, a model repeatedly attends to tokens it has already processed. Their keys and values can be retained so they do not have to be recomputed from scratch at every new token.

With MHA, each previous token contributes a key and a value for every head. With MQA, each previous token contributes only one shared key and one shared value. If there are $h_q$ query heads, the persistent K/V representation is smaller by a factor of $h_q$.

This reduction is the main motivation for MQA. The attention score work does not disappear because every query head still compares against the sequence. What becomes smaller is the K/V projection and the amount of K/V data that must be retained and read.

There is a tradeoff. MHA lets every head learn its own key and value representation. MQA asks all query heads to work with the same one. The compact representation is useful for inference, but it gives the model less freedom on the K/V side.

## Causal behavior stays the same

MQA does not change the meaning of a causal mask. At position $t$, every query head may use keys from positions up to $t$ and must ignore later positions.

Apply the mask to the scores before softmax. The same position mask can be shared across query heads, while the unmasked scores remain different because the queries differ.

When causal mode is disabled, every query position may attend to every key position.

## The one-head boundary

If $h_q=1$, there is one query head, one key head, and one value head. Nothing is being shared across multiple queries. The mechanism becomes single-head scaled dot-product attention followed by $W_o$.

This is a valuable test because a correct MQA implementation should agree with equivalent one-head MHA when their projections are the same.

## The minimal tensor picture

For input hidden states with shape $(B,S,d_{\mathrm{model}})$:

- queries are arranged as $(B,h_q,S,d_k)$,
- the shared keys and values are arranged as $(B,1,S,d_k)$,
- broadcasting produces one score matrix per query head,
- concatenating the head outputs returns $(B,S,d_{\mathrm{model}})$.

The size-one head dimension on K and V is the visual signature of MQA. It communicates that there is one K/V head, not $h_q$ separate heads that happen to contain similar numbers.

## Common mistakes to avoid

- Giving $W_k$ and $W_v$ model-width outputs creates ordinary MHA.
- Splitting K and V into $h_q$ learned heads contradicts the sharing rule.
- Copying along the sequence or batch axis instead of the head axis mixes unrelated data.
- Scaling by the model width is wrong; each query-key dot product has width $d_k$.
- The causal mask must be applied before softmax.
- Query-head outputs must be concatenated before $W_o$.

The clean mental model is: many learned ways to ask, one shared memory to ask against.
