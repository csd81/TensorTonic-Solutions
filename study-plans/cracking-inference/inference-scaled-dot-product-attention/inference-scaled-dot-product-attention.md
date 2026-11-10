# Scaled Dot-Product Attention

Attention answers a simple question: when a token is being updated, which other tokens contain useful information for it?

Imagine the phrase “the animal was tired because it ran.” When the model updates the representation of “it,” the useful context is not every word equally. “Animal” is probably more relevant than “because.” Attention gives the model a learned way to make that choice for every token and then blend the useful information into a new representation.

Scaled dot-product attention is the small mathematical operation that performs this selection. Multi-head attention, multi-query attention, and the other mechanisms that follow all reuse this same core operation.

## Queries, keys, and values

A useful analogy is searching a library catalog. A search request is compared with the labels on many records. Once a matching record is found, the catalog returns the information stored in that record.

Attention separates those three roles:

- A **query** describes what the current token is looking for.
- A **key** describes what each available token can be matched on.
- A **value** contains the information that will actually be returned.

The query is compared with keys, never directly with values. This distinction matters. A key helps decide whether a token is relevant, while its value supplies the content used in the output.

For one query vector $q$ and one key vector $k$, the comparison is their dot product:

$$
q \cdot k
$$

Vectors pointing in similar directions produce a larger score. Vectors that are poorly aligned produce a smaller or negative score. These scores are learned notions of relevance because the queries and keys themselves come from learned projections.

## From scores to a useful output

Suppose one query is compared with three keys and produces scores $2.1$, $0.4$, and $-0.7$. Those numbers are not yet convenient weights. They may be negative, and they do not add up to one.

Softmax converts them into a distribution, perhaps $0.80$, $0.15$, and $0.05$. The three weights are positive and sum to one. The output for this query is then

$$
0.80v_1 + 0.15v_2 + 0.05v_3
$$

This weighted sum is the real result of attention. The query does not simply choose one token. It can take most of its information from one value while still mixing in smaller contributions from others.

If a different query is used, its comparisons with the same keys will change, so it receives a different mixture of the values. This is how each position builds its own context-aware representation.

## Why the dot product is scaled

If query and key vectors have many components, their dot product is a sum of many products. Its typical magnitude grows with the vector width. Large positive and negative scores make softmax extremely confident, so one position can receive almost all the probability even before the model has learned a useful distinction.

The Transformer corrects this by dividing every score by the square root of the query-key width, denoted by $d_k$:

$$
\text{scaled score} = \frac{q \cdot k}{\sqrt{d_k}}
$$

The square root is not an arbitrary decoration. Under common assumptions, the variance of the dot product grows in proportion to $d_k$, so division by $\sqrt{d_k}$ keeps score magnitudes roughly comparable as the vector width changes.

Scaling happens before softmax. Dividing the probabilities afterward would no longer produce a distribution that sums to one.

## The complete matrix operation

In practice, the model processes every query together. Stack queries into $Q$, keys into $K$, and values into $V$. The full operation is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}} + M\right)V
$$

Read the formula from left to right:

1. $QK^{\mathsf T}$ compares every query with every key.
2. Division by $\sqrt{d_k}$ keeps those comparisons well scaled.
3. The optional mask $M$ blocks relationships that are not allowed.
4. Softmax turns each query row into attention weights.
5. Multiplication by $V$ forms a weighted combination of value vectors.

Softmax must run across the key positions. Each query needs its own distribution over the available keys. Normalizing across queries would answer a different question and would mix decisions made by different output positions.

## A small numerical example

Take one query $q=[1,0]$, two keys $k_1=[1,0]$ and $k_2=[0,1]$, and values $v_1=[1,2]$ and $v_2=[3,4]$.

The raw comparisons are

$$
q \cdot k_1 = 1,
\qquad
q \cdot k_2 = 0
$$

Here $d_k=2$, so the scaled scores are approximately $[0.7071,0]$. Softmax turns them into weights of approximately $[0.6698,0.3302]$. The output is therefore

$$
0.6698[1,2] + 0.3302[3,4]
=
[1.6604,2.6604]
$$

The first value contributes more because its key matches the query more strongly, but the second value is not discarded. With $q=[0,1]$, the preference reverses because the second key now has the larger dot product.

This example captures the entire mechanism. The larger batched implementation performs the same reasoning for every query position at once.

## What the mask means

Sometimes a query must not use every key. In causal language modeling, a token cannot look at later tokens because those tokens have not been generated yet. A general attention mask can also hide padding or other invalid positions.

This problem uses a boolean mask where **true** means blocked. Conceptually, blocked scores are replaced by negative infinity before softmax. Their exponential becomes zero, so they receive exactly zero attention weight. The permitted positions are then normalized among themselves.

Masking after softmax is incorrect. It would remove some probabilities without redistributing their mass, so the remaining weights would no longer sum to one. Masking the values is also insufficient because the forbidden positions would still influence the normalization.

The problem guarantees that every query has at least one permitted key. Without that guarantee, a fully blocked row would ask softmax to normalize a row containing only negative infinity, which does not define a meaningful distribution.

## Self-attention and cross-attention

The same function supports two common situations.

In **self-attention**, queries, keys, and values describe the same sequence. The number of query positions and key positions is usually equal.

In **cross-attention**, the queries can come from one sequence while keys and values come from another. A short target sequence might attend to a longer source sequence. For that reason, your implementation must not assume that the query length equals the key length.

Keys and values must share a sequence length because every key identifies the value stored at the same position. Queries may have a different sequence length because they are the requests being answered.

## The few shapes that matter here

Let $B$ be the batch size, $S_q$ the number of queries, $S_k$ the number of keys, and $d_v$ the value width.

- $Q$ has shape $(B,S_q,d_k)$.
- $K$ has shape $(B,S_k,d_k)$, so its feature width matches $Q$.
- $V$ has shape $(B,S_k,d_v)$, so its sequence length matches $K$.
- The score matrix has shape $(B,S_q,S_k)$, and the output has shape $(B,S_q,d_v)$.

These shapes follow directly from the story: each of the $S_q$ queries assigns a weight to each of the $S_k$ key-value records, then combines value vectors of width $d_v$.

## Common mistakes to avoid

- Forgetting the transpose on $K$ prevents queries from being compared with all keys.
- Scaling by $\sqrt{d_v}$ is wrong because the scores are dot products over the query-key width $d_k$.
- Applying softmax along the wrong axis gives each query the wrong normalization.
- Treating **true** as permitted reverses the mask used by this problem.
- Applying the mask after softmax leaves invalid normalization.
- Assuming $S_q=S_k$ breaks valid cross-attention inputs.

A correct implementation should be understandable in the same order as the formula: compare, scale, mask, normalize, and combine values.
