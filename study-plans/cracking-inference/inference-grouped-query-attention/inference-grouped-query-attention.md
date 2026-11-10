# Grouped-Query Attention

Multi-head attention gives every query head a private key and value head. Multi-query attention moves to the other extreme: all query heads share one key and one value head.

Grouped-query attention chooses a point between those extremes. It keeps several K/V heads, but fewer than the number of query heads. Each K/V head is shared by a small group of query heads.

You can picture a company with several research teams. Every researcher has an individual question, but researchers in the same team share one reference desk. There are fewer reference desks than researchers, yet the entire company is not forced to use a single desk.

## Why introduce groups?

MHA offers the most freedom because every head learns its own queries, keys, and values. It also retains the largest K/V representation.

MQA makes that representation much smaller by sharing one K/V head across all query heads. This is efficient, but it is an aggressive form of sharing.

GQA exposes a direct tradeoff. More K/V groups preserve more independent key and value representations. Fewer groups make the K/V representation smaller. The number of query heads stays unchanged, so the model still has many distinct ways to ask which tokens are relevant.

## The group assignment

Let $h_q$ be the number of query heads and $h_{\mathrm{kv}}$ the number of key/value heads. Equal-sized groups require

$$
h_q \bmod h_{\mathrm{kv}} = 0
$$

The number of query heads sharing one K/V head is

$$
g = \frac{h_q}{h_{\mathrm{kv}}}
$$

If there are 8 query heads and 2 K/V heads, then $g=4$. Query heads 0 through 3 use K/V head 0, while query heads 4 through 7 use K/V head 1.

If there are 8 query heads and 4 K/V heads, then $g=2$. The assignments become $(0,1)$, $(2,3)$, $(4,5)$, and $(6,7)$.

The mapping for query head $i$ is

$$
\operatorname{kv}(i) = \left\lfloor\frac{i}{g}\right\rfloor
$$

The floor operation creates consecutive groups. A round-robin order would represent a different mapping and would not match this problem.

## What is learned independently?

Every query head has its own query features. K/V head 0 and K/V head 1 also have different learned features from each other. Sharing occurs only among query heads assigned to the same K/V head.

For a query head $i$, attention is

$$
H_i
=
\operatorname{softmax}\left(
\frac{Q_iK_{\operatorname{kv}(i)}^{\mathsf T}}{\sqrt{d_k}} + M
\right)
V_{\operatorname{kv}(i)}
$$

Two query heads in the same group use identical keys and values, but their attention weights can differ because $Q_i$ differs. Two query heads in different groups can differ on both the query side and the K/V side.

This is why grouping does not mean that several queries are averaged into one. Every query head survives as an independent computation and returns its own output. The group only decides which key and value representation that query is allowed to use. Keeping those two ideas separate prevents most conceptual mistakes in GQA.

## How the projections encode GQA

As before, one query head has width

$$
d_k = \frac{d_{\mathrm{model}}}{h_q}
$$

The query projection produces all $h_q$ query heads, so its output width is $d_{\mathrm{model}}$. The key and value projections produce only $h_{\mathrm{kv}}$ heads, so each has output width $h_{\mathrm{kv}}d_k$.

This is the defining asymmetry. If K and V are projected to $h_qd_k$, the result is MHA. If they are projected to only $d_k$, the result is MQA. GQA uses an intermediate width.

After projection, keep K and V compact while identifying their $h_{\mathrm{kv}}$ heads. Then repeat each K/V head $g$ consecutive times along the head axis so that it lines up with the query heads assigned to it.

For four query heads and two K/V heads, the conceptual expansion is

$$
[K_0,K_1]
\longrightarrow
[K_0,K_0,K_1,K_1]
$$

Values follow the same expansion. The repeated entries represent shared data, not additional learned heads.

## A worked grouping example

Take a model width of 8, four query heads, and two K/V heads. Each head has width 2, and each K/V head serves two query heads.

The query projection yields four query heads: $Q_0,Q_1,Q_2,Q_3$. The compact key projection yields $K_0,K_1$, and the value projection yields $V_0,V_1$.

The attention calculations pair them as follows:

- $Q_0$ and $Q_1$ use $K_0,V_0$.
- $Q_2$ and $Q_3$ use $K_1,V_1$.

All four query heads return an output. Those four outputs are concatenated to restore the model width and then passed through $W_o$. K/V sharing changes how attention reads context, but it does not reduce the number of query-head outputs.

## GQA contains MHA and MQA as boundaries

The boundary cases provide the best correctness tests.

When $h_{\mathrm{kv}}=h_q$, every group has size one. Each query head receives its own K/V head, which is exactly MHA.

When $h_{\mathrm{kv}}=1$, there is one group containing all query heads. Every query shares one K/V head, which is exactly MQA.

An intermediate value is genuine GQA. Your implementation should reach all three cases through the same grouping rule rather than through separate attention algorithms.

## Why the K/V memory becomes smaller

For each token, MHA keeps K/V features for $h_q$ heads, while GQA keeps them for $h_{\mathrm{kv}}$ heads. The relative K/V size is therefore

$$
\frac{h_{\mathrm{kv}}}{h_q}
$$

With 8 query heads and 2 K/V heads, the K/V representation is one quarter of the MHA size. The attention score work still involves 8 query heads. GQA reduces the K/V side, not the number of query decisions.

This is why GQA is best understood as a capacity and memory compromise between MHA and MQA, not as a completely new attention formula.

## Causal attention

Grouping does not alter the time rule. In causal mode, every query head may use the current and earlier token positions but not later positions. Apply the mask to scores before softmax.

The same position mask can be used for every group. K/V sharing decides which representation a head reads, while causality decides which token positions it is allowed to read.

## The minimal tensor picture

Queries are arranged as $(B,h_q,S,d_k)$. Compact keys and values begin as $(B,h_{\mathrm{kv}},S,d_k)$ and are expanded along the head axis to align with the $h_q$ query heads. Attention then produces $h_q$ outputs, which are concatenated back to $(B,S,d_{\mathrm{model}})$.

The important thing to verify is not merely the expanded shape. The order must match the consecutive group assignment.

## Common mistakes to avoid

- Failing to reject head counts where $h_q$ is not divisible by $h_{\mathrm{kv}}$ creates uneven groups.
- Computing $d_k$ from the K/V head count gives query heads the wrong width.
- Alternating K/V heads in round-robin order creates the wrong mapping.
- Expanding keys and values differently makes a query score one memory but retrieve from another.
- Merging only $h_{\mathrm{kv}}$ outputs is wrong because outputs belong to query heads.
- The causal mask still belongs before softmax.

The central idea is small enough to remember: several questions share one memory within each group.
