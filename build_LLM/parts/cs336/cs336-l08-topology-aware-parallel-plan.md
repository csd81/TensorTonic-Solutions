# Choose a Topology-Aware Parallel Plan

Large training runs combine several kinds of parallelism at once, and each kind has a different memory saving and a different communication pattern. The best combination depends on the cluster, because GPUs inside a node talk over fast links while traffic between nodes is much slower. This problem centres on searching every combination of data, tensor, pipeline and context parallelism, keeping the ones that fit, and picking the one that sends the least data over the slow links.

## Four ways to split the work

Each kind of parallelism has a degree, the number of GPUs it spreads one piece of work over.

- **Data parallelism, degree $d$.** Each group of GPUs gets a different slice of the batch, and their gradients are averaged.
- **Tensor parallelism, degree $t$.** Individual weight matrices are split so each GPU computes part of every layer.
- **Pipeline parallelism, degree $p$.** Each GPU holds a consecutive block of layers.
- **Context parallelism, degree $c$.** Each sequence is split along its length, which matters for very long contexts.

A plan uses every GPU exactly once, so the degrees must multiply to the GPU count $G$:

$$
d \, t \, p \, c = G
$$

Divisibility rules follow from what each kind splits. The data degree must divide the global batch size $B$, and the context degree must divide the sequence length. Very long sequences also come with a minimum context degree, because one sequence's attention activations would not fit on a GPU without splitting it at least that many ways.

## Topology decides where traffic goes

Tensor and context parallelism exchange activations inside every layer, many times per step. They need the fast links within a node, so this exercise's cost model requires $t \cdot c$ to be at most the number of GPUs per node. Data and pipeline parallelism communicate less often, once per step for gradients or once per stage boundary for activations, so they can span nodes.

## Which degrees shrink memory

With $S$ bytes of model state and $A$ bytes of activations per sample, each GPU holds

$$
M_s = \left\lceil \frac{S}{tp} \right\rceil, \qquad M_a = \left\lceil \frac{A (B / d)}{tc} \right\rceil
$$

Tensor and pipeline parallelism divide the model state, because each GPU holds only part of the weights. Activations depend on the local batch $B / d$, and tensor and context parallelism divide them further. The cost model leaves the pipeline degree out of the activation term as a simplification, since each stage still holds activations for every microbatch in flight. Data parallelism alone leaves the full model state on every GPU. A plan is feasible only if $M_s + M_a$ fits in one GPU's memory.

## Scoring by slow traffic first

The cost model estimates inter-node and intra-node bytes:

$$
I_{\text{out}} = 2 M_s \frac{d - 1}{d} + M_a \frac{p - 1}{p}, \qquad I_{\text{in}} = 2 M_a \left( \frac{t - 1}{t} + \frac{c - 1}{c} \right)
$$

The first term of $I_{\text{out}}$ is a ring all-reduce over the data-parallel group, using the per-GPU model-state shard $M_s$ as the stand-in for gradient bytes. In a ring all-reduce each GPU sends about $2(d - 1)/d$ times the data, which is where that factor comes from. The second term is activations crossing pipeline boundaries, and $I_{\text{in}}$ covers the tensor and context exchanges inside a node.

Inter-node traffic is compared first because it is the slowest, and the remaining entries break ties. Among equals, fewer pipeline stages mean a smaller bubble, the idle time at the start and end of each pipeline step. Larger data parallelism is preferred next because it needs no changes to the layers, and smaller tensor and context degrees break any remaining tie.

## A four-GPU search

Take 4 GPUs with 2 per node, $S = 600$ bytes of model state, $A = 50$ bytes per sample, batch size 8, sequence length 16, 500 bytes of memory per GPU and a minimum context degree of 1.

Several plans can be ruled out before any memory arithmetic. Any plan with $t \cdot c = 4$ needs more GPUs than one node has. Pure data parallelism, $(d, t, p, c) = (4, 1, 1, 1)$, keeps all 600 bytes of model state on each GPU and does not fit.

Three feasible plans show the trade-offs:

- $(2, 1, 2, 1)$ needs $300 + 200 = 500$ bytes, with $I_{\text{out}} = 300 + 100 = 400$ and $I_{\text{in}} = 0$.
- $(1, 2, 2, 1)$ needs $150 + 200 = 350$ bytes, with $I_{\text{out}} = 100$ and $I_{\text{in}} = 200$.
- $(1, 1, 2, 2)$ needs $300 + 200 = 500$ bytes, also with $I_{\text{out}} = 100$ and $I_{\text{in}} = 200$.

The last two tie on both traffic terms, on $p = 2$ and on $d = 1$. The tie falls to the smaller tensor degree, so the chosen plan is $(1, 1, 2, 2)$, using exactly the full 500 bytes of memory. Data parallelism lost because its gradient all-reduce crosses nodes.

## Common mistakes to avoid

- **Letting tensor and context parallelism span nodes.** Their product must fit within one node.
- **Ignoring divisibility.** The data degree must divide the batch and the context degree must divide the sequence length.
- **Forgetting the minimum context degree.** Plans below it are infeasible even if they fit in memory.
- **Rounding memory down.** Both memory terms round up to whole bytes.
- **Scoring in the wrong order.** Inter-node bytes decide first, and the later entries only break ties.
- **Ignoring memory when traffic looks good.** Pure data parallelism has no intra-node traffic but keeps the full model state on every GPU, so it often fails the memory check.
