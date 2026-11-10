# Expert-Parallel Dispatch Across Ranks

In a sparse Mixture of Experts layer, each token is routed to one or more experts. Under expert parallelism, different experts live on different ranks, so routed token copies must be grouped by destination before an all-to-all exchange.

This problem prepares that communication layout. It does not run the network exchange or the experts. It returns the permutations, counts, expert IDs, and weights needed to send and later restore the routed items.

## Begin with token-route pairs

Selected expert IDs and routing weights have shape $(T,k)$. Each token contributes $k$ logical routes.

For token $t$ and route slot $r$, the canonical flattened position is

$$
i=tk+r
$$

Flattening in row-major order lists every route for token 0, then every route for token 1, and so on.

This flattened sequence defines original order. Every permutation and inverse permutation is interpreted relative to it.

## Map experts to destination ranks

The expert-to-rank tensor says which rank owns each expert. For flattened expert ID $e_i$, destination rank is

$$
d_i=owner[e_i]
$$

Several experts may belong to one rank, and the same expert can appear in many routes. Destination is determined by ownership, not by token index or route weight.

The world size may include ranks that own no selected route for this input.

## A concrete example

Suppose selected expert IDs are

$$
\begin{bmatrix}
0&2\\
1&0
\end{bmatrix}
$$

The flattened expert sequence is $[0,2,1,0]$.

Let experts 0, 1, and 2 belong to ranks $[1,0,1]$. Destination ranks are therefore

$$
[1,1,0,1]
$$

Grouping by ascending destination rank places original flat position 2 first, followed by positions 0, 1, and 3. The dispatch permutation is

$$
[2,0,1,3]
$$

Per-rank send counts are $[1,3]$.

If routing weights in original flat order are $[0.6,0.4,0.3,0.7]$, reordered weights become $[0.3,0.6,0.4,0.7]$. The corresponding reordered expert IDs are $[1,0,2,0]$. Every position still describes the same original route; only communication order changed.

## Why grouping is stable

Routes are sorted by destination rank, but routes going to the same rank retain their original relative order.

In the example, rank 1 receives original positions 0, 1, and 3 in that order. A stable sort guarantees this behavior.

Stability makes the layout deterministic and preserves token-route order inside each destination segment. An unstable grouping could still send the correct items but make output association harder to reason about and fail the contract.

The dispatch permutation contains original indices arranged in send order.

## Reorder expert IDs and weights together

The reordered expert IDs are

$$
E_{send}=E_{flat}[perm]
$$

Routing weights must use the identical permutation:

$$
W_{send}=W_{flat}[perm]
$$

Expert IDs and weights form one logical record. Sorting only IDs assigns weights to the wrong routes while leaving all tensor shapes valid.

The token representations themselves would follow the same route permutation in a complete dispatcher, but they are outside this function’s outputs.

## Per-rank send counts

Counts have one entry for every rank from zero through $world\_size-1$.

If destination ranks are $[1,1,0,1]$, counts are $[1,3]$. With world size four, the same destinations produce $[1,3,0,0]$.

Zero entries are required. Collective communication code uses a fixed world-sized count vector to determine segment boundaries, including peers receiving nothing.

Counts also describe contiguous slices in the reordered arrays. With counts $[1,3]$, rank 0 receives the first item and rank 1 receives the next three. A prefix sum of counts identifies segment offsets without inspecting every destination again.

The count sum must equal the number of routes:

$$
\sum_r count_r=Tk
$$

This is a strong structural check.

## The inverse permutation

After remote experts process the grouped items, their results arrive in dispatch order. The inverse permutation restores original flattened token-route order.

If $perm[j]$ gives the original index sent at position $j$, define

$$
inverse[perm[j]]=j
$$

Then for any flattened payload $X$,

$$
X[perm][inverse]=X
$$

The inverse is not simply the original permutation unless that particular permutation happens to be self-inverse.

Restoring route order is necessary before reshaping back to $(T,k)$ and aggregating expert contributions per token.

For the example permutation $[2,0,1,3]$, the inverse is $[1,2,0,3]$. Applying that inverse to values returned in dispatch order places the result for original position 0 first, original position 1 second, and so on. This second reordering is what reconnects remote expert outputs with their token and route slots.

## Relationship to earlier MoE routing

Earlier problems selected top-k experts and weights, then grouped token routes by expert for local computation.

Expert parallelism adds an ownership layer. Routes are first grouped by the rank that owns the expert so an all-to-all can move them. A destination rank may then group its received routes by local expert.

This function begins with already selected experts and weights. It must not rerun the router, renormalize weights, or change expert choices.

It also does not balance load across ranks. If routing sends every item to one owner, the count vector is intentionally skewed. Changing destinations for balance would send routes to ranks that do not own the selected experts.

## Preparing an all-to-all payload

The reordered arrays are arranged as rank-contiguous segments, which is the layout expected by a variable-size all-to-all exchange. Each peer learns how many items it should receive from the send-count vector.

Actual communication would include token representations and might exchange receive counts as well. Those tensors and network operations are outside this function, but the permutation and counts are the bookkeeping they depend on.

After expert computation, returned values follow the same dispatched record order. The inverse permutation restores canonical route order before any top-k weighted aggregation.

## Output shapes

There are $Tk$ flattened routes. Dispatch permutation, inverse permutation, reordered expert IDs, and reordered weights each have length $Tk$.

Per-rank send counts have length equal to world size.

Integer outputs identify indices, experts, or counts. Reordered weights preserve their floating-point values and remain aligned with their expert IDs.

## Complexity and memory

Let $R=Tk$ be the number of routes. Stable sorting by destination costs $O(R\log R)$ in the specified general approach.

Mapping destinations, counting, building the inverse, and applying the permutation are $O(R+world\_size)$.

Permutation and reordered payload arrays use $O(R)$ memory, while counts use $O(world\_size)$.

The function estimates no network time and sends no data. It only prepares a deterministic communication layout.

## Common mistakes to avoid

- Flattening column-first changes the canonical token-route order.
- Sorting by expert ID instead of owning rank fails to create rank-contiguous send segments.
- Using an unstable sort can reorder routes within one destination.
- Omitting zero-count ranks produces a count vector shorter than world size.
- Reordering expert IDs without weights breaks each route record.
- Constructing the inverse in the same direction as the dispatch permutation does not restore original order.
- Deduplicating repeated experts loses legitimate routes from different tokens or slots.
- Rerunning top-k routing changes inputs that this function must preserve.

The preparation is a reversible reordering. Flatten token routes in a fixed order, group them stably by expert owner for communication, and retain the inverse mapping that puts returned expert results back where they began.
