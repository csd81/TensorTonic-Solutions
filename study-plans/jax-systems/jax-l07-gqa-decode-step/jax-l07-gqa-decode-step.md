# Grouped-Query Decode Step

During generation, a transformer runs one decode step per new token, and in every layer that step compares the new token's query with all the keys stored in the KV cache. Grouped-query attention, usually shortened to GQA, lets several query heads share one cached key and value head, which shrinks the cache and the memory traffic that dominates decoding. This page starts from a single head attending over a short cache, adds the scaling and the masking that a fixed-size cache needs, and then shows how query heads are grouped onto shared cache heads without copying the cache. One full decode step of this kind is what you implement in this problem.

## One query against a cache

Start with one sequence and one head. The new token has a query vector $q$ of size D, the head size. The cache holds a key $K_j$ and a value $V_j$ for each earlier slot $j$. Attention scores each slot by how well its key matches the query, turns the scores into weights with a softmax, and returns the weighted average of the values.

$$
s_j = \frac{q \cdot K_j}{\sqrt{D}}, \qquad w_j = \frac{e^{s_j}}{\sum_m e^{s_m}}, \qquad o = \sum_j w_j V_j
$$

The output $o$ has the same size as one value vector. During decoding there is only one query per sequence, so the scores form a single row with one entry per slot, which is much smaller than the square score matrix of training.

## Why the scores are divided by the square root of D

A dot product of two vectors with D entries adds up D products. If the entries are roughly independent with variance 1, the sum has variance D, so scores grow with the head size. Large scores push the softmax towards putting almost all the weight on one slot, and the gradients and the mixing both become poor. Dividing by $\sqrt{D}$ brings the variance back to about 1 whatever the head size. With $D = 2$ the scale factor is $1/\sqrt{2} \approx 0.7071$.

## Stale slots and the length

An inference engine keeps the cache at a fixed number of slots S, so that the compiled decode step has the same shapes for every token. Only the first few slots of each sequence hold real keys and values, and a length per sequence says how many. The slots from the length onward contain whatever was there before, such as entries from an earlier, longer request. They are ordinary finite numbers, so nothing in the array itself marks them as invalid.

If those stale slots take part in the softmax, they steal weight from the real tokens. Here is an example with $D = 2$, the query $q = (1, 1)$, and three slots whose keys are $(1, 0)$, $(0, 2)$ and $(3, 3)$, and whose values are $(1, 0)$, $(0, 1)$ and $(10, 10)$. Only the first two slots are valid, so the length is 2.

- **With the mask.** The valid scores are $1/\sqrt{2} \approx 0.7071$ and $2/\sqrt{2} \approx 1.4142$. The softmax gives weights of about $0.3302$ and $0.6698$, so the output is about $(0.3302, 0.6698)$.
- **Without the mask.** The stale key scores $6/\sqrt{2} \approx 4.2426$ and takes about $0.9189$ of the weight. The output jumps to about $(9.2159, 9.2434)$, which is mostly the stale value.

Masking sets the score of every invalid slot to negative infinity before the softmax. Its exponential is 0, so it gets exactly zero weight, and the valid weights renormalize among themselves. Every sequence keeps at least one valid slot, the new token's own entry, so the softmax always has something to divide by.

## Masks with traced lengths

In a batch, each sequence has its own length, so lengths is an int32 array with one entry per sequence. Under jax.jit that array is traced, which means it is replaced by a stand-in that knows only its shape and dtype while the function is compiled. Slicing the cache as keys[:length] would produce an array whose size depends on the data, and JAX cannot compile that.

The fixed-shape alternative is to compare slot numbers with lengths. The slot numbers $0, 1, \dots, S-1$ form a row, the lengths form a column, and comparing them with broadcasting gives a (B, S) array of booleans that is true exactly for the valid slots of each sequence. With $S = 4$ and lengths $(1, 3)$, the mask rows are $(\text{T}, \text{F}, \text{F}, \text{F})$ and $(\text{T}, \text{T}, \text{T}, \text{F})$. jnp.where then picks the real score or negative infinity element by element, and the shapes never change.

## Sharing cache heads across query heads

Multi-head attention gives every query head its own key and value head. During decoding the whole cache is read from memory at every step, so the cache size sets the speed. GQA keeps Hq query heads but stores only Hkv key and value heads, where Hkv divides Hq. Each cache head serves a group of $G = Hq / Hkv$ consecutive query heads.

$$
g(h) = \left\lfloor \frac{h}{G} \right\rfloor
$$

Here $h$ is the query head index and $g(h)$ is the cache head it reads. With $Hq = 4$ and $Hkv = 2$ the group size is 2, so query heads 0 and 1 read cache head 0, and query heads 2 and 3 read cache head 1. Two special cases fall out of the same rule. With $Hkv = Hq$ every group has one member and GQA is ordinary multi-head attention. With $Hkv = 1$ every query head shares one cache head, which is called multi-query attention.

The saving is direct. A model with 32 query heads and 8 cache heads stores a cache 4 times smaller than the same model with 32 cache heads, and each decode step reads 4 times less cache data.

## Grouping without copying the cache

One way to line up query heads with cache heads is to repeat each cache head G times along the head axis, so that the cache has Hq heads and ordinary multi-head attention applies. That gives the right numbers, but it builds a copy of the cache that is G times larger, which throws away the memory saving GQA was designed for.

The cheaper way reshapes the query instead. The query of shape (B, Hq, D) can be viewed as (B, Hkv, G, D), because consecutive query heads belong to the same group. Head $h$ lands at cache head $\lfloor h / G \rfloor$ and group member $h \bmod G$, which is exactly the mapping above. Now each cache head appears once, and the G members of its group are an extra axis that broadcasts against it. jnp.einsum, which describes a contraction with a short subscript string, can multiply over the D axis while keeping the batch, cache head, group member and slot axes. Reshaping the result back to (B, Hq, D) restores the original head order.

## Precision of the arithmetic

Caches are often stored in bfloat16, a 16-bit format that keeps only two to three significant decimal digits, to halve their memory. The stored numbers are what they are, but the arithmetic on them should not lose more. Converting the cache to float32 before computing scores, the softmax and the weighted sum keeps each step accurate, while summing a long row of products in bfloat16 would add a rounding error at every addition. XLA usually fuses this conversion into the computation that reads the cache, so it costs little extra memory traffic.

## Where this step sits in decoding

A full decode step in one layer projects the new token into its query, key and value, applies rotary embeddings to the query and key, writes the key and value into the cache at the current position, and then runs this attention against the cache with the length increased by one. The output then goes through the output projection and the rest of the layer. This problem isolates the attention part, with the cache already updated and the lengths already counting the new token.

## Common mistakes to avoid

- **Mapping heads with a modulo.** Query head h reads cache head h // G, and using h % Hkv sends neighbouring query heads to different cache heads.
- **Masking one slot too many or too few.** The valid slots are those with index less than the length, so a test of less than or equal lets one stale slot in.
- **Skipping the mask.** Stale slots hold finite numbers, so without a mask they receive real weight and can dominate the output.
- **Leaving out the square root of D.** Unscaled scores make the softmax too sharp, and the outputs move noticeably even for small heads.
- **Slicing the cache by a traced length.** Under jax.jit a length has no concrete value, so the cache has to stay at its full size and be masked instead.
- **Doing the arithmetic in bfloat16.** Scores, softmax and the weighted sum should run in float32 after converting the cache.
