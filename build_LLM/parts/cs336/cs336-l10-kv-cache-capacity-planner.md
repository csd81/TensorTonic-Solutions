# Plan KV-Cache Capacity

When a model serves many users at once, every active request keeps its own key-value cache in GPU memory, and those caches quickly outgrow the model itself. How many requests fit decides how much text the GPU can generate per second. The heart of this problem is turning a cache-size formula into a batch limit and then into a speed estimate from a simple memory-bandwidth model.

## What occupies GPU memory during serving

Serving needs three things in memory. The model weights take $M_{\text{model}} = P b_p$ bytes for $P$ parameters at $b_p$ bytes each. A reserved amount $R$ covers the runtime, temporary buffers and fragmentation. Everything left over is available for KV caches.

Each request stores a key and a value for every layer, every cached token and every key-value head:

$$
M_{\text{request}} = 2 L S H_{kv} D b_e
$$

Here $L$ is the number of layers, $S$ the cached sequence length, $H_{kv}$ the number of key-value heads, $D$ the head width and $b_e$ the bytes per stored element. The factor of 2 counts keys and values. The planner assumes every request holds a full $S$-token cache, the worst case, because a request's cache grows to that length as it generates.

The query-head count does not appear. With grouped-query attention, many query heads share each key-value head, and only the shared heads are stored. This is the main reason grouped-query attention exists.

## How many requests fit

With total GPU memory $M$, the batch is the number of whole request caches that fit in what remains:

$$
B = \max\left(0, \left\lfloor \frac{M - R - M_{\text{model}}}{M_{\text{request}}} \right\rfloor \right)
$$

Rounding down keeps only complete caches, since half a cache cannot serve a request. If the memory left after the model and reserve is smaller than one request's cache, or is negative, no request fits and the batch is zero.

## Decode speed from memory bandwidth

Generating one token for every request in the batch requires reading all the model weights once and every request's full cache. Each byte read takes part in very little arithmetic, so the step's time is set by memory bandwidth $W$, in bytes per second:

$$
T = \frac{M_{\text{model}} + B M_{\text{request}}}{W}
$$

One step produces $B$ tokens, one per request, so throughput is $B / T$ tokens per second, or zero when no request fits.

Batching is what makes serving efficient. The weights are read once per step whatever the batch size, so their cost is shared across all requests. The caches are read in full for every request, so their cost grows with the batch and eventually dominates.

## A realistic seven-billion-parameter example

Take a 7-billion-parameter model in bfloat16, so $M_{\text{model}} = 14 \times 10^9$ bytes, with 32 layers, 32 key-value heads of width 128, and caches of 4096 tokens stored in bfloat16. Each request needs

$$
2 \times 32 \times 4096 \times 32 \times 128 \times 2 = 2{,}147{,}483{,}648
$$

bytes, about $2.15 \times 10^9$, which is exactly 2 GiB because a GiB is $2^{30}$ bytes. On a GPU with $80 \times 10^9$ bytes of memory and $4 \times 10^9$ reserved, $62 \times 10^9$ bytes remain for caches, and $62 \times 10^9 / 2{,}147{,}483{,}648 \approx 28.9$ rounds down to 28 requests.

At a bandwidth of $2 \times 10^{12}$ bytes per second, a step reads $14 \times 10^9 + 28 \times 2{,}147{,}483{,}648 \approx 74.1 \times 10^9$ bytes and takes about 0.0371 seconds, for about 755 tokens per second.

Switching to grouped-query attention with 8 key-value heads cuts each cache to a quarter of the size, about $0.54 \times 10^9$ bytes. The remaining memory then holds about 115.5 caches, so 115 requests fit, about four times as many, and throughput rises to about 3037 tokens per second while the step time barely changes.

## What the model leaves out

Real servers allocate cache memory in pages as sequences grow, mix requests of different lengths, and spend some time on computation that this model ignores. The bandwidth model is still the standard first estimate, because decoding a large model is limited by how fast memory can be read.

## Common mistakes to avoid

- **Using query heads for cache size.** Only key-value heads are stored, so the query-head count does not enter the formula.
- **Forgetting the factor of 2.** Every layer stores both keys and values.
- **Rounding the batch up.** Only whole request caches fit, so the batch rounds down.
- **Allowing a negative batch.** When the model and reserve exceed memory, the batch is zero.
- **Reading the weights once per request.** The weights are read once per step and shared by the whole batch.
- **Mixing decimal and binary units.** A GiB is about 7 percent larger than $10^9$ bytes, which changes the request count.
