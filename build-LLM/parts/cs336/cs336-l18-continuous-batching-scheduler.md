# Simulate Continuous Decode Batching

An inference server handles many requests of different lengths at once. If it waited for every request in a batch to finish before starting new ones, short requests would leave GPU slots empty while long ones kept running. Continuous batching, introduced in Orca (Yu et al., 2022) and used by servers such as vLLM, admits and retires requests at every decode step. The heart of this problem is simulating that scheduler step by step under a fixed KV-cache budget.

## Why static batches waste the GPU

With static batching, a batch of requests starts together and ends when its longest request finishes. A request that needs 10 tokens sits idle in the batch while another generates 1,000, and new requests wait outside even though GPU memory and compute are free.

Continuous batching treats each decode step as a chance to rebalance. Finished requests leave immediately, and waiting requests join as soon as there is room, so the batch stays full and throughput rises sharply.

## Reserving cache memory

Each active request needs KV-cache memory for its prompt and every token it generates. In this simplified scheduler, to guarantee a request can always finish once started, it reserves the full footprint, prompt tokens plus maximum new tokens, at admission and keeps it until the request completes. A request is admitted only if its footprint fits in the capacity left by everything already reserved. Production servers such as vLLM instead allocate cache pages on demand and pause a request if memory runs out, which uses memory more fully at the cost of more complex bookkeeping.

If the next waiting request does not fit, the scheduler still considers later, smaller ones. This avoids head-of-line blocking, where memory sits unused behind one large request. The trade-off is that a large request can be delayed for a long time if small ones keep arriving.

The simulation treats a request's prefill as happening when it is admitted and counts only decode steps.

## Admission twice around each decode

Within a step, admission happens twice around the decode. Newly arrived requests get a first chance at admission. Then every active request generates one token, and those that reach their maximum length finish and release their reservations. The freed capacity is used straight away by a second chance at admission. Requests admitted at that point missed this step's decode, which has already run, so their first token comes on the next step.

Actual memory use is each active request's prompt plus the tokens it has generated so far, which is usually well below the reserved total. That gap is the price of full reservation.

## Three requests on a budget of six

Take a cache capacity of 6 tokens. Request A arrives at step 0 with 2 prompt tokens and 1 to generate, a footprint of 3. Request B also arrives at step 0 with 2 prompt tokens and 2 to generate, a footprint of 4. Request C arrives at step 1 with 1 prompt token and 1 to generate, a footprint of 2.

At step 0 the first pass admits A, using 3 of 6. B needs 4 and does not fit. A generates its only token and finishes, freeing its reservation, so the second pass admits B. The step records A and B as admitted, A as finished, B as active, and 2 tokens in use, B's prompt with nothing generated yet.

At step 1, C arrives, and the first pass admits it alongside B's reservation of 4, for a total of 6. Both decode. C finishes, and B has generated 1 of its 2 tokens, so 3 tokens are in use. At step 2, B generates its last token and finishes, leaving the cache empty.

B joined in the same step that A left, whereas static batching would have held it until a whole new batch could start.

## Common mistakes to avoid

- **Admitting by actual use instead of reservation.** Admission checks the full footprint so a started request can always finish.
- **Stopping at the first request that does not fit.** Later, smaller requests are still considered, which keeps memory from sitting idle.
- **Letting newly admitted requests decode in the same step.** Requests from the second pass start generating on the next step.
- **Forgetting to release a finished request's reservation before the second pass.** Waiting requests would stay blocked for an extra step.
- **Reporting reserved tokens as cache use.** The gap between reserved and actual use is exactly what the cache-use figure exposes.
