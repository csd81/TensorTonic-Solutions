# Dynamic Request Batching

Processing several requests together can use hardware more efficiently than launching each one separately. Waiting for a batch to fill, however, adds queueing latency. A dynamic batcher balances those concerns by dispatching on either size or time.

This problem simulates a forming batch. It closes immediately when it reaches the maximum size, or at a fixed deadline measured from the first request in that batch.

## The two dispatch triggers

Let the maximum batch size be $B_{max}$ and the maximum queue delay be $D$.

When the first request enters an empty batch at time $a_f$, the deadline becomes

$$
d=a_f+D
$$

The deadline never moves when later requests join. Moving it would let the first request wait longer than the configured maximum.

The batch dispatches at the earlier of:

- the arrival time that fills position $B_{max}$,
- the fixed deadline $d$.

Every request in the batch receives the same batch ID and dispatch time.

## Requests are considered chronologically

Input arrival times are not required to be sorted. Scheduling first orders request indices by arrival time.

Equal arrival times are ordered by original request index. This stable tie rule makes the output deterministic and preserves the caller’s order among simultaneous arrivals.

The returned assignment and start-time tensors still correspond to original indices. Chronological processing order is an internal view, not a new output ordering.

## Dispatch by size

Suppose $B_{max}=3$, the delay is 5, and requests arrive at times 0, 1, and 2.

The first request sets a deadline at 5. The second joins at time 1. The third joins at time 2 and fills the batch, so all three dispatch at time 2.

The size trigger avoids waiting until the deadline after enough work is already available.

The next arriving request begins a new batch and establishes a new deadline from its own arrival.

When several requests share the filling arrival time, stable ordering decides which request completes the current batch and which requests begin the next one. The original index rule makes this boundary reproducible.

## Dispatch by deadline

Now suppose only two requests arrive at times 0 and 1 with the same size limit and delay.

The batch never reaches size three, so it dispatches at time 5, which is the first request’s arrival plus the maximum delay.

The second request waits four units, while the first waits five. The limit is defined from the oldest request in the forming batch.

The final nonempty batch also dispatches at its deadline. End of input does not mean it should dispatch immediately, because the simulation models the allowed waiting period.

Batch start time is therefore a dispatch time rather than the arrival time of the first member. Every request in a timeout-triggered batch may have a different wait, but they all begin model execution together at the deadline.

## The exact-deadline boundary

A request arriving exactly at the current deadline does not join the expiring batch.

At time $d$, expiration is processed first. The old batch dispatches at $d$, then the new request starts the next batch and receives deadline $d+D$.

This event ordering must be explicit because another scheduler could reasonably allow the equal-time arrival to join. The stored contract chooses expiration first.

The operational test is therefore $arrival\geq deadline$ before adding the current request.

## A mixed example

Let maximum size be two, delay be three, and arrivals in original order be $[4,0,1,4]$.

Chronological order is request 1 at time 0, request 2 at time 1, then requests 0 and 3 at time 4 in index order.

Requests 1 and 2 fill batch 0 at time 1. Requests 0 and 3 then fill batch 1 at time 4.

Returned assignments are placed back in original order. Requests 0 and 3 receive batch 1 and time 4, while requests 1 and 2 receive batch 0 and time 1.

This example shows why processing order and output order must be handled separately.

## Maximum delay of zero

Zero queue delay is valid. Each request dispatches at its arrival time in a separate batch, including requests sharing an arrival time, because deadline expiration takes priority over the next equal-time arrival.

## What the scheduler returns

For $N$ input requests, batch assignment has shape $(N)$ and contains integer batch IDs beginning at zero.

Batch start time also has shape $(N)$. All requests assigned to the same batch have the same start time.

Batch IDs reflect dispatch sequence. They do not need to match request indices, and a request that appears first in the input may belong to a later chronological batch.

## Latency and throughput tradeoff

A larger maximum batch size can improve computational efficiency when enough requests arrive quickly. A longer delay gives the queue more opportunity to form such batches.

Both settings can increase how long early requests wait. The delay provides a hard queueing boundary in this simulation.

This function does not execute the batches or estimate model latency. It only determines membership and dispatch time from arrivals and the two batching limits.

Queue delay and execution latency should not be confused. The returned start time tells when queued work leaves the batcher; any time spent running the model would occur afterward and is not included in this simulation.

## Difference from continuous batching

Dynamic batching here forms a request-level group before dispatch. Once a request is assigned, the simulation does not replace completed members during execution.

Continuous batching, covered by the next problem, revisits membership at every decode step and can reuse a freed slot immediately.

Keeping those models separate avoids adding token-level behavior to a request-arrival scheduler.

## Complexity and memory

Sorting $N$ requests costs $O(N\log N)$. The scheduling pass adds and dispatches each request once, costing $O(N)$.

Chronological indices, the forming batch, and outputs use $O(N)$ memory.

No time-step simulation is needed because only arrivals, fill events, and fixed deadlines can change the forming batch.

## Common mistakes to avoid

- Processing the input tensor as already sorted gives incorrect batches for unsorted arrivals.
- Breaking equal-time ties arbitrarily makes assignments nondeterministic.
- Resetting the deadline whenever a request joins can exceed the oldest request’s delay limit.
- Checking expiration after adding an equal-time arrival violates the deadline boundary rule.
- Dispatching a full batch at its deadline instead of the filling arrival adds unnecessary wait.
- Dispatching the final partial batch immediately at end of input ignores its fixed deadline.
- Returning results in chronological order detaches them from original request indices.

The scheduler can be understood as a repeated timer: the first queued request starts a fixed clock, later requests may fill the batch before it rings, and whichever event happens first closes that batch.
