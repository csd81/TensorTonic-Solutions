# Disaggregated Prefill and Decode Serving

Prefill and decode place different demands on inference hardware. Prefill processes many prompt tokens together, while decode repeatedly produces one new token. Disaggregated serving assigns these phases to separate replica pools.

The separation allows each pool to queue and scale independently, but it introduces a handoff: the KV cache created by prefill must reach a decode replica before generation can begin.

This problem simulates replica assignment, transfer delay, first-token time, and completion time.

## The request path

Every request follows four stages in order:

1. wait for a prefill replica,
2. process its prompt,
3. transfer its KV cache,
4. wait for and run on a decode replica.

No stage can begin before the previous one completes. Prefill and decode use independent availability arrays because they occupy different pools.

The simulation uses token rates rather than modeling individual model operations.

## Prefill scheduling order

Requests enter prefill scheduling by ascending arrival time. Equal arrival times use original request index as the tie-break.

For each request, choose the prefill replica with the earliest current availability time. Equal availability chooses the lowest replica index.

This policy chooses earliest availability, not highest processing rate or earliest predicted completion. Replica rates affect duration only after assignment.

The deterministic rule makes assignments reproducible even when several replicas begin idle at time zero.

## Prefill start and end

For request $i$ assigned to prefill replica $p$, start time is

$$
s^{pre}_i=\max(a_i,A^{pre}_p)
$$

where $a_i$ is arrival and $A^{pre}_p$ is that replica’s next-available time.

With prompt length $P_i$ and replica rate $r^{pre}_p$ tokens per second, duration is

$$
d^{pre}_i=\frac{P_i}{r^{pre}_p}
$$

and end time is

$$
e^{pre}_i=s^{pre}_i+d^{pre}_i
$$

The selected replica’s availability becomes this end time.

## KV-cache transfer

Decode cannot use the request until its KV cache has moved from the prefill pool.

The transfer model contains a variable term and a fixed term:

$$
d^{transfer}_i=\frac{P_i}{BW}+L_{fixed}
$$

Here bandwidth is expressed in prompt-token equivalents per second, following the problem’s simplified units. A longer prompt creates a larger cache and therefore a longer variable transfer.

The request becomes decode-ready at

$$
k_i=e^{pre}_i+d^{transfer}_i
$$

It cannot enter the decode queue before $k_i$, even if a decode replica is idle.

## Decode order uses readiness

Requests are scheduled for decode in ascending $k_i$, with original index breaking equal ready times.

This order can differ from arrival order. A later request with a short prompt may finish prefill and transfer before an earlier request with a long prompt.

Using original arrival order for decode would make an already-ready request wait behind another whose KV cache has not arrived. The handoff time is the causal eligibility boundary.

## Decode assignment and duration

For a decode-ready request, choose the replica with earliest availability, again breaking ties by lowest index.

Decode starts at

$$
s^{dec}_i=\max(k_i,A^{dec}_d)
$$

If the chosen rate is $r^{dec}_d$, the first token appears after one token duration:

$$
f_i=s^{dec}_i+\frac{1}{r^{dec}_d}
$$

For output length $O_i$, completion is

$$
c_i=s^{dec}_i+\frac{O_i}{r^{dec}_d}
$$

The replica remains occupied until $c_i$.

## Understanding the TTFT field

The stored simulator reports $f_i$, the first-token time on the shared simulation clock, in its TTFT output field. This value includes arrival position on the clock, prefill queueing and service, transfer, decode queueing, and one decode-token duration.

If a consumer wants elapsed TTFT from request arrival, it can calculate $f_i-a_i$. The reference contract and tests use the reported first-token time directly.

The important distinction is that KV-ready time is not first-token time. A request may wait for a decode replica after transfer, and generating the first token also takes time.

## A one-request example

Suppose a request arrives at time 0 with 100 prompt tokens and 4 output tokens. Prefill rate is 50 tokens per second, transfer bandwidth is 100 tokens per second with fixed latency 0.1, and decode rate is 10 tokens per second.

Prefill takes 2 seconds. Transfer takes $100/100+0.1=1.1$ seconds, so the request is decode-ready at 3.1.

With an idle decode replica, decode begins at 3.1. The first token time is 3.2, and completion is

$$
3.1+\frac{4}{10}=3.5
$$

This timeline shows why transfer must be included between the two phases.

## Independent bottlenecks

A slow or undersized prefill pool creates a prefill queue. A slow decode pool creates a decode queue even when KV transfers arrive quickly.

Low transfer bandwidth or high fixed latency delays every decode-ready time and may leave decode replicas idle while state is in transit.

The separate availability arrays let the simulation expose these bottlenecks through timings. It does not automatically rebalance replicas between pools.

Disaggregation can reduce interference between phases, but it also duplicates deployment concerns and requires cache movement. This exercise models the handoff cost without judging whether separation is beneficial for a particular system.

## Output shapes

For $N$ requests, assigned replicas have shape $(N,2)$. Column zero contains the prefill replica index and column one the decode replica index.

First-token and completion times each have shape $(N)$ and remain in original request order, even though internal scheduling uses arrival and ready-time orderings.

Replica indices are integers. Timing outputs use floating point because rates and transfer latency can produce fractional seconds.

## Complexity and memory

Sorting requests by arrival and KV-ready time costs $O(N\log N)$.

Scanning $P$ prefill replicas and $D$ decode replicas for each request costs $O(N(P+D))$ with the direct minimum search used by this problem.

Per-request and availability state uses $O(N+P+D)$ memory. The simulator does not allocate KV-cache tensors or execute model work.

## Common mistakes to avoid

- Using one shared availability pool allows prefill work to block decode replicas directly.
- Choosing the fastest-rate replica instead of the earliest-available replica changes the assignment policy.
- Starting transfer before prefill ends violates phase causality.
- Scheduling decode by original arrival ignores when each KV cache actually becomes ready.
- Omitting fixed transfer latency underestimates short transfers.
- Treating KV-ready time as first-token time ignores decode queueing and one-token service.
- Computing completion with one token instead of the complete output length releases decode capacity too early.
- Returning results in scheduling order rather than original request order breaks row alignment.

The simulation is a two-queue pipeline joined by a state transfer. Prefill completion creates a cache, transfer makes that cache eligible for decode, and independent replica availability determines the waiting time on each side.
