# Continuous Batching

Requests generating different numbers of tokens do not finish together. In a fixed batch, a short request can leave an unused slot while the system waits for longer requests to complete.

Continuous batching revisits the active set at every decode step. Completed requests leave, and waiting requests can take their slots at the next step.

This problem simulates that policy with discrete arrival steps, required token counts, and a maximum number of active sequences.

## The unit of progress

One simulation step represents one decode iteration. Every active request receives exactly one token of progress during that step.

If request $i$ needs $r_i$ tokens, its remaining count decreases by one each time it appears in an active schedule row.

A request cannot receive two tokens in one step, even when capacity is otherwise unused. The simulator models parallel progress across active sequences, not repeated progress for one sequence.

## Order within every step

Each step follows a precise order:

1. identify free active-sequence slots,
2. admit eligible waiting requests into those slots,
3. record the active schedule for the step,
4. give one token to every active request,
5. mark newly completed requests and free their slots.

Admission happens before progress, so a request arriving at step $s$ can be active in row $s$ when capacity is available.

Completion happens after that row’s work. A slot freed at step $s$ becomes available for new admission at step $s+1$.

## Eligibility and admission order

A request is eligible when its arrival step is at most the current step and it has never been admitted.

When more eligible requests are waiting than free slots, choose smaller arrival steps first. Equal arrivals are ordered by original request index.

Once admitted, a request stays active until it completes. There is no preemption or rotation among unfinished requests in this exercise.

The admitted state must be separate from active state. A completed request is no longer active, but it must not become eligible for admission again.

## A simple capacity example

Suppose three requests arrive at steps $[0,0,1]$, each needs two tokens, and capacity is two.

At step 0, requests 0 and 1 are admitted. The active row is

$$
[1,1,0]
$$

Both receive one token and each has one remaining.

At step 1, request 2 has arrived, but no slot is free. Requests 0 and 1 are active again, receive their second tokens, and complete at step 1.

At step 2, request 2 takes a free slot. It remains active at steps 2 and 3 and completes at step 3.

The schedule is therefore

$$
\begin{bmatrix}
1&1&0\\
1&1&0\\
0&0&1\\
0&0&1
\end{bmatrix}
$$

Completion steps are $[1,1,3]$.

## Completion-step meaning

The completion value is the step in which the request receives its final required token.

A request admitted with one required token completes in that same step. If it arrives at step 4 and runs immediately, its completion step is 4.

The value is not the following step when its slot becomes visible as free. Confusing these two moments introduces an off-by-one error.

Every completion value must correspond to the final row where that request is active.

## Immediate slot reuse

Immediate reuse means no artificial empty step is inserted after completion.

If a request finishes during step 5, another waiting request can be admitted at the beginning of step 6. It cannot join halfway through step 5 because the active row and one-token work for that step have already been determined.

This next-step boundary makes the schedule deterministic and mirrors iteration-level admission.

## Idle periods

The simulation begins at step zero and continues through steps where nothing is active.

If the first request arrives at step 3, rows 0, 1, and 2 contain all false values. At step 3, the request becomes eligible.

Skipping idle rows would shift all later completion indices and violate the meaning of arrival steps.

The simulation terminates because every request requires at least one token and admitted active requests make one unit of progress per step.

## Schedule shape

For $N$ requests and $T$ simulated steps, the active schedule has shape $(T,N)$.

Row $s$ identifies every request that receives one decode token at step $s$. Column $i$ traces the lifetime of request $i$.

The completion tensor has shape $(N)$ and stays in original request order.

The final schedule row includes at least one request receiving its last token. No trailing all-false row is added after all requests complete.

## Why continuous batching helps

Iteration-level admission can keep capacity occupied when request lengths vary. A short request releases its slot without waiting for longer sequences that entered around the same time.

The benefit depends on workload and execution details, but the simulator captures the core scheduling behavior: membership changes at token boundaries.

It does not model prompt processing, KV-cache memory, per-step duration, or different token execution costs. Every active sequence consumes one logical slot and one unit of work per step.

Capacity limits simultaneous sequences, not total tokens. A request needing many tokens occupies one slot across many rows, while a one-token request occupies one slot for only its completion row.

## Compared with dynamic request batching

The preceding dynamic batcher chooses request groups before execution based on size and queue delay.

Continuous batching manages an already decoding population. Its capacity is a number of active sequences, and its decision point is every token step.

A real server may use both ideas in different parts of its pipeline. This exercise keeps only the continuous decode simulation.

## Complexity and memory

Let $T$ be the number of simulated steps and $N$ the number of requests. A direct implementation checks request states and records a row each step, costing $O(TN)$ time.

The explicit schedule uses $O(TN)$ memory. Current remaining counts, admission flags, active flags, and completion times use $O(N)$.

Production schedulers need not retain the complete history, but this function returns it for verification.

## Common mistakes to avoid

- Giving progress before admission delays newly arrived requests by one step.
- Admitting by original index without considering arrival step violates first-arrived priority.
- Re-admitting a completed request confuses inactive with never admitted.
- Freeing a slot before recording the step removes the request from the row where it earns its final token.
- Waiting an extra empty step before reuse contradicts immediate next-step admission.
- Skipping idle steps changes the coordinate system used by arrivals and completion values.
- Allowing one active request multiple tokens per step changes the simulated scheduling policy.

The schedule is a sequence of snapshots. At each token boundary, fill open slots with the oldest eligible requests, let every active request advance once, and release those that have just completed.
