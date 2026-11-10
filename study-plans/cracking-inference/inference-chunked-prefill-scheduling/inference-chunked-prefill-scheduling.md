# Chunked Prefill Scheduling

A long prompt can require substantial prefill work before its first output token. If the scheduler processes that prompt as one indivisible job, requests already streaming tokens may wait through the entire prefill.

Chunked prefill splits prompt work into bounded pieces. Each iteration has a shared token budget, spends that budget on ready decode work first, then uses any remainder for prompt chunks.

This problem simulates that exact decode-priority policy and returns an event history.

## Two kinds of work

Every request begins in prefill. It must process all prompt tokens before it becomes eligible for decode.

During prefill, one event may process several prompt tokens, limited by the chunk size and remaining iteration budget. During decode, one event processes exactly one token.

The phase marker in the output distinguishes these events: zero means prefill and one means decode.

A request that finishes prefill during iteration $s$ may begin decoding no earlier than iteration $s+1$.

## The shared iteration budget

Let $M$ be the maximum tokens per iteration. At the start of each iteration, remaining budget equals $M$.

Every scheduled decode token subtracts one. Every prefill event subtracts its chunk size. The sum of event token counts for an iteration must never exceed $M$.

Unused budget is allowed. For example, the system may be idle before the next request arrives, or remaining budget may be too small only after all eligible work is exhausted.

The budget is shared across requests and phases rather than reset for each request.

## Decode has strict priority

At an iteration, all admitted requests already in decode are considered before any prefill request.

Each ready decode request receives at most one token, processed in original request-index order in this stored policy. If decode work consumes the complete budget, no prefill event occurs that iteration.

This priority protects the cadence of requests whose output is already streaming. It may delay TTFT for requests still waiting on prompt work, which is the intended policy tradeoff.

Priority must be implemented structurally by finishing the decode scheduling pass before beginning prefill scheduling.

## Bounded prefill chunks

After decode, prefill requests use the remaining budget in first-come-first-served order. Arrival step is the primary key and original request index breaks ties.

For request $i$, a prefill event processes

$$
c_i=\min(B,C,P_i)
$$

where $B$ is remaining iteration budget, $C$ is the configured chunk limit, and $P_i$ is that request’s remaining prompt tokens.

This minimum enforces all three boundaries. A final chunk may be smaller than $C$, and a budget-limited chunk may be smaller even when the request has much more work left.

## A single-request example

Suppose a request arrives at iteration 0 with five prompt tokens and two decode tokens. The iteration budget is three and chunk size is two.

Iteration 0 has no decode work, so prefill processes two tokens. Three prompt tokens remain.

Iteration 1 processes another two-token prefill chunk. One prompt token remains.

Iteration 2 processes the final one-token chunk. The request enters decode state but cannot decode in the same iteration.

Iterations 3 and 4 each process one decode token, completing the request.

The unused budget in several iterations is valid because the chunk cap prevents a single request from consuming more prompt tokens in one iteration.

## Interleaving two requests

Suppose request A has finished prefill and still needs decode tokens, while request B is waiting on a long prompt. With budget four, A receives one decode token first.

The remaining budget of three can then be used for B’s prefill, subject to the chunk limit. Both phases appear in the same iteration, but decode appears first in scheduling priority.

On the next iteration, A again receives its decode token before B receives another prompt chunk.

This interleaving is the purpose of chunking. B makes bounded progress without forcing A to pause for B’s complete prompt.

## Arrival and admission

A request becomes admitted when its arrival step is at most the current iteration. It cannot receive prefill or decode work before that point.

Several requests can arrive together. Their prefill order uses original index as the tie-break.

An admitted request remains in the prefill candidate set until its prompt count reaches zero. It then moves permanently into decode state and eventually into done state.

Separate state flags prevent completed requests from reappearing and prevent newly prefilled requests from decoding too early.

## The event table

Each returned row contains four integers:

1. iteration,
2. request ID,
3. phase, with zero for prefill and one for decode,
4. tokens processed.

Decode rows always have token count one. Prefill rows have a positive count no larger than the chunk limit, remaining prompt, or available budget.

The table has one row per scheduled event, not one row per iteration. An iteration can produce several rows when multiple requests receive work.

Idle iterations produce no event row, but the internal iteration counter must still advance so later arrivals retain their correct time.

## Policy boundaries

This simulator does not choose chunks by shortest prompt, remaining work, or model priority. It uses arrival time and index for prefill candidates.

It does not batch a newly completed prefill into decode during the same iteration. The phase transition takes effect at the next boundary.

It also does not model how long one token of prefill or decode takes on hardware. The shared token budget is the complete resource model for the exercise.

## Complexity and memory

For $N$ requests and $T$ iterations, a direct simulator examines request state across iterations, giving $O(TN)$ time.

Current state uses $O(N)$ memory. The returned history uses $O(E)$ rows for $E$ scheduled events.

Long prompts create at most approximately their token count divided by chunk size prefill events, with additional splits possible when remaining iteration budget is smaller than the chunk limit.

## Common mistakes to avoid

- Scheduling prefill before ready decode work violates the explicit priority policy.
- Resetting the token budget for each request allows an iteration to exceed its global limit.
- Processing a complete long prompt in one event ignores the chunk cap.
- Letting a request decode in the iteration where prefill finishes breaks the phase boundary.
- Sorting prefills only by index ignores earlier arrival steps.
- Giving more than one decode token to a request in one iteration changes the decode policy.
- Recording one table row per iteration loses separate request and phase events.
- Failing to advance through idle iterations prevents future requests from arriving.

The scheduler protects streaming work while allowing prompts to advance in controlled pieces. Every iteration pays for decode first, then spends the remaining budget on the oldest eligible prefill chunks.
