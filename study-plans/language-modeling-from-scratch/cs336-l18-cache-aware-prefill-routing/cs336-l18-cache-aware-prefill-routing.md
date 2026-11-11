# Route Cache-Aware Prefill Requests

A large inference deployment runs many copies of a model, called workers, and a router decides which worker handles each request. Sending a request to a worker that already holds part of its KV cache saves prefill work, but always chasing cache hits can overload a few workers. The heart of this problem is ranking the workers that can fit each request by cache reuse first and load second, and keeping each worker's load up to date as requests are assigned.

## Why routing should know about caches

Each worker keeps a prefix cache of recent prompts, and the router learns each worker's reusable prefix length for a request by matching it against that worker's prefix trie. If a worker has cached a long prefix of the request, it only needs to run prefill on the uncached remainder. Routing the same request to a different worker would recompute the whole prompt.

Load matters too. A worker that receives every request with a popular system prompt gets far more work than the others, and its queue grows while the rest sit idle. A good router balances the two, preferring cache hits but spreading load when hits are equal.

## Two kinds of load

Each worker tracks two numbers. Assigned tokens count the full length of every request it has taken, and in this problem they are checked against the worker's capacity using the full request length. That is a conservative rule, since in a real prefix cache the cached part is already resident and shared. Uncached load counts only the prefill work a worker has been given, the part of each request that was not already cached, which is what actually costs compute.

A worker is feasible for a request only if its assigned tokens plus the request's full length stay within its capacity.

## Ranking feasible workers

Among the feasible workers, the router uses a strict priority order. The longest cached prefix comes first, since it saves the most prefill. Among equal prefixes, the lowest projected uncached load wins, which is the same as preferring the worker with the least prefill work already queued, because the request's uncached part is identical on every tied worker. After that, the worker left with the most remaining capacity wins, keeping room for future requests, and a fixed rule settles any final tie.

A strict order never trades away a cache hit for better balance, which keeps the router simple and predictable. In this scheme capacity is the only brake on cache-chasing, and production routers often add a load threshold as well, sending a request elsewhere when its best cache match is too busy.

The chosen worker's assigned tokens grow by the full request length and its uncached load grows by the request's uncached part. A request that no worker can fit never reaches a worker, so no load is charged for it.

## Three requests and three workers

Take workers 0, 1 and 2 with capacities 16, 30 and 30, and three requests of 10, 10 and 6 tokens.

The first request has a cached prefix of 6 tokens on worker 0 and none elsewhere. All workers are feasible, and worker 0 has the longest prefix, so it gets the request. Its assigned tokens become 10 and its uncached load becomes $10 - 6 = 4$.

The second request has prefixes of 6 on worker 0 and 2 on worker 2. Worker 0 would reach $10 + 10 = 20$ assigned tokens, over its capacity of 16, so it is infeasible despite the better prefix. Worker 2's prefix of 2 beats worker 1's 0, so worker 2 takes it, with uncached load $10 - 2 = 8$.

The third request has no cached prefix anywhere. All three workers are feasible, since worker 0 would reach exactly 16. The projected uncached loads are $4 + 6 = 10$, $0 + 6 = 6$ and $8 + 6 = 14$, so worker 1, the least loaded, takes it. The assignments are workers 0, 2 and 1, and the final uncached loads are 4, 6 and 8. The second request shows why the state must update after every assignment, since worker 0 only became infeasible because of the first request.

## Where this sits in a serving stack

Cache-aware routing works alongside the prefix cache on each worker and the continuous-batching scheduler inside it. The router decides which worker, the prefix cache decides how much prefill can be skipped, and the scheduler decides when the request runs. Systems such as SGLang use this kind of routing to raise cache hit rates across many workers.

## Common mistakes to avoid

- **Checking capacity with the uncached part only.** Feasibility uses the full request length in this problem.
- **Adding the full request to uncached load.** Only the uncached remainder adds prefill work.
- **Picking an infeasible worker for its cache hit.** Capacity rules out a worker before any ranking.
- **Ranking every request against the initial worker state.** Each assignment changes load and capacity, which is why worker 0 is infeasible for the second request in the example.
- **Comparing projected loads without adding the request.** The projection includes the request's uncached part, which matters once prefixes differ.
