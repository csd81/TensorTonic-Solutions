# Selecting a Configuration Under a Latency SLO

Inference deployments often have several candidate configurations. More GPUs or a larger batch may increase throughput, but can change latency and hourly cost.

The cheapest row in a benchmark table is not useful if it cannot carry the required load or violates the latency objective. This problem first filters by hard service constraints, then chooses the lowest-cost feasible option with deterministic tie rules.

## Reading the candidate table

Each candidate row contains five values in this order:

1. GPU count,
2. batch size,
3. throughput,
4. P95 latency,
5. hourly cost.

The original row index identifies the configuration returned to the caller.

Column order is part of the contract. Swapping throughput and latency can produce a plausible calculation that selects the wrong deployment.

## Two hard feasibility constraints

Candidate $i$ is feasible when

$$
q_i\geq q_{required}
$$

and

$$
l_i\leq l_{max}
$$

where $q_i$ is throughput and $l_i$ is P95 latency.

Both conditions must hold. Excess throughput does not compensate for a latency violation, and excellent latency does not compensate for insufficient capacity.

Feasibility is checked before cost. An infeasible candidate cannot win even if it is free.

## Why P95 is a service objective

P95 latency summarizes tail behavior rather than only the typical request. A maximum P95 requirement says the deployment should keep that measured percentile at or below a threshold.

This selector trusts each candidate’s supplied measurement. It does not calculate percentiles, predict performance, or account for uncertainty in the benchmark.

The required throughput may represent requests per second or tokens per second, but every candidate and the requirement must use the same unit.

## Exact boundaries are feasible

The comparisons are non-strict. A candidate with throughput exactly equal to the required rate passes the capacity constraint.

A candidate with P95 exactly equal to the maximum allowed latency also passes.

Using greater-than for throughput or less-than for latency would incorrectly reject configurations that meet the stated SLO exactly.

Floating-point comparisons follow the supplied values directly. The problem defines no tolerance around the boundary.

## Cost is the primary objective

After filtering, choose the feasible candidate with the smallest hourly cost.

The selector does not maximize throughput, minimize latency, or calculate a blended score. Those metrics are constraints once their thresholds are met; cost decides among survivors.

For example, a configuration delivering 500 requests per second is not preferred over one delivering 200 when the requirement is 150 and the 200-rate option is cheaper, assuming both satisfy latency.

This distinction prevents overprovisioning from silently becoming the objective.

## A simple example

Suppose the requirement is throughput at least 100 and P95 latency at most 80.

Consider three rows:

- candidate 0 has throughput 90, latency 50, and cost 1,
- candidate 1 has throughput 120, latency 90, and cost 2,
- candidate 2 has throughput 110, latency 70, and cost 3.

Candidate 0 fails throughput. Candidate 1 fails latency. Candidate 2 is the only feasible option, so it is selected despite having the highest listed cost.

Cost comparison begins only after those failures are removed.

## Deterministic tie-breaking

Several feasible candidates may have the same hourly cost. The required priority is:

1. lower cost,
2. fewer GPUs,
3. smaller batch size,
4. lower original row index.

This can be represented by the ordered key

$$
(cost_i,gpu_i,batch_i,i)
$$

Lexicographic comparison examines later fields only when every earlier field ties.

Throughput and latency do not appear in the tie key because they were already used as feasibility constraints and are not secondary objectives in this contract.

## Why original index is last

Two rows may be identical across cost, GPU count, and batch size. Choosing the lower original index makes the result stable and traceable.

Sorting the candidate matrix before selection can lose that original identity unless indices are carried along. A single pass naturally preserves it.

Updating the best candidate only when the full key is smaller also leaves an exactly tied earlier row in place.

## No feasible configuration

It is possible that every candidate violates capacity, latency, or both.

The required result is the sentinel pair

$$
(-1,-1.0)
$$

The negative index indicates that no row was selected, and the negative cost indicates there is no valid hourly cost to report.

The function should not fall back to the least-bad infeasible option because that would silently violate a hard service requirement.

## Output contract

Selected index is a scalar INT64 tensor. Hourly cost is a scalar float64 tensor.

For a feasible result, the cost must come from column four of the selected original row. Recomputing cost from GPU count is outside the contract.

Scalar tensors have no candidate dimension. Returning a length-one vector changes the documented shape even though it contains one value.

## Capacity planning interpretation

The selector assumes candidate measurements already reflect their GPU count, batch size, model, hardware, and workload.

It does not interpolate between candidates or create a new configuration. It chooses one supplied row exactly.

Hourly cost is minimized for the stated required rate and latency SLO. Changes to either threshold can change the feasible set and therefore the selected row.

The result is only as reliable as the candidate data and whether benchmark conditions match production traffic.

## Complexity and memory

For $C$ candidates, checking constraints and comparing keys takes $O(C)$ time.

Only the current best key and index are needed, so auxiliary memory is $O(1)$.

The candidate matrix already exists as input, and the function creates no structure proportional to deployment size beyond reading its rows.

## Common mistakes to avoid

- Selecting the cheapest row before feasibility checks can violate capacity or latency.
- Using OR between the two constraints accepts candidates that fail one hard requirement.
- Using strict inequalities rejects exact-boundary configurations.
- Maximizing throughput among feasible rows changes the cost-minimization objective.
- Using throughput or latency as an undocumented tie-break changes deterministic results.
- Preferring larger batches before GPU count reverses the required tie order.
- Returning a row from sorted order instead of its original index breaks traceability.
- Choosing an infeasible fallback instead of the sentinel hides the absence of a valid deployment.

The decision process is deliberately simple: remove every row that breaks a hard service constraint, then compare the remaining rows by cost, GPU count, batch size, and original position in that exact order.

Hard constraints always take precedence over an attractive price.
