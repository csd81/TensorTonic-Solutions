# Tensor-Parallel All-Reduce Cost

Tensor parallelism splits a model operation across several ranks. Each rank computes a partial result, and some layers must combine those partial results before execution can continue.

An all-reduce produces the same reduced tensor on every participating rank. This problem estimates communication for a ring all-reduce and repeats that estimate across a specified number of identical collectives.

## Message size in bytes

If the tensor contains $N$ elements and each uses $b$ bytes, the full logical message size is

$$
M=Nb
$$

Element count alone is not network volume. One million FP16 elements occupy two million bytes, while the same number of FP32 elements occupy four million bytes.

The ring formulas use $M$ as the size of the tensor being reduced, not the size of a shard already divided by world size.

## The two ring phases

A ring all-reduce can be understood as two phases:

- reduce-scatter combines chunks while leaving each rank with one reduced chunk,
- all-gather circulates those reduced chunks so every rank receives the full result.

With world size $p$, each phase takes $p-1$ communication steps. Total ring steps are

$$
S=2(p-1)
$$

At each step, a rank transfers one chunk of size $M/p$ under the simplified model.

## Per-rank byte volume

Multiplying steps by chunk size gives bytes moved by one rank for one collective:

$$
V_{rank}=2(p-1)\frac{M}{p}
$$

Equivalently,

$$
V_{rank}=2\frac{p-1}{p}M
$$

This is a per-rank figure. It is not summed across all ranks.

As $p$ grows, the factor approaches two, so each rank communicates nearly twice the full message size across both phases.

## A four-rank example

Suppose the logical tensor is two million bytes and world size is four.

The ring uses six steps. Each step transfers a half-million-byte chunk, so each rank moves

$$
6\times500{,}000=3{,}000{,}000\text{ bytes}
$$

The factor is $2(4-1)/4=1.5$ times the full message.

Aggregate traffic across the four ranks would be twelve million bytes, but this function reports three million because its convention is explicitly per rank.

If link bandwidth is one billion bytes per second and per-hop latency is one microsecond, the bandwidth term is $0.003$ seconds and the six-step latency term is $0.000006$ seconds. One collective is estimated at $0.003006$ seconds. Repeating it ten times gives $30{,}000{,}000$ communicated bytes and $0.03006$ seconds per rank.

The example also shows why the terms should remain in consistent base units. Treating the bandwidth as gigabytes per second without converting it to bytes per second would make the division incorrect.

## Bandwidth time

Let link bandwidth be $BW$ bytes per second. Moving the per-rank volume takes ideally

$$
t_{bw}=\frac{V_{rank}}{BW}
$$

Higher bandwidth reduces this component. Doubling the message doubles it.

The formula assumes the effective link bandwidth applies to the ring transfer as modeled. It does not account for topology contention, protocol overhead, or incomplete bandwidth utilization.

## Per-hop latency

Every ring step also carries a fixed latency $\alpha$. The latency component is

$$
t_{lat}=S\alpha
$$

One collective therefore costs

$$
t_{one}=S\alpha+\frac{V_{rank}}{BW}
$$

Large messages are often dominated by the bandwidth term. For many small messages, repeated step latency can become important even when total bytes are modest.

## Repeated collectives

Tensor-parallel layers may invoke the same shaped collective several times. If the count is $C$, totals per rank are

$$
V_{total}=CV_{rank}
$$

and

$$
t_{total}=Ct_{one}
$$

Both bytes and time scale with the count. Multiplying only the byte result would understate latency, while multiplying only time would make outputs inconsistent.

The model treats collectives as sequential costs. It does not overlap them with compute or each other.

## World size one

With one rank, no network reduction is required. The step count is

$$
2(1-1)=0
$$

so bytes and time are both zero, regardless of message size or collective count.

This boundary follows naturally from the formula and is a useful test. A one-rank program may still perform a local operation, but that local work is not communication.

## More ranks do not make communication free

Increasing tensor-parallel world size divides computation and parameter shards across more devices, but all-reduce communication remains.

Per-rank ring volume approaches twice the message size, while step count grows linearly with $p-1$. The latency term therefore becomes more sensitive to larger worlds.

This problem estimates only collective cost. It does not calculate the compute saved by tensor parallelism, so it cannot decide whether a larger world size improves total latency.

## Many small collectives versus one large collective

Suppose two strategies communicate the same total bytes. One uses a single large all-reduce, while the other uses many small all-reduces.

Their bandwidth terms may be similar, but the second strategy pays the ring-step latency term repeatedly. The collective-count input makes this effect visible.

This is why bytes alone are insufficient for communication modeling. Message frequency and fixed latency matter alongside volume.

Conversely, merging messages can reduce repeated latency only when the computation permits those collectives to be combined. The estimator accepts the count as a fact and does not claim that fusion is always possible.

## Output contract and units

The function returns two floating-point 64-bit values:

1. total communicated bytes per rank across all repetitions,
2. total communication time in seconds per rank.

Bytes may be fractional in the arithmetic because of division by world size, even though physical transfers occur in discrete chunks. The estimator follows the given closed-form model.

Bandwidth must be supplied in bytes per second and per-hop latency in seconds. Mixing gigabytes per second or microseconds without conversion creates large unit errors.

## Complexity

The estimator uses constant scalar arithmetic, so its own time and memory are $O(1)$.

Reported communication grows with message size, world size through the ring factors, and collective count, but the function does not simulate individual steps or allocate message-sized buffers.

## Common mistakes to avoid

- Reporting aggregate bytes across ranks violates the per-rank convention.
- Using only $p-1$ steps omits either reduce-scatter or all-gather.
- Sending the full message at every step overestimates ring volume by a factor of $p$.
- Omitting bytes per element confuses tensor elements with network bytes.
- Modeling only bandwidth misses fixed latency for each ring step.
- Forgetting to multiply both bytes and time by collective count makes the outputs inconsistent.
- Returning nonzero network cost at world size one contradicts the absence of communication.
- Mixing seconds with microseconds or bytes with gigabytes breaks dimensional consistency.

The ring estimate separates three ideas cleanly: message size determines chunk volume, world size determines steps and per-rank share, and the hardware contributes bandwidth and fixed latency costs.
