# Calculate Collective Effective Bandwidth

When you time a collective such as an all-reduce, dividing the payload size by the time gives a number that changes with the number of GPUs and cannot be compared with the speed of the links. A model of how a ring algorithm moves data corrects for how much each rank actually sends. This problem centres on computing that per-rank traffic and turning it into an effective bandwidth.

## Why raw payload over time is misleading

Suppose an all-reduce of a gradient tensor takes 2 milliseconds on 2 GPUs and 3.5 milliseconds on 8 GPUs. It looks like the network got slower. In fact each GPU sends $2S \cdot \tfrac{7}{8}$ bytes on 8 GPUs against $2S \cdot \tfrac{1}{2}$ on 2, which is 1.75 times as much, so the links may be running at the same speed in both cases.

To judge whether communication is running at the hardware's speed, you need the bytes each rank really sends, under a model of how the collective is implemented.

## How a ring moves data

In a ring algorithm the $R$ ranks are arranged in a circle, and each one sends only to its neighbour. For a reduce-scatter of a payload of $S$ bytes, the payload is split into $R$ chunks. Over $R - 1$ steps each rank passes one chunk to its neighbour and adds the chunk it receives to its own, until every rank holds one fully summed chunk. Each rank sends $R - 1$ chunks of $S / R$ bytes, which is

$$
S \, \frac{R - 1}{R}
$$

bytes in total. An all-gather runs the same pattern without the additions. An all-to-all, though not a ring, also sends $R - 1$ of its $R$ chunks to other ranks, so all three move this amount per rank.

An all-reduce is a reduce-scatter followed by an all-gather, so it makes two such passes.

## Counting passes gives per-rank traffic

Let $c$ be the number of passes, 2 for all-reduce and 1 for all-gather, reduce-scatter and all-to-all. The modeled bytes per rank and the effective bandwidth over a measured time $t$ are

$$
A = c S \frac{R - 1}{R}, \qquad B = \frac{A}{t}
$$

With a single rank, $R - 1 = 0$, so no communication is needed and both values are zero.

The factor $(R - 1)/R$ approaches 1 as the group grows, so per-rank traffic never exceeds $cS$ however many GPUs join. Beyond a handful of ranks it barely grows at all, which keeps the bandwidth cost of a ring nearly flat. The number of steps, and with it the latency, still grows with $R$, which is why libraries switch to other algorithms for very large groups.

## Why this number is useful

$B$ can be compared directly with the bandwidth of the links between GPUs, whatever the group size. If an all-reduce measures close to the link speed, the links are being used efficiently, and further gains have to come from faster hardware, sending less data, or hiding the communication behind computation. If it measures far below, something else is wrong, such as messages that are too small to fill the links or contention with other traffic.

NVIDIA's NCCL benchmarks report the same quantity under the name bus bandwidth.

## Comparing collectives on eight GPUs

Take a payload of 8,000,000 bytes on $R = 8$ ranks, with each collective taking 2 milliseconds.

For an all-reduce, $c = 2$:

$$
A = 2 \times 8{,}000{,}000 \times \frac{7}{8} = 14{,}000{,}000, \qquad B = \frac{14{,}000{,}000}{0.002} = 7 \times 10^{9}
$$

so the effective bandwidth is 7 billion bytes per second. An all-gather of the same payload in the same time models half the traffic, 7,000,000 bytes, and therefore 3.5 billion bytes per second.

Judged by payload alone, both collectives would show $8{,}000{,}000 / 0.002 = 4 \times 10^9$ bytes per second and look identical, even though the all-reduce moved twice as much data per rank and was using the links twice as hard.

## Common mistakes to avoid

- **Using one pass for all-reduce.** All-reduce is a reduce-scatter plus an all-gather, so it counts two passes.
- **Forgetting the factor $(R - 1)/R$.** Each rank keeps one of the $R$ chunks and sends only the other $R - 1$.
- **Using $1/R$ in place of $(R - 1)/R$.** That models each rank sending one chunk when it actually sends $R - 1$, which badly undercounts traffic on large groups.
- **Treating a single rank as an error.** One rank sends nothing, so traffic and bandwidth are both zero.
- **Comparing payload over time with link speed.** It misjudges ring traffic by the factor $c(R - 1)/R$.
