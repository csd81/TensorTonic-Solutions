# Measure Kernel Timeline Utilization

A GPU runs work on many streaming multiprocessors (SMs) at once, and a profiler can record when each SM is busy. Gaps in that timeline are lost performance, often caused by small kernels that leave most SMs idle or by waits between one kernel ending and the next starting. Kernel fusion combines several kernels into one to close those gaps. The heart of this problem is turning per-SM busy intervals into utilization numbers and comparing a separate-kernel timeline with a fused one.

## Why idle SMs matter

A modern GPU has around a hundred SMs, and a kernel only uses the hardware well if it keeps all of them busy until it finishes. Two patterns waste time. A kernel may not have enough work to fill every SM, so some sit idle the whole time. A sequence of small kernels may also leave gaps between launches from launch overhead and synchronization, and each kernel spends extra time writing intermediate results to memory for the next kernel to read back.

Fusion addresses both. A fused kernel keeps intermediate results on chip instead of writing them out, and it gives each SM a longer, continuous stretch of work. FlashAttention is the best-known example, fusing the steps of attention into one kernel.

## Merging intervals gives true busy time

Each interval records one SM, a start time and an end time. Intervals on the same SM can touch, when two pieces of work run back to back, or overlap, because an SM can host several thread blocks at once and the profiler records each block's interval. Counting them separately would count some time twice, so touching or overlapping intervals on each SM are merged before measuring busy time. A zero-length record carries no work, and if it were counted it could stretch the makespan without adding anything.

The makespan $T$ runs from time 0 to the latest end of any real work. With $N$ SMs and merged busy time $T_{\text{busy},s}$ on SM $s$, utilization is

$$
U = \frac{\sum_s T_{\text{busy},s}}{N T}
$$

which is the fraction of all available SM time spent on real work. A utilization of 1 means every SM was busy for the whole makespan. A fused kernel may run on a different number of SMs from the separate kernels, so each timeline's utilization uses its own $N$. An empty timeline has makespan 0, and its utilization is defined as 0.

Utilization here measures busy time, not arithmetic efficiency. A busy SM can still be stalled waiting on memory, so a high timeline utilization does not by itself mean high FLOP utilization.

## Idle tails reveal uneven work

Each SM's idle tail is the time from its last piece of real work to the end of the makespan. Large idle tails on a few SMs mean the work was unevenly divided, so those SMs finished early and waited while others ran on.

## Speedup comes from makespan

The speedup is the separate makespan divided by the fused makespan, with two empty timelines counting as a speedup of 1. Makespan is wall-clock time, so its ratio is the speedup a user actually sees. Utilization alone could rise even if a job got slower, for example if every SM were kept busy with redundant work.

## A separate and a fused timeline

Take two SMs. In the separate timeline, SM 0 runs from 0 to 2, then from 2 to 3, then from 4 to 5, and SM 1 runs from 0 to 1. The first two intervals on SM 0 touch, so they merge into one busy stretch from 0 to 3, and the third adds another unit, for 4 units of busy time. SM 1 is busy for 1 unit.

The makespan is 5, so utilization is

$$
U = \frac{4 + 1}{2 \times 5} = 0.5
$$

and the idle tails are 0 for SM 0, which works until the end, and 4 for SM 1, which stops at time 1.

In the fused timeline, SM 0 runs from 0 to 3 and SM 1 from 0 to 2. The total busy time is still 5, the same work, but the makespan drops to 3, giving utilization $5 / 6 \approx 0.833333$ and idle tails of 0 and 1. The speedup is $5 / 3 \approx 1.666667$. Fusion removed the gap between SM 0's kernels and spread the work more evenly, so the same amount of work finished sooner.

## Common mistakes to avoid

- **Adding overlapping intervals separately.** Touching or overlapping intervals on one SM must be merged before summing.
- **Merging across SMs.** Each SM's intervals are merged on their own.
- **Dividing by the busy SMs only.** Utilization divides by every SM times the makespan, including SMs that did nothing.
- **Using the earliest start instead of time 0 for the makespan.** Utilization would be overstated when work starts late.
- **Inverting the speedup.** A value below 1 would wrongly say fusion slowed the job down.
