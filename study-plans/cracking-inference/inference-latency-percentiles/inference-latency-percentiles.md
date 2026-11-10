# P50, P95, and P99 Inference Latency

Inference requests do not all finish in the same amount of time. One may arrive while the server is idle, while another waits behind a large batch or competes for memory. A single average hides this spread.

Percentiles describe the distribution from the user’s point of view. P50 represents a typical request, while P95 and P99 reveal progressively slower tail requests.

This problem calculates those three values from observed latency samples using one exact interpolation convention.

## What a percentile means

After sorting latencies from fastest to slowest, a percentile identifies a position in that ordered list.

P50 is the median. Roughly half the observations are at or below it. P95 marks the high-latency region where about 95 percent of observations are at or below the reported value. P99 focuses even farther into the tail.

A smaller percentile is not automatically more important. P50 summarizes ordinary behavior, while P99 helps expose rare but painful delays that an average can conceal.

## Sorting is the first step

Suppose the recorded latencies are

$$
[40,10,30,20]
$$

Their input order reflects collection order, not rank. Percentile calculation uses the ascending values

$$
[10,20,30,40]
$$

The original tensor should remain unchanged. Sorting can occur internally, but the function must not rearrange the caller’s samples in place.

Repeated values remain separate observations. A list such as $[10,10,10,100]$ contains four measurements, and the three repeated tens each occupy a rank.

## The rank convention

For percentile fraction $p$ between zero and one and $n$ sorted samples, this exercise uses

$$
r=p(n-1)
$$

The factor $n-1$ places $p=0$ exactly at index zero and $p=1$ exactly at the last index.

The requested fractions are $0.50$, $0.95$, and $0.99$. A fractional rank lies between two observed values and requires interpolation.

Percentile libraries support several rank conventions. Nearest-rank, lower, higher, midpoint, and linear rules can return different answers on the same small dataset. The stored contract specifically requires the linear $p(n-1)$ convention.

## Linear interpolation

Let

$$
j=\lfloor r\rfloor,\qquad k=\lceil r\rceil
$$

and let the fractional part be

$$
f=r-j
$$

For sorted samples $v$, the percentile is

$$
P_p=v_j+f(v_k-v_j)
$$

When $r$ is an integer, $j=k$ and the observed sample is returned exactly. When $r$ lies between indices, $f$ determines how far to move from the lower value toward the upper value.

## A complete four-sample example

Using sorted values $[10,20,30,40]$, P50 has rank

$$
0.50(4-1)=1.5
$$

It lies halfway between 20 and 30, so P50 is 25.

P95 has rank $0.95\times3=2.85$. It lies 85 percent of the way from 30 to 40, giving 38.5.

P99 has rank $0.99\times3=2.97$, giving 39.7.

These interpolated numbers need not appear among the original measurements. They estimate a location between neighboring ranks according to the declared convention.

## Why tail percentiles matter

Imagine 99 requests finish in 100 milliseconds and one finishes in 10 seconds. The average rises, but it does not say how concentrated the problem is. P50 stays near the common experience, while a high tail percentile moves toward the slow observation.

Comparing P50 with P99 shows whether latency is tightly clustered or has a long tail. A large gap can point toward queueing bursts, cold starts, uneven prompt lengths, or resource contention, although this function only calculates the statistics and does not diagnose their cause.

Percentiles should always be reported with the sample window and workload context. This exercise receives only a tensor of samples, so those operational details remain outside its output.

The three requested percentiles are summaries of the same sample set, not three separate experiments. Their values must be monotonic: P50 cannot exceed P95, and P95 cannot exceed P99 for valid data.

## The single-sample case

With one observation, $n-1=0$. Every requested percentile has rank zero.

Therefore P50, P95, and P99 all equal the sole latency. There is no second point to interpolate with, and no special approximation is required.

This is a useful boundary test because code that assumes distinct lower and upper indices may fail unnecessarily.

## Repeated values

Repeated observations follow the same formula. If both neighboring values are equal, interpolation returns that shared value because their difference is zero.

No deduplication should occur. Removing duplicates changes the empirical distribution by discarding how often a latency was observed.

For example, $[10,10,10,100]$ describes a very different workload from $[10,100]$, even though they contain the same unique values.

## Validate before calculating

Inputs are guaranteed to be one-dimensional, nonempty, finite and nonnegative. No input-validation code is required.

Zero is valid. A request measured as zero latency may be unusual, but it remains finite and nonnegative.

## Output contract and dtype

The result is a floating-point tensor with three entries in this exact order:

1. P50,
2. P95,
3. P99.

Returning the correct values in another order is still incorrect because downstream consumers interpret columns by position.

Floating-point output is necessary because interpolation can produce fractions even when every input sample is a whole number.

## Complexity and memory

Sorting $n$ observations takes $O(n\log n)$ time in a conventional implementation. Once sorted, each requested percentile takes constant work, and there are only three.

Working storage is commonly $O(n)$ when sorting into a new tensor. The output itself has constant size.

For the small tensors in this exercise, clarity and the exact convention matter more than specialized approximate percentile structures used by large monitoring systems.

## Common mistakes to avoid

- Calculating ranks on unsorted values makes array position depend on arrival order rather than latency.
- Using $pn$ instead of $p(n-1)$ implements a different percentile convention.
- Choosing the nearest observation instead of interpolating changes small-sample answers.
- Deduplicating repeated samples changes the empirical distribution.
- Sorting the caller’s tensor in place introduces an unexpected side effect.
- Returning P99 before P95 violates the fixed output order.
- Using only the mean does not answer a percentile question.

The calculation is a disciplined reading of an ordered sample set: sort without mutating the input, locate each fractional rank with $p(n-1)$, and interpolate between the two neighboring observations.
