# TTFT, TPOT, ITL, and Token Throughput

Streaming language-model responses have more than one kind of latency. A user first waits for output to begin, then watches tokens arrive over time. Combining those experiences into one duration makes it difficult to tell whether a system is slow before generation or slow during generation.

This problem calculates separate metrics from request arrivals and valid token timestamps. It also calculates aggregate token throughput across the complete observation window.

## Reading one request timeline

For request $r$, let $a_r$ be its arrival time. Let its valid output-token timestamps be

$$
t_{r,0},t_{r,1},\ldots,t_{r,n_r-1}
$$

The first-token time is $t_{r,0}$, and $n_r$ is the number of valid output tokens.

The timeline has two useful regions. The interval from arrival to the first token measures initial waiting and prompt processing. The intervals between later tokens measure streaming cadence.

## Time to first token

Time to first token, or TTFT, is

$$
TTFT_r=t_{r,0}-a_r
$$

If a request arrives at time 2.0 and emits its first token at 2.4, its TTFT is 0.4 time units.

TTFT can include queueing, prompt processing, model execution, and delivery delays represented by the timestamps. The function does not separate those causes; it measures the user-visible wait encoded by the inputs.

Every request has one TTFT because every valid request produces at least one output token.

## Inter-token latency

Inter-token latency, or ITL, looks at consecutive output timestamps:

$$
\Delta_{r,j}=t_{r,j}-t_{r,j-1}\qquad j=1,\ldots,n_r-1
$$

For timestamps $[0.2,0.5,0.9]$, the intervals are $[0.3,0.4]$. Their mean is $0.35$.

Only gaps after the first token are included. TTFT is not treated as an inter-token gap because it measures a different phase of the request.

Non-decreasing timestamps ensure the intervals are nonnegative. Equal timestamps produce a zero interval, which is allowed by the stated input contract.

## Time per output token

For more than one output token, this exercise defines TPOT as

$$
TPOT_r=\frac{t_{r,n_r-1}-t_{r,0}}{n_r-1}
$$

The denominator is the number of intervals, not the number of tokens. Three output tokens create two gaps.

TPOT equals mean ITL here. The consecutive differences telescope:

$$
\sum_{j=1}^{n_r-1}(t_{r,j}-t_{r,j-1})=t_{r,n_r-1}-t_{r,0}
$$

Dividing both sides by $n_r-1$ gives the same average. Calculating both metrics through their specified forms provides a useful consistency check.

## The one-token convention

A request with one output token has no inter-token interval. Dividing by $n_r-1$ would divide by zero.

This problem defines both TPOT and mean ITL as exactly zero for that request. Zero is a documented placeholder meaning there was no gap to measure.

It should not be interpreted as evidence that the model could generate subsequent tokens instantly. The request simply provides no observation of decoding cadence.

TTFT remains meaningful for a one-token response and is calculated normally.

## Padded timestamp rows

Requests can produce different token counts, but the timestamp input is a rectangular matrix. Shorter rows contain padding after their valid entries.

The output-token count determines the valid prefix of each row:

$$
T_r=token\_timestamps[r,:n_r]
$$

Every per-request calculation must use this prefix only. Padding may contain zero, a huge value, or any other number, so checking the padding value itself is not a reliable mask.

The first valid timestamp equals the supplied first-token time. The last valid timestamp is at index $n_r-1$, not necessarily in the matrix’s last column.

## Aggregate token throughput

Throughput answers how many valid output tokens were emitted per unit of wall-clock time across all requests.

Total tokens are

$$
N_{total}=\sum_r n_r
$$

The observation window begins at the earliest arrival and ends at the latest valid final-token timestamp:

$$
W=\max_r t_{r,n_r-1}-\min_r a_r
$$

Aggregate throughput is

$$
throughput=\frac{N_{total}}{W}
$$

This is one batch-level value, not the average of per-request token rates.

## A two-request example

Suppose both requests arrive at time 0. Request A emits tokens at $[0.1,0.2,0.3,0.4]$, while request B emits two tokens at $[0.2,0.5]$.

Their TTFT values are 0.1 and 0.2. Request A has three intervals of 0.1, so TPOT and mean ITL are 0.1. Request B has one interval of 0.3, so both metrics are 0.3.

Six valid tokens are emitted between the earliest arrival at 0 and the latest completion at 0.5. Aggregate throughput is

$$
\frac{6}{0.5}=12
$$

tokens per unit time.

Any padded entries after request B’s second token must be ignored, even if they are larger than 0.5.

## Why the measurement window matters

Adding per-request rates would double-count periods when requests overlap. Averaging them would ignore idle time before or between requests.

The global window preserves concurrency and idle gaps as they actually appear in the supplied timestamps. If all requests pause for a shared stall, the window grows and throughput falls.

Using the earliest first-token time as the start would omit queueing and prompt time. This contract begins at the earliest request arrival.

Using the padded matrix maximum as the end can let meaningless padding dominate the result. The end must come from valid final tokens only.

## Shapes and outputs

Arrival times, first-token times, and output-token counts each have shape $(R)$. The padded timestamp matrix has shape $(R,T_{max})$.

TTFT, TPOT, and mean ITL each return one value per request with shape $(R)$. Aggregate throughput is a scalar.

The request rows stay aligned across every input and output. Sorting requests is unnecessary and could detach a metric from its original request index.

## Complexity and memory

Let $N_{valid}$ be the total number of valid timestamps. Slicing valid prefixes and calculating gaps requires $O(N_{valid})$ time.

Per-request output and last-valid-time storage use $O(R)$ memory. A fully vectorized mask over the padded matrix may use $O(RT_{max})$, while a row-wise calculation can avoid materializing that mask.

## Common mistakes to avoid

- Including TTFT in ITL mixes initial waiting with streaming cadence.
- Dividing TPOT by the number of tokens instead of the number of gaps underestimates it.
- Reading the last padded column as every request’s completion time lets padding corrupt the metrics.
- Returning NaN for a one-token request violates the required zero convention.
- Averaging per-request throughput does not equal aggregate tokens divided by the global window.
- Starting the throughput window at the first emitted token excludes earlier service activity.
- Treating zero padding as universally invalid fails because padding is unconstrained and valid timestamps may also be zero.

The reliable approach is to respect each request’s valid timeline: separate the initial wait from later token gaps, then measure total output across the earliest-arrival to latest-valid-completion window.
