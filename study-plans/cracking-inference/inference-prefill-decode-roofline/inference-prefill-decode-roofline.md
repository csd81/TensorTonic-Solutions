# Prefill and Decode with the Roofline Model

Counting operations alone does not predict how long inference takes. Hardware must also move model data from memory into compute units. A phase can wait mainly for arithmetic or mainly for memory traffic.

The roofline model combines these limits. This problem applies a deliberately simple model to prefill and decode, producing work, bytes moved, arithmetic intensity, attainable performance, and a latency lower bound for each phase.

## First understand the two phases

Prefill processes all prompt tokens together. Matrix operations can reuse loaded weights across many tokens in the batch and prompt.

Decode generates tokens one step at a time. Under this exercise’s model, the complete weights are read again for every generated step.

Both phases use the same model parameters. Their different opportunities for weight reuse create different arithmetic intensities.

## Estimating parameter count

The problem uses the first-order approximation

$$
P=12Ld^2
$$

where $L$ is the layer count and $d$ is the model width.

This estimate represents dominant transformer matrix parameters. It is a modeling convention, not an exact count for every architecture.

Attention’s quadratic sequence term, embeddings, normalization, biases, cache traffic, and other smaller costs are excluded by the contract. Adding them would produce a different estimator.

## Counting floating-point work

A multiply-accumulate is counted as two floating-point operations, so processing $T$ tokens costs

$$
F=2PT
$$

Prefill processes batch size $B$ and prompt length $S$:

$$
F_{pre}=2PBS
$$

Decode produces $G$ tokens for each batch item:

$$
F_{dec}=2PBG
$$

Generated length scales total decode work because the model performs one step after another.

## Modeling bytes moved

Let $e$ be bytes per parameter element.

Prefill reads the model weights once for the whole batched prompt pass in this simplified model:

$$
M_{pre}=Pe
$$

Decode reads those weights once per generated step:

$$
M_{dec}=PeG
$$

Batch size does not multiply either byte formula here. The model assumes one weight read can serve the batch at a step.

These expressions model dominant weight traffic only. They do not attempt a complete hardware trace.

## Arithmetic intensity

Arithmetic intensity measures useful floating-point work per byte moved:

$$
I=\frac{F}{M}
$$

Substituting the prefill formulas gives

$$
I_{pre}=\frac{2BS}{e}
$$

For decode, generated length cancels:

$$
I_{dec}=\frac{2B}{e}
$$

Prefill intensity grows with prompt length because loaded weights are reused across prompt tokens. Decode intensity does not grow with generated length because both work and weight bytes repeat for every step.

This cancellation is an important correctness check.

## The two hardware ceilings

Let $P_{peak}$ be peak compute throughput in FLOP/s and $BW$ be memory bandwidth in bytes/s.

Compute cannot exceed $P_{peak}$. Memory bandwidth supports at most

$$
P_{bandwidth}=BW\times I
$$

FLOP/s at arithmetic intensity $I$.

Attainable performance is the lower ceiling:

$$
P_{attain}=\min(P_{peak},BW\times I)
$$

When the bandwidth ceiling is lower, the phase is bandwidth-bound. When peak compute is lower, the phase is compute-bound.

## The ridge point

The ceilings meet at arithmetic intensity

$$
I_{ridge}=\frac{P_{peak}}{BW}
$$

A phase below this intensity is limited by bandwidth in the idealized model. A phase above it is limited by compute.

For hardware with 100 trillion FLOP/s and 1 trillion bytes/s bandwidth, the ridge point is 100 FLOP/byte.

With two-byte elements and batch one, decode intensity is $2/2=1$ FLOP/byte, far below the ridge. A prompt of 512 tokens gives prefill intensity $2\times512/2=512$ FLOP/byte, above the ridge.

This example explains why prefill and decode can stress different hardware resources under the problem’s assumptions.

## Latency lower bound

The compute time cannot be less than

$$
t_{compute}=\frac{F}{P_{peak}}
$$

The memory time cannot be less than

$$
t_{memory}=\frac{M}{BW}
$$

The roofline latency is the larger value:

$$
t=\max\left(\frac{F}{P_{peak}},\frac{M}{BW}\right)
$$

This equals $F/P_{attain}$ under the same model. Calculating both forms directly provides a useful consistency check.

The result is a lower bound because real execution also has kernel overhead, imperfect utilization, synchronization, and other traffic omitted from the formulas.

## Reading the output table

The output has two rows and five columns. Row zero is prefill and row one is decode.

The columns are:

1. total floating-point operations,
2. total bytes moved,
3. arithmetic intensity in FLOP per byte,
4. attainable performance in FLOP/s,
5. estimated latency in seconds.

Keeping units explicit prevents mistakes such as comparing bytes with gigabytes per second or returning milliseconds where seconds are required.

Floating-point 64-bit output helps preserve large counts and ratios without integer truncation.

## How inputs change the phases

Increasing model width raises parameter count quadratically because of $d^2$. Increasing layer count raises it linearly. Both phase work and weight traffic grow with parameter count.

Increasing prompt length raises prefill work but leaves its modeled weight bytes unchanged, so prefill intensity rises.

Increasing generated length raises decode work and bytes by the same factor. Decode intensity and attainable FLOP/s remain unchanged, while total latency grows.

Increasing batch size raises work in both phases without increasing the modeled weight reads, so arithmetic intensity increases.

Changing element size affects bytes but not FLOPs. Larger elements reduce intensity and increase the memory-time bound.

## Estimator complexity versus modeled work

The function evaluates a fixed number of scalar formulas, so its own time and memory complexity are $O(1)$.

The numbers it reports can be enormous. Constant-time estimation does not mean the modeled inference is constant work.

No tensors proportional to model size, prompt length, or generated length should be created. The function returns only a $2\times5$ metric tensor.

## Common mistakes to avoid

- Counting a multiply-accumulate as one operation misses the required factor of two.
- Multiplying prefill weight bytes by prompt length removes the modeled reuse that distinguishes the phases.
- Omitting generated length from decode bytes makes decode intensity incorrectly grow with generation length.
- Multiplying bytes by batch size contradicts this problem’s shared weight-read model.
- Taking the larger performance ceiling predicts throughput hardware cannot deliver.
- Taking the smaller latency component ignores the active bottleneck.
- Mixing FLOP/s with bytes/s without arithmetic intensity produces incompatible units.
- Adding attention-quadratic or KV-cache terms changes the explicitly simplified model.
- Treating the latency result as a measured runtime ignores real overheads outside the roofline bound.

The roofline view asks one practical question: does this phase have enough arithmetic work per byte to use the hardware’s compute capacity? Prefill often gains intensity through weight reuse across many prompt tokens, while decode repeatedly moves the same weights for relatively little work at each step.
