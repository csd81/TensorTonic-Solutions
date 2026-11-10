# Model Memory and Minimum GPU Count

Before serving a model, we need to know whether its complete inference state fits in GPU memory. Parameter storage is usually the largest component, but it is not the only one. KV cache, activations, temporary runtime allocations, and safety headroom also consume capacity.

This problem builds a simple byte estimate from those components and calculates the smallest whole number of GPUs whose usable memory can hold it.

## Begin with model weights

If a model has $P$ parameters and each parameter uses $b$ bytes, weight storage is

$$
W=Pb
$$

A seven-billion-parameter model using two bytes per parameter needs about fourteen billion bytes for weights under this decimal byte convention.

The bytes-per-parameter value may be fractional. Four-bit weights can be represented as $0.5$ bytes per parameter in this estimator. Metadata such as quantization scales must already be reflected in the caller’s chosen value or another supplied memory term because the function does not infer it.

Parameter count and byte precision must not be confused. Quantization changes $b$, while model size changes $P$.

## Add runtime state

The modeled base footprint contains three components:

$$
B_{base}=W+K+A
$$

where $K$ is KV-cache bytes and $A$ is activation bytes.

KV cache grows with active sequences and cached context under the caller’s workload assumptions. Activation memory covers working tensors represented by the input estimate.

Both must be included before calculating headroom. Looking only at model weights can make a deployment appear to fit even though it fails as soon as requests begin running.

This formula does not independently derive KV-cache or activation sizes. Earlier problems covered KV-cache accounting; here those estimates enter as byte totals.

## Runtime overhead and headroom

Allocators, workspaces, framework state, fragmentation, and unmodeled runtime needs can require memory beyond the three explicit components. The problem represents that reserve as a fraction $r$.

Total required bytes are

$$
B_{total}=\left\lceil B_{base}(1+r)\right\rceil
$$

An overhead fraction of $0.10$ means add ten percent of the complete base footprint.

The multiplication applies after weights, KV cache, and activations are summed. Applying overhead to weights alone would leave no reserve for the runtime state.

The result rounds upward because a fractional byte estimate cannot be satisfied by rounding down.

## A complete example

Suppose a model has seven billion parameters at two bytes each. Its KV cache estimate is two billion bytes, activations use half a billion bytes, and runtime overhead is ten percent.

Weight bytes are fourteen billion. The base footprint is

$$
14+2+0.5=16.5\text{ billion bytes}
$$

After overhead,

$$
16.5\times1.10=18.15\text{ billion bytes}
$$

If each GPU offers sixteen billion usable bytes, one GPU is insufficient and two are required.

The example uses decimal byte units. Fractional calculations use binary64 floating-point arithmetic, and ceilings apply to those computed values without intermediate rounding. No unit conversion is performed.

## From required bytes to GPU count

Let $U$ be usable bytes per GPU. The minimum count is

$$
G=\left\lceil\frac{B_{total}}{U}\right\rceil
$$

Ceiling is essential. If required memory is one byte larger than one GPU’s usable capacity, the estimate must return two GPUs.

When required bytes equal capacity exactly, one GPU is sufficient. The formula does not add an extra device at an exact boundary.

The returned count is a capacity result. It does not guarantee that the model can be partitioned across that many devices with suitable kernels or communication topology.

## Usable versus advertised memory

The input is usable bytes per GPU, not necessarily the device’s advertised physical memory.

A deployment may reserve memory for the driver, communication libraries, monitoring, or other processes. The caller can reflect those restrictions by passing a lower usable capacity.

Runtime overhead applies to the modeled workload footprint, while usable capacity expresses how much of each device is available. Mixing those ideas or subtracting the same reserve twice can overstate the GPU count.

This exercise trusts the supplied usable capacity and does not apply another per-GPU margin.

## Fractional parameter storage

Sub-byte weight formats make $W$ a floating-point intermediate. For example, an odd parameter count multiplied by $0.5$ can produce a half-byte mathematical estimate.

The complete post-overhead total is rounded upward once to a whole byte. Rounding each component prematurely can produce a slightly different result.

The returned total and GPU count use 64-bit integers because large models and caches can exceed 32-bit ranges.

Floating-point arithmetic is part of the contract because bytes per parameter and overhead fraction are floats. Within the stated problem, the final ceiling defines the required integer result.

## What the estimate includes

The calculation includes:

- model weight bytes,
- supplied KV-cache bytes,
- supplied activation bytes,
- a multiplicative runtime-overhead fraction.

It does not include network bandwidth, compute throughput, load balancing, model replication, or whether tensors can be split at arbitrary byte boundaries.

Those concerns matter for a real distributed deployment, but adding them would change this focused capacity calculation.

## Output and units

The result is a length-two INT64 tensor:

1. total required bytes,
2. minimum GPU count.

The first entry is bytes, not gigabytes. The second is a dimensionless whole-device count.

Keeping raw bytes avoids ambiguity between decimal gigabytes and binary gibibytes. Unit conversion belongs at the presentation layer.

## Complexity

The estimator uses a fixed number of scalar operations, so its time and auxiliary memory are $O(1)$.

It does not allocate tensors proportional to parameter count or required memory. The reported bytes describe a hypothetical deployment footprint rather than memory used by the calculation itself.

## Common mistakes to avoid

- Omitting KV cache or activations underestimates runtime memory.
- Applying overhead only to weights ignores reserve for the other modeled components.
- Rounding down fractional bytes can make an infeasible plan appear to fit.
- Using floor division for GPU count fails whenever there is a nonzero remainder.
- Treating fractional bytes per parameter as invalid prevents sub-byte estimates.
- Using advertised capacity when the input already specifies usable capacity changes the caller’s assumption.
- Returning gigabytes instead of bytes violates the output unit.
- Interpreting the minimum count as a performance guarantee goes beyond a memory-capacity estimate.

The calculation is a two-stage capacity check: estimate the complete footprint with headroom, round it upward to bytes, then divide by usable per-GPU memory and round upward to whole devices.

Both upward roundings protect capacity rather than estimating an average.
