# Synchronize Data-Parallel Gradients

Data parallelism is the simplest way to train on many GPUs. Every GPU holds a full copy of the model, processes a different slice of the batch, and the GPUs average their gradients before each update so all copies stay identical. The core of this exercise is computing that average correctly, including ranks that produced no gradient, and applying the same update to every replica.

## Why data parallelism works

The gradient of a loss averaged over a batch is the average of the gradients over its parts. If a batch is split evenly across $R$ ranks, where a rank is one participating GPU, and each rank computes the gradient of its own mean loss, averaging those $R$ gradients gives the gradient of the full batch. Each rank then applies the same update to the same starting parameters, so after an initial broadcast of the parameters the replicas stay in sync without sending the parameters again.

In practice the averaging is done with an all-reduce, which sums the gradients across ranks so every rank can divide by $R$. This step runs after every rank's backward pass and before the optimizer step. This problem simulates it, together with an SGD update, on a single machine.

## The mean gradient

Let $g_{r,j}$ be rank $r$'s gradient for parameter $j$. The mean gradient is

$$
\bar{g}_j = \frac{1}{R} \sum_{r=0}^{R-1} g_{r,j}
$$

Some ranks may have no gradient for a parameter. An embedding row for a token that never appears in one rank's slice of the batch is a typical case, and PyTorch leaves that gradient as None. A missing gradient counts as zero, and the divisor stays $R$. Dividing by the number of ranks that happened to produce a gradient would give those ranks extra weight and make the result depend on which ranks used the parameter.

## The SGD update

With learning rate $\eta$, each replica of parameter $j$ becomes

$$
\theta'_j = \theta_j - \eta \, \bar{g}_j
$$

All replicas start from the same $\theta_j$ and use the same $\bar{g}_j$, so they finish with identical values. Each rank still needs its own copy of the result, because in a real system every GPU stores its parameters in its own memory.

## Accumulate in float32, store in the parameter's dtype

Gradients, and sometimes the parameters themselves, are often stored in bfloat16. Adding many bfloat16 values directly loses precision, since each addition rounds to about two or three significant digits, so the summing belongs in float32.

The mean that the step hands back has the same dtype as the parameter, matching the dtype a real all-reduce on bfloat16 tensors returns. The update then starts from that rounded mean even though it is computed in float32, and rounding the mean first can change the final parameter slightly. Real systems that want the extra accuracy communicate gradients in float32.

## A missing-gradient example

Take $R = 3$ ranks and one parameter with value $[2.0]$. Rank 0's gradient is $[1.0]$, rank 1 has no gradient, and rank 2's gradient is $[5.0]$. The learning rate is 0.5.

The missing gradient counts as zero, so

$$
\bar{g} = \frac{1.0 + 0 + 5.0}{3} = 2.0, \qquad \theta' = 2.0 - 0.5 \times 2.0 = 1.0
$$

All three ranks end with $[1.0]$. Averaging over only the two ranks that had gradients would give $\bar{g} = 3.0$ and $\theta' = 0.5$, a step 50 percent too large.

## Scaling beyond plain data parallelism

Every rank in plain data parallelism stores the full model, its gradients and its optimizer state, which limits the model size to what one GPU can hold. Sharded variants such as ZeRO and FSDP split those tensors across ranks and replace the all-reduce with a reduce-scatter and an all-gather. The averaging rule in this problem stays the same in all of them.

## Common mistakes to avoid

- **Dividing by the number of gradients present.** A missing gradient adds zero to the sum, and the divisor stays the full rank count $R$.
- **Summing in bfloat16.** Low-precision additions lose accuracy, so the sum is accumulated in float32.
- **Skipping the rounding of the mean.** The update starts from the mean stored in the parameter's dtype.
- **Sharing or overwriting tensors.** Every rank's updated parameters live in its own memory, so the originals stay untouched and no two ranks share a tensor.
- **Updating with each rank's own gradient.** Replicas drift apart unless every rank applies the same averaged gradient.
