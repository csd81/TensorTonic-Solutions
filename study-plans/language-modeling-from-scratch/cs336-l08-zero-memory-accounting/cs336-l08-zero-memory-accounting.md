# Account for ZeRO Memory by Stage

Plain data parallelism keeps a full copy of the parameters, gradients and optimizer state on every GPU, so adding GPUs never lets you train a bigger model. ZeRO, the Zero Redundancy Optimizer (Rajbhandari et al., 2020), removes that duplication in three stages by splitting more and more of the model state across the GPUs. This problem centres on how much persistent memory each GPU, called a rank, needs at each stage.

## Where the memory goes

Training a model keeps three kinds of persistent state on each rank. The parameters are the weights themselves. The gradients hold one value per weight after the backward pass. The optimizer state holds whatever the optimizer tracks, such as Adam's two running averages and, in mixed-precision training, a float32 master copy of the weights.

The optimizer state is usually the largest. With 16-bit parameters and gradients and float32 optimizer state, each parameter costs 2 bytes of weights, 2 bytes of gradient and 12 bytes of optimizer state, 4 bytes each for the float32 master weight and Adam's two running averages. The ZeRO paper uses this accounting with fp16 training.

Let $P$, $G$ and $O$ be the total parameter, gradient and optimizer-state bytes, and let $R$ be the number of ranks.

## Data parallelism replicates everything

In plain data parallelism, often called DDP after PyTorch's DistributedDataParallel, every rank holds all three in full:

$$
M_{\text{DDP}} = P + G + O
$$

Every rank applies the same averaged gradient to the same weights, so all $R$ copies of the parameters, gradients and optimizer state are identical, and all but one copy of each is redundant.

## The collectives that make sharding work

Data parallelism relies on three collective operations. An all-reduce sums a tensor across all ranks and leaves the full sum on every rank. A reduce-scatter also sums, but each rank keeps only its own slice of the result. An all-gather does the reverse, so every rank collects every slice to rebuild the full tensor. A reduce-scatter followed by an all-gather moves the same amount of data as one all-reduce.

## Three stages of sharding

Each ZeRO stage shards one more category. Sharding a pool of bytes means splitting it into $R$ pieces with each rank keeping one, so the busiest rank holds $\lceil \text{pool} / R \rceil$ bytes, rounded up because a split rarely comes out even.

**Stage 1** shards the optimizer state. Each rank updates only its own slice of the parameters, then an all-gather gives every rank the full new weights.

$$
M_{\text{ZeRO1}} = P + G + \left\lceil \frac{O}{R} \right\rceil
$$

**Stage 2** also shards the gradients. A reduce-scatter replaces the all-reduce, so each rank receives only the summed gradients for its own slice.

$$
M_{\text{ZeRO2}} = P + \left\lceil \frac{G + O}{R} \right\rceil
$$

**Stage 3** shards the parameters too. Each layer's full weights are gathered just before that layer runs and freed right after, so nothing is kept in full between uses. PyTorch's Fully Sharded Data Parallel (FSDP) applies the same idea.

$$
M_{\text{ZeRO3}} = \left\lceil \frac{P + G + O}{R} \right\rceil
$$

In each formula the sharded categories are treated as one pool and rounded up together. Peak memory in a real run is higher than these numbers, because activations and the temporary gathered buffers come on top of this persistent state.

## What sharding costs

Stages 1 and 2 cost about the same communication as plain data parallelism, because their reduce-scatter and all-gather together move the same data as one all-reduce. Stage 3 adds an all-gather of the parameters in both the forward and backward passes, roughly 1.5 times the traffic of plain data parallelism. That extra traffic is the price of fitting a model whose state is far larger than one GPU's memory.

## A three-rank example

Take $P = 200$, $G = 200$ and $O = 1000$ bytes on $R = 3$ ranks. Plain data parallelism needs $200 + 200 + 1000 = 1400$ bytes per rank. The three ZeRO stages need

$$
200 + 200 + \lceil 1000 / 3 \rceil = 734, \qquad 200 + \lceil 1200 / 3 \rceil = 600, \qquad \lceil 1400 / 3 \rceil = 467
$$

bytes per rank. Stage 1 alone almost halves the memory, because the optimizer state was most of the total. The uneven divisions show why every division rounds up, since splitting 1000 bytes over 3 ranks leaves one rank holding 334.

The ZeRO paper's 7.5 billion parameter example shows the same pattern at real scale. It needs 120 GB per GPU with plain data parallelism, and about 31 GB, 17 GB and 2 GB per GPU with the three stages on 64 GPUs.

## Common mistakes to avoid

- **Sharding the wrong categories at a stage.** Stage 1 shards only optimizer state, stage 2 adds gradients, and stage 3 adds parameters.
- **Rounding down.** The busiest rank holds the larger piece, so every division rounds up.
- **Rounding each category separately.** The sharded categories form one pool that is divided and rounded once.
- **Dividing with floats.** Float division loses exactness for very large byte counts, so results can come out off by one or more.
- **Expecting savings with one rank.** With $R = 1$ every stage needs the same memory as plain data parallelism.
