# <span style="font-size: 20px;">Gradient Accumulation</span>

## The Memory-Batch Size Tradeoff

Larger batch sizes produce lower-variance gradient estimates, leading to smoother training. However, large batches require proportionally more GPU memory. When the desired batch size exceeds available memory, gradient accumulation lets you simulate a large batch by processing it in smaller chunks.

## How It Works

Instead of computing the gradient on one large batch and updating immediately, you:

1. Process a micro-batch and compute its gradient
2. Accumulate (sum) this gradient into a buffer
3. Repeat for $K$ micro-batches
4. Average the accumulated gradient by dividing by $K$
5. Apply one parameter update using this averaged gradient

The effective batch size is $\text{micro\_batch} \times K$, but memory usage equals processing just one micro-batch.

## Mathematical Equivalence

For a dataset split into micro-batches $B_1, \ldots, B_K$:

$$
\nabla_{\text{accum}} = \frac{1}{K} \sum_{k=1}^{K} \nabla_{B_k} f(w)
$$

This averaged gradient approximates the gradient over the combined batch $B_1 \cup \cdots \cup B_K$, assuming the weights $w$ are held fixed across the $K$ micro-batches.

## Key Detail

The weights must NOT be updated between micro-batches within one accumulation window. All $K$ gradients are computed at the same $w$, then one update is applied. This is what makes accumulation equivalent to a single large-batch step.

## Practical Impact

Gradient accumulation is essential for training large models (LLMs, vision transformers) where batch sizes of 256+ are needed for stable training but only a handful of samples fit in GPU memory at once.