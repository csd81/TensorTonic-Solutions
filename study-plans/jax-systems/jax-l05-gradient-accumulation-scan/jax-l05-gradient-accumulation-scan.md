# Gradient Accumulation with Scan

Gradient accumulation lets a model train with a batch that is too large to fit in memory at once. The batch is split into smaller microbatches, each microbatch's gradient is computed in turn, and the results are combined into the gradient of the whole batch. This page explains why the combination is exact, why a Python loop is the wrong way to express it in JAX, and how lax.scan runs the same step over every microbatch while keeping the compiled program small.

## Why a large batch may not fit

A training step runs the model forward on a batch and then backward to get the gradients. The backward pass needs intermediate values from the forward pass, such as the hidden activations of every layer, so those values are kept in memory until the gradient is done. Their size grows in proportion to the number of examples in the batch.

For a large model, the activations of a full batch can be many times larger than the parameters, so a batch size that trains well can need far more memory than the device has, even when an eighth of that batch fits comfortably. The batch size is still worth keeping large, because it controls how noisy each gradient is, and a training recipe tuned for one batch size often behaves differently at another.

Gradient accumulation separates the two. The optimizer sees the gradient of the full batch, while the device only ever holds the activations of one microbatch.

## The average of averages

The loss of a batch is the mean of the per-example losses, and the gradient of a mean is the mean of the gradients. A full batch of $B$ examples split into $M$ microbatches of $S = B / M$ examples each therefore gives

$$
\nabla L_{\text{full}} = \frac{1}{B} \sum_{n=1}^{B} \nabla \ell_n = \frac{1}{M} \sum_{j=0}^{M-1} \left( \frac{1}{S} \sum_{n \in \text{microbatch } j} \nabla \ell_n \right)
$$

Here $\ell_n$ is the loss of example $n$ and $\nabla$ means the gradient with respect to the parameters. The bracket is the mean gradient of microbatch $j$, so the full-batch gradient is the plain average of the $M$ microbatch gradients. This works only because every microbatch holds the same number of examples, which makes all the inner means equally weighted.

Take a model with one parameter $w$ that predicts $w x$, with squared error $(w x - y)^2$ as the per-example loss. Its gradient for one example is $2 x (w x - y)$. Use $w = 1$, inputs $x = [1, 2, 3, 4]$ and targets $y = [2, 2, 2, 2]$. The residuals $w x - y$ are $-1, 0, 1, 2$, and the per-example gradients are $-2, 0, 6, 16$, whose mean is 5.

With two microbatches of two examples each, the first microbatch has mean gradient $(-2 + 0) / 2 = -1$ and the second has $(6 + 16) / 2 = 11$. Their average is 5, the same as the full batch. The losses combine the same way, since the full-batch loss is 1.5 and the two microbatch losses are 0.5 and 2.5.

Now split the same batch unevenly into one example and three examples. The microbatch gradients are $-2$ and $22 / 3 \approx 7.333$, and their plain average is about 2.667, which is wrong. An uneven split needs each microbatch weighted by its size, which is why accumulation code usually insists on equal microbatches.

## A Python loop gets unrolled

The obvious way to accumulate is a Python for loop that computes one microbatch gradient per iteration and adds it to a running total. That gives the right numbers, but under jax.jit it builds a very different program from what it appears to.

When jit traces a function, it runs the Python code once and records every array operation into a program called a jaxpr. A Python loop runs during tracing, so each iteration records its own full copy of the forward and backward pass. For the two-layer model of this problem with D, H and K all equal to 4 and a batch of 64 examples, a Python loop that slices out each microbatch traces to 87 operations with 2 microbatches, 321 with 8 and 1,257 with 32. Compile time grows with the program, so accumulating over many microbatches with a Python loop makes every new configuration slow to compile.

## How scan runs one step many times

The function jax.lax.scan expresses a loop whose body is traced only once. It takes a body function, an initial carry and a stacked array of inputs. The carry is the state that flows from one iteration to the next, and the stacked inputs have a leading axis that scan walks along, handing the body one slice per iteration.

$$
(\text{carry}_{j+1}, \; \text{out}_j) = f(\text{carry}_j, \; \text{xs}[j]), \qquad j = 0, 1, \dots, M - 1
$$

Here $f$ is the body function, $\text{xs}[j]$ is slice $j$ of the stacked inputs, and $\text{out}_j$ is an optional per-step output that scan stacks into an array. The compiler sees one body and a loop count, so the program has the same size whatever $M$ is. A scan-based version of the same computation traces to 12 top-level operations for 2, 8 or 32 microbatches alike.

The carry can be any pytree, for example a tuple holding a scalar and a dict of arrays. It must have the same structure, shapes and dtypes on every iteration, because the compiled loop reuses one memory layout for it. If the initial carry is an int32 zero and the body adds a float32 value to it, scan raises a TypeError saying that the carry input and carry output must have equal types. Starting the carry from float32 zeros of the right shapes avoids that.

## Laying out the microbatches

Scan needs the microbatches stacked along a new leading axis. An array of shape (B, D) can be reshaped to (M, S, D) without copying any data. Because rows are stored one after another, the reshape groups consecutive rows, so slice $j$ holds rows $jS$ to $(j + 1)S - 1$.

The inputs and the targets have to be grouped in the same way, so that each example stays paired with its own target. Reshaping x and y with different layouts would silently pair examples with the wrong targets, and the loss would no longer be the loss of the batch.

## Running totals live in the carry

The formula above is an average of $M$ terms, and an average can be built up one term at a time. Keeping a running sum of the microbatch losses and a running sum of the microbatch gradients, and dividing both by $M$ once all microbatches are done, gives exactly the averages of the formula. Dividing each term by $M$ as it arrives gives the same result.

The gradient of a microbatch is a pytree with the structure of the parameters, so the running gradient sum is a pytree of that structure too, and it fits naturally into the carry next to the scalar loss sum. The matrix products of the model all happen inside the body on arrays with $S$ rows instead of $B$, which is what keeps the activation memory small.

## The cost of accumulating

Accumulation trades time for memory. The peak activation memory now depends on $S$ instead of $B$, but the $M$ microbatches run one after another, and a small microbatch may not keep the hardware fully busy. The parameters and the running gradient total stay in memory for the whole loop.

In a full training step, accumulation comes first. The averaged gradient then goes through clipping and into the optimizer, exactly as if it had come from one large batch. When training also runs on several devices, each device usually accumulates over its own microbatches before the devices average their gradients once, which keeps communication to one exchange per step.

## Common mistakes to avoid

- **Looping in Python.** The loop is unrolled during tracing into one copy of the model per microbatch, and the program and its compile time grow with the number of microbatches.
- **Summing without dividing.** The total of the microbatch gradients is $M$ times the full-batch gradient, so the result must be divided by the number of microbatches.
- **Pairing inputs with the wrong targets.** Reshaping x and y with different layouts mixes examples with other examples' targets.
- **Differentiating through the scan.** Calling jax.grad on a function that scans over the microbatch losses builds a separate backward scan that keeps the activations of every microbatch, which defeats the memory saving.
- **Computing the gradient outside the scan.** A full-batch forward or backward pass next to the scan holds the activations of the whole batch and defeats the memory saving.
- **Starting the carry with the wrong dtype.** A carry whose dtype or structure changes between iterations makes lax.scan raise a TypeError.
- **Using unequal microbatches.** A plain average of microbatch means is exact only when every microbatch has the same number of examples.