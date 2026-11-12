# Gradient Accumulation Equivalence

Large batches often do not fit in GPU memory at once. Gradient accumulation solves this by splitting the batch into microbatches, running a backward pass on each, and adding their gradients before taking a single optimizer step. The heart of this problem is scaling each microbatch so the accumulated gradient matches the full-batch gradient, even when microbatches have different sizes.

## Why accumulate gradients

Activation memory grows with the number of examples processed at once. If a batch of 1,024 examples does not fit, four microbatches of 256 might. Running them one after another and adding their gradients gives the same update as the full batch, while only one microbatch's activations are in memory at any time.

The cost is time, since the microbatches run sequentially. The benefit is that the update matches the large-batch update, apart from tiny floating-point differences in the order of summation, for models without batch-dependent layers such as batch normalization.

## The full-batch gradient

For a linear model with weights $w$ of length $D$, the mean squared error over all $N$ examples is

$$
L = \frac{1}{N} \sum_m \sum_{i=1}^{N_m} \left( X_{m,i} w - y_{m,i} \right)^2
$$

where $m$ selects a microbatch, $N_m$ is its size, $X_{m,i}$ is the input row of example $i$ in that microbatch and $y_{m,i}$ is its target. Differentiating gives

$$
\nabla_w L = \frac{2}{N} \sum_m \sum_{i=1}^{N_m} \left( X_{m,i} w - y_{m,i} \right) X_{m,i}
$$

The SGD update with learning rate $\eta$ is then $w_{\text{new}} = w - \eta \nabla_w L$.

## Scale by the total count

The full gradient is a sum over microbatches, so it can be built piece by piece. If each microbatch's squared errors are summed and divided by the total example count $N$, the microbatch gradients add up to exactly $\nabla_w L$.

The tempting alternative is to give each microbatch its own mean loss and then average the microbatch results. That gives every microbatch equal weight, which means an example in a small microbatch counts more than an example in a large one. The two approaches only agree when all microbatches have the same size.

## A worked example with unequal sizes

Take $w = [1]$ and learning rate 0.1. The first microbatch has one example with input 1 and target 0. The second has three examples, each with input 1 and target 2. So $N = 4$.

The residuals $X w - y$ are $1$ in the first microbatch and $-1$ for each example in the second. Scaling by the total count gives

$$
\nabla_w L = \frac{2}{4} \left( 1 \cdot 1 + 3 \cdot (-1) \cdot 1 \right) = \frac{2}{4} \cdot (-2) = -1
$$

and the update gives $w_{\text{new}} = 1 - 0.1 \times (-1) = 1.1$.

If each microbatch used its own mean loss instead, the first microbatch's gradient would be $2 \times 1 = 2$ and the second's would be $2 \times (-1) = -2$. Their average is 0, which would leave the weight unchanged even though three of the four examples want it to grow.

## How autograd accumulates

In PyTorch, a leaf tensor is one created directly rather than computed from other tensors, such as a model parameter. When it requires gradients, its grad attribute holds a running gradient. Calling backward on a loss adds the new gradient into that attribute instead of overwriting it. Gradient accumulation relies on this, because one backward pass per microbatch builds up the full gradient there.

This also means the accumulation has to start from a parameter whose gradient is empty. If the parameter already carries a gradient from somewhere else, it would be mixed into the result, and the gradient the caller already has must stay untouched.

After the last microbatch, one SGD step uses the complete accumulated gradient. Stepping after every microbatch would make several small updates with partial gradients, which is a different algorithm.

## Common mistakes to avoid

- **Averaging microbatch means.** Unequal microbatch sizes then give some examples more weight than others.
- **Stepping after each microbatch.** The update must happen once, after all gradients have been accumulated.
- **Zeroing gradients between microbatches.** Clearing the gradient throws away the contributions already collected.
- **Starting from the caller's parameter.** An existing gradient leaks into the result, and backward or an in-place update changes data the caller owns.
- **Keeping the update inside autograd.** Computing the new weights from a tensor that still tracks gradients records the SGD step in the graph, which wastes memory and is not part of the update.
