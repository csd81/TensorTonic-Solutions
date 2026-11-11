# Per-Example Gradient Clipping

Differentially private training, known as DP-SGD (Abadi et al., 2016), limits how much any single training example can change a model. It does this by computing a separate gradient for every example, shrinking each one that is too large, and only then averaging them. This page explains why the clipping must happen per example, how jax.vmap turns a one-example gradient into a batch of gradients, and how the clipping rule treats each gradient, which together make up the function you write here.

## Clipping one gradient by hand

Start with a model that has two parameters and a limit $C = 1$ on the length of any gradient. A gradient of $(3, 4)$ has length $\sqrt{3^2 + 4^2} = 5$, which is above the limit, so clipping divides it by 5 and gives $(0.6, 0.8)$, a vector that points the same way and has length exactly 1. A second gradient of $(0.3, 0.4)$ has length 0.5, which is already within the limit, so clipping leaves it as it is.

## Why a single example must have bounded influence

A model can memorize rare training examples, and an attacker who can probe the model may then learn whether a particular record was in the training set. Differential privacy is a guarantee that the trained model would look almost the same whether or not any one example was included.

To give that guarantee, the training step must bound how far one example can move the parameters. The step averages per-example gradients, so bounding each per-example gradient to a maximum length $C$ bounds the contribution of any one example to $C / N$ in a batch of $N$. Noise added afterwards hides whatever influence remains. This problem builds the clipping and averaging part of that step, which comes before the noise.

## The batch gradient hides the pieces

The usual training step differentiates the mean loss over the batch. By linearity, that gradient is the mean of the per-example gradients, but the transform adds them up during the backward pass and returns only the total. Once they are summed, there is no way to tell which example contributed what, so there is nothing left to clip.

Clipping the averaged gradient does not help either. One example with a huge gradient dominates the average, and shrinking the average shrinks every example's contribution equally, so the outlier still controls the direction of the update. DP-SGD needs each example's gradient on its own.

## vmap turns one example into a batch

The cleanest way to get per-example gradients in JAX is to write the loss for a single example and let JAX vectorize it. jax.vmap takes a function written for one input and returns a function that runs it across a batch, adding a leading batch axis to its inputs and outputs. It does not loop in Python. It traces the function once and rewrites every operation into its batched form.

The in_axes argument says which axis of each argument holds the batch. An entry of 0 means the argument is split along its first axis, one slice per example, and an entry of None means the argument is shared by every example. For per-example gradients, the parameters are shared and the features and targets are split.

Applying jax.vmap to jax.grad of the one-example loss gives a function whose output has the same structure as the parameters, with an extra leading axis of length $N$ on every leaf. If the parameters are a dict with a weight vector of length $D$ and a scalar bias, the result is a dict with an $(N, D)$ array of weight gradients and an $(N,)$ array of bias gradients. Row $n$ of each is exactly the gradient for example $n$.

A Python loop over the examples would give the same numbers, but it would launch a separate small computation for every example and grow slower with the batch size. The vmapped version is one batched program, so its cost is close to that of a single ordinary gradient on the whole batch.

## Measuring the size of a whole gradient

Clipping needs one length per example, so the gradient for example $n$ is treated as a single vector $g_n$ that concatenates every parameter's gradient. Its L2 norm is

$$
\lVert g_n \rVert = \sqrt{\sum_{i=1}^{D} \big(g_n^{w}\big)_i^{2} + \big(g_n^{b}\big)^{2}}
$$

where $g_n^{w}$ is the weight part and $g_n^{b}$ is the bias part. Taking separate norms for the weights and the bias would bound each part to $C$, so the full vector could have length up to $\sqrt{2}\,C$, and the privacy accounting that assumes length $C$ would be wrong.

For the linear model with squared error, $\ell_n = \frac{1}{2}(x_n \cdot w + b - y_n)^2$, the error $r_n = x_n \cdot w + b - y_n$ sets everything. The weight gradient is $r_n x_n$ and the bias gradient is $r_n$, so an example the model already fits perfectly has $r_n = 0$ and a zero gradient.

## Shrinking without stretching

The clipping rule rescales a gradient only when it is too long.

$$
\bar g_n = \frac{g_n}{\max\big(1, \lVert g_n \rVert / C\big)}
$$

If $\lVert g_n \rVert \le C$, the divisor is 1 and the gradient is unchanged. If it is longer, the divisor is $\lVert g_n \rVert / C$, which rescales it to length exactly $C$ while keeping its direction. The rule never lengthens a short gradient, because that would give small examples more weight than they earned.

The form with a maximum also handles a zero gradient safely. Writing the factor as $C / \lVert g_n \rVert$ without the cap would divide by zero for a perfectly fitted example, and zero times infinity is NaN, which would poison the whole average.

## A two-example walk-through

Take $w = [1, 0]$, $b = 0$, $C = 1$, and two examples with targets 0, the first with features $[3, 4]$ and the second with features $[0.5, 0]$.

The errors are $r_1 = 3$ and $r_2 = 0.5$, so the gradients, written as the weight part followed by the bias part, are $g_1 = ([9, 12], 3)$ and $g_2 = ([0.25, 0], 0.5)$. Their norms are $\sqrt{81 + 144 + 9} = 15.297058$ and $\sqrt{0.0625 + 0.25} = 0.559017$.

The first gradient is longer than $C$, so dividing it by 15.297058 gives $([0.588348, 0.784465], 0.196116)$, while the second is shorter and keeps its values. The average of the two clipped gradients is $([0.419174, 0.392232], 0.348058)$.

Compare this with averaging first. The plain average is $([4.625, 6], 1.75)$ with norm $7.775161$, and clipping it gives $([0.594843, 0.771688], 0.225076)$. That direction is almost entirely the first example's, while in the per-example version the second example still moves the weights noticeably.

## Where this step sits in DP-SGD

A full DP-SGD step clips every per-example gradient, sums the clipped gradients, adds Gaussian noise whose scale is proportional to $C$, divides by the batch size, and applies the update. The noise is what turns bounded influence into a privacy guarantee, and a privacy accountant tracks how much privacy is spent over many steps. Per-example clipping is also used outside privacy, as a way to stop a few noisy or mislabeled examples from destabilizing training.

## Common mistakes to avoid

- **Clipping the averaged gradient.** The average has already mixed the examples together, so one outlier still controls the update direction.
- **Clipping each parameter separately.** Separate norms for the weights and the bias let the full per-example gradient exceed $C$.
- **Scaling every gradient to length C.** The factor must be capped at 1, otherwise short gradients are stretched and small examples gain influence.
- **Dividing by a zero norm.** A perfectly fitted example has a zero gradient, and an uncapped factor of $C$ divided by its norm turns the average into NaN.
- **Mapping over the parameters.** The parameters are shared by every example, so their entry in in_axes must be None while the features and targets use axis 0.
- **Reporting the clipped norms.** The norms in the result are the lengths before clipping, which are what tell you how often clipping is active.
