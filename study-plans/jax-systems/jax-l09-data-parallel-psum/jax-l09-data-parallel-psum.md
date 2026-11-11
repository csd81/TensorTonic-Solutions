# Data-Parallel Step with psum

Data parallelism is the most common way to train on many devices. Every device keeps a full copy of the model, works on its own slice of the batch, and the devices then agree on one shared gradient so that every copy takes the same step. This page builds the idea from averages of a few numbers, then shows how jax.shard_map and the collectives psum and pmean express it, and ends with a trap in how JAX differentiates replicated weights inside shard_map.

## Splitting the batch

A training step has two expensive parts, the forward pass and the backward pass, and both scale with the number of examples in the batch. If 8 devices each take one eighth of the batch, each device does one eighth of the work, and all 8 run at the same time.

The weights are small compared with the work done on them, so every device gets the full set. In the language of the previous problem, the batch arrays are split along the data axis of the mesh and the weights are replicated. After the backward pass, each device holds a gradient computed from its own examples only. The 8 gradients differ, because the 8 slices of data differ, and the devices have to combine them before anyone updates the weights.

## Averages of equal-sized pieces

Start with plain numbers. The four values 1, 3, 2 and 6 have mean 3. Split them into two halves of two values each. The first half has mean 2 and the second half has mean 4, and the average of those two means is again 3.

That works because both halves have the same size. With $S$ shards of $M$ values each, the overall mean of all $SM$ values is

$$
\frac{1}{SM} \sum_{s=1}^{S} \sum_{m=1}^{M} v_{s,m} = \frac{1}{S} \sum_{s=1}^{S} \left( \frac{1}{M} \sum_{m=1}^{M} v_{s,m} \right),
$$

where $v_{s,m}$ is value $m$ of shard $s$. The right side is the mean of the shard means. If the shards had different sizes the identity would break. The three values 1, 3 and 2 have mean 2, but splitting them into the shards (1) and (3, 2) gives shard means 1 and 2.5, whose average is 1.75. This is why data-parallel training splits the batch into equal parts.

## The gradient splits the same way

The loss of a batch is a mean over examples, and the gradient of a mean is the mean of the gradients, because differentiation is linear. Write $L_s$ for the loss of shard $s$, computed with the same formula as the full loss but over its own $M = N/S$ examples. Then

$$
L = \frac{1}{S} \sum_{s=1}^{S} L_s, \qquad \nabla L = \frac{1}{S} \sum_{s=1}^{S} \nabla L_s.
$$

So each device can compute $\nabla L_s$ entirely on its own, with no communication, and only the final averaging step needs the devices to talk. The averaged gradient is exactly the gradient that one device would have computed on the whole batch, with no approximation involved.

## Combining values across devices

JAX gives the averaging step a name. Inside per-device code, jax.lax.psum(v, "data") adds the value $v$ from every device along the data axis and hands the total back to all of them. jax.lax.pmean(v, "data") does the same and divides by the number of devices on that axis. Both are called all-reduce operations, because every device contributes a value and every device receives the reduced result.

After pmean, every device holds the same averaged gradient, so every device applies the same update and the replicated weights stay identical. The cost of the all-reduce depends on the size of the gradient, which equals the size of the weights, and not on the batch size. That makes data parallelism attractive when the batch is large and the model is moderate.

## Writing per-device code with shard_map

jax.shard_map turns a function written for one device's block into a function on the global arrays. You give it the mesh, an in_specs entry for every input and an out_specs entry for every output. An in_specs entry of P("data") splits the first dimension of that input across the data axis, so with $N = 16$ examples on 8 devices each device's body receives 2 rows. An entry of P() hands every device the full array.

Inside the body, shapes are local. A batch array of global shape (16, 3) appears as shape (2, 3), and any mean you take inside the body is a mean over the 2 local examples. The out_specs entries describe how to assemble the results. P() says the output is the same on every device, so JAX returns one copy, and JAX checks that this claim is true. A value that still differs between devices cannot leave through P() until a collective such as pmean has made it identical.

## Replicated weights and summed gradients

JAX tracks, for every value inside shard_map, whether it varies across devices. The batch block is device-varying, since every device holds different examples. The weights entered through P(), so they are device-invariant. This tracking changes how jax.grad behaves.

When a device-invariant weight is used in a computation with device-varying data, JAX inserts an implicit broadcast from the invariant value to a varying one. The derivative of that broadcast is a sum over devices. So jax.grad of a shard loss with respect to a replicated weight already returns $\sum_s \nabla L_s$, the gradient summed over all devices, and it is the same on every device. A pmean applied after that leaves the sum unchanged, because averaging 8 equal numbers returns the same number, and the update ends up 8 times too large.

There are two clean ways out. One is to mark the weights as device-varying at the start of the body with jax.lax.pcast(w, "data", to="varying"). Each device then differentiates its own copy, gets its own $\nabla L_s$, and pmean averages the 8 of them. The other is to apply pmean to the loss before differentiating, so the derivative of the averaged loss carries the factor $1/S$ through the implicit sum.

## A worked step on eight devices

Take a scalar model $\hat y = w x$ with no bias and $w = 0.5$, and 8 examples, one per device, with inputs 1, -1, 2, 0, 1, 3, -2, 1 and targets 2, -1, 3, 1, 2, 5, -3, 1. The residuals $w x - y$ are -1.5, 0.5, -2, -1, -1.5, -3.5, 2 and -0.5. Each device's loss is its squared residual, and the average of the 8 squared residuals is 3.28125.

The gradient of $(wx - y)^2$ with respect to $w$ is $2x(wx - y)$. The 8 local gradients are -3, -1, -8, 0, -3, -21, -8 and -1, and their mean is -5.625. A step with learning rate 0.1 moves $w$ to $0.5 + 0.1 \cdot 5.625 = 1.0625$.

Running this in shard_map confirms both behaviours. With pcast applied to $w$ first, the gradient after pmean is -5.625. Without it, jax.grad already returns the sum of the local gradients, -45, and pmean keeps it at -45, which is 8 times the correct value.

## Where this step sits in training

A real training loop repeats this step thousands of times. Frameworks often hide the communication by letting jax.jit place the all-reduce for you, based on the input shardings. Writing it with shard_map makes every piece visible, the split of the batch, the local backward pass and the single all-reduce of the gradient, which is the layout that larger schemes such as fully sharded data parallelism start from.

## Common mistakes to avoid

- **Summing instead of averaging.** psum adds the 8 shard gradients, which is 8 times the batch gradient, so either use pmean or divide the psum by the number of devices.
- **Averaging an already summed gradient.** A gradient taken inside shard_map with respect to replicated weights is already a sum over devices, and pmean on it changes nothing.
- **Dividing by the global batch size inside the body.** The body only sees N/8 examples, so a shard loss that divides by N is 8 times too small before the average.
- **Taking the mean over outputs as well.** The loss sums over the K outputs and averages over examples, so a mean over every entry divides by K once too often.
- **Reporting the loss after the update.** The loss belongs to the weights that produced the gradient, which are the weights before the step.
- **Forgetting that Explicit meshes demand pre-sharded inputs.** jax.make_mesh gives Explicit axes by default, and then shard_map rejects inputs whose sharding does not already match in_specs.