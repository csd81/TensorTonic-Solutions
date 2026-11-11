Gradient accumulation lets you simulate a larger effective batch size by running several forward-backward passes over small micro-batches, summing gradients, and performing a single weight update. This is standard when GPU memory cannot fit the desired batch size.

## Why batch size matters

* Larger batches give more stable gradient estimates and often smoother convergence
* Memory for activations and intermediate values grows with batch size during backprop
* When the target batch size $B$ does not fit, we process $K$ micro-batches of size $B/K$ and accumulate their gradients

## How accumulation works

* For each micro-batch: forward pass, compute loss, call .backward(). PyTorch adds the new gradients into the existing .grad tensors instead of overwriting them
* Do not call optimizer.zero_grad() between micro-batches within the accumulation window
* After $K$ micro-batches: divide the accumulated gradient by $K$ to average, then take one optimizer step and zero gradients before the next accumulation cycle
* Wrap the weight update in torch.no_grad() so the in-place update is not recorded in the computational graph

## Mathematical justification

For a loss that is a sum over samples, the gradient of the sum is the sum of the gradients. So averaging gradients over $K$ micro-batches is equivalent to one backward pass over the concatenated batch, up to the same scaling:

$$
\frac{1}{K} \sum_{k=1}^{K} \nabla_\theta L_k = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell_i
$$

when $B = K \times \text{micro-batch size}$ and $L_k$ is the loss for micro-batch $k$. So one step after $K$ accumulations matches one step with a batch of size $B$, while using only $B/K$ samples in memory at a time.

## Implementation details

* After .backward(), param.grad holds the sum of gradients from all backward calls since the last zero. Divide by $K$ before applying the optimizer if your optimizer expects the mean gradient over the effective batch
* If you use a loss that already averages over the micro-batch, the accumulated .grad is the sum of $K$ averaged gradients; dividing by $K$ then gives the mean over the full effective batch
* Batch normalization and similar layers see only the micro-batch; for very small micro-batches, consider nn.GroupNorm or running stats over the full effective batch if needed
