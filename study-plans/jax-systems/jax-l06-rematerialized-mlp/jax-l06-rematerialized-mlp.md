# Rematerialized MLP Block

Training a deep network needs memory for the weights and the optimizer, and also for every intermediate value the backward pass will read later. For large models those saved activations are often the biggest item of all. Rematerialization, also called gradient checkpointing, throws some of them away after the forward pass and recomputes them when the backward pass needs them. This page explains what gets saved, what jax.checkpoint changes, and how to use it on the residual MLP block in this problem.

## What the backward pass needs to remember

Start with one scalar function, $f(x) = \sin(\sin(x))$. Its derivative by the chain rule is

$$
f'(x) = \cos(\sin(x)) \cdot \cos(x).
$$

To evaluate this at $x = 0.5$, you need $\sin(0.5) \approx 0.4794$ from the forward pass and the two cosines that use it. The result is about 0.7786. Reverse-mode differentiation, which is what jax.grad uses, runs the forward pass first, keeps the values the derivative rules will need (called residuals), and then walks backwards multiplying local derivatives.

For a network layer, the residuals are whole tensors. A tanh hidden layer $h = \tanh(z)$ has derivative $1 - h^2$, so the backward pass needs $h$, a tensor of shape $(B, H)$ where $B$ is the number of examples and $H$ is the hidden width. A matrix multiply $z = x W$ needs its input $x$ to form the gradient of $W$. Every layer leaves something like this behind.

The total adds up quickly. A hidden activation with 1024 tokens and width 4096 in float32 takes 16 MiB, and a stack of 8 such layers keeps 128 MiB alive just for that one kind of residual. Real models have dozens of layers, several residuals per layer and much longer sequences, so activation memory often decides the largest batch that fits on a device.

## Trading compute for memory

Rematerialization makes a different trade. Wrap a piece of the computation in a function and mark it as checkpointed. During the forward pass, only the inputs of that function are saved. During the backward pass, the function is run forward again from those inputs to regenerate its internal values, and then its gradient is computed as usual.

The cost is one extra forward pass of the wrapped piece. A backward pass usually costs about twice a forward pass, so recomputing everything adds roughly a third to the work of a training step. In exchange, the memory for the internal activations of each checkpointed piece exists only briefly, while that piece is being differentiated. The gradients are mathematically identical, because the same operations run on the same inputs. For the sine example, the derivative at 0.5 is about 0.7786 with or without the checkpoint.

## What jax.checkpoint does to the program

JAX transforms functions by tracing them into a small intermediate language called a jaxpr, a list of primitive operations such as sin, mul, dot_general and tanh. You can print the jaxpr of any function with jax.make_jaxpr, which is the best way to see what a transformation really did.

For the gradient of the plain sine example, the jaxpr computes sin and cos of the input during the forward part, keeps both, and multiplies them in the backward part. For the gradient of the checkpointed version, the forward part only computes what the output needs, and the backward part sits inside a nested equation named remat2, which is how this JAX version labels a checkpointed region. Inside it, sin of the saved input is computed a second time before the cosines are taken. That nested equation, containing a recomputed forward operation, is the visible signature of rematerialization.

The function jax.checkpoint (also exported as jax.remat) takes a function and returns one with the same inputs and outputs. It does nothing visible in an ordinary call. Its effect appears only when the function is differentiated, which is why the checkpoint has to wrap code that sits inside the function you pass to jax.grad or jax.value_and_grad. Wrapping only the final loss arithmetic, which contains no tanh, recomputes almost nothing and leaves the expensive activations saved.

## The residual MLP block

The block in this problem is the feed-forward part of a transformer layer, simplified to one hidden layer with a tanh activation and a residual connection.

$$
z = x W_1 + b_1, \qquad h = \tanh(z), \qquad \text{out} = x + h W_2 + b_2
$$

Here $x$ has shape $(B, D)$ with $D$ features per example, $W_1$ has shape $(D, H)$, $b_1$ has shape $(H)$, $W_2$ has shape $(H, D)$ and $b_2$ has shape $(D)$. The residual connection adds the input back, so the block only has to learn a correction to $x$. The loss is the mean of the squared differences between out and a target $y$ of the same shape, averaged over all $B \cdot D$ entries.

The residuals of this block are $x$, $z$ or $h$, and the products along the way. With a checkpoint around the block, only $x$ and the parameters are kept, and $h$ is recomputed when the gradient of $W_2$ and the tanh derivative need it.

## A one-feature example

Shrink everything to single numbers to see the gradients. Take $x = 1$, $W_1 = 0.5$, $b_1 = 0$, $W_2 = 2$, $b_2 = 0$ and target $y = 1.5$, with one example so the mean is the single squared error.

The forward pass gives $z = 0.5$, $h = \tanh(0.5) \approx 0.4621$ and $\text{out} = 1 + 2 \cdot 0.4621 \approx 1.9242$. The error is about 0.4242 and the loss is about 0.1800.

The backward pass starts from the derivative of the squared error, $2 \cdot 0.4242 \approx 0.8485$. That is directly the gradient of $b_2$. The gradient of $W_2$ multiplies it by $h$, giving about 0.3921. Moving into the hidden layer, the gradient with respect to $h$ is $0.8485 \cdot W_2 \approx 1.6969$, and the tanh derivative $1 - h^2 \approx 0.7864$ turns it into about 1.3346 for $z$. Since $x = 1$, that is also the gradient of $W_1$, and it is the gradient of $b_1$ as well.

Notice that $h$ was used twice in the backward pass, for the $W_2$ gradient and for the tanh derivative. Without a checkpoint it is stored. With a checkpoint, the backward pass recomputes $z$ and $h$ from $x$ and the parameters, and every gradient comes out the same.

## Choosing what to save

By default, jax.checkpoint saves nothing except the inputs of the wrapped function and recomputes everything inside it. JAX also accepts a policy argument that keeps some intermediate values. For example, jax.checkpoint_policies.dots_saveable keeps the results of matrix multiplications and recomputes only the cheap elementwise operations such as tanh. Policies like this let you spend memory where recomputation would be slow and save it where recomputation is cheap.

In deep networks the usual pattern is to checkpoint each layer or each group of layers. Then only the inputs of each layer are saved, and during the backward pass one layer at a time is recomputed and differentiated. With $L$ layers grouped into segments of about $\sqrt{L}$ layers, activation memory grows like $\sqrt{L}$ instead of $L$, at the price of one extra forward pass.

## Common mistakes to avoid

- **Checkpointing outside the differentiated function.** jax.checkpoint only changes how gradients are computed, so wrapping code that is not inside the function given to jax.grad saves no memory.
- **Saving everything with the policy.** A policy such as everything_saveable keeps every intermediate value, so the checkpoint recomputes nothing and saves no memory.
- **Wrapping the wrong piece.** A checkpoint around the final loss arithmetic recomputes almost nothing, while the tanh hidden layer, which produces the large activation, stays saved.
- **Dropping the residual connection.** The block output is $x$ plus the MLP branch, and leaving out $x$ changes both the loss and every gradient.
- **Summing instead of averaging.** The loss is a mean over all $B \cdot D$ entries, and a sum scales every gradient by that count.
- **Mixing NumPy into the traced function.** Converting a traced value with NumPy inside the function fails during tracing, so the whole computation must use JAX operations.
- **Expecting different gradients.** Rematerialization changes memory and compute only, so a checkpointed gradient that disagrees with the plain one points to a bug elsewhere.