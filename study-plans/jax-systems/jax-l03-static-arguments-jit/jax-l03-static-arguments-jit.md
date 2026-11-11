# Static Arguments in jit

JAX can compile a Python function into one fast program with jax.jit, and most JAX training and inference code runs that way. Compilation works by running the function once with placeholder values that know their shape and dtype but not their contents. That is a problem for any argument that decides a shape, such as the number of attention heads, because the compiler has to know every shape before it can build the program. This page explains how jit sees your arguments, why a head count has to be marked static, and how a feature vector is split into heads, normalized head by head, and merged back.

## What jit does the first time it sees a function

Take a tiny function that doubles a vector and adds one. When you call the jitted version with a float32 vector of length 3, JAX does not run your Python code on the numbers. It calls the function with a stand-in object called a tracer, which carries only the shape (3,) and the dtype float32. Every jax.numpy operation the tracer passes through is written down, and the record is a small program called a jaxpr. For this function the jaxpr has two equations, one multiply by 2.0 and one add of 1.0, both on a float32 array of shape (3,).

JAX then hands the jaxpr to the XLA compiler, which produces machine code, and stores it in a cache keyed by the shapes and dtypes of the inputs. The next call with another length-3 float32 vector skips Python entirely and runs the cached code. A call with a vector of length 5 does not match the cache, so JAX traces and compiles again.

So inside a jitted function you never see real values. You see tracers, and the only facts they can answer in Python are their shape, their dtype and their number of dimensions.

## Why a traced number cannot set a shape

Plain Python integers passed to a jitted function are traced too. If you pass the number 2, the function receives a tracer for an int32 scalar, and its value is unknown at trace time. Arithmetic on it is fine, since the program can add or multiply it like any other value. Using it as a dimension is impossible, because an XLA program has fixed shapes and the compiler needs the actual number while it builds the program.

You can see this with a jitted function that reshapes a length-8 vector into n rows. Calling it with n equal to 2 raises a TypeError saying that shapes must be 1D sequences of concrete values of integer type, and the message adds that the concrete value was not available because it depends on the argument n. The same failure happens if you compute a dimension from it, for example the width divided by n.

## Marking an argument static

jax.jit accepts two options that fix this. static_argnames takes the names of arguments, and static_argnums takes their positions. A static argument is passed to your Python code as the ordinary Python value, so the reshape receives a real integer. In exchange, the value becomes part of the cache key, and it has to be hashable (an int, a float, a string or a tuple, but never an array).

The cost shows up as extra compilations. In a small experiment, a jitted function with a static n was called four times on length-8 vectors, with n equal to 2, then 4, then 2 again, then 2 with a different vector of the same shape. The Python body ran only twice, once for each distinct value of n. The third and fourth calls reused the program compiled for n equal to 2. This is the right trade for settings such as the head count, which take a handful of values over the life of a model. It is the wrong trade for anything that changes every step, such as a learning rate, which should stay a traced value.

## Splitting a feature vector into heads

Multi-head attention cuts each token's feature vector of width $D$ into $H$ equal pieces of width $D_h = D / H$, and each piece is processed by its own head. For an input $x$ of shape $(B, T, D)$, where $B$ is the batch size and $T$ is the number of tokens, the natural target layout is $(B, H, T, D_h)$. In that layout the head axis sits next to the batch axis, so every later operation can treat each head as an independent sequence.

Head $h$ has to receive the same block of features from every token, features $h D_h$ to $(h + 1) D_h - 1$. A reshape only regroups entries in their stored order, which is token by token, so it can cut each token into blocks but cannot move the head axis in front of the token axis. Placing the heads first is a change in the order of the axes, and that is a different kind of operation from a reshape.

## Why a single reshape scrambles the heads

It is tempting to reshape straight to $(B, H, T, D_h)$, because the total number of entries matches. Take one sequence with two tokens of width 4, with token values $[1, 2, 3, 4]$ and $[5, 6, 7, 8]$, and two heads of width 2.

- **With the heads cut from each token.** Head 0 holds $[1, 2]$ and $[5, 6]$, which are the first halves of both tokens. Head 1 holds $[3, 4]$ and $[7, 8]$.
- **With a single direct reshape.** Head 0 holds $[1, 2]$ and $[3, 4]$, which are both halves of the first token. Head 1 holds both halves of the second token.

The direct reshape gives the right shape with the wrong contents, since each head has become one token's whole vector instead of the same slice across all tokens. The two layouts only agree when there is a single token or a single head, which is why this bug can hide on small inputs.

## Normalizing each head on its own

Some attention variants normalize every head's vector before it is used, so that one head with very large values cannot dominate the dot products of the others. This is RMS normalization applied per head. For one head vector $h$ of width $D_h$ with entries $h_1, \dots, h_{D_h}$,

$$
\operatorname{rms}(h) = \sqrt{\frac{1}{D_h} \sum_{j=1}^{D_h} h_j^2 + \epsilon}, \qquad y = \frac{h}{\operatorname{rms}(h)},
$$

where $j$ indexes the entries, $\epsilon$ is a small positive constant, and $y$ is the normalized head vector with the same shape as $h$. After this step every head vector has a root mean square of almost exactly 1.

Here is one token of width 4, $[2, 0, 1, 1]$, split into two heads with $\epsilon = 10^{-6}$. Head 0 is $[2, 0]$, its mean square is 2, its rms is about 1.414214, and its normalized vector is about $[1.414213, 0]$. Head 1 is $[1, 1]$, its mean square is 1, and its normalized vector is about $[0.9999995, 0.9999995]$, slightly below 1 because of $\epsilon$.

Normalizing the whole vector at once would give a different answer. The mean square of all four entries is 1.5, so every entry is divided by about 1.224745, which gives roughly $[1.632993, 0, 0.816496, 0.816496]$. The second head no longer has unit size, because the first head's large entry set the scale for both.

## What epsilon protects against

A head whose entries are all zero has a mean square of 0, and dividing by its square root would produce NaN values. Adding $\epsilon$ inside the square root keeps the denominator at $\sqrt{10^{-6}} = 0.001$, so a zero head stays exactly zero. Where $\epsilon$ sits matters for small inputs. Inside the square root it acts on the mean square, while adding it after the square root acts on the rms itself, and the two give visibly different results once the entries are around $10^{-3}$.

## Merging heads back

The merged output must be the exact inverse of the split, so that head 0's values return to features 0 to $D_h - 1$ of their own token, head 1's values to the next block, and so on. Any operation that moved an axis during the split has to be undone before the head blocks are joined, or the features land in the wrong tokens. The split and the merge both need $H$ and $D_h$ as plain integers, which is the reason the head count has to be static for the whole function.

## Common mistakes to avoid

- **Leaving num_heads traced.** jax.jit traces every argument by default, so a reshape that depends on the head count fails with a TypeError about concrete values.
- **Reshaping straight to the head layout.** A direct reshape to $(B, H, T, D_h)$ has the right shape but gives each head a whole token instead of the same slice of every token.
- **Merging without the inverse transpose.** Reshaping $(B, H, T, D_h)$ directly to $(B, T, D)$ interleaves tokens and heads, so the merged features land in the wrong places.
- **Normalizing over the full width.** One rms for the whole vector lets a large head shrink the others, so the heads no longer have unit size.
- **Moving epsilon outside the square root.** That changes the result for small values and no longer matches the definition of RMS normalization.
- **Marking an array as static.** Static values must be hashable, so passing x as static raises an error, and a static value that changes every call recompiles every call.