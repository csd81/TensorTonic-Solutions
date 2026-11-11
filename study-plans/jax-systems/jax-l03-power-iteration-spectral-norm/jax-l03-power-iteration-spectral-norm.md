# Spectral Norm by Power Iteration

The spectral norm of a matrix is the largest factor by which it can stretch a vector. It is the quantity behind spectral normalization, which keeps a layer from amplifying its inputs too much, and it shows up in stability analysis and in optimizers that work with the geometry of weight matrices. A full singular value decomposition gives it exactly, but that is expensive, so training code usually estimates it with a few rounds of power iteration. This page builds power iteration from a diagonal example, explains why it converges and how fast, and shows how to run a loop with a traced iteration count inside jit using jax.lax.fori_loop.

## What the spectral norm measures

Write $w$ for a matrix of shape $(M, N)$. It maps a vector $v$ of length $N$ to the vector $w v$ of length $M$. The stretch factor of a particular direction is $\lVert w v \rVert / \lVert v \rVert$, where $\lVert \cdot \rVert$ is the ordinary Euclidean length. The spectral norm $\sigma_1$ is the largest stretch factor over all nonzero $v$, and it equals the largest singular value of $w$.

For the diagonal matrix with entries 2 and 1, a vector along the first axis is stretched by 2, a vector along the second axis by 1, and every other direction by something in between. The spectral norm is 2, and the direction that achieves it is $[1, 0]$.

## Why repeated multiplication finds the top direction

Multiplying a vector by $w$ and then by its transpose $w^\top$ applies the square matrix $w^\top w$, which stretches each singular direction by the square of its singular value. Start the diagonal example from $[1, 1]$. One application of $w^\top w$ gives $[4, 1]$, a second gives $[16, 1]$, a third $[64, 1]$, and a fifth $[1024, 1]$. The first component is multiplied by $2^2 = 4$ every round while the second is multiplied by $1^2 = 1$, so the vector swings towards the top direction.

In general the share of the second direction shrinks by a factor of $(\sigma_2 / \sigma_1)^2$ per round, where $\sigma_2$ is the second-largest singular value. In the example that factor is $0.25$, which is why the direction settles quickly.

## Normalizing on every step

Without rescaling, the lengths above grow like $4^k$ after $k$ rounds, and with larger singular values or more rounds the numbers overflow float32. With singular values below 1 they shrink towards 0 instead. Power iteration therefore rescales to unit length after every multiplication, which keeps the direction and throws away the size. Writing $u$ for the unit vector on the output side, one round is

$$
u \leftarrow \frac{w v}{\lVert w v \rVert}, \qquad v \leftarrow \frac{w^\top u}{\lVert w^\top u \rVert}.
$$

The start vector is rescaled to unit length first. The direction it points in is the only thing that matters for the iteration, but its length would otherwise leak into an estimate taken after zero rounds.

## Reading off the estimate

After the last round, $v$ is a unit vector close to the top input direction, so the stretch factor of $v$ is an estimate of $\sigma_1$,

$$
\sigma \approx \lVert w v \rVert, \qquad u = \frac{w v}{\sigma},
$$

where $u$ is the matching unit output direction. The same number can be written as $u^\top w v$, which is the form used by spectral normalization. Because the stretch factor of any unit vector is at most $\sigma_1$, the estimate approaches the true value from below.

For the diagonal example started from $[1, 1]$, the unit start vector is about $[0.707107, 0.707107]$, and its stretch factor is about 1.581139. After one round, $v$ is about $[0.970143, 0.242536]$ and the estimate is about 1.955385. After two rounds it is about 1.997080, after three about 1.999817, and after four about 1.999989. Each round removes roughly three quarters of the remaining error in the direction.

## How the gap controls speed

A wide gap between the top two singular values makes power iteration fast, and a narrow gap makes it slow. For a diagonal matrix with entries 2 and 1.9, started from the same vector, the shrink factor per round is $(1.9 / 2)^2 = 0.9025$, and the estimate is only about 1.955744 after one round, 1.974103 after five and 1.988866 after ten. A fixed number of rounds therefore gives an estimate whose quality depends on the matrix, and the exact value after $k$ rounds depends on both the start vector and $k$. That is why the start vector is passed in instead of drawn at random, and why the iteration count is part of the definition of the result.

Spectral normalization in neural networks leans on the same fact from the other side. It runs a single round per training step and carries the vector $u$ over to the next step, so over many steps the estimate keeps improving while each step stays cheap.

## Loops with a traced count under jit

Inside a jitted function the iteration count may arrive as a traced int32 scalar instead of a Python integer, for example when it is a field of a configuration that the caller does not want to recompile for. A Python for loop over range of that count needs a real integer while the program is being traced, so it fails with a TracerIntegerConversionError.

jax.lax.fori_loop takes a lower bound, an upper bound, a body function and an initial value. The body receives the loop index and the current value, and returns the next value, which must keep the same shapes and dtypes. fori_loop accepts a traced upper bound, and what it compiles to depends on the bounds. With plain Python integers it becomes a scan with a fixed length, and with a traced bound it becomes a while loop, which checks the index against the bound at run time. A small doubling loop shows both cases. Tracing it with a Python count of 3 records one scan, while tracing it with a traced count records one while.

The two forms behave differently under differentiation. Reverse-mode gradients through a while loop are not supported, and JAX raises a ValueError saying that reverse-mode differentiation does not work for a fori_loop with dynamic start or stop values. With a fixed Python count the loop is a scan, and gradients flow through it. Estimating a spectral norm for monitoring or for normalization usually does not need gradients through the iterations, so a traced count is a reasonable choice here.

## Zero iterations

With a count of 0 the loop body never runs, and the estimate is the stretch factor of the unit start vector. That is a lower bound on the spectral norm and is often far from it, as the value 1.581139 in the diagonal example shows. It is still a well-defined result, and the loop has to handle it without special code.

## Common mistakes to avoid

- **Looping with Python's range on a traced count.** The count is a tracer under jit, so the loop cannot read it and tracing fails.
- **Skipping the normalization of the start vector.** The direction is unchanged, but with zero rounds the estimate is scaled by the start vector's length.
- **Running one round too many or too few.** Before convergence every round changes the estimate, so the count has to match exactly.
- **Letting the vectors grow without rescaling.** Repeated multiplication without unit rescaling overflows or underflows float32 after enough rounds.
- **Reading the estimate from the wrong side.** The length of $w^\top u$ is a different number from $\lVert w v \rVert$ until the iteration has converged.
- **Changing the loop value's shape inside the body.** fori_loop needs the value to keep the same shape and dtype on every round, so returning a vector of a different length fails to trace.