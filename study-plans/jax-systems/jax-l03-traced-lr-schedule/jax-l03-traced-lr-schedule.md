# Traced Learning Rate Schedule

Almost every large model is trained with a learning rate that changes over time, rising quickly at the start and then falling slowly to a small final value. In JAX the whole training step is usually compiled with jax.jit, and the current step count is one of its traced inputs, so the schedule has to be computed from a traced number. That rules out the most natural way to write a schedule, which is a Python if statement that checks which phase the step is in. This page builds the warmup and cosine schedule from small numbers, shows why the if statement fails under jit, and explains how to compute both phases and select between them.

## The schedule in small numbers

Picture a short run with a peak learning rate of 0.2, a final learning rate of 0.02, a warmup of 4 steps and 12 steps in total. During warmup the learning rate climbs in a straight line from 0, so steps 0, 1, 2 and 3 use 0, 0.05, 0.1 and 0.15. Step 4 is the first step after warmup, and it uses the full peak of 0.2.

From there the learning rate follows half of a cosine wave down to the final value. Step 6 uses about 0.17364, step 8 sits exactly halfway at 0.11, step 10 uses about 0.04636, and step 12 reaches 0.02. Any step beyond 12 stays at 0.02. The curve falls slowly at first, fastest in the middle, and slowly again as it approaches the floor.

## Why training warms up

At the very start of training the weights are random and the optimizer has almost no history. Adaptive optimizers such as Adam divide each update by a running estimate of the gradient's size, and after only a few steps that estimate is noisy, so a full-size learning rate can produce a few huge updates that push the model into a bad region. A linear warmup keeps the first updates small while those estimates settle, then hands over to the full rate.

## Why the decay follows a cosine

Late in training, smaller steps let the weights settle into a low point of the loss instead of bouncing around it. A cosine curve gives a smooth path from the peak to the floor with no sudden jumps. It spends a while near the peak, where learning is fastest, and a while near the floor, where fine adjustment happens.

## The formula

Write $s$ for the step, $W$ for warmup_steps, $S$ for total_steps, $\eta_{\max}$ for peak_lr and $\eta_{\min}$ for min_lr. During warmup, which covers $s < W$, the learning rate is

$$
\eta(s) = \eta_{\max} \cdot \frac{s}{W}.
$$

For $s \ge W$, define the fraction of the decay phase that has passed, capped at 1,

$$
p = \min\left(\frac{s - W}{S - W},\ 1\right), \qquad \eta(s) = \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos(\pi p)\right).
$$

At $p = 0$ the cosine equals 1, so the rate is $\eta_{\max}$, which matches the end of warmup. At $p = 1$ the cosine equals $-1$, so the rate is $\eta_{\min}$. The cap on $p$ keeps every later step at the floor.

## Why a Python if fails under jit

When jax.jit traces the training step, the step count arrives as a tracer, a placeholder that knows it is an int32 array of some shape but does not know its value. A Python if statement needs a real True or False right now, while the program is still being built. JAX cannot give one, so it raises a TracerBoolConversionError the moment the comparison is used as a condition. A tiny jitted function that returns 1.0 if its input is below 4 and 2.0 otherwise fails in exactly this way.

The same rule covers other ways of pulling a Python value out of a tracer. Calling float or int on the step, or using it in Python's min or max, needs a concrete number, and fails in the same way under jit.

## Computing both phases and selecting

The way out is to stop choosing a branch and compute both. You compute the warmup value and the cosine value for the step, then pick the right one with jnp.where, which takes a boolean array and two candidate arrays and returns, element by element, the first candidate where the condition is true and the second where it is false. The condition itself, the comparison of the step with the warmup length, is an ordinary traced array of booleans, so nothing needs to be known in Python.

For example, jnp.where applied to the condition that steps 0, 4 and 8 are below 4, with candidates 1.0 and 2.0, returns $[1, 2, 2]$. Because the selection happens element by element, the same code works for a single step and for a whole vector of steps. That is useful for plotting a schedule, and for logging the learning rate of many steps at once.

Computing both branches does mean that the unselected branch is evaluated too. For a schedule this costs almost nothing, but it matters for safety. A branch that divides by zero or takes the cosine of a huge number still runs, and its result is simply discarded, so each branch should produce finite values for every input it might see. Clamping $p$ into the range from 0 to 1 with jnp.clip handles both the cap and steps that are still in warmup, where $s - W$ is negative.

## Where lax.cond fits

jax.lax.cond is the other traced conditional. It takes a single boolean and two functions, and the compiled program runs only the chosen function. Its condition must be a scalar. Given the boolean vector $[\text{True}, \text{False}]$, it raises a TypeError saying the predicate must be a scalar, so it cannot handle a vector of steps on its own. If you map a lax.cond over a vector with jax.vmap, JAX converts it into an elementwise select, the same operation that jnp.where produces. lax.cond pays off when the branches are expensive and only one should run, which is never the case for a learning rate.

## What the cap on progress does

Without the cap, steps beyond the end keep moving along the cosine wave. In the small example, step 16 would have a progress of 1.5 and a learning rate of 0.11 again, and step 20 would have a progress of 2 and climb all the way back to the peak of 0.2. A run that trains a little longer than planned, or a schedule evaluated past its end for a plot, would then see the learning rate shoot back up. The cap keeps it pinned at the floor.

## Where the schedule runs in training

In a typical JAX training loop, the optimizer state holds an integer step counter, and the jitted update function reads it, computes the learning rate from the schedule, scales the update, and increments the counter. The schedule's settings (the peak, the floor and the two lengths) stay fixed for the whole run, so they can be ordinary Python numbers that the compiler folds into the program as constants. Only the step changes from call to call, and because it stays a traced value the update compiles exactly once.

## Common mistakes to avoid

- **Branching with a Python if on the step.** The step is a tracer under jit, so the comparison cannot become a Python boolean and tracing stops with an error.
- **Using lax.cond for a vector of steps.** Its condition must be a single boolean, so a vector of steps raises a TypeError.
- **Starting warmup one step late.** Using $(s + 1) / W$ shifts the whole ramp, so step 0 no longer starts at a learning rate of 0.
- **Forgetting to cap the progress.** Steps past total_steps then climb back up the cosine wave instead of staying at the floor.
- **Measuring progress from step 0.** The decay phase starts at step $W$, so dividing $s$ by $S$ instead of $s - W$ by $S - W$ makes the curve start below the peak.
- **Dropping the floor.** The cosine moves between $\eta_{\max}$ and $\eta_{\min}$, so leaving out $\eta_{\min}$ sends the final rate to 0.