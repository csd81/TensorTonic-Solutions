# Karras Euler Sampler

Sampling from a diffusion model can be pictured as following a smooth path from pure noise down to clean data, where the sampler only gets to look at the path at a handful of points. Think of driving a winding road at night and checking the map only at a few stops, so the stops are best placed where the road bends most. Karras et al. (2022) showed that placing those points mostly near the clean end, and moving between them with a slightly smarter rule, gives high quality samples in a few dozen network calls. This page builds their sampler from a single coordinate, starting with one plain step and ending with the full noise schedule and the Euler and Heun rules you implement in this problem.

## Noise measured by one number

Earlier problems mixed signal and noise with weights whose squares add to 1. Karras et al. use a different but equivalent bookkeeping, where the clean data $y$ is kept at full size and noise is simply added on top.

$$
x = y + \sigma n
$$

Here $n$ is unit Gaussian noise and $\sigma$ is the noise level, a single number. At $\sigma = 0$ the sample is clean, and at a large $\sigma$ such as 80 the data is buried under noise. This view is called variance exploding, because the total variance grows with $\sigma$ instead of staying at 1. A model trained in the other bookkeeping can be rescaled into this one, so the sampler below applies to both.

## A path from noise to data

Song et al. (2021) showed that there is a deterministic path, the probability flow ODE, that carries a sample from a large noise level down to the data while matching the right noisy distribution at every $\sigma$ along the way. Written with a denoiser $D(x, \sigma)$ that estimates the clean sample, the path obeys the following slope.

$$
\frac{dx}{d\sigma} = \frac{x - D(x, \sigma)}{\sigma}
$$

The numerator $x - D(x, \sigma)$ is the part of $x$ that the denoiser believes is noise, and dividing by $\sigma$ turns it into the noise per unit of noise level. As $\sigma$ decreases, the sample moves against this slope and sheds noise. The sampler's job is to follow the path from $\sigma_{\max}$ down to 0 with as few slope evaluations as possible, because each one costs a network call.

## An exact denoiser for Gaussian data

A real sampler uses a trained network as $D$. To study the solver on its own, this problem uses data that is Gaussian with mean $\mu$ (data_mean) and standard deviation $s$ (data_std) in every coordinate. For that data the best possible denoiser, the expected clean value given $x$, has a closed form.

$$
D(x, \sigma) = \frac{s^2 x + \sigma^2 \mu}{s^2 + \sigma^2}
$$

At large $\sigma$ it returns almost $\mu$, because a very noisy sample says little about the data. At small $\sigma$ it returns almost $x$. With an exact denoiser, any error in the final samples comes only from the step schedule and the solver, which makes the effect of each choice easy to see.

## One Euler step

The simplest way to follow a path is to evaluate the slope where you stand and walk straight along it. With the current level $\sigma_i$ and the next level $\sigma_{i+1}$, the slope is $d_i = (x - D(x, \sigma_i)) / \sigma_i$ and the Euler proposal moves by $(\sigma_{i+1} - \sigma_i)\, d_i$. Because $\sigma$ decreases, that step size is negative.

Take one coordinate with $\mu = 0$ and $s = 1$, and walk from $\sigma = 3$ to $\sigma = 1$. Starting from a unit noise value of $0.5$, the sample at $\sigma = 3$ is $3 \cdot 0.5 = 1.5$, since the starting point must carry noise of size $\sigma_{\max}$.

1. The denoiser gives $D(1.5, 3) = 1.5 / 10 = 0.15$.
2. The slope is $(1.5 - 0.15) / 3 = 0.45$.
3. The Euler step lands at $1.5 + (1 - 3) \cdot 0.45 = 0.6$.

For this simple data the true path is known exactly, $x(\sigma) = 1.5 \sqrt{1 + \sigma^2} / \sqrt{10}$, which is about $0.67082$ at $\sigma = 1$. Euler fell short, because the slope changes along the way and a single evaluation at the start cannot see that.

## Heun's correction

Heun's method takes the Euler result as a proposal, evaluates the slope again at the proposal, and redoes the step with the average of the two slopes. In the example the proposal $0.6$ at $\sigma = 1$ gives $D = 0.3$ and a second slope of $0.3$. The average slope is $0.375$, so the corrected step lands at $1.5 - 2 \cdot 0.375 = 0.75$.

Heun now overshoots the true value $0.67082$ by about $0.079$, against Euler's undershoot of about $0.071$ on this single large step. The real gain shows over many small steps, because averaging the two slopes cancels the leading error term. Euler's error shrinks in proportion to the step size, while Heun's shrinks with its square, at the price of a second denoiser call per step. With ten levels from 3 down to 0.05 and $\rho = 7$, the same starting value ends about $0.043$ below the true end value with Euler and only about $0.007$ above it with Heun.

## The final step to zero

The last step goes from the smallest level to exactly 0, and the second slope there would divide by zero. Algorithm 1 of Karras et al. therefore takes a plain Euler step at the end, for both methods.

With an exact denoiser that final Euler step lands precisely on the denoiser output, since $x - \sigma\, (x - D) / \sigma = D$. In the example, the step from $\sigma = 1$ to 0 halves the sample, giving $0.3$ for Euler and $0.375$ for Heun against the true end value of about $0.47434$. The remaining gap comes from taking only two very large steps, which is where a good schedule helps.

## Why steps should cluster at low noise

The path changes slowly at high noise, where everything looks like noise, and quickly at low noise, where fine detail forms. Evenly spaced levels waste most of the budget at the noisy end. Karras et al. space the levels evenly in $\sigma^{1/\rho}$ instead, for $i = 0, \dots, N - 1$ with $N$ = num_steps.

$$
\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N - 1} \left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right) \right)^{\rho}
$$

Taking the $\rho$-th root compresses large values, the even spacing happens in that compressed scale, and raising back to the power $\rho$ stretches the gaps at the noisy end. With five levels from 20 down to 0.05, $\rho = 1$ gives evenly spaced levels of about 20, 15.01, 10.03, 5.04 and 0.05. With $\rho = 7$, the value the paper recommends, they become about 20, 6.747, 1.863, 0.385 and 0.05, so the last three levels all sit below 2, while with $\rho = 1$ only the last one does. After the last level, a final $\sigma_N = 0$ is appended so the sampler finishes on a clean sample.

## From one coordinate to a batch

In the problem, initial_noise has shape $(B, \dots)$ and holds unit noise, and the sampler starts from $\sigma_{\max}$ times that noise. The noise level is shared by every sample, and the denoiser acts on each coordinate independently, so each update is a scalar step applied to the whole tensor at once. The function returns both the schedule, which has $N + 1$ entries including the final zero, and the final samples.

## Numerical behaviour

Each slope divides by $\sigma_i$, which is tiny near $\sigma_{\min}$. The difference $x - D(x, \sigma)$ is also tiny there, so the ratio stays moderate, but rounding in single precision is amplified. A schedule that clusters levels near the clean end also avoids one huge jump from a large level to a tiny one, which keeps the error small.

## Common mistakes to avoid

- **Ordering the noise levels upward.** The path must run from sigma_max down to sigma_min, and running it upward adds noise instead of removing it.
- **Forgetting the appended zero.** Without it the sampler stops at sigma_min with residual noise and the schedule has the wrong length.
- **Applying the Heun correction on the last step.** The second slope divides by a zero noise level and turns the samples into infinities or NaN.
- **Starting from unit noise.** The first level is sigma_max, so the starting point must be the noise scaled by sigma_max to match that level.
- **Using only the second slope in the Heun step.** The method needs the average of both slopes, and one slope alone is a first-order method with a larger error.
- **Spacing the levels evenly in sigma.** That ignores rho and spends most steps at high noise where the path barely changes.