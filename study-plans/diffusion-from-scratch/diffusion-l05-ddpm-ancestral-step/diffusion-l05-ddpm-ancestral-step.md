# DDPM Ancestral Step

Generating with a diffusion model works a little like restoring an old photo by hand, removing a thin layer of grime at a time until a clean picture appears. Ancestral sampling, the original sampler of Ho et al. (2020), starts from pure noise and takes many small backward steps. Each step asks the network how much noise is still in the sample, uses that guess to estimate where the sample stood one step earlier, and then adds back a small amount of fresh randomness. This page builds one such step, starting from a single number and ending with the batch update you implement in this problem.

## What one backward step needs

The forward process adds a little Gaussian noise at every step. With $\beta_t$ the variance added by step $t$ (betas at index $t$) and $\alpha_t = 1 - \beta_t$ the fraction that step keeps, one forward step is

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t .
$$

When each $\beta_t$ is small, the reverse of one step is also close to a Gaussian. So a backward step needs only two things, a centre for the likely values of $x_{t-1}$ and a spread around that centre. The sampler moves to the centre and then draws a random offset with the right spread.

The difficulty is the centre. The exact reverse distribution depends on the whole data distribution, which is unknown. Once the clean sample $x_0$ is known, though, the reverse step has an exact Gaussian answer, the forward posterior $q(x_{t-1} \mid x_t, x_0)$ from the previous section. The network cannot know $x_0$, but it can estimate it, and that estimate is what the predicted noise provides.

## From predicted noise to a clean estimate

Recall the closed form forward formula, with $\bar\alpha_t$ the product of $\alpha_0$ through $\alpha_t$.

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

The network returns a guess $\hat\epsilon$ (predicted_noise) of the noise inside $x_t$. Plugging that guess into the formula and solving for the clean part gives an estimate of $x_0$. For one number, take a schedule with betas $= (0.04, 0.1)$, so the alphas are $(0.96, 0.9)$ and the cumulative values are $\bar\alpha_0 = 0.96$ and $\bar\alpha_1 = 0.864$. At $t = 1$, with $x_t = 0.7$ and $\hat\epsilon = -0.4$, the clean estimate is $(0.7 + \sqrt{0.136} \cdot 0.4) / \sqrt{0.864} \approx 0.9118$.

## The centre of the step

The posterior mean from the previous section mixes the clean estimate and the current sample. In the example its two coefficients are about $0.7204$ on the clean estimate and $0.2790$ on $x_t$, so the centre is about $0.7204 \cdot 0.9118 + 0.2790 \cdot 0.7 \approx 0.8522$.

Doing that in two stages works, but the clean estimate can be substituted into the posterior mean once and simplified. The derivation runs as follows.

1. The posterior mean is a combination $c_0 x_0 + c_t x_t$, where $c_0$ uses $\beta_t$ and $\sqrt{\bar\alpha_{t-1}}$, and $c_t$ uses $1 - \bar\alpha_{t-1}$ and $\sqrt{\alpha_t}$.
2. Replacing $x_0$ with the estimate turns the mean into one term in $x_t$ and one term in $\hat\epsilon$.
3. The coefficient on $x_t$ simplifies to $1 / \sqrt{\alpha_t}$, using $\bar\alpha_t = \alpha_t \bar\alpha_{t-1}$ and $\beta_t + \alpha_t = 1$.
4. The coefficient on $\hat\epsilon$ simplifies to $-\beta_t / (\sqrt{\alpha_t} \sqrt{1 - \bar\alpha_t})$.

Together these give the form that Ho et al. wrote in their sampling algorithm.

$$
\mu = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\epsilon \right)
$$

Read it from the inside out. The bracket removes a scaled portion of the predicted noise from $x_t$, and the division by $\sqrt{\alpha_t}$ undoes the shrink that one forward step applied. In the example the noise coefficient is $0.1 / \sqrt{0.136} \approx 0.2712$, the bracket is $0.7 + 0.2712 \cdot 0.4 \approx 0.8085$, and dividing by $\sqrt{0.9} \approx 0.9487$ gives $\mu \approx 0.8522$, the same centre as the two-stage route.

Two different alphas appear in the formula, and they do different jobs. The single-step $\alpha_t$ undoes one step of shrinking, while the cumulative $\bar\alpha_t$ measures how much total noise $x_t$ carries.

## Choosing the spread

The DDPM paper fixes the reverse variance instead of learning it, and offers two natural choices. They sit at the two ends of a reasonable range, and variance_type picks one of them.

- **The fixed_large choice.** The variance is $\beta_t$, the variance that one forward step added.
- **The fixed_small choice.** The variance is the exact posterior variance $\tilde\beta_t = \beta_t (1 - \bar\alpha_{t-1}) / (1 - \bar\alpha_t)$, which is right when $x_0$ is known.

Since $\bar\alpha_{t-1}$ is larger than $\bar\alpha_t$, the ratio inside $\tilde\beta_t$ is below 1, so fixed_small always adds less noise. Ho et al. reported similar sample quality for both on images, and Nichol and Dhariwal (2021) later learned an interpolation between them, which is why libraries keep both options.

The new sample is the centre plus a random offset, $x_{t-1} = \mu + \sigma_t z$, where $z$ (noise) is fresh standard Gaussian noise. Both choices are variances, so each one passes through a square root to become the standard deviation $\sigma_t$ before it scales $z$.

## Finishing the example

Continue at $t = 1$ with $\mu \approx 0.8522$ and a fresh draw $z = 0.5$.

1. For fixed_small, the variance is $0.1 \cdot 0.04 / 0.136 \approx 0.02941$, its square root is about $0.1715$, and $x_{t-1} \approx 0.8522 + 0.0857 = 0.9379$.
2. For fixed_large, the standard deviation is $\sqrt{0.1} \approx 0.3162$, and $x_{t-1} \approx 0.8522 + 0.1581 = 1.0103$.

The same centre leads to noticeably different samples, so the choice of spread matters at every step of a long chain.

## Why the last step adds no noise

At $t = 0$ the result is the final output. Any noise added there would land directly in the generated sample as grain that no later step could remove, so the sampling algorithm of Ho et al. returns the centre alone at $t = 0$.

The posterior variance agrees with this rule. Timesteps are 0-based and the plan defines $\bar\alpha_{-1} = 1$, so $1 - \bar\alpha_{-1} = 0$ and $\tilde\beta_0 = 0$. The fixed_large choice gives $\beta_0 > 0$, though, so the sampler has to skip the noise there explicitly.

The centre stays well behaved at this boundary. Since $1 - \bar\alpha_0 = \beta_0$, the noise coefficient at $t = 0$ equals $\beta_0 / \sqrt{\beta_0} = \sqrt{\beta_0}$, which is $0.2$ for the example schedule. At every later step it is smaller than $\sqrt{\beta_t}$.

## From one number to a batch

In the problem, x_t, predicted_noise and noise all have shape $(B, \dots)$, for example $(B, C, H, W)$. The step uses one integer t for the whole batch, so $\beta_t$, $\alpha_t$, $\bar\alpha_t$ and $\sigma_t$ are plain scalars that scale every element alike. The cumulative values are built from betas inside the function, and the update leaves x_t unchanged.

## Where this step sits in the sampler

A full DDPM sampler sets $x_{T-1}$ to pure noise and applies this step for $t = T - 1$ down to $0$, calling the network once per step. With a typical $T = 1000$ that is a thousand network evaluations per sample, which motivates the faster samplers later in this section.

## Common mistakes to avoid

- **Adding noise at t = 0.** The last step is the output itself, so any noise there stays in the final sample as visible grain.
- **Swapping the two variance options.** Using $\beta_t$ for fixed_small or the posterior variance for fixed_large gives the wrong amount of randomness at every step.
- **Dividing by the square root of the cumulative alpha.** The centre is scaled by $1 / \sqrt{\alpha_t}$ for one step, and using $\sqrt{\bar\alpha_t}$ inflates the sample at late steps.
- **Multiplying the noise by the variance.** The noise scale is the standard deviation, so the variance has to pass through a square root first.
- **Using the square root of beta as the noise coefficient.** The predicted noise is weighted by $\beta_t / \sqrt{1 - \bar\alpha_t}$, which equals $\sqrt{\beta_t}$ only at $t = 0$.
- **Reading the previous cumulative alpha at index t minus 1 when t is 0.** Negative indexing wraps to the last entry, so the value before the first step must be taken as 1.