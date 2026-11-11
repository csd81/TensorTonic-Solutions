# DDIM Step

The original diffusion sampler walks back from noise one tiny step at a time, often a thousand of them, like going down a staircase one stair at a time when most stairs could safely be skipped. DDIM (Song et al., 2021) lets the same trained network take large jumps. At each jump it first estimates the clean sample, then places a new sample at the earlier noise level by mixing that estimate with exactly the right amount of noise. A knob called eta decides how much of that noise is freshly drawn and how much is reused from the current sample. This page builds one jump from a single number and ends with the batch update you implement in this problem.

## Why big jumps are allowed

The DDPM sampler treats generation as the exact reverse of a chain of small steps, so it has to visit every training timestep. Song et al. noticed that the DDPM training loss only ever uses the closed form marginals, the distribution of $x_t$ given $x_0$ at each single timestep. It never looks at how consecutive timesteps are linked.

Many different reverse processes share those same marginals, and some of them can move between any two timesteps directly. A network trained with the ordinary noise prediction loss works for all of them, so no retraining is needed. Picking 50 timesteps out of 1000 then gives a sampler that is roughly twenty times faster.

## A jump in two moves

Recall the forward formula with the cumulative fraction $\bar\alpha_t$ (alpha_bars at index t).

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

The jump goes from timestep t to an earlier timestep t_prev, whose cumulative fraction is $\bar\alpha_{\text{prev}}$. The first move inverts the formula with the network's noise guess $\hat\epsilon$ (predicted_noise) to get a clean estimate.

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \hat\epsilon}{\sqrt{\bar\alpha_t}}
$$

The second move builds a sample that looks like it came from the forward formula at t_prev. It needs signal $\sqrt{\bar\alpha_{\text{prev}}}\, \hat x_0$ and noise with total variance $1 - \bar\alpha_{\text{prev}}$.

The simplest choice of noise is the predicted noise itself, scaled to the new level. Take one number with $\bar\alpha_t = 0.0784$ and $\bar\alpha_{\text{prev}} = 0.64$, so the square roots are $0.28$ and $0.96$ at t, and $0.8$ and $0.6$ at t_prev. With $x_t = 0.4$ and $\hat\epsilon = 0.3$, the clean estimate is $(0.4 - 0.96 \cdot 0.3) / 0.28 = 0.4$. The new sample is $0.8 \cdot 0.4 + 0.6 \cdot 0.3 = 0.5$.

That jump used no randomness at all. Reusing the predicted noise keeps the new sample consistent with the sample it came from, which is what makes large jumps safe.

## Splitting the noise budget

The noise variance $1 - \bar\alpha_{\text{prev}}$ is a budget that can be split between two sources. A part $\sigma^2$ comes from fresh standard Gaussian noise $z$ (noise), and the rest reuses the predicted noise direction.

$$
x_{\text{prev}} = \sqrt{\bar\alpha_{\text{prev}}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{\text{prev}} - \sigma^2}\, \hat\epsilon + \sigma z
$$

The two noise terms are independent with unit variance each, so their variances add up to $(1 - \bar\alpha_{\text{prev}} - \sigma^2) + \sigma^2$, which is the full budget. Whatever $\sigma$ is, the new sample has the right total noise level for t_prev. In the deterministic jump above, $\sigma = 0$ and the whole budget went to the predicted noise.

## What eta controls

The amount of fresh noise is set by eta, written $\eta$.

$$
\sigma = \eta \sqrt{\frac{1 - \bar\alpha_{\text{prev}}}{1 - \bar\alpha_t}} \sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{\text{prev}}}}
$$

The second square root alone is the standard deviation of a forward jump from t_prev to t. The first factor shrinks it to the posterior value, the spread that remains when $x_t$ is also known. With $\eta = 1$ and adjacent timesteps, $\sigma^2$ equals the DDPM posterior variance $\tilde\beta_t$, so DDIM reproduces ancestral DDPM sampling. With $\eta = 0$ the sampler is fully deterministic, and the same starting noise always gives the same output. Values in between scale the fresh noise smoothly.

Continue the example with $z = -1$.

1. The first factor is $\sqrt{0.36 / 0.9216} = 0.625$ and the second is $\sqrt{1 - 0.1225} \approx 0.93675$.
2. With $\eta = 1$, $\sigma \approx 0.58547$ and $\sigma^2 \approx 0.34277$, so the direction coefficient is $\sqrt{0.36 - 0.34277} \approx 0.13125$.
3. The new sample is $0.32 + 0.13125 \cdot 0.3 - 0.58547 \approx -0.22609$.
4. With $\eta = 0.5$, $\sigma \approx 0.29273$, the direction coefficient is about $0.52374$, and the new sample is about $0.18439$.

Most of the budget moves from the predicted direction to fresh noise as eta grows, and the sample moves away from the deterministic value $0.5$.

## Landing on the clean sample

The final jump of a sampler goes to the clean end of the chain. Timesteps are 0-based, so this is written as t_prev = -1, and the plan defines $\bar\alpha_{-1} = 1$. Then $1 - \bar\alpha_{\text{prev}} = 0$, so $\sigma = 0$ for any eta, the direction coefficient is 0, and the update returns $\hat x_0$ itself.

A negative index in Python silently reads the last entry of alpha_bars, which is the noisiest timestep. A sampler with that bug ends by adding a large amount of noise to its output.

## Clipping the clean estimate

Image models are trained on data scaled to $[-1, 1]$, but at high noise levels $\hat x_0$ can fall far outside that range. Clamping it to $[-1, 1]$ keeps samples in range and was used in the original DDPM and DDIM code.

After clamping, the old noise guess no longer matches. The pair of clean estimate and noise should still satisfy the forward formula at $x_t$, so the noise is recomputed from $x_t$ and the clamped estimate. Without that step the update mixes a clamped signal with a noise direction that belongs to the unclamped one.

Take $x_t = 0.6$ and $\hat\epsilon = 0.2$ at the same two timesteps, with $\eta = 0$.

1. The clean estimate is $(0.6 - 0.96 \cdot 0.2) / 0.28 \approx 1.45714$, and without clipping the update gives about $1.28571$.
2. Clamping gives $\hat x_0 = 1$, and the recomputed noise is $(0.6 - 0.28) / 0.96 \approx 0.33333$.
3. The clipped update is $0.8 + 0.6 \cdot 0.33333 = 1.0$, while keeping the old noise would give $0.92$.

## From one number to a batch

In the problem, x_t, predicted_noise and noise have shape $(B, \dots)$, for example $(B, C, H, W)$. One integer t and one integer t_prev apply to every sample, so every coefficient is a plain scalar. The function returns the new sample along with the clean estimate it used, which samplers often display as a preview of the final image.

## Where this step sits in the pipeline

A DDIM sampler chooses a decreasing list of timesteps, starts from pure noise, and calls the network once per entry. Each call is followed by this update with t_prev set to the next entry in the list, and the final call uses t_prev = -1. Because the deterministic version maps noise to images with a fixed rule, it can also be run in reverse to encode a real image into noise.

## Common mistakes to avoid

- **Indexing alpha_bars at t_prev = -1.** Python wraps to the noisiest entry, so the final step adds noise to the output instead of returning the clean estimate.
- **Keeping the old noise after clipping.** The update then combines a clamped signal with a noise direction from the unclamped estimate, which breaks the forward relation at x_t.
- **Dropping the first factor of sigma.** The remaining factor is the forward jump spread, which is larger than the posterior value and makes eta = 1 noisier than DDPM.
- **Ignoring sigma in the direction term.** The reused direction must give up exactly sigma squared of variance, otherwise the total noise level of the new sample is too high.
- **Skipping the clamp under the direction square root.** Rounding can push the term slightly below 0 when sigma uses almost the whole budget, and the root then returns NaN.
- **Treating eta as an on or off switch.** Eta scales sigma continuously, so values between 0 and 1 must give intermediate amounts of fresh noise.