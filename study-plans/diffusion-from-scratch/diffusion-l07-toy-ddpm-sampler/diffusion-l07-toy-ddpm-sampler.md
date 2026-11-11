# Toy DDPM Sampler

Sampling is the moment a diffusion model creates something new. It starts from pure static and cleans it up a little at a time, like slowly bringing a blurry photo into focus. At every step the trained network guesses which part of the current sample is noise, and the sampler removes a measured share of it while adding back a touch of fresh randomness. This page builds the reverse loop one piece at a time, from the schedule and a single backward step on one number up to the full loop over every timestep, which is the capstone sampler you implement in this problem. Most of the work lies in doing each step in the right order with the right index.

## Building the schedule from the betas

The forward process adds a little Gaussian noise at each of $T$ timesteps, numbered $t = 0, 1, \dots, T - 1$. Step $t$ has a small variance $\beta_t$, given by betas[t], and keeps the fraction $\alpha_t = 1 - \beta_t$ of the variance it receives. The fraction of the clean data's variance that survives up to and including step $t$ is the cumulative product

$$
\bar\alpha_t = \alpha_0\, \alpha_1 \cdots \alpha_t .
$$

Take $T = 2$ with betas $(0.2, 0.5)$. Then $\alpha_0 = 0.8$ and $\alpha_1 = 0.5$, so $\bar\alpha_0 = 0.8$ and $\bar\alpha_1 = 0.4$. The reverse loop also needs the value before the first step, and the convention is $\bar\alpha_{-1} = 1$, since nothing has been lost before any noise is added.

With these values the forward closed form jumps straight to any timestep, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, where $\epsilon$ is standard Gaussian noise. The sampler runs this process backwards.

## Guessing the clean sample from a noise prediction

The network looks at the current sample $x_t$ and its timestep and predicts the noise $\hat\epsilon$ it contains. If that prediction were perfect, the forward closed form could be solved for the clean sample. Using the prediction in its place gives the predicted clean sample

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \hat\epsilon}{\sqrt{\bar\alpha_t}} .
$$

With $\bar\alpha_1 = 0.4$, $x_1 = 1$ and $\hat\epsilon = 0.5$, the guess is $(1 - \sqrt{0.6} \cdot 0.5) / \sqrt{0.4} \approx 0.968766$. Jumping straight to this guess would be a poor sampler, because the guess from a very noisy sample is blurry. The reverse step uses it more carefully.

## The exact backward step when the clean sample is known

The distribution of $x_{t-1}$ given only $x_t$ is intractable. If the clean sample were also known, though, the posterior $q(x_{t-1} \mid x_t, x_0)$ would be an exact Gaussian (Ho et al., 2020). The sampler plugs in $\hat x_0$ for the unknown clean sample. The posterior mean is a weighted blend of the guess and the current sample,

$$
\mu = \frac{\beta_t \sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_t}\, \hat x_0 + \frac{(1 - \bar\alpha_{t-1}) \sqrt{\alpha_t}}{1 - \bar\alpha_t}\, x_t ,
$$

and its variance depends only on the schedule,

$$
\tilde\beta_t = \frac{\beta_t (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} .
$$

The next sample is $x_{t-1} = \mu + \sqrt{\tilde\beta_t}\, z$, where $z$ is fresh standard Gaussian noise. Early in the loop, $\bar\alpha_t$ is small and the weight on $x_t$ dominates, so the sample moves only a little towards the guess. Near the end the weight on $\hat x_0$ takes over.

Continuing the example at $t = 1$, the two weights are about $0.745356$ for $\hat x_0$ and $0.235702$ for $x_1$, so $\mu \approx 0.957778$. The variance is $0.5 \cdot 0.2 / 0.6 \approx 0.166667$, whose square root is about $0.408248$.

## The last step adds no noise

At $t = 0$ the convention $\bar\alpha_{-1} = 1$ makes $1 - \bar\alpha_{-1} = 0$. The variance $\tilde\beta_0$ is then 0, the weight on $x_0$ inside $\mu$ is 0, and the weight on $\hat x_0$ is $\beta_0 / (1 - \alpha_0) = 1$. The final step therefore returns exactly the predicted clean sample, and adding noise there would only corrupt the output.

Ho et al. noted that using $\beta_t$ itself as the variance gives similar image quality, but that is a different sampler, and its variance does not vanish at $t = 0$. This problem uses the posterior variance $\tilde\beta_t$.

## Keeping the guess in range

Early in the loop $\bar\alpha_t$ is small, so dividing by $\sqrt{\bar\alpha_t}$ amplifies any error in $\hat\epsilon$. With $\bar\alpha_t = 0.04$, an error of $0.1$ in the predicted noise moves $\hat x_0$ by about $0.49$. A guess from $x_t = 0.5$ and $\hat\epsilon = -0.6$ lands at about $5.44$, far outside the range of data scaled to $[-1, 1]$, and the posterior mean then drags the sample towards an impossible value.

When the data is known to lie in a bounded range, clamping $\hat x_0$ to $[-c, c]$ before computing $\mu$ removes the overshoot, where $c$ is clip_value. This is the clip_denoised option of the original DDPM code. Clipping is a modelling choice, so data without a known range should not be clamped, which is why the threshold is optional. Clamping the sample $x_{t-1}$ instead of $\hat x_0$ is a different operation that fights the injected noise and leaves the overshooting guess inside $\mu$.

## Walking the timesteps and the noise rows

The loop starts from x_T, the sample at timestep $T - 1$, of shape $(N, D)$. It visits $t = T - 1, T - 2, \dots, 0$, so iteration $k$ handles timestep $t = T - 1 - k$. Every random draw is supplied in advance as noise with shape $(T, N, D)$, and iteration $k$ reads row $k$. Indexing the noise by $t$ instead would walk the rows backwards, which gives a valid looking but different chain. The last row is never used, because the final step adds no noise.

The loop also records the possibly clamped $\hat x_0$ of every iteration, in the same order, which shows how the guess sharpens from blurry to final.

## The network and its timestep input

The noise prediction uses the same tiny network as the training step in this plan.

$$
\hat\epsilon = \operatorname{SiLU}\big( [\, x_t,\ e(t) \,]\, W_1 + b_1 \big)\, W_2 + b_2
$$

The weights come from params, and $\operatorname{SiLU}(z) = z \cdot \operatorname{sigmoid}(z)$. The embedding $e(t)$ is the sinusoidal one, with half equal to embed_dim // 2, frequencies $f_i = \exp(-\ln(10000)\, i / \text{half})$, the cosine half first and the sine half second, and one zero appended for an odd embed_dim. All N samples sit at the same timestep, so they share one embedding row.

## Two reverse steps by hand

Put the pieces together for $T = 2$ with betas $(0.2, 0.5)$ and one number. Suppose the network predicts $\hat\epsilon = 0.5$ at $t = 1$ and $\hat\epsilon = -0.2$ at $t = 0$, the starting value is $x_1 = 1$, and the first noise row holds $0.3$. No clipping is used.

1. At $t = 1$ the guess is $\hat x_0 \approx 0.968766$ and the posterior mean is $\mu \approx 0.957778$.
2. The noise term is $0.408248 \cdot 0.3 \approx 0.122474$, so the sample at timestep 0 is about $1.080253$.
3. At $t = 0$ the guess is $(1.080253 + \sqrt{0.2} \cdot 0.2) / \sqrt{0.8} \approx 1.307759$.
4. The final step returns this guess unchanged, so the generated sample is about $1.307759$.

The recorded guesses are about $0.968766$ and $1.307759$, and the second noise row is never read.

## Numerical behaviour

The coefficients divide by $1 - \bar\alpha_t$, which is small only when the early betas are tiny. In float32, computing $1 - \bar\alpha_t$ for $\bar\alpha_t$ close to 1 loses digits, so practical samplers either keep moderate betas, as toy schedules do, or precompute the schedule in float64. The division by $\sqrt{\bar\alpha_t}$ is the other sensitive spot, and clipping is the usual guard against it.

## Where this sits in the diffusion pipeline

This loop is the generation stage of DDPM. The network trained with the noise prediction step is frozen, and the sampler turns noise into data in $T$ network calls. Faster samplers such as DDIM reuse the same predicted clean sample and change only how $x_{t-1}$ is formed, and evaluation metrics such as FID are computed on the samples this loop produces.

## Common mistakes to avoid

- **Indexing noise by t instead of by the step counter.** The rows are consumed in the wrong order, which produces a different chain from the same random draws.
- **Adding noise at t = 0.** The final step should return the posterior mean, and extra noise blurs the generated sample.
- **Clipping the predicted clean sample when no clip value is given.** A default range like $[-1, 1]$ silently squashes data that legitimately lies outside it.
- **Using betas[t] as the variance.** That is the larger fixed variance choice, which differs from the posterior variance at every step and stays nonzero at t = 0.
- **Using alphas[t] where alpha_bars[t] belongs.** The predicted clean sample must invert the cumulative forward process, and a single step's alpha describes only one step.
- **Clamping the sample instead of the predicted clean sample.** The clipping then fights the injected noise and leaves the overshooting guess in the posterior mean.
- **Forgetting that alpha_bars at index minus one is 1.** Wrapping around to the last schedule entry at t = 0 gives a nonzero variance and the wrong final sample.