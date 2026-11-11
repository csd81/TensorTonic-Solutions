# Min-SNR Loss Weighting

When a diffusion model trains, every example in a batch is noised to a different level, from almost clean to almost pure static. Averaging their losses with equal weight sounds fair, yet it lets some noise levels pull far harder on the network than others, a bit like a course where the easiest homework counts the most towards the final grade. Min-SNR weighting gives each example a weight based on how much signal it still contains and caps the weight of the easiest ones, which helps training converge faster. This page builds that weight from a single noise level, shows how it changes with the network's target, and ends with the weighted batch loss you implement in this problem.

## Measuring how much signal is left

Recall the forward formula, where the clean data keeps the fraction $\bar\alpha_t$ of the variance and the noise supplies the rest.

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

The signal to noise ratio of a timestep compares those two variances directly.

$$
\mathrm{snr}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}
$$

Three values give a feel for it. With $\bar\alpha_t = 0.96$ the ratio is $0.96 / 0.04 = 24$, and the sample is almost clean. With $\bar\alpha_t = 0.6$ it is $1.5$, close to the point where signal and noise are equal. With $\bar\alpha_t = 0.15$ it is about $0.1765$, and the sample is mostly noise. The ratio falls from very large values at the first timesteps towards 0 at the end of the schedule, so one number summarizes the whole noise level.

## Why equal weights are not neutral

The plain training loss draws a timestep for every sample and averages the squared errors with equal weight. Hang et al. (2023) argued that this treats training as many tasks at once, one per noise level, and that the tasks pull the network in conflicting directions. Some weightings let the nearly clean timesteps dominate the gradient even though those tasks are easy, while others spend capacity on very noisy timesteps where little can be learned.

The trouble is that a squared error means different things at different noise levels. The cleanest way to see this is to measure every error in the same units, the units of the clean sample, because a clean sample estimate is what a sampler finally needs.

## Errors in clean sample units

Suppose the network predicts noise, and its guess is off by $\delta$ at one element. Solving the forward formula for the clean value shows that a noise error of $\delta$ produces a clean value error of $\delta \sqrt{1 - \bar\alpha_t} / \sqrt{\bar\alpha_t}$. Squaring both sides gives a clean relation between the two squared errors.

$$
\lVert \epsilon - \hat\epsilon \rVert^2 = \mathrm{snr}(t)\, \lVert x_0 - \hat x_0 \rVert^2
$$

Try it at $\bar\alpha_t = 0.96$, where the weights are about $0.9798$ and $0.2$. A noise error of $0.5$ becomes a clean value error of $0.5 \cdot 0.2 / 0.9798 \approx 0.1021$. The squared errors are $0.25$ and about $0.01042$, and their ratio is exactly 24, the signal to noise ratio.

So the plain noise loss secretly weights each clean sample error by $\mathrm{snr}(t)$. At almost clean timesteps that hidden weight is huge, which is exactly the imbalance Hang et al. observed.

## Capping the weight

Min-SNR fixes the imbalance in clean sample units first. For a network that predicts the clean sample, the weight is the ratio itself, capped from above by a constant $\gamma$ (gamma).

$$
w_{x_0}(t) = \min\left(\mathrm{snr}(t), \gamma\right)
$$

Below the cap, this weight equals the hidden weight of the plain noise loss, so noisy timesteps are trained exactly as before. Above the cap, the weight stops growing, so nearly clean timesteps can no longer dominate. Hang et al. found $\gamma = 5$ to work well and reported reaching the same image quality several times faster than with the plain loss.

The cap acts only from above. Values of the ratio below $\gamma$ pass through unchanged, and raising them up to $\gamma$ would amplify the noisiest timesteps, which is the opposite of the intended effect.

## Moving the weight to other targets

Networks that predict noise or velocity produce errors in other units, so the clean sample weight has to be translated. The relation above says a squared noise error is $\mathrm{snr}(t)$ times a squared clean error, so to apply the clean sample weight to a noise loss you divide it by $\mathrm{snr}(t)$. The result is 1 wherever the ratio is below $\gamma$ and falls below 1 only for nearly clean timesteps.

The velocity $v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1 - \bar\alpha_t}\, x_0$ needs the same substitution once more. Its squared error turns out to be $\mathrm{snr}(t) + 1$ times the squared clean error, so the velocity weight divides the capped ratio by $\mathrm{snr}(t) + 1$. That weight is always below 1 and stays finite even as the ratio approaches 0. The problem statement collects the three rows, and prediction_type selects one of them.

## A three-sample example

Take $\gamma = 4$ and three samples whose timesteps give $\bar\alpha = 0.96$, $0.6$ and $0.15$, so the ratios are $24$, $1.5$ and about $0.1765$. Only the first ratio exceeds the cap, so the capped values are $4$, $1.5$ and $0.1765$. Let the unweighted per-sample losses be $0.12$, $0.45$ and $0.7$.

1. For a noise target the weights are $4 / 24 \approx 0.1667$, then 1 and 1. The weighted losses are $0.02$, $0.45$ and $0.7$, and their mean is $0.39$.
2. For a clean sample target the weights are $4$, $1.5$ and $0.1765$. The weighted losses are $0.48$, $0.675$ and about $0.1235$, and their mean is about $0.4262$.
3. For a velocity target the weights are $4 / 25 = 0.16$, $1.5 / 2.5 = 0.6$ and $0.1765 / 1.1765 = 0.15$. The weighted losses are $0.0192$, $0.27$ and $0.105$, and their mean is $0.1314$.

The noise row shows the main effect for the most common setup. The nearly clean sample keeps only a sixth of its loss, while the two noisier samples keep their full loss.

## From one sample to a batch

In the problem, per_sample_mse has shape $(B,)$ and holds one unweighted loss per sample, such as the values from the noise prediction loss. Each sample has its own timestep, so each one reads its own entry of alpha_bars at its 0-based timestep and gets its own ratio and weight. The batch loss is the mean of the products of each weight with its own loss.

## Edge cases

If $\bar\alpha_t$ were exactly 1 the ratio would be infinite, and if it were exactly 0 the noise weight would divide 0 by 0. Schedules used with this weighting keep $\bar\alpha_t$ strictly between those values, so every weight is finite. With schedules that reach zero signal at the last step, implementations either use velocity prediction, whose weight stays finite there, or keep the ratio away from 0.

An unknown target name should raise a ValueError. Silently using the wrong row changes the loss by large factors without any visible failure.

## Where the weighting sits in training

Min-SNR weighting is applied after the per-sample losses are computed and before they are averaged into the scalar that is backpropagated. It needs only the timesteps and the schedule, so it adds almost no cost to a training step. Other schemes, such as the P2 weighting of Choi et al. (2022), plug into the same place.

## Common mistakes to avoid

- **Using the same formula for every target.** The cap is defined in clean sample units, so a noise or velocity loss needs its conversion factor or the weights are off by up to a factor of snr + 1.
- **Dividing the velocity weight by snr.** The velocity error converts with snr + 1, and dividing by snr overweights the noisiest timesteps.
- **Taking square roots in the ratio.** The ratio compares variances, and a ratio of standard deviations moves every sample relative to the cap.
- **Clamping from below instead of above.** Raising small ratios to gamma makes the noisiest timesteps dominate the loss.
- **Multiplying the mean weight by the mean loss.** The product of two averages ignores which sample carries which weight.
- **Summing instead of averaging over the batch.** The loss then grows with the batch size and changes the effective learning rate.