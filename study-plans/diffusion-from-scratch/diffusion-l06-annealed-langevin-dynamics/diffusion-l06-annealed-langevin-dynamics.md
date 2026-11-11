# Annealed Langevin Dynamics

Annealed Langevin dynamics is the procedure that turns a score model into new samples. Picture marbles scattered over a hilly landscape that roll towards the valleys while the ground is shaken, with the shaking calmed bit by bit so they settle in the right places. Score based diffusion models use this loop to move random starting points towards realistic data, taking large steps while the picture is blurry and small careful steps once it sharpens. This page builds the sampler one piece at a time, starting from a single point climbing a single hill and ending with the full loop over a ladder of noise levels that you implement in this problem. The score here is exact, so the sampler itself is the only thing to get right.

## Climbing the log density with the score

The score of a density $p(x)$ is $\nabla_x \log p(x)$, an arrow that points towards regions of higher probability. The simplest way to use it is gradient ascent on the log density. Pick a small positive step size $\alpha$ and move the point a little along the arrow.

$$
x \leftarrow x + \frac{\alpha}{2}\, \nabla_x \log p(x)
$$

Take a one-dimensional Gaussian with mean 0 and variance 1, whose score is $-x$. Starting at $x = 2$ with $\alpha = 0.5$, one step gives $2 + 0.25 \cdot (-2) = 1.5$. Repeating the step moves the point closer and closer to 0.

That is the problem with plain ascent. Every starting point ends at the peak, so a thousand runs give a thousand copies of the same value. A sampler needs the points to spread out the way the data does, with most near the peak and some further away.

## Adding noise turns climbing into sampling

Langevin dynamics adds a fresh random kick to every step. With $z$ a standard Gaussian draw of the same shape as $x$, the update becomes

$$
x \leftarrow x + \frac{\alpha}{2}\, \nabla_x \log p(x) + \sqrt{\alpha}\, z .
$$

The drift pulls the point uphill and the kick pushes it around at random. The factor $\frac{1}{2}$ on the drift and the $\sqrt{\alpha}$ on the kick are tied together, and that exact balance is what makes the chain settle into the distribution $p$ itself. A Gaussian with mean 0 and variance $v$ shows why, since its score is $-x / v$.

1. One step maps $x$ to $(1 - \frac{\alpha}{2v})\, x + \sqrt{\alpha}\, z$.
2. If the chain has settled with variance $V$, the next point must have the same variance, so $V = (1 - \frac{\alpha}{2v})^2 V + \alpha$.
3. Solving for $V$ gives $V = v / (1 - \frac{\alpha}{4v})$, which approaches the true variance $v$ as $\alpha$ shrinks.

With $v = 1$ and $\alpha = 0.1$, the chain settles at a variance of about $1.0256$, very close to 1. If the $\frac{1}{2}$ were dropped, the same reasoning gives about $0.5263$, roughly half the true spread. The balance between drift and kick is part of the method.

## One noise level is not enough

A score trained on clean data alone is poor exactly where sampling starts. Random starting points lie far from the data, where the training set has no examples, so the learned arrows there point anywhere. Real data also tends to sit near a thin surface inside a high-dimensional space, where the clean score is badly behaved.

Blurring the data with Gaussian noise of level $\sigma$ fixes both problems. The blurred density covers all of space and its score is reliable everywhere. A large $\sigma$ gives trustworthy arrows far from the data but washes out the detail, while a small $\sigma$ keeps the detail but is reliable only close to the data.

Annealing uses both. The chain starts at the largest level and hands its points down a ladder of decreasing levels $\sigma_0 > \sigma_1 > \dots > \sigma_{L-1}$, where L is the number of levels and the index $i$ starts at 0. The last level is called $\sigma_{\text{last}}$, so $\sigma_{\text{last}} = \sigma_{L-1}$. Song and Ermon (2019) introduced this sampler for their noise conditional score networks.

## The score of a Gaussian blurred by noise

A trained network would normally supply the score at each level. This problem uses a data distribution whose blurred score is known in closed form, which isolates the sampler. The data is a Gaussian with mean $\mu$, a vector of shape $(D)$, and variance $v$ on every axis. Adding independent noise of level $\sigma_i$ adds $\sigma_i^2$ to the variance, because independent variances add. The blurred density is a Gaussian with variance $v + \sigma_i^2$, and its score at level $i$ is

$$
s_i(x) = -\frac{x - \mu}{v + \sigma_i^2} .
$$

The arrow points straight at the mean and grows with distance. A larger $\sigma_i$ gives a gentler pull, which matches the blurrier picture at that level. With $\mu = 0$ and $v = 1$, a point at $x = 2$ gets a score of $-2 / (1 + 9) = -0.2$ at $\sigma_i = 3$ and $-2 / (1 + 2.25) \approx -0.6154$ at $\sigma_i = 1.5$.

## Matching the step size to the noise level

A single step size cannot suit every level. At a large $\sigma_i$ the score is small, so a step tuned for the last level would barely move the points. At a small $\sigma_i$ the score is large, so a step tuned for the first level would overshoot. Song and Ermon set

$$
\alpha_i = \eta\, \frac{\sigma_i^2}{\sigma_{\text{last}}^2},
$$

where $\eta$ is the base step size, the value used at the last level. The reasoning compares the size of the drift with the size of the kick.

1. The score at level $i$ has a typical size of about $1 / \sigma_i$, so the drift has size about $\alpha_i / \sigma_i$.
2. The kick has size about $\sqrt{\alpha_i}$, so the ratio of drift to kick is about $\sqrt{\alpha_i} / \sigma_i$.
3. Choosing $\alpha_i$ proportional to $\sigma_i^2$ makes that ratio the same at every level.

Every level then spends its steps with the same balance between moving uphill and exploring. The first level takes the largest steps and the last level takes steps of exactly $\eta$.

## Walking down the ladder

The full sampler starts from the points in x_init, an array of shape $(N, D)$ with N points of D features. For each level $i$ from 0 to $L - 1$ it applies the Langevin update steps_per_sigma times, recomputing the score from the current points before every step. Each step uses its own noise slice $z_{i,j}$ of shape $(N, D)$, where $j$ counts the steps inside level $i$ from 0. The slices are stored level by level, so all the steps of level 0 come first, then all the steps of level 1, and so on.

All N points move independently. They share the levels, step sizes and score function, but each row has its own noise and its own path.

## A two-level worked example

Take one point $x = 2$ in one dimension, data mean $\mu = 0$, data variance $v = 1$, levels $(3.0, 1.5)$, one step per level, base step size $\eta = 0.2$ and noise values $0.5$ and $-1.0$.

1. At level 0 the step size is $\alpha_0 = 0.2 \cdot 3.0^2 / 1.5^2 = 0.8$ and the score is $-0.2$.
2. The drift is $0.4 \cdot (-0.2) = -0.08$ and the kick is $\sqrt{0.8} \cdot 0.5 \approx 0.4472$, so $x$ becomes about $2.3672$.
3. At level 1 the step size is $\alpha_1 = 0.2$ and the score is $-2.3672 / 3.25 \approx -0.7284$.
4. The drift is about $-0.0728$ and the kick is $\sqrt{0.2} \cdot (-1.0) \approx -0.4472$, so $x$ ends at about $1.8472$.

The first level took a step four times larger than the second, exactly as the ratio $\sigma_0^2 / \sigma_{\text{last}}^2 = 4$ demands. The second score is larger than the first even though the point moved further from the mean, because the smaller noise level pulls harder.

## Keeping every step stable

The update is stable only when the drift does not overshoot the mean. For the blurred Gaussian score, one step multiplies the distance to the mean by $1 - \alpha_i / (2(v + \sigma_i^2))$. Substituting the step size shows that this factor stays between 0 and 1 at every level when $\eta$ is at most $2 \sigma_{\text{last}}^2$. A larger $\eta$ makes the points jump past the mean and oscillate or diverge, and a learned score behaves the same way. In the example above the limit is $2 \cdot 1.5^2 = 4.5$, far above the $\eta = 0.2$ used.

Annealed Langevin dynamics was the first sampler for score based models. Song et al. (2021) later showed that this noise ladder and the DDPM schedule are two discretizations of stochastic differential equations, and their predictor corrector samplers still use Langevin steps as the corrector.

## Common mistakes to avoid

- **Keeping the step size constant across levels.** The chain then barely moves at the large noise levels and overshoots or jitters at the small ones.
- **Normalizing by the first sigma instead of the last.** The ratio $\sigma_i^2 / \sigma_0^2$ shrinks every step below its intended size, so the final level uses a tiny fraction of the base step size.
- **Reading the noise slices in the wrong order.** Swapping the roles of the level index and the step index gives a chain that looks valid but no longer follows the given sequence of draws.
- **Dropping the factor one half on the drift.** The drift then doubles relative to the kick, and the chain settles into a distribution about half as spread as the data.
- **Using the clean data score at every level.** The variance must include $\sigma_i^2$, otherwise every level pulls as hard as an unblurred density would.
- **Computing the score once per level.** The score depends on the current points, so it has to be recomputed before every single step.
- **Updating x_init in place.** The caller's starting points are overwritten, which corrupts any later reuse of them.