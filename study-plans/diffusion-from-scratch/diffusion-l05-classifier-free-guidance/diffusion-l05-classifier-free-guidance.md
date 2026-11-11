# Classifier-Free Guidance

A model that draws pictures from text prompts tends to follow the prompt loosely, like an artist who hears the request but drifts towards whatever is most typical. Classifier-free guidance makes it listen harder. At every sampling step it asks the model twice, once with the prompt and once without, and then pushes the answer further in the direction that the prompt added. Strong pushes also inflate the prediction, so a second, optional step shrinks it back to a sensible size. This page builds both pieces from a single number and ends with the per-sample guided prediction you implement in this problem.

## Two answers to the same question

A conditional diffusion model takes the noisy sample $x_t$ and a condition $y$, such as a class label or a text prompt, and predicts the noise. Ho and Salimans (2022) trained one network to work both with and without the condition. During training the condition is dropped for a fraction of examples, often about 10 percent, and replaced by an empty token, so the network also learns to predict without any prompt.

At sampling time the same network is called twice on the same $x_t$. The conditional prediction $c$ (cond_pred) uses the prompt, and the unconditional prediction $u$ (uncond_pred) uses the empty token. Both have the shape of the sample.

Start with one element where $c = 0.6$ and $u = 0.3$. The two answers differ by $0.3$, and that difference is the part of the prediction that exists only because of the prompt.

## Following the difference further

The difference $c - u$ is a direction that points from "anything plausible" towards "plausible and matching the prompt". Guidance starts at the unconditional answer and walks along that direction by a factor $w$, the guidance scale (guidance_scale).

$$
g = u + w\,(c - u)
$$

For the single element, $w = 0$ gives $0.3$, the unconditional answer, and $w = 1$ gives $0.6$, the plain conditional answer. With $w = 5$ the result is $0.3 + 5 \cdot 0.3 = 1.8$, which overshoots well past the conditional answer. Overshooting is the point, because it exaggerates whatever the prompt contributed. Typical text-to-image settings use $w$ between about 5 and 10.

The order of the two predictions matters. Swapping them gives $c + w\,(u - c)$, which walks away from the prompt, so higher scales produce samples that avoid it.

## Why the difference acts like a classifier

Earlier work steered samples with a separate classifier trained on noisy images, adding the gradient of $\log p(y \mid x_t)$ to the model's score (Dhariwal and Nichol, 2021). A noise prediction is proportional to the negative score, $-\nabla_x \log p(x_t)$, and by Bayes rule the gap between the conditional and unconditional scores is exactly $\nabla_x \log p(y \mid x_t)$.

So $c - u$ is, up to a constant, the gradient that a classifier would have provided. Multiplying it by $w$ is like using a classifier whose push has strength $w$, without training any classifier at all. That is where the name classifier-free comes from.

## Why strong guidance overexposes samples

Extrapolation also changes the size of the prediction. The guided prediction is $c$ plus $(w - 1)$ copies of the gap, so whenever the gap is not tiny, $g$ has a much larger spread than either $c$ or $u$. When that oversized noise prediction is turned into a clean image estimate, the estimate is over-corrected, which shows up as blown-out colours and harsh contrast at high guidance scales.

Lin et al. (2024) proposed a simple fix. Measure the spread of each sample with its standard deviation, and scale the guided prediction so that its spread matches the spread of the conditional prediction. The conditional prediction is the target because it is a prediction the model was trained to make at a realistic scale.

## Rescaling one sample

Let $s(\cdot)$ be the population standard deviation over all elements of one sample, meaning the square root of the mean squared deviation from the sample's own mean. The rescaled prediction multiplies the guided one by a single scalar per sample.

$$
r = g \cdot \frac{s(c)}{s(g)}
$$

Fully rescaled predictions can look flat, so the final answer blends the two with a factor $\phi$ (guidance_rescale) between 0 and 1.

$$
\text{result} = \phi\, r + (1 - \phi)\, g
$$

With $\phi = 0$ the result is the plain guided prediction, and with $\phi = 1$ it is fully rescaled. Lin et al. reported that $\phi = 0.7$ works well. The rescaling multiplies by one scalar, so it keeps the direction that guidance chose and changes only its size.

## A three-element example

Take one sample with three elements, $c = (0.6, 0, -0.3)$ and $u = (0.3, 0.2, -0.2)$, and use $w = 5$ and $\phi = 0.5$.

1. The gap is $c - u = (0.3, -0.2, -0.1)$, so $g = u + 5 (c - u) = (1.8, -0.8, -0.7)$.
2. The mean of $c$ is $0.1$ and its deviations are $(0.5, -0.1, -0.4)$. Their squares average to $0.42 / 3 = 0.14$, so $s(c) \approx 0.37417$.
3. The mean of $g$ is also $0.1$, with deviations $(1.7, -0.9, -0.8)$. Their squares average to $4.34 / 3 \approx 1.44667$, so $s(g) \approx 1.20277$.
4. The ratio is $0.37417 / 1.20277 \approx 0.31109$, so $r \approx (0.55995, -0.24887, -0.21776)$.
5. The blend with $\phi = 0.5$ is about $(1.17998, -0.52443, -0.45888)$.

The guided spread was more than three times the conditional spread, and the blend removes about half of that excess. Had the target been the unconditional spread, about $0.21602$, every value would have been shrunk much further, which changes the brightness and contrast of the result.

## From one sample to a batch

In the problem, cond_pred and uncond_pred have shape $(B, \dots)$, for example $(B, C, H, W)$. The guidance formula is elementwise, so it applies to the whole batch at once. The rescale step is per sample, though, so each sample gets its own pair of standard deviations, taken over all of its non-batch elements, and its own scalar ratio. A ratio computed over the whole batch would couple unrelated images, letting one high-contrast image change the brightness of another.

## Population versus sample standard deviation

A standard deviation can divide the squared deviations by $n$ or by $n - 1$. This problem uses the population version that divides by $n$. If both standard deviations used the same divisor the choice would cancel in the ratio, but mixing the two scales every rescaled value by $\sqrt{n / (n - 1)}$, which is about $1.2247$ for three elements. The ratio is undefined when a guided sample has no variation at all, which happens only in degenerate cases.

## Where guidance sits in a sampler

Guidance wraps the network call inside any sampler, such as DDPM or DDIM. At every step the sampler evaluates the model on the conditional and unconditional inputs, often in one doubled batch, combines the two outputs with this function, and passes the guided prediction to the update rule as if it came from a single model. The cost is a second network evaluation per step.

## Common mistakes to avoid

- **Swapping the conditional and unconditional predictions.** The formula then extrapolates away from the condition, so a higher scale makes samples ignore the prompt more.
- **Rescaling to the spread of the unconditional prediction.** The target spread comes from the conditional prediction, and the other one changes the brightness and contrast of every sample.
- **Inverting the blend weights.** The rescale factor weights the rescaled tensor, so swapping $\phi$ and $1 - \phi$ gives no rescaling at all at $\phi = 1$.
- **Computing one standard deviation over the whole batch.** Each sample has its own spread, and a batch-wide ratio couples unrelated samples together.
- **Using the default sample standard deviation.** It divides by $n - 1$, and applying it to only one of the two spreads leaves every rescaled value off by a constant factor.