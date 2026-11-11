# Denoising Score Matching Loss

A score model learns, for any point in space, which direction leads towards more realistic data. Think of a compass that always points uphill towards the places where real examples are most common. Score based diffusion models generate data by starting from random noise and following this compass step by step, so they need a way to train it. The catch is that the correct direction is never written down anywhere in the training data. This page builds the idea one piece at a time, starting from a single number and ending with the noise-weighted loss for a whole batch that you implement in this problem.

## A score is an arrow towards likely data

Start with a density $p(x)$ over a single number $x$. Its score is the derivative of its logarithm, written $\nabla_x \log p(x)$. The score is positive where the density grows to the right and negative where it grows to the left, so stepping in the direction of the score moves a point towards regions of higher probability.

For the standard Gaussian, with mean 0 and variance 1, the log density is $-x^2 / 2$ plus a constant, and its derivative gives the score $-x$. At $x = 2$ the score is $-2$, an arrow pointing back towards the mean with a length that grows the further away the point sits. At $x = 0$ the score is 0, because the point already sits at the peak.

In D dimensions the score is a vector of D partial derivatives, one arrow component per feature. A network $s_\theta(x)$ that outputs this vector field is called a score model. A useful property is that the score never needs the normalizing constant of $p$, since a constant disappears when you take the derivative of the logarithm.

## The true score of real data is never observed

The obvious way to train a score model would be to compare its output with $\nabla_x \log p(x)$ at many data points and minimize the squared difference. This cannot be done as written. A training set contains samples of the data, and nobody knows the density those samples came from, let alone its derivative.

Hyvarinen (2005) found a rewrite of this loss that removes the unknown score, but it needs the trace of the Jacobian of the model. For an image with a hundred thousand pixels that means a hundred thousand backward passes per example, which is far too expensive. Diffusion models use a cheaper trick instead.

## Adding noise makes the target known

Vincent (2011) noticed that the problem becomes easy once each clean sample is blurred with Gaussian noise of a chosen size. Take a clean value $x_0$, draw a standard Gaussian number $\epsilon$, and pick a noise level $\sigma > 0$. The noisy value is

$$
\tilde x = x_0 + \sigma \epsilon .
$$

Given $x_0$, the noisy value follows a Gaussian with mean $x_0$ and variance $\sigma^2$. The score of that one Gaussian is known exactly.

1. The log density of $\tilde x$ given $x_0$ is $-(\tilde x - x_0)^2 / (2\sigma^2)$ plus a constant.
2. Its derivative with respect to $\tilde x$ is $-(\tilde x - x_0) / \sigma^2$.
3. The difference $\tilde x - x_0$ equals $\sigma \epsilon$, so the score simplifies to $-\epsilon / \sigma$.

Take $x_0 = 0.8$, $\sigma = 0.5$ and $\epsilon = -0.6$. The noisy value is $0.8 - 0.3 = 0.5$. The formula in step 2 gives $-(0.5 - 0.8) / 0.25 = 1.2$, and the shortcut in step 3 gives $0.6 / 0.5 = 1.2$ as well. The target is positive because the noise pushed the value to the left of the clean point, and the arrow has to point back to the right.

Vincent showed that regressing a model onto this conditional target has the same minimizer as regressing onto the score of the whole noisy data distribution. The model therefore learns the score of the data smoothed at level $\sigma$, using nothing more than samples and the noise drawn for them.

## Small noise levels shout louder

One noise level is not enough. A small $\sigma$ gives a sharp score near the data but leaves the empty space between data points unexplored, because noisy samples never land there. A large $\sigma$ covers all of space but blurs the detail away. Song and Ermon (2019) trained a single network on a whole ladder of noise levels, with each sample in a batch drawn at its own level.

Mixing levels creates a balance problem. The target $-\epsilon / \sigma$ has a typical size of $1 / \sigma$, so the errors at small levels are far larger. Suppose $\epsilon = 1$ and the model predicts a score of 0. At $\sigma = 0.1$ the target is $-10$ and half the squared error is $50$. At $\sigma = 2$ the target is $-0.5$ and half the squared error is $0.125$. The small level contributes 400 times more to the loss, so its gradients would drown out everything else.

The fix is a weight $\lambda(\sigma) = \sigma^2$ on each sample's error. Multiplying 50 by $0.01$ and $0.125$ by $4$ gives $0.5$ in both cases, so the two levels now count equally. The weight exactly cancels the $1 / \sigma^2$ growth of the squared target.

## The full per-sample loss

For a sample with D features, the error is summed over the features, halved, and multiplied by the weight. Writing $s$ for the model's predicted score at $\tilde x$, the per-sample loss is

$$
\ell = \frac{1}{2}\, \sigma^2 \sum_{d=1}^{D} \left( s_d + \frac{\epsilon_d}{\sigma} \right)^2 .
$$

The plus sign appears because the target is $-\epsilon / \sigma$, so the error $s_d - (-\epsilon_d / \sigma)$ becomes $s_d + \epsilon_d / \sigma$. Moving the $\sigma^2$ inside the square gives the equivalent form $\frac{1}{2} \sum_d (\sigma s_d + \epsilon_d)^2$, which compares the scaled score $\sigma s$ with the negative noise. That quantity has the same size at every noise level, which is the whole point of the weighting.

## A toy score model with a built-in scale

This problem replaces the network with a linear model. The noisy row $\tilde x$ of length D is multiplied by a weight $W$ of shape $(D, D)$, a bias $b$ of shape $(D)$ is added, and the result is divided by $\sigma$.

$$
s = \frac{\tilde x W + b}{\sigma}
$$

The division mirrors a common design in real score networks, which produce an output of unit size and divide it by the noise level so the score has the right magnitude at every $\sigma$. With this model the weighted loss reduces to $\frac{1}{2} \lVert \tilde x W + b + \epsilon \rVert^2$, a handy way to check your numbers.

## A two-sample worked example

Use $D = 2$, the weight with rows $(1.0, 0.0)$ and $(0.5, -1.0)$, and bias $(0.1, 0.0)$.

1. The first sample has $x_0 = (0.6, -0.2)$, $\epsilon = (1.0, 0.5)$ and $\sigma = 0.4$, so $\tilde x = (1.0, 0.0)$.
2. The linear output is $(1.1, 0.0)$, and dividing by $0.4$ gives the predicted score $(2.75, 0.0)$.
3. The target is $-\epsilon / \sigma = (-2.5, -1.25)$, so the error is $(5.25, 1.25)$ and half its squared sum is $14.5625$.
4. Multiplying by $\sigma^2 = 0.16$ gives $\ell = 2.33$. The short form agrees, since $\frac{1}{2}((1.1 + 1.0)^2 + (0.0 + 0.5)^2) = 2.33$.
5. The second sample has $x_0 = (0, 0)$, $\epsilon = (-1.0, 1.0)$ and $\sigma = 3$, so $\tilde x = (-3, 3)$ and the linear output is $(-1.4, -3.0)$.
6. Its predicted score is about $(-0.4667, -1.0)$ and its target is about $(0.3333, -0.3333)$, so half the squared error is about $0.5422$.
7. Multiplying by $\sigma^2 = 9$ gives $\ell = 4.88$, and the batch loss is the mean $(2.33 + 4.88) / 2 = 3.605$.

Without the weighting the first sample would contribute $14.5625$ against $0.5422$, about 27 times more. With it, the large-noise sample carries the larger loss, because its scaled prediction $\sigma s$ lies further from the negative noise.

## One noise level per sample in a batch

In a batch, x0 and noise have shape $(B, D)$ and sigmas has shape $(B)$, one level per row. Each level has to scale every feature of its own row in the same way when forming $\tilde x$, dividing the model output, building the target and applying the weight. That means the $B$ values must line up with the batch axis, which takes a reshape into a column before they meet a $(B, D)$ tensor. The batch loss is the plain mean of the $B$ per-sample losses.

## Where this loss sits in diffusion

This loss is the score based twin of the DDPM noise prediction loss. If a network predicts the noise $\hat\epsilon$, the score it implies is $-\hat\epsilon / \sigma$, and the weighted loss becomes $\frac{1}{2} \lVert \epsilon - \hat\epsilon \rVert^2$. Song et al. (2021) showed that both views describe the same model. The trained score is exactly what annealed Langevin dynamics, the next problem in this plan, uses to turn noise into samples.

## Common mistakes to avoid

- **Dropping the sigma squared weight.** Small noise levels then produce errors many orders of magnitude larger than large ones and swamp the training signal.
- **Flipping the sign of the target.** A target of $+\epsilon / \sigma$ points away from the clean data, so the model learns to push samples into empty space.
- **Dividing the noise by sigma squared.** The target is $-(\tilde x - x_0) / \sigma^2$, which already equals $-\epsilon / \sigma$, so dividing $\epsilon$ itself by $\sigma^2$ counts one factor of $\sigma$ twice.
- **Averaging over the features instead of summing.** The loss uses the full squared norm, and a mean divides every per-sample value by D.
- **Forgetting the factor one half.** Every loss doubles, and the values no longer agree with the definition.
- **Lining sigmas up with the feature axis.** A $(B)$ vector broadcast against a $(B, D)$ tensor without a reshape pairs noise levels with features instead of samples, or fails when B and D differ.