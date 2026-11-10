## <span style="font-size: 20px;">Bayesian Updating with the Beta-Binomial Model</span>

Bayesian inference provides a principled framework for updating beliefs as new data arrives. Unlike frequentist methods that treat parameters as fixed unknowns, the Bayesian approach represents uncertainty about parameters using probability distributions. The Beta-Binomial model is the canonical example of conjugate Bayesian analysis, where the math works out in closed form.

### The Beta Prior

The Beta distribution $\text{Beta}(\alpha, \beta)$ is defined on $[0, 1]$ and is ideal for modeling uncertainty about a probability parameter $\theta$. Its probability density function is:

$$f(\theta; \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

where $B(\alpha, \beta)$ is the Beta function. The parameters $\alpha$ and $\beta$ can be interpreted as "pseudo-counts" of prior successes and failures. The prior mean and variance are:

$$E[\theta] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}(\theta) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$$

The sum $\alpha + \beta$ controls the concentration: larger values produce tighter distributions, representing more confident prior beliefs.

### The Binomial Likelihood

Given a probability parameter $\theta$, the likelihood of observing $s$ successes in $n = s + f$ trials follows the Binomial distribution:

$$P(s | \theta, n) = \binom{n}{s} \theta^s (1 - \theta)^f$$

The key insight is that this likelihood, viewed as a function of $\theta$, has the same functional form as a Beta distribution: $\theta^s(1-\theta)^f$.

### Conjugate Update Rule

When we observe $s$ successes and $f$ failures from a Binomial likelihood, the posterior distribution remains in the Beta family:

$$\text{Prior: } \theta \sim \text{Beta}(\alpha, \beta) \quad \xrightarrow{\text{observe } s, f} \quad \text{Posterior: } \theta \sim \text{Beta}(\alpha + s, \beta + f)$$

This is the defining property of a **conjugate prior**: the posterior belongs to the same distribution family as the prior. The update is purely arithmetic - just add the observed counts to the prior parameters. This computational convenience is why conjugate priors are so popular in practice.

### The Posterior Mean as a Weighted Average

The posterior mean can be written as a weighted combination of the prior mean and the data:

$$E[\theta | \text{data}] = \frac{\alpha + s}{\alpha + \beta + s + f} = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{s}{n}$$

where $n = s + f$. The weight given to the prior versus the data depends on the "strength" of the prior ($\alpha + \beta$) relative to the sample size ($n$). Weak priors are quickly overwhelmed by data; strong priors require more evidence to shift. This provides an elegant mechanism for regularization.

### Uninformative and Informative Priors

The choice $\text{Beta}(1, 1)$ gives the uniform distribution over $[0, 1]$, representing complete prior ignorance. This is an uninformative prior where the posterior is driven entirely by the data. Jeffreys' prior $\text{Beta}(0.5, 0.5)$ is another common non-informative choice with better theoretical properties under reparameterization. In contrast, an informative prior like $\text{Beta}(100, 100)$ encodes a strong belief that $\theta \approx 0.5$ and requires substantial data to overcome.

### Credible Intervals vs. Confidence Intervals

Bayesian credible intervals have a direct probabilistic interpretation: a 95% credible interval means there is a 95% posterior probability that $\theta$ lies in the interval, given the data and prior. This differs fundamentally from frequentist confidence intervals, which describe the long-run coverage probability of the interval construction procedure rather than making probability statements about the parameter itself.

### Applications in Machine Learning

Bayesian updating powers many practical systems. **Thompson sampling** uses posterior Beta distributions for multi-armed bandit problems, sampling from each arm's posterior to balance exploration and exploitation naturally. **Bayesian A/B testing** provides richer conclusions than frequentist tests by computing the full posterior probability that one variant outperforms another. In **online learning** and recommendation systems, Beta-Binomial models enable efficient real-time updates as user interaction data streams in, making them ideal for click-through rate estimation and content personalization.