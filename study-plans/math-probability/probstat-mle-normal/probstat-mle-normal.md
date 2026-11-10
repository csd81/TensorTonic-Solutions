## <span style="font-size: 20px;">Maximum Likelihood Estimation for the Normal Distribution</span>

Maximum Likelihood Estimation (MLE) is a foundational method in statistics and machine learning for estimating the parameters of a probability distribution given observed data. The core principle is elegant: find the parameter values that make the observed data most probable. MLE was formalized by R.A. Fisher in the 1920s and remains one of the most widely used estimation frameworks today.

### The Likelihood Function

Given $n$ independent observations $x_1, x_2, \ldots, x_n$ drawn from a normal distribution $\mathcal{N}(\mu, \sigma^2)$, the likelihood function is the joint probability density:

$$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

Working with products is inconvenient, so we take the natural logarithm to obtain the **log-likelihood**:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

The log-likelihood is a monotonic transformation, so maximizing it is equivalent to maximizing the original likelihood. This transformation converts products into sums, making differentiation straightforward.

### Deriving the MLE Estimates

To find the maximum, we take partial derivatives and set them to zero.

**For $\mu$:** Taking $\frac{\partial \ell}{\partial \mu} = 0$ gives:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

Solving yields the MLE of the mean:

$$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

This is simply the sample mean, which aligns with intuition.

**For $\sigma^2$:** Taking $\frac{\partial \ell}{\partial \sigma^2} = 0$ gives:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \mu)^2 = 0$$

Solving yields the MLE of the variance:

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2$$

The MLE of the standard deviation is therefore $\hat{\sigma}_{\text{MLE}} = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$. Notice this divides by $n$, not $n-1$. The Bessel-corrected version (dividing by $n-1$) gives an unbiased estimate of variance, but MLE produces the **biased estimator** that maximizes likelihood. The bias is $-\sigma^2/n$, which vanishes as $n$ grows.

### Properties of MLE

MLE estimators enjoy several desirable asymptotic properties:

- **Consistency**: As $n \to \infty$, the MLE converges in probability to the true parameter value.
- **Asymptotic efficiency**: The MLE achieves the Cramer-Rao lower bound asymptotically, meaning no consistent estimator has lower variance for large samples.
- **Asymptotic normality**: For large $n$, the MLE is approximately normally distributed around the true parameter with variance equal to the inverse Fisher information.
- **Invariance**: If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$. This is why we can take the square root of the variance MLE to get the standard deviation MLE.

### Fisher Information

The **Fisher information** $I(\theta) = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]$ quantifies how much information a sample carries about the parameter. For the normal distribution, $I(\mu) = n/\sigma^2$ and $I(\sigma^2) = n/(2\sigma^4)$. The Cramer-Rao bound states that $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$, establishing a fundamental limit on estimation precision.

### Applications in Machine Learning

MLE is everywhere in modern ML. Fitting a Gaussian distribution to data is the simplest case, but the same principle underlies logistic regression (maximizing Bernoulli likelihood), neural network training (minimizing cross-entropy loss is equivalent to maximizing likelihood), and generative models like VAEs and normalizing flows. The Expectation-Maximization (EM) algorithm uses MLE in its M-step to handle latent variables, powering Gaussian Mixture Models, Hidden Markov Models, and many other unsupervised learning methods.