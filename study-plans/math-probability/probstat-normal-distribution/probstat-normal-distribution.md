## <span style="font-size: 20px;">Normal (Gaussian) Distribution</span>

The normal distribution is the most important continuous probability distribution in statistics and machine learning. Its ubiquity stems from the Central Limit Theorem, its mathematical tractability, and its natural appearance in physical and social phenomena.

### Probability Density Function

A random variable $X \sim \mathcal{N}(\mu, \sigma^2)$ has the PDF:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Parameters:
- $\mu$: mean (location parameter) - determines the center of the bell curve.
- $\sigma^2$: variance (scale parameter) - determines the spread. $\sigma$ is the standard deviation.

The PDF is symmetric about $\mu$, and its shape is the familiar bell curve. Larger $\sigma$ produces a wider, flatter curve; smaller $\sigma$ produces a taller, narrower one. The peak of the density occurs at $x = \mu$ with value $\frac{1}{\sigma\sqrt{2\pi}}$.

### The Standard Normal Distribution

When $\mu = 0$ and $\sigma = 1$, we get the **standard normal** $Z \sim \mathcal{N}(0, 1)$. Any normal variable can be standardized via the **z-score**:

$$Z = \frac{X - \mu}{\sigma}$$

The z-score tells us how many standard deviations $X$ is from the mean. A z-score of 2 means the observation is 2 standard deviations above the mean. Standardization allows us to use a single table or function for all normal distributions.

### The 68-95-99.7 Rule (Empirical Rule)

The empirical rule states that for a normal distribution:

| Range | Probability |
|-------|------------|
| $\mu \pm 1\sigma$ | 68.27% |
| $\mu \pm 2\sigma$ | 95.45% |
| $\mu \pm 3\sigma$ | 99.73% |

This rule provides quick mental estimates for probabilities. An observation more than 3 standard deviations from the mean has less than a 0.3% chance under the normal model, making it a candidate for an outlier or anomaly.

### CDF and Quantile Function

The **cumulative distribution function** $\Phi(x)$ gives $P(X \leq x)$. It has no closed-form expression and is computed numerically:

$$\Phi(x) = \int_{-\infty}^{x} f(t)\, dt$$

The **quantile function** (inverse CDF) $\Phi^{-1}(p)$ returns the value below which a fraction $p$ of the distribution falls. This is used to compute critical values for hypothesis tests and confidence intervals. For example, $\Phi^{-1}(0.975) = 1.96$, the critical value for a 95% two-sided confidence interval.

### Central Limit Theorem

The CLT states that the sum (or mean) of a large number of independent, identically distributed random variables approaches a normal distribution, regardless of the original distribution:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

This is why the normal distribution appears everywhere: averages, measurement errors, test statistics, and regression residuals all tend toward normality. The CLT justifies using normal-based inference even when the underlying data is not normally distributed, provided the sample size is sufficient.

### Properties of the Normal Distribution

The normal distribution has several convenient properties:
- **Linear combinations**: If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then $aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2, a^2\sigma_1^2 + b^2\sigma_2^2)$.
- **Maximum entropy**: Among all distributions with given mean and variance, the normal has the highest entropy, making it the least informative (most conservative) choice.

### Applications in Machine Learning

**Loss functions and Gaussian noise**: The MSE loss function implicitly assumes Gaussian-distributed errors. Minimizing MSE is equivalent to maximum likelihood estimation under the assumption $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

**Batch normalization**: BatchNorm transforms activations to have approximately zero mean and unit variance, stabilizing training and enabling higher learning rates.

**Weight initialization**: Methods like Xavier and He initialization draw weights from normal distributions with carefully chosen variances to maintain stable gradient flow through deep networks.

**Gaussian processes**: A GP defines a distribution over functions where any finite collection of function values is jointly normal. GPs provide uncertainty-aware predictions and are used in Bayesian optimization for hyperparameter tuning.