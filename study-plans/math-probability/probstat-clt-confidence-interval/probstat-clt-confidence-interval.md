## <span style="font-size: 20px;">Confidence Intervals via the Central Limit Theorem</span>

The **Central Limit Theorem** states that the sum (or mean) of a large number of independent, identically distributed random variables converges to a normal distribution, regardless of the original distribution's shape:

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

This convergence allows us to build confidence intervals even when the underlying data is not normal. The CLT is the theoretical foundation that makes most of frequentist inference practical.

### Constructing a Confidence Interval

A confidence interval provides a range of plausible values for the population mean $\mu$. Using the CLT:

$$\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot SE \qquad \text{where } SE = \frac{s}{\sqrt{n}}$$

Here $s$ is the sample standard deviation (with Bessel's correction, dividing by $n-1$), and $z_{\alpha/2}$ is the critical value from the standard normal distribution. The quantity $z_{\alpha/2} \cdot SE$ is the half-width of the interval.

### Critical Values for Common Confidence Levels

| Confidence Level | $z_{\alpha/2}$ | Interval Width Factor |
|:---:|:---:|:---:|
| 90% | 1.645 | 3.29 SE |
| 95% | 1.960 | 3.92 SE |
| 99% | 2.576 | 5.15 SE |

For a 95% confidence interval, we use $z = 1.96$, so the interval is $\bar{x} \pm 1.96 \cdot SE$. Notice how moving from 95% to 99% confidence increases the interval width by about 31%. Higher confidence requires a wider interval.

### Correct Interpretation

A 95% confidence interval does **not** mean "there is a 95% probability that $\mu$ lies in this interval." The parameter $\mu$ is fixed (not random) - it either is or is not in the interval. The correct interpretation is: **if we repeated the sampling process many times, 95% of the resulting intervals would contain the true $\mu$**. This is a statement about the procedure, not about any single interval.

This distinction matters in practice. A single 95% CI either contains $\mu$ or it does not - we just do not know which. The "95%" describes our long-run confidence in the method.

### Margin of Error

The **margin of error** is the half-width of the confidence interval:

$$ME = z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

The margin of error decreases with the square root of $n$. To cut the margin of error in half, you need four times the sample size. This reflects the diminishing returns of additional data collection. Going from $n=100$ to $n=400$ halves the margin of error, but going from $n=400$ to $n=1600$ only halves it again.

### Sample Size Determination

To achieve a desired margin of error $ME$ at confidence level $1 - \alpha$:

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{ME}\right)^2$$

This formula is critical for experiment planning - determining how much data to collect before starting. When $\sigma$ is unknown, use a pilot study estimate or a conservative upper bound.

### When the CLT Approximation is Poor

The CLT is an asymptotic result. For small samples from heavily skewed or heavy-tailed distributions, the normal approximation may be poor. Warning signs include: sample sizes below 30, visible skewness in the data, presence of extreme outliers, and bounded distributions (like proportions near 0 or 1). In these cases, consider bootstrap confidence intervals or exact methods.

### Applications in ML and Data Science

**A/B testing**: Confidence intervals on conversion rate differences tell you not just whether a change is significant, but the plausible range of its effect size. A CI of [0.001, 0.050] says the improvement is real but could be tiny.

**Model evaluation uncertainty**: A model with 92% accuracy on 500 test samples has a 95% CI of roughly $92\% \pm 2.4\%$. Reporting just the point estimate without this uncertainty is misleading.

**Hyperparameter tuning**: Cross-validation means and their CIs help distinguish genuinely better configurations from noise. If two configurations have overlapping CIs, the difference may not be reliable.

**Reporting results**: Academic papers increasingly require confidence intervals rather than just p-values, because CIs convey both the direction and magnitude of an effect.