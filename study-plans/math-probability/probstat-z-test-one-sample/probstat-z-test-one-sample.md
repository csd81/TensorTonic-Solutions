## <span style="font-size: 20px;">One-Sample Z-Test</span>

The **one-sample z-test** determines whether a sample mean $\bar{x}$ differs significantly from a hypothesized population mean $\mu_0$, when the population standard deviation $\sigma$ is known. It is the simplest and most foundational hypothesis test.

### Hypotheses

For a two-tailed test:

$$H_0: \mu = \mu_0 \qquad H_a: \mu \neq \mu_0$$

The null hypothesis $H_0$ states that the true population mean equals the hypothesized value. The alternative $H_a$ states that it differs in either direction. One-tailed variants test $H_a: \mu > \mu_0$ or $H_a: \mu < \mu_0$ when there is a prior reason to expect a specific direction.

### Test Statistic

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

This standardizes the difference between the sample mean and hypothesized mean by the standard error. Under $H_0$, this statistic follows a standard normal distribution $N(0, 1)$. The further $\bar{x}$ is from $\mu_0$ relative to the standard error, the larger $|z|$ becomes, and the stronger the evidence against $H_0$.

### Decision Rule

**Using critical values**: Reject $H_0$ if $|z| > z_{\alpha/2}$, where $z_{\alpha/2}$ is the critical value (e.g., 1.96 for $\alpha = 0.05$, 2.576 for $\alpha = 0.01$).

**Using p-values**: The p-value for a two-tailed test is:

$$p = 2 \cdot P(Z > |z|) = 2 \cdot [1 - \Phi(|z|)]$$

Reject $H_0$ if $p < \alpha$. The p-value approach is preferred because it provides the exact strength of evidence, not just a binary decision.

### Type I and Type II Errors

| | $H_0$ True | $H_0$ False |
|:---|:---:|:---:|
| Reject $H_0$ | Type I error ($\alpha$) | Correct (Power) |
| Fail to reject | Correct | Type II error ($\beta$) |

- **Type I error** ($\alpha$): Rejecting a true null hypothesis (false positive). The significance level controls this probability directly.
- **Type II error** ($\beta$): Failing to reject a false null hypothesis (false negative). This depends on the true effect size, sample size, and $\alpha$.
- **Power** ($1 - \beta$): The probability of correctly rejecting a false $H_0$. Power increases with sample size, effect size, and $\alpha$. A well-designed study targets power of at least 0.80.

### The Relationship Between Power and Sample Size

For a two-tailed z-test detecting effect size $\delta = \mu - \mu_0$:

$$n = \left(\frac{(z_{\alpha/2} + z_\beta) \cdot \sigma}{\delta}\right)^2$$

This formula shows that to detect smaller effects, you need quadratically more data. Halving the detectable effect size requires four times the sample size.

### Confidence Interval Connection

A $(1-\alpha)$ confidence interval for $\mu$ from the z-test is:

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

There is a direct duality: $H_0: \mu = \mu_0$ is rejected at level $\alpha$ if and only if $\mu_0$ falls outside the $(1-\alpha)$ CI. This means the CI contains all values of $\mu_0$ that would not be rejected by the test.

### Assumptions

1. The observations are independent
2. The population standard deviation $\sigma$ is known (rare in practice; when unknown, use the t-test instead)
3. The sampling distribution of $\bar{X}$ is approximately normal (guaranteed by CLT for large $n$, or if the population is normal)

### Applications

**Quality control**: Testing whether a manufacturing process produces items with a target mean measurement when historical process variance is well-established.

**Monitoring model performance**: If baseline model accuracy has a known standard deviation from repeated evaluations, a z-test can determine if a new model's accuracy is significantly different.

**A/B testing foundations**: The z-test logic extends directly to two-sample comparisons. Understanding the one-sample case is essential before tackling two-sample tests and A/B test frameworks.

**Anomaly detection**: In monitoring systems, z-tests can flag when a metric (latency, error rate, throughput) deviates significantly from its historical mean. If the metric's historical standard deviation is well-characterized, the z-test provides an immediate significance assessment.

### Limitations

The z-test's assumption that $\sigma$ is known is its primary limitation. In practice, $\sigma$ is almost never known exactly. When it must be estimated from data, the t-test (which accounts for this estimation uncertainty) is the appropriate choice. The z-test remains important conceptually because it illustrates hypothesis testing mechanics without the added complexity of the t-distribution, and it is asymptotically equivalent to the t-test for large samples.