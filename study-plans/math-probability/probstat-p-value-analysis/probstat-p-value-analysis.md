## <span style="font-size: 20px;">P-Value Analysis</span>

The **p-value** is the probability of observing a test statistic as extreme as (or more extreme than) the one actually observed, assuming the null hypothesis $H_0$ is true:

$$p = P(T \geq t_{\text{obs}} \mid H_0) \quad \text{(right-tailed)}$$

$$p = P(T \leq t_{\text{obs}} \mid H_0) \quad \text{(left-tailed)}$$

$$p = 2 \cdot P(T \geq |t_{\text{obs}}| \mid H_0) \quad \text{(two-tailed)}$$

The p-value is a measure of the evidence against $H_0$. Smaller p-values indicate stronger evidence that $H_0$ is false.

### What a P-Value Is NOT

This is critical and frequently misunderstood:

- The p-value is **not** the probability that $H_0$ is true. $P(H_0 \mid \text{data})$ requires Bayesian analysis with a prior.
- The p-value is **not** the probability that the result occurred by chance.
- The p-value is **not** the probability of making a Type I error (that is $\alpha$, which you set before the test).
- The p-value is **not** the probability of replicating the result.

The p-value answers: "If there were truly no effect, how surprising would my data be?" A small p-value means the data are unlikely under $H_0$, which gives us reason to doubt $H_0$.

### Computing P-Values for Different Distributions

For a **normal** (z) distribution: use $\Phi(z)$ for left-tail, $1 - \Phi(z)$ for right-tail.

For a **t-distribution** with $df$ degrees of freedom: use the t CDF and survival function similarly, but the distribution shape depends on $df$.

For a **chi-square** distribution: always use the right tail (survival function), since $\chi^2$ is always non-negative and only large values indicate poor fit.

### Common Significance Thresholds

| Threshold | Label | Convention |
|:---:|:---:|:---|
| $p < 0.10$ | Marginal | Sometimes used in exploratory work |
| $p < 0.05$ | Significant | Most common threshold |
| $p < 0.01$ | Highly significant | Stronger evidence |
| $p < 0.001$ | Very highly significant | Very strong evidence |

These thresholds are conventions, not natural laws. The 0.05 threshold was popularized by Ronald Fisher but was never intended to be a rigid boundary. Many fields are moving toward reporting exact p-values and letting readers judge significance.

### One-Tailed vs Two-Tailed

A **one-tailed test** has a directional alternative ($\mu > \mu_0$ or $\mu < \mu_0$) and places all $\alpha$ in one tail. A **two-tailed test** ($\mu \neq \mu_0$) splits $\alpha$ between both tails. Two-tailed tests are more conservative and are the default in most applications.

For a given test statistic, the two-tailed p-value is exactly twice the one-tailed p-value (in the relevant direction). One-tailed tests are appropriate only when there is a strong prior reason to expect the effect in a specific direction and the opposite direction is scientifically meaningless.

### The Multiple Testing Problem

When performing $m$ independent tests at level $\alpha$, the probability of at least one false positive is:

$$1 - (1-\alpha)^m$$

With 20 tests at $\alpha = 0.05$, there is a 64% chance of at least one false positive. The **Bonferroni correction** addresses this by testing each hypothesis at level $\alpha/m$, controlling the family-wise error rate. The **Benjamini-Hochberg procedure** is a less conservative alternative that controls the false discovery rate (FDR).

### P-Hacking

**P-hacking** refers to practices that artificially produce small p-values: testing many hypotheses and only reporting significant ones, optional stopping (checking results repeatedly and stopping when $p < 0.05$), or selectively choosing analysis methods. These practices inflate false positive rates far beyond the nominal $\alpha$. Pre-registration of analysis plans helps combat p-hacking.

### Effect Size vs Statistical Significance

A statistically significant result is not necessarily practically important. With a large enough sample, even trivially small effects become significant. **Effect sizes** (Cohen's $d$, correlation coefficients, odds ratios) measure the magnitude of an effect independent of sample size and should always accompany p-values.

### Applications

**Feature selection**: Testing which features have significant predictive power, with multiple testing correction to avoid false discoveries.

**Hyperparameter significance**: Determining whether observed performance differences between model configurations are genuine or due to random variation.

**Publishing ML results**: Reporting p-values alongside confidence intervals provides a complete picture of both statistical significance and practical importance.