## <span style="font-size: 20px;">One-Sample T-Test</span>

The **one-sample t-test** tests whether a sample mean differs significantly from a hypothesized value $\mu_0$ when the population standard deviation is **unknown** and must be estimated from the data. This is far more common in practice than the z-test, since we rarely know $\sigma$ in advance.

### Student's t-Distribution

When we replace the known $\sigma$ with the sample standard deviation $s$, the test statistic no longer follows a standard normal distribution. Instead, it follows a **t-distribution** with $n-1$ degrees of freedom:

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \sim t_{n-1}$$

The t-distribution was discovered by William Sealy Gosset, who published under the pseudonym "Student" while working at the Guinness brewery. It has **heavier tails** than the standard normal, reflecting the additional uncertainty from estimating $\sigma$. With fewer observations, the tails are heavier (more probability in extreme values). As the degrees of freedom increase, the t-distribution converges to the normal distribution - by $df \approx 30$, they are nearly indistinguishable.

### Degrees of Freedom

The degrees of freedom $df = n - 1$ represent the number of independent pieces of information available to estimate variability. We lose one degree of freedom because we used the data to estimate $\bar{x}$ before computing $s$. The $n$ deviations $x_i - \bar{x}$ must sum to zero, so only $n-1$ are free to vary.

### Test Procedure

For a two-tailed test with hypotheses $H_0: \mu = \mu_0$ vs $H_a: \mu \neq \mu_0$:

1. Compute the sample mean $\bar{x}$ and sample standard deviation $s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$
2. Compute $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$
3. Compute $p = 2 \cdot P(T_{n-1} > |t|)$
4. Reject $H_0$ if $p < \alpha$

### Confidence Interval

The $(1-\alpha)$ confidence interval is:

$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

Note that $t_{\alpha/2, n-1} > z_{\alpha/2}$ for finite $n$, so t-based confidence intervals are wider than z-based intervals. This appropriately reflects the extra uncertainty from estimating $\sigma$. For example, with $n = 10$ and 95% confidence, the t critical value is 2.262 compared to the z value of 1.960.

### Paired t-Test

When data come in pairs (before/after measurements, matched subjects), the **paired t-test** applies the one-sample t-test to the differences $d_i = x_{i,\text{after}} - x_{i,\text{before}}$:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

This is equivalent to testing $H_0: \mu_d = 0$ and is more powerful than an independent two-sample test because it controls for subject-level variability. Each subject serves as its own control, removing between-subject variance.

### Effect Size: Cohen's d

The standardized effect size for a one-sample t-test is:

$$d = \frac{\bar{x} - \mu_0}{s}$$

Convention classifies $|d| = 0.2$ as small, $|d| = 0.5$ as medium, and $|d| = 0.8$ as large. Reporting effect sizes alongside p-values gives a complete picture of both statistical and practical significance.

### Assumptions

1. **Independence**: Observations are independent of each other
2. **Normality**: The population (or differences for paired test) is approximately normal. The t-test is fairly robust to this assumption for $n \geq 30$ due to the CLT
3. **No extreme outliers**: Heavy outliers can inflate $s$ and distort results. Consider robust alternatives like the Wilcoxon signed-rank test when outliers are present

### Applications

**Comparing model performances**: Given $k$-fold cross-validation scores, a paired t-test can determine if model A significantly outperforms model B by testing whether the mean score difference is zero.

**Feature significance**: Testing whether a feature's coefficient differs significantly from zero in a regression model uses t-tests (this is what the standard regression output reports).

**Pre/post analysis**: Evaluating whether a system change (new algorithm, updated data pipeline) significantly altered a metric by comparing matched before-and-after measurements.