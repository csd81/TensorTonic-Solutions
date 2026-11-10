## <span style="font-size: 20px;">A/B Test Significance: Two-Proportion Z-Test</span>

An **A/B test** compares two variants (control vs treatment) to determine whether the treatment produces a statistically significant change in a binary outcome such as conversion rate, click-through rate, or signup rate. It is the standard method for data-driven product decisions.

### Setup

Let $p_c = X_c / n_c$ be the control conversion rate and $p_t = X_t / n_t$ be the treatment conversion rate, where $X$ is the number of conversions and $n$ is the number of visitors.

The hypotheses are:

$$H_0: p_c = p_t \qquad H_a: p_c \neq p_t$$

We test whether the observed difference in proportions is large enough to be unlikely under the assumption that both groups have the same underlying conversion rate.

### Pooled Proportion

Under $H_0$, both groups have the same conversion rate. We estimate it with the **pooled proportion**:

$$\hat{p} = \frac{X_c + X_t}{n_c + n_t}$$

This pools all observations as if there were no difference between groups, giving the best estimate of the common rate under the null. The pooled proportion is used in the denominator of the test statistic because under $H_0$ we assume a single common rate.

### Test Statistic

The two-proportion z-statistic is:

$$z = \frac{p_t - p_c}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_c} + \frac{1}{n_t}\right)}}$$

Under $H_0$ and for sufficiently large samples (generally $n\hat{p} \geq 5$ and $n(1-\hat{p}) \geq 5$ for each group), $z \sim N(0, 1)$. The p-value for a two-tailed test is $p = 2 \cdot [1 - \Phi(|z|)]$.

### Lift

The **lift** quantifies the relative improvement:

$$\text{lift} = \frac{p_t - p_c}{p_c}$$

A lift of 0.05 means the treatment increased the conversion rate by 5% relative to control. Lift communicates practical significance - a statistically significant result with 0.1% lift may not justify the engineering cost of implementation. Always consider minimum detectable effect (MDE) during planning.

### Confidence Interval for the Difference

A $(1-\alpha)$ CI for $p_t - p_c$ (not using the pooled proportion, since we want to estimate the actual difference):

$$p_t - p_c \pm z_{\alpha/2} \sqrt{\frac{p_c(1-p_c)}{n_c} + \frac{p_t(1-p_t)}{n_t}}$$

This interval is more informative than the binary reject/fail-to-reject decision because it shows the plausible range of the true difference. If the CI is [0.001, 0.030], you know the effect is real but small.

### Sample Size Planning

To detect an effect of size $\delta = p_t - p_c$ with power $1 - \beta$ at significance level $\alpha$:

$$n \approx \frac{(z_{\alpha/2} + z_\beta)^2 \cdot [p_c(1-p_c) + p_t(1-p_t)]}{\delta^2}$$

Running an underpowered test wastes resources - you are unlikely to detect a real effect even if it exists. For a typical web experiment with baseline conversion of 5% and MDE of 1 percentage point, you need roughly 3,800 visitors per group at 80% power and $\alpha = 0.05$.

### Practical vs Statistical Significance

A result can be statistically significant but practically meaningless (tiny effect, huge sample) or practically important but statistically insignificant (meaningful effect, insufficient sample). Always consider both. The confidence interval bridges this gap by showing the range of plausible effect sizes.

### Sequential Testing and Peeking

Repeatedly checking results during an experiment inflates the false positive rate. If you check once per day for 30 days at $\alpha = 0.05$, the actual false positive rate can exceed 20%. **Sequential testing** methods (e.g., always-valid p-values, group sequential designs) allow periodic checking while maintaining error control. These methods spend the $\alpha$ budget across interim analyses.

### Applications

**Product experimentation**: Tech companies run thousands of A/B tests annually to optimize user experience, pricing, and features.

**Conversion optimization**: Testing landing pages, email subject lines, checkout flows, and recommendation algorithms.

**Multi-armed bandits**: An alternative to A/B testing that adaptively allocates traffic to better-performing variants, trading off exploration and exploitation rather than waiting for a fixed-horizon test to conclude.