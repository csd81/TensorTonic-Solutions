## <span style="font-size: 20px;">Binomial Distribution</span>

The binomial distribution models the number of successes in a fixed number of independent trials, each with the same probability of success. It is one of the most fundamental discrete distributions in statistics and machine learning.

### Setup and Parameters

A random variable $X \sim \text{Binomial}(n, p)$ counts the number of successes in $n$ independent Bernoulli trials, where each trial succeeds with probability $p$ and fails with probability $q = 1 - p$.

Parameters:
- $n$: number of trials (positive integer)
- $p$: probability of success on each trial ($0 \leq p \leq 1$)

The support is $k \in \{0, 1, 2, \ldots, n\}$.

### Probability Mass Function (PMF)

The probability of observing exactly $k$ successes is:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

Each term has a clear interpretation:
- $\binom{n}{k}$: the number of ways to choose which $k$ of the $n$ trials are successes.
- $p^k$: the probability that those $k$ trials all succeed.
- $(1-p)^{n-k}$: the probability that the remaining trials all fail.

The PMF is bell-shaped and symmetric when $p = 0.5$, right-skewed when $p < 0.5$, and left-skewed when $p > 0.5$.

### Mean and Variance

$$E[X] = np \qquad \text{Var}(X) = np(1-p)$$

The mean follows from linearity of expectation: $X = \sum_{i=1}^n X_i$ where each $X_i \sim \text{Bernoulli}(p)$, so $E[X] = \sum E[X_i] = np$. The variance uses the independence of trials: $\text{Var}(X) = \sum \text{Var}(X_i) = np(1-p)$.

The variance is maximized when $p = 0.5$ and approaches zero as $p \to 0$ or $p \to 1$. This reflects the intuition that outcomes are most uncertain when success and failure are equally likely.

### Connection to the Bernoulli Distribution

The Bernoulli distribution is the special case $\text{Binomial}(1, p)$. A Bernoulli trial has only two outcomes (success/failure, 1/0). The binomial is the sum of $n$ independent Bernoulli trials, making it the natural generalization for counting successes.

### Tail Probability

The **tail probability** $P(X \geq t)$ is the probability of observing at least $t$ successes:

$$P(X \geq t) = \sum_{k=t}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

This is useful for hypothesis testing: if we observe $t$ or more successes, how surprising is this under the null hypothesis? The tail probability quantifies this as a p-value.

### Normal Approximation

For large $n$, the Central Limit Theorem guarantees:

$$\frac{X - np}{\sqrt{np(1-p)}} \xrightarrow{d} \mathcal{N}(0, 1)$$

The rule of thumb is that the approximation is reasonable when $np \geq 5$ and $n(1-p) \geq 5$. A continuity correction (adding $\pm 0.5$ to the boundary value) improves accuracy when using the normal CDF to approximate binomial probabilities.

### Cumulative Distribution Function

The CDF $F(k) = P(X \leq k)$ is the sum of the PMF up to $k$. There is no closed-form expression, but it can be related to the regularized incomplete beta function. In practice, it is computed by summing PMF values or using library functions.

### Applications in Machine Learning

**Click-through rate (CTR) modeling**: Each user impression is a Bernoulli trial (click or no-click). The total number of clicks follows a binomial distribution, enabling confidence intervals for CTR estimates and informing decisions about ad placement.

**A/B testing**: Comparing two variants uses binomial proportions. The number of conversions in each group is binomial, and the test determines whether the observed difference in conversion rates is statistically significant or could arise by chance.

**Dropout regularization**: During training, each neuron is independently retained with probability $p$ (a Bernoulli trial). The number of active neurons in a layer follows $\text{Binomial}(n, p)$. At test time, weights are scaled by $p$ to match the expected activation.

**Binary classification metrics**: With $n$ test samples, the number of correct predictions follows a binomial distribution (under certain assumptions), which allows construction of confidence intervals for accuracy and enables comparison between models.