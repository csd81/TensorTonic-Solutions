## <span style="font-size: 20px;">Discrete Moments</span>

Moments are numerical summaries that characterize the shape of a probability distribution. The first two moments - expected value and variance - are the most commonly used, capturing the center and spread of a distribution respectively.

### Expected Value (First Moment)

The **expected value** (or mean) of a discrete random variable $X$ with values $x_1, x_2, \ldots$ and corresponding probabilities $p_1, p_2, \ldots$ is:

$$E[X] = \sum_{i} x_i \cdot p_i$$

The expected value is a probability-weighted average: it is the long-run average value of $X$ over many independent realizations. Note that $E[X]$ need not be a value that $X$ can actually take. For example, the expected value of a fair die roll is $E[X] = (1+2+3+4+5+6)/6 = 3.5$, which is not a possible outcome.

### Second Moment (Raw)

The **second moment** (or mean of the square) is:

$$E[X^2] = \sum_{i} x_i^2 \cdot p_i$$

This measures the average squared value. By itself, $E[X^2]$ is less intuitive, but it is essential for computing the variance. The second moment is always non-negative and satisfies $E[X^2] \geq (E[X])^2$ by Jensen's inequality (since $x^2$ is convex).

### Variance and Standard Deviation

The **variance** measures the expected squared deviation from the mean:

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

The second form, known as the **computational formula** or **shortcut formula**, is more efficient to compute: calculate $E[X]$ and $E[X^2]$ in a single pass over the data, then subtract. The equivalence follows from expanding $(X - \mu)^2$:

$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

The **standard deviation** $\sigma = \sqrt{\text{Var}(X)}$ has the same units as $X$ and is more interpretable. A standard deviation of 5 means that typical deviations from the mean are around 5 units.

Variance is always non-negative: $\text{Var}(X) \geq 0$, with equality if and only if $X$ is a constant (deterministic).

### Linearity of Expectation

One of the most powerful properties of expectation is **linearity**:

$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

This holds **regardless of whether $X$ and $Y$ are independent**. It is used extensively in algorithm analysis, where we decompose a complex random variable into simpler indicator variables. For example, the expected number of fixed points of a random permutation is 1, proved elegantly by summing $n$ indicator variables each with expectation $1/n$.

### Variance of Sums

For the variance of a sum:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

If $X$ and $Y$ are **independent**, the covariance is zero, so:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

This explains why averaging $n$ independent observations reduces variance by a factor of $n$: $\text{Var}(\bar{X}) = \text{Var}(X)/n$. This is the mathematical basis for the standard error of the mean.

### Moment Generating Functions

The **moment generating function** (MGF) is defined as:

$$M_X(t) = E[e^{tX}] = \sum_{i} e^{t x_i} p_i$$

The $k$-th moment is obtained by differentiating $k$ times and evaluating at $t=0$: $E[X^k] = M_X^{(k)}(0)$. MGFs are useful for proving distributional results: if two random variables have the same MGF, they have the same distribution. The MGF of a sum of independent variables is the product of individual MGFs, which greatly simplifies derivations.

### Applications in Machine Learning

**Reinforcement learning**: The expected reward $E[R] = \sum_r r \cdot P(R=r)$ is the quantity that RL algorithms optimize. Policy gradient methods estimate this expectation through sampling. The variance of returns determines the stability of training and motivates variance reduction techniques like baselines and advantage estimation.

**Risk assessment and decision theory**: In portfolio theory, the mean represents expected return and the variance represents risk. Mean-variance optimization (Markowitz theory) finds the portfolio with the best risk-return tradeoff. Similarly, in decision theory, expected utility $E[U(X)] = \sum_i U(x_i) p_i$ guides optimal decision-making under uncertainty.

**Loss function analysis**: Understanding the expected loss $E[L(Y, \hat{Y})]$ and its variance helps compare models. A model with slightly higher expected loss but much lower variance may be preferable in practice, reflecting the classical bias-variance tradeoff.