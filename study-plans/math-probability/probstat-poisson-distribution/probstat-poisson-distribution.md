## <span style="font-size: 20px;">Poisson Distribution</span>

The Poisson distribution models the number of events occurring in a fixed interval of time or space, when events happen independently at a constant average rate. It is the go-to distribution for counting rare or sporadic events.

### Definition and PMF

A random variable $X \sim \text{Poisson}(\lambda)$ has the probability mass function:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

where $\lambda > 0$ is both the mean and the rate parameter. Unlike the binomial, the Poisson has no upper bound on $k$ (though probabilities become vanishingly small for large $k$). The distribution is right-skewed for small $\lambda$ and approaches symmetry as $\lambda$ grows.

### Mean and Variance

A remarkable property of the Poisson distribution is that mean and variance are equal:

$$E[X] = \lambda \qquad \text{Var}(X) = \lambda$$

This property, called **equidispersion**, is often used as a diagnostic: if observed data has variance much larger than the mean (**overdispersion**), the Poisson model may be inappropriate, and alternatives like the negative binomial should be considered. Conversely, variance smaller than the mean (**underdispersion**) also signals model misfit.

### Connection to the Binomial

The Poisson arises as a limit of the binomial when:
- $n \to \infty$ (many trials)
- $p \to 0$ (each trial has small probability)
- $np \to \lambda$ (the expected count stays finite)

$$\lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k} = \frac{\lambda^k e^{-\lambda}}{k!}$$

This makes the Poisson the natural model for counting rare events among many opportunities. Whenever we have a large number of independent trials each with a small probability of success, the Poisson approximation is appropriate.

### The Poisson Process

A **Poisson process** is a continuous-time stochastic process where events arrive independently at a constant rate $\lambda$:

- The number of events in any interval of length $t$ follows $\text{Poisson}(\lambda t)$.
- The time between consecutive events follows $\text{Exponential}(\lambda)$.
- Events in non-overlapping intervals are independent.

The Poisson process is the continuous-time analog of the Bernoulli process and serves as a building block for more complex stochastic models.

### CDF and Tail Probabilities

The cumulative distribution function is:

$$F(k) = P(X \leq k) = \sum_{i=0}^{k} \frac{\lambda^i e^{-\lambda}}{i!}$$

The tail probability $P(X > k) = 1 - F(k)$ answers questions like "what is the probability of more than $k$ events?" There is no closed-form expression, so it is computed by summing PMF values.

### Special Case: Zero Events

The probability of observing no events is:

$$P(X = 0) = e^{-\lambda}$$

This decreases exponentially with $\lambda$. For $\lambda = 1$, there is a 36.8% chance of zero events; for $\lambda = 5$, only 0.67%. This probability is important in practice - it represents the chance that nothing happens in the observation period.

### Sum of Poissons

If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$. This **reproductive property** makes the Poisson convenient for aggregating counts from multiple independent sources.

### Applications in Machine Learning

**Web traffic modeling**: The number of requests to a server in a given second often follows a Poisson distribution. This enables capacity planning: $P(X > \text{threshold})$ gives the probability of exceeding server capacity.

**Rate limiting and anomaly detection**: If normal traffic follows $\text{Poisson}(\lambda)$, an observed count far exceeding $\lambda$ (in the tail of the distribution) signals an anomaly, such as a DDoS attack or viral content.

**Natural language processing**: The Poisson distribution models word frequencies in documents. The number of occurrences of a rare word in a text of fixed length approximately follows a Poisson distribution, which is used in topic models and information retrieval.

**Queueing theory**: Arrival processes in queueing systems are typically modeled as Poisson. This determines expected wait times, queue lengths, and server utilization - metrics relevant to load balancing in distributed ML serving systems.