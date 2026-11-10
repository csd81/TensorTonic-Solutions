## <span style="font-size: 20px;">Monte Carlo Estimation of Pi</span>

Monte Carlo methods use randomness to solve problems that might be deterministic in principle. Estimating $\pi$ via random sampling is the classic introduction to this powerful family of techniques that underpins much of modern computational statistics and machine learning. The name comes from the Monte Carlo casino in Monaco, reflecting the role of chance in these methods.

### The Geometric Argument

Consider a unit square $[0, 1] \times [0, 1]$ with a quarter-circle of radius 1 inscribed inside it. The area of the quarter-circle is $\frac{\pi}{4}$, while the area of the square is $1$. If we uniformly sample random points in the square, the probability that a point falls inside the quarter-circle equals the ratio of areas:

$$P(\text{inside}) = \frac{\text{Area of quarter-circle}}{\text{Area of square}} = \frac{\pi}{4}$$

A point $(x, y)$ lies inside the quarter-circle if and only if $x^2 + y^2 \leq 1$. By counting the fraction of random points that satisfy this condition and multiplying by 4, we obtain an estimate:

$$\hat{\pi} = 4 \cdot \frac{\text{points inside}}{\text{total points}}$$

### Convergence and Error Analysis

By the **Law of Large Numbers**, this estimate converges to the true value of $\pi$ as the number of points $n \to \infty$. The convergence rate is $O(1/\sqrt{n})$, meaning that to halve the error, we need four times as many points. This relatively slow convergence is characteristic of naive Monte Carlo methods.

The standard error of the estimate can be derived from the Bernoulli variance. Each point is an independent trial with success probability $p = \pi/4$. The variance of the indicator is $p(1-p)$, so:

$$SE(\hat{\pi}) = 4 \cdot \sqrt{\frac{p(1-p)}{n}} = \frac{4\sqrt{p(1-p)}}{\sqrt{n}}$$

With $p \approx 0.785$, for $n = 10{,}000$ points the standard error is roughly $0.016$, giving about 2 correct decimal digits.

### Variance Reduction Techniques

Several techniques can improve convergence beyond naive sampling:

- **Stratified sampling**: Divide the domain into a grid of subregions and sample a fixed number of points within each cell. This ensures uniform coverage across the domain and reduces the clustering effects that random sampling can produce, lowering variance without additional computational cost.
- **Importance sampling**: Sample more densely from regions that contribute more to the quantity being estimated, then reweight each sample to correct for the non-uniform sampling density. This is especially powerful when the integrand has large variations.
- **Antithetic variables**: For each sample $(x, y)$, also use $(1-x, 1-y)$. These paired samples introduce negative correlation that reduces the overall variance of the estimate.
- **Quasi-Monte Carlo**: Replace pseudo-random sequences with low-discrepancy sequences (Sobol, Halton) that cover the space more evenly, achieving convergence rates up to $O(1/n)$ in favorable cases.

### The Monte Carlo Principle

The deeper principle is that Monte Carlo methods convert integration problems into averaging problems. For any integrable function $f$ over a domain $D$:

$$\int_D f(x) \, dx \approx \frac{|D|}{n} \sum_{i=1}^{n} f(x_i)$$

where $x_i$ are uniformly sampled from $D$ and $|D|$ is the volume of the domain. The pi estimation is a special case where $f$ is the indicator function of the quarter-circle and $|D| = 1$. This framework generalizes to arbitrarily high dimensions, where traditional numerical quadrature methods break down due to the curse of dimensionality but Monte Carlo maintains its $O(1/\sqrt{n})$ rate regardless of dimension.

### Applications in Machine Learning

Monte Carlo methods are pervasive in ML. **Markov Chain Monte Carlo (MCMC)** methods like Metropolis-Hastings and Hamiltonian Monte Carlo enable Bayesian inference for complex models where posterior distributions are analytically intractable. In **reinforcement learning**, Monte Carlo policy evaluation estimates value functions by averaging returns from sampled trajectories. Monte Carlo integration powers the reparameterization trick in variational autoencoders and is used for computing marginal likelihoods in model selection. Even **dropout** in neural networks can be interpreted as approximate Bayesian inference via Monte Carlo sampling of sub-network architectures. In quantitative finance, Monte Carlo simulation is the standard method for pricing complex derivatives and estimating portfolio risk metrics like Value-at-Risk.