<span style="font-size: 14px;">Computing expected values is one of the most fundamental operations in machine learning. The expected loss over a dataset, the expected reward in reinforcement learning, the evidence lower bound (ELBO) in variational autoencoders - all require evaluating integrals of the form $E[f(X)] = \int f(x) p(x) dx$. When this integral has a closed-form solution, we use it. When it does not, we turn to numerical integration.</span>

## <span style="font-size: 14px;">Expected Value as an Integral</span>

<span style="font-size: 14px;">For a continuous random variable $X$ with probability density function $p(x)$, the expected value of a function $f(X)$ is:</span>

$$
E[f(X)] = \int_{-\infty}^{\infty} f(x) \, p(x) \, dx
$$

<span style="font-size: 14px;">When $X \sim N(\mu, \sigma^2)$, the density is:</span>

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

<span style="font-size: 14px;">The Gaussian density decays exponentially fast outside a few standard deviations of the mean. Beyond $\mu \pm 5\sigma$, the density is on the order of $10^{-6}$ of its peak value, so truncating the integral to the interval $[\mu - 5\sigma, \mu + 5\sigma]$ introduces negligible error.</span>

### <span style="font-size: 14px;">The Specific Problem: $E[X^2]$</span>

<span style="font-size: 14px;">For this problem, $f(x) = x^2$, so we compute:</span>

$$
\begin{aligned}
E[X^2] &= \int_{-\infty}^{\infty} x^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) dx
\end{aligned}
$$

<span style="font-size: 14px;">This has a well-known analytical solution. By the identity $E[X^2] = \text{Var}(X) + (E[X])^2$:</span>

$$
E[X^2] = \sigma^2 + \mu^2
$$

<span style="font-size: 14px;">The analytical answer serves as ground truth for verifying the numerical methods. The real value of this exercise is building the tools for cases where no such formula exists - which is the typical situation in machine learning applications.</span>

## <span style="font-size: 14px;">Numerical Integration: The Basic Idea</span>

<span style="font-size: 14px;">Numerical integration (quadrature) approximates a definite integral by evaluating the integrand at a finite set of points and combining those values with appropriate weights:</span>

$$
\int_a^b g(x) \, dx \approx \sum_{i=0}^{n} w_i \, g(x_i)
$$

<span style="font-size: 14px;">Different quadrature methods differ in their choice of evaluation points $x_i$ and weights $w_i$. The goal is to make the approximation as accurate as possible with as few function evaluations as possible.</span>

<span style="font-size: 14px;">For our problem, the integrand is $g(x) = x^2 \cdot p(x)$, and we integrate over $[a, b] = [\mu - 5\sigma, \mu + 5\sigma]$ using $n$ equally-spaced subintervals with step size $h = (b - a) / n$.</span>

## <span style="font-size: 14px;">The Trapezoidal Rule</span>

<span style="font-size: 14px;">The trapezoidal rule approximates the integrand as a piecewise linear function. On each subinterval $[x_i, x_{i+1}]$, the function is approximated by the straight line connecting $(x_i, g(x_i))$ and $(x_{i+1}, g(x_{i+1}))$. The area under this trapezoid is:</span>

$$
\int_{x_i}^{x_{i+1}} g(x) \, dx \approx \frac{h}{2} [g(x_i) + g(x_{i+1})]
$$

<span style="font-size: 14px;">Summing over all $n$ subintervals:</span>

$$
\int_a^b g(x) \, dx \approx \frac{h}{2} [g(x_0) + 2g(x_1) + 2g(x_2) + \cdots + 2g(x_{n-1}) + g(x_n)]
$$

<span style="font-size: 14px;">where $x_i = a + ih$ for $i = 0, 1, \ldots, n$. The interior points have coefficient 2 because each is shared by two adjacent trapezoids.</span>

### <span style="font-size: 14px;">Error Analysis</span>

<span style="font-size: 14px;">The error of the trapezoidal rule over a single interval is:</span>

$$
\text{Error}_{\text{single}} = -\frac{h^3}{12} g''(\xi_i)
$$

<span style="font-size: 14px;">for some $\xi_i \in [x_i, x_{i+1}]$. Summing over $n$ intervals, the total error is:</span>

$$
\text{Error}_{\text{total}} = -\frac{(b-a)h^2}{12} g''(\xi) = O(h^2)
$$

<span style="font-size: 14px;">for some $\xi \in [a, b]$. The trapezoidal rule is second-order accurate: doubling the number of intervals (halving $h$) reduces the error by a factor of 4.</span>

<span style="font-size: 14px;">The trapezoidal rule is exact for linear functions (since a trapezoid perfectly fits a line). For convex functions, it overestimates the integral; for concave functions, it underestimates.</span>

## <span style="font-size: 14px;">Simpson's Rule</span>

<span style="font-size: 14px;">Simpson's rule achieves higher accuracy by fitting a quadratic (parabola) through three consecutive points. On each pair of subintervals $[x_{2i}, x_{2i+2}]$, it uses the values at $x_{2i}$, $x_{2i+1}$, and $x_{2i+2}$:</span>

$$
\int_{x_{2i}}^{x_{2i+2}} g(x) \, dx \approx \frac{h}{3} [g(x_{2i}) + 4g(x_{2i+1}) + g(x_{2i+2})]
$$

<span style="font-size: 14px;">This formula is derived by integrating the unique quadratic polynomial that passes through the three points. Summing over $n/2$ pairs of intervals (where $n$ must be even):</span>

$$
\int_a^b g(x) \, dx \approx \frac{h}{3} [g(x_0) + 4g(x_1) + 2g(x_2) + 4g(x_3) + 2g(x_4) + \cdots + 4g(x_{n-1}) + g(x_n)]
$$

<span style="font-size: 14px;">The pattern of coefficients is: 1, 4, 2, 4, 2, ..., 4, 2, 4, 1. The interior odd-indexed points get coefficient 4, interior even-indexed points get coefficient 2, and the endpoints get coefficient 1.</span>

### <span style="font-size: 14px;">Derivation via Lagrange Interpolation</span>

<span style="font-size: 14px;">To understand where the coefficients 1, 4, 1 come from, consider three equally-spaced points $x_0, x_1 = x_0 + h, x_2 = x_0 + 2h$. The Lagrange interpolating polynomial through $(x_0, g_0), (x_1, g_1), (x_2, g_2)$ is:</span>

$$
\begin{aligned}
P(x) &= g_0 \frac{(x-x_1)(x-x_2)}{(x_0-x_1)(x_0-x_2)} \\
&\phantom{=}+ g_1 \frac{(x-x_0)(x-x_2)}{(x_1-x_0)(x_1-x_2)} \\
&\phantom{=}+ g_2 \frac{(x-x_0)(x-x_1)}{(x_2-x_0)(x_2-x_1)}
\end{aligned}
$$

<span style="font-size: 14px;">Integrating $P(x)$ from $x_0$ to $x_2$ and simplifying yields:</span>

$$
\int_{x_0}^{x_2} P(x) \, dx = \frac{h}{3}(g_0 + 4g_1 + g_2)
$$

<span style="font-size: 14px;">The factor of 4 for the midpoint arises because the parabola gives more weight to the center of the interval, where it has the most information about the function's curvature.</span>

### <span style="font-size: 14px;">Error Analysis</span>

<span style="font-size: 14px;">Despite fitting a quadratic (degree 2), Simpson's rule is exact for polynomials up to degree 3. This "superconvergence" occurs because the symmetry of the three-point formula causes the cubic error term to vanish. The leading error term is:</span>

$$
\text{Error}_{\text{single}} = -\frac{h^5}{90} g^{(4)}(\xi_i)
$$

<span style="font-size: 14px;">Summing over $n/2$ pairs:</span>

$$
\text{Error}_{\text{total}} = -\frac{(b-a)h^4}{180} g^{(4)}(\xi) = O(h^4)
$$

<span style="font-size: 14px;">Simpson's rule is fourth-order accurate: doubling the number of intervals reduces the error by a factor of 16. This means that for smooth functions, Simpson's rule converges much faster than the trapezoidal rule.</span>

## <span style="font-size: 14px;">Comparing the Two Methods</span>

<span style="font-size: 14px;">For the same number of function evaluations ($n + 1$), Simpson's rule is typically far more accurate than the trapezoidal rule. With $n = 100$ on a smooth integrand like a Gaussian-weighted polynomial, the trapezoidal rule might give 4-8 digits of accuracy while Simpson's gives 8-12 digits.</span>

<span style="font-size: 14px;">However, this advantage depends on the integrand being smooth. If the function has discontinuities, sharp corners, or rapid oscillations, both methods may struggle, and the higher-order method does not necessarily win. For our integrand $x^2 \cdot p(x)$, which is infinitely differentiable and decays rapidly, both methods work well, but Simpson's rule is significantly more accurate.</span>

## <span style="font-size: 14px;">Application: ELBO in Variational Autoencoders</span>

<span style="font-size: 14px;">In a variational autoencoder (VAE), the training objective involves the evidence lower bound:</span>

$$
\text{ELBO} = E_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))
$$

<span style="font-size: 14px;">The first term is an expectation over the approximate posterior $q(z|x)$, which typically has no closed form. In practice, this is approximated using Monte Carlo sampling (which is a stochastic form of numerical integration), but the underlying mathematical structure is the same: we need to compute an expected value by integrating over a probability distribution.</span>

<span style="font-size: 14px;">For low-dimensional latent spaces, deterministic quadrature (like trapezoidal or Simpson's rule) can be more efficient than Monte Carlo, which converges at the slow rate $O(1/\sqrt{n})$ regardless of dimension. The trapezoidal rule at $O(1/n^2)$ and Simpson's at $O(1/n^4)$ are much faster in one dimension. However, quadrature scales exponentially with dimension (the "curse of dimensionality"), which is why Monte Carlo dominates for high-dimensional problems.</span>

## <span style="font-size: 14px;">Application: Policy Gradient in RL</span>

<span style="font-size: 14px;">The policy gradient theorem states that the gradient of the expected return is:</span>

$$
\begin{aligned}
\nabla_\theta E[R] &= E_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]
\end{aligned}
$$

<span style="font-size: 14px;">This is another expected value that requires integration over the policy distribution. In continuous action spaces with low-dimensional actions, numerical integration can provide lower-variance gradient estimates than the standard REINFORCE estimator. Gaussian quadrature methods (a more sophisticated family that includes the trapezoidal and Simpson's rules as special cases) are particularly effective here.</span>

## <span style="font-size: 14px;">Implementation Considerations</span>

### <span style="font-size: 14px;">Vectorized Computation</span>

<span style="font-size: 14px;">Both the trapezoidal and Simpson's rules benefit from vectorized implementation. Instead of looping over evaluation points, we can compute all function values at once using numpy's array operations. The evaluation points are $x_i = a + ih$ for $i = 0, \ldots, n$, generated by `np.linspace(a, b, n+1)`. The integrand values $g(x_i) = x_i^2 \cdot p(x_i)$ are computed in one vectorized call. Then the weighted sum uses appropriate coefficient arrays.</span>

### <span style="font-size: 14px;">Numerical Stability of the Gaussian PDF</span>

<span style="font-size: 14px;">When computing $p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$, it is important to use the standard library implementation (`scipy.stats.norm.pdf` or equivalent) rather than coding the formula directly. Direct computation can overflow or underflow for extreme $x$ values. However, since we truncate to $[\mu - 5\sigma, \mu + 5\sigma]$, the exponent stays within $[-12.5, 0]$, so direct computation is safe in this range.</span>

### <span style="font-size: 14px;">Choosing the Number of Intervals</span>

<span style="font-size: 14px;">How many intervals $n$ do we need? For the trapezoidal rule with error $O(h^2)$, where $h = 10\sigma/n$:</span>

$$
\text{Error} \approx \frac{(10\sigma)^3}{12n^2} \max|g''|
$$

<span style="font-size: 14px;">For Simpson's rule:</span>

$$
\text{Error} \approx \frac{(10\sigma)^5}{180n^4} \max|g^{(4)}|
$$

<span style="font-size: 14px;">With $n = 100$ and $\sigma = 1$, the trapezoidal error is roughly $O(10^{-2})$ while Simpson's error is roughly $O(10^{-6})$. For machine-precision accuracy, Simpson's rule with $n = 1000$ is more than sufficient, while the trapezoidal rule would need $n = 10^6$ or more.</span>

## <span style="font-size: 14px;">Worked Example: Standard Normal, $E[X^2]$</span>

<span style="font-size: 14px;">Consider $X \sim N(0, 1)$ and $f(x) = x^2$. The analytical answer is $E[X^2] = 0^2 + 1^2 = 1$. The integrand is $g(x) = x^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$, and we integrate over $[-5, 5]$.</span>

<span style="font-size: 14px;">With a small number of intervals, the trapezoidal rule already gives a reasonable approximation because the endpoint values of the integrand are negligible (the Gaussian density decays rapidly). As $n$ increases, the trapezoidal error decreases quadratically while Simpson's error decreases quartically. This means Simpson's rule converges dramatically faster: for the same number of function evaluations, it achieves several additional digits of accuracy. The difference in convergence rate between the two methods is one of the most striking demonstrations of how higher-order polynomial approximation can improve numerical integration.</span>

## <span style="font-size: 14px;">Beyond Trapezoidal and Simpson's</span>

<span style="font-size: 14px;">More advanced quadrature methods include:</span>

<span style="font-size: 14px;">Gaussian quadrature: chooses both points and weights optimally for polynomial integrands. Gauss-Hermite quadrature is specifically designed for integrals weighted by the Gaussian density, making it ideal for computing expectations under normal distributions.</span>

<span style="font-size: 14px;">Adaptive quadrature: subdivides the interval more finely where the integrand varies rapidly and less finely where it is smooth. This is what `scipy.integrate.quad` implements.</span>

<span style="font-size: 14px;">Monte Carlo integration: uses random samples instead of deterministic points. The error is $O(1/\sqrt{n})$ regardless of dimension, making it the method of choice for high-dimensional integrals despite its slow convergence.</span>

<span style="font-size: 14px;">Understanding the trapezoidal and Simpson's rules provides the foundation for all of these methods. The core idea - approximate the integral by evaluating the function at discrete points and combining with weights - is universal.</span>

## <span style="font-size: 14px;">Why the Truncation to $\mu \pm 5\sigma$ Works</span>

<span style="font-size: 14px;">The Gaussian distribution has extremely thin tails. The probability of $|X - \mu| > k\sigma$ is bounded by:</span>

$$
P(|X - \mu| > k\sigma) \leq \frac{2}{k\sqrt{2\pi}} e^{-k^2/2}
$$

<span style="font-size: 14px;">For $k = 5$: $P(|X - \mu| > 5\sigma) \approx 5.7 \times 10^{-7}$. The contribution of the tail to $E[X^2]$ is bounded by $E[X^2 \cdot \mathbf{1}_{|X-\mu|>5\sigma}]$, which for reasonable $\mu$ and $\sigma$ is negligible compared to the numerical integration error. Even with the extremely accurate Simpson's rule at $n = 1000$, the truncation error is far smaller than the quadrature error.</span>

<span style="font-size: 14px;">This truncation strategy generalizes: for any expectation under a Gaussian, integrating over $\mu \pm 5\sigma$ captures more than 99.99994% of the probability mass. In practice, $\mu \pm 4\sigma$ (99.9937%) is often sufficient, but using $5\sigma$ provides a comfortable margin.</span>