# <span style="font-size: 20px;">Gaussian Mixture Models (EM)</span>

<span style="font-size: 14px;">A Gaussian Mixture Model assumes that the data is generated from a mixture of</span> $k$ <span style="font-size: 14px;">Gaussian distributions. The EM algorithm iteratively estimates the parameters of these distributions by alternating between computing soft assignments (E-step) and updating parameters (M-step).</span>

---

## <span style="font-size: 16px;">EM Algorithm</span>

### <span style="font-size: 16px;">E-step (Expectation)</span>

<span style="font-size: 14px;">Compute the responsibility of each component for each data point:</span>

$$
r_{ij} = \frac{\pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j,\, \Sigma_j)}{\displaystyle\sum_{l=1}^{k} \pi_l \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_l,\, \Sigma_l)}
$$

### <span style="font-size: 16px;">M-step (Maximization)</span>

<span style="font-size: 14px;">Update parameters using the soft assignments:</span>

$$
N_j = \sum_{i=1}^{n} r_{ij}, \quad \pi_j = \frac{N_j}{n}
$$

$$
\boldsymbol{\mu}_j = \frac{1}{N_j}\sum_{i=1}^{n} r_{ij} \mathbf{x}_i
$$

$$
\Sigma_j = \frac{1}{N_j}\sum_{i=1}^{n} r_{ij} (\mathbf{x}_i - \boldsymbol{\mu}_j)(\mathbf{x}_i - \boldsymbol{\mu}_j)^\top
$$

---

## <span style="font-size: 16px;">Covariance Matrix Types</span>

<span style="font-size: 14px;">The parameterization of the covariance matrices significantly affects model capacity, computational cost, and the risk of overfitting:</span>

- <span style="font-size: 14px;">**Full covariance**: each component has its own unconstrained $d \times d$ covariance matrix. Parameters per component: $\frac{d(d+1)}{2}$ (symmetric). Total covariance parameters: $k \cdot \frac{d(d+1)}{2} \approx k \cdot d^2$. Can capture arbitrarily shaped elliptical clusters but requires the most data to estimate reliably.</span>
- <span style="font-size: 14px;">**Diagonal covariance**: $\Sigma_j = \text{diag}(\sigma_{j1}^2, \ldots, \sigma_{jd}^2)$. Parameters: $k \cdot d$. Assumes features are independent within each component. Clusters are axis-aligned ellipses.</span>
- <span style="font-size: 14px;">**Spherical covariance**: $\Sigma_j = \sigma_j^2 I$. Parameters: $k$. Each component is a sphere in feature space. This is the most constrained form.</span>
- <span style="font-size: 14px;">**Tied covariance**: all components share a single covariance matrix $\Sigma$. Parameters: $\frac{d(d+1)}{2}$ (independent of $k$). Useful when cluster shapes are similar but locations differ.</span>

---

## <span style="font-size: 16px;">GMM vs. K-Means</span>

- <span style="font-size: 14px;">K-Means uses hard assignments (each point belongs to exactly one cluster). GMM uses soft assignments (each point has a probability of belonging to each cluster)</span>
- <span style="font-size: 14px;">K-Means assumes spherical clusters. GMM can model elliptical clusters via covariance matrices</span>
- <span style="font-size: 14px;">K-Means is a special case of GMM where covariances are fixed to</span> $\sigma^2 I$ <span style="font-size: 14px;">and</span> $\sigma \to 0$

<span style="font-size: 14px;">**Formal connection**: when all components share a spherical covariance $\Sigma_j = \sigma^2 I$ with equal mixing weights, the responsibility $r_{ij}$ in the E-step becomes:</span>

$$
r_{ij} = \frac{\exp\bigl(-\|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2 / 2\sigma^2\bigr)}{\displaystyle\sum_l \exp\bigl(-\|\mathbf{x}_i - \boldsymbol{\mu}_l\|^2 / 2\sigma^2\bigr)}
$$

<span style="font-size: 14px;">As $\sigma \to 0$, this softmax becomes a hard argmin, assigning each point to its nearest centroid - exactly K-Means.</span>

---

## <span style="font-size: 16px;">Connection to ELBO and Variational Inference</span>

<span style="font-size: 14px;">EM can be understood through the lens of variational inference. The log-likelihood of the observed data $\mathbf{X}$ under a latent variable model is:</span>

$$
\begin{aligned}
\log p(\mathbf{X} | \theta) &= \underbrace{\mathbb{E}_{q(Z)}[\log p(\mathbf{X}, Z | \theta)] - \mathbb{E}_{q(Z)}[\log q(Z)]}_{\text{ELBO}(q, \theta)} \\
&\quad + \text{KL}(q(Z) \| p(Z | \mathbf{X}, \theta))
\end{aligned}
$$

<span style="font-size: 14px;">where $Z$ represents the latent cluster assignments and $q(Z)$ is a variational distribution.</span>

- <span style="font-size: 14px;">**E-step**: set $q(Z) = p(Z | \mathbf{X}, \theta)$ (the true posterior over assignments). This makes the KL divergence zero, so the ELBO equals the log-likelihood.</span>
- <span style="font-size: 14px;">**M-step**: maximize the ELBO with respect to $\theta$ while holding $q$ fixed. This increases the log-likelihood.</span>

<span style="font-size: 14px;">This perspective explains why EM monotonically increases the log-likelihood and why it converges to a local maximum (not necessarily the global one).</span>

---

## <span style="font-size: 16px;">The Singularity Problem</span>

<span style="font-size: 14px;">A critical failure mode of GMM occurs when a Gaussian component "collapses" onto a single data point. If a component mean lands exactly on a data point and the covariance shrinks to zero, the likelihood of that point becomes infinite:</span>

$$\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \Sigma) \propto |\Sigma|^{-1/2} \to \infty \quad \text{as } \Sigma \to 0$$

<span style="font-size: 14px;">This means the log-likelihood is unbounded above, and EM can diverge. Prevention strategies:</span>

- <span style="font-size: 14px;">**Covariance regularization**: add $\epsilon I$ to each covariance matrix after every M-step, enforcing a minimum eigenvalue. This is the most common fix.</span>
- <span style="font-size: 14px;">**Minimum component size**: if a component's effective count drops below a threshold, reinitialize it randomly.</span>
- <span style="font-size: 14px;">**Bayesian priors**: place an inverse-Wishart prior on the covariance matrices, which prevents collapse by penalizing near-singular matrices.</span>

---

## <span style="font-size: 16px;">Convergence</span>

<span style="font-size: 14px;">EM monotonically increases the log-likelihood at each step. It converges to a local maximum, which depends on initialization. Like K-Means, multiple restarts with different seeds help find better solutions.</span>

---

## <span style="font-size: 16px;">Numerical Issues</span>

- <span style="font-size: 14px;">**Singular covariances**: if a component collapses onto a single point, the covariance becomes singular. Adding a small regularization term</span> $\epsilon I$ <span style="font-size: 14px;">prevents this</span>
- <span style="font-size: 14px;">**Underflow**: Gaussian densities can be very small for high-dimensional data. Working in log-space or using the log-sum-exp trick helps</span>

---

## <span style="font-size: 16px;">Model Selection</span>

- <span style="font-size: 14px;">**BIC (Bayesian Information Criterion)**:</span> $\text{BIC} = -2 \ln L + p \ln n$ <span style="font-size: 14px;">where</span> $p$ <span style="font-size: 14px;">is the number of parameters. Lower BIC is better</span>
- <span style="font-size: 14px;">**AIC (Akaike Information Criterion)**:</span> $\text{AIC} = -2 \ln L + 2p$
- <span style="font-size: 14px;">Both balance model fit against complexity to choose</span> $k$

<span style="font-size: 14px;">**Counting parameters for BIC/AIC**: for a GMM with $k$ components, $d$ dimensions, and full covariance:</span>

- <span style="font-size: 14px;">Means: $k \cdot d$</span>
- <span style="font-size: 14px;">Covariances: $k \cdot \frac{d(d+1)}{2}$</span>
- <span style="font-size: 14px;">Mixing weights: $k - 1$ (they sum to 1)</span>
- <span style="font-size: 14px;">Total: $p = k \cdot d + k \cdot \frac{d(d+1)}{2} + (k-1)$</span>

<span style="font-size: 14px;">BIC penalizes complexity more heavily than AIC (especially for large datasets) and tends to select simpler models.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why not just use K-Means?**</span>
  <span style="font-size: 14px;">A: GMM provides probability estimates, handles non-spherical clusters, and gives a principled density estimate. K-Means is faster but less flexible.</span>

- <span style="font-size: 14px;">**Q: What if a component gets no data?**</span>
  <span style="font-size: 14px;">A: Its responsibility becomes zero and the M-step divides by zero. Regularization and reinitialization help. Some implementations remove empty components.</span>

- <span style="font-size: 14px;">**Q: Can EM get stuck?**</span>
  <span style="font-size: 14px;">A: Yes, at local maxima. Use multiple random initializations or K-Means initialization for the means.</span>

---