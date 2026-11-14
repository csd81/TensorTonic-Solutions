## What Is Generalized Advantage Estimation (GAE)?

Generalized Advantage Estimation is a technique for computing **advantage function estimates** that balances bias and variance. It provides a smooth interpolation between high-bias, low-variance estimates and low-bias, high-variance estimates.

Introduced by Schulman et al. (2016), GAE is widely used in policy gradient methods like PPO and TRPO.

---

## The Advantage Function

The advantage function measures how much better an action is compared to the average:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**Interpretation:**
- $A > 0$: Action $a$ is better than average in state $s$
- $A < 0$: Action $a$ is worse than average
- $A = 0$: Action $a$ is exactly average

The advantage is used in policy gradient methods to weight the gradient update.

---

## Why Use Advantages?

Using advantages instead of returns reduces variance in policy gradient updates:

**With returns:**

$$
\nabla_\theta J \approx E\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

**With advantages:**

$$
\nabla_\theta J \approx E\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]
$$

Subtracting the baseline $V(s)$ does not change the expected gradient but reduces variance.

---

## The TD Error (TD Residual)

The one-step TD error is defined as:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This is an estimate of the advantage $A(s_t, a_t)$ with:
- Low variance (uses one step of actual reward)
- High bias (bootstraps from $V(s_{t+1})$)

---

## n-Step Advantage Estimates

We can use different horizons for advantage estimation:

**1-step (TD error):**

$$
\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**2-step:**

$$
\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)
$$

**n-step:**

$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
$$

**$\infty$-step (Monte Carlo):**

$$
\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - V(s_t)
$$

---

## The Bias-Variance Tradeoff

**Short horizon (small n):**
- Low variance: Only a few random rewards involved
- High bias: Heavily relies on value function estimate

**Long horizon (large n):**
- High variance: Many random rewards accumulated
- Low bias: Less reliance on value function estimate

GAE provides a principled way to blend these estimates.

---

## GAE: The Formula

GAE computes a weighted average of all n-step advantage estimates:

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where:
- $\gamma$ is the discount factor
- $\lambda \in [0, 1]$ is the GAE parameter controlling bias-variance tradeoff

---

## Understanding the $\lambda$ Parameter

**$\lambda = 0$:**

$$
\hat{A}_t = \delta_t
$$
Pure TD error. High bias, low variance.

**$\lambda = 1$:**

$$
\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - V(s_t)
$$
Monte Carlo advantage. Low bias, high variance.

**$\lambda \in (0, 1)$:**
Interpolates between TD and MC. Typical values: 0.9 to 0.99.

---

## Recursive Computation

GAE can be computed efficiently using backward recursion:

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
$$

**Algorithm (backward from end of trajectory):**

1. Set $\hat{A}_T = 0$ (or $\delta_{T-1}$ for last step)
2. For $t = T-1, T-2, ..., 0$:
   - Compute $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
   - Compute $\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}$

---

## Worked Example

**Trajectory:** 4 timesteps with rewards and values

- $t=0$: $r_0 = 1$, $V(s_0) = 5$
- $t=1$: $r_1 = 2$, $V(s_1) = 6$
- $t=2$: $r_2 = 3$, $V(s_2) = 4$
- $t=3$: $r_3 = 4$, $V(s_3) = 3$
- Terminal: $V(s_4) = 0$

**Parameters:** $\gamma = 0.9$, $\lambda = 0.8$

---

**Step 1: Compute TD errors**

$\delta_3 = r_3 + \gamma V(s_4) - V(s_3) = 4 + 0.9(0) - 3 = 1$

$\delta_2 = r_2 + \gamma V(s_3) - V(s_2) = 3 + 0.9(3) - 4 = 3 + 2.7 - 4 = 1.7$

$\delta_1 = r_1 + \gamma V(s_2) - V(s_1) = 2 + 0.9(4) - 6 = 2 + 3.6 - 6 = -0.4$

$\delta_0 = r_0 + \gamma V(s_1) - V(s_0) = 1 + 0.9(6) - 5 = 1 + 5.4 - 5 = 1.4$

---

**Step 2: Compute GAE backward**

$\hat{A}_3 = \delta_3 = 1$ (last step)

$\hat{A}_2 = \delta_2 + \gamma\lambda\hat{A}_3 = 1.7 + 0.9(0.8)(1) = 1.7 + 0.72 = 2.42$

$\hat{A}_1 = \delta_1 + \gamma\lambda\hat{A}_2 = -0.4 + 0.9(0.8)(2.42) = -0.4 + 1.74 = 1.34$

$\hat{A}_0 = \delta_0 + \gamma\lambda\hat{A}_1 = 1.4 + 0.9(0.8)(1.34) = 1.4 + 0.96 = 2.36$

---

## Handling Episode Boundaries

At terminal states, there is no next state value:

$$
\delta_{T-1} = r_{T-1} + \gamma \cdot 0 - V(s_{T-1}) = r_{T-1} - V(s_{T-1})
$$

For non-terminal truncation (e.g., max episode length):

$$
\delta_{T-1} = r_{T-1} + \gamma V(s_T) - V(s_{T-1})
$$

Use the "done" flag to distinguish between true termination and truncation.

---

## GAE in Practice

**Typical hyperparameters:**
- $\gamma = 0.99$: Standard discount factor
- $\lambda = 0.95$: Balances bias-variance well

**In PPO:**
1. Collect trajectory using current policy
2. Compute value estimates $V(s_t)$ for all states
3. Compute GAE advantages $\hat{A}_t$
4. Normalize advantages: $\hat{A}_t = \frac{\hat{A}_t - \mu}{\sigma}$
5. Use advantages in policy gradient update

---

## Why GAE Works Well

**Exponential weighting:**

$(\gamma\lambda)^l$ weights decrease exponentially. Recent TD errors matter most, but distant information still contributes.

**Smooth interpolation:**

Unlike choosing a fixed n-step, GAE smoothly blends all horizons.

**Variance reduction:**

For typical $\lambda$ values (0.9-0.99), GAE achieves much lower variance than pure Monte Carlo while maintaining acceptable bias.

---

## Comparison of Advantage Estimators

**TD Error ($\lambda = 0$):**
- Only uses immediate reward and one bootstrap
- Very low variance
- High bias if value function is inaccurate

**Monte Carlo ($\lambda = 1$):**
- Uses complete return
- Unbiased given value function
- High variance from reward randomness

**GAE ($\lambda \in (0.9, 0.99)$):**
- Best of both worlds
- Moderate variance and bias
- Empirically works best in most cases

---

## Normalizing Advantages

After computing GAE, advantages are typically normalized:

$$
\hat{A}_t^{norm} = \frac{\hat{A}_t - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + \epsilon}
$$

**Benefits:**
- Stabilizes policy gradient updates
- Ensures consistent learning rate effectiveness
- Prevents very large or small gradients

---

## GAE with Value Function Fitting

GAE advantages are also used to compute targets for value function updates:

**Value target:**

$$
V_{target}(s_t) = \hat{A}_t + V(s_t) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} + V(s_t)
$$

Or equivalently, using the $\lambda$-return:

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

---

## Computational Complexity

**Time:** $O(T)$ where $T$ is trajectory length
- One backward pass through the trajectory
- Each step: constant time operations

**Space:** $O(T)$
- Store all TD errors (or compute on the fly)
- Store all GAE values

Very efficient for typical trajectory lengths.

---

## Common Implementation Details

**Vectorization:**

Process multiple trajectories in parallel using batch operations.

**Handling varying lengths:**

Pad shorter trajectories and use masks to ignore padding.

**GPU acceleration:**

TD errors and GAE can be computed efficiently on GPU.

---

## GAE vs Other Advantage Methods

**Actor-Critic with TD error:**
- Uses $\delta_t$ directly as advantage
- Simpler but higher bias

**n-step returns:**
- Fixed horizon
- GAE is a soft version with exponential averaging

**V-trace (IMPALA):**
- For off-policy correction
- Uses importance sampling truncation

---

## Theoretical Properties

**Bias:**

GAE introduces bias through the value function approximation. If $V$ is exact, the estimator is unbiased for $\lambda = 1$.

**Variance:**

Variance decreases as $\lambda$ decreases. The variance is bounded by the Monte Carlo variance.

**Convergence:**

Policy gradient with GAE converges to a local optimum under standard conditions.