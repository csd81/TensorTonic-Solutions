## What Is Policy Evaluation?

Policy evaluation is the process of computing the **value function** for a given policy $\pi$. The value function tells us how good it is to be in each state when following the policy.

In Monte Carlo (MC) policy evaluation, we estimate values by **averaging returns** from actual experience, rather than using a model of the environment.

---

## The Value Function

The state-value function $V^\pi(s)$ is the expected return starting from state $s$ and following policy $\pi$:

$$
V^\pi(s) = E_\pi[G_t | S_t = s]
$$

where the return $G_t$ is the sum of discounted rewards:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The discount factor $\gamma \in [0, 1]$ determines how much we value future rewards.

---

## Monte Carlo Approach

Monte Carlo methods learn from **complete episodes** of experience. The key idea:

1. Generate episodes by following the policy
2. For each state visited, record the return that followed
3. Average the returns to estimate the value

**No model required:** We learn directly from experience, not from transition probabilities.

**Episodic tasks only:** Episodes must terminate for us to compute returns.

---

## First-Visit vs Every-Visit MC

**First-Visit MC:**

Only use the return following the **first** visit to state $s$ in each episode.

If an episode visits state $s$ multiple times, only the first occurrence counts.

**Every-Visit MC:**

Use the return following **every** visit to state $s$ in each episode.

If state $s$ appears 3 times in an episode, we get 3 return samples.

Both converge to $V^\pi(s)$ as the number of episodes approaches infinity.

---

## First-Visit MC Algorithm

**Initialize:**
- $V(s) \leftarrow$ arbitrary for all $s$
- $\text{Returns}(s) \leftarrow$ empty list for all $s$

**Repeat for each episode:**

1. Generate episode: $S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T$

2. $G \leftarrow 0$

3. For $t = T-1, T-2, ..., 0$:
   - $G \leftarrow \gamma G + R_{t+1}$
   - If $S_t$ not in $\{S_0, S_1, ..., S_{t-1}\}$:
     - Append $G$ to $\text{Returns}(S_t)$
     - $V(S_t) \leftarrow \text{average}(\text{Returns}(S_t))$

---

## Computing Returns: Worked Example

**Episode:** $S_0 \to S_1 \to S_2 \to \text{Terminal}$

**Rewards:** $R_1 = 1$, $R_2 = 2$, $R_3 = 3$

**Discount factor:** $\gamma = 0.9$

**Computing returns backward:**

$G_2 = R_3 = 3$

$G_1 = R_2 + \gamma G_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7$

$G_0 = R_1 + \gamma G_1 = 1 + 0.9 \times 4.7 = 1 + 4.23 = 5.23$

**Updates:**
- $\text{Returns}(S_2)$.append(3)
- $\text{Returns}(S_1)$.append(4.7)
- $\text{Returns}(S_0)$.append(5.23)

---

## Why Compute Returns Backward?

Computing returns backward from the end of the episode is efficient:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

This recursive relationship allows us to compute all returns in a single backward pass through the episode, using $O(T)$ time instead of $O(T^2)$.

---

## Incremental Mean Update

Instead of storing all returns and computing the average, we can update incrementally:

$$
V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}[G_t - V(S_t)]
$$

where $N(S_t)$ is the number of times state $S_t$ has been visited.

**Equivalently:**

$$
V_{new} = V_{old} + \alpha [G - V_{old}]
$$

where $\alpha = 1/N$ for exact averaging.

---

## Constant Learning Rate

In non-stationary environments or for faster adaptation, use a constant learning rate:

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]
$$

where $\alpha \in (0, 1]$ is fixed.

**Properties:**
- Recent returns have more influence
- Forgets old data exponentially
- Can track changing value functions
- Does not converge to exact average

---

## Detailed Example: Gridworld

**Setup:** 4-state gridworld, policy always moves right

States: A $\to$ B $\to$ C $\to$ Terminal

Reward: +1 at each step, $\gamma = 0.9$

**Episode 1:** A $\to$ B $\to$ C $\to$ T

Returns:
- $G(C) = 1$
- $G(B) = 1 + 0.9(1) = 1.9$
- $G(A) = 1 + 0.9(1.9) = 2.71$

**After Episode 1:**
- $V(A) = 2.71$
- $V(B) = 1.9$
- $V(C) = 1.0$

**Episode 2:** A $\to$ B $\to$ C $\to$ T (same trajectory)

Same returns. After averaging:
- $V(A) = (2.71 + 2.71)/2 = 2.71$
- $V(B) = 1.9$
- $V(C) = 1.0$

Values stabilize because the policy is deterministic.

---

## Handling Stochastic Policies

With a stochastic policy, different episodes may take different paths, leading to different returns from the same state.

**Example:** From state A, policy goes right with 70% probability, left with 30%.

Different episodes will produce different returns. The Monte Carlo average estimates the expected return under this stochastic policy.

---

## Variance in Monte Carlo Estimates

MC estimates can have **high variance** because:

1. Returns depend on entire episode trajectories
2. Long episodes have more variability
3. Stochastic environments add randomness

**Reducing variance:**
- Use more episodes
- Use baseline subtraction (in policy gradient methods)
- Consider TD methods for lower variance (but introducing bias)

---

## Bias in Monte Carlo Estimates

MC estimates are **unbiased**: the expected value of the estimate equals the true value.

$$
E[G_t | S_t = s] = V^\pi(s)
$$

This is because we directly sample actual returns, not bootstrap from other estimates.

---

## Convergence Properties

**First-Visit MC:**
- Unbiased estimator of $V^\pi(s)$
- Converges to $V^\pi(s)$ as visits $\to \infty$
- Sample average converges by Law of Large Numbers

**Every-Visit MC:**
- Also converges to $V^\pi(s)$
- Biased for finite samples (correlated samples within episode)
- Often more data-efficient in practice

---

## Advantages of Monte Carlo Methods

**1. Model-free:**

No need for transition probabilities or reward function.

**2. Unbiased:**

Estimates converge to true values without systematic error.

**3. Works with sampled experience:**

Can learn from real interactions or simulated episodes.

**4. Simple to understand and implement:**

Just average returns.

---

## Disadvantages of Monte Carlo Methods

**1. Requires complete episodes:**

Cannot learn from incomplete trajectories.

**2. High variance:**

Returns can vary significantly between episodes.

**3. Slow learning:**

Updates only happen after episode completion.

**4. Memory:**

May need to store entire episodes before processing.

---

## Monte Carlo vs Dynamic Programming

**Dynamic Programming:**
- Requires full model (transitions, rewards)
- Updates based on successor state values
- Bootstrapping: $V(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$

**Monte Carlo:**
- Model-free, learns from experience
- Updates based on complete returns
- No bootstrapping: $V(s) \approx \text{average of sampled } G_t$

---

## Monte Carlo vs Temporal Difference

**Monte Carlo:**
- Updates after episode ends
- Uses actual complete return $G_t$
- Unbiased, high variance

**Temporal Difference (TD):**
- Updates after each step
- Uses estimated return: $R_{t+1} + \gamma V(S_{t+1})$
- Biased (uses estimate), lower variance

MC and TD represent two ends of a spectrum, with n-step methods in between.

---

## Importance Sampling (Off-Policy MC)

To evaluate policy $\pi$ using data from behavior policy $b$:

$$
V^\pi(s) = E_b\left[\frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)} G_t \Bigg| S_t = s\right]
$$

The importance sampling ratio reweights returns to correct for the different action probabilities.

This allows learning about one policy while following another.