## What Is SARSA?

SARSA is an **on-policy** temporal difference (TD) control algorithm that learns action-value functions $Q(s, a)$. The name comes from the quintuple used in each update: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

Unlike Q-learning, SARSA learns the value of the policy being followed, including exploration.

---

## On-Policy vs Off-Policy

**On-policy (SARSA):**
- Learns the value of the policy being used to collect data
- Behavior policy = target policy
- Updates use the action actually taken next

**Off-policy (Q-learning):**
- Learns the value of the optimal policy regardless of behavior
- Behavior policy $\neq$ target policy
- Updates use the greedy action (max Q)

---

## The SARSA Update Rule

Given the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$

**Key difference from Q-learning:** Uses $Q(S_{t+1}, A_{t+1})$ instead of $\max_{a'} Q(S_{t+1}, a')$.

---

## Understanding the Update

**TD Target:**

$$
\text{target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})
$$

This is the estimated return using:
- Actual reward $R_{t+1}$
- Discounted Q-value of the **actual** next state-action pair

**TD Error:**

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

The update moves $Q(S_t, A_t)$ toward the target.

---

## SARSA Algorithm

**Initialize:** $Q(s, a)$ arbitrarily for all $s, a$

**Repeat for each episode:**

1. Initialize state $S$
2. Choose action $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)

3. **Repeat for each step:**
   - Take action $A$, observe reward $R$ and next state $S'$
   - Choose action $A'$ from $S'$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
   - Update: $Q(S, A) \leftarrow Q(S, A) + \alpha[R + \gamma Q(S', A') - Q(S, A)]$
   - $S \leftarrow S'$, $A \leftarrow A'$

4. Until $S$ is terminal

---

## Worked Example

**Setup:**
- Current state $S_t$, action $A_t$
- $Q(S_t, A_t) = 5.0$
- Reward: $R_{t+1} = 2$
- Next state $S_{t+1}$
- Next action chosen by $\epsilon$-greedy: $A_{t+1}$
- $Q(S_{t+1}, A_{t+1}) = 7.0$
- Learning rate: $\alpha = 0.1$
- Discount factor: $\gamma = 0.9$

**Step 1: Compute TD target**

$$
\text{target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) = 2 + 0.9 \times 7 = 2 + 6.3 = 8.3
$$

**Step 2: Compute TD error**

$$
\delta = 8.3 - 5.0 = 3.3
$$

**Step 3: Update Q-value**

$$
Q(S_t, A_t) \leftarrow 5.0 + 0.1 \times 3.3 = 5.0 + 0.33 = 5.33
$$

---

## SARSA with Epsilon-Greedy

SARSA is typically used with an $\epsilon$-greedy policy:

**Action selection:**

$$
A = \begin{cases} \arg\max_a Q(S, a) & \text{with probability } 1 - \epsilon \\ \text{random action} & \text{with probability } \epsilon \end{cases}
$$

**Important:** Both $A_t$ and $A_{t+1}$ are selected using the same $\epsilon$-greedy policy. The update reflects the value of this exploratory policy.

---

## What SARSA Learns

SARSA learns $Q^\pi$ where $\pi$ is the $\epsilon$-greedy policy.

**This is not $Q^*$!**

The learned Q-values account for the fact that the agent will sometimes explore (take random actions). This affects the value estimates.

**Implication:** SARSA may learn "safer" policies that account for potential exploration mistakes.

---

## The Cliff Walking Example

Classic example showing SARSA vs Q-learning difference:

**Environment:**
- Grid with a cliff along the bottom edge
- Agent starts at bottom-left, goal at bottom-right
- Falling off cliff gives large negative reward and resets to start

**Q-learning:** Learns optimal path (along cliff edge) because it evaluates greedy policy. During training with $\epsilon$-greedy, occasionally falls off cliff.

**SARSA:** Learns safer path (away from cliff) because it evaluates the $\epsilon$-greedy policy, accounting for occasional random actions near the cliff.

---

## SARSA vs Q-Learning Update

**SARSA:**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

**Q-Learning:**

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]
$$

The only difference is in the bootstrap term:
- SARSA: Uses $Q(S_{t+1}, A_{t+1})$ (action actually taken)
- Q-learning: Uses $\max_{a'} Q(S_{t+1}, a')$ (best action)

---

## Terminal States

When $S_{t+1}$ is a terminal state:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} - Q(S_t, A_t)]
$$

There is no next action, so no bootstrap term. The target is just the final reward.

---

## Expected SARSA

A variant that uses the expected Q-value under the policy:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t)\right]
$$

**Advantages:**
- Lower variance than SARSA (no randomness in next action)
- Still on-policy (uses $\pi$)
- With greedy $\pi$, reduces to Q-learning

**For $\epsilon$-greedy:**

$$
\sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') = (1-\epsilon) \max_{a'} Q(S_{t+1}, a') + \frac{\epsilon}{|A|} \sum_{a'} Q(S_{t+1}, a')
$$

---

## Convergence

SARSA converges to $Q^\pi$ under standard conditions:

**1. All state-action pairs visited infinitely often**

**2. Learning rate schedule:**

$$
\sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty
$$

**3. GLIE (Greedy in the Limit with Infinite Exploration):**

The policy must converge to greedy as exploration decreases.

With these conditions, SARSA converges to $Q^*$.

---

## Learning Rate and Stability

**High $\alpha$ (e.g., 0.5):**
- Fast adaptation
- May be unstable
- Good for non-stationary problems

**Low $\alpha$ (e.g., 0.01):**
- Slow learning
- More stable
- Better asymptotic accuracy

**Typical:** $\alpha \in [0.1, 0.5]$ for tabular SARSA

---

## Advantages of SARSA

**1. On-policy safety:**
Learns values that account for exploration, leading to safer behavior.

**2. More realistic evaluation:**
Values reflect actual expected returns under the behavior policy.

**3. Well-suited for:**
- Risk-sensitive applications
- When exploration can be dangerous
- Online learning scenarios

---

## Disadvantages of SARSA

**1. May not learn optimal policy:**
Values are for $\epsilon$-greedy policy, not optimal policy.

**2. Exploration affects values:**
Q-values change as $\epsilon$ decays.

**3. Less sample efficient:**
Must follow current policy; cannot reuse old data as effectively.

---

## n-Step SARSA

Extends SARSA to use n-step returns:

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})
$$

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[G_t^{(n)} - Q(S_t, A_t)]
$$

Balances bias (from bootstrapping) and variance (from returns).

---

## SARSA($\lambda$)

Combines all n-step returns using eligibility traces:

$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

**Implementation with eligibility traces:**

1. Initialize traces $e(s, a) = 0$ for all $s, a$
2. On each step:
   - $\delta = R + \gamma Q(S', A') - Q(S, A)$
   - $e(S, A) = e(S, A) + 1$ (or replace/accumulate)
   - For all $s, a$: $Q(s, a) \leftarrow Q(s, a) + \alpha \delta e(s, a)$
   - For all $s, a$: $e(s, a) \leftarrow \gamma \lambda e(s, a)$

---

## When to Use SARSA vs Q-Learning

**Use SARSA when:**
- Safety during learning matters
- Actions have real consequences
- You want to learn the value of exploratory behavior

**Use Q-learning when:**
- You want to learn the optimal policy directly
- Sample efficiency is important
- You can separate behavior and target policies

---

## SARSA in Deep RL

While tabular SARSA is less common in deep RL, the on-policy concept extends:

**Actor-Critic methods:**
- On-policy evaluation of current policy
- Similar spirit to SARSA

**PPO, A2C:**
- On-policy algorithms
- Learn value of current policy

The key insight of SARSA (learning the policy you follow) remains relevant in modern RL.