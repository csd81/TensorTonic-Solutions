## The Reinforcement Learning Setup

In reinforcement learning (RL), an **agent** interacts with an **environment** over a sequence of time steps. At each time step $t$:

1. The agent observes the current state $s_t$.
2. The agent takes an action $a_t$ according to its policy.
3. The environment returns a reward $r_t$ and transitions to a new state $s_{t+1}$.

This continues until the episode ends (the agent reaches a terminal state or a maximum number of steps). The full sequence of states, actions, and rewards is called a **trajectory** or **episode**:

$$
s_0, a_0, r_0, \; s_1, a_1, r_1, \; s_2, a_2, r_2, \; \ldots, \; s_T, a_T, r_T
$$

The agent's goal is to learn a policy that maximizes the total reward it accumulates over an episode. But "total reward" needs to be defined carefully, and that is where returns come in.

---

## Returns: Measuring Long-Term Reward

The **return** $G_t$ at time step $t$ is the total accumulated reward from that point onward. The simplest version is the **discounted return**:

$$
G_t = r_t + \gamma \, r_{t+1} + \gamma^2 \, r_{t+2} + \gamma^3 \, r_{t+3} + \cdots = \sum_{k=0}^{T-t} \gamma^k \, r_{t+k}
$$

where $\gamma \in [0, 1]$ is the **discount factor**.

**Why discount?** There are several reasons:

- **Mathematical convenience**: Without discounting ($\gamma = 1$), the return can be infinite in non-terminating environments. Discounting guarantees the sum converges as long as rewards are bounded.
- **Uncertainty**: Future rewards are less certain. A reward far in the future depends on many uncertain transitions and decisions. Discounting reflects this uncertainty.
- **Time preference**: In many real-world settings, sooner is better. A dollar today is worth more than a dollar next year.

**Extreme cases:**

- $\gamma = 0$: The agent is completely myopic. $G_t = r_t$, and only the immediate reward matters.
- $\gamma = 1$: No discounting at all. All future rewards are valued equally. This only works in finite-horizon (episodic) settings where the episode is guaranteed to end.

**Recursive structure**: The return satisfies a natural recursion:

$$
G_t = r_t + \gamma \, G_{t+1}
$$

The return at time $t$ is the immediate reward plus the discounted return from the next step onward. At the final time step $T$, there are no future rewards, so $G_T = r_T$.

---

## The Value Function

The **state value function** $V(s)$ gives the expected return from state $s$ under a given policy:

$$
V(s) = \mathbb{E}[G_t \mid s_t = s]
$$

It answers: "on average, how much total reward will the agent collect if it starts in state $s$ and follows its current policy?"

$V(s)$ is an expectation, meaning it averages over all the randomness in future transitions and actions. In any single episode, the actual return $G_t$ from state $s_t$ will differ from $V(s_t)$. Sometimes the agent will do better than expected, sometimes worse.

The value function is estimated by the **critic** in actor-critic methods. It can be learned using temporal-difference (TD) learning, Monte Carlo estimation, or other techniques. For this problem, $V$ is given to you as a precomputed array.

---

## The Advantage Function

The **advantage** at time step $t$ is:

$$
A_t = G_t - V(s_t)
$$

It measures the difference between what actually happened ($G_t$, the realized return) and what was expected ($V(s_t)$, the average return from that state).

**Interpreting the sign:**

- $A_t > 0$ (positive advantage): The actual return exceeded the expected value. Whatever the agent did at time $t$ led to a better-than-average outcome. This action should be reinforced.
- $A_t < 0$ (negative advantage): The actual return fell short of the expected value. The agent did worse than average. This action should be discouraged.
- $A_t = 0$: The outcome was exactly as expected. No signal to adjust the policy.

The advantage does not just tell you whether the return was high or low in absolute terms. It tells you whether it was high or low *relative to what you expected from that state*. This distinction is critical.

---

## Why Not Just Use Returns?

A natural question: why bother subtracting $V(s_t)$? Why not just use $G_t$ directly to update the policy?

The answer is **variance reduction**.

Consider a simple policy gradient algorithm like REINFORCE. The basic gradient update is:

$$
\nabla J \propto \sum_t G_t \cdot \nabla \ln \pi(a_t \mid s_t)
$$

This says: increase the probability of actions that led to high returns, decrease the probability of actions that led to low returns. The problem is that $G_t$ can be large and noisy. If all rewards in the environment are positive (say, between 50 and 100), then $G_t$ is always positive, and the gradient pushes every action's probability up, just by different amounts. The signal is drowned in noise.

Now subtract a **baseline** $b(s_t)$ from the return:

$$
\nabla J \propto \sum_t (G_t - b(s_t)) \cdot \nabla \ln \pi(a_t \mid s_t)
$$

The value function $V(s_t)$ is the optimal baseline (in the sense of minimizing variance). By subtracting it:

- Actions that led to better-than-expected outcomes get positive weight (pushed up).
- Actions that led to worse-than-expected outcomes get negative weight (pushed down).

The gradient now has a clear directional signal rather than an always-positive magnitude that differs only slightly between good and bad actions. This makes learning much faster and more stable.

**Mathematically, subtracting any baseline that depends only on the state does not introduce bias into the gradient estimate. It only reduces variance.** This is a fundamental result in policy gradient theory.

---

## A Concrete Example

Consider an episode with 3 time steps, $\gamma = 1$ (no discounting), and value estimates $V = [0.5, 1.0, 1.5]$:

**t=0**: state=0, reward=1, return = 1 + 2 + 3 = 6, V(0) = 0.5, advantage = 6 - 0.5 = **5.5**

**t=1**: state=1, reward=2, return = 2 + 3 = 5, V(1) = 1.0, advantage = 5 - 1.0 = **4.0**

**t=2**: state=2, reward=3, return = 3, V(2) = 1.5, advantage = 3 - 1.5 = **1.5**

All advantages are positive, meaning at every step the agent did better than expected. The advantage at $t=0$ is the largest because the return accumulated over all remaining steps and the baseline was small.

Now with $\gamma = 0$ (myopic) and $V = [0, 0, 0]$:

**t=0**: reward=1, return=1, V(0)=0, advantage = 1 - 0 = **1.0**

**t=1**: reward=2, return=2, V(1)=0, advantage = 2 - 0 = **2.0**

**t=2**: reward=3, return=3, V(2)=0, advantage = 3 - 0 = **3.0**

With no discounting and zero baselines, the advantage equals the immediate reward at each step. Future rewards do not contribute because $\gamma = 0$.

---

## The Role of the Discount Factor in Advantages

The discount factor $\gamma$ directly controls how far into the future the return looks:

- With high $\gamma$ (close to 1), $G_t$ includes rewards from many future steps. The advantage reflects whether the entire remainder of the episode went well or poorly.
- With low $\gamma$ (close to 0), $G_t$ is dominated by the immediate reward $r_t$. The advantage is mostly about whether the immediate outcome was better than expected.

In practice, moderate values like $\gamma = 0.99$ are common. This means a reward 100 steps in the future is worth $0.99^{100} \approx 0.366$ of its nominal value, so it still contributes meaningfully but is discounted relative to immediate rewards.

---

## Advantage in Actor-Critic Methods

The advantage function is the bridge between the **actor** (the policy) and the **critic** (the value function) in actor-critic algorithms.

**Actor-Critic architecture:**

- The **critic** learns $V(s)$ by minimizing the error between predicted values and observed returns.
- The **actor** learns the policy $\pi(a \mid s)$ by using the advantage $A_t$ to weight its gradient updates.

This creates a feedback loop:

1. The critic provides increasingly accurate value estimates.
2. Better value estimates produce lower-variance advantage estimates.
3. Lower-variance advantages help the actor learn more effectively.
4. A better policy generates new trajectories, which the critic uses to refine its estimates.

**Major algorithms built on advantages:**

- **A2C / A3C** (Advantage Actor-Critic): The original formulation. Uses $A_t = G_t - V(s_t)$ directly, just as in this problem.
- **PPO** (Proximal Policy Optimization): Uses advantage estimates to compute a clipped surrogate objective that prevents the policy from changing too much in one update. PPO is one of the most widely used RL algorithms in practice.
- **GAE** (Generalized Advantage Estimation): Instead of using the full Monte Carlo return $G_t$, GAE blends different-length return estimates using an exponential weighting controlled by a parameter $\lambda$. This provides a tunable bias-variance tradeoff for the advantage.

---

## Bias-Variance Tradeoff in Advantage Estimation

The version of advantage in this problem, $A_t = G_t - V(s_t)$, uses the **Monte Carlo return** $G_t$. This has an important property: it is **unbiased**. The actual return $G_t$ is a direct sample of what happened, with no approximation.

However, it can have **high variance** because $G_t$ depends on every reward from time $t$ to the end of the episode. A single unlucky transition far in the future can make $G_t$ very different from one episode to the next, even from the same starting state.

An alternative is the **TD (temporal difference) advantage**:

$$
A_t^{\text{TD}} = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This only looks one step ahead and uses the value function to estimate the rest. It has lower variance (because it does not depend on the full trajectory) but introduces **bias** (because $V$ is an approximation and may be inaccurate).

GAE generalizes both: with $\lambda = 1$ it reduces to the Monte Carlo advantage (unbiased, high variance), and with $\lambda = 0$ it reduces to the TD advantage (biased, low variance). Choosing $\lambda$ in between gives a practical tradeoff.

---

## Where Advantage Computation Shows Up

**Game-playing agents**: AlphaGo, OpenAI Five (Dota 2), and other game-playing systems use advantage-based policy gradient methods. The advantage tells the agent which moves were better than its current strategy and by how much.

**Robotics**: Policy gradient methods with advantage estimation train robots to walk, grasp objects, and perform complex manipulation tasks. The advantage signal helps separate the effect of a single motor command from the overall difficulty of the task.

**Language model fine-tuning (RLHF)**: Reinforcement Learning from Human Feedback uses PPO (which relies on advantages) to fine-tune language models. The advantage measures whether a particular response was better or worse than what the model typically produces.

**Resource management**: RL agents that manage server allocation, network routing, or energy systems use advantage-based updates to learn policies that maximize throughput or minimize cost.