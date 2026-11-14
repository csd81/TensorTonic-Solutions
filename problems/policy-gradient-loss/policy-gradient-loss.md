## The Reinforcement Learning Setup

In reinforcement learning:
- **Agent**: the learner/decision maker
- **Environment**: the world the agent interacts with
- **State**: the current situation
- **Action**: what the agent can do
- **Reward**: feedback from the environment

The goal: learn a policy $\pi(a|s)$ that maximizes cumulative reward.

---

## What Is a Policy?

A policy maps states to actions. Two types:

**Deterministic policy:**

$$
a = \pi(s)
$$
Given state $s$, always take action $a$.

**Stochastic policy:**

$$
\pi(a|s) = P(a|s)
$$
Given state $s$, sample action $a$ from a probability distribution.

Policy gradient methods use stochastic policies parameterized by neural networks.

---

## The Objective

We want to maximize expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

Where:
- $\theta$ are the policy network parameters
- $\tau$ is a trajectory (sequence of states, actions, rewards)
- $R(\tau)$ is the total reward of the trajectory

The challenge: how do we take gradients through sampling actions?

---

## The Policy Gradient Theorem

The gradient of the objective is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]
$$

This is remarkable: we can estimate the gradient by:
1. Sampling trajectories using the current policy
2. Computing log probabilities of the actions taken
3. Weighting by the total reward

No need to differentiate through the environment!

---

## REINFORCE Algorithm

The simplest policy gradient method (Williams, 1992):

**For each episode:**
1. Run the policy, collect trajectory: $s_0, a_0, r_0, s_1, a_1, r_1, ...$
2. Compute return: $R = \sum_t r_t$
3. Compute loss: $L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot R$
4. Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L$

The loss is the negative of the policy gradient objective (we minimize loss, which maximizes reward).

---

## Understanding the Loss

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot R
$$

Breaking this down:

**$\log \pi_\theta(a_t|s_t)$:** How likely was this action under our policy?

**$R$:** How good was the outcome?

**The product:** If $R > 0$ (good outcome), increase probability of actions taken. If $R < 0$ (bad outcome), decrease probability of actions taken.

**The negative sign:** Because we minimize loss but want to maximize reward.

---

## The Credit Assignment Problem

Basic REINFORCE uses total episode reward for all actions:

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot R_{\text{total}}
$$

Problem: early actions get credit/blame for late rewards, even if they were unrelated.

**Solution: reward-to-go**

Use only future rewards for each action:

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot R_t
$$

Where $R_t = \sum_{t'=t}^{T} r_{t'}$ is the sum of rewards from time $t$ onward.

This makes sense: action at time $t$ can only affect future rewards, not past.

---

## Variance Reduction: Baselines

Policy gradients have high variance. A common fix is subtracting a baseline:

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot (R_t - b(s_t))
$$

Where $b(s_t)$ is a baseline (any function of state, not action).

**Why it works:**
- Subtracting a constant does not change the expected gradient
- But it can dramatically reduce variance
- Actions better than average get positive weight, worse get negative

**Common baselines:**
- Average reward over the batch
- Value function $V(s_t)$ (estimated by another network)

---

## Advantage Function

The advantage measures how much better an action is compared to average:

$$
A(s, a) = Q(s, a) - V(s)
$$

Where:
- $Q(s, a)$ is the expected return starting from state $s$, taking action $a$
- $V(s)$ is the expected return starting from state $s$

Using advantage as the weight:

$$
L = -\sum_t \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)
$$

This is the foundation of Actor-Critic methods and PPO.

---

## Numerical Example

Episode with 3 timesteps:

**Timestep 0:** state = s0, action = a0, log_prob = -0.5, reward = 1
**Timestep 1:** state = s1, action = a1, log_prob = -1.0, reward = 0
**Timestep 2:** state = s2, action = a2, log_prob = -0.3, reward = 10

Reward-to-go:
- R0 = 1 + 0 + 10 = 11
- R1 = 0 + 10 = 10
- R2 = 10

Loss (no baseline):

$$
L = -[(-0.5)(11) + (-1.0)(10) + (-0.3)(10)]
$$

$$
L = -[5.5 + 10 + 3] = -18.5
$$

Since all returns are positive, all actions get reinforced.

---

## The Gradient

$$
\nabla_\theta L = -\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t
$$

For a neural network policy with softmax output:
- $\nabla_\theta \log \pi = \nabla_\theta z_a - \sum_a \pi(a|s) \nabla_\theta z_a$
- Similar to cross-entropy gradient, but weighted by return

The gradient:
- Increases probability of actions with positive advantage
- Decreases probability of actions with negative advantage
- Magnitude depends on how good/bad the action was

---

## Entropy Regularization

Adding entropy bonus to encourage exploration:

$$
L = -\sum_t [\log \pi_\theta(a_t|s_t) \cdot A_t + \beta H(\pi(\cdot|s_t))]
$$

Where $H$ is entropy and $\beta$ is a small coefficient (e.g., 0.01).

**Why entropy helps:**
- Prevents premature convergence to deterministic policy
- Encourages exploration of different actions
- Makes training more stable

---

## Common Issues and Solutions

**High variance:**
- Use baselines (value function)
- Use larger batches
- Use advantage estimation (GAE)

**Sample inefficiency:**
- Policy gradient uses each sample once
- Off-policy methods (PPO, SAC) reuse samples
- Importance sampling corrections

**Unstable updates:**
- Large policy updates can be catastrophic
- PPO clips the policy ratio
- TRPO uses KL divergence constraint

---

## Where Policy Gradient Is Used

- **Game playing**: Atari, board games, video games
- **Robotics**: learning motor control
- **Recommendation systems**: sequential decision making
- **Language models**: RLHF (Reinforcement Learning from Human Feedback)
- **Any sequential decision problem** with delayed rewards

Modern methods like PPO, SAC, and TD3 all build on the policy gradient foundation with various improvements for stability and sample efficiency.