# <span style="font-size: 20px;">Discounted Returns</span>

<span style="font-size: 14px;">The **discounted return** $G_t$ is the total reward an agent accumulates from timestep $t$ until the end of an episode, with rewards further in the future weighted progressively less by a **discount factor** $\gamma$. It is the single quantity that every value function, policy gradient, and Bellman equation is ultimately built on: the agent's entire objective is to maximize the expected return, so getting the return right is the foundation of all of reinforcement learning.</span>

---

## Definition

<span style="font-size: 14px;">For a trajectory of rewards $r_t, r_{t+1}, \ldots, r_{T-1}$, the return at timestep $t$ is the discounted sum of all rewards collected from $t$ onward:</span>

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k \, r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t-1} r_{T-1}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$r_{t+k}$ is the reward received $k$ steps after time $t$. Note the indexing convention used here, the reward at offset $k$ is $r_{t+k}$, so the very first term $\gamma^0 r_t$ is the reward received at the current step</span>
* <span style="font-size: 14px;">$\gamma \in [0, 1]$ is the discount factor that controls how much future rewards are worth relative to immediate ones</span>
* <span style="font-size: 14px;">$T$ is the trajectory length, so the final reward in the sum is $r_{T-1}$ and the offset $k$ runs from $0$ to $T-t-1$</span>

<span style="font-size: 14px;">The return is a **random variable** in a stochastic environment because the reward sequence depends on stochastic transitions and a possibly stochastic policy. Value functions are defined as expectations of $G_t$: the state value is $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ and the action value is $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$. The deterministic per-trajectory return computed here is the inner sample that those expectations average over.</span>

---

## <span style="font-size: 16px;">The Role of $\gamma$</span>

<span style="font-size: 14px;">The discount factor encodes how the agent values delayed reward. Each step further into the future multiplies the reward by another factor of $\gamma$, so a reward $n$ steps away is worth only $\gamma^n$ of its face value. The behavior at the extremes is the clearest way to build intuition:</span>

* <span style="font-size: 14px;">**$\gamma = 0$ (myopic):** $G_t = r_t$. The agent is completely shortsighted and cares only about the immediate reward. Every $\gamma^k$ for $k \geq 1$ is zero, so the entire tail of the trajectory vanishes</span>
* <span style="font-size: 14px;">**$\gamma$ near $1$ (farsighted):** distant rewards retain almost their full weight. The agent plans far ahead and is willing to forgo immediate reward for larger long-term gains</span>
* <span style="font-size: 14px;">**$\gamma = 1$ (undiscounted):** every reward counts equally. This is only well defined for episodic tasks where $T$ is finite, otherwise the sum can diverge</span>

<span style="font-size: 14px;">A useful interpretation is an **effective horizon**: with discount $\gamma$, rewards beyond roughly $\frac{1}{1-\gamma}$ steps contribute little. For $\gamma = 0.9$ the horizon is about $10$ steps; for $\gamma = 0.99$ it is about $100$; for $\gamma = 0.999$ about $1000$. Raising $\gamma$ toward $1$ lengthens the planning horizon but also increases the variance of return estimates, because more noisy future rewards enter the sum and each contributes its own randomness. This horizon-versus-variance tradeoff is one reason $\gamma$ is often set below the value the task nominally calls for.</span>

---

## <span style="font-size: 16px;">Why Discount at All</span>

<span style="font-size: 14px;">There are three independent justifications for discounting, and they reinforce each other:</span>

* <span style="font-size: 14px;">**Mathematical convergence.** For an infinite-horizon task with bounded rewards $|r| \leq R_{\max}$, the geometric series $\sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1-\gamma}$ is finite whenever $\gamma < 1$. Without discounting an infinite stream of positive rewards would sum to infinity and the objective would be undefined, so no policy could be ranked above another</span>
* <span style="font-size: 14px;">**Uncertainty about the future.** A reward we might collect many steps from now is less certain to actually be realized, since the environment is stochastic and the episode may terminate at any point. Discounting expresses a preference for reward we are more confident about, much like an interest rate prices the risk of a future payment</span>
* <span style="font-size: 14px;">**Bias toward near-term reward.** Discounting biases the agent toward collecting reward sooner. Two trajectories with the same total undiscounted reward are ranked by how early the reward arrives, the one that front-loads reward has the higher return</span>

<span style="font-size: 14px;">This near-term bias is a feature, not a bug. It mirrors economic time-value-of-money reasoning and keeps the optimization well posed. It also has a clean probabilistic reading: a constant per-step termination probability of $1 - \gamma$ turns an undiscounted infinite-horizon problem into exactly the discounted formulation, so $\gamma$ can be seen as the probability the episode survives one more step.</span>

---

## <span style="font-size: 16px;">The Recursive Form</span>

<span style="font-size: 14px;">The return satisfies a one-step recursion that is the seed of every Bellman equation. Pulling the first term out of the sum:</span>

$$
G_t = r_t + \gamma \big( r_{t+1} + \gamma r_{t+2} + \cdots \big) = r_t + \gamma \, G_{t+1}
$$

<span style="font-size: 14px;">So the return at time $t$ equals the immediate reward plus $\gamma$ times the return at the next step. This recursion is exactly why returns can be computed in a single backward pass: start from the end of the trajectory and accumulate. It also directly motivates the **Bellman expectation equation** $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$, which replaces the random future return $G_{t+1}$ with its expected value $V^\pi(s')$. In this sense the discounted return is the bridge between a single sampled trajectory and the smooth value functions that dynamic programming solves for.</span>

<span style="font-size: 14px;">The same recursion underlies **temporal-difference learning**. The TD target $r_t + \gamma V(s_{t+1})$ is a one-step bootstrap of $G_t$, and the $n$-step return $r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$ interpolates between the full Monte Carlo return (this problem, with no bootstrap) and a one-step TD estimate. Understanding $G_t$ exactly is what makes those approximations interpretable.</span>

---

## Worked Example (3-step chain)

<span style="font-size: 14px;">Consider a short trajectory with rewards $r_0 = 1$, $r_1 = 2$, $r_2 = 3$ (so $T = 3$) and discount $\gamma = 0.5$. The task asks for $G_t$ for every $t \in \{0, 1, 2\}$. Working backward is the cleanest route:</span>

<span style="font-size: 14px;">1. **$G_2$** (offsets $k = 0$ only): $G_2 = \gamma^0 r_2 = 3$</span>

<span style="font-size: 14px;">2. **$G_1$** (offsets $k = 0, 1$): $G_1 = r_1 + \gamma r_2 = 2 + 0.5 \cdot 3 = 3.5$</span>

<span style="font-size: 14px;">3. **$G_0$** (offsets $k = 0, 1, 2$): $G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.5 \cdot 2 + 0.25 \cdot 3 = 1 + 1 + 0.75 = 2.75$</span>

<span style="font-size: 14px;">The output is the list $[2.75, 3.5, 3.0]$. Notice how the recursion checks out: $G_1 = r_1 + \gamma G_2 = 2 + 0.5 \cdot 3 = 3.5$ and $G_0 = r_0 + \gamma G_1 = 1 + 0.5 \cdot 3.5 = 2.75$. Computing right-to-left with the rule $G_t = r_t + \gamma G_{t+1}$ (seeding $G_T = 0$) is the standard $O(T)$ implementation and avoids recomputing the geometric weights for every $t$.</span>

<span style="font-size: 14px;">Reading off the list, $G_0 = 2.75$ is the value of starting at the front of this exact reward sequence, while $G_2 = 3.0$ exceeds it because the only remaining reward is the large undiscounted $r_2$. A later index is not always smaller than an earlier one, the magnitude depends entirely on the rewards that still lie ahead.</span>

---

## <span style="font-size: 16px;">Effect of Changing $\gamma$</span>

<span style="font-size: 14px;">Using the same rewards $[1, 2, 3]$, the return $G_0$ shifts sharply with the discount:</span>

* <span style="font-size: 14px;">$\gamma = 0$: $G_0 = 1$ (only the immediate reward survives)</span>
* <span style="font-size: 14px;">$\gamma = 0.5$: $G_0 = 2.75$ (the case above)</span>
* <span style="font-size: 14px;">$\gamma = 0.9$: $G_0 = 1 + 0.9 \cdot 2 + 0.81 \cdot 3 = 1 + 1.8 + 2.43 = 5.23$</span>
* <span style="font-size: 14px;">$\gamma = 1$: $G_0 = 1 + 2 + 3 = 6$ (undiscounted total)</span>

<span style="font-size: 14px;">As $\gamma$ rises, the later large reward $r_2 = 3$ contributes more, pulling $G_0$ upward. This is why $\gamma$ is treated as part of the problem specification, not a free hyperparameter: it defines what "best" even means for the agent. Two MDPs that differ only in $\gamma$ can have entirely different optimal policies, one preferring a small immediate reward and the other waiting for a larger delayed one.</span>

---

## <span style="font-size: 16px;">From Return to Objective</span>

<span style="font-size: 14px;">The return is the per-trajectory quantity, but the agent optimizes its **expectation**. For a fixed policy $\pi$ the objective in an episodic task is $J(\pi) = \mathbb{E}_\pi[G_0]$, the expected return from the start state. Maximizing $J$ over policies is what reinforcement learning means. Because $G_0$ is built from the same rewards under different transition outcomes, the expectation marginalizes over every source of randomness: the action sampled from $\pi(\cdot \mid s)$ and the next state sampled from $P(\cdot \mid s, a)$.</span>

For the stated start-state discounted objective, the policy gradient is

$$
\nabla_\theta J=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right].
$$

Here, theta denotes policy parameters, pi is the policy, s and a are the state and action at time t, T is the trajectory length, and the expectation averages trajectories generated by the policy. Gamma is the discount factor and G is the return defined above. The outer discount measures time from the episode start; the discount inside each return measures time from that action. This connection does not change the return values requested by the problem.

---

## <span style="font-size: 16px;">Stochastic Return Example</span>

<span style="font-size: 14px;">Returns generalize cleanly to stochastic outcomes through expectation. Suppose from a state the agent receives reward $r_0 = 1$, then with probability $0.5$ continues to a branch yielding $r_1 = 4$ and with probability $0.5$ to a branch yielding $r_1 = 0$, after which the episode ends. With $\gamma = 0.9$:</span>

* <span style="font-size: 14px;">Good branch return: $G_0 = 1 + 0.9 \cdot 4 = 4.6$</span>
* <span style="font-size: 14px;">Bad branch return: $G_0 = 1 + 0.9 \cdot 0 = 1.0$</span>
* <span style="font-size: 14px;">Expected return: $V = 0.5 \cdot 4.6 + 0.5 \cdot 1.0 = 2.8$</span>

<span style="font-size: 14px;">The deterministic return formula computes each branch's $G_0$; the value function is then the probability-weighted average. This is precisely the structure the Bellman equations formalize, and it shows why the single-trajectory return is the atomic building block: every value, every Q-estimate, and every policy-gradient term is an average of returns like these.</span>

---

## <span style="font-size: 16px;">Complexity and Numerical Notes</span>

<span style="font-size: 14px;">A backward sweep computes all $T$ returns in $O(T)$ time and $O(T)$ space, performing one multiply and one add per step. A naive forward implementation that re-sums the tail for each $t$ costs $O(T^2)$ and is both slower and more error prone. Because each term is bounded by $\gamma^k R_{\max}$, the partial sums are numerically stable for $\gamma < 1$, but the explicit powers $\gamma^k$ underflow to zero once $k$ is large, which is harmless in the recursive form (no powers are stored) but silently truncates the tail in a power-accumulating forward loop.</span>

---

## <span style="font-size: 16px;">Discount Values in Practice</span>

<span style="font-size: 14px;">The choice of $\gamma$ is tied to the time scale on which reward signals matter, and common settings reflect that:</span>

* <span style="font-size: 14px;">**$\gamma = 0.99$** is the default in most deep RL benchmarks (Atari, MuJoCo control). With an effective horizon near $100$ steps it captures fairly long-range consequences while keeping return variance manageable</span>
* <span style="font-size: 14px;">**$\gamma = 0.9$ to $0.95$** suits tasks where reward is fairly immediate and long horizons would only inject noise, such as short control episodes</span>
* <span style="font-size: 14px;">**$\gamma$ very close to $1$ (e.g. $0.999$)** is used when the meaningful payoff is far in the future, as in long board games, but it slows credit assignment and demands more samples to estimate returns reliably</span>

<span style="font-size: 14px;">An alternative to discounting is the **average-reward** formulation, which optimizes the long-run reward rate rather than a discounted sum and is preferred for genuinely continuing tasks with no natural episode boundary. For the finite-trajectory setting here, the discounted return is both well defined and exactly computable, so it is the right tool. The key mental model is that $\gamma$ is not tuning the algorithm, it is defining the agent's preferences over the timing of reward.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one in the reward index.** Different texts write the return with reward $r_{t+k+1}$ (Sutton and Barto) versus $r_{t+k}$ (used here). They differ only in whether the step-$t$ reward is indexed $r_t$ or $r_{t+1}$. Mixing conventions shifts every term by one and produces returns that are off by a factor of $\gamma$. Always match the indexing the problem actually specifies, here the first term is $\gamma^0 r_t$.</span>
* <span style="font-size: 14px;">**Forward summation in $O(T^2)$.** Computing each $G_t$ with its own inner loop over $k$ is quadratic and recomputes the same discounted tails repeatedly. The backward recursion $G_t = r_t + \gamma G_{t+1}$ is linear and numerically cleaner because it never materializes $\gamma^k$ as a separate power.</span>
* <span style="font-size: 14px;">**Accumulating $\gamma^k$ by repeated multiplication.** Maintaining a running power `g *= gamma` inside a forward loop drifts in floating point over long horizons and underflows to $0$ once $\gamma^k$ falls below machine precision, silently dropping the tail of the sum. The recursive form sidesteps this entirely.</span>
* <span style="font-size: 14px;">**Using $\gamma = 1$ on a non-terminating task.** Undiscounted returns are only finite when the episode actually ends. Applying $\gamma = 1$ to an infinite-horizon problem makes $G_t$ diverge and breaks any downstream value estimate built on it.</span>

---