# <span style="font-size: 20px;">Epsilon-Greedy Action Selection</span>

<span style="font-size: 14px;">The $\epsilon$-greedy policy is the simplest knob for trading off **exploration** and **exploitation** in a multi-armed bandit or reinforcement learning agent. With probability $1 - \epsilon$ it **exploits** by picking the action with the highest estimated value, and with probability $\epsilon$ it **explores** by picking a uniformly random action. This single parameter $\epsilon \in [0, 1]$ controls how much the agent is willing to sacrifice short-term reward to gather information about uncertain actions.</span>

---

## <span style="font-size: 16px;">The Explore-Exploit Dilemma</span>

<span style="font-size: 14px;">A bandit agent faces $A$ actions (arms), each with an unknown true mean reward $\mu_a$. The agent only ever observes the reward of the arm it actually pulls, so its value estimates $Q[a]$ are noisy and based on limited samples. This creates a fundamental tension:</span>

* <span style="font-size: 14px;">**Exploitation:** pull the arm that currently looks best to maximize immediate reward. The risk is that the current best is only best because of lucky early samples, and the true best arm was undersampled.</span>
* <span style="font-size: 14px;">**Exploration:** pull a different arm to refine its estimate, accepting a likely worse immediate reward in exchange for information that may pay off later.</span>

<span style="font-size: 14px;">A purely greedy agent ($\epsilon = 0$) commits permanently to whichever arm happened to look best after the first few pulls, and can lock onto a suboptimal arm forever. A purely random agent ($\epsilon = 1$) learns every arm's value accurately but never acts on that knowledge. $\epsilon$-greedy interpolates between these extremes.</span>

---

## <span style="font-size: 16px;">The Policy Distribution</span>

<span style="font-size: 14px;">Let $a^*$ be the greedy action, the index of the maximum value estimate. The $\epsilon$-greedy policy assigns probability mass as follows:</span>

$$
\pi(a) = \begin{cases} (1 - \epsilon) + \dfrac{\epsilon}{A} & \text{if } a = a^* \\[6pt] \dfrac{\epsilon}{A} & \text{if } a \neq a^* \end{cases}
$$

<span style="font-size: 14px;">The intuition behind the two terms for the greedy action:</span>

* <span style="font-size: 14px;">With probability $1 - \epsilon$ the agent chooses to exploit, and the exploit branch always selects $a^*$. This contributes $1 - \epsilon$ to $\pi(a^*)$.</span>
* <span style="font-size: 14px;">With probability $\epsilon$ the agent explores uniformly over all $A$ actions, and $a^*$ is one of them. This contributes $\epsilon / A$ to every action, including $a^*$.</span>

<span style="font-size: 14px;">Summing these two channels gives $(1 - \epsilon) + \epsilon / A$ for the greedy action, while every non-greedy action receives only the uniform exploration share $\epsilon / A$. A quick sanity check confirms the distribution normalizes:</span>

$$
\left[(1 - \epsilon) + \frac{\epsilon}{A}\right] + (A - 1)\frac{\epsilon}{A} = (1 - \epsilon) + \frac{A\epsilon}{A} = 1
$$

<span style="font-size: 14px;">When the max value is tied across several actions, the convention here is to break ties by lowest index: $a^* = \min\{ a : Q[a] = \max_b Q[b] \}$. Only that single index gets the exploitation bonus; the other tied actions are treated as ordinary non-greedy actions.</span>

---

## <span style="font-size: 16px;">Value Estimation Behind the Policy</span>

<span style="font-size: 14px;">The value estimates $Q[a]$ that the policy ranks are typically maintained by an incremental sample average. After observing reward $R$ for arm $a$ on its $n$-th pull:</span>

$$
Q_{n+1}[a] = Q_n[a] + \frac{1}{n}\left(R - Q_n[a]\right)
$$

<span style="font-size: 14px;">This update is the running mean written in error-correction form: the new estimate moves toward the observed reward by a step proportional to the prediction error $R - Q_n[a]$. The $\epsilon$-greedy policy then acts only on the current $Q$ values; the exploration it injects is what makes those estimates trustworthy, because every arm keeps getting sampled at rate at least $\epsilon / A$.</span>

<span style="font-size: 14px;">For nonstationary problems where the arm means drift over time, the $1/n$ step is replaced by a constant step size $\alpha \in (0, 1]$:</span>

$$
Q_{n+1}[a] = Q_n[a] + \alpha\left(R - Q_n[a]\right)
$$

<span style="font-size: 14px;">This turns the estimate into an exponentially weighted average that forgets old rewards, so the agent can track a changing environment. The choice between $1/n$ and constant $\alpha$ is orthogonal to the exploration parameter $\epsilon$, but the two interact: tracking with constant $\alpha$ usually pairs with a residual $\epsilon_{\min} > 0$ so the agent keeps probing in case the best arm changes.</span>

<span style="font-size: 14px;">**Optimistic initialization** is a complementary trick. Setting all initial $Q[a]$ to an optimistically high value forces even a greedy agent to try every arm at least once, because each pull disappoints the inflated estimate and drives it down toward reality. This injects a burst of early exploration without any randomness, and is sometimes combined with a small $\epsilon$ for the long-run residual exploration.</span>

---

## <span style="font-size: 16px;">Decay Schedules for $\epsilon$</span>

<span style="font-size: 14px;">A fixed $\epsilon$ never stops exploring, which is wasteful once the estimates have converged. In practice $\epsilon$ is annealed over time $t$ so the agent explores aggressively early and exploits later. Common schedules:</span>

* <span style="font-size: 14px;">**Linear decay:** $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 - c\,t)$ shrinks $\epsilon$ at a constant rate down to a floor.</span>
* <span style="font-size: 14px;">**Exponential decay:** $\epsilon_t = \epsilon_{\min} + (\epsilon_0 - \epsilon_{\min}) e^{-t / \tau}$ decays smoothly with time constant $\tau$.</span>
* <span style="font-size: 14px;">**Inverse-time decay:** $\epsilon_t = 1 / t$ (or $c / t$) gives the theoretically motivated rate, since the per-step exploration cost shrinks just fast enough to keep total exploration manageable.</span>

<span style="font-size: 14px;">A small residual $\epsilon_{\min}$ is usually kept so the agent can recover if the environment is nonstationary and the best arm changes over time.</span>

---

## <span style="font-size: 16px;">Regret Analysis</span>

<span style="font-size: 14px;">**Regret** measures the cumulative gap between the reward an oracle that always plays the optimal arm $\mu^*$ would earn and what the agent actually earns over $T$ rounds:</span>

$$
\text{Regret}(T) = T\mu^* - \sum_{t=1}^{T} \mathbb{E}[\mu_{a_t}] = \sum_{a} \Delta_a\, \mathbb{E}[N_a(T)]
$$

<span style="font-size: 14px;">where $\Delta_a = \mu^* - \mu_a$ is the suboptimality gap of arm $a$ and $N_a(T)$ is the number of times it was pulled. Regret is the central currency for comparing exploration strategies: the better the strategy, the slower its regret grows.</span>

<span style="font-size: 14px;">With a **fixed** $\epsilon$, the agent pulls each suboptimal arm with probability at least $\epsilon / A$ on every single round, forever. Therefore $\mathbb{E}[N_a(T)] \approx (\epsilon / A) T$ grows **linearly** in $T$, and so does the regret:</span>

$$
\text{Regret}(T) \approx \frac{\epsilon}{A} T \sum_{a \neq a^*} \Delta_a = \Theta(T)
$$

<span style="font-size: 14px;">Linear regret means the per-round regret never vanishes; the agent keeps paying the exploration tax indefinitely. This is the key weakness of fixed $\epsilon$-greedy. A properly tuned decaying schedule such as $\epsilon_t = c / t$ can achieve $O(\log T)$ regret, matching the logarithmic lower bound that UCB1 and Thompson Sampling reach with smarter, uncertainty-aware exploration that does not waste pulls on arms already known to be bad.</span>

<span style="font-size: 14px;">The Lai and Robbins (1985) lower bound establishes that no consistent bandit algorithm can do better than logarithmic regret asymptotically: any method must pull each suboptimal arm at least $\Omega(\log T / \Delta_a)$ times just to be statistically confident it is suboptimal. Fixed $\epsilon$-greedy overshoots this floor by a factor that grows linearly with $T$, because it never reduces the rate at which it samples losing arms. The decaying-$\epsilon$ schedule is the cheapest way to claw back toward the optimal rate, but it requires knowing roughly how fast to decay; UCB1 and Thompson Sampling reach the same logarithmic order without that tuning, which is the central motivation for the methods that follow in this section.</span>

---

## <span style="font-size: 16px;">Why $\epsilon$-Greedy Is the Baseline</span>

<span style="font-size: 14px;">$\epsilon$-greedy is the default exploration strategy that every other method is measured against. Its appeal is almost entirely practical:</span>

* <span style="font-size: 14px;">**Trivial to implement.** It needs only the value estimates and a single coin flip per step. There is no model of uncertainty, no posterior, no confidence bound to maintain, which is why it dominates introductory treatments and serves as a sanity-check baseline in deep RL where the action-value function is a neural network.</span>
* <span style="font-size: 14px;">**Domain-agnostic.** Because exploration is uniform over actions, the method makes no assumption about reward structure. It works for Bernoulli arms, Gaussian arms, and large discrete action spaces alike.</span>
* <span style="font-size: 14px;">**Composable with function approximation.** In Deep Q-Networks (Mnih et al., 2015), $\epsilon$-greedy is applied directly on the network's Q-outputs with $\epsilon$ annealed from $1.0$ to $0.1$ over the first million frames. The simplicity that makes it weak in the tabular regret sense makes it robust when $Q$ is approximate and noisy.</span>

<span style="font-size: 14px;">Its weakness is that exploration is **undirected**: it spends the same effort re-sampling an arm already known to be terrible as it does on a promising but uncertain arm. UCB1 and Thompson Sampling fix exactly this by steering exploration toward arms with high uncertainty, which is what lets them reach logarithmic rather than linear regret.</span>

---

## <span style="font-size: 16px;">Greedy in the Limit With Infinite Exploration</span>

<span style="font-size: 14px;">The condition that separates a converging $\epsilon$-greedy agent from a permanently suboptimal one is called **GLIE**: Greedy in the Limit with Infinite Exploration. A schedule is GLIE if two properties hold:</span>

* <span style="font-size: 14px;">**Infinite exploration:** every action is selected infinitely often as $t \to \infty$, so $N_a(t) \to \infty$ for all $a$. This guarantees each value estimate $Q[a]$ converges to its true mean $\mu_a$.</span>
* <span style="font-size: 14px;">**Greedy in the limit:** the policy becomes greedy with respect to the converged estimates, $\epsilon_t \to 0$, so eventually the agent stops paying the exploration tax.</span>

<span style="font-size: 14px;">The schedule $\epsilon_t = 1 / t$ satisfies both: the harmonic series $\sum_t 1/t$ diverges (infinite exploration) while $\epsilon_t \to 0$ (greedy in the limit). A fixed $\epsilon$ satisfies the first condition but not the second, which is precisely why it explores forever and incurs linear regret. GLIE is the formal reason a decay schedule is not just a heuristic but a requirement for asymptotic optimality.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Suppose $Q = [0.1,\ 0.5,\ 0.5,\ 0.2]$, so $A = 4$ and $\epsilon = 0.2$. The max value is $0.5$, attained at indices $1$ and $2$; breaking ties by lowest index gives $a^* = 1$.</span>

<span style="font-size: 14px;">1. **Uniform share:** every action gets $\epsilon / A = 0.2 / 4 = 0.05$.</span>

<span style="font-size: 14px;">2. **Greedy bonus:** action $1$ additionally gets $1 - \epsilon = 0.8$, for a total of $0.8 + 0.05 = 0.85$.</span>

<span style="font-size: 14px;">3. **Assemble:** $\pi = [0.05,\ 0.85,\ 0.05,\ 0.05]$. Note action $2$ ties for the max value but is treated as non-greedy, so it gets only the uniform share.</span>

<span style="font-size: 14px;">4. **Check:** $0.05 + 0.85 + 0.05 + 0.05 = 1.0$. Rounding each to 4 decimals leaves the values unchanged.</span>

<span style="font-size: 14px;">As a second example, take $Q = [1.0,\ 0.3,\ 0.7]$ with $\epsilon = 0.3$, so $A = 3$. The unique max is at index $0$, giving $a^* = 0$. The uniform share is $0.3 / 3 = 0.1$. Action $0$ gets $0.7 + 0.1 = 0.8$, and actions $1$ and $2$ each get $0.1$. The result $\pi = [0.8,\ 0.1,\ 0.1]$ again sums to $1$. Notice how raising $\epsilon$ from $0.2$ to $0.3$ flattens the distribution: the greedy action loses mass to the explorers, which is exactly the explore-exploit dial in action.</span>

---

## <span style="font-size: 16px;">Comparison With Other Exploration Schemes</span>

<span style="font-size: 14px;">Placing $\epsilon$-greedy alongside the other strategies in this section clarifies what its uniform exploration costs:</span>

* <span style="font-size: 14px;">**Softmax / Boltzmann:** instead of exploring all non-greedy arms equally, it explores them in proportion to their estimated value, so near-greedy arms are tried more than clearly bad ones. This is value-aware but still ignores uncertainty.</span>
* <span style="font-size: 14px;">**UCB1 (Auer et al., 2002):** explores deterministically by adding an optimism bonus that shrinks as an arm is sampled more. It targets uncertainty directly and achieves $O(\log T)$ regret, where fixed $\epsilon$-greedy is $\Theta(T)$.</span>
* <span style="font-size: 14px;">**Thompson Sampling:** maintains a posterior over each arm's mean and samples from it, so exploration is automatically proportional to remaining uncertainty. It also reaches logarithmic regret with strong empirical performance.</span>

<span style="font-size: 14px;">The throughline across all four methods is the same regret currency: each one is ultimately judged by how slowly $\sum_a \Delta_a\, \mathbb{E}[N_a(T)]$ grows. $\epsilon$-greedy sits at the bottom of that hierarchy precisely because uniform exploration cannot stop pulling arms with large $\Delta_a$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the uniform share on the greedy action.** A common bug sets $\pi(a^*) = 1 - \epsilon$ and leaves out the extra $\epsilon / A$. The probabilities then sum to $1 - \epsilon + (A-1)\epsilon/A < 1$, so the distribution does not normalize and any downstream sampling is biased.</span>
* <span style="font-size: 14px;">**Mishandling ties.** When several actions share the max value, the policy must assign the exploit bonus to exactly one of them (here the lowest index). Splitting the bonus across all tied actions, or giving it to all of them, changes the total mass and the resulting distribution.</span>
* <span style="font-size: 14px;">**Assuming fixed $\epsilon$ converges to optimal behavior.** A constant $\epsilon$ keeps pulling bad arms at rate $\epsilon / A$ forever, producing linear regret. Without a decay schedule the agent never settles on the best arm even after infinite rounds.</span>
* <span style="font-size: 14px;">**Confusing the policy probability with the action taken.** This task returns the full distribution $\pi(a)$, not a single sampled action. Returning an argmax or a one-hot vector ignores the exploration mass entirely.</span>

---