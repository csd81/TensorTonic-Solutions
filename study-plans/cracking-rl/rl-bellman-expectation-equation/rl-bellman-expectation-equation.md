# <span style="font-size: 20px;">Bellman Expectation Equation</span>

<span style="font-size: 14px;">The **Bellman expectation equation** expresses the value of a state under a fixed policy $\pi$ as the expected immediate reward plus the discounted value of the successor state. It is the recursive consistency condition that any correct value function must satisfy, and it is the computational core of **policy evaluation**, the step that turns a policy into the value function that scores it.</span>

---

## The Equation

<span style="font-size: 14px;">For a state $s$, the value under policy $\pi$ averages over both the action the policy might take and the next state the environment might transition to:</span>

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \, \big[ R(s, a, s') + \gamma \, V^\pi(s') \big]
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$\pi(a \mid s)$ is the probability the policy assigns to action $a$ in state $s$. The outer sum is the expectation over actions</span>
* <span style="font-size: 14px;">$P(s' \mid s, a)$ is the probability of landing in state $s'$ after taking action $a$ in $s$. The inner sum is the expectation over transitions</span>
* <span style="font-size: 14px;">$R(s, a, s')$ is the expected immediate reward on the $(s, a, s')$ transition</span>
* <span style="font-size: 14px;">$\gamma \in [0, 1]$ is the discount factor, weighting the value of the successor state relative to the immediate reward</span>

<span style="font-size: 14px;">The task computes one **synchronous backup**: it reads the current estimate $V$ and produces $V_{new}(s)$ for every state using that same old $V$ on the right-hand side, without iterating to convergence. A single backup is one application of the Bellman expectation operator.</span>

---

## <span style="font-size: 16px;">Where It Comes From</span>

<span style="font-size: 14px;">Recall that the value function is the expected discounted return $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ and that the return obeys the one-step recursion $G_t = r_t + \gamma G_{t+1}$. Substituting the recursion into the expectation and using linearity:</span>

$$
V^\pi(s) = \mathbb{E}_\pi[\, r_t + \gamma G_{t+1} \mid S_t = s\,] = \mathbb{E}_\pi[\, r_t + \gamma V^\pi(S_{t+1}) \mid S_t = s\,]
$$

<span style="font-size: 14px;">The step replacing $\mathbb{E}[G_{t+1}]$ with $V^\pi(S_{t+1})$ is the **bootstrapping** move, it expresses one value in terms of other values. Expanding the expectation over the policy and the transition dynamics gives back the full nested sum. The equation is therefore not an algorithm in itself, it is an identity the true $V^\pi$ must satisfy. Policy evaluation finds the $V^\pi$ that makes the identity hold everywhere.</span>

---

## <span style="font-size: 16px;">Reading the Two Sums</span>

<span style="font-size: 14px;">The two nested sums encode the two sources of randomness in a Markov decision process, and keeping them straight is the whole skill:</span>

* <span style="font-size: 14px;">**Outer sum over $a$, weighted by $\pi(a \mid s)$.** This averages over the agent's own choices. A deterministic policy collapses this to a single term, $\pi(a \mid s) = 1$ for one action and $0$ for the rest, so only that action's branch survives</span>
* <span style="font-size: 14px;">**Inner sum over $s'$, weighted by $P(s' \mid s, a)$.** This averages over the environment's response. A deterministic environment collapses it to one successor state</span>

<span style="font-size: 14px;">Inside the bracket, $R(s, a, s') + \gamma V^\pi(s')$ is the value of one concrete $(a, s')$ outcome: get the immediate reward, then inherit the discounted value of wherever you landed. The double sum is just the probability-weighted average of these outcome values, which is the textbook definition of an expectation.</span>

---

## <span style="font-size: 16px;">Matrix Form and Exact Solution</span>

<span style="font-size: 14px;">Because the equation is **linear** in $V^\pi$ (there is no max anywhere), it can be written as a linear system. Define the policy-induced transition matrix $P^\pi$ with entries $P^\pi_{ss'} = \sum_a \pi(a \mid s) P(s' \mid s, a)$ and the expected reward vector $R^\pi_s = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) R(s, a, s')$. Then:</span>

$$
V^\pi = R^\pi + \gamma P^\pi V^\pi \quad \Longrightarrow \quad V^\pi = (I - \gamma P^\pi)^{-1} R^\pi
$$

<span style="font-size: 14px;">For $\gamma < 1$ the matrix $I - \gamma P^\pi$ is invertible, so the value function has a unique closed-form solution. This is why policy evaluation always has a well-defined answer. Solving the $S \times S$ system directly costs $O(S^3)$; iterating the backup instead avoids the inverse and is preferred when $S$ is large. The single backup in this problem is one iteration of that iterative scheme.</span>

---

## Worked Example (2-state chain)

<span style="font-size: 14px;">Consider two states $\{0, 1\}$ and a single action so the policy is trivial ($\pi(a \mid s) = 1$). Let $\gamma = 0.9$ and the current estimate be $V = [0, 0]$. From state $0$ the agent goes to state $1$ with probability $1$, earning reward $5$. From state $1$ it stays in state $1$ with probability $1$, earning reward $1$.</span>

<span style="font-size: 14px;">1. **Backup state $0$:** only successor is $s' = 1$, so $V_{new}(0) = 1 \cdot [5 + 0.9 \cdot V(1)] = 5 + 0.9 \cdot 0 = 5$</span>

<span style="font-size: 14px;">2. **Backup state $1$:** only successor is $s' = 1$, so $V_{new}(1) = 1 \cdot [1 + 0.9 \cdot V(1)] = 1 + 0.9 \cdot 0 = 1$</span>

<span style="font-size: 14px;">After one synchronous backup, $V_{new} = [5, 1]$. Crucially, state $0$ used the **old** $V(1) = 0$, not the freshly computed value, that is what synchronous means. The output is $[5.0, 1.0]$.</span>

<span style="font-size: 14px;">If this backup were repeated to convergence, the values would approach the fixed point: state $1$ satisfies $V(1) = 1 + 0.9 V(1)$, giving $V(1) = \frac{1}{1 - 0.9} = 10$, and then $V(0) = 5 + 0.9 \cdot 10 = 14$. The single step lands partway there; iteration closes the gap geometrically.</span>

---

## <span style="font-size: 16px;">Stochastic Policy Example</span>

<span style="font-size: 14px;">The full power of the equation shows when both sums are nontrivial. Consider state $s$ with two actions. The policy is stochastic: $\pi(a_0 \mid s) = 0.7$, $\pi(a_1 \mid s) = 0.3$. Let $\gamma = 0.9$ and current values $V = [V(0), V(1), V(2)] = [0, 10, 4]$ over three states.</span>

* <span style="font-size: 14px;">Action $a_0$ leads to state $1$ with probability $1$, reward $2$: branch value $= 2 + 0.9 \cdot 10 = 11$</span>
* <span style="font-size: 14px;">Action $a_1$ leads to state $1$ with prob $0.5$ (reward $0$) and state $2$ with prob $0.5$ (reward $6$): branch value $= 0.5(0 + 0.9 \cdot 10) + 0.5(6 + 0.9 \cdot 4) = 0.5 \cdot 9 + 0.5 \cdot 9.6 = 9.3$</span>

<span style="font-size: 14px;">Combining over the policy: $V_{new}(s) = 0.7 \cdot 11 + 0.3 \cdot 9.3 = 7.7 + 2.79 = 10.49$. The inner sum handled the environment's randomness for $a_1$; the outer sum mixed the two actions by their policy probabilities. Each layer is a clean expectation, computed innermost first.</span>

---

## <span style="font-size: 16px;">Action-Value Connection</span>

<span style="font-size: 14px;">It is often cleaner to compute the per-action branch values explicitly. Define the action value</span>

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \, \big[ R(s, a, s') + \gamma \, V^\pi(s') \big]
$$

<span style="font-size: 14px;">which is exactly the inner sum, the value of committing to action $a$ once and then following $\pi$. The state value is then the policy-weighted average of these:</span>

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \, Q^\pi(s, a)
$$

<span style="font-size: 14px;">In the example above, $Q^\pi(s, a_0) = 11$ and $Q^\pi(s, a_1) = 9.3$, and $V_{new}(s) = 0.7 \cdot 11 + 0.3 \cdot 9.3 = 10.49$. Decomposing the backup this way separates the two sums into two clear stages and is the natural structure for an implementation: compute every $Q(s, a)$, then collapse over actions with the policy.</span>

---

## <span style="font-size: 16px;">Complexity of One Backup</span>

<span style="font-size: 14px;">A single synchronous sweep visits every state, every action, and every successor once, so the cost is $O(S \cdot A \cdot S) = O(S^2 A)$ time when the transition model is dense, or $O(S \cdot A \cdot \bar{d})$ where $\bar{d}$ is the average number of reachable successors when $P$ is sparse. Space is $O(S)$ for the snapshot of old values plus the new array. The exact linear-algebra solution via $(I - \gamma P^\pi)^{-1}$ costs $O(S^3)$, so iterating cheap $O(S^2 A)$ backups is the standard choice for large state spaces, trading exactness for scalability.</span>

---

## <span style="font-size: 16px;">Synchronous vs Asynchronous Backups</span>

<span style="font-size: 14px;">A backup can use the old values uniformly or the newest available values, and the distinction changes the result of a single step:</span>

* <span style="font-size: 14px;">**Synchronous (Jacobi-style):** compute all $V_{new}(s)$ from a frozen copy of the old $V$, then overwrite. This is the behavior the problem requires. It is order-independent, the same input always gives the same output regardless of the state-visiting order</span>
* <span style="font-size: 14px;">**Asynchronous (Gauss-Seidel-style):** update states in place so later states in the sweep see already-updated earlier values. This often converges faster over many sweeps but makes a single sweep depend on the visiting order</span>

<span style="font-size: 14px;">For this one-backup task, synchronous semantics are essential: read every right-hand-side value from the old array. Accidentally updating in place would let state $0$ in the example above see the new $V(1)$ and produce a different, order-dependent answer.</span>

---

## <span style="font-size: 16px;">Convergence as a Contraction</span>

<span style="font-size: 14px;">Iterating the Bellman expectation backup converges because the operator is a **$\gamma$-contraction** in the max norm. For any two value estimates $U$ and $V$:</span>

$$
\| T^\pi U - T^\pi V \|_\infty \leq \gamma \, \| U - V \|_\infty
$$

<span style="font-size: 14px;">Each backup shrinks the worst-case error by at least a factor of $\gamma$. By the Banach fixed-point theorem there is a unique fixed point, namely $V^\pi$, and repeated application converges to it from any starting estimate. The single backup performed here is one such error-reducing step. Convergence is geometric, so smaller $\gamma$ means faster evaluation but a shorter effective horizon.</span>

---

## <span style="font-size: 16px;">Terminal States and Boundary Conditions</span>

<span style="font-size: 14px;">Terminal (absorbing) states need explicit handling or the backup will produce nonsense. A terminal state by definition yields no further reward, so its value is fixed at $V(s_{\text{term}}) = 0$ and it should never be updated by the backup. The standard convention is to treat a terminal state as transitioning to itself with reward $0$, which makes $V = 0 + \gamma \cdot 0 = 0$ a self-consistent fixed point. If a terminal state is mistakenly given an outgoing transition to a high-value state, its value inflates and that error bleeds into every predecessor through the recursion.</span>

<span style="font-size: 14px;">For finite-horizon problems the boundary condition is different: the value at the final timestep is just the expected immediate reward with no discounted tail, and earlier timesteps fold in successors via the recursion. The infinite-horizon discounted form used here assumes a stationary value function that does not depend on the timestep, which is the right model for the synchronous backup the task specifies.</span>

---

## <span style="font-size: 16px;">Role in the Larger Picture</span>

<span style="font-size: 14px;">The Bellman expectation equation is the **evaluation** half of every dynamic-programming method. Policy iteration alternates full policy evaluation (solving this equation for the current $\pi$) with greedy policy improvement. The action-value form $Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)[R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]$ is the same identity expressed over state-action pairs and underlies SARSA. The key contrast to remember is with the Bellman optimality equation: this equation evaluates a given policy and is linear, whereas optimality maximizes over actions and is nonlinear.</span>

---

## <span style="font-size: 16px;">Why Bootstrapping Helps</span>

<span style="font-size: 14px;">A naive way to evaluate a policy is Monte Carlo: roll out many full trajectories and average the observed returns at each state. That works but is slow and high variance, because each estimate waits for an entire episode to finish and absorbs all the noise along the way. The Bellman expectation equation instead **bootstraps**, it estimates a state's value from the current estimates of its immediate successors, propagating information one step at a time.</span>

<span style="font-size: 14px;">The benefit is that good information spreads through the state space quickly. In the 2-state example, knowing state $1$'s value lets state $0$ be evaluated in one backup without simulating a single full trajectory from state $0$. The cost is that the estimate is only as good as the successor estimates it leans on, which is why a single backup is generally not the final answer and iteration is needed to wash out the initial guess. This trade, lower variance and faster propagation in exchange for bootstrap bias, is the defining feature of dynamic-programming and temporal-difference methods relative to Monte Carlo.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**In-place updates when synchronous is required.** Overwriting $V(s)$ during the sweep lets later computations read freshly updated neighbors, turning a synchronous backup into a Gauss-Seidel step. For a single-backup spec this yields a different, order-dependent result. Snapshot the old $V$ and read every right-hand-side value from the snapshot.</span>
* <span style="font-size: 14px;">**Dropping the policy weighting.** The outer sum must be weighted by $\pi(a \mid s)$, not a plain average or a max over actions. Replacing it with a max silently turns the expectation equation into the optimality equation and computes the wrong quantity for a stochastic policy.</span>
* <span style="font-size: 14px;">**Putting $\gamma$ on the reward instead of the successor value.** The discount multiplies $V^\pi(s')$, not $R(s, a, s')$. Writing $\gamma R + V$ or $\gamma[R + V]$ misweights immediate versus future reward and breaks the contraction property that guarantees convergence.</span>
* <span style="font-size: 14px;">**Misusing the reward indexing $R(s, a, s')$.** When the reward depends on the landing state $s'$, it sits inside the inner sum and is averaged with $P(s' \mid s, a)$. Pulling a state-action reward $R(s, a)$ outside the inner sum is fine, but treating an $(s, a, s')$ reward as if it were constant over $s'$ drops the transition-dependent term and biases the value.</span>

---