# <span style="font-size: 20px;">Bellman Optimality Equation</span>

<span style="font-size: 14px;">The **Bellman optimality equation** defines the value of a state under an optimal policy as the maximum expected return achievable by picking the best action now and continuing optimally thereafter. Unlike the expectation equation, which evaluates a fixed policy, this equation characterizes the best possible value $V^*$ directly. The single $\max$ over actions makes it **nonlinear**, which is what separates control (finding the best policy) from mere evaluation.</span>

---

## The Equation

<span style="font-size: 14px;">For a state $s$, the optimal value is obtained by choosing the action that maximizes the expected immediate reward plus discounted successor value:</span>

$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \, \big[ R(s, a, s') + \gamma \, V^*(s') \big]
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$\max_a$ selects the single best action rather than averaging over a policy. This is the defining difference from the Bellman expectation equation</span>
* <span style="font-size: 14px;">$P(s' \mid s, a)$ is the transition probability, and the inner sum is the expectation over stochastic successors</span>
* <span style="font-size: 14px;">$R(s, a, s')$ is the expected immediate reward on the $(s, a, s')$ transition</span>
* <span style="font-size: 14px;">$\gamma \in [0, 1]$ discounts the successor value relative to the immediate reward</span>

<span style="font-size: 14px;">The task computes one **synchronous backup**: read the current estimate $V$, and for every state produce $V_{new}(s)$ by maximizing over actions using that same old $V$ on the right-hand side. This is one application of the Bellman optimality operator $T^*$, not iteration to convergence.</span>

---

## <span style="font-size: 16px;">Where It Comes From</span>

<span style="font-size: 14px;">The optimal value is defined as the best achievable expected return, $V^*(s) = \max_\pi V^\pi(s)$. The **principle of optimality** (Bellman, 1957) states that an optimal policy has the property that, whatever the first action and resulting state, the remaining decisions must themselves be optimal from that state. Applying the return recursion $G_t = r_t + \gamma G_{t+1}$ and the fact that continuing optimally from $s'$ yields $V^*(s')$, the best first action is the one maximizing the expected immediate reward plus discounted optimal continuation. That is exactly the optimality equation.</span>

<span style="font-size: 14px;">This decomposition is what makes dynamic programming possible: a global optimization over entire sequences of actions reduces to a local, per-state choice that only needs the optimal values of the immediate successors. The recursion stitches these local choices into a globally optimal policy.</span>

---

## <span style="font-size: 16px;">Why the Max Makes It Nonlinear</span>

<span style="font-size: 14px;">The Bellman expectation equation is linear in $V$: it is a weighted sum, so it can be solved with a single matrix inverse $(I - \gamma P^\pi)^{-1} R^\pi$. The optimality equation replaces the policy-weighted sum $\sum_a \pi(a \mid s) (\cdots)$ with $\max_a (\cdots)$. The max of linear functions is **piecewise linear and convex**, not linear, so there is no matrix inverse that solves it in one shot.</span>

<span style="font-size: 14px;">This nonlinearity is the entire reason iterative methods like value iteration exist. The system $V^* = T^* V^*$ has a unique solution, but it must be reached by repeatedly applying the operator rather than inverting a matrix. The nonlinearity also encodes the key fact of optimal control: the optimal action in a state can flip discontinuously as successor values change, because the argmax is not a smooth function of those values.</span>

---

## <span style="font-size: 16px;">The Inner Expectation Stays Linear</span>

<span style="font-size: 14px;">A common confusion is to think the max applies to the environment's randomness. It does not. The agent chooses the action, so the max is over the agent's controllable choice. The environment's response is still an **expectation**, averaged with the transition probabilities $P(s' \mid s, a)$. Defining the optimal action value</span>

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \, \big[ R(s, a, s') + \gamma \, V^*(s') \big]
$$

<span style="font-size: 14px;">makes this clean: $Q^*(s, a)$ is a linear expectation over successors for a fixed action, and $V^*(s) = \max_a Q^*(s, a)$ applies the max only at the top. The agent maximizes over what it controls and averages over what it does not. Swapping these, taking a max over successor states, would model an adversarial environment, a different problem entirely.</span>

---

## <span style="font-size: 16px;">Extracting the Greedy Policy</span>

<span style="font-size: 14px;">Once $V^*$ is known, the optimal policy is **greedy** with respect to it: in each state, take the action whose backup achieves the max.</span>

$$
\pi^*(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a) \, \big[ R(s, a, s') + \gamma \, V^*(s') \big]
$$

<span style="font-size: 14px;">There is always a **deterministic** optimal policy in a finite MDP, because the max is attained by at least one action, so the agent never needs to randomize. This is a fundamental result: while optimal stochastic policies can exist (when actions tie), a deterministic one is always available. Greedy extraction is a one-step lookahead using the optimal values, which is why solving for $V^*$ effectively solves the control problem.</span>

<span style="font-size: 14px;">A crucial caveat: greedy extraction is only guaranteed optimal when applied to the **true** $V^*$. Acting greedily with respect to an intermediate, not-yet-converged value estimate can produce a suboptimal policy, because a state's apparent best action may change once successor values settle. This is why value iteration runs the backup to convergence before reading off the policy, and why a single backup yields a correct one-step value update but not necessarily the final policy. The policy implied by the optimal values is sometimes called the **greedy policy of $V^*$**, and the equivalence between solving for $V^*$ and finding $\pi^*$ is what makes value-based control work.</span>

---

## Worked Example (2-state, 2-action)

<span style="font-size: 14px;">States $\{0, 1\}$, $\gamma = 0.9$, current estimate $V = [0, 0]$. State $0$ has two actions:</span>

* <span style="font-size: 14px;">Action $a_0$: go to state $1$ with prob $1$, reward $1$. Branch value $= 1 + 0.9 \cdot 0 = 1$</span>
* <span style="font-size: 14px;">Action $a_1$: stay in state $0$ with prob $1$, reward $3$. Branch value $= 3 + 0.9 \cdot 0 = 3$</span>

<span style="font-size: 14px;">State $1$ has one action that stays in state $1$ with reward $2$: branch value $= 2 + 0.9 \cdot 0 = 2$.</span>

<span style="font-size: 14px;">1. **Backup state $0$:** $V_{new}(0) = \max(1, 3) = 3$, achieved by $a_1$</span>

<span style="font-size: 14px;">2. **Backup state $1$:** $V_{new}(1) = 2$</span>

<span style="font-size: 14px;">After one synchronous backup, $V_{new} = [3, 2]$, and the greedy action in state $0$ is $a_1$. Both backups read the old $V$, so state $0$ used $V(1) = 0$, not any updated value. The output of a single optimality backup is $[3.0, 2.0]$.</span>

<span style="font-size: 14px;">If iterated to convergence the values would grow: state $0$ would satisfy $V(0) = \max(1 + 0.9 V(1), \ 3 + 0.9 V(0))$. Solving the self-loop branch gives $V(0) = 3 / (1 - 0.9) = 30$, and indeed $a_1$ remains optimal because $30$ beats the $a_0$ branch. One backup lands at $3$; iteration drives it toward $30$.</span>

---

## <span style="font-size: 16px;">Optimality vs Expectation Side by Side</span>

<span style="font-size: 14px;">The two equations share the same inner expectation but differ in how they combine actions:</span>

* <span style="font-size: 14px;">**Expectation:** $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$, a policy-weighted average. Linear, evaluates a given $\pi$, solvable by matrix inverse</span>
* <span style="font-size: 14px;">**Optimality:** $V^*(s) = \max_a Q^*(s, a)$, a max over actions. Nonlinear, finds the best policy, solvable only iteratively</span>

<span style="font-size: 14px;">A subtle but important point: $V^*$ is not the value of any single fixed action chosen everywhere. It is the value of the policy that picks the locally best action in every state, with each state's choice consistent with the others through the shared $V^*$. This mutual consistency is exactly what the fixed-point equation enforces.</span>

---

## <span style="font-size: 16px;">Stochastic Transition Example</span>

<span style="font-size: 14px;">When an action has stochastic outcomes, the inner expectation matters. Consider state $s$ with $\gamma = 1$ and current values $V = [V(1), V(2)] = [10, 0]$ for two successor states. State $s$ offers:</span>

* <span style="font-size: 14px;">A **safe** action: go to state $2$ with prob $1$, reward $4$. Branch value $= 4 + 1 \cdot 0 = 4$</span>
* <span style="font-size: 14px;">A **risky** action: go to state $1$ with prob $0.4$ (reward $0$) or state $2$ with prob $0.6$ (reward $0$). Branch value $= 0.4(0 + 10) + 0.6(0 + 0) = 4$</span>

<span style="font-size: 14px;">Both branches average to $4$, so $V_{new}(s) = \max(4, 4) = 4$ and the actions tie. Lowering the risky action's success probability to $0.3$ gives its branch value $0.3 \cdot 10 = 3 < 4$, so the safe action wins. The max correctly compares the **expected** outcomes of each action, not their best-case outcomes, which is why the inner sum must be a probability-weighted average before the max is applied.</span>

---

## <span style="font-size: 16px;">Relation to Value Iteration</span>

<span style="font-size: 14px;">A single optimality backup is the atomic step of **value iteration**, which simply applies $T^*$ repeatedly until the values stop changing: $V_{k+1} = T^* V_k$. Because $T^*$ is a contraction, the sequence converges geometrically to $V^*$ regardless of the initial $V_0$, typically taken as all zeros. The greedy policy extracted from the converged $V^*$ is optimal. Understanding this one backup in isolation, exactly the quantity this problem computes, is therefore the prerequisite for understanding the full algorithm: value iteration is nothing more than this backup run to a fixed point.</span>

---

## <span style="font-size: 16px;">Complexity of One Backup</span>

<span style="font-size: 14px;">One synchronous optimality backup evaluates every action's expected value for every state, then maxes. With a dense model this is $O(S^2 A)$ time, the same as one expectation backup, because the max over $A$ actions is no more expensive than a weighted sum over $A$ actions. Space is $O(S)$ for the value snapshot. The nonlinearity adds no asymptotic cost per backup; its price is paid in needing iteration rather than a single matrix solve, since no closed-form inverse exists for the max operator.</span>

---

## <span style="font-size: 16px;">Contraction and Uniqueness</span>

<span style="font-size: 14px;">The optimality operator $T^*$ is a **$\gamma$-contraction** in the max norm, just like the expectation operator:</span>

$$
\| T^* U - T^* V \|_\infty \leq \gamma \, \| U - V \|_\infty
$$

<span style="font-size: 14px;">The key technical fact is that the max operator is **non-expansive**: $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$. Composing it with the $\gamma$-discounted expectation yields a contraction. By the Banach fixed-point theorem, $T^*$ has a unique fixed point $V^*$, and repeatedly applying it converges there from any start. A single optimality backup is one such error-shrinking step, reducing the worst-case error by at least a factor of $\gamma$.</span>

<span style="font-size: 14px;">The contraction also bounds how far one backup can be from the answer. After $k$ backups from $V_0$, the error satisfies $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$, so convergence is geometric with rate $\gamma$. A practical stopping rule uses the change between successive iterates: if $\|V_{k+1} - V_k\|_\infty < \epsilon$, then $\|V_{k+1} - V^*\|_\infty < \frac{\gamma \epsilon}{1 - \gamma}$. This is why $\gamma$ close to $1$ both lengthens the horizon and slows convergence, the same factor that controls foresight controls the iteration speed.</span>

---

## <span style="font-size: 16px;">Tie-Breaking and Determinism</span>

<span style="font-size: 14px;">When two actions achieve the same max value, the value $V^*(s)$ is unambiguous (both give the same number), but the greedy action is not unique. Implementations must adopt a deterministic tie-break rule, commonly the **lowest action index**, so that policy extraction is reproducible. Ties are common in symmetric problems and at intermediate iterations before values have separated. The value backup itself is unaffected by the tie, only the recovered policy depends on the rule.</span>

---

## <span style="font-size: 16px;">Terminal States</span>

<span style="font-size: 14px;">Terminal states must keep value $V^*(s_{\text{term}}) = 0$ and should not be backed up, since no further action or reward follows. The usual convention treats a terminal state as self-absorbing with reward $0$, making $0$ a self-consistent fixed point. If a terminal state is accidentally backed up over real actions, its inflated value propagates backward through the max in every predecessor and corrupts the entire solution. Goal states that grant a one-time reward are modeled by placing that reward on the transition **into** the terminal state, not on the terminal state's own (nonexistent) future.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Taking the max over successor states instead of actions.** The max belongs over the agent's actions; the environment's transitions are always averaged with $P(s' \mid s, a)$. Maxing over $s'$ models an adversary and gives the wrong, pessimistic value. The structure is max-over-$a$, expectation-over-$s'$.</span>
* <span style="font-size: 14px;">**In-place updates breaking synchronous semantics.** A single optimality backup must read every right-hand-side value from the old $V$. Updating in place lets a state's max use freshly updated neighbors, producing an order-dependent result that does not match the specified one-backup operator. Snapshot $V$ first.</span>
* <span style="font-size: 14px;">**Inconsistent tie-breaking in policy extraction.** When actions tie for the argmax, an arbitrary or floating-point-sensitive choice yields non-reproducible policies and can oscillate across iterations. Fix a deterministic rule such as the lowest index, and apply it consistently.</span>
* <span style="font-size: 14px;">**Discounting the reward rather than the successor value.** The discount multiplies $V^*(s')$, not $R(s, a, s')$. Writing $\gamma[R + V]$ or $\gamma R + V$ misweights the immediate reward and destroys the contraction guarantee, so iteration no longer converges to the true optimum.</span>

---