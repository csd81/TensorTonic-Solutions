# <span style="font-size: 20px;">Policy Iteration</span>

<span style="font-size: 14px;">**Policy iteration** solves a finite Markov decision process by alternating two steps until the policy stops changing: **policy evaluation**, which computes the value function of the current policy, and **policy improvement**, which makes the policy greedy with respect to those values. It converges to the optimal value function $V^*$ and an optimal deterministic policy $\pi^*$ in a finite number of iterations, a guarantee value iteration does not share.</span>

---

## The Two Steps

<span style="font-size: 14px;">**Policy evaluation** computes $V$ for the current deterministic policy $\pi$ by sweeping the Bellman expectation backup until the values settle:</span>

$$
V[s] \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma \, V[s'] \right]
$$

<span style="font-size: 14px;">This sweep repeats until $\max_s |V_{\text{new}}[s] - V[s]| < \texttt{eval\_tol}$. Because the policy is fixed, there is no max here, the equation is linear in $V$.</span>

<span style="font-size: 14px;">**Policy improvement** then greedifies the policy with respect to the freshly computed values:</span>

$$
\pi_{\text{new}}(s) \leftarrow \arg\max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \, V[s'] \right]
$$

<span style="font-size: 14px;">Ties in the argmax are broken by selecting the lowest action index. The outer loop alternates these two steps and terminates when no state changes its action. The output is the pair $(V, \pi)$ with values rounded to 4 decimals and integer action indices.</span>

---

## <span style="font-size: 16px;">The Idea</span>

<span style="font-size: 14px;">Policy iteration separates the two concerns that value iteration fuses together. Evaluation answers "how good is my current plan?" exactly, and improvement answers "given that, what is the locally best action everywhere?" Each improvement produces a strictly better policy (or signals optimality), so the algorithm climbs a staircase of policies, each provably no worse than the last, until it reaches the top.</span>

<span style="font-size: 14px;">The crucial structural fact is that the space of deterministic policies in a finite MDP is **finite**, with $A^S$ members. Since every improvement step yields a different and strictly better policy until convergence, and a policy can be revisited only if it failed to improve (which means it is optimal), the algorithm must terminate after finitely many iterations. In practice it converges in remarkably few outer iterations, often a handful even for large state spaces.</span>

<span style="font-size: 14px;">The two phases play complementary roles that mirror the two Bellman equations. Evaluation solves the linear Bellman **expectation** equation for the fixed current policy, giving an exact reading of that policy's worth. Improvement applies a single Bellman **optimality** style backup, the max over actions, to turn those readings into a better policy. Policy iteration is thus a disciplined dialogue between the two equations, where evaluation supplies trustworthy values and improvement spends them on a better plan.</span>

---

## <span style="font-size: 16px;">The Policy Improvement Theorem</span>

<span style="font-size: 14px;">The correctness of the whole method rests on one result. Define the action value of taking $a$ once then following $\pi$:</span>

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \, V^\pi(s') \right]
$$

<span style="font-size: 14px;">The **policy improvement theorem** states that if a new policy $\pi'$ satisfies $Q^\pi(s, \pi'(s)) \geq V^\pi(s)$ for every state, then $V^{\pi'}(s) \geq V^\pi(s)$ for every state. In words: if at each state the new action looks at least as good under the old values, the new policy is genuinely at least as good everywhere.</span>

<span style="font-size: 14px;">The greedy policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$ trivially satisfies the precondition, since the max is at least the value of the action $\pi$ itself would have taken. So greedification can never make the policy worse. If greedification changes no action, then $V^\pi = \max_a Q^\pi$, which is exactly the Bellman optimality equation, so $\pi$ is optimal. This is why "no action changed" is a valid and exact stopping criterion.</span>

---

## <span style="font-size: 16px;">Why the Improvement Is Strict</span>

<span style="font-size: 14px;">If the greedy policy differs from the current one in at least one state, the improvement is **strict**: $V^{\pi'}(s) > V^\pi(s)$ for at least that state, with no state getting worse. The argument unrolls the improvement theorem's inequality:</span>

$$
V^\pi(s) \leq Q^\pi(s, \pi'(s)) \leq Q^{\pi'}(s, \pi'(s)) = V^{\pi'}(s)
$$

<span style="font-size: 14px;">repeatedly applying the bootstrap. Strict improvement is what rules out cycling: the algorithm cannot revisit a policy it has already left, because values only increase. Combined with the finiteness of the policy space, this guarantees the staircase has a top and the loop ends.</span>

---

## Worked Example (2-state, 2-action)

<span style="font-size: 14px;">States $\{0, 1\}$, $\gamma = 0.9$, deterministic transitions. State $0$: $a_0$ stays in $0$ (reward $1$), $a_1$ moves to $1$ (reward $0$). State $1$: $a_0$ stays in $1$ (reward $5$), $a_1$ moves to $0$ (reward $0$). Start with the policy $\pi = [a_0, a_1]$ (stay in $0$, leave $1$).</span>

<span style="font-size: 14px;">1. **Evaluate $\pi = [a_0, a_1]$:** state $0$ stays in $0$: $V(0) = 1 + 0.9 V(0)$. State $1$ moves to $0$: $V(1) = 0 + 0.9 V(0)$. Solving: $V(0) = 1/(1-0.9) = 10$, $V(1) = 0.9 \cdot 10 = 9$. So $V = [10, 9]$</span>

<span style="font-size: 14px;">2. **Improve:** state $0$: $Q(0, a_0) = 1 + 0.9 \cdot 10 = 10$, $Q(0, a_1) = 0 + 0.9 \cdot 9 = 8.1$, so keep $a_0$. State $1$: $Q(1, a_0) = 5 + 0.9 \cdot 9 = 13.1$, $Q(1, a_1) = 0 + 0.9 \cdot 10 = 9$, so switch to $a_0$. New policy $\pi = [a_0, a_0]$ (changed)</span>

<span style="font-size: 14px;">3. **Evaluate $\pi = [a_0, a_0]$:** $V(0) = 1 + 0.9 V(0) = 10$; $V(1) = 5 + 0.9 V(1) = 50$. So $V = [10, 50]$</span>

<span style="font-size: 14px;">4. **Improve:** state $0$: $Q(0, a_0) = 1 + 0.9 \cdot 10 = 10$, $Q(0, a_1) = 0.9 \cdot 50 = 45$, so switch to $a_1$. State $1$: $Q(1, a_0) = 5 + 0.9 \cdot 50 = 50$ beats $a_1$, keep $a_0$. New policy $\pi = [a_1, a_0]$ (changed)</span>

<span style="font-size: 14px;">5. **Evaluate $\pi = [a_1, a_0]$:** $V(1) = 5 + 0.9 V(1) = 50$; $V(0) = 0.9 \cdot 50 = 45$. So $V = [45, 50]$</span>

<span style="font-size: 14px;">6. **Improve:** state $0$: $Q(0, a_1) = 0.9 \cdot 50 = 45$ beats $Q(0, a_0) = 1 + 0.9 \cdot 45 = 41.5$, keep $a_1$. State $1$: keep $a_0$. No action changed, so stop</span>

<span style="font-size: 14px;">The result is $V^* = [45, 50]$ and $\pi^* = [a_1, a_0]$, matching the value-iteration answer for the same MDP. Notice the policy converged in three improvement steps, and each evaluation was solved exactly via the linear self-loop equations.</span>

---

## <span style="font-size: 16px;">The Algorithm Step by Step</span>

<span style="font-size: 14px;">1. **Initialize** an arbitrary deterministic policy $\pi$, for example $\pi(s) = 0$ (lowest action index) for every state, and values $V(s) = 0$</span>

<span style="font-size: 14px;">2. **Policy evaluation:** repeatedly sweep $V[s] \leftarrow \sum_{s'} P(s' \mid s, \pi(s))[R(s, \pi(s), s') + \gamma V[s']]$ over all states until $\max_s |V_{\text{new}}[s] - V[s]| < \texttt{eval\_tol}$</span>

<span style="font-size: 14px;">3. **Policy improvement:** for each state compute $Q(s, a)$ for every action and set $\pi_{\text{new}}(s) = \arg\max_a Q(s, a)$, breaking ties toward the lowest index</span>

<span style="font-size: 14px;">4. **Check stability:** if $\pi_{\text{new}} = \pi$ (no state changed its action), stop and return $(V, \pi)$; otherwise set $\pi \leftarrow \pi_{\text{new}}$ and go back to step 2</span>

<span style="font-size: 14px;">The values from the final evaluation are already consistent with the final stable policy, so no extra evaluation is needed after the loop ends.</span>

---

## <span style="font-size: 16px;">Stochastic Transition Example</span>

<span style="font-size: 14px;">Improvement compares expected action values, so stochastic outcomes are handled inside the $Q$ computation. Suppose during an improvement step a state $s$ has values-so-far $V = [V(1), V(2)] = [20, 4]$, $\gamma = 0.9$, and two actions:</span>

* <span style="font-size: 14px;">$a_0$: deterministic to state $2$, reward $6$. $Q(s, a_0) = 6 + 0.9 \cdot 4 = 9.6$</span>
* <span style="font-size: 14px;">$a_1$: to state $1$ with prob $0.5$ (reward $0$), to state $2$ with prob $0.5$ (reward $0$). $Q(s, a_1) = 0.5(0.9 \cdot 20) + 0.5(0.9 \cdot 4) = 9 + 1.8 = 10.8$</span>

<span style="font-size: 14px;">Improvement picks $a_1$ since $10.8 > 9.6$, even though $a_0$ has the larger immediate reward. The greedy step weighs the full discounted expectation of each action, not the one-step reward, which is exactly what makes the improved policy provably better under the current values.</span>

---

## <span style="font-size: 16px;">Evaluation: Iterative vs Exact</span>

<span style="font-size: 14px;">Policy evaluation can be done two ways. The **iterative** form sweeps the expectation backup until $\texttt{eval\_tol}$ is met, costing $O(S^2)$ per sweep (for a fixed policy each state has one action). The **exact** form solves the linear system $V = (I - \gamma P^\pi)^{-1} R^\pi$ directly in $O(S^3)$, since with a fixed policy the Bellman expectation equation is linear.</span>

<span style="font-size: 14px;">The problem specifies the iterative form with an $\texttt{eval\_tol}$ stopping test, which is the standard choice when $S$ is large enough that the cubic solve is expensive. A key practical point: evaluation does not need to run to machine precision. Even a loose tolerance still yields a value good enough for improvement to make progress, which leads to the modified algorithm below.</span>

---

## <span style="font-size: 16px;">Relation to Value Iteration and Modified Policy Iteration</span>

<span style="font-size: 14px;">Value iteration is the extreme case of policy iteration where evaluation is **truncated to a single sweep** before improving. Full policy iteration sits at the other extreme, evaluating to convergence each time. Between them lies **modified policy iteration**, which runs a fixed small number of evaluation sweeps per improvement, often the most efficient choice in practice.</span>

* <span style="font-size: 14px;">**Policy iteration:** few outer iterations, each expensive (full evaluation). Best when evaluation is cheap relative to the number of policies</span>
* <span style="font-size: 14px;">**Value iteration:** many cheap sweeps, no separate evaluation. Best when the state space is large and full evaluation is costly</span>
* <span style="font-size: 14px;">**Modified policy iteration:** a tunable middle ground that usually dominates both on wall-clock time</span>

---

## <span style="font-size: 16px;">Why Convergence Is Finite, Not Just Asymptotic</span>

<span style="font-size: 14px;">Value iteration converges asymptotically: the values approach $V^*$ geometrically but, strictly speaking, reach it only in the limit, so a tolerance is needed to stop. Policy iteration is different in kind. The object that converges is the **policy**, which lives in a finite set, and each iteration moves to a strictly better member of that set. There is no infinite tail: once the greedy policy equals the current policy, the Bellman optimality equation holds and the answer is exact.</span>

<span style="font-size: 14px;">This is the deeper reason policy iteration usually needs only a handful of outer iterations. Each improvement is a discrete jump that can reshuffle many actions at once, whereas a value-iteration sweep only nudges values by one step of lookahead. The cost is that every jump requires fully evaluating a policy, which is why the per-iteration work is heavier. The trade between few-expensive-iterations and many-cheap-sweeps is the central design choice between the two algorithms.</span>

---

## <span style="font-size: 16px;">Warm-Starting Evaluation</span>

<span style="font-size: 14px;">A practical optimization is to start each policy evaluation from the value function of the previous policy rather than from zero. Because consecutive policies differ in only a few states, their value functions are close, so warm-starting cuts the number of evaluation sweeps sharply. The improvement theorem guarantees the new policy's value dominates the old one's, so the previous values are a lower bound and a sound starting point. This is safe as long as evaluation still runs to its $\texttt{eval\_tol}$, since the goal is the value of the new policy, not the old one.</span>

---

## <span style="font-size: 16px;">Complexity and Convergence Rate</span>

<span style="font-size: 14px;">Each evaluation sweep is $O(S^2)$ for a dense model (one action per state), and each improvement is $O(S^2 A)$ since it maxes over all actions. The number of evaluation sweeps per iteration depends on $\texttt{eval\_tol}$ and $\gamma$. The number of **outer** iterations is typically very small, and policy iteration is known to converge at least as fast as value iteration in iteration count, often dramatically faster, because each improvement makes a discrete jump to a strictly better policy rather than a continuous nudge of values.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Not re-initializing or carrying stale values into evaluation incorrectly.** Each evaluation must compute the value of the current policy. Warm-starting evaluation from the previous policy's values is fine and speeds convergence, but the sweep must still run until $\texttt{eval\_tol}$ is met, otherwise the improvement step sees values that do not reflect the current policy and may pick wrong actions or fail to converge.</span>
* <span style="font-size: 14px;">**Inconsistent tie-breaking causing oscillation.** When two actions tie in the argmax, an unstable choice can make the policy flip back and forth between equally-good actions forever, so the "no action changed" test never fires. A fixed lowest-index rule makes ties deterministic and guarantees termination.</span>
* <span style="font-size: 14px;">**Using the max in policy evaluation.** Evaluation must use the fixed policy action $\pi(s)$, not a max over actions. Slipping a max into the evaluation backup turns it into value iteration and breaks the clean separation, often producing values that do not match the policy being improved.</span>
* <span style="font-size: 14px;">**Comparing rounded values for the argmax or the change test.** Rounding to 4 decimals is for output only. Using rounded values inside the evaluation tolerance check or the improvement argmax can cause spurious action flips or premature stopping. Keep the internal computation in full precision.</span>

---