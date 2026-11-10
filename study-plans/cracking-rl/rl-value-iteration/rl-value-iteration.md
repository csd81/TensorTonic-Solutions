# <span style="font-size: 20px;">Value Iteration</span>

<span style="font-size: 14px;">**Value iteration** computes the optimal value function $V^*$ of a finite Markov decision process by applying the Bellman optimality backup repeatedly until the values stop changing. It interleaves a single sweep of evaluation and improvement into one update, so it never needs to fully evaluate a policy. Once $V^*$ has converged, the optimal policy $\pi^*$ is recovered by acting greedily with respect to it.</span>

---

## The Update

<span style="font-size: 14px;">Starting from $V_0(s) = 0$ for all $s$, value iteration repeats the Bellman optimality backup:</span>

$$
V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \, V_k(s') \right]
$$

<span style="font-size: 14px;">Each sweep updates every state once using the previous sweep's values. Iteration continues until the values converge, measured by the max-norm change falling below a tolerance:</span>

$$
\max_s |V_{k+1}(s) - V_k(s)| < \texttt{tol}
$$

<span style="font-size: 14px;">or until a cap of $\texttt{max\_iters}$ sweeps is reached. After convergence, the greedy policy is extracted with one final lookahead:</span>

$$
\pi^*(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \, V^*(s') \right]
$$

<span style="font-size: 14px;">where $P[s][a][s']$ is the transition probability, $R[s][a][s']$ is the reward on the $(s, a, s')$ transition, and $\gamma \in [0, 1)$ is the discount factor. The output is the pair $(V, \pi)$ with values rounded to 4 decimals and integer action indices, ties broken toward the lowest index.</span>

---

## <span style="font-size: 16px;">The Idea</span>

<span style="font-size: 14px;">Value iteration sidesteps the central difficulty of solving the Bellman optimality equation directly: the equation is nonlinear because of the max over actions, so there is no matrix inverse that returns $V^*$ in one step. Instead, the algorithm treats the optimality equation $V^* = T^* V^*$ as a fixed-point problem and reaches the fixed point by **repeated application** of the operator $T^*$.</span>

<span style="font-size: 14px;">Conceptually, $V_k(s)$ can be read as the optimal value achievable in at most $k$ steps. The first sweep from $V_0 = 0$ yields the best one-step reward; the second sweep folds in the best two-step plans; after enough sweeps the finite-horizon optimum has converged to the infinite-horizon $V^*$. This interpretation also explains why initialization to zero is safe: it corresponds to "zero remaining value," which the backups then correct upward (or downward) as horizon information accumulates.</span>

<span style="font-size: 14px;">This is what makes value iteration strictly simpler than policy iteration: it never commits to a policy during the value sweeps. The implicit policy changes freely from sweep to sweep as the max picks whichever action currently looks best, and only crystallizes into the final greedy policy once the values have settled. Folding the improvement into the backup itself is the key efficiency: there is no separate evaluation phase to run to convergence.</span>

---

## <span style="font-size: 16px;">Why It Converges</span>

<span style="font-size: 14px;">The optimality operator $T^*$ is a **$\gamma$-contraction** in the max norm:</span>

$$
\| T^* U - T^* V \|_\infty \leq \gamma \, \| U - V \|_\infty
$$

<span style="font-size: 14px;">Each sweep shrinks the worst-case distance to $V^*$ by at least a factor of $\gamma$. The contraction holds because the inner expectation scales errors by $\gamma$ and the max over actions is non-expansive. By the Banach fixed-point theorem, $T^*$ has a unique fixed point $V^*$, and the iterates converge to it geometrically from any starting estimate:</span>

$$
\| V_k - V^* \|_\infty \leq \gamma^k \, \| V_0 - V^* \|_\infty
$$

<span style="font-size: 14px;">This is why $\gamma < 1$ is required for guaranteed convergence: at $\gamma = 1$ the operator is only non-expansive, not contractive, and convergence can fail. The bound also makes the runtime predictable: the number of sweeps to reach error $\epsilon$ scales like $\frac{\log(1/\epsilon)}{\log(1/\gamma)}$, so values of $\gamma$ near $1$ converge slowly even though each sweep is cheap.</span>

---

## <span style="font-size: 16px;">The Algorithm Step by Step</span>

<span style="font-size: 14px;">1. **Initialize** $V(s) = 0$ for every state $s$</span>

<span style="font-size: 14px;">2. **Sweep:** for each state $s$, compute $Q(s, a) = \sum_{s'} P(s' \mid s, a)[R(s, a, s') + \gamma V(s')]$ for every action and set $V_{new}(s) = \max_a Q(s, a)$, reading every $V(s')$ from the previous sweep's values</span>

<span style="font-size: 14px;">3. **Measure change:** compute $\delta = \max_s |V_{new}(s) - V(s)|$, then replace $V$ with $V_{new}$</span>

<span style="font-size: 14px;">4. **Check termination:** if $\delta < \texttt{tol}$ or the sweep count reaches $\texttt{max\_iters}$, stop; otherwise return to step 2</span>

<span style="font-size: 14px;">5. **Extract policy:** for each state set $\pi(s) = \arg\max_a Q(s, a)$ using the converged $V$, breaking ties toward the lowest action index</span>

---

## Worked Example (2-state, 2-action)

<span style="font-size: 14px;">States $\{0, 1\}$, $\gamma = 0.9$, deterministic transitions. State $0$: action $a_0$ stays in $0$ with reward $1$; action $a_1$ moves to $1$ with reward $0$. State $1$: action $a_0$ stays in $1$ with reward $5$; action $a_1$ moves to $0$ with reward $0$.</span>

<span style="font-size: 14px;">Start $V_0 = [0, 0]$.</span>

<span style="font-size: 14px;">1. **Sweep 1:** $V(0) = \max(1 + 0.9 \cdot 0, \ 0 + 0.9 \cdot 0) = 1$. $V(1) = \max(5 + 0.9 \cdot 0, \ 0 + 0.9 \cdot 0) = 5$. So $V_1 = [1, 5]$, $\delta = 5$</span>

<span style="font-size: 14px;">2. **Sweep 2:** $V(0) = \max(1 + 0.9 \cdot 1, \ 0 + 0.9 \cdot 5) = \max(1.9, 4.5) = 4.5$. $V(1) = \max(5 + 0.9 \cdot 5, \ 0 + 0.9 \cdot 1) = \max(9.5, 0.9) = 9.5$. So $V_2 = [4.5, 9.5]$, $\delta = 4.5$</span>

<span style="font-size: 14px;">3. The values keep growing and converge to the fixed point $V^* = [45, 50]$, where state $1$ satisfies $V(1) = 5 + 0.9 V(1) \Rightarrow V(1) = 50$ (self-loop), and state $0$ prefers moving to state $1$: $V(0) = 0.9 \cdot 50 = 45$, which beats staying ($1 + 0.9 \cdot 45 = 41.5$)</span>

<span style="font-size: 14px;">The greedy policy is $\pi^* = [a_1, a_0]$: from state $0$ move to the high-value state $1$, and in state $1$ stay to keep collecting the reward $5$. Note that in sweep 1 the greedy action in state $0$ looked like $a_0$, but as state $1$'s value grew the optimal action flipped to $a_1$, which is exactly why the policy must be extracted only after $V$ converges.</span>

---

## <span style="font-size: 16px;">Stochastic Transition Example</span>

<span style="font-size: 14px;">When actions have stochastic outcomes the inner expectation does real work. Consider a single state $s$ with two actions and successor values $V = [V(1), V(2)] = [10, 2]$, $\gamma = 0.9$:</span>

* <span style="font-size: 14px;">Action $a_0$: deterministic to state $2$, reward $4$. $Q(s, a_0) = 4 + 0.9 \cdot 2 = 5.8$</span>
* <span style="font-size: 14px;">Action $a_1$: to state $1$ with prob $0.6$ (reward $0$), to state $2$ with prob $0.4$ (reward $0$). $Q(s, a_1) = 0.6(0 + 0.9 \cdot 10) + 0.4(0 + 0.9 \cdot 2) = 0.6 \cdot 9 + 0.4 \cdot 1.8 = 5.4 + 0.72 = 6.12$</span>

<span style="font-size: 14px;">The backup gives $V_{new}(s) = \max(5.8, 6.12) = 6.12$ and greedy action $a_1$. The probabilistic action wins because its high-value branch (state $1$) outweighs the safe option in expectation. The algorithm always compares expected, not best-case, action values, which the inner sum computes before the max selects.</span>

---

## <span style="font-size: 16px;">The Q-Value Decomposition</span>

<span style="font-size: 14px;">A clean implementation computes the action values explicitly and reuses them for both the value update and the policy extraction:</span>

$$
Q(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \, V(s') \right]
$$

<span style="font-size: 14px;">Then $V_{new}(s) = \max_a Q(s, a)$ and, after convergence, $\pi^*(s) = \arg\max_a Q(s, a)$. Computing the same $Q(s, a)$ table once per state avoids duplicating the lookahead between the value max and the policy argmax, and it makes the tie-break rule explicit: scan actions in increasing index order and keep the first action that attains the running maximum. This structure also generalizes directly to **Q-value iteration**, $Q_{k+1}(s, a) = \sum_{s'} P[R + \gamma \max_{a'} Q_k(s', a')]$, which is the tabular ancestor of Q-learning and deep Q-networks.</span>

---

## <span style="font-size: 16px;">Synchronous vs In-Place Sweeps</span>

<span style="font-size: 14px;">The update above is **synchronous** (Jacobi): every $V_{new}(s)$ reads the previous sweep's values, and the array is swapped only after all states are computed. An **in-place** (Gauss-Seidel) variant updates $V(s)$ immediately so later states in the sweep see fresher values; it often converges in fewer sweeps but makes the per-sweep result depend on state ordering.</span>

<span style="font-size: 14px;">For a deterministic specification with a fixed convergence criterion, the synchronous form is the safe default because it is order-independent and reproducible. The convergence guarantee holds for both, since both are contractions toward the same fixed point. The choice affects only the trajectory of intermediate iterates, not the final $V^*$.</span>

---

## <span style="font-size: 16px;">Termination and the Stopping Rule</span>

<span style="font-size: 14px;">The $\texttt{tol}$ test on $\max_s |V_{k+1}(s) - V_k(s)|$ is a proxy for closeness to $V^*$. The contraction gives a rigorous bound linking the two:</span>

$$
\| V_{k+1} - V^* \|_\infty \leq \frac{\gamma}{1 - \gamma} \, \| V_{k+1} - V_k \|_\infty
$$

<span style="font-size: 14px;">So a small successive change does not by itself mean small error: the factor $\frac{\gamma}{1-\gamma}$ blows up as $\gamma \to 1$. The $\texttt{max\_iters}$ cap is a safety valve that guarantees termination even when $\texttt{tol}$ is never met, which matters when $\gamma$ is close to $1$ and convergence is slow. A correct implementation stops on whichever condition fires first and then extracts the policy from whatever $V$ it has.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Each sweep costs $O(S^2 A)$ time for a dense transition model: for every state and action, sum over $S$ successors. With a sparse model the cost drops to $O(S A \bar{d})$ where $\bar{d}$ is the mean number of reachable successors. The number of sweeps to reach tolerance $\epsilon$ is $O\!\left(\frac{\log(1/\epsilon)}{1 - \gamma}\right)$, so total work scales with both the state-action size and the effective horizon. Space is $O(S)$ beyond the model itself, since only the current and next value arrays are needed. Policy extraction is a single extra $O(S^2 A)$ sweep.</span>

---

## <span style="font-size: 16px;">Initialization and Rounding</span>

<span style="font-size: 14px;">The problem fixes $V_0(s) = 0$ for all states. Because $T^*$ is a contraction with a unique fixed point, the starting estimate does not affect the answer, only the number of sweeps. Zero is the conventional choice and corresponds to assuming no future value before any backup. A warm start from a good guess can converge faster but is not required here.</span>

<span style="font-size: 14px;">Rounding to 4 decimals is applied to the **reported** values, not to the running estimates: rounding inside the loop would inject error into every subsequent backup and could prevent the $\texttt{tol}$ test from ever passing. Keep the iteration in full precision and round only the final $V$ for output. The tolerance comparison should likewise use the unrounded successive change, so the loop terminates based on true convergence rather than rounding artifacts.</span>

---

## <span style="font-size: 16px;">Why a Single Sweep Is Not Enough</span>

<span style="font-size: 14px;">A single optimality backup propagates value information exactly one step: after sweep 1 only immediate rewards are reflected, after sweep 2 two-step plans, and so on. Information from a distant high-reward state takes as many sweeps to reach a far-away state as the number of transitions between them. This is the same one-step propagation seen in the worked example, where state $0$'s preferred action only became clear once state $1$'s value had grown over several sweeps. The discount $\gamma$ controls how strongly that distant information survives the journey, and the contraction guarantees the cumulative effect of all sweeps converges to the exact $V^*$.</span>

---

## <span style="font-size: 16px;">Value Iteration vs Policy Iteration</span>

<span style="font-size: 14px;">Both find $V^*$ and $\pi^*$, but they organize the work differently:</span>

* <span style="font-size: 14px;">**Value iteration** does one optimality backup per state per sweep and never solves a policy fully. It can take many cheap sweeps when $\gamma$ is near $1$</span>
* <span style="font-size: 14px;">**Policy iteration** fully evaluates the current policy (solving a linear system or sweeping to convergence), then improves it greedily. It converges in very few outer iterations but each is more expensive</span>

<span style="font-size: 14px;">Value iteration can be seen as policy iteration with the evaluation step truncated to a single sweep. In practice it is the simpler algorithm to implement and the default when the state space is large enough that full policy evaluation is costly.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Extracting the policy from an unconverged $V$.** The greedy action in a state can flip as successor values grow. Reading the policy before convergence (as the worked example shows for state $0$) yields a suboptimal policy. Extract $\pi^*$ only after the value sweep has stopped.</span>
* <span style="font-size: 14px;">**In-place updates when synchronous semantics are expected.** Overwriting $V(s)$ mid-sweep changes the intermediate iterates and the convergence trajectory. For a reproducible synchronous spec, compute all new values from a snapshot of the old array, then swap.</span>
* <span style="font-size: 14px;">**Inconsistent or floating-point-sensitive tie-breaking.** When actions tie for the argmax, comparing values that differ only by rounding noise can pick different actions on different runs. Use a strict lowest-index rule and avoid letting the 4-decimal rounding leak into the comparison used for argmax.</span>
* <span style="font-size: 14px;">**Ignoring the $\texttt{max\_iters}$ cap or using $\gamma = 1$.** Without the cap a slow-converging run (large $\gamma$) may never satisfy $\texttt{tol}$ and loops forever. With $\gamma = 1$ the operator is not a contraction and values can diverge. The spec requires $\gamma < 1$ and a finite iteration cap precisely to keep the loop well behaved.</span>

---