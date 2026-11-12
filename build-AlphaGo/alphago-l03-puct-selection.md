# Understanding PUCT Action Selection

At an MCTS node, action selection must balance two goals. Search should revisit actions that have produced good values, while still exploring promising actions that have not received enough attention. PUCT combines these goals in one score for every legal edge and selects the highest score.

## Exploitation through the mean value

For action $a$, the visit count $N_a$ records how many simulations used that edge, and the total value $W_a$ stores the values accumulated from those simulations. The mean value is:

$$
Q_a=\frac{W_a}{N_a}
$$

This division is meaningful only when the edge has been visited. The problem defines an unvisited edge's mean as zero:

$$
Q_a=
\begin{cases}
0, & N_a=0 \\
W_a/N_a, & N_a>0
\end{cases}
$$

$Q_a$ is the exploitation part of the score. A larger mean says that simulations through this action have been better from the node player's perspective. Using the mean rather than total value prevents an edge from looking good merely because it has been visited many times.

The zero-visit case should be assigned directly rather than computed by division. Dividing zero by zero produces an undefined value that can contaminate the score vector and action selection.

## Exploration through the policy prior

The policy prior $P_a$ expresses how promising the network considered the action when the node was expanded. PUCT turns that prior into an exploration bonus:

$$
U_a=cP_a\frac{\sqrt{T}}{1+N_a}
$$

Here $c$ is the positive exploration constant and $T$ is the sum of all edge visit counts at the node:

$$
T=\sum_b N_b
$$

The complete score is:

$$
S_a=Q_a+U_a
$$

Each factor in the bonus has a clear role. A larger prior increases exploration for actions favoured by the network. The square root of total node visits makes exploration pressure grow as the node receives more search. The denominator reduces the bonus for an edge that has already been visited repeatedly.

Adding one to the edge visit count keeps the denominator valid for an unvisited action. The numerator's total visit count is used exactly as supplied by the sum, without adding one.

## What happens on the first selection

When every edge visit count is zero, the total $T$ is zero. Its square root is also zero, so every exploration bonus is zero. Every unvisited $Q$ is zero as well. All legal actions therefore receive score zero on this first call.

The required tie rule selects the lowest legal action index. This may seem surprising because the priors differ, but it follows directly from the given formula, which does not add one to total visits. The policy prior begins affecting selection after the node has at least one recorded visit.

Changing the numerator to the square root of $T+1$ would make priors active immediately, but that is a different PUCT definition and would fail this problem's examples.

## How visits change the balance

Suppose two actions have similar mean values and priors, but one has been visited much more often. The heavily visited action has a larger denominator, so its exploration bonus is smaller. The less visited action receives more pressure to be examined.

An action with a strong mean value can still win despite a smaller exploration bonus because $Q$ is added directly. An unvisited action can become competitive through a large prior and the smallest possible denominator. PUCT does not enforce a fixed exploration schedule; the relative scores decide at every node visit.

The exploration constant $c$ controls the overall strength of the bonus. A larger value gives priors and under-visited edges more influence, while a smaller value makes the empirical mean values more dominant. This problem receives $c$ as a supplied positive value and applies it directly.

## Illegal actions

The score array covers the complete action space, but selection must consider only legal actions. After computing the numeric scores, every illegal entry is replaced with negative infinity.

Negative infinity is useful because any finite legal score is larger. It preserves alignment in the returned score vector while guaranteeing that a standard maximum operation cannot select an illegal edge.

The legal mask is authoritative even if an illegal action has a high mean or total value. Its final selection score is still negative infinity; illegal priors are zero under this problem's input contract.

## Deterministic tie breaking

More than one legal action can have the same maximum score. NumPy's first-maximum behaviour returns the earliest index, which matches the required lowest-index tie rule as long as illegal entries have already been replaced by negative infinity.

The selected index must be returned as an ordinary Python integer. NumPy's maximum-index operation produces a NumPy integer scalar, which has the right numeric value but not the exact requested type.

Deterministic ties are important for reproducible examples and tests. No random choice is part of this selection problem.

## Numeric representation

The complete score vector uses 64-bit floating point. Visit counts remain integers, while total values and priors are interpreted as floating-point quantities. The total visit sum is used in floating-point arithmetic before taking its square root.

Mean values should be calculated only at positions with positive visits. Starting with a zero floating-point vector and filling visited positions avoids division warnings and preserves the defined zero for unvisited edges.

## Common mistakes to avoid

- Dividing total value by zero visits creates undefined mean values.
- Using total value directly instead of the mean rewards visit volume rather than average outcome.
- Adding one to total visits changes the first-selection behaviour required by the formula.
- Omitting one from the edge denominator causes division by zero for unvisited actions.
- Applying the legal mask after choosing the maximum can select a forbidden action.
- Randomly breaking ties violates the required lowest-index result.
- Returning only legal scores loses alignment with the complete action space.

PUCT selection is one comparison built from two sources of evidence: $Q$ summarizes what simulations have observed, while the prior-weighted bonus encourages search where the network and low visit count suggest more information may be useful.

---