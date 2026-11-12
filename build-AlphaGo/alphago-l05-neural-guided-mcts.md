# Understanding Deterministic Neural-Guided MCTS

Neural-guided MCTS combines a network's policy and value estimates with repeated tree search. The policy provides priors for newly expanded edges, simulations gather visit and value statistics, and the final root visit counts become a search policy. This problem joins those pieces on a supplied finite two-player tree.

## The supplied tree

Each nonterminal state has a strictly increasing list of legal actions and a matching transition for each action. Policy logits are aligned with that legal-action list rather than with a larger global action vector.

Every state also records the current player. Player signs alternate across edges, and both network values and terminal values are expressed from the current player's perspective at the state where they are read.

The graph is guaranteed to be a tree. A state has one path from the root, so the search does not need transposition handling or repeated-state detection.

## Expanding a state

Expansion creates one statistics record per legal action. The state's policy logits are converted to priors using a stable softmax. Subtract the largest logit before exponentiation, then normalize the weights:

$$
P_a=\frac{\exp(l_a-m)}{\sum_b\exp(l_b-m)}
$$

Every new edge begins with visit count $N=0$, accumulated value $W=0$, and mean value $Q=0$. Its prior $P$ remains fixed after expansion.

The root is expanded before the simulation loop. This means the first simulation begins by selecting one root edge rather than treating the root itself as a newly discovered leaf.

Other nonterminal states are expanded only when a traversal reaches them for the first time. Expansion and evaluation happen together at that first visit, and traversal stops there for the current simulation.

## Selecting an edge with PUCT

At an expanded nonterminal state, calculate the total number of visits across its outgoing edges:

$$
T=\sum_b N(s,b)
$$

Each action receives the score:

$$
S(s,a)=Q(s,a)+c_{\mathrm{puct}}P(s,a)\frac{\sqrt{T}}{1+N(s,a)}
$$

$Q$ is the current mean result for the player who selects at that state. The second term favours actions with strong network priors and fewer visits. The exploration coefficient controls how strongly that term influences selection.

The total is used exactly as written, with no added one. When all outgoing visits are zero, $T=0$, every exploration bonus is zero, and all $Q$ values are also zero. The first selection is therefore a tie.

Legal actions are supplied in increasing order, and exact ties select the lowest action. Updating the selected action only when a score is strictly larger preserves this deterministic rule.

## Traversing one simulation

A simulation starts at the root and repeatedly selects a PUCT action while the current state is nonterminal and already expanded. Each selected state-action pair is appended to the path, then the transition moves to its child.

Traversal stops at the first state that is either terminal or unexpanded. It does not continue through a newly expanded node during the same simulation. This gives each simulation one leaf evaluation.

There are two leaf cases:

- A terminal state uses its supplied terminal value.
- An unexpanded nonterminal state is expanded and uses its supplied neural value prediction.

The leaf value is interpreted from the current player at that leaf. No policy is needed at a terminal state because there are no outgoing edges to expand.

## Backing the value through the path

Every selected edge receives one visit and one value update. The edge statistics belong to the player who selected that edge, which is the current player at its parent state.

Compare that parent player with the leaf player. If they are the same, the edge receives the leaf value. If they differ, it receives the negated leaf value. Because player signs alternate and the game is zero-sum, this expresses the same outcome from the edge player's perspective.

For the adjusted value $v_a$:

$$
N'_a=N_a+1
$$

$$
W'_a=W_a+v_a
$$

$$
Q'_a=\frac{W'_a}{N'_a}
$$

Backup walks over every selected edge from leaf toward root. The path order used for updates does not change the stored tree, but reverse traversal reflects how the leaf evaluation propagates upward.

## Building the root outputs

After all simulations, collect root statistics in the root's legal-action order. The first output contains integer visit counts, and the second contains 64-bit floating-point mean values.

The third output is the root policy derived from visits. At zero temperature, select the earliest maximum visit count and return a one-hot policy. Since root actions are already increasing, the first maximum also corresponds to the lowest action tie break.

At positive temperature, use visit weights raised to the reciprocal temperature and normalize them. A stable log-space form handles small temperatures:

$$
\pi_a=\frac{N_a^{1/\tau}}{\sum_b N_b^{1/\tau}}
$$

The simulation count is positive, so at least one root edge has a positive visit. Zero-visit edges receive zero positive-temperature probability.

The returned arrays are aligned to legal-action order, not indexed by raw action number. If root actions are 2 and 5, the first output entry describes action 2 and the second describes action 5.

## Common mistakes to avoid

- Failing to expand the root before simulation changes the first traversal and visit totals.
- Adding one to total node visits changes the stated PUCT behaviour.
- Breaking score ties randomly makes the search nondeterministic.
- Continuing through a newly expanded state gives one simulation more than one evaluation.
- Treating a leaf value as belonging to every player without sign adjustment corrupts $W$ and $Q$.
- Replacing accumulated value instead of adding to it discards earlier simulations.
- Returning arrays in raw action-index order ignores the compact legal-action contract.
- Applying softmax to root visits at the end is not the temperature visit policy.

The integrated search repeats one precise cycle: select through expanded nodes with PUCT, stop at the first terminal or new state, evaluate once, back up from the correct player perspectives, and finally turn root visits into the requested policy.

---