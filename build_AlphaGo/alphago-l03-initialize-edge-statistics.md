# Understanding MCTS Edge Initialization

When search expands a new node, it creates one edge for every action in the complete action space. Each edge needs a small set of statistics that later simulations will update. At initialization, no edge has been visited, so the network's policy logits provide the only reason to prefer one legal action over another.

## The four edge statistics

The returned dictionary contains four aligned arrays. The entry at one action index describes that same action in every array:

- $N$ is the number of times the edge has been visited.
- $W$ is the total value accumulated through the edge.
- $Q$ is the mean value of the edge.
- $P$ is the prior probability assigned from the policy network.

Fresh edges have not participated in any simulation. Their visit counts are zero, no value has been accumulated, and there is no empirical mean value yet. This problem represents that initial state by setting $N$, $W$, and $Q$ to zero for every action.

The prior $P$ is different. It is available immediately because the policy network has already produced a logit for every action. Those logits must be converted into probabilities over legal actions only.

## Why legality is applied before softmax

An illegal action must never receive an edge prior. Giving it even a small probability would waste probability mass and could let later selection logic consider an action that the game rules forbid.

The legal mask therefore selects the logits that participate in softmax. Illegal logits are excluded completely, rather than being assigned a large negative replacement and left in the normalization approximately. Their final priors are exact zeros.

If $\mathcal{L}$ is the set of legal actions, the prior for a legal action $a$ is:

$$
P_a=\frac{\exp(l_a-m)}{\sum_{b\in\mathcal{L}}\exp(l_b-m)}
$$

Here $l_a$ is the action's logit and $m$ is the largest logit among legal actions. For an illegal action, $P_a$ is exactly zero.

The denominator contains only legal actions. As a result, legal priors sum to one even when most of the complete action space is unavailable.

## What logits mean

A policy logit is a raw preference score, not a probability. Logits may be negative, positive, very large, or very small. Only their relative values matter to softmax.

Equal legal logits produce equal priors. If two legal actions have the same logit and they are the only legal actions, each receives one half. If exactly one action is legal, its prior is one regardless of its logit because it receives all available probability mass.

An illegal action with the largest raw logit still receives zero. Legality is a rule constraint, so it takes precedence over the network's preference.

## Why subtract the largest legal logit

Directly exponentiating a large positive logit can overflow floating-point arithmetic. For example, a logit near one thousand produces an exponential too large for ordinary numeric representation. Softmax avoids this without changing the probabilities by subtracting the same constant from every participating logit.

Choosing the largest legal logit makes the biggest shifted value zero and all others nonpositive. Their exponentials are therefore at most one:

$$
\exp(l_a-m) \leq 1
$$

Softmax is unchanged because the common factor cancels between numerator and denominator. For any constant $c$:

$$
\frac{\exp(l_a-c)}{\sum_b \exp(l_b-c)}
=
\frac{\exp(l_a)}{\sum_b \exp(l_b)}
$$

The maximum must be taken over legal logits, not the complete vector. Illegal values do not participate in the distribution and should not influence its numerical reference point.

## Alignment with the complete action space

The prior calculation works on a compact selection of legal logits, but the returned $P$ array must align with every original action. A useful mental model is to begin with a zero prior array of the complete length, calculate probabilities for the legal selection, and place them back only at legal indices.

The same length is used for $N$, $W$, and $Q$. This alignment lets later MCTS code select one action index and read or update all four statistics without translating between different index systems. The pass action remains in its normal final position when it is legal.

## Required numeric types

Visit counts use 64-bit integers because they count discrete simulations. Total values, mean values, and priors use 64-bit floating point because they store fractional quantities.

Although every initial value in $W$ and $Q$ is zero, their floating type matters for later updates. An integer array would discard fractional backed-up values. Likewise, $P$ must hold softmax probabilities rather than rounded integers.

## Useful invariants after initialization

A correctly initialized node has several properties that can be checked directly:

- Every visit count, total value, and mean value is zero.
- Every illegal prior is exactly zero.
- Legal priors are nonnegative; very small probabilities can underflow to zero in floating-point arithmetic.
- The priors over legal actions sum to one.
- All four arrays have the same length and action alignment.

These are stronger checks than looking at one example. They describe what must remain true for any valid logits and mask accepted by the problem.

## Common mistakes to avoid

- Applying softmax to every logit gives probability mass to illegal actions.
- Zeroing illegal probabilities after a full softmax leaves legal probabilities summing to less than one.
- Subtracting the largest value after exponentiation does not prevent overflow.
- Using the largest complete-space logit lets an illegal action influence the stability shift.
- Returning only compact legal arrays loses alignment with the complete action space.
- Using floating-point visit counts or integer priors violates the required statistic types.
- Deriving $Q$ by dividing zero total value by zero visits creates undefined values; fresh means are defined as zero here.

Initialization gives search a clean starting point: every empirical statistic is zero, while a stable legal-only softmax turns the network's raw preferences into aligned edge priors.

---