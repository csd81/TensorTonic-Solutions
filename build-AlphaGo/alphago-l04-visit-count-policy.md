# Understanding the Visit-Count Search Policy

After MCTS has run many simulations from the root, its visit counts summarize how search distributed attention across actions. Self-play converts those counts into a policy, then selects a move from that policy. Temperature controls whether selection is exploratory or nearly deterministic.

## From search counts to probabilities

Each legal action has a nonnegative visit count $N_a$. At positive temperature $\tau$, the unnormalized weight is:

$$
w_a=N_a^{1/\tau}
$$

The legal weights are normalized to form the search policy:

$$
\pi_a=\frac{N_a^{1/\tau}}{\sum_{b\in\mathcal{L}}N_b^{1/\tau}}
$$

Illegal actions receive probability zero and do not appear in the denominator. The result remains aligned with the complete action space, including pass.

When temperature is one, the exponent is one and the policy is simply proportional to visit counts. An action visited three times as often as another receives three times its probability.

## What temperature changes

Temperature changes the contrast between nonzero counts. When $\tau$ is below one, the exponent $1/\tau$ is greater than one, so larger counts are amplified more strongly. The policy becomes sharper and favours the most visited actions.

When $\tau$ is above one, the exponent lies between zero and one. Count differences are compressed, producing a flatter distribution over actions with positive visits.

Temperature does not make a zero count positive. At any positive temperature, zero raised to a positive power remains zero. An unvisited legal action therefore receives zero unless every legal action is unvisited and the special uniform rule applies.

## The all-zero legal case

If every legal visit count is zero, the positive-temperature formula has a zero denominator. Rather than divide by zero, the problem defines a uniform legal policy.

If there are $k$ legal actions, each receives:

$$
\frac{1}{k}
$$

Illegal actions still receive zero. This gives a valid probability distribution even before search has distinguished among legal choices.

The check concerns legal visits only. Counts at illegal entries must not prevent the uniform fallback, because illegal actions are excluded from the policy.

## Numerically stable positive-temperature weights

Direct exponentiation can overflow when counts are large and temperature is very small. The same weights can be computed in log space for positive counts:

$$
\log w_a=\frac{\log N_a}{\tau}
$$

Subtract the largest legal log weight before exponentiation, then normalize the resulting finite weights. The common subtraction does not change the final probabilities because it multiplies every weight by the same factor.

Only positive counts enter the logarithm. Taking the logarithm of zero is undefined, so zero-count actions keep their zero probability without participating in the log-space calculation.

## Zero temperature is a separate rule

At temperature zero, the positive-temperature formula cannot be evaluated because it would require division by zero in the exponent. The problem instead defines deterministic maximum-visit selection.

Find the largest visit count among legal actions, choose the lowest legal index with that count, and return a one-hot policy at that action. Every other entry is zero.

The lowest-index rule makes ties deterministic. It applies even when all legal counts are zero: every legal action ties for the maximum, so the earliest legal action is selected.

No uniform random draw is used at zero temperature. The supplied draw is absent on this branch.

## Sampling at positive temperature

At positive temperature, the selected action is sampled from the policy using the supplied uniform draw. Visit legal actions in increasing index order and accumulate their probabilities. Select the first action whose cumulative probability is strictly greater than the draw.

The word strictly matters. If a draw lies exactly on a cumulative boundary, it belongs to the next probability interval rather than the one that just ended. Using greater than or equal would select the earlier action and violate the required boundary convention.

Suppose the first legal action has probability one quarter. A draw below one quarter selects it, while a draw exactly equal to one quarter moves to the next interval.

Floating-point sums can finish just below one even when the mathematical total is one. Keeping the final legal action as a fallback ensures that every valid draw below one still produces an action.

## Policy and action must agree

The function returns both the complete policy and the selected action. At zero temperature, the selected action is the only position with probability one. At positive temperature, the action must be selected from the returned distribution using the supplied draw.

Repeated calls with the same inputs and draw are deterministic. Random-number generation is outside this function; it consumes the provided draw rather than generating a new one.

The policy uses 64-bit floating point, while the action is an ordinary Python integer. Converting the index matters because NumPy selection operations often return NumPy integer scalars.

## Common mistakes to avoid

- Including illegal counts in normalization gives forbidden actions probability mass.
- Evaluating the positive-temperature formula at zero temperature causes division by zero.
- Taking the logarithm of zero visits creates invalid values.
- Forgetting the all-zero fallback leaves a zero denominator.
- Randomly breaking maximum-count ties violates the lowest-index rule.
- Using greater than or equal changes exact cumulative-boundary sampling.
- Generating a new random draw ignores the supplied deterministic input.
- Returning a compact legal-only policy loses complete action alignment.

Visit-count policy conversion turns search effort into a training target and a move choice. Positive temperature reshapes and samples the legal counts, zero temperature selects the earliest maximum deterministically, and both branches keep illegal probability exactly zero.

---