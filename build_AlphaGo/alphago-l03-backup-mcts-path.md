# Understanding MCTS Path Backup

One MCTS simulation travels from the root through a sequence of selected edges and eventually reaches a leaf. The leaf is evaluated once, but that result must inform every edge that led to it. Backup is the operation that carries the leaf value backward and updates the visit statistics along the path.

## What the path records mean

The path is stored in root-to-leaf order. Each record describes one selected edge and contains its state identifier, action, player, visit count, total value, mean value, and prior.

The player field is especially important. It identifies the player who selected that edge, so the edge's value statistics must be expressed from that player's perspective. The leaf value, however, initially arrives from the leaf player's perspective. Backup must reconcile those viewpoints.

The statistics have the following roles:

- $N$ counts how many simulations have passed through the edge.
- $W$ is the sum of all perspective-adjusted values backed through it.
- $Q$ is the empirical mean value, equal to total value divided by visits.
- $P$ is the policy prior assigned when the edge was created.

Only $N$, $W$, and $Q$ change during backup. The prior and identifying fields describe the edge itself and must remain unchanged.

## Why values need a perspective

A positive value means good for the player whose perspective is being used. In a two-player zero-sum game, what is good for one player is bad for the other. Changing perspective therefore negates the value.

If a leaf value is $0.6$ for Black, the same outcome is $-0.6$ for White. If the value is zero, negation has no visible effect because the evaluation is neutral. The sign has meaning only together with the player it belongs to.

This problem does not assume that every neighbouring path record must have a different player. Instead, it tells us to negate only when the stored player changes while walking backward. That rule handles ordinary alternating turns and also preserves the correct sign if two consecutive records happen to carry the same player.

## Walking from leaf to root

Although the returned records keep root-to-leaf order, propagation begins at the leaf end. The deepest selected edge is the first edge that should receive the leaf evaluation, followed by its parent, continuing until the root edge.

Start with two pieces of propagation state:

- the current value, initialized to the supplied leaf value,
- the current perspective, initialized to the supplied leaf player.

At each record in reverse order, compare the record's player with the current perspective. If they differ, negate the current value. If they match, keep the value unchanged. The resulting number is the value from that edge player's perspective.

After updating the edge, its player becomes the current perspective for the next step toward the root. This comparison-based approach is safer than blindly negating at every record because it follows the actual stored player sequence.

## Updating one edge

Every backed-up simulation contributes exactly one additional visit:

$$
N'_a=N_a+1
$$

The perspective-adjusted value $v_a$ is added to the accumulated total:

$$
W'_a=W_a+v_a
$$

The new mean must be calculated from the updated total and updated visit count:

$$
Q'_a=\frac{W'_a}{N'_a}
$$

The order expressed by these equations matters. Computing the mean with the old visit count or the old total produces the wrong average. There is no division-by-zero problem after incrementing because the new visit count is at least one.

Suppose an edge has already been visited three times with total value $0.75$. Its mean is $0.25$. If the new perspective-adjusted result is $-0.5$, the updated total becomes $0.25$, the visit count becomes four, and the new mean becomes $0.0625$. Backup incorporates the new observation rather than replacing the earlier history.

## A perspective example

Imagine a two-edge path where the deepest edge was selected by White, the root edge was selected by Black, and the leaf value is positive from Black's perspective. At the deepest edge, the stored player differs from the leaf player, so the value is negated and White receives a negative update. Moving to the root changes perspective back to Black, so the value is negated again and Black receives the original positive value.

If two adjacent records both identify Black, the sign does not change between them. Both edges store the evaluation from the same perspective. This is why checking player identity is more precise than assuming one sign flip per list element.

## Preserve path order and metadata

The state identifier, action, player, and prior remain exactly as supplied. Converting the prior to an ordinary floating value is acceptable under the stored solution, but its numeric value must not change. Backup is not a second policy calculation.

## Why a deep copy is required

The supplied path may be retained by other search code. Updating its dictionaries directly would silently change the caller's stored records. A shallow copy of the outer list is not enough because the dictionaries inside would still be shared.

A deep copy creates independent records before any statistic is changed. The function returns the updated copy while the original path remains untouched.

An empty path is a valid boundary case. There are no edges to update, so the result is simply a new empty list. The leaf value does not create a record by itself.

## Common mistakes to avoid

- Reversing the returned records changes the required root-to-leaf layout.
- Negating at every record ignores the stored player sequence.
- Never negating treats one player's favourable result as favourable for both players.
- Replacing $W$ with the new value discards all earlier simulations.
- Calculating $Q$ before incrementing $N$ uses stale statistics.
- Updating only a shallow copy still mutates the original dictionaries.
- Returning the reverse traversal order changes the required path layout.
- Modifying the policy prior mixes backup with edge initialization.

Path backup takes one leaf evaluation and turns it into consistent evidence for every selected edge. Reverse traversal supplies the direction, player comparisons supply the sign, and the updated visit total and mean preserve everything learned from earlier simulations.

---