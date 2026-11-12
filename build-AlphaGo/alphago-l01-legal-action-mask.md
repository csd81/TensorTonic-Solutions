# Understanding the Legal Go Action Mask

A policy network assigns a preference to every possible action, but not every action is allowed in the current position. The legal action mask is the bridge between the game rules and those action scores. It marks each playable intersection as true, each forbidden intersection as false, and reserves one final true or false entry for passing.

## How actions are represented

The board intersections are flattened in row-major order. On a board with side length $n$, an intersection action ranges from 0 through $n^2-1$. Its board coordinates are recovered by dividing by the side length:

$$
\text{row} = \left\lfloor \frac{a}{n} \right\rfloor
$$

$$
\text{column} = a \bmod n
$$

Here, a is the intersection action index and n is the board side length. The row is the integer quotient of a divided by n, and the column is the remainder.

The pass action uses index $n^2$, immediately after all intersections. The resulting Boolean mask therefore has $n^2+1$ entries. Keeping this convention consistent is essential because the same indices will later be used by a policy vector and by move application.

## Terminal states have no legal actions

The first question is whether the state is already terminal. In this problem, the game ends after two consecutive passes or when the move count reaches two times the square of the board size. Once either condition is true, no intersection and no pass should be offered. The required result is an all-false mask.

For a nonterminal state, pass is always legal and occupies the final entry. Intersection actions must be tested against occupancy, capture, suicide, and positional superko.

## Occupied points are immediately illegal

A stone can only be placed on an empty intersection. If a candidate coordinate already contains Black or White, its mask entry is false without any simulation. Empty points require a temporary result board because later rules depend on the position after the stone is placed and captures are removed.

Each candidate must use its own board copy. Testing one move must not change the board used to test another move, and the original board supplied by the caller must remain unchanged. Without independent copies, legality results would depend on the order in which actions happened to be checked.

## Captures must be resolved before suicide

After placing the candidate stone, examine adjacent opposing groups. A group is captured when it has no liberties in the temporary position. Every stone in such a group is removed.

The order matters. A move may initially appear to leave the new stone with no empty neighbour, yet capture an opposing group and create liberties where those stones were removed. If suicide is checked before captures, this legal capturing move will be rejected.

Several adjacent points can belong to the same opposing group. Captured coordinates should therefore be collected distinctly before removal. This prevents duplicate counting and avoids unnecessary repeated writes, while still allowing more than one separate opposing group to be captured by a single move.

## The suicide rule

Once all captures have been removed, find the connected group containing the newly played stone and collect its liberties. If that group has no liberties, the move is suicide and is illegal.

The check applies to the complete new group, not only to the placed stone. The new stone might connect to a friendly chain that has a liberty elsewhere. Conversely, an apparently open local shape can still have no liberty if all neighbouring points are occupied and no connected friendly stone reaches an empty point.

This gives the correct local rule sequence:

- Place the stone on a copy.
- Find and remove adjacent opposing groups with no liberties.
- Find the played stone's resulting group.
- Reject the move if that group still has no liberties.

## Positional superko

Even a move that survives occupancy, capture, and suicide checks can be illegal because of repetition. Positional superko forbids a move when its resulting board is identical to any previously seen board supplied to the function.

This comparison concerns the whole board after captures. Checking only the candidate coordinate, the number of captured stones, or the immediately previous board is not enough. The history can contain an older repeated position, and the rule asks whether the complete result matches any supplied position.

Pass is handled separately in this exercise. It remains legal in every nonterminal state even though it leaves the stones unchanged. Intersection moves are the ones simulated and compared against the position history.

## Interpreting the finished mask

Each intersection entry means that the corresponding action can be applied legally from the current state. The final entry means that the player may pass. The mask should use NumPy's Boolean type so it can be combined directly with later policy calculations without confusing numeric scores with legality flags.

On an empty nonterminal board of side length at least two, every intersection and pass are legal when the history contains only that empty board. On a 1-by-1 board, placement has no liberties, so only pass is legal. In a position where every empty point would be suicide, all intersection entries can be false while pass remains true. In a terminal state, even pass becomes false because there is no next action.

## Common mistakes to avoid

- Checking suicide before removing captures rejects legal capturing moves.
- Reusing one temporary board across candidates lets earlier simulations corrupt later answers.
- Examining only the new stone's immediate neighbours misses liberties elsewhere in its connected group.
- Comparing only with the latest board implements a simple ko check, not the supplied positional-superko history.
- Forgetting the extra pass entry shifts the action convention used by the rest of the system.
- Marking pass true before handling terminal state produces an action after the game has ended.

The mask is a compact summary of the rules. For every intersection, it answers whether placing a stone, resolving captures, checking the resulting group, and checking the complete history leads to a valid position. Pass occupies the final entry, and terminal states close every option.

---