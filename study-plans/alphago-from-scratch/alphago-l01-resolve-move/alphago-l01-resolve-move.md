# Understanding Go State Transitions

This problem receives an action that is already known to be legal and turns the current game state into the next one. Its purpose is state transition, not move validation. That distinction keeps the task focused: the action does not need to be checked for occupancy, suicide, or positional superko again, but it must be applied with the correct capture, counter, player, and terminal rules.

## A move changes more than the board

A Go state contains several pieces of information. The stones are only one part. After every action, the next state must also record:

- which player moves next,
- how many opposing stones were captured,
- how many consecutive passes have occurred,
- how many total actions have been played,
- whether the game is now terminal.

Returning these values together ensures that later search code sees one consistent transition. Updating the board correctly while forgetting a counter would create a state that looks plausible but behaves incorrectly on the next turn.

## Decoding the action

For a board with side length $n$, intersection actions use row-major indices from 0 through $n^2-1$. The coordinate for an intersection action is:

$$
\text{row} = \left\lfloor \frac{a}{n} \right\rfloor
$$

$$
\text{column} = a \bmod n
$$

Here, a is the intersection action index and n is the board side length. The row is the integer quotient of a divided by n, and the column is the remainder.

The special action $n^2$ means pass. This shared action convention lets a policy vector, a legal mask, and the state transition all refer to moves using the same index.

Every action increments the move count by one, including pass. The next player is always the opponent, represented by negating the current value: Black changes from 1 to -1, and White changes from -1 to 1.

## Resolving a pass

A pass places no stone and captures nothing. The next board must contain exactly the same values as the current board, but it should still be returned as a separate copy so callers cannot accidentally mutate the original state through the returned array.

Passing increases the consecutive-pass count. It does not reset that count because the purpose of the counter is to detect two passes in succession. The transition after a pass therefore has:

- an unchanged board copy,
- the other player to move,
- a captured count of zero,
- one more consecutive pass,
- one more total move,
- a newly calculated terminal flag.

If this is the second consecutive pass, the game ends. The move cap can also end the game on a pass, so both terminal conditions still need to be considered.

## Resolving an intersection move

An intersection action begins by copying the board and placing the current player's stone at the decoded coordinate. The supplied action is guaranteed legal, so the destination is empty and the resulting friendly group will not be suicidal after captures.

Next, inspect only the orthogonally adjacent opposing stones. A newly placed stone can remove a liberty only from opposing groups that touch its coordinate. Distant groups cannot become captured from this move.

For each adjacent opponent stone, trace its complete connected group and collect its liberties. If no liberty remains, every stone in that group is captured and must be changed to empty on the result board.

It is possible for two adjacent opponent coordinates to belong to the same group. Collecting captured coordinates in a set ensures that the group contributes each stone once. It is also possible for one move to capture multiple separate groups, so the collection must combine all qualifying groups before the captured count is calculated.

## Why group liberties decide capture

Capture applies to a connected group as a unit. An individual opposing stone may have no empty neighbour of its own while remaining connected to another stone that has a liberty. Removing stones one by one from local appearance would therefore be wrong.

The correct question is whether the entire adjacent opposing group has any liberty after the new stone is placed. If its liberty set is empty, the whole group is removed. If even one liberty remains anywhere along the group, every stone stays.

Because the action is already legal, this function does not need to test the new friendly group for suicide or compare the result with earlier boards. Those rules belong to legal action generation. Repeating them here would require history that this function does not receive and would blur the contract between validation and transition.

## Updating counters after a played stone

An intersection action breaks any sequence of passes, so the consecutive-pass count resets to zero. The total move count still increases by one. The captured count is the number of distinct opposing stones removed from the result board.

The move cap used by this problem is two times the square of the board size:

$$
\text{move cap} = 2n^2
$$

After an intersection action, the state becomes terminal when the updated move count reaches that cap. Two-pass termination cannot occur on this branch because playing a stone resets the pass count.

## Preserving the original state

The function must not write into the supplied board. Search algorithms frequently branch from one position and explore many possible actions. If one transition mutates the shared parent board, every other branch begins from corrupted state.

Creating a copy before placement gives the next state its own board. The original remains a stable description of the parent position. This is true for pass as well: although no values change, returning a copy preserves the same ownership rule for every transition.

## Common mistakes to avoid

- Mutating the input board makes sibling search branches interfere with one another.
- Removing only the adjacent stone instead of its full group implements the wrong capture rule.
- Counting the same captured group from two neighbouring points inflates the captured total.
- Incrementing move count only for played stones forgets that pass is an action.
- Leaving the pass count unchanged after an intersection move can cause a later single pass to end the game incorrectly.
- Computing terminal status from the old counters misses the action that crosses a terminal threshold.

The transition is deterministic: decode the legal action, work on a copy, resolve any adjacent captures, switch the player, update both counters, and decide whether the resulting state has ended. Each returned value describes that same next moment in the game.

---