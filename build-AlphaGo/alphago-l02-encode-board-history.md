# Understanding AlphaGo Zero Board History Encoding

A neural network cannot consume a Go board as an informal picture. It needs a fixed collection of numeric planes whose meaning stays consistent from one position to the next. This problem converts the current board and recent history into that representation while expressing stones from the viewpoint of the player whose turn it is.

## Why one board is not the whole state

The current arrangement of stones is central, but recent positions provide useful context about how that arrangement was reached. AlphaGo Zero represents the current board together with up to seven earlier boards. The history is supplied newest first, so its first entry is the current position, its second entry is the immediately preceding position, and so on.

A fixed network input cannot grow or shrink with the available history. The encoding always reserves space for eight positions. When a game has fewer than eight recorded boards, the unused historical slots are filled with zeros.

## Two planes for every position

Each historical board becomes two binary planes:

- The first plane marks stones belonging to the current player.
- The second plane marks stones belonging to the opponent.

A marked intersection contains 1.0 and every other intersection contains 0.0. Empty points are therefore zero in both planes.

This is a current-player-relative representation. If Black is to play, Black stones appear in the first plane and White stones appear in the second. If White is to play, White stones appear in the first plane and Black stones appear in the second. The meaning of the channel remains stable as turns change: first means mine, second means the opponent's.

That symmetry is valuable because the network can learn patterns such as the safety of my group or pressure on the opponent's stones without needing entirely separate channel meanings for Black and White.

## Channel order across time

For each of the eight reserved history steps, the current-player plane comes before the opponent plane. Since the input history is already newest first, the channels are arranged as:

$$
X_0, Y_0, X_1, Y_1, \ldots, X_7, Y_7
$$

Here $X_t$ marks the current player's stones at history step $t$, and $Y_t$ marks the opponent's stones at the same step. The subscript describes age, not player identity. Step zero is the newest board.

Eight positions with two planes each produce sixteen stone planes. Their order is part of the data contract. Reversing the history or grouping all current-player planes before all opponent planes would preserve the same raw facts but attach them to the wrong channel meanings.

## Padding missing history

Early in a game, there may be only one or two boards available. The remaining reserved steps contribute pairs of all-zero planes. A zero pair means no board was supplied for that age; it does not claim that a historical board was empty.

Each padding plane must match the board's height, width, device, and required floating-point type. Creating padding on a default CPU while the history lives on a GPU would make the planes impossible to stack. Taking device and board size from the supplied history keeps every channel compatible.

## The final colour plane

One additional plane records which colour is to play. Every point in this plane has the same value:

- The plane is filled with 1.0 when Black is to play.
- The plane is filled with 0.0 when White is to play.

This plane may seem repetitive because every intersection contains the same number, but a convolutional network reads local regions. Repeating the turn information across the board makes it available at every spatial location without requiring a separate nonspatial input path.

The sixteen historical stone planes plus this colour plane give seventeen channels in total. Each channel has the same board-sized grid, so the result is a tensor with seventeen planes stacked along the first dimension.

## Creating indicator planes

For an available board, comparison provides the cleanest interpretation. Comparing every intersection with the current player's value produces a Boolean map of current-player stones. Comparing with the negated player value produces the opponent map. Converting those maps to 32-bit floating point gives the required zeros and ones.

This approach handles both turns with the same rule. There is no need for separate Black and White branches beyond the final colour plane. It also leaves empty intersections unmarked automatically because zero equals neither player value.

The history tensors may arrive with another numeric type, but the output must always use 32-bit floating point. Explicit conversion prevents the result from inheriting an integer or higher-precision type. The conversion should retain the original device.

## Stacking without changing the inputs

Once sixteen historical planes and the colour plane have been collected, stacking them creates one tensor. Stacking adds the channel dimension while preserving the board coordinates within each plane.

No input board needs to be edited. Comparisons create new tensors, and padding and colour planes are newly allocated. Preserving the history is important because the same board tensors may still be part of the environment state or may be reused by other search branches.

The method also works with noncontiguous board tensors because elementwise comparison depends on logical values, not a particular memory layout. Operations should avoid assumptions that the board data occupies one contiguous block.

## Common mistakes to avoid

- Encoding fixed Black and White planes ignores the required current-player perspective.
- Reading history oldest first moves every board into the wrong time channel.
- Adding only one zero plane for a missing step breaks the two-plane pairing.
- Omitting the final colour plane produces sixteen channels instead of seventeen.
- Creating padding on the CPU fails when the history tensors use another device.
- Allowing the output to inherit integer input types violates the 32-bit floating-point contract.
- Reusing or editing input tensors risks changing the stored game history.

The representation is fixed and interpretable: eight newest-first board positions, two perspective-relative planes per position, zero pairs for missing history, and one final plane indicating whether Black moves next. This gives the policy-value network a consistent numeric view of both space and recent time.

---