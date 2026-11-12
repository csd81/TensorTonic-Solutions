# Understanding Dihedral Go Symmetry

A square Go position can be rotated or reflected without changing the strategic meaning of the game. If the board is transformed, the move probabilities must follow the same coordinates, while the game outcome stays unchanged. This creates additional valid training records from one self-play position.

## The eight symmetries of a square

A square has eight spatial symmetries. Four are rotations, and four begin with a left-to-right reflection followed by those same rotations.

The transform identifiers in this problem have an exact meaning:

- Identifier 0 leaves the square unchanged.
- Identifier 1 rotates it 90 degrees counterclockwise.
- Identifier 2 rotates it 180 degrees.
- Identifier 3 rotates it 270 degrees counterclockwise.
- Identifier 4 reflects left to right with no later rotation.
- Identifier 5 reflects left to right, then rotates 90 degrees counterclockwise.
- Identifier 6 reflects left to right, then rotates 180 degrees.
- Identifier 7 reflects left to right, then rotates 270 degrees counterclockwise.

The order for the final four matters. Reflecting and then rotating is not generally the same coordinate mapping as rotating and then reflecting. Following the supplied identifier convention keeps the output consistent with the examples and with any inverse mapping used elsewhere.

## Transform every state plane together

The state contains several board-sized planes. Each plane describes information at the same intersections, such as current stones, opponent stones, older positions, or the colour-to-play indicator.

The transform acts on the final two axes, which represent board rows and columns. The plane axis is preserved. Every plane receives the same rotation or reflection so that all channels continue to describe one aligned position.

Applying different transforms to different planes would create a state that never existed. A stone location in one channel would no longer correspond to the same board coordinate in another channel.

For the colour-to-play plane, every value is constant, so rotation has no visible effect. It should still pass through the same spatial operation because the function treats all planes consistently.

## The policy has a spatial part and a pass part

The policy contains one probability for every board intersection followed by one final probability for pass. The board probabilities use row-major action order, so they can be reshaped into a square grid with the board's side length.

That square policy grid must receive exactly the same spatial transform as the state. After transformation, it is flattened back into row-major action order.

This alignment is the central correctness condition. If a move had high probability at one intersection before transformation, the transformed policy must place that same probability at the transformed intersection. Rotating only the state would pair a board position with targets for different coordinates.

The pass action is not attached to any intersection. It remains the final policy entry and keeps its original probability. It must be separated before the board policy is reshaped, then appended unchanged after the transformed board entries are flattened.

## Why the value does not change

The scalar value describes the outcome or desirability of the position from the current player's perspective. Rotating or reflecting the board does not exchange the players, change the rules, or alter who eventually wins.

Therefore the value is returned unchanged. A positive target remains positive, a negative target remains negative, and a neutral value remains zero. Spatial augmentation changes coordinates, not perspective.

This differs from colour inversion, which would exchange player meaning and could require a sign change. Colour inversion is not one of the eight transforms in this problem.

## Reflection before rotation

For identifiers 4 through 7, first reverse the column direction. This is a left-to-right reflection: the leftmost column becomes the rightmost column, and vice versa. Rows remain in place during the reflection.

After reflection, rotate counterclockwise by the identifier minus four quarter-turns. Identifier 4 therefore performs only reflection, while identifier 7 performs reflection followed by three quarter-turns.

Because the transform is applied to the complete last two axes, it works for any square board size, including a one-by-one board where every symmetry has the same visible result.

## Preserve dtypes

This problem does not request numeric conversion. The state might contain integer indicators, Boolean values, or floating features, while the policy has its own probability dtype. Each transformed array must preserve the dtype of its corresponding input.

Using NumPy rotation, flipping, reshaping, and concatenation without an explicit conversion preserves those types. The scalar value is returned as supplied rather than converted to another representation.

## Return independent arrays

NumPy rotations and flips can return views that share memory with their source. A view may look like a new array, but writing into it can affect the original input or another returned object.

The required outputs must be newly allocated. Making explicit copies after the state and policy-grid transforms ensures that later batch processing cannot modify the original training record. The pass entry should also be copied before concatenation.

The inputs themselves must never be edited in place. Augmentation creates an additional record; it does not replace the stored original.

## Common mistakes to avoid

- Rotating the state but not the board policy misaligns inputs and targets.
- Including pass in the square reshape either fails or moves a nonspatial action.
- Using clockwise rotation reverses the required identifier convention.
- Rotating before reflecting gives different results for the final four identifiers.
- Transforming the plane axis mixes channels instead of board coordinates.
- Negating the value confuses spatial symmetry with player inversion.
- Returning rotation or flip views violates the independent-array requirement.
- Converting dtypes adds behaviour that the problem explicitly excludes.

Dihedral augmentation is one consistent coordinate change applied to every spatial part of a training record. The state planes and board-action probabilities move together, while pass, player perspective, value, and numeric types remain fixed.

---