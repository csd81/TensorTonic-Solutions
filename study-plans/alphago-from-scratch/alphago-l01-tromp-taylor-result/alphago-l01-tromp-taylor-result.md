# Understanding Tromp-Taylor Area Scoring

This problem decides whether a Go position is finished and, only when it is finished, calculates the result. The scoring rule is easier to understand when the board is separated into occupied intersections and connected empty regions.

## First decide whether the game is over

Scoring a position that can still change would be premature, so terminal detection comes first. Under this problem's rules, the state is terminal when either of these conditions holds:

- The players have passed two times in succession.
- The number of played actions reaches two times the square of the board size.

The second condition is a practical move cap. On a board with side length $n$, the cap is:

$$
2n^2
$$

If neither condition is true, the function returns a false terminal flag and no scores or winner. Those absent values communicate that the result is not defined yet. Returning provisional numbers would make it too easy for later code to mistake an ongoing game for a completed one.

## Area scoring in plain language

Tromp-Taylor scoring is area based. A player's score includes the intersections occupied by that player's stones and the empty intersections that form territory belonging only to that player. White also receives komi, a fixed number of points that compensates for Black moving first.

The stone part is direct:

- Every Black stone contributes one point to Black.
- Every White stone contributes one point to White.

The empty part requires more care. Empty intersections are not judged one at a time. They are divided into connected regions using vertical and horizontal adjacency, just as stones are divided into groups.

## Empty regions and their boundaries

Take one unvisited empty intersection and trace every empty intersection connected to it. Together they form one region. While exploring the region, record the colours of any adjacent stones. That collection of colours is the region's boundary.

There are three meaningful outcomes:

- If the boundary contains only Black, the entire region belongs to Black.
- If the boundary contains only White, the entire region belongs to White.
- If the boundary contains both colours, the region is neutral and belongs to neither player.

An empty board, or an empty region with no bordering stones, also awards no territory. Its boundary contains neither colour, so it does not satisfy the requirement of being surrounded exclusively by one player.

The key idea is that ownership belongs to a whole connected region. Suppose a large empty area winds through several rows. A White stone touching one distant edge makes White part of that region's boundary even if most points are closer to Black stones. Looking only at the immediate neighbours of each empty point would wrongly split one region into conflicting local decisions.

## Flood filling each region once

A flood fill provides the needed region. A stack or queue stores empty points waiting to be examined, while a visited collection ensures that each empty point becomes part of exactly one region.

For each point in the region, inspect its valid orthogonal neighbours:

- An empty neighbour is part of the same region and should be explored.
- A Black neighbour adds Black to the boundary colours.
- A White neighbour adds White to the boundary colours.
- A direction outside the board contributes nothing.

When the frontier becomes empty, the size of the region and its complete boundary are known. The region size is added only when that boundary contains exactly one colour. A shared boundary remains neutral, and an empty boundary remains neutral as well.

The global visited collection is important because the outer board scan will encounter many points from the same region. Once a region has been scored, all of its points must be skipped in later iterations. Otherwise the same territory would be counted repeatedly.

## Building the two scores

Black begins with the number of Black stones. White begins with the number of White stones plus komi. Owned empty regions are then added to the corresponding score.

In mathematical form, if $B$ and $W$ are the sets of stones and $R$ ranges over connected empty regions, the scores are:

$$
S_B = |B| + \sum_{R:\operatorname{boundary}(R)=\{1\}} |R|
$$

$$
S_W = |W| + \sum_{R:\operatorname{boundary}(R)=\{-1\}} |R| + k
$$

Here $k$ is komi. The formulas say exactly what the flood fill does: count stones, then add the sizes of regions bordered by only one colour.

The required scores use 64-bit floating-point values. This matters because komi may contain a fractional point. Starting the score values with the requested NumPy type keeps integer stone counts and fractional komi in a consistent representation.

## Choosing the winner

After both scores are complete, compare them:

- Black wins when the Black score is larger, so the winner value is 1.
- White wins when the White score is larger, so the winner value is -1.
- Equal scores produce a winner value of 0.

The values 1 and -1 label the winning player; they are not score signs. Black counts occupied area and territory, while White also includes komi, which may be negative.

## Common mistakes to avoid

- Scoring before checking the terminal conditions returns a result for an ongoing game.
- Awarding each empty point from its immediate neighbours can misclassify a large connected region.
- Treating a region that touches both colours as territory gives points that should remain neutral.
- Flood filling without a global visited collection counts the same region more than once.
- Forgetting komi changes White's score and can change the winner.
- Modifying the board to mark visited points breaks the promise that the input remains unchanged.

The complete mental model is: establish that the game has ended, count occupied area, partition all empty intersections into connected regions, award only regions whose full boundary has one colour, add komi to White, and compare the final totals.

---