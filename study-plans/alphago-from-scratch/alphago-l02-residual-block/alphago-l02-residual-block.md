# Understanding an AlphaGo Residual Block

A residual block transforms a board representation while preserving a direct path for the information that entered the block. Instead of asking two convolutions to construct an entirely new representation, the block asks them to learn a useful correction and adds that correction back to the original features.

## What enters the block

The input is a batch of feature maps. Each channel can be thought of as a different description of the same board locations. Earlier layers may have learned channels that respond to stones, liberties, local shapes, or combinations that do not have a simple human name.

Every operation in this block preserves the batch size, channel count, board height, and board width. This is necessary because the residual branch must eventually be added element by element to the unchanged shortcut.

## The shortcut and the residual branch

The block has two paths. The shortcut carries the input directly to the addition near the end. The residual branch passes the same input through two convolution and normalization stages. If the branch produces a correction $R(X)$, the addition forms:

$$
X + R(X)
$$

This structure gives the block an easy way to preserve useful features. When a channel already contains good information, the residual branch can make a small adjustment instead of recreating that information from scratch. If the branch becomes zero, the block reduces to applying the final ReLU to the shortcut.

## Why the convolutions use padding

Both convolutions use a three by three kernel. Such a kernel lets every output location combine information from its local neighbourhood. A point can respond to the feature values at its own location and the surrounding locations one step away.

Without padding, a three by three convolution would shrink the board after each use. Padding by one intersection on every side keeps the spatial dimensions unchanged. Stride one ensures that the kernel produces an output at every board location rather than skipping positions.

No convolution bias is used in this exercise. The normalization stage already has a learned shift for each channel, so an additional convolution bias would be redundant for the defined computation.

## Per-channel batch normalization

After each convolution, the result is normalized independently for every output channel. For one channel, the mean and population variance are computed across all batch items and all board locations. The channel axis is not averaged away because each channel has its own statistics.

For an activation $z$, the normalized value is:

$$
\widehat{z} = \frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

The supplied channel scale and shift then produce:

$$
\operatorname{BN}(z)=\gamma\widehat{z}+\beta
$$

The variance here is the population variance, which divides by the number of values rather than using the sample correction. Computing it as the mean of squared differences from the mean matches the problem exactly.

The positive constant epsilon prevents division by zero when a channel has no variation. This occurs naturally in small examples where every convolution output in a channel is identical. In that case, the centered values are zero, the normalized values remain zero, and the learned shift determines the channel output.

The scale and shift contain one value per channel. They need to act across every item and every board location in that channel. Conceptually, each value is expanded over the batch and spatial dimensions while remaining distinct across channels.

## The two stages are not identical

The first convolution is followed by normalization and ReLU:

$$
H=\operatorname{ReLU}\left(\operatorname{BN}_1\left(\operatorname{Conv}_1(X)\right)\right)
$$

ReLU replaces negative values with zero and leaves positive values unchanged. This introduces a nonlinear transformation between the two convolutions.

The second convolution is followed by normalization, but its result is not passed through ReLU immediately. It is first added to the original shortcut, and only then is ReLU applied:

$$
Y=\operatorname{ReLU}\left(X+\operatorname{BN}_2\left(\operatorname{Conv}_2(H)\right)\right)
$$

Moving that final ReLU before the addition would define a different block. The residual correction is allowed to contain negative values so it can reduce as well as increase shortcut features. The activation is applied after the two paths have been combined.

## A useful zero-branch check

Suppose both convolution weight tensors contain only zeros and both normalization shifts are zero. Each convolution output is zero, normalization remains zero, and the residual branch contributes nothing. The output is then simply the input passed through ReLU.

This explains why a positive input value survives, while a negative input value becomes zero. It is also a strong conceptual check: the shortcut is added after the second normalization, so a zero branch must not erase the input.

## Numeric type and device

The required computation uses 32-bit floating point. The input, convolution weights, channel scales, and shifts may arrive in other floating-point dtypes, so they should be interpreted consistently before arithmetic. The result must stay on the same device as the input.

The function must not modify any inputs. Convolution, normalization, addition, and ReLU can all produce new tensors, so there is no need for in-place changes.

## Common mistakes to avoid

- Omitting padding shrinks the residual branch and makes shortcut addition impossible.
- Averaging across the channel dimension mixes statistics that must remain separate.
- Using sample variance produces different values from the required population variance.
- Applying ReLU after the second normalization but before shortcut addition changes the residual block.
- Forgetting to add the original input removes the shortcut entirely.
- Adding a convolution bias introduces a term that is not part of the supplied parameters.
- Leaving some parameters in another type or on another device breaks the required output contract.

The complete idea is that two normalized convolutions learn a board-aware correction while the shortcut preserves the incoming representation. The first ReLU sits inside the residual branch, and the second acts only after the correction and shortcut have been combined.

---