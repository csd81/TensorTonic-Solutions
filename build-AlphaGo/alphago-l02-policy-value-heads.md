# Understanding the Policy and Value Heads

The shared network features describe one Go position, but the search needs two different predictions from them. The policy head estimates which actions look promising, while the value head estimates how favourable the position is for the player whose turn it is. These heads share the same input features and then use separate parameters for their separate jobs.

## One representation, two questions

The feature tensor entering the heads has already been processed by the network trunk. Its channels contain spatial patterns over the board. The two branches read this common representation but do not share their later convolution, normalization, or linear parameters.

The policy branch asks: how should preference be distributed across every board action and pass? The value branch asks: what single outcome estimate summarizes this position from the current player's perspective?

Keeping these questions in one network is useful because both depend on understanding the same board. A shape that makes an action promising can also affect the likely winner, yet each output needs its own final transformation.

## The policy branch

The policy branch begins with a one by one convolution that produces two channels. A one by one kernel does not mix neighbouring board locations. At each location, it learns a new combination of the shared feature channels. This compresses the trunk representation while preserving the board layout.

The two policy channels are normalized independently across the batch and spatial axes. Each channel uses the supplied scale and shift, followed by ReLU. The board-shaped activations are then flattened for each batch item and passed through a linear layer.

The linear layer produces one number for every intersection action plus one final number for pass. These numbers are logits. A logit is an unrestricted score, so it can be positive, negative, or zero.

The function must return raw logits. It does not apply softmax and does not remove illegal actions. Those decisions belong to later search logic, where the legal mask is known and the probabilities can be normalized over only legal actions. Applying softmax here would change the required output and make legal-only normalization less direct.

For feature-map height H and width W, the number of policy outputs is:

$$
HW+1
$$

The extra output is pass, using the same final action convention as the legal move mask.

## The value branch

The value branch starts from the same shared features but uses its own one by one convolution. This convolution produces one channel, which is normalized, shifted, scaled, and passed through ReLU.

After flattening, a hidden linear layer combines evidence from the entire board. ReLU follows this hidden projection. A final linear layer reduces the hidden representation to one scalar per batch item.

That scalar is passed through the hyperbolic tangent:

$$
v=\tanh(z)
$$

The result lies between -1 and 1. A value near 1 means the position is judged favourable for the current player, a value near -1 means it is judged unfavourable, and a value near 0 indicates a balanced or uncertain estimate. The exact training meaning is not computed in this problem, but the bounded output is the value signal expected by search.

The final single-value dimension should be removed so the branch returns one value per batch item rather than a column with an unnecessary trailing dimension.

## Normalization inside each branch

Both heads use the same normalization rule but separate statistics and parameters. For each output channel, compute the mean and population variance over all batch items and board positions:

$$
\widehat{z}=\frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

Then apply the channel's supplied scale and shift:

$$
\operatorname{BN}(z)=\gamma\widehat{z}+\beta
$$

The policy convolution has two channels, so it has two means, two variances, two scales, and two shifts. The value convolution has one of each. Statistics from the policy branch must not be reused for the value branch because their activations and channel counts differ.

The variance is the population variance, calculated as the mean squared distance from the channel mean. The positive normalization constant keeps a constant channel well defined. When every value in a channel is identical, the centered activation is zero and the learned shift remains after normalization.

## Flattening preserves batch ownership

Flattening should combine only the channel and spatial dimensions within each batch item. It must not merge different batch items together. Every position in the batch represents a separate Go state and must produce its own policy vector and value.

The expected policy linear weights assume a particular flattened feature count, and the value hidden weights assume another. If the batch dimension is flattened accidentally, both the mathematical meaning and the matrix dimensions become wrong.

## Numeric type, device, and input safety

Every calculation and both outputs must use 32-bit floating point on the feature tensor's device. Supplied weights, biases, scales, and shifts should participate using that same type and device. This avoids accidental precision differences and device conflicts.

The shared feature tensor and parameter tensors must remain unchanged. The two heads are readers of the same representation. Neither branch should edit the features before the other branch uses them.

## Common mistakes to avoid

- Applying softmax inside the policy head returns probabilities instead of the required logits.
- Masking illegal actions here mixes game-state logic into a branch that does not receive the legal mask.
- Sharing normalization statistics between the two heads confuses different activations.
- Flattening across the batch dimension combines separate positions into one example.
- Omitting the pass output produces too few policy logits.
- Forgetting tanh leaves the value estimate unbounded.
- Returning the value with an extra one-element dimension violates the required one-value-per-item result.

The policy and value heads are two specialized readings of one shared board representation. The policy branch preserves action-specific preferences as raw logits, while the value branch compresses the position to one bounded current-player estimate.

---