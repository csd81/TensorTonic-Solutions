# Stable LatentMoE

Stable LatentMoE combines two kinds of expert computation. Shared experts process every token at full model width, while routed experts process only selected tokens in a smaller latent space. This lets the model offer many specialized experts without sending the full-width representation through every selected expert.

## Shared and routed paths

The shared path captures transformations that are useful for all tokens. In this problem there are exactly two shared experts, and both process the original token independently. Their outputs are added directly to the final result.

The routed path is selective. A token is first compressed from model width into latent width:

$$
z = W_{\downarrow}x
$$

A router then assigns the token to a small set of experts. Only those selected experts run, and each works entirely in latent width. Their weighted result is normalized and projected back to model width.

Keeping these paths separate is important. Shared experts do not use the latent token in this exercise, and routed experts do not process the original full-width token.

## Select with bias, weight without bias

The router turns its projection into raw scores with sigmoid:

$$
r = \operatorname{sigmoid}(W_rx)
$$

Add the current expert bias when selecting top-k routes:

$$
\mathcal{T}_k(x)=\operatorname{TopK}(r+b,k)
$$

The bias changes which experts are selected, but mixture weights use only the chosen raw scores:

$$
p_i = \frac{r_i}{\sum_{j\in\mathcal{T}_k(x)}r_j}
$$

The selected weights therefore sum to one for every token. This is the same separation used by Quantile Balancing: bias controls load, while raw confidence controls the mixture.

## Run the selected latent experts

Every routed expert is a small feed-forward network using SiTU-GLU. It projects the latent token into gate and up branches, applies the smoothly capped gated activation, and uses its down projection to return to latent width.

For the selected set, combine expert outputs with the mixture weights:

$$
u = \sum_{i\in\mathcal{T}_k(x)} p_i E_i^{\mathrm{routed}}(z)
$$

Experts that were not selected must not contribute. A straightforward implementation may calculate selected experts token by token; efficiency tricks are not part of the conceptual requirement.

## Normalize before returning to model width

The routed aggregate can vary in scale depending on the token and chosen experts. Stable LatentMoE applies RMS normalization to $u$ immediately before the latent up-projection:

$$
y_{\mathrm{routed}} = W_{\uparrow}\operatorname{RMSNorm}(u)
$$

The order matters. Normalizing after the up-projection changes the computation, while normalizing individual expert outputs before mixing is also a different operation.

The final result adds both shared experts and the routed path:

$$
y = E_1^{\mathrm{shared}}(x)+E_2^{\mathrm{shared}}(x)+y_{\mathrm{routed}}
$$

Each shared expert also uses SiTU-GLU, but its projections operate at model width.

## A routing example

Suppose a token has raw router scores $[0.2,0.7,0.6]$, current bias $[0.5,0,0]$, and selects two experts. Biased selection scores are $[0.7,0.7,0.6]$, so experts 0 and 1 are selected under the implementation's top-k tie behavior.

Their mixture weights use raw values $0.2$ and $0.7$, not the biased values. The normalized weights are approximately $0.222$ and $0.778$. If the two latent expert outputs are $u_0$ and $u_1$, the aggregate is

$$
u = 0.222u_0 + 0.778u_1
$$

The example shows why bias and mixture weight must stay separate. Expert 0 became selectable because of bias, but it does not receive an artificially large contribution weight.

## Implementation order

- Compute the latent token representation and raw sigmoid router scores.
- Add current bias only for top-k selection.
- Gather selected raw scores and normalize them per token.
- Run the chosen latent experts with their own SiTU-GLU projections.
- Weight and sum selected expert outputs to form the latent aggregate.
- RMS-normalize the aggregate and apply the latent up-projection.
- Run both full-width shared experts on the original tokens and add all three paths.
- Return final output, selected indices, mixture weights, and latent aggregate.

## Common mistakes to avoid

- **Sending shared experts the latent token.** They operate on the original model-width input.
- **Using biased scores as mixture weights.** Bias affects selection only.
- **Normalizing each routed expert separately.** Normalize the weighted aggregate before the up-projection.
- **Running only one shared expert.** This exercise requires both supplied shared experts.
- **Returning the normalized aggregate.** The requested latent aggregate is the weighted expert result before RMS normalization.
