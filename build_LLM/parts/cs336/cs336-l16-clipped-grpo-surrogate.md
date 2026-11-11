# Compute a Clipped GRPO Surrogate

Policy-gradient training reuses each batch of sampled responses for a few update steps, and without a limit the policy can change far more than the data supports. The clipped surrogate objective from PPO (Schulman et al., 2017), which GRPO also uses, caps how much credit an update can take for moving each token's probability. The heart of this problem is computing that clipped objective over the response tokens selected by a mask, which leaves out prompt and padding tokens, along with a diagnostic that shows how often the cap is active.

## Why limit each update

Responses are sampled from the old policy, the model as it was when the batch was generated. After one gradient step the policy has changed, and the batch no longer reflects what the new policy would produce. A large step can make the update unreliable, sometimes collapsing a model that was training well.

The probability ratio measures how far the new policy has moved for each token:

$$
r_{it} = \exp\left( \operatorname{clip}\left(\ell^{\text{new}}_{it} - \ell^{\text{old}}_{it}, -60, 60\right) \right)
$$

where $\ell^{\text{new}}_{it}$ and $\ell^{\text{old}}_{it}$ are the new and old log probabilities of token $t$ in sequence $i$. A ratio of 1 means no change, and 1.5 means the new policy finds the token 50 percent more likely. In float32 the exponential overflows a little above 88, so the log ratio is bounded before exponentiating.

At the start of an update the ratio is 1, and the gradient of $r_{it} A_i$ equals the advantage $A_i$ times the gradient of the token's log probability, which is the ordinary policy gradient. As the policy drifts, the ratio reweights the old samples so they still estimate the new policy's objective.

## Taking the pessimistic minimum

With clip width $\epsilon$, each token's surrogate is

$$
s_{it} = \min\left( r_{it} A_i, \ \operatorname{clip}(r_{it}, 1 - \epsilon, 1 + \epsilon) A_i \right)
$$

The first term is the plain policy-gradient objective, and the second clips the ratio to the interval from $1 - \epsilon$ to $1 + \epsilon$. Taking the smaller of the two gives a pessimistic objective.

For a positive advantage, raising the token's probability helps, but any credit for raising it beyond $1 + \epsilon$ is removed, so there is no incentive to push further in one step. For a negative advantage, lowering the probability helps, and credit for lowering it below $1 - \epsilon$ is removed. In the other direction, when a change made things worse, the minimum keeps the full, unclipped penalty, so the objective never hides a harmful move.

Averaging the negative surrogate over every selected token in the batch gives each token equal weight, so long responses contribute more terms than short ones. How to weight long and short responses is a choice of its own, which the length-normalization problem explores.

## Measuring how often clipping happens

The clipped fraction is the share of selected tokens whose ratio has left the clip interval. A value near zero means almost every token's ratio has stayed within the interval, so the cap is rarely in play. A large value means updates are hitting the limit often, which usually suggests a learning rate that is too high or too many reuse steps per batch. Averaging the surrogate within each response shows which responses dominate the update.

## Four tokens, three outside the interval

Take two sequences of two tokens each, all selected, with $\epsilon = 0.2$, so the interval is 0.8 to 1.2.

The first sequence has advantage $+1$ and ratios 1.3 and 0.7. For the first token, $\min(1.3, 1.2) = 1.2$, so the extra credit above 1.2 is removed. For the second, $\min(0.7, 0.8) = 0.7$, the full penalty for having lowered a good token. Its mean is 0.95.

The second sequence has advantage $-1$ and ratios 0.7 and 1.0. For the first token, $\min(-0.7, -0.8) = -0.8$, so the credit for lowering a bad token below 0.8 is removed. The second token is unchanged and contributes $-1$. Its mean is $-0.9$.

The loss is $-(1.2 + 0.7 - 0.8 - 1.0) / 4 = -0.025$. Three of the four ratios, 1.3, 0.7 and 0.7, lie outside the interval, so the clipped fraction is 0.75, even though only two of those three surrogates were actually capped.

## Common mistakes to avoid

- **Taking the maximum of the two terms.** The objective uses the minimum, which is what makes it pessimistic.
- **Clipping the advantage.** Only the ratio is clipped, and the advantage multiplies both terms.
- **Averaging per sequence before averaging tokens.** The loss is the mean over all selected tokens in the batch.
- **Forgetting the sign of the loss.** The loss is the negative surrogate, because the surrogate is to be maximized.
- **Exponentiating the raw log difference.** A large gap between new and old log probabilities overflows to infinity and poisons the whole loss.
