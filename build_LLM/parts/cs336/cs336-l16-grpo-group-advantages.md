# Normalize GRPO Group Advantages

Group Relative Policy Optimization (GRPO), introduced with DeepSeekMath (Shao et al., 2024), trains a language model with reinforcement learning without training a second network to predict expected reward. It samples several responses to the same prompt and judges each one against its siblings. The heart of this problem is turning raw rewards into advantages by normalizing within each prompt's group, and recognizing groups that carry no learning signal.

## Why compare responses to the same prompt

A policy-gradient update needs an advantage for each response, a number that says how much better or worse it was than expected. Proximal Policy Optimization (PPO) estimates the expected reward with a learned value network, a second model about as large as the policy that must be trained alongside it and roughly doubles the memory needed.

GRPO replaces the value network with a simple baseline. For each prompt it samples a group of responses, and the average reward of that group stands in for the expected reward. A response that beats its siblings gets a positive advantage, and one that falls short gets a negative advantage. Comparing within a prompt also removes differences in difficulty between prompts, since an easy prompt where every response scores well does not make all of them look good.

## Centering and scaling within a group

For response $i$ in group $g$ with reward $r_i$, let $\mu_g$ be the group's mean reward and $\sigma_g$ its population standard deviation, computed with a divisor equal to the group size. The advantage is

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

where $\epsilon$ is a small positive constant that keeps the division finite when the spread is tiny but nonzero. Subtracting the mean makes the advantages in each group sum to zero. Dividing by the standard deviation puts every group on a comparable scale, so a group whose rewards range from 0 to 1 and one whose rewards range from 0 to 100 produce updates of similar size.

## Groups with nothing to learn

If every response in a group gets the same reward, there is no way to tell which was better. Mathematically the standard deviation is zero and every advantage would be zero. In floating point, though, the computed mean of identical rewards can differ from them by a rounding error, and dividing that tiny residue by $\sigma_g + \epsilon$, itself around $10^{-8}$, can turn noise into a large, meaningless advantage. A constant group is therefore detected directly and given exactly zero advantages.

Such a group is inactive, since it contributes nothing to the advantage-weighted policy-gradient term. Inactive groups are common in practice. With a correctness reward, a prompt that is too easy, where every response is right, or too hard, where every response is wrong, gives an inactive group. Counting active groups shows how much of a batch is actually teaching the model, and some training recipes filter or resample prompts to keep that fraction high.

## A rare success stands out

Take two groups. The first has four responses with rewards 1, 0, 0 and 0, so only one response solved the problem. The second has two responses that both scored 1.

The first group's mean is 0.25, and its population standard deviation is

$$
\sigma = \sqrt{\frac{0.75^2 + 3 \times 0.25^2}{4}} = \sqrt{0.1875} \approx 0.433013
$$

The successful response gets $0.75 / 0.433013 \approx 1.732051$, and each failure gets $-0.25 / 0.433013 \approx -0.577350$. The single success receives a large positive advantage, three times the size of each failure's penalty, because succeeding was rare in that group.

The second group has identical rewards, so both advantages are 0 and the group is inactive. Only one of the two groups is active.

## Where advantages go next

Each response's advantage is shared by all of its tokens. The clipped surrogate objective, a PPO-style loss that limits how far one update can move each token's probability, multiplies each token's change by that advantage, so the update favors whichever responses did best on each prompt.

## Common mistakes to avoid

- **Normalizing over the whole batch.** Each prompt's group has its own mean and standard deviation.
- **Using the sample standard deviation.** For a group of two, dividing by one instead of two shrinks every advantage by a factor of $\sqrt{2}$.
- **Relying on epsilon to handle constant groups.** Rounding error in the mean divided by a tiny denominator produces large spurious advantages, so constant groups must be caught explicitly.
- **Forgetting to subtract the mean.** Unnormalized rewards would push every response the same way on easy prompts.
