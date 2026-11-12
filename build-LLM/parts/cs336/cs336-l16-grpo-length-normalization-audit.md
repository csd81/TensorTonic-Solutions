# Compare Sequence and Token Length Normalization

A policy-gradient loss for language models has to combine many tokens per response, and there is more than one way to do it. Summing over a response's tokens and averaging over them give different weights to long and short responses, and that choice shapes how long the model's answers become. The heart of this problem is computing both conventions for each sequence and measuring how far apart they are.

## Where the choice comes from

The policy-gradient objective for a response is its advantage times its log probability, and its gradient pushes that log probability up or down. A response's log probability is the sum of its tokens' log probabilities, so the natural contribution is

$$
g_i^{\text{sum}} = A_i \sum_t m_{it} \ell_{it}
$$

where $A_i$ is the response's advantage, $\ell_{it}$ is the log probability of token $t$, and $m_{it}$ is 1 for response tokens and 0 for prompt and padding tokens. The original GRPO objective instead divided each response's sum by its number of selected tokens $n_i$:

$$
g_i^{\text{avg}} = \frac{g_i^{\text{sum}}}{n_i}
$$

The clipped surrogate loss averages over every selected token in the whole batch, which is a third option. It gives every token equal weight like the sum, but its overall scale depends on how many tokens the batch contains.

## Why the convention matters

Under the summed form, every token carries the same weight, so a 500-token response contributes 50 times as many terms as a 10-token one. Under the averaged form, every response carries the same total weight, so each token in a long response counts for less.

That difference creates a bias. With averaging, each token of a long wrong answer is pushed down less than each token of a short wrong answer, because the response's penalty is spread over more tokens. Training therefore discourages long wrong answers less than short ones, which shows up as responses that grow steadily during training without getting better. Liu et al. (2025), in the paper that introduced Dr. GRPO, argued that this length normalization biases training toward longer incorrect responses, and their corrected objective drops the per-response division along with the per-group standard-deviation scaling.

## One number for the gap between conventions

A single number summarizes how much the two conventions differ across a batch:

$$
R = \frac{\sum_i |g_i^{\text{sum}}|}{\sum_i |g_i^{\text{avg}}|}
$$

Absolute values keep positive and negative contributions from canceling. If every response has the same length $n$, the ratio is exactly $n$, since averaging just rescales everything. When lengths differ, the ratio is an average of the response lengths weighted by the size of each averaged contribution, so it always lies between the shortest and longest lengths.

## A short right answer and a long wrong one

Take two responses. The first has advantage $+1$ and a single selected token with log probability $-2$. The second has advantage $-1$ and five selected tokens, each with log probability $-0.5$.

The summed contributions are

$$
g_1^{\text{sum}} = 1 \times (-2) = -2, \qquad g_2^{\text{sum}} = -1 \times (5 \times -0.5) = 2.5
$$

Averaging divides by 1 and 5, giving $-2$ and $0.5$. The ratio of total magnitudes is $(2 + 2.5) / (2 + 0.5) = 1.8$.

Under summing, the long wrong answer contributes more than the short right answer, as its five tokens each get pushed down. Under averaging, its weight shrinks to a quarter of the short answer's, so the update mostly reinforces the right answer and barely discourages the long wrong one. Over many steps, the weak penalty on long wrong answers lets them grow longer.

## Choosing a convention

Summing removes the length bias but lets long responses dominate the gradient, so real systems often divide the summed loss by a fixed constant, the batch's token count or the number of responses. Each choice changes the effective weight of every response, and with it the model's incentive to write longer or shorter answers.

## Common mistakes to avoid

- **Including masked tokens.** Prompt and padding tokens are excluded from both the sum and the count.
- **Dividing by the full sequence length.** The average divides by the number of selected tokens only, so padding would shrink every contribution.
- **Letting signs cancel in the ratio.** The ratio compares sums of absolute values.
- **Dropping the advantage's sign.** A negative advantage must flip the contribution, or wrong answers would be reinforced.
