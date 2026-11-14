# Conditional Likelihood

LLaDA has no left-to-right factorization, so it cannot score a response by multiplying next-token probabilities. Instead it estimates a lower bound on the conditional log-likelihood $\log p_\theta(r_0 \mid p_0)$ by masking parts of the response and asking the mask predictor to recover them. This problem implements the fixed-count estimator of Eq. 6, which the paper uses for every likelihood-based benchmark because its variance is much lower than the estimator that falls directly out of training.

## From an exact product to a diffusion bound

An autoregressive model computes $\log p(r_0 \mid p_0) = \sum_i \log p(r_0^i \mid p_0, r_0^{1:i-1})$ exactly in one forward pass. LLaDA defines its model distribution as the result of a reverse masking process, so the exact likelihood is a marginal over every unmasking trajectory and has no closed form. What the paper can compute is the training objective, which Eq. 4 shows to be an upper bound on the negative log-likelihood. Negating any Monte Carlo estimate of that bound therefore gives an estimate of a lower bound on $\log p_\theta(r_0 \mid p_0)$, which is enough to rank candidate answers in multiple-choice evaluation.

The mask predictor here is the same bidirectional network used for generation. It reads the prompt $p_0$, which is never masked, together with a partially masked response, and predicts every masked token in one pass.

## Two estimators of the same bound

The SFT loss in Eq. 5 suggests the first estimator. Draw a ratio $t \sim U[0,1]$, mask each response token independently with probability $t$, and weight the masked log-probabilities by $1/t$. The number of masked tokens is then random, and for a short response it often lands far from $tL$, so the weight $1/t$ multiplies a sum that holds too many or too few terms.

Eq. 6 removes that source of noise. With $L$ the response length, it draws a count $l$ uniformly from $\{1, \dots, L\}$, masks exactly $l$ tokens chosen without replacement to form the noisy response $r_l$, and weights by $L/l$:

$$
-\mathbb{E}_{l, r_0, r_l}\left[\frac{L}{l} \sum_{i=1}^{L} \mathbb{1}[r_l^i = \mathrm{M}] \log p_\theta(r_0^i \mid p_0, r_l)\right]
$$

Here $\mathrm{M}$ is the mask token. The masked fraction is now exactly $l/L$, so the weight always matches the number of terms it multiplies. Both forms have the same expectation, but Appendix A.2 reports that the random-ratio form needs over 1000 Monte Carlo samples for stable results, while the fixed-count form is stable with 128, which is the setting used for evaluation in Appendix B.6.

## Conventions this exercise fixes

The paper draws each mask count independently. The official evaluation code instead spreads the counts of a mini-batch evenly over $\{1, \dots, L\}$ from one random start, a variance-reduction choice the paper does not mention. This exercise takes the counts as an input, so both schemes are valid inputs and the estimator itself is unchanged.

The paper says only that $l$ tokens are chosen uniformly without replacement. The code builds a boolean row whose first $l$ entries are true and permutes it, so offset $j$ is masked exactly when $\pi_j < l$, where $\pi$ is the permutation. The exercise uses this code convention. Reading the first $l$ entries of $\pi$ as positions selects a different set whenever $\pi$ is not its own inverse.

## One response, two samples

Take a prompt of length $P = 1$ and a response of length $L = 3$ whose true tokens have log-probabilities $g = (-0.5, -1.2, -0.3)$ at offsets 0, 1 and 2. The first sample has $l = 2$ and $\pi = (2, 0, 1)$. The offsets with $\pi_j < 2$ are 1 and 2, so the masked sum is $-1.2 - 0.3 = -1.5$, and the weight $3/2$ gives $e_1 = -2.25$. The second sample has $l = 3$, which masks everything, so $e_2 = -2.0$ with weight 1. The Monte Carlo mean is $-2.125$.

Reading $\pi_0, \pi_1 = 2, 0$ as positions would mask offsets 2 and 0 instead and give $1.5 \times (-0.8) = -1.2$, which shows how much the indexing convention matters.

## Common mistakes to avoid

- **Weighting by $l/L$.** The inverse ratio $L/l$ is what corrects for seeing only a fraction of the tokens, so the flipped weight shrinks the estimate exactly when few tokens are masked.
- **Dropping the factor $L$.** A weight of $1/l$ turns each sample into an average masked log-probability, which is a different quantity and no longer the bound of Eq. 6.
- **Reading the permutation as positions.** The mask follows $\pi_{k,j} < l_k$, and taking the first $l_k$ permutation entries as indices only agrees when the permutation is an involution.
- **Forgetting the prompt offset.** Response offset $j$ sits at sequence position $P + j$ in both the noisy copy and the logits, and masking or reading positions from zero mixes prompt and response.
- **Averaging over masked tokens.** Each sample sums its masked log-probabilities before the weight is applied, so dividing by the masked count applies $1/l$ twice.
- **Returning the loss.** The bound in Eq. 6 is a negative log-likelihood, while this function returns its negation, the log-likelihood estimate.