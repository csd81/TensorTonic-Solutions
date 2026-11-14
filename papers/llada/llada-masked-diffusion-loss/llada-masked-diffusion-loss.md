# Masked Diffusion Loss

The masked diffusion loss is the training objective that turns LLaDA's mask predictor into a principled generative model. It takes the cross-entropy of the clean token at every masked position, divides each term by the mask probability of its sequence, and normalizes each sequence by its response length. This problem implements that weighted, length-normalized cross-entropy for pretraining and supervised fine-tuning in one function, given a mask that was already drawn.

## Absorbing masks instead of Gaussian noise

LLaDA is a discrete diffusion model. Its forward process replaces tokens with an absorbing mask token instead of adding Gaussian noise, and its network predicts clean tokens with a softmax instead of predicting noise. Because the unmasked tokens of a noisy sequence are exactly the clean tokens, the paper shows that the predictor needs no timestep input, so the loss only ever compares predicted distributions with the original tokens at masked positions.

Write $x_0$ for a clean sequence of length $L$, $t$ for the mask ratio, $x_t$ for the noisy sequence, and $\mathrm{M}$ for the mask token. Eq. 3 of the paper defines the pretraining loss as

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[\frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \mathrm{M}] \log p_\theta(x_0^i \mid x_t)\right]
$$

## Why the inverse ratio is there

A masked language model such as MaskGIT averages the cross-entropy over masked tokens and stops there. LLaDA keeps the $1/t$ factor, and the paper stresses that this factor is what links the objective to maximum likelihood. Eq. 4 states that the loss upper-bounds the negative log-likelihood of the model distribution, so lowering it is guaranteed to raise a likelihood bound.

The weight also has an intuitive role. At ratio $t$ about $tL$ tokens are masked, so the plain sum grows with the ratio. Dividing by $t$ puts lightly masked and heavily masked sequences on a common scale, and a lightly masked sequence, which asks the model to fill in a few tokens from rich context, still contributes in proportion to its difficulty.

The released code does not divide by $t$ itself. It draws $t$ from $[0, 1)$ and uses the mask probability $p = (1 - \epsilon) t + \epsilon$ with $\epsilon = 10^{-3}$, which keeps the weight $1/p$ at or below 1000. This exercise follows the code and receives $p$ directly, so the weight is $1/p$.

## Pretraining and fine-tuning share one objective

Supervised fine-tuning splits each sequence into a prompt $p_0$ and a response $r_0$ and masks only the response, which is Eq. 5. Appendix B.1 pads short responses with end-of-sequence tokens and treats that padding as part of the response, so padded tokens are masked and scored like any other response token. A prompt length of zero turns fine-tuning back into pretraining.

The sources disagree on how to normalize. Eq. 3 and Eq. 5 are expectations of a sum with no length divisor. Algorithm 1 divides by $tL$ and Algorithm 2 by $tL'$, where $L'$ is the response length. The pretraining snippet in the official guidelines divides the summed weighted loss by batch size times sequence length, and the fine-tuning snippet divides each token term by its own sequence's response length before dividing by batch size. That snippet mixes singular and plural variable names, but the intended computation is unambiguous. This exercise fixes the fine-tuning form, a per-sequence division by $L'_b = L - P_b$ followed by a batch mean, which reduces exactly to the pretraining code when every prompt length is zero.

The choice matters once response lengths differ. A single token mean across the batch lets long responses dominate, while the per-sequence form gives every sequence equal weight in the batch average. Dividing every row by $L$ instead of $L'_b$ shrinks the loss of sequences with long prompts for no reason. This loss trains the same predictor that later drives the reverse sampling process and the conditional likelihood estimator, so its weighting decides what those procedures inherit.

## One fine-tuning row by hand

Take one sequence with $L = 4$, prompt length $P = 2$, vocabulary size 2 and mask probability $p = 0.5$. Both response positions are masked and both clean tokens are 0. Position 2 has logits $(1, 0)$, so its cross-entropy is $\log(1 + e^{-1}) = 0.3133$. Position 3 has logits $(0, 1)$, so its cross-entropy is $\log(1 + e) = 1.3133$.

Dividing by $p$ gives weighted losses $0.6265$ and $2.6265$, which sum to $3.2530$. The response length is $L' = 2$, so the loss is $1.6265$. Dividing by $L = 4$ would report half of that, and the unweighted MaskGIT mean would report $0.8133$, which is also half.

## Common mistakes to avoid

- **Dropping the inverse ratio.** An unweighted masked mean is the MaskGIT objective and no longer bounds the negative log-likelihood.
- **Multiplying by the ratio.** The weight is $1/p$, so heavily masked sequences are divided by a number near one and lightly masked ones are scaled up.
- **Scoring unmasked positions.** Clean context tokens carry no loss, even when the logits there are wrong.
- **Dividing every row by the full length.** Fine-tuning normalizes by $L - P_b$, which counts the padded end tokens but not the prompt.
- **Taking one token mean across the batch.** Each sequence is normalized on its own and the batch average comes afterward.
- **Computing the softmax and then its logarithm.** Exponentials of logits near 1000 overflow, so use a log-softmax that subtracts the maximum first.