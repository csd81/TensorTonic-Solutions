# Forward Masking

LLaDA is a masked diffusion language model, and its forward process decides how clean text becomes corrupted training input. A mask ratio is drawn once per sequence, each token is hidden behind a reserved mask token with that probability, and the mask predictor is then trained to recover the hidden tokens. This problem covers that corruption step for both pretraining and supervised fine-tuning, with every random quantity supplied as an input.

## Absorbing masks instead of Gaussian noise

Continuous diffusion models such as DDPM corrupt data by adding Gaussian noise whose scale grows with time. LLaDA works on discrete tokens, so it replaces a token with a single absorbing mask token and never perturbs its value. Once a token is masked it carries no information about the original, and the mask predictor has to infer it from the visible context.

Writing $x_0$ for a clean sequence of length $L$ and $t \in [0, 1]$ for the time, the paper's forward process masks each token independently:

$$
q_{t \mid 0}\left(x_t^i \mid x_0^i\right) = \begin{cases} t, & x_t^i = \text{M} \\ 1 - t, & x_t^i = x_0^i \end{cases}
$$

Here M is the mask token and $x_t^i$ is position $i$ of the noisy sequence. At $t = 0$ the text is untouched, and at $t = 1$ every token is masked, which is the fully masked starting point of generation.

## One ratio per sequence, not one fixed ratio

BERT also hides tokens, but it uses one fixed ratio for every sequence. LLaDA draws a fresh ratio per sequence, so across training the model sees inputs that range from almost clean to almost fully masked. That spread is what turns masked prediction into a generative model, because the reverse process has to fill in text at every level of corruption, including the fully masked state where generation begins.

The ratio is shared by every token of its sequence. If each token got its own ratio, the number of masked tokens per sequence would concentrate around the average ratio, and the inputs would no longer follow the forward process that the training loss assumes.

## The shifted ratio in the official code

The paper writes the time as uniform on $[0, 1]$ in its loss and as uniform on $(0, 1]$ in its training algorithms. The official reference code draws $t$ from $[0, 1)$ and masks with the shifted probability

$$
p = (1 - \epsilon)\, t + \epsilon, \qquad \epsilon = 10^{-3}
$$

This exercise fixes that code form and takes both $t$ and $\epsilon$ as inputs. The shift keeps $p$ at least $\epsilon$, which matters because the training loss divides each masked token's cross-entropy by $p$, and a probability of zero would make that weight infinite. As a consequence, a ratio of exactly zero can still mask a token whose draw falls below $\epsilon$.

## Prompts stay clean during fine-tuning

For supervised fine-tuning, each example is a prompt $p_0$ followed by a response $r_0$. LLaDA leaves the prompt unchanged and masks only response tokens, because the fine-tuned model is meant to learn the conditional distribution of the response given a fully visible prompt. The prompt length is counted from the start of the sequence, and the rule is the same for pretraining with a prompt length of zero.

Short responses are padded with end-of-sequence tokens to a common length, and those padding tokens are treated as part of the response. They can be masked like any other response token, which teaches the model to predict where a response ends. This forward process feeds directly into the masked diffusion loss, which reads the mask indicator and the probabilities this function returns.

## A two-token prompt with two masked answers

Take a sequence of five tokens with a prompt length of 2, a ratio $t = 0.3$, and $\epsilon = 0.01$. The shifted probability is $0.99 \times 0.3 + 0.01 = 0.307$. Suppose the draws are $0.05, 0.6, 0.2, 0.31$ and $0.1$.

- Positions 0 and 1 are prompt positions, so they stay clean even though the draw at position 0 is below 0.307.
- Position 2 has draw 0.2, which is below 0.307, so it is masked.
- Position 3 has draw 0.31, just above 0.307, so it keeps its token.
- Position 4 has draw 0.1 and is masked.

The mask indicator is false, false, true, false, true, and exactly two response tokens are replaced by the mask token.

## Common mistakes to avoid

- **Flipping the comparison.** A token is masked when its draw is below the probability, so masking draws above it effectively uses the ratio $1 - p$.
- **Dropping the epsilon shift.** Comparing draws with the raw ratio changes the threshold slightly and returns probabilities that can be zero.
- **Masking the prompt.** Fine-tuning must leave every position before the prompt length untouched, even when its draw falls below the threshold.
- **Counting the prompt from the end.** The prompt occupies the first positions, so the protected region starts at position 0.
- **Using a separate ratio per token.** Every position of a sequence compares against the same probability.
- **Returning per-token probabilities.** The code repeats the probability across positions internally, but this function returns one value per sequence.