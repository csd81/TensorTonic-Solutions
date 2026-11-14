# Classifier-Free Guidance

Classifier-free guidance is an inference-time knob that LLaDA borrows from image diffusion. It runs the mask predictor once with the prompt and once with the prompt hidden, then pushes the prediction further in the direction the prompt moved it. This problem covers the unsupervised variant that Appendix A.2 of the LLaDA paper (Nie et al., 2025) adopts from earlier masked diffusion work. It shows how the prompt-free input is built, how both branches share one forward pass, and how the two predictions are combined.

## What the mask predictor already offers

LLaDA's mask predictor reads a sequence in which some tokens have been replaced by the mask token $M$, and it predicts a distribution over the clean token at every masked position. During sampling, the prompt $p_0$ is fixed and the response $r_t$ is partially masked. The conditional prediction $p_\theta(r_0 \mid p_0, r_t)$ uses everything the prompt says.

Pretraining masks each token at a random ratio, so the model has already seen sequences in which large spans, including whole leading spans, are hidden. Replacing every prompt token with $M$ therefore produces an input the model can handle without any extra training, and its prediction $p_\theta(r_0 \mid m, r_t)$ reflects what the response would look like with no prompt at all. Here $m$ is a sequence of mask tokens of the prompt's length. This is why the paper calls the method unsupervised. Image diffusion models usually need a separate dropout of the conditioning signal during training to learn the unconditional branch, while LLaDA gets it from the masking it already learns.

## Conditional against prompt-masked prediction

Comparing the two branches isolates the prompt's effect. A token whose probability rises when the prompt is visible is one the prompt supports. Guidance with scale $w \ge 0$ amplifies that rise. Eq. 16 of the paper writes the guided predictor in probability space as

$$
\tilde{p}_\theta(r_0 \mid p_0, r_t) \propto \frac{p_\theta(r_0 \mid p_0, r_t)^{1+w}}{p_\theta(r_0 \mid m, r_t)^{w}}
$$

With $w = 0$ it is the ordinary conditional predictor. Larger $w$ rewards tokens the prompt favors and penalizes tokens the unconditional branch already liked, which trades diversity for prompt alignment. Appendix B.3 searches $w$ over 0.5, 1, 1.5 and 2. The main results keep guidance off so that the comparison with autoregressive models stays fair.

## Probability form against logit form

The official sampler and likelihood code combine logits instead of probabilities. With conditional logits $z_c$ and unconditional logits $z_u$ at one position, both vectors of length $V$, they compute

$$
\tilde{z} = z_u + (1 + w)(z_c - z_u) = (1 + w) z_c - w z_u
$$

A logit vector differs from its log-softmax only by one constant per position. Taking the logarithm of Eq. 16 gives $(1+w)\log p_c - w \log p_u$ plus a constant, which equals $(1+w) z_c - w z_u$ plus another constant. A softmax removes both constants, so the two forms give the same distribution. This exercise follows the code by returning the guided logits, and it also returns their log-softmax so the match with Eq. 16 is visible. The prompt-masked copy replaces every prompt position with the mask token and leaves the response untouched, including response positions that are already masked.

The code also concatenates the conditional and prompt-masked rows into one batch, runs the network once, and splits the output in half. Each row is processed independently, so the batched call returns the same logits as two separate calls while paying for one forward pass. The tiny predictor supplied here is a stand-in for the real bidirectional Transformer. It adds the sequence mean of the embeddings to every position, so hiding the prompt changes every response logit, just as bidirectional attention does in the full model.

## A two-token check

Take one position with conditional logits $z_c = (1.0, 0.0)$ and unconditional logits $z_u = (0.4, 0.2)$, and set $w = 0.5$. The guided logits are $1.5 \times (1.0, 0.0) - 0.5 \times (0.4, 0.2) = (1.3, -0.1)$. Their log-softmax is $(-0.220, -1.620)$, so the first token gets probability 0.802. The conditional branch alone would give it $\sigma(1.0) = 0.731$. The prompt raised the first token's advantage over the second from 0.2 to 1.0, and guidance stretched that gap to 1.4.

## Common mistakes to avoid

- **Dropping the one in the factor.** Using $w(z_c - z_u)$ instead of $(1 + w)(z_c - z_u)$ moves the prediction toward the unconditional branch, and at $w = 0$ it returns the unconditional logits.
- **Swapping the branches.** Extrapolating from the conditional logits toward the unconditional ones pushes predictions away from the prompt.
- **Masking the wrong span.** The unconditional copy hides the prompt and keeps the response, because guidance asks what the response looks like without the prompt.
- **Masking every position.** A fully masked copy discards the partially revealed response as well, so the two branches no longer differ only in the prompt.
- **Mixing probabilities linearly.** Eq. 16 is a ratio of powers, which becomes linear only in log space, and a linear mix of probabilities can even turn negative.
- **Calling the predictor twice.** Two calls give the same numbers on paper but double the forward passes the official code saves by batching.
- **Returning unnormalized logits as log-probabilities.** Guided logits carry an arbitrary per-position offset, so they need a log-softmax before they are probabilities.