# Score Multiple-Choice Continuations

Many language model benchmarks ask a model to pick the best of several answers without generating any text. HellaSwag, a sentence-completion benchmark, and MMLU, a multi-subject exam benchmark, are both often run this way. The model scores each candidate continuation by how likely it finds those tokens, and the highest score wins. The heart of this problem is summing the right token log probabilities for each choice and picking the winner.

## Scoring answers without generation

Generating an answer and then parsing it is fragile, especially for small or base models that do not follow instructions reliably. Likelihood scoring avoids that. The prompt is followed by each candidate answer in turn, the model reads the combined text, and the log probabilities of the answer tokens are added up. A model that understands the question should assign more probability to the correct answer's tokens than to the wrong ones.

Every choice shares the same prompt, so only the continuation tokens are scored. A mask marks which positions belong to the continuation, and prompt and padding positions are excluded.

## Adding up log probabilities over the answer

For batch item $b$, choice $c$ and position $t$, let $z_{bct}$ be the logits over the vocabulary, $x_{bct}$ the target token and $m_{bct}$ the mask. The score of a choice is

$$
s_{bc} = \sum_t m_{bct} \log \operatorname{softmax}(z_{bct})_{x_{bct}}
$$

Each selected position adds the log probability of its target token, and masked positions add nothing. Log probabilities are at most 0, so scores are negative or zero, and a score closer to 0 means the model found the continuation more likely. The predicted choice is the one with the highest score:

$$
c_b^* = \operatorname*{arg\,max}_c \, s_{bc}
$$

A choice whose mask selects nothing contributes no evidence, so its score stays at the neutral value of 0.

In a causal language model, the logits at position $t$ predict the token at position $t + 1$, so scoring code normally shifts the targets one step left. Here each target already sits at the same position as the logits that predict it, so no further shift is applied.

## Why the sum favors short answers

The plain sum is the log probability of the whole answer as one event, which is why it is the default when answers have similar lengths. Every extra token multiplies in another probability below 1, though, so its log probability pushes the sum further down. Summing therefore penalizes long continuations, even when each of their tokens is predicted well.

Evaluation suites handle this in different ways. Some divide the sum by the number of tokens or characters to compare average likelihood per token, and some compare against the probability of the continuation without the question. This problem uses the plain sum.

## A length effect in one example

Take one question with two choices. Choice A is a single token that the model gives probability 0.25. Choice B is three tokens, each given probability 0.6. Their scores are

$$
s_A = \log 0.25 \approx -1.386294, \qquad s_B = 3 \log 0.6 \approx -1.532477
$$

Choice A wins, even though every token of B was predicted more confidently than A's only token. Under per-token averaging, B would score $\log 0.6 \approx -0.511$ against A's $-1.386$ and win instead. The two scoring rules can disagree, which is why benchmark reports state which one they use.

## Why log-softmax beats the log of a softmax

When one logit is far larger than the others, the softmax probability of a low-scoring token can underflow to exactly 0, and its logarithm becomes negative infinity. A log-softmax subtracts the largest logit before exponentiating and computes the log probabilities directly, which keeps every value finite and accurate.

## Where likelihood scoring fits in evaluation

Likelihood scoring is cheap and deterministic, since it needs one forward pass per choice and no sampling, so it is popular for tracking base models during pretraining. Its weakness is that it measures what the model finds likely, which is not always what it would say when asked to answer freely. Many evaluation suites report both a likelihood-based score and a generation-based score for this reason.

## Common mistakes to avoid

- **Scoring prompt tokens.** Only masked continuation positions add to a choice's score.
- **Forgetting the mask on padding.** Padded positions still have logits, and adding them changes a choice's score.
- **Shifting already-aligned targets.** An extra shift scores each token with the wrong position's prediction.
- **Picking the lowest score.** Scores are log likelihoods, so the highest score is the most likely choice.
- **Taking the log of a softmax.** A log-softmax avoids underflow for large logit gaps.
