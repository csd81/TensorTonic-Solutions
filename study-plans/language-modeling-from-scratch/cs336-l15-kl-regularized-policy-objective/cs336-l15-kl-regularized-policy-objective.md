# Score a KL-Regularized Policy Batch

Reinforcement learning from human feedback (RLHF) improves a language model by rewarding good responses, but chasing reward alone lets the model drift into strange text that fools the reward model. The standard fix subtracts a penalty for moving away from a reference model. The heart of this problem is computing that penalty from sampled log probabilities, applying it to each response's reward, and averaging into the training objective.

## Why a reward alone is not enough

A reward model is a learned approximation of human preferences, and like any learned model it has blind spots. A policy optimized hard against it can find responses that score highly without being good, such as repeating phrases the reward model likes or drifting into odd styles. This is called reward hacking.

Keeping the policy close to a trusted reference model, usually the supervised fine-tuned model it started from, limits how far it can drift. InstructGPT (Ouyang et al., 2022) and earlier summarization work (Stiennon et al., 2020) both trained against reward minus a penalty on the KL divergence from the reference.

## Estimating KL divergence from samples

Let $\pi$ be the policy and $\pi_{\text{ref}}$ the reference model. The KL divergence between them is

$$
D_{\text{KL}}(\pi \,\|\, \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[ \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} \right]
$$

the expected log ratio of their probabilities over responses $y$ sampled from the policy. Computing the expectation exactly would sum over every possible response, which is impossible, so it is replaced by the log ratio of one sampled response.

A sequence's log probability is the sum of its tokens' log probabilities, so the difference of two sequence log probabilities is the sum of the per-token differences. For response $i$ and token $t$, let $\ell^\pi_{it}$ and $\ell^{\text{ref}}_{it}$ be the policy's and the reference's log probabilities of the sampled token, and let $m_{it}$ mark tokens that belong to the response. The sampled sequence divergence is

$$
\widehat{D}_i = \sum_t m_{it} \left( \ell^\pi_{it} - \ell^{\text{ref}}_{it} \right)
$$

In expectation over responses sampled from the policy, this equals the true KL divergence, which is never negative. A single sample can still come out negative, when the reference happened to find that particular response more likely, and the estimate must keep that sign. Clamping negative values to 0 would bias the average upward. The divergence of a whole sequence grows with its length, just like its log probability, so the sum is left unnormalized.

## The penalized reward and objective

With reward $r_i$ and penalty coefficient $\beta$, the penalized reward is

$$
\widetilde{r}_i = r_i - \beta \widehat{D}_i
$$

and the objective for a batch of $B$ responses is their mean, $J = \frac{1}{B} \sum_i \widetilde{r}_i$, which the policy update tries to increase. A larger $\beta$ keeps the policy closer to the reference, and $\beta = 0$ ignores the reference entirely. In PPO-style RLHF, these penalized rewards replace the raw reward model scores as the signal the policy-gradient update maximizes.

## A batch with one negative divergence

Take two responses with $\beta = 0.1$.

The first response has three tokens, all in the response, with policy log probabilities $[-0.5, -1.0, -0.2]$ and reference log probabilities $[-1.0, -1.0, -0.5]$. The differences are 0.5, 0 and 0.3, so $\widehat{D}_1 = 0.8$. With reward 2, the penalized reward is $2 - 0.1 \times 0.8 = 1.92$.

The second response has policy log probabilities $[-2, -1]$ and reference log probabilities $[-1.5, -1.2]$, but only its first token is in the response. That token gives $\widehat{D}_2 = -2 - (-1.5) = -0.5$, and the masked second token contributes nothing. With reward 1, the penalized reward is $1 - 0.1 \times (-0.5) = 1.05$, slightly above the raw reward.

The objective is $(1.92 + 1.05) / 2 = 1.485$. The second response gained a little because the reference gave its selected token a higher log probability than the policy did. The penalty term therefore rewards the policy for moving that probability back toward the reference.

## Common mistakes to avoid

- **Clamping negative divergences.** Single-sample estimates can be negative, and clamping them biases the KL penalty upward.
- **Dividing by response length.** The sequence divergence is a plain sum over selected tokens.
- **Applying the mask after summing.** Masked positions must be zeroed before the sum, or prompt and padding tokens leak into the divergence.
- **Subtracting in the wrong order.** The divergence is the policy log probability minus the reference log probability, and reversing it rewards drifting away.
- **Averaging the divergence over the batch before penalizing.** Each response gets its own penalty, and the mean is taken only at the end.
