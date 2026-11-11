# Implement the DPO Loss

Direct Preference Optimization (Rafailov et al., 2023) trains a language model on preference pairs without a separate reward model or reinforcement learning loop. It turns the RLHF objective into a simple classification loss on pairs of preferred and rejected responses. The heart of this problem is computing that loss from four sequence log probabilities per pair, along with the implicit rewards and accuracy used to monitor training.

## From RLHF to a classification loss

RLHF fits a reward model to preference pairs, then optimizes the policy to maximize reward minus a KL penalty against a reference model. That pipeline needs a separately trained reward model, usually a value model as well, and an RL loop that can be unstable.

The DPO paper showed that the optimal policy for that KL-regularized objective has a closed form, which can be rearranged to express the reward in terms of the policy itself,

$$
r(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

where $\pi$ is the policy being trained, $\pi_{\text{ref}}$ is the frozen reference model, $\beta$ is the KL strength, and $Z(x)$ depends only on the prompt $x$. The Bradley-Terry model gives the chance that a chosen response beats a rejected one as $\sigma(r^+ - r^-)$, the sigmoid of their reward difference. Both responses share the prompt, so $\beta \log Z(x)$ is identical for them and cancels in that difference, and the policy can be trained directly on preference pairs with a logistic loss.

## Comparing the policy's margin with the reference's

For pair $i$, let $\ell_i^+$ and $\ell_i^-$ be the policy's log probabilities of the preferred and rejected responses, each summed over the response's tokens as the closed-form derivation requires, and let $\ell_{\text{ref},i}^+$ and $\ell_{\text{ref},i}^-$ be the reference model's. The logit is

$$
z_i = \beta \left[ (\ell_i^+ - \ell_i^-) - (\ell_{\text{ref},i}^+ - \ell_{\text{ref},i}^-) \right]
$$

It compares how strongly the policy prefers the chosen response with how strongly the reference already did. A positive logit means the policy has moved toward the chosen response relative to where it started.

## The loss and label smoothing

The basic DPO loss is $-\log \sigma(z_i)$, the logistic loss for predicting that the chosen response wins. Human preference labels are noisy, so a smoothed version assumes each label is flipped with probability $\epsilon$:

$$
L_i = -(1 - \epsilon) \log \sigma(z_i) - \epsilon \log \sigma(-z_i)
$$

With $\epsilon = 0$ this is plain DPO. A small positive $\epsilon$ stops the loss from pushing the logit toward infinity on pairs that might be mislabeled. The batch loss is the mean of $L_i$.

When a logit is large in magnitude, one of $\sigma(z_i)$ or $\sigma(-z_i)$ underflows to 0, so taking the log of a sigmoid directly gives $\log 0$ in one of the two terms. A log-sigmoid function avoids that underflow.

## Reading the implicit rewards during training

Each response's implicit reward is $\beta$ times its log probability ratio:

$$
r_i^+ = \beta (\ell_i^+ - \ell_{\text{ref},i}^+), \qquad r_i^- = \beta (\ell_i^- - \ell_{\text{ref},i}^-)
$$

The logit is exactly $r_i^+ - r_i^-$. Preference accuracy, the fraction of pairs where the chosen reward beats the rejected one, tracks how often the policy has moved in the right direction. Watching the two rewards separately shows whether the model is raising the chosen responses, lowering the rejected ones, or both.

## A pair that looks wrong but is right

Take two pairs with $\beta = 0.1$ and no smoothing.

In the first pair the policy gives the chosen response a log probability of $-10$ and the rejected one $-12$, while the reference gives both $-11$. The logit is $0.1 \times (2 - 0) = 0.2$, the loss is $-\log \sigma(0.2) \approx 0.598139$, and the rewards are 0.1 and $-0.1$.

In the second pair the policy gives the chosen response $-8$ and the rejected one $-7$, so in raw terms it still prefers the rejected response. The reference gives $-9$ and $-7.5$, an even stronger preference for the rejected one. The logit is $0.1 \times (-1 - (-1.5)) = 0.05$, which is positive, and the loss is about 0.668460. The rewards are 0.1 and 0.05, so this pair counts as correct.

The mean loss is about 0.633299 and the accuracy is 1. DPO judges the policy against where it started, so a policy that has moved toward the chosen response is rewarded even if its raw probabilities still favor the other one. With $\epsilon = 0.1$ the per-pair losses rise to about 0.618139 and 0.673460, because the smoothing term adds a cost to every positive logit.

## Common mistakes to avoid

- **Leaving out the reference model.** Without the reference margin, the loss rewards preferences the policy already had before training.
- **Swapping chosen and rejected.** The chosen log probability comes first in each difference, and swapping them flips the sign of every logit.
- **Taking the log of a sigmoid directly.** Large logits underflow unless a log-sigmoid is used.
- **Applying smoothing to the wrong term.** The $\epsilon$ weight goes on $\log \sigma(-z_i)$.
- **Using only one response's log ratio.** The logit needs both the chosen and the rejected ratios.
