# Perform Exact Speculative Sampling

Decoding a large model one token at a time is slow because each step reads the whole model from memory to produce a single token. Speculative sampling (Leviathan et al., 2023; Chen et al., 2023) lets a small draft model propose several tokens, then has the large target model check them all in one pass. The heart of this problem is the accept-and-replace rule that makes the final tokens follow the target model's distribution exactly, as if the target had generated them alone.

## Why speculate

A decode step of a large model is limited by memory bandwidth, so checking several tokens in one forward pass costs about the same as generating one. A small draft model is cheap enough to propose a handful of tokens quickly. If most proposals are accepted, the target model produces several tokens per pass instead of one.

The challenge is doing this without changing what the model says. A naive scheme that keeps any proposal the target assigns reasonable probability would bias the output toward the draft model. The acceptance rule below avoids that bias completely.

## Accepting a proposal

At draft position $i$, let $q_i(v)$ and $p_i(v)$ be the draft and target probabilities of token $v$, and let $x_i$ be the token the draft proposed. The proposal is accepted with probability

$$
a_i = \min\left( 1, \frac{p_i(x_i)}{q_i(x_i)} \right)
$$

If the target assigns at least as much probability to the token as the draft did, it is always accepted. If the target assigns less, the token is accepted with probability equal to the ratio of the target's probability to the draft's. In practice a uniform random value between 0 and 1 is compared with $a_i$. The ratio is undefined when $q_i(x_i) = 0$, a case that can arise from rounding in real systems, and treating any positive target probability as full acceptance keeps the rule consistent.

Positions are checked in order, and the first rejection ends the round, because every later proposal was conditioned on the rejected token.

## Replacing a rejected token

When proposal $i$ is rejected, a replacement is drawn from the residual distribution

$$
r_i(v) = \frac{\max(p_i(v) - q_i(v), 0)}{\sum_u \max(p_i(u) - q_i(u), 0)}
$$

where $u$ ranges over the vocabulary. It keeps only the tokens the target likes more than the draft does, in proportion to the difference. These are exactly the tokens the acceptance step under-produced.

## Why the output matches the target exactly

The draft proposes token $v$ with probability $q_i(v)$ and keeps it with probability $\min(1, p_i(v)/q_i(v))$, so $v$ is emitted directly with probability $\min(p_i(v), q_i(v))$. The total chance of a rejection is $\sum_u \max(p_i(u) - q_i(u), 0)$, and a rejection then picks $v$ with probability $r_i(v)$, which adds exactly $\max(p_i(v) - q_i(v), 0)$. The two parts sum to $p_i(v)$.

This guarantee makes the method exact, so the output has the same distribution as plain sampling from the target, and only the speed changes. Comparing one uniform random value against the running total of $r_i$ picks each token with probability equal to its share of the total, which is the standard way to sample from a discrete distribution.

## Where a round ends

A round emits the accepted prefix followed by the replacement, then stops. If every proposal is accepted, the round returns the proposals unchanged. The full algorithm then samples one extra token for free, because the target's forward pass has already produced the distribution for the next position, but this problem stops at the proposals.

## A three-proposal example

The draft proposed tokens 2, 1 and 0 over a vocabulary of three.

At the first position the draft probabilities are $[0.1, 0.3, 0.6]$ and the target probabilities are $[0.3, 0.3, 0.4]$, so token 2 has $a_0 = 0.4 / 0.6 = 0.666667$. An acceptance value of 0.5 accepts it. At the second position the draft gives token 1 probability 0.5 and the target 0.6, so $a_1 = 1$ and token 1 is accepted whatever the random value.

At the third position the draft probabilities are $[0.4, 0.4, 0.2]$ and the target probabilities are $[0.1, 0.3, 0.6]$. Token 0 has $a_2 = 0.1 / 0.4 = 0.25$, and an acceptance value of 0.7 rejects it. The positive differences are $[0, 0, 0.4]$, which normalize to $[0, 0, 1]$, so the replacement is token 2 for any residual random value. The round emits $[2, 1, 2]$, with two accepted proposals and a rejection at position 2.

## Common mistakes to avoid

- **Accepting when the random value equals the threshold.** Acceptance requires the value to be strictly less than $a_i$.
- **Continuing after a rejection.** Later proposals depend on the rejected token, so the round stops.
- **Sampling the replacement from the target distribution.** Drawing from $p$ gives extra probability to tokens the draft already covered, which biases the output toward the draft model.
- **Forgetting the zero-draft case.** A draft probability of zero gives acceptance 1 if the target probability is positive and 0 otherwise.
- **Adding a bonus token after full acceptance.** This problem returns the proposals unchanged when all are accepted.
- **Computing the residual in float32.** Small differences $p - q$ can round to zero or turn slightly negative, which distorts the residual.
