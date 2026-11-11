# Speculative Decoding Acceptance

A large language model generates one token per forward pass, and each pass is slow because it has to read all of the model's weights from memory. Speculative decoding (Leviathan et al., 2023, and Chen et al., 2023) lets a small, fast draft model guess several tokens ahead and then has the large target model check all of the guesses in a single pass. A careful accept and reject rule keeps the output distributed exactly as if the target model had generated every token itself. This page builds that rule from a single token, shows why it is exact, and then turns the whole verification step into fixed-shape JAX array operations, which is what you implement here.

## Drafting and verifying

The draft model proposes G tokens one after another, $x_0, x_1, \dots, x_{G-1}$, sampling each from its own distribution $q_i$ over the vocabulary. The target model then runs once over the prompt plus all G draft tokens. Because a transformer produces a next-token distribution at every input position, that single pass gives the target distribution $p_i$ at each draft position, plus one more distribution $p_G$ for the position after the last draft token.

Verification walks through the draft tokens in order and decides for each one whether the target model would accept it. The first rejected token ends the step, and a replacement token is drawn in its place. If every draft token is accepted, the extra distribution $p_G$ supplies one bonus token for free. Either way a step yields between 1 and G + 1 new tokens for one pass of the large model.

## Accepting a single draft token

Start with one position and a vocabulary of three tokens. Suppose the draft distribution is $q = [0.6, 0.3, 0.1]$ and the target distribution is $p = [0.3, 0.5, 0.2]$. The draft model is overconfident about token 0 and underconfident about token 1.

The acceptance rule keeps a draft token $x$ with probability $\min(1, p(x) / q(x))$. In practice you draw a uniform number $u$ between 0 and 1 and accept when

$$
u < \frac{p(x)}{q(x)}
$$

which happens with exactly that probability. For token 0 the ratio is $0.3 / 0.6 = 0.5$, so a draft of token 0 is accepted half the time. For tokens 1 and 2 the ratios are $0.5 / 0.3$ and $0.2 / 0.1$, both above 1, so those drafts are always accepted. The draft model proposed token 0 more often than the target wants, and the rule throws away exactly the excess.

## The residual distribution fills the gap

When a draft is rejected, the step still has to produce a token, and it has to make up for the probability that the rejections removed. The replacement is drawn from the residual distribution, the positive part of the difference between the target and the draft, normalized to sum to 1.

$$
r(v) = \frac{\max\big(p(v) - q(v), 0\big)}{\sum_{w} \max\big(p(w) - q(w), 0\big)}
$$

Here $v$ is a vocabulary entry and $w$ runs over all of them. In the example, $p - q = [-0.3, 0.2, 0.1]$, the positive part is $[0, 0.2, 0.1]$, and normalizing gives $r = [0, 2/3, 1/3]$. Token 0 gets nothing because it was already over-proposed, while tokens 1 and 2 receive the probability that the draft model under-proposed.

## Why the output matches the target exactly

The probability that the step outputs token 0 is the chance of drafting it times the chance of keeping it, $0.6 \cdot 0.5 = 0.3$. Rejections only happen on token 0, so the total rejection probability is $0.6 \cdot 0.5 = 0.3$ as well. Token 1 is output when it is drafted, with probability $0.3$, or after a rejection when the residual picks it, $0.3 \cdot 2/3 = 0.2$, giving $0.5$ in total. Token 2 gets $0.1 + 0.3 \cdot 1/3 = 0.2$.

The result is $[0.3, 0.5, 0.2]$, exactly the target distribution. A simulation with 200,000 draws gave frequencies of about $[0.3012, 0.5004, 0.1984]$. The same argument works for any $p$ and $q$, because the total rejection probability always equals the total mass of $\max(p - q, 0)$, which here is 0.3 in both cases. Speculative decoding therefore changes the speed of generation without changing what the model says, as long as the rule is followed exactly.

## A chain of tests that stops at the first failure

With G draft tokens, each position has its own test with its own uniform $u_i$, ratio $p_i(x_i) / q_i(x_i)$ and pass or fail result. Draft token $i$ only counts if every draft token before it was accepted, because token $i$ was proposed as the continuation of those earlier tokens. Once position $n$ fails, the draft tokens after it were conditioned on a token that is now discarded, and they are thrown away even if their own tests would pass.

The number of accepted tokens $n$ is therefore the length of the run of passes at the start. For pass flags $[1, 1, 0, 1]$, the run is 2, and the final pass is ignored. The replacement token is then drawn from the residual of $p_n$ and $q_n$ at position $n$. When all G tests pass, $n = G$ and there is no draft distribution at that position, so the bonus token is drawn from $p_G$ alone, normalized the same way.

## How many tokens a step produces

If every draft token passes its test with the same probability $\alpha$, the expected number of tokens per step is

$$
\frac{1 - \alpha^{G+1}}{1 - \alpha}
$$

which counts the accepted run plus the one replacement or bonus token. With $\alpha = 0.8$ and G = 4 drafts, a step yields about 3.36 tokens for one pass of the large model. With $\alpha = 0.5$ the same G gives about 1.94, and with $\alpha = 0.8$ and G = 8 it gives about 4.33, which shows the diminishing return of drafting further ahead. A good draft model, one whose $q$ is close to $p$, is what makes the method pay off.

## Drawing a token from uniform noise

Sampling from a distribution with a given uniform number uses the inverse of the cumulative distribution. Add up the probabilities in vocabulary order to get the cumulative values $r(0)$, $r(0) + r(1)$, and so on up to 1. The sampled token is the first index whose cumulative value exceeds the uniform $s$. For $r = [0, 2/3, 1/3]$ the cumulative values are about $[0, 0.6667, 1]$, so $s = 0.5$ gives token 1 and $s = 0.8$ gives token 2. Each token is chosen with probability equal to the width of its interval, and a token with zero probability has an interval of zero width and can never be chosen.

This problem supplies all the uniforms as inputs, both the acceptance uniforms $u_i$ and one sampling uniform per row. In a real JAX program they would come from jax.random with an explicit key, and passing them in makes every result reproducible.

## Verification without data-dependent loops

The natural way to describe the chain is a loop that breaks at the first failure, but under jax.jit that loop cannot exist. The pass flags are traced arrays, so a Python if statement on one of them raises an error during tracing, and every batch row may stop at a different position.

The run length therefore has to come from array operations that look at all G pass flags of every row at once. The quantity needed is the length of the initial run of ones in a vector of 0 or 1 flags, and operations along the draft axis can produce it with no branch at all, so the compiled program is the same whatever the flags turn out to be.

Each row then needs the distributions at its own stopping position, so the program reads a different position in each row with one gather. The output layout is fixed as well. A row always has G + 1 slots, holding the accepted drafts, then the replacement or bonus token, then a padding value of -1, whatever the run length is. The serving system reads the run length to know how many slots are real.

## Common mistakes to avoid

- **Counting every passing test.** A pass after the first failure does not count, so the accepted count is the length of the initial run of passes.
- **Comparing the uniform with p alone.** The test uses the ratio of target to draft probability, and dropping the division by $q$ rejects good drafts and breaks the exactness argument.
- **Sampling the replacement from p.** After a rejection the token must come from the residual $\max(p - q, 0)$, otherwise tokens the draft over-proposed are sampled too often.
- **Forgetting to normalize the residual.** The residual sums to the rejection probability, which is below 1, so its raw cumulative values pick the wrong token.
- **Reading the next position after a rejection.** The replacement uses $p_n$ and $q_n$ at the rejected position $n$ itself.
- **Leaving stale tokens in the padding.** Slots after the replacement token must be -1, so draft tokens past the first rejection never look like output.