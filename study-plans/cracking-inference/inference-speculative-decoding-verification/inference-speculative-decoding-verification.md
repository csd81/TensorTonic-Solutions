# Speculative Decoding and Token Verification

Autoregressive generation usually asks a large target model for one token at a time. Speculative decoding introduces a smaller draft model that proposes several tokens, then lets the target model verify those proposals.

The draft model is useful because producing candidates can be cheaper. The target model remains the authority: an acceptance rule and a correction distribution ensure that verification follows the target probabilities rather than blindly trusting the draft.

This problem begins after both models’ probability distributions have been supplied. It verifies draft tokens in order, stops at the first rejection, and emits one additional sampled token.

## Two distributions at each draft position

At draft position $i$, let $q_i$ be the draft distribution and $p_i$ be the target distribution. The proposed token is $x_i$.

The verifier only needs the two probabilities assigned to that proposed token when deciding whether to accept it:

$$
q_i(x_i)\quad\text{and}\quad p_i(x_i)
$$

If the target gives the proposal at least as much probability as the draft does, the token can always be accepted. If the draft gives it more probability, it is accepted only some of the time.

## The acceptance probability

The required acceptance probability is

$$
\alpha_i=\min\left(1,\frac{p_i(x_i)}{q_i(x_i)}\right)
$$

A supplied uniform draw $u_i$ accepts the token exactly when

$$
u_i<\alpha_i
$$

The strict inequality matters at the boundary.

Suppose the draft assigns probability $0.4$ to its proposed token and the target assigns $0.6$. The ratio is $1.5$, which is capped at one, so every valid draw accepts it.

If the draft probability is $0.5$ and the target probability is $0.2$, then $\alpha_i=0.4$. A draw of $0.15$ accepts the proposal, while a draw of $0.40$ rejects it because acceptance requires the draw to be strictly smaller.

## The zero draft-probability edge case

The inputs deliberately allow a proposed token whose draft probability is zero. Such a proposal would not normally be sampled from that draft distribution, but the verifier must still handle the adversarial input without division by zero.

Under this problem’s explicit contract, set the acceptance probability to zero when $q_i(x_i)=0$. The supplied proposal is therefore rejected at that position. Valid inputs guarantee a positive residual sum at every rejection, including this zero-draft-probability case; a rejection with zero residual mass is outside the input domain.

This branch should be handled before attempting the ratio. Producing infinity and relying on later operations would contradict the required behavior.

## Verification stops at the first rejection

Draft tokens are checked from position zero onward. Every accepted token is appended to the emitted prefix.

As soon as one token is rejected:

- that rejected draft token is not emitted,
- no later draft token is checked or emitted,
- one correction token is sampled for the rejected position,
- verification for this group ends.

Later proposals depend on the rejected token being part of their draft history. Once that token is replaced, those later proposals no longer belong to the verified continuation.

If the third draft token is rejected after the first two were accepted, the accepted count is two, the returned prefix contains those two tokens, and the correction token is returned separately.

## Constructing the correction distribution

At a rejection, sampling directly from the target distribution would overcount probability mass already represented by accepted draft proposals. The required correction uses the positive residual between target and draft probabilities:

$$
r_i(x)=\max(0,p_i(x)-q_i(x))
$$

Negative differences become zero. Then normalize the residual:

$$
\hat r_i(x)=\frac{r_i(x)}{\sum_j r_i(j)}
$$

Suppose

$$
q_i=[0.6,0.3,0.1]
$$

and

$$
p_i=[0.2,0.5,0.3]
$$

The raw difference is $[-0.4,0.2,0.2]$. Clamping gives $[0,0.2,0.2]$, and normalization gives the correction distribution $[0,0.5,0.5]$.

Token 0 receives no correction probability because the draft already assigned it more mass than the target. The residual concentrates on tokens underrepresented by the draft.

## Sampling the correction token

The final supplied draw is used with inverse-CDF sampling. For a distribution $s$, choose the first token index whose cumulative probability is strictly greater than the draw $u$:

$$
t=\min\left\{j:\sum_{x=0}^{j}s(x)>u\right\}
$$

For the residual distribution $[0,0.5,0.5]$, cumulative probabilities are $[0,0.5,1]$. A draw of $0.2$ selects token 1, while a draw of $0.5$ selects token 2 because the search is right-sided.

The correction is sampled once and returned separately from the accepted prefix. Together they define the continuation.

## When every draft token is accepted

If all $K$ proposed tokens pass verification, the accepted prefix contains all $K$ draft tokens. The procedure also samples one bonus token from the target distribution at the next position.

That distribution is the extra row in the target input, at index $K$. It has no aligned draft distribution because the draft proposal group has already ended.

The same final uniform draw and right-sided inverse-CDF rule select the bonus token. The returned prefix has length $K$, the accepted count is $K$, and the bonus token is returned separately.

This bonus token is different from a correction. A correction uses the residual distribution at the first rejected position. A bonus uses the target’s own next-position distribution after complete acceptance.

## Following the shapes

The draft token IDs have shape $(K)$. Draft distributions have shape $(K,V)$ for vocabulary size $V$.

Target distributions have shape $(K+1,V)$. Rows zero through $K-1$ verify the aligned draft positions, and row $K$ supplies the bonus distribution.

Uniform draws have shape $(K+1)$. The first $K$ draws are acceptance tests. The last draw is reserved for the single correction or bonus sample.

The function returns a tuple of three integer PyTorch tensors: the one-dimensional accepted prefix, the scalar accepted count from zero through $K$, and the scalar next-token ID. The correction or bonus token is not appended to the returned prefix.

## Why deterministic draws are supplied

Acceptance and final sampling are probabilistic operations. Supplying the draws as inputs makes every branch reproducible.

The verifier must not generate additional random numbers. Doing so would make identical inputs produce different answers and would prevent exact testing of boundary cases.

Draw placement also matters. The final draw is not consumed for a later acceptance test after an earlier rejection. It is reserved for the correction or bonus token.

## Complexity and memory

At most $K$ draft positions are checked. Looking up the proposed token’s two probabilities is constant work per checked position.

If a rejection occurs, building and sampling the residual examines all $V$ vocabulary entries once. If every token is accepted, sampling the bonus distribution also examines $V$ entries.

The worst-case running time is therefore $O(K+V)$ for the supplied distributions. The input storage itself is $O(KV)$, while temporary residual and cumulative vectors use $O(V)$ memory.

This function does not run either model. It verifies probability tensors that have already been calculated.

## Common mistakes to avoid

- Comparing the target and draft probabilities without taking their ratio gives the wrong acceptance chance.
- Forgetting to cap the ratio at one can produce an invalid probability above one.
- Accepting when the draw equals $\alpha_i$ violates the strict boundary rule.
- Dividing by a zero draft probability raises an error and misses the required rejection case.
- Continuing to verify tokens after the first rejection uses proposals based on an invalidated history.
- Sampling a correction from the full target distribution ignores the positive residual rule.
- Normalizing target minus draft before clamping can leave negative probabilities.
- Using the last target row for a rejection confuses the bonus distribution with the correction distribution.
- Drawing more than one correction or bonus token changes the output contract.

Speculative verification is a controlled handoff between two distributions. The draft proposes a short path, the target accepts only the portion allowed by the probability ratio, and the residual or bonus sample supplies the one token that safely continues generation.
