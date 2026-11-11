# Top-k and Top-p Sampling

At every decoding step a language model produces one score per vocabulary token, called a logit, and the inference code has to turn those scores into a single chosen token. Always taking the highest score gives dull, repetitive text, while sampling from the full distribution occasionally picks absurd tokens from the long tail. Temperature, top-k and top-p (also called nucleus sampling) are the three standard controls that sit between these extremes. This page builds them one at a time on tiny vocabularies, then shows how a uniform random number turns the final distribution into a token, and how to write all of it with fixed shapes so it compiles under jax.jit.

## From logits to probabilities

The softmax turns a row of logits $z_0, \dots, z_{V-1}$ over a vocabulary of V tokens into probabilities that are positive and sum to 1.

$$
p_j = \frac{e^{z_j}}{\sum_{m} e^{z_m}}
$$

With three tokens and logits $(2, 1, 0)$, the probabilities are about $(0.6652, 0.2447, 0.0900)$. The biggest logit gets the biggest share, but every token keeps some chance. Subtracting the largest logit from every logit before exponentiating does not change the result, and it stops $e^{z}$ from overflowing when logits are large.

## Temperature reshapes the distribution

Temperature is a positive number $\tau$ that divides every logit before the softmax. Dividing by a value below 1 stretches the gaps between logits, and dividing by a value above 1 squeezes them.

With the same logits $(2, 1, 0)$ and $\tau = 0.5$, the scaled logits are $(4, 2, 0)$ and the probabilities become about $(0.8668, 0.1173, 0.0159)$, so the top token dominates. With $\tau = 2$, the scaled logits are $(1, 0.5, 0)$ and the probabilities become about $(0.5065, 0.3072, 0.1863)$, which is much flatter. Temperature keeps the ranking of tokens the same and changes how sharply the probability is concentrated on the top of the ranking. That is why it is applied first, before any filter that looks at the ranking.

## Top-k keeps a fixed number of candidates

Top-k keeps the k tokens with the largest scaled logits and removes all the others. Removing a token means giving it probability 0, and the cleanest way to do that is to set its logit to negative infinity before the softmax, since $e^{-\infty} = 0$. The softmax then spreads the whole probability over the survivors.

For logits $(2, 1, 0)$ with $k = 2$, the third token is removed and the softmax of $(2, 1)$ gives about $(0.7311, 0.2689)$. Both survivors gained probability, because the share that the removed token held is now divided between them. With $k = 1$ only the top token survives, and sampling becomes the same as taking the highest score.

The weakness of top-k is that k is the same at every step. When the model is very sure, k candidates may include tokens with almost no probability, and when the model is unsure, k may cut off reasonable choices.

## Top-p adapts the number of candidates

Top-p keeps the smallest group of most likely tokens whose total probability reaches a threshold p. When the model is confident, a single token may already reach p, and when it is unsure, many tokens are needed. The group of kept tokens is called the nucleus.

To find the nucleus, sort the probabilities from largest to smallest and look at the mass ranked before each token, meaning the total probability of all tokens ranked above it. A token joins the nucleus when that mass is still less than p. This rule always keeps the top token, whose mass before it is 0, and it also keeps the token that pushes the running total past p.

Take sorted probabilities $(0.5, 0.3, 0.15, 0.05)$ and $p = 0.7$. The masses ranked before each token are $(0, 0.5, 0.8, 0.95)$. The first two are below 0.7, so the nucleus is the first two tokens, which together hold 0.8. The second token is kept even though the total it brings, 0.8, is above p, since without it the nucleus would hold only 0.5. Dividing by 0.8 renormalizes the nucleus to $(0.625, 0.375)$.

## The order of the steps matters

The three controls are combined in a fixed order. Temperature scales the logits, top-k removes all but the k best, the softmax runs over the survivors, and top-p is applied to those probabilities. Changing the order changes the answer. Running top-p on the full softmax before top-k, for example, measures the mass against a different distribution, so the nucleus can come out larger or smaller. In this problem the order is the one stated in the description, which matches common inference libraries.

## Turning a uniform number into a token

After filtering, each row has a final distribution with zeros for removed tokens. Inverse-CDF sampling picks a token from it using one number $u$ drawn uniformly from $[0, 1)$. Walk through the vocabulary in index order and keep a running total of probabilities, called the cumulative distribution $c_j$. The chosen token is the first index whose running total is greater than $u$.

Picture the interval from 0 to 1 cut into consecutive pieces, one per token, with lengths equal to the probabilities. The number $u$ lands in exactly one piece, and a token with probability 0.375 owns a piece of length 0.375, so it is chosen 37.5 percent of the time. A token with probability 0 owns an empty piece and can never be chosen.

Suppose the vocabulary has four tokens and the filtered, renormalized probabilities in index order are $(0, 0.375, 0, 0.625)$. The running totals are $(0, 0.375, 0.375, 1)$. With $u = 0.2$ the first total above 0.2 is at index 1, and with $u = 0.7$ it is at index 3. With $u = 0$ the walk also lands on index 1, because the total at index 0 is 0, which is not greater than 0, so a removed token can never be chosen. The order of the walk is part of the definition. Walking the tokens from most to least likely visits token 3 first with a running total of 0.625 and then token 1, so $u = 0.7$ would pick token 1 instead. Both orders give each token the same overall chance, but a fixed $u$ only gives a reproducible answer when the order is fixed too.

## Why the noise is passed in

In a real sampler, $u$ comes from jax.random.uniform with a PRNG key, which is JAX's explicit random state. Passing the uniform numbers in as an input separates the randomness from the sampling rule, which makes every call deterministic. The same logits and the same $u$ always give the same token, which is what makes sampling code testable and makes two runs of a server reproducible.

## Fixed shapes under jit

Under jax.jit every array argument is traced, which means the function is run once with stand-in values that know only their shape and dtype, and every intermediate array must have a shape that can be worked out from the input shapes alone. A boolean-mask index such as probs[probs > 0.1] returns as many elements as pass the test, so its shape depends on the data, and JAX refuses to trace it.

Filtering therefore keeps the full vocabulary axis everywhere. jnp.where chooses between two arrays element by element without changing shapes, so removed tokens can be given a logit of negative infinity or a probability of 0 while the array stays of shape (B, V). Values that set shapes, such as top_k, have to be static, meaning ordinary Python numbers fixed when the function is compiled. lax.top_k, which returns the k largest values of each row, needs k to be static for exactly this reason, since k is the length of its output.

## Common mistakes to avoid

- **Dropping the token that crosses top_p.** The nucleus keeps every token whose mass ranked before it is below top_p, so the token that pushes the total past top_p stays in.
- **Forgetting to renormalize after top-p.** Without dividing by the nucleus mass, the running total never reaches 1 and large values of $u$ fall past the last kept token.
- **Applying temperature after the softmax.** Temperature divides the logits, and dividing probabilities instead gives a distribution that does not sum to 1.
- **Applying top-p before top-k.** The nucleus is measured on the softmax of the top-k survivors, so swapping the order changes which tokens are kept.
- **Walking the cumulative sum in sorted order.** The token is chosen by walking the vocabulary in index order, and the sorted order gives a different token for the same $u$.
- **Filtering with boolean indexing.** Under jax.jit an index that keeps only some tokens has a data-dependent shape, so filters must keep the full vocabulary and use masks.
