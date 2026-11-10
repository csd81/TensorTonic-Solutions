# Beam Search Step

Greedy decoding keeps only the single best token at every step, and that can lock a model into a sequence that looked good early but scores poorly overall. Beam search keeps the K best partial sequences, called beams, alive at once and lets them compete at every step. This page starts from two beams and a three-token vocabulary, shows how candidates are scored and ranked, and then covers the bookkeeping that makes beam search tricky in practice, which is moving every beam's history and cache to follow the beam it came from.

## Scoring a sequence by adding log probabilities

A language model gives a probability for each next token. The probability of a whole sequence is the product of the probabilities of its tokens, and taking logs turns that product into a sum. Each beam therefore keeps a running score, the sum of the log probabilities of its tokens so far. A larger score, meaning one closer to zero, is a more likely sequence.

Extending a beam with score $s_k$ by a token $v$ whose log probability on that beam is $\log p_k(v)$ gives a candidate with total

$$
c_{k,v} = s_k + \log p_k(v)
$$

where $k$ indexes the current beams from 0 to K minus 1, and $v$ indexes the vocabulary from 0 to V minus 1. Every beam can be extended by every token, so one step produces K times V candidates.

## Two beams and three tokens

Take K = 2 beams with scores $s_0 = -0.5$ and $s_1 = -1.2$, and a vocabulary of V = 3 tokens. Beam 0 has history [2] and beam 1 has history [1]. The next-token log probabilities are $[-0.4, -1.6, -2.3]$ on beam 0 and $[-0.1, -2.5, -3.0]$ on beam 1.

The totals for beam 0 are $[-0.9, -2.1, -2.8]$ and for beam 1 are $[-1.3, -3.7, -4.2]$. The two best of all six are $-0.9$, from beam 0 with token 0, and $-1.3$, from beam 1 with token 0. The new beams are [2, 0] with score $-0.9$ and [1, 0] with score $-1.3$.

Notice that beam 1's best token had log probability $-0.1$, far better than beam 0's best at $-0.4$. Beam 0 still ranks first because its history was more likely. The ranking compares whole sequences, which is the entire point of carrying the running score.

## When one beam takes every place

Run one more step from the beams above, with scores $-0.9$ and $-1.3$. Suppose the log probabilities are $[-0.3, -0.6, -2.9]$ on beam 0 and $[-1.1, -0.9, -1.6]$ on beam 1. The totals are $[-1.2, -1.5, -3.8]$ for beam 0 and $[-2.4, -2.2, -2.9]$ for beam 1.

The two best totals, $-1.2$ and $-1.5$, both come from beam 0, with tokens 0 and 1. Both new beams extend the same parent, so both start from its history [2, 0] and become [2, 0, 0] and [2, 0, 1]. The old beam 1, [1, 0], has no children and drops out.

This case is why the selection must look at all K times V candidates together. Picking the best token of each beam separately would have kept [1, 0, 1] at $-2.2$ and thrown away [2, 0, 1] at $-1.5$, a much more likely sequence.

## Ranking all candidates with one top-k

Laying the totals out as a (K, V) grid and flattening it into one row of K times V entries turns the selection into a single top-k problem. In flat order, candidate $(k, v)$ sits at index $n = kV + v$, so beam 0's candidates come first, then beam 1's, and so on. In the first step above, the winners sat at flat indices 0 and 3.

Going back from a flat index to the candidate uses integer division and the remainder. The parent beam is $\lfloor n / V \rfloor$ and the token is $n \bmod V$. Index 3 with V = 3 gives parent 1 and token 0, which is exactly the second winner.

lax.top_k returns the K largest entries along the last axis, sorted from largest to smallest, together with their indices. With a batch of rows of shape (B, K V), it handles every batch row independently in one call. When two entries are equal, it returns the lower index first, so for an input of [1, 3, 3, 2] the top two are the 3s at indices 1 and 2. Real vocabularies are large, and with K = 4 and V = 50,000 each step ranks 200,000 candidates per row, which is why one fused top-k is much better than any loop.

## The first step needs one live beam

At the very first step there is only one sequence, the prompt, but the arrays still have K beams. If all K beams start with score 0 and the same log probabilities, every top token appears K times among the candidates. Take two beams with scores 0 and 0 and the same log probabilities $[-0.2, -1.9, -2.6]$ on both. Top-k with K = 2 returns $-0.2$ twice, at flat indices 0 and 3, which is token 0 on beam 0 and token 0 on beam 1. The second beam is then wasted on a duplicate.

The usual fix gives beam 0 a score of 0 and every other beam a very low score such as $-10^9$. All winners then come from beam 0 on the first step. With beam 1's score set to $-10^9$ in the example above, the top two become $-0.2$ and $-1.9$, at flat indices 0 and 1, which are tokens 0 and 1 of beam 0. After that step every beam is distinct and the search proceeds normally.

## Reordering histories by parent

After selection, new beam $i$ continues the sequence of beam parents[i]. Its history is the parent's history with the new token appended, so the old histories must be gathered along the beam axis with the parent indices before the token is attached. Appending tokens to the old histories in their old order would attach each token to the wrong sequence whenever a parent has zero or several children.

With a batch, each batch row has its own parents. The gather must use the parents of row $b$ for the beams of row $b$, which is a per-row gather along axis 1. Indexing axis 1 with the parents of only the first row, or broadcasting the parents against the wrong axis, mixes rows together. jax.vmap is a clean way to express a per-row operation. It takes a function written for one row, such as picking beams of one row with that row's parents, and maps it over the batch axis of all its arguments at once. jnp.take_along_axis is an alternative that gathers along one axis with an index array of matching rank.

## Reordering every cache leaf

A real decoder keeps more than histories per beam. The attention keys and values of every layer, any position counters and any other per-beam state live in a cache, and all of it must follow the parents in exactly the same way. If the histories are reordered but the cache is not, beam $i$ continues text from one parent while attending to the keys and values of another, and the model's next predictions are computed from the wrong context.

JAX represents such nested containers as pytrees, any nesting of dicts, lists and tuples whose leaves are arrays. jax.tree_util.tree_map applies a function to every leaf of a pytree and returns a new pytree with the same structure. Because each leaf starts with the axes (B, K), one reorder function works for all of them, whatever the remaining shape and dtype, and the code does not need to know the cache's key names or how many leaves it has.

## Shapes stay fixed from step to step

Beam search under jit fits well with fixed shapes. The number of beams stays K, the scores stay (B, K), and each step adds exactly one column to the histories. The selection itself is data dependent, since which parents win depends on the numbers, but that dependence lives entirely inside index arrays and never changes a shape. A Python if statement that inspects a score, or a conversion of a traced index to a Python int, fails during tracing.

## Common mistakes to avoid

- **Choosing the best tokens of each beam separately.** Beams must compete in one ranking over all K times V candidates, otherwise a strong beam cannot take several places.
- **Forgetting the running scores.** Ranking by the new token's log probability alone ignores how likely each beam's history already is.
- **Recovering the parent with the remainder.** The parent is the flat index divided by V, and the token is the remainder, so swapping them or dividing by K gives wrong beams.
- **Appending tokens without reordering the histories.** Each new token must be attached to its parent's history, or tokens end up on sequences they were never scored with.
- **Reordering some cache leaves and not others.** Every per-beam array must follow the parents, including integer counters, or the cache no longer matches the histories.
- **Using one batch row's parents for every row.** Each row of the batch has its own beams, so the gather must use that row's parent indices.