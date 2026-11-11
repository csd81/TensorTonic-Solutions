# Canonicalize Pairwise Preference Data

Preference training methods such as reward modeling and Direct Preference Optimization (DPO) learn from pairs of responses where one is preferred over the other. Human labelers often rank several responses at once, because ranking four answers is faster than judging six separate pairs. The heart of this problem is expanding each complete ranking into every winner-loser pair it implies, in a fixed order, so the training data is the same every time it is built.

## Why rankings become pairs

Most reward models and DPO rest on the Bradley-Terry model. It models the chance that response $a$ beats response $b$ as $\sigma(r_a - r_b)$, where $r$ is a scalar score for each response and $\sigma$ is the logistic sigmoid. That formula compares exactly two responses, so every training example needs one preferred and one rejected response.

A ranking of $K$ responses contains more information than a single pair. The top response beats every other response, the second beats every response below it, and so on. Turning the ranking into pairs captures all of that, and InstructGPT (Ouyang et al., 2022) trained its reward model on all the pairs from each ranking of 4 to 9 responses.

## How many pairs a ranking gives

A complete ranking of $K$ candidates, with no ties, yields one pair for each unordered pair of candidates, which is

$$
\binom{K}{2} = \frac{K (K - 1)}{2}
$$

pairs. In each one the better-ranked candidate is the winner, and rank 1 is the best. Four candidates give 6 pairs and nine candidates give 36.

Pairs are formed only between candidates of the same prompt. Comparing responses to different prompts would make no sense, since the two responses answer different questions.

This problem assumes complete rankings with no ties. Real labeling interfaces often allow a labeler to mark two responses as equally good, and a tied pair says nothing about which response is better, so pipelines usually drop it or give it a label of one half. With a strict ranking every pair carries a clear winner.

## A canonical order

The same ranking can be expanded into pairs in many orders, and a different order changes how data is batched and shuffled, which changes training. A canonical order makes the output reproducible. Any fixed rule works as long as it depends only on the ranks and the prompt order, so the same records always expand to the same list. Here, within each prompt, pairs are listed by the winner's rank and then by the loser's rank.

## A shuffled ranking of four

Take one prompt whose four candidates are listed as c with rank 3, a with rank 1, d with rank 4 and b with rank 2. Sorting by rank gives the order a, b, c, d.

Expanding in canonical order gives six pairs:

- a beats b, a beats c, a beats d.
- b beats c, b beats d.
- c beats d.

Candidate a, the best, appears as the winner three times, and d, the worst, appears only as a loser. The order the candidates were listed in did not affect the result.

## Pairs from one ranking are not independent

All six pairs share the same prompt, and each candidate appears in three of them. If these pairs are scattered across training as independent examples, the model sees the same responses again and again and can overfit to them. InstructGPT handled this by keeping all pairs from one prompt together in a single batch element, which a canonical expansion that keeps each prompt's pairs adjacent makes easy.

The number of pairs also grows quickly with $K$. A ranking of nine responses gives 36 pairs, so a single heavily ranked prompt can dominate a dataset of pairs unless its pairs are weighted or grouped.

## Common mistakes to avoid

- **Pairing only adjacent ranks.** Every better-worse combination is a pair, so a ranking of $K$ candidates gives $K(K-1)/2$ pairs.
- **Using the listed order as the ranking.** Candidates must be sorted by their rank field first.
- **Pairing candidates across prompts.** Pairs only form within one prompt's candidates.
- **Including both orientations of a pair.** Each combination appears once, with the better candidate as winner.
- **Treating pairs from one ranking as independent samples.** They share candidates, so scattering them across batches overweights those responses.
