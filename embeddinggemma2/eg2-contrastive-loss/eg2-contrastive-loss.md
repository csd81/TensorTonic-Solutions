# Contrastive Loss

The contrastive loss is the main objective of the EmbeddingGemma 2 finetuning stage, where it replaces the distillation loss that dominated earlier training. It pulls each query toward its positive target and pushes it away from everything else in the batch. This problem covers the paper's extended form, which adds same-tower negatives and a duplicate mask on top of the familiar InfoNCE loss.

## From InfoNCE to four kinds of negatives

Plain InfoNCE treats a batch of $B$ query and target pairs as $B$ classification problems. Query $q_i$ must pick its own target $p_i^+$ out of all $B$ targets, so the denominator holds the positive plus the $B - 1$ other targets of the batch. These in-batch negatives are free but mostly easy.

Hard negatives fix that weakness. Each example brings $H$ mined targets $p_{i,h}^-$ that look relevant but are wrong, which forces the model to separate near misses. With in-batch and hard negatives together, the denominator is $Z_i^P + Z_i^{HN} + Z_i^{IB}$.

EmbeddingGemma 2 adds a fourth term. The same-tower term $Z_i^{STN}$ compares $q_i$ with other queries $q_j$ and $p_i^+$ with other targets $p_j^+$.

## Same-tower negatives keep the towers mixed

A loss that only compares queries with targets has a blind spot. It can be satisfied by an embedding space in which all queries live in one region and all targets live in another, as long as each query is slightly closer to its own target than to the others. The paper adds same-tower negatives, following Moiseev et al., 2023, "to prevent queries and targets from clustering into disjoint subspaces." Pushing $q_i$ away from $q_j$ and $p_i^+$ away from $p_j^+$ spreads each tower across the sphere, so a query and its target end up close in absolute terms and not only relative to the batch.

## The mask removes duplicates and the diagonal

Large training mixtures contain repeated texts. If two examples share a query, the positive of one is a correct answer for the other, and treating it as a negative punishes the model for being right. The indicator $\mathbb{1}_{TN}(i, j)$ is zero whenever queries $i$ and $j$ are duplicates or positives $i$ and $j$ are duplicates, and that zero removes example $j$ from all three in-batch and same-tower sums of row $i$.

The same rule handles the diagonal for free. Example $i$ is always a duplicate of itself, so $\mathbb{1}_{TN}(i, i) = 0$. Without that, $Z_i^{IB}$ would count the positive a second time, and $Z_i^{STN}$ would add $e^{1/\tau}$ twice for the self-similarities of $q_i$ and $p_i^+$. Hard negatives are never masked.

In this exercise, duplicates are decided by integer labels supplied with the batch, and the temperature and the number of hard negatives are inputs. The paper fixes none of these values, so the exercise chooses them.

## Why hardness weighting was dropped

EmbeddingGemma 1 multiplied each hard negative by a weight that grew with how similar it was to the query. EmbeddingGemma 2 removes it, because the paper says it "amplifies noisy or weakly associated positive pairs, especially during early training, and compounds gradient scale imbalances between tasks of different modalities." Every negative now contributes its plain exponential, which leaves the loss as a single log-softmax over one row of logits. The paper uses this loss with a small weight in text pre-finetuning and as the main objective in multimodal finetuning, where it lets the model move past imitating its teacher. A listwise Plackett-Luce variant for ranked data reuses the same terms.

## A duplicate that turns the loss off

Take $B = 2$, $d = 2$, $H = 0$ and $\tau = 1$ with $q_1 = p_1^+ = (1, 0)$ and $q_2 = p_2^+ = (0, 1)$. With distinct labels, row 1 has a positive logit of $1$ and three masked-in logits of $0$, for $q_1$ against $p_2^+$, $q_1$ against $q_2$ and $p_1^+$ against $p_2^+$. Its term is $\log(e + 3) - 1 = 1.7437 - 1 = 0.7437$, and row 2 is the same by symmetry, so the loss is $0.7437$.

Now give both queries the same label. The mask zeroes $j = 2$ for row 1 and $j = 1$ for row 2, so each denominator holds only its positive and each term is $\log e - 1 = 0$. The loss falls to $0$, because the batch no longer contains any trusted negative.

## Common mistakes to avoid

- **Skipping the same-tower term.** The result is the standard loss with hard negatives, and it no longer stops the two towers from drifting apart.
- **Adding only the query side of the same-tower term.** Both $q_i$ against $q_j$ and $p_i^+$ against $p_j^+$ enter the denominator.
- **Leaving out the mask.** The diagonal then leaks into the in-batch and same-tower sums, and duplicates are punished as negatives.
- **Building the mask from query labels alone.** A pair is also masked when only the positives are duplicates.
- **Using raw dot products.** The inputs are not normalized, so every similarity, including those with hard negatives, must be a cosine.
- **Multiplying by the temperature.** Logits are similarities divided by $\tau$, and multiplying flattens the softmax instead of sharpening it.
- **Summing over the batch.** The loss is a mean over the $B$ rows, and a sum makes its scale depend on batch size.