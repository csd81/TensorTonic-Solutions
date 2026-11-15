# Matryoshka Objective

The Matryoshka objective is the outer loop of EmbeddingGemma 2's training losses. It takes the contrastive, spread-out and distillation losses and applies each of them again to several prefixes of the same embedding, so that the first 512, 256 or 128 coordinates of a 768-dimensional vector work as embeddings on their own. This problem is about that outer loop, which means choosing the prefixes, renormalizing them and combining the per-width losses.

## Why one width is not enough

A model trained only at its full width spreads information across all coordinates in whatever way the losses happen to favor. Nothing encourages the first coordinates to carry the most useful signal. If a user keeps the first 128 coordinates of such an embedding to save storage, the result is close to an arbitrary projection, and retrieval quality can collapse.

Matryoshka representation learning (Kusupati et al., 2022) fixes this by training every nested prefix at once. The paper describes the idea as "splitting each loss into k separate losses applied to k overlapping sub-dimensions of the embedding." Because the 128-wide prefix is itself part of the 256-wide prefix, gradients from the small widths pull the most important information toward the front, while the full-width loss still uses the remaining coordinates for finer detail. One forward pass produces every width, and serving at a smaller width costs nothing beyond slicing.

## Renormalizing every prefix

The full embedding has unit length, but its prefixes do not. If the first $m$ coordinates of a query $q_i \in \mathbb{R}^d$ hold a fraction $r$ of its squared norm, the prefix has length $\sqrt{r}$. A deployed model that truncates to width $m$ renormalizes before computing similarities, so training should see the same vectors. For each width the prefix becomes

$$
\hat{q}_i = \frac{q_{i,1:m}}{\lVert q_{i,1:m} \rVert_2}
$$

and the same rule applies to every positive and every hard negative. This matters most for the losses that assume unit inputs. The spread-out regularizer reads inner products directly as cosine similarities, and the distillation loss measures squared Euclidean distance to a unit teacher vector, so an unnormalized prefix makes both of them wrong. The contrastive loss normalizes internally and would hide the mistake, which is why the weights decide which errors are visible.

The teacher side needs no extra work in this loop. The distillation helper truncates the teacher to the student's current width and renormalizes it, which matches how EmbeddingGemma 2 distills from Gemini Embedding 2 without a learned projection.

## Different stages, different weights

The three weights $w_C$, $w_S$ and $w_D$ let one function express every stage of the recipe. Text pre-finetuning uses almost only distillation, with a small contrastive weight, and multimodal pre-finetuning uses distillation alone. Multimodal finetuning replaces distillation with the contrastive loss and adds the spread-out regularizer. The paper explains the switch by noting that the teacher's embedding space "is not optimized for the student's architecture, size, and MRL sub-dimensions." Table 3 shows the effect. Continuing distillation for a compute-equivalent run moves MMEB-v2 from 52.27 to 52.28, while the contrastive finetuning stage reaches 55.46.

The paper does not say how the per-width losses are weighted against each other. The exercise chooses an unweighted sum over widths, so every prefix counts equally, and it passes the three loss weights in as inputs. Using one sum rather than a mean keeps each width's gradient the same size no matter how many widths are listed.

## Two widths by hand

Take a batch of two identical examples, so every batch mean equals the value for one example, each with student query $q = (0.6, 0.8, 0.0)$, student positive $p^+ = (0.8, 0.6, 0.0)$, no hard negatives, widths $3$ and $1$, and weights $w_C = 0$, $w_S = 0$, $w_D = 1$. Let the teacher query equal the student query, and let the teacher positive be $(-0.6, 0.8, 0.0)$. At width $3$ the query distance is $0$, and the positive distance is $1.4^2 + 0.2^2 = 2.0$, so $\ell_3 = 2.0$.

At width $1$ the query prefix $(0.6)$ and its teacher prefix both renormalize to $(1.0)$, which gives distance $0$. The student positive prefix $(0.8)$ becomes $(1.0)$, while the teacher prefix $(-0.6)$ becomes $(-1.0)$, so the distance is $2^2 = 4$ and $\ell_1 = 4$. The total is $2 + 4 = 6$. The short prefix points the opposite way from the teacher even though the full vectors were only moderately apart, and only the nested objective sees that.

## Common mistakes to avoid

- **Skipping the renormalization.** A raw prefix is shorter than one, so the spread-out and distillation terms measure the wrong geometry at every width except the full one.
- **Slicing from the end.** The nested prefixes are the first coordinates, and taking the last ones trains a different set of sub-embeddings than the one users truncate to.
- **Forgetting the hard negatives.** They also need truncation and renormalization, and the distillation term reveals the difference whenever its weight is positive.
- **Averaging over widths.** The exercise sums the per-width losses, and dividing by the number of widths shrinks every gradient.
- **Training only the full width.** Computing the losses once at $d$ leaves the shorter prefixes untrained, which removes the point of the objective.
- **Returning a Python list.** The per-width losses should come back as one tensor in the order of the widths.