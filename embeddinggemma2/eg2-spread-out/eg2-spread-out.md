# Spread-Out Regularizer

The spread-out regularizer is one of the three losses that shape the EmbeddingGemma 2 embedding space during multimodal finetuning. It looks only at pairs of inputs that have nothing to do with each other and pushes their embeddings toward right angles. This problem covers the exact form the paper uses, which keeps the second moment of pairwise similarity and drops everything else from the original regularizer.

## Why embedding spaces collapse into cones

A contrastive loss only asks that a query sits closer to its own target than to the batch negatives. It says nothing about where everything else ends up, and trained encoders often place all of their outputs inside a narrow cone on the unit sphere. Two random inputs then have a cosine similarity of 0.6 or 0.7 even when they are unrelated, so the useful signal lives in a thin band of similarity values.

That crowding hurts in three ways. Thresholds on retrieval scores become fragile because unrelated pairs already score high. Approximate nearest neighbor indexes partition the space poorly when every point sits in the same region. Quantization rounds coordinates to a few levels, and when all vectors point roughly the same way, that rounding error is large relative to the small angular gaps that separate them. The paper cites quantization robustness and efficient approximate search as reasons for the regularizer.

## From the global orthogonal regularizer to one term

Zhang et al., 2017 proposed the global orthogonal regularizer. It compares the statistics of random pairs with those of two points drawn uniformly from the unit sphere in $\mathbb{R}^d$, which have mean inner product $0$ and mean squared inner product $1/d$. Writing $M_1$ for the mean of $q_i^\top q_j$ over unrelated pairs and $M_2$ for the mean of $(q_i^\top q_j)^2$, the original loss is

$$
\mathcal{L}_{\text{GOR}} = M_1^2 + \max\left(0, M_2 - \frac{1}{d}\right)
$$

The first term pulls the average similarity to zero. The second penalizes similarities that are spread wider than on the uniform sphere, and the hinge stops it once that level is reached.

EmbeddingGemma 2 follows EmbeddingGemma 1 and keeps only $M_2$, with no hinge. The paper reports that the second moment alone also drives the mean term toward its target, which makes sense because a small mean square forces most individual similarities, and therefore their mean, close to zero. Dropping the hinge means the loss keeps pushing toward exact orthogonality instead of stopping at $1/d$. The quantity for one tower is the mean of the squared off-diagonal entries of its Gram matrix $q q^\top$, where $q$ has shape $(B, d)$ and the Gram matrix has shape $(B, B)$.

## Two towers, never mixed

The loss is computed separately for queries and for positive targets and then summed. Cross-tower products $q_i^\top p^+_j$ are left out on purpose. The contrastive loss already handles those, and it needs $q_i^\top p^+_i$ to be large, so a spread-out term over cross pairs would fight the main objective on the diagonal. Inside one tower, row $i$ and row $j$ with $i \neq j$ come from different training examples, which makes them the closest available stand-in for a random unrelated pair.

The diagonal is excluded for a similar reason. Every row has unit norm, so $q_i^\top q_i = 1$ always, and including it would add a constant of $1/(B-1)$ per tower that carries no gradient and distorts the scale. In the paper's recipe, this regularizer joins the contrastive and Matryoshka losses in the finetuning stage, after distillation has placed the model near its teacher.

## Three vectors in the plane

Take $B = 3$ and $d = 2$ with queries $q_1 = (1, 0)$, $q_2 = (0, 1)$ and $q_3 = (0.7071, 0.7071)$. The off-diagonal inner products are $q_1^\top q_2 = 0$, $q_1^\top q_3 = 0.7071$ and $q_2^\top q_3 = 0.7071$. Their squares are $0$, $0.5$ and $0.5$. Each unordered pair appears twice among the $B(B-1) = 6$ ordered pairs, so the query term is $2 \times 1.0 / 6 = 0.3333$.

Now let the positives be $(1, 0)$, $(0, 1)$ and $(-1, 0)$. The products are $0$, $-1$ and $0$, so the only nonzero square is $1$, and the positive term is $2 \times 1 / 6 = 0.3333$. The loss is $0.6667$. The antipodal pair costs as much as an identical pair, because squaring removes the sign. Three directions cannot all be orthogonal in two dimensions, so this loss can never reach zero here, and the minimum spreads the error as evenly as possible.

## Common mistakes to avoid

- **Including the diagonal.** Unit rows give $q_i^\top q_i = 1$, so every self-pair adds a constant and the loss can no longer reach zero for orthogonal rows.
- **Dividing by $B^2$.** There are $B(B-1)$ ordered off-diagonal pairs, and the wrong count shrinks the loss most for small batches.
- **Using absolute values.** The paper squares each similarity, which weights large similarities far more than small ones and keeps the loss smooth at zero.
- **Mixing the towers.** Products between a query and a positive belong to the contrastive loss, and penalizing them here works against matching pairs.
- **Averaging the two towers.** The paper adds the query term and the positive term, so halving their sum changes the effective weight of the regularizer.
- **Counting only the query tower.** Targets can collapse just as easily as queries, and the second term keeps them spread as well.