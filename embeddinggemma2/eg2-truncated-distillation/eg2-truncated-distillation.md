# Truncated Distillation

Truncated distillation is the loss EmbeddingGemma 2 uses to inherit the embedding space of Gemini Embedding 2, a far larger multimodal embedding model. The student is trained to place each input exactly where the teacher places it, after the teacher vector has been cut down to the student width and rescaled to unit length. This problem covers that matching loss for queries, positive targets and hard negative targets.

## Matching embeddings instead of scores

Most retrieval distillation transfers relevance scores. The teacher rates how well each document fits a query, and the student is trained to reproduce those ratings, often through a KL divergence over a candidate list. That signal is indirect, because many different embedding geometries produce the same ranking of a handful of candidates.

Embedding matching, introduced by Kim et al., 2023, asks for more. The student embedding $u_i$ should land on the teacher embedding of the same input, measured by the squared distance $\lVert u_i - \tilde{u}_i \rVert_2^2$. Matching positions fixes every pairwise score at once, including scores against inputs that never appeared together in a batch. EmbeddingGemma 2 applies the match to queries $q$, positives $p^+$ and hard negatives $p^-$, as EmbeddingGemma 1 did. Matching the hard negatives transfers the teacher's ability to separate a query from the near misses that look relevant, which is exactly the skill that is hardest to learn from labels alone.

## Learned projection versus truncation

The two models disagree on width. The student emits $d = 768$ dimensions while the teacher emits more, so the vectors cannot be subtracted directly. Kim et al. solve this with a learned projection that maps the student space into the teacher width, trained jointly with the student. The projection adds parameters, its own initialization and a component that is thrown away after training, and the student can partly offload the matching onto it instead of learning the geometry itself.

EmbeddingGemma 2 removes the projection by changing the teacher side. Gemini Embedding 2 is trained to support lower output dimensions, so a prefix of its vector is already a meaningful embedding. The paper simply truncates the teacher to the student width. Truncation alone leaves a vector shorter than one, because the dropped tail carried part of the norm, so the prefix is divided by its own norm:

$$
\tilde{t} = \frac{t_{1:d}}{\lVert t_{1:d} \rVert_2}
$$

The paper says the teacher is truncated but does not say which coordinates survive. This exercise chooses the first $d$, which is the same prefix convention Matryoshka embeddings use when a vector is shortened at serving time. The order of operations matters. Normalizing the full teacher vector and then cutting it produces a target whose length depends on how much energy sat in the tail, so it is generally not a unit vector and can never be matched by a unit student row.

## Where the loss sits in training

Distillation carries the early stages. Text pre-finetuning uses it almost exclusively, with a small contrastive weight, and multimodal pre-finetuning uses nothing else, which lets the student absorb how the teacher arranges images, audio, video and text in one space. The finetuning stage then drops distillation entirely, because the teacher space was never optimized for the student's size or its truncated sub-dimensions. The three terms are weighted uniformly, and with several hard negatives per example the negative term averages over them, so an example with more negatives does not count more than one with fewer.

## A query that is close, a negative that is not

Take $B = 1$, $H = 2$, $d = 2$ and $D = 4$. The teacher query is $(1, 1, 10, -10)$, whose prefix $(1, 1)$ normalizes to $(0.7071, 0.7071)$. The student query $(0.6, 0.8)$ differs by $(-0.1071, 0.0929)$, so its term is $0.0115 + 0.0086 = 0.0201$. The teacher positive $(0, -2, 1, 1)$ becomes $(0, -1)$, which the student matches exactly.

The teacher negatives are $(4, 3, 0, 0)$ and $(1, 0, 5, 5)$, which become $(0.8, 0.6)$ and $(1, 0)$. The student negatives $(0.6, 0.8)$ and $(1, 0)$ contribute $0.04 + 0.04 = 0.08$ and $0$, and their mean is $0.04$. The loss is $0.0201 + 0 + 0.04 = 0.0601$. Normalizing before truncating would instead turn the teacher query into roughly $(0.0704, 0.0704)$, and the query term alone would jump to about $0.81$.

## Common mistakes to avoid

- **Normalizing before truncating.** The target then has a norm well below one whenever the tail is large, and the loss chases a point the student cannot reach.
- **Skipping the renormalization.** A bare prefix has the wrong length, so even a perfectly aligned student keeps a positive loss.
- **Keeping the last coordinates.** The prefix carries the coarse structure of the teacher space, while the tail holds fine detail that does not stand on its own.
- **Summing over hard negatives.** The paper averages over the negatives of each example, so a sum scales the term with $H$.
- **Taking an element-wise mean.** The loss is a squared norm per row, and dividing by $d$ shrinks every term by the width.
- **Dropping the hard negative term.** It is the part that teaches the student to tell close negatives apart from true positives.
- **Averaging over an empty negatives axis.** With no hard negatives that mean is undefined, while the term should simply be zero.