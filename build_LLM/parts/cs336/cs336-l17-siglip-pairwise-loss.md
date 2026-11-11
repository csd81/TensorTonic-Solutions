# Implement the SigLIP Pairwise Loss

SigLIP (Zhai et al., 2023) trains image and text encoders on paired data like CLIP, but replaces CLIP's batch-wide softmax with an independent yes-or-no decision for every image-text pair. That small change removes the need to normalize over the whole batch and makes training cheaper in memory and communication, especially at large batch sizes. The heart of this problem is building the matrix of pair logits, labeling the matched pairs, and averaging a logistic loss over every pair.

## From picking one caption to judging every pair

CLIP treats each image as a classification problem over all captions in the batch. Its softmax couples every entry in a row, and in the symmetric loss every column too, so each device must gather all embeddings in the batch before it can normalize. Each row's loss also depends on how many captions compete in the softmax, so its value shifts as the batch grows.

SigLIP instead decides, for each image and each caption separately, whether the two belong together. For a batch of $B$ image-caption pairs, each of the $B^2$ image-caption combinations becomes its own binary classification. The $B$ matched pairs on the diagonal are positives and the $B^2 - B$ others are negatives. No term depends on any other, so the loss can be computed in blocks across devices without gathering the whole matrix.

## Pair logits with a learned scale and bias

Each embedding is first divided by its length, so only its direction matters, and a zero vector is protected from a division by zero. For normalized image vector $\widehat{v}_i$ and text vector $\widehat{t}_j$, the pair logit is

$$
\ell_{ij} = a \, \widehat{v}_i^{\mathsf T} \widehat{t}_j + b
$$

The cosine similarity lies between $-1$ and $1$, so the multiplier $a$ stretches it into a useful range for a sigmoid. The bias $b$ shifts every logit. It matters because negatives vastly outnumber positives, and starting with a strongly negative bias, $-10$ in the SigLIP paper, makes the model predict "no match" by default so the many easy negatives do not swamp early training. The paper learns both, storing the multiplier as the exponential of a learned number so it stays positive and starting it at 10. This problem passes in the multiplier itself.

## The pairwise logistic loss

Each pair gets a label $z_{ij}$, which is $1$ on the diagonal and $-1$ elsewhere. We call $z_{ij} \ell_{ij}$ the signed logit. A correct prediction makes it large and positive in both cases, for positives with a high logit and negatives with a low one. The loss is the mean negative log-sigmoid of the signed logits over all pairs:

$$
L = -\frac{1}{B^2} \sum_{i=1}^{B} \sum_{j=1}^{B} \log \sigma(z_{ij} \ell_{ij})
$$

where $\sigma(z) = 1 / (1 + e^{-z})$. For a large negative signed logit the sigmoid rounds to 0 and its log becomes infinite, so the loss must be computed in log space rather than by taking the log of a sigmoid that has already rounded to zero.

## One confusing negative dominates

Take two pairs with multiplier 10 and bias $-5$. The image embeddings are $[1, 0]$ and $[0.6, 0.8]$, and the text embeddings are $[0.8, 0.6]$ and $[0, 1]$, all already of length 1. The cosine similarities are 0.8 and 0 in the first row and 0.96 and 0.8 in the second, so the logits are

$$
\ell = \begin{bmatrix} 3 & -5 \\ 4.6 & 3 \end{bmatrix}
$$

The two positives give signed logits of 3, each costing $-\log \sigma(3) \approx 0.048587$. The negative pair of image 1 with text 2 has signed logit 5 and costs only about 0.006715. The other negative, image 2 with text 1, has signed logit $-4.6$, since the model wrongly thinks they match, and costs about 4.610002.

The mean over four pairs is about 1.178473, almost all of it from that one confusing negative. Every pair is scored on its own, so this term's gradient depends only on image 2 and text 1, with no normalization over the other pairs in the batch.

## Why the independent loss scales well

Because no softmax couples the entries, a large batch can be split across devices. Each device keeps its own images and receives the other devices' text embeddings one block at a time, computing the loss for each block as it arrives. SigLIP was trained this way with batches of tens of thousands of pairs, and it outperformed CLIP's softmax loss at smaller batch sizes.

## Common mistakes to avoid

- **Using a softmax over rows or columns.** A softmax couples the pairs, so the loss would no longer be SigLIP's and could not be computed block by block.
- **Labeling negatives with 0.** The signed form needs $-1$ for negatives so that $z_{ij} \ell_{ij}$ is positive when the prediction is right.
- **Averaging over only the diagonal.** The mean runs over all $B^2$ pairs, and the negatives carry most of the signal.
- **Skipping normalization.** Without dividing by the lengths, the logits depend on the vectors' norms as well as their directions.
- **Initializing the bias near zero in a real training run.** With $B^2 - B$ negatives, a zero bias makes the early loss dominated by easy negatives.
