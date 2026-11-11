# Blockwise Attention with Online Softmax

Attention compares every query with every key, so the plain way of computing it builds a score matrix whose size grows with the product of the two sequence lengths. Flash-style attention never builds that matrix. It walks over the keys one block at a time and keeps three small running quantities per query that are enough to produce the exact softmax result at the end. This page derives that online softmax from the stable softmax, works through a tiny example, and shows how lax.scan expresses the loop in JAX.

## Attention needs a softmax over every key

For one attention head, the queries $q$ have shape $(T_q, d)$, and the keys $k$ and values $v$ have shape $(T_k, d)$, where $T_q$ and $T_k$ are the numbers of query and key positions and $d$ is the head width. Each query $i$ scores every key $j$, turns its scores into weights with a softmax, and averages the values with those weights.

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad O_i = \frac{\sum_j e^{s_{ij}} \, v_j}{\sum_j e^{s_{ij}}}
$$

Here $s_{ij}$ is a score and $O_i$ is the output row for query $i$. Dividing by $\sqrt{d}$ keeps the scores from growing with the head width. The scores for all queries form a $(T_q, T_k)$ matrix. With $T_q = T_k = 4096$ in float32 that matrix takes 64 MiB for a single head of a single example, and it has to be written out and read back. Moving those bytes, more than the arithmetic, is what makes plain attention slow on long sequences.

## The max trick keeps the exponentials finite

The exponential grows very fast, and float32 overflows to infinity once its input passes about 88. Scores of 1000 and 999 would give two infinities and an undefined ratio. The fix uses the fact that the softmax does not change when the same constant is subtracted from every score, because the factor $e^{-c}$ cancels between the numerator and the denominator.

$$
\frac{e^{s_j}}{\sum_t e^{s_t}} = \frac{e^{s_j - m}}{\sum_t e^{s_t - m}}, \qquad m = \max_t s_t
$$

Subtracting the maximum $m$ makes every exponent zero or negative, so every exponential lies between 0 and 1. For the scores 1000 and 999 the shifted exponents are 0 and -1, and the weights come out as about 0.7311 and 0.2689. The catch is that this needs the maximum over every key before any exponential is taken, which seems to require seeing the whole row at once.

## Merging partial results when the maximum changes

The online softmax removes that requirement. Process the keys in blocks and keep three running quantities for each query.

- **The running maximum $m$.** The largest score seen so far.
- **The running sum $l$.** The sum of $e^{s - m}$ over the keys seen so far.
- **The running output $o$.** The sum of $e^{s - m} v$ over the keys seen so far, a vector of width $d$.

All three use the current $m$ as their reference point. When a new block arrives with scores $s_j$, the new maximum is $m' = \max(m, \max_j s_j)$. The old sum and output were computed relative to $m$, and multiplying them by $e^{m - m'}$ moves them to the new reference, since $e^{s - m} \cdot e^{m - m'} = e^{s - m'}$. Then the new block's contributions are added.

$$
c = e^{m - m'}, \qquad l' = c \, l + \sum_j e^{s_j - m'}, \qquad o' = c \, o + \sum_j e^{s_j - m'} \, v_j
$$

Here $c$ is the correction factor, between 0 and 1. If the maximum did not change, $c = 1$ and the old totals are kept as they are. After the last block, the output row is $o / l$, which equals the full softmax average exactly, because both $o$ and $l$ now use the true maximum.

## A four-key example

Take one query whose scores against four keys are 1.0, 0.5, 3.0 and 2.0, with scalar values 10, 20, 30 and 40, split into two blocks of two keys.

The direct softmax uses the maximum 3.0. The shifted exponentials are about 0.1353, 0.0821, 1 and 0.3679, which sum to about 1.5853. The weighted sum of values is about 47.7102, so the output is about 30.0954.

The online version starts from $m = -\infty$, $l = 0$ and $o = 0$.

1. The first block has scores 1.0 and 0.5, so the maximum becomes 1.0. The correction is $e^{-\infty} = 0$, which simply clears the empty starting totals. The exponentials are 1 and about 0.6065, so $l \approx 1.6065$ and $o \approx 10 + 0.6065 \cdot 20 \approx 22.1306$. At this point $o / l \approx 13.7754$, the softmax average of the first two keys alone.
2. The second block has scores 3.0 and 2.0, so the maximum rises to 3.0 and the correction is $e^{1 - 3} \approx 0.1353$. The old totals shrink to about 0.2174 and 2.9951. The new exponentials are 1 and about 0.3679, giving $l \approx 1.5853$ and $o \approx 47.7102$.

The final $o / l$ is about 30.0954, the same as the direct softmax. Skipping the correction in step 2 gives about 22.4736 instead, because the first block would then count as if its scores were two points higher than they are.

## Writing the loop with lax.scan

A Python for loop over blocks would work in eager mode, but under jax.jit it is unrolled into one copy of the block computation per iteration, so the compiled program grows with the number of blocks. The primitive lax.scan expresses the same loop as a single compiled body that runs repeatedly.

The function lax.scan takes a step function, an initial carry and a sequence of inputs. It slices the inputs along their leading axis and calls the step once per slice, passing the current carry and the slice, and the step returns the new carry plus an optional per-step output. For blockwise attention the carry is the running maximum of shape $(T_q)$, the running sum of shape $(T_q)$ and the running output of shape $(T_q, d)$, and the per-slice input is one block of keys and the matching block of values. Reshaping a $(T_k, d)$ array into $(T_k / b, b, d)$, where $b$ is the block size, gives exactly that leading axis, and the scan then runs $T_k / b$ steps.

Inside each step the scores form only a $(T_q, b)$ matrix, so the largest temporary is $b / T_k$ of the full score matrix. With $T_q = T_k = 4096$ and $b = 128$, that is 2 MiB instead of 64 MiB. The values the step needs from outside, such as the queries, can be captured from the enclosing function and are shared by every step.

## Causal masking inside the blocks

A decoder lets query position $i$ attend only to key positions $j \le i$. The full formula adds minus infinity to every forbidden score, so its exponential is exactly 0. In blockwise form, each step needs the global positions of its keys, which are the block's starting offset plus 0 to $b - 1$. The step can carry a block counter or receive the offsets as an extra scanned input.

Masked scores need care in the running maximum. If every score a query has seen so far were minus infinity, both $m$ and $m'$ would be minus infinity and $m - m'$ would be NaN. With causal attention over equal-length sequences this never happens, because key 0 sits in the first block and every query may attend to it, so every running maximum is finite after the first step. Later blocks that are fully masked for a query leave its maximum unchanged and add exponentials of 0.

## Where this sits in a real kernel

Flash attention on a GPU runs the same recurrence, with each block of keys and values loaded once into fast on-chip memory and reused by a block of queries. The arithmetic is the same as plain attention, but the memory traffic falls from the size of the score matrix to roughly the size of the inputs and outputs. This problem keeps the whole query set in one piece and blocks only the keys, which is enough to see the recurrence and the scan structure.

## Common mistakes to avoid

- **Forgetting the correction factor.** Adding a new block without rescaling the old sum and output mixes exponentials taken against different maxima, which gives a wrong average whenever the maximum rises.
- **Rescaling the sum but not the output.** Both running quantities share one reference point, so both must be multiplied by the same factor.
- **Exponentiating raw scores.** Large scores overflow to infinity in float32, and the result turns into NaN.
- **Starting the running maximum at zero.** When every score of a query is far below zero, all its exponentials underflow to 0 and the final division becomes 0 by 0, while a start of minus infinity lets the first block set the true reference.
- **Building the full score matrix.** Computing all $T_q \times T_k$ scores and then scanning defeats the memory saving the blocks exist for.
- **Using local positions in the causal mask.** Each block's key positions must include the block's starting offset, otherwise every block is masked as if it were the first.
- **Dropping the $\sqrt{d}$ scale.** The scores are divided by the square root of the head width before the softmax.