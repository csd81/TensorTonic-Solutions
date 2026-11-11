# Causal Grouped-Query Attention

Grouped-query attention (GQA), introduced by Ainslie et al. (2023), lets several query heads share one key head and one value head. Llama 2 70B, Llama 3 and Mistral all use it to shrink the memory needed for the key-value cache during generation. The heart of this problem is mapping each query head to the right shared head, applying the causal mask, and doing it without copying the shared heads.

## Why share key and value heads

During text generation, a model stores the keys and values of every previous token so it does not recompute them. This key-value cache grows with the number of layers, the sequence length, the batch size and the number of key and value heads. With long contexts and large batches it can take more memory than the model weights.

Standard multi-head attention gives every query head its own key and value heads. Multi-query attention goes to the other extreme and shares a single key and value head across all query heads, which saves the most memory but hurts quality. GQA sits between them. With 32 query heads and 8 key-value heads, the cache shrinks by a factor of 4 while quality stays close to full multi-head attention.

## Which query heads share which key-value head

With $H_q$ query heads and $H_{kv}$ key-value heads, each group holds $G = H_q / H_{kv}$ query heads. Groups are contiguous blocks, so query head $h$ (counting from 0) uses key-value head

$$
\operatorname{kvhead}(h) = \left\lfloor \frac{h}{G} \right\rfloor
$$

With 8 query heads and 2 key-value heads, $G = 4$. Query heads 0 to 3 share key-value head 0, and query heads 4 to 7 share key-value head 1. Taking $h \bmod H_{kv}$ would interleave the groups and pair heads with the wrong keys.

When $H_{kv} = H_q$ the formula gives ordinary multi-head attention, and when $H_{kv} = 1$ it gives multi-query attention.

## The causal mask

A language model predicts each token from the tokens before it, so a query at position $i$ may only look at keys at positions $j \le i$. The scores for later keys are set to negative infinity before the softmax:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^{\mathsf T}}{\sqrt{D}} + M \right) V
$$

Here $Q$ is one query head, $K$ and $V$ are its shared key and value heads, $D$ is the head width, and $M_{ij}$ is 0 when $j \le i$ and $-\infty$ when $j > i$. Since $e^{-\infty} = 0$, masked keys get exactly zero weight. The softmax runs over key positions, and the first query always has at least one visible key, itself, so no row is entirely masked.

## Sharing without copying

A simple implementation repeats each key-value head $G$ times so the tensors line up with the queries. That allocates $G$ temporary copies of every key and value head on each forward pass, spending memory and bandwidth that the sharing was meant to save.

A better approach groups the query heads by the key-value head they use. Each key-value head can then be broadcast across its $G$ query heads in the score and output products without being copied. Because the groups are contiguous, this grouping matches the $\lfloor h / G \rfloor$ mapping exactly.

## Two heads sharing one key-value head

Take one key-value head with head width $D = 2$ and two positions. The keys are $k_0 = [1, 1]$ and $k_1 = [0, 2]$, and the values are $v_0 = [1, 0]$ and $v_1 = [0, 1]$. Two query heads in the same group both read these keys and values.

At position 1, the first query head has query $[1, 2]$. Both keys are visible, and the scaled scores are

$$
\frac{[1, 2] \cdot [1, 1]}{\sqrt{2}} = 2.121320, \qquad \frac{[1, 2] \cdot [0, 2]}{\sqrt{2}} = 2.828427
$$

The softmax gives weights $0.330238$ and $0.669762$, so the output is $[0.330238, 0.669762]$.

The second query head has query $[4, 0]$ at the same position. Its scores are $2.828427$ and $0$, its weights are $0.944193$ and $0.055807$, and its output is $[0.944193, 0.055807]$. The two heads read the same keys and values and still produce different outputs.

At position 0, either head could only see $k_0$, so its single weight would be 1 and its output would be exactly $v_0$, whatever its query.

## Why the softmax runs in float32

Bfloat16 keeps only about two to three significant digits, so softmax probabilities and their weighted sum lose accuracy at that precision. These steps run in float32 before the result is converted back.

## Common mistakes to avoid

- **Interleaving the groups.** Query head $h$ uses key-value head $\lfloor h / G \rfloor$, and taking $h \bmod H_{kv}$ pairs it with the wrong head.
- **Repeating key and value heads.** Copying them costs the memory and bandwidth that sharing was meant to save.
- **Masking the diagonal.** Each query may attend to its own position, so only strictly later keys are masked.
- **Masking after the softmax.** Setting probabilities to zero after normalizing leaves rows that no longer sum to 1.
- **Scaling by the wrong width.** The divisor is the square root of the head width $D$.
- **Running the softmax over the wrong axis.** Probabilities are normalized over key positions.
