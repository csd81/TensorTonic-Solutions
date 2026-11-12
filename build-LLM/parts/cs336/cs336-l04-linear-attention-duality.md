# Parallel and Recurrent Linear Attention

Softmax attention compares every query with every key, so its cost grows with the square of the sequence length. Linear attention removes the softmax, and that one change lets the same computation run either in parallel like a transformer or step by step like a recurrent network. The heart of this problem is computing both forms and seeing that they produce the same outputs.

## The cost of softmax attention

In causal softmax attention, the output at position $t$ is a weighted average of all values up to $t$, with weights from a softmax over the scores $q_t \cdot k_s$. The softmax couples all the scores in a row, so each new token has to look back at every earlier key. Each generated token therefore costs more than the last, and the key-value cache grows with the sequence.

## Removing the softmax

Without the softmax, the output at position $t$ is the sum of the values weighted by the raw scores:

$$
y_t = \sum_{s \le t} (q_t \cdot k_s) \, v_s
$$

Here $q_t$ is the query at position $t$, and $k_s$ and $v_s$ are the key and value at an earlier or equal position $s$. Queries and keys have width $D_k$ and values have width $D_v$. The sum includes $s = t$, so each output uses its own token. Because there is no normalization, the scores can be negative and the weights do not sum to 1.

## The recurrent form

Because $q_t \cdot k_s = q_t^{\mathsf T} k_s$ and matrix products can be regrouped, each term $(q_t \cdot k_s) v_s$ can be rewritten as $q_t^{\mathsf T} (k_s v_s^{\mathsf T})$, where $k_s v_s^{\mathsf T}$ is a $D_k \times D_v$ outer product. The query then factors out of the sum:

$$
y_t = q_t^{\mathsf T} \sum_{s \le t} k_s v_s^{\mathsf T}
$$

The sum is a running state that can be updated one token at a time. Starting from $S_0 = 0$,

$$
S_t = S_{t-1} + k_t v_t^{\mathsf T}, \qquad y_t = q_t^{\mathsf T} S_t
$$

Each step writes the new key-value outer product into the state and then reads the state with the current query. The state has a fixed size of $D_k \times D_v$ however long the sequence gets, so generation costs the same at every step and needs no growing cache. Because the state accumulates a product at every step, the running sums are kept in float32 to limit rounding error.

## The parallel form

During training the whole sequence is available at once, and a loop over positions would leave the GPU mostly idle. Stacking the queries, keys and values as matrices $Q$, $K$ and $V$, with one row per position, gives all outputs at once as

$$
Y = \left( Q K^{\mathsf T} \odot L \right) V
$$

where $L$ is a lower-triangular matrix of ones that keeps each position and the ones before it, and $\odot$ is elementwise multiplication. This is ordinary attention with the softmax replaced by a causal mask, and it needs no sequential loop.

This equivalence is called the duality of linear attention. Models can therefore train with the parallel form and generate with the recurrent form, and both give the same outputs up to floating-point rounding.

## Both forms on two tokens

Take a sequence of two tokens with $D_k = 2$ and $D_v = 1$:

$$
k_1 = [1, 0], \ v_1 = [2], \ q_1 = [1, 1], \qquad k_2 = [1, 1], \ v_2 = [1], \ q_2 = [0, 1]
$$

In the recurrent form, the first write and the second write give the states

$$
S_1 = k_1 v_1^{\mathsf T} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}, \qquad S_2 = S_1 + k_2 v_2^{\mathsf T} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}
$$

Reading them gives $y_1 = q_1^{\mathsf T} S_1 = 2$ and $y_2 = q_2^{\mathsf T} S_2 = 1$.

In the parallel form, $y_1 = (q_1 \cdot k_1) v_1 = 1 \times 2 = 2$ and $y_2 = (q_2 \cdot k_1) v_1 + (q_2 \cdot k_2) v_2 = 0 \times 2 + 1 \times 1 = 1$. The outputs match, and the final state is $S_2$.

## Where this leads

Plain linear attention never forgets, since every write stays in the state forever with equal weight. Without the softmax's normalization the model is also less selective about which tokens to focus on. Later architectures such as Mamba 2 and Gated DeltaNet add decay and correction terms to the state update, and they keep a parallel training form alongside the recurrent one.

## Common mistakes to avoid

- **Excluding the current token.** The state is updated before it is read, so each output includes its own key and value.
- **Writing the outer product in the wrong order.** The state has shape $D_k \times D_v$, built from $k_t v_t^{\mathsf T}$.
- **Masking the diagonal in the parallel form.** Only entries for later positions are removed.
- **Adding a softmax or normalization.** This problem uses raw scores with no normalization.
- **Sharing state across the batch.** Each batch element has its own state.
- **Accumulating along the wrong axis.** The running sum is over sequence positions, so summing over the key width or the batch mixes unrelated quantities.
