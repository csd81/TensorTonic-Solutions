# Extended WY Chunk Transition

The extended WY representation is the algebraic step that makes Gated DeltaNet trainable in chunks. It rewrites the product of many gated Householder transitions inside one chunk as an identity minus a sum of rank-one terms, and it writes the chunk's accumulated memory as a sum of decayed outer products. This problem builds those two lists of vectors and the affine map $S_{\text{out}} = S_{\text{in}} F + G$ that they define.

## Why one token at a time is too slow

The gated delta rule of Equation 10 updates the state $S_t \in \mathbb{R}^{d_v \times d_k}$ as

$$
S_t = S_{t-1}\,\alpha_t \left(I - \beta_t k_t k_t^\top\right) + \beta_t v_t k_t^\top
$$

where $k_t$ is the key, $v_t$ the value, $\alpha_t = e^{g_t}$ the decay and $\beta_t$ the write strength. This problem uses the paper's orientation, with the state shaped value width by key width and the transition multiplied on the right, while the official code stores the transpose.

Running this recurrence token by token is a chain of matrix products that cannot use parallel hardware well. A chunkwise algorithm instead needs to know, for a chunk of $C$ tokens, how the incoming state maps to the outgoing one. Expanding the recurrence over the chunk gives exactly that map, with a transition $F$ equal to the ordered product $\prod_i \alpha_i (I - \beta_i k_i k_i^\top)$ and an offset $G$ that collects every write.

## Explicit products against the WY form

Multiplying the $C$ Householder matrices directly costs $C$ products of $d_k \times d_k$ matrices, which is wasteful because each factor differs from the identity by only a rank-one term. Bischof and Van Loan's WY representation, used by DeltaNet in Equation 4, observes that such a product stays the identity minus a rank-$C$ correction,

$$
P = I - \sum_{i=1}^{C} w_i k_i^\top
$$

with vectors $w_i$ built by a short recursion over earlier positions. The correction lives in the span of the chunk's keys, so storing $C$ vectors of length $d_k$ replaces storing a dense matrix. When $C$ is smaller than $d_k$, the saving is large, and the products $k_i^\top k_r$ that drive the recursion form a small $C \times C$ table that maps well onto matrix units.

The same trick applies to the write term. DeltaNet represents its offset as $H = \sum_i u_i k_i^\top$ with value vectors $u_i$ from Equation 5. Each $u_r$ is the value minus what earlier writes would already predict along the current key, which is the delta rule's error correction expressed in closed form.

## What the gates change

Gated DeltaNet multiplies every transition by a scalar decay. Writing $\gamma_r = \exp(\sum_{i \le r} g_i)$ for the chunk-local cumulative decay, Section 3.3 shows that the gated product is simply $F = \gamma_C P$. The scalars commute with every matrix, so the key side needs no new recursion, and this problem keeps the ungated Equation 4 vectors for it.

The value side does change. A write made at position $i$ has been decayed by $\gamma_r / \gamma_i$ by the time position $r$ corrects against it, so the gated recursion multiplies each correction term by that ratio. The offset then decays each rank-one term to the end of the chunk with $\gamma_C / \gamma_i$. Appendix A proves this by induction, and its second line contains a slip that writes $k_i^\top k_i k_{t+1}^\top$ where the next line needs $k_i^\top k_{t+1}$, so the corrected expression is the one to trust.

The result is a pair of matrices that summarize the chunk. Every chunkwise step of training reuses them, combined with decayed queries and keys, so getting the decay ratios right here decides whether the whole forward pass matches the recurrence.

## A two-token chunk by hand

Take $d_k = 2$, $d_v = 1$, keys $k_1 = (1, 0)$ and $k_2 = (0.6, 0.8)$, write strengths $\beta = (1, 0.5)$, values $v = (1, 2)$, and decays $\alpha_1 = 1$, $\alpha_2 = 0.5$, so $\gamma = (1, 0.5)$.

- **Key vectors.** The first is $w_1 = (1, 0)$. With $k_1^\top k_2 = 0.6$, the second is $w_2 = 0.5\left((0.6, 0.8) - 0.6\,(1, 0)\right) = (0, 0.4)$.
- **Transition.** The correction $w_1 k_1^\top + w_2 k_2^\top$ has rows $(1, 0)$ and $(0.24, 0.32)$, so $P$ has rows $(0, 0)$ and $(-0.24, 0.68)$, and $F = 0.5P$.
- **Value vectors.** The first is $\tilde{u}_1 = 1$, and the second is $\tilde{u}_2 = 0.5\,(2 - 1 \times 0.5 \times 0.6) = 0.85$.
- **Offset.** $G = 0.5 \times 1 \times (1, 0) + 0.85\,(0.6, 0.8) = (1.01, 0.68)$, which matches two steps of the recurrence from a zero state.

## Common mistakes to avoid

- **Putting the decay ratio into the key recursion.** The transition only picks up the global factor $\gamma_C$, so the key vectors must use plain inner products.
- **Dropping the ratio from the value recursion.** Without $\gamma_r / \gamma_i$ the corrections subtract writes that the decay has already shrunk.
- **Inverting the ratio.** The factor is $\gamma_r / \gamma_i \le 1$ for $i < r$, and its reciprocal grows instead of decaying.
- **Forgetting $\gamma_C$ in the transition.** The incoming state must be decayed across the whole chunk.
- **Leaving out $\gamma_C / \gamma_i$ in the offset.** Early writes would then reach the chunk end at full strength.
- **Including the current position in the correction sums.** Each recursion subtracts only earlier positions $i < r$.