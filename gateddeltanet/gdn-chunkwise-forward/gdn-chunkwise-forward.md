# Chunkwise Training

Chunkwise training is how Gated DeltaNet turns a strictly sequential memory update into work that runs on matrix units. It cuts the sequence into chunks, carries one state across each chunk boundary, and computes every output inside a chunk with a handful of matrix products. This problem covers the parts around the UT transform, namely the decayed state carry, the pseudo-values and the decay-masked interaction inside a chunk.

## Three ways to run the same recurrence

Write the state of one head as $S_t \in \mathbb{R}^{d_v \times d_k}$, with keys and queries of width $d_k$ and values of width $d_v$. The gated delta rule updates it as $S_t = S_{t-1}\alpha_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ and reads $o_t = S_t q_t$, where $\alpha_t \in (0, 1]$ is the decay and $\beta_t$ is the write strength.

- **The recurrent form.** It follows that update one token at a time. It needs constant memory, but every step waits for the previous one, so the hardware runs a long chain of small matrix-vector products.
- **The fully parallel form.** It writes every output as a sum over all earlier tokens with an $L \times L$ score matrix. With the delta rule the transitions do not commute, and this form costs quadratic time and memory.
- **The chunkwise form.** It mixes the two. It is parallel inside a chunk and recurrent across chunks, so the cost is linear in the sequence length.

## From Mamba2 and DeltaNet to the gated chunk form

Equation 1 gives the chunkwise form of linear attention with decay, which is the Mamba2 case. Queries decay to the chunk's first position, keys decay to its last position, the carried state is multiplied by the total decay $\gamma_C$, and in-chunk scores are masked by $\Gamma$, with entry $\gamma_i / \gamma_j$ on and below the diagonal. Here $\gamma_r$ is the product of the decays from the chunk's first position up to $r$, restarted in every chunk.

Equations 8 and 9 give DeltaNet's chunkwise form with no decay. The delta rule erases what the state already stores along each key, so the chunk writes corrected values $U - W S^\top$ built from its WY vectors $W$ and $U$. The state update is $S + (U - WS^\top)^\top K$, and the in-chunk term uses those corrected values with the plain causal mask.

Gated DeltaNet's Section 3.3 form needs both ideas at once. The decay terms come from Mamba2 and the correction comes from DeltaNet, which gives the pseudo-values $\Delta = \tilde{U} - \overleftarrow{W} S_n^\top$. Each output is $\overleftarrow{Q} S_n^\top + (QK^\top \odot \Gamma)\Delta$, and the next state is $\gamma_{C_n} S_n + \Delta^\top \overrightarrow{K}$. The supplied helper builds $\overleftarrow{W}$ from the ungated triangular matrix scaled by $\gamma$, and $\tilde{U}$ from the matrix masked by $\Gamma$.

## Conventions this problem fixes

- The printed output equation in Section 3.3 uses the plain causal mask, but expanding Equation 10 shows that the intra-chunk term needs the decay-aware mask with the diagonal included, so this problem treats the printed mask as a typo.
- The state is stored value width by key width, as in the paper, whereas the official kernel stores its transpose.
- The official code multiplies queries by $d_k^{-1/2}$, but the paper never scales them, and this problem applies no scale.
- The final chunk may be shorter than the chunk size. It is processed as it is, so its total decay $\gamma_{C_n}$ uses only its own positions.

## Two chunks of one token each

Take one head with $d_k = 2$ and $d_v = 1$, a zero initial state, chunk size one, keys $k_1 = (1, 0)$ and $k_2 = (0.6, 0.8)$, values $v_1 = 2$ and $v_2 = 1$, write strengths $\beta_1 = 0.5$ and $\beta_2 = 1$, and decays $\alpha_1 = 1$ and $\alpha_2 = 0.5$.

In the first chunk the pseudo-value is $\tilde{U} = 0.5 \times 2 = 1$, because the incoming state is zero. The next state is $1 \times (1, 0) = (1, 0)$.

In the second chunk $\gamma_1 = 0.5$, the decayed key vector is $\overleftarrow{w} = 0.5 \times (0.6, 0.8) = (0.3, 0.4)$, and $\tilde{U} = 1$. The incoming state predicts $(0.3, 0.4) \cdot (1, 0) = 0.3$, so the pseudo-value is $1 - 0.3 = 0.7$. The next state is $0.5 \times (1, 0) + 0.7 \times (0.6, 0.8) = (0.92, 0.56)$.

The direct recurrence $S_2 = 0.5 \times (1, 0)(I - k_2 k_2^\top) + k_2$ gives the same $(0.92, 0.56)$.

## Common mistakes to avoid

- **Using the plain causal mask.** Without $\Gamma$ the earlier positions in a chunk are not decayed before they reach a later query, and the outputs drift from the recurrence whenever the decay is below one.
- **Forgetting $\gamma_{C_n}$ on the carried state.** The state must lose the whole chunk's decay before the new writes are added.
- **Leaving keys undecayed in the state update.** A key written early in a chunk must be scaled by $\gamma_{C_n} / \gamma_r$, since the later positions decay it further.
- **Accumulating the decay across the whole sequence.** The cumulative decay restarts in every chunk, and a global cumulative sum mixes chunks.
- **Building pseudo-values from the outgoing state.** The correction must use the state the chunk starts from, because that is what the queries and keys of the chunk see.