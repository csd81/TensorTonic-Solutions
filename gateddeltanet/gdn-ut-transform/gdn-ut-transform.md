# UT Transform

The UT transform is the step of Gated DeltaNet's chunkwise training that computes, for one chunk, the vectors summarizing how its delta-rule writes interfere. The paper first defines these WY vectors by recursions over positions. This problem replaces them with two unit lower triangular solves and shows why only the value side carries the decay.

## Why a chunk needs WY vectors at all

Gated DeltaNet keeps a state $S \in \mathbb{R}^{d_v \times d_k}$, and each token applies $S \leftarrow S\,\alpha_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$. Here $k_t$ is a unit key, $v_t$ a value, $\alpha_t = e^{g_t}$ the decay and $\beta_t$ the write strength. The WY representation compresses a chunk of $C$ such transitions into $I - \sum_r w_r k_r^\top$ with one vector $w_r$ per position, plus matching value vectors $\tilde{u}_r$, built in the Extended WY Representation problem by the recursions of Equation 4 and Section 3.3.

Vector $w_r$ needs every earlier $w_i$, so a chunk takes $C$ dependent steps of vector arithmetic, which suits tensor cores poorly.

## From recursion to triangular solve

Stack the vectors as rows. The key recursion $w_r = \beta_r (k_r - \sum_{i<r} w_i\, k_i^\top k_r)$ says that row $r$ of $W$ plus a weighted sum of the earlier rows equals $\beta_r k_r$. Writing $A = \operatorname{strictLower}(\operatorname{diag}(\beta) K K^\top)$, this is the linear system $(I_C + A) W = \operatorname{diag}(\beta) K$. Since $I_C + A$ is unit lower triangular, this gives $W = T K$ with $T = (I_C + A)^{-1} \operatorname{diag}(\beta)$ from Equation 6, which Yang et al., 2025 adopt from the UT transform of Joffrain et al., 2006.

The sequential recursion needs $C$ dependent vector updates. The triangular solve needs one Gram matrix $K K^\top$, one small solve and one matrix product with $K$ or $V$.

Both forms give identical vectors in exact arithmetic, but the second moves almost all work into matrix products, and its only sequential part acts on a small $C \times C$ matrix.

## Why only the value side uses the decay mask

Inside a chunk, let $\gamma_r = \exp(\sum_{i \le r} g_i)$ be the cumulative decay and $\Gamma_{ij} = \gamma_i / \gamma_j$ for $i \ge j$. The paper shows that the decayed transition product equals $\gamma_C$ times the ungated product $I - \sum_r w_r k_r^\top$. The decay factors are scalars, so they collect into one number in front. The key side therefore keeps the ungated matrix of Equation 6, and the decay enters only afterward through $\overleftarrow{w}_r = \gamma_r w_r$.

A value written at position $i$ has been decayed by $\gamma_r / \gamma_i$ by the time position $r$ reads it, so its correction term in the recursion for $\tilde{u}_r$ carries that ratio. In matrix form the ratio multiplies the Gram matrix element-wise, which gives $\tilde{T}$ with $\Gamma \odot K K^\top$. This exercise fixes that split, matching Section 3.3 and the official reference code, so the key vectors use the ungated $T$ and the value vectors use the gated $\tilde{T}$.

With every $g_r = 0$, $\Gamma$ is all ones on and below the diagonal, the two T matrices coincide, and the result reduces to DeltaNet's Equations 6 and 7.

These outputs feed the chunkwise forward pass, which subtracts $\overleftarrow{W} S^\top$ from $\tilde{U}$ to form pseudo-values, so the key vectors are measured from the start of the chunk. The paper's state orientation, value width by key width, is used throughout.

## Two keys at a right angle after correction

Take $C = 2$ with keys $k_1 = (0, 1)$ and $k_2 = (0.6, 0.8)$, so $k_1^\top k_2 = 0.8$. Let $\beta = (1, 0.5)$, $g = (-0.2, -1.0)$ and values $v_1 = 2$, $v_2 = 1$.

- The plain strictly lower entry is $\beta_2 \cdot 0.8 = 0.4$, so $T$ has rows $(1, 0)$ and $(-0.4, 0.5)$.
- The decay ratio is $\gamma_2 / \gamma_1 = e^{-1.0} = 0.3679$, so the gated entry is $0.1472$ and $\tilde{T}$ has rows $(1, 0)$ and $(-0.1472, 0.5)$.
- Row two of $TK$ is $-0.4\,(0, 1) + 0.5\,(0.6, 0.8) = (0.3, 0)$, which is orthogonal to $k_1$. The correction removed the part of $k_2$ the first write covers.
- With $\gamma = (0.8187, 0.3012)$, the decayed key vectors are $(0, 0.8187)$ and $(0.0904, 0)$.
- The gated value vectors are $2$ and $-0.1472 \cdot 2 + 0.5 = 0.2057$.

## Common mistakes to avoid

- **Gating the key side.** Using $\Gamma \odot K K^\top$ for $T$ double counts the decay, because the key side receives it once through $\gamma_r$ afterward.
- **Keeping the diagonal.** Including it adds $\beta_r k_r^\top k_r$ to every pivot, so the system stops being unit triangular.
- **Scaling columns instead of rows.** The write strength belongs to the row index, $\operatorname{diag}(\beta) K K^\top$, and scaling columns gives the transpose of the intended interaction.
- **Transposing the decay ratio.** The mask uses $\gamma_i / \gamma_j$ for $i \ge j$, which is at most one, while its reciprocal explodes under strong decay.
- **Dropping the right factor.** Without the trailing $\operatorname{diag}(\beta)$, the diagonal of $T$ becomes one and every vector loses its write strength.
- **Scaling to the wrong end.** The key vectors are decayed to the start of the chunk with $\gamma_r$, and the end-of-chunk factor $\gamma_C / \gamma_r$ belongs to the keys in the state update.