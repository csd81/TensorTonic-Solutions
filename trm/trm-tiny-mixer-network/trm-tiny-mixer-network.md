# Tiny Mixer Network

The Tiny Recursive Model (TRM) of Jolicoeur-Martineau, 2025 solves puzzles by applying one tiny network many times instead of one large network once. This problem builds that network in its attention-free form. Every layer mixes information across positions with a SwiGLU along the sequence axis, then mixes channels with a second SwiGLU. The recursion problems feed it the sum of the question, answer and latent embeddings.

## Mixing positions with a matrix instead of attention

Self-attention lets every position read every other position through scores computed from the content. Its parameters live in a projection of size $[D, 3D]$ for the queries, keys and values, so the same weights handle any sequence length $L$. The paper notes that this suits long contexts where $L \gg D$.

The token-mixing alternative in Section 4.5 applies an MLP along the sequence axis, in the spirit of MLP-Mixer. Transposing the hidden states turns each channel into a vector of length $L$, and the token weights $G^t_\ell, U^t_\ell \in \mathbb{R}^{L \times H_t}$ and $W^t_\ell \in \mathbb{R}^{H_t \times L}$ act on that vector. Mixing is then a learned function of absolute positions. The network therefore only accepts the exact length it was built for.

The two choices trade off in a predictable way.

- **Attention generalizes over length.** Its weights never mention a position, so a model trained on one grid size can in principle read another.
- **The sequence MLP fixes the length.** Its $[L, L]$-scale matrices are cheap when $L \le D$ and impose a rigid positional structure.

## Why Sudoku gets the mixer and large grids keep attention

A 9 by 9 Sudoku has 81 cells and the paper's hidden width is 512, so $L \le D$ and the sequence MLP is small. The board never changes shape, and its rules tie fixed cells together, which suits a positional mixing matrix. Table 1 reports that dropping self-attention lifts Sudoku-Extreme accuracy from 74.7% to 87.4%. For Maze-Hard and ARC-AGI the inputs are 30 by 30 grids with 900 positions, which makes the $[L, L]$ matrices large, and the paper finds the MLP version worse there, so it keeps self-attention for those tasks.

## Two layers and more recursion beat four layers

Section 4.4 tried to scale capacity by adding layers and saw generalization fall, which the paper attributes to overfitting on roughly a thousand training puzzles. Halving the depth to two layers while raising the number of recursions to keep compute similar raised Sudoku-Extreme accuracy from 79.5% to 87.4%, as Table 1 shows. The recursion problems supply depth by calling this shallow block repeatedly.

## Post-norm on two different axes

The paper lists RMSNorm, no biases and SwiGLU but does not say where the norm sits. The released code uses post-norm with a gain-free RMS normalization, and this problem fixes that convention with $\epsilon = 10^{-5}$ by default. After token mixing, the code normalizes the transposed tensor over its last axis, which is the sequence axis, and only then swaps back. The exercise follows that order, so the first normalization rescales each channel across positions and the second rescales each position across channels.

Three smaller conventions come from the same code. The gate and up projections are stored as separate tensors here, while the code fuses them into one projection and splits it with a chunk. The SiLU is applied to the gate branch and the up branch stays linear. The released normalization casts to float32 internally, while this exercise computes it in the input dtype.

## Mixing two positions by hand

Take one sequence with $L = D = 2$, hidden state rows $(1, 2)$ and $(3, 0)$, one layer, $H_t = 1$, and zero channel weights. Let the token weights be $G^t = (1, 0)^\top$, $U^t = (1, 1)^\top$ and $W^t = (1, 0)$. The transposed rows are channel one $(1, 3)$ and channel two $(2, 0)$.

- **Gate and up values.** Channel one gives gate $1$ and up $4$, and channel two gives gate $2$ and up $2$.
- **Token update.** $\operatorname{SiLU}(1) \cdot 4 = 2.9242$ and $\operatorname{SiLU}(2) \cdot 2 = 3.5232$, and $W^t$ writes both values into position one only.
- **Sequence-axis norm.** Channel one becomes $(3.9242, 3)$ and normalizes to $(1.1235, 0.8589)$. Channel two becomes $(5.5232, 0)$ and normalizes to $(1.4142, 0)$.
- **Channel-axis norm.** Position one, $(1.1235, 1.4142)$, ends as $(0.8797, 1.1073)$, and position two ends as $(1.4142, 0)$.

The value 3 from position two reached position one, which is the cross-position flow that attention would otherwise provide.

## Common mistakes to avoid

- **Normalizing channels after token mixing.** The first normalization runs over positions while the tensor is still transposed, and swapping back first changes every value.
- **Skipping the transpose.** Multiplying the untransposed states by the token weights mixes channels, and it only runs at all when $L = D$.
- **Using pre-norm.** The released block adds the residual first and normalizes the sum.
- **Dropping a residual.** Both halves keep the identity path, and losing either one discards the incoming state.
- **Putting the SiLU on the up branch.** The gate branch receives the nonlinearity and the up branch scales it linearly.
- **Reusing the first layer's weights.** Every layer has its own token and channel weights, so the loop must index the layer each time.