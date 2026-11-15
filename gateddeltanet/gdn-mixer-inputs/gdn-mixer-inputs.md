# Token Mixer Inputs

The token mixer of Gated DeltaNet (Yang, Kautz and Hatamizadeh, 2025) reads its memory through queries, writes to it with keys and values, and controls it with a decay and a write strength. This problem builds that input path from the hidden states. Queries, keys and values pass through a linear projection, a short causal convolution and SiLU, and queries and keys are L2-normalized per head, while the two gates come from linear projections alone. The path also keeps a small convolution cache so the layer can continue a sequence one token at a time.

## Why a short convolution sits in front of the recurrence

The gated delta rule updates its state with one key and one value per token, so on its own each write sees only the current position. A depthwise causal convolution of width $K_c$ lets every query, key and value channel mix the last $K_c$ projected tokens before the recurrence begins, which gives the layer a cheap local view of nearby tokens without attention. In Table S.1, removing it raises average perplexity from 27.35 to 28.95.

The convolution is causal, so the output at token $t$ reads only tokens up to $t$. In this exercise it is written as a cross-correlation in which the weight at index $K_c - 1$ multiplies the current token and index 0 multiplies the oldest token in the window, and it has no bias, matching the official layer.

## L2 normalization and the feature map

After SiLU, each query and key head vector $a \in \mathbb{R}^{d_k}$ is divided by $\sqrt{\sum_i a_i^2 + \epsilon}$. The delta rule erases the memory along the current key with the factor $I - \beta_t k_t k_t^\top$, and this factor is a proper projection-style update only when $k_t$ has unit length. Without it, a long key could overshoot the stored association, so the paper adds it for training stability. The epsilon formula and its default of $10^{-6}$ come from the flash-linear-attention module the official layer imports, because the paper does not give them.

Table S.1 also compares normalizations and feature maps. Swapping L2 for L1 normalization costs about three perplexity points with every activation tried, for example 30.18 with SiLU. Keeping L2 and changing the activation matters much less, with ReLU at 27.67 and $1 + \operatorname{ELU}$ at 27.58, but SiLU still gives the best result. Values are not normalized, because their scale is stored content.

## Gates that skip the convolution

The paper states that the decay $\alpha$ and the write strength $\beta$ use linear projections only. The paper defers the decay parameterization to Mamba2, and this exercise follows the official layer:

$$
g_{t,h} = -\exp(A_{\log,h}) \operatorname{softplus}(x_t W_{\alpha,h} + b_{dt,h})
$$

The softplus keeps the step size positive, and the factor $\exp(A_{\log,h})$ sets a learned rate per head, so the log-decay $g$ is never positive and the decay $\alpha = \exp(g)$ stays in $(0, 1]$. A plain sigmoid decay would also stay in that range, but it saturates at both ends, while the softplus form grows linearly for large logits and can drive the decay all the way to zero when a token should clear the memory.

## The cache that makes the layer resumable

To continue a sequence later, the convolution needs the last $K_c - 1$ projected rows of each branch, taken before the convolution is applied. This exercise stores each cache as $(B, K_c - 1, C)$ with the most recent row last, which is its own layout choice. The official code keeps the transposed layout. Because the cache keeps inputs and not outputs, running a sequence in two calls gives exactly the same result as one call.

## One channel through the path

Take one query channel with kernel width $K_c = 3$, weights $(0.5, 0, 1)$, cache rows $1$ and $-1$ with the most recent last, and a new projected value of $2$. The window is $(1, -1, 2)$, so the convolution gives $0.5 \cdot 1 + 0 \cdot (-1) + 1 \cdot 2 = 2.5$, and SiLU gives $2.5 \cdot \operatorname{sigmoid}(2.5) = 2.3104$. The new cache is $(-1, 2)$. For the gates, a decay logit of $1$ with $A_{\log} = 0$ and zero time-step bias gives $g = -\operatorname{softplus}(1) = -1.3133$, a decay of $0.2689$. A head vector $(3, 4)$ normalizes to about $(0.6, 0.8)$.

## Common mistakes to avoid

- **Centered padding.** Padding on both sides lets a token read future positions, which breaks causality and changes every output.
- **Flipping the kernel.** A true convolution reverses the weights, so the current token would be multiplied by the oldest weight.
- **Applying SiLU before the convolution.** The activation belongs after the local mixing.
- **Normalizing across all heads at once.** Each head of width $d_k$ is its own unit vector, and pooling heads shrinks every head.
- **Normalizing the values.** Only queries and keys are L2-normalized.
- **Dropping the time-step bias or using a sigmoid decay.** Both change the Mamba2 gate and its range of decays.
- **Caching convolution outputs.** The cache must hold pre-convolution rows, or a split call no longer matches a single call.