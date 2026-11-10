# <span style="font-size: 20px;">Transformer Encoder Block</span>

<span style="font-size: 14px;">The Transformer encoder block is the core repeating unit in BERT, the encoder side of T5/BART, and Vision Transformers. Understanding its exact structure - the interplay of self-attention, feed-forward networks, residual connections, and normalization - is essential for any deep learning interview.</span>

---

## <span style="font-size: 16px;">Block Architecture</span>

<span style="font-size: 14px;">Each encoder block contains two sublayers:</span>

<span style="font-size: 14px;">**Sublayer 1 - Multi-Head Self-Attention**: the input $x$ serves as queries, keys, and values. This allows each token to attend to all other tokens in the sequence, building contextualized representations. The output has the same shape as the input.</span>

<span style="font-size: 14px;">**Sublayer 2 - Position-wise Feed-Forward Network (FFN)**: two linear transformations with a ReLU activation in between. "Position-wise" means the same FFN is applied independently to each position (token). The FFN expands the dimension from $d_{\text{model}}$ to $d_{\text{ff}}$ (typically $4 \times d_{\text{model}}$) and then projects back:</span>

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2
$$

<span style="font-size: 14px;">Each sublayer is wrapped with a residual connection and layer normalization:</span>

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

---

## <span style="font-size: 16px;">Why Residual Connections Matter</span>

<span style="font-size: 14px;">Residual connections (He et al., 2016) solve two critical problems in deep Transformers:</span>

<span style="font-size: 14px;">**1. Gradient flow**: without residuals, gradients must flow through every sublayer during backpropagation. In a 12-layer BERT or 96-layer GPT-3, this leads to vanishing gradients. The residual path provides a "gradient highway" where gradients can skip sublayers entirely: $\partial L / \partial x = \partial L / \partial \text{output} \cdot (I + \partial \text{Sublayer}/\partial x)$. The identity term ensures gradients always flow.</span>

<span style="font-size: 14px;">**2. Feature refinement**: each sublayer adds a small refinement to the input rather than completely transforming it. This makes training easier because the sublayer only needs to learn the residual (the difference from identity), not the entire transformation. Early in training, sublayer outputs are small, so the block approximates the identity function.</span>

---

## <span style="font-size: 16px;">The Feed-Forward Network</span>

<span style="font-size: 14px;">The FFN is applied independently and identically to each position. Despite its simplicity (just two linear layers), it serves a crucial role: it provides the model's "computation" capacity. Self-attention can only compute weighted averages of value vectors - linear combinations. The FFN adds nonlinearity through ReLU, allowing the model to compute arbitrary functions of the attended features.</span>

<span style="font-size: 14px;">The expansion ratio $d_{\text{ff}} / d_{\text{model}}$ (typically 4) creates a bottleneck architecture. The expansion to $d_{\text{ff}}$ allows the network to work in a higher-dimensional space where complex patterns are easier to represent. The projection back to $d_{\text{model}}$ compresses this into a compact representation. Recent work suggests individual FFN neurons often correspond to interpretable features or concepts.</span>

<span style="font-size: 14px;">**Modern variants**: LLaMA and other recent models replace ReLU with SwiGLU: $\text{FFN}(x) = (W_1 x \odot \text{Swish}(W_3 x)) W_2$. This uses a gating mechanism that empirically improves performance. GPT-NeoX uses GELU instead of ReLU.</span>

---

## <span style="font-size: 16px;">Post-LN vs Pre-LN</span>

<span style="font-size: 14px;">The original Transformer and BERT use **Post-LN** (normalize after the residual add):</span>

$$
x = \text{LN}(x + \text{Sublayer}(x))
$$

<span style="font-size: 14px;">Most modern architectures (GPT-2/3, LLaMA, Mistral) use **Pre-LN** (normalize before the sublayer):</span>

$$
x = x + \text{Sublayer}(\text{LN}(x))
$$

<span style="font-size: 14px;">Pre-LN is preferred because: (1) it keeps the residual stream unnormalized, preserving gradient magnitude across layers; (2) it does not require learning rate warm-up; (3) the final output is unnormalized, which can be handled by an extra LayerNorm at the end. Post-LN can achieve slightly better final performance but is harder to train, especially in deeper models.</span>

<span style="font-size: 14px;">This problem implements Post-LN to match the original Transformer specification.</span>

---


## <span style="font-size: 16px;">Scaling and Modern Variants</span>

<span style="font-size: 14px;">**Depth scaling.** BERT-base uses 12 encoder blocks; BERT-large uses 24. Increasing depth generally improves representational power but with diminishing returns and increased training instability. Techniques like pre-norm, careful initialization, and learning rate warmup become critical at scale.</span>

<span style="font-size: 14px;">**Sparse attention.** For long documents, some encoder blocks replace full self-attention with sparse patterns (local windows, global tokens, or random connections). Longformer and BigBird use this approach to process documents of 4096+ tokens with linear complexity.</span>

<span style="font-size: 14px;">**Mixture of Experts (MoE).** The feed-forward network in each block can be replaced with multiple expert FFNs, where a router selects the top-k experts per token. This scales model capacity without proportionally scaling compute, used in Switch Transformer and Mixtral.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why is the FFN called "position-wise"?**</span>
  <span style="font-size: 14px;">A: The same weight matrices $W_1$ and $W_2$ are applied independently to each position in the sequence. There is no interaction between positions in the FFN - all cross-position information exchange happens in the self-attention sublayer. This is equivalent to applying a shared MLP to each token independently.</span>

- <span style="font-size: 14px;">**Q: What is the parameter count formula for an encoder block?**</span>
  <span style="font-size: 14px;">A: MHA: $4(d_{\text{model}}^2 + d_{\text{model}})$ (four linear projections with bias). FFN: $2 \cdot d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}} + d_{\text{model}}$ (two linear layers with biases). LayerNorm: $4 \cdot d_{\text{model}}$ (two norms, each with weight and bias). Total: $4d^2 + 2d \cdot d_{\text{ff}} + 5d + d_{\text{ff}}$ where $d = d_{\text{model}}$.</span>

- <span style="font-size: 14px;">**Q: Why do we need both self-attention and FFN? Could one work alone?**</span>
  <span style="font-size: 14px;">A: Self-attention computes weighted averages, which are linear operations on the values. Without the FFN, stacking attention layers can only compute increasingly complex linear combinations - the representational power would be limited. The FFN's nonlinearity (ReLU/GELU) is what allows the Transformer to approximate arbitrary functions. Conversely, FFN alone has no cross-position interaction, so it cannot capture dependencies between tokens.</span>

- <span style="font-size: 14px;">**Q: How does the encoder handle variable-length sequences?**</span>
  <span style="font-size: 14px;">A: A padding mask is passed to the self-attention: positions corresponding to padding tokens are blocked (set to $-10^9$ before softmax). This prevents real tokens from attending to padding. The FFN is position-wise so it naturally handles any length. The LayerNorm operates per-token so it is also length-agnostic.</span>

- <span style="font-size: 14px;">**Q: In BERT, how many encoder blocks are stacked and what are the dimensions?**</span>
  <span style="font-size: 14px;">A: BERT-Base uses 12 encoder blocks with $d_{\text{model}} = 768$, $h = 12$ heads, $d_{\text{ff}} = 3072$ (4x expansion). BERT-Large uses 24 blocks with $d_{\text{model}} = 1024$, $h = 16$, $d_{\text{ff}} = 4096$. The total parameter count (including embeddings) is 110M for Base and 340M for Large.</span>

---