# <span style="font-size: 20px;">Arcee Trinity Transformer Block</span>

<span style="font-size: 14px;">The Arcee Trinity Transformer Block is the fundamental repeating unit of the Arcee Trinity architecture. Each block takes a hidden state of shape $(B, S, d)$ and transforms it through a gated attention sub-layer followed by a coarse MoE feedforward sub-layer, both wrapped in sandwich normalization with depth scaling. The model stacks many such blocks sequentially.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">A Trinity block refines token representations through two sub-layers: attention (inter-token communication) and feedforward (per-token computation). Four modifications distinguish it from a standard Transformer block:</span>

* <span style="font-size: 14px;">**Gated attention:** A learned sigmoid gate element-wise modulates the attention output before the residual connection, giving the model fine-grained control over which attention features to pass through.</span>
* <span style="font-size: 14px;">**Coarse-grained MoE FFN:** Instead of a single dense feedforward network, the intermediate dimension is split into expert groups. A sigmoid router selects top-$k$ groups per token, making the FFN conditionally sparse.</span>
* <span style="font-size: 14px;">**Sandwich normalization:** Both sub-layers apply RMSNorm twice -- once before the sub-layer (pre-norm) and once after (post-norm) -- rather than the single pre-norm used in standard Transformer blocks.</span>
* <span style="font-size: 14px;">**Depth scaling:** The post-normed sub-layer output is divided by $\sqrt{2L + 1}$ (where $L$ is the 0-based layer index) before being added to the residual stream. This prevents gradient explosion in deep stacks.</span>

<span style="font-size: 14px;">The block processes its input through a strict 6-step pipeline. No step can be reordered or omitted without breaking the architecture.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{B \times S \times d}$ be the input to the block and $L$ be the 0-based layer index.</span>

### <span style="font-size: 14px;">Step 1: Pre-norm before attention</span>

$$
\hat{x} = \text{RMSNorm}(x, \gamma_{\text{attn\_pre}})
$$

### <span style="font-size: 14px;">Step 2: Gated causal attention</span>

$$
Q = \hat{x} W_q^T, \quad K = \hat{x} W_k^T, \quad V = \hat{x} W_v^T
$$

$$
\text{attn\_out} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \text{causal\_mask}\right) V
$$

$$
\text{gate} = \sigma(\hat{x} \cdot W_{\text{gate}}^T)
$$

$$
\text{gated} = \text{gate} \odot (\text{attn\_out} \cdot W_o^T)
$$

### <span style="font-size: 14px;">Step 3: Post-norm + depth scale + residual (attention)</span>

$$
h = x + \frac{\text{RMSNorm}(\text{gated}, \gamma_{\text{attn\_post}})}{\sqrt{2L + 1}}
$$

### <span style="font-size: 14px;">Step 4: Pre-norm before FFN</span>

$$
\hat{h} = \text{RMSNorm}(h, \gamma_{\text{ffn\_pre}})
$$

### <span style="font-size: 14px;">Step 5: Coarse MoE FFN</span>

$$
\text{moe\_out} = \sum_{i \in \text{TopK}} w_i \cdot W_{\text{down}}[:, s_i] \cdot \bigl(\text{swish}(W_{\text{gate}}^{\text{ffn}}[s_i]\, \hat{h}) \odot W_{\text{up}}[s_i]\, \hat{h}\bigr)
$$

<span style="font-size: 14px;">where $s_i$ is the slice of the intermediate dimension belonging to expert $i$, and $w_i$ are the renormalized sigmoid router weights.</span>

### <span style="font-size: 14px;">Step 6: Post-norm + depth scale + residual (FFN)</span>

$$
\text{output} = h + \frac{\text{RMSNorm}(\text{moe\_out}, \gamma_{\text{ffn\_post}})}{\sqrt{2L + 1}}
$$

### <span style="font-size: 14px;">RMSNorm definition</span>

$$
\text{RMSNorm}(x, \gamma) = \frac{x \cdot \gamma}{\sqrt{\text{mean}(x^2, \text{dim}=-1) + \epsilon}}
$$

<span style="font-size: 14px;">Each sub-layer has its own pair of $\gamma$ vectors: $\gamma_{\text{attn\_pre}}, \gamma_{\text{attn\_post}}$ for attention, and $\gamma_{\text{ffn\_pre}}, \gamma_{\text{ffn\_post}}$ for FFN. These are four separate learned parameter vectors per block.</span>

---

## <span style="font-size: 16px;">Step-by-Step Walkthrough</span>

### <span style="font-size: 14px;">Step 1: Pre-norm before attention</span>

<span style="font-size: 14px;">The input $x$ is passed through RMSNorm with $\gamma_{\text{attn\_pre}}$, stabilizing the input so the RMS of each token's feature vector is approximately 1. The original $x$ is preserved for the residual connection in Step 3.</span>

### <span style="font-size: 14px;">Step 2: Gated causal attention</span>

<span style="font-size: 14px;">The normalized $\hat{x}$ is projected into Q, K, V. Scaled dot-product attention with a causal mask is computed, and the result is projected through $W_o$.</span>

<span style="font-size: 14px;">A gate vector $\sigma(\hat{x} \cdot W_{\text{gate}}^T)$ produces values in $[0, 1]$ for each feature dimension. The gate and the projected attention output are multiplied element-wise, allowing the model to selectively suppress or amplify individual attention features. The gate input is the same pre-normed $\hat{x}$, not the raw $x$.</span>

### <span style="font-size: 14px;">Step 3: Post-norm + depth scale + residual (attention)</span>

<span style="font-size: 14px;">The gated output passes through a second RMSNorm with separate parameters $\gamma_{\text{attn\_post}}$ -- this is the "sandwich" part. The post-normed result is divided by $\sqrt{2L + 1}$ and added to the original $x$.</span>

<span style="font-size: 14px;">The depth scale shrinks with depth: layer 0 divides by $1.0$, layer 5 by $\approx 3.32$, layer 31 by $\approx 7.94$. This progressively dampens each layer's contribution.</span>

### <span style="font-size: 14px;">Step 4: Pre-norm before FFN</span>

<span style="font-size: 14px;">The residual output $h$ from Step 3 is normalized with $\gamma_{\text{ffn\_pre}}$. The un-normed $h$ is preserved for Step 6.</span>

### <span style="font-size: 14px;">Step 5: Coarse MoE FFN</span>

<span style="font-size: 14px;">The normalized $\hat{h}$ enters the coarse-grained MoE. The intermediate dimension is partitioned into $N$ expert groups. A sigmoid router scores each expert, selects top-$k$, renormalizes weights to sum to 1, and runs only the selected SwiGLU experts on their slices. Weighted outputs are summed.</span>

<span style="font-size: 14px;">"Coarse-grained" means all experts share the same weight matrices but operate on different contiguous slices. This is more parameter-efficient than fine-grained MoE with fully independent expert weights.</span>

### <span style="font-size: 14px;">Step 6: Post-norm + depth scale + residual (FFN)</span>

<span style="font-size: 14px;">Same structure as Step 3: post-norm with $\gamma_{\text{ffn\_post}}$, divide by $\sqrt{2L + 1}$, add to $h$ (the Step 3 residual, not the original $x$). This produces the final block output.</span>

---

## <span style="font-size: 16px;">How It Differs from a Standard Transformer Block</span>

<span style="font-size: 14px;">A standard pre-norm Transformer block (as in GPT-2, LLaMA) follows this pattern:</span>

$$
h = x + \text{Attention}(\text{LayerNorm}(x))
$$

$$
\text{output} = h + \text{FFN}(\text{LayerNorm}(h))
$$

<span style="font-size: 14px;">The Trinity block modifies every part of this pattern:</span>

* <span style="font-size: 14px;">**Normalization:** Standard uses one pre-norm per sub-layer (2 norms total). Trinity uses pre-norm AND post-norm per sub-layer (4 norms total), each with independent learned parameters.</span>
* <span style="font-size: 14px;">**Attention gating:** Standard applies $W_o$ projection and adds directly to the residual. Trinity interposes a sigmoid gate between $W_o$ projection and the post-norm, element-wise modulating the attention output.</span>
* <span style="font-size: 14px;">**FFN type:** Standard uses a single dense FFN (SwiGLU in modern models). Trinity uses coarse-grained MoE with sigmoid routing, activating only a subset of the intermediate dimension per token.</span>
* <span style="font-size: 14px;">**Depth scaling:** Standard has no per-layer scaling. Trinity divides each sub-layer's contribution by $\sqrt{2L + 1}$ before the residual addition, making deeper layers contribute less.</span>
* <span style="font-size: 14px;">**Residual path:** Standard adds the sub-layer output directly. Trinity interposes post-norm and depth scaling before the residual addition. The residual stream itself is never normalized.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Stacking blocks into a full model</span>

<span style="font-size: 14px;">Arcee Trinity stacks identical blocks (same architecture, different parameters) sequentially. Before the first block, token embeddings are looked up with no absolute positional embedding. After the final block, a final RMSNorm and an untied linear head produce logits.</span>

### <span style="font-size: 14px;">Interleaved RoPE/NoPE pattern</span>

<span style="font-size: 14px;">Although every block has the same 6-step structure, the attention sub-layer differs across blocks in how it handles positional encoding. Trinity uses an interleaved pattern controlled by a `rope_ratio` parameter:</span>

$$
\text{is\_nope} = (\text{layer\_idx} + 1) \bmod (\text{rope\_ratio} + 1) = 0
$$

<span style="font-size: 14px;">With `rope_ratio = 3`, the cycle is: RoPE, RoPE, RoPE, NoPE, RoPE, RoPE, RoPE, NoPE, and so on. RoPE layers apply Rotary Position Embeddings to Q and K before computing attention scores. NoPE layers skip positional encoding entirely, computing attention based purely on content similarity.</span>

<span style="font-size: 14px;">This interleaving combines position-aware attention (RoPE) with position-invariant attention (NoPE), letting NoPE layers capture abstract semantic patterns independent of token position.</span>

### <span style="font-size: 14px;">Depth scaling across the full stack</span>

<span style="font-size: 14px;">The depth scale factor $\frac{1}{\sqrt{2L + 1}}$ uses each block's 0-based index $L$. In a 32-block model, the first block ($L=0$) scales by $1/\sqrt{1} = 1.0$ (no attenuation), while the last block ($L=31$) scales by $1/\sqrt{63} \approx 0.126$. Both sub-layers within the same block use the same scale factor.</span>

### <span style="font-size: 14px;">Parameter isolation per block</span>

<span style="font-size: 14px;">Each block has independent parameters: attention weights ($W_q, W_k, W_v, W_o, W_{\text{gate}}$), FFN weights ($W_{\text{gate}}^{\text{ffn}}, W_{\text{up}}, W_{\text{down}}$), router weights, and four RMSNorm $\gamma$ vectors. No weights are shared between blocks.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a single token through one Trinity block with $d = 4$, layer index $L = 2$. We use simplified values throughout.</span>

### <span style="font-size: 14px;">Input</span>

$$
x = [1.0, \; -0.5, \; 0.8, \; -0.3]
$$

### <span style="font-size: 14px;">Step 1: Pre-norm (attention)</span>

<span style="font-size: 14px;">Compute RMS: $\text{rms} = \sqrt{\frac{1.0^2 + (-0.5)^2 + 0.8^2 + (-0.3)^2}{4}} = \sqrt{\frac{1.0 + 0.25 + 0.64 + 0.09}{4}} = \sqrt{0.495} \approx 0.7036$</span>

<span style="font-size: 14px;">With $\gamma_{\text{attn\_pre}} = [1.0, 1.0, 1.0, 1.0]$ (for simplicity):</span>

$$
\hat{x} = \frac{x}{0.7036} \approx [1.4213, \; -0.7107, \; 1.1371, \; -0.4264]
$$

### <span style="font-size: 14px;">Step 2: Gated attention</span>

<span style="font-size: 14px;">Suppose after the full attention computation (Q, K, V projections, scaled dot-product, output projection), we get:</span>

$$
\text{attn\_proj} = W_o \cdot \text{attn\_out} = [0.4, \; 0.2, \; -0.1, \; 0.6]
$$

<span style="font-size: 14px;">The gate is computed from $\hat{x}$. Suppose $W_{\text{gate}}^T \hat{x}$ produces pre-sigmoid values $[0.5, -1.0, 0.8, 0.0]$:</span>

$$
\text{gate} = \sigma([0.5, -1.0, 0.8, 0.0]) = [0.622, \; 0.269, \; 0.690, \; 0.500]
$$

$$
\text{gated} = [0.622 \times 0.4, \; 0.269 \times 0.2, \; 0.690 \times (-0.1), \; 0.500 \times 0.6]
$$

$$
\text{gated} = [0.249, \; 0.054, \; -0.069, \; 0.300]
$$

<span style="font-size: 14px;">Notice how the gate suppressed the second dimension (0.269 scale) while passing the first and fourth dimensions more strongly.</span>

### <span style="font-size: 14px;">Step 3: Post-norm + depth scale + residual (attention)</span>

<span style="font-size: 14px;">RMS of gated $\approx 0.200$. With $\gamma_{\text{attn\_post}} = [1,1,1,1]$: post-normed $= [1.245, 0.270, -0.345, 1.500]$.</span>

<span style="font-size: 14px;">Depth scale for $L = 2$: $\frac{1}{\sqrt{5}} \approx 0.4472$</span>

$$
h = [1.0, -0.5, 0.8, -0.3] + 0.4472 \times [1.245, 0.270, -0.345, 1.500] = [1.557, -0.379, 0.646, 0.371]
$$

### <span style="font-size: 14px;">Step 4: Pre-norm (FFN)</span>

<span style="font-size: 14px;">RMS of $h \approx 0.883$. With $\gamma_{\text{ffn\_pre}} = [1,1,1,1]$:</span>

$$
\hat{h} = \frac{h}{0.883} \approx [1.763, -0.429, 0.731, 0.420]
$$

### <span style="font-size: 14px;">Step 5: Coarse MoE FFN</span>

<span style="font-size: 14px;">Intermediate dim 8, split into 4 experts of size 2. Router selects top-2 (expert 0, expert 3) with weights $w_0 = 0.6, w_3 = 0.4$. Each expert computes SwiGLU on its slice:</span>

$$
\text{moe\_out} = 0.6 \times [0.3, -0.1, 0.5, 0.2] + 0.4 \times [-0.2, 0.4, 0.1, -0.3] = [0.100, 0.100, 0.340, 0.000]
$$

### <span style="font-size: 14px;">Step 6: Post-norm + depth scale + residual (FFN)</span>

<span style="font-size: 14px;">RMS of moe\_out $\approx 0.1842$. Post-normed $= [0.543, 0.543, 1.846, 0.000]$. Same depth scale $0.4472$:</span>

$$
\text{output} = [1.557, -0.379, 0.646, 0.371] + 0.4472 \times [0.543, 0.543, 1.846, 0.000] = [1.800, -0.136, 1.471, 0.371]
$$

<span style="font-size: 14px;">This final vector is passed as input to the next block (or to the final RMSNorm if this is the last block).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">1. Wrong order of operations around the gate</span>

<span style="font-size: 14px;">The gate must be applied after the output projection $W_o$ but before the post-norm. The correct sequence is: attention -> $W_o$ projection -> gate modulation -> post-norm -> depth scale -> residual. A common mistake is applying the gate after post-norm, which changes what the gate modulates and breaks the sandwich norm structure.</span>

### <span style="font-size: 14px;">2. Applying depth scale to the wrong tensor</span>

<span style="font-size: 14px;">The depth scale must be applied to the post-normed sub-layer output only, not to the residual sum. The formula is $h = x + \frac{\text{post\_norm}(\text{sub\_out})}{\sqrt{2L+1}}$, not $h = \frac{x + \text{post\_norm}(\text{sub\_out})}{\sqrt{2L+1}}$. Scaling the full sum would attenuate the residual stream itself.</span>

### <span style="font-size: 14px;">3. Sharing norm parameters between sub-layers</span>

<span style="font-size: 14px;">Each block has four independent RMSNorm $\gamma$ vectors: $\gamma_{\text{attn\_pre}}, \gamma_{\text{attn\_post}}, \gamma_{\text{ffn\_pre}}, \gamma_{\text{ffn\_post}}$. A common error is using only two (one pre, one post) and sharing them between the attention and FFN sub-layers. Each sub-layer must have its own dedicated pair.</span>

### <span style="font-size: 14px;">4. Residual connection placement errors</span>

<span style="font-size: 14px;">The attention residual connects to block input $x$. The FFN residual connects to $h$ (attention output), not $x$. Writing $x + \text{ffn\_out}$ instead of $h + \text{ffn\_out}$ skips the attention contribution entirely.</span>

### <span style="font-size: 14px;">5. Confusing layer_idx across blocks</span>

<span style="font-size: 14px;">The depth scale factor uses the block's position in the full model stack (0-based `layer_idx`), not a sub-layer counter within the block. Both sub-layers (attention and FFN) in block $L$ use the same scale factor $\frac{1}{\sqrt{2L+1}}$. A mistake is to use $L$ for attention and $L+1$ for FFN within the same block, or to start counting from 1 instead of 0.</span>

### <span style="font-size: 14px;">6. Forgetting the causal mask</span>

<span style="font-size: 14px;">The attention must include a causal mask for both RoPE and NoPE layers. Omitting it turns the model into a bidirectional encoder.</span>

### <span style="font-size: 14px;">7. Mixing up the RoPE/NoPE decision scope</span>

<span style="font-size: 14px;">RoPE vs. NoPE affects only whether rotary embeddings are applied to Q and K. It does not change the FFN, gating, normalization, or depth scaling. The 6-step structure is identical for both block types.</span>

---