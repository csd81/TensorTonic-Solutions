# <span style="font-size: 20px;">DeepSeek V3 Transformer Block</span>

<span style="font-size: 14px;">The DeepSeek V3 Transformer Block is the fundamental repeating unit of the DeepSeek V3 architecture. Each block takes a hidden state of shape $(B, S, d)$ and transforms it through two sub-layers -- Multi-head Latent Attention (MLA) and a feedforward sub-layer -- both wrapped in pre-norm RMSNorm with residual connections. The feedforward is conditionally either dense SwiGLU or sparse MoE with shared and routed experts, depending on the block's layer index.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A DeepSeek V3 block refines token representations through two sub-layers: attention (inter-token communication) and feedforward (per-token computation). Three choices distinguish it from a standard Transformer:</span>

* <span style="font-size: 14px;">**Multi-head Latent Attention (MLA):** Instead of standard MHA, MLA compresses the KV cache through low-rank joint compression. Q and K are decomposed into a compressed latent plus decoupled RoPE components, drastically reducing KV cache memory at inference.</span>
* <span style="font-size: 14px;">**Conditional FFN type:** The first `num_dense` layers (3 in DeepSeek V3) use dense SwiGLU. All remaining layers (3 through 60) use sparse MoE combining shared experts (always active) with routed experts (token-conditionally selected).</span>
* <span style="font-size: 14px;">**Pre-norm architecture:** Both sub-layers apply RMSNorm before computation. The residual connection adds the sub-layer output back to the un-normed input.</span>

<span style="font-size: 14px;">The block follows a strict 6-step pipeline. The FFN type at Step 5 is determined by layer index, not by any learned gate.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{B \times S \times d}$ be the input to the block, $L$ be the 0-based layer index, and $N_{\text{dense}}$ be the number of dense-FFN layers (3 in DeepSeek V3).</span>

### <span style="font-size: 14px;">Step 1: Pre-norm before attention</span>

$$
\hat{x} = \text{RMSNorm}(x, \gamma_{\text{attn}})
$$

<span style="font-size: 14px;">The original un-normed $x$ is preserved for the residual connection in Step 3.</span>

### <span style="font-size: 14px;">Step 2: Multi-head Latent Attention (MLA)</span>

<span style="font-size: 14px;">The normalized $\hat{x}$ enters MLA. The query side compresses into a low-rank latent, then up-projects to produce the content query plus a decoupled RoPE query:</span>

$$
c_Q = \hat{x} \, W_{DQ}, \quad [q_C; \, q_R] = \text{RMSNorm}(c_Q) \, [W_{UQ}; \, W_{QR}]
$$

<span style="font-size: 14px;">The KV side jointly compresses $\hat{x}$ into a shared latent, from which K and V are up-projected. A separate projection produces the decoupled RoPE key:</span>

$$
c_{KV} = \hat{x} \, W_{DKV}, \quad [k_C; \, v] = c_{KV} \, W_{UKV}, \quad k_R = \hat{x} \, W_{KR}
$$

<span style="font-size: 14px;">RoPE is applied only to $q_R$ and $k_R$. The full Q and K concatenate content and RoPE parts:</span>

$$
Q = [q_C; \; \text{RoPE}(q_R)], \quad K = [k_C; \; \text{RoPE}(k_R)]
$$

$$
\text{attn\_out} = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_h}} + \text{causal\_mask}\right) V \cdot W_O
$$

<span style="font-size: 14px;">During inference, only $c_{KV}$ and $k_R$ are cached per token (not full K and V).</span>

### <span style="font-size: 14px;">Step 3: Residual connection (attention)</span>

$$
h = x + \text{attn\_out}
$$

<span style="font-size: 14px;">The attention output is added to the original un-normed $x$, not to $\hat{x}$.</span>

### <span style="font-size: 14px;">Step 4: Pre-norm before FFN</span>

$$
\hat{h} = \text{RMSNorm}(h, \gamma_{\text{ffn}})
$$

<span style="font-size: 14px;">The un-normed $h$ is preserved for the residual in Step 6.</span>

### <span style="font-size: 14px;">Step 5: Feedforward (conditional on layer index)</span>

<span style="font-size: 14px;">**Case A: Dense SwiGLU (when $L < N_{\text{dense}}$):**</span>

$$
\text{ffn\_out} = \left(\text{SiLU}(\hat{h} \, W_{\text{gate}}) \odot (\hat{h} \, W_{\text{up}})\right) W_{\text{down}}
$$

<span style="font-size: 14px;">Standard SwiGLU: all tokens share the same dense weights. No routing, no expert selection.</span>

<span style="font-size: 14px;">**Case B: Sparse MoE with shared + routed experts (when $L \geq N_{\text{dense}}$):**</span>

<span style="font-size: 14px;">The MoE layer combines $N_s$ shared experts (always active) with $N_r$ routed experts ($K_r$ selected per token). Each expert is a SwiGLU FFN with independent weights.</span>

$$
g_i = \text{softmax}\!\left(\text{Sigmoid}(s_i) \cdot \frac{s_i}{\|s_i\|_2}\right), \quad s_i = \hat{h} \cdot e_i^T
$$

$$
\text{routed} = \sum_{i \in \text{TopK}_r} g_i \cdot \text{FFN}_i^{\text{routed}}(\hat{h})
$$

$$
\text{shared} = \sum_{j=1}^{N_s} \text{FFN}_j^{\text{shared}}(\hat{h})
$$

$$
\text{ffn\_out} = \text{shared} + \text{routed}
$$

<span style="font-size: 14px;">The scores $s_i$ pass through sigmoid, are multiplied by their L2-normalized version, and a softmax over the top-$K_r$ selected experts produces the final weights $g_i$. Shared expert outputs are summed without any gating.</span>

### <span style="font-size: 14px;">Step 6: Residual connection (FFN)</span>

$$
\text{output} = h + \text{ffn\_out}
$$

<span style="font-size: 14px;">The FFN output is added to the un-normed $h$ from Step 3. This sum becomes the input to the next block.</span>

### <span style="font-size: 14px;">RMSNorm definition</span>

$$
\text{RMSNorm}(x, \gamma) = \frac{x \cdot \gamma}{\sqrt{\text{mean}(x^2, \text{dim}=-1) + \epsilon}}
$$

<span style="font-size: 14px;">Each sub-layer has its own $\gamma$: $\gamma_{\text{attn}}$ for attention pre-norm and $\gamma_{\text{ffn}}$ for FFN pre-norm. Two separate learned vectors per block.</span>

---

## <span style="font-size: 16px;">How It Differs from a Standard Transformer Block</span>

<span style="font-size: 14px;">A standard pre-norm Transformer block (as in LLaMA, GPT-2) follows this pattern:</span>

$$
h = x + \text{MHA}(\text{LayerNorm}(x))
$$

$$
\text{output} = h + \text{FFN}(\text{LayerNorm}(h))
$$

<span style="font-size: 14px;">The DeepSeek V3 block modifies the attention mechanism and the feedforward sub-layer:</span>

* <span style="font-size: 14px;">**Attention mechanism:** Standard MHA maintains separate K and V projections per head. MLA replaces this with low-rank joint KV compression, reducing per-token KV cache from $2 n_h d_h$ to $d_c + d_R$ values.</span>
* <span style="font-size: 14px;">**RoPE handling:** Standard Transformers apply RoPE to full Q and K vectors. MLA decouples a separate RoPE component ($q_R, k_R$) from the content components ($q_C, k_C$), because compressed $c_{KV}$ cannot carry position information.</span>
* <span style="font-size: 14px;">**FFN type:** Standard uses a single dense FFN for all layers. DeepSeek V3 uses dense SwiGLU for the first 3 layers and sparse MoE (shared + routed experts) for the remaining 58, activating only a fraction of total parameters per token.</span>
* <span style="font-size: 14px;">**Expert routing:** MoE layers use sigmoid activation with L2 normalization and top-K selection, rather than standard softmax-only routing. Shared experts ensure every token receives baseline processing regardless of routing decisions.</span>
* <span style="font-size: 14px;">**Dense prefix:** The first 3 layers are always dense, providing a stable representation foundation before introducing MoE sparsity.</span>

---

## <span style="font-size: 16px;">The FFN Routing Decision</span>

<span style="font-size: 14px;">The FFN type is a hard-coded architectural decision based on layer index, not a learned gate:</span>

$$
\text{FFN}(\hat{h}) = \begin{cases} \text{DenseSwiGLU}(\hat{h}) & \text{if } L < N_{\text{dense}} \\ \text{SharedExperts}(\hat{h}) + \text{RoutedMoE}(\hat{h}) & \text{if } L \geq N_{\text{dense}} \end{cases}
$$

<span style="font-size: 14px;">In DeepSeek V3, $N_{\text{dense}} = 3$, meaning layers 0, 1, and 2 use dense SwiGLU, while layers 3 through 60 use the MoE configuration.</span>

<span style="font-size: 14px;">Why a dense prefix? Early layers form basic representations from raw embeddings. Sparse MoE here would give different tokens fundamentally different transformations before meaningful representations exist. By layer 3, representations are rich enough for conditional expert selection.</span>

<span style="font-size: 14px;">In MoE layers, shared experts act as a safety net: every token passes through them regardless of routing. Routed experts provide specialized processing on top.</span>

<span style="font-size: 14px;">Dense and MoE blocks have entirely different parameter sets. A dense block has one $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ triple. An MoE block has separate triples per expert plus router centroids $e_i$. These are not interchangeable.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Full model architecture</span>

<span style="font-size: 14px;">DeepSeek V3 stacks 61 blocks (same pipeline, different parameters) sequentially. Before the first block, token embeddings are looked up ($d = 7168$). After the last block, a final RMSNorm and linear head produce logits.</span>

### <span style="font-size: 14px;">Layer configuration</span>

<span style="font-size: 14px;">Layers 0-2 use dense SwiGLU FFN; layers 3-60 use MoE FFN. All 61 blocks use the same MLA attention -- the dense/MoE distinction applies only to the feedforward sub-layer.</span>

### <span style="font-size: 14px;">MoE configuration</span>

<span style="font-size: 14px;">Each MoE layer has 1 shared expert and 256 routed experts, with $K_r = 8$ selected per token. Each expert is a SwiGLU FFN with intermediate dimension 2048. Only $1 + 8 = 9$ experts are active per token, giving the model 671B total parameters but only ~37B active parameters per token.</span>

### <span style="font-size: 14px;">MLA dimensions</span>

<span style="font-size: 14px;">MLA uses 128 heads with $d_h = 128$, compressed KV dimension $d_c = 512$, RoPE dimension $d_R = 64$, and query compression dimension 1536. The KV cache stores $d_c + d_R = 576$ values per token vs. $2 \times 128 \times 128 = 32768$ in standard MHA (~57x reduction).</span>

### <span style="font-size: 14px;">Auxiliary losses in MoE layers</span>

<span style="font-size: 14px;">MoE layers use auxiliary load-balancing losses to prevent expert collapse. DeepSeek V3 also introduces auxiliary-loss-free balancing via per-expert bias terms adjusted during training.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace one token through a block with $d = 4$, $d_c = 2$, $d_R = 1$, 2 heads with $d_h = 2$. We show both FFN paths: $L = 1$ (dense) and $L = 5$ (MoE).</span>

### <span style="font-size: 14px;">Input</span>

$$
x = [1.0, \; -0.5, \; 0.8, \; -0.3]
$$

### <span style="font-size: 14px;">Step 1: Pre-norm (attention)</span>

<span style="font-size: 14px;">Compute RMS: $\text{rms} = \sqrt{\frac{1.0^2 + 0.5^2 + 0.8^2 + 0.3^2}{4}} = \sqrt{\frac{1.0 + 0.25 + 0.64 + 0.09}{4}} = \sqrt{0.495} \approx 0.7036$</span>

<span style="font-size: 14px;">With $\gamma_{\text{attn}} = [1.0, 1.0, 1.0, 1.0]$:</span>

$$
\hat{x} = \frac{x}{0.7036} \approx [1.421, \; -0.711, \; 1.137, \; -0.426]
$$

### <span style="font-size: 14px;">Step 2: MLA attention</span>

<span style="font-size: 14px;">Compress KV: $c_{KV} = \hat{x} \, W_{DKV} = [0.6, \; -0.2]$ (cached). Decoupled RoPE key: $k_R = \hat{x} \, W_{KR} = [0.35]$ (also cached).</span>

<span style="font-size: 14px;">Up-project K and V from $c_{KV}$, compute Q via its own compression path, apply RoPE to decoupled components, run causal attention, and project through $W_O$:</span>

$$
\text{attn\_out} = [0.30, \; 0.15, \; -0.10, \; 0.25]
$$

### <span style="font-size: 14px;">Step 3: Residual (attention)</span>

$$
h = x + \text{attn\_out} = [1.0 + 0.30, \; -0.5 + 0.15, \; 0.8 + (-0.10), \; -0.3 + 0.25]
$$

$$
h = [1.30, \; -0.35, \; 0.70, \; -0.05]
$$

### <span style="font-size: 14px;">Step 4: Pre-norm (FFN)</span>

<span style="font-size: 14px;">RMS of $h$: $\sqrt{\frac{1.69 + 0.1225 + 0.49 + 0.0025}{4}} = \sqrt{0.5763} \approx 0.7591$</span>

$$
\hat{h} = \frac{h}{0.7591} \approx [1.713, \; -0.461, \; 0.922, \; -0.066]
$$

### <span style="font-size: 14px;">Step 5A: Dense SwiGLU (layer L = 1, which is less than 3)</span>

<span style="font-size: 14px;">Since $L = 1 < N_{\text{dense}} = 3$, this layer uses the dense SwiGLU path. With intermediate dimension 8:</span>

$$
\text{ffn\_out} = \left(\text{SiLU}(\hat{h} \, W_{\text{gate}}) \odot (\hat{h} \, W_{\text{up}})\right) W_{\text{down}} = [0.20, \; -0.10, \; 0.35, \; 0.05]
$$

### <span style="font-size: 14px;">Step 5B: Sparse MoE (layer L = 5, which is at least 3)</span>

<span style="font-size: 14px;">Since $L = 5 \geq N_{\text{dense}} = 3$, this layer uses the MoE path. Suppose 1 shared expert and 4 routed experts with $K_r = 2$.</span>

<span style="font-size: 14px;">Router scores: $s = \hat{h} \cdot E^T = [0.8, \; -0.3, \; 0.5, \; 1.2]$.</span>

<span style="font-size: 14px;">Top-2 selection picks experts 3 ($s=1.2$) and 0 ($s=0.8$). Apply sigmoid + normalization + softmax to get routing weights $g_3 = 0.58, \; g_0 = 0.42$.</span>

<span style="font-size: 14px;">Each selected expert and the shared expert compute SwiGLU independently:</span>

$$
\text{shared} = \text{FFN}_{\text{shared}}(\hat{h}) = [0.12, \; -0.08, \; 0.18, \; 0.04]
$$

$$
\text{routed} = 0.42 \times [0.15, \; 0.05, \; -0.10, \; 0.20] + 0.58 \times [-0.05, \; 0.12, \; 0.25, \; -0.08] = [0.034, \; 0.091, \; 0.103, \; 0.038]
$$

$$
\text{ffn\_out} = \text{shared} + \text{routed} = [0.154, \; 0.011, \; 0.283, \; 0.078]
$$

### <span style="font-size: 14px;">Step 6: Residual (FFN)</span>

<span style="font-size: 14px;">Using the MoE result from Step 5B:</span>

$$
\text{output} = h + \text{ffn\_out} = [1.454, \; -0.339, \; 0.983, \; 0.028]
$$

<span style="font-size: 14px;">This vector is the input to the next block (or to the final RMSNorm if this is the last block).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">1. Applying MoE routing to a dense-prefix layer</span>

<span style="font-size: 14px;">The first 3 layers ($L < 3$) must use dense SwiGLU, not MoE. The dense-prefix layers have a single set of FFN weights and no router at all. Attempting to route through experts on these layers will fail because expert weight tensors do not exist for those blocks.</span>

### <span style="font-size: 14px;">2. Applying dense SwiGLU to an MoE layer</span>

<span style="font-size: 14px;">The reverse mistake: using a single dense FFN for layers that should use MoE ($L \geq 3$). This collapses the sparse expert structure into a single transformation, discarding routing entirely. The conditional must be strict: $L < N_{\text{dense}}$ for dense, $L \geq N_{\text{dense}}$ for MoE.</span>

### <span style="font-size: 14px;">3. Forgetting the residual connections</span>

<span style="font-size: 14px;">Both sub-layers require a residual that adds the sub-layer output to the un-normed input. The correct pattern is $h = x + f(\text{RMSNorm}(x))$, not $h = \hat{x} + f(\hat{x})$. Omitting the residual entirely destroys the gradient highway and prevents deep stacking.</span>

### <span style="font-size: 14px;">4. Wrong RMSNorm placement (post-norm instead of pre-norm)</span>

<span style="font-size: 14px;">DeepSeek V3 uses pre-norm: RMSNorm is applied before the sub-layer, and the residual adds the raw sub-layer output to the un-normed input. Post-norm (normalizing after the residual addition) has different training dynamics. Using post-norm when the model expects pre-norm produces incorrect outputs and instability.</span>

### <span style="font-size: 14px;">5. Forgetting shared experts in MoE layers</span>

<span style="font-size: 14px;">The MoE output is the sum of shared and routed expert outputs. Omitting shared experts removes the guaranteed baseline processing. The shared expert output is added directly (no routing weight), so it must not be multiplied by any gate value.</span>

### <span style="font-size: 14px;">6. Applying RoPE to content Q/K components in MLA</span>

<span style="font-size: 14px;">In MLA, RoPE is applied only to the decoupled components $q_R$ and $k_R$, not to the content components $q_C$ and $k_C$. Applying RoPE to the full Q or K would inject positional information into components derived from the compressed latent $c_{KV}$, breaking the KV cache compression scheme. The decoupled design keeps $c_{KV}$ position-independent so it can be up-projected at inference without re-applying positional encoding.</span>