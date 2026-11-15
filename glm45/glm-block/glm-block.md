# <span style="font-size: 20px;">GLM-4.5 Transformer Block</span>

<span style="font-size: 14px;">The GLM-4.5 transformer block (Zeng et al., 2025) is the repeated unit inside the 46-layer GLM-4.5 model. It is a pre-norm, two-residual decoder block: an RMSNorm-gated GQA self-attention sublayer with QK-norm and partial RoPE, then an RMSNorm-gated FFN sublayer that is **dense SwiGLU for the first layer only** and a **group-routed sparse MoE with a shared expert** for every layer after.</span>

---

## <span style="font-size: 16px;">High-Level Shape of the Block</span>

<span style="font-size: 14px;">The block reads one residual stream $x \in \mathbb{R}^{n \times H}$ and returns another tensor of the same shape. Two sublayers fire in sequence, each gated by RMSNorm on the way in and added back to the residual stream on the way out:</span>

$$
x \;\leftarrow\; x + \text{Attn}\bigl(\text{RMSNorm}(x;\,\gamma_1)\bigr)
$$

$$
x \;\leftarrow\; x + \text{FFN}\bigl(\text{RMSNorm}(x;\,\gamma_2)\bigr)
$$

<span style="font-size: 14px;">The FFN dispatch depends on the layer index:</span>

* <span style="font-size: 14px;">**Dense SwiGLU** when $\text{layer\_idx} < \text{first\_k\_dense\_replace}$. In stock GLM-4.5 this is exactly layer 0.</span>
* <span style="font-size: 14px;">**Sparse MoE with shared expert** for every layer where $\text{layer\_idx} \geq \text{first\_k\_dense\_replace}$. In stock GLM-4.5 this is layers 1 through 45.</span>

---

## <span style="font-size: 16px;">Sublayer 1: Pre-Norm + GQA Attention</span>

<span style="font-size: 14px;">RMSNorm normalizes along the last axis with no mean subtraction:</span>

$$
\text{RMSNorm}(x;\,\gamma) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{H}\sum_i x_i^2 + \epsilon}}
$$

<span style="font-size: 14px;">The normalized $t \in \mathbb{R}^{n \times H}$ is projected to a fused QKV tensor and split:</span>

* <span style="font-size: 14px;">$Q \in \mathbb{R}^{n \times H_q \times d}$ with $H_q$ query heads.</span>
* <span style="font-size: 14px;">$K, V \in \mathbb{R}^{n \times H_{kv} \times d}$ with $H_{kv}$ key-value heads (GQA: $H_q$ is an integer multiple of $H_{kv}$).</span>
* <span style="font-size: 14px;">In GLM-4.5, QKV uses a bias term (unusual among modern open LLMs, kept from earlier GLM versions).</span>

<span style="font-size: 14px;">Three GLM-4.5-specific operations are applied to $Q$ and $K$ before the dot product:</span>

* <span style="font-size: 14px;">**QK-norm.** RMSNorm is applied independently to $Q$ and $K$ along the head dimension, each with its own learned gain ($\gamma_q, \gamma_k \in \mathbb{R}^d$). This is the main GLM diff from Llama and stabilizes attention logits at scale.</span>
* <span style="font-size: 14px;">**Partial RoPE.** Only the first $\text{rope\_dim}$ channels of each head get the rotary embedding. The remaining $d - \text{rope\_dim}$ channels pass through untouched.</span>
* <span style="font-size: 14px;">**GQA expansion.** $K$ and $V$ are broadcast (no parameters) from $H_{kv}$ heads to $H_q$ heads by repeating each KV head $H_q / H_{kv}$ times along a new axis.</span>

<span style="font-size: 14px;">Standard causal scaled dot-product attention then runs per head:</span>

$$
\text{Attn}_h(Q, K, V) = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}} + M\right) V_h
$$

<span style="font-size: 14px;">where $M$ is the causal mask. The per-head outputs are concatenated and passed through an output projection $W_o$ (with bias $b_o$ in GLM-4.5). The full attention output is added to the residual stream.</span>

---

## <span style="font-size: 16px;">Partial RoPE Mechanics</span>

<span style="font-size: 14px;">RoPE rotates pairs of channels by an angle that depends on token position. GLM-4.5 only rotates the first $\text{rope\_dim}$ channels per head. With doubled cos/sin (the half-table $(n, \text{rope\_dim}/2)$ concatenated with itself to width $\text{rope\_dim}$) the rotation is:</span>

$$
y_{\text{rot}} = x_{\text{rot}} \odot \cos + \text{rotate\_half}(x_{\text{rot}}) \odot \sin
$$

$$
y = \text{concat}\bigl[y_{\text{rot}},\;x_{\text{pass}}\bigr]
$$

* <span style="font-size: 14px;">$x_{\text{rot}}$ is the first $\text{rope\_dim}$ channels and $x_{\text{pass}}$ is the remaining $d - \text{rope\_dim}$ channels.</span>
* <span style="font-size: 14px;">$\text{rotate\_half}$ swaps and signs the two halves of $x_{\text{rot}}$: $[x_1, x_2] \mapsto [-x_2, x_1]$.</span>
* <span style="font-size: 14px;">When $\text{rope\_dim} = d$, partial RoPE reduces to full RoPE.</span>

---

## <span style="font-size: 16px;">Sublayer 2: Pre-Norm + FFN Dispatch</span>

<span style="font-size: 14px;">After RMSNorm with $\gamma_2$, the block branches by layer index. The dispatch is decided per layer at construction time, not per token.</span>

### <span style="font-size: 14px;">Dense SwiGLU (layer 0 only)</span>

$$
\text{FFN}_{\text{dense}}(t) = \bigl(\text{SiLU}(t W_{\text{gate}}^\top) \odot (t W_{\text{up}}^\top)\bigr) W_{\text{down}}^\top
$$

<span style="font-size: 14px;">where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{I \times H}$ and $W_{\text{down}} \in \mathbb{R}^{H \times I}$. $I$ is the dense intermediate size. This is the same gated-MLP shape used by Llama and most modern decoder LMs.</span>

### <span style="font-size: 14px;">Sparse MoE + Shared Expert (every layer after the first)</span>

<span style="font-size: 14px;">Each MoE FFN has three pieces:</span>

* <span style="font-size: 14px;">**Router.** A linear projection $W_g \in \mathbb{R}^{E \times H}$ produces logits, then $\sigma(\cdot)$ scores each of $E = $ `n_routed_experts` experts per token. GLM-4.5 uses sigmoid (independent expert scores), not softmax.</span>
* <span style="font-size: 14px;">**Routed experts.** $E$ expert-specific SwiGLU FFNs. Each token routes to `num_experts_per_tok` experts using a group-restricted top-k.</span>
* <span style="font-size: 14px;">**Shared expert.** A single dense SwiGLU FFN that runs for every token, regardless of routing. Its output is added to the routed mixture.</span>

---

## <span style="font-size: 16px;">Group-Restricted Top-k Routing</span>

<span style="font-size: 14px;">Experts are partitioned into `n_group` equal groups, with `n_routed_experts / n_group` experts per group. The router:</span>

<span style="font-size: 14px;">1. **Sigmoid-score every expert.** $s = \sigma(t W_g^\top) \in \mathbb{R}^{n \times E}$.</span>

<span style="font-size: 14px;">2. **Score each group** as the sum of its top-2 expert scores within the group.</span>

<span style="font-size: 14px;">3. **Select top `topk_group` groups** per token and mask scores in all other groups to $-\infty$.</span>

<span style="font-size: 14px;">4. **Take top-`num_experts_per_tok` experts** from the unmasked scores; gather their pre-mask sigmoid scores as raw weights.</span>

<span style="font-size: 14px;">5. **Normalize and scale.** If `norm_topk_prob` is set, divide weights by their sum; then multiply by `routed_scaling_factor`. GLM-4.5 ships with both.</span>

<span style="font-size: 14px;">Group restriction reduces the effective expert search space per token, which improves load balancing and lets the model use many more experts without making routing decisions trivially imbalanced.</span>

---

## <span style="font-size: 16px;">Shared Expert</span>

<span style="font-size: 14px;">The shared expert is a normal SwiGLU FFN evaluated on every token, in parallel with the routed mixture. Final FFN output is:</span>

$$
\text{FFN}_{\text{moe}}(t) = \sum_{i=1}^{k} w_i\,\text{Expert}_{e_i}(t) + \text{SharedExpert}(t)
$$

<span style="font-size: 14px;">The shared expert always sees the token, even when routing is poor or experts disagree. It absorbs common patterns and frees the routed experts to specialize. GLM-4.5 uses `shared_intermediate_size` for this expert (often equal to `moe_intermediate_size`).</span>

---

## <span style="font-size: 16px;">Composing the Block</span>

<span style="font-size: 14px;">Putting the pieces together, one forward pass through a GLM-4.5 block does:</span>

<span style="font-size: 14px;">1. $t = \text{RMSNorm}(x;\,\gamma_1)$</span>

<span style="font-size: 14px;">2. $a = \text{GLMAttention}(t;\,W_{qkv}, b_{qkv}, W_o, b_o, \gamma_q, \gamma_k, \cos, \sin, \text{rope\_dim}, H_q, H_{kv}, d)$</span>

<span style="font-size: 14px;">3. $x \leftarrow x + a$</span>

<span style="font-size: 14px;">4. $t_2 = \text{RMSNorm}(x;\,\gamma_2)$</span>

<span style="font-size: 14px;">5. $y = \text{FFN}_{\text{dense}}(t_2)$ if $\text{layer\_idx} < \text{first\_k\_dense\_replace}$, else $y = \text{FFN}_{\text{moe}}(t_2) = \text{RoutedMixture}(t_2) + \text{SharedExpert}(t_2)$</span>

<span style="font-size: 14px;">6. **Return** $x + y$</span>

---

## <span style="font-size: 16px;">Comparison with Sibling Blocks</span>

* <span style="font-size: 14px;">**Llama block (Touvron et al., 2023):** full RoPE on the entire head, no QK-norm, dense SwiGLU FFN at every layer, no MoE, no shared expert. Same residual structure (pre-norm, two residuals) but no FFN dispatch.</span>
* <span style="font-size: 14px;">**GPT-OSS block (`gpto-block`):** every FFN is MoE, no dense layers. Attention has learned attention sinks and alternates sliding-window vs full attention by layer index. Routing is a simple softmax-after-topk without group restriction. No QK-norm, RoPE is full-width.</span>
* <span style="font-size: 14px;">**DeepSeek-V3 block (`ds3-*` family):** MLA (multi-head latent attention) with decoupled RoPE rather than GQA, but shares the group-restricted routing and shared-expert structure that GLM-4.5 adopted.</span>
* <span style="font-size: 14px;">**Sandwich-norm block (`at-block`):** post-norm around each sublayer plus depth-scaling. GLM-4.5 sticks with pre-norm only.</span>

---

## <span style="font-size: 16px;">Why These Specific Choices in GLM-4.5</span>

* <span style="font-size: 14px;">**Pre-norm + two residuals** is the dominant LLM topology for a reason: gradients flow through the residual stream untouched by normalization, which keeps very deep networks (46 layers here) trainable.</span>
* <span style="font-size: 14px;">**QK-norm** addresses a real failure mode at scale: as $H_q$ and $d$ grow, dot products $Q K^\top / \sqrt{d}$ can drift to extreme magnitudes, causing softmax saturation. Normalizing Q and K to unit RMS per head clamps logits before they enter softmax.</span>
* <span style="font-size: 14px;">**Partial RoPE** lets the model dedicate part of each head to position-invariant content while still encoding position in the remaining channels, a compromise that worked well in GLM-4 and is preserved here.</span>
* <span style="font-size: 14px;">**Dense layer 0** stabilizes the very first FFN step before tokens have any routing signal. Routing depends on $t_2$, and at layer 0 that representation is still close to raw embeddings: not enough signal for sigmoid routing to be informative. A single dense layer fixes this without inflating parameter count.</span>
* <span style="font-size: 14px;">**Group-restricted routing** scales to many experts. GLM-4.5 has hundreds of routed experts; without group restriction the top-k operation over all of them is both expensive and prone to load imbalance.</span>
* <span style="font-size: 14px;">**Shared expert** absorbs token-generic computation that does not benefit from specialization. It is the most-used FFN in the network and is roughly the analog of a dense FFN running in parallel with the experts.</span>

---

## <span style="font-size: 16px;">Worked Example (tiny config)</span>

<span style="font-size: 14px;">Take $n = 2$ tokens, $H = 4$, $H_q = 2$, $H_{kv} = 1$, $d = 2$, $\text{rope\_dim} = 2$. Layer 0 with $\text{first\_k\_dense\_replace} = 1$ goes through the dense branch.</span>

<span style="font-size: 14px;">1. Normalize: $t = \text{RMSNorm}(x;\,\gamma_1)$. For each row, divide by its RMS then multiply by $\gamma_1$.</span>

<span style="font-size: 14px;">2. Project to QKV: $(n + 2n_{kv})d = (2 + 2)\cdot 2 = 8$ columns. Split into $Q$ of shape $(2, 2, 2)$, $K, V$ of shape $(2, 1, 2)$.</span>

<span style="font-size: 14px;">3. QK-norm on $Q$ and $K$ along the last axis; partial RoPE rotates both channels of each head (since $\text{rope\_dim} = d$ here, partial RoPE equals full RoPE).</span>

<span style="font-size: 14px;">4. GQA: tile $K$ and $V$ from 1 head to 2 (each kv head feeds 2 query heads).</span>

<span style="font-size: 14px;">5. Causal SDPA per head, concat outputs, apply $W_o + b_o$.</span>

<span style="font-size: 14px;">6. Add to residual, RMSNorm with $\gamma_2$, then dense SwiGLU because layer 0 < 1.</span>

<span style="font-size: 14px;">7. Add the FFN result to the residual stream and return.</span>

<span style="font-size: 14px;">If we now switch to $\text{layer\_idx} = 1$, step 6 instead runs the MoE branch: route each of the 2 tokens to `num_experts_per_tok` experts via group-restricted top-k, sum the weighted expert outputs, and add the shared expert evaluated on the same $t_2$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Post-norm instead of pre-norm.** Applying RMSNorm after the residual ($x = \text{RMSNorm}(x + \text{Sublayer}(x))$) is a different architecture and trains poorly without warmup tricks. GLM-4.5 normalizes the input to the sublayer only.</span>
* <span style="font-size: 14px;">**Missing the second residual.** Returning $y$ instead of $x + y$ at the FFN step throws away half of the block. The residual is what keeps the network trainable at depth 46.</span>
* <span style="font-size: 14px;">**Wrong FFN dispatch.** Using dense SwiGLU at every layer (or MoE at every layer including 0) silently breaks the model: layer 0 needs dense, layers 1+ need MoE. The condition is `layer_idx < first_k_dense_replace`, strict less-than.</span>
* <span style="font-size: 14px;">**Forgetting QK-norm.** Skipping the RMSNorm on $Q$ and $K$ matches the Llama recipe, not GLM. At small head counts the difference is small but with the full $H_q = 96$ heads of GLM-4.5 the softmax saturates and training stalls.</span>
* <span style="font-size: 14px;">**Rotating all $d$ channels instead of $\text{rope\_dim}$ channels.** When $\text{rope\_dim} < d$ the trailing channels must pass through. Easy to miss because most reference Llama code applies RoPE to the full head.</span>
* <span style="font-size: 14px;">**Forgetting the shared expert in the MoE branch.** A pure routed mixture trains worse and loses the absolute baseline that the shared expert provides. The shared expert is added to the routed sum, not gated by it.</span>
* <span style="font-size: 14px;">**Forgetting `routed_scaling_factor`.** GLM-4.5 multiplies the (optionally normalized) top-k weights by a scalar before the expert mixture. Skipping it shrinks the FFN signal by exactly that factor.</span>
* <span style="font-size: 14px;">**Mixing softmax with sigmoid in the router.** GLM-4.5 routes with sigmoid (per-expert independent scores) so the raw weights do not sum to 1. The `norm_topk_prob` flag handles renormalization. Replacing sigmoid with softmax silently changes routing behavior.</span>
* <span style="font-size: 14px;">**Wrong KV repetition for GQA.** Tile each KV head to $q_{\text{mult}} = H_q / H_{kv}$ query heads in order. Striding wrong (e.g. interleaving) misaligns heads with their queries.</span>
* <span style="font-size: 14px;">**Causal mask off-by-one.** GLM-4.5 attends to all earlier positions including itself. The mask is upper-triangular with diagonal kept (i.e. `triu(..., diagonal=1)` for the $-\infty$ region).</span>

---
