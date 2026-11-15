# <span style="font-size: 20px;">Full GLM-4.5 Forward Pass</span>

<span style="font-size: 14px;">The forward pass of GLM-4.5 (Zhipu AI, 2025) chains a token embedding lookup, a stack of transformer blocks, a final RMSNorm, and an output projection. The interesting structural choice is the **dense prefix**: layer 0 uses a vanilla SwiGLU MLP while every subsequent layer uses a Mixture-of-Experts (MoE) block with group-limited routing and a shared expert. This problem asks you to wire these pieces together end-to-end.</span>

---

## <span style="font-size: 16px;">The Pipeline</span>

<span style="font-size: 14px;">Given a flat sequence of $n$ token ids, the model emits a $(n, \text{vocab})$ logit tensor through five stages:</span>

* <span style="font-size: 14px;">**Embedding lookup**: $x = E[\text{ids}]$ where $E \in \mathbb{R}^{V \times d}$, producing $x \in \mathbb{R}^{n \times d}$.</span>
* <span style="font-size: 14px;">**Dense prefix block** (layer 0): pre-norm GQA self-attention with partial RoPE and QK norm, then residual, then pre-norm dense SwiGLU MLP, then residual.</span>
* <span style="font-size: 14px;">**MoE blocks** (layers 1..N-1): same attention sublayer, but the MLP is replaced by `gate_W` + routed experts + a shared expert, all summed and added back via residual.</span>
* <span style="font-size: 14px;">**Final RMSNorm**: a learned per-feature gain $\gamma_f$ rescales the residual stream so the logit projection sees a calibrated distribution.</span>
* <span style="font-size: 14px;">**Output projection**: $\text{logits} = x \, W_{\text{lm}}$ where $W_{\text{lm}} \in \mathbb{R}^{d \times V}$.</span>

$$
\text{logits}_t = (\text{RMSNorm}(h^{(N)}_t)) \cdot W_{\text{lm}}
$$

<span style="font-size: 14px;">Here $h^{(N)}$ is the residual stream after the last block. Everything below details what $h^{(N)}$ accumulates.</span>

---

## <span style="font-size: 16px;">Layer 0: The Dense Prefix</span>

<span style="font-size: 14px;">The first block follows the standard pre-norm recipe:</span>

$$
h \leftarrow h + \text{Attn}(\text{RMSNorm}(h))
$$

$$
h \leftarrow h + \text{SwiGLU}(\text{RMSNorm}(h))
$$

<span style="font-size: 14px;">Where SwiGLU is a three-matrix gated MLP:</span>

$$
\text{SwiGLU}(z) = \big( (z W_g^\top) \odot \sigma(z W_g^\top) \cdot (z W_u^\top) \big) W_d^\top
$$

<span style="font-size: 14px;">**Why a dense prefix?** Empirically MoE routing is unstable on the very first block because the residual stream is still dominated by raw embedding signal: token-level features have not yet been mixed across positions, so the router has nothing semantic to route on. GLM-4.5 (like DeepSeek-V3 and DeepSeek-V2) keeps the first `first_k_dense_replace` layers dense to give the router a usable input distribution before it starts.</span>

<span style="font-size: 14px;">For GLM-4.5 the published config is `first_k_dense_replace = 1` with 46 total layers, so 1 dense block plus 45 MoE blocks.</span>

---

## <span style="font-size: 16px;">Layers 1..N-1: The MoE Block</span>

<span style="font-size: 14px;">The attention sublayer is identical to layer 0. The MLP is replaced by a routed mixture plus a shared expert:</span>

$$
\text{MoE}(z) = \sum_{k=1}^{K} w_{k} \, E_{i_k}(z) + \text{SharedExpert}(z)
$$

<span style="font-size: 14px;">Where $E_e$ is a SwiGLU expert with its own $(W_g^e, W_u^e, W_d^e)$, $i_k$ are the indices of the selected experts, and $w_k$ are the normalized router weights. The shared expert is a single SwiGLU MLP that processes every token, so the MoE term only has to learn the **specialized** residual on top of a generic transformation.</span>

### <span style="font-size: 14px;">Group-Limited Routing</span>

<span style="font-size: 14px;">GLM-4.5 inherits group-limited topk routing from DeepSeek-V2. The $E$ routed experts are partitioned into $G$ groups of $E/G$ experts each (the paper calls this **node-restricted** routing because each group maps to one device). For each token:</span>

* <span style="font-size: 14px;">Compute raw gate scores $s = \sigma(z \, W_{\text{gate}}^\top) \in \mathbb{R}^E$ (sigmoid, not softmax).</span>
* <span style="font-size: 14px;">Add a per-expert bias used purely for selection: $\tilde{s} = s + b$.</span>
* <span style="font-size: 14px;">Group score for group $g$ is the sum of the top-2 biased scores inside that group.</span>
* <span style="font-size: 14px;">Pick the top `topk_group` groups; mask all experts outside them to $-\infty$.</span>
* <span style="font-size: 14px;">Take the top `num_experts_per_tok` experts inside the surviving groups.</span>
* <span style="font-size: 14px;">Renormalize the selected raw scores (not biased), then multiply by `routed_scaling_factor`.</span>

<span style="font-size: 14px;">**Why two scores?** The bias $b$ is updated by a load-balancing controller outside the gradient path. Using $\tilde{s}$ for selection lets the controller redirect tokens to under-utilized experts; using $s$ for the final weight keeps the gradient signal honest. This is the **auxiliary-loss-free balancing** scheme from DeepSeek-V3 and is reused verbatim in GLM-4.5.</span>

### <span style="font-size: 14px;">Why Sigmoid Gate</span>

<span style="font-size: 14px;">A softmax gate forces $\sum_e s_e = 1$, which couples experts: pulling one score up pushes others down. A sigmoid gate scores each expert independently. Combined with the explicit `norm_topk_prob` renormalization after topk, this gives the router more freedom to learn graded confidence per expert before competition is enforced.</span>

---

## <span style="font-size: 16px;">Attention Sublayer (Identical Across Layers)</span>

<span style="font-size: 14px;">Inside every block the attention path is:</span>

* <span style="font-size: 14px;">Pre-norm: $z = \text{RMSNorm}(h)$.</span>
* <span style="font-size: 14px;">Project: $Q = z W_q^\top$ reshaped to $(n, n_h, d_h)$, $K = z W_k^\top$ reshaped to $(n, n_{kv}, d_h)$, similarly $V$.</span>
* <span style="font-size: 14px;">QK norm: apply a per-head RMSNorm to each $Q$ head and each $K$ head with learned gain.</span>
* <span style="font-size: 14px;">Partial RoPE: rotate only the first `rope_dim` channels of each $Q$ and $K$ head, leaving the trailing channels as a position-invariant component.</span>
* <span style="font-size: 14px;">GQA expand: tile $K$ and $V$ from $n_{kv}$ heads to $n_h$ heads with `repeat_interleave`.</span>
* <span style="font-size: 14px;">Scaled dot-product with a strict causal mask.</span>
* <span style="font-size: 14px;">Standard softmax (no attention sinks, unlike GPT-OSS).</span>
* <span style="font-size: 14px;">Concatenate heads, project: $\text{out} = (\text{attn}) W_o^\top$, add residual.</span>

<span style="font-size: 14px;">QK norm stabilises the dot-product scale before softmax; partial RoPE preserves a non-positional subspace that the model can use for content-only matching. Both are direct carry-overs from DeepSeek-V3.</span>

---

## <span style="font-size: 16px;">Comparison With Sibling Forward Passes</span>

* <span style="font-size: 14px;">**at-forward (LLaMA-style)**: dense SwiGLU MLP at every layer, no MoE, full RoPE on the entire head, no QK norm.</span>
* <span style="font-size: 14px;">**gpto-forward (GPT-OSS)**: MoE at every layer (no dense prefix), softmax router with learned attention sinks, alternating sliding-window and full causal masks.</span>
* <span style="font-size: 14px;">**ds3-forward (DeepSeek-V3)**: closest cousin. Same group-limited router, same sigmoid gate, same dense prefix idea. The big DeepSeek-V3 wrinkle is Multi-Latent Attention (MLA) with decoupled RoPE on a small head; GLM-4.5 uses plain GQA with partial RoPE instead.</span>
* <span style="font-size: 14px;">**glm-forward (this problem)**: dense prefix of `first_k_dense_replace` layers, then MoE with sigmoid + group-limited router + bias-corrected selection + shared expert, GQA attention with partial RoPE + QK norm, no sliding window, no attention sinks.</span>

---

## <span style="font-size: 16px;">GLM-4.5 Configuration</span>

<span style="font-size: 14px;">The shipped GLM-4.5 weights use:</span>

* <span style="font-size: 14px;">$L = 46$ transformer layers with `first_k_dense_replace = 1` (1 dense, 45 MoE).</span>
* <span style="font-size: 14px;">$d = 4096$ hidden, $V = 151{,}552$ vocab, max positions $131{,}072$.</span>
* <span style="font-size: 14px;">$n_h = 96$ query heads, $n_{kv} = 8$ KV heads, $d_h = 128$ head dim, `rope_dim = 64` (half-RoPE).</span>
* <span style="font-size: 14px;">Dense MLP intermediate $= 12{,}288$. MoE intermediate $= 1408$ per expert. Shared expert intermediate $= 1408$.</span>
* <span style="font-size: 14px;">$E = 160$ routed experts, $K = 8$ active per token, $G = 8$ groups, `topk_group = 4`, `routed_scaling_factor = 2.5`.</span>
* <span style="font-size: 14px;">Total $\approx 358$B parameters, $\approx 32$B active per token.</span>

<span style="font-size: 14px;">For this problem the dimensions are scaled down dramatically (hidden 4-8, vocab 4-6, 2-5 layers) so each test case fits in the runner. The math is identical.</span>

---

## <span style="font-size: 16px;">Numerical Walk-Through</span>

<span style="font-size: 14px;">Take $n = 2$ tokens, $d = 4$, $V = 4$, $L = 2$ with `first_k_dense_replace = 1`. After embedding, $x \in \mathbb{R}^{2 \times 4}$.</span>

<span style="font-size: 14px;">1. **Layer 0 (dense)**: RMSNorm $\rightarrow$ GQA attention $\rightarrow$ residual $\rightarrow$ RMSNorm $\rightarrow$ SwiGLU MLP $\rightarrow$ residual. $x$ stays $2 \times 4$.</span>

<span style="font-size: 14px;">2. **Layer 1 (MoE)**: same attention sublayer. Then RMSNorm $\rightarrow$ compute gate logits $g = z W_{\text{gate}}^\top \in \mathbb{R}^{2 \times 2}$ for 2 experts. With $G = 1$ group and `topk_group = 1`, the group mask is a no-op. With `num_experts_per_tok = 1`, pick the highest biased score per token, look up the corresponding raw sigmoid score, multiply by `routed_scaling_factor`, run that expert's SwiGLU on $z$, accumulate. Add the shared expert SwiGLU output. Add to residual.</span>

<span style="font-size: 14px;">3. **Final norm**: rescale $x$ with $\gamma_f$.</span>

<span style="font-size: 14px;">4. **lm_head**: $\text{logits} = x \, W_{\text{lm}} \in \mathbb{R}^{2 \times 4}$. The runner checks every entry to $10^{-6}$.</span>

---

## <span style="font-size: 16px;">Implementation Map</span>

<span style="font-size: 14px;">Each per-layer dict in `blocks` carries everything the layer needs. The schema differs by layer type because dense and MoE layers have disjoint MLP weights:</span>

* <span style="font-size: 14px;">**Both layer types**: `attn_norm_g`, `W_q`, `W_k`, `W_v`, `W_o`, `q_norm_g`, `k_norm_g`, `mlp_norm_g`.</span>
* <span style="font-size: 14px;">**Dense layers only**: `W_gate`, `W_up`, `W_down`.</span>
* <span style="font-size: 14px;">**MoE layers only**: `gate_W`, `e_score_correction_bias`, `experts` (list of dicts with `W_gate`, `W_up`, `W_down`), `shared_W_gate`, `shared_W_up`, `shared_W_down`.</span>

<span style="font-size: 14px;">The layer index `li` is the dispatch signal: `if li < first_k_dense_replace` use the dense branch, else use the MoE branch.</span>

---

## <span style="font-size: 16px;">Complexity and Memory</span>

<span style="font-size: 14px;">The forward cost is dominated by a few terms per block:</span>

* <span style="font-size: 14px;">**Attention scores**: $O(n^2 \, n_h \, d_h)$ per layer, the only term that scales with $n^2$.</span>
* <span style="font-size: 14px;">**QKV projections + output projection**: $O(n \, d \, (n_h + 2 n_{kv}) \, d_h + n \, d^2)$ per layer.</span>
* <span style="font-size: 14px;">**Dense MLP** (layer 0): $O(n \, d \, d_{\text{ff}})$ with three matrices.</span>
* <span style="font-size: 14px;">**MoE MLP**: only the selected $K$ experts run per token, so cost is $O(n \, K \, d \, d_{\text{moe}})$, plus the shared expert contributes $O(n \, d \, d_{\text{shared}})$.</span>
* <span style="font-size: 14px;">**lm_head**: $O(n \, d \, V)$, often the single largest matmul because $V \approx 150{,}000$.</span>

<span style="font-size: 14px;">Memory at inference is dominated by the KV cache (one $K$ and one $V$ tensor per layer, sized $n \, n_{kv} \, d_h$) and by the MoE expert weights, which sit in memory whether or not they are selected for a given token.</span>

---

## <span style="font-size: 16px;">Modern Variants</span>

* <span style="font-size: 14px;">**DeepSeek-V3** uses MLA attention with a tiny RoPE head decoupled from a larger no-RoPE head, plus the same group-limited router. GLM-4.5 simplifies attention back to GQA + partial RoPE.</span>
* <span style="font-size: 14px;">**Qwen-2 MoE** drops the shared expert and uses a softmax gate. The expert specialization burden falls entirely on the routed experts.</span>
* <span style="font-size: 14px;">**Mixtral 8x7B** uses a small softmax gate over 8 experts with `topk = 2`, no group restriction. Suitable for single-node inference.</span>
* <span style="font-size: 14px;">**GPT-OSS** alternates sliding-window and full attention by layer index, uses attention sinks, and routes every layer (no dense prefix).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping the final RMSNorm.** Running `lm_head` on raw residual leaves the logits at the magnitude of the accumulated residual stream, which after 46 blocks is large. Cross-entropy still trains, but every problem test compares element-wise to a small tolerance and fails immediately.</span>
* <span style="font-size: 14px;">**Treating layer 0 as MoE.** Forgetting the dense prefix means dereferencing `lw["gate_W"]` on a layer 0 dict that does not contain it. The runner raises a `KeyError` or, if you defensively `.get` it, returns the wrong tensor. The branch must be `li < first_k_dense_replace`, not `li > 0` (those are equivalent only when `first_k_dense_replace == 1`).</span>
* <span style="font-size: 14px;">**Wrong embedding axis.** `embed_W[input_ids]` indexes rows. Writing `embed_W.T[input_ids]` or `embed_W[:, input_ids]` silently produces a tensor of the wrong shape downstream and trips the attention reshape.</span>
* <span style="font-size: 14px;">**Transposing lm_head.** `lm_head_W` is $(d, V)$ so $x \, W_{\text{lm}}$ is the correct multiplication. Using $W_{\text{lm}}^\top$ flips the output shape to $(n, d)$, which the runner catches via the shape assertion.</span>
* <span style="font-size: 14px;">**Dropping the last residual.** Computing the MoE output and assigning $x = \text{moe\_out}$ instead of $x = x + \text{moe\_out}$ wipes the attention contribution from that block. The bug looks identical to a healthy forward except logits are off by a constant residual stream.</span>
* <span style="font-size: 14px;">**Iterating layers in the wrong order.** The block list is ordered from layer 0 to layer N-1. Reversing it not only reorders computation but also makes layer 0 (dense) come last, which is structurally wrong: the dense MLP runs on an already-MoE-mixed residual and the MoE layers run on raw embeddings.</span>
* <span style="font-size: 14px;">**Confusing the two router scores.** Selection uses $s + b$, weighting uses $s$. Using the biased score as the weight injects the controller's load-balancing signal into the forward pass and biases the output systematically toward whichever experts the controller is trying to fill.</span>
* <span style="font-size: 14px;">**Forgetting the shared expert.** GLM-4.5 always adds a single SwiGLU MLP that processes every token regardless of routing. Skipping it removes the model's "default" transformation and the residual diverges from the reference.</span>
* <span style="font-size: 14px;">**Applying full RoPE.** Only the first `rope_dim` channels of each head are rotated. Rotating the entire head destroys the position-invariant subspace and is a common copy-paste bug from LLaMA-style implementations.</span>

---
