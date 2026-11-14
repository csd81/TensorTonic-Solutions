# <span style="font-size: 20px;">GPT-OSS Transformer Block</span>

<span style="font-size: 14px;">The GPT-OSS transformer block is the unit that repeats $L$ times to make the full model. It composes a **pre-norm GQA attention sublayer** with learned softmax sinks and an alternating sliding window, plus a **pre-norm sparse MoE sublayer** with top-4 routing over 128 clamped SwiGLU experts (in the open-weights release).</span>

---

## <span style="font-size: 16px;">What This Block Does</span>

<span style="font-size: 14px;">Each block transforms $x \in \mathbb{R}^{n \times H}$ into another tensor of the same shape: mixing information across preceding tokens via attention, then applying a nonlinear per-token transform via MoE. Residual streams flow around both sublayers.</span>

* <span style="font-size: 14px;">**Input shape:** $(n_{\text{tokens}}, H)$ where $H$ is the hidden size (no batch dim in the reference torch implementation).</span>
* <span style="font-size: 14px;">**Output shape:** same as input.</span>
* <span style="font-size: 14px;">**Sublayer 1:** pre-RMSNorm, QKV projection, RoPE on Q and K, scaled dot-product attention with **GQA + causal mask + optional sliding window + learned sinks**, output projection, residual add.</span>
* <span style="font-size: 14px;">**Sublayer 2:** pre-RMSNorm, router logits, top-$k$ expert selection with softmax-normalized weights, per-expert clamped SwiGLU MLP, weighted sum, residual add.</span>

---

## <span style="font-size: 16px;">Pre-Norm vs Post-Norm</span>

<span style="font-size: 14px;">The original Transformer paper (Vaswani et al., 2017) wrapped each sublayer as $\text{LN}(x + \text{Sublayer}(x))$ (post-norm). Modern LLMs almost universally adopt the **pre-norm** variant introduced by Xiong et al. (2020):</span>

$$
x \leftarrow x + \text{Sublayer}(\text{LN}(x))
$$

<span style="font-size: 14px;">Pre-norm keeps the residual stream un-normalized: the path from input embedding to logits is a clean sum of contributions, which has two practical consequences:</span>

* <span style="font-size: 14px;">**Gradient flow.** The identity branch is never squashed by a normalization at the joins, so signal magnitudes do not collapse with depth.</span>
* <span style="font-size: 14px;">**Trainability without warmup tricks.** Post-norm transformers typically need very small learning rates and long warmup to avoid divergence in early steps. Pre-norm is more forgiving.</span>

<span style="font-size: 14px;">GPT-OSS uses **RMSNorm** (Zhang and Sennrich, 2019), the variance-only variant of LayerNorm. For a vector $x \in \mathbb{R}^H$ with scale parameter $\gamma \in \mathbb{R}^H$:</span>

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

<span style="font-size: 14px;">There is no mean subtraction and no learned bias. RMSNorm is used in LLaMA, Mistral, Qwen, and GPT-OSS for both speed (one less reduction) and slightly better empirical behavior at scale.</span>

---

## <span style="font-size: 16px;">The Attention Sublayer</span>

<span style="font-size: 14px;">Concretely, the attention path looks like:</span>

$$
t = \text{RMSNorm}(x), \quad [q, k, v] = t W_{qkv}^\top + b_{qkv}
$$

<span style="font-size: 14px;">QKV is a single fused projection. With $H_q = $ num_attention_heads, $H_{kv} = $ num_key_value_heads, $d = $ head_dim, the output dimension is $(H_q + 2 H_{kv}) \cdot d$. After slicing, $q$ is reshaped to $(n, H_{kv}, q_{\text{mult}}, d)$ and $k, v$ to $(n, H_{kv}, d)$, where $q_{\text{mult}} = H_q / H_{kv}$. This nested layout makes the **grouped-query** structure explicit: $q_{\text{mult}}$ queries per kv head share one K and one V.</span>

<span style="font-size: 14px;">RoPE is then applied to $q$ and $k$ using the precomputed $\cos$ and $\sin$ tables. The rotation pairs the lower and upper halves of each head:</span>

$$
\text{RoPE}(x) = \big[x_{:d/2} \odot \cos - x_{d/2:} \odot \sin,\ x_{d/2:} \odot \cos + x_{:d/2} \odot \sin\big]
$$

<span style="font-size: 14px;">The $\cos$ and $\sin$ tables themselves come from a **YaRN-scaled** inverse-frequency schedule (Peng et al., 2023) that lets GPT-OSS extrapolate to 128K context from a 4K pretraining window. This block consumes those tables as inputs and does not recompute them.</span>

<span style="font-size: 14px;">The attention itself is scaled dot-product attention with three tweaks:</span>

* <span style="font-size: 14px;">**Grouped-query expansion.** Each $k, v$ head is broadcast across the $q_{\text{mult}}$ query heads sharing it. This is purely a tensor reshape: no extra parameters, but the same K, V values participate in $q_{\text{mult}}$ dot products.</span>
* <span style="font-size: 14px;">**Causal + sliding-window mask.** A standard upper-triangular causal mask ensures position $i$ never attends to $j > i$. On even-indexed layers GPT-OSS additionally bands the mask: tokens older than `sliding_window` positions are also masked. On odd-indexed layers full causal attention is used.</span>
* <span style="font-size: 14px;">**Learned attention sinks.** Each head has a scalar logit $s_h$ that is concatenated to its key-axis logits before softmax. Softmax then normalizes over $n+1$ entries; the sink "extra" probability is dropped after the softmax (it is not multiplied into any value).</span>

<span style="font-size: 14px;">Putting these together, for one head with query $q_i$ and stacked keys $K_{1:i}$:</span>

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{j' \le i, j' \ge i - w + 1} \exp(q_i^\top k_{j'} / \sqrt{d}) + \exp(s_h)}
$$

$$
o_i = \sum_{j \le i, j \ge i - w + 1} \alpha_{ij}\, v_j
$$

<span style="font-size: 14px;">The output is reshaped back to $(n, H_q d)$, projected through $W_o$, and added to the residual stream.</span>

---

## <span style="font-size: 16px;">Why Alternate Full and Windowed Attention</span>

<span style="font-size: 14px;">Sliding-window attention (Beltagy et al., Longformer 2020) keeps the per-layer cost linear in sequence length, but loses access to distant tokens. Full attention is quadratic but global. GPT-OSS, like Mistral and Gemma, **alternates** the two:</span>

* <span style="font-size: 14px;">**Even-indexed layers (`layer_idx % 2 == 0`):** sliding window of `config.sliding_window` (128 in the open release). Cheap, focuses on local structure.</span>
* <span style="font-size: 14px;">**Odd-indexed layers:** full causal attention. Expensive but each token can reach any earlier token.</span>

<span style="font-size: 14px;">Two layers of this pattern together let any token reach any earlier token: the windowed layer locally, then the full layer globally. The effective receptive field grows quickly while average per-layer cost stays close to linear in $n$. KV-cache memory of windowed layers is also bounded by the window.</span>

<span style="font-size: 14px;">In code:</span>

$$
\text{sliding\_window}_{\text{block}} = \begin{cases} \text{config.sliding\_window} & \text{layer\_idx} \% 2 == 0 \\ 0 & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">A common implementation bug is to flip the parity (apply sliding window on odd layers). Public test cases here include both even and odd variants of the same input precisely to catch that.</span>

---

## <span style="font-size: 16px;">Why Attention Sinks</span>

<span style="font-size: 14px;">Without sinks, softmax forces $\sum_j \alpha_{ij} = 1$ over a small windowed set of keys. If none of those keys is informative, the attention is forced to allocate full probability mass somewhere, producing spuriously confident outputs. Xiao et al. (2024) showed empirically that vanilla LLMs handle this by routing all "junk" attention into the first few tokens (their term: a "sink").</span>

<span style="font-size: 14px;">GPT-OSS bakes that mechanism in explicitly:</span>

* <span style="font-size: 14px;">**Per-head learned scalar.** Each head learns one extra logit $s_h$ that participates in the softmax denominator but contributes no value vector.</span>
* <span style="font-size: 14px;">**Softmax now sums to less than 1 over real keys.** This frees the head from having to pick a winner when no key is relevant.</span>
* <span style="font-size: 14px;">**Stabilizes sliding-window layers.** Because windowed layers have a small key set, they are especially prone to spurious normalization. Sinks restore a graceful "abstain" option.</span>

<span style="font-size: 14px;">Numerically, a very large positive $s_h$ collapses all real-key probability toward zero, effectively muting the head. A very negative $s_h$ pushes the sink out of contention, recovering vanilla softmax. The hidden tests exercise both extremes.</span>

---

## <span style="font-size: 16px;">The MoE Sublayer</span>

<span style="font-size: 14px;">After the first residual, a second pre-RMSNorm feeds into the MoE FFN:</span>

$$
t_2 = \text{RMSNorm}(x), \quad \ell = t_2 W_g^\top + b_g \in \mathbb{R}^{n \times E}
$$

<span style="font-size: 14px;">where $E$ is the number of experts (128 in GPT-OSS). For each token, the **top-$k$** experts (4 in GPT-OSS) are selected by logit value, and their logits are normalized by a softmax restricted to the top-$k$ set:</span>

$$
\text{top\_idx} = \text{argsort}(-\ell)[:, :k], \quad w_{ij} = \text{softmax}(\ell[i, \text{top\_idx}_i])_j
$$

<span style="font-size: 14px;">Each selected expert is a **clamped SwiGLU** MLP. Expert $e$ has parameters $W_1^e \in \mathbb{R}^{2I \times H}, b_1^e \in \mathbb{R}^{2I}, W_2^e \in \mathbb{R}^{H \times I}, b_2^e \in \mathbb{R}^H$, where $I$ is the intermediate size. The forward is:</span>

$$
z = W_1^e t_2 + b_1^e, \quad z_g = z[\dots, ::2], \quad z_l = z[\dots, 1::2]
$$

$$
\text{SwiGLU}(z) = \text{clamp}(z_g, \cdot, L) \cdot \sigma\!\big(\alpha \cdot \text{clamp}(z_g, \cdot, L)\big) \odot (\text{clamp}(z_l, -L, L) + 1)
$$

$$
y^e = W_2^e \text{SwiGLU}(z) + b_2^e
$$

<span style="font-size: 14px;">where $\alpha = 1.702$, $L$ is `swiglu_limit` (7.0 default), and **the linear branch gets a `+1` bias**, an idiosyncrasy of GPT-OSS that is easy to miss. The clamp prevents the GLU pre-activation from exploding under fp8 / mxfp4 weights and stabilizes training.</span>

<span style="font-size: 14px;">The block output for token $i$ is then:</span>

$$
\text{moe}_i = \sum_{j=1}^{k} w_{ij} \cdot y^{\text{top\_idx}_{ij}}
$$

<span style="font-size: 14px;">followed by the second residual: $\text{output}_i = x_i + \text{moe}_i$.</span>

<span style="font-size: 14px;">Sparse MoE buys parameter count without paying the matching FLOP cost: only $k$ of $E$ experts run per token. GPT-OSS-120B has 128 experts but activates only 4 per token, so per-token FFN compute matches a dense model of $I = 4 \cdot 2880$ while parameter count is $32\times$ larger.</span>

---

## <span style="font-size: 16px;">Dataflow</span>

<span style="font-size: 14px;">In words, the dataflow through one block is:</span>

* <span style="font-size: 14px;">$x_0 = \text{input}$</span>
* <span style="font-size: 14px;">$h_1 = \text{RMSNorm}(x_0, \gamma_1)$</span>
* <span style="font-size: 14px;">$q, k, v = \text{slice}(h_1 W_{qkv}^\top + b_{qkv})$ then reshape into GQA layout</span>
* <span style="font-size: 14px;">$q, k = \text{RoPE}(q, \cos, \sin), \text{RoPE}(k, \cos, \sin)$</span>
* <span style="font-size: 14px;">$a = \text{SDPA}(q, k, v, \text{sinks}, \text{sw})$ with $\text{sw} = $ window on even layers, $0$ on odd</span>
* <span style="font-size: 14px;">$x_1 = x_0 + a W_o^\top + b_o$  (first residual)</span>
* <span style="font-size: 14px;">$h_2 = \text{RMSNorm}(x_1, \gamma_2)$</span>
* <span style="font-size: 14px;">$\ell = h_2 W_g^\top + b_g, \quad (\text{idx}, w) = \text{top\_k\_softmax}(\ell, k)$</span>
* <span style="font-size: 14px;">$m_i = \sum_j w_{ij} \cdot \text{Expert}_{\text{idx}_{ij}}(h_2[i])$ where each expert is a clamped SwiGLU MLP</span>
* <span style="font-size: 14px;">$x_2 = x_1 + m$  (second residual)</span>
* <span style="font-size: 14px;">$\text{output} = x_2$</span>

---

## <span style="font-size: 16px;">Comparison with LLaMA and DeepSeek-V3 Blocks</span>

<span style="font-size: 14px;">Three modern open-weight blocks side by side:</span>

* <span style="font-size: 14px;">**LLaMA block.** Pre-RMSNorm, GQA attention with full causal mask in every layer, RoPE on Q and K, dense SwiGLU FFN. No sinks, no sliding window, no MoE. The simplest of the three.</span>
* <span style="font-size: 14px;">**DeepSeek-V3 block.** Pre-RMSNorm, Multi-Latent Attention (MLA) with low-rank compressed KV and decoupled RoPE on a separate small head. Dense SwiGLU for the first few layers (the "dense prefix"), MoE with a shared expert plus routed experts for the rest. No sliding window, no sinks.</span>
* <span style="font-size: 14px;">**GPT-OSS block.** Pre-RMSNorm, GQA attention with sinks and alternating sliding window, RoPE with YaRN scaling, clamped SwiGLU MoE with top-4 routing over 128 experts and no shared expert. The most aggressive sparsity of the three plus the most ornate attention.</span>

<span style="font-size: 14px;">All three follow the same pre-norm + residual skeleton; the differences are in what fills the two sublayer slots.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $n=2$, $H=4$, $H_{kv}=1$, $q_{\text{mult}}=2$, $d=2$, $E=2$, $k=1$, $I=4$, $\text{sw}=1$. With small concrete weights and $x = [[0.27, -0.46, -0.92, -0.97], [0.63, 0.83, 0.21, 0.46]]$:</span>

* <span style="font-size: 14px;">**Layer 0 (even, sliding window=1 active):** token 1 attends only to itself (positions older than 1 step are masked). Output row 1 differs from the no-window run.</span>
* <span style="font-size: 14px;">**Layer 1 (odd, sliding window disabled):** token 1 attends to both positions 0 and 1. Output row 1 is different from the even-layer case.</span>
* <span style="font-size: 14px;">**Row 0 in both layers is identical:** the first token has only itself in its receptive field under both even and odd configurations, so the windowing flag has no effect there.</span>

<span style="font-size: 14px;">This identity-on-token-0 / divergence-on-later-tokens pattern is the signature of correct alternation handling.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Alternation parity flipped.** Applying the sliding window on odd layers (or every layer, or no layer) is the most common bug. It silently changes the receptive-field schedule and produces wrong outputs on at least half the layers. Always test even/odd pairs with the same input.</span>
* <span style="font-size: 14px;">**Forgetting the second residual.** The MoE sublayer must add its output back to the post-attention residual stream. Returning $\text{moe}(t_2)$ alone discards the attention contribution and produces a much worse model.</span>
* <span style="font-size: 14px;">**Post-norm by accident.** Writing $\text{LN}(x + \text{Sublayer}(x))$ instead of $x + \text{Sublayer}(\text{LN}(x))$ is the original Vaswani style but does not match GPT-OSS, and it changes the residual magnitudes throughout the stack.</span>
* <span style="font-size: 14px;">**Norm scope.** RMSNorm normalizes the feature axis $H$, not the token axis. Normalizing across tokens breaks positional information.</span>
* <span style="font-size: 14px;">**RoPE on Q only (or K only).** Both Q and K must be rotated; rotating only one half breaks the relative-position property because the dot product $q_i^\top k_j$ no longer depends only on $i - j$.</span>
* <span style="font-size: 14px;">**Skipping GQA expansion.** K and V have $H_{kv}$ heads while Q has $H_q = q_{\text{mult}} \cdot H_{kv}$ heads. Forgetting to broadcast K and V along the $q_{\text{mult}}$ axis silently zeros out most attention contributions or causes shape errors.</span>
* <span style="font-size: 14px;">**Dropping the sink from the softmax denominator.** Sinks must participate in the normalization but then be dropped from the value-side sum. Including them in the value sum (or omitting them from the softmax) gives a completely different distribution.</span>
* <span style="font-size: 14px;">**Wrong SwiGLU formula.** The clamped SwiGLU here uses $\alpha = 1.702$, asymmetric clamps (none-to-$L$ on the GLU side, $-L$-to-$L$ on the linear side), and a **+1** added to the linear branch. Missing the $\alpha$ or the +1 produces a related but incorrect activation.</span>
* <span style="font-size: 14px;">**Router top-$k$ normalization.** The top-$k$ gating weights must be re-softmaxed over the chosen experts only. Using the raw logits (or softmax over all $E$ then slicing) gives weights that do not sum to 1 and biases the mixture.</span>
* <span style="font-size: 14px;">**Interleaved vs concatenated SwiGLU split.** GPT-OSS uses $x[\dots, ::2]$ and $x[\dots, 1::2]$ (interleaved), not concatenated halves.</span>

---
