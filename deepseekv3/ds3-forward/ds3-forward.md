# <span style="font-size: 20px;">Full DeepSeek V3 Forward Pass</span>

<span style="font-size: 14px;">The full forward pass of DeepSeek V3 maps token IDs to logits over the vocabulary through four stages: token embedding, a stack of transformer blocks (dense prefix + MoE), a final RMSNorm, and a linear LM head. Key deviations from a vanilla transformer: no positional embedding at input, untied embedding and head weights, a dense-block prefix before MoE blocks, and no softmax at the output.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The forward pass is the function $f: \mathbb{Z}^{T} \to \mathbb{R}^{T \times V}$, where $T$ is the sequence length and $V$ is the vocabulary size. It produces unnormalized log-probabilities (logits) for every position over every vocabulary entry. During training these feed into cross-entropy loss; during inference, decoding strategies operate on the raw logits.</span>

<span style="font-size: 14px;">A key property: no positional embedding is added at the input stage. Positional information enters through RoPE applied inside each attention layer, so the embedding layer produces position-agnostic representations and position awareness is injected at every attention computation.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Stage 1 -- Token Embedding:**</span>

$$
H^{(0)} = \text{Embed}(x) \in \mathbb{R}^{T \times d}
$$

<span style="font-size: 14px;">where $x = [x_1, x_2, \ldots, x_T]$ is the input token ID sequence and $d$ is the hidden dimension. Each token ID indexes a row of the embedding matrix $W_e \in \mathbb{R}^{V \times d}$.</span>

<span style="font-size: 14px;">**Stage 2 -- Transformer Blocks:**</span>

$$
H^{(\ell)} = \text{Block}_\ell(H^{(\ell-1)}), \quad \ell = 1, 2, \ldots, N
$$

<span style="font-size: 14px;">Each block applies MLA attention followed by a feed-forward network, each with a residual connection and pre-RMSNorm. The first $N_{\text{dense}}$ blocks use a dense FFN; the remaining blocks use MoE.</span>

<span style="font-size: 14px;">**Stage 3 -- Final RMSNorm:**</span>

$$
H^{(\text{final})} = \text{RMSNorm}(H^{(N)})
$$

<span style="font-size: 14px;">**Stage 4 -- LM Head:**</span>

$$
\text{logits} = H^{(\text{final})} \cdot W_h^{\top} \in \mathbb{R}^{T \times V}
$$

<span style="font-size: 14px;">where $W_h \in \mathbb{R}^{V \times d}$ is the head weight matrix, separate from $W_e$. No bias, no softmax.</span>

---

## <span style="font-size: 16px;">Stage 1: Token Embedding</span>

<span style="font-size: 14px;">The embedding layer is a lookup table. Given token ID $x_t \in \{0, 1, \ldots, V-1\}$, the embedding is:</span>

$$
h_t^{(0)} = W_e[x_t, :] \in \mathbb{R}^d
$$

<span style="font-size: 14px;">This is a row selection (index operation), not a matrix multiplication. For $T$ tokens, the output is $H^{(0)} \in \mathbb{R}^{T \times d}$.</span>

<span style="font-size: 14px;">Critical design choices:</span>

* <span style="font-size: 14px;">**No positional embedding.** Unlike the original transformer, DeepSeek V3 does not inject position information here. The output is purely semantic -- identical tokens at different positions produce identical vectors.</span>
* <span style="font-size: 14px;">**No scaling factor.** Some architectures multiply embeddings by $\sqrt{d}$ to match positional encodings. Since there are none, this scaling is omitted.</span>
* <span style="font-size: 14px;">**Position enters later via RoPE.** RoPE rotates query and key vectors inside each attention layer based on position. Position awareness is refreshed every layer rather than being a one-time additive signal.</span>

<span style="font-size: 14px;">With vocabulary 129,280 and hidden dimension 7,168, $W_e$ contains approximately 926 million parameters.</span>

---

## <span style="font-size: 16px;">Stage 2: Transformer Blocks</span>

<span style="font-size: 14px;">The model stacks $N$ transformer blocks, each transforming $H^{(\ell-1)} \in \mathbb{R}^{T \times d}$ into $H^{(\ell)} \in \mathbb{R}^{T \times d}$. The blocks split into two regimes based on the feed-forward network used.</span>

### <span style="font-size: 14px;">Block Structure (All Blocks)</span>

<span style="font-size: 14px;">Every block follows this pattern:</span>

$$
H_{\text{mid}}^{(\ell)} = H^{(\ell-1)} + \text{MLA}(\text{RMSNorm}(H^{(\ell-1)}))
$$

$$
H^{(\ell)} = H_{\text{mid}}^{(\ell)} + \text{FFN}_\ell(\text{RMSNorm}(H_{\text{mid}}^{(\ell)}))
$$

<span style="font-size: 14px;">Residual connections ensure gradient flow through the full depth. RMSNorm before each sub-layer (pre-norm) stabilizes training for very deep models.</span>

### <span style="font-size: 14px;">Dense Prefix Blocks ($\ell = 1, \ldots, N_{\text{dense}}$)</span>

<span style="font-size: 14px;">The first $N_{\text{dense}}$ blocks use a standard dense feed-forward network:</span>

$$
\text{FFN}_{\text{dense}}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot (W_{\text{up}} \cdot x))
$$

<span style="font-size: 14px;">This is a SwiGLU-style FFN: gate and up projections expand the dimension, SiLU activates the gate branch, elementwise multiplication combines them, and the down projection maps back to $d$.</span>

<span style="font-size: 14px;">The dense prefix exists for stability. Early layers build foundational representations before routing to experts is meaningful. Routing on insufficiently processed embeddings would be near-random.</span>

### <span style="font-size: 14px;">MoE Blocks ($\ell = N_{\text{dense}}+1, \ldots, N$)</span>

<span style="font-size: 14px;">The remaining blocks replace the dense FFN with a Mixture-of-Experts FFN. Each MoE block has a router selecting $K_r$ experts from $N_e$ routed experts, plus a shared expert that processes every token:</span>

$$
\text{FFN}_{\text{MoE}}(x) = \text{FFN}_{\text{shared}}(x) + \sum_{i \in \text{TopK}(s)} g_i \cdot \text{FFN}_i(x)
$$

<span style="font-size: 14px;">where $s$ is the router score vector and $g_i$ are gating weights. Each expert FFN has the same SwiGLU structure but smaller intermediate dimensions. The shared expert ensures baseline capacity; routed experts provide specialization. Although MoE layers contribute to the 671B total, only a subset is active per token (roughly 37B active parameters).</span>

### <span style="font-size: 14px;">Sequential Execution</span>

<span style="font-size: 14px;">Blocks execute strictly in order. Each block's input depends on the previous block's output; no skip connection bypasses entire blocks. The hidden dimension $d$ remains constant.</span>

---

## <span style="font-size: 16px;">Stage 3: Final RMSNorm</span>

<span style="font-size: 14px;">After the last transformer block, a final RMSNorm is applied:</span>

$$
H^{(\text{final})} = \text{RMSNorm}(H^{(N)})
$$

<span style="font-size: 14px;">RMSNorm normalizes each token's hidden vector independently:</span>

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

<span style="font-size: 14px;">where $\gamma \in \mathbb{R}^d$ is a learnable scale vector (initialized to ones). Unlike LayerNorm, RMSNorm does not subtract the mean -- it only divides by the root-mean-square magnitude.</span>

<span style="font-size: 14px;">This standardizes hidden-state magnitudes before the projection to vocabulary size. Without it, scale varies wildly across positions, making logits unreliable. The pre-norm blocks already apply RMSNorm before each sub-layer, but the last block's output is an un-normalized residual sum -- the final RMSNorm ensures the LM head input is well-scaled.</span>

---

## <span style="font-size: 16px;">Stage 4: Language Model Head</span>

<span style="font-size: 14px;">The LM head is a single linear projection from hidden dimension to vocabulary size:</span>

$$
\text{logits} = H^{(\text{final})} \cdot W_h^{\top} \in \mathbb{R}^{T \times V}
$$

<span style="font-size: 14px;">Key design decisions:</span>

* <span style="font-size: 14px;">**Untied weights.** $W_h \in \mathbb{R}^{V \times d}$ is completely separate from $W_e$. Many models tie these to save parameters, but DeepSeek V3 keeps them independent so the embedding space (input semantics) can differ from the output space (next-token prediction).</span>
* <span style="font-size: 14px;">**No bias.** The projection is $W_h^{\top} x$ with no additive bias.</span>
* <span style="font-size: 14px;">**No softmax.** Output is raw logits. Softmax is applied externally: fused into cross-entropy during training, optionally applied during inference depending on the decoding strategy.</span>

<span style="font-size: 14px;">Output shape is $T \times V$. For autoregressive modeling, logit at position $t$ predicts position $t+1$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">DeepSeek V3 was introduced in "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024). The architecture specifications:</span>

* <span style="font-size: 14px;">**Total layers ($N$):** 61 transformer blocks.</span>
* <span style="font-size: 14px;">**Dense prefix ($N_{\text{dense}}$):** First 3 blocks use dense FFNs; blocks 4 through 61 use MoE.</span>
* <span style="font-size: 14px;">**Hidden dimension ($d$):** 7,168.</span>
* <span style="font-size: 14px;">**Vocabulary size ($V$):** 129,280 tokens.</span>
* <span style="font-size: 14px;">**Total parameters:** 671 billion.</span>
* <span style="font-size: 14px;">**Active parameters per token:** Approximately 37 billion. Only $K_r = 8$ of 256 routed experts are selected per token, plus 1 shared expert per MoE layer.</span>
* <span style="font-size: 14px;">**Attention:** Multi-head Latent Attention (MLA) with low-rank key-value compression, reducing KV cache size without sacrificing quality.</span>
* <span style="font-size: 14px;">**Position encoding:** RoPE applied within attention, not at the embedding layer. DeepSeek V3 uses a decoupled RoPE variant where rotary embeddings are applied to separate rope-specific query and key vectors rather than the compressed latent representations.</span>

<span style="font-size: 14px;">Training used auxiliary-loss-free load balancing and Multi-Token Prediction (MTP). During standard inference, only the four-stage pipeline above is used (MTP heads are not active). The model was trained on 14.8 trillion tokens, with MoE keeping per-token compute comparable to a dense 37B model.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified model with $d = 4$, $V = 6$, $N = 2$ blocks (1 dense, 1 MoE), and input tokens $x = [2, 0, 5]$.</span>

### <span style="font-size: 14px;">Step 1: Token Embedding</span>

<span style="font-size: 14px;">Embedding matrix $W_e \in \mathbb{R}^{6 \times 4}$:</span>

$$
W_e = \begin{bmatrix} 0.3 & -0.1 & 0.5 & 0.2 \\ 0.1 & 0.4 & -0.3 & 0.6 \\ -0.2 & 0.7 & 0.1 & -0.4 \\ 0.5 & 0.2 & -0.6 & 0.3 \\ -0.1 & -0.3 & 0.4 & 0.8 \\ 0.6 & 0.0 & -0.2 & 0.1 \end{bmatrix}
$$

<span style="font-size: 14px;">Look up rows for IDs $[2, 0, 5]$:</span>

* <span style="font-size: 14px;">Token 2: $h_1^{(0)} = [-0.2, 0.7, 0.1, -0.4]$</span>
* <span style="font-size: 14px;">Token 0: $h_2^{(0)} = [0.3, -0.1, 0.5, 0.2]$</span>
* <span style="font-size: 14px;">Token 5: $h_3^{(0)} = [0.6, 0.0, -0.2, 0.1]$</span>

<span style="font-size: 14px;">No positional embedding is added. These vectors enter the first block directly.</span>

### <span style="font-size: 14px;">Step 2: Transformer Blocks (Simplified)</span>

<span style="font-size: 14px;">After the dense block (block 1) and MoE block (block 2), suppose the output hidden states are:</span>

* <span style="font-size: 14px;">$h_1^{(2)} = [1.2, -0.5, 0.8, -0.3]$</span>
* <span style="font-size: 14px;">$h_2^{(2)} = [-0.4, 1.1, 0.2, 0.7]$</span>
* <span style="font-size: 14px;">$h_3^{(2)} = [0.9, 0.3, -0.6, 1.0]$</span>

### <span style="font-size: 14px;">Step 3: Final RMSNorm</span>

<span style="font-size: 14px;">Apply RMSNorm with $\gamma = [1, 1, 1, 1]$. For token 1:</span>

$$
\text{RMS}(h_1^{(2)}) = \sqrt{\frac{1.44 + 0.25 + 0.64 + 0.09}{4}} = \sqrt{0.605} \approx 0.7778
$$

$$
h_1^{(\text{final})} = \frac{[1.2, -0.5, 0.8, -0.3]}{0.7778} \approx [1.543, -0.643, 1.029, -0.386]
$$

<span style="font-size: 14px;">For token 2:</span>

$$
\text{RMS}(h_2^{(2)}) = \sqrt{\frac{0.16 + 1.21 + 0.04 + 0.49}{4}} = \sqrt{0.475} \approx 0.6892
$$

$$
h_2^{(\text{final})} \approx [-0.580, 1.596, 0.290, 1.016]
$$

<span style="font-size: 14px;">For token 3:</span>

$$
\text{RMS}(h_3^{(2)}) = \sqrt{\frac{0.81 + 0.09 + 0.36 + 1.00}{4}} = \sqrt{0.565} \approx 0.7517
$$

$$
h_3^{(\text{final})} \approx [1.197, 0.399, -0.798, 1.331]
$$

### <span style="font-size: 14px;">Step 4: LM Head</span>

<span style="font-size: 14px;">Head weight matrix $W_h \in \mathbb{R}^{6 \times 4}$ (separate from $W_e$):</span>

$$
W_h = \begin{bmatrix} 0.4 & -0.2 & 0.3 & 0.1 \\ -0.1 & 0.5 & 0.2 & -0.3 \\ 0.2 & 0.1 & -0.4 & 0.6 \\ 0.3 & -0.5 & 0.1 & 0.4 \\ -0.3 & 0.2 & 0.5 & -0.1 \\ 0.1 & 0.3 & -0.1 & 0.2 \end{bmatrix}
$$

<span style="font-size: 14px;">Logits for token 1: $\text{logit}_j = h_1^{(\text{final})} \cdot W_h[j, :]$</span>

* <span style="font-size: 14px;">Vocab 0: $(1.543)(0.4) + (-0.643)(-0.2) + (1.029)(0.3) + (-0.386)(0.1) = 0.617 + 0.129 + 0.309 - 0.039 = 1.016$</span>
* <span style="font-size: 14px;">Vocab 1: $(-0.154) + (-0.322) + (0.206) + (0.116) = -0.154$</span>
* <span style="font-size: 14px;">Vocab 2: $(0.309) + (-0.064) + (-0.412) + (-0.232) = -0.399$</span>

<span style="font-size: 14px;">The full logit vector has 6 entries. The highest logit (vocab 0, at 1.016) is the model's top prediction for the next token after position 1. No softmax is applied -- these raw values are the forward pass output.</span>

<span style="font-size: 14px;">Notice $W_h \neq W_e$. If weights were tied, logits would use the same matrix that produced embeddings. Untied weights let the model learn a different projection for output prediction.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Adding Positional Embeddings at the Input</span>

<span style="font-size: 14px;">Adding sinusoidal or learned positional embeddings to token embeddings follows the original transformer recipe, but DeepSeek V3 intentionally omits this. Position enters exclusively through RoPE in attention. Adding positional embeddings at input means the model gets position info twice (embedding + RoPE), corrupting representations.</span>

### <span style="font-size: 14px;">Tying Embedding and Head Weights</span>

<span style="font-size: 14px;">Many implementations tie $W_h = W_e$ by default. DeepSeek V3 uses untied weights. Tying forces the input and output spaces to be identical, constraining the model. Sharing weights means the logit computation uses the wrong matrix. Always initialize $W_e$ and $W_h$ as independent parameters.</span>

### <span style="font-size: 14px;">Applying Softmax in the Forward Pass</span>

<span style="font-size: 14px;">The forward pass returns raw logits, not probabilities. Applying softmax inside the forward pass and then passing to cross-entropy loss (which expects logits) effectively applies softmax twice, producing near-uniform predictions and destroying gradients.</span>

### <span style="font-size: 14px;">Wrong Block Ordering (MoE Before Dense)</span>

<span style="font-size: 14px;">The dense prefix blocks must come first (blocks 1 through $N_{\text{dense}}$), followed by MoE blocks. Reversing this order forces the router to make expert selection decisions on raw, minimally processed embeddings. Router scores would be effectively random, leading to poor expert utilization and collapsed training. The dense prefix builds foundational representations the router can meaningfully differentiate.</span>

### <span style="font-size: 14px;">Forgetting the Final RMSNorm</span>

<span style="font-size: 14px;">It is easy to go directly from the last transformer block to the LM head, skipping the final RMSNorm. The last block outputs $H^{(N)}$, an un-normalized residual sum. Without normalization, hidden-state magnitudes vary wildly, causing logits to have inconsistent scales. This makes training unstable and inference unreliable. The final RMSNorm is not optional.</span>

### <span style="font-size: 14px;">Scaling Embeddings by $\sqrt{d}$</span>

<span style="font-size: 14px;">The original transformer multiplies embeddings by $\sqrt{d_{\text{model}}}$ to match the scale of positional encodings. Since DeepSeek V3 has no positional encodings at the embedding stage, this scaling is not applied. Including it inflates initial hidden states, disrupting the initialization and norm scales throughout the network.</span>