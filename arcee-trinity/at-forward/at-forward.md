# <span style="font-size: 20px;">Full Arcee Trinity Forward Pass</span>

<span style="font-size: 14px;">The full Arcee Trinity forward pass transforms a sequence of token IDs into logits over the vocabulary. It chains four stages: token embedding, a stack of Transformer blocks, a final RMSNorm, and a linear LM head. The output is raw logits (no softmax).</span>

<span style="font-size: 14px;">This is the capstone of the Arcee Trinity architecture. Every component studied in isolation -- gated attention, sandwich normalization, depth scaling, coarse MoE, interleaved RoPE/NoPE -- is assembled here into a single coherent model.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The forward pass is a four-stage pipeline:</span>

* <span style="font-size: 14px;">**Stage 1 -- Token Embedding:** Each token ID is mapped to a dense vector via a learned embedding table. No positional embedding is added here because Arcee Trinity handles position inside each block via RoPE/NoPE.</span>
* <span style="font-size: 14px;">**Stage 2 -- N Transformer Blocks:** The embedded sequence passes through $N$ blocks sequentially. Each block applies gated attention followed by coarse MoE FFN, both wrapped in sandwich normalization with depth scaling.</span>
* <span style="font-size: 14px;">**Stage 3 -- Final RMSNorm:** Stabilizes the hidden states before projection so the LM head receives representations at a consistent scale.</span>
* <span style="font-size: 14px;">**Stage 4 -- LM Head:** A linear projection from hidden dimension to vocabulary size produces logits. No softmax is applied. Embedding and LM head weights are untied.</span>

<span style="font-size: 14px;">The input is a tensor of token IDs with shape $(B, T)$ where $B$ is batch size and $T$ is sequence length. The output is a logits tensor of shape $(B, T, V)$ where $V$ is vocabulary size.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Stage 1 -- Embedding lookup:**</span>

$$
h^{(0)} = W_{\text{embed}}[\text{token\_ids}]
$$

<span style="font-size: 14px;">where $W_{\text{embed}} \in \mathbb{R}^{V \times d}$, $V$ is vocabulary size, $d$ is hidden dimension, and $h^{(0)} \in \mathbb{R}^{B \times T \times d}$.</span>

<span style="font-size: 14px;">**Stage 2 -- Each Transformer block $i$ (for $i = 0, 1, \ldots, N-1$):**</span>

<span style="font-size: 14px;">Attention sub-layer with sandwich norm and depth scaling:</span>

$$
h^{(i+0.5)} = h^{(i)} + \frac{\text{RMSNorm}\bigl(\text{gate} \odot \text{Attn}(\text{RMSNorm}(h^{(i)}))\bigr)}{\sqrt{2i + 1}}
$$

<span style="font-size: 14px;">FFN sub-layer with sandwich norm and depth scaling:</span>

$$
h^{(i+1)} = h^{(i+0.5)} + \frac{\text{RMSNorm}\bigl(\text{CoarseMoE}(\text{RMSNorm}(h^{(i+0.5)}))\bigr)}{\sqrt{2i + 1}}
$$

<span style="font-size: 14px;">**Stage 3 -- Final RMSNorm:**</span>

$$
h_f = \text{RMSNorm}(h^{(N)}) = \frac{h^{(N)} \cdot \gamma_f}{\sqrt{\text{mean}((h^{(N)})^2) + \epsilon}}
$$

<span style="font-size: 14px;">**Stage 4 -- LM head projection:**</span>

$$
\text{logits} = h_f \cdot W_{\text{head}}^T
$$

<span style="font-size: 14px;">where $W_{\text{head}} \in \mathbb{R}^{V \times d}$ is the LM head weight matrix, separate from $W_{\text{embed}}$.</span>

---

## <span style="font-size: 16px;">Stage 1: Token Embedding</span>

<span style="font-size: 14px;">The embedding layer is a lookup table $W_{\text{embed}} \in \mathbb{R}^{V \times d}$. Given a token ID $t$, the embedding is the $t$-th row. For a sequence of $T$ tokens, this produces a tensor of shape $(B, T, d)$.</span>

<span style="font-size: 14px;">Critically, Arcee Trinity does **not** add positional embedding at this stage. Standard Transformers (e.g., GPT-2) add sinusoidal or learned positional embeddings before the first block. Arcee Trinity instead injects position inside each block through RoPE, which rotates Q and K during attention. The embedding stage is purely semantic -- tokens map to meaning vectors with no position information.</span>

<span style="font-size: 14px;">Why this design?</span>

* <span style="font-size: 14px;">**RoPE is relative, not absolute:** It encodes relative positions through rotation angles on Q and K. No additive embedding at the input is needed.</span>
* <span style="font-size: 14px;">**Per-layer flexibility:** Different layers can use different position strategies. Arcee Trinity exploits this with its interleaved RoPE/NoPE pattern.</span>
* <span style="font-size: 14px;">**Separation of concerns:** The embedding layer handles vocabulary mapping; each block handles positional encoding independently.</span>

---

## <span style="font-size: 16px;">Stage 2: Transformer Blocks</span>

<span style="font-size: 14px;">The core of the model is a stack of $N$ blocks applied sequentially. Each block has the same structural template but its own learned parameters. Block $i$ transforms $h^{(i)}$ into $h^{(i+1)}$ through two sub-layers.</span>

<span style="font-size: 14px;">**Processing within each block:**</span>

* <span style="font-size: 14px;">**Pre-norm (attention):** $\hat{x} = \text{RMSNorm}(h^{(i)}, \gamma_{\text{attn\_pre}})$</span>
* <span style="font-size: 14px;">**Gated attention:** Compute Q, K, V from $\hat{x}$. Apply RoPE to Q, K (or skip for NoPE layers). Compute causal self-attention. Apply sigmoid gate: $\text{gated} = \sigma(\hat{x} W_g^T) \odot (\text{attn} \cdot W_o^T)$.</span>
* <span style="font-size: 14px;">**Post-norm + depth scaling + residual:** $h^{(i+0.5)} = h^{(i)} + \text{RMSNorm}(\text{gated}, \gamma_{\text{attn\_post}}) / \sqrt{2i+1}$</span>
* <span style="font-size: 14px;">**Pre-norm (FFN):** $\hat{h} = \text{RMSNorm}(h^{(i+0.5)}, \gamma_{\text{ffn\_pre}})$</span>
* <span style="font-size: 14px;">**Coarse MoE FFN:** Sigmoid router selects top-$k$ expert groups. Each group computes SwiGLU. Outputs are weighted-summed with renormalized router weights.</span>
* <span style="font-size: 14px;">**Post-norm + depth scaling + residual:** $h^{(i+1)} = h^{(i+0.5)} + \text{RMSNorm}(\text{moe\_out}, \gamma_{\text{ffn\_post}}) / \sqrt{2i+1}$</span>

<span style="font-size: 14px;">Key architectural details across the block stack:</span>

* <span style="font-size: 14px;">**Sandwich normalization:** Every sub-layer has RMSNorm both before and after. This prevents activation magnitudes from growing unchecked within the sub-layer.</span>
* <span style="font-size: 14px;">**Depth scaling by $1/\sqrt{2i+1}$:** Early blocks contribute at full scale; deeper blocks contribute less. This prevents residual stream magnitude from growing as $O(\sqrt{N})$.</span>
* <span style="font-size: 14px;">**Interleaved RoPE/NoPE:** Most blocks apply RoPE. A subset (e.g., every 4th layer) skip positional encoding entirely (NoPE), forcing position-independent representations that generalize to unseen sequence lengths.</span>
* <span style="font-size: 14px;">**Sigmoid gating on attention:** A per-dimension sigmoid gate controls how much of the attention output flows forward, providing finer control than a simple linear output projection.</span>

---

## <span style="font-size: 16px;">Stage 3: Final RMSNorm</span>

<span style="font-size: 14px;">After the last Transformer block, $h^{(N)}$ passes through a final RMSNorm:</span>

$$
h_f = \frac{h^{(N)} \cdot \gamma_f}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} (h^{(N)}_j)^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\gamma_f \in \mathbb{R}^d$ is a learned scale vector and $\epsilon$ (typically $10^{-6}$) prevents division by zero.</span>

<span style="font-size: 14px;">Why is this necessary?</span>

* <span style="font-size: 14px;">**Scale normalization:** The residual stream accumulates contributions from $N$ blocks. The final RMSNorm ensures consistent scale regardless of depth or input.</span>
* <span style="font-size: 14px;">**Stable logit magnitudes:** Without normalization, hidden state norms vary wildly, making logits unstable, training volatile, and temperature-based sampling unreliable.</span>
* <span style="font-size: 14px;">**Universal practice:** All modern LLMs (LLaMA, Mistral, Qwen) include a final norm before the LM head. Omitting it degrades both training stability and generation quality.</span>

<span style="font-size: 14px;">This final RMSNorm is separate from the sandwich norms inside each block. It has its own learned $\gamma_f$ parameter vector.</span>

---

## <span style="font-size: 16px;">Stage 4: LM Head</span>

<span style="font-size: 14px;">The LM head is a single linear layer projecting from $d$ to $V$:</span>

$$
\text{logits} = h_f \cdot W_{\text{head}}^T, \quad W_{\text{head}} \in \mathbb{R}^{V \times d}
$$

<span style="font-size: 14px;">The output logits have shape $(B, T, V)$. No softmax is applied -- the model outputs raw logits.</span>

<span style="font-size: 14px;">**Why no softmax?** Cross-entropy loss applies log-softmax internally for numerical stability. Applying softmax in the model would cause double-softmax with near-zero gradients. Raw logits also allow downstream temperature scaling and top-k/top-p filtering.</span>

<span style="font-size: 14px;">**Why untied weights?** Some architectures (e.g., GPT-2) tie the embedding and LM head ($W_{\text{head}} = W_{\text{embed}}$). Arcee Trinity uses separate matrices because:</span>

* <span style="font-size: 14px;">**Different roles:** The embedding maps tokens into a space for contextual processing. The LM head maps hidden states into a space for next-token prediction. These are different functions.</span>
* <span style="font-size: 14px;">**More capacity:** Untying adds $V \times d$ parameters but gives more expressiveness. For large models this cost is marginal.</span>
* <span style="font-size: 14px;">**No gradient conflict:** With tied weights, gradients from the loss and from the embedding lookup compete on the same matrix. Untying eliminates this.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Arcee Trinity is a decoder-only Transformer. The full model configuration:</span>

* <span style="font-size: 14px;">**Hidden dimension ($d$):** 4096</span>
* <span style="font-size: 14px;">**Number of layers ($N$):** 36</span>
* <span style="font-size: 14px;">**Attention heads:** 32 query heads, 8 KV heads (Grouped Query Attention)</span>
* <span style="font-size: 14px;">**Head dimension:** $4096 / 32 = 128$</span>
* <span style="font-size: 14px;">**Intermediate dimension:** 14336 (FFN expansion ~3.5x)</span>
* <span style="font-size: 14px;">**Vocabulary size ($V$):** 128,256</span>
* <span style="font-size: 14px;">**Context length:** 131,072 tokens (128K)</span>

<span style="font-size: 14px;">The embedding contributes ~525M parameters ($128256 \times 4096$), and the untied LM head adds another ~525M. Each of the 36 blocks contains attention projections ($W_q, W_k, W_v, W_o, W_g$), four RMSNorm layers, and coarse MoE FFN weights. The final RMSNorm adds just 4096 parameters.</span>

<span style="font-size: 14px;">The interleaved RoPE/NoPE pattern means roughly 27 layers use RoPE and 9 use NoPE (every 4th layer). This is fixed at architecture definition time. Each block's FFN is a sigmoid-routed coarse MoE selecting expert groups that share the same weight matrices.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace 3 token IDs through a simplified model with $d = 4$, $V = 8$, $N = 2$ blocks.</span>

<span style="font-size: 14px;">**Input token IDs:** $[2, 5, 7]$ (sequence length $T = 3$).</span>

<span style="font-size: 14px;">**Stage 1 -- Embedding lookup.** Look up rows 2, 5, 7 from $W_{\text{embed}} \in \mathbb{R}^{8 \times 4}$:</span>

* <span style="font-size: 14px;">**Token 2:** $h_0^{(0)} = [0.3, -0.5, 0.8, 0.1]$</span>
* <span style="font-size: 14px;">**Token 5:** $h_1^{(0)} = [-0.2, 0.7, 0.4, -0.6]$</span>
* <span style="font-size: 14px;">**Token 7:** $h_2^{(0)} = [0.9, 0.1, -0.3, 0.5]$</span>

<span style="font-size: 14px;">No positional embedding added. These raw semantic vectors enter the block stack.</span>

<span style="font-size: 14px;">**Stage 2 -- Block 0 ($i = 0$, depth scale $= 1/\sqrt{1} = 1.0$).** Focus on token 2:</span>

<span style="font-size: 14px;">Pre-norm: $\text{RMS} = \sqrt{(0.09 + 0.25 + 0.64 + 0.01)/4} = \sqrt{0.2475} \approx 0.4975$. With $\gamma = [1,1,1,1]$: $\hat{x} = [0.603, -1.005, 1.608, 0.201]$.</span>

<span style="font-size: 14px;">After gated attention (Q/K/V projections, causal attention, sigmoid gating), suppose the gated output is $[0.12, -0.08, 0.15, -0.04]$. Post-norm gives $[0.53, -0.35, 0.66, -0.18]$. With depth scale $1.0$: $h^{(0.5)} = [0.3 + 0.53,\; -0.5 - 0.35,\; 0.8 + 0.66,\; 0.1 - 0.18] = [0.83, -0.85, 1.46, -0.08]$.</span>

<span style="font-size: 14px;">After coarse MoE FFN with the same sandwich norm pattern, suppose block 0 output: $h^{(1)} = [1.02, -0.64, 1.31, 0.15]$.</span>

<span style="font-size: 14px;">**Stage 2 -- Block 1 ($i = 1$, depth scale $= 1/\sqrt{3} \approx 0.577$).** Same process but sub-layer contributions are attenuated by ~42%. After block 1: $h^{(2)} = [0.95, -0.48, 1.22, 0.28]$.</span>

<span style="font-size: 14px;">**Stage 3 -- Final RMSNorm.** For $h^{(2)} = [0.95, -0.48, 1.22, 0.28]$:</span>

$$
\text{RMS} = \sqrt{\frac{0.9025 + 0.2304 + 1.4884 + 0.0784}{4}} = \sqrt{0.6749} \approx 0.8216
$$

<span style="font-size: 14px;">With $\gamma_f = [1,1,1,1]$: $h_f = [1.156, -0.584, 1.485, 0.341]$.</span>

<span style="font-size: 14px;">**Stage 4 -- LM Head.** Project through $W_{\text{head}} \in \mathbb{R}^{8 \times 4}$ (untied from embedding). Each row dot-products with $h_f$:</span>

* <span style="font-size: 14px;">**Logit 0:** $[0.2, 0.1, -0.3, 0.5] \cdot h_f = 0.231 - 0.058 - 0.446 + 0.171 = -0.102$</span>
* <span style="font-size: 14px;">**Logit 1:** $[0.4, -0.2, 0.6, 0.1] \cdot h_f = 0.462 + 0.117 + 0.891 + 0.034 = 1.504$</span>
* <span style="font-size: 14px;">**Logit 2:** $[-0.1, 0.8, 0.2, -0.4] \cdot h_f = -0.116 - 0.467 + 0.297 - 0.136 = -0.422$</span>

<span style="font-size: 14px;">Continuing for all 8 vocabulary entries produces the full logit vector. The highest logit (logit 1 at 1.504) is the model's predicted next token. No softmax applied.</span>

<span style="font-size: 14px;">**Key observations:** The embedding contributes no positional information -- only blocks (via RoPE) know token order. Deeper blocks contribute less due to $1/\sqrt{2i+1}$ scaling. The final RMSNorm ensures consistent magnitude before the LM head. The LM head is independent of the embedding matrix.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Adding positional embedding at the embedding stage.**</span>

<span style="font-size: 14px;">Arcee Trinity uses RoPE inside each block, not an additive positional embedding at the input. Adding a learned or sinusoidal positional embedding to $h^{(0)}$ would double-count position information and degrade length generalization.</span>

<span style="font-size: 14px;">**2. Tying embedding and LM head weights.**</span>

<span style="font-size: 14px;">Weight tying ($W_{\text{head}} = W_{\text{embed}}$) is incorrect for Arcee Trinity. The two matrices are independently parameterized. Tying forces one matrix to serve two different functions and creates gradient conflicts.</span>

<span style="font-size: 14px;">**3. Applying softmax in the forward pass.**</span>

<span style="font-size: 14px;">The model outputs raw logits. Applying softmax then using `CrossEntropyLoss` (which applies log-softmax internally) produces double-softmax, which is mathematically wrong and causes near-zero gradients.</span>

<span style="font-size: 14px;">**4. Wrong block ordering.**</span>

<span style="font-size: 14px;">The order is: attention pre-norm, gated attention, attention post-norm + depth scaling + residual, FFN pre-norm, coarse MoE, FFN post-norm + depth scaling + residual. Swapping sub-layers or misplacing the post-norm breaks sandwich normalization.</span>

<span style="font-size: 14px;">**5. Forgetting the final RMSNorm.**</span>

<span style="font-size: 14px;">Without it, hidden states enter the LM head with unconstrained magnitude, causing wildly varying logits, training instability, and unreliable generation.</span>

<span style="font-size: 14px;">**6. Constant depth scaling factor across all blocks.**</span>

<span style="font-size: 14px;">The factor $1/\sqrt{2i+1}$ varies with block index $i$. Using a constant like $1/\sqrt{N}$ everywhere loses the property that early blocks contribute more and deeper blocks less.</span>

<span style="font-size: 14px;">**7. Applying RoPE to every layer.**</span>

<span style="font-size: 14px;">The interleaved pattern deliberately skips RoPE on certain layers. Applying RoPE everywhere eliminates the NoPE layers that learn position-agnostic representations.</span>

---