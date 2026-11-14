# <span style="font-size: 20px;">Full GPT-OSS Forward Pass</span>

<span style="font-size: 14px;">The forward pass of GPT-OSS (OpenAI, 2025) maps a sequence of token ids to a sequence of logits over the vocabulary. It is the wiring layer that glues every previous component (embedding, YaRN-scaled RoPE, sliding/full attention, attention sinks, GQA, MoE routing, RMSNorm) into a single end-to-end function. Building this assembly correctly is the difference between a model that emits coherent text and one that outputs garbage at the first layer.</span>

---

## <span style="font-size: 16px;">Where This Sits in the Architecture</span>

<span style="font-size: 14px;">GPT-OSS ships in two flavours: the 20B variant uses 24 transformer blocks and the 120B variant uses 36 blocks. Both share the same forward template:</span>

* <span style="font-size: 14px;">**Token embedding lookup** maps integer ids to dense vectors</span>
* <span style="font-size: 14px;">**N transformer blocks** stacked sequentially, each one updating the hidden state in place</span>
* <span style="font-size: 14px;">**Final RMSNorm** stabilises the last hidden state before projection</span>
* <span style="font-size: 14px;">**LM head** projects back to vocabulary logits (weight-tied with the embedding in the released checkpoints)</span>

<span style="font-size: 14px;">No positional embedding is added at the token level. Position information enters only through RoPE inside attention, which is exactly the LLaMA / DeepSeek pattern.</span>

---

## <span style="font-size: 16px;">The Pipeline in One Equation</span>

<span style="font-size: 14px;">Given input token ids $t \in \mathbb{Z}^{n}$, embedding $E \in \mathbb{R}^{V \times d}$, blocks $f_0, \ldots, f_{L-1}$, final norm scale $\gamma_f$, and head $H \in \mathbb{R}^{d \times V}$:</span>

$$
h^{(0)} = E[t]
$$

$$
h^{(i+1)} = f_i\big(h^{(i)},\, \cos,\, \sin,\, i\big) \quad \text{for } i = 0, \ldots, L-1
$$

$$
\text{logits} = \text{RMSNorm}(h^{(L)},\, \gamma_f) \cdot H
$$

<span style="font-size: 14px;">Output shape is $(n, V)$. The cos/sin tables are computed once per call from the YaRN-scaled inverse frequencies for the full sequence length $n$.</span>

---

## <span style="font-size: 16px;">Step 1: Token Embedding Lookup</span>

<span style="font-size: 14px;">Embedding is a gather, not a matmul. Given vocab matrix $E$ of shape $(V, d)$ and token ids $t$ of shape $(n,)$, the embedded sequence is:</span>

$$
h^{(0)}_i = E[t_i]
$$

* <span style="font-size: 14px;">In PyTorch: $\texttt{embed\_W[input\_ids]}$ where $\texttt{input\_ids}$ is a long tensor. The result has shape $(n, d)$</span>
* <span style="font-size: 14px;">No bias, no activation, no positional add. Just a row lookup</span>
* <span style="font-size: 14px;">GPT-OSS uses vocab size $V = 201088$ and hidden $d = 2880$. Embedding therefore stores roughly 579 M parameters: a single matrix that dominates memory at small layer counts</span>

---

## <span style="font-size: 16px;">Step 2: Precompute YaRN cos/sin Once</span>

<span style="font-size: 14px;">RoPE needs $(\cos, \sin)$ tables of shape $(n, d_h / 2)$. Computing them inside each layer would be wasteful since they depend only on positions and head dim, not on activations. The forward function builds them once per call.</span>

<span style="font-size: 14px;">The inverse frequencies follow YaRN:</span>

$$
\text{inv\_freq}_k = \theta^{-2k/d_h}
$$

<span style="font-size: 14px;">If $\text{scaling\_factor} > 1$, GPT-OSS applies the YaRN ramp: extrapolate for high-frequency dimensions, interpolate for low-frequency ones. The ramp is a clipped linear interpolation:</span>

$$
\text{ramp}_k = \text{clip}\!\left(\frac{k - \text{low}}{\text{high} - \text{low}},\, 0,\, 1\right)
$$

$$
\text{inv\_freq}_k \leftarrow (1 - \text{ramp}_k) \cdot \text{inv\_freq}_k + \text{ramp}_k \cdot \frac{\text{inv\_freq}_k}{\text{scaling\_factor}}
$$

<span style="font-size: 14px;">For $\text{scaling\_factor} = 1$ the YaRN branch is a no-op and inv_freq stays at its base $\theta$-spaced values. Then:</span>

$$
\text{angles}_{p,k} = p \cdot \text{inv\_freq}_k, \quad \cos = \cos(\text{angles}), \quad \sin = \sin(\text{angles})
$$

* <span style="font-size: 14px;">Compute once before the layer loop, broadcast inside each attention call</span>
* <span style="font-size: 14px;">All layers share the same cos/sin: the alternation pattern controls masking, not rotation</span>
* <span style="font-size: 14px;">Slice $\text{cos}[:n]$ for shorter sequences if you cache for the longest seen length</span>

---

## <span style="font-size: 16px;">Step 3: Alternating Attention Across Layers</span>

<span style="font-size: 14px;">GPT-OSS interleaves two attention patterns to balance global reach and local efficiency:</span>

* <span style="font-size: 14px;">**Even layers** ($i \bmod 2 = 0$) use sliding-window attention with window size 128. Each query token can only attend to the last 128 keys</span>
* <span style="font-size: 14px;">**Odd layers** ($i \bmod 2 = 1$) use full causal attention. Each query can attend to all previous tokens up to the start of the sequence</span>

<span style="font-size: 14px;">The implementation receives the layer index $i$ and chooses the mask accordingly. The sliding mask is the causal mask combined with a band restriction: position $j$ is masked from query $i$ iff $i - j \geq w$ or $j > i$.</span>

<span style="font-size: 14px;">This alternation is identical to the Gemma family pattern. The motivation is straightforward:</span>

* <span style="font-size: 14px;">Sliding layers reduce KV-cache size and attention compute at long context</span>
* <span style="font-size: 14px;">Full layers preserve a path for long-range information to flow</span>
* <span style="font-size: 14px;">Alternating, rather than batching all sliding then all full, keeps the global pathway active throughout depth</span>

<span style="font-size: 14px;">Inside each block, attention also uses GQA expansion (8 KV heads against 64 query heads in 20B) and per-head learned sinks in the softmax denominator. Those are details handled inside the block; the forward pass only has to feed the layer index in.</span>

---

## <span style="font-size: 16px;">Step 4: Stack the Blocks</span>

<span style="font-size: 14px;">Every block is the same recipe, parameterised only by its weights and its layer index. In pseudocode:</span>

<span style="font-size: 14px;">1. $h_n = \text{RMSNorm}(h, \gamma_{\text{attn}})$</span>

<span style="font-size: 14px;">2. $h = h + \text{Attn}(h_n, \cos, \sin, i)$</span>

<span style="font-size: 14px;">3. $h_n = \text{RMSNorm}(h, \gamma_{\text{mlp}})$</span>

<span style="font-size: 14px;">4. $h = h + \text{MoE}(h_n)$</span>

<span style="font-size: 14px;">The residual must be preserved after each sublayer. Skipping one residual on the last layer (a common bug) silently degrades output norms without crashing.</span>

* <span style="font-size: 14px;">Pre-LN is used throughout: norm is applied before the sublayer, not after the residual add</span>
* <span style="font-size: 14px;">The MoE replaces the dense FFN at every layer in GPT-OSS (unlike DeepSeek V3 which keeps the first few layers dense)</span>
* <span style="font-size: 14px;">Each MoE has 128 experts with top-4 routing in 20B (32 experts top-4 in the cards, check release notes for exact counts)</span>

---

## <span style="font-size: 16px;">Step 5: Final RMSNorm and LM Head</span>

<span style="font-size: 14px;">After the last block, apply RMSNorm one more time using the model-level scale $\gamma_f$:</span>

$$
h_{\text{out}} = \text{RMSNorm}(h^{(L)}, \gamma_f) = \frac{h^{(L)}}{\sqrt{\text{mean}(h^{(L)\,2}) + \epsilon}} \cdot \gamma_f
$$

<span style="font-size: 14px;">Then project to vocabulary logits:</span>

$$
\text{logits} = h_{\text{out}} \cdot H
$$

<span style="font-size: 14px;">where $H \in \mathbb{R}^{d \times V}$. The released GPT-OSS checkpoints tie $H = E^\top$: the lm_head is just the embedding matrix transposed. From a memory standpoint this halves the parameter cost of the head; from a learning standpoint it forces input and output token spaces to share the same geometry.</span>

* <span style="font-size: 14px;">This problem keeps both $E$ and $H$ as separate inputs so the user can verify shapes without inferring the tie</span>
* <span style="font-size: 14px;">In production, $H$ is constructed by transposing $E$ at load time</span>
* <span style="font-size: 14px;">The final norm is not optional. Without it the logits inherit unbounded scale from accumulated residual additions</span>

---

## <span style="font-size: 16px;">Worked Example (Tiny Config)</span>

<span style="font-size: 14px;">Configure $L = 2$ layers, $d = 4$, head_dim $= 2$, 1 KV head, $q_\text{mult} = 2$, 2 experts top-1, intermediate $= 4$, vocab $= 4$, sliding_window $= 2$, scaling_factor $= 1.0$. Feed $t = [1, 2]$:</span>

* <span style="font-size: 14px;">$h^{(0)} = E[[1, 2]]$ has shape $(2, 4)$</span>
* <span style="font-size: 14px;">YaRN: $\text{inv\_freq} = [1.0, 1/100]$. Positions $[0, 1]$. cos shape $(2, 1)$</span>
* <span style="font-size: 14px;">Layer 0 is even: sliding window 2. With 2 tokens, the window does not exclude any non-causal position</span>
* <span style="font-size: 14px;">Layer 1 is odd: full causal. Same effective mask as layer 0 for $n = 2$</span>
* <span style="font-size: 14px;">After two blocks, apply RMSNorm with $\gamma_f$, then matmul by $H \in \mathbb{R}^{4 \times 4}$</span>
* <span style="font-size: 14px;">Output logits shape $(2, 4)$</span>

<span style="font-size: 14px;">For 5 or more tokens, the sliding window starts excluding far-back positions on even layers. That is when the alternation actually changes outputs versus a uniform-full-attention pipeline.</span>

---

## <span style="font-size: 16px;">What the Paper Says</span>

<span style="font-size: 14px;">The GPT-OSS model card describes a "decoder-only transformer with alternating local and global attention, learned attention sinks, mixture-of-experts feed-forward, and YaRN-extended RoPE for long context". Each phrase corresponds to one box in the forward pass:</span>

* <span style="font-size: 14px;">**Decoder-only**: only the causal forward pipeline, no encoder cross-attention</span>
* <span style="font-size: 14px;">**Alternating attention**: even-layer sliding, odd-layer full, controlled by the layer index passed to the block</span>
* <span style="font-size: 14px;">**Attention sinks**: learned per-head scalar in the softmax denominator, lives inside the attention call</span>
* <span style="font-size: 14px;">**MoE feed-forward**: dispatched inside the block, no dense FFN at any layer</span>
* <span style="font-size: 14px;">**YaRN RoPE**: cos/sin tables computed at the forward level, then passed to every layer</span>

<span style="font-size: 14px;">The 128K context window is achieved by setting initial_context_length $= 4096$ and scaling_factor $= 32$, which extends the effective context by 32x via the YaRN ramp.</span>

---

## <span style="font-size: 16px;">Why This Order Matters</span>

* <span style="font-size: 14px;">**Embed first.** Attention needs token vectors before it can compute Q/K/V. Trying to apply norm or attention to raw integer ids will crash or silently broadcast nonsense</span>
* <span style="font-size: 14px;">**RoPE precompute before the layer loop.** Putting cos/sin computation inside each layer multiplies the cost by $L$ for no benefit. The tables only depend on position</span>
* <span style="font-size: 14px;">**Layer index threaded through the block.** Without the index, the block cannot decide whether to apply sliding or full attention. A common mistake is to bake the pattern into the block constructor instead of the forward arg; this works in PyTorch but is fragile</span>
* <span style="font-size: 14px;">**Final norm after the last residual.** RMSNorm before lm_head is what calibrates logit scale. Dropping it makes early-training logits explode and is hard to debug from the loss alone</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Let $n$ be sequence length, $d$ hidden, $L$ layers, $E$ experts, $k$ top-k, $I$ intermediate. Per-token compute is dominated by:</span>

* <span style="font-size: 14px;">Attention: $O(L \cdot n \cdot d)$ for projections plus $O(L \cdot n^2 \cdot d_h)$ for scores. Sliding layers reduce $n^2$ to $n \cdot w$</span>
* <span style="font-size: 14px;">MoE: $O(L \cdot k \cdot d \cdot I)$ per token (only $k$ experts active out of $E$)</span>
* <span style="font-size: 14px;">LM head: $O(n \cdot d \cdot V)$ which is often the largest single matmul at small batch</span>

<span style="font-size: 14px;">At 128K context with 32 sliding layers and 32 full layers, sliding cuts roughly half of the attention compute compared to a uniform full-attention model.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the final RMSNorm.** The logits magnitude inherits accumulated residual norm. Without final norm, output distributions are flat or peaked in unpredictable ways. The trained $\gamma_f$ exists exactly to calibrate this scale</span>
* <span style="font-size: 14px;">**Applying full attention on every layer.** Numerically the outputs are close on short sequences, so this can silently pass unit tests. The bug appears at long context where the sliding-window KV-cache contract is violated and memory blows up</span>
* <span style="font-size: 14px;">**Wrong embedding axis.** $E[t]$ versus $E.T[t]$ versus $t \cdot E$ are all syntactically valid in PyTorch but only one gives the right shape. Confusing $(V, d)$ vs $(d, V)$ for the embedding matrix is the single most common porting bug</span>
* <span style="font-size: 14px;">**LM head transposed.** The head is $(d, V)$ in this problem's convention, so logits $= h \cdot H$. If you load it as $(V, d)$ and forget to transpose, the matmul either crashes on shape or silently produces a $(n, d)$ tensor when $V = d$</span>
* <span style="font-size: 14px;">**Layer index off by one.** Passing $i + 1$ or the reversed index swaps which layers do sliding versus full. Outputs differ subtly. Always pass the natural 0-indexed layer position</span>
* <span style="font-size: 14px;">**Recomputing cos/sin inside the layer.** Correct but wasteful. Also a source of bugs if the per-layer call uses the wrong sequence length (e.g. only the current token in a generation step)</span>
* <span style="font-size: 14px;">**Mutating the residual.** Each block must write $h = h + \text{sublayer}(\text{norm}(h))$, not $h = \text{sublayer}(\text{norm}(h))$. Dropping the residual on any layer destroys the gradient highway and silently degrades quality</span>
* <span style="font-size: 14px;">**Floating-point accumulation in fp32 only.** The reference implementation uses float64 to keep numerical comparisons strict at the test level. Production runs in bf16 and accepts small numerical drift, but your verification should use float64 to avoid masking real bugs</span>

---

