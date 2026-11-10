# <span style="font-size: 20px;">Transformer Decoder Block</span>

<span style="font-size: 14px;">The Transformer decoder block is the core unit in GPT-style language models and the decoder side of sequence-to-sequence architectures like T5 and BART. Its three-sublayer structure - masked self-attention, cross-attention, and FFN - is what enables autoregressive generation with conditioning on an encoded source.</span>

---

## <span style="font-size: 16px;">Decoder Block Architecture</span>

<span style="font-size: 14px;">The decoder block extends the encoder block with an additional cross-attention sublayer. Each sublayer follows the Post-LN pattern: $\text{LN}(x + \text{Sublayer}(x))$.</span>

<span style="font-size: 14px;">**Sublayer 1 - Masked Self-Attention**: the decoder attends to its own previous outputs. A causal mask ensures position $i$ can only attend to positions $j \leq i$. This is what makes the decoder autoregressive: each token can only depend on tokens that have already been generated.</span>

<span style="font-size: 14px;">**Sublayer 2 - Cross-Attention**: queries come from the decoder's current state, while keys and values come from the encoder output. This allows the decoder to "read" the encoded source sequence at every layer. For example, in translation, each decoder position attends to relevant parts of the source sentence.</span>

<span style="font-size: 14px;">**Sublayer 3 - Feed-Forward Network**: identical to the encoder FFN. Provides per-position nonlinear computation after the two attention sublayers have gathered contextual information.</span>

---

## <span style="font-size: 16px;">Causal Masking</span>

<span style="font-size: 14px;">The causal (look-ahead) mask is a lower-triangular matrix:</span>

$$
M_{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}
$$

<span style="font-size: 14px;">Positions where $M_{ij} = 0$ are set to $-10^9$ before softmax, making their attention weight effectively zero. This ensures:</span>

- <span style="font-size: 14px;">**Training**: all positions can be trained in parallel (teacher forcing), but each position only sees past tokens, matching the inference-time constraint.</span>
- <span style="font-size: 14px;">**Inference**: tokens are generated one at a time, and the mask naturally prevents seeing future tokens that have not been generated yet.</span>
- <span style="font-size: 14px;">**KV cache**: during inference, previously computed keys and values are cached and reused, so only the new token's Q/K/V need to be computed. The mask ensures consistency between the cached and new computations.</span>

---

## <span style="font-size: 16px;">Cross-Attention: Connecting Encoder and Decoder</span>

<span style="font-size: 14px;">Cross-attention is the bridge between encoder and decoder. The key insight is the asymmetric source of Q, K, V:</span>

- <span style="font-size: 14px;">$Q$ comes from the decoder (what information does this decoder position need?)</span>
- <span style="font-size: 14px;">$K$ comes from the encoder (what information does each source position offer?)</span>
- <span style="font-size: 14px;">$V$ comes from the encoder (the actual information to retrieve)</span>

<span style="font-size: 14px;">This means the attention scores measure compatibility between decoder queries and encoder keys. Each decoder position can attend to any source position, allowing flexible alignment. In translation, this replaces the fixed alignment models used in pre-Transformer seq2seq.</span>

<span style="font-size: 14px;">**Note**: in decoder-only models (GPT, LLaMA), there is no cross-attention sublayer. The encoder output is absent, and the model conditions only on its own past outputs. This simplifies the block to just masked self-attention + FFN.</span>

---

## <span style="font-size: 16px;">Decoder-Only vs Encoder-Decoder</span>

<span style="font-size: 14px;">Two dominant Transformer architectures exist in practice:</span>

- <span style="font-size: 14px;">**Encoder-decoder** (T5, BART, original Transformer): uses full decoder blocks with all three sublayers. Best for tasks with distinct input/output sequences (translation, summarization). The cross-attention allows flexible conditioning on the entire source.</span>
- <span style="font-size: 14px;">**Decoder-only** (GPT, LLaMA, Mistral): uses decoder blocks without cross-attention (only masked self-attention + FFN). Input and output share the same sequence. Dominant for language modeling and general-purpose LLMs because of simplicity and the ability to handle any task as text-to-text generation.</span>

<span style="font-size: 14px;">The success of decoder-only architectures suggests that cross-attention may not be strictly necessary - the model can learn to attend to "source" information within the same sequence using self-attention alone. However, encoder-decoder models are more parameter-efficient for tasks with a clear input-output distinction.</span>

---


## <span style="font-size: 16px;">Efficient Inference Patterns</span>

<span style="font-size: 14px;">**KV-Cache.** During autoregressive generation, each decoder block caches the K and V tensors from all previously generated tokens. When generating token t, only the new token's Q, K, V are computed, and the cached K/V from tokens 1..t-1 are reused. This converts generation from O(t^2) to O(t) per step but requires O(t * d_model * num_layers) memory.</span>

<span style="font-size: 14px;">**Speculative decoding.** A small "draft" model generates several candidate tokens quickly, then the large model verifies them in a single forward pass (since verification is parallel). Accepted tokens are free; rejected ones trigger regeneration from the rejection point. This can achieve 2-3x speedup without changing output quality.</span>

<span style="font-size: 14px;">**Continuous batching.** In serving, different requests finish at different times. Continuous batching (also called iteration-level scheduling) allows new requests to join the batch as old ones complete, maximizing GPU utilization. This requires careful KV-cache memory management but dramatically improves throughput in production.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does the decoder need both self-attention and cross-attention?**</span>
  <span style="font-size: 14px;">A: Self-attention lets the decoder build coherent output by attending to its own previously generated tokens (language modeling). Cross-attention lets it condition on the source sequence (reading the input). Without self-attention, the decoder could not maintain output coherence. Without cross-attention, it could not access the source information at every layer.</span>

- <span style="font-size: 14px;">**Q: How does the decoder handle different source and target lengths?**</span>
  <span style="font-size: 14px;">A: Self-attention operates on the target sequence (shape $T_{\text{tgt}} \times T_{\text{tgt}}$). Cross-attention has Q of shape $T_{\text{tgt}}$ and K, V of shape $T_{\text{src}}$, producing scores of shape $T_{\text{tgt}} \times T_{\text{src}}$. The output always has the target sequence length. This naturally handles variable-length input-output pairs.</span>

- <span style="font-size: 14px;">**Q: What happens if you remove the causal mask from the decoder?**</span>
  <span style="font-size: 14px;">A: During training, the model could "cheat" by looking at future target tokens. It would learn to simply copy the next token from the input instead of predicting it. At inference time, future tokens do not exist, so the model's behavior would be inconsistent with training. The causal mask ensures the same information constraint during both training and inference.</span>

- <span style="font-size: 14px;">**Q: Why is the parameter count of a decoder block larger than an encoder block?**</span>
  <span style="font-size: 14px;">A: The decoder has an additional cross-attention sublayer with its own 4 linear projections and 1 LayerNorm. So it has 10 linear layers (vs 6 in encoder), 3 LayerNorms (vs 2), and correspondingly more parameters: $8d_{\text{model}}^2 + 2d_{\text{model}} \cdot d_{\text{ff}}$ plus bias terms, compared to $4d_{\text{model}}^2 + 2d_{\text{model}} \cdot d_{\text{ff}}$ for the encoder.</span>

- <span style="font-size: 14px;">**Q: In practice, how does GPT differ from this decoder block?**</span>
  <span style="font-size: 14px;">A: GPT removes the cross-attention sublayer entirely (no encoder to attend to). It also uses Pre-LN instead of Post-LN, and typically replaces ReLU with GELU. The input is the concatenation of context + generation, and the causal mask ensures autoregressive processing. Modern GPT variants also use RoPE and RMSNorm.</span>

---