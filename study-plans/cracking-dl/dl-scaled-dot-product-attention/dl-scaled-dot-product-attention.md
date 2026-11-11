# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Scaled dot-product attention (Vaswani et al., 2017) is the atomic operation underlying every Transformer model - from BERT and GPT to Vision Transformers and diffusion models. Understanding this mechanism at the mathematical level is non-negotiable for any DL interview in 2026.</span>

---

## <span style="font-size: 16px;">The Attention Mechanism</span>

<span style="font-size: 14px;">Attention answers the question: "given a query, which keys are most relevant, and what information (values) should I retrieve?" It is a soft dictionary lookup:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

<span style="font-size: 14px;">Step by step:</span>

<span style="font-size: 14px;">**1. Compute similarity scores**: $S = QK^T \in \mathbb{R}^{n_q \times n_k}$. Each entry $S_{ij}$ is the dot product between query $i$ and key $j$, measuring how much query $i$ should attend to position $j$.</span>

<span style="font-size: 14px;">**2. Scale**: divide by $\sqrt{d_k}$. Without scaling, when $d_k$ is large, the dot products grow in magnitude (variance proportional to $d_k$), pushing softmax into regions where gradients are extremely small. Scaling by $\sqrt{d_k}$ keeps the variance at 1 regardless of dimension.</span>

<span style="font-size: 14px;">**3. Mask (optional)**: set certain positions to $-\infty$ (or $-10^9$ in practice) before softmax. After softmax, these positions become 0. This is used for causal (autoregressive) masking and padding masking.</span>

<span style="font-size: 14px;">**4. Softmax**: normalize each row to a probability distribution. Each query now has a set of attention weights that sum to 1 over all keys.</span>

<span style="font-size: 14px;">**5. Weighted sum**: multiply weights by values. The output for each query is a weighted combination of all value vectors, where the weights reflect relevance.</span>

---

## <span style="font-size: 16px;">Why Scale by $\sqrt{d_k}$?</span>

<span style="font-size: 14px;">Consider queries and keys with independent components, each with mean 0 and variance 1. The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$ (sum of $d_k$ independent products, each with variance 1).</span>

<span style="font-size: 14px;">For $d_k = 64$ (a typical value), the dot products have standard deviation $\sqrt{64} = 8$. This means many scores would be far from 0, and softmax would produce near-one-hot distributions. The gradients through softmax would be nearly zero (saturation), making training extremely slow.</span>

<span style="font-size: 14px;">Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping the softmax in a regime where gradients flow well. This is similar in spirit to the scaling in Xavier initialization and batch normalization.</span>

---

## <span style="font-size: 16px;">Masking</span>

<span style="font-size: 14px;">Two types of masks are used in Transformers:</span>

<span style="font-size: 14px;">**Causal (look-ahead) mask**: a lower-triangular matrix that prevents position $i$ from attending to positions $j > i$. This is essential for autoregressive generation (GPT, decoder side of T5). Without it, the model could "cheat" by looking at future tokens during training.</span>

<span style="font-size: 14px;">**Padding mask**: prevents attention to padding tokens in variable-length sequences within a batch. Without it, the model would waste capacity attending to meaningless padding positions.</span>

<span style="font-size: 14px;">Implementation: masked positions are set to $-10^9$ (not exactly $-\infty$ to avoid NaN in gradients). After softmax, these positions have weight $\approx 0$. Setting to 0 directly before softmax would be wrong because softmax would still assign some probability to those positions.</span>

---

## <span style="font-size: 16px;">Backward Pass Through Attention</span>

<span style="font-size: 14px;">Given upstream gradient $\partial L / \partial O$ where $O = WV$ and $W = \text{softmax}(S/\sqrt{d_k})$:</span>

<span style="font-size: 14px;">**Gradient for V:**</span>

$$
\frac{\partial L}{\partial V} = W^T \cdot \frac{\partial L}{\partial O}
$$

<span style="font-size: 14px;">**Gradient for attention weights:**</span>

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial O} \cdot V^T
$$

<span style="font-size: 14px;">**Through softmax** (the non-trivial part): for each row $i$, the softmax Jacobian is $\text{diag}(w_i) - w_i w_i^T$. This gives:</span>

$$
\frac{\partial L}{\partial S_i} = w_i \odot \bigl(\frac{\partial L}{\partial W_i} - \langle \frac{\partial L}{\partial W_i},\, w_i \rangle\bigr)
$$

<span style="font-size: 14px;">where the inner product sums over the key dimension. This compact form avoids materializing the full Jacobian matrix.</span>

<span style="font-size: 14px;">**Gradient for Q and K:**</span>

$$
\begin{aligned}
\frac{\partial L}{\partial Q} &= \frac{\partial L}{\partial S} \cdot K \cdot \frac{1}{\sqrt{d_k}} \\[6pt]
\frac{\partial L}{\partial K} &= \left(\frac{\partial L}{\partial S}\right)^T \cdot Q \cdot \frac{1}{\sqrt{d_k}}
\end{aligned}
$$

---

## <span style="font-size: 16px;">Attention Complexity and Alternatives</span>

<span style="font-size: 14px;">Standard attention has $O(n^2 d)$ time and $O(n^2)$ memory for sequence length $n$. This quadratic scaling is the primary bottleneck for long sequences. Key alternatives:</span>

- <span style="font-size: 14px;">**Flash Attention** (Dao et al., 2022): exact attention with tiling to reduce memory I/O. Does not reduce computational complexity but achieves 2-4x wall-clock speedup by avoiding materializing the full attention matrix in HBM</span>
- <span style="font-size: 14px;">**Sparse attention** (Beltagy et al., 2020): attend only to local windows and fixed stride positions. Reduces to $O(n\sqrt{n})$</span>
- <span style="font-size: 14px;">**Linear attention** (Katharopoulos et al., 2020): replace $\text{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^T V)$ using a kernel function, achieving $O(nd^2)$</span>
- <span style="font-size: 14px;">**Multi-query attention** (Shazeer, 2019): share K and V across heads, reducing KV-cache memory during inference</span>
- <span style="font-size: 14px;">**Grouped-query attention** (Ainslie et al., 2023): compromise between multi-head and multi-query, used in LLaMA 2 and Mistral</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why scale by $\sqrt{d_k}$ and not $d_k$ or some learned parameter?**</span>
  <span style="font-size: 14px;">A: The dot product of two random vectors with i.i.d. components of variance 1 has variance $d_k$. Dividing by $\sqrt{d_k}$ normalizes the variance to 1, keeping softmax inputs in a numerically stable range. Using $d_k$ would over-flatten; a learned parameter adds complexity without clear benefit since the scaling is principled.</span>

- <span style="font-size: 14px;">**Q: What happens if you remove the softmax?**</span>
  <span style="font-size: 14px;">A: Without softmax, attention weights are unnormalized and can be negative. This is "linear attention." It removes the quadratic bottleneck (can be computed as $Q(K^T V)$ in $O(nd^2)$) but loses the sharp selection property - softmax creates a competition where one or few keys dominate, which is important for precise information retrieval.</span>

- <span style="font-size: 14px;">**Q: How does causal masking enable autoregressive generation?**</span>
  <span style="font-size: 14px;">A: By preventing position $i$ from attending to positions $j > i$, the output at each position depends only on past tokens. This means during inference, new tokens can be generated one at a time by appending to the KV cache without recomputing attention for previous positions. Without the mask, the model would need future tokens to compute current outputs.</span>

- <span style="font-size: 14px;">**Q: What is the KV cache and why does it matter for inference?**</span>
  <span style="font-size: 14px;">A: During autoregressive generation, each new token only needs to attend to all previous tokens. The KV cache stores the key and value projections for all previously generated tokens, so only the new token's Q, K, V need to be computed at each step. This reduces per-token inference from $O(n \cdot d)$ to $O(d)$ compute (excluding the cache lookup). The cache grows linearly with sequence length, which is the primary memory bottleneck for long-context LLMs.</span>

- <span style="font-size: 14px;">**Q: Why is Flash Attention faster if it does the same computation?**</span>
  <span style="font-size: 14px;">A: Standard attention materializes the full $n \times n$ attention matrix in GPU high-bandwidth memory (HBM), which is slow for large $n$. Flash Attention tiles the computation so that each block fits in fast SRAM (on-chip memory), never materializing the full matrix. It computes exact attention but with far fewer HBM reads/writes, making it IO-bound rather than compute-bound.</span>

---