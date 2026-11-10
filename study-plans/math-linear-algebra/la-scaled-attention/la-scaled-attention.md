## Scaled Dot-Product Attention

<span style="font-size: 14px;">The attention mechanism is the core building block of the Transformer architecture. It computes a weighted combination of value vectors, where the weights are determined by the similarity between query and key vectors:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

---

## Step by Step

<span style="font-size: 14px;">1. **Compute scores**: $S = QK^T \in \mathbb{R}^{n \times m}$ where $S_{ij} = q_i^T k_j$ measures how much query $i$ attends to key $j$</span>

<span style="font-size: 14px;">2. **Scale**: divide by $\sqrt{d_k}$ to prevent the dot products from growing large in magnitude</span>

<span style="font-size: 14px;">3. **Softmax**: apply row-wise softmax to get attention weights $W = \text{softmax}(S / \sqrt{d_k})$, where each row sums to 1</span>

<span style="font-size: 14px;">4. **Weighted sum**: output $O = WV$ where each output row is a weighted combination of value vectors</span>

---

## Why Scale by $\sqrt{d_k}$?

<span style="font-size: 14px;">If $q$ and $k$ are random vectors with independent entries of mean 0 and variance 1, then $q^T k = \sum_{j=1}^{d_k} q_j k_j$ has mean 0 and variance $d_k$ (each product $q_j k_j$ has variance 1, and they are independent, so variances add). As $d_k$ grows, the dot products grow in magnitude, pushing the softmax into regions of extremely small gradients. For example, with $d_k = 512$, the standard deviation of the scores is $\sqrt{512} \approx 22.6$, meaning scores routinely exceed 20, where the softmax is nearly a hard argmax. Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping the softmax in a regime with useful gradients.</span>

---

## Worked Example (3 tokens, $d_k = 2$)

<span style="font-size: 14px;">Let $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{pmatrix}$, $V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{pmatrix}$. Scores: $QK^T = \begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{pmatrix}$. Scaled by $\sqrt{2}$: $\begin{pmatrix} 0.71 & 0 & 0.35 \\ 0 & 0.71 & 0.35 \\ 0.71 & 0.71 & 0.71 \end{pmatrix}$. After row-wise softmax, token 1 attends most to key 1, token 2 attends most to key 2, and token 3 attends equally to all keys. The output blends value vectors according to these weights.</span>

---

## Softmax Function

<span style="font-size: 14px;">The row-wise softmax converts raw scores into probabilities:</span>

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

<span style="font-size: 14px;">For numerical stability, subtract the maximum: $\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$.</span>

---

## Multi-Head Attention

<span style="font-size: 14px;">Instead of a single attention function, multi-head attention runs $h$ parallel attention heads with different learned projections: $Q_i = QW_i^Q$, $K_i = KW_i^K$, $V_i = VW_i^V$ where each projection has dimension $d_k = d_{\text{model}} / h$. The outputs are concatenated and linearly projected:</span>

$$
\text{MultiHead}(Q,K,V) = [\text{head}_1; \ldots; \text{head}_h] W^O
$$

<span style="font-size: 14px;">Different heads can attend to different aspects: one head might capture syntactic relationships, another semantic similarity, another positional patterns.</span>

---

## Masking and Complexity

<span style="font-size: 14px;">In decoder (autoregressive) models, a **causal mask** prevents each position from attending to future positions. This is implemented by setting the upper-triangular entries of the score matrix to $-\infty$ before softmax, so those positions get zero attention weight. The computational complexity of attention is $O(n^2 d_k)$ where $n$ is the sequence length, due to the $n \times n$ score matrix. This quadratic cost in sequence length is the main bottleneck for long sequences and has motivated efficient attention variants (linear attention, sparse attention, FlashAttention).</span>

---

## Self-Attention

<span style="font-size: 14px;">In self-attention, $Q$, $K$, and $V$ are all derived from the same input $X$ via learned linear projections: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Each position attends to all other positions, allowing the model to capture long-range dependencies regardless of sequence distance. Attention can be interpreted as a soft dictionary lookup: queries search for matching keys, and the output is a weighted combination of the corresponding values.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Transformer architecture**: the foundation of GPT, BERT, and virtually all modern large language models. Stacking multi-head self-attention layers with feedforward networks creates powerful sequence models.</span>
* <span style="font-size: 14px;">**Vision Transformers (ViT)**: attention applied to image patches enables powerful image classification and generation models, often rivaling or surpassing convolutional networks.</span>
* <span style="font-size: 14px;">**Cross-attention**: when $Q$ comes from one sequence and $K, V$ from another, cross-attention enables tasks like machine translation, text-to-image generation, and protein structure prediction (AlphaFold).</span>
* <span style="font-size: 14px;">**Attention as soft lookup**: attention can be interpreted as a differentiable dictionary lookup, where queries retrieve a soft combination of values based on key similarity. This perspective connects attention to kernel methods and Nadaraya-Watson regression.</span>