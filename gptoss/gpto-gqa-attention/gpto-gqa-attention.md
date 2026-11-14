# <span style="font-size: 20px;">GQA Attention with Sinks</span>

<span style="font-size: 14px;">The GPT-OSS scaled-dot-product attention (SDPA) layer composes four ideas in a single forward pass: **scaled dot-product attention**, **grouped-query attention (GQA)**, an **optional sliding window** for memory-efficient long context, and **learned per-head softmax sinks**. This single function replaces what most papers describe across three or four separate sections.</span>

---

## <span style="font-size: 16px;">What the layer computes</span>

<span style="font-size: 14px;">Inputs are already projected and (in the real model) already carry rotary positional information. The layer takes:</span>

* <span style="font-size: 14px;">$Q \in \mathbb{R}^{n \times H_{kv} \times g \times d}$ where $H_{kv}$ is the number of KV heads and $g$ is the query multiplier (so total query heads $= H_{kv} \cdot g$)</span>
* <span style="font-size: 14px;">$K, V \in \mathbb{R}^{n \times H_{kv} \times d}$, one set of keys/values per KV head, shared across the $g$ queries that point at it</span>
* <span style="font-size: 14px;">$\mathbf{s} \in \mathbb{R}^{H_{kv} \cdot g}$, a learned sink logit per total query head</span>
* <span style="font-size: 14px;">$\texttt{sm\_scale}$, typically $1/\sqrt{d}$</span>
* <span style="font-size: 14px;">$\texttt{sliding\_window}$, $0$ for full causal attention, else the window width $W$</span>

<span style="font-size: 14px;">Output is the per-token concatenation across all query heads, flattened to shape $(n, H_{kv} \cdot g \cdot d)$.</span>

---

## <span style="font-size: 16px;">Core equation</span>

<span style="font-size: 14px;">For one (kv_head, q_mult) slot indexed by $(h, m)$, query position $q$, key position $k$:</span>

$$
\alpha_{h,m,q,k} = \texttt{sm\_scale} \cdot \langle Q_{q,h,m}, K_{k,h} \rangle + M_{q,k}
$$

<span style="font-size: 14px;">where $M_{q,k}$ is the causal (and optional sliding-window) mask. The softmax includes the head sink in the denominator:</span>

$$
w_{h,m,q,k} = \frac{\exp(\alpha_{h,m,q,k})}{\exp(s_{h,m}) + \sum_{k'} \exp(\alpha_{h,m,q,k'})}
$$

$$
O_{q,h,m} = \sum_k w_{h,m,q,k} \cdot V_{k,h}
$$

<span style="font-size: 14px;">Note that the sink is in the denominator but does NOT appear in the numerator that multiplies $V$. Each row of $w$ sums to less than 1.</span>

---

## <span style="font-size: 16px;">Scaled dot-product attention recap</span>

<span style="font-size: 14px;">Vanilla scaled dot-product attention from Vaswani et al. 2017 computes $\text{softmax}(QK^T/\sqrt{d})V$. The $\sqrt{d}$ scaling keeps the variance of $\langle q, k \rangle$ from growing with $d$, which is critical because softmax saturates for large inputs. If you skip the scale, gradients vanish for moderately wide heads and the model fails to train.</span>

<span style="font-size: 14px;">Causal masking adds $-\infty$ to scores where the key position is in the future (k > q). After softmax, those positions get weight 0, so the model cannot peek at tokens it has not generated yet.</span>

---

## <span style="font-size: 16px;">Grouped-query attention (GQA)</span>

<span style="font-size: 14px;">**Multi-head attention (MHA)** in the original Transformer keeps independent $K$ and $V$ projections per head. With $H$ heads, the KV cache during inference stores $H \cdot d$ floats per token, per layer. For a 70B-class model with 64 heads and 4096 sequence length, the cache becomes the dominant cost.</span>

<span style="font-size: 14px;">**Multi-query attention (MQA)** (Shazeer 2019) shrinks the cache by tying all heads to a single $K, V$ pair. The cache becomes $d$ floats per token, but the model loses representational capacity.</span>

<span style="font-size: 14px;">**Grouped-query attention (GQA)** (Ainslie et al. 2023) interpolates: keep $H_{kv}$ KV heads and let each be shared by $g = H_q / H_{kv}$ query heads. The cache becomes $H_{kv} \cdot d$ floats per token. GQA is now standard across LLaMA 2/3, Mistral, GPT-OSS, and most modern open models.</span>

<span style="font-size: 14px;">Special cases of the same layer:</span>

* <span style="font-size: 14px;">$g = 1$, $H_{kv} = H_q$: the layer reduces to MHA</span>
* <span style="font-size: 14px;">$H_{kv} = 1$, $g = H_q$: the layer reduces to MQA</span>
* <span style="font-size: 14px;">Any $1 < H_{kv} < H_q$: GQA proper</span>

<span style="font-size: 14px;">In code, GQA is implemented by **broadcasting** $K$ and $V$ along the new $g$ axis: $K$ of shape $(n, H_{kv}, d)$ becomes $(n, H_{kv}, g, d)$ without copying memory. Each $K$ vector is shared by $g$ queries, so the einsum still works elementwise.</span>

---

## <span style="font-size: 16px;">Sliding window attention</span>

<span style="font-size: 14px;">Full causal attention has cost $O(n^2 d)$ per layer. For long context (32K+ tokens) this is unaffordable. **Sliding window attention** (Beltagy et al. 2020, Longformer; Mistral 2023) restricts each query to attend to the previous $W$ keys, dropping cost to $O(n \cdot W \cdot d)$.</span>

<span style="font-size: 14px;">The mask is constructed from two pieces:</span>

* <span style="font-size: 14px;">Causal piece: $-\infty$ on the strict upper triangle, $\texttt{torch.triu(..., diagonal=1)}$. This blocks future keys.</span>
* <span style="font-size: 14px;">Window piece: $-\infty$ on entries more than $W$ below the diagonal, $\texttt{torch.tril(..., diagonal=-W)}$. This blocks keys too far in the past.</span>

<span style="font-size: 14px;">Adding both masks gives a band of finite values exactly $W$ wide along the diagonal. With $W = 0$ the window mask is skipped and we get plain causal attention.</span>

<span style="font-size: 14px;">GPT-OSS alternates layers: even-indexed layers use $W = 128$ sliding window, odd-indexed layers use full causal attention. This hybrid pattern gives most of the inference savings without losing the ability to reach back arbitrarily far on some layers.</span>

---

## <span style="font-size: 16px;">Learned softmax sinks</span>

<span style="font-size: 14px;">A standard softmax forces $\sum_k w_k = 1$. Even when no key is a good match, the softmax must allocate all of its weight somewhere. In long-context decoder models this manifests as **attention sinks**: the first few tokens consistently absorb large fractions of attention weight regardless of content (Xiao et al. 2024). The model invents a sink token because it cannot output "attend to nothing".</span>

<span style="font-size: 14px;">GPT-OSS bakes the sink directly into the layer. Each query head learns a scalar logit $s_{h,m}$. That logit is concatenated as an extra entry into the softmax denominator. After normalization, the sink column is dropped from the output:</span>

$$
w_k = \frac{\exp(\alpha_k)}{\exp(s) + \sum_{k'} \exp(\alpha_{k'})}, \quad \sum_k w_k = 1 - \frac{\exp(s)}{\exp(s) + \sum_{k'} \exp(\alpha_{k'})}
$$

<span style="font-size: 14px;">Intuition:</span>

* <span style="font-size: 14px;">If $s \to -\infty$, the sink contributes 0 and we recover vanilla softmax</span>
* <span style="font-size: 14px;">If $s$ is large, the sink absorbs most of the mass and all $w_k$ shrink toward 0</span>
* <span style="font-size: 14px;">The model learns per-head how much weight to "throw away"</span>

<span style="font-size: 14px;">From the GPT-OSS paper card: "Attention layers use a learned attention sink per-head, where the denominator of the softmax has a further additive value." The sink behaves like a virtual key with zero value vector, but parametrized in logit space so the model can adjust it through gradient descent without needing a separate value embedding.</span>

---

## <span style="font-size: 16px;">How the three combine</span>

<span style="font-size: 14px;">The full forward pass weaves the pieces tightly:</span>

<span style="font-size: 14px;">1. **Expand K and V** via broadcasting to $(n, H_{kv}, g, d)$ so the einsum over heads sees one $K, V$ per query.</span>

<span style="font-size: 14px;">2. **Score** with $\texttt{einsum("qhmd,khmd->hmqk", Q, K_e)}$ to produce a tensor $(H_{kv}, g, n_q, n_k)$ of dot products.</span>

<span style="font-size: 14px;">3. **Scale and mask**: multiply by $\texttt{sm\_scale}$, add the causal+window mask broadcast over heads.</span>

<span style="font-size: 14px;">4. **Concatenate sink** as a column of shape $(H_{kv}, g, n_q, 1)$ filled with $s_{h,m}$. The combined tensor has shape $(H_{kv}, g, n_q, n_k + 1)$.</span>

<span style="font-size: 14px;">5. **Stable softmax**: subtract the per-row max (over the combined axis) before exponentiating. The sink may be a large positive value, so taking the max only over original scores would overflow.</span>

<span style="font-size: 14px;">6. **Drop the sink column** from $W$ via $W = W[..., :-1]$. The sink stays in the denominator, but does not multiply any value.</span>

<span style="font-size: 14px;">7. **Aggregate values** with $\texttt{einsum("hmqk,khmd->qhmd", W, V_e)}$. Reshape to $(n, H_{kv} \cdot g \cdot d)$.</span>

---

## <span style="font-size: 16px;">GPT-OSS specifics</span>

<span style="font-size: 14px;">From the GPT-OSS model card (the 20B variant):</span>

* <span style="font-size: 14px;">$H_q = 64$ query heads, $H_{kv} = 8$ KV heads, so $g = 8$</span>
* <span style="font-size: 14px;">$d_{\text{head}} = 64$ per head, $d_{\text{model}} = 64 \times 64 = 4096$</span>
* <span style="font-size: 14px;">Sliding window $W = 128$ on even layers, full attention on odd layers</span>
* <span style="font-size: 14px;">$\texttt{sm\_scale} = 1 / \sqrt{64} = 0.125$</span>
* <span style="font-size: 14px;">One sink scalar per query head, so 64 sink parameters per layer</span>

<span style="font-size: 14px;">The KV cache savings: vanilla MHA would store $64 \cdot 64 = 4096$ floats of $K$ and 4096 of $V$ per token per layer. With GQA at $H_{kv} = 8$, that drops to $8 \cdot 64 = 512$ each, an $8\times$ reduction. Combined with sliding window on half the layers, total cache cost roughly halves again.</span>

---

## <span style="font-size: 16px;">Worked numerical example</span>

<span style="font-size: 14px;">Take the smallest non-trivial case: $n=2$, $H_{kv}=1$, $g=1$, $d=2$, $\texttt{sm\_scale}=0.5$, $\texttt{sliding\_window}=0$, single sink $s = 0$.</span>

<span style="font-size: 14px;">$Q_0 = [1, 0]$, $Q_1 = [0, 1]$, $K_0 = [1, 0]$, $K_1 = [0, 1]$, $V_0 = [1, 2]$, $V_1 = [3, 4]$.</span>

<span style="font-size: 14px;">Step 1, raw scores $\langle Q_q, K_k \rangle$:</span>

$$
QK^T = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

<span style="font-size: 14px;">Step 2, scale by 0.5 and add causal mask:</span>

$$
\alpha = \begin{pmatrix} 0.5 & -\infty \\ 0 & 0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">Step 3, append sink column ($s = 0$, so $\exp(0) = 1$):</span>

$$
[\alpha \mid s] = \begin{pmatrix} 0.5 & -\infty & 0 \\ 0 & 0.5 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">Step 4, softmax over the combined row. Row 0: $\exp$ gives $(1.6487, 0, 1)$, sum $= 2.6487$. Weights $(0.6225, 0, 0.3775)$. Drop sink: $w_0 = (0.6225, 0)$.</span>

<span style="font-size: 14px;">Row 1: $\exp$ gives $(1, 1.6487, 1)$, sum $= 3.6487$. Weights $(0.2740, 0.4519, 0.2741)$. Drop sink: $w_1 = (0.2740, 0.4519)$.</span>

<span style="font-size: 14px;">Step 5, output $O_q = \sum_k w_{q,k} V_k$:</span>

<span style="font-size: 14px;">$O_0 = 0.6225 \cdot [1, 2] + 0 \cdot [3, 4] = [0.6225, 1.2449]$</span>

<span style="font-size: 14px;">$O_1 = 0.2740 \cdot [1, 2] + 0.4519 \cdot [3, 4] = [1.6297, 2.3556]$</span>

<span style="font-size: 14px;">Note that $w_0$ sums to 0.6225, not 1. The remaining 0.3775 of attention mass leaks to the sink and never reaches any value vector.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the GQA broadcast.** If you forget to expand $K$ and $V$ to shape $(n, H_{kv}, g, d)$, the einsum runs but only uses one query group per kv head. Outputs for $g > 1$ are wrong. This is silent: shapes line up if you tile naively at the end.</span>

* <span style="font-size: 14px;">**Max over the wrong axis.** The max-subtraction must include the sink. If you compute the max only over the original scores, a large positive sink overflows $\exp$ and you get NaN. Always max over the concatenated tensor.</span>

* <span style="font-size: 14px;">**Slicing the sink in the wrong order.** Concatenate the sink as the LAST column, then slice with $\texttt{W[..., :-1]}$. If you concatenate at the front and forget to slice $\texttt{W[..., 1:]}$, the value einsum will multiply $V_0$ by the sink weight, producing garbage.</span>

* <span style="font-size: 14px;">**Wrong sliding-window diagonal.** The window mask uses $\texttt{torch.tril(..., diagonal=-W)}$, not $\texttt{diagonal=-W-1}$ or $\texttt{diagonal=W}$. Off-by-one here either lets in keys that should be blocked or blocks keys you wanted to keep.</span>

* <span style="font-size: 14px;">**Sliding window with $W = 0$.** $W = 0$ means "full causal attention", not "block everything". Guard with $\texttt{if sliding\_window > 0}$ before adding the window mask.</span>

* <span style="font-size: 14px;">**Forgetting $\texttt{sm\_scale}$.** Without scaling, large $\langle Q, K \rangle$ values push softmax to one-hot. The model trains to extreme weights and gradients collapse. Always scale before the mask is added.</span>

* <span style="font-size: 14px;">**Sink shape confusion.** Sinks has shape $(H_{kv} \cdot g,)$. You must reshape to $(H_{kv}, g, 1, 1)$ before broadcasting to the score tensor. Forgetting the reshape silently broadcasts wrong if total heads happens to divide nicely.</span>

* <span style="font-size: 14px;">**Dtypes.** Use float64 internally for stability. The reference uses $-\infty$ explicitly via $\texttt{torch.full(..., -float('inf'))}$ rather than a large negative constant; mixing the two can cause subtle differences when the row otherwise contains only finite negatives.</span>

* <span style="font-size: 14px;">**Einsum index order.** The convention $\texttt{"qhmd,khmd->hmqk"}$ contracts on $d$ and keeps $h, m$ shared. Swapping query and key axis names (e.g. typing $\texttt{"khmd,qhmd->hmqk"}$) silently transposes the score matrix and produces an anti-causal pattern that the mask cannot fix.</span>

* <span style="font-size: 14px;">**Output reshape.** The final $\texttt{reshape(n\_tokens, -1)}$ flattens heads in the order $(H_{kv}, g, d)$. Downstream the output projection expects this exact ordering. Reshaping via $\texttt{transpose}$ first or flattening in a different order silently shuffles the heads.</span>

---

## <span style="font-size: 16px;">Connections to real models</span>

<span style="font-size: 14px;">Once you can read this layer, you can read the rest of the modern decoder zoo:</span>

* <span style="font-size: 14px;">**LLaMA 2/3** uses GQA without sinks or sliding window. Set $\mathbf{s} \to -\infty$ and $W = 0$ and you recover its attention.</span>

* <span style="font-size: 14px;">**Mistral 7B** uses GQA with sliding window 4096 on every layer. Set $\mathbf{s} \to -\infty$ and $W = 4096$.</span>

* <span style="font-size: 14px;">**Gemma 2** alternates sliding-window and full attention every other layer, similar to GPT-OSS, but without learned sinks.</span>

* <span style="font-size: 14px;">**GPT-OSS** is the first widely released open model to bake sinks into the base architecture rather than as a streaming-attention hack at inference time.</span>

<span style="font-size: 14px;">In all four, the cache footprint scales as $H_{kv} \cdot d$ per token per layer rather than $H_q \cdot d$, and the layer is implemented as one fused kernel at production scale (Flash Attention 3, xformers). The PyTorch version here is the unfused reference that those kernels are validated against.</span>

---