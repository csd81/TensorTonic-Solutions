# <span style="font-size: 20px;">Text Generation Sampling</span>

---

## <span style="font-size: 16px;">Core Concept</span>

<span style="font-size: 14px;">Sampling-based text generation produces diverse outputs by randomly selecting tokens from the model's predicted probability distribution, rather than always picking the most likely token (greedy) or maintaining multiple candidates (beam search). Two key controls shape the sampling distribution: temperature scaling adjusts how "peaked" or "flat" the distribution is, and top-k filtering restricts sampling to only the k most probable tokens.</span>

---

## <span style="font-size: 16px;">Temperature Scaling</span>

### <span style="font-size: 14px;">How It Works</span>

<span style="font-size: 14px;">Before applying softmax, the logits are divided by a temperature parameter $T > 0$:</span>

$$p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

### <span style="font-size: 14px;">Effect on Distribution</span>

<span style="font-size: 14px;">**Low temperature** ($T < 1$): Makes the distribution more peaked. The highest-probability tokens become even more dominant, approaching greedy decoding as $T \to 0$. Produces more predictable, repetitive text.</span>

<span style="font-size: 14px;">**High temperature** ($T > 1$): Flattens the distribution toward uniform. Lower-probability tokens get a bigger share, producing more diverse but potentially less coherent text.</span>

<span style="font-size: 14px;">**$T = 1$**: No change - uses the model's raw probabilities as-is.</span>

---

## <span style="font-size: 16px;">Top-k Filtering</span>

### <span style="font-size: 14px;">Algorithm</span>

<span style="font-size: 14px;">After computing the temperature-scaled probabilities:</span>

1. <span style="font-size: 14px;">Sort tokens by probability (descending)</span>
2. <span style="font-size: 14px;">Keep only the top $k$ tokens</span>
3. <span style="font-size: 14px;">Set all other probabilities to zero</span>
4. <span style="font-size: 14px;">Renormalize the remaining probabilities so they sum to 1</span>

### <span style="font-size: 14px;">Purpose</span>

<span style="font-size: 14px;">Top-k filtering prevents sampling from the long tail of very low-probability tokens that can produce nonsensical output. Even with high temperature, the model never selects tokens outside the top k. When $k \geq V$ (vocabulary size), no filtering occurs.</span>

---

## <span style="font-size: 16px;">Underlying Transformer</span>

### <span style="font-size: 14px;">Causal Self-Attention</span>

<span style="font-size: 14px;">The transformer uses causal (autoregressive) attention identical to GPT-style models. Each position can only attend to itself and earlier positions, enforced by masking the upper triangle of the attention score matrix to $-10^9$ before softmax. Scores are scaled by $1/\sqrt{D}$.</span>

### <span style="font-size: 14px;">Forward Pass</span>

<span style="font-size: 14px;">For a token sequence $[t_0, \ldots, t_{N-1}]$:</span>

1. <span style="font-size: 14px;">Embed: $X_i = E_{tok}[t_i] + E_{pos}[i]$</span>
2. <span style="font-size: 14px;">Attention: $Q, K, V = XW_Q, XW_K, XW_V$; causal mask; $X_1 = \text{softmax}(\text{scores}) \cdot V + X$</span>
3. <span style="font-size: 14px;">FFN: $X_2 = \text{ReLU}(X_1 W_1 + b_1) W_2 + b_2 + X_1$</span>
4. <span style="font-size: 14px;">Output: $\text{logits} = X_2[-1] \cdot W_{out} + b_{out}$</span>

---

## <span style="font-size: 16px;">Optional Downstream Sampling</span>

<span style="font-size: 14px;">This exercise returns the temperature-scaled, top-k-filtered probability distribution and does not draw a token, so no seed is required or supplied. A downstream generator can use a seeded random number generator to sample reproducibly from that distribution.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

### <span style="font-size: 14px;">What is top-p (nucleus) sampling?</span>
<span style="font-size: 14px;">Instead of a fixed count k, top-p keeps the smallest set of tokens whose cumulative probability exceeds a threshold p (e.g., 0.9). This adapts to the shape of the distribution - keeping more tokens when the distribution is flat and fewer when it is peaked.</span>

### <span style="font-size: 14px;">Can temperature and top-k be combined?</span>
<span style="font-size: 14px;">Yes, and they commonly are. Temperature is applied first (to the logits before softmax), then top-k filtering is applied to the resulting probabilities. The temperature reshapes the distribution, and top-k truncates the tail.</span>

### <span style="font-size: 14px;">When is sampling preferred over beam search?</span>
<span style="font-size: 14px;">Sampling is preferred for creative or open-ended tasks (story writing, dialogue, brainstorming) where diversity matters. Beam search tends to produce high-probability but repetitive and "boring" text. Sampling with appropriate temperature and top-k produces more natural, varied outputs.</span>

### <span style="font-size: 14px;">What is the repetition penalty?</span>
<span style="font-size: 14px;">A technique that reduces the probability of tokens that have already appeared in the generated text. It divides the logit of previously generated tokens by a penalty factor before softmax, encouraging the model to produce diverse vocabulary.</span>

---
