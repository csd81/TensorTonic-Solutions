# <span style="font-size: 20px;">Masked Language Modeling - Theory</span>

---

## <span style="font-size: 16px;">Overview</span>

<span style="font-size: 14px;">Masked Language Modeling (MLM) is the pre-training objective used by BERT. A percentage of input tokens are replaced with a special [MASK] token, and the model must predict the original tokens at those masked positions. Unlike autoregressive models that predict left-to-right, MLM allows bidirectional context - the model sees tokens on both sides of the mask when making predictions.</span>

---

## <span style="font-size: 16px;">1. Masking Strategy</span>

<span style="font-size: 14px;">Given a sequence of token IDs and a list of positions to mask:</span>

<span style="font-size: 14px;">1. Record the original token at each masked position (these are the labels)</span>
<span style="font-size: 14px;">2. Replace those positions with the designated mask_token_id in the input</span>

<span style="font-size: 14px;">The model never sees the original tokens at masked positions - it must infer them from surrounding context.</span>

---

## <span style="font-size: 16px;">2. Bidirectional Encoding</span>

<span style="font-size: 14px;">The masked sequence is processed through a single-layer transformer encoder (identical to BERT's architecture):</span>

### <span style="font-size: 14px;">Embeddings</span>

$$X_i = E_{token}[\tilde{t}_i] + E_{position}[i]$$

<span style="font-size: 14px;">where $\tilde{t}_i$ is the token ID after masking.</span>

### <span style="font-size: 14px;">Bidirectional Self-Attention</span>

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
$$\text{attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V$$
$$X_1 = \text{attn} + X$$

### <span style="font-size: 14px;">FFN + Residual</span>

$$X_2 = \text{ReLU}(X_1 W_1 + b_1) W_2 + b_2 + X_1$$

---

## <span style="font-size: 16px;">3. Prediction at Masked Positions</span>

<span style="font-size: 14px;">For each masked position $m$, the model projects the representation to vocabulary logits:</span>

$$\text{logits}_m = X_2[m] \cdot W_{out} + b_{out}$$
$$\text{probs}_m = \text{softmax}(\text{logits}_m)$$
$$\text{predicted}_m = \arg\max(\text{logits}_m)$$

<span style="font-size: 14px;">Only masked positions are predicted - the model does not need to predict unmasked tokens.</span>

---

## <span style="font-size: 16px;">4. Accuracy and Loss</span>

<span style="font-size: 14px;">For each masked position, we compute the cross-entropy loss:</span>

$$\text{loss}_m = -\log(\text{probs}_m[\text{label}_m])$$

<span style="font-size: 14px;">The total MLM loss is the mean across all masked positions. Accuracy counts how many masked positions were predicted correctly.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">**Q: Why mask 15% of tokens in BERT's original paper?**</span>
<span style="font-size: 14px;">If too few tokens are masked, training is slow because there are few prediction targets per sequence. If too many are masked, the model has too little context to make good predictions. 15% was found empirically to balance training efficiency and context availability.</span>

<span style="font-size: 14px;">**Q: In the original BERT, why are not all selected tokens replaced with [MASK]?**</span>
<span style="font-size: 14px;">BERT replaces 80% of selected tokens with [MASK], 10% with a random token, and 10% unchanged. This reduces the mismatch between pre-training (which has [MASK] tokens) and fine-tuning (which does not). In this simplified version, we mask all selected positions.</span>

<span style="font-size: 14px;">**Q: How does MLM differ from causal language modeling?**</span>
<span style="font-size: 14px;">Causal LM (GPT-style) predicts each token using only left context. MLM predicts masked tokens using both left and right context. This makes MLM better for understanding tasks but not directly usable for generation, since the model expects to see some of the "future" tokens.</span>

<span style="font-size: 14px;">**Q: What is whole-word masking?**</span>
<span style="font-size: 14px;">When using subword tokenization, masking individual subword pieces can be too easy (the model just looks at adjacent pieces of the same word). Whole-word masking masks all subword tokens of a selected word together, forcing the model to use broader context.</span>

---
