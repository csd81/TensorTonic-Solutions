# <span style="font-size: 20px;">Next Sentence Prediction - Theory</span>

---

## <span style="font-size: 16px;">Overview</span>

<span style="font-size: 14px;">Next Sentence Prediction (NSP) is one of BERT's two pre-training objectives (alongside MLM). Given two segments A and B concatenated into a single sequence, the model predicts whether B is the actual next sentence following A in the original text (IsNext) or a random sentence (NotNext). The [CLS] token's representation is used as a pooled representation for this binary classification.</span>

---

## <span style="font-size: 16px;">1. Input Construction</span>

<span style="font-size: 14px;">The input sequence is formed by concatenating segment A and segment B. Each token gets three embeddings:</span>

$$X_i = E_{token}[t_i] + E_{position}[i] + E_{segment}[s_i]$$

<span style="font-size: 14px;">where $s_i = 0$ for tokens in segment A and $s_i = 1$ for tokens in segment B. The first token (position 0) is typically a [CLS] token used for classification.</span>

---

## <span style="font-size: 16px;">2. Bidirectional Encoding</span>

<span style="font-size: 14px;">The full concatenated sequence is encoded with bidirectional self-attention (no causal mask):</span>

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
$$\text{attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V$$
$$X_1 = \text{attn} + X$$
$$X_2 = \text{FFN}(X_1) + X_1$$

<span style="font-size: 14px;">where $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$</span>

---

## <span style="font-size: 16px;">3. [CLS] Pooling</span>

<span style="font-size: 14px;">BERT uses the representation at position 0 (the [CLS] token) as a summary of the entire sequence. Before classification, this representation is passed through a pooling layer with tanh activation:</span>

$$h_{pool} = \tanh(X_2[0] \cdot W_{pool} + b_{pool})$$

<span style="font-size: 14px;">The tanh squashes the representation to $[-1, 1]$, acting as a non-linear bottleneck that encourages the model to compress sequence-level information into a fixed-size vector.</span>

---

## <span style="font-size: 16px;">4. Binary Classification</span>

<span style="font-size: 14px;">The pooled representation is projected to 2 logits (IsNext vs NotNext):</span>

$$\text{logits} = h_{pool} \cdot W_{cls} + b_{cls}$$
$$\text{probs} = \text{softmax}(\text{logits})$$
$$\text{prediction} = \arg\max(\text{logits})$$

<span style="font-size: 14px;">Class 0 typically means "IsNext" (the sentences are consecutive) and class 1 means "NotNext" (random pairing).</span>

---

## <span style="font-size: 16px;">5. Cross-Entropy Loss</span>

<span style="font-size: 14px;">Given the true label (0 or 1):</span>

$$\text{loss} = -\log(\text{probs}[\text{label}])$$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">**Q: Why was NSP later dropped in models like RoBERTa?**</span>
<span style="font-size: 14px;">Research showed NSP provides minimal benefit and may even hurt performance. The task is too easy - the model can often distinguish random pairs just from topic mismatch rather than learning coherence. RoBERTa showed that removing NSP and training with longer sequences improved results.</span>

<span style="font-size: 14px;">**Q: What is the role of segment embeddings?**</span>
<span style="font-size: 14px;">Segment embeddings tell the model which tokens belong to which sentence. Without them, the model cannot distinguish where segment A ends and segment B begins. This is critical for tasks like NSP, question answering, and natural language inference where two segments interact.</span>

<span style="font-size: 14px;">**Q: Why use tanh pooling instead of directly classifying from X_2[0]?**</span>
<span style="font-size: 14px;">The tanh pooling layer adds a learnable non-linear transformation that can adapt the [CLS] representation for the specific classification task. During pre-training, this layer learns to extract sentence-level features. During fine-tuning, it can be further adapted.</span>

<span style="font-size: 14px;">**Q: How is the training data for NSP constructed?**</span>
<span style="font-size: 14px;">50% of the time, B is the actual next sentence after A (label=0, IsNext). 50% of the time, B is a random sentence from the corpus (label=1, NotNext). This balanced dataset ensures the model cannot succeed by always predicting one class.</span>

---
