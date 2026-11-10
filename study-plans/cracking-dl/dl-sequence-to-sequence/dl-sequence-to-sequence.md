# <span style="font-size: 20px;">Seq2Seq with Teacher Forcing</span>

<span style="font-size: 14px;">The sequence-to-sequence (Seq2Seq) model (Sutskever et al., 2014; Cho et al., 2014) is the foundational architecture for mapping variable-length input sequences to variable-length output sequences. It powered the first neural machine translation systems and remains conceptually important as the precursor to the Transformer's encoder-decoder design.</span>

---

## <span style="font-size: 16px;">Encoder-Decoder Architecture</span>

<span style="font-size: 14px;">The model has two separate RNNs:</span>

<span style="font-size: 14px;">**Encoder**: reads the source sequence $(x_1, \dots, x_T)$ and compresses it into a fixed-size context vector - the final hidden state $(h_T, c_T)$:</span>

$$
h_t^{\text{enc}} = \text{LSTM}(\text{Embed}(x_t),\; h_{t-1}^{\text{enc}})
$$

<span style="font-size: 14px;">**Decoder**: generates the target sequence $(y_1, \dots, y_S)$ one token at a time, initialized with the encoder's final state:</span>

$$
\begin{aligned}
h_t^{\text{dec}} = \text{LSTM}(\text{Embed}(y_{t-1}),\; h_{t-1}^{\text{dec}}), \\
\hat{y}_t = \text{softmax}(W \cdot h_t^{\text{dec}} + b)
\end{aligned}
$$

<span style="font-size: 14px;">The critical bottleneck is that the entire source sequence must be compressed into a single fixed-size vector. This limitation motivated the attention mechanism (Bahdanau et al., 2015), which allows the decoder to look back at all encoder states.</span>

---

## <span style="font-size: 16px;">Teacher Forcing</span>

<span style="font-size: 14px;">During training, the decoder needs an input token at each step. Two strategies:</span>

<span style="font-size: 14px;">**With teacher forcing** (ratio = 1.0): feed the ground-truth previous token $y_{t-1}$ as input. This provides perfect context and makes training fast and stable, but creates a mismatch with inference where ground truth is unavailable.</span>

<span style="font-size: 14px;">**Without teacher forcing** (ratio = 0.0): feed the model's own prediction $\hat{y}_{t-1} = \arg\max(\text{logits})$. This matches inference behavior but early in training the predictions are poor, causing error accumulation - one wrong prediction leads to garbage input for all subsequent steps.</span>

<span style="font-size: 14px;">**Scheduled sampling** (Bengio et al., 2015): start with high teacher forcing ratio and gradually decrease it during training. This gives the model good gradient signal early while preparing it for inference-time autoregressive generation.</span>

<span style="font-size: 14px;">The exposure bias problem - the discrepancy between training (sees ground truth) and inference (sees own predictions) - remains a fundamental challenge. It motivated techniques like beam search, minimum risk training, and reinforcement learning approaches to sequence generation.</span>

---

## <span style="font-size: 16px;">The Information Bottleneck</span>

<span style="font-size: 14px;">The encoder compresses an entire source sequence into a single vector of size $H$ (the hidden dimension). For short sequences this works well, but for long sequences critical information is lost. Empirically, Seq2Seq without attention degrades significantly on sequences longer than about 20 tokens.</span>

<span style="font-size: 14px;">Sutskever et al. found that reversing the source sequence improved performance - by placing the most recent tokens closest to the decoder, the gradient path for early target tokens was shortened. This hack became unnecessary once attention was introduced.</span>

<span style="font-size: 14px;">The parameter count for a single-layer Seq2Seq model:</span>
- <span style="font-size: 14px;">Embeddings: $V_s \times E + V_t \times E$ (source and target vocabularies times embedding dimension)</span>
- <span style="font-size: 14px;">Each LSTM layer: $4H(E + H) + 8H$ for layer 0, $4H(H + H) + 8H$ for subsequent layers</span>
- <span style="font-size: 14px;">Output projection: $H \times V_t + V_t$</span>

---

## <span style="font-size: 16px;">Beam Search</span>

<span style="font-size: 14px;">At inference time, greedy decoding (always picking argmax) is suboptimal because a locally good choice may lead to a globally poor sequence. Beam search maintains the top-$k$ candidate sequences at each step:</span>

<span style="font-size: 14px;">1. Start with $k$ copies of the initial state</span>
<span style="font-size: 14px;">2. At each step, expand each beam by all vocabulary tokens, score them, and keep only the top-$k$ partial sequences</span>
<span style="font-size: 14px;">3. Stop when all beams have produced an end-of-sequence token</span>

<span style="font-size: 14px;">Beam search with $k = 4$ to $10$ typically improves BLEU scores by 1-2 points over greedy decoding. However, increasing $k$ beyond 10 often hurts quality due to the "beam search curse" - longer, more probable sequences tend to be repetitive.</span>

---

## <span style="font-size: 16px;">Practical Design Choices</span>

- <span style="font-size: 14px;">**Shared vs separate embeddings**: when source and target languages share vocabulary (or subword units), tying embedding weights reduces parameters and improves low-resource performance</span>
- <span style="font-size: 14px;">**Number of layers**: 2-4 LSTM layers is typical. Deeper models need residual connections between layers</span>
- <span style="font-size: 14px;">**Bidirectional encoder**: encoding left-to-right and right-to-left, then concatenating (or projecting) hidden states, gives the decoder richer context. Common in production NMT systems</span>
- <span style="font-size: 14px;">**Dropout**: applied between LSTM layers (not within gates). PyTorch's LSTM dropout parameter handles this automatically for multi-layer LSTMs</span>
- <span style="font-size: 14px;">**Label smoothing**: softens the one-hot target distribution during training, improving generalization. Used in nearly all modern translation systems</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: What is exposure bias and how does it affect Seq2Seq?**</span>
  <span style="font-size: 14px;">A: During training with teacher forcing, the decoder always sees ground-truth inputs. At inference, it sees its own (potentially wrong) predictions. This mismatch causes error accumulation: one wrong token corrupts all subsequent context. Scheduled sampling, where teacher forcing ratio decreases during training, partially addresses this.</span>

- <span style="font-size: 14px;">**Q: Why does the basic Seq2Seq struggle with long sequences?**</span>
  <span style="font-size: 14px;">A: The entire source must be compressed into a single fixed-size vector. For long sequences, early information gets overwritten. Attention solves this by letting the decoder access all encoder hidden states directly, creating a dynamic context vector at each step.</span>

- <span style="font-size: 14px;">**Q: How does beam search improve over greedy decoding?**</span>
  <span style="font-size: 14px;">A: Greedy decoding picks the locally best token at each step, which may lead to globally suboptimal sequences. Beam search explores multiple candidates in parallel, finding higher-probability sequences. However, it only approximates the true best sequence (which would require exponential search) and can degenerate for large beam widths.</span>

- <span style="font-size: 14px;">**Q: Why use separate encoder and decoder instead of a single model?**</span>
  <span style="font-size: 14px;">A: The source and target sequences may have different lengths, vocabularies, and structures. Separating them allows each to specialize: the encoder learns to compress, the decoder learns to generate. The interface between them (the context vector or attention) is the learned "meaning representation."</span>

- <span style="font-size: 14px;">**Q: How did this architecture evolve into the Transformer?**</span>
  <span style="font-size: 14px;">A: First, attention was added (Bahdanau, 2015) to let the decoder look at all encoder states. Then self-attention was added to both encoder and decoder. Finally, the recurrence was removed entirely (Vaswani, 2017), replacing sequential LSTM processing with parallel attention layers. The encoder-decoder structure persists in models like T5 and BART.</span>

---