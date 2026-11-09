# <span style="font-size: 20px;">Sentence Embeddings</span>

<span style="font-size: 14px;">Sentence embeddings map variable-length sentences to fixed-dimensional vectors that capture semantic meaning. While word embeddings represent individual tokens, many NLP tasks (semantic search, clustering, classification) require a single vector per sentence. The challenge is aggregating token-level representations into a sentence-level representation that preserves meaning.</span>

---

## <span style="font-size: 16px;">Pooling Strategies</span>

<span style="font-size: 14px;">Given a sentence with $T$ tokens and their embeddings $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_T \in \mathbb{R}^D$, pooling reduces these to a single vector $\mathbf{s} \in \mathbb{R}^D$:</span>

### <span style="font-size: 14px;">Mean Pooling</span>

$$\mathbf{s} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{e}_t$$

<span style="font-size: 14px;">The simplest and most widely used approach. Treats all tokens equally. Works surprisingly well in practice and is the default in Sentence-BERT and many production systems.</span>

### <span style="font-size: 14px;">Max Pooling</span>

$$s_d = \max_{t=1}^{T} e_{t,d} \quad \text{for each dimension } d$$

<span style="font-size: 14px;">Takes the maximum value across all tokens for each dimension independently. Captures the most prominent feature in each dimension. Useful when specific keywords matter more than overall meaning.</span>

### <span style="font-size: 14px;">Weighted Mean Pooling</span>

$$\mathbf{s} = \frac{\sum_{t=1}^{T} w_t \, \mathbf{e}_t}{\sum_{t=1}^{T} w_t}$$

<span style="font-size: 14px;">Weight each token's contribution differently. Common weighting schemes include IDF weights (downweight frequent words like "the", "is") and SIF (Smooth Inverse Frequency) weights $w_t = a / (a + p_t)$ where $p_t$ is the word's unigram probability and $a$ is a hyperparameter (typically $a = 10^{-3}$).</span>

---

## <span style="font-size: 16px;">SIF Weighting</span>

<span style="font-size: 14px;">Smooth Inverse Frequency (SIF) is a principled weighting scheme from Arora et al. (2017). The intuition: frequent words carry less discriminative information. Given a corpus, compute unigram probabilities $p_w$ for each word, then weight each token by:</span>

$$w_t = \frac{a}{a + p(w_t)}$$

<span style="font-size: 14px;">where $a$ is a smoothing constant. Rare words get weight close to 1, while common words get weight close to $a / (a + 1) \approx a$ (very small when $a \ll 1$). This simple scheme outperforms unweighted mean pooling on most sentence similarity benchmarks.</span>

---

## <span style="font-size: 16px;">Computing Word Frequencies</span>

<span style="font-size: 14px;">To compute SIF weights, you need unigram probabilities. Given a corpus of sentences:</span>

* <span style="font-size: 14px;">Count total occurrences of each word across all sentences</span>
* <span style="font-size: 14px;">Divide each count by the total number of tokens to get $p(w)$</span>
* <span style="font-size: 14px;">Words absent from the embedding vocabulary still use their observed corpus frequencies; their zero vectors contribute zero to the numerator, but their weights remain in the weighted-mean denominator.</span>

---

## <span style="font-size: 16px;">Implementation Considerations</span>

* <span style="font-size: 14px;">**Embedding lookup**: Map each token to its embedding vector. Unknown tokens (OOV) should use a zero vector</span>
* <span style="font-size: 14px;">**Empty sentences**: Return a zero vector for sentences with no tokens</span>
* <span style="font-size: 14px;">**Normalization**: In production systems, sentence embeddings are often L2-normalized before computing cosine similarity, but the raw pooled vector is the standard output</span>

---

## <span style="font-size: 16px;">Comparison of Pooling Methods</span>

| <span style="font-size: 14px;">Method</span> | <span style="font-size: 14px;">Strengths</span> | <span style="font-size: 14px;">Weaknesses</span> |
|---|---|---|
| <span style="font-size: 14px;">Mean</span> | <span style="font-size: 14px;">Simple, robust baseline</span> | <span style="font-size: 14px;">Diluted by stop words</span> |
| <span style="font-size: 14px;">Max</span> | <span style="font-size: 14px;">Captures salient features</span> | <span style="font-size: 14px;">Loses positional/frequency info</span> |
| <span style="font-size: 14px;">Weighted Mean</span> | <span style="font-size: 14px;">Downweights noise words</span> | <span style="font-size: 14px;">Requires corpus statistics</span> |

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Mean pooling vs [CLS] token?** Mean pooling averages all token embeddings. [CLS] uses a single special token's representation. For BERT without fine-tuning, mean pooling outperforms [CLS]; with fine-tuning, [CLS] can be better since it is explicitly trained as a sentence representation</span>
* <span style="font-size: 14px;">**Why does SIF work?** SIF approximates removing the first principal component of the embedding space, which typically captures word frequency rather than meaning. Downweighting frequent words achieves a similar effect</span>
* <span style="font-size: 14px;">**When would max pooling outperform mean?** In tasks where specific keywords or phrases are strong signals (e.g., sentiment analysis where one strong word like "terrible" dominates). Mean pooling is better for tasks requiring holistic sentence understanding</span>

---