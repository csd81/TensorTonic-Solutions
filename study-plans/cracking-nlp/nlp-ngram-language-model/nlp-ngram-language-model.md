# <span style="font-size: 20px;">N-gram Language Model</span>

<span style="font-size: 14px;">An n-gram language model estimates the probability of the next word given the previous $n-1$ words. It is the simplest language model and forms the foundation for understanding more complex approaches like neural language models. Despite its simplicity, n-gram models remain useful for spelling correction, speech recognition, and as baselines in NLP research.</span>

---

## <span style="font-size: 16px;">The Chain Rule and Markov Assumption</span>

<span style="font-size: 14px;">The probability of a sentence $w_1, w_2, \ldots, w_T$ can be decomposed using the chain rule:</span>

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

<span style="font-size: 14px;">The Markov assumption truncates the history to the previous $n-1$ words:</span>

$$P(w_t \mid w_1, \ldots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

---

## <span style="font-size: 16px;">N-gram Models</span>

* <span style="font-size: 14px;">**Unigram** ($n=1$): $P(w_t) = \frac{\text{count}(w_t)}{\sum_w \text{count}(w)}$. Ignores all context</span>
* <span style="font-size: 14px;">**Bigram** ($n=2$): $P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$. Conditions on the previous word</span>
* <span style="font-size: 14px;">**Trigram** ($n=3$): $P(w_t \mid w_{t-2}, w_{t-1}) = \frac{\text{count}(w_{t-2}, w_{t-1}, w_t)}{\text{count}(w_{t-2}, w_{t-1})}$. Conditions on the previous two words</span>

<span style="font-size: 14px;">In general, for an n-gram model:</span>

$$P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = \frac{\text{count}(w_{t-n+1}, \ldots, w_t)}{\text{count}(w_{t-n+1}, \ldots, w_{t-1})}$$

---

## <span style="font-size: 16px;">Start and End Tokens</span>

<span style="font-size: 14px;">To model sentence boundaries, special tokens are added:</span>

* <span style="font-size: 14px;">**Start token** (<span style="font-family:monospace; font-size:13px;">\<s\></span>): Prepended $n-1$ times to mark the beginning. For a bigram model, one <span style="font-family:monospace; font-size:13px;">\<s\></span> is added; for a trigram, two are added</span>
* <span style="font-size: 14px;">**End token** (<span style="font-family:monospace; font-size:13px;">\</s\></span>): Appended once to mark the end of a sentence</span>

<span style="font-size: 14px;">These tokens allow the model to learn which words likely start or end sentences. For example, $P(\text{"The"} \mid \text{\<s\>})$ captures the probability of "The" being the first word.</span>

---

## <span style="font-size: 16px;">Laplace (Add-1) Smoothing</span>

<span style="font-size: 14px;">Without smoothing, any unseen n-gram has zero probability, which makes entire sentences have zero probability. Laplace smoothing adds 1 to every n-gram count:</span>

$$P_{\text{Laplace}}(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = \frac{\text{count}(w_{t-n+1}, \ldots, w_t) + 1}{\text{count}(w_{t-n+1}, \ldots, w_{t-1}) + V}$$

<span style="font-size: 14px;">where $V$ is the vocabulary size (number of unique word types including <span style="font-family:monospace; font-size:13px;">\</s\></span> but not <span style="font-family:monospace; font-size:13px;">\<s\></span>). The denominator adds $V$ because we add 1 to each of the $V$ possible next words, so the counts must sum correctly.</span>

---

## <span style="font-size: 16px;">Sentence Probability</span>

<span style="font-size: 14px;">To compute the probability of a sentence, pad it with start/end tokens, then multiply the conditional probabilities of each token given its context:</span>

$$P(s) = \prod_{t=1}^{T+1} P_{\text{Laplace}}(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

<span style="font-size: 14px;">where $w_{T+1}$ is the end token. In practice, work with log probabilities to avoid numerical underflow:</span>

$$\log P(s) = \sum_{t=1}^{T+1} \log P_{\text{Laplace}}(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

---

## <span style="font-size: 16px;">Text Generation</span>

<span style="font-size: 14px;">To generate text from an n-gram model:</span>

* <span style="font-size: 14px;">Start with $n-1$ start tokens as context</span>
* <span style="font-size: 14px;">At each step, compute the probability distribution over all vocabulary words given the current context</span>
* <span style="font-size: 14px;">Sample the next word from this distribution (or take the argmax for greedy generation)</span>
* <span style="font-size: 14px;">Update the context window and repeat until the end token is generated or a maximum length is reached</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why not just use higher-order n-grams?** The number of possible n-grams grows exponentially with $n$ as $V^n$. Most n-grams will never be observed in the training data, leading to severe sparsity. In practice, trigrams are often the practical limit without massive corpora</span>
* <span style="font-size: 14px;">**Laplace vs Kneser-Ney smoothing?** Laplace smoothing is simple but distributes too much probability mass to unseen events. Kneser-Ney uses absolute discounting and a continuation probability that considers how many different contexts a word appears in, producing much better estimates</span>
* <span style="font-size: 14px;">**How do n-gram models relate to neural LMs?** Both model $P(w_t \mid \text{context})$, but n-gram models use explicit count tables while neural models learn continuous representations. Neural models generalize better to unseen contexts through shared parameters, but n-gram models are faster to train and more interpretable</span>

---