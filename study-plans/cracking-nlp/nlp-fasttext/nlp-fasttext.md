# <span style="font-size: 20px;">FastText</span>

<span style="font-size: 14px;">FastText extends Word2Vec by representing each word as a bag of character n-grams. Instead of learning a single embedding per word, it learns embeddings for subword units and composes word representations by summing them. This allows FastText to generate embeddings for words never seen during training (out-of-vocabulary words) by leveraging shared morphological structure.</span>

---

## <span style="font-size: 16px;">Character N-grams</span>

<span style="font-size: 14px;">Each word is augmented with boundary markers and decomposed into character n-grams of lengths between $n_{min}$ and $n_{max}$:</span>

* <span style="font-size: 14px;">Word "cat" with boundary markers becomes <span style="font-family:monospace; font-size:13px;">\<cat\></span></span>
* <span style="font-size: 14px;">With $n_{min}=3$, $n_{max}=4$: <span style="font-family:monospace; font-size:13px;">\<ca, cat, at\>, \<cat, cat\></span></span>
* <span style="font-size: 14px;">The full padded word (e.g., <span style="font-family:monospace; font-size:13px;">\<cat\></span>) is included if its length falls within $[n_{min}, n_{max}]$</span>

<span style="font-size: 14px;">Morphologically related words like "cat" and "cats" share n-grams (<span style="font-family:monospace; font-size:13px;">\<ca, cat, \<cat</span>), allowing the model to transfer learned representations between them.</span>

---

## <span style="font-size: 16px;">Word Representation</span>

<span style="font-size: 14px;">A word's embedding is the sum of its character n-gram embeddings:</span>

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}(w)} \mathbf{z}_g$$

<span style="font-size: 14px;">where $\mathcal{G}(w)$ is the set of character n-grams for word $w$ and $\mathbf{z}_g$ is the learned embedding for n-gram $g$. This replaces the single word vector from Word2Vec with a compositional representation.</span>

---

## <span style="font-size: 16px;">Training with Skip-gram</span>

<span style="font-size: 14px;">FastText uses the same Skip-gram with Negative Sampling objective as Word2Vec, but the center word vector $\mathbf{v}_c$ is now the sum of its n-gram embeddings rather than a single learned vector:</span>

$$\mathcal{L} = -\log \sigma\left(\mathbf{u}_o \cdot \sum_{g \in \mathcal{G}(c)} \mathbf{z}_g\right) - \sum_{i=1}^{k} \log \sigma\left(-\mathbf{u}_{n_i} \cdot \sum_{g \in \mathcal{G}(c)} \mathbf{z}_g\right)$$

<span style="font-size: 14px;">Gradients flow back through the sum to each individual n-gram embedding. Since $\mathbf{v}_c = \sum_g \mathbf{z}_g$, the gradient $\partial \mathcal{L}/\partial \mathbf{z}_g = \partial \mathcal{L}/\partial \mathbf{v}_c$ for all $g$ - each n-gram receives the full center word gradient.</span>

---

## <span style="font-size: 16px;">OOV Word Handling</span>

<span style="font-size: 14px;">The key advantage of FastText over Word2Vec is handling out-of-vocabulary (OOV) words at inference time:</span>

* <span style="font-size: 14px;">Extract character n-grams from the OOV word</span>
* <span style="font-size: 14px;">Sum the embeddings of n-grams that were seen during training</span>
* <span style="font-size: 14px;">If no n-grams match, return a zero vector</span>

<span style="font-size: 14px;">For example, "cats" was never in training but shares n-grams with "cat". The composed embedding captures the morphological similarity, producing a reasonable vector for the unseen word.</span>

---

## <span style="font-size: 16px;">Comparison with Word2Vec</span>

| <span style="font-size: 14px;">Aspect</span> | <span style="font-size: 14px;">Word2Vec</span> | <span style="font-size: 14px;">FastText</span> |
|---|---|---|
| <span style="font-size: 14px;">Unit of representation</span> | <span style="font-size: 14px;">Whole word</span> | <span style="font-size: 14px;">Character n-grams</span> |
| <span style="font-size: 14px;">OOV handling</span> | <span style="font-size: 14px;">None</span> | <span style="font-size: 14px;">Compose from subword n-grams</span> |
| <span style="font-size: 14px;">Morphology</span> | <span style="font-size: 14px;">Not captured</span> | <span style="font-size: 14px;">Naturally captured via shared n-grams</span> |
| <span style="font-size: 14px;">Vocabulary size</span> | <span style="font-size: 14px;">Number of unique words</span> | <span style="font-size: 14px;">Number of unique n-grams (larger)</span> |
| <span style="font-size: 14px;">Training speed</span> | <span style="font-size: 14px;">Faster</span> | <span style="font-size: 14px;">Slower (more parameters per word)</span> |

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why character n-grams over BPE subwords?** Character n-grams are language-agnostic and require no tokenizer training. BPE is data-driven and captures more meaningful units, but needs a separate training step</span>
* <span style="font-size: 14px;">**How does FastText handle misspellings?** Misspelled words share most n-grams with the correct spelling, so their composed embeddings are similar. This makes FastText naturally robust to typos</span>
* <span style="font-size: 14px;">**Why sum n-grams instead of averaging?** Summing preserves more information about the number of matching subwords. Longer words with more n-grams get larger-magnitude embeddings, but in practice, normalization is applied downstream if needed</span>

---