# <span style="font-size: 20px;">Embedding Similarity</span>

<span style="font-size: 14px;">Measuring similarity between embeddings is fundamental to NLP applications like semantic search, clustering, deduplication, and recommendation systems. Given two vectors in the same embedding space, different distance and similarity metrics capture different notions of "closeness". Choosing the right metric depends on whether you care about direction, magnitude, or both.</span>

---

## <span style="font-size: 16px;">Cosine Similarity</span>

<span style="font-size: 14px;">Measures the cosine of the angle between two vectors, capturing directional similarity independent of magnitude:</span>

$$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|} = \frac{\sum_{i} a_i b_i}{\sqrt{\sum_{i} a_i^2} \, \sqrt{\sum_{i} b_i^2}}$$

<span style="font-size: 14px;">Range: $[-1, 1]$. Value of 1 means identical direction, 0 means orthogonal, -1 means opposite direction. This is the most widely used similarity metric for text embeddings because it is invariant to vector scaling - a document mentioned twice does not become "more similar" just because its embedding is doubled.</span>

---

## <span style="font-size: 16px;">Dot Product</span>

<span style="font-size: 14px;">The unnormalized inner product of two vectors:</span>

$$\text{dot}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i$$

<span style="font-size: 14px;">Range: $(-\infty, +\infty)$. Unlike cosine similarity, the dot product is sensitive to vector magnitude. Two vectors can have high dot product either because they point in similar directions or because they have large magnitudes. This is useful when magnitude carries meaning (e.g., document importance or confidence).</span>

---

## <span style="font-size: 16px;">Euclidean Distance</span>

<span style="font-size: 14px;">The L2 distance between two vectors in the embedding space:</span>

$$\text{euclidean}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_{i} (a_i - b_i)^2}$$

<span style="font-size: 14px;">Range: $[0, +\infty)$. Lower values mean more similar. Euclidean distance measures the straight-line distance between two points. It is sensitive to both direction and magnitude, making it most appropriate when embeddings are normalized to unit length (where it becomes monotonically related to cosine similarity).</span>

---

## <span style="font-size: 16px;">Nearest Neighbor Search</span>

<span style="font-size: 14px;">Given a query embedding and a corpus of candidate embeddings, find the $k$ most similar candidates. The brute-force approach computes the similarity between the query and every candidate, then returns the top $k$:</span>

* <span style="font-size: 14px;">For cosine similarity and dot product: sort in descending order (higher is more similar)</span>
* <span style="font-size: 14px;">For Euclidean distance: sort in ascending order (lower is more similar)</span>

<span style="font-size: 14px;">Brute-force search has $O(n \cdot d)$ time complexity for $n$ candidates of dimension $d$. For large-scale retrieval, approximate methods (FAISS, Annoy, HNSW) trade exactness for speed.</span>

---

## <span style="font-size: 16px;">Relationship Between Metrics</span>

<span style="font-size: 14px;">For unit-normalized vectors ($\|\mathbf{a}\| = \|\mathbf{b}\| = 1$):</span>

* <span style="font-size: 14px;">Cosine similarity equals the dot product: $\text{cosine}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b}$</span>
* <span style="font-size: 14px;">Euclidean distance relates to cosine similarity: $\|\mathbf{a} - \mathbf{b}\|^2 = 2(1 - \text{cosine}(\mathbf{a}, \mathbf{b}))$</span>
* <span style="font-size: 14px;">Ranking by cosine, dot product, or negative Euclidean distance all produce the same order</span>

<span style="font-size: 14px;">This is why many production systems L2-normalize embeddings at indexing time: it makes all three metrics equivalent and allows using the fastest one (dot product).</span>

---

## <span style="font-size: 16px;">Edge Cases</span>

* <span style="font-size: 14px;">**Zero vectors**: Cosine similarity is undefined when either vector has zero norm. Convention is to return 0.0</span>
* <span style="font-size: 14px;">**Identical vectors**: Cosine similarity is 1.0, Euclidean distance is 0.0</span>
* <span style="font-size: 14px;">**High dimensions**: In very high-dimensional spaces, distances between random points concentrate around the same value (curse of dimensionality), making nearest neighbor search less discriminative</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**When to use cosine vs dot product?** Use cosine when embeddings have variable magnitudes and you only care about semantic direction. Use dot product when magnitude is meaningful (e.g., Maximum Inner Product Search in recommendation systems where magnitude encodes item popularity)</span>
* <span style="font-size: 14px;">**How to speed up nearest neighbor search?** Approximate nearest neighbor (ANN) algorithms like HNSW, IVF, or product quantization. They build index structures that prune the search space, achieving sub-linear query time at the cost of occasionally missing the true nearest neighbor</span>
* <span style="font-size: 14px;">**Why normalize embeddings?** Normalization makes cosine similarity equivalent to dot product, which is faster to compute (no division). It also makes Euclidean distance monotonically related to cosine similarity, so any metric gives the same ranking</span>

---