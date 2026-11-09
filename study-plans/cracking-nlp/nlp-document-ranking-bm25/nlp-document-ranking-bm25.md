# <span style="font-size: 20px;">Document Ranking (BM25)</span>

## Core Idea

BM25 (Best Matching 25) is the standard ranking function used in information retrieval. Given a query and a collection of documents, BM25 scores each document based on how well it matches the query, accounting for term frequency, document length, and corpus-level term importance.

## The BM25 Formula

For a query q and document d:

$$\text{BM25}(d, q) = \sum_{t \in q} IDF(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where:
- $f(t, d)$ is the frequency of term $t$ in document $d$
- $|d|$ is the document length (number of tokens)
- $\text{avgdl}$ is the average document length across the corpus
- $k_1$ and $b$ are tunable parameters

## IDF Component

The inverse document frequency for BM25 uses:

$$
IDF(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)
$$

where $N$ is the total number of documents and $n(t)$ is the number of documents containing term $t$. This differs from the standard TF-IDF formula. Terms appearing in many documents get low IDF; rare terms get high IDF.

## TF Saturation

The numerator/denominator structure creates a saturation effect: as term frequency increases, the score grows sub-linearly. The first occurrence of a term contributes the most; additional occurrences contribute progressively less. Parameter $k_1$ controls the saturation speed:
- $k_1 = 0$: TF is completely ignored (binary matching)
- $k_1 \to \infty$: no saturation (raw TF)
- Typical: $k_1 = 1.2$ to $1.5$

## Length Normalization

The $b$ parameter controls how much document length affects scoring:
- $b = 0$: no length normalization (longer documents are not penalized)
- $b = 1$: full length normalization (scores are fully normalized by document length relative to average)
- Typical: $b = 0.75$

The key insight is that BM25 normalizes by **relative** document length ($|d|/\text{avgdl}$), not absolute length. A document twice the average length is penalized the same regardless of whether the average is 50 or 500 tokens.

---

## Common Interview Follow-ups

### How does BM25 differ from TF-IDF cosine similarity?

BM25 has three key differences: (1) TF saturation prevents long documents with many term repetitions from dominating, (2) the IDF formula is different (can be negative for very common terms in some variants), and (3) length normalization uses the average document length as a reference point rather than the vector norm. BM25 generally outperforms TF-IDF cosine for ad-hoc retrieval tasks.

### Why is BM25 still used despite neural retrieval models?

BM25 is extremely efficient (inverted index lookup), requires no training data, and provides a strong baseline. Modern search pipelines use BM25 as a first-stage retriever to select candidate documents, then re-rank with neural models. This two-stage approach combines BM25's efficiency with neural models' semantic understanding.

### What happens when a query term appears in all documents?

Its IDF becomes log(0.5/(N+0.5)+1) which is close to 0 (slightly positive with the +1). This effectively ignores terms that don't discriminate between documents, which is the correct behavior: a term in every document provides no ranking signal.

### How would you tune k1 and b?

Cross-validation on relevance-labeled query-document pairs. Typical starting points: k1=1.5, b=0.75. Short documents (tweets) often benefit from lower b. Verbose queries benefit from lower k1. Some systems tune per-field (title vs body) with different parameters.

### What is BM25F?

BM25F extends BM25 to handle multiple document fields (title, body, URL) with different weights. Instead of scoring each field separately and combining, BM25F computes a weighted combination of field-level term frequencies before applying the BM25 formula. This is more principled than score-level combination.
