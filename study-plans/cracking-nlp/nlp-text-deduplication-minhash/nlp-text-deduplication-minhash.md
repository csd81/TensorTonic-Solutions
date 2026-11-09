# <span style="font-size: 20px;">Text Deduplication (MinHash)</span>

---

## <span style="font-size: 16px;">Core Concept</span>

<span style="font-size: 14px;">MinHash is a locality-sensitive hashing technique that efficiently estimates the Jaccard similarity between sets. In text deduplication, each document is represented as a set of character n-grams (shingles), and MinHash signatures are computed to quickly identify near-duplicate document pairs without comparing every shingle directly.</span>

---

## <span style="font-size: 16px;">The MinHash Pipeline</span>

### <span style="font-size: 14px;">Step 1: Shingling</span>

<span style="font-size: 14px;">Convert each document into a set of overlapping character n-grams. For a document "hello" with shingle size 3, the shingles are {"hel", "ell", "llo"}. Shingles capture local character patterns, so documents with similar text will share many shingles.</span>

### <span style="font-size: 14px;">Step 2: Hash Functions</span>

<span style="font-size: 14px;">Define $k$ hash functions of the form $h(x) = (ax + b) \bmod p$ where $p$ is a large prime (e.g., $2^{31}-1$), and $a, b$ are random coefficients. Each shingle string is first converted to an integer using a deterministic polynomial hash, then the $k$ hash functions are applied.</span>

### <span style="font-size: 14px;">Step 3: MinHash Signature</span>

<span style="font-size: 14px;">For each document and each hash function $h_k$, the MinHash value is the minimum hash value over all shingles in the document:</span>

$$
\text{sig}_k(D) = \min_{s \in D} h_k(s)
$$

<span style="font-size: 14px;">The signature is a vector of $k$ MinHash values. This compact representation (typically 50-200 integers) replaces the potentially large shingle set.</span>

### <span style="font-size: 14px;">Step 4: Similarity Estimation</span>

<span style="font-size: 14px;">The estimated Jaccard similarity between two documents is the fraction of hash positions where their signatures agree:</span>

$$
\hat{J}(A, B) = \frac{|\{k : \text{sig}_k(A) = \text{sig}_k(B)\}|}{k}
$$

<span style="font-size: 14px;">By the MinHash theorem, $\Pr[\min h(A) = \min h(B)] = J(A, B)$, so this estimate is unbiased and concentrates around the true Jaccard similarity as $k$ increases.</span>

---

## <span style="font-size: 16px;">Why MinHash Works</span>

<span style="font-size: 14px;">Consider a random permutation of all possible shingles. The minimum element in each set under this permutation depends on the union of the two sets. The probability that both sets have the same minimum equals the fraction of the union that is in the intersection - which is exactly the Jaccard similarity $|A \cap B| / |A \cup B|$. Hash functions approximate random permutations.</span>

---

## <span style="font-size: 16px;">Greedy Deduplication</span>

<span style="font-size: 14px;">After computing pairwise similarities, deduplication proceeds greedily: iterate through documents in order. For each document, if its estimated similarity to any earlier non-duplicate document meets the threshold, mark it as a duplicate of that earlier document. Keep only non-duplicate documents.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">1. **Time complexity**: Without MinHash, comparing N documents with S average shingles takes $O(N^2 \cdot S)$. With MinHash, signature computation is $O(N \cdot S \cdot k)$ and comparison is $O(N^2 \cdot k)$, where $k \ll S$.</span>

<span style="font-size: 14px;">2. **LSH banding**: For large N, even $O(N^2)$ pairwise comparisons are expensive. Locality-Sensitive Hashing (LSH) groups signature rows into bands and hashes each band, so only documents that collide in at least one band are compared. This reduces comparisons to near-linear time.</span>

<span style="font-size: 14px;">3. **Shingle size trade-off**: Small shingles (2-3 chars) are more likely to match across unrelated documents, reducing precision. Large shingles (8-10 chars) require near-exact overlap, reducing recall. Character shingles of size 4-5 or word shingles of size 2-3 are common choices.</span>

<span style="font-size: 14px;">4. **MinHash vs SimHash**: MinHash estimates Jaccard similarity (set overlap). SimHash estimates cosine similarity (vector angle). MinHash is better for sparse set comparison; SimHash is better for dense vector comparison.</span>

<span style="font-size: 14px;">5. **Weighted MinHash**: For TF-IDF or other weighted representations, weighted MinHash generalizes the technique to bags (multisets) where element frequency matters.</span>

---