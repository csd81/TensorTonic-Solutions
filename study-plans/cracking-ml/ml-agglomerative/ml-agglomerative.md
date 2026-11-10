# <span style="font-size: 20px;">Agglomerative Clustering</span>

<span style="font-size: 14px;">Agglomerative clustering is a bottom-up hierarchical clustering method. It starts with each point as its own cluster and iteratively merges the closest pair of clusters. The result is a tree-like structure called a dendrogram, which can be cut at any level to produce a flat clustering.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Initialize: each data point is its own cluster</span>
2. <span style="font-size: 14px;">Compute pairwise distances between all clusters</span>
3. <span style="font-size: 14px;">Find the two closest clusters and merge them</span>
4. <span style="font-size: 14px;">Update inter-cluster distances</span>
5. <span style="font-size: 14px;">Repeat until the desired number of clusters is reached</span>

---

## <span style="font-size: 16px;">Linkage Methods</span>

- **<span style="font-size: 14px;">Single linkage</span>**: <span style="font-size: 14px;">distance between nearest points. Can discover elongated and non-convex clusters, but is susceptible to "chaining" where clusters connect through sparse bridges</span>
- **<span style="font-size: 14px;">Complete linkage</span>**: <span style="font-size: 14px;">distance between farthest points. Produces compact, roughly equal-sized clusters, but is sensitive to outliers</span>
- **<span style="font-size: 14px;">Average linkage</span>**: <span style="font-size: 14px;">mean distance between all pairs. A compromise between single and complete, often produces good results in practice</span>

---

## <span style="font-size: 16px;">Lance-Williams Update</span>

<span style="font-size: 14px;">When merging clusters</span> $A$ <span style="font-size: 14px;">and</span> $B$ <span style="font-size: 14px;">into</span> $A \cup B$<span style="font-size: 14px;">, the distance to another cluster</span> $C$ <span style="font-size: 14px;">can be computed from existing distances using the Lance-Williams formula, avoiding recomputation from scratch:</span>

- <span style="font-size: 14px;">**Single**:</span> $d(A \cup B, C) = \min(d(A, C), d(B, C))$
- <span style="font-size: 14px;">**Complete**:</span> $d(A \cup B, C) = \max(d(A, C), d(B, C))$
- <span style="font-size: 14px;">**Average**:</span> $d(A \cup B, C) = \frac{|A| \cdot d(A, C) + |B| \cdot d(B, C)}{|A| + |B|}$

---

## <span style="font-size: 16px;">Dendrograms</span>

<span style="font-size: 14px;">The merge history forms a binary tree (dendrogram). Cutting at a height</span> $h$ <span style="font-size: 14px;">produces a flat clustering where all merges below</span> $h$ <span style="font-size: 14px;">are kept and those above are separated. This allows exploring clusterings at multiple granularities without rerunning the algorithm.</span>

---

## <span style="font-size: 16px;">Complexity</span>

- <span style="font-size: 14px;">Naive:</span> $O(n^3)$ <span style="font-size: 14px;">time,</span> $O(n^2)$ <span style="font-size: 14px;">space</span>
- <span style="font-size: 14px;">With priority queues:</span> $O(n^2 \log n)$
- <span style="font-size: 14px;">Single linkage can be done in</span> $O(n^2)$ <span style="font-size: 14px;">using the minimum spanning tree</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How do you choose the number of clusters?**</span>
  <span style="font-size: 14px;">A: Cut the dendrogram where there is a large gap in merge distances (similar to the elbow method). Inconsistency coefficient is another heuristic.</span>

- <span style="font-size: 14px;">**Q: Single vs. complete linkage?**</span>
  <span style="font-size: 14px;">A: Single linkage finds elongated clusters but chains. Complete linkage finds compact clusters but is sensitive to outliers. Ward's method (minimizes within-cluster variance) is often the best default.</span>

- <span style="font-size: 14px;">**Q: What is Ward's method?**</span>
  <span style="font-size: 14px;">A: A linkage that merges clusters to minimize the total within-cluster sum of squares. It tends to produce balanced, spherical clusters and is equivalent to K-Means in some sense.</span>

---