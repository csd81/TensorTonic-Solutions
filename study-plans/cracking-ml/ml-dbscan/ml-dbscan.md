# <span style="font-size: 20px;">DBSCAN</span>

<span style="font-size: 14px;">DBSCAN (Ester et al., 1996) is a density-based clustering algorithm that can discover clusters of arbitrary shape and automatically identifies noise points. It is one of the most cited algorithms in data mining.</span>

---

## <span style="font-size: 16px;">Key Concepts</span>

- **<span style="font-size: 14px;">Core point</span>**: <span style="font-size: 14px;">has at least</span> `min_samples` <span style="font-size: 14px;">points within distance</span> $\varepsilon$ <span style="font-size: 14px;">(including itself)</span>
- **<span style="font-size: 14px;">Border point</span>**: <span style="font-size: 14px;">not a core point but within</span> $\varepsilon$ <span style="font-size: 14px;">of a core point</span>
- **<span style="font-size: 14px;">Noise point</span>**: <span style="font-size: 14px;">neither core nor border, labeled -1</span>
- **<span style="font-size: 14px;">Density-reachable</span>**: <span style="font-size: 14px;">point</span> $q$ <span style="font-size: 14px;">is density-reachable from</span> $p$ <span style="font-size: 14px;">if there is a chain of core points connecting them</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">For each unvisited point</span> $p$<span style="font-size: 14px;">:</span>
   - <span style="font-size: 14px;">Mark</span> $p$ <span style="font-size: 14px;">as visited</span>
   - <span style="font-size: 14px;">Find all neighbors within</span> $\varepsilon$
   - <span style="font-size: 14px;">If</span> $|\text{neighbors}| < \texttt{min\_samples}$<span style="font-size: 14px;">, mark as noise</span>
   - <span style="font-size: 14px;">Otherwise, start a new cluster and expand:</span>
     - <span style="font-size: 14px;">Add all neighbors to the cluster</span>
     - <span style="font-size: 14px;">For each unvisited neighbor, find its neighbors. If it is also a core point, add its neighbors to the expansion queue</span>
     - <span style="font-size: 14px;">Continue until no more points can be added</span>

---

## <span style="font-size: 16px;">Choosing Parameters</span>

- $\varepsilon$<span style="font-size: 14px;">: the neighborhood radius. Use a k-distance plot (sort k-nearest-neighbor distances) and look for an elbow</span>
- `min_samples`<span style="font-size: 14px;">: rule of thumb is</span> $d + 1$ <span style="font-size: 14px;">or</span> $2d$ <span style="font-size: 14px;">where</span> $d$ <span style="font-size: 14px;">is the dimensionality. Higher values produce more conservative clusters</span>

---

## <span style="font-size: 16px;">Advantages</span>

- <span style="font-size: 14px;">Does not require specifying</span> $k$ <span style="font-size: 14px;">(number of clusters)</span>
- <span style="font-size: 14px;">Can find clusters of arbitrary shape (non-convex, elongated, etc.)</span>
- <span style="font-size: 14px;">Robust to outliers (they become noise)</span>
- <span style="font-size: 14px;">Deterministic for core points (border point assignment can vary with processing order)</span>

---

## <span style="font-size: 16px;">Limitations</span>

- <span style="font-size: 14px;">Struggles with clusters of varying density (a single</span> $\varepsilon$ <span style="font-size: 14px;">cannot fit both dense and sparse clusters)</span>
- <span style="font-size: 14px;">Performance degrades in high dimensions (distances become less meaningful)</span>
- <span style="font-size: 14px;">Naive implementation is</span> $O(n^2)$<span style="font-size: 14px;">; with spatial indexing (KD-tree) it can be</span> $O(n \log n)$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How does DBSCAN handle varying density?**</span>
  <span style="font-size: 14px;">A: Poorly with a single epsilon. HDBSCAN extends DBSCAN to handle varying densities by building a hierarchy of clusterings and extracting the most stable clusters.</span>

- <span style="font-size: 14px;">**Q: What is OPTICS?**</span>
  <span style="font-size: 14px;">A: An extension that produces an ordering of points augmented with reachability distances, from which clusterings at multiple density levels can be extracted.</span>

- <span style="font-size: 14px;">**Q: How do you scale DBSCAN?**</span>
  <span style="font-size: 14px;">A: Use spatial indexing structures like KD-trees or ball trees for neighbor queries. For very large datasets, use approximate methods or distributed implementations.</span>

---