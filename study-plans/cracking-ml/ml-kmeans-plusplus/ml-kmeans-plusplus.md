# <span style="font-size: 20px;">K-Means++ Initialization</span>

<span style="font-size: 14px;">K-Means++ (Arthur and Vassilvitskii, 2007) is an initialization method for K-Means that provides a provably better starting point than random initialization. It selects initial centroids that are well-spread across the data.</span>

---

## <span style="font-size: 16px;">The Problem with Random Init</span>

<span style="font-size: 14px;">Standard K-Means picks</span> $k$ <span style="font-size: 14px;">random points as initial centroids. If two centroids start in the same cluster, the algorithm may converge to a poor local minimum. K-Means++ addresses this by ensuring centroids are spread apart.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Choose the first centroid</span> $c_1$ <span style="font-size: 14px;">uniformly at random from the data points</span>
2. <span style="font-size: 14px;">For</span> $i = 2, \ldots, k$<span style="font-size: 14px;">:</span>
   - <span style="font-size: 14px;">For each point</span> $x$<span style="font-size: 14px;">, compute</span> $D(x) = \min_{c \in \{c_1, \ldots, c_{i-1}\}} \|x - c\|$
   - <span style="font-size: 14px;">Choose the next centroid</span> $c_i$ <span style="font-size: 14px;">with probability</span> $P(x) = \frac{D(x)^2}{\sum_j D(x_j)^2}$

<span style="font-size: 14px;">Points that are far from all existing centroids are more likely to be chosen, ensuring good spread.</span>

---

## <span style="font-size: 16px;">Theoretical Guarantee</span>

<span style="font-size: 14px;">K-Means++ guarantees that the expected value of the K-Means objective after initialization is at most</span> $O(\log k)$ <span style="font-size: 14px;">times the optimal. In contrast, random initialization provides no such guarantee and can be arbitrarily bad.</span>

---

## <span style="font-size: 16px;">Why D-squared Weighting?</span>

- <span style="font-size: 14px;">Points far from existing centroids are likely in under-represented regions</span>
- <span style="font-size: 14px;">Squared distance (not linear) gives a stronger preference for distant points</span>
- <span style="font-size: 14px;">This is not deterministic - randomness prevents pathological cases where the farthest point is an outlier</span>

---

## <span style="font-size: 16px;">Practical Impact</span>

- <span style="font-size: 14px;">Typically leads to fewer iterations of K-Means</span>
- <span style="font-size: 14px;">Produces better final clusterings on average</span>
- <span style="font-size: 14px;">The initialization cost is</span> $O(n \cdot k \cdot d)$<span style="font-size: 14px;">, which is dominated by the K-Means iterations themselves</span>
- <span style="font-size: 14px;">Used as the default initialization in scikit-learn's KMeans</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why squared distance instead of linear?**</span>
  <span style="font-size: 14px;">A: Squared distance gives more weight to far-away points, making it more likely to pick points from different clusters. Linear distance would not spread centroids as effectively.</span>

- <span style="font-size: 14px;">**Q: Can K-Means++ still produce bad initializations?**</span>
  <span style="font-size: 14px;">A: Yes, it is still randomized. But the probability of a bad initialization is much lower. Running multiple times and keeping the best is still recommended.</span>

- <span style="font-size: 14px;">**Q: What is K-Means||?**</span>
  <span style="font-size: 14px;">A: A scalable variant that selects multiple candidates per round in parallel, reducing the number of passes over the data. Useful for very large datasets.</span>

---