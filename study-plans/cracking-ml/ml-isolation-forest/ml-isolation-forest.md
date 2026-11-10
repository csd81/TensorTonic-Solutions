# <span style="font-size: 20px;">Isolation Forest</span>

<span style="font-size: 14px;">Isolation Forest (Liu, Ting, and Zhou, 2008) is an unsupervised anomaly detection algorithm based on a simple principle: anomalies are few and different, so they are easier to isolate than normal points.</span>

---

## <span style="font-size: 16px;">Key Insight</span>

<span style="font-size: 14px;">Random partitions isolate anomalies quickly (short path) because they sit in sparse regions. Normal points require many splits to isolate because they are surrounded by similar points. This is analogous to how binary search trees give shorter paths for extreme values.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

### <span style="font-size: 16px;">Training Phase</span>

1. <span style="font-size: 14px;">For each of</span> $T$ <span style="font-size: 14px;">trees:</span>
   - <span style="font-size: 14px;">Sample</span> $\psi$ <span style="font-size: 14px;">points without replacement</span>
   - <span style="font-size: 14px;">Build an isolation tree by recursively:</span>
     - <span style="font-size: 14px;">Randomly selecting a feature</span>
     - <span style="font-size: 14px;">Randomly selecting a split value between min and max</span>
     - <span style="font-size: 14px;">Partitioning into left (< split) and right (>= split)</span>
   - <span style="font-size: 14px;">Stop when depth reaches</span> $\lceil \log_2 \psi \rceil$ <span style="font-size: 14px;">or node has</span> $\leq 1$ <span style="font-size: 14px;">points</span>

### <span style="font-size: 16px;">Scoring Phase</span>

2. <span style="font-size: 14px;">For each point, compute the average path length across all trees</span>
3. <span style="font-size: 14px;">Normalize by</span> $c(\psi)$<span style="font-size: 14px;">, the expected path length in a balanced BST</span>

---

## <span style="font-size: 16px;">Path Length Normalization</span>

<span style="font-size: 14px;">The function</span> $c(n)$ <span style="font-size: 14px;">approximates the average path length of an unsuccessful search in a Binary Search Tree with</span> $n$ <span style="font-size: 14px;">nodes:</span>

$$
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
$$

<span style="font-size: 14px;">where</span> $H(k) = \ln(k) + \gamma$ <span style="font-size: 14px;">(</span>$\gamma \approx 0.5772$<span style="font-size: 14px;"> is the Euler-Mascheroni constant). This normalization allows scores to be compared across datasets of different sizes.</span>

---

## <span style="font-size: 16px;">Score Interpretation</span>

- <span style="font-size: 14px;">Score close to 1: likely anomaly (short average path)</span>
- <span style="font-size: 14px;">Score close to 0.5: normal (average path similar to expected)</span>
- <span style="font-size: 14px;">Score below 0.5: very normal (deep in dense regions)</span>

---

## <span style="font-size: 16px;">Advantages</span>

- <span style="font-size: 14px;">Linear time complexity:</span> $O(T \cdot \psi \cdot \log \psi)$ <span style="font-size: 14px;">for training</span>
- <span style="font-size: 14px;">No need to model normal data - directly isolates anomalies</span>
- <span style="font-size: 14px;">Works well in high dimensions</span>
- <span style="font-size: 14px;">No distance computations needed</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why random splits instead of optimal splits?**</span>
  <span style="font-size: 14px;">A: Random splits avoid overfitting and are computationally cheap. The averaging over many trees handles the randomness.</span>

- <span style="font-size: 14px;">**Q: How does this compare to LOF?**</span>
  <span style="font-size: 14px;">A: Local Outlier Factor computes density ratios, which is $O(n^2)$. Isolation Forest is $O(n \log n)$ and scales better.</span>

- <span style="font-size: 14px;">**Q: What about high-dimensional data?**</span>
  <span style="font-size: 14px;">A: Isolation Forest handles high dimensions naturally since each split only uses one feature. However, in very high dimensions, the signal can be diluted.</span>

---