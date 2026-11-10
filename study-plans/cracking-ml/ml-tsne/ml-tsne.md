# <span style="font-size: 20px;">t-SNE Implementation</span>

<span style="font-size: 14px;">t-SNE (van der Maaten and Hinton, 2008) is a nonlinear dimensionality reduction technique primarily used for visualization. It is especially good at preserving local structure - points that are close in high dimensions remain close in the embedding.</span>

---

## <span style="font-size: 16px;">High-Dimensional Affinities</span>

<span style="font-size: 14px;">For each point</span> $i$<span style="font-size: 14px;">, define the conditional probability of</span> $j$ <span style="font-size: 14px;">being its neighbor:</span>

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

<span style="font-size: 14px;">where</span> $\sigma_i$ <span style="font-size: 14px;">is set via binary search to achieve a target perplexity. The joint probability is symmetrized:</span> $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$

---

## <span style="font-size: 16px;">Perplexity</span>

$$
\text{Perp}(P_i) = 2^{H(P_i)}
$$

<span style="font-size: 14px;">where</span> $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ <span style="font-size: 14px;">is the Shannon entropy. Perplexity roughly controls the effective number of neighbors. Typical values range from 5 to 50.</span>

---

## <span style="font-size: 16px;">Low-Dimensional Affinities (Student-t)</span>

<span style="font-size: 14px;">In the output space, the Student-t distribution with one degree of freedom (Cauchy) is used:</span>

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

<span style="font-size: 14px;">The heavy tails of the t-distribution allow nearby points to be faithfully represented while giving distant points more room, alleviating the "crowding problem" of SNE.</span>

---

## <span style="font-size: 16px;">Optimization</span>

<span style="font-size: 14px;">The KL divergence is minimized using gradient descent with momentum:</span>

$$
\frac{\partial \text{KL}}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij}) (1 + \|y_i - y_j\|^2)^{-1} (y_i - y_j)
$$

<span style="font-size: 14px;">Momentum starts at 0.5 and increases to 0.8 after 100 iterations. The initial embedding is sampled from a small Gaussian to break symmetry.</span>

---

## <span style="font-size: 16px;">Practical Tips</span>

- <span style="font-size: 14px;">**Perplexity**: should be less than the number of points. For small datasets, use small perplexity (2-10)</span>
- <span style="font-size: 14px;">**Learning rate**: typical values are 10-1000. Too small causes convergence to poor local minima; too large causes oscillation</span>
- <span style="font-size: 14px;">**Iterations**: at least 250-1000 for convergence. More for larger datasets</span>
- <span style="font-size: 14px;">**Early exaggeration**: multiplying P by 4 for the first 50 iterations can help form clearer clusters (not required here)</span>

---

## <span style="font-size: 16px;">Limitations</span>

- <span style="font-size: 14px;">Non-parametric: cannot embed new points without rerunning</span>
- <span style="font-size: 14px;">Global structure is not preserved (distances between clusters are meaningless)</span>
- <span style="font-size: 14px;">Non-convex optimization: different runs produce different embeddings</span>
- <span style="font-size: 14px;">Computationally expensive:</span> $O(n^2)$ <span style="font-size: 14px;">for exact implementation. Barnes-Hut approximation reduces to</span> $O(n \log n)$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why Student-t instead of Gaussian in the output space?**</span>
  <span style="font-size: 14px;">A: The heavy-tailed t-distribution allows moderate distances in the input space to map to larger distances in the output, alleviating the crowding problem where all clusters collapse together.</span>

- <span style="font-size: 14px;">**Q: How is t-SNE different from PCA?**</span>
  <span style="font-size: 14px;">A: PCA preserves global variance and is linear. t-SNE preserves local neighborhood structure and is nonlinear. PCA is deterministic; t-SNE is stochastic.</span>

- <span style="font-size: 14px;">**Q: What is UMAP?**</span>
  <span style="font-size: 14px;">A: Uniform Manifold Approximation and Projection. It is faster, preserves more global structure, and has a theoretical foundation in topological data analysis. It has largely replaced t-SNE in practice.</span>

---