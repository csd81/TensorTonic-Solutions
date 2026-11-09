# <span style="font-size: 20px;">GloVe (Global Vectors)</span>

<span style="font-size: 14px;">GloVe learns word embeddings by factorizing the global word-word co-occurrence matrix. Unlike Word2Vec's predictive approach, GloVe explicitly captures corpus-wide statistics, combining the benefits of count-based methods (like LSA) with the linear substructure properties of prediction-based methods.</span>

---

## <span style="font-size: 16px;">Co-occurrence Matrix</span>

<span style="font-size: 14px;">The foundation of GloVe is a co-occurrence matrix $X$ where entry $X_{ij}$ counts how often words $i$ and $j$ appear near each other within a context window. GloVe uses distance-weighted counts: if two words are $d$ positions apart, they contribute $1/d$ to the co-occurrence count. This gives more weight to closer context words.</span>

<span style="font-size: 14px;">For a window size of $w$, word pairs up to $w$ positions apart are counted. The matrix is symmetric: $X_{ij} = X_{ji}$.</span>

---

## <span style="font-size: 16px;">The GloVe Objective</span>

<span style="font-size: 14px;">GloVe minimizes a weighted least squares objective over all non-zero co-occurrence entries:</span>

$$J = \sum_{i,j: X_{ij} > 0} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$\mathbf{w}_i$ and $\tilde{\mathbf{w}}_j$ are the word and context embedding vectors</span>
* <span style="font-size: 14px;">$b_i$ and $\tilde{b}_j$ are scalar bias terms</span>
* <span style="font-size: 14px;">$f(X_{ij})$ is a weighting function that prevents very frequent co-occurrences from dominating</span>

---

## <span style="font-size: 16px;">Weighting Function</span>

<span style="font-size: 14px;">The weighting function $f$ caps the influence of high-frequency pairs:</span>

$$f(x) = \begin{cases} (x / x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$

<span style="font-size: 14px;">Typical values are $x_{max} = 100$ and $\alpha = 3/4$. Without this weighting, common word pairs like ("the", "of") would dominate the loss, drowning out signal from rarer, more informative pairs.</span>

---

## <span style="font-size: 16px;">Training with AdaGrad</span>

<span style="font-size: 14px;">GloVe uses AdaGrad (Adaptive Gradient) for optimization. Each parameter maintains a running sum of squared gradients, and the learning rate is scaled by $1/\sqrt{G}$ where $G$ is this accumulated sum:</span>

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}} \cdot g_t$$

<span style="font-size: 14px;">AdaGrad naturally gives larger updates to infrequent parameters and smaller updates to frequent ones. The squared gradient accumulators are initialized to 1.0 (as in the original implementation) to provide numerical stability for the first updates.</span>

---

## <span style="font-size: 16px;">Gradient Computation</span>

<span style="font-size: 14px;">For a non-zero entry $(i, j)$ with co-occurrence $X_{ij}$:</span>

<span style="font-size: 14px;">Let $\delta = f(X_{ij}) \cdot (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})$</span>

* <span style="font-size: 14px;">$\nabla_{\mathbf{w}_i} = \delta \cdot \tilde{\mathbf{w}}_j$</span>
* <span style="font-size: 14px;">$\nabla_{\tilde{\mathbf{w}}_j} = \delta \cdot \mathbf{w}_i$</span>
* <span style="font-size: 14px;">$\nabla_{b_i} = \delta$</span>
* <span style="font-size: 14px;">$\nabla_{\tilde{b}_j} = \delta$</span>

---

## <span style="font-size: 16px;">Final Embeddings</span>

<span style="font-size: 14px;">Since the co-occurrence matrix is symmetric, $\mathbf{w}$ and $\tilde{\mathbf{w}}$ are exposed to the same information and are equivalent up to random initialization. The original paper found that summing both vectors ($\mathbf{w} + \tilde{\mathbf{w}}$) as the final embedding slightly outperforms using either alone.</span>

---

## <span style="font-size: 16px;">GloVe vs Word2Vec</span>

* <span style="font-size: 14px;">**Word2Vec** scans local windows and learns from individual (center, context) pairs - it never sees the full co-occurrence picture at once</span>
* <span style="font-size: 14px;">**GloVe** precomputes the entire co-occurrence matrix, then optimizes a global objective. This makes it potentially more efficient since the matrix is computed once</span>
* <span style="font-size: 14px;">Levy and Goldberg (2014) showed that Word2Vec's skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, connecting both approaches theoretically</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why log of co-occurrence?** The model learns $\mathbf{w}_i^T \tilde{\mathbf{w}}_j \approx \log X_{ij}$. Operating in log space compresses the dynamic range and ensures the dot product captures ratios of co-occurrence probabilities, which encode meaning</span>
* <span style="font-size: 14px;">**Why sum W and W_tilde?** Both matrices encode the same information due to the symmetric objective. Summing them averages out noise from random initialization, similar to ensemble averaging</span>
* <span style="font-size: 14px;">**Why AdaGrad over SGD?** AdaGrad automatically adapts learning rates per parameter. Rare word pairs get larger effective updates, which is critical since the co-occurrence matrix is highly skewed</span>

---