## Principal Component Analysis (PCA)

<span style="font-size: 14px;">PCA finds the directions of maximum variance in data. Given $n$ data points in $\mathbb{R}^d$, PCA finds an orthogonal basis where the first axis captures the most variance, the second captures the most remaining variance, and so on.</span>

---

## Algorithm

<span style="font-size: 14px;">1. **Center**: subtract the mean $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ from each data point</span>

<span style="font-size: 14px;">2. **Covariance**: compute the sample covariance matrix</span>

$$
C = \frac{1}{n-1} \tilde{X}^T \tilde{X}
$$

<span style="font-size: 14px;">where $\tilde{X}$ is the centered data matrix</span>

<span style="font-size: 14px;">3. **Eigendecompose**: compute eigenvalues $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$ and eigenvectors $v_1, \ldots, v_d$ of $C$</span>

<span style="font-size: 14px;">4. **Project**: the projected data is $X_{\text{proj}} = \tilde{X} V_k$ where $V_k = [v_1 | \cdots | v_k]$</span>

---

## Worked Example (2D)

<span style="font-size: 14px;">Suppose we have 4 points: $(1,2), (3,4), (5,6), (7,8)$. After centering (mean = $(4,5)$), the centered data is $(-3,-3), (-1,-1), (1,1), (3,3)$. The covariance matrix is $C = \frac{1}{3}\begin{pmatrix} 20 & 20 \\ 20 & 20 \end{pmatrix}$. The eigenvalues are $\lambda_1 = 40/3$ and $\lambda_2 = 0$. The first eigenvector is $(1/\sqrt{2}, 1/\sqrt{2})^T$ (the diagonal direction). All variance lies along one component, confirming the data is perfectly collinear.</span>

---

## Explained Variance

<span style="font-size: 14px;">The explained variance ratio for each component is:</span>

$$
\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
$$

<span style="font-size: 14px;">The cumulative sum helps choose $k$: if 95% of variance is captured by $k$ components, little information is lost by discarding the rest. This is visualized using a **scree plot**, which shows eigenvalues in decreasing order. The "elbow" - where the curve bends sharply - suggests a natural cutoff. Common strategies: (1) choose $k$ at the elbow, (2) choose $k$ so cumulative variance exceeds 95%, or (3) use cross-validation.</span>

---

## Connection to SVD

<span style="font-size: 14px;">PCA is intimately connected to the SVD. If $\tilde{X} = U \Sigma V^T$ is the SVD of the centered data matrix, then:</span>

* <span style="font-size: 14px;">The right singular vectors $V$ are the principal components (eigenvectors of $C$)</span>
* <span style="font-size: 14px;">The eigenvalues of $C$ are $\lambda_i = \sigma_i^2 / (n-1)$</span>
* <span style="font-size: 14px;">The projected data is $\tilde{X} V_k = U_k \Sigma_k$</span>

<span style="font-size: 14px;">In practice, computing PCA via SVD is more numerically stable than forming the covariance matrix explicitly. The covariance approach squares the condition number, while SVD works directly with the data matrix. For large sparse datasets, truncated SVD (computing only the top $k$ singular values) is far more efficient than full eigendecomposition.</span>

---

## Limitations

<span style="font-size: 14px;">PCA has important limitations to be aware of:</span>

* <span style="font-size: 14px;">**Linear only**: PCA finds linear projections. If the data lies on a nonlinear manifold (e.g., a Swiss roll), PCA fails to capture the true structure. **Kernel PCA** addresses this by applying PCA in a kernel-induced feature space, enabling nonlinear dimensionality reduction.</span>
* <span style="font-size: 14px;">**Sensitive to scale**: features with larger magnitudes dominate the covariance. Always standardize features (zero mean, unit variance) before PCA unless the scale differences are meaningful.</span>
* <span style="font-size: 14px;">**Variance is not always importance**: PCA maximizes variance, but high-variance directions may be noise rather than signal. Supervised methods like LDA or PLS may be more appropriate when labels are available.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Dimensionality reduction**: reduce $d$-dimensional data to $k$ dimensions while preserving maximum variance, enabling visualization and faster downstream models</span>
* <span style="font-size: 14px;">**Noise reduction**: by keeping only the top components, PCA removes low-variance directions that often correspond to noise</span>
* <span style="font-size: 14px;">**Feature decorrelation**: PCA produces uncorrelated features, which helps algorithms that assume feature independence</span>
* <span style="font-size: 14px;">**Eigenfaces**: in face recognition, PCA on face images finds the principal "face modes" - the eigenfaces - enabling compact representation and matching</span>
* <span style="font-size: 14px;">**Preprocessing**: PCA is commonly used before training to reduce input dimensionality, often followed by whitening to make features unit variance</span>