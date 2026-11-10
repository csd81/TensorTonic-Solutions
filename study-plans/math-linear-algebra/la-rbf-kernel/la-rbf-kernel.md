## The RBF (Gaussian) Kernel

<span style="font-size: 14px;">The Radial Basis Function (RBF) kernel, also called the Gaussian kernel, measures similarity between two vectors using the squared Euclidean distance:</span>

$$
k(x, y) = \exp(-\gamma \|x - y\|^2) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)
$$

<span style="font-size: 14px;">where $\gamma = 1/(2\sigma^2)$. As the distance increases, the kernel value decays smoothly from 1 (identical points) toward 0 (distant points).</span>

---

## The Kernel Trick and Feature Space

<span style="font-size: 14px;">The kernel trick allows algorithms to operate in a high-dimensional feature space without explicitly computing the coordinates. The RBF kernel implicitly maps data into an **infinite-dimensional** feature space $\phi: \mathbb{R}^d \to \mathcal{H}$ such that:</span>

$$
k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}
$$

<span style="font-size: 14px;">This follows from the Taylor expansion of the exponential: $\exp(-\gamma\|x-y\|^2) = \exp(-\gamma\|x\|^2) \cdot \exp(-\gamma\|y\|^2) \cdot \exp(2\gamma x^Ty) = \exp(-\gamma\|x\|^2) \cdot \exp(-\gamma\|y\|^2) \cdot \sum_{k=0}^{\infty} \frac{(2\gamma)^k (x^Ty)^k}{k!}$. Each term in the sum corresponds to a dimension in the feature space, giving infinitely many features. We can compute inner products in this infinite-dimensional space in $O(d)$ time using just the kernel function.</span>

---

## Kernel Matrix Properties

<span style="font-size: 14px;">For any kernel function, the kernel matrix $K$ (also called the Gram matrix) has important properties:</span>

* <span style="font-size: 14px;">**Symmetric**: $K_{ij} = K_{ji}$ because $k(x_i, x_j) = k(x_j, x_i)$</span>
* <span style="font-size: 14px;">**Positive semi-definite**: $z^T K z \geq 0$ for all $z \in \mathbb{R}^n$. This is guaranteed by **Mercer's theorem**: a continuous symmetric function $k$ is a valid kernel (corresponds to an inner product in some feature space) if and only if the kernel matrix is positive semi-definite for all finite sets of points.</span>
* <span style="font-size: 14px;">**Diagonal is 1**: $K_{ii} = \exp(0) = 1$ for RBF</span>
* <span style="font-size: 14px;">**Entries in $(0, 1]$**: all entries are positive for RBF</span>

---

## Effect of Gamma

<span style="font-size: 14px;">The $\gamma$ parameter controls the "reach" of each data point:</span>

* <span style="font-size: 14px;">**Large $\gamma$** (small $\sigma$): the Gaussian is narrow, so the kernel decays quickly. Each point influences only its close neighbors. This leads to complex, tightly-fitting decision boundaries that risk overfitting. In the extreme, each training point becomes its own island.</span>
* <span style="font-size: 14px;">**Small $\gamma$** (large $\sigma$): the Gaussian is wide, so distant points remain similar. This produces smooth, simple decision boundaries that may underfit. In the extreme ($\gamma \to 0$), all kernel values approach 1 and no structure is captured.</span>

<span style="font-size: 14px;">A common default is $\gamma = 1/d$ (one over the number of features) or $\gamma = 1/(2 \cdot \text{median}(\|x_i - x_j\|^2))$, the median heuristic.</span>

---

## Comparison with Other Kernels

<span style="font-size: 14px;">The RBF kernel is the most commonly used kernel, but alternatives exist:</span>

* <span style="font-size: 14px;">**Linear kernel**: $k(x,y) = x^Ty$. Equivalent to no kernel (operates in original space). Fastest, but only finds linear boundaries.</span>
* <span style="font-size: 14px;">**Polynomial kernel**: $k(x,y) = (x^Ty + c)^d$. Maps to a finite-dimensional feature space of dimension $\binom{d + n}{n}$. Can capture polynomial interactions but may be numerically unstable for high degrees.</span>
* <span style="font-size: 14px;">**RBF kernel**: infinite-dimensional feature space. Universal approximator - can approximate any continuous function on a compact set with enough data. Most flexible but requires tuning $\gamma$.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Support Vector Machines**: RBF kernel SVM creates nonlinear decision boundaries by operating in the implicit feature space. The kernel matrix replaces all dot products in the SVM dual formulation.</span>
* <span style="font-size: 14px;">**Gaussian Processes**: the RBF kernel (called the squared exponential kernel in GP literature) defines the covariance structure of GP priors, producing smooth function posteriors. The length-scale $\sigma$ controls how quickly the function varies.</span>
* <span style="font-size: 14px;">**Kernel PCA**: PCA in the feature space $\mathcal{H}$ can capture nonlinear structure. Only the kernel matrix is needed, not the explicit feature map.</span>
* <span style="font-size: 14px;">**Kernel density estimation**: RBF kernels are used as smoothing kernels in KDE for nonparametric density estimation, where $\sigma$ acts as the bandwidth parameter.</span>