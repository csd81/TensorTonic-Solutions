## Eigenvalues and Eigenvectors

<span style="font-size: 14px;">An eigenvector of a square matrix $A$ is a nonzero vector $v$ whose direction is unchanged by the linear transformation $A$. The scalar $\lambda$ by which it is scaled is the eigenvalue:</span>

$$
A v = \lambda v, \qquad v \neq 0
$$

<span style="font-size: 14px;">Rearranging gives $(A - \lambda I)v = 0$, which has nonzero solutions only when $\det(A - \lambda I) = 0$. This is the characteristic equation, a polynomial of degree $n$ in $\lambda$ whose roots are the eigenvalues.</span>

---

## Worked Example (2x2)

<span style="font-size: 14px;">Consider $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. The characteristic equation is:</span>

$$
\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0
$$

<span style="font-size: 14px;">So $\lambda_1 = 5$, $\lambda_2 = 2$. For $\lambda_1 = 5$: solving $(A - 5I)v = 0$ gives $v_1 = (1, 1)^T$. For $\lambda_2 = 2$: solving $(A - 2I)v = 0$ gives $v_2 = (1, -2)^T$. Geometrically, $A$ stretches space by a factor of 5 along the direction $(1,1)$ and by a factor of 2 along $(1,-2)$.</span>

---

## Diagonalization

<span style="font-size: 14px;">If $A$ has $n$ linearly independent eigenvectors, we can form the matrix $V = [v_1 | v_2 | \cdots | v_n]$ and the diagonal matrix $D = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$, giving the eigendecomposition:</span>

$$
A = V D V^{-1}
$$

<span style="font-size: 14px;">This factorization makes matrix powers trivial: $A^k = V D^k V^{-1}$, where $D^k = \mathrm{diag}(\lambda_1^k, \ldots, \lambda_n^k)$. It also reveals the spectral radius $\rho(A) = \max|\lambda_i|$, which controls convergence of iterative methods.</span>

<span style="font-size: 14px;">Not every matrix is diagonalizable. A matrix is called **defective** if it has fewer than $n$ linearly independent eigenvectors. This happens when an eigenvalue has algebraic multiplicity greater than its geometric multiplicity - for example, $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has eigenvalue 1 with multiplicity 2 but only one eigenvector. Defective matrices require the Jordan normal form instead.</span>

---

## Symmetric Matrices

<span style="font-size: 14px;">For real symmetric matrices ($A = A^T$), the spectral theorem guarantees:</span>

* <span style="font-size: 14px;">All eigenvalues are real</span>
* <span style="font-size: 14px;">Eigenvectors corresponding to distinct eigenvalues are orthogonal</span>
* <span style="font-size: 14px;">$A = V D V^T$ where $V$ is orthogonal ($V^{-1} = V^T$)</span>

<span style="font-size: 14px;">This is especially important because many matrices in ML are symmetric: covariance matrices, kernel matrices, and Hessians.</span>

---

## Properties

* <span style="font-size: 14px;">$\mathrm{tr}(A) = \sum \lambda_i$ - trace equals sum of eigenvalues</span>
* <span style="font-size: 14px;">$\det(A) = \prod \lambda_i$ - determinant equals product of eigenvalues</span>
* <span style="font-size: 14px;">$A$ is invertible if and only if all eigenvalues are nonzero</span>
* <span style="font-size: 14px;">Eigenvalues of $A^T$ are the same as those of $A$</span>
* <span style="font-size: 14px;">Eigenvalues of $A^k$ are $\lambda_i^k$ - this is the basis for the **power method**, which computes the dominant eigenvector by repeatedly multiplying a random vector by $A$ and normalizing</span>

---

## Applications in ML

* <span style="font-size: 14px;">**PCA**: the principal components are eigenvectors of the covariance matrix, sorted by eigenvalue magnitude. Larger eigenvalues correspond to directions of greater variance.</span>
* <span style="font-size: 14px;">**Spectral clustering**: uses eigenvectors of the graph Laplacian $L = D - W$ to embed data into a low-dimensional space before clustering. The smallest nonzero eigenvalues encode community structure.</span>
* <span style="font-size: 14px;">**PageRank**: Google's PageRank is the dominant eigenvector (eigenvalue 1) of a modified adjacency matrix. The power method iteratively refines page importance scores until convergence.</span>
* <span style="font-size: 14px;">**Convergence analysis**: the condition number $\kappa = |\lambda_{\max}|/|\lambda_{\min}|$ of the Hessian determines gradient descent convergence rate. Large condition numbers indicate ill-conditioned problems with slow convergence.</span>
* <span style="font-size: 14px;">**Markov chains**: the stationary distribution is the eigenvector corresponding to eigenvalue 1 of the transition matrix. The second-largest eigenvalue controls the mixing time of the chain.</span>