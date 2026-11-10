## Cholesky Decomposition

<span style="font-size: 14px;">The Cholesky decomposition factors a symmetric positive definite (SPD) matrix $A$ into the product of a lower triangular matrix $L$ and its transpose:</span>

$$
A = LL^T
$$

<span style="font-size: 14px;">This is essentially a "square root" of a matrix. The decomposition exists and is unique (with positive diagonal) if and only if $A$ is symmetric positive definite - meaning $A = A^T$ and $x^T A x > 0$ for all nonzero $x$.</span>

---

## Algorithm

<span style="font-size: 14px;">The entries of $L$ are computed column by column. For column $j$:</span>

$$
L_{jj} = \sqrt{A_{jj} - \sum_{k=0}^{j-1} L_{jk}^2}
$$

$$
L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{k=0}^{j-1} L_{ik} L_{jk} \right), \qquad i > j
$$

<span style="font-size: 14px;">If $A_{jj} - \sum_{k} L_{jk}^2 \leq 0$ at any step, the matrix is not positive definite and the decomposition fails. The algorithm processes each column from left to right, and each row within a column from top to bottom, making it straightforward to implement.</span>

---

## Efficiency and Cost

<span style="font-size: 14px;">Cholesky requires roughly $n^3/3$ operations, which is half the cost of LU decomposition ($2n^3/3$ operations). This makes it the preferred factorization when you know the matrix is SPD. It is also numerically stable without pivoting, unlike general LU decomposition. For comparison:</span>

* <span style="font-size: 14px;">LU decomposition: $\approx 2n^3/3$ flops</span>
* <span style="font-size: 14px;">Cholesky decomposition: $\approx n^3/3$ flops</span>
* <span style="font-size: 14px;">Solving via Cholesky (forward + back substitution): $2n^2$ flops after factorization</span>

---

## Existence and Uniqueness

<span style="font-size: 14px;">**Existence**: if $A$ is SPD, then $A_{11} > 0$ (take $x = e_1$), guaranteeing the first diagonal entry. By induction on the Schur complement $A_{22} - A_{21}A_{11}^{-1}A_{12}$ (which is also SPD), the algorithm succeeds at every step. **Uniqueness**: if $A = L_1 L_1^T = L_2 L_2^T$ with positive diagonals, then $L_2^{-1} L_1$ is both lower triangular and orthogonal, hence the identity. So $L_1 = L_2$.</span>

---

## Connection to Matrix Square Root

<span style="font-size: 14px;">The Cholesky factor $L$ can be viewed as the "square root" of $A$ in the sense that $A = LL^T$. This is not the unique positive-definite square root $A^{1/2}$ (which satisfies $A^{1/2} A^{1/2} = A$ with $A^{1/2}$ also SPD), but it is much cheaper to compute. The positive-definite square root requires an eigendecomposition: $A^{1/2} = V D^{1/2} V^T$.</span>

---

## Positive Definiteness Test

<span style="font-size: 14px;">Attempting a Cholesky decomposition is one of the most efficient ways to test if a symmetric matrix is positive definite. If the algorithm completes without encountering a non-positive value under the square root, the matrix is SPD. If it fails, the matrix is not. This is faster than computing all eigenvalues.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Sampling from multivariate Gaussians**: to sample $x \sim \mathcal{N}(\mu, \Sigma)$, compute $L = \mathrm{chol}(\Sigma)$, sample $z \sim \mathcal{N}(0, I)$, and set $x = \mu + Lz$. This works because $\mathrm{Cov}(Lz) = L I L^T = \Sigma$. This is the standard method in variational autoencoders and Monte Carlo simulations.</span>
* <span style="font-size: 14px;">**Solving SPD systems**: solving $Ax = b$ becomes two triangular solves: $Ly = b$ (forward substitution) then $L^T x = y$ (back substitution), each costing $O(n^2)$.</span>
* <span style="font-size: 14px;">**Gaussian processes**: the log-likelihood involves $\log \det(K)$ and $K^{-1}y$, both efficiently computed via Cholesky of the kernel matrix $K$. The log-determinant is $2 \sum \log L_{ii}$.</span>
* <span style="font-size: 14px;">**Kalman filters**: the predict and update steps involve covariance matrices that are SPD, and Cholesky factorization provides numerically stable updates. The square-root Kalman filter maintains the Cholesky factor directly.</span>
* <span style="font-size: 14px;">**Optimization and preconditioning**: Newton's method requires solving $H \Delta x = -g$ where $H$ is the Hessian. If $H$ is SPD (as at a local minimum), Cholesky is the method of choice. Incomplete Cholesky factorizations serve as preconditioners for conjugate gradient methods on large sparse systems.</span>