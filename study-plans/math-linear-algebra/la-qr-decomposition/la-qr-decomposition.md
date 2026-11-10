## QR Decomposition

<span style="font-size: 14px;">The QR decomposition factors a matrix $A \in \mathbb{R}^{m \times n}$ (with $m \geq n$) into the product of an orthogonal matrix $Q$ and an upper triangular matrix $R$:</span>

$$
A = QR
$$

<span style="font-size: 14px;">The columns of $Q$ form an orthonormal basis for $\mathcal{C}(A)$, and $R$ encodes the coordinates of each column of $A$ in that basis.</span>

---

## Gram-Schmidt Process

<span style="font-size: 14px;">The classical Gram-Schmidt algorithm builds $Q$ column by column. Given columns $a_1, \ldots, a_n$ of $A$:</span>

$$
u_k = a_k - \sum_{j=1}^{k-1} \frac{\langle a_k, q_j \rangle}{\langle q_j, q_j \rangle} q_j, \qquad q_k = \frac{u_k}{\|u_k\|}
$$

<span style="font-size: 14px;">Each $q_k$ is obtained by orthogonalizing $a_k$ against all previously computed $q_j$, then normalizing. The entries of $R$ are $R_{jk} = \langle a_k, q_j \rangle$ for $j < k$ and $R_{kk} = \|u_k\|$.</span>

---

## Worked Example (3x2)

<span style="font-size: 14px;">Let $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$. Step 1: $q_1 = a_1 / \|a_1\| = (1, 0, 1)^T / \sqrt{2}$. Step 2: $u_2 = a_1 - \langle a_2, q_1 \rangle q_1 = (1,1,0)^T - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,0,1)^T = (1/2, 1, -1/2)^T$, then $q_2 = u_2 / \|u_2\|$. The resulting $R$ is upper triangular with $R_{11} = \sqrt{2}$, $R_{12} = 1/\sqrt{2}$, $R_{22} = \|u_2\| = \sqrt{3/2}$.</span>

---

## Householder Reflections

<span style="font-size: 14px;">An alternative and numerically more stable approach uses Householder reflections. A Householder reflector $H = I - 2vv^T$ (where $\|v\|=1$) reflects vectors across the hyperplane orthogonal to $v$. By choosing $v$ to zero out subdiagonal entries column by column, we transform $A$ into upper triangular form:</span>

$$
H_n \cdots H_2 H_1 A = R \implies A = (H_1 H_2 \cdots H_n) R = QR
$$

<span style="font-size: 14px;">Since each $H_i$ is orthogonal, their product $Q$ is also orthogonal.</span>

---

## Comparison of Methods

<span style="font-size: 14px;">Three main algorithms exist for computing QR:</span>

* <span style="font-size: 14px;">**Classical Gram-Schmidt**: intuitive but numerically unstable due to accumulated rounding errors in the projections. Modified Gram-Schmidt improves stability by recomputing projections against updated vectors.</span>
* <span style="font-size: 14px;">**Householder reflections**: the standard choice in numerical libraries (LAPACK, NumPy). Uses $2n^2(m - n/3)$ flops and is backward stable. Does not form $Q$ explicitly unless requested.</span>
* <span style="font-size: 14px;">**Givens rotations**: zeroes out one element at a time using plane rotations. Preferred for sparse or banded matrices where only a few elements need elimination.</span>

---

## Uniqueness

<span style="font-size: 14px;">If $A$ has full column rank, the QR decomposition is unique provided we require $R_{kk} > 0$. Without this sign convention, each column of $Q$ could be negated along with the corresponding row of $R$.</span>

---

## Solving Ax = b via QR

<span style="font-size: 14px;">From $A = QR$, the system $Ax = b$ becomes $QRx = b$. Since $Q$ is orthogonal, $Q^TQ = I$, so $Rx = Q^Tb$. Because $R$ is upper triangular, this is solved efficiently by back-substitution in $O(n^2)$. This avoids forming $A^TA$ and is numerically far more stable: the condition number is $\kappa(A)$ rather than $\kappa(A)^2$ as with normal equations.</span>

---

## Applications in ML and Numerical Computing

* <span style="font-size: 14px;">**Least squares**: solving $Ax = b$ via QR is more numerically stable than normal equations, especially when $A$ is ill-conditioned. This is what NumPy's `np.linalg.lstsq` uses internally.</span>
* <span style="font-size: 14px;">**QR algorithm for eigenvalues**: the iterative QR algorithm is the standard method for computing all eigenvalues of a matrix. Each step computes $A_k = Q_k R_k$ then forms $A_{k+1} = R_k Q_k$, and $A_k$ converges to upper triangular (Schur) form.</span>
* <span style="font-size: 14px;">**Numerical stability**: QR is preferred over Cholesky-based normal equations when the condition number of $A$ is large, because forming $A^TA$ squares the condition number.</span>
* <span style="font-size: 14px;">**Orthogonalization**: QR provides an orthonormal basis for the column space, useful in iterative solvers like GMRES and Arnoldi iteration for large-scale eigenvalue problems.</span>