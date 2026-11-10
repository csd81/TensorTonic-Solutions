## The Moore-Penrose Pseudoinverse

<span style="font-size: 14px;">The Moore-Penrose pseudoinverse $A^+$ is the unique matrix satisfying all four Penrose conditions:</span>

$$
\begin{align}
A A^+ A &= A \\
A^+ A A^+ &= A^+ \\
(A A^+)^T &= A A^+ \\
(A^+ A)^T &= A^+ A
\end{align}
$$

<span style="font-size: 14px;">The first condition says $A^+$ acts as a generalized inverse. The second says $A^+$ is also a generalized inverse of $A^+A$. The third and fourth conditions require $AA^+$ and $A^+A$ to be symmetric - they are the orthogonal projections onto $\mathcal{C}(A)$ and $\mathcal{C}(A^T)$ respectively. For an invertible square matrix, $A^+ = A^{-1}$. For non-square or singular matrices, $A^+$ provides a best-fit generalization.</span>

---

## SVD-Based Computation

<span style="font-size: 14px;">Given the SVD $A = U \Sigma V^T$, the pseudoinverse is:</span>

$$
A^+ = V \Sigma^+ U^T
$$

<span style="font-size: 14px;">where $\Sigma^+$ is formed by taking the reciprocal of each nonzero singular value and transposing the matrix. If $\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_r, 0, \ldots, 0)$, then $\Sigma^+ = \mathrm{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)^T$. This computation is numerically stable and handles rank-deficient matrices naturally by ignoring zero (or near-zero) singular values.</span>

---

## Special Cases

* <span style="font-size: 14px;">**Full column rank** ($m \geq n$, rank $= n$): $A^+ = (A^TA)^{-1}A^T$ (left inverse). Here $A^+A = I_n$.</span>
* <span style="font-size: 14px;">**Full row rank** ($n \geq m$, rank $= m$): $A^+ = A^T(AA^T)^{-1}$ (right inverse). Here $AA^+ = I_m$.</span>
* <span style="font-size: 14px;">**Invertible**: $A^+ = A^{-1}$</span>
* <span style="font-size: 14px;">**Zero matrix**: $A^+ = 0^T$</span>

---

## Minimum-Norm Least Squares Solution

<span style="font-size: 14px;">The pseudoinverse provides the minimum-norm least squares solution to $Ax = b$:</span>

$$
x^* = A^+ b = \arg\min_{x: \|Ax - b\| = \min} \|x\|
$$

<span style="font-size: 14px;">This means $x^* = A^+b$ minimizes $\|Ax - b\|$ (closest to $b$), and among all minimizers, it has the smallest $\|x\|$. For overdetermined systems ($m > n$) with full column rank, this reduces to the ordinary least squares solution $(A^TA)^{-1}A^Tb$. For underdetermined systems ($n > m$) with full row rank, there are infinitely many exact solutions $Ax = b$, and $A^+b = A^T(AA^T)^{-1}b$ selects the one with minimum Euclidean norm.</span>

---

## Practical Computation

<span style="font-size: 14px;">In NumPy, `np.linalg.pinv(A)` computes $A^+$ via SVD. The function `np.linalg.lstsq(A, b)` solves the least squares problem and internally uses the pseudoinverse approach. A key numerical consideration is choosing a threshold for "zero" singular values: values below $\sigma_1 \cdot \epsilon \cdot \max(m, n)$ (where $\epsilon$ is machine epsilon) are typically treated as zero. This threshold is the `rcond` parameter in NumPy.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Linear regression**: the coefficient vector $\hat{\beta} = X^+ y$ gives the least squares fit, handling rank-deficient design matrices gracefully where $(X^TX)^{-1}$ does not exist.</span>
* <span style="font-size: 14px;">**Underdetermined systems**: in compressed sensing and neural network analysis, $A^+b$ gives the minimum-norm solution among infinitely many solutions.</span>
* <span style="font-size: 14px;">**Regularization connection**: the pseudoinverse is the limit of the ridge regression solution as $\lambda \to 0$: $\lim_{\lambda \to 0} (A^TA + \lambda I)^{-1}A^T = A^+$. This explains why ridge regression is well-conditioned even when the pseudoinverse is sensitive to small singular values.</span>
* <span style="font-size: 14px;">**Control theory and robotics**: pseudoinverses compute optimal control inputs for systems with more actuators than constraints. In robotics, the pseudoinverse of the Jacobian maps desired end-effector velocities to joint velocities for redundant manipulators.</span>