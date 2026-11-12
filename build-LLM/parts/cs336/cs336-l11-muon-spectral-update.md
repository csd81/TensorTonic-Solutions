# Orthogonalize a Muon Momentum Update

Muon is an optimizer for the weight matrices of a neural network that replaces each momentum matrix with its nearest orthogonal matrix, or semi-orthogonal for non-square matrices, before taking a step. It has trained transformers faster than AdamW in several published comparisons, including Kimi's Moonlight models. The heart of this problem is computing that orthogonalized update from a singular value decomposition and applying it to the parameter.

## Why orthogonalize an update

The gradient of a weight matrix is usually dominated by a few directions, so its singular values span a wide range. A plain momentum step moves the weights far along those few strong directions and barely at all along the many weak ones, even though the weak directions may still carry useful signal.

Orthogonalizing keeps the directions of the update but sets every singular value to 1, so every direction takes a step of the same size. The largest singular value of a matrix is its spectral norm, so Muon fixes the update's spectral norm at 1, which is where the name spectral update comes from. Practical implementations usually also multiply the update by a shape-dependent factor so the learning rate behaves consistently across layers of different shapes, which this problem leaves out.

## Momentum first

Let $W_t$ be the parameter, $G_t$ its gradient, $B_{t-1}$ the previous momentum and $\mu$ the momentum coefficient. The momentum is updated first:

$$
B_t = \mu B_{t-1} + G_t
$$

Orthogonalizing the momentum instead of the raw gradient keeps the averaging effect of momentum, because the direction comes from several recent steps. The reference Muon implementation uses a Nesterov variant by default, and this problem uses plain momentum.

## The orthogonal factor from the SVD

For a momentum matrix with $m$ rows and $n$ columns, let $r = \min(m, n)$. Its compact singular value decomposition is

$$
B_t = U \Sigma V^{\mathsf T}
$$

where the columns of $U$ and $V$ are orthonormal singular vectors and the diagonal matrix $\Sigma$ holds the $r$ singular values. Dropping $\Sigma$ gives the update

$$
O_t = U V^{\mathsf T}
$$

The compact form matters for non-square matrices. A full SVD would make $U$ a square $m \times m$ matrix, and $UV^{\mathsf T}$ would no longer match the parameter's shape. With the compact form, $U$ is $m \times r$ and $V^{\mathsf T}$ is $r \times n$, so their product is $m \times n$.

$O_t$ is the closest matrix to $B_t$, in Frobenius norm, whose singular values are all 1, sometimes called its polar factor. For a tall matrix its columns are orthonormal, and for a wide matrix its rows are. The parameter then moves against this direction with learning rate $\eta$:

$$
W_{t+1} = W_t - \eta O_t
$$

## Why the SVD runs in float32 and production uses Newton-Schulz

In this problem the SVD is computed in float32 even when the weights are stored in bfloat16, because standard SVD routines need higher precision, and PyTorch's SVD does not accept bfloat16 at all. The result is converted back to the parameter's dtype before the update.

A full SVD is expensive for large matrices, so production Muon (Jordan et al., 2024) approximates $UV^{\mathsf T}$ with a few Newton-Schulz iterations that use only matrix multiplications and run comfortably in bfloat16. This problem uses the exact SVD so the result is easy to check. Muon is also applied only to two-dimensional hidden weight matrices, while embeddings, output heads, biases and normalization scales are usually trained with AdamW.

## An off-diagonal example

Take a $2 \times 2$ parameter of all ones, previous momentum $\begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}$, gradient $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, momentum coefficient 0.5 and learning rate 0.1.

The new momentum is

$$
B_t = 0.5 \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 2 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

The first factor is orthogonal and the second is diagonal, so the singular values are 2 and 1 and the orthogonal part is the swap matrix:

$$
O_t = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

The momentum pushed the top-right entry twice as hard as the bottom-left one, but the orthogonalized update moves them equally. The new parameter is $\begin{bmatrix} 1 & 0.9 \\ 0.9 & 1 \end{bmatrix}$.

## Common mistakes to avoid

- **Orthogonalizing the raw gradient.** The SVD is applied to the updated momentum.
- **Keeping the singular values.** The update is $UV^{\mathsf T}$, with $\Sigma$ dropped.
- **Using the full SVD.** Only the compact form keeps the update the same shape as a tall or wide parameter.
- **Multiplying $U$ by $V$ instead of $V^{\mathsf T}$.** $U$ is $m \times r$ and $V^{\mathsf T}$ is $r \times n$, so only that product has the parameter's shape and the right values.
- **Running the SVD in bfloat16.** The decomposition runs in float32 and the update is converted back afterward.
- **Scaling the update by the momentum's size.** The orthogonalized update has every singular value equal to 1 whatever the gradient's magnitude, and the learning rate alone sets the step size.
