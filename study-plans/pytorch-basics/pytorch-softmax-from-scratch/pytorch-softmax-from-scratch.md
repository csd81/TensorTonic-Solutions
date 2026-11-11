The softmax function converts a vector of raw scores (logits) into a valid probability distribution. It is one of the most important functions in deep learning, appearing in classification heads, attention mechanisms, and reinforcement learning policies.

## Definition

Given a vector $z = (z_0, z_1, \ldots, z_{C-1})$ of $C$ logits, softmax maps it to a probability vector $p$ where:

$$
p_j = \frac{e^{z_j}}{\sum_{k=0}^{C-1} e^{z_k}}
$$

Key properties of the output:

* Every element is positive: $p_j > 0$ for all $j$
* All elements sum to 1: $\sum_j p_j = 1$
* Larger logits get larger probabilities
* The relative ordering of values is preserved

## Why Not Just Normalize?

A simpler normalization like $p_j = z_j / \sum_k z_k$ does not work because logits can be negative, producing negative "probabilities." Another attempt, $p_j = |z_j| / \sum_k |z_k|$, loses information about which logit is largest. The exponential function solves both problems: it maps any real number to a positive number, and it preserves ordering (larger input gives larger output).

## The Numerical Stability Problem

The naive formula has a critical flaw. If any logit is large (e.g., $z_j = 1000$), then $e^{1000}$ overflows to infinity in float32 (which can represent values up to about $3.4 \times 10^{38}$). The result is NaN (infinity divided by infinity).

Conversely, if all logits are very negative (e.g., $z_j = -1000$), all exponentials underflow to 0, and we get 0/0 = NaN.

These are not edge cases. In practice, logits can easily reach hundreds or thousands during training, especially early on when weights are not yet tuned.

## The Max-Subtraction Trick

The fix is elegant: subtract the maximum logit from each element before exponentiating:

$$
p_j = \frac{e^{z_j - m}}{\sum_{k} e^{z_k - m}}, \quad m = \max_k z_k
$$

This is mathematically identical to the original formula. To see why, factor out $e^m$ from both numerator and denominator:

$$
\begin{aligned}
\frac{e^{z_j}}{\sum_k e^{z_k}}
&= \frac{e^{-m} \cdot e^{z_j}}{e^{-m} \cdot \sum_k e^{z_k}} \\
&= \frac{e^{z_j - m}}{\sum_k e^{z_k - m}}
\end{aligned}
$$

After subtracting $m$:

* The largest exponent becomes $e^{m - m} = e^0 = 1$, so nothing overflows
* Smaller values may underflow to 0, but that is harmless: it just means those classes get probability near 0
* The denominator is at least 1, so no division by zero

This trick costs almost nothing: one pass to find the max, then a subtraction. Every production softmax implementation uses it.

## Batched Softmax

In practice, softmax is applied row-wise to a 2-D tensor of shape $(N, C)$ where $N$ is the batch size and $C$ is the number of classes. Each row is an independent logit vector, and softmax normalizes each row independently. The max subtraction is also done per row:

$$
m_i = \max_{k} z_{i,k}
$$

In PyTorch, this means using dim=1 and keepdim=True so the shapes broadcast correctly during subtraction and division.

## Softmax Temperature

A generalized version divides logits by a scalar temperature $T$ before applying softmax:

$$
p_j = \frac{e^{z_j / T}}{\sum_k e^{z_k / T}}
$$

* $T = 1$: standard softmax
* $T \to 0$: the distribution approaches a one-hot vector (argmax), with all probability on the largest logit
* $T \to \infty$: the distribution approaches uniform $p_j = 1/C$

Temperature scaling is used in knowledge distillation (training a small model to mimic a large one), calibration (making model confidence more accurate), and sampling from language models (controlling creativity vs. determinism).

## Derivative of Softmax

The Jacobian of softmax with respect to the logits is:

$$
\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)
$$

where $\delta_{ij}$ is the Kronecker delta (1 if $i = j$, 0 otherwise). This can be written in matrix form as:

$$
J = \text{diag}(p) - p p^T
$$

When combined with cross-entropy loss, the gradient simplifies dramatically to $\hat{p} - y$ (predicted minus true), which is why these two functions are almost always used together.

## Softmax in Attention Mechanisms

Beyond classification, softmax is a core component of the attention mechanism in transformers:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Here softmax is applied to the scaled dot products to produce attention weights, which are then used to compute a weighted sum of values. The numerical stability of softmax is critical here because the dot products $QK^T$ can have large magnitudes.

## Softmax vs Sigmoid

For binary classification (2 classes), softmax with 2 outputs is equivalent to sigmoid on the difference of the two logits:

$$
\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)
$$

So for binary tasks, you can use either a single logit with sigmoid or two logits with softmax. In practice, the single-logit approach is preferred because it has fewer parameters.

## Common Mistakes

* **Forgetting the max subtraction**: Works on small logits, fails silently on large ones with NaN outputs
* **Wrong dimension**: Applying softmax along dim=0 (across the batch) instead of dim=1 (across classes) gives a completely wrong distribution
* **Not keeping dimensions**: When subtracting the max or dividing by the sum, failing to use keepdim=True causes shape mismatches due to broadcasting rules
* **Double softmax**: Applying softmax to values that are already probabilities (e.g., after a previous softmax) makes the distribution more uniform, losing information

## Where Softmax Appears

* **Classification output layers**: Converts final-layer logits to class probabilities
* **Attention weights**: Normalizes query-key scores in transformers
* **Mixture models**: Produces mixing coefficients from gating networks
* **Reinforcement learning**: Converts action logits to a policy distribution for sampling
* **Gumbel-softmax**: A differentiable approximation to categorical sampling, enabling gradient-based training of discrete choices

## Relationship to LogSoftmax

When you need log-probabilities (e.g., for cross-entropy or NLL loss), computing $\log(\text{softmax}(z))$ naively is unstable because softmax can produce values very close to 0. Instead, compute log-softmax directly:

$$
\log p_j = z_j - m - \log \sum_k e^{z_k - m}
$$

This avoids the double numerical hazard (overflow in exp, underflow in log) and is what PyTorch's F.log_softmax does internally.