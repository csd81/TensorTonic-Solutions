# Understanding the AlphaGo Policy-Value Loss

The network learns two predictions at once: a policy over actions and a value for the position. Its training objective measures both predictions against self-play targets and adds an L2 penalty for the supplied trainable parameters. The total loss is the sum of these three scalar terms.

## The policy target comes from search

For each batch item, the target policy $\pi$ is the normalized distribution derived from MCTS visit counts. It may place probability on several actions, so it is not necessarily a one-hot label.

The network produces raw policy logits. These logits must be converted into log-probabilities across the action dimension. The policy loss is cross-entropy with the target distribution:

$$
\ell_p=-\frac{1}{B}\sum_{i=1}^{B}\sum_{a=1}^{A}\pi_{i,a}\log p_{i,a}
$$

$B$ is the number of positions in the batch, $A$ is the number of complete actions, $i$ identifies a position, and $a$ identifies an action.

For one position, the inner sum measures how much log-probability the network assigned where search placed target mass. The negative sign turns larger target-aligned probabilities into a smaller loss. The outer mean gives every batch item equal weight regardless of action count.

## Why log-softmax is used

Computing softmax probabilities first and then taking their logarithm can be numerically unstable. Very unfavoured actions may underflow to zero, and the logarithm of zero is negative infinity.

Log-softmax combines normalization and logarithm in a stable operation. For logit $l_a$:

$$
\log p_a=l_a-\log\sum_b\exp(l_b)
$$

The library implementation uses stable arithmetic internally. Log-softmax must act across actions within each batch item. Applying it across the batch would compare unrelated Go positions and produce invalid distributions.

The target policy is multiplied directly by log-probabilities. This supports soft search targets naturally. Applying an index-based classification loss would discard the distribution when more than one action has target probability.

## The value loss

The value head predicts one scalar $v_i$ for each position, while the self-play result provides target $z_i$. Their mean squared error is:

$$
\ell_v=\frac{1}{B}\sum_{i=1}^{B}(v_i-z_i)^2
$$

Squaring makes positive and negative errors contribute equally and penalizes larger errors more strongly. Averaging across the batch keeps the scale comparable when batch size changes.

Predicted values and target values are flat batch vectors of shape (B,), with one value per batch item.

The targets lie between -1 and 1 and represent outcomes from the same player perspective used by the encoded state and value prediction. This function does not change signs; it assumes the supplied targets are already aligned.

## L2 regularization

The regularization term discourages large values in the supplied parameter tensors. For coefficient $\lambda$ and parameter tensors $\theta_j$:

$$
\ell_r=\lambda\sum_j\lVert\theta_j\rVert_2^2
$$

The squared L2 norm means summing the square of every scalar in every supplied tensor. It is not the norm itself, so no square root is taken. It is also not averaged by parameter count or batch size because the problem specifies a direct sum.

After the squares are accumulated, the result is multiplied by the nonnegative coefficient. A zero coefficient makes the contribution zero. An empty parameter collection also produces a scalar zero on the same device and with the same numeric type as the other losses.

Only the supplied parameter tensors are regularized. The function should not search a model object for additional parameters or automatically exclude any supplied biases.

## The total and return values

The complete objective is:

$$
\ell=\ell_p+\ell_v+\ell_r
$$

All four quantities are scalar PyTorch tensors. The required return order is total loss, policy loss, value loss, and regularization loss. Returning the components is useful because training code can inspect which part dominates without recomputing the objective.

The total is a sum, not an average of the three terms. The policy and value terms already contain their specified batch means, while the regularization term has its own coefficient.

## Preserving automatic differentiation

Every calculation must remain in PyTorch tensor operations. Converting losses to Python numbers, NumPy arrays, or detached tensors breaks the computation graph. The training step needs gradients to flow from the total loss back through policy logits, predicted values, and regularized parameters.

Log-softmax, multiplication, summation, mean, subtraction, squaring, and addition are differentiable. Constructing the empty regularization scalar from an existing tensor keeps its device and type compatible with the other terms.

The returned component losses should also remain tensors. Calling a scalar extraction method may be useful for logging elsewhere, but it does not belong inside this function.

## Common mistakes to avoid

- Applying softmax along the batch dimension normalizes unrelated positions together.
- Taking a logarithm after an unstable manual softmax can create infinities.
- Reducing the policy term across the whole batch before respecting each target row changes weighting.
- Using absolute error instead of squared error changes the value objective.
- Taking a square root computes an L2 norm rather than the required squared penalty.
- Averaging parameter squares changes the stated regularization term.
- Regularizing tensors that were not supplied expands the objective unexpectedly.
- Converting any loss to a Python number breaks automatic differentiation.
- Returning the four terms in another order violates the function contract.

The loss teaches the shared network from two self-play signals while controlling parameter magnitude. Stable policy cross-entropy follows the MCTS distribution, mean squared error follows the game outcome, and the weighted sum of parameter squares supplies the explicit regularizer.

---