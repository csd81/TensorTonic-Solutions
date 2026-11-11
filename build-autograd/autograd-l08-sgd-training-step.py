import torch


def sgd_training_step(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    learning_rate: float,
) -> tuple:
    """
    Returns old loss, new loss, updated weights, updated bias, weight gradients, and bias gradient.
    """
    lr = float(learning_rate)

    # 1. Forward pass before update
    preds = torch.tanh(torch.matmul(inputs, weights) + bias)
    diff = preds - targets
    loss_before = (diff * diff).sum()

    # 2. Manual gradient derivation
    delta = 2.0 * diff * (1.0 - preds * preds)
    grad_w = torch.matmul(inputs.transpose(0, 1), delta)
    grad_b = delta.sum()

    # 3. Out-of-place simultaneous parameter update
    new_weights = weights - lr * grad_w
    new_bias = bias - lr * grad_b

    # 4. Recompute loss after update
    new_preds = torch.tanh(torch.matmul(inputs, new_weights) + new_bias)
    new_diff = new_preds - targets
    loss_after = (new_diff * new_diff).sum()

    return loss_before, loss_after, new_weights, new_bias, grad_w, grad_b
