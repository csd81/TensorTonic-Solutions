import torch


def squared_error_loss_gradients(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    diff = predictions - targets
    loss = (diff * diff).sum()
    grads = 2.0 * diff

    return loss, grads
