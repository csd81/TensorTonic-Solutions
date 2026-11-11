import torch


def mlp_forward(
    inputs: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    current = inputs
    layer_outputs = []

    for W, b in zip(weights, biases):
        current = torch.tanh(torch.matmul(W, current) + b)
        layer_outputs.append(current)

    return current, layer_outputs
