import torch


def dense_layer_forward(
    inputs: torch.Tensor,
    weight_matrix: torch.Tensor,
    biases: torch.Tensor,
    nonlinear: bool,
) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    preactivations = torch.matmul(weight_matrix, inputs) + biases

    if nonlinear:
        return torch.tanh(preactivations)

    return preactivations
