import torch


def scalar_neuron_module(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    nonlinear: bool,
) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    preactivation = (inputs * weights).sum() + bias

    if nonlinear:
        return torch.tanh(preactivation)

    return preactivation
