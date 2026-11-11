import math
import torch


def initialize_weights(
    fan_in: int, fan_out: int, method: str
) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    shape = (fan_out, fan_in)

    if method == "xavier_uniform":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return (2.0 * torch.rand(shape, dtype=torch.float32) - 1.0) * limit

    elif method == "xavier_normal":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.randn(shape, dtype=torch.float32) * std

    elif method == "he_uniform":
        limit = math.sqrt(6.0 / fan_in)
        return (2.0 * torch.rand(shape, dtype=torch.float32) - 1.0) * limit

    elif method == "he_normal":
        std = math.sqrt(2.0 / fan_in)
        return torch.randn(shape, dtype=torch.float32) * std

    else:
        raise ValueError(f"Unsupported initialization method: {method}")
