import torch

def muon_spectral_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    previous_momentum: torch.Tensor,
    momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    orig_dtype = parameter.dtype

    # 1. Accumulate momentum in the input dtype without mutating inputs
    new_momentum = momentum_coefficient * previous_momentum + gradient

    # 2. Compute compact SVD and U @ V^T in float32
    u, _, vh = torch.linalg.svd(new_momentum.to(torch.float32), full_matrices=False)
    orthogonalized_fp32 = u @ vh

    # 3. Restore orthogonalized update to input dtype before updating the parameter
    orthogonalized_update = orthogonalized_fp32.to(orig_dtype)
    new_parameter = parameter - learning_rate * orthogonalized_update

    return {
        "new_parameter": new_parameter,
        "new_momentum": new_momentum,
        "orthogonalized_update": orthogonalized_update,
    }
