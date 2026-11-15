import torch

def per_head_muon(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    previous_momentum: torch.Tensor,
    num_heads: int,
    momentum_coefficient: float,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the updated parameter, momentum, and orthogonalized update.
    """
    orig_dtype = parameter.dtype
    orig_device = parameter.device
    rows, cols = parameter.shape

    # Cast to parameter's device/dtype (upcasting low-precision floats to float32 for SVD stability)
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    param = parameter.to(device=orig_device, dtype=work_dtype)
    grad = gradient.to(device=orig_device, dtype=work_dtype)
    prev_mom = previous_momentum.to(device=orig_device, dtype=work_dtype)

    # 1. Update momentum: M_t = mu * M_{t-1} + G_t
    updated_momentum = momentum_coefficient * prev_mom + grad

    # 2. Partition output rows into equal attention-head blocks: (num_heads, head_dim, cols)
    head_blocks = updated_momentum.view(num_heads, rows // num_heads, cols)

    # 3. Batched compact SVD across all heads: M_t^{(h)} = U_h @ Sigma_h @ V_h^T
    # torch.linalg.svd(..., full_matrices=False) handles arbitrary (m, n) shapes and batches cleanly
    U, _, Vh = torch.linalg.svd(head_blocks, full_matrices=False)

    # 4. Compute polar factor O_h = U_h @ V_h^T and restore original (rows, cols) shape
    orthogonalized_update = torch.matmul(U, Vh).reshape(rows, cols)

    # 5. Update parameters: Theta_{t+1} = Theta_t - eta * O
    updated_parameter = param - learning_rate * orthogonalized_update

    return {
        "updated_parameter": updated_parameter.to(dtype=orig_dtype),
        "updated_momentum": updated_momentum.to(dtype=orig_dtype),
        "orthogonalized_update": orthogonalized_update.to(dtype=orig_dtype),
    }
