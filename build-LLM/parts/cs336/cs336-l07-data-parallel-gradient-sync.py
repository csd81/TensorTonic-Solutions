import torch

def data_parallel_gradient_sync(parameter_replicas: list, gradient_replicas: list, learning_rate: float) -> dict:
    """
    Returns a dict: mean_gradients (tensor list), updated_replicas (rank-ordered tensor lists).
    """
    num_ranks = len(parameter_replicas)
    num_params = len(parameter_replicas[0])

    mean_gradients = []
    base_updated_params = []

    for j in range(num_params):
        ref_param = parameter_replicas[0][j]
        orig_dtype = ref_param.dtype
        device = ref_param.device

        # Accumulate available gradients in float32 across all ranks
        grad_sum = torch.zeros_like(ref_param, dtype=torch.float32, device=device)
        for r in range(num_ranks):
            grad = gradient_replicas[r][j]
            if grad is not None:
                grad_sum.add_(grad.to(torch.float32))

        # Divide by total rank count R (missing gradients count as 0 in the numerator, not skipped)
        mean_grad_fp32 = grad_sum / num_ranks

        # Store mean gradient in the parameter's original dtype first
        mean_grad = mean_grad_fp32.to(orig_dtype)
        mean_gradients.append(mean_grad)

        # Perform the SGD step in float32 using the quantized mean_grad, then cast back
        updated_fp32 = ref_param.to(torch.float32) - float(learning_rate) * mean_grad.to(torch.float32)
        base_updated_params.append(updated_fp32.to(orig_dtype))

    # Broadcast independent clones of the updated parameters to every rank
    updated_replicas = [
        [param.clone() for param in base_updated_params]
        for _ in range(num_ranks)
    ]

    return {
        "mean_gradients": mean_gradients,
        "updated_replicas": updated_replicas,
    }
