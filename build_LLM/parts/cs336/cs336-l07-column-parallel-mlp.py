import torch

def column_parallel_mlp(
    x: torch.Tensor,
    weight_shards: list[torch.Tensor],
    bias_shards: list[torch.Tensor] | None = None,
) -> dict:
    """
    Returns a dict: local_activations (tensor list), full_activation (tensor).
    """
    local_activations = []

    for r, w_r in enumerate(weight_shards):
        # Compute local projection Y_r = X @ W_r
        y_r = x @ w_r

        # Add optional bias shard b_r (broadcasts across batch dimension B)
        if bias_shards is not None and bias_shards[r] is not None:
            y_r = y_r + bias_shards[r]

        local_activations.append(y_r)

    # Concatenate all rank outputs along the feature dimension in rank order
    full_activation = torch.cat(local_activations, dim=-1)

    return {
        "local_activations": local_activations,
        "full_activation": full_activation,
    }
