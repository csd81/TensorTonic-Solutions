import torch


def recursive_parameter_collection(
    weights: list[torch.Tensor], biases: list[torch.Tensor]
) -> tuple:
    """
    Returns a list of scalar parameter views and its integer count.
    """
    params = []

    for W, b in zip(weights, biases):
        out_features, in_features = W.shape
        for row_idx in range(out_features):
            for col_idx in range(in_features):
                params.append(W[row_idx, col_idx])
            params.append(b[row_idx])

    return params, len(params)
