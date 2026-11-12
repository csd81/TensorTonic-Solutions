import torch

def classifier_free_guidance(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, guidance_scale: float, guidance_rescale: float) -> torch.Tensor:
    """
    Returns the guided prediction with the shape and dtype of cond_pred.
    """
    g = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    if guidance_rescale == 0.0:
        return g

    # Reduce over all non-batch dimensions (1, 2, ..., ndim - 1)
    reduce_dims = tuple(range(1, cond_pred.ndim))

    # Compute population standard deviation (correction=0) per sample
    std_cond = torch.std(cond_pred, dim=reduce_dims, correction=0, keepdim=True)
    std_g = torch.std(g, dim=reduce_dims, correction=0, keepdim=True)

    rescaled = g * (std_cond / std_g)
    return guidance_rescale * rescaled + (1.0 - guidance_rescale) * g
