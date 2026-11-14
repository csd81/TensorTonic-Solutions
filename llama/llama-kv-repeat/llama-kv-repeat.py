import torch

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Returns a float64 tensor with repeated key-value heads.
    """
    if n_rep == 1:
        return kv
        
    B, H_kv, S, D_h = kv.shape
    return (
        kv[:, :, None, :, :]
        .expand(B, H_kv, n_rep, S, D_h)
        .reshape(B, H_kv * n_rep, S, D_h)
    )
