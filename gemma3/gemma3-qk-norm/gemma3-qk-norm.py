import torch

def qk_norm(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    gamma_q: torch.Tensor, 
    gamma_k: torch.Tensor, 
    eps: float = 1e-6
) -> dict[str, torch.Tensor]:
    """
    Returns normalized query and key tensors in a dictionary.
    """
    # Ensure float64 precision
    Q = Q.to(torch.float64)
    K = K.to(torch.float64)
    gamma_q = gamma_q.to(torch.float64)
    gamma_k = gamma_k.to(torch.float64)

    # Compute RMS along the last dimension (head_dim)
    rms_q = torch.sqrt(torch.mean(Q ** 2, dim=-1, keepdim=True) + eps)
    rms_k = torch.sqrt(torch.mean(K ** 2, dim=-1, keepdim=True) + eps)

    # Normalize and apply learned gain (broadcasts automatically across B, H, S)
    q_norm = (Q / rms_q) * gamma_q
    k_norm = (K / rms_k) * gamma_k

    return {
        "query": q_norm,
        "key": k_norm
    }
