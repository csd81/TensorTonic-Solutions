import torch

def kv_reconstruct(c_kv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor, num_heads: int) -> dict:
    """
    Returns reconstructed content keys and values in a dictionary.
    """
    # Ensure computations and outputs use torch.float64
    c_kv = c_kv.to(torch.float64)
    W_uk = W_uk.to(torch.float64)
    W_uv = W_uv.to(torch.float64)

    B, S, _ = c_kv.shape

    # Up-project latent vector to content keys and values
    # k_proj: (B, S, num_heads * d_k), v_proj: (B, S, num_heads * d_v)
    k_proj = c_kv @ W_uk.T
    v_proj = c_kv @ W_uv.T

    # Infer per-head widths independently
    d_k = k_proj.shape[-1] // num_heads
    d_v = v_proj.shape[-1] // num_heads

    # Reshape to (B, S, num_heads, head_dim) and transpose to (B, num_heads, S, head_dim)
    k_content = k_proj.view(B, S, num_heads, d_k).transpose(1, 2).contiguous()
    value = v_proj.view(B, S, num_heads, d_v).transpose(1, 2).contiguous()

    return {
        "k_content": k_content,
        "value": value
    }
