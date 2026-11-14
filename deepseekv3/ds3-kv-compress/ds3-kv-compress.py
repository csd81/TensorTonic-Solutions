import torch


def kv_compress(x: torch.Tensor, W_dkv: torch.Tensor) -> torch.Tensor:
  """
    Returns compressed KV latents of shape (batch, sequence, latent_width).
    
    x: shape (B, S, D) in torch.float64
    W_dkv: shape (C, D) in torch.float64
    """
  # Project x onto the transpose of W_dkv: (B, S, D) @ (D, C) -> (B, S, C)
  c_kv = torch.matmul(x, W_dkv.t())
  return c_kv
