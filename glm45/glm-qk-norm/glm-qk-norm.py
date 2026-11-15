import torch


def qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    eps: float = 1e-5,
) -> dict:
  def apply_rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    # x shape: (..., d)
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    normed = (x.float() / rms) * gamma.float()
    return normed.to(x.dtype)

  # Apply RMSNorm independently to q and k along the last dimension (head_dim)
  q_norm = apply_rms_norm(q.to(torch.float64), gamma_q.to(torch.float64))
  k_norm = apply_rms_norm(k.to(torch.float64), gamma_k.to(torch.float64))

  return {"q_norm": q_norm, "k_norm": k_norm}
