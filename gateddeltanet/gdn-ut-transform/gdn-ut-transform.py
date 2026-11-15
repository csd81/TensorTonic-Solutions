import torch


def ut_transform(
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns the ungated T, the gated T, the decayed WY key vectors and the gated value vectors."""
  B, C, d_k = keys.shape
  device = keys.device
  dtype = keys.dtype

  # Cumulative decay gamma_r = exp(sum_{i=1}^r g_i)
  gamma = torch.exp(torch.cumsum(log_decay, dim=-1))  # (B, C)

  # Decay-aware mask Gamma_ij = gamma_i / gamma_j for i >= j else 0
  gamma_ratio = gamma.unsqueeze(-1) / gamma.unsqueeze(-2)
  mask_lower = torch.tril(
      torch.ones((C, C), device=device, dtype=dtype), diagonal=0
  )
  Gamma = gamma_ratio * mask_lower

  # Gram matrix K K^T
  KKt = torch.matmul(keys, keys.transpose(-1, -2))

  # Diagonal matrix of beta
  diag_beta = torch.diag_embed(beta)

  def strict_lower(mat):
    return torch.tril(mat, diagonal=-1)

  I_C = torch.eye(C, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

  # 1) Ungated T matrix
  A_ungated = I_C + strict_lower(torch.matmul(diag_beta, KKt))
  T_ungated = torch.linalg.solve_triangular(A_ungated, diag_beta, upper=False)

  # 2) Gated T_tilde matrix
  A_gated = I_C + strict_lower(torch.matmul(diag_beta, Gamma * KKt))
  T_gated = torch.linalg.solve_triangular(A_gated, diag_beta, upper=False)

  # 3) Decayed WY key vectors
  TK = torch.matmul(T_ungated, keys)
  W_leftarrow = gamma.unsqueeze(-1) * TK

  # 4) Gated value vectors
  U_tilde = torch.matmul(T_gated, values)

  return T_ungated, T_gated, W_leftarrow, U_tilde
