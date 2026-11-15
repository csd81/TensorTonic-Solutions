import torch


def gated_delta_online_updates(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    write_strength: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns the five next states and their retrieval residuals."""
  B, dv, dk = state.shape
  alpha = decay.view(B, 1, 1)
  beta = write_strength.view(B, 1, 1)

  # Outer product v k^T: shape (B, dv, dk)
  vkT = value.unsqueeze(2) * key.unsqueeze(1)

  # 1. Linear attention: S + v k^T
  S_la = state + vkT

  # 2. Mamba2: alpha * S + v k^T
  S_mamba2 = alpha * state + vkT

  # 3. Longhorn: epsilon = beta / (1 + beta * k^T k)
  k_norm_sq = (key ** 2).sum(dim=1, keepdim=True).unsqueeze(2)  # (B, 1, 1)
  eps = beta / (1.0 + beta * k_norm_sq)  # (B, 1, 1)

  I = torch.eye(dk, device=state.device, dtype=state.dtype).unsqueeze(
      0
  )  # (1, dk, dk)
  kkT = key.unsqueeze(2) * key.unsqueeze(1)  # (B, dk, dk)
  term_longhorn = I - eps * kkT
  S_longhorn = torch.matmul(state, term_longhorn) + eps * vkT

  # 4. DeltaNet: S * (I - beta * k k^T) + beta * v k^T
  term_deltanet = I - beta * kkT
  S_deltanet = torch.matmul(state, term_deltanet) + beta * vkT

  # 5. Gated DeltaNet: S * (alpha * (I - beta * k k^T)) + beta * v k^T
  term_gdn = alpha * (I - beta * kkT)
  S_gdn = torch.matmul(state, term_gdn) + beta * vkT

  # Stack states in order: (5, B, dv, dk)
  states_stack = torch.stack(
      [S_la, S_mamba2, S_longhorn, S_deltanet, S_gdn], dim=0
  )

  # Compute retrieval residuals: r = || S' k - v ||_2
  # S' @ k: (5, B, dv, dk) @ (1, B, dk, 1) -> (5, B, dv, 1)
  key_col = key.view(1, B, dk, 1)
  S_prime_k = torch.matmul(states_stack, key_col).squeeze(3)  # (5, B, dv)
  residuals = torch.norm(S_prime_k - value.unsqueeze(0), p=2, dim=2)  # (5, B)

  return states_stack, residuals
