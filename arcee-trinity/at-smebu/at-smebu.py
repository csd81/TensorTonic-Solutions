import torch


def smebu_update(
    expert_bias: torch.Tensor,
    momentum_buffer: torch.Tensor,
    expert_load: torch.Tensor,
    learning_rate: float,
    momentum: float,
    kappa: float,
) -> dict:
  """Returns updated expert bias and momentum buffer in a dictionary."""
  # Ensure float64 precision
  expert_bias = expert_bias.to(torch.float64)
  momentum_buffer = momentum_buffer.to(torch.float64)
  expert_load = expert_load.to(torch.float64)

  # 1. Compute mean load
  n_mean = torch.mean(expert_load)

  # 2. Normalize load violation and soft clamp with tanh
  v_i = (n_mean - expert_load) / n_mean
  v_tilde_i = torch.tanh(kappa * v_i)

  # 3. Scale by learning_rate and center the update to mean zero
  delta_b_i = learning_rate * v_tilde_i
  delta_b_i = delta_b_i - torch.mean(delta_b_i)

  # 4. Update momentum buffer and expert bias
  m_i_prime = momentum * momentum_buffer + (1.0 - momentum) * delta_b_i
  b_i_prime = expert_bias + m_i_prime

  return {"expert_bias": b_i_prime, "momentum_buffer": m_i_prime}
