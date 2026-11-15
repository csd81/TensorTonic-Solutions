import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
  return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)


def tiny_mixer_network(
    h: torch.Tensor,
    token_gate: torch.Tensor,
    token_up: torch.Tensor,
    token_down: torch.Tensor,
    channel_gate: torch.Tensor,
    channel_up: torch.Tensor,
    channel_down: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
  """Returns the hidden states after every attention-free TRM layer."""
  num_layers = token_gate.shape[0]

  for l in range(num_layers):
    # 1. Token half
    # Swap last two axes: h (B, L, D) -> A (B, D, L)
    A = h.transpose(-1, -2)

    tg = token_gate[l]
    tu = token_up[l]
    td = token_down[l]

    gate_t = F.silu(A @ tg) * (A @ tu)
    up_t = gate_t @ td

    A_prime = rms_norm(A + up_t, dim=-1, eps=eps)
    h_prime = A_prime.transpose(-1, -2)

    # 2. Channel half
    cg = channel_gate[l]
    cu = channel_up[l]
    cd = channel_down[l]

    gate_c = F.silu(h_prime @ cg) * (h_prime @ cu)
    up_c = gate_c @ cd

    h_prime_prime = rms_norm(h_prime + up_c, dim=-1, eps=eps)

    h = h_prime_prime

  return h
