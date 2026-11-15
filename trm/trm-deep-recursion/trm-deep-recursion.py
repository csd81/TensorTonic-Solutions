import torch


def tiny_net(h: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
  """Returns the stand-in TRM network output with the same shape as h."""
  for layer in range(params["w_gate"].shape[0]):
    gate = torch.nn.functional.silu(h @ params["w_gate"][layer])
    up = h @ params["w_up"][layer]
    h = h + (gate * up) @ params["w_down"][layer]
    h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-5)
  return h


def latent_recursion(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int, params: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns the refined answer and the final latent after one recursion process."""
  for _ in range(n):
    z = tiny_net(x + y + z, params)
    y = tiny_net(y + z, params)
  return y, z


def deep_recursion(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int,
    T: int,
    params: dict[str, torch.Tensor],
    prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns the detached answer carry, the detached latent carry, the answer logits, and the halting logits."""
  y_j, z_j = y, z

  # First T - 1 recursion processes without gradient tracking
  if T > 1:
    with torch.no_grad():
      for _ in range(T - 1):
        y_j, z_j = latent_recursion(x, y_j, z_j, n, params)

  # Last recursion process with gradient tracking
  y_T, z_T = latent_recursion(x, y_j, z_j, n, params)

  # Compute answer logits, dropping prefix positions P
  # y_T shape: (B, L, D), w_out shape: (D, V)
  answer_logits = y_T[:, prefix_len:, :] @ params["w_out"]

  # Compute halting logit reading position zero of the refined answer
  # w_halt shape: (D,), b_halt shape: scalar / 0-dim tensor
  halting_logits = y_T[:, 0, :] @ params["w_halt"] + params["b_halt"]

  # Detach the carries for the next supervision step
  return y_T.detach(), z_T.detach(), answer_logits, halting_logits
