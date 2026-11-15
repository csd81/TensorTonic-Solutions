import torch


def tiny_net(h: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
  """Returns the stand-in TRM network output with the same shape as h."""
  for layer in range(params["w_gate"].shape[0]):
    gate = torch.nn.functional.silu(h @ params["w_gate"][layer])
    up = h @ params["w_up"][layer]
    h = h + (gate * up) @ params["w_down"][layer]
    h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-5)
  return h


def trm_inner(
    y: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    params: dict[str, torch.Tensor],
    n: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns the detached refined answer, the detached refined latent, the halting logits, and the continue logits."""

  def recursion(y, z):
    for _ in range(n):
      z = tiny_net(x + y + z, params)
    return tiny_net(y + z, params), z

  with torch.no_grad():
    for _ in range(T - 1):
      y, z = recursion(y, z)
    y, z = recursion(y, z)
  q = y[:, 0] @ params["w_q"] + params["b_q"]
  return y.detach(), z.detach(), q[:, 0], q[:, 1]

def adaptive_halting_step(
    halted: torch.Tensor,
    steps: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    x_new: torch.Tensor,
    y_init: torch.Tensor,
    z_init: torch.Tensor,
    params: dict[str, torch.Tensor],
    n: int,
    T: int,
    max_steps: int,
    training: bool,
    mode: str,
    explore_min_steps: torch.Tensor,
) -> tuple:
  B, L, D = y.shape
  r_b = halted.bool()
  y_init_b = y_init.view(1, 1, -1).expand(B, L, D)
  z_init_b = z_init.view(1, 1, -1).expand(B, L, D)

  y_start = torch.where(r_b.view(B, 1, 1), y_init_b, y)
  z_start = torch.where(r_b.view(B, 1, 1), z_init_b, z)
  x_start = torch.where(r_b.view(B, 1, 1), x_new, x)
  start_steps = torch.where(halted, torch.zeros_like(steps), steps)

  y_next, z_next, halt_logits, continue_logits = trm_inner(
      y_start, z_start, x_start, params, n, T
  )
  new_steps = start_steps + 1
  at_limit = new_steps >= max_steps
  new_halted = at_limit
  continue_target = None

  if training and max_steps > 1:
    signal = (
        halt_logits > 0
        if mode.lower() == "trm"
        else halt_logits > continue_logits
    )
    new_halted = (at_limit | signal) & (new_steps >= explore_min_steps)
    if mode.lower() == "hrm":
      _, _, next_halt, next_continue = trm_inner(
          y_next, z_next, x_start, params, n, T
      )
      max_q = torch.maximum(next_halt, next_continue)
      continue_target = torch.sigmoid(torch.where(at_limit, next_halt, max_q))

  return (
      y_next,
      z_next,
      x_start,
      new_steps.to(torch.int64),
      new_halted.to(torch.bool),
      halt_logits,
      continue_target,
  )
