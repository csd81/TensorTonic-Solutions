import torch


def situ_glu(
    input_tensor: torch.Tensor,
    gate_projection: torch.Tensor,
    up_projection: torch.Tensor,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
  """
    Returns the bounded element-wise gated activation tensor.
    """
  # Linear projections: x @ W^T (or matmul with transposed weight matrices)
  g = torch.matmul(input_tensor, gate_projection.transpose(-1, -2))
  u = torch.matmul(input_tensor, up_projection.transpose(-1, -2))

  # Bounded gate branch: [beta_1 * tanh(g / beta_1) * sigmoid(g)]
  bounded_gate = (gate_cap * torch.tanh(g / gate_cap)) * torch.sigmoid(g)

  # Bounded up branch: [beta_2 * tanh(u / beta_2)]
  bounded_up = up_cap * torch.tanh(u / up_cap)

  # Element-wise weave
  return bounded_gate * bounded_up
