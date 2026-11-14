import torch


def moe_route(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    gate_b: torch.Tensor,
    top_k: int,
) -> dict:
  """Returns top_indices and top_weights in a dictionary."""
  # 1. Compute gate logits G = X @ W_g + b_g
  logits = torch.matmul(x, gate_W) + gate_b  # Shape: (n_tokens, n_experts)

  # 2. Find top-k values and their indices in descending order
  top_weights_raw, top_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)

  # 3. Apply softmax only to the selected top-k logits
  top_weights = torch.softmax(top_weights_raw.to(torch.float64), dim=-1)

  return {
      "top_indices": top_indices.to(torch.int64),
      "top_weights": top_weights.to(torch.float64),
  }
