import torch


def select_inference_configuration(
    candidates: torch.Tensor,
    required_rate: float,
    max_p95_latency: float,
) -> tuple:
  """Returns (selected_index, hourly_cost): scalar int64 and float64 tensors; (-1, -1.0) if infeasible."""
  # Column mapping: [gpu_count, batch_size, throughput, p95_latency, hourly_cost]
  throughput = candidates[:, 2]
  p95_latency = candidates[:, 3]
  hourly_cost = candidates[:, 4]
  gpu_count = candidates[:, 0]
  batch_size = candidates[:, 1]

  # Check feasibility conditions: q_i >= required_rate and ell_i <= max_p95_latency
  feasible_mask = (throughput >= required_rate) & (
      p95_latency <= max_p95_latency
  )

  if not torch.any(feasible_mask):
    return (
        torch.tensor(-1, dtype=torch.int64),
        torch.tensor(-1.0, dtype=torch.float64),
    )

  feasible_indices = torch.where(feasible_mask)[0]

  best_idx = None
  best_key = None

  for idx in feasible_indices:
    i = idx.item()
    # Sort keys: (hourly_cost, gpu_count, batch_size, original_index)
    key = (
        hourly_cost[i].item(),
        gpu_count[i].item(),
        batch_size[i].item(),
        i,
    )
    if best_key is None or key < best_key:
      best_key = key
      best_idx = i

  selected_index = torch.tensor(best_idx, dtype=torch.int64)
  selected_cost = candidates[best_idx, 4].to(torch.float64)

  return (selected_index, selected_cost)
