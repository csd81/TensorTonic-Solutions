import torch


def quantile_balancing(
    router_scores: torch.Tensor,
    current_bias: torch.Tensor,
    selected_count: int,
) -> dict[str, torch.Tensor]:
  """
    Returns a dictionary containing routes, mixture weights, expert loads, and the next centered bias.
    """
  num_tokens, num_experts = router_scores.shape
  bias = current_bias.to(dtype=router_scores.dtype, device=router_scores.device)

  # 1. Biased scores and Top-(k+1) selection
  biased_scores = router_scores + bias.unsqueeze(0)
  topk_plus_1_vals, topk_plus_1_idx = torch.topk(
      biased_scores, k=selected_count + 1, dim=-1
  )

  # Top-k indices are the current routes; the (k+1)-th biased score is alpha_i
  selected_experts = topk_plus_1_idx[:, :selected_count]
  alpha = topk_plus_1_vals[:, selected_count]  # Shape: (num_tokens,)

  # 2. Mixture weights from raw (unbiased) router scores of selected experts
  selected_raw_scores = torch.gather(router_scores, dim=-1, index=selected_experts)
  mixture_weights = selected_raw_scores / selected_raw_scores.sum(
      dim=-1, keepdim=True
  )

  # 3. Expert loads across all tokens
  expert_loads = torch.bincount(
      selected_experts.reshape(-1), minlength=num_experts
  )

  # 4. Quantile margin computation for the next bias
  target_load = (num_tokens * selected_count) // num_experts
  margins = router_scores - alpha.unsqueeze(1)  # s_{i, j} - alpha_i

  # Take the (q + 1)-th largest margin per expert (0-indexed rank q along dim=0)
  sorted_margins, _ = torch.sort(margins, dim=0, descending=True)
  q_plus_1_margin = sorted_margins[target_load]

  # Negate and center across experts
  raw_next_bias = -q_plus_1_margin
  next_bias = raw_next_bias - raw_next_bias.mean()

  return {
      "selected_experts": selected_experts,
      "mixture_weights": mixture_weights,
      "expert_loads": expert_loads,
      "next_bias": next_bias,
  }
