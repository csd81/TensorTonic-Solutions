import torch


def greedy_decode(input_ids: list, step_logits: torch.Tensor) -> list:
  """
    Returns the prompt and greedily generated token IDs.
    """
  # Copy input_ids so the original list is not mutated
  output_ids = list(input_ids)

  # Handle edge case where 0 steps are generated
  if step_logits.numel() > 0:
    # Find the index of the highest logit along the last dimension (V)
    # torch.argmax automatically resolves ties by returning the first index
    next_tokens = torch.argmax(step_logits, dim=-1).tolist()
    output_ids.extend(next_tokens)

  return output_ids
