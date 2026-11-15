import torch


def moe_router(
    x: torch.Tensor, expert_centroids: torch.Tensor, top_k: int
) -> dict:
  """
    Returns selected expert indices and normalized weights in a dictionary.
    """
  # 1. Compute raw affinity logits: (B, S, D) @ (D, E) -> (B, S, E)
  logits = torch.matmul(x, expert_centroids.transpose(0, 1))

  # 2. Apply sigmoid independently per expert (no softmax across experts)
  scores = torch.sigmoid(logits)

  # 3. Select the top_k affinities and their corresponding expert indices
  top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

  # 4. Normalize only the selected top_k affinities so they sum to 1
  weights = top_scores / top_scores.sum(dim=-1, keepdim=True)

  return {
      "indices": top_indices.to(torch.int64),
      "weights": weights.to(torch.float64),
  }
