import torch

def sigmoid_router(x: torch.Tensor, router_weight: torch.Tensor, expert_bias: torch.Tensor, top_k: int) -> dict:
    """
    Returns selected expert indices and normalized weights in a dictionary.
    """
    # x shape: (B, S, D)
    # router_weight shape: (E, D) -> x @ router_weight.T -> (B, S, E)
    logits = torch.matmul(x, router_weight.T)
    
    # Compute independent sigmoid scores
    scores = torch.sigmoid(logits)  # shape: (B, S, E)
    
    # Add expert_bias only for ranking/selection
    biased_scores = scores + expert_bias  # shape: (B, S, E)
    
    # Select top_k indices based on biased_scores
    # topk returns (values, indices)
    topk_indices = torch.topk(biased_scores, k=top_k, dim=-1, largest=True, sorted=True).indices.to(torch.int64)
    
    # Gather the original (unbiased) sigmoid scores at the selected indices
    # We need to expand topk_indices or use gather to extract the corresponding scores
    topk_scores = torch.gather(scores, dim=-1, index=topk_indices)  # shape: (B, S, top_k)
    
    # Normalize selected scores across the selected experts so they sum to 1
    # Avoid division by zero by adding a small epsilon or using safety if needed, though sums > 0 here
    normalized_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    
    # Ensure float64 for weights and int64 for indices as required
    normalized_weights = normalized_weights.to(torch.float64)
    topk_indices = topk_indices.to(torch.int64)
    
    return {
        "indices": topk_indices,
        "weights": normalized_weights
    }
