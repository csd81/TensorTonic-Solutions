import torch

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (expert_indices, routing_weights), tensors shaped (num_tokens, top_k).
    """
    # To break ties by lower expert index, we can invert the index or sort carefully.
    # A standard way: use torch.topk, but torch.topk might not guarantee lower index tie-breaking.
    # Instead, we can sort with a custom key or sort descending by logit, ascending by index.
    
    T, E = router_logits.shape
    device = router_logits.device
    
    # We want to sort by:
    # 1. logit descending (so negate it or use argsort)
    # 2. expert index ascending
    # Create an index tensor [0, 1, ..., E-1] broadcasted to (T, E)
    expert_indices_all = torch.arange(E, device=device).expand(T, E)
    
    # Sort logits descending. To handle tie-breaking (lower index first), 
    # we can sort by (-logits, expert_indices) or sort indices based on values.
    # Let's do it per token:
    sorted_logits, sorted_indices = torch.sort(router_logits, descending=True, dim=-1)
    
    # Wait, PyTorch's torch.sort is stable, but sorting descending puts larger values first.
    # If there are ties, we need smaller index first. A clean way is:
    # Sort ascending by index first, then sort/select by logit, or use a custom sort key:
    # Key: (-logit, index)
    custom_keys = torch.stack([-router_logits, expert_indices_all], dim=-1) # shape: (T, E, 2)
    # Lexsort or argsort:
    # We can sort indices 0..E-1 for each row based on (-logit, index)
    # Alternatively:
    
    expert_indices = torch.empty((T, top_k), dtype=torch.int64, device=device)
    routing_weights = torch.empty((T, top_k), dtype=torch.float32, device=device)
    
    for i in range(T):
        logits_i = router_logits[i]
        # Create list of (logit, -index) or similar? No, lower index preferred means smaller index comes first if logits tie.
        # So we sort by key: (-logit[idx], idx)
        sorted_pairs = sorted(
            range(E),
            key=lambda idx: (-logits_i[idx].item(), idx)
        )
        top_indices = torch.tensor(sorted_pairs[:top_k], device=device, dtype=torch.int64)
        top_logits = logits_i[top_indices]
        
        # Compute softmax over top_k logits
        weights = torch.softmax(top_logits, dim=-1)
        
        expert_indices[i] = top_indices
        routing_weights[i] = weights
        
    return expert_indices, routing_weights
