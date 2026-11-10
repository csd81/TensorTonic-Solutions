import torch

def prepare_expert_parallel_dispatch(
    selected_expert_ids: torch.Tensor,
    expert_to_rank: torch.Tensor,
    routing_weights: torch.Tensor,
    world_size: int,
) -> tuple:
    """
    Returns (dispatch_permutation, per_rank_send_counts, inverse_permutation, 
             reordered_expert_ids, reordered_weights): four integer tensors and one floating tensor.
    """
    # 1. Flatten the inputs to represent the sequence of all routes (i = t * k + r)
    flat_expert_ids = selected_expert_ids.flatten()
    flat_weights = routing_weights.flatten()
    
    # 2. Map each selected expert to its destination rank
    target_ranks = expert_to_rank[flat_expert_ids]
    
    # 3. Perform a stable sort by destination rank to get the dispatch permutation
    dispatch_permutation = torch.argsort(target_ranks, stable=True)
    
    # 4. Count how many routes are dispatched to each destination rank
    per_rank_send_counts = torch.bincount(target_ranks, minlength=world_size)
    
    # 5. Invert the dispatch permutation to find the send position for each original route
    inverse_permutation = torch.argsort(dispatch_permutation)
    
    # 6. Reorder expert IDs and routing weights into their scheduled send order
    reordered_expert_ids = flat_expert_ids[dispatch_permutation]
    reordered_weights = flat_weights[dispatch_permutation]
    
    return (
        dispatch_permutation,
        per_rank_send_counts,
        inverse_permutation,
        reordered_expert_ids,
        reordered_weights
    )
