import torch

def verify_speculative_tokens(
    draft_token_ids: torch.Tensor,
    draft_distributions: torch.Tensor,
    target_distributions: torch.Tensor,
    uniform_draws: torch.Tensor,
) -> tuple:
    """
    Returns (accepted_prefix, accepted_count, next_token), 
    as 1D, scalar, and scalar integer PyTorch tensors.
    """
    K = draft_token_ids.shape[0]
    accepted_indices = []
    rejected_at = None
    
    for i in range(K):
        x_i = draft_token_ids[i].item()
        q_i_xi = draft_distributions[i, x_i].item()
        p_i_xi = target_distributions[i, x_i].item()
        
        # Calculate acceptance probability
        if q_i_xi == 0:
            a_i = 0.0
        else:
            a_i = min(1.0, p_i_xi / q_i_xi)
            
        # Check acceptance condition
        if uniform_draws[i].item() < a_i:
            accepted_indices.append(x_i)
        else:
            rejected_at = i
            break
            
    # Sample the next token using the final draw
    final_draw = uniform_draws[-1].item()
    
    if rejected_at is not None:
        # Sample from the residual distribution
        p_i = target_distributions[rejected_at]
        q_i = draft_distributions[rejected_at]
        residual = torch.clamp(p_i - q_i, min=0.0)
        residual_dist = residual / residual.sum()
        
        # Sample based on cumulative probability
        cum_sum = 0.0
        next_token_id = 0
        for idx, prob in enumerate(residual_dist):
            cum_sum += prob.item()
            if cum_sum > final_draw:
                next_token_id = idx
                break
    else:
        # All tokens accepted -> sample bonus token from target_distributions[K]
        bonus_dist = target_distributions[K]
        cum_sum = 0.0
        next_token_id = 0
        for idx, prob in enumerate(bonus_dist):
            cum_sum += prob.item()
            if cum_sum > final_draw:
                next_token_id = idx
                break
                
    accepted_prefix = torch.tensor(accepted_indices, dtype=torch.int64)
    accepted_count = torch.tensor(len(accepted_indices), dtype=torch.int64)
    next_token = torch.tensor(next_token_id, dtype=torch.int64)
    
    return (accepted_prefix, accepted_count, next_token)
