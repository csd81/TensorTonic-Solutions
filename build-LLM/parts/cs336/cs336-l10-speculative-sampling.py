import torch

def speculative_sampling(
    draft_tokens: torch.Tensor, 
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor, 
    acceptance_random_values: torch.Tensor,
    residual_random_value: int | float,
) -> dict:
    """
    Returns a dict: tokens (int64 tensor), accepted_count (int), rejected_at (int or None).
    """
    # Initialize state variables
    accepted_count = 0
    rejected_at = None
    K = draft_tokens.shape[0]
    
    # Cast probability tensors to float64 for exact computation
    draft_probs_64 = draft_probabilities.to(torch.float64)
    target_probs_64 = target_probabilities.to(torch.float64)
    
    for i in range(K):
        token_id = draft_tokens[i].item()
        q_i = draft_probs_64[i, token_id].item()
        p_i = target_probs_64[i, token_id].item()
        
        # Calculate acceptance probability 'a_i'
        if q_i == 0.0:
            a_i = 1.0 if p_i > 0.0 else 0.0
        else:
            a_i = min(1.0, p_i / q_i)
            
        # Accept condition: supplied value must be strictly less than a_i
        if acceptance_random_values[i].item() < a_i:
            accepted_count += 1
        else:
            # First rejection reached
            rejected_at = i
            break
            
    if rejected_at is None:
        # All tokens accepted: return draft unchanged without an extra token
        return {
            "tokens": draft_tokens.to(torch.int64),
            "accepted_count": accepted_count,
            "rejected_at": None
        }
    
    # Construct the residual distribution for the rejection index
    p_row = target_probs_64[rejected_at]
    q_row = draft_probs_64[rejected_at]
    
    # Calculate max(p_i(v) - q_i(v), 0)
    diff = torch.clamp(p_row - q_row, min=0.0)
    r_i = diff / torch.sum(diff)
    
    # Find the sampled index from the cumulative replacement distribution
    cum_sum = torch.cumsum(r_i, dim=0)
    
    # Filter token indices where cumulative probability is strictly greater than residual_random_value
    valid_indices = torch.where(cum_sum > residual_random_value)[0]
    
    if valid_indices.numel() > 0:
        sampled_token = valid_indices[0].item()
    else:
        # Fallback to the last index with positive replacement probability due to rounding precision errors
        sampled_token = torch.where(r_i > 0.0)[0][-1].item()
        
    # Emit accepted prefix followed by the replacement token
    final_tokens = torch.empty(rejected_at + 1, dtype=torch.int64, device=draft_tokens.device)
    if rejected_at > 0:
        final_tokens[:rejected_at] = draft_tokens[:rejected_at]
    final_tokens[rejected_at] = sampled_token
    
    return {
        "tokens": final_tokens,
        "accepted_count": accepted_count,
        "rejected_at": rejected_at
    }
