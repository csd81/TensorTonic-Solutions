import torch

def topk_moe_router(
    x: torch.Tensor, 
    w_router: torch.Tensor, 
    expert_outputs: torch.Tensor, 
    shared_output: torch.Tensor, 
    k: int
) -> dict:
    """
    Returns tensors in a dict: output, selected_experts, selected_weights, token_fraction, probability_mass.
    """
    orig_dtype = x.dtype
    orig_device = x.device
    
    # 1. Compute router scores in float32
    # x: (T, D), w_router: (D, E) -> scores: (T, E)
    scores = torch.matmul(x.to(dtype=torch.float32), w_router.to(dtype=torch.float32))
    
    # 2. Compute probability mass 'm' over all experts using full softmax
    prob_mass_all = torch.softmax(scores, dim=-1)
    probability_mass = prob_mass_all.mean(dim=0)
    
    # 3. Select top-k experts with stable sorting to break ties by lower expert index
    # torch.sort with descending=True and stable=True maintains the ascending index order for ties
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    
    selected_scores = sorted_scores[:, :k]
    selected_experts = sorted_indices[:, :k]
    
    # 4. Compute routing weights (gating weights) alpha via softmax over the selected top-k scores
    selected_weights = torch.softmax(selected_scores, dim=-1)
    
    # 5. Compute token fraction 'f'
    # For each expert, find the fraction of tokens that selected it
    T, E = scores.shape
    # Create an indicator matrix of shape (T, E) where 1 indicates the expert was selected
    template = torch.zeros((T, E), device=orig_device, dtype=torch.float32)
    indicator = template.scatter(1, selected_experts, 1.0)
    token_fraction = indicator.mean(dim=0)
    
    # 6. Combine expert outputs using the weights
    # expert_outputs: (T, E, Dout)
    # Gather the outputs of the selected experts for each token -> (T, k, Dout)
    Dout = expert_outputs.shape[-1]
    expanded_indices = selected_experts.unsqueeze(-1).expand(-1, -1, Dout)
    selected_expert_outputs = torch.gather(expert_outputs.to(dtype=torch.float32), 1, expanded_indices)
    
    # Multiply by weights and sum over the k selected experts -> (T, Dout)
    # selected_weights: (T, k) -> unsqueeze to (T, k, 1) for broadcasting
    weighted_outputs = selected_expert_outputs * selected_weights.unsqueeze(-1)
    summed_expert_outputs = weighted_outputs.sum(dim=1)
    
    # Final output combines the shared output and the routed expert outputs
    output = shared_output.to(dtype=torch.float32) + summed_expert_outputs
    
    # 7. Cast back to match specifications
    return {
        "output": output.to(dtype=orig_dtype),
        "selected_experts": selected_experts.to(dtype=torch.int64),
        "selected_weights": selected_weights.to(dtype=orig_dtype),
        "token_fraction": token_fraction.to(dtype=orig_dtype),
        "probability_mass": probability_mass.to(dtype=orig_dtype)
    }
