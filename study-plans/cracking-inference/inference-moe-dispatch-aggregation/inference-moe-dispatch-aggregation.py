import torch

def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Computes a sparse mixture-of-experts forward pass using the supplied 
    router logits and bias-free expert networks.
    
    Shapes:
        token_states:  (T, D)  where T is token count, D is model dimension
        router_logits: (T, E)  where E is expert count
        w_in:          (E, D, F) where F is expert hidden width
        w_out:         (E, F, D)
        top_k:         int (1 to E)
    """
    # 1. Identify the top-k experts for each token.
    # torch.topk is stable by default, prioritizing lower indices during ties.
    topk_logits, topk_indices = torch.topk(router_logits, k=top_k, dim=-1, largest=True, sorted=True)
    
    # 2. Normalize weights using softmax over only the selected top-k experts.
    routing_weights = torch.softmax(topk_logits, dim=-1)
    
    # 3. Initialize the aggregated final output tensor.
    output = torch.zeros_like(token_states)
    
    # 4. Dispatch tokens and aggregate across their selected experts.
    num_tokens = token_states.shape[0]
    for t in range(num_tokens):
        x_t = token_states[t] # Shape: (D,)
        
        for k_idx in range(top_k):
            expert_idx = topk_indices[t, k_idx].item()
            weight = routing_weights[t, k_idx]
            
            # Compute expert output: f_e(x) = ReLU(x @ W_in) @ W_out
            h = torch.relu(x_t @ w_in[expert_idx]) # Shape: (F,)
            y_expert = h @ w_out[expert_idx]        # Shape: (D,)
            
            # Weighted aggregation
            output[t] += weight * y_expert
            
    return output
