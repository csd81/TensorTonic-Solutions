import torch
import torch.nn.functional as F

def sparse_moe_ffn(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    shared_W_gate: torch.Tensor,
    shared_W_up: torch.Tensor,
    shared_W_down: torch.Tensor,
    top_k: int
) -> torch.Tensor:
    """
    Returns shared-plus-routed expert outputs with the same shape as x.
    """
    B, S, D = x.shape
    E = router_weight.shape[0]
    
    # Flatten batch and sequence dimensions for clean token-wise routing: (T, D)
    T = B * S
    x_flat = x.reshape(T, D)
    
    # 1. Shared Expert (SwiGLU): runs on every token
    shared_gate = F.silu(x_flat @ shared_W_gate.T)
    shared_up = x_flat @ shared_W_up.T
    shared_out = (shared_gate * shared_up) @ shared_W_down.T  # (T, D)
    
    # 2. Routing: biased scores for selection, unbiased scores for weighting
    affinities = torch.sigmoid(x_flat @ router_weight.T)       # (T, E)
    biased_scores = affinities + expert_bias                   # (T, E)
    
    _, topk_indices = torch.topk(biased_scores, k=top_k, dim=-1)  # (T, top_k)
    
    # Gather original unbiased affinities at selected indices and normalize
    selected_affinities = torch.gather(affinities, dim=-1, index=topk_indices)
    weights = selected_affinities / selected_affinities.sum(dim=-1, keepdim=True)  # (T, top_k)
    
    # 3. Routed Experts (SwiGLU): compute only for assigned tokens
    routed_out = torch.zeros_like(x_flat)
    for e in range(E):
        # Find (token_idx, k_pos) pairs assigned to expert e
        token_idx, k_idx = torch.where(topk_indices == e)
        if token_idx.numel() == 0:
            continue
            
        x_e = x_flat[token_idx]                                # (N_e, D)
        gate_e = F.silu(x_e @ W_gate[e].T)                     # (N_e, I)
        up_e = x_e @ W_up[e].T                                 # (N_e, I)
        out_e = (gate_e * up_e) @ W_down[e].T                  # (N_e, D)
        
        w_e = weights[token_idx, k_idx].unsqueeze(-1)          # (N_e, 1)
        routed_out.index_add_(0, token_idx, w_e * out_e)
        
    # 4. Combine shared and routed outputs without residual connection
    out = (shared_out + routed_out).reshape(B, S, D)
    return out.to(torch.float64)
