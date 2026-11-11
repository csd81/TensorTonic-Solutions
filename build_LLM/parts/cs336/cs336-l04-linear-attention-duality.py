import torch

def linear_attention_duality(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: parallel_output, recurrent_output, final_state.
    """
    # Keep track of original metadata for the final cast
    orig_dtype = q.dtype
    device = q.device
    
    # Cast to float32 for stable intermediate computations
    q_f32 = q.to(dtype=torch.float32)
    k_f32 = k.to(dtype=torch.float32)
    v_f32 = v.to(dtype=torch.float32)
    
    B, S, Dk = q_f32.shape
    Dv = v_f32.shape[2]
    
    # --- Parallel Formulation ---
    # Compute the full dot-product similarity matrix: (B, S, S)
    attn_weights = torch.bmm(q_f32, k_f32.transpose(1, 2))
    
    # Mask out the future tokens (inclusive causal mask)
    causal_mask = torch.tril(torch.ones(S, S, device=device))
    attn_weights = attn_weights * causal_mask
    
    # Compute parallel outputs: (B, S, Dv)
    parallel_output = torch.bmm(attn_weights, v_f32)
    
    # --- Recurrent Formulation ---
    recurrent_output = torch.zeros(B, S, Dv, dtype=torch.float32, device=device)
    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    
    for t in range(S):
        # Extract vectors for the current step and unsqueeze for outer/matrix products
        q_t = q_f32[:, t, :].unsqueeze(1)  # (B, 1, Dk)
        k_t = k_f32[:, t, :].unsqueeze(2)  # (B, Dk, 1)
        v_t = v_f32[:, t, :].unsqueeze(1)  # (B, 1, Dv)
        
        # S_t = S_{t-1} + k_t * v_t^T
        state = state + torch.bmm(k_t, v_t)
        
        # y_t = q_t^T * S_t
        y_t = torch.bmm(q_t, state)  # (B, 1, Dv)
        recurrent_output[:, t, :] = y_t.squeeze(1)
        
    return {
        "parallel_output": parallel_output.to(dtype=orig_dtype),
        "recurrent_output": recurrent_output.to(dtype=orig_dtype),
        "final_state": state.to(dtype=orig_dtype)
    }
