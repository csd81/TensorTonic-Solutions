import torch
import torch.nn.functional as F

def kda_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_logits: torch.Tensor,
    write_strength: torch.Tensor,
    output_gate_logits: torch.Tensor,
    output_projection: torch.Tensor,
    initial_state: torch.Tensor,
    g_min: float = -5.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing sequence outputs and the final recurrent state.
    """
    orig_dtype = query.dtype
    
    # Upcast to float32 for numerical stability during recurrence if using low precision
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    
    q = query.to(compute_dtype)
    k = key.to(compute_dtype)
    v = value.to(compute_dtype)
    z = decay_logits.to(compute_dtype)
    beta = write_strength.to(compute_dtype)
    gate_logits = output_gate_logits.to(compute_dtype)
    proj = output_projection.to(compute_dtype)
    
    # Clone initial_state to avoid mutating the input tensor
    state = initial_state.to(compute_dtype).clone()  # (B, H, D_k, D_v)
    
    B, S, H, D_k = q.shape
    D_v = v.shape[-1]
    
    # Precompute elementwise retention factor alpha_t: (B, S, H, D_k)
    alpha = torch.exp(g_min * torch.sigmoid(z))
    
    # Ensure write_strength broadcasts cleanly against (B, H, 1, D_v)
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1)  # (B, S, H, 1)
        
    readouts = []
    
    for t in range(S):
        q_t = q[:, t]          # (B, H, D_k)
        k_t = k[:, t]          # (B, H, D_k)
        v_t = v[:, t]          # (B, H, D_v)
        alpha_t = alpha[:, t]  # (B, H, D_k)
        beta_t = beta[:, t]    # (B, H, 1)
        
        # 1. Apply diagonal decay to previous state: Diag(alpha_t) @ S_{t-1}
        decayed_state = state * alpha_t.unsqueeze(-1)  # (B, H, D_k, D_v)
        
        # 2. Compute delta rule update: S_t = decayed_state + beta_t * k_t @ (v_t - decayed_state^T @ k_t)^T
        # Project decayed_state along k_t: (B, H, D_v)
        k_proj = torch.einsum("bhk,bhkv->bhv", k_t, decayed_state)
        delta = (v_t - k_proj) * beta_t  # (B, H, D_v)
        state = decayed_state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        
        # 3. Read from the updated state: o_tilde_t = S_t^T @ q_t -> (B, H, D_v)
        o_t = torch.einsum("bhkv,bhk->bhv", state, q_t)
        readouts.append(o_t)
        
    # Stack along sequence dimension: (B, S, H, D_v)
    o = torch.stack(readouts, dim=1)
    
    # 4. Independent RMS normalization per head over D_v
    rms = torch.sqrt(torch.mean(o.pow(2), dim=-1, keepdim=True) + eps)
    o_norm = o / rms
    
    # 5. Apply sigmoid output gate
    o_gated = o_norm * torch.sigmoid(gate_logits)
    
    # 6. Concatenate heads -> (B, S, H * D_v) and apply output projection
    o_concat = o_gated.reshape(B, S, H * D_v)
    outputs = F.linear(o_concat, proj)
    
    return {
        "outputs": outputs.to(orig_dtype),
        "final_state": state.to(orig_dtype),
    }
