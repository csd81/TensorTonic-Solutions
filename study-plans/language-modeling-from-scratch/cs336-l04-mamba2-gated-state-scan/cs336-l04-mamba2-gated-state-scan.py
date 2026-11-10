import torch

def mamba2_gated_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> dict:
    """
    Implements a gated recurrent state scan matching Mamba 2 formulation.
    All intermediate operations are performed in float32.
    """
    B, S, Dk = q.shape
    Dv = v.shape[-1]
    device = q.device
    orig_dtype = q.dtype
    
    # Cast all inputs to float32 for intermediate calculations
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    gamma_f32 = gamma.float()
    
    # Initialize states and output container
    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    outputs = torch.zeros(B, S, Dv, dtype=torch.float32, device=device)
    
    for t in range(S):
        # Extract components for time step t and expand dimensions for matrix ops
        g = gamma_f32[:, t, None, None]      # (B, 1, 1)
        kt = k_f32[:, t, :, None]            # (B, Dk, 1)
        vt = v_f32[:, t, None, :]            # (B, 1, Dv)
        qt = q_f32[:, t, :, None]            # (B, Dk, 1)
        
        # S_t = gamma_t * S_{t-1} + k_t @ v_t^T
        state = g * state + torch.matmul(kt, vt)
        
        # y_t = q_t^T @ S_t
        yt = torch.matmul(qt.transpose(-1, -2), state)  # (B, 1, Dv)
        outputs[:, t, :] = yt.squeeze(1)
        
    return {
        "outputs": outputs.to(orig_dtype),
        "final_state": state.to(orig_dtype)
    }
