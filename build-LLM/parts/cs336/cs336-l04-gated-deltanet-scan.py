import torch

def gated_deltanet_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> dict:
    """
    Computes the Gated DeltaNet state update and recurrent token-by-token sequence scan.
    
    Args:
        q: Query tensor of shape (B, S, Dk)
        k: Key tensor of shape (B, S, Dk)
        v: Value tensor of shape (B, S, Dv)
        gamma: Decay gate tensor of shape (B, S)
        beta: Update gate tensor of shape (B, S)
        
    Returns:
        A dictionary containing:
            - "outputs": (B, S, Dv) tensor
            - "final_state": (B, Dk, Dv) tensor
    """
    B, S, Dk = q.shape
    Dv = v.shape[2]
    device = q.device
    
    # Cast all tensors to float32 for intermediate precision requirements
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    gamma_f32 = gamma.to(torch.float32)
    beta_f32 = beta.to(torch.float32)
    
    # Initialize the hidden state matrix and output accumulation tensor
    S_state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    outputs = torch.zeros(B, S, Dv, dtype=torch.float32, device=device)
    
    for t in range(S):
        q_t = q_f32[:, t, :]  # (B, Dk)
        k_t = k_f32[:, t, :]  # (B, Dk)
        v_t = v_f32[:, t, :]  # (B, Dv)
        g_t = gamma_f32[:, t].view(B, 1, 1)  # (B, 1, 1)
        b_t = beta_f32[:, t].view(B, 1, 1)   # (B, 1, 1)
        
        # 1. Compute erasure term: k_t^T * S_{t-1} -> (B, 1, Dv)
        kt_S = torch.bmm(k_t.unsqueeze(1), S_state)
        # Form the outer product k_t * (k_t^T * S_{t-1}) -> (B, Dk, Dv)
        erase = torch.bmm(k_t.unsqueeze(2), kt_S)
        
        # 2. Compute writing term: k_t * v_t^T -> (B, Dk, Dv)
        write = torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
        
        # 3. Complete gated update step for the state matrix S_t
        S_state = g_t * (S_state - b_t * erase) + b_t * write
        
        # 4. Generate the step output token: y_t = q_t^T * S_t -> (B, 1, Dv)
        y_t = torch.bmm(q_t.unsqueeze(1), S_state).squeeze(1)
        outputs[:, t, :] = y_t

    return {
        "outputs": outputs.to(q.dtype),
        "final_state": S_state.to(q.dtype)
    }
