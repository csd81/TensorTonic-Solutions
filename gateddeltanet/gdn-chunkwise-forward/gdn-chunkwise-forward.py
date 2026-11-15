import torch

def ut_chunk_helper(keys: torch.Tensor, values: torch.Tensor, beta: torch.Tensor, log_decay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = keys.shape[-2]
    cumulative = torch.cumsum(log_decay, dim=-1)
    below = torch.ones(chunk, chunk, dtype=torch.bool, device=keys.device).tril(-1)
    decay_mask = torch.exp(cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~below, 0.0)
    scores = beta.unsqueeze(-1) * (keys @ keys.transpose(-1, -2))
    eye = torch.eye(chunk, dtype=keys.dtype, device=keys.device)
    weights = torch.diag_embed(beta)
    plain = torch.linalg.solve_triangular(eye + scores.masked_fill(~below, 0.0), weights, upper=False, unitriangular=True)
    gated = torch.linalg.solve_triangular(eye + scores * decay_mask, weights, upper=False, unitriangular=True)
    return torch.exp(cumulative).unsqueeze(-1) * (plain @ keys), gated @ values

def chunkwise_gated_delta(
    queries: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor, 
    log_decay: torch.Tensor, 
    beta: torch.Tensor, 
    initial_state: torch.Tensor, 
    chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the outputs, the incoming state of every chunk, and the final state.
    """
    seq_len = queries.shape[-2]
    state = initial_state.clone()
    
    outputs = []
    chunk_states = []
    
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        c_len = end - start
        
        # Record incoming state for this chunk
        chunk_states.append(state)
        
        # Slice current chunk
        q = queries[..., start:end, :]
        k = keys[..., start:end, :]
        v = values[..., start:end, :]
        g = log_decay[..., start:end]
        b = beta[..., start:end]
        
        # 1. UT Transform for decayed keys (W_left) and gated values (U_tilde)
        w_left, u_tilde = ut_chunk_helper(k, v, b, g)
        
        # 2. Cumulative decay gamma within the chunk
        cum_g = torch.cumsum(g, dim=-1) # (B, H, C_n)
        gamma = torch.exp(cum_g) # (B, H, C_n)
        gamma_end = gamma[..., -1:] # (B, H, 1)
        
        # 3. Decay-aware causal mask Gamma (inclusive of diagonal, i >= j)
        causal_mask = torch.ones((c_len, c_len), dtype=torch.bool, device=queries.device).tril(0)
        gamma_mat = torch.exp(cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).masked_fill(~causal_mask, 0.0)
        
        # 4. Decay queries to chunk start and keys to chunk end
        q_left = gamma.unsqueeze(-1) * q
        k_right = (gamma_end / gamma).unsqueeze(-1) * k
        
        # 5. Pseudo-values: Delta = U_tilde - W_left @ S_n^T
        delta = u_tilde - (w_left @ state.transpose(-1, -2))
        
        # 6. Chunk output: O_n = Q_left @ S_n^T + ((Q @ K^T) * Gamma) @ Delta
        attn_scores = (q @ k.transpose(-1, -2)) * gamma_mat
        o_chunk = (q_left @ state.transpose(-1, -2)) + (attn_scores @ delta)
        outputs.append(o_chunk)
        
        # 7. Next incoming state: S_{n+1} = gamma_{C_n} * S_n + Delta^T @ K_right
        state = gamma_end.unsqueeze(-1) * state + (delta.transpose(-1, -2) @ k_right)
        
    final_outputs = torch.cat(outputs, dim=-2)
    all_chunk_states = torch.stack(chunk_states, dim=-3)
    
    return final_outputs, all_chunk_states, state
