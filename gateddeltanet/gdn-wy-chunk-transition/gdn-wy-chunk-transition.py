import torch

def wy_chunk_transition(
    keys: torch.Tensor, 
    values: torch.Tensor, 
    beta: torch.Tensor, 
    log_decay: torch.Tensor, 
    state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C, d_k = keys.shape
    d_v = values.shape[-1]
    device, dtype = keys.device, keys.dtype

    g_cumsum = torch.cumsum(log_decay, dim=1)  # (B, C)
    gamma = torch.exp(g_cumsum)                # (B, C)
    gamma_C = gamma[:, -1:]                    # (B, 1)

    w = torch.zeros(B, C, d_k, device=device, dtype=dtype)
    u_tilde = torch.zeros(B, C, d_v, device=device, dtype=dtype)

    for r in range(C):
        k_r = keys[:, r, :]        # (B, d_k)
        v_r = values[:, r, :]      # (B, d_v)
        beta_r = beta[:, r:r+1]    # (B, 1)
        gamma_r = gamma[:, r:r+1]  # (B, 1)

        if r == 0:
            w_r = beta_r * k_r
            u_tilde_r = beta_r * v_r
        else:
            k_prev = keys[:, :r, :]                 # (B, r, d_k)
            w_prev = w[:, :r, :]                    # (B, r, d_k)
            u_prev = u_tilde[:, :r, :]              # (B, r, d_v)
            gamma_prev = gamma[:, :r].unsqueeze(-1) # (B, r, 1)

            dots = torch.sum(k_prev * k_r.unsqueeze(1), dim=-1, keepdim=True)  # (B, r, 1)

            w_sum = torch.sum(w_prev * dots, dim=1)  # (B, d_k)
            w_r = beta_r * (k_r - w_sum)

            decay_ratio = gamma_r.unsqueeze(-1) / gamma_prev  # (B, r, 1)
            u_term = u_prev * decay_ratio * dots              # (B, r, d_v)
            u_sum = torch.sum(u_term, dim=1)                  # (B, d_v)
            u_tilde_r = beta_r * (v_r - u_sum)

        w[:, r, :] = w_r
        u_tilde[:, r, :] = u_tilde_r

    I = torch.eye(d_k, device=device, dtype=dtype).unsqueeze(0)  # (1, d_k, d_k)

    wk_outer = torch.matmul(w.transpose(1, 2), keys)  # (B, d_k, d_k)
    F = gamma_C.unsqueeze(-1) * (I - wk_outer)        # (B, d_k, d_k)

    scale_G = (gamma_C / gamma).unsqueeze(-1)  # (B, C, 1)
    scaled_u_tilde = u_tilde * scale_G         # (B, C, d_v)
    G = torch.matmul(scaled_u_tilde.transpose(1, 2), keys)  # (B, d_v, d_k)

    S_out = torch.matmul(state, F) + G

    return w, u_tilde, F, G, S_out
