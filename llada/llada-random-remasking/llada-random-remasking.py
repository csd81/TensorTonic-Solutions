import torch

def llada_random_remasking(step_logits: torch.Tensor, uniforms: torch.Tensor, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    N, L, V = step_logits.shape
    device = step_logits.device
    dtype = step_logits.dtype

    # Initialize response trajectory with shape (N + 1, L) filled with mask_id
    trajectory = torch.full((N + 1, L), mask_id, dtype=torch.int64, device=device)
    
    # Current response starts as all mask_id
    r_current = torch.full((L,), mask_id, dtype=torch.int64, device=device)
    trajectory[0] = r_current.clone()

    for k in range(N):
        # Time parameters
        t_k = (N - k) / N
        s_k = (N - k - 1) / N
        rho_k = s_k / t_k if t_k != 0 else 0.0

        # Greedy predictions: argmax over vocabulary
        y_hat_k = torch.argmax(step_logits[k], dim=-1).to(torch.int64)

        u_k = uniforms[k]
        r_next = torch.empty_like(r_current)

        is_masked = (r_current == mask_id)
        not_masked = ~is_masked

        # 1. If r_i(k) != mask_id, it never changes again
        r_next[not_masked] = r_current[not_masked]

        # 2. If r_i(k) == mask_id and u_k,i < rho_k, remask it
        remask_mask = is_masked & (u_k < rho_k)
        r_next[remask_mask] = mask_id

        # 3. If r_i(k) == mask_id and u_k,i >= rho_k, take greedy prediction
        reveal_mask = is_masked & (u_k >= rho_k)
        r_next[reveal_mask] = y_hat_k[reveal_mask]

        r_current = r_next
        trajectory[k + 1] = r_current.clone()

    final_response = trajectory[-1]
    return final_response, trajectory
