import torch

def glm_mtp(
    h_prev: torch.Tensor,
    next_token_embed: torch.Tensor,
    norm_h_g: torch.Tensor,
    norm_e_g: torch.Tensor,
    proj_W: torch.Tensor,
    final_norm_g: torch.Tensor,
    lm_head_W: torch.Tensor,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns float64 MTP logits with shape (n_tokens, vocab_size).
    """
    # Convert raw Python lists from the test runner into float64 tensors
    h_prev = torch.as_tensor(h_prev, dtype=torch.float64)
    next_token_embed = torch.as_tensor(next_token_embed, dtype=torch.float64)
    norm_h_g = torch.as_tensor(norm_h_g, dtype=torch.float64)
    norm_e_g = torch.as_tensor(norm_e_g, dtype=torch.float64)
    proj_W = torch.as_tensor(proj_W, dtype=torch.float64)
    final_norm_g = torch.as_tensor(final_norm_g, dtype=torch.float64)
    lm_head_W = torch.as_tensor(lm_head_W, dtype=torch.float64)

    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps_rms)
        return x * rms * weight

    # 1. Normalize the previous hidden state and next token embedding separately
    norm_h = rms_norm(h_prev, norm_h_g)
    norm_e = rms_norm(next_token_embed, norm_e_g)

    # 2. Concatenate along the feature dimension -> (n_tokens, 2H)
    fused = torch.cat([norm_h, norm_e], dim=-1)

    # 3. Project back to hidden width H -> (n_tokens, H)
    u = fused @ proj_W

    # 4. Apply final RMSNorm and project to vocabulary logits -> (n_tokens, vocab_size)
    u_norm = rms_norm(u, final_norm_g)
    logits = u_norm @ lm_head_W

    return logits
