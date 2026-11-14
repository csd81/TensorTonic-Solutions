import torch

def llada_forward_masking(
    clean_ids: torch.Tensor,
    mask_ratios: torch.Tensor,
    uniform_draws: torch.Tensor,
    prompt_lengths: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the noisy token IDs, the boolean mask indicator, and the per-sequence mask probabilities.
    """
    B, L = clean_ids.shape
    device = clean_ids.device

    # 1. Compute per-sequence mask probabilities: p_b = (1 - eps) * t_b + eps
    # Keep the exact floating dtype of mask_ratios
    mask_probs = (1.0 - eps) * mask_ratios + eps

    # 2. Check where uniform draws fall below p_b: shape (B, L)
    draw_below_prob = uniform_draws < mask_probs.unsqueeze(1)

    # 3. Check where token positions are at or after the prompt length: shape (B, L)
    positions = torch.arange(L, device=device).unsqueeze(0)
    after_prompt = positions >= prompt_lengths.unsqueeze(1)

    # 4. Combine conditions into the boolean mask indicator
    mask_indicator = draw_below_prob & after_prompt

    # 5. Construct noisy token IDs without modifying clean_ids
    noisy_ids = torch.where(mask_indicator, mask_id, clean_ids)

    return noisy_ids, mask_indicator, mask_probs
