import torch

def clipped_grpo_surrogate(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
    advantages: torch.Tensor, token_mask: torch.Tensor,
    clip_epsilon: int | float = 0.2,
) -> dict:
    """
    Returns a dict: loss and clipped_fraction (scalars), per_sequence_surrogate (tensor).
    """
    dtype = new_log_probs.dtype
    device = new_log_probs.device

    eps_t = torch.tensor(clip_epsilon, dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)
    low = one - eps_t
    high = one + eps_t

    log_ratio = torch.clamp(new_log_probs - old_log_probs, min=-60.0, max=60.0)
    ratios = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)
    unclipped = ratios * adv
    clipped = torch.clamp(ratios, min=low, max=high) * adv
    surrogate = torch.minimum(unclipped, clipped)

    mask_f = token_mask.to(dtype)
    masked_surrogate = torch.where(token_mask, surrogate, torch.zeros_like(surrogate))

    # Per-sequence mean over selected tokens (0 when a sequence has no selected tokens)
    seq_token_counts = mask_f.sum(dim=1)
    seq_surrogate_sums = masked_surrogate.sum(dim=1)
    per_sequence_surrogate = torch.where(
        seq_token_counts > 0,
        seq_surrogate_sums / seq_token_counts.clamp(min=1),
        torch.zeros_like(seq_surrogate_sums),
    )

    # Global token-level loss and clipped fraction across all selected tokens
    total_tokens = mask_f.sum()
    if total_tokens > 0:
        loss = -(masked_surrogate.sum() / total_tokens)
        is_clipped = token_mask & ((ratios < low) | (ratios > high))
        clipped_fraction = is_clipped.to(dtype).sum() / total_tokens
    else:
        loss = torch.zeros((), dtype=dtype, device=device)
        clipped_fraction = torch.zeros((), dtype=dtype, device=device)

    return {
        "loss": loss,
        "per_sequence_surrogate": per_sequence_surrogate,
        "clipped_fraction": clipped_fraction,
    }
