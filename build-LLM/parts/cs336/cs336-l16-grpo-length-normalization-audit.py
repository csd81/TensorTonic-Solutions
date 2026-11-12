import torch

def grpo_length_normalization_audit(
    token_log_probs: torch.Tensor, advantages: torch.Tensor, token_mask: torch.Tensor,
) -> dict:
    """
    Returns a dict: sequence_summed and token_averaged (tensors), magnitude_ratio (scalar tensor).
    """
    dtype = token_log_probs.dtype
    device = token_log_probs.device

    masked_log_probs = torch.where(
        token_mask,
        token_log_probs,
        torch.zeros_like(token_log_probs),
    )
    sequence_summed = advantages * masked_log_probs.sum(dim=1)

    token_counts = token_mask.to(dtype).sum(dim=1)
    safe_counts = torch.where(token_counts > 0, token_counts, torch.ones_like(token_counts))
    token_averaged = torch.where(
        token_counts > 0,
        sequence_summed / safe_counts,
        torch.zeros_like(sequence_summed),
    )

    numer = sequence_summed.abs().sum()
    denom = token_averaged.abs().sum()
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    magnitude_ratio = torch.where(
        denom > 0,
        numer / safe_denom,
        torch.zeros((), dtype=dtype, device=device),
    )

    return {
        "sequence_summed": sequence_summed,
        "token_averaged": token_averaged,
        "magnitude_ratio": magnitude_ratio,
    }
