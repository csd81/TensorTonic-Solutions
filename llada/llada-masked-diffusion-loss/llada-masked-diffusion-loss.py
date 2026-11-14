import torch
import torch.nn.functional as F

def llada_masked_diffusion_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_prob: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the scalar masked diffusion loss and the weighted per-token losses.
    """
    B, L, V = logits.shape
    dtype = logits.dtype
    device = logits.device

    # Ensure inputs match the device and dtype of logits without mutating them
    clean_ids = clean_ids.to(device=device, dtype=torch.long)
    mask_f = mask.to(device=device, dtype=dtype)
    mask_prob = mask_prob.to(device=device, dtype=dtype)
    prompt_lengths = prompt_lengths.to(device=device, dtype=dtype)

    # 1. Numerically stable per-token cross-entropy: -log_softmax(z)[clean_ids]
    log_probs = F.log_softmax(logits, dim=-1)
    token_nll = -log_probs.gather(dim=-1, index=clean_ids.unsqueeze(-1)).squeeze(-1)

    # 2. Apply mask and weight each sequence by 1 / p_b
    weighted_losses = (token_nll * mask_f) / mask_prob.unsqueeze(-1)

    # 3. Normalize each sequence by its response length L'_b = L - P_b, then average over batch
    response_lengths = (L - prompt_lengths).to(dtype=dtype)
    seq_losses = weighted_losses.sum(dim=-1) / response_lengths
    loss = seq_losses.mean()

    return loss, weighted_losses
