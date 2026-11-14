import torch

def token_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Returns the log-softmax probability of each target token.
    """
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

def llada_conditional_log_likelihood(
    sequence: torch.Tensor,
    prompt_length: int,
    mask_counts: torch.Tensor,
    permutations: torch.Tensor,
    logits: torch.Tensor,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the noisy copies, the per-sample estimates and their mean.
    """
    num_samples, total_len, _ = logits.shape
    response_len = total_len - prompt_length
    device = logits.device
    dtype = logits.dtype

    # Align tensors to logits' device without mutating inputs
    seq = sequence.to(device=device)
    counts = mask_counts.to(device=device)
    perms = permutations.to(device=device)

    # 1. Response mask: m_{k, j} = 1[pi_{k, j} < l_k], shape (samples, L)
    resp_mask = perms < counts.unsqueeze(-1)

    # 2. Construct noisy copies of shape (samples, P + L)
    noisy_copies = seq.unsqueeze(0).expand(num_samples, total_len).clone()
    noisy_copies[:, prompt_length:][resp_mask] = mask_id

    # 3. Compute log-probabilities at response positions using the supplied helper
    resp_logits = logits[:, prompt_length:, :]
    resp_targets = seq[prompt_length:].unsqueeze(0).expand(num_samples, response_len)
    g = token_log_probs(resp_logits, resp_targets)

    # 4. Per-sample estimates e_k = (L / l_k) * sum_j(m_{k, j} * g_{k, j}) and Monte Carlo mean
    masked_sum = (g * resp_mask.to(dtype=dtype)).sum(dim=-1)
    weights = torch.tensor(response_len, device=device, dtype=dtype) / counts.to(dtype=dtype)
    estimates = weights * masked_sum
    mean_estimate = estimates.mean()

    return noisy_copies, estimates, mean_estimate
