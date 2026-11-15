import torch
import torch.nn.functional as F

def multi_teacher_opd_reward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    domain_indices: torch.Tensor,
    effort_indices: torch.Tensor,
    sampled_tokens: torch.Tensor,
    clip_threshold: float,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing clipped rewards and selected teacher token log probabilities.
    """
    orig_dtype = student_logits.dtype
    orig_device = student_logits.device
    batch_size = student_logits.shape[0]

    # Ensure index tensors are long and on the target device
    batch_idx = torch.arange(batch_size, device=orig_device)
    d_idx = domain_indices.to(device=orig_device, dtype=torch.long)
    e_idx = effort_indices.to(device=orig_device, dtype=torch.long)
    token_idx = sampled_tokens.to(device=orig_device, dtype=torch.long).unsqueeze(-1)

    # 1. Select the (B, S, V) teacher logits for each batch item b using (d_b, e_b, b)
    selected_teacher_logits = teacher_logits[d_idx, e_idx, batch_idx].to(
        device=orig_device, dtype=orig_dtype
    )

    # 2. Compute log probabilities over vocabulary V and gather sampled token log-probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(selected_teacher_logits, dim=-1)

    student_token_log_probs = student_log_probs.gather(dim=-1, index=token_idx).squeeze(-1)
    teacher_token_log_probs = teacher_log_probs.gather(dim=-1, index=token_idx).squeeze(-1)

    # 3. Stop gradient and clip the log-probability gap to [-R_max, R_max]
    raw_gap = (teacher_token_log_probs - student_token_log_probs).detach()
    rewards = torch.clamp(raw_gap, min=-clip_threshold, max=clip_threshold)

    return {
        "rewards": rewards.to(dtype=orig_dtype, device=orig_device),
        "teacher_token_log_probs": teacher_token_log_probs.to(dtype=orig_dtype, device=orig_device),
    }
