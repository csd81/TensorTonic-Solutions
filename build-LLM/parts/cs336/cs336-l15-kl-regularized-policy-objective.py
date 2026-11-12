import torch

def kl_regularized_policy_objective(
    policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor,
    rewards: torch.Tensor, response_mask: torch.Tensor, beta: int | float,
) -> dict:
    """
    Returns a dict: objective (scalar), penalized_rewards and sampled_divergences (float32 tensors).
    """
    policy_lp = policy_log_probs.to(torch.float32)
    ref_lp = reference_log_probs.to(torch.float32)
    rewards_f32 = rewards.to(torch.float32)
    mask_f32 = response_mask.to(torch.float32)

    token_log_ratios = (policy_lp - ref_lp) * mask_f32
    sampled_divergences = token_log_ratios.sum(dim=-1)
    penalized_rewards = rewards_f32 - float(beta) * sampled_divergences

    if penalized_rewards.numel() == 0:
        objective = torch.tensor(0.0, dtype=torch.float32, device=policy_lp.device)
    else:
        objective = penalized_rewards.mean()

    return {
        "objective": objective,
        "penalized_rewards": penalized_rewards,
        "sampled_divergences": sampled_divergences,
    }
