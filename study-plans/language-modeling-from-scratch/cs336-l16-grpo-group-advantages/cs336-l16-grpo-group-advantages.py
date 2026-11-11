import torch

def grpo_group_advantages(
    rewards: torch.Tensor, group_ids: torch.Tensor, epsilon: int | float = 1e-8,
) -> dict:
    """
    Returns a dict: advantages (reward dtype), active_group_mask (Boolean tensor), active_group_count (int).
    """
    dtype = rewards.dtype
    device = rewards.device

    if rewards.numel() == 0:
        return {
            "advantages": torch.empty_like(rewards),
            "active_group_mask": torch.empty(0, dtype=torch.bool, device=device),
            "active_group_count": 0,
        }

    num_groups = int(group_ids.max().item()) + 1

    # Detect constant-reward groups directly via group min == group max
    group_mins = torch.full((num_groups,), float("inf"), dtype=dtype, device=device)
    group_maxs = torch.full((num_groups,), float("-inf"), dtype=dtype, device=device)
    group_mins.scatter_reduce_(0, group_ids, rewards, reduce="amin", include_self=True)
    group_maxs.scatter_reduce_(0, group_ids, rewards, reduce="amax", include_self=True)

    active_group_mask = group_maxs > group_mins
    active_group_count = int(active_group_mask.sum().item())

    # Compute group means and population standard deviations
    group_sums = torch.zeros(num_groups, dtype=dtype, device=device)
    group_counts = torch.zeros(num_groups, dtype=dtype, device=device)
    group_sums.scatter_add_(0, group_ids, rewards)
    group_counts.scatter_add_(0, group_ids, torch.ones_like(rewards))
    group_means = group_sums / group_counts

    gathered_means = group_means[group_ids]
    sq_diff = (rewards - gathered_means) ** 2
    group_sq_diff_sums = torch.zeros(num_groups, dtype=dtype, device=device)
    group_sq_diff_sums.scatter_add_(0, group_ids, sq_diff)
    group_stds = torch.sqrt(group_sq_diff_sums / group_counts)

    eps_t = torch.tensor(epsilon, dtype=dtype, device=device)
    advantages = (rewards - gathered_means) / (group_stds[group_ids] + eps_t)

    # Enforce exact zeros for constant-reward groups
    advantages = torch.where(active_group_mask[group_ids], advantages, torch.zeros_like(advantages))

    return {
        "advantages": advantages,
        "active_group_mask": active_group_mask,
        "active_group_count": active_group_count,
    }
