import math
from collections import defaultdict

def grpo_advantages(rewards: list, group_indices: list, eps: float = 1e-8) -> list:
    groups = defaultdict(list)
    for idx, (r, g) in enumerate(zip(rewards, group_indices)):
        groups[g].append((idx, r))

    advantages = [0.0] * len(rewards)

    for group_id, items in groups.items():
        n = len(items)
        if n <= 1:
            continue

        group_rewards = [r for _, r in items]
        if all(r == group_rewards[0] for r in group_rewards):
            continue

        mean = sum(group_rewards) / n
        variance = sum((r - mean) ** 2 for r in group_rewards) / n
        std = math.sqrt(variance)

        if std == 0.0:
            continue

        for idx, r in items:
            adv = (r - mean) / (std + eps)
            val = round(adv, 4)
            advantages[idx] = 0.0 if val == -0.0 else val

    return advantages
