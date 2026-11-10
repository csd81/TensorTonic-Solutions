from collections import defaultdict


def rloo_advantages(rewards: list, group_indices: list) -> list:
  """Returns the leave-one-out advantage list, rounded to four decimals."""
  # Group rewards by their group ID
  group_sums = defaultdict(float)
  group_counts = defaultdict(int)

  for r, g in zip(rewards, group_indices):
    group_sums[g] += r
    group_counts[g] += 1

  advantages = []
  for r, g in zip(rewards, group_indices):
    count = group_counts[g]
    # The sum of other samples in the group is group_sum - r
    other_sum = group_sums[g] - r
    mean_other = other_sum / (count - 1)
    adv = r - mean_other
    advantages.append(round(float(adv), 4))

  return advantages
