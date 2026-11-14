def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
  """Returns the generalized advantage estimate at every timestep."""
  T = len(rewards)
  advantages = [0.0] * T
  running_gae = 0.0

  for t in range(T - 1, -1, -1):
    delta = rewards[t] + gamma * values[t + 1] - values[t]
    running_gae = delta + gamma * lam * running_gae
    advantages[t] = float(running_gae)

  return advantages
