def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
  """Returns the REINFORCE loss with a mean-return baseline."""
  T = len(rewards)
  returns = [0.0] * T
  running_return = 0.0

  for t in range(T - 1, -1, -1):
    running_return = rewards[t] + gamma * running_return
    returns[t] = running_return

  mean_return = sum(returns) / T
  advantages = [g - mean_return for g in returns]

  loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / T
  return float(loss)
