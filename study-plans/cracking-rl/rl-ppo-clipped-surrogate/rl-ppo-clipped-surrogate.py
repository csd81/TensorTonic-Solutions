import math


def ppo_clipped_loss(
    log_probs_new: list, log_probs_old: list, advantages: list, clip_eps: float
) -> float:
  """Returns the PPO clipped loss as a float rounded to four decimals."""
  T = len(log_probs_new)
  total_loss = 0.0
  for t in range(T):
    r_t = math.exp(log_probs_new[t] - log_probs_old[t])
    A_t = advantages[t]
    term1 = r_t * A_t
    clipped_r_t = max(1.0 - clip_eps, min(1.0 + clip_eps, r_t))
    term2 = clipped_r_t * A_t
    L_t = min(term1, term2)
    total_loss += L_t
  loss = -(1.0 / T) * total_loss
  return round(loss, 4)
