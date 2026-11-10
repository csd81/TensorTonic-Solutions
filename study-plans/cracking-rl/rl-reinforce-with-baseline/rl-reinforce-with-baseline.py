def reinforce_baseline_loss(log_probs: list, returns: list, baselines: list) -> float:
  """
    Returns the baseline-adjusted loss as a float rounded to four decimals.
    """
  T = len(log_probs)
  total = sum(
      ell_t * (G_t - b_t)
      for ell_t, G_t, b_t in zip(log_probs, returns, baselines)
  )
  loss = -(1.0 / T) * total
  return round(float(loss), 4)
