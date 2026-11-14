import math


def label_smoothing_loss(
    predictions: list, target: int, epsilon: float
) -> float:
  """Returns cross-entropy loss for the smoothed target distribution."""
  K = len(predictions)
  smooth_other = epsilon / K
  smooth_target = (1.0 - epsilon) + smooth_other

  loss = 0.0
  for i, p in enumerate(predictions):
    q_i = smooth_target if i == target else smooth_other
    loss -= q_i * math.log(p)

  return float(loss)
