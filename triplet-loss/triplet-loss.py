import numpy as np


def triplet_loss(
    anchor: list, positive: list, negative: list, margin: float = 1.0
) -> float:
  """Returns the loss as a float."""
  a = np.atleast_2d(np.asarray(anchor, dtype=np.float64))
  p = np.atleast_2d(np.asarray(positive, dtype=np.float64))
  n = np.atleast_2d(np.asarray(negative, dtype=np.float64))

  pos_dist = np.sum((a - p) ** 2, axis=1)
  neg_dist = np.sum((a - n) ** 2, axis=1)

  losses = np.maximum(0.0, pos_dist - neg_dist + margin)
  return float(np.mean(losses))
