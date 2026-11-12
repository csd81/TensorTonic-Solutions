import numpy as np


def contrastive_loss(
    a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean"
) -> float:
  """Returns the loss as a float."""
  a_arr = np.atleast_2d(np.asarray(a, dtype=np.float64))
  b_arr = np.atleast_2d(np.asarray(b, dtype=np.float64))
  y_arr = np.asarray(y, dtype=np.float64)

  distances = np.linalg.norm(a_arr - b_arr, axis=1)

  similar_loss = y_arr * (distances**2)
  dissimilar_loss = (1.0 - y_arr) * (np.maximum(0.0, margin - distances) ** 2)
  losses = similar_loss + dissimilar_loss

  if reduction == "sum":
    return float(np.sum(losses))
  return float(np.mean(losses))
