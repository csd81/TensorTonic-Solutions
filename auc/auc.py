import numpy as np


def auc(fpr: list, tpr: list) -> float:
  """Returns the area as a float."""
  fpr = np.asarray(fpr)
  tpr = np.asarray(tpr)

  # Compute interval widths (x_{i+1} - x_i)
  dx = np.diff(fpr)

  # Compute average TPR values for each interval ((y_i + y_{i+1}) / 2)
  avg_y = (tpr[:-1] + tpr[1:]) / 2.0

  # Sum the trapezoid areas
  return float(np.sum(dx * avg_y))
