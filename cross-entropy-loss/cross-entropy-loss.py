import numpy as np


def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
  """Returns the mean multiclass cross-entropy loss as a Python float."""
  probs = np.asarray(y_pred, dtype=np.float64)
  targets = np.asarray(y_true, dtype=np.int64)

  correct_probs = probs[np.arange(len(targets)), targets]
  loss = -np.mean(np.log(correct_probs))

  return float(loss)
