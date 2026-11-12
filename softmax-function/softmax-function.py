import numpy as np


def softmax(x: list) -> np.ndarray:
  """Returns stable softmax probabilities as a NumPy array matching the shape of x."""
  arr = np.asarray(x, dtype=np.float64)
  shifted = arr - np.max(arr, axis=-1, keepdims=True)
  exp_vals = np.exp(shifted)
  return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
