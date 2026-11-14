import numpy as np


def local_response_normalization(
    x: np.ndarray, k: float, n: int, alpha: float, beta: float
) -> np.ndarray:
  """Returns the float64 channel-normalized tensor."""
  x = np.asarray(x, dtype=np.float64)
  C = x.shape[-1]
  r = n // 2

  # Square activations and zero-pad along the channel axis to handle boundary clipping
  sq = x**2
  padded = np.pad(sq, ((0, 0), (0, 0), (0, 0), (r, r)), mode="constant")

  # Compute the sliding window sum of length n along the last axis in O(B*H*W*C) time
  csum = np.cumsum(padded, axis=-1)
  csum = np.pad(csum, ((0, 0), (0, 0), (0, 0), (1, 0)), mode="constant")
  window_sum = csum[..., n : n + C] - csum[..., :C]

  # Apply the LRN scaling formula
  denom = (k + (alpha / n) * window_sum) ** beta
  return x / denom
