import math
import numpy as np


def gelu(x: list) -> np.ndarray:
  """Returns a NumPy array with the same shape as x."""
  arr = np.asarray(x, dtype=np.float64)
  vec_erf = np.vectorize(math.erf, otypes=[np.float64])
  return 0.5 * arr * (1.0 + vec_erf(arr / math.sqrt(2.0)))
