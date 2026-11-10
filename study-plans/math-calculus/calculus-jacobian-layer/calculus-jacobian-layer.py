import numpy as np


def jacobian_network_layer(W: list, b: list, x: list) -> dict:
  """Returns layer values and analytical and numerical Jacobians."""
  W = np.array(W, dtype=float)
  b = np.array(b, dtype=float)
  x = np.array(x, dtype=float)

  # Pre-activation z = Wx + b
  z = W @ x + b

  # Activation f = sigmoid(z)
  f = 1.0 / (1.0 + np.exp(-z))

  # Derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))
  s_prime = f * (1.0 - f)

  # Analytical Jacobian J_ij = d f_i / d x_j = sigma'(z_i) * W_ij
  analytical_J = np.diag(s_prime) @ W

  # Numerical Jacobian using central differences with h = 0.00001
  h = 0.00001
  m = len(f)
  n = len(x)
  numerical_J = np.zeros((m, n))

  for j in range(n):
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[j] += h
    x_minus[j] -= h

    f_plus = 1.0 / (1.0 + np.exp(-(W @ x_plus + b)))
    f_minus = 1.0 / (1.0 + np.exp(-(W @ x_minus + b)))

    numerical_J[:, j] = (f_plus - f_minus) / (2.0 * h)

  return {
      "z": z.tolist(),
      "f": f.tolist(),
      "analytical_jacobian": analytical_J.tolist(),
      "numerical_jacobian": numerical_J.tolist(),
  }
