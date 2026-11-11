import torch


def neuron_gradient_check(
    inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, h: float
) -> tuple:
    """
    Returns five tensors: analytic/numerical weight gradients, analytic/numerical bias gradients, maximum error.
    """
    device = inputs.device
    dtype = torch.float64

    # 1. Cast inputs and step size to float64 on the target device
    x = inputs.to(device=device, dtype=dtype)
    w = weights.to(device=device, dtype=dtype)
    b = bias.to(device=device, dtype=dtype)
    h_t = torch.tensor(h, device=device, dtype=dtype)

    # 2. Forward pass and analytic gradients
    z = torch.dot(w, x) + b
    y = torch.tanh(z)
    dy = 1.0 - y * y

    analytic_w = dy * x
    analytic_b = dy

    # 3. Numerical gradients using forward finite differences
    # Perturbing w_i by +h shifts pre-activation z by +h * x_i
    y_w_pert = torch.tanh(z + h_t * x)
    numerical_w = (y_w_pert - y) / h_t

    # Perturbing b by +h shifts pre-activation z by +h
    y_b_pert = torch.tanh(z + h_t)
    numerical_b = (y_b_pert - y) / h_t

    # 4. Maximum absolute disagreement across all parameters
    max_err = torch.abs(analytic_b - numerical_b)
    if x.numel() > 0:
        max_w_err = torch.max(torch.abs(analytic_w - numerical_w))
        max_err = torch.maximum(max_err, max_w_err)

    return analytic_w, numerical_w, analytic_b, numerical_b, max_err
