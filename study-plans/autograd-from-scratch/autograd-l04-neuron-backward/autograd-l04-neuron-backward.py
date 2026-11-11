import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    dtype = inputs.dtype
    device = inputs.device

    # Do the math in float64 when asked for it; promote float16 to float32
    # so tanh and the 1 - y^2 term keep enough precision.
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    x = inputs.to(compute_dtype)
    w = weights.to(compute_dtype)
    b = bias.to(compute_dtype)
    g = upstream_gradient.to(compute_dtype)

    a = (x * w).sum() + b          # preactivation (sum of empty -> 0, keeps b)
    y = torch.tanh(a)              # output
    delta = g * (1.0 - y * y)      # dL/da = g * tanh'(a)

    dx = delta * w                 # dL/dx_i = delta * w_i
    dw = delta * x                 # dL/dw_i = delta * x_i
    db = delta                     # dL/db   = delta

    return (y.to(dtype), dx.to(dtype), dw.to(dtype), db.to(dtype))