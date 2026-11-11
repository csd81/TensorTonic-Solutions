import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    # Clone and enable gradient tracking on the leaf tensor to avoid modifying the original input
    x = values.clone().detach().requires_grad_(True)
    
    # Forward pass: compute the function y = sum(x^3 + 2x)
    y = torch.sum(x**3 + 2 * x)
    
    # Backward pass: compute the gradients of y with respect to x
    y.backward()
    
    # Return the accumulated gradients
    return x.grad
