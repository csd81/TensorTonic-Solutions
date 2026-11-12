import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    # Keep track of the original dtype to restore it at the end
    orig_dtype = x.dtype
    
    # Cast to float32 for high-precision arithmetic during normalization
    x_f32 = x.to(torch.float32)
    g_f32 = g.to(torch.float32)
    
    # Calculate the root mean square over the last dimension
    rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + epsilon)
    
    # Normalize and scale
    output = (x_f32 / rms) * g_f32
    
    return output.to(orig_dtype)
