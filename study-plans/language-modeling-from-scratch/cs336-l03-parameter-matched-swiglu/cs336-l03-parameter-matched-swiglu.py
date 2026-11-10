import math
import torch

def parameter_matched_swiglu(
    x: torch.Tensor, w_g: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    # Extract dimensions
    d = x.shape[-1]
    H = w_g.shape[1]
    
    # Calculate target hidden width h
    h = min(H, max(1, math.floor(base_params / (3 * d) + 0.5)))
    parameter_count = 3 * d * h
    
    # Track original dtype to restore it later
    orig_dtype = x.dtype
    
    # Cast tensors to float32 for high-precision intermediate computation
    x_f32 = x.to(torch.float32)
    w_g_sliced = w_g[:, :h].to(torch.float32)
    w_v_sliced = w_v[:, :h].to(torch.float32)
    w_o_sliced = w_o[:h, :].to(torch.float32)
    
    # Compute the SwiGLU forward pass
    gate = torch.nn.functional.silu(x_f32 @ w_g_sliced)
    value = x_f32 @ w_v_sliced
    gated_activation = gate * value
    output = gated_activation @ w_o_sliced
    
    # Restore original dtype
    output = output.to(orig_dtype)
    
    return {
        "output": output,
        "hidden_width": h,
        "parameter_count": parameter_count
    }
