import torch
import torch.nn.functional as F

def time_residual_block(x: torch.Tensor, time_embedding: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the block output of shape (B, C_out, H, W) with the dtype of x.
    """
    # 1. Project the timestep embedding: tau = SiLU(e) @ W_t^T + b_t
    tau = F.linear(F.silu(time_embedding), params["time_weight"], params["time_bias"])
    
    # 2. First convolution block + timestep conditioning
    h = F.group_norm(x, num_groups, weight=params["norm1_weight"], bias=params["norm1_bias"], eps=1e-5)
    h = F.silu(h)
    h = F.conv2d(h, params["conv1_weight"], bias=params["conv1_bias"], stride=1, padding=1)
    
    # Reshape tau from (B, C_out) to (B, C_out, 1, 1) to broadcast across spatial dimensions (H, W)
    h = h + tau[:, :, None, None]
    
    # 3. Second convolution block
    h = F.group_norm(h, num_groups, weight=params["norm2_weight"], bias=params["norm2_bias"], eps=1e-5)
    h = F.silu(h)
    h = F.conv2d(h, params["conv2_weight"], bias=params["conv2_bias"], stride=1, padding=1)
    
    # 4. Skip connection
    if "skip_weight" in params:
        skip = F.conv2d(x, params["skip_weight"], bias=params["skip_bias"], stride=1, padding=0)
    else:
        skip = x
        
    return skip + h
