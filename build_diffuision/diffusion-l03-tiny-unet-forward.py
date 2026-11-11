import math
import torch
import torch.nn.functional as F

def tiny_unet_forward(x: torch.Tensor, timesteps: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the network output of shape (B, C_out, H, W) with the dtype of x.
    """
    dtype = x.dtype
    device = x.device
    B = x.shape[0]
    
    # 1. Timestep Embedding
    # Find E from time_linear1_weight shape (T, E)
    E = params['time_linear1_weight'].shape[1]
    
    # fi = 10000 ** (-2i / E) for i < E // 2
    i = torch.arange(E // 2, dtype=dtype, device=device)
    f = 10000.0 ** (-2.0 * i / E)
    
    # Compute arguments (B, E // 2)
    args = timesteps.to(dtype).unsqueeze(1) * f.unsqueeze(0)
    
    # emb(t) joins cosines and then sines
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    
    # 2. Time MLP
    e = F.linear(emb, params['time_linear1_weight'], params['time_linear1_bias'])
    e = F.silu(e)
    e = F.linear(e, params['time_linear2_weight'], params['time_linear2_bias'])
    
    # Helper function for Residual Blocks
    def run_res_block(x_in, prefix):
        # GN1 -> SiLU -> Conv1
        out = F.group_norm(x_in, num_groups, params[f'{prefix}_norm1_weight'], params[f'{prefix}_norm1_bias'], eps=1e-5)
        out = F.silu(out)
        out = F.conv2d(out, params[f'{prefix}_conv1_weight'], params[f'{prefix}_conv1_bias'], padding=1)
        
        # Time projection addition
        time_proj = F.linear(F.silu(e), params[f'{prefix}_time_weight'], params[f'{prefix}_time_bias'])
        out = out + time_proj.unsqueeze(-1).unsqueeze(-1)
        
        # GN2 -> SiLU -> Conv2
        out = F.group_norm(out, num_groups, params[f'{prefix}_norm2_weight'], params[f'{prefix}_norm2_bias'], eps=1e-5)
        out = F.silu(out)
        out = F.conv2d(out, params[f'{prefix}_conv2_weight'], params[f'{prefix}_conv2_bias'], padding=1)
        
        # Skip connection
        if f'{prefix}_skip_weight' in params:
            skip = F.conv2d(x_in, params[f'{prefix}_skip_weight'], params[f'{prefix}_skip_bias'])
        else:
            skip = x_in
            
        return out + skip

    # 3. Forward Pass through U-Net
    # Input conv
    h1_in = F.conv2d(x, params['conv_in_weight'], params['conv_in_bias'], padding=1)
    h1 = run_res_block(h1_in, 'down_res')
    
    # Downsample (stride 2)
    h1_down = F.conv2d(h1, params['downsample_weight'], params['downsample_bias'], stride=2, padding=1)
    
    # Mid block
    mid = run_res_block(h1_down, 'mid_res')
    
    # Upsample (Repeat 2x2 then conv)
    u = F.interpolate(mid, scale_factor=2.0, mode='nearest')
    u = F.conv2d(u, params['upsample_weight'], params['upsample_bias'], padding=1)
    
    # Concatenate channels with u first
    u_h1 = torch.cat([u, h1], dim=1)
    
    # Up block
    h2 = run_res_block(u_h1, 'up_res')
    
    # Final output layers
    out = F.group_norm(h2, num_groups, params['out_norm_weight'], params['out_norm_bias'], eps=1e-5)
    out = F.silu(out)
    out = F.conv2d(out, params['conv_out_weight'], params['conv_out_bias'], padding=1)
    
    return out
