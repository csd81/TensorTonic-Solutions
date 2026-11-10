import torch

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> tuple:
    """
    Returns (quantized, scale, dequantized):
    int8 codes, broadcastable float scales, float reconstruction.
    """
    # 1. Normalize negative axis indices
    if channel_axis < 0:
        channel_axis += x.dim()
        
    # 2. Find the maximum absolute value along the channel axis
    # We construct a reduction list of all axes except the channel axis
    reduction_dims = [i for i in range(x.dim()) if i != channel_axis]
    
    # Compute max absolute value for each channel
    if reduction_dims:
        max_vals = torch.amax(torch.abs(x), dim=reduction_dims)
    else:
        max_vals = torch.abs(x)
        
    # 3. Calculate per-channel scales
    scales = max_vals / 127.0
    
    # 4. Handle all-zero channels by replacing 0.0 scale with 1.0
    scales = torch.where(scales == 0.0, torch.ones_like(scales), scales)
    
    # 5. Reshape scale tensor to be broadcastable with the original input tensor x
    # It must have size 1 on every axis except the channel_axis
    scale_shape = [1] * x.dim()
    scale_shape[channel_axis] = x.size(channel_axis)
    broadcast_scales = scales.view(scale_shape)
    
    # 6. Quantize: divide by scale, round to nearest even, and clip to [-127, 127]
    # torch.round uses round-to-nearest-even by default
    q = torch.clamp(torch.round(x / broadcast_scales), -127, 127).to(torch.int8)
    
    # 7. Dequantize to get the reconstructed float tensor
    x_hat = q.to(torch.float32) * broadcast_scales
    
    return q, broadcast_scales, x_hat
