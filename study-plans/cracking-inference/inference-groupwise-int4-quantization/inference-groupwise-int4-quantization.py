import torch

def groupwise_int4_quantize(weight: torch.Tensor, group_size: int) -> tuple:
    """
    Returns (codes, scales, dequantized): 
    int8 codes, float row-group scales, float reconstruction.
    """
    R, C = weight.shape
    device = weight.device
    
    # Calculate number of groups N per row, rounded up
    N = (C + group_size - 1) // group_size
    
    codes = torch.empty((R, C), dtype=torch.int8, device=device)
    scales = torch.empty((R, N), dtype=torch.float32, device=device)
    dequant = torch.empty((R, C), dtype=torch.float32, device=device)
    
    for r in range(R):
        for g in range(N):
            start_c = g * group_size
            end_c = min(start_c + group_size, C)
            group_weights = weight[r, start_c:end_c]
            
            # An all-zero group uses scale one
            max_val = torch.max(torch.abs(group_weights)).item()
            if max_val == 0.0:
                s = 1.0
            else:
                s = max_val / 7.0
            
            scales[r, g] = s
            
            # Round to nearest integer (half-to-even via torch.round) and clip to [-7, 7]
            q = torch.clamp(torch.round(group_weights / s), -7.0, 7.0).to(torch.int8)
            codes[r, start_c:end_c] = q
            dequant[r, start_c:end_c] = q.to(torch.float32) * s
            
    return codes, scales, dequant
