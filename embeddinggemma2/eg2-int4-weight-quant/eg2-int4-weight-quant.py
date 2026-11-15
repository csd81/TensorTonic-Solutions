import torch

def quantize_int4(weight: torch.Tensor, mode: str, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of int8 codes, per-group scales and the dequantized weight.
    """
    n_out, n_in = weight.shape
    
    # Determine group size based on mode
    if mode == "per_block":
        group_size = block_size
    elif mode == "per_channel":
        group_size = n_in
    else:
        raise ValueError(f"Unsupported mode: {mode}")
        
    num_groups = n_in // group_size
    
    # Reshape to (n_out, num_groups, group_size) using reshape() to safely handle non-contiguous inputs
    w_grouped = weight.reshape(n_out, num_groups, group_size)
    
    # Calculate symmetric scale per group: max(|w|) / 7
    max_abs = w_grouped.abs().amax(dim=-1)
    scales = max_abs / 7.0
    
    # Avoid division by zero for all-zero groups
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    
    # Quantize: scale, round half-to-even (torch.round's default), and clamp to [-8, 7]
    scaled = w_grouped / safe_scales.unsqueeze(-1)
    codes_grouped = torch.clamp(torch.round(scaled), -8, 7).to(torch.int8)
    
    # Ensure all-zero groups have code 0 (even though 0 / 1 = 0, this makes it explicit)
    codes_grouped = torch.where(scales.unsqueeze(-1) == 0, torch.zeros_like(codes_grouped), codes_grouped)
    
    # Dequantize back to original dtype
    dequant_grouped = codes_grouped.to(weight.dtype) * scales.unsqueeze(-1)
    
    # Restore original 2D matrix shape
    codes = codes_grouped.reshape(n_out, n_in)
    dequantized = dequant_grouped.reshape(n_out, n_in)
    
    return codes, scales, dequantized
