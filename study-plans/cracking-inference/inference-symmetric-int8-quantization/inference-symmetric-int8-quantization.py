import torch

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> tuple:
    """
    Returns (quantized, scale, dequantized): 
    int8 codes, scalar float scale, float reconstruction.
    """
    # Find the maximum absolute value in the tensor
    max_val = torch.max(torch.abs(x))
    
    # Handle the all-zero tensor edge case
    if max_val == 0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        quantized = torch.zeros_like(x, dtype=torch.int8)
        dequantized = torch.zeros_like(x, dtype=torch.float32)
        return quantized, scale, dequantized

    # Calculate scale factor
    scale = max_val / 127.0
    
    # Quantize: divide by scale, round to nearest even, and clip to [-127, 127]
    # torch.round() handles round-to-nearest-even by default
    quantized = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    
    # Dequantize (reconstruct) the floating-point values
    dequantized = quantized.to(torch.float32) * scale
    
    return quantized, scale, dequantized
