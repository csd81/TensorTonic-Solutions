import torch

_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

def mxfp4_dequant(packed: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    Returns dequantized float32 blocks of shape (n_blocks, block_size).
    """
    # Ensure device consistency if needed
    device = packed.device
    lut = _FP4_LUT.to(device)

    # Extract low and high nibbles from each uint8 byte
    # Low nibble: packed & 0x0F
    # High nibble: (packed >> 4) & 0x0F
    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F

    # Interleave low and high nibbles to preserve element order (32 elements per block)
    # Shape: (n_blocks, 16, 2) -> reshape to (n_blocks, 32)
    nibbles = torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(packed.shape[0], block_size)

    # Map nibbles through the E2M1 lookup table
    vals = lut[nibbles.long()]

    # Compute E8M0 scale factor: 2^(s - 127)
    # scales is torch.uint8, cast to int32/float32 for exponent calculation
    scale_factors = torch.pow(2.0, scales.to(torch.float32) - 127.0).unsqueeze(1)

    # Multiply each block's values by its scale factor
    dequantized = vals * scale_factors

    return dequantized.to(torch.float32)
