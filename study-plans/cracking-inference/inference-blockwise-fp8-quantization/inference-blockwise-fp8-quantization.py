import torch

def blockwise_fp8_quantize(
    x: torch.Tensor,
    row_block_size: int,
    col_block_size: int,
    fp8_max: float,
) -> tuple:
    """
    Returns (scaled, block_scales, dequantized), floating tensors with one scale per 2D block.
    """
    R, C = x.shape
    num_row_blocks = (R + row_block_size - 1) // row_block_size
    num_col_blocks = (C + col_block_size - 1) // col_block_size

    block_scales = torch.zeros((num_row_blocks, num_col_blocks), dtype=x.dtype, device=x.device)
    scaled = torch.zeros_like(x)
    dequantized = torch.zeros_like(x)

    for r_idx in range(num_row_blocks):
        r_start = r_idx * row_block_size
        r_end = min(r_start + row_block_size, R)

        for c_idx in range(num_col_blocks):
            c_start = c_idx * col_block_size
            c_end = min(c_start + col_block_size, C)

            block = x[r_start:r_end, c_start:c_end]
            abs_max = torch.max(torch.abs(block))

            if abs_max == 0:
                scale = 1.0
            else:
                scale = abs_max / fp8_max

            block_scales[r_idx, c_idx] = scale

            # Scale and clip to [-fp8_max, fp8_max]
            y_block = torch.clamp(block / scale, -fp8_max, fp8_max)
            # Reconstruct (X-hat)
            x_hat_block = y_block * scale

            scaled[r_start:r_end, c_start:c_end] = y_block
            dequantized[r_start:r_end, c_start:c_end] = x_hat_block

    return scaled, block_scales, dequantized
