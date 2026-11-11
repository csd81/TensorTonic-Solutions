import torch

def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:
    """
    Returns a dict: tokens (image dtype), coordinates (int64 tensor).
    """
    batch, channels, height, width = images.shape
    device = images.device

    grid_h = height // patch_height
    grid_w = width // patch_width
    patch_count = grid_h * grid_w
    patch_dim = channels * patch_height * patch_width

    # Reshape (B, C, H, W) -> (B, C, grid_h, patch_height, grid_w, patch_width)
    # Permute -> (B, grid_h, grid_w, C, patch_height, patch_width)
    # Flatten -> (B, patch_count, patch_dim)
    tokens = (
        images.reshape(batch, channels, grid_h, patch_height, grid_w, patch_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, patch_count, patch_dim)
    )

    rows = torch.arange(grid_h, dtype=torch.int64, device=device)
    cols = torch.arange(grid_w, dtype=torch.int64, device=device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    coordinates = torch.stack([grid_rows.reshape(-1), grid_cols.reshape(-1)], dim=1)

    return {
        "tokens": tokens,
        "coordinates": coordinates,
    }
