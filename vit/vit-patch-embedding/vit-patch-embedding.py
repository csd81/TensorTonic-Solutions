import numpy as np


def patch_embed(
    image: np.ndarray,
    patch_size: int,
    W_proj: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
  """
    Returns float64 patch embeddings with shape (B, N, D).
    """
  B, H, W, C = image.shape

  # 1. Calculate the number of complete patches along height and width
  grid_h = H // patch_size
  grid_w = W // patch_size

  # 2. Crop out incomplete patches along the bottom and right boundaries
  cropped = image[:, : grid_h * patch_size, : grid_w * patch_size, :]

  # 3. Reshape into (B, grid_h, patch_size, grid_w, patch_size, C)
  patches = cropped.reshape(B, grid_h, patch_size, grid_w, patch_size, C)

  # 4. Transpose to (B, grid_h, grid_w, patch_size, patch_size, C)
  # This keeps patch grid order (row-major) and intra-patch order (row-major + channels) intact
  patches = patches.transpose(0, 1, 3, 2, 4, 5)

  # 5. Flatten the grid into N patches and each patch into a vector of size (P * P * C)
  N = grid_h * grid_w
  flattened_patches = patches.reshape(B, N, patch_size * patch_size * C)

  # 6. Apply linear projection and bias: z_p = x_p @ W + b
  embeddings = flattened_patches @ W_proj + bias

  return embeddings.astype(np.float64)
