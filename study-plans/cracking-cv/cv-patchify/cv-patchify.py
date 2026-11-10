import torch

def patchify(images: list, patch_size: int) -> list:
    """
    Splits images of shape (B, C, H, W) into non-overlapping square patches 
    of side patch_size, returning a nested list of shape (B, N, P * P * C).
    """
    # 1. Convert input list to float64 tensor as required
    tensor = torch.tensor(images, dtype=torch.float64)
    B, C, H, W = tensor.shape
    P = patch_size
    
    # 2. Compute the number of patches along height and width
    num_patches_h = H // P
    num_patches_w = W // P
    
    # 3. Reshape to split H and W into patches
    # (B, C, H, W) -> (B, C, num_patches_h, P, num_patches_w, P)
    x = tensor.view(B, C, num_patches_h, P, num_patches_w, P)
    
    # 4. Permute to order: Row-major patches, then Row -> Col -> Channel within patch
    # Target order: (B, num_patches_h, num_patches_w, P, P, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    
    # 5. Flatten into (B, N, P * P * C)
    num_patches = num_patches_h * num_patches_w
    patch_dim = P * P * C
    x = x.view(B, num_patches, patch_dim)
    
    # 6. Round to 4 decimal places and convert back to nested Python list
    return torch.round(x, decimals=4).tolist()
