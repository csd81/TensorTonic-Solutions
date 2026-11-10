import torch

def window_partition_reverse(x: list, window_size: int) -> dict:
    """
    Returns a dictionary with partitioned and reversed, both nested float lists.
    """
    # Convert input to a float64 PyTorch tensor
    t = torch.tensor(x, dtype=torch.float64)
    B, H, W, C = t.shape
    
    nh = H // window_size
    nw = W // window_size
    
    # 1. Window Partition
    # Split H and W dimensions into (number of windows, window_size)
    t_part = t.view(B, nh, window_size, nw, window_size, C)
    # Permute to group by batch, window row, and window col: (B, nh, nw, ws, ws, C)
    t_part = t_part.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Flatten the first three dimensions to get the total number of windows
    t_part = t_part.view(-1, window_size, window_size, C)
    
    # 2. Window Reverse (Reconstruct original shape)
    t_rev = t_part.view(B, nh, nw, window_size, window_size, C)
    t_rev = t_rev.permute(0, 1, 3, 2, 4, 5).contiguous()
    t_rev = t_rev.view(B, H, W, C)
    
    # Round to 4 decimal places and convert back to nested Python lists
    partitioned_list = torch.round(t_part, decimals=4).tolist()
    reversed_list = torch.round(t_rev, decimals=4).tolist()
    
    return {
        "partitioned": partitioned_list,
        "reversed": reversed_list
    }
