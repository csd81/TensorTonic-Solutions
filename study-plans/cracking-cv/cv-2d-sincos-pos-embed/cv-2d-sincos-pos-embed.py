import numpy as np

def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> list:
    """
    Returns a float list of shape (grid_h * grid_w, embed_dim), rounded to 4 decimals.
    """
    # Each axis (row and col) gets half of the total embedding dimension
    axis_dim = embed_dim // 2
    
    # Generate coordinates for the grid in row-major order
    rows = np.arange(grid_h, dtype=np.float64)
    cols = np.arange(grid_w, dtype=np.float64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing='ij')
    grid_r = grid_r.flatten()
    grid_c = grid_c.flatten()
    
    # Compute the frequencies (omega) for the axis embedding
    # i ranges from 0 to (axis_dim / 2) - 1
    omega = 1.0 / (10000.0 ** (np.arange(axis_dim // 2, dtype=np.float64) * 2 / axis_dim))
    
    # Compute 1D sinusoidal embeddings for rows
    out_r = np.outer(grid_r, omega)
    emb_r = np.concatenate([np.sin(out_r), np.cos(out_r)], axis=1)
    
    # Compute 1D sinusoidal embeddings for columns
    out_c = np.outer(grid_c, omega)
    emb_c = np.concatenate([np.sin(out_c), np.cos(out_c)], axis=1)
    
    # Concatenate row and column embeddings
    emb = np.concatenate([emb_r, emb_c], axis=1)
    
    # Round to 4 decimal places and return as a nested list
    return np.round(emb, 4).tolist()
