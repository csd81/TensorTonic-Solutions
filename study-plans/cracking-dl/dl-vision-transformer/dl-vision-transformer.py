import numpy as np

def vit_patch_embed(image: list, patch_size: int, W_proj: list, b_proj: list, W_cls: list, W_pos: list) -> dict:
    """
    Returns a dictionary of lists: patches, patch_embeddings, tokens, output.
    """
    # Convert input lists to NumPy arrays with float64 precision
    img_arr = np.array(image, dtype=np.float64)
    W_proj_arr = np.array(W_proj, dtype=np.float64)
    b_proj_arr = np.array(b_proj, dtype=np.float64)
    W_cls_arr = np.array(W_cls, dtype=np.float64)
    W_pos_arr = np.array(W_pos, dtype=np.float64)
    
    C, H, W = img_arr.shape
    p = patch_size
    
    # Calculate the number of complete patches along height and width
    num_patches_h = H // p
    num_patches_w = W // p
    
    # Extract patches in row-major order and flatten in (C, p, p) order
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = img_arr[:, i*p:(i+1)*p, j*p:(j+1)*p]
            patches.append(patch.flatten())
            
    P = np.array(patches)
    
    # Linear projection: E = P * W_proj^T + b_proj
    E = P @ W_proj_arr.T + b_proj_arr
    
    # Prepend the class token
    tokens = np.vstack([W_cls_arr, E])
    
    # Add positional embeddings elementwise
    output = tokens + W_pos_arr
    
    # Round results to 4 decimal places and convert back to Python nested lists
    return {
        'patches': np.round(P, 4).tolist(),
        'patch_embeddings': np.round(E, 4).tolist(),
        'tokens': np.round(tokens, 4).tolist(),
        'output': np.round(output, 4).tolist()
    }
