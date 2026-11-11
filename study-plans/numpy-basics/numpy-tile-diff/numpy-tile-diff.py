import numpy as np

def tile_diff(data: list, reps: int) -> np.ndarray:
    """
    Returns float64 slices of tiled values and next-row differences with a final zero row.
    """
    # 1. Convert to a float64 numpy array and tile it vertically
    T = np.tile(np.array(data, dtype=np.float64), (reps, 1))
    
    # 2. Allocate an empty array for differences filled with zeros
    D = np.zeros_like(T)
    
    # 3. Compute consecutive row differences for all but the last row (T[i+1] - T[i])
    D[:-1] = T[1:] - T[:-1]
    
    # 4. Stack T and D along a new first axis to match the shape (2, m * reps, n)
    return np.stack([T, D], axis=0)
