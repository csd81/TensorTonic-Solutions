import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    # Convert inputs to 1D float64 NumPy arrays
    vector_a = np.array(a, dtype=np.float64)
    vector_b = np.array(b, dtype=np.float64)
    
    # Calculate Euclidean norms (L2 norms)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Return 0.0 if either vector is a zero vector (norm is 0)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    # Compute dot product
    dot_product = np.dot(vector_a, vector_b)
    
    # Calculate and return cosine similarity as a standard Python float
    return float(dot_product / (norm_a * norm_b))
