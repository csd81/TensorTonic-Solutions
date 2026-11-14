import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    
    # Column vector of positions: shape (seq_length, 1)
    positions = np.arange(seq_length, dtype=np.float64)[:, np.newaxis]
    
    # Even dimension indices (2i): shape (d_model // 2,)
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float64) * (-np.log(10000.0) / d_model)
    )
    
    # Compute angles: shape (seq_length, d_model // 2)
    angles = positions * div_term
    
    # Assign sine to even indices and cosine to odd indices
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    
    return pe
