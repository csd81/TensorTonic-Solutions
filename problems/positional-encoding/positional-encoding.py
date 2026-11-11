import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model) containing positional encodings.
    """
    # 1. Create a column vector for positions: shape (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # 2. Create a row vector for the dimension indices: shape (d_model,)
    j = np.arange(d_model)
    
    # 3. Calculate the 2i term for every dimension index using integer division
    # Even columns (2i) and odd columns (2i + 1) use the same divisor factor
    div_term = base ** ((j // 2 * 2) / d_model)
    
    # 4. Compute all angles via broadcasting: shape (seq_len, d_model)
    angles = pos / div_term
    
    # 5. Initialize the output matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # 6. Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe
