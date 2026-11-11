def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    H = len(X)
    W = len(X[0])
    
    H_out = H // pool_size
    W_out = W // pool_size
    
    out = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            # Extract the pool_size x pool_size window and find its maximum
            window_max = max(
                X[r][c]
                for r in range(i * pool_size, (i + 1) * pool_size)
                for c in range(j * pool_size, (j + 1) * pool_size)
            )
            row.append(window_max)
        out.append(row)
        
    return out
