def average_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping average-pooled windows.
    """
    H = len(X)
    W = len(X[0])
    
    H_out = H // pool_size
    W_out = W // pool_size
    
    pool_area = pool_size * pool_size
    out = []
    
    for i in range(H_out):
        row = []
        for j in range(W_out):
            window_sum = 0.0
            for a in range(pool_size):
                for b in range(pool_size):
                    window_sum += X[i * pool_size + a][j * pool_size + b]
            row.append(window_sum / pool_area)
        out.append(row)
        
    return out
