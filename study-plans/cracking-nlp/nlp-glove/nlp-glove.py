import numpy as np

def glove(
    corpus: list, 
    embedding_dim: int = 10, 
    window_size: int = 2, 
    epochs: int = 50, 
    learning_rate: float = 0.05, 
    x_max: float = 100, 
    alpha: float = 0.75, 
    seed: int = 42
) -> list:
    """
    Trains a compact GloVe model with distance-weighted co-occurrences and AdaGrad.
    Returns the summed center and context embedding matrices as a Python list of lists.
    """
    # 1. Flatten tokens and establish sorted vocabulary
    tokens = [token for doc in corpus for token in doc]
    vocab = sorted(list(set(tokens)))
    w2i = {word: idx for idx, word in enumerate(vocab)}
    V = len(vocab)
    
    # 2. Build the distance-weighted co-occurrence matrix X
    X = np.zeros((V, V), dtype=np.float64)
    for i in range(len(tokens)):
        for j in range(i + 1, min(len(tokens), i + window_size + 1)):
            distance = j - i
            idx_i = w2i[tokens[i]]
            idx_j = w2i[tokens[j]]
            
            if idx_i == idx_j:
                X[idx_i][idx_i] += 2.0 / distance
            else:
                X[idx_i][idx_j] += 1.0 / distance
                X[idx_j][idx_i] += 1.0 / distance
                
    # Extract nonzero coordinates in row-major order
    nonzero_pairs = [(i, j) for i in range(V) for j in range(V) if X[i, j] > 0]
    
    # 3. Initialization using legacy generator seed
    np.random.seed(seed)
    W = np.random.randn(V, embedding_dim).astype(np.float64) * 0.01
    W_tilde = np.random.randn(V, embedding_dim).astype(np.float64) * 0.01
    b = np.zeros(V, dtype=np.float64)
    b_tilde = np.zeros(V, dtype=np.float64)
    
    # Initialize elementwise squared-gradient AdaGrad accumulators to one
    grad_W_acc = np.ones_like(W)
    grad_W_tilde_acc = np.ones_like(W_tilde)
    grad_b_acc = np.ones_like(b)
    grad_b_tilde_acc = np.ones_like(b_tilde)
    
    # 4. AdaGrad online updates
    for epoch in range(epochs):
        for i, j in nonzero_pairs:
            x_ij = X[i, j]
            
            # Compute weighting function value f(x)
            f_x = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            
            # Compute inner error term
            diff = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - np.log(x_ij)
            common_grad = f_x * diff
            
            # Compute gradients based on pre-update parameters
            dW = common_grad * W_tilde[j]
            dW_tilde = common_grad * W[i]
            db = common_grad
            db_tilde = common_grad
            
            # Accumulate historical squared gradients
            grad_W_acc[i] += dW ** 2
            grad_W_tilde_acc[j] += dW_tilde ** 2
            grad_b_acc[i] += db ** 2
            grad_b_tilde_acc[j] += db_tilde ** 2
            
            # Update parameters using updated denominators
            W[i] -= learning_rate * dW / np.sqrt(grad_W_acc[i])
            W_tilde[j] -= learning_rate * dW_tilde / np.sqrt(grad_W_tilde_acc[j])
            b[i] -= learning_rate * db / np.sqrt(grad_b_acc[i])
            b_tilde[j] -= learning_rate * db_tilde / np.sqrt(grad_b_tilde_acc[j])
            
    # Combine the center and context matrices and format output
    combined_embeddings = W + W_tilde
    return np.round(combined_embeddings, 4).tolist()
