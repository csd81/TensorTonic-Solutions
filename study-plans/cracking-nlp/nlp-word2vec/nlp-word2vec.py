import numpy as np

def word2vec(corpus: list, embedding_dim: int = 10, window_size: int = 2, num_negatives: int = 5, epochs: int = 50, learning_rate: float = 0.01, seed: int = 42) -> list:
    """
    Trains a deterministic skip-gram model with negative sampling using legacy NumPy random generation.
    Returns the input embedding matrix as a Python list of float lists, rounded to four decimals.
    """
    # 1. Flatten the documents in order
    tokens = []
    for doc in corpus:
        tokens.extend(doc)
    
    # 2. Define vocabulary ordered by Python string sorting
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    
    # Compute token counts for the negative-sampling distribution
    counts = {w: 0 for w in vocab}
    for w in tokens:
        counts[w] += 1
        
    # P(w) proportional to f(w)^(3/4)
    p_weights = np.array([counts[vocab[i]]**0.75 for i in range(vocab_size)], dtype=np.float64)
    P = p_weights / np.sum(p_weights)
    
    # 3. Deterministic Weight Initialization
    rng = np.random.RandomState(seed)
    W_in = rng.randn(vocab_size, embedding_dim).astype(np.float64) * 0.01
    W_out = rng.randn(vocab_size, embedding_dim).astype(np.float64) * 0.01
    
    # 4. Generate center-context pairs across flat text
    pairs = []
    n_tokens = len(tokens)
    for c_idx in range(n_tokens):
        left_bound = max(0, c_idx - window_size)
        right_bound = min(n_tokens, c_idx + window_size + 1)
        for o_idx in range(left_bound, right_bound):
            if o_idx == c_idx:
                continue
            pairs.append((w2i[tokens[c_idx]], w2i[tokens[o_idx]]))
            
    # Sigmoid activation helper with required clip boundary
    def sigmoid(x):
        clipped = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))
        
    # 5. Stochastic Gradient Descent Training Loop
    for epoch in range(epochs):
        for c_w, o_w in pairs:
            # Draw all K negatives in one call per pair
            negatives = rng.choice(vocab_size, size=num_negatives, replace=True, p=P)
            
            # Snap input center snapshot vector to freeze it within this pair's update chain
            v_c = W_in[c_w].copy()
            grad_c = np.zeros_like(v_c)
            
            # --- Positive context target update ---
            score_pos = np.dot(v_c, W_out[o_w])
            sig_pos = sigmoid(score_pos)
            g_pos = 1.0 - sig_pos
            
            grad_c += g_pos * W_out[o_w]
            W_out[o_w] += learning_rate * g_pos * v_c
            
            # --- Negative samples sequentially applied updates ---
            for neg_w in negatives:
                score_neg = np.dot(v_c, W_out[neg_w])
                sig_neg = sigmoid(score_neg)
                g_neg = -sig_neg
                
                grad_c += g_neg * W_out[neg_w]
                W_out[neg_w] += learning_rate * g_neg * v_c
                
            # Accumulate and execute center vector update exactly once at completion of pair processing
            W_in[c_w] += learning_rate * grad_c
            
    return np.round(W_in, 4).tolist()
