import numpy as np

def fasttext(
    corpus: list, 
    queries: list, 
    embedding_dim: int = 10, 
    window_size: int = 2, 
    num_negatives: int = 5, 
    epochs: int = 50, 
    learning_rate: float = 0.01, 
    min_n: int = 3, 
    max_n: int = 6, 
    seed: int = 42
) -> list:
    """
    Trains a simplified subword FastText skip-gram model using NumPy float64.
    Returns a Python list of float lists in query order, each of length embedding_dim,
    rounded to four decimal places.
    """
    # 1. Helper function to extract character n-grams
    def extract_ngrams(word: str, min_len: int, max_len: int) -> list:
        wrapped = f"<{word}>"
        ngrams = []
        for n in range(min_len, max_len + 1):
            for i in range(len(wrapped) - n + 1):
                ngrams.append(wrapped[i:i+n])
        return ngrams

    # 2. Flatten document corpus & build word vocabularies
    flat_corpus = [word for doc in corpus for word in doc]
    word_vocab = sorted(list(set(flat_corpus)))
    word_to_idx = {w: i for i, w in enumerate(word_vocab)}
    
    # Compute unigram negative-sampling probability P(w)
    word_counts = {w: 0 for w in word_vocab}
    for w in flat_corpus:
        word_counts[w] += 1
    freqs = np.array([word_counts[w] for w in word_vocab], dtype=np.float64)
    P = (freqs ** 0.75) / np.sum(freqs ** 0.75)

    # 3. Construct the sorted unique training subword n-gram vocabulary
    all_training_ngrams = []
    for w in flat_corpus:
        all_training_ngrams.extend(extract_ngrams(w, min_n, max_n))
    ngram_vocab = sorted(list(set(all_training_ngrams)))
    ngram_to_idx = {g: i for i, g in enumerate(ngram_vocab)}

    # 4. Initialize embedding matrices using NumPy's legacy seed
    np.random.seed(seed)
    W_in = np.random.randn(len(ngram_vocab), embedding_dim).astype(np.float64) * 0.01
    W_out = np.random.randn(len(word_vocab), embedding_dim).astype(np.float64) * 0.01

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 5. Training loop (epochs -> center words -> context words -> negative sampling updates)
    for epoch in range(epochs):
        for center_idx in range(len(flat_corpus)):
            center_word = flat_corpus[center_idx]
            center_ngrams = extract_ngrams(center_word, min_n, max_n)
            center_ngram_idxs = [ngram_to_idx[g] for g in center_ngrams]
            
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                context_idx = center_idx + offset
                if 0 <= context_idx < len(flat_corpus):
                    context_word = flat_corpus[context_idx]
                    o_idx = word_to_idx[context_word]
                    
                    # Jointly sample K negatives for this center-context pair
                    neg_idxs = np.random.choice(len(word_vocab), size=num_negatives, replace=True, p=P)
                    
                    # Compute static word embedding sum v_w
                    v_w = np.sum(W_in[center_ngram_idxs], axis=0)
                    grad_v = np.zeros(embedding_dim, dtype=np.float64)
                    
                    # Positive context vector update
                    u_o = W_out[o_idx]
                    dot_pos = np.clip(np.dot(v_w, u_o), -500.0, 500.0)
                    sig_pos = sigmoid(dot_pos)
                    
                    grad_v += (1.0 - sig_pos) * u_o
                    W_out[o_idx] += learning_rate * (1.0 - sig_pos) * v_w
                    
                    # Sequential negative context vector updates
                    for neg_idx in neg_idxs:
                        u_n = W_out[neg_idx]
                        dot_neg = np.clip(np.dot(v_w, u_n), -500.0, 500.0)
                        sig_neg = sigmoid(dot_neg)
                        
                        grad_v += (-sig_neg) * u_n
                        W_out[neg_idx] += learning_rate * (-sig_neg) * v_w
                        
                    # Backpropagate accumulated gradient once per subword occurrence
                    for g_idx in center_ngram_idxs:
                        W_in[g_idx] += learning_rate * grad_v

    # 6. Generate outputs for query words
    results = []
    for q in queries:
        q_ngrams = extract_ngrams(q, min_n, max_n)
        q_idxs = [ngram_to_idx[g] for g in q_ngrams if g in ngram_to_idx]
        
        if len(q_idxs) == 0:
            v_q = np.zeros(embedding_dim, dtype=np.float64)
        else:
            v_q = np.sum(W_in[q_idxs], axis=0)
            
        results.append([round(float(val), 4) for val in v_q])
        
    return results
