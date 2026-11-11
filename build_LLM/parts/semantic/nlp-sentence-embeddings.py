import numpy as np

def sentence_embeddings(sentences: list, vocab: dict, method: str = 'mean', sif_a: float = 0.001) -> list:
    """
    Returns one list of float coordinates per sentence using the specified pooling method.
    """
    # 1. Infer embedding width from vocab, default to 1 if vocab is empty
    if vocab:
        width = len(next(iter(vocab.values())))
    else:
        width = 1

    # 2. Pre-calculate frequencies for SIF pooling if required
    if method == 'sif':
        # Count frequencies of all tokens across all input sentences
        token_counts = {}
        total_tokens = 0
        for sentence in sentences:
            for token in sentence:
                token_counts[token] = token_counts.get(token, 0) + 1
                total_tokens += 1
        
        # Calculate weights for each unique token in the inputs
        token_weights = {}
        for token, count in token_counts.items():
            p_w = count / total_tokens if total_tokens > 0 else 0
            token_weights[token] = sif_a / (sif_a + p_w)

    output = []

    for sentence in sentences:
        # Handle empty sentences
        if not sentence:
            output.append(np.zeros(width, dtype=np.float64).tolist())
            continue

        # Get or create token vectors (zero vectors for unknown tokens)
        vectors = []
        for token in sentence:
            if token in vocab:
                vectors.append(vocab[token])
            else:
                vectors.append(np.zeros(width, dtype=np.float64))
        
        vectors = np.array(vectors, dtype=np.float64)

        # Apply the chosen pooling method
        if method == 'mean':
            pooled_vector = np.mean(vectors, axis=0)
            
        elif method == 'max':
            pooled_vector = np.max(vectors, axis=0)
            
        elif method == 'sif':
            # Extract the pre-calculated weights for each token occurrence in the sentence
            weights = np.array([token_weights[token] for token in sentence], dtype=np.float64)
            sum_weights = np.sum(weights)
            
            if sum_weights == 0:
                pooled_vector = np.zeros(width, dtype=np.float64)
            else:
                # Weighted average: sum(w_i * v_i) / sum(w_i)
                pooled_vector = np.sum(vectors * weights[:, np.newaxis], axis=0) / sum_weights

        output.append(pooled_vector.tolist())

    return output
