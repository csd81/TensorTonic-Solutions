import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Map each vocabulary word to its corresponding index
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    # Initialize a 1D integer array of zeros
    vector = np.zeros(len(vocab), dtype=int)
    
    # Count occurrences of in-vocabulary tokens
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] += 1
            
    return vector
