def bag_of_words(corpus: list) -> dict:
    """
    Returns a dictionary with vocab (token indices) and vectors (integer count lists).
    """
    # 1. Gather all unique tokens from the corpus
    unique_tokens = set()
    for doc in corpus:
        for token in doc:
            unique_tokens.add(token)
            
    # 2. Sort tokens in Python string order and build vocabulary dictionary
    sorted_tokens = sorted(list(unique_tokens))
    vocab = {token: index for index, token in enumerate(sorted_tokens)}
    
    # 3. Create frequency vectors for each document
    vectors = []
    for doc in corpus:
        # Initialize a count row for the current document with zeros
        row = [0] * len(vocab)
        for token in doc:
            if token in vocab:
                row[vocab[token]] += 1
        vectors.append(row)
        
    return {
        "vocab": vocab,
        "vectors": vectors
    }
