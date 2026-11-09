def language_model(corpus: list, contexts: list, alpha: float = 1) -> list:
    """
    Returns one predicted next-word string per context.
    """
    from collections import defaultdict

    # 1. Build the vocabulary of distinct corpus tokens
    vocab_set = set()
    for sentence in corpus:
        for token in sentence:
            vocab_set.add(token)
    
    # Sort the vocabulary to easily handle ties alphabetically by Python string order
    vocab = sorted(list(vocab_set))
    V_size = len(vocab)
    
    # 2. Count adjacent pairs within each sentence only
    # context_counts[c] stores the total number of bigrams starting with c: C(c)
    # bigram_counts[c][w] stores the count of adjacent pair (c, w): C(c, w)
    context_counts = defaultdict(int)
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            c = sentence[i]
            w = sentence[i + 1]
            bigram_counts[c][w] += 1
            context_counts[c] += 1
            
    predictions = []
    
    # 3. Predict the next word for each context
    for c in contexts:
        # Uniform distribution case (unseen contexts or alpha > 0 uniform ranking)
        # If the context 'c' has no outgoing training bigrams, or under uniform additive smoothing,
        # we can determine the winner directly.
        # Since alpha >= 0, uniform additive smoothing changes probabilities but NOT their relative ranking
        # for a fixed context with existing counts. 
        # If C(c) == 0 (unseen or sentence-final-only context), any word in the vocabulary gives equal probability, 
        # so the alphabetically first word wins.
        if context_counts[c] == 0:
            predictions.append(vocab[0] if vocab else "")
            continue
            
        # If the context exists, find the word 'w' that maximizes C(c, w).
        # We iterate over the sorted vocabulary so ties are naturally broken by alphabetical order.
        best_w = None
        max_count = -1
        
        for w in vocab:
            count = bigram_counts[c][w]
            if count > max_count:
                max_count = count
                best_w = w
                
        predictions.append(best_w)
        
    return predictions
