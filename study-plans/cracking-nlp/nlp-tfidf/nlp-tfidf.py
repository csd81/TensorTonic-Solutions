import math

def tfidf(corpus: list) -> dict:
    """
    Returns a dictionary with vocab (token indices) and tfidf (float lists).
    """
    # 1. Build and sort the unique vocabulary
    unique_tokens = set()
    for doc in corpus:
        for token in doc:
            unique_tokens.add(token)
            
    sorted_vocab = sorted(list(unique_tokens))
    vocab_dict = {token: idx for idx, token in enumerate(sorted_vocab)}
    
    # 2. Compute Document Frequency (DF) and IDF for each token
    N = len(corpus)
    idf_dict = {}
    
    for token in sorted_vocab:
        # Count how many documents contain the token at least once
        df = sum(1 for doc in corpus if token in doc)
        # Compute unsmoothed natural log IDF
        idf_dict[token] = math.log(N / df)
        
    # 3. Compute TF-IDF matrix
    tfidf_matrix = []
    
    for doc in corpus:
        doc_vector = []
        for token in sorted_vocab:
            # Raw term count in the current document
            tf = doc.count(token)
            # Compute final TF-IDF score
            score = tf * idf_dict[token]
            # Round to 4 decimal places
            doc_vector.append(round(score, 4))
        tfidf_matrix.append(doc_vector)
        
    return {
        "vocab": vocab_dict,
        "tfidf": tfidf_matrix
    }
