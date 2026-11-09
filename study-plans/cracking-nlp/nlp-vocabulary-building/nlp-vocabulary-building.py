from collections import Counter

def build_vocabulary(corpus: list, max_vocab_size: int) -> dict:
    """
    Returns a dictionary with vocab (token IDs) and encoded (integer lists).
    """
    # Reserve the standard tokens at IDs 0 and 1
    vocab = {"<pad>": 0, "<unk>": 1}
    reserved = {"<pad>", "<unk>"}
    
    # 1. Count occurrences of distinct tokens, ignoring reserved tokens
    counts = Counter()
    for sentence in corpus:
        for token in sentence:
            if token not in reserved:
                counts[token] += 1
            
    # 2. Sort tokens by decreasing count, breaking ties by ascending string order
    sorted_tokens = sorted(counts.keys(), key=lambda token: (-counts[token], token))
    
    # 3. Limit available vocabulary slots based on max_vocab_size
    allowed_slots = max(0, max_vocab_size - 2)
    kept_tokens = sorted_tokens[:allowed_slots]
    
    # 4. Map the kept tokens to sequential integers starting at ID 2
    for i, token in enumerate(kept_tokens):
        vocab[token] = i + 2
        
    # 5. Encode the corpus using the finalized vocabulary mapping
    encoded = []
    for sentence in corpus:
        encoded_sentence = [vocab.get(token, 1) for token in sentence]
        encoded.append(encoded_sentence)
        
    return {"vocab": vocab, "encoded": encoded}