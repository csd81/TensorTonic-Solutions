from collections import Counter

def bpe_train(text: str, target_vocab_size: int) -> dict:
    """
    Returns merge rules and newly created byte tokens in a dictionary.
    """
    # Convert text to initial UTF-8 byte token IDs (0-255)
    tokens = list(text.encode("utf-8"))
    
    # Map each token ID to its underlying list of raw byte values
    vocab = {i: [i] for i in range(256)}
    
    merge_rules = []
    new_tokens = []
    next_id = 256

    while next_id < target_vocab_size and len(tokens) >= 2:
        # 1. Count all adjacent pairs
        pair_counts = Counter(zip(tokens, tokens[1:]))
        if not pair_counts:
            break

        # 2. Select most frequent pair; break ties by smallest first ID, then second ID
        best_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p[0], p[1]))

        # 3. Replace non-overlapping occurrences from left to right
        new_sequence = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_sequence.append(next_id)
                i += 2
            else:
                new_sequence.append(tokens[i])
                i += 1
        tokens = new_sequence

        # 4. Record the merge rule and expand the new token's raw bytes
        p0, p1 = best_pair
        merged_bytes = vocab[p0] + vocab[p1]
        vocab[next_id] = merged_bytes
        
        merge_rules.append([p0, p1])
        new_tokens.append(merged_bytes)
        
        next_id += 1

    return {
        "merge_rules": merge_rules,
        "new_tokens": new_tokens,
    }
