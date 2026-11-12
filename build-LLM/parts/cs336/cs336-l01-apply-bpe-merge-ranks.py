def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    if not text:
        return []
    
    # Start with the UTF-8 byte IDs
    tokens = list(text.encode("utf-8"))
    
    # Apply each rule sequentially in list order
    for left_id, right_id, new_id in merges:
        new_tokens = []
        i = 0
        n = len(tokens)
        
        while i < n:
            # Check if the pair matches from left to right
            if i < n - 1 and tokens[i] == left_id and tokens[i + 1] == right_id:
                new_tokens.append(new_id)
                i += 2  # Skip both since they are merged
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        tokens = new_tokens
        
    return tokens

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    if not ids:
        return ""
        
    # Reconstruct the complete byte sequence
    all_bytes = []
    for token_id in ids:
        all_bytes.extend(vocab[token_id])
        
    # Interpret the complete sequence as UTF-8
    return bytes(all_bytes).decode("utf-8", errors="replace")
