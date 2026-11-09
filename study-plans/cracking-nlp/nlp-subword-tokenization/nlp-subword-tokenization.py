def subword_tokenize(word: str, vocab: list) -> list:
    """
    Returns a list of subword strings, including continuation markers.
    """
    # Convert vocabulary to a set for O(1) lookups
    vocab_set = set(vocab)
    tokens = []
    
    i = 0
    n = len(word)
    
    while i < n:
        longest_match = None
        longest_len = 0
        
        # Check every possible prefix length from current position 'i'
        for j in range(i + 1, n + 1):
            substring = word[i:j]
            
            # Determine how this token should look in the vocabulary
            if i == 0:
                token_to_check = substring
            else:
                token_to_check = f"##{substring}"
                
            # If it exists and is longer than our previous match, track it
            if token_to_check in vocab_set:
                if len(substring) > longest_len:
                    longest_match = token_to_check
                    longest_len = len(substring)
        
        if longest_match is not None:
            # Append the longest matching token found
            tokens.append(longest_match)
            i += longest_len
        else:
            # Fallback rule: take a single character
            fallback_char = word[i]
            if i == 0:
                tokens.append(fallback_char)
            else:
                tokens.append(f"##{fallback_char}")
            i += 1
            
    return tokens
