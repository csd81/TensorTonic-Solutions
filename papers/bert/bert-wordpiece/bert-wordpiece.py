def wordpiece_tokenize(text: str, vocab: dict, unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    output_tokens = []
    
    # Lowercase and split on whitespace
    words = text.lower().split()
    
    for word in words:
        # Check max word length constraint
        if len(word) > max_word_length:
            output_tokens.append(unk_token)
            continue
            
        is_bad = False
        start = 0
        sub_tokens = []
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Greedy longest-match-first search from the end of the word backwards
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                    
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
                
            # If no matching subword was found, the whole word becomes [UNK]
            if cur_substr is None:
                is_bad = True
                break
                
            sub_tokens.append(cur_substr)
            start = end
            
        if is_bad:
            output_tokens.append(unk_token)
        else:
            output_tokens.extend(sub_tokens)
            
    return output_tokens
