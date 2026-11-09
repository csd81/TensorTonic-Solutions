def segment_sentences(text: str) -> list[str]:
    """
    Splits text into sentences based on a case-sensitive boundary rule and
    a specific list of abbreviations.
    """
    abbreviations = {"Mr", "Mrs", "Dr", "Jr", "Sr", "vs", "etc", "i.e", "e.g", "U.S"}
    sentences = []
    
    start_idx = 0
    n = len(text)
    
    i = 0
    while i < n:
        char = text[i]
        
        # Check for sentence-ending punctuation candidates
        if char in {'.', '!', '?'}:
            is_boundary = False
            
            # Rule 1: Always a boundary if it's the end of the entire text
            if i == n - 1:
                is_boundary = True
            else:
                # Rule 2: Must be followed by whitespace
                if text[i + 1].isspace():
                    # Look ahead past all whitespace to find the next character
                    next_idx = i + 1
                    while next_idx < n and text[next_idx].isspace():
                        next_idx += 1
                    
                    # Boundary if at end of text or followed by an uppercase letter
                    if next_idx == n or text[next_idx].isupper():
                        is_boundary = True
            
            # Rule 3: Abbreviation check for periods ('.') only
            if is_boundary and char == '.':
                # Scan backwards until whitespace or the start of the current sentence
                prev_idx = i - 1
                while prev_idx >= start_idx and not text[prev_idx].isspace():
                    prev_idx -= 1
                
                # Extract the token preceding the period
                token = text[prev_idx + 1:i]
                
                if token in abbreviations:
                    is_boundary = False
            
            # If confirmed boundary, extract and trim the sentence
            if is_boundary:
                sentence = text[start_idx:i + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start_idx = i + 1
                
        i += 1
        
    # Append any remaining nonempty trailing text
    if start_idx < n:
        trailing = text[start_idx:].strip()
        if trailing:
            sentences.append(trailing)
            
    return sentences