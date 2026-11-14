def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    step = chunk_size - overlap
    chunks = []
    
    for i in range(0, len(tokens), step):
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop once a chunk reaches the end of the token list
        if i + chunk_size >= len(tokens):
            break
            
    return chunks
