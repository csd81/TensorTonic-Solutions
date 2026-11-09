def masked_language_model(tokens: list, mask_positions: list, logits: list, vocab: list) -> list:
    """
    Returns one predicted vocabulary string per supplied mask position.
    """
    predictions = []
    
    for row in logits:
        # Find the index of the maximum value.
        # Python's max() with enumerate automatically breaks ties by the smaller 
        # index because it evaluates sequentially and keeps the first maximum seen.
        best_idx, _ = max(enumerate(row), key=lambda x: x[1])
        
        # Map the index to the corresponding vocabulary word
        predictions.append(vocab[best_idx])
        
    return predictions
