def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    counts = {}
    for sentence in sentences:
        for token in sentence:
            counts[token] = counts.get(token, 0) + 1
    return counts
