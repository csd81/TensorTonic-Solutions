def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    stopword_set = set(stopwords)
    return [token for token in tokens if token not in stopword_set]
