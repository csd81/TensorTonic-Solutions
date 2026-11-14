def ordinal_encoding(values: list, ordering: list) -> list:
    """
    Returns the ordinal index of every input value.
    """
    rank_map = {category: idx for idx, category in enumerate(ordering)}
    return [rank_map[val] for val in values]
