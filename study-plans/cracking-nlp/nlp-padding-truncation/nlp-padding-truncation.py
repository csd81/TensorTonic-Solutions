def pad_and_truncate(sequences: list, max_length: int, pad_value: int = 0) -> list:
    """
    Returns one integer list of length max_length per input sequence.
    """
    result = []
    for seq in sequences:
        # Truncate if longer than max_length, otherwise keep a copy of seq
        truncated = seq[:max_length]
        # Pad with pad_value if shorter than max_length
        padded = truncated + [pad_value] * (max_length - len(truncated))
        result.append(padded)
    return result
