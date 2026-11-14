def bpe_encode(text: str, merge_rules: list, new_tokens: list) -> list:
    """
    Returns encoded token IDs.
    """
    tokens = list(text.encode("utf-8"))

    for i, (pair_a, pair_b) in enumerate(merge_rules):
        new_id = 256 + i
        merged = []
        idx = 0
        while idx < len(tokens):
            if (
                idx + 1 < len(tokens)
                and tokens[idx] == pair_a
                and tokens[idx + 1] == pair_b
            ):
                merged.append(new_id)
                idx += 2
            else:
                merged.append(tokens[idx])
                idx += 1
        tokens = merged

    return tokens


def bpe_decode(token_ids: list, new_tokens: list) -> str:
    """
    Returns decoded UTF-8 text.
    """
    vocab = [[b] for b in range(256)] + new_tokens

    byte_list = []
    for tid in token_ids:
        byte_list.extend(vocab[tid])

    return bytes(byte_list).decode("utf-8")
