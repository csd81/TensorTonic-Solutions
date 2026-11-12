def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    if vocab_size <= 256:
        return {"vocab": [], "merges": []}
    
    # Track the byte sequences using standard dict matching the platform's reference structure
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    sequences = [list(text.encode("utf-8")) for text in corpus]
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = {}
        for sequence in sequences:
            for left, right in zip(sequence, sequence[1:]):
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Follow the platform's precise structural tuple tie-breaker
        left, right = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        
        new_id = len(vocab)
        vocab[new_id] = vocab[left] + vocab[right]
        merges.append([int(left), int(right), int(new_id)])

        updated_sequences = []
        for sequence in sequences:
            updated = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
                    updated.append(new_id)
                    index += 2
                else:
                    updated.append(sequence[index])
                    index += 1
            updated_sequences.append(updated)
        sequences = updated_sequences

    learned_vocab = [
        [int(token_id), [int(b) for b in vocab[token_id]]]
        for token_id in range(256, len(vocab))
    ]
    return {"vocab": learned_vocab, "merges": merges}
