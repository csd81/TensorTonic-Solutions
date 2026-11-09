def bpe_tokenize(word_freqs: dict, num_merges: int) -> dict:
    # Initialize word splits by separating into individual characters.
    # The final character gets '</w>' attached to it instead of being a separate symbol.
    splits = {}
    for word, freq in word_freqs.items():
        if not word:
            continue
        chars = list(word)
        chars[-1] += "</w>"
        splits[word] = chars

    merges = []

    for _ in range(num_merges):
        # Count frequencies of all adjacent pairs across all words
        pair_counts = {}
        for word, freq in word_freqs.items():
            chars = splits[word]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            break

        # Find the pair with the maximum frequency-weighted count.
        # Ties are broken by alphabetical order of the first symbol, then the second.
        best_pair = min(
            pair_counts.keys(),
            key=lambda p: (-pair_counts[p], p[0], p[1])
        )

        merges.append([best_pair[0], best_pair[1]])

        # Apply the merge to all word splits (non-overlapping, left-to-right)
        target_first, target_second = best_pair
        merged_symbol = target_first + target_second

        for word in word_freqs:
            chars = splits[word]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == target_first and chars[i+1] == target_second:
                    new_chars.append(merged_symbol)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            splits[word] = new_chars

    # Gather the final remaining symbols across all active splits
    vocab_set = set()
    for chars in splits.values():
        vocab_set.update(chars)
    
    return {
        "merges": merges,
        "vocab": sorted(list(vocab_set))
    }
