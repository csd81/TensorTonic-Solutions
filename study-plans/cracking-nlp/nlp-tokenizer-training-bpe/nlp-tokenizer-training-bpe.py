from collections import defaultdict

def train_bpe(word_freqs: dict, num_merges: int) -> dict:
    # 1. Initialize character-level splits with the </w> token
    splits = {}
    vocab = set(['</w>'])
    
    for word in word_freqs:
        # Construct the initial split list: individual characters + '</w>'
        word_splits = list(word) + ['</w>']
        splits[word] = word_splits
        for symbol in word_splits:
            vocab.add(symbol)
            
    merges = []
    merge_frequencies = []
    
    # 2. Perform BPE merges
    for _ in range(num_merges):
        pair_counts = defaultdict(int)
        
        # Count all adjacent pairs weighted by word frequencies
        for word, word_splits in splits.items():
            freq = word_freqs[word]
            for i in range(len(word_splits) - 1):
                pair = (word_splits[i], word_splits[i+1])
                pair_counts[pair] += freq
                
        if not pair_counts:
            break
            
        # Find the maximum frequency
        max_freq = max(pair_counts.values())
        
        # Collect candidates and break ties alphabetically
        candidates = [pair for pair, count in pair_counts.items() if count == max_freq]
        best_pair = min(candidates, key=lambda p: (p[0], p[1]))
        
        # Record the merge
        merges.append([best_pair[0], best_pair[1]])
        merge_frequencies.append(max_freq)
        
        # Define the newly created symbol
        new_symbol = best_pair[0] + best_pair[1]
        vocab.add(new_symbol)
        
        # 3. Replace non-overlapping occurrences left-to-right
        new_splits = {}
        for word, word_splits in splits.items():
            replaced_splits = []
            i = 0
            while i < len(word_splits):
                if i < len(word_splits) - 1 and word_splits[i] == best_pair[0] and word_splits[i+1] == best_pair[1]:
                    replaced_splits.append(new_symbol)
                    i += 2  # Skip past the matched pair (left-to-right, non-overlapping)
                else:
                    replaced_splits.append(word_splits[i])
                    i += 1
            new_splits[word] = replaced_splits
            
        splits = new_splits

    # 4. Prepare the sorted vocabulary list and counts
    sorted_vocab = sorted(list(vocab))
    
    return {
        "merges": merges,
        "vocab": sorted_vocab,
        "vocab_size": len(sorted_vocab),
        "final_splits": splits,
        "merge_frequencies": merge_frequencies
    }
