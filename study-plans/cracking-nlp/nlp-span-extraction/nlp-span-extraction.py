def span_extraction(tokens: list, start_logits: list, end_logits: list, max_span_length: int) -> dict:
    n = len(tokens)
    best_score = float('-inf')
    best_start = 0
    best_end = 0

    # Iterate through all possible start positions
    for i in range(n):
        max_j = min(n, i + max_span_length)
        for j in range(i, max_j):
            score = start_logits[i] + end_logits[j]
            
            # Tie-breaking logic check
            if score > best_score:
                best_score = score
                best_start, best_end = i, j
            elif score == best_score:
                current_len = j - i + 1
                best_len = best_end - best_start + 1
                if current_len < best_len:
                    best_start, best_end = i, j
                elif current_len == best_len:
                    if i < best_start:
                        best_start, best_end = i, j

    return {
        "start": best_start,
        "end": best_end,
        "span": tokens[best_start:best_end + 1],
        "score": round(float(best_score), 4)  # <-- Cast to float here
    }
