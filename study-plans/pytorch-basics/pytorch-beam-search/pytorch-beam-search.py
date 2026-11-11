def beam_search(log_probs_fn, start_token: int, end_token: int, beam_width: int, max_len: int) -> list[int]:
    """
    Returns token IDs including the start token but excluding a terminal end token.
    """
    if start_token == end_token:
        return []

    # Active beams stored as: (score, tokens_list)
    active_beams = [(0.0, [start_token])]
    # Completed beams stored as: (score, tokens_list_without_end_token)
    completed_beams = []

    for _ in range(max_len):
        if not active_beams:
            break

        candidates = []
        # beam_idx preserves existing beam order for tie-breaking
        for beam_idx, (score, tokens) in enumerate(active_beams):
            log_probs = log_probs_fn(tokens)
            # token_id is naturally iterated in ascending order (0, 1, ...)
            for token_id, token_score in enumerate(log_probs):
                candidates.append((score + token_score, beam_idx, token_id, tokens))

        # Sort by:
        # 1. Descending total score (-cand[0])
        # 2. Ascending previous beam order (cand[1])
        # 3. Ascending next-token ID (cand[2])
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        top_candidates = candidates[:beam_width]

        next_active_beams = []
        for total_score, _, token_id, tokens in top_candidates:
            if token_id == end_token:
                # Exclude terminal end_token when storing completed beams
                completed_beams.append((total_score, tokens))
            else:
                next_active_beams.append((total_score, tokens + [token_id]))

        active_beams = next_active_beams

    # Combine completed beams first (in completion order), then surviving active beams
    all_final_beams = completed_beams + active_beams

    # Python's max() with a key returns the first item encountered on ties,
    # naturally favoring completed beams in completion order over active beams.
    best_score, best_tokens = max(all_final_beams, key=lambda x: x[0])
    return best_tokens
