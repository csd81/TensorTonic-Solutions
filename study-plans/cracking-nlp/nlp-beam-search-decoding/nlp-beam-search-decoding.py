import math

def beam_search(prob_table: dict, beam_width: int, max_length: int, start_token: str, end_token: str) -> list:
    """
    Decode a token sequence using beam search and a supplied next-token probability table.
    Returns the selected token list, including start and excluding the terminal end token.
    """
    # Active beams stored as a list of tuples: (cumulative_log_score, sequence_tuple)
    active_beams = [(0.0, (start_token,))]
    completed_beams = []
    
    # max_length counts the start token and any generated end token,
    # allowing at most max_length - 1 rounds of token generation.
    for _ in range(max_length - 1):
        if not active_beams:
            break
            
        candidates = []
        # Expand active beams in their current order to preserve generation order
        for score, seq in active_beams:
            last_token = seq[-1]
            if last_token in prob_table and prob_table[last_token]:
                # Directly iterate to respect insertion order for tie-breaking
                for nxt in prob_table[last_token]:
                    prob = prob_table[last_token][nxt]
                    new_score = score + math.log(prob)
                    new_seq = seq + (nxt,)
                    candidates.append((new_score, new_seq))
                    
        # If an entire round has no extensions, retain the current unfinished beams and stop
        if not candidates:
            break
            
        # Rank all candidates together. Python's built-in sort is stable,
        # which preserves generation order for equal scores.
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Retain at most beam_width candidates
        retained = candidates[:beam_width]
        next_active = []
        
        for score, seq in retained:
            if seq[-1] == end_token:
                # Saved as completed; they do not expand or consume slots in later rounds
                completed_beams.append((score, seq))
            else:
                next_active.append((score, seq))
        active_beams = next_active

    # 1. Prefer the highest-scoring saved completion
    if completed_beams:
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed_beams[0][1]
        return list(best_seq[:-1]) # Omit the end_token
        
    # 2. If nothing completed, use the best remaining unfinished beam
    if active_beams:
        active_beams.sort(key=lambda x: x[0], reverse=True)
        return list(active_beams[0][1])
        
    # 3. If none remains, use the start token alone
    return [start_token]
