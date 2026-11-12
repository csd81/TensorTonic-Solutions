import copy

def sample_recent_replay_batch(
    games: list | tuple,
    window_size: int,
    position_indices: list | tuple,
) -> tuple[list, int]:
    """
    Returns independent sampled records and the eligible position count.
    """
    # 1. Slice the games to only keep the last `window_size` games
    recent_games = games[-window_size:] if window_size < len(games) else games
    
    # 2. Flatten all positions from the recent games into a stable linear pool
    eligible_pool = []
    for game in recent_games:
        for position in game:
            eligible_pool.append(position)
            
    # 3. Retrieve the total number of eligible positions
    eligible_count = len(eligible_pool)
    
    # 4. Map the requested indices to deep-copied records to ensure absolute independence
    sampled_records = [copy.deepcopy(eligible_pool[idx]) for idx in position_indices]
    
    return sampled_records, eligible_count
