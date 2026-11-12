import numpy as np

def legal_go_action_mask(board: list, player: int, seen_positions: list, consecutive_passes: int, move_count: int) -> np.ndarray:
    """
    Returns a Boolean NumPy array of legality flags of length n^2 + 1, with the pass action last.
    """
    board_arr = np.array(board, dtype=int)
    n = board_arr.shape[0]
    mask = np.zeros(n * n + 1, dtype=bool)
    
    # 1. Check if the game is already in a terminal state
    if consecutive_passes >= 2 or move_count >= 2 * n * n:
        return mask  # Every flag is False
    
    # Pass is always legal in any non-terminal state
    mask[-1] = True
    
    # Helper to find a stone group and its liberties via BFS/DFS
    def get_group_and_liberties(b: np.ndarray, start_r: int, start_c: int):
        color = b[start_r, start_c]
        if color == 0:
            return set(), set()
        
        group = set()
        liberties = set()
        stack = [(start_r, start_c)]
        group.add((start_r, start_c))
        
        while stack:
            curr_r, curr_c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    if b[nr, nc] == color:
                        if (nr, nc) not in group:
                            group.add((nr, nc))
                            stack.append((nr, nc))
                    elif b[nr, nc] == 0:
                        liberties.add((nr, nc))
        return group, liberties

    # Convert all historically seen boards to a list of arrays for fast match checks
    seen_arrays = [np.array(sp, dtype=int) for sp in seen_positions]

    # Evaluate legality for every intersection on the board
    for r in range(n):
        for c in range(n):
            # A move must be played on an empty intersection
            if board_arr[r, c] != 0:
                continue
            
            # Simulate placing the player's stone
            next_board = board_arr.copy()
            next_board[r, c] = player
            
            # Find and remove any dead opposing groups
            opp = -player
            captured_stones = set()
            visited_opp = np.zeros((n, n), dtype=bool)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and next_board[nr, nc] == opp:
                    if not visited_opp[nr, nc]:
                        g, libs = get_group_and_liberties(next_board, nr, nc)
                        for gr, gc in g:
                            visited_opp[gr, gc] = True
                        if len(libs) == 0:
                            captured_stones.update(g)
            
            # Lift captured stones off the board
            for cr, cc in captured_stones:
                next_board[cr, cc] = 0
                
            # Verify that the newly placed group has at least 1 liberty (Suicide Prevention)
            _, own_libs = get_group_and_liberties(next_board, r, c)
            if len(own_libs) == 0:
                continue
                
            # Positional Superko Check: State must not duplicate any previously encountered grid
            superko_violation = False
            for sp in seen_arrays:
                if np.array_equal(next_board, sp):
                    superko_violation = True
                    break
            
            if not superko_violation:
                mask[r * n + c] = True
                
    return mask
