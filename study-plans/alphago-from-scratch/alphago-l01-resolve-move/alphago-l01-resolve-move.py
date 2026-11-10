import numpy as np

def resolve_go_action(board: list, player: int, action: int, consecutive_passes: int, move_count: int) -> tuple:
    """
    Applies a legal Go action to a nonterminal state and resolves captures.
    
    Returns: 
        A six-item tuple: (next_board, next_player, captured_count, 
                           consecutive_pass_count, move_count, terminal)
    """
    board = np.array(board, dtype=int)
    n = board.shape[0]
    
    next_player = -player
    next_move_count = move_count + 1
    
    # 1. Handle Pass Action
    if action == n * n:
        next_consecutive_passes = consecutive_passes + 1
        captured = 0
        terminal = (next_consecutive_passes >= 2) or (next_move_count >= 2 * n * n)
        return board, next_player, captured, next_consecutive_passes, next_move_count, bool(terminal)
        
    # 2. Handle Stone Placement
    r, c = action // n, action % n
    board[r, c] = player
    next_consecutive_passes = 0
    
    # Identify and remove dead opponent groups
    opponent = -player
    visited = np.zeros_like(board, dtype=bool)
    captured = 0
    
    for i in range(n):
        for j in range(n):
            if board[i, j] == opponent and not visited[i, j]:
                # Explore the connected string (group) via BFS
                group = []
                liberties = set()
                queue = [(i, j)]
                visited[i, j] = True
                
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    group.append((curr_r, curr_c))
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if board[nr, nc] == 0:
                                liberties.add((nr, nc))
                            elif board[nr, nc] == opponent and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                                
                # If the group has zero liberties, it is captured
                if len(liberties) == 0:
                    captured += len(group)
                    for gr, gc in group:
                        board[gr, gc] = 0
                        
    # 3. Determine Terminal State
    terminal = (next_consecutive_passes >= 2) or (next_move_count >= 2 * n * n)
    
    return board, next_player, captured, next_consecutive_passes, next_move_count, bool(terminal)
