import numpy as np

def tromp_taylor_result(board: list, komi: float, consecutive_passes: int, move_count: int) -> tuple:
    """
    Scores a terminal Go board using Tromp-Taylor area scoring.
    
    Returns: 
        A four-item tuple: (terminal_flag, black_score, white_score, winner)
        If the game is unfinished, returns (False, None, None, None).
    """
    # Convert to NumPy array to safely preserve and query the board
    board_arr = np.array(board)
    h, w = board_arr.shape
    
    # 1. Check termination conditions
    is_terminal = (consecutive_passes >= 2) or (move_count >= 2 * h * w)
    if not is_terminal:
        return (False, None, None, None)
    
    # 2. Count live stones on the board
    black_score = float(np.sum(board_arr == 1))
    white_score = float(np.sum(board_arr == -1)) + float(komi)
    
    # Track visited empty intersections
    visited = np.zeros_like(board_arr, dtype=bool)
    
    # 3. Analyze connected empty regions (0s)
    for r in range(h):
        for c in range(w):
            if board_arr[r, c] == 0 and not visited[r, c]:
                # Start a BFS/Flood-fill for this empty territory
                component_size = 0
                adjacent_colors = set()
                queue = [(r, c)]
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component_size += 1
                    
                    # Explore 4-directional neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbor_val = board_arr[nr, nc]
                            if neighbor_val == 0:
                                if not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                            else:
                                # Found a bordering stone color
                                adjacent_colors.add(neighbor_val)
                
                # 4. Award empty territory if it is uniquely bordered by one color
                if len(adjacent_colors) == 1:
                    border_color = list(adjacent_colors)[0]
                    if border_color == 1:
                        black_score += component_size
                    elif border_color == -1:
                        white_score += component_size

    # 5. Determine the winner
    if black_score > white_score:
        winner = 1
    elif white_score > black_score:
        winner = -1
    else:
        winner = 0
        
    return (True, black_score, white_score, winner)
