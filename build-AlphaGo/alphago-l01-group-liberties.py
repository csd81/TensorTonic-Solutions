import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    # Convert to a numpy array for easy shape handling
    board_arr = np.array(board)
    rows, cols = board_arr.shape
    
    target_color = board_arr[row, col]
    
    # Track visited states to avoid duplicate processing
    visited_group = set()
    liberties = set()
    
    # BFS queue initialized with the starting coordinate
    queue = [(row, col)]
    visited_group.add((row, col))
    
    while queue:
        r, c = queue.pop(0)
        
        # Check all 4 orthogonal directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            # Ensure the neighbor is within board boundaries
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_val = board_arr[nr, nc]
                
                if neighbor_val == target_color:
                    if (nr, nc) not in visited_group:
                        visited_group.add((nr, nc))
                        queue.append((nr, nc))
                elif neighbor_val == 0:
                    # An empty cell is a liberty
                    liberties.add((nr, nc))
                    
    # Format and sort coordinates by row, then column
    sorted_group = sorted([list(coord) for coord in visited_group], key=lambda x: (x[0], x[1]))
    sorted_liberties = sorted([list(coord) for coord in liberties], key=lambda x: (x[0], x[1]))
    
    return sorted_group, sorted_liberties
