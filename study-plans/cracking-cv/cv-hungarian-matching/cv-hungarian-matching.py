import numpy as np

def hungarian_matching(cost_matrix: list) -> dict:
    """
    Returns a dictionary with row_indices, col_indices (int lists), 
    and total_cost (float) rounded to 4 decimals.
    """
    # 1. Initialization and Matrix Shape handling
    C_orig = np.array(cost_matrix, dtype=np.float64)
    N, M = C_orig.shape
    
    # Track transposition to handle rectangular cases uniformly
    is_transposed = False
    if N > M:
        C = C_orig.T.copy()
        is_transposed = True
    else:
        C = C_orig.copy()
        
    n_rows, n_cols = C.shape  # Here n_rows <= n_cols always
    
    # 2. Step 1: Subtract row minimums
    for r in range(n_rows):
        C[r] -= np.min(C[r])
        
    # 3. Step 2: Subtract column minimums (only if square; if rectangular, columns could be 0 anyway)
    if n_rows == n_cols:
        for c in range(n_cols):
            C[:, c] -= np.min(C[:, c])
            
    # Helper to find an initial tentative assignment
    def get_initial_assignment(mat):
        row_ind = -np.ones(n_rows, dtype=int)
        col_ind = -np.ones(n_cols, dtype=int)
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isclose(mat[r, c], 0.0) and col_ind[c] == -1:
                    row_ind[r] = c
                    col_ind[c] = r
                    break
        return row_ind
    
    # Initialize assignment state
    # row_to_col[r] stores the assigned column for row r, or -1 if unassigned
    row_to_col = get_initial_assignment(C)
    
    # 4. Main Hungarian Optimization Loop
    while np.any(row_to_col == -1):
        # Find all zeros
        is_zero = np.isclose(C, 0.0)
        
        # Track covered rows and columns
        row_covered = np.zeros(n_rows, dtype=bool)
        col_covered = np.zeros(n_cols, dtype=bool)
        
        # Cover columns containing assignments
        for r in range(n_rows):
            if row_to_col[r] != -1:
                col_covered[row_to_col[r]] = True
                
        # Alternating path search / Cover updates
        marked_zeros = {} # maps uncovered zero coordinates (r, c) -> prime/starred state
        
        def step_4_5():
            nonlocal row_to_col
            # Find an uncovered zero
            uncovered_r, uncovered_c = -1, -1
            for r in range(n_rows):
                if not row_covered[r]:
                    for c in range(n_cols):
                        if not col_covered[c] and is_zero[r, c]:
                            uncovered_r, uncovered_c = r, c
                            break
                if uncovered_r != -1:
                    break
            
            # If no uncovered zero exists, we need to adjust potential values (Step 6)
            if uncovered_r == -1:
                return "step6"
            
            # "Prime" the zero
            marked_zeros[uncovered_r] = uncovered_c
            
            # If there is an assigned zero in the same row...
            assigned_c = row_to_col[uncovered_r]
            if assigned_c != -1:
                # Cover this row and uncover the column of the assigned zero
                row_covered[uncovered_r] = True
                col_covered[assigned_c] = False
                return "continue"
            else:
                # Step 5: Construct augmenting path
                path = [(uncovered_r, uncovered_c)]
                while True:
                    # Find a starred zero in the column of the latest primed zero
                    last_c = path[-1][1]
                    starred_r = -1
                    for r in range(n_rows):
                        if row_to_col[r] == last_c:
                            starred_r = r
                            break
                    if starred_r == -1:
                        break
                    
                    # Find the primed zero in that starred row
                    primed_c = marked_zeros[starred_r]
                    path.append((starred_r, last_c))
                    path.append((starred_r, primed_c))
                
                # Unstar starred zeros, star primed zeros along the path
                for r, c in path:
                    if row_to_col[r] == c:
                        row_to_col[r] = -1
                    elif marked_zeros.get(r) == c:
                        row_to_col[r] = c
                return "break"
        
        # Run step 4 until we either augment or need matrix modification
        status = "continue"
        while status == "continue":
            status = step_4_5()
            
        if status == "step6":
            # Step 6: Find the minimum uncovered value
            min_val = np.inf
            for r in range(n_rows):
                if not row_covered[r]:
                    for c in range(n_cols):
                        if not col_covered[c]:
                            if C[r, c] < min_val:
                                min_val = C[r, c]
            
            # Modify the matrix based on coverage
            for r in range(n_rows):
                if row_covered[r]:
                    C[r, :] += min_val
            for c in range(n_cols):
                if not col_covered[c]:
                    C[:, c] -= min_val

    # 5. Format and map indices back to original matrix shapes
    final_rows = []
    final_cols = []
    
    if is_transposed:
        # Transposed back means our rows are the original columns
        orig_assigned_cols = np.arange(n_rows)
        orig_assigned_rows = row_to_col
        
        # Sort indices by original row index ascending
        sorted_pairs = sorted(zip(orig_assigned_rows, orig_assigned_cols))
        final_rows = [int(r) for r, _ in sorted_pairs]
        final_cols = [int(c) for _, c in sorted_pairs]
    else:
        for r in range(n_rows):
            final_rows.append(r)
            final_cols.append(int(row_to_col[r]))
            
    # Calculate original total cost before rounding
    total_cost = 0.0
    for r, c in zip(final_rows, final_cols):
        total_cost += C_orig[r, c]
        
    return {
        "row_indices": final_rows,
        "col_indices": final_cols,
        "total_cost": float(round(total_cost, 4))
    }
