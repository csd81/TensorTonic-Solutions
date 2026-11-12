import copy

def backup_mcts_path(path: list, leaf_value: float, leaf_player: int) -> list:
    """
    Returns updated copies of the path records in their original order.
    """
    if not path:
        return []
    
    # Create a deep copy to ensure records are independent of the input
    updated_path = copy.deepcopy(path)
    
    for edge in updated_path:
        # Determine the value from this edge's player perspective
        v_a = leaf_value if edge["player"] == leaf_player else -leaf_value
        
        # Update statistics
        edge["N"] += 1
        edge["W"] += v_a
        edge["Q"] = edge["W"] / edge["N"]
        
    return updated_path
