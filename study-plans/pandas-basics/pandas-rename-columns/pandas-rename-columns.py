import pandas as pd

def rename_columns(data: dict, rename_map: dict) -> dict:
    """
    Returns a dictionary mapping renamed column names to their value lists.
    """
    first_seen_order = []
    seen = set()
    latest_values = {}
    
    for old_name, values in data.items():
        # Determine the new name using the rename map
        new_name = rename_map.get(old_name, old_name)
        
        # Track the first position/appearance of the new name
        if new_name not in seen:
            seen.add(new_name)
            first_seen_order.append(new_name)
            
        # Overwrite to ensure the last column's values are kept upon collision
        latest_values[new_name] = values
        
    # Reconstruct the dictionary in the order of first appearance
    return {name: latest_values[name] for name in first_seen_order}
