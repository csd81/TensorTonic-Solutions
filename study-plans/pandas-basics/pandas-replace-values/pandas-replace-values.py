import pandas as pd

def replace_values(data: dict, column: str, old_val: object, new_val: object) -> dict:
    """
    Returns data as a dictionary of updated lists and count as an integer.
    """
    # Create a copy of the original data dictionary to avoid modifying the input
    updated_data = {k: list(v) for k, v in data.items()}
    
    count = 0
    if column in updated_data:
        # Replace occurrences and track the count
        updated_list = []
        for item in updated_data[column]:
            if item == old_val:
                updated_list.append(new_val)
                count += 1
            else:
                updated_list.append(item)
        updated_data[column] = updated_list
        
    return {
        "data": updated_data,
        "count": count
    }
