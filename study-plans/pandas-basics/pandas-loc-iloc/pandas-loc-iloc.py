import pandas as pd

def iloc_selection(data: dict, row: int, col: int) -> list:
    """
    Returns [element, row_values, col_values], with both value sequences as lists.
    """
    # 1. Create the DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # 2. Extract the specific scalar element at (row, col)
    element = df.iloc[row, col]
    
    # 3. Extract the entire row at the given position and convert to a list
    row_values = df.iloc[row].tolist()
    
    # 4. Extract the entire column at the given position and convert to a list
    col_values = df.iloc[:, col].tolist()
    
    # 5. Return the expected list format
    return [element, row_values, col_values]
