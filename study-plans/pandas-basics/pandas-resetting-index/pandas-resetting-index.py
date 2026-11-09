import pandas as pd

def reset_index_demo(data: dict, index_col: str) -> list:
    """
    Returns [columns_before_reset, columns_after_reset], both lists of strings.
    """
    # 1. Create the initial DataFrame from the provided dictionary
    df = pd.DataFrame(data)
    
    # 2. Set the specified column as the row index
    df = df.set_index(index_col)
    
    # Get the column names list before the reset (as requested by the prompt)
    # df.columns.tolist() contains all remaining columns in their original order
    columns_before_reset = df.columns.tolist()
    
    # 3. Restore the index back as a data column
    df = df.reset_index()
    
    # Get the column names list after the reset
    # This automatically puts the former index column first, followed by others
    columns_after_reset = df.columns.tolist()
    
    return [columns_before_reset, columns_after_reset]
