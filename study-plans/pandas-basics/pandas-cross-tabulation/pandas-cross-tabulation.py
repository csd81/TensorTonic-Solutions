import pandas as pd

def cross_tab(data: dict, row_col: str, col_col: str) -> dict:
    """
    Returns a dictionary from column labels to dictionaries of row labels and integer counts.
    """
    # 1. Convert the input dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # 2. Compute the cross-tabulation frequency table
    # pd.crosstab automatically includes all observed combinations and defaults missing ones to 0
    ct_df = pd.crosstab(df[row_col], df[col_col])
    
    # 3. Convert to a nested dictionary
    # By default, columns (col_col) become outer keys and rows (row_col) become inner keys
    return ct_df.to_dict()
