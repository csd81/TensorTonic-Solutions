import pandas as pd

def unstack_to_wide(data: dict, index_col: str, columns_col: str, values_col: str) -> dict:
    # Convert input dictionary to a DataFrame
    df = pd.DataFrame(data)
    
    # Pivot the table to reshape from long to wide
    wide_df = df.pivot(index=index_col, columns=columns_col, values=values_col)
    
    # Order rows by ascending identifier
    wide_df = wide_df.sort_index()
    
    # Restore index_col as a standard column
    wide_df = wide_df.reset_index()
    
    # Convert the wide DataFrame back into a dictionary of lists
    return wide_df.to_dict(orient='list')
