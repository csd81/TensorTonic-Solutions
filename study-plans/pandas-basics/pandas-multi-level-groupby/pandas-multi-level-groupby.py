import pandas as pd

def multi_groupby(data: dict, group_cols: list, target_col: str, aggfunc: str) -> dict:
    """
    Groups the data by multiple specified columns, applies the requested aggregation 
    on the target column, and returns the flattened result.
    """
    # 1. Convert the input dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # 2. Perform the multi-level group-by and apply the aggregation function
    grouped = df.groupby(group_cols)[target_col].agg(aggfunc)
    
    # 3. Reset the index to flatten the resulting MultiIndex DataFrame/Series back into a flat table
    flat_df = grouped.reset_index()
    
    # 4. Convert the flat DataFrame to a dictionary (e.g., standard column orientation)
    return flat_df.to_dict(orient="list")
