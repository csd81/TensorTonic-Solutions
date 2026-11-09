import pandas as pd

def melt_dataframe(data: dict, id_vars: list, value_vars: list) -> dict:
    """
    Returns identifier column lists together with variable and value lists.
    """
    # 1. Convert the input dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # 2. Unpivot the DataFrame from wide to long format
    # If value_vars is empty, pd.melt automatically uses all non-id_vars columns
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars if value_vars else None)
    
    # 3. Convert back to a dictionary of lists (orient='list')
    return melted_df.to_dict(orient='list')
