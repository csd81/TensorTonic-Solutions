import pandas as pd

def create_pivot(data: dict, index: str, columns: str, values: str, aggfunc: str) -> dict:
    """
    Returns a dictionary from column labels to dictionaries of row labels and aggregates.
    """
    # 1. Load the input dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # 2. Generate the pivot table, ensuring missing combinations are filled with 0
    pivot_df = pd.pivot_table(
        df, 
        index=index, 
        columns=columns, 
        values=values, 
        aggfunc=aggfunc, 
        fill_value=0
    )
    
    # 3. Ensure 'count' aggregation explicitly returns integer types
    if aggfunc == "count":
        pivot_df = pivot_df.astype(int)
        
    # 4. Convert to a nested dictionary (outer keys = columns, inner keys = row labels)
    return pivot_df.to_dict()
