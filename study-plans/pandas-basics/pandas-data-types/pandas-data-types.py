import pandas as pd

def data_types_overview(data: dict) -> dict:
    """
    Returns a dictionary with dtypes, type_counts, and num_columns.
    """
    # Create the DataFrame using pandas default inference
    df = pd.DataFrame(data)
    
    # 1. Map each column name to its string representation of the dtype
    dtypes_map = {col: str(dtype) for col in df.columns for dtype in [df[col].dtype]}
    
    # 2. Count the occurrences of each unique dtype string
    type_counts = {}
    for dtype_str in dtypes_map.values():
        type_counts[dtype_str] = type_counts.get(dtype_str, 0) + 1
        
    # 3. Get the total number of columns
    num_columns = len(df.columns)
    
    return {
        "dtypes": dtypes_map,
        "type_counts": type_counts,
        "num_columns": num_columns
    }
