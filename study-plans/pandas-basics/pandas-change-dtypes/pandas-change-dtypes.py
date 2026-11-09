import pandas as pd

def change_dtype(data: dict, column: str, target_type: str) -> list:
    # 1. Create the DataFrame from the dictionary of lists
    df = pd.DataFrame(data)
    
    # 2. Capture the dtypes before the conversion
    dtypes_before = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 3. Map the user-friendly target_type string to the specific pandas dtype
    type_mapping = {
        "int": "int64",
        "float": "float64",
        "str": "object"
    }
    pandas_type = type_mapping[target_type]
    
    # 4. Cast the specified column to the target dtype
    df[column] = df[column].astype(pandas_type)
    
    # 5. Capture the dtypes after the conversion
    dtypes_after = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 6. Return both dictionaries as a list
    return [dtypes_before, dtypes_after]
