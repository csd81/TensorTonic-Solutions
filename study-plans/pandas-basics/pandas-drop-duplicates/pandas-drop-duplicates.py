import pandas as pd

def drop_duplicates(data: dict) -> list:
    # Create DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Record row counts
    rows_before = len(df)
    
    # Drop duplicate rows, keeping the first occurrence
    df_cleaned = df.drop_duplicates()
    rows_after = len(df_cleaned)
    
    # Convert cleaned DataFrame back to a dictionary of lists
    cleaned_data = df_cleaned.to_dict(orient='list')
    
    return [rows_before, rows_after, cleaned_data]
