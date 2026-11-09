import pandas as pd

def select_column(data: dict, column: str) -> dict:
    """
    Returns a dictionary with values as a list and length as an integer.
    """
    # Create the DataFrame from the dictionary of column lists
    df = pd.DataFrame(data)
    
    # Select the column using bracket notation to handle spaces and attribute names safely
    selected_series = df[column]
    
    # Convert the selected column back to a Python list
    values_list = selected_series.tolist()
    
    # Return the dictionary format matching the requirements
    return {
        "values": values_list,
        "length": len(values_list)
    }
