import pandas as pd

def inspect_dataframe(data: dict) -> dict:
    df = pd.DataFrame(data)
    rows, cols = df.shape
    return {
        "rows": int(rows),
        "cols": int(cols),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_values": int(df.size),
    }