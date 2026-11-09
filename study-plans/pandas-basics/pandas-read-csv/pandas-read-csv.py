import pandas as pd

def create_dataframe(data: dict) -> dict:
    df = pd.DataFrame(data)

    return {
        "data": df.to_dict(orient="list"),
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
    }