import numpy as np

def value_addition_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns an addition node with id, data, grad, op, and the original parents.
    """
    total = np.float64(left["data"]) + np.float64(right["data"])

    return {
        "id": output_id,
        "data": float(total),
        "grad": 0.0,
        "op": "+",
        "parents": [left, right],
    }
