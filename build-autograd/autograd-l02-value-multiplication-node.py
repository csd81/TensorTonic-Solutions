import numpy as np

def value_multiplication_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns the node dictionary with id, data, grad, op, and parents.
    """
    product = np.float64(left["data"]) * np.float64(right["data"])

    return {
        "id": output_id,
        "data": float(product),
        "grad": 0.0,
        "op": "*",
        "parents": [left, right],
    }
