import numpy as np

def build_expression_graph(leaves: list, operations: list) -> tuple:
    """
    Returns a tuple of the node list and final node ID.
    """
    nodes = []
    values_by_id = {}

    for leaf in leaves:
        node_id = leaf["id"]
        val = np.float64(leaf["data"])
        values_by_id[node_id] = val
        nodes.append({
            "id": node_id,
            "data": float(val),
            "grad": 0.0,
            "op": "",
            "parents": [],
        })

    for operation in operations:
        node_id = operation["id"]
        op = operation["op"]
        left_id = operation["left"]
        right_id = operation["right"]

        left_val = values_by_id[left_id]
        right_val = values_by_id[right_id]

        if op == "+":
            result = left_val + right_val
        elif op == "*":
            result = left_val * right_val
        else:
            raise ValueError(f"Unsupported operation: {op}")

        values_by_id[node_id] = result
        nodes.append({
            "id": node_id,
            "data": float(result),
            "grad": 0.0,
            "op": op,
            "parents": [left_id, right_id],
        })

    final_id = nodes[-1]["id"]
    return (nodes, final_id)
