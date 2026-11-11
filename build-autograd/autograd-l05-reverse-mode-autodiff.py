import numpy as np


def reverse_mode_autodiff(leaves: list, operations: list, output_id: str) -> tuple:
    """
    Returns a tuple of the float output and a leaf-ID-to-float-gradient dictionary.
    """
    # 1. Index leaves and operations without mutating inputs
    leaf_values = {leaf["id"]: float(leaf["value"]) for leaf in leaves}
    op_map = {op["id"]: op for op in operations}

    # 2. Build topological order of reachable nodes via post-order DFS
    visited = set()
    topo_order = []

    def dfs(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        if nid in op_map:
            for parent_id in op_map[nid]["parents"]:
                dfs(parent_id)
        topo_order.append(nid)

    dfs(output_id)

    # 3. Forward pass: compute and cache intermediate values
    values = {}
    for nid in topo_order:
        if nid in leaf_values:
            values[nid] = leaf_values[nid]
        else:
            op_node = op_map[nid]
            op_type = op_node["op"]
            parents = op_node["parents"]

            if op_type == "add":
                values[nid] = float(values[parents[0]] + values[parents[1]])
            elif op_type == "mul":
                values[nid] = float(values[parents[0]] * values[parents[1]])
            elif op_type == "tanh":
                values[nid] = float(np.tanh(np.float64(values[parents[0]])))

    # 4. Backward pass: seed output gradient and propagate in reverse
    grads = {nid: 0.0 for nid in topo_order}
    grads[output_id] = 1.0

    for nid in reversed(topo_order):
        if nid not in op_map:
            continue

        op_node = op_map[nid]
        op_type = op_node["op"]
        parents = op_node["parents"]
        g = grads[nid]

        if op_type == "add":
            grads[parents[0]] += g
            grads[parents[1]] += g
        elif op_type == "mul":
            a = values[parents[0]]
            b = values[parents[1]]
            grads[parents[0]] += g * b
            grads[parents[1]] += g * a
        elif op_type == "tanh":
            t = values[nid]
            grads[parents[0]] += g * (1.0 - t * t)

    # 5. Collect reachable leaf gradients in original input order
    leaf_grads = {
        leaf["id"]: float(grads[leaf["id"]])
        for leaf in leaves
        if leaf["id"] in visited
    }

    return float(values[output_id]), leaf_grads
