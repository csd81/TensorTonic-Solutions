import numpy as np


def autodiff_gradient_check(
    leaves: list, operations: list, output_id: str, h: float
) -> tuple:
    """
    Returns a tuple of analytic gradients, numerical gradients, and maximum error.
    """
    base_leaves = {leaf["id"]: float(leaf["value"]) for leaf in leaves}
    op_map = {op["id"]: op for op in operations}
    step = float(h)

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

    def evaluate(leaf_vals: dict) -> dict:
        vals = {}
        for nid in topo_order:
            if nid in leaf_vals:
                vals[nid] = leaf_vals[nid]
            else:
                op_node = op_map[nid]
                op_type = op_node["op"]
                parents = op_node["parents"]

                if op_type == "add":
                    vals[nid] = float(vals[parents[0]] + vals[parents[1]])
                elif op_type == "mul":
                    vals[nid] = float(vals[parents[0]] * vals[parents[1]])
                elif op_type == "tanh":
                    vals[nid] = float(np.tanh(np.float64(vals[parents[0]])))
        return vals

    base_values = evaluate(base_leaves)
    base_output = base_values[output_id]

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
            a = base_values[parents[0]]
            b = base_values[parents[1]]
            grads[parents[0]] += g * b
            grads[parents[1]] += g * a
        elif op_type == "tanh":
            t = base_values[nid]
            grads[parents[0]] += g * (1.0 - t * t)

    reachable_leaf_ids = [
        leaf["id"] for leaf in leaves if leaf["id"] in visited
    ]

    analytic_grads = {lid: float(grads[lid]) for lid in reachable_leaf_ids}

    numerical_grads = {}
    for lid in reachable_leaf_ids:
        perturbed_leaves = dict(base_leaves)
        perturbed_leaves[lid] += step
        perturbed_output = evaluate(perturbed_leaves)[output_id]
        numerical_grads[lid] = float((perturbed_output - base_output) / step)

    max_error = 0.0
    for lid in reachable_leaf_ids:
        err = abs(analytic_grads[lid] - numerical_grads[lid])
        if err > max_error:
            max_error = float(err)

    return analytic_grads, numerical_grads, float(max_error)
