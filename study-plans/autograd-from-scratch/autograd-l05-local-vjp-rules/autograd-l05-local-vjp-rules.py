def local_vjp(
    operation: str, inputs: list, output: float, upstream_gradient: float
) -> list:
    """
    Returns a list of float gradient contributions in input order.
    """
    g = float(upstream_gradient)

    if operation == "add":
        return [g, g]

    if operation == "mul":
        a = float(inputs[0])
        b = float(inputs[1])
        return [g * b, g * a]

    if operation == "tanh":
        t = float(output)
        return [g * (1.0 - t * t)]

    raise ValueError(f"Unsupported operation: {operation}")
