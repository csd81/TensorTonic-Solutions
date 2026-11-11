def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    def count_elements(shapes: list[list[int]]) -> int:
        total_elements = 0
        for shape in shapes:
            # An empty shape collection denotes a scalar with 1 element
            if not shape:
                total_elements += 1
                continue
            
            # Product of all dimensions
            prod = 1
            for dim in shape:
                prod *= dim
            total_elements += prod
        return total_elements

    # Calculate element counts
    num_param_elements = count_elements(param_shapes)
    num_act_elements = count_elements(activation_shapes)

    # Determine optimizer state multiplier
    opt_multiplier = {
        "sgd": 0,
        "adagrad": 1,
        "adam": 2
    }[optimizer.lower()]

    # Compute category byte sizes
    parameters = num_param_elements * param_bytes_per_element
    gradients = num_param_elements * grad_bytes_per_element
    activations = num_act_elements * activation_bytes_per_element
    optimizer_state = num_param_elements * opt_multiplier * optimizer_bytes_per_element
    
    total = parameters + gradients + activations + optimizer_state

    return {
        "parameters": parameters,
        "gradients": gradients,
        "activations": activations,
        "optimizer_state": optimizer_state,
        "total": total
    }
