import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    a_val = np.float64(a)
    b_val = np.float64(b)
    c_val = np.float64(c)
    f_val = np.float64(f)
    h_val = np.float64(h)

    def forward(x_a: np.float64, x_b: np.float64, x_c: np.float64, x_f: np.float64) -> np.float64:
        e = x_a * x_b + x_c
        return e * x_f

    # Forward pass
    e_val = a_val * b_val + c_val
    loss = e_val * f_val

    # Analytic gradients via chain rule: [dL/da, dL/db, dL/dc, dL/df]
    dL_de = f_val
    dL_df = e_val
    dL_da = dL_de * b_val
    dL_db = dL_de * a_val
    dL_dc = dL_de * np.float64(1.0)

    analytic_grads = [dL_da, dL_db, dL_dc, dL_df]

    # Numerical gradients via forward differences
    num_da = (forward(a_val + h_val, b_val, c_val, f_val) - loss) / h_val
    num_db = (forward(a_val, b_val + h_val, c_val, f_val) - loss) / h_val
    num_dc = (forward(a_val, b_val, c_val + h_val, f_val) - loss) / h_val
    num_df = (forward(a_val, b_val, c_val, f_val + h_val) - loss) / h_val

    numerical_grads = [num_da, num_db, num_dc, num_df]

    # Maximum absolute pairwise difference
    max_error = max(abs(ag - ng) for ag, ng in zip(analytic_grads, numerical_grads))

    return (
        float(loss),
        [float(g) for g in analytic_grads],
        [float(g) for g in numerical_grads],
        float(max_error),
    )
