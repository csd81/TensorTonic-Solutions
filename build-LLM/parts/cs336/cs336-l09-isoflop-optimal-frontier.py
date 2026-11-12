import torch

def isoflop_optimal_frontier(
    compute_budgets: torch.Tensor,
    parameter_counts: torch.Tensor,
    terminal_losses: torch.Tensor,
) -> dict:
    """
    Returns a dict: optima (list of dicts), frontier_scale, frontier_exponent.
    """
    budgets = compute_budgets.to(dtype=torch.float64)
    params = parameter_counts.to(dtype=torch.float64)
    losses = terminal_losses.to(dtype=torch.float64)

    optima = []
    optimal_log_n_list = []

    num_budgets = budgets.shape[0]
    for i in range(num_budgets):
        c_val = budgets[i]
        x = torch.log(params[i])
        y = losses[i]

        # Center x around its mean for better numerical conditioning of the Vandermonde matrix
        x_mean = x.mean()
        u = x - x_mean
        design = torch.stack([u ** 2, u, torch.ones_like(u)], dim=-1)

        # Solve unweighted least squares: L(u) = a * u^2 + b_u * u + c_u
        coeffs = torch.linalg.lstsq(design, y).solution
        a_c, b_u, c_u = coeffs[0], coeffs[1], coeffs[2]

        # Vertex in centered coordinates: u* = -b_u / (2 * a_c)
        u_star = -b_u / (2.0 * a_c)
        x_star = x_mean + u_star

        # Minimum loss at the vertex: L_C(x*) = a_c * (u*)^2 + b_u * u* + c_u
        min_loss = a_c * (u_star ** 2) + b_u * u_star + c_u

        n_c = torch.exp(x_star)
        d_c = c_val / (6.0 * n_c)

        optimal_log_n_list.append(x_star)
        optima.append({
            "compute": float(c_val.item()),
            "parameters": float(n_c.item()),
            "tokens": float(d_c.item()),
            "loss": float(min_loss.item()),
        })

    # Fit unweighted OLS line across recovered optima: log(N_C) = log(k) + alpha * log(C)
    log_c = torch.log(budgets)
    log_n = torch.stack(optimal_log_n_list)

    log_c_mean = log_c.mean()
    log_c_centered = log_c - log_c_mean

    alpha = torch.sum(log_c_centered * (log_n - log_n.mean())) / torch.sum(log_c_centered ** 2)
    log_k = log_n.mean() - alpha * log_c_mean
    k = torch.exp(log_k)

    return {
        "optima": optima,
        "frontier_scale": float(k.item()),
        "frontier_exponent": float(alpha.item()),
    }
