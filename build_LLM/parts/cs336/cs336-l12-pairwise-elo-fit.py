import math
import torch

def pairwise_elo_fit(
    num_models: int,
    model_a: torch.Tensor,
    model_b: torch.Tensor,
    outcomes: torch.Tensor,
    steps: int,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict: ratings, rankings, fitted_probabilities, component_ids (tensors), connected (bool).
    """
    device = outcomes.device
    idx_a = model_a.to(dtype=torch.int64, device=device)
    idx_b = model_b.to(dtype=torch.int64, device=device)
    y = outcomes.to(dtype=torch.float64, device=device)

    num_comparisons = int(y.shape[0])
    
    ratings = torch.zeros((num_models,), dtype=torch.float64, device=device)

    scale = math.log(10.0) / 400.0
    lr = float(learning_rate)

    if num_comparisons > 0 and steps > 0:
        grad_scale = scale / num_comparisons
        for _ in range(steps):
            # Predicted probability that model_a beats model_b
            diff = ratings[idx_b] - ratings[idx_a]
            probs = 1.0 / (1.0 + torch.pow(10.0, diff / 400.0))

            # Residual: (y_n - p_n)
            residuals = y - probs

            # Accumulate +residual for model_a and -residual for model_b
            grad = torch.zeros((num_models,), dtype=torch.float64, device=device)
            grad.index_add_(0, idx_a, residuals)
            grad.index_add_(0, idx_b, -residuals)
            grad.mul_(grad_scale)

            # Simultaneous gradient ascent step followed by zero-mean centering
            ratings = ratings + lr * grad
            ratings = ratings - ratings.mean()

    # Final predicted probabilities (works even when steps == 0 or num_comparisons == 0)
    if num_comparisons > 0:
        final_diff = ratings[idx_b] - ratings[idx_a]
        fitted_probabilities = 1.0 / (1.0 + torch.pow(10.0, final_diff / 400.0))
    else:
        fitted_probabilities = torch.empty((0,), dtype=torch.float64, device=device)

    # Sort models by descending rating, breaking ties by lower model index
    # Using stable sort on (-ratings) after arranging indices 0..N-1 preserves index order on ties
    rankings = torch.argsort(-ratings, stable=True).to(dtype=torch.int64, device=device)

    # Connected components via Union-Find (always linking to the smaller root index)
    parent = list(range(num_models))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(idx_a.tolist(), idx_b.tolist()):
        ru, rv = find(u), find(v)
        if ru != rv:
            if ru < rv:
                parent[rv] = ru
            else:
                parent[ru] = rv

    # Assign contiguous component IDs 0, 1, ... in increasing order of smallest model index
    root_to_comp = {}
    comp_ids_list = []
    for i in range(num_models):
        r = find(i)
        if r not in root_to_comp:
            root_to_comp[r] = len(root_to_comp)
        comp_ids_list.append(root_to_comp[r])

    component_ids = torch.tensor(comp_ids_list, dtype=torch.int64, device=device)
    connected = bool(len(root_to_comp) == 1)

    return {
        "ratings": ratings,
        "rankings": rankings,
        "fitted_probabilities": fitted_probabilities,
        "component_ids": component_ids,
        "connected": connected,
    }
