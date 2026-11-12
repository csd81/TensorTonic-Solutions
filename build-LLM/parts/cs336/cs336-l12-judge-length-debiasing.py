import torch

def judge_length_debiasing(
    num_models: int,
    model_a: torch.Tensor,
    model_b: torch.Tensor,
    length_differences: torch.Tensor,
    outcomes: torch.Tensor,
    steps: int,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict: length_coefficient (float), debiased_probabilities and model_win_rates (float64 tensors).
    """
    device = outcomes.device
    idx_a = model_a.to(dtype=torch.int64, device=device)
    idx_b = model_b.to(dtype=torch.int64, device=device)
    delta = length_differences.to(dtype=torch.float64, device=device)
    y = outcomes.to(dtype=torch.float64, device=device)

    m = int(y.shape[0])
    utilities = torch.zeros((num_models,), dtype=torch.float64, device=device)
    beta = torch.tensor(0.0, dtype=torch.float64, device=device)
    lr = float(learning_rate)

    if m > 0 and steps > 0:
        for _ in range(steps):
            # Forward pass with length covariate: logit_n = u_{a_n} - u_{b_n} + beta * delta_n
            logits = utilities[idx_a] - utilities[idx_b] + beta * delta
            probs = torch.sigmoid(logits)

            # Gradient of mean BCE loss w.r.t. logit_n is (p_n - y_n) / M
            grad_logits = (probs - y) / m

            # Gradients for model utilities and length coefficient from pre-update parameters
            grad_u = torch.zeros((num_models,), dtype=torch.float64, device=device)
            grad_u.index_add_(0, idx_a, grad_logits)
            grad_u.index_add_(0, idx_b, -grad_logits)
            grad_beta = torch.sum(grad_logits * delta)

            # Simultaneous gradient descent update and zero-mean centering of utilities
            utilities = utilities - lr * grad_u
            utilities = utilities - utilities.mean()
            beta = beta - lr * grad_beta

    if m > 0:
        # Evaluate debiased A-win probabilities at zero length difference (delta = 0)
        debiased_logits = utilities[idx_a] - utilities[idx_b]
        debiased_probabilities = torch.sigmoid(debiased_logits)

        # Accumulate win probability sums and appearance counts for each model
        win_prob_sum = torch.zeros((num_models,), dtype=torch.float64, device=device)
        win_prob_sum.index_add_(0, idx_a, debiased_probabilities)
        win_prob_sum.index_add_(0, idx_b, 1.0 - debiased_probabilities)

        counts = torch.zeros((num_models,), dtype=torch.float64, device=device)
        ones = torch.ones((m,), dtype=torch.float64, device=device)
        counts.index_add_(0, idx_a, ones)
        counts.index_add_(0, idx_b, ones)

        # Unseen models (counts == 0) default to 0.5
        model_win_rates = torch.where(
            counts > 0,
            win_prob_sum / torch.clamp(counts, min=1.0),
            torch.full((num_models,), 0.5, dtype=torch.float64, device=device),
        )
    else:
        debiased_probabilities = torch.empty((0,), dtype=torch.float64, device=device)
        model_win_rates = torch.full((num_models,), 0.5, dtype=torch.float64, device=device)

    return {
        "length_coefficient": float(beta.item()),
        "debiased_probabilities": debiased_probabilities,
        "model_win_rates": model_win_rates,
    }
