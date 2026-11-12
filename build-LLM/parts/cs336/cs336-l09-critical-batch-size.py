import torch

def critical_batch_size(
    batch_sizes: torch.Tensor,
    steps_to_target: torch.Tensor,
    reached_mask: torch.Tensor,
) -> dict:
    mask = reached_mask.to(dtype=torch.bool)
    
    # Cast to float64 before multiplying to prevent int32 overflow and float32 precision loss
    all_examples = batch_sizes.double() * steps_to_target.double()
    valid_steps = steps_to_target[mask].double()
    valid_examples = all_examples[mask]

    min_steps = float(torch.min(valid_steps).item())
    min_examples = float(torch.min(valid_examples).item())

    return {
        "examples": [
            val if reached else None
            for val, reached in zip(all_examples.tolist(), mask.tolist())
        ],
        "min_steps": min_steps,
        "min_examples": min_examples,
        "critical_batch_size": min_examples / min_steps,
    }
