import math

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    """
    Returns the cosine-annealed learning rate for the requested step.
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    return float(lr)
